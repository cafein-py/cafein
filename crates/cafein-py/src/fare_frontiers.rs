//! The cutoff-pruned (time, fare) frontier product bindings.

use super::*;
use cafein_core::routers::fare_frontier::{frontier, push_walk, FareFrontierInputs, FrontierRow};
use cafein_core::routers::router::Request;
use rayon::prelude::*;

use crate::cost_matrices::fare_tables;
use crate::points::{request_offsets, unsnapped};

/// Rejects a zero step or a grid too fine for its window — an
/// unbounded sample list would materialise before routing.
fn validated_step(window: u32, departure_step: Option<u32>) -> PyResult<()> {
    let Some(step) = departure_step else {
        return Ok(());
    };
    if step == 0 {
        return Err(PyValueError::new_err(
            "departure_step must be a positive number of seconds",
        ));
    }
    if (window as u64).div_ceil(step as u64) > 100_000 {
        return Err(PyValueError::new_err(
            "window / departure_step samples the window more than \
             100000 times; widen the step or narrow the window",
        ));
    }
    Ok(())
}

fn validated_cutoffs(cutoffs: &[f64]) -> PyResult<()> {
    if cutoffs.is_empty()
        || cutoffs
            .iter()
            .any(|cutoff| !cutoff.is_finite() || *cutoff < 0.0)
        || cutoffs.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(PyValueError::new_err(
            "cutoffs must be a non-empty ascending list of finite, \
             non-negative amounts",
        ));
    }
    Ok(())
}

fn frontier_columns(
    py: Python<'_>,
    cells: &[Vec<Vec<Option<FrontierRow>>>],
) -> PyResult<Py<PyDict>> {
    let mut from_index: Vec<u32> = Vec::new();
    let mut to_index: Vec<u32> = Vec::new();
    let mut cutoff_column: Vec<f64> = Vec::new();
    let mut travel_time: Vec<u32> = Vec::new();
    let mut fare: Vec<f64> = Vec::new();
    let mut rides: Vec<u32> = Vec::new();
    for (i, slots) in cells.iter().enumerate() {
        for (j, rows) in slots.iter().enumerate() {
            for row in rows.iter().flatten() {
                from_index.push(i as u32);
                to_index.push(j as u32);
                cutoff_column.push(row.cutoff);
                travel_time.push(row.travel_time);
                fare.push(row.fare);
                rides.push(row.rides as u32);
            }
        }
    }
    let out = PyDict::new(py);
    out.set_item("from_index", from_index)?;
    out.set_item("to_index", to_index)?;
    out.set_item("cutoff", cutoff_column)?;
    out.set_item("travel_time_s", travel_time)?;
    out.set_item("fare", fare)?;
    out.set_item("rides", rides)?;
    Ok(out.unbind())
}

#[pymethods]
impl TransportNetwork {
    /// The stop-to-stop (time, fare) frontier over a departure window:
    /// per (pair, cutoff) the minimum travel time with its exact fare
    /// and rides, flattened into columns. Rule-based fare tables only;
    /// each origin boards at its stop, and the direct walk joins each
    /// cell as the zero-fare candidate. Internal.
    #[pyo3(signature = (from_stops, to_stops, date, departure, window, fares, cutoffs,
                        max_transfers = 7, max_duration = None, exact = true,
                        departure_step = None))]
    #[allow(clippy::too_many_arguments)]
    fn _fare_frontier_table(
        &self,
        py: Python<'_>,
        from_stops: Vec<String>,
        to_stops: Vec<String>,
        date: &str,
        departure: &str,
        window: u32,
        fares: Bound<'_, PyDict>,
        cutoffs: Vec<f64>,
        max_transfers: u8,
        max_duration: Option<u32>,
        exact: bool,
        departure_step: Option<u32>,
    ) -> PyResult<Py<PyDict>> {
        if window == 0 {
            return Err(PyValueError::new_err(
                "window must be a positive number of seconds",
            ));
        }
        validated_step(window, departure_step)?;
        validated_cutoffs(&cutoffs)?;
        let tables = fare_tables(
            &fares,
            self.feed.routes.len(),
            self.build.timetable.stop_count() as usize,
        )?;
        let cafein_core::fares::FareTables::RuleBased(rules) = &tables else {
            return Err(PyValueError::new_err(
                "the fare frontier prices rule-based structures only; a \
                 zone structure's journeys price through journey_frontiers \
                 and annotate_fares",
            ));
        };
        let origins = from_stops
            .iter()
            .map(|stop| self.resolve_stop(stop))
            .collect::<PyResult<Vec<_>>>()?;
        let destinations = to_stops
            .iter()
            .map(|stop| Ok(vec![(self.resolve_stop(stop)?, 0u32)]))
            .collect::<PyResult<Vec<_>>>()?;
        let departure = parse_time(departure)?;
        if departure as u64 + window as u64 > u32::MAX as u64 {
            return Err(PyValueError::new_err(
                "departure + window overflows the router clock; narrow \
                 the window",
            ));
        }
        let active_services = self.active_services(date)?;
        let active_services_previous = self.active_services_previous(date)?;
        let inputs = FareFrontierInputs {
            timetable: &self.build.timetable,
            transfers: &self.transfers,
            fares: rules,
            cutoffs: &cutoffs,
            max_duration,
            departure_step,
            exact,
        };
        let cells: Vec<Vec<Vec<Option<_>>>> = py.allow_threads(|| {
            origins
                .par_iter()
                .map(|&origin| {
                    let request = Request {
                        departure,
                        access: vec![(origin, 0)],
                        egress: Vec::new(),
                        active_services: active_services.clone(),
                        active_services_previous: active_services_previous.clone(),
                        max_transfers,
                        exclusions: None,
                    };
                    let mut rows = frontier(&inputs, &request, &destinations, window);
                    // The direct walk joins each cell as the zero-fare
                    // candidate, exactly as the walking journey joins
                    // the shipped frontier products — boarding still
                    // starts at the origin stop.
                    for (slot, egress) in destinations.iter().enumerate() {
                        for &(stop, _) in egress {
                            let walk = if stop == origin {
                                Some(0)
                            } else {
                                self.transfers
                                    .from_stop(origin)
                                    .iter()
                                    .find(|edge| edge.to == stop)
                                    .map(|edge| edge.duration)
                            };
                            if let Some(seconds) = walk {
                                if max_duration.is_none_or(|cap| seconds <= cap) {
                                    push_walk(&mut rows[slot], &cutoffs, departure, seconds);
                                }
                            }
                        }
                    }
                    rows
                })
                .collect()
        });
        frontier_columns(py, &cells)
    }

    /// The point-to-point (time, fare) frontier: walking access and
    /// egress over the street network, the direct walk joining each
    /// cell as the zero-fare candidate, unsnapped points reported.
    /// Internal.
    #[pyo3(signature = (origins, destinations, date, departure, window, fares, cutoffs,
                        max_transfers = 7, max_duration = None, exact = true,
                        departure_step = None,
                        walking_speed_kmph = 3.6, max_walking_time = 7200.0,
                        max_snap_distance = 1600.0))]
    #[allow(clippy::too_many_arguments)]
    fn _fare_frontier_table_from_points(
        &self,
        py: Python<'_>,
        origins: Vec<(f64, f64)>,
        destinations: Vec<(f64, f64)>,
        date: &str,
        departure: &str,
        window: u32,
        fares: Bound<'_, PyDict>,
        cutoffs: Vec<f64>,
        max_transfers: u8,
        max_duration: Option<u32>,
        exact: bool,
        departure_step: Option<u32>,
        walking_speed_kmph: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
    ) -> PyResult<Py<PyDict>> {
        if window == 0 {
            return Err(PyValueError::new_err(
                "window must be a positive number of seconds",
            ));
        }
        validated_step(window, departure_step)?;
        validated_cutoffs(&cutoffs)?;
        let tables = fare_tables(
            &fares,
            self.feed.routes.len(),
            self.build.timetable.stop_count() as usize,
        )?;
        let cafein_core::fares::FareTables::RuleBased(rules) = &tables else {
            return Err(PyValueError::new_err(
                "the fare frontier prices rule-based structures only; a \
                 zone structure's journeys price through journey_frontiers \
                 and annotate_fares",
            ));
        };
        let streets = self.installed_streets()?;
        let speed =
            validated_walking_speed(walking_speed_kmph, max_walking_time, max_snap_distance)?;
        validate_points(&origins)?;
        validate_points(&destinations)?;
        let departure = parse_time(departure)?;
        if departure as u64 + window as u64 > u32::MAX as u64 {
            return Err(PyValueError::new_err(
                "departure + window overflows the router clock; narrow \
                 the window",
            ));
        }
        let active_services = self.active_services(date)?;
        let active_services_previous = self.active_services_previous(date)?;
        let inputs = FareFrontierInputs {
            timetable: &self.build.timetable,
            transfers: &self.transfers,
            fares: rules,
            cutoffs: &cutoffs,
            max_duration,
            departure_step,
            exact,
        };
        let (cells, unsnapped_from, unsnapped_to) = py.allow_threads(|| {
            let mut linked = streets.link_pointsets(
                &[&origins, &destinations],
                speed,
                max_walking_time,
                max_snap_distance,
            );
            let destination_links = linked.pop().expect("two point sets");
            let origin_links = linked.pop().expect("two point sets");
            let unsnapped_from = unsnapped(&origin_links);
            let unsnapped_to = unsnapped(&destination_links);
            let slots: Vec<Vec<(StopIdx, u32)>> = destination_links
                .iter()
                .map(|links| request_offsets(links.as_deref().unwrap_or(&[])))
                .collect();
            let walk = streets.walk_matrix(
                &origins,
                &destinations,
                speed,
                max_walking_time,
                max_snap_distance,
            );
            let cells: Vec<Vec<Vec<Option<FrontierRow>>>> = origin_links
                .par_iter()
                .enumerate()
                .map(|(i, links)| {
                    let request = Request {
                        departure,
                        access: request_offsets(links.as_deref().unwrap_or(&[])),
                        egress: Vec::new(),
                        active_services: active_services.clone(),
                        active_services_previous: active_services_previous.clone(),
                        max_transfers,
                        exclusions: None,
                    };
                    let mut rows = frontier(&inputs, &request, &slots, window);
                    for (j, cell) in walk[i].iter().enumerate() {
                        if let Some((seconds, _)) = cell {
                            if max_duration.is_none_or(|cap| *seconds <= cap) {
                                push_walk(&mut rows[j], &cutoffs, departure, *seconds);
                            }
                        }
                    }
                    rows
                })
                .collect();
            (cells, unsnapped_from, unsnapped_to)
        });
        let out = frontier_columns(py, &cells)?;
        let bound = out.bind(py);
        bound.set_item("unsnapped_from", unsnapped_from)?;
        bound.set_item("unsnapped_to", unsnapped_to)?;
        Ok(out)
    }
}
