//! The cutoff-pruned (time, fare) frontier product bindings.

use super::*;
use cafein_core::routers::fare_frontier::{fold_cutoffs, frontier, FareFrontierInputs};
use cafein_core::routers::router::Request;

use crate::cost_matrices::fare_tables;

#[pymethods]
impl TransportNetwork {
    /// The stop-to-stop (time, fare) frontier over a departure window:
    /// per (pair, cutoff) the minimum travel time with its exact fare
    /// and rides, flattened into columns. Rule-based fare tables only;
    /// each origin boards at its stop, and the direct walk joins each
    /// cell as the zero-fare candidate. Internal.
    #[pyo3(signature = (from_stops, to_stops, date, departure, window, fares, cutoffs,
                        max_transfers = 7, max_duration = None))]
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
    ) -> PyResult<Py<PyDict>> {
        if window == 0 {
            return Err(PyValueError::new_err(
                "window must be a positive number of seconds",
            ));
        }
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
        let active_services = self.active_services(date)?;
        let active_services_previous = self.active_services_previous(date)?;
        let inputs = FareFrontierInputs {
            timetable: &self.build.timetable,
            transfers: &self.transfers,
            fares: rules,
            cutoffs: &cutoffs,
            max_duration,
        };
        let cells: Vec<Vec<Vec<Option<_>>>> = py.allow_threads(|| {
            origins
                .iter()
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
                    let mut arrivals = frontier(&inputs, &request, &destinations, window);
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
                                    cafein_core::routers::fare_frontier::push_walk(
                                        &mut arrivals[slot],
                                        departure,
                                        seconds,
                                    );
                                }
                            }
                        }
                    }
                    arrivals
                        .iter()
                        .map(|slot_arrivals| fold_cutoffs(slot_arrivals, &cutoffs))
                        .collect()
                })
                .collect()
        });
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
}
