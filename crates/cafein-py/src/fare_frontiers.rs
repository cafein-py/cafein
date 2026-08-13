//! The cutoff-pruned (time, fare) frontier product bindings.

use super::*;
use cafein_core::routers::fare_frontier::{frontier, push_walk, FareFrontierInputs, FrontierRow};
use cafein_core::routers::router::Request;
use cafein_core::routers::zone_frontier::{
    chain, zone_frontier_cutoffs, zone_frontier_warm, ChainStep, ZoneFrontierInputs, MONEY_EPSILON,
    SEED_NODE,
};
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

    /// The stop-to-stop (time, fare) frontier under a zone fare
    /// structure: the exact zone-ticket state machine, the same
    /// product shape and departure disciplines as the rule-based
    /// frontier. Always exact — there is no fast discipline to trade
    /// down to. Internal.
    #[pyo3(signature = (from_stops, to_stops, date, departure, window, fares, cutoffs,
                        max_transfers = 7, max_duration = None,
                        departure_step = None))]
    #[allow(clippy::too_many_arguments)]
    fn _zone_fare_frontier_table(
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
        let cafein_core::fares::FareTables::Zone(zones) = &tables else {
            return Err(PyValueError::new_err(
                "the zone frontier takes a zone fare structure",
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
        let top_cutoff = cutoffs.last().copied().unwrap_or(0.0);
        let inputs = ZoneFrontierInputs {
            timetable: &self.build.timetable,
            transfers: &self.transfers,
            fares: zones,
            top_cutoff,
            max_duration,
            departure_step,
            seed_walks: true,
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
                    let mut rows =
                        zone_frontier_cutoffs(&inputs, &request, &destinations, window, &cutoffs);
                    // The direct walk joins each cell as the zero-fare
                    // candidate, exactly as on the rule-based product.
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

    /// The point-to-point (time, fare) frontier under a zone fare
    /// structure: street access and egress, the direct walk as the
    /// zero-fare candidate, unsnapped points reported — the zone
    /// engine with seed walks off, since the street rows arrive
    /// walk-complete. Internal.
    #[pyo3(signature = (origins, destinations, date, departure, window, fares, cutoffs,
                        max_transfers = 7, max_duration = None,
                        departure_step = None,
                        walking_speed_kmph = 3.6, max_walking_time = 7200.0,
                        max_snap_distance = 1600.0))]
    #[allow(clippy::too_many_arguments)]
    fn _zone_fare_frontier_table_from_points(
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
        let cafein_core::fares::FareTables::Zone(zones) = &tables else {
            return Err(PyValueError::new_err(
                "the zone frontier takes a zone fare structure",
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
        let top_cutoff = cutoffs.last().copied().unwrap_or(0.0);
        let inputs = ZoneFrontierInputs {
            timetable: &self.build.timetable,
            transfers: &self.transfers,
            fares: zones,
            top_cutoff,
            max_duration,
            departure_step,
            seed_walks: false,
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
                    let mut rows =
                        zone_frontier_cutoffs(&inputs, &request, &slots, window, &cutoffs);
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

/// Refines a fare-objective fold in place to the exact zone-fare
/// optimum. Per origin: the fold's fares warm-start per-slot money
/// bounds (a fold fare at the tariff's cheapest product is provably
/// optimal and skips the exact engine — the only bound sound under
/// walks); fold-less slots climb the ceiling staircase from the
/// dearest single product, doubling to the hard cap
/// `(max_transfers + 1) x dearest`, past which emptiness means
/// genuinely unreachable. Winning chains are reconstructed from the
/// arena into full cost rows.
#[allow(clippy::too_many_arguments)]
pub(crate) fn refine_zone_fare_rows(
    timetable: &Timetable,
    transfers: &Transfers,
    zones: &cafein_core::fares::ZoneFares,
    inputs: &CostInputs<'_>,
    requests: &[Request],
    slots: &[Vec<(StopIdx, u32, f64)>],
    slot_to: &[u32],
    access_meters: Option<&[HashMap<StopIdx, f64>]>,
    seed_walks: bool,
    window: u32,
    budget: Option<u32>,
    rows: &mut [Vec<CostRow>],
) {
    let cheapest = zones
        .products
        .iter()
        .map(|product| product.price)
        .fold(f64::INFINITY, f64::min);
    let dearest = zones
        .products
        .iter()
        .map(|product| product.price)
        .fold(0.0f64, f64::max);
    if zones.products.is_empty() || !dearest.is_finite() {
        return;
    }
    // Slots close with a negative sentinel, never zero: a zero-price
    // product is legal and a zero ceiling must still run the engine.
    const CLOSED: f64 = f64::NEG_INFINITY;
    // A duration cap bounds every feasible fare through the tariff's
    // own structure: with a product covering every zone on unlimited
    // boardings, a journey lasting at most `within` partitions into
    // at most floor(within / window) + 1 such tickets. On HSL that
    // turns the staircase's 8-ticket worst case into two ABCDE
    // tickets — the difference between proving a cell empty in one
    // bounded search and climbing through combinatorial ones.
    let union = zones
        .products
        .iter()
        .fold(0u128, |mask, product| mask | product.zones);
    let span_cap = budget.and_then(|within| {
        zones
            .products
            .iter()
            .filter(|product| {
                product.zones & union == union && product.transfers == cafein_core::fares::NO_FARE
            })
            .filter_map(|product| {
                if product.duration.is_infinite() {
                    return Some(product.price);
                }
                // The engine expires tickets on whole seconds; a
                // window that truncates to zero covers nothing and
                // must not enter the bound.
                let window = product.duration.floor();
                if window.is_nan() || window < 1.0 {
                    return None;
                }
                let tickets = (within as f64 / window).floor() + 1.0;
                Some(tickets * product.price)
            })
            .fold(None, |best: Option<f64>, cap| {
                Some(best.map_or(cap, |value| value.min(cap)))
            })
    });
    let destinations: Vec<Vec<(StopIdx, u32)>> = slots
        .iter()
        .map(|links| {
            links
                .iter()
                .map(|&(stop, seconds, _)| (stop, seconds))
                .collect()
        })
        .collect();
    // Static fare-reachability: rides only alight at zoned stops (a
    // zone-less stop has no zone bit, so no ticket ever covers it)
    // and walks never chain, so every priced arrival at an egress
    // stop requires it zoned or one footpath from a zoned stop. A
    // fold-less slot failing both — and not walkable from the seeds —
    // can never price: it closes with no search, instead of climbing
    // the staircase's whole-horizon rungs to prove emptiness.
    let stop_count = timetable.stop_count() as usize;
    let zoned = |stop: StopIdx| {
        zones
            .stop_zone
            .get(stop.0 as usize)
            .is_some_and(|&zone| zone != cafein_core::fares::NO_FARE && zone < 128)
    };
    let mut from_zoned = vec![false; stop_count];
    for stop in 0..stop_count as u32 {
        if zoned(StopIdx(stop)) {
            for edge in transfers.from_stop(StopIdx(stop)) {
                from_zoned[edge.to.0 as usize] = true;
            }
        }
    }
    let statically_priceable: Vec<bool> = slots
        .iter()
        .map(|links| {
            links
                .iter()
                .any(|&(stop, _, _)| zoned(stop) || from_zoned[stop.0 as usize])
        })
        .collect();
    // Each origin's search holds bags and a reconstruction arena;
    // host-wide fan-out multiplies that peak (an unguarded run once
    // OOM-crashed a workstation). Refinement runs in its own pool,
    // never wider than four workers or the surrounding pool.
    let workers = rayon::current_num_threads().clamp(1, 4);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(workers).build();
    let per_origin = |origin: usize, origin_rows: &mut Vec<CostRow>, request: &Request| {
        {
            {
                let hard_cap = span_cap
                    .unwrap_or(f64::INFINITY)
                    .min((request.max_transfers as f64 + 1.0) * dearest);
                let fold_at: HashMap<u32, usize> = origin_rows
                    .iter()
                    .enumerate()
                    .map(|(index, row)| (row.to, index))
                    .collect();
                // Working bounds: CLOSED marks a settled slot (zero
                // stays a legitimate ceiling under zero-fare tariffs), a
                // fold fare is a real journey's price and hence a valid
                // ceiling — its travel time seeds the engine's deadline
                // index — and fold-less slots enter the staircase.
                let seed_reach = |slot: usize| {
                    slots[slot].iter().any(|&(stop, _, _)| {
                        request.access.iter().any(|&(access, _)| access == stop)
                            || (seed_walks
                                && request.access.iter().any(|&(access, _)| {
                                    transfers
                                        .from_stop(access)
                                        .iter()
                                        .any(|edge| edge.to == stop)
                                }))
                    })
                };
                let mut staircase = vec![false; slots.len()];
                let mut seeds: Vec<Option<u32>> = vec![None; slots.len()];
                let mut working: Vec<f64> = (0..slots.len())
                    .map(|slot| match fold_at.get(&slot_to[slot]) {
                        Some(&index) => {
                            let fare = origin_rows[index].fare;
                            if !fare.is_finite() {
                                if statically_priceable[slot] || seed_reach(slot) {
                                    staircase[slot] = true;
                                    dearest
                                } else {
                                    CLOSED
                                }
                            } else if fare <= cheapest + MONEY_EPSILON {
                                CLOSED
                            } else {
                                seeds[slot] = Some(origin_rows[index].seconds);
                                fare
                            }
                        }
                        None => {
                            if statically_priceable[slot] || seed_reach(slot) {
                                staircase[slot] = true;
                                dearest
                            } else {
                                CLOSED
                            }
                        }
                    })
                    .collect();
                if working.iter().all(|&bound| bound == CLOSED) {
                    return;
                }
                // A staircase slot the timetable cannot reach at all would
                // hold the money bound at its ceiling for the whole run
                // and every escalation; one earliest-arrival pass closes
                // those soundly (the engine's journeys are a subset of
                // earliest-arrival reachability).
                if staircase.iter().any(|&climbing| climbing) {
                    let arrivals = Raptor.one_to_all(timetable, transfers, request);
                    for slot in 0..slots.len() {
                        if staircase[slot]
                            && !slots[slot]
                                .iter()
                                .any(|&(stop, _, _)| arrivals[stop.0 as usize].is_some())
                        {
                            staircase[slot] = false;
                            working[slot] = CLOSED;
                        }
                    }
                    if working.iter().all(|&bound| bound == CLOSED) {
                        return;
                    }
                }
                let mut exact: Vec<Option<CostRow>> = vec![None; slots.len()];
                // One single-slot search per unsettled cell: its money
                // bound is the cell's own warm fare and its deadline the
                // cell's own best arrival — both far tighter than any
                // joint search's maxima, which is what the measurements
                // showed (per-cell searches beat one shared search by an
                // order of magnitude). Escalation stays per cell.
                for slot in 0..slots.len() {
                    while working[slot] > CLOSED {
                        let engine_inputs = ZoneFrontierInputs {
                            timetable,
                            transfers,
                            fares: zones,
                            top_cutoff: working[slot],
                            max_duration: budget,
                            departure_step: None,
                            seed_walks,
                        };
                        let (zone_rows, arena) = zone_frontier_warm(
                            &engine_inputs,
                            request,
                            &destinations[slot..slot + 1],
                            window,
                            vec![working[slot]],
                            &seeds[slot..slot + 1],
                        );
                        match &zone_rows[0] {
                            // Nothing beat the fold's own journey: the
                            // fold row stands as the exact answer.
                            Some(row) if row.node == SEED_NODE => {
                                working[slot] = CLOSED;
                            }
                            Some(row) if row.fare <= working[slot] + MONEY_EPSILON => {
                                exact[slot] = Some(zone_cost_row(
                                    slot_to[slot],
                                    row,
                                    &arena,
                                    timetable,
                                    request,
                                    &slots[slot],
                                    transfers,
                                    inputs,
                                    access_meters.map(|maps| &maps[origin]),
                                ));
                                working[slot] = CLOSED;
                            }
                            _ => {
                                if staircase[slot]
                                    && working[slot] < hard_cap
                                    && working[slot] > 0.0
                                {
                                    working[slot] = (working[slot] * 2.0).min(hard_cap);
                                } else {
                                    working[slot] = CLOSED;
                                }
                            }
                        }
                    }
                }
                let refined: Vec<CostRow> = (0..slots.len())
                    .filter_map(|slot| {
                        exact[slot].take().or_else(|| {
                            fold_at
                                .get(&slot_to[slot])
                                .map(|&index| origin_rows[index].clone())
                        })
                    })
                    .collect();
                *origin_rows = refined;
            }
        }
    };
    match pool {
        Ok(pool) => pool.install(|| {
            rows.par_iter_mut()
                .zip(requests.par_iter())
                .enumerate()
                .for_each(|(origin, (origin_rows, request))| {
                    per_origin(origin, origin_rows, request)
                })
        }),
        // No pool means no bound: one origin at a time beats the
        // surrounding pool's full width for these arena-heavy
        // searches.
        Err(_) => {
            rows.iter_mut().zip(requests.iter()).enumerate().for_each(
                |(origin, (origin_rows, request))| per_origin(origin, origin_rows, request),
            )
        }
    }
}

/// Rebuilds one exact winner's cost columns from its arena chain:
/// transit meters and grams over the ridden legs, walk meters over
/// the seed, transfer, and egress links, and the ridden geometry.
#[allow(clippy::too_many_arguments)]
fn zone_cost_row(
    to: u32,
    row: &cafein_core::routers::zone_frontier::ZoneRow,
    arena: &[cafein_core::routers::zone_frontier::Node],
    timetable: &Timetable,
    request: &Request,
    egress_links: &[(StopIdx, u32, f64)],
    transfers: &Transfers,
    inputs: &CostInputs<'_>,
    access_meters: Option<&HashMap<StopIdx, f64>>,
) -> CostRow {
    let (_, seed_stop, steps) = chain(arena, row.node);
    let mut transit_meters = 0.0f64;
    let mut walk_meters = 0.0f64;
    let mut grams = 0.0f64;
    let mut resolved = true;
    let mut parts: Vec<Vec<(f64, f64)>> = Vec::new();
    let mut at = seed_stop;
    let mut last_alight = NO_STOP;
    let walk_between = |from: StopIdx, to: StopIdx| {
        transfers
            .from_stop(from)
            .iter()
            .find(|edge| edge.to == to)
            .map(|edge| edge.meters)
            .unwrap_or(0.0)
    };
    match access_meters {
        // Point access rows carry their own walked meters.
        Some(meters) => walk_meters += meters.get(&seed_stop).copied().unwrap_or(0.0),
        // Stop queries walk a footpath when the seed boarded elsewhere.
        None => {
            if request.access.iter().all(|&(stop, _)| stop != seed_stop) {
                if let Some(&(origin, _)) = request.access.first() {
                    walk_meters += walk_between(origin, seed_stop);
                }
            }
        }
    }
    for step in &steps {
        match *step {
            ChainStep::Ride {
                trip,
                board_position,
                alight_position,
                ..
            } => {
                let meters = inputs
                    .geometry
                    .leg_distance(trip, board_position, alight_position)
                    as f64;
                transit_meters += meters;
                let factor = inputs.factors[trip.0 as usize];
                if factor.is_finite() {
                    grams += meters / 1000.0 * factor;
                } else {
                    resolved = false;
                }
                if inputs.with_geometry {
                    if let Some(shapes) = inputs.leg_geometry {
                        parts.push(shapes.leg_coordinates(trip, board_position, alight_position));
                    }
                }
                let stops = timetable.pattern_stops(timetable.trip_pattern(trip));
                at = stops[alight_position as usize];
                last_alight = at.0;
            }
            ChainStep::Walk { from, to } => {
                walk_meters += walk_between(from, to);
                at = to;
            }
        }
    }
    // The egress link that joined the destination, when one did.
    walk_meters += egress_links
        .iter()
        .find(|&&(stop, seconds, _)| stop == at && seconds == row.egress_seconds)
        .map(|&(_, _, meters)| meters)
        .unwrap_or(0.0);
    let rode = steps
        .iter()
        .any(|step| matches!(step, ChainStep::Ride { .. }));
    CostRow {
        to,
        access_stop: if rode { seed_stop.0 } else { NO_STOP },
        egress_stop: last_alight,
        seconds: row.travel_time,
        rides: row.rides as u32,
        transit_meters,
        walk_meters,
        street_meters: 0.0,
        rental_transfers: 0,
        emission_grams: if resolved { grams } else { f64::NAN },
        fare: row.fare,
        geometry: if inputs.with_geometry {
            Some(wkb_multi_line_string(&parts))
        } else {
            None
        },
    }
}
