//! The matrix entry points: one-to-all, cost, least-cost, and
//! percentile forms, fanned out over origins.

use super::*;

impl TransitRouter for Raptor {
    fn route(
        &self,
        timetable: &Timetable,
        transfers: &Transfers,
        request: &Request,
    ) -> Vec<Journey> {
        Search::new(timetable, transfers, request).profile(&[request.departure])
    }
}

impl Raptor {
    /// Earliest arrival at every stop for a single departure.
    ///
    /// One run serves all destinations — the matrix primitive: matrices
    /// are computed origin by origin, never per OD pair. The request's
    /// egress list is not consulted; unreachable stops are `None`.
    pub fn one_to_all(
        &self,
        timetable: &Timetable,
        transfers: &Transfers,
        request: &Request,
    ) -> Vec<Option<u32>> {
        let mut search = Search::new(timetable, transfers, request);
        search.run(request.departure);
        search.arrivals()
    }

    /// The fastest journey's aggregated costs from each request to each
    /// destination — the cost-matrix computation, fanned out over the
    /// origins with rayon like [`Raptor::one_to_all_many`].
    ///
    /// Rows come back per origin, reachable destinations only, in
    /// destination order. Deterministic regardless of scheduling.
    pub fn cost_matrix(
        &self,
        timetable: &Timetable,
        transfers: &Transfers,
        inputs: &CostInputs<'_>,
        requests: &[Request],
        destinations: &[StopIdx],
    ) -> Vec<Vec<CostRow>> {
        requests
            .par_iter()
            .map_init(
                || None,
                |pooled: &mut Option<Search>, request| {
                    let search = match pooled {
                        Some(search) if search.rounds == request.max_transfers as usize + 1 => {
                            search.reset(request);
                            search
                        }
                        _ => pooled.insert(Search::new(timetable, transfers, request)),
                    };
                    search.run(request.departure);
                    destinations
                        .iter()
                        .filter_map(|&stop| search.costs_to(stop, request.departure, inputs, None))
                        .collect()
                },
            )
            .collect()
    }

    /// The fastest journey's aggregated costs from each request to each
    /// destination *point*, joined through the points' egress link
    /// tables — the pointset cost matrix.
    ///
    /// `access_meters` gives each request's access-walk lengths keyed by
    /// entry stop; `egress` gives each destination point's
    /// `(stop, seconds, meters)` links. A point's travel time is the
    /// minimum over its links of the arrival at the link's stop plus the
    /// egress walk; its costs are the winning journey's, with the access
    /// and egress walks added to `walk_meters`.
    pub fn cost_matrix_to_points(
        &self,
        timetable: &Timetable,
        transfers: &Transfers,
        inputs: &CostInputs<'_>,
        requests: &[Request],
        access_meters: &[HashMap<StopIdx, f64>],
        egress: &[Vec<(StopIdx, u32, f64)>],
    ) -> Vec<Vec<CostRow>> {
        assert_eq!(requests.len(), access_meters.len());
        requests
            .par_iter()
            .zip(access_meters.par_iter())
            .map_init(
                || None,
                |pooled: &mut Option<Search>, (request, access)| {
                    let search = match pooled {
                        Some(search) if search.rounds == request.max_transfers as usize + 1 => {
                            search.reset(request);
                            search
                        }
                        _ => pooled.insert(Search::new(timetable, transfers, request)),
                    };
                    search.run(request.departure);
                    egress
                        .iter()
                        .enumerate()
                        .filter_map(|(point, links)| {
                            search.costs_to_point(
                                point as u32,
                                links,
                                request.departure,
                                inputs,
                                access,
                            )
                        })
                        .collect()
                },
            )
            .collect()
    }

    /// Earliest arrival at every stop for each request — the matrix
    /// computation, fanned out over the origins with rayon.
    ///
    /// Search state is pooled: a worker reuses its buffers across the
    /// origins it processes instead of reallocating per query. Shared
    /// inputs are immutable, and each result depends only on its own
    /// request, so the output is deterministic regardless of how rayon
    /// schedules the origins.
    pub fn one_to_all_many(
        &self,
        timetable: &Timetable,
        transfers: &Transfers,
        requests: &[Request],
    ) -> Vec<Vec<Option<u32>>> {
        requests
            .par_iter()
            .map_init(
                || None,
                |pooled: &mut Option<Search>, request| {
                    let search = match pooled {
                        Some(search) if search.rounds == request.max_transfers as usize + 1 => {
                            search.reset(request);
                            search
                        }
                        _ => pooled.insert(Search::new(timetable, transfers, request)),
                    };
                    search.run(request.departure);
                    search.arrivals()
                },
            )
            .collect()
    }

    /// The objective-best journey's aggregated costs within a
    /// travel-time budget, per destination, over a departure window —
    /// the emissions/fare counterpart of [`Raptor::cost_matrix`]. One
    /// descending range scan per origin; each pass's improved
    /// per-round arrivals are reconstructed and folded, so the
    /// candidates are the profile's (departure, arrival, rides)-Pareto
    /// set and a cell holds its lowest-objective member within the
    /// budget (no budget: within the window's reach). Destinations
    /// with no qualifying candidate (a resolved emission, a priceable
    /// fare) are absent.
    #[allow(clippy::too_many_arguments)]
    pub fn least_cost_matrix(
        &self,
        timetable: &Timetable,
        transfers: &Transfers,
        inputs: &CostInputs<'_>,
        requests: &[Request],
        destinations: &[StopIdx],
        window: u32,
        budget: Option<u32>,
        objective: Objective,
    ) -> Vec<Vec<CostRow>> {
        requests
            .par_iter()
            .map_init(
                || None,
                |pooled: &mut Option<Search>, request| {
                    let search = match pooled {
                        Some(search) if search.rounds == request.max_transfers as usize + 1 => {
                            search.reset(request);
                            search
                        }
                        _ => pooled.insert(Search::new(timetable, transfers, request)),
                    };
                    let departures = departure_candidates(timetable, request, window);
                    let mut thresholds = vec![UNREACHED; destinations.len() * (search.rounds + 1)];
                    let mut best: Vec<Option<CostRow>> = vec![None; destinations.len()];
                    for &departure in &departures {
                        search.run(departure);
                        search.fold_best(
                            departure,
                            destinations,
                            budget,
                            inputs,
                            None,
                            objective,
                            &mut thresholds,
                            &mut best,
                        );
                    }
                    best.into_iter().flatten().collect()
                },
            )
            .collect()
    }

    /// [`Raptor::least_cost_matrix`] over destination points, joined
    /// through the points' egress link tables like
    /// [`Raptor::cost_matrix_to_points`].
    #[allow(clippy::too_many_arguments)]
    pub fn least_cost_matrix_to_points(
        &self,
        timetable: &Timetable,
        transfers: &Transfers,
        inputs: &CostInputs<'_>,
        requests: &[Request],
        access_meters: &[HashMap<StopIdx, f64>],
        egress: &[Vec<(StopIdx, u32, f64)>],
        window: u32,
        budget: Option<u32>,
        objective: Objective,
    ) -> Vec<Vec<CostRow>> {
        assert_eq!(requests.len(), access_meters.len());
        requests
            .par_iter()
            .zip(access_meters.par_iter())
            .map_init(
                || None,
                |pooled: &mut Option<Search>, (request, access)| {
                    let search = match pooled {
                        Some(search) if search.rounds == request.max_transfers as usize + 1 => {
                            search.reset(request);
                            search
                        }
                        _ => pooled.insert(Search::new(timetable, transfers, request)),
                    };
                    let departures = departure_candidates(timetable, request, window);
                    let mut thresholds = vec![UNREACHED; egress.len() * (search.rounds + 1)];
                    let mut best: Vec<Option<CostRow>> = vec![None; egress.len()];
                    for &departure in &departures {
                        search.run(departure);
                        search.fold_best_points(
                            departure,
                            egress,
                            budget,
                            inputs,
                            access,
                            objective,
                            &mut thresholds,
                            &mut best,
                        );
                    }
                    best.into_iter().flatten().collect()
                },
            )
            .collect()
    }

    /// Travel-time percentiles over a departure window, per origin —
    /// the windowed matrix primitive, fanned out over origins with rayon.
    ///
    /// Every minute mark within `[departure, departure + window)` is
    /// evaluated: one descending rRAPTOR scan per origin on shared
    /// state yields, at each mark, the earliest arrival when leaving at
    /// or after it. The samples are therefore the full minute-level
    /// departure population, and the returned values are exact
    /// nearest-rank percentiles of the travel-time distribution across
    /// the window. Walking-only reachability (the access list) is
    /// departure-independent and overlays every sample.
    ///
    /// Returns, per request, `stop_count × percentiles.len()` travel
    /// times flat by stop; `u32::MAX` marks an unreachable percentile.
    pub fn percentile_matrix(
        &self,
        timetable: &Timetable,
        transfers: &Transfers,
        requests: &[Request],
        window: u32,
        percentiles: &[f64],
    ) -> Vec<Vec<u32>> {
        let stop_count = timetable.stop_count() as usize;
        requests
            .par_iter()
            .map_init(
                || None,
                |pooled: &mut Option<Search>, request| {
                    let arrivals = window_samples(pooled, timetable, transfers, request, window);
                    let access_floor = access_floor(stop_count, request);
                    let mut out = Vec::with_capacity(stop_count * percentiles.len());
                    let mut samples = vec![0u32; arrivals.len()];
                    for stop in 0..stop_count {
                        for (sample, (mark, marked)) in samples.iter_mut().zip(&arrivals) {
                            *sample = travel_time(marked[stop], *mark, access_floor[stop]);
                        }
                        samples.sort_unstable();
                        for &percentile in percentiles {
                            out.push(nearest_rank(&samples, percentile));
                        }
                    }
                    out
                },
            )
            .collect()
    }

    /// Travel-time percentiles over a departure window to destination
    /// points, joined through the points' egress link tables — the
    /// windowed pointset matrix. Semantics follow
    /// [`Raptor::percentile_matrix`], with each mark's arrival at a
    /// point being the minimum over its links of the arrival at the
    /// link's stop plus the egress walk.
    ///
    /// Returns, per request, `egress.len() × percentiles.len()` travel
    /// times flat by point; `u32::MAX` marks an unreachable percentile.
    pub fn percentile_matrix_to_points(
        &self,
        timetable: &Timetable,
        transfers: &Transfers,
        requests: &[Request],
        egress: &[Vec<(StopIdx, u32, f64)>],
        window: u32,
        percentiles: &[f64],
    ) -> Vec<Vec<u32>> {
        let stop_count = timetable.stop_count() as usize;
        requests
            .par_iter()
            .map_init(
                || None,
                |pooled: &mut Option<Search>, request| {
                    let arrivals = window_samples(pooled, timetable, transfers, request, window);
                    let access_floor = access_floor(stop_count, request);
                    propagate_point_percentiles(
                        &arrivals,
                        &access_floor,
                        stop_count,
                        egress,
                        percentiles,
                    )
                },
            )
            .collect()
    }

    /// Routes over a departure window: the Pareto set of journeys over
    /// (departure, arrival, rides) for departures within
    /// `[request.departure, request.departure + window)`.
    ///
    /// Each journey's departure is the latest time the origin can be left
    /// to catch it, capped at the window's final second — a journey that
    /// leaves within the window but waits for a ride beyond it is
    /// reported with that final second as its departure. So unlike
    /// [`TransitRouter::route`] — which answers "leaving exactly at the
    /// requested time" — the result enumerates the distinct departure
    /// choices the window offers, sorted by departure and then by ride
    /// count. A zero-length window has no departures.
    pub fn route_range(
        &self,
        timetable: &Timetable,
        transfers: &Transfers,
        request: &Request,
        window: u32,
    ) -> Vec<Journey> {
        let departures = departure_candidates(timetable, request, window);
        Search::new(timetable, transfers, request).profile(&departures)
    }
}

/// The origin departure times within `[request.departure,
/// request.departure + window)` at which some trip becomes catchable: one
/// candidate per active-service trip departure at an access stop, shifted
/// back by the stop's access duration. Descending, deduplicated. Shared
/// with the TBTR range router so both enumerate identical windows.
pub(crate) fn departure_candidates(
    timetable: &Timetable,
    request: &Request,
    window: u32,
) -> Vec<u32> {
    // Widened so a window reaching past u32::MAX cannot clip candidates.
    let end = request.departure as u64 + window as u64;
    let mut candidates = Vec::new();
    for &(stop, duration) in &request.access {
        for pattern_stop in timetable.patterns_at_stop(stop) {
            let position = pattern_stop.position as usize;
            // Boarding at a pattern's last position is pointless.
            if position + 1 == timetable.pattern_stops(pattern_stop.pattern).len() {
                continue;
            }
            for trip in timetable.pattern_trips(pattern_stop.pattern) {
                let service = timetable.trip_service(trip) as usize;
                let stored = timetable.trip_stop_times(trip)[position].departure;
                let active_today = request
                    .active_services
                    .get(service)
                    .copied()
                    .unwrap_or(false);
                let active_previous = request
                    .active_services_previous
                    .get(service)
                    .copied()
                    .unwrap_or(false);
                // Today's trips depart at their stored time; the previous
                // day's a day earlier.
                let departures = [
                    active_today.then_some(stored),
                    active_previous
                        .then(|| stored.checked_sub(DAY_SECONDS))
                        .flatten(),
                ];
                for trip_departure in departures.into_iter().flatten() {
                    let Some(origin_departure) = trip_departure.checked_sub(duration) else {
                        continue;
                    };
                    if origin_departure >= request.departure && (origin_departure as u64) < end {
                        candidates.push(origin_departure);
                    }
                }
            }
        }
    }
    // The window's final second is always a candidate: it covers journeys
    // that leave within the window and wait for a ride beyond it.
    if window > 0 {
        candidates.push((end - 1).min(u32::MAX as u64) as u32);
    }
    candidates.sort_unstable_by(|left, right| right.cmp(left));
    candidates.dedup();
    candidates
}

/// One descending rRAPTOR scan over a request's departure window: for
/// every minute mark within `[departure, departure + window)`,
/// ascending, the per-stop earliest arrivals when leaving at or after
/// that mark. The pooled search is rebuilt when the round count changes.
pub(super) fn window_samples<'a>(
    pooled: &mut Option<Search<'a>>,
    timetable: &'a Timetable,
    transfers: &'a Transfers,
    request: &'a Request,
    window: u32,
) -> Vec<(u32, Vec<u32>)> {
    let search = match pooled {
        Some(search) if search.rounds == request.max_transfers as usize + 1 => {
            search.reset(request);
            search
        }
        _ => pooled.insert(Search::new(timetable, transfers, request)),
    };
    let sample_count = (window as u64).div_ceil(60).max(1) as u32;
    let mut samples = Vec::with_capacity(sample_count as usize);
    for step in (0..sample_count).rev() {
        let Some(mark) = request.departure.checked_add(step * 60) else {
            continue;
        };
        // One pass per minute mark, descending, bags shared (range-RAPTOR).
        // A pass at `mark` boards every trip departing at or after `mark`, so
        // after it the bags hold exactly the earliest arrivals for leaving at
        // or after `mark` — running extra passes at the individual trip
        // departures in between adds nothing to the minute-mark samples.
        search.run(mark);
        samples.push((
            mark,
            search
                .best
                .last()
                .expect("search always has a round")
                .clone(),
        ));
    }
    samples.reverse();
    samples
}

/// Propagates one origin's per-mark stop arrivals to destination points —
/// the door-to-door percentile reduction shared by the RAPTOR and TBTR
/// windowed point matrices. `arrivals` holds `(mark, per-stop earliest
/// arrival)` samples ascending by mark; `egress` gives each point's
/// `(stop, seconds, meters)` links. Transposes the samples into a
/// stop-major table (R5's `invertTravelTimes`) so a destination reads each
/// nearby stop's iterations as one contiguous block, floors every sample
/// by the point's departure-independent walking-only time, and emits
/// nearest-rank percentiles per point.
pub(crate) fn propagate_point_percentiles(
    arrivals: &[(u32, Vec<u32>)],
    access_floor: &[u32],
    stop_count: usize,
    egress: &[Vec<(StopIdx, u32, f64)>],
    percentiles: &[f64],
) -> Vec<u32> {
    let iterations = arrivals.len();
    let mut to_stop = vec![UNREACHED; stop_count * iterations];
    let mut departures = vec![0u32; iterations];
    for (iteration, (mark, marked)) in arrivals.iter().enumerate() {
        departures[iteration] = *mark;
        for (stop, &at_stop) in marked.iter().enumerate() {
            to_stop[stop * iterations + iteration] = at_stop;
        }
    }
    let mut out = Vec::with_capacity(egress.len() * percentiles.len());
    let mut at_point = vec![UNREACHED; iterations];
    let mut samples = vec![0u32; iterations];
    for links in egress {
        // The walking-only floor through any link is departure-independent,
        // like the access floor.
        let mut walk_floor = UNREACHED;
        for &(stop, seconds, _) in links {
            let floor = access_floor[stop.0 as usize];
            if floor != UNREACHED {
                walk_floor = walk_floor.min(floor.saturating_add(seconds));
            }
        }
        for slot in at_point.iter_mut() {
            *slot = UNREACHED;
        }
        // Propagate every iteration from each nearby stop, reading the
        // stop's contiguous iteration block.
        for &(stop, seconds, _) in links {
            let base = stop.0 as usize * iterations;
            for (slot, &at_stop) in at_point.iter_mut().zip(&to_stop[base..base + iterations]) {
                if at_stop == UNREACHED {
                    continue;
                }
                if let Some(arrival) = at_stop.checked_add(seconds).filter(|&at| at != UNREACHED) {
                    *slot = (*slot).min(arrival);
                }
            }
        }
        for (sample, (&at, &mark)) in samples.iter_mut().zip(at_point.iter().zip(&departures)) {
            *sample = travel_time(at, mark, walk_floor);
        }
        samples.sort_unstable();
        for &percentile in percentiles {
            out.push(nearest_rank(&samples, percentile));
        }
    }
    out
}

/// Walking-only travel times from the request's access list, by stop.
pub(crate) fn access_floor(stop_count: usize, request: &Request) -> Vec<u32> {
    let mut floor = vec![UNREACHED; stop_count];
    for &(stop, duration) in &request.access {
        let slot = &mut floor[stop.0 as usize];
        *slot = (*slot).min(duration);
    }
    floor
}

/// One travel-time sample: the transit arrival over the mark, floored
/// by the departure-independent walking-only time.
pub(crate) fn travel_time(arrival: u32, mark: u32, walk_floor: u32) -> u32 {
    let transit = if arrival == UNREACHED {
        UNREACHED
    } else {
        arrival - mark
    };
    transit.min(walk_floor)
}

/// The nearest-rank percentile of ascending samples; ranks exactly
/// between two samples round up (the upper median), keeping the
/// convention reproducible across languages.
pub(crate) fn nearest_rank(sorted: &[u32], percentile: f64) -> u32 {
    let position = (percentile / 100.0) * (sorted.len() - 1) as f64;
    sorted[((position + 0.5).floor() as usize).min(sorted.len() - 1)]
}
