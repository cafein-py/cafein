//! RAPTOR: earliest-arrival routing for a single departure time, and its
//! range (rRAPTOR) extension for a departure window.
//!
//! Round-based: round `k` finds the earliest arrivals reachable with
//! exactly `k` rides. Within a pattern, the earliest catchable trip at a
//! stop position is found by binary search over departures, which is valid
//! at every position because the timetable's patterns are FIFO chains.
//!
//! The range query runs one pass per candidate departure time, in
//! decreasing order, on shared state: arrivals found for a later departure
//! stay feasible for every earlier one, so each pass explores only what
//! its departure improves, and journeys dominated by a later departure are
//! never emitted.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::geometry::{wkb_multi_line_string, LegGeometry, TripGeometry};
use crate::journey::{Journey, Leg};
use crate::router::{Request, TransitRouter};
use crate::timetable::{PatternIdx, StopIdx, Timetable, TripIdx};
use crate::transfers::Transfers;

/// The RAPTOR router.
pub struct Raptor;

/// Aggregated costs of the fastest journey to one destination.
#[derive(Debug, Clone, PartialEq)]
pub struct CostRow {
    /// The destination's index: a stop index for stop matrices, a
    /// destination-point index for pointset matrices.
    pub to: u32,
    /// Travel time in seconds from the requested departure.
    pub seconds: u32,
    /// Number of transit legs; 0 for a destination reached on foot.
    pub rides: u32,
    /// Distance ridden on transit, in meters.
    pub transit_meters: f64,
    /// Distance walked on transfers and access links, in meters.
    pub walk_meters: f64,
    /// Grams CO₂e over the ridden legs; NaN when a ridden trip has no
    /// emission factor.
    pub emission_grams: f64,
    /// The ridden legs' geometry as a WKB MultiLineString, when asked
    /// for and leg geometries are installed.
    pub geometry: Option<Vec<u8>>,
}

/// Everything the cost reconstruction reads besides the search state.
pub struct CostInputs<'a> {
    /// Per-trip cumulative distances (drives meters and emissions).
    pub geometry: &'a TripGeometry,
    /// Grams CO₂e per passenger-kilometer per trip, indexed by trip;
    /// NaN marks a trip without a resolved factor.
    pub factors: &'a [f64],
    /// Leg polylines; required to emit geometries.
    pub leg_geometry: Option<&'a LegGeometry>,
    /// Emit each row's WKB MultiLineString.
    pub with_geometry: bool,
}

const UNREACHED: u32 = u32::MAX;

/// Keeps the cleaner of an existing candidate and a challenger: fewer
/// grams win, equal grams resolve toward the shorter travel time. NaN
/// grams (an unresolved emission factor) never qualify.
fn fold_cleaner(current: &mut Option<CostRow>, challenger: CostRow) {
    if challenger.emission_grams.is_nan() {
        return;
    }
    let better = match current {
        None => true,
        Some(row) => {
            challenger.emission_grams < row.emission_grams
                || (challenger.emission_grams == row.emission_grams
                    && challenger.seconds < row.seconds)
        }
    };
    if better {
        *current = Some(challenger);
    }
}

/// Seconds in a service day: a previous-day trip's stored times are shifted
/// back by this to place it on the queried day's clock.
const DAY_SECONDS: u32 = 86_400;

/// How a stop's arrival time in a round was achieved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Label {
    Unreached,
    /// Reached directly from the origin.
    Access,
    /// Alighted from a trip boarded at `board_position` of its pattern.
    /// `day_offset` is subtracted from the trip's stored times to place
    /// them on the queried day (nonzero for a previous-day trip).
    Transit {
        trip: TripIdx,
        board_position: u16,
        alight_position: u16,
        day_offset: u32,
    },
    /// Walked from another stop reached by transit in the same round.
    Transfer {
        from_stop: StopIdx,
        duration: u32,
    },
}

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

    /// The cleanest journey's aggregated costs within a travel-time
    /// budget, per destination, over a departure window — the emissions
    /// counterpart of [`Raptor::cost_matrix`]. One descending range
    /// scan per origin; each pass's improved per-round arrivals are
    /// reconstructed and folded, so the candidates are the profile's
    /// (departure, arrival, rides)-Pareto set and a cell holds its
    /// lowest-emission member within the budget (no budget: within the
    /// window's reach). Destinations with no qualifying resolved-
    /// emissions candidate are absent.
    #[allow(clippy::too_many_arguments)]
    pub fn least_emission_matrix(
        &self,
        timetable: &Timetable,
        transfers: &Transfers,
        inputs: &CostInputs<'_>,
        requests: &[Request],
        destinations: &[StopIdx],
        window: u32,
        budget: Option<u32>,
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
                    let mut cleanest: Vec<Option<CostRow>> = vec![None; destinations.len()];
                    for &departure in &departures {
                        search.run(departure);
                        search.fold_cleanest(
                            departure,
                            destinations,
                            budget,
                            inputs,
                            None,
                            &mut thresholds,
                            &mut cleanest,
                        );
                    }
                    cleanest.into_iter().flatten().collect()
                },
            )
            .collect()
    }

    /// [`Raptor::least_emission_matrix`] over destination points, joined
    /// through the points' egress link tables like
    /// [`Raptor::cost_matrix_to_points`].
    #[allow(clippy::too_many_arguments)]
    pub fn least_emission_matrix_to_points(
        &self,
        timetable: &Timetable,
        transfers: &Transfers,
        inputs: &CostInputs<'_>,
        requests: &[Request],
        access_meters: &[HashMap<StopIdx, f64>],
        egress: &[Vec<(StopIdx, u32, f64)>],
        window: u32,
        budget: Option<u32>,
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
                    let mut cleanest: Vec<Option<CostRow>> = vec![None; egress.len()];
                    for &departure in &departures {
                        search.run(departure);
                        search.fold_cleanest_points(
                            departure,
                            egress,
                            budget,
                            inputs,
                            access,
                            &mut thresholds,
                            &mut cleanest,
                        );
                    }
                    cleanest.into_iter().flatten().collect()
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
                    let mut out = Vec::with_capacity(egress.len() * percentiles.len());
                    let mut samples = vec![0u32; arrivals.len()];
                    for links in egress {
                        // The walking-only floor through any link is
                        // departure-independent, like the access floor.
                        let mut walk_floor = UNREACHED;
                        for &(stop, seconds, _) in links {
                            let floor = access_floor[stop.0 as usize];
                            if floor != UNREACHED {
                                walk_floor = walk_floor.min(floor.saturating_add(seconds));
                            }
                        }
                        for (sample, (mark, marked)) in samples.iter_mut().zip(&arrivals) {
                            let mut at_point = UNREACHED;
                            for &(stop, seconds, _) in links {
                                let at_stop = marked[stop.0 as usize];
                                if at_stop == UNREACHED {
                                    continue;
                                }
                                if let Some(arrival) =
                                    at_stop.checked_add(seconds).filter(|&at| at != UNREACHED)
                                {
                                    at_point = at_point.min(arrival);
                                }
                            }
                            *sample = travel_time(at_point, *mark, walk_floor);
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
/// back by the stop's access duration. Descending, deduplicated.
fn departure_candidates(timetable: &Timetable, request: &Request, window: u32) -> Vec<u32> {
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
fn window_samples<'a>(
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
    let candidates = departure_candidates(timetable, request, window);
    let mut next_candidate = 0;
    let sample_count = (window as u64).div_ceil(60).max(1) as u32;
    let mut samples = Vec::with_capacity(sample_count as usize);
    for step in (0..sample_count).rev() {
        let Some(mark) = request.departure.checked_add(step * 60) else {
            continue;
        };
        while next_candidate < candidates.len() && candidates[next_candidate] >= mark {
            search.run(candidates[next_candidate]);
            next_candidate += 1;
        }
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

/// Walking-only travel times from the request's access list, by stop.
fn access_floor(stop_count: usize, request: &Request) -> Vec<u32> {
    let mut floor = vec![UNREACHED; stop_count];
    for &(stop, duration) in &request.access {
        let slot = &mut floor[stop.0 as usize];
        *slot = (*slot).min(duration);
    }
    floor
}

/// One travel-time sample: the transit arrival over the mark, floored
/// by the departure-independent walking-only time.
fn travel_time(arrival: u32, mark: u32, walk_floor: u32) -> u32 {
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
fn nearest_rank(sorted: &[u32], percentile: f64) -> u32 {
    let position = (percentile / 100.0) * (sorted.len() - 1) as f64;
    sorted[((position + 0.5).floor() as usize).min(sorted.len() - 1)]
}

/// RAPTOR state shared by the passes of one query.
struct Search<'a> {
    timetable: &'a Timetable,
    transfers: &'a Transfers,
    request: &'a Request,
    rounds: usize,
    /// Per-round arrival times; `tau[k][stop]` is the earliest arrival at
    /// `stop` with exactly `k` rides, over all departures processed so far.
    tau: Vec<Vec<u32>>,
    labels: Vec<Vec<Label>>,
    /// Prefix minimum of `tau`: `best[k][stop]` is the earliest arrival at
    /// `stop` with at most `k` rides, over all departures processed so
    /// far. Pruning at a round must not consult later rounds — a faster
    /// but more-rides arrival from a later departure does not dominate
    /// the fewer-ride option an earlier departure still offers.
    best: Vec<Vec<u32>>,
    marked: Vec<StopIdx>,
    is_marked: Vec<bool>,
    /// First marked position per pattern for the current round.
    queue_position: Vec<u16>,
    queued_patterns: Vec<PatternIdx>,
}

impl<'a> Search<'a> {
    fn new(timetable: &'a Timetable, transfers: &'a Transfers, request: &'a Request) -> Self {
        let stop_count = timetable.stop_count() as usize;
        let rounds = request.max_transfers as usize + 1;
        Search {
            timetable,
            transfers,
            request,
            rounds,
            tau: vec![vec![UNREACHED; stop_count]; rounds + 1],
            labels: vec![vec![Label::Unreached; stop_count]; rounds + 1],
            best: vec![vec![UNREACHED; stop_count]; rounds + 1],
            marked: Vec::new(),
            is_marked: vec![false; stop_count],
            queue_position: vec![u16::MAX; timetable.pattern_count() as usize],
            queued_patterns: Vec::new(),
        }
    }

    /// Clears the per-query state for a new request, reusing the
    /// allocated buffers. The request must keep the round count: callers
    /// pooling a search across origins hold `max_transfers` fixed.
    fn reset(&mut self, request: &'a Request) {
        debug_assert_eq!(self.rounds, request.max_transfers as usize + 1);
        self.request = request;
        for tau in &mut self.tau {
            tau.fill(UNREACHED);
        }
        for labels in &mut self.labels {
            labels.fill(Label::Unreached);
        }
        for best in &mut self.best {
            best.fill(UNREACHED);
        }
        self.marked.clear();
        self.is_marked.fill(false);
        // `queue_position` needs no reset: every pass restores it to
        // u16::MAX as patterns are dequeued.
    }

    /// The earliest arrival at every stop over all processed departures,
    /// with any number of rides; unreachable stops are `None`.
    fn arrivals(&self) -> Vec<Option<u32>> {
        self.best
            .last()
            .expect("search always has a round")
            .iter()
            .map(|&arrival| (arrival != UNREACHED).then_some(arrival))
            .collect()
    }

    /// The fastest journey's aggregated costs to a destination point
    /// over its egress links; `None` when no link's stop is reachable.
    fn costs_to_point(
        &self,
        point: u32,
        egress: &[(StopIdx, u32, f64)],
        departure: u32,
        inputs: &CostInputs<'_>,
        access_meters: &HashMap<StopIdx, f64>,
    ) -> Option<CostRow> {
        let best = self.best.last().expect("search always has a round");
        let mut chosen: Option<(u32, StopIdx, f64)> = None;
        for &(stop, seconds, meters) in egress {
            let at_stop = best[stop.0 as usize];
            if at_stop == UNREACHED {
                continue;
            }
            let Some(arrival) = at_stop.checked_add(seconds).filter(|&at| at != UNREACHED) else {
                continue;
            };
            if chosen.is_none_or(|(current, _, _)| arrival < current) {
                chosen = Some((arrival, stop, meters));
            }
        }
        let (arrival, stop, egress_meters) = chosen?;
        let mut row = self.costs_to(stop, departure, inputs, Some(access_meters))?;
        row.to = point;
        row.seconds = arrival - departure;
        row.walk_meters += egress_meters;
        Some(row)
    }

    /// The fastest journey's aggregated costs to `stop`, walking the
    /// label chain of the round that achieves the earliest arrival;
    /// `None` when the stop is unreachable.
    fn costs_to(
        &self,
        stop: StopIdx,
        departure: u32,
        inputs: &CostInputs<'_>,
        access_meters: Option<&HashMap<StopIdx, f64>>,
    ) -> Option<CostRow> {
        let mut best_round = 0;
        let mut best_arrival = UNREACHED;
        for round in 0..=self.rounds {
            let arrival = self.tau[round][stop.0 as usize];
            if arrival < best_arrival {
                best_arrival = arrival;
                best_round = round;
            }
        }
        if best_arrival == UNREACHED {
            return None;
        }
        let mut row = self.walk_costs(stop, best_round, inputs, access_meters);
        row.seconds = best_arrival - departure;
        Some(row)
    }

    /// The aggregated costs of the journey behind `tau[round][stop]`,
    /// walking its label chain. The caller has checked the label is
    /// reached; `seconds` is left at zero for the caller to fill (the
    /// travel time depends on which departure the caller charges).
    fn walk_costs(
        &self,
        stop: StopIdx,
        best_round: usize,
        inputs: &CostInputs<'_>,
        access_meters: Option<&HashMap<StopIdx, f64>>,
    ) -> CostRow {
        let timetable = self.timetable;
        let mut round = best_round;
        let mut at = stop;
        let mut rides = 0u32;
        let mut transit_meters = 0.0;
        let mut walk_meters = 0.0;
        let mut grams = 0.0;
        let mut resolved = true;
        let mut legs: Vec<(TripIdx, u16, u16)> = Vec::new();
        loop {
            match self.labels[round][at.0 as usize] {
                Label::Transit {
                    trip,
                    board_position,
                    alight_position,
                    day_offset: _,
                } => {
                    rides += 1;
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
                        legs.push((trip, board_position, alight_position));
                    }
                    at = timetable.pattern_stops(timetable.trip_pattern(trip))
                        [board_position as usize];
                    round -= 1;
                }
                Label::Transfer {
                    from_stop,
                    duration: _,
                } => {
                    // Transfers are deduplicated per stop pair: the one
                    // edge found is the one routing relaxed.
                    walk_meters += self
                        .transfers
                        .from_stop(from_stop)
                        .iter()
                        .find(|transfer| transfer.to == at)
                        .map(|transfer| transfer.meters)
                        .unwrap_or(0.0);
                    at = from_stop;
                }
                Label::Access => {
                    if let Some(access) = access_meters {
                        walk_meters += access.get(&at).copied().unwrap_or(0.0);
                    }
                    break;
                }
                Label::Unreached => unreachable!("cost reconstruction hit an unreached label"),
            }
        }
        let geometry = match (inputs.with_geometry, inputs.leg_geometry) {
            (true, Some(shapes)) => {
                let parts: Vec<Vec<(f64, f64)>> = legs
                    .iter()
                    .rev()
                    .map(|&(trip, board, alight)| shapes.leg_coordinates(trip, board, alight))
                    .collect();
                Some(wkb_multi_line_string(&parts))
            }
            _ => None,
        };
        CostRow {
            to: stop.0,
            seconds: 0,
            rides,
            transit_meters,
            walk_meters,
            emission_grams: if resolved { grams } else { f64::NAN },
            geometry,
        }
    }

    /// Folds a pass's improved arrivals into each destination's cleanest
    /// candidate: the lowest-emission journey within the travel-time
    /// budget seen so far, ties resolved toward the shorter travel time.
    ///
    /// `thresholds` carries the best arrival per (destination, round)
    /// across the descending departures; an arrival that does not
    /// improve its threshold left `tau` unchanged, so it has no label
    /// chain to reconstruct — the candidates are exactly the profile's
    /// (departure, arrival, rides)-Pareto set, the same set
    /// `journey_frontier` sees. Unresolved emissions (NaN) never
    /// qualify.
    #[allow(clippy::too_many_arguments)]
    fn fold_cleanest(
        &self,
        departure: u32,
        destinations: &[StopIdx],
        budget: Option<u32>,
        inputs: &CostInputs<'_>,
        access_meters: Option<&HashMap<StopIdx, f64>>,
        thresholds: &mut [u32],
        cleanest: &mut [Option<CostRow>],
    ) {
        for (slot, &stop) in destinations.iter().enumerate() {
            let thresholds = &mut thresholds[slot * (self.rounds + 1)..][..self.rounds + 1];
            // At-most-k-ride thresholds — `collect`'s admission rule —
            // so the ride candidates match the profile's. Round 0 joins
            // as the zero-ride floor: the access-seeded stops (for a
            // stop matrix, the origin itself at 0 s and 0 g — footpaths
            // only relax after a ride, as in the time-optimising
            // matrix); for points the caller's direct-walk overlay
            // plays that part instead.
            for round in 0..=self.rounds {
                let arrival = self.tau[round][stop.0 as usize];
                if arrival >= thresholds[round] {
                    continue;
                }
                for threshold in &mut thresholds[round..] {
                    *threshold = (*threshold).min(arrival);
                }
                let seconds = arrival - departure;
                if budget.is_some_and(|budget| seconds > budget) {
                    continue;
                }
                let mut row = self.walk_costs(stop, round, inputs, access_meters);
                row.seconds = seconds;
                fold_cleaner(&mut cleanest[slot], row);
            }
        }
    }

    /// [`Search::fold_cleanest`] joined through each destination point's
    /// egress links: a point's per-round arrival is the minimum over its
    /// links of the stop arrival plus the egress walk.
    #[allow(clippy::too_many_arguments)]
    fn fold_cleanest_points(
        &self,
        departure: u32,
        egress: &[Vec<(StopIdx, u32, f64)>],
        budget: Option<u32>,
        inputs: &CostInputs<'_>,
        access_meters: &HashMap<StopIdx, f64>,
        thresholds: &mut [u32],
        cleanest: &mut [Option<CostRow>],
    ) {
        for (point, links) in egress.iter().enumerate() {
            let thresholds = &mut thresholds[point * (self.rounds + 1)..][..self.rounds + 1];
            // Rounds from 1, at-most-k-ride thresholds over the joined
            // point arrival — exactly `collect`'s admission rule (the
            // per-stop pruning inside `tau` does not survive the link
            // join, so the suffix update is what enforces dominance by
            // fewer-ride journeys here). The walking-only alternative
            // arrives via the caller's direct-walk overlay, as it does
            // for `journey_frontier`.
            for round in 1..=self.rounds {
                let mut chosen: Option<(u32, StopIdx, f64)> = None;
                for &(stop, seconds, meters) in links {
                    let at_stop = self.tau[round][stop.0 as usize];
                    if at_stop == UNREACHED {
                        continue;
                    }
                    let Some(arrival) = at_stop.checked_add(seconds).filter(|&at| at != UNREACHED)
                    else {
                        continue;
                    };
                    if chosen.is_none_or(|(current, _, _)| arrival < current) {
                        chosen = Some((arrival, stop, meters));
                    }
                }
                let Some((arrival, stop, egress_meters)) = chosen else {
                    continue;
                };
                if arrival >= thresholds[round] {
                    continue;
                }
                for threshold in &mut thresholds[round..] {
                    *threshold = (*threshold).min(arrival);
                }
                let seconds = arrival - departure;
                if budget.is_some_and(|budget| seconds > budget) {
                    continue;
                }
                let mut row = self.walk_costs(stop, round, inputs, Some(access_meters));
                row.to = point as u32;
                row.seconds = seconds;
                row.walk_meters += egress_meters;
                fold_cleaner(&mut cleanest[point], row);
            }
        }
    }

    /// Runs one pass per departure — which must be strictly decreasing —
    /// and returns the journeys no later departure dominates, sorted by
    /// departure and then by ride count.
    fn profile(mut self, departures: &[u32]) -> Vec<Journey> {
        let mut journeys = Vec::new();
        // The best arrival emitted with at most `round` rides so far; a
        // pass may only emit strictly earlier arrivals, so journeys
        // dominated by a later departure never surface.
        let mut thresholds = vec![UNREACHED; self.rounds + 1];
        for &departure in departures {
            self.run(departure);
            self.collect(departure, &mut thresholds, &mut journeys);
        }
        journeys.sort_by_key(|journey| (journey.departure, journey.rides()));
        journeys
    }

    /// One RAPTOR pass from `departure`, improving the shared state.
    fn run(&mut self, departure: u32) {
        let timetable = self.timetable;
        let request = self.request;

        // Leftover marks from the previous pass describe stops whose
        // labels are already final for later departures; they carry no
        // work for this one.
        for stop in std::mem::take(&mut self.marked) {
            self.is_marked[stop.0 as usize] = false;
        }

        for &(stop, duration) in &request.access {
            // Skip access whose arrival cannot be represented below the
            // UNREACHED sentinel; wrapping or saturating would corrupt it.
            let Some(arrival) = departure
                .checked_add(duration)
                .filter(|&at| at != UNREACHED)
            else {
                continue;
            };
            let index = stop.0 as usize;
            if arrival < self.tau[0][index] {
                self.tau[0][index] = arrival;
                self.labels[0][index] = Label::Access;
                for best in &mut self.best {
                    best[index] = best[index].min(arrival);
                }
                if !self.is_marked[index] {
                    self.is_marked[index] = true;
                    self.marked.push(stop);
                }
            }
        }

        for round in 1..=self.rounds {
            self.queued_patterns.clear();
            let mut marked = std::mem::take(&mut self.marked);
            for stop in marked.drain(..) {
                self.is_marked[stop.0 as usize] = false;
                for pattern_stop in timetable.patterns_at_stop(stop) {
                    let slot = &mut self.queue_position[pattern_stop.pattern.0 as usize];
                    if *slot == u16::MAX {
                        self.queued_patterns.push(pattern_stop.pattern);
                    }
                    if pattern_stop.position < *slot {
                        *slot = pattern_stop.position;
                    }
                }
            }
            self.marked = marked;

            for index in 0..self.queued_patterns.len() {
                let pattern = self.queued_patterns[index];
                let start_position = self.queue_position[pattern.0 as usize];
                self.queue_position[pattern.0 as usize] = u16::MAX;
                let stops = timetable.pattern_stops(pattern);
                let mut current: Option<(TripIdx, u16, u32)> = None;

                for position in start_position as usize..stops.len() {
                    let stop = stops[position].0 as usize;

                    if let Some((trip, board_position, day_offset)) = current {
                        let arrival = timetable.trip_stop_times(trip)[position]
                            .arrival
                            .saturating_sub(day_offset);
                        if arrival < self.best[round][stop] {
                            self.tau[round][stop] = arrival;
                            for best in &mut self.best[round..] {
                                best[stop] = best[stop].min(arrival);
                            }
                            self.labels[round][stop] = Label::Transit {
                                trip,
                                board_position,
                                alight_position: position as u16,
                                day_offset,
                            };
                            if !self.is_marked[stop] {
                                self.is_marked[stop] = true;
                                self.marked.push(stops[position]);
                            }
                        }
                    }

                    // Try to catch an earlier trip from this stop, using the
                    // previous round's arrival. The arrival handling above
                    // has already recorded any improvement at this position;
                    // boarding at a pattern's last position is pointless
                    // because there is no later stop to alight at, and other
                    // patterns serving this stop are queued separately.
                    let reached = self.tau[round - 1][stop];
                    if reached == UNREACHED || position + 1 == stops.len() {
                        continue;
                    }
                    let can_catch_earlier = match current {
                        Some((trip, _, day_offset)) => {
                            reached
                                <= timetable.trip_stop_times(trip)[position]
                                    .departure
                                    .saturating_sub(day_offset)
                        }
                        None => true,
                    };
                    if can_catch_earlier {
                        if let Some((trip, day_offset)) =
                            earliest_trip(timetable, request, pattern, position, reached)
                        {
                            // Board the earlier-departing vehicle; across
                            // service days trip index no longer orders
                            // departures, so compare the shifted times.
                            let departure = timetable.trip_stop_times(trip)[position]
                                .departure
                                .saturating_sub(day_offset);
                            let replaces = match current {
                                Some((current_trip, _, current_offset)) => {
                                    departure
                                        < timetable.trip_stop_times(current_trip)[position]
                                            .departure
                                            .saturating_sub(current_offset)
                                }
                                None => true,
                            };
                            if replaces {
                                current = Some((trip, position as u16, day_offset));
                            }
                        }
                    }
                }
            }

            // Relax one footpath hop from every stop improved by transit,
            // leaving from the transit arrivals as they stand now — a
            // transfer improving a marked stop must not chain into that
            // stop's own outgoing transfers within the round.
            let transit_marked: Vec<(StopIdx, u32)> = self
                .marked
                .iter()
                .map(|&stop| (stop, self.tau[round][stop.0 as usize]))
                .collect();
            for (stop, departure_at_stop) in transit_marked {
                for transfer in self.transfers.from_stop(stop) {
                    let Some(arrival) = departure_at_stop
                        .checked_add(transfer.duration)
                        .filter(|&at| at != UNREACHED)
                    else {
                        continue;
                    };
                    let to = transfer.to.0 as usize;
                    if arrival < self.best[round][to] {
                        self.tau[round][to] = arrival;
                        for best in &mut self.best[round..] {
                            best[to] = best[to].min(arrival);
                        }
                        self.labels[round][to] = Label::Transfer {
                            from_stop: stop,
                            duration: transfer.duration,
                        };
                        if !self.is_marked[to] {
                            self.is_marked[to] = true;
                            self.marked.push(transfer.to);
                        }
                    }
                }
            }

            if self.marked.is_empty() {
                break;
            }
        }
    }

    /// Emits the pass's journeys: per round, the best egress arrival, if
    /// it strictly beats everything already emitted with no more rides.
    fn collect(&self, departure: u32, thresholds: &mut [u32], journeys: &mut Vec<Journey>) {
        for round in 1..=self.rounds {
            let mut best_egress: Option<(u32, StopIdx, u32)> = None;
            for &(stop, duration) in &self.request.egress {
                let at_stop = self.tau[round][stop.0 as usize];
                if at_stop == UNREACHED {
                    continue;
                }
                let Some(arrival) = at_stop.checked_add(duration).filter(|&at| at != UNREACHED)
                else {
                    continue;
                };
                if best_egress.is_none_or(|(current, _, _)| arrival < current) {
                    best_egress = Some((arrival, stop, duration));
                }
            }
            let Some((arrival, egress_stop, egress_duration)) = best_egress else {
                continue;
            };
            if arrival >= thresholds[round] {
                continue;
            }
            for threshold in &mut thresholds[round..] {
                *threshold = (*threshold).min(arrival);
            }
            journeys.push(self.reconstruct(departure, round, egress_stop, egress_duration));
        }
    }

    /// Walks the labels backwards from the egress stop.
    ///
    /// Labels written by later-departure passes stay valid here: their
    /// board times are at or after the (only ever improved) arrival of the
    /// previous round, so the reconstructed chain is time-consistent.
    fn reconstruct(
        &self,
        departure: u32,
        round: usize,
        egress_stop: StopIdx,
        egress_duration: u32,
    ) -> Journey {
        let timetable = self.timetable;
        let mut legs = Vec::new();
        let departure_at_stop = self.tau[round][egress_stop.0 as usize];
        // Cannot saturate: collect only emits in-range egress arrivals.
        legs.push(Leg::Egress {
            from_stop: egress_stop,
            departure: departure_at_stop,
            arrival: departure_at_stop.saturating_add(egress_duration),
        });

        let mut current_round = round;
        let mut stop = egress_stop;
        loop {
            match self.labels[current_round][stop.0 as usize] {
                Label::Transit {
                    trip,
                    board_position,
                    alight_position,
                    day_offset,
                } => {
                    let pattern = timetable.trip_pattern(trip);
                    let pattern_stops = timetable.pattern_stops(pattern);
                    let times = timetable.trip_stop_times(trip);
                    let board_stop = pattern_stops[board_position as usize];
                    legs.push(Leg::Transit {
                        trip,
                        board_stop,
                        alight_stop: stop,
                        board_position,
                        alight_position,
                        board_time: times[board_position as usize]
                            .departure
                            .saturating_sub(day_offset),
                        alight_time: times[alight_position as usize]
                            .arrival
                            .saturating_sub(day_offset),
                    });
                    stop = board_stop;
                    current_round -= 1;
                }
                Label::Transfer {
                    from_stop,
                    duration,
                } => {
                    let arrival = self.tau[current_round][stop.0 as usize];
                    legs.push(Leg::Transfer {
                        from_stop,
                        to_stop: stop,
                        departure: arrival - duration,
                        arrival,
                    });
                    stop = from_stop;
                }
                Label::Access => {
                    legs.push(Leg::Access {
                        to_stop: stop,
                        departure,
                        arrival: self.tau[0][stop.0 as usize],
                    });
                    break;
                }
                Label::Unreached => unreachable!("journey reconstruction hit an unreached label"),
            }
        }
        legs.reverse();

        Journey {
            departure,
            arrival: departure_at_stop.saturating_add(egress_duration),
            legs,
        }
    }
}

/// The earliest trip of `pattern` boardable at `position` no earlier than
/// `reached`, and the day offset to subtract from its stored times. Today's
/// services board at their stored times; the previous day's board a day
/// earlier, so their over-midnight tail is reachable in the small hours.
/// The two are compared on the queried day's clock and the earlier one wins.
fn earliest_trip(
    timetable: &Timetable,
    request: &Request,
    pattern: PatternIdx,
    position: usize,
    reached: u32,
) -> Option<(TripIdx, u32)> {
    let today = earliest_active_trip(
        timetable,
        &request.active_services,
        pattern,
        position,
        reached,
    )
    .map(|trip| (trip, 0));
    // A previous-day trip stored at time `t` runs at `t - DAY_SECONDS`, so
    // it is boardable from `reached` when `t >= reached + DAY_SECONDS`.
    let previous = reached
        .checked_add(DAY_SECONDS)
        .and_then(|threshold| {
            earliest_active_trip(
                timetable,
                &request.active_services_previous,
                pattern,
                position,
                threshold,
            )
        })
        .map(|trip| (trip, DAY_SECONDS));
    match (today, previous) {
        (Some(today), Some(previous)) => {
            let departure = |(trip, offset): (TripIdx, u32)| {
                timetable.trip_stop_times(trip)[position]
                    .departure
                    .saturating_sub(offset)
            };
            Some(if departure(previous) < departure(today) {
                previous
            } else {
                today
            })
        }
        (today, None) => today,
        (None, previous) => previous,
    }
}

/// The earliest trip of `pattern` departing `position` at or after `reached`
/// whose service is `active`. Valid because departures at every position are
/// sorted within a FIFO pattern.
fn earliest_active_trip(
    timetable: &Timetable,
    active: &[bool],
    pattern: PatternIdx,
    position: usize,
    reached: u32,
) -> Option<TripIdx> {
    let range = timetable.pattern_trip_range(pattern);
    let (mut low, mut high) = (range.start, range.end);
    while low < high {
        let mid = low + (high - low) / 2;
        if timetable.trip_stop_times(TripIdx(mid))[position].departure < reached {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    (low..range.end).map(TripIdx).find(|&trip| {
        active
            .get(timetable.trip_service(trip) as usize)
            .copied()
            .unwrap_or(false)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::DistanceProvenance;
    use crate::timetable::{StopTime, TimetableBuilder};

    fn time(at: u32) -> StopTime {
        StopTime {
            arrival: at,
            departure: at,
        }
    }

    /// Five stops. Pattern A rides 0→1→2, pattern B rides 1→3, and stop 2
    /// has a 50-second footpath to stop 4, from which pattern C rides 4→3.
    fn network() -> (Timetable, Transfers) {
        let mut builder = TimetableBuilder::new(5);
        let a = builder
            .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 0)
            .unwrap();
        let b = builder.add_pattern(&[StopIdx(1), StopIdx(3)], 1).unwrap();
        let c = builder.add_pattern(&[StopIdx(4), StopIdx(3)], 2).unwrap();
        // Two trips on A so boarding must pick the second when the first
        // has already left.
        builder
            .add_trip(a, vec![time(100), time(200), time(300)], 0, 0)
            .unwrap();
        builder
            .add_trip(a, vec![time(700), time(800), time(900)], 1, 0)
            .unwrap();
        // B departs stop 1 at 250, reachable from A's first trip (arr 200).
        builder
            .add_trip(b, vec![time(250), time(400)], 2, 0)
            .unwrap();
        // A later B trip on an inactive service would be wrong to board.
        builder
            .add_trip(b, vec![time(500), time(600)], 3, 1)
            .unwrap();
        // C departs stop 4 at 400; stop 4 is only reachable by footpath
        // from stop 2 (arr 300 + 50).
        builder
            .add_trip(c, vec![time(400), time(1000)], 4, 0)
            .unwrap();
        let timetable = builder.finish();
        let transfers = Transfers::from_edges(5, &[(StopIdx(2), StopIdx(4), 50, 50.0)]).unwrap();
        (timetable, transfers)
    }

    fn request(from: StopIdx, to: StopIdx, departure: u32) -> Request {
        Request {
            departure,
            access: vec![(from, 0)],
            egress: vec![(to, 0)],
            active_services: vec![true, false],
            active_services_previous: Vec::new(),
            max_transfers: 3,
        }
    }

    #[test]
    fn times_overflowing_the_representable_range_are_unreachable() {
        // Access additions near the u32 limit must neither wrap nor
        // collide with the UNREACHED sentinel; such paths simply stay
        // unreachable instead of producing bogus arrivals.
        let (timetable, transfers) = network();
        for departure in [u32::MAX - 5, u32::MAX - 10] {
            let mut nearly_out_of_time = request(StopIdx(0), StopIdx(3), departure);
            nearly_out_of_time.access = vec![(StopIdx(0), 10)];
            assert_eq!(
                Raptor.route(&timetable, &transfers, &nearly_out_of_time),
                Vec::new()
            );
        }
    }

    #[test]
    fn window_percentiles_match_per_minute_runs() {
        // The windowed scan's samples must equal fresh single-departure
        // runs at every minute mark; percentiles follow nearest-rank.
        let (timetable, transfers) = network();
        let window = 600;
        let percentiles = [0.0, 50.0, 100.0];
        let mut request = request(StopIdx(0), StopIdx(3), 0);
        request.egress = Vec::new();
        let rows = Raptor.percentile_matrix(
            &timetable,
            &transfers,
            std::slice::from_ref(&request),
            window,
            &percentiles,
        );
        let stop_count = timetable.stop_count() as usize;
        for stop in 0..stop_count {
            let mut samples: Vec<u32> = (0..window / 60)
                .map(|step| {
                    let mark = step * 60;
                    let mut fresh = request.clone();
                    fresh.departure = mark;
                    match Raptor.one_to_all(&timetable, &transfers, &fresh)[stop] {
                        Some(arrival) => arrival - mark,
                        None => UNREACHED,
                    }
                })
                .collect();
            samples.sort_unstable();
            for (at, &percentile) in percentiles.iter().enumerate() {
                assert_eq!(
                    rows[0][stop * percentiles.len() + at],
                    nearest_rank(&samples, percentile),
                    "stop {stop} percentile {percentile}"
                );
            }
        }
        // The pointset variant joins the same samples over egress links.
        let egress = vec![vec![(StopIdx(3), 30, 25.0), (StopIdx(4), 10, 8.0)]];
        let point_rows = Raptor.percentile_matrix_to_points(
            &timetable,
            &transfers,
            std::slice::from_ref(&request),
            &egress,
            window,
            &percentiles,
        );
        let mut samples: Vec<u32> = (0..window / 60)
            .map(|step| {
                let mark = step * 60;
                let mut fresh = request.clone();
                fresh.departure = mark;
                let arrivals = Raptor.one_to_all(&timetable, &transfers, &fresh);
                let mut best = UNREACHED;
                for &(stop, seconds, _) in &egress[0] {
                    if let Some(at) = arrivals[stop.0 as usize] {
                        best = best.min(at + seconds);
                    }
                }
                if best == UNREACHED {
                    UNREACHED
                } else {
                    best - mark
                }
            })
            .collect();
        samples.sort_unstable();
        for (at, &percentile) in percentiles.iter().enumerate() {
            assert_eq!(point_rows[0][at], nearest_rank(&samples, percentile));
        }
    }

    #[test]
    fn cost_rows_aggregate_the_fastest_journey() {
        // Distances per trip: pattern A trips 1200 m over three stops,
        // B trips 800 m, C 2000 m; factors 10/10/20/20/30 g/pkm.
        let (timetable, transfers) = network();
        let geometry = TripGeometry::from_trips(
            &timetable,
            vec![
                (
                    TripIdx(0),
                    vec![0.0, 500.0, 1200.0],
                    DistanceProvenance::CrowFly,
                ),
                (
                    TripIdx(1),
                    vec![0.0, 500.0, 1200.0],
                    DistanceProvenance::CrowFly,
                ),
                (TripIdx(2), vec![0.0, 800.0], DistanceProvenance::CrowFly),
                (TripIdx(3), vec![0.0, 800.0], DistanceProvenance::CrowFly),
                (TripIdx(4), vec![0.0, 2000.0], DistanceProvenance::CrowFly),
            ],
        )
        .unwrap();
        let factors = [10.0, 10.0, 20.0, 20.0, 30.0];
        let inputs = CostInputs {
            geometry: &geometry,
            factors: &factors,
            leg_geometry: None,
            with_geometry: false,
        };
        let mut request = request(StopIdx(0), StopIdx(3), 0);
        request.egress = Vec::new();
        let rows = Raptor.cost_matrix(
            &timetable,
            &transfers,
            &inputs,
            std::slice::from_ref(&request),
            &[StopIdx(3), StopIdx(4)],
        );
        assert_eq!(rows.len(), 1);
        // To stop 3: ride A 0→1 (500 m, 10 g/pkm), ride B 1→3 (800 m,
        // 20 g/pkm), arriving 400 with no walking.
        let to_3 = &rows[0][0];
        assert_eq!((to_3.to, to_3.seconds, to_3.rides), (3, 400, 2));
        assert_eq!(to_3.transit_meters, 1300.0);
        assert_eq!(to_3.walk_meters, 0.0);
        assert!((to_3.emission_grams - 21.0).abs() < 1e-9);
        assert_eq!(to_3.geometry, None);
        // To stop 4: ride A 0→2 (1200 m), then the 50 m footpath.
        let to_4 = &rows[0][1];
        assert_eq!((to_4.to, to_4.seconds, to_4.rides), (4, 350, 1));
        assert_eq!(to_4.transit_meters, 1200.0);
        assert_eq!(to_4.walk_meters, 50.0);
        assert!((to_4.emission_grams - 12.0).abs() < 1e-9);
        // An unresolved factor (NaN) poisons only the affected row.
        let partial = [10.0, 10.0, f64::NAN, f64::NAN, 30.0];
        let inputs = CostInputs {
            factors: &partial,
            ..inputs
        };
        let rows = Raptor.cost_matrix(
            &timetable,
            &transfers,
            &inputs,
            std::slice::from_ref(&request),
            &[StopIdx(3), StopIdx(4)],
        );
        assert!(rows[0][0].emission_grams.is_nan());
        assert!((rows[0][1].emission_grams - 12.0).abs() < 1e-9);
    }

    #[test]
    fn point_rows_join_over_egress_links() {
        let (timetable, transfers) = network();
        let geometry = TripGeometry::from_trips(
            &timetable,
            vec![
                (
                    TripIdx(0),
                    vec![0.0, 500.0, 1200.0],
                    DistanceProvenance::CrowFly,
                ),
                (
                    TripIdx(1),
                    vec![0.0, 500.0, 1200.0],
                    DistanceProvenance::CrowFly,
                ),
                (TripIdx(2), vec![0.0, 800.0], DistanceProvenance::CrowFly),
                (TripIdx(3), vec![0.0, 800.0], DistanceProvenance::CrowFly),
                (TripIdx(4), vec![0.0, 2000.0], DistanceProvenance::CrowFly),
            ],
        )
        .unwrap();
        let factors = [10.0, 10.0, 20.0, 20.0, 30.0];
        let inputs = CostInputs {
            geometry: &geometry,
            factors: &factors,
            leg_geometry: None,
            with_geometry: false,
        };
        let mut request = request(StopIdx(0), StopIdx(3), 0);
        request.egress = Vec::new();
        let access: HashMap<StopIdx, f64> = [(StopIdx(0), 120.0)].into_iter().collect();
        // Point 0 leaves from stop 3; point 1 prefers stop 4's shorter
        // egress over stop 3's long one.
        let egress = vec![
            vec![(StopIdx(3), 30, 25.0)],
            vec![(StopIdx(3), 1000, 900.0), (StopIdx(4), 10, 8.0)],
        ];
        let rows = Raptor.cost_matrix_to_points(
            &timetable,
            &transfers,
            &inputs,
            std::slice::from_ref(&request),
            std::slice::from_ref(&access),
            &egress,
        );
        let point_0 = &rows[0][0];
        assert_eq!((point_0.to, point_0.seconds, point_0.rides), (0, 430, 2));
        assert_eq!(point_0.transit_meters, 1300.0);
        // Access 120 m plus egress 25 m; no transfer on this journey.
        assert_eq!(point_0.walk_meters, 145.0);
        assert!((point_0.emission_grams - 21.0).abs() < 1e-9);
        let point_1 = &rows[0][1];
        assert_eq!((point_1.to, point_1.seconds, point_1.rides), (1, 360, 1));
        assert_eq!(point_1.transit_meters, 1200.0);
        // Access 120 m, the 50 m footpath to stop 4, egress 8 m.
        assert_eq!(point_1.walk_meters, 178.0);
        assert!((point_1.emission_grams - 12.0).abs() < 1e-9);
    }

    #[test]
    fn many_origins_match_single_runs() {
        // The parallel fan-out must agree with per-request runs; enough
        // duplicated requests make the workers reuse pooled state.
        let (timetable, transfers) = network();
        let origins = [StopIdx(0), StopIdx(1), StopIdx(2), StopIdx(4)];
        let requests: Vec<Request> = (0..8)
            .flat_map(|_| origins)
            .map(|origin| {
                let mut request = request(origin, StopIdx(3), 0);
                request.egress = Vec::new();
                request
            })
            .collect();
        let rows = Raptor.one_to_all_many(&timetable, &transfers, &requests);
        assert_eq!(rows.len(), requests.len());
        for (request, row) in requests.iter().zip(&rows) {
            assert_eq!(row, &Raptor.one_to_all(&timetable, &transfers, request));
        }
    }

    #[test]
    fn routes_a_direct_ride() {
        let (timetable, transfers) = network();
        let journeys = Raptor.route(&timetable, &transfers, &request(StopIdx(0), StopIdx(2), 0));
        assert_eq!(journeys.len(), 1);
        let journey = &journeys[0];
        assert_eq!(journey.arrival, 300);
        assert_eq!(journey.rides(), 1);
        assert_eq!(
            journey.legs[1],
            Leg::Transit {
                trip: TripIdx(0),
                board_stop: StopIdx(0),
                alight_stop: StopIdx(2),
                board_position: 0,
                alight_position: 2,
                board_time: 100,
                alight_time: 300,
            }
        );
    }

    #[test]
    fn boards_the_next_trip_when_the_first_has_left() {
        let (timetable, transfers) = network();
        let journeys = Raptor.route(
            &timetable,
            &transfers,
            &request(StopIdx(0), StopIdx(2), 150),
        );
        assert_eq!(journeys.len(), 1);
        assert_eq!(journeys[0].arrival, 900);
    }

    #[test]
    fn transfers_at_a_shared_stop() {
        let (timetable, transfers) = network();
        let journeys = Raptor.route(&timetable, &transfers, &request(StopIdx(0), StopIdx(3), 0));
        // One ride cannot reach stop 3; two rides via stop 1 arrive at 400.
        assert_eq!(journeys.len(), 1);
        let journey = &journeys[0];
        assert_eq!(journey.arrival, 400);
        assert_eq!(journey.rides(), 2);
    }

    #[test]
    fn walks_a_footpath_after_a_ride() {
        let (timetable, transfers) = network();
        // Ride A to stop 2 (arr 300), walk the 50-second footpath to 4.
        let journeys = Raptor.route(&timetable, &transfers, &request(StopIdx(0), StopIdx(4), 0));
        assert_eq!(journeys.len(), 1);
        let journey = &journeys[0];
        assert_eq!(journey.arrival, 350);
        assert!(matches!(
            journey.legs[2],
            Leg::Transfer {
                from_stop: StopIdx(2),
                to_stop: StopIdx(4),
                departure: 300,
                arrival: 350,
            }
        ));
    }

    #[test]
    fn transfers_relax_a_single_hop_from_transit_arrivals() {
        // Footpaths 1→2 and 2→3 without a closing 1→3 edge: the walk out
        // of stop 2 must leave from its transit arrival (500), not chain
        // onto the walk that just improved stop 2 in the same round.
        let mut builder = TimetableBuilder::new(4);
        let to_a = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
        let to_b = builder.add_pattern(&[StopIdx(0), StopIdx(2)], 1).unwrap();
        builder
            .add_trip(to_a, vec![time(0), time(100)], 0, 0)
            .unwrap();
        builder
            .add_trip(to_b, vec![time(0), time(500)], 1, 0)
            .unwrap();
        let timetable = builder.finish();
        let transfers = Transfers::from_edges(
            4,
            &[
                (StopIdx(1), StopIdx(2), 30, 30.0),
                (StopIdx(2), StopIdx(3), 50, 50.0),
            ],
        )
        .unwrap();
        let journeys = Raptor.route(&timetable, &transfers, &request(StopIdx(0), StopIdx(3), 0));
        assert_eq!(journeys.len(), 1);
        assert_eq!(journeys[0].arrival, 550);
    }

    #[test]
    fn footpaths_from_the_origin_are_the_access_lists_job() {
        let (timetable, transfers) = network();
        // Stop 2 only rides A at its last position; the footpath 2→4 is
        // not relaxed from the origin itself, by contract.
        let journeys = Raptor.route(
            &timetable,
            &transfers,
            &request(StopIdx(2), StopIdx(3), 260),
        );
        assert_eq!(journeys.len(), 0);
    }

    #[test]
    fn skips_trips_of_inactive_services() {
        let (timetable, transfers) = network();
        // Departing at 260: B's active trip (dep 250) is gone; the service-1
        // trip at 500 exists but must not be boarded.
        let mut req = request(StopIdx(1), StopIdx(3), 260);
        let journeys = Raptor.route(&timetable, &transfers, &req);
        assert_eq!(journeys.len(), 0);
        // With service 1 active the 500 trip works.
        req.active_services = vec![true, true];
        let journeys = Raptor.route(&timetable, &transfers, &req);
        assert_eq!(journeys.len(), 1);
        assert_eq!(journeys[0].arrival, 600);
    }

    #[test]
    fn emits_the_pareto_set_over_rides_and_arrival() {
        let mut builder = TimetableBuilder::new(3);
        // A slow direct pattern and a faster two-ride alternative.
        let direct = builder.add_pattern(&[StopIdx(0), StopIdx(2)], 0).unwrap();
        let first = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 1).unwrap();
        let second = builder.add_pattern(&[StopIdx(1), StopIdx(2)], 2).unwrap();
        builder
            .add_trip(direct, vec![time(0), time(1000)], 0, 0)
            .unwrap();
        builder
            .add_trip(first, vec![time(0), time(100)], 1, 0)
            .unwrap();
        builder
            .add_trip(second, vec![time(150), time(300)], 2, 0)
            .unwrap();
        let timetable = builder.finish();
        let transfers = Transfers::empty(3);
        let journeys = Raptor.route(
            &timetable,
            &transfers,
            &Request {
                departure: 0,
                access: vec![(StopIdx(0), 0)],
                egress: vec![(StopIdx(2), 0)],
                active_services: vec![true],
                active_services_previous: Vec::new(),
                max_transfers: 3,
            },
        );
        assert_eq!(journeys.len(), 2);
        assert_eq!((journeys[0].rides(), journeys[0].arrival), (1, 1000));
        assert_eq!((journeys[1].rides(), journeys[1].arrival), (2, 300));
    }

    #[test]
    fn chooses_between_access_and_egress_alternatives() {
        let (timetable, transfers) = network();
        // Origin can reach stop 0 slowly or stop 1 quickly; destination is
        // reachable from stop 2 or stop 3.
        let journeys = Raptor.route(
            &timetable,
            &transfers,
            &Request {
                departure: 0,
                access: vec![(StopIdx(0), 90), (StopIdx(1), 10)],
                egress: vec![(StopIdx(2), 500), (StopIdx(3), 20)],
                active_services: vec![true, false],
                active_services_previous: Vec::new(),
                max_transfers: 3,
            },
        );
        // Best: board B at stop 1 (reached at 10, dep 250), arrive 3 at
        // 400, egress 20 → 420 with one ride. Riding A from 0 to 2 then
        // egress 500 gives 800; two rides cannot beat 420.
        assert_eq!(journeys.len(), 1);
        let journey = &journeys[0];
        assert_eq!(journey.arrival, 420);
        assert_eq!(journey.rides(), 1);
        assert_eq!(
            journey.legs[0],
            Leg::Access {
                to_stop: StopIdx(1),
                departure: 0,
                arrival: 10,
            }
        );
    }

    #[test]
    fn terminal_stops_still_board_their_other_patterns() {
        // Stop 1 is the terminus of pattern X (0→1) and the start of
        // pattern Y (1→2); arriving at the terminus must still allow
        // boarding Y through its own pattern membership.
        let mut builder = TimetableBuilder::new(3);
        let x = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
        let y = builder.add_pattern(&[StopIdx(1), StopIdx(2)], 1).unwrap();
        builder.add_trip(x, vec![time(0), time(100)], 0, 0).unwrap();
        builder
            .add_trip(y, vec![time(150), time(250)], 1, 0)
            .unwrap();
        let timetable = builder.finish();
        let transfers = Transfers::empty(3);
        let journeys = Raptor.route(
            &timetable,
            &transfers,
            &Request {
                departure: 0,
                access: vec![(StopIdx(0), 0)],
                egress: vec![(StopIdx(2), 0)],
                active_services: vec![true],
                active_services_previous: Vec::new(),
                max_transfers: 3,
            },
        );
        assert_eq!(journeys.len(), 1);
        assert_eq!(journeys[0].arrival, 250);
        assert_eq!(journeys[0].rides(), 2);
    }

    #[test]
    fn one_to_all_reports_earliest_arrivals_everywhere() {
        let (timetable, transfers) = network();
        let arrivals =
            Raptor.one_to_all(&timetable, &transfers, &request(StopIdx(0), StopIdx(0), 0));
        // Origin at the departure time; ride A to 1 and 2; B onward to 3;
        // the footpath reaches 4.
        assert_eq!(arrivals[0], Some(0));
        assert_eq!(arrivals[1], Some(200));
        assert_eq!(arrivals[2], Some(300));
        assert_eq!(arrivals[3], Some(400));
        assert_eq!(arrivals[4], Some(350));
        // Departing after the last useful trips, nothing is reachable
        // beyond the origin itself.
        let late = Raptor.one_to_all(&timetable, &transfers, &request(StopIdx(3), StopIdx(0), 0));
        assert_eq!(late[3], Some(0));
        assert_eq!(late[0], None);
    }

    #[test]
    fn respects_the_transfer_limit() {
        let (timetable, transfers) = network();
        let mut req = request(StopIdx(0), StopIdx(3), 0);
        req.max_transfers = 0;
        let journeys = Raptor.route(&timetable, &transfers, &req);
        assert_eq!(journeys.len(), 0);
    }

    /// One pattern 0→1 with three rides: 100→300, 200→400, 300→500.
    fn frequent_network() -> (Timetable, Transfers) {
        let mut builder = TimetableBuilder::new(2);
        let a = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
        for departure in [100, 200, 300] {
            builder
                .add_trip(a, vec![time(departure), time(departure + 200)], 0, 0)
                .unwrap();
        }
        (builder.finish(), Transfers::empty(2))
    }

    #[test]
    fn range_emits_one_journey_per_feasible_departure() {
        let (timetable, transfers) = frequent_network();
        let journeys = Raptor.route_range(
            &timetable,
            &transfers,
            &request(StopIdx(0), StopIdx(1), 0),
            250,
        );
        // Departures 100 and 200 fall in [0, 250); each ride is the
        // latest-departure way to its arrival, so both survive. The
        // window's final second waits for the 300 ride.
        let profile: Vec<_> = journeys
            .iter()
            .map(|journey| (journey.departure, journey.arrival, journey.rides()))
            .collect();
        assert_eq!(profile, vec![(100, 300, 1), (200, 400, 1), (249, 500, 1)]);
        // Each journey departs the origin at its stated departure time.
        for journey in &journeys {
            assert_eq!(
                journey.legs[0],
                Leg::Access {
                    to_stop: StopIdx(0),
                    departure: journey.departure,
                    arrival: journey.departure,
                }
            );
        }
    }

    #[test]
    fn range_window_is_half_open() {
        let (timetable, transfers) = frequent_network();
        let journeys = Raptor.route_range(
            &timetable,
            &transfers,
            &request(StopIdx(0), StopIdx(1), 100),
            100,
        );
        // [100, 200) holds the 100 departure; the ride at 200 is only
        // reached by waiting from the window's final second.
        let profile: Vec<_> = journeys
            .iter()
            .map(|journey| (journey.departure, journey.arrival))
            .collect();
        assert_eq!(profile, vec![(100, 300), (199, 400)]);
        // A zero-length window has no departures at all.
        let none = Raptor.route_range(
            &timetable,
            &transfers,
            &request(StopIdx(0), StopIdx(1), 100),
            0,
        );
        assert!(none.is_empty());
    }

    #[test]
    fn range_waits_past_the_window_when_the_next_ride_is_later() {
        let (timetable, transfers) = frequent_network();
        // No ride departs within [0, 50), but leaving at its final second
        // and waiting catches the ride at 100.
        let journeys = Raptor.route_range(
            &timetable,
            &transfers,
            &request(StopIdx(0), StopIdx(1), 0),
            50,
        );
        let profile: Vec<_> = journeys
            .iter()
            .map(|journey| (journey.departure, journey.arrival, journey.rides()))
            .collect();
        assert_eq!(profile, vec![(49, 300, 1)]);
    }

    #[test]
    fn range_keeps_fewer_ride_options_from_earlier_departures() {
        // Departing at 200, a two-ride chain arrives at 320; departing at
        // 100, the direct ride arrives at 400. Neither dominates the
        // other — the direct journey needs fewer rides — so the faster
        // later pass must not prune the earlier pass's direct label.
        let mut builder = TimetableBuilder::new(3);
        let direct = builder.add_pattern(&[StopIdx(0), StopIdx(2)], 0).unwrap();
        let first = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 1).unwrap();
        let second = builder.add_pattern(&[StopIdx(1), StopIdx(2)], 2).unwrap();
        builder
            .add_trip(direct, vec![time(100), time(400)], 0, 0)
            .unwrap();
        builder
            .add_trip(first, vec![time(200), time(240)], 1, 0)
            .unwrap();
        builder
            .add_trip(second, vec![time(250), time(320)], 2, 0)
            .unwrap();
        let timetable = builder.finish();
        let transfers = Transfers::empty(3);
        let journeys = Raptor.route_range(
            &timetable,
            &transfers,
            &request(StopIdx(0), StopIdx(2), 0),
            201,
        );
        let profile: Vec<_> = journeys
            .iter()
            .map(|journey| (journey.departure, journey.arrival, journey.rides()))
            .collect();
        assert_eq!(profile, vec![(100, 400, 1), (200, 320, 2)]);
    }

    #[test]
    fn range_drops_journeys_dominated_by_later_departures() {
        // A slow ride at 100 and an express at 150 that arrives earlier:
        // departing at 100 offers nothing the 150 departure does not beat.
        let mut builder = TimetableBuilder::new(2);
        let local = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
        let express = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 1).unwrap();
        builder
            .add_trip(local, vec![time(100), time(400)], 0, 0)
            .unwrap();
        builder
            .add_trip(express, vec![time(150), time(250)], 1, 0)
            .unwrap();
        let timetable = builder.finish();
        let transfers = Transfers::empty(2);
        let journeys = Raptor.route_range(
            &timetable,
            &transfers,
            &request(StopIdx(0), StopIdx(1), 0),
            200,
        );
        assert_eq!(journeys.len(), 1);
        assert_eq!((journeys[0].departure, journeys[0].arrival), (150, 250));
    }

    #[test]
    fn range_keeps_extra_rides_only_when_strictly_earlier() {
        // Departing at 200, one direct ride arrives at 500. Departing at
        // 100, a two-ride chain arrives at 300; the direct ride is also
        // catchable then but no longer beats anything.
        let mut builder = TimetableBuilder::new(3);
        let direct = builder.add_pattern(&[StopIdx(0), StopIdx(2)], 0).unwrap();
        let first = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 1).unwrap();
        let second = builder.add_pattern(&[StopIdx(1), StopIdx(2)], 2).unwrap();
        builder
            .add_trip(direct, vec![time(200), time(500)], 0, 0)
            .unwrap();
        builder
            .add_trip(first, vec![time(100), time(150)], 1, 0)
            .unwrap();
        builder
            .add_trip(second, vec![time(160), time(300)], 2, 0)
            .unwrap();
        let timetable = builder.finish();
        let transfers = Transfers::empty(3);
        let journeys = Raptor.route_range(
            &timetable,
            &transfers,
            &request(StopIdx(0), StopIdx(2), 0),
            300,
        );
        let profile: Vec<_> = journeys
            .iter()
            .map(|journey| (journey.departure, journey.arrival, journey.rides()))
            .collect();
        assert_eq!(profile, vec![(100, 300, 2), (200, 500, 1)]);
    }

    #[test]
    fn range_shifts_candidates_by_the_access_duration() {
        let (timetable, transfers) = frequent_network();
        let journeys = Raptor.route_range(
            &timetable,
            &transfers,
            &Request {
                departure: 0,
                access: vec![(StopIdx(0), 50)],
                egress: vec![(StopIdx(1), 0)],
                active_services: vec![true],
                active_services_previous: Vec::new(),
                max_transfers: 3,
            },
            200,
        );
        // Catching the rides at 100 and 200 means leaving at 50 and 150;
        // the window's final second waits for the ride at 300.
        let departures: Vec<_> = journeys.iter().map(|journey| journey.departure).collect();
        assert_eq!(departures, vec![50, 150, 199]);
        assert_eq!(
            journeys[0].legs[0],
            Leg::Access {
                to_stop: StopIdx(0),
                departure: 50,
                arrival: 100,
            }
        );
    }

    #[test]
    fn range_skips_candidates_of_inactive_services() {
        let (timetable, transfers) = network();
        // B's service-1 trip departs stop 1 at 500 but never runs.
        let journeys = Raptor.route_range(
            &timetable,
            &transfers,
            &request(StopIdx(1), StopIdx(3), 0),
            600,
        );
        let departures: Vec<_> = journeys.iter().map(|journey| journey.departure).collect();
        assert_eq!(departures, vec![250]);
    }

    #[test]
    fn range_walks_footpaths_per_departure() {
        let (timetable, transfers) = network();
        // Only A's first trip (dep 100) reaches stop 4 in time for C: ride
        // to stop 2 (arr 300), walk 50 s, catch C at 400.
        let journeys = Raptor.route_range(
            &timetable,
            &transfers,
            &request(StopIdx(0), StopIdx(3), 0),
            800,
        );
        let profile: Vec<_> = journeys
            .iter()
            .map(|journey| (journey.departure, journey.arrival, journey.rides()))
            .collect();
        assert_eq!(profile, vec![(100, 400, 2)]);
    }

    /// A trip stored past midnight on the previous service day is boardable
    /// early on the queried day, shifted back one day.
    fn over_midnight_network() -> (Timetable, Transfers) {
        let mut builder = TimetableBuilder::new(2);
        let pattern = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
        // 25:00 → 25:10 the previous day is 01:00 → 01:10 on this one.
        builder
            .add_trip(pattern, vec![time(90_000), time(90_600)], 0, 0)
            .unwrap();
        (builder.finish(), Transfers::empty(2))
    }

    #[test]
    fn boards_the_previous_days_over_midnight_trip() {
        let (timetable, transfers) = over_midnight_network();
        let base = Request {
            departure: 0,
            access: vec![(StopIdx(0), 0)],
            egress: vec![(StopIdx(1), 0)],
            active_services: vec![false],
            active_services_previous: vec![false],
            max_transfers: 1,
        };

        // Neither day runs the service: the night trip is unreachable.
        assert!(Raptor.route(&timetable, &transfers, &base).is_empty());

        // Today alone runs it at its stored 25:00 — reachable only by
        // waiting out the whole day, arriving 25:10.
        let today = Request {
            active_services: vec![true],
            ..base.clone()
        };
        let journeys = Raptor.route(&timetable, &transfers, &today);
        assert_eq!(journeys.len(), 1);
        assert_eq!(journeys[0].arrival, 90_600);

        // Active the day before, the same trip runs at 01:00 → 01:10 here.
        let previous = Request {
            active_services_previous: vec![true],
            ..base.clone()
        };
        let journeys = Raptor.route(&timetable, &transfers, &previous);
        assert_eq!(journeys.len(), 1);
        assert_eq!(journeys[0].arrival, 90_600 - 86_400);
        let Leg::Transit {
            board_time,
            alight_time,
            ..
        } = journeys[0].legs[1]
        else {
            panic!("expected a transit leg, got {:?}", journeys[0].legs);
        };
        assert_eq!(
            (board_time, alight_time),
            (90_000 - 86_400, 90_600 - 86_400)
        );

        // Both days active: the earlier previous-day run wins.
        let both = Request {
            active_services: vec![true],
            active_services_previous: vec![true],
            ..base.clone()
        };
        let journeys = Raptor.route(&timetable, &transfers, &both);
        assert_eq!(journeys.len(), 1);
        assert_eq!(journeys[0].arrival, 90_600 - 86_400);
    }

    #[test]
    fn range_profiles_previous_day_over_midnight_trips() {
        let (timetable, transfers) = over_midnight_network();
        let request = Request {
            departure: 0,
            access: vec![(StopIdx(0), 0)],
            egress: vec![(StopIdx(1), 0)],
            active_services: vec![false],
            active_services_previous: vec![true],
            max_transfers: 1,
        };
        // The window covers 00:00–02:00; the shifted 01:00 departure lands
        // in it and profiles as leaving at 01:00, arriving 01:10.
        let journeys = Raptor.route_range(&timetable, &transfers, &request, 2 * 3600);
        let profile: Vec<_> = journeys
            .iter()
            .map(|journey| (journey.departure, journey.arrival, journey.rides()))
            .collect();
        assert_eq!(profile, vec![(3_600, 4_200, 1)]);
    }
}
