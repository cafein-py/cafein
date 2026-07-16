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

use crate::fares::{FareLeg, FareTables};
use crate::geometry::{wkb_multi_line_string, LegGeometry, TripGeometry};
use crate::journey::{Journey, Leg};
use crate::path_key::{challenger_wins, PathToken};
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
    /// The journey's fare under the fare tables; NaN when the journey
    /// cannot be priced, or when no tables were given.
    pub fare: f64,
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
    /// Fare tables to price each row's journey with; `None` leaves
    /// fares NaN.
    pub fares: Option<&'a FareTables>,
}

const UNREACHED: u32 = u32::MAX;

/// What the windowed candidate fold minimises.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Objective {
    /// Grams CO₂e; unresolved (NaN) emissions never qualify.
    Emissions,
    /// The journey fare; unpriceable (NaN) journeys never qualify.
    Fare,
}

impl Objective {
    fn key(self, row: &CostRow) -> f64 {
        match self {
            Objective::Emissions => row.emission_grams,
            Objective::Fare => row.fare,
        }
    }
}

/// Keeps the better of an existing candidate and a challenger on the
/// objective: a lower key wins, equal keys resolve toward the shorter
/// travel time. NaN keys never qualify.
pub(crate) fn fold_better(
    current: &mut Option<CostRow>,
    challenger: CostRow,
    objective: Objective,
) {
    let key = objective.key(&challenger);
    if key.is_nan() {
        return;
    }
    let better = match current {
        None => true,
        Some(row) => {
            key < objective.key(row)
                || (key == objective.key(row) && challenger.seconds < row.seconds)
        }
    };
    if better {
        *current = Some(challenger);
    }
}

/// Appends the destination-to-origin canonical tokens of the chain
/// behind `labels[round][stop]` and returns its root departure. Reads
/// the same label chains `walk_costs` walks, in the time/topology
/// domain only.
fn chain_tokens_into(
    labels: &[Vec<Label>],
    timetable: &Timetable,
    stop: usize,
    round: usize,
    out: &mut Vec<PathToken>,
) -> u32 {
    let mut at = stop;
    let mut r = round;
    loop {
        match labels[r][at] {
            Label::Transit {
                trip,
                board_position,
                alight_position,
                day_offset,
            } => {
                out.push(PathToken::Ride {
                    trip: trip.0,
                    day_offset,
                    board: board_position,
                    alight: alight_position,
                });
                let pattern = timetable.trip_pattern(trip);
                at = timetable.pattern_stops(pattern)[board_position as usize].0 as usize;
                r -= 1;
            }
            Label::Transfer {
                from_stop,
                duration,
            } => {
                out.push(PathToken::Walk {
                    from: from_stop.0,
                    to: at as u32,
                    duration,
                });
                at = from_stop.0 as usize;
            }
            Label::Access {
                departure,
                duration,
            } => {
                out.push(PathToken::Access {
                    stop: at as u32,
                    duration,
                });
                return departure;
            }
            Label::Unreached => unreachable!("canonical key walked an unreached label"),
        }
    }
}

/// Seconds in a service day: a previous-day trip's stored times are shifted
/// back by this to place it on the queried day's clock.
const DAY_SECONDS: u32 = 86_400;

/// How a stop's arrival time in a round was achieved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Label {
    Unreached,
    /// Reached directly from the origin, leaving at `departure` and
    /// walking `duration` — the chain root the canonical path key
    /// compares.
    Access {
        departure: u32,
        duration: u32,
    },
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

/// RAPTOR state shared by the passes of one query.
pub(crate) struct Search<'a> {
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
    /// Reusable scratch for canonical-key comparisons on exact ties.
    key_scratch_a: Vec<PathToken>,
    key_scratch_b: Vec<PathToken>,
}

impl<'a> Search<'a> {
    pub(crate) fn new(
        timetable: &'a Timetable,
        transfers: &'a Transfers,
        request: &'a Request,
    ) -> Self {
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
            key_scratch_a: Vec::new(),
            key_scratch_b: Vec::new(),
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
        let mut fare_legs: Vec<FareLeg> = Vec::new();
        loop {
            match self.labels[round][at.0 as usize] {
                Label::Transit {
                    trip,
                    board_position,
                    alight_position,
                    day_offset,
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
                    let pattern = timetable.trip_pattern(trip);
                    let board_stop = timetable.pattern_stops(pattern)[board_position as usize];
                    if inputs.fares.is_some() {
                        fare_legs.push(FareLeg {
                            route: timetable.pattern_route(pattern),
                            board_stop: board_stop.0,
                            alight_stop: at.0,
                            board_time: timetable.trip_stop_times(trip)[board_position as usize]
                                .departure
                                .saturating_sub(day_offset),
                        });
                    }
                    at = board_stop;
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
                Label::Access { .. } => {
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
        let fare = match inputs.fares {
            Some(tables) => {
                // The chain walks back to front; pricing needs rides in
                // order.
                fare_legs.reverse();
                tables.price(&fare_legs)
            }
            None => f64::NAN,
        };
        CostRow {
            to: stop.0,
            seconds: 0,
            rides,
            transit_meters,
            walk_meters,
            emission_grams: if resolved { grams } else { f64::NAN },
            fare,
            geometry,
        }
    }

    /// Folds a pass's improved arrivals into each destination's best
    /// candidate on the objective: the lowest-key journey within the
    /// travel-time budget seen so far, ties resolved toward the
    /// shorter travel time.
    ///
    /// `thresholds` carries the best arrival per (destination, round)
    /// across the descending departures; an arrival that does not
    /// improve its threshold left `tau` unchanged, so it has no label
    /// chain to reconstruct — the candidates are exactly the profile's
    /// (departure, arrival, rides)-Pareto set, the same set
    /// `journey_frontier` sees. NaN keys (an unresolved emission
    /// factor, an unpriceable journey) never qualify.
    #[allow(clippy::too_many_arguments)]
    fn fold_best(
        &self,
        departure: u32,
        destinations: &[StopIdx],
        budget: Option<u32>,
        inputs: &CostInputs<'_>,
        access_meters: Option<&HashMap<StopIdx, f64>>,
        objective: Objective,
        thresholds: &mut [u32],
        best: &mut [Option<CostRow>],
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
                fold_better(&mut best[slot], row, objective);
            }
        }
    }

    /// [`Search::fold_best`] joined through each destination point's
    /// egress links: a point's per-round arrival is the minimum over its
    /// links of the stop arrival plus the egress walk.
    #[allow(clippy::too_many_arguments)]
    fn fold_best_points(
        &self,
        departure: u32,
        egress: &[Vec<(StopIdx, u32, f64)>],
        budget: Option<u32>,
        inputs: &CostInputs<'_>,
        access_meters: &HashMap<StopIdx, f64>,
        objective: Objective,
        thresholds: &mut [u32],
        best: &mut [Option<CostRow>],
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
                fold_better(&mut best[point], row, objective);
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

    /// Records an alighting from `trip` at `position`: a strict
    /// improvement writes through; an exact same-round tie keeps the
    /// canonical chain and re-marks the stop so the replacement
    /// propagates.
    #[allow(clippy::too_many_arguments)]
    fn alight_at(
        &mut self,
        timetable: &Timetable,
        round: usize,
        stops: &[StopIdx],
        position: usize,
        trip: TripIdx,
        board_position: u16,
        day_offset: u32,
    ) {
        let stop = stops[position].0 as usize;
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
        } else if arrival == self.tau[round][stop] && arrival != UNREACHED {
            let mut challenger = std::mem::take(&mut self.key_scratch_a);
            let mut incumbent = std::mem::take(&mut self.key_scratch_b);
            challenger.clear();
            challenger.push(PathToken::Ride {
                trip: trip.0,
                day_offset,
                board: board_position,
                alight: position as u16,
            });
            let root = chain_tokens_into(
                &self.labels,
                timetable,
                stops[board_position as usize].0 as usize,
                round - 1,
                &mut challenger,
            );
            incumbent.clear();
            let incumbent_root =
                chain_tokens_into(&self.labels, timetable, stop, round, &mut incumbent);
            if challenger_wins(root, &challenger, incumbent_root, &incumbent) {
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
            self.key_scratch_a = challenger;
            self.key_scratch_b = incumbent;
        }
    }

    /// One RAPTOR pass from `departure`, improving the shared state.
    pub(crate) fn run(&mut self, departure: u32) {
        let timetable = self.timetable;
        let request = self.request;
        let has_previous = request
            .active_services_previous
            .iter()
            .any(|&active| active);

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
                self.labels[0][index] = Label::Access {
                    departure,
                    duration,
                };
                for best in &mut self.best {
                    best[index] = best[index].min(arrival);
                }
                if !self.is_marked[index] {
                    self.is_marked[index] = true;
                    self.marked.push(stop);
                }
            } else if arrival == self.tau[0][index] && arrival != UNREACHED {
                let mut challenger = std::mem::take(&mut self.key_scratch_a);
                let mut incumbent = std::mem::take(&mut self.key_scratch_b);
                challenger.clear();
                challenger.push(PathToken::Access {
                    stop: stop.0,
                    duration,
                });
                incumbent.clear();
                let incumbent_root =
                    chain_tokens_into(&self.labels, timetable, index, 0, &mut incumbent);
                if challenger_wins(departure, &challenger, incumbent_root, &incumbent) {
                    self.labels[0][index] = Label::Access {
                        departure,
                        duration,
                    };
                    if !self.is_marked[index] {
                        self.is_marked[index] = true;
                        self.marked.push(stop);
                    }
                }
                self.key_scratch_a = challenger;
                self.key_scratch_b = incumbent;
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
                let mut currents: [Option<(TripIdx, u16)>; 2] = [None, None];

                for position in start_position as usize..stops.len() {
                    let stop = stops[position].0 as usize;

                    for (current, day_offset) in currents.into_iter().zip([0, DAY_SECONDS]) {
                        if let Some((trip, board_position)) = current {
                            self.alight_at(
                                timetable,
                                round,
                                stops,
                                position,
                                trip,
                                board_position,
                                day_offset,
                            );
                        }
                    }

                    // Try to catch an earlier trip from this stop, using the
                    // previous round's arrival. The arrival handling above
                    // has already recorded any improvement at this position;
                    // boarding at a pattern's last position is pointless
                    // because there is no later stop to alight at, and other
                    // patterns serving this stop are queued separately. The
                    // day streams board independently: merged across days
                    // departures are not FIFO — yesterday's tail can depart
                    // later here yet arrive earlier downstream — so each
                    // stream rides its own earliest trip and the arrival
                    // writes settle the competition.
                    let reached = self.tau[round - 1][stop];
                    if reached == UNREACHED || position + 1 == stops.len() {
                        continue;
                    }
                    for (stream, current) in currents.iter_mut().enumerate() {
                        let active: &[bool] = if stream == 0 {
                            &request.active_services
                        } else if has_previous {
                            &request.active_services_previous
                        } else {
                            continue;
                        };
                        // A previous-day trip stored at `t` runs at
                        // `t - DAY_SECONDS`, so it is boardable from
                        // `reached` when `t >= reached + DAY_SECONDS`.
                        let threshold = if stream == 0 {
                            reached
                        } else {
                            match reached.checked_add(DAY_SECONDS) {
                                Some(threshold) => threshold,
                                None => continue,
                            }
                        };
                        let can_catch_earlier = match *current {
                            Some((trip, _)) => {
                                threshold <= timetable.trip_stop_times(trip)[position].departure
                            }
                            None => true,
                        };
                        if !can_catch_earlier {
                            continue;
                        }
                        let Some(trip) =
                            earliest_active_trip(timetable, active, pattern, position, threshold)
                        else {
                            continue;
                        };
                        // Stored times order a single day's FIFO stream.
                        let replaces = match *current {
                            Some((current_trip, _)) => {
                                timetable.trip_stop_times(trip)[position].departure
                                    < timetable.trip_stop_times(current_trip)[position].departure
                            }
                            None => true,
                        };
                        if replaces {
                            *current = Some((trip, position as u16));
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
                    } else if arrival == self.tau[round][to] && arrival != UNREACHED {
                        let mut challenger = std::mem::take(&mut self.key_scratch_a);
                        let mut incumbent = std::mem::take(&mut self.key_scratch_b);
                        challenger.clear();
                        challenger.push(PathToken::Walk {
                            from: stop.0,
                            to: transfer.to.0,
                            duration: transfer.duration,
                        });
                        let root = chain_tokens_into(
                            &self.labels,
                            timetable,
                            stop.0 as usize,
                            round,
                            &mut challenger,
                        );
                        incumbent.clear();
                        let incumbent_root =
                            chain_tokens_into(&self.labels, timetable, to, round, &mut incumbent);
                        if challenger_wins(root, &challenger, incumbent_root, &incumbent) {
                            self.labels[round][to] = Label::Transfer {
                                from_stop: stop,
                                duration: transfer.duration,
                            };
                            if !self.is_marked[to] {
                                self.is_marked[to] = true;
                                self.marked.push(transfer.to);
                            }
                        }
                        self.key_scratch_a = challenger;
                        self.key_scratch_b = incumbent;
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
                Label::Access { .. } => {
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
#[path = "raptor_tests.rs"]
mod tests;
