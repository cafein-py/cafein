//! McTBTR: multicriteria trip-based routing over (arrival, emissions).
//!
//! This module holds the multicriteria transfer set — the precompute
//! stage. Witt's transfer reduction is unsound under a second
//! criterion (a covering transfer may ride a dirtier vehicle), so both
//! generation and reduction become dominance-aware:
//!
//! - Generation boards, per (line, position), the earliest catchable
//!   trip **plus** the strictly-decreasing-factor suffix of later
//!   trips — transferring to a later-but-cleaner vehicle can hold a
//!   true Pareto point. The same-line skip applies only to siblings
//!   whose factor is no cleaner; U-turn drops stay (the
//!   alight-one-stop-earlier alternative rides strictly less distance
//!   on the current trip and the identical distance on the boarded
//!   one). Trips without a resolved factor are never boarded and get
//!   no transfers — journeys riding them are excluded from emissions
//!   frontiers by contract.
//! - Reduction keeps a transfer iff some onward stop (directly or over
//!   a footpath) accepts its (arrival, grams) into a per-stop Pareto
//!   bag. Grams sit on the trip-absolute scale — the cumulative
//!   distance along the current trip times its factor — so every
//!   compared journey shares the "riding this trip" prefix and
//!   dominance is invariant to the actual boarding position, the same
//!   argument as McRAPTOR's same-trip rider reduction. Dominance is
//!   exact (no bucketing): the reduced set must stay complete for
//!   every query bucket.
//!
//! Every transfer the time reduction keeps improves an arrival, so it
//! also enters the bags: with every factor resolved, the multicriteria
//! set is a superset of the time set (unresolved-factor trips are the
//! deliberate exception — the time set boards them, this set never
//! does).

use rayon::prelude::*;

use crate::exhaustive::quantized;
use crate::geometry::TripGeometry;
use crate::journey::{Journey, Leg};
use crate::mcraptor::Bag;
use crate::raptor::departure_candidates;
use crate::router::Request;
use crate::tbtr::{
    earliest_boardable, u_turn, DayView, TransferSet, TransferSetBuild, TripTransfer, ViewTrip,
};
use crate::timetable::{StopIdx, Timetable};
use crate::transfers::Transfers;

/// Builds the multicriteria transfer set of a day view, fanned out
/// over virtual trips with rayon. `factors` are grams CO₂e per
/// passenger-kilometre per backing trip, NaN where unresolved.
pub fn transfer_set(
    view: &DayView,
    timetable: &Timetable,
    footpaths: &Transfers,
    geometry: &TripGeometry,
    factors: &[f64],
) -> TransferSetBuild {
    let per_trip: Vec<(Vec<Vec<TripTransfer>>, usize)> = (0..view.trip_count())
        .into_par_iter()
        .map_init(
            || Bags::new(timetable.stop_count()),
            |bags, trip| {
                let trip = ViewTrip(trip);
                let mut generated = generate(view, timetable, footpaths, factors, trip);
                let count = generated.iter().map(Vec::len).sum();
                reduce(
                    view,
                    timetable,
                    footpaths,
                    geometry,
                    factors,
                    trip,
                    bags,
                    &mut generated,
                );
                (generated, count)
            },
        )
        .collect();
    TransferSet::assemble(per_trip)
}

/// Grams CO₂e accumulated along `trip` from its first position to
/// `position` — the trip-absolute scale reduction compares on.
fn absolute_grams(
    geometry: &TripGeometry,
    view: &DayView,
    factors: &[f64],
    trip: ViewTrip,
    position: u16,
) -> f64 {
    let backing = view.backing(trip);
    geometry.leg_distance(backing, 0, position) as f64 / 1000.0 * factors[backing.0 as usize]
}

/// The feasible multicriteria transfers of one virtual trip, per alight
/// position.
fn generate(
    view: &DayView,
    timetable: &Timetable,
    footpaths: &Transfers,
    factors: &[f64],
    trip: ViewTrip,
) -> Vec<Vec<TripTransfer>> {
    let line = view.line_of(trip);
    let pattern = view.line_pattern(line);
    let offset = view.line_day_offset(line);
    let stops = timetable.pattern_stops(pattern);
    let times = view.stored_times(timetable, trip);
    let mut per_position: Vec<Vec<TripTransfer>> = vec![Vec::new(); stops.len()];
    let factor = factors[view.backing(trip).0 as usize];
    if !factor.is_finite() {
        return per_position;
    }
    let alight_from = view.first_boardable(trip) as usize + 1;
    for (alight, kept) in per_position.iter_mut().enumerate().skip(alight_from) {
        let arrival = times[alight].arrival - offset;
        let stop = stops[alight];
        let mut board_from = |at: StopIdx, ready: u32| {
            for served in timetable.patterns_at_stop(at) {
                for candidate_line in view.lines_of_pattern(served.pattern).into_iter().flatten() {
                    let candidate =
                        earliest_boardable(view, timetable, candidate_line, served.position, ready);
                    let Some(first) = candidate else { continue };
                    // The earliest catchable trip plus the later trips
                    // whose factor strictly improves on every earlier
                    // boardable one.
                    let mut cleanest = f64::INFINITY;
                    for rank in first.0..view.line_trips(candidate_line).end {
                        let boarded = ViewTrip(rank);
                        let boarded_factor = factors[view.backing(boarded).0 as usize];
                        if !boarded_factor.is_finite() || boarded_factor >= cleanest {
                            continue;
                        }
                        cleanest = boarded_factor;
                        if candidate_line == line
                            && boarded.0 >= trip.0
                            && served.position as usize >= alight
                            && boarded_factor >= factor
                        {
                            continue;
                        }
                        if u_turn(
                            view,
                            timetable,
                            stops,
                            times,
                            offset,
                            alight,
                            boarded,
                            served.position,
                        ) {
                            continue;
                        }
                        kept.push(TripTransfer {
                            trip: boarded,
                            position: served.position,
                        });
                    }
                }
            }
        };
        board_from(stop, arrival);
        for footpath in footpaths.from_stop(stop) {
            board_from(footpath.to, arrival.saturating_add(footpath.duration));
        }
    }
    per_position
}

/// The dominance-aware reduction for one virtual trip: back-to-front
/// over the alight positions, a transfer survives iff riding the
/// boarded trip onward lands a (arrival, grams) point no kept
/// alternative dominates, at some stop — directly or over a footpath.
#[allow(clippy::too_many_arguments)]
fn reduce(
    view: &DayView,
    timetable: &Timetable,
    footpaths: &Transfers,
    geometry: &TripGeometry,
    factors: &[f64],
    trip: ViewTrip,
    bags: &mut Bags,
    per_position: &mut [Vec<TripTransfer>],
) {
    bags.clear();
    let offset = view.line_day_offset(view.line_of(trip));
    let stops = timetable.pattern_stops(view.line_pattern(view.line_of(trip)));
    let times = view.stored_times(timetable, trip);
    let alight_from = view.first_boardable(trip) as usize + 1;
    for alight in (alight_from..stops.len()).rev() {
        let arrival = times[alight].arrival - offset;
        let staying = absolute_grams(geometry, view, factors, trip, alight as u16);
        bags.improve(stops[alight], arrival, staying);
        for footpath in footpaths.from_stop(stops[alight]) {
            bags.improve(
                footpath.to,
                arrival.saturating_add(footpath.duration),
                staying,
            );
        }
        let alighted = staying;
        per_position[alight].retain(|transfer| {
            let boarded = transfer.trip;
            let boarded_offset = view.day_offset(boarded);
            let boarded_factor = factors[view.backing(boarded).0 as usize];
            let boarded_stops = timetable.pattern_stops(view.line_pattern(view.line_of(boarded)));
            let boarded_times = view.stored_times(timetable, boarded);
            let boarded_from = absolute_grams(geometry, view, factors, boarded, transfer.position);
            let mut keeps = false;
            for k in transfer.position as usize + 1..boarded_stops.len() {
                let reached = boarded_times[k].arrival - boarded_offset;
                let ridden =
                    absolute_grams(geometry, view, factors, boarded, k as u16) - boarded_from;
                debug_assert!(boarded_factor.is_finite());
                let grams = alighted + ridden;
                if bags.improve(boarded_stops[k], reached, grams) {
                    keeps = true;
                }
                for footpath in footpaths.from_stop(boarded_stops[k]) {
                    if bags.improve(
                        footpath.to,
                        reached.saturating_add(footpath.duration),
                        grams,
                    ) {
                        keeps = true;
                    }
                }
            }
            keeps
        });
    }
}

/// Per-stop (arrival, grams) Pareto bags with cheap reuse: only the
/// touched stops reset between trips.
struct Bags {
    labels: Vec<Vec<(u32, f64)>>,
    touched: Vec<u32>,
}

impl Bags {
    fn new(stop_count: u32) -> Bags {
        Bags {
            labels: vec![Vec::new(); stop_count as usize],
            touched: Vec::new(),
        }
    }

    fn clear(&mut self) {
        for &stop in &self.touched {
            self.labels[stop as usize].clear();
        }
        self.touched.clear();
    }

    /// Pareto insert; returns whether the point entered the bag.
    fn improve(&mut self, stop: StopIdx, arrival: u32, grams: f64) -> bool {
        let bag = &mut self.labels[stop.0 as usize];
        for &(at, g) in bag.iter() {
            if at <= arrival && g <= grams {
                return false;
            }
        }
        if bag.is_empty() {
            self.touched.push(stop.0);
        }
        bag.retain(|&(at, g)| !(arrival <= at && grams <= g));
        bag.push((arrival, grams));
        true
    }
}

/// How a segment came to board its trip.
#[derive(Debug, Clone, Copy)]
enum SegOrigin {
    Access {
        stop: StopIdx,
        seconds: u32,
    },
    Transfer {
        parent: u32,
        alight: u16,
    },
    Walked {
        parent: u32,
        alight: u16,
        duration: u32,
    },
}

/// One boarded trip during the scan; `grams` are the journey's grams
/// at boarding.
#[derive(Debug, Clone, Copy)]
struct Segment {
    trip: ViewTrip,
    board: u16,
    grams: f64,
    departure: u32,
    origin: SegOrigin,
}

/// The per-(trip, round) Pareto bag over (board, κ): boarding no later
/// along the pattern with a κ in the same or a cleaner bucket covers
/// every alight the newcomer could make. Equal slots refine toward the
/// exact-cleaner κ.
#[derive(Debug, Clone, Default)]
struct TripBag {
    entries: Vec<(u16, f64, i64)>,
}

impl TripBag {
    fn admits(&mut self, board: u16, kappa: f64, key: i64) -> bool {
        for &(b, k, kk) in &self.entries {
            if b <= board && kk <= key && !(b == board && kk == key && kappa < k) {
                return false;
            }
        }
        self.entries
            .retain(|&(b, _, kk)| !(board <= b && key <= kk));
        self.entries.push((board, kappa, key));
        true
    }
}

/// A destination frontier entry: the leaf segment, where it alighted,
/// and how the egress was joined.
#[derive(Debug, Clone, Copy)]
struct Arrived {
    departure: u32,
    arrival: u32,
    key: i64,
    grams: f64,
    leaf: u32,
    alight: u16,
    /// A final footpath hop before the egress, when joined via one.
    walk: Option<(StopIdx, u32)>,
}

/// The multicriteria trip-based engine for one query date: the day
/// view, the dominance-aware transfer set, and per-trip factors.
/// Queries return the same journeys McRAPTOR's would — verified
/// against it and the exhaustive oracle — via segment scanning.
pub struct McTbtrEngine<'a> {
    timetable: &'a Timetable,
    footpaths: &'a Transfers,
    geometry: &'a TripGeometry,
    factors: &'a [f64],
    view: DayView,
    set: TransferSet,
}

impl<'a> McTbtrEngine<'a> {
    pub fn for_date(
        timetable: &'a Timetable,
        footpaths: &'a Transfers,
        geometry: &'a TripGeometry,
        factors: &'a [f64],
        active_services: &[bool],
        active_services_previous: &[bool],
    ) -> McTbtrEngine<'a> {
        let view = DayView::for_date(timetable, active_services, active_services_previous);
        // Same-stop transfers only — installed footpaths relax at
        // query time (the hybrid the time engine uses), so the dense
        // transitively closed set never enters the precompute.
        let none = Transfers::empty(timetable.stop_count());
        let set = transfer_set(&view, timetable, &none, geometry, factors).transfers;
        McTbtrEngine {
            timetable,
            footpaths,
            geometry,
            factors,
            view,
            set,
        }
    }

    /// The Pareto set over (arrival, emissions bucket) for a single
    /// departure, as full journeys.
    pub fn route(&self, request: &Request, bucket: f64) -> Vec<Journey> {
        self.profile(request, &[request.departure], bucket)
    }

    /// The departure-window profile over (departure, arrival,
    /// emissions bucket).
    pub fn route_range(&self, request: &Request, window: u32, bucket: f64) -> Vec<Journey> {
        let departures = departure_candidates(self.timetable, request, window);
        self.profile(request, &departures, bucket)
    }

    fn profile(&self, request: &Request, departures: &[u32], bucket: f64) -> Vec<Journey> {
        assert!(
            bucket.is_finite() && bucket > 0.0,
            "the emissions bucket must be positive"
        );
        let rounds = request.max_transfers as usize + 1;
        let key = |grams: f64| (grams / bucket).floor() as i64;
        let mut arena: Vec<Segment> = Vec::new();
        let mut queue: Vec<Vec<u32>> = vec![Vec::new(); rounds + 1];
        let mut trip_bags: Vec<TripBag> =
            vec![TripBag::default(); self.view.trip_count() as usize * rounds];
        let mut stop_bags: Vec<Bag> = vec![Bag::default(); self.timetable.stop_count() as usize];
        let mut destination: Vec<Arrived> = Vec::new();
        for &departure in departures {
            // Seed: board from every access stop.
            for &(stop, seconds) in &request.access {
                let ready = departure.saturating_add(seconds);
                let admitted = stop_bags[stop.0 as usize].insert(ready, 0.0, key(0.0));
                if !admitted {
                    continue;
                }
                self.board(
                    stop,
                    ready,
                    0.0,
                    departure,
                    |_, _| SegOrigin::Access { stop, seconds },
                    0,
                    key,
                    &mut arena,
                    &mut trip_bags,
                    &mut queue[1],
                );
            }
            for round in 1..=rounds {
                let segments = std::mem::take(&mut queue[round]);
                for index in segments {
                    self.scan(
                        index,
                        round,
                        rounds,
                        request,
                        key,
                        &mut arena,
                        &mut trip_bags,
                        &mut stop_bags,
                        &mut destination,
                        &mut queue,
                    );
                }
            }
        }
        let mut journeys: Vec<Journey> = destination
            .iter()
            .map(|arrived| self.assemble(arrived, &arena))
            .collect();
        journeys.sort_by_key(|journey| (journey.departure, journey.arrival, journey.rides()));
        journeys
    }

    /// Boards the earliest catchable trip of every line serving `stop`,
    /// plus the strictly-decreasing-factor suffix, admitting through
    /// the (board, κ) bags of `round + 1`... the caller passes the
    /// target round's queue.
    #[allow(clippy::too_many_arguments)]
    fn board(
        &self,
        stop: StopIdx,
        ready: u32,
        grams: f64,
        departure: u32,
        origin: impl Fn(ViewTrip, u16) -> SegOrigin,
        round: usize,
        key: impl Fn(f64) -> i64,
        arena: &mut Vec<Segment>,
        trip_bags: &mut [TripBag],
        queue: &mut Vec<u32>,
    ) {
        let rounds_stride = trip_bags.len() / self.view.trip_count().max(1) as usize;
        for served in self.timetable.patterns_at_stop(stop) {
            for line in self
                .view
                .lines_of_pattern(served.pattern)
                .into_iter()
                .flatten()
            {
                let Some(first) =
                    earliest_boardable(&self.view, self.timetable, line, served.position, ready)
                else {
                    continue;
                };
                let mut cleanest = f64::INFINITY;
                for rank in first.0..self.view.line_trips(line).end {
                    let trip = ViewTrip(rank);
                    let factor = self.factors[self.view.backing(trip).0 as usize];
                    if !factor.is_finite() || factor >= cleanest {
                        continue;
                    }
                    cleanest = factor;
                    let ridden = absolute_grams(
                        self.geometry,
                        &self.view,
                        self.factors,
                        trip,
                        served.position,
                    );
                    let kappa = quantized(grams - ridden);
                    let slot = rank as usize * rounds_stride + round;
                    if trip_bags[slot].admits(served.position, kappa, key(kappa)) {
                        let index = arena.len() as u32;
                        arena.push(Segment {
                            trip,
                            board: served.position,
                            grams,
                            departure,
                            origin: origin(trip, served.position),
                        });
                        queue.push(index);
                    }
                }
            }
        }
    }

    /// Scans one segment: alight everywhere ahead, join the egress
    /// (directly and over one footpath), relax query-time footpaths
    /// into next-round boardings, and expand the precomputed transfers.
    #[allow(clippy::too_many_arguments)]
    fn scan(
        &self,
        index: u32,
        round: usize,
        rounds: usize,
        request: &Request,
        key: impl Fn(f64) -> i64 + Copy,
        arena: &mut Vec<Segment>,
        trip_bags: &mut [TripBag],
        stop_bags: &mut [Bag],
        destination: &mut Vec<Arrived>,
        queue: &mut [Vec<u32>],
    ) {
        let segment = arena[index as usize];
        let trip = segment.trip;
        let line = self.view.line_of(trip);
        let offset = self.view.line_day_offset(line);
        let stops = self.timetable.pattern_stops(self.view.line_pattern(line));
        let times = self.view.stored_times(self.timetable, trip);
        let boarded_from =
            absolute_grams(self.geometry, &self.view, self.factors, trip, segment.board);
        for alight in segment.board as usize + 1..stops.len() {
            let arrival = times[alight].arrival - offset;
            let ridden =
                absolute_grams(self.geometry, &self.view, self.factors, trip, alight as u16)
                    - boarded_from;
            let grams = quantized(segment.grams + ridden);
            let stop = stops[alight];
            // Direct egress joins.
            for &(egress, seconds) in &request.egress {
                if egress == stop {
                    self.join(
                        destination,
                        key,
                        segment.departure,
                        arrival.saturating_add(seconds),
                        grams,
                        index,
                        alight as u16,
                        None,
                    );
                }
            }
            // Query-time footpaths, gated on stop-bag improvement (the
            // T4b semantics); improving walks board and join.
            if stop_bags[stop.0 as usize].insert(arrival, grams, key(grams)) {
                for footpath in self.footpaths.from_stop(stop) {
                    let reached = arrival.saturating_add(footpath.duration);
                    if !stop_bags[footpath.to.0 as usize].insert(reached, grams, key(grams)) {
                        continue;
                    }
                    for &(egress, seconds) in &request.egress {
                        if egress == footpath.to {
                            self.join(
                                destination,
                                key,
                                segment.departure,
                                reached.saturating_add(seconds),
                                grams,
                                index,
                                alight as u16,
                                Some((footpath.to, footpath.duration)),
                            );
                        }
                    }
                    if round < rounds {
                        let duration = footpath.duration;
                        self.board(
                            footpath.to,
                            reached,
                            grams,
                            segment.departure,
                            |_, _| SegOrigin::Walked {
                                parent: index,
                                alight: alight as u16,
                                duration,
                            },
                            round,
                            key,
                            arena,
                            trip_bags,
                            &mut queue[round + 1],
                        );
                    }
                }
            }
            // Precomputed transfers, pruned against the destination.
            if round < rounds && !self.pruned(destination, key, arrival, grams) {
                for transfer in self.set.from_trip_position(trip, alight as u16) {
                    let boarded = transfer.trip;
                    let ridden = absolute_grams(
                        self.geometry,
                        &self.view,
                        self.factors,
                        boarded,
                        transfer.position,
                    );
                    let kappa = quantized(grams - ridden);
                    let stride = rounds;
                    let slot = boarded.0 as usize * stride + round;
                    if trip_bags[slot].admits(transfer.position, kappa, key(kappa)) {
                        let next = arena.len() as u32;
                        arena.push(Segment {
                            trip: boarded,
                            board: transfer.position,
                            grams,
                            departure: segment.departure,
                            origin: SegOrigin::Transfer {
                                parent: index,
                                alight: alight as u16,
                            },
                        });
                        queue[round + 1].push(next);
                    }
                }
            }
        }
    }

    /// Whether every continuation from (arrival, grams) is already
    /// dominated at the destination — passes descend, so every stored
    /// entry departs no earlier than the current pass.
    fn pruned(
        &self,
        destination: &[Arrived],
        key: impl Fn(f64) -> i64,
        arrival: u32,
        grams: f64,
    ) -> bool {
        let key = key(grams);
        destination
            .iter()
            .any(|entry| entry.arrival <= arrival && entry.key <= key)
    }

    /// Inserts a destination candidate under (departure desc, arrival,
    /// bucket) dominance with equal-slot refinement — McRAPTOR's
    /// destination-bag rules.
    #[allow(clippy::too_many_arguments)]
    fn join(
        &self,
        destination: &mut Vec<Arrived>,
        key: impl Fn(f64) -> i64,
        departure: u32,
        arrival: u32,
        grams: f64,
        leaf: u32,
        alight: u16,
        walk: Option<(StopIdx, u32)>,
    ) {
        let key = key(grams);
        for entry in destination.iter() {
            if entry.departure >= departure
                && entry.arrival <= arrival
                && entry.key <= key
                && !(entry.departure == departure
                    && entry.arrival == arrival
                    && entry.key == key
                    && grams < entry.grams)
            {
                return;
            }
        }
        destination.retain(|entry| {
            !(departure >= entry.departure && arrival <= entry.arrival && key <= entry.key)
        });
        destination.push(Arrived {
            departure,
            arrival,
            key,
            grams,
            leaf,
            alight,
            walk,
        });
    }

    /// Walks a winning segment chain back into the journey contract.
    fn assemble(&self, arrived: &Arrived, arena: &[Segment]) -> Journey {
        let mut legs = Vec::new();
        let mut segment = &arena[arrived.leaf as usize];
        let mut alight = arrived.alight;
        let leaf_stops = self
            .timetable
            .pattern_stops(self.view.line_pattern(self.view.line_of(segment.trip)));
        let leaf_times = self.view.stored_times(self.timetable, segment.trip);
        let leaf_offset = self.view.day_offset(segment.trip);
        let alight_arrival = leaf_times[alight as usize].arrival - leaf_offset;
        match arrived.walk {
            Some((stop, duration)) => {
                let reached = alight_arrival.saturating_add(duration);
                legs.push(Leg::Egress {
                    from_stop: stop,
                    departure: reached,
                    arrival: arrived.arrival,
                });
                legs.push(Leg::Transfer {
                    from_stop: leaf_stops[alight as usize],
                    to_stop: stop,
                    departure: alight_arrival,
                    arrival: reached,
                });
            }
            None => {
                legs.push(Leg::Egress {
                    from_stop: leaf_stops[alight as usize],
                    departure: alight_arrival,
                    arrival: arrived.arrival,
                });
            }
        }
        loop {
            let trip = segment.trip;
            let line = self.view.line_of(trip);
            let offset = self.view.line_day_offset(line);
            let stops = self.timetable.pattern_stops(self.view.line_pattern(line));
            let times = self.view.stored_times(self.timetable, trip);
            legs.push(Leg::Transit {
                trip: self.view.backing(trip),
                board_stop: stops[segment.board as usize],
                alight_stop: stops[alight as usize],
                board_position: segment.board,
                alight_position: alight,
                board_time: times[segment.board as usize].departure - offset,
                alight_time: times[alight as usize].arrival - offset,
            });
            let board_stop = stops[segment.board as usize];
            match segment.origin {
                SegOrigin::Access { stop, seconds } => {
                    legs.push(Leg::Access {
                        to_stop: stop,
                        departure: segment.departure,
                        arrival: segment.departure.saturating_add(seconds),
                    });
                    break;
                }
                SegOrigin::Transfer { parent, alight: at }
                | SegOrigin::Walked {
                    parent, alight: at, ..
                } => {
                    let parent_segment = &arena[parent as usize];
                    let parent_line = self.view.line_of(parent_segment.trip);
                    let parent_stops = self
                        .timetable
                        .pattern_stops(self.view.line_pattern(parent_line));
                    let parent_stop = parent_stops[at as usize];
                    if parent_stop != board_stop {
                        let left = self.view.stored_times(self.timetable, parent_segment.trip)
                            [at as usize]
                            .arrival
                            - self.view.line_day_offset(parent_line);
                        let duration = match segment.origin {
                            SegOrigin::Walked { duration, .. } => duration,
                            _ => self
                                .footpaths
                                .from_stop(parent_stop)
                                .iter()
                                .find(|footpath| footpath.to == board_stop)
                                .map(|footpath| footpath.duration)
                                .unwrap_or(0),
                        };
                        legs.push(Leg::Transfer {
                            from_stop: parent_stop,
                            to_stop: board_stop,
                            departure: left,
                            arrival: left.saturating_add(duration),
                        });
                    }
                    alight = at;
                    segment = parent_segment;
                }
            }
        }
        legs.reverse();
        Journey {
            departure: arrived.departure,
            arrival: arrived.arrival,
            legs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::DistanceProvenance;
    use crate::timetable::{StopTime, TimetableBuilder, TripIdx};

    fn time(at: u32) -> StopTime {
        StopTime {
            arrival: at,
            departure: at,
        }
    }

    /// Line A rides 0→1→2; a fast dirty line and a slow clean line both
    /// ride 1→3.
    fn forked() -> (Timetable, TripGeometry) {
        let mut builder = TimetableBuilder::new(4);
        let a = builder
            .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 0)
            .unwrap();
        let fast = builder.add_pattern(&[StopIdx(1), StopIdx(3)], 1).unwrap();
        let slow = builder.add_pattern(&[StopIdx(1), StopIdx(3)], 2).unwrap();
        builder
            .add_trip(a, vec![time(0), time(100), time(300)], 0, 0)
            .unwrap();
        builder
            .add_trip(fast, vec![time(120), time(400)], 1, 0)
            .unwrap();
        builder
            .add_trip(slow, vec![time(150), time(600)], 2, 0)
            .unwrap();
        let timetable = builder.finish();
        let geometry = TripGeometry::from_trips(
            &timetable,
            vec![
                (
                    TripIdx(0),
                    vec![0.0, 500.0, 1000.0],
                    DistanceProvenance::CrowFly,
                ),
                (TripIdx(1), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
                (TripIdx(2), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
            ],
        )
        .unwrap();
        (timetable, geometry)
    }

    fn boarded(set: &TransferSet, trip: ViewTrip, position: u16) -> Vec<(u32, u16)> {
        let mut list: Vec<(u32, u16)> = set
            .from_trip_position(trip, position)
            .iter()
            .map(|transfer| (transfer.trip.0, transfer.position))
            .collect();
        list.sort_unstable();
        list
    }

    #[test]
    fn keeps_the_cleaner_slower_transfer() {
        let (timetable, geometry) = forked();
        let view = DayView::universal(&timetable);
        let footpaths = Transfers::empty(4);
        // The time reduction sees the slow line arriving later
        // everywhere and drops it.
        let time_set = TransferSet::for_view(&view, &timetable, &footpaths).transfers;
        assert_eq!(boarded(&time_set, ViewTrip(0), 1), vec![(1, 0)]);
        // Clean-but-slow (factor 10 vs 100): a true Pareto move, kept.
        let factors = [50.0, 100.0, 10.0];
        let mc = transfer_set(&view, &timetable, &footpaths, &geometry, &factors).transfers;
        assert_eq!(boarded(&mc, ViewTrip(0), 1), vec![(1, 0), (2, 0)]);
        // With the slow line just as dirty, the mc reduction drops it
        // too — and matches the time set exactly here.
        let uniform = [50.0, 100.0, 100.0];
        let same = transfer_set(&view, &timetable, &footpaths, &geometry, &uniform).transfers;
        assert_eq!(boarded(&same, ViewTrip(0), 1), vec![(1, 0)]);
    }

    #[test]
    fn the_mc_set_covers_the_time_set() {
        let (timetable, geometry) = forked();
        let view = DayView::universal(&timetable);
        let footpaths = Transfers::empty(4);
        let time_set = TransferSet::for_view(&view, &timetable, &footpaths).transfers;
        for factors in [[50.0, 100.0, 10.0], [50.0, 50.0, 50.0], [1.0, 2.0, 3.0]] {
            let mc = transfer_set(&view, &timetable, &footpaths, &geometry, &factors).transfers;
            for trip in 0..view.trip_count() {
                let stops = timetable
                    .pattern_stops(view.line_pattern(view.line_of(ViewTrip(trip))))
                    .len();
                for position in 0..stops as u16 {
                    for kept in time_set.from_trip_position(ViewTrip(trip), position) {
                        assert!(
                            mc.from_trip_position(ViewTrip(trip), position)
                                .contains(kept),
                            "time-kept transfer missing under factors {factors:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn mixed_factor_lines_board_the_cleaner_later_trip() {
        // One connecting line with a dirty early trip and a clean later
        // one: generation must board both.
        let mut builder = TimetableBuilder::new(4);
        let a = builder
            .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 0)
            .unwrap();
        let b = builder.add_pattern(&[StopIdx(1), StopIdx(3)], 1).unwrap();
        builder
            .add_trip(a, vec![time(0), time(100), time(300)], 0, 0)
            .unwrap();
        builder
            .add_trip(b, vec![time(120), time(400)], 1, 0)
            .unwrap();
        builder
            .add_trip(b, vec![time(200), time(500)], 2, 0)
            .unwrap();
        let timetable = builder.finish();
        let geometry = TripGeometry::from_trips(
            &timetable,
            vec![
                (
                    TripIdx(0),
                    vec![0.0, 500.0, 1000.0],
                    DistanceProvenance::CrowFly,
                ),
                (TripIdx(1), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
                (TripIdx(2), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
            ],
        )
        .unwrap();
        let view = DayView::universal(&timetable);
        let footpaths = Transfers::empty(4);
        let mixed = [50.0, 100.0, 10.0];
        let mc = transfer_set(&view, &timetable, &footpaths, &geometry, &mixed).transfers;
        assert_eq!(boarded(&mc, ViewTrip(0), 1), vec![(1, 0), (2, 0)]);
        // Uniform factors collapse to the earliest-trip rule.
        let uniform = [50.0, 100.0, 100.0];
        let mc = transfer_set(&view, &timetable, &footpaths, &geometry, &uniform).transfers;
        assert_eq!(boarded(&mc, ViewTrip(0), 1), vec![(1, 0)]);
    }

    #[test]
    fn same_line_reboarding_survives_only_toward_cleaner_siblings() {
        // One line, dirty trip then clean trip: alighting the dirty
        // trip mid-pattern may re-board the cleaner sibling at the same
        // position; with uniform factors the sibling stays skipped.
        let mut builder = TimetableBuilder::new(3);
        let a = builder
            .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 0)
            .unwrap();
        builder
            .add_trip(a, vec![time(0), time(100), time(300)], 0, 0)
            .unwrap();
        builder
            .add_trip(a, vec![time(50), time(200), time(400)], 1, 0)
            .unwrap();
        let timetable = builder.finish();
        let geometry = TripGeometry::from_trips(
            &timetable,
            vec![
                (
                    TripIdx(0),
                    vec![0.0, 500.0, 1000.0],
                    DistanceProvenance::CrowFly,
                ),
                (
                    TripIdx(1),
                    vec![0.0, 500.0, 1000.0],
                    DistanceProvenance::CrowFly,
                ),
            ],
        )
        .unwrap();
        let view = DayView::universal(&timetable);
        let footpaths = Transfers::empty(3);
        let mixed = [100.0, 10.0];
        let mc = transfer_set(&view, &timetable, &footpaths, &geometry, &mixed).transfers;
        assert_eq!(boarded(&mc, ViewTrip(0), 1), vec![(1, 1)]);
        let uniform = [100.0, 100.0];
        let mc = transfer_set(&view, &timetable, &footpaths, &geometry, &uniform).transfers;
        assert_eq!(boarded(&mc, ViewTrip(0), 1), vec![]);
    }

    #[test]
    fn unresolved_factors_are_excluded() {
        let (timetable, geometry) = forked();
        let view = DayView::universal(&timetable);
        let footpaths = Transfers::empty(4);
        // The fast line's factor is unresolved: never boarded.
        let factors = [50.0, f64::NAN, 10.0];
        let mc = transfer_set(&view, &timetable, &footpaths, &geometry, &factors).transfers;
        assert_eq!(boarded(&mc, ViewTrip(0), 1), vec![(2, 0)]);
        // An unresolved source trip gets no transfers at all.
        let factors = [f64::NAN, 100.0, 10.0];
        let mc = transfer_set(&view, &timetable, &footpaths, &geometry, &factors).transfers;
        assert_eq!(boarded(&mc, ViewTrip(0), 1), vec![]);
    }

    use crate::exhaustive::pareto_oracle;
    use crate::mcraptor;

    fn grams_of(journey: &Journey, geometry: &TripGeometry, factors: &[f64]) -> f64 {
        quantized(
            journey
                .legs
                .iter()
                .map(|leg| match leg {
                    Leg::Transit {
                        trip,
                        board_position,
                        alight_position,
                        ..
                    } => {
                        geometry.leg_distance(*trip, *board_position, *alight_position) as f64
                            / 1000.0
                            * factors[trip.0 as usize]
                    }
                    _ => 0.0,
                })
                .sum(),
        )
    }

    fn triples(
        journeys: &[Journey],
        geometry: &TripGeometry,
        factors: &[f64],
    ) -> Vec<(u32, f64, u32)> {
        let mut list: Vec<(u32, f64, u32)> = journeys
            .iter()
            .map(|journey| {
                (
                    journey.arrival,
                    grams_of(journey, geometry, factors),
                    journey.rides() as u32,
                )
            })
            .collect();
        list.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.partial_cmp(&b.1).unwrap()));
        list
    }

    /// The three-way check: with a vanishing bucket, the trip-based
    /// engine, McRAPTOR, and the exhaustive oracle must produce the
    /// same frontier.
    fn engines_agree(
        timetable: &Timetable,
        geometry: &TripGeometry,
        footpaths: &Transfers,
        factors: &[f64],
        origin: StopIdx,
        egress: StopIdx,
        max_transfers: u8,
    ) -> Vec<(u32, f64, u32)> {
        let view = DayView::universal(timetable);
        let request = Request {
            departure: 0,
            access: vec![(origin, 0)],
            egress: vec![(egress, 0)],
            active_services: vec![true; 8],
            active_services_previous: vec![false; 8],
            max_transfers,
        };
        let points = pareto_oracle(
            &view,
            timetable,
            footpaths,
            geometry,
            factors,
            0,
            &request.access,
            &request.egress,
            max_transfers,
        );
        let oracle: Vec<(u32, f64, u32)> = points
            .iter()
            .map(|point| (point.arrival, point.grams, point.rides))
            .collect();
        let raptor = mcraptor::route(
            &view, timetable, footpaths, geometry, factors, &request, 1e-6,
        );
        assert_eq!(triples(&raptor, geometry, factors), oracle, "mcraptor");
        let engine = McTbtrEngine::for_date(
            timetable,
            footpaths,
            geometry,
            factors,
            &request.active_services,
            &request.active_services_previous,
        );
        let tbtr = engine.route(&request, 1e-6);
        assert_eq!(triples(&tbtr, geometry, factors), oracle, "mctbtr");
        oracle
    }

    #[test]
    fn the_engine_matches_the_oracle_on_the_forked_fixture() {
        let (timetable, geometry) = forked();
        let footpaths = Transfers::empty(4);
        let frontier = engines_agree(
            &timetable,
            &geometry,
            &footpaths,
            &[50.0, 100.0, 10.0],
            StopIdx(0),
            StopIdx(3),
            3,
        );
        // Fast-dirty and slow-clean two-ride journeys: both true points.
        assert_eq!(frontier, vec![(400, 125.0, 2), (600, 35.0, 2)]);
        engines_agree(
            &timetable,
            &geometry,
            &footpaths,
            &[50.0, 100.0, 100.0],
            StopIdx(0),
            StopIdx(3),
            3,
        );
    }

    #[test]
    fn the_engine_matches_the_oracle_over_footpaths() {
        // Ride, walk a footpath, ride again — the walked transfer is a
        // query-time relaxation, not a precomputed one.
        let mut builder = TimetableBuilder::new(4);
        let first = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
        let second = builder.add_pattern(&[StopIdx(2), StopIdx(3)], 1).unwrap();
        builder
            .add_trip(first, vec![time(0), time(100)], 0, 0)
            .unwrap();
        builder
            .add_trip(second, vec![time(200), time(300)], 1, 0)
            .unwrap();
        let timetable = builder.finish();
        let geometry = TripGeometry::from_trips(
            &timetable,
            vec![
                (TripIdx(0), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
                (TripIdx(1), vec![0.0, 2000.0], DistanceProvenance::CrowFly),
            ],
        )
        .unwrap();
        let footpaths = Transfers::from_edges(
            4,
            &[
                (StopIdx(1), StopIdx(2), 50, 50.0),
                (StopIdx(2), StopIdx(1), 50, 50.0),
            ],
        )
        .unwrap();
        let frontier = engines_agree(
            &timetable,
            &geometry,
            &footpaths,
            &[10.0, 20.0],
            StopIdx(0),
            StopIdx(3),
            3,
        );
        assert_eq!(frontier, vec![(300, 50.0, 2)]);
    }

    #[test]
    fn loop_backs_walk_nowhere() {
        // The routers' shared regression shape: ride out and back, then
        // walk — suppressed by the access-seeded stop bags.
        let mut builder = TimetableBuilder::new(3);
        let out = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
        let back = builder.add_pattern(&[StopIdx(1), StopIdx(0)], 1).unwrap();
        let direct = builder.add_pattern(&[StopIdx(0), StopIdx(2)], 2).unwrap();
        builder
            .add_trip(out, vec![time(10), time(50)], 0, 0)
            .unwrap();
        builder
            .add_trip(back, vec![time(60), time(100)], 1, 0)
            .unwrap();
        builder
            .add_trip(direct, vec![time(20), time(200)], 2, 0)
            .unwrap();
        let timetable = builder.finish();
        let geometry = TripGeometry::from_trips(
            &timetable,
            vec![
                (TripIdx(0), vec![0.0, 400.0], DistanceProvenance::CrowFly),
                (TripIdx(1), vec![0.0, 400.0], DistanceProvenance::CrowFly),
                (TripIdx(2), vec![0.0, 800.0], DistanceProvenance::CrowFly),
            ],
        )
        .unwrap();
        let footpaths = Transfers::from_edges(
            3,
            &[
                (StopIdx(0), StopIdx(2), 30, 30.0),
                (StopIdx(2), StopIdx(0), 30, 30.0),
            ],
        )
        .unwrap();
        let frontier = engines_agree(
            &timetable,
            &geometry,
            &footpaths,
            &[10.0, 10.0, 100.0],
            StopIdx(0),
            StopIdx(2),
            3,
        );
        assert_eq!(frontier, vec![(200, 80.0, 1)]);
    }

    #[test]
    fn profiles_the_departure_window() {
        let mut builder = TimetableBuilder::new(2);
        let line = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
        builder
            .add_trip(line, vec![time(100), time(300)], 0, 0)
            .unwrap();
        builder
            .add_trip(line, vec![time(200), time(400)], 1, 0)
            .unwrap();
        let timetable = builder.finish();
        let geometry = TripGeometry::from_trips(
            &timetable,
            vec![
                (TripIdx(0), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
                (TripIdx(1), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
            ],
        )
        .unwrap();
        let footpaths = Transfers::empty(2);
        let factors = [50.0, 50.0];
        let request = Request {
            departure: 50,
            access: vec![(StopIdx(0), 0)],
            egress: vec![(StopIdx(1), 0)],
            active_services: vec![true],
            active_services_previous: vec![false],
            max_transfers: 1,
        };
        let engine = McTbtrEngine::for_date(
            &timetable,
            &footpaths,
            &geometry,
            &factors,
            &request.active_services,
            &request.active_services_previous,
        );
        let journeys = engine.route_range(&request, 200, 1e-6);
        let profile: Vec<(u32, u32)> = journeys
            .iter()
            .map(|journey| (journey.departure, journey.arrival))
            .collect();
        assert_eq!(profile, vec![(100, 300), (200, 400)]);
    }

    #[test]
    fn u_turns_stay_dropped() {
        // Alight A at stop 2, walk back to stop 1, and re-ride the
        // 1→2 segment on B although B was catchable by alighting A at
        // stop 1 directly: the classic U-turn, dropped even though B
        // is cleaner — the earlier alight saves grams on A and rides
        // the identical distance on B. Boarding B at stop 2 itself
        // stays a plain, kept transfer.
        let mut builder = TimetableBuilder::new(4);
        let a = builder
            .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 0)
            .unwrap();
        let b = builder
            .add_pattern(&[StopIdx(1), StopIdx(2), StopIdx(3)], 1)
            .unwrap();
        builder
            .add_trip(a, vec![time(0), time(100), time(200)], 0, 0)
            .unwrap();
        builder
            .add_trip(b, vec![time(250), time(350), time(450)], 1, 0)
            .unwrap();
        let timetable = builder.finish();
        let geometry = TripGeometry::from_trips(
            &timetable,
            vec![
                (
                    TripIdx(0),
                    vec![0.0, 500.0, 1000.0],
                    DistanceProvenance::CrowFly,
                ),
                (
                    TripIdx(1),
                    vec![0.0, 500.0, 1000.0],
                    DistanceProvenance::CrowFly,
                ),
            ],
        )
        .unwrap();
        let footpaths = Transfers::from_edges(
            4,
            &[
                (StopIdx(1), StopIdx(2), 30, 30.0),
                (StopIdx(2), StopIdx(1), 30, 30.0),
            ],
        )
        .unwrap();
        let view = DayView::universal(&timetable);
        let factors = [100.0, 10.0];
        let mc = transfer_set(&view, &timetable, &footpaths, &geometry, &factors).transfers;
        // From (A, position 2): the walk-back boarding of B at
        // position 0 is the U-turn and is gone; boarding B at stop 2
        // (position 1) survives.
        assert_eq!(boarded(&mc, ViewTrip(0), 2), vec![(1, 1)]);
    }
}
