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
use crate::fares::FareLeg;
use crate::geometry::{wkb_multi_line_string, TripGeometry};
use crate::journey::{Journey, Leg};
use crate::mcraptor::Bag;
use crate::raptor::{departure_candidates, CostInputs, CostRow};
use crate::router::Request;
use crate::tbtr::{
    earliest_boardable, u_turn, DayView, TransferSet, TransferSetBuild, TripTransfer, ViewTrip,
};
use crate::timetable::{StopIdx, Timetable, TripIdx};
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

/// The sentinel leaf of a zero-ride (access floor) matrix winner.
const ACCESS_LEAF: u32 = u32::MAX;

/// One matrix winner: the cleanest (then fastest) point folded for a
/// destination, with the chain to rebuild its cost row.
#[derive(Debug, Clone, Copy)]
struct Winner {
    grams: f64,
    seconds: u32,
    leaf: u32,
    alight: u16,
    /// The point was reached over a final footpath hop.
    walked: bool,
}

/// Per-destination fold state for the emissions matrix, mirroring the
/// McRAPTOR matrix fold: candidates fold per pass at creation (an
/// end-of-search bag readout would lose budget-qualifying candidates
/// to cross-pass evictions), lower grams win, ties resolve toward the
/// shorter travel time, a travel-time budget disqualifies outright.
struct MatrixSink<'a> {
    /// Per stop: destination slot + 1, or 0 when not a destination.
    slots: &'a [u32],
    budget: Option<u32>,
    best: &'a mut [Option<Winner>],
}

impl MatrixSink<'_> {
    fn fold(
        &mut self,
        stop: StopIdx,
        seconds: u32,
        grams: f64,
        leaf: u32,
        alight: u16,
        walked: bool,
    ) {
        let slot = self.slots[stop.0 as usize];
        if slot == 0 {
            return;
        }
        if self.budget.is_some_and(|budget| seconds > budget) {
            return;
        }
        let best = &mut self.best[slot as usize - 1];
        let better = match best {
            None => true,
            Some(winner) => {
                grams < winner.grams || (grams == winner.grams && seconds < winner.seconds)
            }
        };
        if better {
            *best = Some(Winner {
                grams,
                seconds,
                leaf,
                alight,
                walked,
            });
        }
    }
}

/// Per-destination-slot frontier state for the batched product: every
/// egress join the one-pair search would make feeds its slot's
/// destination frontier instead, under the same `join` rules, so a
/// batched cell's journeys equal the single-pair query's. Stop mode
/// joins a destination stop's own alights and footpath walks with no
/// final walk; door-to-door mode (`egress_active`) walks each join
/// through the per-stop final-egress map, the walking-only journey
/// being the caller's overlay.
struct FrontierSink<'a> {
    /// Per stop: destination slot + 1, or 0 when not a destination.
    slots: &'a [u32],
    /// Per stop: `(destination slot, walk seconds, walk meters)` final
    /// egress; consulted only in door-to-door mode.
    egress: &'a [Vec<(u32, u32, f64)>],
    egress_active: bool,
    /// Per slot: the destination frontier the cell assembles from.
    bags: &'a mut [Vec<Arrived>],
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
    set: std::borrow::Cow<'a, TransferSet>,
}

impl<'a> McTbtrEngine<'a> {
    /// The date's multicriteria transfer set alone — what a caller
    /// caches to skip the expensive precompute on later engines
    /// (`from_set`). Keyed by the date's view and the factors; the
    /// query-time footpaths never enter the precompute, so one set
    /// serves every footpath choice.
    pub fn transfers_for_date(
        timetable: &Timetable,
        geometry: &TripGeometry,
        factors: &[f64],
        active_services: &[bool],
        active_services_previous: &[bool],
    ) -> TransferSet {
        let view = DayView::for_date(timetable, active_services, active_services_previous);
        // Same-stop transfers only — installed footpaths relax at
        // query time (the hybrid the time engine uses), so the dense
        // transitively closed set never enters the precompute.
        let none = Transfers::empty(timetable.stop_count());
        transfer_set(&view, timetable, &none, geometry, factors).transfers
    }

    pub fn for_date(
        timetable: &'a Timetable,
        footpaths: &'a Transfers,
        geometry: &'a TripGeometry,
        factors: &'a [f64],
        active_services: &[bool],
        active_services_previous: &[bool],
    ) -> McTbtrEngine<'a> {
        let set = Self::transfers_for_date(
            timetable,
            geometry,
            factors,
            active_services,
            active_services_previous,
        );
        let view = DayView::for_date(timetable, active_services, active_services_previous);
        McTbtrEngine {
            timetable,
            footpaths,
            geometry,
            factors,
            view,
            set: std::borrow::Cow::Owned(set),
        }
    }

    /// The engine over a **prebuilt** multicriteria transfer set — the
    /// reused path when the caller cached the date's set
    /// (`transfers_for_date`), skipping the dominance-aware precompute.
    /// The set must have been built for these `active_services` and
    /// these `factors`; only the cheap per-engine state (the day view)
    /// is rebuilt.
    pub fn from_set(
        timetable: &'a Timetable,
        footpaths: &'a Transfers,
        geometry: &'a TripGeometry,
        factors: &'a [f64],
        active_services: &[bool],
        active_services_previous: &[bool],
        set: &'a TransferSet,
    ) -> McTbtrEngine<'a> {
        let view = DayView::for_date(timetable, active_services, active_services_previous);
        McTbtrEngine {
            timetable,
            footpaths,
            geometry,
            factors,
            view,
            set: std::borrow::Cow::Borrowed(set),
        }
    }

    /// The Pareto set over (arrival, emissions bucket) for a single
    /// departure, as full journeys.
    pub fn route(&self, request: &Request, bucket: f64) -> Vec<Journey> {
        self.profile(request, &[request.departure], bucket, &mut None)
    }

    /// The departure-window profile over (departure, arrival,
    /// emissions bucket).
    pub fn route_range(&self, request: &Request, window: u32, bucket: f64) -> Vec<Journey> {
        let departures = departure_candidates(self.timetable, request, window);
        self.profile(request, &departures, bucket, &mut None)
    }

    fn profile(
        &self,
        request: &Request,
        departures: &[u32],
        bucket: f64,
        fold: &mut Option<MatrixSink<'_>>,
    ) -> Vec<Journey> {
        let (arena, destination) = self.passes(request, departures, bucket, fold, &mut None);
        let mut journeys: Vec<Journey> = destination
            .iter()
            .map(|arrived| self.assemble(arrived, &arena))
            .collect();
        journeys.sort_by_key(|journey| (journey.departure, journey.arrival, journey.rides()));
        journeys
    }

    /// The pass loop shared by the journey profile and the matrix:
    /// returns the segment arena and the destination frontier.
    fn passes(
        &self,
        request: &Request,
        departures: &[u32],
        bucket: f64,
        fold: &mut Option<MatrixSink<'_>>,
        frontier: &mut Option<FrontierSink<'_>>,
    ) -> (Vec<Segment>, Vec<Arrived>) {
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
                // rides = 0: the trip-based engine ranks rounds in its
                // trip bags, so its stop bags dominate on (arrival, key)
                // only — see mcraptor::Bag::insert.
                let admitted = stop_bags[stop.0 as usize].insert(ready, 0.0, key(0.0), 0);
                if !admitted {
                    continue;
                }
                if let Some(sink) = fold {
                    // The zero-ride floor of the origin's own cell.
                    sink.fold(
                        stop,
                        ready.saturating_sub(departure),
                        0.0,
                        ACCESS_LEAF,
                        0,
                        false,
                    );
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
                        fold,
                        frontier,
                    );
                }
            }
        }
        (arena, destination)
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
        fold: &mut Option<MatrixSink<'_>>,
        frontier: &mut Option<FrontierSink<'_>>,
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
            // Alights, direct egress joins, and query-time footpaths are
            // all gated on stop-bag improvement (the T4b semantics) —
            // the same admission McRAPTOR's egress check applies, so a
            // dominated arrival (the origin's own access seed included)
            // never joins a destination. Rank the rides used (this
            // segment's round) so a later-departure pass that reached
            // the stop on more rides cannot suppress a cleaner
            // fewer-rides arrival across the profile — the same
            // cross-pass soundness the McRAPTOR bag needs.
            if stop_bags[stop.0 as usize].insert(arrival, grams, key(grams), round as u8) {
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
                if let Some(sink) = frontier {
                    self.frontier_join(
                        sink,
                        stop,
                        key,
                        segment.departure,
                        arrival,
                        grams,
                        index,
                        alight as u16,
                        None,
                    );
                }
                if let Some(sink) = fold {
                    sink.fold(
                        stop,
                        arrival - segment.departure,
                        grams,
                        index,
                        alight as u16,
                        false,
                    );
                }
                for footpath in self.footpaths.from_stop(stop) {
                    let reached = arrival.saturating_add(footpath.duration);
                    if !stop_bags[footpath.to.0 as usize].insert(
                        reached,
                        grams,
                        key(grams),
                        round as u8,
                    ) {
                        continue;
                    }
                    if let Some(sink) = fold {
                        sink.fold(
                            footpath.to,
                            reached - segment.departure,
                            grams,
                            index,
                            alight as u16,
                            true,
                        );
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
                    if let Some(sink) = frontier {
                        self.frontier_join(
                            sink,
                            footpath.to,
                            key,
                            segment.departure,
                            reached,
                            grams,
                            index,
                            alight as u16,
                            Some((footpath.to, footpath.duration)),
                        );
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

    /// Routes an egress join into every destination slot the stop
    /// serves, under the one-pair `join` rules: the slot's own stop in
    /// stop mode (no final walk), the per-stop final-egress map in
    /// door-to-door mode (`arrival` is before the final walk).
    #[allow(clippy::too_many_arguments)]
    fn frontier_join(
        &self,
        sink: &mut FrontierSink<'_>,
        stop: StopIdx,
        key: impl Fn(f64) -> i64 + Copy,
        departure: u32,
        arrival: u32,
        grams: f64,
        leaf: u32,
        alight: u16,
        walk: Option<(StopIdx, u32)>,
    ) {
        if !sink.egress_active {
            let slot = sink.slots[stop.0 as usize];
            if slot != 0 {
                self.join(
                    &mut sink.bags[slot as usize - 1],
                    key,
                    departure,
                    arrival,
                    grams,
                    leaf,
                    alight,
                    walk,
                );
            }
            return;
        }
        for index in 0..sink.egress[stop.0 as usize].len() {
            let (slot, seconds, _) = sink.egress[stop.0 as usize][index];
            self.join(
                &mut sink.bags[slot as usize],
                key,
                departure,
                arrival.saturating_add(seconds),
                grams,
                leaf,
                alight,
                walk,
            );
        }
    }

    /// The batched Pareto frontiers over this engine's candidate set:
    /// per request × destination slot, the (departure, arrival,
    /// emissions bucket) Pareto journeys of the departure window —
    /// each cell exactly the one-pair `route_range` set, the McTBTR
    /// counterpart of `mcraptor::frontier_matrix`. Stop mode takes the
    /// destination stops; door-to-door mode (`egress_active`) a
    /// per-stop final-egress map over `slot_count` destination points,
    /// the walking-only journey being the caller's overlay. One engine
    /// build (one transfer set) serves every origin, fanned out with
    /// rayon.
    #[allow(clippy::too_many_arguments)]
    pub fn frontier_matrix(
        &self,
        requests: &[Request],
        destinations: &[StopIdx],
        egress: &[Vec<(u32, u32, f64)>],
        egress_active: bool,
        slot_count: usize,
        window: u32,
        bucket: f64,
    ) -> Vec<Vec<Vec<Journey>>> {
        if egress_active {
            assert_eq!(
                egress.len(),
                self.timetable.stop_count() as usize,
                "the egress map must be per stop"
            );
        } else {
            assert_eq!(
                destinations.len(),
                slot_count,
                "stop mode takes one slot per destination stop"
            );
        }
        // A stop holds one slot, so repeated destination stops share the
        // first occurrence's slot and their cells are re-expanded to the
        // requested order after assembly.
        let mut slots = vec![0u32; self.timetable.stop_count() as usize];
        let mut cell_of: Vec<usize> = Vec::with_capacity(destinations.len());
        let mut unique = 0usize;
        for &stop in destinations {
            let slot = slots[stop.0 as usize];
            if slot == 0 {
                unique += 1;
                slots[stop.0 as usize] = unique as u32;
                cell_of.push(unique - 1);
            } else {
                cell_of.push(slot as usize - 1);
            }
        }
        let bag_count = if egress_active { slot_count } else { unique };
        requests
            .par_iter()
            .map(|request| {
                let departures = departure_candidates(self.timetable, request, window);
                let mut bags: Vec<Vec<Arrived>> = vec![Vec::new(); bag_count];
                let mut sink = Some(FrontierSink {
                    slots: &slots,
                    egress,
                    egress_active,
                    bags: &mut bags,
                });
                let (arena, _) = self.passes(request, &departures, bucket, &mut None, &mut sink);
                let cells: Vec<Vec<Journey>> = bags
                    .iter()
                    .map(|bag| {
                        let mut journeys: Vec<Journey> = bag
                            .iter()
                            .map(|arrived| self.assemble(arrived, &arena))
                            .collect();
                        journeys.sort_by_key(|journey| {
                            (journey.departure, journey.arrival, journey.rides())
                        });
                        journeys
                    })
                    .collect();
                if egress_active || unique == destinations.len() {
                    return cells;
                }
                cell_of.iter().map(|&cell| cells[cell].clone()).collect()
            })
            .collect()
    }

    /// The least-emissions cost matrix over this engine's candidate
    /// set: per origin–destination cell, the cleanest journey (ties
    /// toward the shorter travel time) among the (departure, arrival,
    /// emissions bucket) Pareto candidates of the departure window,
    /// optionally within a travel-time `budget`. The McTBTR
    /// counterpart of the McRAPTOR matrix: one engine build serves
    /// every origin, fanned out with rayon.
    pub fn least_emissions_matrix(
        &self,
        inputs: &CostInputs<'_>,
        requests: &[Request],
        destinations: &[StopIdx],
        window: u32,
        budget: Option<u32>,
        bucket: f64,
    ) -> Vec<Vec<CostRow>> {
        let mut slots = vec![0u32; self.timetable.stop_count() as usize];
        for (index, stop) in destinations.iter().enumerate() {
            slots[stop.0 as usize] = index as u32 + 1;
        }
        requests
            .par_iter()
            .map(|request| {
                let departures = departure_candidates(self.timetable, request, window);
                let mut best: Vec<Option<Winner>> = vec![None; destinations.len()];
                let mut fold = Some(MatrixSink {
                    slots: &slots,
                    budget,
                    best: &mut best,
                });
                let (arena_out, _) =
                    self.passes(request, &departures, bucket, &mut fold, &mut None);
                best.into_iter()
                    .enumerate()
                    .filter_map(|(slot, winner)| {
                        winner.map(|winner| {
                            self.cost_row(inputs, &winner, &arena_out, destinations[slot])
                        })
                    })
                    .collect()
            })
            .collect()
    }

    /// Walks a matrix winner's chain into a cost row, mirroring the
    /// McRAPTOR matrix reconstruction; the sentinel leaf is the
    /// origin's zero-ride floor.
    fn cost_row(
        &self,
        inputs: &CostInputs<'_>,
        winner: &Winner,
        arena: &[Segment],
        to: StopIdx,
    ) -> CostRow {
        let mut rides = 0u32;
        let mut transit_meters = 0.0;
        let mut walk_meters = 0.0;
        let mut legs: Vec<(TripIdx, u16, u16)> = Vec::new();
        let mut fare_legs: Vec<FareLeg> = Vec::new();
        if winner.leaf != ACCESS_LEAF {
            let mut segment = &arena[winner.leaf as usize];
            let mut alight = winner.alight;
            if winner.walked {
                let stops = self
                    .timetable
                    .pattern_stops(self.view.line_pattern(self.view.line_of(segment.trip)));
                walk_meters += self
                    .footpaths
                    .from_stop(stops[alight as usize])
                    .iter()
                    .find(|footpath| footpath.to == to)
                    .map(|footpath| footpath.meters)
                    .unwrap_or(0.0);
            }
            loop {
                let trip = segment.trip;
                let backing = self.view.backing(trip);
                rides += 1;
                transit_meters +=
                    inputs.geometry.leg_distance(backing, segment.board, alight) as f64;
                if inputs.with_geometry {
                    legs.push((backing, segment.board, alight));
                }
                if inputs.fares.is_some() {
                    let pattern = self.timetable.trip_pattern(backing);
                    let stops = self.timetable.pattern_stops(pattern);
                    fare_legs.push(FareLeg {
                        route: self.timetable.pattern_route(pattern),
                        board_stop: stops[segment.board as usize].0,
                        alight_stop: stops[alight as usize].0,
                        board_time: self.timetable.trip_stop_times(backing)[segment.board as usize]
                            .departure
                            .saturating_sub(self.view.day_offset(trip)),
                    });
                }
                let board_stop = self
                    .timetable
                    .pattern_stops(self.view.line_pattern(self.view.line_of(trip)))
                    [segment.board as usize];
                match segment.origin {
                    SegOrigin::Access { .. } => break,
                    SegOrigin::Transfer { parent, alight: at }
                    | SegOrigin::Walked {
                        parent, alight: at, ..
                    } => {
                        let parent_segment = &arena[parent as usize];
                        let parent_stops = self.timetable.pattern_stops(
                            self.view
                                .line_pattern(self.view.line_of(parent_segment.trip)),
                        );
                        let parent_stop = parent_stops[at as usize];
                        if parent_stop != board_stop {
                            walk_meters += self
                                .footpaths
                                .from_stop(parent_stop)
                                .iter()
                                .find(|footpath| footpath.to == board_stop)
                                .map(|footpath| footpath.meters)
                                .unwrap_or(0.0);
                        }
                        alight = at;
                        segment = parent_segment;
                    }
                }
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
                fare_legs.reverse();
                tables.price(&fare_legs)
            }
            None => f64::NAN,
        };
        CostRow {
            to: to.0,
            seconds: winner.seconds,
            rides,
            transit_meters,
            walk_meters,
            emission_grams: winner.grams,
            fare,
            geometry,
        }
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
#[path = "mctbtr_tests.rs"]
mod tests;
