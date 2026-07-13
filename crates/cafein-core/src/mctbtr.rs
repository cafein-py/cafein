//! McTBTR: multicriteria trip-based routing over (arrival, emissions).
//!
//! This module holds the multicriteria transfer set — the precompute
//! stage. Witt's transfer reduction is unsound under a second
//! criterion (a covering transfer may ride a dirtier vehicle), so the
//! set is built by **global candidate/witness enumeration** instead
//! (Baum et al. 2023's integrated preprocessing, over (arrival,
//! grams)): from every source stop and departure, ride the
//! earliest-or-strictly-cleaner first trips, transfer same-stop onto
//! second trips under the same boarding frontier, and keep a transfer only
//! when a two-trip journey using it survives every witness in the
//! per-stop Pareto bags (strict dominance always evicts; an exact tie
//! resolves by a context-independent order — witness over candidate,
//! then lower label identity — so every enumeration context elects
//! the same canonical journey). Grams anchor at the real
//! boarding, strictly stronger than a trip-local reduction: a
//! transfer survives only where a genuine origin context needs it,
//! and the set is deliberately **not** a superset of the time set.
//! The same-line skip applies only to siblings whose factor is no
//! cleaner; U-turn drops stay (the alight-one-stop-earlier
//! alternative rides strictly less distance on the current trip and
//! the identical distance on the boarded one). Trips without a
//! resolved factor are never boarded and get no transfers — journeys
//! riding them are excluded from emissions frontiers by contract.
//! The set is footpath-blind: walked transfers are the query's job
//! (it boards over installed footpaths from every scanned alight), so
//! one set serves every footpath choice. Dominance is exact (no
//! bucketing): the set must stay complete for every query bucket.

use rayon::prelude::*;

use crate::exhaustive::quantized;
use crate::fares::FareLeg;
use crate::geometry::{wkb_multi_line_string, TripGeometry};
use crate::journey::{Journey, Leg};
use crate::mcraptor::Bag;
use crate::raptor::{departure_candidates, CostInputs, CostRow};
use crate::router::Request;
use crate::tbtr::{
    earliest_boardable, DayView, TransferSet, TransferSetBuild, TripTransfer, ViewTrip,
};
use crate::timetable::{StopIdx, Timetable, TripIdx};
use crate::transfers::{Transfer, Transfers};

/// Builds the multicriteria transfer set of a day view by **global
/// candidate/witness enumeration** (Baum et al. 2023's integrated
/// preprocessing, adapted to (arrival, grams) and cafein's trip-based
/// set): from every source stop, a descending run per unique departure
/// time boards the first-trip frontier, transfers same-stop onto
/// second-trip candidates, and keeps a trip-transfer only when a two-trip journey
/// using it survives every witness — one-trip journeys and competing
/// alternatives from the same origin context. Fanned out over source
/// stops with rayon; `factors` are grams CO₂e per passenger-kilometre
/// per backing trip, NaN where unresolved. `TransferSetBuild::generated`
/// reports the kept-transfer count (the enumeration never materializes
/// an unreduced set).
pub fn transfer_set(
    view: &DayView,
    timetable: &Timetable,
    geometry: &TripGeometry,
    factors: &[f64],
) -> TransferSetBuild {
    let chains = CleanerChains::build(view, factors);
    let per_source: Vec<Vec<(u32, u16, TripTransfer)>> = (0..timetable.stop_count())
        .into_par_iter()
        .map_init(
            || SetSearch::new(view, timetable, geometry, factors, &chains),
            |search, stop| search.run(StopIdx(stop)),
        )
        .collect();
    let mut all: Vec<(u32, u16, TripTransfer)> = per_source.into_iter().flatten().collect();
    all.sort_unstable_by_key(|&(trip, alight, transfer)| {
        (trip, alight, transfer.trip.0, transfer.position)
    });
    all.dedup();
    let mut per_trip: Vec<(Vec<Vec<TripTransfer>>, usize)> = (0..view.trip_count())
        .map(|trip| {
            let line = view.line_of(ViewTrip(trip));
            let positions = timetable.pattern_stops(view.line_pattern(line)).len();
            (vec![Vec::new(); positions], 0)
        })
        .collect();
    for (trip, alight, transfer) in all {
        let (positions, kept) = &mut per_trip[trip as usize];
        positions[alight as usize].push(transfer);
        *kept += 1;
    }
    TransferSet::assemble(per_trip)
}

/// The boarding candidates of every line, precomputed: from any first
/// boardable trip, the trips worth boarding are the first one with a
/// resolved factor followed by its strictly-decreasing-factor chain —
/// exactly the set the naive suffix walk keeps, without visiting the
/// skipped trips. `first_candidate[rank]` is the first rank at or after
/// `rank` in the same line whose factor resolves; `next_cleaner[rank]`
/// is the next rank after `rank` in the same line whose factor is
/// strictly cleaner. `u32::MAX` ends a chain.
pub(crate) struct CleanerChains {
    first_candidate: Vec<u32>,
    next_cleaner: Vec<u32>,
}

impl CleanerChains {
    pub(crate) fn build(view: &DayView, factors: &[f64]) -> CleanerChains {
        let trips = view.trip_count() as usize;
        let mut first_candidate = vec![u32::MAX; trips];
        let mut next_cleaner = vec![u32::MAX; trips];
        let mut stack: Vec<u32> = Vec::new();
        for line in 0..view.line_count() {
            // Backward over the line: the classic next-smaller-element
            // stack yields each rank's next strictly cleaner trip, and
            // the running first resolved rank fills `first_candidate`.
            stack.clear();
            let mut first = u32::MAX;
            for rank in view.line_trips(line).rev() {
                let factor = factors[view.backing(ViewTrip(rank)).0 as usize];
                if factor.is_finite() {
                    while let Some(&top) = stack.last() {
                        let top_factor = factors[view.backing(ViewTrip(top)).0 as usize];
                        if top_factor >= factor {
                            stack.pop();
                        } else {
                            break;
                        }
                    }
                    next_cleaner[rank as usize] = stack.last().copied().unwrap_or(u32::MAX);
                    stack.push(rank);
                    first = rank;
                }
                first_candidate[rank as usize] = first;
            }
        }
        CleanerChains {
            first_candidate,
            next_cleaner,
        }
    }

    /// The boarding candidates from the first boardable rank, in order:
    /// the first resolved trip, then each strictly cleaner successor.
    pub(crate) fn candidates(&self, first: u32) -> CleanerChain<'_> {
        CleanerChain {
            chains: self,
            next: self.first_candidate[first as usize],
            started: false,
        }
    }
}

pub(crate) struct CleanerChain<'a> {
    chains: &'a CleanerChains,
    next: u32,
    started: bool,
}

impl Iterator for CleanerChain<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        if self.started && self.next != u32::MAX {
            self.next = self.chains.next_cleaner[self.next as usize];
        }
        self.started = true;
        (self.next != u32::MAX).then_some(self.next)
    }
}

/// Grams CO₂e accumulated along `trip` from its first position to
/// `position`; callers subtract the boarding position's value to
/// anchor a leg's grams at the real boarding.
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

/// A one-trip label during the set enumeration: the journey's arrival
/// and grams at a stop, and — when it rode a trip — the alight event a
/// kept transfer would depart from. `None` marks the zero-trip witness
/// (staying at the source).
#[derive(Debug, Clone, Copy)]
struct OneCtx {
    arrival: u32,
    grams: f64,
    event: Option<(ViewTrip, u16)>,
}

/// A two-trip label: the journey's arrival and grams, and the candidate
/// transfer it would emit — `None` for witnesses (one- or zero-trip
/// journeys mirrored into the two-trip bags).
#[derive(Debug, Clone, Copy)]
struct TwoCtx {
    arrival: u32,
    grams: f64,
    pair: Option<(ViewTrip, u16, TripTransfer)>,
}

impl OneCtx {
    /// The label's context-independent identity, witnesses first.
    fn rank(&self) -> (u8, u32, u16) {
        match self.event {
            None => (0, 0, 0),
            Some((trip, position)) => (1, trip.0, position),
        }
    }
}

/// Whether `a` dominates `b` in a one-trip bag: strictly better on
/// (arrival, grams), or an exact tie resolved by the global identity
/// order — witnesses beat candidates, then the lower event wins. The
/// tie order is context-independent so every enumeration context
/// elects the same canonical label, keeping witness chains
/// substitutable across contexts (an inconsistent tie-break can evict
/// a transfer in every context that needs it, each citing the other).
fn dominates_one(a: &OneCtx, b: &OneCtx) -> bool {
    if a.arrival > b.arrival || a.grams > b.grams {
        return false;
    }
    if a.arrival < b.arrival || a.grams < b.grams {
        return true;
    }
    a.rank() <= b.rank()
}

/// The two-trip counterpart of [`dominates_one`], with the candidate
/// transfer pair as the identity.
fn dominates_two(a: &TwoCtx, b: &TwoCtx) -> bool {
    if a.arrival > b.arrival || a.grams > b.grams {
        return false;
    }
    if a.arrival < b.arrival || a.grams < b.grams {
        return true;
    }
    a.rank() <= b.rank()
}

impl TwoCtx {
    /// The label's context-independent identity, witnesses first.
    fn rank(&self) -> (u8, u32, u16, u32, u16) {
        match self.pair {
            None => (0, 0, 0, 0, 0),
            Some((first, alight, transfer)) => {
                (1, first.0, alight, transfer.trip.0, transfer.position)
            }
        }
    }

    /// Whether `other` is the same frontier point — used at extraction
    /// to detect a candidate its bag has since evicted.
    fn same(&self, other: &TwoCtx) -> bool {
        self.arrival == other.arrival
            && self.grams.to_bits() == other.grams.to_bits()
            && match (self.pair, other.pair) {
                (None, None) => true,
                (Some(a), Some(b)) => a == b,
                _ => false,
            }
    }
}

/// A per-worker enumeration from one source stop: per-stop one- and
/// two-trip Pareto bags under [`dominates_one`]/[`dominates_two`] —
/// strict dominance always evicts, an exact tie resolves by the
/// global identity order — shared across the source's descending
/// departure runs
/// (a witness from a later departure is catchable from every earlier
/// context, so its evictions are sound), with each run's candidates
/// extracted **at the end of that run**, before any earlier-departure
/// label exists.
struct SetSearch<'a> {
    view: &'a DayView,
    timetable: &'a Timetable,
    geometry: &'a TripGeometry,
    factors: &'a [f64],
    chains: &'a CleanerChains,
    one: Vec<Vec<OneCtx>>,
    two: Vec<Vec<TwoCtx>>,
    touched: Vec<u32>,
    marked: Vec<bool>,
    out: Vec<(u32, u16, TripTransfer)>,
}

impl<'a> SetSearch<'a> {
    fn new(
        view: &'a DayView,
        timetable: &'a Timetable,
        geometry: &'a TripGeometry,
        factors: &'a [f64],
        chains: &'a CleanerChains,
    ) -> SetSearch<'a> {
        let stops = timetable.stop_count() as usize;
        SetSearch {
            view,
            timetable,
            geometry,
            factors,
            chains,
            one: vec![Vec::new(); stops],
            two: vec![Vec::new(); stops],
            touched: Vec::new(),
            marked: vec![false; stops],
            out: Vec::new(),
        }
    }

    /// Every unique origin departure at `source`, descending.
    fn departures_at(&self, source: StopIdx) -> Vec<u32> {
        let mut departures = Vec::new();
        for served in self.timetable.patterns_at_stop(source) {
            let positions = self.timetable.pattern_stops(served.pattern).len();
            if served.position as usize + 1 >= positions {
                continue;
            }
            for line in self
                .view
                .lines_of_pattern(served.pattern)
                .into_iter()
                .flatten()
            {
                let offset = self.view.line_day_offset(line);
                for rank in self.view.line_trips(line) {
                    let times = self.view.stored_times(self.timetable, ViewTrip(rank));
                    // A previous-day trip's pre-midnight events are
                    // yesterday's departures, not boardable today.
                    if let Some(departure) = times[served.position as usize]
                        .departure
                        .checked_sub(offset)
                    {
                        departures.push(departure);
                    }
                }
            }
        }
        departures.sort_unstable_by(|a, b| b.cmp(a));
        departures.dedup();
        departures
    }

    fn touch(&mut self, stop: StopIdx) {
        if !self.marked[stop.0 as usize] {
            self.marked[stop.0 as usize] = true;
            self.touched.push(stop.0);
        }
    }

    fn insert_one(&mut self, stop: StopIdx, label: OneCtx) -> bool {
        let bag = &self.one[stop.0 as usize];
        if bag.iter().any(|entry| dominates_one(entry, &label)) {
            return false;
        }
        self.touch(stop);
        let bag = &mut self.one[stop.0 as usize];
        bag.retain(|entry| !dominates_one(&label, entry));
        bag.push(label);
        true
    }

    fn insert_two(&mut self, stop: StopIdx, label: TwoCtx) -> bool {
        let bag = &self.two[stop.0 as usize];
        if bag.iter().any(|entry| dominates_two(entry, &label)) {
            return false;
        }
        self.touch(stop);
        let bag = &mut self.two[stop.0 as usize];
        bag.retain(|entry| !dominates_two(&label, entry));
        bag.push(label);
        true
    }

    fn run(&mut self, source: StopIdx) -> Vec<(u32, u16, TripTransfer)> {
        for &stop in &self.touched {
            self.one[stop as usize].clear();
            self.two[stop as usize].clear();
            self.marked[stop as usize] = false;
        }
        self.touched.clear();
        self.out.clear();
        for departure in self.departures_at(source) {
            self.run_departure(source, departure);
        }
        std::mem::take(&mut self.out)
    }

    fn run_departure(&mut self, source: StopIdx, departure: u32) {
        // The zero-trip witness: staying at the source.
        self.insert_one(
            source,
            OneCtx {
                arrival: departure,
                grams: 0.0,
                event: None,
            },
        );
        self.insert_two(
            source,
            TwoCtx {
                arrival: departure,
                grams: 0.0,
                pair: None,
            },
        );
        // Round 1: the first-trip frontier from the source.
        let mut fresh: Vec<(StopIdx, OneCtx)> = Vec::new();
        for served in self.timetable.patterns_at_stop(source) {
            let stops = self.timetable.pattern_stops(served.pattern);
            if served.position as usize + 1 >= stops.len() {
                continue;
            }
            for line in self
                .view
                .lines_of_pattern(served.pattern)
                .into_iter()
                .flatten()
            {
                let Some(first) =
                    earliest_boardable(self.view, self.timetable, line, served.position, departure)
                else {
                    continue;
                };
                let offset = self.view.line_day_offset(line);
                for rank in self.chains.candidates(first.0) {
                    let trip = ViewTrip(rank);
                    let times = self.view.stored_times(self.timetable, trip);
                    let boarded_from = absolute_grams(
                        self.geometry,
                        self.view,
                        self.factors,
                        trip,
                        served.position,
                    );
                    for alight in served.position as usize + 1..stops.len() {
                        let arrival = times[alight].arrival - offset;
                        let grams = quantized(
                            absolute_grams(
                                self.geometry,
                                self.view,
                                self.factors,
                                trip,
                                alight as u16,
                            ) - boarded_from,
                        );
                        let label = OneCtx {
                            arrival,
                            grams,
                            event: Some((trip, alight as u16)),
                        };
                        if self.insert_one(stops[alight], label) {
                            fresh.push((stops[alight], label));
                            self.insert_two(
                                stops[alight],
                                TwoCtx {
                                    arrival,
                                    grams,
                                    pair: None,
                                },
                            );
                        }
                    }
                }
            }
        }
        // Round 2: transfer same-stop onto second-trip candidates —
        // walked transfers never enter the set; the query boards over
        // installed footpaths from every scanned alight itself.
        let mut inserted: Vec<(StopIdx, TwoCtx)> = Vec::new();
        for (stop, context) in fresh {
            let Some((first_trip, alight)) = context.event else {
                continue;
            };
            self.board_second(
                stop,
                context.arrival,
                first_trip,
                alight,
                context.grams,
                &mut inserted,
            );
        }
        // Extraction seals this departure's candidates before any
        // earlier-departure label can exist.
        for (stop, label) in inserted {
            if self.two[stop.0 as usize]
                .iter()
                .any(|entry| entry.same(&label))
            {
                let (trip, alight, transfer) = label.pair.expect("recorded labels are candidates");
                let record = (trip.0, alight, transfer);
                if !self.out.contains(&record) {
                    self.out.push(record);
                }
            }
        }
    }

    /// Boards every second-trip candidate at `board` ready at `ready`,
    /// riding it everywhere ahead, under the first trip's same-line
    /// skip. The `u_turn` predicate is structurally unreachable here:
    /// it requires boarding at the first trip's *previous* stop, while
    /// a same-stop transfer boards at the alight stop itself — equal
    /// only where a pattern repeats a stop consecutively, and that
    /// repeat's later context is bag-dominated before round 2 runs
    /// (it arrives no earlier and rides no less than the first
    /// visit), so no candidate the filter could drop is generated.
    fn board_second(
        &mut self,
        board: StopIdx,
        ready: u32,
        first_trip: ViewTrip,
        alight: u16,
        grams: f64,
        inserted: &mut Vec<(StopIdx, TwoCtx)>,
    ) {
        let first_line = self.view.line_of(first_trip);
        let first_factor = self.factors[self.view.backing(first_trip).0 as usize];
        for served in self.timetable.patterns_at_stop(board) {
            let stops = self.timetable.pattern_stops(served.pattern);
            if served.position as usize + 1 >= stops.len() {
                continue;
            }
            for line in self
                .view
                .lines_of_pattern(served.pattern)
                .into_iter()
                .flatten()
            {
                let Some(first) =
                    earliest_boardable(self.view, self.timetable, line, served.position, ready)
                else {
                    continue;
                };
                let offset = self.view.line_day_offset(line);
                for rank in self.chains.candidates(first.0) {
                    let boarded = ViewTrip(rank);
                    let boarded_factor = self.factors[self.view.backing(boarded).0 as usize];
                    if line == first_line
                        && boarded.0 >= first_trip.0
                        && served.position >= alight
                        && boarded_factor >= first_factor
                    {
                        continue;
                    }
                    let pair = Some((
                        first_trip,
                        alight,
                        TripTransfer {
                            trip: boarded,
                            position: served.position,
                        },
                    ));
                    let times = self.view.stored_times(self.timetable, boarded);
                    let boarded_from = absolute_grams(
                        self.geometry,
                        self.view,
                        self.factors,
                        boarded,
                        served.position,
                    );
                    for pos in served.position as usize + 1..stops.len() {
                        let arrival = times[pos].arrival - offset;
                        let ridden = absolute_grams(
                            self.geometry,
                            self.view,
                            self.factors,
                            boarded,
                            pos as u16,
                        ) - boarded_from;
                        let label = TwoCtx {
                            arrival,
                            grams: quantized(grams + ridden),
                            pair,
                        };
                        if self.insert_two(stops[pos], label) {
                            inserted.push((stops[pos], label));
                        }
                    }
                }
            }
        }
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

/// Per-search work counters and round-level phase timers, owned by one
/// `passes` call (no shared atomics: parallel origins each fill their
/// own and the caller reduces them afterwards). Increments are plain
/// integer adds so the instrumentation-off cost stays negligible; the
/// report only prints under `CAFEIN_MCTBTR_PROF`.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct SearchStats {
    /// Closure edge records read from the CSR (one per edge per sweep).
    pub closure_edge_records_loaded: u64,
    /// Label-against-edge relaxations (per admitted label per edge; in
    /// the label-major loop this equals the records loaded).
    pub closure_label_edge_relaxations: u64,
    pub segments_enqueued: u64,
    pub segments_scanned_live: u64,
    pub suffix_context_evaluations: u64,
    pub direct_scan_ns: u64,
    pub expand_ns: u64,
    pub walk_board_ns: u64,
}

impl SearchStats {
    fn absorb(&mut self, other: &SearchStats) {
        self.closure_edge_records_loaded += other.closure_edge_records_loaded;
        self.closure_label_edge_relaxations += other.closure_label_edge_relaxations;
        self.segments_enqueued += other.segments_enqueued;
        self.segments_scanned_live += other.segments_scanned_live;
        self.suffix_context_evaluations += other.suffix_context_evaluations;
        self.direct_scan_ns += other.direct_scan_ns;
        self.expand_ns += other.expand_ns;
        self.walk_board_ns += other.walk_board_ns;
    }

    fn report(&self, label: &str) {
        if std::env::var("CAFEIN_MCTBTR_PROF").is_err() {
            return;
        }
        eprintln!(
            "MCTBTR-STATS {label} closure_edge_records_loaded={} \
             closure_label_edge_relaxations={} segments_enqueued={} \
             segments_scanned_live={} suffix_context_evaluations={} \
             direct_scan_ms={} expand_ms={} walk_board_ms={}",
            self.closure_edge_records_loaded,
            self.closure_label_edge_relaxations,
            self.segments_enqueued,
            self.segments_scanned_live,
            self.suffix_context_evaluations,
            self.direct_scan_ns / 1_000_000,
            self.expand_ns / 1_000_000,
            self.walk_board_ns / 1_000_000,
        );
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

/// A next-round boarding discovered by a round's join sweep, executed
/// by its expansion sweep under the round's pruning envelope.
struct WalkBoard {
    segment: u32,
    alight: u16,
    to: StopIdx,
    reached: u32,
    grams: f64,
    duration: u32,
}

/// The one-pair round's pruning envelope: per arrival, the cleanest
/// key the destination has reached at or before it. A candidate at or
/// above the envelope is dominated — its continuations only grow in
/// arrival and grams, and passes descend, so nothing it leads to can
/// join a new frontier entry (the same plain (arrival ≤, key ≤)
/// dominance the one-pair pruning has always used). The bound is
/// non-increasing in arrival while a segment's alights only grow in
/// both axes, so a pruned alight ends its segment's expansion
/// outright. The batched product runs unpruned: an all-slots envelope
/// is a max over every destination rebuilt from every accumulated
/// frontier each round — measured far costlier at scale than the
/// expansion it trims, and one unserved slot disables it entirely.
struct PruneEnvelope {
    arrivals: Vec<u32>,
    bounds: Vec<i64>,
}

impl PruneEnvelope {
    /// Never prunes: the destination has no entry yet, or the search
    /// has no one-pair destination at all (the batched frontier and
    /// emissions-matrix fold modes).
    fn none() -> PruneEnvelope {
        PruneEnvelope {
            arrivals: Vec::new(),
            bounds: Vec::new(),
        }
    }

    fn build(entries: &[Arrived]) -> PruneEnvelope {
        // The destination's entries sorted by arrival under prefix-min
        // keys: from each arrival, the cleanest key already achieved.
        if entries.is_empty() {
            return PruneEnvelope::none();
        }
        let mut sorted: Vec<(u32, i64)> = entries
            .iter()
            .map(|entry| (entry.arrival, entry.key))
            .collect();
        sorted.sort_unstable();
        let mut prefix = i64::MAX;
        for slot in &mut sorted {
            prefix = prefix.min(slot.1);
            slot.1 = prefix;
        }
        sorted.dedup_by(|later, earlier| {
            if earlier.0 == later.0 {
                earlier.1 = later.1;
                true
            } else {
                false
            }
        });
        let (arrivals, bounds) = sorted.into_iter().unzip();
        PruneEnvelope { arrivals, bounds }
    }

    fn prunes(&self, arrival: u32, key: i64) -> bool {
        match self
            .arrivals
            .partition_point(|&at| at <= arrival)
            .checked_sub(1)
        {
            None => false,
            Some(index) => self.bounds[index] <= key,
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
    chains: CleanerChains,
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
        transfer_set(&view, timetable, geometry, factors).transfers
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
        let chains = CleanerChains::build(&view, factors);
        McTbtrEngine {
            timetable,
            footpaths,
            geometry,
            factors,
            view,
            set: std::borrow::Cow::Owned(set),
            chains,
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
        let chains = CleanerChains::build(&view, factors);
        McTbtrEngine {
            timetable,
            footpaths,
            geometry,
            factors,
            view,
            set: std::borrow::Cow::Borrowed(set),
            chains,
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
        let mut stats = SearchStats::default();
        let (arena, destination) =
            self.passes(request, departures, bucket, fold, &mut None, &mut stats);
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
        stats: &mut SearchStats,
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
                // The join sweep first: alights feed the stop bags and
                // every destination (tightening the frontiers), and the
                // admitted footpath boardings are recorded. The
                // expansion sweep then runs under the round's fully
                // tightened pruning envelope — Baum et al.'s improved
                // trip-based query, on both expansion channels. Only
                // the one-pair profile prunes; see [`PruneEnvelope`].
                let mut walk_boards: Vec<WalkBoard> = Vec::new();
                let mut admitted: Vec<(u32, u16)> = Vec::new();
                stats.segments_scanned_live += segments.len() as u64;
                let scan_started = std::time::Instant::now();
                for &index in &segments {
                    self.scan_joins(
                        index,
                        round,
                        request,
                        key,
                        &arena,
                        &mut stop_bags,
                        &mut destination,
                        fold,
                        frontier,
                        (round < rounds).then_some(&mut walk_boards),
                        &mut admitted,
                        stats,
                    );
                }
                stats.direct_scan_ns += scan_started.elapsed().as_nanos() as u64;
                if round < rounds {
                    // Only the one-pair profile prunes; the batched
                    // frontier and fold modes never build an envelope.
                    let envelope = if frontier.is_none() && fold.is_none() {
                        PruneEnvelope::build(&destination)
                    } else {
                        PruneEnvelope::none()
                    };
                    let expand_started = std::time::Instant::now();
                    self.expand_admitted(
                        &admitted,
                        round,
                        rounds,
                        key,
                        &envelope,
                        &mut arena,
                        &mut trip_bags,
                        &mut queue,
                    );
                    stats.expand_ns += expand_started.elapsed().as_nanos() as u64;
                    let walk_started = std::time::Instant::now();
                    for board in walk_boards {
                        if envelope.prunes(board.reached, key(board.grams)) {
                            continue;
                        }
                        let departure = arena[board.segment as usize].departure;
                        let (parent, alight, duration) =
                            (board.segment, board.alight, board.duration);
                        self.board(
                            board.to,
                            board.reached,
                            board.grams,
                            departure,
                            |_, _| SegOrigin::Walked {
                                parent,
                                alight,
                                duration,
                            },
                            round,
                            key,
                            &mut arena,
                            &mut trip_bags,
                            &mut queue[round + 1],
                        );
                    }
                    stats.walk_board_ns += walk_started.elapsed().as_nanos() as u64;
                }
            }
        }
        stats.segments_enqueued += arena.len() as u64;
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
                for rank in self.chains.candidates(first.0) {
                    let trip = ViewTrip(rank);
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

    /// The join sweep over one segment: alight everywhere ahead, feed
    /// the stop bags, join the egress (directly and over one footpath)
    /// into every destination, and record the admitted footpath
    /// boardings for the expansion sweep.
    #[allow(clippy::too_many_arguments)]
    fn scan_joins(
        &self,
        index: u32,
        round: usize,
        request: &Request,
        key: impl Fn(f64) -> i64 + Copy,
        arena: &[Segment],
        stop_bags: &mut [Bag],
        destination: &mut Vec<Arrived>,
        fold: &mut Option<MatrixSink<'_>>,
        frontier: &mut Option<FrontierSink<'_>>,
        mut walk_boards: Option<&mut Vec<WalkBoard>>,
        admitted: &mut Vec<(u32, u16)>,
        stats: &mut SearchStats,
    ) {
        let segment = arena[index as usize];
        let mut closure_edges = 0u64;
        let trip = segment.trip;
        let line = self.view.line_of(trip);
        let offset = self.view.line_day_offset(line);
        let stops = self.timetable.pattern_stops(self.view.line_pattern(line));
        let times = self.view.stored_times(self.timetable, trip);
        let boarded_from =
            absolute_grams(self.geometry, &self.view, self.factors, trip, segment.board);
        stats.suffix_context_evaluations +=
            (stops.len() - segment.board as usize).saturating_sub(1) as u64;
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
            if !stop_bags[stop.0 as usize].insert(arrival, grams, key(grams), round as u8) {
                continue;
            }
            admitted.push((index, alight as u16));
            self.join_direct_alight(
                request,
                key,
                segment.departure,
                stop,
                arrival,
                grams,
                index,
                alight as u16,
                destination,
                fold,
                frontier,
            );
            for footpath in self.footpaths.from_stop(stop) {
                closure_edges += 1;
                self.relax_closure_edge(
                    request,
                    key,
                    round,
                    footpath,
                    segment.departure,
                    arrival,
                    grams,
                    index,
                    alight as u16,
                    stop_bags,
                    destination,
                    fold,
                    frontier,
                    &mut walk_boards,
                );
            }
        }
        stats.closure_edge_records_loaded += closure_edges;
        stats.closure_label_edge_relaxations += closure_edges;
    }

    /// The direct-alight sink joins for a bag-admitted on-trip
    /// arrival: egress destinations, the frontier, and the fold.
    #[allow(clippy::too_many_arguments)]
    fn join_direct_alight(
        &self,
        request: &Request,
        key: impl Fn(f64) -> i64 + Copy,
        departure: u32,
        stop: StopIdx,
        arrival: u32,
        grams: f64,
        index: u32,
        alight: u16,
        destination: &mut Vec<Arrived>,
        fold: &mut Option<MatrixSink<'_>>,
        frontier: &mut Option<FrontierSink<'_>>,
    ) {
        for &(egress, seconds) in &request.egress {
            if egress == stop {
                self.join(
                    destination,
                    key,
                    departure,
                    arrival.saturating_add(seconds),
                    grams,
                    index,
                    alight,
                    None,
                );
            }
        }
        if let Some(sink) = frontier {
            self.frontier_join(
                sink, stop, key, departure, arrival, grams, index, alight, None,
            );
        }
        if let Some(sink) = fold {
            sink.fold(stop, arrival - departure, grams, index, alight, false);
        }
    }

    /// The closure-target operations for one admitted point relaxed
    /// over one footpath edge: bag admission at the target, then the
    /// fold, egress, frontier, and WalkBoard joins.
    #[allow(clippy::too_many_arguments)]
    fn relax_closure_edge(
        &self,
        request: &Request,
        key: impl Fn(f64) -> i64 + Copy,
        round: usize,
        footpath: &Transfer,
        departure: u32,
        arrival: u32,
        grams: f64,
        index: u32,
        alight: u16,
        stop_bags: &mut [Bag],
        destination: &mut Vec<Arrived>,
        fold: &mut Option<MatrixSink<'_>>,
        frontier: &mut Option<FrontierSink<'_>>,
        walk_boards: &mut Option<&mut Vec<WalkBoard>>,
    ) {
        let reached = arrival.saturating_add(footpath.duration);
        if !stop_bags[footpath.to.0 as usize].insert(reached, grams, key(grams), round as u8) {
            return;
        }
        if let Some(sink) = fold {
            sink.fold(footpath.to, reached - departure, grams, index, alight, true);
        }
        for &(egress, seconds) in &request.egress {
            if egress == footpath.to {
                self.join(
                    destination,
                    key,
                    departure,
                    reached.saturating_add(seconds),
                    grams,
                    index,
                    alight,
                    Some((footpath.to, footpath.duration)),
                );
            }
        }
        if let Some(sink) = frontier {
            self.frontier_join(
                sink,
                footpath.to,
                key,
                departure,
                reached,
                grams,
                index,
                alight,
                Some((footpath.to, footpath.duration)),
            );
        }
        if let Some(boards) = walk_boards.as_deref_mut() {
            boards.push(WalkBoard {
                segment: index,
                alight,
                to: footpath.to,
                reached,
                grams,
                duration: footpath.duration,
            });
        }
    }

    /// The expansion sweep, under the round's pruning envelope: the
    /// join sweep's bag-admitted (segment, alight) pairs relax the
    /// precomputed transfers — a dominated alight's transfers are
    /// covered by its dominator's, whose reduction-complete transfer
    /// set reaches every outcome no later, no dirtier, and on no more
    /// rides, with a departure no earlier (passes descend). A pruned
    /// pair ends its segment's expansion outright: arrivals and grams
    /// only grow along the trip while the envelope's bound never rises.
    #[allow(clippy::too_many_arguments)]
    fn expand_admitted(
        &self,
        admitted: &[(u32, u16)],
        round: usize,
        rounds: usize,
        key: impl Fn(f64) -> i64 + Copy,
        envelope: &PruneEnvelope,
        arena: &mut Vec<Segment>,
        trip_bags: &mut [TripBag],
        queue: &mut [Vec<u32>],
    ) {
        let mut at = 0;
        while at < admitted.len() {
            let (index, _) = admitted[at];
            let segment = arena[index as usize];
            let trip = segment.trip;
            let line = self.view.line_of(trip);
            let offset = self.view.line_day_offset(line);
            let times = self.view.stored_times(self.timetable, trip);
            let boarded_from =
                absolute_grams(self.geometry, &self.view, self.factors, trip, segment.board);
            while at < admitted.len() && admitted[at].0 == index {
                let alight = admitted[at].1;
                at += 1;
                let arrival = times[alight as usize].arrival - offset;
                let ridden = absolute_grams(self.geometry, &self.view, self.factors, trip, alight)
                    - boarded_from;
                let grams = quantized(segment.grams + ridden);
                if envelope.prunes(arrival, key(grams)) {
                    while at < admitted.len() && admitted[at].0 == index {
                        at += 1;
                    }
                    break;
                }
                for transfer in self.set.from_trip_position(trip, alight) {
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
                                alight,
                            },
                        });
                        queue[round + 1].push(next);
                    }
                }
            }
        }
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
        let collected: Vec<(Vec<Vec<Journey>>, SearchStats)> = requests
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
                let mut stats = SearchStats::default();
                let (arena, _) = self.passes(
                    request,
                    &departures,
                    bucket,
                    &mut None,
                    &mut sink,
                    &mut stats,
                );
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
                let cells = if egress_active || unique == destinations.len() {
                    cells
                } else {
                    cell_of.iter().map(|&cell| cells[cell].clone()).collect()
                };
                (cells, stats)
            })
            .collect();
        let mut reduced = SearchStats::default();
        let mut result = Vec::with_capacity(collected.len());
        for (cells, stats) in collected {
            reduced.absorb(&stats);
            result.push(cells);
        }
        reduced.report("frontier_matrix");
        result
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
                let (arena_out, _) = self.passes(
                    request,
                    &departures,
                    bucket,
                    &mut fold,
                    &mut None,
                    &mut SearchStats::default(),
                );
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
