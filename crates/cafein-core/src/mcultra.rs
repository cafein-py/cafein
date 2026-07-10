//! McULTRA shortcut computation (multicriteria: arrival time, emissions),
//! a transcription of the KIT reference `kit-algo/ULTRA`
//! (`Algorithms/RAPTOR/ULTRA/{McShortcutSearch,McBuilder}.h`), the way
//! [`ultra`](crate::ultra) transcribes `{ShortcutSearch,Builder}.h`.
//!
//! It applies the same `ShortcutSearch → McShortcutSearch` change — scalar
//! arrival labels become per-vertex Pareto **bags** — to cafein's bicriteria
//! search, with **two deliberate adaptations** (see plans/mcultra-plan.md):
//!
//! 1. The second Pareto criterion is **emissions** (grams CO₂e), not KIT's
//!    walking distance. Emissions accrue on *trips*, not walks: `grams` is
//!    carried unchanged across transfer edges (walking is zero-emission) and
//!    grows only in the route scans, by `factor(trip) × leg_distance` — the
//!    McRAPTOR rule ([`mcraptor`](crate::mcraptor)).
//! 2. cafein's `(trip, position)` timetable model, as the bicriteria port does.
//!
//! Domination is **exact** `(arrival, grams)` — a cleaner-but-later transfer
//! therefore survives that the bicriteria (arrival-only) search prunes. The
//! one-ride and two-ride bags stay separate (KIT's `OneTripBag`/`TwoTripsBag`),
//! keeping the transfers axis the consuming McRAPTOR relaxes.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

use rayon::prelude::*;

use crate::exhaustive::quantized;
use crate::geometry::TripGeometry;
use crate::tbtr::{earliest_boardable, DayView, ViewTrip};
use crate::timetable::{StopIdx, Timetable};
use crate::transfers::Transfers;
use crate::ultra::{
    collect_departures, dedup_stops, stations, walk_from, Departure, Shortcut, NEVER,
};

/// The McULTRA shortcut set: intermediate transfers Pareto-sufficient for
/// `(arrival, transfers, emissions)`, computed in parallel over source stops.
/// `factors` is the per-`TripIdx` emission-factor array (NaN = no factor, that
/// trip is skipped), aligned to `DayView::universal`; `min_time`/`max_time`
/// bound the source-departure window. Deduplicated to the shortest walk per
/// origin→destination pair.
pub fn compute_mcultra_shortcuts(
    view: &DayView,
    timetable: &Timetable,
    transfers: &Transfers,
    geometry: &TripGeometry,
    factors: &[f64],
    min_time: u32,
    max_time: u32,
) -> Vec<Shortcut> {
    let stop_count = timetable.stop_count();
    let station = stations(transfers, stop_count);
    let mut all: Vec<Shortcut> = (0..stop_count)
        .into_par_iter()
        .filter(|&stop| station[stop as usize].0 == stop)
        .map_init(
            || {
                Search::new(
                    view, timetable, transfers, geometry, factors, &station, stop_count,
                )
            },
            |search, stop| search.run(StopIdx(stop), min_time, max_time),
        )
        .flatten()
        .collect();
    all.sort_unstable_by(|a, b| {
        (a.origin, a.destination)
            .cmp(&(b.origin, b.destination))
            .then(a.seconds.cmp(&b.seconds))
    });
    all.dedup_by_key(|shortcut| (shortcut.origin, shortcut.destination));
    all
}

/// A one-trip label reaching a stop after one trip and a following walk.
/// `origin`/`origin_arrival` name the shortcut origin (the trip's alight stop
/// and its arrival there) when the journey is a candidate — its trip was
/// boarded in the source station — else `origin` is `None` (a witness).
/// `meters` is the walked distance of the intermediate transfer since the
/// alight, so an emitted shortcut's `seconds` = `arrival − origin_arrival`.
#[derive(Debug, Clone, Copy)]
struct OneLabel {
    arrival: u32,
    grams: f64,
    origin: Option<StopIdx>,
    origin_arrival: u32,
    meters: f64,
}

/// A two-trip label reaching a stop after two trips (and walks). `shortcut`
/// carries the intermediate transfer to emit if the journey is a candidate that
/// survives the final search, else `None` (a witness).
#[derive(Debug, Clone, Copy)]
struct TwoLabel {
    arrival: u32,
    grams: f64,
    shortcut: Option<Shortcut>,
}

impl OneLabel {
    fn is_candidate(&self) -> bool {
        self.origin.is_some()
    }
}

impl TwoLabel {
    fn is_candidate(&self) -> bool {
        self.shortcut.is_some()
    }

    /// Whether `other` names the same frontier point — `(arrival, grams, kind)` —
    /// used on pop to detect a queued label its bag has since evicted (a witness
    /// took its place), so a witnessed-away candidate does not emit.
    fn same(&self, other: &TwoLabel) -> bool {
        self.arrival == other.arrival
            && self.grams.to_bits() == other.grams.to_bits()
            && self.is_candidate() == other.is_candidate()
    }
}

/// Whether `(a_arrival, a_grams)` weakly Pareto-dominates `(b_arrival, b_grams)`.
#[inline]
fn dominates(a_arrival: u32, a_grams: f64, b_arrival: u32, b_grams: f64) -> bool {
    a_arrival <= b_arrival && a_grams <= b_grams
}

/// On an exact `(arrival, grams)` tie a witness dominates a candidate (the
/// shortcut is then unnecessary), while a strict dominance always wins — the
/// multicriteria form of the bicriteria port's strict-improvement rule. `a` is
/// the potential dominator, `b` the label under threat of eviction.
#[inline]
fn out_dominates(
    a_arrival: u32,
    a_grams: f64,
    a_candidate: bool,
    b_arrival: u32,
    b_grams: f64,
    b_candidate: bool,
) -> bool {
    if !dominates(a_arrival, a_grams, b_arrival, b_grams) {
        return false;
    }
    if a_arrival < b_arrival || a_grams < b_grams {
        return true; // strict on one axis
    }
    // Exact tie: `a` wins unless it is a candidate and `b` a witness.
    !a_candidate || b_candidate
}

/// Inserts `label` into a one-trip bag under the dominance above, evicting what
/// it dominates. Returns whether it entered the frontier.
fn insert_one(bag: &mut Vec<OneLabel>, label: OneLabel) -> bool {
    if bag.iter().any(|e| {
        out_dominates(
            e.arrival,
            e.grams,
            e.is_candidate(),
            label.arrival,
            label.grams,
            label.is_candidate(),
        )
    }) {
        return false;
    }
    bag.retain(|e| {
        !out_dominates(
            label.arrival,
            label.grams,
            label.is_candidate(),
            e.arrival,
            e.grams,
            e.is_candidate(),
        )
    });
    bag.push(label);
    true
}

/// Inserts `label` into a two-trip bag; same dominance rule as [`insert_one`].
fn insert_two(bag: &mut Vec<TwoLabel>, label: TwoLabel) -> bool {
    if bag.iter().any(|e| {
        out_dominates(
            e.arrival,
            e.grams,
            e.is_candidate(),
            label.arrival,
            label.grams,
            label.is_candidate(),
        )
    }) {
        return false;
    }
    bag.retain(|e| {
        !out_dominates(
            label.arrival,
            label.grams,
            label.is_candidate(),
            e.arrival,
            e.grams,
            e.is_candidate(),
        )
    });
    bag.push(label);
    true
}

/// A per-worker multicriteria shortcut search from one source stop.
struct Search<'a> {
    view: &'a DayView,
    timetable: &'a Timetable,
    transfers: &'a Transfers,
    geometry: &'a TripGeometry,
    factors: &'a [f64],
    station: &'a [StopIdx],
    stop_count: u32,
    source_rep: StopIdx,
    direct: Vec<u32>,
    emitted: HashSet<(u32, u32)>,
    shortcuts: Vec<Shortcut>,
    /// Round-0 walk arrival per stop (grams 0); a witness in all rounds.
    zero: Vec<u32>,
    /// Per-stop one-trip and two-trip Pareto bags.
    one: Vec<Vec<OneLabel>>,
    two: Vec<Vec<TwoLabel>>,
    updated_by_route: Vec<StopIdx>,
    updated_by_transfer: Vec<StopIdx>,
}

impl<'a> Search<'a> {
    fn new(
        view: &'a DayView,
        timetable: &'a Timetable,
        transfers: &'a Transfers,
        geometry: &'a TripGeometry,
        factors: &'a [f64],
        station: &'a [StopIdx],
        stop_count: u32,
    ) -> Search<'a> {
        let n = stop_count as usize;
        Search {
            view,
            timetable,
            transfers,
            geometry,
            factors,
            station,
            stop_count,
            source_rep: StopIdx(0),
            direct: Vec::new(),
            emitted: HashSet::new(),
            shortcuts: Vec::new(),
            zero: vec![NEVER; n],
            one: vec![Vec::new(); n],
            two: vec![Vec::new(); n],
            updated_by_route: Vec::new(),
            updated_by_transfer: Vec::new(),
        }
    }

    fn run(&mut self, source: StopIdx, min_time: u32, max_time: u32) -> Vec<Shortcut> {
        if self.station[source.0 as usize] != source {
            return Vec::new();
        }
        self.source_rep = source;
        self.direct = walk_from(self.transfers, source, self.stop_count);
        self.emitted.clear();
        self.shortcuts.clear();
        let departures = collect_departures(
            self.view,
            self.timetable,
            &self.direct,
            source,
            self.station,
            min_time,
            max_time,
        );
        for departure in departures {
            self.run_departure(&departure);
        }
        std::mem::take(&mut self.shortcuts)
    }

    fn run_departure(&mut self, departure: &Departure) {
        self.clear_labels();
        self.relax_initial(departure.time);
        self.scan_initial_routes(departure);
        self.intermediate_dijkstra();
        self.scan_routes();
        self.final_dijkstra();
    }

    fn clear_labels(&mut self) {
        for value in self.zero.iter_mut() {
            *value = NEVER;
        }
        for bag in self.one.iter_mut() {
            bag.clear();
        }
        for bag in self.two.iter_mut() {
            bag.clear();
        }
        self.updated_by_route.clear();
        self.updated_by_transfer.clear();
    }

    /// Round-0 walk: every stop reachable from the source at `departure + walk`,
    /// grams 0 — a witness in all rounds.
    fn relax_initial(&mut self, departure: u32) {
        for stop in 0..self.stop_count as usize {
            if self.direct[stop] == NEVER {
                continue;
            }
            let arrival = departure.saturating_add(self.direct[stop]);
            self.zero[stop] = arrival;
            insert_one(
                &mut self.one[stop],
                OneLabel {
                    arrival,
                    grams: 0.0,
                    origin: None,
                    origin_arrival: arrival,
                    meters: 0.0,
                },
            );
            insert_two(
                &mut self.two[stop],
                TwoLabel {
                    arrival,
                    grams: 0.0,
                    shortcut: None,
                },
            );
        }
    }

    /// Round 1: board the departure's first-trip segments (from the source
    /// station), riding the emissions frontier, recording one-trip labels.
    fn scan_initial_routes(&mut self, departure: &Departure) {
        self.updated_by_route.clear();
        for &(line, position) in &departure.routes {
            let stops = self.timetable.pattern_stops(self.view.line_pattern(line));
            let board = stops[position as usize];
            let ready = self.zero[board.0 as usize];
            if ready == NEVER {
                continue;
            }
            let candidate = self.station[board.0 as usize] == self.source_rep;
            let rides = self.frontier(line, position, ready);
            for (trip, factor) in rides {
                let offset = self.view.line_day_offset(line);
                let times = self.view.stored_times(self.timetable, trip);
                let backing = self.view.backing(trip);
                for pos in position as usize + 1..stops.len() {
                    let stop = stops[pos];
                    let arrival = times[pos].arrival.saturating_sub(offset);
                    let meters = self.geometry.leg_distance(backing, position, pos as u16) as f64;
                    let grams = quantized(meters / 1000.0 * factor);
                    let label = OneLabel {
                        arrival,
                        grams,
                        origin: candidate.then_some(stop),
                        origin_arrival: arrival,
                        meters: 0.0,
                    };
                    if insert_one(&mut self.one[stop.0 as usize], label) {
                        self.updated_by_route.push(stop);
                        // A one-trip journey is a witness for two-trip journeys
                        // reaching the same stop (fewer rides, same criteria).
                        insert_two(
                            &mut self.two[stop.0 as usize],
                            TwoLabel {
                                arrival,
                                grams,
                                shortcut: None,
                            },
                        );
                    }
                }
            }
        }
        dedup_stops(&mut self.updated_by_route);
    }

    /// Round 2: board the routes serving the intermediate-transfer-updated
    /// stops, riding the emissions frontier, recording two-trip labels that
    /// carry the shortcut they would emit.
    fn scan_routes(&mut self) {
        let boarding = std::mem::take(&mut self.updated_by_transfer);
        self.updated_by_route.clear();
        for board in boarding {
            let contexts: Vec<OneLabel> = self.one[board.0 as usize].clone();
            for served in self.timetable.patterns_at_stop(board) {
                let pattern_stops = self.timetable.pattern_stops(served.pattern);
                if served.position as usize + 1 >= pattern_stops.len() {
                    continue;
                }
                for line in self
                    .view
                    .lines_of_pattern(served.pattern)
                    .into_iter()
                    .flatten()
                {
                    for context in &contexts {
                        self.ride_second(line, served.position, board, *context);
                    }
                }
            }
        }
        dedup_stops(&mut self.updated_by_route);
    }

    /// Rides trip 2 on `line` from `board` for one boarding `context` (a
    /// one-trip label reaching `board`), inserting two-trip labels downstream.
    fn ride_second(&mut self, line: u32, position: u16, board: StopIdx, context: OneLabel) {
        let shortcut = self.candidate_shortcut(board, &context);
        let rides = self.frontier(line, position, context.arrival);
        let stops = self.timetable.pattern_stops(self.view.line_pattern(line));
        for (trip, factor) in rides {
            let offset = self.view.line_day_offset(line);
            let times = self.view.stored_times(self.timetable, trip);
            let backing = self.view.backing(trip);
            for pos in position as usize + 1..stops.len() {
                let stop = stops[pos];
                let arrival = times[pos].arrival.saturating_sub(offset);
                let meters = self.geometry.leg_distance(backing, position, pos as u16) as f64;
                let grams = quantized(context.grams + meters / 1000.0 * factor);
                let label = TwoLabel {
                    arrival,
                    grams,
                    shortcut,
                };
                if insert_two(&mut self.two[stop.0 as usize], label) {
                    self.updated_by_route.push(stop);
                }
            }
        }
    }

    /// The shortcut a two-trip journey boarding trip 2 at `board` would emit:
    /// walk from the candidate's origin (a distinct alight, not already
    /// emitted) to `board`, else `None` for a witness boarding.
    fn candidate_shortcut(&self, board: StopIdx, context: &OneLabel) -> Option<Shortcut> {
        match context.origin {
            Some(origin) if origin != board && !self.emitted.contains(&(origin.0, board.0)) => {
                Some(Shortcut {
                    origin,
                    destination: board,
                    seconds: context.arrival - context.origin_arrival,
                    meters: context.meters,
                })
            }
            _ => None,
        }
    }

    /// The emissions frontier of trips on `line` boardable at `position` after
    /// `ready`: the earliest boardable trip and every later trip whose factor
    /// strictly improves — the `(arrival, grams)` Pareto trips (mcraptor's rule).
    /// Trips without a finite factor are skipped.
    fn frontier(&self, line: u32, position: u16, ready: u32) -> Vec<(ViewTrip, f64)> {
        let Some(first) = earliest_boardable(self.view, self.timetable, line, position, ready)
        else {
            return Vec::new();
        };
        let mut rides = Vec::new();
        let mut cleanest = f64::INFINITY;
        for rank in first.0..self.view.line_trips(line).end {
            let trip = ViewTrip(rank);
            let factor = self.factors[self.view.backing(trip).0 as usize];
            if !factor.is_finite() || factor >= cleanest {
                continue;
            }
            cleanest = factor;
            rides.push((trip, factor));
        }
        rides
    }

    /// The intermediate transfer: a multicriteria Dijkstra over the transfer
    /// graph from the one-trip labels, carrying `grams` unchanged (walks do not
    /// emit) and growing `arrival`/`meters` per edge. Every settled stop boards
    /// trip 2 (its witness labels prune downstream candidates; its candidate
    /// labels seed shortcuts). Bounded like the bicriteria port (Baum §3.3): once
    /// no candidate label is still queued, the rest can only witness stops those
    /// candidates already settled, so the search stops.
    fn intermediate_dijkstra(&mut self) {
        let transfers = self.transfers;
        let seeds = std::mem::take(&mut self.updated_by_route);
        let mut heap: BinaryHeap<Reverse<Queued<OneLabel>>> = BinaryHeap::new();
        let mut pending: usize = 0;
        for stop in &seeds {
            // The intermediate transfer is measured from the trip-1 alight.
            for label in self.one[stop.0 as usize].iter_mut() {
                label.meters = 0.0;
            }
            for label in &self.one[stop.0 as usize] {
                heap.push(Reverse(Queued::new(
                    label.arrival,
                    label.grams,
                    *stop,
                    *label,
                )));
                if label.is_candidate() {
                    pending += 1;
                }
            }
        }
        self.updated_by_transfer.clear();
        let mut settled: HashSet<StopIdx> = HashSet::new();
        while let Some(Reverse(item)) = heap.pop() {
            let label = item.label;
            if label.is_candidate() {
                pending -= 1;
            }
            if settled.insert(item.stop) {
                self.updated_by_transfer.push(item.stop);
            }
            for edge in transfers.from_stop(item.stop) {
                let next = OneLabel {
                    arrival: label.arrival.saturating_add(edge.duration),
                    grams: label.grams,
                    origin: label.origin,
                    origin_arrival: label.origin_arrival,
                    meters: label.meters + edge.meters,
                };
                if insert_one(&mut self.one[edge.to.0 as usize], next) {
                    heap.push(Reverse(Queued::new(
                        next.arrival,
                        next.grams,
                        edge.to,
                        next,
                    )));
                    if next.is_candidate() {
                        pending += 1;
                    }
                    // Witnessify into the two-trip bag (as round 1 does).
                    insert_two(
                        &mut self.two[edge.to.0 as usize],
                        TwoLabel {
                            arrival: next.arrival,
                            grams: next.grams,
                            shortcut: None,
                        },
                    );
                }
            }
            if pending == 0 {
                break;
            }
        }
    }

    /// The final transfer: a multicriteria Dijkstra over the transfer graph from
    /// the two-trip labels. A candidate label that settles still on its bag's
    /// frontier — no witness has dominated it — emits its shortcut. Bounded on
    /// pending candidates like [`intermediate_dijkstra`].
    fn final_dijkstra(&mut self) {
        let transfers = self.transfers;
        let seeds = std::mem::take(&mut self.updated_by_route);
        let mut heap: BinaryHeap<Reverse<Queued<TwoLabel>>> = BinaryHeap::new();
        let mut pending: usize = 0;
        for stop in &seeds {
            for label in &self.two[stop.0 as usize] {
                heap.push(Reverse(Queued::new(
                    label.arrival,
                    label.grams,
                    *stop,
                    *label,
                )));
                if label.is_candidate() {
                    pending += 1;
                }
            }
        }
        while let Some(Reverse(item)) = heap.pop() {
            let label = item.label;
            if label.is_candidate() {
                pending -= 1;
                // Skip a candidate its bag has since evicted (a witness dominated
                // it): its shortcut is unnecessary and must not be emitted.
                if !self.two[item.stop.0 as usize]
                    .iter()
                    .any(|e| e.same(&label))
                {
                    if pending == 0 {
                        break;
                    }
                    continue;
                }
            }
            if let Some(shortcut) = label.shortcut {
                if self
                    .emitted
                    .insert((shortcut.origin.0, shortcut.destination.0))
                {
                    self.shortcuts.push(shortcut);
                }
            }
            for edge in transfers.from_stop(item.stop) {
                let next = TwoLabel {
                    arrival: label.arrival.saturating_add(edge.duration),
                    grams: label.grams,
                    shortcut: label.shortcut,
                };
                if insert_two(&mut self.two[edge.to.0 as usize], next) {
                    heap.push(Reverse(Queued::new(
                        next.arrival,
                        next.grams,
                        edge.to,
                        next,
                    )));
                    if next.is_candidate() {
                        pending += 1;
                    }
                }
            }
            if pending == 0 {
                break;
            }
        }
    }
}

/// A multicriteria-Dijkstra queue item carrying its label. Ordered by
/// `(arrival, grams)` — `grams` held as a bit pattern, a settled quantized
/// non-negative value (never NaN), so the bit order matches the numeric order —
/// so the heap (under `Reverse`) pops the earliest, then cleanest, label. The
/// label rides along; a label evicted from its bag before it is popped is
/// harmless (it relaxes into dominated non-inserts, or emits a
/// superfluous-but-safe shortcut), so no staleness check is needed.
#[derive(Debug, Clone, Copy)]
struct Queued<L> {
    arrival: u32,
    grams_bits: u64,
    stop: StopIdx,
    label: L,
}

impl<L> Queued<L> {
    fn new(arrival: u32, grams: f64, stop: StopIdx, label: L) -> Queued<L> {
        Queued {
            arrival,
            grams_bits: grams.to_bits(),
            stop,
            label,
        }
    }
    fn key(&self) -> (u32, u64, u32) {
        (self.arrival, self.grams_bits, self.stop.0)
    }
}

impl<L> PartialEq for Queued<L> {
    fn eq(&self, other: &Queued<L>) -> bool {
        self.key() == other.key()
    }
}
impl<L> Eq for Queued<L> {}
impl<L> Ord for Queued<L> {
    fn cmp(&self, other: &Queued<L>) -> std::cmp::Ordering {
        self.key().cmp(&other.key())
    }
}
impl<L> PartialOrd for Queued<L> {
    fn partial_cmp(&self, other: &Queued<L>) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
#[path = "mcultra_tests.rs"]
mod tests;
