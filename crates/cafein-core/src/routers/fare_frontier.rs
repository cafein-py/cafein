//! The cutoff-pruned (time, fare) frontier search.
//!
//! A profile (range) search over a departure window whose labels carry
//! the exact accumulated fare and the rule-based calculator's full
//! continuation state ([`FareState`]). Fare-blind dominance cannot
//! produce this product — a slower-but-cheaper journey is exactly the
//! low-cutoff winner — so the per-stop bags conjoin arrival and rides
//! with [`state_dominates`], gated per query by the tables'
//! monotonicity checks. The cutoff list trims at two sound points
//! only: labels whose best continuation cannot fit the top cutoff are
//! pruned, and the per-cutoff fold runs as admitted labels reach
//! destination-linked stops, never inside dominance.
//!
//! The product is reconstruction-free — per (pair, cutoff) the
//! minimum travel time with its exact fare and rides — so labels are
//! plain bag entries with no parent chains. Walking is free and
//! relaxes the **closed** transfer closure; unclosed sets (merged or
//! carriage) are out of contract here.

use crate::fares::{state_dominates, FareState, RuleFares};
use crate::routers::raptor::earliest_active_trip;
use crate::routers::router::Request;
use crate::timetable::{StopIdx, Timetable, TripIdx};
use crate::transfers::Transfers;

const UNREACHED: u32 = u32::MAX;
/// Money comparisons tolerate binary-float dust: a 0.1 + 0.2 fare
/// must fit a 0.3 cutoff.
const MONEY_EPSILON: f64 = 1e-9;
const DAY_SECONDS: u32 = 86_400;

/// The network-and-tariff inputs one frontier query runs on.
pub struct FareFrontierInputs<'a> {
    pub timetable: &'a Timetable,
    /// The closed walking closure.
    pub transfers: &'a Transfers,
    pub fares: &'a RuleFares,
    /// Ascending monetary cutoffs; the top one bounds the search.
    pub cutoffs: &'a [f64],
    /// A bound on `arrival − departure`, seconds; `None` unbounded.
    pub max_duration: Option<u32>,
    /// `Some(step)`: rasterise the departure window every `step`
    /// seconds, R5's discipline — every reported journey is real and
    /// waits from its sampled departure, so travel times are measured
    /// against the grid, not the exact event. `None`: one pass per
    /// exact (trip departure − access walk) event — wait-free
    /// departures, the shipped frontier products' event semantics,
    /// far more passes on street-linked origins. A journey catchable
    /// only by waiting past the last in-window event belongs to the
    /// grid, so neither mode's travel times bound the other's at a
    /// window's edge. The caller bounds `window / step`; the bindings
    /// reject more than 100,000 samples.
    pub departure_step: Option<u32>,
    /// `true`: the oracle-exact state bags. `false`: the fast
    /// discipline — the state relation with its gates assumed
    /// monotone, and only the earliest boardable trip explored, the
    /// r5r trade. A kept row's fare is always its real fare; a
    /// cheaper journey may be missed where discounts and transfer
    /// windows interact (tariffs without integrations are exact
    /// either way).
    pub exact: bool,
}

/// One journey candidate at a destination: everything the per-cutoff
/// fold reads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Arrived {
    pub departure: u32,
    pub arrival: u32,
    pub rides: u8,
    /// The capped journey fare; 0 for a walk-only chain.
    pub fare: f64,
}

impl Arrived {
    pub fn travel_time(&self) -> u32 {
        self.arrival - self.departure
    }
}

/// One row of the folded product: the winner under one cutoff.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrontierRow {
    pub cutoff: f64,
    pub travel_time: u32,
    pub fare: f64,
    pub rides: u8,
}

/// A label's fare side: `None` before the first boarding (walking is
/// free), the calculator's state after it. The kinds never compare —
/// a boarded label may integrate where a fresh one pays full, and a
/// fresh label's zero total is below any boarded one's.
#[derive(Debug, Clone, Copy, PartialEq)]
enum FareSide {
    Unboarded,
    Boarded(FareState),
}

#[derive(Debug, Clone, Copy)]
struct Entry {
    arrival: u32,
    rides: u8,
    side: FareSide,
}

/// The per-query dominance gates, computed once.
#[derive(Clone, Copy)]
struct Gates {
    discounts_monotone: bool,
    freshness_monotone: bool,
    /// The tables' largest single-integration saving: the transfer
    /// allowance bound of the conservative relation below.
    margin: f64,
    max_discounts: u32,
    exact: bool,
    /// Whether any fare type rides free after the first boarding.
    unlimited_types: bool,
    /// Bit `t` set: type `t` rides free after the first boarding.
    /// Types past 128 are conservatively treated as unlimited.
    unlimited_mask: u128,
}

fn gates_unlimited(gates: Gates, kind: u32) -> bool {
    kind >= 128 || (gates.unlimited_mask >> kind) & 1 == 1
}

/// A discount-exhausted state's window, route, and previous full fare
/// can never matter again — no integration remains to read them — so
/// they collapse to canonical sentinels and the state niches merge.
/// The unlimited-transfers branch still reads the previous type, which
/// stays.
fn canonicalize(fares: &RuleFares, state: FareState) -> FareState {
    if state.discounts < fares.max_discounted_transfers {
        return state;
    }
    FareState {
        previous_route: u32::MAX,
        previous_board: u32::MAX,
        previous_fare: f64::INFINITY,
        ..state
    }
}

/// A state's remaining discount budget.
fn remaining(entry: &Entry, max_discounts: u32) -> u32 {
    match &entry.side {
        FareSide::Unboarded => max_discounts,
        FareSide::Boarded(state) => max_discounts.saturating_sub(state.discounts),
    }
}

fn total(entry: &Entry) -> f64 {
    match &entry.side {
        FareSide::Unboarded => 0.0,
        FareSide::Boarded(state) => state.total,
    }
}

/// The R5-style transfer-allowance relation: `b`'s continuation can
/// undercut `a`'s by at most one margin per remaining discount (an
/// earlier label can replicate any suffix), so `a` dominates whenever
/// its total clears `b`'s by that bound. The bound is blind to
/// unlimited-transfer types — their free rides are not discounts and
/// are unbounded in count — so when `b`'s previous type rides free,
/// only an equal previous type may use this rule. Pricing `a`'s
/// suffix at full fare at worst also needs monotone tables — a
/// harmful pair's forced integration is a surcharge the savings-only
/// margin cannot bound — so non-monotone tariffs skip the rule.
/// Sound beside the exact relation; this is what keeps real-feed
/// bags small.
fn conservatively_dominates(a: &Entry, b: &Entry, gates: Gates) -> bool {
    if !gates.discounts_monotone {
        return false;
    }
    if let FareSide::Boarded(state) = &b.side {
        if gates.unlimited_types
            && gates_unlimited(gates, state.previous_type)
            && !matches!(&a.side, FareSide::Boarded(other)
                if other.previous_type == state.previous_type)
        {
            return false;
        }
    }
    a.arrival <= b.arrival
        && a.rides <= b.rides
        && total(a) <= total(b) - gates.margin * remaining(b, gates.max_discounts) as f64
}

fn dominates(a: &Entry, b: &Entry, gates: Gates) -> bool {
    if conservatively_dominates(a, b, gates) {
        return true;
    }
    if a.arrival > b.arrival || a.rides > b.rides {
        return false;
    }
    if !gates.exact {
        // The fast discipline: the state gates are assumed monotone,
        // so far fewer niches survive. Every kept row's fare is real,
        // but a cheaper journey can be missed where discounts and
        // windows interact — the r5r trade, stated in the docs.
        return match (&a.side, &b.side) {
            (FareSide::Unboarded, FareSide::Unboarded) => true,
            (FareSide::Boarded(a), FareSide::Boarded(b)) => {
                a.previous_type == b.previous_type
                    && a.total <= b.total
                    && a.discounts <= b.discounts
                    && a.previous_board >= b.previous_board
            }
            _ => false,
        };
    }
    match (&a.side, &b.side) {
        (FareSide::Unboarded, FareSide::Unboarded) => true,
        (FareSide::Boarded(a), FareSide::Boarded(b)) => {
            state_dominates(a, b, gates.discounts_monotone, gates.freshness_monotone)
        }
        _ => false,
    }
}

/// An unsorted nondominated bag, mirroring the McRAPTOR pattern.
#[derive(Default, Clone)]
struct Bag {
    entries: Vec<Entry>,
}

impl Bag {
    fn insert(&mut self, candidate: Entry, gates: Gates) -> bool {
        for entry in &self.entries {
            if dominates(entry, &candidate, gates) {
                return false;
            }
        }
        self.entries
            .retain(|entry| !dominates(&candidate, entry, gates));
        self.entries.push(candidate);
        true
    }
}

/// One boarded run along a pattern: the trip, its day stream, and the
/// label state after boarding. Entries on the same trip share every
/// future arrival, so only same-trip entries compare while riding.
#[derive(Clone, Copy)]
struct Riding {
    trip: TripIdx,
    day_offset: u32,
    rides: u8,
    state: FareState,
}

/// The per-origin frontier search: bags per stop, persisting across
/// the window's descending departure passes.
pub struct FareFrontierSearch<'a> {
    inputs: &'a FareFrontierInputs<'a>,
    request: &'a Request,
    gates: Gates,
    /// `state.total − margin × remaining discounts` (and the cap)
    /// must fit the top cutoff, or the label is dead.
    margin: f64,
    top_cutoff: f64,
    bags: Vec<Bag>,
    marked: Vec<StopIdx>,
    is_marked: Vec<bool>,
    /// Per stop: the destination slots it serves, with egress seconds
    /// — the fold runs at insert time, only over admitted entries.
    destination_map: Vec<Vec<(u32, u32)>>,
    /// Per destination slot × cutoff: the current winner — only the
    /// per-cutoff minimum ever matters, so no arrival set is kept.
    rows: Vec<Vec<Option<FrontierRow>>>,
}

impl<'a> FareFrontierSearch<'a> {
    pub fn new(
        inputs: &'a FareFrontierInputs<'a>,
        request: &'a Request,
        destinations: &[Vec<(StopIdx, u32)>],
    ) -> FareFrontierSearch<'a> {
        assert!(
            inputs.transfers.closed(),
            "the fare frontier relaxes the closed walking closure only"
        );
        let stop_count = inputs.timetable.stop_count() as usize;
        let discounts_monotone = inputs.fares.discounts_are_monotone();
        // Freshness is safe only when the budget covers every boarding
        // the query can make (and the tables are monotone at all).
        let freshness_monotone = discounts_monotone
            && inputs.fares.max_discounted_transfers > (request.max_transfers as u32).min(254);
        FareFrontierSearch {
            inputs,
            request,
            gates: Gates {
                discounts_monotone,
                freshness_monotone,
                margin: inputs.fares.max_discount_margin(),
                max_discounts: inputs.fares.max_discounted_transfers,
                exact: inputs.exact,
                unlimited_types: inputs.fares.unlimited_transfers.iter().any(|&on| on),
                unlimited_mask: inputs
                    .fares
                    .unlimited_transfers
                    .iter()
                    .take(128)
                    .enumerate()
                    .fold(
                        0u128,
                        |mask, (kind, &on)| {
                            if on {
                                mask | (1 << kind)
                            } else {
                                mask
                            }
                        },
                    ),
            },
            margin: inputs.fares.max_discount_margin(),
            top_cutoff: inputs.cutoffs.last().copied().unwrap_or(f64::INFINITY),
            bags: vec![Bag::default(); stop_count],
            marked: Vec::new(),
            is_marked: vec![false; stop_count],
            destination_map: {
                let mut map: Vec<Vec<(u32, u32)>> = vec![Vec::new(); stop_count];
                for (slot, egress) in destinations.iter().enumerate() {
                    for &(stop, seconds) in egress {
                        map[stop.0 as usize].push((slot as u32, seconds));
                    }
                }
                map
            },
            rows: vec![vec![None; inputs.cutoffs.len()]; destinations.len()],
        }
    }

    /// Folds one just-admitted entry into the per-cutoff winners of
    /// the destination slots its stop serves.
    fn collect(&mut self, stop: StopIdx, entry: &Entry, departure: u32) {
        if self.destination_map[stop.0 as usize].is_empty() {
            return;
        }
        let fare = match &entry.side {
            FareSide::Unboarded => 0.0,
            FareSide::Boarded(state) => self.inputs.fares.capped_total(state),
        };
        if fare.is_nan() || fare > self.top_cutoff + MONEY_EPSILON {
            return;
        }
        for index in 0..self.destination_map[stop.0 as usize].len() {
            let (slot, seconds) = self.destination_map[stop.0 as usize][index];
            let Some(arrival) = entry
                .arrival
                .checked_add(seconds)
                .filter(|&at| at != UNREACHED)
            else {
                continue;
            };
            if let Some(cap) = self.inputs.max_duration {
                if arrival.saturating_sub(departure) > cap {
                    continue;
                }
            }
            fold_candidate(
                &mut self.rows[slot as usize],
                self.inputs.cutoffs,
                Arrived {
                    departure,
                    arrival,
                    rides: entry.rides,
                    fare,
                },
            );
        }
    }

    /// Whether a boarded label can still fit the top cutoff: its total
    /// minus the largest remaining saving, against the cap.
    fn alive(&self, state: &FareState) -> bool {
        if state.total.is_nan() {
            return false;
        }
        let remaining = self
            .inputs
            .fares
            .max_discounted_transfers
            .saturating_sub(state.discounts) as f64;
        let floor = (state.total - self.margin * remaining).max(0.0);
        floor.min(self.inputs.fares.fare_cap) <= self.top_cutoff + MONEY_EPSILON
    }

    fn mark(&mut self, stop: StopIdx) {
        if !self.is_marked[stop.0 as usize] {
            self.is_marked[stop.0 as usize] = true;
            self.marked.push(stop);
        }
    }

    /// Runs one departure pass; passes must come in strictly
    /// decreasing departure order, exactly as in the other profile
    /// engines. Seeds never walk: the caller chooses the access
    /// semantics — the stop-to-stop product deliberately boards at
    /// the origin only, while a caller wanting walk-first boardings
    /// supplies closure-composed access rows.
    pub fn pass(&mut self, departure: u32) {
        // The pass's arrival horizon: an intermediate arrival past the
        // duration cap already exceeds it for every continuation, so
        // such labels prune everywhere below.
        let horizon = self
            .inputs
            .max_duration
            .and_then(|cap| departure.checked_add(cap))
            .unwrap_or(UNREACHED);
        // Access seeding: unboarded labels, fare zero.
        let mut seeds: Vec<(StopIdx, Entry)> = Vec::new();
        for &(stop, seconds) in &self.request.access {
            let Some(arrival) = departure
                .checked_add(seconds)
                .filter(|&at| at != UNREACHED && at <= horizon)
            else {
                continue;
            };
            seeds.push((
                stop,
                Entry {
                    arrival,
                    rides: 0,
                    side: FareSide::Unboarded,
                },
            ));
        }
        for (stop, entry) in seeds {
            if self.bags[stop.0 as usize].insert(entry, self.gates) {
                self.mark(stop);
                self.collect(stop, &entry, departure);
            }
        }

        // `rides` is a u8: the 255th transfer would be the 256th ride,
        // so the budget clamps at 254 exactly as McRAPTOR's does.
        let rounds = (self.request.max_transfers as usize).min(254) + 1;
        for round in 1..=rounds {
            let marked = std::mem::take(&mut self.marked);
            for &stop in &marked {
                self.is_marked[stop.0 as usize] = false;
            }
            if marked.is_empty() {
                break;
            }
            // Pattern queue: the minimum touched position per pattern.
            let timetable = self.inputs.timetable;
            let mut queued: Vec<(u32, u16)> = Vec::new();
            let mut queue_position: std::collections::HashMap<u32, u16> =
                std::collections::HashMap::new();
            for &stop in &marked {
                for pattern_stop in timetable.patterns_at_stop(stop) {
                    let slot = queue_position
                        .entry(pattern_stop.pattern.0)
                        .or_insert(u16::MAX);
                    if pattern_stop.position < *slot {
                        *slot = pattern_stop.position;
                    }
                }
            }
            queued.extend(queue_position);
            queued.sort_unstable();

            let mut writes: Vec<(StopIdx, Entry)> = Vec::new();
            for (pattern, start_position) in queued {
                self.scan_pattern(
                    crate::timetable::PatternIdx(pattern),
                    start_position as usize,
                    round,
                    horizon,
                    &mut writes,
                );
            }
            for (stop, entry) in writes {
                if self.bags[stop.0 as usize].insert(entry, self.gates) {
                    self.mark(stop);
                    self.collect(stop, &entry, departure);
                }
            }

            // Walking phase over the closed closure: fare unchanged.
            let sources: Vec<StopIdx> = self.marked.clone();
            let mut walks: Vec<(StopIdx, Entry)> = Vec::new();
            for &source in &sources {
                let candidates: Vec<Entry> = self.bags[source.0 as usize]
                    .entries
                    .iter()
                    .filter(|entry| entry.rides as usize == round)
                    .copied()
                    .collect();
                for edge in self.inputs.transfers.from_stop(source) {
                    for entry in &candidates {
                        let Some(arrival) = entry
                            .arrival
                            .checked_add(edge.duration)
                            .filter(|&at| at != UNREACHED && at <= horizon)
                        else {
                            continue;
                        };
                        walks.push((
                            edge.to,
                            Entry {
                                arrival,
                                rides: entry.rides,
                                side: entry.side,
                            },
                        ));
                    }
                }
            }
            for (stop, entry) in walks {
                if self.bags[stop.0 as usize].insert(entry, self.gates) {
                    self.mark(stop);
                    self.collect(stop, &entry, departure);
                }
            }
        }
    }

    /// Scans one pattern from `start_position`, carrying the boarded
    /// runs; alights collect into `writes`.
    fn scan_pattern(
        &mut self,
        pattern: crate::timetable::PatternIdx,
        start_position: usize,
        round: usize,
        horizon: u32,
        writes: &mut Vec<(StopIdx, Entry)>,
    ) {
        let timetable = self.inputs.timetable;
        let stops = timetable.pattern_stops(pattern);
        let route = timetable.pattern_route(pattern);
        let mut riding: Vec<Riding> = Vec::new();
        for position in start_position..stops.len() {
            let stop = stops[position];
            // Alight everything riding here.
            for run in &riding {
                let times = timetable.trip_stop_times(run.trip);
                let arrival = times[position].arrival.saturating_sub(run.day_offset);
                if arrival > horizon {
                    continue;
                }
                writes.push((
                    stop,
                    Entry {
                        arrival,
                        rides: run.rides,
                        side: FareSide::Boarded(run.state),
                    },
                ));
            }
            if position + 1 == stops.len() {
                continue;
            }
            // Board from the stop's bag: entries with fewer rides than
            // this round, each on its own earliest trip per stream.
            let boardable: Vec<Entry> = self.bags[stop.0 as usize]
                .entries
                .iter()
                .filter(|entry| (entry.rides as usize) < round)
                .copied()
                .collect();
            let trip_range = timetable.pattern_trip_range(pattern);
            for entry in boardable {
                for (stream, active) in [
                    &self.request.active_services,
                    &self.request.active_services_previous,
                ]
                .into_iter()
                .enumerate()
                {
                    if active.iter().all(|&on| !on) {
                        continue;
                    }
                    let day_offset = if stream == 0 { 0 } else { DAY_SECONDS };
                    let Some(threshold) = entry.arrival.checked_add(day_offset) else {
                        continue;
                    };
                    let Some(first) = earliest_active_trip(
                        timetable,
                        active,
                        self.request.exclusions.as_deref(),
                        pattern,
                        position,
                        threshold,
                    ) else {
                        continue;
                    };
                    // Boarding time is fare state: a later trip keeps
                    // a fresher window that can win a cutoff — but only
                    // while a discount remains to exploit it, so an
                    // exhausted label boards the earliest trip alone.
                    // The horizon cuts trips departing past the
                    // duration cap.
                    let explores_later = self.inputs.exact
                        && match &entry.side {
                            FareSide::Unboarded => self.inputs.fares.max_discounted_transfers > 0,
                            FareSide::Boarded(state) => {
                                state.discounts < self.inputs.fares.max_discounted_transfers
                            }
                        };
                    let last = if explores_later {
                        trip_range.end
                    } else {
                        first.0 + 1
                    };
                    for trip in first.0..last {
                        let trip = TripIdx(trip);
                        if timetable.trip_stop_times(trip)[position]
                            .departure
                            .saturating_sub(day_offset)
                            > horizon
                        {
                            break;
                        }
                        if !active
                            .get(timetable.trip_service(trip) as usize)
                            .copied()
                            .unwrap_or(false)
                        {
                            continue;
                        }
                        if let Some(excluded) = self.request.exclusions.as_deref() {
                            if excluded.excludes_trip(trip) {
                                continue;
                            }
                        }
                        self.board(entry, route, trip, position, day_offset, &mut riding);
                    }
                }
            }
        }
    }

    /// Boards one label onto one trip, inserting the run unless a
    /// same-trip run dominates it.
    #[allow(clippy::too_many_arguments)]
    fn board(
        &self,
        entry: Entry,
        route: u32,
        trip: TripIdx,
        position: usize,
        day_offset: u32,
        riding: &mut Vec<Riding>,
    ) {
        let timetable = self.inputs.timetable;
        let board_time = timetable.trip_stop_times(trip)[position]
            .departure
            .saturating_sub(day_offset);
        let state = match entry.side {
            FareSide::Unboarded => self.inputs.fares.board_first(route, board_time),
            FareSide::Boarded(state) => self.inputs.fares.board_next(&state, route, board_time),
        };
        // An unpriceable or cutoff-dead label is dropped: it can never
        // win any cell.
        let Some(state) = state
            .map(|state| canonicalize(self.inputs.fares, state))
            .filter(|state| self.alive(state))
        else {
            return;
        };
        let run = Riding {
            trip,
            day_offset,
            rides: entry.rides + 1,
            state,
        };
        // Same-trip runs share every future arrival, so only their
        // states compare; different trips both ride on.
        let mut admitted = true;
        riding.retain(|other| {
            if other.trip != run.trip || other.day_offset != run.day_offset {
                return true;
            }
            if state_dominates(
                &other.state,
                &run.state,
                self.gates.discounts_monotone,
                self.gates.freshness_monotone,
            ) && other.rides <= run.rides
            {
                admitted = false;
                return true;
            }
            !(state_dominates(
                &run.state,
                &other.state,
                self.gates.discounts_monotone,
                self.gates.freshness_monotone,
            ) && run.rides <= other.rides)
        });
        if admitted {
            riding.push(run);
        }
    }

    /// The folded product: per slot, the per-cutoff winners.
    pub fn into_rows(self) -> Vec<Vec<Option<FrontierRow>>> {
        self.rows
    }
}

/// Offers one candidate to every cutoff it fits, keeping the
/// lexicographic (travel time, fare, rides) minimum per cutoff.
pub fn fold_candidate(rows: &mut [Option<FrontierRow>], cutoffs: &[f64], candidate: Arrived) {
    for (slot, &cutoff) in cutoffs.iter().enumerate() {
        if candidate.fare > cutoff + MONEY_EPSILON {
            continue;
        }
        let row = FrontierRow {
            cutoff,
            travel_time: candidate.travel_time(),
            fare: candidate.fare,
            rides: candidate.rides,
        };
        let better = match &rows[slot] {
            None => true,
            Some(current) => {
                (row.travel_time, row.fare, row.rides)
                    < (current.travel_time, current.fare, current.rides)
            }
        };
        if better {
            rows[slot] = Some(row);
        }
    }
}

/// Injects the direct walking alternative into a cell's arrivals: a
/// zero-fare, zero-ride candidate at the walk's duration.
pub fn push_walk(rows: &mut [Option<FrontierRow>], cutoffs: &[f64], departure: u32, seconds: u32) {
    let Some(arrival) = departure.checked_add(seconds).filter(|&at| at != UNREACHED) else {
        return;
    };
    fold_candidate(
        rows,
        cutoffs,
        Arrived {
            departure,
            arrival,
            rides: 0,
            fare: 0.0,
        },
    );
}

/// Folds one cell's arrivals per cutoff: the minimum travel time with
/// fare within the cutoff, ties to the lexicographic minimum of
/// (travel time, fare, rides); `None` where nothing fits.
/// `fold_candidate` applies the same rule incrementally.
pub fn fold_cutoffs(arrivals: &[Arrived], cutoffs: &[f64]) -> Vec<Option<FrontierRow>> {
    cutoffs
        .iter()
        .map(|&cutoff| {
            let mut best: Option<FrontierRow> = None;
            for arrived in arrivals {
                if arrived.fare > cutoff + MONEY_EPSILON {
                    continue;
                }
                let row = FrontierRow {
                    cutoff,
                    travel_time: arrived.travel_time(),
                    fare: arrived.fare,
                    rides: arrived.rides,
                };
                let better = match &best {
                    None => true,
                    Some(current) => {
                        (row.travel_time, row.fare, row.rides)
                            < (current.travel_time, current.fare, current.rides)
                    }
                };
                if better {
                    best = Some(row);
                }
            }
            best
        })
        .collect()
}

#[cfg(test)]
#[path = "fare_frontier/tests.rs"]
mod tests;

/// The rasterised window: departures every `step` seconds within
/// `[departure, departure + window)`, descending — the same half-open
/// window the event enumeration searches.
pub fn sampled_departures(departure: u32, window: u32, step: u32) -> Vec<u32> {
    let step = step.max(1) as u64;
    let mut departures = Vec::new();
    let mut offset = 0u64;
    while offset < window as u64 {
        // Samples stop below the `UNREACHED` sentinel; the bindings
        // reject windows crossing the clock outright.
        let Some(at) = u32::try_from(departure as u64 + offset)
            .ok()
            .filter(|&at| at != UNREACHED)
        else {
            break;
        };
        departures.push(at);
        offset += step;
    }
    departures.reverse();
    departures
}

/// One origin's frontier over the departure window: one pass per
/// sampled or event departure, descending.
pub fn frontier(
    inputs: &FareFrontierInputs<'_>,
    request: &Request,
    destinations: &[Vec<(StopIdx, u32)>],
    window: u32,
) -> Vec<Vec<Option<FrontierRow>>> {
    let departures = match inputs.departure_step {
        Some(step) => sampled_departures(request.departure, window, step),
        None => crate::routers::raptor::departure_candidates(inputs.timetable, request, window),
    };
    let mut search = FareFrontierSearch::new(inputs, request, destinations);
    for &departure in &departures {
        search.pass(departure);
    }
    search.into_rows()
}
