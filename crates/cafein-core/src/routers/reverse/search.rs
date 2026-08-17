//! The reverse search: frontier bags of (latest departure, achieved
//! arrival) labels growing from the destination's egress toward the
//! origins, round by round. The labels carry no path chains — journeys
//! are materialized by replaying the forward engine over the elected
//! (departure, rides, arrival) tuples, so both query directions always
//! return identical journeys.

use super::*;
use crate::routers::raptor::Raptor;
use crate::routers::router::TransitRouter;
use crate::transfers::Transfers;

/// Sentinel for an unfixed achieved arrival (a pending, pre-transit
/// label) and for unreachable values in outputs.
pub(crate) const UNSET: u32 = u32::MAX;

const DAY_SECONDS: u32 = 86_400;

/// One stop's per-mark per-round winners, as `reverse_profile_states`
/// returns them.
pub type MarkWinners = Vec<Vec<(u16, u32, u32)>>;

#[derive(Debug, Clone, Copy)]
struct RLabel {
    stop: StopIdx,
    /// Latest feasible departure from the stop, absolute seconds.
    departure: u32,
    /// The chain's final arrival at the destination; `UNSET` while the
    /// label is pending (no transit leg has fixed it yet).
    achieved: u32,
    /// Pending labels only: the egress walk waiting behind the
    /// deadline-side ride. Walks never extend pending labels (the
    /// exact-transfer mirror), so this never accumulates.
    trailing: u32,
    /// Whether the suffix starts with a ride — the only forward-legal
    /// journey opening, and therefore the only kind that may serve as
    /// an origin-side candidate.
    ride_rooted: bool,
    /// The egress seed this chain serves — set at seeding, inherited
    /// unchanged through rides and walks, ignored by dominance and by
    /// every untagged consumer. Multi-destination queries read it to
    /// attribute each winner to its destination.
    seed: u32,
}

impl RLabel {
    fn fixed(&self) -> bool {
        self.achieved != UNSET
    }
}

/// The reusable reverse state: the per-round frontier bags.
pub struct ReverseState {
    /// `bags[round][stop]` holds the frontier labels.
    bags: Vec<Vec<Vec<RLabel>>>,
    /// Stops whose round-`k` bag changed (the next scan's queue).
    marked: Vec<StopIdx>,
    is_marked: Vec<bool>,
    /// Labels a ride inserted this round, for the transfer phase.
    ride_touched: Vec<RLabel>,
}

impl ReverseState {
    pub fn new() -> ReverseState {
        ReverseState {
            bags: Vec::new(),
            marked: Vec::new(),
            is_marked: Vec::new(),
            ride_touched: Vec::new(),
        }
    }
}

impl Default for ReverseState {
    fn default() -> Self {
        ReverseState::new()
    }
}

/// Whether `a` may prune `b` within one bag. Pending and fixed labels
/// never dominate each other (a pending label's eventual arrival is
/// unknown until a ride fixes it), and a walk-rooted label never
/// prunes a ride-rooted one: only ride-rooted labels can open a
/// journey at an origin, so a dominating walk root must not evict the
/// candidate — the eviction would silently erase the origin's direct
/// journeys wherever a sibling platform sees a later train.
fn dominates(a: &RLabel, b: &RLabel) -> bool {
    if b.ride_rooted && !a.ride_rooted {
        return false;
    }
    match (a.fixed(), b.fixed()) {
        (true, true) => a.departure >= b.departure && a.achieved <= b.achieved,
        (false, false) => a.departure >= b.departure && a.trailing <= b.trailing,
        _ => false,
    }
}

pub(crate) struct ReverseSearch<'a> {
    timetable: &'a Timetable,
    reversed: &'a ReversedTransfers,
    request: &'a Request,
    /// Tagged multi-destination runs isolate dominance per seed:
    /// labels of different destinations never prune each other, so
    /// every destination's own winner survives the shared frontier —
    /// the fan-out's per-destination answers stay exactly
    /// reproducible. Untagged runs keep the shared frontier.
    seed_isolated: bool,
    /// Optional owner map for tagged runs: `seed_of[i]` tags egress
    /// entry `i`. Several egress entries (a point destination's many
    /// street links) share one owner seed, so isolation groups by
    /// DESTINATION — one isolated frontier per link would blow the
    /// bags up quadratically. `None` tags each entry by its index.
    seed_of: Option<&'a [u32]>,
}

impl<'a> ReverseSearch<'a> {
    pub(crate) fn new(
        timetable: &'a Timetable,
        reversed: &'a ReversedTransfers,
        request: &'a Request,
    ) -> ReverseSearch<'a> {
        ReverseSearch {
            timetable,
            reversed,
            request,
            seed_isolated: false,
            seed_of: None,
        }
    }

    pub(crate) fn new_seed_isolated(
        timetable: &'a Timetable,
        reversed: &'a ReversedTransfers,
        request: &'a Request,
        seed_of: Option<&'a [u32]>,
    ) -> ReverseSearch<'a> {
        ReverseSearch {
            timetable,
            reversed,
            request,
            seed_isolated: true,
            seed_of,
        }
    }

    fn stop_excluded(&self, stop: StopIdx) -> bool {
        self.request
            .exclusions
            .as_deref()
            .is_some_and(|excluded| excluded.excludes_stop(stop))
    }

    /// Inserts `label` into `state.bags[round][stop]` under frontier
    /// dominance; returns whether it was admitted. Exact ties keep the
    /// incumbent: the choice is output-neutral, since the tuples are
    /// equal and journeys never come from the labels themselves.
    fn insert(&self, state: &mut ReverseState, round: usize, label: RLabel) -> bool {
        let stop = label.stop.0 as usize;
        let isolated = self.seed_isolated;
        if state.bags[round][stop].iter().any(|incumbent| {
            (!isolated || incumbent.seed == label.seed) && dominates(incumbent, &label)
        }) {
            return false;
        }
        state.bags[round][stop].retain(|incumbent| {
            (isolated && incumbent.seed != label.seed) || !dominates(&label, incumbent)
        });
        state.bags[round][stop].push(label);
        if !state.is_marked[stop] {
            state.is_marked[stop] = true;
            state.marked.push(label.stop);
        }
        true
    }

    /// Runs the rounds; afterwards `state.bags[k][stop]` holds the
    /// fixed frontier of `k`-ride continuations from each stop.
    pub(crate) fn run(&self, state: &mut ReverseState) {
        let deadline = self.request.departure;
        let rounds = self.request.max_transfers as usize + 2;
        let stops = self.timetable.stop_count() as usize;
        state.bags.clear();
        state.bags.resize_with(rounds, || vec![Vec::new(); stops]);
        state.marked.clear();
        state.is_marked.clear();
        state.is_marked.resize(stops, false);

        // Round 0: pending labels at the egress stops. Walks never
        // extend these (the exact-transfer mirror), so the trailing
        // walk is fixed here once.
        for (index, &(stop, walk)) in self.request.egress.iter().enumerate() {
            let seed = self.seed_of.map_or(index as u32, |owners| owners[index]);
            if self.stop_excluded(stop) {
                continue;
            }
            let Some(departure) = deadline.checked_sub(walk) else {
                continue;
            };
            self.insert(
                state,
                0,
                RLabel {
                    stop,
                    departure,
                    achieved: UNSET,
                    trailing: walk,
                    ride_rooted: false,
                    seed,
                },
            );
            // Forward journeys may end ride → transfer → egress (the
            // egress composes over walk-improved labels), so exactly
            // one incoming transfer extends each egress seed before
            // the deadline-side ride fixes the chain — independently
            // of the direct label's admission: a competing seed's
            // extension may dominate the direct label at this stop,
            // yet extending the dominator again would take a second,
            // forbidden transfer, so this seed's own extensions stay
            // necessary.
            for &(from, duration) in self.reversed.into_stop(stop) {
                if self.stop_excluded(from) {
                    continue;
                }
                let Some(extended) = departure.checked_sub(duration) else {
                    continue;
                };
                self.insert(
                    state,
                    0,
                    RLabel {
                        stop: from,
                        departure: extended,
                        achieved: UNSET,
                        trailing: walk + duration,
                        ride_rooted: false,
                        seed,
                    },
                );
            }
        }

        let has_previous = self
            .request
            .active_services_previous
            .iter()
            .any(|&active| active);
        for round in 1..rounds {
            let queue: Vec<StopIdx> = std::mem::take(&mut state.marked);
            for stop in &queue {
                state.is_marked[stop.0 as usize] = false;
            }
            if queue.is_empty() {
                break;
            }
            // The route queue: per pattern, the highest marked position
            // (everything upstream of it can improve).
            let mut patterns: std::collections::HashMap<PatternIdx, u16> =
                std::collections::HashMap::new();
            for &stop in &queue {
                for pattern_stop in self.timetable.patterns_at_stop(stop) {
                    let entry = patterns
                        .entry(pattern_stop.pattern)
                        .or_insert(pattern_stop.position);
                    if pattern_stop.position > *entry {
                        *entry = pattern_stop.position;
                    }
                }
            }
            state.ride_touched.clear();
            let mut queue: Vec<(PatternIdx, u16)> = patterns.into_iter().collect();
            queue.sort_unstable_by_key(|&(pattern, _)| pattern.0);
            for (pattern, max_position) in queue {
                self.scan_pattern(state, round, pattern, max_position, has_previous);
            }
            // The transfer phase: walks extend ride-fixed labels only,
            // along genuine incoming edges.
            let touched = std::mem::take(&mut state.ride_touched);
            for label in &touched {
                for &(from, duration) in self.reversed.into_stop(label.stop) {
                    if self.stop_excluded(from) {
                        continue;
                    }
                    let Some(departure) = label.departure.checked_sub(duration) else {
                        continue;
                    };
                    self.insert(
                        state,
                        round,
                        RLabel {
                            stop: from,
                            departure,
                            achieved: label.achieved,
                            trailing: 0,
                            ride_rooted: false,
                            seed: label.seed,
                        },
                    );
                }
            }
            state.ride_touched = touched;
        }
    }

    /// One pattern's reverse scan: positions from the highest marked
    /// down to the first, carrying the onboard continuations.
    fn scan_pattern(
        &self,
        state: &mut ReverseState,
        round: usize,
        pattern: PatternIdx,
        max_position: u16,
        has_previous: bool,
    ) {
        let stops = self.timetable.pattern_stops(pattern);
        // (trip, offset, achieved, alight_position, seed)
        let mut onboard: Vec<(TripIdx, u32, u32, u16, u32)> = Vec::new();
        for position in (0..=max_position as usize).rev() {
            let stop = stops[position];
            if self.stop_excluded(stop) {
                continue;
            }
            // 1) Board here: contribute every onboard continuation.
            let contributions: Vec<RLabel> = onboard
                .iter()
                .filter(|&&(_, _, _, alight, _)| (alight as usize) > position)
                .filter_map(|&(trip, offset, achieved, _, seed)| {
                    let departure = self.timetable.trip_stop_times(trip)[position]
                        .departure
                        .checked_sub(offset)?;
                    Some(RLabel {
                        stop,
                        departure,
                        achieved,
                        trailing: 0,
                        ride_rooted: true,
                        seed,
                    })
                })
                .collect();
            for label in contributions {
                if self.insert(state, round, label) {
                    state.ride_touched.push(label);
                }
            }
            // 2) Alight here: merge the previous round's labels into
            // the onboard set.
            let previous: Vec<RLabel> = state.bags[round - 1][stop.0 as usize].clone();
            for label in previous {
                for (active, offset) in [
                    (&self.request.active_services, 0u32),
                    (&self.request.active_services_previous, DAY_SECONDS),
                ] {
                    if offset != 0 && !has_previous {
                        continue;
                    }
                    // A previous-day trip stored at `t` arrives at
                    // `t − DAY_SECONDS`; it is alightable when
                    // `t ≤ departure + DAY_SECONDS`.
                    let Some(reached) = label.departure.checked_add(offset) else {
                        continue;
                    };
                    let Some(latest) = crate::routers::raptor::latest_active_trip(
                        self.timetable,
                        active,
                        self.request.exclusions.as_deref(),
                        pattern,
                        position,
                        reached,
                    ) else {
                        continue;
                    };
                    // A fixed label's achieved arrival is independent of
                    // the trip, so the latest boardable one dominates
                    // every earlier trip. A pending label's achieved is
                    // SET by this deadline-side boarding: the boardable
                    // trips form a staircase of nondominated (upstream
                    // departure, achieved) pairs, and every step must
                    // enter the onboard set — pending labels exist only
                    // at the egress stops, so the enumeration is tiny.
                    let range = self.timetable.pattern_trip_range(pattern);
                    let earliest = if label.fixed() { latest.0 } else { range.start };
                    for raw in (earliest..=latest.0).rev() {
                        let trip = TripIdx(raw);
                        let usable = active
                            .get(self.timetable.trip_service(trip) as usize)
                            .copied()
                            .unwrap_or(false)
                            && !self
                                .request
                                .exclusions
                                .as_deref()
                                .is_some_and(|excluded| excluded.excludes_trip(trip));
                        if !usable {
                            continue;
                        }
                        // A previous-day trip whose stored time sits
                        // below the day offset ran yesterday's daytime:
                        // it has no query-day clock position at all.
                        let Some(arrival) = self.timetable.trip_stop_times(trip)[position]
                            .arrival
                            .checked_sub(offset)
                        else {
                            continue;
                        };
                        let achieved = if label.fixed() {
                            label.achieved
                        } else {
                            arrival.saturating_add(label.trailing)
                        };
                        // Light dedupe: one onboard entry per (trip,
                        // offset), keeping the earliest achieved; the
                        // stop bags' dominance is authoritative. A
                        // seed-isolated run dedupes per seed too —
                        // merging across seeds would silently discard
                        // a destination that shares the trip.
                        match onboard.iter_mut().find(|entry| {
                            entry.0 == trip
                                && entry.1 == offset
                                && (!self.seed_isolated || entry.4 == label.seed)
                        }) {
                            Some(entry) => {
                                if achieved < entry.2 {
                                    entry.2 = achieved;
                                    entry.3 = position as u16;
                                    entry.4 = label.seed;
                                }
                            }
                            None => {
                                onboard.push((trip, offset, achieved, position as u16, label.seed))
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The latest-departure journeys from every access stop to the
/// destination described by `request.egress`, for the deadline in
/// `request.departure` — the complete-journey order's nondominated
/// set: (latest departure, fewest rides), earliest achieved arrival
/// breaking ties. Legs come from replaying the forward engine at each
/// elected departure, so an arrive-by journey is identical to the
/// depart-at answer for the departure it discovers.
pub fn reverse_route(
    timetable: &Timetable,
    transfers: &Transfers,
    reversed: &ReversedTransfers,
    request: &Request,
) -> Vec<Journey> {
    let search = ReverseSearch::new(timetable, reversed, request);
    let mut state = ReverseState::new();
    search.run(&mut state);

    let selected = mark_candidates(&state, request, request.departure);
    replay(timetable, transfers, request, &selected)
}

/// One deadline mark's complete-journey candidates: every fixed
/// ride-rooted label at an access stop with `achieved ≤ mark`,
/// composed with its access walk — forward journeys always begin
/// access → ride, so only ride-rooted labels are legal here — then
/// the complete-journey order's nondominated prefix.
fn mark_candidates(state: &ReverseState, request: &Request, mark: u32) -> Vec<(u32, usize, u32)> {
    let mut candidates: Vec<(u32, usize, u32)> = Vec::new();
    for &(stop, walk) in &request.access {
        for (round, bags) in state.bags.iter().enumerate() {
            for label in &bags[stop.0 as usize] {
                if !label.fixed() || !label.ride_rooted || label.achieved > mark {
                    continue;
                }
                let Some(composed) = label.departure.checked_sub(walk) else {
                    continue;
                };
                candidates.push((composed, round, label.achieved));
            }
        }
    }
    // The complete-journey order: departure desc, rides asc, achieved
    // asc; then the nondominated prefix over (departure, rides) with
    // the achieved tie-break.
    candidates.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let mut selected = Vec::new();
    let mut best_rides = usize::MAX;
    for (departure, rides, achieved) in candidates {
        let dominated = best_rides <= rides
            && selected
                .iter()
                .any(|&(d, r, _)| d >= departure && r <= rides && (d > departure || r < rides));
        let duplicate = selected
            .iter()
            .any(|&(d, r, _)| d == departure && r == rides);
        if dominated || duplicate {
            continue;
        }
        best_rides = best_rides.min(rides);
        selected.push((departure, rides, achieved));
    }
    selected
}

/// Materializes candidate tuples through the forward engine, latest
/// departure first — the arc's replay guarantee.
fn replay(
    timetable: &Timetable,
    transfers: &Transfers,
    request: &Request,
    selected: &[(u32, usize, u32)],
) -> Vec<Journey> {
    let mut journeys = Vec::new();
    for &(departure, rides, achieved) in selected {
        let forward = Request {
            departure,
            ..request.clone()
        };
        let journey = Raptor
            .route(timetable, transfers, &forward)
            .into_iter()
            .find(|journey| journey.arrival == achieved && journey.rides() == rides)
            .expect("the forward replay contains the elected journey");
        debug_assert_eq!(journey.departure, departure);
        journeys.push(journey);
    }
    journeys
}

/// The deadline profile's journeys over ascending minute `marks`: one
/// reverse run at the final mark, each mark's complete-journey Pareto
/// candidates collected, the union deduplicated on exact tuples, and
/// every distinct journey materialized by forward replay — sorted by
/// (departure desc, rides asc, achieved asc), the union's members
/// each being some mark's answer (never globally re-filtered: a later
/// mark's later-departing winner does not erase an earlier mark's).
/// `walk` is a direct walking alternative's seconds: each mark's
/// Pareto selection then competes against the walk placed to arrive
/// exactly at that mark (zero rides), so the union is exactly the
/// union of the marks' single-deadline answers — a duration test
/// alone would keep journeys a mark's own walk dominates. Walk
/// winners come back as the second element, (departure, arrival)
/// placements the caller renders itself.
pub fn reverse_route_profile(
    timetable: &Timetable,
    transfers: &Transfers,
    reversed: &ReversedTransfers,
    request: &Request,
    marks: &[u32],
    walk: Option<u32>,
) -> (Vec<Journey>, Vec<(u32, u32)>) {
    let search = ReverseSearch::new(timetable, reversed, request);
    let mut state = ReverseState::new();
    search.run(&mut state);
    // Each access stop's frontier evaluates once through the rolling
    // per-mark evaluator; a mark's candidate pool is then the tiny
    // per-round winner set per access stop, never a rescan of every
    // label per mark.
    let profiles: Vec<(u32, MarkWinners)> = request
        .access
        .iter()
        .map(|&(stop, walk_seconds)| {
            let states: Vec<(u16, u32, u32)> = state
                .bags
                .iter()
                .enumerate()
                .flat_map(|(round, bags)| {
                    bags[stop.0 as usize]
                        .iter()
                        .filter(|label| label.fixed() && label.ride_rooted)
                        .map(move |label| (round as u16, label.departure, label.achieved))
                })
                .collect();
            (walk_seconds, reverse_profile_states(&states, marks))
        })
        .collect();
    let mut union: Vec<(u32, usize, u32)> = Vec::new();
    for (at, &mark) in marks.iter().enumerate() {
        let mut candidates: Vec<(u32, usize, u32)> = Vec::new();
        for (walk_seconds, profile) in &profiles {
            for &(round, departure, achieved) in &profile[at] {
                if let Some(composed) = departure.checked_sub(*walk_seconds) {
                    candidates.push((composed, round as usize, achieved));
                }
            }
        }
        if let Some(seconds) = walk {
            if let Some(placed) = mark.checked_sub(seconds) {
                candidates.push((placed, 0, mark));
            }
        }
        candidates.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        let mut kept: Vec<(u32, usize, u32)> = Vec::new();
        let mut best_rides = usize::MAX;
        for (departure, rides, achieved) in candidates {
            let dominated = best_rides <= rides
                && kept
                    .iter()
                    .any(|&(d, r, _)| d >= departure && r <= rides && (d > departure || r < rides));
            let duplicate = kept.iter().any(|&(d, r, _)| d == departure && r == rides);
            if dominated || duplicate {
                continue;
            }
            best_rides = best_rides.min(rides);
            kept.push((departure, rides, achieved));
        }
        for candidate in kept {
            if !union.contains(&candidate) {
                union.push(candidate);
            }
        }
    }
    union.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let walks: Vec<(u32, u32)> = union
        .iter()
        .filter(|&&(_, rides, _)| rides == 0)
        .map(|&(departure, _, achieved)| (departure, achieved))
        .collect();
    let transit: Vec<(u32, usize, u32)> = union
        .into_iter()
        .filter(|&(_, rides, _)| rides > 0)
        .collect();
    (replay(timetable, transfers, request, &transit), walks)
}

/// Per-origin per-round fixed frontier states: `result[stop]` lists
/// `(round, latest departure, achieved arrival)` — ride counts come
/// with the rounds, so consumers composing access walks apply the
/// complete-journey order after composition and only then derive
/// durations.
pub fn reverse_one_to_all(
    timetable: &Timetable,
    reversed: &ReversedTransfers,
    request: &Request,
) -> Vec<Vec<(u16, u32, u32)>> {
    let search = ReverseSearch::new(timetable, reversed, request);
    let mut state = ReverseState::new();
    search.run(&mut state);
    collect_states(timetable, &state)
}

/// `reverse_one_to_all` for many requests, fanned out in parallel with
/// per-worker state reuse — one request per arrive-by matrix
/// destination. Each finished run folds into `fold`'s compact value
/// (typically one dense column over the requested origins) inside the
/// worker, so only one run's per-stop frontiers are ever held per
/// worker — never O(destinations × stops) retained nested state.
pub fn reverse_one_to_all_fold<T, F>(
    timetable: &Timetable,
    reversed: &ReversedTransfers,
    requests: &[Request],
    fold: F,
) -> Vec<T>
where
    T: Send,
    F: Fn(usize, &[Vec<(u16, u32, u32)>]) -> T + Sync,
{
    use rayon::prelude::*;
    requests
        .par_iter()
        .enumerate()
        .map_init(ReverseState::new, |state, (index, request)| {
            let search = ReverseSearch::new(timetable, reversed, request);
            search.run(state);
            fold(index, &collect_states(timetable, state))
        })
        .collect()
}

/// The deadline profile of one stop's frontier states: for each
/// ascending deadline mark, each round's winner — the latest-departure
/// state among that round's states with `achieved ≤ mark` (earliest
/// achieved breaking exact departure ties). One `T2` run's frontier
/// serves every mark because dominance can only prune a state in
/// favour of one that serves the same mark at least as well; the
/// consumers compose access walks and apply the complete-journey
/// order across rounds after composition, exactly as on the
/// single-deadline surfaces. Sorting each round's states by achieved
/// and rolling a best pointer over the ascending marks makes this
/// O(states log states + rounds × marks), never O(states × marks).
pub fn reverse_profile_states(
    states: &[(u16, u32, u32)],
    marks: &[u32],
) -> Vec<Vec<(u16, u32, u32)>> {
    // One sort orders by (round, achieved); grouping consecutive
    // rounds is then a single pass — O(states log states) setup with
    // no per-state scans.
    let mut sorted: Vec<(u16, u32, u32)> = states
        .iter()
        .map(|&(round, departure, achieved)| (round, achieved, departure))
        .collect();
    sorted.sort_unstable();
    let mut per_round: Vec<(u16, Vec<(u32, u32)>)> = Vec::new();
    for (round, achieved, departure) in sorted {
        match per_round.last_mut() {
            Some((held, entries)) if *held == round => entries.push((achieved, departure)),
            _ => per_round.push((round, vec![(achieved, departure)])),
        }
    }
    let mut cursors = vec![0usize; per_round.len()];
    let mut bests: Vec<Option<(u32, u32)>> = vec![None; per_round.len()];
    let mut result = Vec::with_capacity(marks.len());
    for &mark in marks {
        let mut winners = Vec::new();
        for (slot, (round, entries)) in per_round.iter_mut().enumerate() {
            while cursors[slot] < entries.len() && entries[cursors[slot]].0 <= mark {
                let (achieved, departure) = entries[cursors[slot]];
                let wins = match bests[slot] {
                    None => true,
                    Some((held_departure, held_achieved)) => {
                        departure > held_departure
                            || (departure == held_departure && achieved < held_achieved)
                    }
                };
                if wins {
                    bests[slot] = Some((departure, achieved));
                }
                cursors[slot] += 1;
            }
            if let Some((departure, achieved)) = bests[slot] {
                winners.push((*round, departure, achieved));
            }
        }
        result.push(winners);
    }
    result
}

/// The per-stop fixed ride-rooted frontier states of a finished run:
/// these are journey starts for the composing consumer.
fn collect_states(timetable: &Timetable, state: &ReverseState) -> Vec<Vec<(u16, u32, u32)>> {
    let stops = timetable.stop_count() as usize;
    let mut result = vec![Vec::new(); stops];
    for (round, bags) in state.bags.iter().enumerate() {
        for (stop, bag) in bags.iter().enumerate() {
            for label in bag {
                if label.fixed() && label.ride_rooted {
                    result[stop].push((round as u16, label.departure, label.achieved));
                }
            }
        }
    }
    result
}

/// `collect_states` with each state's egress-seed tag — the one
/// consumer that attributes winners to destinations reads these; the
/// untagged surfaces stay byte-identical.
fn collect_tagged(timetable: &Timetable, state: &ReverseState) -> Vec<Vec<(u16, u32, u32, u32)>> {
    let stops = timetable.stop_count() as usize;
    let mut result = vec![Vec::new(); stops];
    for (round, bags) in state.bags.iter().enumerate() {
        for (stop, bag) in bags.iter().enumerate() {
            for label in bag {
                if label.fixed() && label.ride_rooted {
                    result[stop].push((round as u16, label.departure, label.achieved, label.seed));
                }
            }
        }
    }
    result
}

/// `reverse_one_to_all` with seed tags: per stop the fixed ride-rooted
/// frontier states as (round, departure, achieved, egress seed index).
pub fn reverse_one_to_all_tagged(
    timetable: &Timetable,
    reversed: &ReversedTransfers,
    request: &Request,
    seed_of: Option<&[u32]>,
) -> Vec<Vec<(u16, u32, u32, u32)>> {
    let search = ReverseSearch::new_seed_isolated(timetable, reversed, request, seed_of);
    let mut state = ReverseState::new();
    search.run(&mut state);
    collect_tagged(timetable, &state)
}
