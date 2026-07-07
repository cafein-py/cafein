//! ULTRA shortcut computation (bicriteria: arrival time, number of
//! trips), ported from the KIT reference `kit-algo/ULTRA`
//! (`Algorithms/RAPTOR/ULTRA/{ShortcutSearch,Builder}.h`).
//!
//! A **shortcut** is one intermediate transfer — a walk from the alight
//! stop of one trip to the board stop of the next — that some Pareto
//! optimal two-trip journey needs and no witness journey dominates.
//! Computed once per network (date-independent, over `DayView::universal`)
//! and used as the single-hop transfer set the routing engines relax:
//! a minimal, unrestricted-walking alternative to the transitive-closure
//! footpaths (see plans/ultra-plan.md).
//!
//! This first stage runs over cafein's **stop-to-stop transfer graph**
//! (option (b) in the plan): the three ULTRA searches are Dijkstras over
//! `Transfers`, with stops as the only vertices. Walks are therefore
//! stop-chains rather than exact road paths, so the set is a slightly
//! conservative superset of the road-graph result — exactness (running
//! the searches over the street graph) is a later stage.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use rayon::prelude::*;

use crate::tbtr::{earliest_boardable, DayView, ViewTrip};
use crate::timetable::{StopIdx, Timetable};
use crate::transfers::Transfers;

/// A stop is never reached at this time.
const NEVER: u32 = u32::MAX;

/// The ULTRA shortcut set of a network: the minimal intermediate
/// transfers over the transfer graph, computed in parallel over source
/// stops. `min_time`/`max_time` bound the source-departure window
/// (`0`/`NEVER - 1` for the whole day). Date-independent — build over
/// `DayView::universal`. Deduplicated (one shortcut per origin→destination
/// pair; equal walk times by construction).
pub fn compute_shortcuts(
    view: &DayView,
    timetable: &Timetable,
    transfers: &Transfers,
    min_time: u32,
    max_time: u32,
) -> Vec<Shortcut> {
    let stop_count = timetable.stop_count();
    let station = stations(transfers, stop_count);
    let mut all: Vec<Shortcut> = (0..stop_count)
        .into_par_iter()
        .filter(|&stop| station[stop as usize].0 == stop)
        .map_init(
            || Search::new(view, timetable, transfers, &station, stop_count),
            |search, stop| search.run(StopIdx(stop), min_time, max_time),
        )
        .flatten()
        .collect();
    all.sort_unstable_by_key(|shortcut| (shortcut.origin, shortcut.destination));
    all.dedup_by_key(|shortcut| (shortcut.origin, shortcut.destination));
    all
}

/// One intermediate-transfer shortcut: walk `seconds` from `origin` (a
/// trip's alight stop) to `destination` (the next trip's board stop).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shortcut {
    pub origin: StopIdx,
    pub destination: StopIdx,
    pub seconds: u32,
}

/// One source-departure and the first-trip boardings catchable when
/// leaving the source at that time. Mirrors the reference's
/// `ConsolidatedDepartureLabel`.
#[derive(Debug, Clone)]
struct Departure {
    time: u32,
    /// The `(line, position)` route segments first-boardable at `time`.
    routes: Vec<(u32, u16)>,
}

/// The representative stop of every stop's station, a station being the
/// stops mutually reachable over zero-time transfers (coincident stops).
/// The representative is the least stop id in the component. Only the
/// representative of a station runs the shortcut search.
fn stations(transfers: &Transfers, stop_count: u32) -> Vec<StopIdx> {
    let mut representative: Vec<StopIdx> = (0..stop_count).map(StopIdx).collect();
    // Union-find over zero-duration transfer edges.
    fn find(rep: &mut [StopIdx], stop: u32) -> u32 {
        let mut root = stop;
        while rep[root as usize].0 != root {
            root = rep[root as usize].0;
        }
        let mut current = stop;
        while rep[current as usize].0 != root {
            let next = rep[current as usize].0;
            rep[current as usize] = StopIdx(root);
            current = next;
        }
        root
    }
    for from in 0..stop_count {
        for edge in transfers.from_stop(StopIdx(from)) {
            if edge.duration != 0 {
                continue;
            }
            let a = find(&mut representative, from);
            let b = find(&mut representative, edge.to.0);
            let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
            representative[hi as usize] = StopIdx(lo);
        }
    }
    // Point every stop at its component's least id.
    for stop in 0..stop_count {
        let root = find(&mut representative, stop);
        representative[stop as usize] = StopIdx(root);
    }
    representative
}

/// Dijkstra over the transfer graph from `source`, returning the walking
/// time to every stop (`NEVER` when unreachable). `source` itself is 0.
fn walk_from(transfers: &Transfers, source: StopIdx, stop_count: u32) -> Vec<u32> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let mut distance = vec![NEVER; stop_count as usize];
    distance[source.0 as usize] = 0;
    let mut heap: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::new();
    heap.push(Reverse((0, source.0)));
    while let Some(Reverse((time, stop))) = heap.pop() {
        if time > distance[stop as usize] {
            continue;
        }
        for edge in transfers.from_stop(StopIdx(stop)) {
            let next = time.saturating_add(edge.duration);
            if next < distance[edge.to.0 as usize] {
                distance[edge.to.0 as usize] = next;
                heap.push(Reverse((next, edge.to.0)));
            }
        }
    }
    distance
}

/// The distinct source-departure times in `[min_time, max_time]` and, for
/// each, the first-trip route segments it can catch — a profile of when
/// leaving the source lets a first trip be boarded, given the walk to its
/// boarding stop. Faithful to the reference's `collectDepartures`:
/// `direct` supplies the round-0 walk, `station` the source-station test.
fn collect_departures(
    view: &DayView,
    timetable: &Timetable,
    direct: &[u32],
    source_rep: StopIdx,
    station: &[StopIdx],
    min_time: u32,
    max_time: u32,
) -> Vec<Departure> {
    // (departure time, route segment or a source-station marker).
    let mut labels: Vec<(u32, Option<(u32, u16)>)> = Vec::new();
    for line in 0..view.line_count() {
        let pattern = view.line_pattern(line);
        let stops = timetable.pattern_stops(pattern);
        let offset = view.line_day_offset(line);
        // The cheapest round-0 walk seen along the line so far; a later
        // stop only matters if it is cheaper to walk to.
        let mut minimal_transfer = NEVER;
        for position in 0..stops.len().saturating_sub(1) {
            let walk = direct[stops[position].0 as usize];
            if walk == NEVER || walk > minimal_transfer {
                continue;
            }
            minimal_transfer = walk;
            for rank in view.line_trips(line) {
                let departure = view.stored_times(timetable, ViewTrip(rank))[position]
                    .departure
                    .saturating_sub(offset);
                if departure < minimal_transfer {
                    continue;
                }
                let source_departure = departure - minimal_transfer;
                if source_departure < min_time {
                    continue;
                }
                if source_departure > max_time {
                    break;
                }
                if station[stops[position].0 as usize] == source_rep {
                    labels.push((source_departure, None));
                }
                labels.push((source_departure, Some((line, position as u16))));
            }
        }
    }
    // Descending by time; at equal time the route segments come before
    // the marker (the reference orders the marker last, `noRouteId` being
    // maximal), so a marker delimits a distinct source departure and
    // claims the route segments seen since the previous marker.
    labels.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then(a.1.is_none().cmp(&b.1.is_none()))
            .then(a.1.cmp(&b.1))
    });
    let mut result: Vec<Departure> = vec![Departure {
        time: NEVER,
        routes: Vec::new(),
    }];
    for (time, segment) in labels {
        match segment {
            None => {
                if time == result.last().unwrap().time {
                    continue;
                }
                result.last_mut().unwrap().time = time;
                result.push(Departure {
                    time,
                    routes: Vec::new(),
                });
            }
            Some(route) => result.last_mut().unwrap().routes.push(route),
        }
    }
    result.pop();
    for departure in &mut result {
        departure.routes.sort_unstable();
        departure.routes.dedup();
    }
    result
}

/// A per-worker shortcut search: a two-trip profile RAPTOR from one
/// source stop over the transfer graph, following the reference's
/// `ShortcutSearch`. Each departure runs with fresh labels (no
/// cross-departure timestamp reuse), and the intermediate/final Dijkstras
/// run to completion over the bounded transfer graph (no explicit witness
/// limit — witnesses are never pruned, so no superfluous shortcuts arise
/// from that). Label vectors are reused across departures and sources.
struct Search<'a> {
    view: &'a DayView,
    timetable: &'a Timetable,
    transfers: &'a Transfers,
    station: &'a [StopIdx],
    stop_count: u32,
    source_rep: StopIdx,
    direct: Vec<u32>,
    /// Origin→destination pairs already emitted for this source, so a
    /// later departure neither re-emits nor re-candidates them.
    emitted: HashSet<(u32, u32)>,
    shortcuts: Vec<Shortcut>,
    // Per-departure labels (0/1/2 trips) and candidate parents.
    zero: Vec<u32>,
    one: Vec<u32>,
    two: Vec<u32>,
    /// For a one-trip label: the shortcut origin (the trip's alight stop)
    /// when the journey is a candidate, else `None` (a witness/walk).
    one_parent: Vec<Option<StopIdx>>,
    /// For a two-trip label reached by route: the shortcut destination
    /// (the trip's board stop) when it is a candidate, else `None`.
    two_route_parent: Vec<Option<StopIdx>>,
    updated_by_route: Vec<StopIdx>,
    updated_by_transfer: Vec<StopIdx>,
    /// Per shortcut destination, the final stops of the candidate
    /// journeys using it — cleared as witnesses dominate them.
    dest_candidates: HashMap<StopIdx, HashSet<StopIdx>>,
}

impl<'a> Search<'a> {
    fn new(
        view: &'a DayView,
        timetable: &'a Timetable,
        transfers: &'a Transfers,
        station: &'a [StopIdx],
        stop_count: u32,
    ) -> Search<'a> {
        let n = stop_count as usize;
        Search {
            view,
            timetable,
            transfers,
            station,
            stop_count,
            source_rep: StopIdx(0),
            direct: Vec::new(),
            emitted: HashSet::new(),
            shortcuts: Vec::new(),
            zero: vec![NEVER; n],
            one: vec![NEVER; n],
            two: vec![NEVER; n],
            one_parent: vec![None; n],
            two_route_parent: vec![None; n],
            updated_by_route: Vec::new(),
            updated_by_transfer: Vec::new(),
            dest_candidates: HashMap::new(),
        }
    }

    fn run(&mut self, source: StopIdx, min_time: u32, max_time: u32) -> Vec<Shortcut> {
        // Only a station's representative searches; its members share it.
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
        self.scan_routes(1);
        self.intermediate_dijkstra();
        self.scan_routes(2);
        self.final_dijkstra();
    }

    fn clear_labels(&mut self) {
        for value in self
            .zero
            .iter_mut()
            .chain(self.one.iter_mut())
            .chain(self.two.iter_mut())
        {
            *value = NEVER;
        }
        for parent in self
            .one_parent
            .iter_mut()
            .chain(self.two_route_parent.iter_mut())
        {
            *parent = None;
        }
        self.updated_by_route.clear();
        self.updated_by_transfer.clear();
        self.dest_candidates.clear();
    }

    /// Round-0 walk: seed every stop reachable from the source at
    /// `departure + walk`, valid for all three rounds; they board trip 1.
    fn relax_initial(&mut self, departure: u32) {
        for stop in 0..self.stop_count as usize {
            if self.direct[stop] == NEVER {
                continue;
            }
            let arrival = departure.saturating_add(self.direct[stop]);
            self.zero[stop] = arrival;
            self.one[stop] = arrival;
            self.two[stop] = arrival;
            self.updated_by_transfer.push(StopIdx(stop as u32));
        }
    }

    /// Scan the routes serving the transfer-updated stops, boarding the
    /// earliest catchable trip and riding onward. `round` is 1 or 2.
    fn scan_routes(&mut self, round: u8) {
        let (view, timetable, station, source_rep) =
            (self.view, self.timetable, self.station, self.source_rep);
        let boarding = std::mem::take(&mut self.updated_by_transfer);
        self.updated_by_route.clear();
        for board in boarding {
            let ready = if round == 1 {
                self.zero[board.0 as usize]
            } else {
                self.one[board.0 as usize]
            };
            if ready == NEVER {
                continue;
            }
            for served in timetable.patterns_at_stop(board) {
                let stops = timetable.pattern_stops(served.pattern);
                if served.position as usize + 1 >= stops.len() {
                    continue;
                }
                for line in view.lines_of_pattern(served.pattern).into_iter().flatten() {
                    let Some(trip) =
                        earliest_boardable(view, timetable, line, served.position, ready)
                    else {
                        continue;
                    };
                    let offset = view.line_day_offset(line);
                    let times = view.stored_times(timetable, trip);
                    for position in served.position as usize + 1..stops.len() {
                        let stop = stops[position];
                        let arrival = times[position].arrival.saturating_sub(offset);
                        if round == 1 {
                            if arrival < self.one[stop.0 as usize] {
                                self.one[stop.0 as usize] = arrival;
                                // Candidate iff the trip was boarded within
                                // the source station; its alight is the
                                // shortcut origin.
                                self.one_parent[stop.0 as usize] =
                                    (station[board.0 as usize] == source_rep).then_some(stop);
                                if arrival < self.two[stop.0 as usize] {
                                    self.two[stop.0 as usize] = arrival;
                                }
                                self.updated_by_route.push(stop);
                            }
                        } else if arrival < self.two[stop.0 as usize] {
                            self.two[stop.0 as usize] = arrival;
                            self.two_route_parent[stop.0 as usize] =
                                self.candidate_destination(board);
                            self.updated_by_route.push(stop);
                        }
                    }
                }
            }
        }
        dedup_stops(&mut self.updated_by_route);
    }

    /// Whether boarding trip 2 at `board` makes it a shortcut destination:
    /// `board` was reached via a candidate's intermediate transfer (a
    /// non-trivial walk from a distinct origin) not already a shortcut.
    fn candidate_destination(&self, board: StopIdx) -> Option<StopIdx> {
        match self.one_parent[board.0 as usize] {
            Some(origin) if origin != board && !self.emitted.contains(&(origin.0, board.0)) => {
                Some(board)
            }
            _ => None,
        }
    }

    /// The intermediate transfer: Dijkstra over the transfer graph from
    /// the one-trip arrivals, propagating the shortcut origin. Settled
    /// stops board trip 2.
    fn intermediate_dijkstra(&mut self) {
        let transfers = self.transfers;
        let seeds = std::mem::take(&mut self.updated_by_route);
        let mut heap: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::new();
        for stop in &seeds {
            heap.push(Reverse((self.one[stop.0 as usize], stop.0)));
        }
        self.updated_by_transfer.clear();
        while let Some(Reverse((time, stop))) = heap.pop() {
            if time > self.one[stop as usize] {
                continue;
            }
            self.updated_by_transfer.push(StopIdx(stop));
            for edge in transfers.from_stop(StopIdx(stop)) {
                let next = time.saturating_add(edge.duration);
                let to = edge.to.0 as usize;
                if next < self.one[to] {
                    self.one[to] = next;
                    self.one_parent[to] = self.one_parent[stop as usize];
                    if next < self.two[to] {
                        self.two[to] = next;
                    }
                    heap.push(Reverse((next, edge.to.0)));
                }
            }
        }
    }

    /// The final transfer: Dijkstra over the transfer graph from the
    /// two-trip arrivals. A candidate destination that settles before any
    /// witness dominates it yields a shortcut.
    fn final_dijkstra(&mut self) {
        let transfers = self.transfers;
        let seeds = std::mem::take(&mut self.updated_by_route);
        let mut heap: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::new();
        for stop in &seeds {
            heap.push(Reverse((self.two[stop.0 as usize], stop.0)));
            if let Some(route_parent) = self.two_route_parent[stop.0 as usize] {
                self.dest_candidates
                    .entry(route_parent)
                    .or_default()
                    .insert(*stop);
            }
        }
        while let Some(Reverse((time, stop))) = heap.pop() {
            if self.dest_candidates.is_empty() {
                break;
            }
            if time > self.two[stop as usize] {
                continue;
            }
            for edge in transfers.from_stop(StopIdx(stop)) {
                let next = time.saturating_add(edge.duration);
                let to = edge.to.0 as usize;
                if next < self.two[to] {
                    self.two[to] = next;
                    // A witness reached `to` no later: its candidate dies.
                    if let Some(route_parent) = self.two_route_parent[to].take() {
                        if let Some(set) = self.dest_candidates.get_mut(&route_parent) {
                            set.remove(&edge.to);
                            if set.is_empty() {
                                self.dest_candidates.remove(&route_parent);
                            }
                        }
                    }
                    heap.push(Reverse((next, edge.to.0)));
                }
            }
            // `stop` settled still a candidate destination's final stop:
            // no witness beat it, so the shortcut is optimal.
            if let Some(route_parent) = self.two_route_parent[stop as usize] {
                let origin = self.one_parent[route_parent.0 as usize]
                    .expect("a candidate destination has a shortcut origin");
                let seconds = self.one[route_parent.0 as usize] - self.one[origin.0 as usize];
                if self.emitted.insert((origin.0, route_parent.0)) {
                    self.shortcuts.push(Shortcut {
                        origin,
                        destination: route_parent,
                        seconds,
                    });
                }
                if let Some(set) = self.dest_candidates.remove(&route_parent) {
                    for obsolete in set {
                        self.two_route_parent[obsolete.0 as usize] = None;
                    }
                }
            }
        }
    }
}

/// Sorts and removes duplicate stops in place.
fn dedup_stops(stops: &mut Vec<StopIdx>) {
    stops.sort_unstable();
    stops.dedup();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timetable::TimetableBuilder;

    fn time(at: u32) -> crate::timetable::StopTime {
        crate::timetable::StopTime {
            arrival: at,
            departure: at,
        }
    }

    /// Line A rides 0→1→2 at 100/200/300; line B rides 2→3 at 250/400.
    fn two_lines() -> Timetable {
        let mut builder = TimetableBuilder::new(4);
        let a = builder
            .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 0)
            .unwrap();
        let b = builder.add_pattern(&[StopIdx(2), StopIdx(3)], 1).unwrap();
        builder
            .add_trip(a, vec![time(100), time(200), time(300)], 0, 0)
            .unwrap();
        builder
            .add_trip(b, vec![time(250), time(400)], 1, 0)
            .unwrap();
        builder.finish()
    }

    #[test]
    fn stations_group_zero_time_transfers() {
        // 0↔1 are coincident (0 s); 2 stands alone.
        let transfers = Transfers::from_edges(
            3,
            &[
                (StopIdx(0), StopIdx(1), 0, 0.0),
                (StopIdx(1), StopIdx(0), 0, 0.0),
            ],
        )
        .unwrap();
        let reps = stations(&transfers, 3);
        assert_eq!(reps[0], StopIdx(0));
        assert_eq!(reps[1], StopIdx(0), "stop 1 joins stop 0's station");
        assert_eq!(reps[2], StopIdx(2), "stop 2 is its own station");
    }

    #[test]
    fn walk_from_chains_transfers() {
        // 0→1 (30 s), 1→2 (40 s); the graph is symmetric.
        let transfers = Transfers::from_edges(
            3,
            &[
                (StopIdx(0), StopIdx(1), 30, 30.0),
                (StopIdx(1), StopIdx(0), 30, 30.0),
                (StopIdx(1), StopIdx(2), 40, 40.0),
                (StopIdx(2), StopIdx(1), 40, 40.0),
            ],
        )
        .unwrap();
        let walk = walk_from(&transfers, StopIdx(0), 3);
        assert_eq!(walk, vec![0, 30, 70]);
        let isolated = walk_from(&Transfers::empty(3), StopIdx(0), 3);
        assert_eq!(isolated, vec![0, NEVER, NEVER]);
    }

    #[test]
    fn departures_profile_the_boardable_trips() {
        let timetable = two_lines();
        let view = DayView::universal(&timetable);
        // Source is stop 0 with no walking: only trips boardable at 0.
        let direct = walk_from(&Transfers::empty(4), StopIdx(0), 4);
        let station = stations(&Transfers::empty(4), 4);
        let departures = collect_departures(
            &view,
            &timetable,
            &direct,
            StopIdx(0),
            &station,
            0,
            NEVER - 1,
        );
        // Line A departs stop 0 at 100; that is the one source departure,
        // and it can board line A at position 0.
        assert_eq!(departures.len(), 1, "{departures:?}");
        assert_eq!(departures[0].time, 100);
        assert!(departures[0]
            .routes
            .contains(&(view.line_of(ViewTrip(0)), 0)));
    }

    #[test]
    fn emits_the_needed_intermediate_transfer() {
        // Board line A at 0, alight 1 (t=100), walk 1→2 (50 s), board
        // line B at 2 (departs 200), alight 3. Reaching 3 needs the
        // 1→2 intermediate transfer and nothing else provides it, so it
        // is the one shortcut.
        let mut builder = TimetableBuilder::new(4);
        let a = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
        let b = builder.add_pattern(&[StopIdx(2), StopIdx(3)], 1).unwrap();
        builder.add_trip(a, vec![time(0), time(100)], 0, 0).unwrap();
        builder
            .add_trip(b, vec![time(200), time(300)], 1, 0)
            .unwrap();
        let timetable = builder.finish();
        let view = DayView::universal(&timetable);
        let transfers = Transfers::from_edges(
            4,
            &[
                (StopIdx(1), StopIdx(2), 50, 50.0),
                (StopIdx(2), StopIdx(1), 50, 50.0),
            ],
        )
        .unwrap();
        let shortcuts = compute_shortcuts(&view, &timetable, &transfers, 0, NEVER - 1);
        assert_eq!(
            shortcuts,
            vec![Shortcut {
                origin: StopIdx(1),
                destination: StopIdx(2),
                seconds: 50,
            }]
        );
    }

    #[test]
    fn a_faster_direct_walk_witnesses_away_the_shortcut() {
        // Same as above, but the source can also walk straight to stop 2
        // (30 s) and catch line B without ever riding line A — a witness
        // that dominates the ride-then-walk candidate, so no shortcut.
        let mut builder = TimetableBuilder::new(4);
        let a = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
        let b = builder.add_pattern(&[StopIdx(2), StopIdx(3)], 1).unwrap();
        builder.add_trip(a, vec![time(0), time(100)], 0, 0).unwrap();
        builder
            .add_trip(b, vec![time(200), time(300)], 1, 0)
            .unwrap();
        let timetable = builder.finish();
        let view = DayView::universal(&timetable);
        let transfers = Transfers::from_edges(
            4,
            &[
                (StopIdx(1), StopIdx(2), 50, 50.0),
                (StopIdx(2), StopIdx(1), 50, 50.0),
                (StopIdx(0), StopIdx(2), 30, 30.0),
                (StopIdx(2), StopIdx(0), 30, 30.0),
            ],
        )
        .unwrap();
        let shortcuts = compute_shortcuts(&view, &timetable, &transfers, 0, NEVER - 1);
        assert!(shortcuts.is_empty(), "{shortcuts:?}");
    }
}
