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

use crate::tbtr::DayView;
use crate::timetable::{StopIdx, Timetable};
use crate::transfers::Transfers;

/// A stop is never reached at this time.
const NEVER: u32 = u32::MAX;

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
                let departure = view.stored_times(timetable, crate::tbtr::ViewTrip(rank))[position]
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
            .contains(&(view.line_of(crate::tbtr::ViewTrip(0)), 0)));
    }
}
