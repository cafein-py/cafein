use super::*;
use crate::journey::Leg;
use crate::routers::raptor::Raptor;
use crate::routers::router::Exclusions;
use crate::routers::router::TransitRouter;
use crate::timetable::{StopTime, TimetableBuilder};
use crate::transfers::Transfers;

fn time(at: u32) -> StopTime {
    StopTime {
        arrival: at,
        departure: at,
    }
}

fn request(from: StopIdx, to: StopIdx, deadline: u32, services: usize) -> Request {
    Request {
        departure: deadline,
        access: vec![(from, 0)],
        egress: vec![(to, 0)],
        active_services: vec![true; services],
        active_services_previous: Vec::new(),
        max_transfers: 3,
        exclusions: None,
    }
}

/// The forward network of the RAPTOR tests: pattern A rides 0→1→2,
/// B rides 1→3, a 50-second footpath joins 2→4, and C rides 4→3.
fn network() -> (Timetable, Transfers) {
    let mut builder = TimetableBuilder::new(5);
    let a = builder
        .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 0)
        .unwrap();
    let b = builder.add_pattern(&[StopIdx(1), StopIdx(3)], 1).unwrap();
    let c = builder.add_pattern(&[StopIdx(4), StopIdx(3)], 2).unwrap();
    builder
        .add_trip(a, vec![time(100), time(200), time(300)], 0, 0)
        .unwrap();
    builder
        .add_trip(a, vec![time(700), time(800), time(900)], 1, 0)
        .unwrap();
    builder
        .add_trip(b, vec![time(250), time(400)], 2, 0)
        .unwrap();
    builder
        .add_trip(c, vec![time(400), time(1000)], 3, 0)
        .unwrap();
    let timetable = builder.finish();
    let transfers = Transfers::from_edges(5, &[(StopIdx(2), StopIdx(4), 50, 50.0)]).unwrap();
    (timetable, transfers)
}

#[test]
fn latest_departure_journeys_invert_the_forward_profile() {
    // The exact oracle: sweep every second as a forward departure,
    // keep journeys arriving by the deadline, and take the
    // complete-journey order's nondominated set. The reverse answer
    // must match it exactly.
    let (timetable, transfers) = network();
    let reversed = ReversedTransfers::build(&transfers);
    for deadline in [399, 400, 450, 999, 1000, 1500] {
        let mut oracle: Vec<(u32, usize, u32)> = Vec::new();
        for departure in 0..1200 {
            let mut forward = request(StopIdx(0), StopIdx(3), 0, 4);
            forward.departure = departure;
            for journey in Raptor.route(&timetable, &transfers, &forward) {
                if journey.arrival <= deadline {
                    oracle.push((departure, journey.rides(), journey.arrival));
                }
            }
        }
        oracle.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        let mut expected: Vec<(u32, usize, u32)> = Vec::new();
        for candidate in oracle {
            let dominated = expected.iter().any(|held| {
                held.0 >= candidate.0
                    && held.1 <= candidate.1
                    && (held.0 > candidate.0 || held.1 < candidate.1)
            });
            let duplicate = expected
                .iter()
                .any(|held| held.0 == candidate.0 && held.1 == candidate.1);
            if !dominated && !duplicate {
                expected.push(candidate);
            }
        }
        let journeys = reverse_route(
            &timetable,
            &transfers,
            &reversed,
            &request(StopIdx(0), StopIdx(3), deadline, 4),
        );
        let got: Vec<(u32, usize, u32)> = journeys
            .iter()
            .map(|journey| (journey.departure, journey.rides(), journey.arrival))
            .collect();
        assert_eq!(got, expected, "deadline {deadline}");
    }
}

#[test]
fn equal_departures_keep_the_earlier_arrival() {
    // Two single-ride patterns leave stop 0 at the same second; the
    // slower one arrives 200 seconds later. Equal departure and rides:
    // the earlier arrival must win.
    let mut builder = TimetableBuilder::new(3);
    let fast = builder.add_pattern(&[StopIdx(0), StopIdx(2)], 0).unwrap();
    let slow = builder.add_pattern(&[StopIdx(0), StopIdx(2)], 1).unwrap();
    builder
        .add_trip(fast, vec![time(100), time(300)], 0, 0)
        .unwrap();
    builder
        .add_trip(slow, vec![time(100), time(500)], 1, 0)
        .unwrap();
    let timetable = builder.finish();
    let transfers = Transfers::empty(3);
    let reversed = ReversedTransfers::build(&transfers);
    let journeys = reverse_route(
        &timetable,
        &transfers,
        &reversed,
        &request(StopIdx(0), StopIdx(2), 600, 2),
    );
    assert_eq!(journeys.len(), 1);
    assert_eq!(journeys[0].departure, 100);
    assert_eq!(journeys[0].arrival, 300);
}

#[test]
fn a_shared_prefix_resurrects_the_earlier_arrival() {
    // Continuations from interchange stop 1: X leaves at 200 arriving
    // 300, Y leaves at 250 arriving 500. A single-best label at stop 1
    // would keep only Y (later departure); the shared upstream trip
    // equalizes both complete journeys' departures, where X's earlier
    // arrival must win — the frontier keeps both.
    let mut builder = TimetableBuilder::new(4);
    let upstream = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let x = builder.add_pattern(&[StopIdx(1), StopIdx(3)], 1).unwrap();
    let y = builder.add_pattern(&[StopIdx(1), StopIdx(3)], 2).unwrap();
    builder
        .add_trip(upstream, vec![time(50), time(100)], 0, 0)
        .unwrap();
    builder
        .add_trip(x, vec![time(200), time(300)], 1, 0)
        .unwrap();
    builder
        .add_trip(y, vec![time(250), time(500)], 2, 0)
        .unwrap();
    let timetable = builder.finish();
    let transfers = Transfers::empty(4);
    let reversed = ReversedTransfers::build(&transfers);
    let journeys = reverse_route(
        &timetable,
        &transfers,
        &reversed,
        &request(StopIdx(0), StopIdx(3), 600, 3),
    );
    // One complete journey: departs 50 with 2 rides; its arrival must
    // be X's 300, not Y's 500.
    assert_eq!(journeys.len(), 1);
    assert_eq!(journeys[0].departure, 50);
    assert_eq!(journeys[0].arrival, 300);
    assert_eq!(journeys[0].rides(), 2);
}

#[test]
fn a_far_deadline_reports_the_journeys_own_arrival() {
    let (timetable, transfers) = network();
    let reversed = ReversedTransfers::build(&transfers);
    let journeys = reverse_route(
        &timetable,
        &transfers,
        &reversed,
        &request(StopIdx(0), StopIdx(3), 50_000, 4),
    );
    assert!(!journeys.is_empty());
    for journey in &journeys {
        assert!(journey.arrival <= 1000, "own arrival, never the deadline");
    }
}

#[test]
fn reverse_relaxation_walks_incoming_edges_only() {
    // A ride 0→1, a directed footpath, a ride 3→2. With the edge
    // 1→3 the journey exists — found through stop 3's incoming edges.
    // With the edge stored 3→1 instead, the walk cannot be taken
    // toward 3, and an implementation reading outgoing edges would
    // wrongly find it.
    let build = || {
        let mut builder = TimetableBuilder::new(4);
        let a = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
        let b = builder.add_pattern(&[StopIdx(3), StopIdx(2)], 1).unwrap();
        builder
            .add_trip(a, vec![time(100), time(200)], 0, 0)
            .unwrap();
        builder
            .add_trip(b, vec![time(300), time(400)], 1, 0)
            .unwrap();
        builder.finish()
    };
    let timetable = build();
    let forward_edge = Transfers::from_edges(4, &[(StopIdx(1), StopIdx(3), 100, 100.0)]).unwrap();
    let reversed = ReversedTransfers::build(&forward_edge);
    let journeys = reverse_route(
        &timetable,
        &forward_edge,
        &reversed,
        &request(StopIdx(0), StopIdx(2), 500, 2),
    );
    assert_eq!(journeys.len(), 1);
    assert_eq!(journeys[0].departure, 100);
    assert_eq!(journeys[0].arrival, 400);

    let timetable = build();
    let wrong_way = Transfers::from_edges(4, &[(StopIdx(3), StopIdx(1), 100, 100.0)]).unwrap();
    let reversed = ReversedTransfers::build(&wrong_way);
    let journeys = reverse_route(
        &timetable,
        &wrong_way,
        &reversed,
        &request(StopIdx(0), StopIdx(2), 500, 2),
    );
    assert!(journeys.is_empty(), "the walk only runs 3→1, never 1→3");
}

#[test]
fn previous_day_services_ride_the_offset_stream() {
    // The only trip belongs to yesterday's service, stored a day
    // later: effective times 100→200 on the query day.
    let mut builder = TimetableBuilder::new(2);
    let a = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    builder
        .add_trip(a, vec![time(86_500), time(86_600)], 0, 0)
        .unwrap();
    let timetable = builder.finish();
    let transfers = Transfers::empty(2);
    let reversed = ReversedTransfers::build(&transfers);
    let mut query = request(StopIdx(0), StopIdx(1), 300, 1);
    query.active_services = vec![false];
    query.active_services_previous = vec![true];
    let journeys = reverse_route(&timetable, &transfers, &reversed, &query);
    assert_eq!(journeys.len(), 1);
    assert_eq!(journeys[0].departure, 100);
    assert_eq!(journeys[0].arrival, 200);
    let mut early = request(StopIdx(0), StopIdx(1), 150, 1);
    early.active_services = vec![false];
    early.active_services_previous = vec![true];
    assert!(reverse_route(&timetable, &transfers, &reversed, &early).is_empty());
}

#[test]
fn exclusions_refuse_the_named_supply() {
    let (timetable, transfers) = network();
    let reversed = ReversedTransfers::build(&transfers);
    let mut query = request(StopIdx(0), StopIdx(3), 450, 4);
    let mut stops = vec![false; 5];
    stops[1] = true;
    query.exclusions = Some(std::sync::Arc::new(Exclusions::new(
        stops,
        vec![false; timetable.trip_count() as usize],
        vec![false; 3],
    )));
    assert!(reverse_route(&timetable, &transfers, &reversed, &query).is_empty());
}

#[test]
fn one_to_all_returns_per_round_states_for_composition() {
    // From the deadline side of `network()`, stop 0 reaches the
    // destination with two rides (leave 100, arrive 400); stop 1
    // boards B directly (leave 250, arrive 400).
    let (timetable, transfers) = network();
    let reversed = ReversedTransfers::build(&transfers);
    let states = reverse_one_to_all(
        &timetable,
        &reversed,
        &request(StopIdx(0), StopIdx(3), 450, 4),
    );
    assert!(states[0].contains(&(2, 100, 400)));
    assert!(states[1].contains(&(1, 250, 400)));
}

#[test]
fn unequal_access_walks_order_by_rides_after_composition() {
    // Access walks of 0 and 150 seconds equalize the composed
    // departures of a two-ride and a one-ride candidate at 100 — the
    // per-round states let the consumer prefer the one-ride journey,
    // which a scalar per-stop answer could not express.
    let (timetable, transfers) = network();
    let reversed = ReversedTransfers::build(&transfers);
    let states = reverse_one_to_all(
        &timetable,
        &reversed,
        &request(StopIdx(0), StopIdx(3), 450, 4),
    );
    let composed: Vec<(u32, u16, u32)> = [(0usize, 0u32), (1usize, 150u32)]
        .iter()
        .flat_map(|&(stop, walk)| {
            states[stop].iter().filter_map(move |&(round, dep, ach)| {
                dep.checked_sub(walk).map(|at| (at, round, ach))
            })
        })
        .collect();
    let best = composed
        .iter()
        .max_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)).then(b.2.cmp(&a.2)))
        .copied()
        .unwrap();
    assert_eq!(best, (100, 1, 400), "the one-ride candidate wins the tie");
}

#[test]
fn the_deadline_side_staircase_keeps_earlier_trips() {
    // Two trips of ONE pattern serve the deadline-side leg from stop 1:
    // the later trip leaves at 250 arriving 500, the earlier at 200
    // arriving 300. A latest-trip-only boarding would fix the worse
    // arrival; the shared upstream trip equalizes both journeys'
    // departures, where the earlier arrival must win — the Helsinki
    // oracle caught exactly this within-pattern case.
    let mut builder = TimetableBuilder::new(3);
    let upstream = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let last = builder.add_pattern(&[StopIdx(1), StopIdx(2)], 1).unwrap();
    builder
        .add_trip(upstream, vec![time(50), time(100)], 0, 0)
        .unwrap();
    builder
        .add_trip(last, vec![time(200), time(300)], 1, 0)
        .unwrap();
    builder
        .add_trip(last, vec![time(250), time(500)], 2, 0)
        .unwrap();
    let timetable = builder.finish();
    let transfers = Transfers::empty(3);
    let reversed = ReversedTransfers::build(&transfers);
    let journeys = reverse_route(
        &timetable,
        &transfers,
        &reversed,
        &request(StopIdx(0), StopIdx(2), 600, 3),
    );
    assert_eq!(journeys.len(), 1);
    assert_eq!(journeys[0].departure, 50);
    assert_eq!(journeys[0].arrival, 300);
}

#[test]
fn journeys_may_end_with_a_trailing_transfer() {
    // Forward journeys can end ride → transfer → egress; the reverse
    // engine reaches them through the one-hop initialization extension.
    let mut builder = TimetableBuilder::new(3);
    let a = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    builder
        .add_trip(a, vec![time(100), time(200)], 0, 0)
        .unwrap();
    let timetable = builder.finish();
    let transfers = Transfers::from_edges(3, &[(StopIdx(1), StopIdx(2), 60, 60.0)]).unwrap();
    let reversed = ReversedTransfers::build(&transfers);
    let journeys = reverse_route(
        &timetable,
        &transfers,
        &reversed,
        &request(StopIdx(0), StopIdx(2), 300, 1),
    );
    assert_eq!(journeys.len(), 1);
    assert_eq!(journeys[0].departure, 100);
    assert_eq!(journeys[0].arrival, 260);
    assert!(matches!(journeys[0].legs[2], Leg::Transfer { .. }));
}

#[test]
fn journeys_never_begin_with_a_transfer() {
    // Forward journeys always begin access → ride: an opening walk
    // from the origin stop to the boarding stop is not a journey shape
    // the forward search produces, so the reverse must not invent it.
    let mut builder = TimetableBuilder::new(3);
    let a = builder.add_pattern(&[StopIdx(1), StopIdx(2)], 0).unwrap();
    builder
        .add_trip(a, vec![time(200), time(300)], 0, 0)
        .unwrap();
    let timetable = builder.finish();
    let transfers = Transfers::from_edges(3, &[(StopIdx(0), StopIdx(1), 60, 60.0)]).unwrap();
    let reversed = ReversedTransfers::build(&transfers);
    let journeys = reverse_route(
        &timetable,
        &transfers,
        &reversed,
        &request(StopIdx(0), StopIdx(2), 400, 1),
    );
    assert!(journeys.is_empty());
}

#[test]
fn yesterdays_daytime_trips_have_no_query_day_clock() {
    // A previous-day service whose trip is stored at ordinary daytime
    // hours ran yesterday morning: subtracting the day offset has no
    // representable query-day position, so it must stay unreachable —
    // saturation would invent a fictitious midnight journey.
    let mut builder = TimetableBuilder::new(2);
    let a = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    builder
        .add_trip(a, vec![time(30_000), time(30_600)], 0, 0)
        .unwrap();
    let timetable = builder.finish();
    let transfers = Transfers::empty(2);
    let reversed = ReversedTransfers::build(&transfers);
    let mut query = request(StopIdx(0), StopIdx(1), 40_000, 1);
    query.active_services = vec![false];
    query.active_services_previous = vec![true];
    assert!(reverse_route(&timetable, &transfers, &reversed, &query).is_empty());
}

#[test]
fn route_and_trip_exclusions_mirror_the_forward_engine() {
    let (timetable, transfers) = network();
    let reversed = ReversedTransfers::build(&transfers);
    // Excluding route 1 (pattern B) severs the fast interchange; the
    // walk + C alternative remains.
    let mut query = request(StopIdx(0), StopIdx(3), 1100, 4);
    let mut routes = vec![false; 3];
    routes[1] = true;
    query.exclusions = Some(std::sync::Arc::new(Exclusions::new(
        vec![false; 5],
        vec![false; timetable.trip_count() as usize],
        routes,
    )));
    let journeys = reverse_route(&timetable, &transfers, &reversed, &query);
    assert_eq!(journeys.len(), 1);
    assert_eq!(journeys[0].arrival, 1000);
    // Excluding B's only active trip does the same.
    let mut query = request(StopIdx(0), StopIdx(3), 1100, 4);
    let mut trips = vec![false; timetable.trip_count() as usize];
    trips[2] = true;
    query.exclusions = Some(std::sync::Arc::new(Exclusions::new(
        vec![false; 5],
        trips,
        vec![false; 3],
    )));
    let journeys = reverse_route(&timetable, &transfers, &reversed, &query);
    assert_eq!(journeys.len(), 1);
    assert_eq!(journeys[0].arrival, 1000);
}

#[test]
fn one_to_all_frontiers_keep_the_tie_break_states() {
    // The equal-departure network through one_to_all: the origin's
    // single-round frontier must hold the earlier achieved arrival.
    let mut builder = TimetableBuilder::new(3);
    let fast = builder.add_pattern(&[StopIdx(0), StopIdx(2)], 0).unwrap();
    let slow = builder.add_pattern(&[StopIdx(0), StopIdx(2)], 1).unwrap();
    builder
        .add_trip(fast, vec![time(100), time(300)], 0, 0)
        .unwrap();
    builder
        .add_trip(slow, vec![time(100), time(500)], 1, 0)
        .unwrap();
    let timetable = builder.finish();
    let transfers = Transfers::empty(3);
    let reversed = ReversedTransfers::build(&transfers);
    let states = reverse_one_to_all(
        &timetable,
        &reversed,
        &request(StopIdx(0), StopIdx(2), 600, 2),
    );
    assert_eq!(states[0], vec![(1, 100, 300)]);
    // The shared-prefix network: stop 1's round-1 frontier retains
    // both continuations, and the origin's round-2 state carries the
    // earlier arrival.
    let mut builder = TimetableBuilder::new(4);
    let upstream = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let x = builder.add_pattern(&[StopIdx(1), StopIdx(3)], 1).unwrap();
    let y = builder.add_pattern(&[StopIdx(1), StopIdx(3)], 2).unwrap();
    builder
        .add_trip(upstream, vec![time(50), time(100)], 0, 0)
        .unwrap();
    builder
        .add_trip(x, vec![time(200), time(300)], 1, 0)
        .unwrap();
    builder
        .add_trip(y, vec![time(250), time(500)], 2, 0)
        .unwrap();
    let timetable = builder.finish();
    let transfers = Transfers::empty(4);
    let reversed = ReversedTransfers::build(&transfers);
    let states = reverse_one_to_all(
        &timetable,
        &reversed,
        &request(StopIdx(0), StopIdx(3), 600, 3),
    );
    let mut interchange = states[1].clone();
    interchange.sort_unstable();
    assert_eq!(interchange, vec![(1, 200, 300), (1, 250, 500)]);
    assert_eq!(states[0], vec![(2, 50, 300)]);
}

#[test]
fn exact_ties_elect_the_canonical_representative() {
    // Two patterns with identical times: the elected journey must be
    // the canonical path-key winner whatever order the scan met them.
    let mut builder = TimetableBuilder::new(2);
    let first = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let second = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 1).unwrap();
    builder
        .add_trip(first, vec![time(100), time(300)], 0, 0)
        .unwrap();
    builder
        .add_trip(second, vec![time(100), time(300)], 1, 0)
        .unwrap();
    let timetable = builder.finish();
    let transfers = Transfers::empty(2);
    let reversed = ReversedTransfers::build(&transfers);
    let journeys = reverse_route(
        &timetable,
        &transfers,
        &reversed,
        &request(StopIdx(0), StopIdx(1), 400, 2),
    );
    assert_eq!(journeys.len(), 1);
    let Leg::Transit { trip, .. } = journeys[0].legs[1] else {
        panic!("a transit leg");
    };
    assert_eq!(trip.0, 0, "the canonical (lexicographically smaller) trip");
}

#[test]
fn a_sibling_platforms_later_train_never_evicts_the_direct_journey() {
    // Origin 0 and its sibling platform 1 are joined both ways. The
    // sibling sees a later train to the destination, whose walk-rooted
    // reverse label dominates the origin's direct ride label on both
    // criteria — but a walk root cannot open a journey, so pruning the
    // ride-rooted label would erase the origin's only journey. The
    // Helsinki closure oracle caught exactly this eviction.
    let mut builder = TimetableBuilder::new(3);
    let direct = builder.add_pattern(&[StopIdx(0), StopIdx(2)], 0).unwrap();
    let sibling = builder.add_pattern(&[StopIdx(1), StopIdx(2)], 1).unwrap();
    builder
        .add_trip(direct, vec![time(100), time(300)], 0, 0)
        .unwrap();
    builder
        .add_trip(sibling, vec![time(250), time(400)], 1, 0)
        .unwrap();
    let timetable = builder.finish();
    let transfers = Transfers::from_edges(
        3,
        &[
            (StopIdx(0), StopIdx(1), 120, 100.0),
            (StopIdx(1), StopIdx(0), 120, 100.0),
        ],
    )
    .unwrap();
    let reversed = ReversedTransfers::build(&transfers);
    let journeys = reverse_route(
        &timetable,
        &transfers,
        &reversed,
        &request(StopIdx(0), StopIdx(2), 500, 2),
    );
    assert_eq!(journeys.len(), 1);
    assert_eq!(journeys[0].departure, 100);
    assert_eq!(journeys[0].arrival, 300);
    assert_eq!(journeys[0].rides(), 1);
}

#[test]
fn identical_trips_in_one_pattern_elect_the_canonical_legs() {
    // Two trips of one pattern with identical times, ridden as the
    // upstream leg of a two-ride journey: forward RAPTOR elects the
    // canonical (lowest-key) trip, and the reverse must return the
    // same legs — not silently the highest-index feasible trip.
    let mut builder = TimetableBuilder::new(3);
    let upstream = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let last = builder.add_pattern(&[StopIdx(1), StopIdx(2)], 1).unwrap();
    builder
        .add_trip(upstream, vec![time(100), time(200)], 0, 0)
        .unwrap();
    builder
        .add_trip(upstream, vec![time(100), time(200)], 1, 0)
        .unwrap();
    builder
        .add_trip(last, vec![time(300), time(400)], 2, 0)
        .unwrap();
    let timetable = builder.finish();
    let transfers = Transfers::empty(3);
    let reversed = ReversedTransfers::build(&transfers);
    let reverse = reverse_route(
        &timetable,
        &transfers,
        &reversed,
        &request(StopIdx(0), StopIdx(2), 500, 3),
    );
    assert_eq!(reverse.len(), 1);
    let mut forward_request = request(StopIdx(0), StopIdx(2), reverse[0].departure, 3);
    forward_request.departure = reverse[0].departure;
    let forward = Raptor.route(&timetable, &transfers, &forward_request);
    let matched = forward
        .iter()
        .find(|journey| journey.arrival == reverse[0].arrival)
        .expect("the forward replay finds the same journey");
    assert_eq!(matched.legs, reverse[0].legs);
}

#[test]
fn competing_egress_seeds_are_order_independent() {
    // Egress at stop 3 is fast (10 s) and a walk 2→3 undercuts stop
    // 2's own slow egress (100 s), so the transfer-extended seed from
    // stop 3 dominates stop 2's direct label. The only journey still
    // rides through stop 2's OWN extension to stop 1 — it must
    // survive whichever order the egress entries arrive in.
    let mut builder = TimetableBuilder::new(4);
    let a = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    builder
        .add_trip(a, vec![time(100), time(200)], 0, 0)
        .unwrap();
    let timetable = builder.finish();
    let transfers = Transfers::from_edges(
        4,
        &[
            (StopIdx(1), StopIdx(2), 50, 50.0),
            (StopIdx(2), StopIdx(3), 50, 50.0),
        ],
    )
    .unwrap();
    let reversed = ReversedTransfers::build(&transfers);
    let egress_orders = [
        vec![(StopIdx(3), 10), (StopIdx(2), 100)],
        vec![(StopIdx(2), 100), (StopIdx(3), 10)],
    ];
    let mut results = Vec::new();
    for egress in egress_orders {
        let query = Request {
            departure: 1000,
            access: vec![(StopIdx(0), 0)],
            egress,
            active_services: vec![true],
            active_services_previous: Vec::new(),
            max_transfers: 3,
            exclusions: None,
        };
        results.push(reverse_route(&timetable, &transfers, &reversed, &query));
    }
    assert_eq!(results[0], results[1]);
    assert_eq!(results[0].len(), 1);
    assert_eq!(results[0][0].departure, 100);
    assert_eq!(results[0][0].arrival, 350);
}

#[test]
fn round_256_states_survive_the_widest_transfer_budget() {
    // A 257-stop chain of single-hop patterns forces a 256-ride
    // journey under max_transfers=255 — the widest legal budget. The
    // returned round must be 256, never wrapped to a zero-ride state.
    let stops = 257u32;
    let mut builder = TimetableBuilder::new(stops);
    for hop in 0..stops - 1 {
        let pattern = builder
            .add_pattern(&[StopIdx(hop), StopIdx(hop + 1)], hop)
            .unwrap();
        builder
            .add_trip(pattern, vec![time(10 * hop), time(10 * hop + 5)], 0, 0)
            .unwrap();
    }
    let timetable = builder.finish();
    let transfers = Transfers::empty(stops);
    let reversed = ReversedTransfers::build(&transfers);
    let query = Request {
        departure: 10 * stops,
        access: vec![(StopIdx(0), 0)],
        egress: vec![(StopIdx(stops - 1), 0)],
        active_services: vec![true],
        active_services_previous: Vec::new(),
        max_transfers: 255,
        exclusions: None,
    };
    let states = reverse_one_to_all(&timetable, &reversed, &query);
    let rounds: Vec<u16> = states[0].iter().map(|&(round, _, _)| round).collect();
    assert_eq!(rounds, vec![256]);
}

#[test]
fn the_frontier_evaluates_every_deadline_mark_exactly() {
    // The frontier of one T2 run answers every mark: per mark and
    // round, the evaluation equals an independent single-deadline
    // run's states. The fixture includes a mark whose winner is a
    // state a later mark's winner dominates on departure alone (the
    // early K departing 100 arriving 400 versus the later 700→1000):
    // both survive the frontier, and the evaluation must pick per
    // mark, never globally.
    let (timetable, transfers) = network();
    let reversed = ReversedTransfers::build(&transfers);
    let marks = [400, 450, 999, 1000, 1500];
    let full = reverse_one_to_all(
        &timetable,
        &reversed,
        &request(StopIdx(0), StopIdx(3), 1500, 4),
    );
    for (at, &mark) in marks.iter().enumerate() {
        let single = reverse_one_to_all(
            &timetable,
            &reversed,
            &request(StopIdx(0), StopIdx(3), mark, 4),
        );
        for stop in 0..timetable.stop_count() as usize {
            let mut evaluated = reverse_profile_states(&full[stop], &marks)[at].clone();
            let mut expected: Vec<(u16, u32, u32)> = Vec::new();
            let mut rounds: Vec<u16> = single[stop].iter().map(|&(round, _, _)| round).collect();
            rounds.sort_unstable();
            rounds.dedup();
            for round in rounds {
                let mut best: Option<(u32, u32)> = None;
                for &(held, departure, achieved) in &single[stop] {
                    if held != round {
                        continue;
                    }
                    let wins = match best {
                        None => true,
                        Some((d, a)) => departure > d || (departure == d && achieved < a),
                    };
                    if wins {
                        best = Some((departure, achieved));
                    }
                }
                if let Some((departure, achieved)) = best {
                    expected.push((round, departure, achieved));
                }
            }
            evaluated.sort_unstable();
            expected.sort_unstable();
            assert_eq!(evaluated, expected, "stop {stop} mark {mark}");
        }
    }
}

#[test]
fn early_marks_resurrect_states_later_winners_shadow() {
    // Two direct trips 0→1: the early one departs 100 arriving 200,
    // the late one departs 700 arriving 800. At mark 200 the early
    // trip is the only answer; at mark 800 the late one wins. A
    // global winner would erase the early mark's answer.
    let mut builder = TimetableBuilder::new(2);
    let a = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    builder
        .add_trip(a, vec![time(100), time(200)], 0, 0)
        .unwrap();
    builder
        .add_trip(a, vec![time(700), time(800)], 1, 0)
        .unwrap();
    let timetable = builder.finish();
    let transfers = Transfers::empty(2);
    let reversed = ReversedTransfers::build(&transfers);
    let states = reverse_one_to_all(
        &timetable,
        &reversed,
        &request(StopIdx(0), StopIdx(1), 900, 3),
    );
    let marks = [200, 500, 800];
    let evaluated = reverse_profile_states(&states[0], &marks);
    assert_eq!(evaluated[0], vec![(1, 100, 200)]);
    assert_eq!(evaluated[1], vec![(1, 100, 200)]);
    assert_eq!(evaluated[2], vec![(1, 700, 800)]);
}

#[test]
fn tagged_states_attribute_winners_to_their_seeds() {
    // Two egress seeds compete: stop 3 (seed 0, via pattern B arriving
    // 400) and stop 4 (seed 1, via the 2→4 footpath achieving 350).
    // Tagged runs isolate dominance per seed, so BOTH destinations'
    // chains survive the shared frontier — each destination's own
    // winner stays exactly reproducible — and the reduction picks the
    // shortest winner duration.
    let (timetable, transfers) = network();
    let reversed = ReversedTransfers::build(&transfers);
    let query = Request {
        departure: 1000,
        access: vec![(StopIdx(0), 0)],
        egress: vec![(StopIdx(3), 0), (StopIdx(4), 0)],
        active_services: vec![true; 4],
        active_services_previous: Vec::new(),
        max_transfers: 3,
        exclusions: None,
    };
    let states = reverse_one_to_all_tagged(&timetable, &reversed, &query, None);
    let winner = states[0]
        .iter()
        .copied()
        .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)).then(b.2.cmp(&a.2)))
        .expect("origin reached");
    let (_, departure, achieved, seed) = winner;
    assert_eq!(seed, 1, "the footpath destination is nearest");
    assert_eq!(achieved - departure, 250);
    // Isolation keeps the other destination's chains alive too.
    assert!(states[0].iter().any(|&(_, _, _, seed)| seed == 0));
    // Alone, every chain carries the only seed's tag.
    let only_three = Request {
        egress: vec![(StopIdx(3), 0)],
        ..query.clone()
    };
    let states = reverse_one_to_all_tagged(&timetable, &reversed, &only_three, None);
    assert!(!states[0].is_empty());
    assert!(states[0].iter().all(|&(_, _, _, seed)| seed == 0));
}

#[test]
fn shared_trip_destinations_both_survive_a_tagged_run() {
    // One trip 0→1→2 serves BOTH destinations (stop 1, seed 0; stop
    // 2, seed 1). The onboard dedupe must keep one continuation per
    // seed — merged by (trip, offset) alone, the earlier alighting
    // would silently erase the farther destination from the frontier.
    let mut builder = TimetableBuilder::new(3);
    let a = builder
        .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 0)
        .unwrap();
    builder
        .add_trip(a, vec![time(100), time(200), time(300)], 0, 0)
        .unwrap();
    let timetable = builder.finish();
    let transfers = Transfers::empty(3);
    let reversed = ReversedTransfers::build(&transfers);
    let query = Request {
        departure: 1000,
        access: vec![(StopIdx(0), 0)],
        egress: vec![(StopIdx(1), 0), (StopIdx(2), 0)],
        active_services: vec![true],
        active_services_previous: Vec::new(),
        max_transfers: 3,
        exclusions: None,
    };
    let states = reverse_one_to_all_tagged(&timetable, &reversed, &query, None);
    let seeds: Vec<u32> = states[0].iter().map(|&(_, _, _, seed)| seed).collect();
    assert!(seeds.contains(&0), "the nearer destination survives");
    assert!(seeds.contains(&1), "the farther destination survives");
    let nearer = states[0]
        .iter()
        .find(|&&(_, _, _, seed)| seed == 0)
        .unwrap();
    let farther = states[0]
        .iter()
        .find(|&&(_, _, _, seed)| seed == 1)
        .unwrap();
    assert_eq!(nearer.2 - nearer.1, 100);
    assert_eq!(farther.2 - farther.1, 200);
}

#[test]
fn a_shared_access_egress_stop_never_elects_an_out_and_back() {
    // Stop 1 serves both tables; riding out (1→3) and back (3→1)
    // reaches the destination later than walking straight through,
    // and the forward engine prunes the chain against its own
    // walk-through bound at the shared stop. The replay drops the
    // candidate rather than panicking — the walking alternative is
    // the callers' to place.
    let mut builder = TimetableBuilder::new(4);
    let out = builder.add_pattern(&[StopIdx(1), StopIdx(3)], 0).unwrap();
    let back = builder.add_pattern(&[StopIdx(3), StopIdx(1)], 1).unwrap();
    builder
        .add_trip(out, vec![time(250), time(400)], 0, 0)
        .unwrap();
    builder
        .add_trip(back, vec![time(450), time(600)], 1, 0)
        .unwrap();
    let timetable = builder.finish();
    let transfers = Transfers::from_edges(4, &[]).unwrap();
    let reversed = ReversedTransfers::build(&transfers);
    let request = Request {
        departure: 1000,
        access: vec![(StopIdx(1), 30)],
        egress: vec![(StopIdx(1), 20)],
        active_services: vec![true; 2],
        active_services_previous: Vec::new(),
        max_transfers: 3,
        exclusions: None,
    };
    let journeys = reverse_route(&timetable, &transfers, &reversed, &request);
    assert!(
        journeys.is_empty(),
        "an out-and-back chain is not a journey: {journeys:?}"
    );
}
