use super::*;
use crate::timetable::{StopTime, TimetableBuilder};

fn time(at: u32) -> StopTime {
    StopTime {
        arrival: at,
        departure: at,
    }
}

/// Four stops: a bike-forbidden trip 0→1, a bike-allowed trip 1→2, a
/// walking row 2→3, and in the carriage set additionally the ride row
/// 1→3.
fn fixture() -> (Timetable, Transfers, Transfers, Vec<bool>) {
    let mut builder = TimetableBuilder::new(4);
    let first = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let second = builder.add_pattern(&[StopIdx(1), StopIdx(2)], 1).unwrap();
    builder
        .add_trip(first, vec![time(10), time(20)], 0, 0)
        .unwrap();
    builder
        .add_trip(second, vec![time(30), time(60)], 1, 0)
        .unwrap();
    let timetable = builder.finish();
    let walking = Transfers::from_edges(4, &[(StopIdx(2), StopIdx(3), 100, 100.0)]).unwrap();
    let mut carriage = Transfers::from_edges(
        4,
        &[
            (StopIdx(1), StopIdx(3), 200, 900.0),
            (StopIdx(2), StopIdx(3), 100, 100.0),
        ],
    )
    .unwrap();
    carriage.mark_unclosed();
    // CSR by source: (1→3) is the ride, (2→3) the walking row.
    let ride_edge = vec![true, false];
    (timetable, walking, carriage, ride_edge)
}

fn request(carrying: Vec<(StopIdx, u32)>, free: Vec<(StopIdx, u32)>) -> CarriageRequest {
    CarriageRequest {
        departure: 0,
        carrying_access: carrying,
        free_access: free,
        max_transfers: 3,
    }
}

/// Carrying may not board the forbidden first trip; the Free plane
/// (seeded by leaving the vehicle at the origin) rides it — the
/// optional-carriage contract.
#[test]
fn carrying_never_boards_a_forbidden_trip() {
    let (timetable, walking, carriage, ride_edge) = fixture();
    let inputs = CarriageInputs {
        timetable: &timetable,
        walking: &walking,
        carriage: &carriage,
        ride_edge: &ride_edge,
        carrying_mask: &[false, true],
        park_mask: &[false; 4],
        active_services: &[true],
        active_services_previous: &[],
    };
    let result = search(
        &inputs,
        &request(vec![(StopIdx(0), 0)], vec![(StopIdx(0), 0)]),
    );
    assert_eq!(result.arrivals(FREE)[1], Some(20));
    assert_eq!(result.arrivals(CARRYING)[1], None);
    assert_eq!(result.arrivals(FREE)[2], Some(60));
    assert_eq!(result.arrivals(FREE)[3], Some(160));
}

/// A cycled Carrying access parks at stop 1, entering Free the same
/// round; both planes then ride the allowed trip, and the carriage
/// ride row extends Carrying's transit arrival.
#[test]
fn parking_bridges_the_planes_and_rides_extend_transit() {
    let (timetable, walking, carriage, ride_edge) = fixture();
    let inputs = CarriageInputs {
        timetable: &timetable,
        walking: &walking,
        carriage: &carriage,
        ride_edge: &ride_edge,
        carrying_mask: &[false, true],
        park_mask: &[false, true, false, false],
        active_services: &[true],
        active_services_previous: &[],
    };
    let result = search(
        &inputs,
        &request(vec![(StopIdx(1), 15)], vec![(StopIdx(0), 0)]),
    );
    assert_eq!(result.arrivals(CARRYING)[1], Some(15));
    // Parked at the seed round.
    assert_eq!(result.arrivals(FREE)[1], Some(15));
    assert_eq!(result.arrivals(FREE)[2], Some(60));
    assert_eq!(result.arrivals(FREE)[3], Some(160));
    // Carrying boards the allowed trip and the carriage walking row
    // extends its transit arrival; the ride row from stop 1 relaxes
    // only from transit arrivals (the exact phase), and none lands at
    // stop 1 while carrying.
    assert_eq!(result.arrivals(CARRYING)[2], Some(60));
    assert_eq!(result.arrivals(CARRYING)[3], Some(160));
}

/// The reconstructed chain across a park: cycled access to stop 1,
/// park, the (bike-forbidden) network from there — the Park leg sits
/// between the Carrying access and the Free transit legs, and carried
/// boardings elsewhere report `bike_aboard`.
#[test]
fn reconstruction_crosses_the_park_and_flags_carried_boardings() {
    let (timetable, walking, carriage, ride_edge) = fixture();
    let inputs = CarriageInputs {
        timetable: &timetable,
        walking: &walking,
        carriage: &carriage,
        ride_edge: &ride_edge,
        carrying_mask: &[false, true],
        park_mask: &[false, true, false, false],
        active_services: &[true],
        active_services_previous: &[],
    };
    let result = search(&inputs, &request(vec![(StopIdx(1), 15)], vec![]));
    // Free's best at stop 3: park at 1 (15) → ride 1→2 (60) → walk (160).
    let (round, arrival) = result.best_round(FREE, StopIdx(3)).unwrap();
    assert_eq!(arrival, 160);
    let legs = result.reconstruct(&timetable, FREE, round, StopIdx(3));
    assert_eq!(
        legs,
        vec![
            CarriageLeg::Access {
                plane: CARRYING,
                to_stop: StopIdx(1),
                arrival: 15,
            },
            CarriageLeg::Park {
                stop: StopIdx(1),
                at: 15,
            },
            CarriageLeg::Transit {
                trip: TripIdx(1),
                board_stop: StopIdx(1),
                alight_stop: StopIdx(2),
                board_position: 0,
                alight_position: 1,
                board_time: 30,
                alight_time: 60,
                bike_aboard: false,
            },
            CarriageLeg::Transfer {
                from_stop: StopIdx(2),
                to_stop: StopIdx(3),
                departure: 60,
                arrival: 160,
                ride: false,
            },
        ]
    );
    // The Carrying plane's own journey to 3 carries aboard.
    let (carried_round, _) = result.best_round(CARRYING, StopIdx(3)).unwrap();
    let carried = result.reconstruct(&timetable, CARRYING, carried_round, StopIdx(3));
    assert!(carried.iter().any(|leg| matches!(
        leg,
        CarriageLeg::Transit {
            bike_aboard: true,
            ..
        }
    )));
}

/// A bike-allowed trip on an inactive service must never board while
/// carrying: the mask scan re-checks the service on every candidate.
#[test]
fn carrying_never_boards_an_inactive_permitted_trip() {
    let mut builder = TimetableBuilder::new(3);
    let line = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    // Trip 0: active service, bikes forbidden. Trip 1: inactive
    // service, bikes allowed.
    builder
        .add_trip(line, vec![time(10), time(20)], 0, 0)
        .unwrap();
    builder
        .add_trip(line, vec![time(30), time(40)], 1, 1)
        .unwrap();
    let timetable = builder.finish();
    let walking = Transfers::empty(3);
    let mut carriage = Transfers::empty(3);
    carriage.mark_unclosed();
    let inputs = CarriageInputs {
        timetable: &timetable,
        walking: &walking,
        carriage: &carriage,
        ride_edge: &[],
        carrying_mask: &[false, true],
        park_mask: &[false; 3],
        active_services: &[true, false],
        active_services_previous: &[],
    };
    let result = search(
        &inputs,
        &request(vec![(StopIdx(0), 0)], vec![(StopIdx(0), 0)]),
    );
    // Carrying: trip 0 forbidden, trip 1 inactive — unreachable.
    assert_eq!(result.arrivals(CARRYING)[1], None);
    // Free rides the active (forbidden-for-bikes) trip.
    assert_eq!(result.arrivals(FREE)[1], Some(20));
}

/// A seed that parks at the sole eligible stop walks onward before the
/// first transit round: park-then-walk boarding stops are seeded.
#[test]
fn round_zero_parks_walk_before_the_first_round() {
    let mut builder = TimetableBuilder::new(4);
    // The only line boards at stop 2; the bicycle reaches stop 1.
    let line = builder.add_pattern(&[StopIdx(2), StopIdx(3)], 0).unwrap();
    builder
        .add_trip(line, vec![time(100), time(200)], 0, 0)
        .unwrap();
    let timetable = builder.finish();
    let walking = Transfers::from_edges(4, &[(StopIdx(1), StopIdx(2), 30, 30.0)]).unwrap();
    let mut carriage = Transfers::empty(4);
    carriage.mark_unclosed();
    let inputs = CarriageInputs {
        timetable: &timetable,
        walking: &walking,
        carriage: &carriage,
        ride_edge: &[],
        carrying_mask: &[false],
        park_mask: &[false, true, false, false],
        active_services: &[true],
        active_services_previous: &[],
    };
    // Carrying-only seed at the parking stop 1 (a cycled access).
    let result = search(&inputs, &request(vec![(StopIdx(1), 10)], vec![]));
    // Park at 1 (10), walk to 2 (40), board (dep 100) → 3 at 200.
    assert_eq!(result.arrivals(FREE)[2], Some(40));
    assert_eq!(result.arrivals(FREE)[3], Some(200));
}

/// A brute-force reference: the same possession-state contract with no
/// marking, no horizons, and no FIFO binary search — every trip of
/// every pattern is scanned from every stop each round. One prefix
/// snapshot per round (entry `r` = best within `r` boardings), so the
/// comparison pins every intermediate round, not just the final
/// arrivals. Divergence from `search` means the engine's machinery,
/// not the contract.
fn oracle(inputs: &CarriageInputs<'_>, request: &CarriageRequest) -> Vec<[Vec<Option<u32>>; 2]> {
    const UNREACHED: u32 = u32::MAX;
    let timetable = inputs.timetable;
    let stops = timetable.stop_count() as usize;
    let rounds = request.max_transfers as usize + 1;
    let mut best = [vec![UNREACHED; stops], vec![UNREACHED; stops]];
    for &(stop, seconds) in &request.carrying_access {
        let arrival = request.departure + seconds;
        let slot = &mut best[CARRYING][stop.0 as usize];
        *slot = (*slot).min(arrival);
    }
    for &(stop, seconds) in &request.free_access {
        let arrival = request.departure + seconds;
        let slot = &mut best[FREE][stop.0 as usize];
        *slot = (*slot).min(arrival);
    }
    let park = |best: &mut [Vec<u32>; 2], source: &[u32]| {
        for stop in 0..stops {
            if inputs.park_mask[stop] && source[stop] != UNREACHED {
                best[FREE][stop] = best[FREE][stop].min(source[stop]);
            }
        }
    };
    let free_walks = |best: &mut [Vec<u32>; 2], source: &[u32]| {
        for (stop, &at) in source.iter().enumerate() {
            if at == UNREACHED {
                continue;
            }
            for edge in inputs.walking.from_stop(StopIdx(stop as u32)) {
                let slot = &mut best[FREE][edge.to.0 as usize];
                *slot = (*slot).min(at + edge.duration);
            }
        }
    };
    // Round-zero closure: park, then Free walking over every parking
    // candidate, shadowed or not — the seeds are reduction-closed
    // (walking again would compose closure rows past the budget), but
    // a shadowed park walks fresh under the exact rule.
    let carrying_now = best[CARRYING].clone();
    let mut parked_now = vec![UNREACHED; stops];
    for stop in 0..stops {
        if inputs.park_mask[stop] {
            parked_now[stop] = carrying_now[stop];
        }
    }
    park(&mut best, &carrying_now);
    free_walks(&mut best, &parked_now);
    let snapshot = |best: &[Vec<u32>; 2]| {
        [CARRYING, FREE].map(|plane| {
            best[plane]
                .iter()
                .map(|&at| (at != UNREACHED).then_some(at))
                .collect::<Vec<_>>()
        })
    };
    let mut per_round = vec![snapshot(&best)];
    for _round in 1..=rounds {
        // Transit: brute-force over every trip and boarding position.
        let mut transit = [vec![UNREACHED; stops], vec![UNREACHED; stops]];
        for plane in [CARRYING, FREE] {
            for pattern in 0..timetable.pattern_count() {
                let pattern = crate::timetable::PatternIdx(pattern);
                let pattern_stops = timetable.pattern_stops(pattern);
                let range = timetable.pattern_trip_range(pattern);
                for trip in range.start..range.end {
                    let trip = TripIdx(trip);
                    if plane == CARRYING && !inputs.carrying_mask[trip.0 as usize] {
                        continue;
                    }
                    let times = timetable.trip_stop_times(trip);
                    let streams = [
                        (0u32, inputs.active_services),
                        (86_400, inputs.active_services_previous),
                    ];
                    for (shift, active) in streams {
                        if !active
                            .get(timetable.trip_service(trip) as usize)
                            .copied()
                            .unwrap_or(false)
                        {
                            continue;
                        }
                        for board in 0..pattern_stops.len() - 1 {
                            let ready = best[plane][pattern_stops[board].0 as usize];
                            if ready == UNREACHED {
                                continue;
                            }
                            let Some(threshold) = ready.checked_add(shift) else {
                                continue;
                            };
                            if threshold > times[board].departure {
                                continue;
                            }
                            for alight in board + 1..pattern_stops.len() {
                                let slot = &mut transit[plane][pattern_stops[alight].0 as usize];
                                *slot = (*slot).min(times[alight].arrival.saturating_sub(shift));
                            }
                        }
                    }
                }
            }
        }
        // Non-transit closure in phase order: carriage rows from the
        // round's Carrying transit arrivals, park over everything the
        // round produced, Free walks over Free transit and parks.
        let mut carrying_round = transit[CARRYING].clone();
        for (stop, &at) in transit[CARRYING].iter().enumerate() {
            if at == UNREACHED {
                continue;
            }
            for edge in inputs.carriage.from_stop(StopIdx(stop as u32)) {
                let slot = &mut carrying_round[edge.to.0 as usize];
                *slot = (*slot).min(at + edge.duration);
            }
        }
        for stop in 0..stops {
            best[CARRYING][stop] = best[CARRYING][stop].min(carrying_round[stop]);
        }
        park(&mut best, &carrying_round);
        let mut free_round = transit[FREE].clone();
        for stop in 0..stops {
            if inputs.park_mask[stop] {
                free_round[stop] = free_round[stop].min(carrying_round[stop]);
            }
            best[FREE][stop] = best[FREE][stop].min(free_round[stop]);
        }
        free_walks(&mut best, &free_round);
        per_round.push(snapshot(&best));
    }
    per_round
}

/// The engine matches the brute-force reference on every fixture, per
/// plane and stop.
#[test]
fn engine_matches_the_exhaustive_oracle() {
    let (timetable, walking, carriage, ride_edge) = fixture();
    for park_mask in [
        [false; 4],
        [false, true, false, false],
        [true, true, true, true],
    ] {
        for carrying_mask in [[false, true], [true, true], [false, false]] {
            let inputs = CarriageInputs {
                timetable: &timetable,
                walking: &walking,
                carriage: &carriage,
                ride_edge: &ride_edge,
                carrying_mask: &carrying_mask,
                park_mask: &park_mask,
                active_services: &[true],
                active_services_previous: &[],
            };
            for request in [
                request(vec![(StopIdx(0), 0)], vec![(StopIdx(0), 0)]),
                request(vec![(StopIdx(1), 15)], vec![]),
                request(
                    vec![(StopIdx(0), 5), (StopIdx(1), 40)],
                    vec![(StopIdx(0), 5)],
                ),
            ] {
                let result = search(&inputs, &request);
                let reference = oracle(&inputs, &request);
                // The walking-only baseline: no carried seeds, so a
                // Free result beating it must have parked somewhere.
                let baseline = search(
                    &inputs,
                    &CarriageRequest {
                        departure: request.departure,
                        carrying_access: Vec::new(),
                        free_access: request.free_access.clone(),
                        max_transfers: request.max_transfers,
                    },
                );
                // The door-to-door egress fold, per plane offsets: the
                // engine's prefix arrivals against the independent
                // oracle's snapshots — a carried egress folds from the
                // Carrying plane only (a parked chain lives in Free,
                // so parking removes carried-egress eligibility by
                // construction), a walking egress from Free.
                let egress_sets: [[&[(u32, u32)]; 2]; 2] =
                    [[&[(3, 0)], &[(3, 50)]], [&[], &[(2, 10), (3, 10)]]];
                for egress in &egress_sets {
                    let mut frontier = u32::MAX;
                    for (round, snapshot) in reference.iter().enumerate() {
                        let fold = |value: Option<u32>, seconds: u32| {
                            value.and_then(|at| at.checked_add(seconds))
                        };
                        let mut engine_best: Option<u32> = None;
                        let mut oracle_best: Option<u32> = None;
                        for plane in [CARRYING, FREE] {
                            for &(stop, seconds) in egress[plane] {
                                let at = fold(
                                    result.prefix_arrival(plane, round, StopIdx(stop)),
                                    seconds,
                                );
                                if at.is_some() && (engine_best.is_none() || at < engine_best) {
                                    engine_best = at;
                                }
                                let at = fold(snapshot[plane][stop as usize], seconds);
                                if at.is_some() && (oracle_best.is_none() || at < oracle_best) {
                                    oracle_best = at;
                                }
                            }
                        }
                        assert_eq!(
                            engine_best, oracle_best,
                            "egress fold: round {round}, park {park_mask:?}, \
                             mask {carrying_mask:?}"
                        );
                        if let Some(at) = engine_best {
                            assert!(at <= frontier, "the frontier must never worsen");
                            frontier = at;
                        }
                    }
                }
                for plane in [CARRYING, FREE] {
                    for (round, snapshot) in reference.iter().enumerate() {
                        for stop in 0..4u32 {
                            assert_eq!(
                                result.prefix_arrival(plane, round, StopIdx(stop)),
                                snapshot[plane][stop as usize],
                                "plane {plane}, round {round}, stop {stop}, \
                                 park {park_mask:?}, mask {carrying_mask:?}"
                            );
                        }
                    }
                    // Every chain the route frontier could select: the
                    // distinct achieving rounds across all prefixes.
                    for stop in 0..4u32 {
                        let mut seen = vec![false; result.rounds() + 1];
                        for round in 0..=result.rounds() {
                            let Some(achieved) =
                                result.achieving_round(plane, round, StopIdx(stop))
                            else {
                                continue;
                            };
                            if std::mem::replace(&mut seen[achieved], true) {
                                continue;
                            }
                            let legs =
                                result.reconstruct(&timetable, plane, achieved, StopIdx(stop));
                            assert_chain_legal(&legs, &inputs);
                            let arrival = result
                                .prefix_arrival(plane, round, StopIdx(stop))
                                .expect("an achieving round has a prefix arrival");
                            let beats_walking = plane == FREE
                                && baseline.arrivals(FREE)[stop as usize]
                                    .is_none_or(|base| arrival < base);
                            if beats_walking {
                                assert!(
                                    legs.iter()
                                        .any(|leg| matches!(leg, CarriageLeg::Park { .. })),
                                    "a Free chain beat the walking-only baseline without a park"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Asserts the possession-state discipline over a reconstructed chain:
/// carried boardings only on permitted trips and never after the park,
/// rides never after the park, parks only at eligible stops, and
/// non-decreasing times throughout.
fn assert_chain_legal(legs: &[CarriageLeg], inputs: &CarriageInputs<'_>) {
    let mut parked = false;
    let mut last_time = 0u32;
    for leg in legs {
        match *leg {
            CarriageLeg::Access { arrival, .. } => last_time = arrival,
            CarriageLeg::Park { stop, at } => {
                assert!(
                    inputs.park_mask[stop.0 as usize],
                    "parked at an ineligible stop"
                );
                assert!(!parked, "parked twice");
                parked = true;
                assert!(at >= last_time);
                last_time = at;
            }
            CarriageLeg::Transit {
                trip,
                board_time,
                alight_time,
                bike_aboard,
                ..
            } => {
                if bike_aboard {
                    assert!(!parked, "carried a boarding after the park");
                    assert!(
                        inputs.carrying_mask[trip.0 as usize],
                        "carried a forbidden trip"
                    );
                }
                assert!(board_time >= last_time);
                assert!(alight_time >= board_time);
                last_time = alight_time;
            }
            CarriageLeg::Transfer {
                departure,
                arrival,
                ride,
                ..
            } => {
                if ride {
                    assert!(!parked, "rode the vehicle after the park");
                }
                assert!(departure >= last_time);
                assert!(arrival >= departure);
                last_time = arrival;
            }
        }
    }
}

/// Free access seeds are reduction-closed: walking out of them again
/// would compose closure rows past the budget, so round zero walks
/// only from park-introduced labels.
#[test]
fn closed_free_seeds_do_not_walk_out_again() {
    let (timetable, walking, carriage, ride_edge) = fixture();
    let inputs = CarriageInputs {
        timetable: &timetable,
        walking: &walking,
        carriage: &carriage,
        ride_edge: &ride_edge,
        carrying_mask: &[false, true],
        park_mask: &[false; 4],
        active_services: &[true],
        active_services_previous: &[],
    };
    // The walking row 2→3 must not extend the closed seed at stop 2.
    let result = search(
        &inputs,
        &request(vec![], vec![(StopIdx(2), 5), (StopIdx(3), 200)]),
    );
    assert_eq!(result.arrivals(FREE)[3], Some(200));
    // A park-introduced label at stop 2 walks out legally.
    let inputs = CarriageInputs {
        park_mask: &[false, false, true, false],
        ..inputs
    };
    let result = search(
        &inputs,
        &request(
            vec![(StopIdx(2), 3)],
            vec![(StopIdx(2), 5), (StopIdx(3), 200)],
        ),
    );
    assert_eq!(result.arrivals(FREE)[2], Some(3));
    assert_eq!(result.arrivals(FREE)[3], Some(103));
}

/// A previous-service-day trip whose times run past 24:00: two stops,
/// one over-midnight trip active yesterday only.
fn midnight_fixture() -> (Timetable, Transfers, Transfers, Vec<bool>) {
    let mut builder = TimetableBuilder::new(2);
    let pattern = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    builder
        .add_trip(pattern, vec![time(90_000), time(90_600)], 0, 0)
        .unwrap();
    let timetable = builder.finish();
    let walking = Transfers::from_edges(2, &[]).unwrap();
    let mut carriage = Transfers::from_edges(2, &[]).unwrap();
    carriage.mark_unclosed();
    (timetable, walking, carriage, vec![])
}

/// The previous-day stream boards under the carrying mask exactly as
/// the current one, and the oracle models it identically.
#[test]
fn over_midnight_trips_carry_and_match_the_oracle() {
    let (timetable, walking, carriage, ride_edge) = midnight_fixture();
    for carrying_mask in [[true], [false]] {
        let inputs = CarriageInputs {
            timetable: &timetable,
            walking: &walking,
            carriage: &carriage,
            ride_edge: &ride_edge,
            carrying_mask: &carrying_mask,
            park_mask: &[false; 2],
            active_services: &[false],
            active_services_previous: &[true],
        };
        let request = CarriageRequest {
            departure: 3_000,
            carrying_access: vec![(StopIdx(0), 0)],
            free_access: vec![(StopIdx(0), 0)],
            max_transfers: 3,
        };
        let result = search(&inputs, &request);
        assert_eq!(result.arrivals(FREE)[1], Some(4_200));
        assert_eq!(
            result.arrivals(CARRYING)[1],
            carrying_mask[0].then_some(4_200)
        );
        let reference = oracle(&inputs, &request);
        let last = reference.last().expect("the oracle snapshots every round");
        for plane in [CARRYING, FREE] {
            assert_eq!(result.arrivals(plane), last[plane], "plane {plane}");
        }
    }
}

/// Two parked seeds linked by a walking row: the walk out of the
/// second must leave its own park, never the (cheaper) walked label a
/// first-seed walk wrote over it — the closure's absent composed row
/// was excluded by the movement budget.
#[test]
fn linked_parked_seeds_never_chain_their_walks() {
    let mut builder = TimetableBuilder::new(3);
    let line = builder.add_pattern(&[StopIdx(0), StopIdx(2)], 0).unwrap();
    builder
        .add_trip(line, vec![time(10_000), time(10_100)], 0, 0)
        .unwrap();
    let timetable = builder.finish();
    // 0→1 and 1→2 exist; the composed 0→2 exceeded the closure budget.
    let walking = Transfers::from_edges(
        3,
        &[
            (StopIdx(0), StopIdx(1), 10, 10.0),
            (StopIdx(1), StopIdx(2), 10, 10.0),
        ],
    )
    .unwrap();
    let mut carriage = Transfers::from_edges(3, &[]).unwrap();
    carriage.mark_unclosed();
    let inputs = CarriageInputs {
        timetable: &timetable,
        walking: &walking,
        carriage: &carriage,
        ride_edge: &[],
        carrying_mask: &[true],
        park_mask: &[true; 3],
        active_services: &[true],
        active_services_previous: &[],
    };
    let result = search(
        &inputs,
        &request(vec![(StopIdx(0), 5), (StopIdx(1), 100)], vec![]),
    );
    // Stop 1: the first seed's walk (15) beats the park (100).
    assert_eq!(result.arrivals(FREE)[1], Some(15));
    // Stop 2: only park-at-1-then-walk is legal (110), never 25.
    assert_eq!(result.arrivals(FREE)[2], Some(110));
    // The chain renders its inline park faithfully.
    let (round, _) = result.best_round(FREE, StopIdx(2)).unwrap();
    let legs = result.reconstruct(&timetable, FREE, round, StopIdx(2));
    assert_eq!(
        legs,
        vec![
            CarriageLeg::Access {
                plane: CARRYING,
                to_stop: StopIdx(1),
                arrival: 100,
            },
            CarriageLeg::Park {
                stop: StopIdx(1),
                at: 100,
            },
            CarriageLeg::Transfer {
                from_stop: StopIdx(1),
                to_stop: StopIdx(2),
                departure: 100,
                arrival: 110,
                ride: false,
            },
        ]
    );
}

/// Two same-round transit arrivals linked by a walking row: the walk
/// out of the later one must reconstruct through its own (inline)
/// ride, never through the cheaper walked label that overwrote the
/// source — the closure's absent composed row was budget-excluded.
#[test]
fn overwritten_walk_sources_reconstruct_their_inline_ride() {
    let mut builder = TimetableBuilder::new(4);
    let slow = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let fast = builder.add_pattern(&[StopIdx(0), StopIdx(2)], 1).unwrap();
    builder
        .add_trip(slow, vec![time(10), time(100)], 0, 0)
        .unwrap();
    builder
        .add_trip(fast, vec![time(10), time(50)], 1, 0)
        .unwrap();
    let timetable = builder.finish();
    // 2→1 and 1→3 exist; the composed 2→3 exceeded the closure budget.
    let walking = Transfers::from_edges(
        4,
        &[
            (StopIdx(1), StopIdx(3), 10, 10.0),
            (StopIdx(2), StopIdx(1), 10, 10.0),
        ],
    )
    .unwrap();
    let mut carriage = Transfers::from_edges(4, &[]).unwrap();
    carriage.mark_unclosed();
    let inputs = CarriageInputs {
        timetable: &timetable,
        walking: &walking,
        carriage: &carriage,
        ride_edge: &[],
        carrying_mask: &[false, false],
        park_mask: &[false; 4],
        active_services: &[true],
        active_services_previous: &[],
    };
    let result = search(&inputs, &request(vec![], vec![(StopIdx(0), 0)]));
    // Stop 1's label is the walk from stop 2 (60 < 100); stop 3 keeps
    // the slow ride's legal walk-out (110).
    assert_eq!(result.arrivals(FREE)[1], Some(60));
    assert_eq!(result.arrivals(FREE)[3], Some(110));
    let (round, _) = result.best_round(FREE, StopIdx(3)).unwrap();
    let legs = result.reconstruct(&timetable, FREE, round, StopIdx(3));
    assert_eq!(
        legs,
        vec![
            CarriageLeg::Access {
                plane: FREE,
                to_stop: StopIdx(0),
                arrival: 0,
            },
            CarriageLeg::Transit {
                trip: TripIdx(0),
                board_stop: StopIdx(0),
                alight_stop: StopIdx(1),
                board_position: 0,
                alight_position: 1,
                board_time: 10,
                alight_time: 100,
                bike_aboard: false,
            },
            CarriageLeg::Transfer {
                from_stop: StopIdx(1),
                to_stop: StopIdx(3),
                departure: 100,
                arrival: 110,
                ride: false,
            },
        ]
    );
}

/// A closure-derived Free seed shadows a slower park at the same stop:
/// the seed may not walk again (its composition was budget-excluded),
/// but the shadowed park must — the only legal chain onward.
#[test]
fn shadowed_parks_still_walk_out() {
    let mut builder = TimetableBuilder::new(4);
    let line = builder.add_pattern(&[StopIdx(0), StopIdx(3)], 0).unwrap();
    builder
        .add_trip(line, vec![time(1_000), time(2_000)], 0, 0)
        .unwrap();
    let timetable = builder.finish();
    // 0→1 and 1→2 exist; the composed 0→2 exceeded the closure budget.
    let walking = Transfers::from_edges(
        4,
        &[
            (StopIdx(0), StopIdx(1), 10, 10.0),
            (StopIdx(1), StopIdx(2), 10, 10.0),
        ],
    )
    .unwrap();
    let mut carriage = Transfers::from_edges(4, &[]).unwrap();
    carriage.mark_unclosed();
    let inputs = CarriageInputs {
        timetable: &timetable,
        walking: &walking,
        carriage: &carriage,
        ride_edge: &[],
        carrying_mask: &[false],
        park_mask: &[false, true, false, false],
        active_services: &[true],
        active_services_previous: &[],
    };
    // The Free seed at 1 (10, via the closed reduction) shadows the
    // cycled park (50); stop 2 is reachable only by park-then-walk.
    let result = search(
        &inputs,
        &request(
            vec![(StopIdx(1), 50)],
            vec![(StopIdx(0), 0), (StopIdx(1), 10)],
        ),
    );
    assert_eq!(result.arrivals(FREE)[1], Some(10));
    assert_eq!(result.arrivals(FREE)[2], Some(60));
    let (round, _) = result.best_round(FREE, StopIdx(2)).unwrap();
    let legs = result.reconstruct(&timetable, FREE, round, StopIdx(2));
    assert_eq!(
        legs,
        vec![
            CarriageLeg::Access {
                plane: CARRYING,
                to_stop: StopIdx(1),
                arrival: 50,
            },
            CarriageLeg::Park {
                stop: StopIdx(1),
                at: 50,
            },
            CarriageLeg::Transfer {
                from_stop: StopIdx(1),
                to_stop: StopIdx(2),
                departure: 50,
                arrival: 60,
                ride: false,
            },
        ]
    );
}

/// A round's Free transit arrival shadowed by an earlier walk label:
/// the fresh ride resets the one-transfer allowance, so its walk-out
/// must still relax (the shadower may not walk again), and the chain
/// reconstructs through the inline ride.
#[test]
fn shadowed_free_transit_arrivals_still_walk() {
    let mut builder = TimetableBuilder::new(4);
    let first = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let second = builder.add_pattern(&[StopIdx(1), StopIdx(2)], 1).unwrap();
    builder
        .add_trip(first, vec![time(10), time(20)], 0, 0)
        .unwrap();
    builder
        .add_trip(second, vec![time(25), time(100)], 1, 0)
        .unwrap();
    let timetable = builder.finish();
    // 1→2 and 2→3 exist; the composed 1→3 exceeded the closure budget.
    let walking = Transfers::from_edges(
        4,
        &[
            (StopIdx(1), StopIdx(2), 10, 10.0),
            (StopIdx(2), StopIdx(3), 10, 10.0),
        ],
    )
    .unwrap();
    let mut carriage = Transfers::from_edges(4, &[]).unwrap();
    carriage.mark_unclosed();
    let inputs = CarriageInputs {
        timetable: &timetable,
        walking: &walking,
        carriage: &carriage,
        ride_edge: &[],
        carrying_mask: &[false, false],
        park_mask: &[false; 4],
        active_services: &[true],
        active_services_previous: &[],
    };
    let result = search(&inputs, &request(vec![], vec![(StopIdx(0), 0)]));
    // Round 1: ride to 1 (20), walk to 2 (30). Round 2: the second
    // ride arrives at 2 at 100 — shadowed by the walked 30, but its
    // walk-out is the only legal chain to 3.
    assert_eq!(result.arrivals(FREE)[2], Some(30));
    assert_eq!(result.arrivals(FREE)[3], Some(110));
    let (round, _) = result.best_round(FREE, StopIdx(3)).unwrap();
    let legs = result.reconstruct(&timetable, FREE, round, StopIdx(3));
    assert_eq!(
        legs,
        vec![
            CarriageLeg::Access {
                plane: FREE,
                to_stop: StopIdx(0),
                arrival: 0,
            },
            CarriageLeg::Transit {
                trip: TripIdx(0),
                board_stop: StopIdx(0),
                alight_stop: StopIdx(1),
                board_position: 0,
                alight_position: 1,
                board_time: 10,
                alight_time: 20,
                bike_aboard: false,
            },
            CarriageLeg::Transit {
                trip: TripIdx(1),
                board_stop: StopIdx(1),
                alight_stop: StopIdx(2),
                board_position: 0,
                alight_position: 1,
                board_time: 25,
                alight_time: 100,
                bike_aboard: false,
            },
            CarriageLeg::Transfer {
                from_stop: StopIdx(2),
                to_stop: StopIdx(3),
                departure: 100,
                arrival: 110,
                ride: false,
            },
        ]
    );
}
