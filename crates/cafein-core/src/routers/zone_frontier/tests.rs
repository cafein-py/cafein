use super::*;
use crate::fares::{FareLeg, ZoneProduct};
use crate::timetable::{StopTime, TimetableBuilder};

fn time(at: u32) -> StopTime {
    StopTime {
        arrival: at,
        departure: at,
    }
}

fn product(price: f64, zones: &[u32], duration: f64) -> ZoneProduct {
    ZoneProduct {
        price,
        zones: zones.iter().fold(0u128, |mask, &zone| mask | (1 << zone)),
        duration,
        transfers: u32::MAX,
    }
}

fn request(access: Vec<(StopIdx, u32)>, max_transfers: u8) -> Request {
    Request {
        departure: 0,
        access,
        egress: Vec::new(),
        active_services: vec![true],
        active_services_previous: Vec::new(),
        max_transfers,
        exclusions: None,
    }
}

fn run(
    timetable: &Timetable,
    transfers: &Transfers,
    fares: &ZoneFares,
    request: &Request,
    destination: StopIdx,
    window: u32,
) -> Option<ZoneRow> {
    let inputs = ZoneFrontierInputs {
        timetable,
        transfers,
        fares,
        top_cutoff: 50.0,
        max_duration: None,
        departure_step: None,
        seed_walks: true,
    };
    zone_frontier(&inputs, request, &[vec![(destination, 0)]], window).0[0]
}

/// The counterexample that refuted mask dominance (Codex, 2026-08-12):
/// two labels reach the transfer stop at the same second, one having
/// paid less with an earlier-started ticket. The continuation boards
/// one second after the cheaper label's window died, so the "cheaper"
/// prefix needs a second ticket while the pricier prefix extends. The
/// engine must keep both and report 4.50, never 7.80.
#[test]
fn expiry_outlives_a_cheaper_prefix() {
    // Zones: A=0, C=1. Stops: 0=A-origin(A), 1=X(A), 2=C-origin(C),
    // 3=DEST(C). Products: AA' 3.30 covering {A} for 4800 s; AC 4.50
    // covering {A,C} for 5400 s.
    let mut builder = TimetableBuilder::new(4);
    let p1 = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let p2 = builder.add_pattern(&[StopIdx(2), StopIdx(1)], 1).unwrap();
    let p3 = builder.add_pattern(&[StopIdx(1), StopIdx(3)], 2).unwrap();
    builder
        .add_trip(p1, vec![time(0), time(5_400)], 0, 0)
        .unwrap();
    builder
        .add_trip(p2, vec![time(600), time(5_400)], 1, 0)
        .unwrap();
    builder
        .add_trip(p3, vec![time(5_401), time(6_000)], 2, 0)
        .unwrap();
    let timetable = builder.finish();
    let transfers = Transfers::empty(4);
    let fares = ZoneFares {
        stop_zone: vec![0, 0, 1, 1],
        products: vec![
            product(3.30, &[0], 4_800.0),
            product(4.50, &[0, 1], 5_400.0),
        ],
    };
    let row = run(
        &timetable,
        &transfers,
        &fares,
        &request(vec![(StopIdx(0), 0), (StopIdx(2), 0)], 7),
        StopIdx(3),
        1_200,
    )
    .expect("destination reachable");
    assert!((row.fare - 4.50).abs() < 1e-9, "fare was {}", row.fare);
}

/// Pair A's shape: a feeder alights in zone C, and the SAME trunk
/// trip is boardable at a zone-D stop (a short walk) or a zone-C stop
/// (a slightly longer walk) with identical downstream arrivals. The
/// same-trip riders must coexist — the ABCD extension boarded at D and
/// the ABC extension boarded at C — and the journey prices 4.50, not
/// 5.00.
#[test]
fn same_trip_riders_keep_the_cheaper_boarding() {
    // Zones: A=0, B=1, C=2, D=3. Stops: 0=O(C), 1=J(C), 2=D1(D),
    // 3=C1(C), 4=A1(A), 5=DEST(B).
    let mut builder = TimetableBuilder::new(6);
    let feeder = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let trunk = builder
        .add_pattern(&[StopIdx(2), StopIdx(3), StopIdx(4)], 1)
        .unwrap();
    let metro = builder.add_pattern(&[StopIdx(4), StopIdx(5)], 2).unwrap();
    builder
        .add_trip(feeder, vec![time(0), time(1_000)], 0, 0)
        .unwrap();
    builder
        .add_trip(trunk, vec![time(1_100), time(1_160), time(2_000)], 1, 0)
        .unwrap();
    builder
        .add_trip(metro, vec![time(2_100), time(2_400)], 2, 0)
        .unwrap();
    let timetable = builder.finish();
    // Walks from the junction reach both boardings of the trunk trip.
    let mut transfers = Transfers::from_edges(
        6,
        &[
            (StopIdx(1), StopIdx(2), 60, 60.0),
            (StopIdx(1), StopIdx(3), 160, 160.0),
        ],
    )
    .unwrap();
    transfers.mark_unclosed();
    let fares = ZoneFares {
        stop_zone: vec![2, 2, 3, 2, 0, 1],
        products: vec![
            product(3.30, &[0, 1], 4_800.0),
            product(4.50, &[0, 1, 2], 5_400.0),
            product(5.00, &[0, 1, 2, 3], 6_600.0),
        ],
    };
    let row = run(
        &timetable,
        &transfers,
        &fares,
        &request(vec![(StopIdx(0), 0)], 7),
        StopIdx(5),
        600,
    )
    .expect("destination reachable");
    assert!((row.fare - 4.50).abs() < 1e-9, "fare was {}", row.fare);
}

/// Pair B's shape: the cheaper journey needs more rides at the same
/// arrival. Fare dominance keeps it beside the fewer-rides label.
#[test]
fn a_cheaper_journey_with_more_rides_survives() {
    // Zones: B=0, C=1, D=2. Stops: 0=O(C), 1=H(D), 2=K1(C), 3=K2(C),
    // 4=DEST(B).
    let mut builder = TimetableBuilder::new(5);
    let fast1 = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let fast2 = builder.add_pattern(&[StopIdx(1), StopIdx(4)], 1).unwrap();
    let slow1 = builder.add_pattern(&[StopIdx(0), StopIdx(2)], 2).unwrap();
    let slow2 = builder.add_pattern(&[StopIdx(2), StopIdx(3)], 3).unwrap();
    let slow3 = builder.add_pattern(&[StopIdx(3), StopIdx(4)], 4).unwrap();
    builder
        .add_trip(fast1, vec![time(0), time(900)], 0, 0)
        .unwrap();
    builder
        .add_trip(fast2, vec![time(1_000), time(2_000)], 1, 0)
        .unwrap();
    builder
        .add_trip(slow1, vec![time(0), time(800)], 2, 0)
        .unwrap();
    builder
        .add_trip(slow2, vec![time(900), time(1_200)], 3, 0)
        .unwrap();
    builder
        .add_trip(slow3, vec![time(1_300), time(2_000)], 4, 0)
        .unwrap();
    let timetable = builder.finish();
    let transfers = Transfers::empty(5);
    let fares = ZoneFares {
        stop_zone: vec![1, 2, 1, 1, 0],
        products: vec![
            product(3.30, &[0, 1], 4_800.0),
            product(4.50, &[0, 1, 2], 6_000.0),
        ],
    };
    let row = run(
        &timetable,
        &transfers,
        &fares,
        &request(vec![(StopIdx(0), 0)], 7),
        StopIdx(4),
        600,
    )
    .expect("destination reachable");
    assert!((row.fare - 3.30).abs() < 1e-9, "fare was {}", row.fare);
    assert_eq!(row.rides, 3);
}

/// A boarding past the active ticket's window buys again: the chain
/// prices as two tickets, and agrees with `ZoneFares::price` on the
/// same legs.
#[test]
fn expired_windows_chain_a_second_ticket() {
    // Zones: A=0, B=1. Stops: 0=O(A), 1=M(B), 2=DEST(B).
    let mut builder = TimetableBuilder::new(3);
    let first = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let second = builder.add_pattern(&[StopIdx(1), StopIdx(2)], 1).unwrap();
    builder
        .add_trip(first, vec![time(0), time(500)], 0, 0)
        .unwrap();
    builder
        .add_trip(second, vec![time(2_000), time(2_500)], 1, 0)
        .unwrap();
    let timetable = builder.finish();
    let transfers = Transfers::empty(3);
    let fares = ZoneFares {
        stop_zone: vec![0, 1, 1],
        products: vec![product(3.30, &[0, 1], 1_000.0)],
    };
    let row = run(
        &timetable,
        &transfers,
        &fares,
        &request(vec![(StopIdx(0), 0)], 7),
        StopIdx(2),
        300,
    )
    .expect("destination reachable");
    assert!((row.fare - 6.60).abs() < 1e-9, "fare was {}", row.fare);
    let dp = fares.price(&[
        FareLeg {
            route: 0,
            board_stop: 0,
            alight_stop: 1,
            board_time: 0,
        },
        FareLeg {
            route: 1,
            board_stop: 1,
            alight_stop: 2,
            board_time: 2_000,
        },
    ]);
    assert!(
        (row.fare - dp).abs() < 1e-9,
        "engine {} vs DP {}",
        row.fare,
        dp
    );
}

/// A zone-less origin can never board, but it can walk: the seed
/// relaxes one bounded transfer before the first boarding, fare free
/// and still unboarded, and the journey prices as if boarded at the
/// neighbour.
#[test]
fn zone_less_origins_walk_to_their_boarding() {
    // Zones: A=0; stop 0 has no usable zone. Stops: 0=O(none),
    // 1=S1(A), 2=DEST(A).
    let mut builder = TimetableBuilder::new(3);
    let line = builder.add_pattern(&[StopIdx(1), StopIdx(2)], 0).unwrap();
    builder
        .add_trip(line, vec![time(600), time(1_200)], 0, 0)
        .unwrap();
    let timetable = builder.finish();
    let mut transfers = Transfers::from_edges(3, &[(StopIdx(0), StopIdx(1), 120, 120.0)]).unwrap();
    transfers.mark_unclosed();
    let fares = ZoneFares {
        stop_zone: vec![crate::fares::NO_FARE, 0, 0],
        products: vec![product(3.30, &[0], 4_800.0)],
    };
    let row = run(
        &timetable,
        &transfers,
        &fares,
        &request(vec![(StopIdx(0), 0)], 7),
        StopIdx(2),
        1_200,
    )
    .expect("destination reachable through the seed walk");
    assert!((row.fare - 3.30).abs() < 1e-9, "fare was {}", row.fare);
    assert_eq!(row.rides, 1);
}

/// Brute-force oracle: enumerate every journey up to `max_rides` —
/// boardings chained by same-stop arrival or one bounded walk, a seed
/// walk before the first boarding, a terminal walk after the last
/// alight — price each with `ZoneFares::price`, and fold exactly as
/// the products do.
struct Oracle<'a> {
    timetable: &'a Timetable,
    transfers: &'a Transfers,
    fares: &'a ZoneFares,
    journeys: Vec<(u32, u32, u8, f64)>,
}

impl<'a> Oracle<'a> {
    fn enumerate(
        timetable: &'a Timetable,
        transfers: &'a Transfers,
        fares: &'a ZoneFares,
        origin: StopIdx,
        destination: StopIdx,
        max_rides: u8,
    ) -> Vec<(u32, u32, u8, f64)> {
        let mut oracle = Oracle {
            timetable,
            transfers,
            fares,
            journeys: Vec::new(),
        };
        // Seed positions: the origin at zero offset, plus one walk.
        let mut seeds = vec![(origin, 0u32)];
        for edge in transfers.from_stop(origin) {
            seeds.push((edge.to, edge.duration));
        }
        for &(stop, offset) in &seeds {
            oracle.extend(
                stop,
                offset,
                None,
                offset,
                &mut Vec::new(),
                destination,
                max_rides,
            );
        }
        oracle.journeys
    }

    /// Recurse from `stop`, reachable `ready` seconds into the clock
    /// (for the seed positions, the walk offset itself — boarding at
    /// `t` then means leaving the origin at `t − offset`). `leave` is
    /// the journey's origin-leave time once the first boarding fixed
    /// it; `offset` is the seed walk, read only at that first
    /// boarding.
    #[allow(clippy::too_many_arguments)]
    fn extend(
        &mut self,
        stop: StopIdx,
        ready: u32,
        leave: Option<u32>,
        offset: u32,
        legs: &mut Vec<FareLeg>,
        destination: StopIdx,
        max_rides: u8,
    ) {
        if legs.len() as u8 >= max_rides {
            return;
        }
        for pattern_stop in self.timetable.patterns_at_stop(stop) {
            let stops = self.timetable.pattern_stops(pattern_stop.pattern);
            let position = pattern_stop.position as usize;
            if position + 1 == stops.len() {
                continue;
            }
            for trip in self.timetable.pattern_trips(pattern_stop.pattern) {
                let times = self.timetable.trip_stop_times(trip);
                let board_time = times[position].departure;
                if board_time < ready {
                    continue;
                }
                let journey_leave = leave.unwrap_or(board_time - offset);
                for alight in position + 1..stops.len() {
                    legs.push(FareLeg {
                        route: 0,
                        board_stop: stop.0,
                        alight_stop: stops[alight].0,
                        board_time,
                    });
                    let arrival = times[alight].arrival;
                    self.finish(stops[alight], arrival, journey_leave, legs, destination);
                    self.extend(
                        stops[alight],
                        arrival,
                        Some(journey_leave),
                        0,
                        legs,
                        destination,
                        max_rides,
                    );
                    for edge in self.transfers.from_stop(stops[alight]) {
                        self.extend(
                            edge.to,
                            arrival + edge.duration,
                            Some(journey_leave),
                            0,
                            legs,
                            destination,
                            max_rides,
                        );
                    }
                    legs.pop();
                }
            }
        }
    }

    /// Record the journey when `stop` is the destination or one walk
    /// from it.
    fn finish(
        &mut self,
        stop: StopIdx,
        arrival: u32,
        leave: u32,
        legs: &[FareLeg],
        destination: StopIdx,
    ) {
        let fare = self.fares.price(legs);
        if !fare.is_finite() {
            return;
        }
        if stop == destination {
            self.journeys.push((leave, arrival, legs.len() as u8, fare));
        }
        for edge in self.transfers.from_stop(stop) {
            if edge.to == destination {
                self.journeys
                    .push((leave, arrival + edge.duration, legs.len() as u8, fare));
            }
        }
    }
}

/// Every fixture-tariff combination: the engine's cheapest fold and
/// cutoff rows equal the oracle's folds over the enumerated journeys.
#[test]
fn the_engine_matches_the_enumerator_oracle() {
    // Network: 5 stops in zones A=0/B=1; walks 1<->2 (both ways) and
    // 3->4; three patterns with two trips each, giving chained,
    // walked, and waiting journeys.
    let mut builder = TimetableBuilder::new(5);
    let p1 = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let p2 = builder.add_pattern(&[StopIdx(2), StopIdx(3)], 1).unwrap();
    let p3 = builder.add_pattern(&[StopIdx(1), StopIdx(4)], 2).unwrap();
    builder
        .add_trip(p1, vec![time(100), time(500)], 0, 0)
        .unwrap();
    builder
        .add_trip(p1, vec![time(1_500), time(1_900)], 1, 0)
        .unwrap();
    builder
        .add_trip(p2, vec![time(700), time(1_200)], 2, 0)
        .unwrap();
    builder
        .add_trip(p2, vec![time(2_400), time(2_900)], 3, 0)
        .unwrap();
    builder
        .add_trip(p3, vec![time(1_400), time(2_000)], 4, 0)
        .unwrap();
    builder
        .add_trip(p3, vec![time(3_200), time(3_800)], 5, 0)
        .unwrap();
    let timetable = builder.finish();
    let mut transfers = Transfers::from_edges(
        5,
        &[
            (StopIdx(1), StopIdx(2), 90, 90.0),
            (StopIdx(2), StopIdx(1), 90, 90.0),
            (StopIdx(3), StopIdx(4), 60, 60.0),
        ],
    )
    .unwrap();
    transfers.mark_unclosed();

    let tariffs: Vec<(&str, ZoneFares)> = vec![
        (
            "subadditive",
            ZoneFares {
                stop_zone: vec![0, 0, 1, 1, 1],
                products: vec![
                    product(2.0, &[0], 4_000.0),
                    product(2.0, &[1], 4_000.0),
                    product(3.0, &[0, 1], 4_000.0),
                ],
            },
        ),
        (
            "non_subadditive",
            ZoneFares {
                stop_zone: vec![0, 0, 1, 1, 1],
                products: vec![
                    product(1.0, &[0], 10_000.0),
                    product(1.0, &[1], 10_000.0),
                    product(3.0, &[0, 1], 10_000.0),
                ],
            },
        ),
        (
            "binding_windows",
            ZoneFares {
                stop_zone: vec![0, 0, 1, 1, 1],
                products: vec![
                    product(2.0, &[0, 1], 900.0),
                    product(5.0, &[0, 1], 10_000.0),
                ],
            },
        ),
        (
            // A legal zero-price product: partial coverage, so paid
            // journeys still exist beside free ones — the closed-slot
            // sentinel must not conflate "free" with "settled".
            "zero_fare_zone",
            ZoneFares {
                stop_zone: vec![0, 0, 1, 1, 1],
                products: vec![
                    product(0.0, &[0], 10_000.0),
                    product(2.0, &[0, 1], 10_000.0),
                ],
            },
        ),
        (
            "binding_transfers",
            ZoneFares {
                stop_zone: vec![0, 0, 1, 1, 1],
                products: vec![
                    ZoneProduct {
                        price: 2.0,
                        zones: 0b11,
                        duration: 10_000.0,
                        transfers: 1,
                    },
                    product(5.0, &[0, 1], 10_000.0),
                ],
            },
        ),
    ];

    let window = 3_000u32;
    let cutoffs = [2.0, 3.0, 4.0, 6.0, 12.0];
    for (name, fares) in &tariffs {
        for origin in [StopIdx(0), StopIdx(2)] {
            let mut singles: Vec<Option<(f64, u32, u8)>> = Vec::new();
            for destination in [StopIdx(3), StopIdx(4)] {
                let journeys =
                    Oracle::enumerate(&timetable, &transfers, fares, origin, destination, 4);
                let request = request(vec![(origin, 0)], 3);
                let inputs = ZoneFrontierInputs {
                    timetable: &timetable,
                    transfers: &transfers,
                    fares,
                    top_cutoff: 12.0,
                    max_duration: None,
                    departure_step: None,
                    seed_walks: true,
                };
                // The oracle's window filter mirrors the profile: the
                // journey's leave time falls inside the window.
                let in_window: Vec<_> = journeys
                    .iter()
                    .filter(|&&(leave, _, _, _)| leave < window)
                    .collect();
                let oracle_cheapest = in_window
                    .iter()
                    .map(|&&(leave, arrival, rides, fare)| (fare, arrival - leave, rides))
                    .min_by(|a, b| a.partial_cmp(b).unwrap());
                let (rows, arena) =
                    zone_frontier(&inputs, &request, &[vec![(destination, 0)]], window);
                let engine = rows[0].map(|row| (row.fare, row.travel_time, row.rides));
                // Reconstruction self-check: the winner's chain must
                // re-price to its own fare through the DP, cent for
                // cent, walks contributing no legs.
                if let Some(row) = rows[0] {
                    let (leave, _seed_stop, steps) = chain(&arena, row.node);
                    let mut legs = Vec::new();
                    for step in &steps {
                        if let ChainStep::Ride {
                            trip,
                            day_offset,
                            board_position,
                            alight_position,
                        } = *step
                        {
                            let pattern_stops =
                                timetable.pattern_stops(timetable.trip_pattern(trip));
                            let times = timetable.trip_stop_times(trip);
                            legs.push(FareLeg {
                                route: 0,
                                board_stop: pattern_stops[board_position as usize].0,
                                alight_stop: pattern_stops[alight_position as usize].0,
                                board_time: times[board_position as usize]
                                    .departure
                                    .saturating_sub(day_offset),
                            });
                        }
                    }
                    let repriced = fares.price(&legs);
                    assert!(
                        (repriced - row.fare).abs() < 1e-9,
                        "chain re-prices to {repriced}, row says {} \
                         (tariff {name} {origin:?}->{destination:?})",
                        row.fare
                    );
                    // The leave time anchors the reported travel time.
                    assert!(leave < window);
                }
                assert_eq!(
                    engine, oracle_cheapest,
                    "cheapest mismatch: tariff {name} {origin:?}->{destination:?}"
                );
                singles.push(engine);
                // The cap invariant: a journey makes at most
                // max_transfers + 1 boardings and each buys at most
                // one ticket, so no enumerated fare exceeds that many
                // dearest products — the matrix staircase's hard cap.
                let dearest = fares
                    .products
                    .iter()
                    .map(|product| product.price)
                    .fold(0.0f64, f64::max);
                for &&(_, _, _, fare) in &in_window {
                    assert!(fare <= 4.0 * dearest + 1e-9);
                }
                let cut_rows = zone_frontier_cutoffs(
                    &inputs,
                    &request,
                    &[vec![(destination, 0)]],
                    window,
                    &cutoffs,
                );
                for (slot, &cutoff) in cutoffs.iter().enumerate() {
                    let oracle_row = in_window
                        .iter()
                        .filter(|&&&(_, _, _, fare)| fare <= cutoff + 1e-9)
                        .map(|&&(leave, arrival, rides, fare)| (arrival - leave, fare, rides))
                        .min_by(|a, b| a.partial_cmp(b).unwrap());
                    let engine_row =
                        cut_rows[0][slot].map(|row| (row.travel_time, row.fare, row.rides));
                    assert_eq!(
                        engine_row, oracle_row,
                        "cutoff {cutoff} mismatch: tariff {name} {origin:?}->{destination:?}"
                    );
                }
            }
            // Multi-slot parity: both destinations in one search must
            // equal the single-slot answers — the shared money bound
            // and the deadline index may only discard what cannot win.
            let request = request(vec![(origin, 0)], 3);
            let inputs = ZoneFrontierInputs {
                timetable: &timetable,
                transfers: &transfers,
                fares,
                top_cutoff: 12.0,
                max_duration: None,
                departure_step: None,
                seed_walks: true,
            };
            let slots = [vec![(StopIdx(3), 0)], vec![(StopIdx(4), 0)]];
            let (multi, _) = zone_frontier(&inputs, &request, &slots, window);
            for (slot, single) in singles.iter().enumerate() {
                let row = multi[slot].map(|row| (row.fare, row.travel_time, row.rides));
                assert_eq!(
                    row, *single,
                    "multi-slot mismatch: tariff {name} {origin:?}"
                );
            }
            // Warm bounds at the optima are true upper bounds, so the
            // rows stay exact; seeding the optima on top must yield
            // journeys as good as the seeds, never a fabricated win.
            let warm: Vec<f64> = singles
                .iter()
                .map(|single| single.map_or(0.0, |(fare, _, _)| fare))
                .collect();
            let seeds: Vec<Option<u32>> = singles
                .iter()
                .map(|single| single.map(|(_, travel_time, _)| travel_time))
                .collect();
            let (warmed, _) =
                zone_frontier_warm(&inputs, &request, &slots, window, warm.clone(), &[]);
            for (slot, single) in singles.iter().enumerate() {
                if single.is_none() {
                    continue;
                }
                let row = warmed[slot].map(|row| (row.fare, row.travel_time, row.rides));
                assert_eq!(row, *single, "warm mismatch: tariff {name} {origin:?}");
            }
            let (seeded, _) =
                zone_frontier_warm(&inputs, &request, &slots, window, warm.clone(), &seeds);
            for (slot, single) in singles.iter().enumerate() {
                let Some((fare, travel_time, _)) = *single else {
                    continue;
                };
                let row = seeded[slot].expect("a seeded slot always answers");
                assert!(
                    (row.fare, row.travel_time) <= (fare, travel_time),
                    "a seed outdid the engine: tariff {name} {origin:?}"
                );
                if row.node == SEED_NODE {
                    assert_eq!((row.fare, row.travel_time), (fare, travel_time));
                }
            }
            // A warm bound strictly under a slot's optimum makes that
            // slot come back empty — never a wrong row. Detectable
            // emptiness is what the matrix staircase escalates on.
            let starved: Vec<f64> = warm.iter().map(|&bound| bound - 0.5).collect();
            let (rows, _) = zone_frontier_warm(&inputs, &request, &slots, window, starved, &[]);
            for (slot, single) in singles.iter().enumerate() {
                if single.is_some() {
                    assert!(
                        rows[slot].is_none(),
                        "a starved bound produced a row: tariff {name} {origin:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn point_egress_never_chains_onto_a_transfer_walk() {
    // Ride 0->1, walk 1->2, then a point egress link at 2: two street
    // walks in a row. Point mode must not fold it; the stop query's
    // own single footpath still does.
    let mut builder = TimetableBuilder::new(3);
    let p1 = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    builder
        .add_trip(p1, vec![time(100), time(400)], 0, 0)
        .unwrap();
    let timetable = builder.finish();
    let mut transfers = Transfers::from_edges(3, &[(StopIdx(1), StopIdx(2), 60, 60.0)]).unwrap();
    transfers.mark_unclosed();
    let fares = ZoneFares {
        stop_zone: vec![0, 0, 0],
        products: vec![product(2.0, &[0], 10_000.0)],
    };
    let request = request(vec![(StopIdx(0), 0)], 3);
    let point = ZoneFrontierInputs {
        timetable: &timetable,
        transfers: &transfers,
        fares: &fares,
        top_cutoff: 10.0,
        max_duration: None,
        departure_step: None,
        seed_walks: false,
    };
    let (rows, _) = zone_frontier(&point, &request, &[vec![(StopIdx(2), 120)]], 600);
    assert!(
        rows[0].is_none(),
        "a point egress folded onto a transfer walk"
    );
    let stop = ZoneFrontierInputs {
        seed_walks: true,
        ..point
    };
    let (rows, _) = zone_frontier(&stop, &request, &[vec![(StopIdx(2), 0)]], 600);
    assert_eq!(
        rows[0].map(|row| row.fare),
        Some(2.0),
        "the stop query's single footpath must still fold"
    );
}
