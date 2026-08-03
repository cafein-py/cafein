use super::*;
use crate::routers::router::Request;
use crate::timetable::{StopTime, TimetableBuilder};

fn time(at: u32) -> StopTime {
    StopTime {
        arrival: at,
        departure: at,
    }
}

/// Two parallel services 0→1: an expensive fast bus line (route 0)
/// and a cheap slow one (route 1), plus a walking row 1→2.
fn fixture() -> (Timetable, Transfers, RuleFares) {
    let mut builder = TimetableBuilder::new(3);
    let fast = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let slow = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 1).unwrap();
    builder
        .add_trip(fast, vec![time(100), time(400)], 0, 0)
        .unwrap();
    builder
        .add_trip(fast, vec![time(700), time(1_000)], 1, 0)
        .unwrap();
    builder
        .add_trip(slow, vec![time(100), time(900)], 2, 0)
        .unwrap();
    let timetable = builder.finish();
    let transfers = Transfers::from_edges(3, &[(StopIdx(1), StopIdx(2), 60, 60.0)]).unwrap();
    let fares = RuleFares {
        route_type: vec![0, 1],
        route_fare: vec![5.0, 2.0],
        unlimited_transfers: vec![false, false],
        allow_same_route: vec![false, false],
        pair_fare: vec![f64::NAN; 4],
        max_discounted_transfers: 1,
        transfer_allowance: 3_600.0,
        fare_cap: f64::INFINITY,
    };
    (timetable, transfers, fares)
}

fn request(access: Vec<(StopIdx, u32)>) -> Request {
    Request {
        departure: 0,
        access,
        egress: Vec::new(),
        active_services: vec![true],
        active_services_previous: Vec::new(),
        max_transfers: 3,
        exclusions: None,
    }
}

/// A fare-blind engine would drop the cheap slow journey (later
/// arrival, same rides); the fare-state bags keep it, and the fold
/// hands each cutoff its own winner.
#[test]
fn cheaper_but_slower_journeys_survive_to_their_cutoff() {
    let (timetable, transfers, fares) = fixture();
    let inputs = FareFrontierInputs {
        timetable: &timetable,
        transfers: &transfers,
        fares: &fares,
        cutoffs: &[3.0, 6.0],
        max_duration: None,
    };
    let request = request(vec![(StopIdx(0), 0)]);
    let mut search = FareFrontierSearch::new(&inputs, &request, 1);
    search.pass(0, &[vec![(StopIdx(1), 0)]]);
    let arrivals = search.into_arrivals();
    let rows = fold_cutoffs(&arrivals[0], &[3.0, 6.0]);
    // Under 3.0 only the slow cheap line fits (arrives 900); under
    // 6.0 the fast expensive one wins (arrives 400).
    let low = rows[0].expect("the cheap journey fits the low cutoff");
    assert_eq!(low.travel_time, 900);
    assert!((low.fare - 2.0).abs() < 1e-9);
    let high = rows[1].expect("the fast journey fits the high cutoff");
    assert_eq!(high.travel_time, 400);
    assert!((high.fare - 5.0).abs() < 1e-9);
}

/// The walk-only chain reaches a walking-linked destination at fare
/// zero and never NaNs a cell. Access rows arrive closure-closed, as
/// the engine's contract states.
#[test]
fn walking_chains_price_zero() {
    let (timetable, transfers, fares) = fixture();
    let inputs = FareFrontierInputs {
        timetable: &timetable,
        transfers: &transfers,
        fares: &fares,
        cutoffs: &[3.0],
        max_duration: None,
    };
    let request = request(vec![(StopIdx(1), 30), (StopIdx(2), 90)]);
    let mut search = FareFrontierSearch::new(&inputs, &request, 1);
    search.pass(0, &[vec![(StopIdx(2), 0)]]);
    let arrivals = search.into_arrivals();
    let rows = fold_cutoffs(&arrivals[0], &[3.0]);
    let row = rows[0].expect("walking fits every cutoff");
    assert_eq!(row.travel_time, 90);
    assert_eq!(row.fare, 0.0);
    assert_eq!(row.rides, 0);
}

/// The duration cap drops journeys longer than the bound, exactly as
/// r5r's trip cap does.
#[test]
fn the_duration_cap_bounds_journeys() {
    let (timetable, transfers, fares) = fixture();
    let inputs = FareFrontierInputs {
        timetable: &timetable,
        transfers: &transfers,
        fares: &fares,
        cutoffs: &[6.0],
        max_duration: Some(500),
    };
    let request = request(vec![(StopIdx(0), 0)]);
    let mut search = FareFrontierSearch::new(&inputs, &request, 1);
    search.pass(0, &[vec![(StopIdx(1), 0)]]);
    let arrivals = search.into_arrivals();
    // Only the fast journey (400) fits the 500-second cap.
    assert_eq!(arrivals[0].len(), 1);
    assert_eq!(arrivals[0][0].arrival, 400);
}

/// A brute-force reference with **no fare-aware dominance at all**:
/// every journey (bounded rides, one closure walk after each ride) is
/// enumerated for every window departure, priced post hoc through the
/// journey pricer, and folded per cutoff. Divergence from the engine
/// means the bags or gates, not the contract.
#[allow(clippy::too_many_arguments)]
fn oracle(
    timetable: &Timetable,
    transfers: &Transfers,
    fares: &RuleFares,
    request: &Request,
    destinations: &[Vec<(StopIdx, u32)>],
    departures: &[u32],
    cutoffs: &[f64],
    max_duration: Option<u32>,
) -> Vec<Vec<Option<FrontierRow>>> {
    use crate::fares::FareLeg;

    #[derive(Clone)]
    struct Node {
        stop: StopIdx,
        time: u32,
        rides: u8,
        legs: Vec<FareLeg>,
        walked: bool,
    }

    let mut per_slot: Vec<Vec<Arrived>> = vec![Vec::new(); destinations.len()];
    for &departure in departures {
        let mut stack: Vec<Node> = request
            .access
            .iter()
            .map(|&(stop, seconds)| Node {
                stop,
                time: departure + seconds,
                rides: 0,
                legs: Vec::new(),
                walked: true,
            })
            .collect();
        while let Some(node) = stack.pop() {
            for (slot, egress) in destinations.iter().enumerate() {
                for &(stop, seconds) in egress {
                    if stop != node.stop {
                        continue;
                    }
                    let arrival = node.time + seconds;
                    if let Some(cap) = max_duration {
                        if arrival - departure > cap {
                            continue;
                        }
                    }
                    let fare = fares.price(&node.legs);
                    if fare.is_nan() {
                        continue;
                    }
                    per_slot[slot].push(Arrived {
                        departure,
                        arrival,
                        rides: node.rides,
                        fare,
                    });
                }
            }
            if (node.rides as usize) < request.max_transfers as usize + 1 {
                for pattern in 0..timetable.pattern_count() {
                    let pattern = crate::timetable::PatternIdx(pattern);
                    let stops = timetable.pattern_stops(pattern);
                    let route = timetable.pattern_route(pattern);
                    for board in 0..stops.len().saturating_sub(1) {
                        if stops[board] != node.stop {
                            continue;
                        }
                        let range = timetable.pattern_trip_range(pattern);
                        for trip in range.start..range.end {
                            let trip = TripIdx(trip);
                            if !request
                                .active_services
                                .get(timetable.trip_service(trip) as usize)
                                .copied()
                                .unwrap_or(false)
                            {
                                continue;
                            }
                            let times = timetable.trip_stop_times(trip);
                            if times[board].departure < node.time {
                                continue;
                            }
                            for alight in board + 1..stops.len() {
                                let mut legs = node.legs.clone();
                                legs.push(FareLeg {
                                    route,
                                    board_stop: stops[board].0,
                                    alight_stop: stops[alight].0,
                                    board_time: times[board].departure,
                                });
                                stack.push(Node {
                                    stop: stops[alight],
                                    time: times[alight].arrival,
                                    rides: node.rides + 1,
                                    legs,
                                    walked: false,
                                });
                            }
                        }
                    }
                }
            }
            if !node.walked {
                for edge in transfers.from_stop(node.stop) {
                    stack.push(Node {
                        stop: edge.to,
                        time: node.time + edge.duration,
                        rides: node.rides,
                        legs: node.legs.clone(),
                        walked: true,
                    });
                }
            }
        }
    }
    per_slot
        .iter()
        .map(|arrivals| fold_cutoffs(arrivals, cutoffs))
        .collect()
}

/// A transfer-rich tariff: buses (route 0 on 0→1, route 1 on 1→2) at
/// 4.0 integrating to 6.0 inside a 900-second window, a direct slow
/// cheap line (route 2, 0→2 at 3.0), and a walking row 1→2.
fn transfer_fixture() -> (Timetable, Transfers, RuleFares) {
    let mut builder = TimetableBuilder::new(3);
    let first = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let second = builder.add_pattern(&[StopIdx(1), StopIdx(2)], 1).unwrap();
    let direct = builder.add_pattern(&[StopIdx(0), StopIdx(2)], 2).unwrap();
    builder
        .add_trip(first, vec![time(100), time(400)], 0, 0)
        .unwrap();
    builder
        .add_trip(first, vec![time(1_300), time(1_600)], 1, 0)
        .unwrap();
    builder
        .add_trip(second, vec![time(600), time(800)], 2, 0)
        .unwrap();
    builder
        .add_trip(second, vec![time(1_700), time(1_900)], 3, 0)
        .unwrap();
    builder
        .add_trip(direct, vec![time(200), time(2_400)], 4, 0)
        .unwrap();
    let timetable = builder.finish();
    let transfers = Transfers::from_edges(3, &[(StopIdx(1), StopIdx(2), 700, 700.0)]).unwrap();
    let fares = RuleFares {
        route_type: vec![0, 0, 1],
        route_fare: vec![4.0, 4.0, 3.0],
        unlimited_transfers: vec![false, false],
        allow_same_route: vec![false, false],
        pair_fare: vec![6.0, f64::NAN, f64::NAN, f64::NAN],
        max_discounted_transfers: 1,
        transfer_allowance: 900.0,
        fare_cap: f64::INFINITY,
    };
    (timetable, transfers, fares)
}

/// A tariff whose dominance gates all bite: a scarce discount budget
/// with a short window (freshness must not compare), a weak and a
/// strong integration (previous full fares diverge), and a harmful
/// variant whose pair total exceeds the fulls it replaces.
fn gated_fixture(harmful: bool) -> (Timetable, Transfers, RuleFares) {
    let mut builder = TimetableBuilder::new(3);
    let first = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let second = builder.add_pattern(&[StopIdx(1), StopIdx(2)], 1).unwrap();
    for (departure, arrival, service) in [(100, 300, 0), (500, 700, 1), (1_500, 1_700, 2)] {
        builder
            .add_trip(first, vec![time(departure), time(arrival)], service, 0)
            .unwrap();
    }
    // A cheap slow bus on the same leg: labels at stop 1 then carry
    // the same cutoff class with different exact fares and different
    // previous full fares, which reorder after the next integration.
    let cheap = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 2).unwrap();
    for (departure, arrival, service) in [(150, 750, 6), (900, 1_450, 7)] {
        builder
            .add_trip(cheap, vec![time(departure), time(arrival)], service, 0)
            .unwrap();
    }
    for (departure, arrival, service) in [(800, 1_000, 3), (1_100, 1_300, 4), (1_900, 2_100, 5)] {
        builder
            .add_trip(second, vec![time(departure), time(arrival)], service, 0)
            .unwrap();
    }
    let timetable = builder.finish();
    let transfers = Transfers::from_edges(3, &[]).unwrap();
    let fares = RuleFares {
        route_type: vec![0, 1, 0],
        route_fare: vec![4.8, 4.5, 2.0],
        unlimited_transfers: vec![false, false],
        allow_same_route: vec![false, false],
        pair_fare: vec![if harmful { 12.0 } else { 9.5 }, 5.0, f64::NAN, f64::NAN],
        max_discounted_transfers: 1,
        transfer_allowance: 600.0,
        fare_cap: f64::INFINITY,
    };
    (timetable, transfers, fares)
}

/// The engine matches the fare-blind exhaustive oracle cell for cell
/// across cutoff sets, duration caps, and window departures — on the
/// transfer-rich tariff and on both gated ones, where later trips'
/// fresher windows, scarce discounts, and harmful pairs all bite.
#[test]
fn engine_matches_the_exhaustive_oracle() {
    for (timetable, transfers, fares) in [
        transfer_fixture(),
        gated_fixture(false),
        gated_fixture(true),
    ] {
        sweep(&timetable, &transfers, &fares);
    }
}

fn sweep(timetable: &Timetable, transfers: &Transfers, fares: &RuleFares) {
    let service_count = {
        let mut top = 0;
        for trip in 0..timetable.trip_count() {
            top = top.max(timetable.trip_service(TripIdx(trip)) as usize + 1);
        }
        top
    };
    let request = Request {
        departure: 0,
        access: vec![(StopIdx(0), 0)],
        egress: Vec::new(),
        active_services: vec![true; service_count],
        active_services_previous: Vec::new(),
        max_transfers: 3,
        exclusions: None,
    };
    let destinations = vec![vec![(StopIdx(1), 0)], vec![(StopIdx(2), 0)]];
    for cutoffs in [vec![3.0, 5.0, 8.0], vec![6.0], vec![2.0], vec![9.0, 12.5]] {
        for max_duration in [None, Some(1_000), Some(2_500)] {
            let inputs = FareFrontierInputs {
                timetable,
                transfers,
                fares,
                cutoffs: &cutoffs,
                max_duration,
            };
            let departures =
                crate::routers::raptor::departure_candidates(timetable, &request, 2_000);
            let mut search = FareFrontierSearch::new(&inputs, &request, destinations.len());
            for &departure in &departures {
                search.pass(departure, &destinations);
            }
            let engine: Vec<Vec<Option<FrontierRow>>> = search
                .into_arrivals()
                .iter()
                .map(|arrivals| fold_cutoffs(arrivals, &cutoffs))
                .collect();
            let reference = oracle(
                timetable,
                transfers,
                fares,
                &request,
                &destinations,
                &departures,
                &cutoffs,
                max_duration,
            );
            assert_eq!(
                engine, reference,
                "cutoffs {cutoffs:?}, cap {max_duration:?}"
            );
        }
    }
}
