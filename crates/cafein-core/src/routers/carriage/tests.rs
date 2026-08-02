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
