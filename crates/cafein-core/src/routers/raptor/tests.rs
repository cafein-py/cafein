use super::*;
use crate::geometry::DistanceProvenance;
use crate::timetable::{StopTime, TimetableBuilder};

fn time(at: u32) -> StopTime {
    StopTime {
        arrival: at,
        departure: at,
    }
}

/// Five stops. Pattern A rides 0→1→2, pattern B rides 1→3, and stop 2
/// has a 50-second footpath to stop 4, from which pattern C rides 4→3.
fn network() -> (Timetable, Transfers) {
    let mut builder = TimetableBuilder::new(5);
    let a = builder
        .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 0)
        .unwrap();
    let b = builder.add_pattern(&[StopIdx(1), StopIdx(3)], 1).unwrap();
    let c = builder.add_pattern(&[StopIdx(4), StopIdx(3)], 2).unwrap();
    // Two trips on A so boarding must pick the second when the first
    // has already left.
    builder
        .add_trip(a, vec![time(100), time(200), time(300)], 0, 0)
        .unwrap();
    builder
        .add_trip(a, vec![time(700), time(800), time(900)], 1, 0)
        .unwrap();
    // B departs stop 1 at 250, reachable from A's first trip (arr 200).
    builder
        .add_trip(b, vec![time(250), time(400)], 2, 0)
        .unwrap();
    // A later B trip on an inactive service would be wrong to board.
    builder
        .add_trip(b, vec![time(500), time(600)], 3, 1)
        .unwrap();
    // C departs stop 4 at 400; stop 4 is only reachable by footpath
    // from stop 2 (arr 300 + 50).
    builder
        .add_trip(c, vec![time(400), time(1000)], 4, 0)
        .unwrap();
    let timetable = builder.finish();
    let transfers = Transfers::from_edges(5, &[(StopIdx(2), StopIdx(4), 50, 50.0)]).unwrap();
    (timetable, transfers)
}

fn request(from: StopIdx, to: StopIdx, departure: u32) -> Request {
    Request {
        departure,
        access: vec![(from, 0)],
        egress: vec![(to, 0)],
        active_services: vec![true, false],
        active_services_previous: Vec::new(),
        max_transfers: 3,
        exclusions: None,
    }
}

#[test]
fn times_overflowing_the_representable_range_are_unreachable() {
    // Access additions near the u32 limit must neither wrap nor
    // collide with the UNREACHED sentinel; such paths simply stay
    // unreachable instead of producing bogus arrivals.
    let (timetable, transfers) = network();
    for departure in [u32::MAX - 5, u32::MAX - 10] {
        let mut nearly_out_of_time = request(StopIdx(0), StopIdx(3), departure);
        nearly_out_of_time.access = vec![(StopIdx(0), 10)];
        assert_eq!(
            Raptor.route(&timetable, &transfers, &nearly_out_of_time),
            Vec::new()
        );
    }
}

#[test]
fn window_percentiles_match_per_minute_runs() {
    // The windowed scan's samples must equal fresh single-departure
    // runs at every minute mark; percentiles follow nearest-rank.
    let (timetable, transfers) = network();
    let window = 600;
    let percentiles = [0.0, 50.0, 100.0];
    let mut request = request(StopIdx(0), StopIdx(3), 0);
    request.egress = Vec::new();
    let rows = Raptor.percentile_matrix(
        &timetable,
        &transfers,
        std::slice::from_ref(&request),
        window,
        &percentiles,
    );
    let stop_count = timetable.stop_count() as usize;
    for stop in 0..stop_count {
        let mut samples: Vec<u32> = (0..window / 60)
            .map(|step| {
                let mark = step * 60;
                let mut fresh = request.clone();
                fresh.departure = mark;
                match Raptor.one_to_all(&timetable, &transfers, &fresh)[stop] {
                    Some(arrival) => arrival - mark,
                    None => UNREACHED,
                }
            })
            .collect();
        samples.sort_unstable();
        for (at, &percentile) in percentiles.iter().enumerate() {
            assert_eq!(
                rows[0][stop * percentiles.len() + at],
                nearest_rank(&samples, percentile),
                "stop {stop} percentile {percentile}"
            );
        }
    }
    // The pointset variant joins the same samples over egress links.
    let egress = vec![vec![(StopIdx(3), 30, 25.0), (StopIdx(4), 10, 8.0)]];
    let point_rows = Raptor.percentile_matrix_to_points(
        &timetable,
        &transfers,
        std::slice::from_ref(&request),
        &egress,
        window,
        &percentiles,
    );
    let mut samples: Vec<u32> = (0..window / 60)
        .map(|step| {
            let mark = step * 60;
            let mut fresh = request.clone();
            fresh.departure = mark;
            let arrivals = Raptor.one_to_all(&timetable, &transfers, &fresh);
            let mut best = UNREACHED;
            for &(stop, seconds, _) in &egress[0] {
                if let Some(at) = arrivals[stop.0 as usize] {
                    best = best.min(at + seconds);
                }
            }
            if best == UNREACHED {
                UNREACHED
            } else {
                best - mark
            }
        })
        .collect();
    samples.sort_unstable();
    for (at, &percentile) in percentiles.iter().enumerate() {
        assert_eq!(point_rows[0][at], nearest_rank(&samples, percentile));
    }
}

#[test]
fn cost_rows_aggregate_the_fastest_journey() {
    // Distances per trip: pattern A trips 1200 m over three stops,
    // B trips 800 m, C 2000 m; factors 10/10/20/20/30 g/pkm.
    let (timetable, transfers) = network();
    let geometry = TripGeometry::from_trips(
        &timetable,
        vec![
            (
                TripIdx(0),
                vec![0.0, 500.0, 1200.0],
                DistanceProvenance::CrowFly,
            ),
            (
                TripIdx(1),
                vec![0.0, 500.0, 1200.0],
                DistanceProvenance::CrowFly,
            ),
            (TripIdx(2), vec![0.0, 800.0], DistanceProvenance::CrowFly),
            (TripIdx(3), vec![0.0, 800.0], DistanceProvenance::CrowFly),
            (TripIdx(4), vec![0.0, 2000.0], DistanceProvenance::CrowFly),
        ],
    )
    .unwrap();
    let factors = [10.0, 10.0, 20.0, 20.0, 30.0];
    let inputs = CostInputs {
        geometry: &geometry,
        factors: &factors,
        leg_geometry: None,
        with_geometry: false,
        fares: None,
        rental: None,
        with_pieces: false,
    };
    let mut request = request(StopIdx(0), StopIdx(3), 0);
    request.egress = Vec::new();
    let rows = Raptor.cost_matrix(
        &timetable,
        &transfers,
        &inputs,
        std::slice::from_ref(&request),
        &[StopIdx(3), StopIdx(4)],
    );
    assert_eq!(rows.len(), 1);
    // To stop 3: ride A 0→1 (500 m, 10 g/pkm), ride B 1→3 (800 m,
    // 20 g/pkm), arriving 400 with no walking.
    let to_3 = &rows[0][0];
    assert_eq!((to_3.to, to_3.seconds, to_3.rides), (3, 400, 2));
    assert_eq!(to_3.transit_meters, 1300.0);
    assert_eq!(to_3.walk_meters, 0.0);
    assert!((to_3.emission_grams - 21.0).abs() < 1e-9);
    assert_eq!(to_3.geometry, None);
    // To stop 4: ride A 0→2 (1200 m), then the 50 m footpath.
    let to_4 = &rows[0][1];
    assert_eq!((to_4.to, to_4.seconds, to_4.rides), (4, 350, 1));
    assert_eq!(to_4.transit_meters, 1200.0);
    assert_eq!(to_4.walk_meters, 50.0);
    assert!((to_4.emission_grams - 12.0).abs() < 1e-9);
    // An unresolved factor (NaN) poisons only the affected row.
    let partial = [10.0, 10.0, f64::NAN, f64::NAN, 30.0];
    let inputs = CostInputs {
        factors: &partial,
        ..inputs
    };
    let rows = Raptor.cost_matrix(
        &timetable,
        &transfers,
        &inputs,
        std::slice::from_ref(&request),
        &[StopIdx(3), StopIdx(4)],
    );
    assert!(rows[0][0].emission_grams.is_nan());
    assert!((rows[0][1].emission_grams - 12.0).abs() < 1e-9);
}

#[test]
fn point_rows_join_over_egress_links() {
    let (timetable, transfers) = network();
    let geometry = TripGeometry::from_trips(
        &timetable,
        vec![
            (
                TripIdx(0),
                vec![0.0, 500.0, 1200.0],
                DistanceProvenance::CrowFly,
            ),
            (
                TripIdx(1),
                vec![0.0, 500.0, 1200.0],
                DistanceProvenance::CrowFly,
            ),
            (TripIdx(2), vec![0.0, 800.0], DistanceProvenance::CrowFly),
            (TripIdx(3), vec![0.0, 800.0], DistanceProvenance::CrowFly),
            (TripIdx(4), vec![0.0, 2000.0], DistanceProvenance::CrowFly),
        ],
    )
    .unwrap();
    let factors = [10.0, 10.0, 20.0, 20.0, 30.0];
    let inputs = CostInputs {
        geometry: &geometry,
        factors: &factors,
        leg_geometry: None,
        with_geometry: false,
        fares: None,
        rental: None,
        with_pieces: false,
    };
    let mut request = request(StopIdx(0), StopIdx(3), 0);
    request.egress = Vec::new();
    let access: HashMap<StopIdx, f64> = [(StopIdx(0), 120.0)].into_iter().collect();
    // Point 0 leaves from stop 3; point 1 prefers stop 4's shorter
    // egress over stop 3's long one.
    let egress = vec![
        vec![(StopIdx(3), 30, 25.0)],
        vec![(StopIdx(3), 1000, 900.0), (StopIdx(4), 10, 8.0)],
    ];
    let rows = Raptor.cost_matrix_to_points(
        &timetable,
        &transfers,
        &inputs,
        std::slice::from_ref(&request),
        std::slice::from_ref(&access),
        &egress,
    );
    let point_0 = &rows[0][0];
    assert_eq!((point_0.to, point_0.seconds, point_0.rides), (0, 430, 2));
    assert_eq!(point_0.transit_meters, 1300.0);
    // Access 120 m plus egress 25 m; no transfer on this journey.
    assert_eq!(point_0.walk_meters, 145.0);
    assert!((point_0.emission_grams - 21.0).abs() < 1e-9);
    let point_1 = &rows[0][1];
    assert_eq!((point_1.to, point_1.seconds, point_1.rides), (1, 360, 1));
    assert_eq!(point_1.transit_meters, 1200.0);
    // Access 120 m, the 50 m footpath to stop 4, egress 8 m.
    assert_eq!(point_1.walk_meters, 178.0);
    assert!((point_1.emission_grams - 12.0).abs() < 1e-9);
}

#[test]
fn cost_rows_carry_fares() {
    use crate::fares::{RuleFares, ZoneFares, ZoneProduct, NO_FARE};

    let (timetable, transfers) = network();
    let geometry = TripGeometry::from_trips(
        &timetable,
        vec![
            (
                TripIdx(0),
                vec![0.0, 500.0, 1200.0],
                DistanceProvenance::CrowFly,
            ),
            (
                TripIdx(1),
                vec![0.0, 500.0, 1200.0],
                DistanceProvenance::CrowFly,
            ),
            (TripIdx(2), vec![0.0, 800.0], DistanceProvenance::CrowFly),
            (TripIdx(3), vec![0.0, 800.0], DistanceProvenance::CrowFly),
            (TripIdx(4), vec![0.0, 2000.0], DistanceProvenance::CrowFly),
        ],
    )
    .unwrap();
    let factors = [10.0, 10.0, 20.0, 20.0, 30.0];
    let rules = |allowance: f64| {
        FareTables::RuleBased(RuleFares {
            route_type: vec![0, 0, 0],
            route_fare: vec![2.0, 3.0, 4.0],
            unlimited_transfers: vec![false],
            allow_same_route: vec![false],
            pair_fare: vec![4.5],
            max_discounted_transfers: 1,
            transfer_allowance: allowance,
            fare_cap: f64::INFINITY,
        })
    };
    let mut request = request(StopIdx(0), StopIdx(3), 0);
    request.egress = Vec::new();
    let priced = |tables: &FareTables| {
        let inputs = CostInputs {
            geometry: &geometry,
            factors: &factors,
            leg_geometry: None,
            with_geometry: false,
            fares: Some(tables),
            rental: None,
            with_pieces: false,
        };
        Raptor.cost_matrix(
            &timetable,
            &transfers,
            &inputs,
            std::slice::from_ref(&request),
            &[StopIdx(0), StopIdx(3)],
        )
    };
    // 0→3 boards route 0 at 100 and route 1 at 250: within a 200 s
    // allowance the pair total applies; a 100 s allowance splits it
    // into two full fares. The origin itself rides nothing.
    let rows = priced(&rules(200.0));
    assert_eq!(rows[0][0].fare, 0.0);
    assert!((rows[0][1].fare - 4.5).abs() < 1e-9);
    let rows = priced(&rules(100.0));
    assert!((rows[0][1].fare - 5.0).abs() < 1e-9);
    // Zone pricing reads the boarding and alighting stops' zones.
    let zones = FareTables::Zone(ZoneFares {
        stop_zone: vec![0, 0, 1, 1, NO_FARE],
        products: vec![ZoneProduct {
            price: 2.8,
            zones: 0b11,
            duration: f64::INFINITY,
            transfers: NO_FARE,
        }],
    });
    let rows = priced(&zones);
    assert!((rows[0][1].fare - 2.8).abs() < 1e-9);
    // The windowed fold keeps priceable candidates only and carries
    // the same fares.
    let inputs = CostInputs {
        geometry: &geometry,
        factors: &factors,
        leg_geometry: None,
        with_geometry: false,
        fares: Some(&rules(200.0)),
        rental: None,
        with_pieces: false,
    };
    let rows = Raptor.least_cost_matrix(
        &timetable,
        &transfers,
        &inputs,
        std::slice::from_ref(&request),
        &[StopIdx(0), StopIdx(3)],
        600,
        None,
        Objective::Fare,
    );
    assert_eq!(rows[0][0].fare, 0.0);
    assert!((rows[0][1].fare - 4.5).abs() < 1e-9);
}

#[test]
fn many_origins_match_single_runs() {
    // The parallel fan-out must agree with per-request runs; enough
    // duplicated requests make the workers reuse pooled state.
    let (timetable, transfers) = network();
    let origins = [StopIdx(0), StopIdx(1), StopIdx(2), StopIdx(4)];
    let requests: Vec<Request> = (0..8)
        .flat_map(|_| origins)
        .map(|origin| {
            let mut request = request(origin, StopIdx(3), 0);
            request.egress = Vec::new();
            request
        })
        .collect();
    let rows = Raptor.one_to_all_many(&timetable, &transfers, &requests);
    assert_eq!(rows.len(), requests.len());
    for (request, row) in requests.iter().zip(&rows) {
        assert_eq!(row, &Raptor.one_to_all(&timetable, &transfers, request));
    }
}

#[test]
fn routes_a_direct_ride() {
    let (timetable, transfers) = network();
    let journeys = Raptor.route(&timetable, &transfers, &request(StopIdx(0), StopIdx(2), 0));
    assert_eq!(journeys.len(), 1);
    let journey = &journeys[0];
    assert_eq!(journey.arrival, 300);
    assert_eq!(journey.rides(), 1);
    assert_eq!(
        journey.legs[1],
        Leg::Transit {
            trip: TripIdx(0),
            board_stop: StopIdx(0),
            alight_stop: StopIdx(2),
            board_position: 0,
            alight_position: 2,
            board_time: 100,
            alight_time: 300,
        }
    );
}

#[test]
fn boards_the_next_trip_when_the_first_has_left() {
    let (timetable, transfers) = network();
    let journeys = Raptor.route(
        &timetable,
        &transfers,
        &request(StopIdx(0), StopIdx(2), 150),
    );
    assert_eq!(journeys.len(), 1);
    assert_eq!(journeys[0].arrival, 900);
}

#[test]
fn transfers_at_a_shared_stop() {
    let (timetable, transfers) = network();
    let journeys = Raptor.route(&timetable, &transfers, &request(StopIdx(0), StopIdx(3), 0));
    // One ride cannot reach stop 3; two rides via stop 1 arrive at 400.
    assert_eq!(journeys.len(), 1);
    let journey = &journeys[0];
    assert_eq!(journey.arrival, 400);
    assert_eq!(journey.rides(), 2);
}

#[test]
fn walks_a_footpath_after_a_ride() {
    let (timetable, transfers) = network();
    // Ride A to stop 2 (arr 300), walk the 50-second footpath to 4.
    let journeys = Raptor.route(&timetable, &transfers, &request(StopIdx(0), StopIdx(4), 0));
    assert_eq!(journeys.len(), 1);
    let journey = &journeys[0];
    assert_eq!(journey.arrival, 350);
    assert!(matches!(
        journey.legs[2],
        Leg::Transfer {
            from_stop: StopIdx(2),
            to_stop: StopIdx(4),
            departure: 300,
            arrival: 350,
        }
    ));
}

#[test]
fn transfers_relax_a_single_hop_from_transit_arrivals() {
    // Footpaths 1→2 and 2→3 without a closing 1→3 edge: the walk out
    // of stop 2 must leave from its transit arrival (500), not chain
    // onto the walk that just improved stop 2 in the same round.
    let mut builder = TimetableBuilder::new(4);
    let to_a = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let to_b = builder.add_pattern(&[StopIdx(0), StopIdx(2)], 1).unwrap();
    builder
        .add_trip(to_a, vec![time(0), time(100)], 0, 0)
        .unwrap();
    builder
        .add_trip(to_b, vec![time(0), time(500)], 1, 0)
        .unwrap();
    let timetable = builder.finish();
    let transfers = Transfers::from_edges(
        4,
        &[
            (StopIdx(1), StopIdx(2), 30, 30.0),
            (StopIdx(2), StopIdx(3), 50, 50.0),
        ],
    )
    .unwrap();
    let journeys = Raptor.route(&timetable, &transfers, &request(StopIdx(0), StopIdx(3), 0));
    assert_eq!(journeys.len(), 1);
    assert_eq!(journeys[0].arrival, 550);
}

#[test]
fn footpaths_from_the_origin_are_the_access_lists_job() {
    let (timetable, transfers) = network();
    // Stop 2 only rides A at its last position; the footpath 2→4 is
    // not relaxed from the origin itself, by contract.
    let journeys = Raptor.route(
        &timetable,
        &transfers,
        &request(StopIdx(2), StopIdx(3), 260),
    );
    assert_eq!(journeys.len(), 0);
}

#[test]
fn skips_trips_of_inactive_services() {
    let (timetable, transfers) = network();
    // Departing at 260: B's active trip (dep 250) is gone; the service-1
    // trip at 500 exists but must not be boarded.
    let mut req = request(StopIdx(1), StopIdx(3), 260);
    let journeys = Raptor.route(&timetable, &transfers, &req);
    assert_eq!(journeys.len(), 0);
    // With service 1 active the 500 trip works.
    req.active_services = vec![true, true];
    let journeys = Raptor.route(&timetable, &transfers, &req);
    assert_eq!(journeys.len(), 1);
    assert_eq!(journeys[0].arrival, 600);
}

#[test]
fn emits_the_pareto_set_over_rides_and_arrival() {
    let mut builder = TimetableBuilder::new(3);
    // A slow direct pattern and a faster two-ride alternative.
    let direct = builder.add_pattern(&[StopIdx(0), StopIdx(2)], 0).unwrap();
    let first = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 1).unwrap();
    let second = builder.add_pattern(&[StopIdx(1), StopIdx(2)], 2).unwrap();
    builder
        .add_trip(direct, vec![time(0), time(1000)], 0, 0)
        .unwrap();
    builder
        .add_trip(first, vec![time(0), time(100)], 1, 0)
        .unwrap();
    builder
        .add_trip(second, vec![time(150), time(300)], 2, 0)
        .unwrap();
    let timetable = builder.finish();
    let transfers = Transfers::empty(3);
    let journeys = Raptor.route(
        &timetable,
        &transfers,
        &Request {
            departure: 0,
            access: vec![(StopIdx(0), 0)],
            egress: vec![(StopIdx(2), 0)],
            active_services: vec![true],
            active_services_previous: Vec::new(),
            max_transfers: 3,
            exclusions: None,
        },
    );
    assert_eq!(journeys.len(), 2);
    assert_eq!((journeys[0].rides(), journeys[0].arrival), (1, 1000));
    assert_eq!((journeys[1].rides(), journeys[1].arrival), (2, 300));
}

#[test]
fn chooses_between_access_and_egress_alternatives() {
    let (timetable, transfers) = network();
    // Origin can reach stop 0 slowly or stop 1 quickly; destination is
    // reachable from stop 2 or stop 3.
    let journeys = Raptor.route(
        &timetable,
        &transfers,
        &Request {
            departure: 0,
            access: vec![(StopIdx(0), 90), (StopIdx(1), 10)],
            egress: vec![(StopIdx(2), 500), (StopIdx(3), 20)],
            active_services: vec![true, false],
            active_services_previous: Vec::new(),
            max_transfers: 3,
            exclusions: None,
        },
    );
    // Best: board B at stop 1 (reached at 10, dep 250), arrive 3 at
    // 400, egress 20 → 420 with one ride. Riding A from 0 to 2 then
    // egress 500 gives 800; two rides cannot beat 420.
    assert_eq!(journeys.len(), 1);
    let journey = &journeys[0];
    assert_eq!(journey.arrival, 420);
    assert_eq!(journey.rides(), 1);
    assert_eq!(
        journey.legs[0],
        Leg::Access {
            to_stop: StopIdx(1),
            departure: 0,
            arrival: 10,
        }
    );
}

#[test]
fn terminal_stops_still_board_their_other_patterns() {
    // Stop 1 is the terminus of pattern X (0→1) and the start of
    // pattern Y (1→2); arriving at the terminus must still allow
    // boarding Y through its own pattern membership.
    let mut builder = TimetableBuilder::new(3);
    let x = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let y = builder.add_pattern(&[StopIdx(1), StopIdx(2)], 1).unwrap();
    builder.add_trip(x, vec![time(0), time(100)], 0, 0).unwrap();
    builder
        .add_trip(y, vec![time(150), time(250)], 1, 0)
        .unwrap();
    let timetable = builder.finish();
    let transfers = Transfers::empty(3);
    let journeys = Raptor.route(
        &timetable,
        &transfers,
        &Request {
            departure: 0,
            access: vec![(StopIdx(0), 0)],
            egress: vec![(StopIdx(2), 0)],
            active_services: vec![true],
            active_services_previous: Vec::new(),
            max_transfers: 3,
            exclusions: None,
        },
    );
    assert_eq!(journeys.len(), 1);
    assert_eq!(journeys[0].arrival, 250);
    assert_eq!(journeys[0].rides(), 2);
}

#[test]
fn one_to_all_reports_earliest_arrivals_everywhere() {
    let (timetable, transfers) = network();
    let arrivals = Raptor.one_to_all(&timetable, &transfers, &request(StopIdx(0), StopIdx(0), 0));
    // Origin at the departure time; ride A to 1 and 2; B onward to 3;
    // the footpath reaches 4.
    assert_eq!(arrivals[0], Some(0));
    assert_eq!(arrivals[1], Some(200));
    assert_eq!(arrivals[2], Some(300));
    assert_eq!(arrivals[3], Some(400));
    assert_eq!(arrivals[4], Some(350));
    // Departing after the last useful trips, nothing is reachable
    // beyond the origin itself.
    let late = Raptor.one_to_all(&timetable, &transfers, &request(StopIdx(3), StopIdx(0), 0));
    assert_eq!(late[3], Some(0));
    assert_eq!(late[0], None);
}

#[test]
fn respects_the_transfer_limit() {
    let (timetable, transfers) = network();
    let mut req = request(StopIdx(0), StopIdx(3), 0);
    req.max_transfers = 0;
    let journeys = Raptor.route(&timetable, &transfers, &req);
    assert_eq!(journeys.len(), 0);
}

/// One pattern 0→1 with three rides: 100→300, 200→400, 300→500.
fn frequent_network() -> (Timetable, Transfers) {
    let mut builder = TimetableBuilder::new(2);
    let a = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    for departure in [100, 200, 300] {
        builder
            .add_trip(a, vec![time(departure), time(departure + 200)], 0, 0)
            .unwrap();
    }
    (builder.finish(), Transfers::empty(2))
}

#[test]
fn range_emits_one_journey_per_feasible_departure() {
    let (timetable, transfers) = frequent_network();
    let journeys = Raptor.route_range(
        &timetable,
        &transfers,
        &request(StopIdx(0), StopIdx(1), 0),
        250,
    );
    // Departures 100 and 200 fall in [0, 250); each ride is the
    // latest-departure way to its arrival, so both survive. The
    // window's final second waits for the 300 ride.
    let profile: Vec<_> = journeys
        .iter()
        .map(|journey| (journey.departure, journey.arrival, journey.rides()))
        .collect();
    assert_eq!(profile, vec![(100, 300, 1), (200, 400, 1), (249, 500, 1)]);
    // Each journey departs the origin at its stated departure time.
    for journey in &journeys {
        assert_eq!(
            journey.legs[0],
            Leg::Access {
                to_stop: StopIdx(0),
                departure: journey.departure,
                arrival: journey.departure,
            }
        );
    }
}

#[test]
fn range_window_is_half_open() {
    let (timetable, transfers) = frequent_network();
    let journeys = Raptor.route_range(
        &timetable,
        &transfers,
        &request(StopIdx(0), StopIdx(1), 100),
        100,
    );
    // [100, 200) holds the 100 departure; the ride at 200 is only
    // reached by waiting from the window's final second.
    let profile: Vec<_> = journeys
        .iter()
        .map(|journey| (journey.departure, journey.arrival))
        .collect();
    assert_eq!(profile, vec![(100, 300), (199, 400)]);
    // A zero-length window has no departures at all.
    let none = Raptor.route_range(
        &timetable,
        &transfers,
        &request(StopIdx(0), StopIdx(1), 100),
        0,
    );
    assert!(none.is_empty());
}

#[test]
fn range_waits_past_the_window_when_the_next_ride_is_later() {
    let (timetable, transfers) = frequent_network();
    // No ride departs within [0, 50), but leaving at its final second
    // and waiting catches the ride at 100.
    let journeys = Raptor.route_range(
        &timetable,
        &transfers,
        &request(StopIdx(0), StopIdx(1), 0),
        50,
    );
    let profile: Vec<_> = journeys
        .iter()
        .map(|journey| (journey.departure, journey.arrival, journey.rides()))
        .collect();
    assert_eq!(profile, vec![(49, 300, 1)]);
}

#[test]
fn range_keeps_fewer_ride_options_from_earlier_departures() {
    // Departing at 200, a two-ride chain arrives at 320; departing at
    // 100, the direct ride arrives at 400. Neither dominates the
    // other — the direct journey needs fewer rides — so the faster
    // later pass must not prune the earlier pass's direct label.
    let mut builder = TimetableBuilder::new(3);
    let direct = builder.add_pattern(&[StopIdx(0), StopIdx(2)], 0).unwrap();
    let first = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 1).unwrap();
    let second = builder.add_pattern(&[StopIdx(1), StopIdx(2)], 2).unwrap();
    builder
        .add_trip(direct, vec![time(100), time(400)], 0, 0)
        .unwrap();
    builder
        .add_trip(first, vec![time(200), time(240)], 1, 0)
        .unwrap();
    builder
        .add_trip(second, vec![time(250), time(320)], 2, 0)
        .unwrap();
    let timetable = builder.finish();
    let transfers = Transfers::empty(3);
    let journeys = Raptor.route_range(
        &timetable,
        &transfers,
        &request(StopIdx(0), StopIdx(2), 0),
        201,
    );
    let profile: Vec<_> = journeys
        .iter()
        .map(|journey| (journey.departure, journey.arrival, journey.rides()))
        .collect();
    assert_eq!(profile, vec![(100, 400, 1), (200, 320, 2)]);
}

#[test]
fn range_drops_journeys_dominated_by_later_departures() {
    // A slow ride at 100 and an express at 150 that arrives earlier:
    // departing at 100 offers nothing the 150 departure does not beat.
    let mut builder = TimetableBuilder::new(2);
    let local = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let express = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 1).unwrap();
    builder
        .add_trip(local, vec![time(100), time(400)], 0, 0)
        .unwrap();
    builder
        .add_trip(express, vec![time(150), time(250)], 1, 0)
        .unwrap();
    let timetable = builder.finish();
    let transfers = Transfers::empty(2);
    let journeys = Raptor.route_range(
        &timetable,
        &transfers,
        &request(StopIdx(0), StopIdx(1), 0),
        200,
    );
    assert_eq!(journeys.len(), 1);
    assert_eq!((journeys[0].departure, journeys[0].arrival), (150, 250));
}

#[test]
fn range_keeps_extra_rides_only_when_strictly_earlier() {
    // Departing at 200, one direct ride arrives at 500. Departing at
    // 100, a two-ride chain arrives at 300; the direct ride is also
    // catchable then but no longer beats anything.
    let mut builder = TimetableBuilder::new(3);
    let direct = builder.add_pattern(&[StopIdx(0), StopIdx(2)], 0).unwrap();
    let first = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 1).unwrap();
    let second = builder.add_pattern(&[StopIdx(1), StopIdx(2)], 2).unwrap();
    builder
        .add_trip(direct, vec![time(200), time(500)], 0, 0)
        .unwrap();
    builder
        .add_trip(first, vec![time(100), time(150)], 1, 0)
        .unwrap();
    builder
        .add_trip(second, vec![time(160), time(300)], 2, 0)
        .unwrap();
    let timetable = builder.finish();
    let transfers = Transfers::empty(3);
    let journeys = Raptor.route_range(
        &timetable,
        &transfers,
        &request(StopIdx(0), StopIdx(2), 0),
        300,
    );
    let profile: Vec<_> = journeys
        .iter()
        .map(|journey| (journey.departure, journey.arrival, journey.rides()))
        .collect();
    assert_eq!(profile, vec![(100, 300, 2), (200, 500, 1)]);
}

#[test]
fn range_shifts_candidates_by_the_access_duration() {
    let (timetable, transfers) = frequent_network();
    let journeys = Raptor.route_range(
        &timetable,
        &transfers,
        &Request {
            departure: 0,
            access: vec![(StopIdx(0), 50)],
            egress: vec![(StopIdx(1), 0)],
            active_services: vec![true],
            active_services_previous: Vec::new(),
            max_transfers: 3,
            exclusions: None,
        },
        200,
    );
    // Catching the rides at 100 and 200 means leaving at 50 and 150;
    // the window's final second waits for the ride at 300.
    let departures: Vec<_> = journeys.iter().map(|journey| journey.departure).collect();
    assert_eq!(departures, vec![50, 150, 199]);
    assert_eq!(
        journeys[0].legs[0],
        Leg::Access {
            to_stop: StopIdx(0),
            departure: 50,
            arrival: 100,
        }
    );
}

#[test]
fn range_skips_candidates_of_inactive_services() {
    let (timetable, transfers) = network();
    // B's service-1 trip departs stop 1 at 500 but never runs.
    let journeys = Raptor.route_range(
        &timetable,
        &transfers,
        &request(StopIdx(1), StopIdx(3), 0),
        600,
    );
    let departures: Vec<_> = journeys.iter().map(|journey| journey.departure).collect();
    assert_eq!(departures, vec![250]);
}

#[test]
fn range_walks_footpaths_per_departure() {
    let (timetable, transfers) = network();
    // Only A's first trip (dep 100) reaches stop 4 in time for C: ride
    // to stop 2 (arr 300), walk 50 s, catch C at 400.
    let journeys = Raptor.route_range(
        &timetable,
        &transfers,
        &request(StopIdx(0), StopIdx(3), 0),
        800,
    );
    let profile: Vec<_> = journeys
        .iter()
        .map(|journey| (journey.departure, journey.arrival, journey.rides()))
        .collect();
    assert_eq!(profile, vec![(100, 400, 2)]);
}

/// A trip stored past midnight on the previous service day is boardable
/// early on the queried day, shifted back one day.
fn over_midnight_network() -> (Timetable, Transfers) {
    let mut builder = TimetableBuilder::new(2);
    let pattern = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    // 25:00 → 25:10 the previous day is 01:00 → 01:10 on this one.
    builder
        .add_trip(pattern, vec![time(90_000), time(90_600)], 0, 0)
        .unwrap();
    (builder.finish(), Transfers::empty(2))
}

#[test]
fn boards_the_previous_days_over_midnight_trip() {
    let (timetable, transfers) = over_midnight_network();
    let base = Request {
        departure: 0,
        access: vec![(StopIdx(0), 0)],
        egress: vec![(StopIdx(1), 0)],
        active_services: vec![false],
        active_services_previous: vec![false],
        max_transfers: 1,
        exclusions: None,
    };

    // Neither day runs the service: the night trip is unreachable.
    assert!(Raptor.route(&timetable, &transfers, &base).is_empty());

    // Today alone runs it at its stored 25:00 — reachable only by
    // waiting out the whole day, arriving 25:10.
    let today = Request {
        active_services: vec![true],
        ..base.clone()
    };
    let journeys = Raptor.route(&timetable, &transfers, &today);
    assert_eq!(journeys.len(), 1);
    assert_eq!(journeys[0].arrival, 90_600);

    // Active the day before, the same trip runs at 01:00 → 01:10 here.
    let previous = Request {
        active_services_previous: vec![true],
        ..base.clone()
    };
    let journeys = Raptor.route(&timetable, &transfers, &previous);
    assert_eq!(journeys.len(), 1);
    assert_eq!(journeys[0].arrival, 90_600 - 86_400);
    let Leg::Transit {
        board_time,
        alight_time,
        ..
    } = journeys[0].legs[1]
    else {
        panic!("expected a transit leg, got {:?}", journeys[0].legs);
    };
    assert_eq!(
        (board_time, alight_time),
        (90_000 - 86_400, 90_600 - 86_400)
    );

    // Both days active: the earlier previous-day run wins.
    let both = Request {
        active_services: vec![true],
        active_services_previous: vec![true],
        ..base.clone()
    };
    let journeys = Raptor.route(&timetable, &transfers, &both);
    assert_eq!(journeys.len(), 1);
    assert_eq!(journeys[0].arrival, 90_600 - 86_400);
}

#[test]
fn range_profiles_previous_day_over_midnight_trips() {
    let (timetable, transfers) = over_midnight_network();
    let request = Request {
        departure: 0,
        access: vec![(StopIdx(0), 0)],
        egress: vec![(StopIdx(1), 0)],
        active_services: vec![false],
        active_services_previous: vec![true],
        max_transfers: 1,
        exclusions: None,
    };
    // The window covers 00:00–02:00; the shifted 01:00 departure lands
    // in it and profiles as leaving at 01:00, arriving 01:10.
    let journeys = Raptor.route_range(&timetable, &transfers, &request, 2 * 3600);
    let profile: Vec<_> = journeys
        .iter()
        .map(|journey| (journey.departure, journey.arrival, journey.rides()))
        .collect();
    assert_eq!(profile, vec![(3_600, 4_200, 1)]);
}

/// An unclosed set must not let a better transfer label shadow a transit
/// arrival's own walk extensions. Stop 2 is reached faster by a round-1
/// transfer row (arr 25), yet the round-2 ride into it (arr 60) still
/// extends over the walking row to stop 3 — no closure row covers that
/// composite, so under the closure contract stop 3 stays unreached.
#[test]
fn unclosed_sets_relax_walks_from_shadowed_transit_arrivals() {
    let mut builder = TimetableBuilder::new(4);
    let first = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let second = builder.add_pattern(&[StopIdx(1), StopIdx(2)], 1).unwrap();
    builder
        .add_trip(first, vec![time(10), time(20)], 0, 0)
        .unwrap();
    builder
        .add_trip(second, vec![time(26), time(60)], 1, 0)
        .unwrap();
    let timetable = builder.finish();
    let edges = [
        (StopIdx(1), StopIdx(2), 5, 5.0),
        (StopIdx(2), StopIdx(3), 100, 100.0),
    ];
    let closed = Transfers::from_edges(4, &edges).unwrap();
    let mut unclosed = Transfers::from_edges(4, &edges).unwrap();
    unclosed.mark_unclosed();
    let request = Request {
        departure: 0,
        access: vec![(StopIdx(0), 0)],
        egress: vec![(StopIdx(3), 0)],
        active_services: vec![true],
        active_services_previous: Vec::new(),
        max_transfers: 3,
        exclusions: None,
    };
    assert_eq!(Raptor.one_to_all(&timetable, &closed, &request)[3], None);
    assert_eq!(
        Raptor.one_to_all(&timetable, &unclosed, &request)[3],
        Some(160)
    );
    let journeys = Raptor.route(&timetable, &unclosed, &request);
    let journey = journeys
        .iter()
        .find(|journey| journey.arrival == 160)
        .unwrap();
    assert_eq!(journey.rides(), 2);
    // The walk leaves the shadowed ride's arrival at 60, not the faster
    // transfer label's 25.
    assert_eq!(
        journey.legs[3],
        Leg::Transfer {
            from_stop: StopIdx(2),
            to_stop: StopIdx(3),
            departure: 60,
            arrival: 160,
        }
    );
    assert_eq!(
        journey.legs[2],
        Leg::Transit {
            trip: TripIdx(1),
            board_stop: StopIdx(1),
            alight_stop: StopIdx(2),
            board_position: 0,
            alight_position: 1,
            board_time: 26,
            alight_time: 60,
        }
    );
}

/// The arrive-by pricing pipeline exactly as the bindings run it: one
/// reverse run per destination at the final mark, each origin's
/// per-mark winners elected through the shared `profile_union`, and
/// the forward scan pricing the candidate tuples under the ceiling.
fn arrive_by_rows(
    timetable: &Timetable,
    transfers: &Transfers,
    inputs: &CostInputs<'_>,
    origin: StopIdx,
    destinations: &[StopIdx],
    marks: &[u32],
    budget: Option<u32>,
) -> Vec<CostRow> {
    use crate::routers::reverse;
    use crate::transfers::ReversedTransfers;

    let deadline = *marks.last().unwrap();
    let reversed = ReversedTransfers::build(transfers);
    let allowed: Vec<Vec<(u32, u32, u16)>> = destinations
        .iter()
        .map(|&destination| {
            let mut reverse_request = request(origin, destination, deadline);
            reverse_request.access = Vec::new();
            reverse_request.egress = vec![(destination, 0)];
            let states = reverse::reverse_one_to_all(timetable, &reversed, &reverse_request);
            let profile = reverse::reverse_profile_states(&states[origin.0 as usize], marks);
            reverse::profile_union(&[(0, profile)], marks, None)
                .into_iter()
                .map(|(departure, rides, achieved)| (departure, achieved, rides as u16))
                .collect()
        })
        .collect();
    let mut forward = request(origin, destinations[0], deadline);
    forward.egress = Vec::new();
    Raptor.arrive_by_least_cost_matrix(
        timetable,
        transfers,
        inputs,
        std::slice::from_ref(&forward),
        destinations,
        std::slice::from_ref(&allowed),
        deadline,
        budget,
        Objective::Emissions,
    )[0]
    .clone()
}

#[test]
fn a_time_dominated_cheaper_journey_never_wins_a_deadline_cell() {
    // Two direct rides 0→1: the clean trip departs earlier AND
    // arrives later than the dirty one, so every deadline mark's
    // Pareto set holds only the dirty ride — the profile-only
    // contract prices it, and the cleaner-but-dominated journey
    // never leaks into a cell.
    let mut builder = TimetableBuilder::new(2);
    let dirty = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let clean = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 1).unwrap();
    builder
        .add_trip(dirty, vec![time(100), time(200)], 0, 0)
        .unwrap();
    builder
        .add_trip(clean, vec![time(50), time(210)], 1, 0)
        .unwrap();
    let timetable = builder.finish();
    let transfers = Transfers::from_edges(2, &[]).unwrap();
    let geometry = TripGeometry::from_trips(
        &timetable,
        vec![
            (TripIdx(0), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
            (TripIdx(1), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
        ],
    )
    .unwrap();
    let factors = [100.0, 1.0];
    let inputs = CostInputs {
        geometry: &geometry,
        factors: &factors,
        leg_geometry: None,
        with_geometry: false,
        fares: None,
        rental: None,
        with_pieces: false,
    };
    let rows = arrive_by_rows(
        &timetable,
        &transfers,
        &inputs,
        StopIdx(0),
        &[StopIdx(1)],
        &[240, 300],
        None,
    );
    assert_eq!(rows.len(), 1);
    // The dirty ride: 1 km at 100 g/pkm. The clean 1 g/pkm ride is
    // dominated at both marks and must not price.
    assert!((rows[0].emission_grams - 100.0).abs() < 1e-9);
    assert_eq!((rows[0].seconds, rows[0].rides), (100, 1));
}

#[test]
fn a_non_minute_departure_instant_prices_exactly() {
    // The candidate departs at 130 — no minute mark — and the scan
    // must run at that exact instant, never a rounded one.
    let mut builder = TimetableBuilder::new(2);
    let a = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    builder
        .add_trip(a, vec![time(130), time(207)], 0, 0)
        .unwrap();
    let timetable = builder.finish();
    let transfers = Transfers::from_edges(2, &[]).unwrap();
    let geometry = TripGeometry::from_trips(
        &timetable,
        vec![(TripIdx(0), vec![0.0, 500.0], DistanceProvenance::CrowFly)],
    )
    .unwrap();
    let factors = [10.0];
    let inputs = CostInputs {
        geometry: &geometry,
        factors: &factors,
        leg_geometry: None,
        with_geometry: false,
        fares: None,
        rental: None,
        with_pieces: false,
    };
    let rows = arrive_by_rows(
        &timetable,
        &transfers,
        &inputs,
        StopIdx(0),
        &[StopIdx(1)],
        &[240],
        None,
    );
    assert_eq!(rows.len(), 1);
    assert_eq!((rows[0].seconds, rows[0].rides), (77, 1));
    assert!((rows[0].emission_grams - 5.0).abs() < 1e-9);
}

#[test]
fn an_arrival_between_the_last_mark_and_the_window_end_is_refused() {
    // Marks stop at 60 (a 90-second window holds no further minute
    // mark). The cheap ride arrives at 70 — inside the window's raw
    // end but past the final mark — so it satisfies no profiled
    // deadline and must not price; the pricier ride arriving by 60
    // wins the cell instead.
    let mut builder = TimetableBuilder::new(2);
    let cheap = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let priced = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 1).unwrap();
    builder
        .add_trip(cheap, vec![time(20), time(70)], 0, 0)
        .unwrap();
    builder
        .add_trip(priced, vec![time(10), time(60)], 1, 0)
        .unwrap();
    let timetable = builder.finish();
    let transfers = Transfers::from_edges(2, &[]).unwrap();
    let geometry = TripGeometry::from_trips(
        &timetable,
        vec![
            (TripIdx(0), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
            (TripIdx(1), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
        ],
    )
    .unwrap();
    let factors = [1.0, 50.0];
    let inputs = CostInputs {
        geometry: &geometry,
        factors: &factors,
        leg_geometry: None,
        with_geometry: false,
        fares: None,
        rental: None,
        with_pieces: false,
    };
    let rows = arrive_by_rows(
        &timetable,
        &transfers,
        &inputs,
        StopIdx(0),
        &[StopIdx(1)],
        &[0, 60],
        None,
    );
    assert_eq!(rows.len(), 1);
    assert!((rows[0].emission_grams - 50.0).abs() < 1e-9);
    assert_eq!((rows[0].seconds, rows[0].rides), (50, 1));
    // The ceiling itself, not just the election: hand the fold an
    // allowed set that DOES list the out-of-ceiling cheap tuple. The
    // `arrival > ceiling` guard alone must refuse it.
    let allowed: Vec<Vec<(u32, u32, u16)>> = vec![vec![(20, 70, 1), (10, 60, 1)]];
    let mut pricing_request = request(StopIdx(0), StopIdx(1), 60);
    pricing_request.egress = Vec::new();
    let rows = Raptor.arrive_by_least_cost_matrix(
        &timetable,
        &transfers,
        &inputs,
        std::slice::from_ref(&pricing_request),
        &[StopIdx(1)],
        std::slice::from_ref(&allowed),
        60,
        None,
        Objective::Emissions,
    );
    assert_eq!(rows[0].len(), 1);
    assert!((rows[0][0].emission_grams - 50.0).abs() < 1e-9);
    assert_eq!((rows[0][0].seconds, rows[0][0].rides), (50, 1));
}

#[test]
fn equal_tuple_journeys_price_the_canonical_one() {
    // Two rides share the exact (departure, arrival, rides) tuple but
    // differ in emissions. Candidate identity is the canonical
    // journey: the forward engine's own winner at that departure —
    // the label chain the fold reconstructs — prices the cell, even
    // when the shadowed twin is cheaper.
    let mut builder = TimetableBuilder::new(2);
    let first = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let second = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 1).unwrap();
    builder
        .add_trip(first, vec![time(100), time(200)], 0, 0)
        .unwrap();
    builder
        .add_trip(second, vec![time(100), time(200)], 1, 0)
        .unwrap();
    let timetable = builder.finish();
    let transfers = Transfers::from_edges(2, &[]).unwrap();
    let geometry = TripGeometry::from_trips(
        &timetable,
        vec![
            (TripIdx(0), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
            (TripIdx(1), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
        ],
    )
    .unwrap();
    let factors = [40.0, 5.0];
    let inputs = CostInputs {
        geometry: &geometry,
        factors: &factors,
        leg_geometry: None,
        with_geometry: false,
        fares: None,
        rental: None,
        with_pieces: false,
    };
    // The canonical winner is whatever the forward engine elects at
    // departure 100 — derive it from the engine itself, never assume.
    let mut forward = request(StopIdx(0), StopIdx(1), 100);
    forward.active_services = vec![true, true];
    let journeys = Raptor.route(&timetable, &transfers, &forward);
    assert_eq!(journeys.len(), 1);
    let canonical_trip = journeys[0]
        .legs
        .iter()
        .find_map(|leg| match leg {
            Leg::Transit { trip, .. } => Some(*trip),
            _ => None,
        })
        .unwrap();
    let canonical_grams = factors[canonical_trip.0 as usize];
    let cheaper_grams = factors.iter().copied().fold(f64::INFINITY, f64::min);
    // The fixture only discriminates when the canonical twin is the
    // pricier one — a cell equal to the cheaper twin would also pass
    // a min-pricing bug.
    assert!(canonical_grams > cheaper_grams);

    use crate::routers::reverse;
    use crate::transfers::ReversedTransfers;
    let reversed = ReversedTransfers::build(&transfers);
    let mut reverse_request = request(StopIdx(0), StopIdx(1), 240);
    reverse_request.active_services = vec![true, true];
    reverse_request.access = Vec::new();
    reverse_request.egress = vec![(StopIdx(1), 0)];
    let states = reverse::reverse_one_to_all(&timetable, &reversed, &reverse_request);
    let profile = reverse::reverse_profile_states(&states[0], &[240]);
    let allowed: Vec<Vec<(u32, u32, u16)>> =
        vec![reverse::profile_union(&[(0, profile)], &[240], None)
            .into_iter()
            .map(|(departure, rides, achieved)| (departure, achieved, rides as u16))
            .collect()];
    // One tuple: the twins collapse to a single candidate identity.
    assert_eq!(allowed[0], vec![(100, 200, 1)]);
    let mut pricing_request = request(StopIdx(0), StopIdx(1), 240);
    pricing_request.active_services = vec![true, true];
    pricing_request.egress = Vec::new();
    let rows = Raptor.arrive_by_least_cost_matrix(
        &timetable,
        &transfers,
        &inputs,
        std::slice::from_ref(&pricing_request),
        &[StopIdx(1)],
        std::slice::from_ref(&allowed),
        240,
        None,
        Objective::Emissions,
    );
    assert_eq!(rows[0].len(), 1);
    assert!((rows[0][0].emission_grams - canonical_grams).abs() < 1e-9);
}
