use super::*;
use crate::geometry::DistanceProvenance;
use crate::timetable::{StopTime, TimetableBuilder, TripIdx};

fn time(at: u32) -> StopTime {
    StopTime {
        arrival: at,
        departure: at,
    }
}

/// Line A rides 0→1→2; a fast dirty line and a slow clean line both
/// ride 1→3.
fn forked() -> (Timetable, TripGeometry) {
    let mut builder = TimetableBuilder::new(4);
    let a = builder
        .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 0)
        .unwrap();
    let fast = builder.add_pattern(&[StopIdx(1), StopIdx(3)], 1).unwrap();
    let slow = builder.add_pattern(&[StopIdx(1), StopIdx(3)], 2).unwrap();
    builder
        .add_trip(a, vec![time(0), time(100), time(300)], 0, 0)
        .unwrap();
    builder
        .add_trip(fast, vec![time(120), time(400)], 1, 0)
        .unwrap();
    builder
        .add_trip(slow, vec![time(150), time(600)], 2, 0)
        .unwrap();
    let timetable = builder.finish();
    let geometry = TripGeometry::from_trips(
        &timetable,
        vec![
            (
                TripIdx(0),
                vec![0.0, 500.0, 1000.0],
                DistanceProvenance::CrowFly,
            ),
            (TripIdx(1), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
            (TripIdx(2), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
        ],
    )
    .unwrap();
    (timetable, geometry)
}

fn boarded(set: &TransferSet, trip: ViewTrip, position: u16) -> Vec<(u32, u16)> {
    let mut list: Vec<(u32, u16)> = set
        .from_trip_position(trip, position)
        .iter()
        .map(|transfer| (transfer.trip.0, transfer.position))
        .collect();
    list.sort_unstable();
    list
}

#[test]
fn keeps_the_cleaner_slower_transfer() {
    let (timetable, geometry) = forked();
    let view = DayView::universal(&timetable);
    let footpaths = Transfers::empty(4);
    // The time reduction sees the slow line arriving later
    // everywhere and drops it.
    let time_set = TransferSet::for_view(&view, &timetable, &footpaths).transfers;
    assert_eq!(boarded(&time_set, ViewTrip(0), 1), vec![(1, 0)]);
    // Clean-but-slow (factor 10 vs 100): a true Pareto move, kept.
    let factors = [50.0, 100.0, 10.0];
    let mc = transfer_set(&view, &timetable, &footpaths, &geometry, &factors).transfers;
    assert_eq!(boarded(&mc, ViewTrip(0), 1), vec![(1, 0), (2, 0)]);
    // With the slow line just as dirty, the mc reduction drops it
    // too — and matches the time set exactly here.
    let uniform = [50.0, 100.0, 100.0];
    let same = transfer_set(&view, &timetable, &footpaths, &geometry, &uniform).transfers;
    assert_eq!(boarded(&same, ViewTrip(0), 1), vec![(1, 0)]);
}

#[test]
fn the_mc_set_covers_the_time_set() {
    let (timetable, geometry) = forked();
    let view = DayView::universal(&timetable);
    let footpaths = Transfers::empty(4);
    let time_set = TransferSet::for_view(&view, &timetable, &footpaths).transfers;
    for factors in [[50.0, 100.0, 10.0], [50.0, 50.0, 50.0], [1.0, 2.0, 3.0]] {
        let mc = transfer_set(&view, &timetable, &footpaths, &geometry, &factors).transfers;
        for trip in 0..view.trip_count() {
            let stops = timetable
                .pattern_stops(view.line_pattern(view.line_of(ViewTrip(trip))))
                .len();
            for position in 0..stops as u16 {
                for kept in time_set.from_trip_position(ViewTrip(trip), position) {
                    assert!(
                        mc.from_trip_position(ViewTrip(trip), position)
                            .contains(kept),
                        "time-kept transfer missing under factors {factors:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn mixed_factor_lines_board_the_cleaner_later_trip() {
    // One connecting line with a dirty early trip and a clean later
    // one: generation must board both.
    let mut builder = TimetableBuilder::new(4);
    let a = builder
        .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 0)
        .unwrap();
    let b = builder.add_pattern(&[StopIdx(1), StopIdx(3)], 1).unwrap();
    builder
        .add_trip(a, vec![time(0), time(100), time(300)], 0, 0)
        .unwrap();
    builder
        .add_trip(b, vec![time(120), time(400)], 1, 0)
        .unwrap();
    builder
        .add_trip(b, vec![time(200), time(500)], 2, 0)
        .unwrap();
    let timetable = builder.finish();
    let geometry = TripGeometry::from_trips(
        &timetable,
        vec![
            (
                TripIdx(0),
                vec![0.0, 500.0, 1000.0],
                DistanceProvenance::CrowFly,
            ),
            (TripIdx(1), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
            (TripIdx(2), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
        ],
    )
    .unwrap();
    let view = DayView::universal(&timetable);
    let footpaths = Transfers::empty(4);
    let mixed = [50.0, 100.0, 10.0];
    let mc = transfer_set(&view, &timetable, &footpaths, &geometry, &mixed).transfers;
    assert_eq!(boarded(&mc, ViewTrip(0), 1), vec![(1, 0), (2, 0)]);
    // Uniform factors collapse to the earliest-trip rule.
    let uniform = [50.0, 100.0, 100.0];
    let mc = transfer_set(&view, &timetable, &footpaths, &geometry, &uniform).transfers;
    assert_eq!(boarded(&mc, ViewTrip(0), 1), vec![(1, 0)]);
}

#[test]
fn same_line_reboarding_survives_only_toward_cleaner_siblings() {
    // One line, dirty trip then clean trip: alighting the dirty
    // trip mid-pattern may re-board the cleaner sibling at the same
    // position; with uniform factors the sibling stays skipped.
    let mut builder = TimetableBuilder::new(3);
    let a = builder
        .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 0)
        .unwrap();
    builder
        .add_trip(a, vec![time(0), time(100), time(300)], 0, 0)
        .unwrap();
    builder
        .add_trip(a, vec![time(50), time(200), time(400)], 1, 0)
        .unwrap();
    let timetable = builder.finish();
    let geometry = TripGeometry::from_trips(
        &timetable,
        vec![
            (
                TripIdx(0),
                vec![0.0, 500.0, 1000.0],
                DistanceProvenance::CrowFly,
            ),
            (
                TripIdx(1),
                vec![0.0, 500.0, 1000.0],
                DistanceProvenance::CrowFly,
            ),
        ],
    )
    .unwrap();
    let view = DayView::universal(&timetable);
    let footpaths = Transfers::empty(3);
    let mixed = [100.0, 10.0];
    let mc = transfer_set(&view, &timetable, &footpaths, &geometry, &mixed).transfers;
    assert_eq!(boarded(&mc, ViewTrip(0), 1), vec![(1, 1)]);
    let uniform = [100.0, 100.0];
    let mc = transfer_set(&view, &timetable, &footpaths, &geometry, &uniform).transfers;
    assert_eq!(boarded(&mc, ViewTrip(0), 1), vec![]);
}

#[test]
fn unresolved_factors_are_excluded() {
    let (timetable, geometry) = forked();
    let view = DayView::universal(&timetable);
    let footpaths = Transfers::empty(4);
    // The fast line's factor is unresolved: never boarded.
    let factors = [50.0, f64::NAN, 10.0];
    let mc = transfer_set(&view, &timetable, &footpaths, &geometry, &factors).transfers;
    assert_eq!(boarded(&mc, ViewTrip(0), 1), vec![(2, 0)]);
    // An unresolved source trip gets no transfers at all.
    let factors = [f64::NAN, 100.0, 10.0];
    let mc = transfer_set(&view, &timetable, &footpaths, &geometry, &factors).transfers;
    assert_eq!(boarded(&mc, ViewTrip(0), 1), vec![]);
}

use crate::exhaustive::pareto_oracle;
use crate::mcraptor;

fn grams_of(journey: &Journey, geometry: &TripGeometry, factors: &[f64]) -> f64 {
    quantized(
        journey
            .legs
            .iter()
            .map(|leg| match leg {
                Leg::Transit {
                    trip,
                    board_position,
                    alight_position,
                    ..
                } => {
                    geometry.leg_distance(*trip, *board_position, *alight_position) as f64 / 1000.0
                        * factors[trip.0 as usize]
                }
                _ => 0.0,
            })
            .sum(),
    )
}

fn triples(journeys: &[Journey], geometry: &TripGeometry, factors: &[f64]) -> Vec<(u32, f64, u32)> {
    let mut list: Vec<(u32, f64, u32)> = journeys
        .iter()
        .map(|journey| {
            (
                journey.arrival,
                grams_of(journey, geometry, factors),
                journey.rides() as u32,
            )
        })
        .collect();
    list.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.partial_cmp(&b.1).unwrap()));
    list
}

/// The three-way check: with a vanishing bucket, the trip-based
/// engine, McRAPTOR, and the exhaustive oracle must produce the
/// same frontier.
fn engines_agree(
    timetable: &Timetable,
    geometry: &TripGeometry,
    footpaths: &Transfers,
    factors: &[f64],
    origin: StopIdx,
    egress: StopIdx,
    max_transfers: u8,
) -> Vec<(u32, f64, u32)> {
    let view = DayView::universal(timetable);
    let request = Request {
        departure: 0,
        access: vec![(origin, 0)],
        egress: vec![(egress, 0)],
        active_services: vec![true; 8],
        active_services_previous: vec![false; 8],
        max_transfers,
    };
    let points = pareto_oracle(
        &view,
        timetable,
        footpaths,
        geometry,
        factors,
        0,
        &request.access,
        &request.egress,
        max_transfers,
    );
    let oracle: Vec<(u32, f64, u32)> = points
        .iter()
        .map(|point| (point.arrival, point.grams, point.rides))
        .collect();
    let raptor = mcraptor::route(
        &view,
        timetable,
        footpaths,
        geometry,
        factors,
        &request,
        1e-6,
        0,
        None,
        &[],
    );
    assert_eq!(triples(&raptor, geometry, factors), oracle, "mcraptor");
    let engine = McTbtrEngine::for_date(
        timetable,
        footpaths,
        geometry,
        factors,
        &request.active_services,
        &request.active_services_previous,
    );
    let tbtr = engine.route(&request, 1e-6);
    assert_eq!(triples(&tbtr, geometry, factors), oracle, "mctbtr");
    oracle
}

#[test]
fn the_engine_matches_the_oracle_on_the_forked_fixture() {
    let (timetable, geometry) = forked();
    let footpaths = Transfers::empty(4);
    let frontier = engines_agree(
        &timetable,
        &geometry,
        &footpaths,
        &[50.0, 100.0, 10.0],
        StopIdx(0),
        StopIdx(3),
        3,
    );
    // Fast-dirty and slow-clean two-ride journeys: both true points.
    assert_eq!(frontier, vec![(400, 125.0, 2), (600, 35.0, 2)]);
    engines_agree(
        &timetable,
        &geometry,
        &footpaths,
        &[50.0, 100.0, 100.0],
        StopIdx(0),
        StopIdx(3),
        3,
    );
}

#[test]
fn the_engine_matches_the_oracle_over_footpaths() {
    // Ride, walk a footpath, ride again — the walked transfer is a
    // query-time relaxation, not a precomputed one.
    let mut builder = TimetableBuilder::new(4);
    let first = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let second = builder.add_pattern(&[StopIdx(2), StopIdx(3)], 1).unwrap();
    builder
        .add_trip(first, vec![time(0), time(100)], 0, 0)
        .unwrap();
    builder
        .add_trip(second, vec![time(200), time(300)], 1, 0)
        .unwrap();
    let timetable = builder.finish();
    let geometry = TripGeometry::from_trips(
        &timetable,
        vec![
            (TripIdx(0), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
            (TripIdx(1), vec![0.0, 2000.0], DistanceProvenance::CrowFly),
        ],
    )
    .unwrap();
    let footpaths = Transfers::from_edges(
        4,
        &[
            (StopIdx(1), StopIdx(2), 50, 50.0),
            (StopIdx(2), StopIdx(1), 50, 50.0),
        ],
    )
    .unwrap();
    let frontier = engines_agree(
        &timetable,
        &geometry,
        &footpaths,
        &[10.0, 20.0],
        StopIdx(0),
        StopIdx(3),
        3,
    );
    assert_eq!(frontier, vec![(300, 50.0, 2)]);
}

#[test]
fn loop_backs_walk_nowhere() {
    // The routers' shared regression shape: ride out and back, then
    // walk — suppressed by the access-seeded stop bags.
    let mut builder = TimetableBuilder::new(3);
    let out = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let back = builder.add_pattern(&[StopIdx(1), StopIdx(0)], 1).unwrap();
    let direct = builder.add_pattern(&[StopIdx(0), StopIdx(2)], 2).unwrap();
    builder
        .add_trip(out, vec![time(10), time(50)], 0, 0)
        .unwrap();
    builder
        .add_trip(back, vec![time(60), time(100)], 1, 0)
        .unwrap();
    builder
        .add_trip(direct, vec![time(20), time(200)], 2, 0)
        .unwrap();
    let timetable = builder.finish();
    let geometry = TripGeometry::from_trips(
        &timetable,
        vec![
            (TripIdx(0), vec![0.0, 400.0], DistanceProvenance::CrowFly),
            (TripIdx(1), vec![0.0, 400.0], DistanceProvenance::CrowFly),
            (TripIdx(2), vec![0.0, 800.0], DistanceProvenance::CrowFly),
        ],
    )
    .unwrap();
    let footpaths = Transfers::from_edges(
        3,
        &[
            (StopIdx(0), StopIdx(2), 30, 30.0),
            (StopIdx(2), StopIdx(0), 30, 30.0),
        ],
    )
    .unwrap();
    let frontier = engines_agree(
        &timetable,
        &geometry,
        &footpaths,
        &[10.0, 10.0, 100.0],
        StopIdx(0),
        StopIdx(2),
        3,
    );
    assert_eq!(frontier, vec![(200, 80.0, 1)]);
}

#[test]
fn the_matrix_matches_the_mcraptor_matrix() {
    // Cell for cell against the McRAPTOR matrix on the forked
    // fixture, across budgets — including the zero-ride floor of
    // the origin's own cell and the empty cell a tight budget
    // leaves.
    let (timetable, geometry) = forked();
    let factors = [50.0, 100.0, 10.0];
    let footpaths = Transfers::empty(4);
    let view = DayView::universal(&timetable);
    let inputs = CostInputs {
        geometry: &geometry,
        factors: &factors,
        leg_geometry: None,
        with_geometry: false,
        fares: None,
    };
    let requests = vec![Request {
        departure: 0,
        access: vec![(StopIdx(0), 0)],
        egress: Vec::new(),
        active_services: vec![true],
        active_services_previous: vec![false],
        max_transfers: 3,
    }];
    let destinations = [StopIdx(0), StopIdx(3)];
    let engine = McTbtrEngine::for_date(
        &timetable,
        &footpaths,
        &geometry,
        &factors,
        &requests[0].active_services,
        &requests[0].active_services_previous,
    );
    let no_egress = vec![Vec::new(); timetable.stop_count() as usize];
    for budget in [None, Some(600), Some(400), Some(50)] {
        let tbtr =
            engine.least_emissions_matrix(&inputs, &requests, &destinations, 600, budget, 1e-6);
        let raptor = mcraptor::least_emissions_matrix(
            &view,
            &timetable,
            &footpaths,
            &inputs,
            &requests,
            &destinations,
            &no_egress,
            &vec![Vec::new(); requests.len()],
            false,
            600,
            budget,
            1e-6,
        );
        let cells = |rows: &Vec<Vec<CostRow>>| -> Vec<(u32, u32, u32, f64, f64, f64)> {
            let mut list: Vec<_> = rows[0]
                .iter()
                .map(|row| {
                    (
                        row.to,
                        row.seconds,
                        row.rides,
                        row.transit_meters,
                        row.walk_meters,
                        row.emission_grams,
                    )
                })
                .collect();
            list.sort_by(|a, b| a.partial_cmp(b).unwrap());
            list
        };
        assert_eq!(cells(&tbtr), cells(&raptor), "budget {budget:?}");
    }
    // The unbudgeted cell is the slow clean line, not the fast
    // dirty one.
    let rows = engine.least_emissions_matrix(&inputs, &requests, &destinations, 600, None, 1e-6);
    let cell = rows[0].iter().find(|row| row.to == 3).unwrap();
    assert_eq!((cell.seconds, cell.rides), (600, 2));
    assert!((cell.emission_grams - 35.0).abs() < 1e-9);
}

#[test]
fn profiles_the_departure_window() {
    let mut builder = TimetableBuilder::new(2);
    let line = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    builder
        .add_trip(line, vec![time(100), time(300)], 0, 0)
        .unwrap();
    builder
        .add_trip(line, vec![time(200), time(400)], 1, 0)
        .unwrap();
    let timetable = builder.finish();
    let geometry = TripGeometry::from_trips(
        &timetable,
        vec![
            (TripIdx(0), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
            (TripIdx(1), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
        ],
    )
    .unwrap();
    let footpaths = Transfers::empty(2);
    let factors = [50.0, 50.0];
    let request = Request {
        departure: 50,
        access: vec![(StopIdx(0), 0)],
        egress: vec![(StopIdx(1), 0)],
        active_services: vec![true],
        active_services_previous: vec![false],
        max_transfers: 1,
    };
    let engine = McTbtrEngine::for_date(
        &timetable,
        &footpaths,
        &geometry,
        &factors,
        &request.active_services,
        &request.active_services_previous,
    );
    let journeys = engine.route_range(&request, 200, 1e-6);
    let profile: Vec<(u32, u32)> = journeys
        .iter()
        .map(|journey| (journey.departure, journey.arrival))
        .collect();
    assert_eq!(profile, vec![(100, 300), (200, 400)]);
}

#[test]
fn u_turns_stay_dropped() {
    // Alight A at stop 2, walk back to stop 1, and re-ride the
    // 1→2 segment on B although B was catchable by alighting A at
    // stop 1 directly: the classic U-turn, dropped even though B
    // is cleaner — the earlier alight saves grams on A and rides
    // the identical distance on B. Boarding B at stop 2 itself
    // stays a plain, kept transfer.
    let mut builder = TimetableBuilder::new(4);
    let a = builder
        .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 0)
        .unwrap();
    let b = builder
        .add_pattern(&[StopIdx(1), StopIdx(2), StopIdx(3)], 1)
        .unwrap();
    builder
        .add_trip(a, vec![time(0), time(100), time(200)], 0, 0)
        .unwrap();
    builder
        .add_trip(b, vec![time(250), time(350), time(450)], 1, 0)
        .unwrap();
    let timetable = builder.finish();
    let geometry = TripGeometry::from_trips(
        &timetable,
        vec![
            (
                TripIdx(0),
                vec![0.0, 500.0, 1000.0],
                DistanceProvenance::CrowFly,
            ),
            (
                TripIdx(1),
                vec![0.0, 500.0, 1000.0],
                DistanceProvenance::CrowFly,
            ),
        ],
    )
    .unwrap();
    let footpaths = Transfers::from_edges(
        4,
        &[
            (StopIdx(1), StopIdx(2), 30, 30.0),
            (StopIdx(2), StopIdx(1), 30, 30.0),
        ],
    )
    .unwrap();
    let view = DayView::universal(&timetable);
    let factors = [100.0, 10.0];
    let mc = transfer_set(&view, &timetable, &footpaths, &geometry, &factors).transfers;
    // From (A, position 2): the walk-back boarding of B at
    // position 0 is the U-turn and is gone; boarding B at stop 2
    // (position 1) survives.
    assert_eq!(boarded(&mc, ViewTrip(0), 2), vec![(1, 1)]);
}
