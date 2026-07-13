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

/// Every kept transfer as (trip, alight, boarded trip, position).
fn kept(set: &TransferSet, view: &DayView, timetable: &Timetable) -> Vec<(u32, u16, u32, u16)> {
    let mut list = Vec::new();
    for trip in 0..view.trip_count() {
        let stops = timetable
            .pattern_stops(view.line_pattern(view.line_of(ViewTrip(trip))))
            .len();
        for position in 0..stops as u16 {
            for transfer in set.from_trip_position(ViewTrip(trip), position) {
                list.push((trip, position, transfer.trip.0, transfer.position));
            }
        }
    }
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
    let mc = transfer_set(&view, &timetable, &geometry, &factors).transfers;
    assert_eq!(boarded(&mc, ViewTrip(0), 1), vec![(1, 0), (2, 0)]);
    // With the slow line just as dirty, the mc reduction drops it
    // too — and matches the time set exactly here.
    let uniform = [50.0, 100.0, 100.0];
    let same = transfer_set(&view, &timetable, &geometry, &uniform).transfers;
    assert_eq!(boarded(&same, ViewTrip(0), 1), vec![(1, 0)]);
}

#[test]
fn the_global_set_keeps_exactly_the_needed_transfers() {
    // Per factor configuration: the expected global contents, and the
    // retired per-trip local reduction's complete output on the same
    // configuration — a frozen snapshot captured by running that
    // implementation before its removal. The subset assertion
    // enforces the plan's pruning-only-removes invariant against the
    // baseline; the equality assertion pins the exact contents (the
    // fork's only transfer event is (A, stop 1): a cleaner-but-slower
    // fork holds a Pareto point and survives, an equally-dirty-or-
    // worse one is witnessed away, and every other cell is empty).
    let (timetable, geometry) = forked();
    let view = DayView::universal(&timetable);
    type Case = (
        [f64; 3],
        Vec<(u32, u16, u32, u16)>,
        Vec<(u32, u16, u32, u16)>,
    );
    let cases: [Case; 3] = [
        (
            [50.0, 100.0, 10.0],
            vec![(0, 1, 1, 0), (0, 1, 2, 0)],
            vec![(0, 1, 1, 0), (0, 1, 2, 0)],
        ),
        ([50.0, 50.0, 50.0], vec![(0, 1, 1, 0)], vec![(0, 1, 1, 0)]),
        ([1.0, 2.0, 3.0], vec![(0, 1, 1, 0)], vec![(0, 1, 1, 0)]),
    ];
    for (factors, expected, local_baseline) in cases {
        let build = transfer_set(&view, &timetable, &geometry, &factors);
        let list = kept(&build.transfers, &view, &timetable);
        assert!(
            list.iter()
                .all(|transfer| local_baseline.contains(transfer)),
            "kept a transfer outside the local baseline under {factors:?}"
        );
        assert_eq!(list, expected, "under factors {factors:?}");
        assert_eq!(build.generated, expected.len(), "under factors {factors:?}");
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
    // The retired local reduction's complete output on this fixture
    // (a frozen snapshot captured by running it before its removal):
    // the global set may only remove from it — here it keeps all of
    // it, the dirty early trip and the cleaner later sibling.
    let mixed = [50.0, 100.0, 10.0];
    let local_baseline = [(0, 1, 1, 0), (0, 1, 2, 0)];
    let mc = transfer_set(&view, &timetable, &geometry, &mixed).transfers;
    let list = kept(&mc, &view, &timetable);
    assert!(list.iter().all(|t| local_baseline.contains(t)));
    assert_eq!(list, vec![(0, 1, 1, 0), (0, 1, 2, 0)]);
    // Uniform factors collapse to the earliest-trip rule; the local
    // baseline shrinks to the earliest boarding alone.
    let uniform = [50.0, 100.0, 100.0];
    let local_baseline = [(0, 1, 1, 0)];
    let mc = transfer_set(&view, &timetable, &geometry, &uniform).transfers;
    let list = kept(&mc, &view, &timetable);
    assert!(list.iter().all(|t| local_baseline.contains(t)));
    assert_eq!(list, vec![(0, 1, 1, 0)]);
}

#[test]
fn same_line_reboarding_is_witnessed_away_by_the_direct_boarding() {
    // One line, dirty trip then clean trip: re-boarding the cleaner
    // sibling mid-pattern is globally redundant — any rider who could
    // catch the dirty trip could board the later sibling at the same
    // stop directly, arriving identically with fewer grams and fewer
    // rides. With uniform factors the sibling is never boarded at all.
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
    let mixed = [100.0, 10.0];
    let mc = transfer_set(&view, &timetable, &geometry, &mixed).transfers;
    assert_eq!(kept(&mc, &view, &timetable), vec![]);
    let uniform = [100.0, 100.0];
    let mc = transfer_set(&view, &timetable, &geometry, &uniform).transfers;
    assert_eq!(kept(&mc, &view, &timetable), vec![]);
}

#[test]
fn unresolved_factors_are_excluded() {
    let (timetable, geometry) = forked();
    let view = DayView::universal(&timetable);
    // The fast line's factor is unresolved: never boarded.
    let factors = [50.0, f64::NAN, 10.0];
    let mc = transfer_set(&view, &timetable, &geometry, &factors).transfers;
    assert_eq!(boarded(&mc, ViewTrip(0), 1), vec![(2, 0)]);
    // An unresolved source trip gets no transfers at all.
    let factors = [f64::NAN, 100.0, 10.0];
    let mc = transfer_set(&view, &timetable, &geometry, &factors).transfers;
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
        None,
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
fn the_earlier_alight_is_the_canonical_transfer_point() {
    // A rides 0→1→2 and B rides 1→2→3: alighting A at stop 1
    // boards B at its start; staying aboard to stop 2 and
    // transferring there arrives identically having ridden the
    // dirtier vehicle further, so only the earlier alight's transfer
    // survives. Footpaths never enter the set — walked transfers are
    // the query's job.
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
    let view = DayView::universal(&timetable);
    let factors = [100.0, 10.0];
    let build = transfer_set(&view, &timetable, &geometry, &factors);
    assert_eq!(
        kept(&build.transfers, &view, &timetable),
        vec![(0, 1, 1, 0)]
    );
}

#[test]
fn over_midnight_departures_do_not_underflow() {
    // A previous-day trip's pre-midnight stop events are yesterday's
    // departures: the source runs skip them instead of shifting them
    // below zero.
    let mut builder = TimetableBuilder::new(2);
    let a = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    builder
        .add_trip(a, vec![time(82_800), time(90_000)], 0, 0)
        .unwrap();
    builder
        .add_trip(a, vec![time(100), time(200)], 1, 1)
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
    let view = DayView::for_date(&timetable, &[false, true], &[true, false]);
    let factors = [10.0, 10.0];
    let build = transfer_set(&view, &timetable, &geometry, &factors);
    assert_eq!(kept(&build.transfers, &view, &timetable), vec![]);
}

#[test]
fn the_frontier_matrix_matches_the_one_pair_profile_per_cell() {
    let (timetable, geometry) = forked();
    let factors = [10.0, 100.0, 10.0];
    let footpaths = Transfers::empty(4);
    let engine = McTbtrEngine::for_date(&timetable, &footpaths, &geometry, &factors, &[true], &[]);
    let destinations = [StopIdx(3), StopIdx(2), StopIdx(3)];
    let requests: Vec<Request> = [StopIdx(0), StopIdx(2)]
        .into_iter()
        .map(|origin| Request {
            departure: 0,
            access: vec![(origin, 0)],
            egress: Vec::new(),
            active_services: vec![true],
            active_services_previous: Vec::new(),
            max_transfers: 3,
        })
        .collect();
    let cells = engine.frontier_matrix(
        &requests,
        &destinations,
        &[],
        false,
        destinations.len(),
        1000,
        1e-6,
    );
    let keys = |journeys: &[Journey]| -> Vec<(u32, u32, u32, f64)> {
        journeys
            .iter()
            .map(|journey| {
                (
                    journey.departure,
                    journey.arrival,
                    journey.rides() as u32,
                    grams_of(journey, &geometry, &factors),
                )
            })
            .collect()
    };
    for (request, row) in requests.iter().zip(&cells) {
        for (&destination, cell) in destinations.iter().zip(row) {
            let mut one_pair = request.clone();
            one_pair.egress = vec![(destination, 0)];
            let journeys = engine.route_range(&one_pair, 1000, 1e-6);
            assert_eq!(keys(cell), keys(&journeys));
        }
    }
    // Non-vacuity: the forked fixture's fast-dirty and slow-clean
    // journeys both survive from stop 0 to stop 3, the repeated
    // destination shares its first slot's cell, and the reverse origin
    // reaches nothing.
    assert_eq!(cells[0][0].len(), 2);
    assert_eq!(keys(&cells[0][2]), keys(&cells[0][0]));
    assert!(cells[1].iter().all(Vec::is_empty));
}

#[test]
fn a_prebuilt_transfer_set_answers_like_for_date() {
    let (timetable, geometry) = forked();
    let factors = [10.0, 100.0, 10.0];
    let footpaths = Transfers::empty(4);
    let set = McTbtrEngine::transfers_for_date(&timetable, &geometry, &factors, &[true], &[]);
    let owned = McTbtrEngine::for_date(&timetable, &footpaths, &geometry, &factors, &[true], &[]);
    let borrowed = McTbtrEngine::from_set(
        &timetable,
        &footpaths,
        &geometry,
        &factors,
        &[true],
        &[],
        &set,
    );
    let request = Request {
        departure: 0,
        access: vec![(StopIdx(0), 0)],
        egress: vec![(StopIdx(3), 0)],
        active_services: vec![true],
        active_services_previous: Vec::new(),
        max_transfers: 3,
    };
    let over_owned = owned.route_range(&request, 1000, 1e-6);
    let over_borrowed = borrowed.route_range(&request, 1000, 1e-6);
    assert!(!over_owned.is_empty());
    assert_eq!(
        triples(&over_owned, &geometry, &factors),
        triples(&over_borrowed, &geometry, &factors)
    );
}

#[test]
fn cleaner_chains_match_the_naive_suffix_walk() {
    // One line, six trips with unresolved, dirtier, equal, and cleaner
    // factors interleaved; the chain from any first boardable rank must
    // visit exactly the trips the naive strictly-cleaner walk keeps.
    let mut builder = TimetableBuilder::new(2);
    let a = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    for (index, start) in (0..6).map(|i| (i, 100 + i * 100)) {
        builder
            .add_trip(a, vec![time(start), time(start + 50)], index, 0)
            .unwrap();
    }
    let timetable = builder.finish();
    let view = DayView::universal(&timetable);
    let factors = [f64::NAN, 100.0, 120.0, 80.0, 80.0, 10.0];
    let chains = CleanerChains::build(&view, &factors);
    let naive = |first: u32| -> Vec<u32> {
        let mut kept = Vec::new();
        let mut cleanest = f64::INFINITY;
        for rank in first..view.line_trips(0).end {
            let factor = factors[view.backing(ViewTrip(rank)).0 as usize];
            if !factor.is_finite() || factor >= cleanest {
                continue;
            }
            cleanest = factor;
            kept.push(rank);
        }
        kept
    };
    for first in 0..6 {
        assert_eq!(
            chains.candidates(first).collect::<Vec<_>>(),
            naive(first),
            "first boardable rank {first}"
        );
    }
    assert_eq!(chains.candidates(0).collect::<Vec<_>>(), vec![1, 3, 5]);
}

#[test]
fn the_frontier_matrix_serves_a_slot_only_the_transfer_reaches() {
    // A clean direct line serves the first destination early and
    // dominates the through trip's alight there — but the through
    // trip's later alight is the only path onward to the second
    // destination, over a precomputed transfer. The batched search
    // runs unpruned, so a slot served by nothing but the dominated
    // trip's continuation keeps its cell.
    let mut builder = TimetableBuilder::new(4);
    let direct = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let through = builder
        .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 1)
        .unwrap();
    let onward = builder.add_pattern(&[StopIdx(2), StopIdx(3)], 2).unwrap();
    builder
        .add_trip(direct, vec![time(100), time(200)], 0, 0)
        .unwrap();
    builder
        .add_trip(through, vec![time(110), time(300), time(400)], 1, 0)
        .unwrap();
    builder
        .add_trip(onward, vec![time(450), time(600)], 2, 0)
        .unwrap();
    let timetable = builder.finish();
    let geometry = TripGeometry::from_trips(
        &timetable,
        vec![
            (TripIdx(0), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
            (
                TripIdx(1),
                vec![0.0, 500.0, 1000.0],
                DistanceProvenance::CrowFly,
            ),
            (TripIdx(2), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
        ],
    )
    .unwrap();
    let factors = [10.0, 50.0, 10.0];
    let footpaths = Transfers::empty(4);
    let engine = McTbtrEngine::for_date(&timetable, &footpaths, &geometry, &factors, &[true], &[]);
    let destinations = [StopIdx(1), StopIdx(3)];
    let requests = vec![Request {
        departure: 0,
        access: vec![(StopIdx(0), 0)],
        egress: Vec::new(),
        active_services: vec![true],
        active_services_previous: Vec::new(),
        max_transfers: 2,
    }];
    let cells = engine.frontier_matrix(
        &requests,
        &destinations,
        &[],
        false,
        destinations.len(),
        1000,
        1e-6,
    );
    // The second cell holds both transfer journeys: the plain
    // through-and-onward ride, and the cleaner three-ride alternative
    // that rides the direct line first and only the through trip's
    // second (shorter, hence cleaner) half.
    let mut reached: Vec<(u32, u32, u32)> = cells[0][1]
        .iter()
        .map(|journey| (journey.departure, journey.arrival, journey.rides() as u32))
        .collect();
    reached.sort_unstable();
    assert_eq!(reached, vec![(100, 600, 3), (110, 600, 2)]);
    // … and both cells equal the one-pair profile.
    for (&destination, cell) in destinations.iter().zip(&cells[0]) {
        let mut one_pair = requests[0].clone();
        one_pair.egress = vec![(destination, 0)];
        let journeys = engine.route_range(&one_pair, 1000, 1e-6);
        let keys = |journeys: &[Journey]| -> Vec<(u32, u32, u32)> {
            journeys
                .iter()
                .map(|journey| (journey.departure, journey.arrival, journey.rides() as u32))
                .collect()
        };
        assert_eq!(keys(cell), keys(&journeys));
    }
}

#[test]
fn dominated_through_journeys_stay_out_of_every_cell() {
    // Clean direct lines serve both destinations in round one, so the
    // dirty through trip's continuation is dominated at every
    // destination: whatever the search explores, no cell may carry
    // the through journey — the one-pair parity below pins it.
    let mut builder = TimetableBuilder::new(4);
    let direct_near = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let direct_far = builder.add_pattern(&[StopIdx(0), StopIdx(3)], 1).unwrap();
    let through = builder
        .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 2)
        .unwrap();
    let onward = builder.add_pattern(&[StopIdx(2), StopIdx(3)], 3).unwrap();
    builder
        .add_trip(direct_near, vec![time(100), time(150)], 0, 0)
        .unwrap();
    builder
        .add_trip(direct_far, vec![time(100), time(160)], 1, 0)
        .unwrap();
    builder
        .add_trip(through, vec![time(100), time(400), time(500)], 2, 0)
        .unwrap();
    builder
        .add_trip(onward, vec![time(550), time(700)], 3, 0)
        .unwrap();
    let timetable = builder.finish();
    let geometry = TripGeometry::from_trips(
        &timetable,
        vec![
            (TripIdx(0), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
            (TripIdx(1), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
            (
                TripIdx(2),
                vec![0.0, 1000.0, 2000.0],
                DistanceProvenance::CrowFly,
            ),
            (TripIdx(3), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
        ],
    )
    .unwrap();
    let factors = [10.0, 10.0, 50.0, 50.0];
    let footpaths = Transfers::empty(4);
    let engine = McTbtrEngine::for_date(&timetable, &footpaths, &geometry, &factors, &[true], &[]);
    let destinations = [StopIdx(1), StopIdx(3)];
    let requests = vec![Request {
        departure: 0,
        access: vec![(StopIdx(0), 0)],
        egress: Vec::new(),
        active_services: vec![true],
        active_services_previous: Vec::new(),
        max_transfers: 2,
    }];
    let cells = engine.frontier_matrix(
        &requests,
        &destinations,
        &[],
        false,
        destinations.len(),
        1000,
        1e-6,
    );
    let keys = |journeys: &[Journey]| -> Vec<(u32, u32, u32)> {
        journeys
            .iter()
            .map(|journey| (journey.departure, journey.arrival, journey.rides() as u32))
            .collect()
    };
    // Only the clean directs survive — the through-and-onward journey
    // is dominated at both slots and reaches neither cell.
    assert_eq!(keys(&cells[0][0]), vec![(100, 150, 1)]);
    assert_eq!(keys(&cells[0][1]), vec![(100, 160, 1)]);
    for (&destination, cell) in destinations.iter().zip(&cells[0]) {
        let mut one_pair = requests[0].clone();
        one_pair.egress = vec![(destination, 0)];
        let journeys = engine.route_range(&one_pair, 1000, 1e-6);
        assert_eq!(keys(cell), keys(&journeys));
    }
}

/// The dense-closure fixture (nine stops): two nondominated rides feed
/// closure source stop 1 (four outgoing edges), line C's alights at
/// source stop 6 interleave in segment order, and two onward lines to
/// stop 8 depart different closure targets.
fn dense_closure() -> (Timetable, TripGeometry, Transfers) {
    let mut builder = TimetableBuilder::new(9);
    let dirty = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let clean = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 1).unwrap();
    let second = builder.add_pattern(&[StopIdx(0), StopIdx(6)], 2).unwrap();
    let onward_d = builder.add_pattern(&[StopIdx(2), StopIdx(8)], 3).unwrap();
    let onward_e = builder.add_pattern(&[StopIdx(3), StopIdx(8)], 4).unwrap();
    builder
        .add_trip(dirty, vec![time(0), time(100)], 0, 0)
        .unwrap();
    builder
        .add_trip(dirty, vec![time(300), time(400)], 1, 0)
        .unwrap();
    builder
        .add_trip(clean, vec![time(50), time(150)], 2, 0)
        .unwrap();
    builder
        .add_trip(clean, vec![time(350), time(450)], 3, 0)
        .unwrap();
    builder
        .add_trip(second, vec![time(10), time(90)], 4, 0)
        .unwrap();
    builder
        .add_trip(onward_d, vec![time(300), time(500)], 5, 0)
        .unwrap();
    builder
        .add_trip(onward_d, vec![time(600), time(800)], 6, 0)
        .unwrap();
    builder
        .add_trip(onward_e, vec![time(280), time(520)], 7, 0)
        .unwrap();
    builder
        .add_trip(onward_e, vec![time(620), time(860)], 8, 0)
        .unwrap();
    let timetable = builder.finish();
    let geometry = TripGeometry::from_trips(
        &timetable,
        (0..9)
            .map(|trip| {
                (
                    TripIdx(trip),
                    vec![0.0, 1000.0],
                    DistanceProvenance::CrowFly,
                )
            })
            .collect(),
    )
    .unwrap();
    let footpaths = Transfers::from_edges(
        9,
        &[
            (StopIdx(1), StopIdx(2), 60, 60.0),
            (StopIdx(1), StopIdx(3), 90, 90.0),
            (StopIdx(1), StopIdx(4), 120, 120.0),
            (StopIdx(1), StopIdx(5), 150, 150.0),
            (StopIdx(6), StopIdx(3), 40, 40.0),
            (StopIdx(6), StopIdx(7), 80, 80.0),
        ],
    )
    .unwrap();
    (timetable, geometry, footpaths)
}

/// Dirty line 100, clean line 10, second source 50, onward D 20, E 30.
const DENSE_FACTORS: [f64; 9] = [100.0, 100.0, 10.0, 10.0, 50.0, 20.0, 20.0, 30.0, 30.0];

fn dense_request(egress: StopIdx) -> Request {
    Request {
        departure: 0,
        access: vec![(StopIdx(0), 0)],
        egress: vec![(egress, 0)],
        active_services: vec![true; 9],
        active_services_previous: vec![false; 9],
        max_transfers: 2,
    }
}

/// A journey's profile coordinates: (departure, arrival, grams, rides).
fn coordinates(
    journeys: &[Journey],
    geometry: &TripGeometry,
    factors: &[f64],
) -> Vec<(u32, u32, f64, u32)> {
    let mut list: Vec<(u32, u32, f64, u32)> = journeys
        .iter()
        .map(|journey| {
            (
                journey.departure,
                journey.arrival,
                grams_of(journey, geometry, factors),
                journey.rides() as u32,
            )
        })
        .collect();
    list.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then(a.1.cmp(&b.1))
            .then(a.2.partial_cmp(&b.2).unwrap())
    });
    list
}

#[test]
fn dense_closure_profile_matches_the_exhaustive_oracle() {
    let (timetable, geometry, footpaths) = dense_closure();
    let view = DayView::universal(&timetable);
    let request = dense_request(StopIdx(8));
    // The oracle per departure candidate, folded under profile
    // dominance: a point survives if no point of an equal-or-later
    // departure arrives no later with no more grams.
    let departures = [350u32, 300, 50, 10, 0];
    let mut points: Vec<(u32, u32, f64, u32)> = Vec::new();
    for &departure in &departures {
        for point in pareto_oracle(
            &view,
            &timetable,
            &footpaths,
            &geometry,
            &DENSE_FACTORS,
            departure,
            &request.access,
            &request.egress,
            request.max_transfers,
        ) {
            points.push((departure, point.arrival, point.grams, point.rides));
        }
    }
    let folded: Vec<(u32, u32, f64, u32)> = points
        .iter()
        .filter(|a| {
            !points.iter().any(|b| {
                b.0 >= a.0 && b.1 <= a.1 && b.2 <= a.2 && (b.0 > a.0 || b.1 < a.1 || b.2 < a.2)
            })
        })
        .copied()
        .collect();
    let mut expected = folded;
    expected.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then(a.1.cmp(&b.1))
            .then(a.2.partial_cmp(&b.2).unwrap())
    });
    expected.dedup();
    let engine = McTbtrEngine::for_date(
        &timetable,
        &footpaths,
        &geometry,
        &DENSE_FACTORS,
        &request.active_services,
        &request.active_services_previous,
    );
    let tbtr = engine.route_range(&request, 400, 1e-6);
    assert_eq!(
        coordinates(&tbtr, &geometry, &DENSE_FACTORS),
        expected,
        "mctbtr"
    );
    let raptor = mcraptor::route_range(
        &view,
        &timetable,
        &footpaths,
        &geometry,
        &DENSE_FACTORS,
        &request,
        400,
        1e-6,
        0,
        None,
        &[],
        None,
    );
    assert_eq!(
        coordinates(&raptor, &geometry, &DENSE_FACTORS),
        expected,
        "mcraptor"
    );
}

#[test]
fn dense_closure_frontier_matrix_matches_one_pair_queries() {
    let (timetable, geometry, footpaths) = dense_closure();
    let engine = McTbtrEngine::for_date(
        &timetable,
        &footpaths,
        &geometry,
        &DENSE_FACTORS,
        &[true; 9],
        &[],
    );
    // Direct-alight, closure-walk, and WalkBoard-only destinations,
    // with the last slot repeated.
    let destinations = [StopIdx(1), StopIdx(4), StopIdx(8), StopIdx(8)];
    for bucket in [1e-6, 25.0] {
        let request = Request {
            departure: 0,
            access: vec![(StopIdx(0), 0)],
            egress: Vec::new(),
            active_services: vec![true; 9],
            active_services_previous: vec![],
            max_transfers: 2,
        };
        let cells = engine.frontier_matrix(
            std::slice::from_ref(&request),
            &destinations,
            &[],
            false,
            destinations.len(),
            400,
            bucket,
        );
        for (&destination, cell) in destinations.iter().zip(&cells[0]) {
            let mut one_pair = request.clone();
            one_pair.egress = vec![(destination, 0)];
            let journeys = engine.route_range(&one_pair, 400, bucket);
            assert_eq!(
                coordinates(cell, &geometry, &DENSE_FACTORS),
                coordinates(&journeys, &geometry, &DENSE_FACTORS),
                "destination {destination:?} at bucket {bucket}"
            );
        }
    }
}

/// A leg's full identity for frozen-journey pins.
fn leg_keys(journey: &Journey) -> Vec<(u8, u32, u32, u32, u32)> {
    journey
        .legs
        .iter()
        .map(|leg| match *leg {
            Leg::Access {
                to_stop,
                departure,
                arrival,
            } => (0, to_stop.0, 0, departure, arrival),
            Leg::Transit {
                trip,
                board_stop,
                alight_stop,
                board_time,
                alight_time,
                ..
            } => (
                1,
                board_stop.0 * 100 + alight_stop.0,
                trip.0,
                board_time,
                alight_time,
            ),
            Leg::Transfer {
                from_stop,
                to_stop,
                departure,
                arrival,
            } => (2, from_stop.0 * 100 + to_stop.0, 0, departure, arrival),
            Leg::Egress {
                from_stop,
                departure,
                arrival,
            } => (3, from_stop.0, 0, departure, arrival),
        })
        .collect()
}

#[test]
fn default_bucket_preserves_dense_closure_journeys() {
    // The complete legacy journeys at the default 25 g bucket, every
    // leg included: the loop rewrites must reproduce them exactly.
    let (timetable, geometry, footpaths) = dense_closure();
    let engine = McTbtrEngine::for_date(
        &timetable,
        &footpaths,
        &geometry,
        &DENSE_FACTORS,
        &[true; 9],
        &[],
    );
    let journeys = engine.route_range(&dense_request(StopIdx(8)), 400, 25.0);
    type Frozen = (u32, u32, usize, Vec<(u8, u32, u32, u32, u32)>);
    let frozen: Vec<Frozen> = journeys
        .iter()
        .map(|journey| {
            (
                journey.departure,
                journey.arrival,
                journey.rides(),
                leg_keys(journey),
            )
        })
        .collect();
    assert_eq!(
        frozen,
        vec![
            (
                50,
                500,
                2,
                vec![
                    (0, 0, 0, 50, 50),
                    (1, 1, 2, 50, 150),
                    (2, 102, 0, 150, 210),
                    (1, 208, 5, 300, 500),
                    (3, 8, 0, 500, 500),
                ],
            ),
            (
                350,
                800,
                2,
                vec![
                    (0, 0, 0, 350, 350),
                    (1, 1, 3, 350, 450),
                    (2, 102, 0, 450, 510),
                    (1, 208, 6, 600, 800),
                    (3, 8, 0, 800, 800),
                ],
            ),
        ]
    );
}

#[test]
fn equal_cost_closure_ancestries_are_coordinate_equivalent() {
    // Two distinct ride-walk-ride ancestries with identical departure,
    // arrival, exact grams, and rides. The legacy engine picks one
    // representative; after batching or grouping either whitelisted
    // ancestry is permitted, but the coordinates must not move.
    let mut builder = TimetableBuilder::new(6);
    let p = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let q = builder.add_pattern(&[StopIdx(0), StopIdx(2)], 1).unwrap();
    let r = builder.add_pattern(&[StopIdx(3), StopIdx(5)], 2).unwrap();
    let s = builder.add_pattern(&[StopIdx(4), StopIdx(5)], 3).unwrap();
    builder.add_trip(p, vec![time(0), time(100)], 0, 0).unwrap();
    builder.add_trip(q, vec![time(0), time(120)], 1, 0).unwrap();
    builder
        .add_trip(r, vec![time(300), time(500)], 2, 0)
        .unwrap();
    builder
        .add_trip(s, vec![time(300), time(500)], 3, 0)
        .unwrap();
    let timetable = builder.finish();
    let geometry = TripGeometry::from_trips(
        &timetable,
        vec![
            (TripIdx(0), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
            (TripIdx(1), vec![0.0, 1500.0], DistanceProvenance::CrowFly),
            (TripIdx(2), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
            (TripIdx(3), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
        ],
    )
    .unwrap();
    let footpaths = Transfers::from_edges(
        6,
        &[
            (StopIdx(1), StopIdx(3), 60, 60.0),
            (StopIdx(2), StopIdx(4), 60, 60.0),
        ],
    )
    .unwrap();
    // 20 g/pkm x 1.0 km + 30 x 1.0 == 20 x 1.5 + 20 x 1.0 == 50 g.
    let factors = [20.0, 20.0, 30.0, 20.0];
    let engine =
        McTbtrEngine::for_date(&timetable, &footpaths, &geometry, &factors, &[true; 6], &[]);
    let request = Request {
        departure: 0,
        access: vec![(StopIdx(0), 0)],
        egress: vec![(StopIdx(5), 0)],
        active_services: vec![true; 6],
        active_services_previous: vec![],
        max_transfers: 2,
    };
    let journeys = engine.route_range(&request, 100, 1e-6);
    assert_eq!(
        coordinates(&journeys, &geometry, &factors),
        vec![(0, 500, 50.0, 2)]
    );
    let via_p = vec![
        (0, 0, 0, 0, 0),
        (1, 1, 0, 0, 100),
        (2, 103, 0, 100, 160),
        (1, 305, 2, 300, 500),
        (3, 5, 0, 500, 500),
    ];
    let via_q = vec![
        (0, 0, 0, 0, 0),
        (1, 2, 1, 0, 120),
        (2, 204, 0, 120, 180),
        (1, 405, 3, 300, 500),
        (3, 5, 0, 500, 500),
    ];
    let keys = leg_keys(&journeys[0]);
    assert!(
        keys == via_p || keys == via_q,
        "unexpected ancestry: {keys:?}"
    );
}

#[test]
fn profile_eviction_does_not_remove_a_cleaner_earlier_departure() {
    // Descending passes: the 300-departure pass boards the onward trip
    // at position 1 and scans it; the 0-departure pass later admits a
    // better boarding at position 0, evicting that trip-bag entry. The
    // already-scanned journey must survive in the profile.
    let mut builder = TimetableBuilder::new(6);
    let slow = builder.add_pattern(&[StopIdx(0), StopIdx(2)], 0).unwrap();
    let early = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 1).unwrap();
    let onward = builder
        .add_pattern(&[StopIdx(1), StopIdx(2), StopIdx(5)], 2)
        .unwrap();
    builder
        .add_trip(slow, vec![time(300), time(450)], 0, 0)
        .unwrap();
    builder
        .add_trip(early, vec![time(0), time(80)], 1, 0)
        .unwrap();
    builder
        .add_trip(onward, vec![time(500), time(600), time(700)], 2, 0)
        .unwrap();
    let timetable = builder.finish();
    let geometry = TripGeometry::from_trips(
        &timetable,
        vec![
            (TripIdx(0), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
            (TripIdx(1), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
            (
                TripIdx(2),
                vec![0.0, 1000.0, 2000.0],
                DistanceProvenance::CrowFly,
            ),
        ],
    )
    .unwrap();
    let footpaths = Transfers::empty(6);
    // Dirty slow line, clean early line: the position-0 boarding
    // dominates and evicts the already-scanned position-1 entry.
    let factors = [100.0, 10.0, 20.0];
    let view = DayView::universal(&timetable);
    let request = Request {
        departure: 0,
        access: vec![(StopIdx(0), 0)],
        egress: vec![(StopIdx(5), 0)],
        active_services: vec![true; 6],
        active_services_previous: vec![],
        max_transfers: 2,
    };
    let mut points: Vec<(u32, u32, f64, u32)> = Vec::new();
    for departure in [300u32, 0] {
        for point in pareto_oracle(
            &view,
            &timetable,
            &footpaths,
            &geometry,
            &factors,
            departure,
            &request.access,
            &request.egress,
            request.max_transfers,
        ) {
            points.push((departure, point.arrival, point.grams, point.rides));
        }
    }
    let mut expected: Vec<(u32, u32, f64, u32)> = points
        .iter()
        .filter(|a| {
            !points.iter().any(|b| {
                b.0 >= a.0 && b.1 <= a.1 && b.2 <= a.2 && (b.0 > a.0 || b.1 < a.1 || b.2 < a.2)
            })
        })
        .copied()
        .collect();
    expected.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then(a.1.cmp(&b.1))
            .then(a.2.partial_cmp(&b.2).unwrap())
    });
    expected.dedup();
    assert_eq!(expected.len(), 2, "both passes must survive the fold");
    let engine =
        McTbtrEngine::for_date(&timetable, &footpaths, &geometry, &factors, &[true; 6], &[]);
    let tbtr = engine.route_range(&request, 400, 1e-6);
    assert_eq!(coordinates(&tbtr, &geometry, &factors), expected);
}
