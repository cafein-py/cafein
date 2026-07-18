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
    let tbtr = engine.route(&request, 1e-6, None);
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
fn the_auto_policy_switch_never_changes_fixture_results() {
    use crate::router::{auto_mc_tbtr, auto_time_tbtr};

    // Both engines match the oracle on the fixture (asserted inside), so
    // whichever engine the auto policy selects for a cache state answers
    // identically — the policy switch can never change results.
    let (timetable, geometry) = forked();
    let footpaths = Transfers::empty(4);
    engines_agree(
        &timetable,
        &geometry,
        &footpaths,
        &[50.0, 100.0, 10.0],
        StopIdx(0),
        StopIdx(3),
        3,
    );
    // The multicriteria decision table over that equivalence: uncached,
    // date-mismatched, factor-mismatched, and semantic-fallback states run
    // McRAPTOR; only a matching cache with a supported query runs McTBTR.
    let fingerprint = 7;
    let cached = Some(("2022-02-22", fingerprint));
    assert!(!auto_mc_tbtr(None, "2022-02-22", fingerprint, false));
    assert!(!auto_mc_tbtr(
        Some(("2022-02-21", fingerprint)),
        "2022-02-22",
        fingerprint,
        false
    ));
    assert!(!auto_mc_tbtr(
        Some(("2022-02-22", 8)),
        "2022-02-22",
        fingerprint,
        false
    ));
    assert!(!auto_mc_tbtr(cached, "2022-02-22", fingerprint, true));
    assert!(auto_mc_tbtr(cached, "2022-02-22", fingerprint, false));
    // The time-only table: only a date-matching cached set runs TBTR.
    assert!(!auto_time_tbtr(None, "2022-02-22"));
    assert!(!auto_time_tbtr(Some("2022-02-21"), "2022-02-22"));
    assert!(auto_time_tbtr(Some("2022-02-22"), "2022-02-22"));
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
    let journeys = engine.route_range(&request, 200, 1e-6, None);
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
            let journeys = engine.route_range(&one_pair, 1000, 1e-6, None);
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
    let over_owned = owned.route_range(&request, 1000, 1e-6, None);
    let over_borrowed = borrowed.route_range(&request, 1000, 1e-6, None);
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
        let journeys = engine.route_range(&one_pair, 1000, 1e-6, None);
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
        let journeys = engine.route_range(&one_pair, 1000, 1e-6, None);
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
    let tbtr = engine.route_range(&request, 400, 1e-6, None);
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
            let journeys = engine.route_range(&one_pair, 400, bucket, None);
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
    let journeys = engine.route_range(&dense_request(StopIdx(8)), 400, 25.0, None);
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
    let journeys = engine.route_range(&request, 100, 1e-6, None);
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
    let tbtr = engine.route_range(&request, 400, 1e-6, None);
    assert_eq!(coordinates(&tbtr, &geometry, &factors), expected);
}

#[test]
fn tripbag_eviction_cancels_a_pending_segment() {
    let mut states = [SegmentState::Pending];
    let mut bag = TripBag::default();
    assert!(bag.admits(5, 100.0, 4, 0, |_| panic!("nothing to evict yet")));
    let mut cancelled = Vec::new();
    assert!(bag.admits(3, 50.0, 2, 1, |token| {
        if states[token as usize] == SegmentState::Pending {
            states[token as usize] = SegmentState::Cancelled;
        }
        cancelled.push(token);
    }));
    assert_eq!(cancelled, vec![0]);
    assert_eq!(states[0], SegmentState::Cancelled);
}

#[test]
fn tripbag_eviction_does_not_cancel_a_scanned_segment() {
    let mut states = [SegmentState::Scanned];
    let mut bag = TripBag::default();
    assert!(bag.admits(5, 100.0, 4, 0, |_| ()));
    assert!(bag.admits(3, 50.0, 2, 1, |token| {
        if states[token as usize] == SegmentState::Pending {
            states[token as usize] = SegmentState::Cancelled;
        }
    }));
    assert_eq!(states[0], SegmentState::Scanned);
}

#[test]
fn tripbag_rejection_does_not_allocate_a_segment() {
    let mut bag = TripBag::default();
    assert!(bag.admits(3, 50.0, 2, 0, |_| ()));
    let mut fired = false;
    assert!(!bag.admits(5, 100.0, 4, 1, |_| fired = true));
    assert!(!fired, "a rejection must not evict anything");
    assert_eq!(bag.entries.len(), 1);
    assert_eq!(bag.entries[0].pending_segment, 0);
}

#[test]
fn one_admission_can_cancel_multiple_pending_segments() {
    let mut bag = TripBag::default();
    assert!(bag.admits(5, 100.0, 5, 0, |_| ()));
    assert!(bag.admits(6, 80.0, 4, 1, |_| ()));
    assert_eq!(bag.entries.len(), 2);
    let mut cancelled = Vec::new();
    assert!(bag.admits(3, 20.0, 3, 2, |token| cancelled.push(token)));
    cancelled.sort_unstable();
    assert_eq!(cancelled, vec![0, 1]);
    assert_eq!(bag.entries.len(), 1);
}

/// Two access lines whose riders both walk to stop 3 and board the
/// same onward trip at position 0. The dirtier, later walker boards
/// first (its source stop is swept first); the cleaner walker's
/// admission then evicts that still-pending onward segment.
fn cancelling() -> (Timetable, TripGeometry, Transfers) {
    let mut builder = TimetableBuilder::new(6);
    let dirty = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let clean = builder.add_pattern(&[StopIdx(0), StopIdx(2)], 1).unwrap();
    let onward = builder.add_pattern(&[StopIdx(3), StopIdx(5)], 2).unwrap();
    builder
        .add_trip(dirty, vec![time(0), time(100)], 0, 0)
        .unwrap();
    builder
        .add_trip(clean, vec![time(0), time(90)], 1, 0)
        .unwrap();
    builder
        .add_trip(onward, vec![time(300), time(500)], 2, 0)
        .unwrap();
    let timetable = builder.finish();
    let geometry = TripGeometry::from_trips(
        &timetable,
        vec![
            (TripIdx(0), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
            (TripIdx(1), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
            (TripIdx(2), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
        ],
    )
    .unwrap();
    let footpaths = Transfers::from_edges(
        6,
        &[
            (StopIdx(1), StopIdx(3), 60, 60.0),
            (StopIdx(2), StopIdx(3), 60, 60.0),
        ],
    )
    .unwrap();
    (timetable, geometry, footpaths)
}

const CANCELLING_FACTORS: [f64; 3] = [10.0, 5.0, 20.0];

fn cancelling_request() -> Request {
    Request {
        departure: 0,
        access: vec![(StopIdx(0), 0)],
        egress: vec![(StopIdx(5), 0)],
        active_services: vec![true; 6],
        active_services_previous: vec![],
        max_transfers: 2,
    }
}

#[test]
fn cancelled_segments_never_become_destination_leaves() {
    let (timetable, geometry, footpaths) = cancelling();
    let engine = McTbtrEngine::for_date(
        &timetable,
        &footpaths,
        &geometry,
        &CANCELLING_FACTORS,
        &[true; 6],
        &[],
    );
    let request = cancelling_request();
    let mut fold = None;
    let mut frontier = None;
    let mut stats = SearchStats::default();
    let (arena, destination) = engine.passes(
        &request,
        &[0],
        1e-6,
        None,
        &mut fold,
        &mut frontier,
        &mut stats,
    );
    assert_eq!(stats.segments_cancelled_pending, 1);
    assert_eq!(stats.segments_skipped_cancelled, 1);
    // The clean walker's onward segment is the only destination leaf;
    // the debug chain check inside `passes` walked every ancestry.
    assert_eq!(destination.len(), 1);
    let _ = &arena;
    let reached = destination[0];
    assert_eq!((reached.departure, reached.arrival), (0, 500));
    assert!((reached.grams - 25.0).abs() < 1e-9);
    // The full profile agrees with the exhaustive oracle.
    let view = DayView::universal(&timetable);
    let oracle = pareto_oracle(
        &view,
        &timetable,
        &footpaths,
        &geometry,
        &CANCELLING_FACTORS,
        0,
        &request.access,
        &request.egress,
        request.max_transfers,
    );
    let oracle: Vec<(u32, f64, u32)> = oracle
        .iter()
        .map(|point| (point.arrival, point.grams, point.rides))
        .collect();
    let journeys = engine.route(&request, 1e-6, None);
    assert_eq!(triples(&journeys, &geometry, &CANCELLING_FACTORS), oracle);
}

#[test]
fn cancelled_segments_never_appear_as_segment_parents() {
    let (timetable, geometry, footpaths) = cancelling();
    let engine = McTbtrEngine::for_date(
        &timetable,
        &footpaths,
        &geometry,
        &CANCELLING_FACTORS,
        &[true; 6],
        &[],
    );
    let request = cancelling_request();
    let mut fold = None;
    let mut frontier = None;
    let mut stats = SearchStats::default();
    let (arena, destination) = engine.passes(
        &request,
        &[0],
        1e-6,
        None,
        &mut fold,
        &mut frontier,
        &mut stats,
    );
    assert_eq!(stats.segments_cancelled_pending, 1);
    // The cancelled onward boarding is the dirty walker's child.
    let cancelled: Vec<u32> = (0..arena.len() as u32)
        .filter(|&index| {
            matches!(
                arena[index as usize].origin,
                SegOrigin::Walked { parent: 0, .. }
            )
        })
        .collect();
    assert_eq!(cancelled.len(), 1);
    for segment in &arena {
        match segment.origin {
            SegOrigin::Transfer { parent, .. } | SegOrigin::Walked { parent, .. } => {
                assert_ne!(parent, cancelled[0], "a cancelled segment gained a child");
            }
            SegOrigin::Access { .. } => {}
        }
    }
    for arrived in &destination {
        assert_ne!(arrived.leaf, cancelled[0], "a cancelled segment was a leaf");
    }
}

#[test]
fn cleaner_exact_tie_continues_to_later_witness() {
    // A deliberately unreachable order: the refinable exact tie ahead
    // of a non-tied witness. The scan must continue past the tie and
    // let the later witness reject (and swap to the front).
    let mut bag = Bag::from_entries(vec![(100, 0, 5, 50.0, 1), (90, 0, 4, 40.0, 1)]);
    assert!(!bag.insert(100, 45.0, 5, 1));
    assert_eq!(bag.snapshot()[0], (90, 0, 4, 40.0f64.to_bits(), 1));
}

#[test]
fn non_tie_witness_and_refinable_tie_cannot_coexist() {
    // Inserting the covering witness and the refinable tie in either
    // order leaves a one-entry antichain with the same decision.
    for pair in [
        [(90u32, 40.0f64, 4i64, 1u8), (100, 50.0, 5, 1)],
        [(100, 50.0, 5, 1), (90, 40.0, 4, 1)],
    ] {
        let mut bag = Bag::default();
        for (arrival, grams, key, rides) in pair {
            bag.insert(arrival, grams, key, rides);
        }
        assert_eq!(bag.snapshot().len(), 1);
        assert!(!bag.insert(100, 45.0, 5, 1));
    }
}

/// The strict rejection relation on snapshot tuples.
fn strict_rejects(
    entry: (u32, u32, i64, u64, u8),
    arrival: u32,
    key: i64,
    grams: f64,
    rides: u8,
) -> bool {
    let (e_arrival, e_penalty, e_key, e_grams, e_rides) = entry;
    let e_grams = f64::from_bits(e_grams);
    e_key <= key
        && e_rides <= rides
        && e_arrival <= arrival
        && if e_arrival == arrival && e_penalty == 0 && e_key == key && e_rides == rides {
            grams >= e_grams
        } else {
            e_arrival + e_penalty <= arrival
        }
}

/// A deterministic operation stream for the bag property tests.
fn bag_operations(count: usize) -> Vec<(u32, i64, f64, u8)> {
    let mut state = 0x9e3779b97f4a7c15u64;
    let mut next = move |bound: u64| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) % bound
    };
    (0..count)
        .map(|_| {
            let arrival = 100 + next(60) as u32;
            let key = next(8) as i64;
            let grams = key as f64 * 25.0 + next(25) as f64;
            let rides = next(4) as u8;
            (arrival, key, grams, rides)
        })
        .collect()
}

#[test]
fn strict_bag_remains_an_antichain() {
    // Cleaner buckets arrive later (anti-correlated axes), so genuine
    // trade-offs survive and the final bag is nontrivial.
    let mut bag = Bag::default();
    let mut state = 0x2545f4914f6cdd1du64;
    let mut next = move |bound: u64| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) % bound
    };
    for _ in 0..5_000 {
        let key = next(8) as i64;
        let arrival = 100 + (7 - key as u32) * 12 + next(6) as u32;
        let grams = key as f64 * 25.0 + next(25) as f64;
        let rides = next(4) as u8;
        bag.insert(arrival, grams, key, rides);
    }
    let entries = bag.snapshot();
    assert!(entries.len() > 2);
    for (position, &entry) in entries.iter().enumerate() {
        for (other_position, &other) in entries.iter().enumerate() {
            if position != other_position {
                let (arrival, _, key, grams, rides) = other;
                assert!(
                    !strict_rejects(entry, arrival, key, f64::from_bits(grams), rides),
                    "bag entries must be mutually incomparable"
                );
            }
        }
    }
}

#[test]
fn strict_rejection_relation_is_transitive() {
    let pool = bag_operations(120);
    let rejects = |a: (u32, i64, f64, u8), b: (u32, i64, f64, u8)| {
        strict_rejects((a.0, 0, a.1, a.2.to_bits(), a.3), b.0, b.1, b.2, b.3)
    };
    for &a in &pool {
        for &b in &pool {
            for &c in &pool {
                if rejects(a, b) && rejects(b, c) {
                    assert!(rejects(a, c), "D must be transitive: {a:?} {b:?} {c:?}");
                }
            }
        }
    }
}

#[test]
fn rejecting_entry_swaps_to_front() {
    let mut bag = Bag::default();
    // Three mutually incomparable entries.
    assert!(bag.insert(100, 30.0, 1, 2));
    assert!(bag.insert(110, 5.0, 0, 2));
    assert!(bag.insert(90, 55.0, 2, 2));
    // The witness sits in the deepest slot; rejection swaps it front.
    let witness = (110, 0, 0, 5.0f64.to_bits(), 2);
    assert_eq!(bag.snapshot()[2], witness);
    assert!(!bag.insert(120, 6.0, 0, 2));
    assert_eq!(bag.snapshot()[0], witness);
}

#[test]
fn front_rejection_does_not_change_order() {
    let mut bag = Bag::default();
    assert!(bag.insert(100, 30.0, 1, 2));
    assert!(bag.insert(110, 5.0, 0, 2));
    assert!(bag.insert(90, 55.0, 2, 2));
    assert!(!bag.insert(120, 6.0, 0, 2));
    let order = bag.snapshot();
    // The front witness rejects again: no reordering.
    assert!(!bag.insert(125, 7.0, 0, 2));
    assert_eq!(bag.snapshot(), order);
}

#[test]
fn admitted_entry_swaps_to_front() {
    let mut bag = Bag::default();
    assert!(bag.insert(100, 30.0, 1, 2));
    assert!(bag.insert(110, 5.0, 0, 2));
    assert_eq!(bag.snapshot()[0], (110, 0, 0, 5.0f64.to_bits(), 2));
    assert!(bag.insert(90, 55.0, 2, 2));
    assert_eq!(bag.snapshot()[0], (90, 0, 2, 55.0f64.to_bits(), 2));
}

#[test]
fn swap_to_front_preserves_cleaner_exact_tie_refinement() {
    let mut bag = Bag::default();
    assert!(bag.insert(100, 50.0, 5, 1));
    // An incomparable admission swaps ahead of the tie entry.
    assert!(bag.insert(110, 10.0, 0, 3));
    // The cleaner exact tie still replaces its dirtier sibling.
    assert!(bag.insert(100, 45.0, 5, 1));
    let entries = bag.snapshot();
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0], (100, 0, 5, 45.0f64.to_bits(), 1));
    assert!(!entries.contains(&(100, 0, 5, 50.0f64.to_bits(), 1)));
}

#[test]
fn swap_to_front_preserves_the_rides_axis() {
    let mut bag = Bag::default();
    assert!(bag.insert(100, 50.0, 5, 4));
    assert!(bag.insert(110, 10.0, 0, 4));
    // The same point on fewer rides must still admit and evict.
    assert!(bag.insert(100, 50.0, 5, 2));
    let entries = bag.snapshot();
    assert_eq!(entries.len(), 2);
    assert!(entries.contains(&(100, 0, 5, 50.0f64.to_bits(), 2)));
    assert!(!entries.contains(&(100, 0, 5, 50.0f64.to_bits(), 4)));
}

#[test]
fn self_organising_insert_matches_stable_strict_insert() {
    // The unchanged stable-order insert_slack(penalty 0, slack 0) is
    // the reference: identical decisions and identical entry sets
    // after every operation, only the private order may differ.
    let stops = 6usize;
    let mut stable = vec![Bag::default(); stops];
    let mut organised = vec![Bag::default(); stops];
    let mut state = 0x51ab5f2d9e3779b9u64;
    let mut next = move |bound: u64| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) % bound
    };
    for _ in 0..100_000 {
        let stop = next(stops as u64) as usize;
        let arrival = 100 + next(60) as u32;
        let key = next(8) as i64;
        let grams = key as f64 * 25.0 + next(25) as f64;
        let rides = next(4) as u8;
        let expected = stable[stop].insert_slack(arrival, 0, grams, key, rides, 0);
        let admitted = organised[stop].insert(arrival, grams, key, rides);
        assert_eq!(admitted, expected);
        let mut reference = stable[stop].snapshot();
        let mut permuted = organised[stop].snapshot();
        reference.sort_unstable();
        permuted.sort_unstable();
        assert_eq!(reference, permuted);
    }
}

#[test]
fn probed_self_organising_insert_matches_production_insert() {
    // Production and probed variants must perform identical swaps:
    // ordered snapshots, not just sets.
    let mut production = Bag::default();
    let mut probed = Bag::default();
    for (arrival, key, grams, rides) in bag_operations(20_000) {
        let mut probes = InsertProbes::default();
        let expected = production.insert(arrival, grams, key, rides);
        let admitted = probed.insert_probed(arrival, grams, key, rides, &mut probes);
        assert_eq!(admitted, expected);
        assert_eq!(production.snapshot(), probed.snapshot());
    }
}

#[test]
fn probe_counters_identify_front_and_nonfront_rejections() {
    // The stage's counter identities over a random stream:
    // front + swaps == rejections, sum(histogram) == rejections,
    // swap distance == entries examined - rejections.
    let mut bag = Bag::default();
    let (mut front, mut swaps, mut distance) = (0u64, 0u64, 0u64);
    let (mut rejections, mut examined) = (0u64, 0u64);
    let mut histogram = [0u64; 9];
    for (arrival, key, grams, rides) in bag_operations(20_000) {
        let mut probes = InsertProbes::default();
        if !bag.insert_probed(arrival, grams, key, rides, &mut probes) {
            rejections += 1;
            examined += probes.examined as u64;
            if probes.examined == 1 {
                front += 1;
            } else {
                swaps += 1;
                distance += (probes.examined - 1) as u64;
            }
            histogram[depth_bucket(probes.examined)] += 1;
        }
    }
    assert!(front > 0 && swaps > 0);
    assert_eq!(front + swaps, rejections);
    assert_eq!(histogram.iter().sum::<u64>(), rejections);
    assert_eq!(distance, examined - rejections);
}

/// A cost row's fields with NaN-safe float bits, for equality
/// assertions.
type CellKey = (u32, u32, u32, u64, u64, u64, u64);

#[test]
fn repeated_destination_stops_keep_every_matrix_cell() {
    // Duplicate destination stops used to share one last-wins slot:
    // the earlier occurrences' cells vanished from the matrix.
    let (timetable, geometry) = forked();
    let factors = [50.0, 100.0, 10.0];
    let footpaths = Transfers::empty(4);
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
    let engine = McTbtrEngine::for_date(
        &timetable,
        &footpaths,
        &geometry,
        &factors,
        &requests[0].active_services,
        &requests[0].active_services_previous,
    );
    let destinations = [StopIdx(3), StopIdx(0), StopIdx(3)];
    let rows = engine.least_emissions_matrix(&inputs, &requests, &destinations, 600, None, 1e-6);
    let row = &rows[0];
    assert_eq!(
        row.iter().map(|cell| cell.to).collect::<Vec<_>>(),
        [3, 0, 3]
    );
    assert_eq!(row[0].seconds, row[2].seconds);
    assert_eq!(
        row[0].emission_grams.to_bits(),
        row[2].emission_grams.to_bits()
    );
}

#[test]
fn the_transfer_cap_saturates_at_the_ride_count_limit() {
    // The trip bags hold ride counts in a `u8`: 255 transfers used to
    // wrap the last round's rides to zero.
    let (timetable, geometry) = forked();
    let factors = [50.0, 100.0, 10.0];
    let footpaths = Transfers::empty(4);
    let inputs = CostInputs {
        geometry: &geometry,
        factors: &factors,
        leg_geometry: None,
        with_geometry: false,
        fares: None,
    };
    let request = |max_transfers: u8| {
        vec![Request {
            departure: 0,
            access: vec![(StopIdx(0), 0)],
            egress: Vec::new(),
            active_services: vec![true],
            active_services_previous: vec![false],
            max_transfers,
        }]
    };
    let engine = McTbtrEngine::for_date(
        &timetable,
        &footpaths,
        &geometry,
        &factors,
        &[true],
        &[false],
    );
    let capped =
        engine.least_emissions_matrix(&inputs, &request(254), &[StopIdx(3)], 600, None, 1e-6);
    let saturated =
        engine.least_emissions_matrix(&inputs, &request(255), &[StopIdx(3)], 600, None, 1e-6);
    assert!(!capped[0].is_empty());
    let cells = |rows: &Vec<Vec<CostRow>>| -> Vec<Vec<CellKey>> {
        rows.iter()
            .map(|row| {
                row.iter()
                    .map(|cell| {
                        (
                            cell.to,
                            cell.seconds,
                            cell.rides,
                            cell.transit_meters.to_bits(),
                            cell.walk_meters.to_bits(),
                            cell.emission_grams.to_bits(),
                            cell.fare.to_bits(),
                        )
                    })
                    .collect()
            })
            .collect()
    };
    assert_eq!(cells(&capped), cells(&saturated));
}

#[test]
fn max_slower_bands_match_mcraptor() {
    // The strict frontier at the fork holds the fast dirty journey and
    // the slow clean one; the band keeps or drops the slow entry
    // identically on both engines.
    let (timetable, geometry) = forked();
    let factors = [50.0, 100.0, 10.0];
    let footpaths = Transfers::empty(4);
    let view = DayView::universal(&timetable);
    let request = Request {
        departure: 0,
        access: vec![(StopIdx(0), 0)],
        egress: vec![(StopIdx(3), 0)],
        active_services: vec![true],
        active_services_previous: vec![false],
        max_transfers: 3,
    };
    let engine = McTbtrEngine::for_date(
        &timetable,
        &footpaths,
        &geometry,
        &factors,
        &request.active_services,
        &request.active_services_previous,
    );
    let mut sizes = Vec::new();
    for band in [None, Some(600), Some(0)] {
        let raptor = mcraptor::route(
            &view,
            &timetable,
            &footpaths,
            &geometry,
            &factors,
            &request,
            1e-6,
            0,
            None,
            &[],
            band,
        );
        let tbtr = engine.route(&request, 1e-6, band);
        assert!(!raptor.is_empty(), "band {band:?}");
        assert_eq!(
            triples(&tbtr, &geometry, &factors),
            triples(&raptor, &geometry, &factors),
            "band {band:?}"
        );
        sizes.push(raptor.len());
    }
    // The band genuinely bites: the unrestricted frontier keeps the
    // slow clean journey, the zero band only the fastest.
    assert!(sizes[0] > sizes[2]);
    for band in [None, Some(600), Some(0)] {
        let raptor = mcraptor::route_range(
            &view,
            &timetable,
            &footpaths,
            &geometry,
            &factors,
            &request,
            600,
            1e-6,
            0,
            None,
            &[],
            band,
        );
        let tbtr = engine.route_range(&request, 600, 1e-6, band);
        assert_eq!(
            triples(&tbtr, &geometry, &factors),
            triples(&raptor, &geometry, &factors),
            "window band {band:?}"
        );
    }
}
