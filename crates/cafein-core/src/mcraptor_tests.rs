use super::*;
use crate::exhaustive::pareto_oracle;
use crate::geometry::DistanceProvenance;
use crate::timetable::{StopTime, TimetableBuilder, TripIdx};

fn time(at: u32) -> StopTime {
    StopTime {
        arrival: at,
        departure: at,
    }
}

fn request(access: StopIdx, egress: StopIdx, max_transfers: u8) -> Request {
    Request {
        departure: 0,
        access: vec![(access, 0)],
        egress: vec![(egress, 0)],
        active_services: Vec::new(),
        active_services_previous: Vec::new(),
        max_transfers,
    }
}

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
    let mut triples: Vec<(u32, f64, u32)> = journeys
        .iter()
        .map(|journey| {
            (
                journey.arrival,
                grams_of(journey, geometry, factors),
                journey.rides() as u32,
            )
        })
        .collect();
    triples.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.partial_cmp(&b.1).unwrap()));
    triples
}

fn oracle_triples(points: &[crate::exhaustive::ParetoPoint]) -> Vec<(u32, f64, u32)> {
    points
        .iter()
        .map(|point| (point.arrival, point.grams, point.rides))
        .collect()
}

/// The exhaustive oracle's frontier fixture: a fast dirty direct
/// line, a slow clean one, and a cleaner-but-slower combination
/// over a transfer.
fn frontier_fixture() -> (Timetable, TripGeometry, [f64; 4]) {
    let mut builder = TimetableBuilder::new(4);
    let dirty = builder.add_pattern(&[StopIdx(0), StopIdx(3)], 0).unwrap();
    let clean = builder.add_pattern(&[StopIdx(0), StopIdx(3)], 1).unwrap();
    let combo_a = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 2).unwrap();
    let combo_b = builder.add_pattern(&[StopIdx(1), StopIdx(3)], 3).unwrap();
    builder
        .add_trip(dirty, vec![time(100), time(500)], 0, 0)
        .unwrap();
    builder
        .add_trip(clean, vec![time(100), time(900)], 1, 0)
        .unwrap();
    builder
        .add_trip(combo_a, vec![time(100), time(300)], 2, 0)
        .unwrap();
    builder
        .add_trip(combo_b, vec![time(400), time(700)], 3, 0)
        .unwrap();
    let timetable = builder.finish();
    let geometry = TripGeometry::from_trips(
        &timetable,
        vec![
            (TripIdx(0), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
            (TripIdx(1), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
            (TripIdx(2), vec![0.0, 500.0], DistanceProvenance::CrowFly),
            (TripIdx(3), vec![0.0, 500.0], DistanceProvenance::CrowFly),
        ],
    )
    .unwrap();
    (timetable, geometry, [100.0, 10.0, 100.0, 10.0])
}

#[test]
fn matches_the_oracle_with_a_vanishing_bucket() {
    let (timetable, geometry, factors) = frontier_fixture();
    let view = DayView::universal(&timetable);
    let footpaths = Transfers::empty(4);
    let request = request(StopIdx(0), StopIdx(3), 3);
    let journeys = route(
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
        None,
    );
    let points = pareto_oracle(
        &view,
        &timetable,
        &footpaths,
        &geometry,
        &factors,
        0,
        &request.access,
        &request.egress,
        3,
    );
    assert_eq!(
        triples(&journeys, &geometry, &factors),
        oracle_triples(&points)
    );
    assert_eq!(
        oracle_triples(&points),
        vec![(500, 100.0, 1), (700, 55.0, 2), (900, 10.0, 1)]
    );
}

/// Two frontier journeys — a fast dirty line and a slower cleaner one
/// — plus a middle line that strict Pareto drops as dominated but a
/// time slack keeps as a suboptimal alternative.
fn relaxed_fixture() -> (Timetable, TripGeometry, [f64; 3]) {
    let mut builder = TimetableBuilder::new(2);
    let fast = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let clean = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 1).unwrap();
    let middle = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 2).unwrap();
    builder
        .add_trip(fast, vec![time(100), time(500)], 0, 0)
        .unwrap();
    builder
        .add_trip(clean, vec![time(100), time(700)], 1, 0)
        .unwrap();
    builder
        .add_trip(middle, vec![time(100), time(600)], 2, 0)
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
    (timetable, geometry, [100.0, 60.0, 100.0])
}

#[test]
fn zero_slack_matches_the_strict_frontier() {
    let (timetable, geometry, factors) = relaxed_fixture();
    let view = DayView::universal(&timetable);
    let footpaths = Transfers::empty(2);
    let request = request(StopIdx(0), StopIdx(1), 3);
    let strict = route(
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
        None,
    );
    // Only the fast dirty line and the slow clean one; the middle line
    // is strictly dominated and dropped.
    assert_eq!(
        triples(&strict, &geometry, &factors),
        vec![(500, 100.0, 1), (700, 60.0, 1)]
    );
}

#[test]
fn a_time_slack_keeps_the_suboptimal_middle_line() {
    let (timetable, geometry, factors) = relaxed_fixture();
    let view = DayView::universal(&timetable);
    let footpaths = Transfers::empty(2);
    let request = request(StopIdx(0), StopIdx(1), 3);
    // 200 s exceeds the 100 s the fast line beats the middle line by,
    // so the middle line (600, dominated by 500) is retained.
    let relaxed = route(
        &view,
        &timetable,
        &footpaths,
        &geometry,
        &factors,
        &request,
        1e-6,
        200,
        None,
        &[],
        None,
    );
    assert_eq!(
        triples(&relaxed, &geometry, &factors),
        vec![(500, 100.0, 1), (600, 100.0, 1), (700, 60.0, 1)]
    );
}

#[test]
fn max_options_keeps_the_frontier_over_the_suboptimal() {
    let (timetable, geometry, factors) = relaxed_fixture();
    let view = DayView::universal(&timetable);
    let footpaths = Transfers::empty(2);
    let request = request(StopIdx(0), StopIdx(1), 3);
    // The relaxed set is three journeys; a cap of two keeps the strict
    // frontier and drops the suboptimal middle line.
    let capped = route(
        &view,
        &timetable,
        &footpaths,
        &geometry,
        &factors,
        &request,
        1e-6,
        200,
        Some(2),
        &[],
        None,
    );
    assert_eq!(
        triples(&capped, &geometry, &factors),
        vec![(500, 100.0, 1), (700, 60.0, 1)]
    );
}

/// One line, two trips one headway apart at the same factor: the later
/// trip is strictly dominated and never boarded by the strict line scan.
fn same_line_fixture() -> (Timetable, TripGeometry, [f64; 2]) {
    let mut builder = TimetableBuilder::new(2);
    let line = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    builder
        .add_trip(line, vec![time(100), time(500)], 0, 0)
        .unwrap();
    builder
        .add_trip(line, vec![time(200), time(600)], 1, 0)
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
    (timetable, geometry, [100.0, 100.0])
}

#[test]
fn a_time_slack_boards_the_next_same_line_trip() {
    let (timetable, geometry, factors) = same_line_fixture();
    let view = DayView::universal(&timetable);
    let footpaths = Transfers::empty(2);
    let request = request(StopIdx(0), StopIdx(1), 3);
    // Strict Pareto boards only the earliest trip; the later same-factor
    // trip arrives later at no lower emissions, so it is dropped.
    let strict = route(
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
        None,
    );
    assert_eq!(triples(&strict, &geometry, &factors), vec![(500, 100.0, 1)]);
    // A 200 s slack boards the next departure too — the "one trip later"
    // alternative the strict line scan never surfaces.
    let relaxed = route(
        &view,
        &timetable,
        &footpaths,
        &geometry,
        &factors,
        &request,
        1e-6,
        200,
        None,
        &[],
        None,
    );
    assert_eq!(
        triples(&relaxed, &geometry, &factors),
        vec![(500, 100.0, 1), (600, 100.0, 1)]
    );
}

/// One line, three trips: a dirty first, a much-later clean one, and a
/// middle-factor trip just after the clean one. The middle trip is within
/// slack of the clean frontier trip but far beyond the first departure —
/// only measuring against the nearest no-dirtier boarded trip admits it.
fn later_frontier_fixture() -> (Timetable, TripGeometry, [f64; 3]) {
    let mut builder = TimetableBuilder::new(2);
    let line = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    builder
        .add_trip(line, vec![time(100), time(500)], 0, 0)
        .unwrap();
    builder
        .add_trip(line, vec![time(900), time(1300)], 1, 0)
        .unwrap();
    builder
        .add_trip(line, vec![time(1000), time(1400)], 2, 0)
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
    (timetable, geometry, [100.0, 10.0, 50.0])
}

#[test]
fn a_time_slack_boards_the_next_trip_after_a_later_frontier() {
    let (timetable, geometry, factors) = later_frontier_fixture();
    let view = DayView::universal(&timetable);
    let footpaths = Transfers::empty(2);
    let request = request(StopIdx(0), StopIdx(1), 3);
    // Strict Pareto keeps the two frontier trips only.
    let strict = route(
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
        None,
    );
    assert_eq!(
        triples(&strict, &geometry, &factors),
        vec![(500, 100.0, 1), (1300, 10.0, 1)]
    );
    // A 200 s slack admits the middle-factor trip 100 s after the clean
    // frontier trip — far beyond the first departure, so measuring only
    // against the first would wrongly drop it.
    let relaxed = route(
        &view,
        &timetable,
        &footpaths,
        &geometry,
        &factors,
        &request,
        1e-6,
        200,
        None,
        &[],
        None,
    );
    assert_eq!(
        triples(&relaxed, &geometry, &factors),
        vec![(500, 100.0, 1), (1300, 10.0, 1), (1400, 50.0, 1)]
    );
}

/// Two route-disjoint corridors between the same stops: a fast one on
/// route 0 and a slower one on route 1, same emissions.
fn two_corridor_fixture() -> (Timetable, TripGeometry, [f64; 2]) {
    let mut builder = TimetableBuilder::new(2);
    let fast = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let slow = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 1).unwrap();
    builder
        .add_trip(fast, vec![time(100), time(500)], 0, 0)
        .unwrap();
    builder
        .add_trip(slow, vec![time(100), time(700)], 1, 0)
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
    (timetable, geometry, [50.0, 50.0])
}

/// Two corridors 0→1 (a fast route 0, a slow route 1, equal emissions),
/// then two onward trips 1→2: an early one only the fast corridor's
/// arrival can catch, and a late one either can. Used to check that a
/// penalty on the fast route does not prune its physically-earlier label
/// at the hub, which alone catches the early connection.
fn connection_fixture() -> (Timetable, TripGeometry, [f64; 4]) {
    let mut builder = TimetableBuilder::new(3);
    let fast = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let slow = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 1).unwrap();
    let early = builder.add_pattern(&[StopIdx(1), StopIdx(2)], 2).unwrap();
    let late = builder.add_pattern(&[StopIdx(1), StopIdx(2)], 3).unwrap();
    builder
        .add_trip(fast, vec![time(0), time(100)], 0, 0)
        .unwrap();
    builder
        .add_trip(slow, vec![time(0), time(300)], 1, 0)
        .unwrap();
    builder
        .add_trip(early, vec![time(150), time(200)], 2, 0)
        .unwrap();
    builder
        .add_trip(late, vec![time(350), time(600)], 3, 0)
        .unwrap();
    let timetable = builder.finish();
    let geometry = TripGeometry::from_trips(
        &timetable,
        vec![
            (TripIdx(0), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
            (TripIdx(1), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
            (TripIdx(2), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
            (TripIdx(3), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
        ],
    )
    .unwrap();
    (timetable, geometry, [100.0, 100.0, 100.0, 100.0])
}

#[test]
fn a_route_ban_forces_the_other_corridor() {
    let (timetable, geometry, factors) = two_corridor_fixture();
    let view = DayView::universal(&timetable);
    let footpaths = Transfers::empty(2);
    let request = request(StopIdx(0), StopIdx(1), 3);
    // No ban: the fast corridor wins; the slower same-emissions one is
    // dominated and dropped.
    let all = route(
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
        None,
    );
    assert_eq!(triples(&all, &geometry, &factors), vec![(500, 50.0, 1)]);
    // Ban route 0 (the `u32::MAX` sentinel): only the slower corridor on
    // route 1 remains.
    let banned = route(
        &view,
        &timetable,
        &footpaths,
        &geometry,
        &factors,
        &request,
        1e-6,
        0,
        None,
        &[u32::MAX, 0],
        None,
    );
    assert_eq!(triples(&banned, &geometry, &factors), vec![(700, 50.0, 1)]);
}

#[test]
fn a_soft_route_penalty_flips_the_winner_but_reports_the_true_arrival() {
    let (timetable, geometry, factors) = two_corridor_fixture();
    let view = DayView::universal(&timetable);
    let footpaths = Transfers::empty(2);
    let request = request(StopIdx(0), StopIdx(1), 3);
    // A penalty under the 200 s gap leaves the fast corridor (route 0) winning:
    // its effective arrival 500 + 100 still beats the other's 700, and the
    // journey reports its true 500 — the penalty lives only in the dominance.
    let light = route(
        &view,
        &timetable,
        &footpaths,
        &geometry,
        &factors,
        &request,
        1e-6,
        0,
        None,
        &[100, 0],
        None,
    );
    assert_eq!(triples(&light, &geometry, &factors), vec![(500, 50.0, 1)]);
    // A penalty over the gap flips the winner to route 1 (effective 750 > 700),
    // and the surviving journey still reports its true 700, never 750.
    let heavy = route(
        &view,
        &timetable,
        &footpaths,
        &geometry,
        &factors,
        &request,
        1e-6,
        0,
        None,
        &[250, 0],
        None,
    );
    assert_eq!(triples(&heavy, &geometry, &factors), vec![(700, 50.0, 1)]);
}

#[test]
fn a_penalized_early_label_still_catches_its_connection() {
    let (timetable, geometry, factors) = connection_fixture();
    let view = DayView::universal(&timetable);
    let footpaths = Transfers::empty(3);
    let request = request(StopIdx(0), StopIdx(2), 3);
    // Route 0 reaches the hub at 100, route 1 at 300; a 300 s penalty on route 0
    // pushes its effective arrival (400) past route 1's (300). Dominating on the
    // effective arrival alone would prune the route-0 label at the hub, losing
    // the only corridor that catches the early 1→2 trip (departs 150). Because
    // the stop bag also keeps the physically-earlier label, the search still
    // reaches the destination at the true 200, not route 1's 600.
    let journeys = route(
        &view,
        &timetable,
        &footpaths,
        &geometry,
        &factors,
        &request,
        1e-6,
        0,
        None,
        &[300, 0, 0, 0],
        None,
    );
    let arrivals: Vec<u32> = journeys.iter().map(|journey| journey.arrival).collect();
    assert!(
        arrivals.contains(&200),
        "the penalized early corridor must still reach 200, got {arrivals:?}"
    );
}

#[test]
fn a_wide_bucket_collapses_to_the_fastest_journey() {
    let (timetable, geometry, factors) = frontier_fixture();
    let view = DayView::universal(&timetable);
    let footpaths = Transfers::empty(4);
    let request = request(StopIdx(0), StopIdx(3), 3);
    let journeys = route(
        &view,
        &timetable,
        &footpaths,
        &geometry,
        &factors,
        &request,
        1e9,
        0,
        None,
        &[],
        None,
    );
    assert_eq!(
        triples(&journeys, &geometry, &factors),
        vec![(500, 100.0, 1)]
    );
}

#[test]
fn loop_backs_walk_nowhere() {
    // The oracle's regression shape: ride out and back, then walk
    // to the destination — cleaner and earlier than the direct
    // line, but suppressed by the access label's dominance.
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
    let factors = [10.0, 10.0, 100.0];
    let footpaths = Transfers::from_edges(
        3,
        &[
            (StopIdx(0), StopIdx(2), 30, 30.0),
            (StopIdx(2), StopIdx(0), 30, 30.0),
        ],
    )
    .unwrap();
    let view = DayView::universal(&timetable);
    let request = request(StopIdx(0), StopIdx(2), 3);
    let journeys = route(
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
        None,
    );
    assert_eq!(
        triples(&journeys, &geometry, &factors),
        vec![(200, 80.0, 1)]
    );
}

#[test]
fn waits_for_the_cleaner_trip_on_a_mixed_factor_line() {
    // One line, a dirty early trip and a clean later one: the true
    // frontier holds both, so boarding must look past the earliest
    // boardable trip when a later factor strictly improves.
    let mut builder = TimetableBuilder::new(2);
    let line = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    builder
        .add_trip(line, vec![time(100), time(200)], 0, 0)
        .unwrap();
    builder
        .add_trip(line, vec![time(300), time(400)], 1, 0)
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
    let factors = [100.0, 10.0];
    let view = DayView::universal(&timetable);
    let footpaths = Transfers::empty(2);
    let request = request(StopIdx(0), StopIdx(1), 1);
    let journeys = route(
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
        None,
    );
    let points = pareto_oracle(
        &view,
        &timetable,
        &footpaths,
        &geometry,
        &factors,
        0,
        &request.access,
        &request.egress,
        1,
    );
    assert_eq!(
        triples(&journeys, &geometry, &factors),
        oracle_triples(&points)
    );
    assert_eq!(
        oracle_triples(&points),
        vec![(200, 100.0, 1), (400, 10.0, 1)]
    );
}

#[test]
fn transfers_over_a_footpath_match_the_oracle() {
    // Ride, walk a footpath, ride again — the walked hop must
    // carry its grams unchanged and reconstruct as a transfer leg.
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
    let factors = [10.0, 20.0];
    let footpaths = Transfers::from_edges(
        4,
        &[
            (StopIdx(1), StopIdx(2), 50, 50.0),
            (StopIdx(2), StopIdx(1), 50, 50.0),
        ],
    )
    .unwrap();
    let view = DayView::universal(&timetable);
    let request = request(StopIdx(0), StopIdx(3), 3);
    let journeys = route(
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
        None,
    );
    let points = pareto_oracle(
        &view,
        &timetable,
        &footpaths,
        &geometry,
        &factors,
        0,
        &request.access,
        &request.egress,
        3,
    );
    assert_eq!(
        triples(&journeys, &geometry, &factors),
        oracle_triples(&points)
    );
    assert_eq!(oracle_triples(&points), vec![(300, 50.0, 2)]);
    let legs: Vec<&str> = journeys[0]
        .legs
        .iter()
        .map(|leg| match leg {
            Leg::Access { .. } => "access",
            Leg::Transit { .. } => "transit",
            Leg::Transfer { .. } => "transfer",
            Leg::Egress { .. } => "egress",
        })
        .collect();
    assert_eq!(
        legs,
        vec!["access", "transit", "transfer", "transit", "egress"]
    );
}

#[test]
fn skips_unresolved_factors() {
    let mut builder = TimetableBuilder::new(2);
    let line = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    builder
        .add_trip(line, vec![time(0), time(100)], 0, 0)
        .unwrap();
    let timetable = builder.finish();
    let geometry = TripGeometry::from_trips(
        &timetable,
        vec![(TripIdx(0), vec![0.0, 1000.0], DistanceProvenance::CrowFly)],
    )
    .unwrap();
    let view = DayView::universal(&timetable);
    let request = request(StopIdx(0), StopIdx(1), 1);
    let journeys = route(
        &view,
        &timetable,
        &Transfers::empty(2),
        &geometry,
        &[f64::NAN],
        &request,
        1e-6,
        0,
        None,
        &[],
        None,
    );
    assert!(journeys.is_empty());
}

#[test]
fn the_emissions_matrix_sees_past_the_time_candidates() {
    use crate::raptor::{Objective, Raptor};

    let (timetable, geometry, factors) = frontier_fixture();
    let view = DayView::universal(&timetable);
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
    let destinations = [StopIdx(0), StopIdx(3)];
    let rows = least_emissions_matrix(
        &view,
        &timetable,
        &footpaths,
        &inputs,
        &requests,
        &destinations,
        &vec![Vec::new(); timetable.stop_count() as usize],
        &vec![Vec::new(); requests.len()],
        false,
        600,
        None,
        1e-6,
    );
    // The cleanest journey is the slow clean line — invisible to
    // the interim objective, whose per-round RAPTOR arrivals only
    // ever hold the faster dirty alternatives.
    let cell = rows[0].iter().find(|row| row.to == 3).unwrap();
    assert_eq!((cell.seconds, cell.rides), (800, 1));
    assert!((cell.emission_grams - 10.0).abs() < 1e-9);
    assert_eq!(cell.transit_meters, 1000.0);
    let interim = Raptor.least_cost_matrix(
        &timetable,
        &footpaths,
        &inputs,
        &requests,
        &destinations,
        600,
        None,
        Objective::Emissions,
    );
    let interim_cell = interim[0].iter().find(|row| row.to == 3).unwrap();
    assert!((interim_cell.emission_grams - 100.0).abs() < 1e-9);
    assert!(cell.emission_grams < interim_cell.emission_grams);
    // The origin's own cell is the zero-ride floor.
    let floor = rows[0].iter().find(|row| row.to == 0).unwrap();
    assert_eq!((floor.seconds, floor.rides), (0, 0));
    assert_eq!(floor.emission_grams, 0.0);
}

#[test]
fn a_budget_caps_the_matrix_travel_time() {
    let (timetable, geometry, factors) = frontier_fixture();
    let view = DayView::universal(&timetable);
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
    let cell = |budget: Option<u32>| {
        let rows = least_emissions_matrix(
            &view,
            &timetable,
            &footpaths,
            &inputs,
            &requests,
            &[StopIdx(3)],
            &vec![Vec::new(); timetable.stop_count() as usize],
            &vec![Vec::new(); requests.len()],
            false,
            600,
            budget,
            1e-6,
        );
        rows[0].iter().find(|row| row.to == 3).cloned()
    };
    // Tightening budgets walk the frontier: the clean line, the
    // transfer combination, the dirty direct, then nothing.
    let unbudgeted = cell(None).unwrap();
    assert_eq!((unbudgeted.seconds, unbudgeted.rides), (800, 1));
    let combo = cell(Some(600)).unwrap();
    assert_eq!((combo.seconds, combo.rides), (600, 2));
    assert!((combo.emission_grams - 55.0).abs() < 1e-9);
    let dirty = cell(Some(400)).unwrap();
    assert_eq!((dirty.seconds, dirty.rides), (400, 1));
    assert!((dirty.emission_grams - 100.0).abs() < 1e-9);
    assert!(cell(Some(100)).is_none());
}

#[test]
fn matrix_rows_carry_their_transfer_walks() {
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
    let factors = [10.0, 20.0];
    let footpaths = Transfers::from_edges(
        4,
        &[
            (StopIdx(1), StopIdx(2), 50, 50.0),
            (StopIdx(2), StopIdx(1), 50, 50.0),
        ],
    )
    .unwrap();
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
    let rows = least_emissions_matrix(
        &view,
        &timetable,
        &footpaths,
        &inputs,
        &requests,
        &[StopIdx(3)],
        &vec![Vec::new(); timetable.stop_count() as usize],
        &vec![Vec::new(); requests.len()],
        false,
        100,
        None,
        1e-6,
    );
    let cell = rows[0].iter().find(|row| row.to == 3).unwrap();
    assert_eq!((cell.seconds, cell.rides), (300, 2));
    assert_eq!(cell.transit_meters, 3000.0);
    assert_eq!(cell.walk_meters, 50.0);
    assert!((cell.emission_grams - 50.0).abs() < 1e-9);
    assert!(cell.fare.is_nan());
}

#[test]
fn profiles_the_departure_window() {
    // Two departures of one line inside the window: the profile
    // keeps both journeys, each at the latest departure catching
    // it.
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
    let factors = [50.0, 50.0];
    let view = DayView::universal(&timetable);
    let footpaths = Transfers::empty(2);
    let request = Request {
        departure: 50,
        access: vec![(StopIdx(0), 0)],
        egress: vec![(StopIdx(1), 0)],
        active_services: vec![true],
        active_services_previous: vec![false],
        max_transfers: 1,
    };
    let journeys = route_range(
        &view,
        &timetable,
        &footpaths,
        &geometry,
        &factors,
        &request,
        200,
        1e-6,
        0,
        None,
        &[],
        None,
    );
    let profile: Vec<(u32, u32)> = journeys
        .iter()
        .map(|journey| (journey.departure, journey.arrival))
        .collect();
    assert_eq!(profile, vec![(100, 300), (200, 400)]);
}

#[test]
fn frontier_matrix_matches_the_one_pair_profile_per_cell() {
    let (timetable, geometry, factors) = frontier_fixture();
    let view = DayView::universal(&timetable);
    let footpaths = Transfers::empty(4);
    let destinations = [StopIdx(3), StopIdx(1), StopIdx(3)];
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
    let cells = frontier_matrix(
        &view,
        &timetable,
        &footpaths,
        &geometry,
        &factors,
        &requests,
        &destinations,
        &[],
        false,
        destinations.len(),
        1000,
        1e-6,
        None,
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
            let journeys = route_range(
                &view,
                &timetable,
                &footpaths,
                &geometry,
                &factors,
                &one_pair,
                1000,
                1e-6,
                0,
                None,
                &[],
                None,
            );
            assert_eq!(keys(cell), keys(&journeys));
        }
    }
    // The known frontier of the fixture: from stop 0 to stop 3 the
    // fast dirty direct, the transfer combination, and the clean slow
    // direct all survive; stop 2 reaches neither destination; the
    // repeated destination stop gets the same cell as its first slot.
    assert_eq!(cells[0][0].len(), 3);
    assert!(cells[1][0].is_empty() && cells[1][1].is_empty());
    assert_eq!(keys(&cells[0][2]), keys(&cells[0][0]));
}

#[test]
fn target_pruning_keeps_the_same_bucket_refinement() {
    // A dirty direct ride reaches the destination in round one; a
    // cleaner two-leg journey arrives at the *same* effective time and
    // bucket one round later, through a different egress stop. Target
    // pruning must keep its labels alive — the destination bag accepts
    // it as the same-class refinement — so the reported journey is the
    // clean one, exactly as without pruning.
    let mut builder = TimetableBuilder::new(4);
    let dirty = builder.add_pattern(&[StopIdx(0), StopIdx(3)], 0).unwrap();
    let leg_a = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 1).unwrap();
    let leg_b = builder.add_pattern(&[StopIdx(1), StopIdx(2)], 2).unwrap();
    builder
        .add_trip(dirty, vec![time(100), time(500)], 0, 0)
        .unwrap();
    builder
        .add_trip(leg_a, vec![time(100), time(200)], 1, 0)
        .unwrap();
    builder
        .add_trip(leg_b, vec![time(300), time(500)], 2, 0)
        .unwrap();
    let timetable = builder.finish();
    let geometry = TripGeometry::from_trips(
        &timetable,
        vec![
            (TripIdx(0), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
            (TripIdx(1), vec![0.0, 100.0], DistanceProvenance::CrowFly),
            (TripIdx(2), vec![0.0, 900.0], DistanceProvenance::CrowFly),
        ],
    )
    .unwrap();
    let factors = [100.0, 10.0, 10.0];
    let view = DayView::universal(&timetable);
    let footpaths = Transfers::empty(4);
    let request = Request {
        departure: 0,
        access: vec![(StopIdx(0), 0)],
        egress: vec![(StopIdx(3), 0), (StopIdx(2), 0)],
        active_services: Vec::new(),
        active_services_previous: Vec::new(),
        max_transfers: 3,
    };
    // One bucket holds both journeys, so they compare equal on
    // emissions during the search and the exact grams decide.
    let journeys = route(
        &view,
        &timetable,
        &footpaths,
        &geometry,
        &factors,
        &request,
        1000.0,
        0,
        None,
        &[],
        None,
    );
    assert_eq!(journeys.len(), 1);
    assert_eq!(journeys[0].arrival, 500);
    assert_eq!(journeys[0].rides(), 2);
    assert_eq!(grams_of(&journeys[0], &geometry, &factors), 10.0);
}

#[test]
fn max_slower_trims_the_slow_tail() {
    let (timetable, geometry, factors) = relaxed_fixture();
    let view = DayView::universal(&timetable);
    let footpaths = Transfers::empty(2);
    let request = request(StopIdx(0), StopIdx(1), 3);
    let banded = |band: u32| {
        route(
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
            Some(band),
        )
    };
    // The clean line arrives 200 s after the fast one: a 100 s band
    // drops it, a 300 s band keeps the full frontier.
    assert_eq!(
        triples(&banded(100), &geometry, &factors),
        vec![(500, 100.0, 1)]
    );
    assert_eq!(
        triples(&banded(300), &geometry, &factors),
        vec![(500, 100.0, 1), (700, 60.0, 1)]
    );
}

#[test]
fn max_slower_keeps_the_fastest_past_a_strayed_prefix() {
    // Stop 1's plain bound (150) comes from a two-ride chain that
    // exhausts the transfer budget, while the destination's only
    // journey rides the slow direct line to stop 1 (arriving 400 —
    // beyond bound + band) and continues. The cutoff floor at the
    // destination bound must keep that prefix alive.
    let mut builder = TimetableBuilder::new(5);
    let fast_a = builder.add_pattern(&[StopIdx(0), StopIdx(4)], 0).unwrap();
    let fast_b = builder.add_pattern(&[StopIdx(4), StopIdx(1)], 1).unwrap();
    let slow = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 2).unwrap();
    let onward = builder.add_pattern(&[StopIdx(1), StopIdx(3)], 3).unwrap();
    builder
        .add_trip(fast_a, vec![time(100), time(120)], 0, 0)
        .unwrap();
    builder
        .add_trip(fast_b, vec![time(130), time(150)], 1, 0)
        .unwrap();
    builder
        .add_trip(slow, vec![time(100), time(400)], 2, 0)
        .unwrap();
    builder
        .add_trip(onward, vec![time(450), time(600)], 3, 0)
        .unwrap();
    let timetable = builder.finish();
    let geometry = TripGeometry::from_trips(
        &timetable,
        vec![
            (TripIdx(0), vec![0.0, 100.0], DistanceProvenance::CrowFly),
            (TripIdx(1), vec![0.0, 100.0], DistanceProvenance::CrowFly),
            (TripIdx(2), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
            (TripIdx(3), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
        ],
    )
    .unwrap();
    let factors = [10.0, 10.0, 10.0, 10.0];
    let view = DayView::universal(&timetable);
    let footpaths = Transfers::empty(5);
    let request = request(StopIdx(0), StopIdx(3), 1);
    let journeys = route(
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
        Some(100),
    );
    assert_eq!(journeys.len(), 1);
    assert_eq!(journeys[0].arrival, 600);
    assert_eq!(journeys[0].rides(), 2);
}

#[test]
fn max_slower_anchors_at_the_resolved_bound() {
    // The truly fastest line's factor is unresolved, so the
    // multicriteria search cannot ride it; the band must anchor at the
    // fastest *resolved* arrival or the only reportable journey dies.
    let mut builder = TimetableBuilder::new(2);
    let unresolved = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let resolved = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 1).unwrap();
    builder
        .add_trip(unresolved, vec![time(100), time(500)], 0, 0)
        .unwrap();
    builder
        .add_trip(resolved, vec![time(100), time(800)], 1, 0)
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
    let factors = [f64::NAN, 10.0];
    let view = DayView::universal(&timetable);
    let footpaths = Transfers::empty(2);
    let request = request(StopIdx(0), StopIdx(1), 3);
    let journeys = route(
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
        Some(100),
    );
    assert_eq!(journeys.len(), 1);
    assert_eq!(journeys[0].arrival, 800);
}

#[test]
fn probed_insert_slack_matches_the_stable_insert_slack() {
    // The R0 probe variant must make identical decisions and keep the
    // identical entry order across slack bands, penalties, exact-class
    // refinements, and ride ranks.
    let mut state = 0x9e3779b97f4a7c15u64;
    let mut next = move |bound: u64| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) % bound
    };
    for &slack in &[0u32, 120] {
        let mut stable = Bag::default();
        let mut probed = Bag::default();
        for _ in 0..20_000 {
            let arrival = 100 + next(90) as u32;
            let penalty = [0, 0, 0, 60][next(4) as usize];
            let key = next(6) as i64;
            let grams = key as f64 * 25.0 + next(25) as f64;
            let rides = next(4) as u8;
            let expected = stable.insert_slack(arrival, penalty, grams, key, rides, slack);
            let mut probes = InsertProbes::default();
            let admitted =
                probed.insert_slack_probed(arrival, penalty, grams, key, rides, slack, &mut probes);
            assert_eq!(admitted, expected);
            assert!(probes.examined as usize <= probes.length as usize + 1);
            assert_eq!(stable.snapshot(), probed.snapshot());
        }
    }
}

#[test]
fn probed_insert_slack_matches_on_unreachable_orders() {
    // Deliberately unreachable entry orders (slack bags are not
    // antichains): decisions and order must still agree.
    let entries = vec![
        (100, 0, 5, 50.0, 1),
        (90, 60, 4, 40.0, 1),
        (100, 0, 5, 45.0, 1),
        (95, 0, 4, 40.0, 2),
    ];
    for &slack in &[0u32, 30, 120] {
        let mut stable = Bag::from_entries(entries.clone());
        let mut probed = Bag::from_entries(entries.clone());
        for candidate in [
            (100u32, 0u32, 5i64, 44.0f64, 1u8),
            (96, 0, 4, 39.0, 2),
            (130, 0, 6, 60.0, 3),
            (90, 60, 4, 40.0, 1),
        ] {
            let (arrival, penalty, key, grams, rides) = candidate;
            let expected = stable.insert_slack(arrival, penalty, grams, key, rides, slack);
            let mut probes = InsertProbes::default();
            let admitted =
                probed.insert_slack_probed(arrival, penalty, grams, key, rides, slack, &mut probes);
            assert_eq!(admitted, expected);
            assert_eq!(stable.snapshot(), probed.snapshot());
        }
    }
}

#[test]
fn attribution_counters_hold_their_identities() {
    // Drive a small search with the diagnostics armed and check the
    // R0 identities on the collected counters.
    let mut builder = TimetableBuilder::new(6);
    let feeder = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let onward = builder
        .add_pattern(&[StopIdx(2), StopIdx(3), StopIdx(5)], 1)
        .unwrap();
    builder
        .add_trip(feeder, vec![time(0), time(100)], 0, 0)
        .unwrap();
    builder
        .add_trip(onward, vec![time(300), time(400), time(500)], 1, 0)
        .unwrap();
    let timetable = builder.finish();
    let geometry = TripGeometry::from_trips(
        &timetable,
        vec![
            (TripIdx(0), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
            (
                TripIdx(1),
                vec![0.0, 1000.0, 2000.0],
                DistanceProvenance::CrowFly,
            ),
        ],
    )
    .unwrap();
    let footpaths = Transfers::from_edges(
        6,
        &[
            (StopIdx(1), StopIdx(2), 60, 60.0),
            (StopIdx(1), StopIdx(4), 60, 60.0),
            (StopIdx(3), StopIdx(4), 60, 60.0),
        ],
    )
    .unwrap();
    let factors = [10.0, 20.0];
    let view = DayView::universal(&timetable);
    let mut search = Search::start(
        &view,
        &timetable,
        &footpaths,
        &geometry,
        &factors,
        25.0,
        0,
        &[],
    );
    search.ops = true;
    search.arm_diagnostics();
    let request = request(StopIdx(0), StopIdx(5), 2);
    for departure in [100u32, 0] {
        search.pass(&request, departure, &mut None, &mut None);
    }
    let stats = &search.stats;
    assert!(stats.footpath_bag_calls > 0);
    assert_eq!(
        stats.footpath_bag_calls,
        stats.footpath_bag_rejections + stats.footpath_target_admissions
    );
    assert_eq!(
        stats.footpath_label_edge_relaxations,
        stats.footpath_cutoff_pruned + stats.footpath_target_pruned + stats.footpath_bag_calls
    );
    assert_eq!(stats.rode_labels, stats.rode_represented + stats.rode_stale);
    assert_eq!(
        stats.batch_points_offered,
        stats.batch_points_rejected + stats.batch_points_evicted + stats.batch_points_live
    );
    assert!(stats.route_bag_calls >= stats.route_bag_admissions);
    assert_eq!(
        stats.footpath_reject_depth_histogram.iter().sum::<u64>(),
        stats.footpath_bag_rejections
    );
    assert_eq!(
        stats.footpath_bag_length_histogram.iter().sum::<u64>(),
        stats.footpath_bag_calls
    );
    // The strict shadow armed (slack 0, no penalties) and saw offers.
    assert!(stats.batch_points_offered > 0);
    assert!(stats.batch_edge_records_predicted <= stats.footpath_edge_records_loaded);
}
