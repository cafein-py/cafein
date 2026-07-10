use super::*;
use crate::geometry::DistanceProvenance;
use crate::timetable::{StopTime, TimetableBuilder, TripIdx};

fn time(at: u32) -> StopTime {
    StopTime {
        arrival: at,
        departure: at,
    }
}

#[test]
fn enumerates_the_true_frontier() {
    // A fast dirty direct line (arr 500, 100 g), a slow clean
    // direct line (arr 900, 10 g), and a middle option: dirty then
    // clean over a transfer (arr 700, 55 g) — three true Pareto
    // points, the last invisible to a time-only candidate set that
    // already holds a faster 2-ride journey... here all are
    // distinct; dominance is checked by hand.
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
    // g/pkm: dirty 100, clean 10, combo legs 100 then 10 over
    // half the distance each → 50 + 5 = 55 g.
    let factors = [100.0, 10.0, 100.0, 10.0];
    let view = DayView::universal(&timetable);
    let footpaths = Transfers::empty(4);
    let points = pareto_oracle(
        &view,
        &timetable,
        &footpaths,
        &geometry,
        &factors,
        0,
        &[(StopIdx(0), 0)],
        &[(StopIdx(3), 0)],
        3,
    );
    let triples: Vec<(u32, f64, u32)> = points
        .iter()
        .map(|point| (point.arrival, point.grams, point.rides))
        .collect();
    assert_eq!(
        triples,
        vec![(500, 100.0, 1), (700, 55.0, 2), (900, 10.0, 1)]
    );
}

#[test]
fn loop_backs_walk_nowhere() {
    // Ride out and back to the origin, then walk to the
    // destination: physically feasible and even Pareto-looking
    // (arrives before the direct line), but the routers' label
    // semantics suppress it — the origin's access label dominates
    // the ride-back — and the oracle must match them.
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
    // The loop is cleaner than the direct ride, so a naive
    // enumerator would keep (130, 8 g) beside (200, 80 g).
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
    let points = pareto_oracle(
        &view,
        &timetable,
        &footpaths,
        &geometry,
        &factors,
        0,
        &[(StopIdx(0), 0)],
        &[(StopIdx(2), 0)],
        3,
    );
    let triples: Vec<(u32, f64, u32)> = points
        .iter()
        .map(|point| (point.arrival, point.grams, point.rides))
        .collect();
    assert_eq!(triples, vec![(200, 80.0, 1)]);
}

#[test]
fn unresolved_factors_never_enter_the_frontier() {
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
    let points = pareto_oracle(
        &view,
        &timetable,
        &Transfers::empty(2),
        &geometry,
        &[f64::NAN],
        0,
        &[(StopIdx(0), 0)],
        &[(StopIdx(1), 0)],
        1,
    );
    assert!(points.is_empty());
}
