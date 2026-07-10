use super::*;
use crate::geometry::DistanceProvenance;
use crate::timetable::{StopTime, TimetableBuilder, TripIdx};

fn time(at: u32) -> StopTime {
    StopTime {
        arrival: at,
        departure: at,
    }
}

fn symmetric(edges: &[(u32, u32, u32, f64)]) -> Transfers {
    let mut all = Vec::new();
    for &(a, b, dur, m) in edges {
        all.push((StopIdx(a), StopIdx(b), dur, m));
        all.push((StopIdx(b), StopIdx(a), dur, m));
    }
    Transfers::from_edges(5, &all).unwrap()
}

fn has(shortcuts: &[Shortcut], origin: u32, dest: u32) -> bool {
    shortcuts
        .iter()
        .any(|s| s.origin == StopIdx(origin) && s.destination == StopIdx(dest))
}

/// Line A rides 0→1 (board in the source station); from its alight (1) two
/// intermediate transfers lead to trip 2: 1→2 boards a **fast, dirty** trip
/// to 4 (arr 250, factor 100), 1→3 boards a **slow, clean** one (arr 600,
/// factor 5). The two two-trip journeys to 4 are Pareto-incomparable, so
/// McULTRA must keep **both** shortcuts; bicriteria ULTRA (arrival only)
/// keeps only the fast one.
fn cleaner_but_later() -> (Timetable, TripGeometry, [f64; 3], Transfers) {
    let mut builder = TimetableBuilder::new(5);
    let a = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let fast = builder.add_pattern(&[StopIdx(2), StopIdx(4)], 1).unwrap();
    let slow = builder.add_pattern(&[StopIdx(3), StopIdx(4)], 2).unwrap();
    builder.add_trip(a, vec![time(0), time(100)], 0, 0).unwrap();
    builder
        .add_trip(fast, vec![time(200), time(250)], 1, 0)
        .unwrap();
    builder
        .add_trip(slow, vec![time(200), time(600)], 2, 0)
        .unwrap();
    let timetable = builder.finish();
    let geometry = TripGeometry::from_trips(
        &timetable,
        vec![
            (TripIdx(0), vec![0.0, 500.0], DistanceProvenance::CrowFly),
            (TripIdx(1), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
            (TripIdx(2), vec![0.0, 1000.0], DistanceProvenance::CrowFly),
        ],
    )
    .unwrap();
    let transfers = symmetric(&[(1, 2, 50, 50.0), (1, 3, 50, 50.0)]);
    (timetable, geometry, [10.0, 100.0, 5.0], transfers)
}

#[test]
fn cleaner_but_later_transfer_survives() {
    let (timetable, geometry, factors, transfers) = cleaner_but_later();
    let view = DayView::universal(&timetable);
    let mc = compute_mcultra_shortcuts(
        &view,
        &timetable,
        &transfers,
        &geometry,
        &factors,
        0,
        NEVER - 1,
    );
    assert!(
        has(&mc, 1, 2),
        "McULTRA keeps the fast-dirty transfer: {mc:?}"
    );
    assert!(
        has(&mc, 1, 3),
        "McULTRA keeps the slow-clean transfer bicriteria drops: {mc:?}"
    );
    // Minimality: exactly the two Pareto transfers, no superfluous shortcuts.
    assert_eq!(mc.len(), 2, "{mc:?}");

    let bicriteria = crate::ultra::compute_shortcuts(&view, &timetable, &transfers, 0, NEVER - 1);
    assert!(has(&bicriteria, 1, 2), "bicriteria keeps the fast transfer");
    assert!(
        !has(&bicriteria, 1, 3),
        "bicriteria (arrival only) drops the slow-clean transfer: {bicriteria:?}"
    );
}

#[test]
fn emits_the_single_needed_transfer() {
    // Only one Pareto two-trip journey: McULTRA and bicriteria agree.
    let mut builder = TimetableBuilder::new(5);
    let a = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let b = builder.add_pattern(&[StopIdx(2), StopIdx(3)], 1).unwrap();
    builder.add_trip(a, vec![time(0), time(100)], 0, 0).unwrap();
    builder
        .add_trip(b, vec![time(200), time(300)], 1, 0)
        .unwrap();
    let timetable = builder.finish();
    let view = DayView::universal(&timetable);
    let geometry = TripGeometry::from_trips(
        &timetable,
        vec![
            (TripIdx(0), vec![0.0, 500.0], DistanceProvenance::CrowFly),
            (TripIdx(1), vec![0.0, 500.0], DistanceProvenance::CrowFly),
        ],
    )
    .unwrap();
    let transfers = symmetric(&[(1, 2, 50, 50.0)]);
    let mc = compute_mcultra_shortcuts(
        &view,
        &timetable,
        &transfers,
        &geometry,
        &[10.0, 10.0],
        0,
        NEVER - 1,
    );
    assert_eq!(mc.len(), 1, "{mc:?}");
    assert_eq!(mc[0].origin, StopIdx(1));
    assert_eq!(mc[0].destination, StopIdx(2));
    assert_eq!(mc[0].seconds, 50);
}

#[test]
fn a_direct_walk_witnesses_the_transfer_away() {
    // Same trip-1/trip-2 as above, but the source can walk straight to the
    // trip-2 board stop (0→2, 90 s) faster and cleaner (grams 0) than
    // trip 1 + the 1→2 transfer (arrives 150). The one-trip candidate at 2
    // is dominated, so no shortcut is needed.
    let mut builder = TimetableBuilder::new(5);
    let a = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let b = builder.add_pattern(&[StopIdx(2), StopIdx(3)], 1).unwrap();
    builder.add_trip(a, vec![time(0), time(100)], 0, 0).unwrap();
    builder
        .add_trip(b, vec![time(200), time(300)], 1, 0)
        .unwrap();
    let timetable = builder.finish();
    let view = DayView::universal(&timetable);
    let geometry = TripGeometry::from_trips(
        &timetable,
        vec![
            (TripIdx(0), vec![0.0, 500.0], DistanceProvenance::CrowFly),
            (TripIdx(1), vec![0.0, 500.0], DistanceProvenance::CrowFly),
        ],
    )
    .unwrap();
    // 1→2 is the intermediate transfer; 0→2 (90 s) is the witnessing walk.
    let transfers = symmetric(&[(1, 2, 50, 50.0), (0, 2, 90, 90.0)]);
    let mc = compute_mcultra_shortcuts(
        &view,
        &timetable,
        &transfers,
        &geometry,
        &[10.0, 10.0],
        0,
        NEVER - 1,
    );
    assert!(
        !has(&mc, 1, 2),
        "a faster, cleaner direct walk witnesses the 1→2 transfer away: {mc:?}"
    );
    // Minimality: the witnessed-away transfer leaves no shortcut at all.
    assert!(mc.is_empty(), "{mc:?}");
}

#[test]
fn a_two_trip_witness_does_not_prune_a_one_trip_shortcut() {
    // The transfers axis. A one-trip context reaches board stop 2 via
    // A (0→1) + the 1→2 transfer (arr 150), and boards B (2→3) — the (1,2)
    // shortcut. A *two-trip* journey C1 (0→4) + 4→5 + C2 (5→2) also reaches 2,
    // faster and cleaner (arr 60). Because one-ride and two-ride labels live
    // in separate bags, that two-ride journey must NOT prune the one-ride
    // context, so (1,2) still emits (it would vanish if the bags were merged).
    let mut builder = TimetableBuilder::new(6);
    let a = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let b = builder.add_pattern(&[StopIdx(2), StopIdx(3)], 1).unwrap();
    let c1 = builder.add_pattern(&[StopIdx(0), StopIdx(4)], 2).unwrap();
    let c2 = builder.add_pattern(&[StopIdx(5), StopIdx(2)], 3).unwrap();
    builder.add_trip(a, vec![time(0), time(100)], 0, 0).unwrap();
    builder
        .add_trip(b, vec![time(200), time(300)], 1, 0)
        .unwrap();
    builder.add_trip(c1, vec![time(0), time(30)], 2, 0).unwrap();
    builder
        .add_trip(c2, vec![time(50), time(60)], 3, 0)
        .unwrap();
    let timetable = builder.finish();
    let view = DayView::universal(&timetable);
    let geometry = TripGeometry::from_trips(
        &timetable,
        vec![
            (TripIdx(0), vec![0.0, 500.0], DistanceProvenance::CrowFly),
            (TripIdx(1), vec![0.0, 500.0], DistanceProvenance::CrowFly),
            (TripIdx(2), vec![0.0, 100.0], DistanceProvenance::CrowFly),
            (TripIdx(3), vec![0.0, 100.0], DistanceProvenance::CrowFly),
        ],
    )
    .unwrap();
    // 1→2 (the shortcut under test) and 4→5 (feeds the two-trip C journey).
    let transfers = Transfers::from_edges(
        6,
        &[
            (StopIdx(1), StopIdx(2), 50, 50.0),
            (StopIdx(2), StopIdx(1), 50, 50.0),
            (StopIdx(4), StopIdx(5), 10, 10.0),
            (StopIdx(5), StopIdx(4), 10, 10.0),
        ],
    )
    .unwrap();
    let mc = compute_mcultra_shortcuts(
        &view,
        &timetable,
        &transfers,
        &geometry,
        &[10.0, 10.0, 1.0, 1.0],
        0,
        NEVER - 1,
    );
    assert!(
        has(&mc, 1, 2),
        "the fast, clean two-trip journey to stop 2 must not prune the \
             one-trip (1,2) shortcut context (separate rides bags): {mc:?}"
    );
}
