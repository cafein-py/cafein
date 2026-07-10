use super::*;
use crate::timetable::TimetableBuilder;

fn time(at: u32) -> crate::timetable::StopTime {
    crate::timetable::StopTime {
        arrival: at,
        departure: at,
    }
}

/// Line A rides 0→1→2 at 100/200/300; line B rides 2→3 at 250/400.
fn two_lines() -> Timetable {
    let mut builder = TimetableBuilder::new(4);
    let a = builder
        .add_pattern(&[StopIdx(0), StopIdx(1), StopIdx(2)], 0)
        .unwrap();
    let b = builder.add_pattern(&[StopIdx(2), StopIdx(3)], 1).unwrap();
    builder
        .add_trip(a, vec![time(100), time(200), time(300)], 0, 0)
        .unwrap();
    builder
        .add_trip(b, vec![time(250), time(400)], 1, 0)
        .unwrap();
    builder.finish()
}

#[test]
fn stations_group_zero_time_transfers() {
    // 0↔1 are coincident (0 s); 2 stands alone.
    let transfers = Transfers::from_edges(
        3,
        &[
            (StopIdx(0), StopIdx(1), 0, 0.0),
            (StopIdx(1), StopIdx(0), 0, 0.0),
        ],
    )
    .unwrap();
    let reps = stations(&transfers, 3);
    assert_eq!(reps[0], StopIdx(0));
    assert_eq!(reps[1], StopIdx(0), "stop 1 joins stop 0's station");
    assert_eq!(reps[2], StopIdx(2), "stop 2 is its own station");
}

#[test]
fn walk_from_chains_transfers() {
    // 0→1 (30 s), 1→2 (40 s); the graph is symmetric.
    let transfers = Transfers::from_edges(
        3,
        &[
            (StopIdx(0), StopIdx(1), 30, 30.0),
            (StopIdx(1), StopIdx(0), 30, 30.0),
            (StopIdx(1), StopIdx(2), 40, 40.0),
            (StopIdx(2), StopIdx(1), 40, 40.0),
        ],
    )
    .unwrap();
    let walk = walk_from(&transfers, StopIdx(0), 3);
    assert_eq!(walk, vec![0, 30, 70]);
    let isolated = walk_from(&Transfers::empty(3), StopIdx(0), 3);
    assert_eq!(isolated, vec![0, NEVER, NEVER]);
}

#[test]
fn departures_profile_the_boardable_trips() {
    let timetable = two_lines();
    let view = DayView::universal(&timetable);
    // Source is stop 0 with no walking: only trips boardable at 0.
    let direct = walk_from(&Transfers::empty(4), StopIdx(0), 4);
    let station = stations(&Transfers::empty(4), 4);
    let departures = collect_departures(
        &view,
        &timetable,
        &direct,
        StopIdx(0),
        &station,
        0,
        NEVER - 1,
    );
    // Line A departs stop 0 at 100; that is the one source departure,
    // and it can board line A at position 0.
    assert_eq!(departures.len(), 1, "{departures:?}");
    assert_eq!(departures[0].time, 100);
    assert!(departures[0]
        .routes
        .contains(&(view.line_of(ViewTrip(0)), 0)));
}

#[test]
fn emits_the_needed_intermediate_transfer() {
    // Board line A at 0, alight 1 (t=100), walk 1→2 (50 s), board
    // line B at 2 (departs 200), alight 3. Reaching 3 needs the
    // 1→2 intermediate transfer and nothing else provides it, so it
    // is the one shortcut.
    let mut builder = TimetableBuilder::new(4);
    let a = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let b = builder.add_pattern(&[StopIdx(2), StopIdx(3)], 1).unwrap();
    builder.add_trip(a, vec![time(0), time(100)], 0, 0).unwrap();
    builder
        .add_trip(b, vec![time(200), time(300)], 1, 0)
        .unwrap();
    let timetable = builder.finish();
    let view = DayView::universal(&timetable);
    let transfers = Transfers::from_edges(
        4,
        &[
            (StopIdx(1), StopIdx(2), 50, 50.0),
            (StopIdx(2), StopIdx(1), 50, 50.0),
        ],
    )
    .unwrap();
    let shortcuts = compute_shortcuts(&view, &timetable, &transfers, 0, NEVER - 1);
    assert_eq!(
        shortcuts,
        vec![Shortcut {
            origin: StopIdx(1),
            destination: StopIdx(2),
            seconds: 50,
            meters: 50.0,
        }]
    );
}

#[test]
fn a_faster_direct_walk_witnesses_away_the_shortcut() {
    // Same as above, but the source can also walk straight to stop 2
    // (30 s) and catch line B without ever riding line A — a witness
    // that dominates the ride-then-walk candidate, so no shortcut.
    let mut builder = TimetableBuilder::new(4);
    let a = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let b = builder.add_pattern(&[StopIdx(2), StopIdx(3)], 1).unwrap();
    builder.add_trip(a, vec![time(0), time(100)], 0, 0).unwrap();
    builder
        .add_trip(b, vec![time(200), time(300)], 1, 0)
        .unwrap();
    let timetable = builder.finish();
    let view = DayView::universal(&timetable);
    let transfers = Transfers::from_edges(
        4,
        &[
            (StopIdx(1), StopIdx(2), 50, 50.0),
            (StopIdx(2), StopIdx(1), 50, 50.0),
            (StopIdx(0), StopIdx(2), 30, 30.0),
            (StopIdx(2), StopIdx(0), 30, 30.0),
        ],
    )
    .unwrap();
    let shortcuts = compute_shortcuts(&view, &timetable, &transfers, 0, NEVER - 1);
    assert!(shortcuts.is_empty(), "{shortcuts:?}");
}

#[test]
fn the_shortcut_survives_a_final_transfer() {
    // The candidate journey reaches its destination only after a
    // final walk (3→4) past the second trip. The intermediate 1→2
    // transfer is still the single shortcut; the final walk is never
    // itself an intermediate transfer (nothing boards at 4).
    let mut builder = TimetableBuilder::new(5);
    let a = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let b = builder.add_pattern(&[StopIdx(2), StopIdx(3)], 1).unwrap();
    builder.add_trip(a, vec![time(0), time(100)], 0, 0).unwrap();
    builder
        .add_trip(b, vec![time(200), time(300)], 1, 0)
        .unwrap();
    let timetable = builder.finish();
    let view = DayView::universal(&timetable);
    let transfers = Transfers::from_edges(
        5,
        &[
            (StopIdx(1), StopIdx(2), 50, 50.0),
            (StopIdx(2), StopIdx(1), 50, 50.0),
            (StopIdx(3), StopIdx(4), 30, 30.0),
            (StopIdx(4), StopIdx(3), 30, 30.0),
        ],
    )
    .unwrap();
    let shortcuts = compute_shortcuts(&view, &timetable, &transfers, 0, NEVER - 1);
    assert_eq!(
        shortcuts,
        vec![Shortcut {
            origin: StopIdx(1),
            destination: StopIdx(2),
            seconds: 50,
            meters: 50.0,
        }]
    );
}

#[test]
fn the_shortcut_metres_sum_a_multi_hop_walk() {
    // The intermediate transfer chains 1→2 (30 s / 30 m) and 2→3
    // (20 s / 20 m); the shortcut carries the summed distance, 50 m.
    let mut builder = TimetableBuilder::new(5);
    let a = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let b = builder.add_pattern(&[StopIdx(3), StopIdx(4)], 1).unwrap();
    builder.add_trip(a, vec![time(0), time(100)], 0, 0).unwrap();
    builder
        .add_trip(b, vec![time(200), time(300)], 1, 0)
        .unwrap();
    let timetable = builder.finish();
    let view = DayView::universal(&timetable);
    let transfers = Transfers::from_edges(
        5,
        &[
            (StopIdx(1), StopIdx(2), 30, 30.0),
            (StopIdx(2), StopIdx(1), 30, 30.0),
            (StopIdx(2), StopIdx(3), 20, 20.0),
            (StopIdx(3), StopIdx(2), 20, 20.0),
        ],
    )
    .unwrap();
    let shortcuts = compute_shortcuts(&view, &timetable, &transfers, 0, NEVER - 1);
    assert_eq!(
        shortcuts,
        vec![Shortcut {
            origin: StopIdx(1),
            destination: StopIdx(3),
            seconds: 50,
            meters: 50.0,
        }]
    );
}
