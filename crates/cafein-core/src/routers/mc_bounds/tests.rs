use super::resolved_bounds;
use crate::routers::router::Request;
use crate::tbtr::DayView;
use crate::timetable::{StopIdx, StopTime, TimetableBuilder};
use crate::transfers::Transfers;

fn time(at: u32) -> StopTime {
    StopTime {
        arrival: at,
        departure: at,
    }
}

#[test]
fn bounds_track_rides_across_passes() {
    // The late pass reaches M faster but exhausts the two-ride cap;
    // the early pass needs its own slower one-ride arrival at M to
    // continue to D. A ride-blind sweep suppresses that label and
    // loses D's bound.
    let mut builder = TimetableBuilder::new(4);
    let a1 = builder.add_pattern(&[StopIdx(0), StopIdx(1)], 0).unwrap();
    let a2 = builder.add_pattern(&[StopIdx(1), StopIdx(2)], 1).unwrap();
    let b = builder.add_pattern(&[StopIdx(0), StopIdx(2)], 2).unwrap();
    let cd = builder.add_pattern(&[StopIdx(2), StopIdx(3)], 3).unwrap();
    builder
        .add_trip(a1, vec![time(300), time(320)], 0, 0)
        .unwrap();
    builder
        .add_trip(a2, vec![time(340), time(360)], 1, 0)
        .unwrap();
    builder
        .add_trip(b, vec![time(10), time(380)], 2, 0)
        .unwrap();
    builder
        .add_trip(cd, vec![time(400), time(450)], 3, 0)
        .unwrap();
    let timetable = builder.finish();
    let view = DayView::universal(&timetable);
    let footpaths = Transfers::empty(4);
    let request = Request {
        departure: 0,
        access: vec![(StopIdx(0), 0)],
        egress: vec![(StopIdx(3), 0)],
        active_services: vec![true],
        active_services_previous: vec![false],
        max_transfers: 1,
        exclusions: None,
    };
    let bounds = resolved_bounds(
        &view,
        &timetable,
        &footpaths,
        &[10.0; 4],
        &request,
        &[300, 0],
    );
    // The late pass cannot reach D within two rides; the early one
    // reaches it at 450 through its own one-ride label at M.
    assert_eq!(bounds[0][3], u32::MAX);
    assert_eq!(bounds[1][3], 450);
    assert_eq!(bounds[1][2], 360);
}
