use super::*;
use crate::{Route, RouteType, Stop, Trip};

fn stop(feed_stop: u32) -> Stop {
    Stop {
        feed: 0,
        id: feed_stop.to_string(),
        code: None,
        name: None,
        latitude: None,
        longitude: None,
        parent_station: None,
    }
}

fn trip(id: &str, stop_times: Vec<crate::StopTime>) -> Trip {
    Trip {
        feed: 0,
        id: id.to_string(),
        route: 0,
        service_id: "s".to_string(),
        direction_id: None,
        shape_id: None,
        headsign: None,
        stop_times,
    }
}

fn call(stop: u32, arrival: u32, departure: u32, stop_sequence: u32) -> crate::StopTime {
    crate::StopTime {
        stop,
        arrival: Some(arrival),
        departure: Some(departure),
        stop_sequence,
        shape_dist_traveled: None,
    }
}

fn blank(stop: u32, stop_sequence: u32) -> crate::StopTime {
    crate::StopTime {
        stop,
        arrival: None,
        departure: None,
        stop_sequence,
        shape_dist_traveled: None,
    }
}

#[test]
fn interpolates_blank_interior_stop_times() {
    use cafein_core::timetable::TripIdx;

    let feed = Feed {
        stops: vec![stop(0), stop(1), stop(2), stop(3)],
        routes: vec![Route {
            feed: 0,
            id: "r".to_string(),
            short_name: None,
            long_name: None,
            route_type: RouteType::Bus,
            agency_id: None,
        }],
        trips: vec![
            trip(
                "timepoints",
                vec![
                    call(0, 0, 0, 1),
                    blank(1, 2),
                    blank(2, 3),
                    call(3, 300, 300, 4),
                ],
            ),
            trip("headless", vec![blank(0, 1), call(1, 100, 100, 2)]),
            trip("tailless", vec![call(0, 0, 0, 1), blank(1, 2)]),
        ],
        ..Feed::default()
    };
    let build = build_timetable(&feed).unwrap();
    // The anchored trip rides with evenly spaced interior times; the
    // trips missing a first or last time stay quarantined.
    assert_eq!(build.timetable.trip_count(), 1);
    assert_eq!(build.interpolated, vec![0]);
    let times = build.timetable.trip_stop_times(TripIdx(0));
    assert_eq!(
        times.iter().map(|time| time.arrival).collect::<Vec<_>>(),
        vec![0, 100, 200, 300]
    );
    assert_eq!(times[1].departure, 100);
    let mut dropped: Vec<u32> = build.quarantined.iter().map(|q| q.trip).collect();
    dropped.sort_unstable();
    assert_eq!(dropped, vec![1, 2]);
}

#[test]
fn quarantines_backwards_trips_instead_of_failing() {
    let feed = Feed {
        stops: vec![stop(0), stop(1)],
        routes: vec![Route {
            feed: 0,
            id: "r".to_string(),
            short_name: None,
            long_name: None,
            route_type: RouteType::Bus,
            agency_id: None,
        }],
        trips: vec![
            trip("good", vec![call(0, 0, 0, 1), call(1, 60, 60, 2)]),
            trip("backwards", vec![call(0, 100, 100, 1), call(1, 40, 40, 2)]),
        ],
        ..Feed::default()
    };
    let build = build_timetable(&feed).unwrap();
    assert_eq!(build.timetable.trip_count(), 1);
    assert_eq!(build.quarantined.len(), 1);
    assert_eq!(build.quarantined[0].trip, 1);
}
