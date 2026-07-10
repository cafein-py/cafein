use super::*;
use crate::{Stop, Trip};

#[test]
fn flags_coordinate_less_stops_and_never_running_trips() {
    let feed = Feed {
        stops: vec![
            Stop {
                feed: 0,
                id: "located".to_string(),
                code: None,
                name: None,
                latitude: Some(60.0),
                longitude: Some(24.0),
                parent_station: None,
            },
            Stop {
                feed: 0,
                id: "nowhere".to_string(),
                code: None,
                name: None,
                latitude: None,
                longitude: None,
                parent_station: None,
            },
        ],
        trips: vec![Trip {
            feed: 0,
            id: "ghost-trip".to_string(),
            route: 0,
            service_id: "ghost".to_string(),
            direction_id: None,
            shape_id: None,
            headsign: None,
            stop_times: Vec::new(),
        }],
        ..Feed::default()
    };
    let services = ServiceCalendar::from_feed(&feed);
    let findings = validate_feed(&feed, &services);
    assert_eq!(
        findings,
        vec![
            QaFinding::StopWithoutCoordinates { stop: 1 },
            QaFinding::TripServiceNeverRuns { trip: 0 },
        ]
    );
}
