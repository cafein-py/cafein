//! Feed-level data-quality checks.

use crate::model::Feed;
use crate::service::ServiceCalendar;

/// A data-quality finding from [`validate_feed`].
#[derive(Debug, PartialEq, Eq)]
pub enum QaFinding {
    /// A stop (index into `feed.stops`) has no coordinates and cannot be
    /// placed on the map or connected to the street network.
    StopWithoutCoordinates { stop: u32 },
    /// A trip (index into `feed.trips`) references a service with no
    /// calendar data; the trip never runs on any date.
    TripServiceNeverRuns { trip: u32 },
}

/// Checks `feed` for data problems that degrade routing quality.
///
/// Findings are reported, not fixed: degraded data must stay visible.
/// Trips with broken stop times are quarantined separately during
/// [`build_timetable`](crate::build_timetable).
pub fn validate_feed(feed: &Feed, services: &ServiceCalendar) -> Vec<QaFinding> {
    let mut findings = Vec::new();
    for (index, stop) in feed.stops.iter().enumerate() {
        if stop.latitude.is_none() || stop.longitude.is_none() {
            findings.push(QaFinding::StopWithoutCoordinates { stop: index as u32 });
        }
    }
    for (index, trip) in feed.trips.iter().enumerate() {
        match services.index(trip.feed, &trip.service_id) {
            Some(service) if services.has_calendar_data(service) => {}
            _ => findings.push(QaFinding::TripServiceNeverRuns { trip: index as u32 }),
        }
    }
    findings
}

#[cfg(test)]
mod tests {
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
}
