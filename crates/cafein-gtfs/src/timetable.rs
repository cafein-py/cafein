//! Building a routing timetable from a merged [`Feed`].

use std::collections::hash_map::Entry;
use std::collections::HashMap;

use cafein_core::timetable::{
    PatternIdx, StopIdx, StopTime, Timetable, TimetableBuilder, TimetableError,
};

use crate::service::ServiceCalendar;
use crate::{Error, Feed, RouteIndex};

/// The outcome of building a timetable: the routing structure, the service
/// calendar resolving its trips' service identifiers, and the trips that
/// were quarantined for data-quality problems.
#[derive(Debug)]
pub struct TimetableBuild {
    pub timetable: Timetable,
    /// Resolves the service identifiers carried on timetable trips; combine
    /// [`ServiceCalendar::active_on`] with
    /// [`Timetable::trip_service`](cafein_core::timetable::Timetable::trip_service)
    /// to restrict a query date to its running trips.
    pub services: ServiceCalendar,
    /// Trips excluded from the timetable, with the reason each was dropped.
    pub quarantined: Vec<QuarantinedTrip>,
}

/// A trip excluded from the timetable, as an index into `feed.trips`.
#[derive(Debug)]
pub struct QuarantinedTrip {
    pub trip: u32,
    pub reason: Error,
}

/// Extracts stop-sequence patterns from `feed` and assembles a [`Timetable`].
///
/// Patterns are keyed by `(route, stop sequence)`, so per-pattern data such
/// as trip geometry stays consistent within one route. Timetable stops map
/// 1:1 onto `feed.stops`; trip sources and pattern routes are indices into
/// `feed.trips` and `feed.routes`.
///
/// Trips with data-quality problems — no stop times, a stop time missing
/// both arrival and departure, or times going backwards — are quarantined
/// rather than failing the build; they are reported in
/// [`TimetableBuild::quarantined`].
///
/// The timetable spans the feed's whole service period: every usable trip
/// is included regardless of service day. Service-calendar resolution
/// (restricting a query to the trips active on its date) is layered on top
/// separately and is not part of the timetable build.
pub fn build_timetable(feed: &Feed) -> Result<TimetableBuild, Error> {
    let services = ServiceCalendar::from_feed(feed);
    let mut builder = TimetableBuilder::new(feed.stops.len() as u32);
    let mut pattern_index: HashMap<(RouteIndex, Vec<StopIdx>), PatternIdx> = HashMap::new();
    let mut quarantined = Vec::new();
    for (trip_index, trip) in feed.trips.iter().enumerate() {
        if trip.stop_times.is_empty() {
            quarantined.push(QuarantinedTrip {
                trip: trip_index as u32,
                reason: Error::MissingStopTime {
                    trip_id: trip.id.clone(),
                },
            });
            continue;
        }
        let mut stop_times = Vec::with_capacity(trip.stop_times.len());
        let mut missing_times = false;
        for stop_time in &trip.stop_times {
            let (arrival, departure) = match (stop_time.arrival, stop_time.departure) {
                (Some(arrival), Some(departure)) => (arrival, departure),
                (Some(arrival), None) => (arrival, arrival),
                (None, Some(departure)) => (departure, departure),
                (None, None) => {
                    missing_times = true;
                    break;
                }
            };
            stop_times.push(StopTime { arrival, departure });
        }
        if missing_times {
            quarantined.push(QuarantinedTrip {
                trip: trip_index as u32,
                reason: Error::MissingStopTime {
                    trip_id: trip.id.clone(),
                },
            });
            continue;
        }
        let stops: Vec<StopIdx> = trip
            .stop_times
            .iter()
            .map(|stop_time| StopIdx(stop_time.stop))
            .collect();
        let pattern = match pattern_index.entry((trip.route, stops)) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                let pattern = builder.add_pattern(&entry.key().1, trip.route)?;
                *entry.insert(pattern)
            }
        };
        let service = services
            .index(trip.feed, &trip.service_id)
            .expect("trip services are interned by ServiceCalendar::from_feed");
        match builder.add_trip(pattern, stop_times, trip_index as u32, service) {
            Ok(()) => {}
            Err(error @ TimetableError::NonIncreasingStopTimes { .. }) => {
                quarantined.push(QuarantinedTrip {
                    trip: trip_index as u32,
                    reason: Error::Timetable(error),
                });
            }
            Err(error) => return Err(Error::Timetable(error)),
        }
    }
    Ok(TimetableBuild {
        timetable: builder.finish(),
        services,
        quarantined,
    })
}

#[cfg(test)]
mod tests {
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
}
