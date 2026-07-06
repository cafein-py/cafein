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
    /// Trips (indices into `feed.trips`) whose blank interior stop times
    /// were filled by interpolation between the surrounding timed stops.
    pub interpolated: Vec<u32>,
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
/// Blank interior stop times — legal at non-timepoint stops — are filled
/// by linear interpolation between the surrounding timed stops, as
/// timepoint-only feeds expect of their consumers; repaired trips are
/// reported in [`TimetableBuild::interpolated`]. Trips with data-quality
/// problems — no stop times, a first or last stop without any time, or
/// times going backwards — are quarantined rather than failing the build;
/// they are reported in [`TimetableBuild::quarantined`].
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
    let mut interpolated = Vec::new();
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
        let mut timed: Vec<Option<StopTime>> = trip
            .stop_times
            .iter()
            .map(|stop_time| match (stop_time.arrival, stop_time.departure) {
                (Some(arrival), Some(departure)) => Some(StopTime { arrival, departure }),
                (Some(arrival), None) => Some(StopTime {
                    arrival,
                    departure: arrival,
                }),
                (None, Some(departure)) => Some(StopTime {
                    arrival: departure,
                    departure,
                }),
                (None, None) => None,
            })
            .collect();
        if timed.iter().any(Option::is_none) {
            if interpolate_stop_times(&mut timed).is_err() {
                quarantined.push(QuarantinedTrip {
                    trip: trip_index as u32,
                    reason: Error::MissingStopTime {
                        trip_id: trip.id.clone(),
                    },
                });
                continue;
            }
            interpolated.push(trip_index as u32);
        }
        let stop_times: Vec<StopTime> = timed.into_iter().flatten().collect();
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
        interpolated,
    })
}

/// Fills blank interior stop times by linear interpolation between the
/// surrounding timed stops, evenly by stop position. `Err` means the
/// trip cannot be anchored: its first or last stop carries no time.
/// Backwards anchors are left to the builder's non-increasing check.
fn interpolate_stop_times(times: &mut [Option<StopTime>]) -> Result<(), ()> {
    if times.first().copied().flatten().is_none() || times.last().copied().flatten().is_none() {
        return Err(());
    }
    let mut anchor = 0;
    for index in 1..times.len() {
        let Some(next) = times[index] else { continue };
        let start = times[anchor].expect("anchors are timed").departure;
        let span = next.arrival.saturating_sub(start) as u64;
        let gaps = (index - anchor) as u64;
        for (offset, slot) in times[anchor + 1..index].iter_mut().enumerate() {
            let time = start + (span * (offset as u64 + 1) / gaps) as u32;
            *slot = Some(StopTime {
                arrival: time,
                departure: time,
            });
        }
        anchor = index;
    }
    Ok(())
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
}
