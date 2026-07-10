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
#[path = "timetable_tests.rs"]
mod tests;
