//! Building a routing timetable from a merged [`Feed`].

use std::collections::hash_map::Entry;
use std::collections::HashMap;

use cafein_core::timetable::{PatternIdx, StopIdx, StopTime, Timetable, TimetableBuilder};

use crate::{Error, Feed, RouteIndex};

/// Extracts stop-sequence patterns from `feed` and assembles a [`Timetable`].
///
/// Patterns are keyed by `(route, stop sequence)`, so per-pattern data such
/// as trip geometry stays consistent within one route. Timetable stops map
/// 1:1 onto `feed.stops`; trip sources and pattern routes are indices into
/// `feed.trips` and `feed.routes`. Trips without stop times are skipped.
pub fn build_timetable(feed: &Feed) -> Result<Timetable, Error> {
    let mut builder = TimetableBuilder::new(feed.stops.len() as u32);
    let mut pattern_index: HashMap<(RouteIndex, Vec<StopIdx>), PatternIdx> = HashMap::new();
    for (trip_index, trip) in feed.trips.iter().enumerate() {
        if trip.stop_times.is_empty() {
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
        let mut stop_times = Vec::with_capacity(trip.stop_times.len());
        for stop_time in &trip.stop_times {
            let (arrival, departure) = match (stop_time.arrival, stop_time.departure) {
                (Some(arrival), Some(departure)) => (arrival, departure),
                (Some(arrival), None) => (arrival, arrival),
                (None, Some(departure)) => (departure, departure),
                (None, None) => {
                    return Err(Error::MissingStopTime {
                        trip_id: trip.id.clone(),
                    })
                }
            };
            stop_times.push(StopTime { arrival, departure });
        }
        builder.add_trip(pattern, stop_times, trip_index as u32)?;
    }
    Ok(builder.finish())
}
