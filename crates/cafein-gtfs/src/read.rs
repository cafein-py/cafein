//! Reading GTFS archives or directories into a [`Feed`].

use std::collections::HashMap;
use std::path::Path;

use gtfs_structures::{DirectionType, Gtfs, GtfsReader};

use crate::model::{
    Agency, Calendar, CalendarDate, Feed, FeedIndex, FeedInfo, Route, RouteIndex, Stop, StopIndex,
    StopTime, Trip,
};
use crate::Error;

impl Feed {
    /// Reads a single GTFS feed from a zip archive or a directory.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Feed, Error> {
        Feed::from_paths(&[path])
    }

    /// Reads several GTFS feeds and merges them into one [`Feed`].
    ///
    /// Entities from the n-th input get `feed == n`, so identifiers that
    /// repeat across inputs stay distinct as `(feed, id)` pairs. Within each
    /// feed, entities are indexed in lexicographic identifier order, making
    /// the merge independent of input file ordering.
    pub fn from_paths<P: AsRef<Path>>(paths: &[P]) -> Result<Feed, Error> {
        let mut feed = Feed::default();
        for (feed_index, path) in paths.iter().enumerate() {
            let gtfs = GtfsReader::default()
                .read_shapes(false)
                .read_from_path(path)?;
            append_gtfs(&mut feed, feed_index as FeedIndex, gtfs)?;
        }
        feed.feed_count = paths.len() as FeedIndex;
        Ok(feed)
    }
}

fn append_gtfs(feed: &mut Feed, feed_index: FeedIndex, gtfs: Gtfs) -> Result<(), Error> {
    for agency in gtfs.agencies {
        feed.agencies.push(Agency {
            feed: feed_index,
            id: agency.id,
            name: agency.name,
            timezone: agency.timezone,
        });
    }

    let stop_base = feed.stops.len() as StopIndex;
    let mut stops: Vec<_> = gtfs.stops.into_iter().collect();
    stops.sort_by(|left, right| left.0.cmp(&right.0));
    let mut stop_index_by_id: HashMap<String, StopIndex> = HashMap::with_capacity(stops.len());
    for (offset, (id, stop)) in stops.into_iter().enumerate() {
        stop_index_by_id.insert(id, stop_base + offset as StopIndex);
        feed.stops.push(Stop {
            feed: feed_index,
            id: stop.id.clone(),
            code: stop.code.clone(),
            name: stop.name.clone(),
            latitude: stop.latitude,
            longitude: stop.longitude,
            parent_station: stop.parent_station.clone(),
        });
    }

    let route_base = feed.routes.len() as RouteIndex;
    let mut routes: Vec<_> = gtfs.routes.into_iter().collect();
    routes.sort_by(|left, right| left.0.cmp(&right.0));
    let mut route_index_by_id: HashMap<String, RouteIndex> = HashMap::with_capacity(routes.len());
    for (offset, (id, route)) in routes.into_iter().enumerate() {
        route_index_by_id.insert(id, route_base + offset as RouteIndex);
        feed.routes.push(Route {
            feed: feed_index,
            id: route.id,
            short_name: route.short_name,
            long_name: route.long_name,
            route_type: route.route_type,
            agency_id: route.agency_id,
        });
    }

    let mut trips: Vec<_> = gtfs.trips.into_iter().collect();
    trips.sort_by(|left, right| left.0.cmp(&right.0));
    for (id, trip) in trips {
        let route = *route_index_by_id
            .get(&trip.route_id)
            .ok_or_else(|| Error::UnknownRoute {
                trip_id: id.clone(),
                route_id: trip.route_id.clone(),
            })?;
        let mut stop_times = Vec::with_capacity(trip.stop_times.len());
        for stop_time in &trip.stop_times {
            let stop =
                *stop_index_by_id
                    .get(&stop_time.stop.id)
                    .ok_or_else(|| Error::UnknownStop {
                        trip_id: id.clone(),
                        stop_id: stop_time.stop.id.clone(),
                    })?;
            stop_times.push(StopTime {
                stop,
                arrival: stop_time.arrival_time,
                departure: stop_time.departure_time,
                stop_sequence: stop_time.stop_sequence,
                shape_dist_traveled: stop_time.shape_dist_traveled,
            });
        }
        stop_times.sort_by_key(|stop_time| stop_time.stop_sequence);
        feed.trips.push(Trip {
            feed: feed_index,
            id,
            route,
            service_id: trip.service_id,
            direction_id: trip.direction_id.map(|direction| match direction {
                DirectionType::Outbound => 0,
                DirectionType::Inbound => 1,
            }),
            shape_id: trip.shape_id,
            headsign: trip.trip_headsign,
            stop_times,
        });
    }

    let mut calendars: Vec<_> = gtfs.calendar.into_iter().collect();
    calendars.sort_by(|left, right| left.0.cmp(&right.0));
    for (service_id, calendar) in calendars {
        feed.calendars.push(Calendar {
            feed: feed_index,
            service_id,
            weekdays: [
                calendar.monday,
                calendar.tuesday,
                calendar.wednesday,
                calendar.thursday,
                calendar.friday,
                calendar.saturday,
                calendar.sunday,
            ],
            start_date: calendar.start_date,
            end_date: calendar.end_date,
        });
    }

    let mut calendar_dates: Vec<_> = gtfs.calendar_dates.into_iter().collect();
    calendar_dates.sort_by(|left, right| left.0.cmp(&right.0));
    for (service_id, dates) in calendar_dates {
        for date in dates {
            feed.calendar_dates.push(CalendarDate {
                feed: feed_index,
                service_id: service_id.clone(),
                date: date.date,
                exception: date.exception_type,
            });
        }
    }

    for info in gtfs.feed_info {
        feed.feed_infos.push(FeedInfo {
            feed: feed_index,
            publisher_name: info.name,
            version: info.version,
        });
    }

    Ok(())
}
