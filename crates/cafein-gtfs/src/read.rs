//! Reading GTFS archives or directories into a [`Feed`].

use std::collections::HashMap;
use std::io::{Cursor, Read as _, Write as _};
use std::path::Path;

use gtfs_structures::{Availability, BikesAllowedType, DirectionType, Gtfs, GtfsReader, RawGtfs};

use crate::model::{
    Agency, Calendar, CalendarDate, Feed, FeedIndex, FeedInfo, Route, RouteIndex, SkippedFile,
    Stop, StopIndex, StopTime, Trip,
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
            let feed_index = feed_index as FeedIndex;
            let gtfs = read_gtfs(path.as_ref(), feed_index, &mut feed.skipped_files)?;
            append_gtfs(&mut feed, feed_index, gtfs)?;
        }
        feed.feed_count = paths.len() as FeedIndex;
        Ok(feed)
    }
}

/// Reads one feed, tolerating what routing can do without.
///
/// The strict parser rejects a whole feed over one bad value, so the
/// tables are read raw and repaired before the feed is assembled. A
/// malformed cosmetic colour in routes.txt is retried on an in-memory
/// copy with the colour columns dropped (the input itself is never
/// modified), and the optional tables routing never consults are
/// dropped, a parse failure among them recorded in `skipped` rather
/// than failing the read. Any other failure surfaces.
fn read_gtfs(path: &Path, feed: FeedIndex, skipped: &mut Vec<SkippedFile>) -> Result<Gtfs, Error> {
    let reader = GtfsReader::default().read_shapes(false).raw();
    let mut raw = reader.read_from_path(path)?;
    if let Some(sanitized) = colour_free_copy(&raw, path) {
        raw = reader.read_from_reader(Cursor::new(sanitized))?;
    }
    skip_unused_tables(&mut raw, feed, skipped);
    Gtfs::try_from(raw).map_err(Into::into)
}

/// The colour-less archive copy to retry on when routes.txt failed to
/// parse; `None` when it parsed, or when the copy cannot be assembled
/// (the original failure then surfaces).
fn colour_free_copy(raw: &RawGtfs, path: &Path) -> Option<Vec<u8>> {
    match raw.routes {
        Err(gtfs_structures::Error::CSVError { .. }) => archive_without_colours(path).ok(),
        _ => None,
    }
}

/// Drops the optional tables routing never consults. A parse failure
/// among them is recorded in `skipped`; a clean table goes too, so its
/// cross-references (a transfer to an unknown stop, say) cannot fail
/// the feed either. feed_info.txt is kept when it parses and skipped
/// otherwise. The raw feed is destructured in full, so a table a
/// parser upgrade adds has to be placed here.
fn skip_unused_tables(raw: &mut RawGtfs, feed: FeedIndex, skipped: &mut Vec<SkippedFile>) {
    let RawGtfs {
        fare_attributes,
        fare_rules,
        fare_products,
        fare_media,
        rider_categories,
        frequencies,
        transfers,
        pathways,
        translations,
        ticketing_deep_links,
        ticketing_identifiers,
        feed_info,
        agencies: _,
        stops: _,
        routes: _,
        trips: _,
        stop_times: _,
        calendar: _,
        calendar_dates: _,
        shapes: _,
        files: _,
        source_format: _,
        sha256: _,
        read_duration: _,
    } = raw;
    take_table(fare_attributes, "fare_attributes.txt", feed, skipped);
    take_table(fare_rules, "fare_rules.txt", feed, skipped);
    take_table(fare_products, "fare_products.txt", feed, skipped);
    take_table(fare_media, "fare_media.txt", feed, skipped);
    take_table(rider_categories, "rider_categories.txt", feed, skipped);
    take_table(frequencies, "frequencies.txt", feed, skipped);
    take_table(transfers, "transfers.txt", feed, skipped);
    take_table(pathways, "pathways.txt", feed, skipped);
    take_table(translations, "translations.txt", feed, skipped);
    take_table(
        ticketing_deep_links,
        "ticketing_deep_links.txt",
        feed,
        skipped,
    );
    take_table(
        ticketing_identifiers,
        "ticketing_identifiers.txt",
        feed,
        skipped,
    );
    if matches!(feed_info, Some(Err(_))) {
        take_table(feed_info, "feed_info.txt", feed, skipped);
    }
}

/// Removes an optional table from the raw feed, recording it in
/// `skipped` when it had failed to parse.
fn take_table<T>(
    table: &mut Option<Result<Vec<T>, gtfs_structures::Error>>,
    file_name: &str,
    feed: FeedIndex,
    skipped: &mut Vec<SkippedFile>,
) {
    if let Some(Err(error)) = table.take() {
        skipped.push(SkippedFile {
            feed,
            file_name: file_name.to_string(),
            reason: crate::error_chain(&error),
        });
    }
}

type SanitizeError = Box<dyn std::error::Error + Send + Sync>;

/// An in-memory zip copy of the feed with the colour columns dropped
/// from routes.txt.
fn archive_without_colours(path: &Path) -> Result<Vec<u8>, SanitizeError> {
    let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
    let options = zip::write::SimpleFileOptions::default();
    if path.is_dir() {
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            let mut bytes = std::fs::read(entry.path())?;
            if name == "routes.txt" {
                bytes = without_colour_columns(&bytes)?;
            }
            writer.start_file(name, options)?;
            writer.write_all(&bytes)?;
        }
    } else {
        let mut archive = zip::ZipArchive::new(std::fs::File::open(path)?)?;
        for index in 0..archive.len() {
            let entry = archive.by_index_raw(index)?;
            if entry.is_file() && entry.name().ends_with("routes.txt") {
                let name = entry.name().to_string();
                drop(entry);
                let mut bytes = Vec::new();
                archive.by_index(index)?.read_to_end(&mut bytes)?;
                writer.start_file(name, options)?;
                writer.write_all(&without_colour_columns(&bytes)?)?;
            } else {
                writer.raw_copy_file(entry)?;
            }
        }
    }
    Ok(writer.finish()?.into_inner())
}

/// routes.txt with the `route_color`/`route_text_color` columns removed.
///
/// Record lengths are enforced strictly: a ragged row fails the rewrite
/// (and thereby the retry), so the colour fallback cannot mask a real
/// shape problem in routes.txt.
fn without_colour_columns(bytes: &[u8]) -> Result<Vec<u8>, SanitizeError> {
    let bytes = bytes.strip_prefix(b"\xef\xbb\xbf").unwrap_or(bytes);
    let mut reader = csv::Reader::from_reader(bytes);
    let headers = reader.headers()?.clone();
    let kept: Vec<usize> = headers
        .iter()
        .enumerate()
        .filter(|(_, name)| !matches!(name.trim(), "route_color" | "route_text_color"))
        .map(|(index, _)| index)
        .collect();
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record(kept.iter().map(|&index| &headers[index]))?;
    for record in reader.records() {
        let record = record?;
        writer.write_record(kept.iter().map(|&index| &record[index]))?;
    }
    Ok(writer.into_inner()?)
}

/// The GTFS availability tri-state as a flag: no information and
/// out-of-spec values stay unknown.
fn availability_flag(availability: Availability) -> Option<bool> {
    match availability {
        Availability::Available => Some(true),
        Availability::NotAvailable => Some(false),
        _ => None,
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
            wheelchair_boarding: availability_flag(stop.wheelchair_boarding),
        });
    }
    // A stop that says nothing inherits its parent station's known
    // value (GTFS wheelchair_boarding semantics), one level: only the
    // parent's own field counts, from a snapshot so order cannot chain.
    let own: Vec<Option<bool>> = feed.stops[stop_base as usize..]
        .iter()
        .map(|stop| stop.wheelchair_boarding)
        .collect();
    for offset in 0..own.len() {
        if own[offset].is_some() {
            continue;
        }
        let index = stop_base as usize + offset;
        let inherited = feed.stops[index]
            .parent_station
            .as_ref()
            .and_then(|id| stop_index_by_id.get(id))
            .and_then(|&parent| own[(parent - stop_base) as usize]);
        feed.stops[index].wheelchair_boarding = inherited;
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
            bikes_allowed: match trip.bikes_allowed {
                BikesAllowedType::AtLeastOneBike => Some(true),
                BikesAllowedType::NoBikesAllowed => Some(false),
                // No information and out-of-spec values stay unknown.
                _ => None,
            },
            wheelchair_accessible: availability_flag(trip.wheelchair_accessible),
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

#[cfg(test)]
#[path = "read_tests.rs"]
mod tests;
