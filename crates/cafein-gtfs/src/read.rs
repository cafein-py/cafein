//! Reading GTFS archives or directories into a [`Feed`].

use std::collections::{HashMap, HashSet};
use std::io::{Cursor, Read as _, Write as _};
use std::path::Path;

use gtfs_structures::{
    Availability, BikesAllowedType, DirectionType, Gtfs, GtfsReader, RawFrequency, RawGtfs,
    RawStopTime, RawTrip,
};
use serde::de::DeserializeOwned;

use crate::model::{
    Agency, Calendar, CalendarDate, DroppedRows, Feed, FeedIndex, FeedInfo, Route, RouteIndex,
    SkippedFile, SkippedFrequency, Stop, StopIndex, StopTime, Trip,
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
            let (gtfs, frequencies) = read_gtfs(
                path.as_ref(),
                feed_index,
                &mut feed.skipped_files,
                &mut feed.dropped_rows,
            )?;
            append_gtfs(&mut feed, feed_index, gtfs, frequencies)?;
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
/// than failing the read. frequencies.txt is routing input, so its
/// parse failure surfaces; its rows are returned beside the feed for
/// expansion (the parser's own trip assembly never sees them). A row
/// of trips.txt or stop_times.txt that fails to parse drops the trip it
/// belongs to, recorded in `dropped`, rather than the feed. Any other
/// failure surfaces.
fn read_gtfs(
    path: &Path,
    feed: FeedIndex,
    skipped: &mut Vec<SkippedFile>,
    dropped: &mut Vec<DroppedRows>,
) -> Result<(Gtfs, Vec<RawFrequency>), Error> {
    let reader = GtfsReader::default().read_shapes(false).raw();
    let source = Source::open(path)?;
    let mut raw = match &source {
        Source::Archive(file) => {
            reader.read_from_reader(file.try_clone().map_err(gtfs_structures::Error::IO)?)?
        }
        Source::Directory(path) => reader.read_from_path(path)?,
    };
    if let Some(sanitized) = colour_free_copy(&raw, &source) {
        raw = reader.read_from_reader(Cursor::new(sanitized))?;
    }
    recover_trip_tables(&mut raw, &source, feed, dropped);
    let frequencies = raw.frequencies.take().transpose()?.unwrap_or_default();
    skip_unused_tables(&mut raw, feed, skipped);
    Ok((Gtfs::try_from(raw)?, frequencies))
}

/// The feed's source, opened once so every pass over it (the strict
/// parse, the colour retry, the lenient re-reads) sees the same bytes:
/// an archive is one open file, which replacing the path cannot
/// change; a directory is read table by table.
enum Source {
    Archive(std::fs::File),
    Directory(std::path::PathBuf),
}

impl Source {
    fn open(path: &Path) -> Result<Source, gtfs_structures::Error> {
        if path.is_dir() {
            Ok(Source::Directory(path.to_path_buf()))
        } else if path.is_file() {
            Ok(Source::Archive(std::fs::File::open(path)?))
        } else {
            Err(gtfs_structures::Error::NotFileNorDirectory(
                path.display().to_string(),
            ))
        }
    }

    /// The archive through a fresh handle on the same open file.
    fn archive(&self) -> std::io::Result<Option<zip::ZipArchive<std::fs::File>>> {
        match self {
            Source::Archive(file) => Ok(Some(zip::ZipArchive::new(file.try_clone()?)?)),
            Source::Directory(_) => Ok(None),
        }
    }
}

/// Re-reads trips.txt and stop_times.txt leniently when the strict
/// parse of either failed on a row. A failed row drops the whole trip
/// it belongs to (its trips.txt row and its every stop_times.txt row),
/// so no partial trip and no orphan reaches the feed. A pass that
/// cannot recover safely leaves the table's error in place, and it
/// surfaces.
fn recover_trip_tables(
    raw: &mut RawGtfs,
    source: &Source,
    feed: FeedIndex,
    dropped: &mut Vec<DroppedRows>,
) {
    let mut reports: Vec<(&str, Vec<DroppedRow>, HashSet<String>)> = Vec::new();
    if failed_on_a_row(&raw.trips) {
        if let Some((rows, bad)) = lenient_table::<RawTrip>(source, "trips.txt", Some("trip_id")) {
            let keys = bad.iter().filter_map(|row| row.key.clone()).collect();
            reports.push(("trips.txt", bad, keys));
            raw.trips = Ok(rows);
        }
    }
    if failed_on_a_row(&raw.stop_times) {
        if let Some((rows, bad)) =
            lenient_table::<RawStopTime>(source, "stop_times.txt", Some("trip_id"))
        {
            let keys = bad.iter().filter_map(|row| row.key.clone()).collect();
            reports.push(("stop_times.txt", bad, keys));
            raw.stop_times = Ok(rows);
        }
    }
    if reports.is_empty() {
        return;
    }
    let doomed: HashSet<&String> = reports.iter().flat_map(|(_, _, keys)| keys).collect();
    // The trips a report may take along: those that parsed, plus a
    // dropped trips.txt row's own trip.
    let mut known: HashSet<String> = raw
        .trips
        .as_ref()
        .map(|trips| trips.iter().map(|trip| trip.id.clone()).collect())
        .unwrap_or_default();
    for (file_name, _, keys) in &reports {
        if *file_name == "trips.txt" {
            known.extend(keys.iter().cloned());
        }
    }
    if let Ok(trips) = &mut raw.trips {
        trips.retain(|trip| !doomed.contains(&trip.id));
    }
    if let Ok(stop_times) = &mut raw.stop_times {
        stop_times.retain(|stop_time| !doomed.contains(&stop_time.trip_id));
    }
    for (file_name, bad, keys) in reports {
        dropped.push(DroppedRows {
            feed,
            file_name: file_name.to_string(),
            rows: bad.len() as u32,
            first_line: bad[0].line,
            first_reason: bad[0].reason.clone(),
            trips_dropped: keys.iter().filter(|id| known.contains(*id)).count() as u32,
            services_affected: 0,
        });
    }
}

fn failed_on_a_row<T>(table: &Result<Vec<T>, gtfs_structures::Error>) -> bool {
    matches!(table, Err(gtfs_structures::Error::CSVError { .. }))
}

/// A row the lenient re-read dropped.
struct DroppedRow {
    line: u64,
    /// The row's value in the key column, when the table cascades.
    key: Option<String>,
    reason: String,
}

/// One table read again leniently, from the archive or directory at
/// `path`; see [`lenient_rows`].
fn lenient_table<T: DeserializeOwned>(
    source: &Source,
    file_name: &str,
    key_column: Option<&str>,
) -> Option<(Vec<T>, Vec<DroppedRow>)> {
    lenient_rows(&table_bytes(source, file_name)?, key_column)
}

/// The rows of a table that parse, beside the rows that do not, read
/// as the strict parser reads them (flexible record lengths, trimmed
/// fields) so every row it would accept is kept. `None` when the pass
/// cannot recover safely: the table has no header or, for a cascading
/// table, no key column; a dropped row's key is blank or unreadable (a
/// ragged row shifts its columns); the reader itself fails; no row
/// parses; or none fails.
fn lenient_rows<T: DeserializeOwned>(
    bytes: &[u8],
    key_column: Option<&str>,
) -> Option<(Vec<T>, Vec<DroppedRow>)> {
    let bytes = bytes.strip_prefix(b"\xef\xbb\xbf").unwrap_or(bytes);
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .trim(csv::Trim::All)
        .from_reader(bytes);
    let headers = reader.headers().ok()?.clone();
    let key_index = match key_column {
        Some(name) => Some(headers.iter().position(|header| header == name)?),
        None => None,
    };
    let mut rows = Vec::new();
    let mut dropped = Vec::new();
    for record in reader.records() {
        let record = record.ok()?;
        match record.deserialize::<T>(Some(&headers)) {
            Ok(row) => rows.push(row),
            Err(error) => {
                let key = match key_index {
                    None => None,
                    Some(_) if record.len() != headers.len() => return None,
                    Some(index) => {
                        let key = record.get(index)?;
                        if key.is_empty() {
                            return None;
                        }
                        Some(key.to_string())
                    }
                };
                dropped.push(DroppedRow {
                    line: record.position().map_or(0, |position| position.line()),
                    key,
                    reason: error.to_string(),
                });
            }
        }
    }
    if rows.is_empty() || dropped.is_empty() {
        return None;
    }
    Some((rows, dropped))
}

/// One table's bytes: the archive entry whose file name matches (as
/// the parser resolves it) or the file in a directory.
fn table_bytes(source: &Source, file_name: &str) -> Option<Vec<u8>> {
    let Some(mut archive) = source.archive().ok()? else {
        let Source::Directory(path) = source else {
            unreachable!("a source without an archive is a directory");
        };
        return std::fs::read(path.join(file_name)).ok();
    };
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index).ok()?;
        if entry.is_file()
            && Path::new(entry.name()).file_name() == Some(std::ffi::OsStr::new(file_name))
        {
            let mut bytes = Vec::new();
            entry.read_to_end(&mut bytes).ok()?;
            return Some(bytes);
        }
    }
    None
}

/// The colour-less archive copy to retry on when routes.txt failed to
/// parse; `None` when it parsed, or when the copy cannot be assembled
/// (the original failure then surfaces).
fn colour_free_copy(raw: &RawGtfs, source: &Source) -> Option<Vec<u8>> {
    match raw.routes {
        Err(gtfs_structures::Error::CSVError { .. }) => archive_without_colours(source).ok(),
        _ => None,
    }
}

/// Drops the optional tables routing never consults. A parse failure
/// among them is recorded in `skipped`; a clean table goes too, so its
/// cross-references (a transfer to an unknown stop, say) cannot fail
/// the feed either. feed_info.txt is kept when it parses and skipped
/// otherwise; frequencies.txt was taken out before this. The raw feed
/// is destructured in full, so a table a parser upgrade adds has to be
/// placed here.
fn skip_unused_tables(raw: &mut RawGtfs, feed: FeedIndex, skipped: &mut Vec<SkippedFile>) {
    let RawGtfs {
        fare_attributes,
        fare_rules,
        fare_products,
        fare_media,
        rider_categories,
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
        frequencies: _,
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
fn archive_without_colours(source: &Source) -> Result<Vec<u8>, SanitizeError> {
    let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
    let options = zip::write::SimpleFileOptions::default();
    if let Source::Directory(path) = source {
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
        let mut archive = source.archive()?.expect("an archive source");
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

fn append_gtfs(
    feed: &mut Feed,
    feed_index: FeedIndex,
    gtfs: Gtfs,
    frequencies: Vec<RawFrequency>,
) -> Result<(), Error> {
    let mut rows_by_trip: HashMap<String, Vec<RawFrequency>> = HashMap::new();
    for row in frequencies {
        rows_by_trip
            .entry(row.trip_id.clone())
            .or_default()
            .push(row);
    }
    let mut budget = FREQUENCY_STOP_TIME_BUDGET;
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
        let rows = rows_by_trip.remove(&id);
        let trip = Trip {
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
        };
        match rows {
            None => feed.trips.push(trip),
            Some(rows) => expand_frequencies(feed, feed_index, trip, rows, &mut budget)?,
        }
    }
    let mut unknown: Vec<_> = rows_by_trip.into_iter().collect();
    unknown.sort_by(|left, right| left.0.cmp(&right.0));
    for (trip_id, rows) in unknown {
        for row in rows {
            feed.skipped_frequencies.push(SkippedFrequency {
                feed: feed_index,
                trip_id: trip_id.clone(),
                reason: format!("{}: no such trip in trips.txt", describe_window(&row)),
            });
        }
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

/// The stop times one feed's frequency expansion may create in total,
/// far above any real feed: a ceiling against a malformed row (a
/// one-second headway over a huge window, say) exhausting memory. The
/// feed is refused past it, since dropping runs silently would distort
/// the timetable.
const FREQUENCY_STOP_TIME_BUDGET: u64 = 100_000_000;

/// Replaces a frequencies.txt template with its runs: per row, one copy
/// of the template departing at `start + k·headway` for every k that
/// keeps it before `end`, every stop time shifted alike (both
/// `exact_times` values are treated the same). Rows that cannot be
/// expanded are reported; a template none of whose rows can is omitted
/// and reported, so a template is never routed at its literal times.
/// Every row's cost is charged to `budget` before any run is built.
fn expand_frequencies(
    feed: &mut Feed,
    feed_index: FeedIndex,
    template: Trip,
    rows: Vec<RawFrequency>,
    budget: &mut u64,
) -> Result<(), Error> {
    let base = template
        .stop_times
        .first()
        .and_then(|first| first.departure.or(first.arrival));
    let mut runs: Vec<Trip> = Vec::new();
    for row in rows {
        let outcome = match run_count(base, &row) {
            Err(problem) => Err(problem),
            Ok(count) => {
                let cost = count.saturating_mul(template.stop_times.len() as u64);
                *budget =
                    budget
                        .checked_sub(cost)
                        .ok_or_else(|| Error::FrequencyExpansionTooLarge {
                            trip_id: template.id.clone(),
                            limit: FREQUENCY_STOP_TIME_BUDGET,
                        })?;
                let expanded = expand_row(&template, base, &row);
                if expanded.is_err() {
                    // A rejected row creates nothing, so it costs nothing.
                    *budget += cost;
                }
                expanded
            }
        };
        match outcome {
            Ok(row_runs) => runs.extend(row_runs),
            Err(problem) => feed.skipped_frequencies.push(SkippedFrequency {
                feed: feed_index,
                trip_id: template.id.clone(),
                reason: format!("{}: {problem}", describe_window(&row)),
            }),
        }
    }
    if runs.is_empty() {
        feed.skipped_frequencies.push(SkippedFrequency {
            feed: feed_index,
            trip_id: template.id,
            reason: "omitted: none of its frequencies.txt rows could be expanded".to_string(),
        });
        return Ok(());
    }
    runs.sort_by_key(|run| {
        run.stop_times
            .first()
            .and_then(|first| first.departure.or(first.arrival))
    });
    feed.trips.extend(runs);
    Ok(())
}

/// How many runs a row describes, checked before any run is built, or
/// why the row cannot be expanded.
fn run_count(base: Option<u32>, row: &RawFrequency) -> Result<u64, String> {
    if base.is_none() {
        return Err("the template's first stop has no time".to_string());
    }
    if row.headway_secs == 0 {
        return Err("headway_secs is 0".to_string());
    }
    if row.end_time <= row.start_time {
        return Err("end_time is not after start_time".to_string());
    }
    Ok(u64::from(row.end_time - row.start_time).div_ceil(u64::from(row.headway_secs)))
}

/// One validated row's runs. A run whose shifted times leave the clock
/// the timetable represents (0 to `u32::MAX` seconds) rejects the row.
fn expand_row(template: &Trip, base: Option<u32>, row: &RawFrequency) -> Result<Vec<Trip>, String> {
    let base = base.expect("validated by run_count");
    let mut runs = Vec::new();
    let mut departure = row.start_time;
    while departure < row.end_time {
        let shift = i64::from(departure) - i64::from(base);
        let mut run = template.clone();
        for stop_time in &mut run.stop_times {
            stop_time.arrival = shift_time(stop_time.arrival, shift)?;
            stop_time.departure = shift_time(stop_time.departure, shift)?;
        }
        runs.push(run);
        let Some(next) = departure.checked_add(row.headway_secs) else {
            break;
        };
        departure = next;
    }
    Ok(runs)
}

/// A stop time moved by `shift` seconds; a blank stays blank.
fn shift_time(time: Option<u32>, shift: i64) -> Result<Option<u32>, String> {
    time.map(|time| {
        u32::try_from(i64::from(time) + shift).map_err(|_| {
            "a run would place a stop outside the clock the timetable represents".to_string()
        })
    })
    .transpose()
}

/// A frequencies.txt row's window, for diagnostics.
fn describe_window(row: &RawFrequency) -> String {
    format!(
        "row {}–{} every {} s",
        clock(row.start_time),
        clock(row.end_time),
        row.headway_secs
    )
}

/// Seconds after midnight as `HH:MM:SS` (hours may exceed 24).
fn clock(seconds: u32) -> String {
    format!(
        "{:02}:{:02}:{:02}",
        seconds / 3600,
        seconds % 3600 / 60,
        seconds % 60
    )
}

#[cfg(test)]
#[path = "read_tests.rs"]
mod tests;
