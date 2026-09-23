//! In-memory representation of one or several merged GTFS feeds.

use chrono::NaiveDate;

pub use gtfs_structures::{Exception, RouteType};

/// Index of a source feed within a merged [`Feed`].
pub type FeedIndex = u16;

/// Index into [`Feed::stops`].
pub type StopIndex = u32;

/// Index into [`Feed::routes`].
pub type RouteIndex = u32;

/// One or several GTFS feeds merged into flat, index-linked tables.
///
/// Entities keep their original GTFS identifiers together with the index of
/// the feed they came from; the pair `(feed, id)` is unique across the merge,
/// except that the runs expanded from one `frequencies.txt` template share
/// their template's trip id.
/// Cross-references between tables are resolved to vector indices at read
/// time, so lookups never go through string identifiers.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Feed {
    pub agencies: Vec<Agency>,
    pub stops: Vec<Stop>,
    pub routes: Vec<Route>,
    pub trips: Vec<Trip>,
    pub calendars: Vec<Calendar>,
    pub calendar_dates: Vec<CalendarDate>,
    pub feed_infos: Vec<FeedInfo>,
    /// Number of source feeds merged into this one.
    pub feed_count: FeedIndex,
    /// Optional tables dropped at read time: a diagnostic, never
    /// persisted.
    #[serde(skip)]
    pub skipped_files: Vec<SkippedFile>,
    /// frequencies.txt rows that could not be expanded, and templates
    /// omitted for lack of an expandable row: a diagnostic, never
    /// persisted.
    #[serde(skip)]
    pub skipped_frequencies: Vec<SkippedFrequency>,
    /// Rows dropped at read time from the tables routing consumes, with
    /// the trips or services each drop took along: a diagnostic, never
    /// persisted.
    #[serde(skip)]
    pub dropped_rows: Vec<DroppedRows>,
}

/// An optional GTFS table that failed to parse and was dropped because
/// routing never consults it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkippedFile {
    pub feed: FeedIndex,
    pub file_name: String,
    /// The parse error and its causes.
    pub reason: String,
}

/// The rows of one table that failed to parse and were dropped, with
/// what the drop took along.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DroppedRows {
    pub feed: FeedIndex,
    pub file_name: String,
    pub rows: u32,
    /// The first dropped row's line as the CSV reader counts lines
    /// (blank lines are skipped uncounted, as in the parser's own
    /// errors), and why it failed.
    pub first_line: u64,
    pub first_reason: String,
    /// Trips removed because a dropped row belonged to them.
    pub trips_dropped: u32,
    /// Services that lost a calendar row.
    pub services_affected: u32,
}

/// A frequencies.txt row that could not be expanded into runs, or a
/// template trip omitted because none of its rows could.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkippedFrequency {
    pub feed: FeedIndex,
    pub trip_id: String,
    pub reason: String,
}

/// A transit agency (`agency.txt`).
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Agency {
    pub feed: FeedIndex,
    pub id: Option<String>,
    pub name: String,
    pub timezone: String,
}

/// A stop or station (`stops.txt`).
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Stop {
    pub feed: FeedIndex,
    pub id: String,
    pub code: Option<String>,
    pub name: Option<String>,
    pub latitude: Option<f64>,
    pub longitude: Option<f64>,
    pub parent_station: Option<String>,
    /// GTFS ``wheelchair_boarding``: ``Some(true)`` = accessible,
    /// ``Some(false)`` = not accessible, ``None`` = the feed says
    /// nothing. A stop without a value inherits its parent station's
    /// known value at ingest.
    pub wheelchair_boarding: Option<bool>,
}

/// A route (`routes.txt`).
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Route {
    pub feed: FeedIndex,
    pub id: String,
    pub short_name: Option<String>,
    pub long_name: Option<String>,
    pub route_type: RouteType,
    pub agency_id: Option<String>,
}

/// A trip (`trips.txt`) with its scheduled calls (`stop_times.txt`),
/// ordered by `stop_sequence`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Trip {
    pub feed: FeedIndex,
    pub id: String,
    pub route: RouteIndex,
    pub service_id: String,
    pub direction_id: Option<u8>,
    pub shape_id: Option<String>,
    pub headsign: Option<String>,
    /// GTFS ``bikes_allowed``: ``Some(true)`` = bicycles allowed,
    /// ``Some(false)`` = forbidden, ``None`` = the feed says nothing.
    pub bikes_allowed: Option<bool>,
    /// GTFS ``wheelchair_accessible``: ``Some(true)`` = accessible,
    /// ``Some(false)`` = not accessible, ``None`` = the feed says
    /// nothing.
    pub wheelchair_accessible: Option<bool>,
    pub stop_times: Vec<StopTime>,
}

/// A scheduled call at a stop.
///
/// Times are seconds past the start of the service day. GTFS over-midnight
/// times (`25:30:00`) stay above 86 400 seconds on their original service
/// day instead of wrapping around.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct StopTime {
    pub stop: StopIndex,
    pub arrival: Option<u32>,
    pub departure: Option<u32>,
    pub stop_sequence: u32,
    pub shape_dist_traveled: Option<f32>,
}

/// A weekly service pattern (`calendar.txt`).
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Calendar {
    pub feed: FeedIndex,
    pub service_id: String,
    /// Monday through Sunday.
    pub weekdays: [bool; 7],
    pub start_date: NaiveDate,
    pub end_date: NaiveDate,
}

/// A dated exception to a service pattern (`calendar_dates.txt`).
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct CalendarDate {
    pub feed: FeedIndex,
    pub service_id: String,
    pub date: NaiveDate,
    pub exception: Exception,
}

/// Feed metadata (`feed_info.txt`).
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct FeedInfo {
    pub feed: FeedIndex,
    pub publisher_name: String,
    pub version: Option<String>,
}
