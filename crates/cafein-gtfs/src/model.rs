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
/// the feed they came from; the pair `(feed, id)` is unique across the merge.
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
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Trip {
    pub feed: FeedIndex,
    pub id: String,
    pub route: RouteIndex,
    pub service_id: String,
    pub direction_id: Option<u8>,
    pub shape_id: Option<String>,
    pub headsign: Option<String>,
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
