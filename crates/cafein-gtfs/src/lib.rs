//! GTFS ingest and timetable construction for cafein.
//!
//! Reads one or several GTFS feeds into a [`Feed`]: flat, index-linked
//! tables in which every entity carries the index of its source feed, so
//! identifiers from different agencies cannot collide.

mod model;
mod qa;
mod read;
mod service;
mod timetable;

pub use model::{
    Agency, BlankDefaults, Calendar, CalendarDate, DroppedRows, Exception, Feed, FeedIndex,
    FeedInfo, FrequencyNote, FrequencyRepair, Route, RouteIndex, RouteType, SkippedFile,
    SkippedFrequency, SkippedTableKind, Stop, StopIndex, StopTime, Trip,
};
pub use qa::{validate_feed, QaFinding};
pub use service::{ServiceCalendar, ServiceIndex};
pub use timetable::{build_timetable, QuarantinedTrip, TimetableBuild};

/// Errors raised while reading or merging GTFS feeds.
#[derive(Debug)]
pub enum Error {
    /// The GTFS archive or directory could not be read or parsed.
    Gtfs(gtfs_structures::Error),
    /// One feed of a read could not be read or parsed: [`Error::Gtfs`]
    /// with the feed's path.
    FeedRead {
        path: String,
        error: gtfs_structures::Error,
    },
    /// A trip references a route that is missing from `routes.txt`.
    UnknownRoute { trip_id: String, route_id: String },
    /// A stop time references a stop that is missing from `stops.txt`.
    UnknownStop { trip_id: String, stop_id: String },
    /// A stop time has neither an arrival nor a departure time.
    MissingStopTime { trip_id: String },
    /// A trip has fewer than two stop times, so it can never be ridden.
    TooFewStopTimes { trip_id: String, count: usize },
    /// Expanding frequencies.txt would create more stop times than the
    /// feed-wide ceiling allows.
    FrequencyExpansionTooLarge { trip_id: String, limit: u64 },
    /// The timetable could not be assembled.
    Timetable(cafein_core::timetable::TimetableError),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Gtfs(error) => write!(f, "could not read GTFS feed: {}", read_problem(error)),
            Error::FeedRead { path, error } => {
                write!(
                    f,
                    "could not read GTFS feed '{path}': {}",
                    read_problem(error)
                )
            }
            Error::UnknownRoute { trip_id, route_id } => {
                write!(f, "trip '{trip_id}' references unknown route '{route_id}'")
            }
            Error::UnknownStop { trip_id, stop_id } => {
                write!(f, "trip '{trip_id}' references unknown stop '{stop_id}'")
            }
            Error::MissingStopTime { trip_id } => {
                write!(
                    f,
                    "trip '{trip_id}' has a stop time without arrival and departure"
                )
            }
            Error::TooFewStopTimes { trip_id, count } => write!(
                f,
                "trip '{trip_id}' has {count} stop time(s); a trip needs at least two"
            ),
            Error::FrequencyExpansionTooLarge { trip_id, limit } => write!(
                f,
                "expanding frequencies.txt would create more than {limit} stop times \
                 (reached at trip '{trip_id}')"
            ),
            Error::Timetable(error) => write!(f, "could not assemble timetable: {error}"),
        }
    }
}

/// The parser's error in the reader's words: a missing required file is
/// named as such.
fn read_problem(error: &gtfs_structures::Error) -> String {
    match error {
        gtfs_structures::Error::MissingFile(name) => format!("required file {name} is missing"),
        other => other.to_string(),
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            // A missing file is stated in full above; the parser's own
            // wording adds nothing as a cause.
            Error::Gtfs(error) | Error::FeedRead { error, .. } => match error {
                gtfs_structures::Error::MissingFile(_) => None,
                other => Some(other),
            },
            Error::Timetable(error) => Some(error),
            _ => None,
        }
    }
}

/// An error's message followed by those of its causes, each added only
/// when the text so far does not already state it.
pub fn error_chain(error: &(dyn std::error::Error + 'static)) -> String {
    let mut message = error.to_string();
    let mut cause = error.source();
    while let Some(error) = cause {
        let text = error.to_string();
        if !message.contains(&text) {
            message.push_str(": ");
            message.push_str(&text);
        }
        cause = error.source();
    }
    message
}

impl From<gtfs_structures::Error> for Error {
    fn from(error: gtfs_structures::Error) -> Self {
        Error::Gtfs(error)
    }
}

impl From<cafein_core::timetable::TimetableError> for Error {
    fn from(error: cafein_core::timetable::TimetableError) -> Self {
        Error::Timetable(error)
    }
}
