//! GTFS ingest and timetable construction for cafein.
//!
//! Reads one or several GTFS feeds into a [`Feed`]: flat, index-linked
//! tables in which every entity carries the index of its source feed, so
//! identifiers from different agencies cannot collide.

mod model;
mod read;
mod timetable;

pub use model::{
    Agency, Calendar, CalendarDate, Exception, Feed, FeedIndex, FeedInfo, Route, RouteIndex,
    RouteType, Stop, StopIndex, StopTime, Trip,
};
pub use timetable::{build_timetable, QuarantinedTrip, TimetableBuild};

/// Errors raised while reading or merging GTFS feeds.
#[derive(Debug)]
pub enum Error {
    /// The GTFS archive or directory could not be read or parsed.
    Gtfs(gtfs_structures::Error),
    /// A trip references a route that is missing from `routes.txt`.
    UnknownRoute { trip_id: String, route_id: String },
    /// A stop time references a stop that is missing from `stops.txt`.
    UnknownStop { trip_id: String, stop_id: String },
    /// A stop time has neither an arrival nor a departure time.
    MissingStopTime { trip_id: String },
    /// The timetable could not be assembled.
    Timetable(cafein_core::timetable::TimetableError),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Gtfs(error) => write!(f, "could not read GTFS feed: {error}"),
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
            Error::Timetable(error) => write!(f, "could not assemble timetable: {error}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Gtfs(error) => Some(error),
            Error::Timetable(error) => Some(error),
            _ => None,
        }
    }
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
