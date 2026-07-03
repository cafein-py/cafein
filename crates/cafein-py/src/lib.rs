//! Python bindings for cafein.

use std::collections::HashMap;

use chrono::NaiveDate;
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use cafein_core::journey::{Journey, Leg};
use cafein_core::raptor::Raptor;
use cafein_core::router::{Request, TransitRouter};
use cafein_core::timetable::StopIdx;
use cafein_core::transfers::Transfers;
use cafein_gtfs::{build_timetable, Feed, TimetableBuild};

/// A routable public-transport network built from GTFS data.
#[pyclass]
struct TransportNetwork {
    feed: Feed,
    build: TimetableBuild,
    transfers: Transfers,
    stops_by_id: HashMap<String, StopLookup>,
    stops_by_qualified_id: HashMap<String, StopIdx>,
}

/// Resolution of a raw GTFS stop_id, which merged feeds can duplicate.
#[derive(Clone, Copy)]
enum StopLookup {
    Unique(StopIdx),
    Ambiguous,
}

#[pymethods]
impl TransportNetwork {
    /// Build a network from one or several GTFS zip archives.
    ///
    /// Parameters
    /// ----------
    /// paths : list of str
    ///     Paths to GTFS zip files or directories. Several feeds are
    ///     merged; a stop_id occurring in more than one feed must then be
    ///     qualified as ``<feed_index>:<stop_id>``, with feeds numbered in
    ///     input order.
    #[staticmethod]
    fn from_gtfs(py: Python<'_>, paths: Vec<String>) -> PyResult<TransportNetwork> {
        let feed = Feed::from_paths(&paths).map_err(to_py_error)?;
        let build = build_timetable(&feed).map_err(to_py_error)?;
        if !build.quarantined.is_empty() {
            let message = format!(
                "quarantined {} trip(s) with data-quality problems; routing excludes them",
                build.quarantined.len()
            );
            let warnings = py.import("warnings")?;
            warnings.call_method1(
                "warn",
                (message, py.get_type::<pyo3::exceptions::PyUserWarning>(), 2),
            )?;
        }
        let transfers = Transfers::empty(build.timetable.stop_count());
        let mut stops_by_id = HashMap::with_capacity(feed.stops.len());
        let mut stops_by_qualified_id = HashMap::with_capacity(feed.stops.len());
        for (index, stop) in feed.stops.iter().enumerate() {
            let stop_index = StopIdx(index as u32);
            stops_by_qualified_id.insert(format!("{}:{}", stop.feed, stop.id), stop_index);
            stops_by_id
                .entry(stop.id.clone())
                .and_modify(|entry| *entry = StopLookup::Ambiguous)
                .or_insert(StopLookup::Unique(stop_index));
        }
        Ok(TransportNetwork {
            feed,
            build,
            transfers,
            stops_by_id,
            stops_by_qualified_id,
        })
    }

    /// Number of stops in the network.
    #[getter]
    fn stop_count(&self) -> u32 {
        self.build.timetable.stop_count()
    }

    /// Number of stop-sequence patterns in the network.
    #[getter]
    fn pattern_count(&self) -> u32 {
        self.build.timetable.pattern_count()
    }

    /// Number of trips in the network.
    #[getter]
    fn trip_count(&self) -> u32 {
        self.build.timetable.trip_count()
    }

    /// Number of installed stop-to-stop transfers.
    #[getter]
    fn transfer_count(&self) -> usize {
        self.transfers.edge_count()
    }

    /// The network's stops as `(stop_id, latitude, longitude)` tuples,
    /// with identifiers in their public form (feed-qualified when several
    /// feeds are merged) and coordinates `None` where the feed has none.
    #[getter]
    fn stops(&self) -> Vec<(String, Option<f64>, Option<f64>)> {
        self.feed
            .stops
            .iter()
            .enumerate()
            .map(|(index, stop)| {
                (
                    self.public_stop_id(StopIdx(index as u32)),
                    stop.latitude,
                    stop.longitude,
                )
            })
            .collect()
    }

    /// Install precomputed stop-to-stop transfers (footpaths).
    ///
    /// Parameters
    /// ----------
    /// footpaths : list of (str, str, int)
    ///     ``(from_stop, to_stop, seconds)`` walking edges, with stop
    ///     identifiers as in ``route_between_stops``. The edge list must
    ///     be transitively closed — routing relaxes a single transfer hop
    ///     per round; ``cafein.streets.walking_footpaths`` produces such
    ///     lists.
    fn set_transfers(&mut self, footpaths: Vec<(String, String, u32)>) -> PyResult<()> {
        let mut edges = Vec::with_capacity(footpaths.len());
        for (from, to, duration) in &footpaths {
            edges.push((self.resolve_stop(from)?, self.resolve_stop(to)?, *duration));
        }
        self.transfers = Transfers::from_edges(self.build.timetable.stop_count(), &edges)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(())
    }

    /// Route between two transit stops for a single departure.
    ///
    /// Journeys ride trips and change vehicles at shared stops or over
    /// the transfers installed with ``set_transfers``. Door-to-door
    /// access/egress from arbitrary coordinates joins once the query-time
    /// street search exists; per-leg distance, distance provenance,
    /// geometry, and emissions join once the geometry preprocessing
    /// produces them.
    ///
    /// Parameters
    /// ----------
    /// from_stop : str
    ///     GTFS stop_id of the origin stop; ``<feed_index>:<stop_id>``
    ///     when the id occurs in several merged feeds.
    /// to_stop : str
    ///     GTFS stop_id of the destination stop, qualified the same way.
    ///     Identifiers in the output follow the same convention: raw GTFS
    ///     ids for a single feed, feed-qualified ids for merged feeds.
    /// date : str
    ///     Service date as ``YYYY-MM-DD``.
    /// departure : str
    ///     Departure time at the origin as ``HH:MM:SS``.
    /// max_transfers : int (optional, default: 4)
    ///     Maximum number of transfers between rides.
    ///
    /// Returns
    /// -------
    /// list of dict
    ///     The Pareto set of journeys over (arrival time, number of rides),
    ///     each with its legs. Times are seconds past the service day's
    ///     start.
    #[pyo3(signature = (from_stop, to_stop, date, departure, max_transfers = 4))]
    fn route_between_stops(
        &self,
        py: Python<'_>,
        from_stop: &str,
        to_stop: &str,
        date: &str,
        departure: &str,
        max_transfers: u8,
    ) -> PyResult<Py<PyList>> {
        let origin = self.resolve_stop(from_stop)?;
        let destination = self.resolve_stop(to_stop)?;
        let date = NaiveDate::parse_from_str(date, "%Y-%m-%d")
            .map_err(|error| PyValueError::new_err(format!("invalid date '{date}': {error}")))?;
        let request = Request {
            departure: parse_time(departure)?,
            access: vec![(origin, 0)],
            egress: vec![(destination, 0)],
            active_services: self.build.services.active_on(date),
            max_transfers,
        };
        let journeys = Raptor.route(&self.build.timetable, &self.transfers, &request);
        let result = PyList::empty(py);
        for journey in &journeys {
            result.append(self.journey_to_dict(py, journey)?)?;
        }
        Ok(result.unbind())
    }
}

impl TransportNetwork {
    /// The public form of a stop identifier: raw for a single feed,
    /// `<feed_index>:<id>` when several feeds are merged.
    fn public_stop_id(&self, stop: StopIdx) -> String {
        let stop = &self.feed.stops[stop.0 as usize];
        self.public_id(stop.feed, &stop.id)
    }

    fn public_id(&self, feed: cafein_gtfs::FeedIndex, id: &str) -> String {
        if self.feed.feed_count > 1 {
            format!("{feed}:{id}")
        } else {
            id.to_owned()
        }
    }

    /// Resolves a stop identifier. In merged networks the feed-qualified
    /// form (`<feed_index>:<stop_id>`) takes precedence, so a raw stop_id
    /// that happens to look like another stop's qualified id must itself
    /// be fully qualified.
    fn resolve_stop(&self, stop_id: &str) -> PyResult<StopIdx> {
        if self.feed.feed_count > 1 {
            if let Some(&stop) = self.stops_by_qualified_id.get(stop_id) {
                return Ok(stop);
            }
        }
        match self.stops_by_id.get(stop_id) {
            Some(StopLookup::Unique(stop)) => Ok(*stop),
            Some(StopLookup::Ambiguous) => Err(PyKeyError::new_err(format!(
                "stop_id '{stop_id}' occurs in several feeds; qualify it as '<feed_index>:{stop_id}'"
            ))),
            None => Err(PyKeyError::new_err(format!("unknown stop_id '{stop_id}'"))),
        }
    }

    fn journey_to_dict(&self, py: Python<'_>, journey: &Journey) -> PyResult<Py<PyDict>> {
        let timetable = &self.build.timetable;
        let dict = PyDict::new(py);
        dict.set_item("departure", journey.departure)?;
        dict.set_item("arrival", journey.arrival)?;
        dict.set_item("rides", journey.rides())?;
        let legs = PyList::empty(py);
        for leg in &journey.legs {
            let entry = PyDict::new(py);
            match *leg {
                Leg::Access {
                    to_stop,
                    departure,
                    arrival,
                } => {
                    entry.set_item("type", "access")?;
                    entry.set_item("to_stop", self.public_stop_id(to_stop))?;
                    entry.set_item("departure", departure)?;
                    entry.set_item("arrival", arrival)?;
                }
                Leg::Transit {
                    trip,
                    board_stop,
                    alight_stop,
                    board_time,
                    alight_time,
                    ..
                } => {
                    let source_trip = &self.feed.trips[timetable.trip_source(trip) as usize];
                    let route = &self.feed.routes[source_trip.route as usize];
                    entry.set_item("type", "transit")?;
                    entry.set_item("trip_id", self.public_id(source_trip.feed, &source_trip.id))?;
                    entry.set_item("route_id", self.public_id(route.feed, &route.id))?;
                    entry.set_item("route_short_name", route.short_name.as_deref())?;
                    entry.set_item("board_stop", self.public_stop_id(board_stop))?;
                    entry.set_item("alight_stop", self.public_stop_id(alight_stop))?;
                    entry.set_item("departure", board_time)?;
                    entry.set_item("arrival", alight_time)?;
                }
                Leg::Transfer {
                    from_stop,
                    to_stop,
                    departure,
                    arrival,
                } => {
                    entry.set_item("type", "transfer")?;
                    entry.set_item("from_stop", self.public_stop_id(from_stop))?;
                    entry.set_item("to_stop", self.public_stop_id(to_stop))?;
                    entry.set_item("departure", departure)?;
                    entry.set_item("arrival", arrival)?;
                }
                Leg::Egress {
                    from_stop,
                    departure,
                    arrival,
                } => {
                    entry.set_item("type", "egress")?;
                    entry.set_item("from_stop", self.public_stop_id(from_stop))?;
                    entry.set_item("departure", departure)?;
                    entry.set_item("arrival", arrival)?;
                }
            }
            legs.append(entry)?;
        }
        dict.set_item("legs", legs)?;
        Ok(dict.unbind())
    }
}

/// Parses ``HH:MM:SS`` into seconds past the service day's start; hours may
/// exceed 23 for over-midnight times, following GTFS.
fn parse_time(value: &str) -> PyResult<u32> {
    let parts: Vec<&str> = value.split(':').collect();
    let invalid = || PyValueError::new_err(format!("invalid time '{value}': expected HH:MM:SS"));
    if parts.len() != 3 {
        return Err(invalid());
    }
    let hours: u32 = parts[0].parse().map_err(|_| invalid())?;
    let minutes: u32 = parts[1].parse().map_err(|_| invalid())?;
    let seconds: u32 = parts[2].parse().map_err(|_| invalid())?;
    if minutes > 59 || seconds > 59 {
        return Err(invalid());
    }
    hours
        .checked_mul(3600)
        .and_then(|in_seconds| in_seconds.checked_add(minutes * 60 + seconds))
        .ok_or_else(invalid)
}

fn to_py_error(error: cafein_gtfs::Error) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pymodule]
fn _cafein(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<TransportNetwork>()?;
    Ok(())
}
