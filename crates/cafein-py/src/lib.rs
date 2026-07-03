//! Python bindings for cafein.

use std::collections::HashMap;

use chrono::NaiveDate;
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use cafein_core::geometry::{DistanceProvenance, TripGeometry};
use cafein_core::journey::{Journey, Leg};
use cafein_core::raptor::Raptor;
use cafein_core::router::{Request, TransitRouter};
use cafein_core::timetable::{StopIdx, TripIdx};
use cafein_core::transfers::Transfers;
use cafein_gtfs::{build_timetable, Feed, RouteType, TimetableBuild};

/// A routable public-transport network built from GTFS data.
#[pyclass]
struct TransportNetwork {
    feed: Feed,
    build: TimetableBuild,
    transfers: Transfers,
    geometry: Option<TripGeometry>,
    stops_by_id: HashMap<String, StopLookup>,
    stops_by_qualified_id: HashMap<String, StopIdx>,
    trips_by_public_id: HashMap<String, TripIdx>,
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
        let mut trips_by_public_id = HashMap::with_capacity(build.timetable.trip_count() as usize);
        for index in 0..build.timetable.trip_count() {
            let trip = TripIdx(index);
            let source = &feed.trips[build.timetable.trip_source(trip) as usize];
            let public = if feed.feed_count > 1 {
                format!("{}:{}", source.feed, source.id)
            } else {
                source.id.clone()
            };
            trips_by_public_id.insert(public, trip);
        }
        Ok(TransportNetwork {
            feed,
            build,
            transfers,
            geometry: None,
            stops_by_id,
            stops_by_qualified_id,
            trips_by_public_id,
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

    /// Install per-trip cumulative travel distances.
    ///
    /// Parameters
    /// ----------
    /// distances : list of (str, list of float, str)
    ///     ``(trip_id, cumulative_meters, provenance)`` rows with one
    ///     non-decreasing cumulative distance per stop of the trip, and
    ///     the provenance tier as one of ``shape_dist``, ``shape_linref``,
    ///     ``osm_relation``, ``map_matched``, ``crow_fly``. Trip
    ///     identifiers follow the public convention (feed-qualified when
    ///     several feeds are merged); rows for trips absent from the
    ///     timetable — e.g. quarantined ones — are ignored. Every
    ///     timetable trip must be covered.
    ///     ``cafein.geometry.trip_distances`` produces such lists.
    fn set_trip_distances(&mut self, distances: Vec<(String, Vec<f64>, String)>) -> PyResult<()> {
        let mut entries = Vec::with_capacity(distances.len());
        for (trip_id, cumulative, provenance) in &distances {
            let Some(&trip) = self.trips_by_public_id.get(trip_id) else {
                continue;
            };
            let cumulative: Vec<f32> = cumulative.iter().map(|&value| value as f32).collect();
            entries.push((trip, cumulative, parse_provenance(provenance)?));
        }
        self.geometry = Some(
            TripGeometry::from_trips(&self.build.timetable, entries)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        );
        Ok(())
    }

    /// The public identifiers of the network's routable trips.
    #[getter]
    fn trip_ids(&self) -> Vec<String> {
        self.trips_by_public_id.keys().cloned().collect()
    }

    /// The network's routes as `(route_id, agency_id, route_type)`
    /// tuples, with identifiers in their public form (feed-qualified
    /// when several feeds are merged) and the GTFS route_type as its
    /// numeric code. A route without an explicit agency in a
    /// single-agency feed carries that feed's one agency.
    #[getter]
    fn routes(&self) -> Vec<(String, Option<String>, i32)> {
        self.feed
            .routes
            .iter()
            .map(|route| {
                let agency_id = route.agency_id.clone().or_else(|| {
                    let mut in_feed = self
                        .feed
                        .agencies
                        .iter()
                        .filter(|agency| agency.feed == route.feed);
                    match (in_feed.next(), in_feed.next()) {
                        (Some(only), None) => only.id.clone(),
                        _ => None,
                    }
                });
                (
                    self.public_id(route.feed, &route.id),
                    agency_id.map(|id| self.public_id(route.feed, &id)),
                    route_type_code(&route.route_type),
                )
            })
            .collect()
    }

    /// Number of trips per distance-provenance tier, empty before
    /// ``set_trip_distances``.
    #[getter]
    fn distance_provenance_counts(&self) -> HashMap<&'static str, u32> {
        let mut counts = HashMap::new();
        if let Some(geometry) = &self.geometry {
            for index in 0..self.build.timetable.trip_count() {
                let name = provenance_name(geometry.provenance(TripIdx(index)));
                *counts.entry(name).or_insert(0) += 1;
            }
        }
        counts
    }

    /// Route between two transit stops for a single departure.
    ///
    /// Journeys ride trips and change vehicles at shared stops or over
    /// the transfers installed with ``set_transfers``; transit legs
    /// report their distance and its provenance when trip distances are
    /// installed. Door-to-door access/egress from arbitrary coordinates
    /// joins once the query-time street search exists; leg geometries
    /// and emissions join once the geometry preprocessing produces them.
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
    /// window : int (optional)
    ///     Departure window in seconds. When given, departures within
    ///     ``[departure, departure + window)`` are profiled: the result is
    ///     the Pareto set of journeys over (departure, arrival, rides),
    ///     each journey's departure being the latest time the origin can
    ///     be left to catch it, sorted by departure and then rides. A
    ///     journey that leaves within the window but waits for a ride
    ///     beyond it carries the window's final second as its departure.
    ///
    /// Returns
    /// -------
    /// list of dict
    ///     Without `window`, the Pareto set of journeys over (arrival
    ///     time, number of rides) leaving at the departure time; with it,
    ///     the departure-window profile. Each journey carries its legs;
    ///     times are seconds past the service day's start.
    #[pyo3(signature = (from_stop, to_stop, date, departure, max_transfers = 4, window = None))]
    #[allow(clippy::too_many_arguments)]
    fn route_between_stops(
        &self,
        py: Python<'_>,
        from_stop: &str,
        to_stop: &str,
        date: &str,
        departure: &str,
        max_transfers: u8,
        window: Option<u32>,
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
        let journeys = match window {
            None => Raptor.route(&self.build.timetable, &self.transfers, &request),
            Some(window) => {
                Raptor.route_range(&self.build.timetable, &self.transfers, &request, window)
            }
        };
        let result = PyList::empty(py);
        for journey in &journeys {
            result.append(self.journey_to_dict(py, journey)?)?;
        }
        Ok(result.unbind())
    }

    /// Earliest arrival at every reachable stop for a single departure.
    ///
    /// One RAPTOR run serves all destinations, so travel-time matrices
    /// are assembled origin by origin from this method — never per OD
    /// pair.
    ///
    /// Parameters
    /// ----------
    /// from_stop : str
    ///     GTFS stop_id of the origin stop; ``<feed_index>:<stop_id>``
    ///     when the id occurs in several merged feeds.
    /// date : str
    ///     Service date as ``YYYY-MM-DD``.
    /// departure : str
    ///     Departure time at the origin as ``HH:MM:SS``.
    /// max_transfers : int (optional, default: 4)
    ///     Maximum number of transfers between rides.
    ///
    /// Returns
    /// -------
    /// dict
    ///     Travel time in seconds to every reachable stop, keyed by
    ///     public stop_id; the origin maps to 0 and unreachable stops
    ///     are absent.
    #[pyo3(signature = (from_stop, date, departure, max_transfers = 4))]
    fn travel_times_from_stop(
        &self,
        py: Python<'_>,
        from_stop: &str,
        date: &str,
        departure: &str,
        max_transfers: u8,
    ) -> PyResult<Py<PyDict>> {
        let origin = self.resolve_stop(from_stop)?;
        let date = NaiveDate::parse_from_str(date, "%Y-%m-%d")
            .map_err(|error| PyValueError::new_err(format!("invalid date '{date}': {error}")))?;
        let departure = parse_time(departure)?;
        let request = Request {
            departure,
            access: vec![(origin, 0)],
            egress: Vec::new(),
            active_services: self.build.services.active_on(date),
            max_transfers,
        };
        let arrivals = Raptor.one_to_all(&self.build.timetable, &self.transfers, &request);
        let result = PyDict::new(py);
        for (index, arrival) in arrivals.iter().enumerate() {
            if let Some(arrival) = arrival {
                result.set_item(
                    self.public_stop_id(StopIdx(index as u32)),
                    arrival - departure,
                )?;
            }
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
                    entry.set_item("distance", py.None())?;
                    entry.set_item("distance_provenance", py.None())?;
                }
                Leg::Transit {
                    trip,
                    board_stop,
                    alight_stop,
                    board_position,
                    alight_position,
                    board_time,
                    alight_time,
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
                    match &self.geometry {
                        Some(geometry) => {
                            entry.set_item(
                                "distance",
                                geometry.leg_distance(trip, board_position, alight_position) as f64,
                            )?;
                            entry.set_item(
                                "distance_provenance",
                                provenance_name(geometry.provenance(trip)),
                            )?;
                        }
                        None => {
                            entry.set_item("distance", py.None())?;
                            entry.set_item("distance_provenance", py.None())?;
                        }
                    }
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
                    entry.set_item("distance", py.None())?;
                    entry.set_item("distance_provenance", py.None())?;
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
                    entry.set_item("distance", py.None())?;
                    entry.set_item("distance_provenance", py.None())?;
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

/// The numeric GTFS route_type of a parsed route type; named variants map
/// to their standard codes, extended codes pass through.
fn route_type_code(route_type: &RouteType) -> i32 {
    match route_type {
        RouteType::Tramway => 0,
        RouteType::Subway => 1,
        RouteType::Rail => 2,
        RouteType::Bus => 3,
        RouteType::Ferry => 4,
        RouteType::CableCar => 5,
        RouteType::Gondola => 6,
        RouteType::Funicular => 7,
        RouteType::Coach => 200,
        RouteType::Air => 1100,
        RouteType::Taxi => 1500,
        RouteType::Other(code) => *code as i32,
    }
}

fn provenance_name(tier: DistanceProvenance) -> &'static str {
    match tier {
        DistanceProvenance::ShapeDist => "shape_dist",
        DistanceProvenance::ShapeLinRef => "shape_linref",
        DistanceProvenance::OsmRelation => "osm_relation",
        DistanceProvenance::MapMatched => "map_matched",
        DistanceProvenance::CrowFly => "crow_fly",
    }
}

fn parse_provenance(value: &str) -> PyResult<DistanceProvenance> {
    match value {
        "shape_dist" => Ok(DistanceProvenance::ShapeDist),
        "shape_linref" => Ok(DistanceProvenance::ShapeLinRef),
        "osm_relation" => Ok(DistanceProvenance::OsmRelation),
        "map_matched" => Ok(DistanceProvenance::MapMatched),
        "crow_fly" => Ok(DistanceProvenance::CrowFly),
        other => Err(PyValueError::new_err(format!(
            "unknown distance provenance '{other}'"
        ))),
    }
}

#[pymodule]
fn _cafein(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<TransportNetwork>()?;
    Ok(())
}
