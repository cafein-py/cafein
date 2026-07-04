//! Python bindings for cafein.

use std::collections::HashMap;

use chrono::NaiveDate;
use numpy::{IntoPyArray, PyArray2, PyArray3, PyArrayMethods};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use cafein_core::geometry::{DistanceProvenance, LegGeometry, TripGeometry};
use cafein_core::journey::{Journey, Leg};
use cafein_core::raptor::{CostInputs, Raptor};
use cafein_core::router::{Request, TransitRouter};
use cafein_core::streets::{StopLink, StreetNetwork, WalkedStop};
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
    leg_geometry: Option<LegGeometry>,
    streets: Option<StreetNetwork>,
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

/// A coordinate query's exact walk lengths in meters, keyed by the stop
/// each access or egress link enters the network through.
struct WalkMaps {
    access: HashMap<StopIdx, f64>,
    egress: HashMap<StopIdx, f64>,
}

impl WalkMaps {
    fn new(access: &[WalkedStop], egress: &[WalkedStop]) -> WalkMaps {
        let meters =
            |walks: &[WalkedStop]| walks.iter().map(|walk| (walk.stop, walk.meters)).collect();
        WalkMaps {
            access: meters(access),
            egress: meters(egress),
        }
    }
}

/// The `(stop, seconds)` request offsets of a walking-search result.
fn request_offsets(walks: &[WalkedStop]) -> Vec<(StopIdx, u32)> {
    walks.iter().map(|walk| (walk.stop, walk.seconds)).collect()
}

/// Rejects an empty or out-of-range window/percentile specification.
fn validate_window(window: u32, percentiles: &[f64]) -> PyResult<()> {
    if window == 0 {
        return Err(PyValueError::new_err("window must be at least 1 second"));
    }
    if percentiles.is_empty() {
        return Err(PyValueError::new_err("percentiles must not be empty"));
    }
    for &percentile in percentiles {
        if !percentile.is_finite() || !(0.0..=100.0).contains(&percentile) {
            return Err(PyValueError::new_err(
                "percentiles must be finite and within [0, 100]",
            ));
        }
    }
    Ok(())
}

/// Rejects non-finite point coordinates with the offending index.
fn validate_points(points: &[(f64, f64)]) -> PyResult<()> {
    for (index, &(lat, lon)) in points.iter().enumerate() {
        if !lat.is_finite() || !lon.is_finite() {
            return Err(PyValueError::new_err(format!(
                "point {index} has non-finite coordinates"
            )));
        }
    }
    Ok(())
}

/// The indices of points the linking could not snap.
fn unsnapped(links: &[Option<Vec<WalkedStop>>]) -> Vec<u32> {
    links
        .iter()
        .enumerate()
        .filter_map(|(index, links)| links.is_none().then_some(index as u32))
        .collect()
}

/// Each destination point's `(stop, seconds, meters)` egress table;
/// unsnapped points get an empty table and stay unreachable.
fn egress_tables(links: &[Option<Vec<WalkedStop>>]) -> Vec<Vec<(StopIdx, u32, f64)>> {
    links
        .iter()
        .map(|links| {
            links
                .as_deref()
                .unwrap_or(&[])
                .iter()
                .map(|walk| (walk.stop, walk.seconds, walk.meters))
                .collect()
        })
        .collect()
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
            leg_geometry: None,
            streets: None,
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
    /// footpaths : list of (str, str, int, float)
    ///     ``(from_stop, to_stop, seconds, meters)`` walking edges, with
    ///     stop identifiers as in ``route_between_stops`` and the walked
    ///     street-path length in meters. The edge list must be
    ///     transitively closed — routing relaxes a single transfer hop
    ///     per round; ``cafein.streets.walking_footpaths`` produces such
    ///     lists.
    fn set_transfers(&mut self, footpaths: Vec<(String, String, u32, f64)>) -> PyResult<()> {
        let mut edges = Vec::with_capacity(footpaths.len());
        for (index, (from, to, duration, meters)) in footpaths.iter().enumerate() {
            if !meters.is_finite() || *meters < 0.0 {
                return Err(PyValueError::new_err(format!(
                    "footpath {index} has a negative or non-finite length"
                )));
            }
            edges.push((
                self.resolve_stop(from)?,
                self.resolve_stop(to)?,
                *duration,
                *meters,
            ));
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

    /// Install per-trip leg geometries.
    ///
    /// Parameters
    /// ----------
    /// polylines : list of (list of float, list of float, list of float)
    ///     Deduplicated ``(longitudes, latitudes, measures)`` polylines:
    ///     coordinates in EPSG:4326 with a non-decreasing measure at
    ///     every vertex (e.g. cumulative meters).
    /// trips : list of (str, int, list of float)
    ///     ``(trip_id, polyline, stop_positions)`` rows locating each
    ///     stop of the trip along its polyline, in the polyline's
    ///     measure. Trip identifiers follow the public convention; rows
    ///     for trips absent from the timetable — e.g. quarantined ones —
    ///     are ignored. Every timetable trip must be covered.
    ///     ``cafein.geometry.trip_distances(..., geometries=True)``
    ///     produces this payload.
    fn set_leg_geometries(
        &mut self,
        polylines: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)>,
        trips: Vec<(String, u32, Vec<f64>)>,
    ) -> PyResult<()> {
        let mut entries = Vec::with_capacity(trips.len());
        for (trip_id, polyline, positions) in trips {
            let Some(&trip) = self.trips_by_public_id.get(&trip_id) else {
                continue;
            };
            entries.push((trip, polyline, positions));
        }
        self.leg_geometry = Some(
            LegGeometry::new(&self.build.timetable, &polylines, entries)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        );
        Ok(())
    }

    /// Install the street network for query-time access/egress searches.
    ///
    /// Parameters
    /// ----------
    /// vertex_count : int
    ///     Number of street vertices; edges reference vertices as
    ///     indices below this count.
    /// edges : list of (int, int, float)
    ///     ``(from, to, meters)`` per walking edge (undirected), with
    ///     the edge's cost length in meters.
    /// coordinate_offsets : list of int
    ///     Offsets into the coordinate arrays, one per edge plus a tail:
    ///     edge ``i``'s geometry runs from its ``from`` vertex through
    ///     coordinates ``coordinate_offsets[i]`` up to
    ///     ``coordinate_offsets[i + 1]``.
    /// longitudes, latitudes : list of float
    ///     The flattened edge geometries, in EPSG:4326.
    /// stop_links : list of (str, int, float, float)
    ///     ``(stop_id, edge, fraction, connector_meters)`` snap records
    ///     saying how each stop enters the street graph, with stop
    ///     identifiers as in ``route_between_stops``.
    ///     ``cafein.streets.walking_streets`` produces this payload.
    fn set_street_network(
        &mut self,
        vertex_count: u32,
        edges: Vec<(u32, u32, f64)>,
        coordinate_offsets: Vec<u32>,
        longitudes: Vec<f64>,
        latitudes: Vec<f64>,
        stop_links: Vec<(String, u32, f64, f64)>,
    ) -> PyResult<()> {
        let mut links = Vec::with_capacity(stop_links.len());
        for (stop_id, edge, fraction, connector) in &stop_links {
            links.push(StopLink {
                stop: self.resolve_stop(stop_id)?,
                edge: *edge,
                fraction: *fraction,
                connector: *connector,
            });
        }
        self.streets = Some(
            StreetNetwork::new(
                vertex_count,
                self.build.timetable.stop_count(),
                &edges,
                &coordinate_offsets,
                &longitudes,
                &latitudes,
                links,
            )
            .map_err(|error| PyValueError::new_err(error.to_string()))?,
        );
        Ok(())
    }

    /// Walking times to every transit stop reachable from a coordinate.
    ///
    /// Requires an installed street network. Walking is undirected, so
    /// the same search serves access from an origin and egress to a
    /// destination.
    ///
    /// Parameters
    /// ----------
    /// lat, lon : float
    ///     The coordinate, in EPSG:4326.
    /// walking_speed_kmph : float (optional, default: 3.6)
    ///     Walking speed in km/h, on the network and on the connectors.
    /// max_walking_time : float (optional, default: 600)
    ///     Walking-time cutoff in seconds.
    /// max_snap_distance : float (optional, default: 100)
    ///     Maximum straight-line distance in meters from the coordinate
    ///     to the walking network; a coordinate farther away raises
    ///     ``ValueError``.
    ///
    /// Returns
    /// -------
    /// dict
    ///     Walking time in seconds to each reachable stop, keyed by
    ///     stop_id; stops beyond the cutoff are absent.
    #[pyo3(signature = (lat, lon, walking_speed_kmph = 3.6, max_walking_time = 600.0, max_snap_distance = 100.0))]
    fn access_stops(
        &self,
        py: Python<'_>,
        lat: f64,
        lon: f64,
        walking_speed_kmph: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
    ) -> PyResult<Py<PyDict>> {
        let streets = self.installed_streets()?;
        let speed =
            validated_walking_speed(walking_speed_kmph, max_walking_time, max_snap_distance)?;
        let reached = coordinate_links(
            streets,
            (lat, lon),
            speed,
            max_walking_time,
            max_snap_distance,
            "",
        )?;
        let result = PyDict::new(py);
        for walk in reached {
            result.set_item(self.public_stop_id(walk.stop), walk.seconds)?;
        }
        Ok(result.unbind())
    }

    /// The public identifiers of the network's routable trips.
    #[getter]
    fn trip_ids(&self) -> Vec<String> {
        self.trips_by_public_id.keys().cloned().collect()
    }

    /// The network's routable trips as `(trip_id, route_id)` tuples,
    /// with identifiers in their public form.
    #[getter]
    fn trips(&self) -> Vec<(String, String)> {
        self.trips_by_public_id
            .iter()
            .map(|(public, &trip)| {
                let source = &self.feed.trips[self.build.timetable.trip_source(trip) as usize];
                let route = &self.feed.routes[source.route as usize];
                (public.clone(), self.public_id(route.feed, &route.id))
            })
            .collect()
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
    /// installed. ``route_between_coordinates`` routes door-to-door from
    /// arbitrary coordinates. Legs carry times, stops, distances, and
    /// provenance; transit legs add their geometry as a WKB LineString
    /// when leg geometries are installed. Walk legs carry no geometry
    /// yet.
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
    #[pyo3(signature = (from_stop, to_stop, date, departure, max_transfers = 4, window = None, geometries = true))]
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
        geometries: bool,
    ) -> PyResult<Py<PyList>> {
        let origin = self.resolve_stop(from_stop)?;
        let destination = self.resolve_stop(to_stop)?;
        let request = Request {
            departure: parse_time(departure)?,
            access: vec![(origin, 0)],
            egress: vec![(destination, 0)],
            active_services: self.active_services(date)?,
            max_transfers,
        };
        self.route_request(py, &request, window, None, geometries)
    }

    /// Route door-to-door between two coordinates for a single departure.
    ///
    /// The street network installed with ``set_street_network`` provides
    /// walking access from the origin to nearby stops and egress from
    /// stops to the destination; journeys otherwise behave as in
    /// ``route_between_stops``. Access and egress legs report their
    /// walking distance in meters; a coordinate farther than
    /// ``max_snap_distance`` from the walking network raises
    /// ``ValueError``. Journeys ride at least one trip: a destination
    /// best reached by walking alone yields no journeys.
    ///
    /// Parameters
    /// ----------
    /// origin, destination : (float, float)
    ///     ``(lat, lon)`` coordinates, in EPSG:4326.
    /// date : str
    ///     Service date as ``YYYY-MM-DD``.
    /// departure : str
    ///     Departure time at the origin coordinate as ``HH:MM:SS``.
    /// max_transfers : int (optional, default: 4)
    ///     Maximum number of transfers between rides.
    /// window : int (optional)
    ///     Departure window in seconds, as in ``route_between_stops``.
    /// walking_speed_kmph : float (optional, default: 3.6)
    ///     Walking speed in km/h of the access and egress searches.
    /// max_walking_time : float (optional, default: 600)
    ///     Walking-time cutoff in seconds of each street search.
    /// max_snap_distance : float (optional, default: 100)
    ///     Maximum straight-line distance in meters from each coordinate
    ///     to the walking network.
    ///
    /// Returns
    /// -------
    /// list of dict
    ///     Journeys as in ``route_between_stops``; arrivals include the
    ///     egress walk.
    #[pyo3(signature = (origin, destination, date, departure, max_transfers = 4, window = None, walking_speed_kmph = 3.6, max_walking_time = 600.0, max_snap_distance = 100.0, geometries = true))]
    #[allow(clippy::too_many_arguments)]
    fn route_between_coordinates(
        &self,
        py: Python<'_>,
        origin: (f64, f64),
        destination: (f64, f64),
        date: &str,
        departure: &str,
        max_transfers: u8,
        window: Option<u32>,
        walking_speed_kmph: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
        geometries: bool,
    ) -> PyResult<Py<PyList>> {
        let streets = self.installed_streets()?;
        let speed =
            validated_walking_speed(walking_speed_kmph, max_walking_time, max_snap_distance)?;
        let access = coordinate_links(
            streets,
            origin,
            speed,
            max_walking_time,
            max_snap_distance,
            "origin ",
        )?;
        let egress = coordinate_links(
            streets,
            destination,
            speed,
            max_walking_time,
            max_snap_distance,
            "destination ",
        )?;
        let walks = WalkMaps::new(&access, &egress);
        let request = Request {
            departure: parse_time(departure)?,
            access: request_offsets(&access),
            egress: request_offsets(&egress),
            active_services: self.active_services(date)?,
            max_transfers,
        };
        self.route_request(py, &request, window, Some(&walks), geometries)
    }

    /// Earliest arrival at every reachable stop from a coordinate.
    ///
    /// The counterpart of ``travel_times_from_stop`` for a coordinate
    /// origin: walking access from the coordinate seeds the search, and
    /// one RAPTOR run serves all destinations. Stops within the walking
    /// cutoff appear with their walking time even without riding.
    ///
    /// Parameters
    /// ----------
    /// origin : (float, float)
    ///     ``(lat, lon)`` coordinate, in EPSG:4326.
    /// date : str
    ///     Service date as ``YYYY-MM-DD``.
    /// departure : str
    ///     Departure time at the origin coordinate as ``HH:MM:SS``.
    /// max_transfers : int (optional, default: 4)
    ///     Maximum number of transfers between rides.
    /// walking_speed_kmph : float (optional, default: 3.6)
    ///     Walking speed in km/h of the access search.
    /// max_walking_time : float (optional, default: 600)
    ///     Walking-time cutoff in seconds of the access search.
    /// max_snap_distance : float (optional, default: 100)
    ///     Maximum straight-line distance in meters from the coordinate
    ///     to the walking network; a coordinate farther away raises
    ///     ``ValueError``.
    ///
    /// Returns
    /// -------
    /// dict
    ///     Travel time in seconds to every reachable stop, keyed by
    ///     stop_id; unreachable stops are absent.
    #[pyo3(signature = (origin, date, departure, max_transfers = 4, walking_speed_kmph = 3.6, max_walking_time = 600.0, max_snap_distance = 100.0))]
    #[allow(clippy::too_many_arguments)]
    fn travel_times_from_coordinate(
        &self,
        py: Python<'_>,
        origin: (f64, f64),
        date: &str,
        departure: &str,
        max_transfers: u8,
        walking_speed_kmph: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
    ) -> PyResult<Py<PyDict>> {
        let streets = self.installed_streets()?;
        let speed =
            validated_walking_speed(walking_speed_kmph, max_walking_time, max_snap_distance)?;
        let access = coordinate_links(
            streets,
            origin,
            speed,
            max_walking_time,
            max_snap_distance,
            "origin ",
        )?;
        let departure = parse_time(departure)?;
        let request = Request {
            departure,
            access: request_offsets(&access),
            egress: Vec::new(),
            active_services: self.active_services(date)?,
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
        let departure = parse_time(departure)?;
        let request = Request {
            departure,
            access: vec![(origin, 0)],
            egress: Vec::new(),
            active_services: self.active_services(date)?,
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

    /// Travel times from several stops to every stop, as a matrix.
    ///
    /// One RAPTOR run serves each origin, fanned out over the origins in
    /// parallel with the GIL released; per-worker search state is pooled
    /// across origins. The result is deterministic regardless of
    /// scheduling.
    ///
    /// Parameters
    /// ----------
    /// from_stops : list of str
    ///     GTFS stop_ids of the origin stops; ``<feed_index>:<stop_id>``
    ///     when an id occurs in several merged feeds.
    /// date : str
    ///     Service date as ``YYYY-MM-DD``.
    /// departure : str
    ///     Departure time at every origin as ``HH:MM:SS``.
    /// max_transfers : int (optional, default: 4)
    ///     Maximum number of transfers between rides.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     A ``(len(from_stops), stop_count)`` uint32 array of travel
    ///     times in seconds; row order follows `from_stops`, column
    ///     order follows ``stops``. Unreachable pairs hold the maximum
    ///     uint32 value (4294967295).
    #[pyo3(signature = (from_stops, date, departure, max_transfers = 4))]
    fn travel_time_matrix<'py>(
        &self,
        py: Python<'py>,
        from_stops: Vec<String>,
        date: &str,
        departure: &str,
        max_transfers: u8,
    ) -> PyResult<Bound<'py, PyArray2<u32>>> {
        let origins: Vec<StopIdx> = from_stops
            .iter()
            .map(|stop| self.resolve_stop(stop))
            .collect::<PyResult<_>>()?;
        let departure = parse_time(departure)?;
        let active_services = self.active_services(date)?;
        let requests: Vec<Request> = origins
            .into_iter()
            .map(|origin| Request {
                departure,
                access: vec![(origin, 0)],
                egress: Vec::new(),
                active_services: active_services.clone(),
                max_transfers,
            })
            .collect();
        let stop_count = self.build.timetable.stop_count() as usize;
        let flat: Vec<u32> = py.allow_threads(|| {
            let rows = Raptor.one_to_all_many(&self.build.timetable, &self.transfers, &requests);
            let mut flat = Vec::with_capacity(requests.len() * stop_count);
            for row in rows {
                flat.extend(row.into_iter().map(|arrival| match arrival {
                    Some(arrival) => arrival - departure,
                    None => u32::MAX,
                }));
            }
            flat
        });
        let rows = requests.len();
        flat.into_pyarray(py)
            .reshape([rows, stop_count])
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Travel-time percentiles over a departure window, as a matrix.
    ///
    /// Every minute mark within ``[departure, departure + window)`` is
    /// evaluated through one descending range scan per origin, in
    /// parallel with the GIL released; the returned values are exact
    /// nearest-rank percentiles of the travel-time distribution across
    /// the window's minute marks.
    ///
    /// Parameters
    /// ----------
    /// from_stops : list of str
    ///     GTFS stop_ids of the origin stops.
    /// date : str
    ///     Service date as ``YYYY-MM-DD``.
    /// departure : str
    ///     Window start at every origin as ``HH:MM:SS``.
    /// window : int
    ///     Window length in seconds, at least 1.
    /// percentiles : list of float
    ///     Percentiles in ``[0, 100]``, e.g. ``[10, 50, 90]``.
    /// max_transfers : int (optional, default: 4)
    ///     Maximum number of transfers between rides.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     A ``(len(from_stops), stop_count, len(percentiles))`` uint32
    ///     array of travel times in seconds; unreachable percentiles
    ///     hold the maximum uint32 value (4294967295).
    #[pyo3(signature = (from_stops, date, departure, window, percentiles, max_transfers = 4))]
    #[allow(clippy::too_many_arguments)]
    fn travel_time_percentiles<'py>(
        &self,
        py: Python<'py>,
        from_stops: Vec<String>,
        date: &str,
        departure: &str,
        window: u32,
        percentiles: Vec<f64>,
        max_transfers: u8,
    ) -> PyResult<Bound<'py, PyArray3<u32>>> {
        validate_window(window, &percentiles)?;
        let origins: Vec<StopIdx> = from_stops
            .iter()
            .map(|stop| self.resolve_stop(stop))
            .collect::<PyResult<_>>()?;
        let departure = parse_time(departure)?;
        let active_services = self.active_services(date)?;
        let requests: Vec<Request> = origins
            .into_iter()
            .map(|origin| Request {
                departure,
                access: vec![(origin, 0)],
                egress: Vec::new(),
                active_services: active_services.clone(),
                max_transfers,
            })
            .collect();
        let stop_count = self.build.timetable.stop_count() as usize;
        let flat: Vec<u32> = py.allow_threads(|| {
            Raptor
                .percentile_matrix(
                    &self.build.timetable,
                    &self.transfers,
                    &requests,
                    window,
                    &percentiles,
                )
                .concat()
        });
        let rows = requests.len();
        flat.into_pyarray(py)
            .reshape([rows, stop_count, percentiles.len()])
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Travel-time percentiles over a departure window between
    /// coordinate points — ``travel_time_percentiles`` over linked
    /// points, with each mark's arrival at a destination joined through
    /// its egress links as in ``travel_time_matrix_from_points``.
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``matrix``: a ``(len(origins), len(destinations),
    ///     len(percentiles))`` uint32 array; ``unsnapped_from`` /
    ///     ``unsnapped_to``: indices of points off the walking network.
    #[pyo3(signature = (origins, destinations, date, departure, window, percentiles, max_transfers = 4, walking_speed_kmph = 3.6, max_walking_time = 600.0, max_snap_distance = 100.0))]
    #[allow(clippy::too_many_arguments)]
    fn travel_time_percentiles_from_points(
        &self,
        py: Python<'_>,
        origins: Vec<(f64, f64)>,
        destinations: Vec<(f64, f64)>,
        date: &str,
        departure: &str,
        window: u32,
        percentiles: Vec<f64>,
        max_transfers: u8,
        walking_speed_kmph: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
    ) -> PyResult<Py<PyDict>> {
        let streets = self.installed_streets()?;
        validate_window(window, &percentiles)?;
        let speed =
            validated_walking_speed(walking_speed_kmph, max_walking_time, max_snap_distance)?;
        validate_points(&origins)?;
        validate_points(&destinations)?;
        let departure = parse_time(departure)?;
        let active_services = self.active_services(date)?;
        let destination_count = destinations.len();
        let (flat, unsnapped_from, unsnapped_to) = py.allow_threads(|| {
            let origin_links =
                streets.link_many(&origins, speed, max_walking_time, max_snap_distance);
            let destination_links =
                streets.link_many(&destinations, speed, max_walking_time, max_snap_distance);
            let unsnapped_from = unsnapped(&origin_links);
            let unsnapped_to = unsnapped(&destination_links);
            let requests: Vec<Request> = origin_links
                .iter()
                .map(|links| Request {
                    departure,
                    access: request_offsets(links.as_deref().unwrap_or(&[])),
                    egress: Vec::new(),
                    active_services: active_services.clone(),
                    max_transfers,
                })
                .collect();
            let egress = egress_tables(&destination_links);
            let flat = Raptor
                .percentile_matrix_to_points(
                    &self.build.timetable,
                    &self.transfers,
                    &requests,
                    &egress,
                    window,
                    &percentiles,
                )
                .concat();
            (flat, unsnapped_from, unsnapped_to)
        });
        let result = PyDict::new(py);
        result.set_item(
            "matrix",
            flat.into_pyarray(py)
                .reshape([origins.len(), destination_count, percentiles.len()])
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        )?;
        result.set_item("unsnapped_from", unsnapped_from.into_pyarray(py))?;
        result.set_item("unsnapped_to", unsnapped_to.into_pyarray(py))?;
        Ok(result.unbind())
    }

    /// The fastest journey's aggregated costs per OD pair, long format.
    ///
    /// One RAPTOR run serves each origin, fanned out in parallel with
    /// the GIL released as in ``travel_time_matrix``; each reachable
    /// pair's costs come from walking the winning label chain. Requires
    /// installed trip distances.
    ///
    /// Parameters
    /// ----------
    /// from_stops : list of str
    ///     GTFS stop_ids of the origin stops.
    /// date : str
    ///     Service date as ``YYYY-MM-DD``.
    /// departure : str
    ///     Departure time at every origin as ``HH:MM:SS``.
    /// factors : list of (str, float)
    ///     Grams CO₂e per passenger-kilometer per trip, resolved by
    ///     ``cafein.emissions.trip_factors``; NaN marks a trip without
    ///     a factor, poisoning the emissions of journeys that ride it.
    ///     Rows for unknown trips are ignored.
    /// max_transfers : int (optional, default: 4)
    ///     Maximum number of transfers between rides.
    /// to_stops : list of str (optional)
    ///     Destination stops; every stop when omitted.
    /// geometries : bool (optional, default: False)
    ///     Attach each pair's ridden legs as a WKB MultiLineString;
    ///     requires installed leg geometries.
    ///
    /// Returns
    /// -------
    /// dict
    ///     Equal-length arrays for the reachable pairs: ``from`` (row
    ///     into `from_stops`), ``to`` (index into ``stops``),
    ///     ``travel_time`` (seconds), ``rides``, ``transit_distance``
    ///     and ``walk_distance`` (meters), ``emissions`` (grams CO₂e,
    ///     NaN when unresolved), and with `geometries` a ``geometry``
    ///     list of WKB bytes.
    #[pyo3(signature = (from_stops, date, departure, factors, max_transfers = 4, to_stops = None, geometries = false))]
    #[allow(clippy::too_many_arguments)]
    fn travel_cost_matrix(
        &self,
        py: Python<'_>,
        from_stops: Vec<String>,
        date: &str,
        departure: &str,
        factors: Vec<(String, f64)>,
        max_transfers: u8,
        to_stops: Option<Vec<String>>,
        geometries: bool,
    ) -> PyResult<Py<PyDict>> {
        let Some(geometry) = &self.geometry else {
            return Err(PyValueError::new_err(
                "no trip distances installed; build the network with trip distances enabled",
            ));
        };
        if geometries && self.leg_geometry.is_none() {
            return Err(PyValueError::new_err(
                "no leg geometries installed; build the network with leg geometries enabled",
            ));
        }
        let origins: Vec<StopIdx> = from_stops
            .iter()
            .map(|stop| self.resolve_stop(stop))
            .collect::<PyResult<_>>()?;
        let destinations: Vec<StopIdx> = match to_stops {
            Some(stops) => stops
                .iter()
                .map(|stop| self.resolve_stop(stop))
                .collect::<PyResult<_>>()?,
            None => (0..self.build.timetable.stop_count())
                .map(StopIdx)
                .collect(),
        };
        let mut per_trip = vec![f64::NAN; self.build.timetable.trip_count() as usize];
        for (trip_id, factor) in &factors {
            if let Some(&trip) = self.trips_by_public_id.get(trip_id) {
                per_trip[trip.0 as usize] = *factor;
            }
        }
        let departure = parse_time(departure)?;
        let active_services = self.active_services(date)?;
        let requests: Vec<Request> = origins
            .into_iter()
            .map(|origin| Request {
                departure,
                access: vec![(origin, 0)],
                egress: Vec::new(),
                active_services: active_services.clone(),
                max_transfers,
            })
            .collect();
        let inputs = CostInputs {
            geometry,
            factors: &per_trip,
            leg_geometry: self.leg_geometry.as_ref(),
            with_geometry: geometries,
        };
        let rows = py.allow_threads(|| {
            Raptor.cost_matrix(
                &self.build.timetable,
                &self.transfers,
                &inputs,
                &requests,
                &destinations,
            )
        });

        let total: usize = rows.iter().map(Vec::len).sum();
        let mut from = Vec::with_capacity(total);
        let mut to = Vec::with_capacity(total);
        let mut travel_time = Vec::with_capacity(total);
        let mut rides = Vec::with_capacity(total);
        let mut transit_distance = Vec::with_capacity(total);
        let mut walk_distance = Vec::with_capacity(total);
        let mut emissions = Vec::with_capacity(total);
        let wkbs = PyList::empty(py);
        for (origin, origin_rows) in rows.into_iter().enumerate() {
            for row in origin_rows {
                from.push(origin as u32);
                to.push(row.to);
                travel_time.push(row.seconds);
                rides.push(row.rides);
                transit_distance.push(row.transit_meters);
                walk_distance.push(row.walk_meters);
                emissions.push(row.emission_grams);
                if geometries {
                    match row.geometry {
                        Some(wkb) => wkbs.append(PyBytes::new(py, &wkb))?,
                        None => wkbs.append(py.None())?,
                    }
                }
            }
        }
        let result = PyDict::new(py);
        result.set_item("from", from.into_pyarray(py))?;
        result.set_item("to", to.into_pyarray(py))?;
        result.set_item("travel_time", travel_time.into_pyarray(py))?;
        result.set_item("rides", rides.into_pyarray(py))?;
        result.set_item("transit_distance", transit_distance.into_pyarray(py))?;
        result.set_item("walk_distance", walk_distance.into_pyarray(py))?;
        result.set_item("emissions", emissions.into_pyarray(py))?;
        if geometries {
            result.set_item("geometry", wkbs)?;
        }
        Ok(result.unbind())
    }

    /// Travel times between coordinate points, as a matrix.
    ///
    /// Every point is linked once against the street network (its
    /// walkable stops with access times); one RAPTOR run then serves
    /// each origin, and a destination's time is the minimum over its
    /// links of the arrival at the link's stop plus the egress walk.
    /// Runs in parallel with the GIL released. Requires an installed
    /// street network.
    ///
    /// Parameters
    /// ----------
    /// origins, destinations : list of (float, float)
    ///     ``(lat, lon)`` coordinates, in EPSG:4326.
    /// date : str
    ///     Service date as ``YYYY-MM-DD``.
    /// departure : str
    ///     Departure time at every origin as ``HH:MM:SS``.
    /// max_transfers : int (optional, default: 4)
    ///     Maximum number of transfers between rides.
    /// walking_speed_kmph, max_walking_time, max_snap_distance :
    ///     The street-search options, as in ``access_stops``.
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``matrix``: a ``(len(origins), len(destinations))`` uint32
    ///     array of travel times in seconds, ``2**32 - 1`` where
    ///     unreachable; ``unsnapped_from`` / ``unsnapped_to``: indices
    ///     of points farther than `max_snap_distance` from the walking
    ///     network (their rows/columns are unreachable).
    #[pyo3(signature = (origins, destinations, date, departure, max_transfers = 4, walking_speed_kmph = 3.6, max_walking_time = 600.0, max_snap_distance = 100.0))]
    #[allow(clippy::too_many_arguments)]
    fn travel_time_matrix_from_points(
        &self,
        py: Python<'_>,
        origins: Vec<(f64, f64)>,
        destinations: Vec<(f64, f64)>,
        date: &str,
        departure: &str,
        max_transfers: u8,
        walking_speed_kmph: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
    ) -> PyResult<Py<PyDict>> {
        let streets = self.installed_streets()?;
        let speed =
            validated_walking_speed(walking_speed_kmph, max_walking_time, max_snap_distance)?;
        validate_points(&origins)?;
        validate_points(&destinations)?;
        let departure = parse_time(departure)?;
        let active_services = self.active_services(date)?;
        let stop_count = self.build.timetable.stop_count() as usize;
        let destination_count = destinations.len();
        let (flat, unsnapped_from, unsnapped_to) = py.allow_threads(|| {
            let origin_links =
                streets.link_many(&origins, speed, max_walking_time, max_snap_distance);
            let destination_links =
                streets.link_many(&destinations, speed, max_walking_time, max_snap_distance);
            let unsnapped_from = unsnapped(&origin_links);
            let unsnapped_to = unsnapped(&destination_links);
            let requests: Vec<Request> = origin_links
                .iter()
                .map(|links| Request {
                    departure,
                    access: request_offsets(links.as_deref().unwrap_or(&[])),
                    egress: Vec::new(),
                    active_services: active_services.clone(),
                    max_transfers,
                })
                .collect();
            let egress = egress_tables(&destination_links);
            let rows = Raptor.one_to_all_many(&self.build.timetable, &self.transfers, &requests);
            let mut flat = vec![u32::MAX; requests.len() * destination_count];
            for (origin, arrivals) in rows.iter().enumerate() {
                debug_assert_eq!(arrivals.len(), stop_count);
                for (point, links) in egress.iter().enumerate() {
                    let mut best = u32::MAX;
                    for &(stop, seconds, _) in links {
                        let Some(at_stop) = arrivals[stop.0 as usize] else {
                            continue;
                        };
                        let Some(arrival) =
                            at_stop.checked_add(seconds).filter(|&at| at != u32::MAX)
                        else {
                            continue;
                        };
                        best = best.min(arrival);
                    }
                    if best != u32::MAX {
                        flat[origin * destination_count + point] = best - departure;
                    }
                }
            }
            (flat, unsnapped_from, unsnapped_to)
        });
        let result = PyDict::new(py);
        result.set_item(
            "matrix",
            flat.into_pyarray(py)
                .reshape([origins.len(), destination_count])
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        )?;
        result.set_item("unsnapped_from", unsnapped_from.into_pyarray(py))?;
        result.set_item("unsnapped_to", unsnapped_to.into_pyarray(py))?;
        Ok(result.unbind())
    }

    /// The fastest journey's aggregated costs between coordinate
    /// points, long format — ``travel_cost_matrix`` over linked points.
    ///
    /// Points link once against the street network; a destination's
    /// travel time is the minimum over its links of the arrival plus
    /// the egress walk, and its costs are the winning journey's, with
    /// the access and egress walks counted in ``walk_distance``.
    /// Requires an installed street network and trip distances.
    ///
    /// Returns
    /// -------
    /// dict
    ///     As ``travel_cost_matrix`` — ``from`` and ``to`` index the
    ///     origin and destination point lists — plus
    ///     ``unsnapped_from`` / ``unsnapped_to`` with the indices of
    ///     points off the walking network.
    #[pyo3(signature = (origins, destinations, date, departure, factors, max_transfers = 4, walking_speed_kmph = 3.6, max_walking_time = 600.0, max_snap_distance = 100.0, geometries = false))]
    #[allow(clippy::too_many_arguments)]
    fn travel_cost_matrix_from_points(
        &self,
        py: Python<'_>,
        origins: Vec<(f64, f64)>,
        destinations: Vec<(f64, f64)>,
        date: &str,
        departure: &str,
        factors: Vec<(String, f64)>,
        max_transfers: u8,
        walking_speed_kmph: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
        geometries: bool,
    ) -> PyResult<Py<PyDict>> {
        let streets = self.installed_streets()?;
        let Some(geometry) = &self.geometry else {
            return Err(PyValueError::new_err(
                "no trip distances installed; build the network with trip distances enabled",
            ));
        };
        if geometries && self.leg_geometry.is_none() {
            return Err(PyValueError::new_err(
                "no leg geometries installed; build the network with leg geometries enabled",
            ));
        }
        let speed =
            validated_walking_speed(walking_speed_kmph, max_walking_time, max_snap_distance)?;
        validate_points(&origins)?;
        validate_points(&destinations)?;
        let mut per_trip = vec![f64::NAN; self.build.timetable.trip_count() as usize];
        for (trip_id, factor) in &factors {
            if let Some(&trip) = self.trips_by_public_id.get(trip_id) {
                per_trip[trip.0 as usize] = *factor;
            }
        }
        let departure = parse_time(departure)?;
        let active_services = self.active_services(date)?;
        let inputs = CostInputs {
            geometry,
            factors: &per_trip,
            leg_geometry: self.leg_geometry.as_ref(),
            with_geometry: geometries,
        };
        let (rows, unsnapped_from, unsnapped_to) = py.allow_threads(|| {
            let origin_links =
                streets.link_many(&origins, speed, max_walking_time, max_snap_distance);
            let destination_links =
                streets.link_many(&destinations, speed, max_walking_time, max_snap_distance);
            let unsnapped_from = unsnapped(&origin_links);
            let unsnapped_to = unsnapped(&destination_links);
            let mut requests = Vec::with_capacity(origin_links.len());
            let mut access_meters = Vec::with_capacity(origin_links.len());
            for links in &origin_links {
                let links = links.as_deref().unwrap_or(&[]);
                requests.push(Request {
                    departure,
                    access: request_offsets(links),
                    egress: Vec::new(),
                    active_services: active_services.clone(),
                    max_transfers,
                });
                access_meters.push(
                    links
                        .iter()
                        .map(|walk| (walk.stop, walk.meters))
                        .collect::<HashMap<_, _>>(),
                );
            }
            let egress = egress_tables(&destination_links);
            let rows = Raptor.cost_matrix_to_points(
                &self.build.timetable,
                &self.transfers,
                &inputs,
                &requests,
                &access_meters,
                &egress,
            );
            (rows, unsnapped_from, unsnapped_to)
        });

        let total: usize = rows.iter().map(Vec::len).sum();
        let mut from = Vec::with_capacity(total);
        let mut to = Vec::with_capacity(total);
        let mut travel_time = Vec::with_capacity(total);
        let mut rides = Vec::with_capacity(total);
        let mut transit_distance = Vec::with_capacity(total);
        let mut walk_distance = Vec::with_capacity(total);
        let mut emissions = Vec::with_capacity(total);
        let wkbs = PyList::empty(py);
        for (origin, origin_rows) in rows.into_iter().enumerate() {
            for row in origin_rows {
                from.push(origin as u32);
                to.push(row.to);
                travel_time.push(row.seconds);
                rides.push(row.rides);
                transit_distance.push(row.transit_meters);
                walk_distance.push(row.walk_meters);
                emissions.push(row.emission_grams);
                if geometries {
                    match row.geometry {
                        Some(wkb) => wkbs.append(PyBytes::new(py, &wkb))?,
                        None => wkbs.append(py.None())?,
                    }
                }
            }
        }
        let result = PyDict::new(py);
        result.set_item("from", from.into_pyarray(py))?;
        result.set_item("to", to.into_pyarray(py))?;
        result.set_item("travel_time", travel_time.into_pyarray(py))?;
        result.set_item("rides", rides.into_pyarray(py))?;
        result.set_item("transit_distance", transit_distance.into_pyarray(py))?;
        result.set_item("walk_distance", walk_distance.into_pyarray(py))?;
        result.set_item("emissions", emissions.into_pyarray(py))?;
        if geometries {
            result.set_item("geometry", wkbs)?;
        }
        result.set_item("unsnapped_from", unsnapped_from.into_pyarray(py))?;
        result.set_item("unsnapped_to", unsnapped_to.into_pyarray(py))?;
        Ok(result.unbind())
    }
}

impl TransportNetwork {
    /// The installed street network, or a `ValueError` explaining that
    /// coordinate queries need one.
    fn installed_streets(&self) -> PyResult<&StreetNetwork> {
        self.streets.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "no street network installed; build the network with an OSM extract",
            )
        })
    }

    /// The service-activity flags of a `YYYY-MM-DD` date.
    fn active_services(&self, date: &str) -> PyResult<Vec<bool>> {
        let date = NaiveDate::parse_from_str(date, "%Y-%m-%d")
            .map_err(|error| PyValueError::new_err(format!("invalid date '{date}': {error}")))?;
        Ok(self.build.services.active_on(date))
    }

    /// Runs a request through the router and converts the journeys,
    /// attaching walk-leg distances when the walk lengths are known.
    fn route_request(
        &self,
        py: Python<'_>,
        request: &Request,
        window: Option<u32>,
        walks: Option<&WalkMaps>,
        geometries: bool,
    ) -> PyResult<Py<PyList>> {
        let journeys = match window {
            None => Raptor.route(&self.build.timetable, &self.transfers, request),
            Some(window) => {
                Raptor.route_range(&self.build.timetable, &self.transfers, request, window)
            }
        };
        let result = PyList::empty(py);
        for journey in &journeys {
            result.append(self.journey_to_dict(py, journey, walks, geometries)?)?;
        }
        Ok(result.unbind())
    }

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

    fn journey_to_dict(
        &self,
        py: Python<'_>,
        journey: &Journey,
        walks: Option<&WalkMaps>,
        geometries: bool,
    ) -> PyResult<Py<PyDict>> {
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
                    entry.set_item(
                        "distance",
                        walks.and_then(|walks| walks.access.get(&to_stop)).copied(),
                    )?;
                    entry.set_item("distance_provenance", py.None())?;
                    entry.set_item("geometry", py.None())?;
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
                    let geometry =
                        self.leg_geometry
                            .as_ref()
                            .filter(|_| geometries)
                            .map(|geometry| {
                                wkb_line_string(
                                    py,
                                    &geometry.leg_coordinates(
                                        trip,
                                        board_position,
                                        alight_position,
                                    ),
                                )
                            });
                    entry.set_item("geometry", geometry)?;
                }
                Leg::Transfer {
                    from_stop,
                    to_stop,
                    departure,
                    arrival,
                } => {
                    // Transfers are deduplicated per stop pair, so the
                    // one edge found is the one routing relaxed.
                    let meters = self
                        .transfers
                        .from_stop(from_stop)
                        .iter()
                        .find(|transfer| transfer.to == to_stop)
                        .map(|transfer| transfer.meters);
                    entry.set_item("type", "transfer")?;
                    entry.set_item("from_stop", self.public_stop_id(from_stop))?;
                    entry.set_item("to_stop", self.public_stop_id(to_stop))?;
                    entry.set_item("departure", departure)?;
                    entry.set_item("arrival", arrival)?;
                    entry.set_item("distance", meters)?;
                    entry.set_item("distance_provenance", py.None())?;
                    entry.set_item("geometry", py.None())?;
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
                    entry.set_item(
                        "distance",
                        walks
                            .and_then(|walks| walks.egress.get(&from_stop))
                            .copied(),
                    )?;
                    entry.set_item("distance_provenance", py.None())?;
                    entry.set_item("geometry", py.None())?;
                }
            }
            legs.append(entry)?;
        }
        dict.set_item("legs", legs)?;
        Ok(dict.unbind())
    }
}

/// The walking speed in m/s of validated street-query parameters, or a
/// `ValueError` naming the parameter that is out of range.
fn validated_walking_speed(
    walking_speed_kmph: f64,
    max_walking_time: f64,
    max_snap_distance: f64,
) -> PyResult<f64> {
    if !walking_speed_kmph.is_finite() || walking_speed_kmph <= 0.0 {
        return Err(PyValueError::new_err(
            "walking_speed_kmph must be a positive, finite number",
        ));
    }
    if !max_walking_time.is_finite() || max_walking_time < 0.0 {
        return Err(PyValueError::new_err(
            "max_walking_time must be a non-negative, finite number",
        ));
    }
    if !max_snap_distance.is_finite() || max_snap_distance < 0.0 {
        return Err(PyValueError::new_err(
            "max_snap_distance must be a non-negative, finite number",
        ));
    }
    Ok(walking_speed_kmph / 3.6)
}

/// The stops walkable from a coordinate, or a `ValueError` when the
/// coordinate is invalid or off the network; `side` prefixes the message
/// (e.g. `"origin "`) to name the endpoint.
fn coordinate_links(
    streets: &StreetNetwork,
    coordinate: (f64, f64),
    walking_speed: f64,
    max_walking_time: f64,
    max_snap_distance: f64,
    side: &str,
) -> PyResult<Vec<WalkedStop>> {
    let (lat, lon) = coordinate;
    if !lat.is_finite() || !lon.is_finite() {
        return Err(PyValueError::new_err(format!(
            "{side}lat and lon must be finite"
        )));
    }
    streets
        .access_stops(lat, lon, walking_speed, max_walking_time, max_snap_distance)
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "{side}({lat}, {lon}) is farther than {max_snap_distance} m \
                 from the walking network"
            ))
        })
}

/// Encodes coordinates as a little-endian WKB LineString (XY).
fn wkb_line_string<'py>(py: Python<'py>, coordinates: &[(f64, f64)]) -> Bound<'py, PyBytes> {
    PyBytes::new(py, &cafein_core::geometry::wkb_line_string(coordinates))
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
