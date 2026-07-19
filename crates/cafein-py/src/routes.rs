//! The one-pair and one-to-all time queries.

use super::*;

#[pymethods]
impl TransportNetwork {
    /// Route between two transit stops for a single departure.
    ///
    /// Journeys ride trips and change vehicles at shared stops or over
    /// the transfers installed with ``set_transfers``; transit legs
    /// report their distance and its provenance when trip distances are
    /// installed. ``route_between_coordinates`` routes door-to-door from
    /// arbitrary coordinates. Legs carry times, stops, distances, and
    /// provenance; transit legs add their geometry as a WKB LineString
    /// when leg geometries are installed, and transfer legs their
    /// walked street path when the street network is installed.
    ///
    /// With a whole-day ULTRA set (``compute_ultra_shortcuts``), the two
    /// stops are routed **door-to-door between their coordinates** — the
    /// same unrestricted initial/intermediate/final walking as
    /// ``route_between_coordinates`` — and ``walking_speed_kmph``,
    /// ``max_walking_time``, and ``max_snap_distance`` bound that walking.
    /// Without such a set (or when a stop has no coordinate or is off the
    /// walking network) the query boards at the origin stop and relaxes the
    /// closure transfers, and those three arguments are ignored.
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
    /// max_transfers : int (optional, default: 7)
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
    #[pyo3(signature = (from_stop, to_stop, date, departure, max_transfers = 7, window = None, exclude_routes = vec![], exclude_trips = vec![], exclude_stops = vec![], walking_speed_kmph = 3.6, max_walking_time = 7200.0, max_snap_distance = 1600.0, geometries = true))]
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
        exclude_routes: Vec<String>,
        exclude_trips: Vec<String>,
        exclude_stops: Vec<String>,
        walking_speed_kmph: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
        geometries: bool,
    ) -> PyResult<Py<PyList>> {
        let origin = self.resolve_stop(from_stop)?;
        let destination = self.resolve_stop(to_stop)?;
        let exclusions = self.exclusion_masks(&exclude_routes, &exclude_trips, &exclude_stops)?;
        // An excluded endpoint is unreachable by contract - also on the
        // door-to-door path, which could otherwise reach the stop's
        // coordinates through a neighbour or a direct walk.
        if let Some(excluded) = exclusions.as_deref() {
            if excluded.excludes_stop(origin) || excluded.excludes_stop(destination) {
                return Ok(PyList::empty(py).unbind());
            }
        }
        // With a whole-day ULTRA set, route door-to-door between the stops'
        // coordinates for unrestricted walking; otherwise board at the origin
        // stop and relax the closure (today's behaviour). Exclusions keep
        // the closure path.
        if self.ultra_active() && exclusions.is_none() {
            if let (Some(streets), Some(from_xy), Some(to_xy)) = (
                self.streets.as_ref(),
                self.stop_coordinate(origin),
                self.stop_coordinate(destination),
            ) {
                if streets
                    .snap(from_xy.0, from_xy.1, max_snap_distance)
                    .is_some()
                    && streets.snap(to_xy.0, to_xy.1, max_snap_distance).is_some()
                {
                    return self.route_between_coordinates(
                        py,
                        from_xy,
                        to_xy,
                        date,
                        departure,
                        max_transfers,
                        window,
                        exclude_routes,
                        exclude_trips,
                        exclude_stops,
                        walking_speed_kmph,
                        max_walking_time,
                        max_snap_distance,
                        geometries,
                    );
                }
            }
        }
        let request = Request {
            departure: parse_time(departure)?,
            access: vec![(origin, 0)],
            egress: vec![(destination, 0)],
            active_services: self.active_services(date)?,
            active_services_previous: self.active_services_previous(date)?,
            max_transfers,
            exclusions,
        };
        self.route_request(py, &request, window, None, None, geometries)
    }

    /// Route door-to-door between two coordinates for a single departure.
    ///
    /// The street network installed with ``set_street_network`` provides
    /// walking access from the origin to nearby stops and egress from
    /// stops to the destination; journeys otherwise behave as in
    /// ``route_between_stops``. Access and egress legs report their
    /// walking distance in meters; a coordinate farther than
    /// ``max_snap_distance`` from the walking network raises
    /// ``ValueError``. Walking all the way is a journey too: within
    /// ``max_walking_time`` the result leads with a walking-only
    /// journey (one ``walk`` leg, zero rides), and a journey is dropped
    /// when walking, leaving at that journey's own departure, would
    /// arrive no later. With ``geometries`` (the default), walk legs
    /// carry their walked street path as WKB LineStrings alongside the
    /// transit legs' geometry.
    ///
    /// Parameters
    /// ----------
    /// origin, destination : (float, float)
    ///     ``(lat, lon)`` coordinates, in EPSG:4326.
    /// date : str
    ///     Service date as ``YYYY-MM-DD``.
    /// departure : str
    ///     Departure time at the origin coordinate as ``HH:MM:SS``.
    /// max_transfers : int (optional, default: 7)
    ///     Maximum number of transfers between rides.
    /// window : int (optional)
    ///     Departure window in seconds, as in ``route_between_stops``.
    /// walking_speed_kmph : float (optional, default: 3.6)
    ///     Walking speed in km/h of the access and egress searches.
    /// max_walking_time : float (optional, default: 7200)
    ///     Walking-time cutoff in seconds of each street search.
    /// max_snap_distance : float (optional, default: 1600)
    ///     Maximum straight-line distance in meters from each coordinate
    ///     to the walking network.
    ///
    /// Returns
    /// -------
    /// list of dict
    ///     Journeys as in ``route_between_stops``; arrivals include the
    ///     egress walk.
    #[pyo3(signature = (origin, destination, date, departure, max_transfers = 7, window = None, exclude_routes = vec![], exclude_trips = vec![], exclude_stops = vec![], walking_speed_kmph = 3.6, max_walking_time = 7200.0, max_snap_distance = 1600.0, geometries = true))]
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
        exclude_routes: Vec<String>,
        exclude_trips: Vec<String>,
        exclude_stops: Vec<String>,
        walking_speed_kmph: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
        geometries: bool,
    ) -> PyResult<Py<PyList>> {
        let exclusions = self.exclusion_masks(&exclude_routes, &exclude_trips, &exclude_stops)?;
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
        // The endpoints re-snap for geometry; the searches above prove
        // both snaps exist.
        let ends = CoordinateEnds {
            origin,
            origin_snap: streets
                .snap(origin.0, origin.1, max_snap_distance)
                .expect("origin linked above"),
            destination,
            destination_snap: streets
                .snap(destination.0, destination.1, max_snap_distance)
                .expect("destination linked above"),
        };
        let request = Request {
            departure: parse_time(departure)?,
            access: request_offsets(&access),
            egress: request_offsets(&egress),
            active_services: self.active_services(date)?,
            active_services_previous: self.active_services_previous(date)?,
            max_transfers,
            exclusions: exclusions.clone(),
        };
        // The walking-only alternative: door to door over the streets,
        // no vehicle, available at every departure. It dominates a
        // journey when walking out at that journey's own departure
        // would arrive no later (walking rides nothing), and is
        // dominated only by a faster journey that also rides nothing.
        // A destination at the origin's exact coordinate is a zero
        // walk — snap arithmetic would charge the connector twice.
        let direct = if origin == destination {
            Some((0, 0.0))
        } else {
            streets
                .walk_to_snaps(
                    &ends.origin_snap,
                    &[Some(ends.destination_snap)],
                    speed,
                    max_walking_time,
                )
                .swap_remove(0)
        };
        // One choice for both routing and the leg-distance lookup, so an
        // ULTRA-routed leg is measured in the ULTRA set. Exclusions keep
        // the closure: the shortcut set's witness pruning is not robust
        // under supply removal.
        let transfers = if request.exclusions.is_some() {
            &self.transfers
        } else {
            self.time_transfers()
        };
        let journeys = match window {
            None => Raptor.route(&self.build.timetable, transfers, &request),
            Some(window) => Raptor.route_range(&self.build.timetable, transfers, &request, window),
        };
        let kept: Vec<&Journey> = journeys
            .iter()
            .filter(|journey| match direct {
                Some((walk_seconds, _)) => journey.arrival - journey.departure < walk_seconds,
                None => true,
            })
            .collect();
        let result = PyList::empty(py);
        if let Some(walk) = direct.filter(|_| !kept.iter().any(|journey| journey.rides() == 0)) {
            // Journeys sort by (departure, rides); the walk leaves at the
            // requested departure with zero rides, so it leads the list.
            result.append(self.walk_journey_dict(
                py,
                request.departure,
                walk,
                &ends,
                geometries,
            )?)?;
        }
        for journey in kept {
            result.append(self.journey_to_dict(
                py,
                journey,
                Some(&walks),
                Some(&ends),
                geometries,
                transfers,
            )?)?;
        }
        Ok(result.unbind())
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
    /// max_transfers : int (optional, default: 7)
    ///     Maximum number of transfers between rides.
    /// walking_speed_kmph : float (optional, default: 3.6)
    ///     Walking speed in km/h of the access search.
    /// max_walking_time : float (optional, default: 7200)
    ///     Walking-time cutoff in seconds of the access search.
    /// max_snap_distance : float (optional, default: 1600)
    ///     Maximum straight-line distance in meters from the coordinate
    ///     to the walking network; a coordinate farther away raises
    ///     ``ValueError``.
    ///
    /// Returns
    /// -------
    /// dict
    ///     Travel time in seconds to every reachable stop, keyed by
    ///     stop_id; unreachable stops are absent.
    #[pyo3(signature = (origin, date, departure, max_transfers = 7, walking_speed_kmph = 3.6, max_walking_time = 7200.0, max_snap_distance = 1600.0))]
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
            active_services_previous: self.active_services_previous(date)?,
            max_transfers,
            exclusions: None,
        };
        // Under a whole-day ULTRA set the intermediate transfers use the
        // shortcuts and a bounded final walk (`<= max_walking_time`) reaches
        // the remaining stops; otherwise this is the closure, tau-direct search
        // (`time_transfers` is the closure then, and the fold is skipped).
        let mut arrivals =
            Raptor.one_to_all(&self.build.timetable, self.time_transfers(), &request);
        if self.ultra_active() {
            let egress = self.final_egress(streets, speed, max_walking_time, max_snap_distance);
            self.fold_final_transfers(&mut arrivals, &egress);
        }
        self.arrivals_dict(py, &arrivals, departure)
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
    /// max_transfers : int (optional, default: 7)
    ///     Maximum number of transfers between rides.
    /// walking_speed_kmph, max_walking_time, max_snap_distance : float
    ///     Bound the door-to-door walking under a whole-day ULTRA set
    ///     (defaults 3.6 km/h, 7200 s, 1600 m); ignored otherwise.
    ///
    /// With a whole-day ULTRA set (``compute_ultra_shortcuts``) the origin
    /// stop is treated as its coordinate and every stop is reached
    /// door-to-door — unrestricted initial, intermediate, and final walking;
    /// without it the search boards at the origin stop over the closure.
    ///
    /// Returns
    /// -------
    /// dict
    ///     Travel time in seconds to every reachable stop, keyed by
    ///     public stop_id; unreachable stops are absent. On the closure path
    ///     the origin maps to 0; under a whole-day ULTRA set it is the
    ///     door-to-door time from the origin stop's coordinate and may cost
    ///     the short walk to the platform.
    #[pyo3(signature = (from_stop, date, departure, max_transfers = 7, walking_speed_kmph = 3.6, max_walking_time = 7200.0, max_snap_distance = 1600.0))]
    #[allow(clippy::too_many_arguments)]
    fn travel_times_from_stop(
        &self,
        py: Python<'_>,
        from_stop: &str,
        date: &str,
        departure: &str,
        max_transfers: u8,
        walking_speed_kmph: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
    ) -> PyResult<Py<PyDict>> {
        let origin = self.resolve_stop(from_stop)?;
        let departure = parse_time(departure)?;
        // With a whole-day ULTRA set, treat the origin stop as its coordinate
        // and reach every stop door-to-door (coordinate access, ULTRA
        // intermediate transfers, one final walk bounded by max_walking_time);
        // otherwise board at the origin stop and relax the closure (today's
        // behaviour).
        if self.ultra_active() {
            if let (Some(streets), Some(coordinate)) =
                (self.streets.as_ref(), self.stop_coordinate(origin))
            {
                if streets
                    .snap(coordinate.0, coordinate.1, max_snap_distance)
                    .is_some()
                {
                    let speed = validated_walking_speed(
                        walking_speed_kmph,
                        max_walking_time,
                        max_snap_distance,
                    )?;
                    let access = coordinate_links(
                        streets,
                        coordinate,
                        speed,
                        max_walking_time,
                        max_snap_distance,
                        "origin ",
                    )?;
                    let request = Request {
                        departure,
                        access: request_offsets(&access),
                        egress: Vec::new(),
                        active_services: self.active_services(date)?,
                        active_services_previous: self.active_services_previous(date)?,
                        max_transfers,
                        exclusions: None,
                    };
                    let mut arrivals =
                        Raptor.one_to_all(&self.build.timetable, self.time_transfers(), &request);
                    let egress =
                        self.final_egress(streets, speed, max_walking_time, max_snap_distance);
                    self.fold_final_transfers(&mut arrivals, &egress);
                    return self.arrivals_dict(py, &arrivals, departure);
                }
            }
        }
        let request = Request {
            departure,
            access: vec![(origin, 0)],
            egress: Vec::new(),
            active_services: self.active_services(date)?,
            active_services_previous: self.active_services_previous(date)?,
            max_transfers,
            exclusions: None,
        };
        let arrivals = Raptor.one_to_all(&self.build.timetable, &self.transfers, &request);
        self.arrivals_dict(py, &arrivals, departure)
    }
}
