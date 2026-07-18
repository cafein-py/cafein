//! Shared request plumbing: ids, services, journey assembly, and
//! option parsing.

use super::*;

impl TransportNetwork {
    /// The installed street network, or a `ValueError` explaining that
    /// coordinate queries need one.
    pub(super) fn installed_streets(&self) -> PyResult<&StreetNetwork> {
        self.streets.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "no street network installed; build the network with an OSM extract",
            )
        })
    }

    /// The service-activity flags of a `YYYY-MM-DD` date.
    pub(super) fn active_services(&self, date: &str) -> PyResult<Vec<bool>> {
        let date = NaiveDate::parse_from_str(date, "%Y-%m-%d")
            .map_err(|error| PyValueError::new_err(format!("invalid date '{date}': {error}")))?;
        Ok(self.build.services.active_on(date))
    }

    /// The services running the day before `date`, whose over-midnight
    /// trips reach into it.
    pub(super) fn active_services_previous(&self, date: &str) -> PyResult<Vec<bool>> {
        let date = NaiveDate::parse_from_str(date, "%Y-%m-%d")
            .map_err(|error| PyValueError::new_err(format!("invalid date '{date}': {error}")))?;
        let previous = date
            .pred_opt()
            .ok_or_else(|| PyValueError::new_err(format!("date '{date}' has no previous day")))?;
        Ok(self.build.services.active_on(previous))
    }

    /// Runs a request through the router and converts the journeys,
    /// attaching walk-leg distances when the walk lengths are known.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn route_request(
        &self,
        py: Python<'_>,
        request: &Request,
        window: Option<u32>,
        walks: Option<&WalkMaps>,
        ends: Option<&CoordinateEnds>,
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
            result.append(self.journey_to_dict(
                py,
                journey,
                walks,
                ends,
                geometries,
                &self.transfers,
            )?)?;
        }
        Ok(result.unbind())
    }

    /// A walking-only journey between the query coordinates, as a dict
    /// shaped like ``journey_to_dict``'s: one ``walk`` leg carrying the
    /// exact street distance and, when asked, the walked path.
    pub(super) fn walk_journey_dict(
        &self,
        py: Python<'_>,
        departure: u32,
        (walk_seconds, meters): (u32, f64),
        ends: &CoordinateEnds,
        geometries: bool,
    ) -> PyResult<Py<PyDict>> {
        let arrival = departure.saturating_add(walk_seconds);
        let dict = PyDict::new(py);
        dict.set_item("departure", departure)?;
        dict.set_item("arrival", arrival)?;
        dict.set_item("rides", 0)?;
        let entry = PyDict::new(py);
        entry.set_item("type", "walk")?;
        entry.set_item("departure", departure)?;
        entry.set_item("arrival", arrival)?;
        entry.set_item("distance", meters)?;
        entry.set_item("distance_provenance", py.None())?;
        let geometry = geometries
            .then(|| {
                if ends.origin == ends.destination {
                    // A zero walk degenerates at its own coordinate.
                    let at = (ends.origin.1, ends.origin.0);
                    Some(wkb_line_string(py, &[at, at]))
                } else {
                    self.walk_wkb(
                        py,
                        ends.origin,
                        &ends.origin_snap,
                        ends.destination,
                        &ends.destination_snap,
                    )
                }
            })
            .flatten();
        entry.set_item("geometry", geometry)?;
        let legs = PyList::empty(py);
        legs.append(entry)?;
        dict.set_item("legs", legs)?;
        Ok(dict.unbind())
    }

    /// A stop's coordinates and street snap, for drawing walk legs.
    pub(super) fn stop_walk_endpoint(&self, stop: StopIdx) -> Option<((f64, f64), Snap)> {
        let streets = self.streets.as_ref()?;
        let snap = streets.stop_snap(stop)?;
        let feed_stop = &self.feed.stops[stop.0 as usize];
        Some(((feed_stop.latitude?, feed_stop.longitude?), snap))
    }

    /// The walked street path between two snapped points, as WKB.
    pub(super) fn walk_wkb<'py>(
        &self,
        py: Python<'py>,
        from_point: (f64, f64),
        from_snap: &Snap,
        to_point: (f64, f64),
        to_snap: &Snap,
    ) -> Option<Bound<'py, PyBytes>> {
        let streets = self.streets.as_ref()?;
        let (path, _) = streets.walk_path(from_point, from_snap, to_point, to_snap)?;
        Some(wkb_line_string(py, &path))
    }

    /// The public form of a stop identifier: raw for a single feed,
    /// `<feed_index>:<id>` when several feeds are merged.
    pub(super) fn public_stop_id(&self, stop: StopIdx) -> String {
        let stop = &self.feed.stops[stop.0 as usize];
        self.public_id(stop.feed, &stop.id)
    }

    pub(super) fn public_id(&self, feed: cafein_gtfs::FeedIndex, id: &str) -> String {
        if self.feed.feed_count > 1 {
            format!("{feed}:{id}")
        } else {
            id.to_owned()
        }
    }

    /// A route-index penalty mask for the McRAPTOR diverse search:
    /// `u32::MAX` for banned public route ids (their lines are skipped),
    /// the given seconds for penalized ids (added to a ride's effective
    /// arrival, clamped below the ban sentinel), 0 otherwise. Unknown ids
    /// are ignored and a ban wins over a penalty. Empty in, empty out —
    /// the engine reads a missing index as free.
    pub(super) fn route_penalty_mask(
        &self,
        banned_routes: &[String],
        route_penalties: &[(String, u64)],
    ) -> Vec<u32> {
        if banned_routes.is_empty() && route_penalties.is_empty() {
            return Vec::new();
        }
        let banned: std::collections::HashSet<&str> =
            banned_routes.iter().map(String::as_str).collect();
        // A penalty is clamped below the ban sentinel; the `u64` boundary type
        // absorbs large or accumulated Python values without overflowing.
        let penalties: std::collections::HashMap<&str, u32> = route_penalties
            .iter()
            .map(|(id, seconds)| (id.as_str(), (*seconds).min((u32::MAX - 1) as u64) as u32))
            .collect();
        self.feed
            .routes
            .iter()
            .map(|route| {
                let id = self.public_id(route.feed, &route.id);
                if banned.contains(id.as_str()) {
                    u32::MAX
                } else {
                    penalties.get(id.as_str()).copied().unwrap_or(0)
                }
            })
            .collect()
    }

    /// Resolves a stop identifier. In merged networks the feed-qualified
    /// form (`<feed_index>:<stop_id>`) takes precedence, so a raw stop_id
    /// that happens to look like another stop's qualified id must itself
    /// be fully qualified.
    pub(super) fn resolve_stop(&self, stop_id: &str) -> PyResult<StopIdx> {
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

    pub(super) fn journey_to_dict(
        &self,
        py: Python<'_>,
        journey: &Journey,
        walks: Option<&WalkMaps>,
        ends: Option<&CoordinateEnds>,
        geometries: bool,
        transfers: &Transfers,
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
                    let geometry = ends.filter(|_| geometries).and_then(|ends| {
                        let (point, snap) = self.stop_walk_endpoint(to_stop)?;
                        self.walk_wkb(py, ends.origin, &ends.origin_snap, point, &snap)
                    });
                    entry.set_item("geometry", geometry)?;
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
                    // Look up the walked distance in the same transfer set
                    // routing relaxed (the ULTRA set for point-destination
                    // time routes, else the closure), so an ULTRA-only
                    // shortcut leg still reports its metres. Transfers are
                    // deduplicated per stop pair, so the one edge found is
                    // the one routing relaxed.
                    let meters = transfers
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
                    let geometry = geometries
                        .then(|| {
                            let (from_point, from_snap) = self.stop_walk_endpoint(from_stop)?;
                            let (to_point, to_snap) = self.stop_walk_endpoint(to_stop)?;
                            self.walk_wkb(py, from_point, &from_snap, to_point, &to_snap)
                        })
                        .flatten();
                    entry.set_item("geometry", geometry)?;
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
                    let geometry = ends.filter(|_| geometries).and_then(|ends| {
                        let (point, snap) = self.stop_walk_endpoint(from_stop)?;
                        self.walk_wkb(py, point, &snap, ends.destination, &ends.destination_snap)
                    });
                    entry.set_item("geometry", geometry)?;
                }
            }
            legs.append(entry)?;
        }
        dict.set_item("legs", legs)?;
        Ok(dict.unbind())
    }
}
