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
    /// With ``arrive_by`` the given time is the arrival deadline and the
    /// result is the latest-departure Pareto set — (latest departure,
    /// fewest rides), earliest arrival breaking ties — each journey
    /// leg-identical to the plain answer for the departure it discovers.
    /// The reverse run boards at the origin stop over the closure; a
    /// whole-day ULTRA set is never claimed. With ``arrive_by`` the
    /// window profiles arrival deadlines at its minute marks and the
    /// result is the deadline profile — the union of the marks'
    /// latest-departure Pareto sets.
    ///
    /// Returns
    /// -------
    /// list of dict
    ///     Without `window`, the Pareto set of journeys over (arrival
    ///     time, number of rides) leaving at the departure time; with it,
    ///     the departure-window profile. Each journey carries its legs;
    ///     times are seconds past the service day's start.
    #[pyo3(signature = (from_stop, to_stop, date, departure, max_transfers = 7, window = None, exclude_routes = vec![], exclude_trips = vec![], exclude_stops = vec![], walking_speed_kmph = 3.6, max_walking_time = 7200.0, max_snap_distance = 1600.0, geometries = true, arrive_by = false))]
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
        arrive_by: bool,
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
        if self.ultra_active() && exclusions.is_none() && !arrive_by {
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
                        arrive_by,
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
        if arrive_by {
            return self.reverse_route_request(py, &request, window, geometries);
        }
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
    /// With ``arrive_by`` the given time is the arrival deadline, as in
    /// ``route_between_stops``; the direct walking alternative is placed
    /// to arrive exactly at the deadline and takes its position in the
    /// (latest departure, fewest rides) order.
    ///
    /// Returns
    /// -------
    /// list of dict
    ///     Journeys as in ``route_between_stops``; arrivals include the
    ///     egress walk.
    #[pyo3(signature = (origin, destination, date, departure, max_transfers = 7, window = None, exclude_routes = vec![], exclude_trips = vec![], exclude_stops = vec![], walking_speed_kmph = 3.6, max_walking_time = 7200.0, max_snap_distance = 1600.0, geometries = true, arrive_by = false))]
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
        arrive_by: bool,
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
        let transfers = if arrive_by || request.exclusions.is_some() {
            &self.transfers
        } else {
            self.time_transfers()
        };
        let (journeys, profile_walks) = if arrive_by {
            self.reverse_journeys(&request, window, direct.map(|(seconds, _)| seconds))?
        } else {
            let journeys = match window {
                None => Raptor.route(&self.build.timetable, transfers, &request),
                Some(window) => {
                    Raptor.route_range(&self.build.timetable, transfers, &request, window)
                }
            };
            (journeys, Vec::new())
        };
        // The walking-only alternative's dominance depends on the
        // axis. Depart-at: both leave at the departure, so a journey
        // survives only when strictly faster than walking. Arrive-by
        // at one deadline: the walk leaves at deadline − walk with
        // zero rides, so it dominates every journey leaving no later.
        // Over an arrival window the walk competed inside every
        // mark's Pareto selection in the core, so the returned
        // journeys and walk placements are already exact — nothing is
        // filtered here.
        let windowed = arrive_by && window.is_some();
        let walk_departure = match direct {
            Some((walk_seconds, _)) if arrive_by => request.departure.checked_sub(walk_seconds),
            Some(_) => Some(request.departure),
            None => None,
        };
        let kept: Vec<&Journey> = journeys
            .iter()
            .filter(|journey| match (direct, walk_departure) {
                _ if windowed => true,
                (Some((walk_seconds, _)), _) if !arrive_by => {
                    journey.arrival - journey.departure < walk_seconds
                }
                (Some(_), Some(at)) => journey.departure > at,
                // A walk longer than the whole clock cannot be placed
                // to arrive by the deadline; it dominates nothing.
                _ => true,
            })
            .collect();
        let result = PyList::empty(py);
        if windowed {
            // Two departure-descending streams merge; equal departures
            // put the walk first (zero rides sorts first).
            let mut placements = profile_walks.iter().peekable();
            if let Some((_, meters)) = direct {
                for journey in &kept {
                    while placements
                        .peek()
                        .is_some_and(|&&(departure, _)| departure >= journey.departure)
                    {
                        let &(departure, arrival) = placements.next().expect("peeked");
                        result.append(self.walk_journey_dict(
                            py,
                            departure,
                            (arrival - departure, meters),
                            &ends,
                            geometries,
                        )?)?;
                    }
                    result.append(self.journey_to_dict(
                        py,
                        journey,
                        Some(&walks),
                        Some(&ends),
                        geometries,
                        transfers,
                    )?)?;
                }
                for &(departure, arrival) in placements {
                    result.append(self.walk_journey_dict(
                        py,
                        departure,
                        (arrival - departure, meters),
                        &ends,
                        geometries,
                    )?)?;
                }
            } else {
                for journey in &kept {
                    result.append(self.journey_to_dict(
                        py,
                        journey,
                        Some(&walks),
                        Some(&ends),
                        geometries,
                        transfers,
                    )?)?;
                }
            }
            return Ok(result.unbind());
        }
        let walk_entry = direct
            .filter(|_| !kept.iter().any(|journey| journey.rides() == 0))
            .zip(walk_departure);
        // The forward walk leads (equal departure, zero rides); the
        // single-deadline arrive-by walk takes its place in the
        // (departure desc, rides asc) order.
        let mut walk_pending = walk_entry.filter(|_| arrive_by);
        if let Some((walk, at)) = walk_entry.filter(|_| !arrive_by) {
            result.append(self.walk_journey_dict(py, at, walk, &ends, geometries)?)?;
        }
        for journey in kept {
            if let Some((walk, at)) = walk_pending {
                if journey.departure <= at {
                    result.append(self.walk_journey_dict(py, at, walk, &ends, geometries)?)?;
                    walk_pending = None;
                }
            }
            result.append(self.journey_to_dict(
                py,
                journey,
                Some(&walks),
                Some(&ends),
                geometries,
                transfers,
            )?)?;
        }
        if let Some((walk, at)) = walk_pending {
            result.append(self.walk_journey_dict(py, at, walk, &ends, geometries)?)?;
        }
        Ok(result.unbind())
    }

    /// The street-policy time matrix core: per-origin pre-reduced access
    /// arrays and per-destination pre-reduced egress arrays in, the flat
    /// origins-major seconds out (`None` = unreachable). One `Request` per
    /// origin through the rayon fan-out, then the egress fold per
    /// destination — the points-matrix shape minus the walking machinery,
    /// which the policy path replaces with its reductions. Stays on the
    /// full transfer closure like every policy query. Internal until the
    /// policy surface stabilises.
    #[pyo3(signature = (access_rows, egress_rows, date, departure, max_transfers,
                        exclude_routes = vec![], exclude_trips = vec![], exclude_stops = vec![],
                        transfer_mode = None, router = "auto"))]
    #[allow(clippy::too_many_arguments)]
    fn _time_matrix_with_access(
        &self,
        py: Python<'_>,
        access_rows: Vec<Vec<(String, u32)>>,
        egress_rows: Vec<Vec<(String, u32)>>,
        date: &str,
        departure: &str,
        max_transfers: u8,
        exclude_routes: Vec<String>,
        exclude_trips: Vec<String>,
        exclude_stops: Vec<String>,
        transfer_mode: Option<(String, f64)>,
        router: &str,
    ) -> PyResult<Vec<Vec<Option<u32>>>> {
        if transfer_mode
            .as_ref()
            .is_some_and(|binding| !crate::network::walking_class(&binding.0))
            && !exclude_stops.is_empty()
        {
            // A rental-bearing merged edge hides its pickup and drop
            // stops inside the token; a walking-class set's tokens have
            // no interiors, so its endpoints are exclusion-checked like
            // any transfer edge and the combination is sound.
            return Err(PyValueError::new_err(
                "stop exclusions do not combine with street_policy \
                 transfers= yet; a rental transfer's interior stops are \
                 not exclusion-aware",
            ));
        }
        let departure = parse_time(departure)?;
        let exclusions = self.exclusion_masks(&exclude_routes, &exclude_trips, &exclude_stops)?;
        let active_services = self.active_services(date)?;
        let active_services_previous = self.active_services_previous(date)?;
        let mut requests = Vec::with_capacity(access_rows.len());
        for row in &access_rows {
            let offsets = row
                .iter()
                .map(|(stop, seconds)| Ok((self.resolve_stop(stop)?, *seconds)))
                .collect::<PyResult<Vec<_>>>()?;
            requests.push(Request {
                departure,
                access: offsets,
                egress: Vec::new(),
                active_services: active_services.clone(),
                active_services_previous: active_services_previous.clone(),
                max_transfers,
                exclusions: exclusions.clone(),
            });
        }
        let mut egress = Vec::with_capacity(egress_rows.len());
        for row in &egress_rows {
            egress.push(
                row.iter()
                    .map(|(stop, seconds)| Ok((self.resolve_stop(stop)?, *seconds)))
                    .collect::<PyResult<Vec<_>>>()?,
            );
        }
        let relaxed = self.policy_transfers(transfer_mode.as_ref())?;
        let router = self.resolve_time_router(router, date, exclusions.is_some())?;
        let matrix = py.allow_threads(|| {
            let rows = if router == "tbtr" {
                let engine =
                    self.tbtr_engine(relaxed, date, &active_services, &active_services_previous);
                let accesses: Vec<Vec<(StopIdx, u32)>> = requests
                    .iter()
                    .map(|request| request.access.clone())
                    .collect();
                engine.one_to_all_many(departure, &accesses, max_transfers)
            } else {
                Raptor.one_to_all_many(&self.build.timetable, relaxed, &requests)
            };
            rows.iter()
                .map(|arrivals| {
                    egress
                        .iter()
                        .map(|links| {
                            let mut best = u32::MAX;
                            for &(stop, seconds) in links {
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
                            (best != u32::MAX).then(|| best - departure)
                        })
                        .collect()
                })
                .collect()
        });
        Ok(matrix)
    }

    /// The Pareto journeys from pre-reduced street access and egress
    /// offsets — the street-policy reconstruction path. The offsets arrive
    /// from the time-only reduction (`_reduced_street_offsets`); the run
    /// relaxes the full transfer closure like every policy query, and the
    /// journeys come back as ``route_between_stops``-shaped dicts whose
    /// access and egress legs carry no distances — Python rebuilds them
    /// from the kept `StreetChoice` tokens. Internal until the policy
    /// surface stabilises.
    #[pyo3(signature = (access, egress, date, departure, max_transfers,
                        exclude_routes = vec![], exclude_trips = vec![], exclude_stops = vec![],
                        geometries = true, transfer_mode = None))]
    #[allow(clippy::too_many_arguments)]
    fn _route_with_access(
        &self,
        py: Python<'_>,
        access: Vec<(String, u32)>,
        egress: Vec<(String, u32)>,
        date: &str,
        departure: &str,
        max_transfers: u8,
        exclude_routes: Vec<String>,
        exclude_trips: Vec<String>,
        exclude_stops: Vec<String>,
        geometries: bool,
        transfer_mode: Option<(String, f64)>,
    ) -> PyResult<Py<PyList>> {
        if transfer_mode
            .as_ref()
            .is_some_and(|binding| !crate::network::walking_class(&binding.0))
            && !exclude_stops.is_empty()
        {
            // A rental-bearing merged edge hides its pickup and drop
            // stops inside the token; a walking-class set's tokens have
            // no interiors, so its endpoints are exclusion-checked like
            // any transfer edge and the combination is sound.
            return Err(PyValueError::new_err(
                "stop exclusions do not combine with street_policy \
                 transfers= yet; a rental transfer's interior stops are \
                 not exclusion-aware",
            ));
        }
        let resolve = |offsets: &[(String, u32)]| {
            offsets
                .iter()
                .map(|(stop, seconds)| Ok((self.resolve_stop(stop)?, *seconds)))
                .collect::<PyResult<Vec<_>>>()
        };
        let request = Request {
            departure: parse_time(departure)?,
            access: resolve(&access)?,
            egress: resolve(&egress)?,
            active_services: self.active_services(date)?,
            active_services_previous: self.active_services_previous(date)?,
            max_transfers,
            exclusions: self.exclusion_masks(&exclude_routes, &exclude_trips, &exclude_stops)?,
        };
        let relaxed = self.policy_transfers(transfer_mode.as_ref())?;
        let journeys = py.allow_threads(|| Raptor.route(&self.build.timetable, relaxed, &request));
        let result = PyList::empty(py);
        for journey in &journeys {
            result.append(self.journey_to_dict(py, journey, None, None, geometries, relaxed)?)?;
        }
        Ok(result.unbind())
    }

    /// The arrive-by twin of `_route_with_access`: caller-supplied
    /// access and egress tables through the reverse engine at a single
    /// deadline — journeys latest-departure-first, the forward replay
    /// re-riding the exact same tables. The walking alternative and
    /// every street leg stay the caller's to place and rebuild.
    /// Internal until the policy surface stabilises.
    #[pyo3(signature = (access, egress, date, deadline, max_transfers = 7, exclude_routes = vec![], exclude_trips = vec![], exclude_stops = vec![], geometries = false))]
    #[allow(clippy::too_many_arguments)]
    fn _reverse_route_with_access(
        &self,
        py: Python<'_>,
        access: Vec<(String, u32)>,
        egress: Vec<(String, u32)>,
        date: &str,
        deadline: &str,
        max_transfers: u8,
        exclude_routes: Vec<String>,
        exclude_trips: Vec<String>,
        exclude_stops: Vec<String>,
        geometries: bool,
    ) -> PyResult<Py<PyList>> {
        let resolve = |offsets: &[(String, u32)]| {
            offsets
                .iter()
                .map(|(stop, seconds)| Ok((self.resolve_stop(stop)?, *seconds)))
                .collect::<PyResult<Vec<_>>>()
        };
        let request = Request {
            departure: parse_time(deadline)?,
            access: resolve(&access)?,
            egress: resolve(&egress)?,
            active_services: self.active_services(date)?,
            active_services_previous: self.active_services_previous(date)?,
            max_transfers,
            exclusions: self.exclusion_masks(&exclude_routes, &exclude_trips, &exclude_stops)?,
        };
        self.reverse_route_request(py, &request, None, geometries)
    }

    /// Earliest arrivals from pre-reduced street access offsets — the
    /// street-policy path. The offsets arrive from the time-only reduction
    /// (`_reduced_street_offsets`), and the run relaxes the full transfer
    /// closure: the ULTRA shortcut set models *walking* egress, so policy
    /// queries stay off it until multimodal ULTRA arrives. Internal until
    /// the policy surface stabilises.
    #[pyo3(signature = (access, date, departure, max_transfers, router = "auto", transfer_mode = None, exclude_routes = vec![], exclude_trips = vec![], exclude_stops = vec![], workers=None))]
    #[allow(clippy::too_many_arguments)]
    fn _travel_times_with_access(
        &self,
        py: Python<'_>,
        access: Vec<(String, u32)>,
        date: &str,
        departure: &str,
        max_transfers: u8,
        router: &str,
        transfer_mode: Option<(String, f64)>,
        exclude_routes: Vec<String>,
        exclude_trips: Vec<String>,
        exclude_stops: Vec<String>,
        workers: Option<usize>,
    ) -> PyResult<Py<PyDict>> {
        if transfer_mode
            .as_ref()
            .is_some_and(|binding| !crate::network::walking_class(&binding.0))
            && !exclude_stops.is_empty()
        {
            // A rental-bearing merged edge hides its pickup and drop
            // stops inside the token; a walking-class set's tokens have
            // no interiors, so its endpoints are exclusion-checked like
            // any transfer edge and the combination is sound.
            return Err(PyValueError::new_err(
                "stop exclusions do not combine with street_policy \
                 transfers= yet; a rental transfer's interior stops are \
                 not exclusion-aware",
            ));
        }
        let excluded =
            !(exclude_routes.is_empty() && exclude_trips.is_empty() && exclude_stops.is_empty());
        let exclusions = if excluded {
            self.exclusion_masks(&exclude_routes, &exclude_trips, &exclude_stops)?
        } else {
            None
        };
        let departure = parse_time(departure)?;
        let offsets = access
            .iter()
            .map(|(stop, seconds)| Ok((self.resolve_stop(stop)?, *seconds)))
            .collect::<PyResult<Vec<_>>>()?;
        let active_services = self.active_services(date)?;
        let active_services_previous = self.active_services_previous(date)?;
        // The cached whole-day set is timetable-only, so it serves the
        // merged transfer binding as-is; the resolver rejects an
        // explicit trip-based engine beside effective exclusions.
        let router = self.resolve_time_router(router, date, excluded)?;
        let arrivals = match router {
            "raptor" => {
                let request = Request {
                    departure,
                    access: offsets,
                    egress: Vec::new(),
                    active_services,
                    active_services_previous,
                    max_transfers,
                    exclusions,
                };
                let relaxed = self.policy_transfers(transfer_mode.as_ref())?;
                py.allow_threads(|| {
                    crate::workers::with_workers("travel_times_with_access", workers, || {
                        Raptor.one_to_all(&self.build.timetable, relaxed, &request)
                    })
                })
            }
            // The engine-neutrality arm: the same reduced array through the
            // trip-based engine, for the RAPTOR/TBTR equality tests —
            // merged (unclosed) sets included, whose shadowed arrivals
            // both engines now relax exactly.
            "tbtr" => {
                let relaxed = self.policy_transfers(transfer_mode.as_ref())?;
                py.allow_threads(|| {
                    crate::workers::with_workers("travel_times_with_access", workers, || {
                        self.tbtr_engine(relaxed, date, &active_services, &active_services_previous)
                            .one_to_all(departure, &offsets, max_transfers)
                    })
                })
            }
            other => return Err(invalid_router(other)),
        };
        self.arrivals_dict(py, &arrivals, departure)
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
    /// With ``arrive_by`` the given time is the arrival deadline and the
    /// coordinate becomes the **destination**: the result maps every
    /// origin stop to the travel time of its latest-departure journey
    /// arriving there by the deadline — each journey's own duration.
    /// Stops within the walking cutoff appear with their walking time,
    /// the walk placed to arrive exactly at the deadline. A whole-day
    /// ULTRA set is never claimed by a reverse run.
    #[pyo3(signature = (origin, date, departure, max_transfers = 7, exclude_routes = vec![], exclude_trips = vec![], exclude_stops = vec![], walking_speed_kmph = 3.6, max_walking_time = 7200.0, max_snap_distance = 1600.0, arrive_by = false))]
    #[allow(clippy::too_many_arguments)]
    fn travel_times_from_coordinate(
        &self,
        py: Python<'_>,
        origin: (f64, f64),
        date: &str,
        departure: &str,
        max_transfers: u8,
        exclude_routes: Vec<String>,
        exclude_trips: Vec<String>,
        exclude_stops: Vec<String>,
        walking_speed_kmph: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
        arrive_by: bool,
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
        let exclusions = self.exclusion_masks(&exclude_routes, &exclude_trips, &exclude_stops)?;
        if arrive_by {
            // The coordinate is the destination: its street links serve
            // as the egress, and one reverse run maps every origin stop
            // to its latest-departure journey.
            let links = request_offsets(&access);
            let request = Request {
                departure,
                access: Vec::new(),
                egress: links.clone(),
                active_services: self.active_services(date)?,
                active_services_previous: self.active_services_previous(date)?,
                max_transfers,
                exclusions,
            };
            return self.reverse_times_dict(py, &request, &links);
        }
        let request = Request {
            departure,
            access: request_offsets(&access),
            egress: Vec::new(),
            active_services: self.active_services(date)?,
            active_services_previous: self.active_services_previous(date)?,
            max_transfers,
            exclusions: exclusions.clone(),
        };
        // Under a whole-day ULTRA set the intermediate transfers use the
        // shortcuts and a bounded final walk (`<= max_walking_time`) reaches
        // the remaining stops; otherwise this is the closure, tau-direct search
        // (`time_transfers` is the closure then, and the fold is skipped).
        // Exclusions keep the closure and skip the shortcut fold.
        let mut arrivals = Raptor.one_to_all(
            &self.build.timetable,
            self.exclusion_transfers(&exclusions),
            &request,
        );
        if self.ultra_active() && exclusions.is_none() {
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
    /// With ``arrive_by`` the given time is the arrival deadline and the
    /// stop becomes the **destination**: the result maps every origin
    /// stop to the travel time of its latest-departure journey arriving
    /// there by the deadline — each journey's own duration, the
    /// destination itself at 0. The reverse run relaxes the closure; a
    /// whole-day ULTRA set is never claimed.
    #[pyo3(signature = (from_stop, date, departure, max_transfers = 7, exclude_routes = vec![], exclude_trips = vec![], exclude_stops = vec![], walking_speed_kmph = 3.6, max_walking_time = 7200.0, max_snap_distance = 1600.0, arrive_by = false))]
    #[allow(clippy::too_many_arguments)]
    fn travel_times_from_stop(
        &self,
        py: Python<'_>,
        from_stop: &str,
        date: &str,
        departure: &str,
        max_transfers: u8,
        exclude_routes: Vec<String>,
        exclude_trips: Vec<String>,
        exclude_stops: Vec<String>,
        walking_speed_kmph: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
        arrive_by: bool,
    ) -> PyResult<Py<PyDict>> {
        let origin = self.resolve_stop(from_stop)?;
        let departure = parse_time(departure)?;
        let exclusions = self.exclusion_masks(&exclude_routes, &exclude_trips, &exclude_stops)?;
        if arrive_by {
            let request = Request {
                departure,
                access: Vec::new(),
                egress: vec![(origin, 0)],
                active_services: self.active_services(date)?,
                active_services_previous: self.active_services_previous(date)?,
                max_transfers,
                exclusions,
            };
            return self.reverse_times_dict(py, &request, &[(origin, 0)]);
        }
        // With a whole-day ULTRA set, treat the origin stop as its coordinate
        // and reach every stop door-to-door (coordinate access, ULTRA
        // intermediate transfers, one final walk bounded by max_walking_time);
        // otherwise board at the origin stop and relax the closure (today's
        // behaviour). Exclusions keep the closure.
        if self.ultra_active() && exclusions.is_none() {
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
            exclusions,
        };
        let arrivals = Raptor.one_to_all(&self.build.timetable, &self.transfers, &request);
        self.arrivals_dict(py, &arrivals, departure)
    }
}
