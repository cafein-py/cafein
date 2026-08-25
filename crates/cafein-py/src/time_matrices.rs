//! The travel-time matrices and window percentiles, stop and
//! point forms.

use super::*;

/// An arrive-by cell: the winner's departure, round, achieved
/// arrival, and composing access stop.
type ArriveByCell = Option<(u32, u32, u32, u32)>;

#[pymethods]
impl TransportNetwork {
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
    /// max_transfers : int (optional, default: 7)
    ///     Maximum number of transfers between rides.
    /// router : str (optional, default: "auto")
    ///     The routing engine: ``"raptor"``, or ``"tbtr"`` to build a
    ///     TBTR day engine (view + reduced trip-transfer set) for the
    ///     date and fan the origins out over it. The results are
    ///     identical; TBTR trades a per-date precompute for faster
    ///     scans. The precomputed set covers same-stop transfers;
    ///     installed footpaths relax at query time, RAPTOR-style.
    ///     ``"auto"`` runs on TBTR when the cached set
    ///     (``compute_tbtr_transfers``) matches the date and no whole-day
    ///     ULTRA set serves the query door-to-door, else on RAPTOR.
    /// walking_speed_kmph, max_walking_time, max_snap_distance : float
    ///     Bound the door-to-door walking of the ``"raptor"`` router under a
    ///     whole-day ULTRA set (defaults 3.6 km/h, 7200 s, 1600 m); ignored
    ///     otherwise.
    ///
    /// With a whole-day ULTRA set the ``"raptor"`` router reaches every stop
    /// door-to-door from each origin (the origin treated as its coordinate,
    /// unrestricted initial/intermediate/final walking); a stop that has no
    /// coordinate or is off the walking network keeps the closure
    /// board-at-origin search for its row. The ``"tbtr"`` router keeps the
    /// closure.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     A ``(len(from_stops), stop_count)`` uint32 array of travel
    ///     times in seconds; row order follows `from_stops`, column
    ///     order follows ``stops``. Unreachable pairs hold the maximum
    ///     uint32 value (4294967295).
    #[pyo3(signature = (from_stops, date, departure, max_transfers = 7, router = "auto", exclude_routes = vec![], exclude_trips = vec![], exclude_stops = vec![], walking_speed_kmph = 3.6, max_walking_time = 7200.0, max_snap_distance = 1600.0))]
    #[allow(clippy::too_many_arguments)]
    fn travel_time_matrix<'py>(
        &self,
        py: Python<'py>,
        from_stops: Vec<String>,
        date: &str,
        departure: &str,
        max_transfers: u8,
        router: &str,
        exclude_routes: Vec<String>,
        exclude_trips: Vec<String>,
        exclude_stops: Vec<String>,
        walking_speed_kmph: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
    ) -> PyResult<Bound<'py, PyArray2<u32>>> {
        let (rows, departure) = self.resolved_time_rows(
            py,
            &from_stops,
            date,
            departure,
            max_transfers,
            router,
            &exclude_routes,
            &exclude_trips,
            &exclude_stops,
            walking_speed_kmph,
            max_walking_time,
            max_snap_distance,
        )?;
        let stop_count = self.build.timetable.stop_count() as usize;
        let count = rows.len();
        let mut flat = Vec::with_capacity(count * stop_count);
        for row in rows {
            flat.extend(row.into_iter().map(|arrival| match arrival {
                Some(arrival) => arrival - departure,
                None => u32::MAX,
            }));
        }
        flat.into_pyarray(py)
            .reshape([count, stop_count])
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }
}

impl TransportNetwork {
    /// One-to-all arrival rows for `from_stops` — the shared engine
    /// dispatch behind the time matrix and the accessibility
    /// aggregations, so their costs are identical by construction.
    /// Returns the rows plus the parsed departure the arrivals are
    /// relative to.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn resolved_time_rows(
        &self,
        py: Python<'_>,
        from_stops: &[String],
        date: &str,
        departure: &str,
        max_transfers: u8,
        router: &str,
        exclude_routes: &[String],
        exclude_trips: &[String],
        exclude_stops: &[String],
        walking_speed_kmph: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
    ) -> PyResult<(Vec<Vec<Option<u32>>>, u32)> {
        if !matches!(router, "auto" | "raptor" | "tbtr") {
            return Err(invalid_router(router));
        }
        let origins: Vec<StopIdx> = from_stops
            .iter()
            .map(|stop| self.resolve_stop(stop))
            .collect::<PyResult<_>>()?;
        let departure = parse_time(departure)?;
        let active_services = self.active_services(date)?;
        let active_services_previous = self.active_services_previous(date)?;
        let exclusions = self.exclusion_masks(exclude_routes, exclude_trips, exclude_stops)?;
        // The RAPTOR router routes door-to-door under a whole-day ULTRA set,
        // but only for origins that snap; validate the walking speed up front
        // (error creation needs the GIL that `allow_threads` releases) and only
        // when at least one origin is usable — a matrix whose origins all fall
        // back ignores the walking options, as `travel_times_from_stop` does.
        let ultra_usable = router != "tbtr"
            && exclusions.is_none()
            && self.ultra_active()
            && self.streets.as_ref().is_some_and(|streets| {
                origins.iter().any(|&origin| {
                    self.stop_coordinate(origin).is_some_and(|coordinate| {
                        streets
                            .snap(coordinate.0, coordinate.1, max_snap_distance)
                            .is_some()
                    })
                })
            });
        let ultra_speed = if ultra_usable {
            Some(validated_walking_speed(
                walking_speed_kmph,
                max_walking_time,
                max_snap_distance,
            )?)
        } else {
            None
        };
        // Auto prefers the door-to-door ULTRA path over an engine switch —
        // it must never trade semantics for speed.
        let router = if ultra_usable {
            "raptor"
        } else {
            self.resolve_time_router(router, date, exclusions.is_some())?
        };
        let rows = py.allow_threads(|| {
            let ticker = crate::logging::ProgressTicker::new("travel_time_matrix", origins.len());
            let tick = || ticker.tick();
            if router == "tbtr" {
                let engine = self.tbtr_engine(
                    &self.transfers,
                    date,
                    &active_services,
                    &active_services_previous,
                );
                let accesses: Vec<Vec<(StopIdx, u32)>> =
                    origins.iter().map(|&origin| vec![(origin, 0)]).collect();
                engine.one_to_all_many_with_progress(
                    departure,
                    &accesses,
                    max_transfers,
                    crate::logging::progress_hook(&ticker, &tick),
                )
            } else if let Some(speed) = ultra_speed {
                let streets = self
                    .streets
                    .as_ref()
                    .expect("ultra_speed is set only when a street network is installed");
                self.ultra_matrix_rows(
                    streets,
                    &origins,
                    departure,
                    &active_services,
                    &active_services_previous,
                    max_transfers,
                    speed,
                    max_walking_time,
                    max_snap_distance,
                )
            } else {
                let requests: Vec<Request> = origins
                    .iter()
                    .map(|&origin| Request {
                        departure,
                        access: vec![(origin, 0)],
                        egress: Vec::new(),
                        active_services: active_services.clone(),
                        active_services_previous: active_services_previous.clone(),
                        max_transfers,
                        exclusions: exclusions.clone(),
                    })
                    .collect();
                Raptor.one_to_all_many_with_progress(
                    &self.build.timetable,
                    &self.transfers,
                    &requests,
                    crate::logging::progress_hook(&ticker, &tick),
                )
            }
        });
        Ok((rows, departure))
    }
}

#[pymethods]
impl TransportNetwork {
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
    /// max_transfers : int (optional, default: 7)
    ///     Maximum number of transfers between rides.
    /// router : str (optional, default: "auto")
    ///     ``"raptor"``, or ``"tbtr"`` to answer the window over a TBTR
    ///     day engine — the same reduced trip-transfer set the
    ///     single-departure matrix uses (reusing the cached set from
    ///     ``compute_tbtr_transfers`` when present). The results are
    ///     identical. ``"auto"`` runs on TBTR when the cached set matches
    ///     the date, else on RAPTOR.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     A ``(len(from_stops), stop_count, len(percentiles))`` uint32
    ///     array of travel times in seconds; unreachable percentiles
    ///     hold the maximum uint32 value (4294967295).
    #[pyo3(signature = (from_stops, date, departure, window, percentiles, max_transfers = 7, router = "auto", exclude_routes = vec![], exclude_trips = vec![], exclude_stops = vec![]))]
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
        router: &str,
        exclude_routes: Vec<String>,
        exclude_trips: Vec<String>,
        exclude_stops: Vec<String>,
    ) -> PyResult<Bound<'py, PyArray3<u32>>> {
        validate_window(window, &percentiles)?;
        let exclusions = self.exclusion_masks(&exclude_routes, &exclude_trips, &exclude_stops)?;
        let router = self.resolve_time_router(router, date, exclusions.is_some())?;
        let origins: Vec<StopIdx> = from_stops
            .iter()
            .map(|stop| self.resolve_stop(stop))
            .collect::<PyResult<_>>()?;
        let departure = parse_time(departure)?;
        let active_services = self.active_services(date)?;
        let active_services_previous = self.active_services_previous(date)?;
        let requests: Vec<Request> = origins
            .into_iter()
            .map(|origin| Request {
                departure,
                access: vec![(origin, 0)],
                egress: Vec::new(),
                active_services: active_services.clone(),
                active_services_previous: active_services_previous.clone(),
                max_transfers,
                exclusions: exclusions.clone(),
            })
            .collect();
        let stop_count = self.build.timetable.stop_count() as usize;
        let flat: Vec<u32> = py.allow_threads(|| {
            let ticker = crate::logging::ProgressTicker::new("travel_time_matrix", requests.len());
            let tick = || ticker.tick();
            if router == "tbtr" {
                let engine = self.tbtr_engine(
                    &self.transfers,
                    date,
                    &active_services,
                    &active_services_previous,
                );
                engine
                    .percentile_matrix_with_progress(
                        &requests,
                        window,
                        &percentiles,
                        crate::logging::progress_hook(&ticker, &tick),
                    )
                    .concat()
            } else {
                Raptor
                    .percentile_matrix_with_progress(
                        &self.build.timetable,
                        &self.transfers,
                        &requests,
                        window,
                        &percentiles,
                        crate::logging::progress_hook(&ticker, &tick),
                    )
                    .concat()
            }
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
    ///
    /// ``router="auto"`` (the default) runs on TBTR when the cached time
    /// transfer set (``compute_tbtr_transfers``) matches the date, else on
    /// RAPTOR; explicit values pick the engine directly.
    #[pyo3(signature = (origins, destinations, date, departure, window, percentiles, max_transfers = 7, router = "auto", exclude_routes = vec![], exclude_trips = vec![], exclude_stops = vec![], walking_speed_kmph = 3.6, max_walking_time = 7200.0, max_snap_distance = 1600.0))]
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
        router: &str,
        exclude_routes: Vec<String>,
        exclude_trips: Vec<String>,
        exclude_stops: Vec<String>,
        walking_speed_kmph: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
    ) -> PyResult<Py<PyDict>> {
        if !matches!(router, "auto" | "raptor" | "tbtr") {
            return Err(invalid_router(router));
        }
        let exclusions = self.exclusion_masks(&exclude_routes, &exclude_trips, &exclude_stops)?;
        let router = self.resolve_time_router(router, date, exclusions.is_some())?;
        let streets = self.installed_streets()?;
        validate_window(window, &percentiles)?;
        let speed =
            validated_walking_speed(walking_speed_kmph, max_walking_time, max_snap_distance)?;
        validate_points(&origins)?;
        validate_points(&destinations)?;
        let departure = parse_time(departure)?;
        let active_services = self.active_services(date)?;
        let active_services_previous = self.active_services_previous(date)?;
        let destination_count = destinations.len();
        let (flat, unsnapped_from, unsnapped_to) = py.allow_threads(|| {
            // One stop-search pass links both the origins (access) and the
            // destinations (egress); see StreetNetwork::link_pointsets.
            let mut linked = streets.link_pointsets(
                &[&origins[..], &destinations[..]],
                speed,
                max_walking_time,
                max_snap_distance,
            );
            let destination_links = linked.pop().unwrap();
            let origin_links = linked.pop().unwrap();
            let unsnapped_from = unsnapped(&origin_links);
            let unsnapped_to = unsnapped(&destination_links);
            let requests: Vec<Request> = origin_links
                .iter()
                .map(|links| Request {
                    departure,
                    access: request_offsets(links.as_deref().unwrap_or(&[])),
                    egress: Vec::new(),
                    active_services: active_services.clone(),
                    active_services_previous: active_services_previous.clone(),
                    max_transfers,
                    exclusions: exclusions.clone(),
                })
                .collect();
            let egress = egress_tables(&destination_links);
            let ticker = crate::logging::ProgressTicker::new("travel_time_matrix", requests.len());
            let tick = || ticker.tick();
            let mut flat = if router == "tbtr" {
                let engine = self.tbtr_engine(
                    self.exclusion_transfers(&exclusions),
                    date,
                    &active_services,
                    &active_services_previous,
                );
                engine
                    .percentile_matrix_to_points_with_progress(
                        &requests,
                        &egress,
                        window,
                        &percentiles,
                        crate::logging::progress_hook(&ticker, &tick),
                    )
                    .concat()
            } else {
                Raptor
                    .percentile_matrix_to_points_with_progress(
                        &self.build.timetable,
                        self.exclusion_transfers(&exclusions),
                        &requests,
                        &egress,
                        window,
                        &percentiles,
                        crate::logging::progress_hook(&ticker, &tick),
                    )
                    .concat()
            };
            // A direct walk is departure-independent, so it caps every
            // percentile of a cell's distribution alike.
            let walk = streets.walk_matrix(
                &origins,
                &destinations,
                speed,
                max_walking_time,
                max_snap_distance,
            );
            let planes = percentiles.len();
            for (origin, row) in walk.iter().enumerate() {
                for (point, cell) in row.iter().enumerate() {
                    if let Some((walk_seconds, _)) = cell {
                        let base = (origin * destination_count + point) * planes;
                        for value in &mut flat[base..base + planes] {
                            *value = (*value).min(*walk_seconds);
                        }
                    }
                }
            }
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
    /// max_transfers : int (optional, default: 7)
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
    ///
    /// ``router="auto"`` (the default) runs on TBTR when the cached time
    /// transfer set (``compute_tbtr_transfers``) matches the date, else on
    /// RAPTOR; explicit values pick the engine directly.
    #[pyo3(signature = (origins, destinations, date, departure, max_transfers = 7, router = "auto", exclude_routes = vec![], exclude_trips = vec![], exclude_stops = vec![], walking_speed_kmph = 3.6, max_walking_time = 7200.0, max_snap_distance = 1600.0))]
    #[allow(clippy::too_many_arguments)]
    fn travel_time_matrix_from_points(
        &self,
        py: Python<'_>,
        origins: Vec<(f64, f64)>,
        destinations: Vec<(f64, f64)>,
        date: &str,
        departure: &str,
        max_transfers: u8,
        router: &str,
        exclude_routes: Vec<String>,
        exclude_trips: Vec<String>,
        exclude_stops: Vec<String>,
        walking_speed_kmph: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
    ) -> PyResult<Py<PyDict>> {
        if !matches!(router, "auto" | "raptor" | "tbtr") {
            return Err(invalid_router(router));
        }
        let exclusions = self.exclusion_masks(&exclude_routes, &exclude_trips, &exclude_stops)?;
        let router = self.resolve_time_router(router, date, exclusions.is_some())?;
        let streets = self.installed_streets()?;
        let speed =
            validated_walking_speed(walking_speed_kmph, max_walking_time, max_snap_distance)?;
        validate_points(&origins)?;
        validate_points(&destinations)?;
        let departure = parse_time(departure)?;
        let active_services = self.active_services(date)?;
        let active_services_previous = self.active_services_previous(date)?;
        let stop_count = self.build.timetable.stop_count() as usize;
        let destination_count = destinations.len();
        let (flat, unsnapped_from, unsnapped_to) = py.allow_threads(|| {
            // One stop-search pass links both the origins (access) and the
            // destinations (egress); see StreetNetwork::link_pointsets.
            let mut linked = streets.link_pointsets(
                &[&origins[..], &destinations[..]],
                speed,
                max_walking_time,
                max_snap_distance,
            );
            let destination_links = linked.pop().unwrap();
            let origin_links = linked.pop().unwrap();
            let unsnapped_from = unsnapped(&origin_links);
            let unsnapped_to = unsnapped(&destination_links);
            let requests: Vec<Request> = origin_links
                .iter()
                .map(|links| Request {
                    departure,
                    access: request_offsets(links.as_deref().unwrap_or(&[])),
                    egress: Vec::new(),
                    active_services: active_services.clone(),
                    active_services_previous: active_services_previous.clone(),
                    max_transfers,
                    exclusions: exclusions.clone(),
                })
                .collect();
            let egress = egress_tables(&destination_links);
            let ticker = crate::logging::ProgressTicker::new("travel_time_matrix", requests.len());
            let tick = || ticker.tick();
            let rows: Vec<Vec<Option<u32>>> = if router == "tbtr" {
                let engine = self.tbtr_engine(
                    self.exclusion_transfers(&exclusions),
                    date,
                    &active_services,
                    &active_services_previous,
                );
                let accesses: Vec<Vec<(StopIdx, u32)>> = requests
                    .iter()
                    .map(|request| request.access.clone())
                    .collect();
                engine.one_to_all_many_with_progress(
                    departure,
                    &accesses,
                    max_transfers,
                    crate::logging::progress_hook(&ticker, &tick),
                )
            } else {
                Raptor.one_to_all_many_with_progress(
                    &self.build.timetable,
                    self.exclusion_transfers(&exclusions),
                    &requests,
                    crate::logging::progress_hook(&ticker, &tick),
                )
            };
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
            // Walking directly can beat transit; each cell keeps the
            // faster of the two (one street search per origin).
            let walk = streets.walk_matrix(
                &origins,
                &destinations,
                speed,
                max_walking_time,
                max_snap_distance,
            );
            for (origin, row) in walk.iter().enumerate() {
                for (point, cell) in row.iter().enumerate() {
                    if let Some((walk_seconds, _)) = cell {
                        let at = origin * destination_count + point;
                        flat[at] = flat[at].min(*walk_seconds);
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
}

#[pymethods]
impl TransportNetwork {
    /// The arrive-by stop-matrix core: one reverse one-to-all per
    /// destination stop fills a column — per cell the latest-departure
    /// journey arriving by the deadline (fewest rides, then earliest
    /// arrival, breaking ties), holding its own duration; the
    /// destination itself at 0. The reverse rides the closure — a
    /// whole-day ULTRA set is never claimed — and the Python wrapper
    /// owns axis validation and destination chunking. Internal until
    /// the arrive-by surface stabilises.
    #[pyo3(signature = (from_stops, to_stops, date, deadline, max_transfers = 7, exclude_routes = vec![], exclude_trips = vec![], exclude_stops = vec![]))]
    #[allow(clippy::too_many_arguments)]
    fn _arrive_by_time_matrix<'py>(
        &self,
        py: Python<'py>,
        from_stops: Vec<String>,
        to_stops: Vec<String>,
        date: &str,
        deadline: &str,
        max_transfers: u8,
        exclude_routes: Vec<String>,
        exclude_trips: Vec<String>,
        exclude_stops: Vec<String>,
    ) -> PyResult<Bound<'py, PyArray2<u32>>> {
        let origins: Vec<StopIdx> = from_stops
            .iter()
            .map(|stop| self.resolve_stop(stop))
            .collect::<PyResult<_>>()?;
        let destinations: Vec<StopIdx> = to_stops
            .iter()
            .map(|stop| self.resolve_stop(stop))
            .collect::<PyResult<_>>()?;
        let deadline = parse_time(deadline)?;
        let active_services = self.active_services(date)?;
        let active_services_previous = self.active_services_previous(date)?;
        let exclusions = self.exclusion_masks(&exclude_routes, &exclude_trips, &exclude_stops)?;
        let requests: Vec<Request> = destinations
            .iter()
            .map(|&destination| Request {
                departure: deadline,
                access: Vec::new(),
                egress: vec![(destination, 0)],
                active_services: active_services.clone(),
                active_services_previous: active_services_previous.clone(),
                max_transfers,
                exclusions: exclusions.clone(),
            })
            .collect();
        let destination_count = destinations.len();
        let flat: Vec<u32> = py.allow_threads(|| {
            let reversed = ReversedTransfers::build(&self.transfers);
            // Each run folds into its dense column inside the worker;
            // per-stop frontiers never accumulate across destinations.
            let columns: Vec<Vec<u32>> = reverse::reverse_one_to_all_fold(
                &self.build.timetable,
                &reversed,
                &requests,
                |column, states| {
                    let destination = destinations[column];
                    let excluded = exclusions
                        .as_deref()
                        .is_some_and(|excluded| excluded.excludes_stop(destination));
                    origins
                        .iter()
                        .map(|&origin| {
                            let mut best: Option<(u32, u32, u32)> = None;
                            for &(round, departure, achieved) in &states[origin.0 as usize] {
                                crate::options::arrive_by_winner(
                                    &mut best,
                                    (departure, round as u32, achieved),
                                );
                            }
                            if origin == destination && !excluded {
                                crate::options::arrive_by_winner(
                                    &mut best,
                                    (deadline, 0, deadline),
                                );
                            }
                            best.map_or(u32::MAX, |(departure, _, achieved)| achieved - departure)
                        })
                        .collect()
                },
            );
            let mut flat = vec![u32::MAX; origins.len() * destination_count];
            for (column, cells) in columns.iter().enumerate() {
                for (row, &cell) in cells.iter().enumerate() {
                    flat[row * destination_count + column] = cell;
                }
            }
            flat
        });
        flat.into_pyarray(py)
            .reshape([origins.len(), destination_count])
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// The windowed arrive-by stop-matrix core: deadlines profile at
    /// every minute mark in `[arrival, arrival + window)` from ONE
    /// reverse run per destination at the final mark — the frontier IS
    /// the deadline profile — and each cell holds nearest-rank
    /// percentiles of its per-mark durations (each mark's
    /// latest-departure journey's own duration; `u32::MAX` where a
    /// mark is unreachable). Internal.
    #[pyo3(signature = (from_stops, to_stops, date, arrival, window, percentiles, max_transfers = 7, exclude_routes = vec![], exclude_trips = vec![], exclude_stops = vec![]))]
    #[allow(clippy::too_many_arguments)]
    fn _arrive_by_time_percentiles<'py>(
        &self,
        py: Python<'py>,
        from_stops: Vec<String>,
        to_stops: Vec<String>,
        date: &str,
        arrival: &str,
        window: u32,
        percentiles: Vec<f64>,
        max_transfers: u8,
        exclude_routes: Vec<String>,
        exclude_trips: Vec<String>,
        exclude_stops: Vec<String>,
    ) -> PyResult<Bound<'py, PyArray3<u32>>> {
        validate_window(window, &percentiles)?;
        let origins: Vec<StopIdx> = from_stops
            .iter()
            .map(|stop| self.resolve_stop(stop))
            .collect::<PyResult<_>>()?;
        let destinations: Vec<StopIdx> = to_stops
            .iter()
            .map(|stop| self.resolve_stop(stop))
            .collect::<PyResult<_>>()?;
        let start = parse_time(arrival)?;
        let marks = crate::options::arrival_marks(start, window)?;
        let deadline = *marks.last().expect("a validated window holds a mark");
        let active_services = self.active_services(date)?;
        let active_services_previous = self.active_services_previous(date)?;
        let exclusions = self.exclusion_masks(&exclude_routes, &exclude_trips, &exclude_stops)?;
        let requests: Vec<Request> = destinations
            .iter()
            .map(|&destination| Request {
                departure: deadline,
                access: Vec::new(),
                egress: vec![(destination, 0)],
                active_services: active_services.clone(),
                active_services_previous: active_services_previous.clone(),
                max_transfers,
                exclusions: exclusions.clone(),
            })
            .collect();
        let destination_count = destinations.len();
        let planes = percentiles.len();
        let flat: Vec<u32> = py.allow_threads(|| {
            let reversed = ReversedTransfers::build(&self.transfers);
            let columns: Vec<Vec<u32>> = reverse::reverse_one_to_all_fold(
                &self.build.timetable,
                &reversed,
                &requests,
                |column, states| {
                    let destination = destinations[column];
                    let excluded = exclusions
                        .as_deref()
                        .is_some_and(|excluded| excluded.excludes_stop(destination));
                    let mut cells = Vec::with_capacity(origins.len() * planes);
                    for &origin in &origins {
                        let profile =
                            reverse::reverse_profile_states(&states[origin.0 as usize], &marks);
                        let mut durations: Vec<u32> = profile
                            .iter()
                            .zip(&marks)
                            .map(|(winners, &mark)| {
                                let mut best: Option<(u32, u32, u32)> = None;
                                for &(round, departure, achieved) in winners {
                                    crate::options::arrive_by_winner(
                                        &mut best,
                                        (departure, round as u32, achieved),
                                    );
                                }
                                if origin == destination && !excluded {
                                    crate::options::arrive_by_winner(&mut best, (mark, 0, mark));
                                }
                                best.map_or(u32::MAX, |(departure, _, achieved)| {
                                    achieved - departure
                                })
                            })
                            .collect();
                        durations.sort_unstable();
                        for &percentile in &percentiles {
                            cells.push(cafein_core::raptor::nearest_rank(&durations, percentile));
                        }
                    }
                    cells
                },
            );
            let mut flat = vec![u32::MAX; origins.len() * destination_count * planes];
            for (column, cells) in columns.iter().enumerate() {
                for (row, values) in cells.chunks(planes).enumerate() {
                    let base = (row * destination_count + column) * planes;
                    flat[base..base + planes].copy_from_slice(values);
                }
            }
            flat
        });
        flat.into_pyarray(py)
            .reshape([origins.len(), destination_count, planes])
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// The windowed arrive-by point-matrix core: destination links
    /// seed the final-mark reverse run; each cell composes the
    /// origin's access links over the per-mark profile winners beside
    /// the per-mark direct-walk candidate, and holds nearest-rank
    /// percentiles of the winners' own durations. Internal.
    #[pyo3(signature = (origins, destinations, date, arrival, window, percentiles, max_transfers = 7, exclude_routes = vec![], exclude_trips = vec![], exclude_stops = vec![], walking_speed_kmph = 3.6, max_walking_time = 7200.0, max_snap_distance = 1600.0))]
    #[allow(clippy::too_many_arguments)]
    fn _arrive_by_time_percentiles_from_points(
        &self,
        py: Python<'_>,
        origins: Vec<(f64, f64)>,
        destinations: Vec<(f64, f64)>,
        date: &str,
        arrival: &str,
        window: u32,
        percentiles: Vec<f64>,
        max_transfers: u8,
        exclude_routes: Vec<String>,
        exclude_trips: Vec<String>,
        exclude_stops: Vec<String>,
        walking_speed_kmph: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
    ) -> PyResult<Py<PyDict>> {
        validate_window(window, &percentiles)?;
        let exclusions = self.exclusion_masks(&exclude_routes, &exclude_trips, &exclude_stops)?;
        let streets = self.installed_streets()?;
        let speed =
            validated_walking_speed(walking_speed_kmph, max_walking_time, max_snap_distance)?;
        validate_points(&origins)?;
        validate_points(&destinations)?;
        let start = parse_time(arrival)?;
        let marks = crate::options::arrival_marks(start, window)?;
        let deadline = *marks.last().expect("a validated window holds a mark");
        let active_services = self.active_services(date)?;
        let active_services_previous = self.active_services_previous(date)?;
        let destination_count = destinations.len();
        let planes = percentiles.len();
        let (flat, unsnapped_from, unsnapped_to) = py.allow_threads(|| {
            let mut linked = streets.link_pointsets(
                &[&origins[..], &destinations[..]],
                speed,
                max_walking_time,
                max_snap_distance,
            );
            let destination_links = linked.pop().unwrap();
            let origin_links = linked.pop().unwrap();
            let unsnapped_from = unsnapped(&origin_links);
            let unsnapped_to = unsnapped(&destination_links);
            let requests: Vec<Request> = destination_links
                .iter()
                .map(|links| Request {
                    departure: deadline,
                    access: Vec::new(),
                    egress: request_offsets(links.as_deref().unwrap_or(&[])),
                    active_services: active_services.clone(),
                    active_services_previous: active_services_previous.clone(),
                    max_transfers,
                    exclusions: exclusions.clone(),
                })
                .collect();
            let access = egress_tables(&origin_links);
            let reversed = ReversedTransfers::build(&self.transfers);
            let walk = streets.walk_matrix(
                &origins,
                &destinations,
                speed,
                max_walking_time,
                max_snap_distance,
            );
            let columns: Vec<Vec<u32>> = reverse::reverse_one_to_all_fold(
                &self.build.timetable,
                &reversed,
                &requests,
                |column, states| {
                    let mut cells = Vec::with_capacity(access.len() * planes);
                    for (row, links) in access.iter().enumerate() {
                        type MarkWinners = Vec<Vec<(u16, u32, u32)>>;
                        let profiles: Vec<(u32, MarkWinners)> = links
                            .iter()
                            .map(|&(stop, seconds, _)| {
                                (
                                    seconds,
                                    reverse::reverse_profile_states(
                                        &states[stop.0 as usize],
                                        &marks,
                                    ),
                                )
                            })
                            .collect();
                        let mut durations: Vec<u32> = marks
                            .iter()
                            .enumerate()
                            .map(|(at, &mark)| {
                                let mut best: Option<(u32, u32, u32)> = None;
                                for (seconds, profile) in &profiles {
                                    for &(round, departure, achieved) in &profile[at] {
                                        let Some(composed) = departure.checked_sub(*seconds) else {
                                            continue;
                                        };
                                        crate::options::arrive_by_winner(
                                            &mut best,
                                            (composed, round as u32, achieved),
                                        );
                                    }
                                }
                                if let Some((walk_seconds, _)) = walk[row][column] {
                                    if let Some(placed) = mark.checked_sub(walk_seconds) {
                                        crate::options::arrive_by_winner(
                                            &mut best,
                                            (placed, 0, mark),
                                        );
                                    }
                                }
                                best.map_or(u32::MAX, |(departure, _, achieved)| {
                                    achieved - departure
                                })
                            })
                            .collect();
                        durations.sort_unstable();
                        for &percentile in &percentiles {
                            cells.push(cafein_core::raptor::nearest_rank(&durations, percentile));
                        }
                    }
                    cells
                },
            );
            let mut flat = vec![u32::MAX; origins.len() * destination_count * planes];
            for (column, cells) in columns.iter().enumerate() {
                for (row, values) in cells.chunks(planes).enumerate() {
                    let base = (row * destination_count + column) * planes;
                    flat[base..base + planes].copy_from_slice(values);
                }
            }
            (flat, unsnapped_from, unsnapped_to)
        });
        let result = PyDict::new(py);
        result.set_item(
            "matrix",
            flat.into_pyarray(py)
                .reshape([origins.len(), destination_count, planes])
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        )?;
        result.set_item("unsnapped_from", unsnapped_from.into_pyarray(py))?;
        result.set_item("unsnapped_to", unsnapped_to.into_pyarray(py))?;
        Ok(result.unbind())
    }

    /// The arrive-by point-matrix core: each destination's street
    /// links seed one reverse run; a cell composes the origin's access
    /// links over the returned states beside the direct-walk candidate
    /// placed to arrive exactly at the deadline, and holds the
    /// complete-journey order winner's own duration. Internal until
    /// the arrive-by surface stabilises.
    #[pyo3(signature = (origins, destinations, date, deadline, max_transfers = 7, exclude_routes = vec![], exclude_trips = vec![], exclude_stops = vec![], walking_speed_kmph = 3.6, max_walking_time = 7200.0, max_snap_distance = 1600.0))]
    #[allow(clippy::too_many_arguments)]
    fn _arrive_by_time_matrix_from_points(
        &self,
        py: Python<'_>,
        origins: Vec<(f64, f64)>,
        destinations: Vec<(f64, f64)>,
        date: &str,
        deadline: &str,
        max_transfers: u8,
        exclude_routes: Vec<String>,
        exclude_trips: Vec<String>,
        exclude_stops: Vec<String>,
        walking_speed_kmph: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
    ) -> PyResult<Py<PyDict>> {
        let exclusions = self.exclusion_masks(&exclude_routes, &exclude_trips, &exclude_stops)?;
        let streets = self.installed_streets()?;
        let speed =
            validated_walking_speed(walking_speed_kmph, max_walking_time, max_snap_distance)?;
        validate_points(&origins)?;
        validate_points(&destinations)?;
        let deadline = parse_time(deadline)?;
        let active_services = self.active_services(date)?;
        let active_services_previous = self.active_services_previous(date)?;
        let destination_count = destinations.len();
        let (flat, unsnapped_from, unsnapped_to) = py.allow_threads(|| {
            let mut linked = streets.link_pointsets(
                &[&origins[..], &destinations[..]],
                speed,
                max_walking_time,
                max_snap_distance,
            );
            let destination_links = linked.pop().unwrap();
            let origin_links = linked.pop().unwrap();
            let unsnapped_from = unsnapped(&origin_links);
            let unsnapped_to = unsnapped(&destination_links);
            let requests: Vec<Request> = destination_links
                .iter()
                .map(|links| Request {
                    departure: deadline,
                    access: Vec::new(),
                    egress: request_offsets(links.as_deref().unwrap_or(&[])),
                    active_services: active_services.clone(),
                    active_services_previous: active_services_previous.clone(),
                    max_transfers,
                    exclusions: exclusions.clone(),
                })
                .collect();
            let access = egress_tables(&origin_links);
            let reversed = ReversedTransfers::build(&self.transfers);
            let walk = streets.walk_matrix(
                &origins,
                &destinations,
                speed,
                max_walking_time,
                max_snap_distance,
            );
            // Each run folds into its dense column inside the worker;
            // per-stop frontiers never accumulate across destinations.
            let columns: Vec<Vec<u32>> = reverse::reverse_one_to_all_fold(
                &self.build.timetable,
                &reversed,
                &requests,
                |column, states| {
                    access
                        .iter()
                        .enumerate()
                        .map(|(row, links)| {
                            let mut best: Option<(u32, u32, u32)> = None;
                            for &(stop, seconds, _) in links {
                                for &(round, departure, achieved) in &states[stop.0 as usize] {
                                    let Some(composed) = departure.checked_sub(seconds) else {
                                        continue;
                                    };
                                    crate::options::arrive_by_winner(
                                        &mut best,
                                        (composed, round as u32, achieved),
                                    );
                                }
                            }
                            if let Some((walk_seconds, _)) = walk[row][column] {
                                if let Some(placed) = deadline.checked_sub(walk_seconds) {
                                    crate::options::arrive_by_winner(
                                        &mut best,
                                        (placed, 0, deadline),
                                    );
                                }
                            }
                            best.map_or(u32::MAX, |(departure, _, achieved)| achieved - departure)
                        })
                        .collect()
                },
            );
            let mut flat = vec![u32::MAX; origins.len() * destination_count];
            for (column, cells) in columns.iter().enumerate() {
                for (row, &cell) in cells.iter().enumerate() {
                    flat[row * destination_count + column] = cell;
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

    /// The arrive-by twin of `_time_matrix_with_access`: caller-supplied
    /// per-origin access tables and per-destination egress tables, one
    /// reverse run per destination at a single deadline. Returns, per
    /// cell, the complete-journey winner's departure clock
    /// (`departure`), its own duration (`matrix`), its round
    /// (`rounds`), and the access stop it composed through
    /// (`access_stop`), `2**32 - 1` where unreachable; the walking
    /// alternative is the caller's to fold, with the same
    /// latest-departure rule. Internal until the policy surface
    /// stabilises.
    #[pyo3(signature = (access_rows, egress_rows, date, deadline, max_transfers = 7, exclude_routes = vec![], exclude_trips = vec![], exclude_stops = vec![]))]
    #[allow(clippy::too_many_arguments)]
    fn _arrive_by_time_matrix_with_access(
        &self,
        py: Python<'_>,
        access_rows: Vec<Vec<(String, u32)>>,
        egress_rows: Vec<Vec<(String, u32)>>,
        date: &str,
        deadline: &str,
        max_transfers: u8,
        exclude_routes: Vec<String>,
        exclude_trips: Vec<String>,
        exclude_stops: Vec<String>,
    ) -> PyResult<Py<PyDict>> {
        let exclusions = self.exclusion_masks(&exclude_routes, &exclude_trips, &exclude_stops)?;
        let deadline = parse_time(deadline)?;
        let active_services = self.active_services(date)?;
        let active_services_previous = self.active_services_previous(date)?;
        let resolve = |rows: &[Vec<(String, u32)>]| {
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|(stop, seconds)| Ok((self.resolve_stop(stop)?, *seconds)))
                        .collect::<PyResult<Vec<_>>>()
                })
                .collect::<PyResult<Vec<_>>>()
        };
        let access = resolve(&access_rows)?;
        let egress = resolve(&egress_rows)?;
        let origin_count = access.len();
        let destination_count = egress.len();
        let (departures, durations, rounds, access_stops) = py.allow_threads(|| {
            let requests: Vec<Request> = egress
                .iter()
                .map(|links| Request {
                    departure: deadline,
                    access: Vec::new(),
                    egress: links.clone(),
                    active_services: active_services.clone(),
                    active_services_previous: active_services_previous.clone(),
                    max_transfers,
                    exclusions: exclusions.clone(),
                })
                .collect();
            let reversed = ReversedTransfers::build(&self.transfers);
            let egress_by_stop: Vec<std::collections::HashMap<u32, u32>> = egress
                .iter()
                .map(|links| links.iter().map(|&(stop, walk)| (stop.0, walk)).collect())
                .collect();
            // Each run folds into its dense column inside the worker;
            // per-stop frontiers never accumulate across destinations.
            let columns: Vec<Vec<ArriveByCell>> = reverse::reverse_one_to_all_fold(
                &self.build.timetable,
                &reversed,
                &requests,
                |column, states| {
                    access
                        .iter()
                        .map(|links| {
                            // The complete-journey winner beside the
                            // access stop it composed through.
                            let mut best: ArriveByCell = None;
                            let mut fold = |candidate: (u32, u32, u32, u32)| {
                                // Full-tuple ties break on the stop
                                // index, so the elected access stop —
                                // and the facility priced through it —
                                // is deterministic.
                                let wins = match best {
                                    None => true,
                                    Some(held) => {
                                        candidate.0 > held.0
                                            || (candidate.0 == held.0 && candidate.1 < held.1)
                                            || (candidate.0 == held.0
                                                && candidate.1 == held.1
                                                && candidate.2 < held.2)
                                            || (candidate.0 == held.0
                                                && candidate.1 == held.1
                                                && candidate.2 == held.2
                                                && candidate.3 < held.3)
                                    }
                                };
                                if wins {
                                    best = Some(candidate);
                                }
                            };
                            for &(stop, seconds) in links {
                                for &(round, departure, achieved) in &states[stop.0 as usize] {
                                    let Some(composed) = departure.checked_sub(seconds) else {
                                        continue;
                                    };
                                    fold((composed, round as u32, achieved, stop.0));
                                }
                                // A stop on both tables walks straight
                                // through: the zero-ride cell the forward
                                // matrix keeps too.
                                if let Some(&out) = egress_by_stop[column].get(&stop.0) {
                                    if let Some(through) = seconds
                                        .checked_add(out)
                                        .and_then(|total| deadline.checked_sub(total))
                                    {
                                        fold((through, 0, deadline, stop.0));
                                    }
                                }
                            }
                            best
                        })
                        .collect()
                },
            );
            let cells = origin_count * destination_count;
            let mut departures = vec![u32::MAX; cells];
            let mut durations = vec![u32::MAX; cells];
            let mut rounds = vec![u32::MAX; cells];
            let mut access_stops = vec![u32::MAX; cells];
            for (column, column_cells) in columns.iter().enumerate() {
                for (row, cell) in column_cells.iter().enumerate() {
                    if let Some((departure, round, achieved, stop)) = cell {
                        let at = row * destination_count + column;
                        departures[at] = *departure;
                        durations[at] = achieved - departure;
                        rounds[at] = *round;
                        access_stops[at] = *stop;
                    }
                }
            }
            (departures, durations, rounds, access_stops)
        });
        let result = PyDict::new(py);
        result.set_item(
            "departure",
            departures
                .into_pyarray(py)
                .reshape([origin_count, destination_count])
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        )?;
        result.set_item(
            "matrix",
            durations
                .into_pyarray(py)
                .reshape([origin_count, destination_count])
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        )?;
        result.set_item(
            "rounds",
            rounds
                .into_pyarray(py)
                .reshape([origin_count, destination_count])
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        )?;
        result.set_item(
            "access_stop",
            access_stops
                .into_pyarray(py)
                .reshape([origin_count, destination_count])
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        )?;
        Ok(result.unbind())
    }
}

impl TransportNetwork {
    /// A one-to-all arrival array as a `{public_stop_id: travel_time}` dict,
    /// travel time measured from `departure`; unreachable stops are absent.
    pub(super) fn arrivals_dict(
        &self,
        py: Python<'_>,
        arrivals: &[Option<u32>],
        departure: u32,
    ) -> PyResult<Py<PyDict>> {
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

    /// The RAPTOR one-to-all rows for a stop-origin travel-time matrix under a
    /// whole-day ULTRA set. Origins whose stop coordinate snaps route
    /// door-to-door — coordinate access, `time_transfers()` intermediate
    /// transfers, and one bounded `final_egress` walk folded into the row, all in
    /// parallel over origins; origins that cannot snap fall back to the closure
    /// board-at-origin search. Rows come back in the input origin order. Runs
    /// with the GIL released, so it uses the erasing `access_stops` (no
    /// `ValueError` construction) rather than `coordinate_links`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ultra_matrix_rows(
        &self,
        streets: &StreetNetwork,
        origins: &[StopIdx],
        departure: u32,
        active_services: &[bool],
        active_services_previous: &[bool],
        max_transfers: u8,
        speed: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
    ) -> Vec<Vec<Option<u32>>> {
        use rayon::prelude::*;
        // Partition into door-to-door usable (snappable coordinate) and closure
        // fallback, keeping each origin's input index for the merge.
        let mut usable: Vec<(usize, (f64, f64))> = Vec::new();
        let mut fallback: Vec<(usize, StopIdx)> = Vec::new();
        for (index, &origin) in origins.iter().enumerate() {
            match self.stop_coordinate(origin) {
                Some(coordinate)
                    if streets
                        .snap(coordinate.0, coordinate.1, max_snap_distance)
                        .is_some() =>
                {
                    usable.push((index, coordinate));
                }
                _ => fallback.push((index, origin)),
            }
        }
        let request = |access: Vec<(StopIdx, u32)>| Request {
            departure,
            access,
            egress: Vec::new(),
            active_services: active_services.to_vec(),
            active_services_previous: active_services_previous.to_vec(),
            max_transfers,
            // The door-to-door path runs only without exclusions.
            exclusions: None,
        };
        let mut rows: Vec<Vec<Option<u32>>> = vec![Vec::new(); origins.len()];
        if !usable.is_empty() {
            let requests: Vec<Request> = usable
                .iter()
                .map(|&(_, coordinate)| {
                    let access = streets
                        .access_stops(
                            coordinate.0,
                            coordinate.1,
                            speed,
                            max_walking_time,
                            max_snap_distance,
                        )
                        .unwrap_or_default();
                    request(request_offsets(&access))
                })
                .collect();
            let ticker = crate::logging::ProgressTicker::new("travel_time_matrix", requests.len());
            let tick = || ticker.tick();
            let mut usable_rows = Raptor.one_to_all_many_with_progress(
                &self.build.timetable,
                self.time_transfers(),
                &requests,
                crate::logging::progress_hook(&ticker, &tick),
            );
            // The bounded final-walk egress is origin-independent — build it
            // once and fold it into every usable origin's arrivals.
            let egress = self.final_egress(streets, speed, max_walking_time, max_snap_distance);
            usable_rows.par_iter_mut().for_each(|row| {
                self.fold_final_transfers(row, &egress);
            });
            for (row, &(index, _)) in usable_rows.into_iter().zip(&usable) {
                rows[index] = row;
            }
        }
        if !fallback.is_empty() {
            let requests: Vec<Request> = fallback
                .iter()
                .map(|&(_, origin)| request(vec![(origin, 0)]))
                .collect();
            let ticker = crate::logging::ProgressTicker::new("travel_time_matrix", requests.len());
            let tick = || ticker.tick();
            let fallback_rows = Raptor.one_to_all_many_with_progress(
                &self.build.timetable,
                &self.transfers,
                &requests,
                crate::logging::progress_hook(&ticker, &tick),
            );
            for (row, &(index, _)) in fallback_rows.into_iter().zip(&fallback) {
                rows[index] = row;
            }
        }
        rows
    }
}
