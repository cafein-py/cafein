//! The multicriteria queries: one-pair routes, frontier matrices,
//! and the flat tables.

use super::*;

#[pymethods]
impl TransportNetwork {
    /// The exact time × emissions Pareto set between two stops — the
    /// exhaustive oracle behind the frontier machinery.
    ///
    /// Considers every boardable trip, with gram labels quantized to a
    /// microgram; orders of magnitude slower than the routers: meant for
    /// verifying frontiers and inspecting true Pareto sets at
    /// sampled-pair scale, never for bulk computation. Trips without a
    /// resolved emission factor are skipped — they can never sit on an
    /// emissions frontier. Requires installed trip distances.
    ///
    /// Returns
    /// -------
    /// list of (int, float, int)
    ///     ``(arrival, grams, rides)`` per frontier point, sorted by
    ///     arrival; ``rides`` is the fewest transit legs achieving the
    ///     point.
    #[pyo3(signature = (origin, destination, date, departure, factors, max_transfers = 7))]
    #[allow(clippy::too_many_arguments)]
    fn pareto_oracle(
        &self,
        py: Python<'_>,
        origin: &str,
        destination: &str,
        date: &str,
        departure: &str,
        factors: Vec<(String, f64)>,
        max_transfers: u8,
    ) -> PyResult<Vec<(u32, f64, u32)>> {
        let Some(geometry) = &self.geometry else {
            return Err(PyValueError::new_err(
                "no trip distances installed; build the network with trip distances enabled",
            ));
        };
        let origin = self.resolve_stop(origin)?;
        let destination = self.resolve_stop(destination)?;
        let mut per_trip = vec![f64::NAN; self.build.timetable.trip_count() as usize];
        for (trip_id, factor) in &factors {
            if let Some(&trip) = self.trips_by_public_id.get(trip_id) {
                per_trip[trip.0 as usize] = *factor;
            }
        }
        let departure = parse_time(departure)?;
        let active_services = self.active_services(date)?;
        let active_services_previous = self.active_services_previous(date)?;
        let points = py.allow_threads(|| {
            let view = DayView::for_date(
                &self.build.timetable,
                &active_services,
                &active_services_previous,
            );
            exhaustive::pareto_oracle(
                &view,
                &self.build.timetable,
                &self.transfers,
                geometry,
                &per_trip,
                departure,
                &[(origin, 0)],
                &[(destination, 0)],
                max_transfers,
            )
        });
        Ok(points
            .into_iter()
            .map(|point| (point.arrival, point.grams, point.rides))
            .collect())
    }

    /// Multicriteria journeys between two stops: the Pareto set over
    /// (arrival, emissions bucket) — with a window, over (departure,
    /// arrival, emissions bucket).
    ///
    /// Emissions enter the search as per-trip factors over precomputed
    /// cumulative distances; labels within `bucket` grams of each other
    /// count as equal, so the returned set is exact on arrivals and
    /// within a bucket-sized band on emissions. Trips without a
    /// resolved factor are skipped — journeys riding them can never sit
    /// on an emissions frontier. Requires installed trip distances.
    ///
    /// ``router`` picks the engine: McRAPTOR (``"raptor"``) answers
    /// immediately; McTBTR (``"tbtr"``) precomputes the day's
    /// multicriteria transfer set first — slower for one pair, built
    /// for batch reuse — and returns the same journeys. ``"auto"``
    /// (the default) runs on McTBTR when the cached set
    /// (``compute_mctbtr_transfers``) matches the query's date and
    /// factors and the query asks nothing McTBTR cannot answer, else
    /// on McRAPTOR.
    ///
    /// Returns
    /// -------
    /// list of dict
    ///     Journeys shaped as in ``route_between_stops``.
    #[pyo3(signature = (from_stop, to_stop, date, departure, factors, window = None, max_transfers = 7, bucket = 25.0, router = "auto", walking_speed_kmph = 3.6, max_walking_time = 7200.0, max_snap_distance = 1600.0, geometries = true, slack = 0.0, max_options = None, banned_routes = vec![], route_penalties = vec![], max_slower = None))]
    #[allow(clippy::too_many_arguments)]
    fn mc_route_between_stops(
        &self,
        py: Python<'_>,
        from_stop: &str,
        to_stop: &str,
        date: &str,
        departure: &str,
        factors: Vec<(String, f64)>,
        window: Option<u32>,
        max_transfers: u8,
        bucket: f64,
        router: &str,
        walking_speed_kmph: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
        geometries: bool,
        slack: f64,
        max_options: Option<usize>,
        banned_routes: Vec<String>,
        route_penalties: Vec<(String, u64)>,
        max_slower: Option<u32>,
    ) -> PyResult<Py<PyList>> {
        if !bucket.is_finite() || bucket <= 0.0 {
            return Err(PyValueError::new_err(
                "bucket must be a positive number of grams",
            ));
        }
        if !matches!(router, "auto" | "raptor" | "tbtr") {
            return Err(invalid_router(router));
        }
        if !slack.is_finite() || slack < 0.0 {
            return Err(PyValueError::new_err(
                "slack must be a non-negative number of seconds",
            ));
        }
        if matches!(max_options, Some(0)) {
            return Err(PyValueError::new_err(
                "max_options must be a positive integer",
            ));
        }
        if slack > 0.0 && router == "tbtr" {
            return Err(PyValueError::new_err(
                "relaxed candidates (slack > 0) require router='raptor'",
            ));
        }
        if (!banned_routes.is_empty() || !route_penalties.is_empty()) && router == "tbtr" {
            return Err(PyValueError::new_err(
                "route bans/penalties (diverse candidates) require router='raptor'",
            ));
        }
        if max_slower.is_some()
            && (slack > 0.0 || !banned_routes.is_empty() || !route_penalties.is_empty())
        {
            return Err(PyValueError::new_err(
                "max_slower requires strict pareto candidates",
            ));
        }
        let Some(geometry) = &self.geometry else {
            return Err(PyValueError::new_err(
                "no trip distances installed; build the network with trip distances enabled",
            ));
        };
        let origin = self.resolve_stop(from_stop)?;
        let destination = self.resolve_stop(to_stop)?;
        let mut per_trip = vec![f64::NAN; self.build.timetable.trip_count() as usize];
        for (trip_id, factor) in &factors {
            if let Some(&trip) = self.trips_by_public_id.get(trip_id) {
                per_trip[trip.0 as usize] = *factor;
            }
        }
        // Under a McULTRA set matching this query's factors, route door-to-door
        // between the two stops' coordinates, so the set's unrestricted
        // intermediate walking is paired with a full street-graph initial and
        // final walk — the McRAPTOR analogue of ULTRA `route_between_stops`. The
        // set covers only the intermediate transfers, so it needs those endpoint
        // searches; explicit TBTR and a factor mismatch keep today's
        // board-at-origin closure routing (`"auto"` reroutes and resolves on
        // the coordinate path, whose engines share the McULTRA set).
        if router != "tbtr"
            && !std::ptr::eq(
                self.emissions_transfers(factor_fingerprint(&per_trip)),
                &self.transfers,
            )
        {
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
                    return self.mc_route_between_coordinates(
                        py,
                        from_xy,
                        to_xy,
                        date,
                        departure,
                        factors,
                        window,
                        max_transfers,
                        bucket,
                        walking_speed_kmph,
                        max_walking_time,
                        max_snap_distance,
                        geometries,
                        slack,
                        max_options,
                        banned_routes,
                        route_penalties,
                        max_slower,
                        router,
                    );
                }
            }
        }
        let router = self.resolve_mc_router(
            router,
            date,
            &per_trip,
            slack > 0.0 || !banned_routes.is_empty() || !route_penalties.is_empty(),
        )?;
        let request = Request {
            departure: parse_time(departure)?,
            access: vec![(origin, 0)],
            egress: vec![(destination, 0)],
            active_services: self.active_services(date)?,
            active_services_previous: self.active_services_previous(date)?,
            max_transfers,
        };
        let penalty_mask = self.route_penalty_mask(&banned_routes, &route_penalties);
        let journeys = py.allow_threads(|| {
            if router == "tbtr" {
                let engine = self.mctbtr_engine(
                    &self.transfers,
                    geometry,
                    &per_trip,
                    date,
                    &request.active_services,
                    &request.active_services_previous,
                );
                return match window {
                    None => engine.route(&request, bucket, max_slower),
                    Some(window) => engine.route_range(&request, window, bucket, max_slower),
                };
            }
            let view = DayView::for_date(
                &self.build.timetable,
                &request.active_services,
                &request.active_services_previous,
            );
            let slack = slack.round() as u32;
            match window {
                None => mcraptor::route(
                    &view,
                    &self.build.timetable,
                    &self.transfers,
                    geometry,
                    &per_trip,
                    &request,
                    bucket,
                    slack,
                    max_options,
                    &penalty_mask,
                    max_slower,
                ),
                Some(window) => mcraptor::route_range(
                    &view,
                    &self.build.timetable,
                    &self.transfers,
                    geometry,
                    &per_trip,
                    &request,
                    window,
                    bucket,
                    slack,
                    max_options,
                    &penalty_mask,
                    max_slower,
                ),
            }
        });
        let result = PyList::empty(py);
        for journey in &journeys {
            result.append(self.journey_to_dict(
                py,
                journey,
                None,
                None,
                geometries,
                &self.transfers,
            )?)?;
        }
        Ok(result.unbind())
    }

    /// Multicriteria door-to-door journeys between two coordinates —
    /// the McRAPTOR counterpart of ``route_between_coordinates``.
    ///
    /// Walking access, egress, the walking-only journey, and the
    /// walk-domination rule behave exactly as in
    /// ``route_between_coordinates``; the candidate set is the Pareto
    /// set over (departure, arrival, emissions bucket) as in
    /// ``mc_route_between_stops``. The zero-emission walking-only
    /// journey anchors the clean end: it dominates every journey that
    /// rides yet arrives no earlier than walking out at that journey's
    /// own departure would.
    ///
    /// Returns
    /// -------
    /// list of dict
    ///     Journeys shaped as in ``route_between_coordinates``.
    ///
    /// ``router="auto"`` (the default) runs on McTBTR when the cached
    /// multicriteria transfer set (``compute_mctbtr_transfers``) matches
    /// the query's date and factors and the query asks nothing McTBTR
    /// cannot answer, else on McRAPTOR; explicit values pick the engine
    /// directly.
    #[pyo3(signature = (origin, destination, date, departure, factors, window = None, max_transfers = 7, bucket = 25.0, walking_speed_kmph = 3.6, max_walking_time = 7200.0, max_snap_distance = 1600.0, geometries = true, slack = 0.0, max_options = None, banned_routes = vec![], route_penalties = vec![], max_slower = None, router = "auto"))]
    #[allow(clippy::too_many_arguments)]
    fn mc_route_between_coordinates(
        &self,
        py: Python<'_>,
        origin: (f64, f64),
        destination: (f64, f64),
        date: &str,
        departure: &str,
        factors: Vec<(String, f64)>,
        window: Option<u32>,
        max_transfers: u8,
        bucket: f64,
        walking_speed_kmph: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
        geometries: bool,
        slack: f64,
        max_options: Option<usize>,
        banned_routes: Vec<String>,
        route_penalties: Vec<(String, u64)>,
        max_slower: Option<u32>,
        router: &str,
    ) -> PyResult<Py<PyList>> {
        if !bucket.is_finite() || bucket <= 0.0 {
            return Err(PyValueError::new_err(
                "bucket must be a positive number of grams",
            ));
        }
        if !matches!(router, "auto" | "raptor" | "tbtr") {
            return Err(invalid_router(router));
        }
        if !slack.is_finite() || slack < 0.0 {
            return Err(PyValueError::new_err(
                "slack must be a non-negative number of seconds",
            ));
        }
        if matches!(max_options, Some(0)) {
            return Err(PyValueError::new_err(
                "max_options must be a positive integer",
            ));
        }
        if router == "tbtr"
            && (slack > 0.0 || !banned_routes.is_empty() || !route_penalties.is_empty())
        {
            return Err(PyValueError::new_err(
                "route slacks, bans and penalties require router='raptor'",
            ));
        }
        if max_slower.is_some()
            && (slack > 0.0 || !banned_routes.is_empty() || !route_penalties.is_empty())
        {
            return Err(PyValueError::new_err(
                "max_slower requires strict pareto candidates",
            ));
        }
        let Some(geometry) = &self.geometry else {
            return Err(PyValueError::new_err(
                "no trip distances installed; build the network with trip distances enabled",
            ));
        };
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
        let mut per_trip = vec![f64::NAN; self.build.timetable.trip_count() as usize];
        for (trip_id, factor) in &factors {
            if let Some(&trip) = self.trips_by_public_id.get(trip_id) {
                per_trip[trip.0 as usize] = *factor;
            }
        }
        let request = Request {
            departure: parse_time(departure)?,
            access: request_offsets(&access),
            egress: request_offsets(&egress),
            active_services: self.active_services(date)?,
            active_services_previous: self.active_services_previous(date)?,
            max_transfers,
        };
        // The walking-only alternative, exactly as in
        // route_between_coordinates: zero emissions, available at
        // every departure, dominating whatever rides without arriving
        // earlier.
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
        // Door-to-door emissions: relax the McULTRA set for the intermediate
        // transfers when one is installed for this factor configuration; the
        // access/egress and direct walk above stay unchanged. Both engines
        // share that set, so `"auto"` resolves on the cache alone.
        let router = self.resolve_mc_router(
            router,
            date,
            &per_trip,
            slack > 0.0 || !banned_routes.is_empty() || !route_penalties.is_empty(),
        )?;
        let intermediate = self.emissions_transfers(factor_fingerprint(&per_trip));
        let slack = slack.round() as u32;
        let penalty_mask = self.route_penalty_mask(&banned_routes, &route_penalties);
        let journeys = py.allow_threads(|| {
            if router == "tbtr" {
                let engine = self.mctbtr_engine(
                    intermediate,
                    geometry,
                    &per_trip,
                    date,
                    &request.active_services,
                    &request.active_services_previous,
                );
                return match window {
                    None => engine.route(&request, bucket, max_slower),
                    Some(window) => engine.route_range(&request, window, bucket, max_slower),
                };
            }
            let view = DayView::for_date(
                &self.build.timetable,
                &request.active_services,
                &request.active_services_previous,
            );
            match window {
                None => mcraptor::route(
                    &view,
                    &self.build.timetable,
                    intermediate,
                    geometry,
                    &per_trip,
                    &request,
                    bucket,
                    slack,
                    max_options,
                    &penalty_mask,
                    max_slower,
                ),
                Some(window) => mcraptor::route_range(
                    &view,
                    &self.build.timetable,
                    intermediate,
                    geometry,
                    &per_trip,
                    &request,
                    window,
                    bucket,
                    slack,
                    max_options,
                    &penalty_mask,
                    max_slower,
                ),
            }
        });
        let kept: Vec<&Journey> = journeys
            .iter()
            .filter(|journey| match direct {
                Some((walk_seconds, _)) => journey.arrival - journey.departure < walk_seconds,
                None => true,
            })
            .collect();
        let result = PyList::empty(py);
        if let Some(walk) = direct.filter(|&(walk_seconds, _)| {
            walk_within_band(walk_seconds, request.departure, &journeys, max_slower)
        }) {
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
                // The same set the route relaxed, so transfer legs report the
                // McULTRA walk distance rather than the closure's.
                intermediate,
            )?)?;
        }
        Ok(result.unbind())
    }

    /// Batched multicriteria Pareto frontiers between stops: per
    /// (origin, destination) cell, the journeys of
    /// ``mc_route_between_stops`` with strict pareto candidates, from
    /// one window profile per origin, in parallel with the GIL
    /// released. Routes over the closure footpaths (board at the
    /// origin stop); a McULTRA emissions set upgrades only the
    /// coordinate variant.
    ///
    /// Returns
    /// -------
    /// list of list of list of dict
    ///     ``result[i][j]`` is the journey list from ``from_stops[i]``
    ///     to ``to_stops[j]``, shaped as in ``mc_route_between_stops``.
    ///
    /// ``router="auto"`` (the default) runs on McTBTR when the cached
    /// multicriteria transfer set (``compute_mctbtr_transfers``) matches
    /// the query's date and factors and the query asks nothing McTBTR
    /// cannot answer, else on McRAPTOR; explicit values pick the engine
    /// directly.
    #[pyo3(signature = (from_stops, to_stops, date, departure, factors, window, max_transfers = 7, bucket = 25.0, geometries = false, max_slower = None, router = "auto"))]
    #[allow(clippy::too_many_arguments)]
    fn mc_frontier_matrix(
        &self,
        py: Python<'_>,
        from_stops: Vec<String>,
        to_stops: Vec<String>,
        date: &str,
        departure: &str,
        factors: Vec<(String, f64)>,
        window: u32,
        max_transfers: u8,
        bucket: f64,
        geometries: bool,
        max_slower: Option<u32>,
        router: &str,
    ) -> PyResult<Py<PyList>> {
        if !bucket.is_finite() || bucket <= 0.0 {
            return Err(PyValueError::new_err(
                "bucket must be a positive number of grams",
            ));
        }
        if !matches!(router, "auto" | "raptor" | "tbtr") {
            return Err(invalid_router(router));
        }
        if window == 0 {
            return Err(PyValueError::new_err(
                "window must be a positive number of seconds",
            ));
        }
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
        let destinations: Vec<StopIdx> = to_stops
            .iter()
            .map(|stop| self.resolve_stop(stop))
            .collect::<PyResult<_>>()?;
        let mut per_trip = vec![f64::NAN; self.build.timetable.trip_count() as usize];
        for (trip_id, factor) in &factors {
            if let Some(&trip) = self.trips_by_public_id.get(trip_id) {
                per_trip[trip.0 as usize] = *factor;
            }
        }
        let router = self.resolve_mc_router(router, date, &per_trip, false)?;
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
            })
            .collect();
        let rows = py.allow_threads(|| {
            if router == "tbtr" {
                let engine = self.mctbtr_engine(
                    &self.transfers,
                    geometry,
                    &per_trip,
                    date,
                    &active_services,
                    &active_services_previous,
                );
                return engine.frontier_matrix(
                    &requests,
                    &destinations,
                    &[],
                    false,
                    destinations.len(),
                    window,
                    bucket,
                    max_slower,
                );
            }
            let view = DayView::for_date(
                &self.build.timetable,
                &active_services,
                &active_services_previous,
            );
            mcraptor::frontier_matrix(
                &view,
                &self.build.timetable,
                &self.transfers,
                geometry,
                &per_trip,
                &requests,
                &destinations,
                &[],
                false,
                destinations.len(),
                window,
                bucket,
                max_slower,
            )
        });
        let result = PyList::empty(py);
        for row in &rows {
            let cells = PyList::empty(py);
            for cell in row {
                let journeys = PyList::empty(py);
                for journey in cell {
                    journeys.append(self.journey_to_dict(
                        py,
                        journey,
                        None,
                        None,
                        geometries,
                        &self.transfers,
                    )?)?;
                }
                cells.append(journeys)?;
            }
            result.append(cells)?;
        }
        Ok(result.unbind())
    }

    /// ``mc_frontier_matrix`` between coordinate points, linked through
    /// the street network like the point matrices; each cell matches
    /// ``mc_route_between_coordinates`` with strict pareto candidates —
    /// including the walking-only journey and its domination rule
    /// (transit journeys the walk beats are dropped per cell).
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``journeys`` — ``result[i][j]`` journey lists;
    ///     ``unsnapped_from`` / ``unsnapped_to`` — indices of points
    ///     off the walking network (their cells stay empty).
    ///
    /// ``router="auto"`` (the default) runs on McTBTR when the cached
    /// multicriteria transfer set (``compute_mctbtr_transfers``) matches
    /// the query's date and factors and the query asks nothing McTBTR
    /// cannot answer, else on McRAPTOR; explicit values pick the engine
    /// directly.
    #[pyo3(signature = (origins, destinations, date, departure, factors, window, max_transfers = 7, bucket = 25.0, walking_speed_kmph = 3.6, max_walking_time = 7200.0, max_snap_distance = 1600.0, geometries = false, max_slower = None, router = "auto"))]
    #[allow(clippy::too_many_arguments)]
    fn mc_frontier_matrix_from_points(
        &self,
        py: Python<'_>,
        origins: Vec<(f64, f64)>,
        destinations: Vec<(f64, f64)>,
        date: &str,
        departure: &str,
        factors: Vec<(String, f64)>,
        window: u32,
        max_transfers: u8,
        bucket: f64,
        walking_speed_kmph: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
        geometries: bool,
        max_slower: Option<u32>,
        router: &str,
    ) -> PyResult<Py<PyDict>> {
        if !bucket.is_finite() || bucket <= 0.0 {
            return Err(PyValueError::new_err(
                "bucket must be a positive number of grams",
            ));
        }
        if !matches!(router, "auto" | "raptor" | "tbtr") {
            return Err(invalid_router(router));
        }
        if window == 0 {
            return Err(PyValueError::new_err(
                "window must be a positive number of seconds",
            ));
        }
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
        let streets = self.installed_streets()?;
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
        let router = self.resolve_mc_router(router, date, &per_trip, false)?;
        let departure = parse_time(departure)?;
        let active_services = self.active_services(date)?;
        let active_services_previous = self.active_services_previous(date)?;
        // The same intermediate set the one-pair coordinate route relaxes:
        // the McULTRA emissions set matching these factors, else the closure.
        let intermediate = self.emissions_transfers(factor_fingerprint(&per_trip));
        let (rows, walk, origin_links, destination_links, unsnapped_from, unsnapped_to) = self
            .point_frontier_rows(
                py,
                streets,
                intermediate,
                geometry,
                &per_trip,
                &origins,
                &destinations,
                date,
                departure,
                window,
                max_transfers,
                bucket,
                max_slower,
                router,
                speed,
                max_walking_time,
                max_snap_distance,
                &active_services,
                &active_services_previous,
            );
        let origin_snaps: Vec<Option<Snap>> = origins
            .iter()
            .map(|&(lat, lon)| streets.snap(lat, lon, max_snap_distance))
            .collect();
        let destination_snaps: Vec<Option<Snap>> = destinations
            .iter()
            .map(|&(lat, lon)| streets.snap(lat, lon, max_snap_distance))
            .collect();
        let cells = PyList::empty(py);
        for (i, row) in rows.iter().enumerate() {
            let row_list = PyList::empty(py);
            for (j, cell) in row.iter().enumerate() {
                let cell_list = PyList::empty(py);
                // The walking-only alternative and its domination rule,
                // exactly as the one-pair coordinate route.
                let direct = if origins[i] == destinations[j] {
                    Some((0, 0.0))
                } else {
                    walk[i][j]
                };
                let ends = match (origin_snaps[i], destination_snaps[j]) {
                    (Some(origin_snap), Some(destination_snap)) => Some(CoordinateEnds {
                        origin: origins[i],
                        origin_snap,
                        destination: destinations[j],
                        destination_snap,
                    }),
                    _ => None,
                };
                let banded_walk = direct.filter(|&(walk_seconds, _)| {
                    walk_within_band(walk_seconds, departure, cell, max_slower)
                });
                if let (Some(walk), Some(ends)) = (banded_walk, ends.as_ref()) {
                    cell_list
                        .append(self.walk_journey_dict(py, departure, walk, ends, geometries)?)?;
                }
                let walks = WalkMaps::new(
                    origin_links[i].as_deref().unwrap_or(&[]),
                    destination_links[j].as_deref().unwrap_or(&[]),
                );
                for journey in cell {
                    if let Some((walk_seconds, _)) = direct {
                        if journey.arrival - journey.departure >= walk_seconds {
                            continue;
                        }
                    }
                    cell_list.append(self.journey_to_dict(
                        py,
                        journey,
                        Some(&walks),
                        ends.as_ref(),
                        geometries,
                        intermediate,
                    )?)?;
                }
                row_list.append(cell_list)?;
            }
            cells.append(row_list)?;
        }
        let result = PyDict::new(py);
        result.set_item("journeys", cells)?;
        result.set_item("unsnapped_from", unsnapped_from.into_pyarray(py))?;
        result.set_item("unsnapped_to", unsnapped_to.into_pyarray(py))?;
        Ok(result.unbind())
    }

    /// The batched pareto frontier as flat columns, between stop ids.
    ///
    /// The exact rows of ``mc_frontier_matrix``'s frame — sorted per
    /// cell by (travel_time, emissions) and Pareto-marked — without
    /// materializing any journey payload.
    ///
    /// Returns
    /// -------
    /// dict
    ///     Equal-length columns ``from_index``, ``to_index``,
    ///     ``departure``, ``arrival``, ``travel_time``, ``rides``,
    ///     ``emissions`` (NaN where a transit factor is unresolved),
    ///     and ``frontier``.
    ///
    /// ``router="auto"`` (the default) runs on McTBTR when the cached
    /// multicriteria transfer set (``compute_mctbtr_transfers``) matches
    /// the query's date and factors and the query asks nothing McTBTR
    /// cannot answer, else on McRAPTOR; explicit values pick the engine
    /// directly.
    #[pyo3(signature = (from_stops, to_stops, date, departure, factors, window, max_transfers = 7, bucket = 25.0, max_slower = None, router = "auto"))]
    #[allow(clippy::too_many_arguments)]
    fn mc_frontier_table(
        &self,
        py: Python<'_>,
        from_stops: Vec<String>,
        to_stops: Vec<String>,
        date: &str,
        departure: &str,
        factors: Vec<(String, f64)>,
        window: u32,
        max_transfers: u8,
        bucket: f64,
        max_slower: Option<u32>,
        router: &str,
    ) -> PyResult<Py<PyDict>> {
        if !bucket.is_finite() || bucket <= 0.0 {
            return Err(PyValueError::new_err(
                "bucket must be a positive number of grams",
            ));
        }
        if !matches!(router, "auto" | "raptor" | "tbtr") {
            return Err(invalid_router(router));
        }
        if window == 0 {
            return Err(PyValueError::new_err(
                "window must be a positive number of seconds",
            ));
        }
        let Some(geometry) = &self.geometry else {
            return Err(PyValueError::new_err(
                "no trip distances installed; build the network with trip distances enabled",
            ));
        };
        let origins: Vec<StopIdx> = from_stops
            .iter()
            .map(|stop| self.resolve_stop(stop))
            .collect::<PyResult<_>>()?;
        let destinations: Vec<StopIdx> = to_stops
            .iter()
            .map(|stop| self.resolve_stop(stop))
            .collect::<PyResult<_>>()?;
        let mut per_trip = vec![f64::NAN; self.build.timetable.trip_count() as usize];
        for (trip_id, factor) in &factors {
            if let Some(&trip) = self.trips_by_public_id.get(trip_id) {
                per_trip[trip.0 as usize] = *factor;
            }
        }
        let router = self.resolve_mc_router(router, date, &per_trip, false)?;
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
            })
            .collect();
        let rows = py.allow_threads(|| {
            if router == "tbtr" {
                let engine = self.mctbtr_engine(
                    &self.transfers,
                    geometry,
                    &per_trip,
                    date,
                    &active_services,
                    &active_services_previous,
                );
                return engine.frontier_matrix(
                    &requests,
                    &destinations,
                    &[],
                    false,
                    destinations.len(),
                    window,
                    bucket,
                    max_slower,
                );
            }
            let view = DayView::for_date(
                &self.build.timetable,
                &active_services,
                &active_services_previous,
            );
            mcraptor::frontier_matrix(
                &view,
                &self.build.timetable,
                &self.transfers,
                geometry,
                &per_trip,
                &requests,
                &destinations,
                &[],
                false,
                destinations.len(),
                window,
                bucket,
                max_slower,
            )
        });
        let mut columns = FrontierColumns::default();
        for (i, row) in rows.iter().enumerate() {
            for (j, cell) in row.iter().enumerate() {
                let cell_rows: Vec<(u32, u32, u32, f64)> = cell
                    .iter()
                    .map(|journey| frontier_row(geometry, &per_trip, journey))
                    .collect();
                columns.push_cell(i as u32, j as u32, cell_rows);
            }
        }
        Ok(columns.into_dict(py)?.unbind())
    }

    /// The batched door-to-door pareto frontier as flat columns.
    ///
    /// The exact rows of ``mc_frontier_matrix_from_points``'s frame —
    /// walking-only alternatives and the walk-domination rule
    /// included, sorted per cell by (travel_time, emissions) and
    /// Pareto-marked — without materializing any journey payload.
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``mc_frontier_table``'s columns plus ``unsnapped_from`` /
    ///     ``unsnapped_to`` (indices of points off the walking
    ///     network; their cells stay empty).
    ///
    /// ``router="auto"`` (the default) runs on McTBTR when the cached
    /// multicriteria transfer set (``compute_mctbtr_transfers``) matches
    /// the query's date and factors and the query asks nothing McTBTR
    /// cannot answer, else on McRAPTOR; explicit values pick the engine
    /// directly.
    #[pyo3(signature = (origins, destinations, date, departure, factors, window, max_transfers = 7, bucket = 25.0, walking_speed_kmph = 3.6, max_walking_time = 7200.0, max_snap_distance = 1600.0, max_slower = None, router = "auto"))]
    #[allow(clippy::too_many_arguments)]
    fn mc_frontier_table_from_points(
        &self,
        py: Python<'_>,
        origins: Vec<(f64, f64)>,
        destinations: Vec<(f64, f64)>,
        date: &str,
        departure: &str,
        factors: Vec<(String, f64)>,
        window: u32,
        max_transfers: u8,
        bucket: f64,
        walking_speed_kmph: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
        max_slower: Option<u32>,
        router: &str,
    ) -> PyResult<Py<PyDict>> {
        if !bucket.is_finite() || bucket <= 0.0 {
            return Err(PyValueError::new_err(
                "bucket must be a positive number of grams",
            ));
        }
        if !matches!(router, "auto" | "raptor" | "tbtr") {
            return Err(invalid_router(router));
        }
        if window == 0 {
            return Err(PyValueError::new_err(
                "window must be a positive number of seconds",
            ));
        }
        let Some(geometry) = &self.geometry else {
            return Err(PyValueError::new_err(
                "no trip distances installed; build the network with trip distances enabled",
            ));
        };
        let streets = self.installed_streets()?;
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
        let router = self.resolve_mc_router(router, date, &per_trip, false)?;
        let departure = parse_time(departure)?;
        let active_services = self.active_services(date)?;
        let active_services_previous = self.active_services_previous(date)?;
        // The same intermediate set the one-pair coordinate route relaxes:
        // the McULTRA emissions set matching these factors, else the closure.
        let intermediate = self.emissions_transfers(factor_fingerprint(&per_trip));
        let (rows, walk, _, _, unsnapped_from, unsnapped_to) = self.point_frontier_rows(
            py,
            streets,
            intermediate,
            geometry,
            &per_trip,
            &origins,
            &destinations,
            date,
            departure,
            window,
            max_transfers,
            bucket,
            max_slower,
            router,
            speed,
            max_walking_time,
            max_snap_distance,
            &active_services,
            &active_services_previous,
        );
        let origin_snaps: Vec<Option<Snap>> = origins
            .iter()
            .map(|&(lat, lon)| streets.snap(lat, lon, max_snap_distance))
            .collect();
        let destination_snaps: Vec<Option<Snap>> = destinations
            .iter()
            .map(|&(lat, lon)| streets.snap(lat, lon, max_snap_distance))
            .collect();
        let mut columns = FrontierColumns::default();
        for (i, row) in rows.iter().enumerate() {
            for (j, cell) in row.iter().enumerate() {
                // The walking-only alternative and its domination rule,
                // exactly as the journey form.
                let direct = if origins[i] == destinations[j] {
                    Some((0, 0.0))
                } else {
                    walk[i][j]
                };
                let ends = origin_snaps[i].is_some() && destination_snaps[j].is_some();
                let banded_walk = direct.filter(|&(walk_seconds, _)| {
                    walk_within_band(walk_seconds, departure, cell, max_slower)
                });
                let mut cell_rows: Vec<(u32, u32, u32, f64)> = Vec::new();
                if let (Some((walk_seconds, _)), true) = (banded_walk, ends) {
                    cell_rows.push((departure, departure.saturating_add(walk_seconds), 0, 0.0));
                }
                for journey in cell {
                    if let Some((walk_seconds, _)) = direct {
                        if journey.arrival - journey.departure >= walk_seconds {
                            continue;
                        }
                    }
                    cell_rows.push(frontier_row(geometry, &per_trip, journey));
                }
                columns.push_cell(i as u32, j as u32, cell_rows);
            }
        }
        let result = columns.into_dict(py)?;
        result.set_item("unsnapped_from", unsnapped_from.into_pyarray(py))?;
        result.set_item("unsnapped_to", unsnapped_to.into_pyarray(py))?;
        Ok(result.unbind())
    }
}

impl TransportNetwork {
    /// The batched door-to-door frontier rows and their street
    /// plumbing — point linking, per-origin requests, the per-stop
    /// final-egress map, the engine dispatch, and the direct walk
    /// matrix — shared by the journey and table forms.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn point_frontier_rows(
        &self,
        py: Python<'_>,
        streets: &StreetNetwork,
        intermediate: &Transfers,
        geometry: &TripGeometry,
        per_trip: &[f64],
        origins: &[(f64, f64)],
        destinations: &[(f64, f64)],
        date: &str,
        departure: u32,
        window: u32,
        max_transfers: u8,
        bucket: f64,
        max_slower: Option<u32>,
        router: &str,
        speed: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
        active_services: &[bool],
        active_services_previous: &[bool],
    ) -> PointFrontierParts {
        let stop_count = self.build.timetable.stop_count() as usize;
        py.allow_threads(|| {
            let mut linked = streets.link_pointsets(
                &[origins, destinations],
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
                    active_services: active_services.to_vec(),
                    active_services_previous: active_services_previous.to_vec(),
                    max_transfers,
                })
                .collect();
            // Invert the destination links into the per-stop final-egress
            // map the frontier fold walks.
            let mut egress_map: Vec<Vec<(u32, u32, f64)>> = vec![Vec::new(); stop_count];
            for (slot, links) in destination_links.iter().enumerate() {
                for link in links.as_deref().unwrap_or(&[]) {
                    egress_map[link.stop.0 as usize].push((slot as u32, link.seconds, link.meters));
                }
            }
            let rows = if router == "tbtr" {
                let engine = self.mctbtr_engine(
                    intermediate,
                    geometry,
                    per_trip,
                    date,
                    active_services,
                    active_services_previous,
                );
                engine.frontier_matrix(
                    &requests,
                    &[],
                    &egress_map,
                    true,
                    destinations.len(),
                    window,
                    bucket,
                    max_slower,
                )
            } else {
                let view = DayView::for_date(
                    &self.build.timetable,
                    active_services,
                    active_services_previous,
                );
                mcraptor::frontier_matrix(
                    &view,
                    &self.build.timetable,
                    intermediate,
                    geometry,
                    per_trip,
                    &requests,
                    &[],
                    &egress_map,
                    true,
                    destinations.len(),
                    window,
                    bucket,
                    max_slower,
                )
            };
            let walk = streets.walk_matrix(
                origins,
                destinations,
                speed,
                max_walking_time,
                max_snap_distance,
            );
            (
                rows,
                walk,
                origin_links,
                destination_links,
                unsnapped_from,
                unsnapped_to,
            )
        })
    }
}
