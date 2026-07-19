//! The cost matrices: fastest and windowed least-cost forms, stop
//! and point.

use super::*;

#[pymethods]
impl TransportNetwork {
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
    /// max_transfers : int (optional, default: 7)
    ///     Maximum number of transfers between rides.
    /// to_stops : list of str (optional)
    ///     Destination stops; every stop when omitted.
    /// geometries : bool (optional, default: False)
    ///     Attach each pair's ridden legs as a WKB MultiLineString;
    ///     requires installed leg geometries.
    /// fares : dict (optional)
    ///     Flat fare tables from ``cafein.fares``; prices each pair's
    ///     journey into the ``fare`` array.
    /// router : str (optional, default: "auto")
    ///     ``"auto"`` runs on TBTR when the cached time transfer set
    ///     (``compute_tbtr_transfers``) matches the date — unless a
    ///     whole-day ULTRA set serves the matrix door-to-door, which
    ///     only the RAPTOR path does — else on RAPTOR; explicit values
    ///     pick the engine directly, ``"tbtr"`` keeping the closure.
    ///
    /// Returns
    /// -------
    /// dict
    ///     Equal-length arrays for the reachable pairs: ``from`` (row
    ///     into `from_stops`), ``to`` (index into ``stops``),
    ///     ``travel_time`` (seconds), ``rides``, ``transit_distance``
    ///     and ``walk_distance`` (meters), ``emissions`` (grams CO₂e,
    ///     NaN when unresolved), ``fare`` (NaN without `fares` or when
    ///     unpriceable), and with `geometries` a ``geometry`` list of
    ///     WKB bytes.
    #[pyo3(signature = (from_stops, date, departure, factors, max_transfers = 7, to_stops = None, router = "auto", walking_speed_kmph = 3.6, max_walking_time = 7200.0, max_snap_distance = 1600.0, geometries = false, fares = None))]
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
        router: &str,
        walking_speed_kmph: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
        geometries: bool,
        fares: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDict>> {
        if !matches!(router, "auto" | "raptor" | "tbtr") {
            return Err(invalid_router(router));
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
        let tables = fares
            .map(|spec| {
                fare_tables(
                    &spec,
                    self.feed.routes.len(),
                    self.build.timetable.stop_count() as usize,
                )
            })
            .transpose()?;
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
        let active_services_previous = self.active_services_previous(date)?;
        let inputs = CostInputs {
            geometry,
            factors: &per_trip,
            leg_geometry: self.leg_geometry.as_ref(),
            with_geometry: geometries,
            fares: tables.as_ref(),
        };
        // Under a whole-day set, snappable origins route door-to-door (the stop
        // cost matrix as a point cost matrix over the stops' coordinates);
        // validate the walking speed only when at least one origin is usable.
        let ultra_usable = router != "tbtr"
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
            self.resolve_time_router(router, date)?
        };
        let rows = py.allow_threads(|| {
            if let Some(speed) = ultra_speed {
                let streets = self
                    .streets
                    .as_ref()
                    .expect("ultra_usable implies a street network");
                self.ultra_cost_matrix_rows(
                    streets,
                    &origins,
                    &destinations,
                    departure,
                    &active_services,
                    &active_services_previous,
                    max_transfers,
                    &inputs,
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
                    })
                    .collect();
                if router == "tbtr" {
                    let engine = self.tbtr_engine(
                        &self.transfers,
                        date,
                        &active_services,
                        &active_services_previous,
                    );
                    engine.cost_matrix(&inputs, &requests, &destinations)
                } else {
                    Raptor.cost_matrix(
                        &self.build.timetable,
                        &self.transfers,
                        &inputs,
                        &requests,
                        &destinations,
                    )
                }
            }
        });
        cost_rows_dict(py, rows, geometries)
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
    /// ``router="auto"`` (the default) runs on TBTR when the cached time
    /// transfer set (``compute_tbtr_transfers``) matches the date, else
    /// on RAPTOR; explicit values pick the engine directly.
    ///
    /// Returns
    /// -------
    /// dict
    ///     As ``travel_cost_matrix`` — ``from`` and ``to`` index the
    ///     origin and destination point lists — plus
    ///     ``unsnapped_from`` / ``unsnapped_to`` with the indices of
    ///     points off the walking network.
    #[pyo3(signature = (origins, destinations, date, departure, factors, max_transfers = 7, router = "auto", walking_speed_kmph = 3.6, max_walking_time = 7200.0, max_snap_distance = 1600.0, geometries = false, fares = None))]
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
        router: &str,
        walking_speed_kmph: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
        geometries: bool,
        fares: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDict>> {
        if !matches!(router, "auto" | "raptor" | "tbtr") {
            return Err(invalid_router(router));
        }
        let router = self.resolve_time_router(router, date)?;
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
        let tables = fares
            .map(|spec| {
                fare_tables(
                    &spec,
                    self.feed.routes.len(),
                    self.build.timetable.stop_count() as usize,
                )
            })
            .transpose()?;
        // Walking rides nothing: with tables it is free, without them
        // fares are not computed at all.
        let walk_fare = if tables.is_some() { 0.0 } else { f64::NAN };
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
        let active_services_previous = self.active_services_previous(date)?;
        let inputs = CostInputs {
            geometry,
            factors: &per_trip,
            leg_geometry: self.leg_geometry.as_ref(),
            with_geometry: geometries,
            fares: tables.as_ref(),
        };
        let (rows, unsnapped_from, unsnapped_to) = py.allow_threads(|| {
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
            let mut requests = Vec::with_capacity(origin_links.len());
            let mut access_meters = Vec::with_capacity(origin_links.len());
            for links in &origin_links {
                let links = links.as_deref().unwrap_or(&[]);
                requests.push(Request {
                    departure,
                    access: request_offsets(links),
                    egress: Vec::new(),
                    active_services: active_services.clone(),
                    active_services_previous: active_services_previous.clone(),
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
            let mut rows = if router == "tbtr" {
                let engine = self.tbtr_engine(
                    self.time_transfers(),
                    date,
                    &active_services,
                    &active_services_previous,
                );
                engine.cost_matrix_to_points(&inputs, &requests, &access_meters, &egress)
            } else {
                Raptor.cost_matrix_to_points(
                    &self.build.timetable,
                    self.time_transfers(),
                    &inputs,
                    &requests,
                    &access_meters,
                    &egress,
                )
            };
            // Walking directly can beat transit: such cells become
            // walking-only rows — zero rides, zero emissions, the walk
            // as the distance. The time fill is one street search per
            // origin; with geometries, each *winning* walk cell
            // additionally reconstructs its street path, mirroring the
            // per-row WKB assembly transit rows already pay.
            let walk = streets.walk_matrix(
                &origins,
                &destinations,
                speed,
                max_walking_time,
                max_snap_distance,
            );
            let walk_geometry = |origin: usize, point: usize| -> Option<Vec<u8>> {
                if !geometries {
                    return None;
                }
                let from_point = origins[origin];
                let to_point = destinations[point];
                if from_point == to_point {
                    // A zero walk degenerates at its own coordinate.
                    let at = (from_point.1, from_point.0);
                    return Some(wkb_multi_line_string(&[vec![at, at]]));
                }
                let from = streets.snap(from_point.0, from_point.1, max_snap_distance)?;
                let to = streets.snap(to_point.0, to_point.1, max_snap_distance)?;
                let (path, _) = streets.walk_path(from_point, &from, to_point, &to)?;
                Some(wkb_multi_line_string(&[path]))
            };
            for (origin, origin_rows) in rows.iter_mut().enumerate() {
                let walk_row = &walk[origin];
                let mut reached = vec![false; destinations.len()];
                for row in origin_rows.iter_mut() {
                    reached[row.to as usize] = true;
                    if let Some((walk_seconds, meters)) = walk_row[row.to as usize] {
                        // Ties resolve toward fewer rides, as the
                        // matrix contract promises: an equal-time walk
                        // beats a ridden row.
                        if walk_seconds < row.seconds
                            || (walk_seconds == row.seconds && row.rides > 0)
                        {
                            row.seconds = walk_seconds;
                            row.rides = 0;
                            row.transit_meters = 0.0;
                            row.walk_meters = meters;
                            row.emission_grams = 0.0;
                            row.fare = walk_fare;
                            row.geometry = walk_geometry(origin, row.to as usize);
                        }
                    }
                }
                for (point, cell) in walk_row.iter().enumerate() {
                    if reached[point] {
                        continue;
                    }
                    if let Some((walk_seconds, meters)) = cell {
                        origin_rows.push(CostRow {
                            to: point as u32,
                            seconds: *walk_seconds,
                            rides: 0,
                            transit_meters: 0.0,
                            walk_meters: *meters,
                            emission_grams: 0.0,
                            fare: walk_fare,
                            geometry: walk_geometry(origin, point),
                        });
                    }
                }
                origin_rows.sort_unstable_by_key(|row| row.to);
            }
            (rows, unsnapped_from, unsnapped_to)
        });
        let result = cost_rows_dict(py, rows, geometries)?;
        result
            .bind(py)
            .set_item("unsnapped_from", unsnapped_from.into_pyarray(py))?;
        result
            .bind(py)
            .set_item("unsnapped_to", unsnapped_to.into_pyarray(py))?;
        Ok(result)
    }

    /// The objective-best journey's aggregated costs per OD pair
    /// within a travel-time budget, long format — the emissions/fare
    /// counterpart of ``travel_cost_matrix`` over a departure window.
    ///
    /// The candidates per pair are the departure window's
    /// (departure, arrival, rides)-Pareto set — the same set
    /// ``journey_frontier`` sees — and a cell reports its
    /// lowest-objective member within `budget` (no budget: within the
    /// window's reach), ties resolved toward the shorter travel time.
    /// Pairs with no qualifying candidate (a resolved emission, a
    /// priceable fare) are absent. The ``"fare"`` objective requires
    /// the flat fare tables ``cafein.fares`` produces.
    ///
    /// With ``candidates="pareto"`` (``"emissions"`` objective only)
    /// the candidates per pair are McRAPTOR's (departure, arrival,
    /// emissions bucket) Pareto set instead, which also holds the
    /// cleaner-but-slower journeys the time-optimal set misses; cells
    /// can therefore report strictly lower emissions.
    ///
    /// ``router="auto"`` (the default) runs the pareto candidates on
    /// McTBTR when the cached multicriteria transfer set matches the
    /// query's date and factors and no matching whole-day McULTRA set
    /// serves the stop matrix door-to-door (only the McRAPTOR path does),
    /// else on McRAPTOR. Time and fare candidates run on TBTR when the
    /// cached time transfer set (``compute_tbtr_transfers``) matches the
    /// date, else on RAPTOR; explicit values pick the engine directly.
    #[pyo3(signature = (from_stops, date, departure, window, factors, objective = "emissions", fares = None, budget = None, max_transfers = 7, to_stops = None, candidates = "time", bucket = 25.0, router = "auto", walking_speed_kmph = 3.6, max_walking_time = 7200.0, max_snap_distance = 1600.0, geometries = false))]
    #[allow(clippy::too_many_arguments)]
    fn least_cost_matrix(
        &self,
        py: Python<'_>,
        from_stops: Vec<String>,
        date: &str,
        departure: &str,
        window: u32,
        factors: Vec<(String, f64)>,
        objective: &str,
        fares: Option<Bound<'_, PyDict>>,
        budget: Option<u32>,
        max_transfers: u8,
        to_stops: Option<Vec<String>>,
        candidates: &str,
        bucket: f64,
        router: &str,
        walking_speed_kmph: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
        geometries: bool,
    ) -> PyResult<Py<PyDict>> {
        if !matches!(router, "auto" | "raptor" | "tbtr") {
            return Err(invalid_router(router));
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
        if window == 0 {
            return Err(PyValueError::new_err(
                "window must be a positive number of seconds",
            ));
        }
        let tables = fares
            .map(|spec| {
                fare_tables(
                    &spec,
                    self.feed.routes.len(),
                    self.build.timetable.stop_count() as usize,
                )
            })
            .transpose()?;
        if candidates != "time" && candidates != "pareto" {
            return Err(PyValueError::new_err(
                "candidates must be 'time' or 'pareto'",
            ));
        }
        if candidates == "pareto" {
            if objective != "emissions" {
                return Err(PyValueError::new_err(
                    "pareto candidates support the 'emissions' objective only",
                ));
            }
            if !bucket.is_finite() || bucket <= 0.0 {
                return Err(PyValueError::new_err(
                    "bucket must be a positive number of grams",
                ));
            }
        }
        let objective = parse_objective(objective, tables.as_ref())?;
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
        let active_services_previous = self.active_services_previous(date)?;
        // Under a McULTRA set matching this query's factors, the pareto/raptor
        // matrix routes door-to-door: a location-based initial walk per origin,
        // the shortcut set for the intermediate transfers, and a street final
        // walk folded per destination. Without a matching set (or a street
        // network) it keeps the closure and board-at-origin access; the trip-based
        // engines and the time and fare candidates always keep the closure.
        let stop_count = self.build.timetable.stop_count() as usize;
        let fingerprint = factor_fingerprint(&per_trip);
        let matrix_mcultra = candidates == "pareto"
            && router != "tbtr"
            && !std::ptr::eq(self.emissions_transfers(fingerprint), &self.transfers)
            && self.streets.is_some();
        // Auto prefers the door-to-door McULTRA path over an engine switch;
        // pareto candidates resolve on the McTBTR cache, time and fare
        // candidates on the cached time set.
        let router = if matrix_mcultra {
            "raptor"
        } else if candidates == "pareto" {
            self.resolve_mc_router(router, date, &per_trip, false)?
        } else {
            self.resolve_time_router(router, date)?
        };
        // Origins that do not take a location-based initial walk (no coordinate,
        // no snap, or no stop reachable within the cap) are marked `!located`;
        // routed over the intermediate-only set they would lose the closure's
        // initial footpaths, so they fall back to closure board-at-origin routing
        // below (mirroring the ULTRA matrices' per-row partition).
        let (access, snappable, egress_map, direct_walks) = if matrix_mcultra {
            let streets = self.streets.as_ref().unwrap();
            let speed =
                validated_walking_speed(walking_speed_kmph, max_walking_time, max_snap_distance)?;
            let (access, located) = self.matrix_location_access(
                streets,
                &origins,
                speed,
                max_walking_time,
                max_snap_distance,
            );
            let egress_map = self.matrix_street_egress(
                streets,
                &destinations,
                speed,
                max_walking_time,
                max_snap_distance,
            );
            let direct_walks = self.matrix_direct_walks(
                streets,
                &origins,
                &destinations,
                speed,
                max_walking_time,
                max_snap_distance,
            );
            (access, located, egress_map, direct_walks)
        } else {
            (
                origins
                    .iter()
                    .map(|&origin| vec![(origin, 0u32, 0.0)])
                    .collect(),
                Vec::new(),
                vec![Vec::new(); stop_count],
                Vec::new(),
            )
        };
        let matrix_transfers = if matrix_mcultra {
            self.emissions_transfers(fingerprint)
        } else {
            &self.transfers
        };
        // Split the located access into routing offsets (stop, seconds) and the
        // walk metres reported per boarded stop.
        let access_meters: Vec<Vec<(StopIdx, f64)>> = access
            .iter()
            .map(|offsets| {
                offsets
                    .iter()
                    .map(|&(stop, _, meters)| (stop, meters))
                    .collect()
            })
            .collect();
        let priced = tables.is_some();
        let requests: Vec<Request> = access
            .into_iter()
            .map(|offsets| Request {
                departure,
                access: offsets
                    .into_iter()
                    .map(|(stop, seconds, _)| (stop, seconds))
                    .collect(),
                egress: Vec::new(),
                active_services: active_services.clone(),
                active_services_previous: active_services_previous.clone(),
                max_transfers,
            })
            .collect();
        let inputs = CostInputs {
            geometry,
            factors: &per_trip,
            leg_geometry: self.leg_geometry.as_ref(),
            with_geometry: geometries,
            fares: tables.as_ref(),
        };
        let rows = py.allow_threads(|| {
            if candidates == "pareto" && router == "tbtr" {
                let engine = self.mctbtr_engine(
                    &self.transfers,
                    geometry,
                    &per_trip,
                    date,
                    &active_services,
                    &active_services_previous,
                );
                return engine.least_emissions_matrix(
                    &inputs,
                    &requests,
                    &destinations,
                    window,
                    budget,
                    bucket,
                );
            }
            if candidates == "pareto" {
                let view = DayView::for_date(
                    &self.build.timetable,
                    &active_services,
                    &active_services_previous,
                );
                let mut rows = mcraptor::least_emissions_matrix(
                    &view,
                    &self.build.timetable,
                    matrix_transfers,
                    &inputs,
                    &requests,
                    &destinations,
                    &egress_map,
                    &access_meters,
                    matrix_mcultra,
                    window,
                    budget,
                    bucket,
                );
                if matrix_mcultra && snappable.iter().any(|&located| !located) {
                    // Re-route the unsnappable origins over the closure (board at
                    // the origin, no street walks) and keep the door-to-door rows
                    // only for snappable origins, in input order.
                    let closure_requests: Vec<Request> = origins
                        .iter()
                        .map(|&origin| Request {
                            departure,
                            access: vec![(origin, 0)],
                            egress: Vec::new(),
                            active_services: active_services.clone(),
                            active_services_previous: active_services_previous.clone(),
                            max_transfers,
                        })
                        .collect();
                    let closure_egress = vec![Vec::new(); stop_count];
                    let closure_access_meters = vec![Vec::new(); origins.len()];
                    let closure = mcraptor::least_emissions_matrix(
                        &view,
                        &self.build.timetable,
                        &self.transfers,
                        &inputs,
                        &closure_requests,
                        &destinations,
                        &closure_egress,
                        &closure_access_meters,
                        false,
                        window,
                        budget,
                        bucket,
                    );
                    rows = rows
                        .into_iter()
                        .zip(closure)
                        .zip(&snappable)
                        .map(
                            |((door, closure_row), &located)| {
                                if located {
                                    door
                                } else {
                                    closure_row
                                }
                            },
                        )
                        .collect();
                }
                // Overlay the explicit direct street walks onto each located
                // origin's door-to-door cells (the diagonal is a true zero walk).
                if matrix_mcultra {
                    for (origin_rows, (walks, &located)) in
                        rows.iter_mut().zip(direct_walks.iter().zip(&snappable))
                    {
                        if located {
                            merge_direct_walk_cells(
                                origin_rows,
                                walks,
                                &destinations,
                                budget,
                                priced,
                            );
                        }
                    }
                }
                rows
            } else if router == "tbtr" {
                let engine = self.tbtr_engine(
                    &self.transfers,
                    date,
                    &active_services,
                    &active_services_previous,
                );
                engine.least_cost_matrix(
                    &inputs,
                    &requests,
                    &destinations,
                    window,
                    budget,
                    objective,
                )
            } else {
                Raptor.least_cost_matrix(
                    &self.build.timetable,
                    &self.transfers,
                    &inputs,
                    &requests,
                    &destinations,
                    window,
                    budget,
                    objective,
                )
            }
        });
        cost_rows_dict(py, rows, geometries)
    }

    /// ``least_cost_matrix`` between coordinate points, linked through
    /// the street network like ``travel_cost_matrix_from_points`` —
    /// including the walking-only alternative, whose zero emissions
    /// (and zero fare) win any cell they qualify for within the budget.
    ///
    /// ``router="auto"`` (the default) runs on TBTR when the cached time
    /// transfer set (``compute_tbtr_transfers``) matches the date, else
    /// on RAPTOR; explicit values pick the engine directly.
    #[pyo3(signature = (origins, destinations, date, departure, window, factors, objective = "emissions", fares = None, budget = None, max_transfers = 7, router = "auto", walking_speed_kmph = 3.6, max_walking_time = 7200.0, max_snap_distance = 1600.0, geometries = false))]
    #[allow(clippy::too_many_arguments)]
    fn least_cost_matrix_from_points(
        &self,
        py: Python<'_>,
        origins: Vec<(f64, f64)>,
        destinations: Vec<(f64, f64)>,
        date: &str,
        departure: &str,
        window: u32,
        factors: Vec<(String, f64)>,
        objective: &str,
        fares: Option<Bound<'_, PyDict>>,
        budget: Option<u32>,
        max_transfers: u8,
        router: &str,
        walking_speed_kmph: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
        geometries: bool,
    ) -> PyResult<Py<PyDict>> {
        if !matches!(router, "auto" | "raptor" | "tbtr") {
            return Err(invalid_router(router));
        }
        let router = self.resolve_time_router(router, date)?;
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
        if window == 0 {
            return Err(PyValueError::new_err(
                "window must be a positive number of seconds",
            ));
        }
        let tables = fares
            .map(|spec| {
                fare_tables(
                    &spec,
                    self.feed.routes.len(),
                    self.build.timetable.stop_count() as usize,
                )
            })
            .transpose()?;
        let objective = parse_objective(objective, tables.as_ref())?;
        let walk_fare = if tables.is_some() { 0.0 } else { f64::NAN };
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
        let active_services_previous = self.active_services_previous(date)?;
        let inputs = CostInputs {
            geometry,
            factors: &per_trip,
            leg_geometry: self.leg_geometry.as_ref(),
            with_geometry: geometries,
            fares: tables.as_ref(),
        };
        let (rows, unsnapped_from, unsnapped_to) = py.allow_threads(|| {
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
            let mut requests = Vec::with_capacity(origin_links.len());
            let mut access_meters = Vec::with_capacity(origin_links.len());
            for links in &origin_links {
                let links = links.as_deref().unwrap_or(&[]);
                requests.push(Request {
                    departure,
                    access: request_offsets(links),
                    egress: Vec::new(),
                    active_services: active_services.clone(),
                    active_services_previous: active_services_previous.clone(),
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
            // Single-criterion (time-Pareto candidates, then lowest-objective):
            // it keeps the closure. McULTRA is an emissions-Pareto set, not
            // time-complete, so relaxing it here could drop time-relevant
            // transfers; the emissions-complete coordinate path is the McRAPTOR
            // one (`mc_route_between_coordinates`, `candidates="pareto"`).
            let mut rows = if router == "tbtr" {
                let engine = self.tbtr_engine(
                    &self.transfers,
                    date,
                    &active_services,
                    &active_services_previous,
                );
                engine.least_cost_matrix_to_points(
                    &inputs,
                    &requests,
                    &access_meters,
                    &egress,
                    window,
                    budget,
                    objective,
                )
            } else {
                Raptor.least_cost_matrix_to_points(
                    &self.build.timetable,
                    &self.transfers,
                    &inputs,
                    &requests,
                    &access_meters,
                    &egress,
                    window,
                    budget,
                    objective,
                )
            };
            // The walking-only alternative: zero grams and zero fare,
            // so within the budget it wins any cell (equal-key cells
            // resolve toward the shorter travel time, as everywhere).
            let key = |row: &CostRow| match objective {
                Objective::Emissions => row.emission_grams,
                Objective::Fare => row.fare,
            };
            let walk = streets.walk_matrix(
                &origins,
                &destinations,
                speed,
                max_walking_time,
                max_snap_distance,
            );
            let walk_geometry = |origin: usize, point: usize| -> Option<Vec<u8>> {
                if !geometries {
                    return None;
                }
                let from_point = origins[origin];
                let to_point = destinations[point];
                if from_point == to_point {
                    let at = (from_point.1, from_point.0);
                    return Some(wkb_multi_line_string(&[vec![at, at]]));
                }
                let from = streets.snap(from_point.0, from_point.1, max_snap_distance)?;
                let to = streets.snap(to_point.0, to_point.1, max_snap_distance)?;
                let (path, _) = streets.walk_path(from_point, &from, to_point, &to)?;
                Some(wkb_multi_line_string(&[path]))
            };
            for (origin, origin_rows) in rows.iter_mut().enumerate() {
                let walk_row = &walk[origin];
                let mut reached = vec![false; destinations.len()];
                for row in origin_rows.iter_mut() {
                    reached[row.to as usize] = true;
                    if let Some((walk_seconds, meters)) = walk_row[row.to as usize] {
                        if budget.is_some_and(|budget| walk_seconds > budget) {
                            continue;
                        }
                        if key(row) > 0.0 || (key(row) == 0.0 && walk_seconds < row.seconds) {
                            row.seconds = walk_seconds;
                            row.rides = 0;
                            row.transit_meters = 0.0;
                            row.walk_meters = meters;
                            row.emission_grams = 0.0;
                            row.fare = walk_fare;
                            row.geometry = walk_geometry(origin, row.to as usize);
                        }
                    }
                }
                for (point, cell) in walk_row.iter().enumerate() {
                    if reached[point] {
                        continue;
                    }
                    if let Some((walk_seconds, meters)) = cell {
                        if budget.is_some_and(|budget| *walk_seconds > budget) {
                            continue;
                        }
                        origin_rows.push(CostRow {
                            to: point as u32,
                            seconds: *walk_seconds,
                            rides: 0,
                            transit_meters: 0.0,
                            walk_meters: *meters,
                            emission_grams: 0.0,
                            fare: walk_fare,
                            geometry: walk_geometry(origin, point),
                        });
                    }
                }
                origin_rows.sort_unstable_by_key(|row| row.to);
            }
            (rows, unsnapped_from, unsnapped_to)
        });
        let result = cost_rows_dict(py, rows, geometries)?;
        result
            .bind(py)
            .set_item("unsnapped_from", unsnapped_from.into_pyarray(py))?;
        result
            .bind(py)
            .set_item("unsnapped_to", unsnapped_to.into_pyarray(py))?;
        Ok(result)
    }
}

impl TransportNetwork {
    /// The `CostRow` rows for a stop-origin travel-cost matrix under a
    /// whole-day ULTRA set. Usable (snappable) origins route door-to-door: the
    /// stop cost matrix is the point cost matrix over the stops' coordinates,
    /// so `cost_matrix_to_points` runs with coordinate access and a
    /// location-based per-destination egress — the final walks `link_many` finds
    /// to `d`'s coordinate (which include `d` itself via its connector), the
    /// same egress the point cost matrix uses; `costs_to_point` rebuilds each
    /// final-walk row from its source stop's row with the walk added. A
    /// destination with no coordinate keeps its transit arrival via a
    /// `(d, 0, 0)` seed instead (it cannot be located), matching the one-to-all
    /// time queries. Off-network origins fall back to the closure `cost_matrix`.
    /// Rows come back in input origin order, keyed by global destination stop
    /// index (the point-matrix rows are remapped from destination-list index).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ultra_cost_matrix_rows(
        &self,
        streets: &StreetNetwork,
        origins: &[StopIdx],
        destinations: &[StopIdx],
        departure: u32,
        active_services: &[bool],
        active_services_previous: &[bool],
        max_transfers: u8,
        inputs: &CostInputs<'_>,
        speed: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
    ) -> Vec<Vec<CostRow>> {
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
        let mut rows: Vec<Vec<CostRow>> = vec![Vec::new(); origins.len()];
        if !usable.is_empty() {
            // Per-destination egress, location-based (the destination stop as
            // its coordinate): the final walks link_many finds to that
            // coordinate — undirected walking, so this is the s -> d egress, the
            // same construction the point cost matrix uses.
            let mut link_inputs: Vec<(f64, f64)> = Vec::new();
            let mut link_index: Vec<Option<usize>> = Vec::with_capacity(destinations.len());
            for &destination in destinations {
                match self.stop_coordinate(destination) {
                    Some(coordinate) => {
                        link_index.push(Some(link_inputs.len()));
                        link_inputs.push(coordinate);
                    }
                    None => link_index.push(None),
                }
            }
            let destination_links =
                streets.link_many(&link_inputs, speed, max_walking_time, max_snap_distance);
            let egress: Vec<Vec<(StopIdx, u32, f64)>> = destinations
                .iter()
                .enumerate()
                .map(|(index, &destination)| match link_index[index] {
                    // Located: the walks link_many finds to the stop's coordinate
                    // (which includes the stop itself via its connector), exactly
                    // the point cost matrix's egress — no separate (d, 0, 0)
                    // transit seed. Empty (coordinate off the network) leaves the
                    // destination unreachable, as the coordinate query would.
                    Some(slot) => destination_links[slot]
                        .as_ref()
                        .map(|reached| {
                            reached
                                .iter()
                                .map(|walk| (walk.stop, walk.seconds, walk.meters))
                                .collect()
                        })
                        .unwrap_or_default(),
                    // No coordinate: the stop cannot be located, so keep its
                    // transit arrival via a zero-length final walk at the stop.
                    None => vec![(destination, 0u32, 0.0f64)],
                })
                .collect();
            let coordinates: Vec<(f64, f64)> = usable.iter().map(|&(_, c)| c).collect();
            let origin_links =
                streets.link_many(&coordinates, speed, max_walking_time, max_snap_distance);
            let mut requests = Vec::with_capacity(usable.len());
            let mut access_meters = Vec::with_capacity(usable.len());
            for links in &origin_links {
                let links = links.as_deref().unwrap_or(&[]);
                requests.push(Request {
                    departure,
                    access: request_offsets(links),
                    egress: Vec::new(),
                    active_services: active_services.to_vec(),
                    active_services_previous: active_services_previous.to_vec(),
                    max_transfers,
                });
                access_meters.push(
                    links
                        .iter()
                        .map(|walk| (walk.stop, walk.meters))
                        .collect::<HashMap<_, _>>(),
                );
            }
            let mut usable_rows = Raptor.cost_matrix_to_points(
                &self.build.timetable,
                self.time_transfers(),
                inputs,
                &requests,
                &access_meters,
                &egress,
            );
            for origin_rows in usable_rows.iter_mut() {
                for row in origin_rows.iter_mut() {
                    row.to = destinations[row.to as usize].0;
                }
            }
            for (origin_rows, &(index, _)) in usable_rows.into_iter().zip(&usable) {
                rows[index] = origin_rows;
            }
        }
        if !fallback.is_empty() {
            let requests: Vec<Request> = fallback
                .iter()
                .map(|&(_, origin)| Request {
                    departure,
                    access: vec![(origin, 0)],
                    egress: Vec::new(),
                    active_services: active_services.to_vec(),
                    active_services_previous: active_services_previous.to_vec(),
                    max_transfers,
                })
                .collect();
            let fallback_rows = Raptor.cost_matrix(
                &self.build.timetable,
                &self.transfers,
                inputs,
                &requests,
                destinations,
            );
            for (origin_rows, &(index, _)) in fallback_rows.into_iter().zip(&fallback) {
                rows[index] = origin_rows;
            }
        }
        rows
    }
}

/// Parses the flat fare tables `cafein.fares` produces, validating the
/// arrays against the network's route and stop counts.
pub(super) fn fare_tables(
    spec: &Bound<'_, PyDict>,
    route_count: usize,
    stop_count: usize,
) -> PyResult<FareTables> {
    fn item<'py, T: FromPyObject<'py>>(spec: &Bound<'py, PyDict>, key: &str) -> PyResult<T> {
        spec.get_item(key)?
            .ok_or_else(|| PyValueError::new_err(format!("fare tables are missing {key:?}")))?
            .extract()
    }
    if spec.contains("stop_zone")? {
        let stop_zone: Vec<u32> = item(spec, "stop_zone")?;
        if stop_zone.len() != stop_count {
            return Err(PyValueError::new_err(
                "the fare tables' stop_zone must cover every stop",
            ));
        }
        if stop_zone.iter().any(|&zone| zone != NO_FARE && zone >= 128) {
            return Err(PyValueError::new_err(
                "fare zone indexes must stay below 128",
            ));
        }
        let products: Vec<(f64, u128, f64, u32)> = item(spec, "products")?;
        let products = products
            .into_iter()
            .map(|(price, zones, duration, transfers)| ZoneProduct {
                price,
                zones,
                duration,
                transfers,
            })
            .collect();
        Ok(FareTables::Zone(ZoneFares {
            stop_zone,
            products,
        }))
    } else {
        let tables = RuleFares {
            route_type: item(spec, "route_type")?,
            route_fare: item(spec, "route_fare")?,
            unlimited_transfers: item(spec, "unlimited_transfers")?,
            allow_same_route: item(spec, "allow_same_route")?,
            pair_fare: item(spec, "pair_fare")?,
            max_discounted_transfers: item(spec, "max_discounted_transfers")?,
            transfer_allowance: item(spec, "transfer_allowance")?,
            fare_cap: item(spec, "fare_cap")?,
        };
        let count = tables.unlimited_transfers.len();
        if tables.route_type.len() != route_count || tables.route_fare.len() != route_count {
            return Err(PyValueError::new_err(
                "the fare tables' route arrays must cover every route",
            ));
        }
        if tables.allow_same_route.len() != count || tables.pair_fare.len() != count * count {
            return Err(PyValueError::new_err(
                "the fare tables' type arrays disagree on the type count",
            ));
        }
        if tables
            .route_type
            .iter()
            .any(|&kind| kind != NO_FARE && kind as usize >= count)
        {
            return Err(PyValueError::new_err(
                "the fare tables' route types must index the type arrays",
            ));
        }
        Ok(FareTables::RuleBased(tables))
    }
}

/// Parses the objective a windowed candidate fold minimises.
pub(super) fn parse_objective(objective: &str, fares: Option<&FareTables>) -> PyResult<Objective> {
    match objective {
        "emissions" => Ok(Objective::Emissions),
        "fare" if fares.is_none() => Err(PyValueError::new_err(
            "the 'fare' objective requires fare tables",
        )),
        "fare" => Ok(Objective::Fare),
        other => Err(PyValueError::new_err(format!(
            "objective must be 'emissions' or 'fare', not {other:?}"
        ))),
    }
}

/// Flattens per-origin cost rows into the columnar dict the Python
/// matrices consume: equal-length arrays for the surviving pairs, plus
/// a WKB list when geometries ride along.
pub(super) fn cost_rows_dict(
    py: Python<'_>,
    rows: Vec<Vec<CostRow>>,
    geometries: bool,
) -> PyResult<Py<PyDict>> {
    let total: usize = rows.iter().map(Vec::len).sum();
    let mut from = Vec::with_capacity(total);
    let mut to = Vec::with_capacity(total);
    let mut travel_time = Vec::with_capacity(total);
    let mut rides = Vec::with_capacity(total);
    let mut transit_distance = Vec::with_capacity(total);
    let mut walk_distance = Vec::with_capacity(total);
    let mut emissions = Vec::with_capacity(total);
    let mut fare = Vec::with_capacity(total);
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
            fare.push(row.fare);
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
    result.set_item("fare", fare.into_pyarray(py))?;
    if geometries {
        result.set_item("geometry", wkbs)?;
    }
    Ok(result.unbind())
}

/// Whether the walking-only journey stays within a ``max_slower`` band:
/// its arrival must sit within the band of the fastest returned transit
/// journey (the minimum of the per-pass anchors the search's output
/// filter kept). Without the restriction, or when nothing rides, the
/// walk always stays. Anchoring on the pre-walk-domination journey set
/// is equivalent to anchoring on the emitted rows: a walk-dominated
/// journey travels at least the walk's seconds and departs no earlier
/// than the walk, so it arrives no earlier than the walk itself and can
/// neither keep nor drop the walk differently than the kept set's
/// fastest would.
pub(super) fn walk_within_band(
    walk_seconds: u32,
    departure: u32,
    journeys: &[Journey],
    max_slower: Option<u32>,
) -> bool {
    let Some(band) = max_slower else {
        return true;
    };
    let Some(fastest) = journeys.iter().map(|journey| journey.arrival).min() else {
        return true;
    };
    departure.saturating_add(walk_seconds) <= fastest.saturating_add(band)
}

/// Overlays an origin's explicit direct street walks onto its emissions cost
/// cells: a walking-only journey — zero rides, its walked metres, zero emissions
/// under today's walking factor — wins a destination cell whenever nothing
/// transit-side is cleaner. `walks` is `(destination slot, walk seconds, walk
/// metres)` with the diagonal (origin coordinate == destination coordinate)
/// already zeroed by the caller; a walk beyond the travel-time `budget` is
/// dropped. `priced` prices the walk at zero fare when a fare model is present.
pub(super) fn merge_direct_walk_cells(
    row: &mut Vec<CostRow>,
    walks: &[(u32, u32, f64)],
    destinations: &[StopIdx],
    budget: Option<u32>,
    priced: bool,
) {
    for &(slot, seconds, meters) in walks {
        if budget.is_some_and(|cap| seconds > cap) {
            continue;
        }
        let to = destinations[slot as usize].0;
        let cell = CostRow {
            to,
            seconds,
            rides: 0,
            transit_meters: 0.0,
            walk_meters: meters,
            emission_grams: 0.0,
            fare: if priced { 0.0 } else { f64::NAN },
            geometry: None,
        };
        match row.iter_mut().find(|existing| existing.to == to) {
            Some(existing) => {
                if 0.0 < existing.emission_grams
                    || (existing.emission_grams == 0.0 && seconds < existing.seconds)
                {
                    *existing = cell;
                }
            }
            None => row.push(cell),
        }
    }
}
