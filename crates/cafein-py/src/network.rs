//! Construction, precomputes, statistics, and the engine and
//! router resolvers.

use super::*;

/// The six installed street-attribute arrays, in canonical order, as the
/// inspection getter returns them to Python.
type StreetAttributeArrays = (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u16>);

/// One street access/egress row: public stop id, whole seconds, and the
/// chosen link's snap identity (edge, fraction, connector meters) — the
/// `StreetChoice` token the 12c reduction keeps beside the duration.
type StreetRow = (String, u32, u32, f64, f64);

/// A mode's raw per-stop times beside the mode-masked links they used.
type ModeRow = (Vec<Option<u32>>, Vec<Option<Snap>>);

/// One reduced street choice: stop, seconds, winning mode, its link snap
/// identity (edge, fraction, connector meters), and — when the transfer
/// closure carried it — the seed stop the vehicle actually reached. The
/// `StreetChoice` sidecar the reconstruction stage consumes.
type ReducedChoice = (String, u32, String, u32, f64, f64, Option<String>);

/// The reduction's running winner per stop.
#[derive(Clone, Copy)]
struct Winner {
    seconds: u32,
    rental: bool,
    order: usize,
    snap: Snap,
    via: Option<StopIdx>,
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
        if !build.interpolated.is_empty() {
            let message = format!(
                "interpolated blank stop times on {} trip(s)",
                build.interpolated.len()
            );
            let warnings = py.import("warnings")?;
            warnings.call_method1(
                "warn",
                (message, py.get_type::<pyo3::exceptions::PyUserWarning>(), 2),
            )?;
        }
        let transfers = Transfers::empty(build.timetable.stop_count());
        let (stops_by_id, stops_by_qualified_id, trips_by_public_id) =
            derived_indexes(&feed, &build.timetable);
        Ok(TransportNetwork {
            feed,
            build,
            transfers,
            ultra_transfers: None,
            ultra_window: None,
            mcultra_transfers: None,
            mcultra_window: None,
            mcultra_factors: None,
            tbtr_time_transfers: None,
            mctbtr_transfers: None,
            geometry: None,
            leg_geometry: None,
            streets: None,
            stops_by_id,
            stops_by_qualified_id,
            trips_by_public_id,
            streets_bytes_read: 0,
            multimodal: None,
            multimodal_elevation: None,
            multimodal_modes: None,
            multimodal_links: std::sync::OnceLock::new(),
            multimodal_profiles: std::sync::Mutex::new(Vec::new()),
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

    /// Number of ULTRA shortcuts, or `None` when none are computed.
    #[getter]
    fn ultra_shortcut_count(&self) -> Option<usize> {
        self.ultra_transfers.as_ref().map(|set| set.edge_count())
    }

    /// Number of McULTRA shortcuts, or `None` when none are computed.
    #[getter]
    fn mcultra_shortcut_count(&self) -> Option<usize> {
        self.mcultra_transfers.as_ref().map(|set| set.edge_count())
    }

    /// The source-departure window the McULTRA set was computed for, or `None`.
    #[getter]
    fn mcultra_window(&self) -> Option<(u32, u32)> {
        self.mcultra_window
    }

    /// A fingerprint of the McULTRA set's stored factor vector, or `None`.
    /// For inspection/tests only — the activation gate compares the vector
    /// itself (`same_factors`), never this hash.
    #[getter]
    fn _mcultra_factor(&self) -> Option<u64> {
        self.mcultra_factors.as_deref().map(factor_fingerprint)
    }

    /// Whether an emissions query with these `factors` would relax the installed
    /// McULTRA set (a whole-day set whose factor vector matches) rather than
    /// the closure. Exposes the `emissions_transfers` gate for inspection/tests.
    fn mcultra_active_for(&self, factors: Vec<(String, f64)>) -> bool {
        let mut per_trip = vec![f64::NAN; self.build.timetable.trip_count() as usize];
        for (trip_id, factor) in &factors {
            if let Some(&trip) = self.trips_by_public_id.get(trip_id) {
                per_trip[trip.0 as usize] = *factor;
            }
        }
        !std::ptr::eq(self.emissions_transfers(&per_trip), &self.transfers)
    }

    /// The computed ULTRA shortcuts as `(origin_stop_id, destination_stop_id,
    /// seconds, meters)` tuples, or `None` when none are computed. Sorted by
    /// origin then destination, so two runs over the same network return
    /// byte-identical lists.
    fn ultra_shortcuts(&self) -> Option<Vec<(String, String, u32, f64)>> {
        self.ultra_transfers.as_ref().map(|set| {
            let mut shortcuts = Vec::with_capacity(set.edge_count());
            for from in 0..self.build.timetable.stop_count() {
                let origin = self.public_stop_id(StopIdx(from));
                for edge in set.from_stop(StopIdx(from)) {
                    shortcuts.push((
                        origin.clone(),
                        self.public_stop_id(edge.to),
                        edge.duration,
                        edge.meters,
                    ));
                }
            }
            shortcuts
        })
    }

    /// Precompute and cache the trip-based (TBTR) transfer set for `date`.
    ///
    /// The dominance-aware transfer set is TBTR's amortised asset — "build
    /// once, query many". Caching it lets repeated stop `router="tbtr"` matrix
    /// calls on the same date — single-departure or windowed — reuse it instead
    /// of rebuilding it every call, which is where the trip-based engine
    /// pays off: large batches of queries on one network and date. A query on a
    /// different date rebuilds ad hoc. The cached set is persisted with the
    /// artifact (`save`/`load`); recomputing for a new date replaces it.
    fn compute_tbtr_transfers(&mut self, py: Python<'_>, date: &str) -> PyResult<()> {
        let active = self.active_services(date)?;
        let previous = self.active_services_previous(date)?;
        let timetable = &self.build.timetable;
        let set =
            py.allow_threads(|| TbtrEngine::transfers_for_date(timetable, &active, &previous));
        self.tbtr_time_transfers = Some((date.to_string(), set));
        Ok(())
    }

    /// Whether a cached time-only TBTR transfer set is present
    /// (`compute_tbtr_transfers`).
    #[getter]
    fn has_tbtr_transfers(&self) -> bool {
        self.tbtr_time_transfers.is_some()
    }

    /// Number of transfers in the cached time-only TBTR set, or
    /// `None` when none is computed.
    #[getter]
    fn tbtr_transfer_count(&self) -> Option<usize> {
        self.tbtr_time_transfers.as_ref().map(|(_, set)| set.len())
    }

    /// Precompute and cache the multicriteria TBTR transfer set for a
    /// service date and a per-trip emission-factor configuration.
    ///
    /// Every ``router="tbtr"`` multicriteria query on the same date whose
    /// factors match reuses the cached set instead of rebuilding the
    /// dominance-aware precompute each call — the point of the trip-based
    /// engine: large batches on one network, date, and factor set. A query
    /// on another date or with other factors rebuilds ad hoc. The cached
    /// set is persisted with the artifact (`save`/`load`); recomputing
    /// replaces it.
    fn compute_mctbtr_transfers(
        &mut self,
        py: Python<'_>,
        date: &str,
        factors: Vec<(String, f64)>,
    ) -> PyResult<()> {
        let Some(geometry) = &self.geometry else {
            return Err(PyValueError::new_err(
                "no trip distances installed; build the network with trip distances enabled",
            ));
        };
        let active = self.active_services(date)?;
        let previous = self.active_services_previous(date)?;
        let mut per_trip = vec![f64::NAN; self.build.timetable.trip_count() as usize];
        for (trip_id, factor) in &factors {
            if let Some(&trip) = self.trips_by_public_id.get(trip_id) {
                per_trip[trip.0 as usize] = *factor;
            }
        }
        let timetable = &self.build.timetable;
        let set = py.allow_threads(|| {
            McTbtrEngine::transfers_for_date(timetable, geometry, &per_trip, &active, &previous)
        });
        self.mctbtr_transfers = Some((date.to_string(), per_trip, set));
        Ok(())
    }

    /// Whether a cached multicriteria TBTR transfer set is present
    /// (`compute_mctbtr_transfers`).
    #[getter]
    fn has_mctbtr_transfers(&self) -> bool {
        self.mctbtr_transfers.is_some()
    }

    /// Number of transfers in the cached multicriteria TBTR set, or
    /// `None` when none is computed.
    #[getter]
    fn mctbtr_transfer_count(&self) -> Option<usize> {
        self.mctbtr_transfers.as_ref().map(|(_, _, set)| set.len())
    }

    /// Compute the ULTRA intermediate-transfer shortcuts and store them.
    ///
    /// Runs the shortcut search over the unrestricted stop-to-stop
    /// walking graph derived from the installed street network (so the
    /// network must be built with an OSM extract), keeping the minimal
    /// set of intermediate transfers a Pareto-optimal two-trip journey
    /// needs. The result is held in memory (`ultra_shortcut_count`,
    /// `ultra_shortcuts`). Computed **for the whole service day** (the
    /// default window), it is relaxed by the door-to-door time queries
    /// (`route_between_coordinates`, `route_between_stops`, and the point-set
    /// matrices) in place of the closure transfers, giving them unrestricted
    /// walking; the one-to-all stop-destination time queries and the
    /// emissions/fare engines keep the closure. A partial-window set (a
    /// narrower `min_departure`/
    /// `max_departure`) is stored and inspectable but not relaxed by routing
    /// — a journey's source departure can fall outside a bounded window. The
    /// set and its compute window are persisted by `save` and restored by
    /// `load`, so the heavy run-once preprocessing is reusable.
    /// Returns the number of shortcuts. `walking_speed_kmph` sets the
    /// walking pace and `max_transfer_time` bounds an intermediate walk,
    /// in seconds. `min_departure`/`max_departure` bound the
    /// source-departure times the shortcuts serve, in seconds since
    /// midnight (the whole service day by default); a narrower window
    /// costs proportionally less.
    #[pyo3(signature = (
        walking_speed_kmph = 3.6,
        max_transfer_time = 1800.0,
        min_departure = 0,
        max_departure = u32::MAX - 1,
    ))]
    fn compute_ultra_shortcuts(
        &mut self,
        py: Python<'_>,
        walking_speed_kmph: f64,
        max_transfer_time: f64,
        min_departure: u32,
        max_departure: u32,
    ) -> PyResult<usize> {
        if !walking_speed_kmph.is_finite() || walking_speed_kmph <= 0.0 {
            return Err(PyValueError::new_err(
                "walking_speed_kmph must be a positive, finite number",
            ));
        }
        if !max_transfer_time.is_finite() || max_transfer_time < 0.0 {
            return Err(PyValueError::new_err(
                "max_transfer_time must be a non-negative, finite number",
            ));
        }
        if min_departure > max_departure {
            return Err(PyValueError::new_err(
                "min_departure must not exceed max_departure",
            ));
        }
        let speed = walking_speed_kmph / 3.6;
        let stop_count = self.build.timetable.stop_count();
        let timetable = &self.build.timetable;
        let streets = self.installed_streets()?;
        let set = py
            .allow_threads(|| {
                let dense = streets.stop_transfers(speed, max_transfer_time);
                let graph =
                    Transfers::from_edges(stop_count, &dense).map_err(|error| error.to_string())?;
                let view = DayView::universal(timetable);
                let shortcuts: Vec<Shortcut> =
                    compute_shortcuts(&view, timetable, &graph, min_departure, max_departure);
                // The shortcuts carry the walked distance, so they build a
                // routing-ready transfer set directly.
                let edges: Vec<(StopIdx, StopIdx, u32, f64)> = shortcuts
                    .iter()
                    .map(|shortcut| {
                        (
                            shortcut.origin,
                            shortcut.destination,
                            shortcut.seconds,
                            shortcut.meters,
                        )
                    })
                    .collect();
                Transfers::from_edges(stop_count, &edges).map_err(|error| error.to_string())
            })
            .map_err(PyValueError::new_err)?;
        let count = set.edge_count();
        self.ultra_transfers = Some(set);
        self.ultra_window = Some((min_departure, max_departure));
        Ok(count)
    }

    /// Computes and **installs** the McULTRA (emissions-aware) shortcut set,
    /// returning its edge count. The coordinate emissions engine relaxes it in
    /// place of the closure when a whole-day set is installed and the query's
    /// factors match the ones it was built with (`emissions_transfers`). `factors`
    /// is the `trip_factors` table; trips without a finite factor are skipped.
    /// Requires installed streets and trip distances.
    #[pyo3(signature = (walking_speed_kmph, max_transfer_time, factors, min_departure, max_departure))]
    fn compute_mcultra_shortcuts(
        &mut self,
        py: Python<'_>,
        walking_speed_kmph: f64,
        max_transfer_time: f64,
        factors: Vec<(String, f64)>,
        min_departure: u32,
        max_departure: u32,
    ) -> PyResult<usize> {
        if !walking_speed_kmph.is_finite() || walking_speed_kmph <= 0.0 {
            return Err(PyValueError::new_err(
                "walking_speed_kmph must be a positive, finite number",
            ));
        }
        if !max_transfer_time.is_finite() || max_transfer_time < 0.0 {
            return Err(PyValueError::new_err(
                "max_transfer_time must be a non-negative, finite number",
            ));
        }
        if min_departure > max_departure {
            return Err(PyValueError::new_err(
                "min_departure must not exceed max_departure",
            ));
        }
        let Some(geometry) = &self.geometry else {
            return Err(PyValueError::new_err(
                "no trip distances installed; build the network with trip distances enabled",
            ));
        };
        let mut per_trip = vec![f64::NAN; self.build.timetable.trip_count() as usize];
        for (trip_id, factor) in &factors {
            if let Some(&trip) = self.trips_by_public_id.get(trip_id) {
                per_trip[trip.0 as usize] = *factor;
            }
        }
        let speed = walking_speed_kmph / 3.6;
        let stop_count = self.build.timetable.stop_count();
        let timetable = &self.build.timetable;
        let streets = self.installed_streets()?;
        let set = py
            .allow_threads(|| {
                let dense = streets.stop_transfers(speed, max_transfer_time);
                let graph =
                    Transfers::from_edges(stop_count, &dense).map_err(|error| error.to_string())?;
                let view = DayView::universal(timetable);
                let shortcuts = compute_mcultra_shortcuts(
                    &view,
                    timetable,
                    &graph,
                    geometry,
                    &per_trip,
                    min_departure,
                    max_departure,
                );
                // The shortcuts carry the walked distance, so they build a
                // routing-ready transfer set directly (as the ULTRA path does).
                let edges: Vec<(StopIdx, StopIdx, u32, f64)> = shortcuts
                    .iter()
                    .map(|s| (s.origin, s.destination, s.seconds, s.meters))
                    .collect();
                Transfers::from_edges(stop_count, &edges).map_err(|error| error.to_string())
            })
            .map_err(PyValueError::new_err)?;
        let count = set.edge_count();
        self.mcultra_transfers = Some(set);
        self.mcultra_window = Some((min_departure, max_departure));
        self.mcultra_factors = Some(per_trip);
        Ok(count)
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

    /// Install precomputed stop-to-stop transfers from flat arrays.
    ///
    /// The array form of ``set_transfers``: `stop_ids` names each
    /// snapped stop once, `from_index`/`to_index` are positions into
    /// it, and the per-edge payloads cross as numpy arrays — no
    /// per-edge Python objects. The edge set must be transitively
    /// closed, as in ``set_transfers``;
    /// ``cafein.streets.walking_footpaths`` produces this shape.
    fn set_transfer_arrays(
        &mut self,
        stop_ids: Vec<String>,
        from_index: PyReadonlyArray1<'_, u32>,
        to_index: PyReadonlyArray1<'_, u32>,
        seconds: PyReadonlyArray1<'_, u32>,
        meters: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<()> {
        let resolved: Vec<StopIdx> = stop_ids
            .iter()
            .map(|stop| self.resolve_stop(stop))
            .collect::<PyResult<_>>()?;
        let from_index = from_index.as_slice()?;
        let to_index = to_index.as_slice()?;
        let seconds = seconds.as_slice()?;
        let meters = meters.as_slice()?;
        if from_index.len() != to_index.len()
            || from_index.len() != seconds.len()
            || from_index.len() != meters.len()
        {
            return Err(PyValueError::new_err(
                "footpath arrays must all have the same length",
            ));
        }
        let stop_at = |index: usize, position: u32| {
            resolved.get(position as usize).copied().ok_or_else(|| {
                PyValueError::new_err(format!(
                    "footpath {index} references a position outside stop_ids"
                ))
            })
        };
        let mut edges = Vec::with_capacity(from_index.len());
        for (index, (((&from, &to), &duration), &length)) in from_index
            .iter()
            .zip(to_index)
            .zip(seconds)
            .zip(meters)
            .enumerate()
        {
            if !length.is_finite() || length < 0.0 {
                return Err(PyValueError::new_err(format!(
                    "footpath {index} has a negative or non-finite length"
                )));
            }
            edges.push((stop_at(index, from)?, stop_at(index, to)?, duration, length));
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
        // The McULTRA search used the trip geometry to decide emissions-relevant
        // transfers; new distances invalidate the set (ULTRA is distance-free).
        self.mcultra_transfers = None;
        self.mcultra_window = None;
        self.mcultra_factors = None;
        // The cached McTBTR transfer set reduced against the old distances'
        // emissions; new distances invalidate it too (the time-only TBTR set
        // is distance-free and stays).
        self.mctbtr_transfers = None;
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
        // ULTRA and McULTRA shortcuts are derived from the street network; a new
        // one invalidates them.
        self.ultra_transfers = None;
        self.ultra_window = None;
        self.mcultra_transfers = None;
        self.mcultra_window = None;
        self.mcultra_factors = None;
        Ok(())
    }

    /// Installs the multimodal union street graph — a second street section
    /// with per-arc mode permissions, attributes, and optional elevations,
    /// behind the cycling / e-scooter access and egress the PT integration
    /// adds. The payload is the standalone ``StreetNetwork`` constructor's,
    /// produced by the union OSM extraction. The walking graph and every
    /// query over it are untouched.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (modes, vertex_count, edges, coordinate_offsets, longitudes, latitudes,
                        edge_highway, edge_surface, edge_smoothness, edge_flags,
                        access_forward, access_reverse, facility_forward, facility_reverse,
                        coordinate_elevations = None, elevation_metadata = None))]
    fn set_multimodal_streets(
        &mut self,
        modes: Vec<String>,
        vertex_count: u32,
        edges: Vec<(u32, u32, f64)>,
        coordinate_offsets: Vec<u32>,
        longitudes: Vec<f64>,
        latitudes: Vec<f64>,
        edge_highway: Vec<u8>,
        edge_surface: Vec<u8>,
        edge_smoothness: Vec<u8>,
        edge_flags: Vec<u16>,
        access_forward: Vec<u8>,
        access_reverse: Vec<u8>,
        facility_forward: Vec<u8>,
        facility_reverse: Vec<u8>,
        coordinate_elevations: Option<Vec<f32>>,
        elevation_metadata: Option<(String, f64, String, f64, u32)>,
    ) -> PyResult<()> {
        let (inner, elevation) = crate::streets::build_multimodal_core(
            vertex_count,
            edges,
            coordinate_offsets,
            longitudes,
            latitudes,
            edge_highway,
            edge_surface,
            edge_smoothness,
            edge_flags,
            access_forward,
            access_reverse,
            facility_forward,
            facility_reverse,
            coordinate_elevations,
            elevation_metadata,
        )?;
        self.multimodal = Some(inner);
        self.multimodal_elevation = elevation;
        self.multimodal_modes = Some(modes);
        self.multimodal_links = std::sync::OnceLock::new();
        self.multimodal_profiles = std::sync::Mutex::new(Vec::new());
        Ok(())
    }

    /// Whether the multimodal union street graph is installed.
    #[getter]
    fn has_multimodal_streets(&self) -> bool {
        self.multimodal.is_some()
    }

    /// The pruning modes the multimodal graph was built with, or ``None``.
    #[getter]
    fn street_modes(&self) -> Option<Vec<String>> {
        self.multimodal_modes.clone()
    }

    /// A deterministic checksum over the multimodal graph's encoded arrays —
    /// core CSR, geometry, attributes, and elevations alike. Internal; the
    /// round-trip tests compare it across save/load paths.
    #[getter]
    fn _multimodal_checksum(&self) -> Option<u32> {
        self.multimodal.as_ref().map(|network| {
            let (_, bytes) = crate::artifact::encode_streets(&network.to_parts());
            crate::artifact::crc32(&bytes)
        })
    }

    /// Provenance of the multimodal graph's sampled elevations, or ``None``
    /// without a DEM — the dict ``StreetNetwork.elevation_metadata`` returns.
    #[getter]
    fn multimodal_elevation_metadata(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        self.multimodal_elevation
            .as_ref()
            .map(|meta| crate::streets::elevation_dict(py, meta))
            .transpose()
    }

    /// Per-stop street access times from a coordinate over the multimodal
    /// graph: ``(stop_id, seconds, link_edge, link_fraction,
    /// connector_meters)`` for every stop whose mode-masked link is
    /// reachable within ``max_seconds`` under ``mode`` — the trailing triple
    /// is the chosen link's snap identity, the ``StreetChoice`` token. One
    /// forward directed search serves the whole row. Internal until the 12c
    /// policy surface; the stop links snap lazily on first use at the
    /// walking path's 1600 m stop radius.
    fn _street_access_seconds(
        &self,
        py: Python<'_>,
        latitude: f64,
        longitude: f64,
        mode: &str,
        max_seconds: f64,
    ) -> PyResult<Vec<StreetRow>> {
        let (row, targets) = self.street_row(py, latitude, longitude, mode, max_seconds, false)?;
        Ok(self.stop_rows(row, &targets))
    }

    /// The egress mirror of ``_street_access_seconds``: per-stop times *to*
    /// the coordinate, one reverse directed search serving the whole column.
    fn _street_egress_seconds(
        &self,
        py: Python<'_>,
        latitude: f64,
        longitude: f64,
        mode: &str,
        max_seconds: f64,
    ) -> PyResult<Vec<StreetRow>> {
        let (column, sources) =
            self.street_row(py, latitude, longitude, mode, max_seconds, true)?;
        Ok(self.stop_rows(column, &sources))
    }

    /// The installed stop-to-stop transfers as ``(from_stop_id, to_stop_id,
    /// seconds)``. Internal; the street-policy oracles recompute the
    /// reduction's transfer closure from it.
    fn _transfer_edges(&self) -> Vec<(String, String, u32)> {
        let mut edges = Vec::with_capacity(self.transfers.edge_count());
        for from in 0..self.build.timetable.stop_count() {
            let origin = self.public_stop_id(StopIdx(from));
            for edge in self.transfers.from_stop(StopIdx(from)) {
                edges.push((origin.clone(), self.public_stop_id(edge.to), edge.duration));
            }
        }
        edges
    }

    /// The time-only street reduction (design §7.2): per stop, the fastest
    /// permitted choice across the policy's modes over the multimodal graph.
    /// ``modes`` arrive in declared order as ``(mode, max_seconds,
    /// paid_rental, eligible_stops)`` — ``eligible_stops=None`` leaves the
    /// mode unmasked (walking is never masked). Exact-time ties resolve to
    /// fewer paid rentals, then the declared order. Returns
    /// ``(stop_id, seconds, winning_mode)`` per reachable stop — the
    /// ``(stop, seconds)`` array the time engines consume, plus the mode the
    /// reconstruction sidecar keeps. Internal until the policy surface.
    #[allow(clippy::type_complexity)]
    fn _reduced_street_offsets(
        &self,
        py: Python<'_>,
        latitude: f64,
        longitude: f64,
        egress: bool,
        modes: Vec<(String, f64, bool, Option<Vec<String>>)>,
    ) -> PyResult<Vec<ReducedChoice>> {
        let stop_count = self.build.timetable.stop_count() as usize;
        let mut best: Vec<Option<Winner>> = vec![None; stop_count];
        let mut snapped = false;
        for (order, (mode, max_seconds, rental, eligible)) in modes.iter().enumerate() {
            let mask = match eligible {
                Some(stops) => {
                    let mut mask = vec![false; stop_count];
                    for stop in stops {
                        mask[self.resolve_stop(stop)?.0 as usize] = true;
                    }
                    Some(mask)
                }
                None => None,
            };
            let Some((row, links)) =
                self.try_street_row(py, latitude, longitude, mode, *max_seconds, egress)?
            else {
                continue;
            };
            snapped = true;
            for (stop, seconds) in row.into_iter().enumerate() {
                let Some(seconds) = seconds else { continue };
                if mask.as_ref().is_some_and(|mask| !mask[stop]) {
                    continue;
                }
                let Some(snap) = links[stop] else { continue };
                let candidate = Winner {
                    seconds,
                    rental: *rental,
                    order,
                    snap,
                    via: None,
                };
                // Faster wins; a tie falls to fewer paid rentals, then the
                // declared mode order — deterministic whatever the input.
                let wins = match &best[stop] {
                    None => true,
                    Some(held) => {
                        candidate.seconds < held.seconds
                            || (candidate.seconds == held.seconds
                                && (usize::from(candidate.rental), candidate.order)
                                    < (usize::from(held.rental), held.order))
                    }
                };
                if wins {
                    best[stop] = Some(candidate);
                }
            }
        }
        if !snapped {
            return Err(PyValueError::new_err(
                "coordinate too far from the multimodal street network for \
                 every policy mode",
            ));
        }
        // Close the reduction under the installed stop-to-stop transfers:
        // the engines never relax footpaths out of access-seeded stops,
        // because the walking access array is footpath-closed by
        // construction — so a reduced array must be too, or "ride to one
        // platform, walk to the neighbouring one" would go missing and a
        // better seed could only prune the transit label that used to feed
        // that footpath. The installed set is a closure, so one pass
        // suffices; the winner keeps its street mode. Direction follows the
        // labels: access labels propagate along a transfer (reach the seed,
        // then walk it), egress labels against it (walk it first, then
        // leave from the seed) — asymmetric footpaths stay honest.
        for stop in 0..stop_count {
            for transfer in self.transfers.from_stop(StopIdx(stop as u32)) {
                let to = transfer.to.0 as usize;
                let (source, target) = if egress { (to, stop) } else { (stop, to) };
                let Some(winner) = best[source] else {
                    continue;
                };
                let Some(candidate) = winner.seconds.checked_add(transfer.duration) else {
                    continue;
                };
                let wins = match &best[target] {
                    None => true,
                    Some(held) => {
                        candidate < held.seconds
                            || (candidate == held.seconds
                                && (usize::from(winner.rental), winner.order)
                                    < (usize::from(held.rental), held.order))
                    }
                };
                if wins {
                    // The choice token stays the seed's — the vehicle serves
                    // its own snap and the transfer is walked via `via`.
                    best[target] = Some(Winner {
                        seconds: candidate,
                        via: Some(StopIdx(source as u32)),
                        ..winner
                    });
                }
            }
        }
        Ok(best
            .into_iter()
            .enumerate()
            .filter_map(|(stop, winner)| {
                winner.map(|winner| {
                    (
                        self.public_stop_id(StopIdx(stop as u32)),
                        winner.seconds,
                        modes[winner.order].0.clone(),
                        winner.snap.edge,
                        winner.snap.fraction,
                        winner.snap.connector,
                        winner.via.map(|via| self.public_stop_id(via)),
                    )
                })
            })
            .collect())
    }

    /// Builds and installs a contraction hierarchy over the walking graph, so
    /// the bounded one-to-many searches (`access_stops`, `travel_times_*`, the
    /// stop matrices' access/egress) run as hierarchy queries instead of graph
    /// sweeps, at identical results. Heavy, run-once preprocessing; opt-in.
    /// Requires an installed street network. Persisted by `save` and restored by
    /// `load` (the buckets are rebuilt on load), so it need not be run again.
    fn install_walking_hierarchy(&mut self, py: Python<'_>) -> PyResult<()> {
        let streets = self
            .streets
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("no street network is installed"))?;
        py.allow_threads(|| streets.install_hierarchy());
        Ok(())
    }

    /// Whether a walking contraction hierarchy is installed.
    #[getter]
    fn has_walking_hierarchy(&self) -> bool {
        self.streets
            .as_ref()
            .is_some_and(StreetNetwork::has_hierarchy)
    }

    /// Attaches synthetic multimodal edge attributes to the installed street
    /// network, for exercising the format-12 round-trip before the real
    /// producers (OSM extraction, profile compiler) exist. Internal surface;
    /// each array must match the graph's slot/edge shape.
    fn _install_street_attributes(
        &mut self,
        adj_access: Vec<u8>,
        adj_facility: Vec<u8>,
        edge_highway: Vec<u8>,
        edge_surface: Vec<u8>,
        edge_smoothness: Vec<u8>,
        edge_flags: Vec<u16>,
    ) -> PyResult<()> {
        let streets = self
            .streets
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("no street network is installed"))?;
        streets
            .install_street_attributes(StreetAttributes {
                adj_access,
                adj_facility,
                edge_highway,
                edge_surface,
                edge_smoothness,
                edge_flags,
            })
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Attaches synthetic per-coordinate elevations. Internal surface.
    fn _install_elevations(&mut self, elevations: Vec<f32>) -> PyResult<()> {
        let streets = self
            .streets
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("no street network is installed"))?;
        streets
            .install_elevations(elevations)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// The installed street attributes as `(adj_access, adj_facility,
    /// edge_highway, edge_surface, edge_smoothness, edge_flags)`, or `None`.
    /// Internal inspection surface for the round-trip tests.
    fn _street_attributes(&self) -> Option<StreetAttributeArrays> {
        let attributes = self.streets.as_ref()?.street_attributes()?;
        Some((
            attributes.adj_access.clone(),
            attributes.adj_facility.clone(),
            attributes.edge_highway.clone(),
            attributes.edge_surface.clone(),
            attributes.edge_smoothness.clone(),
            attributes.edge_flags.clone(),
        ))
    }

    /// The installed per-coordinate elevations, or `None`. Internal surface.
    fn _street_elevations(&self) -> Option<Vec<f32>> {
        self.streets.as_ref()?.elevations().map(<[f32]>::to_vec)
    }

    /// The installed street network's `(adjacency_slots, edges, coordinates)`
    /// counts, for sizing synthetic attributes in tests. Internal surface.
    fn _street_attribute_shape(&self) -> Option<(u32, u32, u32)> {
        let streets = self.streets.as_ref()?;
        Some((
            2 * streets.edge_count(),
            streets.edge_count(),
            streets.coordinate_count(),
        ))
    }

    /// The number of street-array descriptors this network would save: the 13
    /// core arrays, plus six for an attribute group and one for elevations.
    /// Internal surface for the walk-only-vs-multimodal descriptor red-check.
    fn _street_descriptor_count(&self) -> Option<usize> {
        let streets = self.streets.as_ref()?;
        Some(
            13 + if streets.street_attributes().is_some() {
                6
            } else {
                0
            } + usize::from(streets.elevations().is_some()),
        )
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
    /// max_walking_time : float (optional, default: 7200)
    ///     Walking-time cutoff in seconds.
    /// max_snap_distance : float (optional, default: 1600)
    ///     Maximum straight-line distance in meters from the coordinate
    ///     to the walking network; a coordinate farther away raises
    ///     ``ValueError``.
    ///
    /// Returns
    /// -------
    /// dict
    ///     Walking time in seconds to each reachable stop, keyed by
    ///     stop_id; stops beyond the cutoff are absent.
    #[pyo3(signature = (lat, lon, walking_speed_kmph = 3.6, max_walking_time = 7200.0, max_snap_distance = 1600.0))]
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
}

impl TransportNetwork {
    /// The engine a time-only query's `router` runs on. `"auto"` resolves
    /// to the trip-based engine only when the cached time transfer set
    /// (`compute_tbtr_transfers`) matches the query's date; explicit values
    /// pass through unchanged.
    pub(super) fn resolve_time_router(
        &self,
        router: &str,
        date: &str,
        needs_raptor: bool,
    ) -> PyResult<&'static str> {
        match router {
            "raptor" => Ok("raptor"),
            "tbtr" if needs_raptor => Err(PyValueError::new_err(
                "route/trip/stop exclusions require router='raptor'",
            )),
            "tbtr" => Ok("tbtr"),
            "auto" => {
                let cached = self.tbtr_time_transfers.as_ref().map(|(d, _)| d.as_str());
                Ok(
                    if cafein_core::router::auto_time_tbtr(cached, date, needs_raptor) {
                        "tbtr"
                    } else {
                        "raptor"
                    },
                )
            }
            other => Err(invalid_router(other)),
        }
    }

    /// The intermediate transfers for a query that may carry
    /// exclusions: any exclusion keeps the closure — the ULTRA shortcut
    /// sets' witness pruning is not robust under supply removal.
    pub(super) fn exclusion_transfers(
        &self,
        exclusions: &Option<std::sync::Arc<Exclusions>>,
    ) -> &Transfers {
        if exclusions.is_some() {
            &self.transfers
        } else {
            self.time_transfers()
        }
    }

    /// The time TBTR engine for a query: over the cached transfer set
    /// when its date matches (`compute_tbtr_transfers`), else built ad
    /// hoc. The query-time `footpaths` vary freely — the precompute
    /// never contains them.
    pub(super) fn tbtr_engine<'a>(
        &'a self,
        footpaths: &'a Transfers,
        date: &str,
        active_services: &'a [bool],
        active_services_previous: &'a [bool],
    ) -> TbtrEngine<'a> {
        if let Some((cached_date, set)) = &self.tbtr_time_transfers {
            if cached_date == date {
                return TbtrEngine::from_set(
                    &self.build.timetable,
                    footpaths,
                    active_services,
                    active_services_previous,
                    set,
                );
            }
        }
        TbtrEngine::for_date(
            &self.build.timetable,
            footpaths,
            active_services,
            active_services_previous,
        )
    }

    /// The engine a multicriteria query's `router` runs on. `"auto"`
    /// resolves to the trip-based engine only when the cached McTBTR set
    /// (`compute_mctbtr_transfers`) matches the query's date and factor
    /// vector and the query asks nothing McTBTR cannot answer
    /// (`needs_raptor`); explicit values pass through unchanged.
    pub(super) fn resolve_mc_router(
        &self,
        router: &str,
        date: &str,
        per_trip: &[f64],
        needs_raptor: bool,
    ) -> PyResult<&'static str> {
        match router {
            "raptor" => Ok("raptor"),
            "tbtr" => Ok("tbtr"),
            "auto" => {
                let cached = self
                    .mctbtr_transfers
                    .as_ref()
                    .map(|(d, factors, _)| (d.as_str(), factors.as_slice()));
                Ok(
                    if cafein_core::router::auto_mc_tbtr(cached, date, per_trip, needs_raptor) {
                        "tbtr"
                    } else {
                        "raptor"
                    },
                )
            }
            other => Err(invalid_router(other)),
        }
    }

    /// The multicriteria TBTR engine for a query: over the cached
    /// transfer set when its date and factor vector match
    /// (`compute_mctbtr_transfers`), else built ad hoc. The query-time
    /// `footpaths` vary freely — the precompute never contains them.
    pub(super) fn mctbtr_engine<'a>(
        &'a self,
        footpaths: &'a Transfers,
        geometry: &'a TripGeometry,
        per_trip: &'a [f64],
        date: &str,
        active_services: &'a [bool],
        active_services_previous: &'a [bool],
    ) -> McTbtrEngine<'a> {
        if let Some((cached_date, factors, set)) = &self.mctbtr_transfers {
            if cached_date == date && same_factors(factors, per_trip) {
                return McTbtrEngine::from_set(
                    &self.build.timetable,
                    footpaths,
                    geometry,
                    per_trip,
                    active_services,
                    active_services_previous,
                    set,
                );
            }
        }
        McTbtrEngine::for_date(
            &self.build.timetable,
            footpaths,
            geometry,
            per_trip,
            active_services,
            active_services_previous,
        )
    }

    /// The intermediate-transfer set for the **point-destination** time
    /// queries: the ULTRA shortcuts only when computed **for the whole
    /// service day**, else the closure footpaths. Used by door-to-door
    /// coordinate routing and the point-set matrices, where the street
    /// access/egress search supplies the initial and final walks, so the
    /// transfer set carries only intermediate transfers. Under a whole-day set
    /// the door-to-door RAPTOR time queries all relax it — `route_between_stops`
    /// (via the coordinate path), and the one-to-all `travel_times_from_stop` /
    /// `travel_times_from_coordinate` / `travel_time_matrix`, which pair it with
    /// a bounded per-destination `final_egress` walk for the final leg (see
    /// `ultra_active`). The emissions/fare engines keep the closure: ULTRA is
    /// not emissions-complete. A partial-window set is not relaxed by routing —
    /// a journey's
    /// source-station departure (after access walking and waiting for a first
    /// trip) can fall outside a bounded window, which would silently drop its
    /// transfers — so only a whole-day set is used.
    pub(super) fn time_transfers(&self) -> &Transfers {
        match (self.ultra_transfers.as_ref(), self.ultra_window) {
            (Some(ultra), Some((0, hi))) if hi >= u32::MAX - 1 => ultra,
            _ => &self.transfers,
        }
    }

    /// Whether a whole-day ULTRA set is installed, i.e. `time_transfers`
    /// returns it — the gate for door-to-door stop routing.
    pub(super) fn ultra_active(&self) -> bool {
        matches!(
            (self.ultra_transfers.as_ref(), self.ultra_window),
            (Some(_), Some((0, hi))) if hi >= u32::MAX - 1
        )
    }

    /// The transfer set the coordinate emissions engines relax for a query
    /// resolving to the factor vector `per_trip`: the whole-day McULTRA set
    /// when one is installed for exactly that factor configuration
    /// (`same_factors`), else the closure. A partial-window or
    /// factor-mismatched set is never silently used (§Factor contract).
    pub(super) fn emissions_transfers(&self, per_trip: &[f64]) -> &Transfers {
        match (
            self.mcultra_transfers.as_ref(),
            self.mcultra_window,
            self.mcultra_factors.as_deref(),
        ) {
            (Some(set), Some((0, hi)), Some(built))
                if hi >= u32::MAX - 1 && same_factors(built, per_trip) =>
            {
                set
            }
            _ => &self.transfers,
        }
    }
}

/// The stop-link snap radius onto the multimodal graph, matching the walking
/// path's default stop radius; the 12c policy surface may parameterize it.
const MULTIMODAL_STOP_SNAP: f64 = 1600.0;

impl TransportNetwork {
    /// The compiled multimodal profile for `mode`, cached by exact
    /// definition equality — the multimodal counterpart of the standalone
    /// network's profile cache.
    /// Any built-in mode compiles, whether or not it was in `street_modes`:
    /// per the documented build contract, the modes select *component
    /// pruning only* — every physical edge is kept — so a mode outside the
    /// list routes correctly, just without its small-island pruning.
    fn multimodal_profile(&self, mode: &str) -> PyResult<CompiledStreetProfile> {
        let network = self.multimodal.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "no multimodal street graph is installed; build with street_modes=",
            )
        })?;
        let definition = crate::streets::profile_definition(mode)?;
        let mut cache = self
            .multimodal_profiles
            .lock()
            .expect("profile cache poisoned");
        if let Some((_, compiled)) = cache.iter().find(|(known, _)| known == &definition) {
            return Ok(compiled.clone());
        }
        let compiled = network
            .compile_profile(&definition)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        cache.push((definition, compiled.clone()));
        Ok(compiled)
    }

    /// The lazily built per-stop links onto the multimodal graph: each stop
    /// snapped per mode bit, equal snaps merged under one mask — so a stop
    /// beside a `foot=no` cycleway links for the bicycle without granting
    /// walking that edge, and the other way round.
    fn multimodal_stop_links(&self, network: &StreetNetwork) -> &[Vec<(Snap, u8)>] {
        use cafein_core::streets::{MODE_BICYCLE, MODE_E_SCOOTER, MODE_WALK};
        self.multimodal_links.get_or_init(|| {
            self.stops()
                .iter()
                .map(|(_, latitude, longitude)| {
                    let (Some(latitude), Some(longitude)) = (latitude, longitude) else {
                        return Vec::new();
                    };
                    let mut links: Vec<(Snap, u8)> = Vec::new();
                    for bit in [MODE_WALK, MODE_BICYCLE, MODE_E_SCOOTER] {
                        let Some(snap) = network.snap_for_mode_bit(
                            *latitude,
                            *longitude,
                            MULTIMODAL_STOP_SNAP,
                            bit,
                        ) else {
                            continue;
                        };
                        match links.iter_mut().find(|(known, _)| {
                            known.edge == snap.edge && known.fraction == snap.fraction
                        }) {
                            Some((_, mask)) => *mask |= bit,
                            None => links.push((snap, bit)),
                        }
                    }
                    links
                })
                .collect()
        })
    }

    /// Each stop's link snap for a mode bit, aligned with the stop indices.
    fn mode_link_targets(&self, network: &StreetNetwork, bit: u8) -> Vec<Option<Snap>> {
        self.multimodal_stop_links(network)
            .iter()
            .map(|candidates| {
                candidates
                    .iter()
                    .find(|(_, mask)| mask & bit != 0)
                    .map(|(snap, _)| *snap)
            })
            .collect()
    }

    /// One mode's per-stop row over the multimodal graph — forward (access)
    /// or reverse (egress) — with the zero-coincident convention applied and
    /// the mode-masked links returned beside it.
    fn street_row(
        &self,
        py: Python<'_>,
        latitude: f64,
        longitude: f64,
        mode: &str,
        max_seconds: f64,
        egress: bool,
    ) -> PyResult<ModeRow> {
        self.try_street_row(py, latitude, longitude, mode, max_seconds, egress)?
            .ok_or_else(|| {
                PyValueError::new_err("coordinate too far from the multimodal street network")
            })
    }

    /// [`street_row`](Self::street_row), answering `None` when the
    /// coordinate has no snap for this mode — a mixed policy skips such a
    /// mode instead of aborting.
    fn try_street_row(
        &self,
        py: Python<'_>,
        latitude: f64,
        longitude: f64,
        mode: &str,
        max_seconds: f64,
        egress: bool,
    ) -> PyResult<Option<ModeRow>> {
        let profile = self.multimodal_profile(mode)?;
        let network = self.multimodal.as_ref().expect("profile lookup checked");
        let links = self.mode_link_targets(network, profile.definition.mode.bit());
        let Some(snap) =
            network.snap_for_profile(latitude, longitude, MULTIMODAL_STOP_SNAP, &profile)
        else {
            return Ok(None);
        };
        let mut row = py.allow_threads(|| {
            if egress {
                network.directed_times_from_snaps(&links, &snap, &profile, max_seconds)
            } else {
                network.directed_times_to_snaps(&snap, &links, &profile, max_seconds)
            }
        });
        self.zero_coincident_stops(&mut row, &links, latitude, longitude, max_seconds);
        Ok(Some((row, links)))
    }

    /// A stop at the query's own coordinate is zero away — routing it through
    /// the network would charge that coordinate's connector twice, exactly
    /// the convention `directed_matrix` applies. The link (and so the snap
    /// token) must exist and the cutoff must admit a zero-length trip.
    fn zero_coincident_stops(
        &self,
        row: &mut [Option<u32>],
        links: &[Option<Snap>],
        latitude: f64,
        longitude: f64,
        max_seconds: f64,
    ) {
        if !max_seconds.is_finite() || max_seconds < 0.0 {
            return;
        }
        for (index, (_, stop_latitude, stop_longitude)) in self.stops().iter().enumerate() {
            if *stop_latitude == Some(latitude)
                && *stop_longitude == Some(longitude)
                && links[index].is_some()
            {
                row[index] = Some(0);
            }
        }
    }

    /// A row of per-stop times as `(public stop id, seconds, link edge, link
    /// fraction, connector meters)` — the trailing triple is the chosen
    /// link's snap identity, the `StreetChoice` token the 12c reduction
    /// keeps beside the winning duration. `None` cells dropped.
    fn stop_rows(&self, row: Vec<Option<u32>>, links: &[Option<Snap>]) -> Vec<StreetRow> {
        row.into_iter()
            .zip(links)
            .enumerate()
            .filter_map(|(stop, (seconds, link))| {
                let value = seconds?;
                let snap = link.as_ref()?;
                Some((
                    self.public_stop_id(StopIdx(stop as u32)),
                    value,
                    snap.edge,
                    snap.fraction,
                    snap.connector,
                ))
            })
            .collect()
    }
}
