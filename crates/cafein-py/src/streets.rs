//! The standalone street network exposed to Python.
//!
//! A `TransportNetwork` carries a street network for walking access and
//! egress; this is the street graph on its own — built from the union OSM
//! extraction, routable by any compiled profile. Journeys never enter it, so
//! it holds no timetable, transfers, or stop links.

use numpy::{IntoPyArray, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use rayon::prelude::*;

use cafein_core::geometry::wkb_line_string;
use cafein_core::streets::{
    Backing, CarCostModel, CarEdgeAttributes, CompiledStreetProfile, EdgeAttributes, MappedStreets,
    Snap, StreetLeg, StreetNetwork as CoreStreetNetwork, StreetProfileDefinition,
};

use crate::artifact::{
    corrupted, crc32, decode_optional_street_arrays, decode_streets, encode_streets, io_error,
    parse_container, validate_street_shape, write_container, ElevationMeta, MappedArtifact,
    MmapMode, StreetsMeta, STREET_ARRAY_ORDER, STREET_ARTIFACT_FORMAT, STREET_ARTIFACT_MAGIC,
};

/// The street-mode names accepted by the public API, with the shipped profile
/// definition each resolves to. The car resolves to its free-flow default;
/// a resolved delay model arrives separately (see [`car_cost_model`]).
pub(super) fn profile_definition(mode: &str) -> PyResult<StreetProfileDefinition> {
    match mode {
        "walk" => Ok(StreetProfileDefinition::walk()),
        "bicycle" => Ok(StreetProfileDefinition::bicycle()),
        "e_bike" => Ok(StreetProfileDefinition::e_bike()),
        "e_scooter" => Ok(StreetProfileDefinition::e_scooter()),
        "car" => Ok(StreetProfileDefinition::car(None)),
        other => Err(PyValueError::new_err(format!(
            "unknown street mode '{other}'; expected one of \
             'walk', 'bicycle', 'e_bike', 'e_scooter', 'car'"
        ))),
    }
}

/// The resolved car delay model as the Python layer passes it: one period's
/// flat numbers — `(group_seconds, groups, ramp_share_high, ramp_share_low,
/// ramp_multiplier, congestion_multiplier)`. Python resolves the period and
/// merges any `delay_model=` override; the numbers validate in
/// `StreetProfileDefinition::validate`.
pub(super) type CarModelPayload = (Vec<f64>, Vec<u8>, f64, f64, f64, f64);

fn car_cost_model(payload: &CarModelPayload) -> PyResult<CarCostModel> {
    let (seconds, groups, share_high, share_low, ramp_multiplier, congestion_multiplier) = payload;
    let group_seconds: [f64; 3] = seconds.as_slice().try_into().map_err(|_| {
        PyValueError::new_err("the car delay model carries one value per road-class group (3)")
    })?;
    Ok(CarCostModel {
        group_seconds,
        groups: groups.clone(),
        ramp_share_high: *share_high,
        ramp_share_low: *share_low,
        ramp_multiplier: *ramp_multiplier,
        congestion_multiplier: *congestion_multiplier,
    })
}

/// One reachable cost-matrix cell, whichever search produced it: the numbers
/// every query reports, and the shape only a `geometries=True` query builds.
#[derive(Clone)]
struct CostCell {
    seconds: u32,
    network_meters: f64,
    connector_meters: f64,
    /// The leg's shape in (longitude, latitude); `None` without geometries.
    geometry: Option<Vec<(f64, f64)>>,
}

impl CostCell {
    fn from_leg(leg: StreetLeg) -> CostCell {
        CostCell {
            seconds: leg.seconds,
            network_meters: leg.network_meters,
            connector_meters: leg.connector_meters,
            geometry: Some(leg.geometry),
        }
    }
}

/// A routable street network built from an OpenStreetMap extract.
#[pyclass]
pub struct StreetNetwork {
    inner: CoreStreetNetwork,
    /// Compiled profiles kept by the exact definition that produced them, so
    /// repeated queries reuse the per-arc millisecond array instead of
    /// rescanning every arc. Keyed by full equality, never by name: a changed
    /// speed or multiplier compiles afresh rather than reusing stale costs.
    profiles: Vec<(StreetProfileDefinition, CompiledStreetProfile)>,
    /// STREETS-section bytes a load explicitly read: the whole section for an
    /// owned load, and just the owned optional arrays for a lazy mapped one.
    /// Zero only for a network built from arrays rather than loaded.
    streets_bytes_read: u64,
    /// The artifact file this network was loaded from, with the header
    /// CRCs read at load time — the streaming fingerprint's stable
    /// cross-process identity; `None` for in-memory builds.
    source: Option<(String, u32, u32)>,
    /// What the installed elevations mean; `None` without elevations.
    elevation: Option<ElevationMeta>,
}

#[pymethods]
impl StreetNetwork {
    /// Builds the network from the union extraction's flat arrays.
    ///
    /// Every attribute array carries one entry per physical edge, in the same
    /// order as `edges`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (vertex_count, edges, coordinate_offsets, longitudes, latitudes,
                        edge_highway, edge_surface, edge_smoothness, edge_flags,
                        access_forward, access_reverse, facility_forward, facility_reverse,
                        coordinate_elevations = None, elevation_metadata = None,
                        car_attributes = None))]
    fn new(
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
        car_attributes: Option<CarPayload>,
    ) -> PyResult<StreetNetwork> {
        let (inner, elevation) = build_multimodal_core(
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
            car_attributes,
        )?;
        Ok(StreetNetwork {
            inner,
            profiles: Vec::new(),
            streets_bytes_read: 0,
            elevation,
            source: None,
        })
    }

    /// Save the street network as a reusable artifact.
    ///
    /// Carries the street graph, its geometry, and the multimodal permission
    /// and attribute arrays behind a versioned, checksummed header, so batch
    /// jobs can ``load`` the file instead of re-running the OSM extraction.
    /// The file is staged beside the destination and atomically renamed into
    /// place, so saving over an artifact never rewrites it under live mapped
    /// readers.
    fn save(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        py.allow_threads(|| {
            // Snapshot inside the closure: `to_parts` copies every street array,
            // which at country scale is the bulk of the save and must not hold
            // the GIL.
            let parts = self.inner.to_parts();
            let (descriptors, streets_bytes) = encode_streets(&parts);
            let meta = bincode::serialize(&StreetsMeta {
                vertex_count: parts.vertex_count,
                links: parts.links.clone(),
                descriptors,
                elevation: self.elevation.clone(),
            })
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
            write_container(
                path,
                STREET_ARTIFACT_MAGIC,
                STREET_ARTIFACT_FORMAT,
                &meta,
                &streets_bytes,
            )
        })
    }

    /// A deterministic content digest over everything `save` would
    /// persist — the artifact identity without writing a file, as a
    /// SHA-256 hex string. Serializes the payload on demand (the
    /// street arrays encode into one transient section buffer); the
    /// streaming fingerprint calls it once per run.
    fn _artifact_checksum(&self, py: Python<'_>) -> PyResult<String> {
        use sha2::{Digest, Sha256};

        if let Some((path, meta_crc, streets_crc)) = &self.source {
            if let Some(digest) =
                py.allow_threads(|| crate::artifact::file_digest(path, *meta_crc, *streets_crc))
            {
                return Ok(digest);
            }
        }
        py.allow_threads(|| {
            let parts = self.inner.to_parts();
            let (descriptors, streets_bytes) = encode_streets(&parts);
            let meta = StreetsMeta {
                vertex_count: parts.vertex_count,
                links: parts.links.clone(),
                descriptors,
                elevation: self.elevation.clone(),
            };
            let mut hasher = Sha256::new();
            hasher.update(STREET_ARTIFACT_MAGIC);
            hasher.update(STREET_ARTIFACT_FORMAT.to_le_bytes());
            bincode::serialize_into(crate::artifact::HashWriter(&mut hasher), &meta)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            hasher.update(b"\0STREETS");
            hasher.update((streets_bytes.len() as u64).to_le_bytes());
            hasher.update(&streets_bytes);
            Ok(crate::artifact::hex_digest(hasher))
        })
    }

    /// Load a street network saved with ``save``.
    ///
    /// ``mmap='auto'`` maps the file and uses the street arrays in place,
    /// falling back to the owned load where mapping is unavailable;
    /// ``'require'`` errors instead of falling back. ``verify`` toggles the
    /// STREETS checksum: default on for owned loads (the bytes are read
    /// anyway), off for mapped loads (the check would page the whole section
    /// in). Artifacts are trusted input, like pickles: load only files you
    /// created.
    #[staticmethod]
    #[pyo3(signature = (path, mmap = "off", verify = None))]
    fn load(
        py: Python<'_>,
        path: &str,
        mmap: &str,
        verify: Option<bool>,
    ) -> PyResult<StreetNetwork> {
        let mode = match mmap {
            "off" => MmapMode::Off,
            "auto" => MmapMode::Auto,
            "require" => MmapMode::Require,
            other => {
                return Err(PyValueError::new_err(format!(
                    "mmap must be 'off', 'auto', or 'require', not '{other}'"
                )))
            }
        };
        let source =
            crate::artifact::read_source(path, STREET_ARTIFACT_MAGIC, STREET_ARTIFACT_FORMAT);
        if mode != MmapMode::Off {
            match py.allow_threads(|| load_street_mapped(path, verify))? {
                Ok((inner, bytes_read, elevation)) => {
                    return Ok(StreetNetwork::adopt(inner, bytes_read, elevation, source))
                }
                Err(reason) if mode == MmapMode::Require => {
                    return Err(PyValueError::new_err(format!(
                        "'{path}' cannot be memory-mapped ({reason}) and \
                         mmap='require' forbids the owned fallback"
                    )))
                }
                Err(_) => {}
            }
        }
        let (inner, bytes_read, elevation) =
            py.allow_threads(|| load_street_owned(path, verify))?;
        Ok(StreetNetwork::adopt(inner, bytes_read, elevation, source))
    }

    #[allow(clippy::too_many_arguments)]
    /// The reached street vertices of a mode spread from `origin`:
    /// `(latitudes, longitudes, costs)` arrays under `mode`, bounded
    /// by `budget` on the chosen `axis` — seconds for `"time"`,
    /// street metres for `"distance"`. A street catchment's target
    /// universe.
    #[pyo3(signature = (origin, mode, budget, axis, max_snap_distance, car_model = None))]
    fn _reached_vertices(
        &mut self,
        py: Python<'_>,
        origin: (f64, f64),
        mode: &str,
        budget: f64,
        axis: &str,
        max_snap_distance: f64,
        car_model: Option<CarModelPayload>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        if axis != "time" && axis != "distance" {
            return Err(PyValueError::new_err(format!(
                "axis must be 'time' or 'distance', not {axis:?}"
            )));
        }
        let index = self.compiled(mode, car_model.as_ref())?;
        let (_, profile) = &self.profiles[index];
        let field: Vec<(f64, f64, f64)> = py.allow_threads(|| {
            let (latitude, longitude) = origin;
            let Some(snap) =
                self.inner
                    .snap_for_profile(latitude, longitude, max_snap_distance, profile)
            else {
                return Vec::new();
            };
            let reached = if axis == "time" {
                self.inner.directed_reached_vertices(&snap, profile, budget)
            } else {
                self.inner
                    .directed_reached_vertices_meters(&snap, profile, budget)
            };
            let positions = self.inner.vertex_positions();
            reached
                .into_iter()
                .filter_map(|(vertex, cost)| {
                    let (lon, lat) = positions[vertex as usize];
                    (lon.is_finite() && lat.is_finite()).then_some((lat, lon, cost))
                })
                .collect()
        });
        let lats: Vec<f64> = field.iter().map(|&(lat, _, _)| lat).collect();
        let lons: Vec<f64> = field.iter().map(|&(_, lon, _)| lon).collect();
        let costs: Vec<f64> = field.iter().map(|&(_, _, c)| c).collect();
        Ok((
            lats.into_pyarray(py).into_any().unbind(),
            lons.into_pyarray(py).into_any().unbind(),
            costs.into_pyarray(py).into_any().unbind(),
        ))
    }

    /// The origins × destinations travel-time matrix under `mode`, in whole
    /// seconds with `u32::MAX` for unreachable.
    ///
    /// Returns `matrix` alongside the index lists of the coordinates that did
    /// not snap, matching the dict the transit point matrices return.
    /// Coordinates are `(latitude, longitude)` in EPSG:4326.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (origins, destinations, mode, max_seconds, max_snap_distance,
                        car_model = None))]
    fn travel_time_matrix(
        &mut self,
        py: Python<'_>,
        origins: Vec<(f64, f64)>,
        destinations: Vec<(f64, f64)>,
        mode: &str,
        max_seconds: f64,
        max_snap_distance: f64,
        car_model: Option<CarModelPayload>,
    ) -> PyResult<Py<PyDict>> {
        let index = self.compiled(mode, car_model.as_ref())?;
        let (_, profile) = &self.profiles[index];
        let destination_count = destinations.len();
        let (rows, unsnapped_from, unsnapped_to) = py.allow_threads(|| {
            self.inner.directed_matrix(
                &origins,
                &destinations,
                profile,
                max_seconds,
                max_snap_distance,
            )
        });
        let mut flat = Vec::with_capacity(rows.len() * destination_count);
        for row in &rows {
            flat.extend(row.iter().map(|cell| cell.unwrap_or(u32::MAX)));
        }
        let table = PyDict::new(py);
        table.set_item(
            "matrix",
            flat.into_pyarray(py)
                .reshape([rows.len(), destination_count])?,
        )?;
        table.set_item("unsnapped_from", unsnapped_from.into_pyarray(py))?;
        table.set_item("unsnapped_to", unsnapped_to.into_pyarray(py))?;
        Ok(table.into())
    }

    /// The origins × destinations cost rows under `mode`, in long format.
    ///
    /// Only the reachable cells are returned — a cost row is far heavier than a
    /// time cell, so a dense matrix of them would be mostly waste. Coordinates
    /// are `(latitude, longitude)`; `geometries` attaches each row's shape as
    /// WKB.
    ///
    /// Without `geometries` the rows come from the metres-only search, so a
    /// time/distance matrix never assembles a shape it would only discard;
    /// the numbers are the reconstructed legs' cell for cell.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (origins, destinations, mode, max_seconds, max_snap_distance,
                        geometries, car_model = None))]
    fn cost_matrix(
        &mut self,
        py: Python<'_>,
        origins: Vec<(f64, f64)>,
        destinations: Vec<(f64, f64)>,
        mode: &str,
        max_seconds: f64,
        max_snap_distance: f64,
        geometries: bool,
        car_model: Option<CarModelPayload>,
    ) -> PyResult<Py<PyDict>> {
        let index = self.compiled(mode, car_model.as_ref())?;
        let (_, profile) = &self.profiles[index];
        let rows = py.allow_threads(|| {
            let snap = |&(latitude, longitude): &(f64, f64)| {
                self.inner
                    .snap_for_profile(latitude, longitude, max_snap_distance, profile)
            };
            let targets: Vec<((f64, f64), Option<Snap>)> = destinations
                .par_iter()
                .map(|&point| (point, snap(&point)))
                .collect();
            let origin_snaps: Vec<Option<Snap>> = origins.par_iter().map(snap).collect();
            // The metres-only search takes the destination snaps on their own,
            // so they are split off once rather than per row.
            let target_snaps: Vec<Option<Snap>> = targets.iter().map(|&(_, snap)| snap).collect();
            // The same-coordinate zero is still a route the cutoff has to
            // admit, so a cutoff admitting nothing leaves the cell unreachable
            // — matching the single route and the time matrix.
            let routable = max_seconds.is_finite() && max_seconds >= 0.0;
            let cells: Vec<Vec<Option<CostCell>>> = origin_snaps
                .par_iter()
                .zip(&origins)
                .map(|(origin, &point)| match origin {
                    None => vec![None; destinations.len()],
                    Some(from) => {
                        let mut row: Vec<Option<CostCell>> = if geometries {
                            self.inner
                                .directed_legs_to_snaps(point, from, &targets, profile, max_seconds)
                                .into_iter()
                                .map(|leg| leg.map(CostCell::from_leg))
                                .collect()
                        } else {
                            self.inner
                                .directed_meters_to_snaps(from, &target_snaps, profile, max_seconds)
                                .into_iter()
                                .zip(&target_snaps)
                                .map(|(cell, target)| {
                                    let (seconds, network_meters) = cell?;
                                    // A cell settles only against a snapped
                                    // destination, so the connectors at both
                                    // ends are the snaps' own.
                                    let to = target.as_ref()?;
                                    Some(CostCell {
                                        seconds,
                                        network_meters,
                                        connector_meters: from.connector + to.connector,
                                        geometry: None,
                                    })
                                })
                                .collect()
                        };
                        if routable {
                            // A coordinate is no distance and no time from
                            // itself. Its shape is the degenerate two-point
                            // line, so it stays a valid LineString.
                            for (cell, (destination, target)) in row.iter_mut().zip(&targets) {
                                if *destination == point && target.is_some() {
                                    *cell = Some(CostCell {
                                        seconds: 0,
                                        network_meters: 0.0,
                                        connector_meters: 0.0,
                                        geometry: geometries
                                            .then(|| vec![(point.1, point.0), (point.1, point.0)]),
                                    });
                                }
                            }
                        }
                        row
                    }
                })
                .collect();
            (cells, origin_snaps, target_snaps)
        });
        let (cells, origin_snaps, target_snaps) = rows;
        let (mut from, mut to, mut seconds) = (Vec::new(), Vec::new(), Vec::new());
        let (mut network, mut connector) = (Vec::new(), Vec::new());
        let shapes = PyList::empty(py);
        for (origin, row) in cells.iter().enumerate() {
            for (destination, cell) in row.iter().enumerate() {
                let Some(cell) = cell else { continue };
                from.push(origin as u32);
                to.push(destination as u32);
                seconds.push(cell.seconds);
                network.push(cell.network_meters);
                connector.push(cell.connector_meters);
                if let Some(geometry) = &cell.geometry {
                    shapes.append(PyBytes::new(py, &wkb_line_string(geometry)))?;
                }
            }
        }
        let unsnapped = |snaps: &[Option<Snap>]| -> Vec<u32> {
            snaps
                .iter()
                .enumerate()
                .filter(|(_, snap)| snap.is_none())
                .map(|(index, _)| index as u32)
                .collect()
        };
        let table = PyDict::new(py);
        table.set_item("from", from.into_pyarray(py))?;
        table.set_item("to", to.into_pyarray(py))?;
        table.set_item("travel_time_s", seconds.into_pyarray(py))?;
        table.set_item("network_distance", network.into_pyarray(py))?;
        table.set_item("connector_distance", connector.into_pyarray(py))?;
        if geometries {
            table.set_item("geometry", shapes)?;
        }
        table.set_item("unsnapped_from", unsnapped(&origin_snaps).into_pyarray(py))?;
        table.set_item("unsnapped_to", unsnapped(&target_snaps).into_pyarray(py))?;
        Ok(table.into())
    }

    /// What the installed elevations mean, as a dict, or `None` when the
    /// network carries no elevations.
    #[getter]
    fn elevation_metadata(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        self.elevation
            .as_ref()
            .map(|meta| elevation_dict(py, meta))
            .transpose()
    }

    /// The stored per-coordinate elevations, aligned with `_coordinates`.
    /// Internal; the elevation intake tests assert on the stored values.
    #[getter]
    fn _coordinate_elevations(&self) -> Option<Vec<f32>> {
        self.inner.elevations().map(<[f32]>::to_vec)
    }

    /// The stored geometry coordinates as `(longitude, latitude)` degrees.
    /// Internal; pairs with `_coordinate_elevations` in tests.
    #[getter]
    fn _coordinates(&self) -> Vec<(f64, f64)> {
        self.inner.coordinates()
    }

    /// The stored coordinates' bounding box as `(west, south, east, north)`
    /// degrees, or `None` for an empty geometry section. Internal; the zone
    /// generators take a network's extent from it.
    #[getter]
    fn _coordinate_bounds(&self) -> Option<(f64, f64, f64, f64)> {
        self.inner.coordinate_bounds()
    }

    /// The stored per-slot car arrays as `(speeds, junctions)`, or `None`
    /// without a car build. Internal; the round-trip tests assert on them.
    #[getter]
    fn _car_attributes(&self) -> Option<(Vec<f32>, Vec<u8>)> {
        self.inner
            .car_attributes()
            .map(|car| (car.adj_car_speed.clone(), car.adj_junction.clone()))
    }

    /// Whether the street arrays are memory-mapped views of the artifact.
    #[getter]
    fn mapped(&self) -> bool {
        self.inner.is_mapped()
    }

    /// STREETS-section bytes the load explicitly read. An owned load reads the
    /// whole section; a lazy mapped load leaves the core CSR and geometry
    /// mapped and reads only the optional multimodal arrays, which are decoded
    /// owned — so it is non-zero but well under the section length. Internal;
    /// the laziness tests assert on it.
    #[getter]
    fn _streets_bytes_read(&self) -> u64 {
        self.streets_bytes_read
    }

    #[getter]
    fn vertex_count(&self) -> u32 {
        self.inner.vertex_count()
    }

    #[getter]
    fn edge_count(&self) -> u32 {
        self.inner.edge_count()
    }

    /// The travel time in whole seconds from `origin` to `destination` under
    /// `mode`, or `None` when the destination is not reachable within
    /// `max_seconds`.
    ///
    /// Coordinates are `(latitude, longitude)` in EPSG:4326. A coordinate that
    /// does not snap to the network raises `ValueError` — unreachable and
    /// unsnappable are different answers.
    #[pyo3(signature = (origin, destination, mode, max_seconds, max_snap_distance,
                        car_model = None))]
    fn travel_time(
        &mut self,
        origin: (f64, f64),
        destination: (f64, f64),
        mode: &str,
        max_seconds: f64,
        max_snap_distance: f64,
        car_model: Option<CarModelPayload>,
    ) -> PyResult<Option<u32>> {
        // Compile first: snapping is profile-aware, so that a bicycle query
        // does not land on a footway it may not enter.
        let index = self.compiled(mode, car_model.as_ref())?;
        let (_, profile) = &self.profiles[index];
        let from = self.snap_endpoint(origin, max_snap_distance, "origin", profile)?;
        let to = self.snap_endpoint(destination, max_snap_distance, "destination", profile)?;
        // A coordinate routes to itself in no time. The connector is the walk
        // between the coordinate and its street, so routing through the network
        // would charge it twice — leaving a point a positive distance from
        // itself. Snapping still runs first, so a coordinate off the network is
        // an error rather than a silent zero.
        if origin == destination {
            return Ok((max_seconds.is_finite() && max_seconds >= 0.0).then_some(0));
        }
        Ok(self
            .inner
            .directed_travel_time(&from, &to, profile, max_seconds))
    }
}

impl StreetNetwork {
    /// Wraps a loaded core network. Profiles are compiled on first query, so a
    /// loaded network starts with an empty cache.
    fn adopt(
        inner: CoreStreetNetwork,
        streets_bytes_read: u64,
        elevation: Option<ElevationMeta>,
        source: Option<(String, u32, u32)>,
    ) -> StreetNetwork {
        StreetNetwork {
            inner,
            profiles: Vec::new(),
            streets_bytes_read,
            elevation,
            source,
        }
    }

    /// Snaps one endpoint, naming it in the error so a failure says which
    /// coordinate was off the network.
    fn snap_endpoint(
        &self,
        (latitude, longitude): (f64, f64),
        max_snap_distance: f64,
        role: &str,
        profile: &CompiledStreetProfile,
    ) -> PyResult<Snap> {
        self.inner
            .snap_for_profile(latitude, longitude, max_snap_distance, profile)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "the {role} ({latitude}, {longitude}) is farther than \
                     {max_snap_distance} m from any street '{}' can use",
                    profile.definition.name
                ))
            })
    }

    /// The position of `mode`'s compiled profile, compiling and caching it on
    /// first use. Returns an index rather than a reference so the cache can be
    /// extended while the network stays borrowed. A resolved car delay model
    /// joins the definition, so each model caches separately by equality.
    fn compiled(&mut self, mode: &str, car_model: Option<&CarModelPayload>) -> PyResult<usize> {
        let mut definition = profile_definition(mode)?;
        if let Some(payload) = car_model {
            if mode != "car" {
                return Err(PyValueError::new_err(
                    "the intersection-delay model applies to mode='car' only",
                ));
            }
            definition.car = Some(car_cost_model(payload)?);
        }
        if let Some(index) = self
            .profiles
            .iter()
            .position(|(cached, _)| *cached == definition)
        {
            return Ok(index);
        }
        let compiled = self
            .inner
            .compile_profile(&definition)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        self.profiles.push((definition, compiled));
        Ok(self.profiles.len() - 1)
    }
}

/// The META payload of a street artifact, and the section bounds its arrays
/// live in.
fn parse_street_meta(
    path: &str,
    bytes: &[u8],
    verify_streets: bool,
) -> PyResult<(StreetsMeta, crate::artifact::ContainerLayout)> {
    let layout = parse_container(path, bytes, STREET_ARTIFACT_MAGIC, STREET_ARTIFACT_FORMAT)?;
    let meta =
        &bytes[layout.meta_offset as usize..(layout.meta_offset + layout.meta_length) as usize];
    if crc32(meta) != layout.meta_crc {
        return Err(corrupted(path, "checksum mismatch"));
    }
    if verify_streets {
        let section = &bytes[layout.streets_offset as usize
            ..(layout.streets_offset + layout.streets_length) as usize];
        if crc32(section) != layout.streets_crc {
            return Err(corrupted(path, "checksum mismatch"));
        }
    }
    let streets_meta: StreetsMeta =
        bincode::deserialize(meta).map_err(|error| PyValueError::new_err(error.to_string()))?;
    if layout.streets_length == 0 {
        return Err(corrupted(path, "missing street section"));
    }
    Ok((streets_meta, layout))
}

/// Loads a street artifact into owned memory — the default path.
type LoadedStreets = (CoreStreetNetwork, u64, Option<ElevationMeta>);

/// The elevation metadata as the Python-facing dict, shared by the
/// standalone and the transit-carried multimodal graphs.
pub(super) fn elevation_dict(py: Python<'_>, meta: &ElevationMeta) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("source", &meta.source)?;
    dict.set_item("sampling_interval", meta.sampling_interval)?;
    dict.set_item("nodata_policy", &meta.nodata_policy)?;
    dict.set_item("coverage", meta.coverage)?;
    dict.set_item("inferred_edges", meta.inferred_edges)?;
    Ok(dict.into())
}

/// The car payload as the Python surface passes it: per-edge
/// `(speed_forward, speed_reverse, junction_forward, junction_reverse)`,
/// all four together or the group absent.
pub(super) type CarPayload = (Vec<f32>, Vec<f32>, Vec<u8>, Vec<u8>);

/// Builds the multimodal street core and its validated elevation metadata
/// from the union extraction's flat arrays — shared by the standalone
/// `StreetNetwork` constructor and the `TransportNetwork`'s multimodal
/// street installer, so both enforce identical invariants.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_multimodal_core(
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
    car_attributes: Option<CarPayload>,
) -> PyResult<(CoreStreetNetwork, Option<ElevationMeta>)> {
    if coordinate_elevations.is_some() != elevation_metadata.is_some() {
        return Err(PyValueError::new_err(
            "coordinate_elevations and elevation_metadata go together: \
             pass both or neither",
        ));
    }
    let inner = CoreStreetNetwork::new_multimodal(
        vertex_count,
        0,
        &edges,
        &coordinate_offsets,
        &longitudes,
        &latitudes,
        Vec::new(),
        EdgeAttributes {
            highway: &edge_highway,
            surface: &edge_surface,
            smoothness: &edge_smoothness,
            flags: &edge_flags,
            access_forward: &access_forward,
            access_reverse: &access_reverse,
            facility_forward: &facility_forward,
            facility_reverse: &facility_reverse,
            car: car_attributes.as_ref().map(
                |(speed_forward, speed_reverse, junction_forward, junction_reverse)| {
                    CarEdgeAttributes {
                        speed_forward,
                        speed_reverse,
                        junction_forward,
                        junction_reverse,
                    }
                },
            ),
        },
        coordinate_elevations.as_deref(),
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let elevation = elevation_metadata.map(
        |(source, sampling_interval, nodata_policy, coverage, inferred_edges)| ElevationMeta {
            source,
            sampling_interval,
            nodata_policy,
            coverage,
            inferred_edges,
        },
    );
    if let Some(meta) = &elevation {
        check_elevation_meta(meta, inner.edge_count()).map_err(PyValueError::new_err)?;
    }
    Ok((inner, elevation))
}

/// The metadata's declared invariants: a positive finite sampling interval,
/// a coverage share within 0..=1, and an inferred count no larger than the
/// edge set it describes. Shared by construction and both load paths.
pub(super) fn check_elevation_meta(meta: &ElevationMeta, edge_count: u32) -> Result<(), String> {
    if !(meta.sampling_interval.is_finite() && meta.sampling_interval > 0.0) {
        return Err("elevation sampling_interval must be positive and finite".into());
    }
    if !(meta.coverage.is_finite() && (0.0..=1.0).contains(&meta.coverage)) {
        return Err("elevation coverage must be a share within 0..=1".into());
    }
    if meta.inferred_edges > edge_count {
        return Err("elevation inferred_edges exceeds the edge count".into());
    }
    Ok(())
}

fn load_street_owned(path: &str, verify: Option<bool>) -> PyResult<LoadedStreets> {
    let bytes = std::fs::read(path).map_err(io_error)?;
    let (streets_meta, layout) = parse_street_meta(path, &bytes, verify.unwrap_or(true))?;
    let elevation = streets_meta.elevation.clone();
    let section = &bytes
        [layout.streets_offset as usize..(layout.streets_offset + layout.streets_length) as usize];
    // A street artifact carries no stops, so no link may reference one.
    let (expected_levels, block_end) =
        validate_street_shape(path, &streets_meta, 0, layout.streets_length, 0)?;
    if block_end != layout.streets_length {
        return Err(corrupted(path, "street array bounds"));
    }
    let parts = decode_streets(path, streets_meta, section, expected_levels)?;
    let inner =
        CoreStreetNetwork::from_parts(parts).map_err(|_| corrupted(path, "street attributes"))?;
    // Metadata is absent exactly when the elevations are, and its fields
    // must hold their declared invariants.
    if inner.elevations().is_some() != elevation.is_some() {
        return Err(corrupted(path, "elevation metadata"));
    }
    if let Some(meta) = &elevation {
        if check_elevation_meta(meta, inner.edge_count()).is_err() {
            return Err(corrupted(path, "elevation metadata"));
        }
    }
    Ok((inner, layout.streets_length, elevation))
}

/// Loads a street artifact with the core arrays as views into a memory map.
/// `Ok(Err(reason))` means mapping is environmentally unavailable and the
/// caller decides between fallback and error; artifact problems are hard
/// errors on every path.
fn load_street_mapped(path: &str, verify: Option<bool>) -> PyResult<Result<LoadedStreets, String>> {
    // Mapped arrays reinterpret the stored little-endian bytes in place.
    if cfg!(target_endian = "big") {
        return Ok(Err("mapped street arrays need a little-endian host".into()));
    }
    if std::env::var_os("CAFEIN_DISABLE_MMAP")
        .is_some_and(|value| !value.is_empty() && value != "0")
    {
        return Ok(Err("disabled by CAFEIN_DISABLE_MMAP".into()));
    }
    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(error) => return Ok(Err(error.to_string())),
    };
    // SAFETY: artifacts are immutable by contract while mapped (see
    // `MappedArtifact`); the map is read-only on our side.
    let map = match unsafe { memmap2::Mmap::map(&file) } {
        Ok(map) => map,
        Err(error) => return Ok(Err(error.to_string())),
    };
    let backing = std::sync::Arc::new(MappedArtifact(map));
    let bytes = backing.bytes();
    let (streets_meta, layout) = parse_street_meta(path, bytes, verify == Some(true))?;
    let elevation = streets_meta.elevation.clone();
    let (_, block_end) = validate_street_shape(path, &streets_meta, 0, layout.streets_length, 0)?;
    if block_end != layout.streets_length {
        return Err(corrupted(path, "street array bounds"));
    }
    let mut streets_read = if verify == Some(true) {
        layout.streets_length
    } else {
        0
    };
    let ranges: Vec<(u64, u64)> = streets_meta
        .descriptors
        .iter()
        .map(|descriptor| (layout.streets_offset + descriptor.offset, descriptor.count))
        .collect();
    // The core CSR/geometry arrays stay mapped; the optional multimodal arrays
    // are decoded owned, paging in only their bytes.
    let section = &bytes
        [layout.streets_offset as usize..(layout.streets_offset + layout.streets_length) as usize];
    let (attributes, car, elevations) =
        decode_optional_street_arrays(section, &streets_meta.descriptors);
    if verify != Some(true) {
        streets_read += streets_meta.descriptors[STREET_ARRAY_ORDER.len()..]
            .iter()
            .map(|descriptor| descriptor.count * descriptor.kind.size())
            .sum::<u64>();
    }
    let spec = MappedStreets {
        backing,
        vertex_count: streets_meta.vertex_count,
        links: streets_meta.links,
        adjacency_offsets: ranges[0],
        adj_targets: ranges[1],
        adj_meters: ranges[2],
        adj_edges: ranges[3],
        endpoints: ranges[4],
        lengths: ranges[5],
        coordinate_offsets: ranges[6],
        lons: ranges[7],
        lats: ranges[8],
        cumulative: ranges[9],
        index_boxes: ranges[10],
        index_payload: ranges[11],
        attributes,
        car,
        elevations,
    };
    let inner =
        CoreStreetNetwork::from_mapped(spec).map_err(|_| corrupted(path, "street array bounds"))?;
    // Metadata is absent exactly when the elevations are, and its fields
    // must hold their declared invariants.
    if inner.elevations().is_some() != elevation.is_some() {
        return Err(corrupted(path, "elevation metadata"));
    }
    if let Some(meta) = &elevation {
        if check_elevation_meta(meta, inner.edge_count()).is_err() {
            return Err(corrupted(path, "elevation metadata"));
        }
    }
    Ok(Ok((inner, streets_read, elevation)))
}

use super::access::{parse_decay, validated_aggregation};
use cafein_core::access::opportunity_sums;

#[pymethods]
impl StreetNetwork {
    /// Decay-weighted opportunity sums between coordinate sets under a
    /// street mode, on the time axis (seconds).
    ///
    /// Costs are exactly `travel_time_matrix`'s directed searches; the
    /// dict carries the row-major `[origin][budget * fields]` values
    /// plus the unsnapped origin/destination indices, mirroring the
    /// matrix surface.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (origins, destinations, opportunities, fields, budgets,
                        decay, decay_param, mode, max_seconds, max_snap_distance,
                        car_model = None))]
    fn _accessibility_to_points(
        &mut self,
        py: Python<'_>,
        origins: Vec<(f64, f64)>,
        destinations: Vec<(f64, f64)>,
        opportunities: Vec<f64>,
        fields: usize,
        budgets: Vec<f64>,
        decay: &str,
        decay_param: Option<f64>,
        mode: &str,
        max_seconds: f64,
        max_snap_distance: f64,
        car_model: Option<CarModelPayload>,
    ) -> PyResult<Py<PyDict>> {
        let decay = parse_decay(decay, decay_param)?;
        validated_aggregation(destinations.len(), &opportunities, fields, &budgets)?;
        let index = self.compiled(mode, car_model.as_ref())?;
        let (_, profile) = &self.profiles[index];
        let width = budgets.len() * fields;
        let (flat, unsnapped_from, unsnapped_to) = py.allow_threads(|| {
            let (rows, unsnapped_from, unsnapped_to) = self.inner.directed_matrix(
                &origins,
                &destinations,
                profile,
                max_seconds,
                max_snap_distance,
            );
            let flat: Vec<f64> = rows
                .par_iter()
                .flat_map_iter(|row| {
                    let costs: Vec<Option<f64>> =
                        row.iter().map(|cell| cell.map(f64::from)).collect();
                    opportunity_sums(&costs, &opportunities, fields, &budgets, &decay)
                })
                .collect();
            (flat, unsnapped_from, unsnapped_to)
        });
        let count = flat.len() / width;
        let table = PyDict::new(py);
        table.set_item("values", flat.into_pyarray(py).reshape([count, width])?)?;
        table.set_item("unsnapped_from", unsnapped_from.into_pyarray(py))?;
        table.set_item("unsnapped_to", unsnapped_to.into_pyarray(py))?;
        Ok(table.into())
    }
}
