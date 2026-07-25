//! The standalone street network exposed to Python.
//!
//! A `TransportNetwork` carries a street network for walking access and
//! egress; this is the street graph on its own — built from the union OSM
//! extraction, routable by any compiled profile. Journeys never enter it, so
//! it holds no timetable, transfers, or stop links.

use numpy::{IntoPyArray, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use cafein_core::streets::{
    Backing, CompiledStreetProfile, EdgeAttributes, MappedStreets, Snap,
    StreetNetwork as CoreStreetNetwork, StreetProfileDefinition,
};

use crate::artifact::{
    corrupted, crc32, decode_optional_street_arrays, decode_streets, encode_streets, io_error,
    parse_container, validate_street_shape, write_container, MappedArtifact, MmapMode, StreetsMeta,
    STREET_ARRAY_ORDER, STREET_ARTIFACT_FORMAT, STREET_ARTIFACT_MAGIC,
};

/// The street-mode names accepted by the public API, with the shipped profile
/// definition each resolves to.
fn profile_definition(mode: &str) -> PyResult<StreetProfileDefinition> {
    match mode {
        "walk" => Ok(StreetProfileDefinition::walk()),
        "bicycle" => Ok(StreetProfileDefinition::bicycle()),
        "e_bike" => Ok(StreetProfileDefinition::e_bike()),
        "e_scooter" => Ok(StreetProfileDefinition::e_scooter()),
        other => Err(PyValueError::new_err(format!(
            "unknown street mode '{other}'; expected one of \
             'walk', 'bicycle', 'e_bike', 'e_scooter'"
        ))),
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
}

#[pymethods]
impl StreetNetwork {
    /// Builds the network from the union extraction's flat arrays.
    ///
    /// Every attribute array carries one entry per physical edge, in the same
    /// order as `edges`.
    #[new]
    #[allow(clippy::too_many_arguments)]
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
    ) -> PyResult<StreetNetwork> {
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
            },
        )
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(StreetNetwork {
            inner,
            profiles: Vec::new(),
            streets_bytes_read: 0,
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
        if mode != MmapMode::Off {
            match py.allow_threads(|| load_street_mapped(path, verify))? {
                Ok((inner, bytes_read)) => return Ok(StreetNetwork::adopt(inner, bytes_read)),
                Err(reason) if mode == MmapMode::Require => {
                    return Err(PyValueError::new_err(format!(
                        "'{path}' cannot be memory-mapped ({reason}) and \
                         mmap='require' forbids the owned fallback"
                    )))
                }
                Err(_) => {}
            }
        }
        let (inner, bytes_read) = py.allow_threads(|| load_street_owned(path, verify))?;
        Ok(StreetNetwork::adopt(inner, bytes_read))
    }

    /// The origins × destinations travel-time matrix under `mode`, in whole
    /// seconds with `u32::MAX` for unreachable.
    ///
    /// Returns `matrix` alongside the index lists of the coordinates that did
    /// not snap, matching the dict the transit point matrices return.
    /// Coordinates are `(latitude, longitude)` in EPSG:4326.
    fn travel_time_matrix(
        &mut self,
        py: Python<'_>,
        origins: Vec<(f64, f64)>,
        destinations: Vec<(f64, f64)>,
        mode: &str,
        max_seconds: f64,
        max_snap_distance: f64,
    ) -> PyResult<Py<PyDict>> {
        let index = self.compiled(mode)?;
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
    fn travel_time(
        &mut self,
        origin: (f64, f64),
        destination: (f64, f64),
        mode: &str,
        max_seconds: f64,
        max_snap_distance: f64,
    ) -> PyResult<Option<u32>> {
        // Compile first: snapping is profile-aware, so that a bicycle query
        // does not land on a footway it may not enter.
        let index = self.compiled(mode)?;
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
    fn adopt(inner: CoreStreetNetwork, streets_bytes_read: u64) -> StreetNetwork {
        StreetNetwork {
            inner,
            profiles: Vec::new(),
            streets_bytes_read,
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
    /// extended while the network stays borrowed.
    fn compiled(&mut self, mode: &str) -> PyResult<usize> {
        let definition = profile_definition(mode)?;
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
fn load_street_owned(path: &str, verify: Option<bool>) -> PyResult<(CoreStreetNetwork, u64)> {
    let bytes = std::fs::read(path).map_err(io_error)?;
    let (streets_meta, layout) = parse_street_meta(path, &bytes, verify.unwrap_or(true))?;
    let section = &bytes
        [layout.streets_offset as usize..(layout.streets_offset + layout.streets_length) as usize];
    // A street artifact carries no stops, so no link may reference one.
    let expected_levels = validate_street_shape(path, &streets_meta, layout.streets_length, 0)?;
    let parts = decode_streets(path, streets_meta, section, expected_levels)?;
    let inner =
        CoreStreetNetwork::from_parts(parts).map_err(|_| corrupted(path, "street attributes"))?;
    Ok((inner, layout.streets_length))
}

/// Loads a street artifact with the core arrays as views into a memory map.
/// `Ok(Err(reason))` means mapping is environmentally unavailable and the
/// caller decides between fallback and error; artifact problems are hard
/// errors on every path.
fn load_street_mapped(
    path: &str,
    verify: Option<bool>,
) -> PyResult<Result<(CoreStreetNetwork, u64), String>> {
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
    validate_street_shape(path, &streets_meta, layout.streets_length, 0)?;
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
    let (attributes, elevations) =
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
        elevations,
    };
    let inner =
        CoreStreetNetwork::from_mapped(spec).map_err(|_| corrupted(path, "street array bounds"))?;
    Ok(Ok((inner, streets_read)))
}
