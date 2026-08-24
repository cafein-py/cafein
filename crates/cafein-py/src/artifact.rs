//! The versioned artifact container: save, load, and the
//! memory-mapped adoption path.

use super::*;

#[pymethods]
impl TransportNetwork {
    /// Save the network as a reusable artifact.
    ///
    /// The artifact carries everything queries need — the timetable,
    /// service calendar, transfers, trip distances, leg geometries,
    /// the street network, and any computed accelerators (ULTRA/McULTRA
    /// shortcut sets, walking hierarchy, cached TBTR transfers) — behind
    /// a versioned header, so batch jobs can ``load`` the same file
    /// read-only instead of rebuilding from GTFS and OSM inputs. The payload carries a checksum, so
    /// on-disk corruption is caught at load time. Build diagnostics
    /// (quarantine reports) are not persisted; their warnings belong
    /// to the build. The file is staged beside the destination and
    /// atomically renamed into place, so saving over an artifact never
    /// rewrites it under live mapped readers.
    fn save(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        let parts = self.streets.as_ref().map(StreetNetwork::to_parts);
        let multimodal_parts = self.multimodal.as_ref().map(StreetNetwork::to_parts);
        py.allow_threads(|| {
            let timer = crate::logging::PhaseTimer::start(
                "cafein.artifact",
                "artifact.save.encode",
                "encoding the artifact payload",
                "encoded the artifact payload",
            );
            let (meta, streets_bytes) = self.encode_artifact(&parts, &multimodal_parts)?;
            timer.finish();
            write_container(path, ARTIFACT_MAGIC, ARTIFACT_FORMAT, &meta, &streets_bytes)
        })
    }

    /// The network's identity for the streaming fingerprint, as a
    /// SHA-256 hex string. A network loaded from an artifact hashes the
    /// **file** (streamed, bound to the load-time header CRCs) — stable
    /// across processes, the cross-process resume identity. An
    /// in-memory build, or one mutated since loading, digests the
    /// content it would save instead — strong but process-local, since
    /// the serialization is not canonical across processes.
    fn _artifact_checksum(&self, py: Python<'_>) -> PyResult<String> {
        if let Some((path, meta_crc, streets_crc)) = &self.source {
            if let Some(digest) = py.allow_threads(|| file_digest(path, *meta_crc, *streets_crc)) {
                return Ok(digest);
            }
        }
        let parts = self.streets.as_ref().map(StreetNetwork::to_parts);
        let multimodal_parts = self.multimodal.as_ref().map(StreetNetwork::to_parts);
        py.allow_threads(|| self.content_digest(&parts, &multimodal_parts))
    }

    /// Load a network saved with ``save``.
    ///
    /// Artifacts written in another format version are refused with a
    /// message naming the writing cafein version, and corrupted
    /// payloads fail their checksum; rebuild from the inputs (or
    /// re-save) with a matching version instead. Artifacts are trusted
    /// input, like pickles: load only files you created.
    ///
    /// ``mmap='auto'`` maps the file and uses the street arrays in
    /// place, falling back to the owned load where mapping is
    /// unavailable; ``'require'`` errors instead of falling back.
    /// ``verify`` toggles the STREETS checksum: default on for owned
    /// loads (the bytes are read anyway), off for mapped loads (the
    /// check would page the whole section in).
    #[staticmethod]
    #[pyo3(signature = (path, mmap = "off", verify = None))]
    fn load(
        py: Python<'_>,
        path: &str,
        mmap: &str,
        verify: Option<bool>,
    ) -> PyResult<TransportNetwork> {
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
        let timer = decode_timer();
        if mode != MmapMode::Off {
            match py.allow_threads(|| load_mapped(path, verify))? {
                Ok(loaded) => {
                    timer.finish();
                    return Ok(rebuilt(loaded));
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
        let loaded = py.allow_threads(|| load_owned(path, verify))?;
        timer.finish();
        Ok(rebuilt(loaded))
    }

    /// Whether the street arrays are memory-mapped views of the loaded
    /// artifact.
    #[getter]
    fn mapped(&self) -> bool {
        self.streets.as_ref().is_some_and(StreetNetwork::is_mapped)
    }

    /// STREETS-section bytes the load explicitly read — 0 for a lazy
    /// mapped load of a walk-only artifact; one carrying the optional
    /// multimodal arrays reports those, which are decoded owned.
    /// Internal; the laziness tests assert on it.
    #[getter]
    fn _streets_bytes_read(&self) -> u64 {
        self.streets_bytes_read
    }
}

impl TransportNetwork {
    /// The bincode META payload and STREETS bytes `save` writes — shared
    /// with `_artifact_checksum` so the checksum is exactly the saved
    /// content's.
    fn encode_artifact(
        &self,
        parts: &Option<cafein_core::streets::StreetNetworkParts>,
        multimodal_parts: &Option<cafein_core::streets::StreetNetworkParts>,
    ) -> PyResult<(Vec<u8>, Vec<u8>)> {
        let (streets_meta, multimodal_meta, streets_bytes) =
            self.encode_street_sections(parts, multimodal_parts);
        let artifact = self.artifact_ref(streets_meta, multimodal_meta);
        let meta = bincode::serialize(&artifact)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok((meta, streets_bytes))
    }

    /// The strong artifact identity: a SHA-256 over the artifact kind and
    /// format, the streamed META serialization, and the STREETS section
    /// with its length — domain-separated, computed without buffering the
    /// META payload (the street arrays still encode into one section
    /// buffer, the run's one network-sized transient).
    fn content_digest(
        &self,
        parts: &Option<cafein_core::streets::StreetNetworkParts>,
        multimodal_parts: &Option<cafein_core::streets::StreetNetworkParts>,
    ) -> PyResult<String> {
        use sha2::{Digest, Sha256};

        let (streets_meta, multimodal_meta, streets_bytes) =
            self.encode_street_sections(parts, multimodal_parts);
        let artifact = self.artifact_ref(streets_meta, multimodal_meta);
        let mut hasher = Sha256::new();
        hasher.update(ARTIFACT_MAGIC);
        hasher.update(ARTIFACT_FORMAT.to_le_bytes());
        bincode::serialize_into(HashWriter(&mut hasher), &artifact)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        hasher.update(b"\0STREETS");
        hasher.update((streets_bytes.len() as u64).to_le_bytes());
        hasher.update(&streets_bytes);
        Ok(hex_digest(hasher))
    }

    /// The STREETS section (walking + multimodal graphs) as `save`
    /// lays it out, with both graphs' metadata.
    fn encode_street_sections(
        &self,
        parts: &Option<cafein_core::streets::StreetNetworkParts>,
        multimodal_parts: &Option<cafein_core::streets::StreetNetworkParts>,
    ) -> (Option<StreetsMeta>, Option<StreetsMeta>, Vec<u8>) {
        {
            let (streets_meta, mut streets_bytes) = match &parts {
                Some(parts) => {
                    let (descriptors, bytes) = encode_streets(parts);
                    (
                        Some(StreetsMeta {
                            vertex_count: parts.vertex_count,
                            links: parts.links.clone(),
                            descriptors,
                            // The walking graph carries no elevations; they
                            // belong to the multimodal graph below.
                            elevation: None,
                        }),
                        bytes,
                    )
                }
                None => (None, Vec::new()),
            };
            // The multimodal union graph's arrays follow the walking arrays
            // inside the one STREETS section, starting on the array
            // alignment; its descriptor offsets are shifted to their final
            // section-relative positions here, so the load path reads both
            // graphs through the same descriptor machinery.
            let multimodal_meta = match &multimodal_parts {
                Some(parts) => {
                    let base =
                        (streets_bytes.len() as u64).div_ceil(ARRAY_ALIGNMENT) * ARRAY_ALIGNMENT;
                    streets_bytes.resize(base as usize, 0);
                    let (mut descriptors, bytes) = encode_streets(parts);
                    for descriptor in &mut descriptors {
                        descriptor.offset += base;
                    }
                    streets_bytes.extend_from_slice(&bytes);
                    Some(StreetsMeta {
                        vertex_count: parts.vertex_count,
                        links: parts.links.clone(),
                        descriptors,
                        elevation: self.multimodal_elevation.clone(),
                    })
                }
                None => None,
            };
            (streets_meta, multimodal_meta, streets_bytes)
        }
    }

    /// The bincode-serializable view of everything `save` persists.
    fn artifact_ref<'a>(
        &'a self,
        streets_meta: Option<StreetsMeta>,
        multimodal_meta: Option<StreetsMeta>,
    ) -> ArtifactRef<'a> {
        {
            ArtifactRef {
                feed: &self.feed,
                timetable: &self.build.timetable,
                services: &self.build.services,
                transfers: &self.transfers,
                geometry: &self.geometry,
                leg_geometry: &self.leg_geometry,
                streets: streets_meta,
                ultra_transfers: &self.ultra_transfers,
                ultra_window: self.ultra_window,
                mcultra_transfers: &self.mcultra_transfers,
                mcultra_window: self.mcultra_window,
                mcultra_factors: &self.mcultra_factors,
                walking_hierarchy: self.streets.as_ref().and_then(StreetNetwork::hierarchy),
                tbtr_time_transfers: &self.tbtr_time_transfers,
                mctbtr_transfers: &self.mctbtr_transfers,
                multimodal: multimodal_meta,
                multimodal_modes: &self.multimodal_modes,
                mode_transfers: self.mode_transfers.as_ref().map(|held| {
                    let mut tokens: Vec<_> = held
                        .tokens
                        .iter()
                        .map(|(&pair, &token)| (pair, token))
                        .collect();
                    tokens.sort_unstable_by_key(|&(pair, _)| pair);
                    PersistedModeTransfersRef {
                        mode: &held.mode,
                        budget: held.budget,
                        set: &held.set,
                        tokens,
                        rental_network_meters: &held.rental_network_meters,
                        rental_edge: &held.rental_edge,
                    }
                }),
                carriage_transfers: self.carriage_transfers.as_ref().map(|held| {
                    PersistedCarriageRef {
                        mode: &held.mode,
                        budget: held.budget,
                        set: &held.set,
                        ride_edge: &held.ride_edge,
                        ride_network_meters: &held.ride_network_meters,
                    }
                }),
            }
        }
    }
}

/// Writes a container: `header | directory | META … pad … | STREETS`.
///
/// The STREETS section starts on `STREETS_ALIGNMENT`, so a mapped load never
/// shares an OS page between the sections; with no street bytes there is
/// nothing to align (or to pad — padding sits outside every section CRC).
/// `magic` and `format` identify the artifact kind, so a network artifact and a
/// street artifact can never be mistaken for one another.
pub(super) fn write_container(
    path: &str,
    magic: &[u8; 8],
    format: u32,
    meta: &[u8],
    streets_bytes: &[u8],
) -> PyResult<()> {
    use std::io::Write;

    let version = env!("CARGO_PKG_VERSION").as_bytes();
    let header = 8 + 4 + 2 + version.len() as u64;
    let directory = 4 + 2 * (2 + 8 + 8 + 4) as u64;
    let meta_offset = header + directory;
    let meta_end = meta_offset + meta.len() as u64;
    let streets_offset = if streets_bytes.is_empty() {
        meta_end
    } else {
        meta_end.div_ceil(STREETS_ALIGNMENT) * STREETS_ALIGNMENT
    };

    // Stage into a sibling temp file and atomically rename over the
    // destination: an artifact must never be rewritten in place under live
    // mapped readers, whose mappings keep the replaced inode valid. The name is
    // unique per process and save, and creation is exclusive, so concurrent
    // saves never share a staging path and a stale file or symlink at it fails
    // the save instead of being written through.
    static SAVE_SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let sequence = SAVE_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let temporary = format!("{path}.tmp-{}-{sequence}", std::process::id());
    let write = || -> PyResult<()> {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(io_error)?;
        let mut writer = std::io::BufWriter::new(file);
        writer.write_all(magic).map_err(io_error)?;
        writer.write_all(&format.to_le_bytes()).map_err(io_error)?;
        writer
            .write_all(&(version.len() as u16).to_le_bytes())
            .map_err(io_error)?;
        writer.write_all(version).map_err(io_error)?;
        writer.write_all(&2u32.to_le_bytes()).map_err(io_error)?;
        for (tag, offset, bytes) in [
            (SECTION_META, meta_offset, meta),
            (SECTION_STREETS, streets_offset, streets_bytes),
        ] {
            writer.write_all(&tag.to_le_bytes()).map_err(io_error)?;
            writer.write_all(&offset.to_le_bytes()).map_err(io_error)?;
            writer
                .write_all(&(bytes.len() as u64).to_le_bytes())
                .map_err(io_error)?;
            writer
                .write_all(&crc32(bytes).to_le_bytes())
                .map_err(io_error)?;
        }
        writer.write_all(meta).map_err(io_error)?;
        let padding = streets_offset - meta_offset - meta.len() as u64;
        writer
            .write_all(&vec![0u8; padding as usize])
            .map_err(io_error)?;
        writer.write_all(streets_bytes).map_err(io_error)?;
        writer.flush().map_err(io_error)?;
        writer.get_ref().sync_all().map_err(io_error)?;
        // Replacing keeps the destination's permissions, as the old
        // truncate-in-place write did.
        if let Ok(metadata) = std::fs::metadata(path) {
            writer
                .get_ref()
                .set_permissions(metadata.permissions())
                .map_err(io_error)?;
        }
        std::fs::rename(&temporary, path).map_err(io_error)
    };
    write().inspect_err(|_| {
        let _ = std::fs::remove_file(&temporary);
    })
}

pub(super) const ARTIFACT_MAGIC: &[u8; 8] = b"CAFEINET";

/// The street-only artifact's magic and format. A distinct magic keeps the two
/// kinds apart: loading one as the other reports the wrong kind rather than
/// failing later in a checksum or a decode.
pub(super) const STREET_ARTIFACT_MAGIC: &[u8; 8] = b"CAFEINST";

// 4: `adj_access` carries the wheelchair permission bit, as in network
// format 19.
// 3: the optional car array group (per-slot driving speeds and junction
// head classes), as in network format 17.
// 2 added optional elevation metadata in `StreetsMeta`, as in network
// format 13.
pub(super) const STREET_ARTIFACT_FORMAT: u32 = 4;

// 20: a persisted wheelchair mode-transfer set is the pure
// walking-class build (no walking-closure union); earlier formats'
// wheelchair sets carried the merge and must be rebuilt.
// 19: `adj_access` carries the wheelchair permission bit (MODE_WHEELCHAIR);
// earlier builds compiled no wheelchair permissions, so earlier formats
// must be rebuilt.
// 18: the feed's stops and trips carry the GTFS wheelchair tri-states
// (`wheelchair_boarding`, `wheelchair_accessible`).
// 17: the STREETS section gains the optional car array group — per-slot
// driving speeds (f32 km/h) and junction head classes (u8) — after the
// elevations, present on car builds only.
// 16 gave the META the optional carriage transfer set with its
// per-edge ride arrays and exact binding, and the feed's trips carry
// the GTFS `bikes_allowed` tri-state.
// 15 added the optional merged mode-transfer set with its tokens,
// per-edge rental arrays, and exact binding.
// 14 added an optional second `StreetsMeta` for the multimodal union
// street graph, whose arrays follow the walking arrays inside the
// STREETS section (descriptor offsets pre-shifted to their position).
// 13 added optional elevation metadata to `StreetsMeta`. Earlier formats
// must be rebuilt.
pub(super) const ARTIFACT_FORMAT: u32 = 20;

/// Section tags in the container directory.
pub(super) const SECTION_META: u16 = 1;

pub(super) const SECTION_STREETS: u16 = 2;

/// The STREETS section starts on this boundary (covers every target
/// platform's page and allocation granularity), so a mapped load never
/// shares an OS page between META and STREETS.
pub(super) const STREETS_ALIGNMENT: u64 = 65_536;

/// Every street array starts 8-byte aligned within the STREETS section.
pub(super) const ARRAY_ALIGNMENT: u64 = 8;

/// The decoded part of the saved network (the META section), borrowed
/// for writing. The street layer's large arrays live in the STREETS
/// section as raw little-endian values; META carries only their
/// descriptor table plus the small link records and scalars.
#[derive(serde::Serialize)]
pub(super) struct ArtifactRef<'a> {
    feed: &'a Feed,
    timetable: &'a Timetable,
    services: &'a ServiceCalendar,
    transfers: &'a Transfers,
    geometry: &'a Option<TripGeometry>,
    leg_geometry: &'a Option<LegGeometry>,
    streets: Option<StreetsMeta>,
    /// The ULTRA shortcut set and the source-departure window it was
    /// computed for, when present; restored so the heavy run-once
    /// preprocessing need not be repeated, and so a partial-window set
    /// is never mistaken for a whole-day one.
    ultra_transfers: &'a Option<Transfers>,
    ultra_window: Option<(u32, u32)>,
    /// The McULTRA (emissions-aware) shortcut set with its window and the
    /// factor vector it was built for; restored so the heavy run-once
    /// preprocessing need not be repeated and the factor contract holds.
    mcultra_transfers: &'a Option<Transfers>,
    mcultra_window: Option<(u32, u32)>,
    mcultra_factors: &'a Option<Vec<f64>>,
    /// The walking contraction hierarchy, when installed; restored so the
    /// run-once contraction need not be repeated. Its one-to-many buckets are
    /// derived state, rebuilt on load rather than persisted.
    walking_hierarchy: Option<&'a ContractionHierarchy>,
    /// The cached time-only TBTR transfer set with the date it was built for,
    /// when present; restored so a loaded network reuses it instead of
    /// rebuilding the dominance-aware set.
    tbtr_time_transfers: &'a Option<(String, cafein_core::tbtr::TransferSet)>,
    /// The cached multicriteria TBTR transfer set with its date and factor
    /// vector, when present.
    mctbtr_transfers: &'a Option<(String, Vec<f64>, cafein_core::tbtr::TransferSet)>,
    /// The multimodal union street graph's meta, when installed. Its
    /// descriptor offsets are already section-relative: the arrays sit after
    /// the walking arrays inside the one STREETS section.
    multimodal: Option<StreetsMeta>,
    /// The pruning modes the multimodal graph was built with.
    multimodal_modes: &'a Option<Vec<String>>,
    /// The merged shared-vehicle transfer set with its exact binding,
    /// when computed; restored so the heavy merge need not be repeated.
    mode_transfers: Option<PersistedModeTransfersRef<'a>>,
    /// The carriage transfer set with its exact binding, when computed.
    carriage_transfers: Option<PersistedCarriageRef<'a>>,
}

/// The carriage set as persisted: rows, per-edge ride identity and
/// meters, and the binding; the unclosed marking is re-applied on load.
#[derive(serde::Serialize)]
struct PersistedCarriageRef<'a> {
    mode: &'a str,
    budget: f64,
    set: &'a Transfers,
    ride_edge: &'a [bool],
    ride_network_meters: &'a [f64],
}

#[derive(serde::Deserialize)]
pub(super) struct PersistedCarriage {
    mode: String,
    budget: f64,
    set: Transfers,
    ride_edge: Vec<bool>,
    ride_network_meters: Vec<f64>,
}

/// The merged mode-transfer set as persisted: the token map flattened
/// into key-sorted pairs so a re-save stays byte-identical, the
/// unclosed marking re-applied on load (a persisted merged set is
/// never a closure). The borrowed twin serializes to the same bytes
/// the owned form deserializes from.
#[derive(serde::Serialize)]
struct PersistedModeTransfersRef<'a> {
    mode: &'a str,
    budget: f64,
    set: &'a Transfers,
    tokens: Vec<((u32, u32), cafein_core::mode_transfers::RentalToken)>,
    rental_network_meters: &'a [f64],
    rental_edge: &'a [bool],
}

#[derive(serde::Deserialize)]
pub(super) struct PersistedModeTransfers {
    mode: String,
    budget: f64,
    set: Transfers,
    tokens: Vec<((u32, u32), cafein_core::mode_transfers::RentalToken)>,
    rental_network_meters: Vec<f64>,
    rental_edge: Vec<bool>,
}

/// The decoded part of the saved network, owned after reading.
#[derive(serde::Deserialize)]
pub(super) struct Artifact {
    feed: Feed,
    timetable: Timetable,
    services: ServiceCalendar,
    transfers: Transfers,
    geometry: Option<TripGeometry>,
    leg_geometry: Option<LegGeometry>,
    streets: Option<StreetsMeta>,
    ultra_transfers: Option<Transfers>,
    ultra_window: Option<(u32, u32)>,
    mcultra_transfers: Option<Transfers>,
    mcultra_window: Option<(u32, u32)>,
    mcultra_factors: Option<Vec<f64>>,
    walking_hierarchy: Option<ContractionHierarchy>,
    tbtr_time_transfers: Option<(String, cafein_core::tbtr::TransferSet)>,
    mctbtr_transfers: Option<(String, Vec<f64>, cafein_core::tbtr::TransferSet)>,
    multimodal: Option<StreetsMeta>,
    multimodal_modes: Option<Vec<String>>,
    mode_transfers: Option<PersistedModeTransfers>,
    carriage_transfers: Option<PersistedCarriage>,
}

/// The street layer's decoded state: link records (endpoints
/// denormalised, so the vertex→link index rebuilds from these alone),
/// scalars, and the descriptor table locating every raw array inside the
/// STREETS section.
#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub(super) struct StreetsMeta {
    pub(super) vertex_count: u32,
    pub(super) links: Vec<StoredLink>,
    pub(super) descriptors: Vec<ArrayDescriptor>,
    /// What the persisted per-coordinate elevations mean; `None` when the
    /// network carries no elevations.
    pub(super) elevation: Option<ElevationMeta>,
}

/// Provenance of the sampled elevations: enough to know what the numbers
/// mean without re-reading the DEM.
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug, PartialEq)]
pub(super) struct ElevationMeta {
    /// The DEM the coordinates were sampled from, as given to the build.
    pub(super) source: String,
    /// The along-edge sampling interval in meters.
    pub(super) sampling_interval: f64,
    /// How missing raster values were treated.
    pub(super) nodata_policy: String,
    /// The finite share of sampled coordinates, 0..=1.
    pub(super) coverage: f64,
    /// Bridge/tunnel edges whose interior was endpoint-interpolated.
    pub(super) inferred_edges: u32,
}

/// One raw array inside the STREETS section. Offsets are relative to the
/// section start (absolute positions come from the section directory), so
/// the descriptor table is complete before the file layout is.
#[derive(serde::Serialize, serde::Deserialize, Clone, Copy, PartialEq, Debug)]
pub(super) struct ArrayDescriptor {
    array: StreetArray,
    pub(super) kind: ArrayKind,
    pub(super) count: u64,
    pub(super) offset: u64,
}

/// The street arrays, in their fixed on-disk order. The first thirteen are
/// the mandatory core graph; the trailing nine are the optional multimodal
/// arrays (a walk-only artifact omits them).
#[derive(serde::Serialize, serde::Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum StreetArray {
    AdjacencyOffsets,
    AdjTargets,
    AdjMeters,
    AdjEdges,
    Endpoints,
    Lengths,
    CoordinateOffsets,
    Lons,
    Lats,
    Cumulative,
    IndexBoxes,
    IndexPayload,
    IndexLevelStarts,
    AdjAccess,
    AdjFacility,
    EdgeHighway,
    EdgeSurface,
    EdgeSmoothness,
    EdgeFlags,
    CoordinateElevations,
    AdjCarSpeed,
    AdjJunction,
}

/// Element type of a raw street array.
#[derive(serde::Serialize, serde::Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum ArrayKind {
    U8,
    U16,
    U32,
    I32,
    F32,
    F64,
}

impl ArrayKind {
    pub(super) fn size(self) -> u64 {
        match self {
            ArrayKind::U8 => 1,
            ArrayKind::U16 => 2,
            ArrayKind::U32 | ArrayKind::I32 | ArrayKind::F32 => 4,
            ArrayKind::F64 => 8,
        }
    }
}

/// The expected street arrays: identity, element kind, and the length
/// each must have, derived from the decoded META scalars. `None` lengths
/// are tied to other arrays and validated by the cross-checks instead.
pub(super) const STREET_ARRAY_ORDER: [(StreetArray, ArrayKind); 13] = [
    (StreetArray::AdjacencyOffsets, ArrayKind::U32),
    (StreetArray::AdjTargets, ArrayKind::U32),
    (StreetArray::AdjMeters, ArrayKind::F64),
    (StreetArray::AdjEdges, ArrayKind::U32),
    (StreetArray::Endpoints, ArrayKind::U32),
    (StreetArray::Lengths, ArrayKind::F64),
    (StreetArray::CoordinateOffsets, ArrayKind::U32),
    (StreetArray::Lons, ArrayKind::I32),
    (StreetArray::Lats, ArrayKind::I32),
    (StreetArray::Cumulative, ArrayKind::F32),
    (StreetArray::IndexBoxes, ArrayKind::I32),
    (StreetArray::IndexPayload, ArrayKind::U32),
    (StreetArray::IndexLevelStarts, ArrayKind::U32),
];

/// The optional multimodal street arrays, in their canonical on-disk order.
/// A present artifact stores an order-preserving subsequence of these after
/// the core arrays: the six attribute arrays as a group (all or none), the
/// elevations independently, and the two car arrays as a group (all or
/// none, and only beside the attribute group).
pub(super) const OPTIONAL_STREET_ARRAY_ORDER: [(StreetArray, ArrayKind); 9] = [
    (StreetArray::AdjAccess, ArrayKind::U8),
    (StreetArray::AdjFacility, ArrayKind::U8),
    (StreetArray::EdgeHighway, ArrayKind::U8),
    (StreetArray::EdgeSurface, ArrayKind::U8),
    (StreetArray::EdgeSmoothness, ArrayKind::U8),
    (StreetArray::EdgeFlags, ArrayKind::U16),
    (StreetArray::CoordinateElevations, ArrayKind::F32),
    (StreetArray::AdjCarSpeed, ArrayKind::F32),
    (StreetArray::AdjJunction, ArrayKind::U8),
];

/// The six attribute arrays that form the all-or-none multimodal group.
pub(super) const STREET_ATTRIBUTE_ARRAYS: [StreetArray; 6] = [
    StreetArray::AdjAccess,
    StreetArray::AdjFacility,
    StreetArray::EdgeHighway,
    StreetArray::EdgeSurface,
    StreetArray::EdgeSmoothness,
    StreetArray::EdgeFlags,
];

/// The two car arrays that form the all-or-none car group.
pub(super) const STREET_CAR_ARRAYS: [StreetArray; 2] =
    [StreetArray::AdjCarSpeed, StreetArray::AdjJunction];

/// A read-only memory map of an artifact file, kept alive by the street
/// network whose arrays point into it. The mapped file must stay
/// unchanged for the mapping's lifetime (see [`Backing`]): replace
/// artifacts by atomic rename, never by editing in place — and keep them
/// out of cloud-synced folders, whose daemons rewrite files in place.
pub(super) struct MappedArtifact(pub(super) memmap2::Mmap);

/// How `load` should back the street arrays.
#[derive(PartialEq, Clone, Copy)]
pub(super) enum MmapMode {
    Off,
    Auto,
    Require,
}

/// The section directory of a parsed container: everything `load` needs
/// to locate the sections, checksums still unchecked.
pub(super) struct ContainerLayout {
    pub(super) meta_offset: u64,
    pub(super) meta_length: u64,
    pub(super) meta_crc: u32,
    pub(super) streets_offset: u64,
    pub(super) streets_length: u64,
    pub(super) streets_crc: u32,
}

/// The stop and trip lookup tables derived from a feed and timetable.
pub(super) type DerivedIndexes = (
    HashMap<String, StopLookup>,
    HashMap<String, StopIdx>,
    HashMap<String, TripIdx>,
);

pub(super) fn derived_indexes(feed: &Feed, timetable: &Timetable) -> DerivedIndexes {
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
    let mut trips_by_public_id = HashMap::with_capacity(timetable.trip_count() as usize);
    for index in 0..timetable.trip_count() {
        let trip = TripIdx(index);
        let source = &feed.trips[timetable.trip_source(trip) as usize];
        let public = if feed.feed_count > 1 {
            format!("{}:{}", source.feed, source.id)
        } else {
            source.id.clone()
        };
        trips_by_public_id.insert(public, trip);
    }
    (stops_by_id, stops_by_qualified_id, trips_by_public_id)
}

pub(super) fn io_error(error: std::io::Error) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// Streams `bincode` output straight into a SHA-256 — the META payload
/// hashes without ever being buffered.
pub(super) struct HashWriter<'a>(pub(super) &'a mut sha2::Sha256);

impl std::io::Write for HashWriter<'_> {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        sha2::Digest::update(self.0, buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// The loaded artifact file's streamed SHA-256, bound to the header
/// CRCs recorded at load time: a file replaced since then fails the
/// binding and the caller falls back to the content digest.
pub(super) fn file_digest(path: &str, meta_crc: u32, streets_crc: u32) -> Option<String> {
    use sha2::Digest;
    use std::io::Read;

    let total = std::fs::metadata(path).ok()?.len();
    let mut file = std::fs::File::open(path).ok()?;
    let mut head = vec![0u8; 4096.min(total as usize)];
    file.read_exact(&mut head).ok()?;
    let layout = parse_header(path, &head, total, ARTIFACT_MAGIC, ARTIFACT_FORMAT)
        .or_else(|_| {
            parse_header(
                path,
                &head,
                total,
                STREET_ARTIFACT_MAGIC,
                STREET_ARTIFACT_FORMAT,
            )
        })
        .ok()?;
    if layout.meta_crc != meta_crc || layout.streets_crc != streets_crc {
        return None;
    }
    let mut hasher = sha2::Sha256::new();
    hasher.update(b"FILE\0");
    let mut meta_state = Crc32::new();
    let mut streets_state = Crc32::new();
    let mut position = 0u64;
    let mut feed = |bytes: &[u8], position: &mut u64| {
        hasher.update(bytes);
        for (range, state) in [
            (
                layout.meta_offset..layout.meta_offset + layout.meta_length,
                &mut meta_state,
            ),
            (
                layout.streets_offset..layout.streets_offset + layout.streets_length,
                &mut streets_state,
            ),
        ] {
            let start = range
                .start
                .max(*position)
                .min(*position + bytes.len() as u64);
            let end = range.end.max(*position).min(*position + bytes.len() as u64);
            if start < end {
                state.update(&bytes[(start - *position) as usize..(end - *position) as usize]);
            }
        }
        *position += bytes.len() as u64;
    };
    feed(&head, &mut position);
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer).ok()?;
        if read == 0 {
            break;
        }
        feed(&buffer[..read], &mut position);
    }
    if meta_state.finish() != meta_crc || streets_state.finish() != streets_crc {
        return None;
    }
    Some(hex_digest(hasher))
}

/// The `(path, meta_crc, streets_crc)` source binding of an artifact
/// file, read from its header alone.
pub(super) fn read_source(path: &str, magic: &[u8; 8], format: u32) -> Option<(String, u32, u32)> {
    use std::io::Read;

    let total = std::fs::metadata(path).ok()?.len();
    let mut file = std::fs::File::open(path).ok()?;
    let mut head = vec![0u8; 4096.min(total as usize)];
    file.read_exact(&mut head).ok()?;
    let layout = parse_header(path, &head, total, magic, format).ok()?;
    Some((
        canonical_source_path(path),
        layout.meta_crc,
        layout.streets_crc,
    ))
}

/// The absolute form a source path is remembered in, so a later
/// working-directory change cannot re-point the identity.
pub(super) fn canonical_source_path(path: &str) -> String {
    std::fs::canonicalize(path)
        .map(|canonical| canonical.to_string_lossy().into_owned())
        .unwrap_or_else(|_| path.to_string())
}

/// A SHA-256 state as its lowercase hex digest.
pub(super) fn hex_digest(hasher: sha2::Sha256) -> String {
    use sha2::Digest;

    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// CRC-32 (IEEE) over the artifact payload.
pub(super) fn crc32(bytes: &[u8]) -> u32 {
    let mut state = Crc32::new();
    state.update(bytes);
    state.finish()
}

/// The container CRC as an incremental state, for streamed reads.
pub(super) struct Crc32(u32);

impl Crc32 {
    pub(super) fn new() -> Crc32 {
        Crc32(u32::MAX)
    }

    pub(super) fn update(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = CRC_TABLE[((self.0 ^ byte as u32) & 0xFF) as usize] ^ (self.0 >> 8);
        }
    }

    pub(super) fn finish(&self) -> u32 {
        !self.0
    }
}

const CRC_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut index = 0;
    while index < 256 {
        let mut value = index as u32;
        let mut bit = 0;
        while bit < 8 {
            value = if value & 1 != 0 {
                0xEDB8_8320 ^ (value >> 1)
            } else {
                value >> 1
            };
            bit += 1;
        }
        table[index] = value;
        index += 1;
    }
    table
};

/// A corrupted-artifact error with the standard rebuild advice.
pub(super) fn corrupted(path: &str, what: &str) -> PyErr {
    PyValueError::new_err(format!(
        "'{path}' is corrupted ({what}); rebuild the network from its \
         inputs and save it again"
    ))
}

/// Serializes a street network's parts into the raw STREETS bytes and the
/// descriptor table locating each array within them.
pub(super) fn encode_streets(parts: &StreetNetworkParts) -> (Vec<ArrayDescriptor>, Vec<u8>) {
    fn push<T: Copy>(
        bytes: &mut Vec<u8>,
        descriptors: &mut Vec<ArrayDescriptor>,
        array: StreetArray,
        kind: ArrayKind,
        values: &[T],
        encode: impl Fn(T) -> [u8; 8],
    ) {
        while !(bytes.len() as u64).is_multiple_of(ARRAY_ALIGNMENT) {
            bytes.push(0);
        }
        descriptors.push(ArrayDescriptor {
            array,
            kind,
            count: values.len() as u64,
            offset: bytes.len() as u64,
        });
        let size = kind.size() as usize;
        for &value in values {
            bytes.extend_from_slice(&encode(value)[..size]);
        }
    }
    fn u8_bytes(value: u8) -> [u8; 8] {
        let mut buffer = [0u8; 8];
        buffer[0] = value;
        buffer
    }
    fn u16_bytes(value: u16) -> [u8; 8] {
        let mut buffer = [0u8; 8];
        buffer[..2].copy_from_slice(&value.to_le_bytes());
        buffer
    }
    fn u32_bytes(value: u32) -> [u8; 8] {
        let mut buffer = [0u8; 8];
        buffer[..4].copy_from_slice(&value.to_le_bytes());
        buffer
    }
    fn i32_bytes(value: i32) -> [u8; 8] {
        let mut buffer = [0u8; 8];
        buffer[..4].copy_from_slice(&value.to_le_bytes());
        buffer
    }
    fn f32_bytes(value: f32) -> [u8; 8] {
        let mut buffer = [0u8; 8];
        buffer[..4].copy_from_slice(&value.to_le_bytes());
        buffer
    }
    fn f64_bytes(value: f64) -> [u8; 8] {
        value.to_le_bytes()
    }

    let mut bytes = Vec::new();
    let mut descriptors = Vec::with_capacity(STREET_ARRAY_ORDER.len());
    use ArrayKind::*;
    use StreetArray::*;
    let d = &mut descriptors;
    let b = &mut bytes;
    push(
        b,
        d,
        AdjacencyOffsets,
        U32,
        &parts.adjacency_offsets,
        u32_bytes,
    );
    push(b, d, AdjTargets, U32, &parts.adj_targets, u32_bytes);
    push(b, d, AdjMeters, F64, &parts.adj_meters, f64_bytes);
    push(b, d, AdjEdges, U32, &parts.adj_edges, u32_bytes);
    push(b, d, Endpoints, U32, &parts.endpoints, u32_bytes);
    push(b, d, Lengths, F64, &parts.lengths, f64_bytes);
    push(
        b,
        d,
        CoordinateOffsets,
        U32,
        &parts.coordinate_offsets,
        u32_bytes,
    );
    push(b, d, Lons, I32, &parts.lons, i32_bytes);
    push(b, d, Lats, I32, &parts.lats, i32_bytes);
    push(b, d, Cumulative, F32, &parts.cumulative, f32_bytes);
    push(b, d, IndexBoxes, I32, &parts.index_boxes, i32_bytes);
    push(b, d, IndexPayload, U32, &parts.index_payload, u32_bytes);
    push(
        b,
        d,
        IndexLevelStarts,
        U32,
        &parts.index_level_starts,
        u32_bytes,
    );
    // The optional multimodal arrays follow the core graph, in canonical
    // order: the attribute group (all six together) then the elevations.
    if let Some(attributes) = &parts.attributes {
        push(b, d, AdjAccess, U8, &attributes.adj_access, u8_bytes);
        push(b, d, AdjFacility, U8, &attributes.adj_facility, u8_bytes);
        push(b, d, EdgeHighway, U8, &attributes.edge_highway, u8_bytes);
        push(b, d, EdgeSurface, U8, &attributes.edge_surface, u8_bytes);
        push(
            b,
            d,
            EdgeSmoothness,
            U8,
            &attributes.edge_smoothness,
            u8_bytes,
        );
        push(b, d, EdgeFlags, U16, &attributes.edge_flags, u16_bytes);
    }
    if let Some(elevations) = &parts.elevations {
        push(b, d, CoordinateElevations, F32, elevations, f32_bytes);
    }
    if let Some(car) = &parts.car {
        push(b, d, AdjCarSpeed, F32, &car.adj_car_speed, f32_bytes);
        push(b, d, AdjJunction, U8, &car.adj_junction, u8_bytes);
    }
    (descriptors, bytes)
}

/// The level-start table a packed index with `leaves` leaves must carry —
/// mirrors the builder in cafein-core.
pub(super) fn expected_level_starts(leaves: usize) -> Vec<u32> {
    let mut levels = vec![0u32];
    let mut total = leaves;
    let mut level_size = leaves;
    while level_size > 1 {
        levels.push(total as u32);
        level_size = level_size.div_ceil(16);
        total += level_size;
    }
    match leaves {
        0 => vec![0, 0],
        1 => vec![0, 1],
        _ => {
            levels.push(total as u32);
            levels
        }
    }
}

/// Validates a street layer from the decoded META alone: descriptor
/// order, sequential aligned layout inside the section starting at
/// `block_start`, counts mutually consistent (so no query indexes out of
/// an array), and well-formed link records. Runs on every load path
/// without touching a single STREETS byte — which is what keeps a mapped
/// load lazy. Returns the level-start table the index must carry and the
/// block's end byte, which the caller checks against what follows (the
/// section end, or the next block), so trailing garbage stays rejected.
pub(super) fn validate_street_shape(
    path: &str,
    meta: &StreetsMeta,
    block_start: u64,
    section_length: u64,
    stop_count: u32,
) -> PyResult<(Vec<u32>, u64)> {
    // The core arrays come first, in fixed order; the optional multimodal
    // arrays follow as an order-preserving subsequence of the canonical
    // optional order (any absent, but never reordered or duplicated).
    if meta.descriptors.len() < STREET_ARRAY_ORDER.len() {
        return Err(corrupted(path, "street descriptor table shape"));
    }
    let (core, optional) = meta.descriptors.split_at(STREET_ARRAY_ORDER.len());
    for (descriptor, &(array, kind)) in core.iter().zip(&STREET_ARRAY_ORDER) {
        if descriptor.array != array || descriptor.kind != kind {
            return Err(corrupted(path, "street descriptor table shape"));
        }
    }
    let mut expected = OPTIONAL_STREET_ARRAY_ORDER.iter().peekable();
    for descriptor in optional {
        loop {
            match expected.next() {
                Some(&(array, kind)) if descriptor.array == array => {
                    if descriptor.kind != kind {
                        return Err(corrupted(path, "street descriptor table shape"));
                    }
                    break;
                }
                Some(_) => continue,
                None => return Err(corrupted(path, "street descriptor table shape")),
            }
        }
    }
    // The arrays must occupy the section exactly as the writer laid them
    // out: sequential, each at the next aligned position — no gaps,
    // overlaps, or aliasing. Checked arithmetic throughout, over the core
    // and any optional arrays alike.
    let mut expected_offset = block_start;
    let mut last_end = block_start;
    for descriptor in &meta.descriptors {
        let extent = descriptor
            .count
            .checked_mul(descriptor.kind.size())
            .and_then(|length| descriptor.offset.checked_add(length));
        let Some(end) = extent.filter(|&end| end <= section_length) else {
            return Err(corrupted(path, "street array bounds"));
        };
        if descriptor.offset != expected_offset {
            return Err(corrupted(path, "street array bounds"));
        }
        last_end = end;
        expected_offset = end.div_ceil(ARRAY_ALIGNMENT) * ARRAY_ALIGNMENT;
    }
    let count = |i: usize| meta.descriptors[i].count;
    let vertices = u64::from(meta.vertex_count);
    let edges = count(5);
    let coordinates = count(7);
    let leaves = count(11) / 2;
    let expected_levels = expected_level_starts(leaves as usize);
    let consistent = count(0) == vertices + 1
        && count(1) == count(2)
        && count(1) == count(3)
        && Some(count(1)) == edges.checked_mul(2)
        && count(4) == count(1)
        && count(6) == edges + 1
        && count(8) == coordinates
        && count(9) == coordinates
        && count(11) % 2 == 0
        // The leaves must be exactly one per consecutive coordinate
        // pair, or snapping would silently skip streets …
        && Some(leaves) == coordinates.checked_sub(edges)
        // … in a tree shaped exactly as its builder shapes it.
        && count(12) == expected_levels.len() as u64
        && Some(count(10)) == u64::from(*expected_levels.last().unwrap()).checked_mul(4);
    if !consistent {
        return Err(corrupted(path, "street array consistency"));
    }
    // Optional arrays: the six attributes are an all-or-none group, and every
    // present optional array must match the graph's slot/edge/coordinate
    // count. The subsequence order was checked above; here it is the group
    // completeness and the lengths.
    let slots = count(1);
    let optional = |array: StreetArray| {
        meta.descriptors[STREET_ARRAY_ORDER.len()..]
            .iter()
            .find(|d| d.array == array)
    };
    let attributes_present = STREET_ATTRIBUTE_ARRAYS
        .iter()
        .filter(|&&a| optional(a).is_some())
        .count();
    if attributes_present != 0 && attributes_present != STREET_ATTRIBUTE_ARRAYS.len() {
        return Err(corrupted(path, "street attribute group"));
    }
    // The car arrays are their own all-or-none group, valid only beside a
    // complete attribute group (a car build is always multimodal).
    let car_present = STREET_CAR_ARRAYS
        .iter()
        .filter(|&&a| optional(a).is_some())
        .count();
    if car_present != 0 && (car_present != STREET_CAR_ARRAYS.len() || attributes_present == 0) {
        return Err(corrupted(path, "street car group"));
    }
    let attribute_length =
        |array: StreetArray, expected: u64| optional(array).is_none_or(|d| d.count == expected);
    let attributes_consistent = attribute_length(StreetArray::AdjAccess, slots)
        && attribute_length(StreetArray::AdjFacility, slots)
        && attribute_length(StreetArray::EdgeHighway, edges)
        && attribute_length(StreetArray::EdgeSurface, edges)
        && attribute_length(StreetArray::EdgeSmoothness, edges)
        && attribute_length(StreetArray::EdgeFlags, edges)
        && attribute_length(StreetArray::CoordinateElevations, coordinates)
        && attribute_length(StreetArray::AdjCarSpeed, slots)
        && attribute_length(StreetArray::AdjJunction, slots);
    if !attributes_consistent {
        return Err(corrupted(path, "street array consistency"));
    }
    for link in &meta.links {
        if u64::from(link.edge) >= edges
            || link.stop.0 >= stop_count
            || !(0.0..=1.0).contains(&link.fraction)
            || !link.connector.is_finite()
            || link.connector < 0.0
            || u64::from(link.from) >= vertices
            || u64::from(link.to) >= vertices
        {
            return Err(corrupted(path, "street link records"));
        }
    }
    Ok((expected_levels, last_end))
}

/// Where the multimodal block begins inside the STREETS section — its first
/// descriptor's offset (0 for an empty table, which the shape validation
/// then rejects).
fn multimodal_block_start(meta: &StreetsMeta) -> u64 {
    meta.descriptors
        .first()
        .map_or(0, |descriptor| descriptor.offset)
}

/// The walking block must end (aligned) exactly where the next block starts,
/// or at the section end when nothing follows — trailing garbage stays
/// rejected with two blocks just as it was with one.
fn check_block_boundary(
    path: &str,
    block_end: u64,
    next_block: Option<u64>,
    section_length: u64,
) -> PyResult<()> {
    let target = next_block.unwrap_or(section_length);
    let aligned = block_end.div_ceil(ARRAY_ALIGNMENT) * ARRAY_ALIGNMENT;
    if block_end == target || (next_block.is_some() && aligned == target) {
        Ok(())
    } else {
        Err(corrupted(path, "street array bounds"))
    }
}

/// Reads the optional multimodal arrays owned, located by descriptor
/// identity — shared by the owned decode and the mapped load, which also
/// keeps them owned. `section` is the whole STREETS section; descriptor
/// offsets are relative to it. The shape validation has confirmed the
/// attribute group is complete and every length matches, so a present group
/// yields `Some` and reading only pages in the optional bytes.
pub(super) fn decode_optional_street_arrays(
    section: &[u8],
    descriptors: &[ArrayDescriptor],
) -> (
    Option<StreetAttributes>,
    Option<CarAttributes>,
    Option<Vec<f32>>,
) {
    fn slice<'a>(section: &'a [u8], descriptor: &ArrayDescriptor) -> &'a [u8] {
        let start = descriptor.offset as usize;
        let end = start + (descriptor.count * descriptor.kind.size()) as usize;
        &section[start..end]
    }
    let find = |array: StreetArray| descriptors.iter().find(|d| d.array == array);
    let read_u8 = |array| find(array).map(|d| slice(section, d).to_vec());
    let read_u16 = |array| {
        find(array).map(|d| {
            slice(section, d)
                .as_chunks::<2>()
                .0
                .iter()
                .map(|chunk| u16::from_le_bytes(*chunk))
                .collect::<Vec<u16>>()
        })
    };
    let read_f32 = |array| {
        find(array).map(|d| {
            slice(section, d)
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_le_bytes(*chunk))
                .collect::<Vec<f32>>()
        })
    };
    let attributes = match (
        read_u8(StreetArray::AdjAccess),
        read_u8(StreetArray::AdjFacility),
        read_u8(StreetArray::EdgeHighway),
        read_u8(StreetArray::EdgeSurface),
        read_u8(StreetArray::EdgeSmoothness),
        read_u16(StreetArray::EdgeFlags),
    ) {
        (
            Some(adj_access),
            Some(adj_facility),
            Some(edge_highway),
            Some(edge_surface),
            Some(edge_smoothness),
            Some(edge_flags),
        ) => Some(StreetAttributes {
            adj_access,
            adj_facility,
            edge_highway,
            edge_surface,
            edge_smoothness,
            edge_flags,
        }),
        _ => None,
    };
    let car = match (
        read_f32(StreetArray::AdjCarSpeed),
        read_u8(StreetArray::AdjJunction),
    ) {
        (Some(adj_car_speed), Some(adj_junction)) => Some(CarAttributes {
            adj_car_speed,
            adj_junction,
        }),
        _ => None,
    };
    (attributes, car, read_f32(StreetArray::CoordinateElevations))
}

/// Decodes the street arrays into owned parts and cross-checks their
/// contents — the value tier a mapped load defers to `verify` and to
/// first use. The shape tier ([`validate_street_shape`]) has run.
pub(super) fn decode_streets(
    path: &str,
    meta: StreetsMeta,
    section: &[u8],
    expected_levels: Vec<u32>,
) -> PyResult<StreetNetworkParts> {
    fn read<T>(
        section: &[u8],
        descriptor: &ArrayDescriptor,
        decode: impl Fn(&[u8]) -> T,
    ) -> Vec<T> {
        let start = descriptor.offset as usize;
        let end = start + (descriptor.count * descriptor.kind.size()) as usize;
        section[start..end]
            .chunks_exact(descriptor.kind.size() as usize)
            .map(decode)
            .collect()
    }
    let u32s = |i: usize| {
        read(section, &meta.descriptors[i], |chunk| {
            u32::from_le_bytes(chunk.try_into().unwrap())
        })
    };
    let i32s = |i: usize| {
        read(section, &meta.descriptors[i], |chunk| {
            i32::from_le_bytes(chunk.try_into().unwrap())
        })
    };
    let f32s = |i: usize| {
        read(section, &meta.descriptors[i], |chunk| {
            f32::from_le_bytes(chunk.try_into().unwrap())
        })
    };
    let f64s = |i: usize| {
        read(section, &meta.descriptors[i], |chunk| {
            f64::from_le_bytes(chunk.try_into().unwrap())
        })
    };

    let (attributes, car, elevations) = decode_optional_street_arrays(section, &meta.descriptors);
    let parts = StreetNetworkParts {
        vertex_count: meta.vertex_count,
        adjacency_offsets: u32s(0),
        adj_targets: u32s(1),
        adj_meters: f64s(2),
        adj_edges: u32s(3),
        endpoints: u32s(4),
        lengths: f64s(5),
        coordinate_offsets: u32s(6),
        lons: i32s(7),
        lats: i32s(8),
        cumulative: f32s(9),
        index_boxes: i32s(10),
        index_payload: u32s(11),
        index_level_starts: u32s(12),
        links: meta.links,
        attributes,
        car,
        elevations,
    };

    // Interior values: offsets monotonic with at least two coordinates
    // per edge and their tails matching the array lengths, ids in range,
    // costs and along-distances well-formed, and the packed index laid
    // out exactly as its builder lays it out, so a corrupted artifact
    // fails loading instead of panicking mid-query.
    let vertices = parts.vertex_count as usize;
    let edges = parts.lengths.len();
    let coordinates = parts.lons.len();
    if parts.adjacency_offsets.first() != Some(&0)
        || !parts.adjacency_offsets.windows(2).all(|w| w[0] <= w[1])
        || parts.adjacency_offsets.last().copied() != u32::try_from(parts.adj_targets.len()).ok()
    {
        return Err(corrupted(path, "street adjacency offsets"));
    }
    if parts.coordinate_offsets.first() != Some(&0)
        || !parts
            .coordinate_offsets
            .windows(2)
            .all(|w| w[1].checked_sub(w[0]).is_some_and(|span| span >= 2))
        || parts.coordinate_offsets.last().copied() != u32::try_from(coordinates).ok()
    {
        return Err(corrupted(path, "street coordinate offsets"));
    }
    if !parts.adj_targets.iter().all(|&v| (v as usize) < vertices)
        || !parts.adj_edges.iter().all(|&e| (e as usize) < edges)
        || !parts.endpoints.iter().all(|&v| (v as usize) < vertices)
    {
        return Err(corrupted(path, "street graph references"));
    }
    // Every adjacency row must restate its edge's endpoints and cost, so
    // the adopted CSR is a faithful view of the edge list, not merely
    // in-range.
    let mut edge_directions = vec![(0u8, 0u8); edges];
    for vertex in 0..vertices {
        let start = parts.adjacency_offsets[vertex] as usize;
        let end = parts.adjacency_offsets[vertex + 1] as usize;
        for slot in start..end {
            let edge = parts.adj_edges[slot] as usize;
            let (from, to) = (parts.endpoints[2 * edge], parts.endpoints[2 * edge + 1]);
            let (source, target) = (vertex as u32, parts.adj_targets[slot]);
            let forward = (source, target) == (from, to);
            let backward = (source, target) == (to, from);
            if !(forward || backward) || parts.adj_meters[slot] != parts.lengths[edge] {
                return Err(corrupted(path, "street adjacency rows"));
            }
            // Each undirected edge must appear once per direction; a loop
            // edge's two identical rows fill whichever slot is still open.
            let counts = &mut edge_directions[edge];
            if forward && counts.0 == 0 {
                counts.0 = 1;
            } else if backward && counts.1 == 0 {
                counts.1 = 1;
            } else {
                return Err(corrupted(path, "street adjacency rows"));
            }
        }
    }
    if edge_directions.iter().any(|&counts| counts != (1, 1)) {
        return Err(corrupted(path, "street adjacency rows"));
    }
    if !parts.adj_meters.iter().all(|&m| m.is_finite() && m >= 0.0)
        || !parts.lengths.iter().all(|&m| m.is_finite() && m >= 0.0)
        || !parts.cumulative.iter().all(|&m| m.is_finite() && m >= 0.0)
    {
        return Err(corrupted(path, "street costs"));
    }
    if parts.index_level_starts != expected_levels {
        return Err(corrupted(path, "street index shape"));
    }
    // The leaves must be exactly the segment set — every consecutive
    // coordinate pair once, none missing or repeated — or snapping would
    // silently skip streets.
    let mut seen = vec![false; coordinates];
    for payload in parts.index_payload.as_chunks::<2>().0 {
        let (edge, segment) = (payload[0] as usize, payload[1] as usize);
        if edge >= edges
            || (segment as u64) < u64::from(parts.coordinate_offsets[edge])
            || (segment as u64 + 1) >= u64::from(parts.coordinate_offsets[edge + 1])
            || std::mem::replace(&mut seen[segment], true)
        {
            return Err(corrupted(path, "street index payloads"));
        }
    }
    for link in &parts.links {
        // The denormalised endpoints must restate the edge's own, since
        // the vertex→link index is rebuilt from them.
        if link.from != parts.endpoints[2 * link.edge as usize]
            || link.to != parts.endpoints[2 * link.edge as usize + 1]
        {
            return Err(corrupted(path, "street link records"));
        }
    }
    Ok(parts)
}

/// Parses and bounds-checks a container's header and section directory.
/// Checksums are the caller's job — the two load paths verify different
/// sections.
pub(super) fn parse_container(
    path: &str,
    bytes: &[u8],
    magic: &[u8; 8],
    expected_format: u32,
) -> PyResult<ContainerLayout> {
    parse_header(path, bytes, bytes.len() as u64, magic, expected_format)
}

/// `parse_container` over a head slice of a file whose full length is
/// known separately — the light path that never reads the sections.
pub(super) fn parse_header(
    path: &str,
    bytes: &[u8],
    total: u64,
    magic: &[u8; 8],
    expected_format: u32,
) -> PyResult<ContainerLayout> {
    let take = |offset: usize, length: usize| -> PyResult<&[u8]> {
        offset
            .checked_add(length)
            .and_then(|end| bytes.get(offset..end))
            .ok_or_else(|| corrupted(path, "truncated header"))
    };
    let kind = if magic == STREET_ARTIFACT_MAGIC {
        "street"
    } else {
        "network"
    };
    if take(0, 8)? != magic {
        return Err(PyValueError::new_err(format!(
            "'{path}' is not a cafein {kind} artifact"
        )));
    }
    let format = u32::from_le_bytes(take(8, 4)?.try_into().unwrap());
    let version_length = u16::from_le_bytes(take(12, 2)?.try_into().unwrap()) as usize;
    let version = String::from_utf8_lossy(take(14, version_length)?).into_owned();
    if format != expected_format {
        return Err(PyValueError::new_err(format!(
            "'{path}' uses {kind} artifact format {format} (written by cafein \
             {version}), which this cafein ({}) cannot read; rebuild \
             the network from its inputs and save it again",
            env!("CARGO_PKG_VERSION"),
        )));
    }
    let mut cursor = 14 + version_length;
    let section_count = u32::from_le_bytes(take(cursor, 4)?.try_into().unwrap());
    cursor += 4;
    if section_count != 2 {
        return Err(corrupted(path, "section directory shape"));
    }
    let mut sections = Vec::new();
    for _ in 0..2 {
        let tag = u16::from_le_bytes(take(cursor, 2)?.try_into().unwrap());
        let offset = u64::from_le_bytes(take(cursor + 2, 8)?.try_into().unwrap());
        let length = u64::from_le_bytes(take(cursor + 10, 8)?.try_into().unwrap());
        let checksum = u32::from_le_bytes(take(cursor + 18, 4)?.try_into().unwrap());
        cursor += 22;
        sections.push((tag, offset, length, checksum));
    }
    let directory_end = cursor as u64;
    match sections.as_slice() {
        &[(SECTION_META, meta_offset, meta_length, meta_crc), (SECTION_STREETS, streets_offset, streets_length, streets_crc)] =>
        {
            let meta_end = meta_offset.checked_add(meta_length);
            let streets_end = streets_offset.checked_add(streets_length);
            if meta_offset < directory_end
                || meta_end.is_none_or(|end| end > streets_offset)
                || streets_end.is_none_or(|end| end > total)
            {
                return Err(corrupted(path, "section bounds"));
            }
            // The writer starts a non-empty STREETS section on the
            // alignment boundary; loads enforce the invariant so a
            // mapped load can rely on it.
            if streets_length > 0 && !streets_offset.is_multiple_of(STREETS_ALIGNMENT) {
                return Err(corrupted(path, "street section alignment"));
            }
            Ok(ContainerLayout {
                meta_offset,
                meta_length,
                meta_crc,
                streets_offset,
                streets_length,
                streets_crc,
            })
        }
        _ => Err(corrupted(path, "section directory shape")),
    }
}

/// A parsed artifact: the decoded META, the adopted street network, and
/// how many STREETS-section bytes the load explicitly read.
pub(super) type LoadedArtifact = (
    Artifact,
    Option<StreetNetwork>,
    Option<StreetNetwork>,
    u64,
    (String, u32, u32),
);

/// Validates a persisted walking hierarchy against the street graph it rides
/// with, before its buckets are rebuilt on load: it must accompany a street
/// network, cover exactly that graph's vertices, and be internally consistent.
/// When the street CSR is already materialised (`csr_read` — every owned load,
/// and a mapped load with `verify`), it must also carry a fingerprint
/// reproducing that CSR, binding the hierarchy to its exact graph. A lazy mapped
/// load skips only that fingerprint step, since recomputing it would page the
/// STREETS section the lazy path deliberately leaves unread — the same trust the
/// lazy path already extends to the street arrays themselves; the shape check
/// (which reads only META) still runs, so a rebuild never indexes out of bounds.
pub(super) fn validate_walking_hierarchy(
    path: &str,
    artifact: &Artifact,
    streets: &Option<StreetNetwork>,
    csr_read: bool,
) -> PyResult<()> {
    if let Some(hierarchy) = &artifact.walking_hierarchy {
        let matches = streets.as_ref().is_some_and(|network| {
            hierarchy.vertex_count() == network.vertex_count()
                && hierarchy.is_consistent()
                && (!csr_read || hierarchy.graph_fingerprint() == network.graph_fingerprint())
        });
        if !matches {
            return Err(corrupted(path, "walking hierarchy shape"));
        }
    }
    Ok(())
}

/// Loads an artifact into owned memory — the default path. `verify`
/// (default on: the bytes are read anyway) toggles the STREETS checksum;
/// META is always checked before anything is decoded.
pub(super) fn load_owned(path: &str, verify: Option<bool>) -> PyResult<LoadedArtifact> {
    let bytes = std::fs::read(path).map_err(io_error)?;
    let layout = parse_container(path, &bytes, ARTIFACT_MAGIC, ARTIFACT_FORMAT)?;
    let meta =
        &bytes[layout.meta_offset as usize..(layout.meta_offset + layout.meta_length) as usize];
    let section = &bytes
        [layout.streets_offset as usize..(layout.streets_offset + layout.streets_length) as usize];
    if crc32(meta) != layout.meta_crc {
        return Err(corrupted(path, "checksum mismatch"));
    }
    if verify.unwrap_or(true) && crc32(section) != layout.streets_crc {
        return Err(corrupted(path, "checksum mismatch"));
    }
    let mut artifact: Artifact =
        bincode::deserialize(meta).map_err(|error| PyValueError::new_err(error.to_string()))?;
    if artifact.streets.is_some() && section.is_empty() {
        return Err(corrupted(path, "missing street section"));
    }
    let stop_count = artifact.timetable.stop_count();
    let next_block = artifact.multimodal.as_ref().map(multimodal_block_start);
    let streets = match artifact.streets.take() {
        Some(streets_meta) => {
            let (expected_levels, block_end) =
                validate_street_shape(path, &streets_meta, 0, layout.streets_length, stop_count)?;
            check_block_boundary(path, block_end, next_block, layout.streets_length)?;
            let parts = decode_streets(path, streets_meta, section, expected_levels)?;
            Some(
                StreetNetwork::from_parts(parts)
                    .map_err(|_| corrupted(path, "street attributes"))?,
            )
        }
        None => None,
    };
    let multimodal = match &artifact.multimodal {
        Some(meta) => {
            let start = multimodal_block_start(meta);
            let (expected_levels, block_end) =
                validate_street_shape(path, meta, start, layout.streets_length, 0)?;
            check_block_boundary(path, block_end, None, layout.streets_length)?;
            let parts = decode_streets(path, meta.clone(), section, expected_levels)?;
            Some(adopt_multimodal(path, parts, meta)?)
        }
        None => None,
    };
    validate_walking_hierarchy(path, &artifact, &streets, true)?;
    Ok((
        artifact,
        streets,
        multimodal,
        layout.streets_length,
        (
            canonical_source_path(path),
            layout.meta_crc,
            layout.streets_crc,
        ),
    ))
}

/// Builds the multimodal graph from decoded parts and revalidates the
/// elevation invariants construction enforces: metadata present exactly when
/// the elevations are, and its declared fields within bounds.
fn adopt_multimodal(
    path: &str,
    parts: StreetNetworkParts,
    meta: &StreetsMeta,
) -> PyResult<StreetNetwork> {
    let network = StreetNetwork::from_parts(parts)
        .map_err(|_| corrupted(path, "multimodal street attributes"))?;
    check_multimodal_elevation(path, &network, meta)?;
    Ok(network)
}

/// The elevation-consistency check both load paths share.
fn check_multimodal_elevation(
    path: &str,
    network: &StreetNetwork,
    meta: &StreetsMeta,
) -> PyResult<()> {
    if network.elevations().is_some() != meta.elevation.is_some() {
        return Err(corrupted(path, "multimodal elevation metadata"));
    }
    if let Some(elevation) = &meta.elevation {
        if crate::streets::check_elevation_meta(elevation, network.edge_count()).is_err() {
            return Err(corrupted(path, "multimodal elevation metadata"));
        }
    }
    Ok(())
}

/// Loads an artifact with the street arrays as views into a memory map.
/// `Ok(Err(reason))` means mapping is environmentally unavailable and the
/// caller decides between fallback and error; artifact problems are hard
/// errors on every path.
pub(super) fn load_mapped(
    path: &str,
    verify: Option<bool>,
) -> PyResult<Result<LoadedArtifact, String>> {
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
    let layout = parse_container(path, bytes, ARTIFACT_MAGIC, ARTIFACT_FORMAT)?;
    let meta =
        &bytes[layout.meta_offset as usize..(layout.meta_offset + layout.meta_length) as usize];
    if crc32(meta) != layout.meta_crc {
        return Err(corrupted(path, "checksum mismatch"));
    }
    // The STREETS checksum would page the whole section in and defeat
    // the lazy load, so it is opt-in here; without it the street content
    // is trusted the way any store trusts its own files — the shape
    // validation and slice bounds checks turn corruption into an error
    // or a wrong result, never into unsoundness.
    let mut streets_read = 0u64;
    if verify == Some(true) {
        let section = &bytes[layout.streets_offset as usize
            ..(layout.streets_offset + layout.streets_length) as usize];
        if crc32(section) != layout.streets_crc {
            return Err(corrupted(path, "checksum mismatch"));
        }
        streets_read = layout.streets_length;
    }
    let mut artifact: Artifact =
        bincode::deserialize(meta).map_err(|error| PyValueError::new_err(error.to_string()))?;
    if artifact.streets.is_some() && layout.streets_length == 0 {
        return Err(corrupted(path, "missing street section"));
    }
    let stop_count = artifact.timetable.stop_count();
    let next_block = artifact.multimodal.as_ref().map(multimodal_block_start);
    let streets = match artifact.streets.take() {
        Some(streets_meta) => {
            let (_, block_end) =
                validate_street_shape(path, &streets_meta, 0, layout.streets_length, stop_count)?;
            check_block_boundary(path, block_end, next_block, layout.streets_length)?;
            let ranges: Vec<(u64, u64)> = streets_meta
                .descriptors
                .iter()
                .map(|descriptor| (layout.streets_offset + descriptor.offset, descriptor.count))
                .collect();
            // The core CSR/geometry arrays stay mapped; the optional
            // multimodal arrays are decoded owned, paging in only their
            // bytes (a walk-only artifact has none and stays fully lazy).
            let section = &bytes[layout.streets_offset as usize
                ..(layout.streets_offset + layout.streets_length) as usize];
            let (attributes, car, elevations) =
                decode_optional_street_arrays(section, &streets_meta.descriptors);
            if verify != Some(true) {
                streets_read += streets_meta.descriptors[STREET_ARRAY_ORDER.len()..]
                    .iter()
                    .map(|descriptor| descriptor.count * descriptor.kind.size())
                    .sum::<u64>();
            }
            let spec = MappedStreets {
                // A clone of the Arc: the map stays borrowed for the
                // multimodal decode below.
                backing: backing.clone(),
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
            Some(
                StreetNetwork::from_mapped(spec)
                    .map_err(|_| corrupted(path, "street array bounds"))?,
            )
        }
        None => None,
    };
    // The multimodal union graph maps exactly as the walking one: core CSR
    // and geometry in place, the optional attribute/elevation arrays decoded
    // owned — the same laziness the street artifact has.
    let multimodal = match &artifact.multimodal {
        Some(meta) => {
            let start = multimodal_block_start(meta);
            let (_, block_end) =
                validate_street_shape(path, meta, start, layout.streets_length, 0)?;
            check_block_boundary(path, block_end, None, layout.streets_length)?;
            let ranges: Vec<(u64, u64)> = meta
                .descriptors
                .iter()
                .map(|descriptor| (layout.streets_offset + descriptor.offset, descriptor.count))
                .collect();
            let section = &bytes[layout.streets_offset as usize
                ..(layout.streets_offset + layout.streets_length) as usize];
            let (attributes, car, elevations) =
                decode_optional_street_arrays(section, &meta.descriptors);
            if verify != Some(true) {
                streets_read += meta.descriptors[STREET_ARRAY_ORDER.len()..]
                    .iter()
                    .map(|descriptor| descriptor.count * descriptor.kind.size())
                    .sum::<u64>();
            }
            let spec = MappedStreets {
                backing: backing.clone(),
                vertex_count: meta.vertex_count,
                links: meta.links.clone(),
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
            let network = StreetNetwork::from_mapped(spec)
                .map_err(|_| corrupted(path, "multimodal street array bounds"))?;
            check_multimodal_elevation(path, &network, meta)?;
            Some(network)
        }
        None => None,
    };
    // Only the `verify` path has paged (and CRC-checked) the STREETS section, so
    // only there is recomputing the CSR fingerprint free of extra reads.
    validate_walking_hierarchy(path, &artifact, &streets, verify == Some(true))?;
    Ok(Ok((
        artifact,
        streets,
        multimodal,
        streets_read,
        (
            canonical_source_path(path),
            layout.meta_crc,
            layout.streets_crc,
        ),
    )))
}

fn decode_timer() -> crate::logging::PhaseTimer {
    crate::logging::PhaseTimer::start(
        "cafein.artifact",
        "artifact.load.decode",
        "decoding the artifact payload",
        "decoded the artifact payload",
    )
}

/// `assemble` with the rebuild phase timed around it.
fn rebuilt(loaded: LoadedArtifact) -> TransportNetwork {
    let timer = crate::logging::PhaseTimer::start(
        "cafein.artifact",
        "artifact.load.rebuild",
        "rebuilding the derived structures",
        "rebuilt the derived structures",
    );
    let network = assemble(loaded);
    timer.finish();
    network
}

/// Assembles a network from a loaded artifact, rebuilding the derived
/// lookup tables.
pub(super) fn assemble(
    (artifact, streets, multimodal, streets_bytes_read, source): LoadedArtifact,
) -> TransportNetwork {
    let Artifact {
        feed,
        timetable,
        services,
        transfers,
        geometry,
        leg_geometry,
        streets: _,
        ultra_transfers,
        ultra_window,
        mcultra_transfers,
        mcultra_window,
        mcultra_factors,
        walking_hierarchy,
        tbtr_time_transfers,
        mctbtr_transfers,
        multimodal: multimodal_meta,
        multimodal_modes,
        mode_transfers,
        carriage_transfers,
    } = artifact;
    let carriage_transfers = carriage_transfers.map(|persisted| {
        let PersistedCarriage {
            mode,
            budget,
            mut set,
            ride_edge,
            ride_network_meters,
        } = persisted;
        // A persisted carriage set is never a closure.
        set.mark_unclosed();
        crate::CarriageTransferSet {
            mode,
            budget,
            set,
            ride_edge,
            ride_network_meters,
        }
    });
    let mode_transfers = mode_transfers.map(|persisted| {
        let PersistedModeTransfers {
            mode,
            budget,
            mut set,
            tokens,
            rental_network_meters,
            rental_edge,
        } = persisted;
        // A persisted merged set is never a closure; the skip-serialized
        // marking must not default it back to one.
        set.mark_unclosed();
        crate::ModeTransferSet {
            mode,
            budget,
            set,
            tokens: tokens.into_iter().collect(),
            rental_network_meters,
            rental_edge,
        }
    });
    let multimodal_elevation = multimodal_meta.and_then(|meta| meta.elevation);
    // The contraction persisted; its buckets are derived state, rebuilt here on
    // the loading thread exactly as `install_hierarchy` builds them for a fresh
    // contraction, so a loaded network matches a freshly built one.
    let mut streets = streets;
    if let (Some(network), Some(hierarchy)) = (streets.as_mut(), walking_hierarchy) {
        network.install_hierarchy_from(hierarchy);
    }
    let build = TimetableBuild {
        timetable,
        services,
        quarantined: Vec::new(),
        interpolated: Vec::new(),
    };
    let (stops_by_id, stops_by_qualified_id, trips_by_public_id) =
        derived_indexes(&feed, &build.timetable);
    TransportNetwork {
        feed,
        build,
        transfers,
        ultra_transfers,
        ultra_window,
        mcultra_transfers,
        mcultra_window,
        mcultra_factors,
        tbtr_time_transfers,
        mctbtr_transfers,
        geometry,
        leg_geometry,
        streets,
        multimodal,
        multimodal_elevation,
        multimodal_modes,
        multimodal_links: std::sync::OnceLock::new(),
        multimodal_profiles: std::sync::Mutex::new(Vec::new()),
        mode_transfers,
        carriage_transfers,
        stops_by_id,
        stops_by_qualified_id,
        trips_by_public_id,
        streets_bytes_read,
        streets_generation: 0,
        transfers_generation: 0,
        source: Some(source),
    }
}
