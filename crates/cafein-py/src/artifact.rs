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
        use std::io::Write;

        let parts = self.streets.as_ref().map(StreetNetwork::to_parts);
        py.allow_threads(|| {
            let (streets_meta, streets_bytes) = match &parts {
                Some(parts) => {
                    let (descriptors, bytes) = encode_streets(parts);
                    (
                        Some(StreetsMeta {
                            vertex_count: parts.vertex_count,
                            links: parts.links.clone(),
                            descriptors,
                        }),
                        bytes,
                    )
                }
                None => (None, Vec::new()),
            };
            let artifact = ArtifactRef {
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
                mcultra_factor: self.mcultra_factor,
                walking_hierarchy: self.streets.as_ref().and_then(StreetNetwork::hierarchy),
                tbtr_time_transfers: &self.tbtr_time_transfers,
                mctbtr_transfers: &self.mctbtr_transfers,
            };
            let meta = bincode::serialize(&artifact)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;

            // Layout: header | directory | META … pad … | STREETS. The
            // STREETS section starts on `STREETS_ALIGNMENT`, so a mapped
            // load never shares an OS page between the sections; without
            // a street network there is nothing to align (or to pad —
            // padding bytes sit outside every section CRC).
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

            // Stage into a sibling temp file and atomically rename over
            // the destination: an artifact must never be rewritten in
            // place under live mapped readers, whose mappings keep the
            // replaced inode valid. The name is unique per process and
            // save, and creation is exclusive, so concurrent saves never
            // share a staging path and a stale file or symlink at it
            // fails the save instead of being written through.
            static SAVE_SEQUENCE: std::sync::atomic::AtomicU64 =
                std::sync::atomic::AtomicU64::new(0);
            let sequence = SAVE_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let temporary = format!("{path}.tmp-{}-{sequence}", std::process::id());
            let write = || -> PyResult<()> {
                let file = std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&temporary)
                    .map_err(io_error)?;
                let mut writer = std::io::BufWriter::new(file);
                writer.write_all(ARTIFACT_MAGIC).map_err(io_error)?;
                writer
                    .write_all(&ARTIFACT_FORMAT.to_le_bytes())
                    .map_err(io_error)?;
                writer
                    .write_all(&(version.len() as u16).to_le_bytes())
                    .map_err(io_error)?;
                writer.write_all(version).map_err(io_error)?;
                writer.write_all(&2u32.to_le_bytes()).map_err(io_error)?;
                for (tag, offset, bytes) in [
                    (SECTION_META, meta_offset, &meta),
                    (SECTION_STREETS, streets_offset, &streets_bytes),
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
                writer.write_all(&meta).map_err(io_error)?;
                let padding = streets_offset - meta_offset - meta.len() as u64;
                writer
                    .write_all(&vec![0u8; padding as usize])
                    .map_err(io_error)?;
                writer.write_all(&streets_bytes).map_err(io_error)?;
                writer.flush().map_err(io_error)?;
                writer.get_ref().sync_all().map_err(io_error)?;
                // Replacing keeps the destination's permissions, as the
                // old truncate-in-place write did.
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
        })
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
        if mode != MmapMode::Off {
            match py.allow_threads(|| load_mapped(path, verify))? {
                Ok(loaded) => return Ok(assemble(loaded)),
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
        Ok(assemble(loaded))
    }

    /// Whether the street arrays are memory-mapped views of the loaded
    /// artifact.
    #[getter]
    fn mapped(&self) -> bool {
        self.streets.as_ref().is_some_and(StreetNetwork::is_mapped)
    }

    /// STREETS-section bytes the load explicitly read — 0 for a lazy
    /// mapped load. Internal; the laziness tests assert on it.
    #[getter]
    fn _streets_bytes_read(&self) -> u64 {
        self.streets_bytes_read
    }
}

pub(super) const ARTIFACT_MAGIC: &[u8; 8] = b"CAFEINET";

// 10: the persisted time TBTR transfer set became tie-complete (same-ride
// equal-arrival competitors retained for cost-row reconstruction); sets
// written by earlier formats lack the competitors and must be rebuilt.
pub(super) const ARTIFACT_FORMAT: u32 = 10;

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
    /// factor-vector fingerprint it was built for; restored so the heavy
    /// run-once preprocessing need not be repeated and the factor contract holds.
    mcultra_transfers: &'a Option<Transfers>,
    mcultra_window: Option<(u32, u32)>,
    mcultra_factor: Option<u64>,
    /// The walking contraction hierarchy, when installed; restored so the
    /// run-once contraction need not be repeated. Its one-to-many buckets are
    /// derived state, rebuilt on load rather than persisted.
    walking_hierarchy: Option<&'a ContractionHierarchy>,
    /// The cached time-only TBTR transfer set with the date it was built for,
    /// when present; restored so a loaded network reuses it instead of
    /// rebuilding the dominance-aware set.
    tbtr_time_transfers: &'a Option<(String, cafein_core::tbtr::TransferSet)>,
    /// The cached multicriteria TBTR transfer set with its date and factor
    /// fingerprint, when present.
    mctbtr_transfers: &'a Option<(String, u64, cafein_core::tbtr::TransferSet)>,
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
    mcultra_factor: Option<u64>,
    walking_hierarchy: Option<ContractionHierarchy>,
    tbtr_time_transfers: Option<(String, cafein_core::tbtr::TransferSet)>,
    mctbtr_transfers: Option<(String, u64, cafein_core::tbtr::TransferSet)>,
}

/// The street layer's decoded state: link records (endpoints
/// denormalised, so the vertex→link index rebuilds from these alone),
/// scalars, and the descriptor table locating every raw array inside the
/// STREETS section.
#[derive(serde::Serialize, serde::Deserialize)]
pub(super) struct StreetsMeta {
    vertex_count: u32,
    links: Vec<StoredLink>,
    descriptors: Vec<ArrayDescriptor>,
}

/// One raw array inside the STREETS section. Offsets are relative to the
/// section start (absolute positions come from the section directory), so
/// the descriptor table is complete before the file layout is.
#[derive(serde::Serialize, serde::Deserialize, Clone, Copy, PartialEq, Debug)]
pub(super) struct ArrayDescriptor {
    array: StreetArray,
    kind: ArrayKind,
    count: u64,
    offset: u64,
}

/// The street arrays, in their fixed on-disk order.
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
}

/// Element type of a raw street array.
#[derive(serde::Serialize, serde::Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum ArrayKind {
    U32,
    I32,
    F32,
    F64,
}

impl ArrayKind {
    fn size(self) -> u64 {
        match self {
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
    meta_offset: u64,
    meta_length: u64,
    meta_crc: u32,
    streets_offset: u64,
    streets_length: u64,
    streets_crc: u32,
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

/// CRC-32 (IEEE) over the artifact payload.
pub(super) fn crc32(bytes: &[u8]) -> u32 {
    const TABLE: [u32; 256] = {
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
    let mut crc = u32::MAX;
    for &byte in bytes {
        crc = TABLE[((crc ^ byte as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    !crc
}

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
/// order, sequential aligned layout inside the section, counts mutually
/// consistent (so no query indexes out of an array), and well-formed
/// link records. Runs on every load path without touching a single
/// STREETS byte — which is what keeps a mapped load lazy. Returns the
/// level-start table the index must carry.
pub(super) fn validate_street_shape(
    path: &str,
    meta: &StreetsMeta,
    section_length: u64,
    stop_count: u32,
) -> PyResult<Vec<u32>> {
    if meta.descriptors.len() != STREET_ARRAY_ORDER.len() {
        return Err(corrupted(path, "street descriptor table shape"));
    }
    // The arrays must occupy the section exactly as the writer laid them
    // out: sequential, each at the next aligned position — no gaps,
    // overlaps, or aliasing. Checked arithmetic throughout.
    let mut expected_offset = 0u64;
    let mut last_end = 0u64;
    for (descriptor, &(array, kind)) in meta.descriptors.iter().zip(&STREET_ARRAY_ORDER) {
        if descriptor.array != array || descriptor.kind != kind {
            return Err(corrupted(path, "street descriptor table shape"));
        }
        let extent = descriptor
            .count
            .checked_mul(kind.size())
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
    if last_end != section_length {
        return Err(corrupted(path, "street array bounds"));
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
    Ok(expected_levels)
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
    for payload in parts.index_payload.chunks_exact(2) {
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
pub(super) fn parse_container(path: &str, bytes: &[u8]) -> PyResult<ContainerLayout> {
    let total = bytes.len() as u64;
    let take = |offset: usize, length: usize| -> PyResult<&[u8]> {
        offset
            .checked_add(length)
            .and_then(|end| bytes.get(offset..end))
            .ok_or_else(|| corrupted(path, "truncated header"))
    };
    if take(0, 8)? != ARTIFACT_MAGIC {
        return Err(PyValueError::new_err(format!(
            "'{path}' is not a cafein network artifact"
        )));
    }
    let format = u32::from_le_bytes(take(8, 4)?.try_into().unwrap());
    let version_length = u16::from_le_bytes(take(12, 2)?.try_into().unwrap()) as usize;
    let version = String::from_utf8_lossy(take(14, version_length)?).into_owned();
    if format != ARTIFACT_FORMAT {
        return Err(PyValueError::new_err(format!(
            "'{path}' uses artifact format {format} (written by cafein \
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
pub(super) type LoadedArtifact = (Artifact, Option<StreetNetwork>, u64);

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
    let layout = parse_container(path, &bytes)?;
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
    let streets = match artifact.streets.take() {
        Some(streets_meta) => {
            let expected_levels =
                validate_street_shape(path, &streets_meta, layout.streets_length, stop_count)?;
            Some(StreetNetwork::from_parts(decode_streets(
                path,
                streets_meta,
                section,
                expected_levels,
            )?))
        }
        None => None,
    };
    validate_walking_hierarchy(path, &artifact, &streets, true)?;
    Ok((artifact, streets, layout.streets_length))
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
    let layout = parse_container(path, bytes)?;
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
    let streets = match artifact.streets.take() {
        Some(streets_meta) => {
            validate_street_shape(path, &streets_meta, layout.streets_length, stop_count)?;
            let ranges: Vec<(u64, u64)> = streets_meta
                .descriptors
                .iter()
                .map(|descriptor| (layout.streets_offset + descriptor.offset, descriptor.count))
                .collect();
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
            };
            Some(
                StreetNetwork::from_mapped(spec)
                    .map_err(|_| corrupted(path, "street array bounds"))?,
            )
        }
        None => None,
    };
    // Only the `verify` path has paged (and CRC-checked) the STREETS section, so
    // only there is recomputing the CSR fingerprint free of extra reads.
    validate_walking_hierarchy(path, &artifact, &streets, verify == Some(true))?;
    Ok(Ok((artifact, streets, streets_read)))
}

/// Assembles a network from a loaded artifact, rebuilding the derived
/// lookup tables.
pub(super) fn assemble(
    (artifact, streets, streets_bytes_read): LoadedArtifact,
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
        mcultra_factor,
        walking_hierarchy,
        tbtr_time_transfers,
        mctbtr_transfers,
    } = artifact;
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
        mcultra_factor,
        tbtr_time_transfers,
        mctbtr_transfers,
        geometry,
        leg_geometry,
        streets,
        stops_by_id,
        stops_by_qualified_id,
        trips_by_public_id,
        streets_bytes_read,
    }
}
