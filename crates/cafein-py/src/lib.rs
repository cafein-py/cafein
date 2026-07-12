//! Python bindings for cafein.

use std::collections::HashMap;

use chrono::NaiveDate;
use numpy::{IntoPyArray, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use cafein_core::ch::ContractionHierarchy;
use cafein_core::exhaustive;
use cafein_core::fares::{FareTables, RuleFares, ZoneFares, ZoneProduct, NO_FARE};
use cafein_core::geometry::{wkb_multi_line_string, DistanceProvenance, LegGeometry, TripGeometry};
use cafein_core::journey::{Journey, Leg};
use cafein_core::mcraptor;
use cafein_core::mctbtr::McTbtrEngine;
use cafein_core::mcultra::compute_mcultra_shortcuts;
use cafein_core::raptor::{CostInputs, CostRow, Objective, Raptor};
use cafein_core::router::{Request, TransitRouter};
use cafein_core::streets::{
    Backing, MappedStreets, Snap, StopLink, StoredLink, StreetNetwork, StreetNetworkParts,
    WalkedStop,
};
use cafein_core::tbtr::{DayView, TbtrEngine};
use cafein_core::timetable::{StopIdx, Timetable, TripIdx};
use cafein_core::transfers::Transfers;
use cafein_core::ultra::{compute_shortcuts, Shortcut};
use cafein_gtfs::{build_timetable, Feed, RouteType, ServiceCalendar, TimetableBuild};

/// A routable public-transport network built from GTFS data.
#[pyclass]
struct TransportNetwork {
    feed: Feed,
    build: TimetableBuild,
    transfers: Transfers,
    /// The ULTRA shortcut set, when computed (`compute_ultra_shortcuts`):
    /// the intermediate transfers as a `Transfers`, derived from the
    /// installed street network. The time engines relax it in place of the
    /// closure `transfers` when it is present; the emissions/fare engines
    /// always use the closure. Persisted with the artifact and restored on
    /// load, scoped by `ultra_window`.
    ultra_transfers: Option<Transfers>,
    /// The source-departure window the ULTRA set was computed for, in
    /// seconds since midnight. A time query outside it falls back to the
    /// closure, so a partial-window set never silently drops transfers.
    ultra_window: Option<(u32, u32)>,
    /// The McULTRA (emissions-aware) shortcut set, when computed
    /// (`compute_mcultra_shortcuts`): the coordinate emissions engines relax it
    /// in place of the closure when a whole-day set is present and the query's
    /// factor vector matches the one it was built for (`mcultra_factor`).
    /// Persisted with the artifact and restored on load, with its window and
    /// factor fingerprint.
    mcultra_transfers: Option<Transfers>,
    mcultra_window: Option<(u32, u32)>,
    /// A fingerprint of the per-trip emission-factor vector the McULTRA set was
    /// built with; a query using different factors falls back to the closure.
    mcultra_factor: Option<u64>,
    /// The cached time-only TBTR transfer set, when computed
    /// (`compute_tbtr_transfers`), keyed by the date string it was built for.
    /// A `router="tbtr"` stop time matrix on the same date — single-departure
    /// or windowed — reuses it (build once, query many); other queries rebuild
    /// ad-hoc. Persisted with the artifact and restored on load, keyed by its
    /// date.
    tbtr_time_transfers: Option<(String, cafein_core::tbtr::TransferSet)>,
    geometry: Option<TripGeometry>,
    leg_geometry: Option<LegGeometry>,
    streets: Option<StreetNetwork>,
    stops_by_id: HashMap<String, StopLookup>,
    stops_by_qualified_id: HashMap<String, StopIdx>,
    trips_by_public_id: HashMap<String, TripIdx>,
    /// STREETS-section bytes the load explicitly read — 0 for a lazy
    /// mapped load; the laziness tests assert on it.
    streets_bytes_read: u64,
}

/// Resolution of a raw GTFS stop_id, which merged feeds can duplicate.
#[derive(Clone, Copy)]
enum StopLookup {
    Unique(StopIdx),
    Ambiguous,
}

/// A coordinate query's exact walk lengths in meters, keyed by the stop
/// each access or egress link enters the network through.
struct WalkMaps {
    access: HashMap<StopIdx, f64>,
    egress: HashMap<StopIdx, f64>,
}

impl WalkMaps {
    fn new(access: &[WalkedStop], egress: &[WalkedStop]) -> WalkMaps {
        let meters =
            |walks: &[WalkedStop]| walks.iter().map(|walk| (walk.stop, walk.meters)).collect();
        WalkMaps {
            access: meters(access),
            egress: meters(egress),
        }
    }
}

/// The `(stop, seconds)` request offsets of a walking-search result.
fn request_offsets(walks: &[WalkedStop]) -> Vec<(StopIdx, u32)> {
    walks.iter().map(|walk| (walk.stop, walk.seconds)).collect()
}

/// A coordinate query's endpoints, for drawing its walk legs.
struct CoordinateEnds {
    origin: (f64, f64),
    origin_snap: Snap,
    destination: (f64, f64),
    destination_snap: Snap,
}

const ARTIFACT_MAGIC: &[u8; 8] = b"CAFEINET";
const ARTIFACT_FORMAT: u32 = 8;
/// Section tags in the container directory.
const SECTION_META: u16 = 1;
const SECTION_STREETS: u16 = 2;
/// The STREETS section starts on this boundary (covers every target
/// platform's page and allocation granularity), so a mapped load never
/// shares an OS page between META and STREETS.
const STREETS_ALIGNMENT: u64 = 65_536;
/// Every street array starts 8-byte aligned within the STREETS section.
const ARRAY_ALIGNMENT: u64 = 8;

/// The decoded part of the saved network (the META section), borrowed
/// for writing. The street layer's large arrays live in the STREETS
/// section as raw little-endian values; META carries only their
/// descriptor table plus the small link records and scalars.
#[derive(serde::Serialize)]
struct ArtifactRef<'a> {
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
}

/// The decoded part of the saved network, owned after reading.
#[derive(serde::Deserialize)]
struct Artifact {
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
}

/// The street layer's decoded state: link records (endpoints
/// denormalised, so the vertex→link index rebuilds from these alone),
/// scalars, and the descriptor table locating every raw array inside the
/// STREETS section.
#[derive(serde::Serialize, serde::Deserialize)]
struct StreetsMeta {
    vertex_count: u32,
    links: Vec<StoredLink>,
    descriptors: Vec<ArrayDescriptor>,
}

/// One raw array inside the STREETS section. Offsets are relative to the
/// section start (absolute positions come from the section directory), so
/// the descriptor table is complete before the file layout is.
#[derive(serde::Serialize, serde::Deserialize, Clone, Copy, PartialEq, Debug)]
struct ArrayDescriptor {
    array: StreetArray,
    kind: ArrayKind,
    count: u64,
    offset: u64,
}

/// The street arrays, in their fixed on-disk order.
#[derive(serde::Serialize, serde::Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
enum StreetArray {
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
enum ArrayKind {
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
const STREET_ARRAY_ORDER: [(StreetArray, ArrayKind); 13] = [
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
struct MappedArtifact(memmap2::Mmap);

impl Backing for MappedArtifact {
    fn bytes(&self) -> &[u8] {
        &self.0
    }
}

/// How `load` should back the street arrays.
#[derive(PartialEq, Clone, Copy)]
enum MmapMode {
    Off,
    Auto,
    Require,
}

/// The section directory of a parsed container: everything `load` needs
/// to locate the sections, checksums still unchecked.
struct ContainerLayout {
    meta_offset: u64,
    meta_length: u64,
    meta_crc: u32,
    streets_offset: u64,
    streets_length: u64,
    streets_crc: u32,
}

/// The stop and trip lookup tables derived from a feed and timetable.
type DerivedIndexes = (
    HashMap<String, StopLookup>,
    HashMap<String, StopIdx>,
    HashMap<String, TripIdx>,
);

fn derived_indexes(feed: &Feed, timetable: &Timetable) -> DerivedIndexes {
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

fn io_error(error: std::io::Error) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// CRC-32 (IEEE) over the artifact payload.
fn crc32(bytes: &[u8]) -> u32 {
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
fn corrupted(path: &str, what: &str) -> PyErr {
    PyValueError::new_err(format!(
        "'{path}' is corrupted ({what}); rebuild the network from its \
         inputs and save it again"
    ))
}

/// Serializes a street network's parts into the raw STREETS bytes and the
/// descriptor table locating each array within them.
fn encode_streets(parts: &StreetNetworkParts) -> (Vec<ArrayDescriptor>, Vec<u8>) {
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
fn expected_level_starts(leaves: usize) -> Vec<u32> {
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
fn validate_street_shape(
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
fn decode_streets(
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
fn parse_container(path: &str, bytes: &[u8]) -> PyResult<ContainerLayout> {
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
type LoadedArtifact = (Artifact, Option<StreetNetwork>, u64);

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
fn validate_walking_hierarchy(
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
fn load_owned(path: &str, verify: Option<bool>) -> PyResult<LoadedArtifact> {
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
fn load_mapped(path: &str, verify: Option<bool>) -> PyResult<Result<LoadedArtifact, String>> {
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
fn assemble((artifact, streets, streets_bytes_read): LoadedArtifact) -> TransportNetwork {
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
        geometry,
        leg_geometry,
        streets,
        stops_by_id,
        stops_by_qualified_id,
        trips_by_public_id,
        streets_bytes_read,
    }
}

/// Parses the flat fare tables `cafein.fares` produces, validating the
/// arrays against the network's route and stop counts.
fn fare_tables(
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
fn parse_objective(objective: &str, fares: Option<&FareTables>) -> PyResult<Objective> {
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
fn cost_rows_dict(
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

/// Rejects an empty or out-of-range window/percentile specification.
fn validate_window(window: u32, percentiles: &[f64]) -> PyResult<()> {
    if window == 0 {
        return Err(PyValueError::new_err("window must be at least 1 second"));
    }
    if percentiles.is_empty() {
        return Err(PyValueError::new_err("percentiles must not be empty"));
    }
    for &percentile in percentiles {
        if !percentile.is_finite() || !(0.0..=100.0).contains(&percentile) {
            return Err(PyValueError::new_err(
                "percentiles must be finite and within [0, 100]",
            ));
        }
    }
    Ok(())
}

/// Rejects non-finite point coordinates with the offending index.
fn validate_points(points: &[(f64, f64)]) -> PyResult<()> {
    for (index, &(lat, lon)) in points.iter().enumerate() {
        if !lat.is_finite() || !lon.is_finite() {
            return Err(PyValueError::new_err(format!(
                "point {index} has non-finite coordinates"
            )));
        }
    }
    Ok(())
}

/// The indices of points the linking could not snap.
fn unsnapped(links: &[Option<Vec<WalkedStop>>]) -> Vec<u32> {
    links
        .iter()
        .enumerate()
        .filter_map(|(index, links)| links.is_none().then_some(index as u32))
        .collect()
}

/// Each destination point's `(stop, seconds, meters)` egress table;
/// unsnapped points get an empty table and stay unreachable.
fn egress_tables(links: &[Option<Vec<WalkedStop>>]) -> Vec<Vec<(StopIdx, u32, f64)>> {
    links
        .iter()
        .map(|links| {
            links
                .as_deref()
                .unwrap_or(&[])
                .iter()
                .map(|walk| (walk.stop, walk.seconds, walk.meters))
                .collect()
        })
        .collect()
}

/// A deterministic, NaN-safe fingerprint of a per-trip emission-factor vector,
/// binding a McULTRA set to the factor configuration it was built with (a query
/// with different factors falls back to the closure). Not a cryptographic digest.
fn factor_fingerprint(per_trip: &[f64]) -> u64 {
    const PRIME: u64 = 0x100000001b3;
    let mut hash = 0xcbf29ce484222325u64;
    for &factor in per_trip {
        hash = (hash ^ factor.to_bits()).wrapping_mul(PRIME);
    }
    (hash ^ per_trip.len() as u64).wrapping_mul(PRIME)
}

/// Overlays an origin's explicit direct street walks onto its emissions cost
/// cells: a walking-only journey — zero rides, its walked metres, zero emissions
/// under today's walking factor — wins a destination cell whenever nothing
/// transit-side is cleaner. `walks` is `(destination slot, walk seconds, walk
/// metres)` with the diagonal (origin coordinate == destination coordinate)
/// already zeroed by the caller; a walk beyond the travel-time `budget` is
/// dropped. `priced` prices the walk at zero fare when a fare model is present.
fn merge_direct_walk_cells(
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
            mcultra_factor: None,
            tbtr_time_transfers: None,
            geometry: None,
            leg_geometry: None,
            streets: None,
            stops_by_id,
            stops_by_qualified_id,
            trips_by_public_id,
            streets_bytes_read: 0,
        })
    }

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

    /// The McULTRA set's stored factor-vector fingerprint, or `None`. For
    /// inspection/tests (the fingerprint binds the set to its factor config).
    #[getter]
    fn _mcultra_factor(&self) -> Option<u64> {
        self.mcultra_factor
    }

    /// Whether an emissions query with these `factors` would relax the installed
    /// McULTRA set (a whole-day set whose factor fingerprint matches) rather than
    /// the closure. Exposes the `emissions_transfers` gate for inspection/tests.
    fn mcultra_active_for(&self, factors: Vec<(String, f64)>) -> bool {
        let mut per_trip = vec![f64::NAN; self.build.timetable.trip_count() as usize];
        for (trip_id, factor) in &factors {
            if let Some(&trip) = self.trips_by_public_id.get(trip_id) {
                per_trip[trip.0 as usize] = *factor;
            }
        }
        !std::ptr::eq(
            self.emissions_transfers(factor_fingerprint(&per_trip)),
            &self.transfers,
        )
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
        let fingerprint = factor_fingerprint(&per_trip);
        self.mcultra_transfers = Some(set);
        self.mcultra_window = Some((min_departure, max_departure));
        self.mcultra_factor = Some(fingerprint);
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
        self.mcultra_factor = None;
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
        self.mcultra_factor = None;
        Ok(())
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
    /// Returns
    /// -------
    /// list of dict
    ///     Without `window`, the Pareto set of journeys over (arrival
    ///     time, number of rides) leaving at the departure time; with it,
    ///     the departure-window profile. Each journey carries its legs;
    ///     times are seconds past the service day's start.
    #[pyo3(signature = (from_stop, to_stop, date, departure, max_transfers = 7, window = None, walking_speed_kmph = 3.6, max_walking_time = 7200.0, max_snap_distance = 1600.0, geometries = true))]
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
        walking_speed_kmph: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
        geometries: bool,
    ) -> PyResult<Py<PyList>> {
        let origin = self.resolve_stop(from_stop)?;
        let destination = self.resolve_stop(to_stop)?;
        // With a whole-day ULTRA set, route door-to-door between the stops'
        // coordinates for unrestricted walking; otherwise board at the origin
        // stop and relax the closure (today's behaviour).
        if self.ultra_active() {
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
                        walking_speed_kmph,
                        max_walking_time,
                        max_snap_distance,
                        geometries,
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
        };
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
    /// Returns
    /// -------
    /// list of dict
    ///     Journeys as in ``route_between_stops``; arrivals include the
    ///     egress walk.
    #[pyo3(signature = (origin, destination, date, departure, max_transfers = 7, window = None, walking_speed_kmph = 3.6, max_walking_time = 7200.0, max_snap_distance = 1600.0, geometries = true))]
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
        walking_speed_kmph: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
        geometries: bool,
    ) -> PyResult<Py<PyList>> {
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
        // ULTRA-routed leg is measured in the ULTRA set.
        let transfers = self.time_transfers();
        let journeys = match window {
            None => Raptor.route(&self.build.timetable, transfers, &request),
            Some(window) => Raptor.route_range(&self.build.timetable, transfers, &request, window),
        };
        let kept: Vec<&Journey> = journeys
            .iter()
            .filter(|journey| match direct {
                Some((walk_seconds, _)) => journey.arrival - journey.departure < walk_seconds,
                None => true,
            })
            .collect();
        let result = PyList::empty(py);
        if let Some(walk) = direct.filter(|_| !kept.iter().any(|journey| journey.rides() == 0)) {
            // Journeys sort by (departure, rides); the walk leaves at the
            // requested departure with zero rides, so it leads the list.
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
                transfers,
            )?)?;
        }
        Ok(result.unbind())
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
    #[pyo3(signature = (origin, date, departure, max_transfers = 7, walking_speed_kmph = 3.6, max_walking_time = 7200.0, max_snap_distance = 1600.0))]
    #[allow(clippy::too_many_arguments)]
    fn travel_times_from_coordinate(
        &self,
        py: Python<'_>,
        origin: (f64, f64),
        date: &str,
        departure: &str,
        max_transfers: u8,
        walking_speed_kmph: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
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
        let request = Request {
            departure,
            access: request_offsets(&access),
            egress: Vec::new(),
            active_services: self.active_services(date)?,
            active_services_previous: self.active_services_previous(date)?,
            max_transfers,
        };
        // Under a whole-day ULTRA set the intermediate transfers use the
        // shortcuts and a bounded final walk (`<= max_walking_time`) reaches
        // the remaining stops; otherwise this is the closure, tau-direct search
        // (`time_transfers` is the closure then, and the fold is skipped).
        let mut arrivals =
            Raptor.one_to_all(&self.build.timetable, self.time_transfers(), &request);
        if self.ultra_active() {
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
    #[pyo3(signature = (from_stop, date, departure, max_transfers = 7, walking_speed_kmph = 3.6, max_walking_time = 7200.0, max_snap_distance = 1600.0))]
    #[allow(clippy::too_many_arguments)]
    fn travel_times_from_stop(
        &self,
        py: Python<'_>,
        from_stop: &str,
        date: &str,
        departure: &str,
        max_transfers: u8,
        walking_speed_kmph: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
    ) -> PyResult<Py<PyDict>> {
        let origin = self.resolve_stop(from_stop)?;
        let departure = parse_time(departure)?;
        // With a whole-day ULTRA set, treat the origin stop as its coordinate
        // and reach every stop door-to-door (coordinate access, ULTRA
        // intermediate transfers, one final walk bounded by max_walking_time);
        // otherwise board at the origin stop and relax the closure (today's
        // behaviour).
        if self.ultra_active() {
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
        };
        let arrivals = Raptor.one_to_all(&self.build.timetable, &self.transfers, &request);
        self.arrivals_dict(py, &arrivals, departure)
    }

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
    /// router : str (optional, default: "raptor")
    ///     The routing engine: ``"raptor"``, or ``"tbtr"`` to build a
    ///     TBTR day engine (view + reduced trip-transfer set) for the
    ///     date and fan the origins out over it. The results are
    ///     identical; TBTR trades a per-date precompute for faster
    ///     scans. The precomputed set covers same-stop transfers;
    ///     installed footpaths relax at query time, RAPTOR-style.
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
    #[pyo3(signature = (from_stops, date, departure, max_transfers = 7, router = "raptor", walking_speed_kmph = 3.6, max_walking_time = 7200.0, max_snap_distance = 1600.0))]
    #[allow(clippy::too_many_arguments)]
    fn travel_time_matrix<'py>(
        &self,
        py: Python<'py>,
        from_stops: Vec<String>,
        date: &str,
        departure: &str,
        max_transfers: u8,
        router: &str,
        walking_speed_kmph: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
    ) -> PyResult<Bound<'py, PyArray2<u32>>> {
        if !matches!(router, "raptor" | "tbtr") {
            return Err(PyValueError::new_err(format!(
                "router must be 'raptor' or 'tbtr', not {router:?}"
            )));
        }
        let origins: Vec<StopIdx> = from_stops
            .iter()
            .map(|stop| self.resolve_stop(stop))
            .collect::<PyResult<_>>()?;
        let departure = parse_time(departure)?;
        let active_services = self.active_services(date)?;
        let active_services_previous = self.active_services_previous(date)?;
        let stop_count = self.build.timetable.stop_count() as usize;
        let count = origins.len();
        // The RAPTOR router routes door-to-door under a whole-day ULTRA set,
        // but only for origins that snap; validate the walking speed up front
        // (error creation needs the GIL that `allow_threads` releases) and only
        // when at least one origin is usable — a matrix whose origins all fall
        // back ignores the walking options, as `travel_times_from_stop` does.
        let ultra_usable = router == "raptor"
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
        let flat: Vec<u32> = py.allow_threads(|| {
            let rows: Vec<Vec<Option<u32>>> = if router == "tbtr" {
                // Reuse the cached transfer set when it was precomputed for
                // this date (`compute_tbtr_transfers`), borrowed by the engine,
                // vs rebuilding the dominance-aware set. Otherwise build ad hoc.
                let cached = self
                    .tbtr_time_transfers
                    .as_ref()
                    .filter(|(cached_date, _)| cached_date.as_str() == date)
                    .map(|(_, set)| set);
                let engine = match cached {
                    Some(set) => TbtrEngine::from_set(
                        &self.build.timetable,
                        &self.transfers,
                        &active_services,
                        &active_services_previous,
                        set,
                    ),
                    None => TbtrEngine::for_date(
                        &self.build.timetable,
                        &self.transfers,
                        &active_services,
                        &active_services_previous,
                    ),
                };
                let accesses: Vec<Vec<(StopIdx, u32)>> =
                    origins.iter().map(|&origin| vec![(origin, 0)]).collect();
                engine.one_to_all_many(departure, &accesses, max_transfers)
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
                    })
                    .collect();
                Raptor.one_to_all_many(&self.build.timetable, &self.transfers, &requests)
            };
            let mut flat = Vec::with_capacity(count * stop_count);
            for row in rows {
                flat.extend(row.into_iter().map(|arrival| match arrival {
                    Some(arrival) => arrival - departure,
                    None => u32::MAX,
                }));
            }
            flat
        });
        flat.into_pyarray(py)
            .reshape([count, stop_count])
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

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
    /// ``router`` picks the engine: McRAPTOR (``"raptor"``, the
    /// default) answers immediately; McTBTR (``"tbtr"``) precomputes
    /// the day's multicriteria transfer set first — slower for one
    /// pair, built for batch reuse — and returns the same journeys.
    ///
    /// Returns
    /// -------
    /// list of dict
    ///     Journeys shaped as in ``route_between_stops``.
    #[pyo3(signature = (from_stop, to_stop, date, departure, factors, window = None, max_transfers = 7, bucket = 25.0, router = "raptor", walking_speed_kmph = 3.6, max_walking_time = 7200.0, max_snap_distance = 1600.0, geometries = true, slack = 0.0, max_options = None, banned_routes = vec![], route_penalties = vec![]))]
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
    ) -> PyResult<Py<PyList>> {
        if !bucket.is_finite() || bucket <= 0.0 {
            return Err(PyValueError::new_err(
                "bucket must be a positive number of grams",
            ));
        }
        if router != "raptor" && router != "tbtr" {
            return Err(PyValueError::new_err("router must be 'raptor' or 'tbtr'"));
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
        // searches; TBTR and a factor mismatch keep today's board-at-origin
        // closure routing.
        if router == "raptor"
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
        };
        let penalty_mask = self.route_penalty_mask(&banned_routes, &route_penalties);
        let journeys = py.allow_threads(|| {
            if router == "tbtr" {
                let engine = McTbtrEngine::for_date(
                    &self.build.timetable,
                    &self.transfers,
                    geometry,
                    &per_trip,
                    &request.active_services,
                    &request.active_services_previous,
                );
                return match window {
                    None => engine.route(&request, bucket),
                    Some(window) => engine.route_range(&request, window, bucket),
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
    #[pyo3(signature = (origin, destination, date, departure, factors, window = None, max_transfers = 7, bucket = 25.0, walking_speed_kmph = 3.6, max_walking_time = 7200.0, max_snap_distance = 1600.0, geometries = true, slack = 0.0, max_options = None, banned_routes = vec![], route_penalties = vec![]))]
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
    ) -> PyResult<Py<PyList>> {
        if !bucket.is_finite() || bucket <= 0.0 {
            return Err(PyValueError::new_err(
                "bucket must be a positive number of grams",
            ));
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
        // access/egress and direct walk above stay unchanged.
        let intermediate = self.emissions_transfers(factor_fingerprint(&per_trip));
        let slack = slack.round() as u32;
        let penalty_mask = self.route_penalty_mask(&banned_routes, &route_penalties);
        let journeys = py.allow_threads(|| {
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
        if let Some(walk) = direct {
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
    /// router : str (optional, default: "raptor")
    ///     ``"raptor"``, or ``"tbtr"`` to answer the window over a TBTR
    ///     day engine — the same reduced trip-transfer set the
    ///     single-departure matrix uses (reusing the cached set from
    ///     ``compute_tbtr_transfers`` when present). The results are
    ///     identical.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     A ``(len(from_stops), stop_count, len(percentiles))`` uint32
    ///     array of travel times in seconds; unreachable percentiles
    ///     hold the maximum uint32 value (4294967295).
    #[pyo3(signature = (from_stops, date, departure, window, percentiles, max_transfers = 7, router = "raptor"))]
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
    ) -> PyResult<Bound<'py, PyArray3<u32>>> {
        validate_window(window, &percentiles)?;
        if !matches!(router, "raptor" | "tbtr") {
            return Err(PyValueError::new_err(format!(
                "router must be 'raptor' or 'tbtr', not {router:?}"
            )));
        }
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
            })
            .collect();
        let stop_count = self.build.timetable.stop_count() as usize;
        let flat: Vec<u32> = py.allow_threads(|| {
            if router == "tbtr" {
                // Reuse the cached transfer set for this date when present,
                // as the single-departure TBTR matrix does; else build ad hoc.
                let cached = self
                    .tbtr_time_transfers
                    .as_ref()
                    .filter(|(cached_date, _)| cached_date.as_str() == date)
                    .map(|(_, set)| set);
                let engine = match cached {
                    Some(set) => TbtrEngine::from_set(
                        &self.build.timetable,
                        &self.transfers,
                        &active_services,
                        &active_services_previous,
                        set,
                    ),
                    None => TbtrEngine::for_date(
                        &self.build.timetable,
                        &self.transfers,
                        &active_services,
                        &active_services_previous,
                    ),
                };
                engine
                    .percentile_matrix(&requests, window, &percentiles)
                    .concat()
            } else {
                Raptor
                    .percentile_matrix(
                        &self.build.timetable,
                        &self.transfers,
                        &requests,
                        window,
                        &percentiles,
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
    #[pyo3(signature = (origins, destinations, date, departure, window, percentiles, max_transfers = 7, walking_speed_kmph = 3.6, max_walking_time = 7200.0, max_snap_distance = 1600.0))]
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
        walking_speed_kmph: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
    ) -> PyResult<Py<PyDict>> {
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
                })
                .collect();
            let egress = egress_tables(&destination_links);
            let mut flat = Raptor
                .percentile_matrix_to_points(
                    &self.build.timetable,
                    self.time_transfers(),
                    &requests,
                    &egress,
                    window,
                    &percentiles,
                )
                .concat();
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
    #[pyo3(signature = (from_stops, date, departure, factors, max_transfers = 7, to_stops = None, walking_speed_kmph = 3.6, max_walking_time = 7200.0, max_snap_distance = 1600.0, geometries = false, fares = None))]
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
        walking_speed_kmph: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
        geometries: bool,
        fares: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDict>> {
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
        let ultra_usable = self.ultra_active()
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
                Raptor.cost_matrix(
                    &self.build.timetable,
                    &self.transfers,
                    &inputs,
                    &requests,
                    &destinations,
                )
            }
        });
        cost_rows_dict(py, rows, geometries)
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
    #[pyo3(signature = (origins, destinations, date, departure, max_transfers = 7, walking_speed_kmph = 3.6, max_walking_time = 7200.0, max_snap_distance = 1600.0))]
    #[allow(clippy::too_many_arguments)]
    fn travel_time_matrix_from_points(
        &self,
        py: Python<'_>,
        origins: Vec<(f64, f64)>,
        destinations: Vec<(f64, f64)>,
        date: &str,
        departure: &str,
        max_transfers: u8,
        walking_speed_kmph: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
    ) -> PyResult<Py<PyDict>> {
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
                })
                .collect();
            let egress = egress_tables(&destination_links);
            let rows =
                Raptor.one_to_all_many(&self.build.timetable, self.time_transfers(), &requests);
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

    /// The fastest journey's aggregated costs between coordinate
    /// points, long format — ``travel_cost_matrix`` over linked points.
    ///
    /// Points link once against the street network; a destination's
    /// travel time is the minimum over its links of the arrival plus
    /// the egress walk, and its costs are the winning journey's, with
    /// the access and egress walks counted in ``walk_distance``.
    /// Requires an installed street network and trip distances.
    ///
    /// Returns
    /// -------
    /// dict
    ///     As ``travel_cost_matrix`` — ``from`` and ``to`` index the
    ///     origin and destination point lists — plus
    ///     ``unsnapped_from`` / ``unsnapped_to`` with the indices of
    ///     points off the walking network.
    #[pyo3(signature = (origins, destinations, date, departure, factors, max_transfers = 7, walking_speed_kmph = 3.6, max_walking_time = 7200.0, max_snap_distance = 1600.0, geometries = false, fares = None))]
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
        walking_speed_kmph: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
        geometries: bool,
        fares: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDict>> {
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
            let mut rows = Raptor.cost_matrix_to_points(
                &self.build.timetable,
                self.time_transfers(),
                &inputs,
                &requests,
                &access_meters,
                &egress,
            );
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
    #[pyo3(signature = (from_stops, date, departure, window, factors, objective = "emissions", fares = None, budget = None, max_transfers = 7, to_stops = None, candidates = "time", bucket = 25.0, router = "raptor", walking_speed_kmph = 3.6, max_walking_time = 7200.0, max_snap_distance = 1600.0, geometries = false))]
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
        if router != "raptor" && router != "tbtr" {
            return Err(PyValueError::new_err("router must be 'raptor' or 'tbtr'"));
        }
        if router == "tbtr" && candidates != "pareto" {
            return Err(PyValueError::new_err(
                "router='tbtr' requires candidates='pareto'",
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
        // network) it keeps the closure and board-at-origin access; TBTR and the
        // time objective always keep the closure.
        let stop_count = self.build.timetable.stop_count() as usize;
        let fingerprint = factor_fingerprint(&per_trip);
        let matrix_mcultra = candidates == "pareto"
            && router == "raptor"
            && !std::ptr::eq(self.emissions_transfers(fingerprint), &self.transfers)
            && self.streets.is_some();
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
                let engine = McTbtrEngine::for_date(
                    &self.build.timetable,
                    &self.transfers,
                    geometry,
                    &per_trip,
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
    #[pyo3(signature = (origins, destinations, date, departure, window, factors, objective = "emissions", fares = None, budget = None, max_transfers = 7, walking_speed_kmph = 3.6, max_walking_time = 7200.0, max_snap_distance = 1600.0, geometries = false))]
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
        walking_speed_kmph: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
        geometries: bool,
    ) -> PyResult<Py<PyDict>> {
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
            let mut rows = Raptor.least_cost_matrix_to_points(
                &self.build.timetable,
                &self.transfers,
                &inputs,
                &requests,
                &access_meters,
                &egress,
                window,
                budget,
                objective,
            );
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
    fn time_transfers(&self) -> &Transfers {
        match (self.ultra_transfers.as_ref(), self.ultra_window) {
            (Some(ultra), Some((0, hi))) if hi >= u32::MAX - 1 => ultra,
            _ => &self.transfers,
        }
    }

    /// Whether a whole-day ULTRA set is installed, i.e. `time_transfers`
    /// returns it — the gate for door-to-door stop routing.
    fn ultra_active(&self) -> bool {
        matches!(
            (self.ultra_transfers.as_ref(), self.ultra_window),
            (Some(_), Some((0, hi))) if hi >= u32::MAX - 1
        )
    }

    /// The transfer set the coordinate emissions engines relax for a query using
    /// factors with fingerprint `factor`: the whole-day McULTRA set when one is
    /// installed for that exact factor configuration, else the closure. A
    /// partial-window or factor-mismatched set is never silently used (§Factor
    /// contract).
    fn emissions_transfers(&self, factor: u64) -> &Transfers {
        match (
            self.mcultra_transfers.as_ref(),
            self.mcultra_window,
            self.mcultra_factor,
        ) {
            (Some(set), Some((0, hi)), Some(built)) if hi >= u32::MAX - 1 && built == factor => set,
            _ => &self.transfers,
        }
    }

    /// A stop's `(latitude, longitude)`, or `None` when the feed omits it.
    fn stop_coordinate(&self, stop: StopIdx) -> Option<(f64, f64)> {
        let stop = &self.feed.stops[stop.0 as usize];
        Some((stop.latitude?, stop.longitude?))
    }

    /// The per-destination final-walk egress for the one-to-all time queries:
    /// a bounded (`max_walking_time`) `link_many` from every stop's coordinate,
    /// giving `egress[t]` the stops within a final walk of `t` and their walk
    /// seconds. The same bounded construction the coordinate query and the cost
    /// matrix use — walking is undirected, so a search from `t` yields the
    /// `s -> t` egress. Built once per query and reused across a matrix's
    /// origins. `None` means the stop has no coordinate (it cannot be located,
    /// so it keeps its bare transit arrival); `Some(list)` treats the stop's
    /// coordinate as the destination — an empty list is a located-but-unreachable
    /// coordinate (its connector exceeds the cap), which gets no arrival, exactly
    /// as `route_between_coordinates` would refuse it.
    fn final_egress(
        &self,
        streets: &StreetNetwork,
        speed: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
    ) -> Vec<Option<Vec<(StopIdx, u32)>>> {
        let stop_count = self.build.timetable.stop_count() as usize;
        let mut coordinates: Vec<(f64, f64)> = Vec::with_capacity(stop_count);
        let mut slots: Vec<Option<usize>> = Vec::with_capacity(stop_count);
        for index in 0..stop_count {
            match self.stop_coordinate(StopIdx(index as u32)) {
                Some(coordinate) => {
                    slots.push(Some(coordinates.len()));
                    coordinates.push(coordinate);
                }
                None => slots.push(None),
            }
        }
        let links = streets.link_many(&coordinates, speed, max_walking_time, max_snap_distance);
        (0..stop_count)
            .map(|index| {
                slots[index].map(|slot| {
                    let mut sources = Vec::new();
                    if let Some(reached) = &links[slot] {
                        sources.extend(reached.iter().map(|walk| (walk.stop, walk.seconds)));
                    }
                    sources
                })
            })
            .collect()
    }

    /// The street egress for the emissions cost matrix, keyed by source stop:
    /// `map[s]` lists `(destination slot, walk seconds, walk meters)` for every
    /// matrix destination reachable by a final walk off stop `s` — the reverse
    /// of `final_egress`, carrying metres for the reported walk distance. A
    /// destination without a coordinate (or unreachable within
    /// `max_walking_time`) contributes no sources, so it is reached only by
    /// alighting there directly.
    fn matrix_street_egress(
        &self,
        streets: &StreetNetwork,
        destinations: &[StopIdx],
        speed: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
    ) -> Vec<Vec<(u32, u32, f64)>> {
        let stop_count = self.build.timetable.stop_count() as usize;
        let mut coordinates: Vec<(f64, f64)> = Vec::with_capacity(destinations.len());
        let mut slot_of: Vec<u32> = Vec::with_capacity(destinations.len());
        for (slot, &destination) in destinations.iter().enumerate() {
            if let Some(coordinate) = self.stop_coordinate(destination) {
                slot_of.push(slot as u32);
                coordinates.push(coordinate);
            }
        }
        let links = streets.link_many(&coordinates, speed, max_walking_time, max_snap_distance);
        let mut map = vec![Vec::new(); stop_count];
        for (index, reached) in links.iter().enumerate() {
            if let Some(reached) = reached {
                let slot = slot_of[index];
                for walk in reached {
                    map[walk.stop.0 as usize].push((slot, walk.seconds, walk.meters));
                }
            }
        }
        // A destination with no coordinate cannot be a door-to-door coordinate,
        // so it is reachable only by a direct alight — give it a bare zero-walk
        // self-entry. A located destination is left to its `link_many` connector:
        // if its coordinate does not snap or lies beyond the cap it carries no
        // entry and is simply unreachable, exactly as the single-pair coordinate
        // route would refuse it (rather than crediting it as a free alight).
        for (slot, &destination) in destinations.iter().enumerate() {
            if self.stop_coordinate(destination).is_none() {
                map[destination.0 as usize].push((slot as u32, 0, 0.0));
            }
        }
        map
    }

    /// The location-based access for the emissions cost matrix, one entry per
    /// origin: the stops within an initial walk of the origin's coordinate — its
    /// own connector included — as `(stop, walk seconds, walk meters)`, plus a
    /// `located` flag. A coordinate that **snaps** is `located` (`true`) even
    /// when no stop is reachable within the cap — its access is then empty (no
    /// transit boarding), but it stays on the door-to-door path so its
    /// direct-walk overlay still applies. Only a missing coordinate or a failed
    /// snap gives the board-at-origin fallback `[(origin, 0, 0)]` with `false`,
    /// routing that origin over the closure rather than the intermediate-only
    /// set. The initial-walk analogue of `matrix_street_egress`; the metres are
    /// threaded into the reported walk distance.
    #[allow(clippy::type_complexity)]
    fn matrix_location_access(
        &self,
        streets: &StreetNetwork,
        origins: &[StopIdx],
        speed: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
    ) -> (Vec<Vec<(StopIdx, u32, f64)>>, Vec<bool>) {
        let mut coordinates: Vec<(f64, f64)> = Vec::with_capacity(origins.len());
        let mut coordinate_of: Vec<Option<usize>> = Vec::with_capacity(origins.len());
        for &origin in origins {
            match self.stop_coordinate(origin) {
                Some(coordinate) => {
                    coordinate_of.push(Some(coordinates.len()));
                    coordinates.push(coordinate);
                }
                None => coordinate_of.push(None),
            }
        }
        let links = streets.link_many(&coordinates, speed, max_walking_time, max_snap_distance);
        origins
            .iter()
            .zip(coordinate_of)
            .map(
                |(&origin, slot)| match slot.and_then(|slot| links[slot].as_ref()) {
                    // Snapped — located even when no stop is reachable within the
                    // cap: it still takes the direct-walk overlay, an empty access
                    // just means no transit boarding. Only a missing coordinate or
                    // a failed snap falls back to closure board-at-origin routing.
                    Some(reached) => (
                        reached
                            .iter()
                            .map(|walk| (walk.stop, walk.seconds, walk.meters))
                            .collect(),
                        true,
                    ),
                    None => (vec![(origin, 0, 0.0)], false),
                },
            )
            .unzip()
    }

    /// The explicit coordinate-to-coordinate direct street walks for the
    /// emissions cost matrix, per origin: `(destination slot, walk seconds, walk
    /// metres)` for every destination the origin's coordinate reaches on foot
    /// within the cap. Built from `walk_matrix`, which snaps both coordinates,
    /// zeroes the same-coordinate diagonal, and returns nothing for a coordinate
    /// that does not snap — so a cell matches the single-pair route's direct
    /// walk rather than inferring one from stop connectors. A stop with no
    /// coordinate contributes and receives no walk.
    fn matrix_direct_walks(
        &self,
        streets: &StreetNetwork,
        origins: &[StopIdx],
        destinations: &[StopIdx],
        speed: f64,
        max_walking_time: f64,
        max_snap_distance: f64,
    ) -> Vec<Vec<(u32, u32, f64)>> {
        let mut origin_coords: Vec<(f64, f64)> = Vec::new();
        let mut origin_row: Vec<Option<usize>> = Vec::with_capacity(origins.len());
        for &origin in origins {
            match self.stop_coordinate(origin) {
                Some(coordinate) => {
                    origin_row.push(Some(origin_coords.len()));
                    origin_coords.push(coordinate);
                }
                None => origin_row.push(None),
            }
        }
        let mut dest_coords: Vec<(f64, f64)> = Vec::new();
        let mut dest_col: Vec<Option<usize>> = Vec::with_capacity(destinations.len());
        for &destination in destinations {
            match self.stop_coordinate(destination) {
                Some(coordinate) => {
                    dest_col.push(Some(dest_coords.len()));
                    dest_coords.push(coordinate);
                }
                None => dest_col.push(None),
            }
        }
        let walk = streets.walk_matrix(
            &origin_coords,
            &dest_coords,
            speed,
            max_walking_time,
            max_snap_distance,
        );
        origin_row
            .iter()
            .map(|&row| match row {
                None => Vec::new(),
                Some(row) => dest_col
                    .iter()
                    .enumerate()
                    .filter_map(|(slot, &col)| {
                        let (seconds, meters) = walk[row][col?]?;
                        Some((slot as u32, seconds, meters))
                    })
                    .collect(),
            })
            .collect()
    }

    /// Folds one **bounded** final walk into a one-to-all arrival array,
    /// location-based: each *located* target stop's arrival becomes the earliest
    /// `arrival[source] + walk(source -> target)` over the sources in
    /// `egress[target]` (within `max_walking_time`). The egress is
    /// `link_many(target.coordinate)`, which includes the target itself via its
    /// connector, so a transit-reached target keeps its own arrival plus that
    /// connector — the arrival *at the stop's coordinate*, matching
    /// `route_between_coordinates`. `egress[target] == None` (no coordinate)
    /// keeps the bare RAPTOR arrival; `Some(empty)` (coordinate unreachable
    /// within `max_walking_time`) yields no arrival, as the coordinate query
    /// would. A single hop over a snapshot of the RAPTOR arrivals, so a final
    /// walk never chains. Used by the one-to-all time queries under a whole-day
    /// ULTRA set.
    fn fold_final_transfers(
        &self,
        arrivals: &mut [Option<u32>],
        egress: &[Option<Vec<(StopIdx, u32)>>],
    ) {
        let reached: Vec<Option<u32>> = arrivals.to_vec();
        for (target, entry) in egress.iter().enumerate() {
            let Some(sources) = entry else {
                continue; // no coordinate: keep the bare transit arrival
            };
            let mut best = None;
            for &(source, walk) in sources {
                let Some(at) = reached[source.0 as usize] else {
                    continue;
                };
                if let Some(candidate) = at.checked_add(walk).filter(|&at| at != u32::MAX) {
                    best = Some(best.map_or(candidate, |current: u32| current.min(candidate)));
                }
            }
            arrivals[target] = best;
        }
    }

    /// A one-to-all arrival array as a `{public_stop_id: travel_time}` dict,
    /// travel time measured from `departure`; unreachable stops are absent.
    fn arrivals_dict(
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
    fn ultra_matrix_rows(
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
            let mut usable_rows =
                Raptor.one_to_all_many(&self.build.timetable, self.time_transfers(), &requests);
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
            let fallback_rows =
                Raptor.one_to_all_many(&self.build.timetable, &self.transfers, &requests);
            for (row, &(index, _)) in fallback_rows.into_iter().zip(&fallback) {
                rows[index] = row;
            }
        }
        rows
    }

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
    fn ultra_cost_matrix_rows(
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

    /// The installed street network, or a `ValueError` explaining that
    /// coordinate queries need one.
    fn installed_streets(&self) -> PyResult<&StreetNetwork> {
        self.streets.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "no street network installed; build the network with an OSM extract",
            )
        })
    }

    /// The service-activity flags of a `YYYY-MM-DD` date.
    fn active_services(&self, date: &str) -> PyResult<Vec<bool>> {
        let date = NaiveDate::parse_from_str(date, "%Y-%m-%d")
            .map_err(|error| PyValueError::new_err(format!("invalid date '{date}': {error}")))?;
        Ok(self.build.services.active_on(date))
    }

    /// The services running the day before `date`, whose over-midnight
    /// trips reach into it.
    fn active_services_previous(&self, date: &str) -> PyResult<Vec<bool>> {
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
    fn route_request(
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
    fn walk_journey_dict(
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
    fn stop_walk_endpoint(&self, stop: StopIdx) -> Option<((f64, f64), Snap)> {
        let streets = self.streets.as_ref()?;
        let snap = streets.stop_snap(stop)?;
        let feed_stop = &self.feed.stops[stop.0 as usize];
        Some(((feed_stop.latitude?, feed_stop.longitude?), snap))
    }

    /// The walked street path between two snapped points, as WKB.
    fn walk_wkb<'py>(
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
    fn public_stop_id(&self, stop: StopIdx) -> String {
        let stop = &self.feed.stops[stop.0 as usize];
        self.public_id(stop.feed, &stop.id)
    }

    fn public_id(&self, feed: cafein_gtfs::FeedIndex, id: &str) -> String {
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
    fn route_penalty_mask(
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
    fn resolve_stop(&self, stop_id: &str) -> PyResult<StopIdx> {
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

    fn journey_to_dict(
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

/// The walking speed in m/s of validated street-query parameters, or a
/// `ValueError` naming the parameter that is out of range.
fn validated_walking_speed(
    walking_speed_kmph: f64,
    max_walking_time: f64,
    max_snap_distance: f64,
) -> PyResult<f64> {
    if !walking_speed_kmph.is_finite() || walking_speed_kmph <= 0.0 {
        return Err(PyValueError::new_err(
            "walking_speed_kmph must be a positive, finite number",
        ));
    }
    if !max_walking_time.is_finite() || max_walking_time < 0.0 {
        return Err(PyValueError::new_err(
            "max_walking_time must be a non-negative, finite number",
        ));
    }
    if !max_snap_distance.is_finite() || max_snap_distance < 0.0 {
        return Err(PyValueError::new_err(
            "max_snap_distance must be a non-negative, finite number",
        ));
    }
    Ok(walking_speed_kmph / 3.6)
}

/// The stops walkable from a coordinate, or a `ValueError` when the
/// coordinate is invalid or off the network; `side` prefixes the message
/// (e.g. `"origin "`) to name the endpoint.
fn coordinate_links(
    streets: &StreetNetwork,
    coordinate: (f64, f64),
    walking_speed: f64,
    max_walking_time: f64,
    max_snap_distance: f64,
    side: &str,
) -> PyResult<Vec<WalkedStop>> {
    let (lat, lon) = coordinate;
    if !lat.is_finite() || !lon.is_finite() {
        return Err(PyValueError::new_err(format!(
            "{side}lat and lon must be finite"
        )));
    }
    streets
        .access_stops(lat, lon, walking_speed, max_walking_time, max_snap_distance)
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "{side}({lat}, {lon}) is farther than {max_snap_distance} m \
                 from the walking network"
            ))
        })
}

/// Encodes coordinates as a little-endian WKB LineString (XY).
fn wkb_line_string<'py>(py: Python<'py>, coordinates: &[(f64, f64)]) -> Bound<'py, PyBytes> {
    PyBytes::new(py, &cafein_core::geometry::wkb_line_string(coordinates))
}

/// Parses ``HH:MM:SS`` into seconds past the service day's start; hours may
/// exceed 23 for over-midnight times, following GTFS.
fn parse_time(value: &str) -> PyResult<u32> {
    let parts: Vec<&str> = value.split(':').collect();
    let invalid = || PyValueError::new_err(format!("invalid time '{value}': expected HH:MM:SS"));
    if parts.len() != 3 {
        return Err(invalid());
    }
    let hours: u32 = parts[0].parse().map_err(|_| invalid())?;
    let minutes: u32 = parts[1].parse().map_err(|_| invalid())?;
    let seconds: u32 = parts[2].parse().map_err(|_| invalid())?;
    if minutes > 59 || seconds > 59 {
        return Err(invalid());
    }
    hours
        .checked_mul(3600)
        .and_then(|in_seconds| in_seconds.checked_add(minutes * 60 + seconds))
        .ok_or_else(invalid)
}

fn to_py_error(error: cafein_gtfs::Error) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// The numeric GTFS route_type of a parsed route type; named variants map
/// to their standard codes, extended codes pass through.
fn route_type_code(route_type: &RouteType) -> i32 {
    match route_type {
        RouteType::Tramway => 0,
        RouteType::Subway => 1,
        RouteType::Rail => 2,
        RouteType::Bus => 3,
        RouteType::Ferry => 4,
        RouteType::CableCar => 5,
        RouteType::Gondola => 6,
        RouteType::Funicular => 7,
        RouteType::Coach => 200,
        RouteType::Air => 1100,
        RouteType::Taxi => 1500,
        RouteType::Other(code) => *code as i32,
    }
}

fn provenance_name(tier: DistanceProvenance) -> &'static str {
    match tier {
        DistanceProvenance::ShapeDist => "shape_dist",
        DistanceProvenance::ShapeLinRef => "shape_linref",
        DistanceProvenance::OsmRelation => "osm_relation",
        DistanceProvenance::MapMatched => "map_matched",
        DistanceProvenance::CrowFly => "crow_fly",
    }
}

fn parse_provenance(value: &str) -> PyResult<DistanceProvenance> {
    match value {
        "shape_dist" => Ok(DistanceProvenance::ShapeDist),
        "shape_linref" => Ok(DistanceProvenance::ShapeLinRef),
        "osm_relation" => Ok(DistanceProvenance::OsmRelation),
        "map_matched" => Ok(DistanceProvenance::MapMatched),
        "crow_fly" => Ok(DistanceProvenance::CrowFly),
        other => Err(PyValueError::new_err(format!(
            "unknown distance provenance '{other}'"
        ))),
    }
}

#[pymodule]
fn _cafein(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<TransportNetwork>()?;
    Ok(())
}
