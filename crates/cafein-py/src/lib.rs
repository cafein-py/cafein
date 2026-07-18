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
    /// The cached multicriteria TBTR transfer set with the date and the
    /// factor fingerprint it was built for (`compute_mctbtr_transfers`).
    mctbtr_transfers: Option<(String, u64, cafein_core::tbtr::TransferSet)>,
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
// 10: the persisted time TBTR transfer set became tie-complete (same-ride
// equal-arrival competitors retained for cost-row reconstruction); sets
// written by earlier formats lack the competitors and must be rebuilt.
const ARTIFACT_FORMAT: u32 = 10;
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
    /// The cached multicriteria TBTR transfer set with its date and factor
    /// fingerprint, when present.
    mctbtr_transfers: &'a Option<(String, u64, cafein_core::tbtr::TransferSet)>,
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
    mctbtr_transfers: Option<(String, u64, cafein_core::tbtr::TransferSet)>,
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

/// The six batched door-to-door frontier parts `point_frontier_rows`
/// hands both the journey and table forms.
type PointFrontierParts = (
    Vec<Vec<Vec<Journey>>>,
    Vec<Vec<Option<(u32, f64)>>>,
    Vec<Option<Vec<WalkedStop>>>,
    Vec<Option<Vec<WalkedStop>>>,
    Vec<u32>,
    Vec<u32>,
);

/// One frontier-table row of a journey: (departure, arrival, rides,
/// emissions). Emissions mirror `cafein.emissions.annotate`: walking
/// legs contribute nothing, each transit leg contributes its
/// leg-distance in kilometres times the trip's factor, summed in leg
/// order, and any unresolved transit factor turns the journey's total
/// into NaN.
fn frontier_row(
    geometry: &TripGeometry,
    per_trip: &[f64],
    journey: &Journey,
) -> (u32, u32, u32, f64) {
    let mut total = 0.0;
    let mut complete = true;
    for leg in &journey.legs {
        if let Leg::Transit {
            trip,
            board_position,
            alight_position,
            ..
        } = leg
        {
            let factor = per_trip[trip.0 as usize];
            if factor.is_nan() {
                complete = false;
            } else {
                let distance =
                    geometry.leg_distance(*trip, *board_position, *alight_position) as f64;
                total += distance / 1000.0 * factor;
            }
        }
    }
    (
        journey.departure,
        journey.arrival,
        journey.rides() as u32,
        if complete { total } else { f64::NAN },
    )
}

/// The flat columns of a batched frontier table, filled cell by cell in
/// the requested (origin, destination) order.
#[derive(Default)]
struct FrontierColumns {
    from_index: Vec<u32>,
    to_index: Vec<u32>,
    departure: Vec<u32>,
    arrival: Vec<u32>,
    travel_time: Vec<u32>,
    rides: Vec<u32>,
    emissions: Vec<f64>,
    frontier: Vec<bool>,
}

impl FrontierColumns {
    /// Appends one cell's rows, sorted and Pareto-marked exactly as the
    /// journey frame: a stable sort by (travel_time, emissions) with
    /// NaN emissions last within equal travel times, and the frontier
    /// mask over (travel_time, emissions) where a NaN row never joins
    /// the frontier and never dominates.
    fn push_cell(&mut self, from: u32, to: u32, mut rows: Vec<(u32, u32, u32, f64)>) {
        rows.sort_by(|a, b| {
            let time = (a.1 - a.0).cmp(&(b.1 - b.0));
            time.then_with(|| match (a.3.is_nan(), b.3.is_nan()) {
                (false, false) => a.3.partial_cmp(&b.3).expect("no NaN"),
                (true, true) => std::cmp::Ordering::Equal,
                (true, false) => std::cmp::Ordering::Greater,
                (false, true) => std::cmp::Ordering::Less,
            })
        });
        for (i, &(departure, arrival, rides, grams)) in rows.iter().enumerate() {
            let time = arrival - departure;
            let on = !grams.is_nan()
                && !rows.iter().enumerate().any(|(j, &(dj, aj, _, gj))| {
                    j != i && !gj.is_nan() && {
                        let tj = aj - dj;
                        tj <= time && gj <= grams && (tj < time || gj < grams)
                    }
                });
            self.from_index.push(from);
            self.to_index.push(to);
            self.departure.push(departure);
            self.arrival.push(arrival);
            self.travel_time.push(time);
            self.rides.push(rides);
            self.emissions.push(grams);
            self.frontier.push(on);
        }
    }

    fn into_dict(self, py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("from_index", self.from_index.into_pyarray(py))?;
        dict.set_item("to_index", self.to_index.into_pyarray(py))?;
        dict.set_item("departure", self.departure.into_pyarray(py))?;
        dict.set_item("arrival", self.arrival.into_pyarray(py))?;
        dict.set_item("travel_time", self.travel_time.into_pyarray(py))?;
        dict.set_item("rides", self.rides.into_pyarray(py))?;
        dict.set_item("emissions", self.emissions.into_pyarray(py))?;
        dict.set_item("frontier", self.frontier.into_pyarray(py))?;
        Ok(dict)
    }
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
fn walk_within_band(
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

/// The error every routing entry raises for an unknown `router` value.
fn invalid_router(router: &str) -> PyErr {
    PyValueError::new_err(format!(
        "router must be 'auto', 'raptor', or 'tbtr', not {router:?}"
    ))
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

mod artifact;
mod cost_matrices;
mod frontiers;
mod network;
mod options;
mod points;
mod routes;
mod time_matrices;

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
