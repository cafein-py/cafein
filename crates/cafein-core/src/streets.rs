//! The query-time street network for walking access and egress.
//!
//! The Python-side build hands the walking graph over as flat arrays:
//! vertices are implicit indices, edges carry their cost length in meters
//! and their geometry. A query snaps a coordinate to its nearest edge
//! through a packed static index over the edge segments — a virtual node at
//! the snap point, with the split edge's cost pro-rated by the fraction each
//! side covers — and runs a cutoff-bounded Dijkstra to collect every transit
//! stop reachable on foot, entering stops through the same snap links the
//! footpath precompute used. Walking is undirected, so one search serves
//! access and egress alike.
//!
//! Edges and vertices are renumbered along a Hilbert curve at build time, so
//! spatially-nearby streets are nearby in every edge-indexed array — the ids
//! are internal, and only exactly-equal snap or cost ties can resolve
//! differently than under the input order.
//!
//! Geometry is stored in geographic coordinates (longitude/latitude); there
//! is no global projection. Distances use a local `cos(latitude)` evaluated
//! at the point's own latitude, so they stay accurate over country-scale
//! latitude ranges (following R5/OpenTripPlanner). Segments are densified to
//! a maximum length at build time, so the local-scale model is exact: the
//! snap foot, the connector distance, and the geometry lengths are all
//! computed in a frame local to the relevant short segment. The packed index
//! stores segment boxes in lon/lat and is used only for envelope queries —
//! never metric nearest-neighbour, whose degree-Euclidean distance would be
//! wrong.
//!
//! Coordinates are assumed to be a contiguous extract within a continuous
//! longitude range — the regional OSM extracts cafein consumes never cross the
//! antimeridian. Distance measurement still uses the shortest signed longitude
//! delta defensively, but the snap envelope is a single longitude interval, so
//! snapping is not supported across ±180°.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use rayon::prelude::*;

use crate::timetable::StopIdx;

/// Longest stored segment, in meters. Segments are densified below this so a
/// single centre-latitude scale represents each one to well under a
/// millimetre even at high latitude.
const MAX_SEGMENT_METERS: f64 = 100.0;

/// Headroom the densifier leaves under `MAX_SEGMENT_METERS`, so re-quantizing
/// the inserted points (≤ ~0.8 cm each) never pushes a segment over the
/// maximum.
const QUANTIZATION_GUARD_METERS: f64 = 0.05;

/// Fixed-point coordinate scale: degrees × 10⁷ stored as `i32`
/// (≈ 1.1 cm of latitude per step; ±180° fits comfortably).
const COORDINATE_SCALE: f64 = 1e7;

/// Children per packed-index node.
const INDEX_NODE_SIZE: usize = 16;

/// A fixed-point degree value from a float one, rounding to the nearest
/// grid step (ties to even).
fn quantize(degrees: f64) -> i32 {
    (degrees * COORDINATE_SCALE)
        .round_ties_even()
        .clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32
}

/// The float degree value of a fixed-point one.
fn degrees(fixed: i32) -> f64 {
    f64::from(fixed) / COORDINATE_SCALE
}

/// A lon/lat bounding box in fixed-point coordinates:
/// `[min_lon, min_lat, max_lon, max_lat]`.
type Envelope = [i32; 4];

fn envelopes_intersect(a: &Envelope, b: &Envelope) -> bool {
    a[0] <= b[2] && b[0] <= a[2] && a[1] <= b[3] && b[1] <= a[3]
}

/// A packed static spatial index over the densified edge segments: leaf
/// boxes sorted by the Hilbert position of their segment's midpoint, parent
/// levels packed bottom-up over runs of `INDEX_NODE_SIZE` children — flat
/// arrays with an implicit tree layout, built once and never mutated.
/// Queried by envelope intersection only. Boxes are exact in the
/// fixed-point grid (the coordinates are grid points), so only query
/// envelopes need outward rounding.
#[derive(Debug)]
struct PackedIndex {
    /// Node boxes: the leaves first (Hilbert order), then each parent
    /// level, the root last.
    boxes: Vec<Envelope>,
    /// Two entries per leaf — `edge index, index of the segment's first
    /// coordinate` — parallel to the leaf boxes.
    payload: Vec<u32>,
    /// Start of each level in `boxes` (leaves at 0), plus a tail.
    level_starts: Vec<u32>,
}

/// Collects the payloads of every leaf whose box intersects the envelope,
/// sorted by payload so callers see a traversal-free order. The arrays
/// are a [`PackedIndex`]'s, whichever backing they live in.
fn query_packed_index(
    boxes: &[Envelope],
    payload: &[u32],
    level_starts: &[u32],
    envelope: &Envelope,
    matches: &mut Vec<(u32, u32)>,
) {
    matches.clear();
    if payload.is_empty() {
        return;
    }
    let levels = level_starts.len() - 1;
    // (global node index, level), starting at the root.
    let mut stack = vec![(boxes.len() - 1, levels - 1)];
    while let Some((node, level)) = stack.pop() {
        if !envelopes_intersect(&boxes[node], envelope) {
            continue;
        }
        if level == 0 {
            matches.push((payload[2 * node], payload[2 * node + 1]));
            continue;
        }
        // A node's children sit in the level below, in its own run.
        let position = node - level_starts[level] as usize;
        let start = level_starts[level - 1] as usize + position * INDEX_NODE_SIZE;
        let end = (start + INDEX_NODE_SIZE).min(level_starts[level] as usize);
        for child in start..end {
            stack.push((child, level - 1));
        }
    }
    matches.sort_unstable();
}

/// Builds the packed index over a densified fixed-point polyline set.
fn build_index(coordinate_offsets: &[u32], lons: &[i32], lats: &[i32]) -> PackedIndex {
    // One item per consecutive coordinate pair, keyed by the Hilbert
    // position of its midpoint on a grid over the extract; ties broken by
    // the payload so the order is a pure function of the geometry.
    let bounds = coordinate_bounds(lons, lats);
    let mut items: Vec<(u64, (u32, u32), Envelope)> = Vec::new();
    for edge in 0..coordinate_offsets.len().saturating_sub(1) {
        let start = coordinate_offsets[edge] as usize;
        let end = coordinate_offsets[edge + 1] as usize;
        for segment in start..end - 1 {
            let (lon_a, lat_a) = (lons[segment], lats[segment]);
            let (lon_b, lat_b) = (lons[segment + 1], lats[segment + 1]);
            let key = hilbert(
                grid_position(
                    ((i64::from(lon_a) + i64::from(lon_b)) / 2) as i32,
                    bounds[0],
                    bounds[2],
                ),
                grid_position(
                    ((i64::from(lat_a) + i64::from(lat_b)) / 2) as i32,
                    bounds[1],
                    bounds[3],
                ),
            );
            items.push((
                key,
                (edge as u32, segment as u32),
                [
                    lon_a.min(lon_b),
                    lat_a.min(lat_b),
                    lon_a.max(lon_b),
                    lat_a.max(lat_b),
                ],
            ));
        }
    }
    items.sort_unstable_by_key(|&(key, payload, _)| (key, payload));

    let count = items.len();
    let level_starts = level_starts_for(count);
    if count == 0 {
        return PackedIndex {
            boxes: Vec::new(),
            payload: Vec::new(),
            level_starts,
        };
    }
    if count == 1 {
        // A single leaf is its own root: one leaf-only level.
        let (_, (edge, segment), envelope) = items[0];
        return PackedIndex {
            boxes: vec![envelope],
            payload: vec![edge, segment],
            level_starts,
        };
    }

    let total = *level_starts.last().unwrap() as usize;
    let mut boxes = Vec::with_capacity(total);
    let mut payload = Vec::with_capacity(count * 2);
    for (_, (edge, segment), envelope) in items {
        boxes.push(envelope);
        payload.push(edge);
        payload.push(segment);
    }
    for level in 1..level_starts.len() - 1 {
        let (start, end) = (
            level_starts[level - 1] as usize,
            level_starts[level] as usize,
        );
        for run in (start..end).step_by(INDEX_NODE_SIZE) {
            let mut merged = boxes[run];
            for child in &boxes[run + 1..(run + INDEX_NODE_SIZE).min(end)] {
                merged[0] = merged[0].min(child[0]);
                merged[1] = merged[1].min(child[1]);
                merged[2] = merged[2].max(child[2]);
                merged[3] = merged[3].max(child[3]);
            }
            boxes.push(merged);
        }
    }
    PackedIndex {
        boxes,
        payload,
        level_starts,
    }
}

/// The level-start offsets of a packed index with `count` leaves: each
/// level's start in the node array (leaves at 0), plus a tail holding the
/// total node count. A pure function of the leaf count, so an adopted
/// index never needs them stored.
fn level_starts_for(count: usize) -> Vec<u32> {
    let mut level_starts = vec![0u32];
    let mut total = count;
    let mut level_size = count;
    while level_size > 1 {
        level_starts.push(total as u32);
        level_size = level_size.div_ceil(INDEX_NODE_SIZE);
        total += level_size;
    }
    match count {
        0 => vec![0, 0],
        // A single leaf is its own root: one leaf-only level.
        1 => vec![0, 1],
        _ => {
            level_starts.push(total as u32);
            level_starts
        }
    }
}

/// The `[min_lon, min_lat, max_lon, max_lat]` bounds of a fixed-point
/// coordinate set.
fn coordinate_bounds(lons: &[i32], lats: &[i32]) -> Envelope {
    let mut bounds = [i32::MAX, i32::MAX, i32::MIN, i32::MIN];
    for (&lon, &lat) in lons.iter().zip(lats) {
        bounds[0] = bounds[0].min(lon);
        bounds[1] = bounds[1].min(lat);
        bounds[2] = bounds[2].max(lon);
        bounds[3] = bounds[3].max(lat);
    }
    bounds
}

/// A coordinate's cell on a 2¹⁶-wide grid over `[min, max]`.
fn grid_position(value: i32, min: i32, max: i32) -> u16 {
    if max <= min {
        return 0;
    }
    let fraction =
        (i64::from(value) - i64::from(min)) as f64 / (i64::from(max) - i64::from(min)) as f64;
    (fraction * f64::from(u16::MAX)).clamp(0.0, f64::from(u16::MAX)) as u16
}

/// A cell's position along the order-16 Hilbert curve (the classic
/// rotate-and-accumulate walk), giving spatially-nearby cells nearby
/// positions.
fn hilbert(x: u16, y: u16) -> u64 {
    const N: u64 = 1 << 16;
    let (mut x, mut y) = (u64::from(x), u64::from(y));
    let mut d: u64 = 0;
    let mut s: u64 = N / 2;
    while s > 0 {
        let rx = u64::from(x & s > 0);
        let ry = u64::from(y & s > 0);
        d += s * s * ((3 * rx) ^ ry);
        // Rotate the quadrant so the curve connects.
        if ry == 0 {
            if rx == 1 {
                x = N - 1 - x;
                y = N - 1 - y;
            }
            std::mem::swap(&mut x, &mut y);
        }
        s /= 2;
    }
    d
}

/// How a stop enters the street graph: snapped onto an edge at a fraction
/// of its cost length, over a straight connector to the snap point.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct StopLink {
    pub stop: StopIdx,
    /// Index of the edge the stop snapped onto.
    pub edge: u32,
    /// Snap position as a fraction of the edge, 0 at its `from` vertex.
    pub fraction: f64,
    /// Straight-line distance from the stop to the snap point, in meters.
    pub connector: f64,
}

/// A stop link as the network stores it: the input [`StopLink`] with its
/// edge's endpoint vertices denormalised in, so a load can rebuild the
/// vertex→link index from the links alone, without the street arrays.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct StoredLink {
    pub stop: StopIdx,
    /// Index of the edge the stop snapped onto (internal numbering).
    pub edge: u32,
    /// Snap position as a fraction of the edge, 0 at its `from` vertex.
    pub fraction: f64,
    /// Straight-line distance from the stop to the snap point, in meters.
    pub connector: f64,
    /// The edge's endpoint vertices.
    pub from: u32,
    pub to: u32,
}

/// A stop reached by the walking search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WalkedStop {
    pub stop: StopIdx,
    /// Walking time in whole seconds, rounded up.
    pub seconds: u32,
    /// The exact walked street-path length in meters, connectors
    /// included.
    pub meters: f64,
}

/// A coordinate snapped onto the street network.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Snap {
    pub edge: u32,
    /// Snap position as a fraction of the edge, 0 at its `from` vertex.
    pub fraction: f64,
    /// Straight-line distance from the coordinate to the snap point, in
    /// meters.
    pub connector: f64,
}

/// Errors raised while assembling a [`StreetNetwork`].
#[derive(Debug, PartialEq)]
pub enum StreetError {
    /// The coordinate offsets do not describe the coordinate arrays.
    InvalidOffsets,
    /// An edge has fewer than two geometry coordinates.
    ShortGeometry { edge: u32 },
    /// An edge has a non-finite coordinate.
    InvalidCoordinates { edge: u32 },
    /// An edge endpoint is not below the vertex count.
    VertexOutOfRange { edge: u32, vertex_count: u32 },
    /// An edge's cost length is negative or not finite.
    InvalidLength { edge: u32 },
    /// A stop link references an edge that does not exist.
    LinkEdgeOutOfRange { link: usize, edge_count: u32 },
    /// A stop link's stop index is not below the stop count.
    StopOutOfRange { stop: u32, stop_count: u32 },
    /// A stop link's fraction is outside [0, 1] or its connector is
    /// negative or not finite.
    InvalidLink { link: usize },
    /// A mapped array range is out of bounds, misaligned, or shaped
    /// inconsistently with the index payload.
    InvalidMapping,
}

impl std::fmt::Display for StreetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StreetError::InvalidOffsets => {
                write!(f, "coordinate offsets do not match the coordinate arrays")
            }
            StreetError::ShortGeometry { edge } => {
                write!(f, "edge {edge} has fewer than two geometry coordinates")
            }
            StreetError::InvalidCoordinates { edge } => {
                write!(f, "edge {edge} has a non-finite coordinate")
            }
            StreetError::VertexOutOfRange { edge, vertex_count } => {
                write!(
                    f,
                    "edge {edge} has an endpoint out of range ({vertex_count} vertices)"
                )
            }
            StreetError::InvalidLength { edge } => {
                write!(f, "edge {edge} has a negative or non-finite length")
            }
            StreetError::LinkEdgeOutOfRange { link, edge_count } => {
                write!(
                    f,
                    "stop link {link} references an edge out of range ({edge_count} edges)"
                )
            }
            StreetError::StopOutOfRange { stop, stop_count } => {
                write!(f, "stop index {stop} is out of range ({stop_count} stops)")
            }
            StreetError::InvalidLink { link } => {
                write!(
                    f,
                    "stop link {link} has a fraction outside [0, 1] or an invalid connector"
                )
            }
            StreetError::InvalidMapping => {
                write!(f, "a mapped street array is out of bounds or misaligned")
            }
        }
    }
}

impl std::error::Error for StreetError {}

/// Reusable per-thread Dijkstra state. The maps are keyed by vertex and
/// cleared (capacity kept) between searches, so a query's memory scales
/// with the vertices it reaches, never with the network's vertex count.
#[derive(Default)]
struct SearchState {
    /// Best known distance in meters, present only for reached vertices.
    distances: HashMap<u32, f64>,
    /// Predecessor `(vertex, edge)` per reached vertex; seeds are absent.
    previous: HashMap<u32, (u32, u32)>,
    /// Pending `(distance bits, vertex)` entries. Non-negative floats
    /// order like their IEEE bit patterns.
    heap: BinaryHeap<Reverse<(u64, u32)>>,
}

impl SearchState {
    fn clear(&mut self) {
        self.distances.clear();
        self.previous.clear();
        self.heap.clear();
    }

    fn distance(&self, vertex: u32) -> f64 {
        self.distances
            .get(&vertex)
            .copied()
            .unwrap_or(f64::INFINITY)
    }
}

thread_local! {
    static SEARCH_STATE: std::cell::RefCell<SearchState> =
        std::cell::RefCell::new(SearchState::default());
    /// Per-thread scratch for the contraction-hierarchy one-to-many query,
    /// reused across a matrix's per-origin searches like `SEARCH_STATE`.
    static CH_SCRATCH: std::cell::RefCell<crate::ch::ChScratch> =
        std::cell::RefCell::new(crate::ch::ChScratch::default());
}

/// An immutable byte store backing mapped street arrays — a read-only
/// memory map of a saved artifact, on the Python side.
///
/// The bytes must never change while any [`StreetNetwork`] holds the
/// store: the network reinterprets them as typed slices, so a mutation
/// would be undefined behaviour, not just a wrong result. Producers
/// replace artifacts by writing a new file and atomically renaming it
/// over the old one — never by editing or truncating in place.
pub trait Backing: Send + Sync + 'static {
    fn bytes(&self) -> &[u8];
}

impl Backing for Vec<u8> {
    fn bytes(&self) -> &[u8] {
        self
    }
}

/// A typed view of a byte range inside a [`Backing`] store, resolved and
/// validated (bounds and alignment) once at adoption.
#[derive(Debug, Clone, Copy)]
struct MappedSlice<T> {
    ptr: *const T,
    len: usize,
}

// The slices point into an immutable `Backing` owned by the same
// `MappedArrays`, so they are as sendable as the store itself.
unsafe impl<T: Send + Sync> Send for MappedSlice<T> {}
unsafe impl<T: Send + Sync> Sync for MappedSlice<T> {}

impl<T> MappedSlice<T> {
    /// Resolves `count` elements at `offset` bytes into `bytes`,
    /// refusing out-of-bounds or misaligned ranges.
    fn new(bytes: &[u8], offset: u64, count: u64) -> Option<MappedSlice<T>> {
        let length = count.checked_mul(std::mem::size_of::<T>() as u64)?;
        let end = offset.checked_add(length)?;
        if end > bytes.len() as u64 {
            return None;
        }
        let ptr = bytes[offset as usize..].as_ptr();
        if !(ptr as usize).is_multiple_of(std::mem::align_of::<T>()) {
            return None;
        }
        Some(MappedSlice {
            ptr: ptr.cast(),
            len: count as usize,
        })
    }

    fn get(&self) -> &[T] {
        // SAFETY: `new` validated bounds and alignment against the
        // backing store, which outlives the slice and never mutates
        // (the `Backing` contract).
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

/// The persisted street arrays as owned vectors — the build and default
/// load path.
#[derive(Debug)]
struct OwnedArrays {
    adjacency_offsets: Vec<u32>,
    adj_targets: Vec<u32>,
    adj_meters: Vec<f64>,
    adj_edges: Vec<u32>,
    endpoints: Vec<u32>,
    lengths: Vec<f64>,
    coordinate_offsets: Vec<u32>,
    lons: Vec<i32>,
    lats: Vec<i32>,
    cumulative: Vec<f32>,
    index_boxes: Vec<Envelope>,
    index_payload: Vec<u32>,
}

/// The persisted street arrays as typed views into a mapped artifact.
struct MappedArrays {
    /// Keeps the mapping alive; every slice below points into it.
    _backing: std::sync::Arc<dyn Backing>,
    adjacency_offsets: MappedSlice<u32>,
    adj_targets: MappedSlice<u32>,
    adj_meters: MappedSlice<f64>,
    adj_edges: MappedSlice<u32>,
    endpoints: MappedSlice<u32>,
    lengths: MappedSlice<f64>,
    coordinate_offsets: MappedSlice<u32>,
    lons: MappedSlice<i32>,
    lats: MappedSlice<i32>,
    cumulative: MappedSlice<f32>,
    index_boxes: MappedSlice<Envelope>,
    index_payload: MappedSlice<u32>,
}

impl std::fmt::Debug for MappedArrays {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MappedArrays")
            .field("lons", &self.lons.len)
            .field("endpoints", &self.endpoints.len)
            .finish_non_exhaustive()
    }
}

/// The street arrays behind either backing. Query code reads through the
/// slice accessors and never touches a concrete field.
#[derive(Debug)]
enum Arrays {
    Owned(OwnedArrays),
    Mapped(MappedArrays),
}

macro_rules! array_accessor {
    ($name:ident, $type:ty) => {
        fn $name(&self) -> &[$type] {
            match self {
                Arrays::Owned(arrays) => &arrays.$name,
                Arrays::Mapped(arrays) => arrays.$name.get(),
            }
        }
    };
}

impl Arrays {
    array_accessor!(adjacency_offsets, u32);
    array_accessor!(adj_targets, u32);
    array_accessor!(adj_meters, f64);
    array_accessor!(adj_edges, u32);
    array_accessor!(endpoints, u32);
    array_accessor!(lengths, f64);
    array_accessor!(coordinate_offsets, u32);
    array_accessor!(lons, i32);
    array_accessor!(lats, i32);
    array_accessor!(cumulative, f32);
    array_accessor!(index_boxes, Envelope);
    array_accessor!(index_payload, u32);
}

/// Where each street array lives inside a mapped artifact: byte offsets
/// into the backing store plus element counts, as the artifact's
/// descriptor table records them. The index level starts are not mapped —
/// they are a pure function of the leaf count and are recomputed.
pub struct MappedStreets {
    pub backing: std::sync::Arc<dyn Backing>,
    pub vertex_count: u32,
    pub links: Vec<StoredLink>,
    /// `(byte offset, element count)` per array; offsets are absolute
    /// within the backing bytes.
    pub adjacency_offsets: (u64, u64),
    pub adj_targets: (u64, u64),
    pub adj_meters: (u64, u64),
    pub adj_edges: (u64, u64),
    pub endpoints: (u64, u64),
    pub lengths: (u64, u64),
    pub coordinate_offsets: (u64, u64),
    pub lons: (u64, u64),
    pub lats: (u64, u64),
    pub cumulative: (u64, u64),
    pub index_boxes: (u64, u64),
    pub index_payload: (u64, u64),
}

/// The walking street graph with its spatial index and stop links.
///
/// The large persisted arrays live behind [`Arrays`] — owned vectors or
/// typed views into a mapped artifact; queries are identical over both.
#[derive(Debug)]
pub struct StreetNetwork {
    arrays: Arrays,
    /// Start of each level in the index boxes (leaves at 0), plus a
    /// tail; derived from the leaf count, tiny, and always owned.
    level_starts: Vec<u32>,
    /// How each snapped stop enters the graph, endpoints denormalised.
    links: Vec<StoredLink>,
    /// `(vertex, link index)` pairs sorted by vertex — every link listed
    /// under both endpoints of its edge — so a search finds the links
    /// near its reached vertices without scanning all links.
    vertex_links: Vec<(u32, u32)>,
    /// An optional contraction hierarchy accelerating the bounded one-to-many
    /// walking searches (`access_stops`/`stop_transfers`/…). Built on demand by
    /// [`install_hierarchy`](Self::install_hierarchy); when absent the searches
    /// use `bounded_dijkstra`. The contraction persists with the artifact; its
    /// buckets are derived state, rebuilt on load.
    hierarchy: Option<ChIndex>,
}

/// A contraction hierarchy plus the one-to-many buckets over the stops'
/// link-endpoint vertices — the acceleration index for `reachable_from_snaps`.
#[derive(Debug)]
struct ChIndex {
    hierarchy: crate::ch::ContractionHierarchy,
    /// Built **unbounded** (over the link-endpoint vertices), so a query at any
    /// finite cutoff is within the buckets' build cutoff.
    buckets: crate::ch::Buckets,
}

impl StreetNetwork {
    /// Builds the network from flat edge arrays and stop links.
    ///
    /// `edges` carries `(from, to, meters)` per edge; edge `i`'s geometry
    /// runs from its `from` vertex through coordinates
    /// `coordinate_offsets[i]..coordinate_offsets[i + 1]` of the
    /// longitude/latitude arrays.
    pub fn new(
        vertex_count: u32,
        stop_count: u32,
        edges: &[(u32, u32, f64)],
        coordinate_offsets: &[u32],
        longitudes: &[f64],
        latitudes: &[f64],
        links: Vec<StopLink>,
    ) -> Result<StreetNetwork, StreetError> {
        if coordinate_offsets.len() != edges.len() + 1
            || coordinate_offsets.first() != Some(&0)
            || coordinate_offsets.last() != Some(&(longitudes.len() as u32))
            || longitudes.len() != latitudes.len()
        {
            return Err(StreetError::InvalidOffsets);
        }
        for (index, &(from, to, meters)) in edges.iter().enumerate() {
            let edge = index as u32;
            let start = coordinate_offsets[index];
            let end = coordinate_offsets[index + 1];
            if end < start || longitudes.len() < end as usize {
                return Err(StreetError::InvalidOffsets);
            }
            if end - start < 2 {
                return Err(StreetError::ShortGeometry { edge });
            }
            for position in start as usize..end as usize {
                if !longitudes[position].is_finite() || !latitudes[position].is_finite() {
                    return Err(StreetError::InvalidCoordinates { edge });
                }
            }
            if from >= vertex_count || to >= vertex_count {
                return Err(StreetError::VertexOutOfRange { edge, vertex_count });
            }
            if !meters.is_finite() || meters < 0.0 {
                return Err(StreetError::InvalidLength { edge });
            }
        }
        for (index, link) in links.iter().enumerate() {
            if link.edge as usize >= edges.len() {
                return Err(StreetError::LinkEdgeOutOfRange {
                    link: index,
                    edge_count: edges.len() as u32,
                });
            }
            if link.stop.0 >= stop_count {
                return Err(StreetError::StopOutOfRange {
                    stop: link.stop.0,
                    stop_count,
                });
            }
            if !(0.0..=1.0).contains(&link.fraction)
                || !link.connector.is_finite()
                || link.connector < 0.0
            {
                return Err(StreetError::InvalidLink { link: index });
            }
        }

        // Quantize the geometry onto the fixed-point grid up front, so
        // every derived structure — permutation keys, densified points,
        // cumulative lengths, index boxes — is a pure function of the
        // stored coordinates.
        let fixed_lons: Vec<i32> = longitudes.iter().map(|&lon| quantize(lon)).collect();
        let fixed_lats: Vec<i32> = latitudes.iter().map(|&lat| quantize(lat)).collect();

        // Hilbert-order the edges by their first coordinate and renumber
        // vertices by first appearance in that order, so spatially-nearby
        // streets are nearby in every edge- and vertex-indexed array. The
        // ids are internal; costs and results are unchanged (only
        // exactly-equal ties could resolve differently than under the
        // input order). Hilbert-cell ties break by the edges' own data —
        // endpoints, length, vertices, then the full geometry — never by
        // the input position, so the layout is the same whatever order the
        // edges arrive in (edges identical in every field are
        // interchangeable, and their links follow them either way).
        let bounds = coordinate_bounds(&fixed_lons, &fixed_lats);
        let mut order: Vec<u32> = (0..edges.len() as u32).collect();
        type EdgeKey = (u64, i32, i32, i32, i32, u64, u32, u32);
        let keys: Vec<EdgeKey> = (0..edges.len())
            .map(|edge| {
                let first = coordinate_offsets[edge] as usize;
                let last = coordinate_offsets[edge + 1] as usize - 1;
                let (from, to, meters) = edges[edge];
                (
                    hilbert(
                        grid_position(fixed_lons[first], bounds[0], bounds[2]),
                        grid_position(fixed_lats[first], bounds[1], bounds[3]),
                    ),
                    fixed_lons[first],
                    fixed_lats[first],
                    fixed_lons[last],
                    fixed_lats[last],
                    meters.to_bits(),
                    from,
                    to,
                )
            })
            .collect();
        let geometry_points = |edge: u32| {
            let start = coordinate_offsets[edge as usize] as usize;
            let end = coordinate_offsets[edge as usize + 1] as usize;
            fixed_lons[start..end]
                .iter()
                .zip(&fixed_lats[start..end])
                .map(|(&lon, &lat)| (lon, lat))
        };
        order.sort_unstable_by(|&a, &b| {
            keys[a as usize]
                .cmp(&keys[b as usize])
                .then_with(|| geometry_points(a).cmp(geometry_points(b)))
        });

        let mut edge_map = vec![0u32; edges.len()];
        let mut vertex_map = vec![u32::MAX; vertex_count as usize];
        let mut next_vertex = 0u32;
        let mut permuted_edges = Vec::with_capacity(edges.len());
        let mut permuted_offsets = Vec::with_capacity(edges.len() + 1);
        let mut permuted_lons = Vec::with_capacity(fixed_lons.len());
        let mut permuted_lats = Vec::with_capacity(fixed_lats.len());
        permuted_offsets.push(0u32);
        for (new_edge, &old_edge) in order.iter().enumerate() {
            edge_map[old_edge as usize] = new_edge as u32;
            let (from, to, meters) = edges[old_edge as usize];
            let mut renumber = |vertex: u32| {
                if vertex_map[vertex as usize] == u32::MAX {
                    vertex_map[vertex as usize] = next_vertex;
                    next_vertex += 1;
                }
                vertex_map[vertex as usize]
            };
            permuted_edges.push((renumber(from), renumber(to), meters));
            let start = coordinate_offsets[old_edge as usize] as usize;
            let end = coordinate_offsets[old_edge as usize + 1] as usize;
            permuted_lons.extend_from_slice(&fixed_lons[start..end]);
            permuted_lats.extend_from_slice(&fixed_lats[start..end]);
            permuted_offsets.push(permuted_lons.len() as u32);
        }
        // Vertices no edge touches keep ids after the connected ones.
        for slot in vertex_map.iter_mut() {
            if *slot == u32::MAX {
                *slot = next_vertex;
                next_vertex += 1;
            }
        }
        let edges = permuted_edges;
        let links: Vec<StoredLink> = links
            .into_iter()
            .map(|link| {
                let edge = edge_map[link.edge as usize];
                let (from, to, _) = edges[edge as usize];
                StoredLink {
                    stop: link.stop,
                    edge,
                    fraction: link.fraction,
                    connector: link.connector,
                    from,
                    to,
                }
            })
            .collect();

        let (dense_offsets, lons, lats, cumulative) =
            densify(&permuted_offsets, &permuted_lons, &permuted_lats);

        let mut adjacency_offsets = vec![0u32; vertex_count as usize + 1];
        for &(from, to, _) in &edges {
            adjacency_offsets[from as usize + 1] += 1;
            adjacency_offsets[to as usize + 1] += 1;
        }
        for vertex in 0..vertex_count as usize {
            adjacency_offsets[vertex + 1] += adjacency_offsets[vertex];
        }
        let mut adj_targets = vec![0u32; edges.len() * 2];
        let mut adj_meters = vec![0f64; edges.len() * 2];
        let mut adj_edges = vec![0u32; edges.len() * 2];
        let mut cursor = adjacency_offsets.clone();
        for (index, &(from, to, meters)) in edges.iter().enumerate() {
            for (a, b) in [(from, to), (to, from)] {
                let slot = cursor[a as usize] as usize;
                adj_targets[slot] = b;
                adj_meters[slot] = meters;
                adj_edges[slot] = index as u32;
                cursor[a as usize] += 1;
            }
        }

        let index = build_index(&dense_offsets, &lons, &lats);
        let endpoints: Vec<u32> = edges.iter().flat_map(|&(from, to, _)| [from, to]).collect();
        let vertex_links = build_vertex_links(&links);

        Ok(StreetNetwork {
            arrays: Arrays::Owned(OwnedArrays {
                adjacency_offsets,
                adj_targets,
                adj_meters,
                adj_edges,
                endpoints,
                lengths: edges.iter().map(|&(_, _, meters)| meters).collect(),
                coordinate_offsets: dense_offsets,
                lons,
                lats,
                cumulative,
                index_boxes: index.boxes,
                index_payload: index.payload,
            }),
            level_starts: index.level_starts,
            links,
            vertex_links,
            hierarchy: None,
        })
    }

    /// Number of street vertices.
    pub fn vertex_count(&self) -> u32 {
        self.arrays.adjacency_offsets().len() as u32 - 1
    }

    /// Number of street edges.
    pub fn edge_count(&self) -> u32 {
        (self.arrays.endpoints().len() / 2) as u32
    }

    /// Number of stop links.
    pub fn link_count(&self) -> usize {
        self.links.len()
    }

    /// Whether the arrays are views into a mapped artifact.
    pub fn is_mapped(&self) -> bool {
        matches!(self.arrays, Arrays::Mapped(_))
    }

    /// Whether a contraction-hierarchy index is installed.
    pub fn has_hierarchy(&self) -> bool {
        self.hierarchy.is_some()
    }

    /// Builds and installs a contraction hierarchy over the walking graph, plus
    /// **unbounded** one-to-many buckets over the stops' link-endpoint vertices,
    /// so the bounded walking searches (`access_stops`/`stop_transfers`/…) run as
    /// hierarchy queries instead of graph sweeps. Heavy (the contraction is
    /// run-once); opt-in, so the default build path is unchanged. Idempotent —
    /// rebuilding replaces the index. The contraction persists with the artifact,
    /// so acceleration survives `save`/`load` without a second contraction.
    pub fn install_hierarchy(&mut self) {
        let hierarchy = crate::ch::ContractionHierarchy::build(
            self.vertex_count(),
            self.arrays.adjacency_offsets(),
            self.arrays.adj_targets(),
            self.arrays.adj_meters(),
        );
        self.install_hierarchy_from(hierarchy);
    }

    /// Installs a prebuilt contraction hierarchy, deriving its one-to-many
    /// buckets from the stops' link-endpoint vertices. Used to restore a
    /// persisted hierarchy on `load`: the run-once contraction is deserialised
    /// and only the buckets — derived state — are rebuilt, exactly as
    /// [`install_hierarchy`](Self::install_hierarchy) builds them for a fresh
    /// contraction, so a loaded network matches a freshly built one.
    pub fn install_hierarchy_from(&mut self, hierarchy: crate::ch::ContractionHierarchy) {
        let mut endpoints: Vec<u32> = self
            .links
            .iter()
            .flat_map(|link| [link.from, link.to])
            .collect();
        endpoints.sort_unstable();
        endpoints.dedup();
        let buckets = hierarchy.buckets(&endpoints, f64::INFINITY);
        self.hierarchy = Some(ChIndex { hierarchy, buckets });
    }

    /// The installed contraction hierarchy, if any — the run-once contraction
    /// result. Persisted with the artifact; the buckets are rebuilt on load.
    pub fn hierarchy(&self) -> Option<&crate::ch::ContractionHierarchy> {
        self.hierarchy.as_ref().map(|index| &index.hierarchy)
    }

    /// A fingerprint of this network's walking-graph CSR, matching what a
    /// hierarchy built over it records: a persisted hierarchy binds to this so a
    /// loaded artifact with a mismatched graph is refused.
    pub fn graph_fingerprint(&self) -> u64 {
        crate::ch::csr_fingerprint(
            self.arrays.adjacency_offsets(),
            self.arrays.adj_targets(),
            self.arrays.adj_meters(),
        )
    }

    /// An edge's `(from, to)` endpoint vertices.
    fn edge_endpoints(&self, edge: u32) -> (u32, u32) {
        let endpoints = self.arrays.endpoints();
        (
            endpoints[2 * edge as usize],
            endpoints[2 * edge as usize + 1],
        )
    }

    /// A stored coordinate as float degrees.
    fn coordinate(&self, position: usize) -> (f64, f64) {
        (
            degrees(self.arrays.lons()[position]),
            degrees(self.arrays.lats()[position]),
        )
    }

    /// A stored cumulative along-distance as f64 meters.
    fn along(&self, position: usize) -> f64 {
        f64::from(self.arrays.cumulative()[position])
    }

    /// The network's serializable state.
    pub fn to_parts(&self) -> StreetNetworkParts {
        StreetNetworkParts {
            vertex_count: self.vertex_count(),
            adjacency_offsets: self.arrays.adjacency_offsets().to_vec(),
            adj_targets: self.arrays.adj_targets().to_vec(),
            adj_meters: self.arrays.adj_meters().to_vec(),
            adj_edges: self.arrays.adj_edges().to_vec(),
            endpoints: self.arrays.endpoints().to_vec(),
            lengths: self.arrays.lengths().to_vec(),
            coordinate_offsets: self.arrays.coordinate_offsets().to_vec(),
            lons: self.arrays.lons().to_vec(),
            lats: self.arrays.lats().to_vec(),
            cumulative: self.arrays.cumulative().to_vec(),
            index_boxes: self
                .arrays
                .index_boxes()
                .iter()
                .flat_map(|envelope| *envelope)
                .collect(),
            index_payload: self.arrays.index_payload().to_vec(),
            index_level_starts: self.level_starts.clone(),
            links: self.links.clone(),
        }
    }

    /// Adopts a network from its serialized parts — nothing street-sized
    /// is rebuilt (the spatial index arrives as arrays); the one derived
    /// rebuild is the L-sized vertex→link index, from the links'
    /// denormalised endpoints.
    pub fn from_parts(parts: StreetNetworkParts) -> StreetNetwork {
        let vertex_links = build_vertex_links(&parts.links);
        StreetNetwork {
            arrays: Arrays::Owned(OwnedArrays {
                adjacency_offsets: parts.adjacency_offsets,
                adj_targets: parts.adj_targets,
                adj_meters: parts.adj_meters,
                adj_edges: parts.adj_edges,
                endpoints: parts.endpoints,
                lengths: parts.lengths,
                coordinate_offsets: parts.coordinate_offsets,
                lons: parts.lons,
                lats: parts.lats,
                cumulative: parts.cumulative,
                index_boxes: parts
                    .index_boxes
                    .chunks_exact(4)
                    .map(|chunk| [chunk[0], chunk[1], chunk[2], chunk[3]])
                    .collect(),
                index_payload: parts.index_payload,
            }),
            level_starts: parts.index_level_starts,
            links: parts.links,
            vertex_links,
            hierarchy: None,
        }
    }

    /// Adopts a network whose arrays stay typed views into a mapped
    /// artifact — no street-sized bytes are read or copied. The caller
    /// (the artifact loader) has validated the layout against the
    /// descriptor table; the bounds and alignment checks here are the
    /// soundness net for constructing the slices, and content stays as
    /// trusted as the mapping (see [`Backing`]).
    pub fn from_mapped(spec: MappedStreets) -> Result<StreetNetwork, StreetError> {
        let bytes = spec.backing.bytes();
        fn slice<T>(
            bytes: &[u8],
            (offset, count): (u64, u64),
        ) -> Result<MappedSlice<T>, StreetError> {
            MappedSlice::new(bytes, offset, count).ok_or(StreetError::InvalidMapping)
        }
        // Envelopes are stored as flat i32 quadruples.
        if !spec.index_boxes.1.is_multiple_of(4) {
            return Err(StreetError::InvalidMapping);
        }
        let arrays = MappedArrays {
            adjacency_offsets: slice(bytes, spec.adjacency_offsets)?,
            adj_targets: slice(bytes, spec.adj_targets)?,
            adj_meters: slice(bytes, spec.adj_meters)?,
            adj_edges: slice(bytes, spec.adj_edges)?,
            endpoints: slice(bytes, spec.endpoints)?,
            lengths: slice(bytes, spec.lengths)?,
            coordinate_offsets: slice(bytes, spec.coordinate_offsets)?,
            lons: slice(bytes, spec.lons)?,
            lats: slice(bytes, spec.lats)?,
            cumulative: slice(bytes, spec.cumulative)?,
            index_boxes: slice(bytes, (spec.index_boxes.0, spec.index_boxes.1 / 4))?,
            index_payload: slice(bytes, spec.index_payload)?,
            _backing: spec.backing,
        };
        let level_starts = level_starts_for(arrays.index_payload.len / 2);
        if *level_starts.last().unwrap() as usize != arrays.index_boxes.len {
            return Err(StreetError::InvalidMapping);
        }
        let vertex_links = build_vertex_links(&spec.links);
        Ok(StreetNetwork {
            arrays: Arrays::Mapped(arrays),
            level_starts,
            links: spec.links,
            vertex_links,
            hierarchy: None,
        })
    }

    /// Snaps a coordinate to its nearest edge within `max_snap_distance`
    /// meters through the packed segment index. Non-finite coordinates or a
    /// non-finite or negative allowance never snap.
    pub fn snap(&self, latitude: f64, longitude: f64, max_snap_distance: f64) -> Option<Snap> {
        if !latitude.is_finite()
            || !longitude.is_finite()
            || !max_snap_distance.is_finite()
            || max_snap_distance < 0.0
        {
            return None;
        }
        // Every segment within `max_snap_distance` is inside this lon/lat
        // envelope; the index is queried by envelope intersection, never by
        // a degree-Euclidean nearest, and each candidate is re-measured
        // exactly. Exact connector ties break by (edge, fraction), so the
        // winner is a function of the built network, not of index internals.
        let envelope = snap_envelope(latitude, longitude, max_snap_distance);
        let mut candidates = Vec::new();
        self.query_index_into(&envelope, &mut candidates);
        let mut best: Option<Snap> = None;
        for (edge, start) in candidates {
            let (connector, fraction) = self.foot_on_segment(latitude, longitude, edge, start);
            if connector <= max_snap_distance
                && best.is_none_or(|current| {
                    connector < current.connector
                        || (connector == current.connector
                            && (edge, fraction) < (current.edge, current.fraction))
                })
            {
                best = Some(Snap {
                    edge,
                    fraction,
                    connector,
                });
            }
        }
        best
    }

    /// Collects the payloads of every packed-index leaf whose box
    /// intersects the envelope, through the array accessors.
    fn query_index_into(&self, envelope: &Envelope, matches: &mut Vec<(u32, u32)>) {
        query_packed_index(
            self.arrays.index_boxes(),
            self.arrays.index_payload(),
            &self.level_starts,
            envelope,
            matches,
        );
    }

    /// The exact connector distance and true-length fraction of a query's
    /// foot on one segment (its first coordinate is `start`), measured in an
    /// equirectangular frame local to the query — exact because segments are
    /// short (densified below `MAX_SEGMENT_METERS`).
    fn foot_on_segment(&self, latitude: f64, longitude: f64, edge: u32, start: u32) -> (f64, f64) {
        let (a, b) = (start as usize, start as usize + 1);
        let (mpd_lon, mpd_lat) = meters_per_degree(latitude);
        let to_xy = |lon: f64, lat: f64| {
            (
                longitude_delta(longitude, lon) * mpd_lon,
                (lat - latitude) * mpd_lat,
            )
        };
        let (lon_a, lat_a) = self.coordinate(a);
        let (lon_b, lat_b) = self.coordinate(b);
        let (ax, ay) = to_xy(lon_a, lat_a);
        let (bx, by) = to_xy(lon_b, lat_b);
        let (dx, dy) = (bx - ax, by - ay);
        let squared = dx * dx + dy * dy;
        // The query sits at the frame origin, so the foot parameter is
        // ((Q - A)·(B - A)) / |B - A|² with Q = 0.
        let t = if squared > 0.0 {
            ((-ax * dx - ay * dy) / squared).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let (px, py) = (ax + t * dx, ay + t * dy);
        let connector = (px * px + py * py).sqrt();

        let end = self.arrays.coordinate_offsets()[edge as usize + 1] as usize;
        let along = self.along(a) + t * (self.along(b) - self.along(a));
        let total = self.along(end - 1);
        let fraction = if total > 0.0 { along / total } else { 0.0 };
        (connector, fraction)
    }

    /// Every transit stop reachable on foot from a coordinate, sorted by
    /// stop index, or `None` when the coordinate is farther than
    /// `max_snap_distance` meters from the network or any parameter is
    /// out of range (the speed must be positive and finite, the cutoffs
    /// non-negative and finite). Walking is undirected, so the same
    /// search answers egress. Seconds round up — understating a walking
    /// time could let routing catch a departure the walk actually
    /// misses — while meters stay the exact street-path length.
    pub fn access_stops(
        &self,
        latitude: f64,
        longitude: f64,
        walking_speed: f64,
        max_seconds: f64,
        max_snap_distance: f64,
    ) -> Option<Vec<WalkedStop>> {
        if !walking_speed.is_finite()
            || walking_speed <= 0.0
            || !max_seconds.is_finite()
            || max_seconds < 0.0
        {
            return None;
        }
        let snap = self.snap(latitude, longitude, max_snap_distance)?;
        Some(self.reachable_from_snaps(&[snap], walking_speed, max_seconds))
    }

    /// Every transit stop reachable on foot from one or more snapped
    /// points, sorted by stop index. The bounded road search plus the
    /// link join shared by [`access_stops`](Self::access_stops) (a
    /// coordinate's single snap) and
    /// [`stop_transfers`](Self::stop_transfers) (all of a source stop's
    /// links). The search is seeded from every source snap's split-edge
    /// endpoints, and the walk to a candidate is the minimum over all
    /// source snaps — so a source that snaps to several edges reaches
    /// through whichever link is nearest, just as a destination stop is
    /// reached through its nearest link. The caller has already
    /// validated the speed and cutoff. The per-vertex distances come from the
    /// installed contraction hierarchy when present, else a `bounded_dijkstra`
    /// over the thread-local search state (reused per worker thread). Seconds
    /// round up (understating a walk could catch a missed departure);
    /// meters stay the exact street-path length.
    fn reachable_from_snaps(
        &self,
        snaps: &[Snap],
        walking_speed: f64,
        max_seconds: f64,
    ) -> Vec<WalkedStop> {
        let cutoff = max_seconds * walking_speed;
        let mut seeds: Vec<(u32, f64)> = Vec::with_capacity(snaps.len() * 2);
        for snap in snaps {
            let (from, to) = self.edge_endpoints(snap.edge);
            let length = self.arrays.lengths()[snap.edge as usize];
            seeds.push((from, snap.connector + snap.fraction * length));
            seeds.push((to, snap.connector + (1.0 - snap.fraction) * length));
        }
        // The reached per-vertex distances, from the contraction hierarchy when
        // installed, else a bounded Dijkstra over the graph. Both feed the same
        // link join, so the `WalkedStop`s are identical.
        match &self.hierarchy {
            Some(index) => CH_SCRATCH.with(|cell| {
                let scratch = &mut cell.borrow_mut();
                index
                    .hierarchy
                    .one_to_many(&index.buckets, &seeds, cutoff, scratch);
                self.link_join(snaps, cutoff, walking_speed, scratch.best())
            }),
            None => SEARCH_STATE.with(|cell| {
                let state = &mut cell.borrow_mut();
                self.bounded_dijkstra(&seeds, cutoff, state);
                self.link_join(snaps, cutoff, walking_speed, &state.distances)
            }),
        }
    }

    /// Joins per-vertex reached `distances` to the stops through their links —
    /// each stop's walk is the minimum over its links of the reached
    /// edge-endpoint distance plus the on-edge offset and connector, and the
    /// direct on-edge walk from a source snap sharing the link's edge. Shared by
    /// the `bounded_dijkstra` and contraction-hierarchy paths of
    /// [`reachable_from_snaps`](Self::reachable_from_snaps), so both return the
    /// same `WalkedStop`s. Seconds round up; meters stay exact.
    fn link_join(
        &self,
        snaps: &[Snap],
        cutoff: f64,
        walking_speed: f64,
        distances: &HashMap<u32, f64>,
    ) -> Vec<WalkedStop> {
        let distance = |vertex: u32| distances.get(&vertex).copied().unwrap_or(f64::INFINITY);
        // Candidate links: those at reached vertices, plus those on any source
        // edge (whose direct on-edge path can be walkable even when neither
        // endpoint is within the cutoff). Sorted so processing order is
        // independent of hash-map iteration order.
        let mut candidates: Vec<u32> = Vec::new();
        for snap in snaps {
            let (from, to) = self.edge_endpoints(snap.edge);
            candidates.extend(self.links_at(from));
            candidates.extend(self.links_at(to));
        }
        for &vertex in distances.keys() {
            candidates.extend(self.links_at(vertex));
        }
        candidates.sort_unstable();
        candidates.dedup();
        let mut nearest: HashMap<StopIdx, f64> = HashMap::new();
        for index in candidates {
            let link = &self.links[index as usize];
            let (link_from, link_to) = (link.from, link.to);
            let link_length = self.arrays.lengths()[link.edge as usize];
            let mut meters = f64::min(
                distance(link_from) + link.fraction * link_length,
                distance(link_to) + (1.0 - link.fraction) * link_length,
            ) + link.connector;
            // A source snap on this link's edge can walk it directly, never
            // reaching an endpoint; keep the shortest over all.
            for snap in snaps {
                if snap.edge == link.edge {
                    let length = self.arrays.lengths()[snap.edge as usize];
                    let direct = snap.connector
                        + (link.fraction - snap.fraction).abs() * length
                        + link.connector;
                    meters = meters.min(direct);
                }
            }
            if meters <= cutoff + 1e-9 {
                // A stop with several links keeps its shortest path.
                nearest
                    .entry(link.stop)
                    .and_modify(|best| *best = best.min(meters))
                    .or_insert(meters);
            }
        }
        let mut reached: Vec<WalkedStop> = nearest
            .into_iter()
            .map(|(stop, meters)| WalkedStop {
                stop,
                seconds: seconds(meters / walking_speed),
                meters,
            })
            .collect();
        reached.sort_unstable_by_key(|walk| walk.stop);
        reached
    }

    /// The walking transfers between every pair of linked stops within
    /// `max_seconds` on foot — the road-network shortest walk, computed
    /// natively and in parallel over source stops. Emitted as
    /// `(from, to, seconds, meters)` edges, self-transfers dropped.
    ///
    /// Each edge is one complete direct road walk (never a chain through
    /// an intermediate stop): the shortest path between two coordinates
    /// runs over streets, not stops, so a bounded search from each source
    /// stop's links already yields the true walk to every stop in range.
    /// A stop that snaps to several edges is searched from all of them,
    /// matching how `access_stops` reaches a stop through any of its
    /// links, so the result stays symmetric. Unlike the footpath closure
    /// this can replace, the set is *not* padded with beyond-cutoff pairs
    /// reachable only by chaining shorter hops — those represent walks
    /// past the cutoff and are excluded by policy; single-hop relaxation
    /// stays complete for every walk within it. Seconds round up and
    /// meters are exact, matching `access_stops`.
    ///
    /// This is a **road-search substrate**, not the ULTRA shortcut set:
    /// it enumerates every within-cutoff walk with no timetable awareness
    /// or witness pruning, so it is the dense (unrestricted) transfer
    /// graph ULTRA's shortcut enumeration consumes and reduces — and the
    /// oracle its minimal output is checked against — not the minimal
    /// shortcut output itself (see plans/ultra-plan.md).
    pub fn stop_transfers(
        &self,
        walking_speed: f64,
        max_seconds: f64,
    ) -> Vec<(StopIdx, StopIdx, u32, f64)> {
        if !walking_speed.is_finite()
            || walking_speed <= 0.0
            || !max_seconds.is_finite()
            || max_seconds < 0.0
        {
            return Vec::new();
        }
        // Group each source stop's links so the search leaves from all of
        // them — a stop can snap to several edges — mirroring how
        // access_stops reaches a stop through any of its links.
        let mut by_stop: HashMap<StopIdx, Vec<Snap>> = HashMap::new();
        for link in &self.links {
            by_stop.entry(link.stop).or_default().push(Snap {
                edge: link.edge,
                fraction: link.fraction,
                connector: link.connector,
            });
        }
        let mut sources: Vec<(StopIdx, Vec<Snap>)> = by_stop.into_iter().collect();
        sources.sort_unstable_by_key(|(stop, _)| *stop);
        let mut edges: Vec<(StopIdx, StopIdx, u32, f64)> = sources
            .into_par_iter()
            .flat_map_iter(|(stop, snaps)| {
                self.reachable_from_snaps(&snaps, walking_speed, max_seconds)
                    .into_iter()
                    .filter(move |walk| walk.stop != stop)
                    .map(move |walk| (stop, walk.stop, walk.seconds, walk.meters))
                    .collect::<Vec<_>>()
            })
            .collect();
        // Deterministic order regardless of worker scheduling.
        edges.sort_unstable_by_key(|&(from, to, _, _)| (from, to));
        edges
    }

    /// Walking times and exact street distances from one snapped point
    /// to many, or `None` where a target is unsnapped or beyond the
    /// cutoff — one bounded search serving a whole matrix row, so a
    /// direct-walk fill costs one street search per origin, never one
    /// per OD pair. Distances include both connectors; seconds round up
    /// like every walk.
    pub fn walk_to_snaps(
        &self,
        from: &Snap,
        targets: &[Option<Snap>],
        walking_speed: f64,
        max_seconds: f64,
    ) -> Vec<Option<(u32, f64)>> {
        if !walking_speed.is_finite()
            || walking_speed <= 0.0
            || !max_seconds.is_finite()
            || max_seconds < 0.0
        {
            return vec![None; targets.len()];
        }
        let cutoff = max_seconds * walking_speed;
        let (from_u, from_v) = self.edge_endpoints(from.edge);
        let from_length = self.arrays.lengths()[from.edge as usize];
        SEARCH_STATE.with(|cell| {
            let state = &mut cell.borrow_mut();
            self.bounded_dijkstra(
                &[
                    (from_u, from.connector + from.fraction * from_length),
                    (from_v, from.connector + (1.0 - from.fraction) * from_length),
                ],
                cutoff,
                state,
            );
            targets
                .iter()
                .map(|target| {
                    let target = target.as_ref()?;
                    let length = self.arrays.lengths()[target.edge as usize];
                    let (u, v) = self.edge_endpoints(target.edge);
                    let mut meters = f64::min(
                        state.distance(u) + target.fraction * length,
                        state.distance(v) + (1.0 - target.fraction) * length,
                    ) + target.connector;
                    if target.edge == from.edge {
                        // Both points sit on one edge: walking between
                        // them never has to reach an endpoint.
                        let direct = from.connector
                            + (target.fraction - from.fraction).abs() * from_length
                            + target.connector;
                        meters = meters.min(direct);
                    }
                    (meters <= cutoff + 1e-9).then(|| (seconds(meters / walking_speed), meters))
                })
                .collect()
        })
    }

    /// Direct walking times and distances between coordinate sets, in
    /// parallel over the origins: per origin, per destination,
    /// `(seconds, meters)` or `None` where either point does not snap
    /// or the walk exceeds the cutoff. A destination at the origin's
    /// exact coordinate is a zero walk — snap-level arithmetic would
    /// charge the connector out to the street and back.
    pub fn walk_matrix(
        &self,
        origins: &[(f64, f64)],
        destinations: &[(f64, f64)],
        walking_speed: f64,
        max_seconds: f64,
        max_snap_distance: f64,
    ) -> Vec<Vec<Option<(u32, f64)>>> {
        let targets: Vec<Option<Snap>> = destinations
            .iter()
            .map(|&(lat, lon)| self.snap(lat, lon, max_snap_distance))
            .collect();
        origins
            .par_iter()
            .map(
                |&origin| match self.snap(origin.0, origin.1, max_snap_distance) {
                    Some(from) => {
                        let mut row =
                            self.walk_to_snaps(&from, &targets, walking_speed, max_seconds);
                        for ((cell, target), &destination) in
                            row.iter_mut().zip(&targets).zip(destinations)
                        {
                            if target.is_some() && destination == origin {
                                *cell = Some((0, 0.0));
                            }
                        }
                        row
                    }
                    None => vec![None; destinations.len()],
                },
            )
            .collect()
    }

    /// The indices of the links whose edge touches a vertex.
    fn links_at(&self, vertex: u32) -> impl Iterator<Item = u32> + '_ {
        let start = self.vertex_links.partition_point(|&(v, _)| v < vertex);
        self.vertex_links[start..]
            .iter()
            .take_while(move |&&(v, _)| v == vertex)
            .map(|&(_, link)| link)
    }

    /// Links many coordinates against the network in parallel: each
    /// point's walkable stops, or `None` where a point does not snap.
    /// One linking serves a whole matrix — per-origin work is then a
    /// transit search plus a table join, never a street search per OD
    /// pair.
    pub fn link_many(
        &self,
        points: &[(f64, f64)],
        walking_speed: f64,
        max_seconds: f64,
        max_snap_distance: f64,
    ) -> Vec<Option<Vec<WalkedStop>>> {
        points
            .par_iter()
            .map(|&(latitude, longitude)| {
                self.access_stops(
                    latitude,
                    longitude,
                    walking_speed,
                    max_seconds,
                    max_snap_distance,
                )
            })
            .collect()
    }

    /// A stop's snap link as a [`Snap`], when the stop is linked. A
    /// stop with several links yields the nearest one.
    pub fn stop_snap(&self, stop: StopIdx) -> Option<Snap> {
        self.links
            .iter()
            .filter(|link| link.stop == stop)
            .min_by(|a, b| a.connector.total_cmp(&b.connector))
            .map(|link| Snap {
                edge: link.edge,
                fraction: link.fraction,
                connector: link.connector,
            })
    }

    /// The walked street path between two snapped coordinates: the
    /// shortest path's geometry in EPSG:4326 — endpoint, connector,
    /// partial edges at the snap fractions, full edges between — and
    /// its length in meters, connectors included. `None` when the snap
    /// points lie in different street components.
    ///
    /// For stop pairs whose stored transfer chained several footpaths
    /// beyond the direct-search cutoff, this direct shortest path can
    /// be shorter than the stored transfer.
    pub fn walk_path(
        &self,
        from_point: (f64, f64),
        from: &Snap,
        to_point: (f64, f64),
        to: &Snap,
    ) -> Option<(Vec<(f64, f64)>, f64)> {
        let from_length = self.arrays.lengths()[from.edge as usize];
        let to_length = self.arrays.lengths()[to.edge as usize];
        // The direct candidate: both points on one edge.
        let direct = (from.edge == to.edge).then(|| {
            from.connector + (from.fraction - to.fraction).abs() * from_length + to.connector
        });

        let (from_u, from_v) = self.edge_endpoints(from.edge);
        let (to_u, to_v) = self.edge_endpoints(to.edge);
        let transit = SEARCH_STATE.with(|cell| {
            let state = &mut cell.borrow_mut();
            self.dijkstra_with_paths(
                &[
                    (from_u, from.connector + from.fraction * from_length),
                    (from_v, from.connector + (1.0 - from.fraction) * from_length),
                ],
                (to_u, to_v),
                state,
            );
            let via_u = state.distance(to_u) + to.fraction * to_length + to.connector;
            let via_v = state.distance(to_v) + (1.0 - to.fraction) * to_length + to.connector;
            let best_transit = if via_u <= via_v { via_u } else { via_v };
            if !best_transit.is_finite() {
                return None;
            }
            let exit = if via_u <= via_v { to_u } else { to_v };
            // Walk the predecessor chain back to the seed vertex.
            let mut vertices = vec![exit];
            let mut edges = Vec::new();
            let mut at = exit;
            while let Some(&(prev, edge)) = state.previous.get(&at) {
                vertices.push(prev);
                edges.push(edge);
                at = prev;
            }
            vertices.reverse();
            edges.reverse();
            Some((vertices, edges, exit, best_transit))
        });
        if let Some((vertices, edges, exit, best_transit)) =
            transit.filter(|&(_, _, _, meters)| direct.is_none_or(|direct| meters < direct))
        {
            let mut path = Vec::new();
            path.push((from_point.1, from_point.0));
            path.push(self.point_at(from.edge, from.fraction));
            // The partial first edge, from the snap point to the seed.
            // A loop edge's endpoints coincide; its cheaper side was
            // seeded, so pick the side by the snap fraction.
            let entry = vertices[0];
            let entry_fraction = if from_u == from_v {
                if from.fraction <= 0.5 {
                    0.0
                } else {
                    1.0
                }
            } else if entry == from_u {
                0.0
            } else {
                1.0
            };
            path.extend(self.edge_slice(from.edge, from.fraction, entry_fraction));
            for (step, &edge) in edges.iter().enumerate() {
                let (u, _) = self.edge_endpoints(edge);
                let forward = vertices[step] == u;
                path.extend(self.edge_slice(
                    edge,
                    if forward { 0.0 } else { 1.0 },
                    if forward { 1.0 } else { 0.0 },
                ));
            }
            // The partial last edge, from the exit vertex to the snap;
            // sides of a loop edge again picked by the snap fraction.
            let exit_fraction = if to_u == to_v {
                if to.fraction <= 0.5 {
                    0.0
                } else {
                    1.0
                }
            } else if exit == to_u {
                0.0
            } else {
                1.0
            };
            path.extend(self.edge_slice(to.edge, exit_fraction, to.fraction));
            path.push(self.point_at(to.edge, to.fraction));
            path.push((to_point.1, to_point.0));
            return Some((dedup_consecutive(path), best_transit));
        }

        let meters = direct?;
        let mut path = vec![(from_point.1, from_point.0)];
        path.push(self.point_at(from.edge, from.fraction));
        path.extend(self.edge_slice(from.edge, from.fraction, to.fraction));
        path.push(self.point_at(to.edge, to.fraction));
        path.push((to_point.1, to_point.0));
        Some((dedup_consecutive(path), meters))
    }

    /// A Dijkstra from the seed vertices into the search state, recording
    /// each reached vertex's predecessor and the edge it was entered
    /// through. Stops once both target vertices are settled — a settled
    /// distance and the predecessor chain behind it are final — so the
    /// work grows with the search ball around the seeds, never with the
    /// network.
    fn dijkstra_with_paths(
        &self,
        sources: &[(u32, f64)],
        targets: (u32, u32),
        state: &mut SearchState,
    ) {
        state.clear();
        let adjacency_offsets = self.arrays.adjacency_offsets();
        let adj_targets = self.arrays.adj_targets();
        let adj_meters = self.arrays.adj_meters();
        let adj_edges = self.arrays.adj_edges();
        for &(vertex, distance) in sources {
            if distance < state.distance(vertex) {
                state.distances.insert(vertex, distance);
                state.previous.remove(&vertex);
                state.heap.push(Reverse((distance.to_bits(), vertex)));
            }
        }
        let (target_a, target_b) = targets;
        let mut settled_a = false;
        let mut settled_b = false;
        while let Some(Reverse((bits, vertex))) = state.heap.pop() {
            let distance = f64::from_bits(bits);
            if distance > state.distance(vertex) {
                continue;
            }
            settled_a |= vertex == target_a;
            settled_b |= vertex == target_b;
            if settled_a && settled_b {
                break;
            }
            let start = adjacency_offsets[vertex as usize] as usize;
            let end = adjacency_offsets[vertex as usize + 1] as usize;
            for slot in start..end {
                let target = adj_targets[slot];
                let next = distance + adj_meters[slot];
                if next < state.distance(target) {
                    state.distances.insert(target, next);
                    state.previous.insert(target, (vertex, adj_edges[slot]));
                    state.heap.push(Reverse((next.to_bits(), target)));
                }
            }
        }
    }

    /// The `(lon, lat)` point at a fraction of an edge's true length.
    fn point_at(&self, edge: u32, fraction: f64) -> (f64, f64) {
        let start = self.arrays.coordinate_offsets()[edge as usize] as usize;
        let end = self.arrays.coordinate_offsets()[edge as usize + 1] as usize;
        let total = self.along(end - 1);
        self.interpolate(start, end, fraction.clamp(0.0, 1.0) * total)
    }

    /// The `(lon, lat)` geometry of an edge between two fractions of its true
    /// length, endpoints interpolated; reversed when `from_fraction >
    /// to_fraction`. Always at least one point.
    fn edge_slice(&self, edge: u32, from_fraction: f64, to_fraction: f64) -> Vec<(f64, f64)> {
        let start = self.arrays.coordinate_offsets()[edge as usize] as usize;
        let end = self.arrays.coordinate_offsets()[edge as usize + 1] as usize;
        let total = self.along(end - 1);
        let (low, high, reversed) = if from_fraction <= to_fraction {
            (from_fraction, to_fraction, false)
        } else {
            (to_fraction, from_fraction, true)
        };
        let low = low.clamp(0.0, 1.0) * total;
        let high = high.clamp(0.0, 1.0) * total;
        let mut slice = Vec::new();
        slice.push(self.interpolate(start, end, low));
        for point in start..end {
            let along = self.along(point);
            if along > low && along < high {
                slice.push(self.coordinate(point));
            }
        }
        if high > low {
            slice.push(self.interpolate(start, end, high));
        }
        if reversed {
            slice.reverse();
        }
        slice
    }

    /// The `(lon, lat)` point at along-edge distance `target` (metres from the
    /// edge's first coordinate), interpolated within the containing segment.
    fn interpolate(&self, start: usize, end: usize, target: f64) -> (f64, f64) {
        let total = self.along(end - 1);
        let target = target.clamp(0.0, total);
        // Largest coordinate index whose cumulative distance is at most the
        // target; the containing segment is `[lo, lo + 1]`.
        let (mut lo, mut hi) = (start, end - 1);
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if self.along(mid) <= target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let span = self.along(lo + 1) - self.along(lo);
        let t = if span > 0.0 {
            (target - self.along(lo)) / span
        } else {
            0.0
        };
        let (lon_lo, lat_lo) = self.coordinate(lo);
        let (lon_hi, lat_hi) = self.coordinate(lo + 1);
        (
            lon_lo + t * (lon_hi - lon_lo),
            lat_lo + t * (lat_hi - lat_lo),
        )
    }

    /// Shortest distances in meters from the source frontier into the
    /// search state; only vertices within `cutoff` are reached.
    fn bounded_dijkstra(&self, sources: &[(u32, f64)], cutoff: f64, state: &mut SearchState) {
        state.clear();
        let adjacency_offsets = self.arrays.adjacency_offsets();
        let adj_targets = self.arrays.adj_targets();
        let adj_meters = self.arrays.adj_meters();
        for &(vertex, distance) in sources {
            if distance <= cutoff + 1e-9 && distance < state.distance(vertex) {
                state.distances.insert(vertex, distance);
                state.heap.push(Reverse((distance.to_bits(), vertex)));
            }
        }
        while let Some(Reverse((bits, vertex))) = state.heap.pop() {
            let distance = f64::from_bits(bits);
            if distance > state.distance(vertex) {
                continue;
            }
            let start = adjacency_offsets[vertex as usize] as usize;
            let end = adjacency_offsets[vertex as usize + 1] as usize;
            for slot in start..end {
                let target = adj_targets[slot];
                let next = distance + adj_meters[slot];
                if next <= cutoff + 1e-9 && next < state.distance(target) {
                    state.distances.insert(target, next);
                    state.heap.push(Reverse((next.to_bits(), target)));
                }
            }
        }
    }
}

/// A [`StreetNetwork`]'s state split for serialization: the flat POD
/// arrays a container stores raw, plus the small link records and scalars
/// it stores decoded. The spatial index rides along as plain arrays;
/// nothing is rebuilt on adoption except the L-sized vertex→link index.
#[derive(Debug, PartialEq)]
pub struct StreetNetworkParts {
    pub vertex_count: u32,
    pub adjacency_offsets: Vec<u32>,
    pub adj_targets: Vec<u32>,
    pub adj_meters: Vec<f64>,
    pub adj_edges: Vec<u32>,
    /// Two entries (`from`, `to`) per edge.
    pub endpoints: Vec<u32>,
    pub lengths: Vec<f64>,
    pub coordinate_offsets: Vec<u32>,
    /// Fixed-point degrees × 10⁷.
    pub lons: Vec<i32>,
    pub lats: Vec<i32>,
    pub cumulative: Vec<f32>,
    /// Flattened index boxes, four entries per node.
    pub index_boxes: Vec<i32>,
    /// Flattened leaf payloads, two entries per leaf.
    pub index_payload: Vec<u32>,
    pub index_level_starts: Vec<u32>,
    pub links: Vec<StoredLink>,
}

/// The `(vertex, link index)` pairs behind [`StreetNetwork::links_at`],
/// sorted by vertex: each link listed under both endpoints of its edge,
/// once when they coincide. The endpoints come from the links themselves,
/// so this rebuilds without touching any street-sized array.
fn build_vertex_links(links: &[StoredLink]) -> Vec<(u32, u32)> {
    let mut vertex_links = Vec::with_capacity(links.len() * 2);
    for (index, link) in links.iter().enumerate() {
        vertex_links.push((link.from, index as u32));
        if link.to != link.from {
            vertex_links.push((link.to, index as u32));
        }
    }
    vertex_links.sort_unstable();
    vertex_links
}

/// Drops consecutive duplicate points, keeping at least two.
fn dedup_consecutive(path: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    let mut coordinates: Vec<(f64, f64)> = Vec::with_capacity(path.len());
    for point in path {
        if coordinates.last() != Some(&point) {
            coordinates.push(point);
        }
    }
    if coordinates.len() == 1 {
        coordinates.push(coordinates[0]);
    }
    coordinates
}

/// The lon/lat envelope containing every segment within `max_snap_distance` of
/// a query, sized at the query's own latitude. The latitude half-width uses a
/// global minimum metres-per-degree-latitude; the longitude half-width uses
/// the minimum metres-per-degree-longitude over the reachable latitude band,
/// so no truly-nearby segment is clipped at any latitude or snap distance.
fn snap_envelope(latitude: f64, longitude: f64, max_snap_distance: f64) -> Envelope {
    // Metres per degree of latitude bottoms out at the equator.
    const MIN_MPD_LAT: f64 = 110_574.0;
    let margin = 1.0 + 1e-6;
    let delta_lat = max_snap_distance / MIN_MPD_LAT * margin;
    let lo_lat = (latitude - delta_lat).clamp(-90.0, 90.0);
    let hi_lat = (latitude + delta_lat).clamp(-90.0, 90.0);
    let min_mpd_lon = meters_per_degree(lo_lat).0.min(meters_per_degree(hi_lat).0);
    let delta_lon = if min_mpd_lon > 1e-9 {
        (max_snap_distance / min_mpd_lon * margin).min(180.0)
    } else {
        180.0
    };
    // Rounded outward onto the fixed-point grid, so the envelope stays a
    // superset of the true one.
    let outward = |degrees: f64, up: bool| -> i32 {
        let scaled = degrees * COORDINATE_SCALE;
        let rounded = if up { scaled.ceil() } else { scaled.floor() };
        rounded.clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32
    };
    [
        outward(longitude - delta_lon, false),
        outward(latitude - delta_lat, false),
        outward(longitude + delta_lon, true),
        outward(latitude + delta_lat, true),
    ]
}

/// Shortest signed longitude difference in degrees, wrapped to `[-180, 180]`
/// so a pair straddling the antimeridian measures the short way.
fn longitude_delta(from: f64, to: f64) -> f64 {
    let delta = (to - from) % 360.0;
    if delta > 180.0 {
        delta - 360.0
    } else if delta < -180.0 {
        delta + 360.0
    } else {
        delta
    }
}

/// The true geometric length between two lon/lat points, in metres, using a
/// local `cos(latitude)` at their midpoint (exact for a short segment).
fn segment_length(lon_a: f64, lat_a: f64, lon_b: f64, lat_b: f64) -> f64 {
    let (mpd_lon, mpd_lat) = meters_per_degree((lat_a + lat_b) / 2.0);
    let dx = longitude_delta(lon_a, lon_b) * mpd_lon;
    let dy = (lat_b - lat_a) * mpd_lat;
    (dx * dx + dy * dy).sqrt()
}

/// Splits every segment longer than `MAX_SEGMENT_METERS` into equal colinear
/// pieces, returning the densified coordinate offsets, geographic coordinates,
/// and per-coordinate cumulative distance from each edge's first point.
fn densify(
    coordinate_offsets: &[u32],
    longitudes: &[i32],
    latitudes: &[i32],
) -> (Vec<u32>, Vec<i32>, Vec<i32>, Vec<f32>) {
    let edge_count = coordinate_offsets.len().saturating_sub(1);
    let mut offsets = Vec::with_capacity(coordinate_offsets.len());
    let mut lons = Vec::new();
    let mut lats = Vec::new();
    let mut cumulative = Vec::new();
    // Inserted points re-quantize onto the grid (≤ ~0.8 cm each), so the
    // split targets a hair under the maximum and no sub-segment exceeds it.
    let target = MAX_SEGMENT_METERS - QUANTIZATION_GUARD_METERS;
    offsets.push(0);
    for edge in 0..edge_count {
        let start = coordinate_offsets[edge] as usize;
        let end = coordinate_offsets[edge + 1] as usize;
        lons.push(longitudes[start]);
        lats.push(latitudes[start]);
        cumulative.push(0.0f32);
        let mut running = 0.0f64;
        for point in start..end - 1 {
            let (lon_a, lat_a) = (degrees(longitudes[point]), degrees(latitudes[point]));
            let (lon_b, lat_b) = (
                degrees(longitudes[point + 1]),
                degrees(latitudes[point + 1]),
            );
            // Bound each sub-piece by the largest metres-per-degree over the
            // segment's latitude band, so none exceeds MAX_SEGMENT_METERS even
            // when the segment spans a wide latitude range. Longitude
            // metres-per-degree peaks toward the equator, latitude toward the
            // poles.
            let mut max_mpd_lon = meters_per_degree(lat_a).0.max(meters_per_degree(lat_b).0);
            if (lat_a <= 0.0) != (lat_b <= 0.0) {
                max_mpd_lon = max_mpd_lon.max(meters_per_degree(0.0).0);
            }
            let max_mpd_lat = meters_per_degree(lat_a).1.max(meters_per_degree(lat_b).1);
            let dx = longitude_delta(lon_a, lon_b).abs() * max_mpd_lon;
            let dy = (lat_b - lat_a).abs() * max_mpd_lat;
            let pieces = ((dx * dx + dy * dy).sqrt() / target).ceil().max(1.0) as usize;
            for k in 1..=pieces {
                let t = k as f64 / pieces as f64;
                let lon = if k == pieces {
                    longitudes[point + 1]
                } else {
                    quantize(lon_a + t * (lon_b - lon_a))
                };
                let lat = if k == pieces {
                    latitudes[point + 1]
                } else {
                    quantize(lat_a + t * (lat_b - lat_a))
                };
                let (prev_lon, prev_lat) = (*lons.last().unwrap(), *lats.last().unwrap());
                running += segment_length(
                    degrees(prev_lon),
                    degrees(prev_lat),
                    degrees(lon),
                    degrees(lat),
                );
                lons.push(lon);
                lats.push(lat);
                cumulative.push(running as f32);
            }
        }
        offsets.push(lons.len() as u32);
    }
    (offsets, lons, lats, cumulative)
}

/// Local meters per degree of (longitude, latitude) on the WGS84 spheroid.
fn meters_per_degree(latitude: f64) -> (f64, f64) {
    let phi = latitude.to_radians();
    let meters_per_lat = 111_132.954 - 559.822 * (2.0 * phi).cos() + 1.175 * (4.0 * phi).cos();
    let meters_per_lon =
        111_412.84 * phi.cos() - 93.5 * (3.0 * phi).cos() + 0.118 * (5.0 * phi).cos();
    (meters_per_lon, meters_per_lat)
}

/// Conservative rounding of a duration to whole seconds: up, with a small
/// tolerance for floating-point noise.
fn seconds(duration: f64) -> u32 {
    (duration - 1e-6).ceil().max(0.0) as u32
}

#[cfg(test)]
#[path = "streets_tests.rs"]
mod tests;
