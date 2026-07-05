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

/// Children per packed-index node.
const INDEX_NODE_SIZE: usize = 16;

/// A lon/lat bounding box: `[min_lon, min_lat, max_lon, max_lat]`.
type Envelope = [f64; 4];

fn envelopes_intersect(a: &Envelope, b: &Envelope) -> bool {
    a[0] <= b[2] && b[0] <= a[2] && a[1] <= b[3] && b[1] <= a[3]
}

/// A packed static spatial index over the densified edge segments: leaf
/// boxes sorted by the Hilbert position of their segment's midpoint, parent
/// levels packed bottom-up over runs of `INDEX_NODE_SIZE` children — flat
/// arrays with an implicit tree layout, built once and never mutated.
/// Queried by envelope intersection only.
#[derive(Debug)]
struct PackedIndex {
    /// Node boxes: the leaves first (Hilbert order), then each parent
    /// level, the root last.
    boxes: Vec<Envelope>,
    /// Per-leaf `(edge index, index of the segment's first coordinate)`,
    /// parallel to the leaf boxes.
    payload: Vec<(u32, u32)>,
    /// Start of each level in `boxes` (leaves at 0), plus a tail.
    level_starts: Vec<u32>,
}

impl PackedIndex {
    /// Collects the payloads of every leaf whose box intersects the
    /// envelope, sorted by payload so callers see a traversal-free order.
    fn query_into(&self, envelope: &Envelope, matches: &mut Vec<(u32, u32)>) {
        matches.clear();
        if self.payload.is_empty() {
            return;
        }
        let levels = self.level_starts.len() - 1;
        // (global node index, level), starting at the root.
        let mut stack = vec![(self.boxes.len() - 1, levels - 1)];
        while let Some((node, level)) = stack.pop() {
            if !envelopes_intersect(&self.boxes[node], envelope) {
                continue;
            }
            if level == 0 {
                matches.push(self.payload[node]);
                continue;
            }
            // A node's children sit in the level below, in its own run.
            let position = node - self.level_starts[level] as usize;
            let start = self.level_starts[level - 1] as usize + position * INDEX_NODE_SIZE;
            let end = (start + INDEX_NODE_SIZE).min(self.level_starts[level] as usize);
            for child in start..end {
                stack.push((child, level - 1));
            }
        }
        matches.sort_unstable();
    }
}

/// Builds the packed index over a densified polyline set.
fn build_index(coordinate_offsets: &[u32], lons: &[f64], lats: &[f64]) -> PackedIndex {
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
                grid_position((lon_a + lon_b) / 2.0, bounds[0], bounds[2]),
                grid_position((lat_a + lat_b) / 2.0, bounds[1], bounds[3]),
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
    let mut level_starts = vec![0u32];
    let mut total = count;
    let mut level_size = count;
    while level_size > 1 {
        level_starts.push(total as u32);
        level_size = level_size.div_ceil(INDEX_NODE_SIZE);
        total += level_size;
    }
    if count == 0 {
        return PackedIndex {
            boxes: Vec::new(),
            payload: Vec::new(),
            level_starts: vec![0, 0],
        };
    }
    if count == 1 {
        // A single leaf is its own root: one leaf-only level.
        let (_, tag, envelope) = items[0];
        return PackedIndex {
            boxes: vec![envelope],
            payload: vec![tag],
            level_starts: vec![0, 1],
        };
    }
    level_starts.push(total as u32);

    let mut boxes = Vec::with_capacity(total);
    let mut payload = Vec::with_capacity(count);
    for (_, tag, envelope) in items {
        boxes.push(envelope);
        payload.push(tag);
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

/// The `[min_lon, min_lat, max_lon, max_lat]` bounds of a coordinate set.
fn coordinate_bounds(lons: &[f64], lats: &[f64]) -> Envelope {
    let mut bounds = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for (&lon, &lat) in lons.iter().zip(lats) {
        bounds[0] = bounds[0].min(lon);
        bounds[1] = bounds[1].min(lat);
        bounds[2] = bounds[2].max(lon);
        bounds[3] = bounds[3].max(lat);
    }
    bounds
}

/// A coordinate's cell on a 2¹⁶-wide grid over `[min, max]`.
fn grid_position(value: f64, min: f64, max: f64) -> u16 {
    if max <= min {
        return 0;
    }
    (((value - min) / (max - min)) * f64::from(u16::MAX)).clamp(0.0, f64::from(u16::MAX)) as u16
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
}

/// The walking street graph with its spatial index and stop links.
#[derive(Debug)]
pub struct StreetNetwork {
    /// CSR offsets into `adjacency`, one entry per vertex plus a tail.
    adjacency_offsets: Vec<u32>,
    /// Outgoing `(target vertex, meters, edge)` entries; edges appear
    /// in both directions.
    adjacency: Vec<(u32, f64, u32)>,
    /// Edge endpoints, one entry per input edge.
    endpoints: Vec<(u32, u32)>,
    /// Edge cost lengths in meters (the OSM way length, which the split
    /// pro-rating distributes; it may differ from the geometric length).
    lengths: Vec<f64>,
    /// Offsets into the coordinate arrays, one per edge plus a tail. The
    /// geometry is densified so every segment is at most
    /// `MAX_SEGMENT_METERS`.
    coordinate_offsets: Vec<u32>,
    /// Edge geometries in geographic coordinates (longitude, latitude).
    lons: Vec<f64>,
    lats: Vec<f64>,
    /// Per-coordinate cumulative true distance from the edge's first point,
    /// in meters; parallel to `lons`/`lats`. The last point of each edge
    /// holds the edge's total geometric length.
    cumulative: Vec<f64>,
    /// How each snapped stop enters the graph.
    links: Vec<StopLink>,
    /// `(vertex, link index)` pairs sorted by vertex — every link listed
    /// under both endpoints of its edge — so a search finds the links
    /// near its reached vertices without scanning all links.
    vertex_links: Vec<(u32, u32)>,
    /// Spatial index over the edge segments in lon/lat, for envelope
    /// queries only.
    index: PackedIndex,
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
        let bounds = coordinate_bounds(longitudes, latitudes);
        let mut order: Vec<u32> = (0..edges.len() as u32).collect();
        type EdgeKey = (u64, u64, u64, u64, u64, u64, u32, u32);
        let keys: Vec<EdgeKey> = (0..edges.len())
            .map(|edge| {
                let first = coordinate_offsets[edge] as usize;
                let last = coordinate_offsets[edge + 1] as usize - 1;
                let (from, to, meters) = edges[edge];
                (
                    hilbert(
                        grid_position(longitudes[first], bounds[0], bounds[2]),
                        grid_position(latitudes[first], bounds[1], bounds[3]),
                    ),
                    longitudes[first].to_bits(),
                    latitudes[first].to_bits(),
                    longitudes[last].to_bits(),
                    latitudes[last].to_bits(),
                    meters.to_bits(),
                    from,
                    to,
                )
            })
            .collect();
        let geometry_bits = |edge: u32| {
            let start = coordinate_offsets[edge as usize] as usize;
            let end = coordinate_offsets[edge as usize + 1] as usize;
            longitudes[start..end]
                .iter()
                .zip(&latitudes[start..end])
                .map(|(&lon, &lat)| (lon.to_bits(), lat.to_bits()))
        };
        order.sort_unstable_by(|&a, &b| {
            keys[a as usize]
                .cmp(&keys[b as usize])
                .then_with(|| geometry_bits(a).cmp(geometry_bits(b)))
        });

        let mut edge_map = vec![0u32; edges.len()];
        let mut vertex_map = vec![u32::MAX; vertex_count as usize];
        let mut next_vertex = 0u32;
        let mut permuted_edges = Vec::with_capacity(edges.len());
        let mut permuted_offsets = Vec::with_capacity(edges.len() + 1);
        let mut permuted_lons = Vec::with_capacity(longitudes.len());
        let mut permuted_lats = Vec::with_capacity(latitudes.len());
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
            permuted_lons.extend_from_slice(&longitudes[start..end]);
            permuted_lats.extend_from_slice(&latitudes[start..end]);
            permuted_offsets.push(permuted_lons.len() as u32);
        }
        // Vertices no edge touches keep ids after the connected ones.
        for slot in vertex_map.iter_mut() {
            if *slot == u32::MAX {
                *slot = next_vertex;
                next_vertex += 1;
            }
        }
        let links: Vec<StopLink> = links
            .into_iter()
            .map(|link| StopLink {
                edge: edge_map[link.edge as usize],
                ..link
            })
            .collect();
        let edges = permuted_edges;

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
        let mut adjacency = vec![(0u32, 0.0, 0u32); edges.len() * 2];
        let mut cursor = adjacency_offsets.clone();
        for (index, &(from, to, meters)) in edges.iter().enumerate() {
            for (a, b) in [(from, to), (to, from)] {
                let slot = cursor[a as usize] as usize;
                adjacency[slot] = (b, meters, index as u32);
                cursor[a as usize] += 1;
            }
        }

        let index = build_index(&dense_offsets, &lons, &lats);
        let endpoints: Vec<(u32, u32)> = edges.iter().map(|&(from, to, _)| (from, to)).collect();
        let vertex_links = build_vertex_links(&links, &endpoints);

        Ok(StreetNetwork {
            adjacency_offsets,
            adjacency,
            endpoints,
            lengths: edges.iter().map(|&(_, _, meters)| meters).collect(),
            coordinate_offsets: dense_offsets,
            lons,
            lats,
            cumulative,
            links,
            vertex_links,
            index,
        })
    }

    /// Number of street vertices.
    pub fn vertex_count(&self) -> u32 {
        self.adjacency_offsets.len() as u32 - 1
    }

    /// Number of street edges.
    pub fn edge_count(&self) -> u32 {
        self.endpoints.len() as u32
    }

    /// Number of stop links.
    pub fn link_count(&self) -> usize {
        self.links.len()
    }

    /// The network's serializable state — everything but the spatial
    /// index, which `from_parts` rebuilds from the geometry.
    pub fn to_parts(&self) -> StreetNetworkParts {
        StreetNetworkParts {
            adjacency_offsets: self.adjacency_offsets.clone(),
            adjacency: self.adjacency.clone(),
            endpoints: self.endpoints.clone(),
            lengths: self.lengths.clone(),
            coordinate_offsets: self.coordinate_offsets.clone(),
            lons: self.lons.clone(),
            lats: self.lats.clone(),
            cumulative: self.cumulative.clone(),
            links: self.links.clone(),
        }
    }

    /// Rebuilds a network from its serialized parts. The stored edge order
    /// is adopted as-is — new saves carry the Hilbert layout; older
    /// artifacts keep their original order, correct either way.
    pub fn from_parts(parts: StreetNetworkParts) -> StreetNetwork {
        let index = build_index(&parts.coordinate_offsets, &parts.lons, &parts.lats);
        let vertex_links = build_vertex_links(&parts.links, &parts.endpoints);
        StreetNetwork {
            adjacency_offsets: parts.adjacency_offsets,
            adjacency: parts.adjacency,
            endpoints: parts.endpoints,
            lengths: parts.lengths,
            coordinate_offsets: parts.coordinate_offsets,
            lons: parts.lons,
            lats: parts.lats,
            cumulative: parts.cumulative,
            links: parts.links,
            vertex_links,
            index,
        }
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
        self.index.query_into(&envelope, &mut candidates);
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
        let (ax, ay) = to_xy(self.lons[a], self.lats[a]);
        let (bx, by) = to_xy(self.lons[b], self.lats[b]);
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

        let end = self.coordinate_offsets[edge as usize + 1] as usize;
        let along = self.cumulative[a] + t * (self.cumulative[b] - self.cumulative[a]);
        let total = self.cumulative[end - 1];
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
        let cutoff = max_seconds * walking_speed;
        let (from, to) = self.endpoints[snap.edge as usize];
        let length = self.lengths[snap.edge as usize];
        SEARCH_STATE.with(|cell| {
            let state = &mut cell.borrow_mut();
            self.bounded_dijkstra(
                &[
                    (from, snap.connector + snap.fraction * length),
                    (to, snap.connector + (1.0 - snap.fraction) * length),
                ],
                cutoff,
                state,
            );
            // The candidate links near the search: those at reached
            // vertices, plus those on the snapped edge itself, whose
            // direct on-edge path can be walkable even when neither
            // endpoint is within the cutoff. Sorted so processing order
            // is independent of hash-map iteration order.
            let mut candidates: Vec<u32> = Vec::new();
            for vertex in [from, to] {
                candidates.extend(self.links_at(vertex));
            }
            for &vertex in state.distances.keys() {
                candidates.extend(self.links_at(vertex));
            }
            candidates.sort_unstable();
            candidates.dedup();
            let mut nearest: HashMap<StopIdx, f64> = HashMap::new();
            for index in candidates {
                let link = &self.links[index as usize];
                let (link_from, link_to) = self.endpoints[link.edge as usize];
                let link_length = self.lengths[link.edge as usize];
                let mut meters = f64::min(
                    state.distance(link_from) + link.fraction * link_length,
                    state.distance(link_to) + (1.0 - link.fraction) * link_length,
                ) + link.connector;
                if link.edge == snap.edge {
                    // Both points sit on one edge: walking between the snap
                    // points never has to reach an endpoint.
                    let direct = snap.connector
                        + (link.fraction - snap.fraction).abs() * length
                        + link.connector;
                    meters = meters.min(direct);
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
            Some(reached)
        })
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
        let from_length = self.lengths[from.edge as usize];
        let to_length = self.lengths[to.edge as usize];
        // The direct candidate: both points on one edge.
        let direct = (from.edge == to.edge).then(|| {
            from.connector + (from.fraction - to.fraction).abs() * from_length + to.connector
        });

        let (from_u, from_v) = self.endpoints[from.edge as usize];
        let (to_u, to_v) = self.endpoints[to.edge as usize];
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
                let (u, _) = self.endpoints[edge as usize];
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
            let start = self.adjacency_offsets[vertex as usize] as usize;
            let end = self.adjacency_offsets[vertex as usize + 1] as usize;
            for &(target, meters, edge) in &self.adjacency[start..end] {
                let next = distance + meters;
                if next < state.distance(target) {
                    state.distances.insert(target, next);
                    state.previous.insert(target, (vertex, edge));
                    state.heap.push(Reverse((next.to_bits(), target)));
                }
            }
        }
    }

    /// The `(lon, lat)` point at a fraction of an edge's true length.
    fn point_at(&self, edge: u32, fraction: f64) -> (f64, f64) {
        let start = self.coordinate_offsets[edge as usize] as usize;
        let end = self.coordinate_offsets[edge as usize + 1] as usize;
        let total = self.cumulative[end - 1];
        self.interpolate(start, end, fraction.clamp(0.0, 1.0) * total)
    }

    /// The `(lon, lat)` geometry of an edge between two fractions of its true
    /// length, endpoints interpolated; reversed when `from_fraction >
    /// to_fraction`. Always at least one point.
    fn edge_slice(&self, edge: u32, from_fraction: f64, to_fraction: f64) -> Vec<(f64, f64)> {
        let start = self.coordinate_offsets[edge as usize] as usize;
        let end = self.coordinate_offsets[edge as usize + 1] as usize;
        let total = self.cumulative[end - 1];
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
            let along = self.cumulative[point];
            if along > low && along < high {
                slice.push((self.lons[point], self.lats[point]));
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
        let total = self.cumulative[end - 1];
        let target = target.clamp(0.0, total);
        // Largest coordinate index whose cumulative distance is at most the
        // target; the containing segment is `[lo, lo + 1]`.
        let (mut lo, mut hi) = (start, end - 1);
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if self.cumulative[mid] <= target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let span = self.cumulative[lo + 1] - self.cumulative[lo];
        let t = if span > 0.0 {
            (target - self.cumulative[lo]) / span
        } else {
            0.0
        };
        (
            self.lons[lo] + t * (self.lons[lo + 1] - self.lons[lo]),
            self.lats[lo] + t * (self.lats[lo + 1] - self.lats[lo]),
        )
    }

    /// Shortest distances in meters from the source frontier into the
    /// search state; only vertices within `cutoff` are reached.
    fn bounded_dijkstra(&self, sources: &[(u32, f64)], cutoff: f64, state: &mut SearchState) {
        state.clear();
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
            let start = self.adjacency_offsets[vertex as usize] as usize;
            let end = self.adjacency_offsets[vertex as usize + 1] as usize;
            for &(target, meters, _) in &self.adjacency[start..end] {
                let next = distance + meters;
                if next <= cutoff + 1e-9 && next < state.distance(target) {
                    state.distances.insert(target, next);
                    state.heap.push(Reverse((next.to_bits(), target)));
                }
            }
        }
    }
}

/// A [`StreetNetwork`]'s serializable state; see
/// [`StreetNetwork::to_parts`].
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct StreetNetworkParts {
    adjacency_offsets: Vec<u32>,
    adjacency: Vec<(u32, f64, u32)>,
    endpoints: Vec<(u32, u32)>,
    lengths: Vec<f64>,
    coordinate_offsets: Vec<u32>,
    lons: Vec<f64>,
    lats: Vec<f64>,
    cumulative: Vec<f64>,
    links: Vec<StopLink>,
}

/// The `(vertex, link index)` pairs behind [`StreetNetwork::links_at`],
/// sorted by vertex: each link listed under both endpoints of its edge,
/// once when they coincide.
fn build_vertex_links(links: &[StopLink], endpoints: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut vertex_links = Vec::with_capacity(links.len() * 2);
    for (index, link) in links.iter().enumerate() {
        let (from, to) = endpoints[link.edge as usize];
        vertex_links.push((from, index as u32));
        if to != from {
            vertex_links.push((to, index as u32));
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
    [
        longitude - delta_lon,
        latitude - delta_lat,
        longitude + delta_lon,
        latitude + delta_lat,
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
    longitudes: &[f64],
    latitudes: &[f64],
) -> (Vec<u32>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let edge_count = coordinate_offsets.len().saturating_sub(1);
    let mut offsets = Vec::with_capacity(coordinate_offsets.len());
    let mut lons = Vec::new();
    let mut lats = Vec::new();
    let mut cumulative = Vec::new();
    offsets.push(0);
    for edge in 0..edge_count {
        let start = coordinate_offsets[edge] as usize;
        let end = coordinate_offsets[edge + 1] as usize;
        lons.push(longitudes[start]);
        lats.push(latitudes[start]);
        cumulative.push(0.0);
        let mut running = 0.0;
        for point in start..end - 1 {
            let (lon_a, lat_a) = (longitudes[point], latitudes[point]);
            let (lon_b, lat_b) = (longitudes[point + 1], latitudes[point + 1]);
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
            let pieces = ((dx * dx + dy * dy).sqrt() / MAX_SEGMENT_METERS)
                .ceil()
                .max(1.0) as usize;
            for k in 1..=pieces {
                let t = k as f64 / pieces as f64;
                let lon = lon_a + t * (lon_b - lon_a);
                let lat = lat_a + t * (lat_b - lat_a);
                let (prev_lon, prev_lat) = (*lons.last().unwrap(), *lats.last().unwrap());
                running += segment_length(prev_lon, prev_lat, lon, lat);
                lons.push(lon);
                lats.push(lat);
                cumulative.push(running);
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
mod tests {
    use super::*;

    // Synthetic networks live around 24°E 60°N; test coordinates are
    // planar meters converted with the local degree lengths, so designed
    // distances hold to well under the one-second rounding step.
    fn lonlat(x: f64, y: f64) -> (f64, f64) {
        let (per_lon, per_lat) = meters_per_degree(60.0);
        (24.0 + x / per_lon, 60.0 + y / per_lat)
    }

    /// A test edge: `(from, to, meters, path)` with the path in planar
    /// meters.
    type TestEdge = (u32, u32, f64, Vec<(f64, f64)>);

    /// Flat-array network builder.
    fn network(
        vertex_count: u32,
        stop_count: u32,
        edges: &[TestEdge],
        links: Vec<StopLink>,
    ) -> Result<StreetNetwork, StreetError> {
        let mut offsets = vec![0u32];
        let mut longitudes = Vec::new();
        let mut latitudes = Vec::new();
        for (_, _, _, path) in edges {
            for &(x, y) in path {
                let (lon, lat) = lonlat(x, y);
                longitudes.push(lon);
                latitudes.push(lat);
            }
            offsets.push(longitudes.len() as u32);
        }
        let flat: Vec<(u32, u32, f64)> = edges
            .iter()
            .map(|&(from, to, meters, _)| (from, to, meters))
            .collect();
        StreetNetwork::new(
            vertex_count,
            stop_count,
            &flat,
            &offsets,
            &longitudes,
            &latitudes,
            links,
        )
    }

    fn link(stop: u32, edge: u32, fraction: f64, connector: f64) -> StopLink {
        StopLink {
            stop: StopIdx(stop),
            edge,
            fraction,
            connector,
        }
    }

    fn straight(from: (f64, f64), to: (f64, f64)) -> Vec<(f64, f64)> {
        vec![from, to]
    }

    /// The `(stop, seconds)` view of a walking-search result.
    fn timed(walks: &[WalkedStop]) -> Vec<(StopIdx, u32)> {
        walks.iter().map(|walk| (walk.stop, walk.seconds)).collect()
    }

    #[test]
    fn snaps_to_the_nearest_edge() {
        let network = network(
            4,
            0,
            &[
                (0, 1, 400.0, straight((0.0, 0.0), (400.0, 0.0))),
                (2, 3, 400.0, straight((0.0, 100.0), (400.0, 100.0))),
            ],
            vec![],
        )
        .unwrap();
        let (lon, lat) = lonlat(100.0, 10.0);
        let snap = network.snap(lat, lon, 100.0).unwrap();
        assert_eq!(snap.edge, 0);
        assert!((snap.fraction - 0.25).abs() < 1e-6);
        assert!((snap.connector - 10.0).abs() < 0.05);
    }

    #[test]
    fn respects_the_snap_distance() {
        let network = network(
            2,
            0,
            &[(0, 1, 400.0, straight((250.0, 0.0), (250.0, 400.0)))],
            vec![],
        )
        .unwrap();
        let (lon, lat) = lonlat(0.0, 0.0);
        // The nearest edge is found whenever the allowance covers it.
        let snap = network.snap(lat, lon, 300.0).unwrap();
        assert_eq!(snap.edge, 0);
        assert!((snap.connector - 250.0).abs() < 0.1);
        assert_eq!(network.snap(lat, lon, 200.0), None);
        assert_eq!(network.access_stops(lat, lon, 1.0, 600.0, 200.0), None);
    }

    #[test]
    fn ignores_out_of_range_query_parameters() {
        let network = network(
            2,
            1,
            &[(0, 1, 400.0, straight((0.0, 0.0), (400.0, 0.0)))],
            vec![link(0, 0, 0.5, 0.0)],
        )
        .unwrap();
        let (lon, lat) = lonlat(100.0, 0.0);
        assert_eq!(network.snap(f64::NAN, lon, 100.0), None);
        assert_eq!(network.snap(lat, f64::INFINITY, 100.0), None);
        assert_eq!(network.snap(lat, lon, f64::NAN), None);
        assert_eq!(network.snap(lat, lon, f64::INFINITY), None);
        assert_eq!(network.snap(lat, lon, -1.0), None);
        assert_eq!(network.access_stops(lat, lon, f64::NAN, 600.0, 100.0), None);
        assert_eq!(network.access_stops(lat, lon, 0.0, 600.0, 100.0), None);
        assert_eq!(
            network.access_stops(lat, lon, f64::INFINITY, 600.0, 100.0),
            None
        );
        assert_eq!(network.access_stops(lat, lon, 1.0, f64::NAN, 100.0), None);
        assert_eq!(network.access_stops(lat, lon, 1.0, -5.0, 100.0), None);
    }

    #[test]
    fn indexes_long_diagonal_edges() {
        // The index holds one entry per polyline segment, so even a
        // 25 km diagonal is found exactly from a query at its middle.
        let network = network(
            2,
            0,
            &[(0, 1, 25_000.0, straight((0.0, 0.0), (20_000.0, 15_000.0)))],
            vec![],
        )
        .unwrap();
        // 50 m perpendicular to the segment's midpoint.
        let (lon, lat) = lonlat(10_000.0 - 30.0, 7_500.0 + 40.0);
        let snap = network.snap(lat, lon, 100.0).unwrap();
        assert_eq!(snap.edge, 0);
        assert!((snap.connector - 50.0).abs() < 0.5);
        assert!((snap.fraction - 0.5).abs() < 1e-3);
    }

    #[test]
    fn survives_huge_snap_allowances() {
        let network = network(
            2,
            1,
            &[(0, 1, 400.0, straight((0.0, 0.0), (400.0, 0.0)))],
            vec![link(0, 0, 0.5, 0.0)],
        )
        .unwrap();
        // The allowance only filters the result, so a finite but absurd
        // value costs nothing and stays correct.
        let (lon, lat) = lonlat(100.0, 0.0);
        let snap = network.snap(lat, lon, 1e12).unwrap();
        assert_eq!(snap.edge, 0);
        assert!(snap.connector < 0.01);
        // Queries far outside the indexed extent behave the same.
        let (far_lon, far_lat) = lonlat(5_000_000.0, 0.0);
        let far = network.snap(far_lat, far_lon, 1e12).unwrap();
        assert_eq!(far.edge, 0);
        assert!(far.connector > 1_000_000.0);
    }

    #[test]
    fn snaps_accurately_across_a_wide_latitude_range() {
        // Two short edges, one at 60°N and one at 70°N. Each snap must
        // measure its connector with the local scale at its own latitude —
        // a single network-mean projection would be ~24% wrong at 70°N.
        let mpd_lon_60 = meters_per_degree(60.0).0;
        let mpd_lon_70 = meters_per_degree(70.0).0;
        let longitudes = [25.0, 25.0, 25.0, 25.0];
        let latitudes = [60.0, 60.01, 70.0, 70.01];
        let offsets = [0u32, 2, 4];
        let edges = [(0u32, 1u32, 1000.0), (2u32, 3u32, 1000.0)];
        let network =
            StreetNetwork::new(4, 0, &edges, &offsets, &longitudes, &latitudes, vec![]).unwrap();

        // 30 m due east of each edge's midpoint snaps at a ~30 m connector,
        // even though 30 m is a different Δlon at each latitude.
        let north = network
            .snap(70.005, 25.0 + 30.0 / mpd_lon_70, 100.0)
            .unwrap();
        assert_eq!(north.edge, 1);
        assert!((north.connector - 30.0).abs() < 0.1, "{}", north.connector);
        assert!((north.fraction - 0.5).abs() < 0.01);

        let south = network
            .snap(60.005, 25.0 + 30.0 / mpd_lon_60, 100.0)
            .unwrap();
        assert_eq!(south.edge, 0);
        assert!((south.connector - 30.0).abs() < 0.1, "{}", south.connector);
    }

    #[test]
    fn densifies_long_segments() {
        // A single 5 km edge is split so every stored segment is short.
        let mpd_lat = meters_per_degree(60.0).1;
        let span = 5_000.0 / mpd_lat;
        let network = StreetNetwork::new(
            2,
            0,
            &[(0u32, 1u32, 5_000.0)],
            &[0u32, 2],
            &[25.0, 25.0],
            &[60.0, 60.0 + span],
            vec![],
        )
        .unwrap();
        let count = network.coordinate_offsets[1] as usize;
        assert!(count >= 51, "expected >=51 densified points, got {count}");
        for pair in network.lats.windows(2) {
            let seg = segment_length(25.0, pair[0], 25.0, pair[1]);
            assert!(seg <= MAX_SEGMENT_METERS + 1e-6, "segment {seg} m too long");
        }
        // Midpoint of the edge is 2500 m along.
        let (_, lat) = network.point_at(0, 0.5);
        assert!((segment_length(25.0, 60.0, 25.0, lat) - 2_500.0).abs() < 1.0);
    }

    #[test]
    fn wraps_longitude_across_the_antimeridian() {
        assert!((longitude_delta(179.99, -179.99) - 0.02).abs() < 1e-9);
        assert!((longitude_delta(-179.99, 179.99) + 0.02).abs() < 1e-9);
        assert!((longitude_delta(10.0, 20.0) - 10.0).abs() < 1e-9);
        // A short segment straddling ±180° measures short, not near-global.
        assert!(segment_length(179.99, 0.0, -179.99, 0.0) < 3_000.0);
    }

    #[test]
    fn densifies_wide_latitude_segments() {
        // Equator to 70°N with some longitude: every sub-piece stays short
        // even though metres-per-degree changes markedly along the segment.
        let network = StreetNetwork::new(
            2,
            0,
            &[(0u32, 1u32, 1000.0)],
            &[0u32, 2],
            &[25.0, 25.5],
            &[0.0, 70.0],
            vec![],
        )
        .unwrap();
        for (lons, lats) in network.lons.windows(2).zip(network.lats.windows(2)) {
            let seg = segment_length(lons[0], lats[0], lons[1], lats[1]);
            assert!(
                seg <= MAX_SEGMENT_METERS + 1e-6,
                "sub-piece {seg} m too long"
            );
        }
    }

    #[test]
    fn walk_paths_follow_the_street() {
        // An L-shaped walk with partial edges at both snap points.
        let network = network(
            3,
            0,
            &[
                (0, 1, 300.0, straight((0.0, 0.0), (300.0, 0.0))),
                (1, 2, 200.0, straight((300.0, 0.0), (300.0, 200.0))),
            ],
            vec![],
        )
        .unwrap();
        let origin = lonlat(100.0, -10.0);
        let target = lonlat(310.0, 100.0);
        let from = network.snap(origin.1, origin.0, 50.0).unwrap();
        let to = network.snap(target.1, target.0, 50.0).unwrap();
        let (path, meters) = network
            .walk_path((origin.1, origin.0), &from, (target.1, target.0), &to)
            .unwrap();
        // 10 m connector + 200 m along the first edge + 100 m up the
        // second + 10 m connector.
        assert!((meters - 320.0).abs() < 0.5);
        let designed = [
            lonlat(100.0, -10.0),
            lonlat(100.0, 0.0),
            lonlat(300.0, 0.0),
            lonlat(300.0, 100.0),
            lonlat(310.0, 100.0),
        ];
        // Densification inserts colinear vertices along the straight edges,
        // so the path passes through the designed corners in order with extra
        // points between them; endpoints match exactly.
        assert_eq!(path.first().copied(), Some(designed[0]), "{path:?}");
        assert_eq!(
            path.last().copied(),
            Some(designed[designed.len() - 1]),
            "{path:?}"
        );
        let mut corner = 0;
        for &point in &path {
            if corner < designed.len()
                && (point.0 - designed[corner].0).abs() < 1e-6
                && (point.1 - designed[corner].1).abs() < 1e-6
            {
                corner += 1;
            }
        }
        assert_eq!(corner, designed.len(), "path {path:?} skips a corner");

        // The same-edge direct case never detours over a vertex.
        let near = lonlat(120.0, 20.0);
        let close = network.snap(near.1, near.0, 50.0).unwrap();
        let (short, direct_meters) = network
            .walk_path((origin.1, origin.0), &from, (near.1, near.0), &close)
            .unwrap();
        assert!((direct_meters - 50.0).abs() < 0.5);
        assert_eq!(short.len(), 4);

        // Disconnected components yield no path.
        let island =
            network // separate component
                .walk_path((origin.1, origin.0), &from, (origin.1, origin.0), &from);
        assert!(island.is_some());
    }

    #[test]
    fn walk_paths_traverse_reversed_edges() {
        // The middle edge is defined against the walking direction, so
        // its geometry must come out reversed.
        let network = network(
            3,
            0,
            &[
                (0, 1, 100.0, straight((0.0, 0.0), (100.0, 0.0))),
                (2, 1, 100.0, straight((200.0, 0.0), (100.0, 0.0))),
            ],
            vec![],
        )
        .unwrap();
        let origin = lonlat(50.0, 0.0);
        let target = lonlat(150.0, 0.0);
        let from = network.snap(origin.1, origin.0, 50.0).unwrap();
        let to = network.snap(target.1, target.0, 50.0).unwrap();
        let (path, meters) = network
            .walk_path((origin.1, origin.0), &from, (target.1, target.0), &to)
            .unwrap();
        assert!((meters - 100.0).abs() < 0.5);
        // Longitudes must increase monotonically along the walk.
        for pair in path.windows(2) {
            assert!(pair[1].0 >= pair[0].0 - 1e-12, "{path:?}");
        }
    }

    #[test]
    fn walk_paths_need_a_connected_street() {
        let network = network(
            4,
            0,
            &[
                (0, 1, 400.0, straight((0.0, 0.0), (400.0, 0.0))),
                (2, 3, 400.0, straight((0.0, 1000.0), (400.0, 1000.0))),
            ],
            vec![],
        )
        .unwrap();
        let origin = lonlat(100.0, 0.0);
        let target = lonlat(100.0, 1000.0);
        let from = network.snap(origin.1, origin.0, 50.0).unwrap();
        let to = network.snap(target.1, target.0, 50.0).unwrap();
        assert!(network
            .walk_path((origin.1, origin.0), &from, (target.1, target.0), &to)
            .is_none());
    }

    #[test]
    fn stop_snaps_prefer_the_nearest_link() {
        let network = network(
            2,
            1,
            &[(0, 1, 400.0, straight((0.0, 0.0), (400.0, 0.0)))],
            vec![link(0, 0, 0.75, 40.0), link(0, 0, 0.25, 10.0)],
        )
        .unwrap();
        let snap = network.stop_snap(StopIdx(0)).unwrap();
        assert!((snap.fraction - 0.25).abs() < 1e-9);
        assert!((snap.connector - 10.0).abs() < 1e-9);
        assert_eq!(network.stop_snap(StopIdx(1)), None);
    }

    #[test]
    fn walk_paths_take_the_short_side_of_a_loop() {
        // A square loop whose endpoints coincide: the walk wraps through
        // the shared vertex, and the drawn sides must be the short ones.
        let network = network(
            1,
            0,
            &[(
                0,
                0,
                400.0,
                vec![
                    (0.0, 0.0),
                    (100.0, 0.0),
                    (100.0, 100.0),
                    (0.0, 100.0),
                    (0.0, 0.0),
                ],
            )],
            vec![],
        )
        .unwrap();
        let origin = lonlat(-10.0, 40.0);
        let target = lonlat(20.0, -10.0);
        let from = network.snap(origin.1, origin.0, 50.0).unwrap();
        let to = network.snap(target.1, target.0, 50.0).unwrap();
        assert!((from.fraction - 0.9).abs() < 1e-6);
        assert!((to.fraction - 0.05).abs() < 1e-6);
        let (path, meters) = network
            .walk_path((origin.1, origin.0), &from, (target.1, target.0), &to)
            .unwrap();
        // 10 m connector + 40 m down + 20 m along + 10 m connector.
        assert!((meters - 80.0).abs() < 0.5, "{meters}");
        let designed = [
            lonlat(-10.0, 40.0),
            lonlat(0.0, 40.0),
            lonlat(0.0, 0.0),
            lonlat(20.0, 0.0),
            lonlat(20.0, -10.0),
        ];
        assert_eq!(path.len(), designed.len(), "{path:?}");
        for (point, expected) in path.iter().zip(designed) {
            assert!((point.0 - expected.0).abs() < 1e-6, "{path:?}");
            assert!((point.1 - expected.1).abs() < 1e-6, "{path:?}");
        }
    }

    #[test]
    fn walks_along_a_shared_edge() {
        // The query and both stops snap onto the same 400 m edge; walking
        // between the snap points never detours over the endpoints.
        let network = network(
            2,
            2,
            &[(0, 1, 400.0, straight((0.0, 0.0), (400.0, 0.0)))],
            vec![link(0, 0, 0.25, 0.0), link(1, 0, 0.75, 0.0)],
        )
        .unwrap();
        let (lon, lat) = lonlat(100.0, 0.0);
        let reached = network.access_stops(lat, lon, 1.0, 600.0, 100.0).unwrap();
        assert_eq!(timed(&reached), vec![(StopIdx(0), 0), (StopIdx(1), 200)]);
        assert!(reached[0].meters.abs() < 0.5);
        assert!((reached[1].meters - 200.0).abs() < 0.5);
    }

    #[test]
    fn walks_a_shared_edge_whose_endpoints_exceed_the_cutoff() {
        // Snap point and stop sit mid-edge on a 2 km edge: both endpoint
        // seeds cost 900/1100 m, beyond the 200 m cutoff, yet the direct
        // on-edge walk (200 m) is within it and must still be found.
        let network = network(
            2,
            1,
            &[(0, 1, 2000.0, straight((0.0, 0.0), (2000.0, 0.0)))],
            vec![link(0, 0, 0.55, 0.0)],
        )
        .unwrap();
        let (lon, lat) = lonlat(900.0, 0.0);
        let reached = network.access_stops(lat, lon, 1.0, 200.0, 100.0).unwrap();
        assert_eq!(timed(&reached), vec![(StopIdx(0), 200)]);
        assert!((reached[0].meters - 200.0).abs() < 0.5);
    }

    #[test]
    fn prorates_split_costs_by_the_edge_length() {
        // The edge's cost length says 800 m although its geometry spans
        // 400 m; pro-rated segments follow the cost length.
        let network = network(
            2,
            1,
            &[(0, 1, 800.0, straight((0.0, 0.0), (400.0, 0.0)))],
            vec![link(0, 0, 0.75, 0.0)],
        )
        .unwrap();
        let (lon, lat) = lonlat(100.0, 0.0);
        let reached = network.access_stops(lat, lon, 1.0, 600.0, 100.0).unwrap();
        assert_eq!(timed(&reached), vec![(StopIdx(0), 400)]);
    }

    #[test]
    fn reaches_stops_through_vertices() {
        // An L-shaped walk: 300 m to the corner, 100 m up the other edge.
        let network = network(
            3,
            1,
            &[
                (0, 1, 300.0, straight((0.0, 0.0), (300.0, 0.0))),
                (1, 2, 200.0, straight((300.0, 0.0), (300.0, 200.0))),
            ],
            vec![link(0, 1, 0.5, 0.0)],
        )
        .unwrap();
        let (lon, lat) = lonlat(0.0, 0.0);
        let reached = network.access_stops(lat, lon, 1.0, 600.0, 100.0).unwrap();
        assert_eq!(timed(&reached), vec![(StopIdx(0), 400)]);
    }

    #[test]
    fn takes_the_cheaper_of_direct_and_detour_paths() {
        // A slow 1000 m edge and a fast 100 m parallel edge between the
        // same vertices: reaching a stop near the slow edge's start from a
        // query near its end is cheaper around the parallel edge (300 m)
        // than straight along the slow edge (800 m).
        let network = network(
            2,
            1,
            &[
                (0, 1, 1000.0, straight((0.0, 0.0), (400.0, 0.0))),
                (0, 1, 100.0, vec![(0.0, 0.0), (200.0, 80.0), (400.0, 0.0)]),
            ],
            vec![link(0, 0, 0.1, 0.0)],
        )
        .unwrap();
        let (lon, lat) = lonlat(360.0, 0.0);
        let reached = network.access_stops(lat, lon, 1.0, 600.0, 50.0).unwrap();
        assert_eq!(timed(&reached), vec![(StopIdx(0), 300)]);
    }

    #[test]
    fn applies_the_walking_time_cutoff() {
        let network = network(
            2,
            2,
            &[(0, 1, 400.0, straight((0.0, 0.0), (400.0, 0.0)))],
            vec![link(0, 0, 0.25, 0.0), link(1, 0, 0.75, 0.0)],
        )
        .unwrap();
        let (lon, lat) = lonlat(0.0, 0.0);
        let reached = network.access_stops(lat, lon, 1.0, 150.0, 100.0).unwrap();
        assert_eq!(timed(&reached), vec![(StopIdx(0), 100)]);
    }

    #[test]
    fn counts_connectors_as_walking() {
        // 10 m to the network, 100 m along it, 20 m out to the stop.
        let network = network(
            2,
            1,
            &[(0, 1, 400.0, straight((0.0, 0.0), (400.0, 0.0)))],
            vec![link(0, 0, 0.5, 20.0)],
        )
        .unwrap();
        let (lon, lat) = lonlat(100.0, 10.0);
        let reached = network.access_stops(lat, lon, 1.0, 600.0, 100.0).unwrap();
        assert_eq!(reached.len(), 1);
        assert!((129..=131).contains(&reached[0].seconds));
    }

    #[test]
    fn rounds_walking_seconds_up() {
        let network = network(
            2,
            1,
            &[(0, 1, 401.0, straight((0.0, 0.0), (400.0, 0.0)))],
            vec![link(0, 0, 0.75, 0.0)],
        )
        .unwrap();
        let (lon, lat) = lonlat(100.0, 0.0);
        let reached = network.access_stops(lat, lon, 1.0, 600.0, 100.0).unwrap();
        // 0.5 × 401 m at 1 m/s is 200.5 s and must not round down; the
        // meters stay exact.
        assert_eq!(timed(&reached), vec![(StopIdx(0), 201)]);
        assert!((reached[0].meters - 200.5).abs() < 1e-9);
    }

    #[test]
    fn keeps_the_fastest_of_duplicate_stop_links() {
        let network = network(
            2,
            1,
            &[(0, 1, 400.0, straight((0.0, 0.0), (400.0, 0.0)))],
            vec![link(0, 0, 0.75, 0.0), link(0, 0, 0.5, 0.0)],
        )
        .unwrap();
        let (lon, lat) = lonlat(100.0, 0.0);
        let reached = network.access_stops(lat, lon, 1.0, 600.0, 100.0).unwrap();
        assert_eq!(timed(&reached), vec![(StopIdx(0), 100)]);
    }

    #[test]
    fn handles_an_empty_network() {
        let network = StreetNetwork::new(0, 5, &[], &[0], &[], &[], vec![]).unwrap();
        assert_eq!(network.snap(60.0, 24.0, 100.0), None);
        assert_eq!(network.access_stops(60.0, 24.0, 1.0, 600.0, 100.0), None);
    }

    #[test]
    fn rejects_inconsistent_input() {
        let edge = |meters| (0u32, 1u32, meters, straight((0.0, 0.0), (400.0, 0.0)));
        assert_eq!(
            StreetNetwork::new(2, 0, &[(0, 1, 400.0)], &[0], &[], &[], vec![]).unwrap_err(),
            StreetError::InvalidOffsets
        );
        assert_eq!(
            StreetNetwork::new(2, 0, &[(0, 1, 400.0)], &[0, 1], &[24.0], &[60.0], vec![])
                .unwrap_err(),
            StreetError::ShortGeometry { edge: 0 }
        );
        assert_eq!(
            StreetNetwork::new(
                2,
                0,
                &[(0, 1, 400.0)],
                &[0, 2],
                &[24.0, f64::NAN],
                &[60.0, 60.0],
                vec![]
            )
            .unwrap_err(),
            StreetError::InvalidCoordinates { edge: 0 }
        );
        assert_eq!(
            network(1, 0, &[edge(400.0)], vec![]).unwrap_err(),
            StreetError::VertexOutOfRange {
                edge: 0,
                vertex_count: 1
            }
        );
        assert_eq!(
            network(2, 0, &[edge(f64::NAN)], vec![]).unwrap_err(),
            StreetError::InvalidLength { edge: 0 }
        );
        assert_eq!(
            network(2, 1, &[edge(400.0)], vec![link(0, 1, 0.5, 0.0)]).unwrap_err(),
            StreetError::LinkEdgeOutOfRange {
                link: 0,
                edge_count: 1
            }
        );
        assert_eq!(
            network(2, 1, &[edge(400.0)], vec![link(1, 0, 0.5, 0.0)]).unwrap_err(),
            StreetError::StopOutOfRange {
                stop: 1,
                stop_count: 1
            }
        );
        assert_eq!(
            network(2, 1, &[edge(400.0)], vec![link(0, 0, 1.5, 0.0)]).unwrap_err(),
            StreetError::InvalidLink { link: 0 }
        );
        assert_eq!(
            network(2, 1, &[edge(400.0)], vec![link(0, 0, 0.5, -1.0)]).unwrap_err(),
            StreetError::InvalidLink { link: 0 }
        );
    }

    /// The inverse Hilbert walk (reference d2xy), for the bijection test.
    fn hilbert_inverse(d: u64) -> (u16, u16) {
        const N: u64 = 1 << 16;
        let (mut x, mut y) = (0u64, 0u64);
        let mut t = d;
        let mut s: u64 = 1;
        while s < N {
            let rx = 1 & (t / 2);
            let ry = 1 & (t ^ rx);
            if ry == 0 {
                if rx == 1 {
                    x = s - 1 - x;
                    y = s - 1 - y;
                }
                std::mem::swap(&mut x, &mut y);
            }
            x += s * rx;
            y += s * ry;
            t /= 4;
            s *= 2;
        }
        (x as u16, y as u16)
    }

    #[test]
    fn hilbert_positions_are_a_bijection() {
        // Round-tripping through the independent inverse walk catches any
        // rotation or accumulation mistake in the forward encoding.
        for x in (0..=u16::MAX).step_by(4099) {
            for y in (0..=u16::MAX).step_by(5273) {
                assert_eq!(hilbert_inverse(hilbert(x, y)), (x, y));
            }
        }
        assert_eq!(hilbert(0, 0), 0);
    }

    #[test]
    fn packed_index_matches_a_linear_scan() {
        // Pseudo-random polylines (fixed LCG seed); every envelope query
        // must return exactly the segments whose boxes intersect it.
        let mut state = 0x2545F4914F6CDD1Du64;
        let mut random = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as f64 / (1u64 << 31) as f64
        };
        let mut offsets = vec![0u32];
        let mut lons = Vec::new();
        let mut lats = Vec::new();
        for _ in 0..120 {
            let points = 2 + (random() * 4.0) as usize;
            for _ in 0..points {
                lons.push(24.0 + random() * 0.5);
                lats.push(60.0 + random() * 0.5);
            }
            offsets.push(lons.len() as u32);
        }
        let index = build_index(&offsets, &lons, &lats);

        let mut scan = Vec::new();
        for edge in 0..offsets.len() - 1 {
            for segment in offsets[edge] as usize..offsets[edge + 1] as usize - 1 {
                scan.push((
                    (edge as u32, segment as u32),
                    [
                        lons[segment].min(lons[segment + 1]),
                        lats[segment].min(lats[segment + 1]),
                        lons[segment].max(lons[segment + 1]),
                        lats[segment].max(lats[segment + 1]),
                    ],
                ));
            }
        }
        let mut matches = Vec::new();
        for _ in 0..200 {
            let (lon, lat) = (24.0 + random() * 0.5, 60.0 + random() * 0.5);
            let (dlon, dlat) = (random() * 0.05, random() * 0.05);
            let envelope = [lon - dlon, lat - dlat, lon + dlon, lat + dlat];
            index.query_into(&envelope, &mut matches);
            let mut expected: Vec<(u32, u32)> = scan
                .iter()
                .filter(|(_, envelope_b)| envelopes_intersect(&envelope, envelope_b))
                .map(|&(tag, _)| tag)
                .collect();
            expected.sort_unstable();
            assert_eq!(matches, expected);
        }
    }

    #[test]
    fn input_edge_order_does_not_change_results() {
        // Two far-apart clusters interleaved in the input: the Hilbert
        // layout normalises both input orders to the same internal one, so
        // every query result — internal ids included — must coincide.
        let edges: Vec<TestEdge> = vec![
            (0, 1, 400.0, straight((0.0, 0.0), (400.0, 0.0))),
            (2, 3, 400.0, straight((5000.0, 5000.0), (5400.0, 5000.0))),
            (1, 4, 400.0, straight((400.0, 0.0), (800.0, 0.0))),
            (3, 5, 400.0, straight((5400.0, 5000.0), (5800.0, 5000.0))),
        ];
        let links = vec![link(0, 0, 0.5, 0.0), link(1, 3, 0.5, 0.0)];
        let forward = network(6, 2, &edges, links.clone()).unwrap();

        let shuffled_edges: Vec<TestEdge> = vec![
            edges[3].clone(),
            edges[1].clone(),
            edges[2].clone(),
            edges[0].clone(),
        ];
        // Links follow their edges to the shuffled positions.
        let shuffled_links = vec![link(0, 3, 0.5, 0.0), link(1, 0, 0.5, 0.0)];
        let shuffled = network(6, 2, &shuffled_edges, shuffled_links).unwrap();

        for &(x, y) in &[(200.0, 10.0), (5600.0, 4990.0), (700.0, -20.0)] {
            let (lon, lat) = lonlat(x, y);
            assert_eq!(
                forward.snap(lat, lon, 100.0),
                shuffled.snap(lat, lon, 100.0)
            );
            assert_eq!(
                forward.access_stops(lat, lon, 1.0, 1200.0, 100.0),
                shuffled.access_stops(lat, lon, 1.0, 1200.0, 100.0)
            );
        }
    }

    #[test]
    fn edges_sharing_a_hilbert_cell_keep_an_input_free_order() {
        // Three edges fan out from one point — identical first coordinate,
        // identical Hilbert key — so the layout's tie-break must come from
        // the edges' own data, never their input position.
        let edges: Vec<TestEdge> = vec![
            (0, 1, 400.0, straight((0.0, 0.0), (400.0, 0.0))),
            (0, 2, 400.0, straight((0.0, 0.0), (0.0, 400.0))),
            (0, 3, 400.0, straight((0.0, 0.0), (-400.0, 0.0))),
        ];
        let links = vec![link(0, 0, 1.0, 0.0), link(1, 1, 1.0, 0.0)];
        let forward = network(4, 2, &edges, links).unwrap();

        let shuffled_edges: Vec<TestEdge> =
            vec![edges[2].clone(), edges[0].clone(), edges[1].clone()];
        let shuffled_links = vec![link(0, 1, 1.0, 0.0), link(1, 2, 1.0, 0.0)];
        let shuffled = network(4, 2, &shuffled_edges, shuffled_links).unwrap();

        for &(x, y) in &[(390.0, 5.0), (-5.0, 390.0), (10.0, 10.0)] {
            let (lon, lat) = lonlat(x, y);
            assert_eq!(
                forward.snap(lat, lon, 100.0),
                shuffled.snap(lat, lon, 100.0)
            );
            assert_eq!(
                forward.access_stops(lat, lon, 1.0, 1200.0, 100.0),
                shuffled.access_stops(lat, lon, 1.0, 1200.0, 100.0)
            );
        }
    }

    #[test]
    fn snaps_on_a_single_segment_network() {
        // One 50 m edge densifies to a single segment: the packed index's
        // one leaf is its own root and must still be found.
        let network = network(
            2,
            1,
            &[(0, 1, 50.0, straight((0.0, 0.0), (50.0, 0.0)))],
            vec![link(0, 0, 1.0, 0.0)],
        )
        .unwrap();
        let (lon, lat) = lonlat(25.0, 5.0);
        let snap = network.snap(lat, lon, 100.0).unwrap();
        assert_eq!(snap.edge, 0);
        assert!((snap.fraction - 0.5).abs() < 1e-6);
        assert!((snap.connector - 5.0).abs() < 0.05);
        let reached = network.access_stops(lat, lon, 1.0, 600.0, 100.0).unwrap();
        assert_eq!(timed(&reached), vec![(StopIdx(0), 30)]);
    }
}
