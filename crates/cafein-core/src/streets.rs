//! The query-time street network for walking access and egress.
//!
//! The Python-side build hands the walking graph over as flat arrays:
//! vertices are implicit indices, edges carry their cost length in meters
//! and their geometry. A query snaps a coordinate to its nearest edge
//! through an R*-tree over the edge segments — a virtual node at the snap
//! point, with the split edge's cost pro-rated by the fraction each side
//! covers — and runs a cutoff-bounded Dijkstra to collect every transit stop
//! reachable on foot, entering stops through the same snap links the
//! footpath precompute used. Walking is undirected, so one search serves
//! access and egress alike.
//!
//! Geometry is stored in geographic coordinates (longitude/latitude); there
//! is no global projection. Distances use a local `cos(latitude)` evaluated
//! at the point's own latitude, so they stay accurate over country-scale
//! latitude ranges (following R5/OpenTripPlanner). Segments are densified to
//! a maximum length at build time, so the local-scale model is exact: the
//! snap foot, the connector distance, and the geometry lengths are all
//! computed in a frame local to the relevant short segment. The R*-tree
//! stores segments in lon/lat and is used only for envelope queries — never
//! metric nearest-neighbour, whose degree-Euclidean distance would be wrong.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use rayon::prelude::*;
use rstar::primitives::{GeomWithData, Line};
use rstar::{RTree, AABB};

use crate::timetable::StopIdx;

/// Longest stored segment, in meters. Segments are densified below this so a
/// single centre-latitude scale represents each one to well under a
/// millimetre even at high latitude.
const MAX_SEGMENT_METERS: f64 = 100.0;

/// One polyline segment in the spatial index (lon/lat), tagged with
/// `(edge index, index of the segment's first coordinate)`.
type EdgeSegment = GeomWithData<Line<[f64; 2]>, (u32, u32)>;

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
    /// Spatial index over the edge segments in lon/lat, for envelope
    /// queries only.
    tree: RTree<EdgeSegment>,
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

        let (dense_offsets, lons, lats, cumulative) =
            densify(coordinate_offsets, longitudes, latitudes);

        let mut adjacency_offsets = vec![0u32; vertex_count as usize + 1];
        for &(from, to, _) in edges {
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

        let tree = build_tree(&dense_offsets, &lons, &lats);

        Ok(StreetNetwork {
            adjacency_offsets,
            adjacency,
            endpoints: edges.iter().map(|&(from, to, _)| (from, to)).collect(),
            lengths: edges.iter().map(|&(_, _, meters)| meters).collect(),
            coordinate_offsets: dense_offsets,
            lons,
            lats,
            cumulative,
            links,
            tree,
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

    /// Rebuilds a network from its serialized parts.
    pub fn from_parts(parts: StreetNetworkParts) -> StreetNetwork {
        let tree = build_tree(&parts.coordinate_offsets, &parts.lons, &parts.lats);
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
            tree,
        }
    }

    /// Snaps a coordinate to its nearest edge within `max_snap_distance`
    /// meters through the segment R*-tree. Non-finite coordinates or a
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
        // envelope; the tree is queried by envelope intersection, never by a
        // degree-Euclidean nearest, and each candidate is re-measured exactly.
        let envelope = snap_envelope(latitude, longitude, max_snap_distance);
        let mut best: Option<Snap> = None;
        for segment in self.tree.locate_in_envelope_intersecting(envelope) {
            let (edge, start) = segment.data;
            let (connector, fraction) = self.foot_on_segment(latitude, longitude, edge, start);
            if connector <= max_snap_distance
                && best.is_none_or(|current| connector < current.connector)
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
        let to_xy = |lon: f64, lat: f64| ((lon - longitude) * mpd_lon, (lat - latitude) * mpd_lat);
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
        let distances = self.bounded_dijkstra(
            &[
                (from, snap.connector + snap.fraction * length),
                (to, snap.connector + (1.0 - snap.fraction) * length),
            ],
            cutoff,
        );
        let mut nearest: HashMap<StopIdx, f64> = HashMap::new();
        for link in &self.links {
            let (link_from, link_to) = self.endpoints[link.edge as usize];
            let link_length = self.lengths[link.edge as usize];
            let mut meters = f64::min(
                distances[link_from as usize] + link.fraction * link_length,
                distances[link_to as usize] + (1.0 - link.fraction) * link_length,
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
        let (distances, previous) = self.dijkstra_with_paths(&[
            (from_u, from.connector + from.fraction * from_length),
            (from_v, from.connector + (1.0 - from.fraction) * from_length),
        ]);
        let (to_u, to_v) = self.endpoints[to.edge as usize];
        let via_u = distances[to_u as usize] + to.fraction * to_length + to.connector;
        let via_v = distances[to_v as usize] + (1.0 - to.fraction) * to_length + to.connector;

        let best_transit = if via_u <= via_v { via_u } else { via_v };
        if direct.is_none_or(|meters| best_transit < meters) && best_transit.is_finite() {
            let exit = if via_u <= via_v { to_u } else { to_v };
            // Walk the predecessor chain back to the seed vertex.
            let mut vertices = vec![exit];
            let mut edges = Vec::new();
            let mut at = exit;
            while let Some((prev, edge)) = previous[at as usize] {
                vertices.push(prev);
                edges.push(edge);
                at = prev;
            }
            vertices.reverse();
            edges.reverse();

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

    /// An unbounded Dijkstra recording each vertex's predecessor and the
    /// edge it was entered through.
    #[allow(clippy::type_complexity)]
    fn dijkstra_with_paths(&self, sources: &[(u32, f64)]) -> (Vec<f64>, Vec<Option<(u32, u32)>>) {
        let count = self.vertex_count() as usize;
        let mut distances = vec![f64::INFINITY; count];
        let mut previous: Vec<Option<(u32, u32)>> = vec![None; count];
        let mut heap = BinaryHeap::new();
        for &(vertex, distance) in sources {
            if distance < distances[vertex as usize] {
                distances[vertex as usize] = distance;
                previous[vertex as usize] = None;
                heap.push(Reverse((distance.to_bits(), vertex)));
            }
        }
        while let Some(Reverse((bits, vertex))) = heap.pop() {
            let distance = f64::from_bits(bits);
            if distance > distances[vertex as usize] {
                continue;
            }
            let start = self.adjacency_offsets[vertex as usize] as usize;
            let end = self.adjacency_offsets[vertex as usize + 1] as usize;
            for &(target, meters, edge) in &self.adjacency[start..end] {
                let next = distance + meters;
                if next < distances[target as usize] {
                    distances[target as usize] = next;
                    previous[target as usize] = Some((vertex, edge));
                    heap.push(Reverse((next.to_bits(), target)));
                }
            }
        }
        (distances, previous)
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

    /// Shortest distances in meters from the source frontier to every
    /// vertex within `cutoff` (`inf` beyond).
    fn bounded_dijkstra(&self, sources: &[(u32, f64)], cutoff: f64) -> Vec<f64> {
        let mut distances = vec![f64::INFINITY; self.vertex_count() as usize];
        // Non-negative floats order like their IEEE bit patterns, which
        // makes them usable as binary-heap keys.
        let mut heap = BinaryHeap::new();
        for &(vertex, distance) in sources {
            if distance <= cutoff + 1e-9 && distance < distances[vertex as usize] {
                distances[vertex as usize] = distance;
                heap.push(Reverse((distance.to_bits(), vertex)));
            }
        }
        while let Some(Reverse((bits, vertex))) = heap.pop() {
            let distance = f64::from_bits(bits);
            if distance > distances[vertex as usize] {
                continue;
            }
            let start = self.adjacency_offsets[vertex as usize] as usize;
            let end = self.adjacency_offsets[vertex as usize + 1] as usize;
            for &(target, meters, _) in &self.adjacency[start..end] {
                let next = distance + meters;
                if next <= cutoff + 1e-9 && next < distances[target as usize] {
                    distances[target as usize] = next;
                    heap.push(Reverse((next.to_bits(), target)));
                }
            }
        }
        distances
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

/// The segment R*-tree over a lon/lat polyline set. Each entry is tagged with
/// `(edge index, its first coordinate's index)`.
fn build_tree(coordinate_offsets: &[u32], lons: &[f64], lats: &[f64]) -> RTree<EdgeSegment> {
    let mut segments = Vec::new();
    for edge in 0..coordinate_offsets.len().saturating_sub(1) {
        let start = coordinate_offsets[edge] as usize;
        let end = coordinate_offsets[edge + 1] as usize;
        for segment in start..end - 1 {
            segments.push(EdgeSegment::new(
                Line::new(
                    [lons[segment], lats[segment]],
                    [lons[segment + 1], lats[segment + 1]],
                ),
                (edge as u32, segment as u32),
            ));
        }
    }
    RTree::bulk_load(segments)
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
fn snap_envelope(latitude: f64, longitude: f64, max_snap_distance: f64) -> AABB<[f64; 2]> {
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
    AABB::from_corners(
        [longitude - delta_lon, latitude - delta_lat],
        [longitude + delta_lon, latitude + delta_lat],
    )
}

/// The true geometric length between two lon/lat points, in metres, using a
/// local `cos(latitude)` at their midpoint (exact for a short segment).
fn segment_length(lon_a: f64, lat_a: f64, lon_b: f64, lat_b: f64) -> f64 {
    let (mpd_lon, mpd_lat) = meters_per_degree((lat_a + lat_b) / 2.0);
    let dx = (lon_b - lon_a) * mpd_lon;
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
            let pieces = (segment_length(lon_a, lat_a, lon_b, lat_b) / MAX_SEGMENT_METERS)
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
}
