//! The query-time street network for walking access and egress.
//!
//! The Python-side build hands the walking graph over as flat arrays:
//! vertices are implicit indices, edges carry their cost length in meters
//! and their geometry. A query snaps a coordinate to its nearest edge
//! through a uniform spatial grid — a virtual node at the snap point,
//! with the split edge's cost pro-rated by the fraction each side covers —
//! and runs a cutoff-bounded Dijkstra to collect every transit stop
//! reachable on foot, entering stops through the same snap links the
//! footpath precompute used. Walking is undirected, so one search serves
//! access and egress alike.
//!
//! Geometry lives in a single local equirectangular plane scaled at the
//! network's mean latitude, which is accurate over the city- and
//! regional-scale extracts one street tile covers; country-scale
//! coverage splits into tiles, each with its own projection.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use crate::timetable::StopIdx;

/// Edge length of a spatial-grid cell, in meters.
const CELL_METERS: f64 = 100.0;

/// How a stop enters the street graph: snapped onto an edge at a fraction
/// of its cost length, over a straight connector to the snap point.
#[derive(Debug, Clone, Copy, PartialEq)]
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
    /// Outgoing `(target vertex, meters)` entries; edges appear in both
    /// directions.
    adjacency: Vec<(u32, f64)>,
    /// Edge endpoints, one entry per input edge.
    endpoints: Vec<(u32, u32)>,
    /// Edge cost lengths in meters (the OSM way length, which the split
    /// pro-rating distributes; it may differ from the geometric length).
    lengths: Vec<f64>,
    /// Offsets into the projected coordinate arrays, one per edge plus a
    /// tail.
    coordinate_offsets: Vec<u32>,
    /// Edge geometries projected to local equirectangular meters.
    xs: Vec<f64>,
    ys: Vec<f64>,
    /// How each snapped stop enters the graph.
    links: Vec<StopLink>,
    /// Edges indexed by the grid cells their geometry passes through.
    grid: HashMap<(i32, i32), Vec<u32>>,
    /// Occupied cell range as `(min_column, max_column, min_row,
    /// max_row)`; `None` when the network has no edges.
    cell_bounds: Option<(i32, i32, i32, i32)>,
    /// Projection origin as `(longitude, latitude)`.
    origin: (f64, f64),
    /// Meters per degree of `(longitude, latitude)` at the origin.
    scale: (f64, f64),
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

        let (origin, scale) = projection(longitudes, latitudes);
        let xs: Vec<f64> = longitudes
            .iter()
            .map(|lon| (lon - origin.0) * scale.0)
            .collect();
        let ys: Vec<f64> = latitudes
            .iter()
            .map(|lat| (lat - origin.1) * scale.1)
            .collect();

        let mut adjacency_offsets = vec![0u32; vertex_count as usize + 1];
        for &(from, to, _) in edges {
            adjacency_offsets[from as usize + 1] += 1;
            adjacency_offsets[to as usize + 1] += 1;
        }
        for vertex in 0..vertex_count as usize {
            adjacency_offsets[vertex + 1] += adjacency_offsets[vertex];
        }
        let mut adjacency = vec![(0u32, 0.0); edges.len() * 2];
        let mut cursor = adjacency_offsets.clone();
        for &(from, to, meters) in edges {
            for (a, b) in [(from, to), (to, from)] {
                let slot = cursor[a as usize] as usize;
                adjacency[slot] = (b, meters);
                cursor[a as usize] += 1;
            }
        }

        let mut grid: HashMap<(i32, i32), Vec<u32>> = HashMap::new();
        let mut cell_bounds: Option<(i32, i32, i32, i32)> = None;
        for index in 0..edges.len() {
            let start = coordinate_offsets[index] as usize;
            let end = coordinate_offsets[index + 1] as usize;
            let mut cells = Vec::new();
            for segment in start..end - 1 {
                let (min_x, max_x) = ordered(xs[segment], xs[segment + 1]);
                let (min_y, max_y) = ordered(ys[segment], ys[segment + 1]);
                for column in cell_of(min_x)..=cell_of(max_x) {
                    for row in cell_of(min_y)..=cell_of(max_y) {
                        cells.push((column, row));
                    }
                }
            }
            cells.sort_unstable();
            cells.dedup();
            for (column, row) in cells {
                grid.entry((column, row)).or_default().push(index as u32);
                cell_bounds = Some(match cell_bounds {
                    None => (column, column, row, row),
                    Some((min_column, max_column, min_row, max_row)) => (
                        min_column.min(column),
                        max_column.max(column),
                        min_row.min(row),
                        max_row.max(row),
                    ),
                });
            }
        }

        Ok(StreetNetwork {
            adjacency_offsets,
            adjacency,
            endpoints: edges.iter().map(|&(from, to, _)| (from, to)).collect(),
            lengths: edges.iter().map(|&(_, _, meters)| meters).collect(),
            coordinate_offsets: coordinate_offsets.to_vec(),
            xs,
            ys,
            links,
            grid,
            cell_bounds,
            origin,
            scale,
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

    /// Snaps a coordinate to its nearest edge within `max_snap_distance`
    /// meters, searching grid cells in expanding rings. Non-finite
    /// coordinates or a non-finite or negative allowance never snap.
    pub fn snap(&self, latitude: f64, longitude: f64, max_snap_distance: f64) -> Option<Snap> {
        if !latitude.is_finite()
            || !longitude.is_finite()
            || !max_snap_distance.is_finite()
            || max_snap_distance < 0.0
        {
            return None;
        }
        let x = (longitude - self.origin.0) * self.scale.0;
        let y = (latitude - self.origin.1) * self.scale.1;
        self.nearest_edge(x, y, max_snap_distance)
            .filter(|&(distance, _, _)| distance <= max_snap_distance)
            .map(|(distance, edge, fraction)| Snap {
                edge,
                fraction,
                connector: distance,
            })
    }

    /// The nearest edge to a projected position as
    /// `(distance, edge, fraction)`, or `None` on an edgeless network.
    ///
    /// An allowance whose ring search would visit more cells than the
    /// grid holds — it spans the whole grid, or the query lies far
    /// outside it — degenerates to a scan of every edge, which bounds
    /// the work whatever the caller allows.
    fn nearest_edge(&self, x: f64, y: f64, max_snap_distance: f64) -> Option<(f64, u32, f64)> {
        let (min_column, max_column, min_row, max_row) = self.cell_bounds?;
        let center = (cell_of(x), cell_of(y));
        let allowance = (max_snap_distance / CELL_METERS).ceil() + 1.0;
        let extent = (center.0 as i64 - min_column as i64)
            .max(max_column as i64 - center.0 as i64)
            .max(center.1 as i64 - min_row as i64)
            .max(max_row as i64 - center.1 as i64)
            .max(0) as f64;
        let mut best: Option<(f64, u32, f64)> = None;
        if allowance >= extent || 4.0 * allowance * allowance >= self.grid.len() as f64 {
            for edge in 0..self.endpoints.len() as u32 {
                let (distance, fraction) = self.project_onto_edge(edge, x, y);
                if best.is_none_or(|(nearest, _, _)| distance < nearest) {
                    best = Some((distance, edge, fraction));
                }
            }
            return best;
        }
        let mut seen = vec![false; self.endpoints.len()];
        for radius in 0..=allowance as i32 {
            // Cells on ring `radius` lie at least `radius - 1` cells away
            // from the query, so a hit at or under that bound is final.
            let reachable = (radius - 1) as f64 * CELL_METERS;
            if reachable > max_snap_distance {
                break;
            }
            if let Some((distance, _, _)) = best {
                if distance <= reachable {
                    break;
                }
            }
            for cell in ring(center, radius) {
                let Some(edges) = self.grid.get(&cell) else {
                    continue;
                };
                for &edge in edges {
                    if std::mem::replace(&mut seen[edge as usize], true) {
                        continue;
                    }
                    let (distance, fraction) = self.project_onto_edge(edge, x, y);
                    if best.is_none_or(|(nearest, _, _)| distance < nearest) {
                        best = Some((distance, edge, fraction));
                    }
                }
            }
        }
        best
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

    /// The distance and linear-referenced fraction of a coordinate's
    /// projection onto an edge's geometry.
    fn project_onto_edge(&self, edge: u32, x: f64, y: f64) -> (f64, f64) {
        let start = self.coordinate_offsets[edge as usize] as usize;
        let end = self.coordinate_offsets[edge as usize + 1] as usize;
        let mut nearest = f64::INFINITY;
        let mut nearest_along = 0.0;
        let mut cumulative = 0.0;
        for segment in start..end - 1 {
            let (ax, ay) = (self.xs[segment], self.ys[segment]);
            let (bx, by) = (self.xs[segment + 1], self.ys[segment + 1]);
            let (dx, dy) = (bx - ax, by - ay);
            let squared = dx * dx + dy * dy;
            let along = if squared > 0.0 {
                (((x - ax) * dx + (y - ay) * dy) / squared).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let (px, py) = (ax + along * dx, ay + along * dy);
            let distance = ((x - px) * (x - px) + (y - py) * (y - py)).sqrt();
            let segment_length = squared.sqrt();
            if distance < nearest {
                nearest = distance;
                nearest_along = cumulative + along * segment_length;
            }
            cumulative += segment_length;
        }
        let fraction = if cumulative > 0.0 {
            nearest_along / cumulative
        } else {
            0.0
        };
        (nearest, fraction)
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
            for &(target, meters) in &self.adjacency[start..end] {
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

/// The projection origin and meters-per-degree scale of a coordinate set.
fn projection(longitudes: &[f64], latitudes: &[f64]) -> ((f64, f64), (f64, f64)) {
    if longitudes.is_empty() {
        return ((0.0, 0.0), meters_per_degree(0.0));
    }
    let count = longitudes.len() as f64;
    let origin = (
        longitudes.iter().sum::<f64>() / count,
        latitudes.iter().sum::<f64>() / count,
    );
    (origin, meters_per_degree(origin.1))
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

/// The grid cell coordinate of a projected position.
fn cell_of(position: f64) -> i32 {
    (position / CELL_METERS).floor() as i32
}

fn ordered(a: f64, b: f64) -> (f64, f64) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

/// The cells at Chebyshev distance `radius` around `center`.
fn ring(center: (i32, i32), radius: i32) -> Vec<(i32, i32)> {
    let (cx, cy) = center;
    if radius == 0 {
        return vec![(cx, cy)];
    }
    let mut cells = Vec::with_capacity(radius as usize * 8);
    for dx in -radius..=radius {
        cells.push((cx.saturating_add(dx), cy.saturating_sub(radius)));
        cells.push((cx.saturating_add(dx), cy.saturating_add(radius)));
    }
    for dy in (1 - radius)..radius {
        cells.push((cx.saturating_sub(radius), cy.saturating_add(dy)));
        cells.push((cx.saturating_add(radius), cy.saturating_add(dy)));
    }
    cells
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
        // The long second edge blows the grid up past the ring search's
        // whole-grid fallback threshold, so the expanding rings really
        // run; the nearest edge sits several rings from the query.
        let network = network(
            4,
            0,
            &[
                (0, 1, 400.0, straight((250.0, 0.0), (250.0, 400.0))),
                (2, 3, 10_000.0, straight((250.0, 1500.0), (250.0, 11_500.0))),
            ],
            vec![],
        )
        .unwrap();
        let (lon, lat) = lonlat(0.0, 0.0);
        let snap = network.snap(lat, lon, 300.0).unwrap();
        assert_eq!(snap.edge, 0);
        // The far edge shifts the mean-latitude scale a little, so the
        // designed 250 m connector holds only to fractions of a meter.
        assert!((snap.connector - 250.0).abs() < 0.5);
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
    fn survives_huge_snap_allowances() {
        let network = network(
            2,
            1,
            &[(0, 1, 400.0, straight((0.0, 0.0), (400.0, 0.0)))],
            vec![link(0, 0, 0.5, 0.0)],
        )
        .unwrap();
        // A finite but absurd allowance degenerates to scanning every
        // edge instead of walking grid rings; results stay correct.
        let (lon, lat) = lonlat(100.0, 0.0);
        let snap = network.snap(lat, lon, 1e12).unwrap();
        assert_eq!(snap.edge, 0);
        assert!(snap.connector < 0.01);
        // The same fallback serves queries far outside the grid.
        let (far_lon, far_lat) = lonlat(5_000_000.0, 0.0);
        let far = network.snap(far_lat, far_lon, 1e12).unwrap();
        assert_eq!(far.edge, 0);
        assert!(far.connector > 1_000_000.0);
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
