//! Walk-path reconstruction along the searched street edges.

use super::*;

impl StreetNetwork {
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
        let from_length = self.arrays().lengths()[from.edge as usize];
        let to_length = self.arrays().lengths()[to.edge as usize];
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
            loop {
                let (prev, edge) = state.previous[at as usize];
                if (prev, edge) == NO_PREVIOUS {
                    break;
                }
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
    pub(super) fn dijkstra_with_paths(
        &self,
        sources: &[(u32, f64)],
        targets: (u32, u32),
        state: &mut SearchState,
    ) {
        state.prepare_with_previous(self.vertex_count() as usize);
        let adjacency_offsets = self.arrays().adjacency_offsets();
        let adj_targets = self.arrays().adj_targets();
        let adj_meters = self.arrays().adj_meters();
        let adj_edges = self.arrays().adj_edges();
        for &(vertex, distance) in sources {
            if distance < state.distance(vertex) {
                state.set_distance(vertex, distance);
                state.previous[vertex as usize] = NO_PREVIOUS;
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
                    state.set_distance(target, next);
                    state.previous[target as usize] = (vertex, adj_edges[slot]);
                    state.heap.push(Reverse((next.to_bits(), target)));
                }
            }
        }
    }

    /// The `(lon, lat)` point at a fraction of an edge's true length.
    pub(super) fn point_at(&self, edge: u32, fraction: f64) -> (f64, f64) {
        let start = self.arrays().coordinate_offsets()[edge as usize] as usize;
        let end = self.arrays().coordinate_offsets()[edge as usize + 1] as usize;
        let total = self.along(end - 1);
        self.interpolate(start, end, fraction.clamp(0.0, 1.0) * total)
    }

    /// The `(lon, lat)` geometry of an edge between two fractions of its true
    /// length, endpoints interpolated; reversed when `from_fraction >
    /// to_fraction`. Always at least one point.
    pub(super) fn edge_slice(
        &self,
        edge: u32,
        from_fraction: f64,
        to_fraction: f64,
    ) -> Vec<(f64, f64)> {
        let start = self.arrays().coordinate_offsets()[edge as usize] as usize;
        let end = self.arrays().coordinate_offsets()[edge as usize + 1] as usize;
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
    pub(super) fn interpolate(&self, start: usize, end: usize, target: f64) -> (f64, f64) {
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
    pub(super) fn bounded_dijkstra(
        &self,
        sources: &[(u32, f64)],
        cutoff: f64,
        state: &mut SearchState,
    ) {
        state.prepare(self.vertex_count() as usize);
        let adjacency_offsets = self.arrays().adjacency_offsets();
        let adj_targets = self.arrays().adj_targets();
        let adj_meters = self.arrays().adj_meters();
        for &(vertex, distance) in sources {
            if distance <= cutoff + 1e-9 && distance < state.distance(vertex) {
                state.set_distance(vertex, distance);
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
                    state.set_distance(target, next);
                    state.heap.push(Reverse((next.to_bits(), target)));
                }
            }
        }
    }
}
