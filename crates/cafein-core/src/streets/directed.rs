//! Directed, profile-aware street search.
//!
//! Where the walking searches in `search.rs` traverse the undirected metres
//! graph, this module routes over a [`CompiledStreetProfile`]'s per-arc
//! millisecond costs: each directed arc (adjacency slot) costs
//! `arc_millis[slot]`, and the `u32::MAX` sentinel marks an arc the mode may
//! not use. It carries the standalone cycling / e-scooter point-to-point routes
//! and time matrices; distances, geometry, and emissions need the predecessor
//! tree this search does not yet record. The walking loop is untouched.
//!
//! Only the **forward** search is implemented here — a door-to-door route and
//! an origins×destinations matrix are forward searches from each origin,
//! reading the label at the destination's snap. The reverse search (for
//! stops that can *reach* a destination) is a PT-egress concern for a later
//! bike-and-ride PR.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::*;

/// Reusable per-thread state for the directed millisecond search, mirroring
/// [`SearchState`](super::search::SearchState) but over `u64` milliseconds.
/// Predecessor recording (for distance/geometry reconstruction) lands with the
/// cost matrices in a later PR; this search records only times.
#[derive(Default)]
pub(super) struct DirectedState {
    /// Best known time in ms per vertex; `u64::MAX` when unreached.
    distances: Vec<u64>,
    /// The vertices this search has written, so it resets only those.
    touched: Vec<u32>,
    /// Pending `(time, vertex)` entries.
    heap: BinaryHeap<Reverse<(u64, u32)>>,
}

impl DirectedState {
    fn prepare(&mut self, vertices: usize) {
        if self.distances.len() < vertices {
            self.distances.resize(vertices, u64::MAX);
        }
        for &vertex in &self.touched {
            self.distances[vertex as usize] = u64::MAX;
        }
        self.touched.clear();
        self.heap.clear();
    }

    fn set(&mut self, vertex: u32, time: u64) {
        let slot = &mut self.distances[vertex as usize];
        if *slot == u64::MAX {
            self.touched.push(vertex);
        }
        *slot = time;
    }

    /// The best known time to `vertex`, or `u64::MAX` when unreached.
    pub(super) fn time(&self, vertex: u32) -> u64 {
        self.distances[vertex as usize]
    }
}

thread_local! {
    static DIRECTED_STATE: std::cell::RefCell<DirectedState> =
        std::cell::RefCell::new(DirectedState::default());
}

/// The on-network millisecond cost of traversing `fraction` of an arc whose
/// whole-edge cost is `arc`, rounded up. `arc` is finite (the arc is permitted).
fn partial_millis(arc: u32, fraction: f64) -> u64 {
    (f64::from(arc) * fraction).ceil() as u64
}

/// The connector's millisecond cost at the profile's connector speed.
fn connector_millis(connector: f64, connector_speed: f64) -> u64 {
    (connector / connector_speed * 1000.0).ceil() as u64
}

impl StreetNetwork {
    /// The adjacency slot carrying edge `edge` from `from` to `to`, if any.
    fn slot_between(&self, from: u32, to: u32, edge: u32) -> Option<usize> {
        let offsets = self.arrays().adjacency_offsets();
        let targets = self.arrays().adj_targets();
        let edges = self.arrays().adj_edges();
        (offsets[from as usize] as usize..offsets[from as usize + 1] as usize)
            .find(|&slot| targets[slot] == to && edges[slot] == edge)
    }

    /// The two directed arcs of a physical edge as `(forward, reverse)` slot
    /// options: `forward` is the `from → to` arc, `reverse` is `to → from`. A
    /// self-loop (`from == to`) has both arcs at the one vertex, so its two
    /// distinct slots are taken directly rather than found by target.
    fn edge_slots(&self, edge: u32) -> (Option<usize>, Option<usize>) {
        let (from, to) = self.edge_endpoints(edge);
        if from == to {
            let offsets = self.arrays().adjacency_offsets();
            let edges = self.arrays().adj_edges();
            let mut slots = (offsets[from as usize] as usize..offsets[from as usize + 1] as usize)
                .filter(|&slot| edges[slot] == edge);
            (slots.next(), slots.next())
        } else {
            (
                self.slot_between(from, to, edge),
                self.slot_between(to, from, edge),
            )
        }
    }

    /// [`edge_slots`](Self::edge_slots) keeping only the directions `profile`
    /// permits.
    fn snap_arcs(
        &self,
        snap: &Snap,
        profile: &CompiledStreetProfile,
    ) -> (Option<usize>, Option<usize>) {
        let (forward, reverse) = self.edge_slots(snap.edge);
        let permitted =
            |slot: Option<usize>| slot.filter(|&slot| profile.arc_millis[slot] != u32::MAX);
        (permitted(forward), permitted(reverse))
    }

    /// Whether `profile` may traverse `edge` in either direction.
    fn edge_permits(&self, edge: u32, profile: &CompiledStreetProfile) -> bool {
        let (forward, reverse) = self.edge_slots(edge);
        [forward, reverse]
            .into_iter()
            .flatten()
            .any(|slot| profile.arc_millis[slot] != u32::MAX)
    }

    /// Snaps a coordinate to the nearest edge `profile` can actually use.
    ///
    /// The plain [`snap`](Self::snap) is mode-blind, so it can land a bicycle
    /// query on a footway the mode may not enter — leaving the search with no
    /// permitted arc to start from and reporting the destination unreachable.
    /// This skips those edges, so a route starts from the nearest street the
    /// mode is allowed on.
    pub fn snap_for_profile(
        &self,
        latitude: f64,
        longitude: f64,
        max_snap_distance: f64,
        profile: &CompiledStreetProfile,
    ) -> Option<Snap> {
        self.snap_filtered(latitude, longitude, max_snap_distance, |edge| {
            self.edge_permits(edge, profile)
        })
    }

    /// The starting frontier for a coordinate snapped at `snap`, leaving toward
    /// the edge's endpoints: `to` over the remaining fraction when the forward
    /// arc is permitted, `from` over the leading fraction when the reverse arc
    /// is permitted. Each cost is the connector plus the proportional on-edge
    /// time.
    fn directed_seeds(&self, snap: &Snap, profile: &CompiledStreetProfile) -> Vec<(u32, u64)> {
        let (from, to) = self.edge_endpoints(snap.edge);
        let (forward, reverse) = self.snap_arcs(snap, profile);
        let connector = connector_millis(snap.connector, profile.definition.connector_speed);
        let mut seeds = Vec::with_capacity(2);
        // Reach `from`: at fraction 0 the snap is already there (no arc needed);
        // otherwise travel the leading fraction against the edge (reverse arc).
        if snap.fraction == 0.0 {
            seeds.push((from, connector));
        } else if let Some(slot) = reverse {
            seeds.push((
                from,
                connector.saturating_add(partial_millis(profile.arc_millis[slot], snap.fraction)),
            ));
        }
        // Reach `to`: at fraction 1 already there; otherwise travel the
        // remaining fraction along the edge (forward arc).
        if snap.fraction == 1.0 {
            seeds.push((to, connector));
        } else if let Some(slot) = forward {
            seeds.push((
                to,
                connector.saturating_add(partial_millis(
                    profile.arc_millis[slot],
                    1.0 - snap.fraction,
                )),
            ));
        }
        seeds
    }

    /// The cost of *arriving* at a coordinate snapped at `snap` from each edge
    /// endpoint: reach the snap point from `from` over the leading fraction
    /// when the forward arc is permitted, from `to` over the remaining fraction
    /// when the reverse arc is permitted, plus the connector.
    fn directed_egress(&self, snap: &Snap, profile: &CompiledStreetProfile) -> Vec<(u32, u64)> {
        let (from, to) = self.edge_endpoints(snap.edge);
        let (forward, reverse) = self.snap_arcs(snap, profile);
        let connector = connector_millis(snap.connector, profile.definition.connector_speed);
        let mut offsets = Vec::with_capacity(2);
        // Arrive at the snap from `from`: at fraction 0 the snap is at `from`;
        // otherwise travel the leading fraction along the edge (forward arc).
        if snap.fraction == 0.0 {
            offsets.push((from, connector));
        } else if let Some(slot) = forward {
            offsets.push((
                from,
                connector.saturating_add(partial_millis(profile.arc_millis[slot], snap.fraction)),
            ));
        }
        // Arrive from `to`: at fraction 1 the snap is at `to`; otherwise travel
        // the remaining fraction against the edge (reverse arc).
        if snap.fraction == 1.0 {
            offsets.push((to, connector));
        } else if let Some(slot) = reverse {
            offsets.push((
                to,
                connector.saturating_add(partial_millis(
                    profile.arc_millis[slot],
                    1.0 - snap.fraction,
                )),
            ));
        }
        offsets
    }

    /// The direct same-edge candidate: when both snaps sit on one edge, the
    /// millisecond cost of travelling directly between them, if the direction
    /// from origin to destination is permitted.
    fn same_edge_millis(
        &self,
        from: &Snap,
        to: &Snap,
        profile: &CompiledStreetProfile,
    ) -> Option<u64> {
        if from.edge != to.edge {
            return None;
        }
        let connectors =
            connector_millis(from.connector, profile.definition.connector_speed).saturating_add(
                connector_millis(to.connector, profile.definition.connector_speed),
            );
        // Same point on the edge: no on-edge travel, so either direction serves.
        let span = (to.fraction - from.fraction).abs();
        if span == 0.0 {
            return Some(connectors);
        }
        // The destination is ahead along the edge (forward) or behind (reverse).
        let (forward, reverse) = self.snap_arcs(from, profile);
        let slot = if to.fraction > from.fraction {
            forward
        } else {
            reverse
        };
        slot.map(|slot| connectors.saturating_add(partial_millis(profile.arc_millis[slot], span)))
    }

    /// Reusable one-to-many directed search: relax each vertex's outgoing arcs
    /// by `profile.arc_millis`, skipping the `u32::MAX` forbidden sentinel,
    /// bounded by `cutoff` milliseconds.
    pub(super) fn directed_dijkstra(
        &self,
        profile: &CompiledStreetProfile,
        sources: &[(u32, u64)],
        cutoff: u64,
        state: &mut DirectedState,
    ) {
        state.prepare(self.vertex_count() as usize);
        let offsets = self.arrays().adjacency_offsets();
        let targets = self.arrays().adj_targets();
        let arc_millis = &profile.arc_millis;
        for &(vertex, time) in sources {
            if time <= cutoff && time < state.time(vertex) {
                state.set(vertex, time);
                state.heap.push(Reverse((time, vertex)));
            }
        }
        while let Some(Reverse((time, vertex))) = state.heap.pop() {
            if time > state.time(vertex) {
                continue;
            }
            let start = offsets[vertex as usize] as usize;
            let end = offsets[vertex as usize + 1] as usize;
            for slot in start..end {
                let cost = arc_millis[slot];
                if cost == u32::MAX {
                    continue;
                }
                let next = time.saturating_add(u64::from(cost));
                let target = targets[slot];
                if next <= cutoff && next < state.time(target) {
                    state.set(target, next);
                    state.heap.push(Reverse((next, target)));
                }
            }
        }
    }

    /// The per-vertex times of a one-to-many directed search, for tests.
    #[cfg(test)]
    pub(super) fn directed_distances(
        &self,
        profile: &CompiledStreetProfile,
        sources: &[(u32, u64)],
        cutoff: u64,
    ) -> Vec<u64> {
        DIRECTED_STATE.with(|cell| {
            let state = &mut cell.borrow_mut();
            self.directed_dijkstra(profile, sources, cutoff, state);
            (0..self.vertex_count()).map(|v| state.time(v)).collect()
        })
    }

    /// The directed travel time in whole seconds (rounded up) from `from` to
    /// `to` under `profile`, or `None` when the destination is not reachable
    /// within `max_seconds`. Combines a forward search from the origin's
    /// directed seeds with the destination's arrival offsets and the direct
    /// same-edge candidate.
    pub fn directed_travel_time(
        &self,
        from: &Snap,
        to: &Snap,
        profile: &CompiledStreetProfile,
        max_seconds: f64,
    ) -> Option<u32> {
        self.directed_times_to_snaps(from, std::slice::from_ref(&Some(*to)), profile, max_seconds)
            [0]
    }

    /// The directed travel times from `from` to each of `targets`, in whole
    /// seconds, or `None` per target that is unsnapped or beyond `max_seconds`.
    ///
    /// One bounded search serves the whole row: every target reads the same
    /// settled labels, so a matrix row costs one street search rather than one
    /// per pair. Sharing [`arrival_millis`](Self::arrival_millis) with the
    /// single route keeps the two answers identical cell for cell.
    pub fn directed_times_to_snaps(
        &self,
        from: &Snap,
        targets: &[Option<Snap>],
        profile: &CompiledStreetProfile,
        max_seconds: f64,
    ) -> Vec<Option<u32>> {
        if !max_seconds.is_finite() || max_seconds < 0.0 {
            return vec![None; targets.len()];
        }
        // Floor, not ceil: an integer-millisecond route is within `max_seconds`
        // exactly when it is `<= floor(max_seconds * 1000)`, so a route just
        // beyond the requested duration is not admitted.
        let cutoff = (max_seconds * 1000.0).floor() as u64;
        let seeds = self.directed_seeds(from, profile);
        DIRECTED_STATE.with(|cell| {
            let state = &mut cell.borrow_mut();
            // With no seed the search settles nothing; `prepare` inside still
            // clears the previous query, so the labels read as unreached.
            self.directed_dijkstra(profile, &seeds, cutoff, state);
            targets
                .iter()
                .map(|target| {
                    let to = target.as_ref()?;
                    let millis = self.arrival_millis(from, to, profile, cutoff, state)?;
                    Some(seconds(millis as f64 / 1000.0))
                })
                .collect()
        })
    }

    /// The best millisecond cost from `from` to `to` given a completed search
    /// from `from`'s seeds: the destination's arrival offsets against the
    /// settled labels, and the direct same-edge candidate.
    fn arrival_millis(
        &self,
        from: &Snap,
        to: &Snap,
        profile: &CompiledStreetProfile,
        cutoff: u64,
        state: &DirectedState,
    ) -> Option<u64> {
        let mut best = self
            .same_edge_millis(from, to, profile)
            .filter(|&ms| ms <= cutoff);
        for (vertex, offset) in self.directed_egress(to, profile) {
            let reached = state.time(vertex);
            if reached != u64::MAX {
                let total = reached.saturating_add(offset);
                if total <= cutoff {
                    best = Some(best.map_or(total, |b| b.min(total)));
                }
            }
        }
        best
    }

    /// The origins × destinations travel-time matrix under `profile`, with the
    /// index lists of the coordinates that did not snap.
    ///
    /// Both sides snap through [`snap_for_profile`](Self::snap_for_profile), so
    /// no query starts or ends on an edge the mode may not use. Origins fan out
    /// over the thread pool; each row depends only on its own origin and rayon
    /// preserves input order, so the matrix is the same however the work is
    /// scheduled. A destination at its origin's own coordinate is zero away —
    /// routing it through the network would charge that coordinate's connector
    /// twice, leaving a point a positive time from itself.
    pub fn directed_matrix(
        &self,
        origins: &[(f64, f64)],
        destinations: &[(f64, f64)],
        profile: &CompiledStreetProfile,
        max_seconds: f64,
        max_snap_distance: f64,
    ) -> (Vec<Vec<Option<u32>>>, Vec<u32>, Vec<u32>) {
        let snap = |&(latitude, longitude): &(f64, f64)| {
            self.snap_for_profile(latitude, longitude, max_snap_distance, profile)
        };
        let target_snaps: Vec<Option<Snap>> = destinations.par_iter().map(snap).collect();
        let origin_snaps: Vec<Option<Snap>> = origins.par_iter().map(snap).collect();
        // The same-coordinate zero is a route the cutoff still has to admit: a
        // cutoff that admits nothing (negative or non-finite) leaves the cell
        // unreachable, exactly as the single route reports it.
        let routable = max_seconds.is_finite() && max_seconds >= 0.0;
        let rows: Vec<Vec<Option<u32>>> = origin_snaps
            .par_iter()
            .zip(origins)
            .map(|(origin, &coordinate)| match origin {
                None => vec![None; destinations.len()],
                Some(from) => {
                    let mut row =
                        self.directed_times_to_snaps(from, &target_snaps, profile, max_seconds);
                    if routable {
                        for ((cell, &destination), target) in
                            row.iter_mut().zip(destinations).zip(&target_snaps)
                        {
                            if destination == coordinate && target.is_some() {
                                *cell = Some(0);
                            }
                        }
                    }
                    row
                }
            })
            .collect();
        let unsnapped = |snaps: &[Option<Snap>]| -> Vec<u32> {
            snaps
                .iter()
                .enumerate()
                .filter(|(_, snap)| snap.is_none())
                .map(|(index, _)| index as u32)
                .collect()
        };
        (rows, unsnapped(&origin_snaps), unsnapped(&target_snaps))
    }
}
