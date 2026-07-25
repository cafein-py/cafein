//! Directed, profile-aware street search.
//!
//! Where the walking searches in `search.rs` traverse the undirected metres
//! graph, this module routes over a [`CompiledStreetProfile`]'s per-arc
//! millisecond costs: each directed arc (adjacency slot) costs
//! `arc_millis[slot]`, and the `u32::MAX` sentinel marks an arc the mode may
//! not use. It carries the standalone cycling / e-scooter point-to-point
//! routes, time matrices, and the distance/geometry reconstruction behind the
//! cost outputs. The walking loop is untouched.
//!
//! Only the **forward** search is implemented here — a door-to-door route and
//! an origins×destinations matrix are forward searches from each origin,
//! reading the label at the destination's snap. The reverse search (for
//! stops that can *reach* a destination) is a PT-egress concern for a later
//! bike-and-ride PR.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use super::*;

/// The distance-provenance tier of a reconstructed street leg.
///
/// cafein computes every edge length from the stored geometry at build time, so
/// a leg's network metres are those lengths and its connectors are geodesic —
/// the exact tier design §8.3 names. The documented fallbacks
/// (`geometry_measured`, `crow_fly`) exist for importers without trustworthy
/// edge lengths, which cafein has none of, so this is the only tier produced.
pub const STREET_DISTANCE_PROVENANCE: &str = "osm_edge_length+geodesic_connector";

/// A reconstructed street leg: the winning path's time, distances, and shape.
///
/// The distances are kept apart because they are measured differently.
/// `network_meters` sums the stored physical edge lengths (and the partial-edge
/// fractions at each end) the route actually traversed; `connector_meters` is
/// the straight line from each coordinate to its snap point, which is not on
/// the network at all. The stored lengths stay authoritative: `geometry` is the
/// reconstructed shape for display, and remeasuring it never replaces them.
#[derive(Debug, Clone, PartialEq)]
pub struct StreetLeg {
    /// Travel time in whole seconds, matching `directed_travel_time`.
    pub seconds: u32,
    pub network_meters: f64,
    pub connector_meters: f64,
    /// The leg's shape in (longitude, latitude), connectors included.
    pub geometry: Vec<(f64, f64)>,
}

/// One end of a snapped edge as the search sees it: the vertex, the cost of
/// reaching it from (or the snap from it), and **which end of the edge it is**,
/// as a fraction.
///
/// The fraction is not recoverable from the vertex alone: a self-loop's two
/// ends are the same vertex, so a reconstruction that inferred the side from
/// the vertex would follow the wrong part of the loop whenever the nearer side
/// is the forbidden one. Carrying it keeps the reported distance and geometry
/// on the arc the search actually used.
#[derive(Debug, Clone, Copy)]
pub(super) struct Endpoint {
    vertex: u32,
    millis: u64,
    fraction: f64,
}

impl Endpoint {
    fn new(vertex: u32, millis: u64, fraction: f64) -> Endpoint {
        Endpoint {
            vertex,
            millis,
            fraction,
        }
    }
}

/// Reusable per-thread state for the directed millisecond search, mirroring
/// [`SearchState`](super::search::SearchState) but over `u64` milliseconds.
/// A time query records only times; the predecessor array behind
/// distance/geometry reconstruction grows only for the searches that read it.
#[derive(Default)]
pub(super) struct DirectedState {
    /// Best known time in ms per vertex; `u64::MAX` when unreached.
    distances: Vec<u64>,
    /// The vertices this search has written, so it resets only those.
    touched: Vec<u32>,
    /// Predecessor `(vertex, edge)` per vertex, [`NO_PREVIOUS`] when unset.
    /// Grown only by a reconstructing search, so a time query never pays for
    /// it.
    previous: Vec<(u32, u32)>,
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

    /// Clears and grows the predecessor array too. Only a reconstructing
    /// search pays: `prepare` itself never touches `previous`, so a time query
    /// neither allocates nor resets it. The stale slots are cleared here,
    /// before `prepare` drops the `touched` list that names them.
    fn prepare_with_previous(&mut self, vertices: usize) {
        for &vertex in &self.touched {
            if let Some(slot) = self.previous.get_mut(vertex as usize) {
                *slot = NO_PREVIOUS;
            }
        }
        self.prepare(vertices);
        if self.previous.len() < vertices {
            self.previous.resize(vertices, NO_PREVIOUS);
        }
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
    fn directed_seeds(&self, snap: &Snap, profile: &CompiledStreetProfile) -> Vec<Endpoint> {
        let (from, to) = self.edge_endpoints(snap.edge);
        let (forward, reverse) = self.snap_arcs(snap, profile);
        let connector = connector_millis(snap.connector, profile.definition.connector_speed);
        let mut seeds = Vec::with_capacity(2);
        // Reach `from`: at fraction 0 the snap is already there (no arc needed);
        // otherwise travel the leading fraction against the edge (reverse arc).
        if snap.fraction == 0.0 {
            seeds.push(Endpoint::new(from, connector, 0.0));
        } else if let Some(slot) = reverse {
            seeds.push(Endpoint::new(
                from,
                connector.saturating_add(partial_millis(profile.arc_millis[slot], snap.fraction)),
                0.0,
            ));
        }
        // Reach `to`: at fraction 1 already there; otherwise travel the
        // remaining fraction along the edge (forward arc).
        if snap.fraction == 1.0 {
            seeds.push(Endpoint::new(to, connector, 1.0));
        } else if let Some(slot) = forward {
            seeds.push(Endpoint::new(
                to,
                connector.saturating_add(partial_millis(
                    profile.arc_millis[slot],
                    1.0 - snap.fraction,
                )),
                1.0,
            ));
        }
        seeds
    }

    /// The cost of *arriving* at a coordinate snapped at `snap` from each edge
    /// endpoint: reach the snap point from `from` over the leading fraction
    /// when the forward arc is permitted, from `to` over the remaining fraction
    /// when the reverse arc is permitted, plus the connector.
    fn directed_egress(&self, snap: &Snap, profile: &CompiledStreetProfile) -> Vec<Endpoint> {
        let (from, to) = self.edge_endpoints(snap.edge);
        let (forward, reverse) = self.snap_arcs(snap, profile);
        let connector = connector_millis(snap.connector, profile.definition.connector_speed);
        let mut offsets = Vec::with_capacity(2);
        // Arrive at the snap from `from`: at fraction 0 the snap is at `from`;
        // otherwise travel the leading fraction along the edge (forward arc).
        if snap.fraction == 0.0 {
            offsets.push(Endpoint::new(from, connector, 0.0));
        } else if let Some(slot) = forward {
            offsets.push(Endpoint::new(
                from,
                connector.saturating_add(partial_millis(profile.arc_millis[slot], snap.fraction)),
                0.0,
            ));
        }
        // Arrive from `to`: at fraction 1 the snap is at `to`; otherwise travel
        // the remaining fraction against the edge (reverse arc).
        if snap.fraction == 1.0 {
            offsets.push(Endpoint::new(to, connector, 1.0));
        } else if let Some(slot) = reverse {
            offsets.push(Endpoint::new(
                to,
                connector.saturating_add(partial_millis(
                    profile.arc_millis[slot],
                    1.0 - snap.fraction,
                )),
                1.0,
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
        sources: &[Endpoint],
        cutoff: u64,
        state: &mut DirectedState,
    ) {
        state.prepare(self.vertex_count() as usize);
        let offsets = self.arrays().adjacency_offsets();
        let targets = self.arrays().adj_targets();
        let arc_millis = &profile.arc_millis;
        for source in sources {
            let (vertex, time) = (source.vertex, source.millis);
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

    /// [`directed_dijkstra`](Self::directed_dijkstra) additionally recording
    /// each vertex's predecessor `(vertex, edge)`.
    ///
    /// A separate loop rather than a flag on the time search: the time matrix
    /// must not pay a branch per relaxation for state it never reads, the same
    /// split the walking `bounded_dijkstra` / `dijkstra_with_paths` pair makes.
    pub(super) fn directed_dijkstra_with_paths(
        &self,
        profile: &CompiledStreetProfile,
        sources: &[Endpoint],
        cutoff: u64,
        state: &mut DirectedState,
    ) {
        state.prepare_with_previous(self.vertex_count() as usize);
        let offsets = self.arrays().adjacency_offsets();
        let targets = self.arrays().adj_targets();
        let edges = self.arrays().adj_edges();
        let arc_millis = &profile.arc_millis;
        for source in sources {
            let (vertex, time) = (source.vertex, source.millis);
            if time <= cutoff && time < state.time(vertex) {
                state.set(vertex, time);
                state.previous[vertex as usize] = NO_PREVIOUS;
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
                    state.previous[target as usize] = (vertex, edges[slot]);
                    state.heap.push(Reverse((next, target)));
                }
            }
        }
    }

    /// The per-vertex times of a one-to-many directed search, for tests.
    /// Sources are `(vertex, millis)`; the seed's edge fraction is irrelevant
    /// to a raw search, so it is filled with zero.
    #[cfg(test)]
    pub(super) fn directed_distances(
        &self,
        profile: &CompiledStreetProfile,
        sources: &[(u32, u64)],
        cutoff: u64,
    ) -> Vec<u64> {
        let seeds: Vec<Endpoint> = sources
            .iter()
            .map(|&(vertex, millis)| Endpoint::new(vertex, millis, 0.0))
            .collect();
        DIRECTED_STATE.with(|cell| {
            let state = &mut cell.borrow_mut();
            self.directed_dijkstra(profile, &seeds, cutoff, state);
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
        for arrival in self.directed_egress(to, profile) {
            let reached = state.time(arrival.vertex);
            if reached != u64::MAX {
                let total = reached.saturating_add(arrival.millis);
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

    /// The reconstructed leg from `from` to `to`, or `None` when unreachable
    /// within `max_seconds`.
    pub fn directed_leg(
        &self,
        from_point: (f64, f64),
        from: &Snap,
        to_point: (f64, f64),
        to: &Snap,
        profile: &CompiledStreetProfile,
        max_seconds: f64,
    ) -> Option<StreetLeg> {
        self.directed_legs_to_snaps(
            from_point,
            from,
            &[(to_point, Some(*to))],
            profile,
            max_seconds,
        )
        .swap_remove(0)
    }

    /// The reconstructed legs from `from` to each of `targets`, one search
    /// serving the whole row.
    ///
    /// Metres from the seed to each settled vertex are memoised, so
    /// destinations sharing a predecessor prefix walk it once.
    pub fn directed_legs_to_snaps(
        &self,
        from_point: (f64, f64),
        from: &Snap,
        targets: &[((f64, f64), Option<Snap>)],
        profile: &CompiledStreetProfile,
        max_seconds: f64,
    ) -> Vec<Option<StreetLeg>> {
        if !max_seconds.is_finite() || max_seconds < 0.0 {
            return vec![None; targets.len()];
        }
        let cutoff = (max_seconds * 1000.0).floor() as u64;
        let seeds = self.directed_seeds(from, profile);
        DIRECTED_STATE.with(|cell| {
            let state = &mut cell.borrow_mut();
            self.directed_dijkstra_with_paths(profile, &seeds, cutoff, state);
            let mut prefix = HashMap::new();
            targets
                .iter()
                .map(|(to_point, target)| {
                    let to = target.as_ref()?;
                    let millis = self.arrival_millis(from, to, profile, cutoff, state)?;
                    Some(self.assemble_leg(
                        from_point,
                        from,
                        *to_point,
                        to,
                        profile,
                        millis,
                        state,
                        &mut prefix,
                    ))
                })
                .collect()
        })
    }

    /// Builds the leg for a destination the search has already reached, whose
    /// best cost is `millis`.
    #[allow(clippy::too_many_arguments)]
    fn assemble_leg(
        &self,
        from_point: (f64, f64),
        from: &Snap,
        to_point: (f64, f64),
        to: &Snap,
        profile: &CompiledStreetProfile,
        millis: u64,
        state: &DirectedState,
        prefix: &mut HashMap<u32, f64>,
    ) -> StreetLeg {
        let connector_meters = from.connector + to.connector;
        let same_edge = self
            .same_edge_millis(from, to, profile)
            .is_some_and(|direct| direct == millis);
        // The winning route is the direct one along the shared edge when that
        // candidate is what produced the best time; otherwise it entered the
        // network and left it at one of the destination edge's endpoints.
        let (network_meters, path) = if same_edge {
            let length = self.arrays().lengths()[from.edge as usize];
            let mut path = vec![
                (from_point.1, from_point.0),
                self.point_at(from.edge, from.fraction),
            ];
            path.extend(self.edge_slice(from.edge, from.fraction, to.fraction));
            path.push(self.point_at(to.edge, to.fraction));
            path.push((to_point.1, to_point.0));
            ((to.fraction - from.fraction).abs() * length, path)
        } else {
            // Take the arrival that produced the winning time, and with it the
            // end of the destination edge it came in at — the vertex alone
            // would not say which end of a self-loop that is.
            let arrival = self
                .directed_egress(to, profile)
                .into_iter()
                .find(|end| {
                    let reached = state.time(end.vertex);
                    reached != u64::MAX && reached.saturating_add(end.millis) == millis
                })
                .expect("the winning arrival is one of the destination's endpoints");
            let exit = arrival.vertex;
            let (vertices, edges) = self.predecessor_chain(exit, state);
            let entry = vertices[0];
            let meters = self.chain_meters(&vertices, &edges, prefix);
            let from_length = self.arrays().lengths()[from.edge as usize];
            let to_length = self.arrays().lengths()[to.edge as usize];
            // The chain roots at a seed, so the seed whose cost is the entry's
            // settled label is the one the winning path left from.
            let entry_fraction = self
                .directed_seeds(from, profile)
                .into_iter()
                .find(|seed| seed.vertex == entry && seed.millis == state.time(entry))
                .map(|seed| seed.fraction)
                .expect("the winning path roots at one of the origin's seeds");
            let exit_fraction = arrival.fraction;
            let partial = (from.fraction - entry_fraction).abs() * from_length
                + (to.fraction - exit_fraction).abs() * to_length;
            let mut path = vec![
                (from_point.1, from_point.0),
                self.point_at(from.edge, from.fraction),
            ];
            path.extend(self.edge_slice(from.edge, from.fraction, entry_fraction));
            for (step, &edge) in edges.iter().enumerate() {
                let (u, _) = self.edge_endpoints(edge);
                let forward = vertices[step] == u;
                let (start, end) = if forward { (0.0, 1.0) } else { (1.0, 0.0) };
                path.extend(self.edge_slice(edge, start, end));
            }
            path.extend(self.edge_slice(to.edge, exit_fraction, to.fraction));
            path.push(self.point_at(to.edge, to.fraction));
            path.push((to_point.1, to_point.0));
            (meters + partial, path)
        };
        StreetLeg {
            seconds: seconds(millis as f64 / 1000.0),
            network_meters,
            connector_meters,
            geometry: dedup_consecutive(path),
        }
    }

    /// The predecessor chain back to the seed, as `(vertices, edges)` in
    /// travel order; `vertices[i]` is the tail of `edges[i]`.
    fn predecessor_chain(&self, exit: u32, state: &DirectedState) -> (Vec<u32>, Vec<u32>) {
        let mut vertices = vec![exit];
        let mut edges = Vec::new();
        let mut at = exit;
        loop {
            let (previous, edge) = state.previous[at as usize];
            if (previous, edge) == NO_PREVIOUS {
                break;
            }
            vertices.push(previous);
            edges.push(edge);
            at = previous;
        }
        vertices.reverse();
        edges.reverse();
        (vertices, edges)
    }

    /// The stored length of every whole edge on the chain, memoised per exit
    /// vertex so destinations sharing a prefix sum it once.
    fn chain_meters(&self, vertices: &[u32], edges: &[u32], prefix: &mut HashMap<u32, f64>) -> f64 {
        let lengths = self.arrays().lengths();
        let mut total = 0.0;
        for (step, &edge) in edges.iter().enumerate() {
            let reached = vertices[step + 1];
            if let Some(&memo) = prefix.get(&reached) {
                total = memo;
                continue;
            }
            total += lengths[edge as usize];
            prefix.insert(reached, total);
        }
        total
    }
}
