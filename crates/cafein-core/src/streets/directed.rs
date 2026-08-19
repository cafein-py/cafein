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
//! Door-to-door routes and origins×destinations matrices are **forward**
//! searches from each origin, reading the label at the destination's snap.
//! The **reverse** search serves the PT-egress side: one search from a
//! destination's arrival offsets over the transposed adjacency labels every
//! vertex with its cost *to* the destination, so a whole column of sources —
//! each stop that can reach it — reads from one settled state.

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
    /// The traversed edges as ``(edge index, traversed fraction, true
    /// traversal seconds)`` — partial snap edges at the ends, 1.0
    /// between, each edge's UNWEIGHTED time under the mode's own
    /// per-edge speed. Connectors are not edges and carry no entry.
    pub edges: Vec<(u32, f64, f64)>,
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

/// The route behind a settled cost, as the reconstruction reads it: the
/// direct ride along the shared snap edge, or the predecessor chain with
/// the partial-edge fractions where the route enters and leaves the snap
/// edges. Meters and geometry both derive from it.
enum WinningRoute {
    SameEdge,
    Chain {
        vertices: Vec<u32>,
        edges: Vec<u32>,
        entry_fraction: f64,
        exit_fraction: f64,
    },
}

/// The safety margin on the A* heuristic's straight-line metres: arc lengths
/// come from the extraction's geodesic while the heuristic measures with the
/// local-planar [`segment_length`], so the two can disagree by well under half
/// a percent. Scaling the distance down (after subtracting a one-metre
/// absolute guard for coordinate quantization) keeps the heuristic admissible
/// across that disagreement at a negligible cost in pruning.
const ASTAR_METRIC_GUARD: f64 = 0.995;

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
    /// Pending `(estimate, time, vertex)` entries of a goal-directed search —
    /// its own heap, so the Dijkstra one keeps its shape.
    astar_heap: BinaryHeap<Reverse<(u64, u64, u32)>>,
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
        self.astar_heap.clear();
    }

    /// How many vertices the last search labelled — the pruning metric the
    /// goal-directedness tests read.
    #[cfg(test)]
    pub(super) fn touched_count(&self) -> usize {
        self.touched.len()
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

    /// The `(vertex, milliseconds)` labels the last search settled.
    pub(super) fn settled(&self) -> impl Iterator<Item = (u32, u64)> + '_ {
        self.touched
            .iter()
            .map(|&vertex| (vertex, self.distances[vertex as usize]))
            .filter(|&(_, time)| time != u64::MAX)
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

/// The connector's millisecond cost at the profile's connector speed.
fn connector_millis(connector: f64, connector_speed: f64) -> u64 {
    (connector / connector_speed * 1000.0).ceil() as u64
}

/// The A* remaining-cost bound. Any route must still reach one of the
/// destination edge's permitted egress endpoints and pay its exact arrival
/// offset (partial edge and connector), so the remaining cost is at least the
/// guarded straight-line metres to the *ball* covering those endpoints —
/// centred between them, shrunk by half their span — at the profile's
/// greatest compiled arc speed, plus the smaller offset. One centre keeps the
/// evaluation to a single hypot on the relaxation hot path; ending on the
/// endpoints keeps the connector out of the bounded segment, so
/// `connector_speed` — which validation does not bound by `max_speed` —
/// cannot break admissibility.
///
/// Distances are measured in one fixed Euclidean frame whose per-degree
/// scales are the minima over the query's reachable latitude band — a true
/// metric (so the bound is consistent) that never exceeds the local planar
/// distance (so it stays a lower bound). Longitude differences are plain:
/// extracts are contiguous and never cross the antimeridian, per the street
/// network's documented contract.
struct AstarHeuristic {
    /// The ball centre, degrees.
    lon: f64,
    lat: f64,
    per_lon: f64,
    per_lat: f64,
    /// Half the endpoints' span plus the one-metre quantization guard.
    slack: f64,
    /// Guarded milliseconds per frame metre.
    scale: f64,
    /// The smaller arrival offset, milliseconds.
    offset: u64,
}

impl AstarHeuristic {
    fn new(
        network: &StreetNetwork,
        seeds: &[Endpoint],
        arrivals: &[Endpoint],
        max_speed: f64,
        cutoff: u64,
    ) -> AstarHeuristic {
        // Targets read the vertex-coordinate table — the same metric the
        // speed bound is measured in, so the whole admissibility argument
        // rests on one coordinate per vertex whatever the input geometry
        // claims elsewhere.
        let targets: Vec<(f64, f64, u64)> = arrivals
            .iter()
            .map(|arrival| {
                let (lon, lat) = network.vertex_coordinates()[arrival.vertex as usize];
                (lon, lat, arrival.millis)
            })
            .collect();
        // Every vertex the search can label is within the cutoff of a seed;
        // 110 km per degree under-divides the true latitude scale, so the
        // band only overshoots. The scales are minimised over the band:
        // metres per longitude degree shrink toward the poles, metres per
        // latitude degree toward the equator.
        let latitudes = seeds
            .iter()
            .map(|seed| {
                let (_, lat) = network.vertex_coordinates()[seed.vertex as usize];
                if lat.is_nan() {
                    0.0
                } else {
                    lat
                }
            })
            .chain(targets.iter().map(|&(_, lat, _)| lat));
        let (low, high) = latitudes.fold((90.0f64, -90.0f64), |(low, high), lat| {
            (low.min(lat), high.max(lat))
        });
        let reach = cutoff as f64 / 1000.0 * max_speed / 110_000.0;
        let low = (low - reach).max(-90.0);
        let high = (high + reach).min(90.0);
        let widest = low.abs().max(high.abs());
        let flattest = if low <= 0.0 && high >= 0.0 {
            0.0
        } else {
            low.abs().min(high.abs())
        };
        let (per_lon, _) = meters_per_degree(widest);
        let per_lon = per_lon.max(0.0);
        let (_, per_lat) = meters_per_degree(flattest);
        // Collapse the (at most two) endpoints into one ball: the centre,
        // half their frame-metre span as slack, the smaller offset.
        let (first_lon, first_lat, first_offset) = targets[0];
        let (lon, lat, slack, offset) = if targets.len() == 1 {
            (first_lon, first_lat, 0.0, first_offset)
        } else {
            let (other_lon, other_lat, other_offset) = targets[1];
            let dx = (first_lon - other_lon) * per_lon;
            let dy = (first_lat - other_lat) * per_lat;
            (
                (first_lon + other_lon) / 2.0,
                (first_lat + other_lat) / 2.0,
                (dx * dx + dy * dy).sqrt() / 2.0,
                first_offset.min(other_offset),
            )
        };
        AstarHeuristic {
            lon,
            lat,
            per_lon,
            per_lat,
            slack: slack + 1.0,
            scale: ASTAR_METRIC_GUARD * 1000.0 / max_speed,
            offset,
        }
    }

    /// `h` at a vertex coordinate. Floored on conversion — rounding down
    /// only loosens the bound; an isolated vertex's NaN coordinate collapses
    /// the distance term to zero, which is merely a weaker bound.
    #[inline]
    fn bound(&self, (lon, lat): (f64, f64)) -> u64 {
        let dx = (lon - self.lon) * self.per_lon;
        let dy = (lat - self.lat) * self.per_lat;
        let meters = (dx * dx + dy * dy).sqrt() - self.slack;
        if meters <= 0.0 {
            return self.offset;
        }
        // A NaN coordinate lands here and casts to zero millis.
        ((meters * self.scale) as u64).saturating_add(self.offset)
    }
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
            |slot: Option<usize>| slot.filter(|&slot| profile.arc_millis()[slot] != u32::MAX);
        (permitted(forward), permitted(reverse))
    }

    /// Whether `profile` may traverse `edge` in either direction.
    pub fn edge_permits(&self, edge: u32, profile: &CompiledStreetProfile) -> bool {
        let (forward, reverse) = self.edge_slots(edge);
        [forward, reverse]
            .into_iter()
            .flatten()
            .any(|slot| profile.arc_millis()[slot] != u32::MAX)
    }

    /// Snaps a coordinate to the nearest edge whose `adj_access` carries
    /// `bit` in either direction — the stop-link builder's mode-aware snap,
    /// needing no compiled profile. `None` on a graph without attributes.
    pub fn snap_for_mode_bit(
        &self,
        latitude: f64,
        longitude: f64,
        max_snap_distance: f64,
        bit: u8,
    ) -> Option<Snap> {
        let attributes = self.street_attributes()?;
        self.snap_filtered(latitude, longitude, max_snap_distance, |edge| {
            let (forward, reverse) = self.edge_slots(edge);
            [forward, reverse]
                .into_iter()
                .flatten()
                .any(|slot| attributes.adj_access[slot] & bit != 0)
        })
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
        // A departing partial charges the junction at the endpoint reached
        // (the slot's head) and never the one behind the snap.
        if snap.fraction == 0.0 {
            seeds.push(Endpoint::new(from, connector, 0.0));
        } else if let Some(slot) = reverse {
            seeds.push(Endpoint::new(
                from,
                connector.saturating_add(profile.departing_partial(slot, snap.fraction)),
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
                connector.saturating_add(profile.departing_partial(slot, 1.0 - snap.fraction)),
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
        // An arriving partial enters the edge through the junction at the
        // slot's tail and stops mid-edge, so only that endpoint charges.
        if snap.fraction == 0.0 {
            offsets.push(Endpoint::new(from, connector, 0.0));
        } else if let Some(slot) = forward {
            offsets.push(Endpoint::new(
                from,
                connector.saturating_add(profile.arriving_partial(slot, snap.fraction)),
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
                connector.saturating_add(profile.arriving_partial(slot, 1.0 - snap.fraction)),
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
        // The destination is ahead along the edge (forward) or behind
        // (reverse). Interior snaps cross no junction; a trip starting
        // exactly at the traversal's tail vertex or ending exactly at its
        // head vertex charges that endpoint, as the graph path would.
        let (forward, reverse) = self.snap_arcs(from, profile);
        let (slot, enters_at_tail, exits_at_head) = if to.fraction > from.fraction {
            (forward, from.fraction == 0.0, to.fraction == 1.0)
        } else {
            (reverse, from.fraction == 1.0, to.fraction == 0.0)
        };
        slot.map(|slot| {
            connectors.saturating_add(profile.same_edge_partial(
                slot,
                span,
                enters_at_tail,
                exits_at_head,
            ))
        })
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
        let arc_millis = profile.arc_millis();
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
        let arc_millis = profile.arc_millis();
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
                if next > cutoff {
                    continue;
                }
                if next < state.time(target) {
                    state.set(target, next);
                    state.previous[target as usize] = (vertex, edges[slot]);
                    state.heap.push(Reverse((next, target)));
                } else if next == state.time(target) && cost > 0 {
                    // An equal-cost offer: keep the smallest (vertex, edge),
                    // so the recorded route is a pure function of the graph
                    // and never of relaxation order — the goal-directed
                    // search resolves the same tie the same way. Positive
                    // cost only: the pointer then always targets a strictly
                    // earlier label, keeping the forest acyclic (a zero-cost
                    // arc keeps its first-found predecessor).
                    let candidate = (vertex, edges[slot]);
                    if candidate < state.previous[target as usize] {
                        state.previous[target as usize] = candidate;
                    }
                }
            }
        }
    }

    /// A vertex's coordinate in (longitude, latitude) degrees — a vertex is
    /// its edge's first or last stored geometry coordinate. NaN for an
    /// isolated vertex, which no search ever reaches.
    fn vertex_coordinate(&self, vertex: u32) -> Option<(f64, f64)> {
        let offsets = self.arrays().adjacency_offsets();
        let slot = offsets[vertex as usize] as usize;
        if slot >= offsets[vertex as usize + 1] as usize {
            return None;
        }
        let edge = self.arrays().adj_edges()[slot] as usize;
        let (from, _) = self.edge_endpoints(edge as u32);
        let coordinates = self.arrays().coordinate_offsets();
        let position = if from == vertex {
            coordinates[edge] as usize
        } else {
            coordinates[edge + 1] as usize - 1
        };
        Some((
            degrees(self.arrays().lons()[position]),
            degrees(self.arrays().lats()[position]),
        ))
    }

    /// Per-vertex `(longitude, latitude)` degrees — the positions the
    /// catchment entries' vertices index into. `NaN` marks an isolated
    /// vertex no search reaches.
    pub fn vertex_positions(&self) -> &[(f64, f64)] {
        self.vertex_coordinates()
    }

    /// The per-vertex coordinate table, built once per network on the first
    /// goal-directed query: the heuristic's per-relaxation read becomes one
    /// contiguous slot instead of a chase through the CSR and geometry
    /// arrays.
    pub(super) fn vertex_coordinates(&self) -> &[(f64, f64)] {
        self.graph.vertex_coordinates.get_or_init(|| {
            (0..self.vertex_count())
                .map(|vertex| {
                    self.vertex_coordinate(vertex)
                        .unwrap_or((f64::NAN, f64::NAN))
                })
                .collect()
        })
    }

    /// The lazily computed greatest chord speed of `profile`'s permitted
    /// arcs, measured over the vertex-coordinate table (see
    /// [`CompiledStreetProfile::max_effective_speed`]). A permitted zero-cost
    /// arc spanning a nonzero chord is free spatial movement — no finite
    /// speed bounds it, so the result is infinite and the heuristic's
    /// distance term vanishes (the search degrades to Dijkstra ordering,
    /// still correct).
    fn chord_speed_bound(&self, profile: &CompiledStreetProfile) -> f64 {
        let coordinates = self.vertex_coordinates();
        let endpoints = self.arrays().endpoints();
        let edges = self.arrays().adj_edges();
        let mut fastest = 0.0f64;
        for (slot, &millis) in profile.arc_millis().iter().enumerate() {
            if millis == u32::MAX {
                continue;
            }
            let edge = edges[slot] as usize;
            let (from_lon, from_lat) = coordinates[endpoints[2 * edge] as usize];
            let (to_lon, to_lat) = coordinates[endpoints[2 * edge + 1] as usize];
            let chord = segment_length(from_lon, from_lat, to_lon, to_lat);
            if chord.is_nan() || chord <= 0.0 {
                continue;
            }
            if millis == 0 {
                return f64::INFINITY;
            }
            fastest = fastest.max(chord / (f64::from(millis) / 1000.0));
        }
        if fastest > 0.0 {
            fastest
        } else {
            profile.definition.max_speed
        }
    }

    /// Goal-directed single-pair search: Dijkstra over the same arcs, ordered
    /// by `g + h` where `h` lower-bounds the remaining cost to the
    /// destination, so the frontier leans toward the target and the loop can
    /// stop as soon as no queued label can beat the best completed total.
    ///
    /// `h` is the guarded straight-line metres toward the permitted egress
    /// endpoints at the lazily measured [`Self::chord_speed_bound`], plus the
    /// smaller exact arrival offset (see [`AstarHeuristic`]) — admissible
    /// because every distance in both the bound and the heuristic reads the
    /// one vertex-coordinate table, so the triangle inequality of that table
    /// alone carries the proof, whatever the stored edge lengths claim. The
    /// loop tolerates label reopening, so correctness needs admissibility
    /// only, not consistency. `best0` carries the direct same-edge candidate.
    ///
    /// Matrix rows keep [`directed_dijkstra`](Self::directed_dijkstra): a
    /// one-to-many search serves every target from one settled state, so a
    /// goal bias has nothing to prune there.
    fn directed_astar(
        &self,
        profile: &CompiledStreetProfile,
        seeds: &[Endpoint],
        to: &Snap,
        best0: Option<u64>,
        cutoff: u64,
        state: &mut DirectedState,
    ) -> Option<u64> {
        let mut best = best0.filter(|&direct| direct <= cutoff);
        let arrivals = self.directed_egress(to, profile);
        if arrivals.is_empty() {
            // No permitted way off the network at the destination: the only
            // possible route is the direct same-edge one.
            return best;
        }
        let max_speed = *profile
            .effective_speed_cache()
            .get_or_init(|| self.chord_speed_bound(profile));
        let heuristic = AstarHeuristic::new(self, seeds, &arrivals, max_speed, cutoff);
        state.prepare(self.vertex_count() as usize);
        let coordinates = self.vertex_coordinates();
        let offsets = self.arrays().adjacency_offsets();
        let targets = self.arrays().adj_targets();
        let arc_millis = profile.arc_millis();
        for seed in seeds {
            if seed.millis <= cutoff && seed.millis < state.time(seed.vertex) {
                state.set(seed.vertex, seed.millis);
                let bound = heuristic.bound(coordinates[seed.vertex as usize]);
                state.astar_heap.push(Reverse((
                    seed.millis.saturating_add(bound),
                    seed.millis,
                    seed.vertex,
                )));
            }
        }
        while let Some(Reverse((estimate, time, vertex))) = state.astar_heap.pop() {
            // Strictly worse only: entries tying the best still settle, so a
            // tied egress endpoint's label is final and the leg assembly
            // resolves ties exactly as a full Dijkstra search would.
            if best.is_some_and(|b| b < estimate) {
                break;
            }
            if time > state.time(vertex) {
                continue;
            }
            for arrival in &arrivals {
                if arrival.vertex == vertex {
                    let total = time.saturating_add(arrival.millis);
                    if total <= cutoff && best.is_none_or(|b| total < b) {
                        best = Some(total);
                    }
                }
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
                    let bound = heuristic.bound(coordinates[target as usize]);
                    state
                        .astar_heap
                        .push(Reverse((next.saturating_add(bound), next, target)));
                }
            }
        }
        best
    }

    /// [`directed_astar`](Self::directed_astar) additionally recording each
    /// vertex's predecessor `(vertex, edge)` — the same time/paths split the
    /// Dijkstra pair makes, for the same reason.
    fn directed_astar_with_paths(
        &self,
        profile: &CompiledStreetProfile,
        seeds: &[Endpoint],
        to: &Snap,
        best0: Option<u64>,
        cutoff: u64,
        state: &mut DirectedState,
    ) -> Option<u64> {
        let mut best = best0.filter(|&direct| direct <= cutoff);
        let arrivals = self.directed_egress(to, profile);
        if arrivals.is_empty() {
            return best;
        }
        let max_speed = *profile
            .effective_speed_cache()
            .get_or_init(|| self.chord_speed_bound(profile));
        let heuristic = AstarHeuristic::new(self, seeds, &arrivals, max_speed, cutoff);
        state.prepare_with_previous(self.vertex_count() as usize);
        let coordinates = self.vertex_coordinates();
        let offsets = self.arrays().adjacency_offsets();
        let targets = self.arrays().adj_targets();
        let edges = self.arrays().adj_edges();
        let arc_millis = profile.arc_millis();
        for seed in seeds {
            if seed.millis <= cutoff && seed.millis < state.time(seed.vertex) {
                state.set(seed.vertex, seed.millis);
                state.previous[seed.vertex as usize] = NO_PREVIOUS;
                let bound = heuristic.bound(coordinates[seed.vertex as usize]);
                state.astar_heap.push(Reverse((
                    seed.millis.saturating_add(bound),
                    seed.millis,
                    seed.vertex,
                )));
            }
        }
        while let Some(Reverse((estimate, time, vertex))) = state.astar_heap.pop() {
            // Strictly worse only: entries tying the best still settle, so a
            // tied egress endpoint's label is final and the leg assembly
            // resolves ties exactly as a full Dijkstra search would.
            if best.is_some_and(|b| b < estimate) {
                break;
            }
            if time > state.time(vertex) {
                continue;
            }
            for arrival in &arrivals {
                if arrival.vertex == vertex {
                    let total = time.saturating_add(arrival.millis);
                    if total <= cutoff && best.is_none_or(|b| total < b) {
                        best = Some(total);
                    }
                }
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
                if next > cutoff {
                    continue;
                }
                if next < state.time(target) {
                    state.set(target, next);
                    state.previous[target as usize] = (vertex, edges[slot]);
                    let bound = heuristic.bound(coordinates[target as usize]);
                    state
                        .astar_heap
                        .push(Reverse((next.saturating_add(bound), next, target)));
                } else if next == state.time(target) && cost > 0 {
                    // The same order-independent, acyclicity-preserving tie
                    // rule as the Dijkstra path search.
                    let candidate = (vertex, edges[slot]);
                    if candidate < state.previous[target as usize] {
                        state.previous[target as usize] = candidate;
                    }
                }
            }
        }
        best
    }

    /// How many vertices A* and cutoff-bounded Dijkstra each label for the
    /// same single-pair query — the goal-directedness and benchmark metric.
    #[cfg(test)]
    pub(super) fn astar_versus_dijkstra_touched(
        &self,
        profile: &CompiledStreetProfile,
        from: &Snap,
        to: &Snap,
        max_seconds: f64,
    ) -> (usize, usize) {
        let cutoff = (max_seconds * 1000.0).floor() as u64;
        let seeds = self.directed_seeds(from, profile);
        let best0 = self.same_edge_millis(from, to, profile);
        let mut state = DirectedState::default();
        self.directed_astar(profile, &seeds, to, best0, cutoff, &mut state);
        let astar = state.touched_count();
        let mut state = DirectedState::default();
        self.directed_dijkstra(profile, &seeds, cutoff, &mut state);
        (astar, state.touched_count())
    }

    /// The lazily built transpose of the adjacency CSR: `(offsets, sources,
    /// slots)`, where vertex `v`'s incoming arcs are
    /// `offsets[v]..offsets[v + 1]`, each arriving from `sources[i]` over
    /// adjacency slot `slots[i]`.
    fn reverse_adjacency(&self) -> &(Vec<u32>, Vec<u32>, Vec<u32>) {
        self.graph.reverse_adjacency.get_or_init(|| {
            let offsets = self.arrays().adjacency_offsets();
            let targets = self.arrays().adj_targets();
            let vertices = self.vertex_count() as usize;
            let mut counts = vec![0u32; vertices + 1];
            for &target in targets {
                counts[target as usize + 1] += 1;
            }
            for vertex in 0..vertices {
                counts[vertex + 1] += counts[vertex];
            }
            let mut sources = vec![0u32; targets.len()];
            let mut slots = vec![0u32; targets.len()];
            let mut cursor: Vec<u32> = counts.clone();
            for vertex in 0..vertices {
                for slot in offsets[vertex] as usize..offsets[vertex + 1] as usize {
                    let position = cursor[targets[slot] as usize] as usize;
                    sources[position] = vertex as u32;
                    slots[position] = slot as u32;
                    cursor[targets[slot] as usize] += 1;
                }
            }
            (counts, sources, slots)
        })
    }

    /// The reverse counterpart of [`directed_dijkstra`](Self::directed_dijkstra):
    /// relaxes *incoming* arcs over the transposed view, so a settled label is
    /// the millisecond cost **from** that vertex **to** the destination whose
    /// egress endpoints seeded the search. Same cutoff and sentinel semantics.
    pub(super) fn directed_dijkstra_reverse(
        &self,
        profile: &CompiledStreetProfile,
        seeds: &[Endpoint],
        cutoff: u64,
        state: &mut DirectedState,
    ) {
        state.prepare(self.vertex_count() as usize);
        let (offsets, sources, slots) = self.reverse_adjacency();
        let arc_millis = profile.arc_millis();
        for seed in seeds {
            if seed.millis <= cutoff && seed.millis < state.time(seed.vertex) {
                state.set(seed.vertex, seed.millis);
                state.heap.push(Reverse((seed.millis, seed.vertex)));
            }
        }
        while let Some(Reverse((time, vertex))) = state.heap.pop() {
            if time > state.time(vertex) {
                continue;
            }
            let start = offsets[vertex as usize] as usize;
            let end = offsets[vertex as usize + 1] as usize;
            for incoming in start..end {
                let cost = arc_millis[slots[incoming] as usize];
                if cost == u32::MAX {
                    continue;
                }
                let next = time.saturating_add(u64::from(cost));
                let target = sources[incoming];
                if next <= cutoff && next < state.time(target) {
                    state.set(target, next);
                    state.heap.push(Reverse((next, target)));
                }
            }
        }
    }

    /// [`directed_dijkstra_reverse`](Self::directed_dijkstra_reverse)
    /// additionally recording each settled vertex's next hop
    /// `(vertex, edge)` toward the destination — the reverse counterpart
    /// of [`directed_dijkstra_with_paths`](Self::directed_dijkstra_with_paths),
    /// with the same order-independent, acyclicity-preserving tie rule.
    pub(super) fn directed_dijkstra_reverse_with_paths(
        &self,
        profile: &CompiledStreetProfile,
        seeds: &[Endpoint],
        cutoff: u64,
        state: &mut DirectedState,
    ) {
        state.prepare_with_previous(self.vertex_count() as usize);
        let (offsets, sources, slots) = self.reverse_adjacency();
        let edges = self.arrays().adj_edges();
        let arc_millis = profile.arc_millis();
        for seed in seeds {
            if seed.millis <= cutoff && seed.millis < state.time(seed.vertex) {
                state.set(seed.vertex, seed.millis);
                state.previous[seed.vertex as usize] = NO_PREVIOUS;
                state.heap.push(Reverse((seed.millis, seed.vertex)));
            }
        }
        while let Some(Reverse((time, vertex))) = state.heap.pop() {
            if time > state.time(vertex) {
                continue;
            }
            let start = offsets[vertex as usize] as usize;
            let end = offsets[vertex as usize + 1] as usize;
            for incoming in start..end {
                let slot = slots[incoming] as usize;
                let cost = arc_millis[slot];
                if cost == u32::MAX {
                    continue;
                }
                let next = time.saturating_add(u64::from(cost));
                let target = sources[incoming];
                if next > cutoff {
                    continue;
                }
                if next < state.time(target) {
                    state.set(target, next);
                    state.previous[target as usize] = (vertex, edges[slot]);
                    state.heap.push(Reverse((next, target)));
                } else if next == state.time(target) && cost > 0 {
                    let candidate = (vertex, edges[slot]);
                    if candidate < state.previous[target as usize] {
                        state.previous[target as usize] = candidate;
                    }
                }
            }
        }
    }

    /// The directed times and network meters from each of `sources` to
    /// `to` — the egress mirror of
    /// [`directed_meters_to_snaps`](Self::directed_meters_to_snaps): one
    /// path-tracking reverse search serves the whole column, chains
    /// toward the destination memoise across sources, and seconds are
    /// identical to [`directed_times_from_snaps`](Self::directed_times_from_snaps)
    /// cell for cell (the direct same-edge candidate wins exact ties, as
    /// the forward reconstruction resolves them). The meters are the
    /// meters of an optimal route: on an exact integer-millisecond tie
    /// between physically different routes, the reverse tree may keep a
    /// different — equally optimal — one than a forward reconstruction
    /// of the same pair, its per-direction choice deterministic either
    /// way. Canonicalising across directions would cost the
    /// one-search-per-column shape, and the time contract is unaffected.
    pub fn directed_meters_from_snaps(
        &self,
        sources: &[Option<Snap>],
        to: &Snap,
        profile: &CompiledStreetProfile,
        max_seconds: f64,
    ) -> Vec<Option<(u32, f64)>> {
        if !max_seconds.is_finite() || max_seconds < 0.0 {
            return vec![None; sources.len()];
        }
        let cutoff = (max_seconds * 1000.0).floor() as u64;
        let seeds = self.directed_egress(to, profile);
        DIRECTED_STATE.with(|cell| {
            let state = &mut cell.borrow_mut();
            self.directed_dijkstra_reverse_with_paths(profile, &seeds, cutoff, state);
            let lengths = self.arrays().lengths();
            let mut suffix: HashMap<u32, (f64, f64)> = HashMap::new();
            sources
                .iter()
                .map(|source| {
                    let from = source.as_ref()?;
                    // The times column's fold, remembering which candidate
                    // produced the winning total.
                    let mut best: Option<(u64, Option<Endpoint>)> = self
                        .same_edge_millis(from, to, profile)
                        .filter(|&direct| direct <= cutoff)
                        .map(|direct| (direct, None));
                    for departure in self.directed_seeds(from, profile) {
                        let reached = state.time(departure.vertex);
                        if reached == u64::MAX {
                            continue;
                        }
                        let total = reached.saturating_add(departure.millis);
                        if total > cutoff {
                            continue;
                        }
                        let better = match &best {
                            None => true,
                            Some((held, _)) => total < *held,
                        };
                        if better {
                            best = Some((total, Some(departure)));
                        }
                    }
                    let (millis, winner) = best?;
                    let meters = match winner {
                        None => (to.fraction - from.fraction).abs() * lengths[from.edge as usize],
                        Some(departure) => {
                            let (chain, exit_fraction) =
                                self.suffix_meters(departure.vertex, &seeds, state, &mut suffix);
                            chain
                                + (from.fraction - departure.fraction).abs()
                                    * lengths[from.edge as usize]
                                + (to.fraction - exit_fraction).abs() * lengths[to.edge as usize]
                        }
                    };
                    Some((seconds(millis as f64 / 1000.0), meters))
                })
                .collect()
        })
    }

    /// Meters from a settled vertex along its recorded chain to the
    /// destination root, and the root's exit fraction — memoised per
    /// vertex so sources sharing a suffix walk it once.
    fn suffix_meters(
        &self,
        start: u32,
        seeds: &[Endpoint],
        state: &DirectedState,
        suffix: &mut HashMap<u32, (f64, f64)>,
    ) -> (f64, f64) {
        let lengths = self.arrays().lengths();
        let mut walk: Vec<(u32, u32)> = Vec::new();
        let mut at = start;
        let (base, exit_fraction) = loop {
            if let Some(&memo) = suffix.get(&at) {
                break memo;
            }
            let (next, edge) = state.previous[at as usize];
            if (next, edge) == NO_PREVIOUS {
                // The chain roots at a seeded arrival offset; its label
                // says which end of the destination edge it is.
                let fraction = seeds
                    .iter()
                    .find(|seed| seed.vertex == at && seed.millis == state.time(at))
                    .map(|seed| seed.fraction)
                    .expect("the chain roots at one of the destination's arrival offsets");
                break (0.0, fraction);
            }
            walk.push((at, edge));
            at = next;
            // The forest is acyclic by construction; a longer chain is
            // corrupt state, and failing beats hanging.
            assert!(
                walk.len() <= self.vertex_count() as usize + 1,
                "predecessor chain exceeds the vertex count"
            );
        };
        let mut total = base;
        for &(vertex, edge) in walk.iter().rev() {
            total += lengths[edge as usize];
            suffix.insert(vertex, (total, exit_fraction));
        }
        (total, exit_fraction)
    }

    /// The directed travel times from each of `sources` to `to`, in whole
    /// seconds — the egress mirror of
    /// [`directed_times_to_snaps`](Self::directed_times_to_snaps): one reverse
    /// search from the destination's arrival offsets serves the whole column,
    /// and each source reads its own departure seeds against the settled
    /// labels plus the direct same-edge candidate.
    pub fn directed_times_from_snaps(
        &self,
        sources: &[Option<Snap>],
        to: &Snap,
        profile: &CompiledStreetProfile,
        max_seconds: f64,
    ) -> Vec<Option<u32>> {
        if !max_seconds.is_finite() || max_seconds < 0.0 {
            return vec![None; sources.len()];
        }
        let cutoff = (max_seconds * 1000.0).floor() as u64;
        let seeds = self.directed_egress(to, profile);
        DIRECTED_STATE.with(|cell| {
            let state = &mut cell.borrow_mut();
            self.directed_dijkstra_reverse(profile, &seeds, cutoff, state);
            sources
                .iter()
                .map(|source| {
                    let from = source.as_ref()?;
                    let mut best = self
                        .same_edge_millis(from, to, profile)
                        .filter(|&direct| direct <= cutoff);
                    for departure in self.directed_seeds(from, profile) {
                        let reached = state.time(departure.vertex);
                        if reached != u64::MAX {
                            let total = reached.saturating_add(departure.millis);
                            if total <= cutoff {
                                best = Some(best.map_or(total, |b| b.min(total)));
                            }
                        }
                    }
                    Some(seconds(best? as f64 / 1000.0))
                })
                .collect()
        })
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
    /// within `max_seconds`. A goal-directed A* search from the origin's
    /// directed seeds toward the destination's arrival offsets, with the
    /// direct same-edge candidate carried in — answer for answer identical to
    /// the Dijkstra row a matrix computes, just settling less of the graph.
    pub fn directed_travel_time(
        &self,
        from: &Snap,
        to: &Snap,
        profile: &CompiledStreetProfile,
        max_seconds: f64,
    ) -> Option<u32> {
        if !max_seconds.is_finite() || max_seconds < 0.0 {
            return None;
        }
        let cutoff = (max_seconds * 1000.0).floor() as u64;
        let seeds = self.directed_seeds(from, profile);
        let best0 = self.same_edge_millis(from, to, profile);
        DIRECTED_STATE.with(|cell| {
            let state = &mut cell.borrow_mut();
            let millis = self.directed_astar(profile, &seeds, to, best0, cutoff, state)?;
            Some(seconds(millis as f64 / 1000.0))
        })
    }

    /// The settled vertices of a bounded directed search from `from`:
    /// `(vertex, seconds)` for every graph vertex within `max_seconds`
    /// under `profile` — a street catchment's target universe.
    /// The settled vertices of a time-bounded multi-seed spread under
    /// `profile` — the profile-aware catchment field, the directed twin
    /// of `walk_field`. Each seed is a snap with its initial seconds (a
    /// stop's transit arrival cost, zero for the origin); the spread
    /// walks only the profile's permitted arcs, so a wheelchair field
    /// never crosses stairs or capped gradients. `(vertex, seconds)`
    /// within `cutoff_seconds`.
    pub fn directed_field(
        &self,
        seeds: &[(Snap, f64)],
        profile: &CompiledStreetProfile,
        cutoff_seconds: f64,
    ) -> Vec<(u32, f64)> {
        if !cutoff_seconds.is_finite() || cutoff_seconds < 0.0 {
            return Vec::new();
        }
        let cutoff = (cutoff_seconds * 1000.0).floor() as u64;
        let mut weighted: Vec<Endpoint> = Vec::with_capacity(seeds.len() * 2);
        for (snap, initial_seconds) in seeds {
            if !initial_seconds.is_finite() || *initial_seconds < 0.0 {
                continue;
            }
            let head_start = (initial_seconds * 1000.0).floor() as u64;
            if head_start > cutoff {
                continue;
            }
            for seed in self.directed_seeds(snap, profile) {
                weighted.push(Endpoint::new(
                    seed.vertex,
                    seed.millis.saturating_add(head_start),
                    seed.fraction,
                ));
            }
        }
        if weighted.is_empty() {
            return Vec::new();
        }
        DIRECTED_STATE.with(|cell| {
            let state = &mut cell.borrow_mut();
            self.directed_dijkstra(profile, &weighted, cutoff, state);
            state
                .settled()
                .filter(|&(_, millis)| millis <= cutoff)
                .map(|(vertex, millis)| (vertex, millis as f64 / 1000.0))
                .collect()
        })
    }

    pub fn directed_reached_vertices(
        &self,
        from: &Snap,
        profile: &CompiledStreetProfile,
        max_seconds: f64,
    ) -> Vec<(u32, f64)> {
        if !max_seconds.is_finite() || max_seconds < 0.0 {
            return Vec::new();
        }
        let cutoff = (max_seconds * 1000.0).floor() as u64;
        let seeds = self.directed_seeds(from, profile);
        DIRECTED_STATE.with(|cell| {
            let state = &mut cell.borrow_mut();
            self.directed_dijkstra(profile, &seeds, cutoff, state);
            state
                .settled()
                .filter(|&(_, millis)| millis <= cutoff)
                .map(|(vertex, millis)| (vertex, millis as f64 / 1000.0))
                .collect()
        })
    }

    /// The settled vertices of a metres-bounded spread from `from`
    /// under `profile`'s permissions — the iso-distance catchment's
    /// target universe: `(vertex, meters)` within `max_meters`.
    /// Directionality follows the profile exactly as the timed
    /// spread does; only permitted arcs are walked, weighted by their
    /// exact street length (`f64` accumulation, as the walking
    /// searches measure it — never per-arc rounding).
    pub fn directed_reached_vertices_meters(
        &self,
        from: &Snap,
        profile: &CompiledStreetProfile,
        max_meters: f64,
    ) -> Vec<(u32, f64)> {
        if !max_meters.is_finite() || max_meters < 0.0 {
            return Vec::new();
        }
        let (from_vertex, to_vertex) = self.edge_endpoints(from.edge);
        let (forward, reverse) = self.snap_arcs(from, profile);
        let length = self.arrays().lengths()[from.edge as usize];
        let mut seeds: Vec<(u32, f64)> = Vec::with_capacity(2);
        if from.fraction == 0.0 || reverse.is_some() {
            seeds.push((from_vertex, from.connector + from.fraction * length));
        }
        if from.fraction == 1.0 || forward.is_some() {
            seeds.push((to_vertex, from.connector + (1.0 - from.fraction) * length));
        }
        let offsets = self.arrays().adjacency_offsets();
        let targets = self.arrays().adj_targets();
        let meters_of = self.arrays().adj_meters();
        let arc_millis = profile.arc_millis();
        super::search::SEARCH_STATE.with(|cell| {
            let state = &mut cell.borrow_mut();
            state.prepare(self.vertex_count() as usize);
            for &(vertex, start) in &seeds {
                if start <= max_meters + 1e-9 && start < state.distance(vertex) {
                    state.set_distance(vertex, start);
                    state.heap.push(Reverse((start.to_bits(), vertex)));
                }
            }
            while let Some(Reverse((bits, vertex))) = state.heap.pop() {
                let reached = f64::from_bits(bits);
                if reached > state.distance(vertex) {
                    continue;
                }
                let start = offsets[vertex as usize] as usize;
                let end = offsets[vertex as usize + 1] as usize;
                for slot in start..end {
                    if arc_millis[slot] == u32::MAX {
                        continue;
                    }
                    let next = reached + meters_of[slot];
                    let target = targets[slot];
                    if next <= max_meters + 1e-9 && next < state.distance(target) {
                        state.set_distance(target, next);
                        state.heap.push(Reverse((next.to_bits(), target)));
                    }
                }
            }
            use super::search::Reached;

            let mut field = Vec::new();
            state.for_each_reached(|vertex, meters| {
                if meters <= max_meters + 1e-9 {
                    field.push((vertex, meters));
                }
            });
            field
        })
    }

    /// The directed travel times from `from` to each of `targets`, in whole
    /// seconds, or `None` per target that is unsnapped or beyond `max_seconds`.
    ///
    /// One bounded search serves the whole row: every target reads the same
    /// settled labels, so a matrix row costs one street search rather than one
    /// per pair. The single route runs a goal-directed search instead, over
    /// the same seeds, egress offsets, and same-edge candidate — the equality
    /// tests hold the two answers identical cell for cell.
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
    /// within `max_seconds` — the goal-directed counterpart of one row cell,
    /// assembling the leg from the A* predecessors.
    pub fn directed_leg(
        &self,
        from_point: (f64, f64),
        from: &Snap,
        to_point: (f64, f64),
        to: &Snap,
        profile: &CompiledStreetProfile,
        max_seconds: f64,
    ) -> Option<StreetLeg> {
        if !max_seconds.is_finite() || max_seconds < 0.0 {
            return None;
        }
        let cutoff = (max_seconds * 1000.0).floor() as u64;
        let seeds = self.directed_seeds(from, profile);
        let best0 = self.same_edge_millis(from, to, profile);
        DIRECTED_STATE.with(|cell| {
            let state = &mut cell.borrow_mut();
            let millis =
                self.directed_astar_with_paths(profile, &seeds, to, best0, cutoff, state)?;
            let mut prefix = HashMap::new();
            Some(self.assemble_leg(
                from_point,
                from,
                to_point,
                to,
                profile,
                None,
                millis,
                state,
                &mut prefix,
            ))
        })
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
        self.legs_to_snaps_impl(from_point, from, targets, profile, None, max_seconds)
    }

    /// [`directed_legs_to_snaps`](Self::directed_legs_to_snaps) under an
    /// objective-weighted profile: the search and route choice ride
    /// `weighted`, while every reported second recomputes the chosen
    /// route under `unweighted` — the weighting bends the choice, never
    /// the clock. `max_seconds` bounds the WEIGHTED (perceived) cost;
    /// the multiplier is at least 1, so every returned leg's true time
    /// is within the budget too.
    pub fn directed_legs_to_snaps_weighted(
        &self,
        from_point: (f64, f64),
        from: &Snap,
        targets: &[((f64, f64), Option<Snap>)],
        weighted: &CompiledStreetProfile,
        unweighted: &CompiledStreetProfile,
        max_seconds: f64,
    ) -> Vec<Option<StreetLeg>> {
        self.legs_to_snaps_impl(
            from_point,
            from,
            targets,
            weighted,
            Some(unweighted),
            max_seconds,
        )
    }

    fn legs_to_snaps_impl(
        &self,
        from_point: (f64, f64),
        from: &Snap,
        targets: &[((f64, f64), Option<Snap>)],
        profile: &CompiledStreetProfile,
        unweighted: Option<&CompiledStreetProfile>,
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
                        unweighted,
                        millis,
                        state,
                        &mut prefix,
                    ))
                })
                .collect()
        })
    }

    /// The directed times and network meters from `from` to each of
    /// `targets` — the row [`directed_times_to_snaps`](Self::directed_times_to_snaps)
    /// computes, each reachable cell additionally carrying the winning
    /// route's ridden meters (partial snap edges included, connectors
    /// not). One path-tracking search serves the whole row and the chain
    /// meters memoise across targets; no geometry is assembled. Seconds
    /// are identical to the times-only row cell for cell.
    pub fn directed_meters_to_snaps(
        &self,
        from: &Snap,
        targets: &[Option<Snap>],
        profile: &CompiledStreetProfile,
        max_seconds: f64,
    ) -> Vec<Option<(u32, f64)>> {
        self.meters_to_snaps_impl(from, targets, profile, None, max_seconds)
    }

    /// [`directed_meters_to_snaps`](Self::directed_meters_to_snaps) under
    /// an objective-weighted profile — the same choice/clock split as
    /// [`directed_legs_to_snaps_weighted`](Self::directed_legs_to_snaps_weighted).
    pub fn directed_meters_to_snaps_weighted(
        &self,
        from: &Snap,
        targets: &[Option<Snap>],
        weighted: &CompiledStreetProfile,
        unweighted: &CompiledStreetProfile,
        max_seconds: f64,
    ) -> Vec<Option<(u32, f64)>> {
        self.meters_to_snaps_impl(from, targets, weighted, Some(unweighted), max_seconds)
    }

    fn meters_to_snaps_impl(
        &self,
        from: &Snap,
        targets: &[Option<Snap>],
        profile: &CompiledStreetProfile,
        unweighted: Option<&CompiledStreetProfile>,
        max_seconds: f64,
    ) -> Vec<Option<(u32, f64)>> {
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
                .map(|target| {
                    let to = target.as_ref()?;
                    let millis = self.arrival_millis(from, to, profile, cutoff, state)?;
                    let route = self.winning_route(from, to, profile, millis, state);
                    let meters = match &route {
                        WinningRoute::SameEdge => {
                            let length = self.arrays().lengths()[from.edge as usize];
                            (to.fraction - from.fraction).abs() * length
                        }
                        WinningRoute::Chain {
                            vertices,
                            edges,
                            entry_fraction,
                            exit_fraction,
                        } => {
                            self.chain_meters(vertices, edges, &mut prefix)
                                + self.partial_meters(from, to, *entry_fraction, *exit_fraction)
                        }
                    };
                    let millis = match unweighted {
                        Some(profile) => self.route_millis(from, to, &route, profile),
                        None => millis,
                    };
                    Some((seconds(millis as f64 / 1000.0), meters))
                })
                .collect()
        })
    }

    /// Builds the leg for a destination the search has already reached, whose
    /// best cost is `millis`. With `unweighted` set, the search (and so the
    /// route choice) ran an objective-weighted profile; the reported seconds
    /// recompute the chosen route under the true costs.
    #[allow(clippy::too_many_arguments)]
    fn assemble_leg(
        &self,
        from_point: (f64, f64),
        from: &Snap,
        to_point: (f64, f64),
        to: &Snap,
        profile: &CompiledStreetProfile,
        unweighted: Option<&CompiledStreetProfile>,
        millis: u64,
        state: &DirectedState,
        prefix: &mut HashMap<u32, f64>,
    ) -> StreetLeg {
        let connector_meters = from.connector + to.connector;
        let route = self.winning_route(from, to, profile, millis, state);
        let (network_meters, path) = match &route {
            WinningRoute::SameEdge => {
                let length = self.arrays().lengths()[from.edge as usize];
                let mut path = vec![
                    (from_point.1, from_point.0),
                    self.point_at(from.edge, from.fraction),
                ];
                path.extend(self.edge_slice(from.edge, from.fraction, to.fraction));
                path.push(self.point_at(to.edge, to.fraction));
                path.push((to_point.1, to_point.0));
                ((to.fraction - from.fraction).abs() * length, path)
            }
            WinningRoute::Chain {
                vertices,
                edges,
                entry_fraction,
                exit_fraction,
            } => {
                let meters = self.chain_meters(vertices, edges, prefix)
                    + self.partial_meters(from, to, *entry_fraction, *exit_fraction);
                let mut path = vec![
                    (from_point.1, from_point.0),
                    self.point_at(from.edge, from.fraction),
                ];
                path.extend(self.edge_slice(from.edge, from.fraction, *entry_fraction));
                for (step, &edge) in edges.iter().enumerate() {
                    let (u, _) = self.edge_endpoints(edge);
                    let forward = vertices[step] == u;
                    let (start, end) = if forward { (0.0, 1.0) } else { (1.0, 0.0) };
                    path.extend(self.edge_slice(edge, start, end));
                }
                path.extend(self.edge_slice(to.edge, *exit_fraction, to.fraction));
                path.push(self.point_at(to.edge, to.fraction));
                path.push((to_point.1, to_point.0));
                (meters, path)
            }
        };
        let millis = match unweighted {
            Some(profile) => self.route_millis(from, to, &route, profile),
            None => millis,
        };
        StreetLeg {
            seconds: seconds(millis as f64 / 1000.0),
            network_meters,
            connector_meters,
            geometry: dedup_consecutive(path),
            edges: self.route_edges(from, to, &route, unweighted.unwrap_or(profile)),
        }
    }

    /// The route's cost under `profile` — how the seed, chain, and egress
    /// compose is exactly what the search summed, so recomputing under the
    /// profile the search ran reproduces its settled cost bit-for-bit, and
    /// recomputing under the TRUE profile prices a weighted search's chosen
    /// route at its true clock. Permissions are identical between the two
    /// (weighting preserves the forbidden sentinel), so every lookup finds
    /// its counterpart.
    fn route_millis(
        &self,
        from: &Snap,
        to: &Snap,
        route: &WinningRoute,
        profile: &CompiledStreetProfile,
    ) -> u64 {
        match route {
            WinningRoute::SameEdge => self
                .same_edge_millis(from, to, profile)
                .expect("the same-edge route is permitted under the same mode"),
            WinningRoute::Chain {
                vertices,
                edges,
                entry_fraction,
                exit_fraction,
            } => {
                let seed = self
                    .directed_seeds(from, profile)
                    .into_iter()
                    .find(|seed| seed.vertex == vertices[0] && seed.fraction == *entry_fraction)
                    .expect("the route's entry seed is permitted under the same mode");
                let exit = *vertices.last().expect("a chain has at least one vertex");
                let egress = self
                    .directed_egress(to, profile)
                    .into_iter()
                    .find(|end| end.vertex == exit && end.fraction == *exit_fraction)
                    .expect("the route's exit is permitted under the same mode");
                let mut total = seed.millis;
                for (step, &edge) in edges.iter().enumerate() {
                    let slot = self.arc_slot(vertices[step], vertices[step + 1], edge);
                    total = total.saturating_add(u64::from(profile.arc_millis()[slot]));
                }
                total.saturating_add(egress.millis)
            }
        }
    }

    /// The route's traversed edges as ``(edge, fraction, seconds)`` —
    /// the partial snap edges at the ends (zero when the route enters at
    /// the snap itself), 1.0 for the whole edges between, each with its
    /// TRUE traversal time under `profile` (street profiles ride
    /// different speeds per edge, so length shares are not time shares).
    /// Connector time is subtracted out of the end partials.
    fn route_edges(
        &self,
        from: &Snap,
        to: &Snap,
        route: &WinningRoute,
        profile: &CompiledStreetProfile,
    ) -> Vec<(u32, f64, f64)> {
        let connector_speed = profile.definition.connector_speed;
        let elapsed = |millis: u64| millis as f64 / 1000.0;
        match route {
            WinningRoute::SameEdge => {
                let connectors = connector_millis(from.connector, connector_speed)
                    .saturating_add(connector_millis(to.connector, connector_speed));
                let on_edge = self
                    .same_edge_millis(from, to, profile)
                    .expect("the same-edge route is permitted under the same mode")
                    .saturating_sub(connectors);
                vec![(
                    from.edge,
                    (to.fraction - from.fraction).abs(),
                    elapsed(on_edge),
                )]
            }
            WinningRoute::Chain {
                vertices,
                edges,
                entry_fraction,
                exit_fraction,
            } => {
                let seed = self
                    .directed_seeds(from, profile)
                    .into_iter()
                    .find(|seed| seed.vertex == vertices[0] && seed.fraction == *entry_fraction)
                    .expect("the route's entry seed is permitted under the same mode");
                let entry = seed
                    .millis
                    .saturating_sub(connector_millis(from.connector, connector_speed));
                let exit = *vertices.last().expect("a chain has at least one vertex");
                let egress = self
                    .directed_egress(to, profile)
                    .into_iter()
                    .find(|end| end.vertex == exit && end.fraction == *exit_fraction)
                    .expect("the route's exit is permitted under the same mode");
                let leave = egress
                    .millis
                    .saturating_sub(connector_millis(to.connector, connector_speed));
                let mut traversed = Vec::with_capacity(edges.len() + 2);
                traversed.push((
                    from.edge,
                    (from.fraction - entry_fraction).abs(),
                    elapsed(entry),
                ));
                for (step, &edge) in edges.iter().enumerate() {
                    let slot = self.arc_slot(vertices[step], vertices[step + 1], edge);
                    traversed.push((edge, 1.0, f64::from(profile.arc_millis()[slot]) / 1000.0));
                }
                traversed.push((to.edge, (to.fraction - exit_fraction).abs(), elapsed(leave)));
                traversed
            }
        }
    }

    /// The adjacency slot of `edge` leaving `tail` for `head`. A
    /// self-loop's two arcs are indistinguishable here and resolve to the
    /// first, mirroring the reconstruction's own direction convention.
    fn arc_slot(&self, tail: u32, head: u32, edge: u32) -> usize {
        let arrays = self.arrays();
        let start = arrays.adjacency_offsets()[tail as usize] as usize;
        let end = arrays.adjacency_offsets()[tail as usize + 1] as usize;
        (start..end)
            .find(|&slot| arrays.adj_edges()[slot] == edge && arrays.adj_targets()[slot] == head)
            .expect("the chain's arc exists in the adjacency")
    }

    /// The winning route behind a settled forward-search cost: the direct
    /// same-edge candidate when that is what produced it, else the
    /// predecessor chain with the partial-edge fractions at both ends.
    fn winning_route(
        &self,
        from: &Snap,
        to: &Snap,
        profile: &CompiledStreetProfile,
        millis: u64,
        state: &DirectedState,
    ) -> WinningRoute {
        let same_edge = self
            .same_edge_millis(from, to, profile)
            .is_some_and(|direct| direct == millis);
        if same_edge {
            return WinningRoute::SameEdge;
        }
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
        let (vertices, edges) = self.predecessor_chain(arrival.vertex, state);
        let entry = vertices[0];
        // The chain roots at a seed, so the seed whose cost is the entry's
        // settled label is the one the winning path left from.
        let entry_fraction = self
            .directed_seeds(from, profile)
            .into_iter()
            .find(|seed| seed.vertex == entry && seed.millis == state.time(entry))
            .map(|seed| seed.fraction)
            .expect("the winning path roots at one of the origin's seeds");
        WinningRoute::Chain {
            vertices,
            edges,
            entry_fraction,
            exit_fraction: arrival.fraction,
        }
    }

    /// The partial lengths ridden on the snap edges at both route ends.
    fn partial_meters(
        &self,
        from: &Snap,
        to: &Snap,
        entry_fraction: f64,
        exit_fraction: f64,
    ) -> f64 {
        let lengths = self.arrays().lengths();
        (from.fraction - entry_fraction).abs() * lengths[from.edge as usize]
            + (to.fraction - exit_fraction).abs() * lengths[to.edge as usize]
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
            // The forest is acyclic by construction; a longer chain is
            // corrupt state, and failing beats hanging.
            assert!(
                vertices.len() <= self.vertex_count() as usize + 1,
                "predecessor chain exceeds the vertex count"
            );
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
