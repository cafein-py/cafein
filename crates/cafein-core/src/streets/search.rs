//! Walking searches: access stops, transfers, point linking, and
//! the walk matrix.

use super::*;

/// A predecessor slot with no entry.
pub(super) const NO_PREVIOUS: (u32, u32) = (u32::MAX, u32::MAX);

/// Reusable per-thread Dijkstra state, dense over the vertices: distances
/// live in a flat array indexed by vertex (infinity when unreached), with
/// a touched list so a search resets only the slots it wrote. The arrays
/// grow once per thread to the network's vertex count; the per-search
/// cost scales with the vertices reached, never with the network.
#[derive(Default)]
pub(super) struct SearchState {
    /// Best known distance in meters per vertex; infinity when unreached.
    pub(super) distances: Vec<f64>,
    /// Predecessor `(vertex, edge)` per vertex; [`NO_PREVIOUS`] when unset
    /// or a seed. Sized lazily — only path reconstruction fills it.
    pub(super) previous: Vec<(u32, u32)>,
    /// The vertices the current search has written.
    pub(super) touched: Vec<u32>,
    /// Pending `(distance bits, vertex)` entries. Non-negative floats
    /// order like their IEEE bit patterns.
    pub(super) heap: BinaryHeap<Reverse<(u64, u32)>>,
}

impl SearchState {
    /// Grows the distance array to cover `vertices` and resets the
    /// previous search's slots; call before seeding.
    pub(super) fn prepare(&mut self, vertices: usize) {
        if self.distances.len() < vertices {
            self.distances.resize(vertices, f64::INFINITY);
        }
        for &vertex in &self.touched {
            self.distances[vertex as usize] = f64::INFINITY;
            // The predecessor array grows only for path-reconstructing
            // searches, so a touched vertex of a larger network can sit
            // past its end; only in-range slots can hold stale entries.
            if let Some(slot) = self.previous.get_mut(vertex as usize) {
                *slot = NO_PREVIOUS;
            }
        }
        self.touched.clear();
        self.heap.clear();
    }

    /// Grows the predecessor array too; only the path-reconstructing
    /// search pays for it.
    pub(super) fn prepare_with_previous(&mut self, vertices: usize) {
        self.prepare(vertices);
        if self.previous.len() < vertices {
            self.previous.resize(vertices, NO_PREVIOUS);
        }
    }

    /// Writes a vertex's distance, recording the first touch.
    #[inline]
    pub(super) fn set_distance(&mut self, vertex: u32, distance: f64) {
        let slot = &mut self.distances[vertex as usize];
        if slot.is_infinite() {
            self.touched.push(vertex);
        }
        *slot = distance;
    }

    #[inline]
    pub(super) fn distance(&self, vertex: u32) -> f64 {
        self.distances[vertex as usize]
    }
}

/// Reached vertices and their walk distances, however the search produced
/// them: the dense per-thread [`SearchState`] or the contraction
/// hierarchy's hash-mapped scratch. Joins consume either through this.
pub(super) trait Reached {
    fn walk(&self, vertex: u32) -> f64;
    fn for_each_reached(&self, apply: impl FnMut(u32, f64));
}

impl Reached for SearchState {
    #[inline]
    fn walk(&self, vertex: u32) -> f64 {
        self.distances[vertex as usize]
    }

    fn for_each_reached(&self, mut apply: impl FnMut(u32, f64)) {
        for &vertex in &self.touched {
            apply(vertex, self.distances[vertex as usize]);
        }
    }
}

impl Reached for HashMap<u32, f64> {
    #[inline]
    fn walk(&self, vertex: u32) -> f64 {
        self.get(&vertex).copied().unwrap_or(f64::INFINITY)
    }

    fn for_each_reached(&self, mut apply: impl FnMut(u32, f64)) {
        for (&vertex, &distance) in self {
            apply(vertex, distance);
        }
    }
}

thread_local! {
    pub(super) static SEARCH_STATE: std::cell::RefCell<SearchState> =
        std::cell::RefCell::new(SearchState::default());
    /// Per-thread scratch for the contraction-hierarchy one-to-many query,
    /// reused across a matrix's per-origin searches like `SEARCH_STATE`.
    static CH_SCRATCH: std::cell::RefCell<crate::ch::ChScratch> =
        std::cell::RefCell::new(crate::ch::ChScratch::default());
}

impl StreetNetwork {
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
    pub(super) fn reachable_from_snaps(
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
                self.link_join(snaps, cutoff, walking_speed, &**state)
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
    pub(super) fn link_join(
        &self,
        snaps: &[Snap],
        cutoff: f64,
        walking_speed: f64,
        reached: &impl Reached,
    ) -> Vec<WalkedStop> {
        let distance = |vertex: u32| reached.walk(vertex);
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
        reached.for_each_reached(|vertex, _| {
            candidates.extend(self.links_at(vertex));
        });
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
            .par_iter()
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
    pub(super) fn links_at(&self, vertex: u32) -> impl Iterator<Item = u32> + '_ {
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

    /// Links one or more coordinate sets by searching **from the stops** (an R5
    /// `LinkedPointSet`): batch-snap the points, then run one bounded search per
    /// stop and join it to the points, so a matrix pays `O(stops)` searches
    /// instead of one per point — and the single stop-search pass serves every
    /// set (a matrix's access and egress) at once. Returns, per set, per point,
    /// the walkable stops (or `None` where a point does not snap): the same stop
    /// sets and rounded seconds as `link_many` on each set, metres to
    /// floating-point round-off (the two search directions sum a path's edges in
    /// opposite order).
    ///
    /// Correct only when the walking graph is symmetric, so a search *from* a
    /// stop yields the distances a search *to* it would; an asymmetric graph is
    /// ineligible and every set falls back to the per-point `link_many`. The
    /// searches are bounded by the query cutoff, so there is no persistent table.
    pub fn link_pointsets(
        &self,
        sets: &[&[(f64, f64)]],
        walking_speed: f64,
        max_seconds: f64,
        max_snap_distance: f64,
    ) -> Vec<Vec<Option<Vec<WalkedStop>>>> {
        if !walking_speed.is_finite()
            || walking_speed <= 0.0
            || !max_seconds.is_finite()
            || max_seconds < 0.0
        {
            return sets.iter().map(|set| vec![None; set.len()]).collect();
        }
        // An asymmetric walking graph cannot be linked from the stop side.
        if !self.is_symmetric() {
            return sets
                .iter()
                .map(|set| self.link_many(set, walking_speed, max_seconds, max_snap_distance))
                .collect();
        }

        // Group each stop's links so its search leaves from all of them, exactly
        // as `access_stops` reaches a stop through any of its links. Searching
        // from every stop is only worth it when there are more points to link
        // than stops to search; below that the per-point search is cheaper.
        let mut by_stop: HashMap<StopIdx, Vec<Snap>> = HashMap::new();
        for link in &self.links {
            by_stop.entry(link.stop).or_default().push(Snap {
                edge: link.edge,
                fraction: link.fraction,
                connector: link.connector,
            });
        }
        let total_points: usize = sets.iter().map(|set| set.len()).sum();
        if total_points < by_stop.len() {
            return sets
                .iter()
                .map(|set| self.link_many(set, walking_speed, max_seconds, max_snap_distance))
                .collect();
        }
        let cutoff = max_seconds * walking_speed;

        // Flatten the sets to one global point list, remembering the set bounds.
        let mut set_offsets = Vec::with_capacity(sets.len() + 1);
        set_offsets.push(0usize);
        let mut points: Vec<(f64, f64)> = Vec::new();
        for set in sets {
            points.extend_from_slice(set);
            set_offsets.push(points.len());
        }

        // Batch snap, then index each snapped point by its edge endpoints (the
        // on-edge distance to each) and by its edge (for same-edge walks).
        let snaps: Vec<Option<Snap>> = points
            .par_iter()
            .map(|&(latitude, longitude)| self.snap(latitude, longitude, max_snap_distance))
            .collect();
        let mut by_vertex: Vec<Vec<(u32, f64)>> = vec![Vec::new(); self.vertex_count() as usize];
        let mut by_edge: HashMap<u32, Vec<(u32, f64, f64)>> = HashMap::new();
        for (point, snap) in snaps.iter().enumerate() {
            let Some(snap) = snap else { continue };
            let (from, to) = self.edge_endpoints(snap.edge);
            let length = self.arrays.lengths()[snap.edge as usize];
            by_vertex[from as usize].push((point as u32, snap.connector + snap.fraction * length));
            by_vertex[to as usize].push((
                point as u32,
                snap.connector + (1.0 - snap.fraction) * length,
            ));
            by_edge.entry(snap.edge).or_default().push((
                point as u32,
                snap.fraction,
                snap.connector,
            ));
        }

        let mut stops: Vec<(StopIdx, Vec<Snap>)> = by_stop.into_iter().collect();
        stops.sort_unstable_by_key(|(stop, _)| *stop);

        // One bounded search per stop (parallel), joined to the points at its
        // reached vertices; each stop emits `(point, stop, metres)` for the
        // points it reaches within the cutoff.
        let reached: Vec<(u32, StopIdx, f64)> = stops
            .into_par_iter()
            .flat_map_iter(|(stop, snaps)| {
                let mut seeds: Vec<(u32, f64)> = Vec::with_capacity(snaps.len() * 2);
                for snap in &snaps {
                    let (from, to) = self.edge_endpoints(snap.edge);
                    let length = self.arrays.lengths()[snap.edge as usize];
                    seeds.push((from, snap.connector + snap.fraction * length));
                    seeds.push((to, snap.connector + (1.0 - snap.fraction) * length));
                }
                let mut best: HashMap<u32, f64> = HashMap::new();
                // Always the graph search here: the contraction hierarchy's
                // one-to-many buckets cover the stop-link endpoint vertices —
                // the join targets when searching *from a point* — while this
                // search joins at the points' own snap endpoints, which the
                // buckets do not cover.
                SEARCH_STATE.with(|cell| {
                    let state = &mut cell.borrow_mut();
                    self.bounded_dijkstra(&seeds, cutoff, state);
                    self.pointset_join(&**state, &snaps, &by_vertex, &by_edge, cutoff, &mut best);
                });
                best.into_iter()
                    .map(move |(point, metres)| (point, stop, metres))
                    .collect::<Vec<_>>()
            })
            .collect();

        // Regroup into per-point walk lists (one `WalkedStop` per reaching stop),
        // sorted by stop; a snapped point that reaches nothing keeps an empty
        // list, an unsnapped point stays `None`.
        let mut walks: Vec<Vec<WalkedStop>> = vec![Vec::new(); points.len()];
        for (point, stop, metres) in reached {
            walks[point as usize].push(WalkedStop {
                stop,
                seconds: seconds(metres / walking_speed),
                meters: metres,
            });
        }
        let mut result = Vec::with_capacity(sets.len());
        for set in 0..sets.len() {
            let mut linked = Vec::with_capacity(sets[set].len());
            for local in 0..sets[set].len() {
                let point = set_offsets[set] + local;
                if snaps[point].is_none() {
                    linked.push(None);
                } else {
                    let mut walked = std::mem::take(&mut walks[point]);
                    walked.sort_unstable_by_key(|walk| walk.stop);
                    linked.push(Some(walked));
                }
            }
            result.push(linked);
        }
        result
    }

    /// One stop's half of the pointset join: offers `walk(stop → point)`
    /// through every reached vertex carrying points, plus the direct
    /// on-edge walk to points sharing one of the stop's link edges,
    /// keeping the minimum per point. Shared by the `bounded_dijkstra`
    /// and contraction-hierarchy paths of
    /// [`link_pointsets`](Self::link_pointsets).
    pub(super) fn pointset_join(
        &self,
        reached: &impl Reached,
        snaps: &[Snap],
        by_vertex: &[Vec<(u32, f64)>],
        by_edge: &HashMap<u32, Vec<(u32, f64, f64)>>,
        cutoff: f64,
        best: &mut HashMap<u32, f64>,
    ) {
        reached.for_each_reached(|vertex, stop_distance| {
            for &(point, point_distance) in &by_vertex[vertex as usize] {
                let metres = point_distance + stop_distance;
                if metres <= cutoff + 1e-9 {
                    best.entry(point)
                        .and_modify(|best| *best = best.min(metres))
                        .or_insert(metres);
                }
            }
        });
        // A point on one of the stop's own link edges can walk it
        // directly, never reaching an endpoint.
        for snap in snaps {
            let Some(on_edge) = by_edge.get(&snap.edge) else {
                continue;
            };
            let length = self.arrays.lengths()[snap.edge as usize];
            for &(point, fraction, connector) in on_edge {
                let metres = connector + (snap.fraction - fraction).abs() * length + snap.connector;
                if metres <= cutoff + 1e-9 {
                    best.entry(point)
                        .and_modify(|best| *best = best.min(metres))
                        .or_insert(metres);
                }
            }
        }
    }

    /// Whether the walking adjacency is symmetric — every directed edge has its
    /// reverse with equal metres. Computed once, then cached; walking is
    /// undirected in the OSM extraction, so it holds.
    pub(super) fn is_symmetric(&self) -> bool {
        *self.symmetric.get_or_init(|| {
            let offsets = self.arrays.adjacency_offsets();
            let targets = self.arrays.adj_targets();
            let meters = self.arrays.adj_meters();
            let mut directed: std::collections::HashSet<(u32, u32, u64)> =
                std::collections::HashSet::with_capacity(targets.len());
            for from in 0..offsets.len().saturating_sub(1) {
                for slot in offsets[from] as usize..offsets[from + 1] as usize {
                    directed.insert((from as u32, targets[slot], meters[slot].to_bits()));
                }
            }
            (0..offsets.len().saturating_sub(1)).all(|from| {
                (offsets[from] as usize..offsets[from + 1] as usize).all(|slot| {
                    directed.contains(&(targets[slot], from as u32, meters[slot].to_bits()))
                })
            })
        })
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
}
