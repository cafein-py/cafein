//! Contraction hierarchy over the undirected walking graph — the acceleration
//! substrate for the `O(stops)` access/egress/stop-transfer searches (see
//! `plans/core-ch-plan.md`). Preprocessing (importance ordering + contraction
//! with a core-only witness search) and a bidirectional point-to-point query
//! with shortcut unpacking (CH-1), plus the **bucket one-to-many** —
//! precomputed per-target [`Buckets`] and a source-side [`one_to_many`] query —
//! for the access/egress workload (CH-2). Installed on a
//! [`StreetNetwork`](crate::streets) via `install_hierarchy` (CH-3), where the
//! stop-destination walking searches run it instead of a graph sweep. Validated
//! against a plain Dijkstra.
//!
//! [`one_to_many`]: ContractionHierarchy::one_to_many
//!
//! The graph is symmetric (walking), so the hierarchy stores a single **upward
//! CSR** — per vertex, the edges to its higher-rank neighbours (originals and
//! shortcuts). Both the forward search (from `s`) and the backward search (from
//! `t`) relax this same upward adjacency, each climbing toward higher ranks.

use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

/// Sentinel for an original edge in `up_middle` (i.e. not a shortcut).
pub const ORIGINAL: u32 = u32::MAX;

/// A contraction hierarchy: per-vertex `rank` (contraction order — lower is
/// contracted earlier / less important) and an upward CSR of edges to
/// higher-rank neighbours. `up_middle[e]` is [`ORIGINAL`] for an original edge,
/// else the middle vertex a shortcut bridges (for unpacking).
#[derive(Debug)]
pub struct ContractionHierarchy {
    rank: Vec<u32>,
    up_offsets: Vec<u32>,
    up_targets: Vec<u32>,
    up_meters: Vec<f64>,
    up_middle: Vec<u32>,
}

/// Precomputed one-to-many buckets over a fixed set of target vertices (built by
/// [`ContractionHierarchy::buckets`]). `entries[v]` lists `(target, distance)`
/// for every target whose bounded upward search settled vertex `v` — the
/// backward half of a bucket-CH query, shared across all sources.
#[derive(Debug)]
pub struct Buckets {
    entries: Vec<Vec<(u32, f64)>>,
    /// The cutoff the buckets were built for. A query cutoff must not exceed it
    /// (see [`ContractionHierarchy::one_to_many`]): the bucket side already
    /// pruned meeting vertices past this bound, so a larger query would silently
    /// omit reachable targets.
    cutoff: f64,
}

/// A vertex's live adjacency during contraction: neighbour → (metres, middle),
/// where `middle` is [`ORIGINAL`] for an original edge, else the shortcut's
/// middle vertex. Parallel edges collapse to the shortest (ties keep the
/// original / lowest-middle for determinism).
type Adjacency = Vec<Vec<Neighbour>>;

#[derive(Clone, Copy)]
struct Neighbour {
    vertex: u32,
    meters: f64,
    middle: u32,
}

impl ContractionHierarchy {
    /// Builds a hierarchy from an undirected graph given as a CSR. `offsets`
    /// has `vertex_count + 1` entries; `targets`/`meters` list each undirected
    /// edge from both endpoints (as [`StreetNetwork`](crate::streets) stores
    /// it). Metres must be finite and non-negative.
    pub fn build(vertex_count: u32, offsets: &[u32], targets: &[u32], meters: &[f64]) -> Self {
        let n = vertex_count as usize;
        let mut adjacency: Adjacency = vec![Vec::new(); n];
        for vertex in 0..n {
            let start = offsets[vertex] as usize;
            let end = offsets[vertex + 1] as usize;
            for slot in start..end {
                insert_edge(
                    &mut adjacency[vertex],
                    targets[slot],
                    meters[slot],
                    ORIGINAL,
                );
            }
        }

        let mut contracted = vec![false; n];
        let mut contracted_neighbours = vec![0u32; n];
        let mut rank = vec![0u32; n];
        // Collected at contraction time: a contracted vertex's live edges are
        // exactly its upward edges (every remaining neighbour outranks it).
        let mut up: Vec<Vec<Neighbour>> = vec![Vec::new(); n];

        // Lazy priority queue on importance (edge difference + contracted
        // neighbours). Ties break by vertex id so the order is deterministic.
        let mut queue: BinaryHeap<Reverse<(i64, u32)>> = BinaryHeap::with_capacity(n);
        for vertex in 0..n {
            let priority = importance(
                vertex as u32,
                &adjacency,
                &contracted,
                &contracted_neighbours,
            );
            queue.push(Reverse((priority, vertex as u32)));
        }

        let mut next_rank = 0u32;
        while let Some(Reverse((priority, vertex))) = queue.pop() {
            if contracted[vertex as usize] {
                continue;
            }
            // Lazy update: recompute importance; if it grew past the new front,
            // reinsert instead of contracting now.
            let current = importance(vertex, &adjacency, &contracted, &contracted_neighbours);
            if current > priority {
                if let Some(&Reverse((front, _))) = queue.peek() {
                    if current > front {
                        queue.push(Reverse((current, vertex)));
                        continue;
                    }
                }
            }

            // Contract `vertex`: its live edges become its upward edges.
            up[vertex as usize] = adjacency[vertex as usize].clone();
            rank[vertex as usize] = next_rank;
            next_rank += 1;
            contracted[vertex as usize] = true;

            let shortcuts = shortcuts_for(vertex, &adjacency, &contracted);
            for shortcut in shortcuts {
                insert_edge(
                    &mut adjacency[shortcut.u as usize],
                    shortcut.w,
                    shortcut.meters,
                    vertex,
                );
                insert_edge(
                    &mut adjacency[shortcut.w as usize],
                    shortcut.u,
                    shortcut.meters,
                    vertex,
                );
            }
            for neighbour in &adjacency[vertex as usize] {
                if !contracted[neighbour.vertex as usize] {
                    contracted_neighbours[neighbour.vertex as usize] += 1;
                    // Importance is recomputed **lazily** when the neighbour is
                    // popped, not eagerly here: an eager recompute runs a witness
                    // search per neighbour per contraction (~O(E) batches), the
                    // dominant preprocessing cost. A stale-high queue key only
                    // delays a vertex — the pop-time recompute reinserts it — so
                    // the hierarchy stays correct, just built far faster.
                }
            }
        }

        // Assemble the upward CSR, keeping only edges to strictly higher-rank
        // neighbours (a contracted vertex's live edges all point upward, but a
        // shortcut recorded on both endpoints must land on the lower-rank one).
        let mut up_offsets = vec![0u32; n + 1];
        for vertex in 0..n {
            let mut kept = 0u32;
            for neighbour in &up[vertex] {
                if rank[neighbour.vertex as usize] > rank[vertex] {
                    kept += 1;
                }
            }
            up_offsets[vertex + 1] = up_offsets[vertex] + kept;
        }
        let total = *up_offsets.last().unwrap() as usize;
        let mut up_targets = vec![0u32; total];
        let mut up_meters = vec![0.0f64; total];
        let mut up_middle = vec![ORIGINAL; total];
        for vertex in 0..n {
            let mut edges: Vec<Neighbour> = up[vertex]
                .iter()
                .copied()
                .filter(|neighbour| rank[neighbour.vertex as usize] > rank[vertex])
                .collect();
            edges.sort_by_key(|neighbour| neighbour.vertex);
            let start = up_offsets[vertex] as usize;
            for (slot, neighbour) in (start..).zip(edges) {
                up_targets[slot] = neighbour.vertex;
                up_meters[slot] = neighbour.meters;
                up_middle[slot] = neighbour.middle;
            }
        }

        ContractionHierarchy {
            rank,
            up_offsets,
            up_targets,
            up_meters,
            up_middle,
        }
    }

    /// The number of vertices.
    pub fn vertex_count(&self) -> u32 {
        self.rank.len() as u32
    }

    /// The number of shortcut edges in the upward CSR (contraction added). A
    /// witness-suppressed contraction adds none; a forced middle adds some.
    pub fn shortcut_count(&self) -> usize {
        self.up_middle.iter().filter(|&&m| m != ORIGINAL).count()
    }

    /// The shortest-walk distance in metres between `source` and `target`, or
    /// `None` if unreachable. Exact — equal to a plain Dijkstra on the original
    /// graph (within floating-point accumulation order).
    pub fn distance(&self, source: u32, target: u32) -> Option<f64> {
        self.query(source, target).map(|(distance, _)| distance)
    }

    /// The shortest walk expanded to the original **vertex** sequence
    /// `[source, .., target]`, or `None` if unreachable — a correctness aid (it
    /// proves shortcuts unpack to a valid original path). It carries only
    /// vertices: parallel original edges between two vertices are collapsed to
    /// the shortest at build time, which keeps distances exact but loses which
    /// street edge a hop used, so **geometry/`walk_path` reconstruction stays on
    /// the original-graph search** (it is per-leg, not the `O(stops)` bottleneck
    /// CH targets); edge-level unpacking would be added only if CH ever serves
    /// geometry.
    pub fn path(&self, source: u32, target: u32) -> Option<Vec<u32>> {
        let (_, meeting) = self.query(source, target)?;
        let forward = self.up_tree_path(source, &self.settle(source), meeting);
        let backward = self.up_tree_path(target, &self.settle(target), meeting);
        // `forward` is source..=meeting; `backward` is target..=meeting; join
        // with the reversed backward tail (meeting..=target), dropping the
        // duplicated meeting vertex.
        let mut path = forward;
        for &vertex in backward.iter().rev().skip(1) {
            path.push(vertex);
        }
        Some(self.unpack(&path))
    }

    /// Bidirectional up-search meeting in the middle: returns the shortest
    /// distance and the meeting vertex.
    fn query(&self, source: u32, target: u32) -> Option<(f64, u32)> {
        let forward = self.settle(source);
        let backward = self.settle(target);
        let mut best: Option<(f64, u32)> = None;
        for (vertex, &distance) in forward.iter().enumerate() {
            if distance.is_finite() && backward[vertex].is_finite() {
                let total = distance + backward[vertex];
                if best.is_none_or(|(current, _)| total < current) {
                    best = Some((total, vertex as u32));
                }
            }
        }
        best
    }

    /// One-sided upward Dijkstra from `origin`, settling every vertex reachable
    /// by climbing the upward CSR. Returns per-vertex distances (`INFINITY` for
    /// unsettled).
    fn settle(&self, origin: u32) -> Vec<f64> {
        let mut distances = vec![f64::INFINITY; self.rank.len()];
        let mut heap: BinaryHeap<Reverse<(u64, u32)>> = BinaryHeap::new();
        distances[origin as usize] = 0.0;
        heap.push(Reverse((0.0f64.to_bits(), origin)));
        while let Some(Reverse((bits, vertex))) = heap.pop() {
            let distance = f64::from_bits(bits);
            if distance > distances[vertex as usize] {
                continue;
            }
            let start = self.up_offsets[vertex as usize] as usize;
            let end = self.up_offsets[vertex as usize + 1] as usize;
            for slot in start..end {
                let target = self.up_targets[slot];
                let next = distance + self.up_meters[slot];
                if next < distances[target as usize] {
                    distances[target as usize] = next;
                    heap.push(Reverse((next.to_bits(), target)));
                }
            }
        }
        distances
    }

    /// A **bounded** upward search from `seeds` (each `(vertex, initial
    /// distance)`): the settled `(vertex, distance)` pairs with
    /// `distance <= cutoff`, climbing the upward CSR. The sparse, seed-and-cutoff
    /// form of [`settle`](Self::settle) that builds and queries buckets.
    fn up_search(&self, seeds: &[(u32, f64)], cutoff: f64) -> Vec<(u32, f64)> {
        let mut distances: HashMap<u32, f64> = HashMap::new();
        let mut heap: BinaryHeap<Reverse<(u64, u32)>> = BinaryHeap::new();
        for &(vertex, distance) in seeds {
            if distance <= cutoff + 1e-9
                && distance < *distances.get(&vertex).unwrap_or(&f64::INFINITY)
            {
                distances.insert(vertex, distance);
                heap.push(Reverse((distance.to_bits(), vertex)));
            }
        }
        while let Some(Reverse((bits, vertex))) = heap.pop() {
            let distance = f64::from_bits(bits);
            if distance > *distances.get(&vertex).unwrap_or(&f64::INFINITY) {
                continue;
            }
            let start = self.up_offsets[vertex as usize] as usize;
            let end = self.up_offsets[vertex as usize + 1] as usize;
            for slot in start..end {
                let target = self.up_targets[slot];
                let next = distance + self.up_meters[slot];
                if next <= cutoff + 1e-9 && next < *distances.get(&target).unwrap_or(&f64::INFINITY)
                {
                    distances.insert(target, next);
                    heap.push(Reverse((next.to_bits(), target)));
                }
            }
        }
        distances.into_iter().collect()
    }

    /// Builds one-to-many **buckets** for a fixed set of `targets` (vertex ids),
    /// each bounded by `cutoff`. The per-target upward searches are independent
    /// and run in parallel; the results scatter into `entries[v]` =
    /// `(target, distance(target, v))`. Built once and reused across sources by
    /// [`one_to_many`](Self::one_to_many).
    pub fn buckets(&self, targets: &[u32], cutoff: f64) -> Buckets {
        let per_target: Vec<(u32, Vec<(u32, f64)>)> = targets
            .par_iter()
            .map(|&target| (target, self.up_search(&[(target, 0.0)], cutoff)))
            .collect();
        let mut entries: Vec<Vec<(u32, f64)>> = vec![Vec::new(); self.rank.len()];
        for (target, settled) in per_target {
            for (vertex, distance) in settled {
                entries[vertex as usize].push((target, distance));
            }
        }
        Buckets { entries, cutoff }
    }

    /// One-to-many shortest distances from `seeds` (a snapped source) to every
    /// bucketed target within `cutoff`. For each vertex the source's bounded
    /// upward search settles, the buckets give the targets reachable through it,
    /// and each target keeps
    /// `min(distance(source, vertex) + distance(target, vertex))` — the bucket-CH
    /// meeting. Bounding both halves by `cutoff` never drops a within-cutoff
    /// target, since the meeting vertex's distance from each side is a prefix of
    /// the (`<= cutoff`) total. Returns `(target, distance)` pairs.
    ///
    /// # Panics
    ///
    /// The `cutoff` must not exceed the buckets' build cutoff. A larger query
    /// cutoff is a contract violation: the bucket side already pruned meeting
    /// vertices past its build cutoff, so reachable targets would be silently
    /// omitted. Build the buckets for at least the largest query cutoff.
    pub fn one_to_many(
        &self,
        buckets: &Buckets,
        seeds: &[(u32, f64)],
        cutoff: f64,
    ) -> Vec<(u32, f64)> {
        assert!(
            cutoff <= buckets.cutoff + 1e-9,
            "one_to_many cutoff {cutoff} exceeds the bucket build cutoff {}; \
             rebuild the buckets for at least this cutoff",
            buckets.cutoff
        );
        let mut best: HashMap<u32, f64> = HashMap::new();
        for (vertex, source_distance) in self.up_search(seeds, cutoff) {
            for &(target, target_distance) in &buckets.entries[vertex as usize] {
                let candidate = source_distance + target_distance;
                if candidate <= cutoff + 1e-9 {
                    best.entry(target)
                        .and_modify(|current| *current = current.min(candidate))
                        .or_insert(candidate);
                }
            }
        }
        best.into_iter().collect()
    }

    /// The upward-tree vertex path `origin..=peak` following predecessors of the
    /// settled distances (recomputed by a min-edge walk down from `peak`).
    fn up_tree_path(&self, origin: u32, distances: &[f64], peak: u32) -> Vec<u32> {
        let mut path = vec![peak];
        let mut vertex = peak;
        while vertex != origin {
            let target_distance = distances[vertex as usize];
            let mut predecessor = None;
            // Find a neighbour `p` (via the upward CSR, which is `p -> vertex`)
            // whose settled distance plus the edge equals `vertex`'s distance.
            for p in 0..self.rank.len() as u32 {
                if !distances[p as usize].is_finite() {
                    continue;
                }
                let start = self.up_offsets[p as usize] as usize;
                let end = self.up_offsets[p as usize + 1] as usize;
                for slot in start..end {
                    if self.up_targets[slot] == vertex
                        && (distances[p as usize] + self.up_meters[slot] - target_distance).abs()
                            < 1e-6
                    {
                        predecessor = Some(p);
                        break;
                    }
                }
                if predecessor.is_some() {
                    break;
                }
            }
            let p = predecessor.expect("up-tree path has a predecessor");
            path.push(p);
            vertex = p;
        }
        path.reverse();
        path
    }

    /// Expands any shortcut hops in a vertex path into original edges.
    fn unpack(&self, path: &[u32]) -> Vec<u32> {
        let mut out = vec![path[0]];
        for window in path.windows(2) {
            self.unpack_edge(window[0], window[1], &mut out);
        }
        out
    }

    fn unpack_edge(&self, from: u32, to: u32, out: &mut Vec<u32>) {
        let middle = self.edge_middle(from, to);
        match middle {
            Some(ORIGINAL) | None => out.push(to),
            Some(mid) => {
                self.unpack_edge(from, mid, out);
                self.unpack_edge(mid, to, out);
            }
        }
    }

    /// The `up_middle` of the edge between `a` and `b` (either direction), if
    /// present in the upward CSR.
    fn edge_middle(&self, a: u32, b: u32) -> Option<u32> {
        for (from, to) in [(a, b), (b, a)] {
            let start = self.up_offsets[from as usize] as usize;
            let end = self.up_offsets[from as usize + 1] as usize;
            for slot in start..end {
                if self.up_targets[slot] == to {
                    return Some(self.up_middle[slot]);
                }
            }
        }
        None
    }
}

/// A shortcut the contraction of `middle` requires between `u` and `w`.
struct Shortcut {
    u: u32,
    w: u32,
    meters: f64,
}

/// The shortcuts contracting `vertex` requires: for each pair of uncontracted
/// neighbours `(u, w)`, a shortcut unless a witness path `u..w` over the current
/// core (excluding `vertex`) is no longer than `d(u,vertex) + d(vertex,w)`.
fn shortcuts_for(vertex: u32, adjacency: &Adjacency, contracted: &[bool]) -> Vec<Shortcut> {
    let neighbours: Vec<Neighbour> = adjacency[vertex as usize]
        .iter()
        .copied()
        .filter(|neighbour| !contracted[neighbour.vertex as usize])
        .collect();
    let mut shortcuts = Vec::new();
    for (i, u) in neighbours.iter().enumerate() {
        for w in neighbours.iter().skip(i + 1) {
            let candidate = u.meters + w.meters;
            if !has_witness(u.vertex, w.vertex, vertex, candidate, adjacency, contracted) {
                shortcuts.push(Shortcut {
                    u: u.vertex,
                    w: w.vertex,
                    meters: candidate,
                });
            }
        }
    }
    shortcuts
}

/// Caps the witness search at this many settled vertices. Past it, the search
/// gives up and a (possibly superfluous) shortcut is added — a longer local
/// search finds a few more witnesses but is much slower, and a superfluous
/// shortcut never corrupts distances (its weight is never below the true path,
/// which the query still finds), so this only trades a slightly denser hierarchy
/// for a far faster contraction.
const WITNESS_SETTLE_LIMIT: usize = 200;

/// Whether a path `source..target` no longer than `limit` exists over the
/// current uncontracted core with `excluded` (the vertex being contracted) and
/// all already-contracted vertices removed — a bounded, settle-limited Dijkstra.
fn has_witness(
    source: u32,
    target: u32,
    excluded: u32,
    limit: f64,
    adjacency: &Adjacency,
    contracted: &[bool],
) -> bool {
    let mut distances: std::collections::HashMap<u32, f64> = std::collections::HashMap::new();
    let mut heap: BinaryHeap<Reverse<(u64, u32)>> = BinaryHeap::new();
    distances.insert(source, 0.0);
    heap.push(Reverse((0.0f64.to_bits(), source)));
    let mut settled = 0usize;
    while let Some(Reverse((bits, vertex))) = heap.pop() {
        let distance = f64::from_bits(bits);
        if distance > limit + 1e-9 {
            break;
        }
        if distance > *distances.get(&vertex).unwrap_or(&f64::INFINITY) {
            continue;
        }
        if vertex == target {
            return distance <= limit + 1e-9;
        }
        settled += 1;
        if settled >= WITNESS_SETTLE_LIMIT {
            return false; // give up — add a shortcut rather than search further
        }
        for neighbour in &adjacency[vertex as usize] {
            let next_vertex = neighbour.vertex;
            if next_vertex == excluded || contracted[next_vertex as usize] {
                continue;
            }
            let next = distance + neighbour.meters;
            if next <= limit + 1e-9 && next < *distances.get(&next_vertex).unwrap_or(&f64::INFINITY)
            {
                distances.insert(next_vertex, next);
                heap.push(Reverse((next.to_bits(), next_vertex)));
            }
        }
    }
    false
}

/// A vertex's importance: edge difference (shortcuts needed minus live degree)
/// plus the number of already-contracted neighbours (spreads contraction out).
/// Lower contracts first.
fn importance(
    vertex: u32,
    adjacency: &Adjacency,
    contracted: &[bool],
    contracted_neighbours: &[u32],
) -> i64 {
    let degree = adjacency[vertex as usize]
        .iter()
        .filter(|neighbour| !contracted[neighbour.vertex as usize])
        .count() as i64;
    let shortcuts = shortcuts_for(vertex, adjacency, contracted).len() as i64;
    (shortcuts - degree) + contracted_neighbours[vertex as usize] as i64
}

/// Inserts (or relaxes) an undirected-half edge, collapsing parallels to the
/// shortest. Ties keep the lower `middle` (an original edge, `ORIGINAL`, sorts
/// last, so a real shortcut middle wins a tie — deterministic either way).
fn insert_edge(neighbours: &mut Vec<Neighbour>, vertex: u32, meters: f64, middle: u32) {
    for existing in neighbours.iter_mut() {
        if existing.vertex == vertex {
            if meters < existing.meters || (meters == existing.meters && middle < existing.middle) {
                existing.meters = meters;
                existing.middle = middle;
            }
            return;
        }
    }
    neighbours.push(Neighbour {
        vertex,
        meters,
        middle,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    /// A plain Dijkstra on the original CSR — the reference the CH must match.
    fn reference(
        vertex_count: u32,
        offsets: &[u32],
        targets: &[u32],
        meters: &[f64],
        source: u32,
    ) -> Vec<f64> {
        let mut distances = vec![f64::INFINITY; vertex_count as usize];
        let mut heap: BinaryHeap<Reverse<(u64, u32)>> = BinaryHeap::new();
        distances[source as usize] = 0.0;
        heap.push(Reverse((0.0f64.to_bits(), source)));
        while let Some(Reverse((bits, vertex))) = heap.pop() {
            let distance = f64::from_bits(bits);
            if distance > distances[vertex as usize] {
                continue;
            }
            let start = offsets[vertex as usize] as usize;
            let end = offsets[vertex as usize + 1] as usize;
            for slot in start..end {
                let target = targets[slot];
                let next = distance + meters[slot];
                if next < distances[target as usize] {
                    distances[target as usize] = next;
                    heap.push(Reverse((next.to_bits(), target)));
                }
            }
        }
        distances
    }

    /// Builds an undirected CSR from an edge list `(a, b, meters)`.
    fn csr(vertex_count: u32, edges: &[(u32, u32, f64)]) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
        let n = vertex_count as usize;
        let mut degree = vec![0u32; n + 1];
        for &(a, b, _) in edges {
            degree[a as usize + 1] += 1;
            degree[b as usize + 1] += 1;
        }
        for vertex in 0..n {
            degree[vertex + 1] += degree[vertex];
        }
        let offsets = degree.clone();
        let total = *offsets.last().unwrap() as usize;
        let mut targets = vec![0u32; total];
        let mut meters = vec![0.0f64; total];
        let mut cursor = offsets.clone();
        for &(a, b, m) in edges {
            let sa = cursor[a as usize] as usize;
            targets[sa] = b;
            meters[sa] = m;
            cursor[a as usize] += 1;
            let sb = cursor[b as usize] as usize;
            targets[sb] = a;
            meters[sb] = m;
            cursor[b as usize] += 1;
        }
        (offsets, targets, meters)
    }

    fn assert_matches(vertex_count: u32, edges: &[(u32, u32, f64)]) {
        let (offsets, targets, meters) = csr(vertex_count, edges);
        let ch = ContractionHierarchy::build(vertex_count, &offsets, &targets, &meters);
        for source in 0..vertex_count {
            let reference = reference(vertex_count, &offsets, &targets, &meters, source);
            for target in 0..vertex_count {
                let expected = reference[target as usize];
                let got = ch.distance(source, target);
                match (expected.is_finite(), got) {
                    (true, Some(distance)) => assert!(
                        (distance - expected).abs() < 1e-6,
                        "d({source},{target}) = {distance}, expected {expected}"
                    ),
                    (false, None) => {}
                    other => panic!("d({source},{target}): expected {expected:?}, got {other:?}"),
                }
            }
        }
    }

    /// A plain multi-seed bounded Dijkstra on the original CSR — the reference
    /// the bucket-CH one-to-many must match.
    fn reference_seeds(
        vertex_count: u32,
        offsets: &[u32],
        targets: &[u32],
        meters: &[f64],
        seeds: &[(u32, f64)],
    ) -> Vec<f64> {
        let mut distances = vec![f64::INFINITY; vertex_count as usize];
        let mut heap: BinaryHeap<Reverse<(u64, u32)>> = BinaryHeap::new();
        for &(vertex, distance) in seeds {
            if distance < distances[vertex as usize] {
                distances[vertex as usize] = distance;
                heap.push(Reverse((distance.to_bits(), vertex)));
            }
        }
        while let Some(Reverse((bits, vertex))) = heap.pop() {
            let distance = f64::from_bits(bits);
            if distance > distances[vertex as usize] {
                continue;
            }
            let start = offsets[vertex as usize] as usize;
            let end = offsets[vertex as usize + 1] as usize;
            for slot in start..end {
                let target = targets[slot];
                let next = distance + meters[slot];
                if next < distances[target as usize] {
                    distances[target as usize] = next;
                    heap.push(Reverse((next.to_bits(), target)));
                }
            }
        }
        distances
    }

    fn assert_one_to_many(
        vertex_count: u32,
        edges: &[(u32, u32, f64)],
        seeds: &[(u32, f64)],
        cutoff: f64,
    ) {
        let (offsets, targets_csr, meters) = csr(vertex_count, edges);
        let ch = ContractionHierarchy::build(vertex_count, &offsets, &targets_csr, &meters);
        let targets: Vec<u32> = (0..vertex_count).collect();
        let buckets = ch.buckets(&targets, cutoff);
        let reference = reference_seeds(vertex_count, &offsets, &targets_csr, &meters, seeds);
        let got: HashMap<u32, f64> = ch
            .one_to_many(&buckets, seeds, cutoff)
            .into_iter()
            .collect();
        for target in 0..vertex_count {
            let expected = reference[target as usize];
            if expected.is_finite() && expected <= cutoff + 1e-9 {
                let distance = got.get(&target).copied();
                assert!(
                    distance.is_some_and(|distance| (distance - expected).abs() < 1e-6),
                    "one_to_many({seeds:?})[{target}] = {distance:?}, expected {expected}"
                );
            } else {
                assert!(
                    got.get(&target)
                        .is_none_or(|&distance| distance > cutoff + 1e-9),
                    "one_to_many({seeds:?})[{target}] = {:?}, expected none (>{cutoff})",
                    got.get(&target)
                );
            }
        }
    }

    fn grid_edges(side: u32) -> Vec<(u32, u32, f64)> {
        let mut edges = Vec::new();
        for r in 0..side {
            for c in 0..side {
                let v = r * side + c;
                if c + 1 < side {
                    edges.push((v, v + 1, 1.0));
                }
                if r + 1 < side {
                    edges.push((v, v + side, 1.0));
                }
            }
        }
        edges
    }

    fn random_edges(n: u32) -> Vec<(u32, u32, f64)> {
        let mut edges: Vec<(u32, u32, f64)> = Vec::new();
        for v in 0..n - 1 {
            let w = 1.0 + ((v as f64 * 7.0 + 3.0) % 9.0);
            edges.push((v, v + 1, w));
        }
        let mut state = 12345u64;
        for _ in 0..80 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let a = (state >> 33) as u32 % n;
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let b = (state >> 33) as u32 % n;
            if a != b {
                let w = 1.0 + ((state >> 40) % 20) as f64;
                edges.push((a, b, w));
            }
        }
        edges
    }

    #[test]
    fn bucket_one_to_many_matches_dijkstra() {
        let edges = grid_edges(4);
        assert_one_to_many(16, &edges, &[(0, 0.0)], f64::INFINITY); // unbounded
        assert_one_to_many(16, &edges, &[(0, 0.0)], 3.0); // bounded prunes far cells
        assert_one_to_many(16, &edges, &[(0, 2.0), (5, 0.0)], 4.0); // multi-seed with offsets
    }

    #[test]
    fn bucket_one_to_many_on_a_random_graph() {
        let edges = random_edges(40);
        assert_one_to_many(40, &edges, &[(0, 0.0)], 15.0);
        assert_one_to_many(40, &edges, &[(7, 1.0), (20, 0.0)], 12.0);
    }

    #[test]
    fn bucket_one_to_many_allows_a_smaller_query_cutoff() {
        // Buckets built for a generous cutoff answer any query bounded by it.
        let edges = grid_edges(4);
        let (offsets, targets_csr, meters) = csr(16, &edges);
        let ch = ContractionHierarchy::build(16, &offsets, &targets_csr, &meters);
        let targets: Vec<u32> = (0..16).collect();
        let buckets = ch.buckets(&targets, 100.0);
        let reference = reference_seeds(16, &offsets, &targets_csr, &meters, &[(0, 0.0)]);
        let got: HashMap<u32, f64> = ch
            .one_to_many(&buckets, &[(0, 0.0)], 3.0)
            .into_iter()
            .collect();
        for target in 0..16u32 {
            let expected = reference[target as usize];
            if expected <= 3.0 + 1e-9 {
                assert!(got
                    .get(&target)
                    .is_some_and(|&d| (d - expected).abs() < 1e-6));
            } else {
                assert!(got.get(&target).is_none_or(|&d| d > 3.0 + 1e-9));
            }
        }
    }

    #[test]
    #[should_panic(expected = "exceeds the bucket build cutoff")]
    fn bucket_one_to_many_rejects_a_larger_query_cutoff() {
        // Querying past the build cutoff would silently omit reachable targets,
        // so it is a hard contract violation rather than a wrong answer.
        let edges = grid_edges(4);
        let (offsets, targets_csr, meters) = csr(16, &edges);
        let ch = ContractionHierarchy::build(16, &offsets, &targets_csr, &meters);
        let buckets = ch.buckets(&(0..16).collect::<Vec<_>>(), 3.0);
        ch.one_to_many(&buckets, &[(0, 0.0)], 10.0);
    }

    #[test]
    fn contraction_forces_and_unpacks_a_shortcut() {
        // A path 0-1-2 whose middle (1) needs the shortcut 0-2 (no witness).
        // Endpoints 0 and 2 are given extra leaves (higher degree → higher
        // importance → contracted later), so vertex 1 — the lowest id in the
        // minimum-importance group — contracts first, forcing the 0-2 shortcut
        // through 1. Without that topology the endpoints would contract first
        // and no shortcut would ever be created (the query would just climb
        // 0-1-2), so the fixture is what makes the test exercise insertion.
        let edges = [
            (0, 1, 1.0),
            (1, 2, 1.0),
            (0, 3, 1.0),
            (0, 4, 1.0),
            (2, 5, 1.0),
            (2, 6, 1.0),
        ];
        let (offsets, targets, meters) = csr(7, &edges);
        let ch = ContractionHierarchy::build(7, &offsets, &targets, &meters);
        assert!(
            ch.shortcut_count() > 0,
            "the middle vertex should have forced a shortcut"
        );
        // 0 -> 2 rides the shortcut and unpacks back through the middle.
        assert_eq!(ch.path(0, 2), Some(vec![0, 1, 2]));
        assert!((ch.distance(0, 2).unwrap() - 2.0).abs() < 1e-6);
        assert_matches(7, &edges);
    }

    #[test]
    fn witness_suppresses_shortcut() {
        // A triangle: contracting any vertex has a witness (the third edge), so
        // no shortcut is added at all — and distances still match.
        let edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.5)];
        let (offsets, targets, meters) = csr(3, &edges);
        let ch = ContractionHierarchy::build(3, &offsets, &targets, &meters);
        assert_eq!(ch.shortcut_count(), 0, "witnesses suppress every shortcut");
        assert_matches(3, &edges);
    }

    #[test]
    fn grid_matches_dijkstra() {
        assert_matches(16, &grid_edges(4));
    }

    #[test]
    fn disconnected_components() {
        // Two components; cross-component distances are unreachable.
        assert_matches(4, &[(0, 1, 2.0), (2, 3, 3.0)]);
    }

    #[test]
    fn weighted_random_graph_matches() {
        assert_matches(40, &random_edges(40));
    }

    #[test]
    fn unpacked_path_length_equals_distance() {
        let edges = grid_edges(4);
        let (offsets, targets, meters) = csr(16, &edges);
        let ch = ContractionHierarchy::build(16, &offsets, &targets, &meters);
        let path = ch.path(0, 15).expect("reachable");
        assert_eq!(path.first(), Some(&0));
        assert_eq!(path.last(), Some(&15));
        // The path is over original edges; its length equals the CH distance.
        let mut length = 0.0;
        for window in path.windows(2) {
            let start = offsets[window[0] as usize] as usize;
            let end = offsets[window[0] as usize + 1] as usize;
            let mut edge = None;
            for slot in start..end {
                if targets[slot] == window[1] {
                    edge = Some(meters[slot]);
                }
            }
            length += edge.expect("consecutive path vertices are adjacent");
        }
        assert!((length - ch.distance(0, 15).unwrap()).abs() < 1e-6);
    }
}
