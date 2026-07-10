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
    let mut scratch = ChScratch::default();
    ch.one_to_many(&buckets, seeds, cutoff, &mut scratch);
    let got = scratch.best();
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
    let mut scratch = ChScratch::default();
    ch.one_to_many(&buckets, &[(0, 0.0)], 3.0, &mut scratch);
    let got = scratch.best();
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
    ch.one_to_many(&buckets, &[(0, 0.0)], 10.0, &mut ChScratch::default());
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
fn consistency_guard_flags_a_tampered_hierarchy() {
    let edges = grid_edges(4);
    let (offsets, targets, meters) = csr(16, &edges);
    let ch = ContractionHierarchy::build(16, &offsets, &targets, &meters);
    assert_eq!(ch.vertex_count(), 16);
    assert!(
        ch.is_consistent(),
        "a freshly built hierarchy is consistent"
    );

    // An out-of-range target — as a corrupted artifact could carry — is caught.
    let mut out_of_range = ContractionHierarchy::build(16, &offsets, &targets, &meters);
    assert!(!out_of_range.up_targets.is_empty());
    out_of_range.up_targets[0] = 999;
    assert!(!out_of_range.is_consistent());

    // A truncated upward-CSR offset array is caught.
    let mut truncated = ContractionHierarchy::build(16, &offsets, &targets, &meters);
    truncated.up_offsets.pop();
    assert!(!truncated.is_consistent());

    // A rank that is no longer a permutation (a duplicate) is caught.
    let mut duplicate_rank = ContractionHierarchy::build(16, &offsets, &targets, &meters);
    duplicate_rank.rank[0] = duplicate_rank.rank[1];
    assert!(!duplicate_rank.is_consistent());

    // A non-finite metre is caught.
    let mut bad_meter = ContractionHierarchy::build(16, &offsets, &targets, &meters);
    assert!(!bad_meter.up_meters.is_empty());
    bad_meter.up_meters[0] = f64::NAN;
    assert!(!bad_meter.is_consistent());

    // The fingerprint reproduces from the same CSR and binds to this graph:
    // a different graph of the same vertex count fingerprints differently.
    assert_eq!(
        ch.graph_fingerprint(),
        csr_fingerprint(&offsets, &targets, &meters)
    );
    let (other_offsets, other_targets, other_meters) =
        csr(16, &[(0, 1, 1.0), (1, 2, 1.0), (3, 4, 2.0)]);
    assert_ne!(
        ch.graph_fingerprint(),
        csr_fingerprint(&other_offsets, &other_targets, &other_meters)
    );
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
