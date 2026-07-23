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

#[test]
fn ch_matches_bounded_dijkstra_on_a_street_network() {
    // A contraction hierarchy built from a StreetNetwork's CSR reproduces
    // its `bounded_dijkstra` shortest walks (the CH-1 integration point,
    // both in the same Hilbert-renumbered vertex space). A path 0-1-2-3
    // (300 m) plus a longer direct 0-3 (350 m) forces interior shortcuts the
    // query must unpack back to 300 m.
    let net = network(
        4,
        0,
        &[
            (0, 1, 100.0, straight((0.0, 0.0), (100.0, 0.0))),
            (1, 2, 100.0, straight((100.0, 0.0), (200.0, 0.0))),
            (2, 3, 100.0, straight((200.0, 0.0), (300.0, 0.0))),
            (0, 3, 350.0, straight((0.0, 0.0), (300.0, 0.0))),
        ],
        vec![],
    )
    .unwrap();
    let ch = crate::ch::ContractionHierarchy::build(
        net.vertex_count(),
        net.arrays().adjacency_offsets(),
        net.arrays().adj_targets(),
        net.arrays().adj_meters(),
    );
    let mut state = SearchState::default();
    for source in 0..net.vertex_count() {
        net.bounded_dijkstra(&[(source, 0.0)], f64::INFINITY, &mut state);
        for target in 0..net.vertex_count() {
            let expected = state.distance(target);
            match ch.distance(source, target) {
                Some(distance) => assert!(
                    (distance - expected).abs() < 1e-6,
                    "ch d({source},{target})={distance} vs bounded_dijkstra {expected}"
                ),
                None => assert!(
                    !expected.is_finite(),
                    "ch says {source}->{target} unreachable, bounded_dijkstra {expected}"
                ),
            }
        }
    }
}

#[test]
fn ch_buckets_match_bounded_dijkstra_on_link_vertices() {
    // A bucket-CH over the stops' link-endpoint vertices reproduces
    // `bounded_dijkstra`'s distances to those vertices from a snapped source
    // (the CH-2 integration point). The stop link-join on top of these vertex
    // distances is validated when CH is wired into `access_stops` (CH-3), so
    // this checks the bucket meeting on a real StreetNetwork CSR.
    let net = network(
        4,
        3,
        &[
            (0, 1, 100.0, straight((0.0, 0.0), (100.0, 0.0))),
            (1, 2, 100.0, straight((100.0, 0.0), (200.0, 0.0))),
            (2, 3, 100.0, straight((200.0, 0.0), (300.0, 0.0))),
            (0, 3, 350.0, straight((0.0, 0.0), (300.0, 0.0))),
        ],
        vec![
            link(0, 0, 0.5, 0.0),
            link(1, 1, 0.5, 0.0),
            link(2, 2, 0.5, 0.0),
        ],
    )
    .unwrap();
    let ch = crate::ch::ContractionHierarchy::build(
        net.vertex_count(),
        net.arrays().adjacency_offsets(),
        net.arrays().adj_targets(),
        net.arrays().adj_meters(),
    );
    let mut targets: Vec<u32> = net
        .links()
        .iter()
        .flat_map(|link| [link.from, link.to])
        .collect();
    targets.sort_unstable();
    targets.dedup();
    let cutoff = 1000.0;
    let buckets = ch.buckets(&targets, cutoff);
    let mut state = SearchState::default();
    // Seed like a snap would: an interior vertex, with a couple of offsets.
    for seeds in [vec![(0u32, 0.0)], vec![(1u32, 10.0), (2u32, 0.0)]] {
        net.bounded_dijkstra(&seeds, cutoff, &mut state);
        let mut scratch = crate::ch::ChScratch::default();
        ch.one_to_many(&buckets, &seeds, cutoff, &mut scratch);
        let got = scratch.best();
        for &target in &targets {
            let expected = state.distance(target);
            if expected <= cutoff + 1e-9 {
                assert!(
                    got.get(&target)
                        .is_some_and(|&d| (d - expected).abs() < 1e-6),
                    "o2m[{target}] = {:?} vs bounded_dijkstra {expected} (seeds {seeds:?})",
                    got.get(&target)
                );
            }
        }
    }
}

#[test]
fn installing_a_hierarchy_keeps_the_walking_results() {
    // `access_stops` and `stop_transfers` return the same stops and walks
    // whether they search the graph (`bounded_dijkstra`) or the installed
    // contraction hierarchy. Distances match within `1e-6` (the hierarchy
    // sums shortcuts in a different order); the stop set and rounded seconds
    // are identical.
    let mut net = network(
        4,
        3,
        &[
            (0, 1, 137.0, straight((0.0, 0.0), (137.0, 0.0))),
            (1, 2, 149.0, straight((137.0, 0.0), (286.0, 0.0))),
            (2, 3, 151.0, straight((286.0, 0.0), (437.0, 0.0))),
            (0, 3, 500.0, straight((0.0, 0.0), (437.0, 0.0))),
        ],
        vec![
            link(0, 0, 0.3, 1.0),
            link(1, 1, 0.5, 2.0),
            link(2, 2, 0.7, 1.5),
        ],
    )
    .unwrap();
    let coord = lonlat(70.0, 0.0); // (lon, lat) near vertex 0's edge
    let base_access = net
        .access_stops(coord.1, coord.0, 1.0, 600.0, 100.0)
        .unwrap();
    let base_transfers = net.stop_transfers(1.0, 600.0);

    net.install_hierarchy();
    assert!(net.has_hierarchy());
    let ch_access = net
        .access_stops(coord.1, coord.0, 1.0, 600.0, 100.0)
        .unwrap();
    let ch_transfers = net.stop_transfers(1.0, 600.0);

    assert_eq!(timed(&ch_access), timed(&base_access));
    for (a, b) in ch_access.iter().zip(&base_access) {
        assert!((a.meters - b.meters).abs() < 1e-6, "{a:?} vs {b:?}");
    }
    let key = |edges: &[(StopIdx, StopIdx, u32, f64)]| -> HashMap<(StopIdx, StopIdx), (u32, f64)> {
        edges.iter().map(|&(f, t, s, m)| ((f, t), (s, m))).collect()
    };
    let (base_map, ch_map) = (key(&base_transfers), key(&ch_transfers));
    assert_eq!(
        base_map.keys().collect::<std::collections::BTreeSet<_>>(),
        ch_map.keys().collect::<std::collections::BTreeSet<_>>()
    );
    for (pair, &(seconds, meters)) in &base_map {
        let &(ch_seconds, ch_meters) = &ch_map[pair];
        assert_eq!(ch_seconds, seconds, "transfer {pair:?} seconds");
        assert!(
            (ch_meters - meters).abs() < 1e-6,
            "transfer {pair:?} meters"
        );
    }
}

/// The `(stop, seconds)` view of a walking-search result.
fn timed(walks: &[WalkedStop]) -> Vec<(StopIdx, u32)> {
    walks.iter().map(|walk| (walk.stop, walk.seconds)).collect()
}

/// Asserts two link results (per point, walkable stops or `None`) agree on the
/// stop set and rounded seconds, metres to 1e-6.
fn assert_links_match(via: &[Option<Vec<WalkedStop>>], many: &[Option<Vec<WalkedStop>>]) {
    assert_eq!(via.len(), many.len(), "point count differs");
    for (a, b) in via.iter().zip(many) {
        match (a, b) {
            (Some(a), Some(b)) => {
                assert_eq!(timed(a), timed(b), "{a:?} vs {b:?}");
                for (x, y) in a.iter().zip(b) {
                    assert!((x.meters - y.meters).abs() < 1e-6, "{x:?} vs {y:?}");
                }
            }
            (None, None) => {}
            _ => panic!("snap disagreement: {a:?} vs {b:?}"),
        }
    }
}

#[test]
fn link_pointsets_matches_link_many() {
    // Linking from the stop side (link_pointsets) returns the same stop sets and
    // rounded seconds as the per-point search (link_many), metres within 1e-6
    // (the two search directions sum a path in opposite order). Checked over a
    // spread of coordinates, both engines, and both a single set and two sets
    // sharing one stop-search pass.
    let mut net = network(
        4,
        3,
        &[
            (0, 1, 137.0, straight((0.0, 0.0), (137.0, 0.0))),
            (1, 2, 149.0, straight((137.0, 0.0), (286.0, 0.0))),
            (2, 3, 151.0, straight((286.0, 0.0), (437.0, 0.0))),
            (0, 3, 500.0, straight((0.0, 0.0), (437.0, 0.0))),
        ],
        vec![
            link(0, 0, 0.3, 1.0),
            link(1, 1, 0.5, 2.0),
            link(2, 2, 0.7, 1.5),
        ],
    )
    .unwrap();
    // The linking APIs take (latitude, longitude); `lonlat` builds the
    // opposite order, so swap.
    let point = |x: f64, y: f64| {
        let (lon, lat) = lonlat(x, y);
        (lat, lon)
    };
    let coords: Vec<(f64, f64)> = vec![
        point(70.0, 0.0),      // mid edge 0
        point(140.0, 0.0),     // near stop 1
        point(400.0, 0.0),     // on edge 2 near stop 2
        point(41.0, 0.0),      // same edge as stop 0's link
        point(5000.0, 5000.0), // beyond the snap distance -> None
    ];
    let check = |net: &StreetNetwork| {
        for &speed in &[1.0, 1.4] {
            for &max_seconds in &[300.0, 900.0] {
                let many = net.link_many(&coords, speed, max_seconds, 100.0);
                // The comparison must not pass vacuously: the on-network
                // points snap and reach stops, the far one stays None.
                let linked = many.iter().filter(|links| links.is_some()).count();
                assert_eq!(linked, 4, "expected 4 snapped points: {many:?}");
                assert!(
                    many.iter().flatten().any(|walks| !walks.is_empty()),
                    "no point reached any stop: {many:?}"
                );
                let single = net.link_pointsets(&[&coords[..]], speed, max_seconds, 100.0);
                assert_eq!(single.len(), 1);
                assert_links_match(&single[0], &many);
                // Two sets share one stop-search pass; each matches its link_many.
                let (a, b) = coords.split_at(2);
                let pair = net.link_pointsets(&[a, b], speed, max_seconds, 100.0);
                assert_eq!(pair.len(), 2);
                assert_links_match(&pair[0], &net.link_many(a, speed, max_seconds, 100.0));
                assert_links_match(&pair[1], &net.link_many(b, speed, max_seconds, 100.0));
            }
        }
    };
    check(&net);
    net.install_hierarchy();
    check(&net);
}

/// Asserts designed walking times, allowing the one extra second that
/// conservative rounding may add when coordinate quantization (≤ ~2 cm
/// per segment) nudges a designed-exact distance past a whole second.
fn assert_walks(walks: &[WalkedStop], designed: &[(u32, u32)]) {
    assert_eq!(walks.len(), designed.len(), "stops differ: {walks:?}");
    for (walk, &(stop, seconds)) in walks.iter().zip(designed) {
        assert_eq!(walk.stop, StopIdx(stop), "stops differ: {walks:?}");
        assert!(
            walk.seconds >= seconds && walk.seconds <= seconds + 1,
            "stop {stop}: {} s, designed {seconds} s",
            walk.seconds
        );
    }
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
    assert!((snap.fraction - 0.25).abs() < 1e-4);
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
    let count = network.arrays().coordinate_offsets()[1] as usize;
    assert!(count >= 51, "expected >=51 densified points, got {count}");
    for pair in network.arrays().lats().windows(2) {
        let seg = segment_length(25.0, degrees(pair[0]), 25.0, degrees(pair[1]));
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
    for (lons, lats) in network
        .arrays()
        .lons()
        .windows(2)
        .zip(network.arrays().lats().windows(2))
    {
        let seg = segment_length(
            degrees(lons[0]),
            degrees(lats[0]),
            degrees(lons[1]),
            degrees(lats[1]),
        );
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

    // The same snapped point routes to itself: a zero-length path.
    let same_point = network.walk_path((origin.1, origin.0), &from, (origin.1, origin.0), &from);
    assert!(same_point.is_some());
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
    assert_walks(&reached, &[(0, 0), (1, 200)]);
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
    assert_walks(&reached, &[(0, 400)]);
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
    assert_walks(&reached, &[(0, 400)]);
}

/// The `(from, to, seconds)` view of a transfer set, dropping the
/// exact meters (checked separately where they matter).
fn transfer_times(edges: &[(StopIdx, StopIdx, u32, f64)]) -> Vec<(u32, u32, u32)> {
    edges
        .iter()
        .map(|&(from, to, seconds, _)| (from.0, to.0, seconds))
        .collect()
}

#[test]
fn stop_transfers_are_direct_walks_within_the_cutoff() {
    // Three stops along one 1000 m edge at 100/400/900 m. Pairwise
    // walks are 300, 500, 800 m; at a 600 m cutoff only the 300 m and
    // 500 m pairs survive — the 800 m pair is past the cutoff and is
    // never padded back in by chaining through the middle stop.
    let network = network(
        2,
        3,
        &[(0, 1, 1000.0, straight((0.0, 0.0), (1000.0, 0.0)))],
        vec![
            link(0, 0, 0.1, 0.0),
            link(1, 0, 0.4, 0.0),
            link(2, 0, 0.9, 0.0),
        ],
    )
    .unwrap();
    let edges = network.stop_transfers(1.0, 600.0);
    assert_eq!(
        transfer_times(&edges),
        vec![(0, 1, 300), (1, 0, 300), (1, 2, 500), (2, 1, 500)],
        "{edges:?}"
    );
    for &(_, _, seconds, meters) in &edges {
        assert!(
            (meters - f64::from(seconds)).abs() < 1.0,
            "meters {meters} vs seconds {seconds}"
        );
    }
}

#[test]
fn stop_transfers_are_symmetric() {
    // Walking is undirected, so every A→B edge has a B→A twin with the
    // same walk. An L of two edges with three stops exercises walks
    // that run through a vertex.
    let network = network(
        3,
        3,
        &[
            (0, 1, 300.0, straight((0.0, 0.0), (300.0, 0.0))),
            (1, 2, 400.0, straight((300.0, 0.0), (300.0, 400.0))),
        ],
        vec![
            link(0, 0, 0.2, 0.0),
            link(1, 0, 0.9, 0.0),
            link(2, 1, 0.5, 0.0),
        ],
    )
    .unwrap();
    let edges = network.stop_transfers(1.0, 900.0);
    for &(from, to, seconds, meters) in &edges {
        let twin = edges
            .iter()
            .find(|&&(a, b, _, _)| a == to && b == from)
            .unwrap_or_else(|| panic!("no twin for {from:?}->{to:?} in {edges:?}"));
        assert_eq!(twin.2, seconds, "asymmetric seconds {edges:?}");
        assert!(
            (twin.3 - meters).abs() < 1e-9,
            "asymmetric meters {edges:?}"
        );
    }
}

#[test]
fn stop_transfers_search_from_all_source_links() {
    // Source stop 0 snaps to both of two disconnected edges; stop 1
    // sits only on the first, stop 2 only on the second. Searching
    // from a single (nearest) link would reach one of them; searching
    // from all links reaches both, symmetrically.
    let network = network(
        4,
        3,
        &[
            (0, 1, 400.0, straight((0.0, 0.0), (400.0, 0.0))),
            (2, 3, 400.0, straight((0.0, 1000.0), (400.0, 1000.0))),
        ],
        vec![
            link(0, 0, 0.5, 0.0),
            link(0, 1, 0.5, 0.0),
            link(1, 0, 0.2, 0.0),
            link(2, 1, 0.8, 0.0),
        ],
    )
    .unwrap();
    let edges = network.stop_transfers(1.0, 600.0);
    assert_eq!(
        transfer_times(&edges),
        vec![(0, 1, 120), (0, 2, 120), (1, 0, 120), (2, 0, 120)],
        "{edges:?}"
    );
}

#[test]
fn stop_transfers_skip_disconnected_stops() {
    // Two separate street components, one stop on each: neither can
    // walk to the other, so the transfer set is empty.
    let network = network(
        4,
        2,
        &[
            (0, 1, 400.0, straight((0.0, 0.0), (400.0, 0.0))),
            (2, 3, 400.0, straight((0.0, 1000.0), (400.0, 1000.0))),
        ],
        vec![link(0, 0, 0.5, 0.0), link(1, 1, 0.5, 0.0)],
    )
    .unwrap();
    assert!(network.stop_transfers(1.0, 3600.0).is_empty());
}

#[test]
fn stop_transfers_reject_invalid_parameters() {
    let network = network(
        2,
        2,
        &[(0, 1, 400.0, straight((0.0, 0.0), (400.0, 0.0)))],
        vec![link(0, 0, 0.25, 0.0), link(1, 0, 0.75, 0.0)],
    )
    .unwrap();
    assert!(network.stop_transfers(0.0, 600.0).is_empty());
    assert!(network.stop_transfers(f64::NAN, 600.0).is_empty());
    assert!(network.stop_transfers(1.0, -1.0).is_empty());
    assert!(network.stop_transfers(1.0, f64::INFINITY).is_empty());
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
    assert_walks(&reached, &[(0, 300)]);
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
    assert_walks(&reached, &[(0, 100)]);
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
    // meters shift only within the coordinate-quantization bound.
    assert_walks(&reached, &[(0, 201)]);
    assert!((reached[0].meters - 200.5).abs() < 0.05);
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
    assert_walks(&reached, &[(0, 100)]);
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
        StreetNetwork::new(2, 0, &[(0, 1, 400.0)], &[0, 1], &[24.0], &[60.0], vec![]).unwrap_err(),
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
    let mut lons: Vec<i32> = Vec::new();
    let mut lats: Vec<i32> = Vec::new();
    for _ in 0..120 {
        let points = 2 + (random() * 4.0) as usize;
        for _ in 0..points {
            lons.push(quantize(24.0 + random() * 0.5));
            lats.push(quantize(60.0 + random() * 0.5));
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
        let envelope = [
            quantize(lon - dlon),
            quantize(lat - dlat),
            quantize(lon + dlon),
            quantize(lat + dlat),
        ];
        query_packed_index(
            &index.boxes,
            &index.payload,
            &index.level_starts,
            &envelope,
            &mut matches,
        );
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

    let shuffled_edges: Vec<TestEdge> = vec![edges[2].clone(), edges[0].clone(), edges[1].clone()];
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
    assert!((snap.fraction - 0.5).abs() < 1e-4);
    assert!((snap.connector - 5.0).abs() < 0.05);
    let reached = network.access_stops(lat, lon, 1.0, 600.0, 100.0).unwrap();
    assert_walks(&reached, &[(0, 30)]);
}

#[test]
fn walks_to_many_snapped_points() {
    // An L of two edges; targets on both edges and off-network.
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
    let snap_at = |x: f64, y: f64| {
        let (lon, lat) = lonlat(x, y);
        network.snap(lat, lon, 100.0)
    };
    let from = snap_at(50.0, 10.0).unwrap();
    let targets = vec![
        snap_at(250.0, 0.0),   // same edge: direct along it
        snap_at(300.0, 150.0), // around the corner
        None,                  // unsnapped point
    ];
    let walks = network.walk_to_snaps(&from, &targets, 1.0, 1200.0);
    // Same edge: 10 m connector + 200 m along; corner: 10 + 250 + 150.
    let (seconds_a, meters_a) = walks[0].unwrap();
    assert!((meters_a - 210.0).abs() < 0.1, "{meters_a}");
    assert!((210..=211).contains(&seconds_a));
    let (seconds_b, meters_b) = walks[1].unwrap();
    assert!((meters_b - 410.0).abs() < 0.1, "{meters_b}");
    assert!((410..=411).contains(&seconds_b));
    assert!(walks[2].is_none());
    // A tight cutoff drops the farther target only.
    let close = network.walk_to_snaps(&from, &targets, 1.0, 300.0);
    assert!(close[0].is_some() && close[1].is_none());
    // The matrix driver agrees with the single-origin search.
    let (from_lon, from_lat) = lonlat(50.0, 10.0);
    let (to_lon, to_lat) = lonlat(300.0, 150.0);
    let matrix = network.walk_matrix(
        &[(from_lat, from_lon)],
        &[(to_lat, to_lon), (89.0, 0.0), (from_lat, from_lon)],
        1.0,
        1200.0,
        100.0,
    );
    assert_eq!(matrix[0][0], walks[1]);
    assert!(matrix[0][1].is_none());
    // The origin's own coordinate is a zero walk, not a trip out to
    // the street and back over the connector.
    assert_eq!(matrix[0][2], Some((0, 0.0)));
}

/// A test backing whose buffer is 8-byte aligned, as a real mapping's
/// is; a plain `Vec<u8>` only guarantees byte alignment.
struct AlignedBytes {
    words: Vec<u64>,
    len: usize,
}

impl AlignedBytes {
    fn from_bytes(bytes: &[u8]) -> AlignedBytes {
        let mut words = vec![0u64; bytes.len().div_ceil(8)];
        // SAFETY: the word buffer is at least `bytes.len()` bytes.
        unsafe {
            std::slice::from_raw_parts_mut(words.as_mut_ptr().cast::<u8>(), bytes.len())
                .copy_from_slice(bytes);
        }
        AlignedBytes {
            words,
            len: bytes.len(),
        }
    }
}

impl Backing for AlignedBytes {
    fn bytes(&self) -> &[u8] {
        // SAFETY: the words hold `len` initialized bytes.
        unsafe { std::slice::from_raw_parts(self.words.as_ptr().cast::<u8>(), self.len) }
    }
}

/// Lays a network's parts out as a mapped artifact would: each array's
/// native-endian bytes at the next 8-byte boundary of one buffer.
fn mapped_from(owned: &StreetNetwork) -> StreetNetwork {
    fn push<T: Copy>(bytes: &mut Vec<u8>, values: &[T]) -> (u64, u64) {
        while !bytes.len().is_multiple_of(8) {
            bytes.push(0);
        }
        let offset = bytes.len() as u64;
        // SAFETY: the arrays are plain-old-data numeric types.
        let raw = unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        };
        bytes.extend_from_slice(raw);
        (offset, values.len() as u64)
    }
    let parts = owned.to_parts();
    let mut bytes = Vec::new();
    let adjacency_offsets = push(&mut bytes, &parts.adjacency_offsets);
    let adj_targets = push(&mut bytes, &parts.adj_targets);
    let adj_meters = push(&mut bytes, &parts.adj_meters);
    let adj_edges = push(&mut bytes, &parts.adj_edges);
    let endpoints = push(&mut bytes, &parts.endpoints);
    let lengths = push(&mut bytes, &parts.lengths);
    let coordinate_offsets = push(&mut bytes, &parts.coordinate_offsets);
    let lons = push(&mut bytes, &parts.lons);
    let lats = push(&mut bytes, &parts.lats);
    let cumulative = push(&mut bytes, &parts.cumulative);
    let index_boxes = push(&mut bytes, &parts.index_boxes);
    let index_payload = push(&mut bytes, &parts.index_payload);
    StreetNetwork::from_mapped(MappedStreets {
        backing: std::sync::Arc::new(AlignedBytes::from_bytes(&bytes)),
        vertex_count: parts.vertex_count,
        links: parts.links,
        adjacency_offsets,
        adj_targets,
        adj_meters,
        adj_edges,
        endpoints,
        lengths,
        coordinate_offsets,
        lons,
        lats,
        cumulative,
        index_boxes,
        index_payload,
    })
    .unwrap()
}

#[test]
fn mapped_networks_match_owned() {
    let owned = network(
        4,
        2,
        &[
            (0, 1, 400.0, straight((0.0, 0.0), (400.0, 0.0))),
            (1, 2, 300.0, straight((400.0, 0.0), (400.0, 300.0))),
            (2, 3, 200.0, straight((400.0, 300.0), (600.0, 300.0))),
        ],
        vec![link(0, 1, 0.5, 10.0), link(1, 2, 1.0, 0.0)],
    )
    .unwrap();
    let mapped = mapped_from(&owned);
    assert!(mapped.is_mapped() && !owned.is_mapped());
    // The mapped view serializes back to the identical parts …
    assert_eq!(mapped.to_parts(), owned.to_parts());
    // … and answers queries identically.
    for &(x, y) in &[(50.0, 5.0), (400.0, 150.0), (590.0, 290.0)] {
        let (lon, lat) = lonlat(x, y);
        assert_eq!(mapped.snap(lat, lon, 100.0), owned.snap(lat, lon, 100.0));
        assert_eq!(
            mapped.access_stops(lat, lon, 1.0, 1200.0, 100.0),
            owned.access_stops(lat, lon, 1.0, 1200.0, 100.0)
        );
    }
    let from = owned.snap(lonlat(50.0, 5.0).1, lonlat(50.0, 5.0).0, 100.0);
    let to = owned.snap(lonlat(590.0, 290.0).1, lonlat(590.0, 290.0).0, 100.0);
    let (from, to) = (from.unwrap(), to.unwrap());
    let from_point = (lonlat(50.0, 5.0).1, lonlat(50.0, 5.0).0);
    let to_point = (lonlat(590.0, 290.0).1, lonlat(590.0, 290.0).0);
    assert_eq!(
        mapped.walk_path(from_point, &from, to_point, &to),
        owned.walk_path(from_point, &from, to_point, &to)
    );
}

#[test]
fn mapped_adoption_refuses_misaligned_or_truncated_ranges() {
    let owned = network(
        2,
        1,
        &[(0, 1, 50.0, straight((0.0, 0.0), (50.0, 0.0)))],
        vec![link(0, 0, 1.0, 0.0)],
    )
    .unwrap();
    let parts = owned.to_parts();
    let backing: std::sync::Arc<dyn Backing> =
        std::sync::Arc::new(AlignedBytes::from_bytes(&[0u8; 64]));
    let spec = |lengths: (u64, u64)| MappedStreets {
        backing: backing.clone(),
        vertex_count: parts.vertex_count,
        links: parts.links.clone(),
        adjacency_offsets: (0, 3),
        adj_targets: (16, 2),
        adj_meters: (24, 2),
        adj_edges: (40, 2),
        endpoints: (48, 2),
        lengths,
        coordinate_offsets: (0, 2),
        lons: (0, 2),
        lats: (0, 2),
        cumulative: (0, 2),
        index_boxes: (0, 4),
        index_payload: (0, 2),
    };
    // An f64 array at a 4-byte offset is misaligned; one past the
    // buffer is out of bounds.
    assert!(StreetNetwork::from_mapped(spec((4, 1))).is_err());
    assert!(StreetNetwork::from_mapped(spec((56, 2))).is_err());
    assert!(StreetNetwork::from_mapped(spec((56, 1))).is_ok());
}
