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
        attributes: None,
        car: None,
        elevations: None,
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
        attributes: None,
        car: None,
        elevations: None,
    };
    // An f64 array at a 4-byte offset is misaligned; one past the
    // buffer is out of bounds.
    assert!(StreetNetwork::from_mapped(spec((4, 1))).is_err());
    assert!(StreetNetwork::from_mapped(spec((56, 2))).is_err());
    assert!(StreetNetwork::from_mapped(spec((56, 1))).is_ok());
}

fn triangle() -> StreetNetwork {
    network(
        4,
        0,
        &[
            (0, 1, 400.0, straight((0.0, 0.0), (400.0, 0.0))),
            (1, 2, 300.0, straight((400.0, 0.0), (400.0, 300.0))),
            (2, 3, 200.0, straight((400.0, 300.0), (600.0, 300.0))),
        ],
        vec![],
    )
    .unwrap()
}

fn uniform_attributes(network: &StreetNetwork, access: u8, flags: u16) -> StreetAttributes {
    let edges = network.edge_count() as usize;
    let slots = 2 * edges;
    StreetAttributes {
        adj_access: vec![access; slots],
        adj_facility: vec![0; slots],
        edge_highway: vec![0; edges],
        edge_surface: vec![0; edges],
        edge_smoothness: vec![0; edges],
        edge_flags: vec![flags; edges],
    }
}

#[test]
fn profile_compiles_permitted_arc_costs() {
    // A permitted arc costs ceil(length / speed * 1000) ms; the bicycle
    // default is 4 m/s.
    let mut net = triangle();
    net.install_street_attributes(uniform_attributes(&net, MODE_WALK | MODE_BICYCLE, 0))
        .unwrap();
    let bike = net
        .compile_profile(&StreetProfileDefinition::bicycle())
        .unwrap();
    let meters = net.arrays().adj_meters().to_vec();
    assert_eq!(bike.arc_millis().len(), meters.len());
    for (&length, &cost) in meters.iter().zip(bike.arc_millis()) {
        assert_eq!(cost, (length / 4.0 * 1000.0).ceil() as u32);
    }
}

#[test]
fn profile_forbids_arcs_missing_the_mode() {
    // With walk-only access, the walking profile permits every arc while the
    // bicycle profile forbids every arc (u32::MAX).
    let mut net = triangle();
    net.install_street_attributes(uniform_attributes(&net, MODE_WALK, 0))
        .unwrap();
    let walk = net
        .compile_profile(&StreetProfileDefinition::walk())
        .unwrap();
    let bike = net
        .compile_profile(&StreetProfileDefinition::bicycle())
        .unwrap();
    assert!(walk.arc_millis().iter().all(|&cost| cost != u32::MAX));
    assert!(bike.arc_millis().iter().all(|&cost| cost == u32::MAX));
}

#[test]
fn profile_applies_class_multipliers() {
    // Halving the base speed on every arc's highway class (code 0) doubles the
    // traversal time.
    let mut net = triangle();
    net.install_street_attributes(uniform_attributes(&net, MODE_WALK | MODE_BICYCLE, 0))
        .unwrap();
    let mut slow = StreetProfileDefinition::bicycle();
    slow.highway_multipliers[0] = 0.5;
    let compiled = net.compile_profile(&slow).unwrap();
    let meters = net.arrays().adj_meters().to_vec();
    for (&length, &cost) in meters.iter().zip(compiled.arc_millis()) {
        assert_eq!(cost, (length / (4.0 * 0.5) * 1000.0).ceil() as u32);
    }
}

#[test]
fn profile_dismount_arc_falls_to_walk_speed() {
    // A dismount arc costs walking time for the bicycle, not bicycle time.
    let mut net = triangle();
    net.install_street_attributes(uniform_attributes(
        &net,
        MODE_WALK | MODE_BICYCLE,
        FLAG_DISMOUNT,
    ))
    .unwrap();
    let bike = net
        .compile_profile(&StreetProfileDefinition::bicycle())
        .unwrap();
    let meters = net.arrays().adj_meters().to_vec();
    for (&length, &cost) in meters.iter().zip(bike.arc_millis()) {
        assert_eq!(cost, (length / 1.0 * 1000.0).ceil() as u32);
    }
}

#[test]
fn profile_walk_compiles_without_attributes() {
    // A graph without installed attributes has no per-mode permissions: the
    // walking profile permits every arc, while a non-walk profile has nothing
    // to route by and is rejected rather than returning an all-forbidden set.
    let net = triangle();
    let walk = net
        .compile_profile(&StreetProfileDefinition::walk())
        .unwrap();
    let meters = net.arrays().adj_meters().to_vec();
    for (&length, &walk_cost) in meters.iter().zip(walk.arc_millis()) {
        assert_eq!(walk_cost, (length * 1000.0).ceil() as u32);
    }
    assert_eq!(
        net.compile_profile(&StreetProfileDefinition::bicycle())
            .unwrap_err(),
        ProfileError::MissingAttributes
    );
}

#[test]
fn compile_rejects_pathological_costs_and_codes() {
    let mut net = triangle();
    net.install_street_attributes(uniform_attributes(&net, MODE_WALK | MODE_BICYCLE, 0))
        .unwrap();
    // A dismount speed above max_speed would break the declared upper bound
    // on on-network speeds, so it is rejected by validation.
    let mut fast_dismount = StreetProfileDefinition::bicycle();
    fast_dismount.dismount_speed = 10.0; // > max_speed 4
    assert_eq!(
        net.compile_profile(&fast_dismount).unwrap_err(),
        ProfileError::MaxSpeedTooLow
    );
    // A physically implausible speed overflows the millisecond range and is
    // rejected rather than silently clamped. (All speeds tiny so the max-speed
    // bound is satisfied and compilation reaches the overflowing cost.)
    let mut crawl = StreetProfileDefinition::bicycle();
    crawl.slope_uphill = 0.0;
    crawl.slope_downhill = 0.0;
    crawl.base_speed = 1e-9;
    crawl.dismount_speed = 1e-9;
    crawl.connector_speed = 1e-9;
    crawl.max_speed = 1e-9;
    assert_eq!(
        net.compile_profile(&crawl).unwrap_err(),
        ProfileError::ArcCostOverflow
    );
    // An out-of-range class code is a drifted-ABI signal and is refused at
    // install, not defaulted to a neutral multiplier.
    let mut fresh = triangle();
    let mut attributes = uniform_attributes(&fresh, MODE_WALK | MODE_BICYCLE, 0);
    attributes.edge_highway[0] = HIGHWAY_CODE_COUNT as u8;
    assert_eq!(
        fresh.install_street_attributes(attributes),
        Err(StreetError::InvalidAttributes)
    );
}

#[test]
fn builtin_profile_speeds_and_modes() {
    assert_eq!(StreetProfileDefinition::walk().base_speed, 1.0);
    assert_eq!(StreetProfileDefinition::bicycle().base_speed, 4.0);
    assert_eq!(StreetProfileDefinition::e_bike().base_speed, 4.0);
    assert_eq!(StreetProfileDefinition::e_scooter().base_speed, 15.0 / 3.6);
    assert_eq!(StreetProfileDefinition::walk().mode, StreetMode::Walk);
    assert_eq!(StreetProfileDefinition::bicycle().mode, StreetMode::Bicycle);
    assert_eq!(
        StreetProfileDefinition::e_scooter().mode,
        StreetMode::EScooter
    );
    // The bicycle's slope model is the owner's methodology, and its max
    // speed covers the steepest clamped descent's credit; the multiplier
    // tables match the class-code counts.
    let bike = StreetProfileDefinition::bicycle();
    assert_eq!((bike.slope_uphill, bike.slope_downhill), (1.0, 0.3));
    assert_eq!(bike.max_speed, 4.0 / (1.0 - 0.3 * MAX_SLOPE));
    assert_eq!(StreetProfileDefinition::e_bike().slope_uphill, 0.0);
    assert_eq!(StreetProfileDefinition::walk().slope_downhill, 0.0);
    assert_eq!(bike.highway_multipliers.len(), HIGHWAY_CODE_COUNT);
    assert_eq!(bike.surface_multipliers.len(), SURFACE_CODE_COUNT);
    assert_eq!(bike.smoothness_multipliers.len(), SMOOTHNESS_CODE_COUNT);
}

#[test]
fn profile_definition_equality_binds_exactly() {
    let mut modified = StreetProfileDefinition::bicycle();
    assert_eq!(modified, StreetProfileDefinition::bicycle());
    modified.surface_multipliers[2] = 0.75;
    assert_ne!(modified, StreetProfileDefinition::bicycle());
    // e-bike routes like a bicycle but is a distinct definition (its vehicle
    // class differs), so equality separates them.
    assert_ne!(
        StreetProfileDefinition::e_bike(),
        StreetProfileDefinition::bicycle()
    );
}

#[test]
fn compile_rejects_invalid_definitions() {
    // Attributes installed so the final valid case reaches (and passes) the
    // compile loop; the validation errors below fire before it is consulted.
    let mut net = triangle();
    net.install_street_attributes(uniform_attributes(&net, MODE_WALK | MODE_BICYCLE, 0))
        .unwrap();
    // A non-finite or non-positive speed is rejected.
    let mut zero = StreetProfileDefinition::bicycle();
    zero.base_speed = 0.0;
    assert_eq!(
        net.compile_profile(&zero).unwrap_err(),
        ProfileError::NonPositiveSpeed
    );
    let mut infinite = StreetProfileDefinition::bicycle();
    infinite.base_speed = f64::INFINITY;
    assert_eq!(
        net.compile_profile(&infinite).unwrap_err(),
        ProfileError::NonPositiveSpeed
    );
    // A wrong-length multiplier table is rejected.
    let mut short = StreetProfileDefinition::bicycle();
    short.surface_multipliers.pop();
    assert_eq!(
        net.compile_profile(&short).unwrap_err(),
        ProfileError::InvalidMultipliers
    );
    // A max_speed below the greatest attainable speed is rejected.
    let mut fast = StreetProfileDefinition::bicycle();
    fast.slope_uphill = 0.0;
    fast.slope_downhill = 0.0;
    fast.highway_multipliers[5] = 2.0; // attainable = 8 m/s > max_speed 4
    assert_eq!(
        net.compile_profile(&fast).unwrap_err(),
        ProfileError::MaxSpeedTooLow
    );
    fast.max_speed = 8.0;
    assert!(net.compile_profile(&fast).is_ok());
}

#[test]
fn load_rejects_invalid_persisted_attributes() {
    // A drifted-ABI or corrupted artifact must be refused on adoption, not
    // compiled into wrong costs. The owned (`from_parts`) and mapped
    // (`from_mapped`) paths run the same attribute check; the owned path is
    // exercised here through the public parts round-trip.
    let mut net = triangle();
    net.install_street_attributes(uniform_attributes(&net, MODE_WALK | MODE_BICYCLE, 0))
        .unwrap();
    // A well-formed set adopts.
    assert!(StreetNetwork::from_parts(net.to_parts()).is_ok());
    // An out-of-range class code is rejected.
    let mut bad_code = net.to_parts();
    bad_code.attributes.as_mut().unwrap().edge_surface[0] = SURFACE_CODE_COUNT as u8;
    assert_eq!(
        StreetNetwork::from_parts(bad_code).unwrap_err(),
        StreetError::InvalidAttributes
    );
    // A wrong-length attribute array is rejected.
    let mut wrong_len = net.to_parts();
    wrong_len.attributes.as_mut().unwrap().edge_highway.pop();
    assert_eq!(
        StreetNetwork::from_parts(wrong_len).unwrap_err(),
        StreetError::InvalidAttributes
    );
}

#[test]
fn dismount_is_a_per_edge_flag_on_both_directions() {
    // `bicycle=dismount` is a way-level flag in `edge_flags`, so it slows both
    // of the dismount edge's arcs and no others.
    let mut net = triangle();
    let mut attributes = uniform_attributes(&net, MODE_WALK | MODE_BICYCLE, 0);
    attributes.edge_flags[1] = FLAG_DISMOUNT;
    net.install_street_attributes(attributes).unwrap();
    let bike = net
        .compile_profile(&StreetProfileDefinition::bicycle())
        .unwrap();
    let meters = net.arrays().adj_meters().to_vec();
    let adj_edges = net.arrays().adj_edges().to_vec();
    for ((&length, &edge), &cost) in meters.iter().zip(&adj_edges).zip(bike.arc_millis()) {
        let speed = if edge == 1 { 1.0 } else { 4.0 };
        assert_eq!(cost, (length / speed * 1000.0).ceil() as u32);
    }
    // The one dismount edge contributes exactly two (both-direction) arcs.
    assert_eq!(adj_edges.iter().filter(|&&edge| edge == 1).count(), 2);
}

#[test]
fn attribute_free_walk_bound_includes_base_speed() {
    // On an attribute-free graph the class multipliers are not applied, so a
    // walk profile runs at base_speed: max_speed below base_speed is rejected
    // even when every multiplier is below one.
    let net = triangle();
    let mut slow = StreetProfileDefinition::walk();
    slow.highway_multipliers.fill(0.5);
    slow.max_speed = 0.5; // below base_speed 1.0
    assert_eq!(
        net.compile_profile(&slow).unwrap_err(),
        ProfileError::MaxSpeedTooLow
    );
    slow.max_speed = 1.0;
    let walk = net.compile_profile(&slow).unwrap();
    let meters = net.arrays().adj_meters().to_vec();
    for (&length, &cost) in meters.iter().zip(walk.arc_millis()) {
        assert_eq!(cost, (length / 1.0 * 1000.0).ceil() as u32);
    }
}

#[test]
fn profile_abi_constants_mirror_the_python_contract() {
    // These must match python/cafein/_osm.py exactly — the raw u8/u16 attribute
    // arrays cross the language boundary as these integers, so a change on
    // either side is a breaking ABI change that must be made on both.
    assert_eq!(
        (MODE_WALK, MODE_BICYCLE, MODE_E_SCOOTER, MODE_CAR),
        (1, 2, 4, 8)
    );
    assert_eq!(
        (
            FLAG_DISMOUNT,
            FLAG_BRIDGE,
            FLAG_TUNNEL,
            FLAG_INDOOR,
            FLAG_STEPS,
            FLAG_SEGREGATED,
            FLAG_LIT,
            FLAG_ROUNDABOUT,
        ),
        (1, 2, 4, 8, 16, 32, 64, 128)
    );
    // The persisted junction head-class vocabulary (`_osm.py`'s JUNCTION_*).
    assert_eq!(
        (
            JUNCTION_TOPOLOGICAL,
            JUNCTION_PRIORITY,
            JUNCTION_SIGNALS,
            JUNCTION_RAMP,
            JUNCTION_CLASS_COUNT,
        ),
        (1, 2, 3, 4, 5)
    );
    assert_eq!(
        (
            HIGHWAY_CODE_COUNT,
            SURFACE_CODE_COUNT,
            SMOOTHNESS_CODE_COUNT
        ),
        (27, 17, 9)
    );
}

#[test]
fn profile_rounds_arc_costs_up() {
    // 3 m/s makes several arcs' millisecond costs fractional; the compiler must
    // round up, not truncate.
    let mut net = triangle();
    net.install_street_attributes(uniform_attributes(&net, MODE_WALK | MODE_BICYCLE, 0))
        .unwrap();
    let mut three = StreetProfileDefinition::bicycle();
    three.slope_uphill = 0.0;
    three.slope_downhill = 0.0;
    three.base_speed = 3.0;
    three.max_speed = 3.0;
    let compiled = net.compile_profile(&three).unwrap();
    let meters = net.arrays().adj_meters().to_vec();
    let mut saw_fractional = false;
    for (&length, &cost) in meters.iter().zip(compiled.arc_millis()) {
        let exact = length / 3.0 * 1000.0;
        assert_eq!(cost, exact.ceil() as u32);
        if exact.fract() != 0.0 {
            saw_fractional = true;
        }
    }
    assert!(
        saw_fractional,
        "expected at least one fractional-millisecond arc"
    );
}

#[test]
fn positive_length_arc_never_costs_zero() {
    // A tiny length with an extreme speed underflows the raw product to `0.0`;
    // a positive-length arc must still cost at least 1 ms.
    let mut net = network(
        2,
        0,
        &[(0, 1, 1e-200, straight((0.0, 0.0), (10.0, 0.0)))],
        vec![],
    )
    .unwrap();
    net.install_street_attributes(uniform_attributes(&net, MODE_WALK | MODE_BICYCLE, 0))
        .unwrap();
    let mut extreme = StreetProfileDefinition::bicycle();
    extreme.slope_uphill = 0.0;
    extreme.slope_downhill = 0.0;
    extreme.base_speed = 1e300;
    extreme.max_speed = 1e300;
    let compiled = net.compile_profile(&extreme).unwrap();
    assert!(compiled.arc_millis().iter().all(|&cost| cost >= 1));
}

// --- Target-directed A* ------------------------------------------------------

/// A deterministic pseudo-random street grid: `width × height` vertices,
/// jittered edge lengths (always at least the straight line), bicycle one-way
/// arcs, dismount edges, and a hilly DEM — everything the compiled costs can
/// vary over, with no wall-clock randomness. Returns the network and each
/// vertex's `(latitude, longitude)`.
fn pseudo_random_grid(width: u32, height: u32, seed: u64) -> (StreetNetwork, Vec<(f64, f64)>) {
    let mut lcg = seed;
    let mut rng = move || {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        lcg >> 33
    };
    let position = |i: u32, j: u32| (f64::from(i) * 130.0, f64::from(j) * 110.0);
    let index = |i: u32, j: u32| j * width + i;
    let both = MODE_WALK | MODE_BICYCLE;
    let mut edges: Vec<TestEdge> = Vec::new();
    let mut access_forward = Vec::new();
    let mut access_reverse = Vec::new();
    let mut flags = Vec::new();
    let mut add = |from: (f64, f64), to: (f64, f64), a: u32, b: u32, roll: (u64, u64)| {
        let planar = ((to.0 - from.0).powi(2) + (to.1 - from.1).powi(2)).sqrt();
        let length = planar * (1.0 + (roll.0 % 25) as f64 / 100.0);
        edges.push((a, b, length, straight(from, to)));
        match roll.0 % 5 {
            0 => {
                access_forward.push(both);
                access_reverse.push(MODE_WALK);
            }
            1 => {
                access_forward.push(MODE_WALK);
                access_reverse.push(both);
            }
            _ => {
                access_forward.push(both);
                access_reverse.push(both);
            }
        }
        flags.push(if roll.1.is_multiple_of(7) {
            FLAG_DISMOUNT
        } else {
            0
        });
    };
    for j in 0..height {
        for i in 0..width {
            if i + 1 < width {
                add(
                    position(i, j),
                    position(i + 1, j),
                    index(i, j),
                    index(i + 1, j),
                    (rng(), rng()),
                );
            }
            if j + 1 < height {
                add(
                    position(i, j),
                    position(i, j + 1),
                    index(i, j),
                    index(i, j + 1),
                    (rng(), rng()),
                );
            }
        }
    }
    let count = edges.len();
    let attrs = Attrs {
        highway: vec![0; count],
        surface: vec![0; count],
        smoothness: vec![0; count],
        flags,
        access_forward,
        access_reverse,
        facility_forward: vec![0; count],
        facility_reverse: vec![0; count],
    };
    let net = elevated_network(width * height, &edges, &attrs, |x, y| {
        ((x / 400.0).sin() * 12.0 + (y / 700.0).cos() * 7.0) as f32
    });
    let coordinates = (0..height)
        .flat_map(|j| (0..width).map(move |i| position(i, j)))
        .map(|(x, y)| {
            let (lon, lat) = lonlat(x, y);
            (lat, lon)
        })
        .collect();
    (net, coordinates)
}

/// Query points scattered off the grid vertices, snapped for `profile`.
fn grid_snaps(
    net: &StreetNetwork,
    coordinates: &[(f64, f64)],
    profile: &CompiledStreetProfile,
    step: usize,
) -> Vec<Option<Snap>> {
    coordinates
        .iter()
        .step_by(step)
        .map(|&(lat, lon)| net.snap_for_profile(lat + 0.0002, lon + 0.0004, 90.0, profile))
        .collect()
}

#[test]
fn astar_matches_dijkstra_cell_for_cell() {
    // The acceptance oracle: on a grid with one-ways, dismounts, slopes, and
    // jittered lengths, the goal-directed single route answers exactly what
    // the Dijkstra matrix row answers — for every origin×destination pair,
    // both profiles, and both a roomy and a tight cutoff (the tight one
    // leaves many cells unreachable).
    let (net, coordinates) = pseudo_random_grid(9, 7, 0xC0FFEE);
    for definition in [
        StreetProfileDefinition::bicycle(),
        StreetProfileDefinition::walk(),
    ] {
        let profile = net.compile_profile(&definition).unwrap();
        let snaps = grid_snaps(&net, &coordinates, &profile, 1);
        for max_seconds in [420.0, 60.0] {
            let mut beyond = 0usize;
            for origin in snaps.iter().flatten() {
                let row = net.directed_times_to_snaps(origin, &snaps, &profile, max_seconds);
                for (cell, target) in row.iter().zip(&snaps) {
                    let single = target
                        .as_ref()
                        .and_then(|to| net.directed_travel_time(origin, to, &profile, max_seconds));
                    assert_eq!(single, *cell, "{} single vs row", definition.name);
                    beyond += usize::from(cell.is_none());
                }
            }
            // The tight cutoff really exercises the unreachable answer.
            if max_seconds == 60.0 {
                assert!(beyond > 0);
            }
        }
    }
}

#[test]
fn astar_legs_match_dijkstra_legs() {
    // The reconstructing variant: time, metres, and geometry of the A* leg
    // are the Dijkstra row leg's, byte for byte, for both profiles — and the
    // tight cutoff exercises identical beyond-cutoff `None` legs.
    let (net, coordinates) = pseudo_random_grid(8, 6, 0xB1C7C1E);
    for definition in [
        StreetProfileDefinition::bicycle(),
        StreetProfileDefinition::walk(),
    ] {
        let profile = net.compile_profile(&definition).unwrap();
        let snaps = grid_snaps(&net, &coordinates, &profile, 2);
        let points: Vec<(f64, f64)> = coordinates
            .iter()
            .step_by(2)
            .map(|&(lat, lon)| (lat + 0.0002, lon + 0.0004))
            .collect();
        for max_seconds in [10_000.0, 60.0] {
            let (mut reached, mut beyond) = (0usize, 0usize);
            for (from_point, origin) in points.iter().zip(&snaps) {
                let Some(from) = origin else { continue };
                let targets: Vec<((f64, f64), Option<Snap>)> =
                    points.iter().copied().zip(snaps.iter().copied()).collect();
                let row =
                    net.directed_legs_to_snaps(*from_point, from, &targets, &profile, max_seconds);
                for ((to_point, target), row_leg) in targets.iter().zip(&row) {
                    let single = target.as_ref().and_then(|to| {
                        net.directed_leg(*from_point, from, *to_point, to, &profile, max_seconds)
                    });
                    assert_eq!(single, *row_leg, "{}", definition.name);
                    reached += usize::from(row_leg.is_some());
                    beyond += usize::from(row_leg.is_none());
                }
            }
            if max_seconds == 60.0 {
                assert!(beyond > 0);
            } else {
                assert!(reached > beyond);
            }
        }
    }
}

#[test]
fn meters_rows_match_the_reconstructed_legs() {
    // The meters rows are the reconstructed legs minus the geometry: the
    // forward row equals `directed_legs_to_snaps` in seconds and network
    // metres cell for cell, and the reverse column equals the forward
    // single legs the same way, its seconds also identical to the
    // times-only column. Both profiles; the tight cutoff exercises
    // identical beyond-cutoff `None` cells.
    let (net, coordinates) = pseudo_random_grid(8, 6, 0xB1C7C1E);
    for definition in [
        StreetProfileDefinition::bicycle(),
        StreetProfileDefinition::walk(),
    ] {
        let profile = net.compile_profile(&definition).unwrap();
        let snaps = grid_snaps(&net, &coordinates, &profile, 2);
        let points: Vec<(f64, f64)> = coordinates
            .iter()
            .step_by(2)
            .map(|&(lat, lon)| (lat + 0.0002, lon + 0.0004))
            .collect();
        for max_seconds in [10_000.0, 60.0] {
            for (from_point, origin) in points.iter().zip(&snaps) {
                let Some(from) = origin else { continue };
                let row = net.directed_meters_to_snaps(from, &snaps, &profile, max_seconds);
                let targets: Vec<((f64, f64), Option<Snap>)> =
                    points.iter().copied().zip(snaps.iter().copied()).collect();
                let legs =
                    net.directed_legs_to_snaps(*from_point, from, &targets, &profile, max_seconds);
                for (cell, leg) in row.iter().zip(&legs) {
                    match (cell, leg) {
                        (Some((seconds, meters)), Some(leg)) => {
                            assert_eq!(*seconds, leg.seconds, "{}", definition.name);
                            assert_eq!(*meters, leg.network_meters, "{}", definition.name);
                        }
                        (None, None) => {}
                        other => panic!("row/leg reachability diverged: {other:?}"),
                    }
                }
            }
            for (to_point, target) in points.iter().zip(&snaps) {
                let Some(to) = target else { continue };
                let column = net.directed_meters_from_snaps(&snaps, to, &profile, max_seconds);
                let times = net.directed_times_from_snaps(&snaps, to, &profile, max_seconds);
                for ((source_point, source), (cell, time)) in
                    points.iter().zip(&snaps).zip(column.iter().zip(&times))
                {
                    assert_eq!(cell.map(|(s, _)| s), *time, "{}", definition.name);
                    let Some(from) = source else { continue };
                    let leg =
                        net.directed_leg(*source_point, from, *to_point, to, &profile, max_seconds);
                    match (cell, leg) {
                        (Some((seconds, meters)), Some(leg)) => {
                            assert_eq!(*seconds, leg.seconds, "{}", definition.name);
                            // The same route, summed from the destination
                            // side instead of the seed's: bit equality is
                            // lost to the accumulation order, nothing else.
                            let tolerance = 1e-9 * leg.network_meters.max(1.0);
                            assert!(
                                (*meters - leg.network_meters).abs() <= tolerance,
                                "{}: {meters} vs {}",
                                definition.name,
                                leg.network_meters
                            );
                        }
                        (None, None) => {}
                        other => panic!("column/leg reachability diverged: {other:?}"),
                    }
                }
            }
        }
    }
}

#[test]
fn a_free_spatial_arc_disables_the_heuristic_but_not_correctness() {
    // A zero-length edge whose endpoints sit apart is free spatial movement:
    // no finite speed bounds it, so the goal bias switches off — and the
    // answers still match Dijkstra, with the reconstruction terminating.
    let edges: Vec<TestEdge> = vec![
        (0, 1, 100.0, straight((0.0, 0.0), (100.0, 0.0))),
        (1, 2, 0.0, straight((100.0, 0.0), (180.0, 0.0))),
        (2, 3, 100.0, straight((180.0, 0.0), (280.0, 0.0))),
    ];
    let net = multimodal_network(4, &edges, &plain_attrs(3)).unwrap();
    let bike = net
        .compile_profile(&StreetProfileDefinition::bicycle())
        .unwrap();
    let origin = lonlat(20.0, 0.0);
    let target = lonlat(260.0, 0.0);
    let from = net.snap(origin.1, origin.0, 40.0).unwrap();
    let to = net.snap(target.1, target.0, 40.0).unwrap();
    let single = net.directed_travel_time(&from, &to, &bike, 3600.0);
    let row = net.directed_times_to_snaps(&from, &[Some(to)], &bike, 3600.0)[0];
    assert_eq!(single, row);
    assert!(single.is_some());
    // The measured bound is infinite — the distance term really is off.
    assert_eq!(bike.effective_speed_cache().get(), Some(&f64::INFINITY));
    let leg = net
        .directed_leg(
            (origin.1, origin.0),
            &from,
            (target.1, target.0),
            &to,
            &bike,
            3600.0,
        )
        .unwrap();
    assert_eq!(leg.seconds, single.unwrap());
}

#[test]
fn unreachable_pairs_answer_none_from_both_searches() {
    // A two-vertex island disconnected from a main edge: routing between
    // them is unreachable, and both searches must say so.
    let edges: Vec<TestEdge> = vec![
        (0, 1, 200.0, straight((0.0, 0.0), (200.0, 0.0))),
        (2, 3, 200.0, straight((5000.0, 0.0), (5200.0, 0.0))),
    ];
    let net = multimodal_network(4, &edges, &plain_attrs(2)).unwrap();
    let bike = net
        .compile_profile(&StreetProfileDefinition::bicycle())
        .unwrap();
    let origin = lonlat(100.0, 0.0);
    let target = lonlat(5100.0, 0.0);
    let from = net.snap(origin.1, origin.0, 50.0).unwrap();
    let to = net.snap(target.1, target.0, 50.0).unwrap();
    assert_eq!(net.directed_travel_time(&from, &to, &bike, 3600.0), None);
    assert_eq!(
        net.directed_times_to_snaps(&from, &[Some(to)], &bike, 3600.0)[0],
        None
    );
    assert!(net
        .directed_leg(
            (origin.1, origin.0),
            &from,
            (target.1, target.0),
            &to,
            &bike,
            3600.0
        )
        .is_none());
}

#[test]
fn tied_routes_reconstruct_identically() {
    // An equal-cost diamond into a destination edge snapped at its middle:
    // the route ties between the two branches and can tie between the
    // destination edge's ends. Both searches must resolve every tie to the
    // same route — the order-independent predecessor rule, not pop order.
    let edges: Vec<TestEdge> = vec![
        // Origin edge west of the diamond.
        (0, 1, 100.0, straight((-100.0, 0.0), (0.0, 0.0))),
        // The two equal branches, north and south.
        (1, 2, 100.0, straight((0.0, 0.0), (50.0, 60.0))),
        (2, 3, 100.0, straight((50.0, 60.0), (100.0, 0.0))),
        (1, 4, 100.0, straight((0.0, 0.0), (50.0, -60.0))),
        (4, 3, 100.0, straight((50.0, -60.0), (100.0, 0.0))),
        // The destination edge east of it.
        (3, 5, 100.0, straight((100.0, 0.0), (200.0, 0.0))),
    ];
    let net = multimodal_network(6, &edges, &plain_attrs(6)).unwrap();
    let bike = net
        .compile_profile(&StreetProfileDefinition::bicycle())
        .unwrap();
    let origin = lonlat(-50.0, 0.0);
    let target = lonlat(150.0, 0.0);
    let from = net.snap(origin.1, origin.0, 30.0).unwrap();
    let to = net.snap(target.1, target.0, 30.0).unwrap();
    let single = net
        .directed_leg(
            (origin.1, origin.0),
            &from,
            (target.1, target.0),
            &to,
            &bike,
            600.0,
        )
        .unwrap();
    let row = net
        .directed_legs_to_snaps(
            (origin.1, origin.0),
            &from,
            &[((target.1, target.0), Some(to))],
            &bike,
            600.0,
        )
        .swap_remove(0)
        .unwrap();
    assert_eq!(single, row);
}

#[test]
fn astar_labels_fewer_vertices_on_a_corridor() {
    // A long east-west corridor with the target a few edges east of the
    // origin: the cutoff-bounded Dijkstra floods both directions, the
    // goal-directed search leans east and labels a fraction of that.
    let count = 200u32;
    let edges: Vec<TestEdge> = (0..count)
        .map(|i| {
            let from = (f64::from(i) * 100.0, 0.0);
            let to = (f64::from(i + 1) * 100.0, 0.0);
            (i, i + 1, 100.0, straight(from, to))
        })
        .collect();
    let net = multimodal_network(count + 1, &edges, &plain_attrs(edges.len())).unwrap();
    let bike = net
        .compile_profile(&StreetProfileDefinition::bicycle())
        .unwrap();
    let origin = lonlat(10_000.0, 0.0);
    let target = lonlat(10_450.0, 0.0);
    let from = net.snap(origin.1, origin.0, 50.0).unwrap();
    let to = net.snap(target.1, target.0, 50.0).unwrap();
    let (astar, dijkstra) = net.astar_versus_dijkstra_touched(&bike, &from, &to, 3600.0);
    assert!(
        astar * 3 < dijkstra,
        "A* labelled {astar} vertices to Dijkstra's {dijkstra}"
    );
    // And the answers still agree.
    assert_eq!(
        net.directed_travel_time(&from, &to, &bike, 3600.0),
        net.directed_times_to_snaps(&from, &[Some(to)], &bike, 3600.0)[0],
    );
}

#[test]
#[ignore = "wall-time benchmark; run manually with -- --ignored"]
fn astar_benchmark() {
    // Long routes across a large grid: prints per-query settled counts and
    // wall time for A* against the Dijkstra row, for manual comparison.
    let width = 200usize;
    let (net, coordinates) = pseudo_random_grid(width as u32, 160, 0x5EED);
    let bike = net
        .compile_profile(&StreetProfileDefinition::bicycle())
        .unwrap();
    let at = |i: usize, j: usize| {
        let (lat, lon) = coordinates[j * width + i];
        net.snap_for_profile(lat + 0.0002, lon + 0.0004, 90.0, &bike)
            .unwrap()
    };
    // Two route shapes at the production default cutoff (7200 s, under which
    // a bounded Dijkstra floods the whole extract): city hops a few km long,
    // and long crossings spanning most of the grid's width. Corner-to-corner
    // would put every vertex on a shortest path and prune nothing — the
    // known A* worst case, not a representative query.
    let city_hops: Vec<(Snap, Snap)> = (0..40)
        .map(|i| (at(60 + i, 40 + i * 2), at(90 + i, 52 + i * 2)))
        .collect();
    let crossings: Vec<(Snap, Snap)> = (0..40)
        .map(|i| (at(6, 30 + i * 2), at(width - 7, 34 + i * 2)))
        .collect();
    for (label, pairs) in [("city hops", &city_hops), ("crossings", &crossings)] {
        let start = std::time::Instant::now();
        let astar: Vec<Option<u32>> = pairs
            .iter()
            .map(|(from, to)| net.directed_travel_time(from, to, &bike, 7200.0))
            .collect();
        let astar_time = start.elapsed();
        let start = std::time::Instant::now();
        let dijkstra: Vec<Option<u32>> = pairs
            .iter()
            .map(|(from, to)| net.directed_times_to_snaps(from, &[Some(*to)], &bike, 7200.0)[0])
            .collect();
        let dijkstra_time = start.elapsed();
        assert_eq!(astar, dijkstra);
        let (mut astar_labels, mut dijkstra_labels) = (0usize, 0usize);
        for (from, to) in pairs.iter() {
            let (a, d) = net.astar_versus_dijkstra_touched(&bike, from, to, 7200.0);
            astar_labels += a;
            dijkstra_labels += d;
        }
        println!(
            "{} {label}: A* {astar_time:?} vs Dijkstra {dijkstra_time:?}; \
             labels {astar_labels} vs {dijkstra_labels}",
            pairs.len()
        );
    }
}

// --- Reverse (egress) directed search ----------------------------------------

#[test]
fn reverse_rows_match_forward_singles_cell_for_cell() {
    // The duality oracle: one reverse search from a destination answers, for
    // every source, exactly what the forward single route answers — one-ways,
    // dismounts, slopes, cutoffs, and unreachable cells included.
    let (net, coordinates) = pseudo_random_grid(9, 7, 0xE64E55);
    for definition in [
        StreetProfileDefinition::bicycle(),
        StreetProfileDefinition::walk(),
    ] {
        let profile = net.compile_profile(&definition).unwrap();
        let snaps = grid_snaps(&net, &coordinates, &profile, 1);
        for max_seconds in [10_000.0, 60.0] {
            for target in snaps.iter().flatten() {
                let column = net.directed_times_from_snaps(&snaps, target, &profile, max_seconds);
                for (cell, source) in column.iter().zip(&snaps) {
                    let single = source.as_ref().and_then(|from| {
                        net.directed_travel_time(from, target, &profile, max_seconds)
                    });
                    assert_eq!(single, *cell, "{} column vs single", definition.name);
                }
            }
        }
    }
}

#[test]
fn a_one_way_splits_access_from_egress() {
    // A bicycle one-way east: reaching the east end is quick, coming back is
    // not — the reverse row must see the asymmetry the forward row sees.
    let edges: Vec<TestEdge> = vec![(0, 1, 200.0, straight((0.0, 0.0), (200.0, 0.0)))];
    let mut attrs = plain_attrs(1);
    attrs.access_reverse = vec![MODE_WALK];
    let net = multimodal_network(2, &edges, &attrs).unwrap();
    let bike = net
        .compile_profile(&StreetProfileDefinition::bicycle())
        .unwrap();
    let west = lonlat(10.0, 0.0);
    let east = lonlat(190.0, 0.0);
    let from = net.snap(west.1, west.0, 50.0).unwrap();
    let to = net.snap(east.1, east.0, 50.0).unwrap();
    let eastbound = net.directed_times_from_snaps(&[Some(from)], &to, &bike, 3600.0)[0];
    let westbound = net.directed_times_from_snaps(&[Some(to)], &from, &bike, 3600.0)[0];
    assert!(eastbound.is_some());
    // The way back has no permitted bicycle arc at all.
    assert_eq!(westbound, None);
}

#[test]
fn mode_bit_snapping_respects_each_bit() {
    // A walk-only edge on the query line and a bicycle-only edge north of
    // it: each mode bit must snap to its own permitted edge, both ways
    // round — the stop-link builder's `foot=no` case.
    let edges: Vec<TestEdge> = vec![
        (0, 1, 200.0, straight((0.0, 0.0), (200.0, 0.0))),
        (2, 3, 200.0, straight((0.0, 60.0), (200.0, 60.0))),
    ];
    let mut attrs = plain_attrs(2);
    attrs.access_forward = vec![MODE_WALK, MODE_BICYCLE];
    attrs.access_reverse = vec![MODE_WALK, MODE_BICYCLE];
    let net = multimodal_network(4, &edges, &attrs).unwrap();
    let (lon, lat) = lonlat(100.0, 10.0);
    let walk = net.snap_for_mode_bit(lat, lon, 300.0, MODE_WALK).unwrap();
    let bike = net
        .snap_for_mode_bit(lat, lon, 300.0, MODE_BICYCLE)
        .unwrap();
    assert_ne!(walk.edge, bike.edge);
    // The walk edge is the nearer one; the bicycle snap crosses to the far
    // edge rather than landing where it may not ride.
    assert!(walk.connector < bike.connector);
    // No e-scooter permission anywhere: no snap at all.
    assert_eq!(net.snap_for_mode_bit(lat, lon, 300.0, MODE_E_SCOOTER), None);
}

// --- Directed profile-aware search -------------------------------------------

fn directed_slot(net: &StreetNetwork, from: u32, to: u32) -> usize {
    let offsets = net.arrays().adjacency_offsets();
    let targets = net.arrays().adj_targets();
    (offsets[from as usize] as usize..offsets[from as usize + 1] as usize)
        .find(|&slot| targets[slot] == to)
        .expect("no arc between the vertices")
}

/// Uniform attributes permitting walk+bicycle everywhere, then clearing the
/// bicycle bit on every directed arc `from → to` (all parallel edges too).
fn bike_access_forbidding(net: &StreetNetwork, forbidden: &[(u32, u32)]) -> StreetAttributes {
    let mut attributes = uniform_attributes(net, MODE_WALK | MODE_BICYCLE, 0);
    let offsets = net.arrays().adjacency_offsets();
    let targets = net.arrays().adj_targets();
    for &(from, to) in forbidden {
        let start = offsets[from as usize] as usize;
        let end = offsets[from as usize + 1] as usize;
        for (offset, &target) in targets[start..end].iter().enumerate() {
            if target == to {
                attributes.adj_access[start + offset] &= !MODE_BICYCLE;
            }
        }
    }
    attributes
}

fn hand_dijkstra(net: &StreetNetwork, arc_millis: &[u32], source: u32) -> Vec<u64> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let offsets = net.arrays().adjacency_offsets();
    let targets = net.arrays().adj_targets();
    let mut dist = vec![u64::MAX; net.vertex_count() as usize];
    let mut heap = BinaryHeap::new();
    dist[source as usize] = 0;
    heap.push(Reverse((0u64, source)));
    while let Some(Reverse((d, v))) = heap.pop() {
        if d > dist[v as usize] {
            continue;
        }
        for slot in offsets[v as usize] as usize..offsets[v as usize + 1] as usize {
            if arc_millis[slot] == u32::MAX {
                continue;
            }
            let next = d + u64::from(arc_millis[slot]);
            let target = targets[slot] as usize;
            if next < dist[target] {
                dist[target] = next;
                heap.push(Reverse((next, targets[slot])));
            }
        }
    }
    dist
}

#[test]
fn directed_dijkstra_matches_a_plain_dijkstra() {
    let mut net = triangle();
    net.install_street_attributes(uniform_attributes(&net, MODE_WALK | MODE_BICYCLE, 0))
        .unwrap();
    let bike = net
        .compile_profile(&StreetProfileDefinition::bicycle())
        .unwrap();
    let expected = hand_dijkstra(&net, bike.arc_millis(), 0);
    let got = net.directed_distances(&bike, &[(0, 0)], u64::MAX);
    assert_eq!(got, expected);
}

#[test]
fn directed_travel_time_is_the_on_edge_plus_arc_time() {
    let mut net = triangle();
    net.install_street_attributes(uniform_attributes(&net, MODE_WALK | MODE_BICYCLE, 0))
        .unwrap();
    let bike = net
        .compile_profile(&StreetProfileDefinition::bicycle())
        .unwrap();
    // A snap mid-edge on the first edge (0,1) to a snap mid-edge on the last
    // edge (2,3), no connectors. The route leaves toward vertex 1, crosses the
    // middle edge (1,2), and arrives on the last edge from vertex 2.
    let from = Snap {
        edge: directed_edge(&net, 0, 1),
        fraction: 0.25,
        connector: 0.0,
    };
    let to = Snap {
        edge: directed_edge(&net, 2, 3),
        fraction: 0.5,
        connector: 0.0,
    };
    let arc = |a, b| bike.arc_millis()[directed_slot(&net, a, b)];
    let leave = (f64::from(arc(0, 1)) * 0.75).ceil() as u64;
    let cross = u64::from(arc(1, 2));
    let arrive = (f64::from(arc(2, 3)) * 0.5).ceil() as u64;
    let expected = seconds((leave + cross + arrive) as f64 / 1000.0);
    assert_eq!(
        net.directed_travel_time(&from, &to, &bike, 1e9),
        Some(expected)
    );
    // The undirected graph is symmetric, so the reverse trip costs the same.
    assert_eq!(
        net.directed_travel_time(&to, &from, &bike, 1e9),
        net.directed_travel_time(&from, &to, &bike, 1e9)
    );
}

fn directed_edge(net: &StreetNetwork, from: u32, to: u32) -> u32 {
    net.arrays().adj_edges()[directed_slot(net, from, to)]
}

#[test]
fn directed_search_respects_one_way_arcs() {
    // Make the middle edge (1,2) one-way in the 1->2 direction only. A trip
    // that must cross it forward still works; the reverse trip has no path.
    let mut net = triangle();
    net.install_street_attributes(bike_access_forbidding(&net, &[(2, 1)]))
        .unwrap();
    let bike = net
        .compile_profile(&StreetProfileDefinition::bicycle())
        .unwrap();
    let on_first = Snap {
        edge: directed_edge(&net, 0, 1),
        fraction: 0.5,
        connector: 0.0,
    };
    let on_last = Snap {
        edge: directed_edge(&net, 2, 3),
        fraction: 0.5,
        connector: 0.0,
    };
    // first -> last crosses (1,2) forward: reachable.
    assert!(net
        .directed_travel_time(&on_first, &on_last, &bike, 1e9)
        .is_some());
    // last -> first would need (2,1), which the bicycle may not use: no path.
    assert_eq!(
        net.directed_travel_time(&on_last, &on_first, &bike, 1e9),
        None
    );
}

#[test]
fn directed_same_edge_direct_and_blocked() {
    let mut net = triangle();
    // Forbid the reverse (1,0) arc of the first edge.
    net.install_street_attributes(bike_access_forbidding(&net, &[(1, 0)]))
        .unwrap();
    let bike = net
        .compile_profile(&StreetProfileDefinition::bicycle())
        .unwrap();
    let edge = directed_edge(&net, 0, 1);
    let near = Snap {
        edge,
        fraction: 0.25,
        connector: 0.0,
    };
    let far = Snap {
        edge,
        fraction: 0.75,
        connector: 0.0,
    };
    // near -> far runs forward (0->1) directly along the edge.
    let arc = bike.arc_millis()[directed_slot(&net, 0, 1)];
    let direct = seconds((f64::from(arc) * 0.5).ceil() / 1000.0);
    assert_eq!(
        net.directed_travel_time(&near, &far, &bike, 1e9),
        Some(direct)
    );
    // far -> near would run reverse (1->0) directly, which is forbidden, and a
    // single edge offers no detour: no path.
    assert_eq!(net.directed_travel_time(&far, &near, &bike, 1e9), None);
}

/// A cycle 0->1->2->0 with a longer parallel edge between 0 and 1, so shortest
/// paths have detours and a parallel-edge choice.
fn loop_network() -> StreetNetwork {
    network(
        3,
        0,
        &[
            (0, 1, 100.0, straight((0.0, 0.0), (100.0, 0.0))),
            (1, 2, 120.0, straight((100.0, 0.0), (100.0, 120.0))),
            (2, 0, 140.0, straight((100.0, 120.0), (0.0, 0.0))),
            (0, 1, 300.0, vec![(0.0, 0.0), (50.0, -80.0), (100.0, 0.0)]),
        ],
        vec![],
    )
    .unwrap()
}

fn hand_dijkstra_bounded(
    net: &StreetNetwork,
    arc_millis: &[u32],
    source: u32,
    cutoff: u64,
) -> Vec<u64> {
    hand_dijkstra(net, arc_millis, source)
        .into_iter()
        .map(|d| if d <= cutoff { d } else { u64::MAX })
        .collect()
}

#[test]
fn directed_search_matches_oracle_over_sources_cutoffs_and_parallels() {
    let mut net = loop_network();
    net.install_street_attributes(uniform_attributes(&net, MODE_WALK | MODE_BICYCLE, 0))
        .unwrap();
    let bike = net
        .compile_profile(&StreetProfileDefinition::bicycle())
        .unwrap();
    for source in 0..net.vertex_count() {
        assert_eq!(
            net.directed_distances(&bike, &[(source, 0)], u64::MAX),
            hand_dijkstra(&net, bike.arc_millis(), source),
        );
        // A bound that reaches some but not all vertices.
        let cutoff = 30_000;
        assert_eq!(
            net.directed_distances(&bike, &[(source, 0)], cutoff),
            hand_dijkstra_bounded(&net, bike.arc_millis(), source, cutoff),
        );
    }
    // The parallel edge choice: 0 -> 1 takes the shorter of the two edges.
    let short = bike.arc_millis()[directed_slot(&net, 0, 1)];
    assert_eq!(
        net.directed_distances(&bike, &[(0, 0)], u64::MAX)[1],
        u64::from(short)
    );
}

#[test]
fn directed_search_detours_around_forbidden_arcs() {
    // Forbid every 0 -> 1 arc; vertex 1 is then reachable from 0 only around
    // the cycle 0 -> 2 -> 1, and the directed search finds that detour.
    let mut net = loop_network();
    net.install_street_attributes(bike_access_forbidding(&net, &[(0, 1)]))
        .unwrap();
    let bike = net
        .compile_profile(&StreetProfileDefinition::bicycle())
        .unwrap();
    let detour = u64::from(bike.arc_millis()[directed_slot(&net, 0, 2)])
        + u64::from(bike.arc_millis()[directed_slot(&net, 2, 1)]);
    assert_eq!(
        net.directed_distances(&bike, &[(0, 0)], u64::MAX)[1],
        detour
    );
    // The oracle agrees over the forbidden graph.
    assert_eq!(
        net.directed_distances(&bike, &[(0, 0)], u64::MAX),
        hand_dijkstra(&net, bike.arc_millis(), 0),
    );
}

#[test]
fn directed_same_edge_falls_back_to_a_detour() {
    // Two snaps on the first 0->1 edge, with its direct reverse (1->0) forbidden.
    // far -> near cannot run back along the edge but detours 1 -> 2 -> 0.
    let mut net = loop_network();
    let edge = directed_edge(&net, 0, 1);
    net.install_street_attributes(bike_access_forbidding(&net, &[(1, 0)]))
        .unwrap();
    let bike = net
        .compile_profile(&StreetProfileDefinition::bicycle())
        .unwrap();
    let near = Snap {
        edge,
        fraction: 0.25,
        connector: 0.0,
    };
    let far = Snap {
        edge,
        fraction: 0.75,
        connector: 0.0,
    };
    // near -> far is the direct forward run.
    assert!(net.directed_travel_time(&near, &far, &bike, 1e9).is_some());
    // far -> near has no direct reverse but a cycle detour exists, so it routes.
    assert!(net.directed_travel_time(&far, &near, &bike, 1e9).is_some());
}

#[test]
fn directed_snap_at_a_vertex_seeds_it_without_a_reverse_arc() {
    // A snap exactly at fraction 0 sits on the `from` vertex; it must seed that
    // vertex even when the reverse arc is forbidden (no arc traversal needed).
    let mut net = loop_network();
    let edge = directed_edge(&net, 0, 1);
    net.install_street_attributes(bike_access_forbidding(&net, &[(1, 0)]))
        .unwrap();
    let bike = net
        .compile_profile(&StreetProfileDefinition::bicycle())
        .unwrap();
    // Origin at vertex 0 (fraction 0 on the 0->1 edge), destination mid-edge on
    // (1,2): reachable forward via vertex 1 despite the forbidden 1->0 arc.
    let origin = Snap {
        edge,
        fraction: 0.0,
        connector: 0.0,
    };
    let dest = Snap {
        edge: directed_edge(&net, 1, 2),
        fraction: 0.5,
        connector: 0.0,
    };
    assert!(net
        .directed_travel_time(&origin, &dest, &bike, 1e9)
        .is_some());
}

fn self_loop_network() -> (StreetNetwork, u32, [usize; 2]) {
    // A self-loop at vertex 1 plus an edge to vertex 0. Returns the network,
    // the loop edge id, and the loop's two distinct adjacency slots.
    let net = network(
        2,
        0,
        &[
            (0, 1, 100.0, straight((0.0, 0.0), (100.0, 0.0))),
            (1, 1, 60.0, vec![(100.0, 0.0), (140.0, 40.0), (100.0, 0.0)]),
        ],
        vec![],
    )
    .unwrap();
    let loop_edge = (0..net.edge_count())
        .find(|&e| net.edge_endpoints(e) == (1, 1))
        .unwrap();
    let edges = net.arrays().adj_edges();
    let slots: Vec<usize> = (net.arrays().adjacency_offsets()[1] as usize
        ..net.arrays().adjacency_offsets()[2] as usize)
        .filter(|&slot| edges[slot] == loop_edge)
        .collect();
    assert_eq!(slots.len(), 2, "a self-loop has two distinct arcs");
    let pair = [slots[0], slots[1]];
    (net, loop_edge, pair)
}

#[test]
fn directed_self_loop_arcs_are_gated_independently() {
    // Forbidding exactly one of the loop's two arcs must leave the other
    // usable, so a mid-loop snap still routes out to vertex 0's edge; forbidding
    // both makes it unreachable. An implementation that merged the two slots
    // would fail the one-forbidden case.
    let route = |forbidden: &[usize]| {
        let (mut net, loop_edge, _) = self_loop_network();
        let mut attributes = uniform_attributes(&net, MODE_WALK | MODE_BICYCLE, 0);
        for &slot in forbidden {
            attributes.adj_access[slot] &= !MODE_BICYCLE;
        }
        net.install_street_attributes(attributes).unwrap();
        let bike = net
            .compile_profile(&StreetProfileDefinition::bicycle())
            .unwrap();
        let on_loop = Snap {
            edge: loop_edge,
            fraction: 0.5,
            connector: 0.0,
        };
        let on_first = Snap {
            edge: directed_edge(&net, 0, 1),
            fraction: 0.5,
            connector: 0.0,
        };
        net.directed_travel_time(&on_loop, &on_first, &bike, 1e9)
    };
    let (_, _, [slot_a, slot_b]) = self_loop_network();
    // Leaving the mid-loop snap uses the surviving arc (7.5 s over half the
    // 60 m loop at 4 m/s) then half the 100 m edge to vertex 0's side (12.5 s).
    assert_eq!(route(&[slot_a]), Some(20));
    assert_eq!(route(&[slot_b]), Some(20));
    assert_eq!(route(&[slot_a, slot_b]), None);
}

#[test]
fn directed_millisecond_accumulation_never_overflows() {
    // A colossal connector must not panic or wrap into a bogus short route; it
    // simply exceeds any sane cutoff and yields no route.
    let mut net = triangle();
    net.install_street_attributes(uniform_attributes(&net, MODE_WALK | MODE_BICYCLE, 0))
        .unwrap();
    let bike = net
        .compile_profile(&StreetProfileDefinition::bicycle())
        .unwrap();
    let huge = Snap {
        edge: directed_edge(&net, 0, 1),
        fraction: 0.5,
        connector: 1e18,
    };
    let normal = Snap {
        edge: directed_edge(&net, 2, 3),
        fraction: 0.5,
        connector: 0.0,
    };
    assert_eq!(
        net.directed_travel_time(&huge, &normal, &bike, 3600.0),
        None
    );
}

#[test]
fn directed_travel_time_respects_a_fractional_cutoff() {
    // A snap at each end of the 400 m first edge routes directly forward: exactly
    // 100 s (400 m / 4 m/s). The cutoff is a floor, so 99.999 s rejects it while
    // 100 s accepts it — a route just beyond the requested duration is excluded.
    let mut net = triangle();
    net.install_street_attributes(uniform_attributes(&net, MODE_WALK | MODE_BICYCLE, 0))
        .unwrap();
    let bike = net
        .compile_profile(&StreetProfileDefinition::bicycle())
        .unwrap();
    let edge = directed_edge(&net, 0, 1);
    let start = Snap {
        edge,
        fraction: 0.0,
        connector: 0.0,
    };
    let end = Snap {
        edge,
        fraction: 1.0,
        connector: 0.0,
    };
    assert_eq!(
        net.directed_travel_time(&start, &end, &bike, 100.0),
        Some(100)
    );
    assert_eq!(net.directed_travel_time(&start, &end, &bike, 99.999), None);
}

// ---- Multimodal construction (`new_multimodal`) ----

/// Per-edge attributes for a multimodal test build, one entry per input edge.
struct Attrs {
    highway: Vec<u8>,
    surface: Vec<u8>,
    smoothness: Vec<u8>,
    flags: Vec<u16>,
    access_forward: Vec<u8>,
    access_reverse: Vec<u8>,
    facility_forward: Vec<u8>,
    facility_reverse: Vec<u8>,
}

/// Builds a network through `new_multimodal`, laying out geometry exactly as
/// [`network`] does.
fn multimodal_network(
    vertex_count: u32,
    edges: &[TestEdge],
    attrs: &Attrs,
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
    StreetNetwork::new_multimodal(
        vertex_count,
        0,
        &flat,
        &offsets,
        &longitudes,
        &latitudes,
        vec![],
        EdgeAttributes {
            highway: &attrs.highway,
            surface: &attrs.surface,
            smoothness: &attrs.smoothness,
            flags: &attrs.flags,
            access_forward: &attrs.access_forward,
            access_reverse: &attrs.access_reverse,
            facility_forward: &attrs.facility_forward,
            facility_reverse: &attrs.facility_reverse,
            car: None,
        },
        None,
    )
}

#[test]
fn new_multimodal_aligns_attributes_through_the_reorder() {
    // Five colinear edges given out of spatial order so the Hilbert reorder
    // permutes them. Each input edge carries its index as a unique highway code
    // and direction-tagged facility bits (`2·i` forward, `2·i + 1` reverse), so
    // every internal edge and directed arc can be traced back to the input edge
    // and direction it came from.
    let edges: Vec<TestEdge> = vec![
        (2, 3, 100.0, straight((200.0, 0.0), (300.0, 0.0))),
        (0, 1, 100.0, straight((0.0, 0.0), (100.0, 0.0))),
        (4, 5, 100.0, straight((400.0, 0.0), (500.0, 0.0))),
        (1, 2, 100.0, straight((100.0, 0.0), (200.0, 0.0))),
        (3, 4, 100.0, straight((300.0, 0.0), (400.0, 0.0))),
    ];
    let n = edges.len();
    let attrs = Attrs {
        highway: (0..n as u8).collect(),
        surface: (0..n as u8).collect(),
        smoothness: (0..n as u8).collect(),
        flags: (0..n).map(|i| (i as u16) * 7 + 1).collect(),
        access_forward: vec![MODE_WALK | MODE_BICYCLE | MODE_E_SCOOTER; n],
        access_reverse: vec![MODE_WALK; n],
        facility_forward: (0..n as u8).map(|i| 2 * i).collect(),
        facility_reverse: (0..n as u8).map(|i| 2 * i + 1).collect(),
    };
    let net = multimodal_network(6, &edges, &attrs).unwrap();
    let built = net.street_attributes().unwrap();

    // Name each internal edge by its geometry, not by any attribute array: its
    // stored first coordinate is the verbatim (quantised) first coordinate of
    // the input edge it came from, and the reorder is defined by that geometry.
    // So `source_edge` is an independent oracle — a permutation applied
    // consistently but wrongly to every attribute array would still fail below.
    let lons = net.arrays().lons();
    let lats = net.arrays().lats();
    let offsets = net.arrays().coordinate_offsets();
    let first_keys: Vec<(i32, i32)> = edges
        .iter()
        .map(|(_, _, _, path)| {
            let (lon, lat) = lonlat(path[0].0, path[0].1);
            (quantize(lon), quantize(lat))
        })
        .collect();
    let source_edge = |internal: usize| -> usize {
        let start = offsets[internal] as usize;
        first_keys
            .iter()
            .position(|&key| key == (lons[start], lats[start]))
            .unwrap()
    };

    // The reorder is non-trivial: at least one internal edge is not its input.
    assert!((0..n).any(|e| source_edge(e) != e));

    // Per-edge codes travel with their edge: highway (verified against the
    // geometry-named source, not assumed), surface, smoothness, and flags.
    for e in 0..n {
        let i = source_edge(e);
        assert_eq!(built.edge_highway[e] as usize, i);
        assert_eq!(built.edge_surface[e] as usize, i);
        assert_eq!(built.edge_smoothness[e] as usize, i);
        assert_eq!(built.edge_flags[e], (i as u16) * 7 + 1);
    }

    // Per-arc permissions land on the correct directed arc: the arc toward an
    // edge's `to` endpoint carries the forward values, toward `from` the reverse.
    let targets = net.arrays().adj_targets();
    let arc_edges = net.arrays().adj_edges();
    for slot in 0..2 * n {
        let e = arc_edges[slot] as usize;
        let i = source_edge(e);
        let (from, to) = net.edge_endpoints(e as u32);
        assert_ne!(from, to);
        let forward = targets[slot] == to;
        assert_eq!(
            built.adj_facility[slot] as usize,
            if forward { 2 * i } else { 2 * i + 1 }
        );
        assert_eq!(
            built.adj_access[slot],
            if forward {
                MODE_WALK | MODE_BICYCLE | MODE_E_SCOOTER
            } else {
                MODE_WALK
            }
        );
    }
}

/// Per-edge car attributes for a car test build, one entry per input edge.
struct CarAttrs {
    speed_forward: Vec<f32>,
    speed_reverse: Vec<f32>,
    junction_forward: Vec<u8>,
    junction_reverse: Vec<u8>,
}

/// Builds a network through `new_multimodal` with the car group installed.
fn car_network(
    vertex_count: u32,
    edges: &[TestEdge],
    attrs: &Attrs,
    car: &CarAttrs,
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
    StreetNetwork::new_multimodal(
        vertex_count,
        0,
        &flat,
        &offsets,
        &longitudes,
        &latitudes,
        vec![],
        EdgeAttributes {
            highway: &attrs.highway,
            surface: &attrs.surface,
            smoothness: &attrs.smoothness,
            flags: &attrs.flags,
            access_forward: &attrs.access_forward,
            access_reverse: &attrs.access_reverse,
            facility_forward: &attrs.facility_forward,
            facility_reverse: &attrs.facility_reverse,
            car: Some(CarEdgeAttributes {
                speed_forward: &car.speed_forward,
                speed_reverse: &car.speed_reverse,
                junction_forward: &car.junction_forward,
                junction_reverse: &car.junction_reverse,
            }),
        },
        None,
    )
}

#[test]
fn new_multimodal_aligns_car_attributes_through_the_reorder() {
    // The same out-of-spatial-order edges as the attribute alignment test;
    // each input edge carries direction-tagged speeds and junction classes,
    // so every directed arc traces back to its input edge and direction.
    let edges: Vec<TestEdge> = vec![
        (2, 3, 100.0, straight((200.0, 0.0), (300.0, 0.0))),
        (0, 1, 100.0, straight((0.0, 0.0), (100.0, 0.0))),
        (4, 5, 100.0, straight((400.0, 0.0), (500.0, 0.0))),
        (1, 2, 100.0, straight((100.0, 0.0), (200.0, 0.0))),
        (3, 4, 100.0, straight((300.0, 0.0), (400.0, 0.0))),
    ];
    let n = edges.len();
    let car = CarAttrs {
        speed_forward: (0..n).map(|i| 10.0 * i as f32 + 1.0).collect(),
        speed_reverse: (0..n).map(|i| 10.0 * i as f32 + 2.0).collect(),
        junction_forward: (0..n as u8).map(|i| i % 5).collect(),
        junction_reverse: (0..n as u8).map(|i| (i + 1) % 5).collect(),
    };
    let net = car_network(6, &edges, &plain_attrs(n), &car).unwrap();
    let built = net.car_attributes().unwrap();

    // The geometry oracle from the attribute alignment test: an internal
    // edge's first stored coordinate identifies the input edge it came from.
    let lons = net.arrays().lons();
    let lats = net.arrays().lats();
    let offsets = net.arrays().coordinate_offsets();
    let first_keys: Vec<(i32, i32)> = edges
        .iter()
        .map(|(_, _, _, path)| {
            let (lon, lat) = lonlat(path[0].0, path[0].1);
            (quantize(lon), quantize(lat))
        })
        .collect();
    let source_edge = |internal: usize| -> usize {
        let start = offsets[internal] as usize;
        first_keys
            .iter()
            .position(|&key| key == (lons[start], lats[start]))
            .unwrap()
    };
    assert!((0..n).any(|e| source_edge(e) != e));

    let targets = net.arrays().adj_targets();
    let arc_edges = net.arrays().adj_edges();
    for slot in 0..2 * n {
        let e = arc_edges[slot] as usize;
        let i = source_edge(e);
        let (from, to) = net.edge_endpoints(e as u32);
        assert_ne!(from, to);
        let forward = targets[slot] == to;
        let (speed, junction) = if forward {
            (car.speed_forward[i], car.junction_forward[i])
        } else {
            (car.speed_reverse[i], car.junction_reverse[i])
        };
        assert_eq!(built.adj_car_speed[slot], speed);
        assert_eq!(built.adj_junction[slot], junction);
    }
}

#[test]
fn car_attributes_round_trip_through_parts() {
    let edges: Vec<TestEdge> = vec![
        (0, 1, 100.0, straight((0.0, 0.0), (100.0, 0.0))),
        (1, 2, 100.0, straight((100.0, 0.0), (200.0, 0.0))),
    ];
    let n = edges.len();
    let car = CarAttrs {
        speed_forward: vec![50.0; n],
        speed_reverse: vec![30.0; n],
        junction_forward: vec![3; n],
        junction_reverse: vec![1; n],
    };
    let net = car_network(3, &edges, &plain_attrs(n), &car).unwrap();
    let parts = net.to_parts();
    assert!(parts.car.is_some());
    let rebuilt = StreetNetwork::from_parts(parts).unwrap();
    assert_eq!(rebuilt.car_attributes(), net.car_attributes());
    assert_eq!(rebuilt.to_parts(), net.to_parts());
}

#[test]
fn from_parts_rejects_malformed_car_groups() {
    let edges: Vec<TestEdge> = vec![(0, 1, 100.0, straight((0.0, 0.0), (100.0, 0.0)))];
    let car = CarAttrs {
        speed_forward: vec![50.0],
        speed_reverse: vec![30.0],
        junction_forward: vec![0],
        junction_reverse: vec![0],
    };
    let net = car_network(2, &edges, &plain_attrs(1), &car).unwrap();
    let corrupt = |mutate: &dyn Fn(&mut StreetNetworkParts)| {
        let mut parts = net.to_parts();
        mutate(&mut parts);
        StreetNetwork::from_parts(parts).unwrap_err()
    };
    // A car group without the attribute group is refused.
    assert_eq!(
        corrupt(&|parts| parts.attributes = None),
        StreetError::InvalidAttributes
    );
    // Truncated arrays, non-positive and non-finite speeds, and junction
    // classes outside the vocabulary are refused.
    let cases: Vec<&dyn Fn(&mut StreetNetworkParts)> = vec![
        &|parts| {
            parts.car.as_mut().unwrap().adj_car_speed.pop();
        },
        &|parts| {
            parts.car.as_mut().unwrap().adj_junction.pop();
        },
        &|parts| parts.car.as_mut().unwrap().adj_car_speed[0] = 0.0,
        &|parts| parts.car.as_mut().unwrap().adj_car_speed[0] = f32::NAN,
        &|parts| parts.car.as_mut().unwrap().adj_car_speed[0] = f32::INFINITY,
        &|parts| parts.car.as_mut().unwrap().adj_junction[0] = JUNCTION_CLASS_COUNT as u8,
    ];
    for mutate in cases {
        assert_eq!(corrupt(mutate), StreetError::InvalidAttributes);
    }
}

#[test]
fn new_multimodal_rejects_invalid_car_inputs() {
    let edges: Vec<TestEdge> = vec![(0, 1, 100.0, straight((0.0, 0.0), (100.0, 0.0)))];
    let build = |car: CarAttrs| car_network(2, &edges, &plain_attrs(1), &car).unwrap_err();
    // A zero speed, a junction class outside the vocabulary, and a length
    // mismatch are each refused at construction.
    assert_eq!(
        build(CarAttrs {
            speed_forward: vec![0.0],
            speed_reverse: vec![30.0],
            junction_forward: vec![0],
            junction_reverse: vec![0],
        }),
        StreetError::InvalidAttributes
    );
    assert_eq!(
        build(CarAttrs {
            speed_forward: vec![50.0],
            speed_reverse: vec![30.0],
            junction_forward: vec![JUNCTION_CLASS_COUNT as u8],
            junction_reverse: vec![0],
        }),
        StreetError::InvalidAttributes
    );
    assert_eq!(
        build(CarAttrs {
            speed_forward: vec![50.0, 60.0],
            speed_reverse: vec![30.0],
            junction_forward: vec![0],
            junction_reverse: vec![0],
        }),
        StreetError::InvalidAttributes
    );
}

// ---- The car profile and its delay model ----

/// The default highway-code → group mapping (motorway/trunk/primary → 1–2,
/// secondary/tertiary → 3, everything else → 4–6).
fn default_groups() -> Vec<u8> {
    let mut groups = vec![2u8; HIGHWAY_CODE_COUNT];
    for code in [1, 3, 5] {
        groups[code] = 0;
    }
    for code in [7, 9] {
        groups[code] = 1;
    }
    groups
}

/// A delay model with distinct per-group values so a group swap cannot pass.
fn test_delay_model() -> CarCostModel {
    CarCostModel {
        group_seconds: [12.0, 8.0, 6.0],
        groups: default_groups(),
        ramp_share_high: 0.75,
        ramp_share_low: 0.5,
        ramp_multiplier: 1.5,
        congestion_multiplier: 1.2,
    }
}

/// The compiled cost of the directed arc `from → to`, from the CSR.
fn arc_between(net: &StreetNetwork, profile: &CompiledStreetProfile, from: u32, to: u32) -> u32 {
    let adjacency = net.arrays().adjacency_offsets();
    let targets = net.arrays().adj_targets();
    let slot = (adjacency[from as usize] as usize..adjacency[from as usize + 1] as usize)
        .find(|&slot| targets[slot] == to)
        .unwrap_or_else(|| panic!("no arc {from}->{to}"));
    profile.arc_millis()[slot]
}

/// A chain of `edges.len()` 100 m elements with per-edge highway codes,
/// speeds, junction head classes, and flags — the car oracle fixture.
/// `junctions[i]` is `(at_from, at_to)` for edge `i`.
#[allow(clippy::type_complexity)]
fn car_oracle_network(
    edges: &[(u32, u32)],
    codes: &[u8],
    speeds: &[f32],
    junctions: &[(u8, u8)],
    flags: &[u16],
) -> StreetNetwork {
    let n = edges.len();
    let test_edges: Vec<TestEdge> = edges
        .iter()
        .enumerate()
        .map(|(i, &(from, to))| {
            let x = 100.0 * i as f64;
            (from, to, 100.0, straight((x, 0.0), (x + 100.0, 0.0)))
        })
        .collect();
    let mut attrs = plain_attrs(n);
    attrs.highway = codes.to_vec();
    for mask in attrs
        .access_forward
        .iter_mut()
        .chain(attrs.access_reverse.iter_mut())
    {
        *mask |= MODE_CAR;
    }
    let vertex_count = edges.iter().map(|&(a, b)| a.max(b)).max().unwrap() + 1;
    let car = CarAttrs {
        speed_forward: speeds.to_vec(),
        speed_reverse: speeds.to_vec(),
        junction_forward: junctions.iter().map(|&(_, at_to)| at_to).collect(),
        junction_reverse: junctions.iter().map(|&(at_from, _)| at_from).collect(),
    };
    attrs.flags = flags.to_vec();
    car_network(vertex_count, &test_edges, &attrs, &car).unwrap()
}

#[test]
fn car_free_flow_is_the_persisted_speeds_alone() {
    // Two elements, 36 and 72 km/h, both endpoints junctions — the default
    // regime (no model) charges nothing anywhere: 100 m at 36 km/h is 10 s,
    // at 72 km/h 5 s, junction classes and flags notwithstanding.
    let net = car_oracle_network(
        &[(0, 1), (1, 2)],
        &[12, 12],
        &[36.0, 72.0],
        &[(1, 1), (1, 3)],
        &[0, FLAG_ROUNDABOUT],
    );
    let profile = net
        .compile_profile(&StreetProfileDefinition::car(None))
        .unwrap();
    assert_eq!(arc_between(&net, &profile, 0, 1), 10_000);
    assert_eq!(arc_between(&net, &profile, 1, 0), 10_000);
    assert_eq!(arc_between(&net, &profile, 1, 2), 5_000);
}

#[test]
fn car_delay_crossing_total_and_terminal_element() {
    // Three residential elements (group 4–6, b = 6) meeting at vertex 1 — a
    // topological junction. Each element charges ½·b for that endpoint and
    // nothing at its dead end, so a path over two of them pays the full
    // crossing b (3 + 3), and each terminal element alone carries just its
    // own share.
    let net = car_oracle_network(
        &[(0, 1), (1, 2), (1, 3)],
        &[12, 12, 12],
        &[36.0, 36.0, 36.0],
        &[(0, 1), (1, 0), (1, 0)],
        &[0, 0, 0],
    );
    let profile = net
        .compile_profile(&StreetProfileDefinition::car(Some(test_delay_model())))
        .unwrap();
    // 10 s free-flow + ½·6 = 13 s, in both directions of every element.
    for (from, to) in [(0, 1), (1, 0), (1, 2), (2, 1), (1, 3), (3, 1)] {
        assert_eq!(arc_between(&net, &profile, from, to), 13_000);
    }
}

#[test]
fn car_delay_mixed_endpoints_charge_each_elements_own_group() {
    // A primary element (b = 12) and a residential one (b = 6), each between
    // a ramp junction (¼·b) and a topological junction (½·b): the shares sum
    // per endpoint and each element reads its own group's b.
    let net = car_oracle_network(
        &[(0, 1), (2, 3)],
        &[5, 12],
        &[36.0, 36.0],
        &[(4, 1), (4, 1)],
        &[0, 0],
    );
    let profile = net
        .compile_profile(&StreetProfileDefinition::car(Some(test_delay_model())))
        .unwrap();
    // Primary: 10 + ¼·12 + ½·12 = 19 s; residential: 10 + 1.5 + 3 = 14.5 s.
    assert_eq!(arc_between(&net, &profile, 0, 1), 19_000);
    assert_eq!(arc_between(&net, &profile, 2, 3), 14_500);
}

#[test]
fn car_delay_ramp_shares_and_the_signalized_ramp_junction() {
    // Ramp elements (motorway_link, group 4–6, b = 6) with one junction
    // endpoint each: at a ramp junction the period share applies — ¾ fast
    // (72 km/h), ½ slow (36 km/h) — while a signalized endpoint gives the
    // ordinary ½ share even to a ramp element (the calibration's hierarchy).
    let net = car_oracle_network(
        &[(0, 1), (2, 3), (4, 5)],
        &[2, 2, 2],
        &[72.0, 36.0, 72.0],
        &[(4, 0), (4, 0), (3, 0)],
        &[0, 0, 0],
    );
    let profile = net
        .compile_profile(&StreetProfileDefinition::car(Some(test_delay_model())))
        .unwrap();
    // Fast ramp at its ramp junction: 5 + 0.75·6 = 9.5 s.
    assert_eq!(arc_between(&net, &profile, 0, 1), 9_500);
    // Slow ramp: 10 + 0.5·6 = 13 s (the low-speed branch).
    assert_eq!(arc_between(&net, &profile, 2, 3), 13_000);
    // Signalized ramp junction: 5 + ½·6 = 8 s.
    assert_eq!(arc_between(&net, &profile, 4, 5), 8_000);
}

#[test]
fn car_delay_multiplier_branches_are_junction_free_only() {
    // Junction-free elements (endpoint classes 0 and priority 2 — an advance
    // sign never charges): the fast ramp takes ×1.5, the fast primary ×1.2,
    // and both slow elements stay at free-flow.
    let net = car_oracle_network(
        &[(0, 1), (2, 3), (4, 5), (6, 7)],
        &[2, 5, 2, 12],
        &[72.0, 72.0, 36.0, 36.0],
        &[(0, 2), (2, 0), (0, 0), (0, 2)],
        &[0, 0, 0, 0],
    );
    let profile = net
        .compile_profile(&StreetProfileDefinition::car(Some(test_delay_model())))
        .unwrap();
    // Fast ramp: 5 × 1.5 = 7.5 s; fast primary: 5 × 1.2 = 6 s.
    assert_eq!(arc_between(&net, &profile, 0, 1), 7_500);
    assert_eq!(arc_between(&net, &profile, 2, 3), 6_000);
    // Slow elements: free-flow, whatever their class.
    assert_eq!(arc_between(&net, &profile, 4, 5), 10_000);
    assert_eq!(arc_between(&net, &profile, 6, 7), 10_000);
}

#[test]
fn car_zero_valued_shares_stay_on_the_penalty_branch() {
    // Junction incidence is a property of the endpoint classes, never of the
    // numeric share values: a zeroed ramp share (or a zeroed b) must yield a
    // zero penalty, not a fall-through onto the multiplier branch.
    let net = car_oracle_network(
        &[(0, 1), (2, 3)],
        &[2, 5],
        &[72.0, 72.0],
        &[(4, 0), (1, 0)],
        &[0, 0],
    );
    let mut zero_share = test_delay_model();
    zero_share.ramp_share_high = 0.0;
    let profile = net
        .compile_profile(&StreetProfileDefinition::car(Some(zero_share)))
        .unwrap();
    // The fast ramp at its ramp junction: free-flow exactly, no ×1.5.
    assert_eq!(arc_between(&net, &profile, 0, 1), 5_000);
    let mut zero_b = test_delay_model();
    zero_b.group_seconds = [0.0, 0.0, 0.0];
    let profile = net
        .compile_profile(&StreetProfileDefinition::car(Some(zero_b)))
        .unwrap();
    // The fast primary at its topological junction: free-flow, no ×1.2.
    assert_eq!(arc_between(&net, &profile, 2, 3), 5_000);
}

#[test]
fn car_partial_traversals_charge_only_crossed_junctions() {
    // Two residential elements (b = 6) joined at vertex 1, both charging
    // ½·b there: full arcs are 10 + 3 = 13 s. Snapped queries must charge a
    // junction exactly when the route crosses it.
    let net = car_oracle_network(
        &[(0, 1), (1, 2)],
        &[12, 12],
        &[36.0, 36.0],
        &[(0, 1), (1, 0)],
        &[0, 0],
    );
    let profile = net
        .compile_profile(&StreetProfileDefinition::car(Some(test_delay_model())))
        .unwrap();
    let snap = |edge: u32, fraction: f64| Snap {
        edge,
        fraction,
        connector: 0.0,
    };
    // Identify the internal edges by their endpoints.
    let edge_at = |from: u32, to: u32| {
        (0..net.edge_count())
            .find(|&edge| {
                let (a, b) = net.edge_endpoints(edge);
                (a, b) == (from, to) || (a, b) == (to, from)
            })
            .unwrap()
    };
    let (e0, e1) = (edge_at(0, 1), edge_at(1, 2));
    // Between two interior points of one element no junction is crossed:
    // half the element at free-flow is 5 s, not half of 13.
    assert_eq!(
        net.directed_travel_time(&snap(e0, 0.25), &snap(e0, 0.75), &profile, 3600.0),
        Some(5)
    );
    // Mid-element to mid-element across vertex 1: each element pays its own
    // ½·b for the one junction crossed — (5 + 3) + (3 + 5) = 16 s, where
    // whole-arc proration would misprice the uncrossed endpoints.
    assert_eq!(
        net.directed_travel_time(&snap(e0, 0.5), &snap(e1, 0.5), &profile, 3600.0),
        Some(16)
    );
    // A same-edge trip whose end sits exactly on the junction vertex crosses
    // it: interior → vertex 1 on e0 is 5 + 3 s (whichever endpoints the
    // stored direction puts at fraction 0 and 1, exactly one end of this
    // edge is its junction), matching what the seed/egress path charges.
    let (_, stored_to) = net.edge_endpoints(e0);
    let junction_fraction = if stored_to == 1 { 1.0 } else { 0.0 };
    assert_eq!(
        net.directed_travel_time(
            &snap(e0, 0.5),
            &snap(e0, junction_fraction),
            &profile,
            3600.0
        ),
        Some(8)
    );
    // The dead-end vertex charges nothing.
    assert_eq!(
        net.directed_travel_time(
            &snap(e0, 0.5),
            &snap(e0, 1.0 - junction_fraction),
            &profile,
            3600.0
        ),
        Some(5)
    );
    // The full 0 → 1 same-edge traversal equals the whole arc: the free-flow
    // base plus the one junction endpoint this edge has — 10 + 3 = 13 s.
    assert_eq!(
        net.directed_travel_time(&snap(e0, 0.0), &snap(e0, 1.0), &profile, 3600.0),
        Some(13)
    );
}

#[test]
fn car_delay_roundabout_interior_replaces_shares() {
    // A roundabout-interior residential element between two topological
    // junctions charges b/4 in place of its 2 × ½·b endpoint shares.
    let net = car_oracle_network(&[(0, 1)], &[12], &[36.0], &[(1, 1)], &[FLAG_ROUNDABOUT]);
    let profile = net
        .compile_profile(&StreetProfileDefinition::car(Some(test_delay_model())))
        .unwrap();
    // 10 + 6/4 = 11.5 s.
    assert_eq!(arc_between(&net, &profile, 0, 1), 11_500);
}

#[test]
fn car_speeds_above_the_default_bound_compile_exactly() {
    // A persisted speed is the edge speed, whatever the profile's fallback
    // bound says: 100 m at a 300 km/h tag compiles unclamped (the
    // goal-directed bound is measured from the compiled costs, so no clamp
    // is needed) — ~1.2 s, not the 1.44 s a 250 km/h ceiling would give.
    let net = car_oracle_network(&[(0, 1)], &[1], &[300.0], &[(0, 0)], &[0]);
    let profile = net
        .compile_profile(&StreetProfileDefinition::car(None))
        .unwrap();
    let unclamped = ((100.0f64 / (300.0 / 3.6)) * 1000.0).ceil() as u32;
    assert_eq!(arc_between(&net, &profile, 0, 1), unclamped);
    assert!(unclamped < 1_440);
}

#[test]
fn car_profiles_reject_bad_models_and_carless_networks() {
    // A malformed model (wrong groups length), a model on a non-car mode,
    // and a car compile over a network without the car group are each
    // refused.
    let mut bad = test_delay_model();
    bad.groups.pop();
    assert_eq!(
        StreetProfileDefinition::car(Some(bad))
            .validate()
            .unwrap_err(),
        ProfileError::InvalidCarModel
    );
    let mut walk = StreetProfileDefinition::walk();
    walk.car = Some(test_delay_model());
    assert_eq!(walk.validate().unwrap_err(), ProfileError::InvalidCarModel);
    let edges: Vec<TestEdge> = vec![(0, 1, 100.0, straight((0.0, 0.0), (100.0, 0.0)))];
    let carless = multimodal_network(2, &edges, &plain_attrs(1)).unwrap();
    assert_eq!(
        carless
            .compile_profile(&StreetProfileDefinition::car(None))
            .unwrap_err(),
        ProfileError::MissingAttributes
    );
}

#[test]
fn new_multimodal_routes_along_permitted_arc_directions() {
    // A two-edge path where bicycles may travel only along the stored geometry
    // direction, while walking is permitted both ways.
    let edges: Vec<TestEdge> = vec![
        (0, 1, 200.0, straight((0.0, 0.0), (200.0, 0.0))),
        (1, 2, 200.0, straight((200.0, 0.0), (400.0, 0.0))),
    ];
    let n = edges.len();
    let attrs = Attrs {
        highway: vec![0; n],
        surface: vec![0; n],
        smoothness: vec![0; n],
        flags: vec![0; n],
        access_forward: vec![MODE_WALK | MODE_BICYCLE; n],
        access_reverse: vec![MODE_WALK; n],
        facility_forward: vec![0; n],
        facility_reverse: vec![0; n],
    };
    let net = multimodal_network(3, &edges, &attrs).unwrap();
    let bike = net
        .compile_profile(&StreetProfileDefinition::bicycle())
        .unwrap();
    let walk = net
        .compile_profile(&StreetProfileDefinition::walk())
        .unwrap();
    let origin = lonlat(10.0, 0.0);
    let target = lonlat(390.0, 0.0);
    let from = net.snap(origin.1, origin.0, 50.0).unwrap();
    let to = net.snap(target.1, target.0, 50.0).unwrap();
    // Bicycle: reachable forward, forbidden against the arc direction.
    assert!(net
        .directed_travel_time(&from, &to, &bike, 3600.0)
        .is_some());
    assert!(net
        .directed_travel_time(&to, &from, &bike, 3600.0)
        .is_none());
    // Walking: permitted both ways.
    assert!(net
        .directed_travel_time(&from, &to, &walk, 3600.0)
        .is_some());
    assert!(net
        .directed_travel_time(&to, &from, &walk, 3600.0)
        .is_some());
}

#[test]
fn new_multimodal_walk_matches_the_walk_only_build() {
    // Identical geometry built two ways: an all-permissive multimodal graph and
    // the attribute-free walk-only graph route walking identically.
    let edges: Vec<TestEdge> = vec![
        (0, 1, 200.0, straight((0.0, 0.0), (200.0, 0.0))),
        (1, 2, 150.0, straight((200.0, 0.0), (350.0, 0.0))),
    ];
    let n = edges.len();
    let every_mode = MODE_WALK | MODE_BICYCLE | MODE_E_SCOOTER;
    let permissive = Attrs {
        highway: vec![0; n],
        surface: vec![0; n],
        smoothness: vec![0; n],
        flags: vec![0; n],
        access_forward: vec![every_mode; n],
        access_reverse: vec![every_mode; n],
        facility_forward: vec![0; n],
        facility_reverse: vec![0; n],
    };
    let multimodal = multimodal_network(3, &edges, &permissive).unwrap();
    let walk_only = network(3, 0, &edges, vec![]).unwrap();
    let walk_m = multimodal
        .compile_profile(&StreetProfileDefinition::walk())
        .unwrap();
    let walk_w = walk_only
        .compile_profile(&StreetProfileDefinition::walk())
        .unwrap();
    let origin = lonlat(10.0, 0.0);
    let target = lonlat(340.0, 0.0);
    let from_m = multimodal.snap(origin.1, origin.0, 50.0).unwrap();
    let to_m = multimodal.snap(target.1, target.0, 50.0).unwrap();
    let from_w = walk_only.snap(origin.1, origin.0, 50.0).unwrap();
    let to_w = walk_only.snap(target.1, target.0, 50.0).unwrap();
    assert_eq!(
        multimodal.directed_travel_time(&from_m, &to_m, &walk_m, 3600.0),
        walk_only.directed_travel_time(&from_w, &to_w, &walk_w, 3600.0)
    );
}

#[test]
fn new_multimodal_rejects_misshaped_or_out_of_range_attributes() {
    let edges: Vec<TestEdge> = vec![
        (0, 1, 100.0, straight((0.0, 0.0), (100.0, 0.0))),
        (1, 2, 100.0, straight((100.0, 0.0), (200.0, 0.0))),
    ];
    let n = edges.len();
    let ok = || Attrs {
        highway: vec![0; n],
        surface: vec![0; n],
        smoothness: vec![0; n],
        flags: vec![0; n],
        access_forward: vec![MODE_WALK; n],
        access_reverse: vec![MODE_WALK; n],
        facility_forward: vec![0; n],
        facility_reverse: vec![0; n],
    };
    // A per-edge slice of the wrong length is rejected before construction.
    let mut short = ok();
    short.highway.pop();
    assert_eq!(
        multimodal_network(3, &edges, &short).unwrap_err(),
        StreetError::InvalidAttributes
    );
    // An out-of-range class code is rejected by the attribute validator.
    let mut bad_code = ok();
    bad_code.highway[0] = HIGHWAY_CODE_COUNT as u8;
    assert_eq!(
        multimodal_network(3, &edges, &bad_code).unwrap_err(),
        StreetError::InvalidAttributes
    );
}

#[test]
fn snap_for_profile_skips_edges_the_mode_may_not_use() {
    // A footway right next to the query and a road slightly farther away. The
    // mode-blind snap takes the footway; a bicycle must skip it and reach the
    // road, or its route would start on an arc it may never traverse.
    let edges: Vec<TestEdge> = vec![
        (0, 1, 100.0, straight((0.0, 0.0), (100.0, 0.0))),
        (2, 3, 100.0, straight((0.0, 40.0), (100.0, 40.0))),
    ];
    let walk_bit = MODE_WALK;
    let both = MODE_WALK | MODE_BICYCLE;
    let attrs = Attrs {
        highway: vec![0; 2],
        surface: vec![0; 2],
        smoothness: vec![0; 2],
        flags: vec![0; 2],
        // Edge 0 (the near one) is walk-only; edge 1 admits bicycles.
        access_forward: vec![walk_bit, both],
        access_reverse: vec![walk_bit, both],
        facility_forward: vec![0; 2],
        facility_reverse: vec![0; 2],
    };
    let net = multimodal_network(4, &edges, &attrs).unwrap();
    let walk = net
        .compile_profile(&StreetProfileDefinition::walk())
        .unwrap();
    let bike = net
        .compile_profile(&StreetProfileDefinition::bicycle())
        .unwrap();
    let (lon, lat) = lonlat(50.0, 5.0); // 5 m from the footway, 35 m from the road
    let blind = net.snap(lat, lon, 100.0).unwrap();
    let walked = net.snap_for_profile(lat, lon, 100.0, &walk).unwrap();
    let biked = net.snap_for_profile(lat, lon, 100.0, &bike).unwrap();
    // Walking keeps the nearest edge; the bicycle takes the farther, usable one.
    assert_eq!(walked, blind);
    assert_ne!(biked.edge, blind.edge);
    assert!(biked.connector > blind.connector);
    // Nothing usable within the allowance still fails to snap.
    assert_eq!(net.snap_for_profile(lat, lon, 10.0, &bike), None);
}

#[test]
fn new_multimodal_orders_coincident_edges_deterministically() {
    // Two geometrically identical parallel edges that differ only in their
    // permissions are not interchangeable: the build must lay them out the same
    // way whatever order they arrive in, so the attributes and routing are
    // stable. Without the attribute tie-breaker the unstable sort would leave
    // their internal numbering — and any coincident snap — input-order-dependent.
    let bike_ok = MODE_WALK | MODE_BICYCLE;
    let walk_only = MODE_WALK;
    let attrs_for = |highway: [u8; 2], forward: [u8; 2], reverse: [u8; 2]| Attrs {
        highway: highway.to_vec(),
        surface: vec![0; 2],
        smoothness: vec![0; 2],
        flags: vec![0; 2],
        access_forward: forward.to_vec(),
        access_reverse: reverse.to_vec(),
        facility_forward: vec![0; 2],
        facility_reverse: vec![0; 2],
    };
    // Edge A allows bikes forward only, edge B in reverse only.
    let edge_a: TestEdge = (0, 1, 200.0, straight((0.0, 0.0), (200.0, 0.0)));
    let edge_b: TestEdge = (0, 1, 200.0, straight((0.0, 0.0), (200.0, 0.0)));
    let net_ab = multimodal_network(
        2,
        &[edge_a.clone(), edge_b.clone()],
        &attrs_for([1, 2], [bike_ok, walk_only], [walk_only, bike_ok]),
    )
    .unwrap();
    let net_ba = multimodal_network(
        2,
        &[edge_b, edge_a],
        &attrs_for([2, 1], [walk_only, bike_ok], [bike_ok, walk_only]),
    )
    .unwrap();

    // Input order changes neither the internal layout's attributes...
    assert_eq!(net_ab.street_attributes(), net_ba.street_attributes());
    // ...nor the routing over it.
    let route = |net: &StreetNetwork| {
        let bike = net
            .compile_profile(&StreetProfileDefinition::bicycle())
            .unwrap();
        let origin = lonlat(10.0, 0.0);
        let target = lonlat(190.0, 0.0);
        let from = net.snap(origin.1, origin.0, 50.0).unwrap();
        let to = net.snap(target.1, target.0, 50.0).unwrap();
        net.directed_travel_time(&from, &to, &bike, 3600.0)
    };
    assert_eq!(route(&net_ab), route(&net_ba));
    assert!(route(&net_ab).is_some());
}

// ---- Distance and geometry reconstruction ----

/// A 4-vertex ladder: a 300 m south side, a 300 m north side, and rungs, with
/// every arc open to walking and cycling.
fn ladder_network() -> StreetNetwork {
    let edges: Vec<TestEdge> = vec![
        (0, 1, 300.0, straight((0.0, 0.0), (300.0, 0.0))),
        (2, 3, 300.0, straight((0.0, 80.0), (300.0, 80.0))),
        (0, 2, 80.0, straight((0.0, 0.0), (0.0, 80.0))),
        (1, 3, 80.0, straight((300.0, 0.0), (300.0, 80.0))),
    ];
    let n = edges.len();
    let both = MODE_WALK | MODE_BICYCLE;
    let attrs = Attrs {
        highway: vec![0; n],
        surface: vec![0; n],
        smoothness: vec![0; n],
        flags: vec![0; n],
        access_forward: vec![both; n],
        access_reverse: vec![both; n],
        facility_forward: vec![0; n],
        facility_reverse: vec![0; n],
    };
    multimodal_network(4, &edges, &attrs).unwrap()
}

fn leg_between(
    net: &StreetNetwork,
    profile: &CompiledStreetProfile,
    from: (f64, f64),
    to: (f64, f64),
) -> StreetLeg {
    let (from_lon, from_lat) = lonlat(from.0, from.1);
    let (to_lon, to_lat) = lonlat(to.0, to.1);
    let from_snap = net
        .snap_for_profile(from_lat, from_lon, 200.0, profile)
        .unwrap();
    let to_snap = net
        .snap_for_profile(to_lat, to_lon, 200.0, profile)
        .unwrap();
    net.directed_leg(
        (from_lat, from_lon),
        &from_snap,
        (to_lat, to_lon),
        &to_snap,
        profile,
        3600.0,
    )
    .unwrap()
}

#[test]
fn a_leg_reports_the_distance_of_the_path_it_took() {
    // Along the south side: 60 m in to 240 m in is 180 m of network, and the
    // two coordinates sit on the edge so neither pays a connector.
    let net = ladder_network();
    let walk = net
        .compile_profile(&StreetProfileDefinition::walk())
        .unwrap();
    let leg = leg_between(&net, &walk, (60.0, 0.0), (240.0, 0.0));
    assert!((leg.network_meters - 180.0).abs() < 1.0);
    assert!(leg.connector_meters < 1.0);
}

#[test]
fn a_leg_reports_its_connectors_separately_from_the_network() {
    // Both coordinates sit 20 m off the south side, so the network distance is
    // unchanged while the connectors add up to roughly 40 m.
    let net = ladder_network();
    let walk = net
        .compile_profile(&StreetProfileDefinition::walk())
        .unwrap();
    let leg = leg_between(&net, &walk, (60.0, -20.0), (240.0, -20.0));
    assert!((leg.network_meters - 180.0).abs() < 1.0);
    assert!((leg.connector_meters - 40.0).abs() < 2.0);
}

#[test]
fn a_leg_agrees_with_the_time_search() {
    // The reconstruction must not disturb the answer: every leg's seconds are
    // the time search's, for both modes and several pairs.
    let net = ladder_network();
    for definition in [
        StreetProfileDefinition::walk(),
        StreetProfileDefinition::bicycle(),
    ] {
        let profile = net.compile_profile(&definition).unwrap();
        for (from, to) in [
            ((10.0, 0.0), (290.0, 0.0)),
            ((10.0, 0.0), (290.0, 80.0)),
            ((150.0, 80.0), (20.0, 0.0)),
        ] {
            let (from_lon, from_lat) = lonlat(from.0, from.1);
            let (to_lon, to_lat) = lonlat(to.0, to.1);
            let from_snap = net
                .snap_for_profile(from_lat, from_lon, 200.0, &profile)
                .unwrap();
            let to_snap = net
                .snap_for_profile(to_lat, to_lon, 200.0, &profile)
                .unwrap();
            let leg = net
                .directed_leg(
                    (from_lat, from_lon),
                    &from_snap,
                    (to_lat, to_lon),
                    &to_snap,
                    &profile,
                    3600.0,
                )
                .unwrap();
            assert_eq!(
                Some(leg.seconds),
                net.directed_travel_time(&from_snap, &to_snap, &profile, 3600.0)
            );
        }
    }
}

#[test]
fn a_same_edge_leg_stays_on_its_edge() {
    // Two snaps on one edge route directly along it: the geometry runs from the
    // origin to the destination without visiting either endpoint.
    let net = ladder_network();
    let walk = net
        .compile_profile(&StreetProfileDefinition::walk())
        .unwrap();
    let leg = leg_between(&net, &walk, (100.0, 0.0), (200.0, 0.0));
    assert!((leg.network_meters - 100.0).abs() < 1.0);
    let (_, south_east) = lonlat(300.0, 0.0);
    assert!(leg
        .geometry
        .iter()
        .all(|&(_, lat)| (lat - south_east).abs() < 1e-9));
    // Ordered from the origin to the destination.
    assert!(leg.geometry.first().unwrap().0 < leg.geometry.last().unwrap().0);
}

#[test]
fn a_leg_detours_when_the_direct_direction_is_forbidden() {
    // The south side is one-way west→east for bicycles, so a bicycle heading
    // east→west may still snap to it (one direction is open) but must ride the
    // ladder around. Snaps are constructed directly, so the test exercises the
    // rejected same-edge candidate rather than a re-snap onto another edge.
    let edges: Vec<TestEdge> = vec![
        (0, 1, 300.0, straight((0.0, 0.0), (300.0, 0.0))),
        (2, 3, 300.0, straight((0.0, 80.0), (300.0, 80.0))),
        (0, 2, 80.0, straight((0.0, 0.0), (0.0, 80.0))),
        (1, 3, 80.0, straight((300.0, 0.0), (300.0, 80.0))),
    ];
    let both = MODE_WALK | MODE_BICYCLE;
    let attrs = Attrs {
        highway: vec![0; 4],
        surface: vec![0; 4],
        smoothness: vec![0; 4],
        flags: vec![0; 4],
        // South side: bicycles only west→east. Everything else is open.
        access_forward: vec![both, both, both, both],
        access_reverse: vec![MODE_WALK, both, both, both],
        facility_forward: vec![0; 4],
        facility_reverse: vec![0; 4],
    };
    let net = multimodal_network(4, &edges, &attrs).unwrap();
    let bike = net
        .compile_profile(&StreetProfileDefinition::bicycle())
        .unwrap();
    let south = net
        .arrays()
        .adj_edges()
        .iter()
        .copied()
        .find(|&edge| {
            net.arrays().lengths()[edge as usize] == 300.0
                && net.point_at(edge, 0.0).1 < net.point_at(edge, 0.5).1 + 1e-9
                && (net.point_at(edge, 0.0).1 - lonlat(0.0, 0.0).1).abs() < 1e-9
        })
        .expect("the south edge");
    // Ride east→west along the south edge: from 290 m back to 10 m.
    let start = Snap {
        edge: south,
        fraction: 290.0 / 300.0,
        connector: 0.0,
    };
    let end = Snap {
        edge: south,
        fraction: 10.0 / 300.0,
        connector: 0.0,
    };
    let start_point = lonlat(290.0, 0.0);
    let end_point = lonlat(10.0, 0.0);
    let leg = net
        .directed_leg(
            (start_point.1, start_point.0),
            &start,
            (end_point.1, end_point.0),
            &end,
            &bike,
            3600.0,
        )
        .unwrap();
    // 10 m to the east corner, up 80, 300 west, down 80, then 10 m east.
    assert!((leg.network_meters - 480.0).abs() < 1.0);
    assert!(leg.connector_meters < 1e-9);
    // The geometry rounds the ladder: it must reach both northern corners.
    let north_east = lonlat(300.0, 80.0);
    let north_west = lonlat(0.0, 80.0);
    // Stored coordinates sit on the fixed-point grid, so compare at grid scale.
    let touches = |target: (f64, f64)| {
        leg.geometry
            .iter()
            .any(|&(lon, lat)| (lon - target.0).abs() < 1e-6 && (lat - target.1).abs() < 1e-6)
    };
    assert!(touches(north_east));
    assert!(touches(north_west));
    // And it is the detour, not the 280 m direct line.
    assert!(leg.network_meters > 400.0);
}

#[test]
fn a_self_loop_leg_follows_the_permitted_side() {
    // A loop edge from vertex 1 back to itself, hung off a stub. The snap sits
    // at a fifth of the way round, so the *near* side is the short way back —
    // but that direction is forbidden, and the leg must report the long way
    // round it actually rides, not the near side its vertex would suggest.
    let edges: Vec<TestEdge> = vec![
        (0, 1, 50.0, straight((0.0, 0.0), (50.0, 0.0))),
        (
            1,
            1,
            400.0,
            vec![
                (50.0, 0.0),
                (150.0, 0.0),
                (150.0, 100.0),
                (50.0, 100.0),
                (50.0, 0.0),
            ],
        ),
    ];
    let both = MODE_WALK | MODE_BICYCLE;
    let attrs = Attrs {
        highway: vec![0; 2],
        surface: vec![0; 2],
        smoothness: vec![0; 2],
        flags: vec![0; 2],
        // The loop runs one way only, in its stored direction.
        access_forward: vec![both, both],
        access_reverse: vec![both, MODE_WALK],
        facility_forward: vec![0; 2],
        facility_reverse: vec![0; 2],
    };
    let net = multimodal_network(2, &edges, &attrs).unwrap();
    let bike = net
        .compile_profile(&StreetProfileDefinition::bicycle())
        .unwrap();
    let loop_edge = (0..net.edge_count())
        .find(|&edge| {
            let (u, v) = net.edge_endpoints(edge);
            u == v
        })
        .expect("the loop edge");
    let stub = 1 - loop_edge;
    // From a fifth of the way round the loop, back to the stub's far end. The
    // near side (a fifth, backwards) is forbidden, so the ride continues
    // forwards around the remaining four fifths.
    let from = Snap {
        edge: loop_edge,
        fraction: 0.2,
        connector: 0.0,
    };
    let to = Snap {
        edge: stub,
        fraction: 0.0,
        connector: 0.0,
    };
    let from_point = net.point_at(loop_edge, 0.2);
    let to_point = net.point_at(stub, 0.0);
    let leg = net
        .directed_leg(
            (from_point.1, from_point.0),
            &from,
            (to_point.1, to_point.0),
            &to,
            &bike,
            3600.0,
        )
        .unwrap();
    // Four fifths of the 400 m loop, then the 50 m stub — not the 80 m
    // near-side shortcut the snap fraction alone would imply.
    assert!((leg.network_meters - (320.0 + 50.0)).abs() < 1.0);
    assert_eq!(
        Some(leg.seconds),
        net.directed_travel_time(&from, &to, &bike, 3600.0)
    );
}
#[test]
fn the_row_form_matches_leg_by_leg() {
    // One search serving many destinations, with its memoised prefix metres,
    // must produce exactly what the per-pair reconstruction does.
    let net = ladder_network();
    let bike = net
        .compile_profile(&StreetProfileDefinition::bicycle())
        .unwrap();
    let origin = lonlat(10.0, 0.0);
    let origin_point = (origin.1, origin.0);
    let from = net
        .snap_for_profile(origin.1, origin.0, 200.0, &bike)
        .unwrap();
    let places = [(290.0, 0.0), (150.0, 80.0), (290.0, 80.0), (20.0, 80.0)];
    let targets: Vec<((f64, f64), Option<Snap>)> = places
        .iter()
        .map(|&(x, y)| {
            let (lon, lat) = lonlat(x, y);
            ((lat, lon), net.snap_for_profile(lat, lon, 200.0, &bike))
        })
        .collect();
    let row = net.directed_legs_to_snaps(origin_point, &from, &targets, &bike, 3600.0);
    for (index, (point, snap)) in targets.iter().enumerate() {
        let expected = net.directed_leg(
            origin_point,
            &from,
            *point,
            snap.as_ref().unwrap(),
            &bike,
            3600.0,
        );
        assert_eq!(row[index], expected);
    }
}

// ---- Elevation intake ----

/// Builds a multimodal network with a per-input-coordinate elevation
/// callback, laying out geometry exactly as [`multimodal_network`] does.
fn elevated_network(
    vertex_count: u32,
    edges: &[TestEdge],
    attrs: &Attrs,
    elevation: impl Fn(f64, f64) -> f32,
) -> StreetNetwork {
    let mut offsets = vec![0u32];
    let mut longitudes = Vec::new();
    let mut latitudes = Vec::new();
    let mut elevations = Vec::new();
    for (_, _, _, path) in edges {
        for &(x, y) in path {
            let (lon, lat) = lonlat(x, y);
            longitudes.push(lon);
            latitudes.push(lat);
            elevations.push(elevation(x, y));
        }
        offsets.push(longitudes.len() as u32);
    }
    let flat: Vec<(u32, u32, f64)> = edges
        .iter()
        .map(|&(from, to, meters, _)| (from, to, meters))
        .collect();
    StreetNetwork::new_multimodal(
        vertex_count,
        0,
        &flat,
        &offsets,
        &longitudes,
        &latitudes,
        vec![],
        EdgeAttributes {
            highway: &attrs.highway,
            surface: &attrs.surface,
            smoothness: &attrs.smoothness,
            flags: &attrs.flags,
            access_forward: &attrs.access_forward,
            access_reverse: &attrs.access_reverse,
            facility_forward: &attrs.facility_forward,
            facility_reverse: &attrs.facility_reverse,
            car: None,
        },
        Some(&elevations),
    )
    .unwrap()
}

fn plain_attrs(count: usize) -> Attrs {
    let both = MODE_WALK | MODE_BICYCLE;
    Attrs {
        highway: vec![0; count],
        surface: vec![0; count],
        smoothness: vec![0; count],
        flags: vec![0; count],
        access_forward: vec![both; count],
        access_reverse: vec![both; count],
        facility_forward: vec![0; count],
        facility_reverse: vec![0; count],
    }
}

#[test]
fn elevations_ride_the_reorder_and_the_densifier() {
    // A 500 m ramp climbing 1 m per 10 m of easting, given as edges out of
    // spatial order so the reorder permutes them. Every stored coordinate —
    // including the ones the densifier inserts — must carry the elevation of
    // its own longitude, which pins both the permutation and the
    // interpolation.
    let edges: Vec<TestEdge> = vec![
        (2, 3, 100.0, straight((200.0, 0.0), (300.0, 0.0))),
        (0, 1, 100.0, straight((0.0, 0.0), (100.0, 0.0))),
        (4, 5, 100.0, straight((400.0, 0.0), (500.0, 0.0))),
        (1, 2, 100.0, straight((100.0, 0.0), (200.0, 0.0))),
        (3, 4, 100.0, straight((300.0, 0.0), (400.0, 0.0))),
    ];
    let net = elevated_network(6, &edges, &plain_attrs(5), |x, _| (x / 10.0) as f32);
    let elevations = net.elevations().expect("elevations installed");
    let lons = net.arrays().lons();
    assert_eq!(elevations.len(), lons.len());
    // The densifier split the 100 m edges (segments cap under 100 m), so
    // there are more stored coordinates than the 10 input ones.
    assert!(elevations.len() > 10);
    let origin_lon = lonlat(0.0, 0.0).0;
    let per_lon = meters_per_degree(60.0).0;
    for (&lon, &elevation) in lons.iter().zip(elevations) {
        let x = (degrees(lon) - origin_lon) * per_lon;
        assert!(
            (f64::from(elevation) - x / 10.0).abs() < 0.05,
            "coordinate at x={x:.1} carries {elevation}"
        );
    }
}

#[test]
fn unavailable_elevation_stays_unavailable_through_densification() {
    // The second input coordinate has no elevation; the densifier must mark
    // every point it inserts against that endpoint as NaN rather than invent
    // values between a known and an unknown.
    let edges: Vec<TestEdge> = vec![(0, 1, 200.0, straight((0.0, 0.0), (200.0, 0.0)))];
    let net = elevated_network(2, &edges, &plain_attrs(1), |x, _| {
        if x > 100.0 {
            f32::NAN
        } else {
            10.0
        }
    });
    let elevations = net.elevations().unwrap();
    assert!(elevations.len() > 2);
    // The first stored coordinate keeps its sampled value; everything
    // interpolated toward the NaN endpoint, and the endpoint itself, is NaN.
    assert_eq!(elevations[0], 10.0);
    assert!(elevations[1..].iter().all(|value| value.is_nan()));
}

#[test]
fn a_build_without_elevations_installs_none() {
    let edges: Vec<TestEdge> = vec![(0, 1, 100.0, straight((0.0, 0.0), (100.0, 0.0)))];
    let net = multimodal_network(2, &edges, &plain_attrs(1)).unwrap();
    assert!(net.elevations().is_none());
}

#[test]
fn a_misshaped_elevation_array_is_rejected() {
    let (lon_a, lat_a) = lonlat(0.0, 0.0);
    let (lon_b, lat_b) = lonlat(100.0, 0.0);
    let attrs = plain_attrs(1);
    let result = StreetNetwork::new_multimodal(
        2,
        0,
        &[(0, 1, 100.0)],
        &[0, 2],
        &[lon_a, lon_b],
        &[lat_a, lat_b],
        vec![],
        EdgeAttributes {
            highway: &attrs.highway,
            surface: &attrs.surface,
            smoothness: &attrs.smoothness,
            flags: &attrs.flags,
            access_forward: &attrs.access_forward,
            access_reverse: &attrs.access_reverse,
            facility_forward: &attrs.facility_forward,
            facility_reverse: &attrs.facility_reverse,
            car: None,
        },
        Some(&[1.0]),
    );
    assert_eq!(result.unwrap_err(), StreetError::InvalidAttributes);
}

#[test]
fn an_infinite_elevation_is_rejected() {
    // The contract is finite metres or NaN; an infinity would poison every
    // slope computed over it, so construction refuses it.
    let (lon_a, lat_a) = lonlat(0.0, 0.0);
    let (lon_b, lat_b) = lonlat(100.0, 0.0);
    let attrs = plain_attrs(1);
    let result = StreetNetwork::new_multimodal(
        2,
        0,
        &[(0, 1, 100.0)],
        &[0, 2],
        &[lon_a, lon_b],
        &[lat_a, lat_b],
        vec![],
        EdgeAttributes {
            highway: &attrs.highway,
            surface: &attrs.surface,
            smoothness: &attrs.smoothness,
            flags: &attrs.flags,
            access_forward: &attrs.access_forward,
            access_reverse: &attrs.access_reverse,
            facility_forward: &attrs.facility_forward,
            facility_reverse: &attrs.facility_reverse,
            car: None,
        },
        Some(&[0.0, f32::INFINITY]),
    );
    assert_eq!(result.unwrap_err(), StreetError::InvalidAttributes);
}

// ---- Slope-aware profile compilation ----

/// The permitted arc costs of a compiled profile, ascending.
fn permitted_costs(compiled: &CompiledStreetProfile) -> Vec<u32> {
    let mut costs: Vec<u32> = compiled
        .arc_millis()
        .iter()
        .copied()
        .filter(|&cost| cost != u32::MAX)
        .collect();
    costs.sort_unstable();
    costs
}

fn near(actual: u32, expected: f64) -> bool {
    (f64::from(actual) - expected).abs() < 100.0
}

#[test]
fn a_flat_dem_compiles_bit_identically_to_no_dem() {
    let edges: Vec<TestEdge> = vec![
        (0, 1, 200.0, straight((0.0, 0.0), (200.0, 0.0))),
        (1, 2, 120.0, straight((200.0, 0.0), (200.0, 120.0))),
    ];
    let flat = multimodal_network(3, &edges, &plain_attrs(2)).unwrap();
    let elevated = elevated_network(3, &edges, &plain_attrs(2), |_, _| 7.5);
    for definition in [
        StreetProfileDefinition::walk(),
        StreetProfileDefinition::bicycle(),
        StreetProfileDefinition::e_scooter(),
    ] {
        assert_eq!(
            flat.compile_profile(&definition).unwrap().arc_millis(),
            elevated.compile_profile(&definition).unwrap().arc_millis(),
        );
    }
}

#[test]
fn a_ramp_charges_the_climb_and_credits_the_descent() {
    // 200 m east at a 10 % grade. The flat bicycle cost is 50_000 ms; the
    // climb multiplies by 1.1, the descent by 1 − 0.3·0.1 = 0.97.
    let edges: Vec<TestEdge> = vec![(0, 1, 200.0, straight((0.0, 0.0), (200.0, 0.0)))];
    let net = elevated_network(2, &edges, &plain_attrs(1), |x, _| (x / 10.0) as f32);
    let bike = net
        .compile_profile(&StreetProfileDefinition::bicycle())
        .unwrap();
    let costs = permitted_costs(&bike);
    assert_eq!(costs.len(), 2);
    assert!(near(costs[0], 48_500.0), "downhill arc: {}", costs[0]);
    assert!(near(costs[1], 55_000.0), "uphill arc: {}", costs[1]);
    // The directions land the right way round: climbing east costs more.
    let origin = lonlat(10.0, 0.0);
    let target = lonlat(190.0, 0.0);
    let from = net.snap(origin.1, origin.0, 50.0).unwrap();
    let to = net.snap(target.1, target.0, 50.0).unwrap();
    let east = net.directed_travel_time(&from, &to, &bike, 3600.0).unwrap();
    let west = net.directed_travel_time(&to, &from, &bike, 3600.0).unwrap();
    assert!(east > west, "east {east} (climb) vs west {west} (descent)");
}

#[test]
fn a_hill_with_equal_endpoints_costs_both_directions_more_than_flat() {
    // Up 10 % for 100 m, back down to the start elevation: a net endpoint
    // slope of zero would compile flat, but each direction climbs one half
    // and descends the other — (1.1 + 0.97) / 2 = 1.035 of flat, both ways.
    let edges: Vec<TestEdge> = vec![(0, 1, 200.0, vec![(0.0, 0.0), (100.0, 0.0), (200.0, 0.0)])];
    let net = elevated_network(2, &edges, &plain_attrs(1), |x, _| {
        ((100.0 - (x - 100.0).abs()) / 10.0) as f32
    });
    let bike = net
        .compile_profile(&StreetProfileDefinition::bicycle())
        .unwrap();
    let costs = permitted_costs(&bike);
    assert_eq!(costs.len(), 2);
    for &cost in &costs {
        assert!(near(cost, 51_750.0), "hill arc: {cost}");
    }
}

#[test]
fn nodata_elevations_and_slope_free_profiles_compile_flat() {
    let edges: Vec<TestEdge> = vec![(0, 1, 200.0, straight((0.0, 0.0), (200.0, 0.0)))];
    let flat = multimodal_network(2, &edges, &plain_attrs(1)).unwrap();
    // Unavailable elevation is flat, never a penalty or a credit.
    let nodata = elevated_network(2, &edges, &plain_attrs(1), |_, _| f32::NAN);
    let bike = StreetProfileDefinition::bicycle();
    assert_eq!(
        nodata.compile_profile(&bike).unwrap().arc_millis(),
        flat.compile_profile(&bike).unwrap().arc_millis()
    );
    // Slope-free profiles ignore even a strong ramp.
    let ramp = elevated_network(2, &edges, &plain_attrs(1), |x, _| (x / 10.0) as f32);
    for definition in [
        StreetProfileDefinition::walk(),
        StreetProfileDefinition::e_bike(),
        StreetProfileDefinition::e_scooter(),
    ] {
        assert_eq!(
            ramp.compile_profile(&definition).unwrap().arc_millis(),
            flat.compile_profile(&definition).unwrap().arc_millis()
        );
    }
}

#[test]
fn dem_spikes_clamp_and_extreme_credits_floor() {
    // A 1000 % grade is a DEM artifact: it clamps to ±100 %, so the climb
    // doubles (×2.0) and the descent credits ×0.7 — never less.
    let edges: Vec<TestEdge> = vec![(0, 1, 200.0, straight((0.0, 0.0), (200.0, 0.0)))];
    let net = elevated_network(2, &edges, &plain_attrs(1), |x, _| (x * 10.0) as f32);
    let bike = net
        .compile_profile(&StreetProfileDefinition::bicycle())
        .unwrap();
    let costs = permitted_costs(&bike);
    assert!(near(costs[0], 35_000.0), "clamped descent: {}", costs[0]);
    assert!(near(costs[1], 100_000.0), "clamped climb: {}", costs[1]);
    // A pathological downhill factor floors at MIN_SLOPE_MULTIPLIER rather
    // than compiling a vanishing (or negative) cost.
    let mut greedy = StreetProfileDefinition::bicycle();
    greedy.slope_downhill = 5.0;
    greedy.max_speed = greedy.base_speed / MIN_SLOPE_MULTIPLIER;
    let floored = permitted_costs(&net.compile_profile(&greedy).unwrap());
    assert!(near(floored[0], 5_000.0), "floored descent: {}", floored[0]);
}

#[test]
fn slope_factors_validate() {
    let mut nan = StreetProfileDefinition::bicycle();
    nan.slope_uphill = f64::NAN;
    assert_eq!(
        nan.validate().unwrap_err(),
        ProfileError::InvalidSlopeFactors
    );
    let mut negative = StreetProfileDefinition::bicycle();
    negative.slope_downhill = -0.1;
    assert_eq!(
        negative.validate().unwrap_err(),
        ProfileError::InvalidSlopeFactors
    );
    // The shipped bicycle's max_speed covers its own downhill credit...
    assert!(StreetProfileDefinition::bicycle().validate().is_ok());
    // ...and a bound that ignores the credit is rejected as too low.
    let mut low = StreetProfileDefinition::bicycle();
    low.max_speed = low.base_speed;
    assert_eq!(low.validate().unwrap_err(), ProfileError::MaxSpeedTooLow);
}

#[test]
fn adoption_rejects_misshaped_or_infinite_persisted_elevations() {
    // Both adoption paths validate elevations like attributes: a wrong-length
    // or infinite array is refused rather than indexed out of alignment. The
    // owned path is exercised through the public parts round-trip; the mapped
    // path runs the same checks.
    let edges: Vec<TestEdge> = vec![(0, 1, 100.0, straight((0.0, 0.0), (100.0, 0.0)))];
    let net = elevated_network(2, &edges, &plain_attrs(1), |_, _| 5.0);
    assert!(StreetNetwork::from_parts(net.to_parts()).is_ok());
    let mut wrong_len = net.to_parts();
    wrong_len.elevations.as_mut().unwrap().pop();
    assert_eq!(
        StreetNetwork::from_parts(wrong_len).unwrap_err(),
        StreetError::InvalidAttributes
    );
    let mut infinite = net.to_parts();
    infinite.elevations.as_mut().unwrap()[0] = f32::INFINITY;
    assert_eq!(
        StreetNetwork::from_parts(infinite).unwrap_err(),
        StreetError::InvalidAttributes
    );
    // The install path refuses an infinity the same way.
    let mut plain = multimodal_network(2, &edges, &plain_attrs(1)).unwrap();
    let count = plain.arrays().lons().len();
    assert_eq!(
        plain
            .install_elevations(vec![f32::INFINITY; count])
            .unwrap_err(),
        StreetError::InvalidAttributes
    );
}

#[test]
fn coincident_edges_differing_only_in_elevation_order_deterministically() {
    // Two edges identical in geometry and attributes but not in their
    // elevation profiles — one of them holding a NaN — must lay out the same
    // way whatever order they arrive in, so the stored profile stays a pure
    // function of the edge set.
    let path = straight((0.0, 0.0), (200.0, 0.0));
    let build = |profiles: [[f32; 2]; 2]| {
        let mut offsets = vec![0u32];
        let mut longitudes = Vec::new();
        let mut latitudes = Vec::new();
        let mut elevations = Vec::new();
        for profile in &profiles {
            for (&(x, y), &elevation) in path.iter().zip(profile) {
                let (lon, lat) = lonlat(x, y);
                longitudes.push(lon);
                latitudes.push(lat);
                elevations.push(elevation);
            }
            offsets.push(longitudes.len() as u32);
        }
        let attrs = plain_attrs(2);
        StreetNetwork::new_multimodal(
            2,
            0,
            &[(0, 1, 200.0), (0, 1, 200.0)],
            &offsets,
            &longitudes,
            &latitudes,
            vec![],
            EdgeAttributes {
                highway: &attrs.highway,
                surface: &attrs.surface,
                smoothness: &attrs.smoothness,
                flags: &attrs.flags,
                access_forward: &attrs.access_forward,
                access_reverse: &attrs.access_reverse,
                facility_forward: &attrs.facility_forward,
                facility_reverse: &attrs.facility_reverse,
                car: None,
            },
            Some(&elevations),
        )
        .unwrap()
    };
    let bits = |net: &StreetNetwork| {
        net.elevations()
            .unwrap()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    };
    let ab = build([[0.0, 20.0], [f32::NAN, 5.0]]);
    let ba = build([[f32::NAN, 5.0], [0.0, 20.0]]);
    assert_eq!(bits(&ab), bits(&ba));
}
