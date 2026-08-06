"""Tier-4 map matching: mode graphs, bounded paths, chain selection."""

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely

from cafein import _map_match


def identity(lons, lats):
    return np.column_stack([np.asarray(lons, float), np.asarray(lats, float)])


def graph_of(edges, kind="rail"):
    """A ModeGraph from ``(u_xy, v_xy)`` directed edge pairs in meters."""
    index = {}
    lonlat = []
    u, v, lengths = [], [], []
    for a, b in edges:
        for point in (a, b):
            if point not in index:
                index[point] = len(lonlat)
                lonlat.append(point)
        u.append(index[a])
        v.append(index[b])
        lengths.append(float(np.hypot(b[0] - a[0], b[1] - a[1])))
    xy = np.asarray(lonlat, dtype=float).reshape(-1, 2)
    return _map_match.ModeGraph(xy, xy, u, v, lengths, kind), index


LINE = [((0, 0), (100, 0)), ((100, 0), (200, 0)), ((200, 0), (300, 0))]
BOTH_WAYS = LINE + [(b, a) for a, b in LINE]


def test_paths_run_grouped_and_cache():
    graph, index = graph_of(BOTH_WAYS)
    a, c, d = index[(0, 0)], index[(200, 0)], index[(300, 0)]
    found = graph.paths({(a, c): 200.0, (a, d): 300.0})
    assert found[(a, c)][0] == pytest.approx(200.0)
    assert found[(a, d)][0] == pytest.approx(300.0)
    assert found[(a, d)][1] == [a, index[(100, 0)], c, d]
    assert (a, c) in graph._paths  # cached for the next pattern
    again = graph.paths({(a, c): 200.0})
    assert again[(a, c)][0] == pytest.approx(200.0)


def test_oneway_edges_have_no_reverse_path():
    graph, index = graph_of(LINE)
    d, a = index[(300, 0)], index[(0, 0)]
    assert graph.paths({(d, a): 300.0})[(d, a)] is None


def test_detour_beyond_bound_refuses():
    # The only route from A to B is a 400 m detour for a 100 m hop:
    # beyond the rail bound (1.5×).
    detour = [
        ((0, 0), (0, 200)),
        ((0, 200), (100, 200)),
        ((100, 200), (100, 0)),
    ]
    graph, index = graph_of(detour + [(b, a) for a, b in detour])
    a, b = index[(0, 0)], index[(100, 0)]
    assert graph.paths({(a, b): 100.0})[(a, b)] is None
    # A generous crow lifts the ceiling above the 500 m detour.
    assert graph.paths({(a, b): 400.0})[(a, b)][0] == pytest.approx(500.0)


def test_failed_search_retries_with_a_wider_bound():
    graph, index = graph_of(BOTH_WAYS)
    a, d = index[(0, 0)], index[(300, 0)]
    assert graph.paths({(a, d): 10.0})[(a, d)] is None  # limit 15 m
    assert graph.paths({(a, d): 300.0})[(a, d)][0] == pytest.approx(300.0)


def test_match_chain_picks_the_reachable_candidates():
    # Two parallel one-way tracks 10 m apart, opposite directions. The
    # stops sit nearer the WRONG track: the chain matcher must pick
    # the eastbound track's vertices anyway.
    eastbound = [((0, 0), (150, 0)), ((150, 0), (300, 0))]
    westbound = [((300, 10), (150, 10)), ((150, 10), (0, 10))]
    graph, index = graph_of(eastbound + westbound)
    stops = np.asarray([[0.0, 8.0], [150.0, 8.0], [300.0, 8.0]])
    segments = _map_match.match_chain(graph, stops)
    assert segments is not None
    assert [length for length, _ in segments] == [
        pytest.approx(150.0),
        pytest.approx(150.0),
    ]
    chain = graph.chain([path for _, path in segments])
    assert chain is not None
    projected, lons, lats = chain
    assert set(lats) == {0.0}  # every vertex on the eastbound track


def test_match_chain_requires_candidates_for_every_stop():
    graph, index = graph_of(BOTH_WAYS)
    stops = np.asarray([[0.0, 0.0], [150.0, 500.0], [300.0, 0.0]])
    assert _map_match.match_chain(graph, stops) is None


def test_chain_deduplicates_joints():
    graph, index = graph_of(BOTH_WAYS)
    a, b, c = index[(0, 0)], index[(100, 0)], index[(200, 0)]
    chain = graph.chain([[a, b], [b, c]])
    assert chain is not None
    projected, lons, lats = chain
    assert lons == [0.0, 100.0, 200.0]


def test_parallel_edges_keep_the_shortest():
    a, b = (0, 0), (100, 0)
    xy = np.asarray([a, b], dtype=float)
    graph = _map_match.ModeGraph(xy, xy, [0, 0], [1, 1], [400.0, 100.0], "rail")
    assert graph.paths({(0, 1): 100.0})[(0, 1)][0] == pytest.approx(100.0)


def rail_way(refs, points, railway="tram", oneway=None):
    return (
        np.asarray(refs, dtype=np.int64),
        [point[0] for point in points],
        [point[1] for point in points],
        railway,
        oneway,
    )


def test_rail_graph_directions():
    ways = [
        rail_way((1, 2), [(0, 0), (1, 0)]),  # bidirectional
        rail_way((2, 3), [(1, 0), (2, 0)], oneway="yes"),  # forward only
        rail_way((3, 4), [(2, 0), (3, 0)], oneway="-1"),  # reverse only
        rail_way((4, 5), [(3, 0), (4, 0)], oneway="alternating"),  # dropped
        rail_way((5, 6), [(4, 0), (5, 0)], railway="rail"),  # other mode
    ]
    graph = _map_match.rail_graph(ways, identity, ("tram", "light_rail"))
    assert graph._matrix.nnz == 4  # 2 + 1 + 1; alternating and rail out
    forward = graph.paths({(0, 2): 2.0})  # node 1 →(both) 2 →(yes) 3
    assert forward[(0, 2)] is not None
    reverse_blocked = graph.paths({(2, 0): 2.0})
    assert reverse_blocked[(2, 0)] is None


def test_rail_graph_joins_shared_node_ids():
    ways = [
        rail_way((1, 2), [(0, 0), (1, 0)]),
        rail_way((2, 3), [(1, 0), (1, 1)]),
    ]
    graph = _map_match.rail_graph(ways, identity, ("tram",))
    assert len(graph._xy) == 3  # the shared node 2 joined
    assert graph.paths({(0, 2): 2.0})[(0, 2)] is not None


def test_rail_graph_keeps_grade_separations_apart():
    # Two tracks cross at an identical COORDINATE but distinct OSM
    # nodes (one bridges the other): they must never junction.
    ways = [
        rail_way((1, 2, 3), [(0, 0), (1, 0), (2, 0)]),
        rail_way((4, 5, 6), [(1, -1), (1, 0), (1, 1)]),
    ]
    graph = _map_match.rail_graph(ways, identity, ("tram",))
    assert len(graph._xy) == 6  # the coincident vertex stays split
    across = graph.paths({(0, 5): 2.0})  # node 1 → node 6
    assert across[(0, 5)] is None


def street_frames(edge_rows, barrier_nodes=()):
    """pyrosm-shaped (nodes, edges) frames from explicit rows, with
    ``u``/``v`` endpoint node ids synthesised per shared endpoint."""
    node_ids = {}

    def node_id(point):
        return node_ids.setdefault(tuple(point), len(node_ids))

    columns = {key for row in edge_rows for key in row} - {"points"}
    data = {column: [row.get(column) for row in edge_rows] for column in columns}
    data["geometry"] = [shapely.LineString(row["points"]) for row in edge_rows]
    data["u"] = [node_id(row["points"][0]) for row in edge_rows]
    data["v"] = [node_id(row["points"][-1]) for row in edge_rows]
    edges = gpd.GeoDataFrame(data, geometry="geometry", crs="EPSG:4326")
    nodes = pd.DataFrame(
        {
            "lon": [point[0] for point, _ in barrier_nodes],
            "lat": [point[1] for point, _ in barrier_nodes],
            "tags": [json.dumps(tags) for _, tags in barrier_nodes],
        }
    )
    return nodes, edges


def test_bus_graph_blocks_at_barriers():
    # The direct street passes a gate: its edge drops, and the only
    # remaining route is the detour around it — the plan's always-on
    # barrier pin.
    edge_rows = [
        {"points": [(0, 0), (100, 0)], "highway": "residential"},
        # The gated edge carries the barrier node mid-geometry.
        {"points": [(100, 0), (150, 0), (200, 0)], "highway": "residential"},
        {"points": [(100, 0), (100, 100)], "highway": "residential"},
        {"points": [(100, 100), (200, 100)], "highway": "residential"},
        {"points": [(200, 100), (200, 0)], "highway": "residential"},
    ]
    nodes, edges = street_frames(
        edge_rows, barrier_nodes=[((150.0, 0.0), {"barrier": "gate"})]
    )
    graph = _map_match.bus_graph(nodes, edges, identity)
    start = graph.candidates(np.asarray([0.0, 0.0]))[0][0]
    end = graph.candidates(np.asarray([200.0, 0.0]))[0][0]
    found = graph.paths({(start, end): 200.0})[(start, end)]
    assert found is not None
    assert found[0] == pytest.approx(400.0)  # around, not through


def test_bus_graph_opens_the_barrier_on_explicit_allow():
    edge_rows = [
        {"points": [(0, 0), (150, 0), (300, 0)], "highway": "residential"},
    ]
    nodes, edges = street_frames(
        edge_rows,
        barrier_nodes=[((150.0, 0.0), {"barrier": "gate", "bus": "yes"})],
    )
    graph = _map_match.bus_graph(nodes, edges, identity)
    start = graph.candidates(np.asarray([0.0, 0.0]))[0][0]
    end = graph.candidates(np.asarray([300.0, 0.0]))[0][0]
    assert graph.paths({(start, end): 300.0})[(start, end)] is not None


def test_bus_graph_includes_bus_only_streets():
    edge_rows = [
        {"points": [(0, 0), (100, 0)], "highway": "residential"},
        {
            "points": [(100, 0), (200, 0)],
            "highway": "residential",
            "access": "no",
            "bus": "yes",
        },
        {"points": [(200, 0), (300, 0)], "highway": "busway"},
    ]
    nodes, edges = street_frames(edge_rows)
    graph = _map_match.bus_graph(nodes, edges, identity)
    start = graph.candidates(np.asarray([0.0, 0.0]))[0][0]
    end = graph.candidates(np.asarray([300.0, 0.0]))[0][0]
    assert graph.paths({(start, end): 300.0})[(start, end)] is not None


def test_missing_barrier_metadata_fails_closed():
    # Without the nodes' tag metadata the barrier rule cannot run:
    # the graph builds empty instead of routing through barriers.
    edge_rows = [{"points": [(0, 0), (100, 0)], "highway": "residential"}]
    nodes, edges = street_frames(edge_rows)
    graph = _map_match.bus_graph(nodes.drop(columns=["tags"]), edges, identity)
    assert graph._matrix.nnz == 0


def test_bus_graph_excludes_bus_banned_streets():
    edge_rows = [
        {"points": [(0, 0), (100, 0)], "highway": "residential", "bus": "no"},
    ]
    nodes, edges = street_frames(edge_rows)
    graph = _map_match.bus_graph(nodes, edges, identity)
    assert graph._matrix.nnz == 0


KANTAKAUPUNKI = "tests/data/kantakaupunki.osm.pbf"


def test_street_read_reaches_the_permission_compiler():
    # The extraction must carry every column the PSV chain consumes —
    # a missing motor_vehicle would silently legalise restricted ways.
    import pathlib

    from cafein import _osm, _osm_tiers

    if not pathlib.Path(KANTAKAUPUNKI).exists():
        pytest.skip("kantakaupunki fixture missing")
    ladder = _osm_tiers.OsmLadder(KANTAKAUPUNKI, frozenset({"bus"}))
    nodes, edges = ladder._streets()
    for column in ("psv", "bus", "vehicle", "motor_vehicle", "access", "oneway"):
        assert column in edges.columns, column
    # Pedestrian streets reach the compiler (explicit bus grants are
    # honoured there); their untagged ways stay denied.
    assert (edges["highway"] == "pedestrian").any()
    forward, reverse, _ = _osm.bus_permissions(edges)
    restricted = (
        (edges["motor_vehicle"] == "no")
        & ~edges["psv"].isin(("yes", "designated"))
        & ~edges["bus"].isin(("yes", "designated"))
    ).to_numpy()
    assert restricted.any()  # the fixture carries real restrictions
    assert not forward[restricted].any() and not reverse[restricted].any()


def test_bus_graph_keeps_grade_separations_apart():
    # A street bridges another: their geometries share the coordinate
    # (100, 0) mid-edge, but no OSM node — never a junction.
    edge_rows = [
        {"points": [(0, 0), (100, 0), (200, 0)], "highway": "residential"},
        {"points": [(100, -100), (100, 0), (100, 100)], "highway": "residential"},
    ]
    nodes, edges = street_frames(edge_rows)
    graph = _map_match.bus_graph(nodes, edges, identity)
    start = graph.candidates(np.asarray([0.0, 0.0]))[0][0]
    end = graph.candidates(np.asarray([100.0, 100.0]))[0][0]
    assert graph.paths({(start, end): 250.0})[(start, end)] is None


def test_short_stub_cannot_fake_a_leg():
    # A 10 m stub sits inside both stops' snap tolerance while the
    # stops are 190 m apart: accepting it would record a near-zero
    # leg — the length floor refuses.
    stub = [((90, 0), (100, 0)), ((100, 0), (90, 0))]
    graph, index = graph_of(stub)
    stops = np.asarray([[0.0, 0.0], [190.0, 0.0]])
    assert _map_match.match_chain(graph, stops) is None


def test_coincident_consecutive_stops_refuse():
    graph, index = graph_of(BOTH_WAYS)
    stops = np.asarray([[0.0, 0.0], [0.0, 0.0], [100.0, 0.0]])
    assert _map_match.match_chain(graph, stops) is None


def test_match_chain_never_collapses_stops_onto_one_vertex():
    # Each stop sits exactly on its own vertex; adjacent candidate
    # sets overlap (60 m spacing, 100 m tolerance). The chain must
    # keep one distinct vertex per stop — no zero-length legs.
    hops = [((0, 0), (60, 0)), ((60, 0), (120, 0))]
    graph, index = graph_of(hops + [(b, a) for a, b in hops])
    stops = np.asarray([[0.0, 0.0], [60.0, 0.0], [120.0, 0.0]])
    segments = _map_match.match_chain(graph, stops)
    assert segments is not None
    assert [length for length, _ in segments] == [
        pytest.approx(60.0),
        pytest.approx(60.0),
    ]
