"""Footpath precompute from OSM walking networks."""

import random
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely

from cafein import streets

# The synthetic networks live around 24°E 60°N; coordinates are given in
# planar meters and converted with the local degree lengths. Designed
# distances hold to a small fraction of a second of walking, but durations
# round up conservatively, so an expected value may come out one second
# high — `assert_footpaths` allows exactly that.
DEG_PER_M_LON = 1 / 55_660
DEG_PER_M_LAT = 1 / 111_412


def lonlat(x, y):
    return (24.0 + x * DEG_PER_M_LON, 60.0 + y * DEG_PER_M_LAT)


def street_network(nodes, edges):
    """pyrosm-like (nodes, edges) frames from `{id: (x, y)}` and
    `(u, v, length[, path])` records."""
    node_frame = pd.DataFrame({"id": list(nodes)})
    rows = []
    for record in edges:
        u, v, length = record[:3]
        path = record[3] if len(record) > 3 else [nodes[u], nodes[v]]
        rows.append(
            {
                "u": u,
                "v": v,
                "length": float(length),
                "geometry": shapely.LineString([lonlat(*point) for point in path]),
            }
        )
    edge_frame = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    return node_frame, edge_frame


def stop(stop_id, x, y):
    lon, lat = lonlat(x, y)
    return (stop_id, lat, lon)


def footpaths(nodes, edges, stops, **options):
    node_frame, edge_frame = street_network(nodes, edges)
    edge_list = streets._network_footpaths(
        stops,
        node_frame,
        edge_frame,
        walking_speed_kmph=options.pop("walking_speed_kmph", 3.6),
        max_walking_time=options.pop("max_walking_time", 600.0),
        max_snap_distance=options.pop("max_snap_distance", 100.0),
    )
    assert not options
    return {(a, b): seconds for a, b, seconds, _ in edge_list}


def payload(nodes, edges, stops, **options):
    node_frame, edge_frame = street_network(nodes, edges)
    _, street = streets._network_streets(
        stops,
        node_frame,
        edge_frame,
        walking_speed_kmph=options.pop("walking_speed_kmph", 3.6),
        max_walking_time=options.pop("max_walking_time", 600.0),
        max_snap_distance=options.pop("max_snap_distance", 100.0),
    )
    assert not options
    return street


def assert_footpaths(result, expected):
    """`expected` maps stop pairs to designed walking seconds."""
    assert set(result) == set(expected)
    for pair, designed in expected.items():
        assert designed <= result[pair] <= designed + 1


STRAIGHT_STREET = {"A": (0, 0), "B": (400, 0)}


def test_stops_on_a_shared_edge_walk_along_it():
    # Both stops split the same 400 m edge; the footpath runs between the
    # snap points, not around via the endpoints.
    result = footpaths(
        STRAIGHT_STREET,
        [("A", "B", 400)],
        [stop("s1", 100, 0), stop("s2", 300, 0)],
    )
    assert_footpaths(result, {("s1", "s2"): 200, ("s2", "s1"): 200})


def test_offset_stops_connect_over_the_snap_distance():
    result = footpaths(
        STRAIGHT_STREET,
        [("A", "B", 400)],
        [stop("s1", 100, 0), stop("s2", 300, 30)],
    )
    assert 229 <= result[("s1", "s2")] <= 232


def test_a_stop_on_an_endpoint_reuses_the_node():
    result = footpaths(
        STRAIGHT_STREET,
        [("A", "B", 400)],
        [stop("s1", 0, 0), stop("s2", 300, 0)],
    )
    assert_footpaths(result, {("s1", "s2"): 300, ("s2", "s1"): 300})


def test_the_split_cost_follows_the_edge_length_not_the_geometry():
    # The edge's `length` says 800 m although its geometry spans 400 m;
    # split segments must redistribute the 800 m cost proportionally.
    result = footpaths(
        STRAIGHT_STREET,
        [("A", "B", 800)],
        [stop("s1", 100, 0), stop("s2", 300, 0)],
    )
    assert_footpaths(result, {("s1", "s2"): 400, ("s2", "s1"): 400})


def test_parallel_edges_keep_the_cheapest():
    # A duplicate slower connection between the same nodes must not be
    # summed into (or replace) the fast one.
    nodes = {"A": (0, 0), "B": (400, 0)}
    edges = [
        ("A", "B", 400),
        ("A", "B", 800, [(0, 0), (200, 100), (400, 0)]),
    ]
    result = footpaths(nodes, edges, [stop("s1", 0, 0), stop("s2", 400, 0)])
    assert_footpaths(result, {("s1", "s2"): 400, ("s2", "s1"): 400})


def test_faster_walking_shortens_footpaths():
    result = footpaths(
        STRAIGHT_STREET,
        [("A", "B", 400)],
        [stop("s1", 100, 0), stop("s2", 300, 0)],
        walking_speed_kmph=7.2,
    )
    assert_footpaths(result, {("s1", "s2"): 100, ("s2", "s1"): 100})


def test_transitive_closure_chains_footpaths_beyond_the_cutoff():
    result = footpaths(
        STRAIGHT_STREET,
        [("A", "B", 400)],
        [stop("s1", 0, 0), stop("s2", 200, 0), stop("s3", 400, 0)],
        max_walking_time=250.0,
    )
    # s1-s3 (400 s) exceeds the direct-search cutoff but is the chain of
    # two direct footpaths, so closure adds it.
    assert 200 <= result[("s1", "s2")] <= 201
    assert 200 <= result[("s2", "s3")] <= 201
    assert 400 <= result[("s1", "s3")] <= 402


def test_durations_round_up_conservatively():
    # Walking times are feasibility constraints: rounding down could let
    # routing catch a departure the walk actually misses. Only genuine
    # floating-point noise is tolerated. Meters stay unrounded.
    durations = np.array([[0.0, 10.4], [9.9999999, 0.0]])
    edges = streets._edge_list(np.array(["a", "b"], dtype=object), durations, 1.25)
    assert {(a, b): s for a, b, s, _ in edges} == {("a", "b"): 11, ("b", "a"): 10}
    meters = {(a, b): m for a, b, _, m in edges}
    assert meters[("a", "b")] == pytest.approx(13.0)
    assert meters[("b", "a")] == pytest.approx(12.5, rel=1e-6)


def test_out_of_range_build_options_are_rejected():
    stops = [stop("s1", 100, 0), stop("s2", 300, 0)]
    for options in [
        {"walking_speed_kmph": float("nan")},
        {"walking_speed_kmph": float("inf")},
        {"walking_speed_kmph": 0.0},
        {"max_walking_time": float("nan")},
        {"max_walking_time": -1.0},
        {"max_snap_distance": float("inf")},
        {"max_snap_distance": -1.0},
    ]:
        with pytest.raises(ValueError, match="finite"):
            footpaths(STRAIGHT_STREET, [("A", "B", 400)], stops, **options)


def test_oversized_stop_sets_are_rejected(monkeypatch):
    # The dense stop-by-stop matrices grow quadratically; builds beyond
    # the ceiling fail fast instead of exhausting memory.
    monkeypatch.setattr(streets, "MAX_FOOTPATH_STOPS", 1)
    with pytest.raises(ValueError, match="snapped stops exceed"):
        footpaths(
            STRAIGHT_STREET,
            [("A", "B", 400)],
            [stop("s1", 100, 0), stop("s2", 300, 0)],
        )


def test_distant_stops_are_left_out():
    with pytest.warns(UserWarning, match="farther than"):
        result = footpaths(
            STRAIGHT_STREET,
            [("A", "B", 400)],
            [stop("s1", 100, 0), stop("s2", 300, 0), stop("far", 200, 500)],
        )
    assert_footpaths(result, {("s1", "s2"): 200, ("s2", "s1"): 200})


def test_stops_without_coordinates_are_left_out():
    with pytest.warns(UserWarning, match="no coordinates"):
        result = footpaths(
            STRAIGHT_STREET,
            [("A", "B", 400)],
            [stop("s1", 100, 0), stop("s2", 300, 0), ("missing", None, None)],
        )
    assert_footpaths(result, {("s1", "s2"): 200, ("s2", "s1"): 200})


def test_disconnected_components_get_local_footpaths_only():
    # An island's stops walk between each other but never to the mainland:
    # disconnected components stay in the graph and Dijkstra leaves
    # cross-component pairs unreachable.
    nodes = {"A": (0, 0), "B": (400, 0), "I1": (0, 60), "I2": (400, 60)}
    edges = [("A", "B", 400), ("I1", "I2", 400)]
    result = footpaths(
        nodes,
        edges,
        [stop("m1", 100, 0), stop("i1", 100, 60), stop("i2", 300, 60)],
    )
    assert 200 <= result[("i1", "i2")] <= 201
    assert ("m1", "i1") not in result
    assert ("i1", "m1") not in result


def test_the_street_payload_mirrors_the_walking_network():
    vertex_count, edge_list, offsets, lons, lats, links = payload(
        STRAIGHT_STREET,
        [("A", "B", 400)],
        [stop("s1", 100, 0), stop("s2", 300, 30)],
    )
    assert vertex_count == 2
    assert edge_list == [(0, 1, 400.0)]
    assert offsets == [0, 2]
    assert lons == pytest.approx([24.0, 24.0 + 400 * DEG_PER_M_LON])
    assert lats == [60.0, 60.0]
    assert [link[:2] for link in links] == [("s1", 0), ("s2", 0)]
    fractions = [link[2] for link in links]
    assert fractions == pytest.approx([0.25, 0.75], rel=1e-3)
    connectors = [link[3] for link in links]
    assert connectors[0] == pytest.approx(0.0, abs=0.1)
    assert connectors[1] == pytest.approx(30.0, rel=0.01)


def test_unsnapped_stops_get_no_street_links():
    with pytest.warns(UserWarning, match="farther than"):
        *_, links = payload(
            STRAIGHT_STREET,
            [("A", "B", 400)],
            [stop("s1", 100, 0), stop("far", 200, 500)],
        )
    assert [link[0] for link in links] == ["s1"]


def test_an_empty_network_yields_an_empty_payload():
    empty_edges = gpd.GeoDataFrame(
        {"u": [], "v": [], "length": []},
        geometry=gpd.GeoSeries([], crs="EPSG:4326"),
    )
    footpath_list, street = streets._network_streets(
        [stop("s1", 100, 0)],
        pd.DataFrame({"id": []}),
        empty_edges,
        walking_speed_kmph=3.6,
        max_walking_time=600.0,
        max_snap_distance=100.0,
    )
    assert footpath_list == []
    assert street == (0, [], [0], [], [], [])


@pytest.fixture(scope="session")
def helsinki_streets(helsinki_gtfs, kantakaupunki_pbf):
    with (
        zipfile.ZipFile(helsinki_gtfs) as archive,
        archive.open("stops.txt") as stops_file,
    ):
        frame = pd.read_csv(stops_file, dtype={"stop_id": str})
    triples = list(
        zip(
            frame["stop_id"],
            frame["stop_lat"].astype(float),
            frame["stop_lon"].astype(float),
        )
    )
    with pytest.warns(UserWarning):
        return streets.walking_streets(str(kantakaupunki_pbf), triples)


@pytest.fixture(scope="session")
def helsinki_footpaths(helsinki_streets):
    return helsinki_streets[0]


def test_helsinki_footpaths_cover_the_extract(helsinki_footpaths):
    # The extract covers central Helsinki only: roughly 1400 of the 8305
    # stops snap onto its walking network, and closure connects the dense
    # center almost completely.
    origins = {from_stop for from_stop, _, _, _ in helsinki_footpaths}
    assert 1_330 <= len(origins) <= 1_440
    assert 1_100_000 <= len(helsinki_footpaths) <= 1_300_000


def test_helsinki_footpaths_pin_known_pairs(helsinki_footpaths):
    lookup = {(a, b): seconds for a, b, seconds, _ in helsinki_footpaths}
    # Kamppi metro platforms sit next to each other; the westbound
    # platform is a short walk from the Kamppi street stops.
    assert lookup[("1040602", "1040601")] == 4
    assert lookup[("1040602", "1040280")] == 20
    assert lookup[("1000102", "1040280")] == 22
    # Meters are the same walks unrounded, at the default 1 m/s.
    meters = {(a, b): m for a, b, _, m in helsinki_footpaths}
    assert 19 <= meters[("1040602", "1040280")] <= 20


def test_helsinki_footpaths_are_symmetric(helsinki_footpaths):
    lookup = {(a, b): seconds for a, b, seconds, _ in helsinki_footpaths}
    assert all(lookup.get((b, a)) == seconds for (a, b), seconds in lookup.items())


def test_helsinki_street_network_covers_the_extract(helsinki_streets):
    vertex_count, edge_list, offsets, lons, lats, links = helsinki_streets[1]
    assert vertex_count > 10_000
    assert len(edge_list) > 10_000
    assert len(offsets) == len(edge_list) + 1
    assert offsets[0] == 0
    assert offsets[-1] == len(lons) == len(lats)
    # Every stop with footpaths snapped, so it also carries a street link.
    origins = {from_stop for from_stop, _, _, _ in helsinki_streets[0]}
    assert {link[0] for link in links} >= origins
    assert all(0 <= link[1] < len(edge_list) for link in links)
    assert all(0.0 <= link[2] <= 1.0 for link in links)
    assert all(0.0 <= link[3] <= 100.0 for link in links)


def test_helsinki_footpaths_are_transitively_closed(helsinki_footpaths):
    lookup = {(a, b): seconds for a, b, seconds, _ in helsinki_footpaths}
    by_origin = {}
    for from_stop, to_stop, seconds, _ in helsinki_footpaths:
        by_origin.setdefault(from_stop, []).append((to_stop, seconds))
    generator = random.Random(0)
    for from_stop in generator.sample(sorted(by_origin), 100):
        for middle, first_leg in generator.sample(
            by_origin[from_stop], min(5, len(by_origin[from_stop]))
        ):
            for to_stop, second_leg in generator.sample(
                by_origin[middle], min(5, len(by_origin[middle]))
            ):
                if to_stop == from_stop:
                    continue
                chained = lookup[(from_stop, to_stop)]
                # Rounding both legs may undercut the chain by a second.
                assert chained <= first_leg + second_leg + 1
