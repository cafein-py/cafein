"""Tier 3 end to end: the opt-in API and withheld-shape accuracy."""

import io
import pathlib
import statistics
import zipfile

import numpy as np
import pandas as pd
import pytest

from cafein import geometry

GTFS = pathlib.Path(__file__).parent / "data" / "helsinki_gtfs.zip"
TRANSIT = pathlib.Path(__file__).parent / "data" / "helsinki-transit.osm.pbf"


def test_osm_tiers_require_the_extract():
    with pytest.raises(ValueError, match="osm_pbf"):
        geometry.trip_distances(str(GTFS), osm_tiers=True)


def test_unknown_mode_names_reject():
    with pytest.raises(ValueError, match="segway"):
        geometry.trip_distances(
            str(GTFS), osm_pbf=str(TRANSIT), osm_tiers=("tram", "segway")
        )


@pytest.fixture(scope="module")
def withheld_feed(tmp_path_factory):
    """The GTFS fixture with tiers 1–2 withheld: no ``shapes.txt`` and
    no ``shape_dist_traveled``, so distances rest on tiers 3 and 5."""
    doctored = tmp_path_factory.mktemp("osm_tiers") / "helsinki_no_shapes.zip"
    with (
        zipfile.ZipFile(GTFS) as src,
        zipfile.ZipFile(doctored, "w", zipfile.ZIP_DEFLATED) as out,
    ):
        for name in ("agency.txt", "routes.txt", "trips.txt", "stops.txt"):
            out.writestr(name, src.read(name))
        stop_times = pd.read_csv(io.BytesIO(src.read("stop_times.txt")), dtype=str)
        stop_times = stop_times.drop(columns=["shape_dist_traveled"])
        out.writestr("stop_times.txt", stop_times.to_csv(index=False))
    return doctored


@pytest.fixture(scope="module")
def pattern_trips():
    """One representative trip per distinct tram (route, stop sequence),
    plus one bus trip pinning per-mode enablement."""
    with zipfile.ZipFile(GTFS) as src:
        routes = pd.read_csv(io.BytesIO(src.read("routes.txt")), dtype=str)
        trips = pd.read_csv(io.BytesIO(src.read("trips.txt")), dtype=str)
        stop_times = pd.read_csv(
            io.BytesIO(src.read("stop_times.txt")),
            dtype=str,
            usecols=["trip_id", "stop_id", "stop_sequence"],
        )
    tram_routes = set(routes.loc[routes["route_type"] == "0", "route_id"])
    tram_trips = trips[trips["route_id"].isin(tram_routes)]
    of_trams = stop_times[stop_times["trip_id"].isin(tram_trips["trip_id"])].copy()
    of_trams["stop_sequence"] = of_trams["stop_sequence"].astype(int)
    sequences = (
        of_trams.sort_values(["trip_id", "stop_sequence"])
        .groupby("trip_id")["stop_id"]
        .apply(tuple)
    )
    route_of = tram_trips.set_index("trip_id")["route_id"]
    representatives = {}
    for trip_id, sequence in sequences.items():
        representatives.setdefault((route_of[trip_id], sequence), trip_id)
    bus_routes = set(routes.loc[routes["route_type"] == "701", "route_id"])
    bus_trip = trips.loc[trips["route_id"].isin(bus_routes), "trip_id"].iloc[0]
    return set(representatives.values()), bus_trip


@pytest.fixture(scope="module")
def tier3_run(withheld_feed, pattern_trips):
    tram_trips, bus_trip = pattern_trips
    results, (polylines, trip_rows) = geometry.trip_distances(
        str(withheld_feed),
        include=tram_trips | {bus_trip},
        geometries=True,
        osm_pbf=str(TRANSIT),
        osm_tiers=("tram",),
    )
    return results, polylines, trip_rows


def test_tram_patterns_resolve_at_tier_3(tier3_run, pattern_trips):
    results, _, _ = tier3_run
    tiers = {tier for _, _, tier in results}
    assert tiers <= {geometry.OSM_RELATION, geometry.CROW_FLY}
    resolved = [row for row in results if row[2] == geometry.OSM_RELATION]
    assert len(resolved) >= 10
    _, bus_trip = pattern_trips
    bus_tier = next(tier for trip, _, tier in results if trip == bus_trip)
    assert bus_tier == geometry.CROW_FLY  # tram-only enablement


def test_tier_3_distances_match_the_withheld_truth(tier3_run):
    results, _, _ = tier3_run
    truth = {
        trip: cumulative[-1]
        for trip, cumulative, tier in geometry.trip_distances(
            str(GTFS), include={row[0] for row in results}
        )
        if tier == geometry.SHAPE_DIST
    }
    errors = [
        abs(cumulative[-1] - truth[trip]) / truth[trip]
        for trip, cumulative, tier in results
        if tier == geometry.OSM_RELATION and trip in truth
    ]
    assert len(errors) >= 10
    assert statistics.median(errors) <= 0.05
    assert max(errors) <= 0.10


def test_tier_3_geometries_are_the_relation_lines(tier3_run):
    results, polylines, trip_rows = tier3_run
    resolved = {trip for trip, _, tier in results if tier == geometry.OSM_RELATION}
    positions_of = {trip: (index, positions) for trip, index, positions in trip_rows}
    for trip in resolved:
        index, positions = positions_of[trip]
        lons, lats, measures = polylines[index]
        # The relation line is denser than the stop chain and the stop
        # positions run monotonically along it.
        assert len(lons) == len(lats) == len(measures) > len(positions)
        assert (np.diff(positions) > 0).all()
        assert 0 <= positions[0] and positions[-1] <= measures[-1] + 1e-6


def build_ladder(
    way_lines,
    platform_points,
    way_role="",
    way_tags=None,
    more=(),
    route="tram",
    modes=("tram",),
):
    """A ladder with one injected relation (ref ``9``, id 77) built
    from explicit way polylines and platform coordinates — plus
    optional additional relations."""
    import shapely

    from cafein import _osm_tiers, _relations

    ways = [
        _relations.RelationMember(
            kind="way",
            id=100 + index,
            role=way_role,
            geometry=shapely.LineString(points),
            tags=dict(way_tags or {}),
        )
        for index, points in enumerate(way_lines)
    ]
    platforms = [
        _relations.RelationMember(
            kind="node",
            id=200 + index,
            role="platform",
            geometry=shapely.Point(*point),
            tags={},
        )
        for index, point in enumerate(platform_points)
    ]
    relation = _relations.RouteRelation(
        id=77,
        route=route,
        ref="9",
        name=None,
        operator=None,
        network=None,
        members=tuple(platforms + ways),
    )
    ladder = _osm_tiers.RelationLadder("unused.osm.pbf", frozenset(modes))
    ladder._relations = [relation, *more]
    return ladder


def synthetic_ladder(way_role="", way_tags=None, lat=60.1, lon0=24.900):
    """The straight east–west specialization: three dense member ways
    along one latitude and four platforms every ~550 m."""
    lons = [lon0 + i * 0.0025 for i in range(13)]
    way_lines = [[(lon, lat) for lon in lons[start : start + 5]] for start in (0, 4, 8)]
    platforms = [(lon0 + i * 0.010, lat) for i in range(4)]
    return build_ladder(way_lines, platforms, way_role=way_role, way_tags=way_tags)


ROUTE_9 = ("r9", "9", None, ())
STOPS_9 = ("a", "b", "c", "d")


def west_east_stops(lat=60.1, lon0=24.900):
    return np.asarray([[lat, lon0 + i * 0.010] for i in range(4)])


def resolve(ladder, latlon, stop_ids=STOPS_9, route=ROUTE_9):
    crow = geometry._crow_fly_cumulative(latlon)
    return ladder.resolve(route, 0, stop_ids, latlon, crow[-1])


def test_reversed_relation_resolves_in_pattern_direction():
    # The pattern travels the member sequence backward: the reversed
    # orientation must win AND the line must validate in the pattern's
    # direction, with the polyline identity carrying the orientation.
    ladder = synthetic_ladder()
    latlon = np.asarray([[60.1, 24.930 - i * 0.010] for i in range(4)])
    resolved = resolve(ladder, latlon)
    assert resolved is not None
    cumulative, identity, along = resolved
    assert identity[1:] == (77, True)
    assert (np.diff(cumulative) > 0).all()
    lons, lats, measures = ladder.polyline(identity)
    assert lons[0] > lons[-1]  # travels east to west
    assert measures[-1] == pytest.approx(cumulative[-1], rel=0.01)


def test_reversal_of_directed_members_refuses():
    # One-way members (or verified-direction rings) cannot legally be
    # travelled backward: the reversed orientation refuses instead of
    # reversing the line wholesale.
    ladder = synthetic_ladder(way_tags={"oneway": "yes"})
    backward = np.asarray([[60.1, 24.930 - i * 0.010] for i in range(4)])
    assert resolve(ladder, backward) is None
    # The forward direction of the same relation still resolves (a
    # distinct stop sequence: ids pin coordinates within one feed).
    assert resolve(ladder, west_east_stops(), stop_ids=("w", "x", "y", "z")) is not None


def test_bus_pattern_never_borrows_a_trolleybus_relation():
    # Same ref, perfect platforms — but the relation is a trolleybus
    # route and the pattern an ordinary bus: the hard mode filter
    # refuses (and the same-mode control resolves).
    lons = [24.900 + i * 0.0025 for i in range(13)]
    way_lines = [[(lon, 60.1) for lon in lons[s : s + 5]] for s in (0, 4, 8)]
    platforms = [(24.900 + i * 0.010, 60.1) for i in range(4)]
    trolley = build_ladder(way_lines, platforms, route="trolleybus", modes=("bus",))
    latlon = west_east_stops()
    crow = geometry._crow_fly_cumulative(latlon)
    assert trolley.resolve(ROUTE_9, 3, STOPS_9, latlon, crow[-1]) is None
    bus = build_ladder(way_lines, platforms, route="bus", modes=("bus",))
    assert bus.resolve(ROUTE_9, 3, STOPS_9, latlon, crow[-1]) is not None


def test_bidirectional_way_stored_against_boarding_order_resolves():
    # One bidirectional member way stored west→east while the
    # boarding members (and the pattern) run east→west: the stored
    # coordinate orientation is arbitrary, so the resolver tries the
    # other orientation and validates it.
    lons = [24.900 + i * 0.0025 for i in range(13)]
    way_lines = [[(lon, 60.1) for lon in lons]]
    platforms = [(24.930 - i * 0.010, 60.1) for i in range(4)]
    ladder = build_ladder(way_lines, platforms)
    latlon = np.asarray([[60.1, 24.930 - i * 0.010] for i in range(4)])
    resolved = resolve(ladder, latlon)
    assert resolved is not None
    cumulative, identity, along = resolved
    assert identity[1:] == (77, True)
    assert (np.diff(cumulative) > 0).all()


def test_implied_motorway_oneway_refuses_reversal():
    # An untagged motorway member implies oneway=yes: the wholesale
    # reversal of the stitched line refuses just like explicit tags.
    ladder = synthetic_ladder(way_tags={"highway": "motorway"})
    backward = np.asarray([[60.1, 24.930 - i * 0.010] for i in range(4)])
    assert resolve(ladder, backward) is None


def test_way_roles_refuse_resolution():
    # PTv1 forward/backward way roles: membership is direction
    # dependent, so tier 3 refuses instead of stitching holes.
    ladder = synthetic_ladder(way_role="forward")
    assert resolve(ladder, west_east_stops()) is None


def test_resolution_cache_is_feed_namespaced():
    # A second feed reusing the same route and stop ids must not
    # inherit the first feed's resolution.
    ladder = synthetic_ladder()
    assert resolve(ladder, west_east_stops()) is not None
    ladder.begin_feed(1)
    shifted = west_east_stops() + [0.02, 0.0]  # ~2 km north: nothing snaps
    assert resolve(ladder, shifted) is None


def test_feeds_project_in_their_own_utm_zone():
    # Feed 2 sits ~15° west, in another UTM zone: its resolution must
    # re-estimate the projection, not reuse feed 1's Helsinki CRS.
    ladder = synthetic_ladder()
    first = resolve(ladder, west_east_stops())
    assert first is not None
    ladder.begin_feed(1)
    ladder._relations = [synthetic_ladder(lon0=10.0)._relations[0]]
    ladder._by_mode = {}
    second = resolve(ladder, west_east_stops(lon0=10.0))
    assert second is not None
    crow = geometry._crow_fly_cumulative(west_east_stops(lon0=10.0))[-1]
    assert second[0][-1] == pytest.approx(crow, rel=0.02)
    assert second[1][0] != first[1][0]  # a different CRS key


def test_sparse_line_fails_the_density_gate():
    # Two-vertex member ways: the stitched line is no denser than the
    # stop sequence — rejected exactly like a stop-chain shape.
    way_lines = [
        [(24.900, 60.1), (24.910, 60.1)],
        [(24.910, 60.1), (24.920, 60.1)],
        [(24.920, 60.1), (24.930, 60.1)],
    ]
    platforms = [(24.900 + i * 0.010, 60.1) for i in range(4)]
    ladder = build_ladder(way_lines, platforms)
    assert resolve(ladder, west_east_stops()) is None


def test_far_stops_fail_the_snap_gate():
    # The relation's track runs ~200 m north of every stop: platforms
    # sit at the stops so matching passes, but the stops do not lie on
    # the geometry.
    lons = [24.900 + i * 0.0025 for i in range(13)]
    way_lines = [[(lon, 60.1018) for lon in lons[s : s + 5]] for s in (0, 4, 8)]
    platforms = [(24.900 + i * 0.010, 60.1) for i in range(4)]
    ladder = build_ladder(way_lines, platforms)
    assert resolve(ladder, west_east_stops()) is None


def test_out_of_order_stops_fail_the_monotone_gate():
    # One adjacent stop pair swapped in the pattern (2 edits over 8
    # stops = 0.25: still accepted by the matcher) — but the along-line
    # positions are no longer monotone, so the gate refuses.
    offsets = (0.0, 0.004, 0.008, 0.012, 0.016, 0.020, 0.024, 0.028)
    platforms = [(24.900 + offset, 60.1) for offset in offsets]
    lons = [24.900 + i * 0.0025 for i in range(13)]
    way_lines = [[(lon, 60.1) for lon in lons[s : s + 5]] for s in (0, 4, 8)]
    ladder = build_ladder(way_lines, platforms)
    pattern_lons = [24.900 + offset for offset in offsets]
    pattern_lons[4], pattern_lons[5] = pattern_lons[5], pattern_lons[4]
    latlon = np.asarray([[60.1, lon] for lon in pattern_lons])
    stop_ids = tuple(f"s{i}" for i in range(8))
    assert resolve(ladder, latlon, stop_ids=stop_ids) is None


def test_implausible_length_fails_the_ratio_gate():
    # A U-shaped line: ~3.3 km of track between stops ~220 m apart —
    # beyond the plausibility band, so tier 3 refuses.
    east = [(24.900 + i * 0.0025, 60.1) for i in range(13)]
    north = [(24.930, 60.1 + i * 0.0005) for i in range(1, 5)]
    west = [(24.930 - i * 0.0025, 60.102) for i in range(1, 13)]
    line = east + north + west
    third = len(line) // 3
    way_lines = [line[: third + 1], line[third : 2 * third + 1], line[2 * third :]]
    platforms = [(24.900, 60.1), (24.900, 60.102)]
    ladder = build_ladder(way_lines, platforms)
    latlon = np.asarray([[60.1, 24.900], [60.102, 24.900]])
    assert resolve(ladder, latlon, stop_ids=("a", "b")) is None


def test_disabled_tiers_change_nothing(withheld_feed, pattern_trips):
    tram_trips, _ = pattern_trips
    subset = set(sorted(tram_trips)[:5])
    plain = geometry.trip_distances(str(withheld_feed), include=subset)
    disabled = geometry.trip_distances(
        str(withheld_feed), include=subset, osm_pbf=str(TRANSIT), osm_tiers=False
    )
    assert disabled == plain
    assert {tier for _, _, tier in plain} == {geometry.CROW_FLY}
