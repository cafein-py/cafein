"""Trip-distance ladder from GTFS shapes."""

import collections
import warnings
import zipfile

import pytest

from cafein import geometry

# Stops sit on the 60°N parallel, 0.01° of longitude (~556.6 m) apart:
# crow-fly hops are ~556.6 m and the three-stop total is ~1113 m.
STOPS = [("A", 60.0, 24.0), ("B", 60.0, 24.01), ("C", 60.0, 24.02)]
CROW_TOTAL = 1113.2


def gtfs_zip(
    path,
    *,
    shape_dist=None,
    shape_points=None,
    raw_shape_rows=None,
    route_type=3,
    drop_coordinates_of=None,
    duplicate_first_stop=False,
):
    """A one-trip feed riding A→B→C, with optional shape data."""
    stop_rows = [
        f"{sid},{sid},," if sid == drop_coordinates_of else f"{sid},{sid},{lat},{lon}"
        for sid, lat, lon in STOPS
    ]
    if duplicate_first_stop:
        stop_rows.append(f"{STOPS[0][0]},duplicate,0.0,0.0")
    stop_time_rows = []
    for position, (sid, _, _) in enumerate(STOPS):
        time = f"08:{position:02d}:00"
        dist = "" if shape_dist is None else f",{shape_dist[position]}"
        stop_time_rows.append(f"T1,{time},{time},{sid},{position + 1}{dist}")
    dist_header = "" if shape_dist is None else ",shape_dist_traveled"
    tables = {
        "agency.txt": [
            "agency_id,agency_name,agency_url,agency_timezone",
            "A,Test,http://example.com,Europe/Helsinki",
        ],
        "stops.txt": ["stop_id,stop_name,stop_lat,stop_lon"] + stop_rows,
        "routes.txt": [
            "route_id,route_short_name,route_type",
            f"R1,1,{route_type}",
        ],
        "trips.txt": [
            "route_id,service_id,trip_id,shape_id",
            "R1,SV,T1,"
            + ("S1" if shape_points is not None or raw_shape_rows is not None else ""),
        ],
        "stop_times.txt": [
            "trip_id,arrival_time,departure_time,stop_id,stop_sequence" + dist_header
        ]
        + stop_time_rows,
        "calendar.txt": [
            "service_id,monday,tuesday,wednesday,thursday,friday,saturday,"
            "sunday,start_date,end_date",
            "SV,1,1,1,1,1,1,1,20220101,20221231",
        ],
    }
    if raw_shape_rows is not None:
        tables["shapes.txt"] = [
            "shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence"
        ] + raw_shape_rows
    elif shape_points is not None:
        tables["shapes.txt"] = [
            "shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence"
        ] + [
            f"S1,{lat},{lon},{sequence}"
            for sequence, (lat, lon) in enumerate(shape_points)
        ]
    with zipfile.ZipFile(path, "w") as archive:
        for name, lines in tables.items():
            archive.writestr(name, "\n".join(lines) + "\n")
    return path


DENSE_SHAPE = [(60.0, 24.0 + 0.0025 * step) for step in range(9)]
"""Nine points along the stops' parallel — denser than the three stops."""


def single(rows):
    assert len(rows) == 1
    trip_id, cumulative, tier = rows[0]
    assert trip_id == "T1"
    return cumulative, tier


def test_geometries_carry_the_stop_chain_without_a_shape(tmp_path):
    feed = gtfs_zip(tmp_path / "feed.zip")
    rows, (polylines, trips) = geometry.trip_distances(feed, geometries=True)
    assert len(rows) == 1
    assert len(polylines) == 1
    lons, lats, measures = polylines[0]
    assert lons == [24.0, 24.01, 24.02]
    assert lats == [60.0, 60.0, 60.0]
    assert measures[0] == 0.0
    assert measures[-1] == pytest.approx(CROW_TOTAL, rel=0.01)
    trip_id, polyline, positions = trips[0]
    assert (trip_id, polyline) == ("T1", 0)
    # Chain positions are the chain's own measures: stops sit on vertices.
    assert positions == measures


def test_geometries_use_the_shape_when_stops_lie_on_it(tmp_path):
    feed = gtfs_zip(tmp_path / "feed.zip", shape_points=DENSE_SHAPE)
    rows, (polylines, trips) = geometry.trip_distances(feed, geometries=True)
    lons, lats, measures = polylines[0]
    assert len(lons) == len(DENSE_SHAPE)
    assert lons[0] == 24.0 and lons[-1] == pytest.approx(24.02)
    assert measures == sorted(measures)
    trip_id, polyline, positions = trips[0]
    assert polyline == 0
    # The stops locate at the chain's crow-fly spacing along the shape.
    assert positions[0] == pytest.approx(0.0, abs=1.0)
    assert positions[1] == pytest.approx(measures[-1] / 2, rel=0.01)
    assert positions[2] == pytest.approx(measures[-1], rel=0.01)


def test_geometries_dedup_polylines_across_trips(tmp_path):
    # Two trips of one shape and stop sequence share one polyline.
    feed = gtfs_zip(tmp_path / "feed.zip", shape_points=DENSE_SHAPE)
    with zipfile.ZipFile(feed, "a") as archive:
        with archive.open("trips.txt") as member:
            lines = member.read().decode().splitlines()
        with archive.open("stop_times.txt") as member:
            stop_times = member.read().decode().splitlines()
        archive.writestr("trips.txt", "\n".join(lines + ["R1,SV,T2,S1"]) + "\n")
        second = [line.replace("T1,", "T2,") for line in stop_times[1:]]
        archive.writestr("stop_times.txt", "\n".join(stop_times + second) + "\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows, (polylines, trips) = geometry.trip_distances(feed, geometries=True)
    assert len(rows) == 2
    assert len(polylines) == 1
    assert {trip_id for trip_id, _, _ in trips} == {"T1", "T2"}
    assert {polyline for _, polyline, _ in trips} == {0}


def test_valid_shape_dist_in_meters_is_used_directly(tmp_path):
    feed = gtfs_zip(tmp_path / "feed.zip", shape_dist=[0, 700, 1400])
    cumulative, tier = single(geometry.trip_distances(feed))
    assert tier == geometry.SHAPE_DIST
    assert cumulative == [0, 700, 1400]


def test_kilometer_shape_dist_is_rescaled(tmp_path):
    feed = gtfs_zip(tmp_path / "feed.zip", shape_dist=[0, 0.7, 1.4])
    cumulative, tier = single(geometry.trip_distances(feed))
    assert tier == geometry.SHAPE_DIST
    assert cumulative == pytest.approx([0, 700, 1400])


def test_decreasing_shape_dist_falls_to_linear_referencing(tmp_path):
    feed = gtfs_zip(
        tmp_path / "feed.zip", shape_dist=[0, 800, 700], shape_points=DENSE_SHAPE
    )
    cumulative, tier = single(geometry.trip_distances(feed))
    assert tier == geometry.SHAPE_LINREF
    assert cumulative == pytest.approx([0, 556.6, 1113.2], rel=0.01)


def test_non_numeric_shape_dist_fails_the_tier(tmp_path):
    feed = gtfs_zip(tmp_path / "feed.zip", shape_dist=[0, "junk", 1400])
    _, tier = single(geometry.trip_distances(feed))
    assert tier == geometry.CROW_FLY


def test_implausible_shape_dist_totals_fail_the_tier(tmp_path):
    # A total twenty times the crow-fly length is neither meters nor km.
    feed = gtfs_zip(tmp_path / "feed.zip", shape_dist=[0, 10_000, 22_000])
    _, tier = single(geometry.trip_distances(feed))
    assert tier == geometry.CROW_FLY


def test_shapes_denser_than_the_stops_are_linear_referenced(tmp_path):
    feed = gtfs_zip(tmp_path / "feed.zip", shape_points=DENSE_SHAPE)
    cumulative, tier = single(geometry.trip_distances(feed))
    assert tier == geometry.SHAPE_LINREF
    assert cumulative == pytest.approx([0, 556.6, 1113.2], rel=0.01)


def test_sparse_shapes_fail_to_crow_fly(tmp_path):
    # Three shape points for three stops: stop-to-stop straight lines
    # dressed up as a shape.
    feed = gtfs_zip(
        tmp_path / "feed.zip",
        shape_points=[(lat, lon) for _, lat, lon in STOPS],
    )
    _, tier = single(geometry.trip_distances(feed))
    assert tier == geometry.CROW_FLY


def test_stops_far_from_the_shape_fail_to_crow_fly(tmp_path):
    # The shape runs ~330 m north of the stops.
    offset = [(lat + 0.003, lon) for lat, lon in DENSE_SHAPE]
    feed = gtfs_zip(tmp_path / "feed.zip", shape_points=offset)
    _, tier = single(geometry.trip_distances(feed))
    assert tier == geometry.CROW_FLY


def test_crow_fly_scales_by_the_mode_detour(tmp_path):
    bus, _ = single(geometry.trip_distances(gtfs_zip(tmp_path / "bus.zip")))
    rail, tier = single(
        geometry.trip_distances(gtfs_zip(tmp_path / "rail.zip", route_type=2))
    )
    assert tier == geometry.CROW_FLY
    assert bus[-1] == pytest.approx(CROW_TOTAL * 1.4, rel=0.01)
    assert rail[-1] == pytest.approx(CROW_TOTAL * 1.2, rel=0.01)


def test_shape_points_sort_numerically_and_skip_unorderable_rows(tmp_path):
    # Twelve points whose 1-based sequences misorder under string sorting
    # ("10" < "2"), written in reversed row order, plus one row without a
    # sequence value that cannot be ordered at all.
    points = [(60.0, 24.0 + 0.002 * step) for step in range(12)]
    rows = [
        f"S1,{lat},{lon},{sequence + 1}" for sequence, (lat, lon) in enumerate(points)
    ]
    rows.append("S1,61.0,25.0,")
    feed = gtfs_zip(tmp_path / "feed.zip", raw_shape_rows=list(reversed(rows)))
    cumulative, tier = single(geometry.trip_distances(feed))
    assert tier == geometry.SHAPE_LINREF
    assert cumulative == pytest.approx([0, 556.6, 1113.2], rel=0.01)


def test_stops_without_coordinates_raise_a_clear_error(tmp_path):
    feed = gtfs_zip(tmp_path / "feed.zip", drop_coordinates_of="B")
    with pytest.raises(ValueError, match="without coordinates.*B"):
        geometry.trip_distances(feed)


def test_duplicate_stop_rows_use_the_first_occurrence(tmp_path):
    feed = gtfs_zip(tmp_path / "feed.zip", duplicate_first_stop=True)
    cumulative, tier = single(geometry.trip_distances(feed))
    assert tier == geometry.CROW_FLY
    assert cumulative[-1] == pytest.approx(CROW_TOTAL * 1.4, rel=0.01)


def test_unreadable_shapes_degrade_to_crow_fly(tmp_path):
    # The appended shapes.txt (a duplicate member wins on read) is missing
    # its coordinate columns entirely; the shape tier degrades with a
    # warning instead of failing the feed.
    feed = gtfs_zip(tmp_path / "feed.zip", raw_shape_rows=["S1,60.0,24.0,1"])
    with zipfile.ZipFile(feed, "a") as archive:
        archive.writestr("shapes.txt", "shape_id,shape_pt_sequence\nS1,1\n")
    with pytest.warns(UserWarning, match="shape tier"):
        _, tier = single(geometry.trip_distances(feed))
    assert tier == geometry.CROW_FLY


def test_tier_one_feeds_never_load_shapes(tmp_path):
    # Valid shape_dist_traveled resolves at tier 1, so the (broken)
    # shapes.txt is never even opened.
    feed = gtfs_zip(
        tmp_path / "feed.zip",
        shape_dist=[0, 700, 1400],
        raw_shape_rows=["S1,60.0,24.0,1"],
    )
    with zipfile.ZipFile(feed, "a") as archive:
        archive.writestr("shapes.txt", "shape_id,shape_pt_sequence\nS1,1\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cumulative, tier = single(geometry.trip_distances(feed))
    assert tier == geometry.SHAPE_DIST
    assert cumulative == [0, 700, 1400]


def test_merged_feeds_qualify_trip_ids(tmp_path):
    first = gtfs_zip(tmp_path / "first.zip")
    second = gtfs_zip(tmp_path / "second.zip")
    rows = geometry.trip_distances([first, second])
    assert [trip_id for trip_id, _, _ in rows] == ["0:T1", "1:T1"]


def test_include_limits_the_ladder_to_given_trips(tmp_path):
    feed = gtfs_zip(tmp_path / "feed.zip")
    assert geometry.trip_distances(feed, include={"other"}) == []
    assert len(geometry.trip_distances(feed, include={"T1"})) == 1
    first = gtfs_zip(tmp_path / "first.zip")
    second = gtfs_zip(tmp_path / "second.zip")
    rows = geometry.trip_distances([first, second], include={"1:T1"})
    assert [trip_id for trip_id, _, _ in rows] == ["1:T1"]


def test_helsinki_distances_come_from_shape_dist(helsinki_gtfs):
    rows = geometry.trip_distances(str(helsinki_gtfs))
    tiers = collections.Counter(tier for _, _, tier in rows)
    # HSL ships complete, valid shape_dist_traveled (in kilometers).
    assert tiers == {geometry.SHAPE_DIST: 195_351}
    lookup = {trip_id: cumulative for trip_id, cumulative, _ in rows}
    korso_to_kapyla = lookup["3001K_20220222_S1_2_0831"]
    # Positions 2 (Korso) and 12 (Käpylä); 16.786 km in the raw tables.
    assert korso_to_kapyla[12] - korso_to_kapyla[2] == pytest.approx(16_786, abs=1)
