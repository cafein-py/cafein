"""TransportNetwork built from the Helsinki GTFS feed shared with r5py."""

import numpy as np
import pytest

from cafein import TransportNetwork

UNREACHABLE = np.uint32(0xFFFFFFFF)


def test_network_statistics(network):
    assert network.stop_count == 8305
    assert network.pattern_count == 1395
    assert network.trip_count == 195_351


def test_routes_the_earliest_direct_k_train(network):
    # Korso -> Käpylä at 08:30 on 2022-02-22 (r5py's canonical departure):
    # the earliest direct ride leaves 08:36:00 and arrives 08:58:00 on trip
    # 3001K_20220222_S1_2_0831, verified independently from the GTFS tables.
    journeys = network.route_between_stops(
        "4810551", "1250551", "2022-02-22", "08:30:00"
    )
    assert journeys

    direct = journeys[0]
    assert direct["rides"] == 1
    assert direct["arrival"] == 8 * 3600 + 58 * 60

    access, transit, egress = direct["legs"]
    assert access["type"] == "access"
    assert transit["type"] == "transit"
    assert transit["trip_id"] == "3001K_20220222_S1_2_0831"
    assert transit["route_short_name"] == "K"
    assert transit["board_stop"] == "4810551"
    assert transit["alight_stop"] == "1250551"
    assert transit["departure"] == 8 * 3600 + 36 * 60
    assert egress["type"] == "egress"

    # Journeys form a Pareto set: more rides only when strictly earlier.
    for earlier, later in zip(journeys, journeys[1:]):
        assert later["rides"] > earlier["rides"]
        assert later["arrival"] < earlier["arrival"]


def test_transit_legs_carry_distance_and_provenance(network):
    # The K train's Korso -> Käpylä distance is 16.786 km straight from
    # the feed's shape_dist_traveled values (recorded in kilometers).
    journeys = network.route_between_stops(
        "4810551", "1250551", "2022-02-22", "08:30:00"
    )
    access, transit, egress = journeys[0]["legs"]
    assert transit["distance"] == pytest.approx(16_786, abs=1)
    assert transit["distance_provenance"] == "shape_dist"
    assert access["distance"] is None
    assert egress["distance"] is None
    assert network.distance_provenance_counts == {"shape_dist": 195_351}


def test_transit_legs_carry_wkb_geometry(network):
    # The K train's leg geometry is the HSL shape sliced between Korso
    # and Käpylä: a dense LineString whose projected length agrees with
    # the shape_dist distance and whose ends sit on the stops.
    import geopandas as gpd
    import shapely

    journeys = network.route_between_stops(
        "4810551", "1250551", "2022-02-22", "08:30:00"
    )
    access, transit, egress = journeys[0]["legs"]
    line = shapely.from_wkb(transit["geometry"])
    assert line.geom_type == "LineString"
    assert shapely.get_num_coordinates(line) > 10
    series = gpd.GeoSeries([line], crs="EPSG:4326")
    length = float(series.to_crs(series.estimate_utm_crs()).length.iloc[0])
    assert length == pytest.approx(transit["distance"], rel=0.01)
    coordinates = {stop: (lon, lat) for stop, lat, lon in network.stops}
    assert line.coords[0] == pytest.approx(coordinates["4810551"], abs=1e-3)
    assert line.coords[-1] == pytest.approx(coordinates["1250551"], abs=1e-3)
    # Walk legs carry no geometry yet.
    assert access["geometry"] is None
    assert egress["geometry"] is None


def test_crow_fly_legs_draw_the_stop_chain(tmp_path):
    import shapely

    feed = build_synthetic_gtfs(tmp_path / "synthetic_gtfs.zip")
    with pytest.warns(UserWarning):
        network = TransportNetwork.from_gtfs([str(feed)])
    journeys = network.route_between_stops("S1", "S2", "2022-02-22", "07:30:00")
    transit = journeys[0]["legs"][1]
    line = shapely.from_wkb(transit["geometry"])
    # No shape in the feed: the geometry is the straight stop chain.
    assert list(line.coords) == [(24.0, 60.0), (24.01, 60.01)]


def test_geometry_output_is_controllable(tmp_path):
    feed = build_synthetic_gtfs(tmp_path / "synthetic_gtfs.zip")
    with pytest.warns(UserWarning):
        network = TransportNetwork.from_gtfs([str(feed)])
    journeys = network.route_between_stops(
        "S1", "S2", "2022-02-22", "07:30:00", geometries=False
    )
    assert journeys[0]["legs"][1]["geometry"] is None
    # Building without leg geometries keeps distances but no polylines.
    with pytest.warns(UserWarning):
        lean = TransportNetwork.from_gtfs([str(feed)], leg_geometries=False)
    transit = lean.route_between_stops("S1", "S2", "2022-02-22", "07:30:00")[0]["legs"][
        1
    ]
    assert transit["geometry"] is None
    assert transit["distance"] is not None


def test_set_leg_geometries_validates_its_payload(tmp_path):
    feed = build_synthetic_gtfs(tmp_path / "synthetic_gtfs.zip")
    with pytest.warns(UserWarning):
        network = TransportNetwork.from_gtfs([str(feed)], trip_distances=False)
    with pytest.raises(ValueError, match="malformed"):
        network.set_leg_geometries(
            [([24.0], [60.0], [0.0])],
            [("T_OK", 0, [0.0, 100.0])],
        )
    with pytest.raises(ValueError, match="no positions"):
        network.set_leg_geometries(
            [([24.0, 24.01], [60.0, 60.01], [0.0, 100.0])],
            [],
        )


def test_a_departure_window_profiles_the_k_trains(network):
    # Korso -> Käpylä over 08:30-09:00: the window holds two direct K
    # trains (08:36->08:58 and 08:56->09:18, straight from the GTFS
    # tables); each journey's departure is the latest feasible leave time,
    # and the window's final second waits for the 09:16 train.
    journeys = network.route_between_stops(
        "4810551", "1250551", "2022-02-22", "08:30:00", window=1800
    )
    direct = [(j["departure"], j["arrival"]) for j in journeys if j["rides"] == 1]
    assert direct == [
        (8 * 3600 + 36 * 60, 8 * 3600 + 58 * 60),
        (8 * 3600 + 56 * 60, 9 * 3600 + 18 * 60),
        (9 * 3600 - 1, 9 * 3600 + 38 * 60),
    ]


def test_travel_times_from_a_stop(network):
    times = network.travel_times_from_stop("4810551", "2022-02-22", "08:30:00")
    assert times["4810551"] == 0
    # The K train reaches Käpylä at 08:58 (see the routing test above).
    assert times["1250551"] == 28 * 60
    # The sample feed carries only Helsinki-bound trips on this corridor,
    # and without footpaths the rail platforms connect to nothing else:
    # the origin plus the twelve downstream stops, verified independently
    # with a time-respecting search over the raw GTFS tables.
    assert len(times) == 13
    # Outside the feed window only the origin itself is reachable.
    assert network.travel_times_from_stop("4810551", "2022-06-01", "08:30:00") == {
        "4810551": 0
    }


def test_travel_times_use_installed_transfers(network, network_with_footpaths):
    # Kamppi's street stop is only reachable from the metro platform over
    # a footpath (see the journey test below): 08:31 M2, alight 08:39,
    # 20 s walk.
    base = network.travel_times_from_stop("1100602", "2022-02-22", "08:30:00")
    assert "1040280" not in base
    walked = network_with_footpaths.travel_times_from_stop(
        "1100602", "2022-02-22", "08:30:00"
    )
    assert walked["1040280"] == 9 * 60 + 20
    # With footpaths the metro feeds the whole central network.
    assert len(walked) > 1_000


def test_travel_time_matrix_matches_single_origin_runs(network):
    origins = ["4810551", "1250551"]
    matrix = network.travel_time_matrix(origins, "2022-02-22", "08:30:00")
    assert matrix.shape == (2, network.stop_count)
    assert matrix.dtype == np.uint32
    stop_order = [stop for stop, _, _ in network.stops]
    for row, origin in enumerate(origins):
        times = network.travel_times_from_stop(origin, "2022-02-22", "08:30:00")
        reachable = {
            stop_order[column]: int(matrix[row, column])
            for column in np.nonzero(matrix[row] != UNREACHABLE)[0]
        }
        assert reachable == times


def test_travel_time_matrix_is_deterministic(network):
    origins = [stop for stop, _, _ in network.stops[:64]]
    first = network.travel_time_matrix(origins, "2022-02-22", "08:30:00")
    second = network.travel_time_matrix(origins, "2022-02-22", "08:30:00")
    assert np.array_equal(first, second)


def test_tbtr_router_matches_raptor(network, network_with_footpaths):
    stops = [stop for stop, _, _ in network.stops][:120]
    raptor = network.travel_time_matrix(stops, "2022-02-22", "08:30:00")
    tbtr = network.travel_time_matrix(stops, "2022-02-22", "08:30:00", router="tbtr")
    assert np.array_equal(raptor, tbtr)
    # With footpaths installed, the walks relax at query time; the
    # matrices must still agree cell for cell.
    raptor = network_with_footpaths.travel_time_matrix(stops, "2022-02-22", "08:30:00")
    tbtr = network_with_footpaths.travel_time_matrix(
        stops, "2022-02-22", "08:30:00", router="tbtr"
    )
    assert np.array_equal(raptor, tbtr)


def test_router_option_is_validated(network):
    with pytest.raises(ValueError, match="router must be"):
        network.travel_time_matrix(
            ["4810551"], "2022-02-22", "08:30:00", router="fastest"
        )
    with pytest.raises(ValueError, match="single-departure"):
        network.travel_time_matrix(
            ["4810551"], "2022-02-22", "08:30:00", window=600, router="tbtr"
        )


def test_travel_time_matrix_rejects_unknown_stops(network):
    with pytest.raises(KeyError, match="no-such-stop"):
        network.travel_time_matrix(
            ["4810551", "no-such-stop"], "2022-02-22", "08:30:00"
        )


def test_travel_time_matrix_accepts_no_origins(network):
    matrix = network.travel_time_matrix([], "2022-02-22", "08:30:00")
    assert matrix.shape == (0, network.stop_count)


def test_no_service_on_a_date_outside_the_feed_window(network):
    journeys = network.route_between_stops(
        "4810551", "1250551", "2022-06-01", "08:30:00"
    )
    assert journeys == []


def test_unknown_stop_raises_a_key_error(network):
    with pytest.raises(KeyError, match="no-such-stop"):
        network.route_between_stops("no-such-stop", "1250551", "2022-02-22", "08:30:00")


def test_invalid_date_and_time_raise_value_errors(network):
    with pytest.raises(ValueError, match="invalid date"):
        network.route_between_stops("4810551", "1250551", "22.2.2022", "08:30:00")
    with pytest.raises(ValueError, match="invalid time"):
        network.route_between_stops("4810551", "1250551", "2022-02-22", "8.30")
    with pytest.raises(ValueError, match="invalid time"):
        network.route_between_stops("4810551", "1250551", "2022-02-22", "1300000:00:00")


def test_merged_feeds_require_qualified_stop_ids(helsinki_gtfs):
    merged = TransportNetwork.from_gtfs([str(helsinki_gtfs), str(helsinki_gtfs)])
    assert merged.stop_count == 2 * 8305
    with pytest.raises(KeyError, match="several feeds"):
        merged.route_between_stops("4810551", "1250551", "2022-02-22", "08:30:00")
    for feed in (0, 1):
        journeys = merged.route_between_stops(
            f"{feed}:4810551", f"{feed}:1250551", "2022-02-22", "08:30:00"
        )
        assert journeys[0]["arrival"] == 8 * 3600 + 58 * 60
        transit = journeys[0]["legs"][1]
        assert transit["board_stop"] == f"{feed}:4810551"
        assert transit["alight_stop"] == f"{feed}:1250551"
        assert transit["trip_id"] == f"{feed}:3001K_20220222_S1_2_0831"


def test_an_osm_extract_installs_transfers(network_with_footpaths):
    assert network_with_footpaths.transfer_count > 1_000_000


def test_street_stops_are_reachable_only_over_footpaths(
    network, network_with_footpaths
):
    # Kalasatama westbound metro platform to a Kamppi street stop: only
    # metro trips serve the platform, so without footpaths the street stop
    # is unreachable.
    assert (
        network.route_between_stops("1100602", "1040280", "2022-02-22", "08:30:00")
        == []
    )
    # With footpaths: the 08:31 M2 reaches the Kamppi platform at 08:39
    # (times straight from the GTFS tables) and the street stop is a
    # 20-second walk away.
    journeys = network_with_footpaths.route_between_stops(
        "1100602", "1040280", "2022-02-22", "08:30:00"
    )
    first = journeys[0]
    assert first["rides"] == 1
    assert first["arrival"] == 8 * 3600 + 39 * 60 + 20

    access, transit, transfer, egress = first["legs"]
    assert access["type"] == "access"
    assert transit["trip_id"] == "31M2_20220222_Ti_2_0817"
    assert transit["route_short_name"] == "M2"
    assert transit["board_stop"] == "1100602"
    assert transit["departure"] == 8 * 3600 + 31 * 60
    assert transit["alight_stop"] == "1040602"
    assert transit["arrival"] == 8 * 3600 + 39 * 60
    assert transfer["type"] == "transfer"
    assert transfer["from_stop"] == "1040602"
    assert transfer["to_stop"] == "1040280"
    assert transfer["arrival"] - transfer["departure"] == 20
    # Transfer legs carry the footpath's exact meters (1 m/s walking).
    assert 19 <= transfer["distance"] <= 20
    assert egress["type"] == "egress"


def stop_coordinates(network, stop_id):
    """A stop's (lat, lon) from the network's stop list."""
    for stop, lat, lon in network.stops:
        if stop == stop_id:
            return lat, lon
    raise KeyError(stop_id)


def test_access_stops_match_the_footpath_pins(network_with_footpaths):
    # Queried from a stop's own coordinates, the street search enters the
    # graph through the same snap link as the footpath precompute, so the
    # pinned Kamppi walking times reappear (the doubled connector and the
    # snapping metric can shift a value across the one-second rounding).
    lat, lon = stop_coordinates(network_with_footpaths, "1040602")
    reached = network_with_footpaths.access_stops(lat, lon)
    assert reached["1040602"] <= 1
    assert 4 <= reached["1040601"] <= 5
    assert 20 <= reached["1040280"] <= 21
    lat, lon = stop_coordinates(network_with_footpaths, "1000102")
    assert 22 <= network_with_footpaths.access_stops(lat, lon)["1040280"] <= 23


def test_access_stops_respect_the_walking_cutoff(network_with_footpaths):
    lat, lon = stop_coordinates(network_with_footpaths, "1040602")
    walkable = network_with_footpaths.access_stops(lat, lon, max_walking_time=600.0)
    assert 30 <= len(walkable) <= 90
    assert all(0 <= seconds <= 600 for seconds in walkable.values())
    # A tighter cutoff filters the same walking times, never changes them.
    nearby = network_with_footpaths.access_stops(lat, lon, max_walking_time=120.0)
    assert set(nearby) < set(walkable)
    assert all(walkable[stop] == seconds for stop, seconds in nearby.items())


def test_faster_walking_reaches_more_stops(network_with_footpaths):
    lat, lon = stop_coordinates(network_with_footpaths, "1040602")
    walked = network_with_footpaths.access_stops(lat, lon)
    ran = network_with_footpaths.access_stops(lat, lon, walking_speed_kmph=7.2)
    assert set(ran) > set(walked)


def test_access_stops_reject_out_of_range_parameters(network_with_footpaths):
    lat, lon = stop_coordinates(network_with_footpaths, "1040602")
    with pytest.raises(ValueError, match="lat and lon"):
        network_with_footpaths.access_stops(float("nan"), lon)
    with pytest.raises(ValueError, match="walking_speed_kmph"):
        network_with_footpaths.access_stops(lat, lon, walking_speed_kmph=float("inf"))
    with pytest.raises(ValueError, match="walking_speed_kmph"):
        network_with_footpaths.access_stops(lat, lon, walking_speed_kmph=0.0)
    with pytest.raises(ValueError, match="max_walking_time"):
        network_with_footpaths.access_stops(lat, lon, max_walking_time=float("nan"))
    with pytest.raises(ValueError, match="max_snap_distance"):
        network_with_footpaths.access_stops(lat, lon, max_snap_distance=-1.0)


def test_access_stops_need_a_street_nearby(network_with_footpaths):
    # Open water south of the extract, far from every walkable way.
    with pytest.raises(ValueError, match="farther than"):
        network_with_footpaths.access_stops(60.10, 24.90)


def test_access_stops_need_a_street_network(network):
    with pytest.raises(ValueError, match="no street network"):
        network.access_stops(60.168, 24.931)


def test_routes_door_to_door_between_coordinates(network_with_footpaths):
    # Kalasatama westbound platform to the Kamppi street stop, queried by
    # their own coordinates: the same 08:31 M2 ride the stop-to-stop
    # oracle pins, but the egress walks straight from the Kamppi metro
    # platform to the destination coordinate instead of transferring
    # first, so the arrivals agree.
    origin = stop_coordinates(network_with_footpaths, "1100602")
    destination = stop_coordinates(network_with_footpaths, "1040280")
    journeys = network_with_footpaths.route_between_coordinates(
        origin, destination, "2022-02-22", "08:30:00"
    )
    stop_journeys = network_with_footpaths.route_between_stops(
        "1100602", "1040280", "2022-02-22", "08:30:00"
    )
    # Walking all the way leads the frontier (zero rides) but is far
    # slower than the ride; the M2 journey follows.
    walk_only, first = journeys[0], journeys[1]
    assert walk_only["rides"] == 0
    (walk_leg,) = walk_only["legs"]
    assert walk_leg["type"] == "walk"
    assert 3_000 <= walk_leg["distance"] <= 6_000
    assert walk_only["arrival"] - walk_only["departure"] >= walk_leg["distance"]
    assert first["rides"] == 1
    assert abs(first["arrival"] - stop_journeys[0]["arrival"]) <= 2

    access, transit, egress = first["legs"]
    assert access["type"] == "access"
    assert access["to_stop"] == "1100602"
    # Walk distances are the exact street-path meters; the leg duration
    # is the same walk rounded up to whole seconds at 1 m/s (3.6 km/h).
    assert 0 <= access["distance"] <= 15
    assert access["distance"] <= access["arrival"] - access["departure"]
    assert transit["trip_id"] == "31M2_20220222_Ti_2_0817"
    assert transit["departure"] == 8 * 3600 + 31 * 60
    assert egress["type"] == "egress"
    assert egress["from_stop"] == "1040602"
    assert 19 <= egress["distance"] <= 23


def test_door_to_door_window_profiles_departures(network_with_footpaths):
    origin = stop_coordinates(network_with_footpaths, "1100602")
    destination = stop_coordinates(network_with_footpaths, "1040280")
    profile = network_with_footpaths.route_between_coordinates(
        origin, destination, "2022-02-22", "08:30:00", window=600
    )
    # The departure-independent walk leads the profile; every ride
    # follows in departure order.
    walk_only, rides = profile[0], profile[1:]
    assert walk_only["rides"] == 0
    assert walk_only["legs"][0]["type"] == "walk"
    assert len(rides) >= 3
    departures = [journey["departure"] for journey in profile]
    assert departures == sorted(departures)
    for journey in rides:
        assert journey["departure"] >= 8 * 3600 + 30 * 60
        assert journey["arrival"] > journey["departure"]
        assert journey["rides"] >= 1


def test_travel_times_from_coordinate_seed_walkable_stops(network_with_footpaths):
    origin = stop_coordinates(network_with_footpaths, "1040602")
    reached = network_with_footpaths.travel_times_from_coordinate(
        origin, "2022-02-22", "08:30:00", max_walking_time=600.0
    )
    walkable = network_with_footpaths.access_stops(*origin, max_walking_time=600.0)
    # Every stop within walking distance appears, at most as far away as
    # the pure walk; transit extends the reach far beyond it.
    for stop, seconds in walkable.items():
        assert reached[stop] <= seconds
    assert reached["1040602"] <= 1
    assert 19 <= reached["1040280"] <= 21
    assert len(reached) > 10 * len(walkable)


def test_door_to_door_needs_streets_and_valid_coordinates(
    network, network_with_footpaths
):
    origin = stop_coordinates(network_with_footpaths, "1100602")
    destination = stop_coordinates(network_with_footpaths, "1040280")
    with pytest.raises(ValueError, match="no street network"):
        network.route_between_coordinates(origin, destination, "2022-02-22", "08:30:00")
    with pytest.raises(ValueError, match="no street network"):
        network.travel_times_from_coordinate(origin, "2022-02-22", "08:30:00")
    with pytest.raises(ValueError, match="origin .* is farther"):
        network_with_footpaths.route_between_coordinates(
            (60.10, 24.90), destination, "2022-02-22", "08:30:00"
        )
    with pytest.raises(ValueError, match="destination .* is farther"):
        network_with_footpaths.route_between_coordinates(
            origin, (60.10, 24.90), "2022-02-22", "08:30:00"
        )
    with pytest.raises(ValueError, match="walking_speed_kmph"):
        network_with_footpaths.route_between_coordinates(
            origin,
            destination,
            "2022-02-22",
            "08:30:00",
            walking_speed_kmph=float("nan"),
        )
    with pytest.raises(ValueError, match="origin lat and lon"):
        network_with_footpaths.travel_times_from_coordinate(
            (float("nan"), 24.9), "2022-02-22", "08:30:00"
        )


def test_a_synthetic_network_routes_door_to_door(tmp_path):
    feed = build_synthetic_gtfs(tmp_path / "synthetic_gtfs.zip")
    with pytest.warns(UserWarning):
        network = TransportNetwork.from_gtfs([str(feed)])
    # One 2 km street edge; S1 sits on its start, S2 at 90 % of its cost
    # length, so the walking cutoff (600 s at 1 m/s) keeps each endpoint
    # within reach of exactly one stop and riding is the only way across.
    network.set_street_network(
        2,
        [(0, 1, 2000.0)],
        [0, 2],
        [24.0, 24.035842],
        [60.0, 60.0],
        [("S1", 0, 0.0, 0.0), ("S2", 0, 0.9, 0.0)],
    )
    # From the edge start (S1's snap point) to the edge end: ride T_OK
    # 08:00 -> 08:10, then walk the outer tenth of the edge's cost length
    # (200 m at 1 m/s) — designed values, exact.
    journeys = network.route_between_coordinates(
        (60.0, 24.0),
        (60.0, 24.035842),
        "2022-02-22",
        "07:30:00",
        max_walking_time=600.0,
    )
    first = journeys[0]
    assert first["arrival"] == 8 * 3600 + 10 * 60 + 200
    access, transit, egress = first["legs"]
    assert access["to_stop"] == "S1"
    assert access["distance"] == 0.0
    assert transit["trip_id"] == "T_OK"
    assert egress["from_stop"] == "S2"
    assert egress["distance"] == pytest.approx(200.0)

    # A destination best reached on foot yields a walking-only journey:
    # ~50 m along the edge (0.0009° of its 0.035842° cost length).
    (walk_only,) = network.route_between_coordinates(
        (60.0, 24.0), (60.0, 24.0009), "2022-02-22", "07:30:00"
    )
    assert walk_only["rides"] == 0
    (walk_leg,) = walk_only["legs"]
    assert walk_leg["type"] == "walk"
    assert walk_leg["distance"] == pytest.approx(50.2, abs=1.0)
    assert walk_only["arrival"] - walk_only["departure"] in (50, 51)

    # A destination at the origin's own coordinate is a zero walk.
    (still,) = network.route_between_coordinates(
        (60.0, 24.0), (60.0, 24.0), "2022-02-22", "07:30:00"
    )
    assert still["rides"] == 0
    assert still["arrival"] == still["departure"]
    assert still["legs"][0]["distance"] == 0.0

    # Beyond the walking cutoff there is neither a ride nor a walk.
    assert (
        network.route_between_coordinates(
            (60.0, 24.0),
            (60.0, 24.0009),
            "2022-02-22",
            "07:30:00",
            max_walking_time=20.0,
        )
        == []
    )


def test_a_synthetic_street_network_answers_access_queries(tmp_path):
    feed = build_synthetic_gtfs(tmp_path / "synthetic_gtfs.zip")
    with pytest.warns(UserWarning):
        network = TransportNetwork.from_gtfs([str(feed)])
    # One 400 m edge running east from S1's coordinates; S1 sits on its
    # start, S2 halfway along. Walking at the default 3.6 km/h (1 m/s),
    # the cost length — not the geometry — sets the times.
    network.set_street_network(
        2,
        [(0, 1, 400.0)],
        [0, 2],
        [24.0, 24.0072],
        [60.0, 60.0],
        [("S1", 0, 0.0, 0.0), ("S2", 0, 0.5, 0.0)],
    )
    reached = network.access_stops(60.0, 24.0)
    assert reached["S1"] == 0
    assert reached["S2"] == 200


def test_set_street_network_validates_its_payload(tmp_path):
    feed = build_synthetic_gtfs(tmp_path / "synthetic_gtfs.zip")
    with pytest.warns(UserWarning):
        network = TransportNetwork.from_gtfs([str(feed)])
    with pytest.raises(ValueError, match="offsets"):
        network.set_street_network(2, [(0, 1, 400.0)], [0], [], [], [])
    with pytest.raises(KeyError, match="unknown stop_id"):
        network.set_street_network(
            2,
            [(0, 1, 400.0)],
            [0, 2],
            [24.0, 24.0072],
            [60.0, 60.0],
            [("missing", 0, 0.5, 0.0)],
        )


def build_synthetic_gtfs(path):
    """A two-stop feed with one good trip, one backwards trip, and a stop
    whose raw id looks like a feed-qualified id."""
    import io
    import zipfile

    tables = {
        "agency.txt": [
            "agency_id,agency_name,agency_url,agency_timezone",
            "A,Test Agency,http://example.com,Europe/Helsinki",
        ],
        "stops.txt": [
            "stop_id,stop_name,stop_lat,stop_lon",
            "S1,First,60.0,24.0",
            "S2,Second,60.01,24.01",
            "0:S1,Colon,60.02,24.02",
        ],
        "routes.txt": [
            "route_id,route_short_name,route_type",
            "R1,1,3",
        ],
        "trips.txt": [
            "route_id,service_id,trip_id",
            "R1,SV,T_OK",
            "R1,SV,T_BACKWARDS",
        ],
        "stop_times.txt": [
            "trip_id,arrival_time,departure_time,stop_id,stop_sequence",
            "T_OK,08:00:00,08:00:00,S1,1",
            "T_OK,08:10:00,08:10:00,S2,2",
            "T_BACKWARDS,09:00:00,09:00:00,S1,1",
            "T_BACKWARDS,08:30:00,08:30:00,S2,2",
        ],
        "calendar.txt": [
            "service_id,monday,tuesday,wednesday,thursday,friday,saturday,"
            "sunday,start_date,end_date",
            "SV,1,1,1,1,1,1,1,20220101,20221231",
        ],
    }
    with zipfile.ZipFile(path, "w") as archive:
        for name, lines in tables.items():
            archive.writestr(name, "\n".join(lines) + "\n")
    return path


def test_quarantined_trips_raise_a_warning(tmp_path):
    feed = build_synthetic_gtfs(tmp_path / "synthetic_gtfs.zip")
    with pytest.warns(UserWarning, match="quarantined 1 trip"):
        network = TransportNetwork.from_gtfs([str(feed)])
    journeys = network.route_between_stops("S1", "S2", "2022-02-22", "07:30:00")
    assert journeys[0]["arrival"] == 8 * 3600 + 10 * 60


def build_timepoint_gtfs(path):
    """A three-stop feed in the shape issue #45 describes: stop times
    blank at the non-timepoint middle stop, and an invalid (non-hex)
    ``route_text_color`` that a strict parser rejects."""
    import zipfile

    tables = {
        "agency.txt": [
            "agency_id,agency_name,agency_url,agency_timezone",
            "A,Test Agency,http://example.com,Europe/Helsinki",
        ],
        "stops.txt": [
            "stop_id,stop_name,stop_lat,stop_lon",
            "S1,First,60.0,24.0",
            "S2,Middle,60.01,24.01",
            "S3,Last,60.02,24.02",
        ],
        "routes.txt": [
            "route_id,route_short_name,route_type,route_text_color",
            "R1,1,3,0",
        ],
        "trips.txt": [
            "route_id,service_id,trip_id",
            "R1,SV,T1",
        ],
        "stop_times.txt": [
            "trip_id,arrival_time,departure_time,stop_id,stop_sequence",
            "T1,08:00:00,08:00:00,S1,1",
            "T1,,,S2,2",
            "T1,08:10:00,08:10:00,S3,3",
        ],
        "calendar.txt": [
            "service_id,monday,tuesday,wednesday,thursday,friday,saturday,"
            "sunday,start_date,end_date",
            "SV,1,1,1,1,1,1,1,20220101,20221231",
        ],
    }
    with zipfile.ZipFile(path, "w") as archive:
        for name, lines in tables.items():
            archive.writestr(name, "\n".join(lines) + "\n")
    return path


def test_timepoint_feeds_are_repaired_at_ingest(tmp_path):
    # Issue #45: blank interior stop times interpolate (with a warning)
    # and an invalid cosmetic route colour no longer rejects the feed.
    feed = build_timepoint_gtfs(tmp_path / "timepoint_gtfs.zip")
    with pytest.warns(UserWarning, match="interpolated blank stop times on 1 trip"):
        network = TransportNetwork.from_gtfs([str(feed)])
    assert [route_id for route_id, _, _ in network.routes] == ["R1"]
    # Boarding at the interpolated middle stop works, halfway between
    # the timepoints.
    journeys = network.route_between_stops("S2", "S3", "2022-02-22", "08:00:00")
    transit = [leg for leg in journeys[0]["legs"] if leg["type"] == "transit"]
    assert transit[0]["departure"] == 8 * 3600 + 5 * 60
    assert journeys[0]["arrival"] == 8 * 3600 + 10 * 60


def test_qualified_ids_take_precedence_over_colon_raw_ids(tmp_path):
    feed = build_synthetic_gtfs(tmp_path / "synthetic_gtfs.zip")
    with pytest.warns(UserWarning):
        merged = TransportNetwork.from_gtfs([str(feed), str(feed)])
    # "0:S1" resolves to feed 0's stop S1, not the raw stop named "0:S1".
    journeys = merged.route_between_stops("0:S1", "0:S2", "2022-02-22", "07:30:00")
    assert journeys[0]["arrival"] == 8 * 3600 + 10 * 60
    assert journeys[0]["legs"][1]["board_stop"] == "0:S1"
    # The colon-named stop stays addressable through full qualification.
    assert merged.route_between_stops("0:0:S1", "0:S2", "2022-02-22", "07:30:00") == []


def test_from_gtfs_accepts_a_single_bare_path(tmp_path):
    feed = build_synthetic_gtfs(tmp_path / "synthetic_gtfs.zip")
    with pytest.warns(UserWarning):
        network = TransportNetwork.from_gtfs(feed)
    assert network.stop_count == 3


def test_routes_carry_the_single_agency_when_implicit(tmp_path):
    # The synthetic feed's routes.txt has no agency_id column; its one
    # agency still resolves for the agency-level emission-factor tier.
    feed = build_synthetic_gtfs(tmp_path / "synthetic_gtfs.zip")
    with pytest.warns(UserWarning):
        network = TransportNetwork.from_gtfs([str(feed)])
    assert network.routes == [("R1", "A", 3)]


def test_stops_are_exposed_with_coordinates(tmp_path):
    feed = build_synthetic_gtfs(tmp_path / "synthetic_gtfs.zip")
    with pytest.warns(UserWarning):
        network = TransportNetwork.from_gtfs([str(feed)])
    # Stops come back in the reader's order: sorted by id within a feed.
    assert network.stops == [
        ("0:S1", 60.02, 24.02),
        ("S1", 60.0, 24.0),
        ("S2", 60.01, 24.01),
    ]


def test_set_transfers_routes_over_footpaths(tmp_path):
    feed = build_synthetic_gtfs(tmp_path / "synthetic_gtfs.zip")
    with pytest.warns(UserWarning):
        network = TransportNetwork.from_gtfs([str(feed)])
    assert network.transfer_count == 0
    network.set_transfers([("S2", "0:S1", 120, 118.5), ("0:S1", "S2", 120, 118.5)])
    assert network.transfer_count == 2

    # Ride to S2 (arrives 08:10), walk the 120-second footpath.
    journeys = network.route_between_stops("S1", "0:S1", "2022-02-22", "07:30:00")
    first = journeys[0]
    assert first["arrival"] == 8 * 3600 + 10 * 60 + 120
    types = [leg["type"] for leg in first["legs"]]
    assert types == ["access", "transit", "transfer", "egress"]
    transfer = first["legs"][2]
    assert transfer["from_stop"] == "S2"
    assert transfer["to_stop"] == "0:S1"
    assert transfer["distance"] == 118.5


def test_set_transfers_rejects_unknown_stops(tmp_path):
    feed = build_synthetic_gtfs(tmp_path / "synthetic_gtfs.zip")
    with pytest.warns(UserWarning):
        network = TransportNetwork.from_gtfs([str(feed)])
    with pytest.raises(KeyError, match="no-such-stop"):
        network.set_transfers([("no-such-stop", "S2", 60, 60.0)])
    with pytest.raises(ValueError, match="non-finite"):
        network.set_transfers([("S1", "S2", 60, float("nan"))])


def test_transfer_arrays_match_the_tuple_form(tmp_path):
    from cafein.streets import Footpaths

    feed = build_synthetic_gtfs(tmp_path / "synthetic_gtfs.zip")
    edges = [("S2", "0:S1", 120, 118.5), ("0:S1", "S2", 120, 118.5)]
    networks = []
    for footpaths in (
        edges,
        Footpaths(["S2", "0:S1"], [0, 1], [1, 0], [120, 120], [118.5, 118.5]),
    ):
        with pytest.warns(UserWarning):
            network = TransportNetwork.from_gtfs([str(feed)])
        network.set_transfers(footpaths)
        networks.append(network)
    tuples, arrays = networks
    assert tuples.transfer_count == arrays.transfer_count == 2
    # Both installs route identically, footpath legs included.
    key = ("S1", "0:S1", "2022-02-22", "07:30:00")
    assert tuples.route_between_stops(*key) == arrays.route_between_stops(*key)


def test_transfer_arrays_are_validated(tmp_path):
    from cafein.streets import Footpaths

    feed = build_synthetic_gtfs(tmp_path / "synthetic_gtfs.zip")
    with pytest.warns(UserWarning):
        network = TransportNetwork.from_gtfs([str(feed)])
    with pytest.raises(KeyError, match="no-such-stop"):
        network.set_transfers(Footpaths(["no-such-stop", "S2"], [0], [1], [60], [60.0]))
    with pytest.raises(ValueError, match="non-finite"):
        network.set_transfers(Footpaths(["S1", "S2"], [0], [1], [60], [float("nan")]))
    with pytest.raises(ValueError, match="outside stop_ids"):
        network._core.set_transfer_arrays(
            ["S1", "S2"],
            np.array([5], dtype=np.uint32),
            np.array([1], dtype=np.uint32),
            np.array([60], dtype=np.uint32),
            np.array([60.0]),
        )
    with pytest.raises(ValueError, match="same length"):
        network._core.set_transfer_arrays(
            ["S1", "S2"],
            np.array([0], dtype=np.uint32),
            np.array([1, 0], dtype=np.uint32),
            np.array([60], dtype=np.uint32),
            np.array([60.0]),
        )
    # Values that would silently wrap into valid uint32 edges are
    # rejected before any narrowing cast.
    with pytest.raises(ValueError, match="unsigned 32-bit"):
        Footpaths(["S1", "S2"], np.array([-1]), [1], [60], [60.0])
    with pytest.raises(ValueError, match="unsigned 32-bit"):
        Footpaths(["S1", "S2"], [0], [1], np.array([2**32], dtype=np.uint64), [60.0])
    with pytest.raises(ValueError, match="integer"):
        Footpaths(["S1", "S2"], np.array([0.5]), [1], [60], [60.0])
    with pytest.raises(ValueError, match="one-dimensional"):
        Footpaths(["S1", "S2"], np.array([[0]]), [1], [60], [60.0])


def test_trip_distances_default_to_the_ladder(tmp_path):
    # The synthetic feed has no shapes: distances fall to crow-fly with
    # the bus detour coefficient (S1->S2 is ~1243 m crow-fly).
    feed = build_synthetic_gtfs(tmp_path / "synthetic_gtfs.zip")
    with pytest.warns(UserWarning):
        network = TransportNetwork.from_gtfs([str(feed)])
    journeys = network.route_between_stops("S1", "S2", "2022-02-22", "07:30:00")
    transit = journeys[0]["legs"][1]
    assert transit["distance"] == pytest.approx(1243 * 1.4, rel=0.01)
    assert transit["distance_provenance"] == "crow_fly"


def test_trip_distances_can_be_disabled(tmp_path):
    feed = build_synthetic_gtfs(tmp_path / "synthetic_gtfs.zip")
    with pytest.warns(UserWarning):
        network = TransportNetwork.from_gtfs([str(feed)], trip_distances=False)
    assert network.distance_provenance_counts == {}
    journeys = network.route_between_stops("S1", "S2", "2022-02-22", "07:30:00")
    transit = journeys[0]["legs"][1]
    assert transit["distance"] is None
    assert transit["distance_provenance"] is None


def test_quarantined_trips_cannot_abort_distance_preprocessing(tmp_path):
    # A quarantined trip visiting a coordinate-less stop must not abort
    # the default build: the ladder only runs over routable trips.
    import zipfile

    tables = {
        "agency.txt": [
            "agency_id,agency_name,agency_url,agency_timezone",
            "A,Test Agency,http://example.com,Europe/Helsinki",
        ],
        "stops.txt": [
            "stop_id,stop_name,stop_lat,stop_lon",
            "S1,First,60.0,24.0",
            "S2,Second,60.01,24.01",
            "X,No coordinates,,",
        ],
        "routes.txt": ["route_id,route_short_name,route_type", "R1,1,3"],
        "trips.txt": [
            "route_id,service_id,trip_id",
            "R1,SV,T_OK",
            "R1,SV,T_BAD",
        ],
        "stop_times.txt": [
            "trip_id,arrival_time,departure_time,stop_id,stop_sequence",
            "T_OK,08:00:00,08:00:00,S1,1",
            "T_OK,08:10:00,08:10:00,S2,2",
            "T_BAD,09:00:00,09:00:00,S1,1",
            "T_BAD,08:30:00,08:30:00,X,2",
        ],
        "calendar.txt": [
            "service_id,monday,tuesday,wednesday,thursday,friday,saturday,"
            "sunday,start_date,end_date",
            "SV,1,1,1,1,1,1,1,20220101,20221231",
        ],
    }
    feed = tmp_path / "quarantine_gtfs.zip"
    with zipfile.ZipFile(feed, "w") as archive:
        for name, lines in tables.items():
            archive.writestr(name, "\n".join(lines) + "\n")
    with pytest.warns(UserWarning, match="quarantined 1 trip"):
        network = TransportNetwork.from_gtfs([str(feed)])
    assert network.distance_provenance_counts == {"crow_fly": 1}


def test_set_trip_distances_validates_input(tmp_path):
    feed = build_synthetic_gtfs(tmp_path / "synthetic_gtfs.zip")
    with pytest.warns(UserWarning):
        network = TransportNetwork.from_gtfs([str(feed)], trip_distances=False)
    # Unknown trips (e.g. quarantined ones) are ignored, but every
    # timetable trip must end up covered.
    with pytest.raises(ValueError, match="no distances"):
        network.set_trip_distances([("no-such-trip", [0.0, 1.0], "crow_fly")])
    with pytest.raises(ValueError, match="unknown distance provenance"):
        network.set_trip_distances([("T_OK", [0.0, 1.0], "guesswork")])
    network.set_trip_distances([("T_OK", [0.0, 1500.0], "map_matched")])
    assert network.distance_provenance_counts == {"map_matched": 1}


def test_walk_legs_carry_street_paths(network_with_footpaths):
    import geopandas as gpd
    import shapely

    def utm_length(line):
        series = gpd.GeoSeries([line], crs="EPSG:4326")
        return float(series.to_crs(series.estimate_utm_crs()).length.iloc[0])

    origin = stop_coordinates(network_with_footpaths, "1100602")
    destination = stop_coordinates(network_with_footpaths, "1040280")
    journeys = network_with_footpaths.route_between_coordinates(
        origin, destination, "2022-02-22", "08:30:00"
    )
    # The walking-only journey draws its whole street path.
    (walk_leg,) = journeys[0]["legs"]
    direct = shapely.from_wkb(walk_leg["geometry"])
    assert direct.geom_type == "LineString"
    assert utm_length(direct) == pytest.approx(walk_leg["distance"], rel=0.02, abs=0.5)
    assert direct.coords[0] == pytest.approx((origin[1], origin[0]), abs=1e-6)
    assert direct.coords[-1] == pytest.approx(
        (destination[1], destination[0]), abs=1e-6
    )
    access, transit, egress = journeys[1]["legs"]
    walk = shapely.from_wkb(access["geometry"])
    assert walk.geom_type == "LineString"
    assert utm_length(walk) == pytest.approx(access["distance"], rel=0.02, abs=0.5)
    assert walk.coords[0] == pytest.approx((origin[1], origin[0]), abs=1e-6)
    walk = shapely.from_wkb(egress["geometry"])
    assert utm_length(walk) == pytest.approx(egress["distance"], rel=0.02, abs=0.5)
    assert walk.coords[-1] == pytest.approx((destination[1], destination[0]), abs=1e-6)

    # Transfer legs draw their street path on stop-to-stop journeys too.
    journeys = network_with_footpaths.route_between_stops(
        "1100602", "1040280", "2022-02-22", "08:30:00"
    )
    access, transit, transfer, egress = journeys[0]["legs"]
    walk = shapely.from_wkb(transfer["geometry"])
    assert utm_length(walk) == pytest.approx(transfer["distance"], rel=0.02, abs=0.5)
    kamppi = stop_coordinates(network_with_footpaths, "1040602")
    street_stop = stop_coordinates(network_with_footpaths, "1040280")
    assert walk.coords[0] == pytest.approx((kamppi[1], kamppi[0]), abs=1e-6)
    assert walk.coords[-1] == pytest.approx((street_stop[1], street_stop[0]), abs=1e-6)
    # Zero-length stop access/egress legs stay without geometry.
    assert access["geometry"] is None
    assert egress["geometry"] is None

    # geometries=False strips walk legs like transit legs.
    journeys = network_with_footpaths.route_between_stops(
        "1100602", "1040280", "2022-02-22", "08:30:00", geometries=False
    )
    assert all(leg["geometry"] is None for leg in journeys[0]["legs"])


def build_over_midnight_gtfs(path):
    """A feed whose only trip runs past midnight on 2022-02-21."""
    import zipfile

    tables = {
        "agency.txt": [
            "agency_id,agency_name,agency_url,agency_timezone",
            "A,Night,http://example.com,Europe/Helsinki",
        ],
        "stops.txt": [
            "stop_id,stop_name,stop_lat,stop_lon",
            "N1,First,60.0,24.0",
            "N2,Second,60.02,24.02",
        ],
        "routes.txt": [
            "route_id,route_short_name,route_type",
            "RN,N,3",
        ],
        "trips.txt": [
            "route_id,service_id,trip_id",
            "RN,NIGHT,T_NIGHT",
        ],
        "stop_times.txt": [
            "trip_id,arrival_time,departure_time,stop_id,stop_sequence",
            "T_NIGHT,25:00:00,25:00:00,N1,1",
            "T_NIGHT,25:20:00,25:20:00,N2,2",
        ],
        "calendar.txt": [
            "service_id,monday,tuesday,wednesday,thursday,friday,saturday,"
            "sunday,start_date,end_date",
            "NIGHT,1,0,0,0,0,0,0,20220221,20220221",
        ],
    }
    with zipfile.ZipFile(path, "w") as archive:
        for name, lines in tables.items():
            archive.writestr(name, "\n".join(lines) + "\n")
    return path


def test_previous_day_over_midnight_trip_is_reachable(tmp_path):
    feed = build_over_midnight_gtfs(tmp_path / "night_gtfs.zip")
    network = TransportNetwork.from_gtfs([str(feed)])

    # The trip runs on Monday 2022-02-21 with a stored time of 25:00, i.e.
    # 01:00 on the 22nd. A 00:30 query on the 22nd catches it as the
    # previous day's over-midnight service.
    journeys = network.route_between_stops("N1", "N2", "2022-02-22", "00:30:00")
    assert journeys
    access, transit, egress = journeys[0]["legs"]
    assert transit["type"] == "transit"
    assert transit["trip_id"] == "T_NIGHT"
    assert transit["departure"] == 3600  # 01:00
    assert transit["arrival"] == 4800  # 01:20
    assert journeys[0]["arrival"] == 4800

    # On its own service day the trip is only reachable by waiting out the
    # day to its stored 25:00.
    same_day = network.route_between_stops("N1", "N2", "2022-02-21", "00:30:00")
    assert same_day
    assert same_day[0]["arrival"] == 25 * 3600 + 20 * 60

    # A day later there is no previous-day service to pull it in.
    assert network.route_between_stops("N1", "N2", "2022-02-23", "00:30:00") == []
