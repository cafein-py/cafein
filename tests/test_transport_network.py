"""TransportNetwork built from the Helsinki GTFS feed shared with r5py."""

import pytest

from cafein import TransportNetwork


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


@pytest.fixture(scope="session")
def network_with_footpaths(helsinki_gtfs, kantakaupunki_pbf):
    with pytest.warns(UserWarning):
        return TransportNetwork.from_gtfs(
            [str(helsinki_gtfs)], osm_pbf=str(kantakaupunki_pbf)
        )


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
    walkable = network_with_footpaths.access_stops(lat, lon)
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
        network_with_footpaths.access_stops(60.14, 24.90)


def test_access_stops_need_a_street_network(network):
    with pytest.raises(ValueError, match="no street network"):
        network.access_stops(60.168, 24.931)


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
    network.set_transfers([("S2", "0:S1", 120), ("0:S1", "S2", 120)])
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


def test_set_transfers_rejects_unknown_stops(tmp_path):
    feed = build_synthetic_gtfs(tmp_path / "synthetic_gtfs.zip")
    with pytest.warns(UserWarning):
        network = TransportNetwork.from_gtfs([str(feed)])
    with pytest.raises(KeyError, match="no-such-stop"):
        network.set_transfers([("no-such-stop", "S2", 60)])


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
