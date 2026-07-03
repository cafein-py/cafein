"""TransportNetwork built from the Helsinki GTFS feed shared with r5py."""

import pytest

from cafein import TransportNetwork


@pytest.fixture(scope="session")
def network(helsinki_gtfs):
    return TransportNetwork.from_gtfs([str(helsinki_gtfs)])


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
