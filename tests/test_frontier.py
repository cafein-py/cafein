"""Time × emissions frontiers over the Helsinki network and a synthetic feed."""

import zipfile

import pytest

from cafein import journey_frontier, least_emissions


def build_two_line_gtfs(path):
    """Two stops joined by a fast two-bus chain (dirty) and a slow direct
    tram (clean), twice each within 08:00–08:30.

    The range-RAPTOR candidate set keeps, per departure, the fastest
    journey of each ride count — so the trade-off spans ride counts: the
    2-ride bus chain wins on time, the 1-ride tram on emissions (shipped
    defaults: bus 92 g/pkm, tram 25 g/pkm). The 08:15 bus chain has a
    long transfer wait, making it slower than the 08:00 chain at equal
    emissions: a candidate that must fall off the frontier.
    """
    tables = {
        "agency.txt": [
            "agency_id,agency_name,agency_url,agency_timezone",
            "A,Test Agency,http://example.com,Europe/Helsinki",
        ],
        "stops.txt": [
            "stop_id,stop_name,stop_lat,stop_lon",
            "A,Origin,60.0,24.0",
            "H,Hub,60.0,24.03",
            "B,Destination,60.0,24.05",
        ],
        "routes.txt": [
            "route_id,route_short_name,route_type",
            "BUS_IN,B1,3",
            "BUS_OUT,B2,3",
            "TRAM,T1,0",
        ],
        "trips.txt": [
            "route_id,service_id,trip_id",
            "BUS_IN,SV,BUS_IN_1",
            "BUS_IN,SV,BUS_IN_2",
            "BUS_OUT,SV,BUS_OUT_1",
            "BUS_OUT,SV,BUS_OUT_2",
            "TRAM,SV,TRAM_1",
            "TRAM,SV,TRAM_2",
        ],
        "stop_times.txt": [
            "trip_id,arrival_time,departure_time,stop_id,stop_sequence",
            "BUS_IN_1,08:00:00,08:00:00,A,1",
            "BUS_IN_1,08:05:00,08:05:00,H,2",
            "BUS_OUT_1,08:07:00,08:07:00,H,1",
            "BUS_OUT_1,08:15:00,08:15:00,B,2",
            "BUS_IN_2,08:15:00,08:15:00,A,1",
            "BUS_IN_2,08:20:00,08:20:00,H,2",
            "BUS_OUT_2,08:30:00,08:30:00,H,1",
            "BUS_OUT_2,08:38:00,08:38:00,B,2",
            "TRAM_1,08:00:00,08:00:00,A,1",
            "TRAM_1,08:30:00,08:30:00,B,2",
            "TRAM_2,08:15:00,08:15:00,A,1",
            "TRAM_2,08:45:00,08:45:00,B,2",
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


def two_line_fares(tram_priced=True):
    """Buses at 4.00, trams at 6.00, and a bus–bus pair total of 10.00
    within a 10-minute allowance — dearer than two full fares, so the
    fast chain (7-minute transfer) pays 10.00 while the slow chain's
    15-minute wait breaks the window and pays 8.00."""
    import pandas as pd

    from cafein.fares import FareStructure

    routes = [("BUS_IN", "BUS", 4.0), ("BUS_OUT", "BUS", 4.0)]
    if tram_priced:
        routes.append(("TRAM", "TRAM", 6.0))
    return FareStructure(
        max_discounted_transfers=1,
        transfer_time_allowance=10.0,
        fares_per_type=pd.DataFrame(
            [
                {
                    "type": kind,
                    "unlimited_transfers": False,
                    "allow_same_route_transfer": False,
                    "use_route_fare": False,
                    "fare": fare,
                }
                for kind, fare in [("BUS", 4.0), ("TRAM", 6.0)]
            ]
        ),
        fares_per_transfer=pd.DataFrame(
            [{"first_leg": "BUS", "second_leg": "BUS", "fare": 10.0}]
        ),
        fares_per_route=pd.DataFrame(
            [
                {
                    "agency_id": "",
                    "agency_name": "",
                    "route_id": route,
                    "route_short_name": "",
                    "route_long_name": "",
                    "mode": kind,
                    "route_fare": fare,
                    "fare_type": kind,
                }
                for route, kind, fare in routes
            ]
        ),
    )


@pytest.fixture()
def two_line_frontier(tmp_path):
    from cafein import TransportNetwork

    feed = build_two_line_gtfs(tmp_path / "two_line_gtfs.zip")
    network = TransportNetwork.from_gtfs([str(feed)])
    return journey_frontier(network, "A", "B", "2022-02-22", "08:00:00", window=1800)


def test_frontier_trades_time_against_emissions(two_line_frontier):
    frame = two_line_frontier
    assert len(frame) == 4
    # Every row's emissions are its ridden meters at the shipped factors
    # (bus 92 g/pkm, tram 25 g/pkm) over the ladder's leg distances.
    for _, row in frame.iterrows():
        expected = sum(
            leg["distance"] / 1000 * (92 if leg["trip_id"].startswith("BUS") else 25)
            for leg in row["journey"]["legs"]
            if leg["type"] == "transit"
        )
        assert row["emissions"] == pytest.approx(expected)
    trams = frame[frame["rides"] == 1]
    buses = frame[frame["rides"] == 2]
    assert len(trams) == 2 and len(buses) == 2
    assert buses["emissions"].min() > trams["emissions"].max()
    assert set(trams["travel_time"]) == {1800}
    assert set(buses["travel_time"]) == {900, 1380}

    # The fast-dirty chain and the slow-clean trams are on the frontier;
    # the 08:15 chain — slower at equal emissions — is not.
    assert set(frame.loc[frame["frontier"], "travel_time"]) == {900, 1800}
    assert set(frame.loc[~frame["frontier"], "travel_time"]) == {1380}

    # The budget view: within 15 minutes only the bus chain qualifies;
    # unconstrained, the tram's lower emissions win; an impossible
    # budget yields nothing.
    assert least_emissions(frame, within=900)["rides"] == 2
    cleanest = least_emissions(frame)
    assert cleanest["rides"] == 1
    assert cleanest["emissions"] == trams["emissions"].iloc[0]
    assert least_emissions(frame, within=60) is None


def test_dominated_candidates_leave_the_frontier(two_line_frontier):
    frame = two_line_frontier
    on_frontier = frame[frame["frontier"]]
    for _, row in frame[~frame["frontier"]].iterrows():
        assert (
            (on_frontier["travel_time"] <= row["travel_time"])
            & (on_frontier["emissions"] <= row["emissions"])
        ).any()
    # Frontier rows sorted by time: emissions strictly decrease across
    # distinct travel times, and tie only at equal travel times.
    times = on_frontier["travel_time"].tolist()
    grams = on_frontier["emissions"].tolist()
    for (t1, g1), (t2, g2) in zip(zip(times, grams), zip(times[1:], grams[1:])):
        assert t1 <= t2
        assert g1 == g2 if t1 == t2 else g1 > g2


def test_fares_join_the_frontier(tmp_path):
    from cafein import TransportNetwork, least_fare

    feed = build_two_line_gtfs(tmp_path / "two_line_gtfs.zip")
    network = TransportNetwork.from_gtfs([str(feed)])
    frame = journey_frontier(
        network,
        "A",
        "B",
        "2022-02-22",
        "08:00:00",
        window=1800,
        fares=two_line_fares(),
    )
    by_time = {row["travel_time"]: row for _, row in frame.iterrows()}
    assert by_time[900]["fare"] == pytest.approx(10.0)
    assert by_time[1380]["fare"] == pytest.approx(8.0)
    assert by_time[1800]["fare"] == pytest.approx(6.0)
    # Cheapness returns the 08:15 bus chain to the frontier: dominated
    # on (time, emissions) — see the base test — but strictly cheaper
    # than the fast chain, whose short transfer pays the pair total.
    assert set(frame.loc[frame["frontier"], "travel_time"]) == {900, 1380, 1800}
    # The budget view over money: cheapest overall is the tram, the
    # bus chains under tightening time budgets, nothing within a minute.
    assert least_fare(frame)["fare"] == pytest.approx(6.0)
    assert least_fare(frame, within=1380)["fare"] == pytest.approx(8.0)
    assert least_fare(frame, within=900)["fare"] == pytest.approx(10.0)
    assert least_fare(frame, within=60) is None
    unpriced = journey_frontier(
        network, "A", "B", "2022-02-22", "08:00:00", window=1800
    )
    with pytest.raises(ValueError, match="carries no fares"):
        least_fare(unpriced)


def test_unpriceable_candidates_leave_the_frontier(tmp_path):
    from cafein import TransportNetwork

    feed = build_two_line_gtfs(tmp_path / "two_line_gtfs.zip")
    network = TransportNetwork.from_gtfs([str(feed)])
    frame = journey_frontier(
        network,
        "A",
        "B",
        "2022-02-22",
        "08:00:00",
        window=1800,
        fares=two_line_fares(tram_priced=False),
    )
    trams = frame[frame["rides"] == 1]
    assert trams["fare"].isna().all()
    assert not trams["frontier"].any()
    assert set(frame.loc[frame["frontier"], "travel_time"]) == {900, 1380}


def test_door_to_door_frontier_anchors_on_walking(network_with_footpaths):
    coordinates = {stop: (lat, lon) for stop, lat, lon in network_with_footpaths.stops}
    frame = journey_frontier(
        network_with_footpaths,
        coordinates["1100602"],
        coordinates["1040280"],
        "2022-02-22",
        "08:30:00",
        window=600,
    )
    # The walking-only journey is the zero-emission anchor …
    walk = frame[frame["rides"] == 0]
    assert len(walk) == 1
    assert walk.iloc[0]["emissions"] == 0.0
    assert bool(walk.iloc[0]["frontier"])
    assert walk.iloc[0]["journey"]["legs"][0]["type"] == "walk"
    # … and the cleanest journey overall, while a transit journey is the
    # fastest and emits at the urban-rail factor.
    assert least_emissions(frame)["rides"] == 0
    fastest = frame.iloc[0]
    assert fastest["rides"] >= 1
    assert fastest["frontier"]
    assert fastest["emissions"] > 0
    budget = least_emissions(frame, within=int(fastest["travel_time"]))
    assert budget["emissions"] == fastest["emissions"]
    # The fast end matches the single-departure oracle: the frontier
    # holds a ride arriving with the pinned fastest journey, and nothing
    # in the window beats that arrival.
    oracle = network_with_footpaths.route_between_coordinates(
        coordinates["1100602"], coordinates["1040280"], "2022-02-22", "08:30:00"
    )
    fastest_arrival = min(journey["arrival"] for journey in oracle)
    assert frame["arrival"].min() == fastest_arrival
    at_oracle = frame[(frame["arrival"] == fastest_arrival) & (frame["rides"] >= 1)]
    assert bool(at_oracle["frontier"].any())
    assert fastest["travel_time"] <= fastest_arrival - (8 * 3600 + 30 * 60)


def test_frontier_rejects_mixed_endpoints(network_with_footpaths):
    with pytest.raises(ValueError, match="both"):
        journey_frontier(
            network_with_footpaths,
            "1100602",
            (60.17, 24.94),
            "2022-02-22",
            "08:30:00",
            window=600,
        )
    with pytest.raises(ValueError, match="coordinate queries"):
        journey_frontier(
            network_with_footpaths,
            "1100602",
            "1040280",
            "2022-02-22",
            "08:30:00",
            window=600,
            max_snap_distance=100.0,
        )


def test_least_fare_survives_unresolved_emissions(network, helsinki_gtfs):
    from cafein import least_fare
    from cafein.fares import zone_fare_structure

    hsl = zone_fare_structure(helsinki_gtfs)
    # Every journey to Suomenlinna rides the factorless ferry: nothing
    # reaches the frontier and least_emissions has nothing to pick, yet
    # the journeys still price and the cheapest one is returned.
    with pytest.warns(UserWarning, match="route_type"):
        frame = journey_frontier(
            network,
            "1080701",
            "1520703",
            "2022-02-22",
            "10:00:00",
            window=3600,
            fares=hsl,
        )
    assert frame["emissions"].isna().all()
    assert not frame["frontier"].any()
    assert least_emissions(frame) is None
    cheapest = least_fare(frame)
    assert cheapest["fare"] == pytest.approx(2.8)
    assert least_fare(frame, within=1) is None


def test_unmatched_factors_poison_but_do_not_block(network):
    # The Suomenlinna ferry has no shipped factor: its journeys carry
    # NaN emissions and never join the frontier.
    with pytest.warns(UserWarning, match="route_type"):
        frame = journey_frontier(
            network, "1080701", "1520703", "2022-02-22", "10:00:00", window=3600
        )
    assert len(frame)
    assert frame["emissions"].isna().any()
    assert not frame.loc[frame["emissions"].isna(), "frontier"].any()
