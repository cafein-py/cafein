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


def test_exhaustive_frontier_agrees_with_hand_checkable_candidates(tmp_path):
    from cafein import TransportNetwork, exhaustive_frontier

    # On the two-line feed the interim candidates at a single departure
    # ARE the true frontier: the fast-dirty bus chain and the
    # slow-clean tram. The oracle must reproduce them exactly.
    feed = build_two_line_gtfs(tmp_path / "two_line_gtfs.zip")
    network = TransportNetwork.from_gtfs([str(feed)])
    true_set = exhaustive_frontier(
        network, "A", "B", "2022-02-22", "08:00:00", max_transfers=4
    )
    interim = journey_frontier(network, "A", "B", "2022-02-22", "08:00:00", window=1)
    assert len(true_set) == 2
    assert true_set["arrival"].tolist() == interim["arrival"].tolist()
    assert true_set["rides"].tolist() == interim["rides"].tolist()
    assert true_set["emissions"].tolist() == pytest.approx(
        interim["emissions"].tolist()
    )


def test_exhaustive_frontier_finds_points_the_interim_misses(network):
    from cafein import exhaustive_frontier

    # The K-train pin: Korso → Käpylä has a single true Pareto point,
    # the direct train.
    direct = exhaustive_frontier(
        network, "4810551", "1250551", "2022-02-22", "08:30:00", max_transfers=4
    )
    assert len(direct) == 1
    assert direct.iloc[0]["arrival"] == 32_280
    assert direct.iloc[0]["rides"] == 1
    assert direct.iloc[0]["emissions"] == pytest.approx(419.65, abs=0.1)

    # The documented blind spot, measured: this pair has a
    # cleaner-but-slower journey with more rides that the interim
    # (time-Pareto) candidate set cannot see.
    origin, destination = "1370104", "4960238"
    true_set = exhaustive_frontier(
        network, origin, destination, "2022-02-22", "08:30:00", max_transfers=4
    )
    interim = journey_frontier(
        network, origin, destination, "2022-02-22", "08:30:00", window=1
    )
    resolved = interim[interim["emissions"].notna()]
    # Soundness: every interim candidate is dominated-or-equalled by a
    # true frontier point (the oracle covers everything reachable).
    for row in resolved.itertuples():
        assert any(
            point.arrival <= row.arrival and point.emissions <= row.emissions + 1e-6
            for point in true_set.itertuples()
        )
    # Incompleteness of the interim: a true point no interim candidate
    # dominates or equals.
    missing = [
        point
        for point in true_set.itertuples()
        if not any(
            row.arrival <= point.arrival and row.emissions <= point.emissions + 1e-6
            for row in resolved.itertuples()
        )
    ]
    assert len(missing) == 1
    assert missing[0].emissions < resolved["emissions"].min()


def test_pareto_candidates_match_the_oracle_on_the_two_line_feed(tmp_path):
    from cafein import TransportNetwork, exhaustive_frontier

    # With a vanishing bucket the McRAPTOR candidate set is the true
    # frontier the oracle enumerates.
    feed = build_two_line_gtfs(tmp_path / "two_line_gtfs.zip")
    network = TransportNetwork.from_gtfs([str(feed)])
    true_set = exhaustive_frontier(
        network, "A", "B", "2022-02-22", "08:00:00", max_transfers=4
    )
    frame = journey_frontier(
        network,
        "A",
        "B",
        "2022-02-22",
        "08:00:00",
        window=1,
        max_transfers=4,
        candidates="pareto",
        bucket=1e-6,
    )
    assert frame["frontier"].all()
    assert frame["arrival"].tolist() == true_set["arrival"].tolist()
    assert frame["rides"].tolist() == true_set["rides"].tolist()
    assert frame["emissions"].tolist() == pytest.approx(true_set["emissions"].tolist())


def test_pareto_window_candidates_cover_the_time_candidates(tmp_path):
    from cafein import TransportNetwork

    # On the two-line feed the time-optimal window candidates are all
    # Pareto-optimal in (departure, arrival, emissions) too, and no
    # further journey is: the two profiles must coincide row for row.
    feed = build_two_line_gtfs(tmp_path / "two_line_gtfs.zip")
    network = TransportNetwork.from_gtfs([str(feed)])
    interim = journey_frontier(network, "A", "B", "2022-02-22", "08:00:00", window=1800)
    pareto = journey_frontier(
        network,
        "A",
        "B",
        "2022-02-22",
        "08:00:00",
        window=1800,
        candidates="pareto",
        bucket=1e-6,
    )
    columns = ["departure", "arrival", "rides", "emissions", "frontier"]
    ordered = [
        frame.sort_values(["departure", "arrival"]) for frame in (interim, pareto)
    ]
    for column in columns:
        assert ordered[0][column].tolist() == pytest.approx(ordered[1][column].tolist())


def test_pareto_candidates_close_the_interim_gap(network):
    from cafein import exhaustive_frontier

    # The measured blind spot of the time candidates (see the oracle
    # test above): McRAPTOR must hold the cleaner-but-slower journey the
    # interim set cannot see.
    origin, destination = "1370104", "4960238"
    true_set = exhaustive_frontier(
        network, origin, destination, "2022-02-22", "08:30:00", max_transfers=4
    )
    exact = journey_frontier(
        network,
        origin,
        destination,
        "2022-02-22",
        "08:30:00",
        window=1,
        max_transfers=4,
        candidates="pareto",
        bucket=1e-6,
    )
    on = exact[exact["frontier"]]
    assert on["arrival"].tolist() == true_set["arrival"].tolist()
    assert on["rides"].tolist() == true_set["rides"].tolist()
    assert on["emissions"].tolist() == pytest.approx(
        true_set["emissions"].tolist(), abs=1e-3
    )
    # At the default 25 g bucket the gap stays closed: a candidate
    # cleaner than everything the interim set holds, and the cleanest
    # candidate within the documented bucket band of the true optimum.
    frame = journey_frontier(
        network,
        origin,
        destination,
        "2022-02-22",
        "08:30:00",
        window=1,
        max_transfers=4,
        candidates="pareto",
    )
    interim = journey_frontier(
        network,
        origin,
        destination,
        "2022-02-22",
        "08:30:00",
        window=1,
        max_transfers=4,
    )
    resolved = interim[interim["emissions"].notna()]
    assert frame["emissions"].min() < resolved["emissions"].min()
    assert frame["emissions"].min() <= true_set["emissions"].min() + 25.0


def test_pareto_candidates_match_the_oracle_over_footpaths(network_with_footpaths):
    from cafein import exhaustive_frontier

    # The strongest cross-engine check: on the network carrying the
    # real (transitively closed) footpath set, a vanishing bucket must
    # reproduce the oracle's frontier — 20 points for this pair.
    origin, destination = "1100602", "1040280"
    true_set = exhaustive_frontier(
        network_with_footpaths,
        origin,
        destination,
        "2022-02-22",
        "08:30:00",
        max_transfers=3,
    )
    exact = journey_frontier(
        network_with_footpaths,
        origin,
        destination,
        "2022-02-22",
        "08:30:00",
        window=1,
        max_transfers=3,
        candidates="pareto",
        bucket=1e-6,
    )
    on = exact[exact["frontier"]]
    assert len(true_set) == 20
    assert on["arrival"].tolist() == true_set["arrival"].tolist()
    assert on["rides"].tolist() == true_set["rides"].tolist()
    assert on["emissions"].tolist() == pytest.approx(
        true_set["emissions"].tolist(), abs=1e-3
    )
    # And over a departure window, the widened set stays sound against
    # the time-optimal profile: every resolved interim candidate is
    # dominated or equalled by a pareto candidate.
    interim = journey_frontier(
        network_with_footpaths,
        origin,
        destination,
        "2022-02-22",
        "08:30:00",
        window=600,
    )
    pareto = journey_frontier(
        network_with_footpaths,
        origin,
        destination,
        "2022-02-22",
        "08:30:00",
        window=600,
        candidates="pareto",
        bucket=1e-6,
    )
    for row in interim[interim["emissions"].notna()].itertuples():
        assert any(
            candidate.departure >= row.departure
            and candidate.arrival <= row.arrival
            and candidate.emissions <= row.emissions + 1e-6
            for candidate in pareto.itertuples()
        )


def test_pareto_candidates_route_door_to_door(network_with_footpaths):
    coordinates = {stop: (lat, lon) for stop, lat, lon in network_with_footpaths.stops}
    origin, destination = coordinates["1100602"], coordinates["1040280"]
    frame = journey_frontier(
        network_with_footpaths,
        origin,
        destination,
        "2022-02-22",
        "08:30:00",
        window=600,
        candidates="pareto",
        bucket=1e-6,
    )
    # The walking-only journey anchors the clean end, exactly as in the
    # time-candidate door-to-door frontier.
    walk = frame[frame["rides"] == 0]
    assert len(walk) == 1
    assert walk.iloc[0]["emissions"] == 0.0
    assert bool(walk.iloc[0]["frontier"])
    assert walk.iloc[0]["journey"]["legs"][0]["type"] == "walk"
    # Whatever rides beats walking — the walk-domination rule.
    transit = frame[frame["rides"] >= 1]
    assert len(transit) > 0
    assert (transit["travel_time"] < walk.iloc[0]["travel_time"]).all()
    # Soundness against the time-optimal door-to-door profile: every
    # resolved interim candidate is dominated or equalled by a pareto
    # candidate, and both engines agree on the fastest arrival.
    interim = journey_frontier(
        network_with_footpaths,
        origin,
        destination,
        "2022-02-22",
        "08:30:00",
        window=600,
    )
    for row in interim[interim["emissions"].notna()].itertuples():
        assert any(
            candidate.departure >= row.departure
            and candidate.arrival <= row.arrival
            and candidate.emissions <= row.emissions + 1e-6
            for candidate in frame.itertuples()
        )
    assert transit["arrival"].min() == interim[interim["rides"] >= 1]["arrival"].min()


def test_the_tbtr_pareto_router_matches_mcraptor(network_with_footpaths):
    from cafein import exhaustive_frontier

    # The full McTBTR stack — factor-aware transfer set plus segment
    # scanning with query-time footpaths — against McRAPTOR and the
    # oracle on the real closed Helsinki footpath network.
    origin, destination = "1100602", "1040280"
    true_set = exhaustive_frontier(
        network_with_footpaths,
        origin,
        destination,
        "2022-02-22",
        "08:30:00",
        max_transfers=3,
    )
    frames = [
        journey_frontier(
            network_with_footpaths,
            origin,
            destination,
            "2022-02-22",
            "08:30:00",
            window=1,
            max_transfers=3,
            candidates="pareto",
            bucket=1e-6,
            router=router,
        )
        for router in ("raptor", "tbtr")
    ]
    for frame in frames:
        on = frame[frame["frontier"]]
        assert on["arrival"].tolist() == true_set["arrival"].tolist()
        assert on["rides"].tolist() == true_set["rides"].tolist()
        assert on["emissions"].tolist() == pytest.approx(
            true_set["emissions"].tolist(), abs=1e-3
        )
    # And over a window, journey for journey. The vanishing bucket
    # makes both searches exact — at coarser buckets each engine may
    # legitimately keep a different same-bucket representative.
    profiles = [
        journey_frontier(
            network_with_footpaths,
            origin,
            destination,
            "2022-02-22",
            "08:30:00",
            window=600,
            max_transfers=3,
            candidates="pareto",
            bucket=1e-6,
            router=router,
        )
        for router in ("raptor", "tbtr")
    ]
    columns = ["departure", "arrival", "rides", "emissions", "frontier"]
    ordered = [
        frame.sort_values(["departure", "arrival"]).reset_index(drop=True)
        for frame in profiles
    ]
    for column in columns:
        assert ordered[0][column].tolist() == pytest.approx(ordered[1][column].tolist())


def test_pareto_candidate_options_are_validated(tmp_path):
    from cafein import TransportNetwork

    feed = build_two_line_gtfs(tmp_path / "two_line_gtfs.zip")
    network = TransportNetwork.from_gtfs([str(feed)])
    with pytest.raises(ValueError, match="candidates"):
        journey_frontier(
            network,
            "A",
            "B",
            "2022-02-22",
            "08:00:00",
            window=1,
            candidates="fastest",
        )
    with pytest.raises(ValueError, match="bucket"):
        journey_frontier(
            network,
            "A",
            "B",
            "2022-02-22",
            "08:00:00",
            window=1,
            candidates="pareto",
            bucket=0.0,
        )
    with pytest.raises(ValueError, match="router"):
        journey_frontier(
            network,
            "A",
            "B",
            "2022-02-22",
            "08:00:00",
            window=1,
            candidates="pareto",
            router="dijkstra",
        )
    with pytest.raises(ValueError, match="candidates='pareto'"):
        journey_frontier(
            network, "A", "B", "2022-02-22", "08:00:00", window=1, router="tbtr"
        )


def _frontier_tuples(frame):
    return sorted(
        (
            int(row.departure),
            int(row.arrival),
            int(row.rides),
            round(float(row.emissions), 3),
        )
        for row in frame.itertuples()
    )


def test_relaxed_candidates_reduce_to_pareto_at_zero_slack(network):
    # slack_seconds=0 reproduces the strict pareto candidate set exactly.
    origin, destination = "1370104", "4960238"
    common = dict(window=600, max_transfers=4)
    pareto = journey_frontier(
        network,
        origin,
        destination,
        "2022-02-22",
        "08:30:00",
        candidates="pareto",
        **common,
    )
    relaxed = journey_frontier(
        network,
        origin,
        destination,
        "2022-02-22",
        "08:30:00",
        candidates="relaxed",
        slack_seconds=0,
        **common,
    )
    assert _frontier_tuples(relaxed) == _frontier_tuples(pareto)


def test_relaxed_candidates_widen_the_pareto_set(network):
    # A positive slack keeps journeys arriving within the band that strict
    # per-stop pruning drops, so the relaxed set is a strict superset of the
    # pareto set.
    origin, destination = "1370104", "4960238"
    common = dict(window=600, max_transfers=4)
    pareto = set(
        _frontier_tuples(
            journey_frontier(
                network,
                origin,
                destination,
                "2022-02-22",
                "08:30:00",
                candidates="pareto",
                **common,
            )
        )
    )
    relaxed = set(
        _frontier_tuples(
            journey_frontier(
                network,
                origin,
                destination,
                "2022-02-22",
                "08:30:00",
                candidates="relaxed",
                slack_seconds=900,
                **common,
            )
        )
    )
    assert pareto <= relaxed
    assert len(relaxed) > len(pareto)


def test_relaxed_max_options_keeps_the_frontier(network):
    # A wide slack surfaces genuinely suboptimal journeys; max_options caps
    # them but never drops a frontier journey. A cap of one returns just the
    # frontier subset, and capping at the full size returns everything.
    origin, destination = "1370104", "4960238"

    def relaxed(**kw):
        return journey_frontier(
            network,
            origin,
            destination,
            "2022-02-22",
            "08:30:00",
            window=600,
            max_transfers=4,
            candidates="relaxed",
            slack_seconds=3600,
            **kw,
        )

    widened = _frontier_tuples(relaxed())
    frontier = _frontier_tuples(relaxed(max_options=1))
    assert set(frontier) <= set(widened)
    assert len(frontier) < len(widened)
    assert _frontier_tuples(relaxed(max_options=len(widened))) == widened


def test_relaxed_candidate_options_are_validated(tmp_path):
    from cafein import TransportNetwork

    feed = build_two_line_gtfs(tmp_path / "two_line_gtfs.zip")
    network = TransportNetwork.from_gtfs([str(feed)])
    with pytest.raises(ValueError, match="candidates='pareto'"):
        journey_frontier(
            network,
            "A",
            "B",
            "2022-02-22",
            "08:00:00",
            window=1,
            candidates="relaxed",
            router="tbtr",
        )
    with pytest.raises(ValueError, match="slack"):
        journey_frontier(
            network,
            "A",
            "B",
            "2022-02-22",
            "08:00:00",
            window=1,
            candidates="relaxed",
            slack_seconds=-5,
        )
    with pytest.raises(ValueError, match="max_options"):
        journey_frontier(
            network,
            "A",
            "B",
            "2022-02-22",
            "08:00:00",
            window=1,
            candidates="relaxed",
            max_options=0,
        )
    with pytest.raises(ValueError, match="max_options"):
        journey_frontier(
            network,
            "A",
            "B",
            "2022-02-22",
            "08:00:00",
            window=1,
            candidates="relaxed",
            max_options=2.5,
        )


def _option_corridors(frame):
    return [
        frozenset(
            leg["route_id"] for leg in journey["legs"] if leg["type"] == "transit"
        )
        for journey in frame["journey"]
    ]


def _journey_signatures(frame):
    return [
        tuple(
            (
                leg["type"],
                leg.get("route_id"),
                int(leg["departure"]),
                int(leg["arrival"]),
            )
            for leg in journey["legs"]
        )
        for journey in frame["journey"]
    ]


def test_diverse_candidates_split_bus_and_tram(tmp_path):
    from cafein import TransportNetwork

    feed = build_two_line_gtfs(tmp_path / "two_line_gtfs.zip")
    network = TransportNetwork.from_gtfs([str(feed)])
    frame = journey_frontier(
        network,
        "A",
        "B",
        "2022-02-22",
        "08:00:00",
        window=1,
        candidates="diverse",
        max_options=3,
    )
    # Two disjoint corridors — the fast bus chain then the tram — even though
    # a third alternative was asked for: penalization stops when the routes
    # run out.
    corridors = _option_corridors(frame)
    assert corridors == [{"BUS_IN", "BUS_OUT"}, {"TRAM"}]
    assert corridors[0].isdisjoint(corridors[1])


def test_diverse_max_options_one_returns_the_fastest(tmp_path):
    from cafein import TransportNetwork

    feed = build_two_line_gtfs(tmp_path / "two_line_gtfs.zip")
    network = TransportNetwork.from_gtfs([str(feed)])
    frame = journey_frontier(
        network,
        "A",
        "B",
        "2022-02-22",
        "08:00:00",
        window=1,
        candidates="diverse",
        max_options=1,
    )
    # Just the single fastest corridor — the 900 s bus chain, not the tram.
    assert len(frame) == 1
    assert _option_corridors(frame) == [{"BUS_IN", "BUS_OUT"}]
    assert int(frame["travel_time"].iloc[0]) == 900


def test_diverse_candidates_are_route_disjoint(network):
    # A pair with several corridors: the alternatives ride mutually disjoint
    # line sets, ordered fastest-first.
    frame = journey_frontier(
        network,
        "1370104",
        "4960238",
        "2022-02-22",
        "08:30:00",
        window=1,
        max_transfers=6,
        candidates="diverse",
        max_options=3,
    )
    corridors = _option_corridors(frame)
    assert len(corridors) >= 2
    for i, first in enumerate(corridors):
        for second in corridors[i + 1 :]:
            assert first.isdisjoint(second)
    arrivals = [int(journey["arrival"]) for journey in frame["journey"]]
    assert arrivals == sorted(arrivals)


def test_diverse_candidates_stop_at_a_single_corridor(network):
    # Only the K train reaches this pair; banning it leaves nothing, so a
    # request for five alternatives returns the one corridor.
    frame = journey_frontier(
        network,
        "4810551",
        "1250551",
        "2022-02-22",
        "08:30:00",
        window=1,
        candidates="diverse",
        max_options=5,
    )
    assert len(frame) == 1


def test_diverse_candidate_options_are_validated(tmp_path):
    from cafein import TransportNetwork

    feed = build_two_line_gtfs(tmp_path / "two_line_gtfs.zip")
    network = TransportNetwork.from_gtfs([str(feed)])
    with pytest.raises(ValueError, match="candidates='pareto'"):
        journey_frontier(
            network,
            "A",
            "B",
            "2022-02-22",
            "08:00:00",
            window=1,
            candidates="diverse",
            router="tbtr",
        )
    with pytest.raises(ValueError, match="max_options"):
        journey_frontier(
            network,
            "A",
            "B",
            "2022-02-22",
            "08:00:00",
            window=1,
            candidates="diverse",
            max_options=0,
        )
    with pytest.raises(ValueError, match="diversity must be"):
        journey_frontier(
            network,
            "A",
            "B",
            "2022-02-22",
            "08:00:00",
            window=1,
            candidates="diverse",
            diversity="closest",
        )
    with pytest.raises(ValueError, match="slack_seconds"):
        journey_frontier(
            network,
            "A",
            "B",
            "2022-02-22",
            "08:00:00",
            window=1,
            candidates="diverse",
            slack_seconds=-1,
        )
    for bad in (-5, 0, "nope"):
        with pytest.raises(ValueError, match="penalty must be"):
            journey_frontier(
                network,
                "A",
                "B",
                "2022-02-22",
                "08:00:00",
                window=1,
                candidates="diverse",
                penalty=bad,
            )
    with pytest.raises(ValueError, match="penalty applies only"):
        journey_frontier(
            network,
            "A",
            "B",
            "2022-02-22",
            "08:00:00",
            window=1,
            candidates="pareto",
            penalty=300,
        )


def test_slack_seconds_defaults_are_per_family(network):
    # The None-sentinel default resolves per family, so existing calls are
    # unchanged: "relaxed" still defaults to a 300 s band, "diverse" to 0
    # (strict pareto per round).
    args = ("1370104", "4960238", "2022-02-22", "08:30:00")
    relaxed_default = journey_frontier(
        network, *args, window=1800, candidates="relaxed"
    )
    relaxed_300 = journey_frontier(
        network, *args, window=1800, candidates="relaxed", slack_seconds=300.0
    )
    assert relaxed_default.equals(relaxed_300)
    diverse_default = journey_frontier(
        network, *args, window=1, max_transfers=6, candidates="diverse", max_options=3
    )
    diverse_0 = journey_frontier(
        network,
        *args,
        window=1,
        max_transfers=6,
        candidates="diverse",
        max_options=3,
        slack_seconds=0.0,
    )
    assert diverse_default.equals(diverse_0)


def test_relaxed_diverse_widens_the_round_pool(network):
    # A positive slack_seconds widens each penalization round's McRAPTOR pool to
    # the relaxed frontier (relaxed × diverse), so "spread" can pick a
    # slightly-suboptimal but more distinct corridor than the strict-pareto pool
    # offers — here a far, slower corridor the strict set never reaches.
    common = dict(
        window=1,
        max_transfers=6,
        candidates="diverse",
        max_options=4,
        diversity="spread",
    )
    args = ("1281160", "1320107", "2022-02-22", "08:30:00")
    strict = journey_frontier(network, *args, slack_seconds=0.0, **common)
    widened = journey_frontier(network, *args, slack_seconds=1800.0, **common)
    # Same fastest seed, still route-disjoint, but a different corridor set that
    # reaches further across the trade-off.
    assert _option_corridors(widened)[0] == _option_corridors(strict)[0]
    assert _option_corridors(widened) != _option_corridors(strict)
    assert widened["travel_time"].max() > strict["travel_time"].max()
    for corridors in (_option_corridors(strict), _option_corridors(widened)):
        for i, first in enumerate(corridors):
            for second in corridors[i + 1 :]:
                assert first.isdisjoint(second)


def test_relaxed_window_is_the_r5py_equivalent(network):
    # candidates="relaxed" over a departure window is R5's detailed-itinerary
    # strategy: a McRAPTOR profile across the window under a per-stop suboptimal
    # slack, with no route penalty. Compared over the same window against
    # "diverse" (iterative route penalization onto disjoint corridors), relaxed
    # does not force route-disjointness — trunk-sharing alternatives survive, as
    # in R5 — and, because disjoint corridors run out while trunk-sharing ones do
    # not, it surfaces more alternatives than the disjoint set.
    args = ("1281160", "1320107", "2022-02-22", "08:30:00")
    common = dict(window=600, max_transfers=6)
    diverse = journey_frontier(
        network,
        *args,
        **common,
        candidates="diverse",
        max_options=4,
        diversity="spread",
    )
    relaxed = journey_frontier(
        network,
        *args,
        **common,
        candidates="relaxed",
        slack_seconds=900,
    )
    div = _option_corridors(diverse)
    rel = _option_corridors(relaxed)
    # diverse forces pairwise route-disjoint corridors...
    assert all(a.isdisjoint(b) for i, a in enumerate(div) for b in div[i + 1 :])
    # ...relaxed does not: at least one pair of options shares a route.
    assert any(not a.isdisjoint(b) for i, a in enumerate(rel) for b in rel[i + 1 :])
    # over the same window, allowing trunk-sharing surfaces more alternatives
    # than forcing disjointness — and diverse returns fewer than its
    # max_options=4 cap, so the gap is exhausted disjoint corridors, not the cap.
    assert len(diverse) < 4
    assert len(relaxed) > len(diverse)
    # like r5py, the alternatives are deduplicated: no two options are the same
    # journey. (Arrivals span more than slack_seconds here — over a window the
    # slack is a per-stop dominance margin, not a global arrival bound.)
    signatures = _journey_signatures(relaxed)
    assert len(signatures) == len(set(signatures))


def test_diverse_time_reproduces_the_default(network):
    # diversity="time" is the default objective, so it reproduces the diverse
    # set returned without the argument.
    common = dict(window=1, max_transfers=6, candidates="diverse", max_options=3)
    default = journey_frontier(
        network, "1370104", "4960238", "2022-02-22", "08:30:00", **common
    )
    explicit = journey_frontier(
        network,
        "1370104",
        "4960238",
        "2022-02-22",
        "08:30:00",
        diversity="time",
        **common,
    )
    assert explicit.equals(default)


def test_diverse_spread_keeps_picking_past_the_walking_journey(network_with_footpaths):
    # A routeless pick — the walking-only journey, spread's clean-slow corner —
    # bans nothing, so the rounds keep selecting from the same pool instead of
    # ending the search: the options hold the walk AND the transit corridors
    # (this pair used to return two options, the fastest corridor and the walk).
    coordinates = {stop: (lat, lon) for stop, lat, lon in network_with_footpaths.stops}
    frame = journey_frontier(
        network_with_footpaths,
        coordinates["1100602"],
        coordinates["1040280"],
        "2022-02-22",
        "08:30:00",
        window=1,
        candidates="diverse",
        diversity="spread",
        max_options=4,
    )
    corridors = _option_corridors(frame)
    transit = [c for c in corridors if c]
    assert len(frame) == 4
    assert sum(1 for c in corridors if not c) == 1  # the walking-only option
    assert len(transit) == 3
    for i, first in enumerate(transit):
        for second in transit[i + 1 :]:
            assert first.isdisjoint(second)


def test_diverse_spread_reaches_across_the_trade_off(network):
    # The same disjoint corridors, but the objective changes which three are
    # kept: "time" takes the three fastest; "spread" seeds on the fastest, then
    # reaches the far (slow-clean) corner the fastest-first set skips.
    common = dict(window=1, max_transfers=6, candidates="diverse", max_options=3)
    fast = journey_frontier(
        network,
        "1370104",
        "4960238",
        "2022-02-22",
        "08:30:00",
        diversity="time",
        **common,
    )
    spread = journey_frontier(
        network,
        "1370104",
        "4960238",
        "2022-02-22",
        "08:30:00",
        diversity="spread",
        **common,
    )
    assert len(spread) == len(fast) == 3
    # Both seed on the same fastest corridor and stay route-disjoint.
    assert _option_corridors(spread)[0] == _option_corridors(fast)[0]
    # Spread reaches a corridor slower than any the fastest-first set kept.
    assert spread["travel_time"].max() > fast["travel_time"].max()
    assert _option_corridors(spread) != _option_corridors(fast)
    # That far corner is cleaner than the fastest-first set's slowest corridor,
    # so the options span the emissions trade-off, not only travel time.
    spread_slowest = spread.loc[spread["travel_time"].idxmax()]
    fast_slowest = fast.loc[fast["travel_time"].idxmax()]
    assert spread_slowest["emissions"] < fast_slowest["emissions"]


def test_diverse_penalty_ban_matches_the_default(network):
    # penalty="ban" is the default hard-disjoint behaviour, unchanged.
    args = ("1281160", "1320107", "2022-02-22", "08:30:00")
    common = dict(
        window=1,
        max_transfers=6,
        candidates="diverse",
        max_options=4,
        diversity="spread",
    )
    default = journey_frontier(network, *args, **common)
    ban = journey_frontier(network, *args, penalty="ban", **common)
    assert _option_corridors(default) == _option_corridors(ban)
    assert _frontier_tuples(default) == _frontier_tuples(ban)


def test_diverse_soft_penalty_shares_trunks_and_finds_more(network):
    # A hard ban forces route-disjoint corridors and dries up fast; a soft
    # penalty makes a used route costly-but-usable, so a corridor sharing a
    # trunk can surface and the set holds more options.
    args = ("1281160", "1320107", "2022-02-22", "08:30:00")
    common = dict(
        window=1,
        max_transfers=6,
        candidates="diverse",
        max_options=5,
        diversity="spread",
    )
    ban = journey_frontier(network, *args, penalty="ban", **common)
    soft = journey_frontier(network, *args, penalty=600, **common)
    bc = _option_corridors(ban)
    sc = _option_corridors(soft)
    # ban is fully route-disjoint...
    assert all(a.isdisjoint(b) for i, a in enumerate(bc) for b in bc[i + 1 :])
    # ...the soft penalty lets two options share a route...
    assert any(not a.isdisjoint(b) for i, a in enumerate(sc) for b in sc[i + 1 :])
    # ...and surfaces more options before drying up.
    assert len(soft) > len(ban)
    # The fastest seed is picked before any penalty applies, so it is unchanged.
    assert soft["travel_time"].min() == ban["travel_time"].min()


def test_diverse_soft_penalty_reports_true_times(network):
    # The penalty steers the search through the dominance only; it must never
    # inflate a reported time. A penalty far larger than any real trip leaves
    # every reported travel time realistic (nowhere near the penalty).
    frame = journey_frontier(
        network,
        "1281160",
        "1320107",
        "2022-02-22",
        "08:30:00",
        window=1,
        max_transfers=6,
        candidates="diverse",
        max_options=4,
        diversity="spread",
        penalty=100000,
    )
    assert len(frame) >= 1
    assert frame["travel_time"].max() < 10000


def test_diverse_soft_penalty_clamps_a_huge_value(network):
    # A penalty far beyond the engine's 32-bit cap (and the binding's integer
    # width) is clamped on accumulation rather than overflowing the boundary
    # conversion, so the call still returns journeys.
    frame = journey_frontier(
        network,
        "1281160",
        "1320107",
        "2022-02-22",
        "08:30:00",
        window=1,
        max_transfers=6,
        candidates="diverse",
        max_options=3,
        diversity="spread",
        penalty=10**20,
    )
    assert len(frame) >= 1


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
