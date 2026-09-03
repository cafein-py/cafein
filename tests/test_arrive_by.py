"""``arrival=`` on the routing calls: the arrive-by time axis."""

import pytest

KORSO, KAPYLA = "4810551", "1250551"
DEADLINE = "2022-02-22 09:30:00"
DEADLINE_S = 9 * 3600 + 30 * 60
KAMPPI = (60.1689, 24.9330)
HAKANIEMI = (60.1795, 24.9520)


def clock(seconds):
    return (
        f"2022-02-22 {seconds // 3600:02d}:"
        f"{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
    )


def tuples(journeys):
    return [(j["departure_s"], j["rides"], j["arrival_s"]) for j in journeys]


def inverted_profile(journeys, deadline_s):
    """The latest-departure Pareto set of a forward window profile."""
    candidates = sorted(
        (
            (j["departure_s"], j["rides"], j["arrival_s"])
            for j in journeys
            if j["arrival_s"] <= deadline_s
        ),
        key=lambda entry: (-entry[0], entry[1], entry[2]),
    )
    kept = []
    for departure, rides, arrival in candidates:
        dominated = any(
            d >= departure and r <= rides and (d > departure or r < rides)
            for d, r, _ in kept
        )
        duplicate = any(d == departure and r == rides for d, r, _ in kept)
        if not dominated and not duplicate:
            kept.append((departure, rides, arrival))
    return kept


def test_time_axis_arguments_are_validated(network, network_with_footpaths):
    from cafein import StreetLegPolicy

    with pytest.raises(ValueError, match="exactly one of departure= or arrival="):
        network.route_between_stops(KORSO, KAPYLA)
    with pytest.raises(ValueError, match="exactly one of departure= or arrival="):
        network.route_between_stops(
            KORSO, KAPYLA, "2022-02-22 08:30:00", arrival=DEADLINE
        )
    with pytest.raises(ValueError, match="exactly one of departure= or arrival="):
        network.travel_times_from_stop(KAPYLA)
    with pytest.raises(ValueError, match="exactly one of departure= or arrival="):
        network.travel_times_from_stop(KAPYLA, "2022-02-22 08:30:00", arrival=DEADLINE)
    # The axes reject each other's windows.
    with pytest.raises(ValueError, match="departure_time_window"):
        network.route_between_stops(
            KORSO, KAPYLA, arrival=DEADLINE, departure_time_window=30
        )
    with pytest.raises(ValueError, match="arrival_time_window"):
        network.route_between_stops(
            KORSO, KAPYLA, arrival=DEADLINE, departure_time_window=30
        )
    with pytest.raises(ValueError, match="beside arrival="):
        network.route_between_stops(
            KORSO, KAPYLA, "2022-02-22 08:30:00", arrival_time_window=30
        )
    # A street policy does not combine with arrival=.
    policy = StreetLegPolicy(access={"walk": 1800}, egress={"walk": 1800})
    with pytest.raises(ValueError, match="street_policy"):
        network_with_footpaths.route_between_coordinates(
            KAMPPI, HAKANIEMI, arrival=DEADLINE, street_policy=policy
        )
    with pytest.raises(ValueError, match="street_policy"):
        network_with_footpaths.travel_times_from_coordinate(
            HAKANIEMI, arrival=DEADLINE, street_policy=policy
        )


def test_the_reverse_answer_inverts_the_forward_window_profile(network):
    # The exact bracket: a forward departure-window profile spanning the
    # morning holds every latest-departure journey the reverse axis may
    # return; its inversion at the deadline is the arrive-by answer.
    reverse = network.route_between_stops(KORSO, KAPYLA, arrival=DEADLINE)
    assert reverse
    profile = network.route_between_stops(
        KORSO, KAPYLA, "2022-02-22 06:00:00", departure_time_window=210
    )
    assert tuples(reverse) == inverted_profile(profile, DEADLINE_S)
    # The inversion pins the deadline and the ordering: arrivals fit the
    # deadline, latest departure first.
    assert all(j["arrival_s"] <= DEADLINE_S for j in reverse)
    departures = [j["departure_s"] for j in reverse]
    assert departures == sorted(departures, reverse=True)
    # Travel time is each journey's own duration: the latest departure
    # rides the 16-minute K train, not the span to the deadline.
    best = reverse[0]
    assert best["arrival_s"] - best["departure_s"] < 30 * 60


def test_the_inversion_holds_over_the_production_closure(network_with_footpaths):
    # The same bracket over the pyrosm-built walking closure — the
    # production transfer set — for the K-train pair and an inner-city
    # pair whose journeys ride the footpaths.
    for origin, destination in [(KORSO, KAPYLA), ("1020453", "1070422")]:
        reverse = network_with_footpaths.route_between_stops(
            origin, destination, arrival=DEADLINE
        )
        assert reverse
        profile = network_with_footpaths.route_between_stops(
            origin, destination, "2022-02-22 06:00:00", departure_time_window=210
        )
        assert tuples(reverse) == inverted_profile(profile, DEADLINE_S)


def test_arrive_by_journeys_are_the_departure_answers(network_with_footpaths):
    # Leg identity, not just tuple identity: each arrive-by journey is
    # byte-identical to the departure= answer for the departure it
    # discovered.
    reverse = network_with_footpaths.route_between_stops(
        KORSO, KAPYLA, arrival=DEADLINE
    )
    assert reverse
    for journey in reverse:
        forward = network_with_footpaths.route_between_stops(
            KORSO, KAPYLA, clock(journey["departure_s"])
        )
        matched = [
            candidate
            for candidate in forward
            if candidate["arrival_s"] == journey["arrival_s"]
            and candidate["rides"] == journey["rides"]
        ]
        assert matched == [journey]


def test_travel_times_flip_to_the_destination(network):
    times = network.travel_times_from_stop(KAPYLA, arrival=DEADLINE)
    reverse = network.route_between_stops(KORSO, KAPYLA, arrival=DEADLINE)
    assert times[KAPYLA] == 0
    best = reverse[0]
    assert times[KORSO] == best["arrival_s"] - best["departure_s"]


def test_coordinate_arrival_times_include_the_walk(network_with_footpaths):
    times = network_with_footpaths.travel_times_from_coordinate(
        HAKANIEMI, arrival=DEADLINE
    )
    assert times
    walks = network_with_footpaths.access_stops(*HAKANIEMI)
    nearest = min(walks, key=walks.get)
    # The nearest stop walks straight to the destination, the walk
    # placed to arrive exactly at the deadline.
    assert times[nearest] == walks[nearest]


def test_the_direct_walk_arrives_exactly_at_the_deadline(network_with_footpaths):
    journeys = network_with_footpaths.route_between_coordinates(
        KAMPPI, HAKANIEMI, arrival=DEADLINE
    )
    walks = [j for j in journeys if j["rides"] == 0]
    assert len(walks) == 1
    walk = walks[0]
    assert walk["legs"][0]["type"] == "walk"
    assert walk["arrival_s"] == DEADLINE_S
    assert walk["departure_s"] == DEADLINE_S - (walk["arrival_s"] - walk["departure_s"])
    departures = [j["departure_s"] for j in journeys]
    assert departures == sorted(departures, reverse=True)
    assert all(j["arrival_s"] <= DEADLINE_S for j in journeys)
    # The walk dominates any ridden journey leaving no later than it,
    # so every kept journey departs strictly after the walk and the
    # walk trails the list.
    assert journeys[-1] == walk
    for journey in journeys:
        if journey["rides"]:
            assert journey["departure_s"] > walk["departure_s"]


def test_exclusions_apply_on_the_reverse_axis(network):
    from cafein import TravelerProfile

    reverse = network.route_between_stops(KORSO, KAPYLA, arrival=DEADLINE)
    ridden = reverse[0]["legs"][1]["route_id"]
    without = network.route_between_stops(
        KORSO, KAPYLA, arrival=DEADLINE, exclude_routes=[ridden]
    )
    assert without
    assert all(
        leg.get("route_id") != ridden for journey in without for leg in journey["legs"]
    )
    # A traveler profile is the same exclusion, spelled as a profile.
    profile = TravelerProfile(exclude_routes=[ridden])
    via_profile = network.route_between_stops(
        KORSO, KAPYLA, arrival=DEADLINE, traveler=profile
    )
    assert via_profile
    assert via_profile == without
    assert all(
        leg.get("route_id") != ridden
        for journey in via_profile
        for leg in journey["legs"]
    )
    # Without the K train the latest alternative leaves before six;
    # the bracket spans the whole early morning.
    forward = network.route_between_stops(
        KORSO,
        KAPYLA,
        "2022-02-22 04:00:00",
        departure_time_window=330,
        exclude_routes=[ridden],
    )
    assert tuples(without) == inverted_profile(forward, DEADLINE_S)


def test_the_matrix_cells_are_the_routing_calls_durations(network):
    from cafein import TravelTimeMatrix

    # The chunk slices the destination axis; pick the block holding
    # Käpylä so the sample has reachable pairs.
    stops = [stop for stop, *_ in network._core.stops]
    block = -(-len(stops) // 400)
    frame = TravelTimeMatrix(
        network,
        origins=[KORSO, "1020453"],
        arrival=DEADLINE,
        chunk=(stops.index(KAPYLA) // block, 400),
        output_time_units="seconds",
    )
    assert KAPYLA in set(frame["to_id"])
    for row in frame.itertuples(index=False):
        journeys = network.route_between_stops(row.from_id, row.to_id, arrival=DEADLINE)
        best = journeys[0]
        assert row.travel_time == best["arrival_s"] - best["departure_s"]
    # The wide matrix chunks the same destination axis.
    matrix = network.travel_time_matrix([KORSO], arrival=DEADLINE, chunk=(0, 400))
    stop_total = network.stop_count
    expected = stop_total // 400 + (1 if stop_total % 400 else 0)
    assert matrix.shape == (1, expected)


def test_the_point_matrix_matches_the_coordinate_call(network_with_footpaths):
    import geopandas as gpd
    from shapely.geometry import Point

    from cafein import TravelTimeMatrix

    coordinates = [KAMPPI, HAKANIEMI]
    points = gpd.GeoDataFrame(
        {"id": ["kamppi", "hakaniemi"]},
        geometry=[Point(lon, lat) for lat, lon in coordinates],
        crs="EPSG:4326",
    )
    frame = TravelTimeMatrix(
        network_with_footpaths,
        origins=points,
        arrival=DEADLINE,
        output_time_units="seconds",
    )
    cells = {
        (row.from_id, row.to_id): row.travel_time
        for row in frame.itertuples(index=False)
    }
    lookup = dict(zip(points["id"], coordinates))
    for (from_id, to_id), seconds in cells.items():
        if from_id == to_id:
            continue
        journeys = network_with_footpaths.route_between_coordinates(
            lookup[from_id], lookup[to_id], arrival=DEADLINE
        )
        best = journeys[0]
        assert seconds == best["arrival_s"] - best["departure_s"]
    assert cells[("kamppi", "hakaniemi")]


def test_itinerary_rows_reconcile_with_the_routing_call(network):
    from cafein import DetailedItineraries

    frame = DetailedItineraries(
        network,
        origins=[KORSO],
        destinations=[KAPYLA],
        arrival=DEADLINE,
        output_time_units="seconds",
    )
    journeys = network.route_between_stops(KORSO, KAPYLA, arrival=DEADLINE)
    assert sorted(frame["option"].unique()) == list(range(len(journeys)))
    for option, journey in enumerate(journeys):
        rows = frame[frame["option"] == option].sort_values("segment")
        assert len(rows) == len(journey["legs"])
        assert list(rows["leg_type"]) == [leg["type"] for leg in journey["legs"]]
        assert rows["departure_s"].iloc[0] == journey["departure_s"]
        assert rows["arrival_s"].iloc[-1] == journey["arrival_s"]
    # Options order latest departure first, like the routing call.
    departures = [journey["departure_s"] for journey in journeys]
    assert departures == sorted(departures, reverse=True)


def test_computer_rejections_on_the_arrival_axis(network, network_with_footpaths):
    import pandas as pd

    from cafein import (
        Accessibility,
        Catchment,
        DetailedItineraries,
        NearestDestinations,
        TravelCostMatrix,
        TravelTimeMatrix,
        travel_cost_table,
    )

    with pytest.raises(ValueError, match="exactly one of departure= or arrival="):
        TravelTimeMatrix(network, origins=[KORSO], departure=DEADLINE, arrival=DEADLINE)
    with pytest.raises(ValueError, match="departure_time_window"):
        TravelTimeMatrix(
            network, origins=[KORSO], arrival=DEADLINE, departure_time_window=30
        )
    with pytest.raises(ValueError, match="router='tbtr'"):
        TravelTimeMatrix(network, origins=[KORSO], arrival=DEADLINE, router="tbtr")
    with pytest.raises(ValueError, match="router='tbtr'"):
        TravelTimeMatrix.to_parquet(
            network,
            origins=[KORSO],
            arrival=DEADLINE,
            router="tbtr",
            output="/tmp/never.parquet",
        )
    with pytest.raises(ValueError, match="candidates='pareto'"):
        DetailedItineraries(
            network,
            origins=[KORSO],
            destinations=[KAPYLA],
            arrival=DEADLINE,
            candidates="pareto",
        )
    with pytest.raises(ValueError, match="exactly one of departure= or arrival="):
        DetailedItineraries(
            network,
            origins=[KORSO],
            destinations=[KAPYLA],
            departure=DEADLINE,
            arrival=DEADLINE,
        )
    with pytest.raises(ValueError, match="exactly one of departure= or arrival="):
        DetailedItineraries(network, origins=[KORSO], destinations=[KAPYLA])
    with pytest.raises(ValueError, match="arrival_time_window"):
        Accessibility(network, [KORSO], [KAPYLA], arrival=DEADLINE, cost="emissions")
    with pytest.raises(ValueError, match="arrival_time_window"):
        Accessibility(
            network,
            [KORSO],
            [KAPYLA],
            arrival=DEADLINE,
            departure_time_window=60,
        )
    with pytest.raises(ValueError, match="exactly one of departure= or arrival="):
        Accessibility(network, [KORSO], [KAPYLA], DEADLINE, arrival=DEADLINE)
    with pytest.raises(ValueError, match="router must be"):
        Accessibility(network, [KORSO], [KAPYLA], arrival=DEADLINE, router="bogus")
    with pytest.raises(ValueError, match="percentiles"):
        Accessibility(network, [KORSO], [KAPYLA], arrival=DEADLINE, percentiles=[50])
    # The cost folds carry their own arrival contract.
    with pytest.raises(ValueError, match="optimize='emissions' or 'money'"):
        TravelCostMatrix(
            network, origins=[KORSO], arrival=DEADLINE, arrival_time_window=20
        )
    with pytest.raises(ValueError, match="arrival_time_window"):
        TravelCostMatrix(
            network, origins=[KORSO], arrival=DEADLINE, optimize="emissions"
        )
    with pytest.raises(ValueError, match="reverse search rides RAPTOR"):
        TravelCostMatrix(
            network,
            origins=[KORSO],
            arrival=DEADLINE,
            arrival_time_window=20,
            optimize="emissions",
            router="tbtr",
        )
    with pytest.raises(ValueError, match="departure_time_window= profiles"):
        TravelCostMatrix(
            network,
            origins=[KORSO],
            arrival=DEADLINE,
            departure_time_window=20,
            optimize="emissions",
        )
    with pytest.raises(ValueError, match="multicriteria arrive-by"):
        TravelCostMatrix(
            network,
            origins=[KORSO],
            arrival=DEADLINE,
            arrival_time_window=20,
            optimize="emissions",
            candidates="pareto",
        )
    with pytest.raises(ValueError, match="reverse search rides RAPTOR"):
        travel_cost_table(
            network,
            origins=[KORSO],
            arrival=DEADLINE,
            arrival_time_window=20,
            optimize="emissions",
            router="tbtr",
        )
    destinations = pd.DataFrame({"id": [KAPYLA], "reachable": [1]})
    for bare in (
        lambda: Accessibility(
            network,
            origins=[KORSO],
            destinations=destinations,
            arrival=DEADLINE,
            cost="emissions",
            budgets=[100.0],
        ),
        lambda: NearestDestinations(
            network,
            origins=[KORSO],
            destinations=[KAPYLA],
            arrival=DEADLINE,
            cost="money",
            k=1,
        ),
        lambda: Catchment(
            network,
            origins=[KAPYLA],
            arrival=DEADLINE,
            cost="emissions",
            budgets=[100.0],
        ),
    ):
        with pytest.raises(ValueError, match="arrival_time_window"):
            bare()
    # The nearest fast path keeps the fan-out's argument rejections.
    with pytest.raises(ValueError, match="apply to a"):
        NearestDestinations(
            network,
            [KORSO],
            [KAPYLA],
            arrival=DEADLINE,
            k=1,
            transport_mode="walk",
        )
    for k in (1, 2):
        with pytest.raises(ValueError, match="max_cost"):
            NearestDestinations(
                network, [KORSO], [KAPYLA], arrival=DEADLINE, k=k, max_cost=0
            )
    # The catchment rejects the departure window on the arrival axis.
    pytest.importorskip("h3")
    with pytest.raises(ValueError, match="departure_time_window"):
        Catchment(
            network_with_footpaths,
            [KAPYLA],
            arrival=DEADLINE,
            departure_time_window=60,
        )


def test_accessibility_scores_recompute_from_the_reverse_durations(network):
    from cafein import Accessibility

    origins = [KORSO, "1020453"]
    destinations = [KAPYLA, "1070422"]
    frame = Accessibility(
        network, origins, destinations, arrival=DEADLINE, budgets=[30, 60]
    )
    durations = {
        (origin, destination): (
            journeys[0]["arrival_s"] - journeys[0]["departure_s"]
            if (
                journeys := network.route_between_stops(
                    origin, destination, arrival=DEADLINE
                )
            )
            else None
        )
        for origin in origins
        for destination in destinations
    }
    for row in frame.itertuples(index=False):
        expected = sum(
            1
            for destination in destinations
            if durations[(row.from_id, destination)] is not None
            and durations[(row.from_id, destination)] <= row.budget * 60
        )
        assert row.accessibility == expected


def test_nearest_destinations_rank_by_reverse_durations(network):
    from cafein import NearestDestinations

    destinations = [KAPYLA, "1070422"]
    frame = NearestDestinations(
        network,
        [KORSO],
        destinations,
        arrival=DEADLINE,
        k=2,
        output_time_units="seconds",
    )
    durations = {}
    for destination in destinations:
        journeys = network.route_between_stops(KORSO, destination, arrival=DEADLINE)
        if journeys:
            durations[destination] = journeys[0]
    expected = sorted(
        (best["arrival_s"] - best["departure_s"], stop)
        for stop, best in durations.items()
    )
    # Unreachable destinations have no rank row — absence, exactly as
    # on the departure axis.
    assert expected
    assert list(frame["destination_id"]) == [stop for _, stop in expected]
    assert list(frame["cost"]) == [cost for cost, _ in expected]


def test_the_arrive_by_catchment_reaches_the_place_by_the_deadline(
    network_with_footpaths,
):
    pytest.importorskip("h3")
    from shapely.geometry import Point

    from cafein import Catchment

    reverse = Catchment(
        network_with_footpaths, [KAPYLA], arrival=DEADLINE, budgets=[15, 30]
    )
    assert set(reverse["budget"]) == {15, 30}
    small = reverse[reverse["budget"] == 15].geometry.iloc[0]
    large = reverse[reverse["budget"] == 30].geometry.iloc[0]
    # Nested budgets nest; the destination's own location belongs to
    # every budget's region.
    assert large.covers(small.buffer(-1e-9))
    stop = dict((s, (lat, lon)) for s, lat, lon in network_with_footpaths.stops)[KAPYLA]
    assert small.covers(Point(stop[1], stop[0]))
    # "Be there by 9:30" differs from "leave at 9:30".
    forward = Catchment(network_with_footpaths, [KAPYLA], DEADLINE, budgets=[30])
    assert not large.equals(forward.geometry.iloc[0])


def test_street_itineraries_place_the_clock_at_the_deadline(helsinki_streets):
    import geopandas as gpd

    from cafein import DetailedItineraries

    places = gpd.GeoDataFrame(
        {"id": ["kamppi", "hakaniemi"]},
        geometry=gpd.points_from_xy([24.9320, 24.9520], [60.1690, 60.1795]),
        crs="EPSG:4326",
    )
    routes = DetailedItineraries(
        helsinki_streets,
        places,
        transport_mode="walk",
        arrival="09:30",
        output_time_units="seconds",
    )
    assert len(routes)
    assert (routes["arrival_s"] == DEADLINE_S).all()
    assert (routes["departure_s"] == DEADLINE_S - routes["travel_time"]).all()
    departing = DetailedItineraries(
        helsinki_streets,
        places,
        transport_mode="walk",
        departure="09:30",
        output_time_units="seconds",
    )
    assert list(routes["travel_time"]) == list(departing["travel_time"])


def test_the_wheelchair_bridge_rejects_the_arrival_axis(multimodal_network):
    pytest.importorskip("h3")
    from cafein import Catchment, TravelerProfile

    with pytest.raises(ValueError, match="does not combine with arrival="):
        Catchment(
            multimodal_network,
            [KAPYLA],
            arrival=DEADLINE,
            traveler=TravelerProfile(wheelchair=True),
        )


def test_catchment_seeds_derive_from_the_reverse_states(network_with_footpaths):
    # The derivational check: the catchment's stop seeds are exactly
    # the reverse one-to-all winners — per stop the routing call's
    # first journey (departure, rides, achieved), and the one-to-all
    # surface reads the same winners' durations.
    reaches = network_with_footpaths._core._arrive_by_reaches(
        [(KAPYLA, 0)], "2022-02-22", "09:30:00", 7, [], [], []
    )
    stops = [stop for stop, *_ in network_with_footpaths._core.stops]
    by_stop = {
        stops[stop]: (departure, rides, achieved)
        for stop, departure, rides, achieved in reaches
    }
    for origin in [KORSO, "1020453"]:
        best = network_with_footpaths.route_between_stops(
            origin, KAPYLA, arrival=DEADLINE
        )[0]
        assert by_stop[origin] == (
            best["departure_s"],
            best["rides"],
            best["arrival_s"],
        )
    times = network_with_footpaths.travel_times_from_stop(KAPYLA, arrival=DEADLINE)
    for stop, (departure, _rides, achieved) in by_stop.items():
        if stop == KAPYLA:
            continue
        assert times[stop] == achieved - departure


WINDOW_START = "2022-02-22 09:00:00"


def test_the_arrival_window_unions_the_per_mark_answers(
    network, network_with_footpaths
):
    # The windowed route call returns the deadline profile: the union
    # of each minute mark's latest-departure Pareto set, every member
    # exactly its mark's single-deadline answer.
    windowed = network.route_between_stops(
        KORSO, KAPYLA, arrival=WINDOW_START, arrival_time_window=30
    )
    assert windowed
    marks = [9 * 3600 + 60 * i for i in range(30)]
    expected = []
    for mark in marks:
        single = network.route_between_stops(KORSO, KAPYLA, arrival=clock(mark))
        for journey in single:
            if journey not in expected:
                expected.append(journey)
    expected.sort(key=lambda j: (-j["departure_s"], j["rides"], j["arrival_s"]))
    assert windowed == expected
    departures = [j["departure_s"] for j in windowed]
    assert departures == sorted(departures, reverse=True)
    # The exact per-mark union over coordinates, walks included: the
    # direct walk competes inside every mark's Pareto selection, so the
    # windowed result equals the union of single-deadline answers.
    windowed = network_with_footpaths.route_between_coordinates(
        KAMPPI, HAKANIEMI, arrival=WINDOW_START, arrival_time_window=10
    )
    assert windowed
    expected = []
    for i in range(10):
        single = network_with_footpaths.route_between_coordinates(
            KAMPPI, HAKANIEMI, arrival=clock(9 * 3600 + 60 * i)
        )
        for journey in single:
            if journey not in expected:
                expected.append(journey)
    key = lambda j: (-j["departure_s"], j["rides"], j["arrival_s"])  # noqa: E731
    assert sorted(windowed, key=key) == sorted(expected, key=key)


def test_windowed_percentiles_recompute_from_single_deadlines(
    network, network_with_footpaths
):
    import geopandas as gpd
    import numpy as np

    from cafein import TravelTimeMatrix

    frame = TravelTimeMatrix(
        network,
        origins=[KORSO],
        arrival=WINDOW_START,
        arrival_time_window=30,
        percentiles=[10, 50, 90],
        chunk=(stop_chunk(network, KAPYLA, 400), 400),
        output_time_units="seconds",
    )
    rows = frame[frame["to_id"] == KAPYLA]
    assert len(rows) == 1
    durations = []
    for i in range(30):
        journeys = network.route_between_stops(
            KORSO, KAPYLA, arrival=clock(9 * 3600 + 60 * i)
        )
        durations.append(
            journeys[0]["arrival_s"] - journeys[0]["departure_s"]
            if journeys
            else np.iinfo(np.uint32).max
        )
    durations.sort()
    for percentile in (10, 50, 90):
        position = (percentile / 100) * (len(durations) - 1)
        expected = durations[min(int(position + 0.5), len(durations) - 1)]
        assert rows[f"travel_time_p{percentile:g}"].iloc[0] == expected
    # The point surface recomputes the same way.
    points = gpd.GeoDataFrame(
        {"id": ["kamppi", "hakaniemi"]},
        geometry=gpd.points_from_xy(
            [KAMPPI[1], HAKANIEMI[1]], [KAMPPI[0], HAKANIEMI[0]]
        ),
        crs="EPSG:4326",
    )
    frame = TravelTimeMatrix(
        network_with_footpaths,
        origins=points,
        arrival=WINDOW_START,
        arrival_time_window=10,
        percentiles=[50],
        output_time_units="seconds",
    )
    row = frame[(frame["from_id"] == "kamppi") & (frame["to_id"] == "hakaniemi")]
    assert len(row) == 1
    durations = []
    for i in range(10):
        journeys = network_with_footpaths.route_between_coordinates(
            KAMPPI, HAKANIEMI, arrival=clock(9 * 3600 + 60 * i)
        )
        durations.append(
            journeys[0]["arrival_s"] - journeys[0]["departure_s"]
            if journeys
            else np.iinfo(np.uint32).max
        )
    durations.sort()
    position = 0.5 * (len(durations) - 1)
    expected = durations[min(int(position + 0.5), len(durations) - 1)]
    assert row["travel_time_p50"].iloc[0] == expected


def stop_chunk(network, stop, n):
    stops = [held for held, *_ in network._core.stops]
    block = -(-len(stops) // n)
    return stops.index(stop) // block


def test_the_windowed_catchment_ranks_per_mark_durations(network_with_footpaths):
    pytest.importorskip("h3")
    from cafein import Catchment

    ranked = Catchment(
        network_with_footpaths,
        [KAPYLA],
        arrival=WINDOW_START,
        arrival_time_window=10,
        budgets=[30],
        percentile=90,
    )
    assert len(ranked)
    # The seeds are derivational: each stop's percentile reach equals
    # the nearest rank of its per-mark single-deadline durations.
    reaches = network_with_footpaths._core._arrive_by_percentile_reaches(
        [(KAPYLA, 0)], "2022-02-22", "09:00:00", 10 * 60, 90.0, 7, [], [], []
    )
    stops = [stop for stop, *_ in network_with_footpaths._core.stops]
    by_stop = {
        stops[stop]: achieved - departure
        for stop, departure, _rides, achieved in reaches
    }
    marks = [9 * 3600 + 60 * i for i in range(10)]
    unreachable = 2**32 - 1
    for origin in [KORSO, "1020453"]:
        durations = sorted(
            network_with_footpaths.travel_times_from_stop(
                KAPYLA, arrival=clock(mark)
            ).get(origin, unreachable)
            for mark in marks
        )
        position = 0.9 * (len(durations) - 1)
        expected = durations[min(int(position + 0.5), len(durations) - 1)]
        if expected == unreachable:
            assert origin not in by_stop
        else:
            assert by_stop[origin] == expected


def test_windowed_accessibility_scores_rank_pessimistically(network):
    from cafein import Accessibility

    frames = {
        percentile: Accessibility(
            network,
            [KORSO],
            [KAPYLA, "1020453"],
            arrival=WINDOW_START,
            arrival_time_window=30,
            percentiles=[percentile],
        )
        for percentile in (10, 90)
    }
    for percentile, frame in frames.items():
        assert set(frame["percentile"]) == {percentile}
    scores = {
        percentile: frame["accessibility"].sum() for percentile, frame in frames.items()
    }
    # Pessimistic durations reach no more than optimistic ones.
    assert scores[90] <= scores[10]


def test_confidence_maps_to_the_symmetric_arrival_percentiles(network):
    from cafein import TravelTimeMatrix

    frame = TravelTimeMatrix(
        network,
        origins=[KORSO],
        arrival=WINDOW_START,
        arrival_time_window=30,
        confidence=0.8,
        chunk=(stop_chunk(network, KAPYLA, 400), 400),
    )
    percentile_columns = [c for c in frame.columns if c.startswith("travel_time_p")]
    assert percentile_columns == [
        "travel_time_p10",
        "travel_time_p50",
        "travel_time_p90",
    ]


def test_streamed_arrive_by_matrix_equals_the_constructor(network, tmp_path):
    import pandas as pd

    from cafein import TravelTimeMatrix
    from test_streaming import _read_aligned

    chunk = (stop_chunk(network, KAPYLA, 400), 400)
    frame = TravelTimeMatrix(
        network,
        origins=[KORSO, "1020453"],
        arrival=DEADLINE,
        chunk=chunk,
        output_time_units="seconds",
    )
    result = TravelTimeMatrix.to_parquet(
        network,
        origins=[KORSO, "1020453"],
        arrival=DEADLINE,
        chunk=chunk,
        output=tmp_path / "arrive.parquet",
        batch_size=7,
        output_time_units="seconds",
    )
    # The destination block splits across batches mid-frame. The
    # arrive-by stream is destination-major, so rows align by key.
    assert result.batches >= 3
    read, expected = _read_aligned(tmp_path / "arrive.parquet", frame)
    keys = ["from_id", "to_id"]
    read = read.sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(read, expected)

    windowed = TravelTimeMatrix(
        network,
        origins=[KORSO, "1020453"],
        arrival=WINDOW_START,
        arrival_time_window=10,
        percentiles=[50],
        chunk=chunk,
        output_time_units="seconds",
    )
    TravelTimeMatrix.to_parquet(
        network,
        origins=[KORSO, "1020453"],
        arrival=WINDOW_START,
        arrival_time_window=10,
        percentiles=[50],
        chunk=chunk,
        output=tmp_path / "windowed.parquet",
        batch_size=7,
        output_time_units="seconds",
    )
    read, expected = _read_aligned(tmp_path / "windowed.parquet", windowed)
    read = read.sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(read, expected)


def test_streamed_arrive_by_accessibility_equals_the_constructor(network, tmp_path):
    import pandas as pd

    from cafein import Accessibility

    origins = [KORSO, "1020453"]
    destinations = [KAPYLA, "1070422"]
    frame = Accessibility(
        network,
        origins,
        destinations,
        arrival=WINDOW_START,
        arrival_time_window=30,
        percentiles=[50],
        budgets=[30, 60],
    )
    Accessibility.to_parquet(
        network,
        origins,
        destinations,
        arrival=WINDOW_START,
        arrival_time_window=30,
        percentiles=[50],
        budgets=[30, 60],
        output=tmp_path / "accessibility.parquet",
        batch_size=1,
    )
    import pyarrow.parquet as parquet

    read = parquet.read_table(tmp_path / "accessibility.parquet").to_pandas()
    keys = ["from_id", "opportunity", "budget", "percentile"]
    read = read[list(frame.columns)].astype({"from_id": str})
    read = read.sort_values(keys).reset_index(drop=True)
    expected = pd.DataFrame(frame).astype({"from_id": str})
    expected = expected.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(read, expected)

    # The single-deadline form streams identically.
    single = Accessibility(
        network, origins, destinations, arrival=DEADLINE, budgets=[30, 60]
    )
    Accessibility.to_parquet(
        network,
        origins,
        destinations,
        arrival=DEADLINE,
        budgets=[30, 60],
        output=tmp_path / "single.parquet",
        batch_size=1,
    )
    read = parquet.read_table(tmp_path / "single.parquet").to_pandas()
    keys = ["from_id", "opportunity", "budget"]
    read = read[list(single.columns)].astype({"from_id": str})
    read = read.sort_values(keys).reset_index(drop=True)
    expected = pd.DataFrame(single).astype({"from_id": str})
    expected = expected.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(read, expected)


def test_arrive_by_resume_refuses_a_different_time_query(network, tmp_path):
    from cafein import TravelTimeMatrix

    chunk = (stop_chunk(network, KAPYLA, 400), 400)
    common = dict(
        origins=[KORSO],
        chunk=chunk,
        batch_size=7,
        output=tmp_path / "run",
        output_time_units="seconds",
    )
    TravelTimeMatrix.to_parquet(network, arrival=DEADLINE, **common)
    # The other axis at the same clock refuses; so does the same
    # deadline under a different window.
    with pytest.raises(ValueError, match="fingerprint|manifest"):
        TravelTimeMatrix.to_parquet(network, departure=DEADLINE, resume=True, **common)
    with pytest.raises(ValueError, match="fingerprint|manifest"):
        TravelTimeMatrix.to_parquet(
            network,
            arrival=DEADLINE,
            arrival_time_window=10,
            percentiles=[50],
            resume=True,
            **common,
        )


def test_the_nearest_fast_path_matches_the_fan_out(network, network_with_footpaths):
    import geopandas as gpd
    import pandas as pd

    from cafein import NearestDestinations

    def rank_one(net, origins, destinations):
        # k=2 rides the per-destination fan-out; its rank-1 rows are the
        # same query answered the slow way.
        fast = NearestDestinations(
            net,
            origins,
            destinations,
            arrival=DEADLINE,
            k=1,
            output_time_units="seconds",
        )
        fanned = NearestDestinations(
            net,
            origins,
            destinations,
            arrival=DEADLINE,
            k=2,
            output_time_units="seconds",
        )
        slow = fanned[fanned["rank"] == 1].reset_index(drop=True)
        pd.testing.assert_frame_equal(fast.reset_index(drop=True), slow)

    rank_one(network, [KORSO, "1020453"], [KAPYLA, "1070422", "1020453"])
    origins = gpd.GeoDataFrame(
        {"id": ["kamppi"]},
        geometry=gpd.points_from_xy([KAMPPI[1]], [KAMPPI[0]]),
        crs="EPSG:4326",
    )
    destinations = gpd.GeoDataFrame(
        {"id": ["hakaniemi", "toolo"]},
        geometry=gpd.points_from_xy([HAKANIEMI[1], 24.9220], [HAKANIEMI[0], 60.1810]),
        crs="EPSG:4326",
    )
    rank_one(network_with_footpaths, origins, destinations)


def test_tied_destinations_attribute_deterministically(network_with_footpaths):
    import geopandas as gpd

    from cafein import NearestDestinations

    # Two DISTINCT destinations at the same coordinate: exactly equal
    # winner durations. The lower-position destination wins, and
    # repeated runs agree.
    origins = gpd.GeoDataFrame(
        {"id": ["kamppi"]},
        geometry=gpd.points_from_xy([KAMPPI[1]], [KAMPPI[0]]),
        crs="EPSG:4326",
    )
    destinations = gpd.GeoDataFrame(
        {"id": ["beta", "alpha"]},
        geometry=gpd.points_from_xy(
            [HAKANIEMI[1], HAKANIEMI[1]], [HAKANIEMI[0], HAKANIEMI[0]]
        ),
        crs="EPSG:4326",
    )
    first = NearestDestinations(
        network_with_footpaths, origins, destinations, arrival=DEADLINE, k=1
    )
    second = NearestDestinations(
        network_with_footpaths, origins, destinations, arrival=DEADLINE, k=1
    )
    assert first.equals(second)
    # "beta" is listed first: position, never name, breaks the tie.
    assert list(first["destination_id"]) == ["beta"]


def test_aliased_stop_destinations_tie_to_the_first_listing(network):
    import pandas as pd

    from cafein import NearestDestinations

    # The same stop listed twice as two destinations: an exact tie the
    # fan-out and the fast path must resolve identically, and an
    # origin that IS the destination answers itself at zero.
    fast = NearestDestinations(
        network,
        [KORSO, KAPYLA],
        [KAPYLA, KAPYLA],
        arrival=DEADLINE,
        k=1,
        output_time_units="seconds",
    )
    fanned = NearestDestinations(
        network,
        [KORSO, KAPYLA],
        [KAPYLA, KAPYLA],
        arrival=DEADLINE,
        k=2,
        output_time_units="seconds",
    )
    slow = fanned[fanned["rank"] == 1].reset_index(drop=True)
    pd.testing.assert_frame_equal(fast.reset_index(drop=True), slow)
    assert list(fast[fast["from_id"] == KAPYLA]["cost"]) == [0.0]


def _profile_emissions(network, origin, destination, window):
    journeys = network.route_between_stops(
        origin, destination, arrival=DEADLINE, arrival_time_window=window
    )
    return [journey["emissions"] for journey in network.annotate_emissions(journeys)]


def test_the_forward_cost_frame_is_pinned_with_the_ceiling_dormant(network):
    # The arrival ceiling threads through every cost fold as an
    # optional parameter; this pins the forward (ceiling-less) frame —
    # row order, dtypes, the exact integer columns, and null placement
    # — so a drifted cell cannot hide behind an unchanged shape.
    import hashlib

    import numpy as np

    from cafein import TravelCostMatrix

    frame = TravelCostMatrix(
        network,
        origins=[KORSO, "1020453", KAPYLA],
        departure="2022-02-22 08:30:00",
        optimize="emissions",
        departure_time_window=20,
        output_time_units="seconds",
    )
    assert len(frame) == 273
    # Older pandas names string columns "object", newer "str"; the id
    # columns accept either while the value columns stay exact.
    dtypes = [str(dtype) for dtype in frame.dtypes]
    assert all(dtype in ("object", "str") for dtype in dtypes[:2])
    assert dtypes[2:] == ["uint32", "uint32", "float64", "float64", "float64"]
    assert int(frame.isna().sum().sum()) == 0
    ids = "\n".join(frame["from_id"] + ">" + frame["to_id"]).encode()
    assert (
        hashlib.sha256(ids).hexdigest()
        == "75c3fae9d5fb3e861189094f39de1a259dd37dd167ed90c1da0b2a9e2b7c652d"
    )
    integers = np.stack(
        [frame["travel_time"].to_numpy(), frame["transfers"].to_numpy()]
    ).astype("uint32")
    assert (
        hashlib.sha256(integers.tobytes()).hexdigest()
        == "ad4b35edef670e779706907592a67111548d58904e1c09bb341729992b8976f8"
    )
    assert abs(frame["emissions"].sum() - 35138.225) < 1e-6
    assert abs(frame["transit_distance_m"].sum() - 1405529.0) < 1e-6
    assert frame["walk_distance_m"].sum() == 0.0
    # A restricted query's full frame, cell by cell.
    small = TravelCostMatrix(
        network,
        origins=[KORSO, KAPYLA],
        destinations=[KORSO, KAPYLA, "1020453"],
        departure="2022-02-22 08:30:00",
        optimize="emissions",
        departure_time_window=20,
        output_time_units="seconds",
    )
    expected = [
        (KORSO, KORSO, 0, 0, 0.0, 0.0, 0.0),
        (KORSO, KAPYLA, 1320, 0, 16786.0, 0.0, 419.65),
        (KAPYLA, KAPYLA, 0, 0, 0.0, 0.0, 0.0),
    ]
    assert len(small) == len(expected)
    for row, want in zip(small.itertuples(index=False), expected):
        assert (row.from_id, row.to_id) == want[:2]
        assert (row.travel_time, row.transfers) == want[2:4]
        assert abs(row.transit_distance_m - want[4]) < 1e-9
        assert abs(row.walk_distance_m - want[5]) < 1e-9
        assert abs(row.emissions - want[6]) < 1e-9


def test_cost_cells_price_the_deadline_profiles_minimum(network):
    # Every cell equals the minimum emissions over the pair's deadline
    # profile, recomputed independently from the windowed arrive-by
    # route plus annotation; self cells answer at zero and pairs whose
    # profile is empty are absent — route parity, cell for cell.
    from cafein import TravelCostMatrix

    stops = [KORSO, "1020453", KAPYLA]
    frame = TravelCostMatrix(
        network,
        origins=stops,
        destinations=stops,
        arrival=DEADLINE,
        arrival_time_window=20,
        optimize="emissions",
        output_time_units="seconds",
    )
    cells = {(row.from_id, row.to_id): row for row in frame.itertuples(index=False)}
    for origin in stops:
        for destination in stops:
            if origin == destination:
                row = cells[(origin, destination)]
                assert row.travel_time == 0 and row.emissions == 0.0
                continue
            values = _profile_emissions(network, origin, destination, 20)
            if not values:
                assert (origin, destination) not in cells
                continue
            assert abs(min(values) - cells[(origin, destination)].emissions) < 1e-9


def test_the_money_axis_prices_the_zone_fares(network, helsinki_gtfs):
    # The HSL zone tariff on the arrival axis: each cell is the
    # cheapest fare among the deadline profile's candidates.
    from cafein import TravelCostMatrix, fares

    hsl = fares.zone_fare_structure(str(helsinki_gtfs), rules="zones")
    frame = TravelCostMatrix(
        network,
        origins=[KORSO],
        destinations=[KAPYLA],
        arrival=DEADLINE,
        arrival_time_window=30,
        optimize="money",
        fares=hsl,
        output_time_units="seconds",
    )
    journeys = network.route_between_stops(
        KORSO, KAPYLA, arrival=DEADLINE, arrival_time_window=30
    )
    priced = fares.annotate_fares(journeys, hsl)
    assert len(frame) == 1
    assert abs(min(j["fare"] for j in priced) - frame["fare"].iloc[0]) < 1e-9


def test_point_cost_cells_take_the_walk_and_the_budgeted_transit(
    network_with_footpaths,
):
    import geopandas as gpd
    from shapely.geometry import Point

    from cafein import TravelCostMatrix

    points = gpd.GeoDataFrame(
        {"id": ["kamppi", "hakaniemi"]},
        geometry=[Point(lon, lat) for lat, lon in [KAMPPI, HAKANIEMI]],
        crs="EPSG:4326",
    )
    frame = TravelCostMatrix(
        network_with_footpaths,
        origins=points.iloc[:1],
        destinations=points,
        arrival=DEADLINE,
        arrival_time_window=20,
        optimize="emissions",
        output_time_units="seconds",
    )
    cells = {row.to_id: row for row in frame.itertuples(index=False)}
    # The walking-only alternative's zero grams win the open cell, and
    # the self cell answers at zero — exactly the forward axis's rule.
    walked = cells["hakaniemi"]
    assert walked.emissions == 0.0 and walked.transfers == 0
    assert walked.walk_distance_m > 0.0 and walked.transit_distance_m == 0.0
    assert cells["kamppi"].travel_time == 0
    # A budget below the walk's duration hands the cell to the
    # cheapest within-budget transit candidate of the profile.
    budgeted = TravelCostMatrix(
        network_with_footpaths,
        origins=points.iloc[:1],
        destinations=points.iloc[1:],
        arrival=DEADLINE,
        arrival_time_window=20,
        optimize="emissions",
        max_travel_time=20,
        output_time_units="seconds",
    )
    journeys = network_with_footpaths.route_between_coordinates(
        KAMPPI, HAKANIEMI, arrival=DEADLINE, arrival_time_window=20
    )
    annotated = network_with_footpaths.annotate_emissions(journeys)
    within = [
        journey["emissions"]
        for journey in annotated
        if journey["rides"] > 0
        and journey["arrival_s"] - journey["departure_s"] <= 1200
    ]
    assert len(budgeted) == 1
    assert budgeted["transit_distance_m"].iloc[0] > 0.0
    assert abs(min(within) - budgeted["emissions"].iloc[0]) < 1e-9


def test_cost_products_ride_the_deadline_profile(network):
    # Accessibility counts destinations whose deadline-profile optimum
    # fits the budget, and NearestDestinations reports that optimum —
    # both against the matrix cell they must share.
    import pandas as pd

    from cafein import Accessibility, NearestDestinations, TravelCostMatrix

    cell = TravelCostMatrix(
        network,
        origins=[KORSO],
        destinations=[KAPYLA],
        arrival=DEADLINE,
        arrival_time_window=20,
        optimize="emissions",
        output_time_units="seconds",
    )["emissions"].iloc[0]
    destinations = pd.DataFrame({"id": [KAPYLA], "reachable": [1]})
    scores = Accessibility(
        network,
        origins=[KORSO],
        destinations=destinations,
        arrival=DEADLINE,
        arrival_time_window=20,
        cost="emissions",
        budgets=[cell - 1.0, cell + 1.0],
    )
    below = scores[scores["budget"] == cell - 1.0]["accessibility"].iloc[0]
    above = scores[scores["budget"] == cell + 1.0]["accessibility"].iloc[0]
    assert (below, above) == (0.0, 1.0)
    nearest = NearestDestinations(
        network,
        origins=[KORSO],
        destinations=[KAPYLA],
        arrival=DEADLINE,
        arrival_time_window=20,
        cost="emissions",
        k=1,
    )
    assert abs(nearest["cost"].iloc[0] - cell) < 1e-9


def test_the_arrive_by_cost_catchment_seeds_the_fitting_stops(
    network_with_footpaths,
):
    # The emissions catchment of a place on the arrival axis: a stop
    # whose deadline-profile optimum fits the budget seeds the region,
    # so its location lies inside; a budget below every candidate
    # leaves it out. Central stops, so the walking field has streets,
    # and a tight walking cutoff so only the seed itself can cover it.
    pytest.importorskip("h3")
    from shapely.geometry import Point

    from cafein import Catchment, TravelCostMatrix

    place, seed_stop = "1010125", "1010108"
    cell = TravelCostMatrix(
        network_with_footpaths,
        origins=[seed_stop],
        destinations=[place],
        arrival=DEADLINE,
        arrival_time_window=5,
        optimize="emissions",
        output_time_units="seconds",
    )["emissions"].iloc[0]
    regions = Catchment(
        network_with_footpaths,
        origins=[place],
        arrival=DEADLINE,
        arrival_time_window=5,
        cost="emissions",
        budgets=[cell - 1.0, cell + 1.0],
        max_walking_time=2,
    )
    seed = next(
        Point(lon, lat)
        for stop, lat, lon in network_with_footpaths.stops
        if stop == seed_stop
    )
    tight = regions[regions["budget"] == cell - 1.0].geometry
    loose = regions[regions["budget"] == cell + 1.0].geometry
    assert not any(region.contains(seed) for region in tight)
    assert any(region.contains(seed) for region in loose)


def test_the_streamed_cost_table_slices_the_destination_axis(network, tmp_path):
    import json

    import pandas as pd
    import pyarrow.parquet as pq

    from cafein import travel_cost_table

    query = dict(
        origins=[KORSO, KAPYLA],
        destinations=[KORSO, KAPYLA, "1020453"],
        arrival=DEADLINE,
        arrival_time_window=20,
        optimize="emissions",
        output_time_units="seconds",
    )
    table = travel_cost_table(network, **query)
    result = travel_cost_table(
        network, output=str(tmp_path / "shards"), batch_size=1, **query
    )
    manifest = json.loads((tmp_path / "shards" / "manifest.json").read_text())
    assert manifest["batch_axis"] == "to"
    assert result.batches == 3
    shards = sorted((tmp_path / "shards").glob("part-*.parquet"))
    streamed = pd.concat([pq.read_table(shard).to_pandas() for shard in shards])
    keys = ["from_id", "to_id"]
    left = table.to_pandas()
    right = streamed
    left[keys] = left[keys].astype(str)
    right[keys] = right[keys].astype(str)
    left = left.sort_values(keys).reset_index(drop=True)
    right = right.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)
    # A resumed run must refuse a different arrival query.
    with pytest.raises(ValueError, match="fingerprint"):
        travel_cost_table(
            network,
            output=str(tmp_path / "shards"),
            batch_size=1,
            resume=True,
            **{**query, "arrival": WINDOW_START},
        )


def test_a_walk_longer_than_every_deadline_is_no_candidate(network_with_footpaths):
    # Near midnight the direct walk cannot be placed to arrive by any
    # mark — the overlay must not invent a walking journey departing
    # the previous day.
    import geopandas as gpd
    from shapely.geometry import Point

    from cafein import TravelCostMatrix

    points = gpd.GeoDataFrame(
        {"id": ["kamppi", "hakaniemi"]},
        geometry=[Point(lon, lat) for lat, lon in [KAMPPI, HAKANIEMI]],
        crs="EPSG:4326",
    )
    frame = TravelCostMatrix(
        network_with_footpaths,
        origins=points.iloc[:1],
        destinations=points.iloc[1:],
        arrival="2022-02-22 00:20:00",
        arrival_time_window=10,
        optimize="emissions",
        output_time_units="seconds",
    )
    deadline_s = 20 * 60 + 9 * 60
    walks = frame[(frame["transit_distance_m"] == 0.0) & (frame["walk_distance_m"] > 0)]
    assert walks.empty
    assert (frame["travel_time"] <= deadline_s).all()
