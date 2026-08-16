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


def test_exactly_one_time_axis_is_required(network):
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


def test_a_window_does_not_combine_with_arrival(network):
    with pytest.raises(ValueError, match="departure_time_window"):
        network.route_between_stops(
            KORSO, KAPYLA, arrival=DEADLINE, departure_time_window=30
        )


def test_a_street_policy_does_not_combine_with_arrival(network_with_footpaths):
    from cafein import StreetLegPolicy

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


def test_journeys_arrive_by_the_deadline_latest_departure_first(network):
    reverse = network.route_between_stops(KORSO, KAPYLA, arrival=DEADLINE)
    assert all(j["arrival_s"] <= DEADLINE_S for j in reverse)
    departures = [j["departure_s"] for j in reverse]
    assert departures == sorted(departures, reverse=True)
    # Travel time is each journey's own duration: the latest departure
    # rides the 16-minute K train, not the span to the deadline.
    best = reverse[0]
    assert best["arrival_s"] - best["departure_s"] < 30 * 60


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


def test_a_traveler_profile_excludes_on_the_reverse_axis(network):
    from cafein import TravelerProfile

    reverse = network.route_between_stops(KORSO, KAPYLA, arrival=DEADLINE)
    ridden = reverse[0]["legs"][1]["route_id"]
    profile = TravelerProfile(exclude_routes=[ridden])
    via_profile = network.route_between_stops(
        KORSO, KAPYLA, arrival=DEADLINE, traveler=profile
    )
    explicit = network.route_between_stops(
        KORSO, KAPYLA, arrival=DEADLINE, exclude_routes=[ridden]
    )
    assert via_profile
    assert via_profile == explicit
    assert all(
        leg.get("route_id") != ridden
        for journey in via_profile
        for leg in journey["legs"]
    )


def test_exclusions_apply_on_the_reverse_axis(network):
    reverse = network.route_between_stops(KORSO, KAPYLA, arrival=DEADLINE)
    ridden = reverse[0]["legs"][1]["route_id"]
    without = network.route_between_stops(
        KORSO, KAPYLA, arrival=DEADLINE, exclude_routes=[ridden]
    )
    assert all(
        leg.get("route_id") != ridden for journey in without for leg in journey["legs"]
    )
    # Without the K train the latest alternative leaves before six;
    # the bracket spans the whole early morning.
    profile = network.route_between_stops(
        KORSO,
        KAPYLA,
        "2022-02-22 04:00:00",
        departure_time_window=330,
        exclude_routes=[ridden],
    )
    assert without
    assert tuples(without) == inverted_profile(profile, DEADLINE_S)


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


def test_the_wide_matrix_chunks_the_destination_axis(network):
    matrix = network.travel_time_matrix([KORSO], arrival=DEADLINE, chunk=(0, 400))
    stops = network.stop_count
    expected = stops // 400 + (1 if stops % 400 else 0)
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


def test_matrix_rejections_on_the_arrival_axis(network):
    from cafein import TravelTimeMatrix

    with pytest.raises(ValueError, match="exactly one of departure= or arrival="):
        TravelTimeMatrix(network, origins=[KORSO], departure=DEADLINE, arrival=DEADLINE)
    with pytest.raises(ValueError, match="departure_time_window"):
        TravelTimeMatrix(
            network, origins=[KORSO], arrival=DEADLINE, departure_time_window=30
        )
    with pytest.raises(ValueError, match="router='tbtr'"):
        TravelTimeMatrix(network, origins=[KORSO], arrival=DEADLINE, router="tbtr")
    with pytest.raises(NotImplementedError, match="do not stream"):
        TravelTimeMatrix.to_parquet(
            network, origins=[KORSO], arrival=DEADLINE, output="/tmp/never.parquet"
        )


def test_itinerary_rejections_on_the_arrival_axis(network):
    from cafein import DetailedItineraries

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


def test_itineraries_require_exactly_one_axis(network):
    from cafein import DetailedItineraries

    with pytest.raises(ValueError, match="exactly one of departure= or arrival="):
        DetailedItineraries(network, origins=[KORSO], destinations=[KAPYLA])


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


def test_product_rejections_on_the_arrival_axis(network, network_with_footpaths):
    from cafein import Accessibility, Catchment

    with pytest.raises(ValueError, match="does not combine with arrival="):
        Accessibility(
            network,
            [KORSO],
            [KAPYLA],
            arrival=DEADLINE,
            cost="emissions",
            departure_time_window=60,
        )
    with pytest.raises(ValueError, match="exactly one of departure= or arrival="):
        Accessibility(network, [KORSO], [KAPYLA], DEADLINE, arrival=DEADLINE)
    with pytest.raises(ValueError, match="router must be"):
        Accessibility(network, [KORSO], [KAPYLA], arrival=DEADLINE, router="bogus")
    with pytest.raises(ValueError, match="percentiles"):
        Accessibility(network, [KORSO], [KAPYLA], arrival=DEADLINE, percentiles=[50])
    pytest.importorskip("h3")
    with pytest.raises(ValueError, match="departure_time_window"):
        Catchment(
            network_with_footpaths,
            [KAPYLA],
            arrival=DEADLINE,
            departure_time_window=60,
        )


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
