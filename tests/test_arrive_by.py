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
