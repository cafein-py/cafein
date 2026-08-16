"""TravelerProfile compilation and the traveler= surfaces."""

import geopandas as gpd
import pandas as pd
import pytest

from cafein import DetailedItineraries, TravelerProfile, TravelTimeMatrix
from cafein.travelers import folded_constraints

# Helsinki wheelchair truth (pinned in test_transport_network too):
# 407 stops accessible, 1965 not, 5933 unsaid; 3422 trips not accessible.
INACCESSIBLE_STOPS = 1965
UNKNOWN_STOPS = 5933
INACCESSIBLE_TRIPS = 3422


def test_the_profile_validates_its_inputs():
    with pytest.raises(ValueError, match="unknown"):
        TravelerProfile(unknown="optimistic")
    with pytest.raises(ValueError, match="walking_speed_kmph"):
        TravelerProfile(walking_speed_kmph=0)
    with pytest.raises(ValueError, match="max_walking_time"):
        TravelerProfile(max_walking_time=-5)
    with pytest.raises(TypeError, match="exclude_stops"):
        TravelerProfile(exclude_stops="1010107")


def test_the_policy_matrix_covers_every_state():
    # A fake core with all three states on stops and trips, so the
    # optimistic/strict split is pinned even where Helsinki has no
    # unknown trips.
    class Core:
        stop_wheelchair_boarding = [
            ("s_yes", True),
            ("s_no", False),
            ("s_unknown", None),
        ]
        trip_wheelchair_accessible = [
            ("t_yes", True),
            ("t_no", False),
            ("t_unknown", None),
        ]

    class Network:
        _core = Core()

    _, trips, stops = TravelerProfile(wheelchair=True)._resolve(Network())
    assert (stops, trips) == (["s_no"], ["t_no"])
    _, trips, stops = TravelerProfile(wheelchair=True, unknown="excluded")._resolve(
        Network()
    )
    assert (sorted(stops), sorted(trips)) == (
        ["s_no", "s_unknown"],
        ["t_no", "t_unknown"],
    )


def test_wheelchair_compilation_policies(network):
    routes, trips, stops = TravelerProfile(wheelchair=True)._resolve(network)
    assert (len(routes), len(trips), len(stops)) == (0, INACCESSIBLE_TRIPS, 1965)
    routes, trips, stops = TravelerProfile(
        wheelchair=True, unknown="excluded"
    )._resolve(network)
    assert len(stops) == INACCESSIBLE_STOPS + UNKNOWN_STOPS
    assert len(trips) == INACCESSIBLE_TRIPS
    # Without wheelchair the feed's fields are not consulted: the
    # profile carries only its manual exclusions.
    routes, trips, stops = TravelerProfile(
        exclude_stops=["1010107"], exclude_routes=["2550"]
    )._resolve(network)
    assert (routes, trips, stops) == (["2550"], [], ["1010107"])
    # Manual exclusions merge with the compiled wheelchair sets.
    _, trips, stops = TravelerProfile(
        wheelchair=True, exclude_stops=["1010419"]
    )._resolve(network)
    assert "1010419" in stops and len(stops) == INACCESSIBLE_STOPS + 1


def test_the_profile_routes_like_hand_built_exclusions(network):
    profile = TravelerProfile(wheelchair=True)
    flags = dict(network._core.stop_wheelchair_boarding)
    _, trips, stops = profile._resolve(network)
    manual = network.route_between_stops(
        "4810551",
        "1250551",
        "2022-02-22 08:30:00",
        exclude_stops=stops,
        exclude_trips=trips,
    )
    profiled = network.route_between_stops(
        "4810551", "1250551", "2022-02-22 08:30:00", traveler=profile
    )
    assert profiled == manual
    # A journey that boards inaccessible supply genuinely reroutes: the
    # plain Pareto set touches inaccessible stops, the wheelchair
    # traveler still arrives — later, on accessible supply only.
    inaccessible = {stop for stop, flag in flags.items() if flag is False}
    plain = network.route_between_stops("1020453", "1070422", "2022-02-22 08:30:00")
    rerouted = network.route_between_stops(
        "1020453", "1070422", "2022-02-22 08:30:00", traveler=profile
    )

    def touched(journeys):
        return {
            leg.get(key)
            for journey in journeys
            for leg in journey["legs"]
            for key in ("board_stop", "alight_stop")
        } - {None}

    assert touched(plain) & inaccessible
    assert rerouted and not (touched(rerouted) & inaccessible)
    assert min(j["arrival_s"] for j in rerouted) > min(j["arrival_s"] for j in plain)
    # An inaccessible destination refuses the wheelchair traveler.
    assert (
        network.route_between_stops(
            "4810551", "1010107", "2022-02-22 08:30:00", traveler=profile
        )
        == []
    )


def test_profile_and_call_exclusions_union(network):
    plain = network.route_between_stops("4810551", "1250551", "2022-02-22 08:30:00")
    ridden = next(
        leg["trip_id"] for leg in plain[0]["legs"] if leg["type"] == "transit"
    )
    profile = TravelerProfile(exclude_stops=["1010107"])
    both = network.route_between_stops(
        "4810551",
        "1250551",
        "2022-02-22 08:30:00",
        exclude_trips=[ridden],
        traveler=profile,
    )
    manual = network.route_between_stops(
        "4810551",
        "1250551",
        "2022-02-22 08:30:00",
        exclude_trips=[ridden],
        exclude_stops=["1010107"],
    )
    assert both == manual
    assert all(leg.get("trip_id") != ridden for j in both for leg in j["legs"])


def test_walking_knobs_fold_and_conflict(network):
    profile = TravelerProfile(walking_speed_kmph=3.0, max_walking_time=10)
    folded = folded_constraints(profile, network, (), (), (), None, None)
    assert folded[3] == 3.0 and folded[4] == 10
    with pytest.raises(ValueError, match="walking_speed_kmph"):
        folded_constraints(profile, network, (), (), (), 5.0, None)
    with pytest.raises(ValueError, match="max_walking_time"):
        folded_constraints(profile, network, (), (), (), None, 15)
    with pytest.raises(ValueError, match="walking_speed_kmph"):
        network.route_between_stops(
            "4810551",
            "1250551",
            "2022-02-22 08:30:00",
            walking_speed_kmph=5.0,
            traveler=profile,
        )
    with pytest.raises(TypeError, match="TravelerProfile"):
        network.route_between_stops(
            "4810551", "1250551", "2022-02-22 08:30:00", traveler="wheelchair"
        )


def test_the_computers_honor_the_profile(network):
    profile = TravelerProfile(wheelchair=True)
    _, trips, stops = profile._resolve(network)
    profiled = TravelTimeMatrix(
        network, ["4810551"], departure="2022-02-22 08:30:00", traveler=profile
    )
    manual = TravelTimeMatrix(
        network,
        ["4810551"],
        departure="2022-02-22 08:30:00",
        exclude_stops=stops,
        exclude_trips=trips,
    )
    pd.testing.assert_frame_equal(pd.DataFrame(profiled), pd.DataFrame(manual))
    itineraries = DetailedItineraries(
        network, ["4810551"], ["1250551"], "2022-02-22 08:30:00", traveler=profile
    )
    hand_built = DetailedItineraries(
        network,
        ["4810551"],
        ["1250551"],
        "2022-02-22 08:30:00",
        exclude_stops=stops,
        exclude_trips=trips,
    )
    pd.testing.assert_frame_equal(pd.DataFrame(itineraries), pd.DataFrame(hand_built))
    from cafein import TravelCostMatrix

    cost_profiled = TravelCostMatrix(
        network,
        ["4810551"],
        ["1250551"],
        "2022-02-22 08:30:00",
        traveler=profile,
    )
    cost_manual = TravelCostMatrix(
        network,
        ["4810551"],
        ["1250551"],
        "2022-02-22 08:30:00",
        exclude_stops=stops,
        exclude_trips=trips,
    )
    pd.testing.assert_frame_equal(
        pd.DataFrame(cost_profiled), pd.DataFrame(cost_manual)
    )
    from cafein import frontier_table

    table_profiled = frontier_table(
        network,
        ["4810551"],
        ["1250551"],
        "2022-02-22 08:30:00",
        10,
        traveler=profile,
    )
    table_manual = frontier_table(
        network,
        ["4810551"],
        ["1250551"],
        "2022-02-22 08:30:00",
        10,
        exclude_stops=stops,
        exclude_trips=trips,
    )
    pd.testing.assert_frame_equal(table_profiled, table_manual)


def test_accessibility_honors_the_profile(network):
    from cafein import Accessibility

    profile = TravelerProfile(wheelchair=True)
    _, trips, stops = profile._resolve(network)
    destinations = pd.DataFrame({"id": ["1250551", "1010107"], "jobs": [10.0, 5.0]})
    profiled = Accessibility(
        network,
        ["4810551"],
        destinations,
        "2022-02-22 08:30:00",
        opportunities="jobs",
        budgets=(45.0,),
        traveler=profile,
    )
    manual = Accessibility(
        network,
        ["4810551"],
        destinations,
        "2022-02-22 08:30:00",
        opportunities="jobs",
        budgets=(45.0,),
        exclude_stops=stops,
        exclude_trips=trips,
    )
    pd.testing.assert_frame_equal(pd.DataFrame(profiled), pd.DataFrame(manual))


def test_street_surfaces_reject_the_traveler(helsinki_streets):
    profile = TravelerProfile(wheelchair=True)
    places = gpd.GeoDataFrame(
        {"id": ["a", "b"]},
        geometry=gpd.points_from_xy([24.9320, 24.9615], [60.1690, 60.2043]),
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="traveler.*no meaning for a street matrix"):
        TravelTimeMatrix(
            helsinki_streets, places, transport_mode="walk", traveler=profile
        )
    with pytest.raises(ValueError, match="traveler.*no meaning for a street matrix"):
        DetailedItineraries(
            helsinki_streets, places, places, transport_mode="walk", traveler=profile
        )
    from cafein import Accessibility

    with pytest.raises(ValueError, match="traveler applies to transit"):
        Accessibility(
            helsinki_streets,
            places,
            places,
            opportunities={"a": 1.0, "b": 1.0},
            transport_mode="walk",
            traveler=profile,
        )


def test_the_wheelchair_traveler_bridges_to_the_street_policy(multimodal_network):
    from cafein import StreetLegPolicy
    from cafein.streets import MAX_ACCESS_EGRESS_TIME

    profile = TravelerProfile(wheelchair=True)
    origin, destination = (60.1580, 24.9350), (60.1870, 24.9610)
    bridged = multimodal_network.route_between_coordinates(
        origin, destination, "2022-02-22 08:30:00", traveler=profile
    )
    assert bridged
    _, trips, stops = profile._resolve(multimodal_network)
    explicit = multimodal_network.route_between_coordinates(
        origin,
        destination,
        "2022-02-22 08:30:00",
        exclude_stops=stops,
        exclude_trips=trips,
        street_policy=StreetLegPolicy(
            access={"wheelchair": MAX_ACCESS_EGRESS_TIME},
            egress={"wheelchair": MAX_ACCESS_EGRESS_TIME},
        ),
    )
    assert bridged == explicit
    # Access and egress ride the wheelchair mode; transfer endpoints
    # respect the traveler's exclusions.
    excluded = set(stops)
    transfers = []
    for journey in bridged:
        for leg in journey["legs"]:
            if leg["type"] in ("access", "egress"):
                assert leg["mode"] == "wheelchair"
            if leg["type"] == "transfer":
                transfers.append(leg)
    # The route genuinely transfers, and never through an excluded stop.
    assert transfers
    for leg in transfers:
        assert leg["from_stop"] not in excluded
        assert leg["to_stop"] not in excluded
    with pytest.raises(ValueError, match="conflict"):
        multimodal_network.route_between_coordinates(
            origin,
            destination,
            "2022-02-22 08:30:00",
            traveler=profile,
            street_policy=StreetLegPolicy(access={"walk": 900}, egress={"walk": 900}),
        )
    with pytest.raises(ValueError, match="fixed speed"):
        multimodal_network.route_between_coordinates(
            origin,
            destination,
            "2022-02-22 08:30:00",
            traveler=profile,
            walking_speed_kmph=5.0,
        )


def test_the_bridge_serves_the_point_computers(multimodal_network):
    from cafein import DetailedItineraries, StreetLegPolicy, TravelTimeMatrix
    from cafein.streets import MAX_ACCESS_EGRESS_TIME

    profile = TravelerProfile(wheelchair=True)
    _, trips, stops = profile._resolve(multimodal_network)
    points = gpd.GeoDataFrame(
        {"id": ["a", "b"]},
        geometry=gpd.points_from_xy([24.9350, 24.9610], [60.1580, 60.1870]),
        crs="EPSG:4326",
    )
    policy = StreetLegPolicy(
        access={"wheelchair": MAX_ACCESS_EGRESS_TIME},
        egress={"wheelchair": MAX_ACCESS_EGRESS_TIME},
    )
    bridged = TravelTimeMatrix(
        multimodal_network, points, departure="2022-02-22 08:30:00", traveler=profile
    )
    manual = TravelTimeMatrix(
        multimodal_network,
        points,
        departure="2022-02-22 08:30:00",
        exclude_stops=stops,
        exclude_trips=trips,
        street_policy=policy,
    )
    pd.testing.assert_frame_equal(pd.DataFrame(bridged), pd.DataFrame(manual))
    itineraries = DetailedItineraries(
        multimodal_network,
        points.iloc[:1],
        points.iloc[1:],
        "2022-02-22 08:30:00",
        traveler=profile,
        geometries=False,
    )
    hand_built = DetailedItineraries(
        multimodal_network,
        points.iloc[:1],
        points.iloc[1:],
        "2022-02-22 08:30:00",
        exclude_stops=stops,
        exclude_trips=trips,
        street_policy=policy,
        geometries=False,
    )
    pd.testing.assert_frame_equal(pd.DataFrame(itineraries), pd.DataFrame(hand_built))
    ends = itineraries[itineraries["leg_type"].isin(["access", "egress"])]
    assert not ends.empty and (ends["mode"] == "wheelchair").all()


def test_the_bridge_serves_the_one_to_all_surface(multimodal_network):
    from cafein import StreetLegPolicy
    from cafein.streets import MAX_ACCESS_EGRESS_TIME

    profile = TravelerProfile(wheelchair=True)
    bridged = multimodal_network.travel_times_from_coordinate(
        (60.1580, 24.9350), "2022-02-22 08:30:00", traveler=profile
    )
    assert bridged
    _, trips, stops = profile._resolve(multimodal_network)
    explicit = multimodal_network.travel_times_from_coordinate(
        (60.1580, 24.9350),
        "2022-02-22 08:30:00",
        exclude_stops=stops,
        exclude_trips=trips,
        street_policy=StreetLegPolicy(
            access={"wheelchair": MAX_ACCESS_EGRESS_TIME},
            egress={"wheelchair": MAX_ACCESS_EGRESS_TIME},
        ),
    )
    assert bridged == explicit
    # An explicitly inaccessible stop never appears among the arrivals.
    assert not set(stops) & set(bridged)


def test_the_bridge_needs_the_wheelchair_build(network_with_footpaths):
    with pytest.raises(
        ValueError, match=r"wheelchair.*street_modes=\('walk', 'wheelchair'\)"
    ):
        network_with_footpaths.route_between_coordinates(
            (60.1690, 24.9320),
            (60.2043, 24.9615),
            "2022-02-22 08:30:00",
            traveler=TravelerProfile(wheelchair=True),
        )


def test_policy_less_point_surfaces_refuse_the_wheelchair_traveler(
    multimodal_network,
    network_with_footpaths,
):
    from cafein import Accessibility, journey_frontier

    profile = TravelerProfile(wheelchair=True)
    with pytest.raises(ValueError, match="cannot honour"):
        journey_frontier(
            multimodal_network,
            (60.1580, 24.9350),
            (60.1870, 24.9610),
            "2022-02-22 08:30:00",
            10,
            traveler=profile,
        )
    points = gpd.GeoDataFrame(
        {"id": ["a"], "jobs": [1.0]},
        geometry=gpd.points_from_xy([24.935], [60.158]),
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="cannot honour"):
        Accessibility(
            multimodal_network,
            points,
            points,
            "2022-02-22 08:30:00",
            opportunities="jobs",
            traveler=profile,
        )
    # The remaining policy-less point surfaces refuse it too.
    pyarrow = pytest.importorskip("pyarrow")  # noqa: F841
    from cafein import frontier_table, travel_cost_table

    with pytest.raises(ValueError, match="cannot honour"):
        travel_cost_table(
            multimodal_network,
            points,
            points,
            "2022-02-22 08:30:00",
            output="unused.parquet",
            traveler=profile,
        )
    with pytest.raises(ValueError, match="cannot honour"):
        frontier_table(
            multimodal_network,
            points,
            points,
            "2022-02-22 08:30:00",
            10,
            traveler=profile,
        )
    # A transit Catchment serves the wheelchair traveler through the
    # profile-aware directed spread: its cells are a subset of the
    # walking catchment's (stairs and capped gradients pruned), and the
    # fixed profile speed rejects the walking knob.
    from cafein import Catchment

    pytest.importorskip("h3")
    wheeled = Catchment(
        multimodal_network,
        ["1020453"],
        "2022-02-22 08:30:00",
        budgets=(15.0,),
        traveler=profile,
    )
    assert not wheeled.empty
    walked = Catchment(
        multimodal_network,
        ["1020453"],
        "2022-02-22 08:30:00",
        budgets=(15.0,),
    )
    assert wheeled.iloc[0].geometry.within(walked.iloc[0].geometry.buffer(1e-6))
    with pytest.raises(ValueError, match="fixed speed"):
        Catchment(
            multimodal_network,
            ["1020453"],
            "2022-02-22 08:30:00",
            budgets=(15.0,),
            traveler=profile,
            walking_speed_kmph=5.0,
        )
    with pytest.raises(ValueError, match="street_modes"):
        Catchment(
            network_with_footpaths,
            ["1020453"],
            "2022-02-22 08:30:00",
            budgets=(15.0,),
            traveler=profile,
        )
    # Point origins ride the synthesized policy for the seeds too; the
    # cost surfaces carry no policy support, so those axes refuse.
    origin_points = gpd.GeoDataFrame(
        {"id": ["o"]},
        geometry=gpd.points_from_xy([24.9350], [60.1580]),
        crs="EPSG:4326",
    )
    geo_wheeled = Catchment(
        multimodal_network,
        origin_points,
        "2022-02-22 08:30:00",
        budgets=(15.0,),
        traveler=profile,
    )
    assert not geo_wheeled.empty
    geo_walked = Catchment(
        multimodal_network,
        origin_points,
        "2022-02-22 08:30:00",
        budgets=(15.0,),
    )
    assert geo_wheeled.iloc[0].geometry.within(geo_walked.iloc[0].geometry.buffer(1e-6))
    with pytest.raises(ValueError, match="snap_distance"):
        Catchment(
            multimodal_network,
            origin_points,
            "2022-02-22 08:30:00",
            budgets=(15.0,),
            snap_distance=500.0,
            traveler=profile,
        )
    with pytest.raises(ValueError, match="time axis"):
        Catchment(
            multimodal_network,
            origin_points,
            "2022-02-22 08:30:00",
            cost="emissions",
            budgets=(500.0,),
            departure_time_window=10,
            traveler=profile,
        )
    with pytest.raises(ValueError, match="outside the network's stop range"):
        multimodal_network._core._catchment_directed_field(
            (60.1580, 24.9350), [(10_000_000, 0.0)], "wheelchair", 900.0, 500.0
        )
    # Stop queries stay timetable-only and pass — the streaming table
    # included, where the profile must equal its explicit exclusions.
    table_profiled = travel_cost_table(
        multimodal_network,
        ["1020453"],
        ["1070422"],
        "2022-02-22 08:30:00",
        traveler=profile,
    )
    _, trips, stops = profile._resolve(multimodal_network)
    table_manual = travel_cost_table(
        multimodal_network,
        ["1020453"],
        ["1070422"],
        "2022-02-22 08:30:00",
        exclude_stops=stops,
        exclude_trips=trips,
    )
    assert table_profiled.equals(table_manual)
    frontier = journey_frontier(
        multimodal_network,
        "1020453",
        "1070422",
        "2022-02-22 08:30:00",
        10,
        traveler=profile,
    )
    assert frontier is not None
