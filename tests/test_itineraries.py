"""DetailedItineraries over the Helsinki network shared with r5py."""

import geopandas as gpd
import pytest

from cafein import DetailedItineraries


def point_frame(network, named_stops):
    """Points at known stops' coordinates, under fresh ids."""
    coordinates = {stop: (lat, lon) for stop, lat, lon in network.stops}
    ids, stops = zip(*named_stops)
    lats, lons = zip(*(coordinates[stop] for stop in stops))
    return gpd.GeoDataFrame(
        {"id": list(ids)},
        geometry=gpd.points_from_xy(lons, lats),
        crs="EPSG:4326",
    )


def test_stop_itinerary_pins_the_k_train(network):
    itineraries = DetailedItineraries(
        network, ["4810551"], ["1250551"], "2022-02-22", "08:30:00"
    )
    assert isinstance(itineraries, gpd.GeoDataFrame)
    assert itineraries.crs == "EPSG:4326"

    option0 = itineraries[itineraries["option"] == 0]
    assert list(option0["leg_type"]) == ["access", "transit", "egress"]
    assert list(option0["segment"]) == [0, 1, 2]
    assert (option0["from_id"] == "4810551").all()
    assert (option0["to_id"] == "1250551").all()

    transit = option0[option0["leg_type"] == "transit"].iloc[0]
    assert transit["trip_id"] == "3001K_20220222_S1_2_0831"
    assert transit["route_short_name"] == "K"
    assert transit["from_stop"] == "4810551"
    assert transit["to_stop"] == "1250551"
    assert transit["departure"] == 8 * 3600 + 36 * 60
    assert transit["arrival"] == 8 * 3600 + 58 * 60
    assert transit["travel_time"] == 22 * 60
    assert transit["distance"] == pytest.approx(16_786, abs=1)
    assert transit["distance_provenance"] == "shape_dist"
    # 16.786 km at the shipped 25 g/pkm urban-rail factor.
    assert transit["emissions"] == pytest.approx(419.65, abs=0.1)
    assert transit["geometry"].geom_type == "LineString"

    # Walk legs carry zero emissions and no ridden-trip fields.
    walks = option0[option0["leg_type"] != "transit"]
    assert (walks["emissions"] == 0.0).all()
    assert walks["trip_id"].isna().all()


def test_options_are_a_pareto_set(network):
    itineraries = DetailedItineraries(
        network, ["4810551"], ["1250551"], "2022-02-22", "08:30:00"
    )
    # Each option is one journey; later options ride more and arrive
    # earlier, matching the routing Pareto contract.
    arrivals, rides = {}, {}
    for option, group in itineraries.groupby("option"):
        arrivals[option] = group["arrival"].max()
        rides[option] = (group["leg_type"] == "transit").sum()
    for option in sorted(arrivals)[1:]:
        assert rides[option] > rides[option - 1]
        assert arrivals[option] < arrivals[option - 1]


def test_geometries_can_be_switched_off(network):
    itineraries = DetailedItineraries(
        network,
        ["4810551"],
        ["1250551"],
        "2022-02-22",
        "08:30:00",
        geometries=False,
    )
    assert itineraries["geometry"].isna().all()
    # The leg records themselves are unaffected.
    transit = itineraries[itineraries["leg_type"] == "transit"].iloc[0]
    assert transit["distance"] == pytest.approx(16_786, abs=1)


def test_door_to_door_itinerary_walks_the_streets(network_with_footpaths):
    coordinates = {stop: (lat, lon) for stop, lat, lon in network_with_footpaths.stops}
    origins = point_frame(network_with_footpaths, [("A", "1100602")])
    destinations = point_frame(network_with_footpaths, [("B", "1040280")])
    itineraries = DetailedItineraries(
        network_with_footpaths, origins, destinations, "2022-02-22", "08:30:00"
    )
    # Option 0 is the walking-only alternative: one walk leg, no stops.
    option0 = itineraries[itineraries["option"] == 0]
    assert list(option0["leg_type"]) == ["walk"]
    walk = option0.iloc[0]
    assert walk["geometry"].geom_type == "LineString"
    assert walk["from_stop"] is None or walk["from_stop"] != walk["from_stop"]
    assert walk["trip_id"] is None or walk["trip_id"] != walk["trip_id"]
    assert walk["emissions"] == 0.0

    option1 = itineraries[itineraries["option"] == 1]
    assert list(option1["leg_type"]) == ["access", "transit", "egress"]

    access = option1[option1["leg_type"] == "access"].iloc[0]
    egress = option1[option1["leg_type"] == "egress"].iloc[0]
    assert access["geometry"].geom_type == "LineString"
    origin_lat, origin_lon = coordinates["1100602"]
    assert access["geometry"].coords[0] == pytest.approx(
        (origin_lon, origin_lat), abs=1e-6
    )
    destination_lat, destination_lon = coordinates["1040280"]
    assert egress["geometry"].coords[-1] == pytest.approx(
        (destination_lon, destination_lat), abs=1e-6
    )
    # The walked ends have no boarding or alighting stop.
    assert access["from_id"] == "A" and access["to_id"] == "B"
    assert access["from_stop"] is None or access["from_stop"] != access["from_stop"]
    assert egress["to_stop"] is None or egress["to_stop"] != egress["to_stop"]


def test_unreachable_pair_yields_an_empty_frame(network):
    itineraries = DetailedItineraries(
        network, ["4810551"], ["4810551"], "2022-02-22", "08:30:00"
    )
    assert isinstance(itineraries, gpd.GeoDataFrame)
    assert len(itineraries) == 0
    assert itineraries.crs == "EPSG:4326"
    assert "geometry" in itineraries.columns


def test_slices_do_not_re_route(network):
    itineraries = DetailedItineraries(
        network, ["4810551"], ["1250551"], "2022-02-22", "08:30:00"
    )
    # A slice is a working GeoDataFrame detached from the network.
    head = itineraries.iloc[:1]
    assert isinstance(head, gpd.GeoDataFrame)
    assert head.crs == "EPSG:4326"
    assert len(head) == 1
    # Concatenation and grouping do not re-trigger routing.
    import pandas as pd

    doubled = pd.concat([itineraries, itineraries], ignore_index=True)
    assert len(doubled) == 2 * len(itineraries)


def test_inputs_must_match_and_be_valid(network):
    points = point_frame(network, [("A", "4810551")])
    with pytest.raises(ValueError, match="both be stop ids or both be"):
        DetailedItineraries(network, ["4810551"], points, "2022-02-22", "08:30:00")
    with pytest.raises(ValueError, match="walking options apply to point"):
        DetailedItineraries(
            network,
            ["4810551"],
            ["1250551"],
            "2022-02-22",
            "08:30:00",
            max_snap_distance=50,
        )
    with pytest.raises(ValueError, match="required for detailed itineraries"):
        DetailedItineraries(network, None, ["1250551"], "2022-02-22", "08:30:00")
