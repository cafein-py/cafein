"""Car routing: the free-flow default, the intersection-delay model, matrices."""

import math
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cafein._cafein")

from cafein import (  # noqa: E402
    DetailedItineraries,
    StreetNetwork,
    TravelCostMatrix,
    TravelTimeMatrix,
)
from cafein import _delays, _osm  # noqa: E402

KAMPPI = (60.1690, 24.9320)
ARABIA = (60.2110, 24.9750)


def scattered_points(count, seed, centre=(60.170, 24.940)):
    """Deterministic building-like locations around central Helsinki."""
    latitude, longitude = centre
    identifiers, latitudes, longitudes = [], [], []
    for index in range(count):
        angle = (seed + index) * 2.399963  # the golden angle
        radius = 100 + (seed * 37 + index * 211) % 900
        identifiers.append(f"point-{seed}-{index}")
        latitudes.append(latitude + radius * math.cos(angle) / 111_320)
        longitudes.append(
            longitude
            + radius * math.sin(angle) / (111_320 * math.cos(math.radians(latitude)))
        )
    return gpd.GeoDataFrame(
        {"id": identifiers},
        geometry=gpd.points_from_xy(longitudes, latitudes),
        crs="EPSG:4326",
    )


@pytest.fixture(scope="module")
def car_network(kantakaupunki_pbf):
    return StreetNetwork.from_osm(
        str(kantakaupunki_pbf),
        modes=("walk", "bicycle", "e_scooter", "car"),
        country="FI",
    )


@pytest.fixture(scope="module")
def origins():
    return scattered_points(4, seed=1)


@pytest.fixture(scope="module")
def destinations():
    return scattered_points(3, seed=2)


def test_the_car_routes_and_beats_walking(car_network):
    car = car_network.travel_time(KAMPPI, ARABIA, mode="car")
    walk = car_network.travel_time(KAMPPI, ARABIA, mode="walk")
    assert car is not None and walk is not None
    assert car < walk


def test_the_default_is_free_flow_and_delays_are_opt_in(car_network):
    free = car_network.travel_time(KAMPPI, ARABIA, mode="car")
    midday = car_network.travel_time(
        KAMPPI, ARABIA, mode="car", intersection_delays=True
    )
    rush = car_network.travel_time(
        KAMPPI, ARABIA, mode="car", intersection_delays=True, profile="rush"
    )
    day = car_network.travel_time(
        KAMPPI, ARABIA, mode="car", intersection_delays=True, profile="day-average"
    )
    assert free < midday <= rush
    assert free < day


def test_delay_options_require_the_gate_and_the_car(car_network):
    with pytest.raises(ValueError, match="intersection_delays=True"):
        car_network.travel_time(KAMPPI, ARABIA, mode="car", profile="rush")
    with pytest.raises(ValueError, match="intersection_delays=True"):
        car_network.travel_time(KAMPPI, ARABIA, mode="car", delay_model={"values": {}})
    with pytest.raises(ValueError, match="mode='car'"):
        car_network.travel_time(
            KAMPPI, ARABIA, mode="bicycle", intersection_delays=True
        )
    with pytest.raises(ValueError, match="unknown profile"):
        car_network.travel_time(
            KAMPPI, ARABIA, mode="car", intersection_delays=True, profile="night"
        )


def test_a_delay_model_override_changes_the_answer(car_network):
    midday = car_network.travel_time(
        KAMPPI, ARABIA, mode="car", intersection_delays=True
    )
    slowed = car_network.travel_time(
        KAMPPI,
        ARABIA,
        mode="car",
        intersection_delays=True,
        delay_model={"values": {"4-6": {"midday": 30.0}}},
    )
    assert slowed > midday


def test_a_carless_build_refuses_car_queries(helsinki_streets):
    with pytest.raises(ValueError):
        helsinki_streets.travel_time(KAMPPI, ARABIA, mode="car")


def test_speed_limit_overrides_thread_through_the_build(kantakaupunki_pbf, car_network):
    # The public build surface forwards `speed_limits=` into the persisted
    # speeds: untagged ways take the overridden default (a sentinel value no
    # Helsinki tag carries), tagged ways keep their maxspeed either way.
    slowed = StreetNetwork.from_osm(
        str(kantakaupunki_pbf),
        modes=("walk", "bicycle", "e_scooter", "car"),
        country="FI",
        speed_limits={"residential_inside": 11, "other_inside": 11},
    )
    baseline = np.asarray(car_network._core._car_attributes[0])
    overridden = np.asarray(slowed._core._car_attributes[0])
    assert len(baseline) == len(overridden)
    assert not (baseline == 11.0).any()
    assert (overridden == 11.0).any()
    assert ((overridden == 11.0) | (overridden == baseline)).all()


def test_the_shipped_values_are_the_jaakkola_calibration():
    # Taulukko 28, seconds per crossing.
    assert _delays.DELAY_MODEL["values"] == {
        "1-2": {"rush": 12.195, "midday": 9.979, "day-average": 11.311},
        "3": {"rush": 11.199, "midday": 6.650, "day-average": 9.439},
        "4-6": {"rush": 10.633, "midday": 7.752, "day-average": 9.362},
    }
    assert _delays.DELAY_MODEL["ramp_multipliers"] == {
        "rush": 2.022762,
        "midday": 1.667750,
        "day-average": 1.884662,
    }
    assert _delays.DELAY_MODEL["congestion_multipliers"] == {
        "rush": 1.2,
        "midday": 1.0,
        "day-average": 1.1,
    }
    assert _delays.DELAY_MODEL["ramp_shares"]["midday"] == 0.75
    assert _delays.RAMP_SHARE_LOW == 0.5


def test_resolve_flattens_the_selected_period():
    payload = _delays.resolve(True)  # midday by default
    seconds, groups, share_high, share_low, ramp, congestion = payload
    assert seconds == [9.979, 6.650, 7.752]
    assert len(groups) == len(_osm.HIGHWAY_CODES)
    # motorway → group 1–2; secondary → 3; residential and the ramp
    # classes → 4–6.
    assert groups[_osm.HIGHWAY_CODES["motorway"]] == 0
    assert groups[_osm.HIGHWAY_CODES["secondary"]] == 1
    assert groups[_osm.HIGHWAY_CODES["residential"]] == 2
    assert groups[_osm.HIGHWAY_CODES["motorway_link"]] == 2
    assert (share_high, share_low) == (0.75, 0.5)
    assert (ramp, congestion) == (1.667750, 1.0)
    assert _delays.resolve(False) is None


def test_resolve_merges_partially_and_rejects_garbage():
    seconds, groups, *_ = _delays.resolve(
        True,
        "rush",
        {"values": {"3": {"rush": 20.0}}, "groups": {"residential": "1-2"}},
    )
    # The named entry changed; its siblings kept the shipped values.
    assert seconds == [12.195, 20.0, 10.633]
    assert groups[_osm.HIGHWAY_CODES["residential"]] == 0
    assert groups[_osm.HIGHWAY_CODES["unclassified"]] == 2
    for bad in [
        {"bogus": 1},
        {"values": {"7": {"rush": 1.0}}},
        {"values": {"3": {"night": 1.0}}},
        {"values": {"3": {"rush": float("nan")}}},
        {"values": {"3": {"rush": -1.0}}},
        {"groups": {"residential": "9"}},
        {"groups": {"skyway": "1-2"}},
        {"ramp_shares": {"night": 0.5}},
    ]:
        with pytest.raises(ValueError):
            _delays.resolve(True, None, bad)


def test_car_matrix_cells_match_single_routes(car_network, origins, destinations):
    matrix = TravelTimeMatrix(car_network, origins, destinations, transport_mode="car")
    assert len(matrix) > 0
    by_pair = {
        (row.from_id, row.to_id): row.travel_time_s for row in matrix.itertuples()
    }
    for _, origin in origins.iterrows():
        for _, destination in destinations.iterrows():
            single = car_network.travel_time(
                (origin.geometry.y, origin.geometry.x),
                (destination.geometry.y, destination.geometry.x),
                mode="car",
            )
            assert by_pair.get((origin["id"], destination["id"])) == single


def test_car_matrices_take_the_delay_options(car_network, origins, destinations):
    free = TravelTimeMatrix(car_network, origins, destinations, transport_mode="car")
    delayed = TravelTimeMatrix(
        car_network,
        origins,
        destinations,
        transport_mode="car",
        intersection_delays=True,
        profile="rush",
    )
    merged = pd.merge(
        free, delayed, on=["from_id", "to_id"], suffixes=("_free", "_rush")
    )
    assert len(merged) > 0
    assert (merged.travel_time_s_rush >= merged.travel_time_s_free).all()
    assert (merged.travel_time_s_rush > merged.travel_time_s_free).any()
    with pytest.raises(ValueError, match="intersection_delays=True"):
        TravelTimeMatrix(
            car_network,
            origins,
            destinations,
            transport_mode="car",
            profile="rush",
        )


def test_car_cost_matrix_reports_unresolved_emissions(
    car_network, origins, destinations
):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        matrix = TravelCostMatrix(
            car_network, origins, destinations, transport_mode="car"
        )
    assert len(matrix) > 0
    assert (matrix.distance_m > 0).all()
    # No shipped car emission factor yet: unresolved, never silently zero.
    assert matrix.emissions.isna().all()
    assert any("emission" in str(warning.message) for warning in caught)
    # A user-supplied factor row resolves it.
    factors = pd.DataFrame(
        [
            {
                "street_mode": "car",
                "vehicle_class": "ICE",
                "service_model": "private",
                "vehicle": 30.0,
                "fuel": 120.0,
                "infrastructure": 10.0,
                "operations": 2.0,
            }
        ]
    )
    priced = TravelCostMatrix(
        car_network, origins, destinations, transport_mode="car", factors=factors
    )
    expected = priced.network_distance_m / 1000.0 * 162.0
    assert np.allclose(priced.emissions, expected)


def test_car_itineraries_carry_geometry(car_network, origins, destinations):
    legs = DetailedItineraries(
        car_network,
        origins.head(2),
        destinations.head(2),
        transport_mode="car",
        intersection_delays=True,
    )
    assert len(legs) > 0
    assert (legs["mode"] == "car").all()
    assert legs.geometry.notna().all()


def test_transit_matrices_reject_the_delay_options(network):
    with pytest.raises(ValueError, match="StreetNetwork car matrix"):
        TravelTimeMatrix(
            network,
            ["1030423"],
            date="2022-02-22",
            departure="08:30:00",
            intersection_delays=True,
        )


def test_the_car_never_serves_transit_access(multimodal_network):
    with pytest.raises(ValueError, match="street-only"):
        multimodal_network._core._street_access_seconds(60.1690, 24.9320, "car", 900.0)
