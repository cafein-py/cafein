"""The monetary cost account of street travel (Gössling et al. 2019)."""

import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cafein._cafein")

from cafein import DetailedItineraries, TravelCostMatrix, costs  # noqa: E402


def scattered_points(count, seed, centre=(60.170, 24.940)):
    """Deterministic building-like locations around central Helsinki."""
    import math

    import geopandas as gpd

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
def origins():
    return scattered_points(3, seed=5)


@pytest.fixture(scope="module")
def destinations():
    return scattered_points(2, seed=6)


def test_the_shipped_table_is_gossling_table_2():
    table = costs.street_costs()
    assert list(table.columns) == [
        "perspective",
        "street_mode",
        "component",
        "cost_per_km",
    ]
    assert costs.CURRENCY == "EUR2017"
    private = table[table.perspective == "private"].set_index("street_mode")
    assert (private.component == "vehicle_operation").all()
    assert private.loc["car", "cost_per_km"] == 0.250
    assert private.loc["bicycle", "cost_per_km"] == 0.047
    assert private.loc["walk", "cost_per_km"] == 0.041
    societal = table[table.perspective == "societal"]
    car = societal[societal.street_mode == "car"].set_index("component")
    assert car.cost_per_km.to_dict() == {
        "climate": 0.011,
        "subsidies": 0.003,
        "air_pollution": 0.007,
        "noise": 0.007,
        "soil_water": 0.005,
        "infrastructure_construction": 0.030,
        "roadway_land": 0.011,
        "parking_land": 0.021,
        "infrastructure_maintenance": 0.004,
        "resources": 0.007,
        "accidents": 0.002,
    }
    assert car.cost_per_km.sum() == pytest.approx(0.108)
    # The health-dominated benefits stay signed, never clamped.
    external = societal.set_index(["street_mode", "component"])
    assert external.loc[("bicycle", "external"), "cost_per_km"] == -0.184
    assert external.loc[("walk", "external"), "cost_per_km"] == -0.370
    # No shipped e-scooter row: the paper carries none.
    assert not (table.street_mode == "e_scooter").any()


def test_street_cost_resolves_modes_and_components():
    assert costs.street_cost("car", "private") == 0.250
    assert costs.street_cost("car", "societal") == pytest.approx(0.108)
    # The e-bike rides the bicycle rows, exactly as in routing.
    assert costs.street_cost("e_bike", "societal") == -0.184
    # Component selection recomputes the derived total.
    assert costs.street_cost(
        "car", "societal", components=["climate", "accidents"]
    ) == pytest.approx(0.013)
    with pytest.raises(ValueError, match="not carried"):
        costs.street_cost("car", "societal", components=["health"])
    with pytest.raises(ValueError, match="component names"):
        costs.street_cost("car", "societal", components=[])
    # An uncovered mode resolves NaN with a warning, never zero.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = costs.street_cost("e_scooter", "societal")
    assert np.isnan(value)
    assert any("unresolved" in str(warning.message) for warning in caught)


def test_cost_overrides_merge_partially_and_validate():
    override = pd.DataFrame(
        [
            {
                "perspective": "societal",
                "street_mode": "car",
                "component": "climate",
                "cost_per_km": 0.020,
            },
            {
                "perspective": "societal",
                "street_mode": "e_scooter",
                "component": "external",
                "cost_per_km": 0.030,
            },
        ]
    )
    # The named component changed; its siblings kept the shipped values,
    # and the new mode row resolves.
    assert costs.street_cost("car", "societal", costs=override) == pytest.approx(
        0.108 - 0.011 + 0.020
    )
    assert costs.street_cost("e_scooter", "societal", costs=override) == 0.030
    for bad in [
        pd.DataFrame(
            [
                {
                    "perspective": "societal",
                    "street_mode": "bus",
                    "component": "climate",
                    "cost_per_km": 0.1,
                }
            ]
        ),
        pd.DataFrame(
            [
                {
                    "perspective": "commercial",
                    "street_mode": "car",
                    "component": "climate",
                    "cost_per_km": 0.1,
                }
            ]
        ),
        pd.DataFrame(
            [
                {
                    "perspective": "societal",
                    "street_mode": "car",
                    "component": "climate",
                    "cost_per_km": float("inf"),
                }
            ]
        ),
        pd.DataFrame(
            [
                {
                    "perspective": "societal",
                    "street_mode": "car",
                    "component": "",
                    "cost_per_km": 0.1,
                }
            ]
        ),
        override.iloc[[0, 0]],  # duplicate key
        pd.DataFrame([{"perspective": "societal", "cost_per_km": 0.1}]),
    ]:
        with pytest.raises(ValueError):
            costs.load_street_costs(bad)


def test_uncovered_modes_stay_nan_even_with_component_columns(
    helsinki_streets, origins, destinations
):
    # The e-scooter has no shipped row: the total and every requested
    # component column are NaN — unresolved, never a KeyError or zero.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scooter = TravelCostMatrix(
            helsinki_streets,
            origins,
            destinations,
            transport_mode="e_scooter",
            perspectives="societal",
            cost_components=["external"],
        )
    assert scooter.cost_societal.isna().all()
    assert scooter.cost_societal_external.isna().all()
    assert any("unresolved" in str(warning.message) for warning in caught)


def test_currency_needs_the_account_and_components_need_strings(
    helsinki_streets, origins, destinations
):
    # A currency without perspectives is rejected, never silently ignored.
    with pytest.raises(ValueError, match="perspectives"):
        TravelCostMatrix(
            helsinki_streets,
            origins,
            destinations,
            transport_mode="bicycle",
            currency="EUR2024",
        )
    # Non-string component names never enter a cost table.
    numeric = pd.DataFrame(
        [
            {
                "perspective": "societal",
                "street_mode": "car",
                "component": 7,
                "cost_per_km": 0.1,
            }
        ]
    )
    with pytest.raises(ValueError, match="a string"):
        costs.load_street_costs(numeric)


def test_cost_columns_ride_the_driven_kilometres(
    helsinki_streets, origins, destinations
):
    matrix = TravelCostMatrix(
        helsinki_streets,
        origins,
        destinations,
        transport_mode="bicycle",
        perspectives=("private", "societal"),
    )
    assert len(matrix) > 0
    kilometres = matrix.network_distance_m / 1000.0
    assert np.allclose(matrix.cost_private, kilometres * 0.047)
    assert np.allclose(matrix.cost_societal, kilometres * -0.184)
    assert (matrix.cost_societal < 0).all()
    assert (matrix.currency == "EUR2017").all()
    # A single perspective as a bare string, with component columns.
    walk = TravelCostMatrix(
        helsinki_streets,
        origins,
        destinations,
        transport_mode="walk",
        perspectives="private",
        cost_components=["vehicle_operation"],
        currency="EUR2024",
    )
    assert np.allclose(walk.cost_private, walk.network_distance_m / 1000.0 * 0.041)
    assert np.allclose(walk.cost_private_vehicle_operation, walk.cost_private)
    assert (walk.currency == "EUR2024").all()
    assert "cost_societal" not in walk.columns
    # An uncovered mode prices NaN, never zero.
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        scooter = TravelCostMatrix(
            helsinki_streets,
            origins,
            destinations,
            transport_mode="e_scooter",
            perspectives="societal",
        )
    assert scooter.cost_societal.isna().all()


def test_cost_options_are_guarded(helsinki_streets, origins, destinations, network):
    with pytest.raises(ValueError, match="perspectives"):
        TravelCostMatrix(
            helsinki_streets,
            origins,
            destinations,
            transport_mode="bicycle",
            cost_components=["vehicle_operation"],
        )
    with pytest.raises(ValueError, match="one perspective"):
        TravelCostMatrix(
            helsinki_streets,
            origins,
            destinations,
            transport_mode="bicycle",
            perspectives=("private", "societal"),
            cost_components=["vehicle_operation"],
        )
    with pytest.raises(ValueError, match="non-empty selection"):
        TravelCostMatrix(
            helsinki_streets,
            origins,
            destinations,
            transport_mode="bicycle",
            perspectives=("commercial",),
        )
    with pytest.raises(ValueError, match="currency"):
        TravelCostMatrix(
            helsinki_streets,
            origins,
            destinations,
            transport_mode="bicycle",
            perspectives="private",
            currency="",
        )
    # Transit surfaces reject the account loudly.
    with pytest.raises(ValueError, match="transit perspective costs"):
        TravelCostMatrix(
            network,
            ["1030423"],
            date="2022-02-22",
            departure="08:30:00",
            perspectives="societal",
        )


def test_itinerary_legs_carry_the_cost_account(helsinki_streets, origins, destinations):
    legs = DetailedItineraries(
        helsinki_streets,
        origins.head(2),
        destinations.head(1),
        transport_mode="bicycle",
        perspectives=("private", "societal"),
    )
    assert len(legs) > 0
    kilometres = legs.network_distance_m / 1000.0
    assert np.allclose(legs.cost_private, kilometres * 0.047)
    assert np.allclose(legs.cost_societal, kilometres * -0.184)
    assert (legs.currency == "EUR2017").all()
