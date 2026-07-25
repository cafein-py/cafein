"""Standalone DetailedItineraries over a StreetNetwork."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cafein._cafein")

from cafein import DetailedItineraries, StreetNetwork, TravelCostMatrix  # noqa: E402
from cafein._cafein import STREET_DISTANCE_PROVENANCE  # noqa: E402

COLUMNS = [
    "from_id",
    "to_id",
    "option",
    "segment",
    "leg_type",
    "mode",
    "departure",
    "arrival",
    "travel_time",
    "distance_m",
    "network_distance_m",
    "connector_distance_m",
    "distance_provenance",
    "geometry",
]


@pytest.fixture(scope="module")
def streets(kantakaupunki_pbf):
    return StreetNetwork.from_osm(str(kantakaupunki_pbf))


@pytest.fixture(scope="module")
def places():
    return gpd.GeoDataFrame(
        {"id": ["kamppi", "hakaniemi", "toolo"]},
        geometry=gpd.points_from_xy(
            [24.9320, 24.9520, 24.9220], [60.1690, 60.1795, 60.1810]
        ),
        crs="EPSG:4326",
    )


def test_columns_and_frame_type(streets, places):
    routes = DetailedItineraries(streets, places, transport_mode="bicycle")
    assert isinstance(routes, gpd.GeoDataFrame)
    assert list(routes.columns) == COLUMNS
    assert routes.crs == "EPSG:4326"
    assert routes.geometry.name == "geometry"
    assert (routes.distance_provenance == STREET_DISTANCE_PROVENANCE).all()


@pytest.mark.parametrize("mode", ["walk", "bicycle", "e_bike", "e_scooter"])
def test_one_leg_per_pair_carrying_its_mode(streets, places, mode):
    routes = DetailedItineraries(streets, places, transport_mode=mode)
    assert len(routes) > 0
    # A standalone street journey is a single leg.
    assert (routes.option == 0).all()
    assert (routes.segment == 0).all()
    assert (routes["mode"] == mode).all()
    # A direct non-walking leg takes its mode as leg_type, as a direct walk does.
    assert (routes.leg_type == routes["mode"]).all()
    # One row per reachable pair, never more.
    pairs = list(zip(routes.from_id, routes.to_id))
    assert len(pairs) == len(set(pairs))


def test_agrees_with_the_cost_matrix(streets, places):
    # Both read the same reconstruction; they must not disagree.
    routes = DetailedItineraries(streets, places, transport_mode="bicycle")
    costs = TravelCostMatrix(streets, places, transport_mode="bicycle")
    columns = [
        "travel_time",
        "distance_m",
        "network_distance_m",
        "connector_distance_m",
    ]
    from_routes = {
        (r.from_id, r.to_id): tuple(getattr(r, c) for c in columns)
        for r in routes.itertuples(index=False)
    }
    from_costs = {
        (r.from_id, r.to_id): tuple(getattr(r, c) for c in columns)
        for r in costs.itertuples(index=False)
    }
    assert from_routes.keys() == from_costs.keys()
    for pair, values in from_routes.items():
        assert values == pytest.approx(from_costs[pair])


def test_absolute_times_are_absent_without_a_departure(streets, places):
    routes = DetailedItineraries(streets, places, transport_mode="bicycle")
    assert routes.departure.isna().all()
    assert routes.arrival.isna().all()
    assert (routes.travel_time >= 0).all()


def test_a_departure_places_the_leg_on_a_clock(streets, places):
    routes = DetailedItineraries(
        streets, places, departure="08:30:00", transport_mode="bicycle"
    )
    start = 8 * 3600 + 30 * 60
    assert (routes.departure == start).all()
    assert (routes.arrival == start + routes.travel_time).all()


def test_geometry_runs_from_origin_to_destination(streets, places):
    routes = DetailedItineraries(streets, places, transport_mode="bicycle")
    coordinates = dict(zip(places["id"], zip(places.geometry.y, places.geometry.x)))
    off_diagonal = routes[routes.from_id != routes.to_id]
    assert len(off_diagonal) > 0
    for row in off_diagonal.itertuples(index=False):
        assert row.geometry.geom_type == "LineString"
        start, end = row.geometry.coords[0], row.geometry.coords[-1]
        # Coordinates are (longitude, latitude); check both, not just one.
        from_lat, from_lon = coordinates[row.from_id]
        to_lat, to_lon = coordinates[row.to_id]
        assert start == pytest.approx((from_lon, from_lat), abs=1e-6)
        assert end == pytest.approx((to_lon, to_lat), abs=1e-6)


def test_the_diagonal_is_a_readable_zero_length_leg(streets, places):
    routes = DetailedItineraries(streets, places, transport_mode="walk")
    diagonal = routes[routes.from_id == routes.to_id]
    assert len(diagonal) == len(places)
    assert (diagonal.travel_time == 0).all()
    # Subscripted, not attribute access: `.distance_m` is geopandas' own method.
    assert (diagonal["distance_m"] == 0).all()
    coordinates = dict(zip(places["id"], zip(places.geometry.y, places.geometry.x)))
    for row in diagonal.itertuples(index=False):
        shape = row.geometry
        assert shape.geom_type == "LineString"
        assert len(shape.coords) == 2
        # Both ends sit on the coordinate itself, so the leg has no extent.
        latitude, longitude = coordinates[row.from_id]
        assert shape.coords[0] == pytest.approx((longitude, latitude), abs=1e-6)
        assert shape.coords[0] == shape.coords[-1]
        assert shape.length == 0.0


def test_geometries_false_drops_the_shapes(streets, places):
    routes = DetailedItineraries(
        streets, places, transport_mode="walk", geometries=False
    )
    assert routes.geometry.isna().all()
    assert (routes.travel_time >= 0).all()


def test_requires_an_explicit_mode(streets, places):
    with pytest.raises(TypeError, match="explicit transport_mode"):
        DetailedItineraries(streets, places)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"date": "2022-02-22"},
        {"max_transfers": 3},
        {"router": "raptor"},
        {"candidates": "pareto"},
        {"bucket": 50.0},
        {"factors": object()},
        {"components": ["vehicle"]},
        {"slack_seconds": 300},
        {"max_options": 3},
        {"diversity": "spread"},
        {"penalty": 60},
        {"exclude_routes": ["1001"]},
        {"walking_speed_kmph": 5.0},
        {"max_walking_time": 600.0},
    ],
)
def test_rejects_transit_only_arguments(streets, places, kwargs):
    name = next(iter(kwargs))
    with pytest.raises(ValueError, match=f"{name}.*no meaning for a street matrix"):
        DetailedItineraries(streets, places, transport_mode="bicycle", **kwargs)


def test_unsnappable_origin_warns(streets, places):
    far = gpd.GeoDataFrame(
        {"id": ["atlantic"]},
        geometry=gpd.points_from_xy([-30.0], [0.0]),
        crs="EPSG:4326",
    )
    with pytest.warns(UserWarning, match="atlantic"):
        routes = DetailedItineraries(streets, far, places, transport_mode="bicycle")
    assert len(routes) == 0


def test_slices_stay_geodataframes(streets, places):
    routes = DetailedItineraries(streets, places, transport_mode="bicycle")
    assert isinstance(routes.iloc[:1], gpd.GeoDataFrame)


@pytest.mark.parametrize("departure", [None, "08:30:00"])
def test_the_column_dtypes_are_the_same_with_or_without_a_departure(
    streets, places, departure
):
    # The clock columns are nullable integer seconds either way: supplying a
    # departure fills them, it does not reshape the shipped schema.
    routes = DetailedItineraries(
        streets, places, departure=departure, transport_mode="bicycle"
    )
    assert routes.departure.dtype == "Int64"
    assert routes.arrival.dtype == "Int64"
    assert routes.travel_time.dtype == np.uint32
    assert routes.option.dtype == np.int64
    assert routes.segment.dtype == np.int64
    for column in ("distance_m", "network_distance_m", "connector_distance_m"):
        assert routes[column].dtype == np.float64
    for column in ("from_id", "to_id", "leg_type", "mode"):
        assert pd.api.types.is_string_dtype(routes[column])
    assert pd.api.types.is_string_dtype(routes.distance_provenance)
    assert routes.geometry.dtype == "geometry"


def test_transit_itineraries_reject_the_street_keywords(network):
    # They were unknown keywords before, so accepting and ignoring them would
    # quietly hand back transit itineraries for a street request.
    with pytest.raises(ValueError, match="is a street mode"):
        DetailedItineraries(
            network,
            ["1010101"],
            ["1010102"],
            "2022-02-22",
            "08:30:00",
            transport_mode="bicycle",
        )
    with pytest.raises(ValueError, match="applies to a StreetNetwork"):
        DetailedItineraries(
            network,
            ["1010101"],
            ["1010102"],
            "2022-02-22",
            "08:30:00",
            max_street_time=600,
        )
