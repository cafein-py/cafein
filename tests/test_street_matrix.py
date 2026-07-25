"""Standalone travel-time matrices over a StreetNetwork."""

import math

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cafein._cafein")

from cafein import StreetNetwork, TravelTimeMatrix  # noqa: E402


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
def streets(kantakaupunki_pbf):
    return StreetNetwork.from_osm(str(kantakaupunki_pbf))


@pytest.fixture(scope="module")
def origins():
    return scattered_points(6, seed=1)


@pytest.fixture(scope="module")
def destinations():
    return scattered_points(5, seed=2)


def coordinates(frame):
    return list(zip(frame.geometry.y, frame.geometry.x))


@pytest.mark.parametrize("mode", ["walk", "bicycle", "e_bike", "e_scooter"])
def test_matrix_cells_match_single_routes(streets, origins, destinations, mode):
    # The oracle: every cell is the answer the one-pair route gives. The matrix
    # runs one search per origin rather than one per pair, so this pins that the
    # shared arrival step produces identical times.
    matrix = TravelTimeMatrix(
        streets, origins, destinations, transport_mode=mode, max_street_time=3600
    )
    cells = {
        (row.from_id, row.to_id): int(row.travel_time)
        for row in matrix.itertuples(index=False)
    }
    for from_id, origin in zip(origins["id"], coordinates(origins)):
        for to_id, destination in zip(destinations["id"], coordinates(destinations)):
            expected = streets.travel_time(
                origin, destination, mode=mode, max_time=3600
            )
            assert cells.get((from_id, to_id)) == expected


def test_matrix_columns_and_long_format(streets, origins, destinations):
    matrix = TravelTimeMatrix(streets, origins, destinations, transport_mode="bicycle")
    assert list(matrix.columns) == ["from_id", "to_id", "travel_time"]
    assert len(matrix) <= len(origins) * len(destinations)
    assert matrix["travel_time"].dtype == np.uint32
    # Slices degrade to plain DataFrames, as the transit matrix does.
    assert type(matrix.iloc[:1]) is pd.DataFrame


def test_diagonal_is_zero_when_destinations_are_the_origins(streets, origins):
    matrix = TravelTimeMatrix(streets, origins, transport_mode="bicycle")
    diagonal = matrix[matrix.from_id == matrix.to_id]
    assert len(diagonal) == len(origins)
    assert (diagonal.travel_time == 0).all()


def test_cycling_beats_walking_over_the_same_pairs(streets, origins, destinations):
    walk = TravelTimeMatrix(
        streets, origins, destinations, transport_mode="walk", max_street_time=3600
    )
    bicycle = TravelTimeMatrix(
        streets, origins, destinations, transport_mode="bicycle", max_street_time=3600
    )
    walked = {(r.from_id, r.to_id): r.travel_time for r in walk.itertuples(index=False)}
    rode = {
        (r.from_id, r.to_id): r.travel_time for r in bicycle.itertuples(index=False)
    }
    shared = set(walked) & set(rode)
    assert shared
    # Off-diagonal pairs: cycling is never slower and usually faster.
    off = [pair for pair in shared if pair[0] != pair[1]]
    assert off
    assert all(rode[pair] <= walked[pair] for pair in off)
    assert sum(rode[pair] < walked[pair] for pair in off) > len(off) / 2


def test_a_tighter_cutoff_drops_cells(streets, origins, destinations):
    generous = TravelTimeMatrix(
        streets, origins, destinations, transport_mode="walk", max_street_time=3600
    )
    tight = TravelTimeMatrix(
        streets, origins, destinations, transport_mode="walk", max_street_time=120
    )
    assert len(tight) < len(generous)
    assert (tight.travel_time <= 120).all()


def test_matrix_is_deterministic(streets, origins, destinations):
    first = TravelTimeMatrix(streets, origins, destinations, transport_mode="bicycle")
    second = TravelTimeMatrix(streets, origins, destinations, transport_mode="bicycle")
    assert first.equals(second)


def test_chunks_partition_origins(streets, origins, destinations):
    whole = TravelTimeMatrix(streets, origins, destinations, transport_mode="bicycle")
    shards = pd.concat(
        [
            TravelTimeMatrix(
                streets,
                origins,
                destinations,
                transport_mode="bicycle",
                chunk=(index, 3),
            )
            for index in range(3)
        ]
    )
    assert len(shards) == len(whole)
    pairs = {(r.from_id, r.to_id) for r in whole.itertuples(index=False)}
    assert {(r.from_id, r.to_id) for r in shards.itertuples(index=False)} == pairs


def test_unsnappable_origins_warn_and_are_absent(streets, destinations):
    far = gpd.GeoDataFrame(
        {"id": ["atlantic"]},
        geometry=gpd.points_from_xy([-30.0], [0.0]),
        crs="EPSG:4326",
    )
    with pytest.warns(UserWarning, match="atlantic"):
        matrix = TravelTimeMatrix(streets, far, destinations, transport_mode="bicycle")
    assert len(matrix) == 0


def test_street_network_requires_an_explicit_mode(streets, origins):
    with pytest.raises(TypeError, match="explicit transport_mode"):
        TravelTimeMatrix(streets, origins)


def test_public_transport_mode_needs_a_transport_network(streets, origins):
    with pytest.raises(ValueError, match="needs a TransportNetwork"):
        TravelTimeMatrix(streets, origins, transport_mode="public_transport")


def test_unknown_mode_is_rejected(streets, origins):
    with pytest.raises(ValueError, match="unknown transport_mode"):
        TravelTimeMatrix(streets, origins, transport_mode="hovercraft")


def test_street_mode_on_a_transport_network_is_rejected(network):
    with pytest.raises(ValueError, match="is a street mode"):
        TravelTimeMatrix(
            network, None, None, "2022-02-22", "08:30:00", transport_mode="bicycle"
        )


def test_max_street_time_on_a_transport_network_is_rejected(network):
    with pytest.raises(ValueError, match="applies to a StreetNetwork"):
        TravelTimeMatrix(
            network, None, None, "2022-02-22", "08:30:00", max_street_time=600
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"date": "2022-02-22"},
        {"departure": "08:30:00"},
        {"max_transfers": 3},
        {"router": "raptor"},
        {"window": 600},
        {"percentiles": [50.0]},
        {"confidence": 90.0},
        {"exclude_routes": ["1001"]},
        {"exclude_trips": ["t1"]},
        {"exclude_stops": ["s1"]},
        {"walking_speed_kmph": 5.0},
        {"max_walking_time": 600.0},
    ],
)
def test_transit_only_arguments_are_rejected(streets, origins, kwargs):
    name = next(iter(kwargs))
    with pytest.raises(ValueError, match=f"{name}.*no meaning for a street matrix"):
        TravelTimeMatrix(streets, origins, transport_mode="bicycle", **kwargs)


def test_stop_id_origins_are_rejected(streets):
    with pytest.raises(TypeError, match="GeoDataFrame of points"):
        TravelTimeMatrix(streets, ["1010101"], transport_mode="bicycle")


@pytest.mark.parametrize("cutoff", [-1.0, float("nan"), float("inf")])
def test_a_cutoff_admitting_nothing_leaves_even_the_diagonal_unreachable(
    streets, origins, cutoff
):
    # The same-coordinate zero is still a route the cutoff must admit, so an
    # unusable cutoff leaves the cell absent — exactly as the single route
    # reports it, which is what makes the matrix and the route agree.
    matrix = TravelTimeMatrix(
        streets, origins, transport_mode="bicycle", max_street_time=cutoff
    )
    assert len(matrix) == 0
    origin = coordinates(origins)[0]
    assert streets.travel_time(origin, origin, mode="bicycle", max_time=cutoff) is None
