"""Standalone travel-time matrices over a StreetNetwork."""

import math
import warnings

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
def streets(helsinki_streets):
    return helsinki_streets


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
        streets,
        origins,
        destinations,
        transport_mode=mode,
        max_street_time=60,
        output_time_units="seconds",
    )
    cells = {
        (row.from_id, row.to_id): int(row.travel_time)
        for row in matrix.itertuples(index=False)
    }
    for from_id, origin in zip(origins["id"], coordinates(origins)):
        for to_id, destination in zip(destinations["id"], coordinates(destinations)):
            expected = streets.travel_time(
                origin, destination, mode=mode, max_travel_time=60
            )
            assert cells.get((from_id, to_id)) == expected


def test_matrix_columns_and_long_format(streets, origins, destinations):
    matrix = TravelTimeMatrix(
        streets,
        origins,
        destinations,
        transport_mode="bicycle",
        output_time_units="seconds",
    )
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
        streets, origins, destinations, transport_mode="walk", max_street_time=60
    )
    bicycle = TravelTimeMatrix(
        streets, origins, destinations, transport_mode="bicycle", max_street_time=60
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
        streets, origins, destinations, transport_mode="walk", max_street_time=60
    )
    tight = TravelTimeMatrix(
        streets, origins, destinations, transport_mode="walk", max_street_time=2
    )
    assert len(tight) < len(generous)
    # Default output is whole minutes: every cell fits the two-minute
    # cutoff in its own unit.
    assert (tight.travel_time <= 2).all()
    exact = TravelTimeMatrix(
        streets,
        origins,
        destinations,
        transport_mode="walk",
        max_street_time=2,
        output_time_units="seconds",
    )
    assert (exact.travel_time <= 120).all()


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


def test_street_matrix_mode_and_input_refusals(streets, origins):
    # The wrong mode names and origin shapes are refused loudly.
    for build, exception, match in [
        (
            lambda: TravelTimeMatrix(
                streets, origins, transport_mode="public_transport"
            ),
            ValueError,
            "needs a TransportNetwork",
        ),
        (
            lambda: TravelTimeMatrix(streets, origins, transport_mode="hovercraft"),
            ValueError,
            "unknown transport_mode",
        ),
        (
            lambda: TravelTimeMatrix(streets, ["1010101"], transport_mode="bicycle"),
            TypeError,
            "GeoDataFrame of points",
        ),
    ]:
        with pytest.raises(exception, match=match):
            build()


def test_street_keywords_on_a_transport_network_are_rejected(network):
    # They were unknown keywords before, so accepting and ignoring them would
    # quietly hand back transit results for a street request.
    for computer, args, kwargs, match in [
        (
            TravelTimeMatrix,
            (None, None, "2022-02-22 08:30:00"),
            {"transport_mode": "bicycle"},
            "is a street mode",
        ),
        (
            TravelTimeMatrix,
            (None, None, "2022-02-22 08:30:00"),
            {"max_street_time": 10},
            "applies to a StreetNetwork",
        ),
        (
            TravelCostMatrix,
            (None, None, "2022-02-22 08:30:00"),
            {"transport_mode": "bicycle"},
            "is a street mode",
        ),
        (
            DetailedItineraries,
            (["1010101"], ["1010102"], "2022-02-22 08:30:00"),
            {"transport_mode": "bicycle"},
            "is a street mode",
        ),
        (
            DetailedItineraries,
            (["1010101"], ["1010102"], "2022-02-22 08:30:00"),
            {"max_street_time": 10},
            "applies to a StreetNetwork",
        ),
    ]:
        with pytest.raises(ValueError, match=match):
            computer(network, *args, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"departure": "2022-02-22 08:30:00"},
        {"max_rides": 3},
        {"router": "raptor"},
        {"departure_time_window": 10},
        {"percentiles": [50.0]},
        {"confidence": 90.0},
        {"exclude_routes": ["1001"]},
        {"exclude_trips": ["t1"]},
        {"exclude_stops": ["s1"]},
        {"walking_speed_kmph": 5.0},
        {"max_walking_time": 10.0},
    ],
)
def test_transit_only_arguments_are_rejected(streets, origins, kwargs):
    name = next(iter(kwargs))
    with pytest.raises(ValueError, match=f"{name}.*no meaning for a street matrix"):
        TravelTimeMatrix(streets, origins, transport_mode="bicycle", **kwargs)


# ---- Cost matrix ----

from cafein import DetailedItineraries, TravelCostMatrix  # noqa: E402
from cafein._cafein import STREET_DISTANCE_PROVENANCE  # noqa: E402


def _on_network_points(streets, shift=0.0):
    """Two coordinates lying on the street network, optionally nudged off it.

    A route's shape runs origin, origin-snap, ..., destination-snap,
    destination, so its second and second-to-last vertices are the snap points
    — exactly on an edge rather than merely near one.
    """
    probe = gpd.GeoDataFrame(
        {"id": ["a", "b"]},
        geometry=gpd.points_from_xy([24.9320, 24.9420], [60.1690, 60.1740]),
        crs="EPSG:4326",
    )
    route = TravelCostMatrix(
        streets,
        probe.iloc[[0]],
        probe.iloc[[1]],
        transport_mode="walk",
        geometries=True,
    )
    shape = route.iloc[0].geometry.coords
    ends = [shape[1], shape[-2]]
    return gpd.GeoDataFrame(
        {"id": ["a", "b"]},
        geometry=gpd.points_from_xy(
            [lon for lon, _ in ends], [lat + shift for _, lat in ends]
        ),
        crs="EPSG:4326",
    )


COST_COLUMNS = [
    "from_id",
    "to_id",
    "travel_time",
    "distance_m",
    "network_distance_m",
    "connector_distance_m",
    "distance_provenance",
    "emissions",
]


def test_cost_matrix_columns_and_dtypes(streets, origins, destinations):
    costs = TravelCostMatrix(
        streets,
        origins,
        destinations,
        transport_mode="bicycle",
        output_time_units="seconds",
    )
    assert list(costs.columns) == COST_COLUMNS
    assert costs.travel_time.dtype == np.uint32
    for column in ("distance_m", "network_distance_m", "connector_distance_m"):
        assert costs[column].dtype == np.float64
    assert pd.api.types.is_string_dtype(costs.distance_provenance)
    assert (costs.distance_provenance == STREET_DISTANCE_PROVENANCE).all()
    assert type(costs.iloc[:1]) is pd.DataFrame


@pytest.mark.parametrize("mode", ["walk", "bicycle", "e_scooter"])
def test_geometries_do_not_change_the_numbers(streets, origins, destinations, mode):
    # Without geometries the rows come from the metres-only search rather than
    # the reconstructed legs, so the two must agree on every cell they report:
    # the same pairs, the same times, and the same distances.
    plain = TravelCostMatrix(streets, origins, destinations, transport_mode=mode)
    shaped = TravelCostMatrix(
        streets, origins, destinations, transport_mode=mode, geometries=True
    )
    assert len(plain) > 0
    columns = [
        "from_id",
        "to_id",
        "travel_time",
        "distance_m",
        "network_distance_m",
        "connector_distance_m",
    ]
    pd.testing.assert_frame_equal(plain[columns], shaped[columns])


def test_geometries_do_not_change_the_diagonal(streets, origins):
    # Destinations default to the origins, so the same-coordinate zero — which
    # neither search settles, both branches write it in — is exercised too.
    plain = TravelCostMatrix(streets, origins, transport_mode="bicycle")
    shaped = TravelCostMatrix(
        streets, origins, transport_mode="bicycle", geometries=True
    )
    diagonal = plain[plain.from_id == plain.to_id]
    assert len(diagonal) == len(origins)
    assert (diagonal.travel_time == 0).all()
    assert (diagonal.distance_m == 0.0).all()
    pd.testing.assert_frame_equal(
        plain[["from_id", "to_id", "travel_time", "distance_m"]],
        shaped[["from_id", "to_id", "travel_time", "distance_m"]],
    )
    # The diagonal's zero-length route must still be a usable LineString:
    # degenerate by nature — a route to the same coordinate has no extent —
    # it must round-trip through WKB as a two-point LineString rather than
    # the one-point shape shapely refuses to read.
    for shape in shaped[shaped.from_id == shaped.to_id].geometry:
        assert shape.geom_type == "LineString"
        assert len(shape.coords) == 2
        assert shape.length == 0.0


def test_cost_matrix_distance_is_the_sum_of_its_parts(streets, origins, destinations):
    costs = TravelCostMatrix(streets, origins, destinations, transport_mode="bicycle")
    assert len(costs) > 0
    total = costs.network_distance_m + costs.connector_distance_m
    assert np.allclose(costs.distance_m, total)
    assert (costs.network_distance_m >= 0).all()
    assert (costs.connector_distance_m >= 0).all()


def test_cost_matrix_times_match_the_time_matrix(streets, origins, destinations):
    # The two computers run the same search; they must not disagree.
    times = TravelTimeMatrix(streets, origins, destinations, transport_mode="bicycle")
    costs = TravelCostMatrix(streets, origins, destinations, transport_mode="bicycle")
    timed = {(r.from_id, r.to_id): r.travel_time for r in times.itertuples(index=False)}
    costed = {
        (r.from_id, r.to_id): r.travel_time for r in costs.itertuples(index=False)
    }
    assert timed == costed


def test_cost_matrix_matches_single_pair_reconstruction(streets, origins, destinations):
    # Each row must equal what a one-pair matrix reports for the same
    # coordinates — time and both reconstructed distances, not just the time.
    whole = TravelCostMatrix(streets, origins, destinations, transport_mode="walk")
    rows = {
        (r.from_id, r.to_id): (
            r.travel_time,
            r.network_distance_m,
            r.connector_distance_m,
        )
        for r in whole.itertuples(index=False)
    }
    assert rows
    for index, from_id in enumerate(origins["id"]):
        for column, to_id in enumerate(destinations["id"]):
            single = TravelCostMatrix(
                streets,
                origins.iloc[[index]],
                destinations.iloc[[column]],
                transport_mode="walk",
            )
            if single.empty:
                assert (from_id, to_id) not in rows
                continue
            only = single.iloc[0]
            assert rows[(from_id, to_id)] == pytest.approx(
                (only.travel_time, only.network_distance_m, only.connector_distance_m)
            )


def test_connectors_are_zero_on_the_network_and_positive_off_it(streets):
    # Coordinates lifted from the network's own geometry snap onto it with no
    # connector; the same pair nudged away pays one at each end.
    costs = TravelCostMatrix(
        streets, _on_network_points(streets), transport_mode="walk"
    )
    off_diagonal = costs[costs.from_id != costs.to_id]
    assert len(off_diagonal) > 0
    assert (off_diagonal.connector_distance_m < 1.0).all()
    assert (off_diagonal.network_distance_m > 0).all()

    offset = _on_network_points(streets, shift=0.0004)  # ~45 m north
    away = TravelCostMatrix(streets, offset, transport_mode="walk")
    away_off_diagonal = away[away.from_id != away.to_id]
    assert (away_off_diagonal.connector_distance_m > 1.0).all()


def test_cycling_detours_cover_more_network_than_walking(
    streets, origins, destinations
):
    # Where a bicycle may not use the most direct way it detours, so its
    # network distance over the shared pairs is longer somewhere and never
    # shorter by more than rounding.
    walk = TravelCostMatrix(
        streets, origins, destinations, transport_mode="walk", max_street_time=60
    )
    bicycle = TravelCostMatrix(
        streets, origins, destinations, transport_mode="bicycle", max_street_time=60
    )
    walked = {
        (r.from_id, r.to_id): r.network_distance_m
        for r in walk.itertuples(index=False)
        if r.from_id != r.to_id
    }
    rode = {
        (r.from_id, r.to_id): r.network_distance_m
        for r in bicycle.itertuples(index=False)
        if r.from_id != r.to_id
    }
    shared = set(walked) & set(rode)
    assert shared
    assert any(rode[pair] > walked[pair] + 1.0 for pair in shared)


def test_cost_matrix_geometry(streets, origins, destinations):
    costs = TravelCostMatrix(
        streets, origins, destinations, transport_mode="bicycle", geometries=True
    )
    assert list(costs.columns) == COST_COLUMNS + ["geometry"]
    assert costs.geometry.dtype == object
    off_diagonal = costs[costs.from_id != costs.to_id]
    assert len(off_diagonal) > 0
    for shape in off_diagonal.geometry:
        assert shape.geom_type == "LineString"
        assert len(shape.coords) >= 2
    # A route's shape starts at its origin and ends at its destination.
    places = dict(zip(origins["id"], coordinates(origins)))
    targets = dict(zip(destinations["id"], coordinates(destinations)))
    row = off_diagonal.iloc[0]
    start, end = row.geometry.coords[0], row.geometry.coords[-1]
    assert abs(start[1] - places[row.from_id][0]) < 1e-6
    assert abs(end[1] - targets[row.to_id][0]) < 1e-6


@pytest.mark.parametrize(
    "kwargs",
    [
        {"departure": "2022-02-22 08:30:00"},
        {"optimize": "emissions"},
        {"router": "raptor"},
        {"max_rides": 3},
        {"departure_time_window": 10},
        {"max_travel_time": 10},
        {"candidates": "pareto"},
        {"bucket": 50.0},
        {"fares": object()},
        {"exclude_routes": ["1001"]},
        {"exclude_trips": ["t1"]},
        {"exclude_stops": ["s1"]},
        {"walking_speed_kmph": 5.0},
        {"max_walking_time": 10.0},
    ],
)
def test_cost_matrix_rejects_transit_only_arguments(streets, origins, kwargs):
    name = next(iter(kwargs))
    with pytest.raises(ValueError, match=f"{name}.*no meaning for a street matrix"):
        TravelCostMatrix(streets, origins, transport_mode="bicycle", **kwargs)


@pytest.mark.parametrize(
    "computer", [TravelTimeMatrix, TravelCostMatrix, DetailedItineraries]
)
def test_a_street_network_requires_an_explicit_mode(streets, origins, computer):
    with pytest.raises(TypeError, match="explicit transport_mode"):
        computer(streets, origins)


@pytest.mark.parametrize("cutoff", [-1.0, float("nan"), float("inf")])
def test_an_unusable_cutoff_is_refused_loudly(streets, origins, cutoff):
    # A negative, NaN, or infinite duration is a caller error, not an
    # empty result — on either matrix and on the one-pair query.
    with pytest.raises(ValueError, match="max_street_time"):
        TravelTimeMatrix(
            streets, origins, transport_mode="bicycle", max_street_time=cutoff
        )
    with pytest.raises(ValueError, match="max_street_time"):
        TravelCostMatrix(
            streets, origins, transport_mode="bicycle", max_street_time=cutoff
        )
    origin = coordinates(origins)[0]
    with pytest.raises(ValueError, match="max_travel_time"):
        streets.travel_time(origin, origin, mode="bicycle", max_travel_time=cutoff)


@pytest.mark.parametrize(
    "computer", [TravelTimeMatrix, TravelCostMatrix, DetailedItineraries]
)
def test_unsnappable_origins_warn_and_are_absent(streets, destinations, computer):
    far = gpd.GeoDataFrame(
        {"id": ["atlantic"]},
        geometry=gpd.points_from_xy([-30.0], [0.0]),
        crs="EPSG:4326",
    )
    with pytest.warns(UserWarning, match="atlantic"):
        frame = computer(streets, far, destinations, transport_mode="bicycle")
    assert len(frame) == 0


# ---- Emissions ----

BICYCLE_ROW = {
    "street_mode": "bicycle",
    "vehicle_class": "conventional",
    "service_model": "private",
    "vehicle": 5.0,
    "fuel": 0.0,
    "infrastructure": 15.0,
    "operations": 0.0,
}


def test_emissions_are_network_kilometres_at_the_resolved_factor(
    streets, origins, destinations
):
    # The hand-calculated example: 20 g/pkm across the four components, so
    # each cell's grams are its network metres — connectors excluded — at 20.
    factors = pd.DataFrame([BICYCLE_ROW])
    costs = TravelCostMatrix(
        streets, origins, destinations, transport_mode="bicycle", factors=factors
    )
    assert len(costs) > 0
    expected = costs.network_distance_m / 1000.0 * 20.0
    assert np.allclose(costs.emissions, expected)
    # Connectors are excluded: pairs with real connectors would differ if the
    # total distance were used instead.
    off = costs[costs.connector_distance_m > 1.0]
    assert len(off) > 0
    assert not np.allclose(off.emissions, off.distance_m / 1000.0 * 20.0)


def test_emission_components_select_a_subset(streets, origins):
    factors = pd.DataFrame([BICYCLE_ROW])
    full = TravelCostMatrix(streets, origins, transport_mode="bicycle", factors=factors)
    vehicle_only = TravelCostMatrix(
        streets,
        origins,
        transport_mode="bicycle",
        factors=factors,
        components=["vehicle"],
    )
    off = full.from_id != full.to_id
    assert np.allclose(vehicle_only.emissions[off], full.emissions[off] * (5.0 / 20.0))


@pytest.mark.parametrize(
    ("mode", "components", "per_km"),
    [
        # The shipped sourced defaults: ITF "Good to Go?" components on the
        # Finland 2020 mix (via cafein-lca), the conventional bicycle's
        # dietary 21 g/km on top, walking the zero baseline.
        ("walk", None, 0.0),
        ("bicycle", None, 37.0),
        ("e_bike", None, 25.0),
        ("e_scooter", None, 36.0),
        # Narrowed to the use phase: walking is free; the conventional
        # bicycle carries the shipped dietary energy factor of 21 g/km;
        # the battery modes their Finnish-mix charging electricity.
        ("walk", ["fuel", "operations"], 0.0),
        ("bicycle", ["fuel", "operations"], 21.0),
        ("e_bike", ["fuel", "operations"], 3.0),
        ("e_scooter", ["fuel", "operations"], 1.0),
    ],
)
def test_default_full_lca_resolves_with_the_shipped_sources(
    streets, origins, mode, components, per_km
):
    kwargs = {} if components is None else {"components": components}
    costs = TravelCostMatrix(streets, origins, transport_mode=mode, **kwargs)
    expected = costs.network_distance_m / 1000.0 * per_km
    assert np.allclose(costs.emissions, expected)


def test_a_user_row_with_a_missing_selected_component_stays_unresolved(
    streets, origins
):
    # A user row resolves the ladder, but an NA in a selected component still
    # reports NA — a partial row must not quietly sum its known parts.
    partial = dict(BICYCLE_ROW)
    partial["fuel"] = float("nan")
    with pytest.warns(UserWarning, match="fuel"):
        costs = TravelCostMatrix(
            streets,
            origins,
            transport_mode="bicycle",
            factors=pd.DataFrame([partial]),
            components=["vehicle", "fuel"],
        )
    assert costs.emissions.isna().all()


def test_empty_components_are_rejected(streets, origins):
    with pytest.raises(ValueError, match="selects nothing"):
        TravelCostMatrix(streets, origins, transport_mode="bicycle", components=[])


MODE_ROW = {
    "street_mode": "bicycle",
    "vehicle": 100.0,
    "fuel": 0.0,
    "infrastructure": 0.0,
    "operations": 0.0,
}
PAIR_ROW = {
    "street_mode": "bicycle",
    "vehicle_class": "conventional",
    "vehicle": 30.0,
    "fuel": 0.0,
    "infrastructure": 10.0,
    "operations": 0.0,
}


@pytest.mark.parametrize(
    ("rows", "per_km"),
    [
        # The shipped table carries an exact-triple sourced row for every
        # mode; a sourced user row at bare street_mode specificity must
        # still win it.
        pytest.param(
            [
                {
                    "street_mode": "bicycle",
                    "vehicle": 10.0,
                    "fuel": 0.0,
                    "infrastructure": 10.0,
                    "operations": 0.0,
                }
            ],
            20.0,
            id="mode-level-beats-shipped",
        ),
        # A mode-level row is shadowed by the class-level pair — the middle
        # rung, with no service_model...
        pytest.param([MODE_ROW, PAIR_ROW], 40.0, id="pair-shadows-mode"),
        # ...which the exact triple shadows in turn.
        pytest.param([MODE_ROW, PAIR_ROW, BICYCLE_ROW], 20.0, id="triple-shadows-pair"),
    ],
)
def test_the_street_ladder_prefers_the_specific_row(streets, origins, rows, per_km):
    costs = TravelCostMatrix(
        streets, origins, transport_mode="bicycle", factors=pd.DataFrame(rows)
    )
    off = costs[costs.from_id != costs.to_id]
    assert len(off) > 0
    assert np.allclose(off.emissions, off.network_distance_m / 1000.0 * per_km)


def test_unmatchable_factor_rows_are_rejected(streets, origins):
    # A row shape the resolver can never match must error, not be silently
    # ignored — the ladder would otherwise fall through to the shipped
    # defaults and answer with their values instead of the user's.
    no_mode = pd.DataFrame(
        [{"vehicle_class": "conventional", "vehicle": 5.0, "fuel": 0.0}]
    )
    with pytest.raises(ValueError, match="no street_mode"):
        TravelCostMatrix(streets, origins, transport_mode="bicycle", factors=no_mode)
    dangling = pd.DataFrame(
        [
            {
                "street_mode": "bicycle",
                "service_model": "private",
                "vehicle": 5.0,
                "fuel": 0.0,
            }
        ]
    )
    with pytest.raises(ValueError, match="without vehicle_class"):
        TravelCostMatrix(streets, origins, transport_mode="bicycle", factors=dangling)


def test_itineraries_agree_with_the_cost_matrix_on_emissions(streets, origins):
    from cafein import DetailedItineraries

    factors = pd.DataFrame([BICYCLE_ROW])
    costs = TravelCostMatrix(
        streets, origins, transport_mode="bicycle", factors=factors
    )
    legs = DetailedItineraries(
        streets, origins, transport_mode="bicycle", factors=factors
    )
    by_pair = {(r.from_id, r.to_id): r.emissions for r in costs.itertuples(index=False)}
    for row in legs.itertuples(index=False):
        assert row.emissions == pytest.approx(by_pair[(row.from_id, row.to_id)])


def test_wheelchair_detours_the_stairs(streets, origins):
    # The longest stairway in the extract (75 m of highway=steps); walking
    # shortcuts straight over it, the wheelchair profile must go around.
    places = gpd.GeoDataFrame(
        {"id": ["top", "bottom"]},
        geometry=gpd.points_from_xy([24.931183, 24.9300526], [60.168853, 60.1684778]),
        crs="EPSG:4326",
    )
    walk = TravelTimeMatrix(
        streets, places, transport_mode="walk", output_time_units="seconds"
    )
    chair = TravelTimeMatrix(
        streets, places, transport_mode="wheelchair", output_time_units="seconds"
    )
    walked = walk.set_index(["from_id", "to_id"])["travel_time"]
    chaired = chair.set_index(["from_id", "to_id"])["travel_time"]
    # Same speed, stricter permissions: never faster, strictly slower here.
    assert chaired[("top", "bottom")] > walked[("top", "bottom")]
    assert chaired[("bottom", "top")] > walked[("bottom", "top")]
    # Same speed, subset permissions: wheelchair times dominate walking
    # everywhere, and match it away from the stairs.
    walk = TravelTimeMatrix(
        streets, origins, transport_mode="walk", output_time_units="seconds"
    )
    chair = TravelTimeMatrix(
        streets, origins, transport_mode="wheelchair", output_time_units="seconds"
    )
    merged = walk.merge(
        chair, on=["from_id", "to_id"], suffixes=("_walk", "_chair")
    ).dropna()
    assert (merged["travel_time_chair"] >= merged["travel_time_walk"]).all()
    assert (merged["travel_time_chair"] == merged["travel_time_walk"]).any()
    # A point halfway up the stairway: walking snaps onto the steps edge,
    # the wheelchair snap must find a permitted street instead — reachable
    # both, never faster wheeled.
    mid_stairs = gpd.GeoDataFrame(
        {"id": ["mid"]},
        geometry=gpd.points_from_xy([24.9306178], [60.1686654]),
        crs="EPSG:4326",
    )
    to_mid_walk = TravelTimeMatrix(
        streets,
        origins,
        mid_stairs,
        transport_mode="walk",
        output_time_units="seconds",
    )
    to_mid_chair = TravelTimeMatrix(
        streets,
        origins,
        mid_stairs,
        transport_mode="wheelchair",
        output_time_units="seconds",
    )
    walked = to_mid_walk.set_index("from_id")["travel_time"].dropna()
    chaired = to_mid_chair.set_index("from_id")["travel_time"].dropna()
    assert not chaired.empty
    shared = walked.index.intersection(chaired.index)
    assert (chaired[shared] >= walked[shared]).all()
