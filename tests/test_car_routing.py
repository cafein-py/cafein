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


def test_the_shipped_car_factors_are_gemmat_table_4():
    from cafein import emissions

    table = emissions.street_factors()
    cars = table[table.street_mode == "car"].set_index("vehicle_class")
    assert (cars.service_model == "private").all()
    # GEMMAT Table 4 per vehicle-km: (vehicle, fuel, infrastructure) with
    # zero operational services, and the published totals.
    expected = {
        "ICE": (24.0, 126.0, 12.0),
        "HEV": (26.0, 94.0, 13.0),
        "PHEV": (32.0, 43.0, 13.0),
        "BEV": (42.0, 16.0, 12.0),
        "FCEV": (38.0, 83.0, 13.0),
    }
    assert set(cars.index) == set(expected)
    for vehicle_class, (vehicle, fuel, infrastructure) in expected.items():
        row = cars.loc[vehicle_class]
        assert (row.vehicle, row.fuel, row.infrastructure, row.operations) == (
            vehicle,
            fuel,
            infrastructure,
            0.0,
        )
    totals = {"ICE": 162.0, "HEV": 133.0, "PHEV": 88.0, "BEV": 70.0, "FCEV": 134.0}
    for vehicle_class, total in totals.items():
        assert emissions.street_factor("car", vehicle_class=vehicle_class) == total
    # The default class is ICE.
    assert emissions.street_factor("car") == 162.0


def test_car_cost_matrix_resolves_emissions_by_class_and_occupancy(
    car_network, origins, destinations
):
    matrix = TravelCostMatrix(car_network, origins, destinations, transport_mode="car")
    assert len(matrix) > 0
    assert (matrix.distance_m > 0).all()
    # The default powertrain is ICE at the driver-only occupancy.
    assert np.allclose(matrix.emissions, matrix.network_distance_m / 1000.0 * 162.0)
    bev = TravelCostMatrix(
        car_network,
        origins,
        destinations,
        transport_mode="car",
        vehicle_class="BEV",
    )
    assert np.allclose(bev.emissions, bev.network_distance_m / 1000.0 * 70.0)
    # Occupancy divides the per-vehicle emissions, never the factors.
    for occupancy in (1.0, 1.6, 2.0, 4.0):
        shared = TravelCostMatrix(
            car_network,
            origins,
            destinations,
            transport_mode="car",
            occupancy=occupancy,
        )
        assert np.allclose(
            shared.emissions,
            shared.network_distance_m / 1000.0 * 162.0 / occupancy,
        )
    # Parking-search metres join the emissions basis before the division.
    parked = TravelCostMatrix(
        car_network,
        origins,
        destinations,
        transport_mode="car",
        parking=(60, 400.0),
        occupancy=2.0,
    )
    assert np.allclose(
        parked.emissions, parked.network_distance_m / 1000.0 * 162.0 / 2.0
    )
    # A user-supplied row still beats the shipped defaults.
    factors = pd.DataFrame(
        [
            {
                "street_mode": "car",
                "vehicle_class": "ICE",
                "service_model": "private",
                "vehicle": 10.0,
                "fuel": 80.0,
                "infrastructure": 8.0,
                "operations": 2.0,
            }
        ]
    )
    priced = TravelCostMatrix(
        car_network, origins, destinations, transport_mode="car", factors=factors
    )
    assert np.allclose(priced.emissions, priced.network_distance_m / 1000.0 * 100.0)


def test_car_geometries_do_not_change_the_numbers(car_network, origins, destinations):
    # A car matrix without geometries takes the metres-only search; the numbers
    # it reports are the reconstructed legs' — delays and parking included,
    # since both ride the same cells.
    columns = [
        "from_id",
        "to_id",
        "travel_time_s",
        "distance_m",
        "network_distance_m",
        "connector_distance_m",
        "emissions",
    ]
    for extra in ({}, {"intersection_delays": True}, {"parking": (60, 400.0)}):
        plain = TravelCostMatrix(
            car_network, origins, destinations, transport_mode="car", **extra
        )
        shaped = TravelCostMatrix(
            car_network,
            origins,
            destinations,
            transport_mode="car",
            geometries=True,
            **extra,
        )
        assert len(plain) > 0
        pd.testing.assert_frame_equal(plain[columns], shaped[columns])


def test_factor_validation_precedes_routing(
    car_network, origins, destinations, monkeypatch
):
    # The emission configuration is resolved before the search runs, so a
    # bad table fails fast and a mutable one cannot change mid-query.
    class NoRouting:
        def __getattr__(self, name):
            raise AssertionError("routing ran before factor validation")

    monkeypatch.setattr(car_network, "_core", NoRouting())
    with pytest.raises(ValueError, match="component"):
        TravelCostMatrix(
            car_network,
            origins,
            destinations,
            transport_mode="car",
            components=["bogus"],
        )


def test_car_emission_options_are_guarded(car_network, origins, destinations):
    for bad in ({"occupancy": 0}, {"occupancy": -1.0}, {"occupancy": float("nan")}):
        with pytest.raises(ValueError, match="occupancy"):
            TravelCostMatrix(
                car_network, origins, destinations, transport_mode="car", **bad
            )
    with pytest.raises(ValueError, match="transport_mode='car'"):
        TravelCostMatrix(
            car_network,
            origins,
            destinations,
            transport_mode="bicycle",
            occupancy=2.0,
        )
    with pytest.raises(ValueError, match="transport_mode='car'"):
        TravelCostMatrix(
            car_network,
            origins,
            destinations,
            transport_mode="walk",
            vehicle_class="BEV",
        )
    # An unknown powertrain resolves nothing: unresolved, never zero.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        unknown = TravelCostMatrix(
            car_network,
            origins,
            destinations,
            transport_mode="car",
            vehicle_class="DIESEL",
        )
    assert unknown.emissions.isna().all()
    assert any("emission" in str(warning.message) for warning in caught)


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
    # The emission options ride the leg surface too.
    shared_bev = DetailedItineraries(
        car_network,
        origins.head(2),
        destinations.head(2),
        transport_mode="car",
        vehicle_class="BEV",
        occupancy=2.0,
    )
    assert np.allclose(
        shared_bev.emissions,
        shared_bev.network_distance_m / 1000.0 * 70.0 / 2.0,
    )


def test_parking_is_off_by_default_and_adds_per_form(car_network):
    base = car_network.travel_time(KAMPPI, ARABIA, mode="car")
    assert car_network.travel_time(KAMPPI, ARABIA, mode="car", parking=False) == base
    assert (
        car_network.travel_time(KAMPPI, ARABIA, mode="car", parking=True) == base + 300
    )
    assert (
        car_network.travel_time(KAMPPI, ARABIA, mode="car", parking=120) == base + 120
    )
    assert (
        car_network.travel_time(KAMPPI, ARABIA, mode="car", parking=(60, 400.0))
        == base + 60
    )
    # Parking composes with the delay model.
    midday = car_network.travel_time(
        KAMPPI, ARABIA, mode="car", intersection_delays=True
    )
    assert (
        car_network.travel_time(
            KAMPPI, ARABIA, mode="car", intersection_delays=True, parking=True
        )
        == midday + 300
    )


def test_parking_rejects_bad_values_and_other_modes(car_network):
    for bad in (
        -1,
        float("nan"),
        (60, -5),
        (float("inf"), 0),
        "lot",
        (1, 2, 3),
        1e12,  # beyond the loud ceiling
    ):
        with pytest.raises(ValueError, match="parking"):
            car_network.travel_time(KAMPPI, ARABIA, mode="car", parking=bad)
    with pytest.raises(ValueError, match="mode='car'"):
        car_network.travel_time(KAMPPI, ARABIA, mode="walk", parking=True)
    # NumPy scalars are numbers too.
    base = car_network.travel_time(KAMPPI, ARABIA, mode="car")
    assert (
        car_network.travel_time(KAMPPI, ARABIA, mode="car", parking=np.int64(120))
        == base + 120
    )
    frame = gpd.GeoDataFrame(
        {"metres": [10.0]},
        geometry=[ARABIA_SQUARE],
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="seconds"):
        car_network.travel_time(KAMPPI, ARABIA, mode="car", parking=frame)
    missing_crs = gpd.GeoDataFrame({"seconds": [60.0]}, geometry=[ARABIA_SQUARE])
    with pytest.raises(ValueError, match="CRS"):
        car_network.travel_time(KAMPPI, ARABIA, mode="car", parking=missing_crs)
    from shapely.geometry import LineString

    lines = gpd.GeoDataFrame(
        {"seconds": [60.0]},
        geometry=[LineString([(24.9, 60.1), (25.0, 60.2)])],
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="Polygon"):
        car_network.travel_time(KAMPPI, ARABIA, mode="car", parking=lines)
    # The snapshot taken at resolve time is immune to later mutation.
    from cafein import _parking

    areas = gpd.GeoDataFrame(
        {"seconds": [60.0]}, geometry=[ARABIA_SQUARE], crs="EPSG:4326"
    )
    resolved = _parking.resolve(areas, "car")
    areas.loc[0, "seconds"] = 9999.0
    seconds, _ = _parking.destination_costs(resolved, [ARABIA])
    assert seconds[0] == 60.0


def _square(latitude, longitude, half=0.004):
    from shapely.geometry import Polygon

    return Polygon(
        [
            (longitude - half, latitude - half),
            (longitude + half, latitude - half),
            (longitude + half, latitude + half),
            (longitude - half, latitude + half),
        ]
    )


ARABIA_SQUARE = _square(*ARABIA)


def test_parking_areas_resolve_by_polygon_with_the_settled_tie_break(car_network):
    base = car_network.travel_time(KAMPPI, ARABIA, mode="car")
    # Two overlapping polygons over Arabia: the largest seconds wins; with
    # equal seconds the largest metres; with both equal the lowest row.
    overlapping = gpd.GeoDataFrame(
        {"seconds": [90.0, 240.0], "metres": [500.0, 100.0]},
        geometry=[ARABIA_SQUARE, _square(*ARABIA, half=0.008)],
        crs="EPSG:4326",
    )
    assert (
        car_network.travel_time(KAMPPI, ARABIA, mode="car", parking=overlapping)
        == base + 240
    )
    equal_seconds = gpd.GeoDataFrame(
        {"seconds": [240.0, 240.0], "metres": [100.0, 900.0]},
        geometry=[ARABIA_SQUARE, _square(*ARABIA, half=0.008)],
        crs="EPSG:4326",
    )
    from cafein import _parking

    seconds, metres = _parking.destination_costs(
        _parking.resolve(equal_seconds, "car"), [ARABIA]
    )
    assert (seconds[0], metres[0]) == (240.0, 900.0)
    full_tie = gpd.GeoDataFrame(
        {"seconds": [240.0, 240.0], "metres": [900.0, 900.0]},
        geometry=[_square(*ARABIA, half=0.008), ARABIA_SQUARE],
        crs="EPSG:4326",
    )
    seconds, metres = _parking.destination_costs(
        _parking.resolve(full_tie, "car"), [ARABIA]
    )
    assert (seconds[0], metres[0]) == (240.0, 900.0)
    # Outside every polygon the shipped constant applies, and a projected
    # frame reprojects onto the points' CRS.
    away = gpd.GeoDataFrame(
        {"seconds": [90.0]}, geometry=[_square(60.9, 24.0)], crs="EPSG:4326"
    )
    assert (
        car_network.travel_time(KAMPPI, ARABIA, mode="car", parking=away) == base + 300
    )
    projected = overlapping.to_crs("EPSG:3067")
    assert (
        car_network.travel_time(KAMPPI, ARABIA, mode="car", parking=projected)
        == base + 240
    )
    # Point-in-polygon is strict: a destination exactly on a polygon
    # boundary is outside and takes the fallback.
    from shapely.geometry import Polygon

    latitude, longitude = ARABIA
    touching = gpd.GeoDataFrame(
        {"seconds": [90.0]},
        geometry=[
            Polygon(
                [
                    (longitude - 0.004, latitude),
                    (longitude + 0.004, latitude),
                    (longitude + 0.004, latitude + 0.008),
                    (longitude - 0.004, latitude + 0.008),
                ]
            )
        ],
        crs="EPSG:4326",
    )
    from cafein import _parking

    seconds, _ = _parking.destination_costs(_parking.resolve(touching, "car"), [ARABIA])
    assert seconds[0] == 300.0


def test_parking_metres_join_distance_and_emissions(car_network, origins, destinations):
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
    plain = TravelCostMatrix(
        car_network, origins, destinations, transport_mode="car", factors=factors
    )
    parked = TravelCostMatrix(
        car_network,
        origins,
        destinations,
        transport_mode="car",
        factors=factors,
        parking=(60, 400.0),
    )
    merged = pd.merge(plain, parked, on=["from_id", "to_id"], suffixes=("", "_p"))
    assert len(merged) == len(plain) == len(parked)
    assert (merged.travel_time_s_p == merged.travel_time_s + 60).all()
    assert np.allclose(merged.network_distance_m_p, merged.network_distance_m + 400.0)
    assert np.allclose(merged.distance_m_p, merged.distance_m + 400.0)
    # The extra metres are driven: they join the emissions basis.
    assert np.allclose(merged.emissions_p, merged.network_distance_m_p / 1000.0 * 162.0)
    # The time matrix gains the same seconds per cell.
    times = TravelTimeMatrix(
        car_network, origins, destinations, transport_mode="car", parking=(60, 400.0)
    )
    with_base = pd.merge(plain, times, on=["from_id", "to_id"], suffixes=("", "_t"))
    assert (with_base.travel_time_s_t == with_base.travel_time_s + 60).all()
    # Itineraries shift arrivals and keep the geometry free of the search.
    legs = DetailedItineraries(
        car_network,
        origins.head(1),
        destinations.head(1),
        departure="08:00:00",
        transport_mode="car",
        parking=(60, 400.0),
    )
    bare = DetailedItineraries(
        car_network,
        origins.head(1),
        destinations.head(1),
        departure="08:00:00",
        transport_mode="car",
    )
    assert (legs.travel_time_s.to_numpy() == bare.travel_time_s.to_numpy() + 60).all()
    assert (legs.arrival_s.to_numpy() == bare.arrival_s.to_numpy() + 60).all()
    assert legs.geometry.iloc[0].equals(bare.geometry.iloc[0])


def test_transit_matrices_reject_the_delay_options(network):
    with pytest.raises(ValueError, match="StreetNetwork car matrix"):
        TravelTimeMatrix(
            network,
            ["1030423"],
            date="2022-02-22",
            departure="08:30:00",
            intersection_delays=True,
        )
    with pytest.raises(ValueError, match="StreetNetwork car matrix"):
        TravelTimeMatrix(
            network,
            ["1030423"],
            date="2022-02-22",
            departure="08:30:00",
            parking=True,
        )


def test_the_car_never_serves_transit_access(multimodal_network):
    with pytest.raises(ValueError, match="street-only"):
        multimodal_network._core._street_access_seconds(60.1690, 24.9320, "car", 900.0)


def test_parking_metres_join_the_cost_basis(car_network, origins, destinations):
    plain = TravelCostMatrix(
        car_network,
        origins,
        destinations,
        transport_mode="car",
        perspectives="private",
    )
    parked = TravelCostMatrix(
        car_network,
        origins,
        destinations,
        transport_mode="car",
        perspectives="private",
        parking=(60, 400.0),
    )
    merged = pd.merge(plain, parked, on=["from_id", "to_id"], suffixes=("", "_p"))
    assert len(merged) == len(plain)
    assert np.allclose(
        merged.cost_private_p, merged.network_distance_m_p / 1000.0 * 0.250
    )
    assert np.allclose(merged.cost_private_p - merged.cost_private, 0.4 * 0.250)
