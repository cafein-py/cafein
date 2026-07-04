"""TravelCostMatrix over the Helsinki network shared with r5py."""

import geopandas as gpd
import numpy as np
import pytest
import shapely

from cafein import TravelCostMatrix, emissions


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


def cost_matrix(network, **kwargs):
    # The HSL feed's ferries have no shipped factor; the warning is part
    # of the resolution contract and irrelevant to most assertions.
    with pytest.warns(UserWarning, match="route_type"):
        return TravelCostMatrix(network, date="2022-02-22", **kwargs)


def test_cost_rows_pin_the_k_train(network):
    matrix = cost_matrix(
        network,
        origins=["4810551"],
        destinations=["1250551"],
        departure="08:30:00",
        geometries=True,
    )
    row = matrix.iloc[0]
    assert (row.from_id, row.to_id) == ("4810551", "1250551")
    assert row.travel_time == 28 * 60
    assert row.transfers == 0
    assert row.transit_distance == pytest.approx(16_786, abs=1)
    assert row.walk_distance == 0.0
    # 16.786 km at the shipped 25 g/pkm urban-rail factor.
    assert row.emissions == pytest.approx(419.65, abs=0.1)
    assert row.geometry.geom_type == "MultiLineString"
    assert shapely.get_num_geometries(row.geometry) == 1


def test_cost_rows_count_walks_and_transfers(network_with_footpaths):
    matrix = cost_matrix(
        network_with_footpaths,
        origins=["1100602"],
        destinations=["1040280", "1250551"],
        departure="08:30:00",
    )
    # The 08:31 M2 to Kamppi, then the pinned 20-second footpath.
    m2 = matrix[matrix.to_id == "1040280"].iloc[0]
    assert m2.travel_time == 9 * 60 + 20
    assert m2.transfers == 0
    assert m2.transit_distance == pytest.approx(4_132, abs=1)
    assert 19 <= m2.walk_distance <= 20
    assert m2.emissions == pytest.approx(4.132 * 25, abs=0.1)
    # Reaching the K train takes a second vehicle.
    korso = matrix[matrix.to_id == "1250551"].iloc[0]
    assert korso.transfers >= 1
    assert korso.walk_distance > 100


def test_cost_matrix_matches_the_travel_time_matrix(network):
    origins = ["4810551", "1250551"]
    matrix = cost_matrix(network, origins=origins, departure="08:30:00")
    times = network.travel_time_matrix(origins, "2022-02-22", "08:30:00")
    stop_order = [stop for stop, _, _ in network.stops]
    for row_index, origin in enumerate(origins):
        rows = matrix[matrix.from_id == origin]
        reachable = np.nonzero(times[row_index] != np.uint32(0xFFFFFFFF))[0]
        assert len(rows) == len(reachable)
        expected = {stop_order[at]: int(times[row_index, at]) for at in reachable}
        assert dict(zip(rows.to_id, rows.travel_time)) == expected
    # The origin reaches itself without riding.
    self_row = matrix[(matrix.from_id == "4810551") & (matrix.to_id == "4810551")].iloc[
        0
    ]
    assert self_row.travel_time == 0
    assert self_row.transfers == 0
    assert self_row.transit_distance == 0.0
    assert self_row.emissions == 0.0


def test_cost_matrix_is_deterministic(network):
    origins = [stop for stop, _, _ in network.stops[:32]]
    first = cost_matrix(network, origins=origins, departure="08:30:00")
    second = cost_matrix(network, origins=origins, departure="08:30:00")
    assert first.equals(second)


def test_cost_matrix_requires_installed_payloads(helsinki_gtfs):
    from cafein import TransportNetwork

    lean = TransportNetwork.from_gtfs([str(helsinki_gtfs)], trip_distances=False)
    with pytest.raises(ValueError, match="no trip distances"):
        cost_matrix(lean, origins=["4810551"], departure="08:30:00")
    slim = TransportNetwork.from_gtfs([str(helsinki_gtfs)], leg_geometries=False)
    with pytest.raises(ValueError, match="no leg geometries"):
        cost_matrix(slim, origins=["4810551"], departure="08:30:00", geometries=True)
    # Costs without geometries stay available on the slim build.
    matrix = cost_matrix(slim, origins=["4810551"], departure="08:30:00")
    assert "geometry" not in matrix.columns
    assert len(matrix) > 0


def test_trip_factors_resolve_the_ladder(network):
    with pytest.warns(UserWarning, match="route_type"):
        factors = dict(emissions.trip_factors(network))
    assert len(factors) == network.trip_count
    assert factors["3001K_20220222_S1_2_0831"] == pytest.approx(25.0)


def test_point_matrices_walk_ride_and_walk(network_with_footpaths):
    origins = point_frame(
        network_with_footpaths,
        [("kalasatama", "1100602"), ("kamppi_metro", "1040602")],
    )
    destinations = point_frame(
        network_with_footpaths,
        [("kamppi_street", "1040280"), ("kapyla", "1250551")],
    )
    matrix = cost_matrix(
        network_with_footpaths,
        origins=origins,
        destinations=destinations,
        departure="08:30:00",
    )
    assert len(matrix) == 4
    # The door-to-door oracle: M2 plus access and egress walks.
    m2 = matrix[
        (matrix.from_id == "kalasatama") & (matrix.to_id == "kamppi_street")
    ].iloc[0]
    assert 558 <= m2.travel_time <= 562
    assert m2.transfers == 0
    assert m2.transit_distance == pytest.approx(4_132, abs=1)
    assert 27 <= m2.walk_distance <= 31
    assert m2.emissions == pytest.approx(4.132 * 25, abs=0.1)
    # A destination best reached on foot appears as a pure walk.
    walk = matrix[
        (matrix.from_id == "kamppi_metro") & (matrix.to_id == "kamppi_street")
    ].iloc[0]
    assert 19 <= walk.travel_time <= 21
    assert walk.transfers == 0
    assert walk.transit_distance == 0.0
    assert 19 <= walk.walk_distance <= 21
    assert walk.emissions == 0.0
    # The point travel-time matrix agrees pair by pair.
    times = network_with_footpaths.travel_time_matrix(
        origins, "2022-02-22", "08:30:00", destinations=destinations
    )
    for row in matrix.itertuples():
        origin = list(origins["id"]).index(row.from_id)
        destination = list(destinations["id"]).index(row.to_id)
        assert times[origin, destination] == row.travel_time


def test_point_matrices_report_unsnapped_points(network_with_footpaths):
    # Open water south of the extract: the point cannot snap.
    sea = gpd.GeoDataFrame(
        {"id": ["sea"]},
        geometry=gpd.points_from_xy([24.90], [60.14]),
        crs="EPSG:4326",
    )
    destinations = point_frame(network_with_footpaths, [("kamppi", "1040280")])
    with pytest.warns(UserWarning, match="off the walking network"):
        matrix = TravelCostMatrix(
            network_with_footpaths,
            sea,
            destinations,
            "2022-02-22",
            "08:30:00",
        )
    assert len(matrix) == 0
    with pytest.warns(UserWarning, match="off the walking network"):
        times = network_with_footpaths.travel_time_matrix(
            sea, "2022-02-22", "08:30:00", destinations=destinations
        )
    assert times.shape == (1, 1)
    assert times[0, 0] == np.uint32(0xFFFFFFFF)


def test_stop_matrices_reject_point_options(network):
    with pytest.raises(ValueError, match="point origins"):
        cost_matrix(
            network,
            origins=["4810551"],
            departure="08:30:00",
            walking_speed_kmph=5.0,
        )
    with pytest.raises(ValueError, match="point origins"):
        network.travel_time_matrix(
            ["4810551"], "2022-02-22", "08:30:00", max_walking_time=300.0
        )


def nearest_rank(sorted_samples, percentile):
    """The core's half-up nearest-rank convention."""
    position = percentile / 100 * (len(sorted_samples) - 1)
    return sorted_samples[int(position + 0.5)]


def test_window_percentiles_match_per_minute_runs(network):
    window = 1800
    percentiles = [0.0, 50.0, 100.0]
    matrix = network.travel_time_matrix(
        ["4810551"],
        "2022-02-22",
        "08:30:00",
        window=window,
        percentiles=percentiles,
    )
    assert matrix.shape == (1, network.stop_count, 3)
    stop_order = [stop for stop, _, _ in network.stops]
    per_minute = []
    for step in range(window // 60):
        mark = 8 * 3600 + 30 * 60 + 60 * step
        clock = f"{mark // 3600:02d}:{mark % 3600 // 60:02d}:00"
        per_minute.append(
            network.travel_times_from_stop("4810551", "2022-02-22", clock)
        )
    unreachable = int(np.uint32(0xFFFFFFFF))
    for column, stop in enumerate(stop_order):
        samples = sorted(times.get(stop, unreachable) for times in per_minute)
        for plane, percentile in enumerate(percentiles):
            assert matrix[0, column, plane] == nearest_rank(samples, percentile), (
                stop,
                percentile,
            )


def test_confidence_maps_to_the_symmetric_interval(network):
    left = network.travel_time_matrix(
        ["4810551"], "2022-02-22", "08:30:00", window=1800, confidence=0.8
    )
    right = network.travel_time_matrix(
        ["4810551"],
        "2022-02-22",
        "08:30:00",
        window=1800,
        percentiles=[10, 50, 90],
    )
    assert np.array_equal(left, right)
    korso = [stop for stop, _, _ in network.stops].index("1250551")
    lower, median, upper = left[0, korso]
    assert lower <= median <= upper
    assert lower < int(np.uint32(0xFFFFFFFF))


def test_point_window_percentiles_keep_walks_constant(network_with_footpaths):
    # A destination best reached on foot does not depend on the
    # departure time: every percentile equals the walk.
    origins = point_frame(network_with_footpaths, [("kamppi_metro", "1040602")])
    destinations = point_frame(network_with_footpaths, [("kamppi_street", "1040280")])
    matrix = network_with_footpaths.travel_time_matrix(
        origins,
        "2022-02-22",
        "08:30:00",
        destinations=destinations,
        window=1800,
        percentiles=[0, 50, 100],
    )
    assert matrix.shape == (1, 1, 3)
    assert 19 <= matrix[0, 0, 0] == matrix[0, 0, 1] == matrix[0, 0, 2] <= 21


def test_window_specifications_are_validated(network):
    with pytest.raises(ValueError, match="require a window"):
        network.travel_time_matrix(
            ["4810551"], "2022-02-22", "08:30:00", percentiles=[50]
        )
    with pytest.raises(ValueError, match="not both"):
        network.travel_time_matrix(
            ["4810551"],
            "2022-02-22",
            "08:30:00",
            window=600,
            percentiles=[50],
            confidence=0.8,
        )
    with pytest.raises(ValueError, match="within"):
        network.travel_time_matrix(
            ["4810551"], "2022-02-22", "08:30:00", window=600, confidence=1.5
        )
    with pytest.raises(ValueError, match="within"):
        network.travel_time_matrix(
            ["4810551"],
            "2022-02-22",
            "08:30:00",
            window=600,
            percentiles=[120],
        )
