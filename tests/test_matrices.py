"""TravelCostMatrix over the Helsinki network shared with r5py."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely

from cafein import TravelCostMatrix, TravelTimeMatrix, emissions


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


def test_least_emission_cells_match_the_frontier(tmp_path):
    from cafein import TransportNetwork, journey_frontier, least_emissions
    from test_frontier import build_two_line_gtfs

    feed = build_two_line_gtfs(tmp_path / "two_line_gtfs.zip")
    network = TransportNetwork.from_gtfs([str(feed)])
    frontier = journey_frontier(
        network, "A", "B", "2022-02-22", "08:00:00", window=1800
    )

    def cell(within=None):
        matrix = TravelCostMatrix(
            network,
            ["A"],
            ["B"],
            "2022-02-22",
            "08:00:00",
            optimize="emissions",
            window=1800,
            within=within,
        )
        rows = matrix[matrix.to_id == "B"]
        return rows.iloc[0] if len(rows) else None

    # The matrix cell is the frontier's least-emission pick, cell for
    # cell: the slow-clean tram unbudgeted, the fast-dirty bus chain
    # within 15 minutes, nothing within a minute.
    cleanest, oracle = cell(), least_emissions(frontier)
    assert cleanest["emissions"] == pytest.approx(oracle["emissions"])
    assert cleanest["travel_time"] == oracle["travel_time"] == 1800
    assert cleanest["transfers"] == 0
    budgeted, oracle = cell(within=900), least_emissions(frontier, within=900)
    assert budgeted["emissions"] == pytest.approx(oracle["emissions"])
    assert budgeted["travel_time"] == oracle["travel_time"] == 900
    assert budgeted["transfers"] == 1
    assert cell(within=60) is None


def test_point_emission_cells_prefer_walking(network_with_footpaths):
    origins = point_frame(network_with_footpaths, [("metro", "1040602")])
    destinations = point_frame(network_with_footpaths, [("street", "1040280")])
    matrix = cost_matrix(
        network_with_footpaths,
        origins=origins,
        destinations=destinations,
        departure="08:30:00",
        optimize="emissions",
        window=600,
    )
    row = matrix.iloc[0]
    assert row.emissions == 0.0
    assert row.transfers == 0
    assert row.transit_distance == 0.0
    assert 19 <= row.travel_time <= 21
    assert 19 <= row.walk_distance <= 21


def test_point_emission_cells_match_the_frontier(network_with_footpaths):
    from cafein import journey_frontier, least_emissions

    coordinates = {stop: (lat, lon) for stop, lat, lon in network_with_footpaths.stops}
    frontier = journey_frontier(
        network_with_footpaths,
        coordinates["1100602"],
        coordinates["1040280"],
        "2022-02-22",
        "08:30:00",
        window=600,
    )
    # A budget below the walking time forces a ride, so the cell must
    # equal the frontier's budgeted least-emission journey exactly.
    oracle = least_emissions(frontier, within=900)
    assert oracle["rides"] >= 1
    matrix = cost_matrix(
        network_with_footpaths,
        origins=point_frame(network_with_footpaths, [("a", "1100602")]),
        destinations=point_frame(network_with_footpaths, [("b", "1040280")]),
        departure="08:30:00",
        optimize="emissions",
        window=600,
        within=900,
    )
    row = matrix.iloc[0]
    assert row.emissions == pytest.approx(oracle["emissions"])
    assert row.travel_time == oracle["travel_time"]
    assert row.transfers == oracle["rides"] - 1


def test_least_fare_cells_match_the_frontier(tmp_path):
    from cafein import TransportNetwork, journey_frontier, least_fare
    from test_frontier import build_two_line_gtfs, two_line_fares

    feed = build_two_line_gtfs(tmp_path / "two_line_gtfs.zip")
    network = TransportNetwork.from_gtfs([str(feed)])
    fares = two_line_fares()
    frontier = journey_frontier(
        network, "A", "B", "2022-02-22", "08:00:00", window=1800, fares=fares
    )

    def cell(within=None):
        matrix = TravelCostMatrix(
            network,
            ["A"],
            ["B"],
            "2022-02-22",
            "08:00:00",
            optimize="fare",
            window=1800,
            within=within,
            fares=fares,
        )
        rows = matrix[matrix.to_id == "B"]
        return rows.iloc[0] if len(rows) else None

    # The matrix cell is the frontier's cheapest pick, budget for
    # budget: the tram unbudgeted, then the out-of-allowance bus chain,
    # then the fast chain at the pair total, nothing within a minute.
    for within in (None, 1380, 900):
        row, oracle = cell(within), least_fare(frontier, within=within)
        assert row["fare"] == pytest.approx(oracle["fare"])
        assert row["travel_time"] == oracle["travel_time"]
    assert cell(within=60) is None


def test_fare_columns_price_the_reported_journeys(network, helsinki_gtfs):
    from cafein import fares as fare_module

    hsl = fare_module.zone_fare_structure(helsinki_gtfs)
    matrix = cost_matrix(
        network,
        origins=["4810551"],
        destinations=["1250551"],
        departure="08:30:00",
        fares=hsl,
    )
    row = matrix.iloc[0]
    # Korso (C) → Käpylä (A) prices at the ABC ticket, and the matrix
    # price equals the routed journey's python-side price.
    assert row.fare == pytest.approx(4.1)
    journeys = network.route_between_stops(
        "4810551", "1250551", "2022-02-22", "08:30:00"
    )
    fare_module.annotate_fares(journeys, hsl)
    fastest = min(journeys, key=lambda journey: journey["arrival"])
    assert row.fare == pytest.approx(fastest["fare"])
    # A seeded rule-based structure prices per boarding: the base fare,
    # one discounted transfer at the pair total (= base), then full
    # fares — so a cell's fare follows its transfer count exactly.
    seeded = fare_module.setup_fare_structure(network, base_fare=3.0)
    bulk = cost_matrix(
        network,
        origins=["4810551", "1040602", "1250551"],
        departure="08:30:00",
        fares=seeded,
    )
    expected = np.where(
        bulk["travel_time"] == 0, 0.0, np.maximum(bulk["transfers"], 1) * 3.0
    )
    assert bulk["fare"].to_numpy() == pytest.approx(expected)


def test_fare_cells_survive_unresolved_emissions(network, helsinki_gtfs):
    from cafein import fares as fare_module

    hsl = fare_module.zone_fare_structure(helsinki_gtfs)
    # Each objective qualifies by its own key: the factorless ferry to
    # Suomenlinna prices at the zone ticket under the fare objective,
    # while the emissions objective has no qualifying candidate.
    cheapest = cost_matrix(
        network,
        origins=["1080701"],
        destinations=["1520703"],
        departure="10:00:00",
        optimize="fare",
        window=3600,
        fares=hsl,
    )
    assert cheapest.iloc[0].fare == pytest.approx(2.8)
    assert np.isnan(cheapest.iloc[0].emissions)
    cleanest = cost_matrix(
        network,
        origins=["1080701"],
        destinations=["1520703"],
        departure="10:00:00",
        optimize="emissions",
        window=3600,
        fares=hsl,
    )
    assert cleanest.empty


def test_point_fare_cells_prefer_walking(network_with_footpaths, helsinki_gtfs):
    from cafein import fares as fare_module

    hsl = fare_module.zone_fare_structure(helsinki_gtfs)
    matrix = cost_matrix(
        network_with_footpaths,
        origins=point_frame(network_with_footpaths, [("metro", "1040602")]),
        destinations=point_frame(network_with_footpaths, [("street", "1040280")]),
        departure="08:30:00",
        optimize="fare",
        window=600,
        fares=hsl,
    )
    row = matrix.iloc[0]
    assert row.fare == 0.0
    assert row.transfers == 0
    assert row.transit_distance == 0.0
    assert 19 <= row.travel_time <= 21


def test_fare_matrices_validate_their_options(network, helsinki_gtfs):
    from cafein import fares as fare_module

    with pytest.raises(ValueError, match="requires a fare structure"):
        TravelCostMatrix(
            network,
            ["4810551"],
            date="2022-02-22",
            departure="08:30:00",
            optimize="fare",
            window=600,
        )
    with pytest.raises(ValueError, match="requires a departure window"):
        TravelCostMatrix(
            network,
            ["4810551"],
            date="2022-02-22",
            departure="08:30:00",
            optimize="fare",
            fares=fare_module.zone_fare_structure(helsinki_gtfs),
        )
    # Without a fare structure no fare column appears.
    plain = cost_matrix(
        network, origins=["4810551"], destinations=["1250551"], departure="08:30:00"
    )
    assert "fare" not in plain.columns


def test_emission_matrices_validate_their_options(network):
    with pytest.raises(ValueError, match="requires a departure window"):
        TravelCostMatrix(
            network,
            ["4810551"],
            date="2022-02-22",
            departure="08:30:00",
            optimize="emissions",
        )
    with pytest.raises(ValueError, match="optimize='emissions'"):
        TravelCostMatrix(
            network,
            ["4810551"],
            date="2022-02-22",
            departure="08:30:00",
            within=600,
        )
    with pytest.raises(ValueError, match="optimize must be"):
        TravelCostMatrix(
            network,
            ["4810551"],
            date="2022-02-22",
            departure="08:30:00",
            optimize="fastest",
        )


def test_emission_cells_never_exceed_the_fastest_journeys_emissions(network):
    fastest = cost_matrix(
        network,
        origins=["4810551"],
        destinations=["1250551"],
        departure="08:30:00",
    )
    cleanest = cost_matrix(
        network,
        origins=["4810551"],
        destinations=["1250551"],
        departure="08:30:00",
        optimize="emissions",
        window=600,
    )
    assert len(fastest) == 1 and len(cleanest) == 1
    assert cleanest.iloc[0].emissions <= fastest.iloc[0].emissions
    # The zero-ride floor: the origin reaches itself at zero cost in the
    # emission mode too.
    self_cell = cost_matrix(
        network,
        origins=["4810551"],
        destinations=["4810551"],
        departure="08:30:00",
        optimize="emissions",
        window=600,
    ).iloc[0]
    assert self_cell.travel_time == 0
    assert self_cell.emissions == 0.0
    assert self_cell.transfers == 0


def test_point_matrices_take_the_direct_walk(tmp_path):
    from cafein import TransportNetwork
    from test_transport_network import build_synthetic_gtfs

    feed = build_synthetic_gtfs(tmp_path / "synthetic_gtfs.zip")
    with pytest.warns(UserWarning):
        network = TransportNetwork.from_gtfs([str(feed)])
    # One 2 km street edge with stops at its start and at 90 % of its
    # cost length. The query points sit at 25 % and 30 %: walking the
    # 100 m between them directly beats any walk through a stop (the
    # nearest is 500 m back) — the cell must hold the direct walk.
    network.set_street_network(
        2,
        [(0, 1, 2000.0)],
        [0, 2],
        [24.0, 24.035842],
        [60.0, 60.0],
        [("S1", 0, 0.0, 0.0), ("S2", 0, 0.9, 0.0)],
    )
    origins = gpd.GeoDataFrame(
        {"id": ["a"]},
        geometry=gpd.points_from_xy([24.0089605], [60.0]),
        crs="EPSG:4326",
    )
    destinations = gpd.GeoDataFrame(
        {"id": ["b"]},
        geometry=gpd.points_from_xy([24.0107526], [60.0]),
        crs="EPSG:4326",
    )
    times = network.travel_time_matrix(
        origins, "2022-02-22", "07:30:00", destinations=destinations
    )
    assert times[0, 0] in (100, 101)
    # A walk is departure-independent: every percentile plane holds it.
    windowed = network.travel_time_matrix(
        origins,
        "2022-02-22",
        "07:30:00",
        destinations=destinations,
        window=600,
        confidence=0.8,
    )
    assert set(windowed[0, 0, :].tolist()) == {times[0, 0]}
    # The cost matrix reports the same walking-only pair: no rides, no
    # transit distance, no emissions, the walk as the distance.
    matrix = TravelCostMatrix(
        network, origins, destinations, "2022-02-22", "07:30:00", geometries=True
    )
    row = matrix.iloc[0]
    assert row["travel_time"] == times[0, 0]
    assert row["transfers"] == 0
    assert row["transit_distance"] == 0.0
    assert row["walk_distance"] == pytest.approx(100.0, abs=0.5)
    assert row["emissions"] == 0.0
    assert row["geometry"].geom_type == "MultiLineString"


def test_point_matrices_report_unsnapped_points(network_with_footpaths):
    # Open water south of the extract: the point cannot snap.
    sea = gpd.GeoDataFrame(
        {"id": ["sea"]},
        geometry=gpd.points_from_xy([24.90], [60.10]),
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


def test_confidence_bounds_equal_their_decimal_percentiles(network):
    # 31 samples put the 5th percentile's half-up rank exactly on a tie:
    # (1 - 0.9) / 2 * 100 must reach the core as 5, not 4.999999999999999.
    left = network.travel_time_matrix(
        ["4810551"], "2022-02-22", "08:30:00", window=1860, confidence=0.9
    )
    right = network.travel_time_matrix(
        ["4810551"],
        "2022-02-22",
        "08:30:00",
        window=1860,
        percentiles=[5, 50, 95],
    )
    assert np.array_equal(left, right)


def test_chunks_partition_the_matrix(network):
    import pandas as pd

    origins = [stop for stop, _, _ in network.stops[:10]]
    full = cost_matrix(
        network, origins=origins, destinations=["1250551"], departure="08:30:00"
    )
    parts = [
        cost_matrix(
            network,
            origins=origins,
            destinations=["1250551"],
            departure="08:30:00",
            chunk=(part, 3),
        )
        for part in range(3)
    ]
    assert pd.concat(parts, ignore_index=True).equals(pd.DataFrame(full))

    matrix = network.travel_time_matrix(origins, "2022-02-22", "08:30:00")
    rows = [
        network.travel_time_matrix(origins, "2022-02-22", "08:30:00", chunk=(part, 3))
        for part in range(3)
    ]
    assert np.array_equal(np.vstack(rows), matrix)


def test_chunk_specifications_are_validated(network):
    for chunk in [(3, 3), (-1, 3), (0, 0)]:
        with pytest.raises(ValueError, match="chunk"):
            network.travel_time_matrix(
                ["4810551"], "2022-02-22", "08:30:00", chunk=chunk
            )


def test_arrow_table_matches_the_dataframe(network, tmp_path):
    pyarrow = pytest.importorskip("pyarrow")
    import pyarrow.parquet
    from cafein import travel_cost_table

    with pytest.warns(UserWarning, match="route_type"):
        table = travel_cost_table(
            network,
            origins=["4810551"],
            destinations=["1250551"],
            date="2022-02-22",
            departure="08:30:00",
            geometries=True,
        )
    frame = cost_matrix(
        network,
        origins=["4810551"],
        destinations=["1250551"],
        departure="08:30:00",
        geometries=True,
    )
    assert pyarrow.types.is_dictionary(table.schema.field("from_id").type)
    assert table.column("from_id").to_pylist() == list(frame.from_id)
    assert table.column("to_id").to_pylist() == list(frame.to_id)
    assert table.column("travel_time").to_pylist() == list(frame.travel_time)
    assert table.column("transfers").to_pylist() == list(frame.transfers)
    assert table.column("emissions").to_pylist() == pytest.approx(list(frame.emissions))
    decoded = shapely.from_wkb(table.column("geometry").to_pylist()[0])
    assert decoded.equals(frame.geometry.iloc[0])
    # The documented shard workflow: write one chunk, read it back.
    shard = tmp_path / "shard-0000.parquet"
    pyarrow.parquet.write_table(table, shard)
    assert pyarrow.parquet.read_table(shard).num_rows == len(frame)


def test_arrow_tables_need_pyarrow(network, monkeypatch):
    import builtins

    from cafein import travel_cost_table

    real_import = builtins.__import__

    def no_pyarrow(name, *args, **kwargs):
        if name == "pyarrow":
            raise ImportError("pyarrow is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pyarrow)
    with pytest.raises(ImportError, match="cafein\\[arrow\\]"):
        travel_cost_table(
            network,
            origins=["4810551"],
            date="2022-02-22",
            departure="08:30:00",
        )


UNREACHABLE = np.iinfo(np.uint32).max


def test_travel_time_matrix_unstacks_the_wide_matrix(network):
    origins = ["4810551", "1250551"]
    wide = network.travel_time_matrix(origins, "2022-02-22", "08:30:00")
    stops = [stop for stop, _lat, _lon in network.stops]
    matrix = TravelTimeMatrix(network, origins, date="2022-02-22", departure="08:30:00")
    assert list(matrix.columns) == ["from_id", "to_id", "travel_time"]
    # Every reachable wide cell is a row; unreachable cells are absent.
    assert len(matrix) == int((wide != UNREACHABLE).sum())
    long = {
        (row.from_id, row.to_id): int(row.travel_time) for row in matrix.itertuples()
    }
    reference = {
        (origin, stops[column]): int(wide[index, column])
        for index, origin in enumerate(origins)
        for column in range(wide.shape[1])
        if wide[index, column] != UNREACHABLE
    }
    assert long == reference
    # The Korso -> Käpylä pair keeps its 28-minute travel time.
    korso = matrix[(matrix.from_id == "4810551") & (matrix.to_id == "1250551")]
    assert int(korso.travel_time.iloc[0]) == 28 * 60
    # Slices degrade to plain DataFrames.
    assert type(matrix.iloc[:1]) is pd.DataFrame


def test_travel_time_matrix_windowed_percentiles(network):
    origins = ["4740551"]
    percentiles = [10, 50, 90]
    wide = network.travel_time_matrix(
        origins,
        "2022-02-22",
        "08:00:00",
        window=1800,
        percentiles=percentiles,
    )
    stops = [stop for stop, _lat, _lon in network.stops]
    matrix = TravelTimeMatrix(
        network,
        origins,
        date="2022-02-22",
        departure="08:00:00",
        window=1800,
        percentiles=percentiles,
    )
    assert list(matrix.columns) == [
        "from_id",
        "to_id",
        "travel_time_p10",
        "travel_time_p50",
        "travel_time_p90",
    ]
    # Each row equals the corresponding wide percentile plane, cell for
    # cell, with unreachable percentile cells read as NaN.
    for row in matrix.itertuples():
        column = stops.index(row.to_id)
        for offset, percentile in enumerate((10, 50, 90)):
            wide_value = wide[0, column, offset]
            long_value = getattr(row, f"travel_time_p{percentile}")
            if wide_value == UNREACHABLE:
                assert np.isnan(long_value)
            else:
                assert long_value == wide_value
    # Percentiles are ordered within a reachable row.
    reachable = matrix.dropna()
    assert (reachable.travel_time_p10 <= reachable.travel_time_p50).all()
    assert (reachable.travel_time_p50 <= reachable.travel_time_p90).all()


def test_travel_time_matrix_over_points(network_with_footpaths):
    origins = point_frame(network_with_footpaths, [("A", "1100602")])
    destinations = point_frame(
        network_with_footpaths, [("B", "1040280"), ("C", "1250551")]
    )
    wide = network_with_footpaths.travel_time_matrix(
        origins, "2022-02-22", "08:30:00", destinations=destinations
    )
    matrix = TravelTimeMatrix(
        network_with_footpaths,
        origins,
        destinations,
        date="2022-02-22",
        departure="08:30:00",
    )
    long = {
        (row.from_id, row.to_id): int(row.travel_time) for row in matrix.itertuples()
    }
    reference = {
        ("A", destination): int(wide[0, column])
        for column, destination in enumerate(["B", "C"])
        if wide[0, column] != UNREACHABLE
    }
    assert long == reference
    assert (matrix.from_id == "A").all()


def test_travel_time_matrix_chunks_partition_origins(network):
    origins = ["4810551", "1250551", "4740551"]
    full = TravelTimeMatrix(network, origins, date="2022-02-22", departure="08:30:00")
    parts = [
        TravelTimeMatrix(
            network, origins, date="2022-02-22", departure="08:30:00", chunk=(k, 3)
        )
        for k in range(3)
    ]
    stitched = pd.concat(parts, ignore_index=True)
    assert len(stitched) == len(full)
    assert set(map(tuple, stitched[["from_id", "to_id"]].to_numpy())) == set(
        map(tuple, full[["from_id", "to_id"]].to_numpy())
    )


def test_travel_time_matrix_requires_date_and_departure(network):
    with pytest.raises(TypeError, match="requires date and departure"):
        TravelTimeMatrix(network, ["4810551"])


def test_travel_time_matrix_defaults_to_all_stops(network):
    stops = [stop for stop, _lat, _lon in network.stops]
    # Omitted origins mean every stop; the first origin chunk keeps the
    # all-stops resolution cheap to exercise.
    matrix = TravelTimeMatrix(
        network,
        date="2022-02-22",
        departure="08:30:00",
        chunk=(0, network.stop_count),
    )
    assert set(matrix.from_id) == {stops[0]}
    assert set(matrix.to_id) <= set(stops)
    assert len(matrix) > 0
