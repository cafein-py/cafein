"""TravelCostMatrix over the Helsinki network shared with r5py."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely

from cafein import TravelCostMatrix, TravelTimeMatrix, emissions

UNREACHABLE_POINT = np.uint32(0xFFFFFFFF)


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
    departure = kwargs.pop("departure", "08:30:00")
    if len(departure) <= len("HH:MM:SS"):
        departure = f"2022-02-22 {departure}"
    with pytest.warns(UserWarning, match="route_type"):
        return TravelCostMatrix(network, departure=departure, **kwargs)


def scattered_points(network, count, seed):
    """Deterministic building-like locations: central stops offset by
    30–300 m in pseudo-random directions — arbitrary points on the
    street network rather than stop coordinates."""
    import math

    south, north, west, east = 60.155, 60.185, 24.91, 24.96
    central = [
        (stop, lat, lon)
        for stop, lat, lon in network.stops
        if lat is not None and south <= lat <= north and west <= lon <= east
    ]
    identifiers, latitudes, longitudes = [], [], []
    for index in range(count):
        _, lat, lon = central[(seed + index * 97) % len(central)]
        angle = (seed + index) * 2.399963  # the golden angle
        radius = 30 + (seed * 13 + index * 53) % 270
        identifiers.append(f"point-{seed}-{index}")
        latitudes.append(lat + radius * math.cos(angle) / 111_320)
        longitudes.append(
            lon + radius * math.sin(angle) / (111_320 * math.cos(math.radians(lat)))
        )
    return gpd.GeoDataFrame(
        {"id": identifiers},
        geometry=gpd.points_from_xy(longitudes, latitudes),
        crs="EPSG:4326",
    )


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
    assert row.travel_time == 28
    assert row.transfers == 0
    assert row.transit_distance_m == pytest.approx(16_786, abs=1)
    assert row.walk_distance_m == 0.0
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
        output_time_units="seconds",
    )
    # The 08:31 M2 to Kamppi, then the pinned 20-second footpath.
    m2 = matrix[matrix.to_id == "1040280"].iloc[0]
    assert m2.travel_time == 9 * 60 + 20
    assert m2.transfers == 0
    assert m2.transit_distance_m == pytest.approx(4_132, abs=1)
    assert 19 <= m2.walk_distance_m <= 20
    assert m2.emissions == pytest.approx(4.132 * 25, abs=0.1)
    # Reaching the K train takes a second vehicle.
    korso = matrix[matrix.to_id == "1250551"].iloc[0]
    assert korso.transfers >= 1
    assert korso.walk_distance_m > 100


def test_cost_matrix_matches_the_travel_time_matrix(network):
    origins = ["4810551", "1250551"]
    matrix = cost_matrix(
        network,
        origins=origins,
        departure="08:30:00",
        output_time_units="seconds",
    )
    times = network.travel_time_matrix(origins, "2022-02-22 08:30:00")
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
    assert self_row.transit_distance_m == 0.0
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
        output_time_units="seconds",
    )
    assert len(matrix) == 4
    # The door-to-door oracle: M2 plus access and egress walks.
    m2 = matrix[
        (matrix.from_id == "kalasatama") & (matrix.to_id == "kamppi_street")
    ].iloc[0]
    assert 558 <= m2.travel_time <= 562
    assert m2.transfers == 0
    assert m2.transit_distance_m == pytest.approx(4_132, abs=1)
    assert 27 <= m2.walk_distance_m <= 31
    assert m2.emissions == pytest.approx(4.132 * 25, abs=0.1)
    # A destination best reached on foot appears as a pure walk.
    walk = matrix[
        (matrix.from_id == "kamppi_metro") & (matrix.to_id == "kamppi_street")
    ].iloc[0]
    assert 19 <= walk.travel_time <= 21
    assert walk.transfers == 0
    assert walk.transit_distance_m == 0.0
    assert 19 <= walk.walk_distance_m <= 21
    assert walk.emissions == 0.0
    # The point travel-time matrix agrees pair by pair.
    times = network_with_footpaths.travel_time_matrix(
        origins, "2022-02-22 08:30:00", destinations=destinations
    )
    for row in matrix.itertuples():
        origin = list(origins["id"]).index(row.from_id)
        destination = list(destinations["id"]).index(row.to_id)
        assert times[origin, destination] == row.travel_time


def test_point_matrices_from_arbitrary_locations(network_with_footpaths):
    # The typical accessibility workload: travel times from a set of
    # "buildings" to a set of "libraries", none of them at a stop.
    network = network_with_footpaths
    buildings = scattered_points(network, 24, seed=7)
    libraries = scattered_points(network, 6, seed=101)
    times = network.travel_time_matrix(
        buildings, "2022-02-22 08:30:00", destinations=libraries
    )
    assert times.shape == (24, 6)
    # Central locations snap onto the walking network, and within the
    # inner city everything connects by walk-ride-walk or on foot.
    assert (times != UNREACHABLE_POINT).all()
    # The long-format cost matrix agrees cell for cell.
    matrix = cost_matrix(
        network,
        origins=buildings,
        destinations=libraries,
        departure="08:30:00",
        output_time_units="seconds",
    )
    assert len(matrix) == 24 * 6
    building_order = list(buildings["id"])
    library_order = list(libraries["id"])
    for row in matrix.itertuples():
        at = building_order.index(row.from_id), library_order.index(row.to_id)
        assert times[at] == row.travel_time
    # Sampled cells equal the per-pair door-to-door routing: the matrix
    # is the bulk face of route_between_coordinates.
    departure = 8 * 3600 + 30 * 60
    for i in range(0, 24, 7):
        for j in range(0, 6, 2):
            origin = (buildings.geometry.y.iloc[i], buildings.geometry.x.iloc[i])
            destination = (libraries.geometry.y.iloc[j], libraries.geometry.x.iloc[j])
            journeys = network.route_between_coordinates(
                origin, destination, "2022-02-22 08:30:00"
            )
            assert journeys
            fastest = min(journey["arrival_s"] for journey in journeys)
            assert fastest - departure == times[i, j]
    # A building off the street network warns and yields no rows.
    nowhere = gpd.GeoDataFrame(
        {"id": ["sea"]},
        geometry=gpd.points_from_xy([24.90], [60.10]),
        crs="EPSG:4326",
    )
    with pytest.warns(UserWarning, match="off the walking network"):
        stranded = TravelCostMatrix(network, nowhere, libraries, "2022-02-22 08:30:00")
    assert stranded.empty
    # Open water south of the extract: the point cannot snap on the
    # wide surface either.
    unsnapped_destinations = point_frame(network, [("kamppi", "1040280")])
    with pytest.warns(UserWarning, match="off the walking network"):
        times = network.travel_time_matrix(
            nowhere, "2022-02-22 08:30:00", destinations=unsnapped_destinations
        )
    assert times.shape == (1, 1)
    assert times[0, 0] == np.uint32(0xFFFFFFFF)


@pytest.mark.parametrize("optimize", ["emissions", "fare"])
def test_least_cost_cells_match_the_frontier(tmp_path, optimize):
    from cafein import TransportNetwork, journey_frontier, least_emissions, least_fare
    from test_frontier import build_two_line_gtfs, two_line_fares

    feed = build_two_line_gtfs(tmp_path / "two_line_gtfs.zip")
    network = TransportNetwork.from_gtfs([str(feed)])
    fares = two_line_fares() if optimize == "fare" else None
    frontier_kwargs = dict(departure_time_window=30, output_time_units="seconds")
    if fares is not None:
        frontier_kwargs["fares"] = fares
    frontier = journey_frontier(
        network, "A", "B", "2022-02-22 08:00:00", **frontier_kwargs
    )

    def cell(max_travel_time=None):
        kwargs = dict(
            output_time_units="seconds",
            optimize=optimize,
            departure_time_window=30,
            max_travel_time=max_travel_time,
        )
        if fares is not None:
            kwargs["fares"] = fares
        matrix = TravelCostMatrix(
            network, ["A"], ["B"], "2022-02-22 08:00:00", **kwargs
        )
        rows = matrix[matrix.to_id == "B"]
        return rows.iloc[0] if len(rows) else None

    if optimize == "emissions":
        # The matrix cell is the frontier's least-emission pick, cell for
        # cell: the slow-clean tram unbudgeted, the fast-dirty bus chain
        # within 15 minutes, nothing within a minute.
        cleanest, oracle = cell(), least_emissions(frontier)
        assert cleanest["emissions"] == pytest.approx(oracle["emissions"])
        assert cleanest["travel_time"] == oracle["travel_time"] == 1800
        assert cleanest["transfers"] == 0
        # The selector's budget is minutes too, whatever the frame's
        # output units.
        budgeted, oracle = cell(max_travel_time=15), least_emissions(
            frontier, max_travel_time=15
        )
        assert budgeted["emissions"] == pytest.approx(oracle["emissions"])
        assert budgeted["travel_time"] == oracle["travel_time"] == 900
        assert budgeted["transfers"] == 1
    else:
        # The matrix cell is the frontier's cheapest pick, budget for
        # budget: the tram unbudgeted, then the out-of-allowance bus chain,
        # then the fast chain at the pair total, nothing within a minute.
        # Both budgets are minutes, whatever the frame's output units.
        for seconds in (None, 1380, 900):
            limit = None if seconds is None else seconds / 60
            row, oracle = cell(limit), least_fare(frontier, max_travel_time=limit)
            assert row["fare"] == pytest.approx(oracle["fare"])
            assert row["travel_time"] == oracle["travel_time"]
    assert cell(max_travel_time=1) is None


def test_pareto_candidates_lower_the_emission_cells(network):
    from cafein import TravelCostMatrix, exhaustive_frontier

    # The measured blind spot (see test_frontier): the interim
    # emissions cell picks among time-optimal journeys only; the pareto
    # candidates hold the strictly cleaner slower one.
    origin, destination = "1370104", "4960238"

    def cell(**kwargs):
        matrix = TravelCostMatrix(
            network,
            [origin],
            [destination],
            "2022-02-22 08:30:00",
            output_time_units="seconds",
            optimize="emissions",
            departure_time_window=0.016666666666666666,
            max_rides=5,
            **kwargs,
        )
        rows = matrix[matrix.to_id == destination]
        return rows.iloc[0] if len(rows) else None

    interim, pareto = cell(), cell(candidates="pareto", bucket=1e-6)
    assert pareto["emissions"] < interim["emissions"]
    # The pareto cell is the true optimum the oracle pins, at that
    # point's own travel time.
    true_set = exhaustive_frontier(
        network,
        origin,
        destination,
        "2022-02-22 08:30:00",
        max_rides=5,
        output_time_units="seconds",
    )
    point = true_set.loc[true_set["emissions"].idxmin()]
    assert pareto["emissions"] == pytest.approx(point["emissions"], abs=1e-3)
    assert pareto["travel_time"] == point["travel_time"]
    # The default bucket keeps the gap closed.
    assert cell(candidates="pareto")["emissions"] < interim["emissions"]


def test_the_tbtr_pareto_matrix_matches_mcraptor(network):
    from cafein import TravelCostMatrix

    # The measured gap pair, cell for cell between the two engines at
    # a vanishing bucket.
    origin, destination = "1370104", "4960238"
    cells = [
        TravelCostMatrix(
            network,
            [origin],
            [destination],
            "2022-02-22 08:30:00",
            optimize="emissions",
            departure_time_window=0.016666666666666666,
            max_rides=5,
            candidates="pareto",
            bucket=1e-6,
            router=router,
        ).iloc[0]
        for router in ("raptor", "tbtr")
    ]
    for column in [
        "travel_time",
        "transfers",
        "transit_distance_m",
        "walk_distance_m",
    ]:
        assert cells[0][column] == cells[1][column]
    assert cells[0]["emissions"] == pytest.approx(cells[1]["emissions"], abs=1e-6)
    # Time candidates ride the trip-based engine too, with identical rows.
    time_cells = [
        TravelCostMatrix(
            network,
            [origin],
            [destination],
            "2022-02-22 08:30:00",
            optimize="emissions",
            departure_time_window=0.016666666666666666,
            max_rides=5,
            router=router,
        )
        for router in ("raptor", "tbtr")
    ]
    assert len(time_cells[0]) > 0
    assert time_cells[1].equals(time_cells[0])


def test_cost_matrix_options_are_validated(network, helsinki_gtfs):
    from cafein import fares as fare_module

    hsl = fare_module.zone_fare_structure(helsinki_gtfs, rules="zones")
    points = scattered_points(network, 2, seed=7)
    shared = dict(departure="2022-02-22 08:30:00")
    for origins, destinations, kwargs, match in (
        (
            ["1370104"],
            ["4960238"],
            dict(
                optimize="emissions",
                departure_time_window=0.016666666666666666,
                candidates="fastest",
            ),
            "candidates",
        ),
        (["1370104"], ["4960238"], dict(candidates="pareto"), "optimize='emissions'"),
        (
            ["1370104"],
            ["4960238"],
            dict(
                optimize="emissions",
                departure_time_window=0.016666666666666666,
                candidates="pareto",
                bucket=0.0,
            ),
            "bucket",
        ),
        # The multicriteria candidates serve stop origins only.
        (
            points,
            points,
            dict(
                optimize="emissions",
                departure_time_window=0.016666666666666666,
                candidates="pareto",
            ),
            "stop origins",
        ),
        (
            ["4810551"],
            None,
            dict(optimize="fare", departure_time_window=10),
            "requires a fare structure",
        ),
        (
            ["4810551"],
            None,
            dict(optimize="fare", fares=hsl),
            "requires departure_time_window",
        ),
        (
            ["4810551"],
            None,
            dict(optimize="emissions"),
            "requires departure_time_window",
        ),
        (["4810551"], None, dict(max_travel_time=10), "optimize='emissions'"),
        (["4810551"], None, dict(optimize="fastest"), "optimize must be"),
    ):
        with pytest.raises(ValueError, match=match):
            TravelCostMatrix(network, origins, destinations, **shared, **kwargs)
    # A time matrix without any time axis is a TypeError.
    with pytest.raises(TypeError, match="requires departure"):
        TravelTimeMatrix(network, ["4810551"])
    # The exact zone-fare engine serves no exclusions.
    with pytest.raises(ValueError, match="exclusions"):
        cost_matrix(
            network,
            origins=["1040601"],
            destinations=["1121601"],
            departure="08:30:00",
            optimize="fare",
            departure_time_window=10,
            fares=hsl,
            exclude_stops=["1121601"],
        )
    # Without a fare structure no fare column appears.
    plain = cost_matrix(
        network, origins=["4810551"], destinations=["1250551"], departure="08:30:00"
    )
    assert "fare" not in plain.columns


@pytest.mark.parametrize("optimize", ["emissions", "fare"])
def test_point_cells_prefer_walking(network_with_footpaths, helsinki_gtfs, optimize):
    from cafein import fares as fare_module

    kwargs = dict(departure_time_window=10)
    if optimize == "fare":
        kwargs["fares"] = fare_module.zone_fare_structure(helsinki_gtfs, rules="zones")
    matrix = cost_matrix(
        network_with_footpaths,
        origins=point_frame(network_with_footpaths, [("metro", "1040602")]),
        destinations=point_frame(network_with_footpaths, [("street", "1040280")]),
        departure="08:30:00",
        output_time_units="seconds",
        optimize=optimize,
        **kwargs,
    )
    row = matrix.iloc[0]
    if optimize == "emissions":
        assert row.emissions == 0.0
        assert 19 <= row.walk_distance_m <= 21
    else:
        assert row.fare == 0.0
    assert row.transfers == 0
    assert row.transit_distance_m == 0.0
    assert 19 <= row.travel_time <= 21


def test_point_emission_cells_match_the_frontier(network_with_footpaths):
    from cafein import journey_frontier, least_emissions

    coordinates = {stop: (lat, lon) for stop, lat, lon in network_with_footpaths.stops}
    frontier = journey_frontier(
        network_with_footpaths,
        coordinates["1100602"],
        coordinates["1040280"],
        "2022-02-22 08:30:00",
        output_time_units="seconds",
        departure_time_window=10,
    )
    # A budget below the walking time forces a ride, so the cell must
    # equal the frontier's budgeted least-emission journey exactly.
    oracle = least_emissions(frontier, max_travel_time=15)
    assert oracle["rides"] >= 1
    matrix = cost_matrix(
        network_with_footpaths,
        origins=point_frame(network_with_footpaths, [("a", "1100602")]),
        destinations=point_frame(network_with_footpaths, [("b", "1040280")]),
        departure="08:30:00",
        output_time_units="seconds",
        optimize="emissions",
        departure_time_window=10,
        max_travel_time=15,
    )
    row = matrix.iloc[0]
    assert row.emissions == pytest.approx(oracle["emissions"])
    assert row.travel_time == oracle["travel_time"]
    assert row.transfers == oracle["rides"] - 1


def test_fare_columns_price_the_reported_journeys(network, helsinki_gtfs):
    from cafein import fares as fare_module

    hsl = fare_module.zone_fare_structure(helsinki_gtfs, rules="zones")
    matrix = cost_matrix(
        network,
        origins=["4810551"],
        destinations=["1250551"],
        departure="08:30:00",
        output_time_units="seconds",
        fares=hsl,
    )
    row = matrix.iloc[0]
    # Korso (C) → Käpylä (A) prices at the ABC ticket, and the matrix
    # price equals the routed journey's python-side price.
    assert row.fare == pytest.approx(4.1)
    journeys = network.route_between_stops("4810551", "1250551", "2022-02-22 08:30:00")
    fare_module.annotate_fares(journeys, hsl)
    fastest = min(journeys, key=lambda journey: journey["arrival_s"])
    assert row.fare == pytest.approx(fastest["fare"])
    # A seeded rule-based structure prices per boarding: the base fare,
    # one discounted transfer at the pair total (= base), then full
    # fares — so a cell's fare follows its transfer count exactly.
    seeded = fare_module.setup_fare_structure(network, base_fare=3.0)
    bulk = cost_matrix(
        network,
        origins=["4810551", "1040602", "1250551"],
        departure="08:30:00",
        output_time_units="seconds",
        fares=seeded,
    )
    expected = np.where(
        bulk["travel_time"] == 0, 0.0, np.maximum(bulk["transfers"], 1) * 3.0
    )
    assert bulk["fare"].to_numpy() == pytest.approx(expected)


def test_fare_cells_survive_unresolved_emissions(network, helsinki_gtfs):
    from cafein import fares as fare_module

    hsl = fare_module.zone_fare_structure(helsinki_gtfs, rules="zones")
    # Each objective qualifies by its own key: the factorless ferry to
    # Suomenlinna prices at the zone ticket under the fare objective,
    # while the emissions objective has no qualifying candidate.
    cheapest = cost_matrix(
        network,
        origins=["1080701"],
        destinations=["1520703"],
        departure="10:00:00",
        optimize="fare",
        departure_time_window=60,
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
        departure_time_window=60,
        fares=hsl,
    )
    assert cleanest.empty


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
        departure_time_window=10,
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
        departure_time_window=10,
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
        origins, "2022-02-22 07:30:00", destinations=destinations
    )
    assert times[0, 0] in (100, 101)
    # A walk is departure-independent: every percentile plane holds it.
    windowed = network.travel_time_matrix(
        origins,
        "2022-02-22 07:30:00",
        destinations=destinations,
        departure_time_window=10,
        confidence=0.8,
    )
    assert set(windowed[0, 0, :].tolist()) == {times[0, 0]}
    # The cost matrix reports the same walking-only pair: no rides, no
    # transit distance, no emissions, the walk as the distance.
    matrix = TravelCostMatrix(
        network,
        origins,
        destinations,
        "2022-02-22 07:30:00",
        geometries=True,
        output_time_units="seconds",
    )
    row = matrix.iloc[0]
    assert row["travel_time"] == times[0, 0]
    assert row["transfers"] == 0
    assert row["transit_distance_m"] == 0.0
    assert row["walk_distance_m"] == pytest.approx(100.0, abs=0.5)
    assert row["emissions"] == 0.0
    assert row["geometry"].geom_type == "MultiLineString"


def test_stop_matrices_gate_walking_options(network):
    # Under a whole-day shortcut set the stop matrices route door-to-door, so the
    # travel-time and cost matrices *accept* the walking options for stop origins
    # (they bound that routing and are ignored without a set): time-optimal under
    # ULTRA, emissions under McULTRA. Stop origins still cannot mix with point
    # destinations.
    accepted = cost_matrix(
        network,
        origins=["4810551"],
        departure="08:30:00",
        max_walking_time=5.0,
    )
    assert len(accepted) >= 1
    accepted_emissions = cost_matrix(
        network,
        origins=["4810551"],
        departure="08:30:00",
        optimize="emissions",
        departure_time_window=30,
        walking_speed_kmph=5.0,
    )
    assert len(accepted_emissions) >= 1
    with pytest.raises(ValueError, match="point origins"):
        network.travel_time_matrix(
            ["4810551"],
            "2022-02-22 08:30:00",
            destinations=point_frame(network, [("d", "4810551")]),
        )
    matrix = network.travel_time_matrix(
        ["4810551"], "2022-02-22 08:30:00", max_walking_time=5.0
    )
    assert matrix.shape == (1, network.stop_count)


def nearest_rank(sorted_samples, percentile):
    """The core's half-up nearest-rank convention."""
    position = percentile / 100 * (len(sorted_samples) - 1)
    return sorted_samples[int(position + 0.5)]


def test_window_percentiles_match_per_minute_runs(network):
    window = 1800
    percentiles = [0.0, 50.0, 100.0]
    matrix = network.travel_time_matrix(
        ["4810551"],
        "2022-02-22 08:30:00",
        departure_time_window=window / 60,
        percentiles=percentiles,
    )
    assert matrix.shape == (1, network.stop_count, 3)
    stop_order = [stop for stop, _, _ in network.stops]
    per_minute = []
    for step in range(window // 60):
        mark = 8 * 3600 + 30 * 60 + 60 * step
        clock = f"{mark // 3600:02d}:{mark % 3600 // 60:02d}:00"
        per_minute.append(
            network.travel_times_from_stop("4810551", f"2022-02-22 {clock}")
        )
    unreachable = int(np.uint32(0xFFFFFFFF))
    for column, stop in enumerate(stop_order):
        samples = sorted(times.get(stop, unreachable) for times in per_minute)
        for plane, percentile in enumerate(percentiles):
            assert matrix[0, column, plane] == nearest_rank(samples, percentile), (
                stop,
                percentile,
            )


@pytest.mark.parametrize(
    "window, confidence, percentiles",
    [
        pytest.param(30, 0.8, [10, 50, 90], id="symmetric-interval"),
        # 31 samples put the 5th percentile's half-up rank exactly on a
        # tie: (1 - 0.9) / 2 * 100 must reach the core as 5, not
        # 4.999999999999999.
        pytest.param(31, 0.9, [5, 50, 95], id="decimal-rank-tie"),
    ],
)
def test_confidence_maps_to_the_symmetric_interval(
    network, window, confidence, percentiles
):
    left = network.travel_time_matrix(
        ["4810551"],
        "2022-02-22 08:30:00",
        departure_time_window=window,
        confidence=confidence,
    )
    right = network.travel_time_matrix(
        ["4810551"],
        "2022-02-22 08:30:00",
        departure_time_window=window,
        percentiles=percentiles,
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
        "2022-02-22 08:30:00",
        destinations=destinations,
        departure_time_window=30,
        percentiles=[0, 50, 100],
    )
    assert matrix.shape == (1, 1, 3)
    assert 19 <= matrix[0, 0, 0] == matrix[0, 0, 1] == matrix[0, 0, 2] <= 21


def test_wide_matrix_specifications_are_validated(network):
    for kwargs, match in (
        (dict(percentiles=[50]), "require departure_time_window"),
        (
            dict(departure_time_window=10, percentiles=[50], confidence=0.8),
            "not both",
        ),
        (dict(departure_time_window=10, confidence=1.5), "within"),
        (dict(departure_time_window=10, percentiles=[120]), "within"),
        (dict(chunk=(3, 3)), "chunk"),
        (dict(chunk=(-1, 3)), "chunk"),
        (dict(chunk=(0, 0)), "chunk"),
    ):
        with pytest.raises(ValueError, match=match):
            network.travel_time_matrix(["4810551"], "2022-02-22 08:30:00", **kwargs)


def test_chunks_partition_the_matrix(network):
    import pandas as pd

    origins = [stop for stop, _, _ in network.stops[:10]]
    full = cost_matrix(
        network,
        origins=origins,
        destinations=["1250551"],
        departure="08:30:00",
        output_time_units="seconds",
    )
    parts = [
        cost_matrix(
            network,
            origins=origins,
            destinations=["1250551"],
            departure="08:30:00",
            output_time_units="seconds",
            chunk=(part, 3),
        )
        for part in range(3)
    ]
    assert pd.concat(parts, ignore_index=True).equals(pd.DataFrame(full))

    matrix = network.travel_time_matrix(origins, "2022-02-22 08:30:00")
    rows = [
        network.travel_time_matrix(origins, "2022-02-22 08:30:00", chunk=(part, 3))
        for part in range(3)
    ]
    assert np.array_equal(np.vstack(rows), matrix)
    # The long-form travel-time matrix partitions its origins too.
    long_origins = ["4810551", "1250551", "4740551"]
    full_long = TravelTimeMatrix(network, long_origins, departure="2022-02-22 08:30:00")
    parts = [
        TravelTimeMatrix(
            network, long_origins, departure="2022-02-22 08:30:00", chunk=(k, 3)
        )
        for k in range(3)
    ]
    stitched = pd.concat(parts, ignore_index=True)
    assert len(stitched) == len(full_long)
    assert set(map(tuple, stitched[["from_id", "to_id"]].to_numpy())) == set(
        map(tuple, full_long[["from_id", "to_id"]].to_numpy())
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
            departure="2022-02-22 08:30:00",
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
            departure="2022-02-22 08:30:00",
        )


UNREACHABLE = np.iinfo(np.uint32).max


def test_travel_time_matrix_unstacks_the_wide_matrix(network):
    origins = ["4810551", "1250551"]
    wide = network.travel_time_matrix(origins, "2022-02-22 08:30:00")
    stops = [stop for stop, _lat, _lon in network.stops]
    matrix = TravelTimeMatrix(
        network,
        origins,
        departure="2022-02-22 08:30:00",
        output_time_units="seconds",
    )
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


def test_travel_time_matrix_accepts_the_tbtr_router(network):
    origins = ["4810551", "1040602", "1250551"]
    raptor = TravelTimeMatrix(network, origins, departure="2022-02-22 08:30:00")
    tbtr = TravelTimeMatrix(
        network, origins, departure="2022-02-22 08:30:00", router="tbtr"
    )
    assert raptor.equals(tbtr)


def test_tbtr_transfer_cache_reuse_matches_ad_hoc(network, network_with_tbtr):
    # compute_tbtr_transfers precomputes the transfer set once; a router="tbtr"
    # time matrix on that date reuses it (build once, query many) and returns
    # the same cells as the ad-hoc build. The set-less shared network answers
    # ad hoc; the session fixture carries the expensive precomputed set.
    origins = ["4810551", "1040602", "1250551"]
    common = dict(departure="2022-02-22 08:30:00")
    assert not network.has_tbtr_transfers
    ad_hoc = TravelTimeMatrix(network, origins, router="tbtr", **common)

    assert network_with_tbtr.has_tbtr_transfers
    cached = TravelTimeMatrix(network_with_tbtr, origins, router="tbtr", **common)
    assert cached.equals(ad_hoc)

    # A query on a different date ignores the cache (ad-hoc build), still correct.
    other = TravelTimeMatrix(
        network_with_tbtr,
        origins,
        departure="2022-02-23 08:30:00",
        router="tbtr",
    )
    other_raptor = TravelTimeMatrix(
        network_with_tbtr, origins, departure="2022-02-23 08:30:00"
    )
    assert other.equals(other_raptor)


def test_travel_time_matrix_windowed_percentiles(network):
    origins = ["4740551"]
    percentiles = [10, 50, 90]
    wide = network.travel_time_matrix(
        origins,
        "2022-02-22 08:00:00",
        departure_time_window=30,
        percentiles=percentiles,
    )
    stops = [stop for stop, _lat, _lon in network.stops]
    matrix = TravelTimeMatrix(
        network,
        origins,
        departure="2022-02-22 08:00:00",
        output_time_units="seconds",
        departure_time_window=30,
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


def test_windowed_tbtr_matches_raptor_and_reuses_cache(network, network_with_tbtr):
    # A windowed router="tbtr" stop matrix returns the same percentile cells
    # as RAPTOR, ad hoc and off the compute_tbtr_transfers cache alike. The
    # set-less shared network answers ad hoc; the session fixture carries the
    # expensive precomputed set.
    origins = ["4810551", "1040602", "1250551"]
    common = dict(
        departure="2022-02-22 08:00:00",
        departure_time_window=30,
        percentiles=[10, 50, 90],
    )
    raptor = TravelTimeMatrix(network, origins, **common)
    ad_hoc = TravelTimeMatrix(network, origins, router="tbtr", **common)
    assert ad_hoc.equals(raptor)
    cached = TravelTimeMatrix(network_with_tbtr, origins, router="tbtr", **common)
    assert cached.equals(ad_hoc)


def test_tbtr_point_matrices_match_raptor(fresh_footpaths_network):
    # Door-to-door coordinate matrices agree cell for cell between the two
    # engines — single departure and windowed percentiles, ad-hoc and cached
    # transfer set alike.
    origins = point_frame(fresh_footpaths_network, [("A", "1100602"), ("B", "1250551")])
    destinations = point_frame(
        fresh_footpaths_network, [("C", "1040280"), ("D", "4810551"), ("E", "1100602")]
    )
    single = dict(departure="2022-02-22 08:30:00")
    windowed = dict(single, departure_time_window=10, percentiles=[10, 50, 90])
    for common in (single, windowed):
        raptor = TravelTimeMatrix(
            fresh_footpaths_network, origins, destinations, **common
        )
        ad_hoc = TravelTimeMatrix(
            fresh_footpaths_network, origins, destinations, router="tbtr", **common
        )
        assert ad_hoc.equals(raptor)
    fresh_footpaths_network.compute_tbtr_transfers("2022-02-22")
    for common in (single, windowed):
        cached = TravelTimeMatrix(
            fresh_footpaths_network, origins, destinations, router="tbtr", **common
        )
        assert cached.equals(
            TravelTimeMatrix(fresh_footpaths_network, origins, destinations, **common)
        )


def test_travel_time_matrix_over_points(network_with_footpaths):
    origins = point_frame(network_with_footpaths, [("A", "1100602")])
    destinations = point_frame(
        network_with_footpaths, [("B", "1040280"), ("C", "1250551")]
    )
    wide = network_with_footpaths.travel_time_matrix(
        origins, "2022-02-22 08:30:00", destinations=destinations
    )
    matrix = TravelTimeMatrix(
        network_with_footpaths,
        origins,
        destinations,
        departure="2022-02-22 08:30:00",
        output_time_units="seconds",
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


def test_travel_time_matrix_defaults_to_all_stops(network):
    stops = [stop for stop, _lat, _lon in network.stops]
    # Omitted origins mean every stop; the first origin chunk keeps the
    # all-stops resolution cheap to exercise.
    matrix = TravelTimeMatrix(
        network,
        departure="2022-02-22 08:30:00",
        chunk=(0, network.stop_count),
    )
    assert set(matrix.from_id) == {stops[0]}
    assert set(matrix.to_id) <= set(stops)
    assert len(matrix) > 0


def test_zone_fare_matrix_reconstructs_the_exact_journey(network, helsinki_gtfs):
    from cafein import fares as fare_module
    from cafein.frontier import fare_frontier

    hsl = fare_module.zone_fare_structure(helsinki_gtfs, rules="zones")
    matrix = cost_matrix(
        network,
        origins=["1040601", "4810551"],
        destinations=["1121601", "1250551"],
        departure="08:30:00",
        optimize="fare",
        departure_time_window=10,
        fares=hsl,
        geometries=True,
        output_time_units="seconds",
    )
    # The A-zone metro hop rides an AB single at the tariff minimum —
    # the lower-bound skip keeps the fold's own row for it — and the
    # C-to-A pair warm-seeds the exact engine; full-chain
    # reconstruction of a cell the engine strictly improves is pinned
    # at metro scale, where such cells exist.
    cell = matrix[(matrix.from_id == "1040601") & (matrix.to_id == "1121601")].iloc[0]
    assert cell.fare == pytest.approx(2.80)
    assert cell.travel_time == 300
    assert cell.transfers == 0
    assert cell.transit_distance_m == pytest.approx(3061, abs=1)
    assert cell.walk_distance_m == pytest.approx(0.0)
    assert cell.emissions == pytest.approx(76.53, abs=0.1)
    assert cell.geometry is not None
    frontier = fare_frontier(
        network,
        ["1040601", "4810551"],
        ["1121601", "1250551"],
        "2022-02-22 08:30:00",
        10,
        hsl,
        cutoffs=[5.70],
        output_time_units="seconds",
    )
    for _, row in frontier.iterrows():
        match = matrix[
            (matrix.from_id == row.from_id) & (matrix.to_id == row.to_id)
        ].iloc[0]
        assert match.fare == pytest.approx(row.fare)
        assert match.travel_time == row.travel_time


def test_zone_fare_point_matrix_prices_and_walks(network_with_footpaths, helsinki_gtfs):
    from cafein import fares as fare_module

    network = network_with_footpaths
    hsl = fare_module.zone_fare_structure(helsinki_gtfs, rules="zones")
    # A near-stop pair rides transit at the exact fare — the budget
    # excludes the direct walk, which would otherwise win the cell at
    # fare zero. The same-point diagonal is the direct walk — fare
    # zero, from the street search alone, never a composition of
    # access and egress walks.
    points = point_frame(
        network, [("near_kamppi", "1040280"), ("near_kapyla", "1250551")]
    )
    matrix = cost_matrix(
        network,
        origins=points,
        destinations=points,
        departure="08:30:00",
        optimize="fare",
        departure_time_window=10,
        max_travel_time=30,
        fares=hsl,
    )
    ride = matrix[
        (matrix.from_id == "near_kamppi") & (matrix.to_id == "near_kapyla")
    ].iloc[0]
    assert ride.fare == pytest.approx(2.80)
    assert ride.transit_distance_m > 0
    walk = matrix[
        (matrix.from_id == "near_kamppi") & (matrix.to_id == "near_kamppi")
    ].iloc[0]
    assert walk.fare == pytest.approx(0.0)
    assert walk.transfers == 0
    assert walk.transit_distance_m == pytest.approx(0.0)


def test_zone_fare_matrix_prices_zero_fare_products(network, helsinki_gtfs):
    from cafein import fares as fare_module

    # A fare-free tariff is legal; the exact engine must price the
    # cells at zero rather than skipping refinement outright.
    import pandas as pd

    hsl = fare_module.zone_fare_structure(helsinki_gtfs, rules="zones")
    free = fare_module.ZoneFareStructure(
        pd.DataFrame(
            [
                {
                    "fare_id": "FREE",
                    "price": "0",
                    "currency_type": "EUR",
                    "payment_method": "0",
                    "transfers": float("nan"),
                    "transfer_duration": "10800",
                }
            ]
        ),
        {"FREE": frozenset({"A", "B", "C", "D"})},
        hsl.stop_zones,
    )
    matrix = cost_matrix(
        network,
        origins=["1040601"],
        destinations=["1121601"],
        departure="08:30:00",
        optimize="fare",
        departure_time_window=10,
        fares=free,
        output_time_units="seconds",
    )
    assert matrix.iloc[0].fare == pytest.approx(0.0)
    assert matrix.iloc[0].travel_time == 300


def test_integer_ids_round_trip_with_their_exact_dtype(network):
    import numpy as np
    import pandas as pd

    from cafein import TravelCostMatrix, TravelTimeMatrix

    for dtype in ("int32", "int64", "uint32"):
        frame = TravelTimeMatrix(
            network,
            origins=np.array([4810551], dtype=dtype),
            departure="2022-02-22 08:30:00",
        )
        assert str(frame["from_id"].dtype) == dtype
    nullable = TravelTimeMatrix(
        network,
        origins=pd.Series([4810551], dtype="Int64"),
        departure="2022-02-22 08:30:00",
    )
    assert str(nullable["from_id"].dtype) == "Int64"
    plain = TravelTimeMatrix(
        network, origins=[4810551], departure="2022-02-22 08:30:00"
    )
    assert str(plain["from_id"].dtype) == "int64"
    # The all-stops destination axis keeps the native string GTFS ids.
    assert str(plain["to_id"].dtype) in ("object", "str")
    # A merge against the user's own integer frame needs no casting.
    joined = plain.merge(
        pd.DataFrame({"from_id": pd.array([4810551]), "population": [7]}),
        on="from_id",
    )
    assert len(joined) == len(plain)
    # Strings stay strings; mixed inputs stay strings.
    strings = TravelTimeMatrix(
        network, origins=["4810551"], departure="2022-02-22 08:30:00"
    )
    assert str(strings["from_id"].dtype) in ("object", "str")
    mixed = TravelTimeMatrix(
        network, origins=[4810551, "1250551"], departure="2022-02-22 08:30:00"
    )
    assert str(mixed["from_id"].dtype) in ("object", "str")
    # Both axes cast on the cost matrix when both were integer-typed.
    cost = TravelCostMatrix(
        network,
        origins=[4810551, 1250551],
        destinations=[1250551],
        departure="2022-02-22 08:30:00",
    )
    assert str(cost["from_id"].dtype) == "int64"
    assert str(cost["to_id"].dtype) == "int64"
    # Shard-schema stability outranks dtype round-tripping on the Arrow
    # surfaces: the ids stay dictionary-encoded strings.
    pyarrow = pytest.importorskip("pyarrow")

    from cafein import travel_cost_table

    table = travel_cost_table(
        network, origins=[4810551], departure="2022-02-22 08:30:00"
    )
    assert pyarrow.types.is_dictionary(table.schema.field("from_id").type)
    assert pyarrow.types.is_string(table.schema.field("from_id").type.value_type)


def _slot_frames(network, origins, slots, **kwargs):
    """The scalar frames of ``slots`` (moment strings) in order."""
    return [
        TravelTimeMatrix(network, origins, departure=slot, **kwargs) for slot in slots
    ]


def test_departure_slots_concatenate_the_scalar_frames(network):
    origins = ["4810551", "1250551", "4740551"]
    slots = ["2022-02-22 08:30:00", "2022-02-22 12:00:00"]
    frame = TravelTimeMatrix(network, origins, departure=slots)
    assert list(frame.columns) == ["from_id", "to_id", "departure_time", "travel_time"]
    expected = pd.concat(
        [
            single.assign(departure_time=slot)
            for single, slot in zip(_slot_frames(network, origins, slots), slots)
        ],
        ignore_index=True,
    )[list(frame.columns)]
    pd.testing.assert_frame_equal(pd.DataFrame(frame), expected)
    # A one-element list is a list: the slot column stays.
    one = TravelTimeMatrix(network, origins, departure=slots[:1])
    assert "departure_time" in one.columns
    pd.testing.assert_frame_equal(
        pd.DataFrame(one.drop(columns="departure_time")),
        pd.DataFrame(TravelTimeMatrix(network, origins, departure=slots[0])),
    )


def test_departure_slot_mapping_labels_the_rows(network):
    origins = ["4810551", "1250551"]
    mapping = {"peak": "2022-02-22 08:30:00", "midday": "2022-02-22 12:00:00"}
    frame = TravelTimeMatrix(network, origins, departure=mapping)
    assert list(frame.columns) == [
        "from_id",
        "to_id",
        "slot",
        "departure_time",
        "travel_time",
    ]
    assert list(frame["slot"].unique()) == ["peak", "midday"]
    by_label = dict(zip(frame["slot"], frame["departure_time"]))
    assert by_label == mapping
    unlabeled = TravelTimeMatrix(network, origins, departure=list(mapping.values()))
    pd.testing.assert_frame_equal(
        pd.DataFrame(frame.drop(columns="slot")), pd.DataFrame(unlabeled)
    )


def test_departure_slots_carry_windows_and_percentiles_per_slot(network):
    origins = ["4740551"]
    slots = ["2022-02-22 08:00:00", "2022-02-22 16:00:00"]
    kwargs = dict(
        departure_time_window=20,
        percentiles=[25, 75],
        output_time_units="seconds",
    )
    frame = TravelTimeMatrix(network, origins, departure=slots, **kwargs)
    assert list(frame.columns) == [
        "from_id",
        "to_id",
        "departure_time",
        "travel_time_p25",
        "travel_time_p75",
    ]
    for single, slot in zip(_slot_frames(network, origins, slots, **kwargs), slots):
        block = frame[frame["departure_time"] == slot].drop(columns="departure_time")
        pd.testing.assert_frame_equal(
            block.reset_index(drop=True), pd.DataFrame(single)
        )


def test_departure_slots_route_points_door_to_door(network_with_footpaths):
    origins = point_frame(network_with_footpaths, [("a", "4810551"), ("b", "1250551")])
    slots = ["2022-02-22 08:30:00", "2022-02-22 09:30:00"]
    frame = TravelTimeMatrix(network_with_footpaths, origins, origins, departure=slots)
    assert list(frame.columns) == ["from_id", "to_id", "departure_time", "travel_time"]
    for slot in slots:
        block = frame[frame["departure_time"] == slot].drop(columns="departure_time")
        single = TravelTimeMatrix(
            network_with_footpaths, origins, origins, departure=slot
        )
        pd.testing.assert_frame_equal(
            block.reset_index(drop=True), pd.DataFrame(single)
        )


def test_arrival_slots_mirror_departure_slots(network_with_footpaths):
    # The arrive-by stop arm spans every stop by design, so the slot
    # identity rides the point arm's small axes.
    points = point_frame(network_with_footpaths, [("a", "4810551"), ("b", "1250551")])
    deadlines = ["2022-02-22 09:30:00", "2022-02-22 13:00:00"]
    frame = TravelTimeMatrix(network_with_footpaths, points, points, arrival=deadlines)
    assert list(frame.columns) == ["from_id", "to_id", "arrival_time", "travel_time"]
    for deadline in deadlines:
        block = frame[frame["arrival_time"] == deadline].drop(columns="arrival_time")
        single = TravelTimeMatrix(
            network_with_footpaths, points, points, arrival=deadline
        )
        pd.testing.assert_frame_equal(
            block.reset_index(drop=True), pd.DataFrame(single)
        )


def test_departure_slots_chunk_the_origins(network):
    origins = ["4810551", "1250551", "4740551"]
    slots = ["2022-02-22 08:30:00", "2022-02-22 12:00:00"]
    full = TravelTimeMatrix(network, origins, departure=slots)
    parts = [
        TravelTimeMatrix(network, origins, departure=slots, chunk=(k, 3))
        for k in range(3)
    ]
    # Every chunk carries both slots; the chunks partition the origins.
    for part in parts:
        assert set(part["departure_time"]) <= set(slots)
    stitched = pd.concat(parts, ignore_index=True)
    keys = ["from_id", "to_id", "departure_time"]
    assert len(stitched) == len(full)
    assert set(map(tuple, stitched[keys].to_numpy())) == set(
        map(tuple, full[keys].to_numpy())
    )


def test_departure_slots_validate_eagerly(network, helsinki_streets):
    origins = ["4810551"]
    with pytest.raises(ValueError, match="dated moments"):
        TravelTimeMatrix(network, origins, departure=["08:30:00"])
    with pytest.raises(ValueError, match="names no slots"):
        TravelTimeMatrix(network, origins, departure=[])
    with pytest.raises(ValueError, match="twice"):
        TravelTimeMatrix(network, origins, departure=["2022-02-22 08:30:00"] * 2)
    with pytest.raises(TypeError, match="labels must be strings"):
        TravelTimeMatrix(network, origins, departure={8: "2022-02-22 08:30:00"})
    with pytest.raises(ValueError, match="give exactly one"):
        TravelTimeMatrix(
            network,
            origins,
            departure=["2022-02-22 08:30:00"],
            arrival=["2022-02-22 09:30:00"],
        )
    # A street matrix carries no timetable: slots are as meaningless
    # there as a single departure.
    points = gpd.GeoDataFrame(
        {"id": ["p", "q"]},
        geometry=gpd.points_from_xy([24.9384, 24.9600], [60.1699, 60.1866]),
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="no meaning for a street matrix"):
        TravelTimeMatrix(
            helsinki_streets,
            points,
            points,
            departure=["2022-02-22 08:30:00", "2022-02-22 12:00:00"],
            transport_mode="walk",
        )


def test_matrix_phase_details_count_the_slots():
    from cafein.matrices import _query_details

    slots = ["2022-02-22 08:30:00", "2022-02-22 12:00:00"]
    assert _query_details({"departure": slots})["slots"] == 2
    assert _query_details({"arrival": {"a": slots[0]}})["slots"] == 1
    assert "slots" not in _query_details({"departure": slots[0]})


def test_departure_slots_share_one_frozen_query(network):
    import types

    origins = ["4810551", "1250551"]
    slots = ["2022-02-22 08:00:00", "2022-02-22 16:00:00"]
    # One-shot iterables must feed every slot, not only the first.
    frame = TravelTimeMatrix(
        network,
        origins,
        departure=slots,
        departure_time_window=10,
        percentiles=(p for p in (25, 75)),
        exclude_routes=(route for route in ["2550"]),
        output_time_units="seconds",
    )
    assert list(frame.columns) == [
        "from_id",
        "to_id",
        "departure_time",
        "travel_time_p25",
        "travel_time_p75",
    ]
    for slot in slots:
        block = frame[frame["departure_time"] == slot].drop(columns="departure_time")
        single = TravelTimeMatrix(
            network,
            origins,
            departure=slot,
            departure_time_window=10,
            percentiles=[25, 75],
            exclude_routes=["2550"],
            output_time_units="seconds",
        )
        pd.testing.assert_frame_equal(
            block.reset_index(drop=True), pd.DataFrame(single)
        )
    # Any mapping labels slots, and dates are canonical ISO.
    proxy = types.MappingProxyType({"peak": "2022-02-22T08:00", "late": slots[1]})
    labeled = TravelTimeMatrix(network, origins, departure=proxy)
    assert list(labeled["slot"].unique()) == ["peak", "late"]
    assert set(labeled["departure_time"]) == set(slots)
    with pytest.raises(ValueError, match="twice"):
        TravelTimeMatrix(network, origins, departure=["2022-02-22T08:00", slots[0]])


def _cost_slot_frame(network, origins, departure, **kwargs):
    with pytest.warns(UserWarning, match="route_type"):
        return TravelCostMatrix(network, origins, departure=departure, **kwargs)


def test_cost_departure_slots_concatenate_the_scalar_frames(network):
    origins = ["4810551", "1250551"]
    slots = ["2022-02-22 08:30:00", "2022-02-22 12:00:00"]
    frame = _cost_slot_frame(network, origins, slots)
    assert list(frame.columns)[:4] == [
        "from_id",
        "to_id",
        "departure_time",
        "travel_time",
    ]
    expected = pd.concat(
        [
            _cost_slot_frame(network, origins, slot).assign(departure_time=slot)
            for slot in slots
        ],
        ignore_index=True,
    )[list(frame.columns)]
    pd.testing.assert_frame_equal(pd.DataFrame(frame), expected)
    labeled = _cost_slot_frame(network, origins, dict(zip(["peak", "midday"], slots)))
    assert list(labeled.columns)[:4] == ["from_id", "to_id", "slot", "departure_time"]
    pd.testing.assert_frame_equal(
        pd.DataFrame(labeled.drop(columns="slot")), pd.DataFrame(frame)
    )


def test_cost_arrival_slots_mirror_departure_slots(network):
    # A cost matrix serves arrival= on the emissions axis (with its
    # window); the time axis rides TravelTimeMatrix.
    origins = ["4810551"]
    destinations = ["4810551", "1250551", "4740551"]
    deadlines = ["2022-02-22 09:30:00", "2022-02-22 13:00:00"]
    kwargs = dict(optimize="emissions", arrival_time_window=10)
    with pytest.warns(UserWarning, match="route_type"):
        frame = TravelCostMatrix(
            network, origins, destinations, arrival=deadlines, **kwargs
        )
    assert list(frame.columns)[:3] == ["from_id", "to_id", "arrival_time"]
    for deadline in deadlines:
        block = frame[frame["arrival_time"] == deadline].drop(columns="arrival_time")
        with pytest.warns(UserWarning, match="route_type"):
            single = TravelCostMatrix(
                network, origins, destinations, arrival=deadline, **kwargs
            )
        pd.testing.assert_frame_equal(
            block.reset_index(drop=True), pd.DataFrame(single)
        )


def test_cost_table_slots_concatenate_the_arrow_tables(network):
    pytest.importorskip("pyarrow")
    from cafein.matrices import travel_cost_table

    origins = ["4810551", "1250551"]
    slots = ["2022-02-22 08:30:00", "2022-02-22 12:00:00"]
    with pytest.warns(UserWarning, match="route_type"):
        table = travel_cost_table(network, origins, departure=slots)
    assert table.column_names[:4] == [
        "from_id",
        "to_id",
        "departure_time",
        "travel_time",
    ]
    parts = []
    for slot in slots:
        with pytest.warns(UserWarning, match="route_type"):
            part = travel_cost_table(network, origins, departure=slot).to_pandas()
        parts.append(part.assign(departure_time=slot))
    expected = pd.concat(parts, ignore_index=True)[table.column_names]
    got = table.to_pandas()
    pd.testing.assert_frame_equal(
        got, expected, check_dtype=False, check_categorical=False
    )


def test_cost_slots_validate_eagerly(network, helsinki_streets):
    origins = ["4810551"]
    with pytest.raises(ValueError, match="dated moments"):
        TravelCostMatrix(network, origins, departure=["08:30:00"])
    points = gpd.GeoDataFrame(
        {"id": ["p", "q"]},
        geometry=gpd.points_from_xy([24.9384, 24.9600], [60.1699, 60.1866]),
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="no meaning for a street matrix"):
        TravelCostMatrix(
            helsinki_streets,
            points,
            points,
            departure=["2022-02-22 08:30:00", "2022-02-22 12:00:00"],
            transport_mode="walk",
        )


def test_walking_only_policy_serves_departure_slots(network_with_footpaths):
    from cafein import StreetLegPolicy

    policy = StreetLegPolicy(access={"walk": 15}, egress={"walk": 15})
    origins = point_frame(network_with_footpaths, [("a", "4810551"), ("b", "1250551")])
    slots = ["2022-02-22 08:30:00", "2022-02-22 12:00:00"]
    frame = TravelTimeMatrix(
        network_with_footpaths, origins, origins, departure=slots, street_policy=policy
    )
    for slot in slots:
        block = frame[frame["departure_time"] == slot].drop(columns="departure_time")
        single = TravelTimeMatrix(
            network_with_footpaths,
            origins,
            origins,
            departure=slot,
            street_policy=policy,
        )
        pd.testing.assert_frame_equal(
            block.reset_index(drop=True), pd.DataFrame(single)
        )


def test_walking_only_policy_serves_cost_slots(network_with_footpaths):
    from cafein import StreetLegPolicy, TravelCostMatrix

    policy = StreetLegPolicy(access={"walk": 15}, egress={"walk": 15})
    origins = point_frame(network_with_footpaths, [("a", "4810551"), ("b", "1250551")])
    slots = ["2022-02-22 08:30:00", "2022-02-22 12:00:00"]
    with pytest.warns(UserWarning, match="route_type"):
        frame = TravelCostMatrix(
            network_with_footpaths,
            origins,
            origins,
            departure=slots,
            street_policy=policy,
        )
    for slot in slots:
        block = frame[frame["departure_time"] == slot].drop(columns="departure_time")
        with pytest.warns(UserWarning, match="route_type"):
            single = TravelCostMatrix(
                network_with_footpaths,
                origins,
                origins,
                departure=slot,
                street_policy=policy,
            )
        pd.testing.assert_frame_equal(
            block.reset_index(drop=True), pd.DataFrame(single)
        )


def test_compare_matrices_aligns_cells():
    import numpy as np

    from cafein.matrices import compare_matrices

    a = pd.DataFrame(
        {
            "from_id": pd.Categorical(["o1", "o1", "o2"]),
            "to_id": ["d1", "d2", "d1"],
            "travel_time": [10.0, 20.0, 0.0],
            "note": ["x", "y", "z"],
        }
    )
    b = pd.DataFrame(
        {
            "from_id": ["o1", "o2", "o3"],
            "to_id": ["d1", "d1", "d1"],
            "travel_time": [12.0, 15.0, 9.0],
            "note": ["x", "y", "z"],
        }
    )
    out = compare_matrices(a, b, ratios=True)
    assert list(out.columns) == [
        "from_id",
        "to_id",
        "status",
        "travel_time_a",
        "travel_time_b",
        "travel_time_delta",
        "travel_time_ratio",
    ]
    by_key = {(row.from_id, row.to_id): row for row in out.itertuples(index=False)}
    assert len(by_key) == 4
    assert by_key[("o1", "d1")].status == "both"
    assert by_key[("o1", "d1")].travel_time_delta == 2.0
    assert by_key[("o1", "d1")].travel_time_ratio == 1.2
    assert by_key[("o1", "d2")].status == "only_a"
    assert np.isnan(by_key[("o1", "d2")].travel_time_delta)
    # A zero on side a: the delta stands, the ratio is NaN.
    assert by_key[("o2", "d1")].travel_time_delta == 15.0
    assert np.isnan(by_key[("o2", "d1")].travel_time_ratio)
    assert by_key[("o3", "d1")].status == "only_b"
    # Slot columns join the key when both sides carry them.
    slotted_a = a.assign(departure_time="2022-02-22 08:30:00")
    slotted_b = b.assign(departure_time="2022-02-22 08:30:00")
    slotted = compare_matrices(slotted_a, slotted_b)
    assert list(slotted.columns)[:4] == [
        "from_id",
        "to_id",
        "departure_time",
        "status",
    ]


def test_compare_matrices_refusals():
    from cafein.matrices import compare_matrices

    def frame(**overrides):
        base = {
            "from_id": ["o1", "o2"],
            "to_id": ["d1", "d1"],
            "travel_time": [10.0, 20.0],
        }
        base.update(overrides)
        return pd.DataFrame(base)

    for a, b, kwargs, match in (
        (frame(slot=["p", "q"]), frame(), {}, "key column on side a"),
        (frame(), frame(from_id=["o1", "o1"]), {}, "duplicate"),
        (
            frame(travel_time_p75=[1.0, 2.0]),
            frame(travel_time_p50=[1.0, 2.0]),
            {},
            "travel_time_p75",
        ),
        (frame(emissions=[5.0, 6.0]), frame(), {}, "side a alone carries"),
        (frame(), frame(), {"columns": ["nope"]}, "no nope column"),
        (
            frame(note=["x", "y"]),
            frame(note=["x", "y"]),
            {"columns": ["note"]},
            "not numeric",
        ),
        (frame(from_id=[1, 2]), frame(), {}, "integer keys"),
        (
            frame(travel_time_p75=[1.0, 2.0]),
            frame(),
            {"columns": ["travel_time"]},
            "percentiles",
        ),
        (
            frame(from_id=pd.to_datetime(["2022-02-22", "2022-02-23"])),
            frame(from_id=pd.to_datetime(["2022-02-22", "2022-02-23"])),
            {},
            "integer keys",
        ),
        (frame().drop(columns="from_id"), frame(), {}, "no from_id"),
        (
            frame(from_id=pd.array([1, "o2"], dtype=object)),
            frame(),
            {},
            "non-string values",
        ),
        (frame(from_id=["o1", None]), frame(), {}, "missing values"),
    ):
        with pytest.raises(ValueError, match=match):
            compare_matrices(a, b, **kwargs)
    # The columns= escape compares the shared subset.
    out = compare_matrices(
        frame(emissions=[5.0, 6.0]), frame(), columns=["travel_time"]
    )
    assert list(out["status"]) == ["both", "both"]
    # Integer keys of different widths merge losslessly as Int64.
    import numpy as np

    widths = compare_matrices(
        frame(from_id=np.array([1, 2], dtype="int32")),
        frame(from_id=np.array([1, 2], dtype="uint64")),
    )
    assert list(widths["status"]) == ["both", "both"]
    assert str(widths["from_id"].dtype) == "Int64"
    # Two unsigned sides keep the unsigned domain — large ids survive.
    big = 2**63 + 7
    unsigned = compare_matrices(
        frame(from_id=np.array([big, big + 1], dtype="uint64")),
        frame(from_id=np.array([big, big + 1], dtype="uint64")),
    )
    assert str(unsigned["from_id"].dtype) == "UInt64"
    assert set(unsigned["from_id"]) == {big, big + 1}
    # Nullable numeric columns compare NA-safely.
    held = compare_matrices(
        frame(travel_time=pd.array([10, pd.NA], dtype="Int64")), frame()
    )
    assert held["travel_time_delta"].iloc[0] == 0.0
    assert np.isnan(held["travel_time_delta"].iloc[1])


def test_the_matrix_classes_delegate_compare(network):
    from cafein.matrices import compare_matrices

    origins = ["4810551", "1250551"]
    peak = TravelTimeMatrix(network, origins, departure="2022-02-22 08:30:00")
    midday = TravelTimeMatrix(network, origins, departure="2022-02-22 12:00:00")
    via_method = peak.compare(midday)
    pd.testing.assert_frame_equal(via_method, compare_matrices(peak, midday))
    assert (via_method["status"] == "both").any()
    assert "travel_time_delta" in via_method.columns
    with pytest.warns(UserWarning, match="route_type"):
        cost_peak = TravelCostMatrix(network, origins, departure="2022-02-22 08:30:00")
        cost_midday = TravelCostMatrix(
            network, origins, departure="2022-02-22 12:00:00"
        )
    ratios = cost_peak.compare(cost_midday, ratios=True)
    assert "emissions_ratio" in ratios.columns
    # Slotted matrices align on the moment key: the shared slot joins
    # as both, the disjoint ones as only_a/only_b.
    early = TravelTimeMatrix(
        network, origins, departure=["2022-02-22 08:30:00", "2022-02-22 12:00:00"]
    )
    late = TravelTimeMatrix(
        network, origins, departure=["2022-02-22 12:00:00", "2022-02-22 16:00:00"]
    )
    slotted = compare_matrices(early, late)
    by_status = dict(slotted.groupby("status")["departure_time"].unique())
    assert list(by_status["both"]) == ["2022-02-22 12:00:00"]
    assert list(by_status["only_a"]) == ["2022-02-22 08:30:00"]
    assert list(by_status["only_b"]) == ["2022-02-22 16:00:00"]


def test_slots_span_service_dates_on_both_axes(network, network_with_footpaths):
    # Two service dates group into two core calls on the departure
    # axis; the arrival axis loops per slot. Both must equal scalars.
    origins = ["4810551", "1250551", "4740551"]
    two_dates = [
        "2022-02-22 08:30:00",
        "2022-02-22 12:00:00",
        "2022-02-23 08:30:00",
    ]
    resolutions = []
    original = type(network)._time_matrix_with_ids

    def counting(self, from_stops, date, departure, *args, **kwargs):
        resolutions.append((date, departure))
        return original(self, from_stops, date, departure, *args, **kwargs)

    type(network)._time_matrix_with_ids = counting
    try:
        frame = TravelTimeMatrix(network, origins, departure=two_dates)
    finally:
        type(network)._time_matrix_with_ids = original
    # One grouped core resolution per service date, each carrying its
    # date's clock list — not one per slot.
    assert [(date, clocks) for date, clocks in resolutions] == [
        ("2022-02-22", ["08:30:00", "12:00:00"]),
        ("2022-02-23", ["08:30:00"]),
    ]
    expected = pd.concat(
        [
            pd.DataFrame(TravelTimeMatrix(network, origins, departure=slot)).assign(
                departure_time=slot
            )
            for slot in two_dates
        ],
        ignore_index=True,
    )[list(frame.columns)]
    pd.testing.assert_frame_equal(pd.DataFrame(frame), expected)
    with pytest.warns(UserWarning, match="route_type"):
        cost = TravelCostMatrix(network, origins[:2], departure=two_dates)
    with pytest.warns(UserWarning, match="route_type"):
        cost_expected = pd.concat(
            [
                pd.DataFrame(
                    TravelCostMatrix(network, origins[:2], departure=slot)
                ).assign(departure_time=slot)
                for slot in two_dates
            ],
            ignore_index=True,
        )[list(cost.columns)]
    pd.testing.assert_frame_equal(pd.DataFrame(cost), cost_expected)
    points = point_frame(network_with_footpaths, [("a", "4810551"), ("b", "1250551")])
    deadlines = ["2022-02-22 09:30:00", "2022-02-23 09:30:00"]
    reverse = TravelTimeMatrix(
        network_with_footpaths, points, points, arrival=deadlines
    )
    for deadline in deadlines:
        block = reverse[reverse["arrival_time"] == deadline].drop(
            columns="arrival_time"
        )
        single = TravelTimeMatrix(
            network_with_footpaths, points, points, arrival=deadline
        )
        pd.testing.assert_frame_equal(
            block.reset_index(drop=True), pd.DataFrame(single)
        )
