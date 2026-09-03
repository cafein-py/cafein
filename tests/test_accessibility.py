"""The accessibility primitive: per-origin opportunity sums (core)."""

import numpy
import pytest

pytest.importorskip("cafein._cafein")

DATE = "2022-02-22"
DEPARTURE = "2022-02-22 08:30:00"
DEPARTURE_TIME = "08:30:00"
UNREACHED = numpy.iinfo("uint32").max


def _stop_sets(network):
    # Served city stops: the head of the id-sorted list is an
    # unserved-station block.
    stops = [stop for stop, lat, lon in network.stops if lat is not None]
    return stops[1000:1005], stops[1000:1050]


def _oracle_times(network, origins, destinations):
    matrix = network._core.travel_time_matrix(
        origins, DATE, DEPARTURE_TIME, 7, "auto", [], [], [], 5.0, 1800.0, 500.0
    )
    column = {stop: at for at, (stop, _, _) in enumerate(network.stops)}
    times = matrix[:, [column[stop] for stop in destinations]].astype("float64")
    times[times == UNREACHED] = numpy.nan
    return times


def _sums(network, origins, destinations, opportunities, budgets, decay, param):
    return network._core._accessibility_from_stops(
        origins,
        destinations,
        list(opportunities),
        1,
        list(budgets),
        decay,
        param,
        DATE,
        DEPARTURE_TIME,
        7,
        "auto",
        [],
        [],
        [],
        5.0,
        1800.0,
        500.0,
    )


def test_step_sums_equal_the_matrix_hand_count(network):
    origins, destinations = _stop_sets(network)
    opportunities = numpy.arange(1.0, len(destinations) + 1)
    times = _oracle_times(network, origins, destinations)
    values = _sums(
        network, origins, destinations, opportunities, (900.0, 1800.0), "step", None
    )
    oracle = numpy.vstack(
        [
            numpy.nansum(numpy.where(times <= budget, opportunities, 0.0), axis=1)
            for budget in (900.0, 1800.0)
        ]
    ).T
    assert values.shape == (len(origins), 2)
    assert numpy.allclose(values, oracle)
    # Monotone non-decreasing in budget.
    assert (values[:, 1] >= values[:, 0]).all()


def test_decay_weights_match_the_formulas(network):
    origins, destinations = _stop_sets(network)
    opportunities = numpy.arange(1.0, len(destinations) + 1)
    times = _oracle_times(network, origins, destinations)

    def masked(weights, support):
        return numpy.nansum(
            numpy.where(times <= support, weights, 0.0) * opportunities,
            axis=1,
            keepdims=True,
        )

    # (decay, parameter): (weights, the support — the ramp keeps its
    # weight to budget + width/2, every other family ends at the budget).
    cases = {
        ("linear", 600.0): (
            numpy.clip((1800.0 + 300.0 - times) / 600.0, 0.0, 1.0),
            2100.0,
        ),
        ("linear_cutoff", None): (numpy.clip(1.0 - times / 1800.0, 0.0, 1.0), 1800.0),
        ("exponential", 600.0): (numpy.exp(-numpy.log(2.0) * times / 600.0), 1800.0),
        ("logistic", 120.0): (
            1.0 / (1.0 + numpy.exp((times - 1800.0) / 120.0)),
            1800.0,
        ),
    }
    for (decay, param), (weights, support) in cases.items():
        values = _sums(
            network, origins, destinations, opportunities, (1800.0,), decay, param
        )
        assert numpy.allclose(values, masked(weights, support)), decay


def test_a_generous_budget_counts_the_reachable_mass(network):
    origins, destinations = _stop_sets(network)
    opportunities = numpy.ones(len(destinations))
    times = _oracle_times(network, origins, destinations)
    values = _sums(
        network, origins, destinations, opportunities, (10_000_000.0,), "step", None
    )
    reachable = numpy.isfinite(times).sum(axis=1, keepdims=True).astype("float64")
    assert numpy.allclose(values, reachable)


def test_street_sums_match_the_street_matrix(helsinki_streets):
    origins = [(60.1699, 24.9384), (60.1866, 24.9600)]
    destinations = [
        (60.1725, 24.9414),
        (60.1798, 24.9509),
        (60.1580, 24.9210),
        (60.2000, 24.9700),
    ]
    table = helsinki_streets._core.travel_time_matrix(
        origins, destinations, "walk", 7200.0, 500.0, None
    )
    times = table["matrix"].astype("float64")
    times[times == UNREACHED] = numpy.nan
    opportunities = numpy.array([5.0, 7.0, 11.0, 13.0])
    result = helsinki_streets._core._accessibility_to_points(
        origins,
        destinations,
        list(opportunities),
        1,
        [900.0, 1800.0],
        "step",
        None,
        "walk",
        7200.0,
        500.0,
    )
    oracle = numpy.vstack(
        [
            numpy.nansum(numpy.where(times <= budget, opportunities, 0.0), axis=1)
            for budget in (900.0, 1800.0)
        ]
    ).T
    assert numpy.allclose(result["values"], oracle)
    assert list(result["unsnapped_from"]) == []
    assert list(result["unsnapped_to"]) == []


def test_the_aggregation_arguments_validate_eagerly(network):
    origins, destinations = _stop_sets(network)
    opportunities = list(numpy.ones(len(destinations)))

    def call(**overrides):
        arguments = {
            "opportunities": opportunities,
            "budgets": [1800.0],
            "decay": "step",
            "param": None,
        }
        arguments.update(overrides)
        return _sums(
            network,
            origins,
            destinations,
            arguments["opportunities"],
            arguments["budgets"],
            arguments["decay"],
            arguments["param"],
        )

    with pytest.raises(ValueError, match="unknown decay"):
        call(decay="gravity")
    with pytest.raises(ValueError, match="decay_param"):
        call(decay="linear")
    with pytest.raises(ValueError, match="positive finite"):
        call(decay="exponential", param=-1.0)
    with pytest.raises(ValueError, match="no decay_param"):
        call(decay="step", param=3.0)
    with pytest.raises(ValueError, match="budgets"):
        call(budgets=[])
    with pytest.raises(ValueError, match="budgets"):
        call(budgets=[float("inf")])
    with pytest.raises(ValueError, match="opportunities carries"):
        call(opportunities=opportunities[:-1])
    with pytest.raises(ValueError, match="finite and non-negative"):
        call(opportunities=[-1.0] + opportunities[1:])


def test_the_computer_matches_the_stop_oracle(network):
    import pandas

    from cafein import Accessibility

    origins, destinations = _stop_sets(network)
    table = pandas.DataFrame(
        {"id": destinations, "jobs": numpy.arange(1.0, len(destinations) + 1)}
    )
    frame = Accessibility(
        network,
        origins,
        table,
        DEPARTURE,
        opportunities="jobs",
        budgets=(15.0, 30.0),
    )
    assert list(frame.columns) == ["from_id", "opportunity", "budget", "accessibility"]
    assert len(frame) == len(origins) * 2
    assert set(frame["opportunity"]) == {"jobs"}
    times = _oracle_times(network, origins, destinations)
    for budget in (15.0, 30.0):
        expected = numpy.nansum(
            numpy.where(times <= budget * 60, table["jobs"].to_numpy(), 0.0), axis=1
        )
        got = frame[frame["budget"] == budget]["accessibility"].to_numpy()
        assert numpy.allclose(got, expected)


def test_the_computer_counts_features_without_opportunities(network):
    from cafein import Accessibility

    origins, destinations = _stop_sets(network)
    frame = Accessibility(network, origins, destinations, DEPARTURE)
    assert set(frame["opportunity"]) == {"count"}
    times = _oracle_times(network, origins, destinations)
    expected = (times <= 1800.0).sum(axis=1).astype("float64")
    assert numpy.allclose(frame["accessibility"].to_numpy(), expected)


def test_the_computer_routes_point_frames_door_to_door(network_with_footpaths):
    geopandas = pytest.importorskip("geopandas")
    from cafein import Accessibility, TravelTimeMatrix

    origins = geopandas.GeoDataFrame(
        {"id": ["kamppi", "kallio"]},
        geometry=geopandas.points_from_xy([24.9316, 24.9500], [60.1688, 60.1841]),
        crs="EPSG:4326",
    )
    destinations = geopandas.GeoDataFrame(
        {"id": ["a", "b", "c"], "seats": [10.0, 20.0, 40.0]},
        geometry=geopandas.points_from_xy(
            [24.9414, 24.9210, 24.9700], [60.1725, 60.1580, 60.2000]
        ),
        crs="EPSG:4326",
    )
    frame = Accessibility(
        network_with_footpaths,
        origins,
        destinations,
        DEPARTURE,
        opportunities="seats",
        budgets=(30.0,),
    )
    matrix = TravelTimeMatrix(
        network_with_footpaths,
        origins,
        destinations,
        DEPARTURE,
        output_time_units="seconds",
    )
    reachable = matrix[matrix.travel_time <= 1800.0]
    seats = dict(zip(destinations["id"], destinations["seats"]))
    expected = {
        origin: sum(seats[to] for to in group["to_id"])
        for origin, group in reachable.groupby("from_id")
    }
    got = dict(zip(frame["from_id"], frame["accessibility"]))
    assert got == expected


def test_the_computer_validates_eagerly(network, helsinki_streets):
    import pandas

    geopandas = pytest.importorskip("geopandas")
    from cafein import Accessibility

    origins, destinations = _stop_sets(network)
    table = pandas.DataFrame({"id": destinations, "jobs": 1.0})
    nulled = table.assign(jobs=[None] + [1.0] * (len(table) - 1))
    complexed = table.assign(jobs=1.0 + 1.0j)
    points = geopandas.GeoDataFrame(
        {"id": ["p"]},
        geometry=geopandas.points_from_xy([24.9384], [60.1699]),
        crs="EPSG:4326",
    )
    transit = (network, origins, table, DEPARTURE)
    stops = (network, origins, destinations, DEPARTURE)
    street = (helsinki_streets, points, points)
    for args, kwargs, error, match in (
        # the aggregation and argument surface
        (transit, {"decay": "gravity"}, ValueError, "unknown decay"),
        (transit, {"decay": "linear"}, ValueError, "decay_params"),
        (transit, {"budgets": "1800"}, TypeError, "budgets"),
        (transit, {"opportunities": "people"}, ValueError, "no column"),
        (transit, {"max_travel_time": 30}, ValueError, "max_travel_time"),
        (
            (network, origins, nulled, DEPARTURE),
            {"opportunities": "jobs"},
            ValueError,
            "null value",
        ),
        ((network, origins, table), {}, TypeError, "requires departure"),
        (transit, {"transport_mode": "walk"}, ValueError, "transport_mode"),
        ((helsinki_streets, origins, table), {}, ValueError, "transport_mode"),
        (transit, {"exclude_routes": "1001"}, TypeError, "exclude_routes"),
        (
            (network, origins, complexed, DEPARTURE),
            {"opportunities": "jobs"},
            ValueError,
            "complex",
        ),
        # street requests reject the transit routing knobs
        (street, {"transport_mode": "walk", "router": "tbtr"}, ValueError, "router"),
        (street, {"transport_mode": "walk", "max_rides": 4}, ValueError, "max_rides"),
        # window knobs reject streets and bad combos
        (stops, {"percentiles": (50,)}, ValueError, "window"),
        (
            street,
            {"transport_mode": "walk", "departure_time_window": 10},
            ValueError,
            "window",
        ),
        # the cost axes
        (stops, {"cost": "calories"}, ValueError, "unknown cost"),
        (stops, {"cost": "distance"}, ValueError, "not an optimizable transit axis"),
        (
            street,
            {"transport_mode": "walk", "cost": "emissions"},
            ValueError,
            "cost engines",
        ),
        (stops, {"cost": "emissions"}, ValueError, "window"),
        (
            stops,
            {
                "cost": "emissions",
                "departure_time_window": 10,
                "percentiles": (25, 75),
            },
            ValueError,
            "percentiles",
        ),
        (
            stops,
            {"cost": "money", "departure_time_window": 10},
            ValueError,
            "fare structure",
        ),
        (stops, {"fares": object()}, ValueError, "fares applies"),
        (
            stops,
            {
                "cost": "money",
                "departure_time_window": 10,
                "fares": object(),
                "factors": object(),
            },
            ValueError,
            "factors and components",
        ),
    ):
        with pytest.raises(error, match=match):
            Accessibility(*args, **kwargs)


def test_percentile_accessibility_matches_the_percentile_matrix(network):
    from cafein import Accessibility

    origins, destinations = _stop_sets(network)
    frame = Accessibility(
        network,
        origins,
        destinations,
        DEPARTURE,
        budgets=(30.0,),
        departure_time_window=10,
        percentiles=(25, 75),
    )
    assert list(frame.columns) == [
        "from_id",
        "opportunity",
        "budget",
        "percentile",
        "accessibility",
    ]
    assert len(frame) == len(origins) * 2
    spread = network._core.travel_time_percentiles(
        origins, DATE, DEPARTURE_TIME, 600, [25, 75], 7, "auto", [], [], []
    )
    column = {stop: at for at, (stop, _, _) in enumerate(network.stops)}
    selection = [column[stop] for stop in destinations]
    for at, percentile in enumerate((25, 75)):
        times = spread[:, selection, at].astype("float64")
        times[times == UNREACHED] = numpy.nan
        expected = (times <= 1800.0).sum(axis=1).astype("float64")
        got = frame[frame["percentile"] == percentile]["accessibility"].to_numpy()
        assert numpy.allclose(got, expected), percentile
    # More of the window's departures reach within budget at the lower
    # percentile: accessibility is non-increasing in the percentile.
    wide = frame.pivot(index="from_id", columns="percentile", values="accessibility")
    assert (wide[25] >= wide[75]).all()


def _cost_surface(network, origins, destinations, optimize, fares=None, window=10):
    from cafein import TravelCostMatrix

    matrix = TravelCostMatrix(
        network,
        origins,
        destinations,
        DEPARTURE,
        optimize=optimize,
        departure_time_window=window,
        fares=fares,
    )
    column = "emissions" if optimize == "emissions" else "money"
    surface = numpy.full((len(origins), len(destinations)), numpy.nan)
    at_origin = {origin: at for at, origin in enumerate(origins)}
    at_dest = {dest: at for at, dest in enumerate(destinations)}
    for _, row in matrix.iterrows():
        surface[at_origin[row["from_id"]], at_dest[row["to_id"]]] = row[column]
    return surface


FARE_WINDOW = 4
"""Minutes of departure window for the shared zone-fare surface."""


@pytest.fixture(scope="module")
def fare_surface(network, helsinki_gtfs):
    """The zone structure and its windowed fare surface, built once."""
    from cafein import fares as fare_module

    structure = fare_module.zone_fare_structure(helsinki_gtfs, rules="zones")
    origins, destinations = _stop_sets(network)
    origins, destinations = origins[:3], destinations[:30]
    surface = _cost_surface(
        network, origins, destinations, "money", structure, window=FARE_WINDOW
    )
    return structure, origins, destinations, surface


@pytest.mark.parametrize("axis", ["emissions", "money"])
def test_cost_axis_accessibility_matches_the_cost_matrix(network, axis, request):
    from cafein import Accessibility

    if axis == "money":
        structure, origins, destinations, surface = request.getfixturevalue(
            "fare_surface"
        )
        window = FARE_WINDOW
        priced = surface[numpy.isfinite(surface)]
        assert priced.size, "the fixture feed prices no pair — fixture drift"
        budgets = (float(numpy.median(priced)),)
    else:
        structure = None
        origins, destinations = _stop_sets(network)
        surface = _cost_surface(network, origins, destinations, axis)
        window = 10
        budgets = (150.0, 600.0)
    frame = Accessibility(
        network,
        origins,
        destinations,
        DEPARTURE,
        cost=axis,
        departure_time_window=window,
        budgets=budgets,
        fares=structure,
    )
    assert "percentile" not in frame.columns
    for budget in budgets:
        expected = numpy.nansum(numpy.where(surface <= budget, 1.0, 0.0), axis=1)
        got = frame[frame["budget"] == budget]["accessibility"].to_numpy()
        assert numpy.allclose(got, expected), budget
    if axis == "emissions":
        # max_travel_time bounds the optimum's journeys: within one
        # minute only an origin's own zero-ride floor can qualify.
        capped = Accessibility(
            network,
            origins,
            destinations,
            DEPARTURE,
            cost=axis,
            departure_time_window=window,
            budgets=budgets,
            max_travel_time=1,
        )
        assert (capped["accessibility"] <= 1).all()
        assert capped["accessibility"].sum() < frame["accessibility"].sum()


def test_street_distance_accessibility_counts_within_metres(helsinki_streets):
    geopandas = pytest.importorskip("geopandas")
    from cafein import Accessibility

    origins = geopandas.GeoDataFrame(
        {"id": ["o1", "o2"]},
        geometry=geopandas.points_from_xy([24.9384, 24.9600], [60.1699, 60.1866]),
        crs="EPSG:4326",
    )
    destinations = geopandas.GeoDataFrame(
        {"id": ["d1", "d2", "d3"]},
        geometry=geopandas.points_from_xy(
            [24.9414, 24.9509, 24.9210], [60.1725, 60.1798, 60.1580]
        ),
        crs="EPSG:4326",
    )
    frame = Accessibility(
        helsinki_streets,
        origins,
        destinations,
        transport_mode="walk",
        cost="distance",
        budgets=(500.0, 2500.0),
    )
    table = helsinki_streets._core.cost_matrix(
        [(60.1699, 24.9384), (60.1866, 24.9600)],
        [(60.1725, 24.9414), (60.1798, 24.9509), (60.1580, 24.9210)],
        "walk",
        7200.0,
        1600.0,
        False,
        None,
    )
    surface = numpy.full((2, 3), numpy.nan)
    surface[table["from"], table["to"]] = (
        table["network_distance"] + table["connector_distance"]
    )
    for budget in (500.0, 2500.0):
        expected = numpy.nansum(numpy.where(surface <= budget, 1.0, 0.0), axis=1)
        got = frame[frame["budget"] == budget]["accessibility"].to_numpy()
        assert numpy.allclose(got, expected), budget
    # The default time axis serves the same street points: counts are
    # monotone in the budget and bounded by the destination count.
    timed = Accessibility(
        helsinki_streets,
        origins,
        destinations,
        transport_mode="walk",
        budgets=(10.0, 30.0),
    )
    assert len(timed) == 4
    wide = timed.pivot(index="from_id", columns="budget", values="accessibility")
    assert (wide[10.0] <= wide[30.0]).all()
    assert (wide[30.0] <= 3.0).all()


def test_empty_origins_and_duplicate_destinations_are_served(network):
    from cafein import Accessibility
    from cafein._cafein import aggregate_opportunity_sums_f64

    # A (0, N) surface keeps its destination dimension.
    empty = aggregate_opportunity_sums_f64(
        numpy.empty((0, 3)), [1.0, 1.0, 1.0], 1, [600.0], "step", None
    )
    assert empty.shape == (0, 1)
    # Repeated destination stops keep every column on the cost axes,
    # exactly as on the time axis.
    origins, destinations = _stop_sets(network)
    doubled = destinations[:3] + destinations[:3]
    single = Accessibility(
        network,
        origins,
        destinations[:3],
        DEPARTURE,
        cost="emissions",
        departure_time_window=10,
        budgets=(600.0,),
    )
    twice = Accessibility(
        network,
        origins,
        doubled,
        DEPARTURE,
        cost="emissions",
        departure_time_window=10,
        budgets=(600.0,),
    )
    assert numpy.allclose(
        twice["accessibility"].to_numpy(), 2 * single["accessibility"].to_numpy()
    )


def _nearest_oracle(matrix, to_ids, k, horizon=None):
    ranked = {}
    for row, costs in enumerate(matrix):
        pairs = [
            (float(cost), position)
            for position, cost in enumerate(costs)
            if cost != UNREACHED and (horizon is None or cost <= horizon)
        ]
        pairs.sort()
        ranked[row] = [(to_ids[position], cost) for cost, position in pairs[:k]]
    return ranked


def test_nearest_destinations_match_the_matrix_sort(network):
    from cafein import NearestDestinations

    origins, destinations = _stop_sets(network)
    wide = network.travel_time_matrix(origins, DEPARTURE)
    column = {stop: at for at, (stop, _, _) in enumerate(network.stops)}
    selected = wide[:, [column[stop] for stop in destinations]]
    frame = NearestDestinations(
        network,
        origins,
        destinations,
        DEPARTURE,
        k=3,
        output_time_units="seconds",
    )
    oracle = _nearest_oracle(selected, destinations, 3)
    for row, origin in enumerate(origins):
        cell = frame[frame["from_id"] == origin]
        assert list(cell["rank"]) == list(range(1, len(oracle[row]) + 1))
        assert list(cell["destination_id"]) == [stop for stop, _ in oracle[row]]
        assert list(cell["cost"]) == [cost for _, cost in oracle[row]]
    # The horizon prunes in the same axis unit (minutes in): ranks
    # beyond it are absent, exactly like unreachable destinations.
    capped = NearestDestinations(
        network,
        origins,
        destinations,
        DEPARTURE,
        k=3,
        max_cost=20,
        output_time_units="seconds",
    )
    capped_oracle = _nearest_oracle(selected, destinations, 3, horizon=1200)
    for row, origin in enumerate(origins):
        cell = capped[capped["from_id"] == origin]
        assert list(cell["destination_id"]) == [stop for stop, _ in capped_oracle[row]]
    # The default output reports the same ranks in whole minutes.
    rounded = NearestDestinations(network, origins, destinations, DEPARTURE, k=3)
    assert list(rounded["destination_id"]) == list(frame["destination_id"])
    assert (rounded["cost"] == numpy.rint(frame["cost"] / 60)).all()


@pytest.mark.parametrize(
    "matrix, k, expected_indices, expected_costs",
    [
        # Ranks break ties deterministically: equal costs keep column
        # order, and unreached ranks are -1 with NaN costs.
        pytest.param(
            [[300, 100, UNREACHED, 100], [50, UNREACHED, UNREACHED, UNREACHED]],
            3,
            [[1, 3, 0], [0, -1, -1]],
            [[100.0, 100.0, 300.0], [50.0, numpy.nan, numpy.nan]],
            id="tie-break",
        ),
        # An oversize k clamps to the columns.
        pytest.param([[120, 60]], 10**9, [[1, 0]], [[60.0, 120.0]], id="oversize-k"),
        # Zero destinations yield empty shapes.
        pytest.param(
            numpy.zeros((3, 0)),
            2,
            [[], [], []],
            numpy.zeros((3, 0)),
            id="zero-destinations",
        ),
    ],
)
def test_nearest_ranks_break_ties_deterministically(
    matrix, k, expected_indices, expected_costs
):
    from cafein import _cafein

    matrix = numpy.asarray(matrix, dtype="uint32")
    indices, costs = _cafein.aggregate_nearest(matrix, k, None)
    assert indices.tolist() == expected_indices
    numpy.testing.assert_array_equal(costs, numpy.asarray(expected_costs))


def test_nearest_windowed_percentile_ranks_match_the_percentile_matrix(network):
    from cafein import NearestDestinations

    origins, destinations = _stop_sets(network)
    wide = network.travel_time_matrix(
        origins, DEPARTURE, departure_time_window=10, percentiles=[75]
    )
    column = {stop: at for at, (stop, _, _) in enumerate(network.stops)}
    selected = wide[:, [column[stop] for stop in destinations], 0]
    frame = NearestDestinations(
        network,
        origins,
        destinations,
        DEPARTURE,
        k=2,
        departure_time_window=10,
        percentile=75,
        output_time_units="seconds",
    )
    oracle = _nearest_oracle(selected, destinations, 2)
    for row, origin in enumerate(origins):
        cell = frame[frame["from_id"] == origin]
        assert list(cell["destination_id"]) == [stop for stop, _ in oracle[row]]


def test_dominance_areas_dissolve_polygon_origins(network):
    geopandas = pytest.importorskip("geopandas")
    from shapely.geometry import box

    from cafein import NearestDestinations

    coordinates = {
        stop: (lat, lon) for stop, lat, lon in network.stops if lat is not None
    }
    origins, destinations = _stop_sets(network)
    frames = geopandas.GeoDataFrame(
        {"id": origins},
        geometry=[
            box(
                coordinates[stop][1] - 0.001,
                coordinates[stop][0] - 0.001,
                coordinates[stop][1] + 0.001,
                coordinates[stop][0] + 0.001,
            )
            for stop in origins
        ],
        crs="EPSG:4326",
    )
    frame = NearestDestinations(network, origins, destinations[:2], DEPARTURE, k=1)
    areas = frame.dominance_areas(frames)
    assert set(areas.columns) == {"destination_id", "geometry", "origins"}
    assert areas["origins"].sum() == frame[frame["rank"] == 1]["from_id"].nunique()
    assert areas.crs == frames.crs
    points = geopandas.GeoDataFrame(
        {"id": origins},
        geometry=geopandas.points_from_xy(
            [coordinates[stop][1] for stop in origins],
            [coordinates[stop][0] for stop in origins],
        ),
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="polygon"):
        frame.dominance_areas(points)


def test_nearest_destinations_validate_eagerly(network):
    from cafein import NearestDestinations

    origins, destinations = _stop_sets(network)
    with pytest.raises(ValueError, match="k must be"):
        NearestDestinations(network, origins, destinations, DEPARTURE, k=0)
    with pytest.raises(ValueError, match="percentile"):
        NearestDestinations(network, origins, destinations, DEPARTURE, percentile=75)
    with pytest.raises(ValueError, match="max_cost"):
        NearestDestinations(
            network,
            origins,
            destinations,
            DEPARTURE,
            cost="emissions",
            departure_time_window=10,
            max_cost=float("inf"),
        )
    with pytest.raises(ValueError, match="departure_time_window"):
        NearestDestinations(network, origins, destinations, DEPARTURE, cost="emissions")
    with pytest.raises(TypeError, match="requires departure"):
        NearestDestinations(network, origins, destinations)


def test_nearest_money_points_keep_their_own_ids(network_with_footpaths, helsinki_gtfs):
    geopandas = pytest.importorskip("geopandas")

    from cafein import NearestDestinations
    from cafein import fares as fare_module

    hsl = fare_module.zone_fare_structure(helsinki_gtfs, rules="zones")

    coordinates = {
        stop: (lat, lon)
        for stop, lat, lon in network_with_footpaths.stops
        if lat is not None
    }
    frame_o = geopandas.GeoDataFrame(
        {"id": [1]},
        geometry=geopandas.points_from_xy(
            [coordinates["1100602"][1]], [coordinates["1100602"][0]]
        ),
        crs="EPSG:4326",
    )
    frame_d = geopandas.GeoDataFrame(
        {"id": [7, 9]},
        geometry=geopandas.points_from_xy(
            [coordinates["1040280"][1], coordinates["1250551"][1]],
            [coordinates["1040280"][0], coordinates["1250551"][0]],
        ),
        crs="EPSG:4326",
    )
    frame = NearestDestinations(
        network_with_footpaths,
        frame_o,
        frame_d,
        DEPARTURE,
        k=2,
        cost="money",
        departure_time_window=10,
        fares=hsl,
    )
    # Integer point-frame ids round-trip in their own dtype — the
    # engines speak strings internally, the frame boundary casts back.
    assert set(frame["destination_id"]) <= set(frame_d["id"])
    assert set(frame["from_id"]) <= set(frame_o["id"])
    assert str(frame["destination_id"].dtype) == str(frame_d["id"].dtype)
    assert str(frame["from_id"].dtype) == str(frame_o["id"].dtype)
    assert len(frame) > 0


def test_dominance_areas_reject_duplicate_origin_ids(network):
    geopandas = pytest.importorskip("geopandas")
    from shapely.geometry import box

    from cafein import NearestDestinations

    origins, destinations = _stop_sets(network)
    frame = NearestDestinations(network, origins, destinations[:2], DEPARTURE, k=1)
    doubled = geopandas.GeoDataFrame(
        {"id": [origins[0], origins[0]]},
        geometry=[box(24.9, 60.1, 24.91, 60.11), box(24.92, 60.1, 24.93, 60.11)],
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="unique origin ids"):
        frame.dominance_areas(doubled)
    # The frame side guards too: ranks computed from repeated origins
    # would replicate polygons through the join.
    repeated = NearestDestinations(
        network, [origins[0], origins[0]], destinations[:2], DEPARTURE, k=1
    )
    single = geopandas.GeoDataFrame(
        {"id": [origins[0]]},
        geometry=[box(24.9, 60.1, 24.91, 60.11)],
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="repeated origin id"):
        repeated.dominance_areas(single)
    # A renamed geometry column is legal input.
    renamed = doubled.iloc[:1].rename_geometry("shape")
    assert len(frame.dominance_areas(renamed)) <= 2


def test_dominance_areas_join_numeric_origin_ids(network_with_footpaths):
    geopandas = pytest.importorskip("geopandas")
    from shapely.geometry import box

    from cafein import NearestDestinations

    network = network_with_footpaths
    stop_origins, destinations = _stop_sets(network)
    coordinates = {
        stop: (lat, lon) for stop, lat, lon in network.stops if lat is not None
    }
    numbered = geopandas.GeoDataFrame(
        {"id": list(range(len(stop_origins)))},
        geometry=[
            box(
                coordinates[stop][1] - 0.001,
                coordinates[stop][0] - 0.001,
                coordinates[stop][1] + 0.001,
                coordinates[stop][0] + 0.001,
            )
            for stop in stop_origins
        ],
        crs="EPSG:4326",
    )
    points = geopandas.GeoDataFrame(
        {"id": list(range(len(stop_origins)))},
        geometry=geopandas.points_from_xy(
            [coordinates[stop][1] for stop in stop_origins],
            [coordinates[stop][0] for stop in stop_origins],
        ),
        crs="EPSG:4326",
    )
    frame = NearestDestinations(
        network, points, points_to_frame(network, destinations[:2]), DEPARTURE, k=1
    )
    areas = frame.dominance_areas(numbered)
    assert len(areas) > 0
    assert areas["origins"].sum() == frame[frame["rank"] == 1]["from_id"].nunique()


def points_to_frame(network, stops):
    import geopandas

    coordinates = {
        stop: (lat, lon) for stop, lat, lon in network.stops if lat is not None
    }
    return geopandas.GeoDataFrame(
        {"id": stops},
        geometry=geopandas.points_from_xy(
            [coordinates[stop][1] for stop in stops],
            [coordinates[stop][0] for stop in stops],
        ),
        crs="EPSG:4326",
    )


def test_merged_feed_aliases_share_their_column(helsinki_gtfs, fares_poa):
    # On a merged feed a unique unqualified id and its qualified form
    # resolve to the same stop; the cost surfaces must fill both
    # columns instead of NaN-ing whichever alias lost the dedupe.
    from cafein import NearestDestinations, TransportNetwork

    poa = helsinki_gtfs.parent / "poa_eptc.zip"
    if not poa.exists():
        pytest.skip("poa_eptc.zip not fetched")
    merged = TransportNetwork.from_gtfs([str(helsinki_gtfs), str(poa)])
    frame = NearestDestinations(
        merged,
        ["0:4810551"],
        ["1250551", "0:1250551"],
        DEPARTURE,
        k=2,
        cost="emissions",
        departure_time_window=10,
    )
    assert list(frame["destination_id"]) == ["1250551", "0:1250551"]
    assert frame["cost"].iloc[0] == frame["cost"].iloc[1]


def test_catchment_membership_matches_the_walk_field_budget_filter(
    network_with_footpaths,
):
    h3 = pytest.importorskip("h3")
    from cafein import Catchment

    frame = Catchment(
        network_with_footpaths,
        ["1100602"],
        DEPARTURE,
        budgets=(10, 20),
        resolution=9,
    )
    assert list(frame["budget"]) == [10.0, 20.0]
    # The rendering oracle: exactly the union of the reached
    # vertices' cells, budget for budget, from the same field.
    arrivals = network_with_footpaths._core.travel_times_from_stop(
        "1100602", "2022-02-22", "08:30:00", 7, [], [], [], 3.6, 7200.0, 1600.0
    )
    reached = list(arrivals)
    indices = list(network_with_footpaths._core._stop_indices(reached))
    coordinates = {
        stop: (lat, lon)
        for stop, lat, lon in network_with_footpaths.stops
        if lat is not None
    }
    origin = coordinates["1100602"]
    seeds = [
        (index, float(arrivals[stop]))
        for index, stop in zip(indices, reached)
        if arrivals[stop] <= 1200
    ]
    lats, lons, seconds = network_with_footpaths._core._catchment_walk_field(
        origin, seeds, 1.0, 1200.0, 1600.0
    )
    for budget_minutes, row in zip((10, 20), frame.itertuples()):
        budget = budget_minutes * 60
        expected = {
            h3.latlng_to_cell(lat, lon, 9)
            for lat, lon, cost in zip(lats, lons, seconds)
            if cost <= budget
        }
        import shapely.geometry

        oracle = shapely.geometry.shape(h3.cells_to_h3shape(expected).__geo_interface__)
        assert row.geometry.equals(oracle)


def test_catchment_budgets_nest(network_with_footpaths):
    pytest.importorskip("h3")
    from cafein import Catchment

    frame = Catchment(
        network_with_footpaths, ["1100602"], DEPARTURE, budgets=(5, 15, 30)
    )
    geometries = list(frame.geometry)
    assert len(geometries) == 3
    for inner, outer in zip(geometries, geometries[1:]):
        assert inner.within(outer.buffer(1e-9))


def test_catchment_off_network_origin_keeps_its_walking_field(network_with_footpaths):
    pytest.importorskip("h3")
    geopandas = pytest.importorskip("geopandas")
    from cafein import Catchment

    # A point with no reachable stop still has a walking catchment:
    # exclude every stop, leaving the origin seed alone.
    stops = [stop for stop, _, _ in network_with_footpaths.stops]
    coordinates = {
        stop: (lat, lon)
        for stop, lat, lon in network_with_footpaths.stops
        if lat is not None
    }
    lat, lon = coordinates["1100602"]
    points = geopandas.GeoDataFrame(
        {"id": ["o"]},
        geometry=geopandas.points_from_xy([lon], [lat]),
        crs="EPSG:4326",
    )
    frame = Catchment(
        network_with_footpaths,
        points,
        DEPARTURE,
        budgets=(10,),
        exclude_stops=stops,
    )
    assert len(frame) == 1
    assert frame.geometry.iloc[0].area > 0


def test_catchment_empty_budget_is_an_absent_row(network_with_footpaths):
    pytest.importorskip("h3")
    from cafein import Catchment

    # One second reaches no street vertex from this stop — its snap
    # connector alone exceeds it — so the tiny budget has no row.
    frame = Catchment(
        network_with_footpaths, ["1100602"], DEPARTURE, budgets=(1 / 60, 10)
    )
    assert list(frame["budget"]) == [10.0]
    # A zero-rounding budget is refused, never silently dropped.
    with pytest.raises(ValueError, match="budgets"):
        Catchment(network_with_footpaths, ["1100602"], DEPARTURE, budgets=(0.001,))


def test_catchment_street_mode_spreads_by_time_and_distance(helsinki_streets):
    pytest.importorskip("h3")
    geopandas = pytest.importorskip("geopandas")
    from cafein import Catchment

    origin = geopandas.GeoDataFrame(
        {"id": ["o"]},
        geometry=geopandas.points_from_xy([24.9384], [60.1699]),
        crs="EPSG:4326",
    )
    timed = Catchment(helsinki_streets, origin, transport_mode="walk", budgets=(5, 10))
    assert len(timed) == 2
    inner, outer = timed.geometry
    assert inner.within(outer.buffer(1e-9))
    metered = Catchment(
        helsinki_streets,
        origin,
        transport_mode="walk",
        cost="distance",
        budgets=(400.0, 800.0),
    )
    assert len(metered) == 2
    assert metered.geometry.iloc[0].within(metered.geometry.iloc[1].buffer(1e-9))
    # An iso-distance area is not an iso-time area rescaled: both
    # exist independently on their own axes.
    assert list(metered["budget"]) == [400.0, 800.0]


def test_catchment_windowed_percentile_rule(network_with_footpaths):
    pytest.importorskip("h3")
    from cafein import Catchment

    # A vertex is reached when the chosen percentile of its stop's
    # arrival distribution fits the budget: the p90 catchment never
    # exceeds the p10 one.
    generous = Catchment(
        network_with_footpaths,
        ["1100602"],
        DEPARTURE,
        budgets=(25,),
        departure_time_window=10,
        percentile=10,
    )
    strict = Catchment(
        network_with_footpaths,
        ["1100602"],
        DEPARTURE,
        budgets=(25,),
        departure_time_window=10,
        percentile=90,
    )
    assert len(generous) == 1 and len(strict) == 1
    assert strict.geometry.iloc[0].within(generous.geometry.iloc[0].buffer(1e-9))


def test_catchment_validates_eagerly(network_with_footpaths, helsinki_streets):
    pytest.importorskip("h3")
    geopandas = pytest.importorskip("geopandas")
    from cafein import Catchment

    points = geopandas.GeoDataFrame(
        {"id": ["p"]},
        geometry=geopandas.points_from_xy([24.9384], [60.1699]),
        crs="EPSG:4326",
    )
    transit = (network_with_footpaths, ["1100602"], DEPARTURE)
    for args, kwargs, error, match in (
        (transit, {"resolution": 22}, ValueError, "resolution"),
        (transit, {"percentile": 75}, ValueError, "percentile"),
        (transit, {"cost": "emissions"}, ValueError, "departure_time_window"),
        ((network_with_footpaths, ["1100602"]), {}, TypeError, "requires departure"),
        (transit, {"budgets": ()}, ValueError, "budgets"),
        # windowed rules serve stop origins only
        (
            (network_with_footpaths, points, DEPARTURE),
            {"departure_time_window": 10},
            ValueError,
            "stop origins",
        ),
        ((helsinki_streets, points), {}, ValueError, "transport_mode"),
        (
            (helsinki_streets, points),
            {"transport_mode": "walk", "router": "tbtr"},
            ValueError,
            "apply to transit",
        ),
        # unusable walking options
        (
            transit,
            {"departure_time_window": 10, "walking_speed_kmph": float("nan")},
            ValueError,
            "walking_speed_kmph",
        ),
        (transit, {"snap_distance": 0}, ValueError, "snap_distance"),
        (
            (network_with_footpaths, ["no-such-stop"], DEPARTURE),
            {},
            KeyError,
            "no-such-stop",
        ),
        (transit, {"router": "fastest"}, ValueError, "router"),
        # the street snap distance validates its whole float envelope
        (
            (helsinki_streets, points),
            {"transport_mode": "walk", "snap_distance": 0},
            ValueError,
            "snap_distance",
        ),
        (
            (helsinki_streets, points),
            {"transport_mode": "walk", "snap_distance": -1},
            ValueError,
            "snap_distance",
        ),
        (
            (helsinki_streets, points),
            {"transport_mode": "walk", "snap_distance": float("nan")},
            ValueError,
            "snap_distance",
        ),
        (
            (helsinki_streets, points),
            {"transport_mode": "walk", "snap_distance": float("inf")},
            ValueError,
            "snap_distance",
        ),
        # The coordinate one-to-all rides RAPTOR: an explicit tbtr
        # request is refused, never silently ignored.
        (
            (network_with_footpaths, points, DEPARTURE),
            {"router": "tbtr"},
            ValueError,
            "rides RAPTOR",
        ),
    ):
        with pytest.raises(error, match=match):
            Catchment(*args, **kwargs)


def test_catchment_point_origin_serves_the_emissions_axis(network_with_footpaths):
    pytest.importorskip("h3")
    geopandas = pytest.importorskip("geopandas")
    from cafein import Catchment

    coordinates = {
        stop: (lat, lon)
        for stop, lat, lon in network_with_footpaths.stops
        if lat is not None
    }
    lat, lon = coordinates["1100602"]
    points = geopandas.GeoDataFrame(
        {"id": ["o"]},
        geometry=geopandas.points_from_xy([lon], [lat]),
        crs="EPSG:4326",
    )
    frame = Catchment(
        network_with_footpaths,
        points,
        DEPARTURE,
        cost="emissions",
        departure_time_window=10,
        budgets=(600.0,),
    )
    assert len(frame) == 1
    assert frame.geometry.iloc[0].area > 0


def test_products_carry_the_inputs_id_dtypes(network):
    import pandas as pd

    from cafein import Accessibility, DetailedItineraries, NearestDestinations

    nearest = NearestDestinations(
        network,
        origins=[4810551],
        destinations=pd.Series([1250551], dtype="Int64"),
        arrival="2022-02-22 09:30:00",
        k=1,
    )
    assert str(nearest["from_id"].dtype) == "int64"
    assert str(nearest["destination_id"].dtype) == "Int64"
    scores = Accessibility(
        network,
        origins=[4810551],
        destinations=pd.DataFrame({"id": [1250551], "reachable": [1]}),
        departure="2022-02-22 08:30:00",
        budgets=[60.0],
    )
    assert str(scores["from_id"].dtype) == "int64"
    legs = DetailedItineraries(
        network,
        origins=[4810551],
        destinations=[1250551],
        departure="2022-02-22 08:30:00",
    )
    assert str(legs["from_id"].dtype) == "int64"
    assert str(legs["to_id"].dtype) == "int64"


def _ramp_sums(surface, opportunities, budget, width):
    """Closed forms over a cost surface (NaN unreached): the ramp with
    ``width`` around ``budget``, the cutoff-anchored linear decay, and
    the mass the ramp carries strictly past the budget — the part a
    search clipped at the budget would lose."""
    costs = numpy.nan_to_num(surface, nan=numpy.inf)
    ramp = numpy.clip((budget + width / 2 - costs) / width, 0.0, 1.0)
    beyond = numpy.where(costs > budget, ramp, 0.0)
    cutoff = numpy.clip(1.0 - costs / budget, 0.0, 1.0)
    return (
        (ramp * opportunities).sum(axis=1),
        (cutoff * opportunities).sum(axis=1),
        (beyond * opportunities).sum(axis=1),
    )


def _median_budget(surface):
    finite = surface[numpy.isfinite(surface)]
    assert finite.size, "no reached destination — fixture drift"
    return float(numpy.median(finite))


def _whole_second_budget(surface):
    """A time budget in whole seconds: budgets are clock inputs and
    round to the second, so a fractional median would shift the ramp."""
    return float(numpy.floor(_median_budget(surface)))


def test_linear_ramp_keeps_its_weight_past_the_budget_on_stop_origins(network):
    from cafein import Accessibility

    origins, destinations = _stop_sets(network)
    seconds = _oracle_times(network, origins, destinations)
    budget = _whole_second_budget(seconds)
    width = budget  # the ramp reaches 0 at 1.5 × budget
    ramp, cutoff, beyond = _ramp_sums(seconds, 1.0, budget, width)
    assert beyond.sum() > 0, "no destination between the budget and the ramp's end"
    minutes = {"budgets": (budget / 60,)}
    linear = Accessibility(
        network,
        origins,
        destinations,
        DEPARTURE,
        decay="linear",
        decay_params={"width": width / 60},
        **minutes,
    )
    assert numpy.allclose(linear["accessibility"].to_numpy(), ramp, rtol=1e-12)
    anchored = Accessibility(
        network, origins, destinations, DEPARTURE, decay="linear_cutoff", **minutes
    )
    assert numpy.allclose(anchored["accessibility"].to_numpy(), cutoff, rtol=1e-12)
    with pytest.raises(ValueError, match="takes no decay_params"):
        Accessibility(
            network,
            origins,
            destinations,
            DEPARTURE,
            decay="linear_cutoff",
            decay_params={"width": 1.0},
            **minutes,
        )
    with pytest.raises(ValueError, match="linear_cutoff"):
        Accessibility(network, origins, destinations, DEPARTURE, decay="ramp")


def test_linear_ramp_keeps_its_weight_past_the_budget_across_a_window(network):
    from cafein import Accessibility, TravelTimeMatrix

    origins, destinations = _stop_sets(network)
    matrix = TravelTimeMatrix(
        network,
        origins,
        departure=DEPARTURE,
        departure_time_window=10,
        percentiles=(25, 75),
        output_time_units="seconds",
    )
    matrix = matrix[matrix["to_id"].isin(destinations)]
    at_origin = {origin: at for at, origin in enumerate(origins)}
    at_dest = {dest: at for at, dest in enumerate(destinations)}
    surfaces = {}
    for percentile in (25, 75):
        surface = numpy.full((len(origins), len(destinations)), numpy.nan)
        for _, row in matrix.iterrows():
            value = row[f"travel_time_p{percentile}"]
            if numpy.isfinite(value):
                surface[at_origin[row["from_id"]], at_dest[row["to_id"]]] = value
        surfaces[percentile] = surface
    budget = _whole_second_budget(surfaces[25])
    width = budget
    frame = Accessibility(
        network,
        origins,
        destinations,
        DEPARTURE,
        departure_time_window=10,
        percentiles=(25, 75),
        budgets=(budget / 60,),
        decay="linear",
        decay_params={"width": width / 60},
    )
    for percentile, surface in surfaces.items():
        ramp, _, beyond = _ramp_sums(surface, 1.0, budget, width)
        assert beyond.sum() > 0
        got = frame[frame["percentile"] == percentile]["accessibility"].to_numpy()
        assert numpy.allclose(got, ramp, rtol=1e-12), percentile


@pytest.mark.parametrize("axis", ["emissions", "money"])
def test_linear_ramp_keeps_its_weight_past_the_budget_on_cost_axes(
    network, axis, request
):
    from cafein import Accessibility

    if axis == "money":
        structure, origins, destinations, surface = request.getfixturevalue(
            "fare_surface"
        )
        window = FARE_WINDOW
    else:
        structure = None
        origins, destinations = _stop_sets(network)
        surface = _cost_surface(network, origins, destinations, axis)
        window = 10
    # The fixture prices these pairs at one or two fare levels: a budget
    # at half the median puts every dearer cost in the ramp's extension.
    finite = surface[numpy.isfinite(surface)]
    budget = 0.5 * _median_budget(surface)
    width = 4.0 * (float(finite.max()) - budget)
    assert budget > 0
    ramp, cutoff, beyond = _ramp_sums(surface, 1.0, budget, width)
    assert beyond.sum() > 0
    shared = dict(
        cost=axis, departure_time_window=window, budgets=(budget,), fares=structure
    )
    linear = Accessibility(
        network,
        origins,
        destinations,
        DEPARTURE,
        decay="linear",
        decay_params={"width": width},  # the axis's own unit, unconverted
        **shared,
    )
    assert numpy.allclose(linear["accessibility"].to_numpy(), ramp, rtol=1e-12)
    anchored = Accessibility(
        network, origins, destinations, DEPARTURE, decay="linear_cutoff", **shared
    )
    assert numpy.allclose(anchored["accessibility"].to_numpy(), cutoff, rtol=1e-12)


def test_linear_ramp_keeps_its_weight_past_the_budget_door_to_door(
    network_with_footpaths,
):
    geopandas = pytest.importorskip("geopandas")
    from cafein import Accessibility, NearestDestinations

    origins = geopandas.GeoDataFrame(
        {"id": ["kamppi", "kallio"]},
        geometry=geopandas.points_from_xy([24.9316, 24.9500], [60.1688, 60.1841]),
        crs="EPSG:4326",
    )
    destinations = geopandas.GeoDataFrame(
        {"id": ["a", "b", "c"], "seats": [10.0, 20.0, 40.0]},
        geometry=geopandas.points_from_xy(
            [24.9414, 24.9210, 24.9700], [60.1725, 60.1580, 60.2000]
        ),
        crs="EPSG:4326",
    )
    # The accessibility dispatch's own per-pair costs (door-to-door
    # walking keeps fractional seconds the matrix rounds away).
    costs = NearestDestinations(
        network_with_footpaths,
        origins,
        destinations,
        DEPARTURE,
        k=3,
        output_time_units="seconds",
    )
    surface = numpy.full((2, 3), numpy.nan)
    at_origin = {"kamppi": 0, "kallio": 1}
    at_dest = {"a": 0, "b": 1, "c": 2}
    for _, row in costs.iterrows():
        surface[at_origin[row["from_id"]], at_dest[row["destination_id"]]] = row["cost"]
    budget = _whole_second_budget(surface)
    width = budget
    seats = destinations["seats"].to_numpy()
    ramp, _, beyond = _ramp_sums(surface, seats, budget, width)
    assert beyond.sum() > 0
    frame = Accessibility(
        network_with_footpaths,
        origins,
        destinations,
        DEPARTURE,
        opportunities="seats",
        budgets=(budget / 60,),
        decay="linear",
        decay_params={"width": width / 60},
    )
    got = dict(zip(frame["from_id"], frame["accessibility"]))
    assert numpy.allclose([got["kamppi"], got["kallio"]], ramp, rtol=1e-12)


def test_linear_ramp_streams_identically(network, tmp_path):
    from cafein import Accessibility

    parquet = pytest.importorskip("pyarrow.parquet")
    origins, destinations = _stop_sets(network)
    seconds = _oracle_times(network, origins, destinations)
    budget = _whole_second_budget(seconds)
    ramp, _, beyond = _ramp_sums(seconds, 1.0, budget, budget)
    assert beyond.sum() > 0
    call = dict(
        budgets=(budget / 60,), decay="linear", decay_params={"width": budget / 60}
    )
    frame = Accessibility(network, origins, destinations, DEPARTURE, **call)
    Accessibility.to_parquet(
        network,
        origins,
        destinations,
        DEPARTURE,
        output=tmp_path / "ramp.parquet",
        batch_size=2,
        **call,
    )
    read = parquet.read_table(tmp_path / "ramp.parquet").to_pandas()
    read = read.astype({"from_id": str}).sort_values("from_id").reset_index(drop=True)
    expected = frame.astype({"from_id": str}).sort_values("from_id")
    assert numpy.allclose(read["accessibility"], expected["accessibility"], rtol=1e-12)
    assert numpy.allclose(expected["accessibility"], ramp, rtol=1e-12)


def test_from_matrix_reproduces_the_routed_computers(network):
    import pandas as pd

    from cafein import Accessibility, NearestDestinations, TravelTimeMatrix

    origins, destinations = _stop_sets(network)
    matrix = TravelTimeMatrix(
        network, origins, departure=DEPARTURE, output_time_units="seconds"
    )
    sub = matrix[matrix["to_id"].isin(destinations)]
    routed = Accessibility(
        network, origins, destinations, DEPARTURE, budgets=(15.0, 30.0)
    )
    framed = Accessibility.from_matrix(
        sub, destinations, budgets=(15.0, 30.0), origins=origins, time_units="seconds"
    )
    pd.testing.assert_frame_equal(pd.DataFrame(framed), pd.DataFrame(routed))
    # An opportunity table weights identically.
    table = pd.DataFrame(
        {"id": destinations, "jobs": numpy.arange(10.0, 10.0 + len(destinations))}
    )
    routed_jobs = Accessibility(
        network, origins, table, DEPARTURE, opportunities="jobs", budgets=(20.0,)
    )
    framed_jobs = Accessibility.from_matrix(
        sub,
        table,
        opportunities="jobs",
        budgets=(20.0,),
        origins=origins,
        time_units="seconds",
    )
    pd.testing.assert_frame_equal(pd.DataFrame(framed_jobs), pd.DataFrame(routed_jobs))
    # The percentile planes of a windowed matrix aggregate per plane.
    windowed = TravelTimeMatrix(
        network,
        origins,
        departure=DEPARTURE,
        departure_time_window=10,
        percentiles=[25, 75],
        output_time_units="seconds",
    )
    windowed = windowed[windowed["to_id"].isin(destinations)]
    routed_p = Accessibility(
        network,
        origins,
        destinations,
        DEPARTURE,
        departure_time_window=10,
        percentiles=(25, 75),
        budgets=(30.0,),
    )
    framed_p = Accessibility.from_matrix(
        windowed,
        destinations,
        budgets=(30.0,),
        origins=origins,
        time_units="seconds",
        percentiles=[25, 75],
    )
    pd.testing.assert_frame_equal(pd.DataFrame(framed_p), pd.DataFrame(routed_p))
    # A single percentile plane aggregates like the pair.
    routed_one = Accessibility(
        network,
        origins,
        destinations,
        DEPARTURE,
        departure_time_window=10,
        percentiles=(25,),
        budgets=(30.0,),
    )
    framed_one = Accessibility.from_matrix(
        windowed,
        destinations,
        budgets=(30.0,),
        origins=pd.DataFrame({"id": origins}),
        time_units="seconds",
        percentiles=[25],
    )
    pd.testing.assert_frame_equal(pd.DataFrame(framed_one), pd.DataFrame(routed_one))
    # The emissions axis reads the cost matrix's column.
    from cafein import TravelCostMatrix

    with pytest.warns(UserWarning, match="route_type"):
        cost = TravelCostMatrix(
            network,
            origins,
            departure=DEPARTURE,
            optimize="emissions",
            departure_time_window=10,
        )
    cost = cost[cost["to_id"].isin(destinations)]
    routed_e = Accessibility(
        network,
        origins,
        destinations,
        DEPARTURE,
        cost="emissions",
        departure_time_window=10,
        budgets=(600.0,),
    )
    framed_e = Accessibility.from_matrix(
        cost, destinations, cost="emissions", budgets=(600.0,), origins=origins
    )
    pd.testing.assert_frame_equal(pd.DataFrame(framed_e), pd.DataFrame(routed_e))
    # NearestDestinations ranks the same frame.
    routed_n = NearestDestinations(
        network, origins, destinations, DEPARTURE, k=3, output_time_units="seconds"
    )
    framed_n = NearestDestinations.from_matrix(
        sub,
        origins=origins,
        destinations=destinations,
        k=3,
        time_units="seconds",
        output_time_units="seconds",
    )
    pd.testing.assert_frame_equal(pd.DataFrame(framed_n), pd.DataFrame(routed_n))


def test_from_matrix_reads_the_former_priced_column(network, helsinki_gtfs):
    # A matrix priced before the column took the objective's name aggregates
    # identically on the money axis.
    import pandas as pd

    from cafein import Accessibility, TravelCostMatrix, fares as fare_module

    structure = fare_module.zone_fare_structure(helsinki_gtfs, rules="zones")
    matrix = TravelCostMatrix(
        network,
        ["1040601"],
        ["1121601"],
        DEPARTURE,
        optimize="money",
        departure_time_window=10,
        fares=structure,
    )
    current = Accessibility.from_matrix(
        matrix, ["1121601"], cost="money", budgets=(3.0,)
    )
    former = Accessibility.from_matrix(
        matrix.rename(columns={"money": "fare"}),
        ["1121601"],
        cost="money",
        budgets=(3.0,),
    )
    pd.testing.assert_frame_equal(pd.DataFrame(former), pd.DataFrame(current))
    assert pd.DataFrame(current)["accessibility"].iloc[0] == 1.0


def test_from_matrix_refusals():
    import pandas as pd

    from cafein import Accessibility, NearestDestinations

    def matrix(**overrides):
        base = {
            "from_id": ["o1", "o2"],
            "to_id": ["d1", "d1"],
            "travel_time": [600.0, 1200.0],
        }
        base.update(overrides)
        return pd.DataFrame(base)

    destinations = ["d1"]
    for frame, kwargs, match in (
        (matrix().drop(columns="to_id"), {}, "no to_id"),
        (matrix(), {"cost": "speed"}, "unknown cost"),
        (matrix(), {"time_units": "hours"}, "time_units"),
        (matrix(), {"cost": "emissions"}, "no emissions column"),
        (matrix(slot=["p", "q"]), {}, "several slot slots"),
        (matrix(from_id=["o1", "o3"]), {"origins": ["o1", "o2"]}, "outside origins="),
        (matrix(to_id=["d1", "dX"]), {}, "outside the destination universe"),
        (
            matrix(from_id=["o1", "o1"]),
            {},
            "duplicate",
        ),
        (matrix(), {"percentiles": [25]}, "no travel_time_p25 column"),
    ):
        with pytest.raises(ValueError, match=match):
            Accessibility.from_matrix(frame, destinations, **kwargs)
    # A one-slot frame passes; NearestDestinations shares the checks.
    single = matrix(from_id=["o1", "o2"], departure_time="2022-02-22 08:30:00")
    frame = Accessibility.from_matrix(single, destinations, budgets=(30.0,))
    assert list(frame["from_id"]) == ["o1", "o2"]
    with pytest.raises(ValueError, match="k must be"):
        NearestDestinations.from_matrix(matrix(), k=0)
    with pytest.raises(ValueError, match="max_cost"):
        NearestDestinations.from_matrix(matrix(), cost="emissions", max_cost=-1)
    with pytest.raises(ValueError, match="percentile"):
        NearestDestinations.from_matrix(matrix(), percentile=101)
    with pytest.raises(ValueError, match="percentile"):
        Accessibility.from_matrix(matrix(), destinations, percentiles=[True])
    with pytest.raises(ValueError, match="'id' column"):
        Accessibility.from_matrix(
            matrix(), destinations, origins=pd.DataFrame({"name": ["o1"]})
        )
    with pytest.raises(ValueError, match="'id' column"):
        Accessibility.from_matrix(matrix(), pd.DataFrame({"name": ["d1"]}))
    with pytest.raises(ValueError, match="names no percentiles"):
        Accessibility.from_matrix(matrix(), destinations, percentiles=())
    # An origin table's id dtype survives to the output.
    typed = matrix(from_id=pd.array([1, 2], dtype="Int32"), to_id=["d1", "d1"])
    frame = Accessibility.from_matrix(
        typed,
        destinations,
        origins=pd.DataFrame({"id": pd.array([1, 2], dtype="Int32")}),
    )
    assert str(frame["from_id"].dtype) == "Int32"


def test_from_matrix_reads_shard_directories(network_with_footpaths, tmp_path):
    import json

    import geopandas
    import pandas as pd

    from cafein import Accessibility, NearestDestinations, TravelTimeMatrix

    coordinates = {stop: (lat, lon) for stop, lat, lon in network_with_footpaths.stops}
    ids = ["4810551", "1250551", "4740551"]
    points = geopandas.GeoDataFrame(
        {"id": ids},
        geometry=geopandas.points_from_xy(
            [coordinates[stop][1] for stop in ids],
            [coordinates[stop][0] for stop in ids],
        ),
        crs="EPSG:4326",
    )
    out = tmp_path / "matrix"
    TravelTimeMatrix.to_parquet(
        network_with_footpaths,
        points,
        points,
        departure=DEPARTURE,
        output=out,
        batch_size=2,
        output_time_units="seconds",
    )
    frame = Accessibility.from_matrix(
        out, ids, budgets=(30.0,), origins=ids, time_units="seconds"
    )
    from cafein import _streaming

    loaded = _streaming.read_shards(out)
    direct = Accessibility.from_matrix(
        loaded, ids, budgets=(30.0,), origins=ids, time_units="seconds"
    )
    pd.testing.assert_frame_equal(pd.DataFrame(frame), pd.DataFrame(direct))
    nearest = NearestDestinations.from_matrix(
        out, origins=ids, destinations=ids, k=2, time_units="seconds"
    )
    assert set(nearest["from_id"]) <= set(ids)
    # The cost producers' directories aggregate the same way.
    from cafein.matrices import TravelCostMatrix, travel_cost_table

    for producer, target in (
        (TravelCostMatrix.to_parquet, tmp_path / "cost"),
        (travel_cost_table, tmp_path / "table"),
    ):
        with pytest.warns(UserWarning, match="route_type"):
            producer(
                network_with_footpaths,
                points,
                points,
                departure=DEPARTURE,
                output=target,
                batch_size=2,
            )
        streamed = Accessibility.from_matrix(
            target, ids, cost="emissions", budgets=(600.0,), origins=ids
        )
        direct = Accessibility.from_matrix(
            _streaming.read_shards(target),
            ids,
            cost="emissions",
            budgets=(600.0,),
            origins=ids,
        )
        pd.testing.assert_frame_equal(pd.DataFrame(streamed), pd.DataFrame(direct))
    # The directory refusals: a tampered shard hash, an incomplete
    # run, a foreign operation, and a missing manifest.
    target = out / "manifest.json"
    pristine = target.read_text()
    manifest = json.loads(pristine)
    manifest["shards"][0]["sha256"] = "0" * 64
    target.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="different content"):
        Accessibility.from_matrix(out, ids)
    manifest = json.loads(pristine)
    manifest["shards"] = manifest["shards"][:1]
    target.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="batch plan"):
        Accessibility.from_matrix(out, ids)
    manifest = json.loads(pristine)
    manifest["shards"][1]["completed"] = False
    target.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incomplete"):
        Accessibility.from_matrix(out, ids)
    manifest = json.loads(pristine)
    manifest["operation"] = "Accessibility.to_parquet"
    target.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not a matrix producer"):
        Accessibility.from_matrix(out, ids)
    target.write_text(pristine)
    # A symlinked shard refuses even though its bytes would verify.
    shard = out / "part-00000.parquet"
    aside = tmp_path / "aside.parquet"
    shard.rename(aside)
    shard.symlink_to(aside)
    with pytest.raises(ValueError, match="regular file"):
        Accessibility.from_matrix(out, ids)
    shard.unlink()
    aside.rename(shard)
    # A multi-slot run refuses on the manifest's authoritative slots.
    slotted = tmp_path / "slotted"
    TravelTimeMatrix.to_parquet(
        network_with_footpaths,
        points,
        points,
        departure=[DEPARTURE, "2022-02-22 12:00:00"],
        output=slotted,
        batch_size=2,
    )
    with pytest.raises(ValueError, match="select one slot"):
        Accessibility.from_matrix(slotted, ids)
    with pytest.raises(ValueError, match="no manifest.json"):
        Accessibility.from_matrix(tmp_path / "nowhere", ids)
