"""The accessibility primitive: per-origin opportunity sums (core)."""

import numpy
import pytest

pytest.importorskip("cafein._cafein")

DATE, DEPARTURE = "2022-02-22", "08:30:00"
UNREACHED = numpy.iinfo("uint32").max


def _stop_sets(network):
    stops = [stop for stop, lat, lon in network.stops if lat is not None]
    return stops[:5], stops[1000:1050]


def _oracle_times(network, origins, destinations):
    matrix = network._core.travel_time_matrix(
        origins, DATE, DEPARTURE, 7, "auto", [], [], [], 5.0, 1800.0, 500.0
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
        DEPARTURE,
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

    def masked(weights):
        return numpy.nansum(
            numpy.where(times <= 1800.0, weights, 0.0) * opportunities,
            axis=1,
            keepdims=True,
        )

    cases = {
        ("linear", 600.0): numpy.clip((1800.0 + 300.0 - times) / 600.0, 0.0, 1.0),
        ("exponential", 600.0): numpy.exp(-numpy.log(2.0) * times / 600.0),
        ("logistic", 120.0): 1.0 / (1.0 + numpy.exp((times - 1800.0) / 120.0)),
    }
    for (decay, param), weights in cases.items():
        values = _sums(
            network, origins, destinations, opportunities, (1800.0,), decay, param
        )
        assert numpy.allclose(values, masked(weights)), decay


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
        DATE,
        DEPARTURE,
        opportunities="jobs",
        budgets=(900.0, 1800.0),
    )
    assert list(frame.columns) == ["from_id", "opportunity", "budget", "accessibility"]
    assert len(frame) == len(origins) * 2
    assert set(frame["opportunity"]) == {"jobs"}
    times = _oracle_times(network, origins, destinations)
    for budget in (900.0, 1800.0):
        expected = numpy.nansum(
            numpy.where(times <= budget, table["jobs"].to_numpy(), 0.0), axis=1
        )
        got = frame[frame["budget"] == budget]["accessibility"].to_numpy()
        assert numpy.allclose(got, expected)


def test_the_computer_counts_features_without_opportunities(network):
    from cafein import Accessibility

    origins, destinations = _stop_sets(network)
    frame = Accessibility(network, origins, destinations, DATE, DEPARTURE)
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
        DATE,
        DEPARTURE,
        opportunities="seats",
        budgets=(1800.0,),
    )
    matrix = TravelTimeMatrix(
        network_with_footpaths, origins, destinations, DATE, DEPARTURE
    )
    reachable = matrix[matrix.travel_time_s <= 1800.0]
    seats = dict(zip(destinations["id"], destinations["seats"]))
    expected = {
        origin: sum(seats[to] for to in group["to_id"])
        for origin, group in reachable.groupby("from_id")
    }
    got = dict(zip(frame["from_id"], frame["accessibility"]))
    assert got == expected


def test_the_computer_serves_street_modes(helsinki_streets):
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
        budgets=(600.0, 1800.0),
    )
    assert len(frame) == 4
    wide = frame.pivot(index="from_id", columns="budget", values="accessibility")
    assert (wide[600.0] <= wide[1800.0]).all()
    assert (wide[1800.0] <= 3.0).all()


def test_the_computer_validates_eagerly(network, helsinki_streets):
    import pandas

    from cafein import Accessibility

    origins, destinations = _stop_sets(network)
    table = pandas.DataFrame({"id": destinations, "jobs": 1.0})

    with pytest.raises(ValueError, match="unknown decay"):
        Accessibility(network, origins, table, DATE, DEPARTURE, decay="gravity")
    with pytest.raises(ValueError, match="decay_params"):
        Accessibility(network, origins, table, DATE, DEPARTURE, decay="linear")
    with pytest.raises(TypeError, match="budgets"):
        Accessibility(network, origins, table, DATE, DEPARTURE, budgets="1800")
    with pytest.raises(ValueError, match="no column"):
        Accessibility(network, origins, table, DATE, DEPARTURE, opportunities="people")
    with pytest.raises(ValueError, match="null value"):
        nulled = table.assign(jobs=[None] + [1.0] * (len(table) - 1))
        Accessibility(network, origins, nulled, DATE, DEPARTURE, opportunities="jobs")
    with pytest.raises(TypeError, match="date and departure"):
        Accessibility(network, origins, table)
    with pytest.raises(ValueError, match="transport_mode"):
        Accessibility(network, origins, table, DATE, DEPARTURE, transport_mode="walk")
    with pytest.raises(ValueError, match="transport_mode"):
        Accessibility(helsinki_streets, origins, table)
    with pytest.raises(TypeError, match="exclude_routes"):
        Accessibility(network, origins, table, DATE, DEPARTURE, exclude_routes="1001")
    with pytest.raises(ValueError, match="complex"):
        Accessibility(
            network,
            origins,
            table.assign(jobs=1.0 + 1.0j),
            DATE,
            DEPARTURE,
            opportunities="jobs",
        )


def test_street_requests_reject_transit_routing_knobs(helsinki_streets):
    geopandas = pytest.importorskip("geopandas")
    from cafein import Accessibility

    points = geopandas.GeoDataFrame(
        {"id": ["p"]},
        geometry=geopandas.points_from_xy([24.9384], [60.1699]),
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="router"):
        Accessibility(
            helsinki_streets, points, points, transport_mode="walk", router="tbtr"
        )
    with pytest.raises(ValueError, match="max_transfers"):
        Accessibility(
            helsinki_streets, points, points, transport_mode="walk", max_transfers=3
        )


def test_percentile_accessibility_matches_the_percentile_matrix(network):
    from cafein import Accessibility

    origins, destinations = _stop_sets(network)
    frame = Accessibility(
        network,
        origins,
        destinations,
        DATE,
        DEPARTURE,
        budgets=(1800.0,),
        window=600,
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
        origins, DATE, DEPARTURE, 600, [25, 75], 7, "auto", [], [], []
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


def test_window_knobs_reject_streets_and_bad_combos(network, helsinki_streets):
    geopandas = pytest.importorskip("geopandas")
    from cafein import Accessibility

    origins, destinations = _stop_sets(network)
    with pytest.raises(ValueError, match="window"):
        Accessibility(
            network, origins, destinations, DATE, DEPARTURE, percentiles=(50,)
        )
    points = geopandas.GeoDataFrame(
        {"id": ["p"]},
        geometry=geopandas.points_from_xy([24.9384], [60.1699]),
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="window"):
        Accessibility(
            helsinki_streets, points, points, transport_mode="walk", window=600
        )
