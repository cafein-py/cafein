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
