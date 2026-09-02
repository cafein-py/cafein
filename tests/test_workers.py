"""Per-call ``workers=``: the local-pool mechanism and its surfaces."""

import io
import dataclasses
import logging

import pandas as pd
import pytest

import cafein
from cafein import _log
from cafein._cafein import _probe_workers

DEPARTURE = "2022-02-22 08:30:00"


@pytest.fixture(autouse=True)
def _pristine_logging():
    root = logging.getLogger("cafein")
    handlers = list(root.handlers)
    level = root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)
    _log._handler = None
    _log._prior_level = None
    del _log._collectors[:]
    _log.sync()


def _served_stops(network, count):
    stops = [stop for stop, lat, lon in network.stops if lat is not None]
    return stops[1000 : 1000 + count]


def _distinct_width():
    ambient = _probe_workers(None)
    return ambient + 1 if ambient < 3 else ambient - 1


def test_probe_reports_requested_widths():
    assert _probe_workers(None) >= 1
    assert _probe_workers(1) == 1
    assert _probe_workers(3) == 3


def test_surfaces_ride_the_requested_pool(network, tmp_path, caplog):
    from cafein import Accessibility, TravelCostMatrix, TravelTimeMatrix

    requested = _distinct_width()
    origins = _served_stops(network, 40)
    small = _served_stops(network, 5)
    surfaces = (
        lambda: TravelTimeMatrix(
            network, origins, departure=DEPARTURE, workers=requested
        ),
        lambda: TravelCostMatrix(network, small, small, DEPARTURE, workers=requested),
        lambda: Accessibility(
            network, small, _served_stops(network, 30), DEPARTURE, workers=requested
        ),
        lambda: TravelTimeMatrix.to_parquet(
            network,
            origins,
            departure=DEPARTURE,
            output=tmp_path / "stream",
            workers=requested,
        ),
        lambda: TravelTimeMatrix.to_parquet(
            network,
            _served_stops(network, 8),
            arrival=DEPARTURE,
            output=tmp_path / "arrive",
            workers=requested,
        ),
    )
    for surface in surfaces:
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="cafein"):
            surface()
        assert any(
            f"on {requested} workers" in record.getMessage()
            for record in caplog.records
        )
    # An unrequested call rides the ambient width.
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="cafein"):
        TravelTimeMatrix(network, origins, departure=DEPARTURE)
    assert any(
        f"on {_probe_workers(None)} workers" in record.getMessage()
        for record in caplog.records
    )


def test_results_are_identical_across_widths(network):
    from cafein import TravelTimeMatrix

    origins = _served_stops(network, 40)
    default = TravelTimeMatrix(network, origins, departure=DEPARTURE)
    narrow = TravelTimeMatrix(network, origins, departure=DEPARTURE, workers=1)
    wide = TravelTimeMatrix(network, origins, departure=DEPARTURE, workers=3)
    pd.testing.assert_frame_equal(pd.DataFrame(default), pd.DataFrame(narrow))
    pd.testing.assert_frame_equal(pd.DataFrame(default), pd.DataFrame(wide))


def test_workers_validates_eagerly(network, tmp_path):
    from cafein import Accessibility, TravelCostMatrix, TravelTimeMatrix

    origins = _served_stops(network, 2)
    for value, exc, match in (
        (True, TypeError, "workers must be an integer"),
        ("4", TypeError, "workers must be an integer"),
        (0, ValueError, "workers must be at least 1"),
    ):
        with pytest.raises(exc, match=match):
            TravelTimeMatrix(network, origins, departure=DEPARTURE, workers=value)
    for value, exc, match in (
        ("nope", ValueError, "could not read"),
        (True, TypeError, "not a bool"),
        (-1, ValueError, "non-negative"),
    ):
        with pytest.raises(exc, match=match):
            TravelTimeMatrix(network, origins, departure=DEPARTURE, max_memory=value)
    with pytest.raises(ValueError, match="could not read"):
        TravelCostMatrix.to_parquet(
            network,
            origins,
            departure=DEPARTURE,
            output=tmp_path / "cost",
            max_memory="nope",
        )
    assert not (tmp_path / "cost").exists()
    # The streaming surfaces validate before writing anything.
    with pytest.raises(ValueError, match="workers must be at least 1"):
        Accessibility.to_parquet(
            network,
            origins,
            _served_stops(network, 5),
            DEPARTURE,
            output=tmp_path / "acc",
            workers=0,
        )


@pytest.fixture(scope="module")
def car_park_network(helsinki_gtfs, kantakaupunki_pbf):
    import warnings

    from cafein import TransportNetwork

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TransportNetwork.from_gtfs(
            [str(helsinki_gtfs)],
            osm_pbf=str(kantakaupunki_pbf),
            street_modes=("walk", "car"),
            country="FI",
        )


def test_car_park_arm_rides_the_requested_pool(car_park_network, caplog):
    import geopandas
    from shapely.geometry import Point

    from cafein import TravelTimeMatrix
    from cafein.policy import CarParkPolicy

    requested = _distinct_width()
    facilities = geopandas.GeoDataFrame(
        {"id": ["pasila"], "search_seconds": [240.0]},
        geometry=[Point(24.9330, 60.1990)],
        crs="EPSG:4326",
    )
    origins = geopandas.GeoDataFrame(
        {"id": ["a"]}, geometry=[Point(24.9130, 60.1980)], crs="EPSG:4326"
    )
    destinations = geopandas.GeoDataFrame(
        {"id": ["b"]}, geometry=[Point(24.9520, 60.1795)], crs="EPSG:4326"
    )
    with caplog.at_level(logging.DEBUG, logger="cafein"):
        TravelTimeMatrix(
            car_park_network,
            origins,
            destinations,
            DEPARTURE,
            street_policy=CarParkPolicy(facilities=facilities),
            workers=requested,
        )
    assert any(
        f"on {requested} workers" in record.getMessage() for record in caplog.records
    )


@pytest.mark.parametrize(
    "arm, engine, cells, row_bytes",
    [
        ("stop time", "time", "origins x stops", 64),
        ("stop cost", "time", "origins x stops", 48),
        ("stop emissions", "multicriteria", "origins x stops", 48),
        ("two slots", "time", "2 x origins x stops", 64 + 64),
        ("chunked", "time", "1 x stops", 64),
        ("network entry", "time", "origins x stops", 4),
        ("street time", "street", "points x points", 64),
        ("street cost", "street", "points x points", 48),
        ("car park", "time", "1", 64),
        ("point time", "time", "points x points", 64),
        ("point cost", "time", "points x points", 48),
        ("stream stop time", "time", "batch x stops", 88),
        ("table", "time", "origins x stops", 48),
        ("stream table", "time", "batch x stops", 72),
        ("stream stop cost", "time", "batch x stops", 72),
        ("stream arrive-by cost", "multicriteria", "batch x origins", 72),
        ("stream street time", "street", "batch x points", 88),
        ("stream point time", "time", "batch x points", 88),
        ("stream point cost", "time", "batch x points", 72),
    ],
)
def test_every_public_call_plans_once(
    network,
    network_with_footpaths,
    helsinki_streets,
    car_park_network,
    arm,
    engine,
    cells,
    row_bytes,
    caplog,
    monkeypatch,
    tmp_path,
):
    import re

    import geopandas
    from shapely.geometry import Point

    from cafein import TravelCostMatrix, TravelTimeMatrix, _memory, travel_cost_table
    from cafein.policy import CarParkPolicy

    calls = []
    real = _memory.plan_call
    distinct = _distinct_width()

    def recording(engine_, size, result_bytes, **kwargs):
        # A width the ambient pool never has: a dispatch that ignored the
        # plan would run on the pool instead and fail the seam check below.
        plan = dataclasses.replace(
            real(engine_, size, result_bytes, **kwargs), width=distinct
        )
        calls.append((engine_, size, result_bytes, kwargs.get("row_bytes"), plan.width))
        return plan

    monkeypatch.setattr(_memory, "plan_call", recording)
    origins = _served_stops(network, 2)
    points = geopandas.GeoDataFrame(
        {"id": [0, 1]},
        geometry=[Point(24.94, 60.17), Point(24.95, 60.175)],
        crs="EPSG:4326",
    )
    facilities = geopandas.GeoDataFrame(
        {"id": ["pasila"], "search_seconds": [240.0]},
        geometry=[Point(24.9330, 60.1990)],
        crs="EPSG:4326",
    )
    frame = lambda lon, lat: geopandas.GeoDataFrame(  # noqa: E731
        {"id": ["p"]}, geometry=[Point(lon, lat)], crs="EPSG:4326"
    )
    build = {
        "stop time": lambda: TravelTimeMatrix(network, origins, departure=DEPARTURE),
        "stop cost": lambda: TravelCostMatrix(network, origins, departure=DEPARTURE),
        "stop emissions": lambda: TravelCostMatrix(
            network,
            origins,
            departure=DEPARTURE,
            optimize="emissions",
            departure_time_window=5,
        ),
        "two slots": lambda: TravelTimeMatrix(
            network, origins, departure=[DEPARTURE, "2022-02-22 09:30"]
        ),
        "street time": lambda: TravelTimeMatrix(
            helsinki_streets, points, points, transport_mode="walk"
        ),
        "street cost": lambda: TravelCostMatrix(
            helsinki_streets, points, points, transport_mode="walk"
        ),
        "chunked": lambda: TravelTimeMatrix(
            network, origins, departure=DEPARTURE, chunk=(0, 2)
        ),
        "network entry": lambda: network.travel_time_matrix(origins, DEPARTURE),
        "table": lambda: travel_cost_table(network, origins, departure=DEPARTURE),
        "stream table": lambda: travel_cost_table(
            network, origins, departure=DEPARTURE, output=tmp_path / "g"
        ),
        "stream stop time": lambda: TravelTimeMatrix.to_parquet(
            network, origins, departure=DEPARTURE, output=tmp_path / "a"
        ),
        "stream stop cost": lambda: TravelCostMatrix.to_parquet(
            network, origins, departure=DEPARTURE, output=tmp_path / "b"
        ),
        "stream arrive-by cost": lambda: TravelCostMatrix.to_parquet(
            network,
            origins,
            origins,
            arrival="2022-02-22 09:30",
            arrival_time_window=5,
            optimize="emissions",
            output=tmp_path / "c",
        ),
        "stream street time": lambda: TravelTimeMatrix.to_parquet(
            helsinki_streets,
            points,
            points,
            transport_mode="walk",
            output=tmp_path / "d",
        ),
        "point time": lambda: TravelTimeMatrix(
            network_with_footpaths, points, points, departure=DEPARTURE
        ),
        "point cost": lambda: TravelCostMatrix(
            network_with_footpaths, points, points, departure=DEPARTURE
        ),
        "stream point time": lambda: TravelTimeMatrix.to_parquet(
            network_with_footpaths,
            points,
            points,
            departure=DEPARTURE,
            output=tmp_path / "e",
        ),
        "stream point cost": lambda: TravelCostMatrix.to_parquet(
            network_with_footpaths,
            points,
            points,
            departure=DEPARTURE,
            output=tmp_path / "f",
        ),
        "car park": lambda: TravelTimeMatrix(
            car_park_network,
            frame(24.9130, 60.1980),
            frame(24.9520, 60.1795),
            DEPARTURE,
            street_policy=CarParkPolicy(facilities=facilities),
        ),
    }
    with caplog.at_level(logging.DEBUG, logger="cafein"):
        build[arm]()
    # One plan per public call, sized by the engine's network and the
    # whole result; every core fan-out runs on its width.
    assert len(calls) == 1
    seen_engine, size, result_bytes, batch_row_bytes, width = calls[0]
    assert seen_engine == engine
    size_of = {"street": helsinki_streets.vertex_count}
    assert size == size_of.get(engine, network.stop_count)
    count = {
        "origins x stops": 2 * network.stop_count,
        "2 x origins x stops": 2 * 2 * network.stop_count,
        "1 x stops": network.stop_count,
        "points x points": 4,
        "1": 1,
        "batch x stops": network.stop_count,
        "batch x origins": 2,
        "batch x points": 2,
    }[cells]
    if cells.startswith("batch"):
        assert result_bytes == 0 and batch_row_bytes == count * row_bytes
    else:
        assert result_bytes == count * row_bytes
    widths = {
        int(m.group(1))
        for m in (
            re.search(r"^(?!probe ).* on (\d+) workers$", r.getMessage())
            for r in caplog.records
        )
        if m
    }
    assert widths == {width} == {distinct}


def test_stream_batches_follow_the_budget(network, tmp_path, monkeypatch):
    import json

    from cafein import TravelCostMatrix, _memory
    from cafein.matrices import _COST_ROW_BYTES, _STREAM_ROW_EXTRA_BYTES, _stream_batch

    # A stream without batch_size= takes the planned batch.
    monkeypatch.setattr(_memory, "resident_bytes", lambda: 0)
    origins = _served_stops(network, 2)
    out = tmp_path / "cost"
    TravelCostMatrix.to_parquet(
        network, origins, departure=DEPARTURE, output=out, max_memory="300M"
    )
    stored = json.loads((out / "manifest.json").read_text())["batch_size"]
    oracle = _memory.plan_call(
        "time",
        network.stop_count,
        0,
        streamed=True,
        row_bytes=network.stop_count * (_COST_ROW_BYTES + _STREAM_ROW_EXTRA_BYTES),
        max_memory="300M",
    )
    assert stored == oracle.batch_rows < 500
    # The entry carries an explicit batch through its plan; a caller
    # that never planned keeps the default.
    TravelCostMatrix.to_parquet(
        network,
        origins,
        departure=DEPARTURE,
        output=tmp_path / "seven",
        batch_size=7,
        max_memory="300M",
    )
    assert (
        json.loads((tmp_path / "seven" / "manifest.json").read_text())["batch_size"]
        == 7
    )
    assert _stream_batch(None, False, out, None) == 500


def test_a_tight_budget_reaches_every_fan_out(
    network, helsinki_streets, car_park_network, caplog, monkeypatch
):
    import re

    import geopandas
    from shapely.geometry import Point

    from cafein import TravelCostMatrix, TravelTimeMatrix, _memory
    from cafein.policy import CarParkPolicy

    # One search costs more than any budget: every plan floors to one
    # worker, and every core seam must run on it.
    monkeypatch.setattr(_memory, "resident_bytes", lambda: 0)
    monkeypatch.setattr(
        _memory, "BYTES_PER_UNIT", {engine: 10**12 for engine in _memory.ENGINES}
    )
    origins = _served_stops(network, 2)
    points = geopandas.GeoDataFrame(
        {"id": [0, 1]},
        geometry=[Point(24.94, 60.17), Point(24.95, 60.175)],
        crs="EPSG:4326",
    )
    facilities = geopandas.GeoDataFrame(
        {"id": ["pasila"], "search_seconds": [240.0]},
        geometry=[Point(24.9330, 60.1990)],
        crs="EPSG:4326",
    )
    frame = lambda lon, lat: geopandas.GeoDataFrame(  # noqa: E731
        {"id": ["p"]}, geometry=[Point(lon, lat)], crs="EPSG:4326"
    )
    tight = {"max_memory": "200M"}
    builders = (
        lambda: TravelTimeMatrix(network, origins, departure=DEPARTURE, **tight),
        lambda: TravelCostMatrix(
            network,
            origins,
            departure=DEPARTURE,
            optimize="emissions",
            departure_time_window=5,
            **tight,
        ),
        lambda: TravelTimeMatrix(
            helsinki_streets, points, points, transport_mode="walk", **tight
        ),
        lambda: TravelTimeMatrix(
            car_park_network,
            frame(24.9130, 60.1980),
            frame(24.9520, 60.1795),
            DEPARTURE,
            street_policy=CarParkPolicy(facilities=facilities),
            **tight,
        ),
    )
    with caplog.at_level(logging.DEBUG, logger="cafein"):
        for build in builders:
            with pytest.warns(UserWarning, match="exceeds the memory budget"):
                build()
    widths = [
        int(m.group(1))
        for m in (
            re.search(r"^(?!probe ).* on (\d+) workers$", r.getMessage())
            for r in caplog.records
        )
        if m
    ]
    assert len(widths) >= len(builders) and set(widths) == {1}


def test_the_network_entry_reports_the_budget_in_its_details(
    network, caplog, monkeypatch
):
    # A zero baseline: a long test session's resident size must not
    # turn the explicit budget into a refusal.
    from cafein import _memory

    monkeypatch.setattr(_memory, "resident_bytes", lambda: 0)
    origins = _served_stops(network, 2)
    with caplog.at_level(logging.INFO, logger="cafein"):
        network.travel_time_matrix(origins, DEPARTURE, max_memory="6G")
    details = [getattr(r, "cafein_details", None) or {} for r in caplog.records]
    assert any(d.get("max_memory") == "6G" for d in details)


def test_a_planning_refusal_follows_the_argument_checks(network, monkeypatch):
    # A budget too small for the result still lets the call's own
    # argument checks speak first; a sound call refuses at its first
    # dispatch, before any search allocates.
    from cafein import TravelTimeMatrix, _memory

    monkeypatch.setattr(_memory, "resident_bytes", lambda: 0)
    origins = _served_stops(network, 2)
    with pytest.raises(ValueError, match="street mode"):
        TravelTimeMatrix(
            network, origins, departure=DEPARTURE, transport_mode="car", max_memory="200M"
        )
    with pytest.raises(ValueError, match="exceed the memory budget"):
        TravelTimeMatrix(network, departure=DEPARTURE, max_memory="200M")
