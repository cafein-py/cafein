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


def test_workers_validates_eagerly(network, tmp_path, caplog):
    import re

    from cafein import (
        Accessibility,
        Catchment,
        NearestDestinations,
        TravelCostMatrix,
        TravelTimeMatrix,
    )

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
    with pytest.raises(ValueError, match="could not read"):
        Accessibility.to_parquet(
            network,
            origins,
            _served_stops(network, 5),
            DEPARTURE,
            output=tmp_path / "acc-budget",
            max_memory="nope",
        )
    assert not (tmp_path / "acc-budget").exists()
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
    # Every accessibility entry refuses the budget before any routing (no
    # core seam runs) or aggregation (an unreadable frame is never reached).
    unread = pd.DataFrame()
    destinations = _served_stops(network, 5)
    for call in (
        lambda: Accessibility(
            network, origins, destinations, DEPARTURE, max_memory="nope"
        ),
        lambda: NearestDestinations(
            network, origins, destinations, DEPARTURE, max_memory="nope"
        ),
        lambda: Catchment(
            network, origins, DEPARTURE, budgets=(10, 20), max_memory="nope"
        ),
        lambda: Accessibility.from_matrix(unread, destinations, max_memory="nope"),
        lambda: NearestDestinations.from_matrix(unread, max_memory="nope"),
    ):
        with caplog.at_level(logging.DEBUG, logger="cafein"):
            _log.sync()
            with pytest.raises(ValueError, match="could not read"):
                call()
    assert not any(
        re.search(r" on \d+ workers$", r.getMessage()) for r in caplog.records
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


def test_a_planning_refusal_follows_the_argument_checks(
    network,
    network_with_footpaths,
    car_park_network,
    multimodal_network,
    monkeypatch,
    tmp_path,
):
    # A budget too small for the result still lets the call's own
    # argument checks speak first; a sound call refuses at its first
    # dispatch, before any search allocates.
    from cafein import TravelTimeMatrix, _memory

    monkeypatch.setattr(_memory, "resident_bytes", lambda: 0)
    origins = _served_stops(network, 2)
    with pytest.raises(ValueError, match="street mode"):
        TravelTimeMatrix(
            network,
            origins,
            departure=DEPARTURE,
            transport_mode="car",
            max_memory="200M",
        )
    with pytest.raises(ValueError, match="exceed the memory budget"):
        TravelTimeMatrix(network, departure=DEPARTURE, max_memory="200M")
    # A refused plan runs no search: with a resident size past any budget,
    # the arrive-by catchment's access search and the car-park
    # composition (its inner offsets search stands in for it) both sit
    # behind the seam.
    import geopandas
    from shapely.geometry import Point

    from cafein import Accessibility, Catchment
    from cafein import network as network_module
    from cafein.policy import CarParkPolicy

    def no_search(*_args, **_kwargs):
        raise AssertionError("a search ran under a refused plan")

    core = network_with_footpaths._core

    class NoAccessSearch:
        def __getattr__(self, name):
            return no_search if name == "access_stops" else getattr(core, name)

    monkeypatch.setattr(network_with_footpaths, "_core", NoAccessSearch())
    monkeypatch.setattr(network_module, "_car_park_offsets", no_search)
    monkeypatch.setattr(_memory, "resident_bytes", lambda: 1 << 40)
    point = geopandas.GeoDataFrame(
        {"id": [0]}, geometry=[Point(24.94, 60.17)], crs="EPSG:4326"
    )
    facilities = geopandas.GeoDataFrame(
        {"id": ["pasila"], "search_seconds": [240.0]},
        geometry=[Point(24.9330, 60.1990)],
        crs="EPSG:4326",
    )
    for build in (
        lambda: Catchment(
            network_with_footpaths,
            point,
            arrival="2022-02-22 09:30",
            budgets=(10, 20),
            max_memory="200M",
        ),
        lambda: Accessibility(
            car_park_network,
            point,
            point,
            DEPARTURE,
            street_policy=CarParkPolicy(facilities=facilities),
            max_memory="200M",
        ),
    ):
        with pytest.raises(ValueError, match="exceed the memory budget"):
            build()
    # Under the forced refusal, a stream's and a catchment's own argument
    # checks still speak first.
    from cafein.travelers import TravelerProfile

    with pytest.raises(ValueError, match="chunk must be"):
        Accessibility.to_parquet(
            network,
            _served_stops(network, 2),
            _served_stops(network, 5),
            DEPARTURE,
            output=tmp_path / "refused",
            chunk=(3, 2),
            max_memory="200M",
        )
    assert not (tmp_path / "refused").exists()
    with pytest.raises(ValueError, match="ride the time axis"):
        Catchment(
            multimodal_network,
            point,
            DEPARTURE,
            cost="emissions",
            departure_time_window=10,
            traveler=TravelerProfile(wheelchair=True),
            max_memory="200M",
        )


@pytest.mark.parametrize(
    "arm, engine, size_of, bytes_of",
    [
        ("accessibility", "time", "stops", "2 x stops x 4"),
        ("accessibility emissions", "multicriteria", "stops", "2 x stops x 8"),
        ("accessibility arrive-by", "multicriteria", "stops", "2 x stops x 8"),
        ("accessibility points", "time", "footpath stops", "2 x 2 x 4"),
        ("accessibility street", "street", "vertices", "2 x 2 x 4"),
        ("nearest", "time", "stops", "2 x stops x 4"),
        ("catchment", "time", "footpath stops", "1 x stops x 4"),
        ("accessibility stream", "time", "stops", "batch: stops x 28"),
        (
            "accessibility arrive-by stream",
            "multicriteria",
            "stops",
            "fixed: 2 x stops x 8",
        ),
        ("accessibility from matrix", "time", "frame", "rows x 32 + 2 x 5 x 8"),
        ("nearest from matrix", "time", "frame", "rows x 32 + 2 x 5 x 8"),
        ("nearest arrive-by", "time", "stops", "2 x stops x 4"),
        ("catchment points", "time", "footpath stops", "2 x footpath stops x 4"),
        ("catchment cost", "multicriteria", "footpath stops", "1 x stops x 8"),
        ("catchment street", "street", "vertices", "2 x vertices x 4"),
        ("accessibility policy", "time", "car park stops", "1 x 1 x 4"),
        ("nearest points", "time", "footpath stops", "2 x 2 x 4"),
        ("nearest street", "street", "vertices", "2 x 2 x 4"),
        ("nearest emissions", "multicriteria", "stops", "2 x stops x 8"),
        ("catchment arrive-by", "time", "footpath stops", "1 x stops x 4"),
        ("accessibility stream points", "time", "footpath stops", "batch: 2 x 28"),
        ("accessibility stream street", "street", "vertices", "batch: 2 x 28"),
        ("accessibility iterator origins", "time", "stops", "2 x stops x 4"),
        ("accessibility explicit percentiles", "time", "stops", "2 x stops x 4"),
        ("accessibility from directory", "time", "frame", "directory"),
    ],
)
def test_accessibility_entries_plan_once(
    network,
    network_with_footpaths,
    helsinki_streets,
    car_park_network,
    arm,
    engine,
    size_of,
    bytes_of,
    caplog,
    monkeypatch,
    tmp_path,
):
    import re

    import geopandas
    from shapely.geometry import Point

    from cafein import (
        Accessibility,
        Catchment,
        NearestDestinations,
        TravelTimeMatrix,
        _log,
        _memory,
    )

    calls = []
    real = _memory.plan_call
    distinct = _distinct_width()

    def recording(engine_, size, result_bytes, **kwargs):
        # A width the pool never has: a dispatch ignoring the plan fails below.
        plan = dataclasses.replace(
            real(engine_, size, result_bytes, **kwargs), width=distinct
        )
        calls.append(
            (
                engine_,
                size,
                result_bytes,
                kwargs.get("row_bytes"),
                kwargs.get("fixed_bytes"),
            )
        )
        return plan

    origins = _served_stops(network, 2)
    destinations = _served_stops(network, 5)
    points = geopandas.GeoDataFrame(
        {"id": [0, 1]},
        geometry=[Point(24.94, 60.17), Point(24.95, 60.175)],
        crs="EPSG:4326",
    )
    from cafein.policy import CarParkPolicy

    facilities = geopandas.GeoDataFrame(
        {"id": ["pasila"], "search_seconds": [240.0]},
        geometry=[Point(24.9330, 60.1990)],
        crs="EPSG:4326",
    )

    def frame_of(lon, lat):
        return geopandas.GeoDataFrame(
            {"id": ["p"]}, geometry=[Point(lon, lat)], crs="EPSG:4326"
        )

    # A stop-origin time matrix covers every stop; built before the
    # planner is patched so only the arm's own plan is recorded.
    frame = TravelTimeMatrix(network, origins, departure=DEPARTURE)
    directory = tmp_path / "streamed"
    TravelTimeMatrix.to_parquet(network, origins, departure=DEPARTURE, output=directory)
    directory_rows = len(frame)
    every_stop = [stop for stop, _latitude, _longitude in network.stops]
    frame = frame[frame["to_id"].isin(destinations)]
    monkeypatch.setattr(_memory, "plan_call", recording)
    # A directory aggregation loads through the loader, then plans.
    from cafein import _streaming

    loads = []
    real_read = _streaming.read_shards

    def loading(path):
        loads.append(len(calls))
        return real_read(path)

    monkeypatch.setattr(_streaming, "read_shards", loading)
    build = {
        "accessibility": lambda: Accessibility(
            network, origins, destinations, DEPARTURE
        ),
        "accessibility emissions": lambda: Accessibility(
            network,
            origins,
            destinations,
            DEPARTURE,
            cost="emissions",
            departure_time_window=10,
            budgets=(600.0,),
        ),
        "accessibility arrive-by": lambda: Accessibility(
            network,
            origins,
            destinations,
            arrival="2022-02-22 09:30",
            arrival_time_window=10,
            cost="emissions",
            budgets=(600.0,),
        ),
        "accessibility points": lambda: Accessibility(
            network_with_footpaths, points, points, DEPARTURE
        ),
        "accessibility street": lambda: Accessibility(
            helsinki_streets, points, points, transport_mode="walk"
        ),
        # The explicit default percentile without a window plans one plane.
        "nearest": lambda: NearestDestinations(
            network, origins, destinations, DEPARTURE, percentile=50
        ),
        "catchment": lambda: Catchment(
            network_with_footpaths,
            ["1100602"],
            DEPARTURE,
            budgets=(10, 20),
            percentile=50,
        ),
        "accessibility stream": lambda: Accessibility.to_parquet(
            network, origins, destinations, DEPARTURE, output=tmp_path / "acc"
        ),
        "accessibility arrive-by stream": lambda: Accessibility.to_parquet(
            network,
            origins,
            destinations,
            arrival="2022-02-22 09:30",
            arrival_time_window=10,
            cost="emissions",
            budgets=(600.0,),
            output=tmp_path / "acc-arrive",
        ),
        "accessibility from matrix": lambda: Accessibility.from_matrix(
            frame, destinations
        ),
        "nearest from matrix": lambda: NearestDestinations.from_matrix(frame),
        "nearest points": lambda: NearestDestinations(
            network_with_footpaths, points, points, DEPARTURE
        ),
        "nearest street": lambda: NearestDestinations(
            helsinki_streets, points, points, transport_mode="walk"
        ),
        "nearest emissions": lambda: NearestDestinations(
            network,
            origins,
            destinations,
            DEPARTURE,
            cost="emissions",
            departure_time_window=10,
        ),
        "catchment arrive-by": lambda: Catchment(
            network_with_footpaths,
            ["1100602"],
            arrival="2022-02-22 09:30",
            arrival_time_window=10,
            budgets=(10, 20),
        ),
        "accessibility stream points": lambda: Accessibility.to_parquet(
            network_with_footpaths, points, points, DEPARTURE, output=tmp_path / "acc-p"
        ),
        "accessibility stream street": lambda: Accessibility.to_parquet(
            helsinki_streets,
            points,
            points,
            transport_mode="walk",
            output=tmp_path / "acc-s",
        ),
        "accessibility iterator origins": lambda: Accessibility(
            network, iter(list(origins)), destinations, DEPARTURE
        ),
        "accessibility explicit percentiles": lambda: Accessibility(
            network,
            origins,
            destinations,
            DEPARTURE,
            departure_time_window=10,
            percentiles=[50.0],
        ),
        "accessibility from directory": lambda: Accessibility.from_matrix(
            directory, every_stop
        ),
        "nearest arrive-by": lambda: NearestDestinations(
            network,
            origins,
            destinations,
            arrival="2022-02-22 09:30",
            arrival_time_window=10,
        ),
        "catchment points": lambda: Catchment(
            network_with_footpaths, points, DEPARTURE, budgets=(10, 20)
        ),
        "catchment cost": lambda: Catchment(
            network_with_footpaths,
            ["1100602"],
            DEPARTURE,
            cost="emissions",
            departure_time_window=10,
            budgets=(600.0,),
        ),
        "catchment street": lambda: Catchment(
            helsinki_streets, points, budgets=(10, 20), transport_mode="walk"
        ),
        "accessibility policy": lambda: Accessibility(
            car_park_network,
            frame_of(24.9130, 60.1980),
            frame_of(24.9520, 60.1795),
            DEPARTURE,
            street_policy=CarParkPolicy(facilities=facilities),
        ),
    }
    calls.clear()
    with caplog.at_level(logging.DEBUG, logger="cafein"):
        # The core's seam lines follow the synced level, as a computer syncs.
        _log.sync()
        build[arm]()
    assert len(calls) == 1
    seen_engine, size, result_bytes, row_bytes, fixed_bytes = calls[0]
    assert seen_engine == engine
    sizes = {
        "stops": network.stop_count,
        "footpath stops": network_with_footpaths.stop_count,
        "vertices": helsinki_streets.vertex_count,
        "car park stops": car_park_network.stop_count,
        "frame": 0,
    }
    assert size == sizes[size_of]
    stops = network.stop_count
    expected = {
        "2 x stops x 4": 2 * stops * 4,
        "2 x stops x 8": 2 * stops * 8,
        "2 x 2 x 4": 16,
        "1 x stops x 4": network_with_footpaths.stop_count * 4,
        "1 x stops x 8": network_with_footpaths.stop_count * 8,
        "2 x footpath stops x 4": 2 * network_with_footpaths.stop_count * 4,
        "2 x vertices x 4": 2 * helsinki_streets.vertex_count * 4,
        "1 x 1 x 4": 4,
        "rows x 32 + 2 x 5 x 8": len(frame) * 32 + 2 * 5 * 8,
    }
    if bytes_of == "directory":
        # The directory loads through the loader first; the plan then
        # covers the loaded frame and the aggregation's surface.
        assert loads == [0]
        assert result_bytes == directory_rows * 32 + 2 * len(every_stop) * 8
    elif bytes_of == "batch: 2 x 28":
        assert result_bytes == 0 and row_bytes == 2 * 28 and not fixed_bytes
    elif bytes_of.startswith("batch:"):
        assert result_bytes == 0 and row_bytes == stops * 28 and not fixed_bytes
    elif bytes_of.startswith("fixed:"):
        assert result_bytes == 0 and fixed_bytes == 2 * stops * 8
    else:
        assert result_bytes == expected[bytes_of]
    widths = {
        int(m.group(1))
        for m in (
            re.search(r"^(?!probe ).* on (\d+) workers$", r.getMessage())
            for r in caplog.records
        )
        if m
    }
    # Two catchment paths run core calls that carry no seam line; their
    # width reaches the core through the same active plan.
    no_seams = {"catchment points", "catchment street"}
    assert widths == (set() if arm in no_seams else {distinct})


def test_the_fare_refinement_runs_on_its_planned_width(
    network, helsinki_gtfs, caplog, monkeypatch
):
    import re

    from cafein import TravelCostMatrix, _memory, fares

    structure = fares.zone_fare_structure(helsinki_gtfs, rules="zones")
    # One priced pair: the refinement's width is the behaviour under test.
    pair = dict(origins=["1040601"], destinations=["1121601"])
    plans = []
    real = _memory.plan_call

    def recording(*args, **kwargs):
        plan = real(*args, **kwargs)
        plans.append(plan)
        return plan

    monkeypatch.setattr(_memory, "plan_call", recording)

    def refinement_seams():
        return [
            int(m.group(1))
            for m in (
                re.search(r"^fare refinement on (\d+) workers$", r.getMessage())
                for r in caplog.records
            )
            if m
        ]

    # A wide budget on a wide host: the plan clears the former four-worker
    # ceiling and the refinement pool follows it.
    monkeypatch.setattr(_memory, "resident_bytes", lambda: 0)
    monkeypatch.setattr(_memory, "ambient_workers", lambda: 8)
    with caplog.at_level(logging.DEBUG, logger="cafein"):
        TravelCostMatrix(
            network,
            departure=DEPARTURE,
            optimize="fare",
            fares=structure,
            departure_time_window=10,
            max_memory="64G",
            **pair,
        )
    (plan,) = plans
    assert plan.refinement_width > 4
    assert refinement_seams() == [plan.refinement_width]
    # An explicit workers= is the whole call's ceiling, the refinement's too.
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="cafein"):
        TravelCostMatrix(
            network,
            departure=DEPARTURE,
            optimize="fare",
            fares=structure,
            departure_time_window=10,
            workers=1,
            **pair,
        )
    assert refinement_seams() == [1]
    # The accessibility computers name the objective money: same phase.
    from cafein import NearestDestinations

    plans.clear()
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="cafein"):
        NearestDestinations(
            network,
            pair["origins"],
            pair["destinations"],
            DEPARTURE,
            cost="money",
            fares=structure,
            departure_time_window=10,
            max_memory="64G",
        )
    (plan,) = plans
    assert plan.refinement_width > 4
    assert refinement_seams() == [plan.refinement_width]
