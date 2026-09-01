"""Per-call ``workers=``: the local-pool mechanism and its surfaces."""

import io
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
    from cafein import Accessibility, TravelTimeMatrix

    origins = _served_stops(network, 2)
    for value, exc, match in (
        (True, TypeError, "workers must be an integer"),
        ("4", TypeError, "workers must be an integer"),
        (0, ValueError, "workers must be at least 1"),
    ):
        with pytest.raises(exc, match=match):
            TravelTimeMatrix(network, origins, departure=DEPARTURE, workers=value)
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
