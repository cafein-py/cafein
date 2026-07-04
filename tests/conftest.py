"""Fixtures shared by the cafein test suite."""

import os
import pathlib

import pytest

DATA_DIRECTORY = pathlib.Path(__file__).parent / "data"


def _data_file(name):
    path = DATA_DIRECTORY / name
    if not path.exists():
        message = (
            f"test data missing at {path}; run `python scripts/fetch_test_data.py`"
        )
        if os.environ.get("CAFEIN_REQUIRE_TEST_DATA"):
            pytest.fail(message)
        pytest.skip(message)
    return path


@pytest.fixture(scope="session")
def helsinki_gtfs():
    """Path to the Helsinki GTFS zip shared with r5py's sample data."""
    return _data_file("helsinki_gtfs.zip")


@pytest.fixture(scope="session")
def kantakaupunki_pbf():
    """Path to the central-Helsinki OSM extract shared with r5py's sample data."""
    return _data_file("kantakaupunki.osm.pbf")


@pytest.fixture(scope="session")
def network(helsinki_gtfs):
    """The Helsinki network with default preprocessing, shared across
    modules because it takes seconds to build."""
    pytest.importorskip("cafein._cafein")
    from cafein import TransportNetwork

    return TransportNetwork.from_gtfs([str(helsinki_gtfs)])


@pytest.fixture(scope="session")
def network_with_footpaths(helsinki_gtfs, kantakaupunki_pbf):
    """The Helsinki network with the OSM extract's walking structures."""
    pytest.importorskip("cafein._cafein")
    from cafein import TransportNetwork

    with pytest.warns(UserWarning):
        return TransportNetwork.from_gtfs(
            [str(helsinki_gtfs)], osm_pbf=str(kantakaupunki_pbf)
        )
