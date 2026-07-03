"""Fixtures shared by the cafein test suite."""

import os
import pathlib

import pytest

DATA_DIRECTORY = pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def helsinki_gtfs():
    """Path to the Helsinki GTFS zip shared with r5py's sample data."""
    path = DATA_DIRECTORY / "helsinki_gtfs.zip"
    if not path.exists():
        message = (
            f"test data missing at {path}; run `python scripts/fetch_test_data.py`"
        )
        if os.environ.get("CAFEIN_REQUIRE_TEST_DATA"):
            pytest.fail(message)
        pytest.skip(message)
    return path
