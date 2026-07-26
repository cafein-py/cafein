"""Fixtures shared by the cafein test suite."""

import os
import pathlib
import time

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
def fares_poa():
    """Path to r5r's saved Porto Alegre fare structure."""
    return _data_file("fares_poa.zip")


@pytest.fixture(scope="session")
def artifact_cache(tmp_path_factory):
    """The directory shared by every worker for cached network artifacts.

    Under ``pytest -n`` each worker's ``getbasetemp()`` is its own
    ``popen-gw*`` subdirectory; the parent is the run's shared directory, so
    an artifact written there is built once for the whole run rather than
    once per worker. Without xdist, ``getbasetemp()`` already **is** the
    run's directory — taking its parent there would land in the persistent
    per-user pytest directory, where a later run could silently reuse a stale
    artifact built by older code.
    """
    base = tmp_path_factory.getbasetemp()
    if os.environ.get("PYTEST_XDIST_WORKER"):
        base = base.parent
    shared = base / "cafein-networks"
    shared.mkdir(parents=True, exist_ok=True)
    return shared


def _cached_network(cache, name, build):
    """A network loaded from the cached artifact, built once if it is missing.

    Building the Helsinki network from GTFS costs ~18 s; loading the artifact
    it saves costs ~1 s, so every fixture and every worker after the first
    pays the cheap path. The lock is an exclusive create: whichever process
    wins builds, the rest wait for the artifact to appear.
    """
    from cafein import TransportNetwork

    artifact = cache / f"{name}.cafein"
    lock = cache / f"{name}.lock"
    deadline = time.monotonic() + 600
    while not artifact.exists():
        try:
            handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() > deadline:
                raise TimeoutError(f"timed out waiting for the {name} artifact")
            time.sleep(0.25)
            continue
        try:
            staged = cache / f"{name}.building"
            build().save(staged)
            staged.replace(artifact)
        finally:
            os.close(handle)
            lock.unlink(missing_ok=True)
    return TransportNetwork.load(artifact)


@pytest.fixture(scope="session")
def network(helsinki_gtfs, artifact_cache):
    """The Helsinki network with default preprocessing.

    Shared across modules, and cached as an artifact so the GTFS build happens
    once per run rather than once per module or worker. Tests that mutate a
    network (installing shortcuts, transfer sets, or street data) must build
    their own instead of taking this one.
    """
    pytest.importorskip("cafein._cafein")
    from cafein import TransportNetwork

    return _cached_network(
        artifact_cache,
        "helsinki",
        lambda: TransportNetwork.from_gtfs([str(helsinki_gtfs)]),
    )


@pytest.fixture(scope="session")
def network_with_footpaths(helsinki_gtfs, kantakaupunki_pbf, artifact_cache):
    """The Helsinki network with the OSM extract's walking structures."""
    pytest.importorskip("cafein._cafein")
    from cafein import TransportNetwork

    def build():
        with pytest.warns(UserWarning):
            return TransportNetwork.from_gtfs(
                [str(helsinki_gtfs)], osm_pbf=str(kantakaupunki_pbf)
            )

    return _cached_network(artifact_cache, "helsinki-footpaths", build)


@pytest.fixture(scope="session")
def helsinki_streets(kantakaupunki_pbf):
    """The standalone street network of the OSM extract, built once.

    The street modules each built their own; one extraction serves them all.
    """
    pytest.importorskip("cafein._cafein")
    from cafein import StreetNetwork

    return StreetNetwork.from_osm(str(kantakaupunki_pbf))
