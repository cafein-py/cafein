"""ULTRA shortcut computation and its persistence."""

import pytest

from cafein import TransportNetwork

# A small intermediate-walk cutoff and a narrow morning source-departure
# window keep these builds quick; the shortcut machinery, not a particular
# set size, is what the tests exercise.
CUTOFF = 300.0
WINDOW = {"min_departure": 28800, "max_departure": 29400}  # 08:00–08:10


@pytest.fixture(scope="module")
def ultra_network(helsinki_gtfs, kantakaupunki_pbf):
    """A Helsinki network with its ULTRA shortcuts computed."""
    pytest.importorskip("cafein._cafein")
    with pytest.warns(UserWarning):
        network = TransportNetwork.from_gtfs(
            [str(helsinki_gtfs)], osm_pbf=str(kantakaupunki_pbf)
        )
    network.compute_ultra_shortcuts(max_transfer_time=CUTOFF, **WINDOW)
    return network


def test_compute_returns_and_stores_a_shortcut_count(ultra_network):
    assert ultra_network.ultra_shortcut_count is not None
    assert ultra_network.ultra_shortcut_count > 0
    shortcuts = ultra_network.ultra_shortcuts
    assert len(shortcuts) == ultra_network.ultra_shortcut_count
    # Shortcuts are (origin, destination, seconds) between distinct stops.
    origin, destination, seconds = shortcuts[0]
    assert origin != destination
    assert isinstance(seconds, int)


def test_compute_is_deterministic(helsinki_gtfs, kantakaupunki_pbf, ultra_network):
    with pytest.warns(UserWarning):
        again = TransportNetwork.from_gtfs(
            [str(helsinki_gtfs)], osm_pbf=str(kantakaupunki_pbf)
        )
    again.compute_ultra_shortcuts(max_transfer_time=CUTOFF, **WINDOW)
    # The whole set — not merely its size — is reproduced, as the plan's
    # thread-independent determinism requires.
    assert again.ultra_shortcuts == ultra_network.ultra_shortcuts


def test_a_new_street_network_clears_the_shortcuts(helsinki_gtfs, kantakaupunki_pbf):
    # Rebuilding the street network invalidates shortcuts derived from the
    # old one, so recomputing is required rather than silently reusing them.
    from cafein import streets

    with pytest.warns(UserWarning):
        net = TransportNetwork.from_gtfs(
            [str(helsinki_gtfs)], osm_pbf=str(kantakaupunki_pbf)
        )
    net.compute_ultra_shortcuts(max_transfer_time=CUTOFF, **WINDOW)
    assert net.ultra_shortcut_count is not None

    _, street_network = streets.walking_streets(str(kantakaupunki_pbf), net.stops)
    net._core.set_street_network(*street_network)
    assert net.ultra_shortcut_count is None
    assert net.ultra_shortcuts is None


def test_count_is_none_before_computation(network):
    assert network.ultra_shortcut_count is None
    assert network.ultra_shortcuts is None


def test_compute_without_a_street_network_errors(network):
    with pytest.raises(ValueError, match="street network"):
        network.compute_ultra_shortcuts()
