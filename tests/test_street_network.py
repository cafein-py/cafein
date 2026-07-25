"""The standalone StreetNetwork: OSM build and coordinate routing."""

import pytest

pytest.importorskip("cafein._cafein")

from cafein import StreetNetwork  # noqa: E402

# Two points in central Helsinki, inside the kantakaupunki extract: the
# Kamppi area and Hakaniemi, roughly 1.8 km apart.
KAMPPI = (60.1690, 24.9320)
HAKANIEMI = (60.1795, 24.9520)
# Well outside the extract (mid-Atlantic), so nothing can snap.
NOWHERE = (0.0, -30.0)


@pytest.fixture(scope="module")
def helsinki_network(kantakaupunki_pbf):
    return StreetNetwork.from_osm(str(kantakaupunki_pbf))


def test_from_osm_builds_a_routable_graph(helsinki_network):
    assert helsinki_network.vertex_count > 1000
    assert helsinki_network.edge_count > 1000
    assert "StreetNetwork" in repr(helsinki_network)


def test_cycling_is_faster_than_walking(helsinki_network):
    walk = helsinki_network.travel_time(KAMPPI, HAKANIEMI, mode="walk")
    bicycle = helsinki_network.travel_time(KAMPPI, HAKANIEMI, mode="bicycle")
    assert walk is not None and bicycle is not None
    assert bicycle < walk


def test_every_shipped_mode_routes(helsinki_network):
    for mode in ("walk", "bicycle", "e_bike", "e_scooter"):
        assert helsinki_network.travel_time(KAMPPI, HAKANIEMI, mode=mode) is not None


def test_repeated_queries_reuse_the_cached_profile(helsinki_network):
    # The compiled profile is cached by exact definition; a second query must
    # return the same answer, not a stale or divergent one.
    first = helsinki_network.travel_time(KAMPPI, HAKANIEMI, mode="bicycle")
    second = helsinki_network.travel_time(KAMPPI, HAKANIEMI, mode="bicycle")
    assert first == second


def test_unreachable_within_the_cutoff_is_none(helsinki_network):
    assert (
        helsinki_network.travel_time(KAMPPI, HAKANIEMI, mode="walk", max_time=1.0)
        is None
    )


def test_a_coordinate_routes_to_itself_in_zero_seconds(helsinki_network):
    # The connector is the walk between a coordinate and its street, so routing
    # a point to itself through the network would charge it twice — leaving a
    # point a positive time from itself, and a matrix a non-zero diagonal.
    for mode in ("walk", "bicycle", "e_scooter"):
        assert helsinki_network.travel_time(KAMPPI, KAMPPI, mode=mode) == 0
    # Zero time fits inside any non-negative cutoff.
    assert helsinki_network.travel_time(KAMPPI, KAMPPI, mode="walk", max_time=0.0) == 0
    # An off-network coordinate is still an error, not a silent zero.
    with pytest.raises(ValueError, match="origin"):
        helsinki_network.travel_time(NOWHERE, NOWHERE, mode="walk")


def test_unsnappable_coordinate_raises_rather_than_returning_none(helsinki_network):
    # Unreachable and unsnappable are different answers: one is None, the
    # other names the offending endpoint.
    with pytest.raises(ValueError, match="origin"):
        helsinki_network.travel_time(NOWHERE, HAKANIEMI, mode="walk")
    with pytest.raises(ValueError, match="destination"):
        helsinki_network.travel_time(KAMPPI, NOWHERE, mode="walk")


def test_unknown_mode_raises(helsinki_network):
    with pytest.raises(ValueError, match="unknown street mode"):
        helsinki_network.travel_time(KAMPPI, HAKANIEMI, mode="hovercraft")


def test_modes_selects_pruning_only(kantakaupunki_pbf):
    # `modes=` chooses which profiles get component pruning; it never drops an
    # edge, so the graph is the same size whatever is listed.
    walk_only = StreetNetwork.from_osm(str(kantakaupunki_pbf), modes=("walk",))
    every_mode = StreetNetwork.from_osm(str(kantakaupunki_pbf))
    assert walk_only.edge_count == every_mode.edge_count
    assert walk_only.vertex_count == every_mode.vertex_count
    # A mode left out of `modes` is still routable, just unpruned.
    assert walk_only.travel_time(KAMPPI, HAKANIEMI, mode="bicycle") is not None


def test_from_osm_rejects_an_unknown_mode(kantakaupunki_pbf):
    with pytest.raises(ValueError, match="unknown street mode"):
        StreetNetwork.from_osm(str(kantakaupunki_pbf), modes=("teleport",))
