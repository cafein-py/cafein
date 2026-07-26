"""The multimodal union street graph carried by a TransportNetwork."""

import numpy as np
import pytest

pytest.importorskip("cafein._cafein")

from cafein import TransportNetwork  # noqa: E402

KAMPPI, HAKANIEMI = (60.1690, 24.9320), (60.1795, 24.9520)
DATE, DEPARTURE = "2022-02-22", "08:30:00"


def test_the_multimodal_graph_is_carried_with_its_metadata(multimodal_network):
    assert multimodal_network.has_multimodal_streets
    meta = multimodal_network.multimodal_elevation_metadata
    assert meta["source"] == "callable"
    assert meta["nodata_policy"] == "nan"
    assert meta["coverage"] == 1.0
    assert meta["inferred_edges"] > 0  # central Helsinki has bridges


def test_walking_stays_bit_for_bit(multimodal_network, network_with_footpaths):
    # The multimodal graph is a second section beside the walking graph, so
    # every walking query answers exactly what a build without it answers.
    with_streets = multimodal_network.route_between_coordinates(
        KAMPPI, HAKANIEMI, DATE, DEPARTURE
    )
    without = network_with_footpaths.route_between_coordinates(
        KAMPPI, HAKANIEMI, DATE, DEPARTURE
    )
    assert with_streets == without
    stops = [stop for stop, _, _ in network_with_footpaths.stops[:40]]
    assert np.array_equal(
        multimodal_network.travel_time_matrix(stops, DATE, DEPARTURE),
        network_with_footpaths.travel_time_matrix(stops, DATE, DEPARTURE),
    )


def test_the_multimodal_section_round_trips(multimodal_network, tmp_path):
    # The session fixture itself came through one save/load; this pins both
    # load paths explicitly, and that walking queries survive them.
    reference = multimodal_network.route_between_coordinates(
        KAMPPI, HAKANIEMI, DATE, DEPARTURE
    )
    path = tmp_path / "multimodal.cafein"
    multimodal_network.save(path)
    checksum = multimodal_network._core._multimodal_checksum
    assert checksum is not None
    for mmap in (False, True):
        loaded = TransportNetwork.load(path, mmap=mmap)
        assert loaded.has_multimodal_streets
        # The graph's content — CSR, geometry, attributes, elevations —
        # survives both load paths, not just its presence.
        assert loaded._core._multimodal_checksum == checksum
        assert (
            loaded.multimodal_elevation_metadata
            == multimodal_network.multimodal_elevation_metadata
        )
        assert (
            loaded.route_between_coordinates(KAMPPI, HAKANIEMI, DATE, DEPARTURE)
            == reference
        )


def test_the_modes_are_exposed_and_persisted(multimodal_network, tmp_path):
    assert multimodal_network.street_modes == ("walk", "bicycle", "e_scooter")
    path = tmp_path / "modes.cafein"
    multimodal_network.save(path)
    assert TransportNetwork.load(path).street_modes == (
        "walk",
        "bicycle",
        "e_scooter",
    )


def test_a_mapped_load_keeps_the_multimodal_core_mapped(multimodal_network, tmp_path):
    # The mapped load decodes only the optional attribute/elevation arrays;
    # the two graphs' core CSR and geometry stay in the map.
    path = tmp_path / "lazy.cafein"
    multimodal_network.save(path)
    loaded = TransportNetwork.load(path, mmap=True)
    assert loaded.has_multimodal_streets
    read = loaded._core._streets_bytes_read
    assert 0 < read < path.stat().st_size / 4


def test_without_street_modes_nothing_is_carried(network):
    assert not network.has_multimodal_streets
    assert network.multimodal_elevation_metadata is None
    assert network.street_modes is None


def test_a_walk_only_save_is_deterministic(network_with_footpaths, tmp_path):
    # The multimodal machinery adds nothing varying to a build without
    # street_modes — including one that carries the walking street graph:
    # saving the same network twice is byte-identical. (Byte identity across
    # a load cycle is not asserted — feed serialization has never guaranteed
    # it — the walking bit-for-bit tests cover behaviour.)
    first = tmp_path / "first.cafein"
    second = tmp_path / "second.cafein"
    network_with_footpaths.save(first)
    network_with_footpaths.save(second)
    assert first.read_bytes() == second.read_bytes()


def test_street_modes_requires_the_extract(helsinki_gtfs):
    with pytest.raises(ValueError, match="osm_pbf"):
        TransportNetwork.from_gtfs(
            [str(helsinki_gtfs)], street_modes=("walk", "bicycle")
        )


def test_dem_requires_street_modes(helsinki_gtfs, kantakaupunki_pbf):
    with pytest.raises(ValueError, match="street_modes"):
        TransportNetwork.from_gtfs(
            [str(helsinki_gtfs)],
            osm_pbf=str(kantakaupunki_pbf),
            dem=lambda lons, lats: lons,
        )
