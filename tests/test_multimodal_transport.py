"""The multimodal union street graph carried by a TransportNetwork."""

import numpy as np
import pytest

pytest.importorskip("cafein._cafein")

from cafein import TransportNetwork  # noqa: E402

KAMPPI, HAKANIEMI = (60.1690, 24.9320), (60.1795, 24.9520)
DATE = "2022-02-22"
DEPARTURE = "2022-02-22 08:30:00"


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
        KAMPPI, HAKANIEMI, DEPARTURE
    )
    without = network_with_footpaths.route_between_coordinates(
        KAMPPI, HAKANIEMI, DEPARTURE
    )
    assert with_streets == without
    stops = [stop for stop, _, _ in network_with_footpaths.stops[:40]]
    assert np.array_equal(
        multimodal_network.travel_time_matrix(stops, DEPARTURE),
        network_with_footpaths.travel_time_matrix(stops, DEPARTURE),
    )


def test_the_multimodal_section_round_trips(multimodal_network, tmp_path):
    # The session fixture itself came through one save/load; this pins both
    # load paths explicitly, and that walking queries survive them.
    reference = multimodal_network.route_between_coordinates(
        KAMPPI, HAKANIEMI, DEPARTURE
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
            loaded.route_between_coordinates(KAMPPI, HAKANIEMI, DEPARTURE) == reference
        )


def test_the_modes_are_exposed_and_persisted(multimodal_network, tmp_path):
    assert multimodal_network.street_modes == (
        "walk",
        "bicycle",
        "e_scooter",
        "wheelchair",
    )
    path = tmp_path / "modes.cafein"
    multimodal_network.save(path)
    assert TransportNetwork.load(path).street_modes == (
        "walk",
        "bicycle",
        "e_scooter",
        "wheelchair",
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
    # The multimodal machinery adds nothing varying to a default build
    # (which now carries the walk multimodal graph beside footpaths):
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
    # street_modes=() opts out of the multimodal graph, so a DEM has
    # nothing to apply to; the walk default would otherwise carry it.
    with pytest.raises(ValueError, match="street_modes"):
        TransportNetwork.from_gtfs(
            [str(helsinki_gtfs)],
            osm_pbf=str(kantakaupunki_pbf),
            street_modes=(),
            dem=lambda lons, lats: lons,
        )


def test_an_extract_carries_walking_by_default(network_with_footpaths):
    # Walking is how public-transport journeys begin and end: with an
    # OSM extract the multimodal graph defaults to ("walk",).
    assert network_with_footpaths.has_multimodal_streets
    assert network_with_footpaths.street_modes == ("walk",)


@pytest.fixture(scope="module")
def opt_out_network(helsinki_gtfs, kantakaupunki_pbf):
    """A footpaths network explicitly built without the multimodal graph."""
    with pytest.warns(UserWarning):
        return TransportNetwork.from_gtfs(
            [str(helsinki_gtfs)],
            osm_pbf=str(kantakaupunki_pbf),
            street_modes=(),
        )


def test_empty_street_modes_opt_out(opt_out_network):
    assert not opt_out_network.has_multimodal_streets
    assert opt_out_network.street_modes is None


def test_street_access_and_egress_rows(multimodal_network):
    # One directed search per row over the multimodal graph, mode-masked
    # stop links: cycling reaches far more stops than walking in the same
    # budget and is at least as fast to nearly all shared ones (slope and
    # one-ways explain the rest); the reverse row sees the asymmetry.
    core = multimodal_network._core
    times = lambda rows: {stop: seconds for stop, seconds, *_ in rows}  # noqa: E731
    bike_rows = core._street_access_seconds(60.1690, 24.9320, "bicycle", 900.0)
    bike = times(bike_rows)
    walk = times(core._street_access_seconds(60.1690, 24.9320, "walk", 900.0))
    assert len(bike) > 2 * len(walk) > 0
    shared = set(bike) & set(walk)
    faster = sum(bike[stop] <= walk[stop] for stop in shared)
    assert faster > 0.9 * len(shared)
    egress = times(core._street_egress_seconds(60.1690, 24.9320, "bicycle", 900.0))
    assert len(set(bike) & set(egress)) > 0.8 * len(bike)
    # Every row carries its chosen link's snap identity — the 12c token.
    for _, seconds, edge, fraction, connector in bike_rows:
        assert seconds >= 0 and edge >= 0
        assert 0.0 <= fraction <= 1.0
        assert connector >= 0.0


def test_street_rows_need_the_multimodal_graph(opt_out_network):
    with pytest.raises(ValueError, match="street_modes"):
        opt_out_network._core._street_access_seconds(60.169, 24.932, "bicycle", 900.0)


def test_synthetic_links_keep_modes_apart_and_merge_equal_snaps(
    fresh_footpaths_network,
):
    # A walk+e-scooter edge south of a real stop and a bicycle-only edge
    # north of it, installed directly: the stop's walk and e-scooter links
    # coincide (one merged mask), the bicycle link is the other edge, and
    # both access and egress rows resolve through the right link.
    network = fresh_footpaths_network
    stop, lat, lon = next((s, la, lo) for s, la, lo in network.stops if la is not None)
    south, north = lat - 0.0005, lat + 0.0011
    zeros = [0, 0]
    network._core.set_multimodal_streets(
        ["walk", "bicycle", "e_scooter"],
        4,
        [(0, 1, 200.0), (2, 3, 200.0)],
        [0, 2, 4],
        [lon - 0.001, lon + 0.001, lon - 0.001, lon + 0.001],
        [south, south, north, north],
        zeros,
        zeros,
        zeros,
        zeros,
        [1 | 4, 2],  # south: walk + e_scooter; north: bicycle only
        [1 | 4, 2],
        zeros,
        zeros,
    )
    rows = {
        mode: {
            row[0]: row
            for row in network._core._street_access_seconds(south, lon, mode, 600.0)
        }
        for mode in ("walk", "bicycle", "e_scooter")
    }
    assert stop in rows["walk"] and stop in rows["bicycle"]
    assert stop in rows["e_scooter"]
    walk_edge = rows["walk"][stop][2]
    # The merged mask: walking and the e-scooter share one link snap...
    assert rows["e_scooter"][stop][2] == walk_edge
    # ...while the bicycle links through the other, permitted edge.
    assert rows["bicycle"][stop][2] != walk_edge
    egress = {
        row[0]: row
        for row in network._core._street_egress_seconds(south, lon, "bicycle", 600.0)
    }
    assert egress[stop][2] == rows["bicycle"][stop][2]


def test_car_attributes_round_trip_through_the_artifact(
    fresh_footpaths_network, tmp_path
):
    # A tiny drivable graph installed with the car group: per-edge driving
    # speeds and junction head classes survive save/load on both load paths,
    # stored beside the multimodal attributes.
    network = fresh_footpaths_network
    _, lat, lon = next((s, la, lo) for s, la, lo in network.stops if la is not None)
    zeros = [0, 0]
    network._core.set_multimodal_streets(
        ["walk", "car"],
        4,
        [(0, 1, 200.0), (2, 3, 200.0)],
        [0, 2, 4],
        [lon - 0.001, lon + 0.001, lon - 0.001, lon + 0.001],
        [lat - 0.0005, lat - 0.0005, lat + 0.0011, lat + 0.0011],
        zeros,
        zeros,
        zeros,
        zeros,
        [1 | 8, 8],
        [1, 0],
        zeros,
        zeros,
        car_attributes=([50.0, 40.0], [30.0, 20.0], [3, 1], [0, 4]),
    )
    stored = network._core._multimodal_car_attributes
    assert stored is not None
    speeds, junctions = stored
    assert sorted(speeds) == [20.0, 30.0, 40.0, 50.0]
    assert sorted(junctions) == [0, 1, 3, 4]
    checksum = network._core._multimodal_checksum
    path = tmp_path / "car.cafein"
    network.save(path)
    for mmap in (False, True):
        loaded = TransportNetwork.load(path, mmap=mmap)
        assert loaded._core._multimodal_car_attributes == stored
        assert loaded._core._multimodal_checksum == checksum


def test_malformed_car_payloads_are_refused(fresh_footpaths_network):
    # A zero speed, a junction class outside the vocabulary, and a length
    # mismatch are each refused at installation.
    network = fresh_footpaths_network
    _, lat, lon = next((s, la, lo) for s, la, lo in network.stops if la is not None)
    zeros = [0, 0]
    cases = [
        ([0.0, 40.0], [30.0, 20.0], [0, 0], [0, 0]),
        ([50.0, 40.0], [30.0, 20.0], [5, 0], [0, 0]),
        ([50.0], [30.0, 20.0], [0, 0], [0, 0]),
    ]
    for car in cases:
        with pytest.raises(ValueError):
            network._core.set_multimodal_streets(
                ["walk", "car"],
                4,
                [(0, 1, 200.0), (2, 3, 200.0)],
                [0, 2, 4],
                [lon - 0.001, lon + 0.001, lon - 0.001, lon + 0.001],
                [lat - 0.0005, lat - 0.0005, lat + 0.0011, lat + 0.0011],
                zeros,
                zeros,
                zeros,
                zeros,
                [1 | 8, 8],
                [1, 0],
                zeros,
                zeros,
                car_attributes=car,
            )


def test_multimodal_walk_rows_agree_with_the_walking_graph(multimodal_network):
    # The same coordinate through the multimodal walk profile and through
    # the legacy walking access path: two different extractions of the same
    # streets, so agreement is near, not bit-for-bit — most stops shared,
    # and the shared times close.
    import statistics

    legacy = multimodal_network.access_stops(60.1690, 24.9320)
    rows = multimodal_network._core._street_access_seconds(
        60.1690, 24.9320, "walk", 900.0
    )
    multimodal = {stop: seconds for stop, seconds, *_ in rows}
    within = {stop: s for stop, s in legacy.items() if s <= 900}
    shared = set(within) & set(multimodal)
    assert len(shared) > 0.5 * len(within)
    differences = [abs(multimodal[stop] - within[stop]) for stop in shared]
    assert statistics.median(differences) <= 30


def test_a_query_at_a_stops_coordinate_is_zero_away(multimodal_network):
    # Routing a stop's own coordinate through the network would charge its
    # connector twice; the rows apply the matrix convention instead.
    probe = multimodal_network._core._street_access_seconds(
        60.1690, 24.9320, "bicycle", 900.0
    )
    coordinates = {s: (la, lo) for s, la, lo in multimodal_network.stops}
    stop, seconds, *_ = probe[0]
    assert seconds > 0
    lat, lon = coordinates[stop]
    rows = {
        r[0]: r
        for r in multimodal_network._core._street_access_seconds(
            lat, lon, "bicycle", 900.0
        )
    }
    assert rows[stop][1] == 0
    egress = {
        r[0]: r
        for r in multimodal_network._core._street_egress_seconds(
            lat, lon, "bicycle", 900.0
        )
    }
    assert egress[stop][1] == 0
