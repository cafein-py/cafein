"""The standalone StreetNetwork: OSM build, coordinate routing, artifacts."""

import math
import multiprocessing

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
def helsinki_network(helsinki_streets):
    return helsinki_streets


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


# ---- Artifacts ----


@pytest.fixture(scope="module")
def artifact(helsinki_network, tmp_path_factory):
    path = tmp_path_factory.mktemp("streets") / "helsinki-streets.cafein"
    helsinki_network.save(path)
    return path


@pytest.fixture(scope="module")
def mmap_available(artifact):
    """Whether this environment can memory-map artifacts."""
    return StreetNetwork.load(artifact, mmap=True).mapped


def _streets_section(path):
    """The (offset, length) of a street artifact's STREETS section."""
    with open(path, "rb") as handle:
        header = handle.read(4096)
    assert header[:8] == b"CAFEINST"
    cursor = 14 + int.from_bytes(header[12:14], "little") + 4
    sections = {}
    for _ in range(2):
        tag = int.from_bytes(header[cursor : cursor + 2], "little")
        offset = int.from_bytes(header[cursor + 2 : cursor + 10], "little")
        length = int.from_bytes(header[cursor + 10 : cursor + 18], "little")
        sections[tag] = (offset, length)
        cursor += 22
    return sections[2]


def test_round_trip_preserves_the_graph_and_its_routes(helsinki_network, artifact):
    loaded = StreetNetwork.load(artifact)
    assert loaded.vertex_count == helsinki_network.vertex_count
    assert loaded.edge_count == helsinki_network.edge_count
    assert not loaded.mapped
    for mode in ("walk", "bicycle", "e_bike", "e_scooter"):
        assert loaded.travel_time(KAMPPI, HAKANIEMI, mode=mode) == (
            helsinki_network.travel_time(KAMPPI, HAKANIEMI, mode=mode)
        )


def test_round_trip_preserves_the_multimodal_permissions(helsinki_network, artifact):
    # A bicycle route only works when the per-arc permissions survive: the
    # profile-aware snap needs them to find a street bikes may use, and the
    # compiler refuses a non-walk profile on a graph with no attributes.
    loaded = StreetNetwork.load(artifact)
    bicycle = loaded.travel_time(KAMPPI, HAKANIEMI, mode="bicycle")
    assert bicycle is not None
    assert bicycle == helsinki_network.travel_time(KAMPPI, HAKANIEMI, mode="bicycle")


def test_mapped_load_routes_identically(helsinki_network, artifact, mmap_available):
    if not mmap_available:
        pytest.skip("memory mapping unavailable in this environment")
    mapped = StreetNetwork.load(artifact, mmap=True)
    assert mapped.mapped
    for mode in ("walk", "bicycle", "e_bike", "e_scooter"):
        assert mapped.travel_time(KAMPPI, HAKANIEMI, mode=mode) == (
            helsinki_network.travel_time(KAMPPI, HAKANIEMI, mode=mode)
        )


def test_mapped_loads_leave_the_core_arrays_unread(artifact, mmap_available):
    # A mapped load reads exactly the optional multimodal arrays, which are
    # decoded owned, and nothing else: the core CSR and geometry stay mapped.
    # Those arrays are 9 bytes per edge — adj_access and adj_facility at one
    # byte per directed arc (2 per edge), the three class codes at one byte per
    # edge, and edge_flags at two — so the expected read is exact, not a bound.
    if not mmap_available:
        pytest.skip("memory mapping unavailable in this environment")
    length = _streets_section(artifact)[1]
    assert length > 0
    mapped = StreetNetwork.load(artifact, mmap=True)
    optional = 9 * mapped.edge_count
    assert mapped._core._streets_bytes_read == optional
    assert optional < length / 4  # the core arrays dominate and stay unread
    owned = StreetNetwork.load(artifact)
    assert owned._core._streets_bytes_read == length
    verified = StreetNetwork.load(artifact, mmap=True, verify=True)
    assert verified._core._streets_bytes_read == length


def test_mmap_require_errors_when_mapping_is_disabled(artifact, monkeypatch):
    monkeypatch.setenv("CAFEIN_DISABLE_MMAP", "1")
    with pytest.raises(ValueError, match="cannot be memory-mapped"):
        StreetNetwork.load(artifact, mmap="require")
    # `auto` falls back to the owned load instead of failing.
    assert not StreetNetwork.load(artifact, mmap=True).mapped


def test_rejects_an_artifact_of_the_wrong_kind(artifact, network, tmp_path):
    from cafein import TransportNetwork

    network_path = tmp_path / "network.cafein"
    network.save(network_path)
    with pytest.raises(ValueError, match="not a cafein street artifact"):
        StreetNetwork.load(network_path)
    with pytest.raises(ValueError, match="not a cafein network artifact"):
        TransportNetwork.load(artifact)


def test_corrupted_payload_fails_its_checksum(artifact, tmp_path):
    damaged = tmp_path / "damaged.cafein"
    payload = bytearray(artifact.read_bytes())
    offset = _streets_section(artifact)[0]
    payload[offset] ^= 0xFF
    damaged.write_bytes(payload)
    with pytest.raises(ValueError, match="checksum mismatch"):
        StreetNetwork.load(damaged)


def test_truncated_artifact_is_refused(artifact, tmp_path):
    truncated = tmp_path / "truncated.cafein"
    truncated.write_bytes(artifact.read_bytes()[:64])
    with pytest.raises(ValueError):
        StreetNetwork.load(truncated)


def test_rejects_an_unknown_mmap_mode(artifact):
    with pytest.raises(ValueError, match="mmap must be"):
        StreetNetwork.load(artifact, mmap="sometimes")


def _mapped_route(args):
    path, mode = args
    network = StreetNetwork.load(path, mmap="require")
    assert network.mapped
    return network.travel_time(KAMPPI, HAKANIEMI, mode=mode)


def test_mapped_artifacts_serve_concurrent_processes(
    helsinki_network, artifact, mmap_available
):
    if not mmap_available:
        pytest.skip("memory mapping unavailable in this environment")
    context = multiprocessing.get_context("spawn")
    with context.Pool(2) as pool:
        results = pool.map(_mapped_route, [(str(artifact), "bicycle")] * 2)
    expected = helsinki_network.travel_time(KAMPPI, HAKANIEMI, mode="bicycle")
    assert results == [expected, expected]


def _two_road_network(near_access, far_access):
    """A walk-only-ish near edge and a parallel far edge, 60 m apart.

    Both run west→east for 200 m; the near edge sits on the query line and the
    far one north of it, so a mode barred from the near edge must snap across
    to the far one. `near_access`/`far_access` are the mode-permission masks.
    """
    from cafein._cafein import StreetNetwork as Core

    per_lat = 111_132.0
    per_lon = 111_320.0 * math.cos(math.radians(60.0))

    def lonlat(x, y):
        return (24.0 + x / per_lon, 60.0 + y / per_lat)

    paths = [
        [lonlat(0.0, 0.0), lonlat(200.0, 0.0)],
        [lonlat(0.0, 60.0), lonlat(200.0, 60.0)],
    ]
    lons, lats, offsets = [], [], [0]
    for path in paths:
        for lon, lat in path:
            lons.append(lon)
            lats.append(lat)
        offsets.append(len(lons))
    zeros = [0, 0]
    return Core(
        4,
        [(0, 1, 200.0), (2, 3, 200.0)],
        offsets,
        lons,
        lats,
        zeros,
        zeros,
        zeros,
        zeros,
        [near_access, far_access],
        [near_access, far_access],
        zeros,
        zeros,
    )


def test_round_trip_preserves_per_arc_permissions(tmp_path):
    # A controlled layout: the near edge admits walking only, the far one also
    # bicycles. A bicycle must therefore snap across to the far edge, so its
    # time differs from walking's. If the reload lost the permissions (or made
    # them permissive) the bicycle would take the near edge and the time would
    # change; if it dropped them entirely, compiling a bicycle profile would
    # fail outright.
    walk_bit, bicycle_bit = 1, 2
    network = StreetNetwork(_two_road_network(walk_bit, walk_bit | bicycle_bit))
    origin, destination = (60.0, 24.0), (59.9999, 24.0 + 200.0 / 55_660.0)
    walk = network.travel_time(origin, destination, mode="walk")
    bicycle = network.travel_time(origin, destination, mode="bicycle")
    assert walk is not None and bicycle is not None
    # The bicycle detours to the far edge, so the two modes disagree.
    assert walk != bicycle

    path = tmp_path / "two-roads.cafein"
    network.save(path)
    for loaded in [StreetNetwork.load(path), StreetNetwork.load(path, mmap=True)]:
        assert loaded.travel_time(origin, destination, mode="walk") == walk
        assert loaded.travel_time(origin, destination, mode="bicycle") == bicycle
