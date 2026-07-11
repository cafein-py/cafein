"""Saving and loading network artifacts."""

import multiprocessing
import os

import numpy as np
import pytest

from cafein import TransportNetwork

# Central Helsinki, covering the stops the hierarchy test routes between; a
# contraction over this cropped walking graph is seconds, not minutes.
HIERARCHY_BBOX = [24.90, 60.15, 25.00, 60.20]  # [min_lon, min_lat, max_lon, max_lat]


@pytest.fixture(scope="module")
def artifact_path(network_with_footpaths, tmp_path_factory):
    """The street-enabled Helsinki network saved to disk once."""
    path = tmp_path_factory.mktemp("artifact") / "helsinki.cafein"
    network_with_footpaths.save(path)
    return path


@pytest.fixture(scope="module")
def reloaded(artifact_path):
    """The street-enabled Helsinki network, round-tripped through disk."""
    return TransportNetwork.load(artifact_path)


@pytest.fixture(scope="module")
def mmap_available(artifact_path):
    """Whether this environment can memory-map artifacts."""
    return TransportNetwork.load(artifact_path, mmap=True).mapped


def _streets_section(path):
    """The (offset, length) of an artifact's STREETS section."""
    with open(path, "rb") as artifact:
        header = artifact.read(4096)
    assert header[:8] == b"CAFEINET"
    cursor = 14 + int.from_bytes(header[12:14], "little") + 4
    sections = {}
    for _ in range(2):
        tag = int.from_bytes(header[cursor : cursor + 2], "little")
        offset = int.from_bytes(header[cursor + 2 : cursor + 10], "little")
        length = int.from_bytes(header[cursor + 10 : cursor + 18], "little")
        sections[tag] = (offset, length)
        cursor += 22
    return sections[2]


def test_round_trip_preserves_the_network(network_with_footpaths, reloaded):
    assert reloaded.stop_count == network_with_footpaths.stop_count
    assert reloaded.trip_count == network_with_footpaths.trip_count
    assert reloaded.transfer_count == network_with_footpaths.transfer_count
    assert reloaded.stops == network_with_footpaths.stops
    assert (
        reloaded.distance_provenance_counts
        == network_with_footpaths.distance_provenance_counts
    )


def test_round_trip_preserves_routing(network_with_footpaths, reloaded):
    # Stop-to-stop journeys, including transfer legs and WKB geometries.
    before = network_with_footpaths.route_between_stops(
        "1100602", "1040280", "2022-02-22", "08:30:00"
    )
    after = reloaded.route_between_stops("1100602", "1040280", "2022-02-22", "08:30:00")
    assert after == before
    # Door-to-door from coordinates exercises the persisted street index.
    coordinates = {stop: (lat, lon) for stop, lat, lon in reloaded.stops}
    origin = coordinates["1100602"]
    destination = coordinates["1040280"]
    assert reloaded.route_between_coordinates(
        origin, destination, "2022-02-22", "08:30:00"
    ) == network_with_footpaths.route_between_coordinates(
        origin, destination, "2022-02-22", "08:30:00"
    )
    assert reloaded.access_stops(*origin) == network_with_footpaths.access_stops(
        *origin
    )


def test_round_trip_preserves_matrices(network_with_footpaths, reloaded):
    origins = ["4810551", "1100602"]
    assert np.array_equal(
        reloaded.travel_time_matrix(origins, "2022-02-22", "08:30:00"),
        network_with_footpaths.travel_time_matrix(origins, "2022-02-22", "08:30:00"),
    )
    assert np.array_equal(
        reloaded.travel_time_matrix(
            origins, "2022-02-22", "08:30:00", window=600, confidence=0.8
        ),
        network_with_footpaths.travel_time_matrix(
            origins, "2022-02-22", "08:30:00", window=600, confidence=0.8
        ),
    )


def test_loaded_networks_save_again(reloaded, tmp_path):
    path = tmp_path / "again.cafein"
    reloaded.save(path)
    twice = TransportNetwork.load(path)
    assert twice.stop_count == reloaded.stop_count
    assert twice.transfer_count == reloaded.transfer_count
    # The staged temp file was renamed into place, not left behind.
    assert [entry.name for entry in tmp_path.iterdir()] == ["again.cafein"]


def test_round_trip_preserves_the_walking_hierarchy(
    helsinki_gtfs, kantakaupunki_pbf, tmp_path
):
    # An installed contraction hierarchy: the run-once contraction persists and
    # its buckets rebuild on load, so a loaded network's walking searches match a
    # freshly contracted one — not the bounded-Dijkstra fallback of a network
    # that lost its hierarchy. Built over a cropped central walking graph, since
    # the serialize/rebuild path under test does not depend on the graph's size.
    with pytest.warns(UserWarning):
        accelerated = TransportNetwork.from_gtfs(
            [str(helsinki_gtfs)],
            osm_pbf=str(kantakaupunki_pbf),
            bounding_box=HIERARCHY_BBOX,
        )
    accelerated._core.install_walking_hierarchy()
    assert accelerated._core.has_walking_hierarchy

    path = tmp_path / "hierarchy.cafein"
    accelerated.save(path)
    restored = TransportNetwork.load(path)
    assert restored._core.has_walking_hierarchy

    coordinates = {stop: (lat, lon) for stop, lat, lon in restored.stops}
    origin = coordinates["1100602"]
    destination = coordinates["1040280"]
    assert restored.access_stops(*origin) == accelerated.access_stops(*origin)
    assert restored.route_between_coordinates(
        origin, destination, "2022-02-22", "08:30:00"
    ) == accelerated.route_between_coordinates(
        origin, destination, "2022-02-22", "08:30:00"
    )

    # A lazy mapped load restores the hierarchy from META and rebuilds its
    # buckets without paging the STREETS section — validating a hierarchy
    # artifact must not defeat the lazy load. Probe the cropped artifact
    # itself, so this test never builds the shared full-network artifact.
    lazy = TransportNetwork.load(path, mmap=True)
    if lazy.mapped:
        assert lazy._core.has_walking_hierarchy
        assert lazy._core._streets_bytes_read == 0
        assert lazy.access_stops(*origin) == accelerated.access_stops(*origin)


def test_round_trip_preserves_the_mcultra_set(artifact_path, tmp_path):
    # The McULTRA emissions-shortcut set persists with its window and factor
    # fingerprint (a bounded window here; the persistence is identical for any
    # window, and all three ride the same META record).
    from cafein import emissions

    net = TransportNetwork.load(artifact_path)
    assert net._core.mcultra_shortcut_count is None
    factors = emissions.trip_factors(net)
    count = net._core.compute_mcultra_shortcuts(3.6, 300.0, factors, 28800, 29100)
    assert count > 0
    assert net._core.mcultra_shortcut_count == count

    path = tmp_path / "mcultra.cafein"
    net.save(path)
    restored = TransportNetwork.load(path)
    assert restored._core.mcultra_shortcut_count == count
    # The window and factor fingerprint round-trip too (they gate activation).
    assert restored._core.mcultra_window == net._core.mcultra_window == (28800, 29100)
    assert restored._core._mcultra_factor == net._core._mcultra_factor
    assert restored._core._mcultra_factor is not None


def test_load_refuses_foreign_and_future_files(tmp_path):
    junk = tmp_path / "junk.cafein"
    junk.write_bytes(b"definitely not a network artifact")
    with pytest.raises(ValueError, match="not a cafein network artifact"):
        TransportNetwork.load(junk)

    future = tmp_path / "future.cafein"
    future.write_bytes(
        b"CAFEINET" + (999).to_bytes(4, "little") + (5).to_bytes(2, "little") + b"9.9.9"
    )
    with pytest.raises(ValueError, match="format 999"):
        TransportNetwork.load(future)

    with pytest.raises(ValueError):
        TransportNetwork.load(tmp_path / "missing.cafein")


def test_load_refuses_previous_format_versions(tmp_path):
    # A version-3 header (the pre-sectioned format) must name the format
    # and the writing version in its rebuild message.
    old = tmp_path / "v3.cafein"
    old.write_bytes(
        b"CAFEINET" + (3).to_bytes(4, "little") + (5).to_bytes(2, "little") + b"0.2.0"
    )
    with pytest.raises(ValueError, match=r"format 3 \(written by cafein 0\.2\.0\)"):
        TransportNetwork.load(old)


def test_load_refuses_corrupted_payloads(tmp_path):
    from test_transport_network import build_synthetic_gtfs

    feed = build_synthetic_gtfs(tmp_path / "synthetic_gtfs.zip")
    with pytest.warns(UserWarning):
        network = TransportNetwork.from_gtfs([str(feed)])
    path = tmp_path / "small.cafein"
    network.save(path)
    blob = bytearray(path.read_bytes())
    blob[-10] ^= 0xFF
    path.write_bytes(bytes(blob))
    with pytest.raises(ValueError, match="checksum mismatch"):
        TransportNetwork.load(path)


def test_load_refuses_corrupted_street_sections(
    network_with_footpaths, mmap_available, tmp_path
):
    # The last byte of a street-enabled artifact sits in the raw STREETS
    # section; flipping it must fail that section's checksum.
    path = tmp_path / "streets.cafein"
    network_with_footpaths.save(path)
    blob = bytearray(path.read_bytes())
    blob[-1] ^= 0xFF
    path.write_bytes(bytes(blob))
    with pytest.raises(ValueError, match="checksum mismatch"):
        TransportNetwork.load(path)
    # A mapped load checksums the streets only when asked: verify=True
    # detects the corruption, the lazy default trusts the content.
    with pytest.raises(ValueError, match="checksum mismatch"):
        TransportNetwork.load(path, mmap=True, verify=True)
    if mmap_available:
        lazy = TransportNetwork.load(path, mmap=True)
        assert lazy.stop_count == network_with_footpaths.stop_count


def test_load_refuses_truncated_payloads(tmp_path):
    from test_transport_network import build_synthetic_gtfs

    feed = build_synthetic_gtfs(tmp_path / "synthetic_gtfs.zip")
    with pytest.warns(UserWarning):
        network = TransportNetwork.from_gtfs([str(feed)])
    path = tmp_path / "small.cafein"
    network.save(path)
    blob = path.read_bytes()
    path.write_bytes(blob[: len(blob) // 2])
    with pytest.raises(ValueError, match="section bounds"):
        TransportNetwork.load(path)


def test_mapped_loads_match_owned(
    network_with_footpaths, artifact_path, mmap_available
):
    if not mmap_available:
        pytest.skip("memory mapping unavailable in this environment")
    mapped = TransportNetwork.load(artifact_path, mmap=True)
    assert mapped.mapped and not network_with_footpaths.mapped
    coordinates = {stop: (lat, lon) for stop, lat, lon in mapped.stops}
    origin = coordinates["1100602"]
    destination = coordinates["1040280"]
    assert mapped.route_between_coordinates(
        origin, destination, "2022-02-22", "08:30:00"
    ) == network_with_footpaths.route_between_coordinates(
        origin, destination, "2022-02-22", "08:30:00"
    )
    assert mapped.access_stops(*origin) == network_with_footpaths.access_stops(*origin)
    origins = ["4810551", "1100602"]
    assert np.array_equal(
        mapped.travel_time_matrix(origins, "2022-02-22", "08:30:00"),
        network_with_footpaths.travel_time_matrix(origins, "2022-02-22", "08:30:00"),
    )


def test_mapped_loads_fall_back_when_mapping_is_unavailable(artifact_path, monkeypatch):
    monkeypatch.setenv("CAFEIN_DISABLE_MMAP", "1")
    fallback = TransportNetwork.load(artifact_path, mmap=True)
    assert not fallback.mapped
    assert fallback.access_stops(60.169, 24.941)
    with pytest.raises(ValueError, match="CAFEIN_DISABLE_MMAP"):
        TransportNetwork.load(artifact_path, mmap="require")
    with pytest.raises(ValueError, match="mmap must be"):
        TransportNetwork.load(artifact_path, mmap="sometimes")


def test_mapped_loads_read_no_street_bytes(artifact_path, mmap_available):
    if not mmap_available:
        pytest.skip("memory mapping unavailable in this environment")
    length = _streets_section(artifact_path)[1]
    assert length > 0
    mapped = TransportNetwork.load(artifact_path, mmap=True)
    assert mapped._core._streets_bytes_read == 0
    owned = TransportNetwork.load(artifact_path)
    assert owned._core._streets_bytes_read == length
    verified = TransportNetwork.load(artifact_path, mmap=True, verify=True)
    assert verified._core._streets_bytes_read == length


def test_mapped_street_pages_stay_cold_after_load(
    kantakaupunki_pbf, mmap_available, tmp_path
):
    # The strong laziness observable: evict the artifact from the page
    # cache, load it mapped, and require the STREETS pages mostly cold —
    # a loader scan of the section would page it back in wholesale.
    #
    # Kernel readahead is the confounder: any fault near the file head
    # speculatively reads one device window ahead, and a window is
    # bounded only by `read_ahead_kb` and the end of the file. The
    # artifact therefore pairs a tiny synthetic feed with the real
    # street network — META is kilobytes, so the loader faults almost
    # nothing and no readahead stream ramps up — and the test skips on
    # devices whose single window could cover the street section anyway
    # (some CI machines configure tens-of-MB readahead, which pulled the
    # whole section in and is indistinguishable from a scan).
    if not mmap_available or not hasattr(os, "posix_fadvise"):
        pytest.skip("needs memory mapping and posix_fadvise")
    import ctypes
    import mmap as mmap_module

    from cafein import streets
    from test_transport_network import build_synthetic_gtfs

    libc = ctypes.CDLL(None, use_errno=True)
    if not hasattr(libc, "mincore"):
        pytest.skip("needs mincore")
    feed = build_synthetic_gtfs(tmp_path / "synthetic_gtfs.zip")
    with pytest.warns(UserWarning):
        network = TransportNetwork.from_gtfs([str(feed)])
        # The synthetic stops lie outside the extract, so they get no
        # links — irrelevant here; the street arrays carry the extract.
        _, payload = streets.walking_streets(str(kantakaupunki_pbf), network.stops)
    network.set_street_network(*payload)
    path = tmp_path / "streets-heavy.cafein"
    network.save(path)
    offset, length = _streets_section(path)
    page = mmap_module.PAGESIZE

    def resident_street_bytes():
        # A private copy-on-write mapping: never written, so mincore sees
        # the shared page-cache pages, while the buffer stays writable
        # (ctypes.from_buffer refuses the read-only mapping's buffer).
        with open(path, "rb") as artifact:
            view = mmap_module.mmap(
                artifact.fileno(),
                0,
                prot=mmap_module.PROT_READ | mmap_module.PROT_WRITE,
                flags=mmap_module.MAP_PRIVATE,
            )
        try:
            buffer = ctypes.c_char.from_buffer(view)
            pages = (view.size() + page - 1) // page
            flags = (ctypes.c_ubyte * pages)()
            failed = libc.mincore(
                ctypes.c_void_p(ctypes.addressof(buffer)),
                ctypes.c_size_t(view.size()),
                flags,
            )
            del buffer
            if failed:
                pytest.skip("mincore is unavailable")
            first = offset // page
            last = (offset + length + page - 1) // page
            return sum(flag & 1 for flag in flags[first:last]) * page
        finally:
            view.close()

    with open(path, "rb") as artifact:
        os.fsync(artifact.fileno())
        os.posix_fadvise(artifact.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    if resident_street_bytes() > 16 * page:
        pytest.skip("the page cache did not evict the artifact")
    # A residency assertion is only meaningful when a single speculative
    # readahead window cannot cover a substantial part of the section.
    device = os.stat(path).st_dev
    bdi = f"/sys/dev/block/{os.major(device)}:{os.minor(device)}/bdi/read_ahead_kb"
    try:
        with open(bdi) as sysfs:
            read_ahead = int(sysfs.read()) * 1024
    except (OSError, ValueError):
        pytest.skip("the device's readahead window is unknown")
    if read_ahead >= length // 4:
        pytest.skip(f"device readahead ({read_ahead} B) can cover the section")
    network = TransportNetwork.load(path, mmap=True)
    assert network.mapped
    # The loader faults only the tiny META, so at most one readahead
    # window (< a quarter of the section, per the guard above) can spill
    # into STREETS; a scan pages in essentially all of it.
    assert length > 8 * 1024 * 1024
    resident = resident_street_bytes()
    assert resident < length // 2, f"resident {resident} B, readahead {read_ahead} B"


def _mapped_walks(args):
    path, lat, lon = args
    network = TransportNetwork.load(path, mmap="require")
    assert network.mapped
    return network.access_stops(lat, lon)


def test_mapped_artifacts_serve_concurrent_processes(
    network_with_footpaths, artifact_path, mmap_available
):
    if not mmap_available:
        pytest.skip("memory mapping unavailable in this environment")
    coordinates = {stop: (lat, lon) for stop, lat, lon in network_with_footpaths.stops}
    lat, lon = coordinates["1100602"]
    context = multiprocessing.get_context("spawn")
    with context.Pool(2) as pool:
        results = pool.map(_mapped_walks, [(str(artifact_path), lat, lon)] * 2)
    expected = network_with_footpaths.access_stops(lat, lon)
    assert results == [expected, expected]
