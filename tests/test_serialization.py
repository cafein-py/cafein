"""Saving and loading network artifacts."""

import numpy as np
import pytest

from cafein import TransportNetwork


@pytest.fixture(scope="module")
def reloaded(network_with_footpaths, tmp_path_factory):
    """The street-enabled Helsinki network, round-tripped through disk."""
    path = tmp_path_factory.mktemp("artifact") / "helsinki.cafein"
    network_with_footpaths.save(path)
    return TransportNetwork.load(path)


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
    # Door-to-door from coordinates exercises the rebuilt street index.
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


def test_load_refuses_truncated_payloads(tmp_path):
    from test_transport_network import build_synthetic_gtfs

    feed = build_synthetic_gtfs(tmp_path / "synthetic_gtfs.zip")
    with pytest.warns(UserWarning):
        network = TransportNetwork.from_gtfs([str(feed)])
    path = tmp_path / "small.cafein"
    network.save(path)
    blob = path.read_bytes()
    path.write_bytes(blob[: len(blob) // 2])
    with pytest.raises(ValueError, match="length mismatch"):
        TransportNetwork.load(path)
