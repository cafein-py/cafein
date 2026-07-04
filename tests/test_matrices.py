"""TravelCostMatrix over the Helsinki network shared with r5py."""

import numpy as np
import pytest
import shapely

from cafein import TravelCostMatrix, emissions


def cost_matrix(network, **kwargs):
    # The HSL feed's ferries have no shipped factor; the warning is part
    # of the resolution contract and irrelevant to most assertions.
    with pytest.warns(UserWarning, match="route_type"):
        return TravelCostMatrix(network, date="2022-02-22", **kwargs)


def test_cost_rows_pin_the_k_train(network):
    matrix = cost_matrix(
        network,
        origins=["4810551"],
        destinations=["1250551"],
        departure="08:30:00",
        geometries=True,
    )
    row = matrix.iloc[0]
    assert (row.from_id, row.to_id) == ("4810551", "1250551")
    assert row.travel_time == 28 * 60
    assert row.transfers == 0
    assert row.transit_distance == pytest.approx(16_786, abs=1)
    assert row.walk_distance == 0.0
    # 16.786 km at the shipped 25 g/pkm urban-rail factor.
    assert row.emissions == pytest.approx(419.65, abs=0.1)
    assert row.geometry.geom_type == "MultiLineString"
    assert shapely.get_num_geometries(row.geometry) == 1


def test_cost_rows_count_walks_and_transfers(network_with_footpaths):
    matrix = cost_matrix(
        network_with_footpaths,
        origins=["1100602"],
        destinations=["1040280", "1250551"],
        departure="08:30:00",
    )
    # The 08:31 M2 to Kamppi, then the pinned 20-second footpath.
    m2 = matrix[matrix.to_id == "1040280"].iloc[0]
    assert m2.travel_time == 9 * 60 + 20
    assert m2.transfers == 0
    assert m2.transit_distance == pytest.approx(4_132, abs=1)
    assert 19 <= m2.walk_distance <= 20
    assert m2.emissions == pytest.approx(4.132 * 25, abs=0.1)
    # Reaching the K train takes a second vehicle.
    korso = matrix[matrix.to_id == "1250551"].iloc[0]
    assert korso.transfers >= 1
    assert korso.walk_distance > 100


def test_cost_matrix_matches_the_travel_time_matrix(network):
    origins = ["4810551", "1250551"]
    matrix = cost_matrix(network, origins=origins, departure="08:30:00")
    times = network.travel_time_matrix(origins, "2022-02-22", "08:30:00")
    stop_order = [stop for stop, _, _ in network.stops]
    for row_index, origin in enumerate(origins):
        rows = matrix[matrix.from_id == origin]
        reachable = np.nonzero(times[row_index] != np.uint32(0xFFFFFFFF))[0]
        assert len(rows) == len(reachable)
        expected = {stop_order[at]: int(times[row_index, at]) for at in reachable}
        assert dict(zip(rows.to_id, rows.travel_time)) == expected
    # The origin reaches itself without riding.
    self_row = matrix[(matrix.from_id == "4810551") & (matrix.to_id == "4810551")].iloc[
        0
    ]
    assert self_row.travel_time == 0
    assert self_row.transfers == 0
    assert self_row.transit_distance == 0.0
    assert self_row.emissions == 0.0


def test_cost_matrix_is_deterministic(network):
    origins = [stop for stop, _, _ in network.stops[:32]]
    first = cost_matrix(network, origins=origins, departure="08:30:00")
    second = cost_matrix(network, origins=origins, departure="08:30:00")
    assert first.equals(second)


def test_cost_matrix_requires_installed_payloads(helsinki_gtfs):
    from cafein import TransportNetwork

    lean = TransportNetwork.from_gtfs([str(helsinki_gtfs)], trip_distances=False)
    with pytest.raises(ValueError, match="no trip distances"):
        cost_matrix(lean, origins=["4810551"], departure="08:30:00")
    slim = TransportNetwork.from_gtfs([str(helsinki_gtfs)], leg_geometries=False)
    with pytest.raises(ValueError, match="no leg geometries"):
        cost_matrix(slim, origins=["4810551"], departure="08:30:00", geometries=True)
    # Costs without geometries stay available on the slim build.
    matrix = cost_matrix(slim, origins=["4810551"], departure="08:30:00")
    assert "geometry" not in matrix.columns
    assert len(matrix) > 0


def test_trip_factors_resolve_the_ladder(network):
    with pytest.warns(UserWarning, match="route_type"):
        factors = dict(emissions.trip_factors(network))
    assert len(factors) == network.trip_count
    assert factors["3001K_20220222_S1_2_0831"] == pytest.approx(25.0)
