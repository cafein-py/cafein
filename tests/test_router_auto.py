"""router="auto": cache-aware engine selection across the query surface."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from test_frontier import build_two_line_gtfs

DATE, DEPARTURE = "2022-02-22", "08:00:00"
FRONTIER_COLUMNS = [
    "departure",
    "arrival",
    "travel_time",
    "rides",
    "emissions",
    "frontier",
]


@pytest.fixture()
def two_line_network(tmp_path):
    from cafein import TransportNetwork

    feed = build_two_line_gtfs(tmp_path / "two_line_gtfs.zip")
    return TransportNetwork.from_gtfs([str(feed)])


def frontier_frame(network, **kwargs):
    from cafein import journey_frontier

    frame = journey_frontier(network, "A", "B", DATE, DEPARTURE, 1800, **kwargs)
    return frame[FRONTIER_COLUMNS]


def test_auto_matches_raptor_without_a_cached_set(two_line_network):
    assert not two_line_network.has_mctbtr_transfers
    auto = frontier_frame(two_line_network, candidates="pareto")
    raptor = frontier_frame(two_line_network, candidates="pareto", router="raptor")
    assert len(auto) > 0
    assert auto.equals(raptor)


def test_auto_rides_a_matching_cached_mctbtr_set(two_line_network, capfd, monkeypatch):
    from cafein import journey_frontiers

    two_line_network.compute_mctbtr_transfers(DATE)
    # The batched product reports search stats under this flag only when
    # the McTBTR engine actually runs — a direct dispatch observable.
    monkeypatch.setenv("CAFEIN_MCTBTR_PROF", "1")
    batched = journey_frontiers(two_line_network, ["A"], ["B"], DATE, DEPARTURE, 1800)
    assert len(batched) > 0
    assert "MCTBTR-STATS" in capfd.readouterr().err
    auto = frontier_frame(two_line_network, candidates="pareto")
    tbtr = frontier_frame(two_line_network, candidates="pareto", router="tbtr")
    assert len(auto) > 0
    assert auto.equals(tbtr)


def test_auto_misses_the_cache_on_another_date_or_factor_set(
    two_line_network, capfd, monkeypatch
):
    from cafein import journey_frontiers

    two_line_network.compute_mctbtr_transfers(DATE)
    monkeypatch.setenv("CAFEIN_MCTBTR_PROF", "1")
    other_day = journey_frontiers(
        two_line_network, ["A"], ["B"], "2022-02-23", DEPARTURE, 1800
    )
    assert len(other_day) > 0
    assert "MCTBTR-STATS" not in capfd.readouterr().err
    partial = journey_frontiers(
        two_line_network, ["A"], ["B"], DATE, DEPARTURE, 1800, components=["vehicle"]
    )
    assert len(partial) > 0
    assert "MCTBTR-STATS" not in capfd.readouterr().err


def test_raptor_only_features_run_under_the_auto_default(two_line_network):
    # With a matching cached set present auto would pick McTBTR; these
    # queries must resolve to McRAPTOR instead and answer identically to it.
    two_line_network.compute_mctbtr_transfers(DATE)
    for kwargs in (
        {"candidates": "relaxed"},
        {"candidates": "diverse", "max_options": 2},
        {"candidates": "pareto", "max_slower": 0},
    ):
        auto = frontier_frame(two_line_network, **kwargs)
        raptor = frontier_frame(two_line_network, router="raptor", **kwargs)
        assert len(auto) > 0
        assert auto.equals(raptor)


def test_explicit_router_constraints_are_unchanged(two_line_network):
    from cafein import TravelCostMatrix, journey_frontier

    args = (two_line_network, "A", "B", DATE, DEPARTURE, 1800)
    with pytest.raises(ValueError, match="requires candidates='pareto'"):
        journey_frontier(*args, candidates="relaxed", router="tbtr")
    with pytest.raises(ValueError, match="max_slower requires router='raptor'"):
        journey_frontier(*args, candidates="pareto", router="tbtr", max_slower=60)
    with pytest.raises(ValueError, match="'auto', 'raptor', or 'tbtr'"):
        journey_frontier(*args, candidates="pareto", router="fastest")
    with pytest.raises(ValueError, match="requires candidates='pareto'"):
        TravelCostMatrix(two_line_network, ["A"], ["B"], DATE, DEPARTURE, router="tbtr")


def test_auto_time_matrix_matches_both_engines(two_line_network):
    args = (["A", "H"], DATE, DEPARTURE)
    raptor = two_line_network.travel_time_matrix(*args, router="raptor")
    assert np.array_equal(two_line_network.travel_time_matrix(*args), raptor)
    two_line_network.compute_tbtr_transfers(DATE)
    tbtr = two_line_network.travel_time_matrix(*args, router="tbtr")
    assert np.array_equal(two_line_network.travel_time_matrix(*args), tbtr)
    assert np.array_equal(raptor, tbtr)


def test_auto_windowed_matrix_matches_raptor(two_line_network):
    from cafein import TravelTimeMatrix

    kwargs = dict(date=DATE, departure=DEPARTURE, window=1800)
    auto = TravelTimeMatrix(two_line_network, ["A"], **kwargs)
    raptor = TravelTimeMatrix(two_line_network, ["A"], router="raptor", **kwargs)
    assert len(auto) > 0
    assert auto.equals(raptor)


def test_auto_batched_frontiers_match_the_cached_engine(two_line_network):
    from cafein import frontier_table, journey_frontiers

    args = (two_line_network, ["A", "H"], ["B"], DATE, DEPARTURE, 1800)
    two_line_network.compute_mctbtr_transfers(DATE)
    tbtr_rows = journey_frontiers(*args, router="tbtr")
    auto_rows = journey_frontiers(*args)
    assert len(auto_rows) == len(tbtr_rows) > 0
    for column in ("from_id", "to_id", *FRONTIER_COLUMNS):
        assert auto_rows[column].tolist() == tbtr_rows[column].tolist()
    flat_auto = frontier_table(*args)
    flat_tbtr = frontier_table(*args, router="tbtr")
    assert len(flat_auto) > 0
    assert flat_auto.equals(flat_tbtr)


def test_auto_pareto_cost_matrix_matches_the_cached_engine(two_line_network):
    from cafein import TravelCostMatrix

    kwargs = dict(optimize="emissions", candidates="pareto", window=1800)
    args = (two_line_network, ["A"], ["B"], DATE, DEPARTURE)
    raptor = TravelCostMatrix(*args, router="raptor", **kwargs)
    assert TravelCostMatrix(*args, **kwargs).equals(raptor)
    two_line_network.compute_mctbtr_transfers(DATE)
    tbtr = TravelCostMatrix(*args, router="tbtr", **kwargs)
    assert len(tbtr) > 0
    assert TravelCostMatrix(*args, **kwargs).equals(tbtr)


def test_detailed_itineraries_pareto_points_accept_tbtr(network_with_footpaths):
    from cafein import DetailedItineraries

    origins = gpd.GeoDataFrame(
        {"id": ["origin"]}, geometry=[Point(24.9330, 60.1689)], crs="EPSG:4326"
    )
    destinations = gpd.GeoDataFrame(
        {"id": ["destination"]}, geometry=[Point(24.9505, 60.1690)], crs="EPSG:4326"
    )
    kwargs = dict(
        date=DATE, departure="08:30:00", candidates="pareto", geometries=False
    )
    tbtr = DetailedItineraries(
        network_with_footpaths, origins, destinations, router="tbtr", **kwargs
    )
    raptor = DetailedItineraries(
        network_with_footpaths, origins, destinations, router="raptor", **kwargs
    )
    assert len(tbtr) > 0
    assert tbtr.drop(columns="geometry").equals(raptor.drop(columns="geometry"))
