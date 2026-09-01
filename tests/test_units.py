"""Seconds-to-minutes conversion of the time columns."""

import geopandas as gpd
import pandas as pd
import pytest

from cafein.units import to_minutes


def frame():
    return pd.DataFrame(
        {
            "from_id": ["a", "b"],
            "travel_time_s": [1983, 600],
            "departure_s": [30600, 30600],
            "arrival_s": [32583, 31200],
            "distance_m": [1982.2, 500.0],
            "distance_provenance": ["osm_edge_length+geodesic_connector"] * 2,
        }
    )


def test_converts_every_seconds_column():
    original = frame()
    minutes = to_minutes(original)
    assert list(minutes.columns) == [
        "from_id",
        "travel_time_min",
        "departure_min",
        "arrival_min",
        "distance_m",
        "distance_provenance",
    ]
    assert minutes.travel_time_min.tolist() == [1983 / 60, 10.0]
    assert minutes.departure_min.tolist() == [510.0, 510.0]
    # Non-time columns ride along untouched.
    assert minutes.distance_m.tolist() == [1982.2, 500.0]
    assert "distance_provenance" in minutes.columns
    # And the source frame itself is left untouched.
    assert "travel_time_s" in original.columns
    assert original.travel_time_s.tolist() == [1983, 600]


def test_converts_only_the_named_columns():
    minutes = to_minutes(frame(), ["travel_time"])
    assert "travel_time_min" in minutes.columns
    # The others keep their seconds.
    assert "departure_s" in minutes.columns
    assert "arrival_s" in minutes.columns
    # A column may be named with its suffix too.
    assert "travel_time_min" in to_minutes(frame(), ["travel_time_s"]).columns
    # A non-seconds column refuses by name.
    with pytest.raises(KeyError, match="not a seconds column"):
        to_minutes(frame(), ["distance"])


def test_a_frame_without_seconds_columns_is_returned_unchanged():
    plain = pd.DataFrame({"from_id": ["a"], "distance_m": [1.0]})
    assert list(to_minutes(plain).columns) == ["from_id", "distance_m"]


def test_keeps_the_frame_type_and_geometry():
    # A GeoDataFrame stays one, with its geometry and CRS intact.
    geo = gpd.GeoDataFrame(
        {"travel_time_s": [60]},
        geometry=gpd.points_from_xy([24.9], [60.1]),
        crs="EPSG:4326",
    )
    minutes = to_minutes(geo)
    assert isinstance(minutes, gpd.GeoDataFrame)
    assert minutes.crs == "EPSG:4326"
    assert minutes.travel_time_min.tolist() == [1.0]


def test_works_on_a_slice_of_a_cafein_frame():
    # The point of a function over a method: cafein's frames degrade to plain
    # pandas on any slice, so a method would already be gone by here.
    sliced = frame()[frame().travel_time_s > 900]
    assert to_minutes(sliced).travel_time_min.tolist() == [1983 / 60]
