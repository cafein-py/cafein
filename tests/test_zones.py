"""OD zone surfaces: the square/H3 generators and polygon matrix intake."""

import sys

import geopandas
import pandas as pd
import pytest
import shapely

from cafein import zones
from cafein.matrices import _point_list

BBOX = (24.90, 60.15, 25.00, 60.20)
CLIP_BOX = shapely.box(24.93, 60.16, 24.97, 60.19)


def test_square_grid_pins_and_determinism():
    grid = zones.square_grid(BBOX, 1000)
    assert list(grid.columns) == ["id", "centroid_lat", "centroid_lon", "geometry"]
    assert str(grid.crs) == "EPSG:4326"
    assert len(grid) == 49
    assert list(grid["id"][:4]) == ["383_6669", "383_6670", "383_6671", "383_6672"]
    assert grid["id"].iloc[-1] == "389_6675"
    assert (grid.geometry.geom_type == "Polygon").all()
    pd.testing.assert_frame_equal(zones.square_grid(BBOX, 1000), grid)


def test_square_grid_lattice_shared_between_overlapping_areas():
    grid = zones.square_grid(BBOX, 1000)
    other = zones.square_grid((24.95, 60.17, 25.05, 60.22), 1000)
    shared = sorted(set(grid["id"]) & set(other["id"]))
    assert len(shared) == 20
    ours = grid.set_index("id").loc[shared]
    theirs = other.set_index("id").loc[shared]
    assert all(a.equals_exact(b, 1e-12) for a, b in zip(ours.geometry, theirs.geometry))
    assert (ours["centroid_lat"] == theirs["centroid_lat"]).all()
    assert (ours["centroid_lon"] == theirs["centroid_lon"]).all()


def test_square_grid_centroids_are_planar_cell_centres():
    import pyproj

    grid = zones.square_grid(BBOX, 1000)
    # The generation CRS at Helsinki is UTM 35N; the id encodes the
    # lattice cell, so its planar centre is computable independently.
    transformer = pyproj.Transformer.from_crs("EPSG:32635", "EPSG:4326", always_xy=True)
    for row in grid.itertuples():
        column, cell_row = (int(part) for part in row.id.split("_"))
        lon, lat = transformer.transform((column + 0.5) * 1000, (cell_row + 0.5) * 1000)
        assert row.centroid_lat == pytest.approx(lat, abs=1e-9)
        assert row.centroid_lon == pytest.approx(lon, abs=1e-9)


def test_square_grid_polygon_clip_keeps_intersecting_cells():
    frame = geopandas.GeoDataFrame(geometry=[CLIP_BOX], crs="EPSG:4326")
    clipped = zones.square_grid(frame, 1000)
    assert len(clipped) == 14
    assert clipped.geometry.intersects(CLIP_BOX).all()
    assert set(clipped["id"]) <= set(zones.square_grid(BBOX, 1000)["id"])


def test_square_grid_crs_handling():
    override = zones.square_grid(BBOX, 1000, crs="EPSG:3857")
    assert len(override) > 0
    with pytest.raises(ValueError, match="projected"):
        zones.square_grid(BBOX, 1000, crs="EPSG:4326")
    with pytest.raises(ValueError, match="metre"):
        zones.square_grid(BBOX, 1000, crs="EPSG:2263")
    with pytest.raises(ValueError, match="CRS"):
        zones.square_grid(geopandas.GeoDataFrame(geometry=[CLIP_BOX]), 1000)
    with pytest.raises(ValueError, match="cell_size"):
        zones.square_grid(BBOX, 0)


def test_square_grid_sparse_multipolygon_stays_small():
    # Distant components must not materialise the union's full bounding
    # rectangle: candidates come per component.
    near = shapely.box(24.90, 60.15, 24.92, 60.17)
    far = shapely.box(27.90, 61.50, 27.92, 61.52)
    frame = geopandas.GeoDataFrame(geometry=[near, far], crs="EPSG:4326")
    grid = zones.square_grid(frame, 100)
    singles = [
        zones.square_grid(geopandas.GeoDataFrame(geometry=[box], crs="EPSG:4326"), 100)
        for box in (near, far)
    ]
    assert len(grid) == sum(len(single) for single in singles)
    assert set(grid["id"]) == set().union(*(set(s["id"]) for s in singles))


def test_square_grid_rejects_non_finite_cell_size():
    for bad in (float("inf"), float("nan")):
        with pytest.raises(ValueError, match="cell_size"):
            zones.square_grid(BBOX, bad)


def test_square_grid_rejects_antimeridian_area():
    frame = geopandas.GeoDataFrame(
        geometry=[shapely.box(179.5, -17.0, 180.0, -16.5)], crs="EPSG:4326"
    )
    with pytest.raises(ValueError, match="antimeridian"):
        zones.square_grid(frame, 1000)


def test_h3_grid_rejects_non_integral_resolution():
    pytest.importorskip("h3")
    with pytest.raises(TypeError, match="integer"):
        zones.h3_grid(BBOX, 8.9)


def test_h3_grid_rejects_antimeridian_area():
    pytest.importorskip("h3")
    with pytest.raises(ValueError, match="antimeridian"):
        zones.h3_grid((178.5, -17.0, 179.9, -16.5), 3)


def test_square_grid_lattice_aligned_polygon_keeps_touching_cells():
    # A polygon exactly on lattice lines: the interior cells are always
    # kept, and any further cell can only be a boundary-touching
    # neighbour from the one-ring candidate expansion.
    aligned = geopandas.GeoDataFrame(
        geometry=[shapely.box(383000, 6669000, 385000, 6671000)],
        crs="EPSG:32635",
    )
    grid = zones.square_grid(aligned, 1000, crs="EPSG:32635")
    interior = {"383_6669", "383_6670", "384_6669", "384_6670"}
    ring = {
        f"{column}_{row}" for column in range(382, 386) for row in range(6668, 6672)
    }
    assert interior <= set(grid["id"]) <= ring


def test_square_grid_ignores_null_and_empty_area_members():
    frame = geopandas.GeoDataFrame(
        geometry=[CLIP_BOX, shapely.Polygon(), None], crs="EPSG:4326"
    )
    grid = zones.square_grid(frame, 1000)
    only = geopandas.GeoDataFrame(geometry=[CLIP_BOX], crs="EPSG:4326")
    pd.testing.assert_frame_equal(grid, zones.square_grid(only, 1000))
    hollow = geopandas.GeoDataFrame(geometry=[shapely.Polygon(), None], crs="EPSG:4326")
    with pytest.raises(ValueError, match="non-empty"):
        zones.square_grid(hollow, 1000)


def test_square_grid_beyond_utm_band_needs_explicit_crs():
    arctic = (0.0, 85.0, 10.0, 86.0)
    with pytest.raises(ValueError, match="UTM"):
        zones.square_grid(arctic, 1000)
    polar = zones.square_grid(arctic, 100000, crs="EPSG:3413")
    assert len(polar) > 0


def test_grids_reject_polar_areas():
    with pytest.raises(ValueError, match="pole"):
        zones.square_grid((0.0, 89.85, 10.0, 89.95), 1000)
    pytest.importorskip("h3")
    with pytest.raises(ValueError, match="pole"):
        zones.h3_grid((0.0, 70.0, 30.0, 75.0), 0)


def test_h3_grid_covers_wide_multi_zone_areas():
    h3 = pytest.importorskip("h3")
    bbox = (0.0, 50.0, 40.0, 55.0)
    hexes = zones.h3_grid(bbox, 3)
    area = shapely.box(*bbox)
    assert hexes.geometry.intersects(area).all()
    inside = h3.geo_to_cells(area, 3)
    assert set(inside) <= set(hexes["id"])


def test_area_validation():
    with pytest.raises(ValueError, match="west < east"):
        zones.square_grid((25.00, 60.15, 24.90, 60.20), 1000)
    with pytest.raises(TypeError, match="bbox"):
        zones.square_grid("everywhere", 1000)
    lines = geopandas.GeoDataFrame(
        geometry=[shapely.LineString([(24.9, 60.15), (25.0, 60.2)])],
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="polygons"):
        zones.square_grid(lines, 1000)


def test_square_grid_from_street_network(helsinki_streets):
    bounds = helsinki_streets._core._coordinate_bounds
    assert bounds is not None
    west, south, east, north = bounds
    grid = zones.square_grid(helsinki_streets, 2000)
    minx, miny, maxx, maxy = grid.total_bounds
    assert minx <= west and miny <= south
    assert maxx >= east and maxy >= north
    assert len(grid) > 0


def test_square_grid_from_transport_network(network, network_with_footpaths):
    grid = zones.square_grid(network_with_footpaths, 2000)
    assert len(grid) > 0
    with pytest.raises(ValueError, match="street network"):
        zones.square_grid(network, 2000)


def test_h3_grid_pins():
    h3 = pytest.importorskip("h3")
    hexes = zones.h3_grid(BBOX, 8)
    assert list(hexes.columns) == ["id", "centroid_lat", "centroid_lon", "geometry"]
    assert len(hexes) == 73
    assert list(hexes["id"][:3]) == [
        "881126d005fffff",
        "881126d007fffff",
        "881126d00dfffff",
    ]
    assert list(hexes["id"]) == sorted(hexes["id"])
    assert all(h3.is_valid_cell(cell) for cell in hexes["id"])
    assert all(h3.int_to_str(h3.str_to_int(cell)) == cell for cell in hexes["id"])
    for row in hexes.itertuples():
        latitude, longitude = h3.cell_to_latlng(row.id)
        assert row.centroid_lat == latitude
        assert row.centroid_lon == longitude


def test_h3_grid_boundary_geometry():
    h3 = pytest.importorskip("h3")
    hexes = zones.h3_grid(BBOX, 8)
    cell = hexes["id"].iloc[0]
    expected = shapely.Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(cell)])
    assert hexes.geometry.iloc[0].equals_exact(expected, 1e-12)


def test_h3_grid_polygon_clip():
    pytest.importorskip("h3")
    frame = geopandas.GeoDataFrame(geometry=[CLIP_BOX], crs="EPSG:4326")
    clipped = zones.h3_grid(frame, 8)
    assert len(clipped) == 23
    assert clipped.geometry.intersects(CLIP_BOX).all()
    assert set(clipped["id"]) <= set(zones.h3_grid(BBOX, 8)["id"])


def test_h3_grid_resolution_validation():
    pytest.importorskip("h3")
    for resolution in (-1, 16):
        with pytest.raises(ValueError, match="resolution"):
            zones.h3_grid(BBOX, resolution)


def test_h3_grid_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "h3", None)
    with pytest.raises(ImportError, match=r"cafein\[h3\]"):
        zones.h3_grid(BBOX, 8)


def test_polygon_intake_uses_centroid_columns():
    grid = zones.square_grid(BBOX, 1000)
    ids, points = _point_list(grid, "origins")
    assert ids == list(grid["id"])
    assert points == list(zip(grid["centroid_lat"], grid["centroid_lon"]))


def test_polygon_intake_computes_utm_centroids():
    grid = zones.square_grid(BBOX, 1000)
    bare = grid[["id", "geometry"]].copy()
    ids, points = _point_list(bare, "origins")
    assert ids == list(grid["id"])
    for (lat, lon), row in zip(points, grid.itertuples()):
        assert lat == pytest.approx(row.centroid_lat, abs=1e-9)
        assert lon == pytest.approx(row.centroid_lon, abs=1e-9)


def test_polygon_intake_rejections():
    grid = zones.square_grid(BBOX, 1000)
    bare = grid[["id", "geometry"]].copy()
    bare.crs = None
    with pytest.raises(ValueError, match="CRS"):
        _point_list(bare, "origins")
    mixed = geopandas.GeoDataFrame(
        {"id": ["a", "b"]},
        geometry=[shapely.Point(24.94, 60.17), CLIP_BOX],
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="points"):
        _point_list(mixed, "origins")


def test_zones_route_as_centroid_points(network_with_footpaths):
    from cafein import TravelTimeMatrix

    frame = geopandas.GeoDataFrame(geometry=[CLIP_BOX], crs="EPSG:4326")
    grid = zones.square_grid(frame, 1000)
    points = geopandas.GeoDataFrame(
        {"id": grid["id"]},
        geometry=geopandas.points_from_xy(grid["centroid_lon"], grid["centroid_lat"]),
        crs="EPSG:4326",
    )
    as_zones = TravelTimeMatrix(
        network_with_footpaths, origins=grid, date="2022-02-22", departure="08:30:00"
    )
    as_points = TravelTimeMatrix(
        network_with_footpaths, origins=points, date="2022-02-22", departure="08:30:00"
    )
    pd.testing.assert_frame_equal(pd.DataFrame(as_zones), pd.DataFrame(as_points))
