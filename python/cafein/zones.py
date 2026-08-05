"""OD zone surfaces: square and H3 hexagonal grids over an area.

Both generators return a polygon GeoDataFrame in EPSG:4326 with ``id``,
``centroid_lat``/``centroid_lon`` (the explicit routing coordinates the
matrix computers consult on polygon frames), and ``geometry`` — ready to
use directly as origins or destinations, routed by centroid.

The ``area`` argument is uniform across both: a ``(west, south, east,
north)`` bbox in EPSG:4326; a polygon GeoDataFrame/GeoSeries (cells kept
by any-intersection with the union of its geometry); or a built
``StreetNetwork``/``TransportNetwork`` (its street extent as the bbox).
"""

import math
import operator

import geopandas
import numpy as np
import pyproj
import shapely


def square_grid(area, cell_size, crs=None):
    """Square cells of ``cell_size`` metres covering ``area``.

    Cells snap to the fixed lattice anchored at the generation CRS's
    origin: a cell's ``column = floor(x / cell_size)`` and ``row =
    floor(y / cell_size)`` from its lower-left corner, with ``id =
    "{column}_{row}"``, columns increasing eastward and rows northward —
    so two overlapping areas gridded in the same CRS produce identical
    cells and ids where they overlap. A bbox or network area covers its
    projected bounds floored/ceiled to ``cell_size`` multiples; a
    polygon area keeps the cells intersecting the union of its geometry.
    Areas touching the antimeridian (±180° longitude) or reaching a
    pole are rejected; null and empty members of a polygon frame are
    ignored.

    Parameters
    ----------
    area : tuple, GeoDataFrame, GeoSeries, StreetNetwork, or TransportNetwork
        A ``(west, south, east, north)`` bbox in EPSG:4326, a polygon
        frame (must carry a CRS), or a built network (street extent).
    cell_size : float
        Cell edge length in metres.
    crs : optional
        The projected, metre-unit CRS to lay the grid out in; defaults
        to the area's local UTM zone (available between -80° and 84°
        latitude — outside that band pass a CRS explicitly, e.g. a
        polar stereographic one). Geographic and non-metre CRSs are
        rejected.

    Returns
    -------
    GeoDataFrame
        Polygon cells in EPSG:4326 with ``id``, ``centroid_lat``,
        ``centroid_lon`` (the cells' planar centres, the coordinates the
        matrix computers route from), and ``geometry``.
    """
    cell = float(cell_size)
    if not (math.isfinite(cell) and cell > 0):
        raise ValueError("cell_size must be a finite, positive number of metres")
    mask, clip = _resolve_area(area)
    _extent_guard(mask, cell)
    generation = _generation_crs(mask, crs)
    mask_generation = (
        geopandas.GeoSeries([mask], crs="EPSG:4326").to_crs(generation).iloc[0]
    )
    # One candidate block per polygon component, deduplicated: a sparse
    # multipolygon never materialises its full bounding rectangle.
    parts = (
        mask_generation.geoms
        if clip and hasattr(mask_generation, "geoms")
        else [mask_generation]
    )
    blocks = []
    for part in parts:
        west, south, east, north = part.bounds
        column_start = math.floor(west / cell)
        column_stop = max(math.ceil(east / cell), column_start + 1)
        row_start = math.floor(south / cell)
        row_stop = max(math.ceil(north / cell), row_start + 1)
        if clip:
            # A boundary on a lattice line touches the cells beyond it,
            # and touching intersects: one extra ring keeps the
            # candidates a superset of every intersecting cell.
            column_start, column_stop = column_start - 1, column_stop + 1
            row_start, row_stop = row_start - 1, row_stop + 1
        blocks.append(
            np.column_stack(
                [
                    np.repeat(
                        np.arange(column_start, column_stop), row_stop - row_start
                    ),
                    np.tile(np.arange(row_start, row_stop), column_stop - column_start),
                ]
            )
        )
    pairs = np.unique(np.concatenate(blocks), axis=0)
    columns, rows = pairs[:, 0], pairs[:, 1]
    xs, ys = columns * cell, rows * cell
    frame = geopandas.GeoDataFrame(
        {"id": [f"{column}_{row}" for column, row in zip(columns, rows)]},
        geometry=shapely.box(xs, ys, xs + cell, ys + cell),
        crs=generation,
    )
    centres = geopandas.GeoSeries(
        geopandas.points_from_xy(xs + cell / 2, ys + cell / 2), crs=generation
    ).to_crs("EPSG:4326")
    frame["centroid_lat"] = centres.y
    frame["centroid_lon"] = centres.x
    if clip:
        frame = frame[frame.geometry.intersects(mask_generation)]
    frame = frame.to_crs("EPSG:4326").reset_index(drop=True)
    return frame[["id", "centroid_lat", "centroid_lon", "geometry"]]


def h3_grid(area, resolution):
    """H3 hexagonal cells at ``resolution`` covering ``area``.

    Keeps every cell intersecting the area (a bbox and a network extent
    behave as their bounding polygon), with ``id`` the canonical H3
    string index and the centroid columns from ``h3.cell_to_latlng`` —
    the cell's canonical centre. Areas touching the antimeridian (±180°
    longitude) or reaching a pole are rejected; null and empty members
    of a polygon frame are ignored. Requires the optional ``h3`` package
    (install ``cafein[h3]``).

    Parameters
    ----------
    area : tuple, GeoDataFrame, GeoSeries, StreetNetwork, or TransportNetwork
        As in `square_grid`.
    resolution : int
        The H3 resolution, ``0``–``15``.

    Returns
    -------
    GeoDataFrame
        Hexagon cells in EPSG:4326 with ``id``, ``centroid_lat``,
        ``centroid_lon``, and ``geometry``, ordered by ``id``.
    """
    h3 = _h3()
    try:
        resolution = operator.index(resolution)
    except TypeError:
        raise TypeError("resolution must be an integer H3 resolution") from None
    if not 0 <= resolution <= 15:
        raise ValueError("resolution must be an H3 resolution between 0 and 15")
    mask, _ = _resolve_area(area)
    # Polyfill keeps centroid-inside cells only; buffering by two edge
    # lengths (comfortably past a cell's circumradius) makes the
    # candidate set a superset of every intersecting cell, filtered back
    # against the unbuffered mask below. The buffer works in degrees at
    # the conservative (largest) per-degree ground length of the area's
    # latitudes, so it oversizes everywhere on the globe away from the
    # guarded antimeridian and poles — no projection involved.
    margin = 2 * h3.average_hexagon_edge_length(resolution, unit="m")
    _extent_guard(mask, margin)
    _, south, _, north = mask.bounds
    latitude = min(max(abs(south), abs(north)) + margin / 110_574, 89.0)
    buffered = mask.buffer(margin / (111_320 * math.cos(math.radians(latitude))))
    cells = sorted(h3.geo_to_cells(buffered, resolution))
    frame = geopandas.GeoDataFrame(
        {"id": cells},
        geometry=[
            shapely.Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(cell)])
            for cell in cells
        ],
        crs="EPSG:4326",
    )
    frame = frame[frame.geometry.intersects(mask)].reset_index(drop=True)
    centres = [h3.cell_to_latlng(cell) for cell in frame["id"]]
    frame["centroid_lat"] = [latitude for latitude, _ in centres]
    frame["centroid_lon"] = [longitude for _, longitude in centres]
    return frame[["id", "centroid_lat", "centroid_lon", "geometry"]]


def _extent_guard(mask, margin_metres):
    """Reject areas whose cells could touch ±180° longitude or a pole —
    cell polygons there would wrap around the world in EPSG:4326."""
    west, south, east, north = mask.bounds
    margin_lat = margin_metres / 110_574
    if north + margin_lat >= 89.9 or south - margin_lat <= -89.9:
        raise ValueError(
            "areas reaching a pole are not supported; keep the area "
            "below ±89.9° latitude (including the cell margin)"
        )
    latitude = min(max(abs(south), abs(north)) + margin_lat, 89.0)
    margin = margin_metres / (111_320 * math.cos(math.radians(latitude)))
    if west - margin <= -180 + 1e-9 or east + margin >= 180 - 1e-9:
        raise ValueError(
            "areas touching the antimeridian (±180° longitude) are not "
            "supported; keep the area clear of it"
        )


def _h3():
    try:
        import h3
    except ImportError as error:
        raise ImportError(
            "h3_grid needs the optional h3 dependency; install " "cafein[h3] or h3"
        ) from error
    return h3


def _resolve_area(area):
    """``area`` as a clip mask: ``(EPSG:4326 geometry, any_intersection)``.

    ``any_intersection`` is True for polygon frames (cells filter by
    intersection with the union) and False for bbox and network extents
    (the whole lattice range is kept).
    """
    bounds = _network_bounds(area)
    if bounds is not None:
        return shapely.box(*bounds), False
    if isinstance(area, (geopandas.GeoDataFrame, geopandas.GeoSeries)):
        geometry = area.geometry if isinstance(area, geopandas.GeoDataFrame) else area
        if area.crs is None:
            raise ValueError("the area polygon frame must carry a CRS")
        geometry = geometry[~(geometry.isna() | geometry.is_empty)]
        if geometry.empty:
            raise ValueError("the area frame has no non-empty geometry")
        kinds = set(geometry.geom_type)
        if not kinds <= {"Polygon", "MultiPolygon"}:
            raise ValueError(
                "the area frame must contain polygons, not " + ", ".join(sorted(kinds))
            )
        return (
            shapely.union_all(geometry.to_crs("EPSG:4326").values),
            True,
        )
    try:
        west, south, east, north = (float(value) for value in area)
    except (TypeError, ValueError):
        raise TypeError(
            "area must be a (west, south, east, north) bbox, a polygon "
            "GeoDataFrame/GeoSeries, or a built StreetNetwork/"
            "TransportNetwork"
        ) from None
    if not (west < east and south < north):
        raise ValueError("the bbox must satisfy west < east and south < north")
    return shapely.box(west, south, east, north), False


def _network_bounds(area):
    """A network's street extent, or ``None`` for non-network areas."""
    core = getattr(area, "_core", None)
    if core is None:
        return None
    if hasattr(core, "_street_coordinate_bounds"):
        bounds = core._street_coordinate_bounds
        if bounds is None:
            raise ValueError(
                "the TransportNetwork has no street network to take an "
                "extent from (build with osm_pbf=)"
            )
        return bounds
    if hasattr(core, "_coordinate_bounds"):
        bounds = core._coordinate_bounds
        if bounds is None:
            raise ValueError("the StreetNetwork stores no coordinates")
        return bounds
    return None


def _generation_crs(mask, crs):
    """The projected metre-unit CRS the grid is laid out in."""
    if crs is None:
        try:
            return geopandas.GeoSeries([mask], crs="EPSG:4326").estimate_utm_crs()
        except RuntimeError:
            raise ValueError(
                "the area has no local UTM zone (UTM covers latitudes "
                "-80° to 84°); pass a projected, metre-unit crs= "
                "explicitly, e.g. a polar stereographic CRS"
            ) from None
    resolved = pyproj.CRS.from_user_input(crs)
    if not resolved.is_projected:
        raise ValueError(
            "crs must be a projected CRS; a geographic one would size "
            "cells in degrees"
        )
    units = {axis.unit_name for axis in resolved.axis_info}
    if units != {"metre"}:
        raise ValueError(
            "crs must use metre units so cell_size means metres, not "
            + ", ".join(sorted(units))
        )
    return resolved
