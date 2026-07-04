"""Per-trip travel distances from GTFS data — the fallback ladder.

Tiers 1, 2, and 5 of the distance ladder, applied per trip at
preprocessing time: validated ``shape_dist_traveled`` values straight
from the feed (``shape_dist``), stops linear-referenced onto the feed's
shape geometry (``shape_linref``), and great-circle distances scaled by a
mode detour coefficient (``crow_fly``). Any validation failing drops the
trip to the next tier. The OSM tiers (route-relation matching, map
matching) arrive with the OSM preprocessing.
"""

import os
import pathlib
import warnings
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely

SHAPE_DIST = "shape_dist"
SHAPE_LINREF = "shape_linref"
CROW_FLY = "crow_fly"

SNAP_TOLERANCE = 100.0
"""Tier-2 validation: maximum stop-to-shape distance, in meters."""

EARTH_RADIUS = 6_371_000.0


def trip_distances(gtfs_paths, include=None, geometries=False):
    """Compute cumulative travel distances for the feeds' trips.

    Parameters
    ----------
    gtfs_paths : path or list of paths
        GTFS zip files or directories, in the order given to
        ``TransportNetwork.from_gtfs``.
    include : set of str (optional)
        Public trip identifiers to compute; other trips are skipped.
        Passing the network's routable trips keeps quarantined trips
        with broken geometry from failing the preprocessing.
    geometries : bool (optional, default: False)
        Also produce the per-trip leg-geometry payload from the same
        pass over the feeds.

    Returns
    -------
    list of (str, list of float, str), or (list, (polylines, trips))
        Without `geometries`, the ``(trip_id, cumulative_meters,
        provenance)`` rows suitable for
        ``TransportNetwork.set_trip_distances``: one cumulative distance
        per stop of the trip, and the ladder tier of the estimate
        (``shape_dist``, ``shape_linref``, or ``crow_fly``). Trip
        identifiers are feed-qualified when several feeds are given.

        With `geometries`, a ``(distances, geometry)`` pair where
        ``geometry`` is the argument tuple of
        ``TransportNetwork.set_leg_geometries``: deduplicated
        ``polylines`` as ``(longitudes, latitudes, measures)`` triples —
        the shape when the trip has one that its stops lie along,
        otherwise the straight stop chain — and ``trips`` as
        ``(trip_id, polyline, stop_positions)`` rows locating each stop
        of the trip on its polyline, in the polyline's measure.
    """
    if isinstance(gtfs_paths, (str, os.PathLike)):
        gtfs_paths = [gtfs_paths]
    qualify = len(gtfs_paths) > 1
    results = []
    polylines = []
    trips = []
    for feed, path in enumerate(gtfs_paths):
        if include is None:
            local = None
        elif qualify:
            prefix = f"{feed}:"
            local = {t[len(prefix) :] for t in include if t.startswith(prefix)}
        else:
            local = set(include)
        rows, feed_polylines, feed_trips = _feed_trip_distances(path, local, geometries)
        base = len(polylines)
        polylines.extend(feed_polylines)
        for trip_id, cumulative, tier in rows:
            public = f"{feed}:{trip_id}" if qualify else trip_id
            results.append((public, cumulative, tier))
        for trip_id, polyline, positions in feed_trips:
            public = f"{feed}:{trip_id}" if qualify else trip_id
            trips.append((public, base + polyline, positions))
    if geometries:
        return results, (polylines, trips)
    return results


def _detour(route_type):
    """Crow-fly detour coefficient of a GTFS route type (extended types
    map to their base mode); rail-bound modes run close to straight."""
    if route_type == 2 or 100 <= route_type < 200:  # rail
        return 1.2
    if route_type == 1 or 400 <= route_type < 500:  # metro
        return 1.2
    if route_type == 0 or 900 <= route_type < 1000:  # tram
        return 1.25
    if route_type == 4 or 1000 <= route_type < 1100:  # ferry
        return 1.2
    return 1.4  # buses and everything else


def _read_table(path, name, **kwargs):
    path = pathlib.Path(path)
    if path.is_dir():
        member = path / name
        return pd.read_csv(member, **kwargs) if member.exists() else None
    with zipfile.ZipFile(path) as archive:
        if name not in archive.namelist():
            return None
        with archive.open(name) as member:
            return pd.read_csv(member, **kwargs)


def _feed_trip_distances(path, include=None, geometries=False):
    """The ladder over one feed: one distance row per included trip, plus
    the feed's deduplicated leg-geometry payload when requested."""
    stop_times = _read_table(
        path,
        "stop_times.txt",
        usecols=lambda column: column
        in {"trip_id", "stop_id", "stop_sequence", "shape_dist_traveled"},
        dtype={"trip_id": str, "stop_id": str},
    ).sort_values(["trip_id", "stop_sequence"], kind="stable")
    if "shape_dist_traveled" not in stop_times:
        stop_times["shape_dist_traveled"] = np.nan
    # Malformed values become NaN and fail tier 1's validation normally
    # instead of aborting the feed.
    stop_times["shape_dist_traveled"] = pd.to_numeric(
        stop_times["shape_dist_traveled"], errors="coerce"
    )
    trips = _read_table(path, "trips.txt", dtype=str).set_index("trip_id")
    routes = _read_table(path, "routes.txt", dtype={"route_id": str})
    route_types = routes.set_index("route_id")["route_type"].astype(int)
    stops = _read_table(path, "stops.txt", dtype={"stop_id": str})
    coordinates = (
        stops.drop_duplicates("stop_id")
        .set_index("stop_id")[["stop_lat", "stop_lon"]]
        .astype(float)
    )

    trip_column = stop_times["trip_id"].to_numpy()
    boundaries = np.flatnonzero(np.r_[True, trip_column[1:] != trip_column[:-1]])
    trip_ids = trip_column[boundaries]
    stop_arrays = np.split(stop_times["stop_id"].to_numpy(), boundaries[1:])
    dist_arrays = np.split(
        stop_times["shape_dist_traveled"].to_numpy(float), boundaries[1:]
    )

    shape_of = trips["shape_id"] if "shape_id" in trips else pd.Series(dtype=object)
    route_of = trips["route_id"]

    # Tier 2's shape geometries load lazily on first use: a feed that
    # resolves fully at tier 1 never touches shapes.txt.
    loaded = []

    def shapes():
        if not loaded:
            loaded.append(_shape_lines(path, stops))
        return loaded[0]

    # Trips sharing shape, mode, stop sequence, and raw distances are one
    # unit of work: compute each unit once, broadcast to its trips. The
    # cache lives within one feed, where a stop id pins its coordinates,
    # so ids can stand in for the coordinates in the key.
    cache = {}
    results = []
    polyline_index = {}
    polylines = []
    trip_geometries = []
    for index, trip_id in enumerate(trip_ids):
        if include is not None and trip_id not in include:
            continue
        shape_id = shape_of.get(trip_id)
        if pd.isna(shape_id):
            shape_id = None
        route_type = int(route_types[route_of[trip_id]])
        stop_ids = tuple(stop_arrays[index])
        key = (shape_id, route_type, stop_ids, dist_arrays[index].tobytes())
        if key not in cache:
            cache[key] = _ladder(
                shape_id,
                route_type,
                stop_ids,
                dist_arrays[index],
                coordinates,
                shapes,
                geometries,
            )
        cumulative, tier, geometry = cache[key]
        results.append((trip_id, cumulative, tier))
        if geometry is None:
            continue
        if geometry[0] == "shape":
            _, used_shape, positions = geometry
            pkey = ("shape", used_shape)
            if pkey not in polyline_index:
                polyline_index[pkey] = len(polylines)
                line, (lon, lat) = shapes()[0][used_shape]
                polylines.append((lon.tolist(), lat.tolist(), _measures(line)))
        else:
            _, positions = geometry
            pkey = ("chain", stop_ids)
            if pkey not in polyline_index:
                polyline_index[pkey] = len(polylines)
                latlon = coordinates.loc[list(stop_ids)].to_numpy()
                polylines.append(
                    (latlon[:, 1].tolist(), latlon[:, 0].tolist(), positions)
                )
        trip_geometries.append((trip_id, polyline_index[pkey], positions))
    return results, polylines, trip_geometries


def _measures(line):
    """Cumulative meters at each vertex of a projected LineString."""
    coordinates = shapely.get_coordinates(line)
    hops = np.hypot(*np.diff(coordinates, axis=0).T)
    return np.concatenate([[0.0], np.cumsum(hops)]).tolist()


def _ladder(
    shape_id, route_type, stop_ids, raw_distances, coordinates, shapes, geometries
):
    """Best-tier-wins distances for one (shape, mode, stop sequence).

    With `geometries`, also how to draw its legs: the shape with the
    stops located along it whenever the stops verifiably lie on it,
    otherwise the straight stop chain.
    """
    latlon = coordinates.loc[list(stop_ids)].to_numpy()
    located = np.isfinite(latlon).all(axis=1)
    if not located.all():
        missing = [sid for sid, ok in zip(stop_ids, located) if not ok]
        raise ValueError(
            "stop(s) without coordinates in a trip's stop sequence: "
            + ", ".join(missing[:5])
        )
    crow = _crow_fly_cumulative(latlon)

    geometry = None
    if geometries:
        geometry = ("chain", crow.tolist())
        if shape_id is not None and shapes() is not None:
            lines, transformer = shapes()
            if shape_id in lines:
                along = _locate_on_shape(lines[shape_id][0], latlon, transformer)
                if along is not None:
                    geometry = ("shape", shape_id, along.tolist())

    cumulative = _from_shape_dist(raw_distances, crow[-1])
    if cumulative is not None:
        return cumulative.tolist(), SHAPE_DIST, geometry
    if shape_id is not None and shapes() is not None:
        lines, transformer = shapes()
        if shape_id in lines:
            cumulative = _linear_referenced(lines[shape_id][0], latlon, transformer)
            if cumulative is not None:
                return cumulative.tolist(), SHAPE_LINREF, geometry
    return (crow * _detour(route_type)).tolist(), CROW_FLY, geometry


def _crow_fly_cumulative(latlon):
    """Cumulative great-circle meters along consecutive stops."""
    lat = np.radians(latlon[:, 0])
    lon = np.radians(latlon[:, 1])
    half = (
        np.sin(np.diff(lat) / 2) ** 2
        + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(np.diff(lon) / 2) ** 2
    )
    hops = 2 * EARTH_RADIUS * np.arcsin(np.sqrt(half))
    return np.concatenate([[0.0], np.cumsum(hops)])


def _from_shape_dist(distances, crow_total):
    """Tier 1: validated ``shape_dist_traveled``, unit-corrected to meters.

    The total is compared against the crow-fly trip length: a plausible
    detour ratio means meters, a thousandth of that means the feed
    records kilometers (some feeds do), anything else fails the tier.
    """
    if np.isnan(distances).any() or (np.diff(distances) < 0).any():
        return None
    total = distances[-1] - distances[0]
    if total <= 0 or crow_total <= 0:
        return None
    ratio = total / crow_total
    if 0.8 <= ratio <= 5:
        scale = 1.0
    elif 0.8e-3 <= ratio <= 5e-3:
        scale = 1000.0
    else:
        return None
    return (distances - distances[0]) * scale


def _linear_referenced(line, latlon, transformer):
    """Tier 2: stops linear-referenced onto the shape, validated.

    On top of `_locate_on_shape`'s checks, the shape must be denser than
    the stop sequence — otherwise it is stop-to-stop straight lines
    dressed up as a shape and adds no distance information.
    """
    if shapely.get_num_coordinates(line) <= len(latlon):
        return None
    along = _locate_on_shape(line, latlon, transformer)
    if along is None:
        return None
    return along - along[0]


def _locate_on_shape(line, latlon, transformer):
    """The stops' absolute positions along a projected shape, validated:
    every stop must lie near the shape, and the positions must be
    monotone (guards against self-intersecting shapes assigning stops
    out of order)."""
    x, y = transformer.transform(latlon[:, 1], latlon[:, 0])
    points = shapely.points(x, y)
    offsets = shapely.distance(line, points)
    if not np.isfinite(offsets).all() or (offsets > SNAP_TOLERANCE).any():
        return None
    along = shapely.line_locate_point(line, points)
    if not np.isfinite(along).all():
        return None
    if (np.diff(along) < 0).any() or along[-1] <= along[0]:
        return None
    return along


def _shape_lines(path, stops):
    """UTM LineStrings per shape_id, with the lon/lat→UTM transformer.

    Returns ``None`` when the feed carries no usable shapes; the shape
    tier is optional by design, so a malformed shapes.txt degrades the
    tier (with a warning) instead of failing the feed.
    """
    try:
        shapes = _read_table(path, "shapes.txt", dtype={"shape_id": str})
        if shapes is None or shapes.empty:
            return None
        # Sequences must sort numerically, and points must have real
        # coordinates; unusable rows are dropped (the tier-2 validations
        # still guard the result).
        for column in ["shape_pt_sequence", "shape_pt_lat", "shape_pt_lon"]:
            shapes[column] = pd.to_numeric(shapes[column], errors="coerce")
        shapes = shapes.dropna(
            subset=["shape_pt_sequence", "shape_pt_lat", "shape_pt_lon"]
        )
        crs = gpd.GeoSeries(
            gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]), crs="EPSG:4326"
        ).estimate_utm_crs()
        transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        x, y = transformer.transform(
            shapes["shape_pt_lon"].to_numpy(), shapes["shape_pt_lat"].to_numpy()
        )
        ordered = shapes.assign(x=x, y=y).sort_values(
            ["shape_id", "shape_pt_sequence"], kind="stable"
        )
        lines = {}
        for shape_id, group in ordered.groupby("shape_id", sort=False):
            if len(group) >= 2:
                lines[shape_id] = (
                    shapely.LineString(np.column_stack([group["x"], group["y"]])),
                    (
                        group["shape_pt_lon"].to_numpy(float),
                        group["shape_pt_lat"].to_numpy(float),
                    ),
                )
        return lines, transformer
    except Exception as error:
        warnings.warn(
            f"shapes could not be used ({error}); trips fall past the shape tier",
            stacklevel=2,
        )
        return None
