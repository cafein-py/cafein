"""Environmental exposure layers on the street network.

``Exposure`` attaches user-named layers — raster bands or vector
value columns, cafein assigns no meaning to the names — to the
street network's edges: a coverage-adjusted (zero-filled) dose mean,
a covered maximum, a coverage share, and one at-or-above length
share per declared threshold, all visible as columns of
``TransportNetwork.streets_gdf``. Ingestion runs once, eagerly, at
construction; routing never touches geometry or GDAL again.

Raster and line ingestion are sampling estimates: values are read at
the midpoints of ``ceil(length / 25 m)`` equal subdivisions of each
edge. Polygon layers are BY DEFAULT rasterized onto a 1 m grid
(``all_touched``, overlaps burned max-wins) and read through the same
samples — a resolution-bounded estimate roughly two orders of
magnitude faster than exact overlay on real zone layers;
``rasterize=None`` opts into the exact length-true overlay with
worst-wins interval claiming (exact on simple edges; self-crossing
or retraced edges claim through a dense-sampling fallback).
"""

from __future__ import annotations

import math
import os
import re

import numpy as np

SAMPLE_STEP_M = 25.0
"""Along-edge sampling step for raster and line layers, metres."""

LINE_MATCH_TOLERANCE_M = 25.0
"""How far a sample may sit from a line feature and still take its
value. Fixed in this arc."""

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_VALUE = "_cafein_exposure_value"
"""The private value column the spatial joins carry — user column
names never meet the join machinery."""
_DERIVED = ("", "_coverage", "_max")

BURN_STRIP_CELLS = 32_000_000
"""Cells per rasterization strip (float32: ~128 MB peak), bounding
memory at metro scale regardless of region size."""


def _threshold_suffix(value):
    """``55`` -> ``"55"``, ``7.5`` -> ``"7_5"``, ``-5`` -> ``"minus5"``.

    Round-trip-safe: ``repr`` of a float is shortest-round-trip, so
    distinct thresholds always render distinct suffixes and two legal
    thresholds can never collide on a column name.
    """
    value = float(value) + 0.0  # normalizes -0.0 to 0.0
    text = str(int(value)) if value.is_integer() else repr(value)
    return text.replace(".", "_").replace("-", "minus").replace("+", "plus")


def _derived_columns(name, thresholds):
    columns = [name + suffix for suffix in _DERIVED]
    for threshold in thresholds:
        columns.append(f"{name}_share_above_{_threshold_suffix(threshold)}")
    return columns


def midpoint_samples(line, step=SAMPLE_STEP_M):
    """Sample points at the midpoints of equal subdivisions of `line`.

    ``n = ceil(length / step)`` subdivisions; every sample represents
    an equal share of the line's length.
    """
    n = max(1, math.ceil(line.length / step))
    return [line.interpolate((i + 0.5) * line.length / n) for i in range(n)]


def _piece_intervals(line, piece):
    """A piece's lineal components as ``[start, end]`` parameters
    along `line`. Interval space is immune to the coordinate noise
    that defeats GEOS union/difference on nearly-coincident linework
    (real zone layers produce exactly that).

    Components parameterize by their MIDPOINT plus half-length to
    each side: endpoint projection is ambiguous on closed edges (both
    endpoints of a full-loop piece project to 0), while a midpoint is
    unambiguous and the interval wraps around the seam of a closed
    line as two pieces.
    """
    import shapely

    total = line.length
    closed = line.is_closed
    intervals = []
    for part in (
        shapely.get_parts(piece)
        if piece.geom_type.startswith(("Multi", "Geometry"))
        else [piece]
    ):
        if part.geom_type != "LineString" or part.length == 0:
            continue
        half = float(part.length) / 2
        midpoint = part.interpolate(0.5, normalized=True)
        center = float(line.project(midpoint))
        low, high = center - half, center + half
        if closed:
            if low < 0:
                intervals.extend([(0.0, high), (total + low, total)])
            elif high > total:
                intervals.extend([(low, total), (0.0, high - total)])
            else:
                intervals.append((low, high))
        else:
            low, high = max(0.0, low), min(total, high)
            if high > low:
                intervals.append((low, high))
    return intervals


def _subtract_intervals(intervals, claimed):
    """`intervals` minus `claimed`, both sorted-merged interval lists."""
    remaining = []
    for low, high in intervals:
        cursor = low
        for c_low, c_high in claimed:
            if c_high <= cursor or c_low >= high:
                continue
            if c_low > cursor:
                remaining.append((cursor, min(c_low, high)))
            cursor = max(cursor, c_high)
            if cursor >= high:
                break
        if cursor < high:
            remaining.append((cursor, high))
    return [(low, high) for low, high in remaining if high > low]


def _merge_intervals(intervals):
    merged = []
    for low, high in sorted(intervals):
        if merged and low <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], high))
        else:
            merged.append((low, high))
    return merged


def _sampled_claim(line, pieces):
    """Worst-wins claiming by dense sampling — the fallback for
    self-crossing or retraced edges, where any 1-D parameterization is
    ambiguous (projection cannot tell traversals apart). Exact in the
    sampling limit; only degenerate edges take this path."""
    import shapely

    step = min(1.0, max(0.05, line.length / 128))
    points = np.asarray(midpoint_samples(line, step=step), dtype=object)
    best = np.full(len(points), -np.inf)
    for value, piece in pieces:
        near = shapely.distance(points, piece) < 1e-6
        best[near] = np.maximum(best[near], value)
    share = line.length / len(points)
    lengths = {}
    for value in best[np.isfinite(best)]:
        lengths[value] = lengths.get(value, 0.0) + share
    return sorted(lengths.items(), key=lambda p: -p[0])


def _claimed_lengths(pieces, line):
    """Worst-wins claiming over lineal pieces of one edge, computed in
    1-D parameter space along `line`: higher values claim their
    stretch first, lower values keep only what is left. Returns
    ``[(value, claimed length)]``."""
    if not line.is_simple:
        return _sampled_claim(line, pieces)
    claimed = []
    lengths = []
    for value, piece in sorted(pieces, key=lambda p: -p[0]):
        intervals = _merge_intervals(_piece_intervals(line, piece))
        kept = _subtract_intervals(intervals, claimed)
        length = sum(high - low for low, high in kept)
        if length > 0:
            lengths.append((value, length))
            claimed = _merge_intervals(claimed + kept)
    return lengths


def _length_products(length, lengths, thresholds):
    """Per-edge products from ``[(value, claimed length)]``."""
    if length == 0:
        return 0.0, 0.0, np.nan, {x: 0.0 for x in thresholds}
    covered = sum(l for _, l in lengths)
    # Weights normalized before multiplying: the mean is a convex
    # combination bounded by the largest |value|, so finite inputs can
    # never overflow into an infinite product.
    mean = sum(v * (l / length) for v, l in lengths)
    coverage = covered / length
    maximum = max((v for v, _ in lengths), default=np.nan)
    shares = {x: sum(l for v, l in lengths if v >= x) / length for x in thresholds}
    return mean, coverage, maximum, shares


def polygon_edge_products(line, pieces, thresholds=()):
    """Per-edge products from ``(value, polygon)`` pieces — exact on
    simple edges; self-crossing or retraced edges claim through the
    dense-sampling fallback.

    Higher values claim their stretch of the line first, so overlaps
    resolve as "the worst thing present"; a boundary touch has zero
    length and contributes nothing. Returns
    ``(mean, coverage, maximum, {threshold: share})``.
    """
    length = line.length
    if length == 0:
        return 0.0, 0.0, np.nan, {x: 0.0 for x in thresholds}
    intersected = [(value, line.intersection(polygon)) for value, polygon in pieces]
    return _length_products(length, _claimed_lengths(intersected, line), thresholds)


def sample_products(values, thresholds=()):
    """Per-edge products from along-edge sample values (NaN = uncovered)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    valid = np.isfinite(values)
    if n == 0 or not valid.any():
        return 0.0, 0.0, np.nan, {x: 0.0 for x in thresholds}
    # Scale before summing: each term is bounded by max|value| / n,
    # so finite samples cannot overflow the accumulator.
    mean = float((np.where(valid, values, 0.0) * (1.0 / n)).sum())
    coverage = float(valid.sum() / n)
    maximum = float(values[valid].max())
    shares = {x: float((valid & (values >= x)).sum() / n) for x in thresholds}
    return mean, coverage, maximum, shares


class Exposure:
    """User-named exposure layers ingested onto the street network.

    Parameters
    ----------
    network : TransportNetwork
        A network built with a street network (``osm_pbf=``).
    thresholds : dict, optional
        ``{layer name: level or sequence of levels}`` — declares
        "minutes at or above X" aggregates. The comparison is
        ``value >= X``, so class-bound data (a noise ``db_low``
        column) counts its own class at its bound. Thresholds live
        here because the at-or-above share is computed in the same
        pass as the mean and cannot be recovered later.
    rasterize : float or None (default: 1.0)
        Cell size in metres for the default rasterized polygon
        ingestion (burned ``all_touched``, max-wins; dilation of up
        to one cell at zone borders is part of the estimate).
        ``None`` selects the exact length-true overlay instead.
    **layers : tuple
        ``name=(source, value)`` — `source` is a raster path or an
        opened rioxarray object (`value` names the band), or a
        GeoDataFrame of polygons or lines (`value` names the numeric
        column). More-is-better data enters as its deficit if it is
        to be minimized against (e.g. ``100 - Comb_GVI``).
    """

    def __init__(self, network, *, thresholds=None, rasterize=1.0, **layers):
        if not layers:
            raise ValueError(
                "Exposure needs at least one layer, e.g. " "noise=(zones_gdf, 'db_low')"
            )
        edges = network.streets_gdf
        if rasterize is not None:
            rasterize = float(rasterize)
            if not (math.isfinite(rasterize) and rasterize > 0):
                raise ValueError(
                    "rasterize must be a positive cell size in metres, "
                    "or None for the exact polygon overlay"
                )
        self._thresholds = _validated_thresholds(thresholds, layers)
        _validate_names(layers, self._thresholds, edges.columns)

        metric_crs = edges.estimate_utm_crs()
        projected = edges.geometry.to_crs(metric_crs)
        # Samples serve raster/line layers and the default rasterized
        # polygon path; built once on first need (the exact polygon
        # opt-in never builds them).
        cache = {}

        def samples():
            if "points" not in cache:
                cache["points"] = _edge_samples(projected)
            return cache["points"]

        self._network = network
        self._edge_count = len(edges)
        # Every layer ingests into staging first; the frame is written
        # only after the whole constructor succeeded, so a failing
        # later layer leaves streets_gdf exactly as it was.
        staged = {}
        for name, spec in layers.items():
            source, value = _validated_spec(name, spec)
            layer_thresholds = self._thresholds.get(name, ())
            products = _ingest(
                name,
                source,
                value,
                projected,
                metric_crs,
                samples,
                layer_thresholds,
                rasterize,
            )
            if not (np.asarray(products["coverage"]) > 0).any():
                raise ValueError(
                    f"layer {name!r} covers no street edge — wrong region " "or CRS?"
                )
            _validate_products(name, products)
            staged[name] = products
        self._layers = staged
        for name, products in staged.items():
            edges[name] = products["mean"]
            edges[f"{name}_coverage"] = products["coverage"]
            edges[f"{name}_max"] = products["max"]
            for threshold, share in products["shares"].items():
                suffix = _threshold_suffix(threshold)
                edges[f"{name}_share_above_{suffix}"] = share

    @property
    def layers(self):
        """The layer names, in declaration order."""
        return tuple(self._layers)

    def thresholds(self, name):
        """The declared thresholds of a layer (possibly empty)."""
        return tuple(self._thresholds.get(name, ()))


def _validate_products(name, products):
    """Every committed per-edge product must be finite (the maximum may
    be NaN on uncovered edges, never infinite)."""
    finite = (
        np.isfinite(products["mean"]).all()
        and np.isfinite(products["coverage"]).all()
        and not np.isinf(products["max"]).any()
        and all(np.isfinite(s).all() for s in products["shares"].values())
    )
    if not finite:
        raise ValueError(
            f"layer {name!r} produced non-finite per-edge products — "
            "the values are too large to aggregate"
        )
    bounded = (products["coverage"] <= 1 + 1e-9).all() and all(
        (s <= 1 + 1e-9).all() for s in products["shares"].values()
    )
    if not bounded:
        raise ValueError(
            f"layer {name!r} produced a coverage or threshold share "
            "above 1 — an ingestion invariant broke; please report this"
        )


def _validated_thresholds(thresholds, layers):
    if thresholds is None:
        return {}
    normalized = {}
    for name, levels in thresholds.items():
        if name not in layers:
            raise ValueError(
                f"thresholds name unknown layer {name!r}; layers are "
                f"{sorted(layers)}"
            )
        try:
            levels = tuple(float(x) for x in levels)
        except TypeError:
            levels = (float(levels),)
        if any(not math.isfinite(x) for x in levels):
            raise ValueError(f"thresholds for {name!r} must be finite")
        normalized[name] = levels
    return normalized


def _validate_names(layers, thresholds, existing):
    existing = set(existing) | {"geometry"}
    seen = {}
    for name in layers:
        if not _NAME.fullmatch(name):
            raise ValueError(f"layer name {name!r} is not a lowercase identifier")
        for column in _derived_columns(name, thresholds.get(name, ())):
            if column in existing:
                raise ValueError(
                    f"layer {name!r} would write column {column!r}, which "
                    "already exists on streets_gdf"
                )
            if column in seen:
                raise ValueError(
                    f"layers {seen[column]!r} and {name!r} both derive the "
                    f"column {column!r}"
                )
            seen[column] = name


def _validated_spec(name, spec):
    if not isinstance(spec, (tuple, list)) or len(spec) != 2:
        raise ValueError(
            f"layer {name!r} must be a (source, value) pair, e.g. "
            "(gdf, 'db_low') or ('air.tif', 'PM25Concentration')"
        )
    source, value = spec
    if not isinstance(value, str):
        raise ValueError(
            f"layer {name!r}: the value selector must be a "
            "band or column name string"
        )
    return source, value


def _edge_samples(projected):
    """Midpoint sample points for every edge, flat, with edge indices.

    One vectorized interpolation call for the whole network — the
    Python-loop equivalent costs minutes at metropolitan edge counts.
    """
    import shapely

    lines = np.asarray(projected.values)
    lengths = shapely.length(lines)
    counts = np.maximum(1, np.ceil(lengths / SAMPLE_STEP_M)).astype(int)
    owners = np.repeat(np.arange(len(lines)), counts)
    starts = np.cumsum(counts) - counts
    within = np.arange(int(counts.sum())) - np.repeat(starts, counts)
    fractions = (within + 0.5) / np.repeat(counts, counts)
    points = shapely.line_interpolate_point(
        np.repeat(lines, counts), fractions, normalized=True
    )
    return points, owners


def _ingest(name, source, value, projected, metric_crs, samples, thresholds, rasterize):
    import geopandas

    if isinstance(source, geopandas.GeoDataFrame):
        return _ingest_vector(
            name, source, value, projected, metric_crs, samples, thresholds, rasterize
        )
    return _ingest_raster(
        name,
        source,
        value,
        metric_crs,
        samples(),
        thresholds,
        len(projected),
    )


def _ingest_vector(
    name, frame, value, projected, metric_crs, samples, thresholds, rasterize
):
    if value not in frame.columns:
        raise ValueError(
            f"layer {name!r} has no column {value!r} "
            f"(columns: {sorted(c for c in frame.columns if c != frame.geometry.name)})"
        )
    if frame.crs is None:
        raise ValueError(
            f"layer {name!r} declares no CRS; set one with " "set_crs before passing it"
        )
    numbers = np.asarray(frame[value], dtype=float)
    bad = ~np.isfinite(numbers)
    if bad.any():
        raise ValueError(
            f"layer {name!r}: {int(bad.sum())} non-finite value(s) in "
            f"column {value!r}"
        )
    if frame.geometry.isna().any():
        raise ValueError(
            f"layer {name!r}: {int(frame.geometry.isna().sum())} missing "
            "geometry(ies)"
        )
    kinds = set(frame.geometry.geom_type.unique())
    polygonal = {"Polygon", "MultiPolygon"}
    linear = {"LineString", "MultiLineString"}
    # A privately named value column: user column names must not
    # collide with the joins' reserved names (index_right, …).
    import geopandas

    sources = geopandas.GeoDataFrame(
        {_VALUE: np.asarray(frame[value], dtype=float)},
        geometry=frame.geometry.values,
        crs=frame.crs,
    ).to_crs(metric_crs)
    if kinds <= polygonal:
        # Real zone datasets routinely carry invalid polygons
        # (self-intersections); repair them rather than refuse, or the
        # published sample data itself would be unusable. Repair can
        # leave lineal remnants — only polygonal components may carry
        # area exposure, so everything else is dropped.
        from shapely import make_valid

        sources = sources.set_geometry(
            [_polygonal(geometry) for geometry in make_valid(sources.geometry.values)]
        )
        if rasterize is not None:
            return _ingest_polygons_rasterized(
                sources, projected, thresholds, rasterize, samples()
            )
        return _ingest_polygons(sources, projected, thresholds)
    if kinds <= linear:
        return _ingest_lines(sources, samples(), thresholds, len(projected))
    raise ValueError(
        f"layer {name!r} mixes or uses unsupported geometry types "
        f"({sorted(str(k) for k in kinds)}); pass a uniformly polygonal "
        "or linear frame"
    )


def _polygonal(geometry):
    """Only the polygonal components of a repaired geometry — lineal
    make_valid remnants must not fake area coverage."""
    import shapely
    from shapely.geometry import MultiPolygon, Polygon

    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    parts = [
        part
        for part in shapely.get_parts(geometry)
        if isinstance(part, (Polygon, MultiPolygon))
    ]
    if not parts:
        return Polygon()
    if len(parts) == 1:
        return parts[0]
    return shapely.union_all(np.asarray(parts, dtype=object))


def _ingest_polygons_rasterized(frame, projected, thresholds, resolution, samples):
    """The default polygon path: burn the zones at `resolution` metres
    (all_touched — center-touch burning silently drops class bands
    thinner than a cell; ascending value order so overlaps resolve
    max-wins) and read the burn at the edges' midpoint samples. A
    resolution-bounded estimate; the exact overlay is the
    ``rasterize=None`` opt-in."""
    try:
        import rasterio.features
        from affine import Affine
    except ImportError as error:
        raise ImportError(
            "the default rasterized polygon ingestion needs rasterio "
            "(install cafein[dem]); pass rasterize=None for the exact "
            "overlay instead"
        ) from error
    import shapely

    points, owners = samples
    n = len(projected)
    xs = shapely.get_x(np.asarray(points))
    ys = shapely.get_y(np.asarray(points))
    layer_bounds = frame.total_bounds
    west = max(float(layer_bounds[0]), float(xs.min())) - resolution
    south = max(float(layer_bounds[1]), float(ys.min())) - resolution
    east = min(float(layer_bounds[2]), float(xs.max())) + resolution
    north = min(float(layer_bounds[3]), float(ys.max())) + resolution
    values = np.full(len(points), np.nan)
    if east > west and north > south:
        width_cells = float(np.ceil((east - west) / resolution))
        height_cells = float(np.ceil((north - south) / resolution))
        # The budget check runs in float space: an absurdly fine
        # resolution overflows to inf, which must still refuse
        # actionably rather than crash on int conversion.
        if not (
            width_cells <= BURN_STRIP_CELLS
            and height_cells <= BURN_STRIP_CELLS * BURN_STRIP_CELLS
        ):
            raise ValueError(
                f"rasterize={resolution:g} needs {width_cells:g} x "
                f"{height_cells:g} cells over this extent, past the "
                f"{BURN_STRIP_CELLS}-cell strip budget — coarsen "
                "rasterize or pass rasterize=None"
            )
        width = max(1, int(width_cells))
        height = max(1, int(height_cells))
        ordered = frame.sort_values(_VALUE)
        shapes = [
            (geometry, value)
            for geometry, value in zip(ordered.geometry.values, ordered[_VALUE].values)
            if not geometry.is_empty
        ]
        strip_rows = max(1, BURN_STRIP_CELLS // width)
        for row0 in range(0, height, strip_rows):
            rows_here = min(strip_rows, height - row0)
            strip_north = north - row0 * resolution
            strip_south = strip_north - rows_here * resolution
            transform = Affine(resolution, 0, west, 0, -resolution, strip_north)
            inside = (
                (xs >= west) & (xs < east) & (ys <= strip_north) & (ys > strip_south)
            )
            if not inside.any():
                continue
            burned = rasterio.features.rasterize(
                shapes,
                out_shape=(rows_here, width),
                transform=transform,
                fill=np.nan,
                dtype="float32",
                all_touched=True,
            )
            cols, rows = (~transform) * (xs[inside], ys[inside])
            cols = np.clip(np.floor(cols).astype(int), 0, width - 1)
            rows = np.clip(np.floor(rows).astype(int), 0, rows_here - 1)
            values[inside] = burned[rows, cols]
    return _sampled_products(values, owners, thresholds, n)


def _ingest_polygons(frame, projected, thresholds):
    import geopandas

    # Positional indices on both operands: the source frame commonly
    # arrives filtered with its original labels retained, and sjoin
    # returns labels, not positions.
    frame = frame.reset_index(drop=True)
    edges = geopandas.GeoDataFrame(geometry=projected.reset_index(drop=True))
    joined = geopandas.sjoin(edges, frame, predicate="intersects")
    n = len(projected)
    mean = np.zeros(n)
    coverage = np.zeros(n)
    maximum = np.full(n, np.nan)
    shares = {x: np.zeros(n) for x in thresholds}
    if joined.empty:
        return {"mean": mean, "coverage": coverage, "max": maximum, "shares": shares}

    import shapely

    # Bulk GEOS over the pair table. Real zone contours carry hundreds
    # of vertices each, so two reductions come before any overlay:
    # a prepared containment test settles the pairs whose edge lies
    # wholly inside the polygon (no overlay needed — the whole edge is
    # the piece), and the rest intersect against the polygon CLIPPED
    # to the edge's envelope, which cuts each overlay from contour
    # size to edge size.
    pair_edges = joined.index.to_numpy(dtype=int)
    pair_polys = joined["index_right"].to_numpy(dtype=int)
    edge_geoms = np.asarray(edges.geometry.values)
    poly_geoms = np.asarray(frame.geometry.values)
    edge_lengths = shapely.length(edge_geoms)

    shapely.prepare(poly_geoms)
    pair_lines = edge_geoms[pair_edges]
    contained = shapely.contains_properly(poly_geoms[pair_polys], pair_lines)

    pieces = np.empty(len(pair_edges), dtype=object)
    pieces[contained] = pair_lines[contained]
    rest = ~contained
    if rest.any():
        # clip_by_rect takes scalar bounds; the per-pair loop is a thin
        # O(vertices) clip, far cheaper than a full overlay against the
        # uncut contour would be.
        rest_polys = poly_geoms[pair_polys[rest]]
        bounds = shapely.bounds(pair_lines[rest])

        def clip(polygon, edge_bounds):
            # Padding keeps the rectangle non-degenerate for
            # axis-aligned edges; a larger clip region can never change
            # the eventual line intersection, only cost a little more.
            xmin, ymin, xmax, ymax = edge_bounds
            pad = max(1e-6, 1e-9 * max(xmax - xmin, ymax - ymin))
            try:
                return shapely.clip_by_rect(
                    polygon, xmin - pad, ymin - pad, xmax + pad, ymax + pad
                )
            except shapely.errors.GEOSException:
                # A rare degenerate ring survives make_valid; that pair
                # falls back to the uncut contour.
                return polygon

        clipped = np.array(
            [
                clip(polygon, edge_bounds)
                for polygon, edge_bounds in zip(rest_polys, bounds)
            ],
            dtype=object,
        )
        rest_lines = pair_lines[rest]
        try:
            pieces[rest] = shapely.intersection(rest_lines, clipped)
        except shapely.errors.GEOSException:
            # clip_by_rect may emit topologically dirty output without
            # raising; repair per pair so one bad clip cannot abort
            # the layer.
            repaired = []
            for line_geom, clip_geom in zip(rest_lines, clipped):
                try:
                    repaired.append(shapely.intersection(line_geom, clip_geom))
                except shapely.errors.GEOSException:
                    repaired.append(
                        shapely.intersection(
                            line_geom, _polygonal(shapely.make_valid(clip_geom))
                        )
                    )
            pieces[rest] = np.array(repaired, dtype=object)
    piece_lengths = shapely.length(pieces)
    keep = piece_lengths > 0
    pair_edges = pair_edges[keep]
    pair_values = frame[_VALUE].to_numpy(dtype=float)[pair_polys[keep]]
    pieces = pieces[keep]
    piece_lengths = piece_lengths[keep]
    if not len(pair_edges):
        # Only point/boundary touches: zero coverage everywhere; the
        # constructor turns that into the empty-overlap refusal.
        return {"mean": mean, "coverage": coverage, "max": maximum, "shares": shares}

    order = np.argsort(pair_edges, kind="stable")
    boundaries = np.flatnonzero(np.diff(pair_edges[order])) + 1
    lines = edges.geometry.values
    simple = shapely.is_simple(lines)
    for group in np.split(order, boundaries):
        edge_index = int(pair_edges[group[0]])
        length = float(edge_lengths[edge_index])
        if len(group) == 1 and simple[edge_index]:
            # One piece cannot overlap itself on a simple edge: its
            # length is the covered length, no claiming. Retraced
            # edges must claim (GEOS collapses their doubled stretch
            # while the edge length counts it twice).
            lengths = [(float(pair_values[group[0]]), float(piece_lengths[group[0]]))]
        else:
            # Multi-piece edges always claim, in interval space: real
            # zone layers overlap with coordinate-noise-level offsets
            # that defeat any GEOS union/difference-based shortcut.
            lengths = _claimed_lengths(
                [(float(pair_values[g]), pieces[g]) for g in group],
                lines[edge_index],
            )
        m, c, mx, s = _length_products(length, lengths, thresholds)
        mean[edge_index] = m
        coverage[edge_index] = c
        maximum[edge_index] = mx
        for x in thresholds:
            shares[x][edge_index] = s[x]
    return {"mean": mean, "coverage": coverage, "max": maximum, "shares": shares}


def _ingest_lines(frame, samples, thresholds, n):
    import geopandas
    import pandas

    points, owners = samples
    # A geometry-only left frame: the right frame already carries the
    # privately named value column only.
    sample_frame = geopandas.GeoDataFrame(geometry=points, crs=frame.crs)
    joined = geopandas.sjoin_nearest(
        sample_frame,
        frame.reset_index(drop=True),
        max_distance=LINE_MATCH_TOLERANCE_M,
        how="left",
    )
    # Equidistant matches duplicate the sample row; reduce to the
    # maximum value per sample — the worst-wins rule.
    per_sample = joined.groupby(level=0)[_VALUE].max()
    per_sample = per_sample.reindex(pandas.RangeIndex(len(points)))
    return _sampled_products(per_sample.values, owners, thresholds, n)


def _ingest_raster(name, source, value, metric_crs, samples, thresholds, n):
    band, transform, crs, nodata = _open_band(name, source, value)
    if crs is None:
        raise ValueError(f"layer {name!r}: the raster declares no CRS")
    from pyproj import Transformer

    points, owners = samples
    import shapely

    xs = shapely.get_x(np.asarray(points))
    ys = shapely.get_y(np.asarray(points))
    to_raster = Transformer.from_crs(metric_crs, crs, always_xy=True)
    rx, ry = to_raster.transform(xs, ys)
    cols, rows = (~transform) * (np.asarray(rx), np.asarray(ry))
    cols = np.floor(cols).astype(int)
    rows = np.floor(rows).astype(int)
    inside = (rows >= 0) & (rows < band.shape[0]) & (cols >= 0) & (cols < band.shape[1])
    # NaN cells (and declared nodata) are "uncovered" — float rasters'
    # missing convention; infinities anywhere in the band are refused,
    # sampled or not.
    if np.isinf(band).any():
        raise ValueError(
            f"layer {name!r}: {int(np.isinf(band).sum())} infinite "
            "value(s) in the raster band"
        )
    values = np.full(len(points), np.nan)
    values[inside] = band[rows[inside], cols[inside]]
    if nodata is not None:
        values[values == nodata] = np.nan
    return _sampled_products(values, owners, thresholds, n)


def _sampled_products(values, owners, thresholds, n):
    """Grouped reductions over the flat sample arrays — O(samples),
    never O(edges x samples). Weights are pre-divided by each edge's
    sample count, so every partial sum stays bounded by the largest
    |value| and finite inputs cannot overflow."""
    values = np.asarray(values, dtype=float)
    owners = np.asarray(owners)
    valid = np.isfinite(values)
    counts = np.bincount(owners, minlength=n).astype(float)
    weight = np.zeros(len(values))
    present = counts[owners] > 0
    weight[present] = 1.0 / counts[owners][present]
    zero_filled = np.where(valid, values, 0.0)
    mean = np.bincount(owners, weights=zero_filled * weight, minlength=n)
    coverage = np.bincount(owners, weights=valid.astype(float) * weight, minlength=n)
    maximum = np.full(n, -np.inf)
    if valid.any():
        np.fmax.at(maximum, owners[valid], values[valid])
    maximum[~np.isfinite(maximum)] = np.nan
    shares = {
        x: np.bincount(
            owners,
            weights=(valid & (values >= x)).astype(float) * weight,
            minlength=n,
        )
        for x in thresholds
    }
    return {"mean": mean, "coverage": coverage, "max": maximum, "shares": shares}


def _open_band(name, source, value):
    """A raster band as ``(array, affine transform, crs, nodata)``."""
    try:
        import rioxarray
    except ImportError as error:
        raise ImportError(
            "raster exposure layers need the optional rioxarray "
            "dependency; install cafein[dem]"
        ) from error

    if isinstance(source, (str, os.PathLike)):
        # Masked + scaled like the DEM ladder: internal masks become
        # NaN ("uncovered"), and scale/offset-encoded bands decode to
        # physical values instead of raw storage integers.
        raster = rioxarray.open_rasterio(
            os.fspath(source), masked=True, mask_and_scale=True
        )
    else:
        raster = source
    data_vars = getattr(raster, "data_vars", None)
    if data_vars is not None:
        if value not in data_vars:
            raise ValueError(
                f"layer {name!r} has no variable {value!r} "
                f"(variables: {sorted(data_vars)})"
            )
        selected = raster[value]
    else:
        selected = _select_band(name, raster, value)
    # The rio accessor names the spatial dims (easting/northing
    # included); squeeze only singleton NON-spatial dims (band, time)
    # and transpose to (y, x) so the array indexing below holds for
    # any source dimension order.
    try:
        # The accessor the USER configured lives on the source object;
        # a variable selected out of a Dataset does not inherit it.
        x_dim, y_dim = raster.rio.x_dim, raster.rio.y_dim
    except Exception:
        try:
            x_dim, y_dim = selected.rio.x_dim, selected.rio.y_dim
        except Exception as error:
            raise ValueError(
                f"layer {name!r}: the raster's spatial dimensions are "
                "not identifiable; set them with rio.set_spatial_dims"
            ) from error
    selected = selected.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim)
    for dim in list(selected.dims):
        if dim not in (x_dim, y_dim) and selected.sizes[dim] == 1:
            selected = selected.squeeze(dim, drop=True)
    if selected.ndim != 2:
        raise ValueError(
            f"layer {name!r}: band {value!r} is not a 2D grid "
            f"(dims: {selected.dims})"
        )
    selected = selected.transpose(y_dim, x_dim)
    return (
        np.asarray(selected.values, dtype=float),
        selected.rio.transform(),
        selected.rio.crs,
        selected.rio.nodata,
    )


def _select_band(name, raster, value):
    """The band whose description names `value` (``"Name [unit]"``
    descriptions match on the name part)."""
    descriptions = raster.attrs.get("long_name")
    if isinstance(descriptions, str):
        descriptions = [descriptions]
    # A DataArray whose own name matches needs no band descriptions.
    if not descriptions and getattr(raster, "name", None) == value:
        return raster
    if raster.ndim == 2:
        raster = raster.expand_dims("band") if "band" not in raster.dims else raster
    bands = list(raster["band"].values) if "band" in raster.coords else []
    if descriptions:
        for index, description in enumerate(descriptions):
            text = str(description)
            if text == value or text.startswith(value + " ["):
                return raster.isel(band=index)
        raise ValueError(
            f"layer {name!r} has no band {value!r} "
            f"(bands: {[str(d) for d in descriptions]})"
        )
    if len(bands) == 1:
        raise ValueError(
            f"layer {name!r}: the raster carries no band descriptions to "
            f"match {value!r} against; a single unnamed band cannot be "
            "selected by name"
        )
    raise ValueError(
        f"layer {name!r}: the raster carries no band descriptions to "
        f"match {value!r} against"
    )
