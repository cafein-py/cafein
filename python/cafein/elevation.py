"""DEM sampling for the street network's per-coordinate elevations.

The DEM is a user-supplied input, like the GTFS feed and the OSM extract;
cafein neither bundles nor fetches elevation. Sampling happens once per build,
outside every routing loop, and produces one ``float32`` value per geometry
coordinate with ``NaN`` for unavailable — the array the Rust core stores
alongside the coordinates.

The backend is a seam, not a hard dependency: ``sample_dem`` accepts a GeoTIFF
path or a sequence of tile paths and reads them through **rioxarray**
(optional dependency), while builders and tests may pass a callable
``(lons, lats) -> elevations`` instead and never touch a raster.
"""

import os

import numpy as np

NODATA_POLICY = "nan"
"""Missing raster values become NaN; nothing is inferred from neighbours."""


def sample_dem(longitudes, latitudes, dem):
    """Elevations for EPSG:4326 coordinates, as float32 with NaN nodata.

    Parameters
    ----------
    longitudes, latitudes : array-like
        Query coordinates in EPSG:4326.
    dem : path, sequence of paths, or callable
        A GeoTIFF path or a sequence of tile paths, opened with rioxarray
        and sampled bilinearly in the DEM's own CRS; or a callable
        ``(lons, lats) -> elevations`` supplying the values directly —
        the swappable seam the tests and advanced builders use.
    """
    longitudes = np.asarray(longitudes, dtype=float)
    latitudes = np.asarray(latitudes, dtype=float)
    if callable(dem):
        # The callable gets read-only copies — never writable views of the
        # builder's geometry arrays — and hands back values we copy in turn,
        # since structure inference mutates the result in place.
        lons, lats = longitudes.copy(), latitudes.copy()
        lons.flags.writeable = False
        lats.flags.writeable = False
        values = np.array(dem(lons, lats), dtype=np.float32, copy=True)
        if values.shape != longitudes.shape:
            raise ValueError(
                f"the elevation callable returned {values.shape} values for "
                f"{longitudes.shape} coordinates"
            )
    else:
        values = _sample_rasters(longitudes, latitudes, dem)
    # NaN is the only unavailable sentinel: infinities — raster nodata quirks
    # or float32 overflow in the cast — must not reach the stored array.
    values[np.isinf(values)] = np.nan
    return values


def _sample_rasters(longitudes, latitudes, dem):
    """Bilinear samples from one GeoTIFF or a mosaic of tiles."""
    try:
        import rioxarray
        import xarray  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "sampling a DEM needs the optional rioxarray dependency; "
            "install cafein[dem] or pass an elevation callable"
        ) from error

    paths = (
        [os.fspath(dem)]
        if isinstance(dem, (str, os.PathLike))
        else [os.fspath(tile) for tile in dem]
    )
    if not paths:
        raise ValueError("dem names no raster tiles")
    # `mask_and_scale` both turns nodata into NaN and decodes any GeoTIFF
    # scale/offset, so scaled DEMs sample as physical elevations.
    rasters = [
        rioxarray.open_rasterio(path, masked=True, mask_and_scale=True).squeeze(
            "band", drop=True
        )
        for path in paths
    ]
    if len(rasters) == 1:
        raster = rasters[0]
    else:
        from rioxarray.merge import merge_arrays

        raster = merge_arrays(rasters)
    if raster.rio.crs is None:
        raise ValueError("the DEM carries no CRS; assign one before sampling")

    # Project the query points into the DEM's CRS — national DEMs are
    # projected while OSM geometry is EPSG:4326 — then interpolate
    # bilinearly. Nodata is already NaN from the masked open above.
    import xarray as xr
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", raster.rio.crs, always_xy=True)
    xs, ys = transformer.transform(longitudes, latitudes)
    sampled = raster.interp(
        x=xr.DataArray(np.asarray(xs), dims="points"),
        y=xr.DataArray(np.asarray(ys), dims="points"),
        method="linear",
    )
    return np.asarray(sampled.values, dtype=np.float32)


def infer_structures(offsets, longitudes, latitudes, elevations, structure_edges):
    """Endpoint-interpolated elevations for bridge and tunnel edges, in place.

    Terrain elevation under a bridge or over a tunnel is not the travelled
    structure, so flagged edges replace their interior coordinates with a
    straight interpolation between the edge's endpoint elevations — the
    documented fallback — weighted by distance along the edge. Returns how
    many edges had an interior rewritten.
    """
    inferred = 0
    for edge in structure_edges:
        start, end = int(offsets[edge]), int(offsets[edge + 1])
        # Only the interior is inferred: the endpoints keep their sampled
        # values, so a two-coordinate edge has nothing to rewrite.
        if end - start <= 2:
            continue
        lons = np.asarray(longitudes[start:end], dtype=float)
        lats = np.asarray(latitudes[start:end], dtype=float)
        # Chord lengths in a frame local to the edge: within one edge the
        # longitude scale is near-constant, so cos(mean latitude) suffices
        # for an interpolation weight.
        scale = np.cos(np.radians(lats.mean()))
        steps = np.hypot(np.diff(lons) * scale, np.diff(lats))
        along = np.concatenate([[0.0], np.cumsum(steps)])
        total = along[-1]
        fraction = along / total if total > 0 else along
        # When either endpoint is unavailable the interior becomes NaN rather
        # than arithmetic on NaN overwriting a finite endpoint.
        first, last = elevations[start], elevations[end - 1]
        if np.isfinite(first) and np.isfinite(last):
            interior = fraction[1:-1].astype(np.float32)
            elevations[start + 1 : end - 1] = (
                first + interior * (last - first)
            ).astype(np.float32)
        else:
            elevations[start + 1 : end - 1] = np.float32("nan")
        inferred += 1
    return inferred


def coverage(elevations):
    """The finite share of sampled elevations, 0..=1 (0 for no coordinates)."""
    total = len(elevations)
    if total == 0:
        return 0.0
    return float(np.isfinite(elevations).sum()) / float(total)
