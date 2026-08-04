"""The car parking search model and its query-time resolution.

A car trip ends with a parking search, costed as time and extra driving
distance (GEMMAT §3.2.3; Fink et al. 2024) — never as a walking leg. The
model is opt-in: with ``parking=`` omitted a car query reports the driving
time without any parking-search cost. ``resolve`` validates the option
into an owned snapshot; ``destination_costs`` turns that snapshot into
per-destination ``(seconds, metres)`` arrays the query surfaces add to
each reachable cell — travel time gains the seconds, the network distance
(and with it the emissions basis) gains the metres, and the path geometry
never shows the search loop.
"""

import math
import numbers

import numpy as np

PARKING_SECONDS = 300.0
"""The shipped constant search time — a stated round product default that
sits inside the 245–322-second total parking penalties Jaakkola (2013,
Taulukko 17) derives for the Helsinki region (corroboration, not
calibration)."""

PARKING_METRES = 0.0
"""The shipped constant search distance."""

_MAX_VALUE = 1e9
"""The loud ceiling on parking seconds and metres: far beyond any real
search, and small enough that rounding and adding to any travel time or
distance stays safely inside the integer output domain."""


def _checked(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"parking {name} must be a number")
    value = float(value)
    if not math.isfinite(value) or value < 0.0 or value > _MAX_VALUE:
        raise ValueError(
            f"parking {name} must be non-negative, finite, and at most "
            f"{_MAX_VALUE:.0e}"
        )
    return value


class _ParkingAreas:
    """An owned, validated snapshot of a parking area frame.

    Snapshotting at resolve time keeps later mutation of the caller's
    frame (values, geometry, or CRS) from bypassing validation — the
    destination assignment happens only after the routing call returns.
    """

    def __init__(self, frame):
        import geopandas as gpd
        from shapely.geometry import MultiPolygon, Polygon

        if "seconds" not in frame.columns:
            raise ValueError("a parking area frame needs a 'seconds' column")
        if frame.crs is None:
            raise ValueError("a parking area frame must carry a CRS")
        geometry = frame.geometry.reset_index(drop=True)
        if not all(isinstance(shape, (Polygon, MultiPolygon)) for shape in geometry):
            raise ValueError("parking areas must be Polygon or MultiPolygon shapes")
        seconds = frame["seconds"].to_numpy(dtype=float, copy=True)
        metres = (
            frame["metres"].to_numpy(dtype=float, copy=True)
            if "metres" in frame.columns
            else np.zeros(len(frame))
        )
        for name, values in (("seconds", seconds), ("metres", metres)):
            if (
                not np.all(np.isfinite(values))
                or (values < 0.0).any()
                or (values > _MAX_VALUE).any()
            ):
                raise ValueError(
                    f"parking {name} must be non-negative, finite, and at "
                    f"most {_MAX_VALUE:.0e}"
                )
        self.rows = gpd.GeoDataFrame(
            {
                "__seconds": seconds,
                "__metres": metres,
                "__row": np.arange(len(frame)),
            },
            geometry=geometry.copy(),
            crs=frame.crs,
        )


def resolve(parking, mode):
    """The validated parking option, or ``None`` when the model is off.

    Forms: ``None``/``False`` → off; ``True`` → the shipped constants; a
    number → that many seconds and 0 metres; a ``(seconds, metres)`` pair →
    both constants; a polygon GeoDataFrame with a ``seconds`` column
    (optional ``metres``) → per-destination values by point-in-polygon.
    The option belongs to the car: any other mode rejects it.
    """
    if parking is None or parking is False:
        return None
    if mode != "car":
        raise ValueError("parking applies to mode='car'")
    if parking is True:
        return (PARKING_SECONDS, PARKING_METRES)
    if isinstance(parking, numbers.Real) and not isinstance(parking, bool):
        return (_checked(parking, "seconds"), 0.0)
    if isinstance(parking, tuple) and len(parking) == 2:
        return (_checked(parking[0], "seconds"), _checked(parking[1], "metres"))
    if hasattr(parking, "geometry"):
        return _ParkingAreas(parking)
    raise ValueError(
        "parking must be True, seconds, a (seconds, metres) pair, or a "
        "polygon GeoDataFrame with a 'seconds' column"
    )


def destination_costs(resolved, destinations):
    """Per-destination ``(seconds, metres)`` arrays for a resolved option.

    `destinations` is a sequence of ``(latitude, longitude)`` pairs. With a
    constant form both arrays are flat. With an area snapshot each
    destination takes its polygon's row by point-in-polygon (a point on a
    boundary is outside); a destination inside several polygons takes the
    row with the largest seconds, ties broken by largest metres and then
    lowest row position — a total, stable ordering — and a destination
    outside every polygon falls back to the shipped constants.
    """
    count = len(destinations)
    if isinstance(resolved, tuple):
        seconds, metres = resolved
        return np.full(count, seconds), np.full(count, metres)

    import geopandas as gpd

    points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(
            [longitude for _, longitude in destinations],
            [latitude for latitude, _ in destinations],
        ),
        crs="EPSG:4326",
    )
    joined = points.sjoin(
        resolved.rows.to_crs(points.crs), how="left", predicate="within"
    )
    # The precedence: largest seconds, then largest metres, then the lowest
    # row position — one row per destination, never duplicated cells.
    joined = joined.sort_values(
        ["__seconds", "__metres", "__row"], ascending=[False, False, True]
    )
    best = joined.groupby(level=0)[["__seconds", "__metres"]].first()
    best = best.reindex(range(count))
    outside = best["__seconds"].isna()
    seconds = best["__seconds"].fillna(PARKING_SECONDS).to_numpy()
    metres = best["__metres"].where(~outside, PARKING_METRES).to_numpy()
    return seconds, metres
