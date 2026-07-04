"""Matrix computers over a transport network."""

import warnings

import numpy as np
import pandas as pd
import shapely


class TravelCostMatrix(pd.DataFrame):
    """The fastest journey's aggregated costs per OD pair, long format.

    A pandas DataFrame with one row per reachable OD pair: ``from_id``
    and ``to_id``, ``travel_time`` (seconds), ``transfers``,
    ``transit_distance`` and ``walk_distance`` (meters), and
    ``emissions`` (grams CO₂e over the ridden legs; NaN where a ridden
    trip has no matching factor row). With ``geometries=True`` each row
    adds ``geometry``, the ridden legs as a shapely MultiLineString in
    EPSG:4326 — convert with
    ``geopandas.GeoDataFrame(matrix, crs="EPSG:4326")``.

    Origins and destinations are either stop identifiers or point
    GeoDataFrames with an ``id`` column. Points are linked once against
    the street network (requires ``osm_pbf=`` at build time): a point's
    travel time is its fastest walk–ride–walk chain, the access and
    egress walks count toward ``walk_distance``, and points off the
    walking network are reported with a warning and yield no rows. From
    stop origins, ``walk_distance`` covers transfers only.

    One RAPTOR run serves each origin, fanned out over all cores; each
    pair's costs come from its fastest journey (ties resolved toward
    fewer rides). Unreachable pairs are absent. Requires a network built
    with trip distances (the default), and with leg geometries for
    ``geometries=True``. Slices and copies degrade to plain DataFrames.

    Parameters
    ----------
    network : TransportNetwork
        The network to compute on.
    origins : list of str, or GeoDataFrame (optional)
        Origin stop_ids (every stop when omitted), or points with an
        ``id`` column.
    destinations : list of str, or GeoDataFrame (optional)
        Destination stop_ids (every stop when omitted), or points; with
        point origins the destinations default to the origins.
    date : str
        Service date as ``YYYY-MM-DD``.
    departure : str
        Departure time at every origin as ``HH:MM:SS``.
    max_transfers : int (optional, default: 4)
        Maximum number of transfers between rides.
    factors : DataFrame or path (optional)
        Extra emission-factor rows layered over the shipped defaults;
        see ``cafein.emissions.load_factors``.
    components : list of str (optional)
        The life-cycle components to include (default: all four); see
        ``cafein.emissions.annotate``.
    geometries : bool (optional, default: False)
        Attach each pair's ridden legs as geometry. Off by default:
        per-pair geometries over large matrices are enormous.
    chunk : (int, int) (optional)
        Compute only origin chunk ``k`` of ``n``: a deterministic
        contiguous block of the resolved origins, so ``n`` batch jobs
        cover all origins disjointly and their shards concatenate.
    walking_speed_kmph, max_walking_time, max_snap_distance : float
        The street-search options for point origins/destinations, as in
        ``TransportNetwork.access_stops``; only valid with points.
    """

    @property
    def _constructor(self):
        return pd.DataFrame

    def __init__(
        self,
        network,
        origins=None,
        destinations=None,
        date=None,
        departure=None,
        *,
        max_transfers=4,
        factors=None,
        components=None,
        geometries=False,
        chunk=None,
        walking_speed_kmph=None,
        max_walking_time=None,
        max_snap_distance=None,
    ):
        table, from_ids, to_ids = _cost_columns(
            network,
            origins,
            destinations,
            date,
            departure,
            max_transfers=max_transfers,
            factors=factors,
            components=components,
            geometries=geometries,
            chunk=chunk,
            walking_speed_kmph=walking_speed_kmph,
            max_walking_time=max_walking_time,
            max_snap_distance=max_snap_distance,
        )
        data = {
            "from_id": np.array(from_ids, dtype=object)[table["from"]],
            "to_id": np.array(to_ids, dtype=object)[table["to"]],
            "travel_time": table["travel_time"],
            "transfers": np.maximum(table["rides"], 1) - 1,
            "transit_distance": table["transit_distance"],
            "walk_distance": table["walk_distance"],
            "emissions": table["emissions"],
        }
        if geometries:
            data["geometry"] = shapely.from_wkb(
                np.array(table["geometry"], dtype=object)
            )
        super().__init__(pd.DataFrame(data))


class TravelTimeMatrix(pd.DataFrame):
    """Travel times per OD pair, long format — the lean r5py-style mode.

    A pandas DataFrame with one row per reachable OD pair: ``from_id``,
    ``to_id``, and ``travel_time`` in seconds. It is the long-format face
    of ``TransportNetwork.travel_time_matrix``: one RAPTOR run serves
    each origin, fanned out over all cores, and the reachable cells of
    the resulting wide matrix are unstacked into rows. Unreachable pairs
    are absent (never a sentinel), so the frame joins straight onto other
    tables. Where travel times only are needed, this is lighter than
    ``TravelCostMatrix``, which also aggregates transfers, distances, and
    emissions.

    With ``window``, every minute mark within ``[departure, departure +
    window)`` is profiled and the ``travel_time`` column is replaced by
    one ``travel_time_p<p>`` column per requested percentile (the median
    by default, or ``confidence`` for the symmetric interval plus the
    median), in seconds and floating-point so an unreachable percentile
    reads as ``NaN``; a pair appears when at least one of its percentiles
    is reachable.

    Origins are either stop identifiers or a point GeoDataFrame with an
    ``id`` column; destinations apply to point origins only — stop
    origins always span every stop (the ``stops`` order). Points are
    linked once against the street network (requires ``osm_pbf=`` at
    build time); points off the walking network are reported with a
    warning and stay unreachable. Slices and copies degrade to plain
    DataFrames.

    Parameters
    ----------
    network : TransportNetwork
        The network to compute on.
    origins : list of str, or GeoDataFrame (optional)
        Origin stop_ids (every stop when omitted), or points with an
        ``id`` column.
    destinations : GeoDataFrame (optional)
        Destination points; defaults to the origins. Only valid with
        point origins — stop origins always span every stop.
    date : str
        Service date as ``YYYY-MM-DD``.
    departure : str
        Departure time at every origin as ``HH:MM:SS``.
    max_transfers : int (optional, default: 4)
        Maximum number of transfers between rides.
    window : int (optional)
        Departure window in seconds; enables percentile columns.
    percentiles : list of float (optional)
        Percentiles in ``[0, 100]`` over the window's departures;
        requires `window`, defaults to ``[50]``.
    confidence : float (optional)
        A level in ``(0, 1)`` mapped to the symmetric percentile
        interval plus the median; requires `window` and excludes
        `percentiles`.
    chunk : (int, int) (optional)
        Compute only origin chunk ``k`` of ``n``: a deterministic
        contiguous block of the resolved origins, so ``n`` batch jobs
        cover all origins disjointly and their rows concatenate.
    walking_speed_kmph, max_walking_time, max_snap_distance : float
        The street-search options for point origins/destinations, as in
        ``TransportNetwork.access_stops``; only valid with points.
    """

    @property
    def _constructor(self):
        return pd.DataFrame

    def __init__(
        self,
        network,
        origins=None,
        destinations=None,
        date=None,
        departure=None,
        *,
        max_transfers=4,
        window=None,
        percentiles=None,
        confidence=None,
        chunk=None,
        walking_speed_kmph=None,
        max_walking_time=None,
        max_snap_distance=None,
    ):
        data = _time_columns(
            network,
            origins,
            date,
            departure,
            max_transfers,
            destinations=destinations,
            window=window,
            percentiles=percentiles,
            confidence=confidence,
            chunk=chunk,
            walking_speed_kmph=walking_speed_kmph,
            max_walking_time=max_walking_time,
            max_snap_distance=max_snap_distance,
        )
        super().__init__(pd.DataFrame(data))


def _time_columns(
    network,
    origins,
    date,
    departure,
    max_transfers,
    *,
    destinations,
    window,
    percentiles,
    confidence,
    chunk,
    walking_speed_kmph,
    max_walking_time,
    max_snap_distance,
):
    """The reachable cells of the travel-time matrix, in long format."""
    if date is None or departure is None:
        raise TypeError("TravelTimeMatrix requires date and departure")
    matrix, from_ids, to_ids, resolved = network._time_matrix_with_ids(
        origins,
        date,
        departure,
        max_transfers,
        destinations=destinations,
        window=window,
        percentiles=percentiles,
        confidence=confidence,
        chunk=chunk,
        walking_speed_kmph=walking_speed_kmph,
        max_walking_time=max_walking_time,
        max_snap_distance=max_snap_distance,
    )
    from_ids = np.asarray(from_ids, dtype=object)
    to_ids = np.asarray(to_ids, dtype=object)
    unreachable = np.iinfo(np.uint32).max
    if resolved is None:
        rows, columns = np.nonzero(matrix != unreachable)
        return {
            "from_id": from_ids[rows],
            "to_id": to_ids[columns],
            "travel_time": matrix[rows, columns],
        }
    rows, columns = np.nonzero((matrix != unreachable).any(axis=2))
    values = matrix[rows, columns, :].astype(float)
    values[values == unreachable] = np.nan
    data = {"from_id": from_ids[rows], "to_id": to_ids[columns]}
    for index, percentile in enumerate(resolved):
        data[f"travel_time_p{percentile:g}"] = values[:, index]
    return data


def travel_cost_table(
    network,
    origins=None,
    destinations=None,
    date=None,
    departure=None,
    *,
    max_transfers=4,
    factors=None,
    components=None,
    geometries=False,
    chunk=None,
    walking_speed_kmph=None,
    max_walking_time=None,
    max_snap_distance=None,
):
    """The travel-cost matrix as a pyarrow Table — the shard-writing form.

    Semantics and parameters follow `TravelCostMatrix`; the output is an
    Arrow table with ``from_id`` and ``to_id`` dictionary-encoded over
    the origin and destination identifiers, the numeric columns wrapping
    the computed arrays zero-copy, and — with ``geometries=True`` — the
    ridden legs as WKB in a binary ``geometry`` column. The batch
    workflow writes one shard per origin chunk::

        network = TransportNetwork.load("network.cafein")
        table = travel_cost_table(network, ..., chunk=(k, n))
        pyarrow.parquet.write_table(table, f"shard-{k:04d}.parquet")

    Shards concatenate trivially. Requires pyarrow (install
    ``cafein[arrow]``).
    """
    try:
        import pyarrow
    except ImportError as error:
        raise ImportError(
            "Arrow tables need the optional pyarrow dependency; install "
            "cafein[arrow] or pyarrow"
        ) from error
    table, from_ids, to_ids = _cost_columns(
        network,
        origins,
        destinations,
        date,
        departure,
        max_transfers=max_transfers,
        factors=factors,
        components=components,
        geometries=geometries,
        chunk=chunk,
        walking_speed_kmph=walking_speed_kmph,
        max_walking_time=max_walking_time,
        max_snap_distance=max_snap_distance,
    )
    columns = {
        "from_id": pyarrow.DictionaryArray.from_arrays(
            pyarrow.array(table["from"]),
            pyarrow.array(from_ids, type=pyarrow.string()),
        ),
        "to_id": pyarrow.DictionaryArray.from_arrays(
            pyarrow.array(table["to"]),
            pyarrow.array(to_ids, type=pyarrow.string()),
        ),
        "travel_time": pyarrow.array(table["travel_time"]),
        "transfers": pyarrow.array(np.maximum(table["rides"], 1) - 1),
        "transit_distance": pyarrow.array(table["transit_distance"]),
        "walk_distance": pyarrow.array(table["walk_distance"]),
        "emissions": pyarrow.array(table["emissions"]),
    }
    if geometries:
        columns["geometry"] = pyarrow.array(
            list(table["geometry"]), type=pyarrow.binary()
        )
    return pyarrow.table(columns)


def _cost_columns(
    network,
    origins,
    destinations,
    date,
    departure,
    *,
    max_transfers,
    factors,
    components,
    geometries,
    chunk,
    walking_speed_kmph,
    max_walking_time,
    max_snap_distance,
):
    """The core's cost arrays plus the origin and destination ids."""
    from cafein import emissions
    from cafein.network import _walk_options

    if date is None or departure is None:
        raise TypeError("TravelCostMatrix requires date and departure")
    trip_factors = emissions.trip_factors(network, factors, components)
    if _is_point_frame(origins) or _is_point_frame(destinations):
        from_ids, origin_points = _point_list(origins, "origins")
        if destinations is None:
            to_ids, destination_points = from_ids, origin_points
        else:
            to_ids, destination_points = _point_list(destinations, "destinations")
        rows = _chunk_slice(len(from_ids), chunk)
        from_ids = from_ids[rows]
        origin_points = origin_points[rows]
        table = network._core.travel_cost_matrix_from_points(
            origin_points,
            destination_points,
            date,
            departure,
            trip_factors,
            max_transfers,
            *_walk_options(walking_speed_kmph, max_walking_time, max_snap_distance),
            geometries,
        )
        _warn_unsnapped(table, from_ids, to_ids)
    else:
        if not (
            walking_speed_kmph is None
            and max_walking_time is None
            and max_snap_distance is None
        ):
            raise ValueError("walking options apply to point origins/destinations")
        stop_ids = [stop for stop, _, _ in network.stops]
        from_ids = list(stop_ids) if origins is None else [str(o) for o in origins]
        from_ids = from_ids[_chunk_slice(len(from_ids), chunk)]
        to_stops = None if destinations is None else [str(d) for d in destinations]
        table = network._core.travel_cost_matrix(
            from_ids,
            date,
            departure,
            trip_factors,
            max_transfers,
            to_stops,
            geometries,
        )
        to_ids = stop_ids
    return table, from_ids, to_ids


def _chunk_slice(count, chunk):
    """The deterministic contiguous origin block ``chunk = (k, n)``
    selects: chunk ``k`` of ``n`` equal blocks (the last possibly
    shorter), covering all origins disjointly across ``k = 0..n-1``."""
    if chunk is None:
        return slice(None)
    index, total = chunk
    index, total = int(index), int(total)
    if total < 1 or not 0 <= index < total:
        raise ValueError("chunk must be (k, n) with 0 <= k < n")
    size = -(-count // total)
    return slice(index * size, min((index + 1) * size, count))


def _is_point_frame(value):
    return value is not None and hasattr(value, "geometry")


def _point_list(frame, role):
    """A point GeoDataFrame's ids and ``(lat, lon)`` pairs, in EPSG:4326."""
    if not _is_point_frame(frame):
        raise TypeError(f"{role} must be a point GeoDataFrame when points are used")
    if "id" not in frame.columns:
        raise ValueError(f"the {role} GeoDataFrame needs an 'id' column")
    if frame.crs is not None:
        frame = frame.to_crs("EPSG:4326")
    geometry = frame.geometry
    if not (geometry.geom_type == "Point").all():
        raise ValueError(f"the {role} GeoDataFrame must contain points")
    ids = [str(identifier) for identifier in frame["id"]]
    return ids, list(zip(geometry.y, geometry.x))


def _warn_unsnapped(table, from_ids, to_ids):
    """Warn about points off the walking network, naming the first few."""
    for key, ids, side in (
        ("unsnapped_from", from_ids, "origin"),
        ("unsnapped_to", to_ids, "destination"),
    ):
        missed = table.get(key)
        if missed is None or not len(missed):
            continue
        named = ", ".join(str(ids[index]) for index in missed[:5])
        suffix = ", …" if len(missed) > 5 else ""
        warnings.warn(
            f"{len(missed)} {side} point(s) are off the walking network "
            f"and unreachable ({named}{suffix})",
            stacklevel=3,
        )
