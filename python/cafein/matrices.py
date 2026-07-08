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
    travel time is its fastest walk–ride–walk chain or the direct street
    walk (within ``max_walking_time``), whichever is faster — a
    walking-only pair reports zero ``transfers``, zero emissions, and
    the walk as ``walk_distance``. The access and egress walks count
    toward ``walk_distance``, and points off the walking network are
    reported with a warning and yield no rows. From stop origins,
    ``walk_distance`` covers transfers only.

    One RAPTOR run serves each origin, fanned out over all cores; each
    pair's costs come from its fastest journey (ties resolved toward
    fewer rides) — or, with ``optimize="emissions"`` or
    ``optimize="fare"``, from the cleanest or cheapest journey of a
    departure window, optionally within a travel-time budget.
    Unreachable pairs are absent. Requires a network built with trip
    distances (the default), and with leg geometries for
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
    max_transfers : int (optional, default: 7)
        Maximum number of transfers between rides.
    optimize : str (optional, default: "time")
        What each cell's journey minimises. ``"time"`` (the default)
        reports the fastest journey. ``"emissions"`` and ``"fare"``
        report the lowest-emission or cheapest journey among the
        departure window's (departure, arrival, rides)-Pareto
        candidates — the same ride candidates ``journey_frontier``
        sees — optionally within the ``within`` travel-time budget. A
        zero-ride floor (zero emissions, zero fare) joins the
        candidates: for stop pairs the origin itself, for point pairs
        the walking-only alternative, which wins any cell it qualifies
        for. Each objective qualifies candidates by its own key: NaN
        emissions drop a candidate under ``"emissions"``, an
        unpriceable fare under ``"fare"`` — pairs with no qualifying
        candidate are absent.
    window : int (optional)
        Departure window in seconds; required with
        ``optimize="emissions"`` and ``optimize="fare"``.
    within : int (optional)
        Travel-time budget in seconds for the windowed optimize modes:
        only journeys at most this long qualify. Unbudgeted, the
        cleanest (cheapest) reachable journey wins.
    candidates : str (optional, default: "time")
        The candidate journey set of the windowed optimize modes.
        ``"pareto"`` (with ``optimize="emissions"``, stop origins and
        destinations) draws each cell's candidates from McRAPTOR's
        (departure, arrival, emissions) Pareto set, which also holds
        the cleaner-but-slower journeys the time-optimal set misses —
        cells can report strictly lower emissions, at more compute per
        origin.
    bucket : float (optional, default: 25.0)
        The emissions bucket width in grams CO₂e of the pareto search,
        as in ``journey_frontier``. Only used with
        ``candidates="pareto"``.
    router : str (optional, default: "raptor")
        The pareto search engine, as in ``journey_frontier``: McRAPTOR
        (``"raptor"``) or McTBTR (``"tbtr"``), which precomputes the
        date's multicriteria transfer set once and fans every origin
        out over it. Only used with ``candidates="pareto"``.
    factors : DataFrame or path (optional)
        Extra emission-factor rows layered over the shipped defaults;
        see ``cafein.emissions.load_factors``.
    components : list of str (optional)
        The life-cycle components to include (default: all four); see
        ``cafein.emissions.annotate``.
    fares : FareStructure or ZoneFareStructure (optional)
        A fare model (see ``cafein.fares``); adds a ``fare`` column
        with each cell's journey priced (NaN where the model cannot
        price it), and is required for ``optimize="fare"``.
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
        max_transfers=7,
        optimize="time",
        window=None,
        within=None,
        factors=None,
        components=None,
        fares=None,
        candidates="time",
        bucket=25.0,
        router="raptor",
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
            optimize=optimize,
            window=window,
            within=within,
            factors=factors,
            components=components,
            fares=fares,
            candidates=candidates,
            bucket=bucket,
            router=router,
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
        if fares is not None:
            data["fare"] = table["fare"]
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
    max_transfers : int (optional, default: 7)
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
    router : str (optional, default: "raptor")
        The routing engine for single-departure stop matrices:
        ``"raptor"``, or ``"tbtr"`` to precompute a TBTR day engine for
        the date and fan the origins out over it; the results are
        identical. Windowed and point matrices run on RAPTOR only.
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
        max_transfers=7,
        window=None,
        percentiles=None,
        confidence=None,
        chunk=None,
        router="raptor",
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
            router=router,
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
    router="raptor",
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
        router=router,
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
    max_transfers=7,
    optimize="time",
    window=None,
    within=None,
    factors=None,
    components=None,
    fares=None,
    geometries=False,
    chunk=None,
    walking_speed_kmph=None,
    max_walking_time=None,
    max_snap_distance=None,
):
    """The travel-cost matrix as a pyarrow Table — the shard-writing form.

    Semantics and parameters follow `TravelCostMatrix` — including the
    windowed optimize modes with their ``window``/``within`` and the
    ``fares`` pricing, though always over the time candidates
    (no ``candidates``/``bucket``); the output is an
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
        optimize=optimize,
        window=window,
        within=within,
        factors=factors,
        components=components,
        fares=fares,
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
    if fares is not None:
        columns["fare"] = pyarrow.array(table["fare"])
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
    optimize="time",
    window=None,
    within=None,
    fares=None,
    candidates="time",
    bucket=25.0,
    router="raptor",
):
    """The core's cost arrays plus the origin and destination ids."""
    from cafein import emissions
    from cafein.network import _walk_options

    if date is None or departure is None:
        raise TypeError("TravelCostMatrix requires date and departure")
    if optimize not in ("time", "emissions", "fare"):
        raise ValueError(
            f"optimize must be 'time', 'emissions', or 'fare', not {optimize!r}"
        )
    if optimize != "time" and window is None:
        raise ValueError(f"optimize={optimize!r} requires a departure window")
    if optimize == "time" and not (window is None and within is None):
        raise ValueError("window and within require optimize='emissions' or 'fare'")
    if optimize == "fare" and fares is None:
        raise ValueError("optimize='fare' requires a fare structure (fares=)")
    if candidates not in ("time", "pareto"):
        raise ValueError("candidates must be 'time' or 'pareto'")
    if router not in ("raptor", "tbtr"):
        raise ValueError("router must be 'raptor' or 'tbtr'")
    if router == "tbtr" and candidates != "pareto":
        raise ValueError("router='tbtr' requires candidates='pareto'")
    if candidates == "pareto":
        if optimize != "emissions":
            raise ValueError("candidates='pareto' requires optimize='emissions'")
        if _is_point_frame(origins) or _is_point_frame(destinations):
            raise ValueError("pareto candidates require stop origins and destinations")
    fare_tables = None if fares is None else fares._flat_tables(network)
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
        walk = _walk_options(walking_speed_kmph, max_walking_time, max_snap_distance)
        if optimize != "time":
            table = network._core.least_cost_matrix_from_points(
                origin_points,
                destination_points,
                date,
                departure,
                window,
                trip_factors,
                optimize,
                fare_tables,
                within,
                max_transfers,
                *walk,
                geometries,
            )
        else:
            table = network._core.travel_cost_matrix_from_points(
                origin_points,
                destination_points,
                date,
                departure,
                trip_factors,
                max_transfers,
                *walk,
                geometries,
                fare_tables,
            )
        _warn_unsnapped(table, from_ids, to_ids)
    else:
        stop_ids = [stop for stop, _, _ in network.stops]
        from_ids = list(stop_ids) if origins is None else [str(o) for o in origins]
        from_ids = from_ids[_chunk_slice(len(from_ids), chunk)]
        to_stops = None if destinations is None else [str(d) for d in destinations]
        if optimize != "time":
            # The emissions/fare (McRAPTOR) stop matrix keeps the closure until
            # McULTRA, so it takes no walking options.
            if not (
                walking_speed_kmph is None
                and max_walking_time is None
                and max_snap_distance is None
            ):
                raise ValueError("walking options apply to point origins/destinations")
            table = network._core.least_cost_matrix(
                from_ids,
                date,
                departure,
                window,
                trip_factors,
                optimize,
                fare_tables,
                within,
                max_transfers,
                to_stops,
                candidates,
                bucket,
                router,
                geometries,
            )
        else:
            # The walking options bound the door-to-door cost matrix under a
            # whole-day ULTRA set; they are ignored on the closure path.
            table = network._core.travel_cost_matrix(
                from_ids,
                date,
                departure,
                trip_factors,
                max_transfers,
                to_stops,
                *_walk_options(walking_speed_kmph, max_walking_time, max_snap_distance),
                geometries,
                fare_tables,
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
