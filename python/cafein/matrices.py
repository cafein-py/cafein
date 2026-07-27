"""Matrix computers over a transport network."""

import warnings

import numpy as np
import pandas as pd
import shapely


class TravelCostMatrix(pd.DataFrame):
    """The fastest journey's aggregated costs per OD pair, long format.

    A pandas DataFrame with one row per reachable OD pair: ``from_id``
    and ``to_id``, ``travel_time`` (seconds), ``transfers``,
    ``transit_distance_m`` and ``walk_distance_m`` (meters), and
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
    the walk as ``walk_distance_m``. The access and egress walks
    count toward ``walk_distance_m``, and points off the walking network are
    reported with a warning and yield no rows. From stop origins,
    ``walk_distance_m`` covers transfers only — unless a whole-day
    (Mc)ULTRA set upgrades the stop matrix to door-to-door routing,
    which walks the access and egress ends too.

    One run of the selected engine (see ``router``) serves each
    origin, fanned out over all cores; each
    pair's costs come from its fastest journey (ties resolved toward
    fewer rides) — or, with ``optimize="emissions"`` or
    ``optimize="fare"``, from the cleanest or cheapest journey of a
    departure window, optionally within a travel-time budget.
    Unreachable pairs are absent. Requires a network built with trip
    distances (the default), and with leg geometries for
    ``geometries=True``. Slices and copies degrade to plain DataFrames.

    Given a ``StreetNetwork`` instead, the matrix is a standalone street
    computation over the compiled profile of ``transport_mode``, bounded
    by ``max_street_time``. Its columns are ``from_id``, ``to_id``,
    ``travel_time``, ``distance_m``, ``network_distance_m``,
    ``connector_distance_m``, ``distance_provenance``, ``emissions``,
    and — with ``geometries=True`` — the route as a shapely LineString. The
    distances carry their unit in the name: ``network_distance_m`` sums
    the stored edge lengths the route traversed and
    ``connector_distance_m`` the straight lines from each coordinate to
    its snap point, and ``distance_m`` is their sum, reported alongside
    rather than instead of them because the two are measured
    differently. ``emissions`` is the ride's
    grams CO₂e — ``network_distance_m`` at the mode's resolved factor
    (see ``cafein.emissions.street_factors``; connectors are excluded,
    and an unresolved factor reports NA rather than a silent zero).
    ``date``, ``departure``, and the other timetable-only arguments are
    rejected; ``factors=`` and ``components=`` configure the factor.

    Parameters
    ----------
    network : TransportNetwork or StreetNetwork
        The network to compute on. A ``StreetNetwork`` takes the
        standalone street path and requires ``transport_mode``.
    origins : list of str, or GeoDataFrame (optional)
        Origin stop_ids (every stop when omitted), or points with an
        ``id`` column. A street matrix needs points.
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
    exclude_routes, exclude_trips, exclude_stops : list of str (optional)
        GTFS ids of supply the journeys must not use - disruption and
        accessibility filters, as in ``route_between_stops``. Runs on
        the RAPTOR engines (``"auto"`` resolves to them).
    router : str (optional, default: "auto")
        The routing engine. With time candidates the engines are RAPTOR
        (``"raptor"``) and TBTR (``"tbtr"``), the latter over the
        cached time transfer set (``compute_tbtr_transfers``) when its
        date matches, else over a set built for the query; ``"auto"``
        runs on TBTR when the cached set matches the date — unless a
        whole-day ULTRA set serves the stop matrix door-to-door, which
        only the RAPTOR path does — else on RAPTOR. With
        ``candidates="pareto"`` the engines are McRAPTOR and McTBTR and
        ``"auto"`` requires the cached multicriteria set
        (``compute_mctbtr_transfers``) to match the query's date and
        factors, mirroring ``journey_frontier``.
    factors : DataFrame or path (optional)
        Extra emission-factor rows layered over the shipped defaults;
        see ``cafein.emissions.load_factors`` — or, for a
        ``StreetNetwork``, street-mode rows for
        ``cafein.emissions.load_street_factors``.
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
        The street-search options for the walking access/egress, as in
        ``TransportNetwork.access_stops``. They bound the walking for point
        origins/destinations, and for stop origins/destinations only when a
        whole-day shortcut set routes them door-to-door; otherwise stop
        matrices ignore them. Only ``max_snap_distance`` applies to a
        ``StreetNetwork``, whose speeds come from the mode's profile.
    transport_mode : str (optional)
        The mode to route. Required for a ``StreetNetwork``, where it is
        one of ``"walk"``, ``"bicycle"``, ``"e_bike"``, ``"e_scooter"``.
    max_street_time : float (optional)
        Cutoff in seconds for a ``StreetNetwork`` matrix (default:
        ``cafein.street_network.MAX_STREET_TIME``, 7200).

    ``street_policy=`` (a ``cafein.StreetLegPolicy``) opens the access
    and egress to the policy's street modes over the multimodal graph
    (build with ``street_modes=``); point origins and destinations only.
    The frame gains ``street_distance_m`` beside the transit and walking
    distances — the meters ridden on street vehicles at the journey
    ends, connectors included — and ``emissions`` adds each vehicle
    mode's street emissions over its network meters (a ``factors=``
    table keyed by ``street_mode`` configures the street ladder, as in
    ``DetailedItineraries``; NaN where unresolved, never a silent
    zero). ``walk_distance_m`` keeps genuine walking: walk-mode ends,
    carried transfer walks, and the mid-journey transfers. Row geometry
    stays the ridden transit legs', as in the legacy matrix — per-leg
    street shapes ride ``DetailedItineraries``. The direct
    walking alternative folds in at the policy's walking access budget,
    and a walking-only policy at one shared budget rides the legacy
    cost matrix bit for bit (its ``street_distance_m`` is identically
    zero). Policy cost matrices run the time-fastest engine arm and do
    not price fares; ``optimize``, ``window``, ``within``, ``fares``,
    ``candidates``, ``router``, and the walking knobs are rejected
    beside a policy rather than silently ignored.
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
        router="auto",
        exclude_routes=(),
        exclude_trips=(),
        exclude_stops=(),
        geometries=False,
        chunk=None,
        walking_speed_kmph=None,
        max_walking_time=None,
        max_snap_distance=None,
        transport_mode=None,
        max_street_time=None,
        street_policy=None,
    ):
        if _is_street_network(network):
            data = _street_cost_columns(
                network,
                origins,
                destinations,
                transport_mode=transport_mode,
                max_street_time=max_street_time,
                max_snap_distance=max_snap_distance,
                chunk=chunk,
                geometries=geometries,
                factors=factors,
                components=components,
                transit_only={
                    "date": date,
                    "departure": departure,
                    "window": window,
                    "within": within,
                    "fares": fares,
                    "walking_speed_kmph": walking_speed_kmph,
                    "max_walking_time": max_walking_time,
                    "max_transfers": None if max_transfers == 7 else max_transfers,
                    "optimize": None if optimize == "time" else optimize,
                    "candidates": None if candidates == "time" else candidates,
                    "bucket": None if bucket == 25.0 else bucket,
                    "router": None if router == "auto" else router,
                    "exclude_routes": tuple(exclude_routes) or None,
                    "exclude_trips": tuple(exclude_trips) or None,
                    "exclude_stops": tuple(exclude_stops) or None,
                    "street_policy": street_policy,
                },
            )
            super().__init__(pd.DataFrame(data))
            return
        if transport_mode is not None and transport_mode != "public_transport":
            raise ValueError(
                f"transport_mode={transport_mode!r} is a street mode; pass a "
                "StreetNetwork to route on it"
            )
        if max_street_time is not None:
            raise ValueError("max_street_time applies to a StreetNetwork matrix")
        if street_policy is not None:
            offending = next(
                (
                    name
                    for name, value in (
                        ("optimize", None if optimize == "time" else optimize),
                        ("window", window),
                        ("within", within),
                        ("fares", fares),
                        ("candidates", None if candidates == "time" else candidates),
                        ("router", None if router == "auto" else router),
                        ("walking_speed_kmph", walking_speed_kmph),
                        ("max_walking_time", max_walking_time),
                        ("max_snap_distance", max_snap_distance),
                    )
                    if value is not None
                ),
                None,
            )
            if offending is not None:
                raise ValueError(
                    f"street_policy does not combine with {offending}; the "
                    "policy carries its own budgets and the policy cost "
                    "matrix runs the time-fastest engine arm, unpriced"
                )
            walk_only, walk_budget = _walking_only_policy(street_policy)
            if walk_only:
                # A walking-only policy IS the legacy cost matrix, at the
                # policy's one walking budget. A street-mode factor table
                # configures street vehicles only, so it never reaches the
                # transit resolver — and walking rides none of them.
                transit_factors, _ = _factor_tables(factors)
                table, from_ids, to_ids = _cost_columns(
                    network,
                    origins,
                    destinations,
                    date,
                    departure,
                    max_transfers=max_transfers,
                    optimize="time",
                    window=None,
                    within=None,
                    factors=transit_factors,
                    components=components,
                    fares=None,
                    candidates="time",
                    bucket=bucket,
                    router="auto",
                    exclude_routes=exclude_routes,
                    exclude_trips=exclude_trips,
                    exclude_stops=exclude_stops,
                    geometries=geometries,
                    chunk=chunk,
                    walking_speed_kmph=None,
                    max_walking_time=walk_budget,
                    max_snap_distance=None,
                )
            else:
                table, from_ids, to_ids = _policy_cost_columns(
                    network,
                    origins,
                    destinations,
                    date,
                    departure,
                    street_policy,
                    max_transfers=max_transfers,
                    factors=factors,
                    components=components,
                    geometries=geometries,
                    chunk=chunk,
                    exclude_routes=exclude_routes,
                    exclude_trips=exclude_trips,
                    exclude_stops=exclude_stops,
                )
            data = {
                "from_id": np.array(from_ids, dtype=object)[table["from"]],
                "to_id": np.array(to_ids, dtype=object)[table["to"]],
                "travel_time_s": table["travel_time_s"],
                "transfers": np.maximum(table["rides"], 1) - 1,
                "transit_distance_m": table["transit_distance"],
                "walk_distance_m": table["walk_distance"],
                # The legacy fast path has no street vehicles, so its
                # street meters are identically zero.
                "street_distance_m": table.get(
                    "street_distance", np.zeros(len(table["travel_time_s"]))
                ),
                "emissions": table["emissions"],
            }
            if geometries:
                data["geometry"] = shapely.from_wkb(
                    np.array(table["geometry"], dtype=object)
                )
            super().__init__(pd.DataFrame(data))
            return
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
            exclude_routes=exclude_routes,
            exclude_trips=exclude_trips,
            exclude_stops=exclude_stops,
            geometries=geometries,
            chunk=chunk,
            walking_speed_kmph=walking_speed_kmph,
            max_walking_time=max_walking_time,
            max_snap_distance=max_snap_distance,
        )
        data = {
            "from_id": np.array(from_ids, dtype=object)[table["from"]],
            "to_id": np.array(to_ids, dtype=object)[table["to"]],
            "travel_time_s": table["travel_time_s"],
            "transfers": np.maximum(table["rides"], 1) - 1,
            "transit_distance_m": table["transit_distance"],
            "walk_distance_m": table["walk_distance"],
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

    Given a ``StreetNetwork`` instead, the matrix is a standalone street
    computation: one bounded search per origin over the compiled profile
    of ``transport_mode`` (``"walk"``, ``"bicycle"``, ``"e_bike"``, or
    ``"e_scooter"``), bounded by ``max_street_time``. It needs no
    timetable, so ``date`` and ``departure`` do not apply, and the
    arguments that only mean something to a timetable — ``max_transfers``,
    ``router``, the departure-window percentiles, the transit exclusions,
    and the walking-speed options, whose speeds come from the profile —
    are rejected. Origins and destinations are point GeoDataFrames;
    a point routes to itself in zero seconds.

    Parameters
    ----------
    network : TransportNetwork or StreetNetwork
        The network to compute on. A ``StreetNetwork`` takes the
        standalone street path and requires ``transport_mode``.
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
    router : str (optional, default: "auto")
        The routing engine: ``"raptor"``, or ``"tbtr"`` to precompute a
        TBTR day engine for the date and fan the origins out over it,
        for stop and point matrices alike; the results are identical.
        ``"auto"`` (the default) runs on TBTR when a cached transfer
        set (``compute_tbtr_transfers``) matches the date, except for
        stop matrices under a whole-day ULTRA set, where only the
        RAPTOR path routes door-to-door and auto prefers it; point
        matrices share the ULTRA set on both engines, so the cache
        alone decides there.
    walking_speed_kmph, max_walking_time, max_snap_distance : float
        The street-search options for the walking access/egress, as in
        ``TransportNetwork.access_stops``. They bound the walking for point
        origins/destinations, and for stop origins/destinations only when a
        whole-day shortcut set routes them door-to-door; otherwise stop
        matrices ignore them. Only ``max_snap_distance`` applies to a
        ``StreetNetwork``, whose speeds come from the mode's profile.
    transport_mode : str (optional)
        The mode to route. Required for a ``StreetNetwork``, where it is
        one of ``"walk"``, ``"bicycle"``, ``"e_bike"``, ``"e_scooter"``;
        for a ``TransportNetwork`` only ``"public_transport"`` (the
        default meaning) applies.
    max_street_time : float (optional)
        Cutoff in seconds for a ``StreetNetwork`` matrix, beyond which a
        destination counts as unreachable (default:
        ``cafein.street_network.MAX_STREET_TIME``, 7200).

    ``street_policy=`` (a ``cafein.StreetLegPolicy``) opens the access
    and egress to the policy's street modes over the carried multimodal
    graph: point-set origins and destinations only, exclusions honoured,
    a walking-only policy identical to the legacy walking matrix. It
    conflicts with the walking knobs, ``router``, and the departure-window
    parameters, which are rejected rather than silently ignored.
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
        router="auto",
        exclude_routes=(),
        exclude_trips=(),
        exclude_stops=(),
        walking_speed_kmph=None,
        max_walking_time=None,
        max_snap_distance=None,
        transport_mode=None,
        max_street_time=None,
        street_policy=None,
    ):
        if _is_street_network(network):
            data = _street_time_columns(
                network,
                origins,
                destinations,
                transport_mode=transport_mode,
                max_street_time=max_street_time,
                max_snap_distance=max_snap_distance,
                chunk=chunk,
                transit_only={
                    "date": date,
                    "departure": departure,
                    "window": window,
                    "percentiles": percentiles,
                    "confidence": confidence,
                    "walking_speed_kmph": walking_speed_kmph,
                    "max_walking_time": max_walking_time,
                    "max_transfers": None if max_transfers == 7 else max_transfers,
                    "router": None if router == "auto" else router,
                    "exclude_routes": tuple(exclude_routes) or None,
                    "exclude_trips": tuple(exclude_trips) or None,
                    "exclude_stops": tuple(exclude_stops) or None,
                    "street_policy": street_policy,
                },
            )
            super().__init__(pd.DataFrame(data))
            return
        if transport_mode is not None and transport_mode != "public_transport":
            raise ValueError(
                f"transport_mode={transport_mode!r} is a street mode; pass a "
                "StreetNetwork to route on it (street legs within a "
                "public-transport journey are a separate feature)"
            )
        if max_street_time is not None:
            raise ValueError("max_street_time applies to a StreetNetwork matrix")
        if street_policy is not None:
            rejected = {
                "window": window,
                "percentiles": percentiles,
                "confidence": confidence,
                "walking_speed_kmph": walking_speed_kmph,
                "max_walking_time": max_walking_time,
                "max_snap_distance": max_snap_distance,
            }
            named = [name for name, value in rejected.items() if value is not None]
            if named or router != "auto":
                offending = ", ".join(named) or f"router={router!r}"
                raise ValueError(
                    f"street_policy does not combine with {offending}; the "
                    "policy carries its own budgets and runs the "
                    "earliest-arrival engine"
                )
            walk_only, walk_budget = _walking_only_policy(street_policy)
            if walk_only:
                # A walking-only policy IS the legacy walking matrix, at the
                # policy's one walking budget.
                data = _time_columns(
                    network,
                    origins,
                    date,
                    departure,
                    max_transfers,
                    destinations=destinations,
                    window=None,
                    percentiles=None,
                    confidence=None,
                    chunk=chunk,
                    router="auto",
                    exclude_routes=exclude_routes,
                    exclude_trips=exclude_trips,
                    exclude_stops=exclude_stops,
                    walking_speed_kmph=None,
                    max_walking_time=walk_budget,
                    max_snap_distance=None,
                )
                super().__init__(pd.DataFrame(data))
                return
            data = _policy_time_columns(
                network,
                origins,
                destinations,
                date,
                departure,
                street_policy,
                max_transfers,
                chunk,
                exclude_routes,
                exclude_trips,
                exclude_stops,
            )
            super().__init__(pd.DataFrame(data))
            return
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
            exclude_routes=exclude_routes,
            exclude_trips=exclude_trips,
            exclude_stops=exclude_stops,
            walking_speed_kmph=walking_speed_kmph,
            max_walking_time=max_walking_time,
            max_snap_distance=max_snap_distance,
        )
        super().__init__(pd.DataFrame(data))


def _is_street_network(network):
    """Whether `network` is a standalone street network rather than a transit one."""
    from cafein.street_network import StreetNetwork

    return isinstance(network, StreetNetwork)


class _StreetQuery:
    """A validated street-matrix query: the resolved point sets and bounds."""

    def __init__(
        self,
        from_ids,
        to_ids,
        origin_points,
        destination_points,
        max_seconds,
        max_snap_distance,
    ):
        self.from_ids = from_ids
        self.to_ids = to_ids
        self.origin_points = origin_points
        self.destination_points = destination_points
        self.max_seconds = max_seconds
        self.max_snap_distance = max_snap_distance


def _street_query(
    origins,
    destinations,
    *,
    transport_mode,
    max_street_time,
    max_snap_distance,
    chunk,
    transit_only,
):
    """Validates a street-matrix call and resolves its points and bounds.

    Shared by every street matrix so their argument rules cannot drift apart.
    `transit_only` maps each timetable-only argument to its value, or to `None`
    when it was left at the default.
    """
    from cafein import street_network, streets

    if transport_mode is None:
        raise TypeError(
            "a StreetNetwork matrix needs an explicit transport_mode, one of "
            f"{', '.join(repr(mode) for mode in street_network.STREET_MODES)}"
        )
    if transport_mode == "public_transport":
        raise ValueError(
            "transport_mode='public_transport' needs a TransportNetwork; a "
            "StreetNetwork carries no timetable"
        )
    if transport_mode not in street_network.STREET_MODES:
        raise ValueError(
            f"unknown transport_mode {transport_mode!r}; expected one of "
            f"{', '.join(repr(mode) for mode in street_network.STREET_MODES)}"
        )
    unsupported = sorted(
        name for name, value in transit_only.items() if value is not None
    )
    if unsupported:
        raise ValueError(
            f"{', '.join(unsupported)} have no meaning for a street matrix, "
            "which carries no timetable and takes its speeds from the profile"
        )
    if not _is_point_frame(origins):
        raise TypeError(
            "a street matrix needs origins as a GeoDataFrame of points; stop "
            "ids belong to a TransportNetwork"
        )
    from_ids, origin_points = _point_list(origins, "origins")
    if destinations is None:
        to_ids, destination_points = list(from_ids), list(origin_points)
    elif not _is_point_frame(destinations):
        raise TypeError(
            "a street matrix needs destinations as a GeoDataFrame of points"
        )
    else:
        to_ids, destination_points = _point_list(destinations, "destinations")
    if chunk is not None:
        span = _chunk_slice(len(from_ids), chunk)
        from_ids, origin_points = from_ids[span], origin_points[span]
    return _StreetQuery(
        from_ids,
        to_ids,
        origin_points,
        destination_points,
        float(
            street_network.MAX_STREET_TIME
            if max_street_time is None
            else max_street_time
        ),
        float(
            streets.MAX_SNAP_DISTANCE
            if max_snap_distance is None
            else max_snap_distance
        ),
    )


def _street_cost_columns(
    network,
    origins,
    destinations,
    *,
    transport_mode,
    max_street_time,
    max_snap_distance,
    chunk,
    geometries,
    transit_only,
    factors=None,
    components=None,
):
    """The reachable cells of a street cost matrix, in long format."""
    from cafein import emissions
    from cafein._cafein import STREET_DISTANCE_PROVENANCE

    query = _street_query(
        origins,
        destinations,
        transport_mode=transport_mode,
        max_street_time=max_street_time,
        max_snap_distance=max_snap_distance,
        chunk=chunk,
        transit_only=transit_only,
    )
    table = network._core.cost_matrix(
        query.origin_points,
        query.destination_points,
        transport_mode,
        query.max_seconds,
        query.max_snap_distance,
        bool(geometries),
    )
    _warn_unsnapped(table, query.from_ids, query.to_ids)
    from_ids = np.asarray(query.from_ids, dtype=object)
    to_ids = np.asarray(query.to_ids, dtype=object)
    network_distance = table["network_distance"]
    connector_distance = table["connector_distance"]
    data = {
        "from_id": from_ids[table["from"]],
        "to_id": to_ids[table["to"]],
        "travel_time_s": table["travel_time_s"],
        # Reported alongside its parts, not instead of them: the two are
        # measured differently (stored edge lengths versus straight connectors).
        "distance_m": network_distance + connector_distance,
        "network_distance_m": network_distance,
        "connector_distance_m": connector_distance,
        "distance_provenance": np.full(
            len(network_distance), STREET_DISTANCE_PROVENANCE, dtype=object
        ),
        # One mode per matrix, so the factor resolves once; connectors are the
        # walk to the vehicle, not vehicle-kilometres, so network metres only.
        "emissions": network_distance
        / 1000.0
        * emissions.street_factor(transport_mode, factors, components),
    }
    if geometries:
        data["geometry"] = shapely.from_wkb(np.array(table["geometry"], dtype=object))
    return data


def _street_time_columns(
    network,
    origins,
    destinations,
    *,
    transport_mode,
    max_street_time,
    max_snap_distance,
    chunk,
    transit_only,
):
    """The reachable cells of a street travel-time matrix, in long format."""
    query = _street_query(
        origins,
        destinations,
        transport_mode=transport_mode,
        max_street_time=max_street_time,
        max_snap_distance=max_snap_distance,
        chunk=chunk,
        transit_only=transit_only,
    )
    from_ids, to_ids = query.from_ids, query.to_ids
    table = network._core.travel_time_matrix(
        query.origin_points,
        query.destination_points,
        transport_mode,
        query.max_seconds,
        query.max_snap_distance,
    )
    _warn_unsnapped(table, from_ids, to_ids)
    matrix = table["matrix"]
    from_ids = np.asarray(from_ids, dtype=object)
    to_ids = np.asarray(to_ids, dtype=object)
    rows, columns = np.nonzero(matrix != np.iinfo(np.uint32).max)
    return {
        "from_id": from_ids[rows],
        "to_id": to_ids[columns],
        "travel_time_s": matrix[rows, columns],
    }


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
    router="auto",
    exclude_routes=(),
    exclude_trips=(),
    exclude_stops=(),
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
        exclude_routes=exclude_routes,
        exclude_trips=exclude_trips,
        exclude_stops=exclude_stops,
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
            "travel_time_s": matrix[rows, columns],
        }
    rows, columns = np.nonzero((matrix != unreachable).any(axis=2))
    values = matrix[rows, columns, :].astype(float)
    values[values == unreachable] = np.nan
    data = {"from_id": from_ids[rows], "to_id": to_ids[columns]}
    for index, percentile in enumerate(resolved):
        data[f"travel_time_p{percentile:g}_s"] = values[:, index]
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
    router="auto",
    exclude_routes=(),
    exclude_trips=(),
    exclude_stops=(),
    walking_speed_kmph=None,
    max_walking_time=None,
    max_snap_distance=None,
):
    """The travel-cost matrix as a pyarrow Table — the shard-writing form.

    Semantics and parameters follow `TravelCostMatrix` — including the
    windowed optimize modes with their ``window``/``within``, the
    ``fares`` pricing, and the ``router`` engine choice, though always
    over the time candidates
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
        router=router,
        exclude_routes=exclude_routes,
        exclude_trips=exclude_trips,
        exclude_stops=exclude_stops,
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
        "travel_time_s": pyarrow.array(table["travel_time_s"]),
        "transfers": pyarrow.array(np.maximum(table["rides"], 1) - 1),
        "transit_distance_m": pyarrow.array(table["transit_distance"]),
        "walk_distance_m": pyarrow.array(table["walk_distance"]),
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
    router="auto",
    exclude_routes=(),
    exclude_trips=(),
    exclude_stops=(),
):
    """The core's cost arrays plus the origin and destination ids."""
    exclusions = (
        [str(route) for route in exclude_routes],
        [str(trip) for trip in exclude_trips],
        [str(stop) for stop in exclude_stops],
    )
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
    if router not in ("auto", "raptor", "tbtr"):
        raise ValueError("router must be 'auto', 'raptor', or 'tbtr'")
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
                router,
                *exclusions,
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
                router,
                *exclusions,
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
            # The emissions (McRAPTOR) stop matrix relaxes a matching whole-day
            # McULTRA set for the pareto objective, routing door-to-door with
            # these walking options; otherwise it keeps the closure and ignores
            # them, as the time path does.
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
                *exclusions,
                *_walk_options(walking_speed_kmph, max_walking_time, max_snap_distance),
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
                router,
                *exclusions,
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


def _walking_only_policy(policy):
    """Whether the policy grants walking only at one shared budget — such a
    policy is the legacy walking path bit for bit. Distinct access and
    egress walking budgets run over the multimodal graph instead."""
    from cafein import streets as _streets

    sides = [
        side if side is not None else {"walk": _streets.MAX_ACCESS_EGRESS_TIME}
        for side in (policy.access, policy.egress)
    ]
    if any(mode != "walk" for side in sides for mode in side):
        return False, None
    budgets = {side["walk"] for side in sides}
    if len(budgets) > 1:
        # Distinct walking budgets cannot map onto the legacy path's one
        # cutoff; such a policy runs over the multimodal graph instead.
        return False, None
    return True, budgets.pop()


def _policy_time_columns(
    network,
    origins,
    destinations,
    date,
    departure,
    policy,
    max_transfers,
    chunk,
    exclude_routes=(),
    exclude_trips=(),
    exclude_stops=(),
):
    """The street-policy travel-time matrix columns: per-point reductions
    through the engine fan-out, the direct walking alternative folded in."""
    from cafein import streets as _streets
    from cafein.policy import reduction_modes

    core = network._core
    if not core.has_multimodal_streets:
        raise ValueError(
            "street_policy needs the multimodal street graph; build with "
            "street_modes="
        )
    if not _is_point_frame(origins):
        raise ValueError(
            "street_policy matrices take point-set origins and destinations"
        )
    # Materialised once: a one-shot iterable must not exhaust between the
    # per-point reductions, and later mutation must not desynchronise them.
    exclude_routes = tuple(str(route) for route in exclude_routes)
    exclude_trips = tuple(str(trip) for trip in exclude_trips)
    exclude_stops = tuple(str(stop) for stop in exclude_stops)
    from_ids, origin_points = _point_list(origins, "origins")
    if destinations is None:
        to_ids, destination_points = from_ids, origin_points
    else:
        to_ids, destination_points = _point_list(destinations, "destinations")
    rows_slice = _chunk_slice(len(from_ids), chunk)
    from_ids = from_ids[rows_slice]
    origin_points = origin_points[rows_slice]
    access_modes = reduction_modes(policy, "access", _streets.MAX_ACCESS_EGRESS_TIME)
    egress_modes = reduction_modes(policy, "egress", _streets.MAX_ACCESS_EGRESS_TIME)

    def reduced(points, egress, modes):
        rows, unsnapped = [], []
        for index, (lat, lon) in enumerate(points):
            try:
                rows.append(
                    [
                        (stop, seconds)
                        for stop, seconds, *_ in core._reduced_street_offsets(
                            lat,
                            lon,
                            egress,
                            modes,
                            exclude_stops=list(exclude_stops),
                        )
                    ]
                )
            except ValueError as error:
                if "too far from the multimodal street network" not in str(error):
                    raise
                # An unsnapped point reaches nothing; its cells are omitted.
                rows.append([])
                unsnapped.append(index)
        return rows, unsnapped

    access_rows, unsnapped_origins = reduced(origin_points, False, access_modes)
    egress_rows, unsnapped_destinations = reduced(
        destination_points, True, egress_modes
    )
    matrix = core._time_matrix_with_access(
        access_rows,
        egress_rows,
        date,
        departure,
        max_transfers,
        exclude_routes=list(exclude_routes),
        exclude_trips=list(exclude_trips),
        exclude_stops=list(exclude_stops),
    )
    # Walking directly can beat riding, exactly as in the walking matrix;
    # the alternative runs over the same multimodal graph at the policy's
    # walking access budget.
    # The direct walking alternative always applies — walking needs no
    # vehicle — at the policy's walking access budget when it names one,
    # else the usual door-to-door cutoff.
    access_budgets = (
        policy.access
        if policy.access is not None
        else {"walk": _streets.MAX_ACCESS_EGRESS_TIME}
    )
    walk_budget = access_budgets.get("walk", _streets.MAX_ACCESS_EGRESS_TIME)
    direct, walk_unsnapped_from, walk_unsnapped_to = core._multimodal_direct_matrix(
        list(origin_points), list(destination_points), "walk", float(walk_budget)
    )
    # A point is unsnapped only when neither the policy's modes nor the
    # direct walking alternative can snap it — a snap fact from both
    # searches, never inferred from reachability.
    _warn_unsnapped(
        {
            "unsnapped_from": sorted(
                set(map(int, unsnapped_origins)) & set(map(int, walk_unsnapped_from))
            ),
            "unsnapped_to": sorted(
                set(map(int, unsnapped_destinations)) & set(map(int, walk_unsnapped_to))
            ),
        },
        from_ids,
        to_ids,
    )
    data = {"from_id": [], "to_id": [], "travel_time_s": []}
    for i, from_id in enumerate(from_ids):
        for j, to_id in enumerate(to_ids):
            best = matrix[i][j]
            if direct is not None and direct[i][j] is not None:
                best = direct[i][j] if best is None else min(best, direct[i][j])
            if best is None:
                # The matrix omits unreachable pairs, as it always has.
                continue
            data["from_id"].append(from_id)
            data["to_id"].append(to_id)
            data["travel_time_s"].append(best)
    return data


def _factor_tables(factors):
    """``factors=`` split by schema for the policy products: a table keyed
    by ``street_mode`` configures the street ladder and leaves the transit
    legs on the shipped defaults; anything else layers over the transit
    ladder as always."""
    import pathlib

    from cafein import emissions

    if factors is None:
        return None, None
    frame = (
        factors
        if isinstance(factors, pd.DataFrame)
        else emissions._read_factor_file(pathlib.Path(factors))
    )
    if "street_mode" in frame.columns:
        return None, frame
    return factors, None


def _policy_cost_columns(
    network,
    origins,
    destinations,
    date,
    departure,
    policy,
    *,
    max_transfers,
    factors,
    components,
    geometries,
    chunk,
    exclude_routes=(),
    exclude_trips=(),
    exclude_stops=(),
):
    """The street-policy cost matrix columns: per-point meters-carrying
    reductions through the engine fan-out, street distances and emissions
    attributed per row from the winning choices, and the direct walking
    alternative folded in over the same multimodal graph."""
    from cafein import emissions
    from cafein import streets as _streets
    from cafein.policy import reduction_modes

    core = network._core
    if not core.has_multimodal_streets:
        raise ValueError(
            "street_policy needs the multimodal street graph; build with "
            "street_modes="
        )
    if not _is_point_frame(origins):
        raise ValueError(
            "street_policy matrices take point-set origins and destinations"
        )
    # Materialised once: a one-shot iterable must not exhaust between the
    # per-point reductions, and later mutation must not desynchronise them.
    exclude_routes = tuple(str(route) for route in exclude_routes)
    exclude_trips = tuple(str(trip) for trip in exclude_trips)
    exclude_stops = tuple(str(stop) for stop in exclude_stops)
    from_ids, origin_points = _point_list(origins, "origins")
    if destinations is None:
        to_ids, destination_points = from_ids, origin_points
    else:
        to_ids, destination_points = _point_list(destinations, "destinations")
    rows_slice = _chunk_slice(len(from_ids), chunk)
    from_ids = from_ids[rows_slice]
    origin_points = origin_points[rows_slice]
    access_modes = reduction_modes(policy, "access", _streets.MAX_ACCESS_EGRESS_TIME)
    egress_modes = reduction_modes(policy, "egress", _streets.MAX_ACCESS_EGRESS_TIME)
    transit_factors, street_factors = _factor_tables(factors)
    trip_factors = emissions.trip_factors(network, transit_factors, components)
    # One resolved per-km factor per granted vehicle mode; NaN keeps an
    # unresolved factor poisoning rather than zeroing its rows. Walking
    # rides no vehicle, so its factor is never read.
    mode_factors = {"walk": 0.0}
    for mode, *_ in access_modes + egress_modes:
        if mode in mode_factors:
            continue
        value = emissions.street_factor(mode, street_factors, components)
        mode_factors[mode] = float("nan") if pd.isna(value) else float(value)

    def reduced(points, egress, modes):
        rows, unsnapped = [], []
        for index, (lat, lon) in enumerate(points):
            try:
                cells = core._reduced_street_rows(
                    lat, lon, egress, modes, exclude_stops=list(exclude_stops)
                )
            except ValueError as error:
                if "too far from the multimodal street network" not in str(error):
                    raise
                # An unsnapped point reaches nothing; its cells are omitted.
                rows.append([])
                unsnapped.append(index)
                continue
            rows.append(
                [
                    (stop, seconds, network_m, connector_m, walk_m, mode_factors[mode])
                    for stop, seconds, mode, network_m, connector_m, walk_m in cells
                ]
            )
        return rows, unsnapped

    access_rows, unsnapped_origins = reduced(origin_points, False, access_modes)
    egress_rows, unsnapped_destinations = reduced(
        destination_points, True, egress_modes
    )
    # The direct walking alternative always applies — walking needs no
    # vehicle — at the policy's walking access budget when it names one,
    # else the usual door-to-door cutoff.
    access_budgets = (
        policy.access
        if policy.access is not None
        else {"walk": _streets.MAX_ACCESS_EGRESS_TIME}
    )
    walk_budget = access_budgets.get("walk", _streets.MAX_ACCESS_EGRESS_TIME)
    table = core._cost_matrix_with_access(
        access_rows,
        egress_rows,
        list(origin_points),
        list(destination_points),
        date,
        departure,
        trip_factors,
        float(walk_budget),
        max_transfers,
        exclude_routes=list(exclude_routes),
        exclude_trips=list(exclude_trips),
        exclude_stops=list(exclude_stops),
        geometries=bool(geometries),
    )
    # A point is unsnapped only when neither the policy's modes nor the
    # direct walking alternative can snap it — a snap fact from both
    # searches, never inferred from reachability.
    _warn_unsnapped(
        {
            "unsnapped_from": sorted(
                set(map(int, unsnapped_origins))
                & set(map(int, table["unsnapped_from"]))
            ),
            "unsnapped_to": sorted(
                set(map(int, unsnapped_destinations))
                & set(map(int, table["unsnapped_to"]))
            ),
        },
        from_ids,
        to_ids,
    )
    return table, from_ids, to_ids
