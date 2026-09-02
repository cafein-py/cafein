"""Matrix computers over a transport network."""

import collections.abc
import dataclasses
import functools
import operator
import warnings

import numpy as np
import pandas as pd

from cafein import _log
import shapely

from cafein._validate import (
    positive_int,
    component_selection,
    freeze_ids,
    id_sequence,
    non_negative_finite,
    restore_id_dtypes,
    sequence_not_string,
)
from cafein import _memory
from cafein._units import duration_seconds, memory_spec
from cafein.travelers import (
    folded_constraints,
    folded_street_policy,
    refuse_wheelchair_streets,
)

#: One long-format cost row (times, transfers, distances, emissions)
#: and one dense time cell, for the planner's result reservations.
_COST_ROW_BYTES = 48
_TIME_CELL_BYTES = 4
#: The long-format time frame at its peak holds, per cell, the dense
#: uint32 matrix (4), the row and column indices from nonzero (16),
#: the two object-id arrays (16), the indexed values (4), and the
#: frame's own copies of the columns while it is built (~24).
_TIME_FRAME_CELL_BYTES = 64
#: Every optional column (a perspective, a component, an exposure
#: layer) is one float64 per row.
_OPTIONAL_COLUMN_BYTES = 8
#: A slot's moment column (a timestamp string per row) and a mapping's
#: label column, as strings in a frame or an Arrow batch.
_MOMENT_COLUMN_BYTES = 64
_SLOT_LABEL_BYTES = 48


def _query_details(arguments):
    """Best-effort shape facts for a matrix phase."""
    details = {}
    for key in ("origins", "destinations"):
        count = _log.sized(arguments.get(key))
        if count is not None:
            details[key] = count
    for key in ("departure_time_window", "arrival_time_window"):
        if arguments.get(key) is not None:
            details["window"] = arguments[key]
    for key in ("departure", "arrival"):
        if isinstance(arguments.get(key), (list, tuple, collections.abc.Mapping)):
            details["slots"] = len(arguments[key])
    if arguments.get("max_memory") is not None:
        details["max_memory"] = arguments["max_memory"]
    chunk = arguments.get("chunk")
    if chunk is not None:
        try:
            details["chunk"] = tuple(int(part) for part in chunk)
        except Exception:
            pass
    return details


def _slot_shape(departure, arrival):
    """``(count, moment_column, label_bytes)``: how many slots a
    departure or arrival argument names, whether the result carries the
    moment column every list or mapping adds, and the bytes per row of
    the label column a mapping adds (``0`` without one)."""
    for value in (departure, arrival):
        if isinstance(value, collections.abc.Mapping):
            # A label column: its longest label's bytes plus an offset,
            # as the plain string column a batch materializes first.
            longest = max((len(str(key).encode("utf-8")) for key in value), default=0)
            return max(1, len(value)), True, longest + 8
        if isinstance(value, (list, tuple)):
            return max(1, len(value)), True, 0
    return 1, False, 0


def _chunked(count, chunk):
    """How many of ``count`` a validated ``chunk`` selects."""
    if chunk is None:
        return count
    try:
        chunk = tuple(int(part) for part in chunk)
        return len(range(count)[_chunk_slice(count, chunk)])
    except (TypeError, ValueError):
        return count  # the body refuses the malformed chunk


def _optional_columns(kwargs, exposure_snapshot):
    """The optional result columns a call selects, for its row estimate."""
    count = 0
    for key in ("perspectives", "cost_components", "components"):
        count += _log.sized(kwargs.get(key)) or 0
    if exposure_snapshot is not None:
        count += len(exposure_snapshot.column_names())
    return count


def _exposure_snapshot(exposure):
    """The reporting snapshot the enclosing call was planned with, so
    the body reports against the surface the plan sized; a fresh one
    for a call nobody planned."""
    active = _memory.active_plan()
    if isinstance(active, _memory.Refusal):
        raise active.error
    if active is not None and active.exposure_snapshot is not None:
        return active.exposure_snapshot
    return exposure._reporting_snapshot()


def _entry_plan(
    network, origins, destinations, departure, kwargs, *, cost, streamed, dense=False
):
    """Plan a public matrix call once: its result (or one batch) and
    its width, from the engine the objective needs and the network's
    size. Percentiles resolve here so a one-shot iterable is read once;
    """
    from cafein.network import _window_percentiles

    workers = kwargs.get("workers")
    if workers is not None:
        workers = positive_int("workers", workers)
    max_memory = kwargs.get("max_memory")
    memory_spec("max_memory", max_memory)
    arrival = kwargs.get("arrival")
    arrive_by = arrival is not None
    street = _is_street_network(network)
    points = street or _is_point_frame(origins)
    origin_count = _log.sized(origins) if origins is not None else None
    if origin_count is None:
        origin_count = 1 if points else network.stop_count
    destination_count = _log.sized(destinations) if destinations is not None else None
    if destination_count is None:
        destination_count = origin_count if points else network.stop_count
    # The chunk slices the fan-out axis: destinations for an arrive-by
    # run, origins otherwise.
    chunk = kwargs.get("chunk")
    if arrive_by:
        destination_count = _chunked(destination_count, chunk)
    else:
        origin_count = _chunked(origin_count, chunk)
    slots, moment_column, label_bytes = _slot_shape(departure, arrival)
    if street:
        engine, size = "street", network._core.vertex_count
    elif cost:
        engine, size = (
            "time" if kwargs.get("optimize", "time") == "time" else "multicriteria"
        ), network.stop_count
    else:
        engine, size = "time", network.stop_count
    if cost:
        row_bytes = _COST_ROW_BYTES + (
            _memory.GEOMETRY_ROW_BYTES if kwargs.get("geometries") else 0
        )
    else:
        window = kwargs.get(
            "arrival_time_window" if arrive_by else "departure_time_window"
        )
        percentiles = kwargs.get("percentiles")
        if not street and (window is not None or percentiles is not None):
            resolved = _window_percentiles(
                duration_seconds(
                    "arrival_time_window" if arrive_by else "departure_time_window",
                    window,
                ),
                percentiles,
                kwargs.get("confidence"),
            )
            if percentiles is not None:
                # A one-shot iterable, read once, reaches the body as a list.
                kwargs["percentiles"] = resolved
            planes = len(resolved) if resolved else 1
        else:
            planes = 1
        row_bytes = (_TIME_CELL_BYTES if dense else _TIME_FRAME_CELL_BYTES) * planes
    exposure = kwargs.get("exposure")
    # One snapshot per call: the plan sizes its columns and the body
    # reports against it; the live object still validates the network.
    snapshot = None if exposure is None else exposure._reporting_snapshot()
    row_bytes += _OPTIONAL_COLUMN_BYTES * _optional_columns(kwargs, snapshot)
    if moment_column:
        row_bytes += _MOMENT_COLUMN_BYTES
    if label_bytes:
        row_bytes += max(_SLOT_LABEL_BYTES, label_bytes)
    kind = "cost" if cost else "time"
    if streamed:
        row_bytes += 24  # the streamed row's index and buffer bytes
        # The batch axis: origins, or destinations for an arrive-by run.
        columns = origin_count if arrive_by else destination_count
        batch = kwargs.get("batch_size")
        if batch is not None:
            batch = _stream_size(batch, False)
        plan = _memory.plan_call(
            engine,
            size,
            0,
            streamed=True,
            row_bytes=columns * row_bytes * slots,
            batch_rows=batch,
            workers=workers,
            max_memory=max_memory,
            label=f"travel {kind} stream",
        )
    else:
        plan = _memory.plan_call(
            engine,
            size,
            origin_count * destination_count * row_bytes * slots,
            workers=workers,
            max_memory=max_memory,
            label=f"travel {kind} matrix",
        )
    return dataclasses.replace(plan, exposure_snapshot=snapshot)


def _frozen_axis(value):
    """An axis argument read once: a one-shot iterable of ids becomes a
    list, so the plan and the body see the same ids; frames, sized
    sequences, strings, and ``None`` pass through."""
    if value is None or isinstance(value, (str, bytes)) or hasattr(value, "__len__"):
        return value
    return list(value)


def _deferred(plan):
    """The plan, or the refusal it raised, deferred to the first dispatch
    so the call's own argument checks refuse first."""
    try:
        return plan()
    except ValueError as error:
        return _memory.Refusal(error)


def _planned_entry(*, cost, streamed=False, method=False, function=False):
    """Plan the decorated public call once and keep its plan active for
    every dispatch inside it. ``method`` marks a network method, whose
    ``self`` is the network and whose positional tail is
    ``(origins, departure, max_rides)``."""

    def decorate(fn):
        if function:

            @functools.wraps(fn)
            def wrapper(
                network, origins=None, destinations=None, departure=None, **kwargs
            ):
                origins, destinations = _frozen_axis(origins), _frozen_axis(
                    destinations
                )
                # "auto": streamed exactly when an output is named.
                streams = (
                    kwargs.get("output") is not None if streamed == "auto" else streamed
                )
                plan = _deferred(
                    lambda: _entry_plan(
                        network,
                        origins,
                        destinations,
                        departure,
                        kwargs,
                        cost=cost,
                        streamed=streams,
                    )
                )
                with _memory.use_plan(plan):
                    return fn(network, origins, destinations, departure, **kwargs)

            return wrapper

        if method:

            @functools.wraps(fn)
            def wrapper(self, origins, departure=None, max_rides=8, **kwargs):
                origins = _frozen_axis(origins)
                if kwargs.get("destinations") is not None:
                    kwargs["destinations"] = _frozen_axis(kwargs["destinations"])
                plan = _deferred(
                    lambda: _entry_plan(
                        self,
                        origins,
                        kwargs.get("destinations"),
                        departure,
                        kwargs,
                        cost=cost,
                        streamed=streamed,
                        dense=True,
                    )
                )
                with _memory.use_plan(plan):
                    return fn(self, origins, departure, max_rides, **kwargs)

            return wrapper

        @functools.wraps(fn)
        def wrapper(
            first, network, origins=None, destinations=None, departure=None, **kwargs
        ):
            origins, destinations = _frozen_axis(origins), _frozen_axis(destinations)
            plan = _deferred(
                lambda: _entry_plan(
                    network,
                    origins,
                    destinations,
                    departure,
                    kwargs,
                    cost=cost,
                    streamed=streamed,
                )
            )
            with _memory.use_plan(plan):
                return fn(first, network, origins, destinations, departure, **kwargs)

        return wrapper

    return decorate


class TravelCostMatrix(pd.DataFrame):
    """The fastest journey's aggregated costs per OD pair, long format.

    A pandas DataFrame with one row per reachable OD pair: ``from_id``
    and ``to_id``, ``travel_time`` (whole minutes rounded to the
    nearest by default; exact seconds with
    ``output_time_units="seconds"``), ``transfers``,
    ``transit_distance_m`` and ``walk_distance_m`` (meters), and
    ``emissions`` (grams CO₂e over the ridden legs; NaN where a ridden
    trip has no matching factor row). With ``geometries=True`` each row
    adds ``geometry``, the ridden legs as a shapely MultiLineString in
    EPSG:4326 — convert with
    ``geopandas.GeoDataFrame(matrix, crs="EPSG:4326")``.

    Origins and destinations are either stop identifiers or
    GeoDataFrames with an ``id`` column — point frames, or polygon
    frames routed from their centroids
    (``centroid_lat``/``centroid_lon`` columns when present — the
    ``cafein.zones`` protocol — otherwise local-UTM centroids). Points
    are linked once against
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

    ``departure=`` (or ``arrival=``) also takes several moments at once:
    a list adds a ``departure_time`` (``arrival_time``) column holding
    each row's moment as ``YYYY-MM-DD HH:MM:SS``, a mapping of labels to
    moments adds a ``slot`` column with the label as well, both placed
    after ``to_id``. The frame is the single-moment frames concatenated
    in slot order — every branch, ``street_policy`` included, applies
    per slot — and the slots must carry dates. A single moment keeps
    the columns as they are; ``travel_cost_table`` and the streaming
    ``to_parquet``/``output=`` forms behave the same — shards carry the
    slot columns and the manifest lists the slots.

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
    ``departure`` and the other timetable-only arguments are
    rejected; ``factors=`` and ``components=`` configure the factor.

    Parameters
    ----------
    network : TransportNetwork or StreetNetwork
        The network to compute on. A ``StreetNetwork`` takes the
        standalone street path and requires ``transport_mode``.
    origins : list of str, or GeoDataFrame (optional)
        Origin stop_ids (every stop when omitted), or points with an
        ``id`` column. A street matrix needs points. Polygon frames
        (``cafein.zones`` surfaces or your own) route from their
        centroids — the ``centroid_lat``/``centroid_lon`` columns when
        present, local-UTM centroids otherwise.
    destinations : list of str, or GeoDataFrame (optional)
        Destination stop_ids (every stop when omitted), or points; with
        point origins the destinations default to the origins.
    departure : datetime.datetime, str, or a sequence/mapping of them (optional)
        Departure at every origin — a datetime, or an ISO string like
        ``"2022-02-22 08:30"``; the service date is its date part.
        Give exactly one of ``departure`` and ``arrival``.
    arrival : datetime.datetime, str, or a sequence/mapping of them (optional)
        Arrival deadline at every destination, in the same forms, with
        ``arrival_time_window`` beside it — the windowed optimize
        modes are the only arrive-by cost modes (``optimize="time"``
        rides ``TravelTimeMatrix``). Each cell prices the pair's
        deadline profile: the exact journeys
        ``route_between_stops(arrival=, arrival_time_window=)``
        returns, replayed forward at their own departure instants
        under the final minute mark's arrival ceiling, so the cell is
        the lowest-objective candidate and a journey time-dominated
        at every deadline never wins. On a zone fare structure the
        cell is the cheapest candidate — the forward axis's exact
        whole-window zone search (and its implicit 120-minute cap)
        stays on the departure axis. One reverse run serves each
        destination, so ``chunk`` slices the destination axis;
        ``router="tbtr"``, ``candidates="pareto"``, and the street-leg
        policies do not combine with it — a ``CarParkPolicy`` serves
        it, windowless.
    arrival_time_window : float or datetime.timedelta (optional)
        The arrival window in minutes beside ``arrival=``: deadlines
        profile at every minute mark within it. Required with the
        arrive-by cost modes; rejected beside ``departure=``.
    max_rides : int (optional, default: 8)
        Maximum number of boarded vehicles per journey (rides, not
        transfers: 8 rides allow 7 transfers).
    optimize : str (optional, default: "time")
        What each cell's journey minimises. ``"time"`` (the default)
        reports the fastest journey. ``"emissions"`` and ``"fare"``
        report the lowest-emission or cheapest journey among the
        departure window's (departure, arrival, rides)-Pareto
        candidates — the same ride candidates ``journey_frontier``
        sees — optionally within the ``max_travel_time`` budget. A
        zero-ride floor (zero emissions, zero fare) joins the
        candidates: for stop pairs the origin itself, for point pairs
        the walking-only alternative, which wins any cell it qualifies
        for. Each objective qualifies candidates by its own key: NaN
        emissions drop a candidate under ``"emissions"``, an
        unpriceable fare under ``"fare"`` — pairs with no qualifying
        candidate are absent. On a **rule-based** fare structure,
        ``optimize="fare"`` returns the lowest-priced journey **among
        the candidates the time-and-ride search retains** — it does
        not search all feasible journeys by fare. A cheaper journey
        may be omitted when it arrives no earlier, uses more rides,
        boards the same trip at a different stop, or loses an
        equal-time canonical-path tie; fares are exact for each
        retained journey, but global fare optimality is not
        guaranteed. On a **zone** fare structure
        (``fares.zone_fare_structure``), that candidate fold is only
        the warm start: every cell is refined by the exact
        zone-ticket engine, and the reported journey is the cheapest
        of **all** journeys within ``max_travel_time`` — slower or
        more-ride journeys on cheaper tickets and multi-ticket chains
        included — with its distances, emissions, and geometry
        reconstructed from the winning chain. The exactness is what
        the 120-minute ``max_travel_time`` default buys its time
        limit for: proving no cheaper journey exists must otherwise
        rule out the whole service day. At metropolitan scale keep a
        bounded ``max_travel_time``; cells whose destinations carry
        no fare zone cannot price and cost the search most.
    departure_time_window : float or datetime.timedelta (optional)
        Departure window in minutes; required with
        ``optimize="emissions"`` and ``optimize="fare"``.
    max_travel_time : float or datetime.timedelta (optional)
        Maximum total travel time in minutes for the windowed optimize
        modes: only journeys at most this long qualify. Unset, the
        cleanest (cheapest) reachable journey wins with no time limit —
        except on a zone fare structure, where ``optimize="fare"``
        defaults to 120 minutes: an exact fare search without a
        time limit must rule out cheaper journeys across the whole
        service day, which is far slower and rarely what an analysis
        means. Pass ``max_travel_time`` explicitly to change the
        limit.
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
    traveler : TravelerProfile (optional)
        One traveler's constraint profile (``cafein.TravelerProfile``):
        its compiled exclusions union the ``exclude_*`` lists, and its
        walking knobs fill the unset walking arguments — a knob set on
        both the call and the profile is rejected.
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
    exposure : Exposure (optional)
        Report exposure per cell, with a ``cafein.Exposure`` built on
        the network being routed. Each reachable cell gains, per
        layer, the journey's ``{layer}_mean``, ``{layer}_max``,
        ``{layer}_coverage``, and one ``{layer}_minutes_above_{X}``
        per declared threshold — the ``exposure_totals()`` arithmetic
        per cell. Street matrices compute exactly from the traversed
        edges' true per-edge times (snap-connector time and any
        parking search outside the basis; no geometry is assembled on
        the default metres-only search). Transit matrices fold the
        time-optimal journey's walk-and-wait skeleton — access,
        egress, and transfer walks at their reconstructed provenance,
        boarding waits sampled at the stops, in-vehicle time excluded
        — with every distinct walk reconstructed once, never per
        cell. Single-departure time-optimal mode only: the windowed
        and Pareto modes, ``arrival=``, ``router='tbtr'`` (the engine
        resolves to RAPTOR), and ``street_policy`` refuse it.
    max_memory : str or int (optional)
        The memory budget this call plans against, as a percentage or a
        size with a binary suffix (``"80%"``, ``"6G"``, or bytes); the
        process default from
        ``cafein.set_max_memory`` (80 % of physical memory) when omitted.
        The fan-out width and, when streaming, the batch size follow from
        it. The budget plans, never caps: hard guarantees stay with the OS.
    chunk : (int, int) (optional)
        Compute only chunk ``k`` of ``n``: a deterministic contiguous
        block of the fan-out axis — the resolved origins with
        ``departure=``, the destinations with ``arrival=`` — so ``n``
        batch jobs cover all pairs disjointly and their shards
        concatenate.
    walking_speed_kmph, max_walking_time, snap_distance : float
        The street-search options for the walking access/egress, as in
        ``TransportNetwork.access_stops``: speed in km/h, walking time
        in minutes (or a timedelta), snap distance in meters. They
        bound the walking for point origins/destinations, and for stop
        origins/destinations only when a whole-day shortcut set routes
        them door-to-door; otherwise stop matrices ignore them. Only
        ``snap_distance`` applies to a ``StreetNetwork``, whose speeds
        come from the mode's profile.
    transport_mode : str (optional)
        The mode to route. Required for a ``StreetNetwork``, where it is
        one of ``"walk"``, ``"bicycle"``, ``"e_bike"``, ``"e_scooter"``,
        ``"car"`` (a car build; the shipped per-powertrain factors price
        emissions with ICE as the default class, and ``factors=`` rows
        still win).
    max_street_time : float or datetime.timedelta (optional)
        Cutoff in minutes for a ``StreetNetwork`` matrix (default: 120
        minutes, ``cafein.street_network.MAX_STREET_TIME`` seconds).
    intersection_delays, profile, delay_model : optional
        Car matrices only. By default car cells are free-flow,
        speed-limit travel times; ``intersection_delays=True`` applies
        the empirical intersection-delay model under ``profile=``
        (``"rush"``, ``"midday"`` — the default — or
        ``"day-average"``), with ``delay_model=`` merging partial
        overrides over the shipped values, as in
        ``StreetNetwork.travel_time``.
    parking : optional
        Car matrices only, off by default. The parking search ending
        each reachable trip, per destination cell: ``True`` → the
        shipped constant (300 s, 0 m), a number → seconds, a
        ``(seconds, metres)`` pair, or a polygon GeoDataFrame with a
        ``seconds`` column (optional ``metres``) resolved by
        point-in-polygon (largest seconds, ties by largest metres then
        lowest row; outside every polygon the shipped constant). The
        seconds join ``travel_time`` and the metres join the driven
        network distance and the emissions basis; geometry never shows
        the search loop, and ``max_street_time`` bounds the driving
        alone.
    occupancy, vehicle_class : optional
        Car matrices only. The shipped car factors are per
        vehicle-kilometre by powertrain (GEMMAT Table 4, Finland's
        energy mix; the default class is ``"ICE"``): ``vehicle_class=``
        selects ``"HEV"``, ``"PHEV"``, ``"BEV"``, ``"FCEV"``, or a user
        row's class, and ``occupancy=`` (at least 1, default 1) divides
        the per-vehicle emissions across the persons carried — the
        factors themselves are never rescaled.
    perspectives, costs, currency, cost_components : optional
        The monetary cost account of a street matrix — a separate
        account from fares, off by default. ``perspectives=`` selects
        ``"private"`` (the vehicle-operation bundle), ``"societal"``
        (the external cost, negatives are benefits), or both, adding a
        ``cost_<perspective>`` column per selection over the driven
        kilometres (parking metres included) from the shipped Gössling
        et al. (2019) Table 2 values — see ``cafein.costs``. ``costs=``
        layers a user table by (perspective, street_mode, component)
        key; ``cost_components=`` (one perspective only) restricts the
        derived total to named components and adds their columns; and
        ``currency=`` (default ``"EUR2017"``) is a declared label
        carried in a ``currency`` column, never a conversion. A mode
        without a matching row prices NaN, never zero.

    Notes
    -----
    ``street_policy=`` (a ``cafein.StreetLegPolicy``) opens the access
    and egress to the policy's street modes over the multimodal graph
    (build with ``street_modes=``); point origins and destinations only.
    With ``transfers={mode: budget}`` the matrix relaxes the merged
    mode-transfer set of ``TransportNetwork.compute_mode_transfers``
    (computed with exactly that binding), and the winning journey's
    rental transfers join the attribution: ride meters in
    ``street_distance_m``, ride grams in ``emissions``, the walking
    rest in ``walk_distance_m``.
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
    not price fares; ``optimize``, ``departure_time_window``,
    ``max_travel_time``, ``fares``, ``candidates``, ``router``, and the
    walking knobs are rejected beside a policy rather than silently
    ignored.

    ``street_policy=`` also takes a ``cafein.policy.CarParkPolicy`` (a network
    built with ``"car"`` in ``street_modes=``): per origin the access
    composes drive-to-facility, parking search, and the facility walk,
    with the ordinary walking access competing beside the car plane and
    winning ties. Egress is ordinary walking; transfers ride the
    installed transfer set, as on every query; the query's walking
    knobs stay active. The drive's metres join
    ``street_distance_m``, its grams (the policy's ``vehicle_class``
    GEMMAT row over its ``occupancy``) join ``emissions``, and a
    ``fee`` column carries the winning facility's parking fee in the
    ``cafein.costs`` currency (EUR2017), zero on rows the walk won. No
    facility column is surfaced — the winner is visible on
    ``DetailedItineraries``. Transit fares stay unpriced; the same
    knobs are rejected as under a street-leg policy except the walking
    knobs, and stop exclusions are rejected too. With ``arrival=`` the
    reverse election picks each cell's latest departure and winning
    facility, and the cell prices that chain at its own departure —
    the fee, metres, and grams belong to the elected facility.

    ``output_time_units=`` selects the ``travel_time`` unit:
    ``"minutes"`` (the default; whole minutes rounded to the nearest)
    or ``"seconds"`` (the engine's exact values). Anything computed
    downstream from the costs themselves — a decay-weighted measure,
    say — needs ``"seconds"``; the rounded minutes do not reproduce
    ``Accessibility``'s weights.
    """

    @property
    def _constructor(self):
        return pd.DataFrame

    @_log.timed_computer(
        "matrix.travel_costs",
        _log.matrix,
        "computing the travel cost matrix",
        "computed the travel cost matrix",
        street_identifier="matrix.streets",
        street_doing="computing the street cost matrix",
        street_done="computed the street cost matrix",
        is_street=lambda network: _is_street_network(network),
        details=_query_details,
    )
    @_planned_entry(cost=True)
    def __init__(
        self,
        network,
        origins=None,
        destinations=None,
        departure=None,
        *,
        arrival=None,
        arrival_time_window=None,
        max_rides=8,
        optimize="time",
        departure_time_window=None,
        max_travel_time=None,
        factors=None,
        components=None,
        fares=None,
        candidates="time",
        bucket=25.0,
        router="auto",
        exclude_routes=(),
        exclude_trips=(),
        traveler=None,
        exclude_stops=(),
        geometries=False,
        chunk=None,
        walking_speed_kmph=None,
        max_walking_time=None,
        snap_distance=None,
        transport_mode=None,
        max_street_time=None,
        street_policy=None,
        intersection_delays=False,
        profile=None,
        delay_model=None,
        parking=None,
        occupancy=None,
        vehicle_class=None,
        perspectives=None,
        costs=None,
        currency=None,
        cost_components=None,
        exposure=None,
        output_time_units="minutes",
        workers=None,
        max_memory=None,
    ):
        if workers is not None:
            workers = positive_int("workers", workers)
        memory_spec("max_memory", max_memory)
        if not _is_street_network(network):
            (
                exclude_routes,
                exclude_trips,
                exclude_stops,
                walking_speed_kmph,
                max_walking_time,
            ) = folded_constraints(
                traveler,
                network,
                exclude_routes,
                exclude_trips,
                exclude_stops,
                walking_speed_kmph,
                max_walking_time,
            )
        if not _is_street_network(network) and _is_point_frame(origins):
            street_policy, max_walking_time = folded_street_policy(
                traveler, network, street_policy, walking_speed_kmph, max_walking_time
            )
        origins, _origin_dtype = freeze_ids(origins)
        destinations, _destination_dtype = freeze_ids(destinations)
        if destinations is None and _is_point_frame(origins):
            # Omitted point destinations default to the origins, and so
            # does their dtype; omitted stop destinations mean every
            # stop (native string ids).
            _destination_dtype = _origin_dtype
        _id_dtypes = {"from_id": _origin_dtype, "to_id": _destination_dtype}
        origins = sequence_not_string("origins", origins)
        destinations = sequence_not_string("destinations", destinations)
        from cafein._units import duration_seconds, validated_output_time_units

        output_time_units = validated_output_time_units(output_time_units)
        if departure is not None and arrival is not None:
            raise ValueError("give exactly one of departure= or arrival=")
        arrive_by = arrival is not None
        from cafein._units import window_axis

        raw_window = window_axis(arrive_by, departure_time_window, arrival_time_window)
        if arrive_by:
            if not _is_street_network(network):
                if candidates != "time":
                    raise ValueError(
                        f"candidates={candidates!r} does not combine with "
                        "arrival=; multicriteria arrive-by is a later arc"
                    )
                if router == "tbtr":
                    raise ValueError(
                        "router='tbtr' does not serve arrival=; the reverse "
                        "search rides RAPTOR"
                    )
                if street_policy is not None:
                    from cafein.policy import CarParkPolicy

                    if not isinstance(street_policy, CarParkPolicy):
                        raise ValueError(
                            "street_policy= (a traveler's street bridge "
                            "included) does not combine with arrival= yet"
                        )
        if max_rides < 1:
            raise ValueError("max_rides must be at least 1")
        max_transfers = max_rides - 1
        window = duration_seconds(
            "arrival_time_window" if arrive_by else "departure_time_window",
            raw_window,
        )
        max_walking_time = duration_seconds("max_walking_time", max_walking_time)
        max_street_time = duration_seconds("max_street_time", max_street_time)
        max_snap_distance = snap_distance
        within = duration_seconds("max_travel_time", max_travel_time)
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
                intersection_delays=intersection_delays,
                profile=profile,
                delay_model=delay_model,
                parking=parking,
                occupancy=occupancy,
                vehicle_class=vehicle_class,
                perspectives=perspectives,
                costs=costs,
                currency=currency,
                cost_components=cost_components,
                exposure=exposure,
                transit_only={
                    "departure": departure,
                    "arrival": arrival,
                    "traveler": traveler,
                    "departure_time_window": window,
                    "max_travel_time": within,
                    "fares": fares,
                    "walking_speed_kmph": walking_speed_kmph,
                    "max_walking_time": max_walking_time,
                    "max_rides": None if max_rides == 8 else max_rides,
                    "optimize": None if optimize == "time" else optimize,
                    "candidates": None if candidates == "time" else candidates,
                    "bucket": None if bucket == 25.0 else bucket,
                    "router": None if router == "auto" else router,
                    "exclude_routes": id_sequence("exclude_routes", exclude_routes)
                    or None,
                    "exclude_trips": id_sequence("exclude_trips", exclude_trips)
                    or None,
                    "exclude_stops": id_sequence("exclude_stops", exclude_stops)
                    or None,
                    "street_policy": street_policy,
                },
                workers=workers,
                max_memory=max_memory,
            )
            super().__init__(
                restore_id_dtypes(
                    pd.DataFrame(_humanize_time_columns(data, output_time_units)),
                    _id_dtypes,
                )
            )
            return
        # The moment(s): a scalar keeps today's frame; a list or mapping
        # adds the slot columns, one block of rows per slot.
        slots, labeled, multi = _time_slots(departure, arrival, arrive_by)
        exclude_routes = id_sequence("exclude_routes", exclude_routes)
        exclude_trips = id_sequence("exclude_trips", exclude_trips)
        exclude_stops = id_sequence("exclude_stops", exclude_stops)
        if chunk is not None:
            chunk = tuple(chunk)
        components = component_selection(components)
        _resolved = None
        if street_policy is None:
            # Every slot reads one snapshot of the mutable inputs —
            # factors, fares, and the resolved endpoints with the chunk
            # applied to its axis — as the streaming form does.
            from cafein import emissions

            _reject_cost_street_args(
                transport_mode,
                max_street_time,
                intersection_delays,
                profile,
                delay_model,
                parking,
                occupancy,
                vehicle_class,
                perspectives,
                costs,
                currency,
                cost_components,
            )
            if exposure is not None:
                _validate_transit_exposure(
                    exposure,
                    network,
                    optimize=optimize,
                    window=window,
                    within=within,
                    candidates=candidates,
                    arrival=arrival,
                    arrive_by=arrive_by,
                    router=router,
                )
            _validate_cost_query(
                slots[0][1],
                slots[0][2],
                optimize,
                window,
                within,
                fares,
                router,
                arrive_by=arrive_by,
            )
            from_ids, to_ids, points, to_stops = _cost_endpoints(
                network, origins, destinations, None if arrive_by else chunk
            )
            if arrive_by and points is None:
                to_axis = to_stops if to_stops is not None else list(to_ids)
                if chunk is not None:
                    to_axis = to_axis[_chunk_slice(len(to_axis), chunk)]
                endpoints = ("stops", from_ids, to_axis)
            elif arrive_by:
                origin_points, destination_points = points
                if chunk is not None:
                    keep = _chunk_slice(len(to_ids), chunk)
                    to_ids = to_ids[keep]
                    destination_points = destination_points[keep]
                endpoints = (
                    "points",
                    from_ids,
                    origin_points,
                    to_ids,
                    destination_points,
                )
            elif points is None:
                endpoints = ("stops", from_ids, to_stops)
            else:
                origin_points, destination_points = points
                endpoints = (
                    "points",
                    from_ids,
                    origin_points,
                    to_ids,
                    destination_points,
                )
            _resolved = (
                emissions.trip_factors(network, factors, components),
                None if fares is None else fares._flat_tables(network),
                endpoints,
            )
        # The Rust slot loop: default-arm slots (time objective,
        # forward axis, no policy, no exposure) group by service date
        # and route in one core call per date over the frozen
        # endpoints; each slot then returns its share of the rows.
        _slot_tables = {}
        if (
            multi
            and street_policy is None
            and not arrive_by
            and optimize == "time"
            and exposure is None
            and _resolved is not None
        ):
            by_date = {}
            for at, (_, slot_date, slot_clock) in enumerate(slots):
                by_date.setdefault(slot_date, []).append((at, slot_clock))
            for slot_date, members in by_date.items():
                table, from_ids, to_ids = _cost_columns(
                    network,
                    origins,
                    destinations,
                    slot_date,
                    [clock for _, clock in members],
                    max_transfers=max_transfers,
                    optimize=optimize,
                    window=window,
                    within=within,
                    factors=factors,
                    components=components,
                    fares=fares,
                    candidates=candidates,
                    bucket=bucket,
                    geometries=geometries,
                    chunk=chunk,
                    router=router,
                    exclude_routes=exclude_routes,
                    exclude_trips=exclude_trips,
                    exclude_stops=exclude_stops,
                    arrive_by=arrive_by,
                    walking_speed_kmph=walking_speed_kmph,
                    max_walking_time=max_walking_time,
                    max_snap_distance=max_snap_distance,
                    _resolved=_resolved,
                    workers=workers,
                    max_memory=max_memory,
                )
                rows_per_slot = len(from_ids)
                origin_index = np.asarray(table["from"])
                slot_of = origin_index // max(rows_per_slot, 1)
                row_count = len(origin_index)
                for i, (at, _) in enumerate(members):
                    keep = slot_of == i
                    share = {}
                    for key, values in table.items():
                        array = np.asarray(values)
                        if (
                            array.ndim >= 1
                            and len(array) == row_count
                            and not key.startswith("unsnapped_")
                        ):
                            share[key] = array[keep]
                        else:
                            share[key] = values
                    share["from"] = origin_index[keep] - i * rows_per_slot
                    _slot_tables[at] = (share, from_ids, to_ids)
        frames = []
        for at, (label, date, clock) in enumerate(slots):
            data = _cost_matrix_data(
                network,
                origins,
                destinations,
                date,
                clock,
                geometries=geometries,
                transport_mode=transport_mode,
                max_street_time=max_street_time,
                intersection_delays=intersection_delays,
                profile=profile,
                delay_model=delay_model,
                parking=parking,
                occupancy=occupancy,
                vehicle_class=vehicle_class,
                perspectives=perspectives,
                costs=costs,
                currency=currency,
                cost_components=cost_components,
                street_policy=street_policy,
                exposure=exposure,
                fares=fares,
                max_transfers=max_transfers,
                optimize=optimize,
                window=window,
                within=within,
                factors=factors,
                components=components,
                candidates=candidates,
                bucket=bucket,
                router=router,
                exclude_routes=exclude_routes,
                exclude_trips=exclude_trips,
                exclude_stops=exclude_stops,
                chunk=chunk,
                arrive_by=arrive_by,
                walking_speed_kmph=walking_speed_kmph,
                max_walking_time=max_walking_time,
                max_snap_distance=max_snap_distance,
                workers=workers,
                max_memory=max_memory,
                arrival=arrival,
                _resolved=_resolved,
                _slot_table=_slot_tables.get(at),
            )
            frame = pd.DataFrame(_humanize_time_columns(data, output_time_units))
            if multi:
                _insert_slot_columns(frame, arrive_by, label, labeled, date, clock)
            frames.append(frame)
        frame = pd.concat(frames, ignore_index=True) if multi else frames[0]
        super().__init__(restore_id_dtypes(frame, _id_dtypes))

    def compare(self, other, *, columns=None, ratios=False):
        """``cafein.matrices.compare_matrices`` of this frame and
        ``other`` — the frame-in function is the durable entry point,
        since a reconstructed frame loses its class."""
        return compare_matrices(self, other, columns=columns, ratios=ratios)

    @classmethod
    def to_parquet(
        cls,
        network,
        origins=None,
        destinations=None,
        departure=None,
        *,
        arrival=None,
        arrival_time_window=None,
        output,
        batch_size=None,
        resume=False,
        max_rides=8,
        optimize="time",
        departure_time_window=None,
        max_travel_time=None,
        factors=None,
        components=None,
        fares=None,
        candidates="time",
        bucket=25.0,
        router="auto",
        exclude_routes=(),
        exclude_trips=(),
        traveler=None,
        exclude_stops=(),
        geometries=False,
        chunk=None,
        walking_speed_kmph=None,
        max_walking_time=None,
        snap_distance=None,
        transport_mode=None,
        max_street_time=None,
        street_policy=None,
        intersection_delays=False,
        profile=None,
        delay_model=None,
        parking=None,
        occupancy=None,
        vehicle_class=None,
        perspectives=None,
        costs=None,
        currency=None,
        cost_components=None,
        exposure=None,
        output_time_units="minutes",
        workers=None,
        max_memory=None,
    ):
        """The cost matrix streamed to Parquet — the constructor's
        semantics with `travel_cost_table`'s ``output=`` behavior.

        The fan-out axis — the origins with ``departure=``, the
        destinations with ``arrival=`` — is processed in
        ``batch_size`` slices (default 500)
        and each batch is written as it completes, so peak memory holds
        one batch — never the whole constructor result. ``output=``
        selects the form by suffix exactly as ``travel_cost_table``
        does and the return value is a :class:`cafein.StreamingResult`.
        ``from_id``/``to_id`` are dictionary-encoded over the shared id
        domains, and the dictionary values are strings whatever the
        input dtype — the streamed surfaces deliberately do not
        round-trip integer ids, keeping every shard's schema
        identical; a street matrix's geometry streams as plain WKB
        binary. A sequence or mapping of moments in
        ``departure=``/``arrival=`` streams every slot: each shard
        carries all slots with the constructor's slot columns (the
        ``slot`` label dictionary-encoded), and the manifest lists the
        slots; streamed rows order batch-major, not the constructor's
        slot-major — rows align by key, never by position.
        ``street_policy`` matrices do not stream yet and are
        rejected. ``resume=True`` continues a matching partial
        directory run exactly as ``travel_cost_table`` does.
        """
        _log.sync()
        if workers is not None:
            workers = positive_int("workers", workers)
        memory_spec("max_memory", max_memory)
        if not _is_street_network(network):
            (
                exclude_routes,
                exclude_trips,
                exclude_stops,
                walking_speed_kmph,
                max_walking_time,
            ) = folded_constraints(
                traveler,
                network,
                exclude_routes,
                exclude_trips,
                exclude_stops,
                walking_speed_kmph,
                max_walking_time,
            )
        if not _is_street_network(network) and _is_point_frame(origins):
            street_policy, max_walking_time = folded_street_policy(
                traveler, network, street_policy, walking_speed_kmph, max_walking_time
            )
        import pyarrow

        from cafein._units import (
            duration_seconds,
            validated_output_time_units,
        )

        output_time_units = validated_output_time_units(output_time_units)
        if departure is not None and arrival is not None:
            raise ValueError("give exactly one of departure= or arrival=")
        arrive_by = arrival is not None
        from cafein._units import window_axis

        raw_window = window_axis(arrive_by, departure_time_window, arrival_time_window)
        if arrive_by:
            if not _is_street_network(network):
                if candidates != "time":
                    raise ValueError(
                        f"candidates={candidates!r} does not combine with "
                        "arrival=; multicriteria arrive-by is a later arc"
                    )
                if router == "tbtr":
                    raise ValueError(
                        "router='tbtr' does not serve arrival=; the reverse "
                        "search rides RAPTOR"
                    )
        slots, labeled, multi = _time_slots(departure, arrival, arrive_by)
        if max_rides < 1:
            raise ValueError("max_rides must be at least 1")
        max_transfers = max_rides - 1
        window = duration_seconds(
            "arrival_time_window" if arrive_by else "departure_time_window",
            raw_window,
        )
        max_walking_time = duration_seconds("max_walking_time", max_walking_time)
        max_street_time = duration_seconds("max_street_time", max_street_time)
        max_snap_distance = snap_distance
        within = duration_seconds("max_travel_time", max_travel_time)
        size = _stream_size(batch_size, resume)
        if street_policy is not None:
            raise NotImplementedError(
                "street_policy matrices do not stream yet; compute the "
                "frame with the constructor instead"
            )
        if chunk is not None:
            chunk = tuple(int(part) for part in chunk)
        if _is_street_network(network):
            resolved = _street_cost_resolution(
                network,
                origins,
                destinations,
                transport_mode=transport_mode,
                max_street_time=max_street_time,
                max_snap_distance=max_snap_distance,
                chunk=chunk,
                transit_only={
                    "departure": departure,
                    "arrival": arrival,
                    "traveler": traveler,
                    "departure_time_window": window,
                    "max_travel_time": within,
                    "fares": fares,
                    "walking_speed_kmph": walking_speed_kmph,
                    "max_walking_time": max_walking_time,
                    "max_rides": None if max_rides == 8 else max_rides,
                    "optimize": None if optimize == "time" else optimize,
                    "candidates": None if candidates == "time" else candidates,
                    "bucket": None if bucket == 25.0 else bucket,
                    "router": None if router == "auto" else router,
                    "exclude_routes": id_sequence("exclude_routes", exclude_routes)
                    or None,
                    "exclude_trips": id_sequence("exclude_trips", exclude_trips)
                    or None,
                    "exclude_stops": id_sequence("exclude_stops", exclude_stops)
                    or None,
                    "street_policy": None,
                },
                factors=factors,
                components=components,
                intersection_delays=intersection_delays,
                profile=profile,
                delay_model=delay_model,
                parking=parking,
                occupancy=occupancy,
                vehicle_class=vehicle_class,
                perspectives=perspectives,
                costs=costs,
                currency=currency,
                cost_components=cost_components,
                exposure=exposure,
            )
            return _stream_street_cost(
                "TravelCostMatrix.to_parquet",
                network,
                resolved,
                geometries,
                chunk,
                output,
                size,
                pyarrow,
                resume=resume,
                output_time_units=output_time_units,
                workers=workers,
                max_memory=max_memory,
            )
        exposure_snapshot = None
        if exposure is not None:
            router = _validate_transit_exposure(
                exposure,
                network,
                optimize=optimize,
                window=window,
                within=within,
                candidates=candidates,
                arrival=arrival,
                arrive_by=arrive_by,
                router=router,
            )
            exposure_snapshot = _exposure_snapshot(exposure)
        _reject_cost_street_args(
            transport_mode,
            max_street_time,
            intersection_delays,
            profile,
            delay_model,
            parking,
            occupancy,
            vehicle_class,
            perspectives,
            costs,
            currency,
            cost_components,
        )
        return _stream_transit_cost(
            "TravelCostMatrix.to_parquet",
            network,
            origins,
            destinations,
            slots,
            labeled,
            multi,
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
            arrive_by=arrive_by,
            walking_speed_kmph=walking_speed_kmph,
            max_walking_time=max_walking_time,
            max_snap_distance=max_snap_distance,
            output=output,
            size=size,
            pyarrow=pyarrow,
            resume=resume,
            output_time_units=output_time_units,
            exposure=exposure_snapshot,
            workers=workers,
            max_memory=max_memory,
        )


class TravelTimeMatrix(pd.DataFrame):
    """Travel times per OD pair, long format — the lean r5py-style mode.

    A pandas DataFrame with one row per reachable OD pair: ``from_id``,
    ``to_id``, and ``travel_time`` — whole minutes rounded to the
    nearest by default, exact seconds with
    ``output_time_units="seconds"``. It is the long-format face
    of ``TransportNetwork.travel_time_matrix``: one run serves each
    origin (each destination with ``arrival=``), fanned out over all
    cores, and the reachable cells of the resulting wide matrix are
    unstacked into rows. Unreachable pairs
    are absent (never a sentinel), so the frame joins straight onto other
    tables. Where travel times only are needed, this is lighter than
    ``TravelCostMatrix``, which also aggregates transfers, distances, and
    emissions.

    The default minutes are rounded to the nearest whole minute:
    anything computed downstream from the costs themselves — a
    decay-weighted measure, say — needs ``output_time_units="seconds"``;
    the rounded minutes do not reproduce ``Accessibility``'s weights.

    With ``departure_time_window``, every minute mark within the
    window is profiled and the ``travel_time`` column is replaced by
    one ``travel_time_p<p>`` column per requested percentile (the
    median by default, or ``confidence`` for the symmetric interval
    plus the median), floating-point in the output units so an
    unreachable percentile reads as ``NaN``; a pair appears when at
    least one of its percentiles is reachable.

    ``departure=`` (or ``arrival=``) also takes several moments at once:
    a list adds a ``departure_time`` (``arrival_time``) column holding
    each row's moment as ``YYYY-MM-DD HH:MM:SS``, a mapping of labels to
    moments adds a ``slot`` column with the label as well, both placed
    after ``to_id``. The frame is the single-moment frames concatenated
    in slot order — windows, percentiles, and ``chunk=`` apply to every
    slot alike — and the slots must carry dates. A single moment keeps
    the columns as they are.

    Origins are either stop identifiers or a point GeoDataFrame with an
    ``id`` column; destinations apply to point origins only — stop
    origins always span every stop (the ``stops`` order). Integer ids
    round-trip: an axis supplied with a uniform integer dtype gets its
    ``from_id``/``to_id`` column back in that exact dtype (the engines
    speak strings internally), while string, mixed, and all-stops axes
    stay strings. Points are
    linked once against the street network (requires ``osm_pbf=`` at
    build time); points off the walking network are reported with a
    warning and stay unreachable. Polygon frames route from their
    centroids (``centroid_lat``/``centroid_lon`` columns when present —
    the ``cafein.zones`` protocol — otherwise local-UTM centroids).
    Slices and copies degrade to plain DataFrames.

    Given a ``StreetNetwork`` instead, the matrix is a standalone street
    computation: one bounded search per origin over the compiled profile
    of ``transport_mode`` (``"walk"``, ``"bicycle"``, ``"e_bike"``,
    ``"e_scooter"``, or ``"car"``), bounded by ``max_street_time``. Car
    cells are free-flow by default; ``intersection_delays=True`` with
    ``profile=`` and ``delay_model=`` applies the intersection-delay
    model, and ``parking=`` adds each destination's parking search
    seconds, as in ``StreetNetwork.travel_time``. It needs no
    timetable, so ``departure`` does not apply, and the
    arguments that only mean something to a timetable — ``max_rides``,
    ``router``, the departure-window percentiles, the transit exclusions,
    and the walking-speed options, whose speeds come from the profile —
    are rejected. Origins and destinations are point GeoDataFrames (or
    polygon frames routed by centroid, as on the transit side);
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
    departure : datetime.datetime, str, or a sequence/mapping of them (optional)
        Departure at every origin — a datetime, or an ISO string like
        ``"2022-02-22 08:30"``; the service date is its date part.
        Give exactly one of ``departure`` and ``arrival``.
    arrival : datetime.datetime, str, or a sequence/mapping of them (optional)
        Arrival deadline at every destination, in the same forms. Each
        row's ``travel_time`` is that pair's latest-departure journey
        arriving by the deadline (fewest rides, then earliest arrival,
        breaking ties) — the journey's own duration, identical to
        ``route_between_stops(arrival=)``. One reverse run serves each
        **destination**, so ``chunk`` slices the destination axis and
        chunked frames still concatenate to full coverage. The reverse
        rides the closure (a whole-day ULTRA set is never claimed);
        the departure-window parameters, ``router="tbtr"``, and
        the street-leg policies do not combine with it (a
        ``CarParkPolicy`` serves it, windowless), and a
        ``StreetNetwork`` matrix has no timetable axis at all.
    max_rides : int (optional, default: 8)
        Maximum number of boarded vehicles per journey (rides, not
        transfers: 8 rides allow 7 transfers).
    departure_time_window : float or datetime.timedelta (optional)
        Departure window in minutes; enables percentile columns.
    percentiles : list of float (optional)
        Percentiles in ``[0, 100]`` over the window's departures;
        requires `departure_time_window`, defaults to ``[50]``.
    confidence : float (optional)
        A level in ``(0, 1)`` mapped to the symmetric percentile
        interval plus the median; requires `departure_time_window` and
        excludes `percentiles`.
    max_memory : str or int (optional)
        The memory budget this call plans against, as a percentage or a
        size with a binary suffix (``"80%"``, ``"6G"``, or bytes); the
        process default from
        ``cafein.set_max_memory`` (80 % of physical memory) when omitted.
        The fan-out width and, when streaming, the batch size follow from
        it. The budget plans, never caps: hard guarantees stay with the OS.
    chunk : (int, int) (optional)
        Compute only chunk ``k`` of ``n``: a deterministic contiguous
        block of the fan-out axis — the resolved origins with
        ``departure=``, the destinations with ``arrival=`` — so ``n``
        batch jobs cover all pairs disjointly and their rows
        concatenate.
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
    walking_speed_kmph, max_walking_time, snap_distance : float
        The street-search options for the walking access/egress, as in
        ``TransportNetwork.access_stops``: speed in km/h, walking time
        in minutes (or a timedelta), snap distance in meters. They
        bound the walking for point origins/destinations, and for stop
        origins/destinations only when a whole-day shortcut set routes
        them door-to-door; otherwise stop matrices ignore them. Only
        ``snap_distance`` applies to a ``StreetNetwork``, whose speeds
        come from the mode's profile.
    transport_mode : str (optional)
        The mode to route. Required for a ``StreetNetwork``, where it is
        one of ``"walk"``, ``"bicycle"``, ``"e_bike"``, ``"e_scooter"``,
        ``"car"``; for a ``TransportNetwork`` only ``"public_transport"``
        (the default meaning) applies.
    max_street_time : float or datetime.timedelta (optional)
        Cutoff in minutes for a ``StreetNetwork`` matrix, beyond which
        a destination counts as unreachable (default: 120 minutes,
        ``cafein.street_network.MAX_STREET_TIME`` seconds).

    Notes
    -----
    ``street_policy=`` (a ``cafein.StreetLegPolicy``) opens the access
    and egress to the policy's street modes over the carried multimodal
    graph: point-set origins and destinations only, exclusions honoured,
    a walking-only policy identical to the legacy walking matrix. With
    ``transfers={mode: budget}`` the matrix relaxes the merged
    mode-transfer set of ``TransportNetwork.compute_mode_transfers``,
    which must be computed with exactly that binding. A carried vehicle
    (``take_aboard=True``, with the carriage set of
    ``TransportNetwork.compute_carriage_transfers`` under an own
    ``transfers=`` grant) runs the possession-state search per origin:
    the bicycle rides along where trips permit, parks at the policy's
    facilities, and every cell is the cross-plane earliest arrival —
    exclusions are rejected beside a carried vehicle. It conflicts with
    the walking knobs, ``router``, and the departure-window parameters,
    which are rejected rather than silently ignored.

    ``street_policy=`` also takes a ``cafein.policy.CarParkPolicy`` (a network
    built with ``"car"`` in ``street_modes=``): per origin the access
    composes drive-to-facility, parking search, and the facility walk,
    with the ordinary walking access competing beside the car plane and
    winning ties; egress is ordinary walking, transfers ride the
    installed transfer set as on every query, and the direct walking
    alternative folds in per cell. Unlike the street-leg policies, the
    query's walking knobs stay active — they time every ordinary walk
    while the policy's budgets bound the car chain. Departure windows,
    percentiles, ``router=`` overrides, and stop exclusions are
    rejected by name; an origin that cannot reach any facility by car
    has its cells omitted with a warning, never a silent walking-only
    row. With ``arrival=`` the same composed tables ride the reverse
    engine: per cell the latest departure arriving by the deadline,
    the walk placed to arrive exactly at it, and chunking slicing the
    destination axis as on every arrive-by matrix.
    """

    @property
    def _constructor(self):
        return pd.DataFrame

    @_log.timed_computer(
        "matrix.travel_times",
        _log.matrix,
        "computing the travel time matrix",
        "computed the travel time matrix",
        street_identifier="matrix.streets",
        street_doing="computing the street time matrix",
        street_done="computed the street time matrix",
        is_street=lambda network: _is_street_network(network),
        details=_query_details,
    )
    @_planned_entry(cost=False)
    def __init__(
        self,
        network,
        origins=None,
        destinations=None,
        departure=None,
        *,
        arrival=None,
        arrival_time_window=None,
        max_rides=8,
        departure_time_window=None,
        percentiles=None,
        confidence=None,
        chunk=None,
        router="auto",
        exclude_routes=(),
        exclude_trips=(),
        traveler=None,
        exclude_stops=(),
        walking_speed_kmph=None,
        max_walking_time=None,
        snap_distance=None,
        transport_mode=None,
        max_street_time=None,
        street_policy=None,
        intersection_delays=False,
        profile=None,
        delay_model=None,
        parking=None,
        output_time_units="minutes",
        workers=None,
        max_memory=None,
    ):
        if workers is not None:
            workers = positive_int("workers", workers)
        memory_spec("max_memory", max_memory)
        if not _is_street_network(network):
            (
                exclude_routes,
                exclude_trips,
                exclude_stops,
                walking_speed_kmph,
                max_walking_time,
            ) = folded_constraints(
                traveler,
                network,
                exclude_routes,
                exclude_trips,
                exclude_stops,
                walking_speed_kmph,
                max_walking_time,
            )
        if not _is_street_network(network) and _is_point_frame(origins):
            street_policy, max_walking_time = folded_street_policy(
                traveler, network, street_policy, walking_speed_kmph, max_walking_time
            )
        origins, _origin_dtype = freeze_ids(origins)
        destinations, _destination_dtype = freeze_ids(destinations)
        if destinations is None and _is_point_frame(origins):
            # Omitted point destinations default to the origins, and so
            # does their dtype; omitted stop destinations mean every
            # stop (native string ids).
            _destination_dtype = _origin_dtype
        _id_dtypes = {"from_id": _origin_dtype, "to_id": _destination_dtype}
        origins = sequence_not_string("origins", origins)
        destinations = sequence_not_string("destinations", destinations)
        from cafein._units import duration_seconds, validated_output_time_units

        output_time_units = validated_output_time_units(output_time_units)
        if departure is not None and arrival is not None:
            raise ValueError("give exactly one of departure= or arrival=")
        arrive_by = arrival is not None
        from cafein._units import window_axis

        raw_window = window_axis(arrive_by, departure_time_window, arrival_time_window)
        if arrive_by:
            if router == "tbtr":
                raise ValueError(
                    "router='tbtr' does not serve arrival=; the reverse "
                    "search rides RAPTOR"
                )
            if street_policy is not None:
                from cafein.policy import CarParkPolicy

                if not isinstance(street_policy, CarParkPolicy):
                    raise ValueError(
                        "street_policy= (a traveler's street bridge included) "
                        "does not combine with arrival= yet"
                    )
        if max_rides < 1:
            raise ValueError("max_rides must be at least 1")
        max_transfers = max_rides - 1
        window = duration_seconds(
            "arrival_time_window" if arrive_by else "departure_time_window",
            raw_window,
        )
        max_walking_time = duration_seconds("max_walking_time", max_walking_time)
        max_street_time = duration_seconds("max_street_time", max_street_time)
        max_snap_distance = snap_distance
        if _is_street_network(network):
            data = _street_time_columns(
                network,
                origins,
                destinations,
                transport_mode=transport_mode,
                max_street_time=max_street_time,
                max_snap_distance=max_snap_distance,
                chunk=chunk,
                intersection_delays=intersection_delays,
                profile=profile,
                delay_model=delay_model,
                parking=parking,
                transit_only={
                    "departure": departure,
                    "arrival": arrival,
                    "traveler": traveler,
                    "departure_time_window": window,
                    "percentiles": percentiles,
                    "confidence": confidence,
                    "walking_speed_kmph": walking_speed_kmph,
                    "max_walking_time": max_walking_time,
                    "max_rides": None if max_rides == 8 else max_rides,
                    "router": None if router == "auto" else router,
                    "exclude_routes": id_sequence("exclude_routes", exclude_routes)
                    or None,
                    "exclude_trips": id_sequence("exclude_trips", exclude_trips)
                    or None,
                    "exclude_stops": id_sequence("exclude_stops", exclude_stops)
                    or None,
                    "street_policy": street_policy,
                },
                workers=workers,
                max_memory=max_memory,
            )
            super().__init__(
                restore_id_dtypes(
                    pd.DataFrame(_humanize_time_columns(data, output_time_units)),
                    _id_dtypes,
                )
            )
            return
        _reject_time_street_args(
            transport_mode,
            max_street_time,
            intersection_delays,
            profile,
            delay_model,
            parking,
        )
        # The moment(s): a scalar keeps today's frame; a list or mapping
        # adds the slot columns, one block of rows per slot.
        slots, labeled, multi = _time_slots(departure, arrival, arrive_by)
        # Every slot sees the same query: one-shot iterables are drained
        # by the first slot otherwise.
        if percentiles is not None:
            percentiles = list(percentiles)
        if chunk is not None:
            chunk = tuple(chunk)
        exclude_routes = id_sequence("exclude_routes", exclude_routes)
        exclude_trips = id_sequence("exclude_trips", exclude_trips)
        exclude_stops = id_sequence("exclude_stops", exclude_stops)
        # The Rust slot loop: stop-origin forward slots group by
        # service date and route in one core call per date — services,
        # the engine's transfer set, the worker pool, and the progress
        # ticker resolve once, and each slot decodes its plane.
        _slot_planes = {}
        if (
            multi
            and street_policy is None
            and not arrive_by
            and window is None
            and percentiles is None
            and confidence is None
            and not _is_street_network(network)
            and (destinations is None or _is_point_frame(origins))
        ):
            by_date = {}
            for at, (_, slot_date, slot_clock) in enumerate(slots):
                by_date.setdefault(slot_date, []).append((at, slot_clock))
            for slot_date, members in by_date.items():
                matrix, from_ids, to_ids, _ = network._time_matrix_with_ids(
                    origins,
                    slot_date,
                    [clock for _, clock in members],
                    max_transfers,
                    destinations=destinations,
                    window=None,
                    percentiles=None,
                    confidence=None,
                    chunk=chunk,
                    router=router,
                    exclude_routes=exclude_routes,
                    exclude_trips=exclude_trips,
                    exclude_stops=exclude_stops,
                    walking_speed_kmph=walking_speed_kmph,
                    max_walking_time=max_walking_time,
                    max_snap_distance=max_snap_distance,
                    workers=workers,
                    max_memory=max_memory,
                )
                rows_per_slot = len(from_ids)
                for i, (at, _) in enumerate(members):
                    plane = matrix[i * rows_per_slot : (i + 1) * rows_per_slot]
                    _slot_planes[at] = (plane, from_ids, to_ids, None)
        frames = []
        for at, (label, date, clock) in enumerate(slots):
            data = _time_matrix_data(
                network,
                origins,
                destinations,
                date,
                clock,
                _matrix=_slot_planes.get(at),
                arrive_by=arrive_by,
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
                street_policy=street_policy,
                max_transfers=max_transfers,
                workers=workers,
                max_memory=max_memory,
            )
            frame = pd.DataFrame(_humanize_time_columns(data, output_time_units))
            if multi:
                _insert_slot_columns(frame, arrive_by, label, labeled, date, clock)
            frames.append(frame)
        frame = pd.concat(frames, ignore_index=True) if multi else frames[0]
        super().__init__(restore_id_dtypes(frame, _id_dtypes))

    def compare(self, other, *, columns=None, ratios=False):
        """``cafein.matrices.compare_matrices`` of this frame and
        ``other`` — the frame-in function is the durable entry point,
        since a reconstructed frame loses its class."""
        return compare_matrices(self, other, columns=columns, ratios=ratios)

    @classmethod
    def to_parquet(
        cls,
        network,
        origins=None,
        destinations=None,
        departure=None,
        *,
        arrival=None,
        arrival_time_window=None,
        output,
        batch_size=None,
        resume=False,
        max_rides=8,
        departure_time_window=None,
        percentiles=None,
        confidence=None,
        chunk=None,
        router="auto",
        exclude_routes=(),
        exclude_trips=(),
        traveler=None,
        exclude_stops=(),
        walking_speed_kmph=None,
        max_walking_time=None,
        snap_distance=None,
        transport_mode=None,
        max_street_time=None,
        street_policy=None,
        intersection_delays=False,
        profile=None,
        delay_model=None,
        parking=None,
        output_time_units="minutes",
        workers=None,
        max_memory=None,
    ):
        """The travel-time matrix streamed to Parquet — the
        constructor's semantics with ``travel_cost_table``'s
        ``output=`` behavior.

        Origins are processed in ``batch_size`` slices (default 500)
        and each batch is written as it completes, so peak memory holds
        one batch — never the whole constructor result. With
        ``arrival=`` the batches slice the **destination axis** instead
        (the fan-out axis, exactly as ``chunk=``), single-deadline and
        ``arrival_time_window=`` forms alike, so shards order
        destination-major; rows align by key, never by position. The
        resume manifest fingerprints the complete time query — axis,
        moment, window, and the resolved percentiles — so a resume
        differing in any of them refuses instead of mixing shards.
        ``output=`` selects the form by suffix exactly as
        ``travel_cost_table`` does and the return value is a
        :class:`cafein.StreamingResult`. ``from_id``/``to_id`` are
        dictionary-encoded over the shared id domains — the dictionary
        values are strings whatever the input dtype, so integer ids
        deliberately do not round-trip on the streamed surface; a
        windowed
        matrix streams its percentile columns. A sequence or mapping
        of moments in ``departure=``/``arrival=`` streams every slot:
        each shard carries all slots with the constructor's slot
        columns (the ``slot`` label dictionary-encoded), and the
        manifest lists the slots; streamed rows order batch-major, not
        the constructor's slot-major — rows align by key, never by
        position. ``street_policy``
        matrices do not stream yet and are rejected. ``resume=True``
        continues a matching partial directory run exactly as
        ``travel_cost_table`` does.
        """
        _log.sync()
        if workers is not None:
            workers = positive_int("workers", workers)
        memory_spec("max_memory", max_memory)
        if not _is_street_network(network):
            (
                exclude_routes,
                exclude_trips,
                exclude_stops,
                walking_speed_kmph,
                max_walking_time,
            ) = folded_constraints(
                traveler,
                network,
                exclude_routes,
                exclude_trips,
                exclude_stops,
                walking_speed_kmph,
                max_walking_time,
            )
        if not _is_street_network(network) and _is_point_frame(origins):
            street_policy, max_walking_time = folded_street_policy(
                traveler, network, street_policy, walking_speed_kmph, max_walking_time
            )
        import pyarrow

        from cafein._units import (
            duration_seconds,
            validated_output_time_units,
        )

        output_time_units = validated_output_time_units(output_time_units)
        if departure is not None and arrival is not None:
            raise ValueError("give exactly one of departure= or arrival=")
        arrive_by = arrival is not None
        from cafein._units import window_axis

        raw_window = window_axis(arrive_by, departure_time_window, arrival_time_window)
        if arrive_by:
            if _is_street_network(network):
                raise ValueError(
                    "arrival applies to transit; a StreetNetwork matrix "
                    "has no timetable axis"
                )
            if router == "tbtr":
                raise ValueError(
                    "router='tbtr' does not serve arrival=; the reverse "
                    "search rides RAPTOR"
                )
            if street_policy is not None:
                from cafein.policy import CarParkPolicy

                if not isinstance(street_policy, CarParkPolicy):
                    raise ValueError(
                        "street_policy= (a traveler's street bridge included) "
                        "does not combine with arrival= yet"
                    )
        slots, labeled, multi = _time_slots(departure, arrival, arrive_by)
        if max_rides < 1:
            raise ValueError("max_rides must be at least 1")
        max_transfers = max_rides - 1
        window = duration_seconds(
            "arrival_time_window" if arrive_by else "departure_time_window",
            raw_window,
        )
        max_walking_time = duration_seconds("max_walking_time", max_walking_time)
        max_street_time = duration_seconds("max_street_time", max_street_time)
        max_snap_distance = snap_distance
        size = _stream_size(batch_size, resume)
        if street_policy is not None:
            raise NotImplementedError(
                "street_policy matrices do not stream yet; compute the "
                "frame with the constructor instead"
            )
        if chunk is not None:
            chunk = tuple(int(part) for part in chunk)
        if _is_street_network(network):
            resolved = _street_time_resolution(
                origins,
                destinations,
                transport_mode=transport_mode,
                max_street_time=max_street_time,
                max_snap_distance=max_snap_distance,
                chunk=chunk,
                transit_only={
                    "departure": departure,
                    "traveler": traveler,
                    "departure_time_window": window,
                    "percentiles": percentiles,
                    "confidence": confidence,
                    "walking_speed_kmph": walking_speed_kmph,
                    "max_walking_time": max_walking_time,
                    "max_rides": None if max_rides == 8 else max_rides,
                    "router": None if router == "auto" else router,
                    "exclude_routes": id_sequence("exclude_routes", exclude_routes)
                    or None,
                    "exclude_trips": id_sequence("exclude_trips", exclude_trips)
                    or None,
                    "exclude_stops": id_sequence("exclude_stops", exclude_stops)
                    or None,
                    "street_policy": None,
                },
                intersection_delays=intersection_delays,
                profile=profile,
                delay_model=delay_model,
                parking=parking,
            )
            return _stream_street_time(
                "TravelTimeMatrix.to_parquet",
                network,
                resolved,
                chunk,
                output,
                size,
                pyarrow,
                resume=resume,
                output_time_units=output_time_units,
                workers=workers,
                max_memory=max_memory,
            )
        _reject_time_street_args(
            transport_mode,
            max_street_time,
            intersection_delays,
            profile,
            delay_model,
            parking,
        )
        return _stream_transit_time(
            "TravelTimeMatrix.to_parquet",
            network,
            origins,
            destinations,
            slots,
            labeled,
            multi,
            max_transfers=max_transfers,
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
            output=output,
            size=size,
            pyarrow=pyarrow,
            resume=resume,
            output_time_units=output_time_units,
            arrive_by=arrive_by,
            workers=workers,
            max_memory=max_memory,
        )


def _time_slots(departure, arrival, arrive_by):
    """The query's moments as ``(slots, labeled, multi)``: the parsed
    ``(label, date, clock)`` triples, whether a mapping labeled them,
    and whether a list or mapping was given (a single moment keeps the
    frame's columns as they are)."""
    from cafein._units import moments

    if arrive_by:
        slots, labeled = moments("arrival", arrival)
        return (
            slots,
            labeled,
            isinstance(arrival, (list, tuple, collections.abc.Mapping)),
        )
    if departure is None:
        return [(None, None, None)], False, False
    slots, labeled = moments("departure", departure)
    return slots, labeled, isinstance(departure, (list, tuple, collections.abc.Mapping))


def _insert_slot_arrow_columns(table, arrive_by, label, labeled, date, clock, pa):
    """The Arrow twin of ``_insert_slot_columns``."""
    at = table.schema.get_field_index("to_id") + 1
    moment = pa.array([f"{date} {clock}"] * table.num_rows, type=pa.string())
    table = table.add_column(
        at, "arrival_time" if arrive_by else "departure_time", moment
    )
    if labeled:
        table = table.add_column(
            at, "slot", pa.array([label] * table.num_rows, type=pa.string())
        )
    return table


def _insert_slot_columns(frame, arrive_by, label, labeled, date, clock):
    """Add a slot's columns after ``to_id``: ``slot`` (the label, for
    mapped slots) and ``departure_time``/``arrival_time`` (the moment
    as ``YYYY-MM-DD HH:MM:SS``)."""
    at = list(frame.columns).index("to_id") + 1
    frame.insert(
        at, "arrival_time" if arrive_by else "departure_time", f"{date} {clock}"
    )
    if labeled:
        frame.insert(at, "slot", label)


def _humanize_time_columns(data, output_time_units):
    """``travel_time_s`` (and percentile columns) → ``travel_time`` in
    the requested output units, keeping column order."""
    from cafein._units import travel_time_output

    out = {}
    for key, value in data.items():
        if key == "travel_time_s":
            out["travel_time"] = travel_time_output(value, output_time_units)
        elif key.startswith("travel_time_p"):
            name = key[: -len("_s")] if key.endswith("_s") else key
            out[name] = travel_time_output(value, output_time_units)
        else:
            out[key] = value
    return out


def _is_street_network(network):
    """Whether `network` is a standalone street network rather than a transit one."""
    from cafein.street_network import StreetNetwork

    return isinstance(network, StreetNetwork)


def _reject_cost_street_args(
    transport_mode,
    max_street_time,
    intersection_delays,
    profile,
    delay_model,
    parking,
    occupancy,
    vehicle_class,
    perspectives,
    costs,
    currency,
    cost_components,
):
    """The cost matrix's street-only arguments, rejected on a transit
    network — shared by the constructor and ``to_parquet``."""
    if transport_mode is not None and transport_mode != "public_transport":
        raise ValueError(
            f"transport_mode={transport_mode!r} is a street mode; pass a "
            "StreetNetwork to route on it"
        )
    if max_street_time is not None:
        raise ValueError("max_street_time applies to a StreetNetwork matrix")
    if (
        intersection_delays
        or profile is not None
        or delay_model is not None
        or parking is not None
        or occupancy is not None
        or vehicle_class is not None
    ):
        raise ValueError(
            "intersection_delays, profile, delay_model, parking, "
            "occupancy, and vehicle_class apply to a StreetNetwork "
            "car matrix"
        )
    if (
        perspectives is not None
        or costs is not None
        or currency is not None
        or cost_components is not None
    ):
        raise ValueError(
            "perspectives, costs, currency, and cost_components price "
            "street kilometres and apply to a StreetNetwork matrix "
            "(transit perspective costs are not supported)"
        )


def _reject_time_street_args(
    transport_mode,
    max_street_time,
    intersection_delays,
    profile,
    delay_model,
    parking,
):
    """The time matrix's street-only arguments, rejected on a transit
    network — shared by the constructor and ``to_parquet``."""
    if transport_mode is not None and transport_mode != "public_transport":
        raise ValueError(
            f"transport_mode={transport_mode!r} is a street mode; pass a "
            "StreetNetwork to route on it (street legs within a "
            "public-transport journey are a separate feature)"
        )
    if max_street_time is not None:
        raise ValueError("max_street_time applies to a StreetNetwork matrix")
    if (
        intersection_delays
        or profile is not None
        or delay_model is not None
        or parking is not None
    ):
        raise ValueError(
            "intersection_delays, profile, delay_model, and parking "
            "apply to a StreetNetwork car matrix"
        )


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
            "a street matrix needs origins as a GeoDataFrame of points (or "
            "polygons routed by centroid); stop ids belong to a "
            "TransportNetwork"
        )
    from_ids, origin_points = _point_list(origins, "origins")
    if destinations is None:
        to_ids, destination_points = list(from_ids), list(origin_points)
    elif not _is_point_frame(destinations):
        raise TypeError(
            "a street matrix needs destinations as a GeoDataFrame of points "
            "(or polygons routed by centroid)"
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
            else non_negative_finite("max_street_time", max_street_time)
        ),
        # A negative or non-finite snap silently unroutes every point
        # Rust-side — an empty matrix, not an error — so it refuses
        # here by its public name.
        float(
            streets.MAX_SNAP_DISTANCE
            if max_snap_distance is None
            else non_negative_finite("snap_distance", max_snap_distance)
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
    intersection_delays=False,
    profile=None,
    delay_model=None,
    parking=None,
    occupancy=None,
    vehicle_class=None,
    perspectives=None,
    costs=None,
    currency=None,
    cost_components=None,
    exposure=None,
    workers=None,
    max_memory=None,
):
    """The reachable cells of a street cost matrix, in long format."""
    components = component_selection(components)
    from cafein._cafein import STREET_DISTANCE_PROVENANCE

    resolved = _street_cost_resolution(
        network,
        origins,
        destinations,
        transport_mode=transport_mode,
        max_street_time=max_street_time,
        max_snap_distance=max_snap_distance,
        chunk=chunk,
        transit_only=transit_only,
        factors=factors,
        components=components,
        intersection_delays=intersection_delays,
        profile=profile,
        delay_model=delay_model,
        parking=parking,
        occupancy=occupancy,
        vehicle_class=vehicle_class,
        perspectives=perspectives,
        costs=costs,
        currency=currency,
        cost_components=cost_components,
        exposure=exposure,
    )
    query = resolved["query"]
    from_index, to_index, numeric, wkb = _street_cost_cells(
        network,
        query,
        geometries=geometries,
        resolved=resolved,
        workers=workers,
        max_memory=max_memory,
    )
    from_ids = np.asarray(query.from_ids, dtype=object)
    to_ids = np.asarray(query.to_ids, dtype=object)
    data = {"from_id": from_ids[from_index], "to_id": to_ids[to_index]}
    data.update(numeric)
    count = len(from_index)
    data["distance_provenance"] = np.full(
        count, STREET_DISTANCE_PROVENANCE, dtype=object
    )
    if resolved["account"] is not None:
        data["currency"] = np.full(count, resolved["account"][2], dtype=object)
    if geometries:
        data["geometry"] = shapely.from_wkb(np.array(wkb, dtype=object))
    # The frame orders provenance before the cost block, then the
    # monetary block with its currency, then the exposure block, then
    # geometry — the exact order the streamed shards emit.
    order = ["from_id", "to_id", "travel_time_s", "distance_m"]
    order += ["network_distance_m", "connector_distance_m", "distance_provenance"]
    order += ["emissions"]
    order += [name for name in data if name.startswith("cost_")]
    if "currency" in data:
        order.append("currency")
    order += [name for name in data if name not in order and name != "geometry"]
    if "geometry" in data:
        order.append("geometry")
    return {name: data[name] for name in order}


def _street_cost_resolution(
    network,
    origins,
    destinations,
    *,
    transport_mode,
    max_street_time,
    max_snap_distance,
    chunk,
    transit_only,
    factors,
    components,
    intersection_delays,
    profile,
    delay_model,
    parking,
    occupancy,
    vehicle_class,
    perspectives,
    costs,
    currency,
    cost_components,
    exposure=None,
):
    """Every result-affecting street-cost input resolved exactly once —
    validation first, then the frozen snapshot the cells (and the
    streaming form's batches) compute from."""
    from cafein import _parking, costs as _costs, emissions
    from cafein.street_network import _resolved_delays

    if exposure is not None:
        exposure._check_network(network)
    resolved_parking = _parking.resolve(parking, transport_mode)
    occupancy, vehicle_class = emissions._car_query_options(
        transport_mode, occupancy, vehicle_class
    )
    query = _street_query(
        origins,
        destinations,
        transport_mode=transport_mode,
        max_street_time=max_street_time,
        max_snap_distance=max_snap_distance,
        chunk=chunk,
        transit_only=transit_only,
    )
    # Resolved after the argument validation but before the routing call:
    # the factor and cost account the query started with are the ones
    # applied, whatever happens to a mutable table while the search holds
    # no GIL — and a bad table fails before the search pays for it.
    factor = emissions.street_factor(
        transport_mode, factors, components, vehicle_class=vehicle_class
    )
    account = _costs.resolve_query(
        transport_mode, perspectives, costs, currency, cost_components
    )
    parking_costs = (
        None
        if resolved_parking is None
        else _parking.destination_costs(resolved_parking, query.destination_points)
    )
    return {
        "transport_mode": transport_mode,
        "query": query,
        "car_model": _resolved_delays(
            transport_mode, intersection_delays, profile, delay_model
        ),
        "parking_costs": parking_costs,
        "occupancy": occupancy,
        "factor": factor,
        "account": account,
        # A frozen copy: the streamed batches and the manifest
        # fingerprint read one state, whatever happens to the caller's
        # Exposure meanwhile — the fare/policy frozen-input pattern.
        "exposure": None if exposure is None else _exposure_snapshot(exposure),
    }


def _street_cost_cells(
    network, query, *, geometries, resolved, workers=None, max_memory=None
):
    """One street-cost batch: origin/destination indices, the numeric
    columns, and the WKB geometries (``None`` without ``geometries``)."""
    transport_mode = resolved["transport_mode"]
    exposure = resolved["exposure"]
    table = network._core.cost_matrix(
        query.origin_points,
        query.destination_points,
        transport_mode,
        query.max_seconds,
        query.max_snap_distance,
        bool(geometries),
        car_model=resolved["car_model"],
        street_edges=exposure is not None,
        workers=_memory.width_or(workers),
    )
    _warn_unsnapped(
        table,
        query.from_ids,
        query.to_ids,
        network=f"the streets the {transport_mode} profile can use",
    )
    network_distance = table["network_distance"]
    connector_distance = table["connector_distance"]
    travel_time_s = table["travel_time_s"]
    if resolved["parking_costs"] is not None:
        # The parking search ends each reachable trip: its seconds join the
        # travel time and its metres the driven network distance (and with
        # it the emissions basis); the geometry never shows the search loop.
        seconds, metres = resolved["parking_costs"]
        to_index = np.asarray(table["to"])
        travel_time_s = travel_time_s + np.rint(seconds[to_index]).astype(np.int64)
        network_distance = network_distance + metres[to_index]
    numeric = {
        "travel_time_s": travel_time_s,
        # Reported alongside its parts, not instead of them: the two are
        # measured differently (stored edge lengths versus straight connectors).
        "distance_m": network_distance + connector_distance,
        "network_distance_m": network_distance,
        "connector_distance_m": connector_distance,
        # One mode per matrix, so the factor resolved once; the
        # connectors are the walk to the vehicle, not vehicle-kilometres,
        # so network metres only. The car's per-vehicle factor divides
        # across the persons carried.
        "emissions": network_distance
        / 1000.0
        * resolved["factor"]
        / resolved["occupancy"],
    }
    if resolved["account"] is not None:
        # Costs ride the same driven kilometres as the emissions —
        # parking-search metres included, connectors excluded — and a
        # missing row's NaN propagates, never a silent zero.
        totals, breakdown, _ = resolved["account"]
        kilometres = network_distance / 1000.0
        for perspective, per_km in totals.items():
            numeric[f"cost_{perspective}"] = kilometres * per_km
        for (perspective, component), per_km in breakdown.items():
            numeric[f"cost_{perspective}_{component}"] = kilometres * per_km
    if exposure is not None:
        from cafein.exposure import _threshold_suffix

        # Each cell's journey is one street leg; its totals are the
        # leg's reporting columns from the traversed-edge provenance
        # (true per-edge seconds — parking never joins the basis).
        reported = [
            exposure.street_leg_columns(edges) for edges in table["street_edges"]
        ]
        minute_columns = {
            f"{name}_minutes_above_{_threshold_suffix(threshold)}"
            for name in exposure.layers
            for threshold in exposure.thresholds(name)
        }
        for column in exposure.column_names():
            values = np.asarray([row[column] for row in reported], dtype=float)
            if column in minute_columns:
                # Journey-total semantics: a reachable cell without a
                # traversed edge (a same-coordinate pair, a
                # connector-only hop) spends 0 minutes at any level,
                # while its mean/max/coverage stay NaN.
                values = np.nan_to_num(values, nan=0.0)
            numeric[column] = values
    geometry = table["geometry"] if geometries else None
    return table["from"], table["to"], numeric, geometry


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
    intersection_delays=False,
    profile=None,
    delay_model=None,
    parking=None,
    workers=None,
    max_memory=None,
):
    """The reachable cells of a street travel-time matrix, in long format."""
    resolved = _street_time_resolution(
        origins,
        destinations,
        transport_mode=transport_mode,
        max_street_time=max_street_time,
        max_snap_distance=max_snap_distance,
        chunk=chunk,
        transit_only=transit_only,
        intersection_delays=intersection_delays,
        profile=profile,
        delay_model=delay_model,
        parking=parking,
    )
    query = resolved["query"]
    rows, columns, travel_time_s = _street_time_cells(
        network, query, resolved, workers=workers, max_memory=max_memory
    )
    from_ids = np.asarray(query.from_ids, dtype=object)
    to_ids = np.asarray(query.to_ids, dtype=object)
    return {
        "from_id": from_ids[rows],
        "to_id": to_ids[columns],
        "travel_time_s": travel_time_s,
    }


def _street_time_resolution(
    origins,
    destinations,
    *,
    transport_mode,
    max_street_time,
    max_snap_distance,
    chunk,
    transit_only,
    intersection_delays,
    profile,
    delay_model,
    parking,
):
    """Every result-affecting street-time input resolved exactly once."""
    from cafein import _parking
    from cafein.street_network import _resolved_delays

    resolved_parking = _parking.resolve(parking, transport_mode)
    query = _street_query(
        origins,
        destinations,
        transport_mode=transport_mode,
        max_street_time=max_street_time,
        max_snap_distance=max_snap_distance,
        chunk=chunk,
        transit_only=transit_only,
    )
    parking_seconds = (
        None
        if resolved_parking is None
        else _parking.destination_costs(resolved_parking, query.destination_points)[0]
    )
    return {
        "transport_mode": transport_mode,
        "query": query,
        "car_model": _resolved_delays(
            transport_mode, intersection_delays, profile, delay_model
        ),
        "parking_seconds": parking_seconds,
    }


def _street_time_cells(network, query, resolved, workers=None, max_memory=None):
    """One street-time batch: cell indices and their travel times."""
    transport_mode = resolved["transport_mode"]
    table = network._core.travel_time_matrix(
        query.origin_points,
        query.destination_points,
        transport_mode,
        query.max_seconds,
        query.max_snap_distance,
        car_model=resolved["car_model"],
        workers=_memory.width_or(workers),
    )
    _warn_unsnapped(
        table,
        query.from_ids,
        query.to_ids,
        network=f"the streets the {transport_mode} profile can use",
    )
    matrix = table["matrix"]
    rows, columns = np.nonzero(matrix != np.iinfo(np.uint32).max)
    travel_time_s = matrix[rows, columns]
    if resolved["parking_seconds"] is not None:
        # Parking seconds join every reachable cell by its destination.
        seconds = resolved["parking_seconds"]
        travel_time_s = travel_time_s + np.rint(seconds[columns]).astype(np.int64)
    return rows, columns, travel_time_s


def _time_matrix_data(
    network,
    origins,
    destinations,
    date,
    departure,
    *,
    arrive_by,
    window,
    _matrix=None,
    percentiles,
    confidence,
    chunk,
    router,
    exclude_routes,
    exclude_trips,
    exclude_stops,
    walking_speed_kmph,
    max_walking_time,
    max_snap_distance,
    street_policy,
    max_transfers,
    workers,
    max_memory=None,
):
    """The long-format columns of one moment's transit time matrix,
    dispatched by street policy exactly as the constructor did."""
    if street_policy is not None:
        from cafein.policy import CarParkPolicy

        if isinstance(street_policy, CarParkPolicy):
            # The query's walking knobs stay ACTIVE beside the car
            # plane, unlike under a StreetLegPolicy.
            named = [
                name
                for name, value in (
                    (
                        (
                            "arrival_time_window"
                            if arrive_by
                            else "departure_time_window"
                        ),
                        window,
                    ),
                    ("percentiles", percentiles),
                    ("confidence", confidence),
                )
                if value is not None
            ]
            if named or router != "auto":
                offending = ", ".join(named) or f"router={router!r}"
                raise ValueError(
                    f"CarParkPolicy does not combine with {offending}; "
                    "the policy matrix runs the earliest-arrival engine"
                )
            if id_sequence("exclude_stops", exclude_stops):
                raise ValueError(
                    "CarParkPolicy does not combine with stop "
                    "exclusions in this stage"
                )
            arm = (
                _car_park_arrive_by_time_columns
                if arrive_by
                else _car_park_time_columns
            )
            return arm(
                network,
                origins,
                destinations,
                date,
                departure,
                street_policy,
                max_transfers,
                chunk,
                (walking_speed_kmph, max_walking_time, max_snap_distance),
                exclude_routes,
                exclude_trips,
                workers=workers,
                max_memory=max_memory,
            )
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
        from cafein.policy import carriage_terms as _carriage_terms

        if _carriage_terms(street_policy) is not None:
            return _carriage_time_columns(
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
                workers=workers,
                max_memory=max_memory,
            )
        walk_only, walk_budget = _walking_only_policy(street_policy)
        if walk_only:
            # A walking-only policy IS the legacy walking matrix, at the
            # policy's one walking budget.
            return _time_columns(
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
                workers=workers,
                max_memory=max_memory,
            )
        return _policy_time_columns(
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
            workers=workers,
            max_memory=max_memory,
        )
    return _time_columns(
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
        arrive_by=arrive_by,
        workers=workers,
        max_memory=max_memory,
        _matrix=_matrix,
    )


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
    arrive_by=False,
    workers=None,
    max_memory=None,
    _matrix=None,
):
    """The reachable cells of the travel-time matrix, in long format.

    ``_matrix`` is one slot's precomputed ``(matrix, from_ids, to_ids,
    resolved)`` plane — the date-grouped core call computes every
    slot's rows at once and each slot decodes its own plane here."""
    if date is None or departure is None:
        raise TypeError("TravelTimeMatrix requires departure or arrival")
    if _matrix is not None:
        matrix, from_ids, to_ids, resolved = _matrix
        return _time_cells(matrix, from_ids, to_ids, resolved)
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
        arrive_by=arrive_by,
        workers=workers,
        max_memory=max_memory,
    )
    return _time_cells(matrix, from_ids, to_ids, resolved)


def _time_cells(matrix, from_ids, to_ids, resolved):
    """One matrix plane decoded into the long-format columns."""
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


def compare_matrices(a, b, *, columns=None, ratios=False):
    """Two matrix frames aligned cell by cell: ``a`` beside ``b``.

    ``a`` and ``b`` are long matrix frames (``TravelTimeMatrix``,
    ``TravelCostMatrix``, or any frame with their columns). Rows align
    on ``from_id`` and ``to_id`` plus whichever slot columns
    (``slot``, ``departure_time``, ``arrival_time``) both sides carry;
    a key column present on one side only, duplicate keys on either
    side, differing percentile column sets, and — without ``columns=``
    — a cost column present on one side only are refused by name.
    Key columns must have compatible dtypes on both sides: both
    integer kinds (merged losslessly as nullable ``Int64``, or
    ``UInt64`` when both sides are unsigned), or both
    string-valued (strings, objects, or a categorical of either,
    normalized to strings before the merge); anything else is refused
    rather than silently coerced.

    The result has one row per key in the union with a ``status``
    column (``both``/``only_a``/``only_b``) and, for every compared
    numeric column ``c``: ``c_a``, ``c_b``, and ``c_delta = c_b -
    c_a`` (NaN where a side is missing), plus ``c_ratio = c_b / c_a``
    with ``ratios=True`` (NaN where ``c_a`` is zero). Non-numeric
    columns (geometry, provenance strings) never join the comparison.
    Values are compared as given — the frames must share their
    ``output_time_units``, which the function cannot know.
    """
    frames = {"a": a, "b": b}
    for name, frame in frames.items():
        for key in ("from_id", "to_id"):
            if key not in frame.columns:
                raise ValueError(f"side {name} has no {key} column")
    keys = ["from_id", "to_id"]
    for key in ("slot", "departure_time", "arrival_time"):
        in_a, in_b = key in a.columns, key in b.columns
        if in_a != in_b:
            raise ValueError(
                f"{key} is a key column on side {'a' if in_a else 'b'} "
                "only; drop it or compute both sides with the same slots"
            )
        if in_a:
            keys.append(key)
    spreads = {
        name: sorted(
            column for column in frame.columns if column.startswith("travel_time_p")
        )
        for name, frame in frames.items()
    }
    if spreads["a"] != spreads["b"]:
        raise ValueError(
            f"side a carries percentiles {spreads['a']} and side b "
            f"{spreads['b']}; recompute one side with the other's "
            "percentiles"
        )

    def key_kind(dtype):
        # Only two key families compare: integers, and strings (plain,
        # object, or a categorical of either). Everything else —
        # floats, datetimes, booleans — is refused, never coerced.
        if isinstance(dtype, pd.CategoricalDtype):
            dtype = dtype.categories.dtype
        if pd.api.types.is_integer_dtype(dtype):
            return "integer"
        if dtype == object or pd.api.types.is_string_dtype(dtype):
            return "string"
        return None

    aligned = {}
    for key in keys:
        kinds = {name: key_kind(frame[key].dtype) for name, frame in frames.items()}
        if kinds["a"] != kinds["b"] or kinds["a"] is None:
            raise ValueError(
                f"{key} is {a[key].dtype} on side a and {b[key].dtype} on "
                "side b; compare integer keys with integer keys or "
                "string-valued keys with string-valued keys"
            )
        if kinds["a"] == "string":
            # An object column is string-valued only if its values are
            # strings: stringifying 1, None, or a timestamp would let
            # distinct keys alias ("1" == str(1)).
            for name, frame in frames.items():
                dtype = frame[key].dtype
                held = (
                    dtype.categories
                    if isinstance(dtype, pd.CategoricalDtype)
                    else pd.unique(frame[key].dropna())
                )
                if not all(isinstance(value, str) for value in held):
                    raise ValueError(
                        f"{key} on side {name} holds non-string values; "
                        "string-valued keys must hold strings"
                    )
        for name, frame in frames.items():
            if frame[key].isna().any():
                raise ValueError(
                    f"{key} on side {name} holds missing values; "
                    "keys identify cells and cannot be NA"
                )
        aligned[key] = kinds["a"]
    for name, frame in frames.items():
        if frame.duplicated(subset=keys).any():
            raise ValueError(
                f"side {name} carries duplicate ({', '.join(keys)}) keys; "
                "a matrix frame has one row per cell"
            )

    def numeric_columns(frame):
        return [
            column
            for column in frame.columns
            if column not in keys
            and pd.api.types.is_numeric_dtype(frame[column])
            and not pd.api.types.is_complex_dtype(frame[column])
        ]

    if columns is None:
        columns_a, columns_b = numeric_columns(a), numeric_columns(b)
        for only, name in (
            (set(columns_a) - set(columns_b), "a"),
            (set(columns_b) - set(columns_a), "b"),
        ):
            if only:
                raise ValueError(
                    f"side {name} alone carries {sorted(only)}; pass "
                    "columns= to compare a shared subset"
                )
        compared = columns_a
    else:
        compared = list(columns)
        for column in compared:
            for name, frame in frames.items():
                if column not in frame.columns:
                    raise ValueError(f"side {name} has no {column} column")
                if not pd.api.types.is_numeric_dtype(
                    frame[column]
                ) or pd.api.types.is_complex_dtype(frame[column]):
                    raise ValueError(
                        f"{column} is not numeric on side {name}; only "
                        "real numeric columns compare"
                    )
    left = pd.DataFrame({key: a[key] for key in keys})
    right = pd.DataFrame({key: b[key] for key in keys})
    for key, kind in aligned.items():
        if kind == "string":
            left[key] = left[key].astype(str)
            right[key] = right[key].astype(str)
        else:
            # One lossless 64-bit representation for the merge:
            # mixed-width or signed/unsigned pairs otherwise promote to
            # float64 and can alias large identifiers. Two unsigned
            # sides keep the unsigned domain.
            target = (
                "UInt64"
                if all(
                    pd.api.types.is_unsigned_integer_dtype(frame[key].dtype)
                    for frame in frames.values()
                )
                else "Int64"
            )
            for side in (left, right):
                try:
                    side[key] = side[key].astype(target)
                except (OverflowError, TypeError, ValueError):
                    raise ValueError(
                        f"{key} holds integers outside the signed 64-bit "
                        "range; cast both sides to strings to compare"
                    ) from None
    for column in compared:
        left[f"{column}_a"] = a[column].to_numpy(dtype="float64", na_value=np.nan)
        right[f"{column}_b"] = b[column].to_numpy(dtype="float64", na_value=np.nan)
    merged = left.merge(right, on=keys, how="outer", indicator=True, sort=False)
    status = merged.pop("_merge").map(
        {"both": "both", "left_only": "only_a", "right_only": "only_b"}
    )
    out = merged[keys].copy()
    out["status"] = status.astype(str)
    for column in compared:
        side_a = merged[f"{column}_a"].to_numpy(dtype=float)
        side_b = merged[f"{column}_b"].to_numpy(dtype=float)
        out[f"{column}_a"] = side_a
        out[f"{column}_b"] = side_b
        out[f"{column}_delta"] = side_b - side_a
        if ratios:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = side_b / side_a
            ratio[side_a == 0] = np.nan
            out[f"{column}_ratio"] = ratio
    return out


@_log.timed_computer(
    "matrix.cost_table",
    _log.matrix,
    "computing the travel cost table",
    "computed the travel cost table",
    street_identifier="matrix.streets",
    street_doing="computing the street cost table",
    street_done="computed the street cost table",
    is_street=lambda network: _is_street_network(network),
    details=_query_details,
)
def travel_cost_table(
    network,
    origins=None,
    destinations=None,
    departure=None,
    *,
    arrival=None,
    arrival_time_window=None,
    max_rides=8,
    optimize="time",
    departure_time_window=None,
    max_travel_time=None,
    factors=None,
    components=None,
    fares=None,
    geometries=False,
    chunk=None,
    router="auto",
    exclude_routes=(),
    exclude_trips=(),
    traveler=None,
    exclude_stops=(),
    walking_speed_kmph=None,
    max_walking_time=None,
    snap_distance=None,
    transport_mode=None,
    max_street_time=None,
    intersection_delays=False,
    profile=None,
    delay_model=None,
    parking=None,
    occupancy=None,
    vehicle_class=None,
    perspectives=None,
    costs=None,
    currency=None,
    cost_components=None,
    exposure=None,
    output=None,
    batch_size=None,
    resume=False,
    output_time_units="minutes",
):
    """The travel-cost matrix as a pyarrow Table — the shard-writing form.

    Semantics and parameters follow `TravelCostMatrix` — including the
    windowed optimize modes with their
    ``departure_time_window``/``max_travel_time``, the arrive-by cost
    axes (``arrival=`` beside ``arrival_time_window=``, whose batches
    slice the destination axis), the slot columns of a sequence or
    mapping of moments (``output=`` streams them too, every shard
    carrying all slots and the manifest listing them), the ``fares``
    pricing, and the ``router`` engine choice, though always
    over the time candidates
    (no ``candidates``/``bucket``); the output is an
    Arrow table with ``from_id`` and ``to_id`` dictionary-encoded over
    the origin and destination identifiers, the numeric columns wrapping
    the computed arrays zero-copy, and — with ``geometries=True`` — the
    ridden legs as WKB in a binary ``geometry`` column. A
    ``StreetNetwork`` computes the street cost matrix exactly as
    ``TravelCostMatrix`` does (``transport_mode=`` required, the car
    and monetary options included, ``exposure=`` columns included; the
    timetable arguments reject as there), as one Arrow table or
    streamed with ``output=``. The batch
    workflow writes one shard per origin chunk::

        network = TransportNetwork.load("network.cafein")
        table = travel_cost_table(network, ..., chunk=(k, n))
        pyarrow.parquet.write_table(table, f"shard-{k:04d}.parquet")

    Shards concatenate trivially. Requires pyarrow (install
    ``cafein[arrow]``). The Arrow surfaces keep ``from_id``/``to_id``
    dictionary-encoded as strings whatever the input dtype: shard
    schema stability across batches and resumes outranks dtype
    round-tripping here.

    With ``output=`` the matrix **streams to disk** instead of
    materialising: the fan-out axis — the origins with ``departure=``,
    the destinations with ``arrival=`` — is processed in
    ``batch_size`` slices
    (default 500) and the return value is a
    :class:`cafein.StreamingResult`, not a table. The form is chosen by
    suffix — a path whose final component ends in ``.parquet``
    (case-insensitive) is a single Parquet file written one row group
    per batch through one writer (splitting only past Parquet's 64 Mi
    rows-per-group cap), any other path is a directory of
    per-batch shards (``part-00000.parquet``, …) beside a
    ``manifest.json`` recording the query fingerprint and each shard's
    origin slice, row count, and completion marker. Existing targets
    are refused, never overwritten. The streamed output concatenates
    bit-for-bit to the unstreamed table (batch-major, the same origin
    order) for a single moment; with several slots each batch carries
    all slots, so streamed rows order batch-major rather than the
    table's slot-major and align by key, never by position. Either
    way ``from_id``/``to_id`` are dictionary-encoded over the same
    shared domains in every batch. Peak memory holds one batch's rows
    (plus the id domains): flat in the total origin count, linear in
    the destination count — a huge destination set splits across jobs.
    ``chunk=`` and ``output=`` compose: a chunked HPC job can itself
    stream its slice. An explicit ``batch_size`` or ``resume=True``
    without ``output=`` is rejected (``resume=False`` is inert).
    ``resume=True`` — directory form only — continues a matching
    partial run: completed shards are skipped untouched (never
    recomputed), a shard from a run killed between its rename and its
    manifest marker is rewritten, and the manifest fingerprint must
    match the query exactly — same network, inputs, parameters,
    ``chunk``, and ``batch_size`` — else the directory is refused,
    never overwritten.
    """
    if not _is_street_network(network):
        (
            exclude_routes,
            exclude_trips,
            exclude_stops,
            walking_speed_kmph,
            max_walking_time,
        ) = folded_constraints(
            traveler,
            network,
            exclude_routes,
            exclude_trips,
            exclude_stops,
            walking_speed_kmph,
            max_walking_time,
        )
    if not _is_street_network(network) and _is_point_frame(origins):
        refuse_wheelchair_streets(traveler, "travel_cost_table")
    try:
        import pyarrow
    except ImportError as error:
        raise ImportError(
            "Arrow tables need the optional pyarrow dependency; install "
            "cafein[arrow] or pyarrow"
        ) from error
    from cafein._units import duration_seconds, validated_output_time_units

    output_time_units = validated_output_time_units(output_time_units)
    if departure is not None and arrival is not None:
        raise ValueError("give exactly one of departure= or arrival=")
    arrive_by = arrival is not None
    from cafein._units import window_axis

    raw_window = window_axis(arrive_by, departure_time_window, arrival_time_window)
    if arrive_by and router == "tbtr":
        raise ValueError(
            "router='tbtr' does not serve arrival=; the reverse " "search rides RAPTOR"
        )
    slots, labeled, multi = _time_slots(departure, arrival, arrive_by)
    date, departure = slots[0][1], slots[0][2]
    if max_rides < 1:
        raise ValueError("max_rides must be at least 1")
    max_transfers = max_rides - 1
    window = duration_seconds(
        "arrival_time_window" if arrive_by else "departure_time_window", raw_window
    )
    within = duration_seconds("max_travel_time", max_travel_time)
    max_walking_time = duration_seconds("max_walking_time", max_walking_time)
    max_snap_distance = snap_distance
    exposure_snapshot = None
    if exposure is not None and not _is_street_network(network):
        router = _validate_transit_exposure(
            exposure,
            network,
            optimize=optimize,
            window=window,
            within=within,
            arrival=arrival,
            arrive_by=arrive_by,
            router=router,
        )
        exposure_snapshot = _exposure_snapshot(exposure)
    if _is_street_network(network):
        chunk = None if chunk is None else tuple(int(part) for part in chunk)
        resolved = _street_cost_resolution(
            network,
            origins,
            destinations,
            transport_mode=transport_mode,
            max_street_time=duration_seconds("max_street_time", max_street_time),
            max_snap_distance=max_snap_distance,
            chunk=chunk,
            transit_only={
                "departure": departure,
                "arrival": arrival,
                "traveler": traveler,
                "departure_time_window": window,
                "max_travel_time": within,
                "fares": fares,
                "walking_speed_kmph": walking_speed_kmph,
                "max_walking_time": max_walking_time,
                "max_rides": None if max_rides == 8 else max_rides,
                "optimize": None if optimize == "time" else optimize,
                "router": None if router == "auto" else router,
                "exclude_routes": id_sequence("exclude_routes", exclude_routes) or None,
                "exclude_trips": id_sequence("exclude_trips", exclude_trips) or None,
                "exclude_stops": id_sequence("exclude_stops", exclude_stops) or None,
            },
            factors=factors,
            components=components,
            intersection_delays=intersection_delays,
            profile=profile,
            delay_model=delay_model,
            parking=parking,
            occupancy=occupancy,
            vehicle_class=vehicle_class,
            perspectives=perspectives,
            costs=costs,
            currency=currency,
            cost_components=cost_components,
            exposure=exposure,
        )
        if output is None:
            if batch_size is not None:
                raise ValueError("batch_size requires output=")
            if resume:
                raise ValueError("resume=True requires output=")
            return _street_arrow_table(
                network, resolved, geometries, pyarrow, output_time_units
            )
        size = _stream_size(batch_size, resume)
        return _stream_street_cost(
            "travel_cost_table",
            network,
            resolved,
            geometries,
            chunk,
            output,
            size,
            pyarrow,
            resume=resume,
            output_time_units=output_time_units,
        )
    _reject_cost_street_args(
        transport_mode,
        max_street_time,
        intersection_delays,
        profile,
        delay_model,
        parking,
        occupancy,
        vehicle_class,
        perspectives,
        costs,
        currency,
        cost_components,
    )
    if output is None:
        if batch_size is not None:
            raise ValueError("batch_size requires output=")
        if resume:
            raise ValueError("resume=True requires output=")
        exclude_routes = id_sequence("exclude_routes", exclude_routes)
        exclude_trips = id_sequence("exclude_trips", exclude_trips)
        exclude_stops = id_sequence("exclude_stops", exclude_stops)
        if chunk is not None:
            chunk = tuple(chunk)
        origins = sequence_not_string("origins", origins)
        destinations = sequence_not_string("destinations", destinations)
        components = component_selection(components)
        from cafein import emissions

        _validate_cost_query(
            slots[0][1],
            slots[0][2],
            optimize,
            window,
            within,
            fares,
            router,
            arrive_by=arrive_by,
        )
        from_ids, to_ids, points, to_stops = _cost_endpoints(
            network, origins, destinations, None if arrive_by else chunk
        )
        if arrive_by and points is None:
            to_axis = to_stops if to_stops is not None else list(to_ids)
            if chunk is not None:
                to_axis = to_axis[_chunk_slice(len(to_axis), chunk)]
            endpoints = ("stops", from_ids, to_axis)
        elif arrive_by:
            origin_points, destination_points = points
            if chunk is not None:
                keep = _chunk_slice(len(to_ids), chunk)
                to_ids = to_ids[keep]
                destination_points = destination_points[keep]
            endpoints = ("points", from_ids, origin_points, to_ids, destination_points)
        elif points is None:
            endpoints = ("stops", from_ids, to_stops)
        else:
            origin_points, destination_points = points
            endpoints = ("points", from_ids, origin_points, to_ids, destination_points)
        _resolved = (
            emissions.trip_factors(network, factors, components),
            None if fares is None else fares._flat_tables(network),
            endpoints,
        )
        tables = []
        for label, date, clock in slots:
            table, from_ids, to_ids = _cost_columns(
                network,
                origins,
                destinations,
                date,
                clock,
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
                arrive_by=arrive_by,
                walking_speed_kmph=walking_speed_kmph,
                max_walking_time=max_walking_time,
                max_snap_distance=max_snap_distance,
                exposure=exposure_snapshot,
                _resolved=_resolved,
            )
            if exposure is not None:
                exposure._check_network(network)
            arrow = _arrow_table(
                table,
                pyarrow.array(from_ids, type=pyarrow.string()),
                pyarrow.array(to_ids, type=pyarrow.string()),
                0,
                fares,
                geometries,
                pyarrow,
                output_time_units,
                exposure=exposure_snapshot,
            )
            if multi:
                arrow = _insert_slot_arrow_columns(
                    arrow, arrive_by, label, labeled, date, clock, pyarrow
                )
            tables.append(arrow)
        return pyarrow.concat_tables(tables) if multi else tables[0]
    size = _stream_size(batch_size, resume)
    return _stream_transit_cost(
        "travel_cost_table",
        network,
        origins,
        destinations,
        slots,
        labeled,
        multi,
        max_transfers=max_transfers,
        optimize=optimize,
        window=window,
        within=within,
        factors=factors,
        components=components,
        fares=fares,
        candidates="time",
        bucket=25.0,
        router=router,
        arrive_by=arrive_by,
        exclude_routes=exclude_routes,
        exclude_trips=exclude_trips,
        exclude_stops=exclude_stops,
        geometries=geometries,
        chunk=chunk,
        walking_speed_kmph=walking_speed_kmph,
        max_walking_time=max_walking_time,
        max_snap_distance=max_snap_distance,
        output=output,
        size=size,
        pyarrow=pyarrow,
        resume=resume,
        output_time_units=output_time_units,
        exposure=exposure_snapshot,
    )


def _stream_transit_cost(
    operation,
    network,
    origins,
    destinations,
    slots,
    labeled,
    multi,
    *,
    max_transfers,
    optimize,
    window,
    within,
    factors,
    components,
    fares,
    candidates,
    bucket,
    router,
    exclude_routes,
    exclude_trips,
    exclude_stops,
    geometries,
    chunk,
    walking_speed_kmph,
    max_walking_time,
    max_snap_distance,
    output,
    size,
    pyarrow,
    arrive_by=False,
    resume=False,
    output_time_units="minutes",
    exposure=None,
    workers=None,
    max_memory=None,
):
    """The transit cost matrix streamed in sliced-axis batches — the
    origins on the departure axis, the destinations (the reverse
    fan-out axis) with ``arrive_by`` — shared by ``travel_cost_table``
    and ``TravelCostMatrix.to_parquet``."""
    # Everything result-affecting resolves and freezes ONCE, before the
    # output is claimed: one-shot iterables drain here, later mutation of
    # the input frames cannot desynchronise batches from the fingerprint,
    # and an invalid query never leaves an empty claimed output behind.
    date, departure = slots[0][1], slots[0][2]
    _validate_cost_query(
        date, departure, optimize, window, within, fares, router, arrive_by=arrive_by
    )
    if candidates not in ("time", "pareto"):
        raise ValueError("candidates must be 'time' or 'pareto'")
    exclude_routes = list(id_sequence("exclude_routes", exclude_routes))
    exclude_trips = list(id_sequence("exclude_trips", exclude_trips))
    exclude_stops = list(id_sequence("exclude_stops", exclude_stops))
    if chunk is not None:
        chunk = tuple(int(part) for part in chunk)
    from_ids, to_ids, points, to_stops = _cost_endpoints(
        network, origins, destinations, None if arrive_by else chunk
    )
    if arrive_by:
        # The sliced (and chunked) axis is the destination selection —
        # the reverse fan-out axis, exactly as the constructor's chunk.
        # A stop table's ``to_id`` domain stays the full one so its
        # rows keep their global keys.
        if points is None:
            to_axis = to_stops if to_stops is not None else list(to_ids)
            if chunk is not None:
                to_axis = to_axis[_chunk_slice(len(to_axis), chunk)]
        else:
            if chunk is not None:
                keep = _chunk_slice(len(to_ids), chunk)
                origin_points_all, destination_points_all = points
                to_ids = to_ids[keep]
                points = (origin_points_all, destination_points_all[keep])
            to_axis = to_ids
    if candidates == "pareto":
        if optimize != "emissions":
            raise ValueError("candidates='pareto' requires optimize='emissions'")
        if points is not None:
            raise ValueError("pareto candidates require stop origins and destinations")
    from cafein import emissions

    trip_factors = emissions.trip_factors(network, factors, components)
    fare_tables = None if fares is None else fares._flat_tables(network)
    columns = [
        "from_id",
        "to_id",
        "travel_time",
        "transfers",
        "transit_distance_m",
        "walk_distance_m",
        "emissions",
    ]
    if fares is not None:
        columns.append("fare")
    if exposure is not None:
        columns += exposure.column_names()
    if geometries:
        columns.append("geometry")
    if multi:
        at = columns.index("to_id") + 1
        columns[at:at] = (["slot"] if labeled else []) + [
            "arrival_time" if arrive_by else "departure_time"
        ]
    parameters = {
        "date": date,
        "departure": departure,
        "slots": None if not multi else [list(slot) for slot in slots],
        "time_axis": "arrival" if arrive_by else "departure",
        "max_transfers": max_transfers,
        "optimize": optimize,
        "window": window,
        "within": within,
        # Sorted for the hash only: the resolver's order is not
        # canonical across processes, the (trip, factor) SET is.
        "factors": sorted(trip_factors),
        "fares": fare_tables,
        "geometries": bool(geometries),
        "chunk": None if chunk is None else list(chunk),
        "candidates": candidates,
        "bucket": bucket,
        "router": router,
        "destinations": to_stops,
        "exclude_routes": exclude_routes,
        "exclude_trips": exclude_trips,
        "exclude_stops": exclude_stops,
        "walking_speed_kmph": walking_speed_kmph,
        "max_walking_time": max_walking_time,
        "max_snap_distance": max_snap_distance,
        # The fingerprint hashes the layer data itself, so a resume with
        # a same-named but different exposure can never wrongly match.
        "exposure": None if exposure is None else exposure._fingerprint(),
        "output_time_units": output_time_units,
    }

    def make_moment(rows, shared_from, shared_to, date, departure):
        if arrive_by and points is None:
            endpoints = ("stops", from_ids, to_axis[rows])
        elif arrive_by:
            origin_points, destination_points = points
            endpoints = (
                "points",
                from_ids,
                origin_points,
                to_ids[rows],
                destination_points[rows],
            )
        elif points is None:
            endpoints = ("stops", from_ids[rows], to_stops)
        else:
            origin_points, destination_points = points
            endpoints = (
                "points",
                from_ids[rows],
                origin_points[rows],
                to_ids,
                destination_points,
            )
        table, _, _ = _cost_columns(
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
            geometries=geometries,
            chunk=chunk,
            router=router,
            exclude_routes=exclude_routes,
            exclude_trips=exclude_trips,
            exclude_stops=exclude_stops,
            arrive_by=arrive_by,
            walking_speed_kmph=walking_speed_kmph,
            max_walking_time=max_walking_time,
            max_snap_distance=max_snap_distance,
            exposure=exposure,
            _resolved=(trip_factors, fare_tables, endpoints),
            workers=workers,
            max_memory=max_memory,
        )
        return _arrow_table(
            table,
            shared_from,
            shared_to,
            0 if arrive_by else rows.start,
            fares,
            geometries,
            pyarrow,
            output_time_units,
            # Point rows key destinations by batch position; stop rows
            # keep their global keys, so only the point form offsets.
            to_offset=rows.start if arrive_by and points is not None else 0,
            exposure=exposure,
        )

    make_batch = _slot_batches(make_moment, slots, labeled, multi, arrive_by, pyarrow)
    return _stream_run(
        operation,
        network,
        columns,
        parameters,
        from_ids,
        to_ids,
        points,
        output,
        size,
        make_batch,
        pyarrow,
        resume=resume,
        slice_axis="to" if arrive_by else "from",
        axis_count=len(to_axis) if arrive_by else None,
        manifest_extra=({"slots": [list(slot) for slot in slots]} if multi else None),
    )


def _slot_batches(make_moment, slots, labeled, multi, arrive_by, pyarrow):
    """Wrap a per-moment batch builder into the slot loop: one table
    per batch carrying every slot's rows, the slot columns appended as
    in the constructors (the label dictionary-encoded for shards)."""

    def make_batch(rows, shared_from, shared_to):
        tables = []
        for label, date, clock in slots:
            arrow = make_moment(rows, shared_from, shared_to, date, clock)
            if multi:
                arrow = _insert_slot_arrow_columns(
                    arrow, arrive_by, label, labeled, date, clock, pyarrow
                )
                if labeled:
                    at = arrow.schema.get_field_index("slot")
                    arrow = arrow.set_column(
                        at,
                        "slot",
                        arrow.column("slot").combine_chunks().dictionary_encode(),
                    )
            tables.append(arrow)
        if not multi:
            return tables[0]
        # The writer's contract is one chunk per column over the shared
        # dictionary domain; concatenation re-chunks, so unify.
        return pyarrow.concat_tables(tables).combine_chunks()

    return make_batch


def _stream_run(
    operation,
    network,
    columns,
    parameters,
    from_ids,
    to_ids,
    points,
    output,
    size,
    make_batch,
    pyarrow,
    resume=False,
    dictionary_columns=("from_id", "to_id"),
    slice_axis="from",
    axis_count=None,
    manifest_extra=None,
):
    """The shared streaming driver: fingerprint, claim, batch, write.

    ``make_batch(rows, shared_from, shared_to)`` returns one batch's
    Arrow table for the sliced-axis slice ``rows`` — the origins by
    default, the destinations with ``slice_axis="to"`` (the arrive-by
    fan-out axis) — inputs must already be frozen by the caller. Used
    by ``travel_cost_table`` and the matrix computers' ``to_parquet``
    classmethods identically. With
    ``resume=True`` (directory form) the completed shards of a matching
    partial run are skipped — their batches never compute — and only
    the remainder routes.
    """
    from cafein import _streaming

    shared_from = pyarrow.array(from_ids, type=pyarrow.string())
    shared_to = pyarrow.array(to_ids, type=pyarrow.string())
    # ``axis_count`` overrides when the sliced axis is a selection
    # narrower than its id domain (a stop cost table's destinations).
    if axis_count is not None:
        count = axis_count
    else:
        count = len(from_ids) if slice_axis == "from" else len(to_ids)
    batches = max(1, -(-count // size))
    fingerprint = _streaming.fingerprint(
        operation,
        columns,
        _streaming.network_digest(network),
        dict(parameters, batch_size=size),
        from_ids,
        to_ids,
        points,
    )
    mode, path = _streaming.resolve_output(output, resume)
    manifest = None
    completed = frozenset()
    if resume:
        manifest, completed = _streaming.prepare_resume(
            path, fingerprint, size, count, (manifest_extra or {}).get("slots")
        )

    def produce():
        for index in range(batches):
            if index in completed:
                continue
            rows = slice(index * size, min((index + 1) * size, count))
            arrow = make_batch(rows, shared_from, shared_to)
            yield index, rows.start, rows.stop, arrow
            # Release this batch before the next one computes: the
            # working set holds one batch, not two.
            del arrow

    manifest_seed = {
        "operation": operation,
        "fingerprint": fingerprint,
        "fingerprint_version": _streaming.FINGERPRINT_VERSION,
        "batch_size": size,
        # Historically named; the sliced axis's length, whichever axis
        # batches (recorded beside it).
        "origin_count": count,
        "batch_axis": slice_axis,
    }
    if manifest_extra:
        manifest_seed.update(manifest_extra)
    shared = {"from_id": shared_from, "to_id": shared_to}
    return _streaming.write_stream(
        mode,
        path,
        produce(),
        manifest_seed,
        {name: shared[name] for name in dictionary_columns},
        manifest=manifest,
    )


def _validate_transit_exposure(
    exposure,
    network,
    *,
    optimize,
    window,
    within,
    candidates="time",
    arrival=None,
    arrive_by=False,
    router="auto",
):
    """The exposure slice is the single-departure, time-optimal mode:
    refuse the windowed, Pareto, and arrive-by combinations by name,
    force the RAPTOR engine (its journey reconstruction carries the
    skeletons), and verify the binding. Returns the resolved router."""
    if router not in ("auto", "raptor", "tbtr"):
        raise ValueError("router must be 'auto', 'raptor', or 'tbtr'")
    for name, value in (
        ("optimize", None if optimize == "time" else optimize),
        ("arrival", arrival if arrive_by else None),
        ("departure_time_window", None if arrive_by else window),
        ("arrival_time_window", window if arrive_by else None),
        ("max_travel_time", within),
        ("candidates", None if candidates == "time" else candidates),
    ):
        if value is not None:
            raise ValueError(
                "exposure= joins the single-departure time-optimal cost "
                f"matrix; it does not combine with {name}="
            )
    if router == "tbtr":
        raise ValueError(
            "exposure= rides the RAPTOR engine's journey reconstruction; "
            "router='tbtr' does not serve it"
        )
    exposure._check_network(network)
    return "raptor"


_NO_STOP = 2**32 - 1
"""The core's absent-stop sentinel on cost rows."""


def _transit_exposure_columns(
    network,
    exposure,
    table,
    *,
    origin_points,
    destination_points,
    max_snap_distance,
):
    """Per-cell exposure totals folded from the journey skeletons — the
    ``exposure_totals()`` arithmetic at matrix scale.

    Every DISTINCT walk component reconstructs once — access per
    (origin, boarding stop), egress per (destination, alighting stop),
    transfers per stored pair, the walking-direct cells per pair — into
    a per-minute ``leg_columns`` record; waits fold straight from the
    sampled stop values. The per-cell combine is ``np.add.at`` over the
    flattened skeleton, never a per-cell reconstruction. A component
    whose walk cannot be reconstructed folds as an uncovered leg,
    exactly as an itinerary leg without provenance does."""
    from cafein import streets as _streets
    from cafein.exposure import _threshold_suffix

    snap_limit = (
        _streets.MAX_SNAP_DISTANCE if max_snap_distance is None else max_snap_distance
    )
    layers = list(exposure.layers)
    thresholds = {name: exposure.thresholds(name) for name in layers}
    cells = len(table["from"])
    weight_total = np.zeros(cells)
    weighted_mean = {name: np.zeros(cells) for name in layers}
    weighted_coverage = {name: np.zeros(cells) for name in layers}
    maxima = {name: np.full(cells, -np.inf) for name in layers}
    minutes = {
        (name, threshold): np.zeros(cells)
        for name in layers
        for threshold in thresholds[name]
    }

    offsets = np.asarray(table["piece_offsets"], dtype=np.int64)
    kind = np.asarray(table["piece_kind"])
    stop = np.asarray(table["piece_stop"], dtype=np.int64)
    other = np.asarray(table["piece_other"], dtype=np.int64)
    seconds = np.asarray(table["piece_seconds"], dtype=float)
    cell_of = np.repeat(np.arange(cells, dtype=np.int64), np.diff(offsets))
    from_index = np.asarray(table["from"], dtype=np.int64)
    to_index = np.asarray(table["to"], dtype=np.int64)

    # Waits fold straight from the sampled stop values, wait_columns
    # semantics: an unsampleable stop counts its time at value 0 with
    # coverage 0, never NaN.
    stop_ids = [identifier for identifier, _, _ in network.stops]
    stop_values = {
        name: np.asarray(
            [
                exposure._stop_values.get(name, {}).get(identifier, np.nan)
                for identifier in stop_ids
            ],
            dtype=float,
        )
        for name in layers
    }
    waits = kind == 3
    if waits.any():
        cell = cell_of[waits]
        wait_seconds = seconds[waits]
        wait_stop = stop[waits]
        np.add.at(weight_total, cell, wait_seconds)
        for name in layers:
            value = stop_values[name][wait_stop]
            covered = np.isfinite(value)
            np.add.at(
                weighted_mean[name], cell, np.where(covered, value, 0.0) * wait_seconds
            )
            np.add.at(
                weighted_coverage[name], cell, covered.astype(float) * wait_seconds
            )
            np.fmax.at(maxima[name], cell[covered], value[covered])
            for threshold in thresholds[name]:
                above = covered & (value >= threshold)
                np.add.at(
                    minutes[(name, threshold)],
                    cell,
                    np.where(above, wait_seconds / 60.0, 0.0),
                )

    # Walk components: one reconstruction per distinct component, one
    # per-minute leg record each, then a vectorized fold. The cache
    # lives on the frozen snapshot, so a streamed run's batches share
    # every reconstruction instead of repeating it per batch.
    walks = ~waits
    core = network._core
    records = getattr(exposure, "_component_cache", None)
    if records is None:
        records = {}
        exposure._component_cache = records

    def leg_record(walked):
        if walked is None:
            return None
        edges, meters = walked
        columns = exposure.leg_columns(edges, 60.0, meters)
        if not np.isfinite(columns[f"{layers[0]}_coverage"]):
            return None
        return columns

    if walks.any():
        indices = np.flatnonzero(walks)
        component = np.full(len(indices), -1, dtype=np.int64)
        component_records = []
        component_ids = {}
        for position, piece in enumerate(indices):
            piece_cell = int(cell_of[piece])
            at = int(stop[piece])
            # Coordinate-keyed: origin/destination indices are local to
            # each sliced call, while the snapshot's cache spans a whole
            # streamed run — the coordinate is the walk's identity.
            if kind[piece] == 0:
                origin = origin_points[int(from_index[piece_cell])]
                key = ("a", origin[0], origin[1], at)
                if key not in records:
                    records[key] = leg_record(
                        core._coordinate_stop_walk_edges(
                            origin[0], origin[1], at, snap_limit
                        )
                    )
            elif kind[piece] == 1:
                destination = destination_points[int(to_index[piece_cell])]
                key = ("e", destination[0], destination[1], at)
                if key not in records:
                    records[key] = leg_record(
                        core._coordinate_stop_walk_edges(
                            destination[0], destination[1], at, snap_limit
                        )
                    )
            else:
                key = ("t", at, int(other[piece]))
                if key not in records:
                    records[key] = leg_record(
                        core._transfer_walk_edges(at, int(other[piece]))
                    )
            columns = records[key]
            if columns is None:
                continue
            if key not in component_ids:
                component_ids[key] = len(component_records)
                component_records.append(columns)
            component[position] = component_ids[key]
        chosen_mask = component >= 0
        if chosen_mask.any():
            cell = cell_of[indices[chosen_mask]]
            walk_seconds = seconds[indices[chosen_mask]]
            chosen = component[chosen_mask]

            def gather(column):
                return np.asarray(
                    [record[column] for record in component_records], dtype=float
                )[chosen]

            np.add.at(weight_total, cell, walk_seconds)
            for name in layers:
                np.add.at(
                    weighted_mean[name], cell, gather(f"{name}_mean") * walk_seconds
                )
                np.add.at(
                    weighted_coverage[name],
                    cell,
                    gather(f"{name}_coverage") * walk_seconds,
                )
                top = gather(f"{name}_max")
                finite = np.isfinite(top)
                np.fmax.at(maxima[name], cell[finite], top[finite])
                for threshold in thresholds[name]:
                    column = f"{name}_minutes_above_{_threshold_suffix(threshold)}"
                    np.add.at(
                        minutes[(name, threshold)],
                        cell,
                        gather(column) * walk_seconds / 60.0,
                    )

    # Walking-direct cells: the whole journey is one walk between the
    # query coordinates; reconstruct per distinct pair.
    if origin_points is not None and destination_points is not None:
        rides = np.asarray(table["rides"])
        travel_seconds = np.asarray(table["travel_time_s"], dtype=float)
        access = np.asarray(table["access_stop"], dtype=np.int64)
        direct = (rides == 0) & (access == _NO_STOP) & (travel_seconds > 0)
        for index in np.flatnonzero(direct):
            origin = origin_points[int(from_index[index])]
            destination = destination_points[int(to_index[index])]
            key = ("d", origin[0], origin[1], destination[0], destination[1])
            if key not in records:
                records[key] = leg_record(
                    core._direct_walk_edges(
                        origin[0],
                        origin[1],
                        destination[0],
                        destination[1],
                        snap_limit,
                    )
                )
            record = records[key]
            if record is None:
                continue
            weight = travel_seconds[index]
            weight_total[index] += weight
            for name in layers:
                weighted_mean[name][index] += record[f"{name}_mean"] * weight
                weighted_coverage[name][index] += record[f"{name}_coverage"] * weight
                top = record[f"{name}_max"]
                if np.isfinite(top):
                    maxima[name][index] = max(maxima[name][index], top)
                for threshold in thresholds[name]:
                    column = f"{name}_minutes_above_{_threshold_suffix(threshold)}"
                    minutes[(name, threshold)][index] += record[column] * weight / 60.0

    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for name in layers:
            out[f"{name}_mean"] = weighted_mean[name] / weight_total
            out[f"{name}_max"] = np.where(
                np.isfinite(maxima[name]), maxima[name], np.nan
            )
            out[f"{name}_coverage"] = weighted_coverage[name] / weight_total
            for threshold in thresholds[name]:
                column = f"{name}_minutes_above_{_threshold_suffix(threshold)}"
                out[column] = minutes[(name, threshold)]
    return out


def _street_arrow_table(network, resolved, geometries, pa, output_time_units):
    """The street cost matrix as one Arrow table — ``travel_cost_table``'s
    street form, the streamed shards' schema in a single batch."""
    from cafein._cafein import STREET_DISTANCE_PROVENANCE
    from cafein._units import travel_time_output

    query = resolved["query"]
    from_index, to_index, numeric, wkb = _street_cost_cells(
        network, query, geometries=geometries, resolved=resolved
    )
    count = len(from_index)
    data = {
        "from_id": pa.DictionaryArray.from_arrays(
            pa.array(from_index),
            pa.array(list(query.from_ids), type=pa.string()),
        ),
        "to_id": pa.DictionaryArray.from_arrays(
            pa.array(to_index),
            pa.array(list(query.to_ids), type=pa.string()),
        ),
        "travel_time": pa.array(
            travel_time_output(numeric["travel_time_s"], output_time_units)
        ),
        "distance_m": pa.array(numeric["distance_m"]),
        "network_distance_m": pa.array(numeric["network_distance_m"]),
        "connector_distance_m": pa.array(numeric["connector_distance_m"]),
        "distance_provenance": pa.array(
            [STREET_DISTANCE_PROVENANCE] * count, type=pa.string()
        ),
        "emissions": pa.array(numeric["emissions"]),
    }
    for name, values in numeric.items():
        if name.startswith("cost_"):
            data[name] = pa.array(values)
    if resolved["account"] is not None:
        data["currency"] = pa.array([resolved["account"][2]] * count, type=pa.string())
    exposure = resolved["exposure"]
    if exposure is not None:
        for name in exposure.column_names():
            data[name] = pa.array(numeric[name])
    if geometries:
        data["geometry"] = pa.array(list(wkb), type=pa.binary())
    return pa.table(data)


def _stream_street_cost(
    operation,
    network,
    resolved,
    geometries,
    chunk,
    output,
    size,
    pa,
    resume=False,
    output_time_units="minutes",
    workers=None,
    max_memory=None,
):
    """The street cost matrix streamed in origin batches over a frozen
    resolution — `TravelCostMatrix.to_parquet`'s street arm."""
    from cafein._cafein import STREET_DISTANCE_PROVENANCE
    from cafein._units import travel_time_output

    query = resolved["query"]
    account = resolved["account"]
    exposure = resolved["exposure"]
    columns = [
        "from_id",
        "to_id",
        "travel_time",
        "distance_m",
        "network_distance_m",
        "connector_distance_m",
        "distance_provenance",
        "emissions",
    ]
    if account is not None:
        totals, breakdown, label = account
        columns += [f"cost_{perspective}" for perspective in totals]
        columns += [
            f"cost_{perspective}_{component}" for perspective, component in breakdown
        ]
        columns.append("currency")
    if exposure is not None:
        columns += exposure.column_names()
    if geometries:
        columns.append("geometry")
    parameters = {
        "transport_mode": resolved["transport_mode"],
        "max_street_time": query.max_seconds,
        "max_snap_distance": query.max_snap_distance,
        "geometries": bool(geometries),
        "chunk": None if chunk is None else list(chunk),
        "car_model": resolved["car_model"],
        "parking_costs": resolved["parking_costs"],
        "occupancy": resolved["occupancy"],
        "factor": resolved["factor"],
        "account": None if account is None else [totals, breakdown, label],
        # The fingerprint hashes the layer data itself, so a resume with
        # a same-named but different exposure can never wrongly match.
        "exposure": None if exposure is None else exposure._fingerprint(),
        "output_time_units": output_time_units,
    }

    def make_batch(rows, shared_from, shared_to):
        batch = _StreetQuery(
            query.from_ids[rows],
            query.to_ids,
            query.origin_points[rows],
            query.destination_points,
            query.max_seconds,
            query.max_snap_distance,
        )
        from_index, to_index, numeric, wkb = _street_cost_cells(
            network,
            batch,
            geometries=geometries,
            resolved=resolved,
            workers=workers,
            max_memory=max_memory,
        )
        count = len(from_index)
        data = {
            "from_id": pa.DictionaryArray.from_arrays(
                pa.array(from_index + rows.start if rows.start else from_index),
                shared_from,
            ),
            "to_id": pa.DictionaryArray.from_arrays(pa.array(to_index), shared_to),
            "travel_time": pa.array(
                travel_time_output(numeric["travel_time_s"], output_time_units)
            ),
            "distance_m": pa.array(numeric["distance_m"]),
            "network_distance_m": pa.array(numeric["network_distance_m"]),
            "connector_distance_m": pa.array(numeric["connector_distance_m"]),
            "distance_provenance": pa.array(
                [STREET_DISTANCE_PROVENANCE] * count, type=pa.string()
            ),
            "emissions": pa.array(numeric["emissions"]),
        }
        for name, values in numeric.items():
            if name.startswith("cost_"):
                data[name] = pa.array(values)
        if account is not None:
            data["currency"] = pa.array([label] * count, type=pa.string())
        if exposure is not None:
            for name in exposure.column_names():
                data[name] = pa.array(numeric[name])
        if geometries:
            data["geometry"] = pa.array(list(wkb), type=pa.binary())
        return pa.table(data)

    return _stream_run(
        operation,
        network,
        columns,
        parameters,
        list(query.from_ids),
        list(query.to_ids),
        (query.origin_points, query.destination_points),
        output,
        size,
        make_batch,
        pa,
        resume=resume,
    )


def _stream_street_time(
    operation,
    network,
    resolved,
    chunk,
    output,
    size,
    pa,
    resume=False,
    output_time_units="minutes",
    workers=None,
    max_memory=None,
):
    """The street time matrix streamed in origin batches over a frozen
    resolution — `TravelTimeMatrix.to_parquet`'s street arm."""
    from cafein._units import travel_time_output

    query = resolved["query"]
    parameters = {
        "transport_mode": resolved["transport_mode"],
        "max_street_time": query.max_seconds,
        "max_snap_distance": query.max_snap_distance,
        "chunk": None if chunk is None else list(chunk),
        "car_model": resolved["car_model"],
        "parking_seconds": resolved["parking_seconds"],
        "output_time_units": output_time_units,
    }

    def make_batch(rows, shared_from, shared_to):
        batch = _StreetQuery(
            query.from_ids[rows],
            query.to_ids,
            query.origin_points[rows],
            query.destination_points,
            query.max_seconds,
            query.max_snap_distance,
        )
        cell_rows, cell_columns, travel_time_s = _street_time_cells(
            network,
            batch,
            resolved,
            workers=workers,
            max_memory=max_memory,
        )
        return pa.table(
            {
                "from_id": pa.DictionaryArray.from_arrays(
                    pa.array(cell_rows + rows.start if rows.start else cell_rows),
                    shared_from,
                ),
                "to_id": pa.DictionaryArray.from_arrays(
                    pa.array(cell_columns), shared_to
                ),
                "travel_time": pa.array(
                    travel_time_output(travel_time_s, output_time_units)
                ),
            }
        )

    return _stream_run(
        operation,
        network,
        ["from_id", "to_id", "travel_time"],
        parameters,
        list(query.from_ids),
        list(query.to_ids),
        (query.origin_points, query.destination_points),
        output,
        size,
        make_batch,
        pa,
        resume=resume,
    )


def _stream_transit_time(
    operation,
    network,
    origins,
    destinations,
    slots,
    labeled,
    multi,
    *,
    max_transfers,
    window,
    percentiles,
    confidence,
    chunk,
    router,
    exclude_routes,
    exclude_trips,
    exclude_stops,
    walking_speed_kmph,
    max_walking_time,
    max_snap_distance,
    output,
    size,
    pyarrow,
    resume=False,
    output_time_units="minutes",
    arrive_by=False,
    workers=None,
    max_memory=None,
):
    """The transit time matrix streamed in sliced-axis batches — the
    origins on the departure axis, the destinations (the fan-out axis)
    with ``arrive_by`` — `TravelTimeMatrix.to_parquet`'s transit
    arm."""
    from cafein._units import travel_time_output
    from cafein.network import _window_percentiles

    date, departure = slots[0][1], slots[0][2]
    if date is None or departure is None:
        raise TypeError("TravelTimeMatrix requires departure or arrival")
    if router not in ("auto", "raptor", "tbtr"):
        raise ValueError(f"router must be 'auto', 'raptor', or 'tbtr', not {router!r}")
    # Frozen once: a one-shot percentile iterable drains here, and every
    # batch routes the resolved list, not the caller's mutable value.
    resolved_percentiles = _window_percentiles(window, percentiles, confidence)
    exclude_routes = list(id_sequence("exclude_routes", exclude_routes))
    exclude_trips = list(id_sequence("exclude_trips", exclude_trips))
    exclude_stops = list(id_sequence("exclude_stops", exclude_stops))
    if chunk is not None:
        chunk = tuple(int(part) for part in chunk)
    from_ids, to_ids, points, _ = _cost_endpoints(
        network, origins, destinations, None if arrive_by else chunk
    )
    if points is None and destinations is not None:
        raise ValueError("destinations apply to point origins")
    if arrive_by and chunk is not None:
        # The arrive-by chunk slices the destination axis, exactly as
        # the constructor's; batches then stream within the chunk.
        columns_keep = _chunk_slice(len(to_ids), chunk)
        to_ids = to_ids[columns_keep]
        if points is not None:
            origin_points_all, destination_points_all = points
            points = (origin_points_all, destination_points_all[columns_keep])
    if points is None:
        destination_frame = None
    else:
        _, destination_points = points
        destination_frame = _point_frame(to_ids, destination_points)
    if resolved_percentiles is None:
        columns = ["from_id", "to_id", "travel_time"]
    else:
        columns = ["from_id", "to_id"] + [
            f"travel_time_p{percentile:g}" for percentile in resolved_percentiles
        ]
    if multi:
        at = columns.index("to_id") + 1
        columns[at:at] = (["slot"] if labeled else []) + [
            "arrival_time" if arrive_by else "departure_time"
        ]
    parameters = {
        "date": date,
        "departure": departure,
        "slots": None if not multi else [list(slot) for slot in slots],
        "time_axis": "arrival" if arrive_by else "departure",
        "max_transfers": max_transfers,
        "window": window,
        "percentiles": resolved_percentiles,
        "chunk": None if chunk is None else list(chunk),
        "router": router,
        "exclude_routes": exclude_routes,
        "exclude_trips": exclude_trips,
        "exclude_stops": exclude_stops,
        "walking_speed_kmph": walking_speed_kmph,
        "max_walking_time": max_walking_time,
        "max_snap_distance": max_snap_distance,
        "output_time_units": output_time_units,
    }

    def make_moment(rows, shared_from, shared_to, date, departure):
        if arrive_by:
            return make_arrival_moment(rows, shared_from, shared_to, date, departure)
        if points is None:
            origins_batch = list(from_ids[rows])
            destinations_batch = None
        else:
            origin_points, _ = points
            origins_batch = _point_frame(from_ids[rows], origin_points[rows])
            destinations_batch = destination_frame
        matrix, _, _, batch_percentiles = network._time_matrix_with_ids(
            origins_batch,
            date,
            departure,
            max_transfers,
            destinations=destinations_batch,
            window=window,
            percentiles=resolved_percentiles,
            confidence=None,
            chunk=None,
            router=router,
            exclude_routes=exclude_routes,
            exclude_trips=exclude_trips,
            exclude_stops=exclude_stops,
            walking_speed_kmph=walking_speed_kmph,
            max_walking_time=max_walking_time,
            max_snap_distance=max_snap_distance,
            workers=workers,
            max_memory=max_memory,
        )
        if batch_percentiles != resolved_percentiles:
            raise ValueError(
                "a batch resolved different percentiles than the frozen "
                "query; the stream never re-resolves"
            )
        unreachable = np.iinfo(np.uint32).max
        if resolved_percentiles is None:
            cell_rows, cell_columns = np.nonzero(matrix != unreachable)
            values = {
                "travel_time": pyarrow.array(
                    travel_time_output(
                        matrix[cell_rows, cell_columns], output_time_units
                    )
                )
            }
        else:
            cell_rows, cell_columns = np.nonzero((matrix != unreachable).any(axis=2))
            spread = matrix[cell_rows, cell_columns, :].astype(float)
            spread[spread == unreachable] = np.nan
            values = {
                f"travel_time_p{percentile:g}": pyarrow.array(
                    travel_time_output(spread[:, index], output_time_units)
                )
                for index, percentile in enumerate(resolved_percentiles)
            }
        return pyarrow.table(
            {
                "from_id": pyarrow.DictionaryArray.from_arrays(
                    pyarrow.array(cell_rows + rows.start if rows.start else cell_rows),
                    shared_from,
                ),
                "to_id": pyarrow.DictionaryArray.from_arrays(
                    pyarrow.array(cell_columns), shared_to
                ),
                **values,
            }
        )

    def make_arrival_moment(rows, shared_from, shared_to, date, departure):
        unreachable = np.iinfo(np.uint32).max
        if points is None:
            if resolved_percentiles is None:
                matrix = network._core._arrive_by_time_matrix(
                    list(from_ids),
                    list(to_ids[rows]),
                    date,
                    departure,
                    max_transfers,
                    exclude_routes,
                    exclude_trips,
                    exclude_stops,
                    workers=_memory.width_or(workers),
                )
            else:
                matrix = network._core._arrive_by_time_percentiles(
                    list(from_ids),
                    list(to_ids[rows]),
                    date,
                    departure,
                    window,
                    resolved_percentiles,
                    max_transfers,
                    exclude_routes,
                    exclude_trips,
                    exclude_stops,
                    workers=_memory.width_or(workers),
                )
        else:
            origin_points, destination_points = points
            matrix, _, _, batch_percentiles = network._time_matrix_with_ids(
                _point_frame(from_ids, origin_points),
                date,
                departure,
                max_transfers,
                destinations=_point_frame(to_ids[rows], destination_points[rows]),
                window=window,
                percentiles=resolved_percentiles,
                confidence=None,
                chunk=None,
                router=router,
                exclude_routes=exclude_routes,
                exclude_trips=exclude_trips,
                exclude_stops=exclude_stops,
                walking_speed_kmph=walking_speed_kmph,
                max_walking_time=max_walking_time,
                max_snap_distance=max_snap_distance,
                arrive_by=True,
                workers=workers,
                max_memory=max_memory,
            )
            if batch_percentiles != resolved_percentiles:
                raise ValueError(
                    "a batch resolved different percentiles than the frozen "
                    "query; the stream never re-resolves"
                )
        matrix = np.asarray(matrix)
        if resolved_percentiles is None:
            cell_rows, cell_columns = np.nonzero(matrix != unreachable)
            values = {
                "travel_time": pyarrow.array(
                    travel_time_output(
                        matrix[cell_rows, cell_columns], output_time_units
                    )
                )
            }
        else:
            cell_rows, cell_columns = np.nonzero((matrix != unreachable).any(axis=2))
            spread = matrix[cell_rows, cell_columns, :].astype(float)
            spread[spread == unreachable] = np.nan
            values = {
                f"travel_time_p{percentile:g}": pyarrow.array(
                    travel_time_output(spread[:, index], output_time_units)
                )
                for index, percentile in enumerate(resolved_percentiles)
            }
        return pyarrow.table(
            {
                "from_id": pyarrow.DictionaryArray.from_arrays(
                    pyarrow.array(cell_rows), shared_from
                ),
                "to_id": pyarrow.DictionaryArray.from_arrays(
                    pyarrow.array(
                        cell_columns + rows.start if rows.start else cell_columns
                    ),
                    shared_to,
                ),
                **values,
            }
        )

    make_batch = _slot_batches(make_moment, slots, labeled, multi, arrive_by, pyarrow)
    return _stream_run(
        operation,
        network,
        columns,
        parameters,
        from_ids,
        to_ids,
        points,
        output,
        size,
        make_batch,
        pyarrow,
        resume=resume,
        slice_axis="to" if arrive_by else "from",
        manifest_extra=({"slots": [list(slot) for slot in slots]} if multi else None),
    )


def _point_frame(ids, coordinates):
    """A frozen point GeoDataFrame rebuilt from resolved ``(lat, lon)``
    pairs — per-batch inputs the caller's mutable frames cannot touch."""
    import geopandas

    return geopandas.GeoDataFrame(
        {"id": list(ids)},
        geometry=geopandas.points_from_xy(
            [longitude for _, longitude in coordinates],
            [latitude for latitude, _ in coordinates],
        ),
        crs="EPSG:4326",
    )


def _stream_size(batch_size, resume):
    """The validated batch size of a streaming call."""
    from cafein import _streaming

    if batch_size is None:
        return _streaming.DEFAULT_BATCH_SIZE
    size = operator.index(batch_size)
    if size < 1:
        raise ValueError("batch_size must be >= 1")
    return size


def _arrow_table(
    table,
    from_dictionary,
    to_dictionary,
    offset,
    fares,
    geometries,
    pa,
    output_time_units,
    to_offset=0,
    exposure=None,
):
    """One batch's Arrow table over the shared id dictionary domains.

    ``offset`` shifts the batch-relative origin indices into the shared
    ``from_dictionary`` domain; ``to_offset`` does the same for
    destination indices where they are batch-relative (the arrive-by
    point batches), and is zero where they already span their domain.
    """
    from cafein._units import travel_time_output

    origin_indices = table["from"] if not offset else table["from"] + offset
    to_indices = table["to"] if not to_offset else table["to"] + to_offset
    columns = {
        "from_id": pa.DictionaryArray.from_arrays(
            pa.array(origin_indices), from_dictionary
        ),
        "to_id": pa.DictionaryArray.from_arrays(pa.array(to_indices), to_dictionary),
        "travel_time": pa.array(
            travel_time_output(np.asarray(table["travel_time_s"]), output_time_units)
        ),
        "transfers": pa.array(np.maximum(table["rides"], 1) - 1),
        "transit_distance_m": pa.array(table["transit_distance"]),
        "walk_distance_m": pa.array(table["walk_distance"]),
        "emissions": pa.array(table["emissions"]),
    }
    if fares is not None:
        columns["fare"] = pa.array(table["fare"])
    if exposure is not None:
        for name in exposure.column_names():
            columns[name] = pa.array(np.asarray(table[name], dtype=float))
    if geometries:
        columns["geometry"] = pa.array(list(table["geometry"]), type=pa.binary())
    return pa.table(columns)


def _cost_endpoints(network, origins, destinations, chunk):
    """The resolved routing inputs, snapshotted once for streaming:
    ``(from_ids, to_ids, points, to_stops)`` — the ``chunk``-sliced
    origin ids, the full destination id domain, the
    ``(origin_points, destination_points)`` pair on point queries
    (``None`` on stop queries), and the stop path's ordered destination
    selection (``None`` on point queries or when every stop applies).
    Mirrors `_cost_columns`'s own resolution and must stay in step."""
    if _is_point_frame(origins) or _is_point_frame(destinations):
        from_ids, origin_points = _point_list(origins, "origins")
        if destinations is None:
            to_ids, destination_points = from_ids, origin_points
        else:
            to_ids, destination_points = _point_list(destinations, "destinations")
        rows = _chunk_slice(len(from_ids), chunk)
        return from_ids[rows], to_ids, (origin_points[rows], destination_points), None
    stop_ids = [stop for stop, _, _ in network.stops]
    from_ids = (
        list(stop_ids) if origins is None else list(id_sequence("origins", origins))
    )
    to_stops = (
        None
        if destinations is None
        else list(id_sequence("destinations", destinations))
    )
    return from_ids[_chunk_slice(len(from_ids), chunk)], stop_ids, None, to_stops


def _cost_matrix_data(
    network,
    origins,
    destinations,
    date,
    departure,
    *,
    geometries,
    transport_mode,
    max_street_time,
    intersection_delays,
    profile,
    delay_model,
    parking,
    occupancy,
    vehicle_class,
    perspectives,
    costs,
    currency,
    cost_components,
    street_policy,
    exposure,
    fares,
    max_transfers,
    optimize,
    window,
    within,
    factors,
    components,
    candidates,
    bucket,
    router,
    exclude_routes,
    exclude_trips,
    exclude_stops,
    chunk,
    arrive_by,
    walking_speed_kmph,
    max_walking_time,
    max_snap_distance,
    workers,
    arrival,
    _resolved=None,
    _slot_table=None,
    max_memory=None,
):
    """The long-format columns of one moment's transit cost matrix,
    dispatched by street policy exactly as the constructor did."""
    _reject_cost_street_args(
        transport_mode,
        max_street_time,
        intersection_delays,
        profile,
        delay_model,
        parking,
        occupancy,
        vehicle_class,
        perspectives,
        costs,
        currency,
        cost_components,
    )
    if street_policy is not None:
        if exposure is not None:
            raise ValueError(
                "exposure= reporting covers transit, walking, and "
                "StreetNetwork computations; street_policy matrices "
                "gain it with a later stage"
            )
        from cafein.policy import CarParkPolicy

        if isinstance(street_policy, CarParkPolicy):
            # The query's walking knobs stay ACTIVE beside the car
            # plane, unlike under a StreetLegPolicy.
            offending = next(
                (
                    name
                    for name, value in (
                        ("optimize", None if optimize == "time" else optimize),
                        (
                            (
                                "arrival_time_window"
                                if arrive_by
                                else "departure_time_window"
                            ),
                            window,
                        ),
                        ("max_travel_time", within),
                        ("fares", fares),
                        (
                            "candidates",
                            None if candidates == "time" else candidates,
                        ),
                        ("router", None if router == "auto" else router),
                    )
                    if value is not None
                ),
                None,
            )
            if offending is not None:
                raise ValueError(
                    f"CarParkPolicy does not combine with {offending}; "
                    "the policy cost matrix runs the time-fastest "
                    "engine arm and transit fares stay unpriced"
                )
            if id_sequence("exclude_stops", exclude_stops):
                raise ValueError(
                    "CarParkPolicy does not combine with stop "
                    "exclusions in this stage"
                )
            arm = (
                _car_park_arrive_by_cost_columns
                if arrive_by
                else _car_park_cost_columns
            )
            return arm(
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
                walk_options=(
                    walking_speed_kmph,
                    max_walking_time,
                    max_snap_distance,
                ),
                exclude_routes=exclude_routes,
                exclude_trips=exclude_trips,
                workers=workers,
                max_memory=max_memory,
            )
        offending = next(
            (
                name
                for name, value in (
                    ("optimize", None if optimize == "time" else optimize),
                    ("departure_time_window", window),
                    ("max_travel_time", within),
                    ("fares", fares),
                    ("candidates", None if candidates == "time" else candidates),
                    ("router", None if router == "auto" else router),
                    ("walking_speed_kmph", walking_speed_kmph),
                    ("max_walking_time", max_walking_time),
                    ("snap_distance", max_snap_distance),
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
        from cafein.policy import reject_carriage as _reject_carriage

        _reject_carriage(street_policy, "matrix computation")
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
                workers=workers,
                max_memory=max_memory,
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
                workers=workers,
                max_memory=max_memory,
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
        return data
    if exposure is not None:
        router = _validate_transit_exposure(
            exposure,
            network,
            optimize=optimize,
            window=window,
            within=within,
            candidates=candidates,
            arrival=arrival,
            arrive_by=arrive_by,
            router=router,
        )
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
        arrive_by=arrive_by,
        walking_speed_kmph=walking_speed_kmph,
        max_walking_time=max_walking_time,
        max_snap_distance=max_snap_distance,
        exposure=None if exposure is None else _exposure_snapshot(exposure),
        workers=workers,
        max_memory=max_memory,
        _resolved=_resolved,
        _slot_table=_slot_table,
    )
    if exposure is not None:
        # Re-verify the binding after the routing (the itineraries
        # precedent): a street graph replaced mid-computation must
        # not pair fresh edge indices with the stale snapshot.
        exposure._check_network(network)
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
    if exposure is not None:
        for column in exposure.column_names():
            data[column] = table[column]
    if geometries:
        data["geometry"] = shapely.from_wkb(np.array(table["geometry"], dtype=object))
    return data


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
    arrive_by=False,
    exposure=None,
    _resolved=None,
    _slot_table=None,
    workers=None,
    max_memory=None,
):
    """The core's cost arrays plus the origin and destination ids.

    ``_slot_table`` is one slot's precomputed ``(table, from_ids,
    to_ids)`` — the date-grouped core call computes every slot's rows
    at once and each slot returns its own share here.

    ``_resolved`` is the streaming form's frozen snapshot: a
    ``(trip_factors, fare_tables, endpoints)`` triple whose endpoints
    are one origin batch — ``("points", from_ids, origin_points,
    to_ids, destination_points)`` or ``("stops", from_ids, to_stops)``
    — replacing the resolution below so mutable inputs are only ever
    read once (`_cost_endpoints` mirrors it and must stay in step)."""
    if _slot_table is not None:
        return _slot_table
    components = component_selection(components)
    exclusions = (
        list(id_sequence("exclude_routes", exclude_routes)),
        list(id_sequence("exclude_trips", exclude_trips)),
        list(id_sequence("exclude_stops", exclude_stops)),
    )
    from cafein import emissions
    from cafein.fares import ZoneFareStructure
    from cafein.network import _walk_options

    _validate_cost_query(
        date, departure, optimize, window, within, fares, router, arrive_by=arrive_by
    )
    # A zone structure's exact fare search needs a time limit to stay
    # fast; 120 minutes of total travel time is the default cap. The
    # arrive-by axis prices pre-enumerated candidates, so it takes no
    # implicit cap.
    if (
        not arrive_by
        and optimize == "fare"
        and within is None
        and isinstance(fares, ZoneFareStructure)
    ):
        within = 7200
    if candidates not in ("time", "pareto"):
        raise ValueError("candidates must be 'time' or 'pareto'")
    if candidates == "pareto":
        if optimize != "emissions":
            raise ValueError("candidates='pareto' requires optimize='emissions'")
        if _is_point_frame(origins) or _is_point_frame(destinations):
            raise ValueError("pareto candidates require stop origins and destinations")

    if _resolved is not None:
        trip_factors, fare_tables, endpoints = _resolved
    else:
        fare_tables = None if fares is None else fares._flat_tables(network)
        trip_factors = emissions.trip_factors(network, factors, components)
        endpoints = None
    point_query = (
        endpoints[0] == "points"
        if endpoints is not None
        else _is_point_frame(origins) or _is_point_frame(destinations)
    )
    if point_query:
        if endpoints is not None:
            _, from_ids, origin_points, to_ids, destination_points = endpoints
        else:
            from_ids, origin_points = _point_list(origins, "origins")
            if destinations is None:
                to_ids, destination_points = from_ids, origin_points
            else:
                to_ids, destination_points = _point_list(destinations, "destinations")
            if arrive_by:
                # The arrive-by chunk slices the destination axis — the
                # reverse fan-out axis — exactly as the time matrix's.
                keep = _chunk_slice(len(to_ids), chunk)
                to_ids = to_ids[keep]
                destination_points = destination_points[keep]
            else:
                rows = _chunk_slice(len(from_ids), chunk)
                from_ids = from_ids[rows]
                origin_points = origin_points[rows]
        walk = _walk_options(walking_speed_kmph, max_walking_time, max_snap_distance)
        if optimize != "time" and arrive_by:
            table = network._core._arrive_by_least_cost_matrix_from_points(
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
                *exclusions,
                *walk,
                geometries,
                workers=_memory.width_or(workers),
            )
        elif optimize != "time":
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
                workers=_memory.width_or(workers),
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
                pieces=exposure is not None,
                workers=_memory.width_or(workers),
            )
            if exposure is not None:
                table.update(
                    _transit_exposure_columns(
                        network,
                        exposure,
                        table,
                        origin_points=origin_points,
                        destination_points=destination_points,
                        max_snap_distance=max_snap_distance,
                    )
                )
        _warn_unsnapped(table, from_ids, to_ids)
    else:
        stop_ids = [stop for stop, _, _ in network.stops]
        if endpoints is not None:
            _, from_ids, to_stops = endpoints
        else:
            from_ids = (
                list(stop_ids)
                if origins is None
                else list(id_sequence("origins", origins))
            )
            to_stops = (
                None
                if destinations is None
                else list(id_sequence("destinations", destinations))
            )
            if arrive_by:
                # The arrive-by chunk slices the destination axis — the
                # reverse fan-out axis — exactly as the time matrix's.
                if to_stops is None:
                    to_stops = list(stop_ids)
                to_stops = to_stops[_chunk_slice(len(to_stops), chunk)]
            else:
                from_ids = from_ids[_chunk_slice(len(from_ids), chunk)]
        if optimize != "time" and arrive_by:
            table = network._core._arrive_by_least_cost_matrix(
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
                *exclusions,
                geometries,
                workers=_memory.width_or(workers),
            )
        elif optimize != "time":
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
                workers=_memory.width_or(workers),
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
                pieces=exposure is not None,
                workers=_memory.width_or(workers),
            )
            if exposure is not None:
                table.update(
                    _transit_exposure_columns(
                        network,
                        exposure,
                        table,
                        origin_points=None,
                        destination_points=None,
                        max_snap_distance=max_snap_distance,
                    )
                )
        to_ids = stop_ids
    return table, from_ids, to_ids


def _validate_cost_query(
    date, departure, optimize, window, within, fares, router, arrive_by=False
):
    """The cost-matrix argument contract, shared by `_cost_columns` and
    the streaming path (which must validate before claiming outputs)."""
    if date is None or departure is None:
        raise TypeError("TravelCostMatrix requires departure")
    if optimize not in ("time", "emissions", "fare"):
        raise ValueError(
            f"optimize must be 'time', 'emissions', or 'fare', not {optimize!r}"
        )
    if arrive_by and optimize == "time":
        raise ValueError(
            "arrival= requires optimize='emissions' or 'fare'; the time "
            "axis rides TravelTimeMatrix"
        )
    if optimize != "time" and window is None:
        name = "arrival_time_window" if arrive_by else "departure_time_window"
        raise ValueError(f"optimize={optimize!r} requires {name}=")
    if optimize == "time" and not (window is None and within is None):
        raise ValueError(
            "departure_time_window and max_travel_time require "
            "optimize='emissions' or 'fare'"
        )
    if optimize == "fare" and fares is None:
        raise ValueError("optimize='fare' requires a fare structure (fares=)")
    if router not in ("auto", "raptor", "tbtr"):
        raise ValueError("router must be 'auto', 'raptor', or 'tbtr'")


def _chunk_slice(count, chunk):
    """The deterministic contiguous axis block ``chunk = (k, n)``
    selects: chunk ``k`` of ``n`` equal blocks (the last possibly
    shorter), covering the caller's axis disjointly across
    ``k = 0..n-1``."""
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
    """A GeoDataFrame's ids and ``(lat, lon)`` pairs, in EPSG:4326.

    Point frames route from their point geometry. Polygon and
    multipolygon frames route from their centroids: the explicit
    ``centroid_lat``/``centroid_lon`` columns when present (the
    ``cafein.zones`` protocol — always EPSG:4326), otherwise centroids
    computed in the frame's local UTM zone. Mixed geometry is rejected.
    """
    if not _is_point_frame(frame):
        raise TypeError(f"{role} must be a point GeoDataFrame when points are used")
    if "id" not in frame.columns:
        raise ValueError(f"the {role} GeoDataFrame needs an 'id' column")
    ids = [str(identifier) for identifier in frame["id"]]
    kinds = set(frame.geometry.geom_type)
    if kinds <= {"Point"}:
        if frame.crs is not None:
            frame = frame.to_crs("EPSG:4326")
        geometry = frame.geometry
        return ids, list(zip(geometry.y, geometry.x))
    if kinds <= {"Polygon", "MultiPolygon"}:
        return ids, _zone_centroids(frame, role)
    raise ValueError(
        f"the {role} GeoDataFrame must contain only points or only "
        "polygon/multipolygon geometries"
    )


def _zone_centroids(frame, role):
    """A polygon frame's routing coordinates as ``(lat, lon)`` pairs."""
    if {"centroid_lat", "centroid_lon"} <= set(frame.columns):
        return list(
            zip(
                (float(value) for value in frame["centroid_lat"]),
                (float(value) for value in frame["centroid_lon"]),
            )
        )
    if frame.crs is None:
        raise ValueError(
            f"the polygon {role} GeoDataFrame needs a CRS (or explicit "
            "centroid_lat/centroid_lon columns) to compute centroids"
        )
    projected = frame.to_crs(frame.estimate_utm_crs())
    centroids = projected.geometry.centroid.to_crs("EPSG:4326")
    return list(zip(centroids.y, centroids.x))


def _warn_unsnapped(table, from_ids, to_ids, network="the walking network"):
    """Warn about points the routed profile cannot snap, naming a few.

    Snapping is profile-specific — the same coordinate can be on the
    walking network yet off the streets the routed mode may use — so the
    street matrices name the routed mode's network instead of the
    walking default the transit paths keep.
    """
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
            f"{len(missed)} {side} point(s) are off {network} "
            f"and unreachable ({named}{suffix})",
            stacklevel=3,
        )


def _walking_only_policy(policy):
    """Whether the policy grants walking only at one shared budget — such a
    policy is the legacy walking path bit for bit. Distinct access and
    egress walking budgets run over the multimodal graph instead."""
    from cafein import streets as _streets

    if policy.transfers:
        # Shared intermediate transfers change the relaxed transfer set;
        # no walking-only fast path applies.
        return False, None
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
    workers=None,
    max_memory=None,
):
    """The street-policy travel-time matrix columns: per-point reductions
    through the engine fan-out, the direct walking alternative folded in."""
    from cafein import streets as _streets
    from cafein.network import _policy_transfer_mode
    from cafein.policy import reduction_modes, reject_carriage

    reject_carriage(policy, "the travel time matrix")
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
    exclude_routes = id_sequence("exclude_routes", exclude_routes)
    exclude_trips = id_sequence("exclude_trips", exclude_trips)
    exclude_stops = id_sequence("exclude_stops", exclude_stops)
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
    transfer_mode = _policy_transfer_mode(policy)

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
                            transfer_mode=transfer_mode,
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
        transfer_mode=transfer_mode,
        workers=_memory.width_or(workers),
    )
    # Walking directly can beat riding, exactly as in the walking matrix;
    # the alternative runs over the same multimodal graph at the policy's
    # walking access budget.
    # The direct walking alternative always applies — walking needs no
    # vehicle — at the policy's walking access budget when it names one,
    # else the usual door-to-door cutoff.
    from cafein.network import _direct_walking_mode

    access_budgets = (
        policy.access
        if policy.access is not None
        else {"walk": _streets.MAX_ACCESS_EGRESS_TIME}
    )
    egress_budgets = (
        policy.egress
        if policy.egress is not None
        else {"walk": _streets.MAX_ACCESS_EGRESS_TIME}
    )
    direct_mode, walk_budget = _direct_walking_mode(access_budgets, egress_budgets)
    direct, walk_unsnapped_from, walk_unsnapped_to = core._multimodal_direct_matrix(
        list(origin_points),
        list(destination_points),
        direct_mode,
        float(walk_budget),
        workers=_memory.width_or(workers),
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


def _car_park_time_matrix(
    network,
    origins,
    destinations,
    date,
    departure,
    policy,
    max_transfers,
    chunk,
    walk_options,
    exclude_routes=(),
    exclude_trips=(),
    workers=None,
    max_memory=None,
):
    """The dense CarParkPolicy travel-time matrix with its id axes.

    Per origin the composed car-and-walking access table, ordinary
    walking egress, the engine fan-out, and the direct walking
    alternative folded in over the installed walking streets at the
    query's walking knobs. Seconds; the uint32 maximum where
    unreachable. An origin that cannot reach any facility by car keeps
    the route surface's refusal semantics: its cells stay unreachable
    and a warning names it, never a silent walking-only row."""
    import copy

    from cafein.network import _car_park_table, _walk_options

    core = network._core
    network._require_car_side()
    if not _is_point_frame(origins):
        raise ValueError(
            "CarParkPolicy matrices take point-set origins and "
            "destinations; the facilities plane needs coordinates"
        )
    policy = copy.deepcopy(policy)
    exclude_routes = id_sequence("exclude_routes", exclude_routes)
    exclude_trips = id_sequence("exclude_trips", exclude_trips)
    from_ids, origin_points = _point_list(origins, "origins")
    if destinations is None:
        to_ids, destination_points = from_ids, origin_points
    else:
        to_ids, destination_points = _point_list(destinations, "destinations")
    rows_slice = _chunk_slice(len(from_ids), chunk)
    from_ids = from_ids[rows_slice]
    origin_points = origin_points[rows_slice]
    walk_options = _walk_options(*walk_options)
    walking_speed, walk_budget, snap_distance = walk_options
    # One pinned generation for the WHOLE matrix: every origin's
    # composition, the egress linking, and the direct-walk fold must
    # read the same street graph; an install mid-computation refuses.
    generation = core._streets_generation
    access_rows, undriven = [], []
    for index, point in enumerate(origin_points):
        try:
            offsets, _tokens, _walking = _car_park_table(
                core, tuple(point), policy, walk_options, False
            )
        except ValueError as error:
            message = str(error)
            if (
                "no facility is reachable by car" not in message
                and "too far from the car streets" not in message
            ):
                raise
            access_rows.append([])
            undriven.append(index)
            continue
        access_rows.append(offsets)
    egress_linked = core._link_walking_stops(
        list(destination_points), walking_speed, walk_budget, snap_distance
    )
    egress_rows = [
        (
            []
            if linked is None
            else [(stop, int(seconds)) for stop, seconds, _meters in linked]
        )
        for linked in egress_linked
    ]
    matrix = core._time_matrix_with_access(
        access_rows,
        egress_rows,
        date,
        departure,
        max_transfers,
        exclude_routes=list(exclude_routes),
        exclude_trips=list(exclude_trips),
        exclude_stops=[],
        transfer_mode=None,
        workers=_memory.width_or(workers),
    )
    # The direct walking alternative rides the installed walking
    # streets — the same walk the route surface offers beside the car
    # chain. An omitted origin stays omitted.
    walk = core._walk_matrix(
        list(origin_points),
        list(destination_points),
        walking_speed,
        walk_budget,
        snap_distance,
    )
    if core._streets_generation != generation:
        raise RuntimeError(
            "the street network was replaced while the park-and-ride "
            "matrix was being computed; rerun the query"
        )
    if undriven:
        named = ", ".join(str(from_ids[index]) for index in undriven[:5])
        suffix = ", …" if len(undriven) > 5 else ""
        warnings.warn(
            f"{len(undriven)} origin point(s) cannot reach any facility "
            f"by car and are omitted ({named}{suffix})",
            stacklevel=3,
        )
    _warn_unsnapped(
        {
            "unsnapped_to": [
                index for index, linked in enumerate(egress_linked) if linked is None
            ]
        },
        from_ids,
        to_ids,
    )
    undriven_rows = set(undriven)
    unreachable = 2**32 - 1
    dense = np.full((len(from_ids), len(to_ids)), unreachable, dtype=np.uint32)
    for i in range(len(from_ids)):
        for j in range(len(to_ids)):
            best = matrix[i][j]
            if i not in undriven_rows and walk[i][j] is not None:
                seconds = int(walk[i][j][0])
                best = seconds if best is None else min(int(best), seconds)
            if best is not None:
                dense[i, j] = int(best)
    return dense, from_ids, to_ids


def _car_park_time_columns(
    network,
    origins,
    destinations,
    date,
    departure,
    policy,
    max_transfers,
    chunk,
    walk_options,
    exclude_routes=(),
    exclude_trips=(),
    workers=None,
    max_memory=None,
):
    """The CarParkPolicy travel-time matrix in the long shape the
    frame takes: unreachable pairs omitted, as the matrix contract
    promises."""
    dense, from_ids, to_ids = _car_park_time_matrix(
        network,
        origins,
        destinations,
        date,
        departure,
        policy,
        max_transfers,
        chunk,
        walk_options,
        exclude_routes,
        exclude_trips,
        workers=workers,
        max_memory=max_memory,
    )
    unreachable = 2**32 - 1
    data = {"from_id": [], "to_id": [], "travel_time_s": []}
    for i, from_id in enumerate(from_ids):
        for j, to_id in enumerate(to_ids):
            if dense[i, j] == unreachable:
                continue
            data["from_id"].append(from_id)
            data["to_id"].append(to_id)
            data["travel_time_s"].append(int(dense[i, j]))
    return data


def _carriage_time_columns(
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
    workers=None,
    max_memory=None,
):
    """The carriage time-matrix columns: per-point per-plane reductions
    through the possession-state fan-out, the cross-plane egress fold
    per cell, and the direct walking alternative folded in over the
    same multimodal graph."""
    from cafein import streets as _streets
    from cafein.network import _policy_transfer_mode
    from cafein.policy import carriage_plane_modes, carriage_terms

    exclude_routes = id_sequence("exclude_routes", exclude_routes)
    exclude_trips = id_sequence("exclude_trips", exclude_trips)
    exclude_stops = id_sequence("exclude_stops", exclude_stops)
    if any((exclude_routes, exclude_trips, exclude_stops)):
        raise ValueError("take_aboard=True does not combine with exclusions yet")
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
    # Snapshot every policy term before the GIL-releasing searches.
    _, vehicle = carriage_terms(policy)
    unknown_rule = vehicle.unknown_bike_trips
    park = (
        None
        if vehicle.facilities == "any_stop"
        else [str(stop) for stop in vehicle.facilities]
    )
    transfer_mode = _policy_transfer_mode(policy)
    access_planes = carriage_plane_modes(
        policy, "access", _streets.MAX_ACCESS_EGRESS_TIME
    )
    egress_planes = carriage_plane_modes(
        policy, "egress", _streets.MAX_ACCESS_EGRESS_TIME
    )
    walk_budget = (policy.access or {}).get("walk", _streets.MAX_ACCESS_EGRESS_TIME)
    from_ids, origin_points = _point_list(origins, "origins")
    if destinations is None:
        to_ids, destination_points = from_ids, origin_points
    else:
        to_ids, destination_points = _point_list(destinations, "destinations")
    rows_slice = _chunk_slice(len(from_ids), chunk)
    from_ids = from_ids[rows_slice]
    origin_points = origin_points[rows_slice]

    def reduced(points, egress, planes):
        from cafein.network import _carrying_offsets

        carrying_rows, free_rows, unsnapped = [], [], []
        for index, (lat, lon) in enumerate(points):
            rows = []
            for plane, plane_modes in enumerate(planes):
                try:
                    reduction = core._reduced_street_offsets(
                        lat, lon, egress, plane_modes
                    )
                except ValueError as error:
                    if "too far from the multimodal street network" not in str(error):
                        raise
                    rows.append([])
                    continue
                # The Carrying access rows carry the reduction's walk
                # flag; the egress rows and the Free plane never walk
                # again, so they stay plain offsets.
                if plane == 0 and not egress:
                    rows.append(_carrying_offsets(reduction))
                else:
                    rows.append([(stop, seconds) for stop, seconds, *_ in reduction])
            if not any(rows):
                # Neither plane snapped; the point's cells are omitted
                # unless the direct walk below still stands.
                unsnapped.append(index)
            carrying_rows.append(rows[0])
            free_rows.append(rows[1])
        return carrying_rows, free_rows, unsnapped

    carrying_rows, free_rows, unsnapped_origins = reduced(
        origin_points, False, access_planes
    )
    carrying_egress_rows, free_egress_rows, unsnapped_destinations = reduced(
        destination_points, True, egress_planes
    )
    matrix = core._carriage_time_matrix(
        carrying_rows,
        free_rows,
        carrying_egress_rows,
        free_egress_rows,
        date,
        departure,
        max_transfers,
        unknown_rule,
        park_stops=park,
        transfer_mode=transfer_mode,
        workers=_memory.width_or(workers),
    )
    # The direct walking alternative always applies — walking needs no
    # vehicle — at the policy's walking access budget when it names one,
    # else the usual door-to-door cutoff.
    direct, walk_unsnapped_from, walk_unsnapped_to = core._multimodal_direct_matrix(
        list(origin_points),
        list(destination_points),
        "walk",
        float(walk_budget),
        workers=_memory.width_or(workers),
    )
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
    workers=None,
    max_memory=None,
):
    """The street-policy cost matrix columns: per-point meters-carrying
    reductions through the engine fan-out, street distances and emissions
    attributed per row from the winning choices, and the direct walking
    alternative folded in over the same multimodal graph."""
    components = component_selection(components)
    from cafein import emissions
    from cafein import streets as _streets
    from cafein.policy import reduction_modes

    from cafein.network import _policy_transfer_mode
    from cafein.policy import reject_carriage

    reject_carriage(policy, "the cost matrix")
    transfer_mode = _policy_transfer_mode(policy)

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
    exclude_routes = id_sequence("exclude_routes", exclude_routes)
    exclude_trips = id_sequence("exclude_trips", exclude_trips)
    exclude_stops = id_sequence("exclude_stops", exclude_stops)
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
    transfer_arg = None
    if transfer_mode is not None:
        # The rental's ride grams join the emissions column, so the
        # transfer mode's shared-fleet factor must resolve.
        mode, budget = transfer_mode
        value = emissions.street_factor(
            mode, street_factors, components, service_model="shared"
        )
        if pd.isna(value):
            raise ValueError(
                f"the {mode} emission factor is unresolved; the cost "
                "matrix attributes rental transfer emissions, so pass "
                "factors= rows resolving it (see "
                "cafein.emissions.load_street_factors)"
            )
        transfer_arg = (mode, budget, float(value) / 1000.0)
    # One resolved per-km factor per granted vehicle mode; NaN keeps an
    # unresolved factor poisoning rather than zeroing its rows. Walking
    # rides no vehicle, so its factor is never read.
    mode_factors = {"walk": 0.0}
    for mode, _, rental, *_ in access_modes + egress_modes:
        if mode in mode_factors:
            continue
        value = emissions.street_factor(
            mode,
            street_factors,
            components,
            service_model="shared" if rental else None,
        )
        mode_factors[mode] = float("nan") if pd.isna(value) else float(value)

    def reduced(points, egress, modes):
        rows, unsnapped = [], []
        for index, (lat, lon) in enumerate(points):
            try:
                cells = core._reduced_street_rows(
                    lat,
                    lon,
                    egress,
                    modes,
                    exclude_stops=list(exclude_stops),
                    transfer_mode=transfer_mode,
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
                    (
                        stop,
                        seconds,
                        network_m,
                        connector_m,
                        walk_m,
                        mode_factors[mode],
                        transfer_network_m,
                        transfer_total_m,
                        mode != "walk" or transfer_rental,
                    )
                    for (
                        stop,
                        seconds,
                        mode,
                        network_m,
                        connector_m,
                        walk_m,
                        transfer_network_m,
                        transfer_total_m,
                        transfer_rental,
                    ) in cells
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
    from cafein.network import _direct_walking_mode

    access_budgets = (
        policy.access
        if policy.access is not None
        else {"walk": _streets.MAX_ACCESS_EGRESS_TIME}
    )
    egress_budgets = (
        policy.egress
        if policy.egress is not None
        else {"walk": _streets.MAX_ACCESS_EGRESS_TIME}
    )
    direct_mode, walk_budget = _direct_walking_mode(access_budgets, egress_budgets)
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
        transfer_mode=transfer_arg,
        direct_mode=direct_mode,
        workers=_memory.width_or(workers),
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


def _car_park_cost_columns(
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
    walk_options,
    exclude_routes=(),
    exclude_trips=(),
    workers=None,
    max_memory=None,
):
    """The CarParkPolicy cost-matrix columns.

    The composed access rows carry the drive metres and the car's
    resolved per-km factor for the engine's sidecar attribution, the
    walking plane and the ordinary walking egress their walked metres,
    and the winning facility's fee joins per row from the returned
    access stop — a ``fee`` column in the ``cafein.costs`` currency,
    zero on rows the walk won. The direct walking alternative folds in
    over the installed walking streets, as on the route surface.
    Transit fares stay unpriced, as on every policy cost matrix, and
    no facility column is surfaced. An origin that cannot reach any
    facility by car keeps the route surface's refusal semantics: its
    cells are omitted and a warning names it."""
    import copy

    from cafein import emissions
    from cafein.network import _car_park_table, _walk_options

    components = component_selection(components)
    core = network._core
    network._require_car_side()
    if not _is_point_frame(origins):
        raise ValueError(
            "CarParkPolicy matrices take point-set origins and "
            "destinations; the facilities plane needs coordinates"
        )
    policy = copy.deepcopy(policy)
    exclude_routes = id_sequence("exclude_routes", exclude_routes)
    exclude_trips = id_sequence("exclude_trips", exclude_trips)
    from_ids, origin_points = _point_list(origins, "origins")
    if destinations is None:
        to_ids, destination_points = from_ids, origin_points
    else:
        to_ids, destination_points = _point_list(destinations, "destinations")
    rows_slice = _chunk_slice(len(from_ids), chunk)
    from_ids = from_ids[rows_slice]
    origin_points = origin_points[rows_slice]
    walk_options = _walk_options(*walk_options)
    walking_speed, walk_budget, snap_distance = walk_options
    transit_factors, street_factors = _factor_tables(factors)
    trip_factors = emissions.trip_factors(network, transit_factors, components)
    # The drive prices as the car — the policy's class row divided by
    # its occupancy; NaN keeps an unresolved factor poisoning rather
    # than zeroing its rows.
    value = emissions.street_factor(
        "car", street_factors, components, vehicle_class=policy.vehicle_class
    )
    car_factor = (
        float("nan") if pd.isna(value) else float(value) / float(policy.occupancy)
    )
    fees = [float(fee) for fee in policy.facilities["fee"]]
    # One pinned generation for the WHOLE matrix: every origin's
    # composition, the egress linking, and the direct-walk fold must
    # read the same street graph; an install mid-computation refuses.
    generation = core._streets_generation
    access_rows, fee_by_stop, undriven = [], [], []
    for index, point in enumerate(origin_points):
        try:
            offsets, tokens, walking = _car_park_table(
                core, tuple(point), policy, walk_options, False
            )
        except ValueError as error:
            message = str(error)
            if (
                "no facility is reachable by car" not in message
                and "too far from the car streets" not in message
            ):
                raise
            access_rows.append([])
            fee_by_stop.append({})
            undriven.append(index)
            continue
        row, stop_fees = [], {}
        for stop, total in offsets:
            token = tokens.get(stop)
            if token is None:
                row.append(
                    (stop, total, 0.0, 0.0, walking[stop][1], 0.0, 0.0, 0.0, False)
                )
                continue
            position, _d, _p, _w, network_m, connector_m, walk_m, _shape = token
            row.append(
                (
                    stop,
                    total,
                    network_m,
                    connector_m,
                    walk_m,
                    car_factor,
                    0.0,
                    0.0,
                    True,
                )
            )
            stop_fees[stop] = fees[position]
        access_rows.append(row)
        fee_by_stop.append(stop_fees)
    egress_linked = core._link_walking_stops(
        list(destination_points), walking_speed, walk_budget, snap_distance
    )
    egress_rows = [
        (
            []
            if linked is None
            else [
                (stop, int(seconds), 0.0, 0.0, meters, 0.0, 0.0, 0.0, False)
                for stop, seconds, meters in linked
            ]
        )
        for linked in egress_linked
    ]
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
        exclude_stops=[],
        geometries=bool(geometries),
        transfer_mode=None,
        direct_mode=None,
        workers=_memory.width_or(workers),
    )
    no_stop = 2**32 - 1
    columns = {
        "from": [int(v) for v in table["from"]],
        "to": [int(v) for v in table["to"]],
        "travel_time_s": [int(v) for v in table["travel_time_s"]],
        "rides": [int(v) for v in table["rides"]],
        "transit_distance": [float(v) for v in table["transit_distance"]],
        "walk_distance": [float(v) for v in table["walk_distance"]],
        "street_distance": [float(v) for v in table["street_distance"]],
        "emissions": [float(v) for v in table["emissions"]],
        "access_stop": [int(v) for v in table["access_stop"]],
    }
    shapes = list(table["geometry"]) if geometries else None
    stops_by_index = [stop for stop, _lat, _lon in core.stops]
    # The direct walking alternative, folded per cell exactly as the
    # engine folds the multimodal one: a strictly faster walk wins, an
    # equal-time walk beats a ridden or driven row, and pairs only
    # walking reaches gain walking-only rows.
    position = {
        (i, j): at for at, (i, j) in enumerate(zip(columns["from"], columns["to"]))
    }
    walk = core._walk_matrix(
        list(origin_points),
        list(destination_points),
        walking_speed,
        walk_budget,
        snap_distance,
    )
    undriven_rows = set(undriven)

    def walk_shape(i, j):
        if not geometries:
            return None
        direct = core._walk_between(
            origin_points[i][0],
            origin_points[i][1],
            destination_points[j][0],
            destination_points[j][1],
            snap_distance,
            True,
        )
        if direct is None or direct[2] is None:
            return None
        # The cost rows' geometry contract is MultiLineString; the
        # rebuilt walk arrives as a bare LineString.
        geometry = shapely.from_wkb(direct[2])
        if geometry.geom_type == "LineString":
            geometry = shapely.MultiLineString([geometry])
        return shapely.to_wkb(geometry)

    for i in range(len(from_ids)):
        if i in undriven_rows:
            continue
        for j in range(len(to_ids)):
            cell = walk[i][j]
            if cell is None:
                continue
            seconds, meters = int(cell[0]), float(cell[1])
            at = position.get((i, j))
            if at is None:
                columns["from"].append(i)
                columns["to"].append(j)
                columns["travel_time_s"].append(seconds)
                columns["rides"].append(0)
                columns["transit_distance"].append(0.0)
                columns["walk_distance"].append(meters)
                columns["street_distance"].append(0.0)
                columns["emissions"].append(0.0)
                columns["access_stop"].append(no_stop)
                if shapes is not None:
                    shapes.append(walk_shape(i, j))
                continue
            stop_index = columns["access_stop"][at]
            car_used = (
                stop_index != no_stop and stops_by_index[stop_index] in fee_by_stop[i]
            )
            if seconds < columns["travel_time_s"][at] or (
                seconds == columns["travel_time_s"][at]
                and (columns["rides"][at] > 0 or car_used)
            ):
                columns["travel_time_s"][at] = seconds
                columns["rides"][at] = 0
                columns["transit_distance"][at] = 0.0
                columns["walk_distance"][at] = meters
                columns["street_distance"][at] = 0.0
                columns["emissions"][at] = 0.0
                columns["access_stop"][at] = no_stop
                if shapes is not None:
                    shapes[at] = walk_shape(i, j)
    if core._streets_generation != generation:
        raise RuntimeError(
            "the street network was replaced while the park-and-ride "
            "matrix was being computed; rerun the query"
        )
    if undriven:
        named = ", ".join(str(from_ids[index]) for index in undriven[:5])
        suffix = ", …" if len(undriven) > 5 else ""
        warnings.warn(
            f"{len(undriven)} origin point(s) cannot reach any facility "
            f"by car and are omitted ({named}{suffix})",
            stacklevel=3,
        )
    _warn_unsnapped(
        {
            "unsnapped_to": [
                index for index, linked in enumerate(egress_linked) if linked is None
            ]
        },
        from_ids,
        to_ids,
    )
    fee_column = [
        (
            fee_by_stop[i].get(stops_by_index[stop_index], 0.0)
            if stop_index != no_stop
            else 0.0
        )
        for i, stop_index in zip(columns["from"], columns["access_stop"])
    ]
    order = sorted(
        range(len(columns["from"])),
        key=lambda at: (columns["from"][at], columns["to"][at]),
    )
    data = {
        "from_id": np.array(from_ids, dtype=object)[
            [columns["from"][at] for at in order]
        ],
        "to_id": np.array(to_ids, dtype=object)[[columns["to"][at] for at in order]],
        "travel_time_s": [columns["travel_time_s"][at] for at in order],
        "transfers": np.maximum([columns["rides"][at] for at in order], 1) - 1,
        "transit_distance_m": [columns["transit_distance"][at] for at in order],
        "walk_distance_m": [columns["walk_distance"][at] for at in order],
        "street_distance_m": [columns["street_distance"][at] for at in order],
        "emissions": [columns["emissions"][at] for at in order],
        "fee": [fee_column[at] for at in order],
    }
    if geometries:
        data["geometry"] = shapely.from_wkb(
            np.array([shapes[at] for at in order], dtype=object)
        )
    return data


def _arrive_by_clock(seconds):
    """A seconds-of-day value back in the ``HH:MM:SS`` engine form."""
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def _car_park_arrive_by_winner(held, candidate):
    """The complete-journey order between arrive-by candidates
    ``(departure, rounds, achieved)``: latest departure, then fewest
    rounds, then earliest arrival — the reverse engine's own rule."""
    if held is None:
        return True
    return (
        candidate[0] > held[0]
        or (candidate[0] == held[0] and candidate[1] < held[1])
        or (
            candidate[0] == held[0]
            and candidate[1] == held[1]
            and candidate[2] < held[2]
        )
    )


def _car_park_arrive_by_time_matrix(
    network,
    origins,
    destinations,
    date,
    deadline,
    policy,
    max_transfers,
    chunk,
    walk_options,
    exclude_routes=(),
    exclude_trips=(),
    workers=None,
    max_memory=None,
):
    """The dense CarParkPolicy arrive-by matrices with their id axes.

    One reverse run per destination at the deadline over the composed
    access tables; per cell the complete-journey winner's own duration,
    departure clock, and winning access-stop index (the uint32 maximum
    where unreachable or where the walk won), the direct walking
    alternative folded in by the same latest-departure rule — placed to
    arrive exactly at the deadline. Chunking slices the destination
    axis, the arrive-by convention. An origin that cannot reach any
    facility by car keeps its cells unreachable, with a warning naming
    it. Returns ``(durations, departures, access_stops, from_ids,
    to_ids)``."""
    import copy

    from cafein.network import _car_park_table, _departure_seconds, _walk_options

    core = network._core
    network._require_car_side()
    if not _is_point_frame(origins):
        raise ValueError(
            "CarParkPolicy matrices take point-set origins and "
            "destinations; the facilities plane needs coordinates"
        )
    policy = copy.deepcopy(policy)
    exclude_routes = id_sequence("exclude_routes", exclude_routes)
    exclude_trips = id_sequence("exclude_trips", exclude_trips)
    from_ids, origin_points = _point_list(origins, "origins")
    if destinations is None:
        to_ids, destination_points = from_ids, origin_points
    else:
        to_ids, destination_points = _point_list(destinations, "destinations")
    # One reverse run fills a column, so chunking slices the
    # destination axis — the arrive-by convention.
    columns = _chunk_slice(len(to_ids), chunk)
    to_ids = to_ids[columns]
    destination_points = destination_points[columns]
    walk_options = _walk_options(*walk_options)
    walking_speed, walk_budget, snap_distance = walk_options
    # One pinned generation for the whole matrix, as on the departure
    # axis: composition, egress linking, and the walk fold must read
    # the same street graph.
    generation = core._streets_generation
    access_rows, undriven = [], []
    for index, point in enumerate(origin_points):
        try:
            offsets, _tokens, _walking = _car_park_table(
                core, tuple(point), policy, walk_options, False
            )
        except ValueError as error:
            message = str(error)
            if (
                "no facility is reachable by car" not in message
                and "too far from the car streets" not in message
            ):
                raise
            access_rows.append([])
            undriven.append(index)
            continue
        access_rows.append(offsets)
    egress_linked = core._link_walking_stops(
        list(destination_points), walking_speed, walk_budget, snap_distance
    )
    egress_rows = [
        (
            []
            if linked is None
            else [(stop, int(seconds)) for stop, seconds, _meters in linked]
        )
        for linked in egress_linked
    ]
    table = core._arrive_by_time_matrix_with_access(
        access_rows,
        egress_rows,
        date,
        deadline,
        max_transfers,
        exclude_routes=list(exclude_routes),
        exclude_trips=list(exclude_trips),
        exclude_stops=[],
        workers=_memory.width_or(workers),
    )
    walk = core._walk_matrix(
        list(origin_points),
        list(destination_points),
        walking_speed,
        walk_budget,
        snap_distance,
    )
    if core._streets_generation != generation:
        raise RuntimeError(
            "the street network was replaced while the park-and-ride "
            "matrix was being computed; rerun the query"
        )
    if undriven:
        named = ", ".join(str(from_ids[index]) for index in undriven[:5])
        suffix = ", …" if len(undriven) > 5 else ""
        warnings.warn(
            f"{len(undriven)} origin point(s) cannot reach any facility "
            f"by car and are omitted ({named}{suffix})",
            stacklevel=3,
        )
    _warn_unsnapped(
        {
            "unsnapped_to": [
                index for index, linked in enumerate(egress_linked) if linked is None
            ]
        },
        from_ids,
        to_ids,
    )
    unreachable = 2**32 - 1
    durations = np.array(table["matrix"], dtype=np.int64)
    departures = np.array(table["departure"], dtype=np.int64)
    rounds = np.array(table["rounds"], dtype=np.int64)
    access_stops = np.array(table["access_stop"], dtype=np.int64)
    deadline_s = _departure_seconds(deadline)
    undriven_rows = set(undriven)
    for i in range(len(from_ids)):
        if i in undriven_rows:
            continue
        for j in range(len(to_ids)):
            cell = walk[i][j]
            if cell is None:
                continue
            walk_seconds = int(cell[0])
            placed = deadline_s - walk_seconds
            if placed < 0:
                continue
            held = (
                None
                if durations[i, j] == unreachable
                else (
                    int(departures[i, j]),
                    int(rounds[i, j]),
                    int(departures[i, j] + durations[i, j]),
                )
            )
            # The walk wins exact ties, as on the route surface.
            walk_candidate = (placed, 0, deadline_s)
            if held is None or not _car_park_arrive_by_winner(walk_candidate, held):
                durations[i, j] = walk_seconds
                departures[i, j] = placed
                rounds[i, j] = 0
                access_stops[i, j] = unreachable
    return durations.astype(np.uint32), departures, access_stops, from_ids, to_ids


def _car_park_arrive_by_time_columns(
    network,
    origins,
    destinations,
    date,
    deadline,
    policy,
    max_transfers,
    chunk,
    walk_options,
    exclude_routes=(),
    exclude_trips=(),
    workers=None,
    max_memory=None,
):
    """The CarParkPolicy arrive-by travel-time matrix in the long
    shape the frame takes: unreachable pairs omitted."""
    durations, _departures, _stops, from_ids, to_ids = _car_park_arrive_by_time_matrix(
        network,
        origins,
        destinations,
        date,
        deadline,
        policy,
        max_transfers,
        chunk,
        walk_options,
        exclude_routes,
        exclude_trips,
        workers=workers,
        max_memory=max_memory,
    )
    unreachable = 2**32 - 1
    data = {"from_id": [], "to_id": [], "travel_time_s": []}
    for i, from_id in enumerate(from_ids):
        for j, to_id in enumerate(to_ids):
            if durations[i, j] == unreachable:
                continue
            data["from_id"].append(from_id)
            data["to_id"].append(to_id)
            data["travel_time_s"].append(int(durations[i, j]))
    return data


def _car_park_arrive_by_cost_columns(
    network,
    origins,
    destinations,
    date,
    deadline,
    policy,
    *,
    max_transfers,
    factors,
    components,
    geometries,
    chunk,
    walk_options,
    exclude_routes=(),
    exclude_trips=(),
    workers=None,
    max_memory=None,
):
    """The CarParkPolicy arrive-by cost-matrix columns.

    The reverse election picks each cell's latest departure and its
    winning access stop; the cell then prices with a forward run at
    that departure seeded with exactly that access row, so the
    attribution — drive metres, grams, and the ``fee`` — belongs to the
    facility the election chose. Walking-won cells price as walks, the
    walk placed to arrive exactly at the deadline. Chunking slices the
    destination axis. Transit fares stay unpriced and no facility
    column is surfaced, as on the departure axis."""
    import copy

    from cafein import emissions
    from cafein.network import _car_park_table, _departure_seconds, _walk_options

    components = component_selection(components)
    core = network._core
    network._require_car_side()
    if not _is_point_frame(origins):
        raise ValueError(
            "CarParkPolicy matrices take point-set origins and "
            "destinations; the facilities plane needs coordinates"
        )
    policy = copy.deepcopy(policy)
    exclude_routes = id_sequence("exclude_routes", exclude_routes)
    exclude_trips = id_sequence("exclude_trips", exclude_trips)
    from_ids, origin_points = _point_list(origins, "origins")
    if destinations is None:
        to_ids, destination_points = from_ids, origin_points
    else:
        to_ids, destination_points = _point_list(destinations, "destinations")
    columns_slice = _chunk_slice(len(to_ids), chunk)
    to_ids = to_ids[columns_slice]
    destination_points = destination_points[columns_slice]
    walk_options = _walk_options(*walk_options)
    walking_speed, walk_budget, snap_distance = walk_options
    transit_factors, street_factors = _factor_tables(factors)
    trip_factors = emissions.trip_factors(network, transit_factors, components)
    value = emissions.street_factor(
        "car", street_factors, components, vehicle_class=policy.vehicle_class
    )
    car_factor = (
        float("nan") if pd.isna(value) else float(value) / float(policy.occupancy)
    )
    fees = [float(fee) for fee in policy.facilities["fee"]]
    generation = core._streets_generation
    # The election and the per-group repricing are separate engine
    # calls; pinning the transfer generation proves they all read the
    # same closure.
    transfers_generation = core._transfers_generation
    access_offsets, rows_by_stop, fee_by_stop, undriven = [], [], [], []
    for index, point in enumerate(origin_points):
        try:
            offsets, tokens, walking = _car_park_table(
                core, tuple(point), policy, walk_options, False
            )
        except ValueError as error:
            message = str(error)
            if (
                "no facility is reachable by car" not in message
                and "too far from the car streets" not in message
            ):
                raise
            access_offsets.append([])
            rows_by_stop.append({})
            fee_by_stop.append({})
            undriven.append(index)
            continue
        by_stop, stop_fees = {}, {}
        for stop, total in offsets:
            token = tokens.get(stop)
            if token is None:
                by_stop[stop] = (
                    stop,
                    total,
                    0.0,
                    0.0,
                    walking[stop][1],
                    0.0,
                    0.0,
                    0.0,
                    False,
                )
                continue
            position, _d, _p, _w, network_m, connector_m, walk_m, _shape = token
            by_stop[stop] = (
                stop,
                total,
                network_m,
                connector_m,
                walk_m,
                car_factor,
                0.0,
                0.0,
                True,
            )
            stop_fees[stop] = fees[position]
        access_offsets.append(offsets)
        rows_by_stop.append(by_stop)
        fee_by_stop.append(stop_fees)
    egress_linked = core._link_walking_stops(
        list(destination_points), walking_speed, walk_budget, snap_distance
    )
    egress_offsets = [
        (
            []
            if linked is None
            else [(stop, int(seconds)) for stop, seconds, _meters in linked]
        )
        for linked in egress_linked
    ]
    egress_rows = [
        (
            []
            if linked is None
            else [
                (stop, int(seconds), 0.0, 0.0, meters, 0.0, 0.0, 0.0, False)
                for stop, seconds, meters in linked
            ]
        )
        for linked in egress_linked
    ]
    table = core._arrive_by_time_matrix_with_access(
        access_offsets,
        egress_offsets,
        date,
        deadline,
        max_transfers,
        exclude_routes=list(exclude_routes),
        exclude_trips=list(exclude_trips),
        exclude_stops=[],
        workers=_memory.width_or(workers),
    )
    walk = core._walk_matrix(
        list(origin_points),
        list(destination_points),
        walking_speed,
        walk_budget,
        snap_distance,
    )
    if undriven:
        named = ", ".join(str(from_ids[index]) for index in undriven[:5])
        suffix = ", …" if len(undriven) > 5 else ""
        warnings.warn(
            f"{len(undriven)} origin point(s) cannot reach any facility "
            f"by car and are omitted ({named}{suffix})",
            stacklevel=3,
        )
    _warn_unsnapped(
        {
            "unsnapped_to": [
                index for index, linked in enumerate(egress_linked) if linked is None
            ]
        },
        from_ids,
        to_ids,
    )
    unreachable = 2**32 - 1
    durations = np.array(table["matrix"], dtype=np.int64)
    departures_at = np.array(table["departure"], dtype=np.int64)
    rounds = np.array(table["rounds"], dtype=np.int64)
    elected_stops = np.array(table["access_stop"], dtype=np.int64)
    deadline_s = _departure_seconds(deadline)
    stops_by_index = [stop for stop, _lat, _lon in core.stops]
    undriven_rows = set(undriven)
    # Per cell: the walking alternative competes by the same
    # latest-departure rule; surviving ridden cells group by their
    # elected (origin, departure, access stop, rides) for the forward
    # pricing, and zero-ride through-stop cells synthesize directly —
    # the forward engine never emits them.
    walk_cells, through_cells, groups = [], [], {}
    for i in range(len(from_ids)):
        if i in undriven_rows:
            continue
        for j in range(len(to_ids)):
            transit = durations[i, j] != unreachable
            cell = walk[i][j]
            placed = None if cell is None else deadline_s - int(cell[0])
            if placed is not None and placed >= 0:
                held = (
                    None
                    if not transit
                    else (
                        int(departures_at[i, j]),
                        int(rounds[i, j]),
                        int(departures_at[i, j] + durations[i, j]),
                    )
                )
                # The walk wins exact ties, as on the route surface.
                walk_candidate = (placed, 0, deadline_s)
                if held is None or not _car_park_arrive_by_winner(walk_candidate, held):
                    walk_cells.append((i, j, placed, int(cell[0]), float(cell[1])))
                    continue
            if not transit:
                continue
            if rounds[i, j] == 0:
                through_cells.append((i, j))
                continue
            key = (
                i,
                int(departures_at[i, j]),
                int(elected_stops[i, j]),
                int(rounds[i, j]),
            )
            groups.setdefault(key, []).append(j)
    columns = {
        "from": [],
        "to": [],
        "travel_time_s": [],
        "rides": [],
        "transit_distance": [],
        "walk_distance": [],
        "street_distance": [],
        "emissions": [],
        "fee": [],
    }
    shapes = [] if geometries else None
    for (i, departure_s, stop_index, rides), cells in sorted(groups.items()):
        stop = stops_by_index[stop_index]
        seed = rows_by_stop[i][stop]
        fee = fee_by_stop[i].get(stop, 0.0)
        # Capped at the elected ride count, the forward time-best
        # through the elected seed IS the complete-journey winner: a
        # faster same-departure chain with more rides cannot displace
        # it, so the cell provably prices the elected journey.
        priced = core._cost_matrix_with_access(
            [[seed]],
            [egress_rows[j] for j in cells],
            [tuple(origin_points[i])],
            [tuple(destination_points[j]) for j in cells],
            date,
            _arrive_by_clock(departure_s),
            trip_factors,
            float(walk_budget),
            rides - 1,
            exclude_routes=list(exclude_routes),
            exclude_trips=list(exclude_trips),
            exclude_stops=[],
            geometries=bool(geometries),
            transfer_mode=None,
            direct_mode=None,
            workers=_memory.width_or(workers),
        )
        found = set()
        for at in range(len(priced["travel_time_s"])):
            j = cells[int(priced["to"][at])]
            found.add(j)
            if (
                int(priced["travel_time_s"][at]) != int(durations[i, j])
                or int(priced["rides"][at]) != rides
            ):
                raise RuntimeError(
                    "an elected arrive-by cell re-priced as a different "
                    "journey; rerun the query"
                )
            columns["from"].append(i)
            columns["to"].append(j)
            columns["travel_time_s"].append(int(priced["travel_time_s"][at]))
            columns["rides"].append(int(priced["rides"][at]))
            columns["transit_distance"].append(float(priced["transit_distance"][at]))
            columns["walk_distance"].append(float(priced["walk_distance"][at]))
            columns["street_distance"].append(float(priced["street_distance"][at]))
            columns["emissions"].append(float(priced["emissions"][at]))
            columns["fee"].append(fee)
            if shapes is not None:
                shapes.append(priced["geometry"][at])
        missing = [j for j in cells if j not in found]
        if missing:
            # The election proved these cells reachable through this
            # very access row; an empty forward answer means the two
            # engines drifted apart.
            raise RuntimeError(
                "an elected arrive-by cell could not be re-priced by "
                "the forward engine; rerun the query"
            )
    for i, j in through_cells:
        # The through-stop zero-ride cell: the composed chain to the
        # elected stop, then the egress walk straight out — priced
        # from the same sidecars the composition carries.
        stop = stops_by_index[int(elected_stops[i, j])]
        row = rows_by_stop[i][stop]
        _stop, _s, network_m, connector_m, walk_m, factor, _tn, _tt, used = row
        egress_m = next(
            (meters for s, _sec, meters in (egress_linked[j] or []) if s == stop),
            0.0,
        )
        grams = network_m / 1000.0 * factor if used and network_m > 0.0 else 0.0
        columns["from"].append(i)
        columns["to"].append(j)
        columns["travel_time_s"].append(int(durations[i, j]))
        columns["rides"].append(0)
        columns["transit_distance"].append(0.0)
        columns["walk_distance"].append(walk_m + egress_m)
        columns["street_distance"].append(network_m + connector_m if used else 0.0)
        columns["emissions"].append(grams)
        columns["fee"].append(fee_by_stop[i].get(stop, 0.0))
        if shapes is not None:
            shapes.append(None)

    def walk_shape(i, j):
        if not geometries:
            return None
        direct = core._walk_between(
            origin_points[i][0],
            origin_points[i][1],
            destination_points[j][0],
            destination_points[j][1],
            snap_distance,
            True,
        )
        if direct is None or direct[2] is None:
            return None
        geometry = shapely.from_wkb(direct[2])
        if geometry.geom_type == "LineString":
            geometry = shapely.MultiLineString([geometry])
        return shapely.to_wkb(geometry)

    for i, j, _placed, walk_seconds, meters in walk_cells:
        columns["from"].append(i)
        columns["to"].append(j)
        columns["travel_time_s"].append(walk_seconds)
        columns["rides"].append(0)
        columns["transit_distance"].append(0.0)
        columns["walk_distance"].append(meters)
        columns["street_distance"].append(0.0)
        columns["emissions"].append(0.0)
        columns["fee"].append(0.0)
        if shapes is not None:
            shapes.append(walk_shape(i, j))
    if core._streets_generation != generation:
        raise RuntimeError(
            "the street network was replaced while the park-and-ride "
            "matrix was being computed; rerun the query"
        )
    if core._transfers_generation != transfers_generation:
        raise RuntimeError(
            "the transfer set was replaced while the park-and-ride "
            "matrix was being computed; rerun the query"
        )
    order = sorted(
        range(len(columns["from"])),
        key=lambda at: (columns["from"][at], columns["to"][at]),
    )
    data = {
        "from_id": np.array(from_ids, dtype=object)[
            [columns["from"][at] for at in order]
        ],
        "to_id": np.array(to_ids, dtype=object)[[columns["to"][at] for at in order]],
        "travel_time_s": [columns["travel_time_s"][at] for at in order],
        "transfers": np.maximum([columns["rides"][at] for at in order], 1) - 1,
        "transit_distance_m": [columns["transit_distance"][at] for at in order],
        "walk_distance_m": [columns["walk_distance"][at] for at in order],
        "street_distance_m": [columns["street_distance"][at] for at in order],
        "emissions": [columns["emissions"][at] for at in order],
        "fee": [columns["fee"][at] for at in order],
    }
    if geometries:
        data["geometry"] = shapely.from_wkb(
            np.array([shapes[at] for at in order], dtype=object)
        )
    return data
