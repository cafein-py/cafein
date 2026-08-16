"""Matrix computers over a transport network."""

import operator
import warnings

import numpy as np
import pandas as pd
import shapely

from cafein._validate import component_selection, id_sequence, sequence_not_string
from cafein.travelers import (
    folded_constraints,
    folded_street_policy,
    refuse_wheelchair_streets,
)


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
    departure : datetime.datetime or str
        Departure at every origin — a datetime, or an ISO string like
        ``"2022-02-22 08:30"``; the service date is its date part.
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
    chunk : (int, int) (optional)
        Compute only origin chunk ``k`` of ``n``: a deterministic
        contiguous block of the resolved origins, so ``n`` batch jobs
        cover all origins disjointly and their shards concatenate.
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

    ``output_time_units=`` selects the ``travel_time`` unit:
    ``"minutes"`` (the default; whole minutes rounded to the nearest)
    or ``"seconds"`` (the engine's exact values).
    """

    @property
    def _constructor(self):
        return pd.DataFrame

    def __init__(
        self,
        network,
        origins=None,
        destinations=None,
        departure=None,
        *,
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
        output_time_units="minutes",
    ):
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
        origins = sequence_not_string("origins", origins)
        destinations = sequence_not_string("destinations", destinations)
        from cafein._units import (
            departure_parts,
            duration_seconds,
            validated_output_time_units,
        )

        output_time_units = validated_output_time_units(output_time_units)
        date, departure = (
            (None, None) if departure is None else departure_parts(departure)
        )
        if max_rides < 1:
            raise ValueError("max_rides must be at least 1")
        max_transfers = max_rides - 1
        window = duration_seconds("departure_time_window", departure_time_window)
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
                transit_only={
                    "departure": departure,
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
            )
            super().__init__(
                pd.DataFrame(_humanize_time_columns(data, output_time_units))
            )
            return
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
            super().__init__(
                pd.DataFrame(_humanize_time_columns(data, output_time_units))
            )
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
        super().__init__(pd.DataFrame(_humanize_time_columns(data, output_time_units)))

    @classmethod
    def to_parquet(
        cls,
        network,
        origins=None,
        destinations=None,
        departure=None,
        *,
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
        output_time_units="minutes",
    ):
        """The cost matrix streamed to Parquet — the constructor's
        semantics with `travel_cost_table`'s ``output=`` behavior.

        Origins are processed in ``batch_size`` slices (default 500)
        and each batch is written as it completes, so peak memory holds
        one batch — never the whole constructor result. ``output=``
        selects the form by suffix exactly as ``travel_cost_table``
        does and the return value is a :class:`cafein.StreamingResult`.
        ``from_id``/``to_id`` are dictionary-encoded over the shared id
        domains; a street matrix's geometry streams as plain WKB
        binary. ``street_policy`` matrices do not stream yet and are
        rejected. ``resume=True`` continues a matching partial
        directory run exactly as ``travel_cost_table`` does.
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
            street_policy, max_walking_time = folded_street_policy(
                traveler, network, street_policy, walking_speed_kmph, max_walking_time
            )
        import pyarrow

        from cafein._units import (
            departure_parts,
            duration_seconds,
            validated_output_time_units,
        )

        output_time_units = validated_output_time_units(output_time_units)
        date, departure = (
            (None, None) if departure is None else departure_parts(departure)
        )
        if max_rides < 1:
            raise ValueError("max_rides must be at least 1")
        max_transfers = max_rides - 1
        window = duration_seconds("departure_time_window", departure_time_window)
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
        return _stream_transit_cost(
            "TravelCostMatrix.to_parquet",
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
            output=output,
            size=size,
            pyarrow=pyarrow,
            resume=resume,
            output_time_units=output_time_units,
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

    With ``departure_time_window``, every minute mark within the
    window is profiled and the ``travel_time`` column is replaced by
    one ``travel_time_p<p>`` column per requested percentile (the
    median by default, or ``confidence`` for the symmetric interval
    plus the median), floating-point in the output units so an
    unreachable percentile reads as ``NaN``; a pair appears when at
    least one of its percentiles is reachable.

    Origins are either stop identifiers or a point GeoDataFrame with an
    ``id`` column; destinations apply to point origins only — stop
    origins always span every stop (the ``stops`` order). Points are
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
    departure : datetime.datetime or str (optional)
        Departure at every origin — a datetime, or an ISO string like
        ``"2022-02-22 08:30"``; the service date is its date part.
        Give exactly one of ``departure`` and ``arrival``.
    arrival : datetime.datetime or str (optional)
        Arrival deadline at every destination, in the same forms. Each
        row's ``travel_time`` is that pair's latest-departure journey
        arriving by the deadline (fewest rides, then earliest arrival,
        breaking ties) — the journey's own duration, identical to
        ``route_between_stops(arrival=)``. One reverse run serves each
        **destination**, so ``chunk`` slices the destination axis and
        chunked frames still concatenate to full coverage. The reverse
        rides the closure (a whole-day ULTRA set is never claimed);
        the departure-window parameters, ``router="tbtr"``, and
        ``street_policy`` do not combine with it, and a
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
    """

    @property
    def _constructor(self):
        return pd.DataFrame

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
    ):
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
        origins = sequence_not_string("origins", origins)
        destinations = sequence_not_string("destinations", destinations)
        from cafein._units import (
            departure_parts,
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
            from cafein._units import arrival_parts

            date, departure = arrival_parts(arrival)
            if router == "tbtr":
                raise ValueError(
                    "router='tbtr' does not serve arrival=; the reverse "
                    "search rides RAPTOR"
                )
            if street_policy is not None:
                raise ValueError(
                    "street_policy= (a traveler's street bridge included) "
                    "does not combine with arrival= yet"
                )
        else:
            date, departure = (
                (None, None) if departure is None else departure_parts(departure)
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
            )
            super().__init__(
                pd.DataFrame(_humanize_time_columns(data, output_time_units))
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
            from cafein.policy import carriage_terms as _carriage_terms

            if _carriage_terms(street_policy) is not None:
                data = _carriage_time_columns(
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
                super().__init__(
                    pd.DataFrame(_humanize_time_columns(data, output_time_units))
                )
                return
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
                super().__init__(
                    pd.DataFrame(_humanize_time_columns(data, output_time_units))
                )
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
            super().__init__(
                pd.DataFrame(_humanize_time_columns(data, output_time_units))
            )
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
            arrive_by=arrive_by,
        )
        super().__init__(pd.DataFrame(_humanize_time_columns(data, output_time_units)))

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
    ):
        """The travel-time matrix streamed to Parquet — the
        constructor's semantics with ``travel_cost_table``'s
        ``output=`` behavior.

        Origins are processed in ``batch_size`` slices (default 500)
        and each batch is written as it completes, so peak memory holds
        one batch — never the whole constructor result. ``output=``
        selects the form by suffix exactly as ``travel_cost_table``
        does and the return value is a :class:`cafein.StreamingResult`.
        ``from_id``/``to_id`` are dictionary-encoded over the shared id
        domains; a windowed matrix streams its percentile columns.
        ``street_policy`` matrices do not stream yet and are rejected.
        ``resume=True`` continues a matching partial directory run
        exactly as ``travel_cost_table`` does.
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
            street_policy, max_walking_time = folded_street_policy(
                traveler, network, street_policy, walking_speed_kmph, max_walking_time
            )
        import pyarrow

        from cafein._units import (
            departure_parts,
            duration_seconds,
            validated_output_time_units,
        )

        output_time_units = validated_output_time_units(output_time_units)
        date, departure = (
            (None, None) if departure is None else departure_parts(departure)
        )
        if max_rides < 1:
            raise ValueError("max_rides must be at least 1")
        max_transfers = max_rides - 1
        window = duration_seconds("departure_time_window", departure_time_window)
        max_walking_time = duration_seconds("max_walking_time", max_walking_time)
        max_street_time = duration_seconds("max_street_time", max_street_time)
        max_snap_distance = snap_distance
        size = _stream_size(batch_size, resume)
        if arrival is not None or arrival_time_window is not None:
            raise NotImplementedError(
                "arrive-by matrices do not stream yet; compute the frame "
                "with the constructor (chunk= slices the destination axis "
                "for batch jobs)"
            )
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
            date,
            departure,
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
        )


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
    )
    query = resolved["query"]
    from_index, to_index, numeric, wkb = _street_cost_cells(
        network, query, geometries=geometries, resolved=resolved
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
    # The frame orders provenance before the cost block, as documented.
    order = ["from_id", "to_id", "travel_time_s", "distance_m"]
    order += ["network_distance_m", "connector_distance_m", "distance_provenance"]
    order += ["emissions"]
    order += [name for name in data if name not in order]
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
):
    """Every result-affecting street-cost input resolved exactly once —
    validation first, then the frozen snapshot the cells (and the
    streaming form's batches) compute from."""
    from cafein import _parking, costs as _costs, emissions
    from cafein.street_network import _resolved_delays

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
    }


def _street_cost_cells(network, query, *, geometries, resolved):
    """One street-cost batch: origin/destination indices, the numeric
    columns, and the WKB geometries (``None`` without ``geometries``)."""
    transport_mode = resolved["transport_mode"]
    table = network._core.cost_matrix(
        query.origin_points,
        query.destination_points,
        transport_mode,
        query.max_seconds,
        query.max_snap_distance,
        bool(geometries),
        car_model=resolved["car_model"],
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
    rows, columns, travel_time_s = _street_time_cells(network, query, resolved)
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


def _street_time_cells(network, query, resolved):
    """One street-time batch: cell indices and their travel times."""
    transport_mode = resolved["transport_mode"]
    table = network._core.travel_time_matrix(
        query.origin_points,
        query.destination_points,
        transport_mode,
        query.max_seconds,
        query.max_snap_distance,
        car_model=resolved["car_model"],
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
):
    """The reachable cells of the travel-time matrix, in long format."""
    if date is None or departure is None:
        raise TypeError("TravelTimeMatrix requires departure or arrival")
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
    departure=None,
    *,
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
    output=None,
    batch_size=None,
    resume=False,
    output_time_units="minutes",
):
    """The travel-cost matrix as a pyarrow Table — the shard-writing form.

    Semantics and parameters follow `TravelCostMatrix` — including the
    windowed optimize modes with their
    ``departure_time_window``/``max_travel_time``, the ``fares``
    pricing, and the ``router`` engine choice, though always
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

    With ``output=`` the matrix **streams to disk** instead of
    materialising: origins are processed in ``batch_size`` slices
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
    order), with ``from_id``/``to_id`` dictionary-encoded over the same
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
    from cafein._units import (
        departure_parts,
        duration_seconds,
        validated_output_time_units,
    )

    output_time_units = validated_output_time_units(output_time_units)
    date, departure = (None, None) if departure is None else departure_parts(departure)
    if max_rides < 1:
        raise ValueError("max_rides must be at least 1")
    max_transfers = max_rides - 1
    window = duration_seconds("departure_time_window", departure_time_window)
    within = duration_seconds("max_travel_time", max_travel_time)
    max_walking_time = duration_seconds("max_walking_time", max_walking_time)
    max_snap_distance = snap_distance
    if output is None:
        if batch_size is not None:
            raise ValueError("batch_size requires output=")
        if resume:
            raise ValueError("resume=True requires output=")
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
        return _arrow_table(
            table,
            pyarrow.array(from_ids, type=pyarrow.string()),
            pyarrow.array(to_ids, type=pyarrow.string()),
            0,
            fares,
            geometries,
            pyarrow,
            output_time_units,
        )
    size = _stream_size(batch_size, resume)
    return _stream_transit_cost(
        "travel_cost_table",
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
        candidates="time",
        bucket=25.0,
        router=router,
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
    )


def _stream_transit_cost(
    operation,
    network,
    origins,
    destinations,
    date,
    departure,
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
    resume=False,
    output_time_units="minutes",
):
    """The transit cost matrix streamed in origin batches — shared by
    ``travel_cost_table`` and ``TravelCostMatrix.to_parquet``."""
    # Everything result-affecting resolves and freezes ONCE, before the
    # output is claimed: one-shot iterables drain here, later mutation of
    # the input frames cannot desynchronise batches from the fingerprint,
    # and an invalid query never leaves an empty claimed output behind.
    _validate_cost_query(date, departure, optimize, window, within, fares, router)
    if candidates not in ("time", "pareto"):
        raise ValueError("candidates must be 'time' or 'pareto'")
    exclude_routes = list(id_sequence("exclude_routes", exclude_routes))
    exclude_trips = list(id_sequence("exclude_trips", exclude_trips))
    exclude_stops = list(id_sequence("exclude_stops", exclude_stops))
    if chunk is not None:
        chunk = tuple(int(part) for part in chunk)
    from_ids, to_ids, points, to_stops = _cost_endpoints(
        network, origins, destinations, chunk
    )
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
    if geometries:
        columns.append("geometry")
    parameters = {
        "date": date,
        "departure": departure,
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
        "output_time_units": output_time_units,
    }

    def make_batch(rows, shared_from, shared_to):
        if points is None:
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
            walking_speed_kmph=walking_speed_kmph,
            max_walking_time=max_walking_time,
            max_snap_distance=max_snap_distance,
            _resolved=(trip_factors, fare_tables, endpoints),
        )
        return _arrow_table(
            table,
            shared_from,
            shared_to,
            rows.start,
            fares,
            geometries,
            pyarrow,
            output_time_units,
        )

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
    )


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
):
    """The shared streaming driver: fingerprint, claim, batch, write.

    ``make_batch(rows, shared_from, shared_to)`` returns one batch's
    Arrow table for the origin slice ``rows`` — inputs must already be
    frozen by the caller. Used by ``travel_cost_table`` and the matrix
    computers' ``to_parquet`` classmethods identically. With
    ``resume=True`` (directory form) the completed shards of a matching
    partial run are skipped — their batches never compute — and only
    the remainder routes.
    """
    from cafein import _streaming

    shared_from = pyarrow.array(from_ids, type=pyarrow.string())
    shared_to = pyarrow.array(to_ids, type=pyarrow.string())
    count = len(from_ids)
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
        manifest, completed = _streaming.prepare_resume(path, fingerprint, size, count)

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
        "origin_count": count,
    }
    shared = {"from_id": shared_from, "to_id": shared_to}
    return _streaming.write_stream(
        mode,
        path,
        produce(),
        manifest_seed,
        {name: shared[name] for name in dictionary_columns},
        manifest=manifest,
    )


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
):
    """The street cost matrix streamed in origin batches over a frozen
    resolution — `TravelCostMatrix.to_parquet`'s street arm."""
    from cafein._cafein import STREET_DISTANCE_PROVENANCE
    from cafein._units import travel_time_output

    query = resolved["query"]
    account = resolved["account"]
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
            network, batch, geometries=geometries, resolved=resolved
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
            network, batch, resolved
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
    date,
    departure,
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
):
    """The transit time matrix streamed in origin batches —
    `TravelTimeMatrix.to_parquet`'s transit arm."""
    from cafein._units import travel_time_output
    from cafein.network import _window_percentiles

    if date is None or departure is None:
        raise TypeError("TravelTimeMatrix requires departure")
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
    from_ids, to_ids, points, _ = _cost_endpoints(network, origins, destinations, chunk)
    if points is None and destinations is not None:
        raise ValueError("destinations apply to point origins")
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
    parameters = {
        "date": date,
        "departure": departure,
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

    def make_batch(rows, shared_from, shared_to):
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
):
    """One batch's Arrow table over the shared id dictionary domains.

    ``offset`` shifts the batch-relative origin indices into the shared
    ``from_dictionary`` domain; destination indices already span theirs.
    """
    from cafein._units import travel_time_output

    origin_indices = table["from"] if not offset else table["from"] + offset
    columns = {
        "from_id": pa.DictionaryArray.from_arrays(
            pa.array(origin_indices), from_dictionary
        ),
        "to_id": pa.DictionaryArray.from_arrays(pa.array(table["to"]), to_dictionary),
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
    _resolved=None,
):
    """The core's cost arrays plus the origin and destination ids.

    ``_resolved`` is the streaming form's frozen snapshot: a
    ``(trip_factors, fare_tables, endpoints)`` triple whose endpoints
    are one origin batch — ``("points", from_ids, origin_points,
    to_ids, destination_points)`` or ``("stops", from_ids, to_stops)``
    — replacing the resolution below so mutable inputs are only ever
    read once (`_cost_endpoints` mirrors it and must stay in step)."""
    components = component_selection(components)
    exclusions = (
        list(id_sequence("exclude_routes", exclude_routes)),
        list(id_sequence("exclude_trips", exclude_trips)),
        list(id_sequence("exclude_stops", exclude_stops)),
    )
    from cafein import emissions
    from cafein.fares import ZoneFareStructure
    from cafein.network import _walk_options

    _validate_cost_query(date, departure, optimize, window, within, fares, router)
    # A zone structure's exact fare search needs a time limit to stay
    # fast; 120 minutes of total travel time is the default cap.
    if optimize == "fare" and within is None and isinstance(fares, ZoneFareStructure):
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
        if endpoints is not None:
            _, from_ids, to_stops = endpoints
        else:
            from_ids = (
                list(stop_ids)
                if origins is None
                else list(id_sequence("origins", origins))
            )
            from_ids = from_ids[_chunk_slice(len(from_ids), chunk)]
            to_stops = (
                None
                if destinations is None
                else list(id_sequence("destinations", destinations))
            )
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


def _validate_cost_query(date, departure, optimize, window, within, fares, router):
    """The cost-matrix argument contract, shared by `_cost_columns` and
    the streaming path (which must validate before claiming outputs)."""
    if date is None or departure is None:
        raise TypeError("TravelCostMatrix requires departure")
    if optimize not in ("time", "emissions", "fare"):
        raise ValueError(
            f"optimize must be 'time', 'emissions', or 'fare', not {optimize!r}"
        )
    if optimize != "time" and window is None:
        raise ValueError(f"optimize={optimize!r} requires departure_time_window=")
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
        list(origin_points), list(destination_points), direct_mode, float(walk_budget)
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
    )
    # The direct walking alternative always applies — walking needs no
    # vehicle — at the policy's walking access budget when it names one,
    # else the usual door-to-door cutoff.
    direct, walk_unsnapped_from, walk_unsnapped_to = core._multimodal_direct_matrix(
        list(origin_points), list(destination_points), "walk", float(walk_budget)
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
