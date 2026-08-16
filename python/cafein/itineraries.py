"""Detailed door-to-door itineraries as a GeoDataFrame."""

import math
from cafein._validate import component_selection, id_sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from cafein.travelers import folded_constraints, folded_street_policy
from cafein.matrices import (
    _factor_tables,
    _is_point_frame,
    _is_street_network,
    _point_list,
    _street_query,
    _warn_unsnapped,
)

_COLUMNS = [
    "from_id",
    "to_id",
    "option",
    "segment",
    "leg_type",
    "departure_s",
    "arrival_s",
    "travel_time_s",
    "from_stop",
    "to_stop",
    "trip_id",
    "route_id",
    "route_short_name",
    "distance_m",
    "distance_provenance",
    "emissions",
    "geometry",
]

# A street-policy frame adds the leg's street mode beside its structural
# position and the rebuilt legs' distance parts beside the total; the
# legacy schema is otherwise unchanged.
_POLICY_COLUMNS = list(_COLUMNS)
_POLICY_COLUMNS.insert(_POLICY_COLUMNS.index("departure_s"), "mode")
_POLICY_COLUMNS.insert(_POLICY_COLUMNS.index("distance_m") + 1, "network_distance_m")
_POLICY_COLUMNS.insert(
    _POLICY_COLUMNS.index("network_distance_m") + 1, "connector_distance_m"
)
_POLICY_COLUMNS.insert(_POLICY_COLUMNS.index("emissions") + 1, "bike_aboard")


class DetailedItineraries(gpd.GeoDataFrame):
    """Full journeys between origins and destinations, one row per leg.

    A GeoDataFrame with one row per leg of every alternative journey
    between each origin and each destination — the time-optimal
    (arrival, rides) set by default, the (arrival, emissions) set with
    ``candidates="pareto"``, that set widened to nearby suboptimal
    journeys with ``candidates="relaxed"``, or distinct-corridor options
    with ``candidates="diverse"``: ``from_id`` and ``to_id``
    (the OD pair), ``option`` (the journey alternative, numbered per OD
    pair), ``segment`` (the leg's position in that journey), and the leg
    itself — ``leg_type`` (``access``, ``transit``, ``transfer``,
    ``egress``, ``walk`` for a walking-only door-to-door journey, or
    ``park`` for a carried vehicle left at a stop — a zero-length
    event under a carriage street policy),
    ``departure_s`` and ``arrival_s`` (clock seconds past the service
    day's start) and ``travel_time`` (whole minutes rounded to the
    nearest by default; exact seconds with
    ``output_time_units="seconds"``), ``from_stop`` and ``to_stop`` (the boarding and alighting
    stops; ``None`` at the walked ends of a door-to-door journey),
    ``trip_id``/``route_id``/``route_short_name`` on transit legs,
    ``distance_m`` (meters) and its ``distance_provenance``, ``emissions``
    (grams CO₂e; ``0`` on walks, ``NaN`` where a ridden trip has no
    matching factor), with ``fares=`` a ``fare`` column — the option's
    whole-journey ticket price, repeated on each of its rows — and
    ``geometry`` — the leg's shape in EPSG:4326,
    a transit polyline or a walked street path, absent where a leg has
    none. Group by ``["from_id", "to_id", "option"]`` to recover whole
    journeys.

    Origins and destinations are either stop identifiers or
    GeoDataFrames with an ``id`` column — point frames, or polygon
    frames routed from their centroids
    (``centroid_lat``/``centroid_lon`` columns when present — the
    ``cafein.zones`` protocol — otherwise local-UTM centroids) — and
    both must be the same kind.
    Stops route with :meth:`TransportNetwork.route_between_stops`;
    points route door-to-door with
    :meth:`TransportNetwork.route_between_coordinates` and need a network
    built with an OSM extract (``osm_pbf=``). Every origin is routed to
    every destination — one search per OD pair — so this detailed mode
    suits focused origin and destination sets, not full matrices.

    Requires a network built with trip distances (the default), and with
    leg geometries for ``geometries=True``. Slices, copies, and other
    pandas operations return ordinary GeoDataFrame views that no longer
    re-route.

    Given a ``StreetNetwork`` instead, each reachable pair is a single
    street leg under ``transport_mode``, bounded by ``max_street_time``:
    ``option`` and ``segment`` are ``0``, ``mode`` names the mode and
    ``leg_type`` repeats it (a direct non-walking leg takes its mode as
    its structural position, as a direct walk already does), and the
    distance columns carry their unit in the name — ``distance_m`` with
    its ``network_distance_m`` and ``connector_distance_m`` parts — plus
    ``distance_provenance``. A street
    network has no timetable, so the departure and arrival columns are
    null unless ``departure`` or ``arrival`` is given purely to place
    the leg on a clock — a departure anchors the start, an arrival
    deadline the end (a bare ``"HH:MM"`` or ``datetime.time`` works
    there) — and
    the timetable-only arguments are rejected, while
    ``factors=`` and ``components=`` configure the mode's emission factor
    (``emissions`` is NA where it is unresolved, never a silent zero).

    Parameters
    ----------
    network : TransportNetwork or StreetNetwork
        The network to route on. A ``StreetNetwork`` takes the standalone
        street path and requires ``transport_mode``.
    origins : list of str, or GeoDataFrame
        Origin stop_ids, or points with an ``id`` column. A street
        network needs points. Polygon frames route from their centroids
        (``centroid_lat``/``centroid_lon`` columns when present — the
        ``cafein.zones`` protocol — otherwise local-UTM centroids).
    destinations : list of str, or GeoDataFrame
        Destination stop_ids, or points with an ``id`` column; the same
        kind as `origins`.
    departure : datetime.datetime or str (optional)
        Departure at every origin — a datetime, or an ISO string like
        ``"2022-02-22 08:30"``; the service date is its date part.
        Give exactly one of ``departure`` and ``arrival``.
    arrival : datetime.datetime or str (optional)
        Arrival deadline at every destination, in the same forms —
        ``candidates="time"`` only. Each pair's rows hold the
        latest-departure Pareto set's full legs, options ordered
        latest departure first, every journey identical to the
        ``departure=`` answer for the departure it discovers. As
        always, each row's ``travel_time`` is its own leg's duration —
        a journey's duration spans its first departure to its final
        arrival, never the span to the deadline. The
        departure-window parameters, ``router="tbtr"``, and
        ``street_policy`` do not combine with it. On a
        ``StreetNetwork`` the deadline is pure clock placement:
        durations are unchanged and every leg's ``arrival_s`` is the
        deadline.
    max_rides : int (optional, default: 8)
        Maximum number of boarded vehicles per journey (rides, not
        transfers: 8 rides allow 7 transfers).
    factors : DataFrame or path (optional)
        Extra emission-factor rows layered over the shipped defaults;
        see ``cafein.emissions.load_factors`` — or, for a
        ``StreetNetwork``, street-mode rows for
        ``cafein.emissions.load_street_factors``.
    components : list of str (optional)
        The life-cycle components to include (default: all four); see
        ``cafein.emissions.annotate``.
    fares : FareStructure or ZoneFareStructure (optional)
        A fare model (see ``cafein.fares``); adds a ``fare`` column.
        A fare prices the whole journey (tickets span legs), so every
        row of an option repeats the option's fare — group by
        ``["from_id", "to_id", "option"]`` and take ``first()`` for
        per-journey fares. ``NaN`` where the structure cannot price a
        journey, ``0.0`` on a walking-only one. Pricing is exact for
        each returned journey and never changes which journeys are
        searched or returned. Rental legs under a ``street_policy``
        (its ``source="shared"`` modes) price from the structure's
        ``street`` tariffs, as in ``cafein.fares.annotate_fares``.
        A ``StreetNetwork`` rejects it with the other timetable-only
        arguments.
    candidates : {"time", "pareto", "relaxed", "diverse"} (default: "time")
        Which alternatives to return per OD pair. ``"time"`` draws the
        time-optimal (arrival, rides) journeys of the RAPTOR engine;
        ``"pareto"`` draws the (arrival, emissions) journeys of the
        McRAPTOR engine — the cleaner-but-slower alternatives the
        time-optimal set misses — at the single given departure;
        ``"relaxed"`` widens the ``"pareto"`` set by a
        ``tolerance_minutes`` band in the per-stop dominance;
        ``"diverse"`` returns
        ``max_options`` distinct alternatives by iterative route
        penalization — by default (``penalty="ban"``) banning each chosen
        corridor's routes so the options ride disjoint line sets, or with a
        numeric ``penalty`` making a used route costly but still usable so
        corridors may share a trunk.
    bucket : float (optional, default: 25.0)
        The emissions bucket width in grams CO₂e for the ``"pareto"``
        search's arrival tie-break; smaller keeps finer emission
        differences apart. Ignored for ``candidates="time"``.
    router : {"auto", "raptor", "tbtr"} (optional, default: "auto")
        The engine backing ``candidates="pareto"``: multicriteria RAPTOR,
        or trip-based (``"tbtr"``), for stop-id and point origins and
        destinations alike. ``"auto"`` (the default) runs on McTBTR when
        a cached transfer set (``compute_mctbtr_transfers``) matches the
        query's date and factors, else on McRAPTOR. ``"tbtr"`` requires
        ``candidates="pareto"``; ``"relaxed"`` and ``"diverse"`` require
        ``"raptor"`` (``"auto"`` resolves to it).
    tolerance_minutes : float or datetime.timedelta (optional, default: None)
        The time-tolerance band in minutes. For ``candidates="relaxed"``
        a journey is kept even when a cleaner or simpler one dominates
        it, as long as that dominator is not more than this much
        earlier (``0`` reproduces ``candidates="pareto"``) — the same
        suboptimal-arrival tolerance as r5py's ``suboptimalMinutes``,
        here at the single given departure (``journey_frontier`` applies
        it across a ``departure_time_window``, the r5py-equivalent
        profile). For ``candidates="diverse"`` a positive value widens
        each penalization round's pool to that relaxed frontier
        (relaxed × diverse). ``None`` takes the per-family default — 5
        minutes for ``"relaxed"`` (r5py's ``suboptimalMinutes``), ``0``
        for ``"diverse"``. Unused for ``"time"`` and ``"pareto"``.
    max_options : int (optional, default: None)
        For ``candidates="relaxed"``, a cap on the suboptimal alternatives
        kept per OD pair — the frontier is always returned and the nearest
        suboptimal journeys fill the rest, ``None`` keeping every journey
        within the tolerance. For ``candidates="diverse"``, the number of
        distinct-corridor alternatives per OD pair (``None`` defaults to
        3); fewer are returned when the disjoint corridors run out.
    diversity : str (optional, default: "time")
        The objective for ``candidates="diverse"``: ``"time"`` picks the
        fastest journey each penalization round (cleaner as tie-break),
        biasing the options toward the fast end of the trade-off;
        ``"spread"`` seeds on the fastest, then each later round picks the
        journey farthest from the already-chosen corridors in the
        normalized (travel_time, emissions) plane, so the options span the
        trade-off. Unused for the other candidate sets.
    penalty : str or float (optional, default: "ban")
        How ``candidates="diverse"`` steers each round off the corridors
        already chosen. ``"ban"`` (default) hard-bans every route a chosen
        corridor rode, so the options ride fully route-disjoint line sets;
        a positive number instead adds that many seconds to a chosen
        route's effective arrival per prior use, so a corridor that mostly
        differs yet shares a trunk can surface (the R5-style soft penalty).
        Unused for the other candidate sets.
    exclude_routes, exclude_trips, exclude_stops : list of str (optional)
        GTFS ids of supply the itineraries must not use — disruption
        and accessibility filters, as in ``route_between_stops``.
        Excluded stops refuse boarding, alighting, transfers, and
        access/egress while vehicles still ride through them; excluded
        origins or destinations yield no rows.
    traveler : TravelerProfile (optional)
        One traveler's constraint profile (``cafein.TravelerProfile``):
        its compiled exclusions union the ``exclude_*`` lists, and its
        walking knobs fill the unset walking arguments — a knob set on
        both the call and the profile is rejected.
    geometries : bool (optional, default: True)
        Attach each leg's geometry. Turn off to skip the geometry work
        when only the leg records are needed.
    walking_speed_kmph, max_walking_time, snap_distance : float
        The street-search options for point origins/destinations, as in
        ``TransportNetwork.route_between_coordinates``: speed in km/h,
        walking time in minutes (or a timedelta), snap distance in
        meters; only valid with points. Only ``snap_distance`` applies
        to a ``StreetNetwork``, whose speeds come from the mode's
        profile.
    transport_mode : str (optional)
        The mode to route. Required for a ``StreetNetwork``, where it is
        one of ``"walk"``, ``"bicycle"``, ``"e_bike"``, ``"e_scooter"``,
        ``"car"`` (a car build). A ``TransportNetwork`` routes public
        transport and takes none.
    max_street_time : float or datetime.timedelta (optional)
        Cutoff in minutes for a ``StreetNetwork``, beyond which a
        destination counts as unreachable (default: 120 minutes,
        ``cafein.street_network.MAX_STREET_TIME`` seconds).
    intersection_delays, profile, delay_model : optional
        Car legs only. Free-flow by default;
        ``intersection_delays=True`` applies the intersection-delay
        model under ``profile=`` (``"rush"``, ``"midday"`` — the
        default — or ``"day-average"``) with ``delay_model=`` partial
        overrides, as in ``StreetNetwork.travel_time``.
    parking : optional
        Car legs only, off by default. The parking search ending each
        leg (``True`` → 300 s / 0 m, a number → seconds, a
        ``(seconds, metres)`` pair, or a per-area polygon GeoDataFrame),
        as in ``StreetNetwork.travel_time``: the seconds join the leg's
        travel time and arrival, the metres its driven distance and
        emissions basis; the leg geometry never shows the search loop.
    occupancy, vehicle_class : optional
        Car legs only. ``vehicle_class=`` selects the powertrain of the
        shipped per-vehicle-km car factors (GEMMAT Table 4; default
        ``"ICE"``) and ``occupancy=`` (at least 1, default 1) divides
        the per-vehicle emissions across the persons carried, as in
        ``TravelCostMatrix``.
    perspectives, costs, currency, cost_components : optional
        The monetary cost account of the street legs, exactly as in
        ``TravelCostMatrix``: off by default, ``cost_<perspective>``
        columns over the driven kilometres from the Gössling et al.
        (2019) defaults with ``costs=`` overrides, per-component
        columns via ``cost_components=`` (one perspective), and the
        declared ``currency`` label column.
    street_policy : StreetLegPolicy (optional)
        Which street modes may serve the access and egress, on what
        vehicle terms (``cafein.StreetLegPolicy``); point origins and
        destinations only, with ``candidates="time"``, ``"pareto"``, or
        ``"relaxed"`` (``"diverse"`` arrives with a later stage). A
        ``transfers={mode: budget}`` grant rides the merged mode-transfer
        set of ``TransportNetwork.compute_mode_transfers`` — time
        candidates only for now — and a transfer whose edge rode a
        rental splits into its walk--ride--walk legs. Under
        the multicriteria candidates each journey end reduces to its
        (seconds, grams) Pareto frontier and the street grams enter the
        McRAPTOR dominance itself, so the options genuinely trade street
        emissions against time — including zero-ride street
        compositions on the frontier. Every granted vehicle mode then
        needs a resolved emission factor (``factors=`` rows keyed by
        ``street_mode``); an unresolved factor is rejected, never
        silently zeroed. The frame gains a
        ``mode`` column beside ``leg_type`` — the street mode of every
        rebuilt access/egress/transfer/walk leg, ``None`` on transit
        legs — and the rebuilt street legs carry their exact distances
        (the ``network_distance_m`` and ``connector_distance_m`` parts
        beside the ``distance_m`` total; null on other legs), the street
        distance provenance, their shapes, and the mode's
        street emissions over its network meters (a ``factors=`` table
        keyed by ``street_mode`` configures the street ladder — see
        ``cafein.emissions.load_street_factors`` — while any other table
        layers over the transit ladder as always; NA where unresolved).
        A choice the transfer closure carried splits into the vehicle
        leg to its seed stop
        plus the walked transfer, so no row blends two modes. A carried
        vehicle (``take_aboard=True``, time candidates) additionally
        adds a ``bike_aboard`` column — ``True`` on the transit rows
        the bicycle rode along — and ``park`` rows where it was left at
        a stop; carriage-set ride transfers render as bicycle legs. A
        walking-only policy at one shared budget rides the legacy
        walking path, and the direct door-to-door alternative stays the
        walking one — direct vehicle journeys ride the standalone street
        products until the vehicle-terms fold arrives. Conflicts with
        the walking options, which are rejected rather than silently
        ignored.
    """

    @property
    def _constructor(self):
        return gpd.GeoDataFrame

    def __init__(
        self,
        network=None,
        origins=None,
        destinations=None,
        departure=None,
        *,
        arrival=None,
        max_rides=8,
        factors=None,
        components=None,
        fares=None,
        candidates="time",
        bucket=25.0,
        router="auto",
        tolerance_minutes=None,
        max_options=None,
        diversity="time",
        penalty="ban",
        exclude_routes=(),
        exclude_trips=(),
        traveler=None,
        exclude_stops=(),
        geometries=True,
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
        if not _is_street_network(network) and hasattr(network, "route_between_stops"):
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
        if (
            not _is_street_network(network)
            and hasattr(network, "route_between_stops")
            and _is_point_frame(origins)
        ):
            street_policy, max_walking_time = folded_street_policy(
                traveler, network, street_policy, walking_speed_kmph, max_walking_time
            )
        from cafein._units import (
            departure_parts,
            duration_seconds,
            validated_output_time_units,
        )

        output_time_units = validated_output_time_units(output_time_units)
        if departure is not None and arrival is not None:
            raise ValueError("give exactly one of departure= or arrival=")
        arrive_by = arrival is not None
        if arrive_by:
            from cafein._units import arrival_parts

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
                raise ValueError(
                    "street_policy= (a traveler's street bridge included) "
                    "does not combine with arrival= yet"
                )
            date, departure = arrival_parts(arrival)
        else:
            date, departure = (
                (None, None) if departure is None else departure_parts(departure)
            )
        if max_rides < 1:
            raise ValueError("max_rides must be at least 1")
        max_transfers = max_rides - 1
        slack_seconds = duration_seconds("tolerance_minutes", tolerance_minutes)
        max_walking_time = duration_seconds("max_walking_time", max_walking_time)
        max_street_time = duration_seconds("max_street_time", max_street_time)
        max_snap_distance = snap_distance
        # Before the reconstruction guard below: a StreetNetwork has no
        # `route_between_stops` either, so it would be mistaken for frame data.
        if _is_street_network(network):
            if street_policy is not None:
                raise ValueError(
                    "street_policy shapes a TransportNetwork's access and "
                    "egress; a StreetNetwork routes one mode directly — pass "
                    "transport_mode instead"
                )
            street_frame = _street_itineraries_frame(
                network,
                origins,
                destinations,
                departure=departure,
                arrive_by=arrive_by,
                transport_mode=transport_mode,
                max_street_time=max_street_time,
                max_snap_distance=max_snap_distance,
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
                    "fares": fares,
                    "traveler": traveler,
                    "tolerance_minutes": slack_seconds,
                    "max_options": max_options,
                    "walking_speed_kmph": walking_speed_kmph,
                    "max_walking_time": max_walking_time,
                    "max_rides": None if max_rides == 8 else max_rides,
                    "candidates": None if candidates == "time" else candidates,
                    "bucket": None if bucket == 25.0 else bucket,
                    "router": None if router == "auto" else router,
                    "diversity": None if diversity == "time" else diversity,
                    "penalty": None if penalty == "ban" else penalty,
                    "exclude_routes": id_sequence("exclude_routes", exclude_routes)
                    or None,
                    "exclude_trips": id_sequence("exclude_trips", exclude_trips)
                    or None,
                    "exclude_stops": id_sequence("exclude_stops", exclude_stops)
                    or None,
                },
            )
            if "travel_time_s" in street_frame.columns:
                from cafein._units import travel_time_output

                position = list(street_frame.columns).index("travel_time_s")
                converted = travel_time_output(
                    street_frame.pop("travel_time_s"), output_time_units
                )
                street_frame.insert(position, "travel_time", converted)
            super().__init__(
                street_frame,
                geometry="geometry",
                crs="EPSG:4326",
            )
            return
        if not hasattr(network, "route_between_stops"):
            # pandas/geopandas reconstruct subclasses by passing data in
            # the first position; wrap it as an ordinary GeoDataFrame.
            super().__init__(network)
            return
        if transport_mode is not None and transport_mode != "public_transport":
            raise ValueError(
                f"transport_mode={transport_mode!r} is a street mode; pass a "
                "StreetNetwork to route on it"
            )
        if max_street_time is not None:
            raise ValueError("max_street_time applies to a StreetNetwork")
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
                "car query"
            )
        if (
            perspectives is not None
            or costs is not None
            or currency is not None
            or cost_components is not None
        ):
            raise ValueError(
                "perspectives, costs, currency, and cost_components price "
                "street kilometres and apply to a StreetNetwork query "
                "(transit perspective costs are not supported)"
            )
        if departure is None and not arrive_by:
            raise ValueError("give exactly one of departure= or arrival=")
        frame = _itineraries_frame(
            network,
            origins,
            destinations,
            date,
            departure,
            max_transfers=max_transfers,
            arrive_by=arrive_by,
            factors=factors,
            components=components,
            fares=fares,
            candidates=candidates,
            bucket=bucket,
            router=router,
            slack_seconds=slack_seconds,
            max_options=max_options,
            diversity=diversity,
            penalty=penalty,
            exclude_routes=exclude_routes,
            exclude_trips=exclude_trips,
            exclude_stops=exclude_stops,
            geometries=geometries,
            walking_speed_kmph=walking_speed_kmph,
            max_walking_time=max_walking_time,
            max_snap_distance=max_snap_distance,
            street_policy=street_policy,
        )
        if "travel_time_s" in frame.columns:
            from cafein._units import travel_time_output

            position = list(frame.columns).index("travel_time_s")
            converted = travel_time_output(
                frame.pop("travel_time_s"), output_time_units
            )
            frame.insert(position, "travel_time", converted)
        super().__init__(frame, geometry="geometry", crs="EPSG:4326")


def _itineraries_frame(
    network,
    origins,
    destinations,
    date,
    departure,
    *,
    max_transfers,
    arrive_by=False,
    factors,
    components,
    fares,
    candidates,
    bucket,
    router,
    slack_seconds,
    max_options,
    diversity,
    penalty,
    exclude_routes,
    exclude_trips,
    exclude_stops,
    geometries,
    walking_speed_kmph,
    max_walking_time,
    max_snap_distance,
    street_policy=None,
):
    import copy

    components = component_selection(components)
    from cafein import emissions
    from cafein.fares import annotate_fares
    from cafein.frontier import _alternative_options, _exclusion_lists

    origin_ids, origin_keys, kind = _endpoints(origins, "origins")
    dest_ids, dest_keys, dest_kind = _endpoints(destinations, "destinations")
    if kind != dest_kind:
        raise ValueError(
            "origins and destinations must both be stop ids or both be "
            "point GeoDataFrames"
        )
    walk = (walking_speed_kmph, max_walking_time, max_snap_distance)
    if kind == "stops" and any(option is not None for option in walk):
        raise ValueError("walking options apply to point origins and destinations")
    if candidates not in ("time", "pareto", "relaxed", "diverse"):
        raise ValueError("candidates must be 'time', 'pareto', 'relaxed', or 'diverse'")
    if router not in ("auto", "raptor", "tbtr"):
        raise ValueError("router must be 'auto', 'raptor', or 'tbtr'")
    if street_policy is not None:
        if kind != "points":
            raise ValueError("street_policy applies to point origins and destinations")
        if candidates == "diverse":
            raise ValueError(
                "street_policy does not support candidates='diverse' yet; "
                "the penalization rounds arrive with a later stage"
            )
        if candidates == "time" and router == "tbtr":
            raise ValueError(
                "street_policy time queries run the RAPTOR arm; "
                "router='tbtr' serves candidates='pareto'"
            )
        if any(option is not None for option in walk):
            raise ValueError(
                "street_policy carries its own budgets; passing "
                "walking_speed_kmph, max_walking_time, or max_snap_distance "
                "beside it is a conflict"
            )
    if router == "tbtr" and candidates != "pareto":
        raise ValueError("router='tbtr' requires candidates='pareto'")
    if router == "auto" and candidates in ("relaxed", "diverse"):
        # Unimplemented on McTBTR; resolve here so every penalization
        # round of a diverse search runs on the same engine.
        router = "raptor"
    slack, options, rounds = _alternative_options(
        candidates, slack_seconds, max_options, diversity, penalty
    )
    exclusions = _exclusion_lists(exclude_routes, exclude_trips, exclude_stops)
    multicriteria = candidates in ("pareto", "relaxed", "diverse")
    transit_factors, street_factors = factors, None
    shared_modes = frozenset()
    if fares is not None:
        # One frozen fare model for the whole frame, like the policy
        # below: every OD pair prices under the same version, even if
        # the caller mutates theirs mid-build.
        fares = copy.deepcopy(fares)
    if street_policy is not None:
        # One frozen policy for the whole frame: every OD pair routes,
        # decorates, and prices emissions under the same version, even
        # if the caller mutates theirs mid-build.
        street_policy = copy.deepcopy(street_policy)
        transit_factors, street_factors = _factor_tables(factors)
        shared_modes = frozenset(
            name
            for name, terms in (street_policy.vehicles or {}).items()
            if terms.source == "shared"
        )
    # The multicriteria (McRAPTOR) candidates need the per-trip factor vector;
    # the time candidates get their emissions from the post-hoc annotation only.
    trip_factors = (
        emissions.trip_factors(network, transit_factors, components)
        if multicriteria
        else None
    )

    records = []
    for origin_id, origin_key in zip(origin_ids, origin_keys):
        for dest_id, dest_key in zip(dest_ids, dest_keys):
            if candidates == "diverse":
                journeys = _route_diverse(
                    network,
                    kind,
                    origin_key,
                    dest_key,
                    date,
                    departure,
                    max_transfers,
                    geometries,
                    walk,
                    router,
                    bucket,
                    trip_factors,
                    factors,
                    components,
                    rounds,
                    diversity,
                    slack,
                    penalty,
                    exclusions,
                )
            else:
                journeys = _route(
                    network,
                    kind,
                    origin_key,
                    dest_key,
                    date,
                    departure,
                    max_transfers,
                    geometries,
                    walk,
                    candidates,
                    router,
                    bucket,
                    slack,
                    options,
                    trip_factors,
                    exclusions,
                    street_policy,
                    street_factors,
                    components,
                    arrive_by=arrive_by,
                )
            if not journeys:
                continue
            network.annotate_emissions(journeys, transit_factors, components)
            if street_policy is not None:
                _street_leg_emissions(
                    journeys, street_factors, components, shared_modes
                )
            if fares is not None:
                annotate_fares(journeys, fares, shared_modes=shared_modes)
            for option, journey in enumerate(journeys):
                for segment, leg in enumerate(journey["legs"]):
                    record = _leg_record(
                        origin_id,
                        dest_id,
                        option,
                        segment,
                        leg,
                        mode=street_policy is not None,
                    )
                    if fares is not None:
                        # A fare prices the whole journey (tickets span
                        # legs); every row of the option repeats it.
                        record["fare"] = journey["fare"]
                    records.append(record)
    columns = _COLUMNS if street_policy is None else _POLICY_COLUMNS
    if fares is not None:
        columns = list(columns)
        columns.insert(columns.index("emissions") + 1, "fare")
    return _to_geodataframe(records, columns)


def _endpoints(value, role):
    """A role's identifiers, routing keys, and kind (stops or points)."""
    if value is None:
        raise ValueError(f"{role} are required for detailed itineraries")
    if _is_point_frame(value):
        ids, points = _point_list(value, role)
        if not ids:
            raise ValueError(f"the {role} GeoDataFrame is empty")
        return ids, points, "points"
    ids = list(id_sequence(role, value))
    if not ids:
        raise ValueError(f"{role} must name at least one stop")
    return ids, ids, "stops"


def _route(
    network,
    kind,
    origin_key,
    dest_key,
    date,
    departure,
    max_transfers,
    geometries,
    walk,
    candidates,
    router,
    bucket,
    slack,
    options,
    trip_factors,
    exclusions,
    street_policy=None,
    street_factors=None,
    components=None,
    arrive_by=False,
):
    """The Pareto-optimal journeys of one OD pair — the time-optimal
    (arrival, rides) set, or the (arrival, emissions) McRAPTOR set with
    ``candidates="pareto"`` / ``"relaxed"``."""
    if candidates in ("pareto", "relaxed"):
        return _route_pareto(
            network,
            kind,
            origin_key,
            dest_key,
            date,
            departure,
            max_transfers,
            geometries,
            walk,
            router,
            bucket,
            slack,
            options,
            trip_factors,
            exclusions,
            street_policy,
            street_factors,
            components,
        )
    if kind == "points":
        walking_speed_kmph, max_walking_time, max_snap_distance = walk
        return network._route_between_coordinates(
            origin_key,
            dest_key,
            date,
            departure,
            max_transfers,
            exclude_routes=exclusions[0],
            exclude_trips=exclusions[1],
            exclude_stops=exclusions[2],
            walking_speed_kmph=walking_speed_kmph,
            max_walking_time=max_walking_time,
            max_snap_distance=max_snap_distance,
            geometries=geometries,
            street_policy=street_policy,
            arrive_by=arrive_by,
        )
    return network._route_between_stops(
        origin_key,
        dest_key,
        date,
        departure,
        max_transfers,
        exclude_routes=exclusions[0],
        exclude_trips=exclusions[1],
        exclude_stops=exclusions[2],
        geometries=geometries,
        arrive_by=arrive_by,
    )


def _route_pareto(
    network,
    kind,
    origin_key,
    dest_key,
    date,
    departure,
    max_transfers,
    geometries,
    walk,
    router,
    bucket,
    slack,
    options,
    trip_factors,
    exclusions,
    street_policy=None,
    street_factors=None,
    components=None,
):
    """The (arrival, emissions) McRAPTOR journeys of one OD pair — the
    cleaner-but-slower alternatives the time-optimal set misses, widened by
    a ``slack``-second slack in the per-stop dominance. Single
    departure (``window=None``)."""
    from cafein.matrices import _walking_only_policy
    from cafein.network import _policy_mc_journeys, _walk_options

    if street_policy is not None:
        from cafein.policy import reject_carriage

        reject_carriage(street_policy, "the multicriteria candidates")
        walk_only, walk_budget = _walking_only_policy(street_policy)
        if not walk_only:
            return _policy_mc_journeys(
                network._core,
                origin_key,
                dest_key,
                date,
                departure,
                max_transfers,
                street_policy,
                exclusions,
                geometries,
                trip_factors,
                street_factors,
                components,
                bucket,
                slack or 0.0,
                options,
                router,
            )
        # A walking-only policy IS the legacy multicriteria walking path,
        # at the policy's one walking budget.
        walk = (None, walk_budget, None)
    if kind == "points":
        return network._core.mc_route_between_coordinates(
            origin_key,
            dest_key,
            date,
            departure,
            trip_factors,
            None,
            max_transfers,
            bucket,
            *_walk_options(*walk),
            geometries,
            slack,
            options,
            exclude_routes=exclusions[0],
            exclude_trips=exclusions[1],
            exclude_stops=exclusions[2],
            router=router,
        )
    return network._core.mc_route_between_stops(
        origin_key,
        dest_key,
        date,
        departure,
        trip_factors,
        None,
        max_transfers,
        bucket,
        router,
        *_walk_options(*walk),
        geometries,
        slack,
        options,
        exclude_routes=exclusions[0],
        exclude_trips=exclusions[1],
        exclude_stops=exclusions[2],
    )


def _route_diverse(
    network,
    kind,
    origin_key,
    dest_key,
    date,
    departure,
    max_transfers,
    geometries,
    walk,
    router,
    bucket,
    trip_factors,
    factors,
    components,
    k,
    diversity,
    slack,
    penalty,
    exclusions,
):
    """``k`` distinct alternatives for one OD pair, by the shared
    ``_diverse_rounds`` loop over single-departure McRAPTOR searches
    (``window=None``). A positive ``slack`` widens each round's pool to the
    relaxed frontier (relaxed × diverse); options come back fastest-first, as
    ``journey_frontier``'s frame sorts them."""
    from cafein.frontier import _diverse_rounds
    from cafein.network import _walk_options

    def search(banned, route_penalties):
        if kind == "points":
            return network._core.mc_route_between_coordinates(
                origin_key,
                dest_key,
                date,
                departure,
                trip_factors,
                None,
                max_transfers,
                bucket,
                *_walk_options(*walk),
                geometries,
                slack,
                None,
                exclude_routes=exclusions[0],
                exclude_trips=exclusions[1],
                exclude_stops=exclusions[2],
                banned_routes=banned,
                route_penalties=route_penalties,
                router=router,
            )
        return network._core.mc_route_between_stops(
            origin_key,
            dest_key,
            date,
            departure,
            trip_factors,
            None,
            max_transfers,
            bucket,
            router,
            *_walk_options(*walk),
            geometries,
            slack,
            None,
            exclude_routes=exclusions[0],
            exclude_trips=exclusions[1],
            exclude_stops=exclusions[2],
            banned_routes=banned,
            route_penalties=route_penalties,
        )

    def annotate(journeys):
        network.annotate_emissions(journeys, factors, components)

    return _diverse_rounds(search, annotate, k, diversity, penalty)


def _leg_record(from_id, to_id, option, segment, leg, mode=False):
    """One leg as a flat record, with its endpoints normalised."""
    leg_type = leg["type"]
    if leg_type == "transit":
        from_stop, to_stop = leg["board_stop"], leg["alight_stop"]
    elif leg_type == "access":
        from_stop, to_stop = None, leg["to_stop"]
    elif leg_type == "egress":
        from_stop, to_stop = leg["from_stop"], None
    elif leg_type == "walk":
        # A door-to-door walking journey never touches a stop.
        from_stop, to_stop = None, None
    elif leg_type == "park":
        # The carried vehicle stays here; a zero-length event.
        from_stop = to_stop = leg["stop"]
    else:
        from_stop, to_stop = leg["from_stop"], leg["to_stop"]
    wkb = leg.get("geometry")
    record = {
        "from_id": from_id,
        "to_id": to_id,
        "option": option,
        "segment": segment,
        "leg_type": leg_type,
        "departure_s": leg["departure_s"],
        "arrival_s": leg["arrival_s"],
        "travel_time_s": leg["arrival_s"] - leg["departure_s"],
        "from_stop": from_stop,
        "to_stop": to_stop,
        "trip_id": leg.get("trip_id"),
        "route_id": leg.get("route_id"),
        "route_short_name": leg.get("route_short_name"),
        "distance_m": leg.get("distance_m"),
        "distance_provenance": leg.get("distance_provenance"),
        "emissions": leg.get("emissions"),
        "geometry": shapely.from_wkb(wkb) if wkb is not None else None,
    }
    if mode:
        # Rebuilt street legs name their mode; a walking-only policy rides
        # the legacy path whose walked legs carry none, so the structural
        # type resolves it there. The distance parts exist on rebuilt
        # street legs only — legacy and transit legs carry nulls.
        record["mode"] = leg.get("mode", None if leg_type == "transit" else "walk")
        if leg_type == "park":
            record["mode"] = None
        record["network_distance_m"] = leg.get("network_distance_m")
        record["connector_distance_m"] = leg.get("connector_distance_m")
        record["bike_aboard"] = bool(leg.get("bike_aboard", False))
    return record


STREET_COLUMNS = [
    "from_id",
    "to_id",
    "option",
    "segment",
    "leg_type",
    "mode",
    "departure_s",
    "arrival_s",
    "travel_time_s",
    "distance_m",
    "network_distance_m",
    "connector_distance_m",
    "distance_provenance",
    "emissions",
    "geometry",
]


def _street_itineraries_frame(
    network,
    origins,
    destinations,
    *,
    departure,
    arrive_by=False,
    transport_mode,
    max_street_time,
    max_snap_distance,
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
    """Street routes as one leg per reachable pair."""
    from cafein import _parking, costs as _costs, emissions
    from cafein._cafein import STREET_DISTANCE_PROVENANCE
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
        chunk=None,
        transit_only=transit_only,
    )
    # Resolved after the argument validation but before the routing call,
    # exactly as the cost matrix does.
    factor = emissions.street_factor(
        transport_mode, factors, components, vehicle_class=vehicle_class
    )
    account = _costs.resolve_query(
        transport_mode, perspectives, costs, currency, cost_components
    )
    table = network._core.cost_matrix(
        query.origin_points,
        query.destination_points,
        transport_mode,
        query.max_seconds,
        query.max_snap_distance,
        bool(geometries),
        car_model=_resolved_delays(
            transport_mode, intersection_delays, profile, delay_model
        ),
    )
    _warn_unsnapped(
        table,
        query.from_ids,
        query.to_ids,
        network=f"the streets the {transport_mode} profile can use",
    )
    from_ids = np.asarray(query.from_ids, dtype=object)
    to_ids = np.asarray(query.to_ids, dtype=object)
    travel_time = table["travel_time_s"]
    network_distance = table["network_distance"]
    if resolved_parking is not None:
        # The parking search ends each leg: seconds join the travel time
        # (and through it the arrival), metres the driven distance and the
        # emissions basis; the leg geometry never shows the search loop.
        seconds, metres = _parking.destination_costs(
            resolved_parking, query.destination_points
        )
        to_index = np.asarray(table["to"])
        travel_time = travel_time + np.rint(seconds[to_index]).astype(np.int64)
        network_distance = network_distance + metres[to_index]
    rows = len(travel_time)
    # A street network has no timetable, so absolute times exist only when a
    # time is supplied to place the leg on a clock: departures anchor
    # the start, an arrival deadline anchors the end — the durations
    # are identical either way.
    if departure is None:
        starts = pd.array([None] * rows, dtype="Int64")
        arrivals = pd.array([None] * rows, dtype="Int64")
    else:
        hours, minutes, seconds = str(departure).split(":")
        moment = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
        times = np.asarray(travel_time, dtype=np.int64)
        if arrive_by:
            starts = pd.array(moment - times, dtype="Int64")
            arrivals = pd.array(np.full(rows, moment, dtype=np.int64), dtype="Int64")
        else:
            starts = pd.array(np.full(rows, moment, dtype=np.int64), dtype="Int64")
            arrivals = pd.array(times + moment, dtype="Int64")
    connector_distance = table["connector_distance"]
    frame = pd.DataFrame(
        {
            "from_id": from_ids[table["from"]],
            "to_id": to_ids[table["to"]],
            # One journey of one leg per pair.
            "option": np.zeros(rows, dtype=np.int64),
            "segment": np.zeros(rows, dtype=np.int64),
            # A direct non-walking leg takes its mode as `leg_type`, as a
            # direct walk already does; `mode` is the unambiguous field.
            "leg_type": np.full(rows, transport_mode, dtype=object),
            "mode": np.full(rows, transport_mode, dtype=object),
            "departure_s": starts,
            "arrival_s": arrivals,
            "travel_time_s": travel_time,
            "distance_m": network_distance + connector_distance,
            "network_distance_m": network_distance,
            "connector_distance_m": connector_distance,
            "distance_provenance": np.full(
                rows, STREET_DISTANCE_PROVENANCE, dtype=object
            ),
            # Post-reconstruction annotation: network metres only — the
            # connectors are the walk to the vehicle, not vehicle-kilometres.
            # The car's per-vehicle factor (resolved above) divides
            # across the persons carried.
            "emissions": network_distance / 1000.0 * factor / occupancy,
        },
        columns=[column for column in STREET_COLUMNS if column != "geometry"],
    )
    if account is not None:
        # Appended after the fixed leg schema: costs ride the same driven
        # kilometres as the emissions, and a missing row's NaN propagates.
        totals, breakdown, label = account
        kilometres = network_distance / 1000.0
        for perspective, per_km in totals.items():
            frame[f"cost_{perspective}"] = kilometres * per_km
        for (perspective, component), per_km in breakdown.items():
            frame[f"cost_{perspective}_{component}"] = kilometres * per_km
        frame["currency"] = label
    if geometries:
        shapes = list(shapely.from_wkb(np.array(table["geometry"], dtype=object)))
    else:
        # The core never built them, so there is nothing to decode.
        shapes = [None] * rows
    geometry = gpd.GeoSeries(shapes, index=frame.index, crs="EPSG:4326")
    return gpd.GeoDataFrame(frame, geometry=geometry, crs="EPSG:4326")


def _street_leg_emissions(journeys, factors, components, shared_modes=frozenset()):
    """Street emissions on the rebuilt vehicle legs, in place.

    Network meters times the mode's resolved per-km factor — a mode in
    ``shared_modes`` (snapshotted from the policy before routing)
    resolves its shared-fleet factors — and the connectors are the walk
    to the vehicle, not vehicle-kilometres. Walking legs keep the
    annotation's zero; an unresolved factor leaves NA, never a silent
    zero. Journey totals grow accordingly.
    """
    from cafein import emissions

    resolved = {}
    for journey in journeys:
        for leg in journey["legs"]:
            mode = leg.get("mode")
            if mode in (None, "walk"):
                continue
            if mode not in resolved:
                shared = mode in shared_modes
                resolved[mode] = emissions.street_factor(
                    mode,
                    factors,
                    components,
                    service_model="shared" if shared else None,
                )
            leg["emissions"] = leg["network_distance_m"] / 1000.0 * resolved[mode]
            total = journey.get("emissions")
            if total is not None:
                journey["emissions"] = total + leg["emissions"]


def _to_geodataframe(records, columns=None):
    """The leg records as a GeoDataFrame with a set geometry and CRS."""
    frame = pd.DataFrame(records, columns=_COLUMNS if columns is None else columns)
    geometry = gpd.GeoSeries(
        frame["geometry"].to_list(), index=frame.index, crs="EPSG:4326"
    )
    frame = frame.drop(columns="geometry")
    return gpd.GeoDataFrame(frame, geometry=geometry, crs="EPSG:4326")
