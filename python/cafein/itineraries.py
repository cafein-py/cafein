"""Detailed door-to-door itineraries as a GeoDataFrame."""

import math

import geopandas as gpd
import pandas as pd
import shapely

from cafein.matrices import _is_point_frame, _point_list

_COLUMNS = [
    "from_id",
    "to_id",
    "option",
    "segment",
    "leg_type",
    "departure",
    "arrival",
    "travel_time",
    "from_stop",
    "to_stop",
    "trip_id",
    "route_id",
    "route_short_name",
    "distance",
    "distance_provenance",
    "emissions",
    "geometry",
]


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
    ``egress``, or ``walk`` for a walking-only door-to-door journey),
    ``departure`` and ``arrival`` and ``travel_time`` in
    seconds, ``from_stop`` and ``to_stop`` (the boarding and alighting
    stops; ``None`` at the walked ends of a door-to-door journey),
    ``trip_id``/``route_id``/``route_short_name`` on transit legs,
    ``distance`` (meters) and its ``distance_provenance``, ``emissions``
    (grams CO₂e; ``0`` on walks, ``NaN`` where a ridden trip has no
    matching factor), and ``geometry`` — the leg's shape in EPSG:4326,
    a transit polyline or a walked street path, absent where a leg has
    none. Group by ``["from_id", "to_id", "option"]`` to recover whole
    journeys.

    Origins and destinations are either stop identifiers or point
    GeoDataFrames with an ``id`` column, and both must be the same kind.
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

    Parameters
    ----------
    network : TransportNetwork
        The network to route on.
    origins : list of str, or GeoDataFrame
        Origin stop_ids, or points with an ``id`` column.
    destinations : list of str, or GeoDataFrame
        Destination stop_ids, or points with an ``id`` column; the same
        kind as `origins`.
    date : str
        Service date as ``YYYY-MM-DD``.
    departure : str
        Departure time at every origin as ``HH:MM:SS``.
    max_transfers : int (optional, default: 7)
        Maximum number of transfers between rides.
    factors : DataFrame or path (optional)
        Extra emission-factor rows layered over the shipped defaults;
        see ``cafein.emissions.load_factors``.
    components : list of str (optional)
        The life-cycle components to include (default: all four); see
        ``cafein.emissions.annotate``.
    candidates : {"time", "pareto", "relaxed", "diverse"} (default: "time")
        Which alternatives to return per OD pair. ``"time"`` draws the
        time-optimal (arrival, rides) journeys of the RAPTOR engine;
        ``"pareto"`` draws the (arrival, emissions) journeys of the
        McRAPTOR engine — the cleaner-but-slower alternatives the
        time-optimal set misses — at the single given departure;
        ``"relaxed"`` widens the ``"pareto"`` set by ``slack_seconds`` to
        the suboptimal journeys arriving within the band; ``"diverse"``
        returns ``max_options`` distinct-corridor alternatives, found by
        iterative route penalization (the fastest journey, then the
        fastest avoiding its routes, and so on) so the options ride
        disjoint line sets.
    bucket : float (optional, default: 25.0)
        The emissions bucket width in grams CO₂e for the ``"pareto"``
        search's arrival tie-break; smaller keeps finer emission
        differences apart. Ignored for ``candidates="time"``.
    router : {"raptor", "tbtr"} (optional, default: "raptor")
        The engine backing ``candidates="pareto"`` between stops:
        multicriteria RAPTOR, or trip-based (``"tbtr"``). ``"tbtr"``
        requires ``candidates="pareto"`` with stop-id origins and
        destinations; ``"relaxed"`` and ``"diverse"`` require ``"raptor"``.
    slack_seconds : float (optional, default: 300.0)
        The time-slack band in seconds for ``candidates="relaxed"``: a
        journey is kept even when a cleaner or simpler one dominates it,
        as long as that dominator is not more than ``slack_seconds``
        earlier. ``0`` reproduces ``candidates="pareto"``. Only used with
        ``candidates="relaxed"``.
    max_options : int (optional, default: None)
        For ``candidates="relaxed"``, a cap on the suboptimal alternatives
        kept per OD pair — the frontier is always returned and the nearest
        suboptimal journeys fill the rest, ``None`` keeping every journey
        within the slack. For ``candidates="diverse"``, the number of
        distinct-corridor alternatives per OD pair (``None`` defaults to
        3); fewer are returned when the disjoint corridors run out.
    geometries : bool (optional, default: True)
        Attach each leg's geometry. Turn off to skip the geometry work
        when only the leg records are needed.
    walking_speed_kmph, max_walking_time, max_snap_distance : float
        The street-search options for point origins/destinations, as in
        ``TransportNetwork.route_between_coordinates``; only valid with
        points.
    """

    @property
    def _constructor(self):
        return gpd.GeoDataFrame

    def __init__(
        self,
        network=None,
        origins=None,
        destinations=None,
        date=None,
        departure=None,
        *,
        max_transfers=7,
        factors=None,
        components=None,
        candidates="time",
        bucket=25.0,
        router="raptor",
        slack_seconds=300.0,
        max_options=None,
        geometries=True,
        walking_speed_kmph=None,
        max_walking_time=None,
        max_snap_distance=None,
    ):
        if not hasattr(network, "route_between_stops"):
            # pandas/geopandas reconstruct subclasses by passing data in
            # the first position; wrap it as an ordinary GeoDataFrame.
            super().__init__(network)
            return
        frame = _itineraries_frame(
            network,
            origins,
            destinations,
            date,
            departure,
            max_transfers=max_transfers,
            factors=factors,
            components=components,
            candidates=candidates,
            bucket=bucket,
            router=router,
            slack_seconds=slack_seconds,
            max_options=max_options,
            geometries=geometries,
            walking_speed_kmph=walking_speed_kmph,
            max_walking_time=max_walking_time,
            max_snap_distance=max_snap_distance,
        )
        super().__init__(frame, geometry="geometry", crs="EPSG:4326")


def _itineraries_frame(
    network,
    origins,
    destinations,
    date,
    departure,
    *,
    max_transfers,
    factors,
    components,
    candidates,
    bucket,
    router,
    slack_seconds,
    max_options,
    geometries,
    walking_speed_kmph,
    max_walking_time,
    max_snap_distance,
):
    from cafein import emissions

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
    if router not in ("raptor", "tbtr"):
        raise ValueError("router must be 'raptor' or 'tbtr'")
    if router == "tbtr" and (candidates != "pareto" or kind != "stops"):
        raise ValueError("router='tbtr' requires candidates='pareto' with stop ids")
    if candidates == "relaxed" and not (
        isinstance(slack_seconds, (int, float))
        and math.isfinite(slack_seconds)
        and slack_seconds >= 0
    ):
        raise ValueError("slack_seconds must be a non-negative number of seconds")
    if candidates in ("relaxed", "diverse") and (
        max_options is not None
        and (
            not isinstance(max_options, int)
            or isinstance(max_options, bool)
            or max_options < 1
        )
    ):
        raise ValueError("max_options must be a positive integer or None")
    multicriteria = candidates in ("pareto", "relaxed", "diverse")
    slack = float(slack_seconds) if candidates == "relaxed" else 0.0
    options = max_options if candidates == "relaxed" else None
    # The multicriteria (McRAPTOR) candidates need the per-trip factor vector;
    # the time candidates get their emissions from the post-hoc annotation only.
    trip_factors = (
        emissions.trip_factors(network, factors, components) if multicriteria else None
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
                    max_options if max_options is not None else 3,
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
                )
            if not journeys:
                continue
            network.annotate_emissions(journeys, factors, components)
            for option, journey in enumerate(journeys):
                for segment, leg in enumerate(journey["legs"]):
                    records.append(
                        _leg_record(origin_id, dest_id, option, segment, leg)
                    )
    return _to_geodataframe(records)


def _endpoints(value, role):
    """A role's identifiers, routing keys, and kind (stops or points)."""
    if value is None:
        raise ValueError(f"{role} are required for detailed itineraries")
    if _is_point_frame(value):
        ids, points = _point_list(value, role)
        if not ids:
            raise ValueError(f"the {role} GeoDataFrame is empty")
        return ids, points, "points"
    ids = [str(identifier) for identifier in value]
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
        )
    if kind == "points":
        walking_speed_kmph, max_walking_time, max_snap_distance = walk
        return network.route_between_coordinates(
            origin_key,
            dest_key,
            date,
            departure,
            max_transfers,
            walking_speed_kmph=walking_speed_kmph,
            max_walking_time=max_walking_time,
            max_snap_distance=max_snap_distance,
            geometries=geometries,
        )
    return network.route_between_stops(
        origin_key, dest_key, date, departure, max_transfers, geometries=geometries
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
):
    """The (arrival, emissions) McRAPTOR journeys of one OD pair — the
    cleaner-but-slower alternatives the time-optimal set misses, widened by
    ``slack`` seconds to the suboptimal ones within the band. Single
    departure (``window=None``)."""
    from cafein.network import _walk_options

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
):
    """``k`` route-disjoint alternatives for one OD pair, by iterative route
    penalization: the shortest-travel-time journey, then the shortest one
    avoiding its routes, and so on until ``k`` are found or the disjoint
    corridors run out. Each round is annotated so the travel-time tie-break
    is by emissions. Single departure (``window=None``)."""
    from cafein.network import _walk_options

    def search(banned):
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
                0.0,
                None,
                banned,
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
            0.0,
            None,
            banned,
        )

    banned = []
    selected = []
    for _ in range(k):
        journeys = search(banned)
        if not journeys:
            break
        network.annotate_emissions(journeys, factors, components)
        pick = min(
            journeys,
            key=lambda journey: (
                journey["arrival"] - journey["departure"],
                journey["emissions"] if journey["emissions"] is not None else math.inf,
            ),
        )
        selected.append(pick)
        routes = {
            leg["route_id"]
            for leg in pick["legs"]
            if leg["type"] == "transit" and leg.get("route_id") is not None
        }
        if not routes:
            break
        banned = banned + [route for route in routes if route not in banned]
    return selected


def _leg_record(from_id, to_id, option, segment, leg):
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
    else:
        from_stop, to_stop = leg["from_stop"], leg["to_stop"]
    wkb = leg.get("geometry")
    return {
        "from_id": from_id,
        "to_id": to_id,
        "option": option,
        "segment": segment,
        "leg_type": leg_type,
        "departure": leg["departure"],
        "arrival": leg["arrival"],
        "travel_time": leg["arrival"] - leg["departure"],
        "from_stop": from_stop,
        "to_stop": to_stop,
        "trip_id": leg.get("trip_id"),
        "route_id": leg.get("route_id"),
        "route_short_name": leg.get("route_short_name"),
        "distance": leg.get("distance"),
        "distance_provenance": leg.get("distance_provenance"),
        "emissions": leg.get("emissions"),
        "geometry": shapely.from_wkb(wkb) if wkb is not None else None,
    }


def _to_geodataframe(records):
    """The leg records as a GeoDataFrame with a set geometry and CRS."""
    frame = pd.DataFrame(records, columns=_COLUMNS)
    geometry = gpd.GeoSeries(
        frame["geometry"].to_list(), index=frame.index, crs="EPSG:4326"
    )
    frame = frame.drop(columns="geometry")
    return gpd.GeoDataFrame(frame, geometry=geometry, crs="EPSG:4326")
