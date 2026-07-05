"""Detailed door-to-door itineraries as a GeoDataFrame."""

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

    A GeoDataFrame with one row per leg of every Pareto-optimal journey
    between each origin and each destination: ``from_id`` and ``to_id``
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
    geometries,
    walking_speed_kmph,
    max_walking_time,
    max_snap_distance,
):
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

    records = []
    for origin_id, origin_key in zip(origin_ids, origin_keys):
        for dest_id, dest_key in zip(dest_ids, dest_keys):
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
):
    """The Pareto-optimal journeys of one OD pair."""
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
