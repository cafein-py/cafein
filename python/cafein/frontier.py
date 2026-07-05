"""Time × emissions Pareto frontiers over departure-window journeys.

The frontier answers "what is the lowest-CO₂ way there, and what does it
cost in time?" from the range-RAPTOR candidate set: the journeys optimal
in (departure, arrival, rides) over a departure window, annotated with
emissions post hoc, reduced to the rows no candidate beats on both
travel time and emissions.

The contract follows from the candidate set: a journey that is slower
*and* rides more vehicles than every time-optimal alternative never
enters the candidates, even if it would be cleaner; slower-but-simpler
journeys (fewer rides) do, and door-to-door queries include the
walking-only journey, whose zero emissions anchor the clean end.
"""

import math

import pandas as pd

from cafein import emissions

_COLUMNS = [
    "departure",
    "arrival",
    "travel_time",
    "rides",
    "emissions",
    "frontier",
    "journey",
]


def journey_frontier(
    network,
    origin,
    destination,
    date,
    departure,
    window,
    *,
    max_transfers=7,
    factors=None,
    components=None,
    walking_speed_kmph=None,
    max_walking_time=None,
    max_snap_distance=None,
    geometries=False,
):
    """The travel time × emissions trade-off between two places.

    Routes the departure window, attaches emissions to every candidate
    journey, and marks the Pareto frontier: the journeys no other
    candidate beats on both travel time and emissions. Requires a
    network built with trip distances (the default).

    Parameters
    ----------
    network : TransportNetwork
        The network to route on.
    origin, destination : str or (float, float)
        Stop ids, or ``(lat, lon)`` coordinates in EPSG:4326 — both of
        the same kind. Coordinate queries route door-to-door and include
        the walking-only journey.
    date : str
        Service date as ``YYYY-MM-DD``.
    departure : str
        Start of the departure window as ``HH:MM:SS``.
    window : int
        Departure window in seconds; candidates leave within
        ``[departure, departure + window)``.
    max_transfers : int (optional, default: 7)
        Maximum number of transfers between rides.
    factors, components : optional
        Emission-factor rows layered over the shipped defaults and the
        LCA components to include, as in ``emissions.annotate``.
    walking_speed_kmph, max_walking_time, max_snap_distance : float
        The street-search options for coordinate queries, as in
        ``route_between_coordinates``; only valid with coordinates.
    geometries : bool (optional, default: False)
        Attach leg geometries to the returned journeys.

    Returns
    -------
    pandas.DataFrame
        One row per candidate journey, sorted by travel time:
        ``departure`` and ``arrival`` (seconds past the service day's
        start), ``travel_time`` (seconds), ``rides``, ``emissions``
        (grams CO₂e; NaN when a ridden trip has no matching factor),
        ``frontier`` (whether the row is Pareto-optimal — NaN-emission
        rows never are), and ``journey``, the annotated journey dict as
        returned by the routing calls.
    """
    stops = isinstance(origin, str), isinstance(destination, str)
    if stops[0] != stops[1]:
        raise ValueError(
            "origin and destination must both be stop ids or both be coordinates"
        )
    if stops[0]:
        if not (
            walking_speed_kmph is None
            and max_walking_time is None
            and max_snap_distance is None
        ):
            raise ValueError("street-search options apply to coordinate queries only")
        journeys = network.route_between_stops(
            origin,
            destination,
            date,
            departure,
            max_transfers,
            window,
            geometries=geometries,
        )
    else:
        journeys = network.route_between_coordinates(
            tuple(origin),
            tuple(destination),
            date,
            departure,
            max_transfers,
            window,
            walking_speed_kmph=walking_speed_kmph,
            max_walking_time=max_walking_time,
            max_snap_distance=max_snap_distance,
            geometries=geometries,
        )
    emissions.annotate(journeys, network, factors, components)
    records = [
        {
            "departure": journey["departure"],
            "arrival": journey["arrival"],
            "travel_time": journey["arrival"] - journey["departure"],
            "rides": journey["rides"],
            "emissions": (
                math.nan if journey["emissions"] is None else journey["emissions"]
            ),
            "journey": journey,
        }
        for journey in journeys
    ]
    frame = pd.DataFrame(records, columns=[c for c in _COLUMNS if c != "frontier"])
    frame["frontier"] = _frontier_mask(
        frame["travel_time"].tolist(), frame["emissions"].tolist()
    )
    return (
        frame[_COLUMNS].sort_values(["travel_time", "emissions"]).reset_index(drop=True)
    )


def least_emissions(frontier, within=None):
    """The lowest-emission journey of a frontier, as its row.

    Parameters
    ----------
    frontier : pandas.DataFrame
        A ``journey_frontier`` result.
    within : float (optional)
        A travel-time budget in seconds; only journeys at most this long
        qualify. The lowest-emission journey within a budget is always a
        frontier row, so the budgeted view is a filter, not a search.

    Returns
    -------
    pandas.Series or None
        The qualifying frontier row with the lowest emissions (ties
        resolved toward the shorter travel time), or ``None`` when no
        journey qualifies.
    """
    rows = frontier[frontier["frontier"]]
    if within is not None:
        rows = rows[rows["travel_time"] <= within]
    if rows.empty:
        return None
    return rows.sort_values(["emissions", "travel_time"]).iloc[0]


def _frontier_mask(times, grams):
    """Which (time, grams) points no other point dominates.

    A point is dominated when another is at least as good on both axes
    and strictly better on one; NaN grams never join the frontier and
    never dominate.
    """
    mask = []
    for i, (time_i, grams_i) in enumerate(zip(times, grams)):
        if math.isnan(grams_i):
            mask.append(False)
            continue
        dominated = any(
            time_j <= time_i
            and grams_j <= grams_i
            and (time_j < time_i or grams_j < grams_i)
            for j, (time_j, grams_j) in enumerate(zip(times, grams))
            if j != i and not math.isnan(grams_j)
        )
        mask.append(not dominated)
    return mask
