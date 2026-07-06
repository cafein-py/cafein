"""Time × emissions Pareto frontiers over departure-window journeys.

The frontier answers "what is the lowest-CO₂ way there, and what does it
cost in time — and money?" from the range-RAPTOR candidate set: the
journeys optimal in (departure, arrival, rides) over a departure window,
annotated with emissions (and fares) post hoc, reduced to the rows no
candidate beats on every annotated criterion.

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
    fares=None,
    walking_speed_kmph=None,
    max_walking_time=None,
    max_snap_distance=None,
    geometries=False,
):
    """The travel time × emissions (× fare) trade-off between two places.

    Routes the departure window, attaches emissions to every candidate
    journey, and marks the Pareto frontier: the journeys no other
    candidate beats on every criterion — travel time and emissions,
    plus fare when a fare structure is given. Requires a network built
    with trip distances (the default).

    With a fare structure (`fares`), every candidate is also priced,
    the frame gains a ``fare`` column, and the fare joins the frontier
    as a third criterion: a slower or dirtier journey stays on the
    frontier when it is strictly cheaper.

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
    fares : FareStructure or ZoneFareStructure (optional)
        A fare model (see ``cafein.fares``); prices every candidate,
        adds the ``fare`` column, and makes the fare the frontier's
        third criterion. NaN marks journeys the model cannot price —
        like NaN emissions, they never join the frontier.
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
        ``frontier`` (whether the row is Pareto-optimal — rows with NaN
        on any criterion never are), and ``journey``, the annotated
        journey dict as returned by the routing calls.
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
    if fares is not None:
        from cafein.fares import annotate_fares

        annotate_fares(journeys, fares)
    records = [
        {
            "departure": journey["departure"],
            "arrival": journey["arrival"],
            "travel_time": journey["arrival"] - journey["departure"],
            "rides": journey["rides"],
            "emissions": (
                math.nan if journey["emissions"] is None else journey["emissions"]
            ),
            **({"fare": journey["fare"]} if fares is not None else {}),
            "journey": journey,
        }
        for journey in journeys
    ]
    columns = [c for c in _COLUMNS if c != "frontier"]
    if fares is not None:
        columns.insert(columns.index("journey"), "fare")
    frame = pd.DataFrame(records, columns=columns)
    frame["frontier"] = _frontier_mask(
        frame["travel_time"].tolist(),
        frame["emissions"].tolist(),
        frame["fare"].tolist() if fares is not None else None,
    )
    ordered = [c for c in _COLUMNS if c != "journey"]
    if fares is not None:
        ordered.append("fare")
    ordered.append("journey")
    return (
        frame[ordered].sort_values(["travel_time", "emissions"]).reset_index(drop=True)
    )


def exhaustive_frontier(
    network,
    origin,
    destination,
    date,
    departure,
    *,
    max_transfers=7,
    factors=None,
    components=None,
):
    """The exact time × emissions Pareto set between two stops.

    A brute-force oracle: every boardable trip is considered, with
    gram labels quantized to a microgram (float noise must not split a
    true point), so the result is the mathematically
    complete frontier for the departure — at a cost orders of magnitude
    above ``journey_frontier``. Use it to verify frontiers or inspect
    true Pareto sets for sampled pairs, never in bulk. Journeys riding
    a trip without a resolved emission factor can never sit on an
    emissions frontier and are excluded outright.

    Unlike ``journey_frontier`` this answers a single departure (no
    window) between stop ids (no coordinates), and returns points, not
    journeys.

    Parameters
    ----------
    network : TransportNetwork
        The network to route on; requires trip distances (the default).
    origin, destination : str
        Stop ids.
    date : str
        Service date as ``YYYY-MM-DD``.
    departure : str
        Departure time as ``HH:MM:SS``.
    max_transfers : int (optional, default: 7)
        Maximum number of transfers between rides.
    factors, components : optional
        Emission-factor rows layered over the shipped defaults and the
        LCA components to include, as in ``emissions.annotate``.

    Returns
    -------
    pandas.DataFrame
        One row per true frontier point, sorted by arrival:
        ``arrival`` and ``travel_time`` (seconds), ``rides`` (the
        fewest transit legs achieving the point), and ``emissions``
        (grams CO₂e).
    """
    trip_factors = emissions.trip_factors(network, factors, components)
    points = network._core.pareto_oracle(
        origin, destination, date, departure, trip_factors, max_transfers
    )
    hours, minutes, seconds = departure.split(":")
    start = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    frame = pd.DataFrame(points, columns=["arrival", "emissions", "rides"])
    frame["travel_time"] = frame["arrival"] - start
    return frame[["arrival", "travel_time", "rides", "emissions"]]


def least_emissions(frontier, within=None):
    """The lowest-emission journey of a frontier, as its row.

    Selects among the candidates with resolved emissions — NaN never
    qualifies — so the budgeted view is a filter, not a second search;
    the pick sits on the frontier whenever its other criteria resolve
    too. This is the same rule the matrix computers' emissions
    objective applies per cell.

    Parameters
    ----------
    frontier : pandas.DataFrame
        A ``journey_frontier`` result.
    within : float (optional)
        A travel-time budget in seconds; only journeys at most this long
        qualify.

    Returns
    -------
    pandas.Series or None
        The qualifying row with the lowest emissions (ties resolved
        toward the shorter travel time), or ``None`` when no journey
        qualifies.
    """
    rows = frontier[frontier["emissions"].notna()]
    if within is not None:
        rows = rows[rows["travel_time"] <= within]
    if rows.empty:
        return None
    return rows.sort_values(["emissions", "travel_time"]).iloc[0]


def least_fare(frontier, within=None):
    """The cheapest journey of a frontier, as its row.

    Selects among the priced candidates — an unpriceable (NaN) fare
    never qualifies, but unresolved emissions do not disqualify a
    journey from being the cheapest — so the pick sits on the frontier
    whenever its emissions also resolve. This is the same rule the
    matrix computers' fare objective applies per cell.

    Parameters
    ----------
    frontier : pandas.DataFrame
        A ``journey_frontier`` result priced with ``fares=``.
    within : float (optional)
        A travel-time budget in seconds; only journeys at most this
        long qualify.

    Returns
    -------
    pandas.Series or None
        The qualifying row with the lowest fare (ties resolved toward
        the shorter travel time, then the lower emissions), or ``None``
        when no journey qualifies.
    """
    if "fare" not in frontier.columns:
        raise ValueError(
            "the frontier carries no fares; pass fares= to journey_frontier"
        )
    rows = frontier[frontier["fare"].notna()]
    if within is not None:
        rows = rows[rows["travel_time"] <= within]
    if rows.empty:
        return None
    return rows.sort_values(["fare", "travel_time", "emissions"]).iloc[0]


def _frontier_mask(times, grams, fares=None):
    """Which candidate points no other point dominates.

    A point is dominated when another is at least as good on every axis
    and strictly better on one; a NaN on any axis keeps a point off the
    frontier and out of the domination tests.
    """
    points = (
        list(zip(times, grams)) if fares is None else list(zip(times, grams, fares))
    )
    valid = [not any(math.isnan(value) for value in point) for point in points]
    mask = []
    for i, point in enumerate(points):
        if not valid[i]:
            mask.append(False)
            continue
        dominated = any(
            valid[j]
            and all(other_axis <= axis for other_axis, axis in zip(other, point))
            and any(other_axis < axis for other_axis, axis in zip(other, point))
            for j, other in enumerate(points)
            if j != i
        )
        mask.append(not dominated)
    return mask
