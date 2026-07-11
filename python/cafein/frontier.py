"""Time × emissions Pareto frontiers over departure-window journeys.

The frontier answers "what is the lowest-CO₂ way there, and what does it
cost in time — and money?" from a candidate journey set annotated with
emissions (and fares) post hoc, reduced to the rows no candidate beats
on every annotated criterion. The candidate set is selected by
``candidates``:

- ``candidates="time"``: the range-RAPTOR set — journeys optimal in
  (departure, arrival, rides). A journey that is slower *and* rides
  more vehicles than every time-optimal alternative never enters this
  set, even if it would be cleaner; slower-but-simpler journeys (fewer
  rides) do, and door-to-door queries include the walking-only journey,
  whose zero emissions anchor the clean end.
- ``candidates="pareto"``: the McRAPTOR set — journeys Pareto-optimal
  in (departure, arrival, emissions), with emissions compared at a
  configurable bucket width during the search. This is the set that
  also holds the cleaner-but-slower-with-more-rides journeys the time
  candidates provably miss; ``exhaustive_frontier`` is its exact,
  brute-force reference.
- ``candidates="relaxed"``: the ``"pareto"`` set widened by a time slack
  in the per-stop dominance — a journey a cleaner or simpler one would
  dominate is kept unless that dominator is more than ``slack_seconds``
  earlier. Taken over a departure ``window`` this matches r5py/R5's
  detailed-itinerary alternatives — a McRAPTOR profile across the window
  under a per-stop suboptimal slack, with no route penalty, so
  trunk-sharing options survive — where ``window`` is r5py's
  ``departure_time_window`` and ``slack_seconds`` its ``suboptimalMinutes``
  (whose 5-minute default is ``slack_seconds``'s 300 s). Because the slack
  acts per stop and departures spread across the window, kept journeys can
  arrive more than ``slack_seconds`` after the fastest.
- ``candidates="diverse"``: distinct-corridor alternatives found by
  iterative route penalization, riding disjoint line sets — unlike
  ``"relaxed"``, the options are forced onto route-disjoint corridors.
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
    candidates="time",
    bucket=25.0,
    router="raptor",
    slack_seconds=None,
    max_options=None,
    diversity="time",
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

    With ``candidates="pareto"`` the candidate set comes from McRAPTOR,
    which searches over (departure, arrival, emissions) directly and so
    also finds the cleaner-but-slower journeys the time-optimal set
    misses; emissions are compared at ``bucket`` grams during the
    search and re-annotated exactly afterwards. ``candidates="relaxed"``
    widens that set by a ``slack_seconds`` slack in the per-stop
    dominance, keeping suboptimal journeys a nearer one would prune
    (capped by ``max_options``), and ``candidates="diverse"`` returns
    ``max_options`` distinct-corridor alternatives by iterative route
    penalization.

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
    candidates : str (optional, default: "time")
        The candidate journey set: ``"time"`` for the range-RAPTOR
        time-optimal journeys, ``"pareto"`` for the McRAPTOR journeys
        Pareto-optimal in (departure, arrival, emissions),
        ``"relaxed"`` for the ``"pareto"`` set widened by a
        ``slack_seconds`` slack in the per-stop dominance — the "a bit
        slower but a real alternative" options that strict Pareto drops —
        or ``"diverse"`` for ``max_options``
        distinct-corridor alternatives, found by iterative route
        penalization (the fastest journey, then the fastest avoiding its
        routes, and so on) so the options ride disjoint line sets. All
        three multicriteria sets require a network with trip distances;
        journeys riding a trip without a resolved emission factor never
        enter them. Coordinate queries route door-to-door either way and
        include the walking-only journey.
    bucket : float (optional, default: 25.0)
        The emissions bucket width in grams CO₂e of the pareto search:
        journeys within one bucket of each other count as equal on
        emissions while searching, bounding its cost. Only used with
        ``candidates="pareto"`` or ``"relaxed"``.
    router : str (optional, default: "raptor")
        The pareto search engine: McRAPTOR (``"raptor"``) answers
        immediately; McTBTR (``"tbtr"``, stop ids only) precomputes the
        date's multicriteria transfer set first — slower for a single
        pair, built for batch reuse — and returns the same journeys.
        Only used with ``candidates="pareto"``; ``"relaxed"`` and
        ``"diverse"`` require ``"raptor"``.
    slack_seconds : float (optional, default: None)
        The time-slack band in seconds. For ``candidates="relaxed"`` a
        journey is kept even when a cleaner or simpler one dominates it,
        as long as that dominator is not more than ``slack_seconds``
        earlier; ``0`` reproduces the strict ``"pareto"`` frontier. For
        ``candidates="diverse"`` a positive value widens each penalization
        round's pool to that relaxed frontier (relaxed × diverse), so a
        round can pick a slightly suboptimal but more distinct corridor.
        ``None`` takes the per-family default — 300 s for ``"relaxed"``
        (r5py's 5-minute ``suboptimalMinutes``), ``0`` (strict pareto per
        round) for ``"diverse"``. Unused for ``"time"`` and ``"pareto"``.
    max_options : int (optional, default: None)
        For ``candidates="relaxed"``, a cap on the suboptimal
        alternatives kept: the strict frontier is always returned in
        full and the suboptimal journeys nearest to it (smallest
        time-gap) fill the rest up to ``max_options``, so the result can
        exceed it when the frontier is larger; ``None`` returns every
        journey within the slack. For ``candidates="diverse"``, the
        number of distinct-corridor alternatives to return (``None``
        defaults to 3); the search may return fewer when disjoint
        corridors run out. Unused for ``"time"`` and ``"pareto"``.
    diversity : str (optional, default: "time")
        The objective for ``candidates="diverse"``: ``"time"`` picks the
        fastest journey each penalization round (cleaner as tie-break), so
        the options bias toward the fast end of the trade-off; ``"spread"``
        seeds on the fastest, then each later round picks the journey
        farthest from the already-chosen corridors in the normalized
        (travel_time, emissions) plane, so the options span the trade-off
        (a fast-dirty one, a slow-clean one, and evenly spaced middles).
        Unused for the other candidate sets.
    walking_speed_kmph, max_walking_time, max_snap_distance : float
        Street-search options for the walking access/egress, as in
        ``route_between_coordinates``. For stop origins/destinations they
        apply only when a whole-day shortcut set routes them door-to-door
        (ULTRA for ``candidates="time"``, McULTRA for ``"pareto"``).
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
    if candidates not in ("time", "pareto", "relaxed", "diverse"):
        raise ValueError("candidates must be 'time', 'pareto', 'relaxed', or 'diverse'")
    if router not in ("raptor", "tbtr"):
        raise ValueError("router must be 'raptor' or 'tbtr'")
    if router == "tbtr" and candidates != "pareto":
        raise ValueError("router='tbtr' requires candidates='pareto'")
    if (
        candidates in ("relaxed", "diverse")
        and slack_seconds is not None
        and not (
            isinstance(slack_seconds, (int, float))
            and math.isfinite(slack_seconds)
            and slack_seconds >= 0
        )
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
    if diversity not in ("time", "spread"):
        raise ValueError("diversity must be 'time' or 'spread'")
    # slack_seconds defaults per family: 300 s for the "relaxed" band, 0 s
    # (strict pareto per round) for "diverse"; a given value applies to either.
    if candidates == "relaxed":
        slack = 300.0 if slack_seconds is None else float(slack_seconds)
    elif candidates == "diverse":
        slack = 0.0 if slack_seconds is None else float(slack_seconds)
    else:
        slack = 0.0
    options = max_options if candidates == "relaxed" else None
    multicriteria = candidates in ("pareto", "relaxed")
    stops = isinstance(origin, str), isinstance(destination, str)
    if stops[0] != stops[1]:
        raise ValueError(
            "origin and destination must both be stop ids or both be coordinates"
        )
    if candidates == "diverse":
        trip_factors = emissions.trip_factors(network, factors, components)
        journeys = _diverse_journeys(
            network,
            stops[0],
            origin,
            destination,
            date,
            departure,
            window,
            max_transfers,
            factors,
            components,
            bucket,
            router,
            trip_factors,
            (walking_speed_kmph, max_walking_time, max_snap_distance),
            geometries,
            max_options if max_options is not None else 3,
            diversity,
            slack,
        )
    elif stops[0]:
        from cafein.network import _walk_options

        if multicriteria:
            trip_factors = emissions.trip_factors(network, factors, components)
            journeys = network._core.mc_route_between_stops(
                origin,
                destination,
                date,
                departure,
                trip_factors,
                window,
                max_transfers,
                bucket,
                router,
                *_walk_options(walking_speed_kmph, max_walking_time, max_snap_distance),
                geometries,
                slack,
                options,
            )
        else:
            journeys = network.route_between_stops(
                origin,
                destination,
                date,
                departure,
                max_transfers,
                window,
                walking_speed_kmph=walking_speed_kmph,
                max_walking_time=max_walking_time,
                max_snap_distance=max_snap_distance,
                geometries=geometries,
            )
    elif multicriteria:
        from cafein.network import _walk_options

        if router == "tbtr":
            raise ValueError("router='tbtr' requires stop ids, not coordinates")
        trip_factors = emissions.trip_factors(network, factors, components)
        journeys = network._core.mc_route_between_coordinates(
            tuple(origin),
            tuple(destination),
            date,
            departure,
            trip_factors,
            window,
            max_transfers,
            bucket,
            *_walk_options(walking_speed_kmph, max_walking_time, max_snap_distance),
            geometries,
            slack,
            options,
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


def _diverse_reference(journeys):
    """The (travel_time, emissions) ranges of the first round's full frontier —
    the stable scale the spread distance normalizes against. Journeys with no
    resolved emissions do not set the emissions range; if none resolve, that
    axis is zero-range and contributes nothing."""
    times = [journey["arrival"] - journey["departure"] for journey in journeys]
    grams = [
        journey["emissions"] for journey in journeys if journey["emissions"] is not None
    ]
    time_range = (min(times), max(times))
    grams_range = (min(grams), max(grams)) if grams else (0.0, 0.0)
    return time_range, grams_range


def _diverse_point(journey, reference):
    """A journey as a point in the normalized (travel_time, emissions) plane.
    Unresolved emissions sit at the reference's dirty end; a zero-range axis
    maps everything to 0 so it cannot skew the distance."""
    (time_lo, time_hi), (grams_lo, grams_hi) = reference
    travel_time = journey["arrival"] - journey["departure"]
    grams = journey["emissions"]
    if grams is None:
        grams = grams_hi
    time = 0.0 if time_hi == time_lo else (travel_time - time_lo) / (time_hi - time_lo)
    emit = 0.0 if grams_hi == grams_lo else (grams - grams_lo) / (grams_hi - grams_lo)
    return time, emit


def _fastest_key(journey):
    """The seed / ``diversity="time"`` order: shortest travel time, cleaner as
    the tie-break (unresolved emissions last)."""
    return (
        journey["arrival"] - journey["departure"],
        journey["emissions"] if journey["emissions"] is not None else math.inf,
    )


def _diverse_pick(journeys, selected, diversity, reference):
    """The penalization round's pick, shared by ``journey_frontier`` and
    ``DetailedItineraries``. The fastest journey seeds round one and drives
    ``diversity="time"``; ``diversity="spread"`` then takes, each later round,
    the journey farthest from the already-selected corridors in the normalized
    (travel_time, emissions) plane (greedy farthest-point dispersion), so the
    options span the trade-off rather than crowding its fast end."""
    if diversity == "time" or not selected:
        return min(journeys, key=_fastest_key)
    chosen = [_diverse_point(journey, reference) for journey in selected]

    def spread_key(journey):
        point = _diverse_point(journey, reference)
        nearest = min(math.hypot(point[0] - c[0], point[1] - c[1]) for c in chosen)
        # Break ties toward the faster journey for a deterministic pick.
        return nearest, -(journey["arrival"] - journey["departure"])

    return max(journeys, key=spread_key)


def _diverse_journeys(
    network,
    is_stop,
    origin,
    destination,
    date,
    departure,
    window,
    max_transfers,
    factors,
    components,
    bucket,
    router,
    trip_factors,
    walk,
    geometries,
    k,
    diversity,
    slack,
):
    """``k`` route-disjoint alternatives by iterative route penalization: each
    round bans the routes every selected journey has ridden, so the alternatives
    use disjoint line sets, and ``_diverse_pick`` chooses the round's journey —
    fastest-first for ``diversity="time"``, or spread across the
    (travel_time, emissions) trade-off for ``diversity="spread"`` — until ``k``
    are found or the search dries up. A positive ``slack`` widens each round's
    McRAPTOR pool to the relaxed frontier (relaxed × diverse), so a round can
    pick a slightly suboptimal but more distinct corridor. The returned frame
    still sorts by travel_time; the objective changes which corridors are
    chosen, not their order."""
    from cafein.network import _walk_options

    def search(banned):
        if is_stop:
            return network._core.mc_route_between_stops(
                origin,
                destination,
                date,
                departure,
                trip_factors,
                window,
                max_transfers,
                bucket,
                router,
                *_walk_options(*walk),
                geometries,
                slack,
                None,
                banned,
            )
        return network._core.mc_route_between_coordinates(
            tuple(origin),
            tuple(destination),
            date,
            departure,
            trip_factors,
            window,
            max_transfers,
            bucket,
            *_walk_options(*walk),
            geometries,
            slack,
            None,
            banned,
        )

    banned = []
    selected = []
    reference = None
    for _ in range(k):
        journeys = search(banned)
        if not journeys:
            break
        emissions.annotate(journeys, network, factors, components)
        if reference is None:
            reference = _diverse_reference(journeys)
        pick = _diverse_pick(journeys, selected, diversity, reference)
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
