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
- ``candidates="relaxed"``: the ``"pareto"`` set widened by a time
  tolerance in the per-stop dominance — a journey a cleaner or simpler
  one would dominate is kept unless that dominator is more than
  ``tolerance_minutes`` earlier. Taken over a
  ``departure_time_window`` this matches r5py/R5's detailed-itinerary
  alternatives — a McRAPTOR profile across the window under a per-stop
  suboptimal tolerance, with no route penalty, so trunk-sharing
  options survive — ``tolerance_minutes`` being r5py's
  ``suboptimalMinutes`` (both default to 5 minutes). Because the
  tolerance acts per stop and departures spread across the window,
  kept journeys can arrive more than the tolerance after the
  fastest.
- ``candidates="diverse"``: distinct alternatives found by iterative
  route penalization. By default (``penalty="ban"``) each round bans a
  chosen corridor's routes, so the options ride route-disjoint line sets;
  a numeric ``penalty`` instead makes a used route costly but still
  usable, so corridors may share a trunk.
"""

import math

import pandas as pd

from cafein import emissions
from cafein._validate import component_selection, id_sequence, sequence_not_string
from cafein.travelers import folded_constraints

_COLUMNS = [
    "departure_s",
    "arrival_s",
    "travel_time_s",
    "rides",
    "emissions",
    "frontier",
    "journey",
]


def journey_frontier(
    network,
    origin,
    destination,
    departure,
    departure_time_window,
    *,
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
    max_slower=None,
    exclude_routes=(),
    exclude_trips=(),
    traveler=None,
    exclude_stops=(),
    walking_speed_kmph=None,
    max_walking_time=None,
    snap_distance=None,
    geometries=False,
    output_time_units="minutes",
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
    widens that set by a ``tolerance_minutes`` band in the per-stop
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
    departure : datetime.datetime or str
        Start of the departure window — a datetime, or an ISO string
        like ``"2022-02-22 08:30"``; the service date is its date part.
    departure_time_window : float or datetime.timedelta
        Departure window in minutes; candidates leave within the
        window.
    max_rides : int (optional, default: 8)
        Maximum number of boarded vehicles per journey (rides, not
        transfers: 8 rides allow 7 transfers).
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
        ``tolerance_minutes`` band in the per-stop dominance — the "a bit
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
    router : str (optional, default: "auto")
        The pareto search engine: McRAPTOR (``"raptor"``) answers
        immediately; McTBTR (``"tbtr"``) precomputes the date's
        multicriteria transfer set first — slower for a single pair,
        built for batch reuse — and returns the same journeys, for stop
        ids and coordinates alike. ``"auto"`` (the default) runs on
        McTBTR when a cached transfer set
        (``compute_mctbtr_transfers``) matches the query's date and
        factors and the query asks nothing McTBTR cannot answer, else
        on McRAPTOR. Only used with ``candidates="pareto"``, where
        ``max_slower`` runs on either engine; ``"relaxed"`` and
        ``"diverse"`` require ``"raptor"`` (``"auto"`` resolves to it —
        the cached set is reduced under strict unpenalized dominance,
        which slack and route penalties would invalidate).
    tolerance_minutes : float or datetime.timedelta (optional, default: None)
        The time-tolerance band in minutes. For ``candidates="relaxed"``
        a journey is kept even when a cleaner or simpler one dominates
        it, as long as that dominator is not more than this much
        earlier; ``0`` reproduces the strict ``"pareto"`` frontier. For
        ``candidates="diverse"`` a positive value widens each penalization
        round's pool to that relaxed frontier (relaxed × diverse), so a
        round can pick a slightly suboptimal but more distinct corridor.
        ``None`` takes the per-family default — 5 minutes for
        ``"relaxed"`` (r5py's ``suboptimalMinutes``), ``0`` (strict
        pareto per round) for ``"diverse"``. Unused for ``"time"`` and
        ``"pareto"``.
    max_options : int (optional, default: None)
        For ``candidates="relaxed"``, a cap on the suboptimal
        alternatives kept: the strict frontier is always returned in
        full and the suboptimal journeys nearest to it (smallest
        time-gap) fill the rest up to ``max_options``, so the result can
        exceed it when the frontier is larger; ``None`` returns every
        journey within the tolerance. For ``candidates="diverse"``, the
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
    penalty : str or float (optional, default: "ban")
        How ``candidates="diverse"`` steers each round off the corridors
        already chosen. ``"ban"`` (default) hard-bans every route a chosen
        corridor rode, so the options ride fully route-disjoint line sets.
        A positive number instead adds that many seconds to a chosen
        route's effective arrival per prior use — costly but still usable —
        so a corridor that mostly differs yet shares a trunk can surface
        (the R5-style soft penalty), and the set can hold more options
        before it dries up. Unused for the other candidate sets.
    exclude_routes, exclude_trips, exclude_stops : list of str (optional)
        GTFS ids of supply the journeys must not use — disruption and
        accessibility filters, as in route_between_stops. Runs on
        the McRAPTOR path ("auto" resolves to it); excluded stops
        refuse boarding, alighting, transfers, and access/egress while
        vehicles still ride through them.
    traveler : TravelerProfile (optional)
        One traveler's constraint profile (``cafein.TravelerProfile``):
        its compiled exclusions union the ``exclude_*`` lists, and its
        walking knobs fill the unset walking arguments — a knob set on
        both the call and the profile is rejected.
    max_slower : float or datetime.timedelta (optional, default: None)
        Restrict the ``"pareto"`` frontier (on either engine) to
        journeys near the fast end: per departure pass, every returned
        journey arrives within ``max_slower`` minutes of the pass's
        fastest resolved-factor arrival, and that fastest journey is
        always among the rows (the walking-only journey is dropped when
        it falls outside the band of the fastest transit journey).
        Within the band the set is best-effort, not complete — the
        in-search pruning is a per-stop (prefix) heuristic, so a journey
        whose final arrival is inside the band may still be excluded
        when its prefix strays outside it. ``None`` (the default) keeps
        today's exact behavior.
    walking_speed_kmph, max_walking_time, snap_distance : float
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
        ``departure_s`` and ``arrival_s`` (seconds past the service
        day's start), ``travel_time`` (in ``output_time_units`` —
        whole minutes by default), ``rides``, ``emissions``
        (grams CO₂e; NaN when a ridden trip has no matching factor),
        ``frontier`` (whether the row is Pareto-optimal — rows with NaN
        on any criterion never are), and ``journey``, the annotated
        journey dict as returned by the routing calls.
    """
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
    from cafein._units import (
        departure_parts,
        duration_seconds,
        humanize_frame_time,
        validated_output_time_units,
    )

    output_time_units = validated_output_time_units(output_time_units)
    date, departure = departure_parts(departure)
    window = duration_seconds("departure_time_window", departure_time_window)
    if max_rides < 1:
        raise ValueError("max_rides must be at least 1")
    max_transfers = max_rides - 1
    slack_seconds = duration_seconds("tolerance_minutes", tolerance_minutes)
    max_walking_time = duration_seconds("max_walking_time", max_walking_time)
    max_snap_distance = snap_distance
    components = component_selection(components)
    if candidates not in ("time", "pareto", "relaxed", "diverse"):
        raise ValueError("candidates must be 'time', 'pareto', 'relaxed', or 'diverse'")
    if router not in ("auto", "raptor", "tbtr"):
        raise ValueError("router must be 'auto', 'raptor', or 'tbtr'")
    if router == "tbtr" and candidates != "pareto":
        raise ValueError("router='tbtr' requires candidates='pareto'")
    if router == "auto" and candidates in ("relaxed", "diverse"):
        # A capability boundary, not a gap: McTBTR's persisted transfer
        # set is reduced under strict unpenalized dominance at build
        # time, and slack or route penalties can invalidate transfers
        # discarded against build-time witnesses. Resolve here so every
        # penalization round of a diverse search runs on one engine.
        router = "raptor"
    max_slower = _validated_max_slower(max_slower, candidates, router)
    exclusions = _exclusion_lists(exclude_routes, exclude_trips, exclude_stops)
    slack, options, rounds = _alternative_options(
        candidates, slack_seconds, max_options, diversity, penalty
    )
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
            rounds,
            diversity,
            slack,
            penalty,
            exclusions,
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
                max_slower=max_slower,
                exclude_routes=exclusions[0],
                exclude_trips=exclusions[1],
                exclude_stops=exclusions[2],
            )
        else:
            journeys = network._route_between_stops(
                origin,
                destination,
                date,
                departure,
                max_transfers,
                window,
                exclude_routes=exclusions[0],
                exclude_trips=exclusions[1],
                exclude_stops=exclusions[2],
                walking_speed_kmph=walking_speed_kmph,
                max_walking_time=max_walking_time,
                max_snap_distance=max_snap_distance,
                geometries=geometries,
            )
    elif multicriteria:
        from cafein.network import _walk_options

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
            max_slower=max_slower,
            exclude_routes=exclusions[0],
            exclude_trips=exclusions[1],
            exclude_stops=exclusions[2],
            router=router,
        )
    else:
        journeys = network._route_between_coordinates(
            tuple(origin),
            tuple(destination),
            date,
            departure,
            max_transfers,
            window,
            exclude_routes=exclusions[0],
            exclude_trips=exclusions[1],
            exclude_stops=exclusions[2],
            walking_speed_kmph=walking_speed_kmph,
            max_walking_time=max_walking_time,
            max_snap_distance=max_snap_distance,
            geometries=geometries,
        )
    emissions.annotate(journeys, network, factors, components)
    if fares is not None:
        from cafein.fares import annotate_fares

        annotate_fares(journeys, fares)
    return humanize_frame_time(_frontier_frame(journeys, fares), output_time_units)


def _frontier_frame(journeys, fares):
    """Annotated journeys as the frontier frame: one row per journey with
    the Pareto mark, sorted by travel time (cleaner as the tie-break)."""
    records = [
        {
            "departure_s": journey["departure_s"],
            "arrival_s": journey["arrival_s"],
            "travel_time_s": journey["arrival_s"] - journey["departure_s"],
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
        frame["travel_time_s"].tolist(),
        frame["emissions"].tolist(),
        frame["fare"].tolist() if fares is not None else None,
    )
    ordered = [c for c in _COLUMNS if c != "journey"]
    if fares is not None:
        ordered.append("fare")
    ordered.append("journey")
    return (
        frame[ordered]
        .sort_values(["travel_time_s", "emissions"])
        .reset_index(drop=True)
    )


def journey_frontiers(
    network,
    origins,
    destinations,
    departure,
    departure_time_window,
    *,
    max_rides=8,
    factors=None,
    components=None,
    fares=None,
    bucket=25.0,
    router="auto",
    max_slower=None,
    exclude_routes=(),
    exclude_trips=(),
    traveler=None,
    exclude_stops=(),
    walking_speed_kmph=None,
    max_walking_time=None,
    snap_distance=None,
    geometries=False,
    output_time_units="minutes",
):
    """Batched ``journey_frontier``: every (origin, destination) cell of two
    point sets, from one window profile per origin.

    The candidate set is the strict McRAPTOR pareto family
    (``candidates="pareto"``): per cell, the journeys Pareto-optimal in
    (departure, arrival, emissions) over the departure window — exactly the
    frame ``journey_frontier`` returns for the same pair. One multicriteria
    profile per origin serves all destinations, and origins run in parallel
    with the GIL released, so a batch costs roughly one search per origin
    rather than one per cell. Requires a network built with trip distances.

    Parameters
    ----------
    network : TransportNetwork
        The network to route on.
    origins, destinations : list of str, or point GeoDataFrame
        Stop ids, or point GeoDataFrames with an ``id`` column (any CRS;
        reprojected to EPSG:4326) — both of the same kind. Coordinate
        queries route door-to-door and include the walking-only journey;
        stop-id queries board at the origin stop and route over the
        footpath closure.
    departure, departure_time_window
        The window's start and its length in minutes, as in
        ``journey_frontier``.
    max_rides, factors, components, fares, bucket, router, max_slower
        As in ``journey_frontier`` (``bucket`` is the pareto search's
        emissions bucket width in grams; ``router="tbtr"`` answers over
        the McTBTR engine — one multicriteria transfer set built per
        call backs every origin — and returns the same journeys;
        ``max_slower`` restricts each cell to its own band of the
        cell's per-pass fastest journey, which always stays among the
        rows, on either engine).
    exclude_routes, exclude_trips, exclude_stops
        As in ``journey_frontier``.
    walking_speed_kmph, max_walking_time, snap_distance : float
        Street-search options for the coordinate queries, as in
        ``route_between_coordinates``.
    geometries : bool (optional, default: False)
        Attach leg geometries to the returned journeys.

    Returns
    -------
    pandas.DataFrame
        The long frame: ``from_id``, ``to_id``, and ``journey_frontier``'s
        columns. Cells follow the requested (origin, destination) order,
        rows within a cell the frame's travel-time sort; a cell with no
        feasible journey contributes no rows.
    """
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
    from cafein._units import (
        departure_parts,
        duration_seconds,
        humanize_frame_time,
        validated_output_time_units,
    )

    output_time_units = validated_output_time_units(output_time_units)
    date, departure = departure_parts(departure)
    window = duration_seconds("departure_time_window", departure_time_window)
    if max_rides < 1:
        raise ValueError("max_rides must be at least 1")
    max_transfers = max_rides - 1
    max_walking_time = duration_seconds("max_walking_time", max_walking_time)
    max_snap_distance = snap_distance
    components = component_selection(components)
    stops = _frontier_ids(origins, "origins"), _frontier_ids(
        destinations, "destinations"
    )
    if (stops[0] is None) != (stops[1] is None):
        raise ValueError(
            "origins and destinations must both be stop ids or both be point frames"
        )
    if router not in ("auto", "raptor", "tbtr"):
        raise ValueError("router must be 'auto', 'raptor', or 'tbtr'")
    max_slower = _validated_max_slower(max_slower, "pareto", router)
    exclusions = _exclusion_lists(exclude_routes, exclude_trips, exclude_stops)
    trip_factors = emissions.trip_factors(network, factors, components)
    if stops[0] is not None:
        from_ids, to_ids = stops
        cells = network._core.mc_frontier_matrix(
            from_ids,
            to_ids,
            date,
            departure,
            trip_factors,
            window,
            max_transfers,
            bucket,
            geometries,
            exclude_routes=exclusions[0],
            exclude_trips=exclusions[1],
            exclude_stops=exclusions[2],
            max_slower=max_slower,
            router=router,
        )
    else:
        from cafein.matrices import _point_list, _warn_unsnapped
        from cafein.network import _walk_options

        from_ids, from_points = _point_list(origins, "origins")
        to_ids, to_points = _point_list(destinations, "destinations")
        table = network._core.mc_frontier_matrix_from_points(
            from_points,
            to_points,
            date,
            departure,
            trip_factors,
            window,
            max_transfers,
            bucket,
            *_walk_options(walking_speed_kmph, max_walking_time, max_snap_distance),
            geometries,
            exclude_routes=exclusions[0],
            exclude_trips=exclusions[1],
            exclude_stops=exclusions[2],
            max_slower=max_slower,
            router=router,
        )
        cells = table["journeys"]
        _warn_unsnapped(table, from_ids, to_ids)
    journeys = [journey for row in cells for cell in row for journey in cell]
    emissions.annotate(journeys, network, factors, components)
    if fares is not None:
        from cafein.fares import annotate_fares

        annotate_fares(journeys, fares)
    frames = []
    for from_id, row in zip(from_ids, cells):
        for to_id, cell in zip(to_ids, row):
            if not cell:
                continue
            frame = _frontier_frame(cell, fares)
            frame.insert(0, "from_id", from_id)
            frame.insert(1, "to_id", to_id)
            frames.append(frame)
    if frames:
        return humanize_frame_time(
            pd.concat(frames, ignore_index=True), output_time_units
        )
    columns = ["from_id", "to_id", *(c for c in _COLUMNS if c != "journey")]
    if fares is not None:
        columns.append("fare")
    columns.append("journey")
    return humanize_frame_time(pd.DataFrame(columns=columns), output_time_units)


def frontier_table(
    network,
    origins,
    destinations,
    departure,
    departure_time_window,
    *,
    max_rides=8,
    factors=None,
    components=None,
    bucket=25.0,
    router="auto",
    max_slower=None,
    exclude_routes=(),
    exclude_trips=(),
    traveler=None,
    exclude_stops=(),
    walking_speed_kmph=None,
    max_walking_time=None,
    snap_distance=None,
    output_time_units="minutes",
):
    """``journey_frontiers`` as one flat frame, without journey payloads.

    The exact rows ``journey_frontiers`` returns for the same arguments,
    minus the ``journey`` column: the per-cell pareto families with
    their ``frontier`` marks, flattened Rust-side into columns instead
    of materializing a Python journey per row. For mass-scale frontier
    campaigns this removes most of the result-building cost; use
    ``journey_frontiers`` when the journey payloads, leg geometries, or
    fares are needed.

    Parameters
    ----------
    network, origins, destinations, departure, departure_time_window
        As in ``journey_frontiers``.
    max_rides, factors, components, bucket, router, max_slower
        As in ``journey_frontiers``.
    exclude_routes, exclude_trips, exclude_stops
        As in ``journey_frontiers``.
    walking_speed_kmph, max_walking_time, snap_distance : float
        Street-search options for the coordinate queries, as in
        ``route_between_coordinates``.

    Returns
    -------
    pandas.DataFrame
        Columns ``from_id``, ``to_id``, ``departure``, ``arrival``,
        ``travel_time``, ``rides``, ``emissions`` (NaN where a transit
        leg's factor is unresolved), and ``frontier``.
    """
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
    from cafein._units import (
        departure_parts,
        duration_seconds,
        humanize_frame_time,
        validated_output_time_units,
    )

    output_time_units = validated_output_time_units(output_time_units)
    date, departure = departure_parts(departure)
    window = duration_seconds("departure_time_window", departure_time_window)
    if max_rides < 1:
        raise ValueError("max_rides must be at least 1")
    max_transfers = max_rides - 1
    max_walking_time = duration_seconds("max_walking_time", max_walking_time)
    max_snap_distance = snap_distance
    components = component_selection(components)
    import numpy as np

    stops = _frontier_ids(origins, "origins"), _frontier_ids(
        destinations, "destinations"
    )
    if (stops[0] is None) != (stops[1] is None):
        raise ValueError(
            "origins and destinations must both be stop ids or both be point frames"
        )
    if router not in ("auto", "raptor", "tbtr"):
        raise ValueError("router must be 'auto', 'raptor', or 'tbtr'")
    max_slower = _validated_max_slower(max_slower, "pareto", router)
    exclusions = _exclusion_lists(exclude_routes, exclude_trips, exclude_stops)
    trip_factors = emissions.trip_factors(network, factors, components)
    if stops[0] is not None:
        from_ids, to_ids = stops
        table = network._core.mc_frontier_table(
            from_ids,
            to_ids,
            date,
            departure,
            trip_factors,
            window,
            max_transfers,
            bucket,
            exclude_routes=exclusions[0],
            exclude_trips=exclusions[1],
            exclude_stops=exclusions[2],
            max_slower=max_slower,
            router=router,
        )
    else:
        from cafein.matrices import _point_list, _warn_unsnapped
        from cafein.network import _walk_options

        from_ids, from_points = _point_list(origins, "origins")
        to_ids, to_points = _point_list(destinations, "destinations")
        table = network._core.mc_frontier_table_from_points(
            from_points,
            to_points,
            date,
            departure,
            trip_factors,
            window,
            max_transfers,
            bucket,
            *_walk_options(walking_speed_kmph, max_walking_time, max_snap_distance),
            exclude_routes=exclusions[0],
            exclude_trips=exclusions[1],
            exclude_stops=exclusions[2],
            max_slower=max_slower,
            router=router,
        )
        _warn_unsnapped(table, from_ids, to_ids)
    from_id = np.asarray(from_ids, dtype=object)
    to_id = np.asarray(to_ids, dtype=object)
    return humanize_frame_time(
        pd.DataFrame(
            {
                "from_id": from_id[table["from_index"]],
                "to_id": to_id[table["to_index"]],
                "departure_s": table["departure_s"].astype("int64"),
                "arrival_s": table["arrival_s"].astype("int64"),
                "travel_time_s": table["travel_time_s"].astype("int64"),
                "rides": table["rides"].astype("int64"),
                "emissions": table["emissions"],
                "frontier": table["frontier"],
            }
        ),
        output_time_units,
    )


def _exclusion_lists(exclude_routes, exclude_trips, exclude_stops):
    """The three exclusion id lists as strings, in one tuple."""
    return (
        list(id_sequence("exclude_routes", exclude_routes)),
        list(id_sequence("exclude_trips", exclude_trips)),
        list(id_sequence("exclude_stops", exclude_stops)),
    )


def _validated_max_slower(max_slower, candidates, router):
    """The validated ``max_slower`` band in whole seconds, or ``None``."""
    if max_slower is None:
        return None
    if candidates != "pareto":
        raise ValueError("max_slower requires candidates='pareto'")
    from cafein._units import duration_seconds

    return duration_seconds("max_slower", max_slower)


def _frontier_ids(values, role):
    """The stop ids of a batched frontier input, or ``None`` for a point
    frame (resolved later); anything else raises."""
    from cafein.matrices import _is_point_frame

    if _is_point_frame(values):
        return None
    ids = list(sequence_not_string(role, values))
    if not ids or not all(isinstance(value, str) for value in ids):
        raise ValueError(
            f"{role} must be a non-empty list of stop ids or a point GeoDataFrame"
        )
    return ids


def _diverse_reference(journeys):
    """The (travel_time, emissions) ranges of the first round's full frontier —
    the stable scale the spread distance normalizes against. Journeys with no
    resolved emissions do not set the emissions range; if none resolve, that
    axis is zero-range and contributes nothing."""
    times = [journey["arrival_s"] - journey["departure_s"] for journey in journeys]
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
    travel_time = journey["arrival_s"] - journey["departure_s"]
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
        journey["arrival_s"] - journey["departure_s"],
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
        return nearest, -(journey["arrival_s"] - journey["departure_s"])

    return max(journeys, key=spread_key)


# The engine caps a route penalty at u32::MAX - 1 (u32::MAX is the ban
# sentinel); clamp accumulated penalties here so an arbitrarily large or
# repeatedly-added value never overflows the binding's integer conversion.
_MAX_PENALTY = 2**32 - 2


def _journey_key(journey):
    """A journey's identity for dedup: its ordered legs by type, route, and
    board/alight times. Distinguishes the same corridor caught at different
    departures, so a soft-penalty round never re-selects an already-kept
    journey."""
    return tuple(
        (
            leg["type"],
            leg.get("route_id"),
            int(leg["departure_s"]),
            int(leg["arrival_s"]),
        )
        for leg in journey["legs"]
    )


def _alternative_options(candidates, slack_seconds, max_options, diversity, penalty):
    """Validates the alternative-set knobs shared by ``journey_frontier`` and
    ``DetailedItineraries`` and resolves their per-family defaults: the time
    slack (300 s for ``"relaxed"``, 0 for ``"diverse"``), the relaxed
    suboptimal cap, and the diverse round count (3 when uncapped)."""
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
    if penalty != "ban" and not (
        isinstance(penalty, (int, float))
        and not isinstance(penalty, bool)
        and math.isfinite(penalty)
        and round(penalty) >= 1
    ):
        raise ValueError("penalty must be 'ban' or a number of seconds >= 1")
    if penalty != "ban" and candidates != "diverse":
        raise ValueError("penalty applies only to candidates='diverse'")
    if candidates == "relaxed":
        slack = 300.0 if slack_seconds is None else float(slack_seconds)
    elif candidates == "diverse":
        slack = 0.0 if slack_seconds is None else float(slack_seconds)
    else:
        slack = 0.0
    options = max_options if candidates == "relaxed" else None
    rounds = max_options if max_options is not None else 3
    return slack, options, rounds


def _diverse_rounds(search, annotate, k, diversity, penalty):
    """The penalization-round loop shared by ``journey_frontier`` and
    ``DetailedItineraries``: pick a journey, make its routes costlier,
    search again — until ``k`` are selected or the alternatives dry up.

    ``search(banned, route_penalties)`` runs one McRAPTOR round;
    ``annotate(journeys)`` attaches emissions in place. ``penalty="ban"``
    hard-bans a chosen corridor's routes so the options ride disjoint line
    sets; a positive ``penalty`` adds that many seconds to a chosen route's
    effective arrival per prior use (clamped at the engine's cap). Picks
    deduplicate against already-selected journeys, and a pick with no transit
    routes — the walking-only journey — changes no costs, so the loop keeps
    selecting from the current pool instead of ending; only a routed pick
    triggers the next search. Returns the selection fastest-first."""
    banned = []
    penalties = {}
    selected = []
    seen = set()
    reference = None
    pool = None
    while len(selected) < k:
        if pool is None:
            pool = search(banned, list(penalties.items()))
            if not pool:
                break
            annotate(pool)
            if reference is None:
                reference = _diverse_reference(pool)
        # A soft penalty (or a routeless pick) leaves chosen journeys in the
        # pool, so drop the ones already kept before picking; a ban removes
        # them at the source.
        fresh = [j for j in pool if _journey_key(j) not in seen]
        if not fresh:
            break
        pick = _diverse_pick(fresh, selected, diversity, reference)
        selected.append(pick)
        seen.add(_journey_key(pick))
        route_ids = [
            leg["route_id"]
            for leg in pick["legs"]
            if leg["type"] == "transit" and leg.get("route_id") is not None
        ]
        if not route_ids:
            # Nothing to ban or penalize: the next search would return this
            # same pool, so keep picking from it.
            continue
        if penalty == "ban":
            # Banning is idempotent, so dedup against what is already banned.
            for route in route_ids:
                if route not in banned:
                    banned.append(route)
        else:
            # One penalty step per route use: a corridor riding a route twice
            # is penalized twice. Clamp below the ban sentinel (_MAX_PENALTY).
            step = int(round(penalty))
            for route in route_ids:
                penalties[route] = min(penalties.get(route, 0) + step, _MAX_PENALTY)
        pool = None
    selected.sort(key=_fastest_key)
    return selected


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
    penalty,
    exclusions,
):
    """``k`` distinct alternatives for the frontier, by the shared
    ``_diverse_rounds`` loop over windowed McRAPTOR searches. A positive
    ``slack`` widens each round's pool to the relaxed frontier
    (relaxed × diverse). The returned frame still sorts by travel_time; the
    objective changes which corridors are chosen, not their order."""
    from cafein.network import _walk_options

    def search(banned, route_penalties):
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
                exclude_routes=exclusions[0],
                exclude_trips=exclusions[1],
                exclude_stops=exclusions[2],
                banned_routes=banned,
                route_penalties=route_penalties,
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
            exclude_routes=exclusions[0],
            exclude_trips=exclusions[1],
            exclude_stops=exclusions[2],
            banned_routes=banned,
            route_penalties=route_penalties,
            router=router,
        )

    def annotate(journeys):
        emissions.annotate(journeys, network, factors, components)

    return _diverse_rounds(search, annotate, k, diversity, penalty)


def exhaustive_frontier(
    network,
    origin,
    destination,
    departure,
    *,
    max_rides=8,
    factors=None,
    components=None,
    output_time_units="minutes",
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
    departure : datetime.datetime or str
        The departure — a datetime, or an ISO string like
        ``"2022-02-22 08:30"``; the service date is its date part.
    max_rides : int (optional, default: 8)
        Maximum number of boarded vehicles per journey (rides, not
        transfers: 8 rides allow 7 transfers).
    factors, components : optional
        Emission-factor rows layered over the shipped defaults and the
        LCA components to include, as in ``emissions.annotate``.

    Returns
    -------
    pandas.DataFrame
        One row per true frontier point, sorted by arrival:
        ``arrival_s`` (clock seconds past the service day's start),
        ``travel_time`` (in ``output_time_units`` — whole minutes by
        default), ``rides`` (the fewest transit legs achieving the
        point), and ``emissions`` (grams CO₂e).
    """
    from cafein._units import (
        departure_parts,
        humanize_frame_time,
        validated_output_time_units,
    )

    output_time_units = validated_output_time_units(output_time_units)
    date, departure = departure_parts(departure)
    if max_rides < 1:
        raise ValueError("max_rides must be at least 1")
    max_transfers = max_rides - 1
    components = component_selection(components)
    trip_factors = emissions.trip_factors(network, factors, components)
    points = network._core.pareto_oracle(
        origin, destination, date, departure, trip_factors, max_transfers
    )
    hours, minutes, seconds = departure.split(":")
    start = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    frame = pd.DataFrame(points, columns=["arrival_s", "emissions", "rides"])
    frame["travel_time_s"] = frame["arrival_s"] - start
    return humanize_frame_time(
        frame[["arrival_s", "travel_time_s", "rides", "emissions"]],
        output_time_units,
    )


def least_emissions(frontier, max_travel_time=None):
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
    max_travel_time : float or datetime.timedelta (optional)
        A travel-time budget in minutes; only journeys whose exact
        duration (``arrival_s - departure_s``) is at most this long
        qualify, whatever the frame's ``output_time_units``.

    Returns
    -------
    pandas.Series or None
        The qualifying row with the lowest emissions (ties resolved
        toward the shorter travel time), or ``None`` when no journey
        qualifies.
    """
    rows = frontier[frontier["emissions"].notna()]
    if max_travel_time is not None:
        from cafein._units import duration_seconds

        budget = duration_seconds("max_travel_time", max_travel_time)
        rows = rows[(rows["arrival_s"] - rows["departure_s"]) <= budget]
    if rows.empty:
        return None
    rows = rows.assign(_exact_s=rows["arrival_s"] - rows["departure_s"])
    return rows.sort_values(["emissions", "_exact_s"]).iloc[0].drop("_exact_s")


def least_fare(frontier, max_travel_time=None):
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
    max_travel_time : float or datetime.timedelta (optional)
        A travel-time budget in minutes; only journeys whose exact
        duration (``arrival_s - departure_s``) is at most this long
        qualify, whatever the frame's ``output_time_units``.

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
    if max_travel_time is not None:
        from cafein._units import duration_seconds

        budget = duration_seconds("max_travel_time", max_travel_time)
        rows = rows[(rows["arrival_s"] - rows["departure_s"]) <= budget]
    if rows.empty:
        return None
    rows = rows.assign(_exact_s=rows["arrival_s"] - rows["departure_s"])
    return rows.sort_values(["fare", "_exact_s", "emissions"]).iloc[0].drop("_exact_s")


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


def fare_frontier(
    network,
    origins,
    destinations,
    departure,
    departure_time_window,
    fares,
    *,
    cutoffs,
    max_rides=8,
    max_travel_time=None,
    exact=True,
    departure_time_step=1,
    walking_speed_kmph=None,
    max_walking_time=None,
    snap_distance=None,
    output_time_units="minutes",
):
    """The cutoff-pruned (time, fare) frontier over a departure window.

    Per origin-destination pair and per fare cutoff: the minimum
    travel time among journeys whose fare fits the cutoff, reported
    with that journey's exact fare and rides — r5r's
    ``pareto_frontier`` shape. Fare enters the engine's dominance
    (labels carry the rule-based calculator's exact state), so a
    slower-but-cheaper journey survives to win its cutoff — no fold
    over the fare-blind products can reproduce this. Ties on travel
    time resolve to the cheapest, then simplest, journey.

    Zone fare structures route through the exact zone-ticket engine —
    stop and point origins and destinations alike; always exact, so
    ``exact=False`` is rejected. Rule-based structures keep their
    engine and its ``exact`` disciplines.

    Parameters
    ----------
    network : TransportNetwork
        The network to route on.
    origins, destinations : list of str, or GeoDataFrames
        Stop ids (each origin boards at its stop, exactly as
        ``route_between_stops`` does), or point GeoDataFrames with an
        ``id`` column routing door-to-door over the street network —
        both the same kind. Either way the direct walk joins each
        cell as the zero-fare candidate.
    departure, departure_time_window
        The window's start (a datetime or an ISO string; the service
        date is its date part) and its length in minutes.
    fares : FareStructure or ZoneFareStructure
        The fare model (``cafein.fares``). Rule-based structures ride
        the rule-based engine with both disciplines; zone structures
        ride the exact zone-ticket engine, which reads the zone
        products only (street rentals and grant restrictions do not
        apply) and accepts ``exact=True`` alone.
    cutoffs : list of float
        Required: the ascending monetary cutoffs to prune and report
        at.
    max_rides : int (optional, default: 8)
        Maximum number of boarded vehicles per journey.
    max_travel_time : float or datetime.timedelta (optional)
        A bound on a journey's duration in minutes (r5r caps at 90
        minutes); ``None`` leaves it unbounded — except on a zone fare
        structure, which defaults to 120 minutes: the exact engine
        must otherwise prove no cheaper journey exists anywhere in
        the service day. Keep it bounded at metropolitan scale;
        destinations carrying no fare zone cannot price and cost the
        search most.
    exact : bool (optional, default: True)
        ``True`` keeps every journey the tariff's fine structure can
        distinguish — the exhaustively verified mode; runtimes grow
        steeply with ``max_travel_time``. ``False`` runs the r5r-style
        discipline (earliest arrival per fare class): exact for
        well-behaved tariffs — every reported fare is real — but a
        cheaper journey can be missed where a scarce discount budget
        interacts with transfer windows; large analyses want this
        mode, as r5r's own frontier does.
    departure_time_step : float or datetime.timedelta (optional, default: 1)
        Minutes between the window's sampled departures — R5's
        per-minute rasterisation. Every reported journey is real and
        waits from its sampled departure, so travel times are
        measured against the grid. ``None`` searches every exact
        (trip departure - access walk) event instead — the wait-free
        event profile the shipped frontier products enumerate, at far
        more search passes on point origins. A journey catchable only
        by waiting past the last in-window event belongs to the grid,
        so neither mode's travel times bound the other's at a
        window's edge.
    walking_speed_kmph, max_walking_time, snap_distance : float
        Street-search options for the point form, as in
        ``route_between_coordinates`` (walking time in minutes);
        rejected beside stop ids. The walking-time bound is clamped to
        ``max_travel_time`` — a longer walk cannot join a journey that
        fits the cap.

    Returns
    -------
    pandas.DataFrame
        Columns ``from_id``, ``to_id``, ``cutoff``, ``travel_time``
        (in ``output_time_units``), ``fare``, and ``rides``; a
        (pair, cutoff) whose cutoff no journey fits is absent.
    """
    from cafein._units import (
        departure_parts,
        duration_seconds,
        humanize_frame_time,
        validated_output_time_units,
    )

    output_time_units = validated_output_time_units(output_time_units)
    date, departure = departure_parts(departure)
    window = duration_seconds("departure_time_window", departure_time_window)
    if max_rides < 1:
        raise ValueError("max_rides must be at least 1")
    max_transfers = max_rides - 1
    max_duration = duration_seconds("max_travel_time", max_travel_time)
    departure_step = duration_seconds("departure_time_step", departure_time_step)
    max_walking_time = duration_seconds("max_walking_time", max_walking_time)
    max_snap_distance = snap_distance
    import pandas as pd

    from cafein import fares as fares_module

    if departure_step is not None:
        step = int(departure_step)
        if step <= 0 or step != departure_step:
            raise ValueError(
                "departure_time_step must be a positive duration, or "
                "None for every departure event"
            )
        departure_step = step
    zone_structure = isinstance(fares, fares_module.ZoneFareStructure)
    if not zone_structure and not isinstance(fares, fares_module.FareStructure):
        raise ValueError("fares must be a cafein.fares.FareStructure")
    # A zone structure's exact fare search needs a time limit to stay
    # fast; 120 minutes of total travel time is the default cap, as on
    # the cost matrices.
    if zone_structure and max_duration is None:
        max_duration = 7200
    if zone_structure and exact is not True:
        raise ValueError(
            "the zone fare frontier is always exact; exact=False is the "
            "rule-based engine's fast discipline"
        )
    from_ids = _frontier_ids(origins, "origins")
    to_ids = _frontier_ids(destinations, "destinations")
    if (from_ids is None) != (to_ids is None):
        raise ValueError(
            "origins and destinations must both be stop ids or both be " "point frames"
        )
    if from_ids is not None and any(
        option is not None
        for option in (walking_speed_kmph, max_walking_time, max_snap_distance)
    ):
        raise ValueError(
            "the walking options shape the point form's street search; "
            "stop-id queries board at their stops"
        )
    if from_ids is None:
        from cafein.matrices import _point_list, _warn_unsnapped
        from cafein.network import _walk_options

        from_ids, from_points = _point_list(origins, "origins")
        to_ids, to_points = _point_list(destinations, "destinations")
        walk = _walk_options(walking_speed_kmph, max_walking_time, max_snap_distance)
        if max_duration is not None:
            walk = (walk[0], min(walk[1], max_duration), walk[2])
        if zone_structure:
            try:
                spec = fares._flat_tables(network)
            except ValueError as error:
                raise ValueError(
                    "the zone fare frontier prices the zone-only reading; "
                    'build the structure with zone_fare_structure(..., rules="zones") '
                    "— route, origin/destination, and agency grants are not "
                    "priceable here yet"
                ) from error
        else:
            spec = fares._flat_tables(network)
        entry = (
            network._core._zone_fare_frontier_table_from_points
            if zone_structure
            else network._core._fare_frontier_table_from_points
        )
        extra = {} if zone_structure else {"exact": exact}
        data = entry(
            from_points,
            to_points,
            date,
            departure,
            window,
            spec,
            [float(cutoff) for cutoff in cutoffs],
            max_transfers=max_transfers,
            max_duration=max_duration,
            departure_step=departure_step,
            **extra,
            **dict(
                zip(
                    (
                        "walking_speed_kmph",
                        "max_walking_time",
                        "max_snap_distance",
                    ),
                    walk,
                )
            ),
        )
        _warn_unsnapped(data, from_ids, to_ids)
    elif zone_structure:
        try:
            spec = fares._flat_tables(network)
        except ValueError as error:
            raise ValueError(
                "the zone fare frontier prices the zone-only reading; "
                'build the structure with zone_fare_structure(..., rules="zones") '
                "— route, origin/destination, and agency grants are not "
                "priceable here yet"
            ) from error
        data = network._core._zone_fare_frontier_table(
            from_ids,
            to_ids,
            date,
            departure,
            window,
            spec,
            [float(cutoff) for cutoff in cutoffs],
            max_transfers=max_transfers,
            max_duration=max_duration,
            departure_step=departure_step,
        )
    else:
        data = network._core._fare_frontier_table(
            from_ids,
            to_ids,
            date,
            departure,
            window,
            fares._flat_tables(network),
            [float(cutoff) for cutoff in cutoffs],
            max_transfers=max_transfers,
            max_duration=max_duration,
            exact=exact,
            departure_step=departure_step,
        )
    frame = pd.DataFrame(
        {
            "from_id": [from_ids[i] for i in data["from_index"]],
            "to_id": [to_ids[j] for j in data["to_index"]],
            "cutoff": data["cutoff"],
            "travel_time_s": data["travel_time_s"],
            "fare": data["fare"],
            "rides": data["rides"],
        }
    )
    return humanize_frame_time(frame, output_time_units)
