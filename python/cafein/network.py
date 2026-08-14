"""The user-facing transport network."""

import os

from cafein._validate import component_selection, id_sequence, sequence_not_string
from cafein._cafein import TransportNetwork as _TransportNetwork


def _gtfs_paths(paths):
    """`paths` as a list of strings; a single bare path is accepted too."""
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    return [os.fspath(path) for path in paths]


def _window_percentiles(window, percentiles, confidence):
    """The percentile list a window/percentiles/confidence spec asks
    for; ``None`` without a window."""
    if window is None:
        if percentiles is not None or confidence is not None:
            raise ValueError(
                "percentiles and confidence require departure_time_window="
            )
        return None
    if percentiles is not None and confidence is not None:
        raise ValueError("pass either percentiles or confidence, not both")
    if confidence is not None:
        if not 0 < confidence < 1:
            raise ValueError("confidence must be within (0, 1)")
        # Rounded so the derived bounds equal their explicit decimal
        # forms; raw float arithmetic (e.g. (1 - 0.9) / 2 * 100 =
        # 4.999999999999999) could otherwise flip a half-up rank tie.
        half = round((1 - confidence) / 2 * 100, 9)
        return [half, 50.0, round(100 - half, 9)]
    if percentiles is None:
        return [50.0]
    return [float(percentile) for percentile in percentiles]


def _walk_options(walking_speed_kmph, max_walking_time, max_snap_distance):
    """Street-query options with the shared defaults filled in."""
    from cafein import streets

    if walking_speed_kmph is None:
        walking_speed_kmph = streets.WALKING_SPEED_KMPH
    if max_walking_time is None:
        max_walking_time = streets.MAX_ACCESS_EGRESS_TIME
    if max_snap_distance is None:
        max_snap_distance = streets.MAX_SNAP_DISTANCE
    return walking_speed_kmph, max_walking_time, max_snap_distance


def _departure_seconds(departure):
    """``HH:MM:SS`` as seconds past the service day's start."""
    hours, minutes, seconds = str(departure).split(":")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)


def _policy_transfer_mode(policy):
    """The policy's transfer binding as a ``(mode, seconds)`` tuple, or
    ``None`` for the installed walking set."""
    if not policy.transfers:
        return None
    ((mode, seconds),) = policy.transfers.items()
    return (mode, float(seconds))


def _policy_reduced(core, point, egress, modes, exclude_stops, transfer_mode=None):
    """One side's reduction: the ``(stop, seconds)`` offsets the engine
    seeds, and the per-stop ``StreetChoice`` tokens the reconstruction
    rebuilds the street legs from."""
    rows = core._reduced_street_offsets(
        point[0],
        point[1],
        egress,
        modes,
        list(id_sequence("exclude_stops", exclude_stops)),
        transfer_mode=transfer_mode,
    )
    offsets = [(stop, seconds) for stop, seconds, *_ in rows]
    tokens = {row[0]: row[1:] for row in rows}
    return offsets, tokens


def _carrying_offsets(rows):
    """The Carrying plane's seeds, each flagged with whether it walked
    to get there — the choice itself is a walk, or the reduction reached
    it over an installed transfer. Such a row has already spent the walk
    a parked vehicle would take, and the carriage engine must not walk
    it again; only a row the vehicle rode may park and walk on."""
    return [(row[0], row[1], row[2] == "walk" or row[-1] is not None) for row in rows]


def _transfer_leg_dicts(
    core, from_stop, to_stop, departure_s, arrival_s, geometries, transfer_mode
):
    """The leg dicts of one relaxed transfer edge: a single walking leg,
    or — when the merged set's edge rode a rental — the walk-ride-walk
    split from its token, the ride's distances from the token itself
    and its shape drawn under the transfer mode's profile. Times split
    by the token's walking seconds; the edge's total stays
    authoritative."""
    token = (
        core._mode_transfer_token(from_stop, to_stop)
        if transfer_mode is not None
        else None
    )
    if token is None:
        walked = core._transfer_leg(
            from_stop, to_stop, geometries, transfer_mode=transfer_mode
        )
        return [
            {
                "type": "transfer",
                "mode": "walk",
                "from_stop": from_stop,
                "to_stop": to_stop,
                "departure_s": departure_s,
                "arrival_s": arrival_s,
                "distance_m": walked[1] if walked is not None else None,
                "distance_provenance": None,
                "geometry": walked[2] if walked is not None else None,
            }
        ]
    pickup, drop, ride_seconds, ride_network, ride_total, pre_seconds, post_seconds = (
        token
    )
    ride_shape = None
    if geometries:
        drawn = core._mode_transfer_ride_leg(from_stop, to_stop)
        if drawn is not None:
            ride_shape = drawn[3]
    legs = []
    at = departure_s
    if pre_seconds > 0 or pickup != from_stop:
        walked = core._transfer_leg(from_stop, pickup, geometries)
        legs.append(
            {
                "type": "transfer",
                "mode": "walk",
                "from_stop": from_stop,
                "to_stop": pickup,
                "departure_s": at,
                "arrival_s": at + pre_seconds,
                "distance_m": walked[1] if walked is not None else None,
                "distance_provenance": None,
                "geometry": walked[2] if walked is not None else None,
            }
        )
        at += pre_seconds
    legs.append(
        {
            "type": "transfer",
            "mode": transfer_mode[0],
            "from_stop": pickup,
            "to_stop": drop,
            "departure_s": at,
            "arrival_s": at + ride_seconds,
            "distance_m": ride_total,
            "network_distance_m": ride_network,
            "connector_distance_m": ride_total - ride_network,
            "distance_provenance": None,
            "geometry": ride_shape,
        }
    )
    at += ride_seconds
    if post_seconds > 0 or drop != to_stop:
        walked = core._transfer_leg(drop, to_stop, geometries)
        legs.append(
            {
                "type": "transfer",
                "mode": "walk",
                "from_stop": drop,
                "to_stop": to_stop,
                "departure_s": at,
                "arrival_s": arrival_s,
                "distance_m": walked[1] if walked is not None else None,
                "distance_provenance": None,
                "geometry": walked[2] if walked is not None else None,
            }
        )
    else:
        legs[-1]["arrival_s"] = arrival_s
    return legs


def _policy_street_legs(
    core, leg, point, tokens, budgets, egress, geometries, transfer_mode=None
):
    """The reconstructed street leg(s) behind one access or egress row.

    The kept token names the winning mode and the stop link it reached;
    the leg rebuilds between the coordinate and that link over the
    multimodal graph. A closure-carried choice (``via``) splits into the
    vehicle leg serving its seed stop plus the walked transfer between
    the seed and the boarded stop, so no leg blends two modes.
    """
    from cafein._cafein import STREET_DISTANCE_PROVENANCE

    stop = leg["from_stop"] if egress else leg["to_stop"]
    # The token's seconds equal the leg's own span — the engine was
    # seeded with them — so the leg times below stand in for them.
    _, mode, edge, fraction, connector, via = tokens[stop]
    seed = via if via is not None and via != stop else stop
    parts = core._multimodal_leg(
        point[0],
        point[1],
        mode,
        seed,
        edge,
        fraction,
        connector,
        egress,
        budgets[mode],
        geometries,
    )
    if parts is None:
        # The reduction proved this link reachable within the budget; a
        # missing rebuild means the two searches drifted apart.
        raise RuntimeError(
            f"the {mode} street leg serving stop {seed!r} could not be "
            "rebuilt from its reduced choice"
        )
    leg_seconds, network_m, connector_m, shape = parts
    street = {
        "type": leg["type"],
        "mode": mode,
        "distance_m": network_m + connector_m,
        "network_distance_m": network_m,
        "connector_distance_m": connector_m,
        "distance_provenance": STREET_DISTANCE_PROVENANCE,
        "geometry": shape,
    }
    if seed == stop:
        street["departure_s"] = leg["departure_s"]
        street["arrival_s"] = leg["arrival_s"]
        end = ("from_stop", seed) if egress else ("to_stop", seed)
        street[end[0]] = end[1]
        return [street]
    # The relaxed edge follows the closure's direction: access moves from
    # the seed the vehicle reached, egress moves *to* the seed it leaves
    # from — asymmetric edges stay honest. Under a merged set the edge
    # itself may ride a rental, so the shared splitter emits its legs.
    if egress:
        # The traveller crosses the transfer first, then the vehicle
        # leaves from the seed's link.
        boundary = max(leg["arrival_s"] - leg_seconds, leg["departure_s"])
        transfer_legs = _transfer_leg_dicts(
            core, stop, seed, leg["departure_s"], boundary, geometries, transfer_mode
        )
        street.update(from_stop=seed, departure_s=boundary, arrival_s=leg["arrival_s"])
        return transfer_legs + [street]
    # Access: the vehicle reaches the seed's link, then the transfer
    # crosses to the boarded stop.
    boundary = min(leg["departure_s"] + leg_seconds, leg["arrival_s"])
    street.update(to_stop=seed, departure_s=leg["departure_s"], arrival_s=boundary)
    transfer_legs = _transfer_leg_dicts(
        core, seed, stop, boundary, leg["arrival_s"], geometries, transfer_mode
    )
    return [street] + transfer_legs


def _policy_journeys(
    core,
    origin,
    destination,
    date,
    departure,
    max_transfers,
    policy,
    exclusions,
    geometries,
):
    """Door-to-door journeys under a street-leg policy.

    Each side reduces over the multimodal graph, the engine routes from
    the reduced offsets, the street legs rebuild from the kept tokens,
    and the direct walking alternative folds in exactly as the legacy
    walking path folds it: a journey is dropped when walking out at its
    own departure would arrive no later, and a walking-only journey
    leads the list unless a kept journey already rides nothing.
    """
    from cafein import streets as _streets
    from cafein._cafein import STREET_DISTANCE_PROVENANCE
    from cafein.policy import reduction_modes

    # One immutable snapshot of the caller's exclusions, reused across
    # every search below — a one-shot iterable or a list mutated between
    # the GIL-releasing calls must not shift the exclusion set mid-query.
    exclude_routes = id_sequence("exclude_routes", exclusions[0])
    exclude_trips = id_sequence("exclude_trips", exclusions[1])
    exclude_stops = id_sequence("exclude_stops", exclusions[2])
    access_modes = reduction_modes(policy, "access", _streets.MAX_ACCESS_EGRESS_TIME)
    egress_modes = reduction_modes(policy, "egress", _streets.MAX_ACCESS_EGRESS_TIME)
    transfer_mode = _policy_transfer_mode(policy)
    origin = tuple(origin)
    destination = tuple(destination)

    def reduced(point, egress_side, modes):
        # A side none of whose modes snap is empty rather than fatal —
        # the direct walking alternative below may still stand, as it
        # does on the policy matrix path. The error is kept: with no
        # walk either, it is the honest answer.
        try:
            offsets, tokens = _policy_reduced(
                core, point, egress_side, modes, exclude_stops, transfer_mode
            )
        except ValueError as error:
            if "too far from the multimodal street network" not in str(error):
                raise
            return [], {}, error
        return offsets, tokens, None

    access, access_tokens, access_error = reduced(origin, False, access_modes)
    egress, egress_tokens, egress_error = reduced(destination, True, egress_modes)
    journeys = core._route_with_access(
        access,
        egress,
        date,
        departure,
        max_transfers,
        list(exclude_routes),
        list(exclude_trips),
        list(exclude_stops),
        geometries,
        transfer_mode=transfer_mode,
    )
    access_budgets = {mode: seconds for mode, seconds, *_ in access_modes}
    egress_budgets = {mode: seconds for mode, seconds, *_ in egress_modes}
    departed = _departure_seconds(departure)
    # The zero-ride composition: ride the street to a stop and leave it
    # on foot without ever boarding. The engine never emits it — its
    # walking ancestor was always dominated by the direct walk, by the
    # triangle inequality on the one walking graph — but a vehicle
    # access can genuinely beat both the walk and every ridden
    # alternative. Both arrays are closed under the installed transfers,
    # so the same-stop minimum covers every through-a-transfer variant.
    egress_seconds = dict(egress)
    composed = None
    for stop, seconds in access:
        other = egress_seconds.get(stop)
        if other is None:
            continue
        total = seconds + other
        if composed is None or total < composed[0]:
            composed = (total, stop, seconds)
    if composed is not None:
        total, stop, access_seconds = composed
        arrival = departed + total
        boundary = departed + access_seconds
        # The Pareto contract over (arrival, rides): the composition
        # rides nothing, so only an equal-or-earlier zero-ride journey
        # dominates it — an earlier *ridden* journey coexists with it,
        # exactly as the walking journey always has. It in turn drops
        # whatever arrives no earlier while riding.
        dominated = any(
            journey["rides"] == 0 and journey["arrival_s"] <= arrival
            for journey in journeys
        )
        if not dominated:
            journeys = [
                journey for journey in journeys if journey["arrival_s"] < arrival
            ]
            journeys.insert(
                0,
                {
                    "departure_s": departed,
                    "arrival_s": arrival,
                    "rides": 0,
                    "legs": [
                        {
                            "type": "access",
                            "to_stop": stop,
                            "departure_s": departed,
                            "arrival_s": boundary,
                        },
                        {
                            "type": "egress",
                            "from_stop": stop,
                            "departure_s": boundary,
                            "arrival_s": arrival,
                        },
                    ],
                },
            )
    for journey in journeys:
        legs = []
        for leg in journey["legs"]:
            if leg["type"] == "access":
                legs.extend(
                    _policy_street_legs(
                        core,
                        leg,
                        origin,
                        access_tokens,
                        access_budgets,
                        False,
                        geometries,
                        transfer_mode,
                    )
                )
            elif leg["type"] == "egress":
                legs.extend(
                    _policy_street_legs(
                        core,
                        leg,
                        destination,
                        egress_tokens,
                        egress_budgets,
                        True,
                        geometries,
                        transfer_mode,
                    )
                )
            elif leg["type"] == "transfer" and transfer_mode is not None:
                # A merged-set edge may ride a rental; the splitter
                # emits its walk-ride-walk legs from the token. Pure
                # walking transfers keep the engine's own leg below.
                legs.extend(
                    _transfer_leg_dicts(
                        core,
                        leg["from_stop"],
                        leg["to_stop"],
                        leg["departure_s"],
                        leg["arrival_s"],
                        geometries,
                        transfer_mode,
                    )
                )
            else:
                leg["mode"] = "walk" if leg["type"] == "transfer" else None
                legs.append(leg)
        journey["legs"] = legs
    # The direct walking alternative rides nothing and needs no stop, so
    # it stands whatever the policy grants; its budget is the policy's
    # walking access budget when walking is granted, else the usual one.
    walk_budget = access_budgets.get("walk", _streets.MAX_ACCESS_EGRESS_TIME)
    direct = core._multimodal_direct_leg(
        origin, destination, "walk", walk_budget, geometries
    )
    if direct is None:
        unsnapped = access_error or egress_error
        if unsnapped is not None:
            # No policy mode snapped on that side and no walk stands in
            # either: the coordinate really is off the street network.
            raise unsnapped
        return journeys
    walk_seconds, network_m, connector_m, shape = direct
    kept = [
        journey
        for journey in journeys
        if journey["arrival_s"] - journey["departure_s"] < walk_seconds
    ]
    if any(journey["rides"] == 0 for journey in kept):
        return kept
    walk = {
        "departure_s": departed,
        "arrival_s": departed + walk_seconds,
        "rides": 0,
        "legs": [
            {
                "type": "walk",
                "mode": "walk",
                "departure_s": departed,
                "arrival_s": departed + walk_seconds,
                "distance_m": network_m + connector_m,
                "network_distance_m": network_m,
                "connector_distance_m": connector_m,
                "distance_provenance": STREET_DISTANCE_PROVENANCE,
                "geometry": shape,
            }
        ],
    }
    return [walk] + kept


def _carriage_journeys(
    core, origin, destination, date, departure, max_transfers, policy, geometries
):
    """Door-to-door carriage journeys: the possession-state search per
    plane-reduced side, the winning chain decorated — street legs from
    the planes' tokens, ride transfers under the carried mode, park
    events and ``bike_aboard`` flags passed through."""
    from cafein import streets as _streets
    from cafein.network import _policy_transfer_mode
    from cafein.policy import carriage_plane_modes, carriage_terms

    mode, vehicle = carriage_terms(policy)
    origin = tuple(origin)
    destination = tuple(destination)
    if not core.has_multimodal_streets:
        raise ValueError(
            "street_policy needs the multimodal street graph; build with "
            "street_modes="
        )
    # Snapshot every policy term before the GIL-releasing searches.
    unknown_rule = vehicle.unknown_bike_trips
    park = (
        None
        if vehicle.facilities == "any_stop"
        else [str(stop) for stop in vehicle.facilities]
    )
    transfer_mode = _policy_transfer_mode(policy)

    # The mode lists complete the snapshot: after this the policy is
    # never read again, so the GIL-releasing searches below cannot mix
    # policy versions.
    side_specs = {
        side: carriage_plane_modes(policy, side, _streets.MAX_ACCESS_EGRESS_TIME)
        for side in ("access", "egress")
    }

    def reduced(point, egress, modes):
        try:
            return _policy_reduced(core, point, egress, modes, ())
        except ValueError as error:
            if "too far from the multimodal street network" not in str(error):
                raise
            return None

    sides = {}
    unsnapped = None
    for side_name, point, egress in (
        ("access", origin, False),
        ("egress", destination, True),
    ):
        carrying_modes, free_modes = side_specs[side_name]
        carrying = reduced(point, egress, carrying_modes)
        free = reduced(point, egress, free_modes)
        if carrying is None and free is None and unsnapped is None:
            # Kept rather than raised: the direct walking alternative
            # below may still stand, exactly as on the policy route.
            unsnapped = ValueError(
                "coordinate too far from the multimodal street network for "
                "every policy mode"
            )
        sides[side_name] = (carrying, free, carrying_modes)
    carr_acc, free_acc, acc_modes = sides["access"]
    carr_egr, free_egr, egr_modes = sides["egress"]

    def carrying_seeds(reduction):
        offsets, tokens = reduction or ([], {})
        return _carrying_offsets([(stop, *tokens[stop]) for stop, _ in offsets])

    journeys = core._carriage_route(
        carrying_seeds(carr_acc),
        (free_acc or ([], {}))[0],
        (carr_egr or ([], {}))[0],
        (free_egr or ([], {}))[0],
        date,
        departure,
        max_transfers,
        unknown_rule,
        geometries,
        park_stops=park,
        transfer_mode=transfer_mode,
    )
    acc_budgets = {name: seconds for name, seconds, *_ in acc_modes}
    egr_budgets = {name: seconds for name, seconds, *_ in egr_modes}
    walk_budgets_acc = {
        "walk": acc_budgets.get("walk", _streets.MAX_ACCESS_EGRESS_TIME)
    }
    walk_budgets_egr = {
        "walk": egr_budgets.get("walk", _streets.MAX_ACCESS_EGRESS_TIME)
    }
    for journey in journeys:
        legs = []
        for leg in journey["legs"]:
            if leg["type"] == "access":
                tokens = (carr_acc if leg["carrying"] else free_acc)[1]
                budgets = acc_budgets if leg["carrying"] else walk_budgets_acc
                legs.extend(
                    _policy_street_legs(
                        core,
                        leg,
                        origin,
                        {leg["to_stop"]: tokens[leg["to_stop"]]},
                        budgets,
                        False,
                        geometries,
                    )
                )
            elif leg["type"] == "egress":
                tokens = (carr_egr if leg["carrying"] else free_egr)[1]
                budgets = egr_budgets if leg["carrying"] else walk_budgets_egr
                legs.extend(
                    _policy_street_legs(
                        core,
                        leg,
                        destination,
                        {leg["from_stop"]: tokens[leg["from_stop"]]},
                        budgets,
                        True,
                        geometries,
                    )
                )
            elif leg["type"] == "transfer" and leg.pop("ride", False):
                drawn = core._carriage_ride_leg(
                    leg["from_stop"], leg["to_stop"], geometries
                )
                network_m = drawn[1] if drawn is not None else None
                connector_m = drawn[2] if drawn is not None else None
                legs.append(
                    {
                        **leg,
                        "mode": mode,
                        "distance_m": (
                            None if drawn is None else network_m + connector_m
                        ),
                        "network_distance_m": network_m,
                        "connector_distance_m": connector_m,
                        "distance_provenance": None,
                        "geometry": drawn[3] if drawn is not None else None,
                    }
                )
            elif leg["type"] == "transfer":
                walked = core._transfer_leg(
                    leg["from_stop"], leg["to_stop"], geometries
                )
                legs.append(
                    {
                        **leg,
                        "mode": "walk",
                        "distance_m": walked[1] if walked is not None else None,
                        "distance_provenance": None,
                        "geometry": walked[2] if walked is not None else None,
                    }
                )
            else:
                # Transit legs keep their bike_aboard flag; park events
                # pass through with no mode.
                leg.setdefault("mode", None)
                legs.append(leg)
        journey["legs"] = legs
    # The direct walking alternative folds in exactly as on the policy
    # route: a journey stands only when strictly faster than walking
    # out, and the walk leads unless a kept journey already rides
    # nothing. The carried vehicle's own direct ride stays with the
    # standalone street products, as on every policy surface.
    from cafein._cafein import STREET_DISTANCE_PROVENANCE

    departed = _departure_seconds(departure)
    direct = core._multimodal_direct_leg(
        origin, destination, "walk", walk_budgets_acc["walk"], geometries
    )
    if direct is None:
        if unsnapped is not None and not journeys:
            raise unsnapped
        return journeys
    walk_seconds, network_m, connector_m, shape = direct
    kept = [
        journey
        for journey in journeys
        if journey["arrival_s"] - journey["departure_s"] < walk_seconds
    ]
    if any(journey["rides"] == 0 for journey in kept):
        return kept
    walk = {
        "departure_s": departed,
        "arrival_s": departed + walk_seconds,
        "rides": 0,
        "legs": [
            {
                "type": "walk",
                "mode": "walk",
                "departure_s": departed,
                "arrival_s": departed + walk_seconds,
                "distance_m": network_m + connector_m,
                "network_distance_m": network_m,
                "connector_distance_m": connector_m,
                "distance_provenance": STREET_DISTANCE_PROVENANCE,
                "geometry": shape,
            }
        ],
    }
    return [walk] + kept


def _policy_mc_journeys(
    core,
    origin,
    destination,
    date,
    departure,
    max_transfers,
    policy,
    exclusions,
    geometries,
    trip_factors,
    street_factors,
    components,
    bucket,
    slack,
    max_options,
    router="auto",
):
    """Door-to-door multicriteria journeys under a street-leg policy.

    Each side reduces to its ``(seconds, grams)`` Pareto frontier, and
    the resolved multicriteria engine — McRAPTOR, or McTBTR when the
    cached transfer set serves the query — seeds and drains those label
    sets with the street grams inside its (arrival, emissions bucket)
    dominance — zero-ride street compositions included, the engines
    drain the seeds themselves — and the street legs rebuild from the
    kept tokens by ``(stop, seconds)``. The direct walking alternative applies the
    multicriteria rule exactly as ``mc_route_between_coordinates`` does:
    journeys must beat it strictly, and it leads the list as the
    zero-emission, zero-ride baseline.
    """
    import pandas as pd

    from cafein import emissions
    from cafein import streets as _streets
    from cafein._cafein import STREET_DISTANCE_PROVENANCE
    from cafein.policy import pareto_reduction_modes

    from cafein.policy import reject_carriage

    reject_carriage(policy, "the multicriteria candidates")
    transfer_mode = _policy_transfer_mode(policy)
    transfer_arg = None
    if transfer_mode is not None:
        mode, budget = transfer_mode
        # The rental's ride grams enter the dominance, so the transfer
        # mode's shared-fleet factor must resolve, exactly as a granted
        # access mode's must.
        value = emissions.street_factor(
            mode, street_factors, components, service_model="shared"
        )
        if pd.isna(value):
            raise ValueError(
                f"the {mode} emission factor is unresolved; the "
                "multicriteria search ranks street emissions, so pass "
                "factors= rows resolving it (see "
                "cafein.emissions.load_street_factors)"
            )
        transfer_arg = (mode, budget, float(value) / 1000.0)
    exclude_routes = id_sequence("exclude_routes", exclusions[0])
    exclude_trips = id_sequence("exclude_trips", exclusions[1])
    exclude_stops = id_sequence("exclude_stops", exclusions[2])
    access_modes = pareto_reduction_modes(
        policy, "access", _streets.MAX_ACCESS_EGRESS_TIME, street_factors, components
    )
    egress_modes = pareto_reduction_modes(
        policy, "egress", _streets.MAX_ACCESS_EGRESS_TIME, street_factors, components
    )
    origin = tuple(origin)
    destination = tuple(destination)

    def frontier(point, egress_side, modes):
        # A side none of whose modes snap is empty rather than fatal, as
        # on every policy path; the error is kept for the no-walk case.
        try:
            rows = core._pareto_street_rows(
                point[0],
                point[1],
                egress_side,
                modes,
                list(exclude_stops),
                transfer_mode=transfer_arg,
            )
        except ValueError as error:
            if "too far from the multimodal street network" not in str(error):
                raise
            return [], {}, error
        labels = (
            [(row[0], row[1], row[2], row[11]) for row in rows]
            if not egress_side
            else [(row[0], row[1], row[2]) for row in rows]
        )
        tokens = {
            (row[0], row[1]): (row[1], row[3], row[4], row[5], row[6], row[10])
            for row in rows
        }
        return labels, tokens, None

    access, access_tokens, access_error = frontier(origin, False, access_modes)
    egress, egress_tokens, egress_error = frontier(destination, True, egress_modes)
    journeys = core._mc_route_with_access(
        access,
        egress,
        date,
        departure,
        trip_factors,
        max_transfers,
        bucket,
        float(slack),
        max_options,
        router,
        list(exclude_routes),
        list(exclude_trips),
        list(exclude_stops),
        geometries,
        transfer_mode=transfer_arg,
    )
    access_budgets = {mode: seconds for mode, seconds, *_ in access_modes}
    egress_budgets = {mode: seconds for mode, seconds, *_ in egress_modes}
    for journey in journeys:
        legs = []
        for leg in journey["legs"]:
            if leg["type"] == "access":
                span = leg["arrival_s"] - leg["departure_s"]
                token = access_tokens[(leg["to_stop"], span)]
                legs.extend(
                    _policy_street_legs(
                        core,
                        leg,
                        origin,
                        {leg["to_stop"]: token},
                        access_budgets,
                        False,
                        geometries,
                        transfer_mode,
                    )
                )
            elif leg["type"] == "egress":
                span = leg["arrival_s"] - leg["departure_s"]
                token = egress_tokens[(leg["from_stop"], span)]
                legs.extend(
                    _policy_street_legs(
                        core,
                        leg,
                        destination,
                        {leg["from_stop"]: token},
                        egress_budgets,
                        True,
                        geometries,
                        transfer_mode,
                    )
                )
            elif leg["type"] == "transfer" and transfer_mode is not None:
                legs.extend(
                    _transfer_leg_dicts(
                        core,
                        leg["from_stop"],
                        leg["to_stop"],
                        leg["departure_s"],
                        leg["arrival_s"],
                        geometries,
                        transfer_mode,
                    )
                )
            else:
                leg["mode"] = "walk" if leg["type"] == "transfer" else None
                legs.append(leg)
        journey["legs"] = legs
    walk_budget = access_budgets.get("walk", _streets.MAX_ACCESS_EGRESS_TIME)
    direct = core._multimodal_direct_leg(
        origin, destination, "walk", walk_budget, geometries
    )
    if direct is None:
        unsnapped = access_error or egress_error
        if unsnapped is not None:
            raise unsnapped
        return journeys
    walk_seconds, network_m, connector_m, shape = direct
    kept = [
        journey
        for journey in journeys
        if journey["arrival_s"] - journey["departure_s"] < walk_seconds
    ]
    departed = _departure_seconds(departure)
    walk = {
        "departure_s": departed,
        "arrival_s": departed + walk_seconds,
        "rides": 0,
        "legs": [
            {
                "type": "walk",
                "mode": "walk",
                "departure_s": departed,
                "arrival_s": departed + walk_seconds,
                "distance_m": network_m + connector_m,
                "network_distance_m": network_m,
                "connector_distance_m": connector_m,
                "distance_provenance": STREET_DISTANCE_PROVENANCE,
                "geometry": shape,
            }
        ],
    }
    return [walk] + kept


class TransportNetwork:
    """A routable public-transport network.

    Built from GTFS timetables and, optionally, an OpenStreetMap extract
    whose walking network provides the stop-to-stop footpath transfers.
    """

    def __init__(self, core):
        self._core = core

    @classmethod
    def from_gtfs(
        cls,
        paths,
        *,
        osm_pbf=None,
        walking_speed_kmph=None,
        max_walking_time=None,
        snap_distance=None,
        bounding_box=None,
        trip_distances=True,
        leg_geometries=True,
        ultra=False,
        street_modes=None,
        dem=None,
        dem_interval=25.0,
        country=None,
        urban_areas=None,
        speed_limits=None,
    ):
        """Build a network from GTFS archives and an optional OSM extract.

        Parameters
        ----------
        paths : path or list of paths
            GTFS zip files or directories, as strings or path-likes; a
            single feed may be given bare. Several feeds are merged; a
            stop_id occurring in more than one feed must then be
            qualified as ``<feed_index>:<stop_id>``, with feeds numbered
            in input order.
        osm_pbf : str (optional)
            Path to an OpenStreetMap PBF extract covering the stops. Its
            walking network is turned into stop-to-stop footpaths (see
            ``cafein.streets.walking_footpaths``) that routing uses as
            transfers, and installed as the street network behind
            coordinate-based access/egress searches (``access_stops``);
            without it the network has neither.
        walking_speed_kmph : float (optional, default: 3.6)
            Walking speed in km/h of the footpath precompute.
        max_walking_time : float or datetime.timedelta (optional, default: 20 minutes)
            Walking-time cutoff of the direct footpath search, in
            minutes; chained footpaths may exceed it.
        snap_distance : float (optional, default: 1600)
            Maximum distance in meters from a stop to the walking
            network; stops farther away get no footpaths.
        bounding_box : sequence of float or shapely geometry (optional)
            Restrict the OSM walking network to this area, as
            ``[min_lon, min_lat, max_lon, max_lat]`` or a shapely
            geometry, so a region-wide extract can be cropped to the
            stops' neighbourhood; stops snap only to the cropped network,
            so those beyond `snap_distance` of it get no footpaths.
            Only meaningful with `osm_pbf`.
        trip_distances : bool (optional, default: True)
            Compute per-trip travel distances through the fallback
            ladder (``cafein.geometry.trip_distances``), so transit legs
            report their distance and its provenance.
        leg_geometries : bool (optional, default: True)
            Also store the trips' polylines, so transit legs report
            their geometry; disable to save memory when geometries are
            never needed. Ignored when `trip_distances` is off.
        street_modes : iterable of str, optional
            Also build the multimodal union street graph from `osm_pbf`
            (e.g. ``("walk", "bicycle", "e_scooter")``, the pruning modes
            of ``StreetNetwork.from_osm``) and carry it with the network —
            the second street section behind cycling / e-scooter access
            and egress. The walking graph and every existing query are
            untouched. Requires `osm_pbf`. With `osm_pbf` the default is
            ``("walk",)`` — walking is how public-transport journeys
            begin and end — and ``()`` opts out of the multimodal graph
            entirely. A malformed value (a bare string, an unknown mode)
            raises before any file is read.
        dem, dem_interval : optional
            Elevation for the multimodal graph, exactly as in
            ``StreetNetwork.from_osm``; only meaningful with
            `street_modes`.
        country, urban_areas, speed_limits : optional
            Car speed configuration, exactly as in
            ``StreetNetwork.from_osm``; only meaningful with ``"car"``
            in `street_modes`.
        ultra : bool (optional, default: False)
            Also compute the whole-day ULTRA intermediate-transfer
            shortcuts (see ``compute_ultra_shortcuts``), giving the
            point-destination time queries unrestricted intermediate
            walking. Requires ``osm_pbf`` and uses ``walking_speed_kmph``.
            It is a heavy, run-once operation (minutes over a metropolitan
            network); ``save`` the result and reuse it. Off by default.

        Notes
        -----
        The build reads the input files more than once (timetable,
        distance ladder, footpaths — and with `street_modes`, a second
        pass over `osm_pbf` for the multimodal graph); they must not
        change underneath it.
        """
        from cafein._units import duration_seconds

        max_walking_time = duration_seconds("max_walking_time", max_walking_time)
        max_snap_distance = snap_distance
        from cafein import street_network as _street_network

        # Every street_modes problem surfaces here, before any file is
        # read — not minutes later, once the timetable is built.
        if street_modes is None:
            street_modes = ("walk",) if osm_pbf is not None else ()
        else:
            street_modes = _street_network.validate_build_modes(street_modes)
        paths = _gtfs_paths(paths)
        if ultra and osm_pbf is None:
            raise ValueError("ultra=True requires an OSM extract; pass osm_pbf=")
        if street_modes and osm_pbf is None:
            raise ValueError("street_modes requires an OSM extract; pass osm_pbf=")
        if dem is not None and not street_modes:
            raise ValueError("dem applies to the multimodal graph; pass street_modes=")
        if "car" not in street_modes and any(
            option is not None for option in (country, urban_areas, speed_limits)
        ):
            raise ValueError(
                "country, urban_areas, and speed_limits configure car speeds; "
                "pass street_modes= including 'car'"
            )
        core = _TransportNetwork.from_gtfs(paths)
        if trip_distances:
            from cafein import geometry

            if leg_geometries:
                distances, polylines = geometry.trip_distances(
                    paths, include=set(core.trip_ids), geometries=True
                )
                core.set_trip_distances(distances)
                core.set_leg_geometries(*polylines)
            else:
                core.set_trip_distances(
                    geometry.trip_distances(paths, include=set(core.trip_ids))
                )
        if osm_pbf is not None:
            from cafein import streets

            if walking_speed_kmph is None:
                walking_speed_kmph = streets.WALKING_SPEED_KMPH
            if max_walking_time is None:
                max_walking_time = streets.MAX_WALKING_TIME
            if max_snap_distance is None:
                max_snap_distance = streets.MAX_SNAP_DISTANCE
            import datetime as _datetime

            footpaths, street_network = streets.walking_streets(
                osm_pbf,
                core.stops,
                walking_speed_kmph=walking_speed_kmph,
                # Internal values are seconds; the public streets API
                # takes minutes or timedeltas — a timedelta is exact.
                max_walking_time=_datetime.timedelta(seconds=max_walking_time),
                snap_distance=max_snap_distance,
                bounding_box=bounding_box,
            )
            core.set_transfer_arrays(
                footpaths.stop_ids,
                footpaths.from_index,
                footpaths.to_index,
                footpaths.seconds,
                footpaths.meters,
            )
            core.set_street_network(*street_network)
            if street_modes:
                core.set_multimodal_streets(
                    list(street_modes),
                    *_street_network.multimodal_payload(
                        osm_pbf,
                        modes=street_modes,
                        bounding_box=bounding_box,
                        dem=dem,
                        dem_interval=dem_interval,
                        country=country,
                        urban_areas=urban_areas,
                        speed_limits=speed_limits,
                    ),
                )
            if ultra:
                core.compute_ultra_shortcuts(walking_speed_kmph)
        return cls(core)

    @property
    def has_multimodal_streets(self):
        """Whether the multimodal union street graph is carried."""
        return self._core.has_multimodal_streets

    @property
    def street_modes(self):
        """The pruning modes the multimodal graph was built with, or ``None``."""
        modes = self._core.street_modes
        return None if modes is None else tuple(modes)

    @property
    def multimodal_elevation_metadata(self):
        """Provenance of the multimodal graph's elevations, or ``None``."""
        return self._core.multimodal_elevation_metadata

    def save(self, path):
        """Save the network as a reusable artifact.

        The artifact carries everything queries need — the timetable,
        service calendar, transfers, trip distances, leg geometries,
        the street network, the ULTRA shortcut set with its compute
        window, and any cached TBTR transfer set (when computed) — so
        batch jobs can ``load`` the same file read-only instead of
        rebuilding from GTFS and OSM inputs.
        Build diagnostics (quarantine warnings) are not persisted.
        The file is staged beside the destination and atomically
        renamed into place, so saving over an existing artifact never
        rewrites it under live memory-mapped readers.

        Parameters
        ----------
        path : path
            Destination file, conventionally ``*.cafein``.
        """
        self._core.save(os.fspath(path))

    @classmethod
    def load(cls, path, *, mmap=False, verify=None):
        """Load a network saved with `save`.

        Artifacts written in another format version are refused with
        a message naming the writing cafein version, and corrupted
        payloads fail their checksum; rebuild from the inputs (or
        re-save) with a matching version instead. Artifacts are
        trusted input, like pickles: load only files you created.

        With ``mmap=True`` the street arrays are used directly from a
        read-only memory map of the file instead of being copied into
        memory: the operating system pages street data in as queries
        touch it and shares those pages between every process mapping
        the same artifact, so per-process memory scales with the region
        a job actually walks, not with the network. A mapped artifact
        must stay unchanged while any process maps it — replace it by
        writing a new file and renaming it over the old one, never by
        editing in place, and keep it out of cloud-synced folders
        (OneDrive and its kin rewrite files in place).

        Parameters
        ----------
        path : path
            An artifact written by `save`.
        mmap : bool or "require"
            ``False`` (default) loads everything into memory. ``True``
            maps the file, falling back to the in-memory load where
            mapping is unavailable; ``"require"`` raises instead of
            falling back.
        verify : bool, optional
            Whether to checksum the street data. Defaults to ``True``
            for in-memory loads (the bytes are read anyway) and
            ``False`` for mapped loads, where the check would page the
            whole street section in and defeat the lazy load.
        """
        modes = {False: "off", True: "auto", "require": "require"}
        if mmap not in modes:
            raise ValueError(f"mmap must be False, True, or 'require', not {mmap!r}")
        return cls(
            _TransportNetwork.load(os.fspath(path), mmap=modes[mmap], verify=verify)
        )

    @property
    def mapped(self):
        """Whether the street arrays are memory-mapped from the artifact."""
        return self._core.mapped

    @property
    def stop_count(self):
        """Number of stops in the network."""
        return self._core.stop_count

    @property
    def pattern_count(self):
        """Number of stop-sequence patterns in the network."""
        return self._core.pattern_count

    @property
    def trip_count(self):
        """Number of trips in the network."""
        return self._core.trip_count

    @property
    def transfer_count(self):
        """Number of installed stop-to-stop transfers."""
        return self._core.transfer_count

    @property
    def ultra_shortcut_count(self):
        """Number of ULTRA shortcuts, or ``None`` if none are computed."""
        return self._core.ultra_shortcut_count

    @property
    def ultra_shortcuts(self):
        """The ULTRA shortcuts as ``(origin_stop_id, destination_stop_id,
        seconds, meters)`` tuples, or ``None`` if none are computed. Sorted
        by origin then destination, so the list is identical across runs
        over the same network."""
        return self._core.ultra_shortcuts()

    @property
    def has_tbtr_transfers(self):
        """Whether a cached time-only TBTR transfer set is present
        (see ``compute_tbtr_transfers``)."""
        return self._core.has_tbtr_transfers

    @property
    def tbtr_transfer_count(self):
        """Number of transfers in the cached time-only TBTR set, or
        ``None`` when none is computed (see ``compute_tbtr_transfers``)."""
        return self._core.tbtr_transfer_count

    @property
    def has_mctbtr_transfers(self):
        """Whether a cached multicriteria TBTR transfer set is present
        (see ``compute_mctbtr_transfers``)."""
        return self._core.has_mctbtr_transfers

    @property
    def mctbtr_transfer_count(self):
        """Number of transfers in the cached multicriteria TBTR set, or
        ``None`` when none is computed (see ``compute_mctbtr_transfers``)."""
        return self._core.mctbtr_transfer_count

    @property
    def stops(self):
        """The stops as ``(stop_id, latitude, longitude)`` tuples."""
        return self._core.stops

    @property
    def routes(self):
        """The routes as ``(route_id, agency_id, route_type)`` tuples,
        with the GTFS route_type as its numeric code."""
        return self._core.routes

    @property
    def trips(self):
        """The routable trips as ``(trip_id, route_id)`` tuples."""
        return self._core.trips

    def annotate_emissions(self, journeys, factors=None, components=None):
        """Attach per-leg and per-journey emissions to routed journeys.

        Parameters
        ----------
        journeys : list of dict
            Journeys from ``route_between_stops`` (with distances, the
            default build).
        factors : DataFrame or path (optional)
            Extra emission-factor rows layered over the shipped
            defaults; see ``cafein.emissions.load_factors``.
        components : list of str (optional)
            The life-cycle components to include (default: all four);
            see ``cafein.emissions.annotate``.

        Returns
        -------
        list of dict
            The journeys, with ``emissions`` (grams CO₂e) on every leg
            and journey; see ``cafein.emissions.annotate``.
        """
        components = component_selection(components)
        from cafein import emissions

        return emissions.annotate(journeys, self, factors, components)

    def set_transfers(self, footpaths):
        """Install precomputed stop-to-stop transfers.

        Parameters
        ----------
        footpaths : Footpaths, or list of (str, str, int, float)
            ``cafein.streets.walking_footpaths``'s array form — whose
            edges cross into the core whole, without per-edge Python
            objects — or ``(from_stop, to_stop, seconds, meters)``
            walking edges. Either way the edge set must be transitively
            closed: routing relaxes a single transfer hop per round.
        """
        from cafein.streets import Footpaths

        if isinstance(footpaths, Footpaths):
            self._core.set_transfer_arrays(
                footpaths.stop_ids,
                footpaths.from_index,
                footpaths.to_index,
                footpaths.seconds,
                footpaths.meters,
            )
        else:
            self._core.set_transfers(footpaths)

    def compute_ultra_shortcuts(
        self,
        *,
        walking_speed_kmph=None,
        max_transfer_time=30.0,
        min_departure="00:00",
        max_departure=None,
    ):
        """Compute the ULTRA intermediate-transfer shortcuts and store them.

        Enumerates, over the unrestricted stop-to-stop walking graph of
        the installed street network, the minimal set of intermediate
        transfers a Pareto-optimal two-trip journey needs (see the ULTRA
        preprocessing of Baum et al.). The network must be built with an
        OSM extract. The result is held in memory (``ultra_shortcut_count``,
        ``ultra_shortcuts``). Computed **for the whole service day** (the
        default window), it is relaxed by the **door-to-door time** queries
        in place of the closure footpaths, giving them unrestricted walking:
        the routing queries (``route_between_coordinates``,
        ``route_between_stops`` — which routes between the stops' coordinates),
        the one-to-all queries (``travel_times_from_stop``,
        ``travel_times_from_coordinate``, and the ``"raptor"``
        ``travel_time_matrix``, which treat a stop origin as its coordinate and
        add one ``final_transfers`` walk), and the point-set matrices
        (``TravelTimeMatrix``/``TravelCostMatrix`` from point origins and
        destinations, ``DetailedItineraries``). The **emissions/fare** queries
        keep the closure — ULTRA is not emissions-complete. A
        partial-window set (a narrower ``min_departure``/``max_departure``)
        is stored and inspectable but not relaxed by routing, since a
        journey's source departure can fall outside a bounded window. The
        set and its compute window are persisted by ``save`` and restored
        by ``load``, so a loaded partial-window set stays unused.

        A whole-day build over a metropolitan network is a heavy,
        run-once operation (minutes, parallel over cores); ``save`` it and
        reuse. Narrowing ``min_departure``/``max_departure`` bounds the
        source-departure set and costs proportionally less.

        Parameters
        ----------
        walking_speed_kmph : float (optional, default: 3.6)
            Walking speed in km/h of the shortcut search.
        max_transfer_time : float or datetime.timedelta (optional, default: 30)
            Cutoff of an intermediate walk, in minutes.
        min_departure : str or datetime.time (optional, default: "00:00")
            Earliest source-departure clock time to serve, as an
            ``"HH:MM"`` string or a ``datetime.time``.
        max_departure : str or datetime.time (optional)
            Latest source-departure clock time to serve, in the same
            forms; the whole service day by default.

        Returns
        -------
        int
            The number of shortcuts computed.
        """
        from cafein import streets
        from cafein._units import clock_time, duration_seconds

        def clock_seconds(name, value):
            hours, minutes, seconds = clock_time(name, value).split(":")
            return int(hours) * 3600 + int(minutes) * 60 + int(seconds)

        transfer_seconds = float(
            duration_seconds("max_transfer_time", max_transfer_time)
        )
        earliest = clock_seconds("min_departure", min_departure)
        if walking_speed_kmph is None:
            walking_speed_kmph = streets.WALKING_SPEED_KMPH
        if max_departure is None:
            return self._core.compute_ultra_shortcuts(
                walking_speed_kmph, transfer_seconds, earliest
            )
        return self._core.compute_ultra_shortcuts(
            walking_speed_kmph,
            transfer_seconds,
            earliest,
            clock_seconds("max_departure", max_departure),
        )

    def compute_tbtr_transfers(self, date):
        """Precompute and cache the trip-based (TBTR) transfer set for `date`.

        The dominance-aware transfer set is TBTR's amortised asset: caching it
        lets repeated stop ``travel_time_matrix(router="tbtr")`` calls on the
        same date — single-departure and windowed alike — reuse it instead of
        rebuilding it every call — the "build once, query many" workload the
        trip-based engine is built for. A query on a different date rebuilds ad
        hoc. The cache is persisted with the artifact (``save``/``load``) and
        re-keyed when computed for a new date; ``has_tbtr_transfers`` reports
        whether one is present.

        Parameters
        ----------
        date : str
            Service date as ``YYYY-MM-DD``.
        """
        return self._core.compute_tbtr_transfers(date)

    def compute_mctbtr_transfers(self, date, factors=None, components=None):
        """Precompute and cache the multicriteria TBTR transfer set.

        The factor-aware transfer set is McTBTR's amortised asset — it is
        far heavier to build than the time-only set, and every
        ``router="tbtr"`` multicriteria query (``journey_frontier``,
        ``journey_frontiers``, the emissions cost matrices) otherwise
        rebuilds it per call. Caching it keys on the date **and** the
        resolved per-trip factors: a query reuses the cache only when its
        ``factors``/``components`` resolve to the same configuration given
        here (the defaults match the defaults). Queries on another date or
        factor set rebuild ad hoc. The cache is persisted with the
        artifact (``save``/``load``) and replaced on recompute;
        ``has_mctbtr_transfers`` reports whether one is present.

        Parameters
        ----------
        date : str
            Service date as ``YYYY-MM-DD``.
        factors, components : optional
            Emission-factor rows layered over the shipped defaults and the
            LCA components to include, as in ``emissions.annotate`` — the
            same arguments the queries will use.
        """
        components = component_selection(components)
        from cafein import emissions

        trip_factors = emissions.trip_factors(self, factors, components)
        return self._core.compute_mctbtr_transfers(date, trip_factors)

    def set_leg_geometries(self, *leg_geometries):
        """Install per-trip leg geometries.

        Parameters
        ----------
        leg_geometries : tuple
            ``(polylines, trips)`` — deduplicated ``(longitudes,
            latitudes, measures)`` polylines and ``(trip_id, polyline,
            stop_positions)`` rows locating each stop of a trip along
            its polyline — as produced (alongside the distances) by
            ``cafein.geometry.trip_distances(..., geometries=True)``.
        """
        self._core.set_leg_geometries(*leg_geometries)

    def set_street_network(self, *street_network):
        """Install the street network for coordinate access/egress.

        Parameters
        ----------
        street_network : tuple
            ``(vertex_count, edges, coordinate_offsets, longitudes,
            latitudes, stop_links)``, as produced (alongside the
            footpaths) by ``cafein.streets.walking_streets``.
        """
        self._core.set_street_network(*street_network)

    def access_stops(
        self,
        lat,
        lon,
        *,
        walking_speed_kmph=None,
        max_walking_time=None,
        snap_distance=None,
    ):
        """Walking times to every transit stop reachable from a coordinate.

        Requires a network built with an OSM extract (``osm_pbf=``).
        Walking is undirected, so the same search serves access from an
        origin and egress to a destination.

        Parameters
        ----------
        lat, lon : float
            The coordinate, in EPSG:4326.
        walking_speed_kmph : float (optional, default: 3.6)
            Walking speed in km/h, on the network and on the connectors.
        max_walking_time : float or datetime.timedelta (optional, default: 120 minutes)
            Walking-time cutoff in minutes.
        snap_distance : float (optional, default: 1600)
            Maximum straight-line distance in meters from the coordinate
            to the walking network; a coordinate farther away raises
            ``ValueError``.

        Returns
        -------
        dict
            Walking time in seconds to each reachable stop, keyed by
            stop_id; stops beyond the cutoff are absent.
        """
        from cafein._units import duration_seconds

        max_walking_time = duration_seconds("max_walking_time", max_walking_time)
        max_snap_distance = snap_distance
        return self._core.access_stops(
            lat,
            lon,
            *_walk_options(walking_speed_kmph, max_walking_time, max_snap_distance),
        )

    def set_trip_distances(self, distances):
        """Install per-trip cumulative travel distances.

        Parameters
        ----------
        distances : list of (str, list of float, str)
            ``(trip_id, cumulative_meters, provenance)`` rows with one
            cumulative distance per stop of the trip;
            ``cafein.geometry.trip_distances`` produces such lists.
        """
        self._core.set_trip_distances(distances)

    @property
    def distance_provenance_counts(self):
        """Number of trips per distance-provenance tier (empty until
        trip distances are installed)."""
        return self._core.distance_provenance_counts

    def route_between_stops(
        self,
        origin,
        destination,
        departure,
        max_rides=8,
        departure_time_window=None,
        *,
        exclude_routes=(),
        exclude_trips=(),
        exclude_stops=(),
        walking_speed_kmph=None,
        max_walking_time=None,
        snap_distance=None,
        geometries=True,
    ):
        """Route between two transit stops.

        Journeys ride trips and change vehicles at shared stops or over
        the installed transfers; transit legs report their distance and
        its provenance when trip distances are installed.
        ``route_between_coordinates`` routes door-to-door from arbitrary
        coordinates, and ``annotate_emissions`` attaches emissions to
        routed journeys. Legs carry times, stops, distances, and
        provenance; transit legs add their geometry as a WKB LineString
        when leg geometries are installed (the default build), and
        transfer legs their walked street path when the street network
        is installed.

        With a whole-day ULTRA set (``compute_ultra_shortcuts``), the two
        stops are routed **door-to-door between their coordinates** — the
        same unrestricted initial/intermediate/final walking as
        ``route_between_coordinates`` — and ``walking_speed_kmph``,
        ``max_walking_time``, and ``snap_distance`` bound that walking.
        Without such a set (or when a stop has no coordinate or is off the
        walking network) the query boards at the origin stop and relaxes
        the closure transfers, and those three arguments are ignored.

        Parameters
        ----------
        origin : str
            GTFS stop_id of the origin stop; ``<feed_index>:<stop_id>``
            when the id occurs in several merged feeds.
        destination : str
            GTFS stop_id of the destination stop, qualified the same way.
            Identifiers in the output follow the same convention.
        departure : datetime.datetime or str
            Departure at the origin — a datetime, or an ISO string like
            ``"2022-02-22 08:30"``; the service date is its date part.
        max_rides : int (optional, default: 8)
            Maximum number of boarded vehicles per journey (rides, not
            transfers: 8 rides allow 7 transfers).
        departure_time_window : float or datetime.timedelta (optional)
            Departure window in minutes. When given, departures within
            the window are profiled: the result is the Pareto set of
            journeys over (departure, arrival, rides), each journey's
            departure being the latest time the origin can be left to
            catch it, sorted by departure and then rides. A journey that
            leaves within the window but waits for a ride beyond it
            carries the window's final second as its departure.
        walking_speed_kmph : float (optional, default: 3.6)
            Walking speed in km/h of the door-to-door searches (whole-day
            ULTRA only; ignored otherwise).
        max_walking_time : float or datetime.timedelta (optional, default: 120 minutes)
            Walking-time cutoff in minutes of each street search
            (whole-day ULTRA only; ignored otherwise).
        snap_distance : float (optional, default: 1600)
            Maximum straight-line distance in meters from a stop to the
            walking network (whole-day ULTRA only; ignored otherwise).
        exclude_routes, exclude_trips, exclude_stops : list of str (optional)
            GTFS ids of supply the journey must not use — for
            disruption scenarios and per-individual accessibility
            filters. An excluded stop refuses boarding, alighting,
            transfers, and access/egress, but vehicles still ride
            through it; an excluded origin or destination yields no
            journeys. Unknown route and trip ids are ignored.

        Returns
        -------
        list of dict
            Without ``departure_time_window``, the Pareto set of
            journeys over (arrival time, number of rides) leaving at the
            departure time; with it, the departure-window profile. Each
            journey carries its legs; the ``*_s`` times are seconds past
            the service day's start.
        """
        from cafein._units import departure_parts, duration_seconds

        date, departure = departure_parts(departure)
        if max_rides < 1:
            raise ValueError("max_rides must be at least 1")
        return self._route_between_stops(
            from_stop=origin,
            to_stop=destination,
            date=date,
            departure=departure,
            max_transfers=max_rides - 1,
            window=duration_seconds("departure_time_window", departure_time_window),
            max_walking_time=duration_seconds("max_walking_time", max_walking_time),
            max_snap_distance=snap_distance,
            exclude_routes=exclude_routes,
            exclude_trips=exclude_trips,
            exclude_stops=exclude_stops,
            walking_speed_kmph=walking_speed_kmph,
            geometries=geometries,
        )

    def _route_between_stops(
        self,
        from_stop,
        to_stop,
        date,
        departure,
        max_transfers=7,
        window=None,
        *,
        exclude_routes=(),
        exclude_trips=(),
        exclude_stops=(),
        walking_speed_kmph=None,
        max_walking_time=None,
        max_snap_distance=None,
        geometries=True,
    ):
        """``route_between_stops`` in core space: split ``date`` +
        ``departure`` strings, ``max_transfers``, and every duration in
        seconds — the form internal callers use to avoid double
        conversion."""
        return self._core.route_between_stops(
            from_stop,
            to_stop,
            date,
            departure,
            max_transfers,
            window,
            list(id_sequence("exclude_routes", exclude_routes)),
            list(id_sequence("exclude_trips", exclude_trips)),
            list(id_sequence("exclude_stops", exclude_stops)),
            *_walk_options(walking_speed_kmph, max_walking_time, max_snap_distance),
            geometries,
        )

    def route_between_coordinates(
        self,
        origin,
        destination,
        departure,
        max_rides=8,
        departure_time_window=None,
        *,
        exclude_routes=(),
        exclude_trips=(),
        exclude_stops=(),
        walking_speed_kmph=None,
        max_walking_time=None,
        snap_distance=None,
        geometries=True,
        street_policy=None,
    ):
        """Route door-to-door between two coordinates.

        Requires a network built with an OSM extract (``osm_pbf=``): the
        street network provides walking access from the origin to nearby
        stops and egress from stops to the destination. Journeys
        otherwise behave as in ``route_between_stops``; access and
        egress legs report their walking distance in meters and — with
        `geometries`, the default — their walked street path as WKB
        LineStrings. Walking all the way is a journey too: within
        `max_walking_time` the result leads with a walking-only journey
        — a single ``walk`` leg, zero rides — and a journey is dropped
        when walking out at that journey's own departure would arrive
        no later than it does.

        Parameters
        ----------
        origin, destination : (float, float)
            ``(lat, lon)`` coordinates, in EPSG:4326. A coordinate
            farther than `snap_distance` from the walking network
            raises ``ValueError``.
        departure : datetime.datetime or str
            Departure at the origin coordinate — a datetime, or an ISO
            string like ``"2022-02-22 08:30"``; the service date is its
            date part.
        max_rides : int (optional, default: 8)
            Maximum number of boarded vehicles per journey (rides, not
            transfers: 8 rides allow 7 transfers).
        departure_time_window : float or datetime.timedelta (optional)
            Departure window in minutes, as in ``route_between_stops``.
        walking_speed_kmph : float (optional, default: 3.6)
            Walking speed in km/h of the access and egress searches.
        max_walking_time : float or datetime.timedelta (optional, default: 120 minutes)
            Walking-time cutoff in minutes of each street search.
        snap_distance : float (optional, default: 1600)
            Maximum straight-line distance in meters from each
            coordinate to the walking network.
        exclude_routes, exclude_trips, exclude_stops : list of str (optional)
            GTFS ids of supply the journey must not use — for
            disruption scenarios and per-individual accessibility
            filters. An excluded stop refuses boarding, alighting,
            transfers, and access/egress, but vehicles still ride
            through it; an excluded origin or destination yields no
            journeys. Unknown route and trip ids are ignored.
        street_policy : StreetLegPolicy, optional
            Which street modes may serve the access and egress, on what
            vehicle terms (``cafein.StreetLegPolicy``). A walking-only
            policy at one shared budget is the current walking path at
            that budget; anything else runs the per-stop time-only
            reduction over the multimodal graph (build with
            ``street_modes=``) and rebuilds the street legs from the
            winning choices — each such leg carries an additional
            ``mode`` beside its exact distances (``network_distance_m``
            and ``connector_distance_m`` parts included), the street
            distance provenance, and its shape, and a choice carried
            through the transfer closure splits into the vehicle leg to
            its seed stop plus the walked transfer. With
            ``transfers={mode: budget}`` (one shared mode; compute the
            set first with :meth:`compute_mode_transfers`) the run
            relaxes the merged mode-transfer set, and a transfer whose
            edge rode a rental splits into its walk--ride--walk legs.
            The direct door-to-door alternative stays the walking one.
            Conflicts with the walking knobs above and with
            ``departure_time_window``, which are rejected rather than
            silently ignored.

        Returns
        -------
        list of dict
            Journeys as in ``route_between_stops``; arrivals include
            the egress walk.
        """
        from cafein._units import departure_parts, duration_seconds

        date, departure = departure_parts(departure)
        if max_rides < 1:
            raise ValueError("max_rides must be at least 1")
        return self._route_between_coordinates(
            origin=origin,
            destination=destination,
            date=date,
            departure=departure,
            max_transfers=max_rides - 1,
            window=duration_seconds("departure_time_window", departure_time_window),
            max_walking_time=duration_seconds("max_walking_time", max_walking_time),
            max_snap_distance=snap_distance,
            exclude_routes=exclude_routes,
            exclude_trips=exclude_trips,
            exclude_stops=exclude_stops,
            walking_speed_kmph=walking_speed_kmph,
            geometries=geometries,
            street_policy=street_policy,
        )

    def _route_between_coordinates(
        self,
        origin,
        destination,
        date,
        departure,
        max_transfers=7,
        window=None,
        *,
        exclude_routes=(),
        exclude_trips=(),
        exclude_stops=(),
        walking_speed_kmph=None,
        max_walking_time=None,
        max_snap_distance=None,
        geometries=True,
        street_policy=None,
    ):
        """``route_between_coordinates`` in core space: split ``date`` +
        ``departure`` strings, ``max_transfers``, and every duration
        in seconds — the form internal callers use to avoid double
        conversion."""
        if street_policy is not None:
            from cafein.matrices import _walking_only_policy

            if any(
                option is not None
                for option in (walking_speed_kmph, max_walking_time, max_snap_distance)
            ):
                raise ValueError(
                    "street_policy carries its own budgets; passing "
                    "walking_speed_kmph, max_walking_time, or "
                    "snap_distance beside it is a conflict"
                )
            if window is not None:
                raise ValueError(
                    "street_policy does not combine with a departure window yet"
                )
            from cafein.policy import carriage_terms

            if carriage_terms(street_policy) is not None:
                exclude_routes = id_sequence("exclude_routes", exclude_routes)
                exclude_trips = id_sequence("exclude_trips", exclude_trips)
                exclude_stops = id_sequence("exclude_stops", exclude_stops)
                if any((exclude_routes, exclude_trips, exclude_stops)):
                    raise ValueError(
                        "take_aboard=True does not combine with exclusions yet"
                    )
                return _carriage_journeys(
                    self._core,
                    origin,
                    destination,
                    date,
                    departure,
                    max_transfers,
                    street_policy,
                    geometries,
                )
            walk_only, walk_budget = _walking_only_policy(street_policy)
            if walk_only:
                # A walking-only policy IS the current walking path, at the
                # policy's walking budget.
                return self._core.route_between_coordinates(
                    tuple(origin),
                    tuple(destination),
                    date,
                    departure,
                    max_transfers,
                    None,
                    list(id_sequence("exclude_routes", exclude_routes)),
                    list(id_sequence("exclude_trips", exclude_trips)),
                    list(id_sequence("exclude_stops", exclude_stops)),
                    *_walk_options(None, walk_budget, None),
                    geometries,
                )
            if not self._core.has_multimodal_streets:
                raise ValueError(
                    "street_policy needs the multimodal street graph; build "
                    "with street_modes="
                )
            return _policy_journeys(
                self._core,
                origin,
                destination,
                date,
                departure,
                max_transfers,
                street_policy,
                (exclude_routes, exclude_trips, exclude_stops),
                geometries,
            )
        return self._core.route_between_coordinates(
            tuple(origin),
            tuple(destination),
            date,
            departure,
            max_transfers,
            window,
            list(id_sequence("exclude_routes", exclude_routes)),
            list(id_sequence("exclude_trips", exclude_trips)),
            list(id_sequence("exclude_stops", exclude_stops)),
            *_walk_options(walking_speed_kmph, max_walking_time, max_snap_distance),
            geometries,
        )

    def compute_carriage_transfers(self, mode, max_transfer_time):
        """Compute the carriage transfer set for a carried ``mode``.

        Per stop pair the faster of the walking closure row and the own
        vehicle's direct ride (ties to walking; ride-only pairs are
        added), each row a single mode. The budget bounds each ride as
        one movement, and queries granting ``transfers={mode:
        max_transfer_time}`` beside a carried vehicle relax exactly this set
        — a missing or differently parameterised set is an error, never
        a silent fallback. Heavy precompute; persisted by ``save`` and
        restored by ``load`` with its exact binding, and dropped when
        the walking closure or the multimodal graph is replaced.

        Parameters
        ----------
        mode : str
            The carried vehicle's street mode (``"bicycle"``).
        max_transfer_time : float or datetime.timedelta
            The per-movement ride budget, in minutes.

        Returns
        -------
        (int, int)
            Total edges in the merged set, and how many are rides.
        """
        from cafein._units import duration_seconds

        seconds = duration_seconds("max_transfer_time", max_transfer_time)
        return self._core._compute_carriage_transfers(mode, float(seconds))

    def compute_mode_transfers(self, mode, max_transfer_time):
        """Compute the merged shared-vehicle transfer set for ``mode``.

        Per stop with a street link for `mode`, one directed search over
        the multimodal graph collects the rental rides to every other
        link within `max_transfer_time` (minutes, or a timedelta); they
        merge into the installed walking
        closure under the one-rental-per-transfer contract — the budget
        bounds a rental-bearing transfer's whole movement (pre-walk,
        ride, post-walk), while pure walking transfers keep the
        installed set's own budget, so the merged set is never weaker
        than the walking one. Queries with
        ``StreetLegPolicy(transfers={mode: budget})`` — the same budget,
        in the policy's seconds — then relax
        this set; a missing or differently bound set is rejected, never
        silently substituted. Heavy precompute; persisted by ``save``
        and restored by ``load`` with its exact binding.

        Returns
        -------
        (int, int)
            The merged set's edge count and how many edges ride a
            rental.
        """
        from cafein._units import duration_seconds

        seconds = duration_seconds("max_transfer_time", max_transfer_time)
        return self._core._compute_mode_transfers(mode, float(seconds))

    def travel_times_from_coordinate(
        self,
        origin,
        departure,
        max_rides=8,
        *,
        exclude_routes=(),
        exclude_trips=(),
        exclude_stops=(),
        walking_speed_kmph=None,
        max_walking_time=None,
        snap_distance=None,
        street_policy=None,
    ):
        """Earliest arrival at every reachable stop from a coordinate.

        The counterpart of ``travel_times_from_stop`` for a coordinate
        origin: walking access from the coordinate seeds one RAPTOR run
        that serves all destinations. Requires a network built with an
        OSM extract (``osm_pbf=``). Stops within the walking cutoff
        appear with their walking time even without riding.

        Parameters
        ----------
        origin : (float, float)
            ``(lat, lon)`` coordinate, in EPSG:4326. A coordinate
            farther than `snap_distance` from the walking network
            raises ``ValueError``.
        departure : datetime.datetime or str
            Departure at the origin coordinate — a datetime, or an ISO
            string like ``"2022-02-22 08:30"``; the service date is its
            date part.
        max_rides : int (optional, default: 8)
            Maximum number of boarded vehicles per journey (rides, not
            transfers: 8 rides allow 7 transfers).
        walking_speed_kmph : float (optional, default: 3.6)
            Walking speed in km/h of the access search.
        max_walking_time : float or datetime.timedelta (optional, default: 120 minutes)
            Walking-time cutoff in minutes of the access search.
        snap_distance : float (optional, default: 1600)
            Maximum straight-line distance in meters from the
            coordinate to the walking network.
        street_policy : StreetLegPolicy, optional
            Which street modes may serve the access, on what vehicle
            terms (``cafein.StreetLegPolicy``). A walking-only policy is
            the current walking path at the policy's budget; an omitted
            side means walking at the usual 7200 s budget. Non-walking
            modes need the multimodal graph (build with
            ``street_modes=``) and run the per-stop time-only reduction
            over it. With ``transfers={mode: budget}`` the run relaxes
            the merged mode-transfer set of
            :meth:`compute_mode_transfers`, which must be computed with
            exactly that binding. Conflicts with the walking knobs above
            and with exclusions, which are rejected rather than
            silently ignored.

        Returns
        -------
        dict
            Travel time in seconds to every reachable stop, keyed by
            stop_id; unreachable stops are absent.
        """
        from cafein._units import departure_parts, duration_seconds

        date, departure = departure_parts(departure)
        if max_rides < 1:
            raise ValueError("max_rides must be at least 1")
        max_transfers = max_rides - 1
        max_walking_time = duration_seconds("max_walking_time", max_walking_time)
        max_snap_distance = snap_distance
        if street_policy is not None:
            from cafein import streets as _streets
            from cafein.policy import reduction_modes

            if any(
                option is not None
                for option in (walking_speed_kmph, max_walking_time, max_snap_distance)
            ):
                raise ValueError(
                    "street_policy carries its own budgets; passing "
                    "walking_speed_kmph, max_walking_time, or "
                    "snap_distance beside it is a conflict"
                )
            exclude_routes = id_sequence("exclude_routes", exclude_routes)
            exclude_trips = id_sequence("exclude_trips", exclude_trips)
            exclude_stops = id_sequence("exclude_stops", exclude_stops)
            if any((exclude_routes, exclude_trips, exclude_stops)):
                raise ValueError("street_policy does not combine with exclusions yet")
            from cafein.network import _policy_transfer_mode
            from cafein.policy import carriage_terms

            modes = reduction_modes(
                street_policy, "access", _streets.MAX_ACCESS_EGRESS_TIME
            )
            transfer_mode = _policy_transfer_mode(street_policy)
            carriage = carriage_terms(street_policy)
            if carriage is not None:
                # The possession-state search: Carrying seeds from the
                # policy reduction, Free seeds from the walking-only
                # reduction — carriage is optional, so every journey
                # without the vehicle stays available.
                mode, vehicle = carriage
                origin = tuple(origin)
                if not self._core.has_multimodal_streets:
                    raise ValueError(
                        "street_policy needs the multimodal street graph; "
                        "build with street_modes="
                    )
                # Snapshot every policy term before the GIL-releasing
                # street searches; the query reads the policy once.
                unknown_rule = vehicle.unknown_bike_trips
                park = (
                    None
                    if vehicle.facilities == "any_stop"
                    else [str(stop) for stop in vehicle.facilities]
                )
                # A carried vehicle's facilities govern parking only:
                # its access may end at any stop (the bicycle boards
                # along), so the reduction runs unmasked for it. A side
                # that snaps for neither plane is fatal; one plane may
                # snap alone (the other seeds empty).
                from cafein.policy import carriage_plane_modes

                carrying_modes, free_modes = carriage_plane_modes(
                    street_policy, "access", _streets.MAX_ACCESS_EGRESS_TIME
                )

                def reduced(plane_modes, carrying_plane=False):
                    try:
                        rows = self._core._reduced_street_offsets(
                            *origin, False, plane_modes
                        )
                    except ValueError as error:
                        if "too far from the multimodal street network" not in str(
                            error
                        ):
                            raise
                        return None
                    if carrying_plane:
                        return _carrying_offsets(rows)
                    return [(stop, seconds) for stop, seconds, *_ in rows]

                carrying = reduced(carrying_modes, carrying_plane=True)
                free = reduced(free_modes)
                if carrying is None and free is None:
                    raise ValueError(
                        "coordinate too far from the multimodal street "
                        "network for every policy mode"
                    )
                return self._core._carriage_travel_times(
                    carrying or [],
                    free or [],
                    date,
                    departure,
                    max_transfers,
                    unknown_rule,
                    park_stops=park,
                    transfer_mode=transfer_mode,
                )
            if transfer_mode is None and all(mode == "walk" for mode, *_ in modes):
                # A walking-only policy IS the current walking path, at the
                # policy's walking budget; a transfers= binding changes the
                # relaxed set, so it never takes this shortcut.
                walk_budget = next(s for mode, s, *_ in modes if mode == "walk")
                return self._core.travel_times_from_coordinate(
                    tuple(origin),
                    date,
                    departure,
                    max_transfers,
                    [],
                    [],
                    [],
                    *_walk_options(walking_speed_kmph, walk_budget, max_snap_distance),
                )
            if not self._core.has_multimodal_streets:
                raise ValueError(
                    "street_policy needs the multimodal street graph; build "
                    "with street_modes="
                )
            access = [
                (stop, seconds)
                for stop, seconds, *_ in self._core._reduced_street_offsets(
                    *tuple(origin), False, modes, transfer_mode=transfer_mode
                )
            ]
            return self._core._travel_times_with_access(
                access, date, departure, max_transfers, transfer_mode=transfer_mode
            )
        return self._core.travel_times_from_coordinate(
            tuple(origin),
            date,
            departure,
            max_transfers,
            list(id_sequence("exclude_routes", exclude_routes)),
            list(id_sequence("exclude_trips", exclude_trips)),
            list(id_sequence("exclude_stops", exclude_stops)),
            *_walk_options(walking_speed_kmph, max_walking_time, max_snap_distance),
        )

    def travel_times_from_stop(
        self,
        origin,
        departure,
        max_rides=8,
        *,
        exclude_routes=(),
        exclude_trips=(),
        exclude_stops=(),
        walking_speed_kmph=None,
        max_walking_time=None,
        snap_distance=None,
    ):
        """Earliest arrival at every reachable stop for a single departure.

        One RAPTOR run serves all destinations, so travel-time matrices
        are assembled origin by origin from this method — never per OD
        pair.

        With a whole-day ULTRA set (``compute_ultra_shortcuts``) the origin
        stop is treated as its coordinate and every stop is reached
        door-to-door — unrestricted initial, intermediate, and final walking,
        bounded by the three walking arguments; without such a set the search
        boards at the origin stop over the closure and those arguments are
        ignored.

        Parameters
        ----------
        origin : str
            GTFS stop_id of the origin stop; ``<feed_index>:<stop_id>``
            when the id occurs in several merged feeds.
        departure : datetime.datetime or str
            Departure at the origin — a datetime, or an ISO string like
            ``"2022-02-22 08:30"``; the service date is its date part.
        max_rides : int (optional, default: 8)
            Maximum number of boarded vehicles per journey (rides, not
            transfers: 8 rides allow 7 transfers).
        walking_speed_kmph : float (optional, default: 3.6)
            Walking speed in km/h of the door-to-door searches (whole-day
            ULTRA only; ignored otherwise).
        max_walking_time : float or datetime.timedelta (optional, default: 120 minutes)
            Walking-time cutoff in minutes of the initial and final
            walks (whole-day ULTRA only; ignored otherwise).
        snap_distance : float (optional, default: 1600)
            Maximum straight-line distance in meters from the origin stop to
            the walking network (whole-day ULTRA only; ignored otherwise).

        Returns
        -------
        dict
            Travel time in seconds to every reachable stop, keyed by
            stop_id; unreachable stops are absent. On the closure path the
            origin maps to 0; under a whole-day ULTRA set it is the
            door-to-door time from the origin stop's coordinate, so the origin
            may cost its short walk to the platform.
        """
        from cafein._units import departure_parts, duration_seconds

        date, departure = departure_parts(departure)
        from_stop = origin
        if max_rides < 1:
            raise ValueError("max_rides must be at least 1")
        max_transfers = max_rides - 1
        max_walking_time = duration_seconds("max_walking_time", max_walking_time)
        max_snap_distance = snap_distance
        return self._core.travel_times_from_stop(
            from_stop,
            date,
            departure,
            max_transfers,
            list(id_sequence("exclude_routes", exclude_routes)),
            list(id_sequence("exclude_trips", exclude_trips)),
            list(id_sequence("exclude_stops", exclude_stops)),
            *_walk_options(walking_speed_kmph, max_walking_time, max_snap_distance),
        )

    def travel_time_matrix(
        self,
        origins,
        departure,
        max_rides=8,
        *,
        exclude_routes=(),
        exclude_trips=(),
        exclude_stops=(),
        destinations=None,
        departure_time_window=None,
        percentiles=None,
        confidence=None,
        chunk=None,
        router="auto",
        walking_speed_kmph=None,
        max_walking_time=None,
        snap_distance=None,
    ):
        """Travel times as a matrix, from stops or from points.

        One RAPTOR run serves each origin, computed in parallel across
        the origins with per-worker state reuse; the result is
        deterministic. This is the bulk primitive travel-time matrices
        are assembled from — never per OD pair.

        With `departure_time_window`, every minute mark within the
        window is evaluated through one descending range
        scan per origin, and the output holds nearest-rank percentiles
        of the travel-time distribution across the window — exact
        values, since the samples are the full minute-level departure
        population. `percentiles` selects them (default: the median);
        `confidence` instead maps a level to the symmetric interval
        plus the median (e.g. ``0.8`` → the 10th, 50th, and 90th
        percentiles), quantifying travel-time variability due to
        departure time within the window.

        Parameters
        ----------
        origins : list of str, or GeoDataFrame
            GTFS stop_ids of the origin stops
            (``<feed_index>:<stop_id>`` when an id occurs in several
            merged feeds), or a point GeoDataFrame with an ``id``
            column. Points are linked once against the street network
            (requires ``osm_pbf=`` at build time); points off the
            walking network are reported with a warning and stay
            unreachable. Polygon frames route from their centroids
            (``centroid_lat``/``centroid_lon`` columns when present —
            the ``cafein.zones`` protocol — otherwise local-UTM
            centroids). Point cells hold the faster of transit and
            walking directly (within ``max_walking_time``), so a pair
            best covered on foot reports its walking time.
        departure : datetime.datetime or str
            Departure at every origin — a datetime, or an ISO string
            like ``"2022-02-22 08:30"``; the service date is its date
            part.
        max_rides : int (optional, default: 8)
            Maximum number of boarded vehicles per journey (rides, not
            transfers: 8 rides allow 7 transfers).
        destinations : GeoDataFrame (optional)
            Destination points; defaults to the origins. Only valid
            with point origins — stop origins always span every stop.
        departure_time_window : float or datetime.timedelta (optional)
            Departure window in minutes; enables percentile output.
        percentiles : list of float (optional)
            Percentiles in ``[0, 100]`` over the window's departures;
            requires `departure_time_window`, defaults to ``[50]``.
        confidence : float (optional)
            A level in ``(0, 1)`` mapped to the symmetric percentile
            interval plus the median; requires `departure_time_window`
            and excludes `percentiles`.
        chunk : (int, int) (optional)
            Compute only origin chunk ``k`` of ``n``: a deterministic
            contiguous block of the resolved origins, so ``n`` batch
            jobs cover all origins disjointly; rows follow the chunk.
        router : str (optional, default: "auto")
            The routing engine: ``"raptor"``, or ``"tbtr"`` to
            precompute a TBTR day engine (Trip-Based Transit Routing:
            Witt's trip-transfer set) for the date and fan the origins
            out over it, for stop and point matrices alike. The results
            are identical. ``"auto"`` (the default) runs on TBTR when a
            cached transfer set (``compute_tbtr_transfers``) matches
            the date, except for stop matrices under a whole-day ULTRA
            set, where only the RAPTOR path routes door-to-door and
            auto prefers it; point matrices share the ULTRA set on
            both engines, so the cache alone decides there.
        walking_speed_kmph, max_walking_time, snap_distance : float
            The street-search options, as in ``access_stops``: speed in
            km/h, walking time in minutes (or a timedelta), snap
            distance in meters. They apply to
            point origins, and to stop origins of the ``"raptor"`` matrix under
            a whole-day ULTRA set (which routes them door-to-door); they are
            ignored for stop origins otherwise.

        Returns
        -------
        numpy.ndarray
            A uint32 array of travel times in seconds — the exact
            engine values, unlike the minute-rounded frame computers —
            origins by all
            stops (column order follows ``stops``) for stop origins,
            origins by destination points for point origins; with
            `window`, one plane per percentile as a third axis, in the
            requested order (lower, median, upper for `confidence`).
            Unreachable pairs hold the maximum uint32 value
            (4294967295).
        """
        from cafein._units import departure_parts, duration_seconds

        date, departure = departure_parts(departure)
        from_stops = origins
        if max_rides < 1:
            raise ValueError("max_rides must be at least 1")
        max_transfers = max_rides - 1
        window = duration_seconds("departure_time_window", departure_time_window)
        max_walking_time = duration_seconds("max_walking_time", max_walking_time)
        max_snap_distance = snap_distance
        from_stops = sequence_not_string("origins", from_stops)
        if destinations is not None:
            destinations = sequence_not_string("destinations", destinations)
        matrix, _from_ids, _to_ids, _percentiles = self._time_matrix_with_ids(
            from_stops,
            date,
            departure,
            max_transfers,
            exclude_routes=exclude_routes,
            exclude_trips=exclude_trips,
            exclude_stops=exclude_stops,
            destinations=destinations,
            window=window,
            percentiles=percentiles,
            confidence=confidence,
            chunk=chunk,
            router=router,
            walking_speed_kmph=walking_speed_kmph,
            max_walking_time=max_walking_time,
            max_snap_distance=max_snap_distance,
        )
        return matrix

    def _time_matrix_with_ids(
        self,
        from_stops,
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
        """The travel-time matrix with its origin and destination id
        axes and the resolved percentile list (``None`` without a
        window). Backs both ``travel_time_matrix`` and the
        ``TravelTimeMatrix`` long-format wrapper, so the two share one
        origin/destination resolution.
        """
        from cafein.matrices import (
            _chunk_slice,
            _is_point_frame,
            _point_list,
            _warn_unsnapped,
        )

        if router not in ("auto", "raptor", "tbtr"):
            raise ValueError(
                f"router must be 'auto', 'raptor', or 'tbtr', not {router!r}"
            )
        percentiles = _window_percentiles(window, percentiles, confidence)
        if _is_point_frame(from_stops):
            from_ids, origin_points = _point_list(from_stops, "origins")
            if destinations is None:
                to_ids, destination_points = from_ids, origin_points
            else:
                to_ids, destination_points = _point_list(destinations, "destinations")
            rows = _chunk_slice(len(from_ids), chunk)
            from_ids = from_ids[rows]
            origin_points = origin_points[rows]
            walk = _walk_options(
                walking_speed_kmph, max_walking_time, max_snap_distance
            )
            if percentiles is None:
                table = self._core.travel_time_matrix_from_points(
                    origin_points,
                    destination_points,
                    date,
                    departure,
                    max_transfers,
                    router,
                    list(id_sequence("exclude_routes", exclude_routes)),
                    list(id_sequence("exclude_trips", exclude_trips)),
                    list(id_sequence("exclude_stops", exclude_stops)),
                    *walk,
                )
            else:
                table = self._core.travel_time_percentiles_from_points(
                    origin_points,
                    destination_points,
                    date,
                    departure,
                    window,
                    percentiles,
                    max_transfers,
                    router,
                    list(id_sequence("exclude_routes", exclude_routes)),
                    list(id_sequence("exclude_trips", exclude_trips)),
                    list(id_sequence("exclude_stops", exclude_stops)),
                    *walk,
                )
            _warn_unsnapped(table, from_ids, to_ids)
            return table["matrix"], from_ids, to_ids, percentiles
        if destinations is not None:
            raise ValueError("destinations apply to point origins")
        to_ids = [stop for stop, _latitude, _longitude in self._core.stops]
        from_stops = list(to_ids) if from_stops is None else list(from_stops)
        from_stops = from_stops[_chunk_slice(len(from_stops), chunk)]
        if percentiles is None:
            # The walking options bound the door-to-door raptor matrix under a
            # whole-day ULTRA set; they are ignored on the closure path.
            matrix = self._core.travel_time_matrix(
                from_stops,
                date,
                departure,
                max_transfers,
                router,
                list(id_sequence("exclude_routes", exclude_routes)),
                list(id_sequence("exclude_trips", exclude_trips)),
                list(id_sequence("exclude_stops", exclude_stops)),
                *_walk_options(walking_speed_kmph, max_walking_time, max_snap_distance),
            )
        else:
            matrix = self._core.travel_time_percentiles(
                from_stops,
                date,
                departure,
                window,
                percentiles,
                max_transfers,
                router,
                list(id_sequence("exclude_routes", exclude_routes)),
                list(id_sequence("exclude_trips", exclude_trips)),
                list(id_sequence("exclude_stops", exclude_stops)),
            )
        return matrix, from_stops, to_ids, percentiles
