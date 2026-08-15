"""Street-leg policies for public-transport queries.

A :class:`StreetLegPolicy` says which street modes may serve a journey's
access and egress and — for shared vehicles, over the merged set of
``TransportNetwork.compute_mode_transfers`` — its intermediate
transfers, each with its own time budget, and under which vehicle
terms. It is one structured object rather than a growing collection of
loosely related mode arguments, and it is deliberately explicit: an own
vehicle names the one side it
serves and where it may be left or picked up, and a shared vehicle names
its availability — nothing is silently assumed.
"""

import math

STREET_POLICY_MODES = ("walk", "bicycle", "e_bike", "e_scooter", "wheelchair")

_OWN_SIDES = ("origin", "destination")


class VehiclePolicy:
    """The vehicle terms behind a non-walking street mode.

    Parameters
    ----------
    source : str
        ``"own"`` — the traveller's vehicle — or ``"shared"`` — a rental
        picked up and released per leg.
    side : str, optional
        Own vehicles only: the one end the vehicle serves. ``"origin"``
        means it is available at the origin, used for access, and left at
        an eligible stop; ``"destination"`` means it is pre-positioned at
        an eligible stop and used for egress. The same own vehicle
        serves both ends only when carried (``take_aboard=True``).
    facilities : iterable of str, or "any_stop"
        The eligible stops: where an own vehicle may be left (``side=
        "origin"``) or picked up (``side="destination"``), or where a
        shared vehicle may be picked up and dropped off. Pass stop ids, or
        the explicit modelling assumption ``"any_stop"`` — eligibility is
        never silently assumed.
    availability : str, optional
        Shared vehicles only: ``"unconstrained"``, the explicit assumption
        that a vehicle is always available at every eligible stop. (An
        availability snapshot arrives with a later stage.)
    take_aboard : bool
        Own vehicles on ``side="origin"`` only: carry the vehicle
        aboard bike-permitted trips (the carriage stage). The routing
        state space gains possession states; see
        ``unknown_bike_trips`` for trips whose GTFS ``bikes_allowed``
        says nothing.
    unknown_bike_trips : str, optional
        With ``take_aboard=True`` only: how trips without a GTFS
        ``bikes_allowed`` value board while carrying —
        ``"forbid"`` (the conservative default) or ``"allow"`` (the
        explicit modelling assumption). Never silently assumed.
    """

    def __init__(
        self,
        *,
        source,
        side=None,
        facilities=None,
        availability=None,
        take_aboard=False,
        unknown_bike_trips=None,
    ):
        if source not in ("own", "shared"):
            raise ValueError(f"source must be 'own' or 'shared', not {source!r}")
        if take_aboard:
            if source != "own":
                raise ValueError(
                    "take_aboard applies to own vehicles; a shared rental "
                    "is picked up and dropped per leg, never carried"
                )
            if side != "origin":
                raise ValueError(
                    "a carried vehicle starts with the traveller; "
                    "take_aboard=True requires side='origin' "
                    "(destination pre-positioning contradicts carriage)"
                )
            if unknown_bike_trips is None:
                unknown_bike_trips = "forbid"
            if unknown_bike_trips not in ("forbid", "allow"):
                raise ValueError(
                    "unknown_bike_trips must be 'forbid' or 'allow' — how "
                    "trips without a GTFS bikes_allowed value board while "
                    "carrying is never silently assumed"
                )
        elif unknown_bike_trips is not None:
            raise ValueError(
                "unknown_bike_trips applies to carried vehicles only; pass "
                "take_aboard=True"
            )
        if source == "own":
            if side not in _OWN_SIDES:
                raise ValueError(
                    "an own vehicle serves exactly one declared side; pass "
                    "side='origin' or side='destination'"
                )
            if availability is not None:
                raise ValueError("availability applies to shared vehicles only")
        else:
            if side is not None:
                raise ValueError(
                    "shared vehicles may serve either end; side= applies to "
                    "own vehicles only"
                )
            if availability != "unconstrained":
                raise ValueError(
                    "a shared vehicle needs its availability stated; pass "
                    "availability='unconstrained' for the explicit "
                    "always-available assumption"
                )
        if facilities is None:
            raise ValueError(
                "eligible stops are never silently assumed; pass facilities= "
                "as stop ids or the explicit 'any_stop' assumption"
            )
        if isinstance(facilities, str):
            if facilities != "any_stop":
                raise ValueError(
                    f"facilities={facilities!r} is not a known selector; "
                    "named facility masks (e.g. bicycle parking data) arrive "
                    "with stop-facility inputs — pass explicit stop ids or "
                    "the 'any_stop' assumption"
                )
        else:
            facilities = tuple(str(stop) for stop in facilities)
            if not facilities:
                raise ValueError("facilities names no stops")
        self.source = source
        self.side = side
        self.facilities = facilities
        self.availability = availability
        self.take_aboard = bool(take_aboard)
        self.unknown_bike_trips = unknown_bike_trips

    def __repr__(self):
        terms = f"source={self.source!r}"
        if self.side is not None:
            terms += f", side={self.side!r}"
        return f"VehiclePolicy({terms})"


class StreetLegPolicy:
    """Which street modes serve access and egress, and on what terms.

    Parameters
    ----------
    access, egress : dict, optional
        ``{mode: seconds}`` time budgets, from ``walk``, ``bicycle``,
        ``e_bike``, ``e_scooter``, and ``wheelchair`` (walking-class:
        no vehicle terms). An omitted dict means walking at
        the query's usual walking budget.
    transfers : dict, optional
        ``{mode: seconds}`` for stop-to-stop transfers mid-journey, one
        mode at a time: a **shared** mode (each transfer a complete
        pickup-travel-drop-off rental, no possession state needed), or
        the **carried own vehicle** (``take_aboard=True``), whose
        possession the carriage engine tracks. The
        budget bounds a rental-bearing transfer's whole movement —
        pre-walk, ride, and post-walk — or a carried vehicle's ride as
        one movement; pure walking transfers keep the installed set's
        own budget. Shared queries require the matching merged set
        precomputed with ``TransportNetwork.compute_mode_transfers``;
        the carried vehicle's set has its own precompute, arriving
        publicly with the carriage engine.
    vehicles : dict, optional
        ``{mode: VehiclePolicy}`` for every non-walking mode used. A
        non-walking mode without vehicle terms is rejected — whose bicycle
        it is, and where it may be left, changes the answer.
    """

    def __init__(self, *, access=None, egress=None, transfers=None, vehicles=None):
        # Omitted sides stay None — the query resolves them to walking at
        # its usual budget — while an explicitly empty dict is rejected
        # there as a policy that grants nothing.
        self.access = None if access is None else _validated_budgets("access", access)
        self.egress = None if egress is None else _validated_budgets("egress", egress)
        if transfers is not None:
            transfers = _validated_budgets("transfers", transfers)
            if "walk" in transfers:
                raise ValueError(
                    "walking transfers are the installed transfer set; "
                    "transfers= names a shared vehicle mode or the "
                    "carried own vehicle"
                )
            if len(transfers) > 1:
                raise ValueError(
                    "one transfer mode at a time for now; each transfer "
                    "set is computed per mode"
                )
        self.transfers = transfers
        vehicles = dict(vehicles or {})
        for mode, policy in vehicles.items():
            if mode not in STREET_POLICY_MODES or mode in ("walk", "wheelchair"):
                raise ValueError(f"vehicles= names an unknown vehicle mode {mode!r}")
            if not isinstance(policy, VehiclePolicy):
                raise ValueError(f"vehicles[{mode!r}] must be a VehiclePolicy")
        for side_name, budgets in (("access", self.access), ("egress", self.egress)):
            for mode in budgets or {}:
                if mode in ("walk", "wheelchair"):
                    # Walking-class modes ride no vehicle: no terms needed.
                    continue
                policy = vehicles.get(mode)
                if policy is None:
                    raise ValueError(
                        f"{side_name} mode {mode!r} needs vehicle terms; pass "
                        f"vehicles={{{mode!r}: VehiclePolicy(...)}}"
                    )
                if policy.source == "own" and not policy.take_aboard:
                    # A carried vehicle may serve both ends — carriage
                    # is what transports it; an uncarried own vehicle
                    # keeps its one declared side.
                    served = "access" if policy.side == "origin" else "egress"
                    if side_name != served:
                        raise ValueError(
                            f"the own {mode} serves {served} only "
                            f"(side={policy.side!r}) without carriage; "
                            f"take_aboard=True lets it serve {side_name} "
                            "too"
                        )
        for mode in self.transfers or {}:
            if mode == "wheelchair":
                # A walking-class transfer set: computed per mode like the
                # shared ones, ridden without vehicle terms.
                continue
            policy = vehicles.get(mode)
            if policy is None:
                raise ValueError(
                    f"transfer mode {mode!r} needs vehicle terms; pass "
                    f"vehicles={{{mode!r}: VehiclePolicy(...)}}"
                )
            if policy.source != "shared" and not policy.take_aboard:
                raise ValueError(
                    "own-vehicle transfers need possession state; carry "
                    "the vehicle with take_aboard=True, or grant a shared "
                    "mode — a shared transfer is a complete "
                    "pickup-travel-drop-off rental"
                )
            if policy.source == "shared" and policy.facilities != "any_stop":
                raise ValueError(
                    f"the merged transfer set rents {mode!r} at every "
                    "linked stop; a facility-masked set is not computed "
                    "yet, so shared transfer vehicles need "
                    "facilities='any_stop'"
                )
        carried = [mode for mode, vehicle in vehicles.items() if vehicle.take_aboard]
        if len(carried) > 1:
            raise ValueError(
                "one carried vehicle per policy; "
                f"{', '.join(sorted(carried))} all declare take_aboard=True"
            )
        if carried:
            mode = carried[0]
            if mode != "bicycle":
                raise ValueError(
                    "carriage is modelled for bicycles (GTFS bikes_allowed "
                    f"governs the boardings); a carried {mode} arrives once "
                    "its own carriage rules are modelled"
                )
            granted = {
                granted_mode
                for budgets in (self.access, self.egress, self.transfers)
                for granted_mode in (budgets or {})
                if granted_mode != "walk"
            }
            if granted - {mode}:
                extra = ", ".join(sorted(granted - {mode}))
                raise ValueError(
                    "a carriage policy grants the carried bicycle and "
                    f"walking only; mixing {extra} with a carried vehicle "
                    "arrives later"
                )
        self.vehicles = vehicles

    def __repr__(self):
        return (
            f"StreetLegPolicy(access={self.access!r}, egress={self.egress!r}, "
            f"transfers={self.transfers!r}, vehicles={self.vehicles!r})"
        )


def _validated_budgets(name, budgets):
    budgets = dict(budgets or {})
    for mode, seconds in budgets.items():
        if mode not in STREET_POLICY_MODES:
            raise ValueError(
                f"{name} names an unknown street mode {mode!r}; expected one "
                f"of {', '.join(STREET_POLICY_MODES)}"
            )
        seconds = float(seconds)
        if not (math.isfinite(seconds) and seconds > 0.0):
            raise ValueError(f"{name}[{mode!r}] must be a positive, finite time budget")
        budgets[mode] = seconds
    return budgets


def reduction_modes(policy, side, walking_budget):
    """The reduction's mode list for one journey side, in declared order.

    Each entry is ``(mode, max_seconds, paid_rental, eligible_stops)`` —
    the shape the core's time-only reduction consumes. Walking is never
    masked; a vehicle mode's eligible stops come from its terms
    (``None`` for the explicit ``any_stop`` assumption). A side the policy
    omitted resolves to walking at `walking_budget`, the query's usual
    walking cutoff.
    """
    budgets = policy.access if side == "access" else policy.egress
    if budgets is None:
        budgets = {"walk": float(walking_budget)}
    if not budgets:
        raise ValueError(f"street_policy grants no {side} modes")
    modes = []
    for mode, seconds in budgets.items():
        if mode in ("walk", "wheelchair"):
            modes.append((mode, seconds, False, None))
            continue
        terms = policy.vehicles[mode]
        eligible = None if terms.facilities == "any_stop" else list(terms.facilities)
        modes.append((mode, seconds, terms.source == "shared", eligible))
    return modes


def carriage_terms(policy):
    """The carried vehicle's ``(mode, VehiclePolicy)``, or ``None``."""
    for mode, vehicle in policy.vehicles.items():
        if vehicle.take_aboard:
            return mode, vehicle
    return None


def carriage_plane_modes(policy, side, default_walk):
    """One side's per-plane reduction mode lists for a carriage query.

    Carrying takes the policy's granted modes with the carried mode
    unmasked (facilities govern parking, never its access) and always
    walks — pushing the vehicle — at the walking budget; Free is the
    walking-only reduction at that budget (carriage is optional, so the
    no-vehicle journeys stay available). Reads the policy completely,
    so callers snapshot by calling this before any GIL-releasing
    search.
    """
    mode, _ = carriage_terms(policy)
    modes = reduction_modes(policy, side, default_walk)
    carrying = [
        (
            (name, seconds, rental, None)
            if name == mode
            else (name, seconds, rental, eligible)
        )
        for name, seconds, rental, eligible in modes
    ]
    budgets = (policy.access if side == "access" else policy.egress) or {}
    walk_budget = float(budgets.get("walk", default_walk))
    free = [("walk", walk_budget, False, None)]
    if all(name != "walk" for name, *_ in carrying):
        carrying.append(("walk", walk_budget, False, None))
    return carrying, free


def reject_carriage(policy, surface):
    """Loudly rejects a carriage policy on a surface that cannot route
    possession states yet."""
    if carriage_terms(policy) is not None:
        raise ValueError(
            f"take_aboard=True is not wired into {surface} yet; "
            "the time-candidate surfaces carry first"
        )


def pareto_reduction_modes(policy, side, walking_budget, factors=None, components=None):
    """The Pareto reduction's mode list for one journey side.

    The time-only list of :func:`reduction_modes` with each mode's
    resolved emission factor appended — the shape the ``(seconds,
    grams)`` frontier reduction consumes. The multicriteria search ranks
    street grams, so an unresolved vehicle factor is rejected here
    rather than silently zeroed or left unrankable; ``factors`` takes
    street-mode rows (see ``cafein.emissions.load_street_factors``).
    """
    import pandas as pd

    from cafein import emissions

    modes = []
    for mode, seconds, rental, eligible in reduction_modes(
        policy, side, walking_budget
    ):
        if mode == "walk":
            modes.append((mode, seconds, rental, eligible, 0.0))
            continue
        value = emissions.street_factor(
            mode,
            factors,
            components,
            service_model="shared" if rental else None,
        )
        if pd.isna(value):
            raise ValueError(
                f"the {mode} emission factor is unresolved; the "
                "multicriteria search ranks street emissions, so pass "
                "factors= rows resolving it (see "
                "cafein.emissions.load_street_factors)"
            )
        modes.append((mode, seconds, rental, eligible, float(value)))
    return modes
