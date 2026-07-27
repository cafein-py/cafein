"""Street-leg policies for public-transport queries.

A :class:`StreetLegPolicy` says which street modes may serve a journey's
access and egress (and, later, its intermediate transfers), each with its
own time budget, and under which vehicle terms. It is one structured
object rather than a growing collection of loosely related mode arguments,
and it is deliberately explicit: an own vehicle names the one side it
serves and where it may be left or picked up, and a shared vehicle names
its availability — nothing is silently assumed.
"""

import math

STREET_POLICY_MODES = ("walk", "bicycle", "e_bike", "e_scooter")

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
        an eligible stop and used for egress. The same own vehicle cannot
        serve both ends.
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
        Must be ``False``: carrying a vehicle aboard changes the routing
        state space and arrives with a later stage.
    """

    def __init__(
        self,
        *,
        source,
        side=None,
        facilities=None,
        availability=None,
        take_aboard=False,
    ):
        if source not in ("own", "shared"):
            raise ValueError(f"source must be 'own' or 'shared', not {source!r}")
        if take_aboard:
            raise ValueError(
                "take_aboard=True (carrying the vehicle aboard) is not yet "
                "supported; it changes the routing state space"
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
        self.take_aboard = False

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
        ``e_bike``, and ``e_scooter``. An omitted dict means walking at
        the query's usual walking budget.
    transfers : dict, optional
        Not consumed yet — rejected if passed. Intermediate transfers ride
        the installed walking transfer set until shared intermediate legs
        arrive (a later stage).
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
            raise ValueError(
                "transfers= is not consumed yet: intermediate transfers ride "
                "the installed walking transfer set; per-mode transfer "
                "budgets arrive with shared intermediate legs"
            )
        self.transfers = None
        vehicles = dict(vehicles or {})
        for mode, policy in vehicles.items():
            if mode not in STREET_POLICY_MODES or mode == "walk":
                raise ValueError(f"vehicles= names an unknown vehicle mode {mode!r}")
            if not isinstance(policy, VehiclePolicy):
                raise ValueError(f"vehicles[{mode!r}] must be a VehiclePolicy")
        for side_name, budgets in (("access", self.access), ("egress", self.egress)):
            for mode in budgets or {}:
                if mode == "walk":
                    continue
                policy = vehicles.get(mode)
                if policy is None:
                    raise ValueError(
                        f"{side_name} mode {mode!r} needs vehicle terms; pass "
                        f"vehicles={{{mode!r}: VehiclePolicy(...)}}"
                    )
                if policy.source == "own":
                    served = "access" if policy.side == "origin" else "egress"
                    if side_name != served:
                        raise ValueError(
                            f"the own {mode} serves {served} only "
                            f"(side={policy.side!r}), so it cannot be an "
                            f"{side_name} mode"
                        )
        self.vehicles = vehicles

    def __repr__(self):
        return (
            f"StreetLegPolicy(access={self.access!r}, egress={self.egress!r}, "
            f"vehicles={self.vehicles!r})"
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
        if mode == "walk":
            modes.append((mode, seconds, False, None))
            continue
        terms = policy.vehicles[mode]
        eligible = None if terms.facilities == "any_stop" else list(terms.facilities)
        modes.append((mode, seconds, terms.source == "shared", eligible))
    return modes
