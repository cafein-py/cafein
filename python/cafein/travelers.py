"""Per-traveler constraint profiles, compiled against a network."""

from cafein._validate import id_sequence


class TravelerProfile:
    """The constraints one traveler carries into every query.

    A declarative bundle passed as ``traveler=`` wherever the routing
    calls and computers take ``exclude_routes``/``exclude_trips``/
    ``exclude_stops``; it is compiled against the query's network at
    call time, so one profile serves any number of networks and calls.

    ``wheelchair=True`` compiles exclusions from the feed's GTFS
    accessibility fields: stops whose ``wheelchair_boarding`` and trips
    whose ``wheelchair_accessible`` mark them not accessible are
    excluded — stops refuse boarding, alighting, transfers, and
    access/egress while vehicles still ride through them, as with
    manual exclusions. A stop without a value inherits its parent
    station's known value at feed ingest. By default unknown supply
    stays usable (most feeds leave most stops unsaid);
    ``unknown="excluded"`` is the strict switch — only supply
    explicitly marked accessible remains, which on sparsely tagged
    feeds can exclude nearly everything. With ``wheelchair=False`` (the
    default) the feed's accessibility fields are not consulted and the
    profile carries only its speeds, walking budget and manual
    exclusions.

    The profile's ``exclude_*`` sequences union with any passed on the
    call itself — constraints add up. The walking knobs are singular:
    a profile that sets the walking speed (``speeds={"walk": …}``) or
    ``max_walking_time`` beside a call-level value for the same knob is
    rejected rather than silently resolved.

    On a door-to-door query over a network built with the
    ``"wheelchair"`` street mode, ``wheelchair=True`` also takes the
    streets: access, egress, and the direct street alternative ride
    the wheelchair profile — no stairs, ``wheelchair=no`` respected,
    slopes adjusting its time on DEM builds — via an internally synthesized
    street policy (an explicit ``street_policy=`` beside it is
    rejected, as is a walking speed: it rides ``speeds["wheelchair"]``).
    Mid-journey transfers ride the computed wheelchair transfer set
    when the network carries one — accessible end to end,
    excluded stops refused at the endpoints — and otherwise the
    installed walking closure with excluded endpoints refused, whose
    walked paths may cross stairs; compute the wheelchair set for the
    fully accessible configuration. ``Catchment`` spreads ride the
    compiled wheelchair profile over the multimodal graph; the other
    surfaces without street-policy support refuse the wheelchair
    traveler on point queries rather than walking silently, and stop
    queries everywhere stay timetable-only.

    Parameters
    ----------
    wheelchair : bool (optional, default: False)
        Compile the wheelchair exclusions from the feed's
        ``wheelchair_boarding`` and ``wheelchair_accessible`` fields.
    unknown : {"usable", "excluded"} (optional, default: "usable")
        Whether supply the feed says nothing about stays usable
        (default) or is excluded too — guaranteed-accessible routing.
        Applies only with ``wheelchair=True``.
    speeds : mapping (optional)
        Street speeds in km/h — ``"walk"``, ``"bicycle"``, ``"e_bike"``,
        ``"e_scooter"``, ``"wheelchair"`` — applied to every call made
        with this profile wherever the call routes that mode: the walking
        speed on any point query, the vehicles on a street-leg policy's
        legs, and ``"wheelchair"`` on a wheelchair traveler's streets. A
        mode set on both the profile and the call is rejected.
    max_walking_time : float or datetime.timedelta (optional)
        Walking-time budget in minutes, as on the calls themselves.
    max_uphill, max_downhill : float (optional)
        The steepest climb and descent a wheelchair traveler takes, as
        slope fractions between 0 and 1 (``0.085`` is 8.5 %), measured over
        each street segment's whole length on DEM builds: a segment longer
        than the measuring run that climbs (or descends) more is not used
        in that direction. Without them slope only costs time. Only with
        ``wheelchair=True``; a computed wheelchair transfer set serves only
        the same limits (``compute_mode_transfers``).
    exclude_stops, exclude_trips, exclude_routes : list of str (optional)
        GTFS ids the traveler must not use, merged into the compiled
        exclusions — personal no-go supply on top of the feed-derived
        constraints.
    """

    def __init__(
        self,
        *,
        wheelchair=False,
        unknown="usable",
        speeds=None,
        max_walking_time=None,
        max_uphill=None,
        max_downhill=None,
        exclude_stops=(),
        exclude_trips=(),
        exclude_routes=(),
    ):
        from cafein._units import duration_seconds
        from cafein._validate import mode_mapping, positive_finite
        from cafein.policy import STREET_POLICY_MODES
        from cafein.street_network import grade_limits

        if unknown not in ("usable", "excluded"):
            raise ValueError("unknown must be 'usable' or 'excluded'")
        speeds = mode_mapping("speeds", speeds, STREET_POLICY_MODES, positive_finite)
        limits = grade_limits(max_uphill, max_downhill)
        if limits and not wheelchair:
            raise ValueError(
                "max_uphill and max_downhill are a wheelchair traveler's grade "
                "limits; set wheelchair=True"
            )
        # Validation only; the raw value is handed to the call sites in
        # the unit their own conversion expects.
        duration_seconds("max_walking_time", max_walking_time)
        self.wheelchair = bool(wheelchair)
        self.unknown = unknown
        self.speeds = speeds
        self.max_walking_time = max_walking_time
        self.max_uphill = limits.get("wheelchair:max_uphill")
        self.max_downhill = limits.get("wheelchair:max_downhill")
        self.exclude_stops = tuple(id_sequence("exclude_stops", exclude_stops))
        self.exclude_trips = tuple(id_sequence("exclude_trips", exclude_trips))
        self.exclude_routes = tuple(id_sequence("exclude_routes", exclude_routes))

    def _resolve(self, network):
        """The profile's ``(routes, trips, stops)`` exclusion id lists
        compiled against `network`, manual exclusions included."""
        stops = list(self.exclude_stops)
        trips = list(self.exclude_trips)
        if self.wheelchair:
            strict = self.unknown == "excluded"
            for stop_id, flag in network._core.stop_wheelchair_boarding:
                if flag is False or (flag is None and strict):
                    stops.append(stop_id)
            for trip_id, flag in network._core.trip_wheelchair_accessible:
                if flag is False or (flag is None and strict):
                    trips.append(trip_id)
        return list(self.exclude_routes), trips, stops


def folded_street_policy(
    traveler,
    network,
    street_policy,
    walking_speed_kmph,
    max_walking_time,
):
    """The door-to-door query's street policy with the traveler's
    wheelchair folded in, and the walking-time knob it consumed.

    A non-wheelchair traveler (or none) passes the policy through. A
    wheelchair traveler synthesizes the wheelchair walking-class policy:
    access and egress at the walking budget, and — when the network
    carries a wheelchair mode-transfer set
    (``compute_mode_transfers("wheelchair", ...)``) — a matching
    transfer grant at the set's own budget, so mid-journey transfers
    ride wheelchair-profile movements with the traveler's excluded
    stops refused at the endpoints (a walking-class set has no interior
    stops, so exclusions combine soundly). Without a computed set,
    transfers ride the installed walking closure with excluded
    endpoints refused; that closure's walked paths may cross stairs,
    so computing the wheelchair set is the accessible-end-to-end
    configuration.
    Passing an explicit ``street_policy`` beside such a traveler is a
    conflict (a policy names its own modes), a network built without
    the wheelchair street mode raises with a rebuild hint, and
    a walking speed is rejected — the wheelchair profile rides
    ``speeds["wheelchair"]``. The synthesized policy carries the street-policy
    query restrictions: a ``departure_time_window`` on a door-to-door
    query is rejected beside it, exactly as beside any policy.
    Returns ``(street_policy, max_walking_time)`` with the walking
    budget consumed into the synthesized policy.
    """
    if traveler is None or not traveler.wheelchair:
        return street_policy, max_walking_time
    if street_policy is not None:
        raise ValueError(
            "a wheelchair traveler synthesizes its own street policy; "
            "passing street_policy= beside it is a conflict — grant the "
            "wheelchair mode in the policy instead"
        )
    if "wheelchair" not in (network._core.street_modes or ()):
        raise ValueError(
            "the wheelchair traveler routes the streets on the wheelchair "
            "mode; build the network with "
            "street_modes=('walk', 'wheelchair') (or load such an artifact)"
        )
    if walking_speed_kmph is not None:
        raise ValueError(
            "speeds['walk'] cannot reshape the wheelchair street "
            "profile, which rides speeds['wheelchair']"
        )
    from cafein import streets as _streets
    from cafein._units import duration_seconds
    from cafein.policy import StreetLegPolicy

    seconds = duration_seconds("max_walking_time", max_walking_time)
    budget = float(_streets.MAX_ACCESS_EGRESS_TIME if seconds is None else seconds)
    transfers = None
    binding = network._core._mode_transfer_binding
    if binding is not None and binding[0] == "wheelchair":
        transfers = {"wheelchair": binding[1]}
    policy = StreetLegPolicy(
        access={"wheelchair": budget},
        egress={"wheelchair": budget},
        transfers=transfers,
    )
    return policy, None


def refuse_wheelchair_streets(traveler, surface):
    """Refuses a wheelchair traveler on a point query of a surface with
    no street-policy support: its access and egress would silently ride
    the plain walking graph, stairs included. Stop queries pass — they
    have no street legs, and the timetable exclusions apply as usual."""
    if traveler is not None and traveler.wheelchair:
        raise ValueError(
            f"{surface} routes point queries over the plain walking "
            "streets and cannot honour the wheelchair traveler's street "
            "side; route between stops, or use a street_policy-capable "
            "computer (the matrices, the itineraries, or the routing "
            "calls)"
        )


def folded_constraints(
    traveler,
    network,
    exclude_routes,
    exclude_trips,
    exclude_stops,
    walking_speed_kmph,
    max_walking_time,
):
    """The call's exclusions and walking knobs with the profile folded
    in: profile exclusions union the call's, a walking knob set on both
    sides is a conflict."""
    if traveler is None:
        return (
            exclude_routes,
            exclude_trips,
            exclude_stops,
            walking_speed_kmph,
            max_walking_time,
        )
    if not isinstance(traveler, TravelerProfile):
        raise TypeError("traveler must be a cafein.TravelerProfile")
    routes, trips, stops = traveler._resolve(network)
    if "walk" in traveler.speeds:
        if walking_speed_kmph is not None:
            raise ValueError(
                "speeds['walk'] is set on both the call and the "
                "traveler profile; set it in one place"
            )
        walking_speed_kmph = traveler.speeds["walk"]
    if traveler.max_walking_time is not None:
        if max_walking_time is not None:
            raise ValueError(
                "max_walking_time is set on both the call and the "
                "traveler profile; set it in one place"
            )
        max_walking_time = traveler.max_walking_time
    return (
        tuple(id_sequence("exclude_routes", exclude_routes)) + tuple(routes),
        tuple(id_sequence("exclude_trips", exclude_trips)) + tuple(trips),
        tuple(id_sequence("exclude_stops", exclude_stops)) + tuple(stops),
        walking_speed_kmph,
        max_walking_time,
    )
