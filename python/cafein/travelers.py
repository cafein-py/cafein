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
    profile carries only its walking knobs and manual exclusions.

    The profile's ``exclude_*`` sequences union with any passed on the
    call itself — constraints add up. The walking knobs are singular:
    a profile that sets ``walking_speed_kmph`` or ``max_walking_time``
    beside a call-level value for the same knob is rejected rather
    than silently resolved.

    Parameters
    ----------
    wheelchair : bool (optional, default: False)
        Compile the wheelchair exclusions from the feed's
        ``wheelchair_boarding`` and ``wheelchair_accessible`` fields.
    unknown : {"usable", "excluded"} (optional, default: "usable")
        Whether supply the feed says nothing about stays usable
        (default) or is excluded too — guaranteed-accessible routing.
        Applies only with ``wheelchair=True``.
    walking_speed_kmph : float (optional)
        Walking speed in km/h, applied to every call made with this
        profile unless the call sets its own.
    max_walking_time : float or datetime.timedelta (optional)
        Walking-time budget in minutes, as on the calls themselves.
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
        walking_speed_kmph=None,
        max_walking_time=None,
        exclude_stops=(),
        exclude_trips=(),
        exclude_routes=(),
    ):
        from cafein._units import duration_seconds

        if unknown not in ("usable", "excluded"):
            raise ValueError("unknown must be 'usable' or 'excluded'")
        if walking_speed_kmph is not None and not walking_speed_kmph > 0:
            raise ValueError("walking_speed_kmph must be positive")
        # Validation only; the raw value is handed to the call sites in
        # the unit their own conversion expects.
        duration_seconds("max_walking_time", max_walking_time)
        self.wheelchair = bool(wheelchair)
        self.unknown = unknown
        self.walking_speed_kmph = walking_speed_kmph
        self.max_walking_time = max_walking_time
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
    if traveler.walking_speed_kmph is not None:
        if walking_speed_kmph is not None:
            raise ValueError(
                "walking_speed_kmph is set on both the call and the "
                "traveler profile; set it in one place"
            )
        walking_speed_kmph = traveler.walking_speed_kmph
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
