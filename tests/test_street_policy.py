"""StreetLegPolicy/VehiclePolicy validation and the time-only reduction."""

import pandas as pd
import pytest

from cafein import StreetLegPolicy, VehiclePolicy

pytestmark = []


def own(side="origin", facilities="any_stop", **terms):
    return VehiclePolicy(source="own", side=side, facilities=facilities, **terms)


def shared(facilities="any_stop"):
    return VehiclePolicy(
        source="shared", facilities=facilities, availability="unconstrained"
    )


def test_walking_only_policies_need_no_vehicle_terms():
    policy = StreetLegPolicy(access={"walk": 1800}, egress={"walk": 1800})
    assert policy.vehicles == {}


def test_unknown_modes_and_bad_budgets_are_rejected():
    with pytest.raises(ValueError, match="unknown street mode"):
        StreetLegPolicy(access={"segway": 600})
    with pytest.raises(ValueError, match="positive, finite time budget"):
        StreetLegPolicy(access={"walk": 0})
    with pytest.raises(ValueError, match="positive, finite time budget"):
        StreetLegPolicy(access={"walk": float("inf")})


def test_non_walk_modes_need_vehicle_terms():
    with pytest.raises(ValueError, match="vehicle terms"):
        StreetLegPolicy(access={"bicycle": 1800})


def test_an_own_vehicle_serves_exactly_one_declared_side():
    with pytest.raises(ValueError, match="one declared side"):
        VehiclePolicy(source="own", facilities="any_stop")
    # side='origin' serves access only.
    with pytest.raises(ValueError, match="serves access only"):
        StreetLegPolicy(egress={"bicycle": 1800}, vehicles={"bicycle": own()})
    policy = StreetLegPolicy(access={"bicycle": 1800}, vehicles={"bicycle": own()})
    assert policy.vehicles["bicycle"].side == "origin"


def test_eligibility_is_never_silently_assumed():
    with pytest.raises(ValueError, match="never silently assumed"):
        VehiclePolicy(source="own", side="origin")
    with pytest.raises(ValueError, match="names no stops"):
        VehiclePolicy(source="own", side="origin", facilities=[])
    terms = VehiclePolicy(source="own", side="origin", facilities=["A", "B"])
    assert terms.facilities == ("A", "B")


def test_shared_vehicles_state_their_availability():
    with pytest.raises(ValueError, match="availability stated"):
        VehiclePolicy(source="shared", facilities="any_stop")
    with pytest.raises(ValueError, match="own vehicles only"):
        VehiclePolicy(
            source="shared",
            side="origin",
            facilities="any_stop",
            availability="unconstrained",
        )
    policy = StreetLegPolicy(
        access={"e_scooter": 900},
        egress={"e_scooter": 900},
        vehicles={"e_scooter": shared()},
    )
    assert policy.vehicles["e_scooter"].availability == "unconstrained"


def test_carriage_terms_validate_and_routing_awaits_the_engine():
    # The carriage validation matrix: own + origin only, the unknown
    # rule explicit, shared never carries.
    carried = own(take_aboard=True)
    assert carried.take_aboard and carried.unknown_bike_trips == "forbid"
    assert own(take_aboard=True, unknown_bike_trips="allow").unknown_bike_trips == (
        "allow"
    )
    with pytest.raises(ValueError, match="never carried"):
        VehiclePolicy(
            source="shared",
            facilities="any_stop",
            availability="unconstrained",
            take_aboard=True,
        )
    with pytest.raises(ValueError, match="side='origin'"):
        own(side="destination", take_aboard=True)
    with pytest.raises(ValueError, match="unknown_bike_trips"):
        own(take_aboard=True, unknown_bike_trips="maybe")
    with pytest.raises(ValueError, match="take_aboard=True"):
        own(unknown_bike_trips="forbid")
    # Own transfers stay rejected without carriage, become legal terms
    # with it — and every query rejects until the carriage engine.
    with pytest.raises(ValueError, match="possession state"):
        StreetLegPolicy(transfers={"bicycle": 900}, vehicles={"bicycle": own()})
    # The terms build; the still-unwired surfaces reject them loudly
    # (see test_carriage_rejects_the_matrix_surfaces).
    StreetLegPolicy(
        access={"bicycle": 600},
        transfers={"bicycle": 900},
        vehicles={"bicycle": carried},
    )
    # A carried bicycle may serve both ends; an uncarried one cannot.
    StreetLegPolicy(
        access={"bicycle": 600},
        egress={"bicycle": 600},
        vehicles={"bicycle": carried},
    )
    with pytest.raises(ValueError, match="take_aboard=True lets it"):
        StreetLegPolicy(
            access={"bicycle": 600},
            egress={"bicycle": 600},
            vehicles={"bicycle": own()},
        )
    with pytest.raises(ValueError, match="carriage is modelled for bicycles"):
        StreetLegPolicy(
            access={"e_scooter": 300},
            vehicles={
                "e_scooter": VehiclePolicy(
                    source="own",
                    side="origin",
                    facilities="any_stop",
                    take_aboard=True,
                )
            },
        )
    with pytest.raises(ValueError, match="carried bicycle and walking only"):
        StreetLegPolicy(
            access={"bicycle": 600, "e_scooter": 300},
            vehicles={
                "bicycle": carried,
                "e_scooter": VehiclePolicy(
                    source="shared",
                    facilities="any_stop",
                    availability="unconstrained",
                ),
            },
        )
    with pytest.raises(ValueError, match="one carried vehicle"):
        StreetLegPolicy(
            access={"bicycle": 600, "e_scooter": 300},
            vehicles={
                "bicycle": carried,
                "e_scooter": VehiclePolicy(
                    source="own",
                    side="origin",
                    facilities="any_stop",
                    take_aboard=True,
                ),
            },
        )


# --- The time-only reduction over the multimodal graph -----------------------

ORIGIN = (60.1690, 24.9320)


def test_the_reduction_keeps_the_fastest_choice_per_stop(multimodal_network):
    pytest.importorskip("cafein._cafein")
    core = multimodal_network._core
    walk = {s: t for s, t, *_ in core._street_access_seconds(*ORIGIN, "walk", 900.0)}
    bike = {s: t for s, t, *_ in core._street_access_seconds(*ORIGIN, "bicycle", 900.0)}
    reduced = {
        stop: (seconds, mode)
        for stop, seconds, mode, *_ in core._reduced_street_offsets(
            *ORIGIN,
            False,
            [("walk", 900.0, False, None), ("bicycle", 900.0, False, None)],
        )
    }
    assert reduced
    # The oracle: per-stop min across the modes, then closed under the
    # installed transfers (ride to one platform, walk to the neighbour) —
    # the closure the engines assume of every access array.
    expected = {
        stop: min(times[stop] for times in (walk, bike) if stop in times)
        for stop in set(walk) | set(bike)
    }
    for origin, to, duration in core._transfer_edges():
        if origin in expected:
            candidate = expected[origin] + duration
            if candidate < expected.get(to, 10**9):
                expected[to] = candidate
    assert {stop: seconds for stop, (seconds, _) in reduced.items()} == expected
    # Where the direct row already wins, a tie falls to the declared order.
    for stop, (seconds, mode) in reduced.items():
        direct = {
            name: times[stop]
            for name, times in (("walk", walk), ("bicycle", bike))
            if stop in times
        }
        if direct and seconds == min(direct.values()):
            best = min(direct.values())
            if direct.get("walk") == best:
                assert mode == "walk"


def test_eligibility_masks_gate_the_vehicle_modes(multimodal_network):
    pytest.importorskip("cafein._cafein")
    core = multimodal_network._core
    bike_rows = core._street_access_seconds(*ORIGIN, "bicycle", 900.0)
    eligible = [bike_rows[0][0], bike_rows[1][0]]
    reduced = core._reduced_street_offsets(
        *ORIGIN,
        False,
        [("walk", 900.0, False, None), ("bicycle", 900.0, False, eligible)],
    )
    winners = {stop for stop, _, mode, *_ in reduced if mode == "bicycle"}
    # The bicycle can only win where it may be left — or one footpath from
    # there (park, then walk the transfer): the closure's neighbourhood.
    allowed = set(eligible)
    for origin, to, _ in core._transfer_edges():
        if origin in set(eligible):
            allowed.add(to)
    assert winners <= allowed
    assert winners & set(eligible)
    # A closure-carried winner names the seed stop the vehicle reached.
    for stop, _, mode, _, _, _, via in reduced:
        if mode == "bicycle" and stop not in set(eligible):
            assert via in set(eligible)
    walk_stops = {s for s, *_ in core._street_access_seconds(*ORIGIN, "walk", 900.0)}
    assert {stop for stop, *_ in reduced} >= walk_stops


def test_ties_fall_to_fewer_paid_rentals(multimodal_network):
    pytest.importorskip("cafein._cafein")
    core = multimodal_network._core
    # A stop's own coordinate ties both modes at zero; the paid rental
    # (declared first) must still lose to the free walk.
    coordinates = {s: (la, lo) for s, la, lo in multimodal_network.stops}
    scooter_stops = {
        s for s, *_ in core._street_access_seconds(*ORIGIN, "e_scooter", 900.0)
    }
    walk_stops = {s for s, *_ in core._street_access_seconds(*ORIGIN, "walk", 900.0)}
    stop = next(iter(scooter_stops & walk_stops))
    # Both raw choices really are zero at the stop's own coordinate.
    for mode_name in ("e_scooter", "walk"):
        rows = {
            s: t
            for s, t, *_ in core._street_access_seconds(
                *coordinates[stop], mode_name, 900.0
            )
        }
        assert rows[stop] == 0
    reduced = dict_by_stop(
        core._reduced_street_offsets(
            *coordinates[stop],
            False,
            [("e_scooter", 900.0, True, None), ("walk", 900.0, False, None)],
        )
    )
    seconds, mode = reduced[stop]
    assert seconds == 0
    assert mode == "walk"


def dict_by_stop(rows):
    return {stop: (seconds, mode) for stop, seconds, mode, *_ in rows}


def test_a_walking_only_policy_is_the_walking_path(multimodal_network):
    pytest.importorskip("cafein._cafein")
    policy = StreetLegPolicy(access={"walk": 1800})
    with_policy = multimodal_network.travel_times_from_coordinate(
        ORIGIN, "2022-02-22", "08:30:00", street_policy=policy
    )
    legacy = multimodal_network.travel_times_from_coordinate(
        ORIGIN, "2022-02-22", "08:30:00", max_walking_time=1800
    )
    assert with_policy == legacy


def test_cycling_access_only_improves_arrivals(multimodal_network):
    pytest.importorskip("cafein._cafein")
    policy = StreetLegPolicy(
        access={"walk": 1800, "bicycle": 1800},
        vehicles={
            "bicycle": VehiclePolicy(source="own", side="origin", facilities="any_stop")
        },
    )
    mixed = multimodal_network.travel_times_from_coordinate(
        ORIGIN, "2022-02-22", "08:30:00", street_policy=policy
    )
    # The apples-to-apples baseline is walking over the same multimodal
    # graph (a walking-only *policy* deliberately takes the legacy walking
    # path instead — covered by the bit-for-bit test above).
    core = multimodal_network._core
    walk_access = [
        (stop, seconds)
        for stop, seconds, *_ in core._reduced_street_offsets(
            *ORIGIN, False, [("walk", 1800.0, False, None)]
        )
    ]
    walking = core._travel_times_with_access(walk_access, "2022-02-22", "08:30:00", 7)
    assert set(mixed) >= set(walking)
    worse = [stop for stop in walking if mixed[stop] > walking[stop]]
    assert not worse
    assert any(mixed[stop] < walking[stop] for stop in walking)


def test_parking_eligibility_gates_the_bicycle(multimodal_network):
    pytest.importorskip("cafein._cafein")
    core = multimodal_network._core
    # Boarding is only permitted where the own bicycle may be left; every
    # other stop must fall back to the walking choice.
    bike_rows = core._street_access_seconds(*ORIGIN, "bicycle", 1800.0)
    parking = [bike_rows[0][0]]
    policy = StreetLegPolicy(
        access={"walk": 1800, "bicycle": 1800},
        vehicles={
            "bicycle": VehiclePolicy(source="own", side="origin", facilities=parking)
        },
    )
    from cafein.policy import reduction_modes

    reduced = core._reduced_street_offsets(
        *ORIGIN, False, reduction_modes(policy, "access", 1800.0)
    )
    for stop, _, mode, _, _, _, via in reduced:
        if mode == "bicycle":
            assert stop in parking or via in parking


def test_a_policy_needs_the_multimodal_graph(network_with_footpaths):
    pytest.importorskip("cafein._cafein")
    policy = StreetLegPolicy(
        access={"walk": 1800, "e_scooter": 900},
        vehicles={
            "e_scooter": VehiclePolicy(
                source="shared", facilities="any_stop", availability="unconstrained"
            )
        },
    )
    with pytest.raises(ValueError, match="street_modes"):
        network_with_footpaths.travel_times_from_coordinate(
            ORIGIN, "2022-02-22", "08:30:00", street_policy=policy
        )


def test_hand_seeded_access_matches_the_stop_query(multimodal_network):
    pytest.importorskip("cafein._cafein")
    # The manually seeded oracle: zero-second access at exactly one stop is
    # the same request travel_times_from_stop makes through its own public
    # path, so the two answers must agree stop for stop.
    core = multimodal_network._core
    stop = next(s for s, la, _ in multimodal_network.stops if la is not None)
    seeded = core._travel_times_with_access([(stop, 0)], "2022-02-22", "08:30:00", 7)
    reference = multimodal_network.travel_times_from_stop(
        stop, "2022-02-22", "08:30:00"
    )
    assert seeded == reference


def test_an_omitted_side_means_walking_at_the_usual_budget(multimodal_network):
    pytest.importorskip("cafein._cafein")
    from cafein import streets

    empty = StreetLegPolicy()
    assert empty.access is None and empty.egress is None
    with_default = multimodal_network.travel_times_from_coordinate(
        ORIGIN, "2022-02-22", "08:30:00", street_policy=empty
    )
    legacy = multimodal_network.travel_times_from_coordinate(
        ORIGIN,
        "2022-02-22",
        "08:30:00",
        max_walking_time=streets.MAX_ACCESS_EGRESS_TIME,
    )
    assert with_default == legacy


def test_policy_and_legacy_walk_knobs_conflict(multimodal_network):
    pytest.importorskip("cafein._cafein")
    with pytest.raises(ValueError, match="conflict"):
        multimodal_network.travel_times_from_coordinate(
            ORIGIN,
            "2022-02-22",
            "08:30:00",
            street_policy=StreetLegPolicy(access={"walk": 900}),
            max_walking_time=1200,
        )


def test_walking_transfers_and_string_selectors_are_rejected():
    with pytest.raises(ValueError, match="walking transfers"):
        StreetLegPolicy(transfers={"walk": 900})
    with pytest.raises(ValueError, match="not a known selector"):
        VehiclePolicy(source="own", side="origin", facilities="bicycle_parking")


def test_reduced_arrays_are_engine_neutral(fresh_footpaths_network):
    pytest.importorskip("cafein._cafein")
    # The same hand-built access array through RAPTOR and TBTR: identical
    # arrivals, so the reduction's consumers are engine-neutral.
    network = fresh_footpaths_network
    network._core.compute_tbtr_transfers("2022-02-22")
    stops = [s for s, la, _ in network.stops if la is not None][:3]
    access = [(stops[0], 0), (stops[1], 240), (stops[2], 611)]
    raptor = network._core._travel_times_with_access(
        access, "2022-02-22", "08:30:00", 7, router="raptor"
    )
    tbtr = network._core._travel_times_with_access(
        access, "2022-02-22", "08:30:00", 7, router="tbtr"
    )
    assert raptor == tbtr


def test_the_public_policy_path_matches_a_hand_built_reduction(multimodal_network):
    pytest.importorskip("cafein._cafein")
    core = multimodal_network._core
    # Hand-build what the policy should compute: per-mode rows, the
    # per-stop min with walk-first ties, then the transfer closure — and
    # feed that array to the engine directly. The public street_policy
    # path must answer identically.
    walk = {s: t for s, t, *_ in core._street_access_seconds(*ORIGIN, "walk", 1200.0)}
    bike = {s: t for s, t, *_ in core._street_access_seconds(*ORIGIN, "bicycle", 600.0)}
    offsets = {}
    for stop in set(walk) | set(bike):
        candidates = [
            (times[stop], name != "walk")
            for name, times in (("walk", walk), ("bicycle", bike))
            if stop in times
        ]
        offsets[stop] = min(candidates)[0]
    for origin, to, duration in core._transfer_edges():
        if origin in offsets:
            candidate = offsets[origin] + duration
            if candidate < offsets.get(to, 10**9):
                offsets[to] = candidate
    seeded = core._travel_times_with_access(
        sorted(offsets.items()), "2022-02-22", "08:30:00", 7
    )
    policy = StreetLegPolicy(
        access={"walk": 1200, "bicycle": 600},
        vehicles={
            "bicycle": VehiclePolicy(source="own", side="origin", facilities="any_stop")
        },
    )
    public = multimodal_network.travel_times_from_coordinate(
        ORIGIN, "2022-02-22", "08:30:00", street_policy=policy
    )
    assert public == seeded


# --- The street-policy travel-time matrix ------------------------------------


def _points_frame(coordinates):
    import geopandas as gpd
    from shapely.geometry import Point

    return gpd.GeoDataFrame(
        {"id": [f"p{i}" for i in range(len(coordinates))]},
        geometry=[Point(lon, lat) for lat, lon in coordinates],
        crs="EPSG:4326",
    )


MATRIX_POINTS = [(60.1690, 24.9320), (60.1795, 24.9520), (60.1580, 24.9350)]


def test_policy_matrix_cells_reconcile_with_single_queries(multimodal_network):
    pytest.importorskip("cafein._cafein")
    from cafein import TravelTimeMatrix
    from cafein import streets as _streets
    from cafein.policy import reduction_modes

    policy = StreetLegPolicy(
        access={"walk": 1200, "bicycle": 900},
        egress={"walk": 1200},
        vehicles={
            "bicycle": VehiclePolicy(source="own", side="origin", facilities="any_stop")
        },
    )
    frame = TravelTimeMatrix(
        multimodal_network,
        _points_frame(MATRIX_POINTS),
        _points_frame(MATRIX_POINTS),
        "2022-02-22",
        "08:30:00",
        street_policy=policy,
    )
    core = multimodal_network._core
    egress_modes = reduction_modes(policy, "egress", _streets.MAX_ACCESS_EGRESS_TIME)
    cells = {(row.from_id, row.to_id): row.travel_time_s for row in frame.itertuples()}
    for i, origin in enumerate(MATRIX_POINTS):
        arrivals = multimodal_network.travel_times_from_coordinate(
            origin, "2022-02-22", "08:30:00", street_policy=policy
        )
        for j, destination in enumerate(MATRIX_POINTS):
            egress = {
                stop: seconds
                for stop, seconds, *_ in core._reduced_street_offsets(
                    *destination, True, egress_modes
                )
            }
            best = min(
                (
                    arrivals[stop] + seconds
                    for stop, seconds in egress.items()
                    if stop in arrivals
                ),
                default=None,
            )
            direct = core._multimodal_direct_matrix(
                [origin], [destination], "walk", 1200.0
            )[0][0][0]
            if direct is not None:
                best = direct if best is None else min(best, direct)
            cell = cells[(f"p{i}", f"p{j}")]
            assert (pd.isna(cell) and best is None) or cell == best


def test_a_walking_only_policy_matrix_is_the_legacy_matrix(multimodal_network):
    pytest.importorskip("cafein._cafein")
    from cafein import TravelTimeMatrix

    policy_frame = TravelTimeMatrix(
        multimodal_network,
        _points_frame(MATRIX_POINTS),
        _points_frame(MATRIX_POINTS),
        "2022-02-22",
        "08:30:00",
        street_policy=StreetLegPolicy(access={"walk": 1200}, egress={"walk": 1200}),
    )
    legacy_frame = TravelTimeMatrix(
        multimodal_network,
        _points_frame(MATRIX_POINTS),
        _points_frame(MATRIX_POINTS),
        "2022-02-22",
        "08:30:00",
        max_walking_time=1200,
    )
    key = ["from_id", "to_id"]
    assert (
        policy_frame.sort_values(key)
        .reset_index(drop=True)
        .equals(legacy_frame.sort_values(key).reset_index(drop=True))
    )
    # And the diagonal is a zero-length trip.
    diagonal = policy_frame[policy_frame.from_id == policy_frame.to_id]
    assert (diagonal.travel_time_s == 0).all()


def test_policy_matrix_honours_exclusions(multimodal_network):
    pytest.importorskip("cafein._cafein")
    from cafein import TravelTimeMatrix

    policy = StreetLegPolicy(
        access={"walk": 900, "bicycle": 900},
        egress={"walk": 900},
        vehicles={
            "bicycle": VehiclePolicy(source="own", side="origin", facilities="any_stop")
        },
    )
    build = lambda **extra: TravelTimeMatrix(  # noqa: E731
        multimodal_network,
        _points_frame(MATRIX_POINTS[:1]),
        _points_frame(MATRIX_POINTS[1:2]),
        "2022-02-22",
        "08:30:00",
        street_policy=policy,
        **extra,
    )
    unrestricted = build()
    # Excluding a rich set of routes must not improve any cell.
    routes = [str(route) for route in range(1, 120)]
    restricted = build(exclude_routes=routes)
    if len(restricted) and len(unrestricted):
        assert restricted.travel_time_s.iloc[0] >= unrestricted.travel_time_s.iloc[0]


def test_policy_matrix_rejects_incompatible_knobs(multimodal_network):
    pytest.importorskip("cafein._cafein")
    from cafein import TravelTimeMatrix

    with pytest.raises(ValueError, match="does not combine"):
        TravelTimeMatrix(
            multimodal_network,
            _points_frame(MATRIX_POINTS),
            _points_frame(MATRIX_POINTS),
            "2022-02-22",
            "08:30:00",
            street_policy=StreetLegPolicy(access={"walk": 1200}),
            max_walking_time=900,
        )


def test_an_unsnapped_matrix_point_warns_and_yields_no_rows(multimodal_network):
    pytest.importorskip("cafein._cafein")
    from cafein import TravelTimeMatrix

    points = MATRIX_POINTS + [(63.0, 28.0)]  # far outside the extract
    policy = StreetLegPolicy(
        access={"walk": 900, "bicycle": 900},
        egress={"walk": 900},
        vehicles={
            "bicycle": VehiclePolicy(source="own", side="origin", facilities="any_stop")
        },
    )
    with pytest.warns(UserWarning, match="origin"):
        frame = TravelTimeMatrix(
            multimodal_network,
            _points_frame(points),
            _points_frame(MATRIX_POINTS),
            "2022-02-22",
            "08:30:00",
            street_policy=policy,
        )
    assert "p3" not in set(frame.from_id)
    assert set(frame.from_id) >= {"p0", "p1", "p2"}


def test_the_direct_walk_survives_a_walkless_access_policy(multimodal_network):
    pytest.importorskip("cafein._cafein")
    from cafein import TravelTimeMatrix

    frame = TravelTimeMatrix(
        multimodal_network,
        _points_frame(MATRIX_POINTS[:1]),
        _points_frame(MATRIX_POINTS[:1]),
        "2022-02-22",
        "08:30:00",
        street_policy=StreetLegPolicy(
            access={"bicycle": 900},
            egress={"walk": 900},
            vehicles={
                "bicycle": VehiclePolicy(
                    source="own", side="origin", facilities="any_stop"
                )
            },
        ),
    )
    # The same coordinate is a zero-length direct walk whatever the
    # access modes.
    assert int(frame.travel_time_s.iloc[0]) == 0


DEST = (60.2043, 24.9615)


def _bike_walk_policy():
    return StreetLegPolicy(
        access={"walk": 900, "bicycle": 900},
        egress={"walk": 900},
        vehicles={"bicycle": own()},
    )


def test_policy_journeys_rebuild_the_street_legs(multimodal_network):
    pytest.importorskip("cafein._cafein")
    from cafein._cafein import STREET_DISTANCE_PROVENANCE
    from cafein.policy import reduction_modes

    core = multimodal_network._core
    policy = _bike_walk_policy()
    journeys = multimodal_network.route_between_coordinates(
        ORIGIN, DEST, "2022-02-22", "08:30:00", street_policy=policy
    )
    assert journeys
    access_tokens = {
        row[0]: row[1:]
        for row in core._reduced_street_offsets(
            *ORIGIN, False, reduction_modes(policy, "access", 7200.0)
        )
    }
    egress_tokens = {
        row[0]: row[1:]
        for row in core._reduced_street_offsets(
            *DEST, True, reduction_modes(policy, "egress", 7200.0)
        )
    }
    for journey in journeys:
        legs = journey["legs"]
        assert legs[0]["departure_s"] == journey["departure_s"]
        assert legs[-1]["arrival_s"] == journey["arrival_s"]
        for before, after in zip(legs, legs[1:]):
            assert before["arrival_s"] <= after["departure_s"]
        for leg in legs:
            if leg["type"] == "transit":
                assert leg["mode"] is None
                continue
            assert leg["mode"] in ("walk", "bicycle")
            if leg["type"] in ("access", "egress", "walk"):
                assert leg["distance_m"] > 0.0
                assert leg["distance_m"] == pytest.approx(
                    leg["network_distance_m"] + leg["connector_distance_m"]
                )
                assert leg["distance_provenance"] == STREET_DISTANCE_PROVENANCE
                assert leg["geometry"] is not None
        # A direct (non-via) end leg's duration is exactly its token's
        # reduced seconds, and its mode is the token's winning mode.
        if legs[0]["type"] == "access":
            token = access_tokens[legs[0]["to_stop"]]
            if token[5] is None:
                assert legs[0]["arrival_s"] - legs[0]["departure_s"] == token[0]
                assert legs[0]["mode"] == token[1]
        if legs[-1]["type"] == "egress":
            token = egress_tokens[legs[-1]["from_stop"]]
            if token[5] is None:
                assert legs[-1]["arrival_s"] - legs[-1]["departure_s"] == token[0]
                assert legs[-1]["mode"] == token[1]


def test_a_walking_only_policy_routes_the_legacy_journeys(multimodal_network):
    pytest.importorskip("cafein._cafein")

    legacy = multimodal_network.route_between_coordinates(
        ORIGIN, DEST, "2022-02-22", "08:30:00"
    )
    policied = multimodal_network.route_between_coordinates(
        ORIGIN,
        DEST,
        "2022-02-22",
        "08:30:00",
        street_policy=StreetLegPolicy(access={"walk": 7200}, egress={"walk": 7200}),
    )
    assert policied == legacy


def test_a_via_choice_splits_into_the_vehicle_and_the_walk(multimodal_network):
    pytest.importorskip("cafein._cafein")
    from cafein.network import _policy_street_legs

    core = multimodal_network._core
    unrestricted = core._reduced_street_offsets(
        *ORIGIN, False, [("bicycle", 900.0, False, None)]
    )
    hub = min(unrestricted, key=lambda row: row[1])
    assert hub[6] is None
    rows = core._reduced_street_offsets(
        *ORIGIN, False, [("bicycle", 900.0, False, [hub[0]])]
    )
    carried = [row for row in rows if row[6] is not None]
    assert carried, "parking at one stop must spread through the closure"
    transfers = {(frm, to): seconds for frm, to, seconds in core._transfer_edges()}
    # Every carried choice names the hub as its seed, keeps the hub's own
    # link snap, and costs the hub's direct time plus exactly one
    # installed transfer - the vehicle only ever reaches the hub, and the
    # closure never composes edges.
    for row in carried:
        assert row[6] == hub[0]
        assert row[3:6] == hub[3:6]
        assert row[1] == hub[1] + transfers[(hub[0], row[0])]
    tokens = {row[0]: row[1:] for row in rows}
    stop, seconds = carried[0][0], carried[0][1]
    departure = 30600
    legs = _policy_street_legs(
        core,
        {
            "type": "access",
            "to_stop": stop,
            "departure_s": departure,
            "arrival_s": departure + seconds,
        },
        ORIGIN,
        tokens,
        {"bicycle": 900.0},
        False,
        True,
    )
    assert len(legs) == 2
    vehicle, walked = legs
    assert vehicle["type"] == "access" and vehicle["to_stop"] == hub[0]
    assert vehicle["mode"] == "bicycle"
    assert walked["type"] == "transfer" and walked["mode"] == "walk"
    assert walked["from_stop"] == hub[0] and walked["to_stop"] == stop
    assert vehicle["departure_s"] == departure
    assert vehicle["arrival_s"] == walked["departure_s"]
    assert walked["arrival_s"] == departure + seconds
    # The rebuilt vehicle leg is exactly the hub's own reduced time, and
    # the walked remainder is exactly the installed transfer.
    assert vehicle["arrival_s"] - vehicle["departure_s"] == hub[1]
    assert walked["arrival_s"] - walked["departure_s"] == transfers[(hub[0], stop)]
    assert vehicle["distance_m"] > 0.0 and vehicle["geometry"] is not None


def test_policy_itineraries_carry_modes_and_street_emissions(multimodal_network):
    pytest.importorskip("cafein._cafein")
    from cafein import DetailedItineraries

    policy = _bike_walk_policy()
    factors = pd.DataFrame(
        {
            "street_mode": ["bicycle"],
            "vehicle": [5.0],
            "fuel": [0.0],
            "infrastructure": [10.0],
            "operations": [6.0],
        }
    )
    frame = DetailedItineraries(
        multimodal_network,
        _points_frame([ORIGIN]),
        _points_frame([DEST]),
        "2022-02-22",
        "08:30:00",
        street_policy=policy,
        factors=factors,
    )
    assert list(frame.columns[:6]) == [
        "from_id",
        "to_id",
        "option",
        "segment",
        "leg_type",
        "mode",
    ]
    transit = frame[frame["leg_type"] == "transit"]
    assert not transit.empty and transit["mode"].isna().all()
    assert (transit["emissions"] > 0.0).all()
    assert transit["network_distance_m"].isna().all()
    assert transit["connector_distance_m"].isna().all()
    street = frame[frame["leg_type"].isin(["access", "egress", "walk"])]
    assert not street.empty and street["mode"].isin(["walk", "bicycle"]).all()
    # The rebuilt legs expose their exact distance parts beside the total.
    assert street["network_distance_m"].notna().all()
    assert list(
        street["network_distance_m"] + street["connector_distance_m"]
    ) == pytest.approx(list(street["distance_m"]))
    assert (street.loc[street["mode"] == "walk", "emissions"] == 0.0).all()
    bicycle = street[street["mode"] == "bicycle"]
    assert not bicycle.empty
    journeys = multimodal_network.route_between_coordinates(
        ORIGIN, DEST, "2022-02-22", "08:30:00", street_policy=policy
    )
    expected = sorted(
        leg["network_distance_m"] / 1000.0 * 21.0
        for journey in journeys
        for leg in journey["legs"]
        if leg.get("mode") == "bicycle"
    )
    assert sorted(bicycle["emissions"]) == pytest.approx(expected)


def test_policy_itineraries_reject_incompatible_knobs(multimodal_network):
    pytest.importorskip("cafein._cafein")
    from cafein import DetailedItineraries

    policy = _bike_walk_policy()
    points = (_points_frame([ORIGIN]), _points_frame([DEST]))
    for kwargs, message in [
        ({"candidates": "diverse"}, "candidates='diverse'"),
        ({"router": "tbtr"}, "router='tbtr'"),
        ({"max_walking_time": 900}, "carries its own budgets"),
    ]:
        with pytest.raises(ValueError, match=message):
            DetailedItineraries(
                multimodal_network,
                *points,
                "2022-02-22",
                "08:30:00",
                street_policy=policy,
                **kwargs,
            )
    with pytest.raises(ValueError, match="point origins and destinations"):
        DetailedItineraries(
            multimodal_network,
            ["1130446"],
            ["1140447"],
            "2022-02-22",
            "08:30:00",
            street_policy=policy,
        )
    with pytest.raises(ValueError, match="departure window"):
        multimodal_network.route_between_coordinates(
            ORIGIN,
            DEST,
            "2022-02-22",
            "08:30:00",
            window=600,
            street_policy=policy,
        )


def test_the_direct_walk_dominates_the_policy_journeys(multimodal_network):
    pytest.importorskip("cafein._cafein")

    journeys = multimodal_network.route_between_coordinates(
        ORIGIN, ORIGIN, "2022-02-22", "08:30:00", street_policy=_bike_walk_policy()
    )
    assert len(journeys) == 1
    walk = journeys[0]
    assert walk["rides"] == 0 and walk["arrival_s"] == walk["departure_s"]
    assert walk["legs"][0]["type"] == "walk"
    assert walk["legs"][0]["mode"] == "walk"
    assert walk["legs"][0]["distance_m"] == 0.0


def test_policy_journeys_honour_exclusions(multimodal_network):
    pytest.importorskip("cafein._cafein")

    policy = _bike_walk_policy()
    journeys = multimodal_network.route_between_coordinates(
        ORIGIN, DEST, "2022-02-22", "08:30:00", street_policy=policy
    )
    boarded = next(
        leg["board_stop"]
        for journey in journeys
        for leg in journey["legs"]
        if leg["type"] == "transit"
    )
    excluded = multimodal_network.route_between_coordinates(
        ORIGIN,
        DEST,
        "2022-02-22",
        "08:30:00",
        street_policy=policy,
        exclude_stops=[boarded],
    )
    for journey in excluded:
        for leg in journey["legs"]:
            for key in ("board_stop", "alight_stop", "from_stop", "to_stop"):
                assert leg.get(key) != boarded


def test_policy_itineraries_reconcile_with_the_time_matrix(multimodal_network):
    pytest.importorskip("cafein._cafein")
    from cafein import DetailedItineraries, TravelTimeMatrix

    policy = _bike_walk_policy()
    itineraries = DetailedItineraries(
        multimodal_network,
        _points_frame([ORIGIN]),
        _points_frame([DEST]),
        "2022-02-22",
        "08:30:00",
        street_policy=policy,
        geometries=False,
    )
    per_option = itineraries.groupby("option").agg(
        departure=("departure_s", "min"), arrival=("arrival_s", "max")
    )
    fastest = int((per_option["arrival"] - per_option["departure"]).min())
    matrix = TravelTimeMatrix(
        multimodal_network,
        _points_frame([ORIGIN]),
        _points_frame([DEST]),
        "2022-02-22",
        "08:30:00",
        street_policy=policy,
    )
    assert int(matrix["travel_time_s"].iloc[0]) == fastest


def test_an_egress_via_choice_walks_the_forward_transfer(multimodal_network):
    pytest.importorskip("cafein._cafein")
    from cafein.network import _policy_street_legs

    core = multimodal_network._core
    unrestricted = core._reduced_street_offsets(
        *DEST, True, [("bicycle", 900.0, False, None)]
    )
    hub = min(unrestricted, key=lambda row: row[1])
    assert hub[6] is None
    rows = core._reduced_street_offsets(
        *DEST, True, [("bicycle", 900.0, False, [hub[0]])]
    )
    carried = [row for row in rows if row[6] is not None]
    assert carried
    transfers = {(frm, to): seconds for frm, to, seconds in core._transfer_edges()}
    # The egress mirror of the closure shape: each carried choice walks
    # exactly one installed forward stop-to-seed edge, then rides.
    for row in carried:
        assert row[6] == hub[0]
        assert row[1] == hub[1] + transfers[(row[0], hub[0])]
    tokens = {row[0]: row[1:] for row in rows}
    stop, seconds = carried[0][0], carried[0][1]
    arrival = 34200
    legs = _policy_street_legs(
        core,
        {
            "type": "egress",
            "from_stop": stop,
            "departure_s": arrival - seconds,
            "arrival_s": arrival,
        },
        DEST,
        tokens,
        {"bicycle": 900.0},
        True,
        True,
    )
    assert len(legs) == 2
    walked, vehicle = legs
    # The traveller walks the closure's forward stop-to-seed edge first,
    # then rides the vehicle from the seed's link to the destination.
    assert walked["type"] == "transfer" and walked["mode"] == "walk"
    assert walked["from_stop"] == stop and walked["to_stop"] == hub[0]
    assert vehicle["type"] == "egress" and vehicle["from_stop"] == hub[0]
    assert vehicle["mode"] == "bicycle"
    assert walked["departure_s"] == arrival - seconds
    assert walked["arrival_s"] == vehicle["departure_s"]
    assert vehicle["arrival_s"] == arrival
    assert vehicle["arrival_s"] - vehicle["departure_s"] == hub[1]
    assert walked["arrival_s"] - walked["departure_s"] == transfers[(stop, hub[0])]
    # The walked meters come from that same forward edge.
    forward = core._transfer_leg(stop, hub[0], False)
    assert forward is not None and walked["distance_m"] == forward[1]


OFFSHORE = (60.05, 24.60)


def test_an_unsnapped_policy_side_degrades_to_the_direct_walk(multimodal_network):
    pytest.importorskip("cafein._cafein")

    policy = _bike_walk_policy()
    journeys = multimodal_network.route_between_coordinates(
        OFFSHORE, OFFSHORE, "2022-02-22", "08:30:00", street_policy=policy
    )
    # No policy mode snaps offshore, but the zero walk to the same
    # coordinate needs no network at all - the query degrades to it
    # instead of failing, as the policy matrix path does.
    assert len(journeys) == 1
    assert journeys[0]["rides"] == 0
    assert journeys[0]["legs"][0]["type"] == "walk"
    with pytest.raises(ValueError, match="too far from the multimodal"):
        multimodal_network.route_between_coordinates(
            OFFSHORE, DEST, "2022-02-22", "08:30:00", street_policy=policy
        )


def _street_factor_rows():
    return pd.DataFrame(
        {
            "street_mode": ["bicycle"],
            "vehicle": [5.0],
            "fuel": [0.0],
            "infrastructure": [10.0],
            "operations": [6.0],
        }
    )


def test_policy_cost_matrix_reconciles_with_the_itineraries(multimodal_network):
    pytest.importorskip("cafein._cafein")
    from cafein import DetailedItineraries, TravelCostMatrix

    policy = _bike_walk_policy()
    factors = _street_factor_rows()
    origins = _points_frame([ORIGIN, MATRIX_POINTS[1]])
    destinations = _points_frame([DEST, MATRIX_POINTS[2]])
    matrix = TravelCostMatrix(
        multimodal_network,
        origins,
        destinations,
        "2022-02-22",
        "08:30:00",
        street_policy=policy,
        factors=factors,
    )
    assert set(matrix.columns) >= {
        "travel_time_s",
        "transfers",
        "transit_distance_m",
        "walk_distance_m",
        "street_distance_m",
        "emissions",
    }
    assert (matrix["street_distance_m"] > 0.0).any()
    for (from_id, to_id), cell in matrix.groupby(["from_id", "to_id"]):
        assert len(cell) == 1
        cell = cell.iloc[0]
        itineraries = DetailedItineraries(
            multimodal_network,
            origins[origins["id"] == from_id],
            destinations[destinations["id"] == to_id],
            "2022-02-22",
            "08:30:00",
            street_policy=policy,
            factors=factors,
            geometries=False,
        )
        options = itineraries.groupby("option").agg(
            departure=("departure_s", "min"),
            arrival=("arrival_s", "max"),
            emissions=("emissions", "sum"),
            distance=("distance_m", "sum"),
        )
        durations = options["arrival"] - options["departure"]
        assert int(cell["travel_time_s"]) == int(durations.min())
        # Where one option alone attains the fastest time, the matrix
        # aggregated exactly that journey: its distances and emissions
        # match the legs' sums.
        fastest = options[durations == durations.min()]
        if len(fastest) == 1:
            assert cell["emissions"] == pytest.approx(
                float(fastest["emissions"].iloc[0]), rel=1e-9
            )
            total = (
                cell["transit_distance_m"]
                + cell["walk_distance_m"]
                + cell["street_distance_m"]
            )
            assert total == pytest.approx(float(fastest["distance"].iloc[0]), rel=1e-9)


def test_a_walking_only_policy_cost_matrix_is_the_legacy_matrix(multimodal_network):
    pytest.importorskip("cafein._cafein")
    from cafein import TravelCostMatrix

    origins = _points_frame(MATRIX_POINTS[:2])
    destinations = _points_frame(MATRIX_POINTS[1:])
    legacy = TravelCostMatrix(
        multimodal_network, origins, destinations, "2022-02-22", "08:30:00"
    )
    policied = TravelCostMatrix(
        multimodal_network,
        origins,
        destinations,
        "2022-02-22",
        "08:30:00",
        street_policy=StreetLegPolicy(access={"walk": 7200}, egress={"walk": 7200}),
    )
    assert (policied["street_distance_m"] == 0.0).all()
    pd.testing.assert_frame_equal(
        policied.drop(columns="street_distance_m").reset_index(drop=True),
        pd.DataFrame(legacy).reset_index(drop=True),
    )


def test_policy_cost_matrix_rejects_incompatible_knobs(multimodal_network):
    pytest.importorskip("cafein._cafein")
    from cafein import TravelCostMatrix

    policy = _bike_walk_policy()
    points = (_points_frame([ORIGIN]), _points_frame([DEST]))
    for kwargs in [
        {"optimize": "emissions", "window": 3600},
        {"router": "tbtr"},
        {"max_walking_time": 900},
        {"candidates": "pareto"},
    ]:
        with pytest.raises(ValueError, match="does not combine"):
            TravelCostMatrix(
                multimodal_network,
                *points,
                "2022-02-22",
                "08:30:00",
                street_policy=policy,
                **kwargs,
            )


def test_policy_cost_matrix_street_emissions_never_zero_silently(multimodal_network):
    pytest.importorskip("cafein._cafein")
    from cafein import TravelCostMatrix

    policy = StreetLegPolicy(
        access={"bicycle": 900},
        egress={"walk": 900},
        vehicles={"bicycle": own()},
    )
    origins = _points_frame([ORIGIN])
    destinations = _points_frame([DEST])
    # A winning user row may carry NA components; the factor then stays
    # unresolved and its rows' emissions poison to NaN, never a silent
    # zero — the shipped sourced defaults resolve only when the user
    # supplies nothing for the mode.
    unresolved_rows = pd.DataFrame(
        {
            "street_mode": ["bicycle"],
            "vehicle": [float("nan")],
            "fuel": [0.0],
            "infrastructure": [0.0],
            "operations": [0.0],
        }
    )
    with pytest.warns(UserWarning, match="unresolved for street mode 'bicycle'"):
        bare = TravelCostMatrix(
            multimodal_network,
            origins,
            destinations,
            "2022-02-22",
            "08:30:00",
            street_policy=policy,
            factors=unresolved_rows,
        )
    ridden_bicycle = bare[(bare["street_distance_m"] > 0.0)]
    assert not ridden_bicycle.empty
    assert ridden_bicycle["emissions"].isna().all()
    priced = TravelCostMatrix(
        multimodal_network,
        origins,
        destinations,
        "2022-02-22",
        "08:30:00",
        street_policy=policy,
        factors=_street_factor_rows(),
    )
    ridden_bicycle = priced[(priced["street_distance_m"] > 0.0)]
    assert not ridden_bicycle.empty
    assert ridden_bicycle["emissions"].notna().all()


def test_the_policy_cost_matrix_zero_walk_is_free(multimodal_network):
    pytest.importorskip("cafein._cafein")
    from cafein import TravelCostMatrix

    frame = TravelCostMatrix(
        multimodal_network,
        _points_frame([ORIGIN]),
        _points_frame([ORIGIN]),
        "2022-02-22",
        "08:30:00",
        street_policy=_bike_walk_policy(),
    )
    cell = frame.iloc[0]
    assert int(cell["travel_time_s"]) == 0
    assert cell["walk_distance_m"] == 0.0
    assert cell["street_distance_m"] == 0.0
    assert cell["emissions"] == 0.0


def test_a_zero_ride_street_composition_is_a_journey(multimodal_network):
    pytest.importorskip("cafein._cafein")

    # Ride the bicycle to a stop and walk away without boarding: the
    # engine never emits this journey (walking compositions are always
    # dominated by the direct walk on one graph), so the policy path
    # composes it from the closed reduced arrays. The pair below is one
    # where it beats every ridden alternative and the direct walk.
    journeys = multimodal_network.route_between_coordinates(
        ORIGIN,
        MATRIX_POINTS[2],
        "2022-02-22",
        "08:30:00",
        street_policy=_bike_walk_policy(),
    )
    assert journeys
    fastest = min(journeys, key=lambda j: j["arrival_s"] - j["departure_s"])
    assert fastest["rides"] == 0
    modes = [leg["mode"] for leg in fastest["legs"]]
    assert "bicycle" in modes
    types = [leg["type"] for leg in fastest["legs"]]
    assert types[0] == "access" and types[-1] == "egress"
    for before, after in zip(fastest["legs"], fastest["legs"][1:]):
        assert before["arrival_s"] <= after["departure_s"]


def test_an_unsnapped_coincident_pair_yields_no_cost_rows(multimodal_network):
    pytest.importorskip("cafein._cafein")
    from cafein import TravelCostMatrix

    # Coincident but off the street network: the zero-walk convention
    # never outranks the unsnapped-row contract - the pair is warned and
    # omitted, exactly as in the time matrix.
    with pytest.warns(UserWarning, match="off the walking network"):
        frame = TravelCostMatrix(
            multimodal_network,
            _points_frame([OFFSHORE]),
            _points_frame([OFFSHORE]),
            "2022-02-22",
            "08:30:00",
            street_policy=_bike_walk_policy(),
        )
    assert frame.empty


def test_a_walking_only_policy_accepts_street_factor_rows(multimodal_network):
    pytest.importorskip("cafein._cafein")
    from cafein import TravelCostMatrix

    # A street-mode factor table configures street vehicles only; the
    # walking-only fast path must not feed it to the transit resolver.
    frame = TravelCostMatrix(
        multimodal_network,
        _points_frame([ORIGIN]),
        _points_frame([DEST]),
        "2022-02-22",
        "08:30:00",
        street_policy=StreetLegPolicy(access={"walk": 7200}, egress={"walk": 7200}),
        factors=_street_factor_rows(),
    )
    assert not frame.empty
    # Transit legs keep the shipped defaults, so ridden rows still carry
    # resolved emissions.
    ridden = frame[frame["transfers"] >= 0]
    assert (ridden["emissions"].notna()).any()


def test_the_composition_coexists_with_faster_transit(multimodal_network):
    pytest.importorskip("cafein._cafein")

    # The (arrival, rides) Pareto contract: an earlier *ridden* journey
    # never dominates the zero-ride composition — they coexist, exactly
    # as the walking journey always has. Here transit is fastest, the
    # bicycle composition beats walking, and both are returned.
    journeys = multimodal_network.route_between_coordinates(
        ORIGIN, DEST, "2022-02-22", "08:30:00", street_policy=_bike_walk_policy()
    )
    ridden = [j for j in journeys if j["rides"] > 0]
    zero = [j for j in journeys if j["rides"] == 0]
    assert ridden and zero
    assert min(j["arrival_s"] for j in ridden) < min(j["arrival_s"] for j in zero)


def test_the_pareto_reduction_degenerates_without_factors(multimodal_network):
    pytest.importorskip("cafein._cafein")

    core = multimodal_network._core
    modes = [("walk", 900.0, False, None), ("bicycle", 900.0, False, None)]
    winners = {
        (stop, seconds)
        for stop, seconds, *_ in core._reduced_street_offsets(*ORIGIN, False, modes)
    }
    rows = core._pareto_street_rows(
        *ORIGIN,
        False,
        [
            (mode, cutoff, rental, eligible, 0.0)
            for mode, cutoff, rental, eligible in modes
        ],
    )
    # Every factor zero: the frontier is exactly the time-only winner,
    # one point per stop, riding free.
    assert {(row[0], row[1]) for row in rows} == winners
    assert len(rows) == len(winners)
    assert all(row[2] == 0.0 for row in rows)


def test_the_pareto_frontier_keeps_the_cleaner_slower_choice(multimodal_network):
    pytest.importorskip("cafein._cafein")

    core = multimodal_network._core
    rows = core._pareto_street_rows(
        *ORIGIN,
        False,
        [("walk", 900.0, False, None, 0.0), ("e_scooter", 900.0, False, None, 100.0)],
    )
    by_stop = {}
    for row in rows:
        by_stop.setdefault(row[0], []).append(row)
    doubles = {stop: points for stop, points in by_stop.items() if len(points) > 1}
    assert doubles, "somewhere the scooter must be faster and dirtier than walking"
    for points in doubles.values():
        fast, slow = points[0], points[-1]
        assert fast[1] < slow[1] and fast[2] > slow[2]
        assert fast[3] == "e_scooter" and slow[3] == "walk"
        assert fast[2] == pytest.approx(fast[7] / 1000.0 * 100.0)
        assert slow[2] == 0.0
    # And per-stop points are sorted by seconds with strictly improving
    # grams — a true frontier.
    for points in by_stop.values():
        seconds = [p[1] for p in points]
        grams = [p[2] for p in points]
        assert seconds == sorted(seconds)
        assert grams == sorted(grams, reverse=True)


def test_an_unresolved_factor_survives_only_where_fastest(multimodal_network):
    pytest.importorskip("cafein._cafein")
    import math

    core = multimodal_network._core
    rows = core._pareto_street_rows(
        *ORIGIN,
        False,
        [
            ("walk", 900.0, False, None, 0.0),
            ("bicycle", 900.0, False, None, float("nan")),
        ],
    )
    walk_seconds = {row[0]: row[1] for row in rows if row[3] == "walk"}
    bicycle = [row for row in rows if row[3] == "bicycle"]
    assert bicycle, "the bicycle is strictly fastest somewhere"
    for row in bicycle:
        assert math.isnan(row[2])
        # NaN grams read as infinitely dirty: the choice survives only
        # where strictly fastest, never as an equal-time alternative.
        if row[0] in walk_seconds:
            assert row[1] < walk_seconds[row[0]]


def test_the_pareto_closure_extends_whole_frontiers(multimodal_network):
    pytest.importorskip("cafein._cafein")

    core = multimodal_network._core
    unrestricted = core._reduced_street_offsets(
        *ORIGIN, False, [("bicycle", 900.0, False, None)]
    )
    hub = min(unrestricted, key=lambda row: row[1])
    rows = core._pareto_street_rows(
        *ORIGIN,
        False,
        [("bicycle", 900.0, False, [hub[0]], 50.0)],
    )
    transfers = {(frm, to): seconds for frm, to, seconds in core._transfer_edges()}
    hub_rows = [row for row in rows if row[0] == hub[0]]
    assert len(hub_rows) == 1 and hub_rows[0][10] is None
    carried = [row for row in rows if row[10] is not None]
    assert carried
    for row in carried:
        assert row[10] == hub[0]
        assert row[1] == hub_rows[0][1] + transfers[(hub[0], row[0])]
        # The vehicle leg is the hub's, so its grams carry unchanged.
        assert row[2] == pytest.approx(hub_rows[0][2])


def _scooter_policy():
    return StreetLegPolicy(
        access={"walk": 900, "e_scooter": 900},
        egress={"walk": 900},
        vehicles={"e_scooter": shared()},
    )


def _scooter_factor_rows():
    return pd.DataFrame(
        {
            "street_mode": ["e_scooter"],
            "vehicle": [10.0],
            "fuel": [1.0],
            "infrastructure": [5.0],
            "operations": [5.0],
        }
    )


def test_mc_policy_options_form_a_true_frontier(multimodal_network):
    pytest.importorskip("cafein._cafein")
    from cafein import DetailedItineraries

    frame = DetailedItineraries(
        multimodal_network,
        _points_frame([ORIGIN]),
        _points_frame([DEST]),
        "2022-02-22",
        "08:30:00",
        candidates="pareto",
        street_policy=_scooter_policy(),
        factors=_scooter_factor_rows(),
        geometries=False,
    )
    options = frame.groupby("option").agg(
        departure=("departure_s", "min"),
        arrival=("arrival_s", "max"),
        grams=("emissions", "sum"),
    )
    durations = (options["arrival"] - options["departure"]).tolist()
    grams = options["grams"].tolist()
    # The cleaner-but-slower alternatives the time-only set misses: more
    # than one option, sorted into a genuine (duration, grams) frontier.
    assert len(durations) > 1
    assert durations == sorted(durations)
    assert grams == sorted(grams, reverse=True)
    # Street emissions ride the scooter's network meters at 21 g/pkm.
    scooter = frame[frame["mode"] == "e_scooter"]
    assert not scooter.empty
    assert scooter["emissions"].equals(scooter["network_distance_m"] / 1000.0 * 21.0)


def test_mc_policy_surfaces_zero_ride_street_compositions(multimodal_network):
    pytest.importorskip("cafein._cafein")
    from cafein import DetailedItineraries

    frame = DetailedItineraries(
        multimodal_network,
        _points_frame([ORIGIN]),
        _points_frame([DEST]),
        "2022-02-22",
        "08:30:00",
        candidates="pareto",
        street_policy=_scooter_policy(),
        factors=_scooter_factor_rows(),
        geometries=False,
    )
    # The engine drains its access seeds: at least one Pareto option
    # rides no transit at all yet uses the scooter — the zero-ride
    # street composition on the emissions frontier.
    zero_ride = [
        option
        for option, legs in frame.groupby("option")
        if not (legs["leg_type"] == "transit").any()
        and (legs["mode"] == "e_scooter").any()
    ]
    assert zero_ride


def test_a_walking_only_policy_rides_the_legacy_pareto_path(multimodal_network):
    pytest.importorskip("cafein._cafein")
    from cafein import DetailedItineraries

    legacy = DetailedItineraries(
        multimodal_network,
        _points_frame([ORIGIN]),
        _points_frame([DEST]),
        "2022-02-22",
        "08:30:00",
        candidates="pareto",
        max_walking_time=1200,
        geometries=False,
    )
    policied = DetailedItineraries(
        multimodal_network,
        _points_frame([ORIGIN]),
        _points_frame([DEST]),
        "2022-02-22",
        "08:30:00",
        candidates="pareto",
        street_policy=StreetLegPolicy(access={"walk": 1200}, egress={"walk": 1200}),
        geometries=False,
    )
    additive = ["mode", "network_distance_m", "connector_distance_m", "bike_aboard"]
    pd.testing.assert_frame_equal(
        pd.DataFrame(policied.drop(columns=additive)).reset_index(drop=True),
        pd.DataFrame(legacy).reset_index(drop=True),
    )


def test_mc_policy_rejects_unresolved_street_factors(multimodal_network):
    pytest.importorskip("cafein._cafein")
    from cafein import DetailedItineraries

    unresolved_rows = pd.DataFrame(
        {
            "street_mode": ["e_scooter"],
            "vehicle": [float("nan")],
            "fuel": [0.0],
            "infrastructure": [0.0],
            "operations": [0.0],
        }
    )
    with pytest.raises(ValueError, match="unresolved"):
        DetailedItineraries(
            multimodal_network,
            _points_frame([ORIGIN]),
            _points_frame([DEST]),
            "2022-02-22",
            "08:30:00",
            candidates="pareto",
            street_policy=_scooter_policy(),
            factors=unresolved_rows,
            geometries=False,
        )
    # Without factors= the shipped sourced defaults resolve the shared
    # scooter out of the box, at its fleet (not private) rate.
    frame = DetailedItineraries(
        multimodal_network,
        _points_frame([ORIGIN]),
        _points_frame([DEST]),
        "2022-02-22",
        "08:30:00",
        candidates="pareto",
        street_policy=_scooter_policy(),
        geometries=False,
    )
    scooter = frame[frame["mode"] == "e_scooter"]
    assert not scooter.empty
    assert scooter["emissions"].equals(scooter["network_distance_m"] / 1000.0 * 108.2)
    with pytest.raises(ValueError, match="diverse"):
        DetailedItineraries(
            multimodal_network,
            _points_frame([ORIGIN]),
            _points_frame([DEST]),
            "2022-02-22",
            "08:30:00",
            candidates="diverse",
            street_policy=_scooter_policy(),
            factors=_scooter_factor_rows(),
            geometries=False,
        )


def test_relaxed_policy_candidates_widen_the_frontier(multimodal_network):
    pytest.importorskip("cafein._cafein")
    from cafein import DetailedItineraries

    shared_kwargs = dict(
        date="2022-02-22",
        departure="08:30:00",
        street_policy=_scooter_policy(),
        factors=_scooter_factor_rows(),
        geometries=False,
    )
    pareto = DetailedItineraries(
        multimodal_network,
        _points_frame([ORIGIN]),
        _points_frame([DEST]),
        candidates="pareto",
        **shared_kwargs,
    )
    relaxed = DetailedItineraries(
        multimodal_network,
        _points_frame([ORIGIN]),
        _points_frame([DEST]),
        candidates="relaxed",
        **shared_kwargs,
    )
    assert relaxed["option"].nunique() >= pareto["option"].nunique()


def test_mc_policy_engines_answer_identically(multimodal_mctbtr_network):
    pytest.importorskip("cafein._cafein")
    from cafein import DetailedItineraries

    frames = {}
    for router in ("raptor", "tbtr", "auto"):
        frames[router] = DetailedItineraries(
            multimodal_mctbtr_network,
            _points_frame([ORIGIN]),
            _points_frame([DEST]),
            "2022-02-22",
            "08:30:00",
            candidates="pareto",
            router=router,
            street_policy=_scooter_policy(),
            factors=_scooter_factor_rows(),
            geometries=False,
        )
    # Engine neutrality with policy label sets: McTBTR (the cached set
    # serves both the explicit and the auto-resolved arm) answers exactly
    # what McRAPTOR answers, zero-ride street compositions included.
    for router in ("tbtr", "auto"):
        pd.testing.assert_frame_equal(
            pd.DataFrame(frames[router]).reset_index(drop=True),
            pd.DataFrame(frames["raptor"]).reset_index(drop=True),
        )
    zero_ride = [
        option
        for option, legs in frames["tbtr"].groupby("option")
        if not (legs["leg_type"] == "transit").any()
        and (legs["mode"] == "e_scooter").any()
    ]
    assert zero_ride


def test_the_shipped_street_factors_resolve_by_service_model():
    pytest.importorskip("cafein._cafein")
    from cafein import emissions

    # ITF "Good to Go?" components on the Finland 2020 mix (cafein-lca),
    # the conventional bicycle's dietary 21 g/km on top; the shared
    # e-scooter follows Judl et al. (2026) 2GEN gross plus the ITF
    # infrastructure graft.
    assert emissions.street_factor("walk") == 0.0
    assert emissions.street_factor("bicycle") == pytest.approx(37.0)
    assert emissions.street_factor("e_bike") == pytest.approx(25.0)
    assert emissions.street_factor("e_scooter") == pytest.approx(36.0)
    assert emissions.street_factor(
        "e_scooter", service_model="shared"
    ) == pytest.approx(108.2)
    # The identity's own service model is the default.
    assert emissions.street_factor(
        "e_scooter", service_model="private"
    ) == pytest.approx(36.0)


def _transfer_policy(budget=600):
    return StreetLegPolicy(
        access={"walk": 900},
        egress={"walk": 900},
        transfers={"e_scooter": budget},
        vehicles={"e_scooter": shared()},
    )


def test_transfer_policies_take_shared_modes_only():
    with pytest.raises(ValueError, match="walking transfers"):
        StreetLegPolicy(transfers={"walk": 600})
    with pytest.raises(ValueError, match="possession state"):
        StreetLegPolicy(
            transfers={"bicycle": 600},
            vehicles={"bicycle": own()},
        )
    with pytest.raises(ValueError, match="vehicle terms"):
        StreetLegPolicy(transfers={"e_scooter": 600})
    with pytest.raises(ValueError, match="one transfer mode at a time"):
        StreetLegPolicy(
            transfers={"e_scooter": 600, "bicycle": 500},
            vehicles={"e_scooter": shared(), "bicycle": shared()},
        )
    with pytest.raises(ValueError, match="any_stop"):
        StreetLegPolicy(
            transfers={"e_scooter": 600},
            vehicles={
                "e_scooter": VehiclePolicy(
                    source="shared",
                    facilities=("1230109",),
                    availability="unconstrained",
                )
            },
        )
    policy = _transfer_policy()
    assert policy.transfers == {"e_scooter": 600.0}
    assert "transfers={'e_scooter': 600.0}" in repr(policy)


def test_transfers_need_the_matching_merged_set(multimodal_network):
    pytest.importorskip("cafein._cafein")

    with pytest.raises(ValueError, match="compute_mode_transfers"):
        multimodal_network.route_between_coordinates(
            ORIGIN,
            DEST,
            "2022-02-22",
            "08:30:00",
            street_policy=_transfer_policy(),
        )


def test_a_mismatched_transfer_binding_is_rejected(multimodal_transfers_network):
    pytest.importorskip("cafein._cafein")

    with pytest.raises(ValueError, match="bound to"):
        multimodal_transfers_network.route_between_coordinates(
            ORIGIN,
            DEST,
            "2022-02-22",
            "08:30:00",
            street_policy=_transfer_policy(budget=500),
        )


def test_the_merged_set_preserves_the_walking_closure(multimodal_transfers_network):
    pytest.importorskip("cafein._cafein")

    core = multimodal_transfers_network._core
    mode, budget, edges, rented = core._mode_transfer_binding
    assert (mode, budget) == ("e_scooter", 600.0)
    walking = len(core._transfer_edges())
    # Never weaker than the set it extends: every walking pair survives,
    # rentals only add or improve.
    assert edges >= walking
    assert 0 < rented < edges


def test_rented_transfers_improve_arrivals_everywhere(multimodal_transfers_network):
    pytest.importorskip("cafein._cafein")

    from cafein import streets as _streets
    from cafein.policy import reduction_modes

    core = multimodal_transfers_network._core
    modes = reduction_modes(
        _transfer_policy(), "access", _streets.MAX_ACCESS_EGRESS_TIME
    )
    access = [
        (stop, seconds)
        for stop, seconds, *_ in core._reduced_street_offsets(*ORIGIN, False, modes)
    ]
    walking = core._travel_times_with_access(access, "2022-02-22", "08:30:00", 7)
    rented = core._travel_times_with_access(
        access, "2022-02-22", "08:30:00", 7, transfer_mode=("e_scooter", 600.0)
    )
    # Identical access seeds, one relaxed set swapped for the other: the
    # merged set is a superset of the walking closure, so arrivals are
    # monotone — never later, somewhere strictly earlier.
    assert set(walking) <= set(rented)
    assert all(rented[stop] <= seconds for stop, seconds in walking.items())
    assert any(rented[stop] < seconds for stop, seconds in walking.items())


def test_rented_transfer_legs_split_from_their_tokens(multimodal_transfers_network):
    pytest.importorskip("cafein._cafein")

    journeys = multimodal_transfers_network.route_between_coordinates(
        ORIGIN, DEST, "2022-02-22", "08:30:00", street_policy=_transfer_policy()
    )
    scooter = [
        leg
        for journey in journeys
        for leg in journey["legs"]
        if leg["type"] == "transfer" and leg.get("mode") == "e_scooter"
    ]
    assert scooter, "somewhere the rental beats walking a transfer"
    for leg in scooter:
        assert leg["arrival_s"] > leg["departure_s"]
        assert leg["distance_m"] == pytest.approx(
            leg["network_distance_m"] + leg["connector_distance_m"]
        )
    matched = 0
    for journey in journeys:
        runs, run = [], []
        for leg in journey["legs"]:
            if leg["type"] == "transfer":
                run.append(leg)
            elif run:
                runs.append(run)
                run = []
        if run:
            runs.append(run)
        for run in runs:
            rides = [leg for leg in run if leg.get("mode") == "e_scooter"]
            if len(rides) != 1:
                # A zero-ride composition can put the access-side and
                # egress-side transfer splits back to back; the outer
                # stop pairs are not recoverable from such a run.
                continue
            # The ride leg spans exactly the token's pickup-to-drop
            # stretch with the token's own distances; the token is keyed
            # by the relaxed edge's outer stop pair.
            (ride,) = rides
            token = multimodal_transfers_network._core._mode_transfer_token(
                run[0]["from_stop"], run[-1]["to_stop"]
            )
            pickup, drop, ride_seconds, ride_network, ride_total = token[:5]
            assert (ride["from_stop"], ride["to_stop"]) == (pickup, drop)
            assert ride["arrival_s"] - ride["departure_s"] == ride_seconds
            assert ride["distance_m"] == pytest.approx(ride_total)
            assert ride["network_distance_m"] == pytest.approx(ride_network)
            matched += 1
        for before, after in zip(journey["legs"], journey["legs"][1:]):
            assert before["arrival_s"] <= after["departure_s"]
    assert matched, "at least one mid-journey rental transfer verifies its token"


def test_the_transfer_matrix_reconciles_with_single_queries(
    multimodal_transfers_network,
):
    pytest.importorskip("cafein._cafein")
    from cafein import TravelTimeMatrix

    policy = _transfer_policy()
    matrix = TravelTimeMatrix(
        multimodal_transfers_network,
        _points_frame([ORIGIN]),
        _points_frame([DEST]),
        "2022-02-22",
        "08:30:00",
        street_policy=policy,
    )
    journeys = multimodal_transfers_network.route_between_coordinates(
        ORIGIN, DEST, "2022-02-22", "08:30:00", street_policy=policy
    )
    fastest = min(j["arrival_s"] - j["departure_s"] for j in journeys)
    assert int(matrix["travel_time_s"].iloc[0]) == fastest


def test_the_cost_matrix_attributes_rental_transfers(multimodal_transfers_network):
    pytest.importorskip("cafein._cafein")
    from cafein import DetailedItineraries, TravelCostMatrix, TravelTimeMatrix

    args = (
        multimodal_transfers_network,
        _points_frame([ORIGIN]),
        _points_frame([DEST]),
        "2022-02-22",
        "08:30:00",
    )
    cost = TravelCostMatrix(*args, street_policy=_transfer_policy())
    time = TravelTimeMatrix(*args, street_policy=_transfer_policy())
    walking = TravelCostMatrix(
        *args, street_policy=StreetLegPolicy(access={"walk": 900}, egress={"walk": 900})
    )
    row = cost.iloc[0]
    # The same engine and set as the time matrix, cell for cell.
    assert row["travel_time_s"] == time.iloc[0]["travel_time_s"]
    assert row["travel_time_s"] < walking.iloc[0]["travel_time_s"]
    # The winning journey rides rental transfers: their street meters
    # and grams are attributed, never silently walked.
    assert row["street_distance_m"] > 0.0
    itinerary = DetailedItineraries(
        *args,
        street_policy=_transfer_policy(),
        geometries=False,
    )
    fastest = min(
        (legs for _, legs in itinerary.groupby("option")),
        key=lambda legs: legs["arrival_s"].max(),
    )
    scooter = fastest[fastest["mode"] == "e_scooter"]
    assert not scooter.empty
    assert row["emissions"] == pytest.approx(fastest["emissions"].sum(), rel=1e-6)
    assert row["street_distance_m"] == pytest.approx(
        scooter["distance_m"].sum(), rel=1e-6
    )
    with pytest.raises(ValueError, match="exclusion-aware"):
        TravelCostMatrix(
            *args,
            street_policy=_transfer_policy(),
            exclude_stops=["1230109"],
        )


def test_stop_exclusions_do_not_combine_with_transfers(multimodal_transfers_network):
    pytest.importorskip("cafein._cafein")
    from cafein import TravelTimeMatrix

    with pytest.raises(ValueError, match="exclusion-aware"):
        multimodal_transfers_network.route_between_coordinates(
            ORIGIN,
            DEST,
            "2022-02-22",
            "08:30:00",
            street_policy=_transfer_policy(),
            exclude_stops=["1230109"],
        )
    with pytest.raises(ValueError, match="exclusion-aware"):
        TravelTimeMatrix(
            multimodal_transfers_network,
            _points_frame([ORIGIN]),
            _points_frame([DEST]),
            "2022-02-22",
            "08:30:00",
            street_policy=_transfer_policy(),
            exclude_stops=["1230109"],
        )


def test_replacing_the_closure_drops_the_merged_set(multimodal_network, artifact_cache):
    pytest.importorskip("cafein._cafein")
    from cafein import TransportNetwork

    network = TransportNetwork.load(artifact_cache / "helsinki-multimodal.cafein")
    network.compute_mode_transfers("e_scooter", 600)
    assert network._core._mode_transfer_binding is not None
    # The merged set folded the replaced closure; it must not survive.
    network._core.set_transfers([])
    assert network._core._mode_transfer_binding is None
    with pytest.raises(ValueError, match="compute_mode_transfers"):
        network.route_between_coordinates(
            ORIGIN, DEST, "2022-02-22", "08:30:00", street_policy=_transfer_policy()
        )


def test_mc_transfers_join_the_frontier(multimodal_transfers_network):
    pytest.importorskip("cafein._cafein")
    from cafein import DetailedItineraries

    def frame(policy):
        return DetailedItineraries(
            multimodal_transfers_network,
            _points_frame([ORIGIN]),
            _points_frame([DEST]),
            "2022-02-22",
            "08:30:00",
            candidates="pareto",
            street_policy=policy,
            factors=_scooter_factor_rows(),
            geometries=False,
        )

    walking = frame(StreetLegPolicy(access={"walk": 900}, egress={"walk": 900}))
    rented = frame(_transfer_policy())
    scooter = rented[
        (rented["leg_type"] == "transfer") & (rented["mode"] == "e_scooter")
    ]
    assert not scooter.empty
    # The ride legs carry the shared fleet's grams over their ridden
    # network meters, exactly as the dominance ranked them.
    assert scooter["emissions"].equals(scooter["network_distance_m"] / 1000.0 * 21.0)
    options = rented.groupby("option").agg(
        departure=("departure_s", "min"),
        arrival=("arrival_s", "max"),
        grams=("emissions", "sum"),
    )
    durations = (options["arrival"] - options["departure"]).tolist()
    grams = options["grams"].tolist()
    assert durations == sorted(durations)
    assert grams == sorted(grams, reverse=True)
    # More than one option with a rental beside cleaner slower ones —
    # the frontier genuinely trades time against rental grams. (The
    # option count may shrink against the walking frontier: a rental
    # option can evict a same-bucket slower walking one.)
    assert len(options) > 1
    best = lambda f: (f.groupby("option")["arrival_s"].max()).min()  # noqa: E731
    assert best(rented) <= best(walking)


def test_mc_relaxed_candidates_take_transfers(multimodal_transfers_network):
    pytest.importorskip("cafein._cafein")
    from cafein import DetailedItineraries

    frame = DetailedItineraries(
        multimodal_transfers_network,
        _points_frame([ORIGIN]),
        _points_frame([DEST]),
        "2022-02-22",
        "08:30:00",
        candidates="relaxed",
        slack_seconds=120,
        street_policy=_transfer_policy(),
        factors=_scooter_factor_rows(),
        geometries=False,
    )
    scooter = frame[(frame["leg_type"] == "transfer") & (frame["mode"] == "e_scooter")]
    assert not scooter.empty


def test_mc_transfer_bindings_are_checked(
    multimodal_network, multimodal_transfers_network
):
    pytest.importorskip("cafein._cafein")
    from cafein import DetailedItineraries

    def query(network, policy, **kwargs):
        return DetailedItineraries(
            network,
            _points_frame([ORIGIN]),
            _points_frame([DEST]),
            "2022-02-22",
            "08:30:00",
            candidates="pareto",
            street_policy=policy,
            factors=_scooter_factor_rows(),
            geometries=False,
            **kwargs,
        )

    with pytest.raises(ValueError, match="compute_mode_transfers"):
        query(multimodal_network, _transfer_policy())
    with pytest.raises(ValueError, match="bound to"):
        query(multimodal_transfers_network, _transfer_policy(budget=500))
    with pytest.raises(ValueError, match="exclusion-aware"):
        query(
            multimodal_transfers_network,
            _transfer_policy(),
            exclude_stops=["1230109"],
        )
    with pytest.raises(ValueError, match="McRAPTOR"):
        multimodal_transfers_network._core._mc_route_with_access(
            [("1230109", 0, 0.0, False)],
            [("1391124", 0, 0.0)],
            "2022-02-22",
            "08:30:00",
            [],
            7,
            25.0,
            router="tbtr",
            transfer_mode=("e_scooter", 600.0, 0.021),
        )


def test_the_merged_set_survives_the_artifact(multimodal_transfers_network, tmp_path):
    pytest.importorskip("cafein._cafein")
    from cafein import TransportNetwork
    from cafein import streets as _streets
    from cafein.policy import reduction_modes

    path = tmp_path / "with-transfers.cafein"
    multimodal_transfers_network.save(path)
    loaded = TransportNetwork.load(path)
    # Double-saving the loaded network stays byte-identical: its token
    # map lands in a freshly seeded HashMap, so this exercises the
    # key-sorted canonicalisation rather than one map's repeated
    # iteration order. (Byte identity across the load cycle itself has
    # never been a feed guarantee.)
    once = tmp_path / "again.cafein"
    twice = tmp_path / "and-again.cafein"
    loaded.save(once)
    loaded.save(twice)
    assert once.read_bytes() == twice.read_bytes()

    assert (
        loaded._core._mode_transfer_binding
        == multimodal_transfers_network._core._mode_transfer_binding
    )
    # The loaded set keeps the unclosed marking: identical access seeds,
    # arrivals monotone against the walking closure — never later,
    # somewhere strictly earlier (a set silently loaded as a closure
    # would lose shadowed walk extensions and fail this).
    core = loaded._core
    modes = reduction_modes(
        _transfer_policy(), "access", _streets.MAX_ACCESS_EGRESS_TIME
    )
    access = [
        (stop, seconds)
        for stop, seconds, *_ in core._reduced_street_offsets(*ORIGIN, False, modes)
    ]
    walking = core._travel_times_with_access(access, "2022-02-22", "08:30:00", 7)
    rented = core._travel_times_with_access(
        access, "2022-02-22", "08:30:00", 7, transfer_mode=("e_scooter", 600.0)
    )
    assert all(rented[stop] <= seconds for stop, seconds in walking.items())
    assert any(rented[stop] < seconds for stop, seconds in walking.items())
    # And the journeys match the saving network's, tokens intact.
    before = multimodal_transfers_network.route_between_coordinates(
        ORIGIN, DEST, "2022-02-22", "08:30:00", street_policy=_transfer_policy()
    )
    after = loaded.route_between_coordinates(
        ORIGIN, DEST, "2022-02-22", "08:30:00", street_policy=_transfer_policy()
    )
    assert before == after


def test_rental_ride_legs_draw_their_shape(multimodal_transfers_network):
    pytest.importorskip("cafein._cafein")
    from shapely import wkb

    journeys = multimodal_transfers_network.route_between_coordinates(
        ORIGIN,
        DEST,
        "2022-02-22",
        "08:30:00",
        street_policy=_transfer_policy(),
        geometries=True,
    )
    rides = [
        leg
        for journey in journeys
        for leg in journey["legs"]
        if leg["type"] == "transfer" and leg.get("mode") == "e_scooter"
    ]
    assert rides
    for leg in rides:
        assert leg["geometry"] is not None
        shape = wkb.loads(bytes(leg["geometry"]))
        assert shape.geom_type == "LineString"
        assert len(shape.coords) > 1


def test_merged_sets_are_engine_neutral(multimodal_transfers_network):
    pytest.importorskip("cafein._cafein")
    from cafein import streets as _streets
    from cafein.policy import reduction_modes

    # The same reduced access array under the merged set through RAPTOR
    # and the trip-based engine: identical arrivals at every stop — both
    # relax shadowed transit arrivals exactly.
    core = multimodal_transfers_network._core
    modes = reduction_modes(
        _transfer_policy(), "access", _streets.MAX_ACCESS_EGRESS_TIME
    )
    access = [
        (stop, seconds)
        for stop, seconds, *_ in core._reduced_street_offsets(*ORIGIN, False, modes)
    ]
    binding = ("e_scooter", 600.0)
    raptor = core._travel_times_with_access(
        access, "2022-02-22", "08:30:00", 7, transfer_mode=binding
    )
    tbtr = core._travel_times_with_access(
        access, "2022-02-22", "08:30:00", 7, router="tbtr", transfer_mode=binding
    )
    assert raptor == tbtr


def test_policy_time_queries_auto_ride_the_cached_tbtr_set(
    multimodal_network, artifact_cache
):
    pytest.importorskip("cafein._cafein")
    from cafein import TransportNetwork
    from cafein import streets as _streets
    from cafein.policy import reduction_modes

    # A private copy: computing the caches mutates network state.
    network = TransportNetwork.load(artifact_cache / "helsinki-multimodal.cafein")
    network.compute_mode_transfers("e_scooter", 600)
    core = network._core
    modes = reduction_modes(
        _transfer_policy(), "access", _streets.MAX_ACCESS_EGRESS_TIME
    )
    access = [
        (stop, seconds)
        for stop, seconds, *_ in core._reduced_street_offsets(*ORIGIN, False, modes)
    ]
    binding = ("e_scooter", 600.0)
    args = (access, "2022-02-22", "08:30:00", 7)
    before = core._travel_times_with_access(*args, transfer_mode=binding)
    from cafein import TravelTimeMatrix

    def public_frame():
        return TravelTimeMatrix(
            network,
            _points_frame([ORIGIN]),
            _points_frame([DEST]),
            "2022-02-22",
            "08:30:00",
            street_policy=_transfer_policy(),
        )

    frame_before = public_frame()
    # With the whole-day set cached, auto rides the trip-based engine —
    # the merged binding included, answering identically.
    core.compute_tbtr_transfers("2022-02-22")
    assert core._travel_times_with_access(*args, transfer_mode=binding) == before
    explicit = core._travel_times_with_access(
        *args, router="raptor", transfer_mode=binding
    )
    assert explicit == before
    rows = core._time_matrix_with_access(
        [access], [[("1391124", 0)]], "2022-02-22", "08:30:00", 7, transfer_mode=binding
    )
    raptor_rows = core._time_matrix_with_access(
        [access],
        [[("1391124", 0)]],
        "2022-02-22",
        "08:30:00",
        7,
        transfer_mode=binding,
        router="raptor",
    )
    assert rows == raptor_rows
    # The public surface: an explicit router beside street_policy stays
    # rejected — auto resolution is internal, never a user knob here —
    # and the cached set leaves the public matrix's answers untouched.
    with pytest.raises(ValueError, match="router"):
        TravelTimeMatrix(
            network,
            _points_frame([ORIGIN]),
            _points_frame([DEST]),
            "2022-02-22",
            "08:30:00",
            street_policy=_transfer_policy(),
            router="raptor",
        )
    assert public_frame().equals(frame_before)


def test_the_carriage_set_builds_binds_and_persists(
    multimodal_network, artifact_cache, tmp_path
):
    pytest.importorskip("cafein._cafein")
    from cafein import TransportNetwork

    network = TransportNetwork.load(artifact_cache / "helsinki-multimodal.cafein")
    core = network._core
    tri = core._trip_bikes_allowed()
    assert len(tri) > 0
    assert set(map(type, tri)) <= {bool, type(None)}
    edges, rides = core._compute_carriage_transfers("bicycle", 600)
    assert core._carriage_transfer_binding == ("bicycle", 600.0, edges, rides)
    walking = len(core._transfer_edges())
    # Rides only replace strictly faster pairs or add new ones.
    assert edges >= walking
    assert 0 < rides <= edges
    # Persistence: the binding and arrays survive the artifact.
    path = tmp_path / "with-carriage.cafein"
    network.save(path)
    loaded = TransportNetwork.load(path)
    assert loaded._core._carriage_transfer_binding == (
        "bicycle",
        600.0,
        edges,
        rides,
    )
    # Replacing the closure drops the set, as with the merged set.
    core.set_transfers([])
    assert core._carriage_transfer_binding is None


def _carried(**terms):
    return VehiclePolicy(
        source="own",
        side="origin",
        facilities="any_stop",
        take_aboard=True,
        **terms,
    )


def _carriage_policy(vehicle, transfers=None, egress=None):
    kwargs = dict(
        access={"bicycle": 600, "walk": 900},
        egress=egress or {"walk": 900},
        vehicles={"bicycle": vehicle},
    )
    if transfers:
        kwargs["transfers"] = transfers
    return StreetLegPolicy(**kwargs)


def test_carriage_travel_times_are_pure_option_value(
    multimodal_network, artifact_cache
):
    pytest.importorskip("cafein._cafein")
    from cafein import TransportNetwork

    network = TransportNetwork.load(artifact_cache / "helsinki-multimodal.cafein")
    baseline_policy = StreetLegPolicy(
        access={"bicycle": 600, "walk": 900},
        egress={"walk": 900},
        vehicles={
            "bicycle": VehiclePolicy(source="own", side="origin", facilities="any_stop")
        },
    )
    args = (ORIGIN, "2022-02-22", "08:30:00")
    baseline = network.travel_times_from_coordinate(
        *args, street_policy=baseline_policy
    )
    # The conservative default on an all-unknown feed: carrying can
    # never board, so carriage equals the no-carriage baseline exactly.
    forbid = network.travel_times_from_coordinate(
        *args, street_policy=_carriage_policy(_carried())
    )
    assert forbid == baseline
    # With the carriage set and the own-transfers grant under
    # unknown='allow', the carried bicycle rides between stops: never
    # worse anywhere, strictly better somewhere.
    network._core._compute_carriage_transfers("bicycle", 900)
    granted = network.travel_times_from_coordinate(
        *args,
        street_policy=_carriage_policy(
            _carried(unknown_bike_trips="allow"), transfers={"bicycle": 900}
        ),
    )
    assert all(granted[stop] <= seconds for stop, seconds in baseline.items())
    assert any(granted[stop] < seconds for stop, seconds in baseline.items())
    # Restricting the parking facilities only removes Free options.
    restricted = network.travel_times_from_coordinate(
        *args,
        street_policy=_carriage_policy(
            VehiclePolicy(
                source="own",
                side="origin",
                facilities=("1230109",),
                take_aboard=True,
                unknown_bike_trips="allow",
            ),
            transfers={"bicycle": 900},
        ),
    )
    assert all(
        restricted.get(stop, 1 << 30) >= seconds for stop, seconds in granted.items()
    )
    # Facilities govern parking only: the carried access still ends at
    # any stop, so the restricted policy still beats the baseline
    # somewhere through carried movement.
    assert any(
        restricted.get(stop, 1 << 30) < seconds for stop, seconds in baseline.items()
    )
    # A bicycle-only access side still walks (pushing) at the default
    # walking budget: identical to granting that budget explicitly.
    from cafein import streets as _streets

    def _side_policy(access):
        return StreetLegPolicy(
            access=access,
            egress={"walk": 900},
            transfers={"bicycle": 900},
            vehicles={"bicycle": _carried(unknown_bike_trips="allow")},
        )

    bike_only = network.travel_times_from_coordinate(
        *args, street_policy=_side_policy({"bicycle": 600})
    )
    explicit = network.travel_times_from_coordinate(
        *args,
        street_policy=_side_policy(
            {"bicycle": 600, "walk": _streets.MAX_ACCESS_EGRESS_TIME}
        ),
    )
    assert bike_only == explicit
    # Binding discipline: the transfers grant needs the exact set.
    with pytest.raises(ValueError, match="bound to"):
        network.travel_times_from_coordinate(
            *args,
            street_policy=_carriage_policy(
                _carried(unknown_bike_trips="allow"), transfers={"bicycle": 600}
            ),
        )


def test_carriage_rejects_the_matrix_surfaces(multimodal_transfers_network):
    pytest.importorskip("cafein._cafein")
    from cafein import DetailedItineraries, TravelCostMatrix

    policy = _carriage_policy(_carried())
    # A walking-only carriage policy must not slip through the legacy
    # fast paths either.
    walk_only = StreetLegPolicy(
        access={"walk": 900},
        egress={"walk": 900},
        vehicles={"bicycle": _carried()},
    )
    frames = (_points_frame([ORIGIN]), _points_frame([DEST]))
    args = (*frames, "2022-02-22", "08:30:00")
    with pytest.raises(ValueError, match="not wired into matrix computation"):
        TravelCostMatrix(multimodal_transfers_network, *args, street_policy=walk_only)
    with pytest.raises(ValueError, match="not wired into matrix computation"):
        TravelCostMatrix(multimodal_transfers_network, *args, street_policy=policy)
    with pytest.raises(ValueError, match="the multicriteria candidates"):
        DetailedItineraries(
            multimodal_transfers_network,
            *args,
            candidates="pareto",
            street_policy=policy,
            factors=_scooter_factor_rows(),
            geometries=False,
        )
    # The walking-only fast path must not slip past the multicriteria
    # rejection either.
    with pytest.raises(ValueError, match="the multicriteria candidates"):
        DetailedItineraries(
            multimodal_transfers_network,
            *args,
            candidates="pareto",
            street_policy=walk_only,
            factors=_scooter_factor_rows(),
            geometries=False,
        )


# A destination whose best carriage journey must park mid-way: the
# probe over strictly-improved stops found this one rides carried,
# parks, and continues on a bike-forbidden trip under walk-only egress.
PARK_DEST = (60.218887, 24.812701)


def test_carriage_routes_carry_park_and_decorate(multimodal_network, artifact_cache):
    pytest.importorskip("cafein._cafein")
    from cafein import TransportNetwork

    network = TransportNetwork.load(artifact_cache / "helsinki-multimodal.cafein")
    network.compute_carriage_transfers("bicycle", 900)
    both_ends = _carriage_policy(
        _carried(unknown_bike_trips="allow"),
        transfers={"bicycle": 900},
        egress={"bicycle": 600, "walk": 900},
    )
    baseline_policy = StreetLegPolicy(
        access={"bicycle": 600, "walk": 900},
        egress={"walk": 900},
        vehicles={
            "bicycle": VehiclePolicy(source="own", side="origin", facilities="any_stop")
        },
    )
    args = (ORIGIN, DEST, "2022-02-22", "08:30:00")
    baseline = network.route_between_coordinates(*args, street_policy=baseline_policy)
    carried = network.route_between_coordinates(*args, street_policy=both_ends)
    assert carried
    # The (arrival, rides) frontier: strictly improving arrivals.
    arrivals = [journey["arrival_s"] for journey in carried]
    assert arrivals == sorted(arrivals, reverse=True)
    journey = min(carried, key=lambda journey: journey["arrival_s"])
    assert journey["arrival_s"] < min(entry["arrival_s"] for entry in baseline)
    legs = journey["legs"]
    transit = [leg for leg in legs if leg["type"] == "transit"]
    assert transit and all(leg["bike_aboard"] for leg in transit)
    assert all(leg["distance_m"] > 0 for leg in transit)
    assert all(leg["geometry"] is not None for leg in transit)
    assert legs[0]["mode"] == "bicycle" and legs[-1]["mode"] == "bicycle"
    # Walk-only egress to the probed destination forces a mid-journey
    # park: carried riding first, the bicycle left at a stop, and every
    # later boarding bike-free.
    walk_egress = _carriage_policy(
        _carried(unknown_bike_trips="allow"), transfers={"bicycle": 900}
    )
    journey = min(
        network.route_between_coordinates(
            ORIGIN, PARK_DEST, "2022-02-22", "08:30:00", street_policy=walk_egress
        ),
        key=lambda journey: journey["arrival_s"],
    )
    kinds = [leg["type"] for leg in journey["legs"]]
    assert "park" in kinds
    parked = kinds.index("park")
    ride = next(leg for leg in journey["legs"] if leg.get("mode") == "bicycle")
    assert ride["network_distance_m"] > 0
    assert journey["legs"][parked]["stop"]
    assert all(
        not leg.get("bike_aboard", False) for leg in journey["legs"][parked + 1 :]
    )
    assert journey["legs"][-1]["mode"] == "walk"


def test_carriage_itineraries_render_the_frame(multimodal_network, artifact_cache):
    pytest.importorskip("cafein._cafein")
    from cafein import DetailedItineraries, TransportNetwork

    network = TransportNetwork.load(artifact_cache / "helsinki-multimodal.cafein")
    network.compute_carriage_transfers("bicycle", 900)
    both_ends = _carriage_policy(
        _carried(unknown_bike_trips="allow"),
        transfers={"bicycle": 900},
        egress={"bicycle": 600, "walk": 900},
    )
    frame = DetailedItineraries(
        network,
        _points_frame([ORIGIN]),
        _points_frame([DEST]),
        "2022-02-22",
        "08:30:00",
        street_policy=both_ends,
        geometries=False,
    )
    assert "bike_aboard" in frame.columns
    assert frame["bike_aboard"].any()
    assert (frame["mode"] == "bicycle").any()
    assert frame.loc[frame["leg_type"] == "transit", "emissions"].notna().all()
    # The parked journey renders its zero-length park row in place.
    walk_egress = _carriage_policy(
        _carried(unknown_bike_trips="allow"), transfers={"bicycle": 900}
    )
    frame = DetailedItineraries(
        network,
        _points_frame([ORIGIN]),
        _points_frame([PARK_DEST]),
        "2022-02-22",
        "08:30:00",
        street_policy=walk_egress,
        geometries=False,
    )
    park = frame[frame["leg_type"] == "park"]
    assert len(park) >= 1
    assert (park["from_stop"] == park["to_stop"]).all()
    assert park["mode"].isna().all()
    assert (park["travel_time_s"] == 0).all()


def test_carriage_time_matrix_matches_the_route_surface(
    multimodal_network, artifact_cache
):
    pytest.importorskip("cafein._cafein")
    from cafein import TransportNetwork, TravelTimeMatrix

    network = TransportNetwork.load(artifact_cache / "helsinki-multimodal.cafein")
    network.compute_carriage_transfers("bicycle", 900)
    policy = _carriage_policy(
        _carried(unknown_bike_trips="allow"),
        transfers={"bicycle": 900},
        egress={"bicycle": 600, "walk": 900},
    )
    frames = (_points_frame([ORIGIN]), _points_frame([DEST, PARK_DEST]))
    args = (*frames, "2022-02-22", "08:30:00")
    matrix = TravelTimeMatrix(network, *args, street_policy=policy)
    departed = 8 * 3600 + 30 * 60
    # Every cell is the route surface's best arrival, door to door.
    for index, destination in enumerate([DEST, PARK_DEST]):
        best = (
            min(
                journey["arrival_s"]
                for journey in network.route_between_coordinates(
                    ORIGIN,
                    destination,
                    "2022-02-22",
                    "08:30:00",
                    street_policy=policy,
                )
            )
            - departed
        )
        cell = matrix.loc[matrix["to_id"] == f"p{index}", "travel_time_s"]
        assert int(cell.iloc[0]) == best
    # Pure option value at the matrix level: the forbid default equals
    # the no-carriage baseline exactly.
    baseline_policy = StreetLegPolicy(
        access={"bicycle": 600, "walk": 900},
        egress={"walk": 900},
        vehicles={
            "bicycle": VehiclePolicy(source="own", side="origin", facilities="any_stop")
        },
    )
    forbid = TravelTimeMatrix(
        network, *args, street_policy=_carriage_policy(_carried())
    )
    baseline = TravelTimeMatrix(network, *args, street_policy=baseline_policy)
    pd.testing.assert_frame_equal(pd.DataFrame(forbid), pd.DataFrame(baseline))
    # A walking-only carriage policy rides the carriage engine while
    # the plain one keeps the legacy fast path (which snaps on the
    # walking graph): every pair the legacy matrix reaches must agree
    # exactly, and the carriage cells cover at least that set — the
    # multimodal snap reaches PARK_DEST where the walking graph
    # cannot.
    walk_only_carriage = StreetLegPolicy(
        access={"walk": 900},
        egress={"walk": 900},
        vehicles={"bicycle": _carried()},
    )
    walk_only = StreetLegPolicy(access={"walk": 900}, egress={"walk": 900})
    wide = (
        _points_frame([ORIGIN, (60.1699, 24.9384)]),
        _points_frame([DEST, PARK_DEST, (60.1866, 24.9600)]),
    )
    carriage_cells = TravelTimeMatrix(
        network, *wide, "2022-02-22", "08:30:00", street_policy=walk_only_carriage
    )
    plain_cells = TravelTimeMatrix(
        network, *wide, "2022-02-22", "08:30:00", street_policy=walk_only
    )
    merged = plain_cells.merge(
        carriage_cells, on=["from_id", "to_id"], suffixes=("_plain", "_carriage")
    )
    assert len(merged) == len(plain_cells)
    assert (merged["travel_time_s_plain"] == merged["travel_time_s_carriage"]).all()
    assert len(carriage_cells) > len(plain_cells)
    # Exclusions are rejected rather than silently dropped.
    with pytest.raises(ValueError, match="does not combine with exclusions"):
        TravelTimeMatrix(
            network, *args, street_policy=policy, exclude_stops=["1230109"]
        )
    # A cached trip-based set never claims the carriage query: the
    # cells are identical with and without it.
    network._core.compute_tbtr_transfers("2022-02-22")
    cached = TravelTimeMatrix(network, *args, street_policy=policy)
    pd.testing.assert_frame_equal(pd.DataFrame(cached), pd.DataFrame(matrix))
