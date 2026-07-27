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
    with pytest.raises(ValueError, match="cannot be an egress"):
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


def test_carrying_aboard_and_non_walk_transfers_are_not_yet():
    with pytest.raises(ValueError, match="take_aboard"):
        own(take_aboard=True)
    with pytest.raises(ValueError, match="not consumed yet"):
        StreetLegPolicy(transfers={"bicycle": 900}, vehicles={"bicycle": own()})


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


def test_transfers_and_string_selectors_are_not_yet():
    with pytest.raises(ValueError, match="not consumed yet"):
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
