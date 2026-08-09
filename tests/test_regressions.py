"""Regression tests guarding specific fixed bugs.

One test per fixed defect; add new ones here rather than in a new file.
"""

import pytest

from cafein import exhaustive_frontier, journey_frontier


def test_mcraptor_window_profile_keeps_cleaner_earlier_journeys(network_with_footpaths):
    """McRAPTOR's departure-window emissions profile must not drop an
    undominated cleaner-but-earlier-departing journey.

    The per-stop label bag is cumulative across the descending profile
    passes; before the fix its dominance ignored the rides used to reach
    a stop, so a later-departure journey that reached an intermediate
    stop with more transfers could suppress an earlier-departure journey
    that reached it with fewer — and thus still had the transfer budget
    for a cleaner continuation. On this pair that dropped the cleanest
    journey entirely: the window frontier collapsed to the single
    latest-departing (dirtiest) point.

    The exhaustive oracle at 08:30 (inside the window) pins the true
    minimum, so McRAPTOR over the window must reach a journey no dirtier
    than it, and must return more than the one dirtiest point.
    """
    origin, destination = "1010419", "4240227"
    oracle = exhaustive_frontier(
        network_with_footpaths,
        origin,
        destination,
        "2022-02-22",
        "08:30:00",
        max_transfers=4,
    )
    cleanest = oracle["emissions"].min()

    frontier = journey_frontier(
        network_with_footpaths,
        origin,
        destination,
        "2022-02-22",
        "08:30:00",
        window=900,
        max_transfers=4,
        candidates="pareto",
        bucket=1e-6,
        router="raptor",
    )
    on_frontier = frontier[frontier["frontier"]]
    transit = on_frontier[on_frontier["rides"] >= 1]

    assert transit["emissions"].min() == pytest.approx(cleanest, abs=1e-3)
    assert len(transit) > 1


def test_mctbtr_window_profile_keeps_cleaner_earlier_journeys(network_with_footpaths):
    """The McTBTR profile has the same cross-pass hazard as McRAPTOR, in
    the stop bags that gate its query-time footpath relaxation.

    Those bags dominate on (arrival, emissions bucket) and persist across
    the descending profile passes; before the fix they ignored the rides
    used to reach a stop, so a later-departure arrival on more rides could
    suppress a cleaner fewer-rides arrival, and the walk that would reach
    a cleaner onward leg was never relaxed. On this pair McTBTR kept a
    journey a few grams dirtier than the oracle's cleanest; ranking the
    rides in the stop bags restores completeness.
    """
    origin, destination = "1010108", "3170218"
    oracle = exhaustive_frontier(
        network_with_footpaths,
        origin,
        destination,
        "2022-02-22",
        "08:30:00",
        max_transfers=4,
    )
    cleanest = oracle["emissions"].min()

    frontier = journey_frontier(
        network_with_footpaths,
        origin,
        destination,
        "2022-02-22",
        "08:30:00",
        window=900,
        max_transfers=4,
        candidates="pareto",
        bucket=1e-6,
        router="tbtr",
    )
    transit = frontier[frontier["frontier"] & (frontier["rides"] >= 1)]

    assert transit["emissions"].min() == pytest.approx(cleanest, abs=1e-3)


def test_factor_loaders_reject_infinities():
    # Both factor loaders accepted +inf component values, which priced
    # infinite emissions instead of failing loudly; NA stays the marker
    # for an unresolved component.
    import pandas as pd
    import pytest

    from cafein import emissions

    street = pd.DataFrame(
        [
            {
                "street_mode": "car",
                "vehicle_class": "ICE",
                "service_model": "private",
                "vehicle": float("inf"),
                "fuel": 0.0,
                "infrastructure": 0.0,
                "operations": 0.0,
            }
        ]
    )
    with pytest.raises(ValueError, match="non-finite"):
        emissions.load_street_factors(street)
    transit = pd.DataFrame([{"route_type": 3, "vehicle": float("inf")}])
    with pytest.raises(ValueError, match="non-finite"):
        emissions.load_factors(transit)


def _install_two_edge_multimodal_graph(network):
    """A walk+e-scooter edge south of the first stop and a bicycle-only
    edge north of it, installed directly — the synthetic multimodal graph
    the mode-separation test uses, with every edge index and mask known."""
    stop, lat, lon = next((s, la, lo) for s, la, lo in network.stops if la is not None)
    south, north = lat - 0.0005, lat + 0.0011
    zeros = [0, 0]
    network._core.set_multimodal_streets(
        ["walk", "bicycle", "e_scooter"],
        4,
        [(0, 1, 200.0), (2, 3, 200.0)],
        [0, 2, 4],
        [lon - 0.001, lon + 0.001, lon - 0.001, lon + 0.001],
        [south, south, north, north],
        zeros,
        zeros,
        zeros,
        zeros,
        [1 | 4, 2],  # south: walk + e_scooter; north: bicycle only
        [1 | 4, 2],
        zeros,
        zeros,
    )
    return stop, lat, lon, south


def test_multimodal_leg_validates_the_street_choice_token(fresh_footpaths_network):
    """The leg rebuild must reject a malformed ``StreetChoice`` token at
    the boundary instead of feeding it to the core.

    Before the fix the caller-supplied ``(edge, fraction, connector)``
    triple became a core ``Snap`` unchecked: an out-of-range edge index
    panicked through unchecked indexing rather than raising ``ValueError``,
    and a non-finite or out-of-range fraction or a negative connector
    produced invalid or understated costs silently. An edge the resolved
    profile cannot traverse was accepted too, seeding the search from an
    arc the mode may not use.
    """
    network = fresh_footpaths_network
    stop, lat, lon, south = _install_two_edge_multimodal_graph(network)
    core = network._core
    rows = {r[0]: r for r in core._street_access_seconds(south, lon, "bicycle", 600.0)}
    _, _, edge, fraction, connector = rows[stop]
    good = core._multimodal_leg(
        south, lon, "bicycle", stop, edge, fraction, connector, False, 600.0, False
    )
    assert good is not None
    malformed = [
        (999, fraction, connector, "out of range"),
        (edge, float("nan"), connector, "fraction"),
        (edge, 1.5, connector, "fraction"),
        (edge, fraction, -1.0, "connector"),
        (edge, fraction, float("inf"), "connector"),
    ]
    for bad_edge, bad_fraction, bad_connector, message in malformed:
        with pytest.raises(ValueError, match=message):
            core._multimodal_leg(
                south,
                lon,
                "bicycle",
                stop,
                bad_edge,
                bad_fraction,
                bad_connector,
                False,
                600.0,
                False,
            )
    # The walk+e-scooter edge is a valid index the bicycle profile may not use.
    walk_rows = {
        r[0]: r for r in core._street_access_seconds(south, lon, "walk", 600.0)
    }
    walk_edge = walk_rows[stop][2]
    assert walk_edge != edge
    with pytest.raises(ValueError, match="not traversable"):
        core._multimodal_leg(
            south, lon, "bicycle", stop, walk_edge, 0.5, 1.0, False, 600.0, False
        )


def test_multimodal_zero_shortcut_validates_snap_and_cutoff(fresh_footpaths_network):
    """The equal-coordinate shortcut on the multimodal access surface must
    snap and check the cutoff before reporting a zero leg.

    Before the fix equal origin and destination coordinates returned a
    zero-duration leg before either endpoint was snapped or ``max_seconds``
    validated, so an equal pair arbitrarily far from the network — or a
    query with a negative or NaN cutoff — was reported reachable, unlike
    the direct matrix and standalone travel-time surfaces.
    """
    network = fresh_footpaths_network
    stop, lat, lon, south = _install_two_edge_multimodal_graph(network)
    core = network._core
    # An equal pair far from every edge is unsnappable, not a zero leg.
    far = (lat + 1.0, lon + 1.0)
    assert core._multimodal_direct_leg(far, far, "walk", 600.0, False) is None
    # A snappable equal pair is a zero leg only when the cutoff admits one.
    origin = (south, lon)
    assert core._multimodal_direct_leg(origin, origin, "walk", -1.0, False) is None
    nan = float("nan")
    assert core._multimodal_direct_leg(origin, origin, "walk", nan, False) is None
    seconds, network_m, connector_m, _ = core._multimodal_direct_leg(
        origin, origin, "walk", 0.0, False
    )
    assert (seconds, network_m, connector_m) == (0, 0.0, 0.0)
    # The stop-coincident shortcut in the leg rebuild obeys the same gate.
    rows = {r[0]: r for r in core._street_access_seconds(lat, lon, "walk", 600.0)}
    _, zero_seconds, edge, fraction, connector = rows[stop]
    assert zero_seconds == 0
    assert (
        core._multimodal_leg(
            lat, lon, "walk", stop, edge, fraction, connector, False, -1.0, False
        )
        is None
    )
    parts = core._multimodal_leg(
        lat, lon, "walk", stop, edge, fraction, connector, False, 600.0, False
    )
    assert parts is not None and parts[0] == 0
