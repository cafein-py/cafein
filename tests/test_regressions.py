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
