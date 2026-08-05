"""The member-way stitcher: chaining, orientation, refusal, ring rule."""

import pathlib
from types import SimpleNamespace

import pytest
import shapely

from cafein import _relations, _stitch

TRANSIT = pathlib.Path(__file__).parent / "data" / "helsinki-transit.osm.pbf"
GAP_FIXTURE = pathlib.Path(__file__).parent / "data" / "helsinki-transit-gap.osm.pbf"


def P(x, y):
    """A local grid point: one unit ≈ 100 m at Helsinki latitudes."""
    return (24.9 + x * 0.0018, 60.1 + y * 0.0009)


def member(points, tags=None, identifier=0):
    return SimpleNamespace(
        geometry=shapely.LineString(points), tags=tags or {}, id=identifier
    )


def unresolved(identifier=99):
    return SimpleNamespace(geometry=None, tags={}, id=identifier)


def refusal(members, **kwargs):
    with pytest.raises(_stitch.StitchRefusal) as caught:
        _stitch.stitch(members, **kwargs)
    return caught.value.reason


CHAIN = [
    member([P(0, 0), P(1, 0), P(2, 0)], identifier=1),
    member([P(2, 0), P(3, 0)], identifier=2),
    member([P(3, 0), P(4, 0), P(5, 0)], identifier=3),
]


def test_perfect_chain_stitches_in_order():
    line = _stitch.stitch(CHAIN)
    assert isinstance(line, shapely.LineString)
    assert line.coords[0] == P(0, 0)
    assert line.coords[-1] == P(5, 0)
    # Joints deduplicate: 8 input vertices, 2 shared.
    assert len(line.coords) == 6


def test_reversed_segment_orients():
    reversed_middle = [
        CHAIN[0],
        member(list(reversed([P(2, 0), P(3, 0)])), identifier=2),
        CHAIN[2],
    ]
    assert _stitch.stitch(reversed_middle).equals(_stitch.stitch(CHAIN))


def test_shuffled_members_chain_by_adjacency():
    shuffled = [CHAIN[2], CHAIN[0], CHAIN[1]]
    line = _stitch.stitch(shuffled)
    # Connectivity never trusts order; direction starts at the first
    # member's free end — here the far terminus.
    assert line.coords[0] == P(5, 0)
    assert line.coords[-1] == P(0, 0)
    assert line.reverse().equals(_stitch.stitch(CHAIN))


def test_single_member_passes_through():
    line = _stitch.stitch([CHAIN[0]])
    assert list(line.coords) == [P(0, 0), P(1, 0), P(2, 0)]


def test_gap_beyond_tolerance_refuses():
    gapped = [
        CHAIN[0],
        member([P(2.5, 0), P(3, 0)], identifier=2),  # ~50 m short
        CHAIN[2],
    ]
    assert refusal(gapped) == "gap"


def test_small_gap_within_tolerance_chains():
    nudged = [
        CHAIN[0],
        member([P(2.00005, 0.00002), P(3, 0)], identifier=2),  # < 1 m off
        CHAIN[2],
    ]
    line = _stitch.stitch(nudged)
    assert line.coords[-1] == P(5, 0)


def test_leftover_branch_refuses():
    # Order-first consumes the mainline; the dangling spur can never
    # connect — the chain does not cover all members.
    branching = CHAIN + [member([P(3, 0), P(3, 1)], identifier=4)]
    assert refusal(branching) == "gap"


def test_ambiguous_continuation_refuses():
    # Order broken (the next-in-order member is far away) and TWO
    # unused members connect at the chain end: more than one
    # continuation candidate.
    ambiguous = [
        CHAIN[0],
        member([P(9, 9), P(10, 9)], identifier=9),
        member([P(2, 0), P(3, 0)], identifier=2),
        member([P(2, 0), P(2, 1)], identifier=4),
    ]
    assert refusal(ambiguous) == "branching"


def test_unresolved_member_refuses():
    assert refusal([CHAIN[0], unresolved()]) == "unresolved-member"


def test_empty_refuses():
    assert refusal([]) == "empty"


RING = [P(10, 0), P(11, 1), P(12, 0), P(11, -1), P(10, 0)]
"""A diamond ring, stored counterclockwise-ish: 0 → top → 2 → bottom."""


def ring_member(tags, identifier=7):
    return member(RING, tags=tags, identifier=identifier)


APPROACH = member([P(8, 0), P(10, 0)], identifier=5)
EXIT = member([P(12, 0), P(14, 0)], identifier=6)


def test_ring_arc_follows_stored_orientation():
    line = _stitch.stitch([APPROACH, ring_member({"junction": "roundabout"}), EXIT])
    coordinates = list(line.coords)
    # Entry at ring vertex 0, exit at vertex 2: the stored-order arc
    # passes the top vertex (index 1), never the bottom one.
    assert P(11, 1) in coordinates
    assert P(11, -1) not in coordinates
    assert coordinates[0] == P(8, 0) and coordinates[-1] == P(14, 0)


def test_ring_arc_from_the_other_side_takes_the_complement():
    line = _stitch.stitch(
        [
            member([P(14, 0), P(12, 0)], identifier=6),
            ring_member({"oneway": "yes"}),
            member([P(10, 0), P(8, 0)], identifier=5),
        ]
    )
    coordinates = list(line.coords)
    # Travelling exit-side first: entry at vertex 2, stored-order arc
    # 2 → bottom → 0.
    assert P(11, -1) in coordinates
    assert P(11, 1) not in coordinates
    assert coordinates[0] == P(14, 0) and coordinates[-1] == P(8, 0)


def test_ring_without_direction_refuses():
    assert refusal([APPROACH, ring_member({}), EXIT]) == "ring-direction"


def test_bidirectional_ring_refuses():
    assert (
        refusal(
            [APPROACH, ring_member({"junction": "roundabout", "oneway": "no"}), EXIT]
        )
        == "ring-direction"
    )


def test_ring_with_one_touch_refuses():
    assert refusal([APPROACH, ring_member({"junction": "roundabout"})]) == "ring-touch"


def test_ring_with_three_touches_refuses():
    third = member([P(11, 1), P(11, 3)], identifier=8)
    assert refusal(
        [APPROACH, ring_member({"junction": "roundabout"}), EXIT, third]
    ) in ("ring-touch", "branching")


def test_shuffled_interior_seed_grows_both_ways():
    # The first listed member is the MIDDLE of the chain: the walk must
    # grow both sides, not report a false gap.
    interior_first = [CHAIN[1], CHAIN[2], CHAIN[0]]
    line = _stitch.stitch(interior_first)
    ends = {line.coords[0], line.coords[-1]}
    assert ends == {P(0, 0), P(5, 0)}
    assert len(line.coords) == 6


def test_malformed_members_refuse():
    empty_geometry = SimpleNamespace(geometry=shapely.LineString(), tags={}, id=41)
    assert refusal([empty_geometry]) == "malformed-member"
    zero_extent = member([P(0, 0), P(0, 0)], identifier=42)
    assert refusal([zero_extent]) == "malformed-member"
    bad = SimpleNamespace(
        geometry=shapely.LineString([(24.9, float("nan")), (24.91, 60.1)]),
        tags={},
        id=43,
    )
    assert refusal([bad]) == "malformed-member"


def test_reversed_oneway_ring_takes_the_reversed_arc():
    # oneway=-1 on a roundabout: travel runs AGAINST stored order, so
    # entry 0 → exit 2 passes the bottom vertex, never the top.
    line = _stitch.stitch(
        [
            APPROACH,
            ring_member({"junction": "roundabout", "oneway": "-1"}),
            EXIT,
        ]
    )
    coordinates = list(line.coords)
    assert P(11, -1) in coordinates
    assert P(11, 1) not in coordinates


def test_unrecognised_oneway_ring_refuses():
    assert (
        refusal(
            [
                APPROACH,
                ring_member({"junction": "roundabout", "oneway": "alternating"}),
                EXIT,
            ]
        )
        == "ring-direction"
    )


def test_reversed_seed_in_cycle_follows_order():
    # A route that returns to its start (cyclic): both seed endpoints
    # connect somewhere, so only the ordered-next member can orient the
    # reversed first way correctly.
    cycle = [
        member(list(reversed([P(0, 0), P(2, 0)])), identifier=1),  # reversed seed
        member([P(2, 0), P(2, 2)], identifier=2),
        member([P(2, 2), P(0, 2)], identifier=3),
        member([P(0, 2), P(0, 0)], identifier=4),
    ]
    line = _stitch.stitch(cycle)
    coordinates = list(line.coords)
    assert coordinates[0] == P(0, 0)
    assert coordinates[1] == P(2, 0)
    assert coordinates[-1] == P(0, 0)


def test_near_loop_open_way_is_not_a_ring():
    # Endpoints ~5 m apart but NOT identical: an ordinary member, not a
    # ring — no direction tags demanded, closure never invented.
    near_loop = member(
        [P(2, 0), P(3, 1), P(4, 0), P(3, -1), P(2.00003, 0.00002)],
        identifier=21,
    )
    tail = member([P(2.00003, 0.00002), P(1, -1)], identifier=22)
    line = _stitch.stitch([CHAIN[0], near_loop, tail])
    assert line.coords[-1] == P(1, -1)


def test_dense_ring_vertices_cluster_to_one_touch():
    # Two ring vertices ~4 m apart near the exit are one touch point.
    dense = [
        P(10, 0),
        P(11, 1),
        P(12, 0),
        P(12.00002, -0.00002),
        P(11, -1),
        P(10, 0),
    ]
    ring = member(dense, tags={"junction": "roundabout"}, identifier=7)
    line = _stitch.stitch([APPROACH, ring, EXIT])
    assert P(11, 1) in list(line.coords)


def test_degenerate_closed_way_is_malformed():
    spike_and_back = member([P(0, 0), P(1, 0), P(0, 0)], identifier=44)
    assert refusal([spike_and_back]) == "malformed-member"


def test_endpoint_touching_two_ring_arcs_refuses():
    # A narrow hairpin ring: the approach endpoint sits within
    # tolerance of two SEPARATED arcs — no arbitrary arc choice.
    hairpin = member(
        [
            P(10, 0),
            P(11, 0.00004),
            P(12, 0.00004),
            P(12, 1),
            P(11, 0.00008),  # returns close beside the outbound arc
            P(10.5, 0.00006),
            P(10, 0),
        ],
        tags={"junction": "roundabout"},
        identifier=7,
    )
    close_approach = member([P(8, 0.00005), P(11, 0.00006)], identifier=5)
    far_exit = member([P(12, 1), P(14, 1)], identifier=6)
    assert refusal([close_approach, hairpin, far_exit]) == "ring-touch"


def test_lone_ring_refuses():
    assert refusal([ring_member({"junction": "roundabout"})]) == "ring-touch"


def route_ways(relation):
    return [m for m in relation.members if m.kind == "way" and m.role == ""]


def test_real_tram_stitches():
    relations = _relations.route_relations(str(TRANSIT))
    tram = next(r for r in relations if r.id == 52918)
    line = _stitch.stitch(route_ways(tram))
    # Tram 1 Eira–Käpylä: a city-scale, densely-vertexed line. The
    # crude degrees→km factor overstates east–west spans at 60°N
    # (longitude degrees are half-length), hence the generous band.
    kilometres = line.length * 111.320
    assert 5 < kilometres < 20
    assert len(line.coords) > 100


def test_real_trams_mostly_stitch():
    relations = _relations.route_relations(str(TRANSIT))
    trams = [r for r in relations if r.route == "tram"]
    outcomes = {}
    for relation in trams:
        try:
            _stitch.stitch(route_ways(relation))
            outcomes[relation.id] = "ok"
        except _stitch.StitchRefusal as refused:
            outcomes[relation.id] = refused.reason
    stitched = sum(1 for value in outcomes.values() if value == "ok")
    # Helsinki tram relations are well maintained: most stitch, and
    # every failure is an explicit reason, never an exception leak.
    assert stitched >= len(trams) // 2, outcomes


def test_gap_fixture_refuses_with_gap():
    relations = _relations.route_relations(str(GAP_FIXTURE))
    gapped = next(r for r in relations if r.id == 52918)
    with pytest.raises(_stitch.StitchRefusal) as caught:
        _stitch.stitch(route_ways(gapped))
    assert caught.value.reason == "gap"
