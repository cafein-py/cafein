"""The tier-3 matcher: eligibility, the operator filter, selection."""

import math

import numpy as np
import pytest

from cafein import _matching, _relations


def relation(
    identifier=1,
    route="tram",
    ref="1",
    name=None,
    operator=None,
    network=None,
):
    return _relations.RouteRelation(
        id=identifier,
        route=route,
        ref=ref,
        name=name,
        operator=operator,
        network=network,
        members=(),
    )


STOP_XY = np.asarray([[0.0, 0.0], [1000.0, 0.0], [2000.0, 0.0], [3000.0, 0.0]])
STOP_IDS = ("s0", "s1", "s2", "s3")


def pattern(short_name="1", long_name=None, agency=(), stop_xy=STOP_XY):
    return _matching.Pattern(
        stop_ids=STOP_IDS[: len(stop_xy)],
        stop_xy=np.asarray(stop_xy, dtype=float),
        short_name=short_name,
        long_name=long_name,
        agency=agency,
    )


def beside_stops(offset=10.0):
    """Canonical positions a few meters from each pattern stop."""
    return STOP_XY + [0.0, offset]


def test_mode_table():
    assert _matching.mode_of(0) == "tram"
    assert _matching.mode_of(900) == "tram"
    assert _matching.mode_of(701) == "bus"
    assert _matching.mode_of(11) == "trolleybus"
    assert _matching.mode_of(109) == "train"
    assert _matching.mode_of(1) == "subway"
    assert _matching.mode_of(4) == "ferry"
    assert _matching.mode_of(7) is None
    assert "light_rail" in _matching.MODE_ROUTES["tram"]
    # The hard mode filter: an ordinary bus never borrows trolleybus
    # relations (nor the reverse).
    assert _matching.MODE_ROUTES["bus"] == ("bus",)
    assert _matching.MODE_ROUTES["trolleybus"] == ("trolleybus",)


def test_perfect_candidate_selected():
    winner, diagnostics = _matching.select(pattern(), [(relation(), beside_stops())])
    assert winner is not None and winner.relation.id == 1
    assert winner.reversed is False
    assert diagnostics[0]["stage"] == "scored"
    assert diagnostics[0]["score"] == 0.0
    assert diagnostics[0]["outcome"] == "selected"
    assert diagnostics[0]["corridor"] == 1.0


def test_ref_agreement_is_exact_after_folding():
    winner, _ = _matching.select(pattern(), [(relation(ref=" 1 "), beside_stops())])
    assert winner is not None
    winner, diagnostics = _matching.select(
        pattern(), [(relation(ref="1A"), beside_stops())]
    )
    assert winner is None
    assert diagnostics[0]["stage"] == "ref"


def test_ref_punctuation_stays_significant():
    # Refs fold case and whitespace only: 1-A and 1A stay distinct.
    winner, diagnostics = _matching.select(
        pattern(short_name="1A"), [(relation(ref="1-A"), beside_stops())]
    )
    assert winner is None
    assert diagnostics[0]["stage"] == "ref"
    winner, _ = _matching.select(
        pattern(short_name="1-a"), [(relation(ref="1-A "), beside_stops())]
    )
    assert winner is not None


def test_missing_ref_only_pairs_with_missing_short_name():
    # A ref-less relation never matches a ref-carrying route, and vice
    # versa; two ref-less sides go through name containment.
    assert (
        _matching.select(pattern(), [(relation(ref=None), beside_stops())])[0] is None
    )
    assert (
        _matching.select(
            pattern(short_name=None, long_name="Eira - Käpylä"),
            [(relation(ref="1"), beside_stops())],
        )[0]
        is None
    )
    winner, _ = _matching.select(
        pattern(short_name=None, long_name="Eira - Käpylä"),
        [(relation(ref=None, name="Tram: Eira–Käpylä"), beside_stops())],
    )
    assert winner is not None
    assert (
        _matching.select(
            pattern(short_name=None, long_name="Eira - Käpylä"),
            [(relation(ref=None, name="Munkkiniemi loop"), beside_stops())],
        )[0]
        is None
    )


def test_corridor_excludes_distant_relations():
    # Half the boarding positions 2 km off the stops: containment 50%.
    far = np.vstack([beside_stops()[:2], STOP_XY[2:] + [0.0, 2000.0]])
    winner, diagnostics = _matching.select(pattern(), [(relation(), far)])
    assert winner is None
    assert diagnostics[0]["stage"] == "corridor"


def test_no_boarding_positions_is_ineligible():
    winner, diagnostics = _matching.select(pattern(), [(relation(), None)])
    assert winner is None
    assert diagnostics[0]["stage"] == "no-boarding"


def test_operator_filter_narrows_to_matches():
    # The mismatching candidate scores best but is disqualified; the
    # tagless candidate loses to the agency match.
    query = pattern(agency=("Helsingin seudun liikenne", "HSL"))
    entries = [
        (relation(identifier=1, network="HSL"), beside_stops(40.0)),
        (relation(identifier=2, network="Other City"), beside_stops(1.0)),
        (relation(identifier=3), beside_stops(1.0)),
    ]
    winner, diagnostics = _matching.select(query, entries)
    assert winner is not None and winner.relation.id == 1
    stages = {d["relation"]: d["stage"] for d in diagnostics}
    assert stages[2] == "operator" and stages[3] == "operator"
    classes = {d["relation"]: d["operator"] for d in diagnostics}
    assert classes == {1: "match", 2: "mismatch", 3: "absent"}


def test_operator_filter_keeps_absent_group_without_matches():
    query = pattern(agency=("HSL",))
    entries = [
        (relation(identifier=1, network="Other City"), beside_stops(1.0)),
        (relation(identifier=2), beside_stops(40.0)),
    ]
    winner, _ = _matching.select(query, entries)
    assert winner is not None and winner.relation.id == 2


def test_operator_filter_skipped_without_agency():
    winner, _ = _matching.select(
        pattern(agency=()),
        [(relation(network="Whoever"), beside_stops())],
    )
    assert winner is not None


def test_agency_id_matches_network_tag():
    # Feeds commonly brand the id (HSL) while the name is the long
    # form; either identity string may pair with either tag.
    query = pattern(agency=("Helsingin seudun liikenne", "HSL"))
    winner, _ = _matching.select(
        query, [(relation(operator="HKL", network="HSL"), beside_stops())]
    )
    assert winner is not None


def test_collapse_prefers_stop_coordinate():
    kinds = [False, True, False]
    xy = np.asarray([[0.0, 0.0], [30.0, 0.0], [500.0, 0.0]])
    canonical = _matching.collapse_positions(kinds, xy)
    assert canonical.shape == (2, 2)
    assert canonical[0].tolist() == [30.0, 0.0]


def test_collapse_keeps_separated_members():
    kinds = [False, False]
    xy = np.asarray([[0.0, 0.0], [60.0, 0.0]])
    assert _matching.collapse_positions(kinds, xy).shape == (2, 2)


def test_gap_positions_never_match():
    # One position beyond snap tolerance becomes a gap symbol: it can
    # only cost, never pair — and two gaps stay distinct.
    off = beside_stops()
    off[2] = [2000.0, 400.0]
    winner, diagnostics = _matching.select(pattern(), [(relation(), off)])
    assert winner is not None
    assert diagnostics[0]["score"] == pytest.approx(0.25)


def test_acceptance_threshold_drops_weak_best():
    off = beside_stops()
    off[2] = [2000.0, 400.0]
    off[3] = [3000.0, 400.0]
    winner, diagnostics = _matching.select(pattern(), [(relation(), off)])
    assert winner is None
    assert diagnostics[0]["score"] == pytest.approx(0.5)


def test_near_tie_drops_both():
    entries = [
        (relation(identifier=1), beside_stops()),
        (relation(identifier=2), beside_stops(20.0)),
    ]
    winner, _ = _matching.select(pattern(), entries)
    assert winner is None


def test_clear_margin_wins():
    off = beside_stops()
    off[2] = [2000.0, 400.0]  # one gap: score 0.25
    entries = [
        (relation(identifier=1), beside_stops()),
        (relation(identifier=2), off),
    ]
    winner, _ = _matching.select(pattern(), entries)
    assert winner is not None and winner.relation.id == 1


def test_reversed_member_order_still_matches():
    winner, diagnostics = _matching.select(
        pattern(), [(relation(), beside_stops()[::-1])]
    )
    assert winner is not None
    assert winner.reversed is True
    assert diagnostics[0]["reversed"] is True
    assert diagnostics[0]["backward"] == 0.0
    assert diagnostics[0]["forward"] > 0.0


def test_edit_distance_basics():
    assert _matching.edit_distance(("a", "b", "c"), ("a", "b", "c")) == 0.0
    assert _matching.edit_distance(("a", "b", "c"), ("a", "x", "c")) == pytest.approx(
        1 / 3
    )
    assert _matching.edit_distance(("a", "b"), ("a", "b", "c", "d")) == pytest.approx(
        0.5
    )
    assert _matching.edit_distance((), ()) == 0.0


def test_boarding_positions_take_roles_and_centroids():
    import shapely

    members = (
        _relations.RelationMember(
            kind="node", id=1, role="platform", geometry=shapely.Point(1, 2), tags={}
        ),
        _relations.RelationMember(
            kind="node",
            id=2,
            role="stop_exit_only",
            geometry=shapely.Point(3, 4),
            tags={},
        ),
        _relations.RelationMember(
            kind="way",
            id=3,
            role="platform",
            geometry=shapely.LineString([(0, 0), (2, 0)]),
            tags={},
        ),
        _relations.RelationMember(kind="way", id=4, role="", geometry=None, tags={}),
        _relations.RelationMember(
            kind="node", id=5, role="stop", geometry=None, tags={}
        ),
    )
    positions = _matching.boarding_positions(
        _relations.RouteRelation(
            id=9,
            route="tram",
            ref="1",
            name=None,
            operator=None,
            network=None,
            members=members,
        )
    )
    assert positions[:3] == [(False, 1.0, 2.0), (True, 3.0, 4.0), (False, 1.0, 0.0)]
    # The unresolved stop member keeps its slot as a coordinate-less
    # position instead of vanishing.
    assert len(positions) == 4
    is_stop, x, y = positions[3]
    assert is_stop is True and math.isnan(x) and math.isnan(y)


def test_unresolved_boarding_members_break_containment():
    # Half the boarding slots are unresolved: the relation must not
    # look perfectly contained on its surviving members alone.
    canonical = np.vstack([beside_stops()[:2], np.full((2, 2), np.nan)])
    winner, diagnostics = _matching.select(pattern(), [(relation(), canonical)])
    assert winner is None
    assert diagnostics[0]["stage"] == "corridor"
    assert diagnostics[0]["corridor"] == pytest.approx(0.5)
