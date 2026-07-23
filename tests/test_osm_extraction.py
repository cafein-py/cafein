"""Union OSM extraction, tag normalisation, permissions, and pruning."""

import numpy as np
import pytest

from cafein import _osm

W, B, S = _osm.WALK, _osm.BICYCLE, _osm.E_SCOOTER


def _perm(**tags):
    forward, reverse, flags, unknown_access, unknown_highway = _osm._row_permissions(
        tags
    )
    return forward, reverse, flags, unknown_access, unknown_highway


# --- The permission compiler (a synthetic tag matrix, no PBF needed) ---------


@pytest.mark.parametrize(
    "tags, forward, reverse, flags",
    [
        # Highway defaults, before any explicit access tags.
        (dict(highway="footway"), W, W, 0),
        (dict(highway="pedestrian"), W, W, 0),
        (dict(highway="steps"), W, W, 0),
        (dict(highway="cycleway"), B | S, B | S, 0),
        (dict(highway="residential"), W | B | S, W | B | S, 0),
        (dict(highway="track"), W | B | S, W | B | S, 0),
        (dict(highway="platform"), W, W, 0),
        # An unrecognised highway value denies both modes by default (only an
        # explicit mode tag opens it) — see the unknown-highway test below.
        (dict(highway="something_new"), 0, 0, 0),
        (dict(highway="something_new", bicycle="yes"), B | S, B | S, 0),
        # A general access DENY overrides the highway default for every mode;
        # a general access ALLOW does not grant a mode the type denies.
        (dict(highway="residential", access="no"), 0, 0, 0),
        (dict(highway="residential", access="private"), 0, 0, 0),
        (dict(highway="footway", access="destination"), W, W, 0),
        (dict(highway="footway", access="yes"), W, W, 0),
        # Mode-specific tags override the general access.
        (dict(highway="residential", foot="no"), B | S, B | S, 0),
        (dict(highway="footway", bicycle="yes"), W | B | S, W | B | S, 0),
        (dict(highway="cycleway", bicycle="no"), 0, 0, 0),
        (dict(highway="cycleway", foot="yes"), W | B | S, W | B | S, 0),
        # bicycle=dismount permits the bike but sets the dismount flag.
        (
            dict(highway="footway", bicycle="dismount"),
            W | B | S,
            W | B | S,
            _osm.FLAG_DISMOUNT,
        ),
        # use_sidepath denies the bicycle on this way.
        (dict(highway="primary", bicycle="use_sidepath"), W, W, 0),
        # vehicle sits between access and bicycle in the hierarchy: vehicle=no
        # denies the bike, vehicle=yes re-grants a bike-permitting way that a
        # general access=no closed, but never grants a type-denied mode.
        (dict(highway="service", vehicle="no"), W, W, 0),
        (dict(highway="service", vehicle="no", bicycle="yes"), W | B | S, W | B | S, 0),
        # access=no denies pedestrians (foot has no vehicle re-grant), while
        # vehicle=yes re-opens the bike on this bike-permitting way.
        (dict(highway="service", access="no", vehicle="yes"), B | S, B | S, 0),
        (dict(highway="footway", vehicle="yes"), W, W, 0),
        # Directionality: oneway blocks the reverse bicycle; foot is unaffected.
        (dict(highway="residential", oneway="yes"), W | B | S, W, 0),
        (dict(highway="residential", oneway="-1"), W, W | B | S, 0),
        (dict(highway="residential", junction="roundabout"), W | B | S, W, 0),
        # An explicit oneway=no overrides a roundabout's implicit direction.
        (
            dict(highway="residential", junction="roundabout", oneway="no"),
            W | B | S,
            W | B | S,
            0,
        ),
        # Cycling exceptions re-open the direction the base oneway blocked.
        (
            {"highway": "residential", "oneway": "yes", "oneway:bicycle": "no"},
            W | B | S,
            W | B | S,
            0,
        ),
        (
            dict(highway="residential", oneway="yes", cycleway="opposite_lane"),
            W | B | S,
            W | B | S,
            0,
        ),
        (
            {
                "highway": "residential",
                "oneway": "yes",
                "cycleway:left": "opposite_track",
            },
            W | B | S,
            W | B | S,
            0,
        ),
        # Contraflow on a reverse (oneway=-1) way re-opens the forward bike.
        (
            dict(highway="residential", oneway="-1", cycleway="opposite_lane"),
            W | B | S,
            W | B | S,
            0,
        ),
        # `oneway:bicycle` honours the boolean aliases, not just "no".
        (
            {"highway": "residential", "oneway": "yes", "oneway:bicycle": "false"},
            W | B | S,
            W | B | S,
            0,
        ),
        (
            {"highway": "residential", "oneway": "yes", "oneway:bicycle": "0"},
            W | B | S,
            W | B | S,
            0,
        ),
        # A modern on-edge side-cycleway running against the oneway re-opens
        # the reverse; the direction qualifier needs the companion lane.
        (
            {
                "highway": "residential",
                "oneway": "yes",
                "cycleway:left": "lane",
                "cycleway:left:oneway": "-1",
            },
            W | B | S,
            W | B | S,
            0,
        ),
        (
            {
                "highway": "residential",
                "oneway": "yes",
                "cycleway:right": "track",
                "cycleway:right:oneway": "no",
            },
            W | B | S,
            W | B | S,
            0,
        ),
        # A `separate` (off-edge) or absent lane does not carry contraflow, so
        # the reverse stays blocked despite the direction qualifier.
        (
            {
                "highway": "residential",
                "oneway": "yes",
                "cycleway:right": "separate",
                "cycleway:right:oneway": "no",
            },
            W | B | S,
            W,
            0,
        ),
        (
            {
                "highway": "residential",
                "oneway": "yes",
                "cycleway:left:oneway": "-1",
            },
            W | B | S,
            W,
            0,
        ),
        # Restrictive access values are not general-public access.
        (dict(highway="service", access="delivery"), 0, 0, 0),
        (dict(highway="track", access="agricultural"), 0, 0, 0),
        (dict(highway="track", access="forestry", foot="yes"), W, W, 0),
        # A dismounted cyclist is pedestrian-like and ignores the oneway.
        (
            dict(highway="residential", oneway="yes", bicycle="dismount"),
            W | B | S,
            W | B | S,
            _osm.FLAG_DISMOUNT,
        ),
        # junction=circular is not implicitly one-way (unlike roundabout).
        (dict(highway="residential", junction="circular"), W | B | S, W | B | S, 0),
        (
            dict(highway="residential", junction="circular", oneway="yes"),
            W | B | S,
            W,
            0,
        ),
        # A modern on-edge side-cycleway direction is relative to the geometry:
        # on a reverse (oneway=-1) base, a forward-running lane re-opens forward.
        (
            {
                "highway": "residential",
                "oneway": "-1",
                "cycleway:left": "lane",
                "cycleway:left:oneway": "yes",
            },
            W | B | S,
            W | B | S,
            0,
        ),
        # …while a reverse-running lane on a reverse base just follows it — the
        # blocked forward stays blocked.
        (
            {
                "highway": "residential",
                "oneway": "-1",
                "cycleway:left": "lane",
                "cycleway:left:oneway": "-1",
            },
            W,
            W | B | S,
            0,
        ),
    ],
)
def test_edge_permission_matrix(tags, forward, reverse, flags):
    got_forward, got_reverse, got_flags, _, _ = _perm(**tags)
    assert got_forward == forward
    assert got_reverse == reverse
    assert got_flags == flags


def test_unknown_access_is_conservative_and_counted():
    # An unrecognised access value neither newly permits nor denies — the
    # highway default stands — and it is reported for diagnostics.
    forward, reverse, _, unknown_access, _ = _perm(highway="residential", access="wat")
    assert forward == reverse == W | B | S
    assert unknown_access
    forward, _, _, unknown_access, _ = _perm(highway="footway", access="wat")
    assert forward == W
    assert unknown_access


def test_unknown_highway_denies_and_is_counted():
    # An unmodelled highway value routes over nothing by default and is
    # reported, so a typo or a new type never silently opens a way.
    forward, reverse, _, _, unknown_highway = _perm(highway="rest_area")
    assert forward == reverse == 0
    assert unknown_highway
    # An explicit mode tag still opens it, and that is not an unknown-access.
    forward, _, _, unknown_access, unknown_highway = _perm(
        highway="rest_area", foot="yes"
    )
    assert forward == W
    assert unknown_highway and not unknown_access


def test_escooter_mirrors_bicycle():
    # The default policy is "bicycle_like": the e-scooter bit tracks the
    # bicycle bit in both directions.
    for tags in (
        dict(highway="cycleway"),
        dict(highway="residential", oneway="yes"),
        dict(highway="footway"),
    ):
        forward, reverse, _, _, _ = _perm(**tags)
        assert bool(forward & S) == bool(forward & B)
        assert bool(reverse & S) == bool(reverse & B)


# --- Class-code normalisation (a synthetic edges frame) ----------------------


def test_normalise_codes_maps_recognised_unknown_and_missing():
    import pandas as pd

    edges = pd.DataFrame(
        {
            "highway": ["steps", "residential", "cycleway"],
            "surface": ["asphalt", "wat", None],  # recognised, unknown, missing
            "smoothness": ["good", None, "wat"],
            "bridge": ["yes", "0", None],  # 0 is false
        }
    )
    highway, surface, smoothness, flags = _osm.normalise_codes(edges)
    assert list(highway) == [
        _osm.HIGHWAY_CODES["steps"],
        _osm.HIGHWAY_CODES["residential"],
        _osm.HIGHWAY_CODES["cycleway"],
    ]
    # asphalt → its code; unknown and missing → 0.
    assert list(surface) == [_osm.SURFACE_CODES["asphalt"], 0, 0]
    assert list(smoothness) == [_osm.SMOOTHNESS_CODES["good"], 0, 0]
    # steps flag on the stepped way; bridge only where truthy (not "0"/missing).
    assert flags[0] & _osm.FLAG_STEPS
    assert bool(flags[0] & _osm.FLAG_BRIDGE)
    assert not (flags[1] & _osm.FLAG_BRIDGE)  # bridge="0" is false
    assert not (flags[2] & _osm.FLAG_BRIDGE)  # bridge missing


# --- Component pruning (synthetic graphs) ------------------------------------


def test_prune_clears_small_components_per_mode():
    # A 50-vertex chain (one component) plus a 2-vertex stub. The stub is below
    # MIN_ISLAND_VERTICES, so both modes lose it; the chain is untouched.
    edges = [(i, i + 1) for i in range(49)] + [(50, 51)]
    u = np.array([a for a, _ in edges])
    v = np.array([b for _, b in edges])
    wb = W | B
    forward = np.full(len(edges), wb, dtype=np.uint8)
    reverse = forward.copy()
    pruned_f, pruned_r = _osm.prune_components_per_profile(u, v, 52, forward, reverse)
    assert (pruned_f[:49] == wb).all()
    assert pruned_f[49] == 0 and pruned_r[49] == 0


def test_prune_judges_connectivity_per_mode_not_on_the_union():
    # Two small bike clusters (20 + 20 vertices, each below MIN_ISLAND_VERTICES)
    # joined only by a single walk-only bridge edge. On the union the whole
    # thing is one 41-vertex component (above threshold), but for bicycles the
    # bridge does not connect the clusters, so each bike cluster is a sub-40
    # component and must be pruned — while walking, connected across the bridge,
    # is kept. A union-based implementation would wrongly keep the bike arcs.
    left = [(i, i + 1) for i in range(19)]  # vertices 0..19 (20 vertices)
    right = [(i, i + 1) for i in range(20, 39)]  # vertices 20..39 (20 vertices)
    bridge = [(19, 20)]  # walk-only connector
    edges = left + right + bridge
    u = np.array([a for a, _ in edges])
    v = np.array([b for _, b in edges])
    wb = W | B
    forward = np.array([wb] * (len(left) + len(right)) + [W], dtype=np.uint8)
    reverse = forward.copy()
    pruned_f, pruned_r = _osm.prune_components_per_profile(u, v, 40, forward, reverse)
    # Bicycle is cleared everywhere (both clusters are sub-threshold for bikes);
    # walking survives on the now-connected 40-vertex component.
    assert (pruned_f & B == 0).all() and (pruned_r & B == 0).all()
    assert (pruned_f[: len(left) + len(right)] & W != 0).all()
    assert pruned_f[-1] & W  # the bridge keeps walking


# --- The union extraction against the pinned Helsinki extract ----------------

_CENTRAL_BBOX = [24.93, 60.16, 24.96, 60.18]


@pytest.fixture(scope="module")
def union_extract(kantakaupunki_pbf):
    """The union network of a central Helsinki bbox, extracted once."""
    return _osm.union_network(str(kantakaupunki_pbf), bounding_box=_CENTRAL_BBOX)


def test_union_extraction_retains_consumed_tags(union_extract):
    # Every tag the compilers consume must survive the extraction; the
    # directional ones in particular are easy to lose to a pyrosm config change.
    _, edges = union_extract
    for tag in ("highway", "oneway", "oneway:bicycle", "junction", "segregated"):
        assert tag in edges.columns, tag
    # oneway ways exist in central Helsinki, so the tag carries real values.
    assert (edges["oneway"] == "yes").any()


def test_union_extraction_shape_and_codes(union_extract):
    nodes, edges = union_extract
    assert len(nodes) > 0 and len(edges) > 0
    highway, surface, smoothness, flags = _osm.normalise_codes(edges)
    assert len(highway) == len(edges)
    # Motor-only ways are filtered out, so no motorway codes appear.
    assert _osm.HIGHWAY_CODES["motorway"] not in set(highway.tolist())
    # Central Helsinki has cycleways, footways, and stepped ways.
    codes = set(highway.tolist())
    assert _osm.HIGHWAY_CODES["cycleway"] in codes
    assert _osm.HIGHWAY_CODES["footway"] in codes
    assert (flags & _osm.FLAG_STEPS != 0).any()


def test_union_permissions_and_diagnostics(union_extract):
    nodes, edges = union_extract
    forward, reverse, flags, diagnostics = _osm.edge_permissions(edges)
    # Most ways are walkable; a smaller share is cyclable; some are one-way for
    # bicycles (forward and reverse differ).
    walkable = (forward & _osm.WALK != 0).mean()
    bikeable = (forward & _osm.BICYCLE != 0).mean()
    assert walkable > 0.8
    assert 0.1 < bikeable < walkable
    assert ((forward & _osm.BICYCLE) != (reverse & _osm.BICYCLE)).any()
    # Walking is undirected: its bit is identical in both directions.
    assert np.array_equal(forward & _osm.WALK, reverse & _osm.WALK)
    # The pinned extract is well-tagged, so few unknown ways of either kind.
    assert diagnostics["unknown_access"] < 0.01 * len(edges)
    assert diagnostics["unknown_highway"] < 0.02 * len(edges)


def test_union_pruning_clears_disconnected_bicycle_arcs(union_extract):
    from cafein.streets import _vertex_endpoints

    nodes, edges = union_extract
    forward, reverse, _, _ = _osm.edge_permissions(edges)
    u, v = _vertex_endpoints(nodes, edges)
    pruned_f, pruned_r = _osm.prune_components_per_profile(
        u, v, len(nodes), forward, reverse
    )
    # Pruning only clears bits, never adds them (both directions are subsets of
    # their inputs).
    assert (pruned_f & ~forward == 0).all()
    assert (pruned_r & ~reverse == 0).all()
    # Most walking survives, while disconnected bicycle arcs are trimmed: the
    # pinned extract has bike stubs, so at least one forward bicycle arc is
    # cleared, and far more bicycle than walking arcs are removed.
    walk_before = int((forward & _osm.WALK != 0).sum())
    walk_after = int((pruned_f & _osm.WALK != 0).sum())
    bike_before = int((forward & _osm.BICYCLE != 0).sum())
    bike_after = int((pruned_f & _osm.BICYCLE != 0).sum())
    assert walk_after > 0.9 * walk_before
    assert bike_after < bike_before
