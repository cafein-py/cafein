"""Union OSM extraction, tag normalisation, permissions, and pruning."""

import numpy as np
import pytest

from cafein import _osm

W, B, S, C = _osm.WALK, _osm.BICYCLE, _osm.E_SCOOTER, _osm.CAR
WC = _osm.WHEELCHAIR
_LEGACY = W | B | S | C


def test_mode_and_flag_bits_match_the_rust_abi():
    # These integer values cross to the Rust profile compiler (streets/profile.rs)
    # as the raw u8/u16 attribute arrays, so a change here is a breaking ABI
    # change and must be mirrored on the Rust side (and vice versa).
    assert (
        _osm.WALK,
        _osm.BICYCLE,
        _osm.E_SCOOTER,
        _osm.CAR,
        _osm.WHEELCHAIR,
    ) == (1, 2, 4, 8, 16)
    assert (
        _osm.FLAG_DISMOUNT,
        _osm.FLAG_BRIDGE,
        _osm.FLAG_TUNNEL,
        _osm.FLAG_INDOOR,
        _osm.FLAG_STEPS,
        _osm.FLAG_SEGREGATED,
        _osm.FLAG_LIT,
        _osm.FLAG_ROUNDABOUT,
    ) == (1, 2, 4, 8, 16, 32, 64, 128)
    assert (
        _osm.JUNCTION_TOPOLOGICAL,
        _osm.JUNCTION_PRIORITY,
        _osm.JUNCTION_SIGNALS,
        _osm.JUNCTION_RAMP,
    ) == (1, 2, 3, 4)
    assert (
        len(_osm.HIGHWAY_CODES),
        len(_osm.SURFACE_CODES),
        len(_osm.SMOOTHNESS_CODES),
    ) == (27, 17, 9)


def test_normalise_codes_sets_the_roundabout_flag():
    import pandas as pd

    edges = pd.DataFrame(
        {
            "highway": ["residential", "residential", "residential"],
            "junction": ["roundabout", "circular", None],
        }
    )
    _, _, _, flags = _osm.normalise_codes(edges)
    assert list(flags & _osm.FLAG_ROUNDABOUT) == [_osm.FLAG_ROUNDABOUT, 0, 0]


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
        (dict(highway="residential"), W | B | S | C, W | B | S | C, 0),
        (dict(highway="track"), W | B | S | C, W | B | S | C, 0),
        (dict(highway="platform"), W, W, 0),
        (dict(highway="trunk"), C, C, 0),
        # A motorway carriageway is implicitly one-way; an explicit false
        # oneway opens it, and links follow their explicit tags only.
        (dict(highway="motorway"), C, 0, 0),
        (dict(highway="motorway", oneway="no"), C, C, 0),
        (dict(highway="motorway_link"), C, C, 0),
        (dict(highway="motorway_link", oneway="yes"), C, 0, 0),
        # The implied one-way is shared state: an explicitly permitted
        # bicycle on a motorway is directional too.
        (dict(highway="motorway", bicycle="yes"), B | S | C, 0, 0),
        # An unrecognised highway value denies every mode by default (only an
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
        (dict(highway="residential", foot="no"), B | S | C, B | S | C, 0),
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
        # use_sidepath denies the bicycle on this way; the car is untouched.
        (dict(highway="primary", bicycle="use_sidepath"), W | C, W | C, 0),
        # vehicle sits between access and bicycle in the hierarchy: vehicle=no
        # denies bike and car, vehicle=yes re-grants what the type permits,
        # but never grants a type-denied mode.
        (dict(highway="service", vehicle="no"), W, W, 0),
        (
            dict(highway="service", vehicle="no", bicycle="yes"),
            W | B | S,
            W | B | S,
            0,
        ),
        # access=no denies pedestrians (foot has no vehicle re-grant), while
        # vehicle=yes re-opens bike and car on this permitting way.
        (dict(highway="service", access="no", vehicle="yes"), B | S | C, B | S | C, 0),
        (dict(highway="footway", vehicle="yes"), W, W, 0),
        # The car chain continues below vehicle: motor_vehicle / motorcar.
        (dict(highway="residential", motor_vehicle="no"), W | B | S, W | B | S, 0),
        (dict(highway="residential", motorcar="no"), W | B | S, W | B | S, 0),
        (
            dict(highway="residential", motor_vehicle="no", motorcar="yes"),
            W | B | S | C,
            W | B | S | C,
            0,
        ),
        (dict(highway="residential", vehicle="no", motorcar="yes"), W | C, W | C, 0),
        (dict(highway="residential", access="no", motorcar="yes"), C, C, 0),
        # The most-specific motorcar= overrides freely, like bicycle= does.
        (dict(highway="footway", motorcar="yes"), W | C, W | C, 0),
        (dict(highway="cycleway", motorcar="yes"), B | S | C, B | S | C, 0),
        # Directionality: oneway blocks the reverse bicycle and car; foot is
        # unaffected.
        (dict(highway="residential", oneway="yes"), W | B | S | C, W, 0),
        (dict(highway="residential", oneway="-1"), W, W | B | S | C, 0),
        (dict(highway="residential", junction="roundabout"), W | B | S | C, W, 0),
        # An explicit oneway=no overrides a roundabout's implicit direction.
        (
            dict(highway="residential", junction="roundabout", oneway="no"),
            W | B | S | C,
            W | B | S | C,
            0,
        ),
        # Cycling exceptions re-open the blocked direction for the bicycle
        # only — the car has no contraflow grants of any kind.
        (
            {"highway": "residential", "oneway": "yes", "oneway:bicycle": "no"},
            W | B | S | C,
            W | B | S,
            0,
        ),
        (
            dict(highway="residential", oneway="yes", cycleway="opposite_lane"),
            W | B | S | C,
            W | B | S,
            0,
        ),
        (
            {
                "highway": "residential",
                "oneway": "yes",
                "cycleway:left": "opposite_track",
            },
            W | B | S | C,
            W | B | S,
            0,
        ),
        # Contraflow on a reverse (oneway=-1) way re-opens the forward bike.
        (
            dict(highway="residential", oneway="-1", cycleway="opposite_lane"),
            W | B | S,
            W | B | S | C,
            0,
        ),
        # `oneway:bicycle` honours the boolean aliases, not just "no".
        (
            {"highway": "residential", "oneway": "yes", "oneway:bicycle": "false"},
            W | B | S | C,
            W | B | S,
            0,
        ),
        (
            {"highway": "residential", "oneway": "yes", "oneway:bicycle": "0"},
            W | B | S | C,
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
            W | B | S | C,
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
            W | B | S | C,
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
            W | B | S | C,
            W,
            0,
        ),
        (
            {
                "highway": "residential",
                "oneway": "yes",
                "cycleway:left:oneway": "-1",
            },
            W | B | S | C,
            W,
            0,
        ),
        # Restrictive access values are not general-public access.
        (dict(highway="service", access="delivery"), 0, 0, 0),
        (dict(highway="track", access="agricultural"), 0, 0, 0),
        (dict(highway="track", access="forestry", foot="yes"), W, W, 0),
        # A dismounted cyclist is pedestrian-like and ignores the oneway; the
        # car still follows it.
        (
            dict(highway="residential", oneway="yes", bicycle="dismount"),
            W | B | S | C,
            W | B | S,
            _osm.FLAG_DISMOUNT,
        ),
        # junction=circular is not implicitly one-way (unlike roundabout).
        (
            dict(highway="residential", junction="circular"),
            W | B | S | C,
            W | B | S | C,
            0,
        ),
        (
            dict(highway="residential", junction="circular", oneway="yes"),
            W | B | S | C,
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
            W | B | S | C,
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
            W | B | S | C,
            0,
        ),
    ],
)
def test_edge_permission_matrix(tags, forward, reverse, flags):
    # The legacy four-mode pins; the wheelchair bit has its own matrix.
    got_forward, got_reverse, got_flags, _, _ = _perm(**tags)
    assert got_forward & _LEGACY == forward
    assert got_reverse & _LEGACY == reverse
    assert got_flags == flags


@pytest.mark.parametrize(
    "tags, allowed",
    [
        # Walkable classes are wheelchair classes; stairs are the veto.
        (dict(highway="footway"), True),
        (dict(highway="pedestrian"), True),
        (dict(highway="elevator"), True),
        (dict(highway="residential"), True),
        (dict(highway="steps"), False),
        # wheelchair=yes rescues only the steps class veto…
        (dict(highway="steps", wheelchair="yes"), True),
        # …and only the literal `yes`: other values, `designated`
        # included, keep the veto.
        (dict(highway="steps", wheelchair="designated"), False),
        # …never an access-ladder denial or a non-walkable class.
        (dict(highway="steps", wheelchair="yes", access="private"), False),
        (dict(highway="steps", wheelchair="yes", foot="no"), False),
        (dict(highway="motorway", wheelchair="yes"), False),
        (dict(highway="trunk", wheelchair="yes"), False),
        # A foot tag never opens stairs for wheels.
        (dict(highway="steps", foot="designated"), False),
        # wheelchair=no denies an otherwise walkable way.
        (dict(highway="footway", wheelchair="no"), False),
        (dict(highway="residential", wheelchair="no"), False),
        # limited and unknown values keep the resolved default.
        (dict(highway="footway", wheelchair="limited"), True),
        (dict(highway="steps", wheelchair="limited"), False),
        # The walk access ladder applies beneath it all.
        (dict(highway="footway", access="private"), False),
        (dict(highway="residential", foot="no"), False),
        (dict(highway="something_new"), False),
        (dict(highway="something_new", foot="yes"), True),
    ],
)
def test_wheelchair_permission_matrix(tags, allowed):
    forward, reverse, _, _, _ = _perm(**tags)
    assert bool(forward & WC) == allowed
    assert bool(reverse & WC) == allowed


def test_unknown_access_is_conservative_and_counted():
    # An unrecognised access value neither newly permits nor denies — the
    # highway default stands — and it is reported for diagnostics.
    forward, reverse, _, unknown_access, _ = _perm(highway="residential", access="wat")
    assert forward == reverse == W | B | S | C | WC
    assert unknown_access
    forward, _, _, unknown_access, _ = _perm(highway="footway", access="wat")
    assert forward == W | WC
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
    assert forward == W | WC
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


@pytest.mark.parametrize(
    "bit",
    [
        # A walk-only bridge edge joining two small bicycle clusters.
        B,
        # A stairway joining two wheelchair clusters: walkable, never
        # wheelable.
        WC,
    ],
    ids=["bicycle", "wheelchair"],
)
def test_prune_judges_connectivity_per_mode_not_on_the_union(bit):
    # Two small clusters (20 + 20 vertices, each below MIN_ISLAND_VERTICES)
    # joined only by a single walk-only connector edge. On the union the
    # whole thing is one 40-vertex component (at the keep threshold), but for the
    # tested mode the connector does not join the clusters, so each cluster
    # is a sub-40 component and must be pruned — while walking, connected
    # across the connector, is kept. A union-based implementation would
    # wrongly keep the pruned mode's arcs.
    left = [(i, i + 1) for i in range(19)]  # vertices 0..19 (20 vertices)
    right = [(i, i + 1) for i in range(20, 39)]  # vertices 20..39 (20 vertices)
    connector = [(19, 20)]  # walk-only
    edges = left + right + connector
    u = np.array([a for a, _ in edges])
    v = np.array([b for _, b in edges])
    both = W | bit
    forward = np.array([both] * (len(left) + len(right)) + [W], dtype=np.uint8)
    reverse = forward.copy()
    pruned_f, pruned_r = _osm.prune_components_per_profile(u, v, 40, forward, reverse)
    # The tested mode is cleared everywhere (both clusters sub-threshold);
    # walking survives on the now-connected 40-vertex component.
    assert (pruned_f & bit == 0).all() and (pruned_r & bit == 0).all()
    assert (pruned_f[: len(left) + len(right)] & W != 0).all()
    assert pruned_f[-1] & W  # the connector keeps walking


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


def test_pruning_touches_only_the_named_modes(union_extract):
    from cafein.streets import _vertex_endpoints

    nodes, edges = union_extract
    forward, reverse, _, _ = _osm.edge_permissions(edges)
    u, v = _vertex_endpoints(nodes, edges)
    bike_only_f, bike_only_r = _osm.prune_components_per_profile(
        u, v, len(nodes), forward, reverse, modes=["bicycle"]
    )
    # The named mode is pruned...
    assert int((bike_only_f & _osm.BICYCLE != 0).sum()) < int(
        (forward & _osm.BICYCLE != 0).sum()
    )
    # ...while every other mode's bits are left exactly as they arrived.
    for bit in (_osm.WALK, _osm.E_SCOOTER):
        assert np.array_equal(bike_only_f & bit, forward & bit)
        assert np.array_equal(bike_only_r & bit, reverse & bit)
    # An unknown mode name refuses loudly.
    with pytest.raises(ValueError, match="unknown street mode"):
        _osm.prune_components_per_profile(
            u, v, len(nodes), forward, reverse, modes=["hovercraft"]
        )


# --- Car speeds ---------------------------------------------------------------


def test_parse_maxspeed_table():
    mph = 1.609344
    cases = [
        ("50", 50.0),
        ("50.5", 50.5),
        ("30 mph", 30 * mph),
        ("30mph", 30 * mph),
        ("none", None),
        ("signals", None),
        ("walk", None),
        ("DE:zone30", None),
        ("-5", None),
        ("0", None),
        ("", None),
        (None, None),
    ]
    for raw, expected in cases:
        got = _osm.parse_maxspeed(raw)
        if expected is None:
            assert got is None, raw
        else:
            assert got == pytest.approx(expected), raw


def test_car_speeds_resolve_tags_then_country_defaults():
    import pandas as pd

    edges = pd.DataFrame(
        {
            "highway": ["residential", "motorway", "primary", "wat", "living_street"],
            "maxspeed": [None, "100", "30 mph", None, None],
            "maxspeed:backward": ["40", None, None, None, None],
        }
    )
    # A tagged maxspeed wins; a directional tag overrides its direction
    # only; untagged ways take the country row's class default, unknown
    # classes the other_* fallback.
    forward, reverse = _osm.car_speeds(edges, country="FI")
    assert list(forward) == pytest.approx([50.0, 100.0, 30 * 1.609344, 50.0, 20.0])
    assert list(reverse) == pytest.approx([40.0, 100.0, 30 * 1.609344, 50.0, 20.0])
    # The rural values apply outside urban polygons; a subdivision row
    # selects state law (US-CA residential 25 mph = 40 km/h), and an
    # unknown subdivision falls back to the country's generic row.
    rural = [False] * len(edges)
    forward, _ = _osm.car_speeds(edges, country="US-CA", urban=rural)
    assert forward[0] == pytest.approx(40.0)
    forward, _ = _osm.car_speeds(edges, country="US-TX", urban=rural)
    generic_us, _ = _osm.car_speeds(edges, country="US", urban=rural)
    assert forward[0] == pytest.approx(generic_us[0])
    with pytest.raises(ValueError, match="ISO 3166"):
        _osm.car_speeds(edges, country="bogus!")
    # An urban-areas polygon frame resolves membership by spatial join.
    import geopandas as gpd
    from shapely.geometry import LineString, Polygon

    geo = gpd.GeoDataFrame(
        {"highway": ["residential", "residential", "residential"]},
        geometry=[
            LineString([(0, 0), (1, 0)]),
            LineString([(5, 5), (6, 5)]),
            # Crosses the polygon boundary: any intersection makes the
            # whole edge urban.
            LineString([(1, 1), (5, 1)]),
        ],
        crs="EPSG:4326",
    )
    urban_areas = gpd.GeoDataFrame(
        geometry=[Polygon([(-1, -1), (2, -1), (2, 2), (-1, 2)])], crs="EPSG:4326"
    )
    forward, _ = _osm.car_speeds(geo, country="FI", urban=urban_areas)
    assert list(forward) == pytest.approx([50.0, 80.0, 50.0])
    # The join reprojects mismatched polygon CRSs onto the edges' own.
    projected = urban_areas.to_crs("EPSG:3067")
    forward, _ = _osm.car_speeds(geo, country="FI", urban=projected)
    assert list(forward) == pytest.approx([50.0, 80.0, 50.0])
    with pytest.raises(ValueError, match="CRS"):
        _osm.car_speeds(
            geo, country="FI", urban=urban_areas.set_crs(None, allow_override=True)
        )
    # Untagged tracks take the low product default, area-invariant —
    # never the ordinary-road fallback — and `speed_limits=` overrides
    # any class of the resolved row.
    track = pd.DataFrame({"highway": ["track", "track"]})
    forward, _ = _osm.car_speeds(track, country="FI", urban=[True, False])
    assert list(forward) == pytest.approx([20.0, 20.0])
    forward, _ = _osm.car_speeds(track, country="FI", speed_limits={"track": 30})
    assert list(forward) == pytest.approx([30.0, 30.0])
    forward, _ = _osm.car_speeds(
        edges, country="FI", speed_limits={"residential_inside": 35}
    )
    assert forward[0] == pytest.approx(35.0)
    # Overrides are validated: unknown classes and non-positive or
    # non-finite speeds are rejected loudly.
    with pytest.raises(ValueError, match="unknown speed_limits"):
        _osm.car_speeds(edges, country="FI", speed_limits={"residental": 30})
    with pytest.raises(ValueError, match="positive km/h"):
        _osm.car_speeds(edges, country="FI", speed_limits={"track": 0})
    with pytest.raises(ValueError, match="positive km/h"):
        _osm.car_speeds(edges, country="FI", speed_limits={"track": float("nan")})
    with pytest.raises(ValueError, match="per edge"):
        _osm.car_speeds(edges, country="FI", urban=[True])
    with pytest.warns(UserWarning, match="no country="):
        _osm.car_speeds(edges)
    with pytest.warns(UserWarning, match="Generic row"):
        _osm.car_speeds(edges, country="XX")


def test_speed_limit_table_is_the_vendored_prototype():
    # Pins the vendored table: a known country row, the Generic fallback,
    # a subdivision row, and that every row prices the fallback classes.
    from cafein._speed_limits import SPEED_LIMITS

    assert SPEED_LIMITS["FI"]["motorway_outside"] == 120
    assert SPEED_LIMITS["FI"]["residential_inside"] == 50
    assert SPEED_LIMITS["GB"]["motorway_outside"] == 113  # 70 mph, stored km/h
    assert SPEED_LIMITS["US-CA"]["motorway_outside"] == 105  # 65 mph
    assert SPEED_LIMITS[""]["residential_inside"] == 50
    # 48 ISO-addressable countries, 22 subdivision rows, one Generic;
    # the source's Northern Cyprus row shares Turkey's code and is
    # deliberately not vendored — TR carries Turkey's own values.
    countries = [key for key in SPEED_LIMITS if len(key) == 2]
    subdivisions = [key for key in SPEED_LIMITS if "-" in key]
    assert (len(countries), len(subdivisions)) == (48, 22)
    assert SPEED_LIMITS["TR"]["motorway_outside"] == 120
    for key, row in SPEED_LIMITS.items():
        assert "other_inside" in row and "other_outside" in row, key


# --- Junction delay classes ---------------------------------------------------


def _junctions(tags_by_vertex, u, v, ways, masks, vertices, highway=None):
    import numpy as np

    tags = [tags_by_vertex.get(i) for i in range(vertices)]
    forward = np.array(masks, dtype=np.uint8)
    if highway is None:
        highway = ["residential"] * len(u)
    return _osm.junction_delay_classes(
        tags, u, v, ways, highway, forward, forward, vertices
    )


def test_junction_classes_by_degree_and_control():
    # Way 10 runs 0→1→2; way 20 crosses it at vertex 1 (3→1→4).
    u, v = [0, 1, 3, 1], [1, 2, 1, 4]
    ways = [10, 10, 20, 20]
    car = [_osm.CAR] * 4
    # Untagged: four drivable approaches meet at 1 — a topological junction.
    fwd, rev = _junctions({}, u, v, ways, car, 5)
    assert list(fwd) == [_osm.JUNCTION_TOPOLOGICAL, 0, _osm.JUNCTION_TOPOLOGICAL, 0]
    assert list(rev) == [0, _osm.JUNCTION_TOPOLOGICAL, 0, _osm.JUNCTION_TOPOLOGICAL]
    # Signals mark every approach and sit atop the hierarchy.
    fwd, rev = _junctions({1: {"highway": "traffic_signals"}}, u, v, ways, car, 5)
    assert list(fwd) == [_osm.JUNCTION_SIGNALS, 0, _osm.JUNCTION_SIGNALS, 0]
    assert list(rev) == [0, _osm.JUNCTION_SIGNALS, 0, _osm.JUNCTION_SIGNALS]
    # A stop mapped onto a real junction is subsumed by it: the junction
    # alone charges, the sign adds nothing (the advance-sign rule).
    fwd, rev = _junctions({1: {"highway": "stop"}}, u, v, ways, car, 5)
    assert list(fwd) == [_osm.JUNCTION_TOPOLOGICAL, 0, _osm.JUNCTION_TOPOLOGICAL, 0]
    assert list(rev) == [0, _osm.JUNCTION_TOPOLOGICAL, 0, _osm.JUNCTION_TOPOLOGICAL]
    # A vertex two footways join is no junction for the car.
    fwd, rev = _junctions({}, u, v, ways, [_osm.CAR, _osm.CAR, _osm.WALK, _osm.WALK], 5)
    assert list(fwd) == [0, 0, 0, 0]
    assert list(rev) == [0, 0, 0, 0]


def test_ramp_junctions_outrank_topology_and_yield_to_signals():
    # Ramp way 20 (motorway_link) leaves road 10 at vertex 1; a second
    # road edge keeps vertex 1 at four approaches.
    u, v = [0, 1, 1], [1, 2, 3]
    ways = [10, 10, 20]
    highway = ["primary", "primary", "motorway_link"]
    car = [_osm.CAR] * 3
    fwd, rev = _junctions({}, u, v, ways, car, 4, highway=highway)
    # Vertex 1 is where the link meets non-link elements: a ramp junction
    # for every approach, outranking the topological class.
    assert list(fwd) == [_osm.JUNCTION_RAMP, 0, 0]
    assert list(rev) == [0, _osm.JUNCTION_RAMP, _osm.JUNCTION_RAMP]
    # Signals on the same vertex win the hierarchy.
    fwd, rev = _junctions(
        {1: {"highway": "traffic_signals"}}, u, v, ways, car, 4, highway=highway
    )
    assert list(fwd) == [_osm.JUNCTION_SIGNALS, 0, 0]
    assert list(rev) == [0, _osm.JUNCTION_SIGNALS, _osm.JUNCTION_SIGNALS]
    # Two links meeting each other only is no ramp junction.
    fwd, rev = _junctions(
        {}, [0, 1], [1, 2], [10, 20], car[:2], 3, highway=["motorway_link"] * 2
    )
    assert list(fwd) == [0, 0]
    assert list(rev) == [0, 0]


def test_priority_nodes_mark_their_own_way_only():
    # A give-way sign at vertex 1, mid-way on way 10 (0→1→2) — vertex 3
    # joins by way 20 only at vertex 2, so vertex 1 lies on one way.
    u, v = [0, 1, 2], [1, 2, 3]
    ways = [10, 10, 20]
    car = [_osm.CAR] * 3
    fwd, rev = _junctions({1: {"highway": "give_way"}}, u, v, ways, car, 4)
    # Both approaches along way 10 are marked; way 20 never touches 1.
    assert list(fwd) == [_osm.JUNCTION_PRIORITY, 0, 0]
    assert list(rev) == [0, _osm.JUNCTION_PRIORITY, 0]
    # direction=forward marks only the approach running with the way.
    fwd, rev = _junctions(
        {1: {"highway": "stop", "direction": "forward"}}, u, v, ways, car, 4
    )
    assert list(fwd) == [_osm.JUNCTION_PRIORITY, 0, 0]
    assert list(rev) == [0, 0, 0]
    # direction=backward marks only the against-the-way approach.
    fwd, rev = _junctions(
        {1: {"highway": "stop", "direction": "backward"}}, u, v, ways, car, 4
    )
    assert list(fwd) == [0, 0, 0]
    assert list(rev) == [0, _osm.JUNCTION_PRIORITY, 0]
