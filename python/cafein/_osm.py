"""Union OpenStreetMap extraction for multimodal street routing.

The walking build (`streets._walking_network`) stays the default; this module
adds the union extraction the cycling / e-scooter modes need: one `pyrosm`
pass over the broadly-filtered network, its tags normalised into flat per-edge
codes and per-direction mode-permission masks, with connectivity pruned
separately for each mode. Nothing here feeds the routing graph yet — that is a
later step; the output is the multimodal edge data the format-12 arrays will
carry.
"""

import numpy as np
import pyrosm
from scipy import sparse
from scipy.sparse import csgraph

from .streets import MIN_ISLAND_VERTICES

# --- Mode permission bits (one per street mode) -------------------------------

WALK = 1 << 0
BICYCLE = 1 << 1
E_SCOOTER = 1 << 2

MODES = {"walk": WALK, "bicycle": BICYCLE, "e_scooter": E_SCOOTER}
"""The street modes and their permission bits. An e-bike reuses the bicycle
bit (same permissions, different speed); the e-scooter has its own bit,
"bicycle_like" by default."""


# --- Per-edge facility / directional flags -----------------------------------

FLAG_DISMOUNT = 1 << 0
"""Bicycles may traverse but must dismount (walk speed)."""
FLAG_BRIDGE = 1 << 1
FLAG_TUNNEL = 1 << 2
FLAG_INDOOR = 1 << 3
FLAG_STEPS = 1 << 4
FLAG_SEGREGATED = 1 << 5
FLAG_LIT = 1 << 6


# --- Normalised class codes (the edge_highway / edge_surface / … arrays) ------

HIGHWAY_CODES = {
    "unknown": 0,
    "motorway": 1,
    "motorway_link": 2,
    "trunk": 3,
    "trunk_link": 4,
    "primary": 5,
    "primary_link": 6,
    "secondary": 7,
    "secondary_link": 8,
    "tertiary": 9,
    "tertiary_link": 10,
    "unclassified": 11,
    "residential": 12,
    "living_street": 13,
    "service": 14,
    "pedestrian": 15,
    "footway": 16,
    "path": 17,
    "cycleway": 18,
    "bridleway": 19,
    "track": 20,
    "steps": 21,
    "corridor": 22,
    "elevator": 23,
    "platform": 24,
    "road": 25,
    "busway": 26,
}
"""Highway values to their `edge_highway` code; unrecognised values map to 0."""

SURFACE_CODES = {
    "unknown": 0,
    "paved": 1,
    "asphalt": 2,
    "concrete": 3,
    "paving_stones": 4,
    "sett": 5,
    "cobblestone": 6,
    "unpaved": 7,
    "compacted": 8,
    "fine_gravel": 9,
    "gravel": 10,
    "ground": 11,
    "dirt": 12,
    "grass": 13,
    "sand": 14,
    "wood": 15,
    "metal": 16,
}
"""Surface values to their `edge_surface` code; unrecognised values map to 0."""

SMOOTHNESS_CODES = {
    "unknown": 0,
    "excellent": 1,
    "good": 2,
    "intermediate": 3,
    "bad": 4,
    "very_bad": 5,
    "horrible": 6,
    "very_horrible": 7,
    "impassable": 8,
}
"""Smoothness values to their `edge_smoothness` code; unknown maps to 0."""


# --- Permission model --------------------------------------------------------

# Default (foot, bicycle) permission per highway class, before any explicit
# access tags, following the OSM wiki defaults and R5's foot/bike traversal
# conventions. A way that passed the exclusion filter but carries an
# unrecognised highway value falls back to `_DEFAULT_HIGHWAY_PERMISSION`.
HIGHWAY_DEFAULTS = {
    "footway": (True, False),
    "pedestrian": (True, False),
    "steps": (True, False),
    "corridor": (True, False),
    "platform": (True, False),
    "path": (True, True),
    "cycleway": (False, True),
    "bridleway": (True, False),
    "track": (True, True),
    "living_street": (True, True),
    "residential": (True, True),
    "service": (True, True),
    "unclassified": (True, True),
    "tertiary": (True, True),
    "tertiary_link": (True, True),
    "secondary": (True, True),
    "secondary_link": (True, True),
    "primary": (True, True),
    "primary_link": (True, True),
    "trunk": (False, False),
    "trunk_link": (False, False),
    "elevator": (True, True),
    "road": (True, True),
    "busway": (False, False),
}
_DEFAULT_HIGHWAY_PERMISSION = (False, False)
"""A retained way with an unrecognised highway value (a typo, a lifecycle
value, or a type cafein does not model yet) denies both modes by default —
only an explicit `foot=`/`bicycle=` tag opens it — and the value is reported,
so routing never silently traverses an unmodelled way."""

_ALLOWED_ACCESS = frozenset(
    {"yes", "designated", "permissive", "destination", "customers", "official"}
)
"""Access values that permit routing: explicit allow, plus destination /
customers (reachable, just usage-restricted) treated as allowed."""

_DENIED_ACCESS = frozenset(
    {
        "no",
        "private",
        "use_sidepath",
        "dismount",
        # Restrictive values that are not general-public access: routable only
        # for their stated purpose, so denied for general routing here (a
        # mode-specific foot=/bicycle= tag still overrides).
        "delivery",
        "agricultural",
        "forestry",
        "permit",
        "military",
    }
)
"""Access values that deny general routing. `dismount`/`use_sidepath` are
handled specially for bicycle before this set is consulted; for foot they
deny."""


def _resolve_mode(default, general, specific, denied_extra=()):
    """Resolve one mode's permission from its highway default, the general
    `access` value, and its mode-specific tag, in precedence order.

    The highway type's implied per-mode default is more specific than the
    general `access` tag, so a general *allow* (``access=yes``/``destination``/
    …) does not grant a mode the type denies — only a general *deny*
    (``access=no``/``private``) overrides the type default. The mode-specific
    tag (``foot=``/``bicycle=``) is most specific and overrides freely. Returns
    (allowed, saw_unknown).
    """
    allowed = default
    unknown = False
    if general is not None:
        if general in _DENIED_ACCESS or general in denied_extra:
            allowed = False
        elif general not in _ALLOWED_ACCESS:
            unknown = True  # conservative: keep the type default
    if specific is not None:
        if specific in _ALLOWED_ACCESS:
            allowed = True
        elif specific in _DENIED_ACCESS or specific in denied_extra:
            allowed = False
        else:
            unknown = True
    return allowed, unknown


_CONTRAFLOW = frozenset(
    {"opposite", "opposite_lane", "opposite_track", "opposite_share_busway"}
)


_FALSE_ONEWAY = frozenset({"no", "false", "0"})

_CYCLEWAY_SIDES = ("cycleway", "cycleway:left", "cycleway:right", "cycleway:both")

_ON_EDGE_CYCLEWAY = (
    frozenset(
        {"lane", "track", "shared_lane", "share_busway", "crossing", "yes", "shared"}
    )
    | _CONTRAFLOW
)
"""Cycleway values that denote a facility on the road edge itself (so a
direction qualifier applies to it). `no`, `separate`, `none`, and a missing
companion are NOT on-edge — the road carries no lane to run contraflow on."""


def _contraflow_reopens(row, reversed_oneway):
    """Whether a mapped on-edge cycle facility re-opens the direction the base
    oneway blocks (forward when `reversed_oneway`, else reverse).

    The legacy ``cycleway=opposite*`` values are defined relative to the
    oneway, so they always re-open the blocked direction. The modern
    ``cycleway:left/right/both:oneway`` values are relative to the stored
    geometry: a false alias is a two-way lane (re-opens either blocked
    direction), while an explicit direction re-opens only the way it permits —
    ``-1`` (running against the geometry) re-opens a forward-blocking base, and
    ``yes`` (running with the geometry) re-opens a reverse-blocking base. The
    modern qualifier is honoured only when the companion ``cycleway:{side}`` is
    an on-edge facility — a ``separate`` or absent lane cannot carry contraflow.
    """
    if any(row.get(side) in _CONTRAFLOW for side in _CYCLEWAY_SIDES):
        return True
    for side in ("cycleway:left", "cycleway:right", "cycleway:both"):
        if row.get(side) not in _ON_EDGE_CYCLEWAY:
            continue
        direction = row.get(f"{side}:oneway")
        if direction is None:
            continue
        if direction in _FALSE_ONEWAY:
            return True
        if reversed_oneway and direction in ("yes", "true", "1"):
            return True
        if not reversed_oneway and direction == "-1":
            return True
    return False


def _bicycle_permission(bike_default, access, vehicle, bicycle):
    """Bicycle permission resolved down the OSM access hierarchy
    (type default → `access` → `vehicle` → `bicycle`), and (dismount, unknown).

    A less-specific *allow* never grants a mode the highway type denies, but a
    more-specific key can re-grant what a less-specific *deny* removed: so
    `vehicle=yes` re-opens a bike-permitting way closed by `access=no`, while
    `vehicle=yes` on a footway (type denies bikes) does not. An explicit
    `bicycle=` overrides everything; `dismount` permits at walk speed (flagged),
    `use_sidepath` denies.
    """
    allowed = bike_default
    unknown = False
    # access / vehicle: a deny propagates; an allow only re-grants a way the
    # type already permits (never grants a type-denied mode).
    for value in (access, vehicle):
        if value is None:
            continue
        if value in _DENIED_ACCESS:
            allowed = False
        elif value in _ALLOWED_ACCESS:
            allowed = bike_default
        else:
            unknown = True
    # explicit bicycle= is most specific and overrides freely.
    dismount = bicycle == "dismount"
    if dismount:
        allowed = True
    elif bicycle is not None:
        if bicycle in _ALLOWED_ACCESS:
            allowed = True
        elif bicycle in _DENIED_ACCESS:
            allowed = False
        else:
            unknown = True
    return allowed, dismount, unknown


def _row_permissions(row):
    """(forward_mask, reverse_mask, flags, unknown_access, unknown_highway).

    `row` maps tag → value (or None). Walking is undirected (pedestrians ignore
    oneway); bicycle and e-scooter are directional, with oneway, roundabouts,
    `oneway:bicycle`, and contraflow cycleway tags resolved per the design's
    precedence ladder. e-scooter mirrors bicycle (the default "bicycle_like"
    policy). An unrecognised highway value denies both modes (only explicit
    mode tags open it) and is reported.
    """
    highway = row.get("highway")
    unknown_highway = highway is not None and highway not in HIGHWAY_DEFAULTS
    foot_default, bike_default = HIGHWAY_DEFAULTS.get(
        highway, _DEFAULT_HIGHWAY_PERMISSION
    )
    access = row.get("access")

    foot_ok, foot_unknown = _resolve_mode(foot_default, access, row.get("foot"))
    bike_ok, dismount, bike_unknown = _bicycle_permission(
        bike_default, access, row.get("vehicle"), row.get("bicycle")
    )

    # Directionality (bicycle/e-scooter only). A roundabout is implicitly
    # one-way unless an explicit false `oneway` overrides it; `junction=circular`
    # is not implicitly directional and follows its explicit `oneway`. A
    # dismounted cyclist is pedestrian-like and, like walking, ignores oneway.
    oneway = row.get("oneway")
    junction = row.get("junction")
    forced_oneway = junction == "roundabout" and oneway not in _FALSE_ONEWAY
    reversed_oneway = oneway == "-1"
    is_oneway = oneway in ("yes", "true", "1") or reversed_oneway or forced_oneway

    bike_forward = bike_ok
    bike_reverse = bike_ok
    if is_oneway and not dismount:
        if reversed_oneway:
            bike_forward = False
        else:
            bike_reverse = False

    oneway_bicycle = row.get("oneway:bicycle")
    if dismount:
        pass  # already bidirectional at walk speed
    elif oneway_bicycle in _FALSE_ONEWAY:
        # Bicycles are exempt from the oneway — both directions (if allowed).
        bike_forward = bike_reverse = bike_ok
    elif oneway_bicycle in ("yes", "true", "1", "-1"):
        bike_forward = bike_ok and oneway_bicycle != "-1"
        bike_reverse = bike_ok and oneway_bicycle == "-1"
    elif is_oneway and _contraflow_reopens(row, reversed_oneway):
        # A contraflow cycleway re-opens the direction the base oneway blocked.
        if reversed_oneway:
            bike_forward = bike_ok
        else:
            bike_reverse = bike_ok

    forward = (
        (WALK if foot_ok else 0)
        | (BICYCLE if bike_forward else 0)
        | (E_SCOOTER if bike_forward else 0)
    )
    reverse = (
        (WALK if foot_ok else 0)
        | (BICYCLE if bike_reverse else 0)
        | (E_SCOOTER if bike_reverse else 0)
    )
    flags = FLAG_DISMOUNT if dismount else 0
    return forward, reverse, flags, (foot_unknown or bike_unknown), unknown_highway


def edge_permissions(edges):
    """Per-edge (access_forward, access_reverse) masks, extra flags, and
    extraction diagnostics.

    The forward direction runs along the way's stored geometry; the reverse
    runs against it. Walking is permitted in both directions alike; bicycle and
    e-scooter follow oneway and its cycling exceptions. Returns
    ``(access_forward, access_reverse, flags, diagnostics)`` where `diagnostics`
    counts edge rows carrying an unrecognised access-hierarchy value —
    ``access``/``foot``/``vehicle``/``bicycle`` (`unknown_access`) — and rows
    with an unmodelled ``highway`` value (`unknown_highway`); `flags`
    OR-combines with `normalise_codes`' class flags.
    """
    columns = {
        tag: _column(edges, tag)
        for tag in (
            "highway",
            "access",
            "foot",
            "bicycle",
            "vehicle",
            "oneway",
            "oneway:bicycle",
            "junction",
            "cycleway",
            "cycleway:left",
            "cycleway:right",
            "cycleway:both",
            "cycleway:left:oneway",
            "cycleway:right:oneway",
            "cycleway:both:oneway",
        )
    }
    n = len(edges)
    forward = np.zeros(n, dtype=np.uint8)
    reverse = np.zeros(n, dtype=np.uint8)
    flags = np.zeros(n, dtype=np.uint16)
    unknown_access = 0
    unknown_highway = 0
    for i in range(n):
        row = {tag: columns[tag][i] for tag in columns}
        f, r, fl, unk_access, unk_highway = _row_permissions(row)
        forward[i] = f
        reverse[i] = r
        flags[i] = fl
        unknown_access += int(unk_access)
        unknown_highway += int(unk_highway)
    diagnostics = {
        "unknown_access": unknown_access,
        "unknown_highway": unknown_highway,
    }
    return forward, reverse, flags, diagnostics


# --- Union extraction filter -------------------------------------------------

_EXCLUDED_HIGHWAY = [
    "abandoned",
    "construction",
    "motor",
    "motorway",
    "motorway_link",
    "proposed",
    "raceway",
]
"""Highway values no street mode we model may use: motor-only and unbuilt."""

_UNION_FILTER = {
    "area": ["yes", "true", "1"],
    "highway": _EXCLUDED_HIGHWAY,
    "service": ["private"],
}
"""The broad exclusion filter: motor-only or unbuilt ways, ways mapped as
areas, and private service ways. Everything else — stairs, footways, paths,
pedestrian streets, platforms, cycleways, tracks, ordinary roads — is retained,
and the per-mode permission compiler decides who may use each."""

_EXTRA_ATTRIBUTES = [
    "vehicle",
    "cycleway:left",
    "cycleway:right",
    "cycleway:both",
    "cycleway:left:oneway",
    "cycleway:right:oneway",
    "cycleway:both:oneway",
    "layer",
    "indoor",
    "incline",
    # Also requested explicitly (though in pyrosm's default highway columns) so
    # the directional and facility logic never silently loses them to a config
    # change: `oneway:bicycle`, `junction`, `segregated`.
    "oneway:bicycle",
    "junction",
    "segregated",
]
"""Tags cafein needs kept on the extracted ways; the first block is not in
pyrosm's default highway columns, the second is requested defensively."""


def union_network(osm_pbf, bounding_box=None):
    """The union street network of a PBF extract, as (nodes, edges).

    One `pyrosm` pass over the broadly-filtered network with the multimodal
    tags retained. Unlike `streets._walking_network`, no mode is filtered out
    here; connectivity is pruned per mode afterwards by
    `prune_components_per_profile`.
    """
    osm = pyrosm.OSM(
        str(osm_pbf),
        bounding_box=bounding_box,
        engine="out_of_core",
        workers="auto",
    )
    network = osm.get_network(
        network_type="all",
        custom_filter=_UNION_FILTER,
        filter_type="exclude",
        extra_attributes=_EXTRA_ATTRIBUTES,
        nodes=True,
    )
    if network is None:
        raise ValueError(f"no routable ways in '{osm_pbf}'")
    return network


def _column(edges, name):
    """A way-tag column as an object array with missing values as `None`, or
    all-`None` when pyrosm dropped the column (a tag absent everywhere in the
    extract yields no column). pyrosm's out-of-core engine returns string
    columns whose missing entries are the literal string ``"nan"`` (and float
    ``NaN`` on the in-memory path), both normalised to `None` here so the plain
    ``is None`` checks downstream are correct."""
    if name not in edges.columns:
        return np.full(len(edges), None, dtype=object)
    values = edges[name].to_numpy(dtype=object)
    return np.array(
        [
            (
                None
                if v is None
                or v == ""
                or v == "nan"
                or (isinstance(v, float) and v != v)
                else v
            )
            for v in values
        ],
        dtype=object,
    )


def normalise_codes(edges):
    """The (edge_highway, edge_surface, edge_smoothness, flags) arrays."""
    highway = _column(edges, "highway")
    surface = _column(edges, "surface")
    smoothness = _column(edges, "smoothness")

    def coded(values, table):
        return np.array(
            [table.get(v, 0) if v is not None else 0 for v in values], dtype=np.uint8
        )

    edge_highway = coded(highway, HIGHWAY_CODES)
    edge_surface = coded(surface, SURFACE_CODES)
    edge_smoothness = coded(smoothness, SMOOTHNESS_CODES)

    flags = np.zeros(len(edges), dtype=np.uint16)
    flags |= np.where(highway == "steps", FLAG_STEPS, 0).astype(np.uint16)
    for tag, bit in (
        ("bridge", FLAG_BRIDGE),
        ("tunnel", FLAG_TUNNEL),
        ("indoor", FLAG_INDOOR),
        ("segregated", FLAG_SEGREGATED),
        ("lit", FLAG_LIT),
    ):
        column = _column(edges, tag)
        present = np.array(
            [v is not None and v not in ("no", "false", "0") for v in column],
            dtype=bool,
        )
        flags |= np.where(present, bit, 0).astype(np.uint16)
    return edge_highway, edge_surface, edge_smoothness, flags


def prune_components_per_profile(u, v, vertex_count, access_forward, access_reverse):
    """Clear each mode's permission from the components too small to route on.

    `u`, `v` are the edges' endpoint vertex indices (as `streets._vertex_
    endpoints` returns them). Connectivity is judged per mode: a union component
    that only connects a mode's streets through a way that mode cannot use is
    not connected for it. For each mode, the sub-`MIN_ISLAND_VERTICES` weak
    components over that mode's permitted arcs have the mode's bit cleared (in
    both directions); the physical edge stays as long as another mode still uses
    it. Returns the pruned (access_forward, access_reverse).
    """
    forward = access_forward.copy()
    reverse = access_reverse.copy()
    for bit in MODES.values():
        usable = ((forward | reverse) & bit) != 0
        if not usable.any():
            continue
        graph = sparse.coo_matrix(
            (np.ones(usable.sum()), (u[usable], v[usable])),
            shape=(vertex_count, vertex_count),
        )
        _, labels = csgraph.connected_components(graph, directed=False)
        sizes = np.bincount(labels, minlength=vertex_count)
        # An edge is on a routable component for this mode when both its
        # endpoints sit in a component of at least MIN_ISLAND_VERTICES.
        small = (sizes[labels[u]] < MIN_ISLAND_VERTICES) | (
            sizes[labels[v]] < MIN_ISLAND_VERTICES
        )
        drop = small & usable
        clear = np.uint8(0xFF ^ bit)
        forward[drop] &= clear
        reverse[drop] &= clear
    return forward, reverse
