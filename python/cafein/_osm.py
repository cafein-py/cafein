"""Union OpenStreetMap extraction for multimodal street routing.

The walking build (`streets._walking_network`) stays the default; this module
adds the union extraction the cycling / e-scooter / car modes need: one
`pyrosm` pass over the broadly-filtered network, its tags normalised into
flat per-edge codes and per-direction mode-permission masks — plus, on a car
build, per-direction driving speeds and junction head classes — with
connectivity pruned separately for each mode. `StreetNetwork.from_osm`
consumes that output as the multimodal edge data the persisted street
arrays carry.
"""

import json
import math
import re
import warnings

import numpy as np
import pyrosm
from scipy import sparse
from scipy.sparse import csgraph

from .streets import MIN_ISLAND_VERTICES

# --- Mode permission bits (one per street mode) -------------------------------

WALK = 1 << 0
BICYCLE = 1 << 1
E_SCOOTER = 1 << 2
CAR = 1 << 3
WHEELCHAIR = 1 << 4

MODES = {
    "walk": WALK,
    "bicycle": BICYCLE,
    "e_scooter": E_SCOOTER,
    "car": CAR,
    "wheelchair": WHEELCHAIR,
}
"""The street modes and their permission bits. An e-bike reuses the bicycle
bit (same permissions, different speed); the e-scooter has its own bit,
"bicycle_like" by default. The wheelchair is walk-like without stairs."""


# --- Per-edge facility / directional flags -----------------------------------

FLAG_DISMOUNT = 1 << 0
"""Bicycles may traverse but must dismount (walk speed)."""
FLAG_BRIDGE = 1 << 1
FLAG_TUNNEL = 1 << 2
FLAG_INDOOR = 1 << 3
FLAG_STEPS = 1 << 4
FLAG_SEGREGATED = 1 << 5
FLAG_LIT = 1 << 6
FLAG_ROUNDABOUT = 1 << 7
"""A roundabout interior (`junction=roundabout`); the car delay model
charges it `b/4` in place of endpoint shares."""


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

# Default (foot, bicycle, car) permission per highway class, before any explicit
# access tags, following the OSM wiki defaults and R5's foot/bike traversal
# conventions. A way that passed the exclusion filter but carries an
# unrecognised highway value falls back to `_DEFAULT_HIGHWAY_PERMISSION`.
HIGHWAY_DEFAULTS = {
    "footway": (True, False, False),
    "pedestrian": (True, False, False),
    "steps": (True, False, False),
    "corridor": (True, False, False),
    "platform": (True, False, False),
    "path": (True, True, False),
    "cycleway": (False, True, False),
    "bridleway": (True, False, False),
    "track": (True, True, True),
    "living_street": (True, True, True),
    "residential": (True, True, True),
    "service": (True, True, True),
    "unclassified": (True, True, True),
    "tertiary": (True, True, True),
    "tertiary_link": (True, True, True),
    "secondary": (True, True, True),
    "secondary_link": (True, True, True),
    "primary": (True, True, True),
    "primary_link": (True, True, True),
    "trunk": (False, False, True),
    "trunk_link": (False, False, True),
    "elevator": (True, True, False),
    "road": (True, True, True),
    "busway": (False, False, False),
    "motorway": (False, False, True),
    "motorway_link": (False, False, True),
}
_DEFAULT_HIGHWAY_PERMISSION = (False, False, False)
"""A retained way with an unrecognised highway value (a typo, a lifecycle
value, or a type cafein does not model yet) denies every mode by default —
only an explicit `foot=`/`bicycle=`/`motorcar=` tag opens it — and the value
is reported, so routing never silently traverses an unmodelled way."""

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


def _wheelchair_permission(foot_default, highway, access, foot, wheelchair):
    """Wheelchair permission: the walk access ladder, then the `steps`
    class veto, then the `wheelchair` tag. `wheelchair=yes` rescues only
    the steps veto — never an access-ladder denial or a non-walkable
    class — and `wheelchair=no` denies everywhere; `limited` and unknown
    values keep the resolved default. Symmetric, like walking."""
    allowed, unknown = _resolve_mode(foot_default, access, foot)
    if highway == "steps" and wheelchair != "yes":
        allowed = False
    if wheelchair == "no":
        allowed = False
    return allowed, unknown


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


def _car_permission(car_default, access, vehicle, motor_vehicle, motorcar):
    """Car permission resolved down the OSM access hierarchy
    (type default → `access` → `vehicle` → `motor_vehicle` → `motorcar`),
    and whether an unknown value was seen.

    The same precedence semantics as the bicycle chain: a less-specific
    *allow* never grants a mode the highway type denies, a deny at any
    level propagates until a more-specific key re-grants, and the
    most-specific `motorcar=` overrides freely.
    """
    allowed = car_default
    unknown = False
    for value in (access, vehicle, motor_vehicle):
        if value is None:
            continue
        if value in _DENIED_ACCESS:
            allowed = False
        elif value in _ALLOWED_ACCESS:
            allowed = car_default
        else:
            unknown = True
    if motorcar is not None:
        if motorcar in _ALLOWED_ACCESS:
            allowed = True
        elif motorcar in _DENIED_ACCESS:
            allowed = False
        else:
            unknown = True
    return allowed, unknown


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
    foot_default, bike_default, car_default = HIGHWAY_DEFAULTS.get(
        highway, _DEFAULT_HIGHWAY_PERMISSION
    )
    access = row.get("access")

    foot_ok, foot_unknown = _resolve_mode(foot_default, access, row.get("foot"))
    chair_ok, _ = _wheelchair_permission(
        foot_default, highway, access, row.get("foot"), row.get("wheelchair")
    )
    bike_ok, dismount, bike_unknown = _bicycle_permission(
        bike_default, access, row.get("vehicle"), row.get("bicycle")
    )
    car_ok, car_unknown = _car_permission(
        car_default,
        access,
        row.get("vehicle"),
        row.get("motor_vehicle"),
        row.get("motorcar"),
    )

    # Directionality (bicycle, e-scooter, and the car). A roundabout is
    # implicitly
    # one-way unless an explicit false `oneway` overrides it; `junction=circular`
    # is not implicitly directional and follows its explicit `oneway`. A
    # dismounted cyclist is pedestrian-like and, like walking, ignores oneway.
    oneway = row.get("oneway")
    junction = row.get("junction")
    # Roundabouts and motorway carriageways are implicitly one-way; only
    # an explicit false `oneway` opens them.
    forced_oneway = (
        junction == "roundabout" or highway == "motorway"
    ) and oneway not in _FALSE_ONEWAY
    reversed_oneway = oneway == "-1"
    is_oneway = oneway in ("yes", "true", "1") or reversed_oneway or forced_oneway
    # Cars follow the base one-way strictly — no contraflow grants of
    # any kind.
    car_oneway = is_oneway
    car_forward = car_ok
    car_reverse = car_ok
    if car_oneway:
        if reversed_oneway:
            car_forward = False
        else:
            car_reverse = False

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
        | (CAR if car_forward else 0)
        | (WHEELCHAIR if chair_ok else 0)
    )
    reverse = (
        (WALK if foot_ok else 0)
        | (BICYCLE if bike_reverse else 0)
        | (E_SCOOTER if bike_reverse else 0)
        | (CAR if car_reverse else 0)
        | (WHEELCHAIR if chair_ok else 0)
    )
    flags = FLAG_DISMOUNT if dismount else 0
    unknown = foot_unknown or bike_unknown or car_unknown
    return forward, reverse, flags, unknown, unknown_highway


def edge_permissions(edges):
    """Per-edge (access_forward, access_reverse) masks, extra flags, and
    extraction diagnostics.

    The forward direction runs along the way's stored geometry; the reverse
    runs against it. Walking is permitted in both directions alike; bicycle and
    e-scooter follow oneway and its cycling exceptions. Returns
    ``(access_forward, access_reverse, flags, diagnostics)`` where `diagnostics`
    counts edge rows carrying an unrecognised access-hierarchy value —
    ``access``/``foot``/``vehicle``/``bicycle``/``motor_vehicle``/
    ``motorcar`` (`unknown_access`) — and rows
    with an unmodelled ``highway`` value (`unknown_highway`); `flags`
    OR-combines with `normalise_codes`' class flags.
    """
    columns = {
        tag: _column(edges, tag)
        for tag in (
            "highway",
            "access",
            "foot",
            "wheelchair",
            "bicycle",
            "vehicle",
            "motor_vehicle",
            "motorcar",
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


# --- Car speeds ---------------------------------------------------------------

_MPH_TO_KMH = 1.609344

_AREA_INVARIANT_SPEED = frozenset({"living_street", "service"})
"""Classes whose legal default does not vary between urban and rural."""

_EXTRA_CLASS_SPEEDS = {"track": 20}
"""Low-speed classes the vendored table does not carry: product defaults
in km/h, area-invariant, overridable through ``speed_limits=``."""


def parse_maxspeed(value):
    """km/h from a ``maxspeed`` tag value, or ``None`` where no numeric
    limit is stated.

    A bare number is km/h; ``NN mph`` converts. Everything else —
    ``none``, ``signals``, ``walk``, zone names, garbage — yields ``None``
    so the class default applies, never infinity.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    if text.endswith("mph"):
        text, factor = text[:-3].strip(), _MPH_TO_KMH
    else:
        factor = 1.0
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed * factor


def validated_country(country):
    """The country selector's syntax checked, warning-free: ``None``
    passes through, anything else must normalise to an ISO 3166-1
    alpha-2 or ISO 3166-2 code. The unknown-code fallback (a warning,
    never an error) stays in ``speed_limit_row``."""
    if country is None:
        return None
    code = str(country).strip().upper()
    if not re.fullmatch(r"[A-Z]{2}(-[A-Z0-9]{1,3})?", code):
        raise ValueError(
            f"country must be an ISO 3166-1 alpha-2 or ISO 3166-2 code, "
            f"not {country!r}"
        )
    return code


def speed_limit_row(country=None):
    """The legal-default speed row for a `country` selector.

    `country` is ISO 3166-1 alpha-2 (``"FI"``) or, where the vendored
    table carries subdivision rows, ISO 3166-2 (``"US-CA"``). The
    fallback chain is subdivision row → the country's generic row → the
    table's Generic row; ``None`` selects Generic with a warning naming
    the option, and a code the table does not carry warns as it falls
    back.
    """
    from cafein._speed_limits import SPEED_LIMITS

    if country is None:
        warnings.warn(
            "no country= given; using the Generic legal default speed "
            "limits for untagged ways",
            UserWarning,
            stacklevel=3,
        )
        return SPEED_LIMITS[""]
    code = validated_country(country)
    for key in (code, code.split("-")[0]):
        if key in SPEED_LIMITS:
            return SPEED_LIMITS[key]
    warnings.warn(
        f"no legal default speed limits for '{code}'; using the Generic row",
        UserWarning,
        stacklevel=3,
    )
    return SPEED_LIMITS[""]


def _class_speed(row, highway, inside):
    """The row's km/h default for a highway class, urban or rural."""
    if highway in _AREA_INVARIANT_SPEED or highway in _EXTRA_CLASS_SPEEDS:
        value = row.get(highway)
        if value is None:
            value = _EXTRA_CLASS_SPEEDS.get(highway)
        if value is not None:
            return value
    elif highway is not None and highway.endswith("_link"):
        value = row.get(highway)
        if value is not None:
            return value
        highway = highway[:-5]
    suffix = "_inside" if inside else "_outside"
    value = row.get(f"{highway}{suffix}") if highway is not None else None
    if value is not None:
        return value
    return row[f"other{suffix}"]


def _inside_urban_areas(edges, urban_areas):
    """Per-edge urban membership by spatial join against area polygons."""
    import geopandas as gpd

    crs = getattr(edges.geometry, "crs", None) or "EPSG:4326"
    if urban_areas.crs is None:
        raise ValueError("urban_areas must carry a CRS")
    frame = gpd.GeoDataFrame(geometry=edges.geometry.reset_index(drop=True), crs=crs)
    # The ACTIVE geometry, whatever its column name — a literal
    # "geometry" lookup could silently read an inactive column.
    areas = gpd.GeoDataFrame(geometry=urban_areas.geometry, crs=urban_areas.crs)
    joined = frame.sjoin(areas.to_crs(crs), how="left")
    inside = joined.groupby(level=0)["index_right"].first().notna()
    return inside.reindex(range(len(frame)), fill_value=False).to_numpy()


def _validated_speed_limits(speed_limits):
    """The override mapping checked against the known class columns."""
    from cafein._speed_limits import SPEED_LIMITS

    known = set(_EXTRA_CLASS_SPEEDS)
    for row in SPEED_LIMITS.values():
        known.update(row)
    overrides = dict(speed_limits)
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise ValueError(
            "unknown speed_limits classes: " + ", ".join(map(str, unknown))
        )
    for key, value in overrides.items():
        if not isinstance(value, (int, float)):
            raise ValueError(f"speed_limits[{key!r}] must be a positive km/h number")
        try:
            number = float(value)
        except OverflowError:
            # An integer beyond float range is a number that cannot be
            # finite; refuse it as such.
            number = math.inf
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"speed_limits[{key!r}] must be a positive km/h number")
        overrides[key] = number
    return overrides


def car_speeds(edges, country=None, urban=None, speed_limits=None):
    """Per-edge (forward_kmh, reverse_kmh) car speeds.

    Per direction: the parsed ``maxspeed:forward``/``:backward``, else the
    parsed ``maxspeed``, else the legal default for the way's class from
    the `country` row. `urban` picks the urban or rural default: a
    polygon GeoDataFrame resolves per edge by spatial join, a per-edge
    boolean array is taken as given, and ``None`` treats every way as
    urban — the conservative default for city-scale extracts, as the
    docs state. `speed_limits` layers user overrides (class column →
    km/h) over the resolved country row.
    """
    row = speed_limit_row(country)
    if speed_limits is not None:
        row = {**row, **_validated_speed_limits(speed_limits)}
    highway = _column(edges, "highway")
    base = _column(edges, "maxspeed")
    forward_tag = _column(edges, "maxspeed:forward")
    backward_tag = _column(edges, "maxspeed:backward")
    n = len(edges)
    if urban is None:
        inside = np.ones(n, dtype=bool)
    elif hasattr(urban, "geometry"):
        inside = _inside_urban_areas(edges, urban)
    else:
        inside = np.asarray(urban, dtype=bool)
        if inside.shape != (n,):
            raise ValueError("urban must be a boolean value per edge")
    forward = np.empty(n, dtype=np.float64)
    reverse = np.empty(n, dtype=np.float64)
    for i in range(n):
        tagged = parse_maxspeed(base[i])
        default = None
        for target, directional in ((forward, forward_tag), (reverse, backward_tag)):
            speed = parse_maxspeed(directional[i])
            if speed is None:
                speed = tagged
            if speed is None:
                if default is None:
                    default = float(_class_speed(row, highway[i], bool(inside[i])))
                speed = default
            target[i] = speed
    return forward, reverse


# --- Junction delay classes ---------------------------------------------------

JUNCTION_TOPOLOGICAL = 1
JUNCTION_PRIORITY = 2
JUNCTION_SIGNALS = 3
JUNCTION_RAMP = 4
"""The persisted junction head classes (0 = no junction): a topological
junction (three or more drivable approaches), a priority sign
(`highway=stop`/`give_way` — recorded, never charged by the delay
engine), signals (`highway=traffic_signals`), and a ramp junction
(an unsignalized endpoint where a `*_link` element meets non-link
drivable elements). Overlaps store one class by the calibration's own
hierarchy: signalized > ramp junction > topological > priority."""

_PRIORITY_NODE = frozenset({"stop", "give_way"})


def node_delay_tags(nodes):
    """Per-node tag dicts for `junction_delay_classes`.

    Reads the union extraction's nodes frame: the ``highway`` control tag
    and its optional ``direction``, from dedicated columns when pyrosm
    split them out, with the free-form ``tags`` mapping (a dict, or its
    JSON string form) as the fallback. Nodes carrying no control tag map
    to ``None``.
    """
    n = len(nodes)

    def column(name):
        if name not in nodes.columns:
            return np.full(n, None, dtype=object)
        return nodes[name].to_numpy(dtype=object)

    def clean(value):
        if value is None or value == "" or value == "nan":
            return None
        if isinstance(value, float) and value != value:
            return None
        return value

    highway = column("highway")
    direction = column("direction")
    tags = column("tags")
    out = []
    for i in range(n):
        h = clean(highway[i])
        d = clean(direction[i])
        t = tags[i]
        if isinstance(t, str):
            try:
                t = json.loads(t)
            except ValueError:
                t = None
        if isinstance(t, dict):
            if h is None:
                h = clean(t.get("highway"))
            if d is None:
                d = clean(t.get("direction"))
        out.append(None if h is None else {"highway": h, "direction": d})
    return out


def junction_delay_classes(
    node_tags, u, v, way_ids, highway, access_forward, access_reverse, vertex_count
):
    """Per-edge (forward_head, reverse_head) junction classes, u8.

    ``forward_head[e]`` is the class stored for the junction a car crosses
    traversing edge ``e`` along its stored geometry (at ``v[e]``);
    ``reverse_head[e]`` covers the opposite traversal into ``u[e]``.
    Signals, ramp junctions, and topological junctions are node
    properties marking every approach; overlaps resolve by the
    signalized > ramp junction > topological hierarchy. `stop`/`give_way`
    survive only where no higher class applies (an advance sign on an
    interior node): they associate by way membership — a node lying on
    exactly one drivable way marks that way's approaches only, honouring
    the node's ``direction=forward/backward`` (relative to that way's
    stored geometry); a node shared by several ways marks every
    approach — the conservative reading of ambiguous mapping. What each
    class costs (priority signs cost nothing; trip endpoints never
    charge) is the routing engine's rule; these arrays only say which
    crossings carry which class.
    """
    u = np.asarray(u)
    v = np.asarray(v)
    way_ids = np.asarray(way_ids)
    highway = np.asarray(highway, dtype=object)
    car = ((np.asarray(access_forward) | np.asarray(access_reverse)) & CAR) != 0
    degree = np.zeros(vertex_count, dtype=np.int64)
    np.add.at(degree, u[car], 1)
    np.add.at(degree, v[car], 1)

    ramp_edge = car & np.array(
        [isinstance(h, str) and h.endswith("_link") for h in highway], dtype=bool
    )
    has_ramp = np.zeros(vertex_count, dtype=bool)
    has_ramp[u[ramp_edge]] = True
    has_ramp[v[ramp_edge]] = True
    nonramp_edge = car & ~ramp_edge
    has_nonramp = np.zeros(vertex_count, dtype=bool)
    has_nonramp[u[nonramp_edge]] = True
    has_nonramp[v[nonramp_edge]] = True

    node_class = np.zeros(vertex_count, dtype=np.uint8)
    node_class[degree >= 3] = JUNCTION_TOPOLOGICAL
    node_class[has_ramp & has_nonramp] = JUNCTION_RAMP
    priority = {}
    for index, tags in enumerate(node_tags):
        if not isinstance(tags, dict):
            continue
        kind = tags.get("highway")
        if kind == "traffic_signals":
            node_class[index] = JUNCTION_SIGNALS
        elif kind in _PRIORITY_NODE and node_class[index] == 0:
            priority[index] = tags

    forward_head = node_class[v].copy()
    reverse_head = node_class[u].copy()
    if not priority:
        return forward_head, reverse_head
    # One pass builds the priority nodes' incident approaches, so each
    # node reads its own edges rather than scanning them all.
    heads = {node: [] for node in priority}
    tails = {node: [] for node in priority}
    for e in range(len(u)):
        into = heads.get(int(v[e]))
        if into is not None:
            into.append(e)
        out = tails.get(int(u[e]))
        if out is not None:
            out.append(e)
    for node, tags in priority.items():
        drivable = {int(way_ids[e]) for e in heads[node] + tails[node] if car[e]}
        # Shared by several ways (or carless): every approach is marked,
        # the conservative reading of ambiguous mapping.
        way = drivable.pop() if len(drivable) == 1 else None
        direction = {"forward": 1, "backward": -1}.get(tags.get("direction"), 0)
        for e in heads[node]:
            if way is None or (way_ids[e] == way and direction != -1):
                forward_head[e] = JUNCTION_PRIORITY
        for e in tails[node]:
            if way is None or (way_ids[e] == way and direction != 1):
                reverse_head[e] = JUNCTION_PRIORITY
    return forward_head, reverse_head


# --- Union extraction filter -------------------------------------------------

_EXCLUDED_HIGHWAY = [
    "abandoned",
    "construction",
    "motor",
    "proposed",
    "raceway",
]
"""Highway values no street mode may use: unbuilt or unmodelled ways."""

_MOTOR_ONLY_HIGHWAY = ["motorway", "motorway_link"]
"""Highway values only the car mode may use: excluded from the extraction
unless a car build asks for them."""


def union_filter(car=False):
    """The broad exclusion filter: unbuilt ways, ways mapped as areas, and
    private service ways — plus the motor-only classes unless a car build
    keeps them. Everything else — stairs, footways, paths, pedestrian
    streets, platforms, cycleways, tracks, ordinary roads — is retained, and
    the per-mode permission compiler decides who may use each."""
    excluded = list(_EXCLUDED_HIGHWAY)
    if not car:
        excluded += _MOTOR_ONLY_HIGHWAY
    return {
        "area": ["yes", "true", "1"],
        "highway": excluded,
        "service": ["private"],
    }


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
    "wheelchair",
    # Also requested explicitly (though in pyrosm's default highway columns) so
    # the directional and facility logic never silently loses them to a config
    # change: `oneway:bicycle`, `junction`, `segregated`.
    "oneway:bicycle",
    "junction",
    "segregated",
    # The car chain and its speeds.
    "motor_vehicle",
    "motorcar",
    "maxspeed",
    "maxspeed:forward",
    "maxspeed:backward",
]
"""Tags cafein needs kept on the extracted ways; the first block is not in
pyrosm's default highway columns, the second is requested defensively."""


def union_network(osm_pbf, bounding_box=None, car=False):
    """The union street network of a PBF extract, as (nodes, edges).

    One `pyrosm` pass over the broadly-filtered network with the multimodal
    tags retained. Unlike `streets._walking_network`, no mode is filtered out
    here; connectivity is pruned per mode afterwards by
    `prune_components_per_profile`. A car build (`car=True`) keeps the
    motor-only highway classes in the extraction.
    """
    osm = pyrosm.OSM(
        str(osm_pbf),
        bounding_box=bounding_box,
        engine="out_of_core",
        workers="auto",
    )
    network = osm.get_network(
        network_type="all",
        custom_filter=union_filter(car=car),
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
    junction = _column(edges, "junction")
    flags |= np.where(junction == "roundabout", FLAG_ROUNDABOUT, 0).astype(np.uint16)
    return edge_highway, edge_surface, edge_smoothness, flags


def prune_components_per_profile(
    u, v, vertex_count, access_forward, access_reverse, modes=None
):
    """Clear each mode's permission from the components too small to route on.

    `u`, `v` are the edges' endpoint vertex indices (as `streets._vertex_
    endpoints` returns them). Connectivity is judged per mode: a union component
    that only connects a mode's streets through a way that mode cannot use is
    not connected for it. For each mode, the sub-`MIN_ISLAND_VERTICES` weak
    components over that mode's permitted arcs have the mode's bit cleared (in
    both directions); the physical edge stays as long as another mode still uses
    it. Returns the pruned (access_forward, access_reverse).

    `modes` names the modes to prune, defaulting to all of them. Pruning is the
    only thing a mode selection changes: no edge is dropped, so a mode left out
    here keeps its raw permissions and can still be compiled later without
    rebuilding the graph.
    """
    if modes is None:
        bits = list(MODES.values())
    else:
        unknown = [mode for mode in modes if mode not in MODES]
        if unknown:
            raise ValueError(
                f"unknown street mode(s) {sorted(unknown)}; "
                f"expected any of {sorted(MODES)}"
            )
        bits = [MODES[mode] for mode in modes]
    forward = access_forward.copy()
    reverse = access_reverse.copy()
    for bit in bits:
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
