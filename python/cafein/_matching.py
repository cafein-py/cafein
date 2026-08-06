"""GTFS-pattern ↔ OSM-route-relation matching — tier 3's matcher.

A deterministic eligibility-then-selection rule, biased to no-match:
mode compatibility, exact folded ``ref`` agreement, and a stop-corridor
containment check gate the candidate set; the operator/network filter
then narrows it; and the surviving variant with the lowest normalized
stop-sequence edit distance wins only when it clears the acceptance
threshold and beats the runner-up by a margin — ties and near-ties
drop, and every candidate's component outcome is recorded for
diagnostics.
"""

import dataclasses
import math
import re

import numpy as np

from cafein.geometry import SNAP_TOLERANCE

#: Pattern-stop buffer radius forming the spatial corridor, in meters —
#: a v1 constant revisited by the calibration sweep.
RELATION_CORRIDOR_METERS = 500.0

#: Fraction of a relation's canonical boarding positions that must fall
#: inside the corridor for eligibility.
CORRIDOR_CONTAINMENT = 0.90

#: Consecutive stop/platform members within this distance collapse to
#: one canonical boarding position, in meters.
STOP_COLLAPSE_METERS = 50.0

#: Normalized edit-distance acceptance threshold and runner-up margin —
#: provisional v1 constants, calibrated by the validation sweep.
EDIT_DISTANCE_ACCEPT = 0.25
EDIT_DISTANCE_MARGIN = 0.10

#: Mode families accepted by ``osm_tiers=`` (keyed by the GTFS side),
#: and the OSM ``route=`` values each matches against.
MODE_ROUTES = {
    "tram": ("tram", "light_rail"),
    "subway": ("subway",),
    "train": ("train",),
    "bus": ("bus",),
    "trolleybus": ("trolleybus",),
    "ferry": ("ferry",),
}


def mode_of(route_type):
    """The GTFS route type's mode family, or ``None`` outside the map
    (extended types collapse to their base mode)."""
    if route_type == 3 or 700 <= route_type < 800:
        return "bus"
    if route_type == 11 or 800 <= route_type < 900:
        return "trolleybus"
    if route_type == 0 or 900 <= route_type < 1000:
        return "tram"
    if route_type == 1 or 400 <= route_type < 500:
        return "subway"
    if route_type == 2 or 100 <= route_type < 200:
        return "train"
    if route_type == 4 or 1000 <= route_type < 1100:
        return "ferry"
    return None


@dataclasses.dataclass(frozen=True)
class Selection:
    """A selected relation and the orientation that won: ``reversed``
    means the pattern travels the member sequence backward."""

    relation: object
    reversed: bool


@dataclasses.dataclass(frozen=True)
class Pattern:
    """One GTFS stop pattern with its route's matching metadata.

    ``stop_xy`` is the stops' projected coordinates in meters, aligned
    with ``stop_ids``; ``agency`` carries the feed's raw agency identity
    strings (name and id) — empty when the feed has no agency.
    """

    stop_ids: tuple
    stop_xy: object
    short_name: str | None
    long_name: str | None
    agency: tuple


_FOLD = re.compile(r"[\W_]+")
_SPACE = re.compile(r"\s+")


def fold(text):
    """The case/whitespace/punctuation-insensitive comparison form —
    for names, where the containment rule permits it."""
    if text is None:
        return ""
    return _FOLD.sub("", str(text).casefold())


def fold_ref(text):
    """The ref comparison form: case and whitespace fold only —
    punctuation stays significant (``1-A`` never equals ``1A``)."""
    if text is None:
        return ""
    return _SPACE.sub("", str(text).casefold())


def boarding_positions(relation):
    """Ordered ``(is_stop, lon, lat)`` boarding members of a relation.

    ``stop``-role and ``platform``-role members (role variants like
    ``stop_exit_only`` included) with materialized geometry; way or
    area platforms contribute their centroid.
    """
    ordered = []
    for member in relation.members:
        role = member.role
        if not (role.startswith("stop") or role.startswith("platform")):
            continue
        if member.geometry is None:
            # An unresolved boarding member (clipped extract, stale
            # data) keeps its slot: it counts against corridor
            # containment and matches nothing in the sequence.
            ordered.append((role.startswith("stop"), math.nan, math.nan))
            continue
        point = member.geometry
        if point.geom_type != "Point":
            point = point.centroid
        ordered.append((role.startswith("stop"), point.x, point.y))
    return ordered


def collapse_positions(kinds, xy):
    """The canonical boarding positions, as an ``(n, 2)`` array.

    Consecutive members within `STOP_COLLAPSE_METERS` are one physical
    boarding location — relations commonly carry both a ``stop`` node
    and a ``platform`` member for it — collapsed to one position that
    prefers the ``stop``-role coordinate.
    """
    canonical = []
    for is_stop, (x, y) in zip(kinds, xy):
        if canonical:
            last = canonical[-1]
            if math.hypot(x - last[0], y - last[1]) <= STOP_COLLAPSE_METERS:
                if is_stop and not last[2]:
                    canonical[-1] = [x, y, True]
                continue
        canonical.append([x, y, is_stop])
    return np.asarray([position[:2] for position in canonical], dtype=float)


def edit_distance(a, b):
    """Unit-cost Levenshtein distance normalized by the longer length."""
    if not a and not b:
        return 0.0
    previous = list(range(len(b) + 1))
    for row, item in enumerate(a, start=1):
        current = [row]
        for column, other in enumerate(b, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (item != other),
                )
            )
        previous = current
    return previous[-1] / max(len(a), len(b))


def select(pattern, entries):
    """The winning relation for a pattern, or ``None`` — plus the
    per-candidate diagnostics either way.

    ``entries`` are mode-compatible ``(relation, canonical_xy)`` pairs
    (``canonical_xy`` from `collapse_positions`, or ``None`` when the
    relation has no boarding members). Selection: exact folded ``ref``
    agreement, corridor containment, the operator/network filter, then
    the lowest normalized edit distance — accepted only under
    `EDIT_DISTANCE_ACCEPT` with an `EDIT_DISTANCE_MARGIN` lead over the
    runner-up.
    """
    diagnostics = []
    survivors = []
    for relation, canonical in entries:
        record = {
            "relation": relation.id,
            "route": relation.route,
            "ref": relation.ref,
            "stage": None,
            "corridor": None,
            "operator": None,
            "forward": None,
            "backward": None,
            "score": None,
            "reversed": None,
            "outcome": None,
        }
        diagnostics.append(record)
        if not _ref_agrees(pattern, relation):
            record["stage"] = "ref"
            continue
        if canonical is None or not len(canonical):
            record["stage"] = "no-boarding"
            continue
        distance = np.hypot(
            canonical[:, 0, None] - pattern.stop_xy[None, :, 0],
            canonical[:, 1, None] - pattern.stop_xy[None, :, 1],
        )
        nearest = distance.min(axis=1)
        record["corridor"] = float((nearest <= RELATION_CORRIDOR_METERS).mean())
        if record["corridor"] < CORRIDOR_CONTAINMENT:
            record["stage"] = "corridor"
            continue
        survivors.append((relation, record, distance, nearest))
    scored = []
    for relation, record, distance, nearest in _operator_filtered(pattern, survivors):
        assigned = _assigned_ids(pattern, distance, nearest)
        forward = edit_distance(pattern.stop_ids, assigned)
        backward = edit_distance(pattern.stop_ids, assigned[::-1])
        record["stage"] = "scored"
        record["forward"] = forward
        record["backward"] = backward
        record["score"] = min(forward, backward)
        record["reversed"] = backward < forward
        scored.append((record["score"], relation, record))
    if not scored:
        return None, diagnostics
    scored.sort(key=lambda entry: entry[0])
    best_score, best, best_record = scored[0]
    if best_score > EDIT_DISTANCE_ACCEPT:
        for _, _, record in scored:
            record["outcome"] = "over-threshold"
        return None, diagnostics
    if len(scored) > 1 and scored[1][0] - best_score < EDIT_DISTANCE_MARGIN:
        for _, _, record in scored:
            record["outcome"] = "near-tie"
        return None, diagnostics
    for _, _, record in scored[1:]:
        record["outcome"] = "runner-up"
    best_record["outcome"] = "selected"
    return Selection(best, bool(best_record["reversed"])), diagnostics


def _ref_agrees(pattern, relation):
    """Exact folded ref agreement; a ref-less relation is eligible only
    for a ref-less route, through name equality-or-containment."""
    short = fold_ref(pattern.short_name)
    ref = fold_ref(relation.ref)
    if short:
        return ref == short
    if ref:
        return False
    name = fold(relation.name)
    long_name = fold(pattern.long_name)
    if not name or not long_name:
        return False
    return name in long_name or long_name in name


def _operator_filtered(pattern, survivors):
    """The operator/network filter: *match* candidates when any exist,
    else the *absent* group — present-and-disagreeing tags always
    disqualify. Skipped when the feed names no agency."""
    identity = {fold(value) for value in pattern.agency if fold(value)}
    if not identity:
        return survivors
    matches = []
    absent = []
    for entry in survivors:
        relation, record = entry[0], entry[1]
        present = [
            fold(tag)
            for tag in (relation.operator, relation.network)
            if tag is not None
        ]
        if any(tag in identity for tag in present if tag):
            record["operator"] = "match"
            matches.append(entry)
        elif not any(present):
            record["operator"] = "absent"
            absent.append(entry)
        else:
            record["operator"] = "mismatch"
            record["stage"] = "operator"
    if matches:
        for entry in absent:
            entry[1]["stage"] = "operator"
        return matches
    return absent


def _assigned_ids(pattern, distance, nearest):
    """Each canonical position's nearest pattern stop within the snap
    tolerance, or a gap symbol that matches nothing."""
    closest = distance.argmin(axis=1)
    return tuple(
        pattern.stop_ids[column] if nearest[row] <= SNAP_TOLERANCE else ("gap", row)
        for row, column in enumerate(closest)
    )
