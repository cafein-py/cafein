"""The member-way stitcher — tier 3's chain builder.

Turns a route relation's way members into one continuous LineString.
Relation member order leads: OSM defines it as the travel order for
``route=*`` relations, the extraction contract pins it, and real routes
revisit intersections (crossings, terminal loops) that adjacency alone
cannot disambiguate — so when the ordered-next member connects it is
taken without polling the others, and the downstream validation gates
(stop snapping, monotone linear referencing, length sanity) backstop a
corrupted order. Only when order breaks — shuffled members — does the
walk fall back to adjacency, and there it continues solely on a unique
candidate. Anything ambiguous or broken **refuses instead of
repairing** (`StitchRefusal` with a reason code): a gap beyond
tolerance, more than one continuation candidate, an unresolvable
member, or a closed ring whose travel direction cannot be verified.
Refusal drops the pattern to the next ladder tier — an honest
fallthrough beats a guessed geometry.
"""

import math

import shapely

#: Endpoints within this distance chain together; anything farther is a
#: gap. A v1 constant, revisited by the calibration sweep.
GAP_TOLERANCE_METERS = 10.0

#: Ring ways with these tag values travel in stored-vertex order.
_RING_DIRECTED = {"junction": ("roundabout", "circular"), "oneway": ("yes",)}

#: Highway values that imply ``oneway=yes`` when untagged.
_IMPLIED_ONEWAY_HIGHWAYS = ("motorway", "motorway_link")


def effective_direction(tags):
    """A way's legal travel direction from its tags: ``1`` stored
    order only, ``-1`` reversed order only, ``0`` both, ``None`` for
    an unrecognised ``oneway`` value. Covers explicit ``oneway``,
    roundabout/circular junctions, and OSM's implied one-way motorway
    semantics."""
    oneway = tags.get("oneway")
    if oneway in ("no", "false", "0"):
        return 0
    if oneway in ("yes", "true", "1"):
        return 1
    if oneway in ("-1", "reverse"):
        return -1
    if oneway is None:
        if tags.get("junction") in _RING_DIRECTED["junction"]:
            return 1
        if tags.get("highway") in _IMPLIED_ONEWAY_HIGHWAYS:
            return 1
        return 0
    return None


class StitchRefusal(ValueError):
    """The relation cannot be stitched; ``reason`` says why.

    Reasons: ``empty``, ``unresolved-member``, ``malformed-member``
    (fewer than two finite coordinates or zero extent), ``gap`` (no
    connectable
    continuation within tolerance), ``branching`` (more than one
    continuation candidate), ``ring-direction`` (a closed ring without
    verified one-way travel), ``ring-touch`` (a ring whose entry or
    exit cannot be placed), ``member-direction`` (the chain traverses
    a one-way member against its legal direction).
    """

    def __init__(self, reason, detail=""):
        super().__init__(reason, detail)
        self.reason = reason
        self.detail = detail

    def __str__(self):
        return f"{self.reason}: {self.detail}" if self.detail else self.reason


def stitch(members, gap_tolerance=GAP_TOLERANCE_METERS):
    """One LineString through every way member, or `StitchRefusal`.

    ``members`` are the relation's route-way members in relation order
    (already role-filtered by the caller): objects with ``geometry``
    (LineString or ``None``) and ``tags`` (the `RelationMember`
    contract). The output direction follows the first member.
    """
    segments = []
    for member in members:
        if member.geometry is None:
            raise StitchRefusal(
                "unresolved-member", f"way {getattr(member, 'id', '?')}"
            )
        coordinates = list(member.geometry.coords)
        if len(coordinates) < 2 or not all(
            math.isfinite(value) for point in coordinates for value in point
        ):
            raise StitchRefusal("malformed-member", f"way {getattr(member, 'id', '?')}")
        if all(_meters(coordinates[0], point) <= 0.01 for point in coordinates):
            raise StitchRefusal(
                "malformed-member", f"zero-extent way {getattr(member, 'id', '?')}"
            )
        # True closure only: an OSM closed way repeats its first node,
        # so the coordinates match exactly — a near-looping open way is
        # an ordinary chain member, never a ring. A closed member too
        # small to be a ring is malformed, not silently open.
        closed = coordinates[0] == coordinates[-1]
        if closed and len(coordinates) < 4:
            raise StitchRefusal(
                "malformed-member",
                f"degenerate closed way {getattr(member, 'id', '?')}",
            )
        segments.append(
            _Segment(
                coordinates[:-1] if closed else coordinates,
                dict(member.tags),
                closed,
                getattr(member, "id", None),
            )
        )
    if not segments:
        raise StitchRefusal("empty", "no way members")
    parts = _walk(segments, gap_tolerance)
    return shapely.LineString(_merge(parts))


class _Segment:
    __slots__ = ("coordinates", "tags", "ring", "id")

    def __init__(self, coordinates, tags, ring, identifier=None):
        self.coordinates = coordinates
        self.tags = tags
        self.ring = ring
        self.id = identifier


def _walk(segments, tolerance):
    """The ordered walk with unique-adjacency fallback.

    The first (non-ring) segment seeds the chain, oriented so its far
    end points at the rest; each step first tries the next unused
    member in relation order, then — if it cannot connect — the unique
    unused member that can. Two candidates refuse as ``branching``,
    none as ``gap``. Rings connect through the explicit ring rule.
    """
    if all(segment.ring for segment in segments):
        raise StitchRefusal("ring-touch", "a ring needs open neighbours")
    order = list(range(len(segments)))
    first = next(index for index in order if not segments[index].ring)
    seed = segments[first]
    used = {first}
    remaining = [index for index in order if index != first]
    coordinates = list(seed.coordinates)
    seed_reversed = False
    if remaining:
        # Orient the seed against the NEXT relation member when that
        # decides a unique direction (relation order is the travel
        # order); fall back to any-connectivity only when it does not.
        ordered_next = [segments[remaining[0]]]
        forward = _connectable(coordinates[-1], ordered_next, tolerance)
        backward = _connectable(coordinates[0], ordered_next, tolerance)
        if backward and not forward:
            coordinates.reverse()
            seed_reversed = True
        elif not forward and not backward:
            others = [segments[index] for index in remaining]
            if not _connectable(coordinates[-1], others, tolerance) and _connectable(
                coordinates[0], others, tolerance
            ):
                coordinates.reverse()
                seed_reversed = True
    parts = [coordinates]
    # Per-part traversal record: (segment, reversed). Ring arcs enter
    # in verified direction (reversed=False); a later global flip may
    # still reverse them, which direction validation refuses.
    traversal = [(seed, seed_reversed)]
    end = coordinates[-1]
    flipped = False
    while remaining:
        step = _next_step(end, segments, remaining, tolerance)
        if step is None:
            if not flipped:
                # An interior seed grows one side first; continue from
                # the chain's other terminus before calling it a gap.
                flipped = True
                parts = [list(reversed(part)) for part in reversed(parts)]
                traversal = [
                    (segment, not was_reversed)
                    for segment, was_reversed in reversed(traversal)
                ]
                end = parts[-1][-1]
                continue
            raise StitchRefusal("gap", "no continuation within tolerance")
        index, part, new_end, reversed_part = step
        parts.append(part)
        traversal.append(
            (segments[index], False if reversed_part is None else reversed_part)
        )
        used.add(index)
        remaining.remove(index)
        end = new_end
    _check_directions(traversal)
    return parts


def _check_directions(traversal):
    """Refuse a chain that traverses any one-way member against its
    legal direction (``junction=roundabout/circular`` implies stored
    order; ``oneway=-1``/``reverse`` demands the reverse). A ring arc
    entered the chain in its verified direction — it only violates
    when the interior-seed flip reversed it afterwards."""
    for segment, was_reversed in traversal:
        if segment.ring:
            if was_reversed:
                raise StitchRefusal(
                    "ring-direction",
                    f"chain growth reversed the ring arc of way {segment.id}",
                )
            continue
        direction = effective_direction(segment.tags)
        if direction == 0:
            continue
        if direction is None:
            raise StitchRefusal(
                "member-direction",
                "unrecognised oneway "
                f"{segment.tags.get('oneway')!r} on way {segment.id}",
            )
        if (direction == 1) == bool(was_reversed):
            raise StitchRefusal(
                "member-direction",
                f"way {segment.id} traversed against its oneway",
            )


def _next_step(end, segments, remaining, tolerance):
    """The continuation from ``end``: relation order first, then the
    unique adjacency candidate."""
    ordered_first = remaining[0]
    step = _connect(end, segments[ordered_first], segments, remaining, tolerance)
    if step is not None:
        return (ordered_first, *step)
    candidates = []
    for index in remaining[1:]:
        attempt = _connect(end, segments[index], segments, remaining, tolerance)
        if attempt is not None:
            candidates.append((index, *attempt))
    if not candidates:
        return None
    if len(candidates) > 1:
        raise StitchRefusal("branching", f"{len(candidates)} continuation candidates")
    return candidates[0]


def _connect(end, segment, segments, remaining, tolerance):
    """``(part, new_end, reversed)`` when ``segment`` continues from
    ``end`` — ``reversed`` tells whether the part runs against the
    stored coordinates, with ``None`` marking a ring arc (its
    direction is verified by the ring rule)."""
    if segment.ring:
        return _connect_ring(end, segment, segments, remaining, tolerance)
    from_start = _meters(end, segment.coordinates[0])
    from_end = _meters(end, segment.coordinates[-1])
    if from_start > tolerance and from_end > tolerance:
        return None
    # Both ends inside tolerance happens on sub-tolerance stubs
    # (crossover track pieces): the closer end wins — a mis-orientation
    # there distorts by less than the tolerance itself.
    if from_start <= from_end:
        return segment.coordinates, segment.coordinates[-1], False
    reversed_part = list(reversed(segment.coordinates))
    return reversed_part, reversed_part[-1], True


def _connect_ring(end, segment, segments, remaining, tolerance):
    """The ring rule: verified direction, entry at the chain end, exit
    toward the next connectable neighbour, arc in stored order."""
    oneway = segment.tags.get("oneway")
    if oneway in ("no", "false", "0"):
        raise StitchRefusal("ring-direction", "explicitly bidirectional ring")
    direction = effective_direction(segment.tags)
    if direction == -1:
        # Legal reversed one-way: travel runs against stored order.
        ring = list(reversed(segment.coordinates))
    elif direction == 1:
        ring = segment.coordinates
    else:
        raise StitchRefusal(
            "ring-direction",
            (
                "no junction=roundabout/circular or recognised oneway"
                if oneway is None
                else f"unrecognised oneway {oneway!r}"
            ),
        )
    entry = _touch_vertex(ring, end, tolerance)
    if entry is None:
        return None
    exits = []
    for index in remaining:
        other = segments[index]
        if other is segment or other.ring:
            continue
        for point in (other.coordinates[0], other.coordinates[-1]):
            vertex = _touch_vertex(ring, point, tolerance)
            if vertex is None:
                continue
            if _meters(ring[vertex], ring[entry]) <= tolerance:
                continue  # the entry side, seen from a neighbour
            # Dense ring vertices within tolerance of each other are
            # ONE touch point, not several.
            for position, existing in enumerate(exits):
                if _meters(ring[vertex], ring[existing]) <= tolerance:
                    exits[position] = min(
                        existing, vertex, key=lambda v: _meters(ring[v], point)
                    )
                    break
            else:
                exits.append(vertex)
    if len(exits) != 1:
        raise StitchRefusal(
            "ring-touch", f"{len(exits)} exit touch points, need exactly 1"
        )
    exit_ = exits[0]
    arc = _arc(ring, entry, exit_)
    return arc, arc[-1], None


def _arc(ring, start, stop):
    """The ring arc from ``start`` to ``stop`` following stored-vertex
    order — the verified travel direction, never the reverse."""
    if start <= stop:
        return ring[start : stop + 1]
    return ring[start:] + ring[: stop + 1]


def _touch_vertex(ring, point, tolerance):
    """The single ring vertex ``point`` touches, or ``None``.

    All in-tolerance vertices cluster by ring adjacency (consecutive
    indices, wrap included, are one physical touch); contact with more
    than one separated arc — a self-near or malformed ring — refuses
    rather than picking an arbitrary arc."""
    hits = [
        position
        for position, vertex in enumerate(ring)
        if _meters(vertex, point) <= tolerance
    ]
    if not hits:
        return None
    clusters = [[hits[0]]]
    for position in hits[1:]:
        if position - clusters[-1][-1] == 1:
            clusters[-1].append(position)
        else:
            clusters.append([position])
    if len(clusters) > 1 and clusters[0][0] == 0 and clusters[-1][-1] == len(ring) - 1:
        clusters[0] = clusters.pop() + clusters[0]
    if len(clusters) > 1:
        raise StitchRefusal(
            "ring-touch", "an endpoint touches the ring in more than one place"
        )
    return min(clusters[0], key=lambda position: _meters(ring[position], point))


def _connectable(point, others, tolerance):
    for segment in others:
        if segment.ring:
            try:
                touched = _touch_vertex(segment.coordinates, point, tolerance)
            except StitchRefusal:
                touched = None
            if touched is not None:
                return True
        elif (
            _meters(point, segment.coordinates[0]) <= tolerance
            or _meters(point, segment.coordinates[-1]) <= tolerance
        ):
            return True
    return False


def _meters(a, b):
    """Equirectangular metres between two (lon, lat) pairs — exact
    enough at chaining tolerances."""
    scale = math.cos(math.radians((a[1] + b[1]) / 2))
    return math.hypot((a[0] - b[0]) * scale, a[1] - b[1]) * 111_320.0


def _merge(parts):
    """The chained coordinate list, joint duplicates dropped."""
    merged = []
    for coordinates in parts:
        for point in coordinates:
            if merged and _meters(merged[-1], point) <= 0.01:
                continue
            merged.append(point)
    if len(merged) < 2:
        raise StitchRefusal("empty", "degenerate geometry")
    return merged
