"""Tier-4 map matching — mode graphs and stop-to-stop shortest paths.

A `ModeGraph` is a directed vertex/edge graph in projected meters with
a KD-tree for stop snapping and a per-pair path cache. Connectivity
rests on OSM node identity, never on coordinates — coincident but
distinct vertices (grade-separated tracks, a bridge over a street)
must not become junctions. Rail-family graphs build from the
node-identity rail-way extraction; the bus graph builds from a pyrosm
street network resolved through the PSV permission chain (endpoints
keyed by their ``u``/``v`` node ids, interior vertices edge-local),
with the graph split at every barrier node whose bus access is not an
explicit allow.

Shortest paths respect that Dijkstra is single-source: consecutive
stop pairs group by snapped source vertex, one bounded run per
distinct source serves all its targets (``limit`` pruned by the detour
bound), and pair results are cached across patterns. A segment whose
stops do not snap, whose target is unreachable within the bound, or
whose path exceeds the detour bound of its crow-fly fails — the
pattern falls through the ladder.
"""

import json

import numpy as np
import shapely
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

from cafein import _osm
from cafein.geometry import SNAP_TOLERANCE

#: Per-segment path length bound, as a multiple of the segment's
#: crow-fly length — provisional v1 constants, calibrated by the
#: validation sweep.
DETOUR_BOUNDS = {"rail": 1.5, "bus": 2.0}

#: The ``railway=`` values backing each rail-family mode.
RAIL_VALUES = {
    "tram": ("tram", "light_rail"),
    "subway": ("subway",),
    "train": ("rail",),
}

#: Mode families the bus street graph serves.
BUS_FAMILIES = ("bus", "trolleybus")

#: Candidate graph vertices considered per stop. Parallel per-direction
#: tracks a few meters apart make the single nearest vertex the wrong
#: one for roughly half the stops — the chain matcher picks among the
#: near candidates instead.
STOP_CANDIDATES = 4


class ModeGraph:
    """One directed mode graph in projected meters."""

    def __init__(self, xy, lonlat, u, v, lengths, kind):
        self.kind = kind
        self._xy = np.asarray(xy, dtype=float)
        self._lonlat = np.asarray(lonlat, dtype=float)
        count = len(self._xy)
        u = np.asarray(u, dtype=np.int64).reshape(-1)
        v = np.asarray(v, dtype=np.int64).reshape(-1)
        weights = np.asarray(lengths, dtype=float).reshape(-1)
        if len(weights):
            # Parallel edges keep the SHORTEST length (csr_matrix sums
            # duplicates, which would corrupt them).
            order = np.lexsort((weights, v, u))
            u, v, weights = u[order], v[order], weights[order]
            first = np.r_[True, (np.diff(u) != 0) | (np.diff(v) != 0)]
            u, v, weights = u[first], v[first], weights[first]
        self._matrix = sparse.csr_matrix((weights, (u, v)), shape=(count, count))
        self._tree = cKDTree(self._xy) if count else None
        self._paths = {}
        self._searched = {}

    def candidates(self, point_xy):
        """The nearest ``(vertex, meters)`` pairs within the stop-snap
        tolerance, closest first — empty when nothing is in range."""
        if self._tree is None:
            return []
        k = min(STOP_CANDIDATES, len(self._xy))
        distances, vertices = self._tree.query(point_xy, k=k)
        distances = np.atleast_1d(distances)
        vertices = np.atleast_1d(vertices)
        return [
            (int(vertex), float(distance))
            for distance, vertex in zip(distances, vertices)
            if distance <= SNAP_TOLERANCE
        ]

    def paths(self, pairs):
        """``{(u, v): (length, vertex_path) | None}`` for the pairs.

        ``pairs`` maps ``(source, target)`` to the segment's crow-fly
        length in meters. Uncached pairs run grouped by source, one
        bounded Dijkstra per distinct source; a pair is ``None`` when
        the target is unreachable within the detour bound or the path
        exceeds ``bound × crow``. Found paths cache forever; a miss
        remembers how far it searched, so a later pattern with a wider
        bound retries instead of inheriting the refusal.
        """
        bound = DETOUR_BOUNDS["bus" if self.kind == "bus" else "rail"]
        by_source = {}
        for (source, target), crow in pairs.items():
            ceiling = bound * max(crow, 1.0)
            if (source, target) in self._paths:
                continue
            if self._searched.get((source, target), 0.0) >= ceiling:
                continue  # already searched at least this far: no path
            by_source.setdefault(source, []).append((target, ceiling))
        for source, targets in by_source.items():
            limit = max(ceiling for _, ceiling in targets)
            distances, predecessors = csgraph.dijkstra(
                self._matrix,
                directed=True,
                indices=source,
                limit=limit,
                return_predecessors=True,
            )
            for target, _ in targets:
                length = float(distances[target])
                if not np.isfinite(length):
                    searched = self._searched.get((source, target), 0.0)
                    self._searched[(source, target)] = max(searched, limit)
                    continue
                path = [target]
                while path[-1] != source:
                    path.append(int(predecessors[path[-1]]))
                self._paths[(source, target)] = (length, path[::-1])
        results = {}
        for pair, crow in pairs.items():
            hit = self._paths.get(pair)
            if hit is None or hit[0] > bound * max(crow, 1.0):
                results[pair] = None
            else:
                results[pair] = hit
        return results

    def chain(self, vertex_paths):
        """The concatenated ``(projected_line, lons, lats)`` of segment
        vertex paths, joint duplicates dropped — ``None`` when the
        chain is degenerate (fewer than two vertices)."""
        vertices = []
        for path in vertex_paths:
            start = 1 if vertices and vertices[-1] == path[0] else 0
            vertices.extend(path[start:])
        if len(vertices) < 2:
            return None
        return (
            shapely.LineString(self._xy[vertices]),
            self._lonlat[vertices, 0].tolist(),
            self._lonlat[vertices, 1].tolist(),
        )


#: Snap-distance weight in the chain objective. Above 1, a chain that
#: collapses two stops onto a shared off-stop vertex (saving path
#: length) always loses to the true per-stop vertices.
SNAP_COST_WEIGHT = 2.0

#: Per-segment lower length-sanity factor: a path shorter than this
#: share of the segment's crow-fly is an under-measurement (a stub
#: inside both stops' snap tolerance faking a near-zero leg), not a
#: shortcut.
SEGMENT_LENGTH_FLOOR = 0.8


def match_chain(graph, stop_xy):
    """The pattern's consistent stop-to-stop segments, or ``None``.

    Each stop gets a set of candidate vertices; a dynamic program
    picks one vertex per stop minimising total path length plus
    `SNAP_COST_WEIGHT` × the snap distances, with every segment inside
    the detour bound of its own crow-fly AND above the length floor
    (`SEGMENT_LENGTH_FLOOR` × crow — a stub within both stops' snap
    tolerance must not fake a near-zero leg), and consecutive stops
    mapped to **distinct** vertices. ``None`` when any stop has no
    candidate, any consecutive stops coincide, or no consistent chain
    survives the gates.
    """
    bound = DETOUR_BOUNDS["bus" if graph.kind == "bus" else "rail"]
    candidate_sets = [graph.candidates(point) for point in stop_xy]
    if any(not candidates for candidates in candidate_sets):
        return None
    crows = np.hypot(*np.diff(np.asarray(stop_xy, dtype=float), axis=0).T)
    if (crows <= 1.0).any():
        # Coincident consecutive stops: degenerate data with no
        # meaningful segment gates — refuse the pattern.
        return None
    pairs = {}
    for i, crow in enumerate(crows):
        for source, _ in candidate_sets[i]:
            for target, _ in candidate_sets[i + 1]:
                if source == target:
                    continue
                pairs[(source, target)] = max(
                    pairs.get((source, target), 0.0), float(crow)
                )
    found = graph.paths(pairs)
    best = {vertex: SNAP_COST_WEIGHT * snap for vertex, snap in candidate_sets[0]}
    back = []
    for i, crow in enumerate(crows):
        ceiling = bound * float(crow)
        floor = SEGMENT_LENGTH_FLOOR * float(crow)
        nxt = {}
        choice = {}
        for target, snap in candidate_sets[i + 1]:
            for source, _ in candidate_sets[i]:
                if source not in best or source == target:
                    continue
                hit = found[(source, target)]
                if hit is None or hit[0] > ceiling or hit[0] < floor:
                    continue
                cost = best[source] + hit[0] + SNAP_COST_WEIGHT * snap
                if target not in nxt or cost < nxt[target]:
                    nxt[target] = cost
                    choice[target] = source
        if not nxt:
            return None
        back.append(choice)
        best = nxt
    vertex = min(best, key=best.get)
    chain = [vertex]
    for choice in reversed(back):
        vertex = choice[vertex]
        chain.append(vertex)
    chain.reverse()
    return [found[pair] for pair in zip(chain, chain[1:])]


def rail_graph(ways, project, values):
    """The directed rail graph of the extraction's matching ways.

    ``ways`` are `cafein._relations.rail_ways` tuples ``(node_refs,
    lons, lats, railway, oneway)``; ``project`` maps (lons, lats) to
    projected xy; ``values`` the ``railway=`` values to keep. Vertices
    join on **node identity**; ``oneway`` resolves through the
    stitcher's effective-direction rule (an unrecognised value keeps
    the way out entirely).
    """
    from cafein import _stitch

    index = {}
    lonlat = []
    segments = []
    for refs, lons, lats, railway, oneway in ways:
        if railway not in values:
            continue
        direction = _stitch.effective_direction(
            {"oneway": None if _missing(oneway) else oneway}
        )
        if direction is None:
            continue
        positions = []
        for ref, lon, lat in zip(refs, lons, lats):
            key = int(ref)
            if key not in index:
                index[key] = len(lonlat)
                lonlat.append((float(lon), float(lat)))
            positions.append(index[key])
        for a, b in zip(positions, positions[1:]):
            segments.append((a, b, direction))
    lonlat = np.asarray(lonlat, dtype=float).reshape(-1, 2)
    xy = project(lonlat[:, 0], lonlat[:, 1]) if len(lonlat) else lonlat
    u, v, lengths = [], [], []
    for a, b, direction in segments:
        length = float(np.hypot(*(xy[a] - xy[b])))
        if direction >= 0:
            u.append(a)
            v.append(b)
            lengths.append(length)
        if direction <= 0:
            u.append(b)
            v.append(a)
            lengths.append(length)
    return ModeGraph(xy, lonlat, u, v, lengths, "rail")


def bus_graph(nodes, edges, project):
    """The directed bus street graph of a pyrosm network.

    ``nodes``/``edges`` come from a pyrosm ``get_network`` read with
    the `cafein._osm._UNBUSABLE_FILTER` exclusion. Edges resolve
    through the PSV permission chain; an edge touching a blocking
    barrier node is dropped whole — over-blocking only costs a detour
    or a fallthrough, never a wrong legal path.
    """
    forward, reverse, _ = _osm.bus_permissions(edges)
    blocked = _blocked_coordinates(nodes)
    if blocked is None:
        # Barrier metadata was not preserved: fail closed — an empty
        # graph resolves nothing rather than routing through barriers.
        empty = np.empty((0, 2))
        return ModeGraph(empty, empty, [], [], [], "bus")
    spans = []
    index = {}
    lonlat = []

    def vertex(key, point):
        if key not in index:
            index[key] = len(lonlat)
            lonlat.append(point)
        return index[key]

    for i, row in enumerate(edges.itertuples()):
        if not (forward[i] or reverse[i]):
            continue
        coordinates = list(row.geometry.coords)
        if blocked and any(point in blocked for point in coordinates):
            continue
        # Junctions exist only at pyrosm's split points: endpoints key
        # by their OSM node id, interior vertices stay edge-local so
        # coincident geometry (grade separations) never joins.
        positions = [vertex(("n", row.u), coordinates[0])]
        for j, point in enumerate(coordinates[1:-1], start=1):
            positions.append(vertex(("e", i, j), point))
        positions.append(vertex(("n", row.v), coordinates[-1]))
        spans.append((np.asarray(positions), bool(forward[i]), bool(reverse[i])))
    lonlat = np.asarray(lonlat, dtype=float).reshape(-1, 2)
    xy = project(lonlat[:, 0], lonlat[:, 1]) if len(lonlat) else lonlat
    u, v, lengths = [], [], []
    for path, go_f, go_r in spans:
        hops = np.hypot(*(xy[path[1:]] - xy[path[:-1]]).T)
        for (a, b), length in zip(zip(path, path[1:]), hops):
            if go_f:
                u.append(int(a))
                v.append(int(b))
                lengths.append(float(length))
            if go_r:
                u.append(int(b))
                v.append(int(a))
                lengths.append(float(length))
    return ModeGraph(xy, lonlat, u, v, lengths, "bus")


def _blocked_coordinates(nodes):
    """The (lon, lat) pairs of barrier nodes that split the bus graph —
    ``None`` when the nodes carry no tag metadata at all (the caller
    fails closed)."""
    blocked = set()
    if nodes is None or "tags" not in nodes.columns:
        return None
    cells = nodes["tags"]
    lons = nodes["lon"].to_numpy(dtype=float)
    lats = nodes["lat"].to_numpy(dtype=float)
    for position, cell in enumerate(cells):
        if cell is None or (isinstance(cell, float) and np.isnan(cell)):
            continue
        if isinstance(cell, str):
            try:
                tags = json.loads(cell)
            except ValueError:
                continue
        elif isinstance(cell, dict):
            tags = cell
        else:
            continue
        barrier = tags.get("barrier")
        if barrier is None:
            continue
        if _osm.bus_barrier_blocks(barrier, tags):
            blocked.add((float(lons[position]), float(lats[position])))
    return blocked


def _missing(value):
    """pyrosm's absent-cell sentinels: None, NaN, and the string nan."""
    if value is None or value == "nan":
        return True
    return isinstance(value, float) and np.isnan(value)
