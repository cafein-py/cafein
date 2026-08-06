"""The OSM distance tiers — relation matching and map matching.

`OsmLadder` owns the per-build OSM state: relations, stitched lines,
canonical boarding positions, mode graphs, and per-pattern results
with their diagnostics. `resolve` runs tier 3 (a matched route
relation, validated with the same gates as a tier-2 shape) and falls
through to tier 4 (stop-to-stop shortest paths on the mode graph,
gated per segment by the detour bound; ferries have no graph and skip
it) — or returns ``None`` and the caller drops to crow-fly.
"""

import geopandas as gpd
import numpy as np
import pyproj
import shapely

from cafein import _map_match, _matching, _relations, _stitch
from cafein.geometry import MAP_MATCHED, OSM_RELATION, _locate_on_shape, _measures

#: Stitched-length plausibility band against the pattern's crow-fly
#: length — tier 1's band, applied to the linear-referenced total.
LENGTH_RATIO = (0.8, 5.0)

_UNSET = object()


class OsmLadder:
    """The OSM-tier context for one `trip_distances` call.

    Each pattern is projected in the UTM CRS estimated from its own
    stops (feeds and patterns need not share a zone); projected caches
    are keyed by CRS, so same-zone patterns share them.
    """

    def __init__(self, pbf_path, modes):
        self._pbf = str(pbf_path)
        self._modes = frozenset(modes)
        self._feed = 0
        self._relations = None
        self._by_mode = {}
        self._projections = {}
        self._transformer = None
        self._crs = None
        self._directed = {}
        self._canonical = {}
        self._lines = {}
        self._resolved = {}
        self._rail_ways = None
        self._street_network = _UNSET
        self._graphs = {}
        self._matched = 0
        self.diagnostics = {}

    def begin_feed(self, feed):
        """Namespace the per-pattern caches by feed — route and stop
        ids repeat freely between independent feeds."""
        self._feed = feed

    def resolve(self, route, route_type, stop_ids, latlon, crow_total):
        """Validated OSM-tier distances for one pattern, or ``None``.

        ``route`` is ``(route_id, short_name, long_name, agency)``;
        a hit returns ``(cumulative, identity, along, tier)`` with the
        stops' absolute positions along the tier's line in meters and
        ``tier`` the provenance (``osm_relation`` or ``map_matched``).
        """
        mode = _matching.mode_of(route_type)
        if mode is None or mode not in self._modes:
            return None
        key = (self._feed, route[0], tuple(stop_ids))
        if key not in self._resolved:
            result = self._resolve(route, mode, stop_ids, latlon, crow_total)
            if result is None:
                result = self._map_matched(mode, stop_ids, latlon)
            self._resolved[key] = result
        return self._resolved[key]

    def polyline(self, identity):
        """The ``(lons, lats, measures)`` payload of an accepted line;
        ``identity`` is the opaque key a resolution returned — a
        ``(crs, relation_id, reversed)`` triple for tier 3, a
        ``("match", n)`` pair for tier 4."""
        projected, lons, lats = self._lines[identity]
        return lons, lats, _measures(projected)

    def _resolve(self, route, mode, stop_ids, latlon, crow_total):
        if not self._set_projection(latlon):
            # No suitable projected CRS for this pattern's area (e.g.
            # near the poles): the pattern falls through.
            key = (self._feed, route[0], tuple(stop_ids))
            self.diagnostics[key] = [{"stage": "no-projection"}]
            return None
        pattern = _matching.Pattern(
            stop_ids=tuple(stop_ids),
            stop_xy=self._project(latlon[:, 1], latlon[:, 0]),
            short_name=route[1],
            long_name=route[2],
            agency=route[3],
        )
        entries = [
            (relation, self._canonical_xy(relation))
            for relation in self._mode_relations(mode)
        ]
        selection, diagnostics = _matching.select(pattern, entries)
        self.diagnostics[(self._feed, route[0], tuple(stop_ids))] = diagnostics
        if selection is None:
            return None
        if self._line(selection.relation, False) is None:
            return None
        # ``selection.reversed`` orders the boarding members; an open
        # way's stored coordinates are independent of that. A directed
        # chain is bound to its legal direction — a neutral one tries
        # the selected orientation, then the other (the monotonicity
        # gate accepts at most one).
        if self._directed[selection.relation.id]:
            orientations = () if selection.reversed else (False,)
        else:
            orientations = (selection.reversed, not selection.reversed)
        for reversed_ in orientations:
            result = self._validated(selection.relation, reversed_, latlon, crow_total)
            if result is not None:
                return result
        return None

    def _map_matched(self, mode, stop_ids, latlon):
        """Tier 4: consecutive-stop shortest paths on the mode graph.

        Every stop must snap to candidate graph vertices and every
        segment of the chosen chain stay inside the detour bound of
        its crow-fly; the accepted segment paths concatenate into the
        pattern geometry.
        """
        if not self._set_projection(latlon):
            return None
        graph = self._graph(mode)
        if graph is None:
            return None
        stop_xy = self._project(latlon[:, 1], latlon[:, 0])
        segments = _map_match.match_chain(graph, stop_xy)
        if segments is None:
            return None
        chained = graph.chain([path for _, path in segments])
        if chained is None:
            return None
        along = np.concatenate([[0.0], np.cumsum([length for length, _ in segments])])
        identity = ("match", self._matched)
        self._matched += 1
        self._lines[identity] = chained
        return along.tolist(), identity, along.tolist(), MAP_MATCHED

    def _graph(self, mode):
        """The mode's graph in the current CRS — ``None`` when the mode
        has no tier-4 graph.

        Ferries never have one (open water). Buses and trolleybuses
        are **structurally excluded for now**: the measured bus pass
        exceeds the plan's memory budget on the metro fixture
        (4.14 GiB added peak RSS against the 2 GiB gate, timing well
        inside its bound — ``scripts/measure_bus_tier4.py``), so bus
        patterns keep tier 3 and the ladder's fallthrough; the gate
        reruns with the calibrated detour bound in the validation
        sweep.
        """
        if mode not in _map_match.RAIL_VALUES:
            return None
        key = ("rail", _map_match.RAIL_VALUES[mode], self._crs)
        if key not in self._graphs:
            if self._rail_ways is None:
                self._rail_ways = _relations.rail_ways(self._pbf)
            self._graphs[key] = _map_match.rail_graph(
                self._rail_ways, self._project, key[1]
            )
        return self._graphs[key]

    def _streets(self):
        """The bus-candidate street network (nodes, edges), read once."""
        if self._street_network is _UNSET:
            import pyrosm

            from cafein import _osm

            osm = pyrosm.OSM(self._pbf, engine="out_of_core", workers="auto")
            self._street_network = osm.get_network(
                network_type="driving+service",
                custom_filter=_osm._UNBUSABLE_FILTER,
                filter_type="exclude",
                nodes=True,
                extra_attributes=[
                    "psv",
                    "bus",
                    "vehicle",
                    "motor_vehicle",
                    "oneway:bus",
                    "oneway:psv",
                ],
            )
        return self._street_network

    def _validated(self, relation, reversed_, latlon, crow_total):
        """The tier-2-style gates over one oriented line: density,
        stop snap, monotonicity, length plausibility."""
        stitched = self._line(relation, reversed_)
        if stitched is None:
            return None
        projected = stitched[0]
        if shapely.get_num_coordinates(projected) <= len(latlon):
            return None
        along = _locate_on_shape(projected, latlon, self._transformer)
        if along is None:
            return None
        total = along[-1] - along[0]
        if crow_total <= 0:
            return None
        if not LENGTH_RATIO[0] <= total / crow_total <= LENGTH_RATIO[1]:
            return None
        identity = (self._crs, relation.id, reversed_)
        return (along - along[0]).tolist(), identity, along.tolist(), OSM_RELATION

    def _mode_relations(self, mode):
        if self._relations is None:
            self._relations = _relations.route_relations(self._pbf)
        if mode not in self._by_mode:
            values = _matching.MODE_ROUTES[mode]
            self._by_mode[mode] = [
                relation for relation in self._relations if relation.route in values
            ]
        return self._by_mode[mode]

    def _canonical_xy(self, relation):
        key = (self._crs, relation.id)
        if key not in self._canonical:
            ordered = _matching.boarding_positions(relation)
            if not ordered:
                self._canonical[key] = None
            else:
                xy = self._project(
                    [entry[1] for entry in ordered],
                    [entry[2] for entry in ordered],
                )
                kinds = [entry[0] for entry in ordered]
                # The projection may have just been (re-)estimated:
                # re-key so same-zone lookups hit.
                key = (self._crs, relation.id)
                self._canonical[key] = _matching.collapse_positions(kinds, xy)
        return self._canonical[key]

    def _line(self, relation, reversed_=False):
        """The relation's oriented stitched line as ``(projected, lons,
        lats)``, ``None`` when stitching refused — cached either way."""
        key = (self._crs, relation.id, reversed_)
        if key in self._lines:
            return self._lines[key]
        if reversed_:
            forward = self._line(relation, False)
            if forward is None or self._directed[relation.id]:
                # One-way members or verified-direction rings: the
                # reverse traversal is not a legal path — refuse.
                self._lines[key] = None
            else:
                projected, lons, lats = forward
                self._lines[key] = (
                    shapely.reverse(projected),
                    lons[::-1],
                    lats[::-1],
                )
            return self._lines[key]
        ways = [
            member
            for member in relation.members
            if member.kind == "way"
            and not member.role.startswith("stop")
            and not member.role.startswith("platform")
        ]
        if any(member.role != "" for member in ways):
            # PTv1 forward/backward (or unknown) way roles make
            # membership direction-dependent: refuse rather than
            # stitch a chain with silent holes.
            self._lines[key] = None
            return None
        self._directed[relation.id] = any(
            _stitch.effective_direction(member.tags) != 0 for member in ways
        )
        try:
            line = _stitch.stitch(ways)
        except _stitch.StitchRefusal:
            self._lines[key] = None
        else:
            coordinates = shapely.get_coordinates(line)
            xy = self._project(coordinates[:, 0], coordinates[:, 1])
            # The projection may have just been (re-)estimated.
            key = (self._crs, relation.id, reversed_)
            self._lines[key] = (
                shapely.LineString(xy),
                coordinates[:, 0].tolist(),
                coordinates[:, 1].tolist(),
            )
        return self._lines[key]

    def _set_projection(self, latlon):
        """Bind the pattern's own UTM CRS (transformers cached per
        CRS); ``False`` when no UTM zone fits the pattern's area."""
        try:
            crs = gpd.GeoSeries(
                gpd.points_from_xy(latlon[:, 1], latlon[:, 0]), crs="EPSG:4326"
            ).estimate_utm_crs()
        except RuntimeError:
            return False
        key = crs.to_epsg() or crs.to_wkt()
        if key not in self._projections:
            self._projections[key] = pyproj.Transformer.from_crs(
                "EPSG:4326", crs, always_xy=True
            )
        self._crs = key
        self._transformer = self._projections[key]
        return True

    def _project(self, lons, lats):
        x, y = self._transformer.transform(np.asarray(lons), np.asarray(lats))
        return np.column_stack([x, y])
