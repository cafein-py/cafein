"""Tier-3 resolution — matched OSM route relations as distance source.

`RelationLadder` owns the per-build OSM state: relations loaded once
from the extract, stitched lines and canonical boarding positions
cached per relation, and per-pattern match results with their
diagnostics. `resolve` returns validated cumulative distances for one
pattern or ``None`` — the caller falls through the ladder either way.
Every accepted geometry passed the same gates as a tier-2 shape:
denser than the stop sequence, every stop within snap tolerance,
monotone linear referencing, and total length inside the plausibility
band of the pattern's crow-fly length.
"""

import geopandas as gpd
import numpy as np
import pyproj
import shapely

from cafein import _matching, _relations, _stitch
from cafein.geometry import _locate_on_shape, _measures

#: Stitched-length plausibility band against the pattern's crow-fly
#: length — tier 1's band, applied to the linear-referenced total.
LENGTH_RATIO = (0.8, 5.0)


class RelationLadder:
    """The tier-3 context for one `trip_distances` call.

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
        self.diagnostics = {}

    def begin_feed(self, feed):
        """Namespace the per-pattern caches by feed — route and stop
        ids repeat freely between independent feeds."""
        self._feed = feed

    def resolve(self, route, route_type, stop_ids, latlon, crow_total):
        """Validated tier-3 distances for one pattern, or ``None``.

        ``route`` is ``(route_id, short_name, long_name, agency)``;
        a hit returns ``(cumulative, relation_id, along)`` with the
        stops' absolute positions along the stitched line in meters.
        """
        mode = _matching.mode_of(route_type)
        if mode is None or mode not in self._modes:
            return None
        key = (self._feed, route[0], tuple(stop_ids))
        if key not in self._resolved:
            self._resolved[key] = self._resolve(
                route, mode, stop_ids, latlon, crow_total
            )
        return self._resolved[key]

    def polyline(self, identity):
        """The ``(lons, lats, measures)`` payload of an accepted line;
        ``identity`` is the ``(crs, relation_id, reversed)`` key a
        resolution returned."""
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
        return (along - along[0]).tolist(), identity, along.tolist()

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
