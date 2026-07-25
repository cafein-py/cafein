"""The standalone street network: build from OSM, route between coordinates."""

import os

import numpy as np
import shapely

from . import _osm, streets
from ._cafein import StreetNetwork as _CoreStreetNetwork

MODES = ("walk", "bicycle", "e_scooter")
"""The modes built by default — those with their own permission bit."""

STREET_MODES = ("walk", "bicycle", "e_bike", "e_scooter")
"""The modes that can be routed. `e_bike` has no permission bit of its own — it
rides the bicycle permissions with its own speed profile — so it is routable
without being a separate build mode."""

MAX_STREET_TIME = 7200.0
"""Default routing cutoff in seconds, matching the street-time ceiling the
transit path already applies to access and egress
(`streets.MAX_ACCESS_EGRESS_TIME`)."""


class StreetNetwork:
    """A routable street network built from an OpenStreetMap extract.

    Unlike `TransportNetwork`, this carries no timetable: it is the street
    graph alone, routable by any street mode. The graph is the *union* of
    every mode's ways — nothing is filtered out by mode — so which modes may
    use a given way is a per-arc permission rather than a separate graph.
    """

    def __init__(self, core):
        self._core = core

    @classmethod
    def from_osm(cls, osm_pbf, *, modes=MODES, bounding_box=None):
        """Build a street network from an OSM PBF extract.

        Parameters
        ----------
        osm_pbf : str or pathlib.Path
            Path to an ``.osm.pbf`` extract.
        modes : iterable of str
            The modes to prune connectivity for, from ``walk``, ``bicycle``,
            and ``e_scooter``. This selects pruning only — every physical edge
            is kept whatever is listed — so a mode left out can still be routed
            later, just without its small-component pruning.
        bounding_box : sequence of float, optional
            ``(min_lon, min_lat, max_lon, max_lat)`` to clip the extract.
        """
        modes = tuple(modes)
        nodes, edges = _osm.union_network(osm_pbf, bounding_box=bounding_box)
        if edges.empty:
            raise ValueError(f"no routable ways in '{osm_pbf}'")
        nodes = nodes.reset_index(drop=True)
        edges = edges.reset_index(drop=True)

        u, v = streets._vertex_endpoints(nodes, edges)
        highway, surface, smoothness, class_flags = _osm.normalise_codes(edges)
        access_forward, access_reverse, access_flags, _ = _osm.edge_permissions(edges)
        flags = class_flags | access_flags
        access_forward, access_reverse = _osm.prune_components_per_profile(
            u, v, len(nodes), access_forward, access_reverse, modes=modes
        )

        lengths = edges["length"].to_numpy(dtype=float)
        geometry = edges.geometry.to_numpy()
        offsets = np.concatenate(
            [[0], np.cumsum(shapely.get_num_coordinates(geometry))]
        )
        coordinates = shapely.get_coordinates(geometry)
        # `adj_facility` is reserved: no profile reads it yet (the compiler
        # routes on the access bits and the per-edge flags).
        facility = np.zeros(len(edges), dtype=np.uint8)
        return cls(
            _CoreStreetNetwork(
                len(nodes),
                list(zip(u.tolist(), v.tolist(), lengths.tolist())),
                offsets.tolist(),
                coordinates[:, 0].tolist(),
                coordinates[:, 1].tolist(),
                highway.tolist(),
                surface.tolist(),
                smoothness.tolist(),
                flags.tolist(),
                access_forward.tolist(),
                access_reverse.tolist(),
                facility.tolist(),
                facility.tolist(),
            )
        )

    def save(self, path):
        """Save the network as a reusable artifact.

        Carries the street graph, its geometry, and the multimodal permission
        and attribute arrays behind a versioned, checksummed header, so batch
        jobs can ``load`` the file instead of re-running the OSM extraction.
        """
        self._core.save(os.fspath(path))

    @classmethod
    def load(cls, path, *, mmap=False, verify=None):
        """Load a network saved with `save`.

        Parameters
        ----------
        path : str or pathlib.Path
            The artifact to load.
        mmap : bool or str
            ``False`` reads the arrays into memory; ``True`` maps the file and
            uses the street arrays in place, falling back to the owned load
            where mapping is unavailable; ``'require'`` errors instead of
            falling back.
        verify : bool, optional
            Whether to checksum the street section. Defaults to on for owned
            loads and off for mapped ones, where the check would page in the
            whole section the mapped load exists to avoid.
        """
        modes = {False: "off", True: "auto", "require": "require"}
        if mmap not in modes:
            raise ValueError(f"mmap must be False, True, or 'require', not {mmap!r}")
        return cls(_CoreStreetNetwork.load(os.fspath(path), modes[mmap], verify))

    @property
    def mapped(self):
        """Whether the street arrays are views of a memory-mapped artifact."""
        return self._core.mapped

    @property
    def vertex_count(self):
        """Number of vertices in the street graph."""
        return self._core.vertex_count

    @property
    def edge_count(self):
        """Number of physical edges in the street graph."""
        return self._core.edge_count

    def travel_time(
        self,
        origin,
        destination,
        *,
        mode,
        max_time=MAX_STREET_TIME,
        max_snap_distance=streets.MAX_SNAP_DISTANCE,
    ):
        """Travel time in whole seconds from `origin` to `destination`.

        Parameters
        ----------
        origin, destination : (float, float)
            ``(lat, lon)`` coordinates in EPSG:4326. A coordinate farther than
            `max_snap_distance` from the network raises ``ValueError``.
        mode : str
            ``walk``, ``bicycle``, ``e_bike``, or ``e_scooter``.
        max_time : float
            Cutoff in seconds; beyond it the destination counts as unreachable.
        max_snap_distance : float
            How far a coordinate may be from the network, in meters.

        Returns
        -------
        int or None
            Seconds, or ``None`` when the destination is not reachable within
            `max_time`.
        """
        return self._core.travel_time(
            tuple(origin),
            tuple(destination),
            mode,
            float(max_time),
            float(max_snap_distance),
        )

    def __repr__(self):
        return f"StreetNetwork({self.vertex_count} vertices, {self.edge_count} edges)"
