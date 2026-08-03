"""The standalone street network: build from OSM, route between coordinates."""

import math
import os

import numpy as np
import shapely

from . import _osm, elevation, streets
from ._cafein import STREET_MAX_SEGMENT_METERS
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
    def from_osm(
        cls, osm_pbf, *, modes=MODES, bounding_box=None, dem=None, dem_interval=25.0
    ):
        """Build a street network from an OSM PBF extract.

        Parameters
        ----------
        osm_pbf : str or pathlib.Path
            Path to an ``.osm.pbf`` extract.
        modes : iterable of str
            The modes to prune connectivity for, from ``walk``, ``bicycle``,
            ``e_scooter``, and ``car``. Listing ``car`` also keeps the
            motor-only highway classes (motorways) in the extraction;
            otherwise the selection changes pruning only — every physical
            edge is kept whatever is listed, so a mode left out can still
            be routed later, just without its small-component pruning.
        bounding_box : sequence of float, optional
            ``(min_lon, min_lat, max_lon, max_lat)`` to clip the extract.
        dem : path, sequence of paths, or callable, optional
            An elevation source: a GeoTIFF path or tile paths sampled through
            the optional rioxarray dependency, or a callable
            ``(lons, lats) -> elevations``. Every geometry coordinate gets an
            elevation (NaN where the DEM has none); bridge and tunnel edges
            interpolate between their endpoints instead of tracking terrain,
            and ``elevation_metadata`` records the provenance. Without it the
            network carries no elevations, as before.
        dem_interval : float
            The along-edge sampling interval in meters (default 25): edge
            geometry is densified to it before sampling so the profile
            between OSM nodes is captured. Capped just under the stored
            geometry's own ~100 m segment limit; ``elevation_metadata``
            records the interval actually used.
        """
        return cls(
            _CoreStreetNetwork(
                *multimodal_payload(
                    osm_pbf,
                    modes=modes,
                    bounding_box=bounding_box,
                    dem=dem,
                    dem_interval=dem_interval,
                )
            )
        )

    @property
    def elevation_metadata(self):
        """Provenance of the sampled elevations, or ``None`` without a DEM.

        A dict of ``source``, ``sampling_interval`` (meters),
        ``nodata_policy``, ``coverage`` (the finite share of sampled
        coordinates), and ``inferred_edges`` (bridges and tunnels whose
        interior was endpoint-interpolated). Persisted with the artifact.
        """
        return self._core.elevation_metadata

    @property
    def _coordinate_elevations(self):
        """The stored per-coordinate elevations; internal, for the tests."""
        return self._core._coordinate_elevations

    @property
    def _coordinates(self):
        """The stored ``(longitude, latitude)`` degrees; internal."""
        return self._core._coordinates

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


def multimodal_payload(
    osm_pbf, *, modes=MODES, bounding_box=None, dem=None, dem_interval=25.0
):
    """The union extraction as the core constructor's argument tuple.

    Shared by ``StreetNetwork.from_osm`` and the ``TransportNetwork`` build,
    so both install the identical multimodal graph from the same inputs.
    """
    modes = tuple(modes)
    nodes, edges = _osm.union_network(
        osm_pbf, bounding_box=bounding_box, car="car" in modes
    )
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
    if dem is not None:
        if not (math.isfinite(dem_interval) and dem_interval > 0):
            raise ValueError("dem_interval must be a positive, finite number")
        # The Rust densifier splits stored segments under its own cap
        # regardless, so sampling coarser than that would let it insert
        # interpolated interiors after structure inference has already
        # counted; capping (with headroom) keeps every stored coordinate
        # in existence — and sampled — before inference runs.
        dem_interval = min(float(dem_interval), STREET_MAX_SEGMENT_METERS - 1.0)
        # Densify to the sampling interval before extracting coordinates,
        # so the elevation profile between OSM nodes is captured.
        # segmentize measures Euclidean degrees, and a degree of latitude
        # is the longest degree there is — 111,694 m at the poles — so
        # dividing by that ceiling bounds every direction: no segment
        # exceeds the interval, and east-west ones oversample (harmless).
        # Extracts are contiguous and never cross the antimeridian, per
        # the street network's documented contract.
        geometry = shapely.segmentize(geometry, dem_interval / 111_700.0)
    offsets = np.concatenate([[0], np.cumsum(shapely.get_num_coordinates(geometry))])
    coordinates = shapely.get_coordinates(geometry)

    elevations = None
    metadata = None
    if dem is not None:
        # Snapshot path inputs up front: they are consumed for sampling
        # and again for the source identifier, so a one-shot iterable or
        # a stateful PathLike must not make the two disagree.
        if not callable(dem):
            dem = (
                os.fspath(dem)
                if isinstance(dem, (str, os.PathLike))
                else tuple(os.fspath(tile) for tile in dem)
            )
        values = elevation.sample_dem(coordinates[:, 0], coordinates[:, 1], dem)
        # The promised finite share of DEM samples — taken before
        # structure inference rewrites bridge and tunnel interiors.
        sampled_coverage = elevation.coverage(values)
        structures = np.nonzero(
            (flags & (_osm.FLAG_BRIDGE | _osm.FLAG_TUNNEL)).astype(bool)
        )[0]
        inferred = elevation.infer_structures(
            offsets, coordinates[:, 0], coordinates[:, 1], values, structures
        )
        if callable(dem):
            source = "callable"
        elif isinstance(dem, str):
            source = dem
        else:
            source = ";".join(dem)
        elevations = values.tolist()
        metadata = (
            source,
            float(dem_interval),
            elevation.NODATA_POLICY,
            sampled_coverage,
            int(inferred),
        )
    # `adj_facility` is reserved: no profile reads it yet (the compiler
    # routes on the access bits and the per-edge flags).
    facility = np.zeros(len(edges), dtype=np.uint8)
    return (
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
        elevations,
        metadata,
    )
