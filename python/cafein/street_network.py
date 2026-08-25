"""The standalone street network: build from OSM, route between coordinates."""

import math
import os

import numpy as np

from cafein import _log
import shapely

from . import _osm, elevation, streets
from ._validate import non_negative_finite
from ._cafein import STREET_MAX_SEGMENT_METERS
from ._cafein import StreetNetwork as _CoreStreetNetwork

MODES = ("walk", "bicycle", "e_scooter", "wheelchair")
"""The modes built by default — those with their own permission bit."""

STREET_MODES = ("walk", "bicycle", "e_bike", "e_scooter", "wheelchair", "car")
"""The modes that can be routed. `e_bike` has no permission bit of its own — it
rides the bicycle permissions with its own speed profile — so it is routable
without being a separate build mode. `car` requires a build with ``"car"`` in
`modes` (the persisted driving speeds and junction classes)."""


def validate_build_modes(modes):
    """`modes` as a validated tuple of build-mode names.

    Rejects a bare string outright — ``str`` is iterable, so
    ``"walk"`` would otherwise dissolve into four one-letter modes —
    and any name without a permission bit, before any file is read.
    """
    if isinstance(modes, str):
        raise TypeError(
            f"street modes must be an iterable of mode names, not the "
            f"string {modes!r} — pass ({modes!r},)"
        )
    modes = tuple(modes)
    seen = set()
    for mode in modes:
        if not isinstance(mode, str) or mode not in _osm.MODES:
            known = ", ".join(sorted(_osm.MODES))
            raise ValueError(
                f"unknown street mode {mode!r}: the build modes are {known}"
            )
        if mode in seen:
            raise ValueError(f"duplicate street mode {mode!r}")
        seen.add(mode)
    return modes


def validate_street_options(
    modes,
    *,
    dem=None,
    dem_interval=25.0,
    country=None,
    urban_areas=None,
    speed_limits=None,
):
    """The multimodal build options validated eagerly.

    Types, values, and the optional-dependency check, with no file
    reads or parsing — a DEM path existence stat is the one filesystem
    touch — so a bad option refuses in milliseconds, before any GTFS
    ingest or OSM extraction begins. Returns the validated snapshots
    ``(modes, dem, country, urban_areas, speed_limits)``: the mode
    tuple, the DEM with paths materialized once (a one-shot iterable
    of tiles survives), the normalised country code, the urban-areas
    snapshot (a copied frame or a materialized one-dimensional
    boolean mask), and the materialized speed-limit overrides — the
    build reads THESE, so a caller's mutable object cannot drift
    between validation and use. A
    malformed country code refuses here; an unknown-but-well-formed
    one keeps warning and falling back at build time."""
    from cafein._validate import positive_finite

    modes = validate_build_modes(modes)
    country = _osm.validated_country(country)
    if dem is not None:
        positive_finite("dem_interval", dem_interval)
        if callable(dem):
            # A callable provably unable to take (lons, lats) refuses
            # now; an uninspectable one (a ufunc, a builtin) passes —
            # the sample-time shape check remains its contract.
            import inspect

            try:
                signature = inspect.signature(dem)
            except (TypeError, ValueError):
                signature = None
            if signature is not None:
                try:
                    signature.bind(None, None)
                except TypeError:
                    raise TypeError(
                        "dem callable must accept two positional "
                        "arguments (lons, lats)"
                    ) from None
        if not callable(dem):
            if isinstance(dem, (str, os.PathLike)):
                dem = os.fspath(dem)
                paths = [dem]
            else:
                dem = tuple(os.fspath(tile) for tile in dem)
                paths = list(dem)
            if not paths:
                raise ValueError("dem names no raster tiles")
            for path in paths:
                if not os.path.exists(path):
                    raise ValueError(f"dem raster {path!r} does not exist")
            # The optional dependency refuses here, not after the
            # extraction and densification have run.
            try:
                import rioxarray  # noqa: F401
                import xarray  # noqa: F401
            except ImportError as error:
                raise ImportError(
                    "sampling a DEM needs the optional rioxarray "
                    "dependency; install cafein[dem] or pass an "
                    "elevation callable"
                ) from error
    if speed_limits is not None:
        speed_limits = _osm._validated_speed_limits(speed_limits)
    if urban_areas is not None:
        if hasattr(urban_areas, "geometry"):
            if getattr(urban_areas, "crs", None) is None:
                raise ValueError("urban_areas must carry a CRS")
            # The snapshot the build reads: a caller's frame cannot
            # drift during the long ingest, and the ACTIVE geometry
            # lands under the canonical column name whatever the
            # caller called it.
            import geopandas as gpd

            urban_areas = gpd.GeoDataFrame(
                geometry=urban_areas.geometry.copy().reset_index(drop=True),
                crs=urban_areas.crs,
            )
        else:
            try:
                mask = np.asarray(urban_areas, dtype=bool)
            except (TypeError, ValueError):
                raise ValueError(
                    "urban_areas must be a polygon GeoDataFrame or a "
                    "one-dimensional boolean mask"
                ) from None
            if mask.ndim != 1:
                raise ValueError(
                    "urban_areas must be a polygon GeoDataFrame or a "
                    "one-dimensional boolean mask"
                )
            # The materialized mask is the snapshot; its per-edge
            # length can only be checked after extraction.
            urban_areas = np.array(mask, copy=True)
    return modes, dem, country, urban_areas, speed_limits


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
        cls,
        osm_pbf,
        *,
        modes=MODES,
        bounding_box=None,
        dem=None,
        dem_interval=25.0,
        country=None,
        urban_areas=None,
        speed_limits=None,
    ):
        """Build a street network from an OSM PBF extract.

        Raises on a malformed `modes` before the extract is read.

        Parameters
        ----------
        osm_pbf : str or pathlib.Path
            Path to an ``.osm.pbf`` extract.
        modes : iterable of str
            The modes to prune connectivity for, from ``walk``, ``bicycle``,
            ``e_scooter``, ``wheelchair``, and ``car``. For the non-car modes the selection
            changes pruning only — every physical edge is kept whatever is
            listed, so a mode left out can still be routed later, just
            without its small-component pruning. ``car`` is the exception:
            listing it keeps the motor-only highway classes (motorways) in
            the extraction and computes the persisted driving speeds and
            junction classes, so car routing requires a build with ``car``
            in `modes`.
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
        country : str, optional
            ISO 3166-1 alpha-2 code (or 3166-2 subdivision, e.g.
            ``"US-CA"``) selecting the legal default speed limits that fill
            ways with no ``maxspeed`` tag. A subdivision falls back to its
            country, an unknown or omitted code to the generic row, each
            fallback with a warning. Car builds only.
        urban_areas : geopandas.GeoDataFrame or array of bool, optional
            Where the urban (inside built-up area) speed defaults apply: a
            polygon layer resolved per edge by intersection, or a
            precomputed per-edge boolean. Omitted, every way counts as
            urban — the conservative default for city extracts. Car builds
            only.
        speed_limits : mapping, optional
            Per-class km/h overrides layered over the resolved country row
            (e.g. ``{"residential_inside": 30}``); unknown classes and
            non-positive values are rejected. Car builds only.
        """
        from cafein._validate import validated_bounding_box

        _log.sync()
        bounding_box = validated_bounding_box(bounding_box)
        # One snapshot serves the build and the phase details alike, so
        # a one-shot iterable cannot be consumed twice; a non-iterable
        # falls through to the validator's own refusal.
        if isinstance(modes, str):
            modes = (modes,)
        else:
            try:
                modes = tuple(modes)
            except TypeError:
                pass
        with _log.phase(
            "build.multimodal",
            _log.build,
            "building the multimodal street graph",
            "built the multimodal street graph",
        ) as ph:
            network = cls(
                _CoreStreetNetwork(
                    *multimodal_payload(
                        osm_pbf,
                        modes=modes,
                        bounding_box=bounding_box,
                        dem=dem,
                        dem_interval=dem_interval,
                        country=country,
                        urban_areas=urban_areas,
                        speed_limits=speed_limits,
                    )
                )
            )
            resolved = list(modes)
            ph.note = ", ".join(resolved)
            ph.details["modes"] = resolved
        return network

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
        _log.sync()
        target = os.fspath(path)
        with _log.phase(
            "artifact.save",
            _log.artifact,
            "saving the street artifact",
            "saved the street artifact",
        ) as ph:
            ph.details["path"] = target
            self._core.save(target)

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
        _log.sync()
        source = os.fspath(path)
        with _log.phase(
            "artifact.load",
            _log.artifact,
            "loading the street artifact",
            "loaded the street artifact",
        ) as ph:
            ph.details["path"] = source
            core = _CoreStreetNetwork.load(source, modes[mmap], verify)
        return cls(core)

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

    @property
    def streets_gdf(self):
        """The street network as a GeoDataFrame of edge lines.

        One row per graph edge, in the graph's edge order — the same
        index space the searches and exposure reporting speak: the
        edge's real polyline (EPSG:4326), ``length_m``, the OSM
        ``highway`` class, and one column per street mode (``walk``,
        ``bicycle``, ``e_scooter``, ``car``, ``wheelchair``) holding
        the edge's permission as ``"both"``, ``"forward"``,
        ``"reverse"``, or ``"no"`` (forward is the stored geometry
        direction). Computed lazily on first access and cached on the
        network.
        """
        if getattr(self, "_streets_gdf_cache", None) is None:
            self._streets_gdf_cache = _edge_layer_frame(self._core._street_edge_layer())
        return self._streets_gdf_cache

    def travel_time(
        self,
        origin,
        destination,
        *,
        mode,
        max_travel_time=None,
        snap_distance=streets.MAX_SNAP_DISTANCE,
        intersection_delays=False,
        profile=None,
        delay_model=None,
        parking=None,
    ):
        """Travel time in whole seconds from `origin` to `destination`.

        Parameters
        ----------
        origin, destination : (float, float)
            ``(lat, lon)`` coordinates in EPSG:4326. A coordinate farther than
            `snap_distance` from the network raises ``ValueError``.
        mode : str
            ``walk``, ``bicycle``, ``e_bike``, ``e_scooter``,
            ``wheelchair``, or ``car``.
        max_travel_time : float or datetime.timedelta (optional)
            Cutoff in minutes; beyond it the destination counts as
            unreachable. Unset uses the shipped street default (120
            minutes).
        snap_distance : float
            How far a coordinate may be from the network, in meters.
        intersection_delays : bool
            Car only. ``False`` (the default) computes free-flow,
            speed-limit-based times; ``True`` applies the empirical
            intersection-delay model (Jaakkola 2013, the calibration behind
            GEMMAT) under the selected `profile`.
        profile : str, optional
            The delay period — ``"rush"``, ``"midday"`` (the default), or
            ``"day-average"``. Requires ``intersection_delays=True``.
        delay_model : mapping, optional
            Partial override of the shipped delay values (keys ``values``,
            ``groups``, ``ramp_multipliers``, ``congestion_multipliers``,
            ``ramp_shares``), merged over them entry by entry. Requires
            ``intersection_delays=True``.
        parking : optional
            Car only, off by default. The parking search ending a car trip,
            costed as time (and, in the richer forms, extra driving
            metres): ``True`` → the shipped constant (300 s, 0 m), a
            number → that many seconds, a ``(seconds, metres)`` pair, or a
            polygon GeoDataFrame with a ``seconds`` column (optional
            ``metres``) resolved per destination by point-in-polygon — a
            destination inside several polygons takes the largest seconds
            (ties: largest metres, then lowest row), outside every polygon
            the shipped constant. Parking adds after the driving search:
            `max_travel_time` bounds the driving alone.

        Returns
        -------
        int or None
            Seconds, or ``None`` when the destination is not reachable within
            `max_travel_time`. Returned times are seconds.
        """
        from cafein._units import duration_seconds

        max_time = (
            MAX_STREET_TIME
            if max_travel_time is None
            else duration_seconds("max_travel_time", max_travel_time)
        )
        max_snap_distance = snap_distance
        from . import _parking

        resolved = _parking.resolve(parking, mode)
        # Materialized once: the routed pair and the parking lookup must see
        # the same coordinates even if a mutable input changes mid-call.
        origin, destination = tuple(origin), tuple(destination)
        seconds = self._core.travel_time(
            origin,
            destination,
            mode,
            float(max_time),
            non_negative_finite("snap_distance", max_snap_distance),
            car_model=_resolved_delays(mode, intersection_delays, profile, delay_model),
        )
        if seconds is None or resolved is None:
            return seconds
        extra, _ = _parking.destination_costs(resolved, [destination])
        return int(seconds + round(float(extra[0])))

    def __repr__(self):
        return f"StreetNetwork({self.vertex_count} vertices, {self.edge_count} edges)"


def _resolved_delays(mode, intersection_delays, profile, delay_model):
    """The core's ``car_model`` payload for a query, or ``None``.

    The delay options belong to the car: any of them set under another mode
    raises rather than being silently ignored.
    """
    from . import _delays

    if mode != "car" and (
        intersection_delays or profile is not None or delay_model is not None
    ):
        raise ValueError(
            "intersection_delays, profile, and delay_model apply to mode='car'"
        )
    return _delays.resolve(intersection_delays, profile, delay_model)


def _edge_layer_frame(rows):
    """One GeoDataFrame row per graph edge from a core edge layer, in
    the graph's edge order: geometry, ``length_m``, and — when the
    graph carries attributes — the highway class and per-mode
    permission columns."""
    import geopandas
    from shapely.geometry import LineString

    from cafein._osm import (
        BICYCLE,
        CAR,
        E_SCOOTER,
        HIGHWAY_CODES,
        WALK,
        WHEELCHAIR,
    )

    data = {"length_m": [meters for _, _, meters, _, _ in rows]}
    if rows and rows[0][3] is not None:
        highway_names = {code: name for name, code in HIGHWAY_CODES.items()}
        data["highway"] = [highway_names[code] for _, _, _, code, _ in rows]
        modes = (
            ("walk", WALK),
            ("bicycle", BICYCLE),
            ("e_scooter", E_SCOOTER),
            ("car", CAR),
            ("wheelchair", WHEELCHAIR),
        )
        directions = {
            (True, True): "both",
            (True, False): "forward",
            (False, True): "reverse",
            (False, False): "no",
        }
        for mode, bit in modes:
            data[mode] = [
                directions[(bool(forward & bit), bool(reverse & bit))]
                for _, _, _, _, (forward, reverse) in rows
            ]
    geometry = [LineString(list(zip(lons, lats))) for lons, lats, _, _, _ in rows]
    return geopandas.GeoDataFrame(data, geometry=geometry, crs="EPSG:4326")


def multimodal_payload(
    osm_pbf,
    *,
    modes=MODES,
    bounding_box=None,
    dem=None,
    dem_interval=25.0,
    country=None,
    urban_areas=None,
    speed_limits=None,
):
    """The union extraction as the core constructor's argument tuple.

    Shared by ``StreetNetwork.from_osm`` and the ``TransportNetwork`` build,
    so both install the identical multimodal graph from the same inputs.
    """
    modes = validate_build_modes(modes)
    if "car" not in modes and any(
        option is not None for option in (country, urban_areas, speed_limits)
    ):
        raise ValueError(
            "country, urban_areas, and speed_limits configure car speeds; "
            "add 'car' to modes to use them"
        )
    modes, dem, country, urban_areas, speed_limits = validate_street_options(
        modes,
        dem=dem,
        dem_interval=dem_interval,
        country=country,
        urban_areas=urban_areas,
        speed_limits=speed_limits,
    )
    with _log.phase(
        "build.multimodal.streets",
        _log.build,
        "extracting the multimodal street union",
        "extracted the multimodal street union",
    ) as ph:
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
        ph.note = f"{len(edges):,} edges"
        ph.details.update(edges=len(edges), modes=list(modes))

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
        # the street network's documented contract. OSM legally carries
        # zero-length ways (consecutive nodes mapped at one position),
        # which GEOS refuses to densify: only edges with real length are
        # segmentized, and degenerate ones keep their stored coordinates.
        measurable = shapely.length(geometry) > 0.0
        geometry = geometry.copy()
        geometry[measurable] = shapely.segmentize(
            geometry[measurable], dem_interval / 111_700.0
        )
    offsets = np.concatenate([[0], np.cumsum(shapely.get_num_coordinates(geometry))])
    coordinates = shapely.get_coordinates(geometry)

    elevations = None
    metadata = None
    if dem is not None:
        with _log.phase(
            "build.multimodal.elevation",
            _log.build,
            "applying the elevation model",
            "applied the elevation model",
        ) as ph:
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
            ph.details["coverage"] = sampled_coverage
            ph.note = f"{sampled_coverage:.0%} DEM coverage"
    car_attributes = None
    if "car" in modes:
        with _log.phase(
            "build.multimodal.speeds",
            _log.build,
            "attributing car speeds and junction delays",
            "attributed car speeds and junction delays",
        ) as ph:
            # Speeds and junction classes read the pruned permissions: the
            # drivable graph the routing engine will actually see.
            speed_forward, speed_reverse = _osm.car_speeds(
                edges, country=country, urban=urban_areas, speed_limits=speed_limits
            )
            junction_forward, junction_reverse = _osm.junction_delay_classes(
                _osm.node_delay_tags(nodes),
                u,
                v,
                edges["id"].to_numpy(),
                _osm._column(edges, "highway"),
                access_forward,
                access_reverse,
                len(nodes),
            )
            car_attributes = (
                speed_forward.tolist(),
                speed_reverse.tolist(),
                junction_forward.tolist(),
                junction_reverse.tolist(),
            )
            ph.details["edges"] = len(edges)
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
        car_attributes,
    )
