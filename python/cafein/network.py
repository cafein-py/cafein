"""The user-facing transport network."""

import os

from cafein._cafein import TransportNetwork as _TransportNetwork


def _gtfs_paths(paths):
    """`paths` as a list of strings; a single bare path is accepted too."""
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    return [os.fspath(path) for path in paths]


def _window_percentiles(window, percentiles, confidence):
    """The percentile list a window/percentiles/confidence spec asks
    for; ``None`` without a window."""
    if window is None:
        if percentiles is not None or confidence is not None:
            raise ValueError("percentiles and confidence require a window")
        return None
    if percentiles is not None and confidence is not None:
        raise ValueError("pass either percentiles or confidence, not both")
    if confidence is not None:
        if not 0 < confidence < 1:
            raise ValueError("confidence must be within (0, 1)")
        # Rounded so the derived bounds equal their explicit decimal
        # forms; raw float arithmetic (e.g. (1 - 0.9) / 2 * 100 =
        # 4.999999999999999) could otherwise flip a half-up rank tie.
        half = round((1 - confidence) / 2 * 100, 9)
        return [half, 50.0, round(100 - half, 9)]
    if percentiles is None:
        return [50.0]
    return [float(percentile) for percentile in percentiles]


def _walk_options(walking_speed_kmph, max_walking_time, max_snap_distance):
    """Street-query options with the shared defaults filled in."""
    from cafein import streets

    if walking_speed_kmph is None:
        walking_speed_kmph = streets.WALKING_SPEED_KMPH
    if max_walking_time is None:
        max_walking_time = streets.MAX_ACCESS_EGRESS_TIME
    if max_snap_distance is None:
        max_snap_distance = streets.MAX_SNAP_DISTANCE
    return walking_speed_kmph, max_walking_time, max_snap_distance


class TransportNetwork:
    """A routable public-transport network.

    Built from GTFS timetables and, optionally, an OpenStreetMap extract
    whose walking network provides the stop-to-stop footpath transfers.
    """

    def __init__(self, core):
        self._core = core

    @classmethod
    def from_gtfs(
        cls,
        paths,
        *,
        osm_pbf=None,
        walking_speed_kmph=None,
        max_walking_time=None,
        max_snap_distance=None,
        trip_distances=True,
        leg_geometries=True,
    ):
        """Build a network from GTFS archives and an optional OSM extract.

        Parameters
        ----------
        paths : path or list of paths
            GTFS zip files or directories, as strings or path-likes; a
            single feed may be given bare. Several feeds are merged; a
            stop_id occurring in more than one feed must then be
            qualified as ``<feed_index>:<stop_id>``, with feeds numbered
            in input order.
        osm_pbf : str (optional)
            Path to an OpenStreetMap PBF extract covering the stops. Its
            walking network is turned into stop-to-stop footpaths (see
            ``cafein.streets.walking_footpaths``) that routing uses as
            transfers, and installed as the street network behind
            coordinate-based access/egress searches (``access_stops``);
            without it the network has neither.
        walking_speed_kmph : float (optional, default: 3.6)
            Walking speed in km/h of the footpath precompute.
        max_walking_time : float (optional, default: 1200)
            Walking-time cutoff of the direct footpath search, in
            seconds; chained footpaths may exceed it.
        max_snap_distance : float (optional, default: 1600)
            Maximum distance in meters from a stop to the walking
            network; stops farther away get no footpaths.
        trip_distances : bool (optional, default: True)
            Compute per-trip travel distances through the fallback
            ladder (``cafein.geometry.trip_distances``), so transit legs
            report their distance and its provenance.
        leg_geometries : bool (optional, default: True)
            Also store the trips' polylines, so transit legs report
            their geometry; disable to save memory when geometries are
            never needed. Ignored when `trip_distances` is off.

        The build reads the input files more than once (timetable,
        distance ladder, footpaths); they must not change underneath it.
        """
        paths = _gtfs_paths(paths)
        core = _TransportNetwork.from_gtfs(paths)
        if trip_distances:
            from cafein import geometry

            if leg_geometries:
                distances, polylines = geometry.trip_distances(
                    paths, include=set(core.trip_ids), geometries=True
                )
                core.set_trip_distances(distances)
                core.set_leg_geometries(*polylines)
            else:
                core.set_trip_distances(
                    geometry.trip_distances(paths, include=set(core.trip_ids))
                )
        if osm_pbf is not None:
            from cafein import streets

            if walking_speed_kmph is None:
                walking_speed_kmph = streets.WALKING_SPEED_KMPH
            if max_walking_time is None:
                max_walking_time = streets.MAX_WALKING_TIME
            if max_snap_distance is None:
                max_snap_distance = streets.MAX_SNAP_DISTANCE
            footpaths, street_network = streets.walking_streets(
                osm_pbf,
                core.stops,
                walking_speed_kmph=walking_speed_kmph,
                max_walking_time=max_walking_time,
                max_snap_distance=max_snap_distance,
            )
            core.set_transfers(footpaths)
            core.set_street_network(*street_network)
        return cls(core)

    def save(self, path):
        """Save the network as a reusable artifact.

        The artifact carries everything queries need — the timetable,
        service calendar, transfers, trip distances, leg geometries,
        and the street network — so batch jobs can ``load`` the same
        file read-only instead of rebuilding from GTFS and OSM inputs.
        Build diagnostics (quarantine warnings) are not persisted.
        The file is staged beside the destination and atomically
        renamed into place, so saving over an existing artifact never
        rewrites it under live memory-mapped readers.

        Parameters
        ----------
        path : path
            Destination file, conventionally ``*.cafein``.
        """
        self._core.save(os.fspath(path))

    @classmethod
    def load(cls, path, *, mmap=False, verify=None):
        """Load a network saved with `save`.

        Artifacts written in another format version are refused with
        a message naming the writing cafein version, and corrupted
        payloads fail their checksum; rebuild from the inputs (or
        re-save) with a matching version instead. Artifacts are
        trusted input, like pickles: load only files you created.

        With ``mmap=True`` the street arrays are used directly from a
        read-only memory map of the file instead of being copied into
        memory: the operating system pages street data in as queries
        touch it and shares those pages between every process mapping
        the same artifact, so per-process memory scales with the region
        a job actually walks, not with the network. A mapped artifact
        must stay unchanged while any process maps it — replace it by
        writing a new file and renaming it over the old one, never by
        editing in place, and keep it out of cloud-synced folders
        (OneDrive and its kin rewrite files in place).

        Parameters
        ----------
        path : path
            An artifact written by `save`.
        mmap : bool or "require"
            ``False`` (default) loads everything into memory. ``True``
            maps the file, falling back to the in-memory load where
            mapping is unavailable; ``"require"`` raises instead of
            falling back.
        verify : bool, optional
            Whether to checksum the street data. Defaults to ``True``
            for in-memory loads (the bytes are read anyway) and
            ``False`` for mapped loads, where the check would page the
            whole street section in and defeat the lazy load.
        """
        modes = {False: "off", True: "auto", "require": "require"}
        if mmap not in modes:
            raise ValueError(f"mmap must be False, True, or 'require', not {mmap!r}")
        return cls(
            _TransportNetwork.load(os.fspath(path), mmap=modes[mmap], verify=verify)
        )

    @property
    def mapped(self):
        """Whether the street arrays are memory-mapped from the artifact."""
        return self._core.mapped

    @property
    def stop_count(self):
        """Number of stops in the network."""
        return self._core.stop_count

    @property
    def pattern_count(self):
        """Number of stop-sequence patterns in the network."""
        return self._core.pattern_count

    @property
    def trip_count(self):
        """Number of trips in the network."""
        return self._core.trip_count

    @property
    def transfer_count(self):
        """Number of installed stop-to-stop transfers."""
        return self._core.transfer_count

    @property
    def stops(self):
        """The stops as ``(stop_id, latitude, longitude)`` tuples."""
        return self._core.stops

    @property
    def routes(self):
        """The routes as ``(route_id, agency_id, route_type)`` tuples,
        with the GTFS route_type as its numeric code."""
        return self._core.routes

    @property
    def trips(self):
        """The routable trips as ``(trip_id, route_id)`` tuples."""
        return self._core.trips

    def annotate_emissions(self, journeys, factors=None, components=None):
        """Attach per-leg and per-journey emissions to routed journeys.

        Parameters
        ----------
        journeys : list of dict
            Journeys from ``route_between_stops`` (with distances, the
            default build).
        factors : DataFrame or path (optional)
            Extra emission-factor rows layered over the shipped
            defaults; see ``cafein.emissions.load_factors``.
        components : list of str (optional)
            The life-cycle components to include (default: all four);
            see ``cafein.emissions.annotate``.

        Returns
        -------
        list of dict
            The journeys, with ``emissions`` (grams CO₂e) on every leg
            and journey; see ``cafein.emissions.annotate``.
        """
        from cafein import emissions

        return emissions.annotate(journeys, self, factors, components)

    def set_transfers(self, footpaths):
        """Install precomputed stop-to-stop transfers.

        Parameters
        ----------
        footpaths : list of (str, str, int, float)
            ``(from_stop, to_stop, seconds, meters)`` walking edges.
            The list must be transitively closed — routing relaxes a
            single transfer hop per round;
            ``cafein.streets.walking_footpaths`` produces such lists.
        """
        self._core.set_transfers(footpaths)

    def set_leg_geometries(self, *leg_geometries):
        """Install per-trip leg geometries.

        Parameters
        ----------
        leg_geometries : tuple
            ``(polylines, trips)`` — deduplicated ``(longitudes,
            latitudes, measures)`` polylines and ``(trip_id, polyline,
            stop_positions)`` rows locating each stop of a trip along
            its polyline — as produced (alongside the distances) by
            ``cafein.geometry.trip_distances(..., geometries=True)``.
        """
        self._core.set_leg_geometries(*leg_geometries)

    def set_street_network(self, *street_network):
        """Install the street network for coordinate access/egress.

        Parameters
        ----------
        street_network : tuple
            ``(vertex_count, edges, coordinate_offsets, longitudes,
            latitudes, stop_links)``, as produced (alongside the
            footpaths) by ``cafein.streets.walking_streets``.
        """
        self._core.set_street_network(*street_network)

    def access_stops(
        self,
        lat,
        lon,
        *,
        walking_speed_kmph=None,
        max_walking_time=None,
        max_snap_distance=None,
    ):
        """Walking times to every transit stop reachable from a coordinate.

        Requires a network built with an OSM extract (``osm_pbf=``).
        Walking is undirected, so the same search serves access from an
        origin and egress to a destination.

        Parameters
        ----------
        lat, lon : float
            The coordinate, in EPSG:4326.
        walking_speed_kmph : float (optional, default: 3.6)
            Walking speed in km/h, on the network and on the connectors.
        max_walking_time : float (optional, default: 7200)
            Walking-time cutoff in seconds.
        max_snap_distance : float (optional, default: 1600)
            Maximum straight-line distance in meters from the coordinate
            to the walking network; a coordinate farther away raises
            ``ValueError``.

        Returns
        -------
        dict
            Walking time in seconds to each reachable stop, keyed by
            stop_id; stops beyond the cutoff are absent.
        """
        return self._core.access_stops(
            lat,
            lon,
            *_walk_options(walking_speed_kmph, max_walking_time, max_snap_distance),
        )

    def set_trip_distances(self, distances):
        """Install per-trip cumulative travel distances.

        Parameters
        ----------
        distances : list of (str, list of float, str)
            ``(trip_id, cumulative_meters, provenance)`` rows with one
            cumulative distance per stop of the trip;
            ``cafein.geometry.trip_distances`` produces such lists.
        """
        self._core.set_trip_distances(distances)

    @property
    def distance_provenance_counts(self):
        """Number of trips per distance-provenance tier (empty until
        trip distances are installed)."""
        return self._core.distance_provenance_counts

    def route_between_stops(
        self,
        from_stop,
        to_stop,
        date,
        departure,
        max_transfers=7,
        window=None,
        *,
        geometries=True,
    ):
        """Route between two transit stops.

        Journeys ride trips and change vehicles at shared stops or over
        the installed transfers; transit legs report their distance and
        its provenance when trip distances are installed.
        ``route_between_coordinates`` routes door-to-door from arbitrary
        coordinates, and ``annotate_emissions`` attaches emissions to
        routed journeys. Legs carry times, stops, distances, and
        provenance; transit legs add their geometry as a WKB LineString
        when leg geometries are installed (the default build), and
        transfer legs their walked street path when the street network
        is installed.

        Parameters
        ----------
        from_stop : str
            GTFS stop_id of the origin stop; ``<feed_index>:<stop_id>``
            when the id occurs in several merged feeds.
        to_stop : str
            GTFS stop_id of the destination stop, qualified the same way.
            Identifiers in the output follow the same convention.
        date : str
            Service date as ``YYYY-MM-DD``.
        departure : str
            Departure time at the origin as ``HH:MM:SS``.
        max_transfers : int (optional, default: 7)
            Maximum number of transfers between rides.
        window : int (optional)
            Departure window in seconds. When given, departures within
            ``[departure, departure + window)`` are profiled: the result
            is the Pareto set of journeys over (departure, arrival,
            rides), each journey's departure being the latest time the
            origin can be left to catch it, sorted by departure and then
            rides. A journey that leaves within the window but waits for
            a ride beyond it carries the window's final second as its
            departure.

        Returns
        -------
        list of dict
            Without `window`, the Pareto set of journeys over (arrival
            time, number of rides) leaving at the departure time; with
            it, the departure-window profile. Each journey carries its
            legs; times are seconds past the service day's start.
        """
        return self._core.route_between_stops(
            from_stop, to_stop, date, departure, max_transfers, window, geometries
        )

    def route_between_coordinates(
        self,
        origin,
        destination,
        date,
        departure,
        max_transfers=7,
        window=None,
        *,
        walking_speed_kmph=None,
        max_walking_time=None,
        max_snap_distance=None,
        geometries=True,
    ):
        """Route door-to-door between two coordinates.

        Requires a network built with an OSM extract (``osm_pbf=``): the
        street network provides walking access from the origin to nearby
        stops and egress from stops to the destination. Journeys
        otherwise behave as in ``route_between_stops``; access and
        egress legs report their walking distance in meters and — with
        `geometries`, the default — their walked street path as WKB
        LineStrings. Walking all the way is a journey too: within
        `max_walking_time` the result leads with a walking-only journey
        — a single ``walk`` leg, zero rides — and a journey is dropped
        when walking out at that journey's own departure would arrive
        no later than it does.

        Parameters
        ----------
        origin, destination : (float, float)
            ``(lat, lon)`` coordinates, in EPSG:4326. A coordinate
            farther than `max_snap_distance` from the walking network
            raises ``ValueError``.
        date : str
            Service date as ``YYYY-MM-DD``.
        departure : str
            Departure time at the origin coordinate as ``HH:MM:SS``.
        max_transfers : int (optional, default: 7)
            Maximum number of transfers between rides.
        window : int (optional)
            Departure window in seconds, as in ``route_between_stops``.
        walking_speed_kmph : float (optional, default: 3.6)
            Walking speed in km/h of the access and egress searches.
        max_walking_time : float (optional, default: 7200)
            Walking-time cutoff in seconds of each street search.
        max_snap_distance : float (optional, default: 1600)
            Maximum straight-line distance in meters from each
            coordinate to the walking network.

        Returns
        -------
        list of dict
            Journeys as in ``route_between_stops``; arrivals include
            the egress walk.
        """
        return self._core.route_between_coordinates(
            tuple(origin),
            tuple(destination),
            date,
            departure,
            max_transfers,
            window,
            *_walk_options(walking_speed_kmph, max_walking_time, max_snap_distance),
            geometries,
        )

    def travel_times_from_coordinate(
        self,
        origin,
        date,
        departure,
        max_transfers=7,
        *,
        walking_speed_kmph=None,
        max_walking_time=None,
        max_snap_distance=None,
    ):
        """Earliest arrival at every reachable stop from a coordinate.

        The counterpart of ``travel_times_from_stop`` for a coordinate
        origin: walking access from the coordinate seeds one RAPTOR run
        that serves all destinations. Requires a network built with an
        OSM extract (``osm_pbf=``). Stops within the walking cutoff
        appear with their walking time even without riding.

        Parameters
        ----------
        origin : (float, float)
            ``(lat, lon)`` coordinate, in EPSG:4326. A coordinate
            farther than `max_snap_distance` from the walking network
            raises ``ValueError``.
        date : str
            Service date as ``YYYY-MM-DD``.
        departure : str
            Departure time at the origin coordinate as ``HH:MM:SS``.
        max_transfers : int (optional, default: 7)
            Maximum number of transfers between rides.
        walking_speed_kmph : float (optional, default: 3.6)
            Walking speed in km/h of the access search.
        max_walking_time : float (optional, default: 7200)
            Walking-time cutoff in seconds of the access search.
        max_snap_distance : float (optional, default: 1600)
            Maximum straight-line distance in meters from the
            coordinate to the walking network.

        Returns
        -------
        dict
            Travel time in seconds to every reachable stop, keyed by
            stop_id; unreachable stops are absent.
        """
        return self._core.travel_times_from_coordinate(
            tuple(origin),
            date,
            departure,
            max_transfers,
            *_walk_options(walking_speed_kmph, max_walking_time, max_snap_distance),
        )

    def travel_times_from_stop(self, from_stop, date, departure, max_transfers=7):
        """Earliest arrival at every reachable stop for a single departure.

        One RAPTOR run serves all destinations, so travel-time matrices
        are assembled origin by origin from this method — never per OD
        pair.

        Parameters
        ----------
        from_stop : str
            GTFS stop_id of the origin stop; ``<feed_index>:<stop_id>``
            when the id occurs in several merged feeds.
        date : str
            Service date as ``YYYY-MM-DD``.
        departure : str
            Departure time at the origin as ``HH:MM:SS``.
        max_transfers : int (optional, default: 7)
            Maximum number of transfers between rides.

        Returns
        -------
        dict
            Travel time in seconds to every reachable stop, keyed by
            stop_id; the origin maps to 0 and unreachable stops are
            absent.
        """
        return self._core.travel_times_from_stop(
            from_stop, date, departure, max_transfers
        )

    def travel_time_matrix(
        self,
        from_stops,
        date,
        departure,
        max_transfers=7,
        *,
        destinations=None,
        window=None,
        percentiles=None,
        confidence=None,
        chunk=None,
        router="raptor",
        walking_speed_kmph=None,
        max_walking_time=None,
        max_snap_distance=None,
    ):
        """Travel times as a matrix, from stops or from points.

        One RAPTOR run serves each origin, computed in parallel across
        the origins with per-worker state reuse; the result is
        deterministic. This is the bulk primitive travel-time matrices
        are assembled from — never per OD pair.

        With `window`, every minute mark within ``[departure,
        departure + window)`` is evaluated through one descending range
        scan per origin, and the output holds nearest-rank percentiles
        of the travel-time distribution across the window — exact
        values, since the samples are the full minute-level departure
        population. `percentiles` selects them (default: the median);
        `confidence` instead maps a level to the symmetric interval
        plus the median (e.g. ``0.8`` → the 10th, 50th, and 90th
        percentiles), quantifying travel-time variability due to
        departure time within the window.

        Parameters
        ----------
        from_stops : list of str, or GeoDataFrame
            GTFS stop_ids of the origin stops
            (``<feed_index>:<stop_id>`` when an id occurs in several
            merged feeds), or a point GeoDataFrame with an ``id``
            column. Points are linked once against the street network
            (requires ``osm_pbf=`` at build time); points off the
            walking network are reported with a warning and stay
            unreachable. Point cells hold the faster of transit and
            walking directly (within ``max_walking_time``), so a pair
            best covered on foot reports its walking time.
        date : str
            Service date as ``YYYY-MM-DD``.
        departure : str
            Departure time at every origin as ``HH:MM:SS``.
        max_transfers : int (optional, default: 7)
            Maximum number of transfers between rides.
        destinations : GeoDataFrame (optional)
            Destination points; defaults to the origins. Only valid
            with point origins — stop origins always span every stop.
        window : int (optional)
            Departure window in seconds; enables percentile output.
        percentiles : list of float (optional)
            Percentiles in ``[0, 100]`` over the window's departures;
            requires `window`, defaults to ``[50]``.
        confidence : float (optional)
            A level in ``(0, 1)`` mapped to the symmetric percentile
            interval plus the median; requires `window` and excludes
            `percentiles`.
        chunk : (int, int) (optional)
            Compute only origin chunk ``k`` of ``n``: a deterministic
            contiguous block of the resolved origins, so ``n`` batch
            jobs cover all origins disjointly; rows follow the chunk.
        router : str (optional, default: "raptor")
            The routing engine for single-departure stop matrices:
            ``"raptor"``, or ``"tbtr"`` to precompute a TBTR day engine
            (Trip-Based Transit Routing: Witt's trip-transfer set) for
            the date and fan the origins out over it. The results are
            identical. Windowed and point matrices run on RAPTOR only,
            and networks with installed footpaths are rejected — the
            transitively closed footpath set is too dense for the TBTR
            precompute as yet.
        walking_speed_kmph, max_walking_time, max_snap_distance : float
            The street-search options for points, as in
            ``access_stops``; only valid with point origins.

        Returns
        -------
        numpy.ndarray
            A uint32 array of travel times in seconds — origins by all
            stops (column order follows ``stops``) for stop origins,
            origins by destination points for point origins; with
            `window`, one plane per percentile as a third axis, in the
            requested order (lower, median, upper for `confidence`).
            Unreachable pairs hold the maximum uint32 value
            (4294967295).
        """
        matrix, _from_ids, _to_ids, _percentiles = self._time_matrix_with_ids(
            from_stops,
            date,
            departure,
            max_transfers,
            destinations=destinations,
            window=window,
            percentiles=percentiles,
            confidence=confidence,
            chunk=chunk,
            router=router,
            walking_speed_kmph=walking_speed_kmph,
            max_walking_time=max_walking_time,
            max_snap_distance=max_snap_distance,
        )
        return matrix

    def _time_matrix_with_ids(
        self,
        from_stops,
        date,
        departure,
        max_transfers,
        *,
        destinations,
        window,
        percentiles,
        confidence,
        chunk,
        walking_speed_kmph,
        max_walking_time,
        max_snap_distance,
        router="raptor",
    ):
        """The travel-time matrix with its origin and destination id
        axes and the resolved percentile list (``None`` without a
        window). Backs both ``travel_time_matrix`` and the
        ``TravelTimeMatrix`` long-format wrapper, so the two share one
        origin/destination resolution.
        """
        from cafein.matrices import (
            _chunk_slice,
            _is_point_frame,
            _point_list,
            _warn_unsnapped,
        )

        if router not in ("raptor", "tbtr"):
            raise ValueError(f"router must be 'raptor' or 'tbtr', not {router!r}")
        if router == "tbtr" and (window is not None or _is_point_frame(from_stops)):
            raise ValueError(
                "router='tbtr' backs single-departure stop matrices only; "
                "windowed and point matrices run on RAPTOR"
            )
        percentiles = _window_percentiles(window, percentiles, confidence)
        if _is_point_frame(from_stops):
            from_ids, origin_points = _point_list(from_stops, "origins")
            if destinations is None:
                to_ids, destination_points = from_ids, origin_points
            else:
                to_ids, destination_points = _point_list(destinations, "destinations")
            rows = _chunk_slice(len(from_ids), chunk)
            from_ids = from_ids[rows]
            origin_points = origin_points[rows]
            walk = _walk_options(
                walking_speed_kmph, max_walking_time, max_snap_distance
            )
            if percentiles is None:
                table = self._core.travel_time_matrix_from_points(
                    origin_points,
                    destination_points,
                    date,
                    departure,
                    max_transfers,
                    *walk,
                )
            else:
                table = self._core.travel_time_percentiles_from_points(
                    origin_points,
                    destination_points,
                    date,
                    departure,
                    window,
                    percentiles,
                    max_transfers,
                    *walk,
                )
            _warn_unsnapped(table, from_ids, to_ids)
            return table["matrix"], from_ids, to_ids, percentiles
        if not (
            destinations is None
            and walking_speed_kmph is None
            and max_walking_time is None
            and max_snap_distance is None
        ):
            raise ValueError("destinations and walking options apply to point origins")
        to_ids = [stop for stop, _latitude, _longitude in self._core.stops]
        from_stops = list(to_ids) if from_stops is None else list(from_stops)
        from_stops = from_stops[_chunk_slice(len(from_stops), chunk)]
        if percentiles is None:
            matrix = self._core.travel_time_matrix(
                from_stops, date, departure, max_transfers, router
            )
        else:
            matrix = self._core.travel_time_percentiles(
                from_stops, date, departure, window, percentiles, max_transfers
            )
        return matrix, from_stops, to_ids, percentiles
