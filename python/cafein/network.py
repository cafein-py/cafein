"""The user-facing transport network."""

import os

from cafein._cafein import TransportNetwork as _TransportNetwork


def _gtfs_paths(paths):
    """`paths` as a list of strings; a single bare path is accepted too."""
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    return [os.fspath(path) for path in paths]


def _walk_options(walking_speed_kmph, max_walking_time, max_snap_distance):
    """Street-query options with the shared defaults filled in."""
    from cafein import streets

    if walking_speed_kmph is None:
        walking_speed_kmph = streets.WALKING_SPEED_KMPH
    if max_walking_time is None:
        max_walking_time = streets.MAX_WALKING_TIME
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
        max_walking_time : float (optional, default: 600)
            Walking-time cutoff of the direct footpath search, in
            seconds; chained footpaths may exceed it.
        max_snap_distance : float (optional, default: 100)
            Maximum distance in meters from a stop to the walking
            network; stops farther away get no footpaths.
        trip_distances : bool (optional, default: True)
            Compute per-trip travel distances through the fallback
            ladder (``cafein.geometry.trip_distances``), so transit legs
            report their distance and its provenance.

        The build reads the input files more than once (timetable,
        distance ladder, footpaths); they must not change underneath it.
        """
        paths = _gtfs_paths(paths)
        core = _TransportNetwork.from_gtfs(paths)
        if trip_distances:
            from cafein import geometry

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
        footpaths : list of (str, str, int)
            ``(from_stop, to_stop, seconds)`` walking edges. The list
            must be transitively closed — routing relaxes a single
            transfer hop per round; ``cafein.streets.walking_footpaths``
            produces such lists.
        """
        self._core.set_transfers(footpaths)

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
        max_walking_time : float (optional, default: 600)
            Walking-time cutoff in seconds.
        max_snap_distance : float (optional, default: 100)
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
        self, from_stop, to_stop, date, departure, max_transfers=4, window=None
    ):
        """Route between two transit stops.

        Journeys ride trips and change vehicles at shared stops or over
        the installed transfers; transit legs report their distance and
        its provenance when trip distances are installed.
        ``route_between_coordinates`` routes door-to-door from arbitrary
        coordinates, and ``annotate_emissions`` attaches emissions to
        routed journeys. Legs carry times, stops, distances, and
        provenance; per-leg geometries are not part of the output yet.

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
        max_transfers : int (optional, default: 4)
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
            from_stop, to_stop, date, departure, max_transfers, window
        )

    def route_between_coordinates(
        self,
        origin,
        destination,
        date,
        departure,
        max_transfers=4,
        window=None,
        *,
        walking_speed_kmph=None,
        max_walking_time=None,
        max_snap_distance=None,
    ):
        """Route door-to-door between two coordinates.

        Requires a network built with an OSM extract (``osm_pbf=``): the
        street network provides walking access from the origin to nearby
        stops and egress from stops to the destination. Journeys
        otherwise behave as in ``route_between_stops``; access and
        egress legs report their walking distance in meters. Journeys
        ride at least one trip: a destination best reached by walking
        alone yields no journeys.

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
        max_transfers : int (optional, default: 4)
            Maximum number of transfers between rides.
        window : int (optional)
            Departure window in seconds, as in ``route_between_stops``.
        walking_speed_kmph : float (optional, default: 3.6)
            Walking speed in km/h of the access and egress searches.
        max_walking_time : float (optional, default: 600)
            Walking-time cutoff in seconds of each street search.
        max_snap_distance : float (optional, default: 100)
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
        )

    def travel_times_from_coordinate(
        self,
        origin,
        date,
        departure,
        max_transfers=4,
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
        max_transfers : int (optional, default: 4)
            Maximum number of transfers between rides.
        walking_speed_kmph : float (optional, default: 3.6)
            Walking speed in km/h of the access search.
        max_walking_time : float (optional, default: 600)
            Walking-time cutoff in seconds of the access search.
        max_snap_distance : float (optional, default: 100)
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

    def travel_times_from_stop(self, from_stop, date, departure, max_transfers=4):
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
        max_transfers : int (optional, default: 4)
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
