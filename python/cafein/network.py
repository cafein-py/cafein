"""The user-facing transport network."""

import os

from cafein._cafein import TransportNetwork as _TransportNetwork


def _gtfs_paths(paths):
    """`paths` as a list of strings; a single bare path is accepted too."""
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    return [os.fspath(path) for path in paths]


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
            transfers; without it the network has no transfers.
        walking_speed_kmph : float (optional, default: 3.6)
            Walking speed in km/h of the footpath precompute.
        max_walking_time : float (optional, default: 600)
            Walking-time cutoff of the direct footpath search, in
            seconds; chained footpaths may exceed it.
        max_snap_distance : float (optional, default: 100)
            Maximum distance in meters from a stop to the walking
            network; stops farther away get no footpaths.
        """
        core = _TransportNetwork.from_gtfs(_gtfs_paths(paths))
        if osm_pbf is not None:
            from cafein import streets

            if walking_speed_kmph is None:
                walking_speed_kmph = streets.WALKING_SPEED_KMPH
            if max_walking_time is None:
                max_walking_time = streets.MAX_WALKING_TIME
            if max_snap_distance is None:
                max_snap_distance = streets.MAX_SNAP_DISTANCE
            core.set_transfers(
                streets.walking_footpaths(
                    osm_pbf,
                    core.stops,
                    walking_speed_kmph=walking_speed_kmph,
                    max_walking_time=max_walking_time,
                    max_snap_distance=max_snap_distance,
                )
            )
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

    def route_between_stops(
        self, from_stop, to_stop, date, departure, max_transfers=4, window=None
    ):
        """Route between two transit stops.

        Journeys ride trips and change vehicles at shared stops or over
        the installed transfers. Door-to-door access and egress from
        arbitrary coordinates, per-leg distance, geometry, and emissions
        arrive with later build steps.

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
