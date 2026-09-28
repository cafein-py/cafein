"""Walking structures from an OpenStreetMap street network.

The build turns a PBF extract into the two walking structures the routing
core consumes. Stop-to-stop footpaths become the transfer edge list: snap
every stop onto its nearest edge, hand the street graph and the snap
links to the routing core, and let it run its cutoff-bounded street
search from every stop's snap point, in parallel over stops. The
cutoff is the walking contract: a transfer between two rides is a
single street-shortest walk within `max_walking_time`, so the direct
bounded search is already complete for routing's one-hop-per-round
relaxation — any chain of footpaths whose total stays inside the
cutoff is a direct footpath by the triangle inequality. The street network itself is handed
over as flat arrays (edges with their geometry, plus the stops' snap
links) for the core's query-time access/egress searches from arbitrary
coordinates. Tiny disconnected components (mapping artifacts, clipped
boundary stubs) are pruned so a snap cannot get trapped on one; larger
disconnected components (genuine islands) stay, and stops snapped onto
different components simply get no footpath between them.
"""

import math
import warnings

from cafein import _log

import geopandas as gpd
import numpy as np
import pandas as pd
import pyrosm
import shapely
from scipy import sparse
from scipy.sparse import csgraph

WALKING_SPEED_KMPH = 3.6
"""Default walking speed, matching r5py's."""

MAX_WALKING_TIME = 1200.0
"""Default cutoff of the direct footpath (transfer) search, in seconds."""

MAX_ACCESS_EGRESS_TIME = 7200.0
"""Default cutoff of the query-time access/egress walk, in seconds.

Distinct from the footpath cutoff: it matches r5py's two-hour walking
cap for door-to-door access and egress, while precomputed transfers keep
the shorter ``MAX_WALKING_TIME`` cutoff."""

MAX_SNAP_DISTANCE = 1600.0
"""Default maximum distance from a stop to the walking network, in meters.

Matches R5's ``LINK_RADIUS_METERS``, the radius within which it links
stops and query points to the street layer (its 300 m constant is only
an initial fast-path search radius), so a stop up to this far from the
walking network still attaches to it over a straight connector."""

MIN_ISLAND_VERTICES = 40
"""Smallest disconnected walking-network component kept, in vertices.

Matches R5's ``MIN_SUBGRAPH_SIZE``. Smaller components are mapping
artifacts or stubs clipped at the extract boundary — snapping into one
traps the walk on a handful of edges — while genuinely walkable islands
are far larger."""


class Footpaths:
    """Transitively closed stop-to-stop walking transfers, as flat arrays.

    ``stop_ids`` names each snapped stop once; ``from_index`` and
    ``to_index`` are positions into it, one edge per element alongside
    its ``seconds`` and ``meters``. The arrays cross into the routing
    core whole — no per-edge Python objects — and iterating yields the
    ``(from_stop, to_stop, seconds, meters)`` tuples of the legacy
    edge-list form.
    """

    __slots__ = ("stop_ids", "from_index", "to_index", "seconds", "meters")

    def __init__(self, stop_ids, from_index, to_index, seconds, meters):
        self.stop_ids = list(stop_ids)
        self.from_index = _uint32_array(from_index, "from_index")
        self.to_index = _uint32_array(to_index, "to_index")
        self.seconds = _uint32_array(seconds, "seconds")
        meters = np.asarray(meters, dtype=np.float64)
        if meters.ndim != 1:
            raise ValueError("meters must be one-dimensional")
        self.meters = np.ascontiguousarray(meters)

    def __len__(self):
        return len(self.seconds)

    def __iter__(self):
        ids = np.asarray(self.stop_ids, dtype=object)
        return zip(
            ids[self.from_index].tolist(),
            ids[self.to_index].tolist(),
            self.seconds.tolist(),
            self.meters.tolist(),
        )


def _uint32_array(values, name):
    """`values` as a contiguous uint32 array, validated before the cast:
    numpy would silently wrap negative or oversized values into valid
    edge indexes or durations."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        return np.zeros(0, dtype=np.uint32)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be an integer array")
    if int(array.min()) < 0 or int(array.max()) > 4_294_967_295:
        raise ValueError(f"{name} values must fit an unsigned 32-bit integer")
    return np.ascontiguousarray(array, dtype=np.uint32)


def walking_footpaths(
    osm_pbf,
    stops,
    *,
    walking_speed_kmph=WALKING_SPEED_KMPH,
    max_walking_time=None,
    snap_distance=MAX_SNAP_DISTANCE,
    bounding_box=None,
):
    """Precompute stop-to-stop walking transfers from an OSM extract.

    Parameters
    ----------
    osm_pbf : str
        Path to an OpenStreetMap PBF extract covering the stops.
    stops : list of (str, float, float)
        ``(stop_id, latitude, longitude)`` triples, as produced by
        ``TransportNetwork.stops``. Stops without coordinates or farther
        than `snap_distance` from the walking network get no
        footpaths.
    walking_speed_kmph : float (optional, default: 3.6)
        Walking speed in km/h, on the network and on the stop connectors.
    max_walking_time : float or datetime.timedelta (optional, default: 20 minutes)
        Walking-time cutoff of the footpath search, in minutes: no
        stop-to-stop transfer exceeds it.
    snap_distance : float (optional, default: 1600)
        Maximum straight-line distance in meters from a stop to its
        nearest walking-network edge.
    bounding_box : sequence of float or shapely geometry (optional)
        Restrict the walking network to this area, as
        ``[min_lon, min_lat, max_lon, max_lat]`` or a shapely geometry;
        ways outside it are dropped, so a region-wide extract can be
        cropped to the stops' neighbourhood. Stops then snap only to the
        cropped network, so a stop with no edge within `snap_distance`
        of it gets no footpaths.

    Returns
    -------
    Footpaths
        The walking edges as flat arrays —
        conservatively rounded seconds plus the exact street-path
        length — suitable for ``TransportNetwork.set_transfers``;
        iterating yields the legacy ``(from_stop, to_stop, seconds,
        meters)`` tuples.
    """
    _log.sync()
    from cafein._units import duration_seconds
    from cafein._validate import (
        non_negative_finite,
        positive_finite,
        validated_bounding_box,
    )

    # The whole parameter set validates before the extract is read: a
    # bad knob fails in milliseconds, never after the PBF parse.
    walking_speed_kmph = positive_finite("walking_speed_kmph", walking_speed_kmph)
    max_walking_time = (
        MAX_WALKING_TIME
        if max_walking_time is None
        else duration_seconds("max_walking_time", max_walking_time)
    )
    max_snap_distance = non_negative_finite("snap_distance", snap_distance)
    bounding_box = validated_bounding_box(bounding_box)
    nodes, edges = _walking_network(osm_pbf, bounding_box)
    return _network_footpaths(
        stops,
        nodes,
        edges,
        walking_speed_kmph=walking_speed_kmph,
        max_walking_time=max_walking_time,
        max_snap_distance=max_snap_distance,
    )


def walking_streets(
    osm_pbf,
    stops,
    *,
    walking_speed_kmph=WALKING_SPEED_KMPH,
    max_walking_time=None,
    snap_distance=MAX_SNAP_DISTANCE,
    bounding_box=None,
):
    """Both walking structures of an OSM extract, from one load.

    Parameters are as in `walking_footpaths`, including `bounding_box`.

    Returns
    -------
    (footpaths, street_network)
        ``footpaths`` as from `walking_footpaths`, and
        ``street_network`` as the argument tuple of
        ``TransportNetwork.set_street_network``: ``(vertex_count, edges,
        coordinate_offsets, longitudes, latitudes, stop_links)``, with
        edges as ``(from, to, meters)`` vertex-index triples, geometry
        coordinates in EPSG:4326 flattened over the offsets, and stop
        links as ``(stop_id, edge, fraction, connector_meters)`` snap
        records.
    """
    _log.sync()
    from cafein._units import duration_seconds
    from cafein._validate import (
        non_negative_finite,
        positive_finite,
        validated_bounding_box,
    )

    # The whole parameter set validates before the extract is read: a
    # bad knob fails in milliseconds, never after the PBF parse.
    walking_speed_kmph = positive_finite("walking_speed_kmph", walking_speed_kmph)
    max_walking_time = (
        MAX_WALKING_TIME
        if max_walking_time is None
        else duration_seconds("max_walking_time", max_walking_time)
    )
    max_snap_distance = non_negative_finite("snap_distance", snap_distance)
    bounding_box = validated_bounding_box(bounding_box)
    nodes, edges = _walking_network(osm_pbf, bounding_box)
    return _network_streets(
        stops,
        nodes,
        edges,
        walking_speed_kmph=walking_speed_kmph,
        max_walking_time=max_walking_time,
        max_snap_distance=max_snap_distance,
    )


_UNWALKABLE_FILTER = {
    "area": ["yes"],
    "highway": [
        "abandoned",
        "construction",
        "motor",
        "motorway",
        "motorway_link",
        "proposed",
        "raceway",
    ],
    "foot": ["no"],
    "service": ["private"],
}
"""Ways pedestrians never walk on: motor-only or unbuilt roads, ways
mapped as areas, and ways that explicitly exclude pedestrians."""


def _walking_network(osm_pbf, bounding_box=None):
    """The walkable street network of a PBF extract, as (nodes, edges).

    Extracted with cafein's own walkability rule rather than pyrosm's
    ``walking`` network type, which drops every ``highway=cycleway`` —
    severing the shared foot-and-cycle paths that carry much of the
    pedestrian traffic in Nordic cities — and every transit platform,
    fragmenting the graph. The exclusion filter here is pyrosm's walking
    filter without those two types, so shared-use paths and platforms
    stay in (R5 applies the same permissive rule).
    """
    # The out-of-core engine streams the PBF instead of loading it into
    # memory, so country-scale extracts parse within bounded RAM.
    with _log.phase(
        "build.streets.read",
        _log.build,
        "reading the OSM walking network",
        "read the OSM walking network",
    ) as ph:
        osm = pyrosm.OSM(
            str(osm_pbf),
            bounding_box=bounding_box,
            engine="out_of_core",
            workers="auto",
        )
        network = osm.get_network(
            network_type="walking",
            custom_filter=_UNWALKABLE_FILTER,
            filter_type="exclude",
            nodes=True,
        )
        if network is None:
            raise ValueError(f"no walkable ways in '{osm_pbf}'")
        ph.note = f"{len(network[1]):,} edges"
        ph.details["edges"] = len(network[1])
    with _log.phase(
        "build.streets.prune",
        _log.build,
        "pruning disconnected street components",
        "pruned disconnected street components",
    ) as ph:
        nodes, edges = _prune_islands(*network)
        ph.note = f"{len(edges):,} edges kept"
        ph.details["edges"] = len(edges)
    if edges.empty:
        raise ValueError(f"no walkable ways in '{osm_pbf}'")
    return nodes, edges


def _prune_islands(nodes, edges):
    """The network without its sub-`MIN_ISLAND_VERTICES` components.

    A nearest-edge snap cannot tell a real street from a disconnected
    stub, so stubs must not exist to be snapped onto.
    """
    u, v = _vertex_endpoints(nodes, edges)
    graph = sparse.coo_matrix((np.ones(len(u)), (u, v)), shape=(len(nodes), len(nodes)))
    _, labels = csgraph.connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    edges = edges[sizes[labels[u]] >= MIN_ISLAND_VERTICES].reset_index(drop=True)
    nodes = nodes[sizes[labels] >= MIN_ISLAND_VERTICES].reset_index(drop=True)
    return nodes, edges


def _network_footpaths(
    stops,
    nodes,
    edges,
    *,
    walking_speed_kmph,
    max_walking_time,
    max_snap_distance,
):
    """The footpath build on an already loaded street network."""
    footpaths, _ = _network_streets(
        stops,
        nodes,
        edges,
        walking_speed_kmph=walking_speed_kmph,
        max_walking_time=max_walking_time,
        max_snap_distance=max_snap_distance,
    )
    return footpaths


def _network_streets(
    stops,
    nodes,
    edges,
    *,
    walking_speed_kmph,
    max_walking_time,
    max_snap_distance,
):
    """Footpaths and the street-network payload on a loaded network."""
    if not (math.isfinite(walking_speed_kmph) and walking_speed_kmph > 0):
        raise ValueError("walking_speed_kmph must be a positive, finite number")
    if not (math.isfinite(max_walking_time) and max_walking_time >= 0):
        raise ValueError("max_walking_time must be a non-negative, finite number")
    if not (math.isfinite(max_snap_distance) and max_snap_distance >= 0):
        raise ValueError("snap_distance must be a non-negative, finite number")
    if max_walking_time > 4_294_967_295:
        raise ValueError("max_walking_time overflows the router clock; narrow it")
    speed = walking_speed_kmph / 3.6  # m/s
    nodes = nodes.reset_index(drop=True)
    edges = edges.reset_index(drop=True)
    stop_points = _stop_points(stops)
    if edges.empty:
        return Footpaths([], [], [], [], []), (0, [], [0], [], [], [])
    if stop_points.empty:
        snapped = pd.DataFrame(columns=["stop_id", "edge", "fraction", "snap_distance"])
    else:
        snapped = _snap_to_edges(stop_points, edges, max_snap_distance)
    with _log.phase(
        "build.streets.graph",
        _log.build,
        "building the walking street graph",
        "built the walking street graph",
    ) as ph:
        payload = _street_payload(nodes, edges, snapped)
        ph.note = f"{len(edges):,} edges, {len(snapped):,} stops"
        ph.details.update(edges=len(edges), stops=len(snapped))
    footpaths = Footpaths([], [], [], [], [])
    if not snapped.empty:
        with _log.phase(
            "build.streets.footpaths",
            _log.build,
            "computing stop-to-stop footpaths",
            "computed stop-to-stop footpaths",
        ) as ph:
            footpaths = _street_footpaths(payload, snapped, speed, max_walking_time)
            ph.note = f"{len(footpaths.seconds):,} footpaths"
            ph.details["footpaths"] = len(footpaths.seconds)
    return footpaths, payload


def _stop_points(stops):
    """The stops that have coordinates, as a point GeoDataFrame."""
    frame = pd.DataFrame(stops, columns=["stop_id", "lat", "lon"])
    located = frame.dropna(subset=["lat", "lon"])
    if len(located) < len(frame):
        warnings.warn(
            f"{len(frame) - len(located)} stop(s) have no coordinates "
            "and get no footpaths",
            stacklevel=2,
        )
    return gpd.GeoDataFrame(
        located[["stop_id"]],
        geometry=gpd.points_from_xy(located["lon"], located["lat"]),
        crs="EPSG:4326",
    ).reset_index(drop=True)


def _snap_to_edges(stop_points, edges, max_snap_distance):
    """Each stop's nearest edge: row position, fraction along it, distance.

    Works in the extract's UTM CRS; the fraction is the snap point's
    linear-referenced position along the edge geometry.
    """
    crs = edges.estimate_utm_crs()
    edge_geometry = edges.geometry.to_crs(crs)
    matched = stop_points.to_crs(crs).sjoin_nearest(
        gpd.GeoDataFrame(geometry=edge_geometry),
        max_distance=max_snap_distance,
        distance_col="snap_distance",
    )
    # Several edges can tie as a stop's nearest; keep the closest match
    # and break exact ties by edge id so the choice is deterministic.
    matched = matched.sort_values(["snap_distance", "index_right"], kind="stable")
    matched = matched[~matched.index.duplicated()]
    matched = matched.sort_index(kind="stable")
    if len(matched) < len(stop_points):
        warnings.warn(
            f"{len(stop_points) - len(matched)} stop(s) are farther than "
            f"{max_snap_distance} m from the walking network and get no "
            "footpaths",
            stacklevel=2,
        )
    nearest = edge_geometry.to_numpy()[matched["index_right"].to_numpy()]
    along = shapely.line_locate_point(nearest, matched.geometry.to_numpy())
    length = shapely.length(nearest)
    return pd.DataFrame(
        {
            "stop_id": matched["stop_id"].to_numpy(),
            "edge": matched["index_right"].to_numpy(),
            "fraction": np.where(
                length > 0, along / np.where(length > 0, length, 1), 0
            ),
            "snap_distance": matched["snap_distance"].to_numpy(),
        }
    )


def _vertex_endpoints(nodes, edges):
    """Each edge's endpoints as vertex indices (node row positions)."""
    node_index = pd.Series(np.arange(len(nodes)), index=nodes["id"].to_numpy())
    u = node_index[edges["u"].to_numpy()].to_numpy()
    v = node_index[edges["v"].to_numpy()].to_numpy()
    return u, v


def _street_payload(nodes, edges, snapped):
    """The street network as the flat arrays the routing core consumes.

    Returns the argument tuple of ``TransportNetwork.set_street_network``:
    ``(vertex_count, edges, coordinate_offsets, longitudes, latitudes,
    stop_links)``.
    """
    u, v = _vertex_endpoints(nodes, edges)
    lengths = edges["length"].to_numpy(dtype=float)
    geometry = edges.geometry.to_numpy()
    offsets = np.concatenate([[0], np.cumsum(shapely.get_num_coordinates(geometry))])
    coordinates = shapely.get_coordinates(geometry)
    return (
        len(nodes),
        list(zip(u.tolist(), v.tolist(), lengths.tolist())),
        offsets.tolist(),
        coordinates[:, 0].tolist(),
        coordinates[:, 1].tolist(),
        list(
            zip(
                snapped["stop_id"].tolist(),
                snapped["edge"].tolist(),
                snapped["fraction"].tolist(),
                snapped["snap_distance"].tolist(),
            )
        ),
    )


def _street_footpaths(payload, snapped, speed, max_walking_time):
    """The footpaths of a street payload, from the core's bounded search.

    The core walks a cutoff-bounded search from every snapped stop's link
    over the payload's graph, in parallel over stops, and returns the
    within-cutoff walks as sparse edges; memory follows the graph and the
    footpaths kept, never the stop pairs. Durations are feasibility
    constraints, so the core rounds them up (with a small tolerance for
    floating-point noise): understating a walking time could let routing
    catch a departure the walk actually misses. Rounding up can carry a
    path past a fractional cutoff, so the kept edges are filtered on the
    rounded seconds — no stored transfer is longer than the cutoff. The
    meters stay exact: every walking cost is a street length over the
    uniform speed.
    """
    from cafein import _cafein

    vertex_count, edges, offsets, longitudes, latitudes, _ = payload
    # One link per snapped stop, in the row order that names `stop_ids`.
    links = list(
        zip(
            range(len(snapped)),
            snapped["edge"].tolist(),
            snapped["fraction"].tolist(),
            snapped["snap_distance"].tolist(),
        )
    )
    from_index, to_index, seconds, meters = _cafein.street_footpaths(
        vertex_count,
        edges,
        offsets,
        longitudes,
        latitudes,
        links,
        speed,
        max_walking_time,
    )
    within = seconds <= max_walking_time
    return Footpaths(
        snapped["stop_id"].tolist(),
        from_index[within],
        to_index[within],
        seconds[within],
        meters[within],
    )


def park_and_ride_facilities(osm_pbf, bounding_box=None):
    """Park-and-ride facilities from OSM, as ``CarParkPolicy`` input.

    A facility is any element carrying a ``park_ride`` tag with a
    value other than ``no`` — ``yes`` and the mode-specific values
    (``bus``, ``train``, ``tram``, ``metro``, ``ferry``, ``hov``)
    alike, standalone or beside ``amenity=parking``; plain parking
    without a qualifying ``park_ride`` value is not a facility. Area
    geometries collapse to ``representative_point()`` (a point
    guaranteed inside), so the frame is point-only by construction.

    Returns a GeoDataFrame in EPSG:4326 with ``id`` (the qualified
    OSM id, ``type/number`` — raw ids are unique per element type
    only),
    ``park_ride`` (the tag value), and point ``geometry`` — ready to
    pass to ``CarParkPolicy(facilities=...)``, which fills the
    ``search_seconds`` and ``fee`` defaults. Tagging completeness is
    the data's own quality judgment, which is why this is a separate,
    deliberate call.
    """
    from cafein._validate import validated_bounding_box

    bounding_box = validated_bounding_box(bounding_box)
    osm = pyrosm.OSM(str(osm_pbf), bounding_box=bounding_box)
    found = osm.get_data_by_custom_criteria(
        custom_filter={"park_ride": True},
        filter_type="keep",
        keep_nodes=True,
        keep_ways=True,
        keep_relations=True,
    )
    columns = ["id", "park_ride", "geometry"]
    if found is None or len(found) == 0:
        return gpd.GeoDataFrame(
            {"id": pd.Series(dtype=object), "park_ride": pd.Series(dtype=object)},
            geometry=gpd.GeoSeries(dtype=object),
            crs="EPSG:4326",
        )
    values = found["park_ride"].astype(str).str.lower()
    found = found[found["park_ride"].notna() & (values != "no")]
    points = found.geometry.representative_point()
    # OSM ids are unique per element type only; the qualified
    # "type/id" form keeps nodes, ways, and relations apart.
    if "osm_type" in found.columns:
        ids = (found["osm_type"].astype(str) + "/" + found["id"].astype(str)).to_numpy(
            dtype=object
        )
    else:
        ids = found["id"].to_numpy(dtype=object)
    frame = gpd.GeoDataFrame(
        {
            "id": ids,
            "park_ride": found["park_ride"].astype(str).to_numpy(dtype=object),
        },
        geometry=points.to_numpy(),
        crs=found.crs or "EPSG:4326",
    ).reset_index(drop=True)
    return frame.to_crs(epsg=4326)[columns]
