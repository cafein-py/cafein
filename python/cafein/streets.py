"""Walking structures from an OpenStreetMap street network.

The build turns a PBF extract into the two walking structures the routing
core consumes. Stop-to-stop footpaths become the transfer edge list: snap
every stop onto its nearest edge (splitting the edge at the snap point),
run a cutoff-bounded one-to-many Dijkstra from every stop, and
transitively close the resulting stop-to-stop times — routing relaxes a
single transfer hop per round, so whenever two footpaths chain, the
chained pair must be a footpath too. The street network itself is handed
over as flat arrays (edges with their geometry, plus the stops' snap
links) for the core's query-time access/egress searches from arbitrary
coordinates. Disconnected walking-network components (islands, clipped
boundary fragments) stay in the graph; stops snapped onto different
components simply get no footpath between them.
"""

import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pyrosm
import shapely
from scipy import sparse
from scipy.sparse import csgraph

WALKING_SPEED_KMPH = 3.6
"""Default walking speed, matching r5py's."""

MAX_WALKING_TIME = 600.0
"""Default cutoff of the direct footpath search, in seconds."""

MAX_SNAP_DISTANCE = 100.0
"""Default maximum distance from a stop to the walking network, in meters."""

_DIJKSTRA_CHUNK = 256


def walking_footpaths(
    osm_pbf,
    stops,
    *,
    walking_speed_kmph=WALKING_SPEED_KMPH,
    max_walking_time=MAX_WALKING_TIME,
    max_snap_distance=MAX_SNAP_DISTANCE,
):
    """Precompute stop-to-stop walking transfers from an OSM extract.

    Parameters
    ----------
    osm_pbf : str
        Path to an OpenStreetMap PBF extract covering the stops.
    stops : list of (str, float, float)
        ``(stop_id, latitude, longitude)`` triples, as produced by
        ``TransportNetwork.stops``. Stops without coordinates or farther
        than `max_snap_distance` from the walking network get no
        footpaths.
    walking_speed_kmph : float (optional, default: 3.6)
        Walking speed in km/h, on the network and on the stop connectors.
    max_walking_time : float (optional, default: 600)
        Walking-time cutoff of the direct footpath search, in seconds.
        Transitive closure may produce chained footpaths that exceed it.
    max_snap_distance : float (optional, default: 100)
        Maximum straight-line distance in meters from a stop to its
        nearest walking-network edge.

    Returns
    -------
    list of (str, str, int)
        Transitively closed ``(from_stop, to_stop, seconds)`` walking
        edges, suitable for ``TransportNetwork.set_transfers``.
    """
    nodes, edges = _walking_network(osm_pbf)
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
    max_walking_time=MAX_WALKING_TIME,
    max_snap_distance=MAX_SNAP_DISTANCE,
):
    """Both walking structures of an OSM extract, from one load.

    Parameters are as in `walking_footpaths`.

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
    nodes, edges = _walking_network(osm_pbf)
    return _network_streets(
        stops,
        nodes,
        edges,
        walking_speed_kmph=walking_speed_kmph,
        max_walking_time=max_walking_time,
        max_snap_distance=max_snap_distance,
    )


def _walking_network(osm_pbf):
    """The walking network of a PBF extract, as pyrosm (nodes, edges)."""
    network = pyrosm.OSM(str(osm_pbf)).get_network(network_type="walking", nodes=True)
    if network is None:
        raise ValueError(f"no walkable ways in '{osm_pbf}'")
    return network


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
    if walking_speed_kmph <= 0:
        raise ValueError("walking_speed_kmph must be positive")
    speed = walking_speed_kmph / 3.6  # m/s
    nodes = nodes.reset_index(drop=True)
    edges = edges.reset_index(drop=True)
    stop_points = _stop_points(stops)
    if edges.empty:
        return [], (0, [], [0], [], [], [])
    if stop_points.empty:
        snapped = pd.DataFrame(columns=["stop_id", "edge", "fraction", "snap_distance"])
    else:
        snapped = _snap_to_edges(stop_points, edges, max_snap_distance)
    footpaths = []
    if not snapped.empty:
        graph, stop_vertices = _routing_graph(nodes, edges, snapped, speed)
        durations = _stop_durations(graph, stop_vertices, max_walking_time)
        closed = _transitive_closure(durations)
        footpaths = _edge_list(snapped["stop_id"].to_numpy(), closed)
    return footpaths, _street_payload(nodes, edges, snapped)


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
    matched = matched[~matched.index.duplicated()]
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


def _routing_graph(nodes, edges, snapped, speed):
    """The walking graph with snap points spliced in, plus stop vertices.

    Vertices are street nodes, then one vertex per distinct interior snap
    point, then one per snapped stop; weights are traversal seconds. A
    split edge's cost is redistributed over its segments proportionally to
    the fraction each segment covers; snap points landing on an endpoint
    reuse the endpoint vertex. Returns the graph and the stop vertices in
    `snapped` row order.
    """
    u, v = _vertex_endpoints(nodes, edges)
    seconds = edges["length"].to_numpy() / speed

    splits = (
        snapped[["edge", "fraction"]][
            (snapped["fraction"] > 0) & (snapped["fraction"] < 1)
        ]
        .drop_duplicates()
        .sort_values(["edge", "fraction"])
        .reset_index(drop=True)
    )
    splits["vertex"] = len(nodes) + np.arange(len(splits))
    edge_ids = splits["edge"].to_numpy()
    fractions = splits["fraction"].to_numpy()
    vertices = splits["vertex"].to_numpy()

    # Chain the split vertices along each edge: a segment from the edge
    # start or the previous snap point into each snap point, and a closing
    # segment from the last snap point to the edge end.
    boundary = edge_ids[1:] != edge_ids[:-1]
    first = np.r_[True, boundary] if len(splits) else np.zeros(0, dtype=bool)
    last = np.r_[boundary, True] if len(splits) else np.zeros(0, dtype=bool)
    previous_vertex = np.roll(vertices, 1)
    previous_fraction = np.roll(fractions, 1)
    into_from = np.where(first, u[edge_ids], previous_vertex)
    into_seconds = (fractions - np.where(first, 0, previous_fraction)) * seconds[
        edge_ids
    ]
    closing_seconds = (1 - fractions[last]) * seconds[edge_ids[last]]

    intact = np.ones(len(edges), dtype=bool)
    intact[edge_ids] = False

    stop_vertices = len(nodes) + len(splits) + np.arange(len(snapped))
    snap_vertex = _snap_vertices(snapped, splits, u, v)

    graph_from = np.concatenate([u[intact], into_from, vertices[last], stop_vertices])
    graph_to = np.concatenate([v[intact], vertices, v[edge_ids[last]], snap_vertex])
    weight = np.concatenate(
        [
            seconds[intact],
            into_seconds,
            closing_seconds,
            snapped["snap_distance"].to_numpy() / speed,
        ]
    )

    # Walking is undirected, so orient each edge low→high and keep the
    # cheapest of any parallel edges (duplicate COO entries would sum).
    unique = (
        pd.DataFrame(
            {
                "a": np.minimum(graph_from, graph_to),
                "b": np.maximum(graph_from, graph_to),
                "weight": weight,
            }
        )
        .groupby(["a", "b"], as_index=False)["weight"]
        .min()
    )
    size = len(nodes) + len(splits) + len(snapped)
    graph = sparse.coo_matrix(
        (unique["weight"], (unique["a"], unique["b"])), shape=(size, size)
    ).tocsr()
    return graph, stop_vertices


def _snap_vertices(snapped, splits, u, v):
    """Each snapped stop's vertex on the street graph."""
    merged = snapped.merge(splits, on=["edge", "fraction"], how="left")
    fraction = merged["fraction"].to_numpy()
    edge = merged["edge"].to_numpy()
    interior = merged["vertex"].fillna(-1).to_numpy(dtype=np.int64)
    return np.where(fraction == 0, u[edge], np.where(fraction == 1, v[edge], interior))


def _stop_durations(graph, stop_vertices, max_walking_time):
    """Stop-to-stop walking seconds within the cutoff (`inf` beyond)."""
    count = len(stop_vertices)
    durations = np.full((count, count), np.inf)
    for start in range(0, count, _DIJKSTRA_CHUNK):
        sources = stop_vertices[start : start + _DIJKSTRA_CHUNK]
        distances = csgraph.dijkstra(
            graph, directed=False, indices=sources, limit=max_walking_time
        )
        durations[start : start + len(sources)] = distances[:, stop_vertices]
    return durations


def _transitive_closure(durations):
    """All-pairs shortest paths over the footpath set itself.

    Whenever two footpaths chain, the chained pair becomes a footpath as
    well; direct footpaths are street shortest paths already, so closure
    never shortens them.
    """
    finite = np.isfinite(durations)
    np.fill_diagonal(finite, False)
    i, j = np.nonzero(finite)
    graph = sparse.coo_matrix((durations[i, j], (i, j)), shape=durations.shape).tocsr()
    return csgraph.dijkstra(graph, directed=False)


def _edge_list(stop_ids, durations):
    """The finite off-diagonal durations as `(from, to, seconds)` edges.

    Durations are feasibility constraints, so they round up (with a small
    tolerance for floating-point noise): understating a walking time could
    let routing catch a departure the walk actually misses.
    """
    finite = np.isfinite(durations)
    np.fill_diagonal(finite, False)
    i, j = np.nonzero(finite)
    seconds = np.ceil(durations[i, j] - 1e-6).astype(np.int64)
    return list(zip(stop_ids[i].tolist(), stop_ids[j].tolist(), seconds.tolist()))
