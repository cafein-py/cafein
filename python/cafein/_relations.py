"""OSM public-transport route-relation extraction — tier 3's data source.

Reads route relations through pyrosm (the only OSM stack): every
relation tagged ``type=route`` with a public-transport ``route=`` value,
its ordered members with roles, member-way geometries and the way tags
the stitcher consumes, and stop/platform member coordinates. Members
whose element falls outside the extract resolve to ``None`` geometry —
the downstream validation gates decide what that costs, extraction never
guesses.
"""

import dataclasses

import numpy as np
import shapely

#: The ``route=`` values tier 3 matches against GTFS routes.
PT_ROUTE_VALUES = (
    "bus",
    "trolleybus",
    "tram",
    "light_rail",
    "subway",
    "train",
    "ferry",
)

#: Way tags kept on member records — what the stitcher's rules consume
#: (ring classification and orientation) plus the mode-sanity tags.
MEMBER_WAY_TAGS = ("junction", "oneway", "highway", "railway")


@dataclasses.dataclass(frozen=True)
class RelationMember:
    """One ordered member of a route relation.

    ``kind`` is ``"way"``, ``"node"``, or ``"relation"``; ``geometry``
    is the way's LineString or the node's Point, or ``None`` when the
    element is not materialized in the extract (boundary-crossing
    members); ``tags`` carries `MEMBER_WAY_TAGS` for ways, empty
    otherwise.
    """

    kind: str
    id: int
    role: str
    geometry: object
    tags: dict


@dataclasses.dataclass(frozen=True)
class RouteRelation:
    """One extracted ``type=route`` relation with ordered members."""

    id: int
    route: str
    ref: str | None
    name: str | None
    operator: str | None
    network: str | None
    members: tuple


def route_relations(source, bounding_box=None):
    """The extract's public-transport route relations, members ordered.

    Parameters
    ----------
    source : str, pathlib.Path, or pyrosm.OSM
        The OSM extract, or an already-open pyrosm reader (opened with
        ``complete_relations=True`` so boundary-crossing member ways
        materialize).
    bounding_box : list, optional
        Passed to pyrosm when ``source`` is a path.

    Returns
    -------
    list of RouteRelation
        Every ``type=route`` relation whose ``route=`` is one of
        `PT_ROUTE_VALUES`, in file order.
    """
    from pyrosm import OSM

    if isinstance(source, OSM):
        osm = source
    else:
        osm = OSM(str(source), bounding_box=bounding_box, complete_relations=True)
    if osm._relations is None:
        _materialize(osm)
    relations = osm._relations
    if relations is None:
        return []
    tags = relations["tags"]
    members = relations["members"]
    ids = relations["id"]

    kept = [
        index
        for index, tag in enumerate(tags)
        if isinstance(tag, dict)
        and tag.get("type") == "route"
        and tag.get("route") in PT_ROUTE_VALUES
    ]
    if not kept:
        return []

    needed_ways = set()
    needed_nodes = set()
    for index in kept:
        member = members[index]
        for member_id, member_type in zip(member["member_id"], member["member_type"]):
            if member_type == b"way":
                needed_ways.add(int(member_id))
            elif member_type == b"node":
                needed_nodes.add(int(member_id))
    way_index = _way_index(osm, needed_ways)
    node_points = _node_points(osm, needed_nodes)

    extracted = []
    for index in kept:
        tag = tags[index]
        member = members[index]
        ordered = tuple(
            _member(member_id, member_type, role, osm, way_index, node_points)
            for member_id, member_type, role in zip(
                member["member_id"], member["member_type"], member["member_role"]
            )
        )
        extracted.append(
            RouteRelation(
                id=int(ids[index]),
                route=tag["route"],
                ref=tag.get("ref"),
                name=tag.get("name"),
                operator=tag.get("operator"),
                network=tag.get("network"),
                members=ordered,
            )
        )
    return extracted


def _materialize(osm):
    """Populate pyrosm's node/way/relation caches.

    pyrosm parses the whole extract into its caches on the first layer
    read whatever the layer's own filter keeps, so the route-filtered
    criteria read is the lightest trigger (it frames only the matching
    handful instead of building a network graph)."""
    osm.get_data_by_custom_criteria(
        custom_filter={"route": list(PT_ROUTE_VALUES)},
        filter_type="keep",
        keep_nodes=False,
        keep_ways=False,
        keep_relations=True,
    )


def rail_network(source, bounding_box=None):
    """The rail-family ways of the extract — tier 4's rail graph source.

    A pyrosm custom-criteria read keeping ``railway=`` tram,
    light_rail, subway, and rail ways, with **service ways excluded**
    (yards, sidings, spurs never carry timetabled patterns). Returns a
    GeoDataFrame of ways with geometry and the ``railway``, ``oneway``,
    and ``service`` columns retained, or an empty frame when the
    extract has no rails.
    """
    import geopandas
    from pyrosm import OSM

    if isinstance(source, OSM):
        osm = source
    else:
        osm = OSM(str(source), bounding_box=bounding_box, complete_relations=True)
    ways = osm.get_data_by_custom_criteria(
        custom_filter={"railway": ["tram", "light_rail", "subway", "rail"]},
        filter_type="keep",
        keep_nodes=False,
        keep_ways=True,
        keep_relations=False,
        osm_keys_to_keep=["railway", "oneway", "service"],
    )
    if ways is None or not len(ways):
        return geopandas.GeoDataFrame(
            columns=["id", "railway", "oneway", "service", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )
    return _service_filtered(ways)


def _service_filtered(ways):
    """The graph-relevant keys lifted to columns and service ways
    dropped — split out so the exclusion is testable against a
    constructed frame, independent of any extract."""
    # Keys outside pyrosm's default column set ride in the JSON ``tags``
    # column; lift the ones the graph builder consumes to real columns.
    for key in ("oneway", "service"):
        if key not in ways.columns:
            ways[key] = [
                _tag_value(cell, key) for cell in ways.get("tags", [None] * len(ways))
            ]
    ways = ways[ways["service"].isna()]
    return ways.reset_index(drop=True)


def _tag_value(cell, key):
    """One key out of pyrosm's ``tags`` column (JSON string or dict)."""
    if isinstance(cell, str):
        import json

        try:
            cell = json.loads(cell)
        except ValueError:
            return None
    if isinstance(cell, dict):
        return cell.get(key)
    return None


def _member(member_id, member_type, role, osm, way_index, node_points):
    kind = member_type.decode() if isinstance(member_type, bytes) else str(member_type)
    member_id = int(member_id)
    geometry = None
    tags = {}
    if kind == "way":
        record = way_index.get(member_id)
        if record is not None:
            geometry = _way_line(osm, record)
            tags = {
                key: record[key]
                for key in MEMBER_WAY_TAGS
                if record.get(key) is not None
            }
    elif kind == "node":
        geometry = node_points.get(member_id)
    return RelationMember(
        kind=kind, id=member_id, role=str(role), geometry=geometry, tags=tags
    )


def _way_index(osm, needed):
    """The needed member-way records by id, one pass over the cache."""
    index = {}
    if not needed or osm._way_records is None:
        return index
    for record in osm._way_records:
        way_id = int(record["id"])
        if way_id in needed:
            index[way_id] = record
    return index


def _node_points(osm, needed):
    """Point geometries for the needed member nodes, where materialized."""
    points = {}
    if not needed or osm._node_coordinates is None:
        return points
    wanted = np.fromiter(needed, dtype=np.int64)
    found, lons, lats = osm._node_coordinates.gather(wanted)
    # ``gather`` returns positions aligned with its hits: recover the
    # ids it actually found by gathering per id only when short.
    if len(found) == len(wanted):
        matched = wanted
    else:
        matched = []
        for node_id in wanted:
            hit, _, _ = osm._node_coordinates.gather(
                np.asarray([node_id], dtype=np.int64)
            )
            if len(hit):
                matched.append(node_id)
        matched = np.asarray(matched, dtype=np.int64)
    for node_id, lon, lat in zip(matched, lons, lats):
        points[int(node_id)] = shapely.Point(float(lon), float(lat))
    return points


def _way_line(osm, record):
    """The way's LineString from its node refs, ``None`` on missing refs."""
    refs = np.asarray(record["nodes"], dtype=np.int64)
    if len(refs) < 2:
        return None
    _, lons, lats = osm._node_coordinates.gather(refs)
    if len(lons) != len(refs):
        return None
    return shapely.LineString(np.column_stack([lons, lats]))
