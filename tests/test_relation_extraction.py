"""The OSM route-relation extraction contract (distance-tier 3 intake).

The committed ``helsinki-transit.osm.pbf`` fixture (a transit-only
subset of the pinned metro clip) carries every PT route relation, so the
contract battery runs everywhere, CI included; only the deep
whole-extract checks need the locally generated metro clip.
"""

import pathlib

import pytest
import shapely

from cafein import _relations

TRANSIT = pathlib.Path(__file__).parent / "data" / "helsinki-transit.osm.pbf"
GAP = pathlib.Path(__file__).parent / "data" / "helsinki-transit-gap.osm.pbf"
MISSING = pathlib.Path(__file__).parent / "data" / "helsinki-transit-missing.osm.pbf"

EXPECTED_COUNTS = {
    "bus": 1835,
    "ferry": 93,
    "train": 56,
    "tram": 22,
    "subway": 4,
    "light_rail": 4,
}


@pytest.fixture(scope="module")
def transit_relations():
    return _relations.route_relations(str(TRANSIT))


def test_kantakaupunki_has_no_route_relations(kantakaupunki_pbf):
    # The central fixture carries only turn restrictions (the r5py
    # pipeline stripped route relations): extraction returns empty
    # rather than failing — the no-data path.
    assert _relations.route_relations(str(kantakaupunki_pbf)) == []


def test_transit_extraction_counts(transit_relations):
    counts = {}
    for relation in transit_relations:
        counts[relation.route] = counts.get(relation.route, 0) + 1
    assert counts == EXPECTED_COUNTS


def test_transit_relation_fields_and_members(transit_relations):
    trams = [r for r in transit_relations if r.route == "tram"]
    assert {"1", "4", "6", "10"} <= {relation.ref for relation in trams}
    sample = next(relation for relation in trams if relation.ref == "1")
    assert sample.network == "HSL"
    assert sample.operator is not None
    roles = {member.role for member in sample.members}
    assert "platform" in roles
    way_members = [m for m in sample.members if m.kind == "way"]
    platform_members = [
        m for m in sample.members if m.kind == "node" and m.role == "platform"
    ]
    assert way_members and platform_members
    assert all(
        isinstance(member.geometry, shapely.Point) for member in platform_members
    )
    # The member ways carry the stitcher's tags; a tram line's ways are
    # railway-tagged — and railway ways do NOT match the route= keep
    # filter, pinning that member materialization is filter-independent.
    assert any(member.tags.get("railway") == "tram" for member in way_members)
    # Coordinates associate with the RIGHT node ids (a swapped
    # id-to-coordinate mapping would corrupt corridor matching):
    # pinned lon/lat of tram 1's first platforms, from the fixture.
    pinned = {
        313998777: (24.9341353, 60.1584709),
        655131553: (24.9359450, 60.1577682),
        6147301569: (24.9414967, 60.1586235),
    }
    by_id = {member.id: member for member in platform_members}
    for node_id, (lon, lat) in pinned.items():
        point = by_id[node_id].geometry
        assert abs(point.x - lon) < 1e-6 and abs(point.y - lat) < 1e-6


def test_ring_tags_survive_extraction(transit_relations):
    # The closed-ring rule consumes junction/oneway from member ways;
    # pinned on known members so tag stripping cannot pass silently.
    bus_75 = next(r for r in transit_relations if r.route == "bus" and r.ref == "75")
    ways = {m.id: m for m in bus_75.members if m.kind == "way"}
    assert ways[341430222].tags.get("junction") == "roundabout"
    assert ways[341430220].tags.get("junction") == "roundabout"
    tram_1 = next(r for r in transit_relations if r.route == "tram" and r.ref == "1")
    tram_ways = {m.id: m for m in tram_1.members if m.kind == "way"}
    assert tram_ways[649755999].tags.get("oneway") == "yes"


def test_fully_contained_mode_resolves_completely(transit_relations):
    # The spatial-vs-filter contract: trams are wholly inside the clip,
    # so every member way and platform node must resolve — no excuses.
    trams = [r for r in transit_relations if r.route == "tram"]
    for relation in trams:
        for member in relation.members:
            if member.kind in ("way", "node"):
                assert member.geometry is not None, (relation.ref, member.id)


#: Tram 1 (relation 52918) as OSM records it in the pinned fixture —
#: an independent literal, so a pyrosm parse that lost or reordered
#: members would fail against it (not just against itself).
TRAM_1_ID = 52918
TRAM_1_MEMBER_COUNT = 181
TRAM_1_FIRST_MEMBERS = [
    ("node", 313998777, "platform"),
    ("node", 655131553, "platform"),
    ("node", 6147301569, "platform"),
    ("node", 310147815, "platform"),
    ("node", 319802167, "platform"),
    ("node", 395244182, "platform"),
]
TRAM_1_LAST_MEMBERS = [
    ("way", 327265252, ""),
    ("way", 327265019, ""),
]
#: Order-sensitive digest of ALL 181 members ("kind:id:role|" joined) —
#: middle-of-sequence reordering or role corruption fails this even
#: though only the ends are pinned literally.
TRAM_1_SEQUENCE_SHA256 = (
    "6734ec56fef9ceb8b0af35755735204d6d494a681763b8ff227ebef02f7ee644"
)


def test_member_order_is_preserved():
    # Both pyrosm's raw arrays AND the wrapper must match the literal
    # pinned sequence — order is the stitcher's ground truth, and a
    # self-referential comparison could not catch pyrosm itself
    # dropping or reordering members.
    from pyrosm import OSM

    osm = OSM(str(TRANSIT), complete_relations=True)
    relations = _relations.route_relations(osm)
    sample = next(r for r in relations if r.id == TRAM_1_ID)
    extracted = [(m.kind, m.id, m.role) for m in sample.members]
    assert len(extracted) == TRAM_1_MEMBER_COUNT
    assert extracted[: len(TRAM_1_FIRST_MEMBERS)] == TRAM_1_FIRST_MEMBERS
    assert extracted[-len(TRAM_1_LAST_MEMBERS) :] == TRAM_1_LAST_MEMBERS
    import hashlib

    sequence = "|".join(f"{kind}:{mid}:{role}" for kind, mid, role in extracted)
    assert hashlib.sha256(sequence.encode()).hexdigest() == TRAM_1_SEQUENCE_SHA256
    raw = osm._relations
    raw_members = next(
        member
        for identifier, member in zip(raw["id"], raw["members"])
        if int(identifier) == TRAM_1_ID
    )
    raw_sequence = [
        (kind.decode() if isinstance(kind, bytes) else str(kind), int(mid), str(role))
        for mid, kind, role in zip(
            raw_members["member_id"],
            raw_members["member_type"],
            raw_members["member_role"],
        )
    ]
    assert raw_sequence == extracted


def test_gap_variant_drops_the_member(transit_relations):
    # The defect fixture deleted one member way of tram 1; pyrosm's
    # deletion policy removes the ref (never dangling), so the gap is
    # geometric: one member fewer and the neighbouring ways no longer
    # touch — the chain break the stitcher must refuse.
    whole = next(r for r in transit_relations if r.route == "tram" and r.ref == "1")
    gapped_relations = _relations.route_relations(str(GAP))
    gapped = next(r for r in gapped_relations if r.route == "tram" and r.ref == "1")
    assert len(gapped.members) == len(whole.members) - 1
    deleted = set(m.id for m in whole.members) - set(m.id for m in gapped.members)
    assert len(deleted) == 1
    # Everything else is untouched: same order with one excision.
    survivors = [m.id for m in whole.members if m.id not in deleted]
    assert [m.id for m in gapped.members] == survivors
    # And the gap is geometrically real: the surviving neighbours of
    # the excised way cannot be chained within any stitcher tolerance
    # (their gap is the deleted way's span, tens of metres).
    position = [m.id for m in whole.members].index(next(iter(deleted)))
    before = whole.members[position - 1].geometry
    after = whole.members[position + 1].geometry
    assert before is not None and after is not None
    gap_degrees = before.distance(after)
    assert gap_degrees * 111_320 > 20


def test_missing_variant_drops_the_relation(transit_relations):
    # The defect fixture deleted one tram-1 relation wholly: the
    # no-match input. Refs are per line, not per relation — the two
    # directions each carry ref "1" — so the pin keys on the relation
    # id (the first ref-1 tram in file order, as the generator picks).
    deleted = next(r for r in transit_relations if r.route == "tram" and r.ref == "1")
    remaining = _relations.route_relations(str(MISSING))
    assert deleted.id not in {relation.id for relation in remaining}
    assert len(remaining) == len(transit_relations) - 1


def test_rail_ways_carry_node_identities():
    ways = _relations.rail_ways(str(TRANSIT))
    assert len(ways) > 100
    for refs, lons, lats, railway, _oneway in ways[:50]:
        assert railway in ("tram", "light_rail", "subway", "rail")
        assert len(refs) == len(lons) == len(lats) >= 2
        assert refs.dtype.kind == "i"


def test_rail_network_extraction():
    rails = _relations.rail_network(str(TRANSIT))
    assert len(rails)
    assert set(rails["railway"].unique()) <= {"tram", "light_rail", "subway", "rail"}
    # Service ways (yards, sidings) are excluded, and the key is lifted
    # out of pyrosm's JSON tags column so the exclusion actually bites.
    assert "service" in rails.columns
    assert rails["service"].isna().all()
    assert "oneway" in rails.columns
    assert set(rails.geometry.geom_type.unique()) <= {"LineString", "MultiLineString"}


def test_service_exclusion_actually_filters():
    # Against a constructed frame — so "no service column exposed"
    # can never masquerade as "service ways excluded".
    import geopandas
    import json

    frame = geopandas.GeoDataFrame(
        {
            "id": [1, 2, 3],
            "railway": ["rail", "rail", "tram"],
            "tags": [
                json.dumps({"service": "yard"}),
                json.dumps({"usage": "main"}),
                None,
            ],
        },
        geometry=[
            shapely.LineString([(0, 0), (1, 1)]),
            shapely.LineString([(1, 1), (2, 2)]),
            shapely.LineString([(2, 2), (3, 3)]),
        ],
        crs="EPSG:4326",
    )
    filtered = _relations._service_filtered(frame)
    assert list(filtered["id"]) == [2, 3]
    assert "service" in filtered.columns and "oneway" in filtered.columns


def test_rail_ways_on_railless_extract(kantakaupunki_pbf):
    # An extract with streets but no rails (and no PT relations): the
    # node-identity reader returns empty without touching relations.
    assert _relations.rail_ways(str(kantakaupunki_pbf)) == []


def test_rail_network_on_railless_extract(kantakaupunki_pbf):
    # The r5py pipeline stripped railway ways from the central fixture
    # along with the relations — which pins the empty-input contract:
    # a graceful empty frame with the promised columns, no exceptions.
    rails = _relations.rail_network(str(kantakaupunki_pbf))
    assert len(rails) == 0
    assert {"railway", "oneway", "service", "geometry"} <= set(rails.columns)


def test_metro_boundary_crossing_members_resolve_to_none(helsinki_metro_pbf):
    # Deep local check: train lines leave the metro clip; their outside
    # members resolve to None (spatial truncation, handled by the
    # ladder's fallthrough) while the rest keep usable geometry.
    relations = _relations.route_relations(str(helsinki_metro_pbf))
    trains = [relation for relation in relations if relation.route == "train"]
    assert trains
    # WAY members specifically (relation-kind members are always None,
    # so counting them would prove nothing about spatial truncation).
    unresolved_ways = sum(
        1
        for relation in trains
        for member in relation.members
        if member.kind == "way" and member.geometry is None
    )
    assert unresolved_ways > 0
    # A pinned boundary-crossing way of train relation 375226 — its
    # geometry leaves the clip and must stay None, never fabricated.
    pinned_relation = next(r for r in trains if r.id == 375226)
    pinned_member = next(m for m in pinned_relation.members if m.id == 422841717)
    assert pinned_member.kind == "way"
    assert pinned_member.geometry is None
    resolved_ways = sum(
        1
        for relation in trains
        for member in relation.members
        if member.kind == "way" and member.geometry is not None
    )
    assert resolved_ways > 0
