"""Generate the OSM distance-tier fixtures (run manually).

Three fixture classes, all produced through pyrosm alone:

- ``tests/data/helsinki-metro.osm.pbf`` — the full Helsinki
  metropolitan clip of the pinned Geofabrik snapshot (streets, rails,
  and route relations; ~100 MB). **Local development artifact only**
  (never hosted, never committed): the street-graph work and the
  calibration sweep need it; tests that do skip when it is absent.
- ``tests/data/helsinki-transit.osm.pbf`` — the transit-only subset
  (route relations + members recursively; a few MB). **Committed to
  the repository** (Henrikki, 2026-08-05), so the extraction-contract
  tests run everywhere, CI included.
- ``tests/data/helsinki-transit-*.osm.pbf`` defect variants — the
  transit subset with a tram member way / a whole relation deleted
  (``write_pbf(delete=...)``), pinning the gap and no-match failure
  modes. Committed beside the subset.

Every output is staged, verified through ``cafein._relations`` (the
same reader the library uses), and atomically published; the recorded
constants below pin the whole pipeline — a changed checksum means the
inputs or pyrosm changed, deliberately or not.
"""

import argparse
import hashlib
import io
import os
import pathlib
import sys
import tempfile
import time
import urllib.request
import zipfile

SNAPSHOT_URL = "https://download.geofabrik.de/europe/finland-220101.osm.pbf"
SNAPSHOT_SHA256 = "2f7cd629a49b3b1f74fdaa9405340a044e8797763fad010a3c5a40df9179a00a"
MARGIN_DEGREES = 0.10
"""Clip margin beyond the GTFS stop bounds (~11 km latitude, ~5.5 km
longitude at Helsinki): room for route geometries that bow outside the
outermost stops."""

PT_ROUTE_VALUES = (
    "bus",
    "trolleybus",
    "tram",
    "light_rail",
    "subway",
    "train",
    "ferry",
)

#: The pinned per-mode relation counts of the metro clip — the
#: verification refuses a drifted regeneration instead of shipping it.
EXPECTED_COUNTS = {
    "bus": 1835,
    "ferry": 93,
    "train": 56,
    "tram": 22,
    "subway": 4,
    "light_rail": 4,
}

#: The tram relation and member way the gap variant deletes, and the
#: relation the no-match variant deletes (tram 1, from the pinned clip).
GAP_RELATION_REF = "1"

#: Representative tag-key counts of the metro clip — the losslessness
#: pins: a write path that started stripping tags (the corruption an
#: overlaying frame could cause) fails these before publication.
EXPECTED_TAG_KEYS = {
    "bus": 472,
    "psv": 515,
    "busway": 36,
    "railway": 8734,
    "gauge": 7455,
    "junction": 1668,
}

#: Recorded output digests — ``write_pbf`` is deterministic for pinned
#: inputs (verified: independent runs hash identically), so any change
#: means an input or pyrosm changed; update deliberately, never blindly.
EXPECTED_SHA256 = {
    # Deliberate regeneration 2026-08-05: the clip switched to the
    # overlay-free empty-frame write (byte-different, semantically
    # identical — counts and tag pins unchanged). Downstream digests
    # repin from this run's published outputs.
    "helsinki-metro.osm.pbf": (
        "f5d07897154b2e4c2b78815f614e312a56827cd2bf528c61a329b2d6a2ddde57"
    ),
    "helsinki-transit.osm.pbf": (
        "9f0f010c00eb0d0231c855fd4abe62c949658e10026f66f0ff40fd4525a274be"
    ),
    "helsinki-transit-gap.osm.pbf": (
        "c7443d7af3afa84c7b26430fabbd1104ae10390429d05d61a0d38f017ca24156"
    ),
    "helsinki-transit-missing.osm.pbf": (
        "6f04bfb0b072267778611c2a6d8e5aa173536b1123d73e85df009fdc87e46078"
    ),
}
DATA = pathlib.Path(__file__).resolve().parent.parent / "tests" / "data"


def sha256(path):
    state = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            state.update(chunk)
    return state.hexdigest()


def publish(build, target, verifier=None):
    """Stage into a sibling temporary, verify the STAGED file (semantic
    verifier plus the recorded digest), and only then replace the
    canonical path — a rejected build never displaces a known-good
    fixture."""
    # The staged name must end in ``.osm.pbf``: pyrosm's read path (the
    # verifier) validates the extension.
    descriptor, staged = tempfile.mkstemp(dir=target.parent, suffix=".staged.osm.pbf")
    os.close(descriptor)
    staged = pathlib.Path(staged)
    try:
        build(staged)
        if verifier is not None:
            verifier(staged)
        digest = sha256(staged)
        expected = EXPECTED_SHA256.get(target.name)
        if expected is not None and digest != expected:
            raise SystemExit(
                f"{target.name}: staged sha256 {digest} != recorded "
                f"{expected}. Content drift — an input or pyrosm changed. "
                "For a deliberate regeneration, update EXPECTED_SHA256."
            )
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    os.replace(staged, target)
    print(
        f"published {target.name}: {target.stat().st_size / 1e6:.1f} MB "
        f"sha256={digest}"
    )


def ensure_snapshot():
    target = DATA / "finland-220101.osm.pbf"
    if not target.exists():
        print(f"downloading {SNAPSHOT_URL} ...", flush=True)

        def fetch(staged):
            urllib.request.urlretrieve(SNAPSHOT_URL, staged)
            digest = sha256(staged)
            if digest != SNAPSHOT_SHA256:
                raise SystemExit(
                    f"snapshot checksum mismatch: {digest}; the pinned "
                    "Geofabrik snapshot should be immutable — refusing"
                )

        publish(fetch, target)
    digest = sha256(target)
    if digest != SNAPSHOT_SHA256:
        raise SystemExit(
            f"snapshot checksum mismatch: {digest} != {SNAPSHOT_SHA256}; "
            "delete the file to re-download"
        )
    print(f"snapshot ok: {target.name}")
    return target


def gtfs_bounds():
    """The GTFS fixture's stop bounds plus the fixed margin."""
    import pandas as pd

    with zipfile.ZipFile(DATA / "helsinki_gtfs.zip") as archive:
        stops = pd.read_csv(io.BytesIO(archive.read("stops.txt")))
    return [
        float(stops["stop_lon"].min()) - MARGIN_DEGREES,
        float(stops["stop_lat"].min()) - MARGIN_DEGREES,
        float(stops["stop_lon"].max()) + MARGIN_DEGREES,
        float(stops["stop_lat"].max()) + MARGIN_DEGREES,
    ]


def open_clip(path):
    from pyrosm import OSM

    osm = OSM(str(path), complete_relations=True)
    osm.get_network(network_type="driving")
    return osm


def pt_relation_frame(osm):
    """The PT route relations as a subset/tag frame (tags as columns —
    ``write_pbf`` treats a matched row's columns as the element's tags,
    so omitting them would strip the relation bare)."""
    import geopandas as gpd
    import pandas as pd

    relations = osm._relations
    rows = []
    for identifier, tag in zip(relations["id"], relations["tags"]):
        if (
            isinstance(tag, dict)
            and tag.get("type") == "route"
            and tag.get("route") in PT_ROUTE_VALUES
        ):
            row = dict(tag)
            row["id"] = int(identifier)
            row["osm_type"] = "relation"
            rows.append(row)
    return gpd.GeoDataFrame(
        pd.DataFrame(rows), geometry=[None] * len(rows), crs="EPSG:4326"
    )


def verify_tag_keys(path):
    """Losslessness proof: the clip keeps the tags an overlaying write
    could strip (bus/PSV permissions, rails, junctions)."""
    from collections import Counter

    from pyrosm import OSM

    osm = OSM(str(path), complete_relations=True)
    osm.get_network(network_type="driving")
    counts = Counter()
    for record in osm._way_records:
        for key in EXPECTED_TAG_KEYS:
            if key in record:
                counts[key] += 1
    mismatches = {
        key: (counts.get(key, 0), expected)
        for key, expected in EXPECTED_TAG_KEYS.items()
        if counts.get(key, 0) != expected
    }
    if mismatches:
        raise SystemExit(
            f"{path.name}: tag-key counts drifted {mismatches} — the write "
            "path may be stripping tags; refusing"
        )
    print(f"  {path.name}: tag-key pins ok {dict(counts)}")


def verify_extraction(path, expect_counts=None, expect_complete=True):
    """The published fixture must satisfy the library's own extraction
    contract — counts per mode, and (unless the fixture is a deliberate
    defect variant) the fully-contained trams resolving every member."""
    from collections import Counter

    from cafein import _relations

    extracted = _relations.route_relations(str(path))
    counts = dict(Counter(relation.route for relation in extracted))
    print(f"  {path.name}: {counts}")
    if expect_counts is not None and counts != expect_counts:
        raise SystemExit(
            f"{path.name}: relation counts {counts} != pinned "
            f"{expect_counts} — inputs or pyrosm changed; refusing"
        )
    if expect_complete:
        trams = [relation for relation in extracted if relation.route == "tram"]
        for relation in trams:
            for member in relation.members:
                if member.kind == "way" and member.geometry is None:
                    raise SystemExit(
                        f"{path.name}: tram {relation.ref} member way "
                        f"{member.id} unresolvable — the fully-contained "
                        "mode must resolve completely; refusing"
                    )
    return extracted


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    snapshot = ensure_snapshot()
    bounds = gtfs_bounds()
    print("clip bounds:", [round(value, 4) for value in bounds])

    from pyrosm import OSM

    metro = DATA / "helsinki-metro.osm.pbf"

    def build_metro(staged):
        import geopandas as gpd

        started = time.time()
        source = OSM(str(snapshot), bounding_box=bounds, complete_relations=True)
        edges = source.get_network(network_type="driving")
        print(
            f"parsed snapshot in {time.time() - started:.0f}s; "
            f"{len(edges)} drive edges"
        )
        # An empty frame writes the whole cache untouched: no matched
        # rows means no tag overlays — the clip is lossless by
        # construction, and the tag-key pins below prove it stays so.
        empty = gpd.GeoDataFrame(
            {"id": [], "osm_type": []}, geometry=[], crs="EPSG:4326"
        )
        source.write_pbf(empty, str(staged))

    def verify_metro(staged):
        verify_extraction(staged, EXPECTED_COUNTS)
        verify_tag_keys(staged)

    publish(build_metro, metro, verifier=verify_metro)

    clip = open_clip(metro)
    frame = pt_relation_frame(clip)
    print("PT relations:", len(frame))
    transit = DATA / "helsinki-transit.osm.pbf"
    publish(
        lambda staged: clip.write_pbf(frame, str(staged), subset_only=True),
        transit,
        verifier=lambda staged: verify_extraction(staged, EXPECTED_COUNTS),
    )
    extracted = verify_extraction(transit, EXPECTED_COUNTS)

    tram = next(
        relation
        for relation in extracted
        if relation.route == "tram" and relation.ref == GAP_RELATION_REF
    )
    middle_way = [m for m in tram.members if m.kind == "way"][
        len([m for m in tram.members if m.kind == "way"]) // 2
    ]
    print(
        f"gap variant: deleting way {middle_way.id} of tram {tram.ref}; "
        f"no-match variant: deleting relation {tram.id}"
    )

    subset = open_clip(transit)
    subset_frame = pt_relation_frame(subset)
    gap = DATA / "helsinki-transit-gap.osm.pbf"
    publish(
        # pyrosm's deletion policy never writes dangling refs (only
        # 'drop'/'error'), so the gap defect is the dropped member: the
        # neighbouring ways no longer touch — the geometric gap the
        # stitcher must refuse.
        lambda staged: subset.write_pbf(
            subset_frame,
            str(staged),
            delete=[("way", middle_way.id)],
        ),
        gap,
        verifier=lambda staged: verify_extraction(staged, expect_complete=False),
    )

    no_match = DATA / "helsinki-transit-missing.osm.pbf"
    publish(
        lambda staged: subset.write_pbf(
            subset_frame, str(staged), delete=[("relation", tram.id)]
        ),
        no_match,
        verifier=lambda staged: verify_extraction(staged),
    )
    print("done; commit the helsinki-transit*.osm.pbf fixtures")


if __name__ == "__main__":
    sys.exit(main())
