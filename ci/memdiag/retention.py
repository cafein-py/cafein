"""Does a built network give its memory back, and what does each kind cost?

1. Build, delete and collect a network three times per kind: anonymous RSS
   that keeps rising means a deleted network is not freed.
2. Save each kind and load the artifacts one after another, keeping them:
   the anonymous RSS each load adds is what a session fixture holding that
   network costs a test worker.
"""

import gc
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
from memprof_plugin import _FIELDS, _read_kb_fields  # noqa: E402

from cafein import TransportNetwork  # noqa: E402

D = "tests/data"
GTFS = [f"{D}/helsinki_gtfs.zip"]
PBF = f"{D}/kantakaupunki.osm.pbf"
KINDS = {
    "gtfs": dict(),
    "footpaths": dict(osm_pbf=PBF),
    "multimodal": dict(
        osm_pbf=PBF, street_modes=("walk", "bicycle", "e_scooter", "wheelchair")
    ),
}


def anon_mb():
    """Anonymous RSS in MB after a gc."""
    gc.collect()
    return _read_kb_fields("/proc/self/status", _FIELDS)["RssAnon"] / 1024


print(f"RETENTION start: anon {anon_mb():.0f} MB", flush=True)
for kind, options in KINDS.items():
    for i in range(3):
        network = TransportNetwork.from_gtfs(GTFS, **options)
        held = anon_mb()
        del network
        print(
            f"RETENTION build {kind} #{i + 1}: {held:.0f} MB while held, "
            f"{anon_mb():.0f} MB after delete",
            flush=True,
        )

folder = tempfile.mkdtemp()
for kind, options in KINDS.items():
    TransportNetwork.from_gtfs(GTFS, **options).save(os.path.join(folder, kind))
print(f"RETENTION before loads: anon {anon_mb():.0f} MB", flush=True)
kept = []
for kind in KINDS:
    before = anon_mb()
    kept.append(TransportNetwork.load(os.path.join(folder, kind)))
    print(
        f"RETENTION load {kind}: +{anon_mb() - before:.0f} MB "
        f"(artifact {os.path.getsize(os.path.join(folder, kind)) / 2**20:.0f} MB on disk)",
        flush=True,
    )
print(f"RETENTION all loaded: anon {anon_mb():.0f} MB", flush=True)
