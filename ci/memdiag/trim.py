"""Is the memory a deleted network leaves behind freed, or still live?

Each network kind runs in a fresh process: build it, delete it, collect,
then ``malloc_trim(0)``; anonymous RSS that falls after the trim was freed
memory glibc had kept, anonymous RSS that stays is live. Then the same
process loads the kind's saved artifact and runs one query, to show what a
loaded network costs.
"""

import gc
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
from memprof_plugin import _FIELDS, _read_kb_fields, trim_heap  # noqa: E402

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


def one_kind(kind):
    """Build, delete, trim, then load and query one kind; print each step."""
    from cafein import TransportNetwork, TravelTimeMatrix

    steps = [("start", anon_mb())]
    network = TransportNetwork.from_gtfs(GTFS, **KINDS[kind])
    steps.append(("built", anon_mb()))
    path = os.path.join(tempfile.mkdtemp(), f"{kind}.cafein")
    network.save(path)
    del network
    steps.append(("deleted", anon_mb()))
    trim_heap()
    steps.append(("trimmed", anon_mb()))
    loaded = TransportNetwork.load(path)
    steps.append(("loaded", anon_mb()))
    stops = [stop_id for stop_id, *_ in loaded.stops[:20]]
    TravelTimeMatrix(loaded, stops, departure="2022-02-22 08:30:00")
    steps.append(("queried", anon_mb()))
    trim_heap()
    steps.append(("trimmed", anon_mb()))
    print(f"TRIM {kind}: " + ", ".join(f"{name} {mb:.0f}" for name, mb in steps), flush=True)


if len(sys.argv) > 1:
    one_kind(sys.argv[1])
else:
    codes = []
    for kind in KINDS:
        codes.append(subprocess.call([sys.executable, __file__, kind]))
        print(f"TRIM {kind} exit {codes[-1]}", flush=True)
    sys.exit(1 if any(code != 0 for code in codes) else 0)
