#!/usr/bin/env python3

"""Window and destination ladders for the batched pareto frontier.

    python ladder.py prep    # build the network + cached McTBTR set once
    python ladder.py drive   # run every rung, one subprocess each

Ladder A: 50 origins x 200 destinations, windows 5/10/15/30/45/60 min.
Ladder B: 50 origins x 15-min window, destinations 50/100/250/500/1000.
Both routers per rung; each rung runs in its own process so peak RSS is
attributable. Results append to ladder-<current date>.csv; every row
carries the run id and the inputs' modification stamps.
"""

import csv
import datetime
import json
import os
import pathlib
import resource
import subprocess
import sys
import time

BENCH = pathlib.Path(
    os.environ.get("GRID_BENCH_DIR")
    or (pathlib.Path.home() / ".cache" / "cafein-bench")
)
REPO = pathlib.Path(__file__).resolve().parent.parent
GTFS = REPO / "tests" / "data" / "helsinki_gtfs.zip"
PBF = BENCH / "Helsinki.osm.pbf"
ARTIFACT = BENCH / "net_ladder.cafein"
CSV_PATH = BENCH / f"ladder-{datetime.date.today().isoformat()}.csv"

DATE = "2022-02-22"
DEPARTURE = "08:30:00"
WALK = dict(walking_speed_kmph=3.6, max_walking_time=1800, max_snap_distance=1600)
ORIGINS = 50
WINDOWS_A = [300, 600, 900, 1800, 2700, 3600]
DESTINATIONS_A = 200
WINDOW_B = 900
DESTINATIONS_B = [50, 100, 250, 500, 1000]
FIELDS = [
    "run_id",
    "inputs_stamp",
    "ladder",
    "router",
    "origins",
    "destinations",
    "window_s",
    "pairs",
    "load_seconds",
    "query_seconds",
    "ms_per_pair",
    "frontier_rows",
    "peak_rss_mb",
    "stamp",
]


def frames(origin_count, destination_count):
    import geopandas as gpd
    import pandas as pd
    from shapely import wkb

    def load(name, count):
        frame = pd.read_parquet(BENCH / name).head(count)
        geometry = [wkb.loads(raw) for raw in frame["geometry"]]
        return gpd.GeoDataFrame({"id": frame["id"]}, geometry=geometry, crs="EPSG:4326")

    return load("origins.parquet", origin_count), load(
        "grid_all.parquet", destination_count
    )


def prep():
    from cafein import TransportNetwork

    started = time.perf_counter()
    network = TransportNetwork.from_gtfs(
        [str(GTFS)], osm_pbf=str(PBF), trip_distances=True
    )
    print(f"build {time.perf_counter() - started:.1f}s", flush=True)
    started = time.perf_counter()
    network.compute_mctbtr_transfers(DATE)
    print(f"mctbtr set {time.perf_counter() - started:.1f}s", flush=True)
    network.save(str(ARTIFACT))
    print(f"saved {ARTIFACT}", flush=True)


def inputs_stamp():
    """The inputs' modification times, pinned per run: a mid-run `prep`
    or point-file replacement changes the stamp and fails the rung."""
    return "|".join(
        str(int((BENCH / name).stat().st_mtime))
        for name in ("net_ladder.cafein", "origins.parquet", "grid_all.parquet")
    )


def peak_rss_mb():
    scale = 1024 * 1024 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


def run(run_id, stamp, ladder, router, destination_count, window):
    from cafein import TransportNetwork
    from cafein.frontier import frontier_table

    if inputs_stamp() != stamp:
        raise SystemExit("the artifact or point files changed mid-run")
    origins, destinations = frames(ORIGINS, destination_count)
    if len(origins) != ORIGINS or len(destinations) != destination_count:
        raise SystemExit(
            f"prepared points are smaller than the rung: {len(origins)} origins, "
            f"{len(destinations)} destinations"
        )
    started = time.perf_counter()
    network = TransportNetwork.load(str(ARTIFACT))
    load_seconds = round(time.perf_counter() - started, 2)
    started = time.perf_counter()
    frame = frontier_table(
        network, origins, destinations, DATE, DEPARTURE, window, router=router, **WALK
    )
    query_seconds = round(time.perf_counter() - started, 2)
    pairs = len(origins) * len(destinations)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "inputs_stamp": stamp,
                "ladder": ladder,
                "router": router,
                "origins": len(origins),
                "destinations": len(destinations),
                "window_s": window,
                "pairs": pairs,
                "load_seconds": load_seconds,
                "query_seconds": query_seconds,
                "ms_per_pair": round(1000 * query_seconds / pairs, 2),
                "frontier_rows": int(len(frame)),
                "peak_rss_mb": round(peak_rss_mb(), 1),
                "stamp": datetime.datetime.now().isoformat(timespec="seconds"),
            }
        ),
        flush=True,
    )


def drive():
    run_id = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    stamp = inputs_stamp()
    failures = 0
    rungs = []
    for window in WINDOWS_A:
        for router in ("raptor", "tbtr"):
            rungs.append(("A", router, DESTINATIONS_A, window))
    for destination_count in DESTINATIONS_B:
        for router in ("raptor", "tbtr"):
            rungs.append(("B", router, destination_count, WINDOW_B))
    fresh = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=FIELDS)
        if fresh:
            writer.writeheader()
        for ladder, router, destination_count, window in rungs:
            print(
                f"RUNG {ladder} {router} dests={destination_count} window={window}",
                flush=True,
            )
            output = subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "run",
                    run_id,
                    stamp,
                    ladder,
                    router,
                    str(destination_count),
                    str(window),
                ],
                capture_output=True,
                text=True,
                env=os.environ,
            )
            line = [
                emitted
                for emitted in output.stdout.splitlines()
                if emitted.startswith("{")
            ]
            if output.returncode != 0 or not line:
                print("FAILED", output.returncode, output.stderr[-500:], flush=True)
                failures += 1
                continue
            record = json.loads(line[-1])
            writer.writerow(record)
            sink.flush()
            print("OK", line[-1], flush=True)
    if failures:
        raise SystemExit(f"{failures} rung(s) failed; the CSV is partial")


if __name__ == "__main__":
    if sys.argv[1] == "prep":
        prep()
    elif sys.argv[1] == "drive":
        drive()
    else:
        run(
            sys.argv[2],
            sys.argv[3],
            sys.argv[4],
            sys.argv[5],
            int(sys.argv[6]),
            int(sys.argv[7]),
        )
