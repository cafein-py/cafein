#!/usr/bin/env python3

"""Footpath-set size and multicriteria query cost, before and after the
bounded-transfer change (#250, issue #141's premise).

    python benchmark_bounded_footpaths.py <label>

Runs against the ladder benchmark's inputs (full HSL feed + BBBike
Helsinki walking network) so the numbers sit beside the 2026-07-13
ladder. One JSON record per run appends to
``$GRID_BENCH_DIR/bounded-footpaths.jsonl``; run the same command at
each revision and compare records by label.
"""

import datetime
import json
import os
import pathlib
import resource
import sys
import time

BENCH = pathlib.Path(
    os.environ.get("GRID_BENCH_DIR")
    or (pathlib.Path.home() / ".cache" / "cafein-bench")
)
REPO = pathlib.Path(__file__).resolve().parent.parent
GTFS = REPO / "tests" / "data" / "helsinki_gtfs.zip"
PBF = BENCH / "Helsinki.osm.pbf"
SINK = BENCH / "bounded-footpaths.jsonl"

DATE = "2022-02-22"
DEPARTURE = "08:30:00"
WALK = dict(walking_speed_kmph=3.6, max_walking_time=1800, max_snap_distance=1600)
ORIGINS = 10
DESTINATIONS = 100
WINDOW = 900


def peak_rss_mb():
    scale = 1024 * 1024 if sys.platform == "darwin" else 1024
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale, 1)


def frames():
    import geopandas as gpd
    import pandas as pd
    from shapely import wkb

    def load(name, count):
        frame = pd.read_parquet(BENCH / name).head(count)
        geometry = [wkb.loads(raw) for raw in frame["geometry"]]
        return gpd.GeoDataFrame({"id": frame["id"]}, geometry=geometry, crs="EPSG:4326")

    return load("origins.parquet", ORIGINS), load("grid_all.parquet", DESTINATIONS)


def main(label):
    from cafein import TransportNetwork
    from cafein.frontier import frontier_table

    record = {
        "label": label,
        "stamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "threads": os.environ.get("RAYON_NUM_THREADS", "auto"),
        "max_walking_time": WALK["max_walking_time"],
    }

    started = time.perf_counter()
    network = TransportNetwork.from_gtfs(
        [str(GTFS)], osm_pbf=str(PBF), trip_distances=True, **WALK
    )
    record["build_seconds"] = round(time.perf_counter() - started, 1)
    record["stops"] = len(list(network.stops))
    record["transfers"] = network.transfer_count
    record["transfers_per_stop"] = round(record["transfers"] / record["stops"], 1)
    record["build_rss_mb"] = peak_rss_mb()

    started = time.perf_counter()
    network.compute_mctbtr_transfers(DATE)
    record["mctbtr_seconds"] = round(time.perf_counter() - started, 1)

    origins, destinations = frames()
    for router in ("raptor", "tbtr"):
        started = time.perf_counter()
        frame = frontier_table(
            network,
            origins,
            destinations,
            DATE,
            DEPARTURE,
            WINDOW,
            router=router,
            **WALK,
        )
        seconds = round(time.perf_counter() - started, 2)
        record[f"{router}_seconds"] = seconds
        record[f"{router}_ms_per_pair"] = round(
            1000 * seconds / (ORIGINS * DESTINATIONS), 2
        )
        record[f"{router}_rows"] = int(len(frame))
    record["peak_rss_mb"] = peak_rss_mb()

    with open(SINK, "a") as sink:
        sink.write(json.dumps(record) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
