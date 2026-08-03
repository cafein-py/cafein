"""Benchmark the cutoff-pruned fare frontier against r5r's pareto_frontier.

Runs the r5r comparison shape on the Porto Alegre sample: 100 sampled
origin stops against every stop, a 10-minute departure window, a
90-minute trip cap, and the vignette's six fare cutoffs — cafein's
``fare_frontier`` in the r5r-parity fast mode, r5r's
``pareto_frontier`` on the same pairs — plus the exhaustively verified
exact mode on a bounded shape beside it.

Requirements: cafein installed, and an R environment with r5r >= 2.4
and its R5 jar cached. Point ``--rscript`` (or ``CAFEIN_RSCRIPT``) at
the environment's Rscript. The test data comes from
``python scripts/fetch_test_data.py``. Manual tool, not part of CI.
"""

import argparse
import os
import pathlib
import shlex
import shutil
import subprocess
import tempfile
import time

from compare_fares_vs_r5r import FARES, FEEDS, PBF, requalified

DATE = "2019-05-13"
DEPARTURE = "14:00:00"
WINDOW = 600
MAX_TRANSFERS = 2  # r5r max_rides = 3
MAX_DURATION = 5_400  # r5r's 90-minute cap
CUTOFFS = [1.0, 4.5, 4.8, 7.2, 8.37, 9.6]
ORIGIN_COUNT = 100
SEED = 7

R_SCRIPT = """
options(java.parameters = "-Xmx8G")
suppressMessages(library(r5r))
suppressMessages(library(data.table))
args <- commandArgs(trailingOnly = TRUE)

build_start <- Sys.time()
network <- build_network(args[1], temp_dir = FALSE)
build_seconds <- as.numeric(Sys.time() - build_start, units = "secs")
fares <- read_fare_structure(args[2])
points <- fread(args[3])
origins <- points[match(fread(args[4])$id, points$id)]

query_start <- Sys.time()
frontier <- pareto_frontier(
    network,
    origins = origins,
    destinations = points,
    mode = c("WALK", "TRANSIT"),
    departure_datetime = as.POSIXct(args[5], tz = "America/Sao_Paulo"),
    time_window = 10L,
    max_trip_duration = 90L,
    fare_structure = fares,
    fare_cutoffs = as.numeric(strsplit(args[6], ",")[[1]]),
    max_rides = 3L
)
query_seconds <- as.numeric(Sys.time() - query_start, units = "secs")
fwrite(frontier, args[7])
cat(sprintf("r5r build %.1f s, query %.1f s, %d rows\\n",
    build_seconds, query_seconds, nrow(frontier)))
"""


def main():
    import pandas as pd

    import cafein
    from cafein import fares
    from cafein.frontier import fare_frontier

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rscript",
        default=os.environ.get("CAFEIN_RSCRIPT", "Rscript"),
        help="Rscript command of an R environment with r5r installed "
        "(may include a wrapper, e.g. 'micromamba run -p <env> Rscript')",
    )
    parser.add_argument("--skip-r5r", action="store_true")
    arguments = parser.parse_args()

    build_start = time.time()
    network = cafein.TransportNetwork.from_gtfs(
        [str(feed) for feed in FEEDS], osm_pbf=str(PBF)
    )
    build_seconds = time.time() - build_start
    structure = requalified(fares.load_fare_structure(FARES), network)
    stop_ids = [stop_id for stop_id, _, _ in network.stops]
    rng = pd.Series(stop_ids).sample(ORIGIN_COUNT, random_state=SEED)
    origins = rng.to_list()
    print(f"cafein build {build_seconds:.1f} s; {len(stop_ids)} stops")

    query_start = time.time()
    fast = fare_frontier(
        network,
        origins,
        stop_ids,
        DATE,
        DEPARTURE,
        WINDOW,
        structure,
        cutoffs=CUTOFFS,
        max_transfers=MAX_TRANSFERS,
        max_duration=MAX_DURATION,
        exact=False,
    )
    fast_seconds = time.time() - query_start
    print(f"cafein fast mode, stops: {len(fast)} rows in {fast_seconds:.1f} s")

    # The apples-to-apples shape: r5r routes coordinates with walking
    # access, so the point form runs the same query.
    import geopandas as gpd
    from shapely.geometry import Point

    coordinates = {
        stop_id: (latitude, longitude)
        for stop_id, latitude, longitude in network.stops
        if latitude is not None
    }

    def frame(ids):
        rows = [(i, *coordinates[i]) for i in ids if i in coordinates]
        return gpd.GeoDataFrame(
            {"id": [row[0] for row in rows]},
            geometry=[Point(row[2], row[1]) for row in rows],
            crs="EPSG:4326",
        )

    points_start = time.time()
    points_fast = fare_frontier(
        network,
        frame(origins),
        frame(stop_ids),
        DATE,
        DEPARTURE,
        WINDOW,
        structure,
        cutoffs=CUTOFFS,
        max_transfers=MAX_TRANSFERS,
        max_duration=MAX_DURATION,
        exact=False,
    )
    points_seconds = time.time() - points_start
    print(
        f"cafein fast mode, points: {len(points_fast)} rows "
        f"in {points_seconds:.1f} s"
    )

    exact_start = time.time()
    exact = fare_frontier(
        network,
        origins[:10],
        stop_ids[:100],
        DATE,
        DEPARTURE,
        WINDOW,
        structure,
        cutoffs=CUTOFFS,
        max_transfers=MAX_TRANSFERS,
        max_duration=MAX_DURATION,
    )
    exact_seconds = time.time() - exact_start
    print(f"cafein exact mode (10×100): {len(exact)} rows in {exact_seconds:.1f} s")

    if arguments.skip_r5r:
        return
    staging = pathlib.Path(tempfile.mkdtemp(prefix="fare-frontier-bench."))
    try:
        for feed in FEEDS:
            shutil.copy(feed, staging / feed.name)
        shutil.copy(PBF, staging / PBF.name)
        frames = []
        for feed in FEEDS:
            import zipfile

            with zipfile.ZipFile(feed) as archive, archive.open("stops.txt") as file:
                frames.append(pd.read_csv(file, dtype={"stop_id": str}))
        stops = pd.concat(frames, ignore_index=True)
        stops = stops.rename(
            columns={"stop_id": "id", "stop_lat": "lat", "stop_lon": "lon"}
        )[["id", "lat", "lon"]]
        points = staging / "points.csv"
        stops.to_csv(points, index=False)
        bare = [
            origin.split(":", 1)[1] if ":" in origin else origin for origin in origins
        ]
        origins_csv = staging / "origins.csv"
        pd.DataFrame({"id": bare}).to_csv(origins_csv, index=False)
        output = staging / "r5r_frontier.csv"
        script = staging / "run.R"
        script.write_text(R_SCRIPT)
        subprocess.run(
            shlex.split(arguments.rscript)
            + [
                str(script),
                str(staging),
                str(FARES),
                str(points),
                str(origins_csv),
                f"{DATE} {DEPARTURE}",
                ",".join(f"{cutoff:.2f}" for cutoff in CUTOFFS),
                str(output),
            ],
            check=True,
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
