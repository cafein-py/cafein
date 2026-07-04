#!/usr/bin/env python3

"""Benchmark cafein against r5py on the shared Helsinki sample data.

Computes an all-to-all stop-to-stop travel time matrix on both engines
and reports build time, matrix time, throughput, and peak memory. Each
engine runs in its own subprocess so the peak-RSS numbers do not bleed
into each other.

    python scripts/benchmark_vs_r5py.py               # sampled stops
    python scripts/benchmark_vs_r5py.py --stops 0     # every stop in the box
    python scripts/benchmark_vs_r5py.py --engine cafein   # one side only

Requirements: cafein installed (with its compiled core); r5py >= 1.0 and
a Java runtime for the comparison side (`mamba install r5py` provides
both). The test data comes from `python scripts/fetch_test_data.py`.

The comparison is as close as the engines' semantics allow, and the
differences are printed with the results: cafein computes stop-to-stop
medians over a one-minute departure window; r5py computes door-to-door
times for point coordinates (here: the stop locations, so access legs are
near-zero) as the median over the same one-minute window — a single
departure sample on both sides. Both networks take their walking
transfers from the same OSM extract. Stops are restricted to the
extract's coverage so both engines can route them.
"""

import argparse
import datetime
import json
import pathlib
import resource
import subprocess
import sys
import time
import zipfile

DATA = pathlib.Path(__file__).parent.parent / "tests" / "data"
GTFS = DATA / "helsinki_gtfs.zip"
PBF = DATA / "kantakaupunki.osm.pbf"

# The walking network's extent in the kantakaupunki extract.
BBOX = (24.846, 60.145, 25.003, 60.256)

DATE = "2022-02-22"
DEPARTURE = "08:30:00"


def stop_selection(count, seed):
    """Stops inside the extract's coverage, optionally sampled."""
    import pandas as pd

    with zipfile.ZipFile(GTFS) as archive, archive.open("stops.txt") as stops_file:
        stops = pd.read_csv(stops_file, dtype={"stop_id": str})
    west, south, east, north = BBOX
    covered = stops[
        stops["stop_lon"].between(west, east) & stops["stop_lat"].between(south, north)
    ]
    if count and count < len(covered):
        covered = covered.sample(count, random_state=seed)
    return covered[["stop_id", "stop_lat", "stop_lon"]].reset_index(drop=True)


def peak_rss_mb():
    scale = 1024 * 1024 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


def run_cafein(stops):
    import numpy as np

    from cafein import TransportNetwork

    started = time.perf_counter()
    network = TransportNetwork.from_gtfs(
        [str(GTFS)], osm_pbf=str(PBF), trip_distances=False
    )
    build_seconds = time.perf_counter() - started

    started = time.perf_counter()
    matrix = network.travel_time_matrix(
        list(stops["stop_id"]), DATE, DEPARTURE, window=60
    )
    column = {stop_id: at for at, (stop_id, _, _) in enumerate(network.stops)}
    selected = matrix[:, [column[stop_id] for stop_id in stops["stop_id"]], 0]
    finite = int((selected != np.uint32(0xFFFFFFFF)).sum())
    matrix_seconds = time.perf_counter() - started
    return build_seconds, matrix_seconds, finite


def run_r5py(stops):
    import geopandas as gpd
    import r5py

    started = time.perf_counter()
    network = r5py.TransportNetwork(str(PBF), [str(GTFS)])
    build_seconds = time.perf_counter() - started

    points = gpd.GeoDataFrame(
        {"id": stops["stop_id"]},
        geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]),
        crs="EPSG:4326",
    )
    departure = datetime.datetime.fromisoformat(f"{DATE}T{DEPARTURE}")
    started = time.perf_counter()
    matrix = r5py.TravelTimeMatrix(
        network,
        origins=points,
        destinations=points,
        departure=departure,
        departure_time_window=datetime.timedelta(minutes=1),
        transport_modes=[r5py.TransportMode.TRANSIT, r5py.TransportMode.WALK],
    )
    matrix_seconds = time.perf_counter() - started
    return build_seconds, matrix_seconds, int(matrix["travel_time"].notna().sum())


def run_engine(engine, stops):
    build_seconds, matrix_seconds, finite = (
        run_cafein(stops) if engine == "cafein" else run_r5py(stops)
    )
    cells = len(stops) ** 2
    print(
        json.dumps(
            {
                "engine": engine,
                "origins": len(stops),
                "build_seconds": round(build_seconds, 2),
                "matrix_seconds": round(matrix_seconds, 2),
                "origins_per_second": round(len(stops) / matrix_seconds, 1),
                "finite_share": round(finite / cells, 3),
                "peak_rss_mb": round(peak_rss_mb(), 1),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stops",
        type=int,
        default=300,
        help="number of stops to sample (0: every covered stop)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--engine",
        choices=["cafein", "r5py"],
        help="run one engine in this process (default: both, in subprocesses)",
    )
    arguments = parser.parse_args()
    stops = stop_selection(arguments.stops, arguments.seed)

    if arguments.engine:
        run_engine(arguments.engine, stops)
        return

    print(
        f"{len(stops)} stops -> {len(stops) ** 2} OD pairs "
        f"({DATE} {DEPARTURE}, single departure)"
    )
    results = []
    for engine in ["cafein", "r5py"]:
        command = [
            sys.executable,
            __file__,
            "--engine",
            engine,
            "--stops",
            str(arguments.stops),
            "--seed",
            str(arguments.seed),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            detail = completed.stderr.strip().splitlines() or [
                f"exit code {completed.returncode}"
            ]
            print(f"{engine}: FAILED\n{detail[-1]}")
            continue
        results.append(json.loads(completed.stdout.strip().splitlines()[-1]))

    if results:
        keys = [
            "engine",
            "build_seconds",
            "matrix_seconds",
            "origins_per_second",
            "finite_share",
            "peak_rss_mb",
        ]
        widths = {key: max(len(key), 12) for key in keys}
        print("  ".join(key.ljust(widths[key]) for key in keys))
        for row in results:
            print("  ".join(str(row[key]).ljust(widths[key]) for key in keys))
    print(
        "\nnotes: cafein computes stop-to-stop medians over a 1-minute "
        "window;\nr5py computes door-to-door medians over the same "
        "1-minute window from the stop\ncoordinates, including "
        "access/egress snapping. Both use the OSM extract for walking "
        "transfers.\nfinite_share differs accordingly."
    )


if __name__ == "__main__":
    main()
