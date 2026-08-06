"""Measure the bus tier-4 preprocessing budget on the metro fixture.

Runs the whole bus map-matching pass — the street-network read, the
PSV-permission graph build, and the stop-to-stop path resolution for
every deduplicated bus pattern of the GTFS feed — and reports wall
time and peak RSS against the plan's budget (10 minutes, 2 GB
additional peak RSS). The measured phase runs in a child process so
its RSS high-water mark is its own: the parent's pandas work cannot
mask or inflate it. Manual, benchmark-style: needs the local Helsinki
metropolitan extract. Exits nonzero when the budget fails.

    python scripts/measure_bus_tier4.py \\
        --osm tests/data/helsinki-metro.osm.pbf \\
        --gtfs tests/data/helsinki_gtfs.zip
"""

import argparse
import io
import json
import resource
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np
import pandas as pd

BUDGET_SECONDS = 600.0
BUDGET_RSS_BYTES = 2 * 1024**3


def peak_rss():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def bus_patterns(gtfs):
    from cafein import _matching

    with zipfile.ZipFile(gtfs) as archive:
        routes = pd.read_csv(io.BytesIO(archive.read("routes.txt")), dtype=str)
        trips = pd.read_csv(io.BytesIO(archive.read("trips.txt")), dtype=str)
        stop_times = pd.read_csv(
            io.BytesIO(archive.read("stop_times.txt")),
            dtype=str,
            usecols=["trip_id", "stop_id", "stop_sequence"],
        )
        stops = pd.read_csv(
            io.BytesIO(archive.read("stops.txt")), dtype={"stop_id": str}
        )
    kinds = routes["route_type"].astype(int).map(_matching.mode_of)
    bus_routes = set(routes.loc[kinds == "bus", "route_id"])
    of_buses = trips[trips["route_id"].isin(bus_routes)]
    times = stop_times[stop_times["trip_id"].isin(of_buses["trip_id"])].copy()
    times["stop_sequence"] = times["stop_sequence"].astype(int)
    sequences = (
        times.sort_values(["trip_id", "stop_sequence"])
        .groupby("trip_id")["stop_id"]
        .apply(tuple)
    )
    route_of = of_buses.set_index("trip_id")["route_id"]
    coordinates = (
        stops.drop_duplicates("stop_id")
        .set_index("stop_id")[["stop_lat", "stop_lon"]]
        .astype(float)
    )
    unique = {}
    for trip_id, sequence in sequences.items():
        unique.setdefault((route_of[trip_id], sequence), sequence)
    return [coordinates.loc[list(seq)].to_numpy() for seq in unique.values()]


def child(osm, pattern_file):
    """The measured phase: graph build + resolution, in a clean
    process whose RSS high-water mark is its own.

    Builds the bus graph directly — the ladder structurally excludes
    buses from tier 4 while the recorded measurement stands, and this
    script is how the gate is re-evaluated."""
    from cafein import _map_match, _osm_tiers

    archive = np.load(pattern_file)
    flat, offsets = archive["flat"], archive["offsets"]
    patterns = [flat[start:stop] for start, stop in zip(offsets[:-1], offsets[1:])]
    baseline = peak_rss()  # imports + the compact pattern arrays
    started = time.monotonic()
    ladder = _osm_tiers.OsmLadder(osm, frozenset({"bus"}))
    ladder._set_projection(patterns[0])
    graph = _map_match.bus_graph(*ladder._streets(), ladder._project)
    built = time.monotonic()
    resolved = 0
    for latlon in patterns:
        stop_xy = ladder._project(latlon[:, 1], latlon[:, 0])
        if _map_match.match_chain(graph, stop_xy) is not None:
            resolved += 1
    print(
        json.dumps(
            {
                "vertices": len(graph._xy),
                "edges": int(graph._matrix.nnz),
                "build_seconds": built - started,
                "total_seconds": time.monotonic() - started,
                "added_rss": peak_rss() - baseline,
                "resolved": resolved,
                "patterns": len(patterns),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--osm", required=True)
    parser.add_argument("--gtfs")
    parser.add_argument("--child-patterns")
    arguments = parser.parse_args()
    if arguments.child_patterns:
        child(arguments.osm, arguments.child_patterns)
        return

    patterns = bus_patterns(arguments.gtfs)
    print(f"{len(patterns)} deduplicated bus patterns")
    with tempfile.NamedTemporaryFile(suffix=".npz") as handle:
        offsets = np.cumsum([0] + [len(p) for p in patterns])
        np.savez(handle.name, flat=np.vstack(patterns), offsets=offsets)
        run = subprocess.run(
            [
                sys.executable,
                __file__,
                "--osm",
                arguments.osm,
                "--child-patterns",
                handle.name,
            ],
            capture_output=True,
            text=True,
        )
    if run.returncode != 0:
        sys.stderr.write(run.stderr)
        sys.exit(run.returncode)
    report = json.loads(run.stdout.strip().splitlines()[-1])
    print(
        f"bus graph: {report['vertices']} vertices, {report['edges']} edges, "
        f"built in {report['build_seconds']:.1f}s"
    )
    print(f"resolved {report['resolved']}/{report['patterns']} patterns")
    print(f"total {report['total_seconds']:.1f}s (budget {BUDGET_SECONDS:.0f}s)")
    print(
        f"added peak RSS {report['added_rss'] / 1024**3:.2f} GiB " "(budget 2.00 GiB)"
    )
    passed = (
        report["total_seconds"] <= BUDGET_SECONDS
        and report["added_rss"] <= BUDGET_RSS_BYTES
    )
    print("budget: " + ("PASS" if passed else "FAIL"))
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
