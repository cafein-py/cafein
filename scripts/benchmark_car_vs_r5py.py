#!/usr/bin/env python3

"""Car benchmark on the Helsinki fixture: r5py parity and delay realism.

Two runs, never conflated:

``--run parity`` (the default) compares cafein's **default car regime** —
free-flow, speed-limit-based, no delay model, no parking — against
r5py's CAR mode, door-to-door over one sampled point set. Both engines
read the **same restriction-free PBF**: the harness writes it with
pyrosm alone (``OSM.write_pbf(..., delete=[("relation", id), ...])``
over every ``type=restriction`` relation), so R5's turn-restriction
handling is neutralised as a difference. cafein's untagged-way speed
defaults are aligned to R5's documented per-class defaults through
``speed_limits=`` for this run only. Reported per engine: build and
matrix wall times; for the comparison: the share of finite cells, the
share agreeing within ±60 s, and the median and p90 absolute
difference (r5py reports whole minutes, so times compare in minutes).

``--run realism`` needs cafein only: the intersection-delay model
(``intersection_delays=True, profile="rush"``) against the free-flow
default over the same pairs, reporting the ratio distribution — a
validation snapshot of the 2009-calibrated values on today's network,
not a recalibration.

Residual, uncontrolled differences of the parity run, stated with the
results: the engines' snapping (R5 links points by its own street-layer
rules — cafein's ``--snap`` default matches R5's documented 1,600 m
street-layer link radius, though the algorithms still differ),
R5's intersection costs (R5 applies small turn costs of its own), and
rounding (R5 reports whole minutes).

    python scripts/benchmark_car_vs_r5py.py                # parity
    python scripts/benchmark_car_vs_r5py.py --run realism  # cafein only
    python scripts/benchmark_car_vs_r5py.py --engine cafein  # one side

Requirements: cafein in the invoking interpreter (pyrosm >= 0.12 for the
restriction stripping); r5py >= 1.0 for the comparison side —
``--r5py-python`` names its interpreter (default: the local
``sustainability-gis`` env, whose bundled JVM is used as JAVA_HOME).
The test data comes from ``python scripts/fetch_test_data.py``.
"""

import argparse
import datetime
import json
import os
import pathlib
import subprocess
import sys
import time

DATA = pathlib.Path(__file__).parent.parent / "tests" / "data"
PBF = DATA / "kantakaupunki.osm.pbf"
STRIPPED = DATA / "kantakaupunki-no-restrictions.osm.pbf"

# The walking network's extent in the kantakaupunki extract; points are
# sampled inside a margin so both engines can snap them.
BBOX = (24.846, 60.145, 25.003, 60.256)
MARGIN = 0.015

DATE = "2022-02-22"
DEPARTURE = "08:30:00"

# R5's documented default car speeds per highway class where OSM carries
# no maxspeed (SpeedConfig defaults, mph converted to km/h). The parity
# run aligns cafein's untagged-way defaults to these via `speed_limits=`;
# tagged ways read the same maxspeed on both engines.
R5_DEFAULT_KMH = {
    "motorway": 65 * 1.609344,
    "motorway_link": 35 * 1.609344,
    "trunk": 55 * 1.609344,
    "trunk_link": 30 * 1.609344,
    "primary": 45 * 1.609344,
    "primary_link": 25 * 1.609344,
    "secondary": 40 * 1.609344,
    "secondary_link": 20 * 1.609344,
    "tertiary": 35 * 1.609344,
    "tertiary_link": 20 * 1.609344,
    "unclassified": 25 * 1.609344,
    "residential": 25 * 1.609344,
    "living_street": 10 * 1.609344,
    "service": 15 * 1.609344,
    "track": 10 * 1.609344,
    "other": 25 * 1.609344,
}


def r5_aligned_speed_limits():
    """The `speed_limits=` override mirroring R5's class defaults.

    Every class the cafein table prices is overridden inside and outside
    urban areas alike, so the two engines differ only where a way
    carries its own maxspeed tag (read identically by both).
    """
    from cafein._speed_limits import SPEED_LIMITS

    classes = set()
    for row in SPEED_LIMITS.values():
        classes.update(row)
    overrides = {}
    unmatched = set()
    for column in classes:
        base = column
        for suffix in ("_inside", "_outside"):
            if column.endswith(suffix):
                base = column[: -len(suffix)]
                break
        if base not in R5_DEFAULT_KMH:
            # `other_*` is cafein's own fallback class; `unclassified_link`
            # has no R5 counterpart. Anything else means the vocabularies
            # drifted and the alignment silently broke.
            if not column.startswith("other") and column != "unclassified_link":
                unmatched.add(column)
            base = "other"
        overrides[column] = R5_DEFAULT_KMH[base]
    if unmatched:
        raise SystemExit(f"classes without an R5 default: {sorted(unmatched)}")
    overrides["track"] = R5_DEFAULT_KMH["track"]
    return overrides


def sample_points(count, seed):
    """Deterministic points inside the extract, as (id, lat, lon) rows."""
    import random

    west, south, east, north = BBOX
    generator = random.Random(seed)
    rows = []
    for index in range(count):
        rows.append(
            (
                f"point-{index}",
                generator.uniform(south + MARGIN, north - MARGIN),
                generator.uniform(west + MARGIN, east - MARGIN),
            )
        )
    return rows


def prepare_stripped_pbf():
    """The restriction-free PBF, written with pyrosm alone.

    Written atomically beside a sidecar that ties the cache to the exact
    source (size and mtime) and records the removed-relation count; any
    mismatch regenerates rather than trusting a stale or foreign file.
    """
    import hashlib

    import pyrosm
    from pyrosm import OSM

    digest = hashlib.sha256(PBF.read_bytes()).hexdigest()
    source = {"sha256": digest, "pyrosm": pyrosm.__version__}
    sidecar = STRIPPED.with_suffix(".json")
    if STRIPPED.exists() and sidecar.exists():
        try:
            recorded = json.loads(sidecar.read_text())
        except ValueError:
            recorded = None
        if recorded and recorded.get("source") == source:
            print(
                f"reusing {STRIPPED.name} "
                f"({recorded['removed_restrictions']} restriction(s) removed)"
            )
            return

    def restriction_ids(osm):
        relations = osm._relations or {}
        if "id" not in relations:
            return []
        return [
            int(identifier)
            for identifier, tags in zip(relations["id"], relations["tags"])
            if isinstance(tags, dict) and tags.get("type") == "restriction"
        ]

    osm = OSM(str(PBF))
    edges = osm.get_network(network_type="driving")
    restrictions = restriction_ids(osm)
    staging = STRIPPED.with_name(
        STRIPPED.name.replace(".osm.pbf", f".staging-{os.getpid()}.osm.pbf")
    )
    osm.write_pbf(edges, str(staging), delete=[("relation", r) for r in restrictions])
    # Verified, not assumed: the written file re-reads with the identical
    # drive network and zero restriction relations before it is adopted.
    check = OSM(str(staging))
    stripped_edges = check.get_network(network_type="driving")
    if len(stripped_edges) != len(edges) or restriction_ids(check):
        staging.unlink()
        raise SystemExit("stripped PBF failed verification; not adopting it")
    staging.replace(STRIPPED)
    sidecar.write_text(
        json.dumps({"source": source, "removed_restrictions": len(restrictions)})
    )
    print(f"wrote {STRIPPED.name}: {len(restrictions)} restriction relation(s) removed")


def run_cafein(points, snap, realism):
    import geopandas as gpd

    from cafein import StreetNetwork, TravelTimeMatrix

    started = time.perf_counter()
    network = StreetNetwork.from_osm(
        str(STRIPPED),
        modes=("walk", "car"),
        country="FI",
        speed_limits=r5_aligned_speed_limits() if not realism else None,
    )
    build_seconds = time.perf_counter() - started
    frame = gpd.GeoDataFrame(
        {"id": [identifier for identifier, _, _ in points]},
        geometry=gpd.points_from_xy(
            [lon for _, _, lon in points], [lat for _, lat, _ in points]
        ),
        crs="EPSG:4326",
    )

    def matrix(**options):
        begun = time.perf_counter()
        cells = TravelTimeMatrix(
            network,
            frame,
            frame,
            transport_mode="car",
            max_snap_distance=snap,
            **options,
        )
        return cells, time.perf_counter() - begun

    free, matrix_seconds = matrix()
    result = {
        "engine": "cafein",
        "build_seconds": round(build_seconds, 2),
        "matrix_seconds": round(matrix_seconds, 3),
        "cells": {
            f"{row.from_id}|{row.to_id}": int(row.travel_time_s)
            for row in free.itertuples()
        },
    }
    if realism:
        rush, rush_seconds = matrix(intersection_delays=True, profile="rush")
        result["rush_matrix_seconds"] = round(rush_seconds, 3)
        result["rush_cells"] = {
            f"{row.from_id}|{row.to_id}": int(row.travel_time_s)
            for row in rush.itertuples()
        }
    print(json.dumps(result))


def run_r5py(points):
    import geopandas as gpd
    import r5py
    from r5py.r5 import travel_time_matrix as ttm_module

    # r5py 1.0.0dev × current geopandas: internal frame operations
    # reconstruct the TravelTimeMatrix subclass through its computing
    # __init__; route the reconstruction to a plain GeoDataFrame.
    ttm_module.TravelTimeMatrix._geodataframe_constructor_with_fallback = classmethod(
        lambda cls, *args, **kwargs: gpd.GeoDataFrame(*args, **kwargs)
    )

    started = time.perf_counter()
    network = r5py.TransportNetwork(str(STRIPPED), [])
    build_seconds = time.perf_counter() - started
    frame = gpd.GeoDataFrame(
        {"id": [identifier for identifier, _, _ in points]},
        geometry=gpd.points_from_xy(
            [lon for _, _, lon in points], [lat for _, lat, _ in points]
        ),
        crs="EPSG:4326",
    )
    departure = datetime.datetime.fromisoformat(f"{DATE}T{DEPARTURE}")
    begun = time.perf_counter()
    matrix = r5py.TravelTimeMatrix(
        network,
        origins=frame,
        destinations=frame,
        departure=departure,
        transport_modes=[r5py.TransportMode.CAR],
    )
    matrix_seconds = time.perf_counter() - begun
    cells = {}
    for row in matrix.dropna(subset=["travel_time"]).itertuples():
        cells[f"{row.from_id}|{row.to_id}"] = int(row.travel_time)
    print(
        json.dumps(
            {
                "engine": "r5py",
                "build_seconds": round(build_seconds, 2),
                "matrix_seconds": round(matrix_seconds, 3),
                "cells": cells,
            }
        )
    )


def compare(cafein_result, r5py_result, points):
    """The parity numbers: coverage, agreement, and the difference shape."""
    total = len(points) ** 2
    cafein_cells = cafein_result["cells"]
    r5py_cells = r5py_result["cells"]
    shared = sorted(set(cafein_cells) & set(r5py_cells))
    differences = [
        cafein_cells[key] / 60.0 - r5py_cells[key] for key in shared
    ]  # minutes; r5py reports whole minutes
    absolute = sorted(abs(value) for value in differences)
    within = sum(1 for value in absolute if value <= 1.0)
    print(f"\nparity over {total} OD pairs ({len(points)} points):")
    print(
        f"  finite cells: cafein {len(cafein_cells) / total:.3f}, "
        f"r5py {len(r5py_cells) / total:.3f}, shared {len(shared)}"
    )
    if not shared:
        raise SystemExit("no shared finite cells; nothing to compare")
    bias = sum(differences) / len(shared)
    print(
        f"  within ±60 s: {within / len(shared):.3f} | "
        f"median |diff| {quantile(absolute, 0.5):.2f} min | "
        f"p90 |diff| {quantile(absolute, 0.9):.2f} min | "
        f"bias {bias:+.2f} min"
    )
    for result in (cafein_result, r5py_result):
        print(
            f"  {result['engine']}: build {result['build_seconds']} s, "
            f"matrix {result['matrix_seconds']} s"
        )


def quantile(values, share):
    """The linearly interpolated `share` quantile of sorted `values`."""
    if not values:
        return float("nan")
    position = share * (len(values) - 1)
    below = int(position)
    above = min(below + 1, len(values) - 1)
    return values[below] + (values[above] - values[below]) * (position - below)


def report_realism(result, points):
    free = result["cells"]
    rush = result["rush_cells"]
    shared = sorted(set(free) & set(rush))
    ratios = sorted(rush[key] / free[key] for key in shared if free[key] > 0)
    if not ratios:
        raise SystemExit("no reachable pairs with positive times; nothing to report")
    slower = sum(1 for ratio in ratios if ratio > 1.0)
    mean = sum(ratios) / len(ratios)
    print(
        f"\nrealism over {len(ratios)} reachable positive-time pairs "
        f"({len(points)} points):"
    )
    print(
        f"  rush/free-flow ratio: median {quantile(ratios, 0.5):.2f}, "
        f"mean {mean:.2f}, p10 {quantile(ratios, 0.1):.2f}, "
        f"p90 {quantile(ratios, 0.9):.2f}, "
        f"share > 1: {slower / len(ratios):.3f}"
    )
    print(
        f"  build {result['build_seconds']} s | free-flow matrix "
        f"{result['matrix_seconds']} s | rush matrix "
        f"{result['rush_matrix_seconds']} s"
    )


def default_r5py_python():
    candidate = pathlib.Path.home() / "mamba" / "envs" / "sustainability-gis"
    return str(candidate / "bin" / "python") if candidate.is_dir() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", choices=["parity", "realism"], default="parity")

    def at_least_two(value):
        count = int(value)
        if count < 2:
            raise argparse.ArgumentTypeError("--points needs at least 2")
        return count

    parser.add_argument("--points", type=at_least_two, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--snap",
        type=float,
        default=1600.0,
        help="cafein snap radius, metres (default: R5's 1600 m link radius)",
    )
    parser.add_argument(
        "--engine",
        choices=["cafein", "r5py"],
        help="run one engine in this process (internal; default: orchestrate)",
    )
    parser.add_argument("--r5py-python", default=default_r5py_python())
    arguments = parser.parse_args()
    points = sample_points(arguments.points, arguments.seed)

    if arguments.engine == "cafein":
        prepare_stripped_pbf()
        run_cafein(points, arguments.snap, realism=arguments.run == "realism")
        return
    if arguments.engine == "r5py":
        if not STRIPPED.exists():
            raise SystemExit(
                f"{STRIPPED.name} is missing; run the cafein side (or the "
                "orchestrator) first — the r5py environment need not carry "
                "pyrosm"
            )
        run_r5py(points)
        return

    prepare_stripped_pbf()
    passthrough = [
        "--run",
        arguments.run,
        "--points",
        str(arguments.points),
        "--seed",
        str(arguments.seed),
        "--snap",
        str(arguments.snap),
    ]

    def run_side(engine, interpreter, environment=None):
        command = [interpreter, __file__, "--engine", engine, *passthrough]
        completed = subprocess.run(
            command, capture_output=True, text=True, env=environment
        )
        if completed.returncode != 0:
            tail = (completed.stderr.strip().splitlines() or ["no stderr"])[-1]
            raise SystemExit(f"{engine} side failed: {tail}")
        return json.loads(completed.stdout.strip().splitlines()[-1])

    cafein_result = run_side("cafein", sys.executable)
    if arguments.run == "realism":
        report_realism(cafein_result, points)
        return
    if not arguments.r5py_python:
        raise SystemExit("no r5py interpreter found; pass --r5py-python")
    environment = dict(os.environ)
    jvm = pathlib.Path(arguments.r5py_python).parent.parent / "lib" / "jvm"
    if jvm.is_dir():
        environment["JAVA_HOME"] = str(jvm)
    r5py_result = run_side("r5py", arguments.r5py_python, environment)
    compare(cafein_result, r5py_result, points)


if __name__ == "__main__":
    main()
