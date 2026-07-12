#!/usr/bin/env python3

"""Benchmark cafein on a 250 m grid over the Helsinki capital region
(Helsinki, Espoo, Vantaa, Kauniainen), and compare its travel-time matrix
against r5py on the same workload.

The region's walking network is the ``helsinki_net.osm.pbf`` extract; a
250 m grid (EPSG:3067) is overlaid on its bounding box and clipped to
cells within 250 m of the walking network, then reprojected to EPSG:4326
for routing, so only reachable land cells remain. The matrix workload is
``K`` sampled origin cells to every grid cell (a full all-to-all 250 m grid
is hundreds of millions of pairs); it scales linearly in origins, so
per-origin throughput extrapolates to the whole grid.

Each benchmark is a separate subcommand so it builds its own network,
reports an isolated peak RSS, and appends one tagged result line to
``results.jsonl``:

    prep          build + save the grid and the origin sample
    cafein-time   cafein TravelTimeMatrix   (vs r5py-time)
    r5py-time     r5py   TravelTimeMatrix   (vs cafein-time)
    cafein-cost   cafein TravelCostMatrix   (time + distance + emissions)
    cafein-pareto cafein journey_frontier   (Pareto time x emissions, sample)

The cafein side must run on a release extension — build it with
``maturin develop --release``; the default ``maturin develop`` produces an
unoptimized dev-profile module whose timings are meaningless.
"""

import argparse
import datetime
import json
import os
import pathlib
import resource
import sys
import tempfile
import time

DATA = pathlib.Path(__file__).parent.parent / "tests" / "data"
GTFS = DATA / "helsinki_gtfs.zip"
PBF = DATA / "helsinki_net.osm.pbf"
GRID_DIR = pathlib.Path(
    os.environ.get("GRID_BENCH_DIR")
    or (pathlib.Path(tempfile.gettempdir()) / "cafein-grid-bench")
)
GRID_ALL = GRID_DIR / "grid_all.parquet"
ORIGINS = GRID_DIR / "origins.parquet"
META = GRID_DIR / "meta.json"
RESULTS = GRID_DIR / "results.jsonl"

DATE = "2022-02-22"
DEPARTURE = "08:30:00"
CELL_M = 250.0
SNAP_M = 250.0

# Routing parameters held identical across engines so the travel-time
# comparison isolates engine performance, not parameter choices. r5py cannot
# route a single departure (it enforces a >=5 min window), so both engines run
# a 5 min departure window and report the median (50th percentile).
WINDOW_S = 300
PERCENTILE = 50
MAX_TRANSFERS = 7  # cafein transfers == r5py max_public_transport_rides - 1
WALK_KMPH = 3.6
MAX_WALK_S = 1800
SNAP_DIST_M = 1600

SHARED_PARAMS = {
    "departure": f"{DATE} {DEPARTURE}",
    "window_s": WINDOW_S,
    "percentile": PERCENTILE,
    "max_transfers": MAX_TRANSFERS,
    "walk_kmph": WALK_KMPH,
    "max_walk_s": MAX_WALK_S,
    "snap_dist_m": SNAP_DIST_M,
}


def peak_rss_mb():
    scale = 1024 * 1024 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


def _record(result):
    # Stamp every product result with the grid it ran against, so the whole
    # run can be verified to have used one consistent point set.
    if result.get("benchmark") != "prep":
        meta = _grid_meta()
        result["grid"] = {
            "cells_reachable": meta.get("grid_cells_reachable"),
            "cell_m": meta.get("cell_m"),
            "snap_m": meta.get("snap_m"),
            "clip_target": meta.get("clip_target"),
            "prep_stamp": meta.get("prep_stamp"),
        }
        # cafein parallelism is RAYON_NUM_THREADS; r5py/R5 uses all cores. Record
        # it so the travel-time comparison is at a known, matched thread count.
        result["threads"] = int(os.environ.get("RAYON_NUM_THREADS") or os.cpu_count())
    result["stamp"] = datetime.datetime.now().isoformat(timespec="seconds")
    with open(RESULTS, "a") as handle:
        handle.write(json.dumps(result) + "\n")
    print(json.dumps(result))


def _grid_meta():
    if META.exists():
        with open(META) as handle:
            return json.load(handle)
    return {}


def _load_points():
    import geopandas as gpd

    if not GRID_ALL.exists():
        raise SystemExit(f"grid not built; run `prep` first ({GRID_ALL} missing)")
    grid = gpd.read_parquet(GRID_ALL)
    origins = gpd.read_parquet(ORIGINS)
    return grid, origins


def prep(origin_count, seed):
    import geopandas as gpd
    import numpy as np
    from pyrosm import OSM
    from shapely.geometry import Point

    GRID_DIR.mkdir(parents=True, exist_ok=True)
    osm = OSM(str(PBF))
    # Clip candidate cells to the walking-network *edges* (line geometries), not
    # only the vertices: a cell can sit next to a long road segment yet far from
    # its nearest node, and should still count as reachable.
    _, edges = osm.get_network("walking", nodes=True)
    edges = edges[["geometry"]].to_crs(3067)
    minx, miny, maxx, maxy = edges.total_bounds

    xs = np.arange(minx + CELL_M / 2, maxx, CELL_M)
    ys = np.arange(miny + CELL_M / 2, maxy, CELL_M)
    gx, gy = np.meshgrid(xs, ys)
    centroids = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in zip(gx.ravel(), gy.ravel())], crs=3067
    )
    total = len(centroids)
    near = gpd.sjoin_nearest(centroids, edges, max_distance=SNAP_M, how="inner")
    near = near[~near.index.duplicated(keep="first")].reset_index(drop=True)
    grid = near[["geometry"]].to_crs(4326)
    grid.insert(0, "id", [f"c{i}" for i in range(len(grid))])
    grid.to_parquet(GRID_ALL)
    origins = grid.sample(min(origin_count, len(grid)), random_state=seed)
    origins.reset_index(drop=True).to_parquet(ORIGINS)

    meta = {
        "cell_m": CELL_M,
        "snap_m": SNAP_M,
        "clip_target": "walking_edges",
        "bbox_3067": [round(v, 1) for v in (minx, miny, maxx, maxy)],
        "grid_cells_total": int(total),
        "grid_cells_reachable": int(len(grid)),
        "origins": int(len(origins)),
        "seed": seed,
        "gtfs": GTFS.name,
        "pbf": PBF.name,
        "prep_stamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(META, "w") as handle:
        json.dump(meta, handle)
    _record({"benchmark": "prep", "full_grid_pairs": int(len(grid)) ** 2, **meta})


def _build_cafein(trip_distances):
    from cafein import TransportNetwork

    started = time.perf_counter()
    network = TransportNetwork.from_gtfs(
        [str(GTFS)], osm_pbf=str(PBF), trip_distances=trip_distances
    )
    return network, round(time.perf_counter() - started, 2), round(peak_rss_mb(), 1)


def _time_column(frame):
    return next(c for c in frame.columns if str(c).startswith("travel_time"))


def cafein_time():
    from cafein import TravelTimeMatrix

    grid, origins = _load_points()
    # Travel time needs no trip distances; build the minimal network r5py's
    # time product is comparable to.
    network, build_seconds, build_rss = _build_cafein(trip_distances=False)
    started = time.perf_counter()
    matrix = TravelTimeMatrix(
        network,
        origins=origins,
        destinations=grid,
        date=DATE,
        departure=DEPARTURE,
        window=WINDOW_S,
        percentiles=[PERCENTILE],
        max_transfers=MAX_TRANSFERS,
        walking_speed_kmph=WALK_KMPH,
        max_walking_time=MAX_WALK_S,
        max_snap_distance=SNAP_DIST_M,
    )
    seconds = time.perf_counter() - started
    pairs = len(origins) * len(grid)
    reachable = int(matrix[_time_column(matrix)].notna().sum())
    _record(
        {
            "benchmark": "cafein-time",
            "engine": "cafein",
            "product": "travel_time",
            "build_config": "trip_distances=False",
            "params": SHARED_PARAMS,
            "origins": int(len(origins)),
            "destinations": int(len(grid)),
            "od_pairs": int(pairs),
            "build_seconds": build_seconds,
            "build_rss_mb": build_rss,
            "matrix_seconds": round(seconds, 2),
            "origins_per_second": round(len(origins) / seconds, 1),
            "pairs_per_second": round(pairs / seconds, 0),
            "reachable_cells": reachable,
            "peak_rss_mb": round(peak_rss_mb(), 1),
        }
    )


def r5py_time():
    import r5py

    grid, origins = _load_points()
    origins = origins[["id", "geometry"]]
    dests = grid[["id", "geometry"]]
    started = time.perf_counter()
    network = r5py.TransportNetwork(str(PBF), [str(GTFS)])
    build_seconds = round(time.perf_counter() - started, 2)
    build_rss = round(peak_rss_mb(), 1)
    departure = datetime.datetime.fromisoformat(f"{DATE}T{DEPARTURE}")
    started = time.perf_counter()
    matrix = r5py.TravelTimeMatrix(
        network,
        origins=origins,
        destinations=dests,
        departure=departure,
        departure_time_window=datetime.timedelta(seconds=WINDOW_S),
        percentiles=[PERCENTILE],
        transport_modes=[r5py.TransportMode.TRANSIT, r5py.TransportMode.WALK],
        speed_walking=WALK_KMPH,
        max_public_transport_rides=MAX_TRANSFERS + 1,
        max_time_walking=datetime.timedelta(seconds=MAX_WALK_S),
    )
    seconds = time.perf_counter() - started
    pairs = len(origins) * len(dests)
    reachable = int(matrix[_time_column(matrix)].notna().sum())
    _record(
        {
            "benchmark": "r5py-time",
            "engine": "r5py",
            "product": "travel_time",
            "build_config": "R5 default (no trip distances)",
            "params": {
                **SHARED_PARAMS,
                "max_public_transport_rides": MAX_TRANSFERS + 1,
                "snap": "r5py default LINK_RADIUS (~1600 m)",
                "note": "r5py caps total trip at max_time=2 h; cafein has no total-time cap",
            },
            "origins": int(len(origins)),
            "destinations": int(len(dests)),
            "od_pairs": int(pairs),
            "build_seconds": build_seconds,
            "build_rss_mb": build_rss,
            "matrix_seconds": round(seconds, 2),
            "origins_per_second": round(len(origins) / seconds, 1),
            "pairs_per_second": round(pairs / seconds, 0),
            "reachable_cells": reachable,
            "peak_rss_mb": round(peak_rss_mb(), 1),
        }
    )


def _single_params():
    # The distance/emissions cost matrix is the canonical single-departure
    # earliest-arrival query (a windowed cost matrix would optimise emissions,
    # a different product). cafein-time-single shares these settings so the two
    # differ only in the distance/emissions work.
    params = {
        k: v for k, v in SHARED_PARAMS.items() if k not in ("window_s", "percentile")
    }
    params["departure_mode"] = "single (08:30, earliest arrival)"
    return params


def cafein_time_single():
    """Single-departure time-only baseline; paired with cafein-cost so their
    difference is the pure cost of computing distances and emissions."""
    from cafein import TravelTimeMatrix

    grid, origins = _load_points()
    network, build_seconds, build_rss = _build_cafein(trip_distances=False)
    started = time.perf_counter()
    matrix = TravelTimeMatrix(
        network,
        origins=origins,
        destinations=grid,
        date=DATE,
        departure=DEPARTURE,
        max_transfers=MAX_TRANSFERS,
        walking_speed_kmph=WALK_KMPH,
        max_walking_time=MAX_WALK_S,
        max_snap_distance=SNAP_DIST_M,
    )
    seconds = time.perf_counter() - started
    pairs = len(origins) * len(grid)
    _record(
        {
            "benchmark": "cafein-time-single",
            "engine": "cafein",
            "product": "travel_time",
            "build_config": "trip_distances=False",
            "params": _single_params(),
            "origins": int(len(origins)),
            "destinations": int(len(grid)),
            "od_pairs": int(pairs),
            "build_seconds": build_seconds,
            "build_rss_mb": build_rss,
            "matrix_seconds": round(seconds, 2),
            "pairs_per_second": round(pairs / seconds, 0),
            "reachable_cells": int(matrix[_time_column(matrix)].notna().sum()),
            "peak_rss_mb": round(peak_rss_mb(), 1),
        }
    )


def cafein_cost():
    """The extra-work benchmark: travel time + distance + emissions in one
    matrix, single departure (pairs with cafein-time-single)."""
    from cafein import TravelCostMatrix

    grid, origins = _load_points()
    network, build_seconds, build_rss = _build_cafein(trip_distances=True)
    started = time.perf_counter()
    matrix = TravelCostMatrix(
        network,
        origins=origins,
        destinations=grid,
        date=DATE,
        departure=DEPARTURE,
        max_transfers=MAX_TRANSFERS,
        walking_speed_kmph=WALK_KMPH,
        max_walking_time=MAX_WALK_S,
        max_snap_distance=SNAP_DIST_M,
    )
    seconds = time.perf_counter() - started
    pairs = len(origins) * len(grid)
    _record(
        {
            "benchmark": "cafein-cost",
            "engine": "cafein",
            "product": "time+distance+emissions",
            "build_config": "trip_distances=True",
            "params": _single_params(),
            "origins": int(len(origins)),
            "destinations": int(len(grid)),
            "od_pairs": int(pairs),
            "build_seconds": build_seconds,
            "build_rss_mb": build_rss,
            "matrix_seconds": round(seconds, 2),
            "pairs_per_second": round(pairs / seconds, 0),
            "reachable_cells": int(matrix["travel_time"].notna().sum()),
            "emissions_finite": int(matrix["emissions"].notna().sum()),
            "peak_rss_mb": round(peak_rss_mb(), 1),
        }
    )


def cafein_pareto(pareto_pairs):
    from cafein.frontier import journey_frontier

    grid, origins = _load_points()
    network, build_seconds, build_rss = _build_cafein(trip_distances=True)
    n_origins = max(1, pareto_pairs // 40)
    sample_origins = origins.iloc[:n_origins]
    sample_dests = grid.sample(min(40, len(grid)), random_state=7)
    pairs = 0
    rows = 0
    started = time.perf_counter()
    for _, o in sample_origins.iterrows():
        for _, d in sample_dests.iterrows():
            frame = journey_frontier(
                network,
                (o.geometry.y, o.geometry.x),
                (d.geometry.y, d.geometry.x),
                DATE,
                DEPARTURE,
                window=WINDOW_S,
                candidates="pareto",
                max_transfers=MAX_TRANSFERS,
                walking_speed_kmph=WALK_KMPH,
                max_walking_time=MAX_WALK_S,
                max_snap_distance=SNAP_DIST_M,
            )
            pairs += 1
            rows += len(frame)
            if pairs >= pareto_pairs:
                break
        if pairs >= pareto_pairs:
            break
    seconds = time.perf_counter() - started
    _record(
        {
            "benchmark": "cafein-pareto",
            "engine": "cafein",
            "product": "pareto_time_x_emissions",
            "build_config": "trip_distances=True",
            "params": {**SHARED_PARAMS, "note": "cafein-only; no r5py equivalent"},
            "pairs": pairs,
            "build_seconds": build_seconds,
            "build_rss_mb": build_rss,
            "pareto_seconds": round(seconds, 2),
            "pairs_per_second": round(pairs / seconds, 2),
            "ms_per_pair": round(seconds / pairs * 1000, 1),
            "frontier_rows": rows,
            "peak_rss_mb": round(peak_rss_mb(), 1),
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "command",
        choices=[
            "prep",
            "cafein-time",
            "r5py-time",
            "cafein-time-single",
            "cafein-cost",
            "cafein-pareto",
        ],
    )
    parser.add_argument("--origins", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pareto-pairs", type=int, default=120)
    args = parser.parse_args()
    if args.command == "prep":
        prep(args.origins, args.seed)
    elif args.command == "cafein-time":
        cafein_time()
    elif args.command == "r5py-time":
        r5py_time()
    elif args.command == "cafein-time-single":
        cafein_time_single()
    elif args.command == "cafein-cost":
        cafein_cost()
    elif args.command == "cafein-pareto":
        cafein_pareto(args.pareto_pairs)


if __name__ == "__main__":
    main()
