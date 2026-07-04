#!/usr/bin/env python3

"""Compare cafein against r5py across the functionality the two share.

For every shared scenario — a travel-time matrix and detailed itineraries —
this runs both engines on the shared Helsinki sample data and records, per
engine, the network build time, the compute time, and peak memory, then reports
how closely the two agree on the results. Each engine runs in its own
subprocess so peak-RSS figures do not bleed into each other (r5py loads a JVM;
cafein does not). Results are printed as tables and, with ``--csv``, written to
a file.

    python scripts/compare_vs_r5py.py                     # both engines, all scenarios
    python scripts/compare_vs_r5py.py --origins 40        # matrix sample size
    python scripts/compare_vs_r5py.py --engine cafein     # one engine only
    python scripts/compare_vs_r5py.py --scenario travel_time_matrix
    python scripts/compare_vs_r5py.py --csv comparison.csv

Requirements: cafein installed (with its compiled core); r5py >= 1.0 and a Java
runtime for the comparison side (``mamba install r5py`` provides both). The test
data comes from ``python scripts/fetch_test_data.py``.

The comparison is as close as the engines' semantics allow: both route
door-to-door from the same points (the stops' own coordinates, so access legs
are near-zero), from the same departure over the same one-minute window, taking
walking transfers from the same OSM extract. Travel times are compared in
minutes (r5py's unit; cafein's seconds are divided by 60). The r5py itinerary
summary follows r5py 1.x column names — adjust ``_summarize_itineraries_r5py``
if your r5py version differs.
"""

import argparse
import datetime
import json
import pathlib
import subprocess
import sys
import tempfile
import time
import zipfile

try:
    import resource
except ImportError:  # resource is Unix-only; peak RSS is unavailable on Windows
    resource = None

DATA = pathlib.Path(__file__).parent.parent / "tests" / "data"
GTFS = DATA / "helsinki_gtfs.zip"
PBF = DATA / "kantakaupunki.osm.pbf"

# The walking network's extent in the kantakaupunki extract.
BBOX = (24.846, 60.145, 25.003, 60.256)
# A central, densely-connected sub-area for the door-to-door itineraries,
# whose per-OD searches raise (rather than warn) on a stop off the network.
ITINERARY_BBOX = (24.90, 60.16, 25.00, 60.20)

DATE = "2022-02-22"
DEPARTURE = "08:30:00"
WINDOW_SECONDS = 60
WALKING_SPEED_KMPH = 3.6
# Detailed itineraries are one search per OD pair, so they run over a small
# central corner of the stop sample.
ITINERARY_POINTS = 5
ITINERARY_SNAP_METERS = 250.0
# Travel times within this many minutes count as agreeing.
TOLERANCE_MINUTES = 1.0

RESULT_PREFIX = "RESULT_JSON:"


def peak_rss_mb():
    if resource is None:
        return float("nan")
    scale = 1024 * 1024 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


def stop_points(count, seed, bbox=BBOX):
    """A point GeoDataFrame at sampled stops inside a bounding box."""
    import geopandas as gpd
    import pandas as pd

    with zipfile.ZipFile(GTFS) as archive, archive.open("stops.txt") as stops_file:
        stops = pd.read_csv(stops_file, dtype={"stop_id": str})
    west, south, east, north = bbox
    covered = stops[
        stops["stop_lon"].between(west, east) & stops["stop_lat"].between(south, north)
    ]
    if count and count < len(covered):
        covered = covered.sample(count, random_state=seed)
    covered = covered.reset_index(drop=True)
    return gpd.GeoDataFrame(
        {"id": covered["stop_id"].astype(str)},
        geometry=gpd.points_from_xy(covered["stop_lon"], covered["stop_lat"]),
        crs="EPSG:4326",
    )


def departure_datetime():
    return datetime.datetime.fromisoformat(f"{DATE}T{DEPARTURE}")


# --- cafein side -----------------------------------------------------------


def cafein_network():
    from cafein import TransportNetwork

    return TransportNetwork.from_gtfs([str(GTFS)], osm_pbf=str(PBF))


def cafein_travel_time_matrix(network, points, _destinations):
    import pandas as pd

    from cafein import TravelTimeMatrix

    matrix = TravelTimeMatrix(
        network,
        origins=points,
        destinations=points,
        date=DATE,
        departure=DEPARTURE,
        window=WINDOW_SECONDS,
    )
    column = "travel_time_p50" if "travel_time_p50" in matrix.columns else "travel_time"
    return pd.DataFrame(
        {
            "from_id": matrix["from_id"].astype(str),
            "to_id": matrix["to_id"].astype(str),
            "travel_time_min": matrix[column].astype(float) / 60.0,
        }
    )


def cafein_detailed_itineraries(network, origins, destinations):
    import pandas as pd

    from cafein import DetailedItineraries

    itineraries = DetailedItineraries(
        network,
        origins=origins,
        destinations=destinations,
        date=DATE,
        departure=DEPARTURE,
        max_snap_distance=ITINERARY_SNAP_METERS,
    )
    frame = pd.DataFrame(itineraries.drop(columns="geometry"))
    if frame.empty:
        return pd.DataFrame(
            columns=["from_id", "to_id", "travel_time_min", "transit_legs"]
        )
    frame["from_id"] = frame["from_id"].astype(str)
    frame["to_id"] = frame["to_id"].astype(str)
    records = []
    for (from_id, to_id, option), group in frame.groupby(
        ["from_id", "to_id", "option"], sort=False
    ):
        records.append(
            {
                "from_id": from_id,
                "to_id": to_id,
                "option": option,
                "travel_time_min": (group["arrival"].max() - group["departure"].min())
                / 60.0,
                "transit_legs": int((group["leg_type"] == "transit").sum()),
            }
        )
    return _best_option(pd.DataFrame(records))


# --- r5py side -------------------------------------------------------------


def r5py_network():
    import r5py

    return r5py.TransportNetwork(str(PBF), [str(GTFS)])


def r5py_modes():
    import r5py

    return [r5py.TransportMode.TRANSIT, r5py.TransportMode.WALK]


def r5py_travel_time_matrix(network, points, _destinations):
    import pandas as pd
    import r5py

    matrix = r5py.TravelTimeMatrix(
        network,
        origins=points,
        destinations=points,
        departure=departure_datetime(),
        departure_time_window=datetime.timedelta(seconds=WINDOW_SECONDS),
        speed_walking=WALKING_SPEED_KMPH,
        transport_modes=r5py_modes(),
    )
    return pd.DataFrame(
        {
            "from_id": matrix["from_id"].astype(str),
            "to_id": matrix["to_id"].astype(str),
            # r5py reports whole minutes; unreachable pairs are NaN.
            "travel_time_min": pd.to_numeric(matrix["travel_time"], errors="coerce"),
        }
    )


def r5py_detailed_itineraries(network, origins, destinations):
    import pandas as pd
    import r5py

    itineraries = r5py.DetailedItineraries(
        network,
        origins=origins,
        destinations=destinations,
        departure=departure_datetime(),
        speed_walking=WALKING_SPEED_KMPH,
        transport_modes=r5py_modes(),
    )
    return _summarize_itineraries_r5py(pd.DataFrame(itineraries))


def _summarize_itineraries_r5py(frame):
    """Per-OD best-journey travel time and transit-leg count from r5py output.

    r5py 1.x returns one row per segment with ``from_id``, ``to_id``,
    ``option``, ``transport_mode``, and timedelta ``travel_time``/``wait_time``.
    Adjust here if your r5py version names these differently.
    """
    import pandas as pd

    if frame.empty:
        return pd.DataFrame(
            columns=["from_id", "to_id", "travel_time_min", "transit_legs"]
        )
    frame = frame.copy()
    frame["from_id"] = frame["from_id"].astype(str)
    frame["to_id"] = frame["to_id"].astype(str)

    def seconds(series):
        if series.empty:
            return 0.0
        if hasattr(series, "dt"):
            return series.dt.total_seconds().sum()
        return float(series.sum())

    def is_transit(modes):
        return modes.astype(str).str.upper().str.contains("TRANSIT").sum()

    records = []
    keys = ["from_id", "to_id", "option"] if "option" in frame else ["from_id", "to_id"]
    for key, group in frame.groupby(keys, sort=False):
        from_id, to_id = (key[0], key[1]) if isinstance(key, tuple) else (key, key)
        total = seconds(group["travel_time"])
        if "wait_time" in group:
            total += seconds(group["wait_time"])
        records.append(
            {
                "from_id": from_id,
                "to_id": to_id,
                "option": key[2] if isinstance(key, tuple) and len(key) > 2 else 0,
                "travel_time_min": total / 60.0,
                "transit_legs": int(is_transit(group["transport_mode"])),
            }
        )
    return _best_option(pd.DataFrame(records))


def _best_option(per_option):
    """The fastest option per OD pair."""
    return (
        per_option.sort_values("travel_time_min")
        .groupby(["from_id", "to_id"], as_index=False)
        .first()[["from_id", "to_id", "travel_time_min", "transit_legs"]]
    )


# --- scenario registry -----------------------------------------------------

SCENARIOS = {
    "travel_time_matrix": {
        "cafein": cafein_travel_time_matrix,
        "r5py": r5py_travel_time_matrix,
        "metric": "travel_time_min",
        "extra": [],
    },
    "detailed_itineraries": {
        "cafein": cafein_detailed_itineraries,
        "r5py": r5py_detailed_itineraries,
        "metric": "travel_time_min",
        "extra": ["transit_legs"],
    },
}


# --- worker (one engine, its own process) ----------------------------------


def run_worker(engine, origins, seed, outdir, scenarios):
    build_started = time.perf_counter()
    network = cafein_network() if engine == "cafein" else r5py_network()
    build_seconds = time.perf_counter() - build_started

    points = stop_points(origins, seed)
    central = stop_points(ITINERARY_POINTS, seed, ITINERARY_BBOX)
    corner = central.iloc[: min(ITINERARY_POINTS, len(central))]

    stats = {"engine": engine, "build_seconds": build_seconds, "scenarios": {}}
    for name in scenarios:
        runner = SCENARIOS[name][engine]
        origins_input, destinations_input = (
            (points, points) if name == "travel_time_matrix" else (corner, corner)
        )
        # A scenario that raises (e.g. an r5py API mismatch) must not sink the
        # other scenarios: record the error and carry on.
        try:
            compute_started = time.perf_counter()
            result = runner(network, origins_input, destinations_input)
            compute_seconds = time.perf_counter() - compute_started
            path = pathlib.Path(outdir) / f"{engine}_{name}.csv"
            result.to_csv(path, index=False)
            stats["scenarios"][name] = {
                "compute_seconds": compute_seconds,
                "n_results": int(len(result)),
                "path": str(path),
            }
        except Exception as error:  # noqa: BLE001 - report any engine failure
            stats["scenarios"][name] = {"error": f"{type(error).__name__}: {error}"}
    stats["peak_rss_mb"] = peak_rss_mb()
    print(RESULT_PREFIX + json.dumps(stats))


# --- parent (orchestration, comparison, reporting) -------------------------


def launch_worker(engine, origins, seed, outdir, scenarios):
    process = subprocess.run(
        [
            sys.executable,
            __file__,
            "--worker",
            engine,
            "--origins",
            str(origins),
            "--seed",
            str(seed),
            "--outdir",
            outdir,
            "--scenario",
            ",".join(scenarios),
        ],
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        reason = (
            process.stderr.strip().splitlines()[-1]
            if process.stderr.strip()
            else "failed"
        )
        print(f"  {engine}: unavailable ({reason})")
        return None
    for line in process.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    print(f"  {engine}: produced no results")
    return None


def compare_scenario(name, cafein_stats, r5py_stats):
    import pandas as pd

    metric = SCENARIOS[name]["metric"]
    left = pd.read_csv(
        cafein_stats["scenarios"][name]["path"], dtype={"from_id": str, "to_id": str}
    )
    right = pd.read_csv(
        r5py_stats["scenarios"][name]["path"], dtype={"from_id": str, "to_id": str}
    )
    merged = left.merge(
        right, on=["from_id", "to_id"], how="outer", suffixes=("_cafein", "_r5py")
    )
    cafein_col, r5py_col = f"{metric}_cafein", f"{metric}_r5py"
    both = merged[merged[cafein_col].notna() & merged[r5py_col].notna()]
    diff = (both[cafein_col] - both[r5py_col]).abs()
    within = float((diff <= TOLERANCE_MINUTES).mean()) if len(both) else float("nan")
    agreement = {
        "scenario": name,
        "pairs_union": int(len(merged)),
        "both_reachable": int(len(both)),
        "only_cafein": int(
            (merged[cafein_col].notna() & merged[r5py_col].isna()).sum()
        ),
        "only_r5py": int((merged[cafein_col].isna() & merged[r5py_col].notna()).sum()),
        "within_tol_share": within,
        "median_abs_diff_min": float(diff.median()) if len(both) else float("nan"),
        "p95_abs_diff_min": float(diff.quantile(0.95)) if len(both) else float("nan"),
        "max_abs_diff_min": float(diff.max()) if len(both) else float("nan"),
    }
    if "transit_legs" in SCENARIOS[name]["extra"] and len(both):
        legs_diff = (both["transit_legs_cafein"] - both["transit_legs_r5py"]).abs()
        agreement["transit_legs_match_share"] = float((legs_diff == 0).mean())
    return agreement


def print_perf(per_engine, scenarios):
    print("\nTiming and memory (per engine)")
    header = (
        f"  {'engine':<8} {'scenario':<22} {'build_s':>8} "
        f"{'compute_s':>10} {'peak_MB':>9} {'rows':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for engine, stats in per_engine.items():
        for name in scenarios:
            scenario = stats["scenarios"].get(name)
            if scenario is None:
                continue
            if "error" in scenario:
                print(
                    f"  {engine:<8} {name:<22} {stats['build_seconds']:>8.2f} "
                    f"{'error':>10} {stats['peak_rss_mb']:>9.1f} {'-':>7}"
                )
                continue
            print(
                f"  {engine:<8} {name:<22} {stats['build_seconds']:>8.2f} "
                f"{scenario['compute_seconds']:>10.2f} {stats['peak_rss_mb']:>9.1f} "
                f"{scenario['n_results']:>7}"
            )
    errors = [
        (engine, name, scenario["error"])
        for engine, stats in per_engine.items()
        for name, scenario in stats["scenarios"].items()
        if "error" in scenario
    ]
    if errors:
        print("\nScenario errors")
        for engine, name, message in errors:
            print(f"  {engine} / {name}: {message}")


def print_agreement(agreements):
    print("\nAgreement (cafein vs r5py, travel time in minutes)")
    header = (
        f"  {'scenario':<22} {'both':>6} {'only_c':>7} {'only_r':>7} "
        f"{'within_1m':>10} {'median':>7} {'p95':>7} {'max':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for agreement in agreements:
        print(
            f"  {agreement['scenario']:<22} {agreement['both_reachable']:>6} "
            f"{agreement['only_cafein']:>7} {agreement['only_r5py']:>7} "
            f"{agreement['within_tol_share']:>10.2%} "
            f"{agreement['median_abs_diff_min']:>7.2f} "
            f"{agreement['p95_abs_diff_min']:>7.2f} {agreement['max_abs_diff_min']:>7.2f}"
        )
        if "transit_legs_match_share" in agreement:
            print(
                f"  {'':<22} transit-leg counts match: "
                f"{agreement['transit_legs_match_share']:.2%}"
            )


def write_csv(path, per_engine, agreements, scenarios):
    import csv

    by_scenario = {agreement["scenario"]: agreement for agreement in agreements}
    fields = [
        "scenario",
        "engine",
        "build_seconds",
        "compute_seconds",
        "peak_rss_mb",
        "n_results",
        "both_reachable",
        "only_cafein",
        "only_r5py",
        "within_tol_share",
        "median_abs_diff_min",
        "p95_abs_diff_min",
        "max_abs_diff_min",
        "transit_legs_match_share",
        "error",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for engine, stats in per_engine.items():
            for name in scenarios:
                scenario = stats["scenarios"].get(name)
                if scenario is None:
                    continue
                agreement = by_scenario.get(name, {})
                writer.writerow(
                    {
                        "scenario": name,
                        "engine": engine,
                        "build_seconds": round(stats["build_seconds"], 3),
                        "compute_seconds": _round(scenario.get("compute_seconds")),
                        "peak_rss_mb": round(stats["peak_rss_mb"], 1),
                        "n_results": scenario.get("n_results", ""),
                        "both_reachable": agreement.get("both_reachable", ""),
                        "only_cafein": agreement.get("only_cafein", ""),
                        "only_r5py": agreement.get("only_r5py", ""),
                        "within_tol_share": _round(agreement.get("within_tol_share")),
                        "median_abs_diff_min": _round(
                            agreement.get("median_abs_diff_min")
                        ),
                        "p95_abs_diff_min": _round(agreement.get("p95_abs_diff_min")),
                        "max_abs_diff_min": _round(agreement.get("max_abs_diff_min")),
                        "transit_legs_match_share": _round(
                            agreement.get("transit_legs_match_share")
                        ),
                        "error": scenario.get("error", ""),
                    }
                )
    print(f"\nWrote {path}")


def _round(value, digits=3):
    return round(value, digits) if isinstance(value, float) else ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=["cafein", "r5py"], help=argparse.SUPPRESS)
    parser.add_argument("--engine", choices=["both", "cafein", "r5py"], default="both")
    parser.add_argument("--origins", type=int, default=25, help="matrix sample size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario", default=",".join(SCENARIOS))
    parser.add_argument("--outdir", help=argparse.SUPPRESS)
    parser.add_argument("--csv", help="also write the results to this CSV path")
    args = parser.parse_args()

    requested = [name for name in args.scenario.split(",") if name]
    unknown = [name for name in requested if name not in SCENARIOS]
    if unknown:
        parser.error(
            f"unknown scenario(s): {', '.join(unknown)}; "
            f"choose from {', '.join(SCENARIOS)}"
        )
    scenarios = requested or list(SCENARIOS)

    if args.worker:
        run_worker(args.worker, args.origins, args.seed, args.outdir, scenarios)
        return

    for path in (GTFS, PBF):
        if not path.exists():
            parser.error(f"missing test data at {path}; run scripts/fetch_test_data.py")

    engines = ["cafein", "r5py"] if args.engine == "both" else [args.engine]
    outdir = tempfile.mkdtemp(prefix="cafein-r5py-")
    print(
        f"Comparing {' and '.join(engines)} on {args.origins} origins "
        f"(itineraries over {ITINERARY_POINTS}); results in {outdir}"
    )

    per_engine = {}
    for engine in engines:
        stats = launch_worker(engine, args.origins, args.seed, outdir, scenarios)
        if stats:
            per_engine[engine] = stats

    if not per_engine:
        print("No engine produced results.")
        return

    print_perf(per_engine, scenarios)

    agreements = []
    if "cafein" in per_engine and "r5py" in per_engine:
        comparable = [
            name
            for name in scenarios
            if "path" in per_engine["cafein"]["scenarios"].get(name, {})
            and "path" in per_engine["r5py"]["scenarios"].get(name, {})
        ]
        agreements = [
            compare_scenario(name, per_engine["cafein"], per_engine["r5py"])
            for name in comparable
        ]
        if agreements:
            print_agreement(agreements)
        skipped = [name for name in scenarios if name not in comparable]
        if skipped:
            print(
                "\n(no agreement for "
                + ", ".join(skipped)
                + " — a scenario errored or produced no results on one side)"
            )
    else:
        print("\n(agreement needs both engines — install r5py and a Java runtime)")

    if args.csv:
        write_csv(args.csv, per_engine, agreements, scenarios)


if __name__ == "__main__":
    main()
