#!/usr/bin/env python3
"""Calibrate the memory planner's constants (``cafein._memory``).

For each engine variant the script runs a fixed origin set at widths
1, 2, and 4, each in a cold subprocess, on networks of different size
— two bounding-box crops of the Helsinki sample feed and the full
feed; the street engine on the OSM extract's walking graph — measuring
each run's computation peak: the child loads the network, takes its
resident size as the baseline, then a monitor thread samples the
resident size every 1 ms while the computation runs; the run's value
is the peak sample minus the baseline. Per network a line through the
widths gives the per-worker slope and the width-independent intercept
(net of the result's own bytes). The largest network decides whether
per-worker state is resolved at all: below the planner's 64 KiB floor
the floor is recorded, else lines through the networks' slopes and
intercepts against their sizes give the per-unit and fixed constants,
floored at zero coefficient by coefficient; a network past the child
deadline is skipped. Every constant is then multiplied by an envelope
factor so no measured point lies above the estimate, and the size
fit's residual is reported as a diagnostic. The geometry row estimate
and the rasterization cell bytes are measured beside them.

Requires psutil (the sampling needs a *current* resident reader) and
the pinned test data (``python scripts/fetch_test_data.py``). Manual
tool, not part of CI; one engine's run stays under ten minutes, and
the per-engine sidecars assemble into one report:

    python scripts/calibrate_memory.py --engines time --report <dir>/<name>-time.md
    python scripts/calibrate_memory.py --assemble <dir>/<name>-*.json \\
        --report <dir>/<name>.md
"""

import argparse
import datetime
import hashlib
import io
import json
import pathlib
import platform
import subprocess
import sys
import tempfile
import textwrap
import time
import zipfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
FEED = ROOT / "tests" / "data" / "helsinki_gtfs.zip"
PBF = ROOT / "tests" / "data" / "kantakaupunki.osm.pbf"
DATE = "2022-02-22"
DEPARTURE = "2022-02-22 08:30:00"
WIDTHS = (1, 2, 4)
#: Bumped whenever the measurement protocol changes; sidecars of
#: different versions never assemble together.
CALIBRATION_VERSION = 1
CHILD_DEADLINE = 90
#: The street child also loads the OSM extract.
STREET_DEADLINE = 240
#: One engine's run, all its children included.
RUN_ALLOWANCE = 600
#: Crops of the sample feed by stop coordinates (lon_min, lat_min,
#: lon_max, lat_max): a small and a larger central box.
CROPS = {
    "crop-small": (24.90, 60.15, 24.97, 60.20),
    "crop-large": (24.80, 60.12, 25.05, 60.26),
}
VARIANTS = {
    "time": ("raptor", "tbtr", "reverse"),
    "multicriteria": ("emissions", "pareto"),
    "fare": ("zone",),
    "street": ("walk",),
}

CHILD = textwrap.dedent("""
    import json, sys, threading, time, warnings
    warnings.filterwarnings("ignore")
    import psutil
    from cafein import TransportNetwork, TravelTimeMatrix, TravelCostMatrix, fares
    from cafein.street_network import StreetNetwork
    import geopandas
    from shapely.geometry import Point
    feed, engine, variant = sys.argv[1], sys.argv[2], sys.argv[3]
    width = int(sys.argv[4])
    DEP, ARR = "%s", "%s"
    if engine == "street":
        streets = StreetNetwork.from_osm(feed, modes=["walk"])
        # Twenty grid points over the extract's centre.
        grid = [(60.160 + 0.006 * i, 24.920 + 0.008 * j) for i in range(4) for j in range(5)]
        origins = geopandas.GeoDataFrame(
            {"id": range(len(grid))},
            geometry=[Point(lon, lat) for lat, lon in grid],
            crs="EPSG:4326",
        )
        size = streets.vertex_count
    else:
        network = TransportNetwork.from_gtfs([feed])
        served = [s for s, lat, lon in network.stops if lat is not None]
        # Origins from the middle of the stop list: the leading ids have
        # no departures at the query time, and their searches end at once.
        counts = {"time": 20, "multicriteria": 10, "fare": 4}
        start = len(served) // 2
        origins = served[start : start + counts[engine]]
        size = network.stop_count
    structure = fares.zone_fare_structure(feed, rules="zones") if variant == "zone" else None
    process = psutil.Process()
    # A cold process: the baseline is the loaded network, and the
    # run's lazy initialisation lands in the width-independent
    # intercept rather than being hidden by a warm-up.
    baseline = process.memory_info().rss
    peak = [baseline]
    stop = threading.Event()
    def monitor():
        while not stop.is_set():
            peak[0] = max(peak[0], process.memory_info().rss)
            time.sleep(0.001)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    if engine == "time" and variant in ("raptor", "tbtr"):
        frame = TravelTimeMatrix(
            network, origins, departure=DEP, router=variant, workers=width
        )
    elif engine == "time":
        frame = TravelTimeMatrix(network, origins[:5], arrival=ARR, workers=width)
    elif engine == "street":
        frame = TravelTimeMatrix(
            streets, origins, origins, transport_mode="walk", workers=width
        )
    elif variant == "emissions":
        frame = TravelCostMatrix(
            network, origins, departure=DEP, optimize="emissions",
            departure_time_window=5, workers=width,
        )
    elif variant == "pareto":
        frame = TravelCostMatrix(
            network, origins, departure=DEP, optimize="emissions",
            departure_time_window=5, candidates="pareto", workers=width,
        )
    else:
        frame = TravelCostMatrix(
            network, origins, departure=DEP, optimize="fare",
            fares=structure, departure_time_window=1, workers=width,
        )
    stop.set(); thread.join()
    # A run shorter than the sampling interval is caught by the
    # resident size it leaves behind: the allocator retains its peak.
    peak[0] = max(peak[0], process.memory_info().rss)
    result_bytes = int(frame.memory_usage(deep=True).sum())
    print(json.dumps({"stops": size, "width": width,
                      "peak": peak[0] - baseline, "result": result_bytes}))
    """) % (DEPARTURE, "2022-02-22 09:30:00")


def crop_feed(source, box, target):
    """Write ``source`` with its stops (and dependent rows) cropped to
    ``box``; trips keep their in-box stop times only."""
    import pandas as pd

    lon_min, lat_min, lon_max, lat_max = box
    with zipfile.ZipFile(source) as archive:
        tables = {name: archive.read(name) for name in archive.namelist()}
    stops = pd.read_csv(io.BytesIO(tables["stops.txt"]), dtype=str)
    lon = stops["stop_lon"].astype(float)
    lat = stops["stop_lat"].astype(float)
    keep = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
    stops = stops[keep]
    kept_ids = set(stops["stop_id"])
    times = pd.read_csv(io.BytesIO(tables["stop_times.txt"]), dtype=str)
    # Only trips wholly inside the box survive: a trip with dropped
    # stops would no longer match its shape.
    inside = times["stop_id"].isin(kept_ids)
    broken = set(times.loc[~inside, "trip_id"])
    times = times[inside & ~times["trip_id"].isin(broken)]
    trips = pd.read_csv(io.BytesIO(tables["trips.txt"]), dtype=str)
    trips = trips[trips["trip_id"].isin(set(times["trip_id"]))]
    # Every stop reference must survive the crop: parent stations
    # outside the box are cleared, transfers between dropped stops go.
    if "parent_station" in stops.columns:
        stops = stops.copy()
        outside = ~stops["parent_station"].isin(kept_ids)
        stops.loc[outside, "parent_station"] = ""
    if "transfers.txt" in tables:
        transfers = pd.read_csv(io.BytesIO(tables["transfers.txt"]), dtype=str)
        transfers = transfers[
            transfers["from_stop_id"].isin(kept_ids)
            & transfers["to_stop_id"].isin(kept_ids)
        ]
        tables["transfers.txt"] = transfers.to_csv(index=False).encode()
    tables["stops.txt"] = stops.to_csv(index=False).encode()
    tables["stop_times.txt"] = times.to_csv(index=False).encode()
    tables["trips.txt"] = trips.to_csv(index=False).encode()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in tables.items():
            archive.writestr(name, payload)


def measure(feed, engine, variant, child_deadline, run_deadline):
    """Every width in its own cold subprocess; ``None`` when a width
    exceeds its deadline or the run's absolute one (the network is
    then skipped for this variant and the report says so)."""
    runs = []
    stops = None
    for width in WIDTHS:
        # Recomputed before every child, so the run's allowance is
        # never spent more than once.
        remaining = run_deadline - time.monotonic()
        if remaining <= 0:
            return None
        try:
            out = subprocess.run(
                [sys.executable, "-c", CHILD, str(feed), engine, variant, str(width)],
                capture_output=True,
                text=True,
                timeout=min(child_deadline, remaining),
                cwd=str(ROOT),
            )
        except subprocess.TimeoutExpired:
            return None
        line = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
        if not line:
            raise RuntimeError(
                f"{engine}/{variant} on {feed.name} at width {width} produced "
                f"no result:\n{out.stderr[-600:]}"
            )
        measured = json.loads(line)
        stops = measured["stops"]
        runs.append(measured)
    return {"stops": stops, "runs": runs}


def fit_line(xs, ys):
    """Least-squares ``y = slope * x + intercept``."""
    slope, intercept = np.polyfit(np.asarray(xs, float), np.asarray(ys, float), 1)
    return float(slope), float(intercept)


def collect(feeds):
    """Measure every engine variant on its networks: one sample per
    network, or a skipped marker past the deadline."""
    samples = []
    for engine, variants in VARIANTS.items():
        networks = {"kantakaupunki": PBF} if engine == "street" else feeds
        child_deadline = STREET_DEADLINE if engine == "street" else CHILD_DEADLINE
        run_deadline = time.monotonic() + RUN_ALLOWANCE
        for variant in variants:
            for name, feed in networks.items():
                # The engine's run stays inside its allowance: every
                # child gets the smaller of its deadline and what remains.
                measured = measure(feed, engine, variant, child_deadline, run_deadline)
                if measured is None:
                    reason = (
                        "the deadline"
                        if run_deadline - time.monotonic() > 0
                        else "the run's allowance"
                    )
                    print(
                        f"{engine:<14}{variant:<10}{name:<12} skipped: past {reason}",
                        flush=True,
                    )
                    samples.append(
                        {"engine": engine, "variant": variant, "network": name}
                    )
                    continue
                sample = {
                    "engine": engine,
                    "variant": variant,
                    "network": name,
                    "size": measured["stops"],
                    "widths": [r["width"] for r in measured["runs"]],
                    "peaks": [r["peak"] for r in measured["runs"]],
                    "results": [r["result"] for r in measured["runs"]],
                }
                samples.append(sample)
                print(
                    f"{engine:<14}{variant:<10}{name:<12}{sample['size']:>7}  "
                    + "  ".join(
                        f"w{w}: {p / 2**20:6.1f} MB"
                        for w, p in zip(sample["widths"], sample["peaks"])
                    ),
                    flush=True,
                )
    return samples


def fit(samples):
    """Per engine: the constants, the envelope factor, the residuals."""
    per_engine = {}
    for engine in dict.fromkeys(s["engine"] for s in samples):
        candidates = []
        variants = dict.fromkeys(s["variant"] for s in samples if s["engine"] == engine)
        for variant in variants:
            chosen = [
                s for s in samples if s["engine"] == engine and s["variant"] == variant
            ]
            skipped = [s["network"] for s in chosen if "size" not in s]
            per_network = []
            for s in chosen:
                if "size" not in s:
                    continue
                slope, intercept = fit_line(s["widths"], s["peaks"])
                # Resident sampling cannot resolve a negative slope: noise.
                per_network.append(
                    (
                        s["size"],
                        max(0.0, slope),
                        intercept - float(np.mean(s["results"])),
                        s["widths"],
                        s["peaks"],
                        s["results"],
                    )
                )
            if not per_network:
                continue
            per_network.sort(key=lambda n: n[0])
            sizes = [n[0] for n in per_network]
            # The largest network decides whether per-worker state is
            # resolved at all: a small crop's search state is
            # legitimately under the planner's 64 KiB floor.
            below_floor = per_network[-1][1] < 64 * 1024
            worst_residual = float("nan")
            if below_floor or len(per_network) < 2:
                unit, fixed = 0.0, float(64 * 1024)
            else:
                unit, fixed = fit_line(sizes, [n[1] for n in per_network])
                unit, fixed = max(0.0, unit), max(0.0, fixed)
                worst_residual = 0.0
                for size, slope, _, _, _, _ in per_network:
                    predicted = unit * size + fixed
                    if predicted > 0:
                        worst_residual = max(
                            worst_residual, abs(slope - predicted) / predicted
                        )
            if len(per_network) >= 2:
                call_unit, call_fixed = fit_line(sizes, [n[2] for n in per_network])
            else:
                call_unit, call_fixed = 0.0, per_network[-1][2]
            call_unit, call_fixed = max(0.0, call_unit), max(0.0, call_fixed)
            # The envelope over every observed peak, against the final
            # constants, so the planner's estimate bounds each point. The
            # residual is a diagnostic of the linear model, never a gate.
            envelope = 1.0
            for size, _, _, widths, peaks, results in per_network:
                for width, peak, result in zip(widths, peaks, results):
                    estimate = (
                        (unit * size + fixed) * width + call_unit * size + call_fixed
                    )
                    # Net of the result, which the planner reserves separately.
                    if estimate > 0:
                        envelope = max(envelope, max(0.0, peak - result) / estimate)
            candidates.append(
                {
                    "variant": variant,
                    "unit": unit * envelope,
                    "fixed": fixed * envelope,
                    "call_unit": call_unit * envelope,
                    "call_fixed": call_fixed * envelope,
                    "envelope": envelope,
                    "residual": worst_residual,
                    "below_floor": below_floor,
                    "skipped": skipped,
                }
            )
        if not candidates:
            continue
        # The engine's constants are the maxima over its variants.
        per_engine[engine] = {
            key: max(c[key] for c in candidates)
            for key in ("unit", "fixed", "call_unit", "call_fixed")
        }
        per_engine[engine]["variants"] = candidates
    return per_engine


def geometry_row_bytes(feed):
    from cafein import TransportNetwork, TravelCostMatrix
    import warnings

    warnings.filterwarnings("ignore")
    network = TransportNetwork.from_gtfs([feed])
    served = [s for s, lat, lon in network.stops if lat is not None]
    # Three mid-list origins to every stop; the leading ids have no
    # departures at the query time and would leave nothing to measure.
    start = len(served) // 2
    frame = TravelCostMatrix(
        network, served[start : start + 3], departure=DEPARTURE, geometries=True
    )
    lengths = frame.geometry.dropna().apply(lambda g: len(g.wkb))
    # The mean over rows that carry a geometry, plus the Arrow offset.
    return int(np.ceil(lengths.mean() + 8)) if len(lengths) else 8


CELL_CHILD = textwrap.dedent("""
    import json, threading, time
    try:
        import psutil
        import numpy as np
        import rasterio.features
        from affine import Affine
        from shapely.geometry import box
    except ImportError:
        print("null")
        raise SystemExit
    process = psutil.Process()
    shape = (2000, 2000)
    baseline = process.memory_info().rss
    peak = [baseline]
    stop = threading.Event()
    def monitor():
        while not stop.is_set():
            peak[0] = max(peak[0], process.memory_info().rss)
            time.sleep(0.001)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    burned = rasterio.features.rasterize(
        [(box(0, 0, 1000, 1000), 1.0)], out_shape=shape,
        transform=Affine(0.5, 0, 0, 0, -0.5, 1000), fill=np.nan,
        dtype="float32", all_touched=True,
    )
    stop.set(); thread.join()
    peak[0] = max(peak[0], process.memory_info().rss)
    print(json.dumps({"peak": peak[0] - baseline, "cells": shape[0] * shape[1]}))
""")


def cell_bytes():
    """The resident bytes one rasterized float32 burn cell costs at the
    rasterization's peak (its shape masks included), measured in a
    cold subprocess; ``None`` without rasterio."""
    out = subprocess.run(
        [sys.executable, "-c", CELL_CHILD],
        capture_output=True,
        text=True,
        timeout=CHILD_DEADLINE,
        cwd=str(ROOT),
    )
    lines = out.stdout.strip().splitlines()
    measured = json.loads(lines[-1]) if lines else None
    if not measured:
        return None
    return max(4, int(np.ceil(measured["peak"] / measured["cells"])))


CHAIN = ("time", "multicriteria", "fare")


def chain_floors(constants):
    """A search that embeds another's is never planned smaller than it:
    multicriteria never below time, fare never below multicriteria.
    Returns the engines whose terms were raised."""
    raised = []
    for lower, upper in zip(CHAIN, CHAIN[1:]):
        if lower in constants and upper in constants:
            for key in ("unit", "fixed", "call_unit", "call_fixed"):
                if constants[lower][key] > constants[upper][key]:
                    constants[upper][key] = constants[lower][key]
                    if upper not in raised:
                        raised.append(upper)
    return raised


def provenance():
    """What a sidecar records about its measurement environment and inputs."""
    return {
        "version": CALIBRATION_VERSION,
        "machine": platform.platform(),
        "python": platform.python_version(),
        "date": datetime.date.today().isoformat(),
        "feed_sha256": _sha256(FEED),
        "pbf_sha256": _sha256(PBF),
        "crops": {name: list(box) for name, box in CROPS.items()},
        "widths": list(WIDTHS),
    }


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render(rows, constants, row_bytes, cells, raised, meta):
    lines = [
        f"# Memory calibration, {meta['date']}",
        "",
        f"Machine: {meta['machine']}, Python {meta['python']}, sample feed "
        f"sha256 {meta['feed_sha256'][:12]}, extract sha256 "
        f"{meta['pbf_sha256'][:12]}. "
        "Networks: two bounding-box crops of the Helsinki sample feed and the "
        "full feed, the street engine on the OSM extract's walking graph; "
        f"widths {WIDTHS}; computation peaks sampled every 1 ms "
        "against a post-load baseline.",
        "",
        "| engine | variant | network | size | "
        + " | ".join(f"w{w}" for w in WIDTHS)
        + " |",
        "|---|---|---|---|" + "---|" * len(WIDTHS),
    ]
    for engine, variant, name, stops, peaks in rows:
        lines.append(
            f"| {engine} | {variant} | {name} | {stops} | "
            + " | ".join(f"{p / 2**20:.1f} MB" for p in peaks)
            + " |"
        )
    lines += ["", "## Constants (envelope-adjusted, rounded up)", "", "```python"]
    for key, name in (
        ("unit", "BYTES_PER_UNIT"),
        ("fixed", "FIXED_BYTES"),
        ("call_unit", "CALL_BYTES_PER_UNIT"),
        ("call_fixed", "CALL_FIXED_BYTES"),
    ):
        values = {engine: int(np.ceil(constants[engine][key])) for engine in constants}
        lines.append(f"{name} = {values!r}")
    lines.append(f"GEOMETRY_ROW_BYTES = {row_bytes}")
    lines.append(f"BYTES_PER_CELL = {cells}")
    lines += ["```", "", "## Fit quality", ""]
    for engine, data in constants.items():
        for c in data["variants"]:
            note = (
                " — per-worker state below RSS resolution; the planner's floor applies"
                if c["below_floor"]
                else ""
            )
            if c["skipped"]:
                note += f" — skipped past the deadline: {', '.join(c['skipped'])}"
            residual = "n/a" if np.isnan(c["residual"]) else f"{c['residual']:.1%}"
            lines.append(
                f"- {engine}/{c['variant']}: envelope {c['envelope']:.2f}, "
                f"size-fit residual {residual}{note}"
            )
    for engine in raised:
        lower = CHAIN[CHAIN.index(engine) - 1]
        lines.append(
            f"- {engine}: terms raised to the {lower} engine's where they "
            f"measured lower (a {engine} search embeds one)"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--report", type=pathlib.Path, default=None)
    parser.add_argument(
        "--engines",
        default=",".join(VARIANTS),
        help="comma-separated engines to calibrate (default: all)",
    )
    parser.add_argument(
        "--crops",
        type=pathlib.Path,
        default=pathlib.Path(tempfile.gettempdir()) / "cafein-calibration-crops",
        help="directory holding the cropped feeds (reused across runs)",
    )
    parser.add_argument(
        "--assemble",
        nargs="+",
        type=pathlib.Path,
        default=None,
        help="merge the JSON sidecars of earlier per-engine runs into one report",
    )
    args = parser.parse_args()
    if args.assemble:
        samples, row_bytes, cells, meta = [], 0, None, None
        for path in args.assemble:
            data = json.loads(path.read_text())
            stored = data["meta"]
            if meta is None:
                meta = dict(stored)
            elif {k: v for k, v in stored.items() if k != "date"} != {
                k: v for k, v in meta.items() if k != "date"
            }:
                raise SystemExit(
                    f"{path} was measured under a different environment "
                    "(version, machine, Python, feed, or widths) than "
                    f"{args.assemble[0]}; sidecars assemble only from one"
                )
            elif stored["date"] not in meta["date"]:
                meta["date"] += ", " + stored["date"]
            samples += data["samples"]
            row_bytes = max(row_bytes, data["row_bytes"])
            if data["cells"] is not None:
                cells = max(cells or 0, data["cells"])
    else:
        for engine in list(VARIANTS):
            if engine not in args.engines.split(","):
                del VARIANTS[engine]
        try:
            import psutil  # noqa: F401
        except ImportError:
            raise SystemExit("calibration needs psutil (pip install psutil)")
        if not FEED.exists():
            raise SystemExit(
                f"{FEED} is missing; run python scripts/fetch_test_data.py"
            )
        args.crops.mkdir(parents=True, exist_ok=True)
        feeds = {}
        for name, crop in CROPS.items():
            target = args.crops / f"{name}.zip"
            if not target.exists():
                crop_feed(FEED, crop, target)
            feeds[name] = target
        feeds["helsinki"] = FEED
        meta = provenance()
        samples = collect(feeds)
        row_bytes = geometry_row_bytes(FEED)
        cells = cell_bytes()
        if args.report:
            # The sidecar keeps the raw samples, so per-engine runs are
            # assembled into one report under one fitting rule.
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "meta": meta,
                        "samples": samples,
                        "row_bytes": row_bytes,
                        "cells": cells,
                    }
                )
            )
    constants = fit(samples)
    rows = [
        (s["engine"], s["variant"], s["network"], s["size"], s["peaks"])
        for s in samples
        if "size" in s
    ]
    raised = chain_floors(constants)
    report = render(rows, constants, row_bytes, cells, raised, meta)
    print(report)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report)


if __name__ == "__main__":
    main()
