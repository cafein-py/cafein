#!/usr/bin/env python3

"""Compare cafein's journey prices against r5r's on the Porto Alegre sample.

Both engines load the identical fare structure (``fares_poa.zip``, r5r's
saved Porto Alegre fares) and route the same sampled stop coordinates
door-to-door on the same feeds. r5r computes its Pareto frontier of
travel time versus monetary cost (R5's suboptimal McRAPTOR sampler with
an in-routing fare calculator); cafein prices its exact range-RAPTOR
window candidates post hoc, times anchored at the window start so
waiting counts the way r5r counts it. The hard check is the fare
arithmetic — every fare cafein charges must be a level the shared
structure can produce — while travel-time gaps at matching fare levels
are reported for judgment only: either engine may find a journey the
other misses (exact profile vs suboptimal sampler), and such rows
compare different journeys, not different pricing.

    python scripts/compare_fares_vs_r5r.py             # sampled stops
    python scripts/compare_fares_vs_r5r.py --pairs 30

Requirements: cafein installed, and an R environment with r5r >= 2.4 and
its R5 jar cached (run ``library(r5r)`` once). Point ``--rscript`` (or
``CAFEIN_RSCRIPT``) at the environment's Rscript; set JAVA_HOME to a
JDK 21+ if rJava needs it. The test data comes from
``python scripts/fetch_test_data.py``. Manual tool, not part of CI.
"""

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import zipfile

DATA = pathlib.Path(__file__).parent.parent / "tests" / "data"
FEEDS = [DATA / "poa_eptc.zip", DATA / "poa_trensurb.zip"]
PBF = DATA / "poa_osm.pbf"
FARES = DATA / "fares_poa.zip"

DATE = "2019-05-13"
DEPARTURE = "14:00:00"
WINDOW = 60
MAX_RIDES = 3
SLACK = 90.0

R_TEMPLATE = """
options(java.parameters = "-Xmx2G")
suppressMessages(library(r5r))
suppressMessages(library(data.table))

network <- build_network("{staging}", temp_dir = FALSE)
fares <- read_fare_structure("{fares}")
points <- fread("{points}")
pairs <- fread("{pairs}")
origins <- points[match(pairs$from_id, points$id)]
destinations <- points[match(pairs$to_id, points$id)]

frontier <- pareto_frontier(
    network,
    origins = origins,
    destinations = destinations,
    mode = c("WALK", "TRANSIT"),
    departure_datetime = as.POSIXct("{date} {departure}"),
    time_window = 1L,
    max_trip_duration = 180L,
    fare_structure = fares,
    fare_cutoffs = c({cutoffs}),
    max_rides = {max_rides}L
)
fwrite(frontier, "{output}")
"""


def stop_sample(count, seed):
    import pandas as pd

    stops = []
    for feed in FEEDS:
        with zipfile.ZipFile(feed) as archive, archive.open("stops.txt") as file:
            stops.append(pd.read_csv(file, dtype={"stop_id": str}))
    stops = pd.concat(stops, ignore_index=True)
    origins = stops.sample(count, random_state=seed).reset_index(drop=True)
    destinations = stops.sample(count, random_state=seed + 1).reset_index(drop=True)
    return stops, list(zip(origins["stop_id"], destinations["stop_id"]))


def requalified(structure, network):
    """The fare structure re-keyed to the network's public route ids.

    A merged cafein network qualifies route ids with their feed index
    (``0:632``); r5r's saved table uses the feeds' bare ids. Rows are
    duplicated under every public id whose bare part matches.
    """
    import pandas as pd

    from cafein import fares

    table = structure.fares_per_route
    bare = {row["route_id"]: row for _, row in table.iterrows()}
    rows = []
    for public, _, _ in network.routes:
        base = public.split(":", 1)[1] if ":" in public else public
        if base in bare:
            row = dict(bare[base])
            row["route_id"] = public
            rows.append(row)
    return fares.FareStructure(
        max_discounted_transfers=structure.max_discounted_transfers,
        transfer_time_allowance=structure.transfer_time_allowance,
        fare_cap=structure.fare_cap,
        fares_per_type=structure.fares_per_type,
        fares_per_transfer=structure.fares_per_transfer,
        fares_per_route=pd.DataFrame(rows, columns=table.columns),
    )


def cafein_candidates(stops, pairs, fare_levels):
    import warnings

    from cafein import TransportNetwork, fares, journey_frontier

    coordinates = {
        row.stop_id: (row.stop_lat, row.stop_lon) for row in stops.itertuples()
    }
    with warnings.catch_warnings():
        # The ingest interpolates the EPTC feed's blank stop times and
        # tolerates its invalid route_text_color, warning about both.
        warnings.simplefilter("ignore")
        network = TransportNetwork.from_gtfs(
            [str(feed) for feed in FEEDS], osm_pbf=str(PBF)
        )
        structure = requalified(fares.load_fare_structure(FARES), network)
        results = {}
        for origin, destination in pairs:
            # Door-to-door from the stops' own coordinates, matching the
            # r5r side's access semantics.
            try:
                frame = journey_frontier(
                    network,
                    coordinates[origin],
                    coordinates[destination],
                    DATE,
                    DEPARTURE,
                    window=WINDOW,
                    max_transfers=MAX_RIDES - 1,
                    fares=structure,
                )
            except (KeyError, ValueError):
                continue
            rows = frame[frame["fare"].notna()]
            if len(rows):
                # r5r anchors travel times at the window start (waiting
                # included); measure cafein's candidates the same way.
                start = sum(
                    int(part) * scale
                    for part, scale in zip(DEPARTURE.split(":"), (3600, 60, 1))
                )
                results[(origin, destination)] = [
                    (row["arrival"] - start, row["fare"]) for _, row in rows.iterrows()
                ]
    return results


def r5r_frontier(rscript, stops, pairs, fare_levels):
    import pandas as pd

    staging = pathlib.Path(tempfile.mkdtemp(prefix="cafein-r5r-"))
    for feed in FEEDS:
        shutil.copy(feed, staging / feed.name)
    shutil.copy(PBF, staging / PBF.name)
    points = staging / "points.csv"
    unique = sorted({stop for pair in pairs for stop in pair})
    frame = stops[stops["stop_id"].isin(unique)].drop_duplicates("stop_id")
    frame = frame.rename(
        columns={"stop_id": "id", "stop_lat": "lat", "stop_lon": "lon"}
    )
    frame[["id", "lat", "lon"]].to_csv(points, index=False)
    pairs_path = staging / "pairs.csv"
    pd.DataFrame(pairs, columns=["from_id", "to_id"]).to_csv(pairs_path, index=False)
    output = staging / "frontier.csv"
    script = staging / "run.R"
    script.write_text(
        R_TEMPLATE.format(
            staging=staging,
            fares=FARES,
            points=points,
            pairs=pairs_path,
            date=DATE,
            departure=DEPARTURE,
            cutoffs=", ".join(f"{level:.2f}" for level in fare_levels),
            max_rides=MAX_RIDES,
            output=output,
        )
    )
    subprocess.run([rscript, str(script)], check=True)
    frontier = pd.read_csv(output, dtype={"from_id": str, "to_id": str})
    results = {}
    for (origin, destination), group in frontier.groupby(["from_id", "to_id"]):
        results[(origin, destination)] = [
            # r5r reports travel times in minutes.
            (row["travel_time"] * 60.0, row["monetary_cost"])
            for _, row in group.iterrows()
        ]
    return results


def fare_levels():
    import pandas as pd

    with zipfile.ZipFile(FARES) as archive:
        types = pd.read_csv(archive.open("fares_per_type.csv"))
        transfers = pd.read_csv(archive.open("fares_per_transfer.csv"))
    singles = sorted(set(types["fare"]))
    pairs = sorted(set(transfers["fare"]))
    # Walking is free; then up to three boardings: singles, transfer
    # pairs, and either with one more full fare.
    levels = {0.0} | set(singles) | set(pairs)
    levels |= {pair + single for pair in pairs for single in singles}
    levels |= {a + b for a in singles for b in singles}
    levels |= {a + b + c for a in singles for b in singles for c in singles}
    return sorted({round(level, 2) for level in levels})


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rscript",
        default=os.environ.get("CAFEIN_RSCRIPT", "Rscript"),
        help="Rscript of an R environment with r5r installed",
    )
    arguments = parser.parse_args()

    levels = fare_levels()
    stops, pairs = stop_sample(arguments.pairs, arguments.seed)
    ours = cafein_candidates(stops, pairs, levels)
    theirs = r5r_frontier(arguments.rscript, stops, pairs, levels)

    shared = sorted(set(ours) & set(theirs))
    print(
        f"{len(pairs)} sampled pairs: cafein priced {len(ours)}, "
        f"r5r reached {len(theirs)} (all-to-all), comparing {len(shared)}"
    )
    if not shared:
        sys.exit("no comparable pairs — check the inputs")

    # The hard invariant: both engines price journeys from the same
    # structure, so every cafein fare must be a level r5r's frontier can
    # express, and where both offer a fare level, the values coincide by
    # construction of the shared table. Travel times are reported but
    # not asserted: cafein's candidates are exact range-RAPTOR profile
    # journeys, while r5r's frontier comes from R5's *suboptimal*
    # McRAPTOR sampler, so either engine may find a journey the other
    # misses — those rows compare different journeys, not different
    # fare arithmetic.
    rounded_levels = {round(level, 2) for level in levels}
    unknown = [
        (pair, fare)
        for pair in shared
        for _, fare in ours[pair]
        if round(fare, 2) not in rounded_levels
    ]
    print(f"cafein fares outside the structure's levels: {len(unknown)}")
    for pair, fare in unknown[:10]:
        print(f"  {pair}: {fare}")

    matched = faster = slower = 0
    gaps = []
    for pair in shared:
        frontier = theirs[pair]
        for time, fare in ours[pair]:
            matching = [t for t, cost in frontier if abs(cost - fare) < 0.005]
            if not matching:
                continue
            matched += 1
            gap = min(matching) - time
            gaps.append(gap)
            if gap > SLACK:
                faster += 1
            elif gap < -SLACK:
                slower += 1
    gaps.sort()
    median = gaps[len(gaps) // 2] if gaps else float("nan")
    print(
        f"fare levels offered by both engines: {matched} "
        f"(cafein faster: {faster}, r5r faster: {slower}, "
        f"median r5r-minus-cafein gap {median:+.0f} s)"
    )
    if unknown:
        sys.exit("FAIL: cafein charged fares outside the shared structure")
    print("OK: every cafein fare is a level of the shared structure")


if __name__ == "__main__":
    main()
