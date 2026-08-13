#!/usr/bin/env python3

"""Issue #246 diagnostic: what zones and ticket chain does a priced
journey actually use?

    python probe_zone_fares.py assign   # compiled stop zones vs the feed
    python probe_zone_fares.py price    # legs, zones and fares per pair

``assign`` needs only the feed. ``price`` builds the metro-scale network
from ``cafein.sampledata.helsinki``.
"""

import csv
import io
import json
import sys
import time
import zipfile

import os

import cafein.sampledata.helsinki as helsinki
from cafein import TransportNetwork
from cafein import fares as fares_module

DATE = "2026-09-08"  # a Tuesday inside the feed's 2026-08-07..10-05 span
DEPARTURE = "08:30:00"
WINDOW = 1800
DESTINATION = "1000109"  # Herttoniemi metro

PRICES = {
    "AB": 3.30,
    "BC": 3.30,
    "CD": 3.30,
    "DE": 3.30,
    "ABC": 4.50,
    "BCD": 4.50,
    "CDE": 4.50,
    "ABCD": 5.00,
    "BCDE": 5.00,
    "ABCDE": 6.10,
}
WINDOWS = {
    "AB": 4800,
    "BC": 4800,
    "CD": 5400,
    "DE": 4800,
    "ABC": 5400,
    "BCD": 6000,
    "CDE": 5400,
    "ABCD": 6600,
    "BCDE": 6000,
    "ABCDE": 6600,
}


def feed_rows():
    with zipfile.ZipFile(helsinki.gtfs) as archive:
        return list(
            csv.DictReader(io.StringIO(archive.read("stops.txt").decode("utf-8-sig")))
        )


def single_ticket(zones):
    covering = [(p, n) for n, p in PRICES.items() if zones <= set(n)]
    return min(covering) if covering else (float("nan"), "?")


def assign():
    """The compiled per-stop zone index against the feed's own column.

    ``stop_zone`` is an index into the sorted distinct ``zone_id``
    values, not a bitmask.
    """
    rows = feed_rows()
    network = TransportNetwork.from_gtfs([str(helsinki.gtfs)])
    structure = fares_module.zone_fare_structure(str(helsinki.gtfs), rules="zones")
    flat = structure._flat_tables(network)
    order = [stop for stop, _, _ in network.stops]
    declared = {r["stop_id"]: (r.get("zone_id") or "").strip() for r in rows}
    letters = sorted({z for z in declared.values() if z})
    index_of = {letter: index for index, letter in enumerate(letters)}
    wrong = [
        (stop, declared.get(stop), mask)
        for stop, mask in zip(order, flat["stop_zone"])
        if index_of.get(declared.get(stop, ""), -1) != mask
    ]
    print(f"zone values (sorted, index order): {letters}")
    print(f"stops: {len(order)}  mismatched against the feed: {len(wrong)}")
    for row in wrong[:10]:
        print("   ", row)
    products = dict(zip(structure.fares["fare_id"], flat["products"]))
    print("products (fare_id -> zone mask):")
    for name, product in products.items():
        covered = "".join(
            letters[bit] for bit in range(len(letters)) if product[1] >> bit & 1
        )
        print(f"    {name:6s} mask={product[1]:#08b} covers={covered}")


def price():
    from cafein import journey_frontier

    rows = feed_rows()
    zones = {r["stop_id"]: (r.get("zone_id") or "").strip() for r in rows}
    names = {r["stop_id"]: r["stop_name"] for r in rows}
    coords = {r["stop_id"]: (float(r["stop_lat"]), float(r["stop_lon"])) for r in rows}

    started = time.perf_counter()
    network = TransportNetwork.from_gtfs(
        [str(helsinki.gtfs)], osm_pbf=str(helsinki.osm_pbf)
    )
    print(f"network built in {time.perf_counter() - started:.1f}s", flush=True)
    structure = fares_module.zone_fare_structure(str(helsinki.gtfs), rules="zones")

    west = sorted(s for s in zones if zones[s] == "C" and coords[s][1] < 24.70)
    north = sorted(s for s in zones if zones[s] == "C" and coords[s][0] > 60.32)
    for origin in west[:3] + north[:3]:
        frame = journey_frontier(
            network,
            origin,
            DESTINATION,
            DATE,
            DEPARTURE,
            window=WINDOW,
            fares=structure,
        )
        journeys = network.route_between_stops(
            origin, DESTINATION, DATE, DEPARTURE, window=WINDOW
        )
        print(
            f"\n=== {origin} {names[origin]!r} zone {zones[origin]} "
            f"-> {names.get(DESTINATION)!r} zone {zones.get(DESTINATION)}"
        )
        if len(frame):
            print(
                "   candidate fares:",
                sorted({round(f, 2) for f in frame["fare"].tolist()}),
                "travel times:",
                sorted({int(t) for t in frame["travel_time_s"].tolist()})[:6],
            )
        for journey in journeys[:2]:
            span, legs = set(), []
            for leg in journey["legs"]:
                if leg["type"] != "transit":
                    continue
                board, alight = leg["board_stop"], leg["alight_stop"]
                span |= {zones.get(board, "?"), zones.get(alight, "?")}
                legs.append(
                    f"{leg.get('route_short_name')}: {zones.get(board)}{board}"
                    f"->{zones.get(alight)}{alight} @{leg['departure_s']}"
                )
            price_, ticket = single_ticket(span)
            duration = journey["arrival_s"] - journey["departure_s"]
            print(
                json.dumps(
                    {
                        "rides": journey["rides"],
                        "duration_s": duration,
                        "zones": "".join(sorted(span)),
                        "single_ticket": ticket,
                        "single_price": price_,
                        "ticket_window_s": WINDOWS.get(ticket),
                        "fits_one_ticket": duration <= WINDOWS.get(ticket, 0),
                        "legs": legs,
                    }
                )
            )


def matrix():
    """The path issue #246 used: does ``optimize="fare"`` return the
    cheapest candidate the frontier can see?"""
    from cafein import TravelCostMatrix, journey_frontier

    rows = feed_rows()
    names = {r["stop_id"]: r["stop_name"] for r in rows}

    started = time.perf_counter()
    network = TransportNetwork.from_gtfs(
        [str(helsinki.gtfs)], osm_pbf=str(helsinki.osm_pbf)
    )
    print(f"network built in {time.perf_counter() - started:.1f}s", flush=True)
    structure = fares_module.zone_fare_structure(str(helsinki.gtfs), rules="zones")

    origins = ["2644203", "2644201", "2644204", "2000004"]
    frame = TravelCostMatrix(
        network,
        origins,
        [DESTINATION],
        DATE,
        DEPARTURE,
        optimize="fare",
        window=WINDOW,
        fares=structure,
    )
    cells = frame[(frame["from_id"].isin(origins)) & (frame["to_id"] == DESTINATION)]
    for _, cell in cells.iterrows():
        candidates = journey_frontier(
            network,
            cell["from_id"],
            DESTINATION,
            DATE,
            DEPARTURE,
            window=WINDOW,
            fares=structure,
        )
        seen = sorted({round(f, 2) for f in candidates["fare"].tolist()})
        print(
            json.dumps(
                {
                    "origin": cell["from_id"],
                    "name": names[cell["from_id"]],
                    "matrix_fare": round(float(cell["fare"]), 2),
                    "matrix_seconds": int(cell["travel_time_s"]),
                    "frontier_fares": seen,
                    "cheapest_available": seen[0] if seen else None,
                    "matrix_picked_cheapest": bool(seen)
                    and abs(float(cell["fare"]) - seen[0]) < 1e-6,
                }
            ),
            flush=True,
        )


def scan():
    """The shipped fare matrix against the exact fare frontier.

    Both now ride the zone-ticket state machine, through different
    folds with different pruning — the matrix warm-starts per-slot
    bounds and an arrival deadline off the fare-blind fold, the
    frontier folds per cutoff with neither. Per scanned cell the two
    must agree to the cent, both ways: an overpriced cell is issue
    #246 back from the dead, an underpriced one a bound pruning a
    journey it must not.
    """
    from cafein import TravelCostMatrix
    from cafein.frontier import fare_frontier

    rows = feed_rows()
    zones = {r["stop_id"]: (r.get("zone_id") or "").strip() for r in rows}
    names = {r["stop_id"]: r["stop_name"] for r in rows}
    coords = {r["stop_id"]: (float(r["stop_lat"]), float(r["stop_lon"])) for r in rows}

    started = time.perf_counter()
    network = TransportNetwork.from_gtfs(
        [str(helsinki.gtfs)], osm_pbf=str(helsinki.osm_pbf)
    )
    print(f"network built in {time.perf_counter() - started:.1f}s", flush=True)
    structure = fares_module.zone_fare_structure(str(helsinki.gtfs), rules="zones")

    far = sorted(
        s
        for s in zones
        if zones[s] in ("C", "D") and (coords[s][1] < 24.72 or coords[s][0] > 60.30)
    )
    # Twelve spread origins keep the scan inside a ten-minute local
    # budget at two workers; the suite pins the measured pairs.
    origins = far[:: max(1, len(far) // 12)][:12]
    print(f"origins: {len(origins)}", flush=True)

    # Both sides run under the same six-hour duration cap: an
    # unbounded exact search must prove no cheaper journey exists
    # anywhere in the service day, which at metro scale takes minutes
    # per origin.
    matrix = TravelCostMatrix(
        network,
        origins,
        [DESTINATION],
        DATE,
        DEPARTURE,
        optimize="fare",
        window=WINDOW,
        within=3 * 3600,
        fares=structure,
    )
    priced = {
        row["from_id"]: (round(float(row["fare"]), 2), int(row["travel_time_s"]))
        for _, row in matrix.iterrows()
    }
    # A frontier row at cutoff c is the FASTEST journey within c, not
    # the cheapest, so a single cutoff cannot witness the minimum
    # fare. Every attainable fare is a sum of ticket prices; probing
    # the ascending attainable sums makes the first cutoff that yields
    # a row pin the exact minimum (a cheaper journey's sum would be an
    # earlier grid point with a row). Four tickets cover the 6-hour
    # duration cap on every HSL window.
    import itertools

    prices = sorted(set(PRICES.values()))
    # The universal (dearest) product bounds the optimum: a journey
    # within the cap partitions into floor(cap / its window) + 1 such
    # tickets. An optimal sum may still chain up to max_transfers + 1
    # = 8 cheaper tickets below that ceiling.
    cap = 3 * 3600
    dearest_id = max(PRICES, key=PRICES.get)
    ceiling = (cap // WINDOWS[dearest_id] + 1) * PRICES[dearest_id]
    sums = set()
    for count in range(1, 9):
        for combo in itertools.combinations_with_replacement(prices, count):
            total = round(sum(combo), 2)
            if total <= ceiling:
                sums.add(total)
    all_sums = sorted(sums)
    cheapest = {}
    for index, origin in enumerate(origins):
        started_origin = time.perf_counter()
        # Matrix-anchored cutoffs keep the cutoff product in its cheap
        # regime: sums at or below the matrix fare witness both sides
        # (a row below it is an overpriced matrix cell; a row at it
        # proves the price is a real journey). A cell the matrix could
        # not price is checked for emptiness at the ceiling alone.
        cell = priced.get(origin)
        if cell is not None:
            cutoffs = [s for s in all_sums if s <= cell[0] + 1e-6] or [all_sums[0]]
        else:
            cutoffs = [round(ceiling, 2)]
        frame = fare_frontier(
            network,
            [origin],
            [DESTINATION],
            DATE,
            DEPARTURE,
            WINDOW,
            structure,
            cutoffs=cutoffs,
            max_duration=cap,
            departure_step=None,
        )
        priced_rows = sorted(
            (
                float(row["cutoff"]),
                round(float(row["fare"]), 2),
                int(row["travel_time_s"]),
            )
            for _, row in frame.iterrows()
            if row["fare"] == row["fare"]
        )
        if priced_rows:
            _, fare, seconds = priced_rows[0]
            cheapest[origin] = (fare, seconds)
        print(
            f"   [{index + 1}/{len(origins)}] {origin}: "
            f"{cheapest.get(origin) or 'no row'} "
            f"({time.perf_counter() - started_origin:.1f}s)",
            flush=True,
        )

    mismatched = 0
    compared = 0
    for origin in origins:
        matrix_cell = priced.get(origin)
        frontier_cell = cheapest.get(origin)
        if matrix_cell is None and frontier_cell is None:
            continue
        compared += 1
        agree = (
            matrix_cell is not None
            and frontier_cell is not None
            and abs(matrix_cell[0] - frontier_cell[0]) <= 1e-6
        )
        if not agree:
            mismatched += 1
            print(
                json.dumps(
                    {
                        "origin": origin,
                        "name": names[origin],
                        "zone": zones[origin],
                        "matrix": matrix_cell,
                        "frontier": frontier_cell,
                    }
                ),
                flush=True,
            )
    print(
        f"\norigins: {len(origins)}  matrix-priced: {len(priced)}  "
        f"frontier-priced: {len(cheapest)}  compared: {compared}  "
        f"mismatched: {mismatched}"
    )
    if mismatched or not compared:
        raise SystemExit(
            "FAIL: matrix and exact frontier disagree"
            if mismatched
            else "inconclusive: no cell priced on either side"
        )


def case():
    """One overpriced pair in detail: the candidates the fare fold saw,
    with and without the D/E zones available."""
    from cafein import journey_frontier

    rows = feed_rows()
    zones = {r["stop_id"]: (r.get("zone_id") or "").strip() for r in rows}
    names = {r["stop_id"]: r["stop_name"] for r in rows}
    network = TransportNetwork.from_gtfs(
        [str(helsinki.gtfs)], osm_pbf=str(helsinki.osm_pbf)
    )
    structure = fares_module.zone_fare_structure(str(helsinki.gtfs), rules="zones")
    barred = [s for s, z in zones.items() if z in ("D", "E", "Ei HSL")]

    for origin in ("9214203", "4340212"):
        for label, excluded in (("unrestricted", ()), ("D/E barred", barred)):
            frame = journey_frontier(
                network,
                origin,
                DESTINATION,
                DATE,
                DEPARTURE,
                window=WINDOW,
                fares=structure,
                router="raptor",
                exclude_stops=excluded,
            )
            print(
                f"\n=== {origin} {names[origin]!r} — {label}: {len(frame)} candidates"
            )
            for _, row in frame.head(6).iterrows():
                print(
                    f"   travel {int(row['travel_time_s'])}s  rides {int(row['rides'])}"
                    f"  fare {row['fare']}  arrival {int(row['arrival_s'])}"
                )
            journeys = network.route_between_stops(
                origin,
                DESTINATION,
                DATE,
                DEPARTURE,
                window=WINDOW,
                exclude_stops=list(excluded),
            )
            for journey in journeys[:2]:
                legs = [
                    f"{leg.get('route_short_name')}:"
                    f"{zones.get(leg['board_stop'])}->{zones.get(leg['alight_stop'])}"
                    for leg in journey["legs"]
                    if leg["type"] == "transit"
                ]
                span = {
                    zones.get(leg[key])
                    for leg in journey["legs"]
                    if leg["type"] == "transit"
                    for key in ("board_stop", "alight_stop")
                }
                print(
                    f"   journey rides={journey['rides']} "
                    f"arrival={journey['arrival_s']} zones={''.join(sorted(span))} "
                    f"legs={legs}"
                )


if __name__ == "__main__":
    # The exact zone refinement holds per-origin search state;
    # unguarded host-wide parallelism recreated an OOM once. Two
    # workers unless the caller says otherwise, matching the recorded
    # benchmark protocol.
    os.environ.setdefault("RAYON_NUM_THREADS", "2")
    {"assign": assign, "price": price, "matrix": matrix, "scan": scan, "case": case}[
        sys.argv[1] if len(sys.argv) > 1 else "assign"
    ]()
