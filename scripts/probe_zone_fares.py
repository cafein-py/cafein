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
    """The shipped fare matrix against the exact fare engine.

    ``optimize="fare"`` folds over (departure, arrival, rides)-Pareto
    candidates, so a cheaper-but-slower journey can be dominated before
    it is ever priced; ``fare_frontier`` carries the fare in the label.
    Any pair where the matrix charges more than the frontier's cheapest
    reachable fare is issue #246 in the wild.
    """
    from cafein import TravelCostMatrix, journey_frontier

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
    origins = far[:: max(1, len(far) // 40)][:40]
    print(f"origins: {len(origins)}", flush=True)

    matrix = TravelCostMatrix(
        network,
        origins,
        [DESTINATION],
        DATE,
        DEPARTURE,
        optimize="fare",
        window=WINDOW,
        fares=structure,
    )
    priced = {
        row["from_id"]: (round(float(row["fare"]), 2), int(row["travel_time_s"]))
        for _, row in matrix.iterrows()
    }
    # No fare-carrying engine exists for zone structures, so the oracle
    # is an exclusion: bar every D and E zone stop and see what the
    # cheapest surviving journey costs. Anything the matrix charges
    # above that is a journey the fare-blind candidate fold discarded.
    barred = [s for s, z in zones.items() if z in ("D", "E", "Ei HSL")]
    print(f"barring {len(barred)} D/E stops for the oracle", flush=True)
    cheapest = {}
    for origin in origins:
        try:
            frame = journey_frontier(
                network,
                origin,
                DESTINATION,
                DATE,
                DEPARTURE,
                window=WINDOW,
                fares=structure,
                router="raptor",
                exclude_stops=barred,
            )
        except Exception as error:  # noqa: BLE001 - reported, not raised
            print(f"   {origin}: oracle failed: {error}", flush=True)
            continue
        priced_rows = [
            (round(float(row["fare"]), 2), int(row["travel_time_s"]))
            for _, row in frame.iterrows()
            if row["fare"] == row["fare"]
        ]
        if priced_rows:
            cheapest[origin] = min(priced_rows)

    worse = 0
    for origin in origins:
        matrix_cell = priced.get(origin)
        frontier_cell = cheapest.get(origin)
        if not matrix_cell or not frontier_cell:
            continue
        if matrix_cell[0] > frontier_cell[0] + 1e-6:
            worse += 1
            print(
                json.dumps(
                    {
                        "origin": origin,
                        "name": names[origin],
                        "zone": zones[origin],
                        "matrix_fare": matrix_cell[0],
                        "matrix_seconds": matrix_cell[1],
                        "frontier_fare": frontier_cell[0],
                        "frontier_seconds": frontier_cell[1],
                    }
                ),
                flush=True,
            )
    print(f"\npairs compared: {len(priced)}  matrix overpriced: {worse}")


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
                    f"{leg.get('route_short_name')}: "
                    f"{zones.get(leg['board_stop'])}{leg['board_stop']}@{leg['departure_s']}"
                    f"->{zones.get(leg['alight_stop'])}{leg['alight_stop']}@{leg['arrival_s']}"
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
    {"assign": assign, "price": price, "matrix": matrix, "scan": scan, "case": case}[
        sys.argv[1] if len(sys.argv) > 1 else "assign"
    ]()
