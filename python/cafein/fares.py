"""Monetary journey costs: fare structures and post-hoc pricing.

Fares are journey-level, not leg-level: discounted transfers, transfer
time windows, zone extents, and caps make the price a function of the
whole leg sequence and its timing. A journey is therefore priced after
routing, from its legs — never inside the routing loop.

Two fare models are supported:

- **Rule-based** (`FareStructure`) mirrors r5r's editable fare
  structure: global settings (``max_discounted_transfers``,
  ``transfer_time_allowance`` in minutes, ``fare_cap``) plus three
  tables — ``fares_per_type`` (per transit mode), ``fares_per_transfer``
  (total prices of mode-pair transfers), and ``fares_per_route``
  (per-route fares). `setup_fare_structure` derives an initial structure
  from a network the way r5r's does; the tables are plain DataFrames to
  edit, and `load_fare_structure` reads a structure saved by r5r (or by
  `save_fare_structure`) so the two tools can share fare definitions.
  Pricing follows r5r's rule-based calculator exactly.
- **Street rentals** price beside either model: a ``street`` tariff
  (per rental mode: an ``unlock`` price plus a ``per_minute`` price,
  billed per started minute) covers the shared-vehicle legs a street
  policy reconstructs. `annotate_fares` names which modes rode rentals
  (``shared_modes`` — own vehicles and walking are free), and a rental
  leg whose mode has no tariff prices NaN, never a silent zero.
- **Zone-based** (`ZoneFareStructure`) prices journeys from GTFS
  ``fare_attributes.txt``/``fare_rules.txt`` — the model Helsinki
  Region Transport ships: a ticket is the cheapest fare valid for a
  stretch of boardings within its transfer window, and a long journey
  chains tickets. A fare's rule rows become alternative **grants** —
  its ``contains_id`` zone set (covering the stretch's boarding and
  alighting zones; with contiguous zone products, as in ring-shaped
  systems, that equals the traversed span), its route set (route-only
  rows), and its origin/destination clauses — any one of which
  validates a stretch, with agency scope bounding them all and a fare
  without rules valid network-wide.
"""

import math
import zipfile

import pandas as pd

#: The compiled core's sentinel for "no fare row" / "no zone".
_NO_FARE = 0xFFFFFFFF

_MODES = {
    0: "TRAM",
    1: "SUBWAY",
    2: "RAIL",
    3: "BUS",
    4: "FERRY",
    5: "CABLE_CAR",
    6: "GONDOLA",
    7: "FUNICULAR",
}

_TYPE_COLUMNS = [
    "type",
    "unlimited_transfers",
    "allow_same_route_transfer",
    "use_route_fare",
    "fare",
]
_TRANSFER_COLUMNS = ["first_leg", "second_leg", "fare"]
_ROUTE_COLUMNS = [
    "agency_id",
    "agency_name",
    "route_id",
    "route_short_name",
    "route_long_name",
    "mode",
    "route_fare",
    "fare_type",
]


class FareStructure:
    """An editable rule-based fare structure, as in r5r.

    Attributes
    ----------
    max_discounted_transfers : int
        How many transfers may receive a discounted (integration) fare;
        later transfers pay the full fare of their leg.
    transfer_time_allowance : float
        Minutes between boardings within which a discounted transfer
        applies; a later boarding pays full fare.
    fare_cap : float
        Ceiling on a journey's total fare (``inf``: uncapped).
    fares_per_type : pandas.DataFrame
        Per transit mode: ``fare``, ``unlimited_transfers`` (rides of
        the same mode are free after the first),
        ``allow_same_route_transfer`` (whether a discounted transfer may
        return to the same route), and ``use_route_fare`` (whether the
        per-route fare overrides the mode fare).
    fares_per_transfer : pandas.DataFrame
        Per ordered mode pair: the *total* price of the two-leg
        combination; a missing pair means no integration (full fares).
    fares_per_route : pandas.DataFrame
        Per route: identity columns, ``route_fare``, and ``fare_type``
        (the mode row the route prices under).
    """

    def __init__(
        self,
        *,
        max_discounted_transfers=1,
        transfer_time_allowance=120.0,
        fare_cap=math.inf,
        fares_per_type=None,
        fares_per_transfer=None,
        fares_per_route=None,
        street=None,
    ):
        self.max_discounted_transfers = int(max_discounted_transfers)
        self.transfer_time_allowance = float(transfer_time_allowance)
        self.fare_cap = float(fare_cap)
        self.fares_per_type = _framed(fares_per_type, _TYPE_COLUMNS)
        self.fares_per_transfer = _framed(fares_per_transfer, _TRANSFER_COLUMNS)
        self.fares_per_route = _framed(fares_per_route, _ROUTE_COLUMNS)
        self.street = _street_tariffs(street)

    def price(self, journey):
        """The journey's fare, mirroring r5r's rule-based calculator.

        Walking is free; the first ride pays its full fare; each further
        ride pays its full fare unless the mode allows unlimited
        transfers (same mode, free) or an in-time discounted transfer
        applies, in which case the pair's total replaces the two full
        fares. Returns NaN when a ridden route has no fare row.
        """
        return self._pricer()(journey)

    def _pricer(self):
        """A journey-pricing closure with the lookups prepared once."""
        types = {str(row["type"]): row for _, row in self.fares_per_type.iterrows()}
        pairs = {
            (str(row["first_leg"]), str(row["second_leg"])): float(row["fare"])
            for _, row in self.fares_per_transfer.iterrows()
        }
        routes = {}
        for _, row in self.fares_per_route.iterrows():
            fare_type = str(row["fare_type"])
            kind = types.get(fare_type)
            if kind is None:
                continue
            full = (
                float(row["route_fare"])
                if bool(kind["use_route_fare"])
                else float(kind["fare"])
            )
            routes[str(row["route_id"])] = (full, fare_type)
        allowance = self.transfer_time_allowance * 60.0

        def full_fare(route_id):
            return routes.get(route_id, (math.nan, None))

        def priced(journey):
            rides = [leg for leg in journey["legs"] if leg["type"] == "transit"]
            if not rides:
                return 0.0
            total, (previous_fare, previous_type) = 0.0, full_fare(rides[0]["route_id"])
            if previous_type is None:
                return math.nan
            total = previous_fare
            previous_route = rides[0]["route_id"]
            previous_board = rides[0]["departure_s"]
            discounts = 0
            for ride in rides[1:]:
                fare, fare_type = full_fare(ride["route_id"])
                if fare_type is None:
                    return math.nan
                # Rides within an unlimited-transfers mode are free and
                # spend neither a discount nor the transfer clock; a
                # later integration prices off this ride's route.
                if fare_type == previous_type and bool(
                    types[fare_type]["unlimited_transfers"]
                ):
                    previous_route = ride["route_id"]
                    previous_fare = fare
                    continue
                pair = pairs.get((previous_type, fare_type))
                allowed = fare_type != previous_type or (
                    bool(types[fare_type]["allow_same_route_transfer"])
                    or ride["route_id"] != previous_route
                )
                in_time = ride["departure_s"] - previous_board <= allowance
                if (
                    discounts < self.max_discounted_transfers
                    and pair is not None
                    and allowed
                    and in_time
                ):
                    # The pair price is the total of both legs; the
                    # first leg's full fare is already counted.
                    total += pair - previous_fare
                    discounts += 1
                else:
                    total += fare
                previous_fare, previous_type = fare, fare_type
                previous_route = ride["route_id"]
                previous_board = ride["departure_s"]
            if math.isfinite(self.fare_cap):
                total = min(total, self.fare_cap)
            return total

        return priced

    def _flat_tables(self, network):
        """The flat arrays the compiled matrix computers price with.

        Route and type identities become positions: ``route_type`` and
        ``route_fare`` follow ``network.routes`` order with the full
        fare (route or type fare) resolved ahead, and ``pair_fare`` is
        the dense type × type total-price matrix (NaN: no integration).
        """
        types = {
            str(row["type"]): (index, row)
            for index, (_, row) in enumerate(self.fares_per_type.iterrows())
        }
        count = len(self.fares_per_type)
        pair_fare = [math.nan] * (count * count)
        for _, row in self.fares_per_transfer.iterrows():
            first = types.get(str(row["first_leg"]))
            second = types.get(str(row["second_leg"]))
            if first is None or second is None:
                continue
            pair_fare[first[0] * count + second[0]] = float(row["fare"])
        routes = {}
        for _, row in self.fares_per_route.iterrows():
            kind = types.get(str(row["fare_type"]))
            if kind is None:
                continue
            index, spec = kind
            full = (
                float(row["route_fare"])
                if bool(spec["use_route_fare"])
                else float(spec["fare"])
            )
            routes[str(row["route_id"])] = (index, full)
        route_type, route_fare = [], []
        for route_id, _, _ in network.routes:
            index, full = routes.get(str(route_id), (_NO_FARE, math.nan))
            route_type.append(index)
            route_fare.append(full)
        return {
            "route_type": route_type,
            "route_fare": route_fare,
            "unlimited_transfers": [
                bool(value) for value in self.fares_per_type["unlimited_transfers"]
            ],
            "allow_same_route": [
                bool(value)
                for value in self.fares_per_type["allow_same_route_transfer"]
            ],
            "pair_fare": pair_fare,
            "max_discounted_transfers": self.max_discounted_transfers,
            "transfer_allowance": self.transfer_time_allowance * 60.0,
            "fare_cap": self.fare_cap,
        }


class ZoneFareStructure:
    """Zone-set fares from GTFS ``fare_attributes``/``fare_rules``.

    A product's restriction dimensions are **alternative grants** — any
    one validates a covered stretch, because that is what real v1 data
    means (HSL's ``D`` ticket carries a zone row, a route-only row, and
    origin/destination rows at once, and plainly covers ordinary
    D-zone trips). A product with no grants is unrestricted. Agency
    scope is the one conjunct, bounding every grant.

    Attributes
    ----------
    fares : pandas.DataFrame
        One row per fare product: ``fare_id``, ``price``,
        ``currency_type``, ``transfers`` (NaN: unlimited within the
        window), ``transfer_duration`` (seconds of validity from the
        first boarding).
    fare_zones : dict
        ``fare_id`` → frozenset of zones: the **zone grant** — a
        stretch whose traversed zones are a subset is covered.
    fare_routes : dict
        ``fare_id`` → frozenset of routes: the **route grant** — a
        stretch whose every leg rides one of them is covered.
    fare_od : dict
        ``fare_id`` → tuple of ``(origin, destination, route)``
        **OD clauses** (each field optional, ``None`` when absent):
        the origin constrains the stretch's first boarding zone, the
        destination its last alighting zone, and a named route binds
        to its clause — every covered leg must ride it.
    agency_routes : dict
        ``fare_id`` → frozenset of the fare's agency's routes — the
        conjunct: when present, every covered leg must ride one,
        whatever grant applies.
    stop_zones : dict
        ``stop_id`` → zone.
    """

    def __init__(
        self,
        fares,
        fare_zones,
        stop_zones,
        street=None,
        fare_routes=None,
        fare_od=None,
        agency_routes=None,
    ):
        self.fares = fares
        # Keys normalize to the strings the pricer looks up: a numeric
        # fare id must not silently drop its restrictions and turn the
        # fare unrestricted.
        self.fare_zones = {
            str(fare): frozenset(zones) for fare, zones in fare_zones.items()
        }
        self.stop_zones = dict(stop_zones)
        self.street = _street_tariffs(street)
        self.fare_routes = {
            str(fare): frozenset(routes) for fare, routes in (fare_routes or {}).items()
        }
        self.fare_od = {
            str(fare): tuple(
                (origin, destination, route) for origin, destination, route in clauses
            )
            for fare, clauses in (fare_od or {}).items()
        }
        self.agency_routes = {
            str(fare): frozenset(routes)
            for fare, routes in (agency_routes or {}).items()
        }

    @property
    def _has_rule_grants(self):
        return bool(self.fare_routes or self.fare_od or self.agency_routes)

    def price(self, journey):
        """The journey's fare: the cheapest chain of tickets.

        A ticket covers a stretch of boardings within its transfer
        window when one of its grants validates it — its zone set
        covers the stretch's boarding and alighting zones, its route
        set covers every leg's route, or an origin/destination clause
        matches the stretch's endpoints — under its agency scope; a
        grant-free ticket is valid network-wide, and a journey longer
        than one window chains tickets. Returns NaN when some leg no
        ticket can cover (an unknown stop zone fails the zone grant
        and any OD endpoint that must match it, never the route
        grant).
        """
        return self._pricer()(journey)

    def _pricer(self):
        products = []
        for _, row in self.fares.iterrows():
            fare_id = str(row["fare_id"])
            zones = self.fare_zones.get(fare_id)
            routes = self.fare_routes.get(fare_id)
            clauses = self.fare_od.get(fare_id, ())
            # A product no rule row restricts is unrestricted — the
            # spec's reading of a fare without fare_rules — and covers
            # any stretch within its window.
            # `transfers` and `transfer_duration` are optional columns:
            # absent means unlimited boardings without a time limit.
            duration = row.get("transfer_duration")
            transfers = row.get("transfers")
            products.append(
                (
                    float(row["price"]),
                    zones,
                    routes,
                    clauses,
                    self.agency_routes.get(fare_id),
                    math.inf if pd.isna(duration) else float(duration),
                    math.inf if pd.isna(transfers) else int(transfers),
                )
            )

        def covers(product, needs, start, end):
            """Whether the product's grants validate legs start..end."""
            _, zones, routes, clauses, agency, _, _ = product
            stretch = needs[start : end + 1]
            if agency is not None and any(
                leg_route not in agency for _, _, _, leg_route in stretch
            ):
                return False
            if zones is None and routes is None and not clauses:
                return True
            if zones is not None:
                union = set()
                known = True
                for leg_zones, _, _, _ in stretch:
                    if leg_zones is None:
                        known = False
                        break
                    union |= leg_zones
                if known and union <= zones:
                    return True
            if routes is not None and all(
                leg_route in routes for _, _, _, leg_route in stretch
            ):
                return True
            first_board = stretch[0][2][0]
            last_alight = stretch[-1][2][1]
            for origin, destination, route in clauses:
                if origin is not None and first_board != origin:
                    continue
                if destination is not None and last_alight != destination:
                    continue
                if route is not None and any(
                    leg_route != route for _, _, _, leg_route in stretch
                ):
                    continue
                return True
            return False

        def priced(journey):
            rides = [leg for leg in journey["legs"] if leg["type"] == "transit"]
            if not rides:
                return 0.0
            needs = []
            for ride in rides:
                board = self.stop_zones.get(ride["board_stop"])
                alight = self.stop_zones.get(ride["alight_stop"])
                zones = (
                    None
                    if board is None or alight is None
                    else frozenset((board, alight))
                )
                needs.append(
                    (
                        zones,
                        ride["departure_s"],
                        (board, alight),
                        ride.get("route_id"),
                    )
                )

            best = {len(needs): 0.0}

            def cost(at):
                if at in best:
                    return best[at]
                cheapest = math.nan
                for product in products:
                    price, _, _, _, _, duration, transfers = product
                    # Every stretch end within the window and transfer
                    # count is tried — an OD-restricted product may
                    # need the next ticket to start mid-window.
                    end = at
                    while True:
                        if (
                            needs[end][1] - needs[at][1] <= duration
                            and (end - at) <= transfers
                            and covers(product, needs, at, end)
                        ):
                            rest = cost(end + 1)
                            candidate = price + rest
                            if math.isnan(cheapest) or candidate < cheapest:
                                cheapest = candidate
                        if (
                            end + 1 >= len(needs)
                            or needs[end + 1][1] - needs[at][1] > duration
                            or (end + 1 - at) > transfers
                        ):
                            break
                        end += 1
                best[at] = cheapest
                return cheapest

            return cost(0)

        return priced

    def _flat_tables(self, network):
        """The flat arrays the compiled matrix computers price with.

        Zones become bit positions: ``stop_zone`` maps each stop (in
        ``network.stops`` order) to its zone index, and each product is
        ``(price, zone bitmask, window seconds, boardings after the
        first)`` with sentinels for "no zone" and "unlimited". The
        compiled pricer carries the zone model only: a structure with
        route, OD, or agency grants is rejected rather than silently
        priced differently from ``annotate_fares``.
        """
        if self._has_rule_grants:
            raise ValueError(
                "matrix fare pricing does not carry route, "
                "origin/destination, or agency fare rules yet; load "
                "with zone_fare_structure(..., rules='zones') for the "
                "zone-only model, or price journeys with annotate_fares"
            )
        unrestricted = sorted(
            str(row["fare_id"])
            for _, row in self.fares.iterrows()
            if str(row["fare_id"]) not in self.fare_zones
        )
        if unrestricted:
            raise ValueError(
                "matrix fare pricing does not carry unrestricted "
                f"products yet ({unrestricted}); price journeys with "
                "annotate_fares"
            )
        zones = sorted(
            {zone for covered in self.fare_zones.values() for zone in covered}
            | set(self.stop_zones.values())
        )
        if len(zones) > 128:
            raise ValueError("matrix fare pricing supports at most 128 zones")
        index = {zone: position for position, zone in enumerate(zones)}
        stop_zone = [
            index.get(self.stop_zones.get(str(stop_id)), _NO_FARE)
            for stop_id, _, _ in network.stops
        ]
        products = []
        for _, row in self.fares.iterrows():
            covered = self.fare_zones.get(str(row["fare_id"]))
            if covered is None:
                continue
            mask = 0
            for zone in covered:
                mask |= 1 << index[zone]
            duration = row.get("transfer_duration")
            transfers = row.get("transfers")
            products.append(
                (
                    float(row["price"]),
                    mask,
                    math.inf if pd.isna(duration) else float(duration),
                    _NO_FARE if pd.isna(transfers) else int(transfers),
                )
            )
        return {"stop_zone": stop_zone, "products": products}


def setup_fare_structure(network, base_fare, by="MODE"):
    """An initial rule-based fare structure derived from a network.

    Every route, mode, and mode pair starts at `base_fare`, as in r5r's
    ``setup_fare_structure``; edit the tables and the global attributes
    to express the actual fare rules.

    Parameters
    ----------
    network : TransportNetwork
        The network whose routes and modes seed the tables.
    base_fare : float
        The price every fare starts from.
    by : str (optional, default: "MODE")
        How routes group into fare types: ``"MODE"`` (one type per
        transit mode) or ``"GENERIC"`` (a single type).

    Returns
    -------
    FareStructure
    """
    if by not in ("MODE", "GENERIC"):
        raise ValueError(f"by must be 'MODE' or 'GENERIC', not {by!r}")
    base_fare = float(base_fare)
    routes = []
    for route_id, short_name, route_type in network.routes:
        mode = _MODES.get(route_type, str(route_type))
        routes.append(
            {
                "agency_id": "",
                "agency_name": "",
                "route_id": route_id,
                "route_short_name": short_name or "",
                "route_long_name": "",
                "mode": mode,
                "route_fare": base_fare,
                "fare_type": mode if by == "MODE" else "GENERIC",
            }
        )
    fares_per_route = pd.DataFrame(routes, columns=_ROUTE_COLUMNS)
    kinds = sorted(set(fares_per_route["fare_type"]))
    fares_per_type = pd.DataFrame(
        [
            {
                "type": kind,
                "unlimited_transfers": False,
                "allow_same_route_transfer": False,
                "use_route_fare": False,
                "fare": base_fare,
            }
            for kind in kinds
        ],
        columns=_TYPE_COLUMNS,
    )
    fares_per_transfer = pd.DataFrame(
        [
            {"first_leg": first, "second_leg": second, "fare": base_fare}
            for first in kinds
            for second in kinds
        ],
        columns=_TRANSFER_COLUMNS,
    )
    return FareStructure(
        fares_per_type=fares_per_type,
        fares_per_transfer=fares_per_transfer,
        fares_per_route=fares_per_route,
    )


def load_fare_structure(path):
    """A rule-based fare structure from an r5r-format zip.

    Reads the layout ``r5r::write_fare_structure`` produces (and
    `save_fare_structure` mirrors): ``global_settings.csv`` plus the
    three fare tables. Debug settings are ignored.
    """
    with zipfile.ZipFile(path) as archive:
        settings = pd.read_csv(archive.open("global_settings.csv"))
        settings = dict(zip(settings["setting"], settings["value"]))
        fares_per_type = pd.read_csv(archive.open("fares_per_type.csv"))
        fares_per_transfer = pd.read_csv(archive.open("fares_per_transfer.csv"))
        fares_per_route = pd.read_csv(
            archive.open("fares_per_route.csv"), dtype={"route_id": str}
        )
    cap = str(settings.get("fare_cap", "Inf"))
    return FareStructure(
        max_discounted_transfers=int(float(settings["max_discounted_transfers"])),
        transfer_time_allowance=float(settings["transfer_time_allowance"]),
        fare_cap=math.inf if cap.lower() in ("inf", "infinity") else float(cap),
        fares_per_type=fares_per_type,
        fares_per_transfer=fares_per_transfer,
        fares_per_route=fares_per_route,
    )


def save_fare_structure(structure, path):
    """Save a rule-based fare structure as an r5r-format zip."""
    cap = structure.fare_cap
    settings = pd.DataFrame(
        {
            "setting": [
                "max_discounted_transfers",
                "transfer_time_allowance",
                "fare_cap",
            ],
            "value": [
                structure.max_discounted_transfers,
                structure.transfer_time_allowance,
                "Inf" if math.isinf(cap) else cap,
            ],
        }
    )
    debug = pd.DataFrame(
        {"setting": ["output_file", "trip_info"], "value": ['""', "MODE"]}
    )
    with zipfile.ZipFile(path, "w") as archive:
        for name, frame in [
            ("global_settings.csv", settings),
            ("fares_per_type.csv", structure.fares_per_type),
            ("fares_per_transfer.csv", structure.fares_per_transfer),
            ("fares_per_route.csv", structure.fares_per_route),
            ("debug_settings.csv", debug),
        ]:
            archive.writestr(name, frame.to_csv(index=False))


def zone_fare_structure(gtfs_path, rules="model"):
    """A zone-based fare structure from a GTFS feed's fare files.

    Reads ``fare_attributes.txt``, ``fare_rules.txt``, and the stops'
    ``zone_id`` column. Every rule row shape becomes a grant:
    ``contains_id`` values aggregate into the fare's zone set whatever
    else their row carries, a row whose only other field is
    ``route_id`` joins the route grant, and a row with either
    endpoint becomes one origin/destination clause holding exactly
    its present fields. Agency scope — ``fare_attributes.agency_id``
    resolved through ``routes.txt`` — compiles into the conjunctive
    agency-route set, and a multi-agency feed whose fares omit
    ``agency_id`` is rejected (the spec requires it). Pass
    ``rules="zones"`` for the pre-grant zone-only reading (route, OD,
    and agency rules ignored) — the model the compiled matrix fare
    path still prices.
    """
    if rules not in ("model", "zones"):
        raise ValueError(f"rules must be 'model' or 'zones', not {rules!r}")

    # Everything reads as verbatim strings: numeric-looking ids must
    # not become numbers (a numeric agency_id would look absent), and
    # pandas' default NA tokens must not eat legal ids like "NA" — a
    # lost restriction would make a product look unrestricted.
    def read(handle):
        return pd.read_csv(handle, dtype=str, keep_default_na=False)

    with zipfile.ZipFile(gtfs_path) as archive:
        names = set(archive.namelist())
        if "fare_attributes.txt" not in names:
            raise ValueError(f"'{gtfs_path}' carries no GTFS fare files")
        attributes = read(archive.open("fare_attributes.txt"))
        # fare_rules.txt is optional: a feed with attributes alone
        # sells unrestricted, network-wide products.
        rule_rows = (
            read(archive.open("fare_rules.txt"))
            if "fare_rules.txt" in names
            else pd.DataFrame(columns=["fare_id"])
        )
        stops = read(archive.open("stops.txt"))
        routes_table = (
            read(archive.open("routes.txt")) if "routes.txt" in names else None
        )
        agency_table = (
            read(archive.open("agency.txt")) if "agency.txt" in names else None
        )
    # The optional numeric columns come back as text; an empty cell is
    # absent, exactly as the NaN the pricer already handles.
    for column in ("transfers", "transfer_duration"):
        if column in attributes.columns:
            attributes[column] = attributes[column].map(
                lambda value: math.nan if not str(value).strip() else value
            )

    def cell(row, column):
        value = getattr(row, column, None)
        return value if isinstance(value, str) and value.strip() else None

    fare_zones = {}
    fare_routes = {}
    fare_od = {}
    for row in rule_rows.itertuples():
        fare = cell(row, "fare_id")
        if fare is None:
            continue
        contains = cell(row, "contains_id")
        if contains is not None:
            fare_zones.setdefault(fare, set()).add(contains)
        origin = cell(row, "origin_id")
        destination = cell(row, "destination_id")
        route = cell(row, "route_id")
        if origin is not None or destination is not None:
            fare_od.setdefault(fare, []).append((origin, destination, route))
        elif route is not None and contains is None:
            # Only a row whose sole field is the route joins the route
            # grant; a contains-bearing row contributes its zone alone,
            # or the fare would turn valid on that route outside its
            # covered zones.
            fare_routes.setdefault(fare, set()).add(route)
    stop_zones = {}
    if "zone_id" in stops.columns:
        stop_zones = {
            row.stop_id: row.zone_id
            for row in stops.itertuples()
            if isinstance(row.zone_id, str) and row.zone_id.strip()
        }
    if rules == "zones":
        # The zone-only reading must not turn a fare whose (dropped)
        # rules were route- or OD-keyed into an unrestricted one:
        # fares with rule rows but no contains rows stay out entirely,
        # exactly the pre-grant behaviour.
        ruled = {
            cell(row, "fare_id")
            for row in rule_rows.itertuples()
            if cell(row, "fare_id") is not None
        }
        keep = attributes["fare_id"].map(
            lambda fare: str(fare) in fare_zones or str(fare) not in ruled
        )
        return ZoneFareStructure(attributes[keep], fare_zones, stop_zones)
    agency_of_route = {}
    if routes_table is not None:
        for row in routes_table.itertuples():
            route = cell(row, "route_id")
            if route is not None:
                agency_of_route[route] = cell(row, "agency_id")
    agencies = set()
    if agency_table is not None:
        for row in agency_table.itertuples():
            agencies.add(cell(row, "agency_id"))
    agency_routes = {}
    if "agency_id" in attributes.columns or len(agencies) > 1:
        # In a single-agency feed, routes may omit agency_id yet
        # belong to the one agency; without routes.txt the scope
        # cannot be resolved and the fare stays unscoped.
        sole_agency = next(iter(agencies)) if len(agencies) == 1 else None
        for row in attributes.itertuples():
            fare = str(row.fare_id)
            agency = cell(row, "agency_id")
            if agency is None:
                if len(agencies) > 1:
                    raise ValueError(
                        f"fare {fare!r} names no agency_id in a "
                        "multi-agency feed; the spec requires it — fix "
                        "the feed or load with rules='zones'"
                    )
                continue
            # Kept even when empty: an agency with no resolvable
            # routes surfaces as an unpriceable fare, never as one
            # valid everywhere.
            agency_routes[fare] = {
                route
                for route, owner in agency_of_route.items()
                if (owner or sole_agency) == agency
            }
    return ZoneFareStructure(
        attributes,
        fare_zones,
        stop_zones,
        fare_routes=fare_routes,
        fare_od=fare_od,
        agency_routes=agency_routes,
    )


def annotate_fares(journeys, structure, shared_modes=()):
    """Attach ``fare`` to journeys, in place, and return them.

    Parameters
    ----------
    journeys : list of dict
        Journeys as returned by the routing calls.
    structure : FareStructure or ZoneFareStructure
        The fare model to price with.
    shared_modes : iterable of str (optional)
        The street modes ridden as rentals under the journeys' street
        policy (the policy's ``source="shared"`` modes). Their legs
        price from the structure's ``street`` tariff — an unlock plus
        started minutes per leg — while own-vehicle and walking legs
        stay free; a rental mode without a tariff prices ``NaN``,
        never a silent zero.
    """
    priced = structure._pricer()
    shared = frozenset(str(mode) for mode in shared_modes)
    for journey in journeys:
        journey["fare"] = priced(journey) + _street_cost(
            journey, structure.street, shared
        )
    return journeys


def _street_tariffs(street):
    """``street`` normalized to ``mode -> (unlock, per_minute)`` floats."""
    tariffs = {}
    for mode, entry in (street or {}).items():
        missing = {"unlock", "per_minute"} - set(entry)
        if missing:
            raise ValueError(
                f"street tariff for {mode!r} is missing {sorted(missing)}; "
                "state a zero component explicitly"
            )
        unlock = float(entry["unlock"])
        per_minute = float(entry["per_minute"])
        if unlock < 0 or per_minute < 0:
            raise ValueError(f"street tariff for {mode!r} must not be negative")
        tariffs[str(mode)] = (unlock, per_minute)
    return tariffs


def _street_cost(journey, street, shared_modes):
    """The journey's rental cost: per shared-mode leg, its mode's unlock
    plus started minutes; ``NaN`` when a ridden rental mode has no
    tariff. Own-vehicle and walking legs are free — a fare is what is
    paid, never an imputed cost."""
    total = 0.0
    for leg in journey["legs"]:
        mode = leg.get("mode")
        if mode is None or mode == "walk" or mode not in shared_modes:
            continue
        tariff = street.get(mode)
        if tariff is None:
            return math.nan
        unlock, per_minute = tariff
        minutes = math.ceil((leg["arrival_s"] - leg["departure_s"]) / 60.0)
        total += unlock + minutes * per_minute
    return total


def _framed(frame, columns):
    """A DataFrame with the expected columns (reordered, validated)."""
    if frame is None:
        return pd.DataFrame(columns=columns)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"fare table is missing columns {missing}")
    return pd.DataFrame(frame)[columns].reset_index(drop=True)
