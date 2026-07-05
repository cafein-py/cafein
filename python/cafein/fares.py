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
- **Zone-based** (`ZoneFareStructure`) prices journeys from GTFS
  ``fare_attributes.txt``/``fare_rules.txt`` with ``contains_id`` zone
  sets — the model Helsinki Region Transport ships: a ticket is the
  cheapest fare whose zone set covers the zones a stretch of the journey
  touches, valid for unlimited boardings within its transfer window, and
  a long journey chains tickets. Legs contribute their boarding and
  alighting stops' zones (with contiguous zone products, as in
  ring-shaped systems, that equals the traversed span). Fare rules keyed
  by route or origin/destination zone pairs are not modelled.
"""

import math
import zipfile

import pandas as pd

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
    ):
        self.max_discounted_transfers = int(max_discounted_transfers)
        self.transfer_time_allowance = float(transfer_time_allowance)
        self.fare_cap = float(fare_cap)
        self.fares_per_type = _framed(fares_per_type, _TYPE_COLUMNS)
        self.fares_per_transfer = _framed(fares_per_transfer, _TRANSFER_COLUMNS)
        self.fares_per_route = _framed(fares_per_route, _ROUTE_COLUMNS)

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
            previous_board = rides[0]["departure"]
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
                in_time = ride["departure"] - previous_board <= allowance
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
                previous_board = ride["departure"]
            if math.isfinite(self.fare_cap):
                total = min(total, self.fare_cap)
            return total

        return priced


class ZoneFareStructure:
    """Zone-set fares from GTFS ``fare_attributes``/``fare_rules``.

    Attributes
    ----------
    fares : pandas.DataFrame
        One row per fare product: ``fare_id``, ``price``,
        ``currency_type``, ``transfers`` (NaN: unlimited within the
        window), ``transfer_duration`` (seconds of validity from the
        first boarding).
    fare_zones : dict
        ``fare_id`` → frozenset of the zones the product covers.
    stop_zones : dict
        ``stop_id`` → zone.
    """

    def __init__(self, fares, fare_zones, stop_zones):
        self.fares = fares
        self.fare_zones = {fare: frozenset(zones) for fare, zones in fare_zones.items()}
        self.stop_zones = dict(stop_zones)

    def price(self, journey):
        """The journey's fare: the cheapest chain of zone tickets.

        Each ticket must cover the zones of every leg it spans (a leg
        contributes its boarding and alighting stops' zones) and every
        boarding it covers must fall within its transfer window; a
        journey longer than one window chains tickets. Returns NaN when
        a leg's zones are outside every product or a stop has no zone.
        """
        return self._pricer()(journey)

    def _pricer(self):
        products = []
        for _, row in self.fares.iterrows():
            zones = self.fare_zones.get(str(row["fare_id"]))
            if zones is None:
                continue
            # `transfers` and `transfer_duration` are optional columns:
            # absent means unlimited boardings without a time limit.
            duration = row.get("transfer_duration")
            transfers = row.get("transfers")
            products.append(
                (
                    float(row["price"]),
                    zones,
                    math.inf if pd.isna(duration) else float(duration),
                    math.inf if pd.isna(transfers) else int(transfers),
                )
            )

        def priced(journey):
            rides = [leg for leg in journey["legs"] if leg["type"] == "transit"]
            if not rides:
                return 0.0
            needs = []
            for ride in rides:
                zones = {
                    self.stop_zones.get(ride["board_stop"]),
                    self.stop_zones.get(ride["alight_stop"]),
                }
                if None in zones:
                    return math.nan
                needs.append((frozenset(zones), ride["departure"]))

            best = {len(needs): 0.0}

            def cost(at):
                if at in best:
                    return best[at]
                cheapest = math.nan
                for price, zones, duration, transfers in products:
                    if not needs[at][0] <= zones:
                        continue
                    # The ticket covers boardings within its window (and
                    # its transfer count), as far as the zones allow.
                    end = at
                    while (
                        end + 1 < len(needs)
                        and needs[end + 1][0] <= zones
                        and needs[end + 1][1] - needs[at][1] <= duration
                        and (end + 1 - at) <= transfers
                    ):
                        end += 1
                    for split in range(at, end + 1):
                        rest = cost(split + 1)
                        candidate = price + rest
                        if math.isnan(cheapest) or candidate < cheapest:
                            cheapest = candidate
                best[at] = cheapest
                return cheapest

            return cost(0)

        return priced


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


def zone_fare_structure(gtfs_path):
    """A zone-based fare structure from a GTFS feed's fare files.

    Reads ``fare_attributes.txt``, the ``contains_id`` rows of
    ``fare_rules.txt``, and the stops' ``zone_id`` column. Fare rules
    keyed by route or origin/destination zones are ignored (fares built
    purely from such rules get no zone set and never price a journey).
    """
    with zipfile.ZipFile(gtfs_path) as archive:
        names = set(archive.namelist())
        if not {"fare_attributes.txt", "fare_rules.txt"} <= names:
            raise ValueError(f"'{gtfs_path}' carries no GTFS fare files")
        attributes = pd.read_csv(
            archive.open("fare_attributes.txt"), dtype={"fare_id": str}
        )
        rules = pd.read_csv(archive.open("fare_rules.txt"), dtype=str)
        stops = pd.read_csv(archive.open("stops.txt"), dtype=str)
    # `contains_id` and `zone_id` are optional columns: a feed whose
    # fare rules are purely route- or origin/destination-keyed yields no
    # zone products, and prices nothing.
    fare_zones = {}
    if "contains_id" in rules.columns:
        contains = rules[rules["contains_id"].notna()]
        fare_zones = {
            fare: set(group["contains_id"])
            for fare, group in contains.groupby("fare_id")
        }
    stop_zones = {}
    if "zone_id" in stops.columns:
        stop_zones = {
            row.stop_id: row.zone_id
            for row in stops.itertuples()
            if isinstance(row.zone_id, str)
        }
    return ZoneFareStructure(attributes, fare_zones, stop_zones)


def annotate_fares(journeys, structure):
    """Attach ``fare`` to journeys, in place, and return them.

    Parameters
    ----------
    journeys : list of dict
        Journeys as returned by the routing calls.
    structure : FareStructure or ZoneFareStructure
        The fare model to price with.
    """
    priced = structure._pricer()
    for journey in journeys:
        journey["fare"] = priced(journey)
    return journeys


def _framed(frame, columns):
    """A DataFrame with the expected columns (reordered, validated)."""
    if frame is None:
        return pd.DataFrame(columns=columns)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"fare table is missing columns {missing}")
    return pd.DataFrame(frame)[columns].reset_index(drop=True)
