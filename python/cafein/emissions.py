"""Per-journey carbon emissions from per-leg distances.

Emission factors live in a long-format, user-overridable table and are
resolved per transit leg through a most-specific-wins ladder over the
leg's GTFS identity: trip_id → route_id → (agency_id, route_type) →
route_type → global default. How detailed the available information is
varies place by place, so factors can be allocated at any of those
levels — a single city line with electric buses gets its own row without
touching the mode averages. Walking legs emit nothing.

Factors are grams CO₂e per passenger-km, occupancy baked in by the table
author, split into life-cycle components (vehicle manufacturing,
fuel/energy, infrastructure, operational services) that sum to the
applied factor — operational-only and full life-cycle emissions come
from the same table by selecting components.
"""

import json
import os
import pathlib
import warnings

import pandas as pd

KEY_COLUMNS = ["trip_id", "route_id", "agency_id", "route_type"]
COMPONENT_COLUMNS = ["vehicle", "fuel", "infrastructure", "operations"]


def default_factors():
    """The shipped factor table, in g CO₂e per passenger-km.

    Life-cycle estimates from the International Transport Forum's LCA
    tool, calibrated to the Finnish electricity mix (Dey, Marín-Flores &
    Tenkanen 2026, Table 5): buses carry the ICE-bus factors, and tram,
    metro, and rail share the metro/urban-train factors. Layer your own
    rows over these through the `factors` argument of `annotate`;
    `vehicle_class_factors` lists the per-powertrain values to build
    such rows from.
    """
    classes = vehicle_class_factors()
    bus = classes.loc["bus-ICE"].to_dict()
    urban_rail = classes.loc["metro-urban-train"].to_dict()
    rows = [
        {"route_type": 3, **bus},
        {"route_type": 0, **urban_rail},
        {"route_type": 1, **urban_rail},
        {"route_type": 2, **urban_rail},
    ]
    return pd.DataFrame(rows).reindex(columns=KEY_COLUMNS + COMPONENT_COLUMNS)


def vehicle_class_factors():
    """Life-cycle factors per vehicle class, in g CO₂e per passenger-km.

    The International Transport Forum's LCA components calibrated to the
    Finnish electricity mix, from Dey, Marín-Flores & Tenkanen (2026,
    Tables 4-5, doi:10.1016/j.scs.2026.107226). Use these to author
    route- or trip-level factor rows — e.g. give a city's electrified
    bus lines the ``bus-BEV`` values, the GEMMAT approach — and, for the
    private-vehicle classes, future direct-mode tables.
    """
    rows = {
        "bus-ICE": (8.0, 72.0, 4.0, 8.0),
        "bus-HEV": (8.0, 53.0, 4.0, 6.0),
        "bus-BEV": (14.0, 10.0, 4.0, 1.0),
        "bus-BEV-2packs": (17.0, 10.0, 4.0, 1.0),
        "bus-FCEV": (11.0, 44.0, 4.0, 5.0),
        "metro-urban-train": (2.0, 12.0, 11.0, 0.0),
        "car-ICE": (24.0, 126.0, 12.0, 0.0),
        "car-HEV": (26.0, 94.0, 13.0, 0.0),
        "car-PHEV": (32.0, 43.0, 13.0, 0.0),
        "car-BEV": (42.0, 16.0, 12.0, 0.0),
        "car-FCEV": (38.0, 83.0, 13.0, 0.0),
    }
    frame = pd.DataFrame.from_dict(rows, orient="index", columns=COMPONENT_COLUMNS)
    frame.index.name = "vehicle_class"
    return frame


def load_factors(source):
    """Load and validate a factor table.

    Parameters
    ----------
    source : DataFrame or path
        A long-format table: any of the key columns ``trip_id``,
        ``route_id``, ``agency_id``, ``route_type`` (empty where not
        applicable) plus one or more of the component columns
        ``vehicle``, ``fuel``, ``infrastructure``, ``operations`` in
        g CO₂e per passenger-km. Paths may point to CSV, JSON (a list of
        mappings), or YAML (the same, via the optional PyYAML
        dependency).

    Returns
    -------
    DataFrame
        The normalized table, with every key and component column
        present and components validated as non-negative numbers.
    """
    if isinstance(source, pd.DataFrame):
        frame = source.copy()
    elif isinstance(source, (str, os.PathLike)):
        frame = _read_factor_file(pathlib.Path(source))
    else:
        raise TypeError(f"cannot load emission factors from {type(source).__name__}")

    unknown = set(frame.columns) - set(KEY_COLUMNS + COMPONENT_COLUMNS)
    if unknown:
        raise ValueError(
            f"unknown factor-table column(s): {', '.join(sorted(unknown))}"
        )
    components = [column for column in COMPONENT_COLUMNS if column in frame.columns]
    if not components:
        raise ValueError(
            "a factor table needs at least one component column "
            f"({', '.join(COMPONENT_COLUMNS)})"
        )
    frame = frame.reindex(columns=KEY_COLUMNS + COMPONENT_COLUMNS)
    # Blank cells mean "not given" whatever the input format wrote them
    # as (CSVs are read without pandas' NA-token guessing, so ids like
    # "NA" survive as real identifiers).
    for column in frame.columns:
        frame[column] = frame[column].map(_blank_to_na)
    for column in COMPONENT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if (frame[column] < 0).any():
            raise ValueError(f"negative values in factor component '{column}'")
    # A keyed row with no component values would resolve as zero
    # emissions and shadow broader rows.
    empty = frame[COMPONENT_COLUMNS].isna().all(axis=1)
    if empty.any():
        raise ValueError(f"{int(empty.sum())} factor row(s) have no component values")
    # Identity keys must compare equal to the string ids journey legs
    # carry, whatever type they were delivered as — including ids that
    # pandas coerced to floats in mixed-key record input.
    for column in ["trip_id", "route_id", "agency_id"]:
        frame[column] = frame[column].map(_key_string, na_action="ignore")
    frame["route_type"] = pd.to_numeric(frame["route_type"], errors="raise")
    located = frame["route_type"].dropna()
    if ((located % 1 != 0) | (located < 0)).any():
        raise ValueError("route_type keys must be non-negative integers")
    return frame


def _blank_to_na(value):
    if isinstance(value, str) and not value.strip():
        return pd.NA
    return value


def _key_string(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def annotate(journeys, network, factors=None, components=None):
    """Attach emissions to journeys, in place, and return them.

    Parameters
    ----------
    journeys : list of dict
        Journeys as returned by ``route_between_stops``, with per-leg
        distances installed (``trip_distances=True``, the default).
    network : TransportNetwork
        The network the journeys were routed on; provides the route
        metadata the factor ladder resolves against.
    factors : DataFrame or path (optional)
        Extra factor rows (see `load_factors`), layered over the shipped
        defaults — at equal specificity, later rows win.
    components : list of str (optional)
        The life-cycle components to include (default: all four). For
        example ``["fuel", "operations"]`` yields operational-scope
        emissions comparable to use-phase inventories.

    Returns
    -------
    list of dict
        The journeys, each leg carrying ``emissions`` in grams CO₂e
        (zero for walking legs, ``None`` where no factor row matches)
        and each journey carrying the summed ``emissions`` (``None`` if
        any of its transit legs is unresolved).
    """
    if components is None:
        components = COMPONENT_COLUMNS
    else:
        unknown = set(components) - set(COMPONENT_COLUMNS)
        if unknown:
            raise ValueError(f"unknown component(s): {', '.join(sorted(unknown))}")
        components = [
            column for column in COMPONENT_COLUMNS if column in set(components)
        ]
        if not components:
            raise ValueError("components must name at least one component column")
    table = default_factors()
    if factors is not None:
        table = pd.concat([table, load_factors(factors)], ignore_index=True)
    resolver = _Resolver(table, components)
    routes = {
        route_id: (agency_id, route_type)
        for route_id, agency_id, route_type in network.routes
    }
    unmatched = set()
    for journey in journeys:
        total = 0.0
        complete = True
        for leg in journey["legs"]:
            if leg["type"] != "transit":
                leg["emissions"] = 0.0
                continue
            if leg.get("distance") is None:
                raise ValueError(
                    "journey legs carry no distances; build the network "
                    "with trip distances enabled"
                )
            agency_id, route_type = routes[leg["route_id"]]
            factor = resolver.resolve(
                leg["trip_id"], leg["route_id"], agency_id, route_type
            )
            if factor is None:
                leg["emissions"] = None
                complete = False
                unmatched.add(route_type)
            else:
                leg["emissions"] = leg["distance"] / 1000.0 * factor
                total += leg["emissions"]
        journey["emissions"] = total if complete else None
    if unmatched:
        warnings.warn(
            f"no emission factor matches route_type(s) {sorted(unmatched)}; "
            "the affected legs carry no emissions",
            stacklevel=2,
        )
    return journeys


def trip_factors(network, factors=None, components=None):
    """Resolve one emission factor per routable trip through the ladder.

    The per-trip form the bulk computations consume: the most specific
    matching factor row for every trip of the network, in grams CO₂e per
    passenger-kilometer.

    Parameters
    ----------
    network : TransportNetwork
        The network whose trips to resolve.
    factors : DataFrame or path (optional)
        Extra factor rows layered over the shipped defaults, as in
        `annotate`.
    components : list of str (optional)
        The life-cycle components to include (default: all four).

    Returns
    -------
    list of (str, float)
        ``(trip_id, factor)`` pairs covering every routable trip;
        ``float("nan")`` marks trips no factor row matches (a warning
        names the affected route types, as in `annotate`).
    """
    if components is None:
        components = COMPONENT_COLUMNS
    else:
        unknown = set(components) - set(COMPONENT_COLUMNS)
        if unknown:
            raise ValueError(f"unknown component(s): {', '.join(sorted(unknown))}")
        components = [
            column for column in COMPONENT_COLUMNS if column in set(components)
        ]
        if not components:
            raise ValueError("components must name at least one component column")
    table = default_factors()
    if factors is not None:
        table = pd.concat([table, load_factors(factors)], ignore_index=True)
    resolver = _Resolver(table, components)
    routes = {
        route_id: (agency_id, route_type)
        for route_id, agency_id, route_type in network.routes
    }
    unmatched = set()
    results = []
    for trip_id, route_id in network.trips:
        agency_id, route_type = routes[route_id]
        factor = resolver.resolve(trip_id, route_id, agency_id, route_type)
        if factor is None:
            unmatched.add(route_type)
            factor = float("nan")
        results.append((trip_id, factor))
    if unmatched:
        warnings.warn(
            f"no emission factor matches route_type(s) {sorted(unmatched)}; "
            "journeys riding the affected trips carry no emissions",
            stacklevel=2,
        )
    return results


def _read_factor_file(path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(
            path,
            dtype={"trip_id": str, "route_id": str, "agency_id": str},
            keep_default_na=False,
        )
    if suffix == ".json":
        records = json.loads(path.read_text(encoding="utf-8"))
    elif suffix in (".yml", ".yaml"):
        try:
            import yaml
        except ImportError as error:
            raise ImportError(
                "reading YAML factor tables needs the optional PyYAML "
                "dependency (pip install pyyaml)"
            ) from error
        records = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"unsupported factor-table format '{path.suffix}'")
    if not isinstance(records, list):
        raise ValueError("a factor file must hold a list of mappings")
    return pd.DataFrame.from_records(records)


class _Resolver:
    """The most-specific-wins ladder over a factor table.

    Rows are bucketed by the most specific key they carry; within a
    bucket, later rows win, so user rows layered after the defaults
    override them at equal specificity.
    """

    def __init__(self, table, components=COMPONENT_COLUMNS):
        factor = table[list(components)].fillna(0.0).sum(axis=1)
        trip = table["trip_id"].notna()
        route = table["route_id"].notna()
        agency = table["agency_id"].notna()
        mode = table["route_type"].notna()
        self.by_trip = dict(zip(table.loc[trip, "trip_id"], factor[trip]))
        by_route = ~trip & route
        self.by_route = dict(zip(table.loc[by_route, "route_id"], factor[by_route]))
        by_agency = ~trip & ~route & agency & mode
        self.by_agency_mode = dict(
            zip(
                zip(
                    table.loc[by_agency, "agency_id"],
                    table.loc[by_agency, "route_type"].astype(int),
                ),
                factor[by_agency],
            )
        )
        by_mode = ~trip & ~route & ~agency & mode
        self.by_mode = dict(
            zip(table.loc[by_mode, "route_type"].astype(int), factor[by_mode])
        )
        rest = table[~trip & ~route & ~agency & ~mode]
        self.default = float(factor[rest.index[-1]]) if len(rest) else None

    def resolve(self, trip_id, route_id, agency_id, route_type):
        candidates = [route_type, _base_route_type(route_type)]
        if trip_id in self.by_trip:
            return self.by_trip[trip_id]
        if route_id in self.by_route:
            return self.by_route[route_id]
        for mode in candidates:
            if mode is not None and (agency_id, mode) in self.by_agency_mode:
                return self.by_agency_mode[(agency_id, mode)]
        for mode in candidates:
            if mode is not None and mode in self.by_mode:
                return self.by_mode[mode]
        return self.default


def _base_route_type(route_type):
    """The base GTFS mode of an extended route_type code, if any."""
    for base, start, end in [
        (2, 100, 200),  # railway services
        (3, 200, 300),  # coach services
        (1, 400, 500),  # urban railway services
        (3, 700, 800),  # bus services
        (3, 800, 900),  # trolleybus
        (0, 900, 1000),  # tram services
        (4, 1000, 1100),  # water transport
        (4, 1200, 1300),  # ferry
        (6, 1300, 1400),  # aerial lift
        (7, 1400, 1500),  # funicular
    ]:
        if start <= route_type < end:
            return base
    return None
