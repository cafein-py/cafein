"""Per-kilometre monetary costs of street travel, by perspective.

A **separate account from fares**: a fare is what is paid to an operator
under a fare structure, while these are per-kilometre cost accounting —
the two are never summed silently, and each follows its own
missing-row-is-NaN rule. The shipped values are Gössling, Choi, Dekker &
Metzler (2019, Ecological Economics 158, doi:10.1016/j.ecolecon.2018.12.016,
Table 2 — EU averages in 2017 euros) under two perspectives:

- ``private`` — the vehicle-operation bundle (fuel, oil and tire wear,
  maintenance, depreciation, parking fees, tolls, insurance), one
  ``vehicle_operation`` component per mode. Travel-time and congestion
  values are excluded by design: time is already cafein's first axis
  and would double-count.
- ``societal`` — the external cost. The car resolves component by
  component; cycling and walking ship the paper's aggregated external
  total as one signed ``external`` component — their health-dominated
  benefits are **negative and reported signed, never clamped**.

Costs are matched by ``(perspective, street_mode)`` only — vehicle class
and service model do not participate (the shipped values are
mode-level; finer user rows use the same columns). Totals are always
derived by summing the selected components, never stored independently.
Currency and base year are metadata: one declared label carried
verbatim beside the outputs, never converted — a caller supplying
override rows declares whatever their values are in, and mixing
currencies is the caller's responsibility. Public-transport perspective
costs are out of scope: the schema is street-only and transit rows are
rejected loudly.
"""

import pathlib
import warnings

import numpy as np
import pandas as pd

PERSPECTIVES = ("private", "societal")

COST_KEY_COLUMNS = ["perspective", "street_mode", "component"]

CURRENCY = "EUR2017"
"""The shipped values' declared currency and base year — a label, never
a conversion."""

_STREET_MODES = ("walk", "bicycle", "e_scooter", "car")
"""The street modes a cost row may name; transit rows are rejected."""


def street_costs():
    """The shipped street cost table, in currency units per kilometre.

    Gössling et al. (2019, Table 2; EU averages, 2017 euros). The
    ``private`` rows are each mode's vehicle-operation bundle; the
    ``societal`` car resolves into the paper's component breakdown
    (summing to its +0.108 €/km total) while cycling and walking carry
    their aggregated external totals signed (−0.184 and −0.370 €/km,
    health-dominated benefits). No row ships for the e-scooter (the
    paper has none): its costs stay unresolved unless user rows supply
    them.
    """
    rows = [
        ("private", "car", "vehicle_operation", 0.250),
        ("private", "bicycle", "vehicle_operation", 0.047),
        ("private", "walk", "vehicle_operation", 0.041),
        ("societal", "car", "climate", 0.011),
        ("societal", "car", "subsidies", 0.003),
        ("societal", "car", "air_pollution", 0.007),
        ("societal", "car", "noise", 0.007),
        ("societal", "car", "soil_water", 0.005),
        ("societal", "car", "infrastructure_construction", 0.030),
        ("societal", "car", "roadway_land", 0.011),
        ("societal", "car", "parking_land", 0.021),
        ("societal", "car", "infrastructure_maintenance", 0.004),
        ("societal", "car", "resources", 0.007),
        ("societal", "car", "accidents", 0.002),
        ("societal", "bicycle", "external", -0.184),
        ("societal", "walk", "external", -0.370),
    ]
    return pd.DataFrame(rows, columns=COST_KEY_COLUMNS + ["cost_per_km"])


def load_street_costs(source):
    """Load and validate a street cost table.

    A long-format table with the columns ``perspective`` (``private`` or
    ``societal``), ``street_mode`` (a street mode — transit rows are
    rejected, never ignored), ``component`` (a non-empty name), and
    ``cost_per_km`` (finite; negative marks a benefit). Paths may point
    to CSV, JSON (a list of mappings), or YAML, as the factor tables do.
    """
    from cafein.emissions import _read_factor_file

    if isinstance(source, pd.DataFrame):
        frame = source.copy()
    else:
        frame = _read_factor_file(pathlib.Path(source))
    missing = [
        column
        for column in COST_KEY_COLUMNS + ["cost_per_km"]
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"cost table misses column(s): {', '.join(missing)}")
    frame = frame.reindex(columns=COST_KEY_COLUMNS + ["cost_per_km"])
    unknown = set(frame["perspective"]) - set(PERSPECTIVES)
    if unknown:
        raise ValueError(
            "unknown cost perspective(s): " + ", ".join(sorted(map(str, unknown)))
        )
    off_street = set(frame["street_mode"]) - set(_STREET_MODES)
    if off_street:
        raise ValueError(
            "cost rows are street-only; unknown street_mode(s): "
            + ", ".join(sorted(map(str, off_street)))
        )
    if not all(isinstance(name, str) and name for name in frame["component"]):
        raise ValueError("every cost row needs a non-empty component name (a string)")
    values = pd.to_numeric(frame["cost_per_km"], errors="raise")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("cost_per_km must be finite (negative marks a benefit)")
    frame["cost_per_km"] = values
    if frame.duplicated(subset=COST_KEY_COLUMNS).any():
        raise ValueError("duplicate (perspective, street_mode, component) cost row(s)")
    return frame.reset_index(drop=True)


def merged_costs(costs=None):
    """The shipped table with a user table layered over it.

    User rows replace shipped rows by their
    ``(perspective, street_mode, component)`` key and add new components
    otherwise — the house partial-merge.
    """
    shipped = street_costs()
    if costs is None:
        return shipped
    user = load_street_costs(costs)
    user_keys = pd.MultiIndex.from_frame(user[COST_KEY_COLUMNS])
    shipped_keys = pd.MultiIndex.from_frame(shipped[COST_KEY_COLUMNS])
    kept = shipped[~shipped_keys.isin(user_keys)]
    return pd.concat([kept, user], ignore_index=True)


def _cost_mode(transport_mode):
    """The cost-table mode a transport mode prices as, e-bike → bicycle."""
    from cafein.emissions import STREET_MODE_IDENTITIES

    if transport_mode not in STREET_MODE_IDENTITIES:
        raise ValueError(
            f"unknown street mode {transport_mode!r}; expected one of "
            f"{', '.join(sorted(STREET_MODE_IDENTITIES))}"
        )
    return STREET_MODE_IDENTITIES[transport_mode][0]


def _resolved(table, mode, perspective, components):
    """``(total, per_component)`` for one pair of an already-merged table.

    An uncovered pair is wholly unresolved: NaN total, NaN per requested
    component, one warning — never a silent zero and never a KeyError.
    A covered pair rejects a selected component it does not carry.
    """
    if perspective not in PERSPECTIVES:
        raise ValueError(
            f"unknown cost perspective {perspective!r}; expected one of "
            f"{', '.join(PERSPECTIVES)}"
        )
    rows = table[(table.perspective == perspective) & (table.street_mode == mode)]
    if rows.empty:
        warnings.warn(
            f"no {perspective} cost row matches street mode '{mode}'; "
            "costs stay unresolved",
            stacklevel=4,
        )
        nan = float("nan")
        return nan, {component: nan for component in components or ()}
    if components is not None:
        known = set(rows.component)
        if isinstance(components, str):
            raise TypeError(
                f"cost_components must be an iterable of component names, "
                f"not the string {components!r} — pass ({components!r},)"
            )
        unknown = set(components) - known
        if unknown:
            raise ValueError(
                f"cost component(s) {', '.join(sorted(unknown))} not carried "
                f"by the {perspective} {mode} rows ({', '.join(sorted(known))})"
            )
        rows = rows[rows.component.isin(components)]
    per_component = dict(zip(rows.component, rows.cost_per_km))
    return float(rows.cost_per_km.sum()), per_component


def street_cost(transport_mode, perspective, costs=None, components=None):
    """The resolved cost per kilometre for one mode and perspective.

    The mode maps to its cost mode as the emissions identities do — the
    e-bike rides the bicycle rows. The result is the sum of the selected
    `components` (default: every component the pair carries); selecting
    a component the pair does not carry is rejected loudly. A pair with
    no rows at all resolves NaN — unresolved, never a silent zero —
    with a warning.
    """
    mode = _cost_mode(transport_mode)
    components = _checked_components(components)
    total, _ = _resolved(merged_costs(costs), mode, perspective, components)
    return total


def _checked_components(components):
    if components is None:
        return None
    if isinstance(components, str):
        raise TypeError(
            f"cost_components must be an iterable of component names, not "
            f"the string {components!r} — pass ({components!r},)"
        )
    chosen = list(components)
    if not chosen or not all(isinstance(name, str) and name for name in chosen):
        raise ValueError("cost_components must be non-empty component names")
    return chosen


def resolve_query(transport_mode, perspectives, costs, currency, cost_components):
    """The validated cost account of a street query, resolved up front.

    Returns ``None`` when `perspectives` is ``None`` (the default —
    costs are opt-in; the other options are then rejected, never
    ignored), else ``(totals, breakdown, currency)``: `totals` maps each
    selected perspective to its per-km cost (the sum of the selected
    components), and `breakdown` maps ``(perspective, component)`` to
    per-km values when `cost_components` requests the per-component
    columns — a selection that requires exactly one perspective, so
    every requested column has one unambiguous vocabulary. The user
    table and the selection are materialized exactly once, so every
    reported number comes from one consistent snapshot.
    """
    if perspectives is None:
        if costs is not None or cost_components is not None or currency is not None:
            raise ValueError(
                "costs=, currency=, and cost_components= configure the cost "
                "account; pass perspectives= to enable it"
            )
        return None
    chosen = [perspectives] if isinstance(perspectives, str) else list(perspectives)
    if not chosen or set(chosen) - set(PERSPECTIVES) or len(set(chosen)) != len(chosen):
        raise ValueError(
            f"perspectives must be a non-empty selection from {PERSPECTIVES}"
        )
    currency = CURRENCY if currency is None else currency
    if not isinstance(currency, str) or not currency:
        raise ValueError("currency must be a non-empty label, e.g. 'EUR2017'")
    cost_components = _checked_components(cost_components)
    if cost_components is not None and len(chosen) != 1:
        raise ValueError(
            "cost_components selects within one perspective's component "
            "vocabulary; pass a single perspective with it"
        )
    mode = _cost_mode(transport_mode)
    table = merged_costs(costs)
    totals, breakdown = {}, {}
    for perspective in chosen:
        total, per_component = _resolved(table, mode, perspective, cost_components)
        totals[perspective] = total
        if cost_components is not None:
            for component in cost_components:
                breakdown[(perspective, component)] = per_component[component]
    return totals, breakdown, currency
