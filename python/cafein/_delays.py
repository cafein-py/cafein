"""The car intersection-delay model and its query-time resolution.

The shipped values are Jaakkola's (2013) calibration — the MSc thesis behind
the MetropAccess-Digiroad drive-time model (regression on HSL 2009
floating-car data), the calibration underlying Tenkanen & Toivonen (2020)
and the GEMMAT framework. ``resolve`` gates the model behind
``intersection_delays`` (the default regime is free-flow), merges a
``delay_model=`` override over the shipped numbers, and flattens the chosen
period into the payload the Rust compiler consumes.
"""

import math

from . import _osm

PROFILES = ("rush", "midday", "day-average")
"""The delay periods: rush (07–09 and 15–17), midday (09–15), and the
day-average (06–22)."""

_GROUPS = ("1-2", "3", "4-6")
"""The functional road-class groups the calibration keys its values by."""

DELAY_MODEL = {
    # Taulukko 28: the per-crossing penalty `b` in seconds, by group and
    # period.
    "values": {
        "1-2": {"rush": 12.195, "midday": 9.979, "day-average": 11.311},
        "3": {"rush": 11.199, "midday": 6.650, "day-average": 9.439},
        "4-6": {"rush": 10.633, "midday": 7.752, "day-average": 9.362},
    },
    # The OSM mapping onto the groups; every drivable class not named here
    # is group 4-6. The `*_link` classes are the ramp category.
    "groups": {
        "motorway": "1-2",
        "trunk": "1-2",
        "primary": "1-2",
        "secondary": "3",
        "tertiary": "3",
    },
    # Liite 14: the multiplier on a junction-free ramp element at or above
    # 70 km/h.
    "ramp_multipliers": {
        "rush": 2.022762,
        "midday": 1.667750,
        "day-average": 1.884662,
    },
    # The multiplier on a junction-free non-ramp element at or above
    # 70 km/h; midday carries none.
    "congestion_multipliers": {"rush": 1.2, "midday": 1.0, "day-average": 1.1},
    # The share of its own `b` a ramp element charges per junction endpoint
    # at or above 70 km/h; below 70 km/h the share is ½ in every period.
    "ramp_shares": {"rush": 0.5, "midday": 0.75, "day-average": 2.0 / 3.0},
}

RAMP_SHARE_LOW = 0.5
"""The below-70-km/h ramp share, every period (the calibration's low-speed
branch); not part of the override surface."""


def _checked_number(value, where):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"delay_model{where} must be a non-negative finite number")
    return float(value)


def _merged(delay_model):
    """The shipped model with a ``delay_model=`` override partially merged.

    Merge semantics mirror ``speed_limits=``: each recognised key overrides
    only the entries it names; unknown keys, groups, classes, periods, and
    malformed numbers are rejected loudly.
    """
    merged = {
        key: {
            name: dict(row) if isinstance(row, dict) else row
            for name, row in table.items()
        }
        for key, table in DELAY_MODEL.items()
    }
    if delay_model is None:
        return merged
    unknown = sorted(set(delay_model) - set(DELAY_MODEL))
    if unknown:
        raise ValueError("unknown delay_model keys: " + ", ".join(map(str, unknown)))
    values = delay_model.get("values", {})
    for group, row in dict(values).items():
        if group not in _GROUPS:
            raise ValueError(
                f"delay_model['values'] group {group!r} is not one of {_GROUPS}"
            )
        for period, seconds in dict(row).items():
            if period not in PROFILES:
                raise ValueError(
                    f"delay_model['values'][{group!r}] period {period!r} is "
                    f"not one of {PROFILES}"
                )
            merged["values"][group][period] = _checked_number(
                seconds, f"['values'][{group!r}][{period!r}]"
            )
    for name, group in dict(delay_model.get("groups", {})).items():
        if name not in _osm.HIGHWAY_CODES:
            raise ValueError(f"delay_model['groups'] class {name!r} is unknown")
        if group not in _GROUPS:
            raise ValueError(
                f"delay_model['groups'][{name!r}] must be one of {_GROUPS}"
            )
        merged["groups"][name] = group
    for key in ("ramp_multipliers", "congestion_multipliers", "ramp_shares"):
        for period, value in dict(delay_model.get(key, {})).items():
            if period not in PROFILES:
                raise ValueError(
                    f"delay_model[{key!r}] period {period!r} is not one of "
                    f"{PROFILES}"
                )
            merged[key][period] = _checked_number(value, f"[{key!r}][{period!r}]")
    return merged


def resolve(intersection_delays=False, profile=None, delay_model=None):
    """``None`` (the free-flow default) or the flat core payload.

    The payload is one period's numbers: ``(group_seconds, groups,
    ramp_share_high, ramp_share_low, ramp_multiplier,
    congestion_multiplier)`` with ``groups`` indexed by highway code.
    ``profile=`` and ``delay_model=`` without ``intersection_delays=True``
    raise — the realistic model is never switched on implicitly and a
    period is never silently ignored.
    """
    if not intersection_delays:
        if profile is not None or delay_model is not None:
            raise ValueError(
                "profile= and delay_model= configure the intersection-delay "
                "model; pass intersection_delays=True to enable it"
            )
        return None
    merged = _merged(delay_model)
    period = "midday" if profile is None else profile
    if period not in PROFILES:
        raise ValueError(f"unknown profile {period!r}; expected one of {PROFILES}")
    groups = [_GROUPS.index("4-6")] * len(_osm.HIGHWAY_CODES)
    for name, group in merged["groups"].items():
        groups[_osm.HIGHWAY_CODES[name]] = _GROUPS.index(group)
    return (
        [merged["values"][group][period] for group in _GROUPS],
        groups,
        merged["ramp_shares"][period],
        RAMP_SHARE_LOW,
        merged["ramp_multipliers"][period],
        merged["congestion_multipliers"][period],
    )
