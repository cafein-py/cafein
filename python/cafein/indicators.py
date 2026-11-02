"""Per-trip indicators beside travel time, emissions and money.

Three indicators, each opt-in through ``indicators=``:

- ``physical_activity`` — MET-hours of walking and cycling: the km in
  each active mode times the mode's rate, its MET (energy use as a
  multiple of rest, the resting 1 MET included) divided by the speed the
  MET was measured or assumed at. MET-hours are the dose unit of the
  mortality evidence (Kelly et al. 2014).
- ``climate_cost`` — the trip's CO₂e emissions times a CO₂ price.
- ``pm10`` and ``pm2_5`` — the particles a car releases into street
  air (exhaust, tyre, brake and road wear, resuspended road dust), in mg
  per person: the km on each road class times a factor per vehicle-km,
  divided by occupancy.

Every shipped value has a ``"middle"`` case, the recommended default,
and a ``"low"`` and a ``"high"`` case where a source gives one; ``low``
is always the smaller number. A case no source gives keeps the middle
value and is marked ``published=False``; a query that uses it warns.
Every row names its source:

- Kelly et al. (2014), International Journal of Behavioral Nutrition and
  Physical Activity 11, 132, doi:10.1186/s12966-014-0132-x.
- Berntsen et al. (2017), International Journal of Behavioral Nutrition
  and Physical Activity 14, 55, doi:10.1186/s12966-017-0513-z.
- Kozey et al. (2010), Medicine & Science in Sports & Exercise 42(9),
  1776-1784, doi:10.1249/mss.0b013e3181d479f2.
- Gojanovic et al. (2011), Medicine & Science in Sports & Exercise 43(11),
  2204-2210, doi:10.1249/mss.0b013e31821cbdc8.
- Payne et al. (2025), Journal of Transport & Health 41, 101963,
  doi:10.1016/j.jth.2024.101963.
- Riiser, Aadland & Nordengen (2026), Frontiers in Sports and Active
  Living 7, 1653833, doi:10.3389/fspor.2025.1653833.
- WHO Regional Office for Europe (2024), Health economic assessment tool
  (HEAT) for walking and for cycling: 2024 update, ISBN 9789289058377.
- European Commission, Handbook on the external costs of transport,
  version 2019 – 1.1, ISBN 978-92-76-18184-2.
- Beddows & Harrison (2021), Atmospheric Environment 244, 117886,
  doi:10.1016/j.atmosenv.2020.117886.
- Hicks, Green & Beevers (2023), Environmental Pollution 336, 122400,
  doi:10.1016/j.envpol.2023.122400.
"""

import collections.abc
import math
import pathlib

import numpy as np
import pandas as pd

from cafein import _log, _validate

INDICATORS = ("physical_activity", "climate_cost", "pm10", "pm2_5")

CASES = ("low", "middle", "high")

ROAD_CLASSES = ("urban", "rural", "motorway")

ACTIVITY_MODES = ("walk", "bicycle", "e_bike", "e_scooter", "wheelchair")
"""The street modes an activity rate may name; a car leg counts zero."""

CO2_PRICE_CURRENCY = "EUR2016"
"""The shipped CO₂ prices' currency and base year — a label."""

_KELLY = "Kelly et al. (2014), p. 3"
_BERNTSEN = "Berntsen et al. (2017), Table 2"
_GOJANOVIC = "Gojanovic et al. (2011), Table 2"
_HEAT = "WHO Regional Office for Europe (2024)"
_BEDDOWS = "Beddows & Harrison (2021)"
_HICKS = "Hicks, Green & Beevers (2023), p. 1"

# street_mode, case, MET, km/h, source, note
_ACTIVITY = [
    (
        "walk",
        "middle",
        4.0,
        5.3,
        _KELLY,
        "assumed; the intensity behind the walking risk",
    ),
    (
        "walk",
        "high",
        4.5,
        1.56 * 3.6,
        "Kozey et al. (2010), Table 4",
        "measured on a level treadmill at 1.56 m/s",
    ),
    ("bicycle", "low", 8.1, 20.7, _BERNTSEN, "measured on 8 riders, flat route"),
    (
        "bicycle",
        "middle",
        6.8,
        14.0,
        f"{_KELLY}; {_HEAT}, p. 63",
        "assumed; the intensity behind the cycling risk",
    ),
    (
        "bicycle",
        "high",
        8.2,
        10.3,
        _GOJANOVIC,
        "measured on 16 sedentary riders, uphill commute (mean grade 3.4 %)",
    ),
    (
        "e_bike",
        "low",
        5.9,
        21.5,
        "Riiser, Aadland & Nordengen (2026), abstract and Table 1",
        "measured on 19 commuters",
    ),
    (
        "e_bike",
        "middle",
        5.5,
        17.0,
        f"{_HEAT}, pp. 62-63",
        "the HEAT expert-panel value",
    ),
    (
        "e_bike",
        "high",
        7.3,
        15.1,
        _GOJANOVIC,
        "measured on 17 sedentary riders at standard assistance, uphill commute",
    ),
    (
        "e_scooter",
        "middle",
        1.6,
        12.0,
        "Payne et al. (2025), pp. 3-4",
        "measured on 42 riders standing on a scooter fixed to a treadmill",
    ),
]

# vehicle_class, road_class, case, PM10, PM2.5 (mg per vehicle-km)
_PARTICLES = [
    ("ICE", "urban", "middle", 40.4, 19.1),
    ("ICE", "rural", "middle", 31.0, 14.5),
    ("ICE", "motorway", "middle", 26.4, 12.6),
    ("BEV", "urban", "low", 31.4, 14.5),
    ("BEV", "urban", "middle", 32.1, 14.8),
    ("BEV", "urban", "high", 35.7, 16.2),
    ("BEV", "rural", "low", 28.6, 12.8),
    ("BEV", "rural", "middle", 28.9, 12.9),
    ("BEV", "rural", "high", 30.5, 13.5),
    ("BEV", "motorway", "low", 27.2, 11.9),
    ("BEV", "motorway", "middle", 27.3, 11.9),
    ("BEV", "motorway", "high", 27.7, 12.0),
]

_PARTICLE_SOURCES = {
    ("ICE", "middle"): (
        f"{_BEDDOWS}, Tables 6 and 8",
        "Euro 6 petrol car of 1349 kg",
    ),
    ("BEV", "middle"): (
        f"{_BEDDOWS}, Table 6 and p. 6",
        "BEV of 1667 kg, 10 % of braking by friction brakes",
    ),
    ("BEV", "low"): (
        f"{_HICKS}; {_BEDDOWS}",
        "5 % friction braking: regeneration cuts brake wear by up to 95 %",
    ),
    ("BEV", "high"): (
        f"{_HICKS}; {_BEDDOWS}",
        "36 % friction braking: the smallest cut, 64 %, across electrified cars",
    ),
}


def co2_prices():
    """The shipped CO₂ prices, per tonne CO₂e in 2016 euros.

    The European Commission handbook's short- and medium-run values, up
    to 2030 (Table 24): the cost of avoiding emissions in line with the
    Paris 1.5–2 °C target, each the mean of the literature's estimates
    without the lowest and highest. The middle value is the one the
    handbook prices all its own tables with; its long-run values
    (2040–2060: 156, 269 and 498) and other published prices can be
    passed as numbers.
    """
    source = "European Commission, Handbook 2019 – 1.1, Table 24"
    return pd.DataFrame(
        [
            {
                "case": case,
                "price": price,
                "currency": CO2_PRICE_CURRENCY,
                "published": True,
                "source": source,
            }
            for case, price in (("low", 60.0), ("middle", 100.0), ("high", 189.0))
        ]
    )


def activity_rates():
    """The shipped activity rates, in MET-hours per km.

    One row per street mode and case: the MET, the speed it belongs to,
    and the rate ``met / speed_kmh``. The middle values are the
    intensities the walking and cycling mortality risks were computed
    with (Kelly et al. 2014; the WHO HEAT tool for the e-bike), the
    e-scooter's a laboratory measurement; the low and high values are the
    most extreme measured pairs of MET and speed, their conditions in
    ``note`` (the high ones include uphill riding). A case no source
    gives keeps the middle value with ``published=False``. No source
    covers the wheelchair at cafein's speed: it has no rows.
    """
    stated = {
        (mode, case): {
            "met": met,
            "speed_kmh": speed,
            "met_hours_per_km": met / speed,
            "source": source,
            "note": note,
        }
        for mode, case, met, speed, source, note in _ACTIVITY
    }
    return _every_case(stated, ["street_mode"])


def particle_factors():
    """The shipped car particle factors, in mg per vehicle-km.

    PM10 and PM2.5 per vehicle class, road class and case: tyre, brake
    and road wear, resuspended road dust, and exhaust, after Beddows &
    Harrison (2021), whose mass law scales each part with the car's
    weight (their Tables 3 and 6 print the PM10 and PM2.5 labels
    swapped; these values use the corrected labels). The BEV cases
    differ in the share of braking done by friction brakes, the rest
    being regenerative: 10 % is the paper's assumption, 5 % and 36 % the
    range of Hicks, Green & Beevers (2023), interpolated linearly
    between the paper's 10 % and friction-only totals. The petrol car
    has no published bounds (its middle value stands for all three
    cases, ``published=False``); hybrid, plug-in hybrid and fuel-cell
    cars have no rows. UK data: no studded tyres or Nordic road dust.
    """
    stated = {
        (vehicle_class, road_class, case): {
            "pm10": pm10,
            "pm2_5": pm2_5,
            **dict(zip(("source", "note"), _PARTICLE_SOURCES[(vehicle_class, case)])),
        }
        for vehicle_class, road_class, case, pm10, pm2_5 in _PARTICLES
    }
    return _every_case(stated, ["vehicle_class", "road_class"])


def _every_case(stated, key_columns):
    """One row per key and case from `stated`, ``{(*key, case): values}``.

    A case no source gives keeps the key's middle values with
    ``published=False`` and no source.
    """
    rows = []
    for key in dict.fromkeys(entry[:-1] for entry in stated):
        middle = stated[(*key, "middle")]
        for case in CASES:
            values = stated.get((*key, case))
            if values is None:
                values = {**middle, "source": "", "note": f"no published {case} value"}
            rows.append(
                {
                    **dict(zip(key_columns, key)),
                    "case": case,
                    **values,
                    "published": (*key, case) in stated,
                }
            )
    return pd.DataFrame(rows)


class IndicatorSettings:
    """One query's indicator settings, resolved once up front.

    `indicators` are the requested columns; `co2_price` is the price per
    tonne when ``climate_cost`` is requested. The rates and factors
    warn once per query for a mode or vehicle class whose case no
    source gives, or which no row covers.
    """

    def __init__(self, indicators, co2_price, activity, particles):
        self.indicators = indicators
        self.co2_price = co2_price
        self._activity = activity
        self._particles = particles
        self._warned = set()

    def climate_cost(self, grams):
        """The climate cost of `grams` CO₂e (a number or an array)."""
        return grams / 1e6 * self.co2_price

    def check_supported(self, supported, where):
        """Refuse every requested indicator `where` does not report."""
        missing = [name for name in self.indicators if name not in supported]
        if missing:
            raise ValueError(
                f"indicators {', '.join(missing)} are not available on {where}"
            )

    def trip_columns(self, legs, emissions):
        """The requested activity and climate columns of trips that ride
        `legs`, ``[(street mode, metres per trip), ...]``, and emit
        `emissions` grams CO₂e per person; metres outside the listed
        modes (transit rides) count no activity."""
        columns = {}
        if "physical_activity" in self.indicators:
            columns["physical_activity"] = (
                sum(
                    np.asarray(metres, dtype=float) * self.activity_rate(mode)
                    for mode, metres in legs
                )
                / 1000.0
            )
        if "climate_cost" in self.indicators:
            columns["climate_cost"] = self.climate_cost(
                np.asarray(emissions, dtype=float)
            )
        return columns

    def street_particles(self, mode, metres_by_class, count, vehicle_class, occupancy):
        """The requested ``pm10`` / ``pm2_5`` columns, mg per person, of
        `count` single-mode street legs in `mode`: a car's metres on each
        road class (`metres_by_class`, ``{road class: metres per leg}``) at
        its `vehicle_class`'s factors (``None``: the default car's),
        divided by `occupancy`. Walking and
        the wheelchair release none; another mode, which no source
        covers, resolves NaN with a warning."""
        wanted = [name for name in ("pm10", "pm2_5") if name in self.indicators]
        if not wanted:
            return {}
        if mode in ("walk", "wheelchair"):
            return {name: np.zeros(count) for name in wanted}
        if mode != "car":
            self._warn_once(
                ("particles", mode),
                f"no particle factor covers street mode '{mode}'; pm10 and "
                "pm2_5 stay unresolved",
            )
            return {name: np.full(count, math.nan) for name in wanted}
        if vehicle_class is None:
            # The car's default powertrain, as its emission factors take it.
            from cafein.emissions import STREET_MODE_IDENTITIES

            vehicle_class = STREET_MODE_IDENTITIES["car"][1]
        totals = np.zeros((2, count))
        for road_class, metres in metres_by_class.items():
            driven = np.asarray(metres, dtype=float) > 0
            if not driven.any():
                # A class no leg drives needs no factor: a missing one
                # must not poison legs that never touch it.
                continue
            factors = np.asarray(self.particle_factor(vehicle_class, road_class))
            kilometres = np.where(driven, np.asarray(metres, dtype=float), 0.0) / 1000.0
            totals += np.where(driven, np.outer(factors, kilometres), 0.0)
        totals /= occupancy
        return {name: totals[("pm10", "pm2_5").index(name)] for name in wanted}

    def fingerprint(self):
        """The resolved settings as plain data, for a stream's manifest."""
        activity = particles = None
        if self._activity is not None:
            rates, unpublished, case = self._activity
            activity = [sorted(rates.items()), sorted(unpublished), case]
        if self._particles is not None:
            factors, unpublished, case = self._particles
            particles = [
                sorted([*key, *values] for key, values in factors.items()),
                sorted(map(list, unpublished)),
                case,
            ]
        return {
            "indicators": list(self.indicators),
            "co2_price": self.co2_price,
            "activity": activity,
            "particles": particles,
        }

    def activity_rate(self, mode):
        """MET-hours per km of a leg in street mode `mode`.

        A car leg counts zero (driving is neither walking nor cycling);
        a mode no row covers resolves NaN with a warning.
        """
        if mode == "car":
            return 0.0
        rates, unpublished, case = self._activity
        if mode not in rates:
            self._warn_once(
                ("activity", mode),
                f"no activity rate covers street mode '{mode}'; "
                "physical_activity stays unresolved",
            )
            return math.nan
        if mode in unpublished:
            self._warn_once(
                ("activity", mode),
                f"activity_rates='{case}': no source gives a {case} value for "
                f"'{mode}'; its middle value is used",
            )
        return rates[mode]

    def particle_factor(self, vehicle_class, road_class):
        """``(pm10, pm2_5)`` in mg per vehicle-km for a car of
        `vehicle_class` on `road_class`; NaN with a warning when no row
        covers the vehicle class."""
        factors, unpublished, case = self._particles
        key = (vehicle_class, road_class)
        if key not in factors:
            self._warn_once(
                ("particles", vehicle_class),
                f"no particle factor covers vehicle class '{vehicle_class}' "
                f"on {road_class} roads; pm10 and pm2_5 stay unresolved",
            )
            return math.nan, math.nan
        if key in unpublished:
            self._warn_once(
                ("particles", vehicle_class),
                f"particle_factors='{case}': no source gives a {case} value for "
                f"'{vehicle_class}'; its middle value is used",
            )
        return factors[key]

    def _warn_once(self, key, message):
        """Warn `message` the first time this query meets `key`."""
        if key not in self._warned:
            self._warned.add(key)
            _log.warn(message)


def resolve_query(indicators, co2_price, activity_rates, particle_factors):
    """The validated indicator settings of a query, resolved up front.

    Returns ``None`` when `indicators` is ``None`` (the default — the
    columns are opt-in, and the value arguments are then rejected, never
    ignored), else an :class:`IndicatorSettings`. Each value argument
    takes a case name (default ``"middle"``) or the caller's own
    values, and is rejected when its indicator is not requested.
    """
    values = {
        "co2_price": co2_price,
        "activity_rates": activity_rates,
        "particle_factors": particle_factors,
    }
    if indicators is None:
        given = [f"{name}=" for name, value in values.items() if value is not None]
        if given:
            raise ValueError(
                f"{', '.join(given)} set indicator values; pass indicators= "
                "to request the columns"
            )
        return None
    chosen = [indicators] if isinstance(indicators, str) else list(indicators)
    if (
        not chosen
        or not all(isinstance(name, str) for name in chosen)
        or set(chosen) - set(INDICATORS)
        or len(set(chosen)) != len(chosen)
    ):
        raise ValueError(f"indicators must be a non-empty selection from {INDICATORS}")
    serves = {
        "co2_price": ("climate_cost",),
        "activity_rates": ("physical_activity",),
        "particle_factors": ("pm10", "pm2_5"),
    }
    for name, value in values.items():
        if value is not None and not set(serves[name]) & set(chosen):
            raise ValueError(
                f"{name}= sets {' and '.join(serves[name])}; add it to indicators="
            )
    return IndicatorSettings(
        tuple(name for name in INDICATORS if name in chosen),
        _co2_price(co2_price) if "climate_cost" in chosen else None,
        _activity(activity_rates) if "physical_activity" in chosen else None,
        _particles(particle_factors) if {"pm10", "pm2_5"} & set(chosen) else None,
    )


def _co2_price(value):
    """The price per tonne a ``co2_price=`` value selects."""
    if value is None or isinstance(value, str):
        case = _validate.choice(
            "co2_price", "middle" if value is None else value, CASES
        )
        prices = co2_prices()
        return float(prices.loc[prices.case == case, "price"].iloc[0])
    return _validate.non_negative_finite("co2_price", value)


def _activity(value):
    """``(rates, unpublished modes, case)`` an ``activity_rates=`` value
    selects; own rates replace the middle case's per mode."""
    table = activity_rates()
    if _names_case(value):
        case = _validate.choice(
            "activity_rates", "middle" if value is None else value, CASES
        )
        rows = table[table.case == case]
        unpublished = set(rows.street_mode[~rows.published])
        return dict(zip(rows.street_mode, rows.met_hours_per_km)), unpublished, case
    if isinstance(value, collections.abc.Mapping):
        own = _validate.mode_mapping(
            "activity_rates", value, ACTIVITY_MODES, _validate.non_negative_finite
        )
    else:
        frame = load_activity_rates(value)
        own = dict(zip(frame.street_mode, frame.met_hours_per_km))
    middle = table[table.case == "middle"]
    rates = dict(zip(middle.street_mode, middle.met_hours_per_km))
    rates.update(own)
    return rates, set(), "middle"


def _particles(value):
    """``(factors, unpublished keys, case)`` a ``particle_factors=`` value
    selects; own rows replace the middle case's per
    ``(vehicle_class, road_class)``."""
    table = particle_factors()
    if _names_case(value):
        case = _validate.choice(
            "particle_factors", "middle" if value is None else value, CASES
        )
        table = table[table.case == case]
        unpublished = set(
            zip(
                table.vehicle_class[~table.published],
                table.road_class[~table.published],
            )
        )
    else:
        middle = table[table.case == "middle"]
        table = pd.concat([middle, load_particle_factors(value)], ignore_index=True)
        unpublished, case = set(), "middle"
    keys = zip(table.vehicle_class, table.road_class)
    return dict(zip(keys, zip(table.pm10, table.pm2_5))), unpublished, case


def _names_case(value):
    """Whether an ``activity_rates=`` or ``particle_factors=`` value names
    a case (``None`` the default one) rather than the caller's own values;
    a string with a file suffix is a table path."""
    return value is None or (isinstance(value, str) and not pathlib.Path(value).suffix)


def load_activity_rates(source):
    """Load and validate an activity rate table.

    Columns ``street_mode`` (one of ``walk``, ``bicycle``, ``e_bike``,
    ``e_scooter``, ``wheelchair``) and ``met_hours_per_km`` (non-negative
    and finite). It may be a DataFrame, a path to CSV, JSON or YAML, or a
    sequence of row mappings; ``activity_rates=`` also takes a plain
    ``{street_mode: met_hours_per_km}`` mapping.
    """
    return _load_table(
        source,
        "activity rate",
        ["street_mode"],
        ["met_hours_per_km"],
        {"street_mode": ACTIVITY_MODES},
    )


def load_particle_factors(source):
    """Load and validate a particle factor table.

    Columns ``vehicle_class`` (a non-empty name), ``road_class`` (one of
    ``urban``, ``rural``, ``motorway``), ``pm10`` and ``pm2_5`` (mg per
    vehicle-km, non-negative and finite). It may be a DataFrame, a path
    to CSV, JSON or YAML, or the mapping forms the factor tables take
    (see ``cafein.emissions.load_factors``).
    """
    return _load_table(
        source,
        "particle factor",
        ["vehicle_class", "road_class"],
        ["pm10", "pm2_5"],
        {"road_class": ROAD_CLASSES},
    )


def _load_table(source, what, keys, values, allowed):
    """`source` as a validated table of `keys` and `values` columns: a
    key column in `allowed` takes only its listed names, any other a
    non-empty name; values are non-negative and finite; keys are
    unique."""
    from cafein.emissions import _table_frame

    frame = _table_frame(source, f"{what}s")
    missing = [column for column in keys + values if column not in frame.columns]
    if missing:
        raise ValueError(f"{what} table misses column(s): {', '.join(missing)}")
    frame = frame.reindex(columns=keys + values)
    for column in keys:
        if column in allowed:
            unknown = set(frame[column]) - set(allowed[column])
            if unknown:
                raise ValueError(
                    f"unknown {column} value(s): "
                    + ", ".join(sorted(map(str, unknown)))
                    + f"; expected one of {', '.join(allowed[column])}"
                )
        elif not all(isinstance(name, str) and name for name in frame[column]):
            raise ValueError(f"every {what} row needs a non-empty {column}")
    for column in values:
        frame[column] = [
            _validate.non_negative_finite(column, value) for value in frame[column]
        ]
    if frame.duplicated(subset=keys).any():
        raise ValueError(f"duplicate ({', '.join(keys)}) {what} row(s)")
    return frame.reset_index(drop=True)
