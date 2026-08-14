"""Cumulative-opportunity accessibility over a network's cost surfaces.

``Accessibility`` counts or decay-weights the opportunities reachable
from every origin within one or more budgets on a chosen cost axis —
minutes of travel time, grams CO2e, fare currency units, or street
metres — in long format: one row per (origin, opportunity field,
budget). The weight formulas live in the compiled core; costs come
from the same engine dispatch as the matrix computers.
"""

import datetime

import numpy as np
import pandas as pd

from cafein._validate import id_sequence, sequence_not_string

#: The parameter each decay family takes (None: no parameter).
DECAY_PARAMETERS = {
    "step": None,
    "linear": "width",
    "exponential": "half_life",
    "logistic": "scale",
}


def _decay_parameter(decay, decay_params):
    """The single float the core expects for `decay`, from the user's
    ``{name: value}`` mapping; errors name the knob."""
    if decay not in DECAY_PARAMETERS:
        raise ValueError(
            f"unknown decay {decay!r}: the decay functions are "
            f"{', '.join(sorted(DECAY_PARAMETERS))}"
        )
    name = DECAY_PARAMETERS[decay]
    if name is None:
        if decay_params is not None:
            raise ValueError(f"decay={decay!r} takes no decay_params")
        return None
    if not isinstance(decay_params, dict) or set(decay_params) != {name}:
        raise ValueError(
            f"decay={decay!r} requires decay_params={{{name!r}: <positive number>}}"
        )
    return float(decay_params[name])


def _budget_list(budgets):
    """`budgets` as floats; the core enforces positive and finite."""
    budgets = sequence_not_string("budgets", budgets)
    try:
        return [float(budget) for budget in budgets]
    except (TypeError, ValueError):
        raise ValueError(f"budgets must be numbers, got {budgets!r}") from None


def _opportunity_columns(destinations, opportunities):
    """The opportunity labels and the row-major values matrix.

    A destinations *table* (anything with columns and an ``id``) can
    carry numeric opportunity columns; a bare id sequence can only be
    counted. ``opportunities=None`` counts features under the label
    ``"count"``. Null values are rejected with a count and the column
    named — a silently zeroed cell would understate accessibility.
    """
    if opportunities is None:
        count = len(destinations)
        return ["count"], np.ones((count, 1), dtype="float64")
    if not hasattr(destinations, "columns"):
        raise ValueError(
            "opportunities need destination columns; pass destinations as a "
            "table with an 'id' column"
        )
    if isinstance(opportunities, str):
        opportunities = [opportunities]
    opportunities = list(opportunities)
    if not opportunities:
        raise ValueError("opportunities names no columns")
    missing = [name for name in opportunities if name not in destinations.columns]
    if missing:
        raise ValueError(
            f"destinations carries no column(s) {missing!r}; available: "
            f"{sorted(map(str, destinations.columns))}"
        )
    values = np.empty((len(destinations), len(opportunities)), dtype="float64")
    for at, name in enumerate(opportunities):
        column = pd.to_numeric(destinations[name], errors="raise")
        if pd.api.types.is_complex_dtype(column):
            raise ValueError(
                f"opportunity column {name!r} is complex-valued; opportunity "
                "values must be real, finite, and non-negative"
            )
        nulls = int(column.isna().sum())
        if nulls:
            raise ValueError(
                f"opportunity column {name!r} carries {nulls} null value(s); "
                "fill or drop them explicitly"
            )
        values[:, at] = column.to_numpy(dtype="float64")
    return [str(name) for name in opportunities], values


def _is_table(value):
    return hasattr(value, "columns")


def _is_geo(value):
    return hasattr(value, "geometry")


def _stop_ids(value, role):
    """Stop ids from a bare sequence or a table's ``id`` column."""
    if _is_table(value):
        if "id" not in value.columns:
            raise ValueError(f"the {role} table needs an 'id' column")
        return [str(identifier) for identifier in value["id"]]
    return list(id_sequence(role, value))


def _transit_cost_surface(
    network,
    origins,
    destinations,
    date,
    departure,
    cost,
    window,
    factors,
    components,
    fares,
    max_transfers,
    max_travel_time,
    router,
    chunk,
    exclusions,
    walk,
):
    """The dense per-destination optimum surface for an emissions or
    money axis: NaN marks pairs the engines emitted no row for —
    unreachable within the window, or carrying an unresolved factor or
    unpriceable fare, none of which can satisfy a finite budget."""
    from cafein import emissions
    from cafein.matrices import (
        _chunk_slice,
        _point_list,
        _validate_cost_query,
        _warn_unsnapped,
    )
    from cafein.network import _walk_options

    from cafein.fares import ZoneFareStructure

    objective = "emissions" if cost == "emissions" else "fare"
    fare_tables = fares._flat_tables(network) if fares is not None else None
    # A zone structure's exact fare search needs a time limit to stay
    # fast; 120 minutes of total travel time is the default cap, as on
    # the cost matrices — max_travel_time overrides it.
    cap = max_travel_time
    if cap is None and isinstance(fares, ZoneFareStructure):
        cap = 7200
    _validate_cost_query(date, departure, objective, window, None, fare_tables, router)
    trip_factors = emissions.trip_factors(network, factors, components)
    exclusions = [list(ids) for ids in exclusions]
    walk = _walk_options(*walk)
    if _is_geo(origins):
        from_ids, origin_points = _point_list(origins, "origins")
        to_ids, destination_points = _point_list(destinations, "destinations")
        rows = _chunk_slice(len(from_ids), chunk)
        from_ids = from_ids[rows]
        origin_points = origin_points[rows]
        table = network._core.least_cost_matrix_from_points(
            origin_points,
            destination_points,
            date,
            departure,
            window,
            trip_factors,
            objective,
            fare_tables,
            cap,
            max_transfers,
            router,
            *exclusions,
            *walk,
            False,
        )
        _warn_unsnapped(table, from_ids, to_ids)
        columns = np.asarray(table["to"], dtype="int64")
    else:
        from_ids = _stop_ids(origins, "origins")
        from_ids = from_ids[_chunk_slice(len(from_ids), chunk)]
        destination_ids = _stop_ids(destinations, "destinations")
        unique_ids = list(dict.fromkeys(destination_ids))
        table = network._core.least_cost_matrix(
            from_ids,
            date,
            departure,
            window,
            trip_factors,
            objective,
            fare_tables,
            cap,
            max_transfers,
            unique_ids,
            "time",
            25.0,
            router,
            *exclusions,
            *walk,
            False,
        )
        # The stop path reports destinations as global stop indices;
        # densify over the unique ids, then expand back so repeated
        # destinations keep every column, exactly as the time path does.
        # Ids resolve through the core's canonical resolver, so
        # qualified ids and merged-feed aliases behave as they do on
        # every routing entry.
        globals_of = network._core._stop_indices(unique_ids)
        lookup = np.full(len(network._core.stops), -1, dtype="int64")
        for at, index in enumerate(globals_of):
            lookup[index] = at
        columns = lookup[np.asarray(table["to"], dtype="int64")]
        to_ids = unique_ids
        unique_at = {stop: at for at, stop in enumerate(unique_ids)}
        expansion = [unique_at[stop] for stop in destination_ids]
    surface = np.full((len(from_ids), len(to_ids)), np.nan, dtype="float64")
    kept = columns >= 0
    surface[np.asarray(table["from"], dtype="int64")[kept], columns[kept]] = np.asarray(
        table[objective], dtype="float64"
    )[kept]
    if not _is_geo(origins):
        surface = surface[:, expansion]
    return surface, from_ids


class Accessibility(pd.DataFrame):
    """Reachable opportunities per origin, budget, and field.

    Long format: ``from_id``, ``opportunity`` (the destination column
    or ``"count"``), ``budget`` (in the cost axis's unit), and
    ``accessibility`` (the
    decay-weighted sum; with the default ``decay="step"``, the exact
    reachable count or mass). With a `departure_time_window`, a
    ``percentile`` column is added and each row holds the
    accessibility at that percentile of the travel-time distribution
    across the window's minute marks (r5-style: percentile costs, then
    weighting — never averaged accessibility values).

    On a ``TransportNetwork``, origins and destinations are either both
    stop-id sequences or both GeoDataFrames with an ``id`` column
    (points, or polygons routed via centroids), routed door to door at
    the ``departure``. On a ``StreetNetwork``, both are
    GeoDataFrames and `transport_mode` names the street mode.

    ``cost`` selects the axis budgets are measured on, with
    ``TravelCostMatrix``'s optimize semantics — the per-destination
    optimum of that axis, not the axis read off the fastest journey:

    - ``"time"`` — minutes (the default; windows/percentiles
      apply).
    - ``"emissions"`` — grams CO2e via the cost engines; requires
      `departure_time_window`, takes `factors`/`components`; a
      destination whose
      optimum is unresolved (an unpriced trip) counts as unreached.
    - ``"money"`` — the fare structure's own currency units (the
      ``fare`` column's units); requires `departure_time_window` and
      `fares`.
    - ``"distance"`` — metres, street networks only (network plus
      connector metres); on transit it is not an optimizable axis and
      raises.

    The emissions and money optima are single values over the window,
    so `percentiles` does not combine with them. ``max_travel_time``
    (minutes or a timedelta) bounds their journeys' total travel time;
    on a zone fare structure the money axis defaults to 120 minutes,
    as on the cost matrices, and the time axis rejects it — the
    budgets already bound time.
    """

    @property
    def _constructor(self):
        return pd.DataFrame

    def __init__(
        self,
        network,
        origins,
        destinations,
        departure=None,
        *,
        opportunities=None,
        cost="time",
        budgets=(30.0,),
        decay="step",
        decay_params=None,
        max_rides=8,
        router="auto",
        departure_time_window=None,
        percentiles=None,
        confidence=None,
        factors=None,
        components=None,
        fares=None,
        chunk=None,
        transport_mode=None,
        max_street_time=None,
        max_travel_time=None,
        exclude_routes=(),
        exclude_trips=(),
        exclude_stops=(),
        walking_speed_kmph=None,
        max_walking_time=None,
        snap_distance=None,
    ):
        from cafein._units import departure_parts, duration_seconds

        date, departure = (
            (None, None) if departure is None else departure_parts(departure)
        )
        if max_rides < 1:
            raise ValueError("max_rides must be at least 1")
        max_transfers = max_rides - 1
        window = duration_seconds("departure_time_window", departure_time_window)
        max_walking_time = duration_seconds("max_walking_time", max_walking_time)
        max_street_time = duration_seconds("max_street_time", max_street_time)
        if max_travel_time is not None and cost not in ("emissions", "money"):
            raise ValueError(
                "max_travel_time bounds the emissions and money axes; "
                "time budgets already bound the time axis"
            )
        max_travel_time = duration_seconds("max_travel_time", max_travel_time)
        max_snap_distance = snap_distance
        from cafein import _cafein
        from cafein.matrices import (
            _chunk_slice,
            _is_street_network,
            _point_list,
            _warn_unsnapped,
        )

        origins = sequence_not_string("origins", origins)
        destinations = sequence_not_string("destinations", destinations)
        exclude_routes = id_sequence("exclude_routes", exclude_routes)
        exclude_trips = id_sequence("exclude_trips", exclude_trips)
        exclude_stops = id_sequence("exclude_stops", exclude_stops)
        if cost == "time" and isinstance(decay_params, dict):
            # Time-axis decay parameters are durations like the
            # budgets: minutes (or timedeltas) in, seconds to the core.
            decay_params = {
                name: duration_seconds("decay_params", value)
                for name, value in decay_params.items()
            }
        decay_param = _decay_parameter(decay, decay_params)
        if cost == "time":
            # Time budgets are durations like every other time input:
            # minutes (or timedeltas); other cost axes keep their own
            # units (grams, euros). The frame's budget column echoes the
            # user's minutes; the core compares seconds.
            raw_budgets = sequence_not_string("budgets", budgets)
            budgets = [
                float(duration_seconds("budgets", budget)) for budget in raw_budgets
            ]
            # The frame's budget column echoes the values as passed.
            budget_column = [
                (
                    budget.total_seconds() / 60
                    if isinstance(budget, datetime.timedelta)
                    else float(budget)
                )
                for budget in raw_budgets
            ]
        else:
            budgets = _budget_list(budgets)
            budget_column = budgets
        labels, values = _opportunity_columns(destinations, opportunities)
        if cost not in ("time", "emissions", "money", "distance"):
            raise ValueError(
                f"unknown cost {cost!r}: the axes are time, emissions, "
                "money, distance"
            )
        if cost != "emissions" and (factors is not None or components is not None):
            raise ValueError("factors and components apply to cost='emissions'")
        if cost != "money" and fares is not None:
            raise ValueError("fares applies to cost='money'")
        if cost == "money" and fares is None:
            raise ValueError("cost='money' requires a fare structure (fares=)")
        if _is_street_network(network) and cost in ("emissions", "money"):
            raise ValueError(
                f"cost={cost!r} needs the transit cost engines; a "
                "StreetNetwork serves cost='time' and cost='distance'"
            )
        if cost in ("emissions", "money"):
            if window is None:
                raise ValueError(
                    f"cost={cost!r} optimizes over a departure window; "
                    "pass departure_time_window="
                )
            if percentiles is not None or confidence is not None:
                raise ValueError(
                    f"cost={cost!r} yields the window's single optimum per "
                    "destination; percentiles apply to cost='time'"
                )

        if _is_street_network(network):
            if transport_mode is None:
                raise ValueError(
                    "a StreetNetwork needs transport_mode= (walk, bicycle, "
                    "e_bike, e_scooter, or car on a car-enabled build)"
                )
            rejected = {
                "date": date,
                "departure": departure,
                "departure_time_window": window,
                "percentiles": percentiles,
                "confidence": confidence,
                "walking_speed_kmph": walking_speed_kmph,
                "max_walking_time": max_walking_time,
            }
            named = [name for name, value in rejected.items() if value is not None]
            if router != "auto":
                named.append("router")
            if max_transfers != 7:
                named.append("max_rides")
            if named or any((exclude_routes, exclude_trips, exclude_stops)):
                offending = ", ".join(named) or "exclusions"
                raise ValueError(
                    f"{offending} apply to transit; a StreetNetwork takes "
                    "transport_mode, max_street_time, and snap_distance"
                )
            from cafein.street_network import MAX_STREET_TIME
            from cafein.streets import MAX_SNAP_DISTANCE

            from_ids, origin_points = _point_list(origins, "origins")
            to_ids, destination_points = _point_list(destinations, "destinations")
            rows = _chunk_slice(len(from_ids), chunk)
            from_ids = from_ids[rows]
            origin_points = origin_points[rows]
            street_seconds = float(
                MAX_STREET_TIME if max_street_time is None else max_street_time
            )
            street_snap = float(
                MAX_SNAP_DISTANCE if max_snap_distance is None else max_snap_distance
            )
            if cost == "distance":
                table = network._core.cost_matrix(
                    origin_points,
                    destination_points,
                    transport_mode,
                    street_seconds,
                    street_snap,
                    False,
                    None,
                )
                _warn_unsnapped(table, from_ids, to_ids, network="the street network")
                matrix = np.full((len(from_ids), len(to_ids)), np.nan, dtype="float64")
                matrix[table["from"], table["to"]] = (
                    table["network_distance"] + table["connector_distance"]
                )
            else:
                table = network._core.travel_time_matrix(
                    origin_points,
                    destination_points,
                    transport_mode,
                    street_seconds,
                    street_snap,
                    None,
                )
                _warn_unsnapped(table, from_ids, to_ids, network="the street network")
                matrix = table["matrix"]
            resolved_percentiles = None
        else:
            if transport_mode is not None or max_street_time is not None:
                raise ValueError(
                    "transport_mode and max_street_time apply to a "
                    "StreetNetwork; a TransportNetwork routes door to door"
                )
            if date is None or departure is None:
                raise TypeError("Accessibility requires departure")
            if _is_geo(origins) != _is_geo(destinations):
                raise ValueError(
                    "origins and destinations must both be stop ids/tables "
                    "or both be point GeoDataFrames"
                )
            if cost == "distance":
                raise ValueError(
                    "cost='distance' is not an optimizable transit axis; "
                    "distances ride along the time-optimal journeys in "
                    "TravelCostMatrix, and street networks serve "
                    "cost='distance' natively"
                )
            if cost in ("emissions", "money"):
                matrix, from_ids = _transit_cost_surface(
                    network,
                    origins,
                    destinations,
                    date,
                    departure,
                    cost,
                    window,
                    factors,
                    components,
                    fares,
                    max_transfers,
                    max_travel_time,
                    router,
                    chunk,
                    (exclude_routes, exclude_trips, exclude_stops),
                    (walking_speed_kmph, max_walking_time, max_snap_distance),
                )
                resolved_percentiles = None
            elif _is_geo(origins):
                matrix, from_ids, _to_ids, resolved_percentiles = (
                    network._time_matrix_with_ids(
                        origins,
                        date,
                        departure,
                        max_transfers,
                        destinations=destinations,
                        window=window,
                        percentiles=percentiles,
                        confidence=confidence,
                        chunk=chunk,
                        walking_speed_kmph=walking_speed_kmph,
                        max_walking_time=max_walking_time,
                        max_snap_distance=max_snap_distance,
                        router=router,
                        exclude_routes=exclude_routes,
                        exclude_trips=exclude_trips,
                        exclude_stops=exclude_stops,
                    )
                )
            else:
                destination_ids = _stop_ids(destinations, "destinations")
                matrix, from_ids, to_ids, resolved_percentiles = (
                    network._time_matrix_with_ids(
                        _stop_ids(origins, "origins"),
                        date,
                        departure,
                        max_transfers,
                        destinations=None,
                        window=window,
                        percentiles=percentiles,
                        confidence=confidence,
                        chunk=chunk,
                        walking_speed_kmph=walking_speed_kmph,
                        max_walking_time=max_walking_time,
                        max_snap_distance=max_snap_distance,
                        router=router,
                        exclude_routes=exclude_routes,
                        exclude_trips=exclude_trips,
                        exclude_stops=exclude_stops,
                    )
                )
                # The all-stops matrix is globally indexed, so the
                # canonical resolver's indices ARE the columns.
                selection = list(network._core._stop_indices(destination_ids))
                matrix = matrix[:, selection, ...]

        flat_values = [float(value) for value in values.ravel()]

        def aggregated(cost_slice):
            cost_slice = np.asarray(cost_slice)
            if cost_slice.dtype.kind == "f":
                return _cafein.aggregate_opportunity_sums_f64(
                    np.ascontiguousarray(cost_slice, dtype="float64"),
                    flat_values,
                    len(labels),
                    budgets,
                    decay,
                    decay_param,
                )
            return _cafein.aggregate_opportunity_sums(
                np.ascontiguousarray(cost_slice, dtype="uint32"),
                flat_values,
                len(labels),
                budgets,
                decay,
                decay_param,
            )

        per_origin = len(budgets) * len(labels)
        columns = {
            "from_id": np.repeat(list(from_ids), per_origin),
            "opportunity": np.tile(labels, len(budgets) * len(from_ids)),
            "budget": np.tile(np.repeat(budget_column, len(labels)), len(from_ids)),
        }
        if resolved_percentiles is None:
            columns["accessibility"] = aggregated(matrix).ravel()
            super().__init__(pd.DataFrame(columns))
            return
        # One aggregation per percentile of the windowed cost
        # distribution: the weights apply to percentile costs, never to
        # averaged accessibility values.
        frames = []
        for at, percentile in enumerate(resolved_percentiles):
            frame = pd.DataFrame(columns)
            frame["percentile"] = percentile
            frame["accessibility"] = aggregated(matrix[:, :, at]).ravel()
            frames.append(frame)
        long = pd.concat(frames, ignore_index=True)
        super().__init__(
            long[["from_id", "opportunity", "budget", "percentile", "accessibility"]]
        )
