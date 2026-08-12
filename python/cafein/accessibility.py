"""Cumulative-opportunity accessibility over a network's time costs.

``Accessibility`` counts or decay-weights the opportunities reachable
from every origin within one or more travel-time budgets, in long
format: one row per (origin, opportunity field, budget). The weight
formulas live in the compiled core; costs come from the same engine
dispatch as the travel-time matrices.
"""

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


class Accessibility(pd.DataFrame):
    """Reachable opportunities per origin, budget, and field.

    Long format: ``from_id``, ``opportunity`` (the destination column
    or ``"count"``), ``budget`` (seconds), and ``accessibility`` (the
    decay-weighted sum; with the default ``decay="step"``, the exact
    reachable count or mass). With a departure `window`, a
    ``percentile`` column is added and each row holds the
    accessibility at that percentile of the travel-time distribution
    across the window's minute marks (r5-style: percentile costs, then
    weighting — never averaged accessibility values).

    On a ``TransportNetwork``, origins and destinations are either both
    stop-id sequences or both GeoDataFrames with an ``id`` column
    (points, or polygons routed via centroids), routed door to door at
    `date` and `departure`. On a ``StreetNetwork``, both are
    GeoDataFrames and `transport_mode` names the street mode.
    """

    @property
    def _constructor(self):
        return pd.DataFrame

    def __init__(
        self,
        network,
        origins,
        destinations,
        date=None,
        departure=None,
        *,
        opportunities=None,
        budgets=(1800.0,),
        decay="step",
        decay_params=None,
        max_transfers=7,
        router="auto",
        window=None,
        percentiles=None,
        confidence=None,
        chunk=None,
        transport_mode=None,
        max_street_time=None,
        exclude_routes=(),
        exclude_trips=(),
        exclude_stops=(),
        walking_speed_kmph=None,
        max_walking_time=None,
        max_snap_distance=None,
    ):
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
        decay_param = _decay_parameter(decay, decay_params)
        budgets = _budget_list(budgets)
        labels, values = _opportunity_columns(destinations, opportunities)

        if _is_street_network(network):
            if transport_mode is None:
                raise ValueError(
                    "a StreetNetwork needs transport_mode= (walk, bicycle, "
                    "e_bike, e_scooter, or car on a car-enabled build)"
                )
            rejected = {
                "date": date,
                "departure": departure,
                "window": window,
                "percentiles": percentiles,
                "confidence": confidence,
                "walking_speed_kmph": walking_speed_kmph,
                "max_walking_time": max_walking_time,
            }
            named = [name for name, value in rejected.items() if value is not None]
            if router != "auto":
                named.append("router")
            if max_transfers != 7:
                named.append("max_transfers")
            if named or any((exclude_routes, exclude_trips, exclude_stops)):
                offending = ", ".join(named) or "exclusions"
                raise ValueError(
                    f"{offending} apply to transit; a StreetNetwork takes "
                    "transport_mode, max_street_time, and max_snap_distance"
                )
            from cafein.street_network import MAX_STREET_TIME
            from cafein.streets import MAX_SNAP_DISTANCE

            from_ids, origin_points = _point_list(origins, "origins")
            to_ids, destination_points = _point_list(destinations, "destinations")
            rows = _chunk_slice(len(from_ids), chunk)
            from_ids = from_ids[rows]
            origin_points = origin_points[rows]
            table = network._core.travel_time_matrix(
                origin_points,
                destination_points,
                transport_mode,
                float(MAX_STREET_TIME if max_street_time is None else max_street_time),
                float(
                    MAX_SNAP_DISTANCE
                    if max_snap_distance is None
                    else max_snap_distance
                ),
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
                raise TypeError("Accessibility requires date and departure")
            if _is_geo(origins) != _is_geo(destinations):
                raise ValueError(
                    "origins and destinations must both be stop ids/tables "
                    "or both be point GeoDataFrames"
                )
            if _is_geo(origins):
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
                column = {stop: at for at, stop in enumerate(to_ids)}
                try:
                    selection = [column[stop] for stop in destination_ids]
                except KeyError as error:
                    raise KeyError(
                        f"unknown destination stop {error.args[0]!r}"
                    ) from None
                matrix = matrix[:, selection, ...]

        flat_values = [float(value) for value in values.ravel()]

        def aggregated(cost_slice):
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
            "budget": np.tile(np.repeat(budgets, len(labels)), len(from_ids)),
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
