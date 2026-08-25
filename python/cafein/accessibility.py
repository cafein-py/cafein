"""Cumulative-opportunity accessibility over a network's cost surfaces.

``Accessibility`` counts or decay-weights the opportunities reachable
from every origin within one or more budgets on a chosen cost axis —
minutes of travel time, grams CO2e, fare currency units, or street
metres — in long format: one row per (origin, opportunity field,
budget). ``NearestDestinations`` ranks each origin's closest ``k``
destinations on the same axes, with ``dominance_areas()`` dissolving
polygon origins into the network-Voronoi map. The aggregation
formulas live in the compiled core; costs come from the same engine
dispatch as the matrix computers.
"""

import datetime
import math

import geopandas as gpd
import numpy as np
import pandas as pd

from cafein import _log

from cafein._validate import (
    freeze_ids,
    id_sequence,
    restore_id_dtypes,
    sequence_not_string,
)
from cafein.travelers import folded_constraints, refuse_wheelchair_streets

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


def _product_time_axis(
    departure,
    arrival,
    *,
    cost,
    window,
    router,
    arrival_window=None,
    percentiles=None,
    confidence=None,
):
    """Exactly one time axis for a product, each window naming its own
    axis; the reverse serves arrive-by queries on RAPTOR only, and the
    cost axes require the arrival window (the deadline profile prices
    a window's candidates, never a bare deadline).
    The arrive-by branch owns the full validation — its stop dispatch
    bypasses the forward matrix layer's checks, so nothing may be
    silently accepted or ignored here. Returns (date, clock,
    arrive_by, active window)."""
    from cafein._units import arrival_parts, departure_parts, window_axis

    if departure is not None and arrival is not None:
        raise ValueError("give exactly one of departure= or arrival=")
    arrive_by = arrival is not None
    active_window = window_axis(arrive_by, window, arrival_window)
    if not arrive_by:
        date, moment = (None, None) if departure is None else departure_parts(departure)
        return date, moment, False, active_window
    if cost in ("emissions", "money") and active_window is None:
        raise ValueError(
            f"cost={cost!r} with arrival= prices the deadline profile "
            "over a window; pass arrival_time_window="
        )
    if cost not in ("time", "emissions", "money"):
        raise ValueError(f"cost={cost!r} does not combine with arrival=")
    if active_window is None and (percentiles is not None or confidence is not None):
        raise ValueError(
            "percentiles= and confidence= rank an arrival window's "
            "duration distribution; pass arrival_time_window="
        )
    if router not in ("auto", "raptor", "tbtr"):
        raise ValueError(f"router must be 'auto', 'raptor', or 'tbtr', not {router!r}")
    if router == "tbtr":
        raise ValueError(
            "router='tbtr' does not serve arrival=; the reverse search " "rides RAPTOR"
        )
    date, moment = arrival_parts(arrival)
    return date, moment, True, active_window


def _nearest_of_set(
    network,
    origins,
    destinations,
    date,
    departure,
    max_transfers,
    horizon,
    chunk,
    exclusions,
    walk_options,
    travel_time_output,
    output_time_units,
):
    """The k=1 arrive-by frame from one tagged reverse run."""
    import numpy as np
    import pandas as pd

    from cafein.matrices import _chunk_slice, _point_list, _warn_unsnapped
    from cafein.network import _walk_options

    exclude_routes, exclude_trips, exclude_stops = exclusions
    if _is_geo(origins) != _is_geo(destinations):
        raise ValueError(
            "origins and destinations must both be stop ids/tables "
            "or both be point GeoDataFrames"
        )
    rows = []
    if _is_geo(origins):
        from_ids, origin_points = _point_list(origins, "origins")
        keep = _chunk_slice(len(from_ids), chunk)
        from_ids = list(from_ids[keep])
        origin_points = list(origin_points[keep])
        to_ids, destination_points = _point_list(destinations, "destinations")
        walking_speed_kmph, max_walking_time, snap_distance = walk_options
        table = network._core._arrive_by_nearest_points(
            origin_points,
            list(destination_points),
            date,
            departure,
            max_transfers,
            list(exclude_routes),
            list(exclude_trips),
            list(exclude_stops),
            *_walk_options(walking_speed_kmph, max_walking_time, snap_distance),
        )
        _warn_unsnapped(table, from_ids, to_ids)
        unreachable = np.iinfo(np.uint32).max
        for identifier, cost_value, seed in zip(
            from_ids, table["costs"], table["seeds"]
        ):
            if cost_value == unreachable:
                continue
            if horizon is not None and cost_value > horizon:
                continue
            rows.append((identifier, to_ids[seed], cost_value))
    else:
        from_ids = _stop_ids(origins, "origins")
        keep = _chunk_slice(len(from_ids), chunk)
        from_ids = list(from_ids[keep])
        to_ids = _stop_ids(destinations, "destinations")
        winners = network._core._arrive_by_nearest(
            from_ids,
            to_ids,
            date,
            departure,
            max_transfers,
            list(exclude_routes),
            list(exclude_trips),
            list(exclude_stops),
        )
        for identifier, winner in zip(from_ids, winners):
            if winner is None:
                continue
            cost_value, seed = winner
            if horizon is not None and cost_value > horizon:
                continue
            rows.append((identifier, to_ids[seed], cost_value))
    costs = travel_time_output(
        # The fan-out's ranking emits float costs; the fast path
        # matches its dtype exactly.
        np.asarray([row[2] for row in rows], dtype="float64"),
        output_time_units,
    )
    return pd.DataFrame(
        {
            "from_id": np.asarray([row[0] for row in rows], dtype=object),
            "rank": np.ones(len(rows), dtype="int64"),
            "destination_id": np.asarray([row[1] for row in rows], dtype=object),
            "cost": costs,
        }
    )


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
    _resolved=None,
    arrive_by=False,
):
    """The dense per-destination optimum surface for an emissions or
    money axis: NaN marks pairs the engines emitted no row for —
    unreachable within the window, or carrying an unresolved factor or
    unpriceable fare, none of which can satisfy a finite budget. With
    ``arrive_by`` the window is an arrival window and each cell prices
    the pair's deadline-profile candidates."""
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
    if _resolved is not None:
        # A streaming run froze the factor and fare resolution once;
        # every batch prices with the same snapshot.
        resolved_factors, fare_tables = _resolved
    else:
        resolved_factors = None
        fare_tables = fares._flat_tables(network) if fares is not None else None
    # A zone structure's exact fare search needs a time limit to stay
    # fast; 120 minutes of total travel time is the default cap, as on
    # the cost matrices — max_travel_time overrides it.
    cap = max_travel_time
    if not arrive_by and cap is None and isinstance(fares, ZoneFareStructure):
        cap = 7200
    _validate_cost_query(
        date,
        departure,
        objective,
        window,
        None,
        fare_tables,
        router,
        arrive_by=arrive_by,
    )
    trip_factors = (
        resolved_factors
        if resolved_factors is not None
        else emissions.trip_factors(network, factors, components)
    )
    exclusions = [list(ids) for ids in exclusions]
    walk = _walk_options(*walk)
    if _is_geo(origins):
        from_ids, origin_points = _point_list(origins, "origins")
        to_ids, destination_points = _point_list(destinations, "destinations")
        rows = _chunk_slice(len(from_ids), chunk)
        from_ids = from_ids[rows]
        origin_points = origin_points[rows]
        if arrive_by:
            table = network._core._arrive_by_least_cost_matrix_from_points(
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
                *exclusions,
                *walk,
                False,
            )
        else:
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
        # Dedupe on the RESOLVED stop, not the id string: a qualified
        # id and its unqualified alias share a global stop, and both
        # columns must carry that stop's costs.
        resolved = list(network._core._stop_indices(destination_ids))
        position_of = {}
        unique_ids = []
        for stop, index in zip(destination_ids, resolved):
            if index not in position_of:
                position_of[index] = len(unique_ids)
                unique_ids.append(stop)
        if arrive_by:
            table = network._core._arrive_by_least_cost_matrix(
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
                *exclusions,
                False,
            )
        else:
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
        # densify over the deduped columns, then expand back so
        # repeated destinations and aliases keep every column, exactly
        # as the time path does.
        lookup = np.full(len(network._core.stops), -1, dtype="int64")
        for index, at in position_of.items():
            lookup[index] = at
        columns = lookup[np.asarray(table["to"], dtype="int64")]
        to_ids = unique_ids
        expansion = [position_of[index] for index in resolved]
    surface = np.full((len(from_ids), len(to_ids)), np.nan, dtype="float64")
    kept = columns >= 0
    surface[np.asarray(table["from"], dtype="int64")[kept], columns[kept]] = np.asarray(
        table[objective], dtype="float64"
    )[kept]
    if not _is_geo(origins):
        surface = surface[:, expansion]
        return surface, from_ids, destination_ids
    return surface, from_ids, to_ids


def _validated_car_park_policy(
    street_policy,
    network,
    origins,
    destinations,
    cost,
    arrive_by,
    window,
    percentiles,
    confidence,
    router,
    exclude_stops,
    surface,
):
    """The eager CarParkPolicy gate for the accessibility pillar: the
    car-park policy only, on the time cost of either axis with
    point-set origins and destinations."""
    if street_policy is None:
        return None
    from cafein.matrices import _is_point_frame, _is_street_network
    from cafein.policy import CarParkPolicy

    if not isinstance(street_policy, CarParkPolicy):
        raise ValueError(
            f"{surface} serves street_policy=CarParkPolicy(...); the "
            "street-leg policies stay on the matrix and itinerary surfaces"
        )
    if _is_street_network(network):
        raise ValueError(
            "CarParkPolicy shapes transit access; a StreetNetwork query "
            "has no transit side"
        )
    if cost != "time":
        raise ValueError(
            f"CarParkPolicy accessibility serves cost='time', not {cost!r}"
        )
    named = [
        name
        for name, value in (
            (
                "arrival_time_window" if arrive_by else "departure_time_window",
                window,
            ),
            ("percentiles", percentiles),
            ("confidence", confidence),
        )
        if value is not None
    ]
    if named or router != "auto":
        offending = ", ".join(named) or f"router={router!r}"
        raise ValueError(
            f"CarParkPolicy does not combine with {offending}; the "
            "policy runs the earliest-arrival engine"
        )
    if exclude_stops:
        raise ValueError(
            "CarParkPolicy does not combine with stop exclusions in this stage"
        )
    # The construction contract outranks the input shape, as on the
    # matrix arms: a missing car side is named first.
    network._require_car_side()
    if not _is_point_frame(origins) or not _is_point_frame(destinations):
        raise ValueError(
            "CarParkPolicy accessibility takes point-set origins and "
            "destinations; the facilities plane needs coordinates"
        )
    return street_policy


def _resolved_cost_matrix(
    network,
    origins,
    destinations,
    date,
    departure,
    cost,
    window,
    percentiles,
    confidence,
    factors,
    components,
    fares,
    max_transfers,
    max_travel_time,
    router,
    chunk,
    transport_mode,
    max_street_time,
    max_snap_distance,
    exclude_routes,
    exclude_trips,
    exclude_stops,
    walking_speed_kmph,
    max_walking_time,
    label,
    street_policy=None,
    _resolved_costs=None,
    arrive_by=False,
):
    """The per-origin cost matrix on the chosen axis, dispatched by
    network kind exactly as the matrix computers dispatch: (matrix,
    from_ids, to_ids, resolved_percentiles). Shared by the
    accessibility-pillar computers so the engine mapping exists once;
    ``to_ids`` aligns with the matrix columns (``None`` when only the
    destination input order aligns them)."""
    from cafein.matrices import (
        _chunk_slice,
        _is_street_network,
        _point_list,
        _warn_unsnapped,
    )

    to_ids = None
    if _is_street_network(network):
        if arrive_by:
            raise ValueError(
                "arrival applies to transit; a StreetNetwork takes "
                "transport_mode, max_street_time, and snap_distance"
            )
        if transport_mode is None:
            raise ValueError(
                "a StreetNetwork needs transport_mode= (walk, bicycle, "
                "e_bike, e_scooter, wheelchair, or car on a car-enabled build)"
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
            raise TypeError(f"{label} requires departure or arrival")
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
            matrix, from_ids, to_ids = _transit_cost_surface(
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
                _resolved=_resolved_costs,
                arrive_by=arrive_by,
            )
            resolved_percentiles = None
        elif _is_geo(origins):
            if street_policy is not None:
                # The composed car-and-walking access table seeds the
                # same engine fan-out the policy matrices ride; the
                # gate validated the axis and the point frames.
                from cafein.matrices import (
                    _car_park_arrive_by_time_matrix,
                    _car_park_time_matrix,
                )

                walk_knobs = (walking_speed_kmph, max_walking_time, max_snap_distance)
                if arrive_by:
                    # A product's chunk stays on the origins — its
                    # scores need every destination — so the matrix
                    # layer's destination chunking is bypassed.
                    origins = origins.iloc[_chunk_slice(len(origins), chunk)]
                    matrix, _departures, _stops, from_ids, to_ids = (
                        _car_park_arrive_by_time_matrix(
                            network,
                            origins,
                            destinations,
                            date,
                            departure,
                            street_policy,
                            max_transfers,
                            None,
                            walk_knobs,
                            exclude_routes,
                            exclude_trips,
                        )
                    )
                    return matrix, from_ids, to_ids, None
                matrix, from_ids, to_ids = _car_park_time_matrix(
                    network,
                    origins,
                    destinations,
                    date,
                    departure,
                    street_policy,
                    max_transfers,
                    chunk,
                    walk_knobs,
                    exclude_routes,
                    exclude_trips,
                )
                return matrix, from_ids, to_ids, None
            if arrive_by:
                # A product's chunk stays on the origins — its scores
                # need every destination — so the matrix layer's
                # destination chunking is bypassed.
                origins = origins.iloc[_chunk_slice(len(origins), chunk)]
            matrix, from_ids, to_ids, resolved_percentiles = (
                network._time_matrix_with_ids(
                    origins,
                    date,
                    departure,
                    max_transfers,
                    destinations=destinations,
                    window=window,
                    percentiles=percentiles,
                    confidence=confidence,
                    chunk=None if arrive_by else chunk,
                    walking_speed_kmph=walking_speed_kmph,
                    max_walking_time=max_walking_time,
                    max_snap_distance=max_snap_distance,
                    router=router,
                    exclude_routes=exclude_routes,
                    exclude_trips=exclude_trips,
                    exclude_stops=exclude_stops,
                    arrive_by=arrive_by,
                )
            )
        elif arrive_by:
            # One reverse run per named destination — never the
            # all-stops fan-out — with the chunk on the origin rows.
            from cafein.network import _window_percentiles

            destination_ids = _stop_ids(destinations, "destinations")
            origin_ids = _stop_ids(origins, "origins")
            origin_ids = list(origin_ids[_chunk_slice(len(origin_ids), chunk)])
            resolved_percentiles = _window_percentiles(window, percentiles, confidence)
            if resolved_percentiles is None:
                matrix = network._core._arrive_by_time_matrix(
                    origin_ids,
                    destination_ids,
                    date,
                    departure,
                    max_transfers,
                    list(exclude_routes),
                    list(exclude_trips),
                    list(exclude_stops),
                )
            else:
                matrix = network._core._arrive_by_time_percentiles(
                    origin_ids,
                    destination_ids,
                    date,
                    departure,
                    window,
                    resolved_percentiles,
                    max_transfers,
                    list(exclude_routes),
                    list(exclude_trips),
                    list(exclude_stops),
                )
            from_ids = origin_ids
            to_ids = destination_ids
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
            to_ids = destination_ids
    return matrix, from_ids, to_ids, resolved_percentiles


def _accessibility_columns(
    network,
    origins,
    destinations,
    date,
    departure,
    cost,
    window,
    percentiles,
    confidence,
    factors,
    components,
    fares,
    max_transfers,
    max_travel_time,
    router,
    chunk,
    transport_mode,
    max_street_time,
    max_snap_distance,
    exclude_routes,
    exclude_trips,
    exclude_stops,
    walking_speed_kmph,
    max_walking_time,
    labels,
    values,
    budgets,
    budget_column,
    decay,
    decay_param,
    label,
    street_policy=None,
    _resolved_costs=None,
    arrive_by=False,
    _surface=None,
):
    """The long accessibility frame for `origins` — the computation
    the constructor and the streaming classmethod share, inputs
    already validated and converted."""
    from cafein import _cafein

    if _surface is not None:
        # A precomputed cost surface (the arrive-by streamer computes
        # its destination fan-out once and slices origin batches from
        # it) skips the per-batch engine dispatch entirely.
        matrix, from_ids, _to_ids, resolved_percentiles = _surface
    else:
        matrix, from_ids, _to_ids, resolved_percentiles = _resolved_cost_matrix(
            network,
            origins,
            destinations,
            date,
            departure,
            cost,
            window,
            percentiles,
            confidence,
            factors,
            components,
            fares,
            max_transfers,
            max_travel_time,
            router,
            chunk,
            transport_mode,
            max_street_time,
            max_snap_distance,
            exclude_routes,
            exclude_trips,
            exclude_stops,
            walking_speed_kmph,
            max_walking_time,
            label=label,
            street_policy=street_policy,
            _resolved_costs=_resolved_costs,
            arrive_by=arrive_by,
        )

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
        return pd.DataFrame(columns)
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
    return long[["from_id", "opportunity", "budget", "percentile", "accessibility"]]


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
    the ``departure`` — or, with ``arrival=`` (exactly one of the
    two), by the arrival deadline: every cost is then the
    latest-departure journey's own duration into the destination, one
    reverse run per destination, with ``chunk`` still slicing the
    origins (a score needs every destination). ``arrival`` rides
    RAPTOR and applies to transit only; ``cost="time"`` takes it bare
    or windowed, while the emissions and money axes require
    ``arrival_time_window=`` beside it and price each destination's
    deadline profile exactly as ``TravelCostMatrix(arrival=)`` does.
    On a ``StreetNetwork``, both are
    GeoDataFrames and `transport_mode` names the street mode.

    ``cost`` selects the axis budgets are measured on, with
    ``TravelCostMatrix``'s optimize semantics — the per-destination
    optimum of that axis, not the axis read off the fastest journey:

    - ``"time"`` — minutes (the default; windows/percentiles
      apply).
    - ``"emissions"`` — grams CO2e via the cost engines; requires its
      axis's window (`departure_time_window`, or `arrival_time_window`
      beside ``arrival=``), takes `factors`/`components`; a
      destination whose
      optimum is unresolved (an unpriced trip) counts as unreached.
    - ``"money"`` — the fare structure's own currency units (the
      ``fare`` column's units); requires its axis's window and
      `fares`.
    - ``"distance"`` — metres, street networks only (network plus
      connector metres); on transit it is not an optimizable axis and
      raises.

    ``Accessibility.to_parquet(...)`` streams the same table to disk
    in origin batches with the matrices' resume manifest, so a
    country-scale run never materialises the whole frame; its streamed
    ``from_id`` stays a dictionary-encoded string whatever the input
    dtype (shard schema stability), even where this constructor would
    restore an integer dtype.

    The emissions and money optima are single values over the window,
    so `percentiles` does not combine with them. ``max_travel_time``
    (minutes or a timedelta) bounds their journeys' total travel time;
    on a zone fare structure the money axis defaults to 120 minutes
    on the departure axis, as on the cost matrices (the arrival axis
    prices pre-enumerated candidates and takes no implicit cap), and
    the time axis rejects it — the budgets already bound time.

    ``street_policy=`` takes a ``cafein.CarParkPolicy`` (a network
    built with ``"car"`` in ``street_modes=``): each origin's access
    composes drive-to-facility, parking search, and the facility walk,
    with the ordinary walking access competing beside the car plane
    and winning ties; egress is ordinary walking and transfers ride
    the installed transfer set, as on every query. The query's
    walking knobs stay active. The policy serves ``cost="time"`` on
    either time axis with point-set origins and destinations — with
    ``arrival=`` each cost is the latest-departure journey's own
    duration through the composed table; windows, percentiles,
    ``router=`` overrides, and stop exclusions are rejected by name,
    and the street-leg policies stay on the matrix and itinerary
    surfaces.
    """

    @property
    def _constructor(self):
        return pd.DataFrame

    @_log.timed_computer(
        "matrix.accessibility",
        _log.matrix,
        "computing accessibility",
        "computed accessibility",
    )
    def __init__(
        self,
        network,
        origins,
        destinations,
        departure=None,
        *,
        arrival=None,
        arrival_time_window=None,
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
        traveler=None,
        exclude_stops=(),
        walking_speed_kmph=None,
        max_walking_time=None,
        snap_distance=None,
        street_policy=None,
    ):
        if hasattr(network, "route_between_stops"):
            (
                exclude_routes,
                exclude_trips,
                exclude_stops,
                walking_speed_kmph,
                max_walking_time,
            ) = folded_constraints(
                traveler,
                network,
                exclude_routes,
                exclude_trips,
                exclude_stops,
                walking_speed_kmph,
                max_walking_time,
            )
        elif traveler is not None:
            raise ValueError(
                "traveler applies to transit; a StreetNetwork takes "
                "transport_mode, max_street_time, and snap_distance"
            )
        from cafein.matrices import _is_point_frame as _points

        if hasattr(network, "route_between_stops") and _points(origins):
            refuse_wheelchair_streets(traveler, "Accessibility")
        from cafein._units import duration_seconds

        date, departure, arrive_by, raw_window = _product_time_axis(
            departure,
            arrival,
            cost=cost,
            window=departure_time_window,
            router=router,
            arrival_window=arrival_time_window,
            percentiles=percentiles,
            confidence=confidence,
        )
        if max_rides < 1:
            raise ValueError("max_rides must be at least 1")
        max_transfers = max_rides - 1
        window = duration_seconds(
            "arrival_time_window" if arrive_by else "departure_time_window",
            raw_window,
        )
        max_walking_time = duration_seconds("max_walking_time", max_walking_time)
        max_street_time = duration_seconds("max_street_time", max_street_time)
        if max_travel_time is not None and cost not in ("emissions", "money"):
            raise ValueError(
                "max_travel_time bounds the emissions and money axes; "
                "time budgets already bound the time axis"
            )
        max_travel_time = duration_seconds("max_travel_time", max_travel_time)
        max_snap_distance = snap_distance
        from cafein.matrices import _is_street_network

        origins, _origin_dtype = freeze_ids(origins)
        _id_dtypes = {"from_id": _origin_dtype}
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
        street_policy = _validated_car_park_policy(
            street_policy,
            network,
            origins,
            destinations,
            cost,
            arrive_by,
            window,
            percentiles,
            confidence,
            router,
            exclude_stops,
            "Accessibility",
        )

        frame = _accessibility_columns(
            network,
            origins,
            destinations,
            date,
            departure,
            cost,
            window,
            percentiles,
            confidence,
            factors,
            components,
            fares,
            max_transfers,
            max_travel_time,
            router,
            chunk,
            transport_mode,
            max_street_time,
            max_snap_distance,
            exclude_routes,
            exclude_trips,
            exclude_stops,
            walking_speed_kmph,
            max_walking_time,
            labels,
            values,
            budgets,
            budget_column,
            decay,
            decay_param,
            label="Accessibility",
            street_policy=street_policy,
            arrive_by=arrive_by,
        )
        super().__init__(restore_id_dtypes(frame, _id_dtypes))

    @classmethod
    def to_parquet(
        cls,
        network,
        origins,
        destinations,
        departure=None,
        *,
        arrival=None,
        arrival_time_window=None,
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
        traveler=None,
        exclude_stops=(),
        walking_speed_kmph=None,
        max_walking_time=None,
        snap_distance=None,
        street_policy=None,
        output,
        batch_size=None,
        resume=False,
    ):
        """The accessibility table streamed to Parquet — the
        constructor's semantics with ``travel_cost_table``'s
        ``output=`` behavior.

        Origins are processed in ``batch_size`` slices (default 500)
        and each batch is written as it completes, so a country-scale
        run never materialises the whole frame. ``output=`` selects
        the form by suffix exactly as ``travel_cost_table`` does and
        the return value is a :class:`cafein.StreamingResult`;
        ``resume=True`` continues a matching partial directory run
        with the same manifest contract. With ``arrival=`` the
        destination fan-out runs once up front and origin batches
        stream from the resulting (origins × destinations) duration
        surface — skinny by construction, so its peak memory is that
        surface plus one batch — and the manifest fingerprints the
        complete time query, refusing a resume that differs in axis,
        moment, window, or percentiles.
        """
        if street_policy is not None:
            raise NotImplementedError(
                "street_policy accessibility does not stream yet; "
                "compute Accessibility with the constructor instead"
            )
        if hasattr(network, "route_between_stops"):
            (
                exclude_routes,
                exclude_trips,
                exclude_stops,
                walking_speed_kmph,
                max_walking_time,
            ) = folded_constraints(
                traveler,
                network,
                exclude_routes,
                exclude_trips,
                exclude_stops,
                walking_speed_kmph,
                max_walking_time,
            )
        elif traveler is not None:
            raise ValueError(
                "traveler applies to transit; a StreetNetwork takes "
                "transport_mode, max_street_time, and snap_distance"
            )
        from cafein.matrices import _is_point_frame as _points

        if hasattr(network, "route_between_stops") and _points(origins):
            refuse_wheelchair_streets(traveler, "Accessibility.to_parquet")
        from cafein._units import duration_seconds

        date, departure, arrive_by, raw_window = _product_time_axis(
            departure,
            arrival,
            cost=cost,
            window=departure_time_window,
            router=router,
            arrival_window=arrival_time_window,
            percentiles=percentiles,
            confidence=confidence,
        )
        if max_rides < 1:
            raise ValueError("max_rides must be at least 1")
        max_transfers = max_rides - 1
        window = duration_seconds(
            "arrival_time_window" if arrive_by else "departure_time_window",
            raw_window,
        )
        max_walking_time = duration_seconds("max_walking_time", max_walking_time)
        max_street_time = duration_seconds("max_street_time", max_street_time)
        if max_travel_time is not None and cost not in ("emissions", "money"):
            raise ValueError(
                "max_travel_time bounds the emissions and money axes; "
                "time budgets already bound the time axis"
            )
        max_travel_time = duration_seconds("max_travel_time", max_travel_time)
        max_snap_distance = snap_distance
        from cafein.matrices import _is_street_network

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

        import pyarrow

        from cafein.matrices import (
            _chunk_slice,
            _point_frame,
            _point_list,
            _stream_run,
            _stream_size,
        )

        size = _stream_size(batch_size, resume)
        geo = _is_geo(origins)
        if geo:
            from_ids, origin_points = _point_list(origins, "origins")
        else:
            from_ids = _stop_ids(origins, "origins")
            origin_points = None
        keep = _chunk_slice(len(from_ids), chunk)
        from_ids = list(from_ids[keep])
        if origin_points is not None:
            origin_points = list(origin_points[keep])
        if _is_geo(destinations):
            to_ids, to_points = _point_list(destinations, "destinations")
            to_ids = list(to_ids)
            to_points = list(to_points)
        elif _is_table(destinations):
            to_ids = _stop_ids(destinations, "destinations")
            to_points = None
        else:
            to_ids = list(id_sequence("destinations", destinations))
            to_points = None
        flat_values = [float(value) for value in values.ravel()]
        resolved_costs = None
        if cost in ("emissions", "money"):
            # One resolution serves every batch and the fingerprint:
            # factor rows, component selection, and fare tables are
            # frozen here, never re-read from the caller's mutables.
            from cafein import emissions as emissions_module

            resolved_costs = (
                emissions_module.trip_factors(network, factors, components),
                None if fares is None else fares._flat_tables(network),
            )
        resolved_percentiles = None
        if cost == "time" and window is not None:
            from cafein.network import _window_percentiles

            resolved_percentiles = _window_percentiles(window, percentiles, confidence)
        columns = ["from_id", "opportunity", "budget", "accessibility"]
        if resolved_percentiles is not None:
            columns = [
                "from_id",
                "opportunity",
                "budget",
                "percentile",
                "accessibility",
            ]
        parameters = {
            "date": date,
            "departure": departure,
            "time_axis": "arrival" if arrive_by else "departure",
            "cost": cost,
            "window": window,
            "budgets": budgets,
            "budget_column": budget_column,
            "decay": decay,
            "decay_param": decay_param,
            "labels": labels,
            "opportunities": flat_values,
            "percentiles": resolved_percentiles,
            "max_transfers": max_transfers,
            "max_travel_time": max_travel_time,
            "router": router,
            "transport_mode": transport_mode,
            "max_street_time": max_street_time,
            "exclude_routes": list(exclude_routes),
            "exclude_trips": list(exclude_trips),
            "exclude_stops": list(exclude_stops),
            "walking_speed_kmph": walking_speed_kmph,
            "max_walking_time": max_walking_time,
            "max_snap_distance": max_snap_distance,
            # Sorted for the hash only, exactly as the matrix
            # streamers record their resolved factor set.
            "factors": (None if resolved_costs is None else sorted(resolved_costs[0])),
            "fares": None if resolved_costs is None else resolved_costs[1],
        }
        per_origin = len(budgets) * len(labels)
        surface = None
        if arrive_by:
            # The destination fan-out runs ONCE; origin batches then
            # stream from the skinny (origins × destinations) surface
            # instead of re-running every destination per batch.
            surface_origins = (
                list(from_ids)
                if origin_points is None
                else _point_frame(from_ids, origin_points)
            )
            surface_destinations = (
                to_ids if to_points is None else _point_frame(to_ids, to_points)
            )
            surface = _resolved_cost_matrix(
                network,
                surface_origins,
                surface_destinations,
                date,
                departure,
                cost,
                window,
                resolved_percentiles,
                None,
                factors,
                components,
                fares,
                max_transfers,
                max_travel_time,
                router,
                None,
                transport_mode,
                max_street_time,
                max_snap_distance,
                exclude_routes,
                exclude_trips,
                exclude_stops,
                walking_speed_kmph,
                max_walking_time,
                label="Accessibility.to_parquet",
                _resolved_costs=resolved_costs,
                arrive_by=True,
            )

        def make_batch(rows, shared_from, shared_to):
            if origin_points is None:
                batch_origins = from_ids[rows]
            else:
                batch_origins = _point_frame(from_ids[rows], origin_points[rows])
            if to_points is None:
                batch_destinations = to_ids
            else:
                batch_destinations = _point_frame(to_ids, to_points)
            batch_surface = None
            if surface is not None:
                matrix, surface_from, surface_to, surface_pct = surface
                batch_surface = (
                    matrix[rows],
                    list(surface_from[rows]),
                    surface_to,
                    surface_pct,
                )
            frame = _accessibility_columns(
                network,
                batch_origins,
                batch_destinations,
                date,
                departure,
                cost,
                window,
                resolved_percentiles,
                None,
                factors,
                components,
                fares,
                max_transfers,
                max_travel_time,
                router,
                None,
                transport_mode,
                max_street_time,
                max_snap_distance,
                exclude_routes,
                exclude_trips,
                exclude_stops,
                walking_speed_kmph,
                max_walking_time,
                labels,
                values,
                budgets,
                budget_column,
                decay,
                decay_param,
                label="Accessibility.to_parquet",
                _resolved_costs=resolved_costs,
                arrive_by=arrive_by,
                _surface=batch_surface,
            )
            block = np.repeat(np.arange(rows.start, rows.stop), per_origin)
            planes = 1 if resolved_percentiles is None else len(resolved_percentiles)
            indices = np.tile(block, planes)
            if len(indices) != len(frame):
                raise ValueError(
                    "a batch resolved a different row structure than the "
                    "frozen query; the stream never re-resolves"
                )
            data = {
                "from_id": pyarrow.DictionaryArray.from_arrays(
                    pyarrow.array(indices), shared_from
                ),
                "opportunity": pyarrow.array(
                    frame["opportunity"], type=pyarrow.string()
                ),
                "budget": pyarrow.array(np.asarray(frame["budget"], dtype="float64")),
            }
            if resolved_percentiles is not None:
                data["percentile"] = pyarrow.array(
                    np.asarray(frame["percentile"], dtype="float64")
                )
            data["accessibility"] = pyarrow.array(
                np.asarray(frame["accessibility"], dtype="float64")
            )
            return pyarrow.table(data)

        return _stream_run(
            "Accessibility.to_parquet",
            network,
            columns,
            parameters,
            from_ids,
            to_ids,
            (origin_points, to_points),
            output,
            size,
            make_batch,
            pyarrow,
            resume=resume,
            dictionary_columns=("from_id",),
        )

    def gini_index(
        self,
        *,
        value="accessibility",
        sociodemographic_data=None,
        population=None,
        group_columns=None,
        dropna=False,
    ):
        """``cafein.equity.gini_index`` over this frame."""
        from cafein import equity

        return equity.gini_index(
            self,
            value=value,
            sociodemographic_data=sociodemographic_data,
            population=population,
            group_columns=group_columns,
            dropna=dropna,
        )

    def lorenz_curve(
        self,
        *,
        value="accessibility",
        sociodemographic_data=None,
        population=None,
        group_columns=None,
        dropna=False,
    ):
        """``cafein.equity.lorenz_curve`` over this frame."""
        from cafein import equity

        return equity.lorenz_curve(
            self,
            value=value,
            sociodemographic_data=sociodemographic_data,
            population=population,
            group_columns=group_columns,
            dropna=dropna,
        )

    def share_ratio(
        self,
        *,
        top=0.10,
        bottom=0.40,
        income=None,
        value="accessibility",
        sociodemographic_data=None,
        population=None,
        group_columns=None,
        dropna=False,
    ):
        """``cafein.equity.share_ratio`` over this frame."""
        from cafein import equity

        return equity.share_ratio(
            self,
            top=top,
            bottom=bottom,
            income=income,
            value=value,
            sociodemographic_data=sociodemographic_data,
            population=population,
            group_columns=group_columns,
            dropna=dropna,
        )

    def palma_ratio(
        self,
        *,
        income,
        value="accessibility",
        sociodemographic_data=None,
        population=None,
        group_columns=None,
        dropna=False,
    ):
        """``cafein.equity.palma_ratio`` over this frame."""
        from cafein import equity

        return equity.palma_ratio(
            self,
            income=income,
            value=value,
            sociodemographic_data=sociodemographic_data,
            population=population,
            group_columns=group_columns,
            dropna=dropna,
        )

    def generalized_entropy(
        self,
        *,
        alpha=1,
        groups=None,
        value="accessibility",
        sociodemographic_data=None,
        population=None,
        group_columns=None,
        dropna=False,
    ):
        """``cafein.equity.generalized_entropy`` over this frame."""
        from cafein import equity

        return equity.generalized_entropy(
            self,
            alpha=alpha,
            groups=groups,
            value=value,
            sociodemographic_data=sociodemographic_data,
            population=population,
            group_columns=group_columns,
            dropna=dropna,
        )

    def theil_t(
        self,
        *,
        groups=None,
        value="accessibility",
        sociodemographic_data=None,
        population=None,
        group_columns=None,
        dropna=False,
    ):
        """``cafein.equity.theil_t`` over this frame."""
        from cafein import equity

        return equity.theil_t(
            self,
            groups=groups,
            value=value,
            sociodemographic_data=sociodemographic_data,
            population=population,
            group_columns=group_columns,
            dropna=dropna,
        )

    def mld(
        self,
        *,
        groups=None,
        value="accessibility",
        sociodemographic_data=None,
        population=None,
        group_columns=None,
        dropna=False,
    ):
        """``cafein.equity.mld`` over this frame."""
        from cafein import equity

        return equity.mld(
            self,
            groups=groups,
            value=value,
            sociodemographic_data=sociodemographic_data,
            population=population,
            group_columns=group_columns,
            dropna=dropna,
        )

    def atkinson(
        self,
        *,
        epsilon=1,
        value="accessibility",
        sociodemographic_data=None,
        population=None,
        group_columns=None,
        dropna=False,
    ):
        """``cafein.equity.atkinson`` over this frame."""
        from cafein import equity

        return equity.atkinson(
            self,
            epsilon=epsilon,
            value=value,
            sociodemographic_data=sociodemographic_data,
            population=population,
            group_columns=group_columns,
            dropna=dropna,
        )

    def kolm(
        self,
        *,
        kappa=1,
        value="accessibility",
        sociodemographic_data=None,
        population=None,
        group_columns=None,
        dropna=False,
    ):
        """``cafein.equity.kolm`` over this frame."""
        from cafein import equity

        return equity.kolm(
            self,
            kappa=kappa,
            value=value,
            sociodemographic_data=sociodemographic_data,
            population=population,
            group_columns=group_columns,
            dropna=dropna,
        )

    def hoover(
        self,
        *,
        value="accessibility",
        sociodemographic_data=None,
        population=None,
        group_columns=None,
        dropna=False,
    ):
        """``cafein.equity.hoover`` over this frame."""
        from cafein import equity

        return equity.hoover(
            self,
            value=value,
            sociodemographic_data=sociodemographic_data,
            population=population,
            group_columns=group_columns,
            dropna=dropna,
        )

    def concentration_index(
        self,
        *,
        income,
        variant="standard",
        bounds=None,
        value="accessibility",
        sociodemographic_data=None,
        population=None,
        group_columns=None,
        dropna=False,
    ):
        """``cafein.equity.concentration_index`` over this frame."""
        from cafein import equity

        return equity.concentration_index(
            self,
            income=income,
            variant=variant,
            bounds=bounds,
            value=value,
            sociodemographic_data=sociodemographic_data,
            population=population,
            group_columns=group_columns,
            dropna=dropna,
        )

    def concentration_curve(
        self,
        *,
        income,
        value="accessibility",
        sociodemographic_data=None,
        population=None,
        group_columns=None,
        dropna=False,
    ):
        """``cafein.equity.concentration_curve`` over this frame."""
        from cafein import equity

        return equity.concentration_curve(
            self,
            income=income,
            value=value,
            sociodemographic_data=sociodemographic_data,
            population=population,
            group_columns=group_columns,
            dropna=dropna,
        )

    def suits(
        self,
        *,
        income,
        value="accessibility",
        sociodemographic_data=None,
        population=None,
        group_columns=None,
        dropna=False,
    ):
        """``cafein.equity.suits`` over this frame."""
        from cafein import equity

        return equity.suits(
            self,
            income=income,
            value=value,
            sociodemographic_data=sociodemographic_data,
            population=population,
            group_columns=group_columns,
            dropna=dropna,
        )

    def fgt_poverty(
        self,
        *,
        poverty_line,
        alpha=0,
        deprived="below",
        value="accessibility",
        sociodemographic_data=None,
        population=None,
        group_columns=None,
        dropna=False,
    ):
        """``cafein.equity.fgt_poverty`` over this frame."""
        from cafein import equity

        return equity.fgt_poverty(
            self,
            poverty_line=poverty_line,
            alpha=alpha,
            deprived=deprived,
            value=value,
            sociodemographic_data=sociodemographic_data,
            population=population,
            group_columns=group_columns,
            dropna=dropna,
        )

    def cost_burden(
        self,
        *,
        cost,
        income,
        threshold=0.10,
        detail=False,
        sociodemographic_data=None,
        population=None,
        group_columns=None,
        dropna=False,
    ):
        """``cafein.equity.cost_burden`` over this frame."""
        from cafein import equity

        return equity.cost_burden(
            self,
            cost=cost,
            income=income,
            threshold=threshold,
            detail=detail,
            sociodemographic_data=sociodemographic_data,
            population=population,
            group_columns=group_columns,
            dropna=dropna,
        )

    def residual_income(
        self,
        *,
        cost,
        income,
        sociodemographic_data=None,
        population=None,
        group_columns=None,
        dropna=False,
    ):
        """``cafein.equity.residual_income`` over this frame."""
        from cafein import equity

        return equity.residual_income(
            self,
            cost=cost,
            income=income,
            sociodemographic_data=sociodemographic_data,
            population=population,
            group_columns=group_columns,
            dropna=dropna,
        )

    def lihc(
        self,
        *,
        cost,
        income,
        poverty_line,
        detail=False,
        sociodemographic_data=None,
        population=None,
        group_columns=None,
        dropna=False,
    ):
        """``cafein.equity.lihc`` over this frame."""
        from cafein import equity

        return equity.lihc(
            self,
            cost=cost,
            income=income,
            poverty_line=poverty_line,
            detail=detail,
            sociodemographic_data=sociodemographic_data,
            population=population,
            group_columns=group_columns,
            dropna=dropna,
        )

    def alkire_foster(
        self,
        *,
        dimensions,
        k,
        detail=False,
        sociodemographic_data=None,
        population=None,
        group_columns=None,
        dropna=False,
    ):
        """``cafein.equity.alkire_foster`` over this frame."""
        from cafein import equity

        return equity.alkire_foster(
            self,
            dimensions=dimensions,
            k=k,
            detail=detail,
            sociodemographic_data=sociodemographic_data,
            population=population,
            group_columns=group_columns,
            dropna=dropna,
        )


class NearestDestinations(pd.DataFrame):
    """The ``k`` nearest destinations per origin on a cost axis.

    A pandas DataFrame with one row per (origin, rank): ``from_id``,
    ``rank`` (1..k), ``destination_id``, and ``cost`` — the
    destination's per-origin optimum on the chosen axis, in the axis's
    unit (the time axis reports whole minutes rounded to the nearest
    by default; ``output_time_units="seconds"`` for the exact values).
    Ranking always uses the exact engine values; ties break
    deterministically by (cost, destination position). A destination
    with no journey within ``max_cost`` has no row, so an origin
    reaching two of three schools yields two rows.

    Origins and destinations follow ``Accessibility``: stop ids or
    GeoDataFrames with an ``id`` column (polygons route via their
    centroids), both of the same kind on a ``TransportNetwork``; a
    ``StreetNetwork`` takes point frames and ``transport_mode``. The
    cost axes, engines, validation, and the ``arrival=`` twin of
    ``departure`` match ``Accessibility``:
    ``cost="emissions"``/``"money"`` require their axis's window
    (``departure_time_window``, or ``arrival_time_window`` beside
    ``arrival=``, pricing each destination's deadline profile) and
    ``fares=`` for money; ``cost="distance"`` is street-only.
    Slices and copies degrade to plain DataFrames.

    Parameters
    ----------
    k : int (optional, default: 1)
        How many nearest destinations to rank per origin.
    max_cost : float or datetime.timedelta (optional)
        The search horizon in the axis's unit — minutes (or a
        timedelta) on the time axis, grams, currency units, or metres
        on the others; destinations beyond it are unreachable. ``None``
        keeps the engine's natural bound.
    percentile : float (optional, default: 50)
        With ``cost="time"`` and a ``departure_time_window``, rank each
        destination by this single percentile of its per-departure
        cost distribution. The output carries no percentile column.
    output_time_units : str (optional, default: "minutes")
        The ``cost`` column's unit on the time axis: ``"minutes"``
        (whole minutes rounded to the nearest) or ``"seconds"`` (the
        exact engine values). Other axes always report their native
        unit.

    The routing knobs (``departure``, ``max_rides``, ``router``, the
    exclusions, and the walking options) follow ``Accessibility``.
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
        arrival=None,
        arrival_time_window=None,
        k=1,
        cost="time",
        max_cost=None,
        percentile=50,
        max_rides=8,
        router="auto",
        departure_time_window=None,
        factors=None,
        components=None,
        fares=None,
        chunk=None,
        transport_mode=None,
        max_street_time=None,
        max_travel_time=None,
        exclude_routes=(),
        exclude_trips=(),
        traveler=None,
        exclude_stops=(),
        walking_speed_kmph=None,
        max_walking_time=None,
        snap_distance=None,
        output_time_units="minutes",
    ):
        if hasattr(network, "route_between_stops"):
            (
                exclude_routes,
                exclude_trips,
                exclude_stops,
                walking_speed_kmph,
                max_walking_time,
            ) = folded_constraints(
                traveler,
                network,
                exclude_routes,
                exclude_trips,
                exclude_stops,
                walking_speed_kmph,
                max_walking_time,
            )
        elif traveler is not None:
            raise ValueError(
                "traveler applies to transit; a StreetNetwork takes "
                "transport_mode, max_street_time, and snap_distance"
            )
        from cafein.matrices import _is_point_frame as _points

        if hasattr(network, "route_between_stops") and _points(origins):
            refuse_wheelchair_streets(traveler, "NearestDestinations")
        from cafein._units import (
            duration_seconds,
            travel_time_output,
            validated_output_time_units,
        )

        output_time_units = validated_output_time_units(output_time_units)
        date, departure, arrive_by, raw_window = _product_time_axis(
            departure,
            arrival,
            cost=cost,
            window=departure_time_window,
            router=router,
            arrival_window=arrival_time_window,
            percentiles=[percentile] if percentile != 50 else None,
        )
        if max_rides < 1:
            raise ValueError("max_rides must be at least 1")
        max_transfers = max_rides - 1
        window = duration_seconds(
            "arrival_time_window" if arrive_by else "departure_time_window",
            raw_window,
        )
        max_walking_time = duration_seconds("max_walking_time", max_walking_time)
        max_street_time = duration_seconds("max_street_time", max_street_time)
        if max_travel_time is not None and cost not in ("emissions", "money"):
            raise ValueError(
                "max_travel_time bounds the emissions and money axes; "
                "the time axis takes max_cost"
            )
        max_travel_time = duration_seconds("max_travel_time", max_travel_time)
        max_snap_distance = snap_distance
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError(f"k must be a positive integer, not {k!r}")
        if isinstance(percentile, bool) or not isinstance(percentile, (int, float)):
            raise ValueError(
                f"percentile is one number in [0, 100], not {percentile!r}"
            )
        percentile = float(percentile)
        if not 0.0 <= percentile <= 100.0:
            raise ValueError(f"percentile must be within [0, 100], not {percentile}")
        if percentile != 50.0 and not (cost == "time" and window is not None):
            raise ValueError(
                "percentile ranks the departure window's cost "
                "distribution; it needs cost='time' and "
                "departure_time_window="
            )
        if cost == "time":
            # The horizon is a duration like every other time input.
            max_cost = duration_seconds("max_cost", max_cost)
            horizon = None if max_cost is None else float(max_cost)
        elif max_cost is not None:
            horizon = float(max_cost)
            if not math.isfinite(horizon) or horizon <= 0.0:
                raise ValueError(
                    f"max_cost must be a positive finite number, not {max_cost!r}"
                )
        else:
            horizon = None
        from cafein import _cafein
        from cafein.matrices import _is_street_network

        origins, _origin_dtype = freeze_ids(origins)
        destinations, _destination_dtype = freeze_ids(destinations)
        _id_dtypes = {
            "from_id": _origin_dtype,
            "destination_id": _destination_dtype,
        }
        origins = sequence_not_string("origins", origins)
        destinations = sequence_not_string("destinations", destinations)
        exclude_routes = id_sequence("exclude_routes", exclude_routes)
        exclude_trips = id_sequence("exclude_trips", exclude_trips)
        exclude_stops = id_sequence("exclude_stops", exclude_stops)
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
        if cost in ("emissions", "money") and window is None:
            raise ValueError(
                f"cost={cost!r} optimizes over a departure window; "
                "pass departure_time_window="
            )
        percentiles = [percentile] if cost == "time" and window is not None else None
        if (
            arrive_by
            and k == 1
            and cost == "time"
            and window is None
            and not _is_street_network(network)
        ):
            # The nearest-of-set fast path: ONE tagged reverse seeded
            # with every destination replaces one run per destination;
            # the winner's tag names the destination. Behavior matches
            # the per-destination fan-out exactly — the shared cost
            # surface's validations included.
            if transport_mode is not None or max_street_time is not None:
                raise ValueError(
                    "transport_mode and max_street_time apply to a "
                    "StreetNetwork; a TransportNetwork routes door to door"
                )
            if horizon is not None and (not math.isfinite(horizon) or horizon <= 0.0):
                raise ValueError(
                    f"max_cost must be a positive finite number, not {max_cost!r}"
                )
            frame = _nearest_of_set(
                network,
                origins,
                destinations,
                date,
                departure,
                max_transfers,
                horizon,
                chunk,
                (exclude_routes, exclude_trips, exclude_stops),
                (walking_speed_kmph, max_walking_time, snap_distance),
                travel_time_output,
                output_time_units,
            )
            super().__init__(restore_id_dtypes(frame, _id_dtypes))
            return
        matrix, from_ids, to_ids, resolved_percentiles = _resolved_cost_matrix(
            network,
            origins,
            destinations,
            date,
            departure,
            cost,
            window,
            percentiles,
            None,
            factors,
            components,
            fares,
            max_transfers,
            max_travel_time,
            router,
            chunk,
            transport_mode,
            max_street_time,
            max_snap_distance,
            exclude_routes,
            exclude_trips,
            exclude_stops,
            walking_speed_kmph,
            max_walking_time,
            label="NearestDestinations",
            arrive_by=arrive_by,
        )
        matrix = np.asarray(matrix)
        if resolved_percentiles is not None:
            matrix = matrix[:, :, 0]
        if matrix.dtype.kind == "f":
            indices, costs = _cafein.aggregate_nearest_f64(
                np.ascontiguousarray(matrix, dtype="float64"), k, horizon
            )
        else:
            indices, costs = _cafein.aggregate_nearest(
                np.ascontiguousarray(matrix, dtype="uint32"), k, horizon
            )
        indices = np.asarray(indices)
        costs = np.asarray(costs)
        origin_rows, ranks = np.nonzero(indices >= 0)
        kept = indices[origin_rows, ranks]
        kept_costs = costs[origin_rows, ranks]
        if cost == "time":
            kept_costs = travel_time_output(kept_costs, output_time_units)
        identifiers = np.asarray(list(from_ids), dtype=object)
        targets = np.asarray(list(to_ids), dtype=object)
        frame = pd.DataFrame(
            {
                "from_id": identifiers[origin_rows],
                "rank": ranks.astype("int64") + 1,
                "destination_id": targets[kept],
                "cost": kept_costs,
            }
        )
        super().__init__(restore_id_dtypes(frame, _id_dtypes))

    def dominance_areas(self, origins):
        """Polygon origins dissolved by their rank-1 destination.

        The network-Voronoi map product: every origin polygon joins the
        area of the destination it reaches first. Returns a
        GeoDataFrame with one row per destination —
        ``destination_id``, the dissolved ``geometry``, and
        ``origins``, how many origin polygons the area absorbed.
        Origins without a rank-1 row (nothing reachable) are absent.
        Raises for point origins: a dissolve needs polygons.
        """
        import geopandas

        if not isinstance(origins, geopandas.GeoDataFrame) or "id" not in origins:
            raise ValueError(
                "dominance_areas needs the origin GeoDataFrame with its " "'id' column"
            )
        polygonal = origins.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        if not polygonal.all():
            raise ValueError(
                "dominance_areas dissolves polygon origins; point "
                "origins have no area to dissolve"
            )
        # The frame's from_id may carry a restored integer dtype and
        # the caller's origin ids any dtype at all; both sides join on
        # the same string key so duplicates cannot hide behind a dtype
        # difference. A fresh two-column frame keeps the join
        # immune to the caller's own column names (a renamed geometry
        # column, a pre-existing destination_id).
        slim = geopandas.GeoDataFrame(
            {"_key": [str(value) for value in origins["id"]]},
            geometry=origins.geometry.values,
            crs=origins.crs,
        )
        duplicated = slim["_key"].duplicated()
        if duplicated.any():
            raise ValueError(
                "dominance_areas needs unique origin ids; "
                f"{int(duplicated.sum())} id(s) repeat"
            )
        first = pd.DataFrame(self[self["rank"] == 1])[["from_id", "destination_id"]]
        repeated = first["from_id"].duplicated()
        if repeated.any():
            raise ValueError(
                "dominance_areas needs unique origin ids; the frame "
                f"ranks {int(repeated.sum())} repeated origin id(s)"
            )
        # The frame's from_id may carry a restored integer dtype;
        # normalise both sides to the same string key for the join.
        first = first.assign(_key=first["from_id"].astype(str))
        joined = slim.merge(first, on="_key", how="inner")
        dissolved = joined[["destination_id", "geometry"]].dissolve(by="destination_id")
        counts = joined.groupby("destination_id").size()
        return geopandas.GeoDataFrame(
            {
                "destination_id": dissolved.index,
                "geometry": dissolved.geometry.values,
                "origins": counts.reindex(dissolved.index).astype("int64").values,
            },
            crs=origins.crs,
        )


def _h3_module():
    try:
        import h3
    except ImportError as error:
        raise ImportError(
            "Catchment renders H3 cells; install the optional extra: "
            "pip install 'cafein[h3]'"
        ) from error
    return h3


class Catchment(gpd.GeoDataFrame):
    """Budget catchments on a cost axis: one cell-union polygon per row.

    A GeoDataFrame with one row per (origin, budget):
    ``from_id``, ``budget`` (echoed as passed — minutes on the time
    axis, the axis's native unit otherwise), and ``geometry``, the
    union of H3 cells at `resolution` over every street-network vertex
    reached within the budget (EPSG:4326). Rows record their budget,
    so banded rings are a difference operation downstream; nested
    budgets yield nested cell unions. An origin reaching nothing under
    a budget has no row — absence, exactly like an unreachable rank.
    Slices and copies are ordinary GeoDataFrame views.

    The target universe is the network's own street vertices — no
    destinations argument. On a ``TransportNetwork`` the door-to-door
    contract seeds the walking spread BOTH from the snapped origin
    itself (an origin with no reachable stop still has a walking
    catchment) AND from every reached stop at its arrival cost: on the
    time axis a vertex is reached when stop arrival plus the walk fits
    the budget; on the emissions/money axes walking is zero-cost and
    each qualifying stop's spread is bounded by ``max_walking_time``
    alone. On a ``StreetNetwork`` the vertices are the mode spread
    under `transport_mode` — seconds on the time axis, street metres
    with ``cost="distance"``.

    Parameters
    ----------
    origins : list of str, or GeoDataFrame
        Stop ids, or points with an ``id`` column (polygons route via
        centroids); typically one or a few origins.
    budgets : sequence of float or datetime.timedelta (optional, default: (30.0,))
        One or more cutoffs — minutes (or timedeltas) on the time
        axis, grams, currency units, or metres on the others.
    resolution : int (optional, default: 9)
        The H3 cell resolution of the rendering; it never affects
        reachability.
    percentile : float (optional, default: 50)
        With ``cost="time"``, a ``departure_time_window``, and stop
        origins: a vertex is reached when this single percentile of
        the per-departure arrival distribution fits the budget. The
        output carries no percentile dimension.

    ``cost``, ``departure``, ``departure_time_window``, ``max_rides``,
    ``router``, ``factors``/``components``/``fares``,
    ``transport_mode``, the exclusions, and the walking options follow
    ``Accessibility``. With ``arrival=`` (exactly one of the two) the
    catchment is the region that can *reach* the origin by the
    deadline. On the time axis stop seeds run in the
    before-deadline domain so the walking field maximizes the
    composed departure, and a cell belongs to a budget when its
    winning journey's own duration fits — a journey arriving well
    before the deadline keeps its cells even when the span to the
    deadline does not. On the emissions and money axes (their
    ``arrival_time_window`` beside ``arrival=``) each stop seeds at
    its deadline-profile optimum toward the origin, so the region is
    everything that can reach the place within the cost budget by
    some profiled deadline. The wheelchair traveler's directed street
    bridge does not combine with ``arrival=`` yet. A wheelchair traveler's catchment rides the
    fixed compiled profile at the multimodal snap radius —
    ``walking_speed_kmph`` and ``snap_distance`` are rejected beside
    it, while ``max_walking_time`` stays configurable. Requires the
    optional ``h3`` extra (``cafein[h3]``).
    """

    @property
    def _constructor(self):
        return gpd.GeoDataFrame

    def __init__(
        self,
        network,
        origins,
        departure=None,
        *,
        arrival=None,
        arrival_time_window=None,
        cost="time",
        budgets=(30.0,),
        resolution=9,
        percentile=50,
        max_rides=8,
        router="auto",
        departure_time_window=None,
        factors=None,
        components=None,
        fares=None,
        chunk=None,
        transport_mode=None,
        exclude_routes=(),
        exclude_trips=(),
        traveler=None,
        exclude_stops=(),
        walking_speed_kmph=None,
        max_walking_time=None,
        snap_distance=None,
    ):
        if hasattr(network, "route_between_stops"):
            (
                exclude_routes,
                exclude_trips,
                exclude_stops,
                walking_speed_kmph,
                max_walking_time,
            ) = folded_constraints(
                traveler,
                network,
                exclude_routes,
                exclude_trips,
                exclude_stops,
                walking_speed_kmph,
                max_walking_time,
            )
        elif traveler is not None:
            raise ValueError(
                "traveler applies to transit; a StreetNetwork takes "
                "transport_mode, max_street_time, and snap_distance"
            )
        from cafein.matrices import _is_point_frame as _points  # noqa: F401

        wheeled = (
            traveler is not None
            and traveler.wheelchair
            and hasattr(network, "route_between_stops")
        )
        if wheeled:
            # The residual spread rides the compiled wheelchair profile
            # over the multimodal graph instead of the walking field.
            if "wheelchair" not in (network._core.street_modes or ()):
                raise ValueError(
                    "the wheelchair traveler routes the streets on the "
                    "wheelchair mode; build the network with "
                    "street_modes=('walk', 'wheelchair') (or load such an "
                    "artifact)"
                )
            if walking_speed_kmph is not None:
                raise ValueError(
                    "walking_speed_kmph cannot reshape the wheelchair "
                    "street profile, which rides its fixed speed"
                )
            if snap_distance is not None:
                # The synthesized policy's transit seeds snap at the
                # fixed multimodal radius; a divergent field radius
                # would mix snapping rules within one catchment.
                raise ValueError(
                    "the wheelchair traveler's street side snaps at the "
                    "multimodal radius; snap_distance= beside it is a "
                    "conflict"
                )
        import shapely.geometry

        from cafein._units import duration_seconds

        h3 = _h3_module()
        date, departure, arrive_by, raw_window = _product_time_axis(
            departure,
            arrival,
            cost=cost,
            window=departure_time_window,
            router=router,
            arrival_window=arrival_time_window,
            percentiles=[percentile] if percentile != 50 else None,
        )
        if arrive_by and wheeled:
            raise ValueError(
                "the wheelchair traveler's street bridge is a directed "
                "surface; it does not combine with arrival= yet"
            )
        if max_rides < 1:
            raise ValueError("max_rides must be at least 1")
        max_transfers = max_rides - 1
        window = duration_seconds(
            "arrival_time_window" if arrive_by else "departure_time_window",
            raw_window,
        )
        max_walking_time = duration_seconds("max_walking_time", max_walking_time)
        max_snap_distance = snap_distance
        if not isinstance(resolution, int) or isinstance(resolution, bool):
            raise ValueError(f"resolution is an H3 level 0-15, not {resolution!r}")
        if not 0 <= resolution <= 15:
            raise ValueError(f"resolution must be within 0-15, not {resolution}")
        if router not in ("auto", "raptor", "tbtr"):
            raise ValueError(
                f"router must be 'auto', 'raptor', or 'tbtr', not {router!r}"
            )
        if isinstance(percentile, bool) or not isinstance(percentile, (int, float)):
            raise ValueError(
                f"percentile is one number in [0, 100], not {percentile!r}"
            )
        percentile = float(percentile)
        if not 0.0 <= percentile <= 100.0:
            raise ValueError(f"percentile must be within [0, 100], not {percentile}")
        from cafein import _cafein  # noqa: F401  (asserts the compiled core)
        from cafein.matrices import _chunk_slice, _is_street_network, _point_list

        origins, _origin_dtype = freeze_ids(origins)
        _id_dtypes = {"from_id": _origin_dtype}
        origins = sequence_not_string("origins", origins)
        exclude_routes = id_sequence("exclude_routes", exclude_routes)
        exclude_trips = id_sequence("exclude_trips", exclude_trips)
        exclude_stops = id_sequence("exclude_stops", exclude_stops)
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
        if cost == "time":
            raw_budgets = sequence_not_string("budgets", budgets)
            budget_values = [
                float(duration_seconds("budgets", budget)) for budget in raw_budgets
            ]
            budget_column = [
                (
                    budget.total_seconds() / 60
                    if isinstance(budget, datetime.timedelta)
                    else float(budget)
                )
                for budget in raw_budgets
            ]
        else:
            budget_values = _budget_list(budgets)
            budget_column = budget_values
        if not budget_values:
            raise ValueError("budgets must name at least one cutoff")
        for budget in budget_values:
            if not math.isfinite(budget) or budget <= 0.0:
                raise ValueError(
                    f"budgets must be positive finite numbers, not {budget!r}"
                )
        street_kind = _is_street_network(network)
        if percentile != 50.0 and not (
            cost == "time" and window is not None and not street_kind
        ):
            raise ValueError(
                "percentile ranks the departure window's arrival "
                "distribution; it needs cost='time', "
                "departure_time_window=, and a TransportNetwork"
            )
        rows = []
        if street_kind:
            if arrive_by:
                raise ValueError(
                    "arrival applies to transit; a StreetNetwork takes "
                    "transport_mode and snap_distance"
                )
            if transport_mode is None:
                raise ValueError(
                    "a StreetNetwork needs transport_mode= (walk, bicycle, "
                    "e_bike, e_scooter, wheelchair, or car on a car-enabled build)"
                )
            if cost not in ("time", "distance"):
                raise ValueError(
                    f"cost={cost!r} needs the transit cost engines; a "
                    "StreetNetwork serves cost='time' and cost='distance'"
                )
            rejected = {
                "departure": departure,
                "departure_time_window": window,
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
                    "transport_mode and snap_distance"
                )
            from cafein.streets import MAX_SNAP_DISTANCE

            snap = float(
                MAX_SNAP_DISTANCE if max_snap_distance is None else max_snap_distance
            )
            if not math.isfinite(snap) or snap <= 0.0:
                raise ValueError(
                    "snap_distance must be a positive finite distance, "
                    f"not {max_snap_distance!r}"
                )
            from_ids, points = _point_list(origins, "origins")
            keep = _chunk_slice(len(from_ids), chunk)
            from_ids = from_ids[keep]
            points = points[keep]
            horizon = max(budget_values)
            for identifier, point in zip(from_ids, points):
                lats, lons, costs = network._core._reached_vertices(
                    point, transport_mode, horizon, cost, snap
                )
                rows.extend(
                    _cell_rows(
                        h3,
                        shapely.geometry,
                        identifier,
                        lats,
                        lons,
                        costs,
                        budget_values,
                        budget_column,
                        resolution,
                    )
                )
        else:
            if transport_mode is not None:
                raise ValueError(
                    "transport_mode applies to a StreetNetwork; a "
                    "TransportNetwork routes door to door"
                )
            if date is None or departure is None:
                raise TypeError("Catchment requires departure or arrival")
            if cost == "distance":
                raise ValueError(
                    "cost='distance' is not an optimizable transit axis; "
                    "street networks serve cost='distance' natively"
                )
            if cost in ("emissions", "money") and window is None:
                raise ValueError(
                    f"cost={cost!r} optimizes over a departure window; "
                    "pass departure_time_window="
                )
            geo = _is_geo(origins)
            if geo and cost == "time" and window is None and router == "tbtr":
                raise ValueError(
                    "the coordinate one-to-all rides RAPTOR; router='tbtr' "
                    "serves stop origins"
                )
            if geo and cost == "time" and window is not None:
                raise ValueError(
                    "windowed time catchments rank per-stop arrival "
                    "distributions over all stops; stop origins serve "
                    "them today"
                )
            from cafein.network import _walk_options

            speed_kmph, walk_cutoff, snap = _walk_options(
                walking_speed_kmph, max_walking_time, max_snap_distance
            )
            speed_kmph = float(speed_kmph)
            walk_cutoff = float(walk_cutoff)
            snap = float(snap)
            if not math.isfinite(speed_kmph) or speed_kmph <= 0.0:
                raise ValueError(
                    "walking_speed_kmph must be a positive finite speed, "
                    f"not {walking_speed_kmph!r}"
                )
            if not math.isfinite(snap) or snap <= 0.0:
                raise ValueError(
                    "snap_distance must be a positive finite distance, "
                    f"not {max_snap_distance!r}"
                )
            stop_ids = [stop for stop, _, _ in network.stops]
            if geo:
                from_ids, points = _point_list(origins, "origins")
            else:
                from_ids = _stop_ids(origins, "origins")
                # The canonical resolver: qualified ids, merged-feed
                # aliases, and unknown-id errors behave as they do on
                # every routing entry.
                resolved = list(network._core._stop_indices(from_ids))
                stops_list = network.stops
                points = []
                for stop, index in zip(from_ids, resolved):
                    _, lat, lon = stops_list[index]
                    if lat is None:
                        raise ValueError(
                            f"origin stop {stop!r} carries no coordinate; "
                            "a catchment spreads from the origin's location"
                        )
                    points.append((lat, lon))
            keep = _chunk_slice(len(from_ids), chunk)
            from_ids = list(from_ids[keep] if geo else from_ids[keep])
            points = list(points[keep])
            if arrive_by and cost == "time":
                # The arrive-by catchment: the region that can REACH
                # the given place by the deadline. Stop seeds run in
                # the before-deadline domain — `deadline − departure`
                # keys maximize the composed departure through the
                # walk — with each seed's achieved arrival and rides
                # as riders; membership judges the winner's own
                # duration, so the field bound extends by the maximum
                # retained seed slack (`deadline − achieved`).
                hours, minutes, seconds = str(departure).split(":")
                deadline_s = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
                if window is not None:
                    # The before-deadline domain anchors at the final
                    # mark: winners serve marks across the window, and
                    # an anchor before a winner's own mark would push
                    # its key negative. Durations are anchor-invariant
                    # (key minus slack), so membership is unchanged.
                    deadline_s += 60 * ((int(window) - 1) // 60)
                speed_ms = speed_kmph / 3.6
                horizon = max(budget_values)
                for identifier, point in zip(from_ids, points):
                    if geo:
                        walks = network._core.access_stops(
                            point[0], point[1], speed_kmph, walk_cutoff, snap
                        )
                        egress = [
                            (stop, int(walk_seconds))
                            for stop, walk_seconds in walks.items()
                        ]
                    else:
                        egress = [(identifier, 0)]
                    if window is None:
                        reaches = network._core._arrive_by_reaches(
                            egress,
                            date,
                            departure,
                            max_transfers,
                            list(exclude_routes),
                            list(exclude_trips),
                            list(exclude_stops),
                        )
                    else:
                        # Deadlines profile over the arrival window and
                        # each stop seeds with its percentile winner.
                        reaches = network._core._arrive_by_percentile_reaches(
                            egress,
                            date,
                            departure,
                            window,
                            percentile,
                            max_transfers,
                            list(exclude_routes),
                            list(exclude_trips),
                            list(exclude_stops),
                        )
                    seeds = []
                    max_slack = 0.0
                    for stop, latest, _rides, achieved in reaches:
                        if achieved - latest > horizon:
                            continue
                        slack = float(deadline_s - achieved)
                        seeds.append((stop, float(deadline_s - latest), _rides, slack))
                        max_slack = max(max_slack, slack)
                    lats, lons, costs = network._core._arrive_by_catchment_walk_field(
                        point, seeds, speed_ms, horizon + max_slack, snap
                    )
                    rows.extend(
                        _cell_rows(
                            h3,
                            shapely.geometry,
                            identifier,
                            lats,
                            lons,
                            costs,
                            budget_values,
                            budget_column,
                            resolution,
                        )
                    )
                import geopandas

                frame = geopandas.GeoDataFrame(
                    {
                        "from_id": [row[0] for row in rows],
                        "budget": [row[1] for row in rows],
                    },
                    geometry=[row[2] for row in rows],
                    crs="EPSG:4326",
                )
                super().__init__(restore_id_dtypes(frame, _id_dtypes))
                return
            surfaces = _catchment_stop_costs(
                network,
                origins,
                from_ids,
                points,
                geo,
                date,
                departure,
                cost,
                window,
                percentile,
                factors,
                components,
                fares,
                max_transfers,
                router,
                chunk,
                (exclude_routes, exclude_trips, exclude_stops),
                (speed_kmph, walk_cutoff, snap),
                stop_ids,
                wheeled,
                arrive_by=arrive_by,
            )
            speed_ms = speed_kmph / 3.6
            for identifier, point, stop_costs in zip(from_ids, points, surfaces):
                if cost == "time":
                    horizon = max(budget_values)
                    seeds = [
                        (index, seconds)
                        for index, seconds in stop_costs
                        if seconds <= horizon
                    ]
                    if wheeled:
                        lats, lons, costs = network._core._catchment_directed_field(
                            point, seeds, "wheelchair", horizon, snap
                        )
                    else:
                        lats, lons, costs = network._core._catchment_walk_field(
                            point, seeds, speed_ms, horizon, snap
                        )
                    rows.extend(
                        _cell_rows(
                            h3,
                            shapely.geometry,
                            identifier,
                            lats,
                            lons,
                            costs,
                            budget_values,
                            budget_column,
                            resolution,
                        )
                    )
                else:
                    for budget, echoed in zip(budget_values, budget_column):
                        seeds = [
                            (index, 0.0)
                            for index, value in stop_costs
                            if value <= budget
                        ]
                        if wheeled:
                            lats, lons, _costs = (
                                network._core._catchment_directed_field(
                                    point, seeds, "wheelchair", walk_cutoff, snap
                                )
                            )
                        else:
                            lats, lons, _costs = network._core._catchment_walk_field(
                                point, seeds, speed_ms, walk_cutoff, snap
                            )
                        cells = {
                            h3.latlng_to_cell(lat, lon, resolution)
                            for lat, lon in zip(lats, lons)
                        }
                        if cells:
                            rows.append(
                                (
                                    identifier,
                                    echoed,
                                    shapely.geometry.shape(
                                        h3.cells_to_h3shape(cells).__geo_interface__
                                    ),
                                )
                            )
        import geopandas

        frame = geopandas.GeoDataFrame(
            {
                "from_id": [row[0] for row in rows],
                "budget": [row[1] for row in rows],
            },
            geometry=[row[2] for row in rows],
            crs="EPSG:4326",
        )
        super().__init__(restore_id_dtypes(frame, _id_dtypes))


def _cell_rows(
    h3,
    geometry_module,
    identifier,
    lats,
    lons,
    costs,
    budget_values,
    budget_column,
    resolution,
):
    """One (from_id, budget, polygon) row per non-empty budget filter
    of a reached-vertex field whose costs share the budgets' unit."""
    rows = []
    costs = np.asarray(costs)
    for budget, echoed in zip(budget_values, budget_column):
        within = costs <= budget
        cells = {
            h3.latlng_to_cell(lat, lon, resolution)
            for lat, lon in zip(np.asarray(lats)[within], np.asarray(lons)[within])
        }
        if cells:
            rows.append(
                (
                    identifier,
                    echoed,
                    geometry_module.shape(h3.cells_to_h3shape(cells).__geo_interface__),
                )
            )
    return rows


def _catchment_stop_costs(
    network,
    origins,
    from_ids,
    points,
    geo,
    date,
    departure,
    cost,
    window,
    percentile,
    factors,
    components,
    fares,
    max_transfers,
    router,
    chunk,
    exclusions,
    walk,
    stop_ids,
    wheeled=False,
    arrive_by=False,
):
    """Per origin, the ``(global stop index, cost)`` pairs of every
    reached stop on the chosen axis — the walking field's seeds. With
    ``arrive_by`` the cost surface transposes: each pair is a stop's
    deadline-profile optimum TO the place, so the field seeds the
    region that can reach it. With
    `wheeled`, point origins reach the stops through the synthesized
    wheelchair street policy — the seeds themselves never ride stairs
    — on the time axis; the cost surfaces carry no policy support yet,
    so a wheeled point origin on the emissions or money axis is
    refused rather than silently walked."""
    import numpy as np

    if cost in ("emissions", "money"):
        if geo and wheeled:
            raise ValueError(
                "wheelchair point-origin catchments ride the time axis; "
                f"the {cost} cost surface has no street-policy support "
                "yet — route from stop origins instead"
            )
        if geo:
            # Point origins ride the point-form cost surface; the
            # stop side is the coordinate-bearing stops as points, so
            # its axis aligns with those stops.
            import geopandas

            located = [
                (stop, lat, lon) for stop, lat, lon in network.stops if lat is not None
            ]
            stop_frame = geopandas.GeoDataFrame(
                {"id": [stop for stop, _, _ in located]},
                geometry=geopandas.points_from_xy(
                    [lon for _, _, lon in located],
                    [lat for _, lat, _ in located],
                ),
                crs="EPSG:4326",
            )
            surface_stops = [stop for stop, _, _ in located]
        else:
            stop_frame = stop_ids
            surface_stops = stop_ids
        if arrive_by:
            # Transposed: the stops are the surface's origins and the
            # (already chunked) places its destinations; each place's
            # seed list is then its column over the stop axis.
            from cafein.matrices import _point_frame

            places = _point_frame(from_ids, points) if geo else list(from_ids)
            surface, _from, _to = _transit_cost_surface(
                network,
                stop_frame,
                places,
                date,
                departure,
                cost,
                window,
                factors,
                components,
                fares,
                max_transfers,
                None,
                router,
                None,
                [list(ids) for ids in exclusions],
                walk,
                arrive_by=True,
            )
            columns = np.asarray(list(network._core._stop_indices(surface_stops)))
            surface = np.asarray(surface)
            return [
                [
                    (int(columns[at]), float(value))
                    for at, value in enumerate(surface[:, place])
                    if np.isfinite(value)
                ]
                for place in range(surface.shape[1])
            ]
        surface, _from, _to = _transit_cost_surface(
            network,
            origins,
            stop_frame,
            date,
            departure,
            cost,
            window,
            factors,
            components,
            fares,
            max_transfers,
            None,
            router,
            chunk,
            [list(ids) for ids in exclusions],
            walk,
        )
        columns = np.asarray(list(network._core._stop_indices(surface_stops)))
        return [
            [
                (int(columns[at]), float(value))
                for at, value in enumerate(row)
                if np.isfinite(value)
            ]
            for row in np.asarray(surface)
        ]
    exclude_routes, exclude_trips, exclude_stops = exclusions
    if geo:
        # Point origins: the door-to-door one-to-all per origin — a
        # dict of arrival seconds keyed by stop id. The walking trio
        # arrives resolved and validated by the constructor. A wheeled
        # origin reaches the stops through the synthesized wheelchair
        # policy, transfer grant included when the set is computed.
        speed, cutoff, snap = (float(value) for value in walk)
        wheel_policy = None
        if wheeled:
            from cafein.policy import StreetLegPolicy

            transfers = None
            binding = network._core._mode_transfer_binding
            if binding is not None and binding[0] == "wheelchair":
                transfers = {"wheelchair": binding[1]}
            wheel_policy = StreetLegPolicy(
                access={"wheelchair": cutoff},
                egress={"wheelchair": cutoff},
                transfers=transfers,
            )
        fields = []
        for point in points:
            if wheel_policy is not None:
                from cafein.network import _policy_transfer_mode
                from cafein.policy import reduction_modes

                modes = reduction_modes(wheel_policy, "access", cutoff)
                transfer_arg = _policy_transfer_mode(wheel_policy)
                access = [
                    (stop, seconds)
                    for stop, seconds, *_ in network._core._reduced_street_offsets(
                        *point, False, modes, transfer_mode=transfer_arg
                    )
                ]
                arrivals = network._core._travel_times_with_access(
                    access,
                    date,
                    departure,
                    max_transfers,
                    transfer_mode=transfer_arg,
                    exclude_routes=list(exclude_routes),
                    exclude_trips=list(exclude_trips),
                    exclude_stops=list(exclude_stops),
                )
            else:
                arrivals = network._core.travel_times_from_coordinate(
                    point,
                    date,
                    departure,
                    max_transfers,
                    list(exclude_routes),
                    list(exclude_trips),
                    list(exclude_stops),
                    speed,
                    cutoff,
                    snap,
                )
            reached_ids = list(arrivals)
            indices = list(network._core._stop_indices(reached_ids))
            fields.append(
                [
                    (int(index), float(arrivals[stop]))
                    for index, stop in zip(indices, reached_ids)
                ]
            )
        return fields
    if window is not None:
        matrix = network._core.travel_time_percentiles(
            list(from_ids),
            date,
            departure,
            int(window),
            [percentile],
            max_transfers,
            router,
            list(exclude_routes),
            list(exclude_trips),
            list(exclude_stops),
        )
        matrix = np.asarray(matrix)[:, :, 0]
    else:
        matrix, _ids, _to, _res = network._time_matrix_with_ids(
            list(from_ids),
            date,
            departure,
            max_transfers,
            destinations=None,
            window=None,
            percentiles=None,
            confidence=None,
            chunk=None,
            router=router,
            exclude_routes=exclude_routes,
            exclude_trips=exclude_trips,
            exclude_stops=exclude_stops,
            walking_speed_kmph=walk[0],
            max_walking_time=walk[1],
            max_snap_distance=walk[2],
        )
        matrix = np.asarray(matrix)
    unreached = np.iinfo(np.uint32).max
    return [
        [(index, float(value)) for index, value in enumerate(row) if value != unreached]
        for row in matrix
    ]
