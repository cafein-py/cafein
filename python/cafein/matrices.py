"""Matrix computers over a transport network."""

import numpy as np
import pandas as pd
import shapely


class TravelCostMatrix(pd.DataFrame):
    """The fastest journey's aggregated costs per OD pair, long format.

    A pandas DataFrame with one row per reachable OD pair: ``from_id``
    and ``to_id`` (stop identifiers), ``travel_time`` (seconds),
    ``transfers``, ``transit_distance`` and ``walk_distance`` (meters),
    and ``emissions`` (grams CO₂e over the ridden legs; NaN where a
    ridden trip has no matching factor row). With ``geometries=True``
    each row adds ``geometry``, the ridden legs as a shapely
    MultiLineString in EPSG:4326 — convert with
    ``geopandas.GeoDataFrame(matrix, crs="EPSG:4326")``.

    One RAPTOR run serves each origin, fanned out over all cores; each
    pair's costs come from its fastest journey (ties resolved toward
    fewer rides). Unreachable pairs are absent. Requires a network built
    with trip distances (the default), and with leg geometries for
    ``geometries=True``. Slices and copies degrade to plain DataFrames.

    Parameters
    ----------
    network : TransportNetwork
        The network to compute on.
    origins : list of str (optional)
        Origin stop_ids; every stop when omitted.
    destinations : list of str (optional)
        Destination stop_ids; every stop when omitted.
    date : str
        Service date as ``YYYY-MM-DD``.
    departure : str
        Departure time at every origin as ``HH:MM:SS``.
    max_transfers : int (optional, default: 4)
        Maximum number of transfers between rides.
    factors : DataFrame or path (optional)
        Extra emission-factor rows layered over the shipped defaults;
        see ``cafein.emissions.load_factors``.
    components : list of str (optional)
        The life-cycle components to include (default: all four); see
        ``cafein.emissions.annotate``.
    geometries : bool (optional, default: False)
        Attach each pair's ridden legs as geometry. Off by default:
        per-pair geometries over large matrices are enormous.
    """

    @property
    def _constructor(self):
        return pd.DataFrame

    def __init__(
        self,
        network,
        origins=None,
        destinations=None,
        date=None,
        departure=None,
        *,
        max_transfers=4,
        factors=None,
        components=None,
        geometries=False,
    ):
        from cafein import emissions

        if date is None or departure is None:
            raise TypeError("TravelCostMatrix requires date and departure")
        stop_ids = np.array([stop for stop, _, _ in network.stops], dtype=object)
        origins = stop_ids.tolist() if origins is None else [str(o) for o in origins]
        to_stops = None if destinations is None else [str(d) for d in destinations]
        table = network._core.travel_cost_matrix(
            origins,
            date,
            departure,
            emissions.trip_factors(network, factors, components),
            max_transfers,
            to_stops,
            geometries,
        )
        data = {
            "from_id": np.array(origins, dtype=object)[table["from"]],
            "to_id": stop_ids[table["to"]],
            "travel_time": table["travel_time"],
            "transfers": np.maximum(table["rides"], 1) - 1,
            "transit_distance": table["transit_distance"],
            "walk_distance": table["walk_distance"],
            "emissions": table["emissions"],
        }
        if geometries:
            data["geometry"] = shapely.from_wkb(
                np.array(table["geometry"], dtype=object)
            )
        super().__init__(pd.DataFrame(data))
