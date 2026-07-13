"""Public-transport routing with per-leg distance and emissions tracking."""

__all__ = [
    "TransportNetwork",
    "TravelCostMatrix",
    "TravelTimeMatrix",
    "DetailedItineraries",
    "travel_cost_table",
    "exhaustive_frontier",
    "journey_frontier",
    "journey_frontiers",
    "frontier_table",
    "least_emissions",
    "least_fare",
    "__version__",
]


def __getattr__(name):
    # Resolved lazily so that the pure-Python modules (cafein.streets)
    # stay importable without the compiled core.
    if name == "TransportNetwork":
        from cafein.network import TransportNetwork

        return TransportNetwork
    if name == "TravelCostMatrix":
        from cafein.matrices import TravelCostMatrix

        return TravelCostMatrix
    if name == "TravelTimeMatrix":
        from cafein.matrices import TravelTimeMatrix

        return TravelTimeMatrix
    if name == "DetailedItineraries":
        from cafein.itineraries import DetailedItineraries

        return DetailedItineraries
    if name == "travel_cost_table":
        from cafein.matrices import travel_cost_table

        return travel_cost_table
    if name == "exhaustive_frontier":
        from cafein.frontier import exhaustive_frontier

        return exhaustive_frontier
    if name == "journey_frontier":
        from cafein.frontier import journey_frontier

        return journey_frontier
    if name == "journey_frontiers":
        from cafein.frontier import journey_frontiers

        return journey_frontiers
    if name == "frontier_table":
        from cafein.frontier import frontier_table

        return frontier_table
    if name == "least_emissions":
        from cafein.frontier import least_emissions

        return least_emissions
    if name == "least_fare":
        from cafein.frontier import least_fare

        return least_fare
    if name == "__version__":
        from cafein._cafein import __version__

        return __version__
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
