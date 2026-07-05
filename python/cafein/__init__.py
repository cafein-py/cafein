"""Public-transport routing with per-leg distance and emissions tracking."""

__all__ = [
    "TransportNetwork",
    "TravelCostMatrix",
    "TravelTimeMatrix",
    "DetailedItineraries",
    "travel_cost_table",
    "journey_frontier",
    "least_emissions",
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
    if name == "journey_frontier":
        from cafein.frontier import journey_frontier

        return journey_frontier
    if name == "least_emissions":
        from cafein.frontier import least_emissions

        return least_emissions
    if name == "__version__":
        from cafein._cafein import __version__

        return __version__
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
