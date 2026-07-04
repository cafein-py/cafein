"""Public-transport routing with per-leg distance and emissions tracking."""

__all__ = ["TransportNetwork", "TravelCostMatrix", "travel_cost_table", "__version__"]


def __getattr__(name):
    # Resolved lazily so that the pure-Python modules (cafein.streets)
    # stay importable without the compiled core.
    if name == "TransportNetwork":
        from cafein.network import TransportNetwork

        return TransportNetwork
    if name == "TravelCostMatrix":
        from cafein.matrices import TravelCostMatrix

        return TravelCostMatrix
    if name == "travel_cost_table":
        from cafein.matrices import travel_cost_table

        return travel_cost_table
    if name == "__version__":
        from cafein._cafein import __version__

        return __version__
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
