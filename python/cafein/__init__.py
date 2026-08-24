"""Public-transport routing with per-leg distance and emissions tracking."""

# Let separately installed portions (cafein.sampledata) resolve under
# this package even when cafein itself is an editable install, whose
# __path__ points at the source tree rather than site-packages.
__path__ = __import__("pkgutil").extend_path(__path__, __name__)

__all__ = [
    "TransportNetwork",
    "StreetNetwork",
    "Exposure",
    "StreetLegPolicy",
    "VehiclePolicy",
    "TravelerProfile",
    "TravelCostMatrix",
    "TravelTimeMatrix",
    "Accessibility",
    "NearestDestinations",
    "Catchment",
    "DetailedItineraries",
    "travel_cost_table",
    "StreamingResult",
    "enable_logging",
    "disable_logging",
    "collect_timings",
    "to_minutes",
    "exhaustive_frontier",
    "journey_frontier",
    "journey_frontiers",
    "frontier_table",
    "fare_frontier",
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
    if name == "StreetNetwork":
        from cafein.street_network import StreetNetwork

        return StreetNetwork
    if name == "Exposure":
        from cafein.exposure import Exposure

        return Exposure
    if name in ("StreetLegPolicy", "VehiclePolicy"):
        from cafein import policy

        return getattr(policy, name)
    if name == "TravelerProfile":
        from cafein.travelers import TravelerProfile

        return TravelerProfile
    if name == "TravelCostMatrix":
        from cafein.matrices import TravelCostMatrix

        return TravelCostMatrix
    if name in ("Accessibility", "NearestDestinations", "Catchment"):
        from cafein import accessibility

        return getattr(accessibility, name)
    if name == "TravelTimeMatrix":
        from cafein.matrices import TravelTimeMatrix

        return TravelTimeMatrix
    if name == "DetailedItineraries":
        from cafein.itineraries import DetailedItineraries

        return DetailedItineraries
    if name == "to_minutes":
        from cafein.units import to_minutes

        return to_minutes
    if name == "travel_cost_table":
        from cafein.matrices import travel_cost_table

        return travel_cost_table
    if name == "StreamingResult":
        from cafein._streaming import StreamingResult

        return StreamingResult
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
    if name == "fare_frontier":
        from cafein.frontier import fare_frontier

        return fare_frontier
    if name == "least_emissions":
        from cafein.frontier import least_emissions

        return least_emissions
    if name == "least_fare":
        from cafein.frontier import least_fare

        return least_fare
    if name in ("enable_logging", "disable_logging", "collect_timings"):
        from cafein import _log

        return getattr(_log, name)
    if name == "__version__":
        from cafein._cafein import __version__

        return __version__
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
