"""Public-transport routing with per-leg distance and emissions tracking."""

__all__ = ["TransportNetwork", "__version__"]


def __getattr__(name):
    # Resolved lazily so that the pure-Python modules (cafein.streets)
    # stay importable without the compiled core.
    if name == "TransportNetwork":
        from cafein.network import TransportNetwork

        return TransportNetwork
    if name == "__version__":
        from cafein._cafein import __version__

        return __version__
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
