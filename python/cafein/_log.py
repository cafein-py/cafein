"""Process and timing visibility: the ``cafein`` logging surface.

Quiet by default: the ``"cafein"`` logger carries a ``NullHandler``
and propagation stays on, so nothing is printed until the user calls
:func:`enable_logging` or configures the stdlib ``logging`` machinery
directly. Phase completions additionally carry structured attributes
(``cafein_phase``, ``cafein_seconds``, ``cafein_details``) and are
delivered to any active :func:`collect_timings` collector
independently of the visible logging configuration.

Logging must never break a computation: every record is delivered
through the exception-guarded :func:`_emit`.
"""

import copy
import functools
import inspect
import logging
import sys
import threading
import time
from contextlib import contextmanager

root = logging.getLogger("cafein")
build = logging.getLogger("cafein.build")
matrix = logging.getLogger("cafein.matrix")
artifact = logging.getLogger("cafein.artifact")
emissions = logging.getLogger("cafein.emissions")
exposure = logging.getLogger("cafein.exposure")

root.addHandler(logging.NullHandler())

# The namespaces the Rust bridge emits to; sync() arms the bridge
# from the minimum of their effective levels. The disarmed sentinel
# sits above every Python level and is never produced by sync().
_RUST_TARGETS = ("cafein.build", "cafein.matrix", "cafein.artifact")
_DISARMED = 2**32 - 1

_LEVELS = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING}

_lock = threading.Lock()
_collectors = []
_handler = None
_prior_level = None


class TimingReport:
    """Structured phase timings collected by :func:`collect_timings`.

    ``phases`` is a list of ``{"phase", "seconds", "details"}`` dicts
    in completion order; ``frame()`` returns the same as a pandas
    DataFrame.
    """

    def __init__(self):
        self.phases = []

    def frame(self):
        import pandas as pd

        return pd.DataFrame(
            {
                "phase": [entry["phase"] for entry in self.phases],
                "seconds": [entry["seconds"] for entry in self.phases],
                "details": [entry["details"] for entry in self.phases],
            }
        )


def _emit(logger, level, message, *, phase=None, seconds=None, details=None):
    """Deliver one record; logging must never break a computation."""
    try:
        if phase is None:
            logger.log(level, message)
            return
        # Every delivery gets a recursively independent snapshot, so a
        # handler mutating its record — or a consumer mutating a report
        # entry — cannot reach the caller's dict or another delivery.
        for report in tuple(_collectors):
            report.phases.append(
                {
                    "phase": phase,
                    "seconds": seconds,
                    "details": copy.deepcopy(dict(details or {})),
                }
            )
        logger.log(
            level,
            message,
            extra={
                "cafein_phase": phase,
                "cafein_seconds": seconds,
                "cafein_details": copy.deepcopy(dict(details or {})),
            },
        )
    except Exception:
        pass


class _Phase:
    """Mutable handle a ``phase`` block fills in as facts appear."""

    def __init__(self):
        self.details = {}
        self.note = None


@contextmanager
def phase(identifier, logger, doing, done):
    """Time one phase: a DEBUG line on entry, an INFO completion with
    the structured attributes on exit. An exception inside the block
    emits no completion — the raised error is the report."""
    handle = _Phase()
    _emit(logger, logging.DEBUG, doing)
    started = time.perf_counter()
    yield handle
    seconds = time.perf_counter() - started
    message = f"{done} in {seconds:.1f} s"
    if handle.note:
        message += f" ({handle.note})"
    _emit(
        logger,
        logging.INFO,
        message,
        phase=identifier,
        seconds=seconds,
        details=handle.details,
    )


def sized(value):
    """``len(value)`` where it exists, else None — best-effort counts
    for phase details."""
    try:
        return len(value)
    except TypeError:
        return None


def timed_computer(
    identifier,
    logger,
    doing,
    done,
    *,
    street_identifier=None,
    street_doing=None,
    street_done=None,
    is_street=None,
    details=None,
):
    """Wrap a computer entry point in its ``matrix.*`` phase.

    Works on frame-building ``__init__`` methods (the row count is
    read off ``self``) and on frame-returning functions alike. With
    ``street_identifier`` and ``is_street``, a street-network call
    swaps to the street identifier and phrases. ``details`` is an
    optional ``(arguments) -> dict`` over the call's bound arguments
    (positional and keyword forms alike), read before the computation.
    """

    def wrap(fn):
        signature = inspect.signature(fn)

        def bound(args, kwargs):
            try:
                return signature.bind_partial(*args, **kwargs).arguments
            except TypeError:
                return {}

        @functools.wraps(fn)
        def timed(*args, **kwargs):
            ident, doing_now, done_now = identifier, doing, done
            arguments = bound(args, kwargs)
            if street_identifier is not None and is_street is not None:
                if is_street(arguments.get("network")):
                    ident = street_identifier
                    doing_now = street_doing or doing
                    done_now = street_done or done
            sync()
            with phase(ident, logger, doing_now, done_now) as handle:
                if details is not None:
                    try:
                        handle.details.update(details(arguments))
                    except Exception:
                        pass
                result = fn(*args, **kwargs)
                rows = sized(result if result is not None else args[0])
                if rows is not None:
                    handle.details["rows"] = rows
                    handle.note = f"{rows:,} rows"
                return result

        return timed

    return wrap


def sync():
    """Arm the Rust bridge from the current logging configuration.

    Passes the minimum effective level over the Rust target
    namespaces, capped at INFO while a collector is active, clamped
    into the bridge's range. A no-op until the bridge exists.
    """
    try:
        from cafein import _cafein

        set_level = getattr(_cafein, "set_log_level", None)
        if set_level is None:
            return
        threshold = min(
            logging.getLogger(target).getEffectiveLevel() for target in _RUST_TARGETS
        )
        if _collectors:
            threshold = min(threshold, logging.INFO)
        set_level(max(0, min(int(threshold), _DISARMED - 1)))
    except Exception:
        pass


def _from_rust(target, levelno, message, phase=None, seconds=None):
    """The dispatch the PyO3 bridge calls for every Rust-side record."""
    logger = logging.getLogger(target)
    if phase is None:
        _emit(logger, levelno, message)
    else:
        _emit(logger, levelno, message, phase=phase, seconds=seconds, details={})


def _install_bridge():
    """Register the dispatch with the compiled bridge, if present."""
    try:
        from cafein import _cafein

        install = getattr(_cafein, "install_log_dispatch", None)
        if install is not None:
            install(_from_rust)
    except Exception:
        pass


_install_bridge()


def _validated_level(level):
    if isinstance(level, bool) or not isinstance(level, (int, str)):
        raise TypeError("level must be an int or one of 'debug', 'info', 'warning'")
    if isinstance(level, int):
        return level
    name = level.lower()
    if name not in _LEVELS:
        raise ValueError("level must be one of 'debug', 'info', 'warning'")
    return _LEVELS[name]


def enable_logging(stream=None, level="info"):
    """Log cafein's phases and timings to a stream.

    Installs a handler on the ``"cafein"`` logger writing to
    ``stream`` (default ``sys.stderr``) and sets the logger's level.
    Calling it again replaces the previously installed handler, never
    duplicating lines; the logger level the user had set before the
    first call is restored by :func:`disable_logging`. With a root
    logger of your own configured, lines can appear twice — set
    ``logging.getLogger("cafein").propagate = False`` to keep them
    out of the root handlers.

    Parameters
    ----------
    stream : writable text stream, optional
        Any object with a ``write`` method; ``sys.stderr`` when None.
    level : int or {"debug", "info", "warning"}
        "info" (default) shows phases and progress; "debug" adds
        phase starting lines.
    """
    global _handler, _prior_level
    if stream is None:
        stream = sys.stderr
    if not callable(getattr(stream, "write", None)):
        raise TypeError(
            "stream must be a writable text stream (an object with a write method)"
        )
    levelno = _validated_level(level)
    with _lock:
        if _handler is None:
            _prior_level = root.level
        else:
            root.removeHandler(_handler)
        handler = logging.StreamHandler(stream)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
        )
        root.addHandler(handler)
        root.setLevel(levelno)
        _handler = handler
    sync()


def disable_logging():
    """Undo :func:`enable_logging`: remove cafein's own handler and
    restore the logger level the user had set before the first
    ``enable_logging`` call. Handlers the user attached are left
    alone; a no-op when ``enable_logging`` was never called."""
    global _handler, _prior_level
    with _lock:
        if _handler is None:
            return
        root.removeHandler(_handler)
        root.setLevel(_prior_level)
        _handler = None
        _prior_level = None
    sync()


@contextmanager
def collect_timings():
    """Collect structured phase timings, yielding a `TimingReport`.

    Collection is independent of the visible logging configuration:
    no handler is attached and no logger level changes, so streams
    configured by :func:`enable_logging` (or plain ``logging``) keep
    printing exactly what they printed before, while every phase
    completed inside the block lands in the report. The collector
    registry is process-global.

    Phase identifiers are dotted paths; a parent phase is an
    aggregate that coexists with its dotted children. The current
    vocabulary: ``build.gtfs`` (the transit core built from GTFS)
    with its children ``build.gtfs.read``, ``build.gtfs.timetable``,
    and ``build.gtfs.indexes``; ``build.streets.read``,
    ``build.streets.prune``, ``build.streets.graph``,
    ``build.streets.footpaths`` (the OSM walking structures);
    ``build.multimodal`` (the multimodal street graph);
    ``artifact.save`` with ``artifact.save.encode``;
    ``artifact.load`` with ``artifact.load.decode`` and
    ``artifact.load.rebuild``; ``emissions.annotate``;
    ``exposure.build``; ``build.tbtr`` and ``build.mctbtr`` (the
    cached engine precomputes); and one identifier per computer,
    shared by its related entry points — ``matrix.travel_times``
    (``TravelTimeMatrix`` and the ``travel_times_*`` surfaces),
    ``matrix.travel_costs`` (``TravelCostMatrix``),
    ``matrix.cost_table`` (``travel_cost_table``),
    ``matrix.itineraries`` (``DetailedItineraries``),
    ``matrix.accessibility`` (``Accessibility``), and
    ``matrix.streets`` (the matrix surfaces over a street network).
    Progress ticks emitted during matrix fan-outs are plain INFO
    lines, not phases, and never enter a report.
    """
    report = TimingReport()
    with _lock:
        _collectors.append(report)
    sync()
    try:
        yield report
    finally:
        with _lock:
            _collectors.remove(report)
        sync()
