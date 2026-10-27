"""Process and timing visibility: the ``cafein`` logging surface.

No log lines by default: the ``"cafein"`` logger carries a
``NullHandler`` and propagation stays on, so nothing is logged until
the user calls :func:`enable_logging` or configures the stdlib
``logging`` machinery directly. Progress shows by default: a long
operation draws a bar on stderr once it has run two seconds (see
:func:`enable_logging`). Phase completions additionally carry
structured attributes (``cafein_phase``, ``cafein_seconds``,
``cafein_details``) and are delivered to any active
:func:`collect_timings` collector independently of the visible
logging configuration.

Logging must never break a computation: every record is delivered
through the exception-guarded :func:`_emit`.
"""

import copy
import functools
import importlib.util
import inspect
import logging
import os
import sys
import threading
import time
import warnings
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


# The package's own source files, which warn() steps out of.
_PACKAGE_PREFIX = os.path.dirname(os.path.abspath(__file__)) + os.sep


@functools.lru_cache(maxsize=256)
def _in_package(filename):
    return os.path.abspath(filename).startswith(_PACKAGE_PREFIX)


def warn(message, category=UserWarning):
    """Warn at the first caller outside the cafein package.

    A fixed ``stacklevel`` lands inside cafein whenever the internal
    call depth changes; stepping over cafein's own frames keeps the
    warning on the line of user code that made the call.
    """
    try:
        frame = sys._getframe(1)
    except ValueError:
        frame = None
    level = 2
    while frame is not None and _in_package(frame.f_code.co_filename):
        frame = frame.f_back
        level += 1
    warnings.warn(message, category, stacklevel=level)


def _emit(
    logger, level, message, *, phase=None, seconds=None, details=None, progress=None
):
    """Deliver one record; logging must never break a computation."""
    try:
        if phase is None:
            if progress is not None:
                label, done, total = progress
                logger.log(
                    level,
                    message,
                    extra={
                        "cafein_progress": {
                            "label": label,
                            "done": done,
                            "total": total,
                        }
                    },
                )
            else:
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
    into the bridge's range, and whether progress is drawn. A no-op
    until the bridge exists.
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
        _cafein.set_progress(_presentation()[0])
    except Exception:
        pass


def _from_rust(
    target,
    levelno,
    message,
    phase=None,
    seconds=None,
    progress=None,
    event=None,
    run=None,
):
    """The dispatch the PyO3 bridge calls for every Rust-side record,
    and for the ``"start"``/``"end"`` events of a fan-out's progress,
    which are drawn but never logged. ``run`` is the fan-out's id."""
    if progress is not None:
        _present(event or "tick", ("rust", run), *progress)
        if event is not None:
            return
    logger = logging.getLogger(target)
    if phase is None:
        _emit(logger, levelno, message, progress=progress)
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


# A bar shows once its operation has run this long; without a terminal,
# an operation that ran this long prints one line with its time instead.
_DELAY = 2.0

# Progress without enable_logging; cafein's test suite turns it off.
_default_progress = True


def _bar_class(stream):
    """``(tqdm class, disable)`` for a bar on ``stream`` (None: stderr).

    On stderr in a Jupyter kernel it is the widget bar when ipywidgets
    is installed, else the text bar, which the notebook redraws in
    place; both show although the kernel's stderr is no terminal.
    Otherwise it is the text bar, hidden unless the stream says it is a
    terminal.
    """
    from tqdm import std

    if stream is None or stream is sys.stderr:
        # A kernel has imported IPython already, so it is looked up,
        # not imported.
        ipython = sys.modules.get("IPython")
        shell = getattr(ipython, "get_ipython", lambda: None)()
        if type(shell).__name__ == "ZMQInteractiveShell":
            if importlib.util.find_spec("ipywidgets") is None:
                return std.tqdm, False
            from tqdm import notebook

            return notebook.tqdm, False
        stream = sys.stderr
    try:
        terminal = bool(stream.isatty())
    except Exception:
        terminal = False
    return std.tqdm, not terminal


def _default_bar_factory(label, total, stream):
    """One operation's bar, shown once it has run ``_DELAY`` seconds and
    cleared when it closes: on ``stream`` for
    ``enable_logging(progress="bar")``, or with ``stream`` None on
    stderr (see :func:`_bar_class`)."""
    tqdm, disable = _bar_class(stream)
    return tqdm(
        total=total,
        desc=label,
        file=sys.stderr if stream is None else stream,
        delay=_DELAY,
        leave=False,
        mininterval=0.2,
        disable=disable,
    )


_bar_factory = _default_bar_factory


class _Run:
    """The bar of one running operation, with its label, stream and
    start time."""

    def __init__(self, bar, label, stream):
        self.bar = bar
        self.label = label
        self.stream = sys.stderr if stream is None else stream
        self.started = time.monotonic()
        # Read now: tqdm marks every bar disabled once it closes.
        self.hidden = bool(getattr(bar, "disable", False))

    def close(self):
        """Close the bar; a hidden one that ran ``_DELAY`` seconds or
        more writes one line with the time it took to its stream."""
        self.bar.close()
        elapsed = time.monotonic() - self.started
        if self.hidden and elapsed >= _DELAY:
            took = self.bar.format_interval(elapsed)
            self.bar.write(f"{self.label} took {took}", file=self.stream)


_runs = {}
_runs_lock = threading.RLock()


def _drop_runs():
    """Close every open bar without its time line: progress was
    reconfigured, and a new start follows the new setting."""
    with _runs_lock:
        runs = list(_runs.values())
        _runs.clear()
        for run in runs:
            try:
                run.bar.close()
            except Exception:
                pass


def _presentation():
    """``(on, stream)``: whether progress is drawn now, and on which
    stream — None for the default bars on stderr."""
    handler = _handler
    if handler is None:
        return _default_progress, None
    return handler._progress == "bar", handler.stream


def _present(event, key, label, done, total):
    """Draw one progress event of the operation run ``key``, shown as
    ``label``: ``"start"`` opens its bar when progress is drawn,
    ``"tick"`` moves an open bar to ``done``, ``"end"`` closes it.
    Never raises."""
    try:
        with _runs_lock:
            if event == "start":
                on, stream = _presentation()
                if on:
                    bar = _bar_factory(label, total, stream)
                    _runs[key] = _Run(bar, label, stream)
            elif event == "tick":
                run = _runs.get(key)
                if run is not None and done > run.bar.n:
                    run.bar.update(done - run.bar.n)
            else:
                run = _runs.pop(key, None)
                if run is not None:
                    run.close()
    except Exception:
        pass


@contextmanager
def reporting(label, logger):
    """Progress of a pyrosm read, shown as cafein's own.

    Yields pyrosm's ``progress`` callable, called as
    ``report(done, 100)``: it moves the bar of ``label`` and, with
    ``logger`` enabled for INFO, logs a line whenever the read passes
    a further 5 %. Yields
    ``False``, which turns pyrosm's own bar off too, when neither
    shows. The bar closes on exit.
    """
    if not (_presentation()[0] or logger.isEnabledFor(logging.INFO)):
        yield False
        return
    key = object()
    started = time.perf_counter()
    logged = 0

    def report(done, total):
        nonlocal logged
        try:
            _present("tick", key, label, done, total)
            percent = int(done * 100 // total)
            if percent >= logged + 5:
                logged = percent - percent % 5
                seconds = time.perf_counter() - started
                _emit(
                    logger,
                    logging.INFO,
                    f"{label} {logged}% ({seconds:.1f} s elapsed)",
                    progress=(label, logged, 100),
                )
        except Exception:
            pass

    _present("start", key, label, 0, 100)
    try:
        yield report
    finally:
        _present("end", key, label, 0, 100)


class _CafeinHandler(logging.StreamHandler):
    """The handler ``enable_logging`` installs: phase and message
    lines to the stream, and tick records as lines when the progress
    mode is "lines" ("bar" draws them as bars instead, False drops
    them). Only cafein's own handler is shaped — the records reach
    every other handler identically."""

    def __init__(self, stream, progress):
        super().__init__(stream)
        self._progress = progress

    def emit(self, record):
        """Write the record, unless it is a tick and progress is not
        shown as lines."""
        if getattr(record, "cafein_progress", None) is None or (
            self._progress == "lines"
        ):
            super().emit(record)


def _validated_progress(progress):
    if progress is False:
        return False
    if isinstance(progress, str) and not isinstance(progress, bool):
        if progress in ("lines", "bar"):
            return progress
        raise ValueError("progress must be 'lines', 'bar', or False")
    raise TypeError("progress must be 'lines', 'bar', or False")


def _validated_level(level):
    if isinstance(level, bool) or not isinstance(level, (int, str)):
        raise TypeError("level must be an int or one of 'debug', 'info', 'warning'")
    if isinstance(level, int):
        return level
    name = level.lower()
    if name not in _LEVELS:
        raise ValueError("level must be one of 'debug', 'info', 'warning'")
    return _LEVELS[name]


def enable_logging(stream=None, level="info", progress="lines"):
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
        "info" (default) shows phases and progress lines; "debug"
        adds phase starting lines.
    progress : {"lines", "bar", False}
        How progress shows while this handler is installed: "lines"
        (default) logs a line at each 5 % of a long operation, "bar"
        draws one bar per operation on ``stream``, and ``False``
        shows none. Without a call to this function, progress draws
        as bars on stderr. A bar shows once its operation has run two
        seconds (a widget inside Jupyter); where the stream is no
        terminal, an operation that took two seconds or more writes
        one line with its time instead.
        ``enable_logging(level="warning", progress=False)`` turns
        progress off without logging anything. Ticks are logged
        identically in every mode, so handlers you attach yourself
        see them at INFO (they carry a ``cafein_progress`` attribute
        with ``label``/``done``/``total``); the bars of pyrosm's OSM
        reads follow this setting too.
    """
    global _handler, _prior_level
    if stream is None:
        stream = sys.stderr
    if not callable(getattr(stream, "write", None)):
        raise TypeError(
            "stream must be a writable text stream (an object with a write method)"
        )
    levelno = _validated_level(level)
    progress = _validated_progress(progress)
    with _lock:
        if _handler is None:
            _prior_level = root.level
        else:
            root.removeHandler(_handler)
            _handler.close()
        handler = _CafeinHandler(stream, progress)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
        )
        root.addHandler(handler)
        root.setLevel(levelno)
        _handler = handler
    _drop_runs()
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
        _handler.close()
        root.setLevel(_prior_level)
        _handler = None
        _prior_level = None
    _drop_runs()
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
    ``build.multimodal`` (the multimodal street graph) with its
    children ``build.multimodal.streets``,
    ``build.multimodal.elevation``, and ``build.multimodal.speeds``;
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


# Arms the bridge for the default progress.
sync()
