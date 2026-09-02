"""The memory budget a computation plans against.

A budget for a Rust core sizes the work up front rather than capping
the runtime: it sizes
the work up front — the fan-out width, the result batch, a
rasterization strip — rather than capping the runtime (Rust has no
heap ceiling; allocations abort on failure, and hard guarantees stay
with the OS). A call reads the process's resident size once, reserves
its known allocations in a fixed order, and plans the width from what
remains; the only overruns are the documented floors, each warned.
"""

import contextlib
import contextvars
import dataclasses
import logging
import os
import sys
import threading
import warnings

from cafein._units import memory_spec

logger = logging.getLogger("cafein")

#: The floor: a resolved budget below it is raised to it, with a
#: warning.
MINIMUM_BUDGET = 200 * 1024**2
#: What a percentage leaves for the rest of the system.
LEAVE_AT_LEAST = 2 * 1024**3
#: The default spec: 80 % of physical memory.
DEFAULT_SPEC = "80%"
#: The per-origin estimate is never below this; the calibration script
#: records the floor when a measured slope falls under it.
MIN_PER_ORIGIN = 64 * 1024
#: The streamed batch never grows past this many rows.
MAX_BATCH_ROWS = 500

#: Calibrated constants (see scripts/calibrate_memory.py and the
#: report it writes to plans/benchmarks/): per engine, the per-origin
#: search state as ``BYTES_PER_UNIT * size + FIXED_BYTES``, and the
#: call's width-independent allocation as ``CALL_BYTES_PER_UNIT * size
#: + CALL_FIXED_BYTES``, where ``size`` is the engine's scaling
#: variable — transit stops, or street-graph vertices for ``street``.
#: The values below are the envelope-adjusted fits from the laptop run
#: recorded in the report; they are conservative, never exact.
ENGINES = ("time", "multicriteria", "fare", "street")
BYTES_PER_UNIT = {"time": 5951, "multicriteria": 5951, "fare": 5951, "street": 0}
FIXED_BYTES = {"time": 65536, "multicriteria": 148298, "fare": 148298, "street": 65536}
CALL_BYTES_PER_UNIT = {"time": 3659, "multicriteria": 3659, "fare": 3659, "street": 0}
CALL_FIXED_BYTES = {
    "time": 8592445,
    "multicriteria": 8592445,
    "fare": 8592445,
    "street": 956924,
}
#: One rasterized float32 burn cell with its shape masks.
BYTES_PER_CELL = 10
#: The mean WKB bytes of one geometry row, offsets included.
GEOMETRY_ROW_BYTES = 11122

_lock = threading.Lock()
#: The plan of the enclosing public call, for the dispatches inside it.
_active = contextvars.ContextVar("cafein_active_plan", default=None)
_default = DEFAULT_SPEC


def set_max_memory(value):
    """Set the process default budget (``None`` restores ``"80%"``).

    The spec is validated for syntax and stored unresolved; physical
    memory is read when a call plans, never here.
    """
    global _default
    spec = memory_spec("max_memory", value)
    with _lock:
        _default = DEFAULT_SPEC if spec is None else value


def max_memory():
    """The process default budget spec, as set."""
    with _lock:
        return _default


def _psutil():
    """The psutil module when importable, else ``None``."""
    try:
        import psutil
    except ImportError:
        return None
    return psutil


def physical_memory():
    """The machine's physical memory in bytes, or ``None`` when no
    reader works (psutil when importable, else the platform's)."""
    psutil = _psutil()
    if psutil is not None:
        # A psutil that imports but cannot read here (a restricted or
        # proc-less environment) falls through to the platform readers.
        with contextlib.suppress(Exception):
            total = int(psutil.virtual_memory().total)
            if total > 0:
                return total
    if hasattr(os, "sysconf"):
        try:
            page, pages = os.sysconf("SC_PAGE_SIZE"), os.sysconf("SC_PHYS_PAGES")
            # sysconf answers -1 for an unknown value rather than raising.
            if page > 0 and pages > 0:
                return int(page * pages)
        except (ValueError, OSError):
            pass
    if sys.platform == "win32":
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_uint32),
                ("dwMemoryLoad", ctypes.c_uint32),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        status = Status()
        status.dwLength = ctypes.sizeof(Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
    return None


def resident_bytes():
    """The process's resident size: current with psutil, else the
    lifetime peak (a conservative stand-in), else ``None`` when no
    reader works."""
    psutil = _psutil()
    if psutil is not None:
        with contextlib.suppress(Exception):
            return int(psutil.Process().memory_info().rss)
    with contextlib.suppress(ImportError, AttributeError, OSError):
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    if sys.platform == "win32":
        with contextlib.suppress(Exception):
            return _windows_resident()
    return None


def _windows_resident():
    """The working-set size from ``GetProcessMemoryInfo``."""
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    # Typed calls: without them ctypes passes the 64-bit pseudo-handle
    # as a 32-bit int and the query fails.
    kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(Counters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    if not psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise OSError("GetProcessMemoryInfo failed")
    return int(counters.WorkingSetSize)


def resolve_budget(spec, stacklevel=3):
    """A spec (``None``: the process default) as bytes, floored at
    200 MiB with a warning."""
    return _resolve(spec, stacklevel + 1)[0]


def _resolve(spec, stacklevel):
    """``(budget, floored)``: the resolved bytes and whether the
    minimum applied."""
    parsed = memory_spec("max_memory", max_memory() if spec is None else spec)
    kind, number = parsed
    if kind == "percent":
        total = physical_memory()
        if total is None:
            raise ValueError(
                "max_memory is a percentage but the machine's memory size "
                "cannot be read here; give an absolute size such as '8G'"
            )
        share = number * total
        budget = int(min(share, total - LEAVE_AT_LEAST))
    else:
        budget = number
    if budget < MINIMUM_BUDGET:
        warnings.warn(
            f"max_memory resolves to {budget} bytes, below the "
            f"{MINIMUM_BUDGET}-byte minimum; planning at the minimum",
            UserWarning,
            stacklevel=stacklevel,
        )
        return MINIMUM_BUDGET, True
    return budget, False


def ambient_workers():
    """The width a fan-out rides without a plan: the core's pool."""
    try:
        from cafein import _cafein

        return int(_cafein._probe_workers(None))
    except Exception:
        return max(1, os.cpu_count() or 1)


@dataclasses.dataclass(frozen=True)
class Plan:
    """One call's memory plan: the headroom it saw, the batch rows it
    reserved (``None`` for an in-memory result), its fan-out width,
    the fare refinement's width when the call has that phase, and the
    exposure reporting snapshot the call was sized with, so the body
    reports against the same frozen surface."""

    headroom: int
    batch_rows: object
    width: int
    refinement_width: object = None
    exposure_snapshot: object = None


def _per_origin(engine, size):
    return max(MIN_PER_ORIGIN, BYTES_PER_UNIT[engine] * size + FIXED_BYTES[engine])


def _call_fixed(engine, size):
    return max(0, CALL_BYTES_PER_UNIT[engine] * size + CALL_FIXED_BYTES[engine])


def _warn(message, stacklevel):
    warnings.warn(message, UserWarning, stacklevel=stacklevel)


def plan_call(
    engine,
    size,
    result_bytes,
    *,
    workers=None,
    max_memory=None,
    label="computation",
    streamed=False,
    row_bytes=None,
    batch_rows=None,
    refinement_engine=None,
    stacklevel=3,
):
    """Plan one call against the budget.

    ``engine`` names the per-origin search-state class (``time``,
    ``multicriteria``, ``fare``, ``street``) and ``size`` its scaling
    variable. ``result_bytes`` is the in-memory result's (or the
    aggregating consumer's intermediate surface's) estimate; a
    streamed call passes ``streamed=True`` with ``row_bytes`` (one
    result row) and ``batch_rows`` (an explicit ``batch_size=``, or
    ``None`` to derive one). ``refinement_engine`` names the fare
    arm's second, sequential phase. ``workers`` and ``max_memory`` are
    the call's explicit knobs; the budget spec ``None`` means the
    process default.
    """
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r}; the engines are {ENGINES}")
    explicit_budget = max_memory is not None
    budget, floored = _resolve(max_memory, stacklevel + 1)
    baseline = resident_bytes()
    if baseline is None:
        raise ValueError(
            f"{label}: the process's resident size cannot be read here, so "
            "nothing can be planned against max_memory; install psutil"
        )
    headroom = max(0, budget - baseline)
    ambient = ambient_workers()
    fixed = _call_fixed(engine, size)
    phases = [(engine, fixed)]
    if refinement_engine is not None:
        phases.append((refinement_engine, _call_fixed(refinement_engine, size)))
    largest_fixed = max(fixed_bytes for _, fixed_bytes in phases)

    floors = ["the minimum budget"] if floored else []
    decisions = []

    def overrun(what):
        floors.append(what)
        _warn(
            f"{label}: {what} exceeds the memory budget ({budget} bytes, "
            f"{baseline} resident, {headroom} headroom); proceeding at the "
            "floor — raise max_memory= to plan wider",
            stacklevel + 1,
        )

    # The result (or batch) reservation, sized against the phase that
    # leaves the least room.
    after_fixed = headroom - largest_fixed
    if streamed:
        if after_fixed <= 0:
            raise ValueError(
                f"{label}: the call's fixed allocation ({largest_fixed} bytes) "
                f"alone exceeds the memory budget's headroom ({headroom} bytes "
                f"of {budget}, {baseline} resident); raise max_memory="
            )
        per_row = max(1, int(row_bytes or 1))
        if batch_rows is None:
            rows = min(MAX_BATCH_ROWS, (after_fixed // 2) // per_row)
            if rows < 1:
                rows = 1
                overrun(f"one batch row ({per_row} bytes)")
        else:
            rows = int(batch_rows)
            if rows * per_row > after_fixed:
                overrun(f"the explicit batch of {rows} rows ({rows * per_row} bytes)")
        reserved = rows * per_row
        batch = rows
    else:
        if largest_fixed + result_bytes > headroom:
            raise ValueError(
                f"{label}: the result ({result_bytes} bytes) and the call's "
                f"fixed allocation ({largest_fixed} bytes) exceed the memory "
                f"budget's headroom ({headroom} bytes of {budget}, {baseline} "
                "resident); narrow the query or raise max_memory="
            )
        reserved = result_bytes
        batch = None

    def width_for(phase_engine, phase_fixed):
        remaining = headroom - phase_fixed - reserved
        per_origin = _per_origin(phase_engine, size)
        planned = remaining // per_origin
        if planned < 1:
            planned = 1
            overrun(f"one {phase_engine} search ({per_origin} bytes)")
        planned = min(planned, ambient)
        if workers is not None:
            if workers <= planned:
                return workers
            if explicit_budget:
                decisions.append(f"workers={workers} narrowed to {planned}")
                return planned
            overrun(f"workers={workers} (the budget plans {planned})")
            return workers
        return planned

    width = width_for(*phases[0])
    refinement = width_for(*phases[1]) if len(phases) > 1 else None
    logger.debug(
        "planned %d workers for %s: %d headroom, %d reserved, %d per origin; "
        "floors: %s; %s",
        width,
        label,
        headroom,
        reserved,
        _per_origin(engine, size),
        ", ".join(floors) or "none",
        "; ".join(decisions) or "no narrowing",
    )
    return Plan(
        headroom=headroom, batch_rows=batch, width=width, refinement_width=refinement
    )


class Refusal:
    """A plan that could not be made: the call's own argument checks
    run first, and the refusal fires at the first dispatch, before any
    search allocates."""

    def __init__(self, error):
        self.error = error


def active_plan():
    """The plan the enclosing public call made, or ``None``."""
    return _active.get()


@contextlib.contextmanager
def use_plan(plan):
    """Make ``plan`` the active one for the dispatches inside."""
    token = _active.set(plan)
    try:
        yield plan
    finally:
        _active.reset(token)


def width_or(workers):
    """The active plan's width, else ``workers`` as given; a deferred
    refusal fires here."""
    plan = _active.get()
    if isinstance(plan, Refusal):
        raise plan.error
    return workers if plan is None else plan.width
