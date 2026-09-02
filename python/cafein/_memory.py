"""The memory budget a computation plans against.

The budget is r5py's ``max_memory`` idea for a Rust core: it sizes
the work up front — the fan-out width, the result batch, a
rasterization strip — rather than capping the runtime (Rust has no
heap ceiling; allocations abort on failure, and hard guarantees stay
with the OS). A call reads the process's resident size once, reserves
its known allocations in a fixed order, and plans the width from what
remains; the only overruns are the documented floors, each warned.
"""

import contextlib
import os
import sys
import threading
import warnings

from cafein._units import memory_spec

#: r5py's floor: a resolved budget below it is raised to it, with a
#: warning.
MINIMUM_BUDGET = 200 * 1024**2
#: What a percentage leaves for the rest of the system.
LEAVE_AT_LEAST = 2 * 1024**3
#: The default spec: 80 % of physical memory.
DEFAULT_SPEC = "80%"
_lock = threading.Lock()
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

    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    if not ctypes.windll.psapi.GetProcessMemoryInfo(
        handle, ctypes.byref(counters), counters.cb
    ):
        raise OSError("GetProcessMemoryInfo failed")
    return int(counters.WorkingSetSize)


def resolve_budget(spec, stacklevel=3):
    """A spec (``None``: the process default) as bytes, floored at
    r5py's 200 MiB with a warning."""
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
