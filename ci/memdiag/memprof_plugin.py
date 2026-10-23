"""pytest plugin: each process's Linux memory breakdown after every test.

``-p memprof_plugin`` with ``MEMPROF_DIR``; one CSV per process. Columns
come from /proc/self/status (VmRSS, RssAnon, RssFile, RssShmem, VmSwap) and
/proc/self/smaps_rollup (Private_Clean + Private_Dirty as ``private``), all
in MB, read after a gc.collect().
"""

import csv
import gc
import os

_FIELDS = ["VmRSS", "RssAnon", "RssFile", "RssShmem", "VmSwap"]
_writer = None
_file = None
_seq = 0


def _read_kb_fields(path, keys):
    """The kB values of ``keys`` in a /proc file of ``name: value kB`` lines."""
    values = dict.fromkeys(keys, 0)
    try:
        with open(path) as handle:
            for line in handle:
                name, _, rest = line.partition(":")
                if name in values:
                    values[name] = int(rest.split()[0])
    except OSError:
        pass
    return values


def _memory_row(phase, nodeid):
    """One CSV row: the phase, the test and the memory columns in MB."""
    status = _read_kb_fields("/proc/self/status", _FIELDS)
    rollup = _read_kb_fields("/proc/self/smaps_rollup", ["Private_Clean", "Private_Dirty"])
    private = rollup["Private_Clean"] + rollup["Private_Dirty"]
    return [phase, nodeid] + [round(status[k] / 1024, 1) for k in _FIELDS] + [
        round(private / 1024, 1)
    ]


def _record_memory(phase, nodeid):
    """Append this process's memory row to its CSV, opening the file first."""
    global _writer, _file, _seq
    gc.collect()
    if _writer is None:
        out = os.environ["MEMPROF_DIR"]
        os.makedirs(out, exist_ok=True)
        worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
        _file = open(os.path.join(out, f"{worker}.csv"), "w", newline="")
        _writer = csv.writer(_file)
        _writer.writerow(
            ["seq", "phase", "nodeid", "rss", "anon", "file", "shmem", "swap", "private"]
        )
    _seq += 1
    _writer.writerow([_seq] + _memory_row(phase, nodeid))
    _file.flush()


def pytest_sessionstart(session):
    if os.environ.get("PYTEST_XDIST_WORKER"):
        _record_memory("start", "")


def pytest_runtest_logfinish(nodeid, location):
    if os.environ.get("PYTEST_XDIST_WORKER"):
        _record_memory("after", nodeid)
