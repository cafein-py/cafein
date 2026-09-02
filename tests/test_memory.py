"""The memory budget: the spec grammar, the process default, and the
call planner's arithmetic."""

import logging
import threading

import pytest
import sys

from cafein import _memory
from cafein._units import memory_spec

GIB = 1024**3
MIB = 1024**2


@pytest.fixture(autouse=True)
def _pristine_default():
    _memory.set_max_memory(None)
    yield
    _memory.set_max_memory(None)


def test_memory_spec_grammar():
    for value, expected in (
        ("80%", ("percent", 0.8)),
        (" 50 % ", ("percent", 0.5)),
        ("8G", ("bytes", 8 * GIB)),
        ("8g", ("bytes", 8 * GIB)),
        ("512MiB", ("bytes", 512 * MIB)),
        ("1.5K", ("bytes", 1536)),
        ("1T", ("bytes", 1024**4)),
        ("4096", ("bytes", 4096)),
        (4096, ("bytes", 4096)),
        (None, None),
    ):
        assert memory_spec("max_memory", value) == expected, value
    for value, error, match in (
        (True, TypeError, "bool"),
        (-1, ValueError, "non-negative"),
        ("101%", ValueError, "percentage"),
        ("0%", ValueError, "percentage"),
        ("8X", ValueError, "could not read"),
        ("lots", ValueError, "could not read"),
        ([8], TypeError, "string"),
    ):
        with pytest.raises(error, match=match):
            memory_spec("max_memory", value)

    # Exact integer arithmetic: no float rounding, no overflow.
    assert memory_spec("m", "9007199254740993") == ("bytes", 9007199254740993)
    assert memory_spec("m", 10**30) == ("bytes", 10**30)
    assert memory_spec("m", str(10**30)) == ("bytes", 10**30)
    assert memory_spec("m", "1.5G") == ("bytes", 1610612736)
    assert memory_spec("m", "2Y") == ("bytes", 2 * 1024**8)


def test_the_process_default_is_set_read_and_reset(monkeypatch):
    assert _memory.max_memory() == "80%"
    _memory.set_max_memory("8G")
    assert _memory.max_memory() == "8G"
    with pytest.raises(ValueError, match="could not read"):
        _memory.set_max_memory("8X")
    assert _memory.max_memory() == "8G"
    # A percentage validates without reading the machine; an
    # unreadable machine surfaces at planning time, by name.
    monkeypatch.setattr(_memory, "physical_memory", lambda: None)
    _memory.set_max_memory("50%")
    with pytest.raises(ValueError, match="absolute size"):
        _memory.resolve_budget(None)
    _memory.set_max_memory(None)
    assert _memory.max_memory() == "80%"
    # Concurrent setters never corrupt the stored spec.
    barrier = threading.Barrier(4)

    def setter(value):
        barrier.wait()
        for _ in range(50):
            _memory.set_max_memory(value)

    threads = [threading.Thread(target=setter, args=(f"{n}G",)) for n in (1, 2, 3, 4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert _memory.max_memory() in {"1G", "2G", "3G", "4G"}


def test_resolve_budget_shares_and_floors(monkeypatch):
    monkeypatch.setattr(_memory, "physical_memory", lambda: 16 * GIB)
    assert _memory.resolve_budget("50%") == 8 * GIB
    # Percentages leave 2 GiB for the system.
    assert _memory.resolve_budget("95%") == 14 * GIB
    assert _memory.resolve_budget("4G") == 4 * GIB
    with pytest.warns(UserWarning, match="below the"):
        assert _memory.resolve_budget("1M") == _memory.MINIMUM_BUDGET


def test_readers_fall_back_when_psutil_cannot_read(monkeypatch):
    # A psutil that imports but fails to read yields to the platform
    # readers, which every CI platform has.
    class Broken:
        class Error(Exception):
            pass

        def virtual_memory(self):
            raise self.Error("no /proc")

        def Process(self):
            raise self.Error("no /proc")

    monkeypatch.setitem(sys.modules, "psutil", Broken())
    assert _memory.physical_memory() > 0
    assert _memory.resident_bytes() > 0
    # Without psutil or resource, Windows reads its working set.
    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setitem(sys.modules, "resource", None)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(_memory, "_windows_resident", lambda: 123)
    assert _memory.resident_bytes() == 123

    def failing():
        raise OSError("no psapi")

    monkeypatch.setattr(_memory, "_windows_resident", failing)
    assert _memory.resident_bytes() is None
