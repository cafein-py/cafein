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


@pytest.fixture
def machine(monkeypatch):
    """A 16 GiB machine with 4 GiB resident and 8 ambient workers,
    with tiny fixed allocations so the arithmetic stays legible."""
    monkeypatch.setattr(_memory, "physical_memory", lambda: 16 * GIB)
    monkeypatch.setattr(_memory, "resident_bytes", lambda: 4 * GIB)
    monkeypatch.setattr(_memory, "ambient_workers", lambda: 8)
    monkeypatch.setattr(
        _memory, "CALL_BYTES_PER_UNIT", dict.fromkeys(_memory.ENGINES, 0)
    )
    monkeypatch.setattr(
        _memory, "CALL_FIXED_BYTES", dict.fromkeys(_memory.ENGINES, MIB)
    )
    monkeypatch.setattr(_memory, "FIXED_BYTES", dict.fromkeys(_memory.ENGINES, 0))
    monkeypatch.setattr(
        _memory,
        "BYTES_PER_UNIT",
        {"time": 100, "multicriteria": 1000, "fare": 12_000, "street": 50},
    )


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


def test_plan_call_reserves_then_plans_the_width(machine, caplog, monkeypatch):
    # The default 80 % of 16 GiB less 4 GiB resident is the headroom;
    # the time engine at 100 bytes/stop over 10_000 stops needs 1 MB
    # per origin, so the width is the ambient cap, not the budget.
    headroom = int(0.8 * 16 * GIB) - 4 * GIB
    with caplog.at_level(logging.DEBUG, logger="cafein"):
        plan = _memory.plan_call("time", 10_000, 2 * GIB, label="matrix")
    assert plan == _memory.Plan(headroom=headroom, batch_rows=None, width=8)
    assert any("planned 8 workers for matrix" in r.getMessage() for r in caplog.records)
    # One DEBUG record per call, the narrowing decision inside it.
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="cafein"):
        _memory.plan_call(
            "fare",
            10_000,
            256 * MIB,
            workers=8,
            max_memory="5G",
            refinement_engine="fare",
        )
    records = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert len(records) == 1 and "workers=8 narrowed" in records[0].getMessage()
    # The minimum-budget floor from the resolution is reported too.
    caplog.clear()
    with monkeypatch.context() as patched:
        patched.setattr(_memory, "resident_bytes", lambda: 0)
        with caplog.at_level(logging.DEBUG, logger="cafein"):
            with pytest.warns(UserWarning, match="minimum"):
                _memory.plan_call("time", 100, 0, max_memory="100M")
    (record,) = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert "the minimum budget" in record.getMessage()
    # The fare engine at 12_000 bytes/stop needs 120 MB per origin; a
    # 5 GiB budget leaves 1 GiB of headroom, minus 1 MiB fixed and the
    # 256 MiB result: 767 MiB // 120 MB = 6 — the budget binds.
    plan = _memory.plan_call("fare", 10_000, 256 * MIB, max_memory="5G", label="fare")
    assert plan.width == 6
    # A narrower explicit workers= wins; a wider one is narrowed under
    # an explicit budget, and a warned overrun under the default.
    assert (
        _memory.plan_call("fare", 10_000, 256 * MIB, workers=3, max_memory="5G").width
        == 3
    )
    assert (
        _memory.plan_call("fare", 10_000, 256 * MIB, workers=8, max_memory="5G").width
        == 6
    )
    _memory.set_max_memory("5G")
    with pytest.warns(UserWarning, match="workers=8"):
        assert _memory.plan_call("fare", 10_000, 256 * MIB, workers=8).width == 8
    # Wider than the ambient pool is still the caller's call.
    with pytest.warns(UserWarning, match="workers=12"):
        assert _memory.plan_call("fare", 10_000, 256 * MIB, workers=12).width == 12
    # The street engine scales by vertices, not stops: 4 M vertices at
    # 50 bytes each is 200 MB per origin, 1023 MiB // 200 MB = 5.
    assert _memory.plan_call("street", 4_000_000, MIB, max_memory="5G").width == 5


def test_plan_call_floors_refusals_and_phases(machine, monkeypatch):
    # A result that cannot fit refuses by name; a search that cannot
    # fit floors at one worker with the warning.
    with pytest.raises(ValueError, match="exceed the memory budget"):
        _memory.plan_call("time", 10_000, 20 * GIB, max_memory="5G")
    with pytest.warns(UserWarning, match="one fare search"):
        plan = _memory.plan_call("fare", 200_000, 900 * MIB, max_memory="5G")
    assert plan.width == 1
    # The fare arm's two sequential phases each plan their own width.
    plan = _memory.plan_call(
        "time", 10_000, 256 * MIB, max_memory="5G", refinement_engine="fare"
    )
    assert plan.width == 8 and plan.refinement_width == 6
    assert (
        _memory.plan_call(
            "time",
            10_000,
            256 * MIB,
            max_memory="5G",
            workers=2,
            refinement_engine="fare",
        ).refinement_width
        == 2
    )
    # Streams: the batch takes at most half the post-fixed headroom,
    # never above 500 rows; an oversized explicit batch is a warned floor.
    plan = _memory.plan_call(
        "time", 10_000, 0, max_memory="5G", streamed=True, row_bytes=MIB
    )
    assert plan.batch_rows == 500
    plan = _memory.plan_call(
        "time", 10_000, 0, max_memory="5G", streamed=True, row_bytes=4 * MIB
    )
    assert plan.batch_rows == (1023 * MIB // 2) // (4 * MIB)
    with pytest.warns(UserWarning, match="explicit batch"):
        plan = _memory.plan_call(
            "time",
            10_000,
            0,
            max_memory="5G",
            streamed=True,
            row_bytes=4 * MIB,
            batch_rows=500,
        )
    assert plan.batch_rows == 500
    with pytest.raises(ValueError, match="fixed allocation"):
        _memory.plan_call(
            "time", 10_000, 0, max_memory="4G", streamed=True, row_bytes=1
        )

    # No resident reader at all is a refusal, never a zero baseline.
    monkeypatch.setattr(_memory, "resident_bytes", lambda: None)
    with pytest.raises(ValueError, match="resident size cannot be read"):
        _memory.plan_call("time", 10_000, 0)


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


def test_the_active_plan_scopes_to_its_call():
    plan = _memory.Plan(headroom=1, batch_rows=None, width=3)
    assert _memory.active_plan() is None and _memory.width_or(7) == 7
    with _memory.use_plan(plan):
        assert _memory.active_plan() is plan and _memory.width_or(7) == 3
    assert _memory.active_plan() is None
