"""The logging surface: progress, enable/disable, phases, timings."""

import io
import logging
import sys

import pandas as pd
import pytest

import cafein
from cafein import _log


@pytest.fixture(autouse=True)
def _pristine_logging():
    """Snapshot and restore the cafein logger and module state."""
    root = logging.getLogger("cafein")
    handlers = list(root.handlers)
    level = root.level
    propagate = root.propagate
    child_levels = {
        name: logging.getLogger(name).level
        for name in ("cafein.build", "cafein.matrix", "cafein.artifact")
    }
    yield
    root.handlers[:] = handlers
    root.setLevel(level)
    root.propagate = propagate
    for name, child_level in child_levels.items():
        logging.getLogger(name).setLevel(child_level)
    _log._handler = None
    _log._prior_level = None
    del _log._collectors[:]
    _log.sync()


def _saved(network, tmp_path, name="net.cafein"):
    path = tmp_path / name
    network.save(path)
    return path


def test_no_log_lines_by_default(network, tmp_path, capsys):
    from cafein import TravelTimeMatrix

    path = _saved(network, tmp_path)
    cafein.TransportNetwork.load(path)
    TravelTimeMatrix(network, _served_stops(network, 40), departure=DEPARTURE)
    assert capsys.readouterr().err == ""
    handlers = logging.getLogger("cafein").handlers
    assert all(isinstance(h, logging.NullHandler) for h in handlers)


def test_enable_logging_lifecycle(network, tmp_path):
    buffer = io.StringIO()
    cafein.enable_logging(stream=buffer)
    # Enabling twice does not duplicate handlers (or lines).
    cafein.enable_logging(stream=buffer)
    path = _saved(network, tmp_path)
    cafein.TransportNetwork.load(path)
    output = buffer.getvalue()
    assert "saved the network artifact in" in output
    assert "loaded the network artifact in" in output
    assert len([line for line in output.splitlines() if "saved" in line]) == 1
    # Disabling restores silence.
    cafein.disable_logging()
    _saved(network, tmp_path, name="silent.cafein")
    assert buffer.getvalue() == output


def test_enable_logging_validates_eagerly():
    for kwargs, exc, match in (
        (dict(stream=42), TypeError, "stream must be a writable text stream"),
        (
            dict(stream=io.StringIO(), level=True),
            TypeError,
            "level must be an int or one of",
        ),
        (
            dict(stream=io.StringIO(), level=b"info"),
            TypeError,
            "level must be an int or one of",
        ),
        (dict(stream=io.StringIO(), level="loud"), ValueError, "level must be one of"),
        (dict(stream=io.StringIO(), progress=True), TypeError, "progress must be"),
        (dict(stream=io.StringIO(), progress=3), TypeError, "progress must be"),
        (
            dict(stream=io.StringIO(), progress="spinner"),
            ValueError,
            "progress must be",
        ),
    ):
        with pytest.raises(exc, match=match):
            cafein.enable_logging(**kwargs)


def test_disable_logging_restores_manual_configuration(network, tmp_path):
    root = logging.getLogger("cafein")
    root.setLevel(logging.DEBUG)
    mine = logging.StreamHandler(io.StringIO())
    root.addHandler(mine)
    try:
        cafein.enable_logging(stream=io.StringIO(), level="info")
        assert root.level == logging.INFO
        cafein.disable_logging()
        assert root.level == logging.DEBUG
        assert mine in root.handlers
        assert not any(
            isinstance(h, logging.StreamHandler) and h is not mine
            for h in root.handlers
        )
    finally:
        root.removeHandler(mine)


def test_disable_logging_is_a_noop_without_enable():
    before = list(logging.getLogger("cafein").handlers)
    cafein.disable_logging()
    assert logging.getLogger("cafein").handlers == before


def test_artifact_phases_carry_structured_attributes(network, tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="cafein"):
        path = _saved(network, tmp_path)
        cafein.TransportNetwork.load(path)
    phases = {
        record.cafein_phase: record
        for record in caplog.records
        if hasattr(record, "cafein_phase")
    }
    assert set(phases) == {
        "artifact.save.encode",
        "artifact.save",
        "artifact.load.decode",
        "artifact.load.rebuild",
        "artifact.load",
    }
    for record in phases.values():
        assert record.cafein_seconds > 0
    for parent in ("artifact.save", "artifact.load"):
        assert phases[parent].cafein_details["path"] == str(path)
    steps = [
        tuple(record.cafein_progress[key] for key in ("label", "done", "total"))
        for record in caplog.records
        if hasattr(record, "cafein_progress")
    ]
    assert steps == [
        ("saving the network", 1, 2),
        ("saving the network", 2, 2),
        ("loading the network", 1, 2),
        ("loading the network", 2, 2),
    ]


def test_from_gtfs_emits_build_phases(kantakaupunki_pbf, tmp_path, caplog):
    from test_frontier import build_two_line_gtfs

    feed = build_two_line_gtfs(tmp_path / "central_gtfs.zip", at=(60.17, 24.93))
    with caplog.at_level(logging.INFO, logger="cafein"):
        cafein.TransportNetwork.from_gtfs(feed, osm_pbf=kantakaupunki_pbf)
    phases = [
        record.cafein_phase
        for record in caplog.records
        if hasattr(record, "cafein_phase")
    ]
    # The exact list also pins the child-before-parent ordering and the
    # absence of build.multimodal.elevation (no DEM in this build).
    assert phases == [
        "build.gtfs.read",
        "build.gtfs.timetable",
        "build.gtfs.indexes",
        "build.gtfs",
        "build.streets.read",
        "build.streets.prune",
        "build.streets.graph",
        "build.streets.footpaths",
        "build.multimodal.streets",
        "build.multimodal",
    ]
    build = next(
        record
        for record in caplog.records
        if getattr(record, "cafein_phase", None) == "build.gtfs"
    )
    assert build.cafein_details["stops"] > 0
    assert build.cafein_details["trips"] > 0
    multimodal = next(
        record
        for record in caplog.records
        if getattr(record, "cafein_phase", None) == "build.multimodal"
    )
    assert multimodal.cafein_details["modes"] == ["walk"]
    steps = [
        record.cafein_progress["done"]
        for record in caplog.records
        if getattr(record, "cafein_progress", {}).get("label")
        == "building the transit core from GTFS"
    ]
    assert steps == [1, 2, 3]
    streets = next(
        record
        for record in caplog.records
        if getattr(record, "cafein_phase", None) == "build.multimodal.streets"
    )
    assert streets.cafein_details["modes"] == ["walk"]
    assert streets.cafein_details["edges"] > 0
    assert all(
        record.cafein_seconds > 0
        for record in caplog.records
        if hasattr(record, "cafein_phase")
    )


def test_gtfs_only_build_emits_no_street_phases(tmp_path, caplog):
    from test_frontier import build_two_line_gtfs

    feed = build_two_line_gtfs(tmp_path / "two_line_gtfs.zip")
    with caplog.at_level(logging.INFO, logger="cafein"):
        cafein.TransportNetwork.from_gtfs(feed)
    phases = [
        record.cafein_phase
        for record in caplog.records
        if hasattr(record, "cafein_phase")
    ]
    assert phases == [
        "build.gtfs.read",
        "build.gtfs.timetable",
        "build.gtfs.indexes",
        "build.gtfs",
    ]


def test_annotate_emits_its_phase(network, caplog):
    journeys = network.route_between_stops("4810551", "1250551", "2022-02-22 08:30:00")
    with caplog.at_level(logging.INFO, logger="cafein"):
        network.annotate_emissions([journeys[0]])
    records = [record for record in caplog.records if hasattr(record, "cafein_phase")]
    assert [record.cafein_phase for record in records] == ["emissions.annotate"]
    assert all(record.cafein_seconds > 0 for record in records)


def test_collect_timings_reports_phases(network, tmp_path):
    with cafein.collect_timings() as report:
        path = _saved(network, tmp_path)
        cafein.TransportNetwork.load(path)
    assert [entry["phase"] for entry in report.phases] == [
        "artifact.save.encode",
        "artifact.save",
        "artifact.load.decode",
        "artifact.load.rebuild",
        "artifact.load",
    ]
    for entry in report.phases:
        assert isinstance(entry["seconds"], float) and entry["seconds"] > 0
    for entry in report.phases:
        if entry["phase"] in ("artifact.save", "artifact.load"):
            assert entry["details"]["path"] == str(path)
    frame = report.frame()
    assert list(frame.columns) == ["phase", "seconds", "details"]
    assert len(frame) == 5
    assert pd.api.types.is_string_dtype(frame["phase"])
    assert pd.api.types.is_float_dtype(frame["seconds"])
    assert frame["details"].dtype == object


@pytest.mark.parametrize(
    "environment", ["stream_level", "child_override", "root_handlers"]
)
def test_collect_timings_ignores_the_logging_environment(
    network, tmp_path, caplog, environment
):
    # The collector's capture is independent of the stream level, of a
    # child-logger override, and of the root handler configuration
    # (which it leaves to its own devices).
    from contextlib import nullcontext

    buffer = io.StringIO()
    if environment == "stream_level":
        cafein.enable_logging(stream=buffer, level="warning")
    elif environment == "child_override":
        logging.getLogger("cafein.artifact").setLevel(logging.WARNING)
    context = (
        caplog.at_level(logging.WARNING)
        if environment == "root_handlers"
        else nullcontext()
    )
    with context:
        with cafein.collect_timings() as report:
            _saved(network, tmp_path)
    assert [entry["phase"] for entry in report.phases] == [
        "artifact.save.encode",
        "artifact.save",
    ]
    if environment == "stream_level":
        assert buffer.getvalue() == ""
    if environment == "root_handlers":
        assert not [r for r in caplog.records if r.name.startswith("cafein")]


@pytest.mark.parametrize("hostile", ["handler", "filter"])
def test_raising_logging_hooks_cannot_break_a_computation(network, tmp_path, hostile):
    calls = []

    class ExplodingHandler(logging.Handler):
        def emit(self, record):
            raise RuntimeError("boom")

        def handle(self, record):
            raise RuntimeError("boom")

    class ExplodingFilter(logging.Filter):
        def filter(self, record):
            calls.append(record)
            raise RuntimeError("boom")

    if hostile == "handler":
        exploding = ExplodingHandler(level=logging.DEBUG)
        target = logging.getLogger("cafein")
        target.addHandler(exploding)
        target.setLevel(logging.DEBUG)
        remove = target.removeHandler
    else:
        # Ancestor filters are not applied to propagated records, so the
        # filter goes on the emitting logger itself.
        exploding = ExplodingFilter()
        target = logging.getLogger("cafein.artifact")
        target.addFilter(exploding)
        logging.getLogger("cafein").setLevel(logging.DEBUG)
        remove = target.removeFilter
    try:
        path = _saved(network, tmp_path)
    finally:
        remove(exploding)
    if hostile == "filter":
        assert calls
    assert path.exists()


def _artifact_stream(network, tmp_path, name="net.cafein"):
    """A save's cafein.artifact output under the current configuration."""
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    emitting = logging.getLogger("cafein.artifact")
    emitting.addHandler(handler)
    try:
        _saved(network, tmp_path, name=name)
    finally:
        emitting.removeHandler(handler)
    return buffer.getvalue()


@pytest.mark.parametrize(
    "configuration", ["manual_child_override", "explicit_notset_root", "negative_level"]
)
def test_unusual_level_configurations_still_arm_the_bridge(
    network, tmp_path, configuration
):
    restore = None
    if configuration == "manual_child_override":
        # A manual child-logger override alone (root at WARNING) arms
        # the Rust bridge.
        logging.getLogger("cafein").setLevel(logging.WARNING)
        logging.getLogger("cafein.artifact").setLevel(logging.INFO)
    elif configuration == "explicit_notset_root":
        # Effective level 0 means "emit everything", not "disarmed".
        plain_root = logging.getLogger()
        restore = (plain_root, plain_root.level)
        plain_root.setLevel(logging.NOTSET)
    else:
        # A negative custom level arms the bridge too.
        logging.getLogger("cafein").setLevel(-5)
    try:
        output = _artifact_stream(network, tmp_path)
    finally:
        if restore is not None:
            restore[0].setLevel(restore[1])
    assert "encoded the artifact payload in" in output


def test_an_oversized_custom_level_suppresses_without_wrapping(network, tmp_path):
    # Armed first, so suppression afterwards is the clamp's doing.
    logging.getLogger("cafein").setLevel(logging.INFO)
    assert "encoded the artifact payload in" in _artifact_stream(network, tmp_path)
    logging.getLogger("cafein").setLevel(10**10)
    assert _artifact_stream(network, tmp_path, name="again.cafein") == ""
    # A collector still captures: its arming is independent of levels.
    with cafein.collect_timings() as report:
        _saved(network, tmp_path, name="collected.cafein")
    assert [entry["phase"] for entry in report.phases] == [
        "artifact.save.encode",
        "artifact.save",
    ]


DEPARTURE = "2022-02-22 08:30:00"


def _served_stops(network, count):
    stops = [stop for stop, lat, lon in network.stops if lat is not None]
    return stops[1000 : 1000 + count]


def test_matrix_computers_emit_their_phases(network, caplog):
    from cafein import (
        Accessibility,
        DetailedItineraries,
        TravelCostMatrix,
        TravelTimeMatrix,
    )

    origins = _served_stops(network, 5)
    with caplog.at_level(logging.INFO, logger="cafein"):
        TravelTimeMatrix(network, origins=origins, departure=DEPARTURE)
        TravelCostMatrix(network, origins, origins, DEPARTURE)
        Accessibility(network, origins, _served_stops(network, 30), DEPARTURE)
        DetailedItineraries(network, ["4810551"], ["1250551"], DEPARTURE)
    records = {
        record.cafein_phase: record
        for record in caplog.records
        if hasattr(record, "cafein_phase")
    }
    assert {
        "matrix.travel_times",
        "matrix.travel_costs",
        "matrix.accessibility",
        "matrix.itineraries",
    } <= set(records)
    assert records["matrix.travel_times"].cafein_details["rows"] > 0
    assert records["matrix.travel_times"].cafein_details["origins"] == 5
    assert records["matrix.travel_times"].cafein_seconds > 0


def test_street_matrices_swap_to_the_street_identifier(helsinki_streets, caplog):
    import geopandas
    from shapely.geometry import Point

    from cafein import TravelTimeMatrix

    points = geopandas.GeoDataFrame(
        {"id": ["a", "b"]},
        geometry=[Point(24.94, 60.17), Point(24.95, 60.18)],
        crs="EPSG:4326",
    )
    with caplog.at_level(logging.INFO, logger="cafein"):
        TravelTimeMatrix(helsinki_streets, points, transport_mode="walk")
    phases = [
        record.cafein_phase
        for record in caplog.records
        if hasattr(record, "cafein_phase")
    ]
    assert phases == ["matrix.streets"]


def test_fanout_ticks_above_the_threshold(network, caplog):
    import re

    from cafein import TravelTimeMatrix

    origins = _served_stops(network, 40)
    buffer = io.StringIO()
    cafein.enable_logging(stream=buffer)
    with caplog.at_level(logging.INFO, logger="cafein"):
        with cafein.collect_timings() as report:
            TravelTimeMatrix(network, origins, departure=DEPARTURE)
    ticks = [line for line in buffer.getvalue().splitlines() if "% (" in line]
    assert len(ticks) == 20
    assert all("travel_time_matrix" in line for line in ticks)
    assert all("origins" in line and "elapsed" in line for line in ticks)
    # Tick percentages never regress: the counts advance strictly.
    counts = [int(re.search(r"\((\d+)/40 origins", line).group(1)) for line in ticks]
    assert counts == sorted(counts) and len(counts) == len(set(counts))
    # Ticks are plain INFO records, not phases: they carry the
    # structured progress triple but none of the phase attributes,
    # and the report holds only the computer's completion.
    tick_records = [record for record in caplog.records if "% (" in record.getMessage()]
    assert len(tick_records) == 20
    assert all(record.levelno == logging.INFO for record in tick_records)
    assert not any(
        hasattr(record, attribute)
        for record in tick_records
        for attribute in ("cafein_phase", "cafein_seconds", "cafein_details")
    )
    assert [entry["phase"] for entry in report.phases] == ["matrix.travel_times"]
    # The records carry the structured progress payload for other
    # handlers, monotone in the done counter.
    progress = [r for r in caplog.records if hasattr(r, "cafein_progress")]
    assert len(progress) == 20
    for record in progress:
        info = record.cafein_progress
        assert info["label"] == "travel_time_matrix"
        assert 0 < info["done"] <= info["total"] == 40
        assert not hasattr(record, "cafein_phase")
    assert [r.cafein_progress["done"] for r in progress] == sorted(
        r.cafein_progress["done"] for r in progress
    )


def test_no_ticks_below_the_threshold(network):
    from cafein import TravelTimeMatrix

    origins = _served_stops(network, 10)
    buffer = io.StringIO()
    cafein.enable_logging(stream=buffer)
    TravelTimeMatrix(network, origins, departure=DEPARTURE)
    output = buffer.getvalue()
    assert not [line for line in output.splitlines() if "% (" in line]
    assert "computed the travel time matrix in" in output


def test_results_are_identical_with_logging_on(network):
    import pandas

    from cafein import TravelTimeMatrix

    origins = _served_stops(network, 40)
    quiet = TravelTimeMatrix(network, origins, departure=DEPARTURE)
    cafein.enable_logging(stream=io.StringIO(), level="debug")
    with cafein.collect_timings():
        loud = TravelTimeMatrix(network, origins, departure=DEPARTURE)
    pandas.testing.assert_frame_equal(pandas.DataFrame(quiet), pandas.DataFrame(loud))


def test_tbtr_precompute_emits_its_phase(network, tmp_path, caplog):
    # A loaded copy, so the shared session fixture is not mutated.
    path = _saved(network, tmp_path)
    copy = cafein.TransportNetwork.load(path)
    with caplog.at_level(logging.INFO, logger="cafein"):
        copy.compute_tbtr_transfers("2022-02-22")
    phases = [
        record.cafein_phase
        for record in caplog.records
        if hasattr(record, "cafein_phase")
    ]
    assert phases == ["build.tbtr"]


def _scattered_points(count, seed=1, centre=(60.170, 24.940)):
    import math

    import geopandas
    from shapely.geometry import Point

    latitude, longitude = centre
    records = []
    for index in range(count):
        angle = (seed + index) * 2.399963  # the golden angle
        radius = 100 + (seed * 37 + index * 211) % 900
        records.append(
            Point(
                longitude + radius * math.sin(angle) / 56_000,
                latitude + radius * math.cos(angle) / 111_320,
            )
        )
    return geopandas.GeoDataFrame(
        {"id": [f"point-{seed}-{index}" for index in range(count)]},
        geometry=records,
        crs="EPSG:4326",
    )


def test_street_cost_matrix_ticks(helsinki_streets):
    from cafein import TravelCostMatrix

    points = _scattered_points(40)
    buffer = io.StringIO()
    cafein.enable_logging(stream=buffer)
    TravelCostMatrix(helsinki_streets, points, points[:5], transport_mode="walk")
    output = buffer.getvalue()
    ticks = [line for line in output.splitlines() if "% (" in line]
    assert ticks and all("street matrix" in line for line in ticks)
    assert "computed the street cost matrix in" in output


def test_policy_cost_matrix_ticks(network_with_footpaths):
    from cafein import StreetLegPolicy, TravelCostMatrix

    points = _scattered_points(60, seed=2)
    buffer = io.StringIO()
    cafein.enable_logging(stream=buffer)
    TravelCostMatrix(
        network_with_footpaths,
        points,
        points[:3],
        DEPARTURE,
        street_policy=StreetLegPolicy(access={"walk": 1800}, egress={"walk": 1800}),
    )
    ticks = [line for line in buffer.getvalue().splitlines() if "% (" in line]
    assert ticks and all("travel_cost_matrix" in line for line in ticks)


def test_details_bind_positional_and_keyword_forms(network, caplog):
    import datetime

    from cafein import TravelTimeMatrix

    origins = _served_stops(network, 5)
    with caplog.at_level(logging.INFO, logger="cafein"):
        TravelTimeMatrix(
            network,
            origins,
            None,
            DEPARTURE,
            departure_time_window=datetime.timedelta(minutes=10),
            chunk=(0, 2),
        )
    record = next(
        record
        for record in caplog.records
        if getattr(record, "cafein_phase", None) == "matrix.travel_times"
    )
    assert record.cafein_details["origins"] == 5
    assert record.cafein_details["window"] == datetime.timedelta(minutes=10)
    assert record.cafein_details["chunk"] == (0, 2)


def test_windowed_matrix_ticks(network):
    import datetime

    from cafein import TravelTimeMatrix

    origins = _served_stops(network, 40)
    buffer = io.StringIO()
    cafein.enable_logging(stream=buffer)
    TravelTimeMatrix(
        network,
        origins,
        departure=DEPARTURE,
        departure_time_window=datetime.timedelta(minutes=10),
    )
    ticks = [line for line in buffer.getvalue().splitlines() if "% (" in line]
    assert len(ticks) == 20
    assert all("travel_time_matrix" in line for line in ticks)


def test_arrive_by_matrix_ticks(network):
    import re

    from cafein import TravelTimeMatrix

    origins = _served_stops(network, 2)
    buffer = io.StringIO()
    cafein.enable_logging(stream=buffer)
    # The arrive-by stop matrix fans out over destinations; one chunk
    # keeps the run small while staying above the tick floor.
    TravelTimeMatrix(network, origins, arrival=DEPARTURE, chunk=(0, 100))
    ticks = [line for line in buffer.getvalue().splitlines() if "% (" in line]
    assert ticks and all("arrive-by fold" in line for line in ticks)
    counts = [int(re.search(r"\((\d+)/", line).group(1)) for line in ticks]
    assert counts == sorted(counts) and len(counts) == len(set(counts))


def test_mixed_partitions_share_one_counter(ultra_network):
    from cafein import TravelTimeMatrix

    central = _served_stops(ultra_network, 20)
    # The farthest stops sit outside the central street extract, so
    # they get no street access and take the fallback partition.
    far = [
        stop
        for stop, _ in sorted(
            (
                (stop, (lat - 60.17) ** 2 + (lon - 24.94) ** 2)
                for stop, lat, lon in ultra_network.stops
                if lat is not None
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )[:20]
    ]
    buffer = io.StringIO()
    cafein.enable_logging(stream=buffer)
    TravelTimeMatrix(ultra_network, central + far, departure=DEPARTURE)
    ticks = [line for line in buffer.getvalue().splitlines() if "% (" in line]
    # A 20/20 usable/fallback split still advances one 40-origin
    # counter: per-partition tickers would emit nothing here.
    assert len(ticks) == 20
    assert all("/40 origins" in line for line in ticks)


@pytest.mark.parametrize("extra", [{}, {"max_slower": 10}], ids=["plain", "restricted"])
def test_frontier_fanout_ticks_once_per_origin(network, extra):
    import re

    from cafein import journey_frontiers

    origins = _served_stops(network, 40)
    buffer = io.StringIO()
    cafein.enable_logging(stream=buffer)
    journey_frontiers(
        network, origins, origins[:2], DEPARTURE, departure_time_window=10, **extra
    )
    ticks = [line for line in buffer.getvalue().splitlines() if "% (" in line]
    # The restriction pass must not inflate the count past the origins.
    assert len(ticks) == 20
    assert all("journey frontiers" in line and "/40 origins" in line for line in ticks)
    counts = [int(re.search(r"\((\d+)/40", line).group(1)) for line in ticks]
    assert counts == sorted(counts) and len(counts) == len(set(counts))


def test_street_artifact_phases(helsinki_streets, tmp_path):
    from cafein import StreetNetwork

    with cafein.collect_timings() as report:
        path = tmp_path / "streets.cafein"
        helsinki_streets.save(path)
        StreetNetwork.load(path)
    assert [entry["phase"] for entry in report.phases] == [
        "artifact.save",
        "artifact.load",
    ]
    for entry in report.phases:
        assert entry["details"]["path"] == str(path)


def test_standalone_street_build_has_the_multimodal_parent(kantakaupunki_pbf, caplog):
    from cafein import StreetNetwork

    # The default-mode build reports the parent phase and its modes.
    with caplog.at_level(logging.INFO, logger="cafein"):
        StreetNetwork.from_osm(str(kantakaupunki_pbf))
    phases = [
        record.cafein_phase
        for record in caplog.records
        if hasattr(record, "cafein_phase")
    ]
    assert phases.index("build.multimodal.streets") < phases.index("build.multimodal")
    parent = next(
        record
        for record in caplog.records
        if getattr(record, "cafein_phase", None) == "build.multimodal"
    )
    assert "walk" in parent.cafein_details["modes"]
    # A generator of modes must still report exact details.
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="cafein"):
        StreetNetwork.from_osm(str(kantakaupunki_pbf), modes=(m for m in ["walk"]))
    parent = next(
        record
        for record in caplog.records
        if getattr(record, "cafein_phase", None) == "build.multimodal"
    )
    assert parent.cafein_details["modes"] == ["walk"]


def test_a_bare_mode_string_still_refuses_before_any_read(tmp_path):
    from cafein import StreetNetwork

    with pytest.raises(TypeError, match="pass"):
        StreetNetwork.from_osm(str(tmp_path / "missing.osm.pbf"), modes="walk")


class _FakeBar:
    def __init__(self, label, total):
        self.label = label
        self.total = total
        self.n = 0
        self.seen = []
        self.refreshed = 0
        self.closed = False

    def refresh(self):
        self.refreshed += 1

    def update(self, step):
        self.n += step
        self.seen.append(self.n)

    def close(self):
        self.closed = True


def test_progress_false_logs_only_phases(network, caplog):
    from cafein import TravelTimeMatrix

    origins = _served_stops(network, 40)
    buffer = io.StringIO()
    cafein.enable_logging(stream=buffer, progress=False)
    with caplog.at_level(logging.INFO, logger="cafein"):
        TravelTimeMatrix(network, origins, departure=DEPARTURE)
    output = buffer.getvalue()
    assert "computed the travel time matrix in" in output
    assert not [line for line in output.splitlines() if "% (" in line]
    # The records themselves still carry every tick for other handlers.
    ticks = [r for r in caplog.records if hasattr(r, "cafein_progress")]
    assert len(ticks) == 20


@pytest.mark.parametrize("setting", ["default", "bar", "off"])
def test_fanouts_draw_one_bar_as_progress_is_set(network, monkeypatch, setting):
    from cafein import TravelTimeMatrix

    bars = []
    buffer = io.StringIO()

    def factory(label, total, stream):
        assert stream is (None if setting == "default" else buffer)
        bar = _FakeBar(label, total)
        bars.append(bar)
        return bar

    origins = _served_stops(network, 40)
    monkeypatch.setattr(_log, "_bar_factory", factory)
    monkeypatch.setattr(_log, "_default_progress", True)
    _log.sync()
    if setting == "bar":
        cafein.enable_logging(stream=buffer, progress="bar")
    elif setting == "off":
        cafein.enable_logging(stream=buffer, level="warning", progress=False)
    TravelTimeMatrix(network, origins, departure=DEPARTURE)
    assert not [line for line in buffer.getvalue().splitlines() if "% (" in line]
    if setting == "off":
        assert bars == [] and buffer.getvalue() == ""
        return
    assert len(bars) == 1
    bar = bars[0]
    assert (bar.label, bar.total, bar.n) == ("travel_time_matrix", 40, 40)
    assert bar.seen == list(range(2, 41, 2))
    assert bar.closed


@pytest.mark.parametrize(
    "setting, delay, line",
    [
        ("default", 0.0, "reading took 00:00\n"),
        ("default", 2.0, ""),
        ("bar", 0.0, "reading took 00:00\n"),
    ],
)
def test_hidden_bars_write_one_line_with_their_time(
    monkeypatch, capsys, setting, delay, line
):
    # Captured stderr and a StringIO are no terminals: the bar is hidden,
    # and an operation that ran the delay or longer writes its time.
    buffer = io.StringIO()
    monkeypatch.setattr(_log, "_default_progress", True)
    monkeypatch.setattr(_log, "_DELAY", delay)
    if setting == "bar":
        cafein.enable_logging(stream=buffer, progress="bar")
    for event, done in (("start", 0), ("tick", 10), ("end", 10)):
        _log._present(event, "run", "reading", done, 10)
    written = buffer.getvalue() if setting == "bar" else capsys.readouterr().err
    assert written == line


class _WriteOnly:
    def write(self, text):
        pass


@pytest.mark.parametrize(
    "kernel, stream, shown",
    [
        (True, None, True),
        (True, io.StringIO(), False),
        (False, _WriteOnly(), False),
    ],
)
def test_bars_show_on_terminals_and_on_a_kernel_s_stderr(
    monkeypatch, kernel, stream, shown
):
    import types

    class ZMQInteractiveShell:
        pass

    shell = ZMQInteractiveShell() if kernel else None
    ipython = types.SimpleNamespace(get_ipython=lambda: shell)
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    _, disable = _log._bar_class(stream)
    assert disable is not shown


def test_overlapping_runs_of_one_label_keep_their_own_bars(monkeypatch):
    bars = []

    def factory(label, total, stream):
        bars.append(_FakeBar(label, total))
        return bars[-1]

    monkeypatch.setattr(_log, "_bar_factory", factory)
    monkeypatch.setattr(_log, "_default_progress", True)
    _log._present("start", "a", "matrix", 0, 10)
    _log._present("start", "b", "matrix", 0, 10)
    _log._present("tick", "a", "matrix", 6, 10)
    _log._present("tick", "b", "matrix", 2, 10)
    _log._present("end", "a", "matrix", 6, 10)
    assert [(bar.n, bar.closed) for bar in bars] == [(6, True), (2, False)]
    _log._present("end", "b", "matrix", 2, 10)
    assert bars[1].closed


def test_shown_bars_redraw_while_a_step_runs(monkeypatch):
    import time

    bars = []

    def factory(label, total, stream):
        bars.append(_FakeBar(label, total))
        return bars[-1]

    monkeypatch.setattr(_log, "_bar_factory", factory)
    monkeypatch.setattr(_log, "_default_progress", True)
    monkeypatch.setattr(_log, "_DELAY", 0.0)
    monkeypatch.setattr(_log, "_REDRAW_SECONDS", 0.01)
    _log._present("start", "a", "reading", 0, 3)
    time.sleep(0.2)
    _log._present("end", "a", "reading", 0, 3)
    redraws = bars[0].refreshed
    assert redraws > 0 and bars[0].closed
    time.sleep(0.05)
    assert bars[0].refreshed == redraws


def test_footpath_search_ticks_once_per_stop():
    from test_streets import STRAIGHT_STREET, footpaths, stop

    buffer = io.StringIO()
    cafein.enable_logging(stream=buffer)
    # 41 stops tick at every third (the 5 % step) and at the last one.
    stops = [stop(f"s{index}", 5 + 9 * index, 0) for index in range(41)]
    footpaths(STRAIGHT_STREET, [("A", "B", 400)], stops)
    ticks = [line for line in buffer.getvalue().splitlines() if "footpath" in line]
    ticks = [line for line in ticks if "% (" in line]
    assert len(ticks) == 14
    assert "(39/41 stops," in ticks[-2] and "(41/41 stops," in ticks[-1]


def test_reconfiguring_progress_closes_open_bars(monkeypatch):
    bars = []

    def factory(label, total, stream):
        bars.append(_FakeBar(label, total))
        return bars[-1]

    monkeypatch.setattr(_log, "_bar_factory", factory)
    monkeypatch.setattr(_log, "_default_progress", True)
    _log._present("start", "a", "matrix", 0, 10)
    cafein.enable_logging(stream=io.StringIO(), level="warning", progress=False)
    assert bars[0].closed
    _log._present("tick", "a", "matrix", 5, 10)
    _log._present("start", "b", "matrix", 0, 10)
    assert (len(bars), bars[0].n) == (1, 0)


@pytest.mark.parametrize(
    "read, label",
    [
        ("streets._walking_network", "reading the OSM walking network"),
        ("_osm.union_network", "reading the OSM street network"),
        (
            "streets.park_and_ride_facilities",
            "reading the OSM park-and-ride facilities",
        ),
    ],
)
def test_pyrosm_reads_show_cafein_progress(monkeypatch, tmp_path, read, label):
    import importlib

    import pyrosm

    bars = []

    def factory(label, total, stream):
        bars.append(_FakeBar(label, total))
        return bars[-1]

    class _OSM:
        def __init__(self, path, progress, **options):
            for done in (0, 50, 100):
                progress(done, 100)
            raise RuntimeError("stop after the progress")

    monkeypatch.setattr(_log, "_bar_factory", factory)
    monkeypatch.setattr(_log, "_default_progress", True)
    monkeypatch.setattr(pyrosm, "OSM", _OSM)
    module, function = read.split(".")
    reader = getattr(importlib.import_module(f"cafein.{module}"), function)
    with pytest.raises(RuntimeError, match="stop after"):
        reader(str(tmp_path / "area.osm.pbf"))
    assert [(bar.label, bar.seen, bar.closed) for bar in bars] == [
        (label, [50, 100], True)
    ]


def test_pyrosm_progress_follows_the_logging_setting():
    buffer = io.StringIO()
    cafein.enable_logging(stream=buffer, level="warning", progress=False)
    with _log.reporting("reading", _log.build) as report:
        assert report is False
    cafein.enable_logging(stream=buffer)
    with _log.reporting("reading", _log.build) as report:
        for done in range(101):
            report(done, 100)
    lines = [line for line in buffer.getvalue().splitlines() if "% (" in line]
    assert len(lines) == 20
    assert "reading 5% (" in lines[0] and "reading 100% (" in lines[-1]
