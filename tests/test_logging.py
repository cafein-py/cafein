"""The logging surface: quiet default, enable/disable, phases, timings."""

import io
import logging

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


def test_quiet_by_default(network, tmp_path, capsys):
    from cafein import TravelTimeMatrix

    path = _saved(network, tmp_path)
    cafein.TransportNetwork.load(path)
    TravelTimeMatrix(network, _served_stops(network, 40), departure=DEPARTURE)
    assert capsys.readouterr().err == ""
    handlers = logging.getLogger("cafein").handlers
    assert all(isinstance(h, logging.NullHandler) for h in handlers)


def test_enable_logging_streams_phase_lines(network, tmp_path):
    buffer = io.StringIO()
    cafein.enable_logging(stream=buffer)
    path = _saved(network, tmp_path)
    cafein.TransportNetwork.load(path)
    output = buffer.getvalue()
    assert "saved the network artifact in" in output
    assert "loaded the network artifact in" in output


def test_enable_logging_twice_does_not_duplicate(network, tmp_path):
    buffer = io.StringIO()
    cafein.enable_logging(stream=buffer)
    cafein.enable_logging(stream=buffer)
    _saved(network, tmp_path)
    lines = [line for line in buffer.getvalue().splitlines() if "saved" in line]
    assert len(lines) == 1


def test_disable_logging_restores_silence(network, tmp_path):
    buffer = io.StringIO()
    cafein.enable_logging(stream=buffer)
    cafein.disable_logging()
    _saved(network, tmp_path)
    assert buffer.getvalue() == ""


def test_enable_logging_validates_eagerly():
    with pytest.raises(TypeError, match="stream must be a writable text stream"):
        cafein.enable_logging(stream=42)
    with pytest.raises(TypeError, match="level must be an int or one of"):
        cafein.enable_logging(stream=io.StringIO(), level=True)
    with pytest.raises(TypeError, match="level must be an int or one of"):
        cafein.enable_logging(stream=io.StringIO(), level=b"info")
    with pytest.raises(ValueError, match="level must be one of"):
        cafein.enable_logging(stream=io.StringIO(), level="loud")


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


def test_from_gtfs_emits_build_phases(helsinki_gtfs, kantakaupunki_pbf, caplog):
    with caplog.at_level(logging.INFO, logger="cafein"):
        cafein.TransportNetwork.from_gtfs(helsinki_gtfs, osm_pbf=kantakaupunki_pbf)
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
    assert all(
        record.cafein_seconds > 0
        for record in caplog.records
        if hasattr(record, "cafein_phase")
    )


def test_gtfs_only_build_emits_no_street_phases(helsinki_gtfs, caplog):
    with caplog.at_level(logging.INFO, logger="cafein"):
        cafein.TransportNetwork.from_gtfs(helsinki_gtfs)
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


def test_collect_timings_is_independent_of_the_stream_level(network, tmp_path):
    buffer = io.StringIO()
    cafein.enable_logging(stream=buffer, level="warning")
    with cafein.collect_timings() as report:
        _saved(network, tmp_path)
    assert buffer.getvalue() == ""
    assert [entry["phase"] for entry in report.phases] == [
        "artifact.save.encode",
        "artifact.save",
    ]


def test_collect_timings_survives_a_child_logger_override(network, tmp_path):
    logging.getLogger("cafein.artifact").setLevel(logging.WARNING)
    with cafein.collect_timings() as report:
        _saved(network, tmp_path)
    assert [entry["phase"] for entry in report.phases] == [
        "artifact.save.encode",
        "artifact.save",
    ]


def test_collect_timings_leaves_root_handlers_to_their_own_config(
    network, tmp_path, caplog
):
    with caplog.at_level(logging.WARNING):
        with cafein.collect_timings() as report:
            _saved(network, tmp_path)
    assert [entry["phase"] for entry in report.phases] == [
        "artifact.save.encode",
        "artifact.save",
    ]
    assert not [r for r in caplog.records if r.name.startswith("cafein")]


def test_raising_handler_cannot_break_a_computation(network, tmp_path):
    class Exploding(logging.Handler):
        def emit(self, record):
            raise RuntimeError("boom")

        def handle(self, record):
            raise RuntimeError("boom")

    handler = Exploding(level=logging.DEBUG)
    root = logging.getLogger("cafein")
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        path = _saved(network, tmp_path)
    finally:
        root.removeHandler(handler)
    assert path.exists()


def test_raising_filter_cannot_break_a_computation(network, tmp_path):
    calls = []

    class Exploding(logging.Filter):
        def filter(self, record):
            calls.append(record)
            raise RuntimeError("boom")

    # Ancestor filters are not applied to propagated records, so the
    # filter goes on the emitting logger itself.
    exploding = Exploding()
    emitting = logging.getLogger("cafein.artifact")
    emitting.addFilter(exploding)
    logging.getLogger("cafein").setLevel(logging.DEBUG)
    try:
        path = _saved(network, tmp_path)
    finally:
        emitting.removeFilter(exploding)
    assert calls
    assert path.exists()


def test_manual_child_override_arms_the_rust_bridge(network, tmp_path):
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    root = logging.getLogger("cafein")
    emitting = logging.getLogger("cafein.artifact")
    root.setLevel(logging.WARNING)
    emitting.setLevel(logging.INFO)
    emitting.addHandler(handler)
    try:
        _saved(network, tmp_path)
    finally:
        emitting.removeHandler(handler)
    assert "encoded the artifact payload in" in buffer.getvalue()


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


def test_an_explicit_notset_root_still_arms_the_bridge(network, tmp_path):
    # Effective level 0 means "emit everything", not "disarmed".
    plain_root = logging.getLogger()
    prior = plain_root.level
    plain_root.setLevel(logging.NOTSET)
    try:
        output = _artifact_stream(network, tmp_path)
    finally:
        plain_root.setLevel(prior)
    assert "encoded the artifact payload in" in output


def test_a_negative_custom_level_arms_the_bridge(network, tmp_path):
    logging.getLogger("cafein").setLevel(-5)
    assert "encoded the artifact payload in" in _artifact_stream(network, tmp_path)


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
    from cafein import Accessibility, TravelCostMatrix, TravelTimeMatrix

    origins = _served_stops(network, 5)
    with caplog.at_level(logging.INFO, logger="cafein"):
        TravelTimeMatrix(network, origins=origins, departure=DEPARTURE)
        TravelCostMatrix(network, origins, origins, DEPARTURE)
        Accessibility(network, origins, _served_stops(network, 30), DEPARTURE)
    records = {
        record.cafein_phase: record
        for record in caplog.records
        if hasattr(record, "cafein_phase")
    }
    assert {
        "matrix.travel_times",
        "matrix.travel_costs",
        "matrix.accessibility",
    } <= set(records)
    assert records["matrix.travel_times"].cafein_details["rows"] > 0
    assert records["matrix.travel_times"].cafein_details["origins"] == 5
    assert records["matrix.travel_times"].cafein_seconds > 0


def test_itineraries_emit_their_phase(network, caplog):
    from cafein import DetailedItineraries

    with caplog.at_level(logging.INFO, logger="cafein"):
        DetailedItineraries(network, ["4810551"], ["1250551"], DEPARTURE)
    phases = [
        record.cafein_phase
        for record in caplog.records
        if hasattr(record, "cafein_phase")
    ]
    assert "matrix.itineraries" in phases


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
    # Ticks are plain INFO records, not phases: no structured
    # attributes, and the report holds only the computer's completion.
    tick_records = [record for record in caplog.records if "% (" in record.getMessage()]
    assert len(tick_records) == 20
    assert all(record.levelno == logging.INFO for record in tick_records)
    assert not any(
        hasattr(record, attribute)
        for record in tick_records
        for attribute in ("cafein_phase", "cafein_seconds", "cafein_details")
    )
    assert [entry["phase"] for entry in report.phases] == ["matrix.travel_times"]


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


def test_tick_percentages_never_regress(network):
    import re

    from cafein import TravelTimeMatrix

    origins = _served_stops(network, 40)
    buffer = io.StringIO()
    cafein.enable_logging(stream=buffer)
    TravelTimeMatrix(network, origins, departure=DEPARTURE)
    counts = [
        int(re.search(r"\((\d+)/40 origins", line).group(1))
        for line in buffer.getvalue().splitlines()
        if "% (" in line
    ]
    assert counts == sorted(counts) and len(counts) == len(set(counts))


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


def test_multimodal_build_children_report(helsinki_gtfs, kantakaupunki_pbf, caplog):
    with caplog.at_level(logging.INFO, logger="cafein"):
        cafein.TransportNetwork.from_gtfs(helsinki_gtfs, osm_pbf=kantakaupunki_pbf)
    phases = [
        record.cafein_phase
        for record in caplog.records
        if hasattr(record, "cafein_phase")
    ]
    assert phases.index("build.multimodal.streets") < phases.index("build.multimodal")
    assert "build.multimodal.elevation" not in phases  # no DEM in this build
    streets = next(
        record
        for record in caplog.records
        if getattr(record, "cafein_phase", None) == "build.multimodal.streets"
    )
    assert streets.cafein_details["modes"] == ["walk"]
    assert streets.cafein_details["edges"] > 0


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


def test_frontier_fanout_ticks_once_per_origin(network):
    import re

    from cafein import journey_frontiers

    origins = _served_stops(network, 40)
    buffer = io.StringIO()
    cafein.enable_logging(stream=buffer)
    journey_frontiers(
        network, origins, origins[:2], DEPARTURE, departure_time_window=10
    )
    ticks = [line for line in buffer.getvalue().splitlines() if "% (" in line]
    assert len(ticks) == 20
    assert all("journey frontiers" in line and "/40 origins" in line for line in ticks)
    counts = [int(re.search(r"\((\d+)/40", line).group(1)) for line in ticks]
    assert counts == sorted(counts) and len(counts) == len(set(counts))


def test_restricted_frontier_ticks_once_per_origin(network):
    from cafein import journey_frontiers

    origins = _served_stops(network, 40)
    buffer = io.StringIO()
    cafein.enable_logging(stream=buffer)
    journey_frontiers(
        network,
        origins,
        origins[:2],
        DEPARTURE,
        departure_time_window=10,
        max_slower=10,
    )
    ticks = [line for line in buffer.getvalue().splitlines() if "% (" in line]
    # The restriction pass must not inflate the count past the origins.
    assert len(ticks) == 20
    assert all("/40 origins" in line for line in ticks)


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


def test_a_generator_of_modes_reports_exact_details(kantakaupunki_pbf, caplog):
    from cafein import StreetNetwork

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
        self.refreshes = 0
        self.closed = False

    def refresh(self):
        self.refreshes += 1

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


def test_progress_bar_drives_one_bar_per_label(network, monkeypatch):
    from cafein import TravelTimeMatrix

    bars = []

    def factory(label, total):
        bar = _FakeBar(label, total)
        bars.append(bar)
        return bar

    monkeypatch.setattr(_log, "_bar_factory", factory)
    origins = _served_stops(network, 40)
    buffer = io.StringIO()
    cafein.enable_logging(stream=buffer, progress="bar")
    TravelTimeMatrix(network, origins, departure=DEPARTURE)
    assert len(bars) == 1
    bar = bars[0]
    assert bar.label == "travel_time_matrix"
    assert bar.total == 40
    assert bar.n == 40
    assert bar.closed
    output = buffer.getvalue()
    assert "computed the travel time matrix in" in output
    assert not [line for line in output.splitlines() if "% (" in line]


def test_progress_bar_without_tqdm_falls_back_to_lines(network, monkeypatch):
    from cafein import TravelTimeMatrix

    monkeypatch.setattr(_log, "_bar_factory", lambda label, total: None)
    origins = _served_stops(network, 40)
    buffer = io.StringIO()
    cafein.enable_logging(stream=buffer, progress="bar")
    with pytest.warns(UserWarning, match="tqdm is not installed"):
        TravelTimeMatrix(network, origins, departure=DEPARTURE)
    assert len([line for line in buffer.getvalue().splitlines() if "% (" in line]) == 20


def test_progress_validates_eagerly():
    with pytest.raises(TypeError, match="progress must be"):
        cafein.enable_logging(stream=io.StringIO(), progress=True)
    with pytest.raises(TypeError, match="progress must be"):
        cafein.enable_logging(stream=io.StringIO(), progress=3)
    with pytest.raises(ValueError, match="progress must be"):
        cafein.enable_logging(stream=io.StringIO(), progress="spinner")


def test_tick_records_carry_structured_progress(network, caplog):
    from cafein import TravelTimeMatrix

    origins = _served_stops(network, 40)
    with caplog.at_level(logging.INFO, logger="cafein"):
        TravelTimeMatrix(network, origins, departure=DEPARTURE)
    ticks = [r for r in caplog.records if hasattr(r, "cafein_progress")]
    assert len(ticks) == 20
    for record in ticks:
        info = record.cafein_progress
        assert info["label"] == "travel_time_matrix"
        assert 0 < info["done"] <= info["total"] == 40
        assert not hasattr(record, "cafein_phase")
    assert [r.cafein_progress["done"] for r in ticks] == sorted(
        r.cafein_progress["done"] for r in ticks
    )
