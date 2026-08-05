"""Streaming ``travel_cost_table``: writers, manifests, fingerprints."""

import subprocess
import sys
import textwrap

import pytest

pyarrow = pytest.importorskip("pyarrow")
import pyarrow.parquet as parquet  # noqa: E402

from cafein import (  # noqa: E402
    _streaming,
    travel_cost_table,
    TravelCostMatrix,
    TravelTimeMatrix,
)

QUERY = {"date": "2022-02-22", "departure": "08:30:00"}


def _origin_stops(network, count):
    return [stop for stop, _, _ in network.stops][:count]


def _read_aligned(path, frame):
    """The streamed output as a frame, dictionary and constant columns
    cast back to the constructor frame's dtypes for exact comparison."""
    import pandas as pd

    read = parquet.read_table(path).to_pandas()
    for name in ("from_id", "to_id", "distance_provenance", "currency"):
        if name in read.columns:
            read[name] = read[name].astype(frame[name].dtype)
    return read, pd.DataFrame(frame)


@pytest.fixture(scope="module")
def car_streets(kantakaupunki_pbf):
    from cafein import StreetNetwork

    return StreetNetwork.from_osm(str(kantakaupunki_pbf), modes=("walk", "car"))


@pytest.fixture(scope="module")
def street_points():
    import geopandas

    return geopandas.GeoDataFrame(
        {"id": ["a", "b", "c", "d", "e"]},
        geometry=geopandas.points_from_xy(
            [24.9384, 24.9241, 24.9600, 24.9450, 24.9300],
            [60.1699, 60.1587, 60.1870, 60.1750, 60.1650],
        ),
        crs="EPSG:4326",
    )


def test_streamed_file_equals_unstreamed(network, tmp_path):
    stops = _origin_stops(network, 120)
    plain = travel_cost_table(network, origins=stops, **QUERY)
    target = tmp_path / "matrix.parquet"
    result = travel_cost_table(
        network, origins=stops, **QUERY, output=target, batch_size=50
    )
    assert isinstance(result, _streaming.StreamingResult)
    assert (result.mode, result.batches, result.manifest) == ("file", 3, None)
    assert result.path == target
    assert result.rows == plain.num_rows
    assert result.schema == plain.schema
    read = parquet.read_table(target)
    assert read.equals(plain)
    assert parquet.ParquetFile(target).num_row_groups == 3
    # The sidecar claim is released on completed publication.
    assert list(tmp_path.glob("*.claim")) == []


def test_streamed_directory_equals_unstreamed(network, tmp_path):
    stops = _origin_stops(network, 120)
    plain = travel_cost_table(network, origins=stops, **QUERY)
    target = tmp_path / "shards"
    result = travel_cost_table(
        network, origins=stops, **QUERY, output=target, batch_size=50
    )
    assert (result.mode, result.batches) == ("directory", 3)
    parts = sorted(target.glob("part-*.parquet"))
    assert [part.name for part in parts] == [
        "part-00000.parquet",
        "part-00001.parquet",
        "part-00002.parquet",
    ]
    joined = pyarrow.concat_tables(parquet.read_table(part) for part in parts)
    assert joined.equals(plain)
    manifest = result.manifest
    assert manifest["operation"] == "travel_cost_table"
    assert manifest["format"] == _streaming.MANIFEST_FORMAT
    assert manifest["fingerprint_version"] == _streaming.FINGERPRINT_VERSION
    assert manifest["batch_size"] == 50
    assert manifest["origin_count"] == 120
    assert [shard["origin_start"] for shard in manifest["shards"]] == [0, 50, 100]
    assert [shard["origin_stop"] for shard in manifest["shards"]] == [50, 100, 120]
    assert all(shard["completed"] for shard in manifest["shards"])
    assert sum(shard["rows"] for shard in manifest["shards"]) == plain.num_rows
    import json

    assert json.loads((target / "manifest.json").read_text()) == manifest


def test_streamed_point_query(network_with_footpaths, tmp_path):
    import geopandas

    coordinates = [(24.9384, 60.1699), (24.9241, 60.1587), (24.9600, 60.1870)]
    points = geopandas.GeoDataFrame(
        {"id": ["a", "b", "c"]},
        geometry=geopandas.points_from_xy(*zip(*coordinates)),
        crs="EPSG:4326",
    )
    plain = travel_cost_table(network_with_footpaths, origins=points, **QUERY)
    result = travel_cost_table(
        network_with_footpaths,
        origins=points,
        **QUERY,
        output=tmp_path / "points",
        batch_size=2,
    )
    parts = sorted((tmp_path / "points").glob("part-*.parquet"))
    joined = pyarrow.concat_tables(parquet.read_table(part) for part in parts)
    assert joined.equals(plain)
    assert result.batches == 2


def test_chunk_composes_with_output(network, tmp_path):
    stops = _origin_stops(network, 100)
    chunked = travel_cost_table(network, origins=stops, **QUERY, chunk=(1, 2))
    result = travel_cost_table(
        network,
        origins=stops,
        **QUERY,
        chunk=(1, 2),
        output=tmp_path / "chunk.parquet",
        batch_size=20,
    )
    assert result.rows == chunked.num_rows
    assert parquet.read_table(tmp_path / "chunk.parquet").equals(chunked)


def test_output_mode_suffix_rule(tmp_path):
    assert _streaming.resolve_output(tmp_path / "table.PARQUET")[0] == "file"
    # The file form reserves a sidecar claim (the final name stays
    # absent until publication): a second job racing the same output
    # fails at resolve time.
    assert not (tmp_path / "table.PARQUET").exists()
    assert (tmp_path / "table.PARQUET.claim").exists()
    with pytest.raises(FileExistsError, match="claimed"):
        _streaming.resolve_output(tmp_path / "table.PARQUET")
    assert _streaming.resolve_output(tmp_path / "shards")[0] == "directory"
    # Path normalisation drops trailing slashes: still the directory form.
    assert _streaming.resolve_output(str(tmp_path / "more") + "/")[0] == "directory"
    existing = tmp_path / "taken.parquet"
    existing.write_bytes(b"")
    with pytest.raises(FileExistsError, match="never overwrites"):
        _streaming.resolve_output(existing)
    as_directory = tmp_path / "dir.parquet"
    as_directory.mkdir()
    with pytest.raises(FileExistsError):
        _streaming.resolve_output(as_directory)
    plain_file = tmp_path / "file"
    plain_file.write_bytes(b"")
    with pytest.raises(ValueError, match="not a directory"):
        _streaming.resolve_output(plain_file)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "leftover").write_bytes(b"")
    with pytest.raises(ValueError, match="not empty"):
        _streaming.resolve_output(occupied)


def test_failed_file_stream_releases_the_claim(tmp_path):
    _streaming.resolve_output(tmp_path / "broken.parquet")

    def explode():
        raise RuntimeError("mid-stream failure")
        yield

    with pytest.raises(RuntimeError, match="mid-stream"):
        _streaming.write_stream("file", tmp_path / "broken.parquet", explode(), {}, {})
    assert list(tmp_path.iterdir()) == []


def test_close_failure_releases_the_claim(tmp_path, monkeypatch):
    import pyarrow.parquet

    class ExplodingWriter(pyarrow.parquet.ParquetWriter):
        def close(self):
            super().close()
            raise OSError("footer write failed")

    monkeypatch.setattr(pyarrow.parquet, "ParquetWriter", ExplodingWriter)
    target = tmp_path / "footer.parquet"
    _streaming.resolve_output(target)
    domain = pyarrow.array(["a"], type=pyarrow.string())
    table = pyarrow.table(
        {
            "from_id": pyarrow.DictionaryArray.from_arrays(
                pyarrow.array([0], type=pyarrow.int64()), domain
            )
        }
    )
    with pytest.raises(OSError, match="footer"):
        _streaming.write_stream(
            "file", target, iter([(0, 0, 1, table)]), {}, {"from_id": domain}
        )
    assert list(tmp_path.iterdir()) == []


def test_manifest_claim_is_exclusive(tmp_path):
    _streaming.write_manifest(tmp_path, {"fingerprint": "x"}, claim=True)
    with pytest.raises(FileExistsError):
        _streaming.write_manifest(tmp_path, {"fingerprint": "x"}, claim=True)
    # Updates replace atomically and leave no temporaries behind.
    _streaming.write_manifest(tmp_path, {"fingerprint": "y"})
    assert [p.name for p in tmp_path.iterdir()] == ["manifest.json"]


def test_failed_file_run_leaves_nothing(network, tmp_path):
    stops = _origin_stops(network, 4)
    target = tmp_path / "doomed.parquet"
    with pytest.raises(ValueError, match="router"):
        travel_cost_table(network, origins=stops, **QUERY, output=target, router="warp")
    assert list(tmp_path.iterdir()) == []


def test_one_shot_exclusions_apply_to_every_batch(network, tmp_path):
    stops = _origin_stops(network, 40)
    routes = [route for route, _, _ in network.routes][:5]
    plain = travel_cost_table(
        network, origins=stops, **QUERY, exclude_routes=list(routes)
    )
    result = travel_cost_table(
        network,
        origins=stops,
        **QUERY,
        exclude_routes=iter(routes),
        output=tmp_path / "one-shot.parquet",
        batch_size=10,
    )
    assert result.batches == 4
    assert parquet.read_table(tmp_path / "one-shot.parquet").equals(plain)


def test_fingerprint_covers_destination_selection(network, tmp_path):
    stops = _origin_stops(network, 12)

    def fingerprint(suffix, destinations):
        return travel_cost_table(
            network,
            origins=stops[:6],
            destinations=destinations,
            **QUERY,
            output=tmp_path / suffix,
            batch_size=6,
        ).manifest["fingerprint"]

    narrow = fingerprint("narrow", stops[6:9])
    wider = fingerprint("wider", stops[6:10])
    assert narrow != wider


def test_streaming_validation(network, tmp_path):
    stops = _origin_stops(network, 4)
    with pytest.raises(ValueError, match="batch_size requires output"):
        travel_cost_table(network, origins=stops, **QUERY, batch_size=10)
    with pytest.raises(ValueError, match="resume=True requires output"):
        travel_cost_table(network, origins=stops, **QUERY, resume=True)
    with pytest.raises(ValueError, match="batch_size"):
        travel_cost_table(
            network, origins=stops, **QUERY, output=tmp_path / "x.parquet", batch_size=0
        )
    with pytest.raises(NotImplementedError, match="resume"):
        travel_cost_table(
            network, origins=stops, **QUERY, output=tmp_path / "y.parquet", resume=True
        )
    # resume=False is inert everywhere.
    table = travel_cost_table(network, origins=stops, **QUERY, resume=False)
    assert table.num_rows > 0


def test_fingerprint_covers_coordinates(network_with_footpaths, tmp_path):
    import geopandas

    def stream(suffix, coordinates):
        points = geopandas.GeoDataFrame(
            {"id": ["a", "b"]},
            geometry=geopandas.points_from_xy(*zip(*coordinates)),
            crs="EPSG:4326",
        )
        return travel_cost_table(
            network_with_footpaths,
            origins=points,
            **QUERY,
            output=tmp_path / suffix,
            batch_size=10,
        ).manifest["fingerprint"]

    base = [(24.9384, 60.1699), (24.9241, 60.1587)]
    first = stream("one", base)
    again = stream("two", base)
    moved = stream("three", [base[0], (24.9241, 60.1588)])
    assert first == again
    assert first != moved


def test_streamed_geometries_and_schema(network_with_footpaths, tmp_path):
    import geopandas

    coordinates = [(24.9384, 60.1699), (24.9241, 60.1587), (24.9600, 60.1870)]
    points = geopandas.GeoDataFrame(
        {"id": ["a", "b", "c"]},
        geometry=geopandas.points_from_xy(*zip(*coordinates)),
        crs="EPSG:4326",
    )
    plain = travel_cost_table(
        network_with_footpaths, origins=points, **QUERY, geometries=True
    )
    result = travel_cost_table(
        network_with_footpaths,
        origins=points,
        **QUERY,
        geometries=True,
        output=tmp_path / "geometry.parquet",
        batch_size=2,
    )
    read = parquet.read_table(tmp_path / "geometry.parquet")
    assert read.equals(plain)
    assert read.schema.field("geometry").type == pyarrow.binary()
    # Deliberately no file-level GeoParquet metadata: later batches could
    # invalidate any bounds a footer claimed.
    metadata = parquet.ParquetFile(tmp_path / "geometry.parquet").metadata.metadata
    assert not metadata or b"geo" not in metadata
    assert result.rows == plain.num_rows


def test_large_batch_stays_one_row_group(network, tmp_path):
    # 600 early Helsinki stops exceed pyarrow's default row-group split
    # (1 Mi rows) in a single batch; the writer must keep batch == group.
    stops = _origin_stops(network, 600)
    result = travel_cost_table(
        network,
        origins=stops,
        **QUERY,
        output=tmp_path / "large.parquet",
        batch_size=600,
    )
    assert result.batches == 1
    assert result.rows > 1_048_576
    assert parquet.ParquetFile(tmp_path / "large.parquet").num_row_groups == 1


def test_writer_rejects_deviating_batches(tmp_path):
    domain = pyarrow.array(["a", "b"], type=pyarrow.string())

    def batch(indices, dictionary, value_type=None):
        value_type = value_type or pyarrow.float64()
        return pyarrow.table(
            {
                "from_id": pyarrow.DictionaryArray.from_arrays(
                    pyarrow.array(indices, type=pyarrow.int64()), dictionary
                ),
                "value": pyarrow.array([1] * len(indices), type=value_type),
            }
        )

    seed = {"operation": "test", "fingerprint": "x", "fingerprint_version": 1}
    (tmp_path / "bad-domain").mkdir()
    (tmp_path / "bad-schema").mkdir()
    deviating_domain = [
        (0, 0, 1, batch([0], domain)),
        (1, 1, 2, batch([1], pyarrow.array(["a", "c"], type=pyarrow.string()))),
    ]
    with pytest.raises(ValueError, match="dictionary domain"):
        _streaming.write_stream(
            "directory",
            tmp_path / "bad-domain",
            iter(deviating_domain),
            seed,
            {"from_id": domain},
        )
    deviating_schema = [
        (0, 0, 1, batch([0], domain)),
        (1, 1, 2, batch([1], domain, value_type=pyarrow.int64())),
    ]
    with pytest.raises(ValueError, match="schema"):
        _streaming.write_stream(
            "directory",
            tmp_path / "bad-schema",
            iter(deviating_schema),
            seed,
            {"from_id": domain},
        )
    (tmp_path / "copied-domain").mkdir()
    copied = [
        (0, 0, 1, batch([0], domain)),
        (1, 1, 2, batch([1], pyarrow.array(["a", "b"], type=pyarrow.string()))),
    ]
    with pytest.raises(ValueError, match="dictionary domain"):
        _streaming.write_stream(
            "directory",
            tmp_path / "copied-domain",
            iter(copied),
            seed,
            {"from_id": domain},
        )
    (tmp_path / "sliced-domain").mkdir()
    sliced = [
        (0, 0, 1, batch([0], domain)),
        (1, 1, 2, batch([0], domain.slice(0, 1))),
    ]
    with pytest.raises(ValueError, match="dictionary domain"):
        _streaming.write_stream(
            "directory",
            tmp_path / "sliced-domain",
            iter(sliced),
            seed,
            {"from_id": domain},
        )
    (tmp_path / "chunked").mkdir()
    dictionary_column = pyarrow.chunked_array(
        [
            pyarrow.DictionaryArray.from_arrays(
                pyarrow.array([0], type=pyarrow.int64()), domain
            )
        ]
        * 2
    )
    chunked = pyarrow.table({"from_id": dictionary_column})
    with pytest.raises(ValueError, match="one chunk"):
        _streaming.write_stream(
            "directory",
            tmp_path / "chunked",
            iter([(0, 0, 2, chunked)]),
            seed,
            {"from_id": domain},
        )


def test_cost_to_parquet_transit_equals_constructor(network, tmp_path):
    import pandas as pd

    stops = _origin_stops(network, 80)
    frame = TravelCostMatrix(network, origins=stops, **QUERY)
    result = TravelCostMatrix.to_parquet(
        network, origins=stops, **QUERY, output=tmp_path / "tcm.parquet", batch_size=30
    )
    assert isinstance(result, _streaming.StreamingResult)
    assert result.batches == 3
    read, expected = _read_aligned(tmp_path / "tcm.parquet", frame)
    pd.testing.assert_frame_equal(read, expected)


def test_time_to_parquet_transit_equals_constructor(network, tmp_path):
    import pandas as pd

    stops = _origin_stops(network, 80)
    frame = TravelTimeMatrix(network, origins=stops, **QUERY)
    TravelTimeMatrix.to_parquet(
        network, origins=stops, **QUERY, output=tmp_path / "ttm.parquet", batch_size=30
    )
    read, expected = _read_aligned(tmp_path / "ttm.parquet", frame)
    pd.testing.assert_frame_equal(read, expected)
    windowed = TravelTimeMatrix(
        network, origins=stops[:20], **QUERY, window=600, confidence=0.8
    )
    TravelTimeMatrix.to_parquet(
        network,
        origins=stops[:20],
        **QUERY,
        window=600,
        confidence=0.8,
        output=tmp_path / "windowed.parquet",
        batch_size=8,
    )
    read, expected = _read_aligned(tmp_path / "windowed.parquet", windowed)
    pd.testing.assert_frame_equal(read, expected)


def test_time_to_parquet_points_default_destinations(network_with_footpaths, tmp_path):
    # destinations=None means every origin — including origins outside
    # the routed batch, which must never narrow the destination set.
    import geopandas
    import pandas as pd

    coordinates = [(24.9384, 60.1699), (24.9241, 60.1587), (24.9600, 60.1870)]
    points = geopandas.GeoDataFrame(
        {"id": ["a", "b", "c"]},
        geometry=geopandas.points_from_xy(*zip(*coordinates)),
        crs="EPSG:4326",
    )
    frame = TravelTimeMatrix(network_with_footpaths, origins=points, **QUERY)
    result = TravelTimeMatrix.to_parquet(
        network_with_footpaths,
        origins=points,
        **QUERY,
        output=tmp_path / "points.parquet",
        batch_size=1,
    )
    assert result.batches == 3
    read, expected = _read_aligned(tmp_path / "points.parquet", frame)
    pd.testing.assert_frame_equal(read, expected)


def test_cost_to_parquet_street_car_equals_constructor(
    car_streets, street_points, tmp_path
):
    import pandas as pd
    import shapely

    options = dict(
        transport_mode="car",
        intersection_delays=True,
        profile="rush",
        parking=True,
        occupancy=2,
        vehicle_class="BEV",
        perspectives=["private", "societal"],
        geometries=True,
    )
    frame = TravelCostMatrix(car_streets, origins=street_points, **options)
    result = TravelCostMatrix.to_parquet(
        car_streets,
        origins=street_points,
        **options,
        output=tmp_path / "car.parquet",
        batch_size=2,
    )
    assert result.batches == 3
    read, expected = _read_aligned(tmp_path / "car.parquet", frame)
    # Geometry streams as plain WKB binary, deliberately without
    # GeoParquet footer metadata.
    expected["geometry"] = shapely.to_wkb(expected["geometry"].values)
    pd.testing.assert_frame_equal(read, expected)
    metadata = parquet.ParquetFile(tmp_path / "car.parquet").metadata.metadata
    assert not metadata or b"geo" not in metadata


def test_time_to_parquet_street_car_equals_constructor(
    car_streets, street_points, tmp_path
):
    import pandas as pd

    options = dict(
        transport_mode="car", intersection_delays=True, profile="rush", parking=True
    )
    frame = TravelTimeMatrix(car_streets, origins=street_points, **options)
    TravelTimeMatrix.to_parquet(
        car_streets,
        origins=street_points,
        **options,
        output=tmp_path / "cartime.parquet",
        batch_size=2,
    )
    read, expected = _read_aligned(tmp_path / "cartime.parquet", frame)
    pd.testing.assert_frame_equal(read, expected)


def test_to_parquet_refusals(network, car_streets, street_points, tmp_path):
    stops = _origin_stops(network, 4)
    with pytest.raises(NotImplementedError, match="street_policy"):
        TravelCostMatrix.to_parquet(
            network,
            origins=stops,
            **QUERY,
            street_policy=object(),
            output=tmp_path / "a.parquet",
        )
    with pytest.raises(NotImplementedError, match="resume"):
        TravelTimeMatrix.to_parquet(
            network, origins=stops, **QUERY, output=tmp_path / "b.parquet", resume=True
        )
    with pytest.raises(ValueError, match="batch_size"):
        TravelCostMatrix.to_parquet(
            network, origins=stops, **QUERY, output=tmp_path / "c.parquet", batch_size=0
        )
    with pytest.raises(ValueError, match="StreetNetwork car matrix"):
        TravelCostMatrix.to_parquet(
            network, origins=stops, **QUERY, parking=True, output=tmp_path / "d.parquet"
        )
    with pytest.raises(ValueError, match="no meaning for a street matrix"):
        TravelCostMatrix.to_parquet(
            car_streets,
            origins=street_points,
            transport_mode="walk",
            date="2022-02-22",
            output=tmp_path / "e.parquet",
        )
    with pytest.raises(ValueError, match="destinations apply to point origins"):
        TravelTimeMatrix.to_parquet(
            network,
            origins=stops,
            destinations=stops[:2],
            **QUERY,
            output=tmp_path / "f.parquet",
        )
    # Failed validation never leaves claimed outputs behind.
    assert list(tmp_path.iterdir()) == []


def test_invalid_queries_never_claim_outputs(network, tmp_path):
    # Deterministic validation runs before the output is claimed — the
    # directory form must not publish a manifest for a doomed query.
    stops = _origin_stops(network, 4)
    with pytest.raises(ValueError, match="pareto"):
        TravelCostMatrix.to_parquet(
            network,
            origins=stops,
            **QUERY,
            candidates="pareto",
            output=tmp_path / "pareto-shards",
        )
    with pytest.raises(ValueError, match="router"):
        TravelTimeMatrix.to_parquet(
            network,
            origins=stops,
            **QUERY,
            router="warp",
            output=tmp_path / "router-shards",
        )
    assert not (tmp_path / "pareto-shards" / "manifest.json").exists()
    assert not (tmp_path / "router-shards" / "manifest.json").exists()


def test_percentile_iterable_freezes_once(network, tmp_path):
    import pandas as pd

    stops = _origin_stops(network, 12)
    frame = TravelTimeMatrix(
        network, origins=stops, **QUERY, window=600, percentiles=[25.0, 75.0]
    )
    TravelTimeMatrix.to_parquet(
        network,
        origins=stops,
        **QUERY,
        window=600,
        percentiles=iter([25.0, 75.0]),
        output=tmp_path / "generator.parquet",
        batch_size=5,
    )
    read, expected = _read_aligned(tmp_path / "generator.parquet", frame)
    pd.testing.assert_frame_equal(read, expected)


def test_operation_distinguishes_fingerprints(network, tmp_path):
    stops = _origin_stops(network, 10)
    table = travel_cost_table(
        network, origins=stops, **QUERY, output=tmp_path / "table", batch_size=5
    )
    method = TravelCostMatrix.to_parquet(
        network, origins=stops, **QUERY, output=tmp_path / "method", batch_size=5
    )
    assert table.manifest["fingerprint"] != method.manifest["fingerprint"]


RSS_SCRIPT = textwrap.dedent("""
    import resource
    import sys
    import warnings

    warnings.filterwarnings("ignore")
    from cafein import TransportNetwork, travel_cost_table

    artifact, output, count, entry = (
        sys.argv[1],
        sys.argv[2],
        int(sys.argv[3]),
        sys.argv[4],
    )
    network = TransportNetwork.load(artifact)
    # The same 250 origins cycled: every batch computes identical rows,
    # so peak memory is comparable across origin counts.
    base = [stop for stop, _, _ in network.stops][:250]
    origins = (base * ((count + 249) // 250))[:count]
    if entry == "classmethod":
        from cafein import TravelCostMatrix

        stream = TravelCostMatrix.to_parquet
    else:
        stream = travel_cost_table
    result = stream(
        network,
        origins=origins,
        date="2022-02-22",
        departure="08:30:00",
        output=output,
        batch_size=250,
    )
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        peak *= 1024
    import pyarrow

    print(peak, result.rows, pyarrow.default_memory_pool().max_memory())
    """)


@pytest.mark.skipif(sys.platform == "win32", reason="resource module is Unix-only")
@pytest.mark.parametrize("entry", ["table", "classmethod"])
def test_streaming_peak_rss_is_flat_in_origins(network, tmp_path, entry):
    """The subprocess RSS guard: growing origins at fixed destinations
    and batch_size must not grow peak memory like the materialised
    table would (pyarrow pool counters print alongside for diagnosis).
    Runs against both ``travel_cost_table`` and the classmethod, so an
    implementation that materialises the constructor result and shards
    it afterwards cannot pass."""
    artifact = tmp_path / "network.cafein"
    network.save(artifact)

    def peak(count, name):
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                RSS_SCRIPT,
                str(artifact),
                str(tmp_path / name),
                str(count),
                entry,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        peak_bytes, rows, arrow_bytes = completed.stdout.split()[-3:]
        print(f"origins={count}: peak={peak_bytes} rows={rows} arrow={arrow_bytes}")
        return int(peak_bytes), int(rows)

    small_peak, small_rows = peak(500, "small.parquet")
    large_peak, large_rows = peak(4000, "large.parquet")
    assert large_rows == 8 * small_rows
    # The materialised rows for 4000 origins outweigh 500's by ~7M rows
    # (several hundred MB before any copies); the streamed run stays
    # approximately flat — the residual is allocator retention, well
    # under the materialisation signal.
    assert large_peak - small_peak < 250 * 1024 * 1024
