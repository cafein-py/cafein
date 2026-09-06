"""Recipe framework: loading, eager validation, input resolution, and running."""

import functools
import hashlib
import json

import pytest

from cafein import recipes


def _valid_recipe():
    return {
        "recipe": "exposure_tradeoff",
        "version": 1,
        "inputs": {
            "streets": {"kind": "file", "path": "streets.pbf"},
            "exposure": {
                "no2": {
                    "kind": "raster",
                    "path": "no2.tif",
                    "value": "concentration",
                    "units": "ug/m3",
                }
            },
            "origins": {"kind": "vector", "path": "origins.geojson", "id_column": "id"},
            "destinations": {
                "kind": "vector",
                "path": "dests.geojson",
                "id_column": "id",
            },
        },
        "parameters": {
            "mode": "bicycle",
            "objective_layer": "no2",
            "weights": [0.5, 1.0],
        },
        "outputs": {"table": "tradeoff.parquet"},
    }


def _write(tmp_path, mutate=None):
    """A recipe file beside touched input files; `mutate` tweaks it first."""
    yaml = pytest.importorskip("yaml")
    for name in ("streets.pbf", "no2.tif", "origins.geojson", "dests.geojson"):
        (tmp_path / name).write_bytes(b"\x00")
    recipe = _valid_recipe()
    if mutate is not None:
        mutate(recipe)
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    return path


def test_validate_resolves_a_recipe(tmp_path):
    pytest.importorskip("yaml")
    # an object-valued keyword in its mapping spelling, built eagerly
    traveler = {"traveler": {"wheelchair": True}}
    resolved = recipes.validate(
        _write(tmp_path, _set(["parameters", "matrix"], traveler))
    )
    assert resolved["recipe"] == "exposure_tradeoff"
    # local paths resolve to absolute files beside the recipe
    assert resolved["inputs"]["streets"]["path"] == (tmp_path / "streets.pbf").resolve()
    assert resolved["inputs"]["exposure"]["no2"]["value"] == "concentration"
    assert resolved["parameters"] == {
        "mode": "bicycle",
        "objective_layer": "no2",
        "weights": [0.5, 1.0],
        "streets": {},
        "exposure": {},
        "matrix": {"traveler": {"wheelchair": True}},
    }
    # the record's view fills in the object's own defaults too
    effective = recipes._effective_parameters(resolved["parameters"])["matrix"]
    assert effective["traveler"]["wheelchair"] is True
    assert effective["traveler"]["unknown"] == "usable"
    assert resolved["outputs"]["table"] == "tradeoff.parquet"


def _set(path_keys, value):
    def mutate(recipe):
        target = recipe
        for key in path_keys[:-1]:
            target = target[key]
        target[path_keys[-1]] = value

    return mutate


def _delete(path_keys):
    def mutate(recipe):
        target = recipe
        for key in path_keys[:-1]:
            target = target[key]
        del target[path_keys[-1]]

    return mutate


@pytest.mark.parametrize(
    "mutate, match",
    [
        (_set(["recipe"], "nope"), "unknown recipe"),
        (_set(["version"], 2), "unsupported recipe version"),
        (_set(["requires"], {"cafein": ">=999"}), "requires cafein"),
        (_delete(["inputs", "origins"]), "missing 'origins'"),
        (_set(["inputs", "streets", "kind"], "place"), "kind must be one of"),
        (_delete(["inputs", "streets", "path"]), "needs 'path'"),
        (_set(["inputs", "streets", "name"], "helsinki"), "does not take 'name'"),
        (_set(["inputs", "streets", "path"], "missing.pbf"), "file not found"),
        (_set(["parameters", "mode"], "car"), "mode must be one of"),
        (_set(["parameters", "objective_layer"], "pm25"), "not a declared exposure"),
        (_set(["parameters", "weights"], []), "non-empty list"),
        (_set(["parameters", "weights"], [-1.0]), "non-negative"),
        (_set(["parameters", "weights"], [1.0, 0.5]), "strictly increasing"),
        (_set(["parameters", "weights"], [0.5, 0.5]), "strictly increasing"),
        (_set(["outputs", "table"], "/tmp/x.parquet"), "relative path"),
        (_set(["outputs", "table"], "../x.parquet"), "relative path"),
        (_set(["outputs", "table"], "C:/out/x.parquet"), "relative path"),
        (_set(["outputs", "table"], "..\\x.parquet"), "relative path"),
        (_set(["parameters", "mod"], "walk"), "unknown key"),
        (_set(["parameters", "matrix"], {"candidates": "time"}), "fixed by the recipe"),
        (_set(["parameters", "streets"], {"modes": ["walk"]}), "fixed by the recipe"),
        (_set(["parameters", "exposure"], {"treshold": 1}), "unknown keyword"),
        (_set(["parameters", "matrix"], "fast"), "must be a mapping of keywords"),
        (_set(["parameters", "matrix"], {"factors": "factors.csv"}), "written as"),
        (
            _set(
                ["parameters", "matrix"],
                {"factors": {"kind": "file", "path": "no2.tif"}},
            ),
            "self-contained",
        ),
        (
            _set(
                ["parameters", "matrix"], {"max_rides": {"kind": "file", "path": "x"}}
            ),
            "does not take a mapping",
        ),
        (
            _set(["parameters", "matrix"], {"street_policy": {"access": {"walk": 1}}}),
            "does not take a mapping",
        ),
        (_set(["parameters", "matrix"], {"traveler": {"foo": 1}}), "traveler.*foo"),
        (_set(["parameters", "matrix"], {"traveler": "wheelchair"}), "mapping of Trav"),
        (
            _set(
                ["parameters", "streets"], {"dem": {"kind": "file", "path": "no.tif"}}
            ),
            "file not found",
        ),
        (
            _set(["inputs", "exposure", "no2", "path"], "origins.geojson"),
            "self-contained",
        ),
        (_set(["bogus"], 1), "unknown key"),
        (_set(["inputs", "bogus"], {}), "unknown key"),
        (_set(["outputs", "table"], "C:x.parquet"), "relative path"),
        (_set(["outputs", "table"], "."), "must name a file"),
        (_set(["outputs", "table"], "out/"), "must name a file"),
        (_set(["outputs", "table"], "tradeoff.arrow"), "must end with .parquet"),
        (_set(["outputs", "table"], "tradeoff.PARQUET"), "must end with .parquet"),
        (_set(["outputs", "table"], "x" * 240 + ".parquet"), "within 247 bytes"),
        (_set(["outputs", "table"], "ä" * 120 + ".parquet"), "within 247 bytes"),
        (_set(["outputs", "table"], "CON.parquet"), "not a portable file name"),
        (_set(["outputs", "table"], "a?.parquet"), "not a portable file name"),
        (_set(["outputs", "table"], "out./t.parquet"), "not a portable file name"),
        (_set(["outputs", "table"], "d" * 256 + "/t.parquet"), "not a portable"),
        (_set(["inputs", "origins", "id_column"], []), "'id_column' must be a string"),
        (_set(["inputs", "exposure", "no2", "value"], True), "must be a column name"),
        (_set(["inputs", "exposure", ""], {"kind": "raster"}), "non-empty strings"),
        (
            _set(
                ["inputs", "exposure", "rasterize"],
                {"kind": "raster", "path": "no2.tif", "value": "c"},
            ),
            "reserved name",
        ),
        (
            _set(
                ["inputs", "exposure", "self"],
                {"kind": "raster", "path": "no2.tif", "value": "c"},
            ),
            "reserved name",
        ),
    ],
)
def test_validate_refuses_by_name(tmp_path, mutate, match):
    pytest.importorskip("yaml")
    with pytest.raises(ValueError, match=match):
        recipes.validate(_write(tmp_path, mutate))


def test_validate_refuses_multi_file_format(tmp_path):
    pytest.importorskip("yaml")
    (tmp_path / "od.shp").write_bytes(b"\x00")
    path = _write(tmp_path, _set(["inputs", "origins", "path"], "od.shp"))
    with pytest.raises(ValueError, match="self-contained"):
        recipes.validate(path)


def test_validate_allows_optional_units(tmp_path):
    pytest.importorskip("yaml")
    resolved = recipes.validate(
        _write(tmp_path, _delete(["inputs", "exposure", "no2", "units"]))
    )
    assert "units" not in resolved["inputs"]["exposure"]["no2"]


def test_load_refuses_duplicate_keys(tmp_path):
    pytest.importorskip("yaml")
    path = tmp_path / "dup.yaml"
    path.write_text("recipe: exposure_tradeoff\nversion: 1\nversion: 2\n")
    with pytest.raises(ValueError, match="duplicate key"):
        recipes.validate(path)


def test_run_writes_the_tradeoff_table_and_provenance(tmp_path, kantakaupunki_pbf):
    """End to end on the Helsinki streets: the exposure integral, the fastest
    baseline, and a provenance record that pins the inputs actually loaded."""
    yaml = pytest.importorskip("yaml")
    geopandas = pytest.importorskip("geopandas")
    pytest.importorskip("pyarrow")
    from shapely.geometry import Point, box

    from cafein import recipes

    # a uniform layer over the corridor, written as a vector file
    zone = geopandas.GeoDataFrame(
        {"level": [61.0]}, geometry=[box(24.90, 60.15, 24.99, 60.20)], crs="EPSG:4326"
    )
    zone.to_file(tmp_path / "no2.geojson", driver="GeoJSON")
    for name, lon, lat in (
        ("origins", 24.9320, 60.1690),
        ("destinations", 24.9520, 60.1795),
    ):
        geopandas.GeoDataFrame(
            {"pid": [name[0]]}, geometry=[Point(lon, lat)], crs="EPSG:4326"
        ).to_file(tmp_path / f"{name}.geojson", driver="GeoJSON")
    recipe = {
        "recipe": "exposure_tradeoff",
        "inputs": {
            "streets": {"kind": "file", "path": str(kantakaupunki_pbf)},
            "exposure": {
                "no2": {
                    "kind": "vector",
                    "path": "no2.geojson",
                    "value": "level",
                    "units": "ug/m3",
                }
            },
            "origins": {
                "kind": "vector",
                "path": "origins.geojson",
                "id_column": "pid",
            },
            "destinations": {
                "kind": "vector",
                "path": "destinations.geojson",
                "id_column": "pid",
            },
        },
        "parameters": {
            "mode": "bicycle",
            "objective_layer": "no2",
            "weights": [0.5, 1.0],
        },
        "outputs": {"table": "tradeoff.parquet"},
    }
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    out = tmp_path / "out"

    frame = recipes.run(path, out_dir=out)

    assert (out / "tradeoff.parquet").is_file()
    for column in (
        "no2_mean",
        "no2_exposure",
        "travel_time",
        "fastest",
        "travel_time_delta",
        "no2_exposure_delta",
    ):
        assert column in frame.columns
    # option 0 is the true fastest: zero deltas, and no route is faster
    fastest = frame[frame["fastest"]]
    assert len(fastest) >= 1
    assert (fastest["travel_time_delta"] == 0).all()
    assert (frame["travel_time_delta"] >= 0).all()
    # the integral identity: exposure = mean concentration × on-street minutes
    # (snap connectors carry no sampled concentration, so on-street ≤ total)
    covered = frame["no2_mean"].notna()
    assert covered.any()
    assert (frame["on_street_time"] <= frame["travel_time"] + 1e-9).all()
    assert frame.loc[covered, "no2_exposure"].tolist() == pytest.approx(
        (frame.loc[covered, "no2_mean"] * frame.loc[covered, "on_street_time"]).tolist()
    )
    record = json.loads((out / "tradeoff.provenance.json").read_text())
    assert record["versions"]["cafein"]
    # the compiled core is recorded separately, and no promised dependency is
    # ever silently dropped from the record
    versions = record["versions"]
    for name in (
        "cafein",
        "cafein_core",
        "geopandas",
        "pyrosm",
        "numpy",
        "shapely",
        "pandas",
        "pyarrow",
    ):
        # real version identifiers, never the "unavailable" fallback or empty
        assert versions[name] and versions[name] != "unavailable"
        assert versions[name][0].isdigit(), (name, versions[name])
    assert (
        record["inputs"]["streets"]["sha256"]
        == hashlib.sha256(kantakaupunki_pbf.read_bytes()).hexdigest()
    )
    assert record["resolved"]["parameters"]["objective_layer"] == "no2"
    assert record["invocation"]["entry_point"] == "python"
    assert record["outputs"]["table"].endswith("tradeoff.parquet")


def test_run_refuses_an_output_that_overwrites_an_input(tmp_path):
    """A .parquet target that is the same file as an input (a hard link here)
    is refused before anything runs."""
    import os

    pytest.importorskip("yaml")
    from cafein import recipes

    path = _write(tmp_path, _set(["outputs", "table"], "alias.parquet"))
    os.link(tmp_path / "no2.tif", tmp_path / "alias.parquet")
    with pytest.raises(ValueError, match="would overwrite an input"):
        recipes.run(path, out_dir=tmp_path)


def _two_route_recipe(
    tmp_path,
    monkeypatch,
    weights=(0.6,),
    mode="walk",
    inputs=None,
    table=None,
    parameters=None,
    received=None,
):
    """A recipe on the synthetic two-corridor network (the OSM loader is
    stubbed, both modes permitted): value 1.0 over the short corridor, the 1.5x
    detour outside it. ``inputs`` replaces the file-kind inputs section;
    ``received`` collects the keywords the stubbed loader is called with."""
    yaml = pytest.importorskip("yaml")
    geopandas = pytest.importorskip("geopandas")
    from shapely.geometry import Point, box
    from test_exposure import _two_route_network

    from cafein._osm import BICYCLE, WALK
    from cafein.street_network import StreetNetwork

    @functools.wraps(StreetNetwork.from_osm)  # keeps the real signature
    def from_osm(*args, **kwargs):
        if received is not None:
            received.update(kwargs)
        return _two_route_network(WALK | BICYCLE)

    monkeypatch.setattr(StreetNetwork, "from_osm", staticmethod(from_osm))
    (tmp_path / "streets.pbf").write_bytes(b"\x00")
    geopandas.GeoDataFrame(
        {"level": [1.0]},
        geometry=[box(24.9290, 60.1690, 24.9364, 60.17005)],
        crs="EPSG:4326",
    ).to_file(tmp_path / "no2.geojson", driver="GeoJSON")
    for name, lon in (("origins", 24.9300), ("destinations", 24.9354)):
        geopandas.GeoDataFrame(
            {"id": [name[0]]}, geometry=[Point(lon, 60.1700)], crs="EPSG:4326"
        ).to_file(tmp_path / f"{name}.geojson", driver="GeoJSON")
    recipe = {
        "recipe": "exposure_tradeoff",
        "inputs": inputs
        or {
            "streets": {"kind": "file", "path": "streets.pbf"},
            "exposure": {
                "no2": {"kind": "vector", "path": "no2.geojson", "value": "level"}
            },
            "origins": {"kind": "vector", "path": "origins.geojson", "id_column": "id"},
            "destinations": {
                "kind": "vector",
                "path": "destinations.geojson",
                "id_column": "id",
            },
        },
        "parameters": {
            "mode": mode,
            "objective_layer": "no2",
            "weights": list(weights),
            **(parameters or {}),
        },
        "outputs": {"table": table or "tradeoff.parquet"},
    }
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    return path


@pytest.mark.parametrize("mode", ["walk", "bicycle"])
def test_pipeline_reports_the_cleaner_slower_alternative(tmp_path, monkeypatch, mode):
    """A weight past the flip point finds the detour: a distinct swept row,
    slower than the fastest, with lower exposure — for either street mode. The
    pipeline also returns the checksum of every input it read."""
    from cafein import recipes

    resolved = recipes.validate(_two_route_recipe(tmp_path, monkeypatch, mode=mode))
    frame, checksums = recipes._RECIPES["exposure_tradeoff"].run(resolved)

    fastest = frame[frame["fastest"]]
    alternatives = frame[~frame["fastest"]]
    assert len(fastest) == 1 and len(alternatives) == 1
    assert (fastest["travel_time_delta"] == 0).all()
    detour = alternatives.iloc[0]
    assert detour["sweep_weight"] == pytest.approx(0.6)
    assert detour["travel_time_delta"] > 0
    assert detour["no2_exposure_delta"] < 0
    assert detour["no2_mean"] == pytest.approx(0.0)
    # the integral identity over the on-street minutes, and one checksum per input
    assert frame["no2_exposure"].tolist() == pytest.approx(
        (frame["no2_mean"] * frame["on_street_time"]).tolist()
    )
    assert set(checksums) == {"streets", "exposure.no2", "origins", "destinations"}
    assert all(len(c["sha256"]) == 64 for c in checksums.values())


def test_snapshots_are_collision_free_and_hash_what_they_copy(tmp_path):
    from cafein import recipes

    a, b = tmp_path / "a.b.geojson", tmp_path / "a_b.geojson"
    a.write_bytes(b"alpha")
    b.write_bytes(b"beta")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checksums = {}
    first = recipes._snapshot(0, "exposure.a.b", {"path": a}, run_dir, checksums)
    second = recipes._snapshot(1, "exposure.a_b", {"path": b}, run_dir, checksums)
    # dotted and underscored layer names no longer share a snapshot file
    assert first["path"] != second["path"]
    assert (
        first["path"].read_bytes() == b"alpha"
        and second["path"].read_bytes() == b"beta"
    )
    assert checksums["exposure.a.b"]["sha256"] == hashlib.sha256(b"alpha").hexdigest()
    assert checksums["exposure.a_b"]["sha256"] == hashlib.sha256(b"beta").hexdigest()
    # a long, many-dotted source name still yields a short, bounded snapshot name
    long_name = tmp_path / ("x." * 100 + "geojson")
    long_name.write_bytes(b"gamma")
    third = recipes._snapshot(
        2, "exposure.long", {"path": long_name}, run_dir, checksums
    )
    assert third["path"].name == "input-02.geojson"


@pytest.mark.parametrize(
    "link, points_at, table",
    [
        ("link", "", "link/tradeoff.parquet"),
        ("tradeoff.provenance.json", "record.json", "tradeoff.parquet"),
    ],
)
def test_run_refuses_a_symlinked_output_component(tmp_path, link, points_at, table):
    """A symlinked directory on the table's path, or a symlinked existing record,
    is refused before anything runs."""
    pytest.importorskip("yaml")
    from cafein import recipes

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "record.json").write_text("{}")
    out = tmp_path / "out"
    out.mkdir()
    (out / link).symlink_to(elsewhere / points_at)
    path = _write(tmp_path, _set(["outputs", "table"], table))
    with pytest.raises(ValueError, match="is a symlink"):
        recipes.run(path, out_dir=out)


@pytest.mark.parametrize("table_landed", [False, True])
def test_run_recovers_the_pair_from_disk_state(tmp_path, monkeypatch, table_landed):
    """An interruption around the table move restores the previous record when
    the new table did not land and keeps the new record when it did; either way
    the published record matches the published table."""
    import os

    pytest.importorskip("pyarrow")
    from cafein import recipes

    class Interrupt(BaseException):
        pass

    out = tmp_path / "out"
    recipes.run(_two_route_recipe(tmp_path, monkeypatch), out_dir=out)
    record, table = out / "tradeoff.provenance.json", out / "tradeoff.parquet"
    first = json.loads(record.read_text())
    real_replace = os.replace

    def interrupted(src, dst):
        if str(dst) == str(table):
            if table_landed:
                real_replace(src, dst)
            raise Interrupt()
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", interrupted)
    with pytest.raises(Interrupt):
        recipes.run(_two_route_recipe(tmp_path, monkeypatch), out_dir=out)
    after = json.loads(record.read_text())
    assert (after == first) is not table_landed
    assert after["outputs"]["sha256"] == hashlib.sha256(table.read_bytes()).hexdigest()
    assert not list(out.glob(".cafein-recipe-*")) and not list(out.glob(".*.lock"))


def test_run_refuses_to_publish_over_a_live_lock(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    from cafein import recipes

    out = tmp_path / "out"
    out.mkdir()
    (out / ".tradeoff.parquet.lock").write_bytes(b"")
    with pytest.raises(ValueError, match="another run is publishing"):
        recipes.run(_two_route_recipe(tmp_path, monkeypatch), out_dir=out)
    # nothing was published under the lock
    assert not (out / "tradeoff.parquet").exists()
    assert not (out / "tradeoff.provenance.json").exists()


def test_validate_refuses_a_geopackage_with_a_live_wal(tmp_path):
    pytest.importorskip("yaml")
    from cafein import recipes

    (tmp_path / "od.gpkg").write_bytes(b"\x00")
    (tmp_path / "od.gpkg-wal").write_bytes(b"\x00")
    path = _write(tmp_path, _set(["inputs", "origins", "path"], "od.gpkg"))
    with pytest.raises(ValueError, match="write-ahead log"):
        recipes.validate(path)


@pytest.mark.parametrize(
    "name, match",
    [("NO2", "lowercase identifier"), ("travel_time", "collides with")],
)
def test_validate_applies_exposures_layer_naming_rule(tmp_path, name, match):
    """Names Exposure would reject are refused at validation, before any data."""
    pytest.importorskip("yaml")
    from cafein import recipes

    spec = {"kind": "raster", "path": "no2.tif", "value": "c"}
    with pytest.raises(ValueError, match=match):
        recipes.validate(_write(tmp_path, _set(["inputs", "exposure", name], spec)))


_SAMPLE_INPUTS = {
    "streets": {"kind": "sample", "name": "tworoute.osm_pbf"},
    "exposure": {"no2": {"kind": "sample", "name": "tworoute.no2", "value": "level"}},
    "origins": {"kind": "sample", "name": "tworoute.origins", "id_column": "id"},
    "destinations": {
        "kind": "sample",
        "name": "tworoute.destinations",
        "id_column": "id",
    },
}


def _fake_sampledata(monkeypatch, tmp_path, tamper=None):
    """Stand in for ``cafein.sampledata`` with a ``tworoute`` region whose pinned
    assets are the two-corridor files in ``tmp_path``, served without a network;
    ``tamper`` names an asset whose pin is made wrong."""
    import dataclasses
    import sys
    import types

    @dataclasses.dataclass
    class Asset:
        name: str
        url: str
        sha256: str
        size: int
        license: str = ""
        attribution: str = ""
        source_stamp: str = ""
        release: str = ""

    parent = types.ModuleType("cafein.sampledata")
    parent.Asset = Asset
    parent.fetch = lambda asset, region: tmp_path / asset.name
    region = types.ModuleType("cafein.sampledata.tworoute")
    region.REGION = "tworoute"
    region.metadata = {}
    files = {"osm_pbf": "streets.pbf", "no2": "no2.geojson"}
    files.update(origins="origins.geojson", destinations="destinations.geojson")
    for key, filename in files.items():
        data = (tmp_path / filename).read_bytes()
        region.metadata[key] = {
            "name": filename,
            "url": "",
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
            "license": "",
            "attribution": "",
            "source_stamp": "",
            "release": "tworoute-2026.09",
        }
    if tamper is not None:
        region.metadata[tamper]["sha256"] = "0" * 64
    monkeypatch.setitem(sys.modules, "cafein.sampledata", parent)
    monkeypatch.setitem(sys.modules, "cafein.sampledata.tworoute", region)
    # an importable submodule that is no region (no metadata table)
    module = types.ModuleType("cafein.sampledata.notaregion")
    monkeypatch.setitem(sys.modules, "cafein.sampledata.notaregion", module)


def test_run_on_sample_inputs_records_the_pins(tmp_path, monkeypatch):
    """Sample inputs validate offline against the region's pins, fetch through
    the sampledata client at run time, and land in the record with their
    release and pinned digest."""
    pytest.importorskip("pyarrow")
    from cafein import recipes

    path = _two_route_recipe(tmp_path, monkeypatch, inputs=_SAMPLE_INPUTS)
    _fake_sampledata(monkeypatch, tmp_path)
    resolved = recipes.validate(path)
    assert resolved["inputs"]["streets"]["path"] is None
    pin = resolved["inputs"]["streets"]["sample"]
    assert pin["asset"] == "tworoute.osm_pbf" and len(pin["sha256"]) == 64

    frame = recipes.run(path, out_dir=tmp_path / "out")

    assert len(frame) == 2 and frame["fastest"].sum() == 1
    record = json.loads((tmp_path / "out" / "tradeoff.provenance.json").read_text())
    for role in ("streets", "exposure.no2", "origins", "destinations"):
        entry = record["inputs"][role]
        assert entry["sample"]["release"] == "tworoute-2026.09"
        assert entry["sha256"] == entry["sample"]["sha256"]
    # the record's resolved recipe shows where the fetched asset was read from
    assert record["resolved"]["inputs"]["streets"]["path"].endswith("streets.pbf")


def test_run_refuses_an_output_that_overwrites_a_fetched_sample(tmp_path, monkeypatch):
    """The overwrite guard sees a sample's fetched path: a table hard-linked to
    the cached asset is refused before the analysis."""
    import os

    pytest.importorskip("pyarrow")
    from cafein import recipes

    path = _two_route_recipe(
        tmp_path, monkeypatch, inputs=_SAMPLE_INPUTS, table="alias.parquet"
    )
    _fake_sampledata(monkeypatch, tmp_path)
    os.link(tmp_path / "streets.pbf", tmp_path / "alias.parquet")
    with pytest.raises(ValueError, match="would overwrite an input"):
        recipes.run(path, out_dir=tmp_path)


def test_run_refuses_a_sample_that_differs_from_its_pin(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    from cafein import recipes

    path = _two_route_recipe(tmp_path, monkeypatch, inputs=_SAMPLE_INPUTS)
    _fake_sampledata(monkeypatch, tmp_path, tamper="osm_pbf")
    with pytest.raises(ValueError, match="not its pin"):
        recipes.run(path, out_dir=tmp_path / "out")


@pytest.mark.parametrize(
    "role, spec, match",
    [
        ("streets", {"kind": "sample", "name": "tworoute"}, "'<region>.<asset>'"),
        (
            "streets",
            {"kind": "sample", "name": "nowhere.osm_pbf"},
            "unknown sample region",
        ),
        (
            "streets",
            {"kind": "sample", "name": "notaregion.osm_pbf"},
            "unknown sample region",
        ),
        (
            "streets",
            {"kind": "sample", "name": "tworoute.nope"},
            "unknown sample asset",
        ),
        ("streets", {"kind": "sample", "name": "tworoute.no2"}, "not a self-contained"),
        (
            "origins",
            {
                "kind": "sample",
                "name": "tworoute.origins",
                "path": "x",
                "id_column": "id",
            },
            "does not take 'path'",
        ),
    ],
)
def test_validate_refuses_bad_sample_inputs(tmp_path, monkeypatch, role, spec, match):
    pytest.importorskip("yaml")
    from cafein import recipes

    inputs = {**_SAMPLE_INPUTS, role: spec}
    path = _two_route_recipe(tmp_path, monkeypatch, inputs=inputs)
    _fake_sampledata(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match=match):
        recipes.validate(path)


def test_validate_resolves_helsinki_sample_pins_offline(tmp_path):
    """Against the real registry: the pins resolve without a download and
    reconstruct the client's own asset type."""
    helsinki = pytest.importorskip("cafein.sampledata.helsinki")
    pytest.importorskip("yaml")
    from cafein import recipes
    from cafein.sampledata import Asset

    if "air_quality" not in helsinki.metadata:
        pytest.skip("this sampledata release carries no air-quality layer")
    inputs = {
        "streets": {"kind": "sample", "name": "helsinki.osm_pbf"},
        "exposure": {
            "no2": {
                "kind": "sample",
                "name": "helsinki.air_quality",
                "value": "NO2Concentration",
                "units": "ug/m3",
            }
        },
        "origins": {
            "kind": "sample",
            "name": "helsinki.poi_library",
            "id_column": "osm_id",
        },
        "destinations": {
            "kind": "sample",
            "name": "helsinki.poi_university",
            "id_column": "osm_id",
        },
    }
    resolved = recipes.validate(_write(tmp_path, _set(["inputs"], inputs)))
    for role, key in (("streets", "osm_pbf"), ("origins", "poi_library")):
        sample = resolved["inputs"][role]["sample"]
        assert sample["sha256"] == helsinki.metadata[key]["sha256"]
        Asset(**{field: sample[field] for field in Asset.__dataclass_fields__})
    assert resolved["inputs"]["exposure"]["no2"]["value"] == "NO2Concentration"


def test_run_applies_grouped_parameters_and_records_defaults(tmp_path, monkeypatch):
    """Group keywords reach their objects (an exposure threshold adds its
    column, a file-valued factor table is snapshotted, hashed, and applied),
    and the record lists every group's effective value, defaults included."""
    pytest.importorskip("pyarrow")
    from cafein import recipes

    (tmp_path / "factors.csv").write_text(
        "street_mode,vehicle_class,service_model,"
        "vehicle,fuel,infrastructure,operations\n"
        "bicycle,conventional,private,100,0,0,0\n"
    )
    parameters = {
        "exposure": {"thresholds": {"no2": 0.5}},
        "matrix": {
            "factors": {"kind": "file", "path": "factors.csv"},
            "geometries": True,
        },
    }
    path = _two_route_recipe(
        tmp_path, monkeypatch, mode="bicycle", parameters=parameters
    )

    frame = recipes.run(path, out_dir=tmp_path / "out")

    assert any(column.startswith("no2_minutes_above") for column in frame.columns)
    # geometries publish as GeoParquet that reads back as lines
    geopandas = pytest.importorskip("geopandas")
    assert isinstance(frame, geopandas.GeoDataFrame)
    published = geopandas.read_parquet(tmp_path / "out" / "tradeoff.parquet")
    assert published.geometry.geom_type.eq("LineString").all()
    # each row keeps its own line: the detour's is the longer one
    length = published.geometry.to_crs("EPSG:3067").length
    assert length[~published["fastest"]].min() > length[published["fastest"]].max()
    # 100 g/km over the network metres: the file's row, not the shipped one
    assert frame["emissions"].tolist() == pytest.approx(
        (frame["network_distance_m"] / 10).tolist()
    )
    record = json.loads((tmp_path / "out" / "tradeoff.provenance.json").read_text())
    assert len(record["inputs"]["parameters.matrix.factors"]["sha256"]) == 64
    effective = record["resolved"]["parameters"]
    assert effective["exposure"]["thresholds"] == {"no2": 0.5}
    assert effective["exposure"]["rasterize"] == 1.0
    assert effective["matrix"]["max_rides"] == 8
    assert effective["matrix"]["factors"]["path"].endswith("factors.csv")


def test_provenance_values_are_canonical():
    import pathlib

    from cafein import recipes

    value = {"a": (1, 2), "p": pathlib.Path("x"), "s": frozenset({2, 1}), "n": None}
    assert recipes._canonical(value, "t") == {
        "a": [1, 2],
        "p": "x",
        "s": [1, 2],
        "n": None,
    }
    with pytest.raises(ValueError, match="t.bad.*cannot be recorded"):
        recipes._canonical({"bad": object()}, "t")


@pytest.mark.parametrize("layer", ["a", None])
def test_run_reads_a_vector_parameter_by_layer(tmp_path, monkeypatch, layer):
    """A vector-valued parameter reaches its object as the named GeoPackage
    layer; a multi-layer file without ``layer:`` is refused by name."""
    geopandas = pytest.importorskip("geopandas")
    pytest.importorskip("pyarrow")
    from shapely.geometry import box

    from cafein import recipes

    zones = tmp_path / "zones.gpkg"
    for name, rows in (("a", 1), ("b", 2)):
        geopandas.GeoDataFrame(
            {"n": range(rows)},
            geometry=[box(24.9, 60.1, 25.0, 60.2)] * rows,
            crs="EPSG:4326",
        ).to_file(zones, layer=name, driver="GPKG")
    spec = {"kind": "file", "path": "zones.gpkg", **({"layer": layer} if layer else {})}
    received = {}
    path = _two_route_recipe(
        tmp_path,
        monkeypatch,
        parameters={"streets": {"urban_areas": spec}},
        received=received,
    )
    if layer is None:
        with pytest.raises(ValueError, match="holds several layers"):
            recipes.run(path, out_dir=tmp_path / "out")
        return
    recipes.run(path, out_dir=tmp_path / "out")
    assert isinstance(received["urban_areas"], geopandas.GeoDataFrame)
    assert len(received["urban_areas"]) == 1
