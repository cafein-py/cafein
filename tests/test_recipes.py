"""Recipe framework: loading, eager validation, input resolution, and running."""

import hashlib

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
    resolved = recipes.validate(_write(tmp_path))
    assert resolved["recipe"] == "exposure_tradeoff"
    # local paths resolve to absolute files beside the recipe
    assert resolved["inputs"]["streets"]["path"] == (tmp_path / "streets.pbf").resolve()
    assert resolved["inputs"]["exposure"]["no2"]["value"] == "concentration"
    assert resolved["parameters"] == {
        "mode": "bicycle",
        "objective_layer": "no2",
        "weights": [0.5, 1.0],
    }
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


def _two_route_recipe(tmp_path, monkeypatch, weights=(0.6,), mode="walk"):
    """A recipe on the synthetic two-corridor network (the OSM loader is
    stubbed, both modes permitted): value 1.0 over the short corridor, the 1.5x
    detour outside it."""
    yaml = pytest.importorskip("yaml")
    geopandas = pytest.importorskip("geopandas")
    from shapely.geometry import Point, box
    from test_exposure import _two_route_network

    from cafein._osm import BICYCLE, WALK
    from cafein.street_network import StreetNetwork

    monkeypatch.setattr(
        StreetNetwork,
        "from_osm",
        staticmethod(lambda *a, **k: _two_route_network(WALK | BICYCLE)),
    )
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
        "inputs": {
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
        },
        "outputs": {"table": "tradeoff.parquet"},
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
