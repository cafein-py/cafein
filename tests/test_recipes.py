"""Recipe framework: loading, eager validation, and input resolution."""

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
        (_set(["inputs", "origins", "id_column"], []), "'id_column' must be a string"),
        (_set(["inputs", "exposure", "no2", "value"], True), "must be a column name"),
        (_set(["inputs", "exposure", ""], {"kind": "raster"}), "non-empty strings"),
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
