"""Declarative analysis recipes.

A *recipe* is a user-authored YAML document describing a whole analysis
pipeline; cafein runs it, so the file is the reproducible method. This module
is the loader, the typed-recipe registry, input resolution, and eager
validation. Executing a recipe type (the ``exposure_tradeoff`` pipeline) and
the provenance record it writes arrive with :func:`run`; this module ships the
framework and :func:`validate`.
"""

from __future__ import annotations

import pathlib

from cafein._validate import choice, non_negative_finite

#: Self-contained single-file formats per input role — so a content checksum
#: (at run time) covers the whole dataset; a shapefile's sidecars would not be.
_STREET_SUFFIXES = (".pbf",)
_RASTER_SUFFIXES = (".tif", ".tiff")
_VECTOR_SUFFIXES = (".gpkg", ".geojson", ".json")

#: The recipe-schema versions this cafein understands.
_SCHEMA_VERSIONS = (1,)


def _load_yaml(path):
    """Parse a recipe file into a mapping, rejecting duplicate keys (PyYAML's
    default silently keeps the last), or raise by name."""
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - trivial guard
        raise ImportError(
            "reading a recipe needs the optional PyYAML dependency "
            "(pip install cafein[yaml] or pyyaml)"
        ) from error

    class _StrictLoader(yaml.SafeLoader):
        pass

    def _no_duplicates(loader, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"{path}: duplicate key '{key}' in the recipe")
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    _StrictLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates
    )
    document = yaml.load(pathlib.Path(path).read_text(encoding="utf-8"), _StrictLoader)
    if not isinstance(document, dict):
        raise ValueError(f"{path}: a recipe must be a YAML mapping")
    return document


def _reject_foreign(mapping, allowed, where):
    for key in mapping:
        if key not in allowed:
            raise ValueError(f"{where}: unknown key '{key}'")


def _check_requires(requires):
    """Enforce an optional ``requires: {cafein: <specifier>}`` version pin."""
    if requires is None:
        return
    if not isinstance(requires, dict):
        raise ValueError("requires: must be a mapping of package to version specifier")
    _reject_foreign(requires, {"cafein"}, "requires")
    spec = requires.get("cafein")
    if spec is None:
        return
    from packaging.specifiers import InvalidSpecifier, SpecifierSet
    from packaging.version import Version

    import cafein

    try:
        specifier = SpecifierSet(str(spec))
    except InvalidSpecifier as error:
        raise ValueError(
            f"requires.cafein: not a version specifier: {spec!r}"
        ) from error
    if Version(cafein.__version__) not in specifier:
        raise ValueError(
            f"this recipe requires cafein {spec}, but {cafein.__version__} is "
            "installed; install a matching cafein or update the recipe"
        )


def _suffix_ok(path, suffixes):
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in suffixes)


# Per-role kind → (required keys, optional keys, self-contained suffixes).
_STREET_KEYS = {"file": (("path",), (), _STREET_SUFFIXES)}
_EXPOSURE_KEYS = {
    "raster": (("path", "value"), ("units",), _RASTER_SUFFIXES),
    "vector": (("path", "value"), ("units", "layer"), _VECTOR_SUFFIXES),
}
_OD_KEYS = {"vector": (("path", "id_column"), ("layer",), _VECTOR_SUFFIXES)}


def _resolve_source(role, layer, spec, recipe_dir, *, keys):
    """Validate + resolve one `kind:`-tagged file input, or raise by name.

    Local paths resolve relative to the recipe's directory, so a recipe + its
    data move together; the file must exist and be a self-contained format.
    """
    where = f"inputs.{role}" + (f".{layer}" if layer else "")
    if not isinstance(spec, dict):
        raise ValueError(f"{where}: must be a mapping with a 'kind'")
    kind = spec.get("kind")
    if kind is None:
        raise ValueError(f"{where}: missing 'kind' (one of {', '.join(keys)})")
    choice(f"{where}.kind", kind, tuple(keys))
    required, optional, suffixes = keys[kind]
    allowed = {"kind", *required, *optional}
    for foreign in set(spec) - allowed:
        raise ValueError(f"{where}: kind '{kind}' does not take '{foreign}'")
    for key in required:
        if spec.get(key) is None:
            raise ValueError(f"{where}: kind '{kind}' needs '{key}'")
    if not isinstance(spec["path"], str):
        raise ValueError(f"{where}: 'path' must be a string")
    for key in ("id_column", "units", "layer"):
        if key in spec and not isinstance(spec[key], str):
            raise ValueError(f"{where}: '{key}' must be a string")
    if "value" in spec and (
        isinstance(spec["value"], bool) or not isinstance(spec["value"], (str, int))
    ):
        raise ValueError(f"{where}: 'value' must be a column name or a band")
    path = (recipe_dir / spec["path"]).resolve()
    if not path.is_file():
        raise ValueError(f"{where}: file not found: {path}")
    if not _suffix_ok(path, suffixes):
        raise ValueError(
            f"{where}: '{path.name}' is not a self-contained {'/'.join(suffixes)} "
            "file (multi-file formats such as shapefiles are not supported)"
        )
    resolved = {"kind": kind, "path": path}
    for key in (*required, *optional):
        if key != "path" and key in spec:
            resolved[key] = spec[key]
    return resolved


def _resolve_inputs(inputs, recipe_dir):
    """Resolve every declared input of the exposure_tradeoff recipe."""
    if not isinstance(inputs, dict):
        raise ValueError("inputs: must be a mapping")
    _reject_foreign(
        inputs, {"streets", "exposure", "origins", "destinations"}, "inputs"
    )
    for required in ("streets", "exposure", "origins", "destinations"):
        if inputs.get(required) is None:
            raise ValueError(f"inputs: missing '{required}'")
    resolved = {
        "streets": _resolve_source(
            "streets", None, inputs["streets"], recipe_dir, keys=_STREET_KEYS
        )
    }
    exposure = inputs["exposure"]
    if not isinstance(exposure, dict) or not exposure:
        raise ValueError("inputs.exposure: must be a non-empty mapping of layers")
    for name in exposure:
        if not isinstance(name, str) or not name:
            raise ValueError("inputs.exposure: layer names must be non-empty strings")
    resolved["exposure"] = {
        name: _resolve_source("exposure", name, spec, recipe_dir, keys=_EXPOSURE_KEYS)
        for name, spec in exposure.items()
    }
    for role in ("origins", "destinations"):
        resolved[role] = _resolve_source(
            role, None, inputs[role], recipe_dir, keys=_OD_KEYS
        )
    return resolved


def _resolve_parameters(parameters, exposure_layers):
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, dict):
        raise ValueError("parameters: must be a mapping")
    _reject_foreign(parameters, {"mode", "objective_layer", "weights"}, "parameters")
    mode = parameters.get("mode", "bicycle")
    choice("parameters.mode", mode, ("bicycle", "walk"))
    objective = parameters.get("objective_layer")
    if objective is None:
        raise ValueError("parameters: missing 'objective_layer'")
    if objective not in exposure_layers:
        raise ValueError(
            f"parameters.objective_layer '{objective}' is not a declared "
            f"exposure layer ({', '.join(exposure_layers)})"
        )
    weights = parameters.get("weights")
    if not isinstance(weights, (list, tuple)) or not weights:
        raise ValueError("parameters.weights: must be a non-empty list of weights")
    weights = [non_negative_finite("parameters.weights", w) for w in weights]
    return {"mode": mode, "objective_layer": objective, "weights": weights}


def _resolve_outputs(outputs):
    if not isinstance(outputs, dict) or not outputs.get("table"):
        raise ValueError("outputs: must declare a 'table' path")
    _reject_foreign(outputs, {"table"}, "outputs")
    if not isinstance(outputs["table"], str):
        raise ValueError("outputs.table: must be a string path")
    table = outputs["table"]
    posix = pathlib.PurePosixPath(table)
    windows = pathlib.PureWindowsPath(table)
    if (
        "\\" in table
        or bool(windows.drive)
        or posix.is_absolute()
        or windows.is_absolute()
        or ".." in posix.parts
        or ".." in windows.parts
    ):
        raise ValueError(
            f"outputs.table '{table}' must be a relative path within the output "
            "root (no absolute paths or '..')"
        )
    return {"table": table}


def validate(path):
    """Validate a recipe file without running it, resolving its inputs.

    Returns the resolved recipe (a mapping the runner consumes); raises
    ``ValueError`` by name on any schema, kind, parameter, output, or
    resolvability problem. Reads inputs' metadata but performs no routing and
    writes nothing.
    """
    path = pathlib.Path(path).resolve()
    document = _load_yaml(path)
    _reject_foreign(
        document,
        {"recipe", "version", "requires", "inputs", "parameters", "outputs"},
        "recipe",
    )
    name = document.get("recipe")
    if name not in _RECIPES:
        known = ", ".join(sorted(_RECIPES)) or "(none registered)"
        raise ValueError(f"unknown recipe '{name}'; known recipes: {known}")
    version = document.get("version", 1)
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version not in _SCHEMA_VERSIONS
    ):
        raise ValueError(f"unsupported recipe version {version!r} (expected 1)")
    _check_requires(document.get("requires"))
    recipe_dir = path.parent
    body = _RECIPES[name].resolve(document, recipe_dir)
    return {
        "recipe": name,
        "version": version,
        "requires": document.get("requires"),
        "recipe_dir": recipe_dir,
        "inputs": body["inputs"],
        "parameters": body["parameters"],
        "outputs": body["outputs"],
    }


class _RecipeType:
    """A registered recipe type: ``resolve`` validates + resolves its sections;
    execution is attached when the pipeline ships (PR 1b)."""

    def __init__(self, name, resolve):
        self.name = name
        self.resolve = resolve


def _resolve_exposure_tradeoff(document, recipe_dir):
    inputs = _resolve_inputs(document.get("inputs"), recipe_dir)
    parameters = _resolve_parameters(
        document.get("parameters"), tuple(inputs["exposure"])
    )
    outputs = _resolve_outputs(document.get("outputs"))
    return {"inputs": inputs, "parameters": parameters, "outputs": outputs}


#: The typed-recipe registry: a recipe type resolves its own sections, so a new
#: analysis type is added by registering it here, not by branching in validate().
_RECIPES = {
    "exposure_tradeoff": _RecipeType("exposure_tradeoff", _resolve_exposure_tradeoff)
}
