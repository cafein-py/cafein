"""Declarative analysis recipes.

A *recipe* is a user-authored YAML document describing a whole analysis
pipeline; cafein runs it, so the file is the reproducible method. This module
is the loader, the typed-recipe registry, input resolution, eager validation
(:func:`validate`), and execution (:func:`run`), which writes each recipe's
outputs beside a run-provenance record of versions and input checksums.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import pathlib
import re

from cafein._validate import choice, non_negative_finite

#: Self-contained single-file formats per input role — so a content checksum
#: (at run time) covers the whole dataset; a shapefile's sidecars would not be.
_STREET_SUFFIXES = (".pbf",)
_RASTER_SUFFIXES = (".tif", ".tiff")
_VECTOR_SUFFIXES = (".gpkg", ".geojson", ".json")
_TABLE_SUFFIXES = (".csv", ".json", ".yaml", ".yml")

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
_STREET_KEYS = {
    "file": (("path",), (), _STREET_SUFFIXES),
    "sample": (("name",), (), _STREET_SUFFIXES),
}
_EXPOSURE_KEYS = {
    "raster": (("path", "value"), ("units",), _RASTER_SUFFIXES),
    "vector": (("path", "value"), ("units", "layer"), _VECTOR_SUFFIXES),
    "sample": (
        ("name", "value"),
        ("units", "layer"),
        _RASTER_SUFFIXES + _VECTOR_SUFFIXES,
    ),
}
_OD_KEYS = {
    "vector": (("path", "id_column"), ("layer",), _VECTOR_SUFFIXES),
    "sample": (("name", "id_column"), ("layer",), _VECTOR_SUFFIXES),
}
#: Parameter keywords with a spelling beyond plain YAML values. A data file is
#: written in the inputs' ``{kind: file|sample}`` form (never a bare path, so it
#: is snapshotted and checksummed): its formats, its extra keys, and how the
#: object receives it (a path, or a vector read into a GeoDataFrame — a
#: multi-layer GeoPackage needing ``layer:``). An object is a mapping of its
#: constructor's keywords; a mapping elsewhere is accepted only where the API
#: value is itself a mapping.
_FILE_KEYWORDS = {
    ("streets", "dem"): (_RASTER_SUFFIXES, (), "path"),
    ("streets", "urban_areas"): (_VECTOR_SUFFIXES, ("layer",), "vector"),
    ("matrix", "factors"): (_TABLE_SUFFIXES, (), "path"),
    ("matrix", "costs"): (_TABLE_SUFFIXES, (), "path"),
}
_OBJECT_KEYWORDS = {
    ("matrix", "traveler"): "TravelerProfile",
    ("matrix", "street_policy"): "StreetLegPolicy",
}
#: Object keywords whose values are themselves objects, keyed by name.
_NESTED_OBJECTS = {"StreetLegPolicy": {"vehicles": "VehiclePolicy"}}
_MAPPING_KEYWORDS = {
    ("exposure", "thresholds"),
    ("streets", "speed_limits"),
    ("matrix", "departure"),
    ("matrix", "arrival"),
}

#: ``Exposure(network, ..., **layers)`` keywords a layer may not be named after.
_RESERVED_LAYER_NAMES = frozenset(
    {"self", "network", "thresholds", "rasterize", "max_memory"}
)


def _sample_pin(name, where):
    """The pinned metadata of the ``cafein.sampledata`` asset ``<region>.<asset>``
    (a key of the region module's ``metadata`` table), resolved offline."""
    import importlib

    parts = name.split(".") if isinstance(name, str) else []
    if len(parts) != 2 or not all(part.isidentifier() for part in parts):
        raise ValueError(
            f"{where}: 'name' must be '<region>.<asset>', e.g. helsinki.osm_pbf"
        )
    region, asset = parts
    try:
        module = importlib.import_module(f"cafein.sampledata.{region}")
    except ImportError:
        raise ValueError(
            f"{where}: unknown sample region '{region}' (is cafein.sampledata "
            "installed?)"
        ) from None
    table = getattr(module, "metadata", None)
    if not isinstance(table, dict):
        raise ValueError(f"{where}: unknown sample region '{region}'")
    if asset not in table:
        raise ValueError(
            f"{where}: unknown sample asset '{name}' (one of "
            f"{', '.join(sorted(table))})"
        )
    return {"asset": name, "region": region, **table[asset]}


def _resolve_source(where, spec, recipe_dir, *, keys):
    """Validate + resolve one `kind:`-tagged input, or raise by name.

    Local paths resolve relative to the recipe's directory, so a recipe + its
    data move together; the file must exist and be a self-contained format. A
    ``sample`` names a pinned ``cafein.sampledata`` asset: its pin is resolved
    here without downloading, the file is fetched when the recipe runs.
    """
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
    for key in ("id_column", "units", "layer"):
        if key in spec and not isinstance(spec[key], str):
            raise ValueError(f"{where}: '{key}' must be a string")
    if "value" in spec and (
        isinstance(spec["value"], bool) or not isinstance(spec["value"], (str, int))
    ):
        raise ValueError(f"{where}: 'value' must be a column name or a band")
    if kind == "sample":
        sample = _sample_pin(spec["name"], where)
        path, filename = None, sample["name"]
    else:
        if not isinstance(spec["path"], str):
            raise ValueError(f"{where}: 'path' must be a string")
        path = (recipe_dir / spec["path"]).resolve()
        if not path.is_file():
            raise ValueError(f"{where}: file not found: {path}")
        filename = path.name
    if not _suffix_ok(pathlib.PurePath(filename), suffixes):
        raise ValueError(
            f"{where}: '{filename}' is not a self-contained {'/'.join(suffixes)} "
            "file (multi-file formats such as shapefiles are not supported)"
        )
    if path is not None and (
        path.suffix.lower() == ".gpkg" and path.with_name(path.name + "-wal").exists()
    ):
        raise ValueError(
            f"{where}: '{path.name}' has a live write-ahead log ({path.name}-wal); "
            "checkpoint the GeoPackage so the file holds all of its state"
        )
    resolved = {"kind": kind, "path": path}
    if kind == "sample":
        resolved["sample"] = sample
    for key in (*required, *optional):
        if key != "path" and key in spec:
            resolved[key] = spec[key]
    return resolved


def _resolve_inputs(inputs, recipe_dir, roles):
    """Resolve every declared input role, or raise by name: a role takes one
    ``kind:``-tagged source, or — declared ``("layers", kinds)`` — a non-empty
    mapping of named sources."""
    if not isinstance(inputs, dict):
        raise ValueError("inputs: must be a mapping")
    _reject_foreign(inputs, set(roles), "inputs")
    resolved = {}
    for role, keys in roles.items():
        spec = inputs.get(role)
        if spec is None:
            raise ValueError(f"inputs: missing '{role}'")
        if _is_layers(keys):
            if not isinstance(spec, dict) or not spec:
                raise ValueError(
                    f"inputs.{role}: must be a non-empty mapping of layers"
                )
            for name in spec:
                if not isinstance(name, str) or not name:
                    raise ValueError(
                        f"inputs.{role}: layer names must be non-empty strings"
                    )
            resolved[role] = {
                name: _resolve_source(
                    f"inputs.{role}.{name}", layer, recipe_dir, keys=keys[1]
                )
                for name, layer in spec.items()
            }
        else:
            resolved[role] = _resolve_source(
                f"inputs.{role}", spec, recipe_dir, keys=keys
            )
    return resolved


def _is_layers(keys):
    """A role declared as a mapping of named sources."""
    return isinstance(keys, tuple) and keys[0] == "layers"


def _input_sources(inputs, roles):
    """``(role, source)`` for every input, a mapping of named sources as
    ``role.name`` — by the roles' declaration, never by a value's shape."""
    for role, value in inputs.items():
        if _is_layers(roles[role]):
            for name, source in value.items():
                yield f"{role}.{name}", source
        else:
            yield role, value


def _check_layer_names(names):
    """Exposure layer names: not one of ``Exposure``'s own keywords, and its
    naming rule (lowercase identifiers outside the cost / travel_time column
    families), applied before any data is touched."""
    for name in names:
        if name in _RESERVED_LAYER_NAMES:
            raise ValueError(
                f"inputs.exposure: '{name}' is a reserved name, not a layer"
            )
    from cafein.exposure import _validate_names

    _validate_names(list(names), {}, ())


def _exposure_groups():
    """The exposure_tradeoff recipe's ``parameters:`` groups: each group's
    target callable, with the keywords the recipe fixes itself (its inputs,
    objective, and sweep) that a recipe may not set."""
    from cafein import Exposure, StreetNetwork, TravelCostMatrix

    return {
        "streets": (StreetNetwork.from_osm, {"osm_pbf", "modes"}),
        "exposure": (Exposure, {"network"}),
        "matrix": (
            TravelCostMatrix,
            {
                "network",
                "origins",
                "destinations",
                "transport_mode",
                "exposure",
                "candidates",
                "optimize",
                "output_time_units",
            },
        ),
    }


def _group_surface(target, fixed):
    """``{keyword: default}`` over the keywords a group may set: the target's
    named parameters minus the fixed ones."""
    import inspect

    surface = {}
    for name, param in inspect.signature(target).parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD) or name in fixed:
            continue
        surface[name] = None if param.default is param.empty else param.default
    return surface


def _build_object(name, mapping, where):
    """The object a mapping spells, built (and so validated) by its own
    constructor; a nested mapping of objects is built the same way."""
    import cafein

    kwargs = dict(mapping)
    for key, nested in _NESTED_OBJECTS.get(name, {}).items():
        if key in kwargs:
            items = kwargs[key]
            if not isinstance(items, dict) or not all(
                isinstance(v, dict) for v in items.values()
            ):
                raise ValueError(
                    f"{where}.{key}: a mapping of {nested} keyword mappings"
                )
            kwargs[key] = {
                k: _build_object(nested, v, f"{where}.{key}.{k}")
                for k, v in items.items()
            }
    try:
        return getattr(cafein, name)(**kwargs)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{where}: {error}") from None


def _effective_object(name, mapping):
    """An object spelling with its constructor's defaults filled in, nested
    objects too."""
    import cafein

    values = {**_group_surface(getattr(cafein, name), set()), **mapping}
    for key, nested in _NESTED_OBJECTS.get(name, {}).items():
        if isinstance(values.get(key), dict):
            values[key] = {
                k: _effective_object(nested, v) for k, v in values[key].items()
            }
    return values


def _resolve_value(group, key, value, recipe_dir):
    """One group keyword's value in its recipe spelling, or raise by name."""
    where = f"parameters.{group}.{key}"
    if (group, key) in _FILE_KEYWORDS:
        if not (isinstance(value, dict) and "kind" in value):
            raise ValueError(
                f"{where}: a data file, written as {{kind: file, path: ...}} or "
                "{kind: sample, name: ...}"
            )
        suffixes, extra, _ = _FILE_KEYWORDS[group, key]
        keys = {
            "file": (("path",), extra, suffixes),
            "sample": (("name",), extra, suffixes),
        }
        return _resolve_source(where, value, recipe_dir, keys=keys)
    if (group, key) in _OBJECT_KEYWORDS:
        if not isinstance(value, dict):
            raise ValueError(
                f"{where}: a mapping of {_OBJECT_KEYWORDS[group, key]} keywords"
            )
        _build_object(_OBJECT_KEYWORDS[group, key], value, where)
        return value
    if isinstance(value, dict) and (group, key) not in _MAPPING_KEYWORDS:
        raise ValueError(f"{where}: does not take a mapping")
    return value


def _parameter_sources(parameters):
    """``(group, key, source)`` for every file-valued parameter."""
    for group, key in _FILE_KEYWORDS:
        values = parameters.get(group)
        if isinstance(values, dict) and key in values:
            yield group, key, values[key]


def _resolve_groups(parameters, groups, recipe_dir):
    """Each group's keywords for the object it configures: named keywords only,
    the ones the recipe fixes refused, a data file in the inputs' ``kind:``
    spelling, an object as a mapping of its keywords (built here, so its own
    validation runs eagerly). Other values are checked by the objects when the
    recipe runs."""
    resolved = {}
    for group, (target, fixed) in groups.items():
        given = parameters.get(group)
        if given is None:
            given = {}
        if not isinstance(given, dict):
            raise ValueError(f"parameters.{group}: must be a mapping of keywords")
        surface = _group_surface(target, fixed)
        values = {}
        for key, value in given.items():
            if key in fixed:
                raise ValueError(
                    f"parameters.{group}.{key}: fixed by the recipe, not a parameter"
                )
            if key not in surface:
                raise ValueError(
                    f"parameters.{group}: unknown keyword '{key}' (one of "
                    f"{', '.join(sorted(surface))})"
                )
            values[key] = _resolve_value(group, key, value, recipe_dir)
        resolved[group] = values
    return resolved


def _resolve_exposure_parameters(parameters, exposure_layers, recipe_dir):
    """The exposure_tradeoff recipe's own parameters, then its groups."""
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, dict):
        raise ValueError("parameters: must be a mapping")
    groups = _exposure_groups()
    _reject_foreign(
        parameters, {"mode", "objective_layer", "weights", *groups}, "parameters"
    )
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
    if any(later <= earlier for earlier, later in zip(weights, weights[1:])):
        raise ValueError("parameters.weights: must be strictly increasing")
    resolved = {"mode": mode, "objective_layer": objective, "weights": weights}
    resolved.update(_resolve_groups(parameters, groups, recipe_dir))
    _canonical(resolved, "parameters")  # every value must be recordable
    return resolved


#: Names that fail on Windows even though POSIX accepts them.
_UNPORTABLE_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')
_WINDOWS_RESERVED = re.compile(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])(\..*)?$", re.I)


def _resolve_outputs(outputs):
    if not isinstance(outputs, dict) or not outputs.get("table"):
        raise ValueError("outputs: must declare a 'table' path")
    _reject_foreign(outputs, {"table"}, "outputs")
    if not isinstance(outputs["table"], str):
        raise ValueError("outputs.table: must be a string path")
    table = outputs["table"]
    posix = pathlib.PurePosixPath(table)
    windows = pathlib.PureWindowsPath(table)
    if posix.name in ("", ".", "..") or table.endswith(("/", "\\")):
        raise ValueError(
            f"outputs.table '{table}' must name a file inside the output root"
        )
    if not posix.name.endswith(".parquet"):
        raise ValueError(
            f"outputs.table '{table}' must end with .parquet, lowercase (the table "
            "is written as Parquet; its provenance and lock derive from the name)"
        )
    if len(posix.name.encode("utf-8")) > 247:
        raise ValueError(
            f"outputs.table name '{posix.name}' must stay within 247 bytes so its "
            "provenance record and lock fit a 255-byte filename limit"
        )
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
    for part in posix.parts:
        if (
            _UNPORTABLE_CHARS.search(part)
            or part.endswith((" ", "."))
            or _WINDOWS_RESERVED.match(part)
            or len(part.encode("utf-8")) > 255
        ):
            raise ValueError(
                f"outputs.table component '{part}' is not a portable file name "
                "(reserved on Windows, a forbidden character, a trailing dot/space, "
                "or over 255 bytes)"
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
    """A registered recipe type: its input ``roles`` (role → the kinds it
    accepts, or ``("layers", kinds)`` for a mapping of named sources), its
    parameter ``groups`` (a callable giving ``{group: (target, fixed)}``),
    ``resolve`` (validates + resolves its sections), and ``run`` (executes the
    resolved recipe, returning ``(frame, checksums)``)."""

    def __init__(self, name, roles, groups, resolve, run):
        self.name = name
        self.roles = roles
        self.groups = groups
        self.resolve = resolve
        self.run = run


_EXPOSURE_ROLES = {
    "streets": _STREET_KEYS,
    "exposure": ("layers", _EXPOSURE_KEYS),
    "origins": _OD_KEYS,
    "destinations": _OD_KEYS,
}


def _resolve_exposure_tradeoff(document, recipe_dir):
    inputs = _resolve_inputs(document.get("inputs"), recipe_dir, _EXPOSURE_ROLES)
    _check_layer_names(inputs["exposure"])
    parameters = _resolve_exposure_parameters(
        document.get("parameters"), tuple(inputs["exposure"]), recipe_dir
    )
    outputs = _resolve_outputs(document.get("outputs"))
    return {"inputs": inputs, "parameters": parameters, "outputs": outputs}


def _file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _layer_names(path, where):
    """The layers of a multi-layer file, or a by-name refusal when they cannot
    be listed — never a silent default layer."""
    import geopandas

    try:
        return geopandas.list_layers(path)["name"].tolist()
    except Exception:
        try:
            import fiona

            return list(fiona.listlayers(path))
        except Exception as error:
            raise ValueError(
                f"{where}: cannot list the layers of '{path.name}'; give 'layer:'"
            ) from error


def _read_vector(path, layer, where):
    """A vector file as a GeoDataFrame; a multi-layer GeoPackage needs `layer:`
    (GeoJSON holds one layer by construction)."""
    import geopandas

    if layer is None and path.suffix.lower() == ".gpkg":
        layers = _layer_names(path, where)
        if len(layers) > 1:
            raise ValueError(
                f"{where}: '{path.name}' holds several layers "
                f"({', '.join(layers)}); pick one with 'layer:'"
            )
    return geopandas.read_file(path, layer=layer)


def _load_points(source, where):
    """Origins/destinations with the declared id column exposed as ``id``,
    keeping its dtype (the matrix preserves input id types)."""
    frame = _read_vector(source["path"], source.get("layer"), where)
    column = source["id_column"]
    if column not in frame.columns:
        raise ValueError(f"{where}: no column '{column}' in {source['path'].name}")
    ids = frame[column]
    if ids.isna().any() or not ids.is_unique:
        raise ValueError(f"{where}: '{column}' values must be non-null and unique")
    return frame.assign(id=ids)


def _load_exposure_sources(exposure):
    """Each layer's ``(source, value)`` for ``Exposure``: a raster path, or a
    vector file read into a GeoDataFrame."""
    layers = {}
    for name, spec in exposure.items():
        if _suffix_ok(spec["path"], _RASTER_SUFFIXES):
            layers[name] = (str(spec["path"]), spec["value"])
        else:
            frame = _read_vector(
                spec["path"], spec.get("layer"), f"inputs.exposure.{name}"
            )
            if spec["value"] not in frame.columns:
                raise ValueError(
                    f"inputs.exposure.{name}: no column '{spec['value']}' in "
                    f"{spec['path'].name}"
                )
            layers[name] = (frame, spec["value"])
    return layers


def _snapshot(index, role, source, run_dir, checksums):
    """Copy one input into the run's private directory under a collision-free
    name, hashing the very bytes copied, so the provenance describes exactly
    what the pipeline reads."""
    digest = hashlib.sha256()
    copy = run_dir / f"input-{index:02d}{source['path'].suffix.lower()}"
    with open(source["path"], "rb") as origin, open(copy, "wb") as target:
        for block in iter(lambda: origin.read(1 << 20), b""):
            digest.update(block)
            target.write(block)
    checksums[role] = {"path": str(source["path"]), "sha256": digest.hexdigest()}
    if "sample" in source:
        if digest.hexdigest() != source["sample"]["sha256"]:
            raise ValueError(
                f"{role}: sample asset {source['sample']['asset']} read as "
                f"{digest.hexdigest()}, not its pin {source['sample']['sha256']}"
            )
        checksums[role]["sample"] = source["sample"]
    return {**source, "path": copy}


def _materialise(source):
    """A source with a local path: a file input as declared, a sample asset
    fetched and verified by ``cafein.sampledata`` (cached after first use)."""
    if source["path"] is not None:
        return source
    from cafein.sampledata import Asset, fetch

    sample = source["sample"]
    pinned = {field: sample[field] for field in Asset.__dataclass_fields__}
    return {**source, "path": fetch(Asset(**pinned), sample["region"])}


def _materialise_inputs(inputs, roles):
    """Every input with a local path (sample assets fetched)."""
    return {
        role: (
            {name: _materialise(source) for name, source in value.items()}
            if _is_layers(roles[role])
            else _materialise(value)
        )
        for role, value in inputs.items()
    }


def _run_exposure_tradeoff(resolved):
    """The fastest-vs-lower-exposure pipeline at matrix scale. Returns the
    trade-off frame and each input's checksum, taken from the snapshot the
    pipeline actually read."""
    import tempfile

    import pandas as pd

    from cafein import Exposure, StreetNetwork, TravelCostMatrix, compare_to_fastest

    inputs, parameters = resolved["inputs"], resolved["parameters"]
    mode, objective = parameters["mode"], parameters["objective_layer"]
    checksums = {}
    with tempfile.TemporaryDirectory(prefix="cafein-recipe-") as run_dir:
        run_dir = pathlib.Path(run_dir)
        roles = list(_input_sources(inputs, _EXPOSURE_ROLES))
        roles += [
            (f"parameters.{g}.{k}", src) for g, k, src in _parameter_sources(parameters)
        ]
        copies = {
            role: _snapshot(index, role, _materialise(source), run_dir, checksums)
            for index, (role, source) in enumerate(roles)
        }
        streets_source = copies["streets"]
        exposure_sources = {n: copies[f"exposure.{n}"] for n in inputs["exposure"]}
        origins_source, destinations_source = copies["origins"], copies["destinations"]

        def group(name):
            """The group's keywords, a file-valued one as its snapshot's path."""
            values = {}
            for key, value in parameters[name].items():
                if (name, key) in _FILE_KEYWORDS:
                    copy = copies[f"parameters.{name}.{key}"]
                    if _FILE_KEYWORDS[name, key][2] == "vector":
                        where = f"parameters.{name}.{key}"
                        value = _read_vector(copy["path"], copy.get("layer"), where)
                    else:
                        value = str(copy["path"])
                elif (name, key) in _OBJECT_KEYWORDS:
                    where = f"parameters.{name}.{key}"
                    value = _build_object(_OBJECT_KEYWORDS[name, key], value, where)
                values[key] = value
            return values

        streets = StreetNetwork.from_osm(
            streets_source["path"], modes=[mode], **group("streets")
        )
        exposure = Exposure(
            streets, **group("exposure"), **_load_exposure_sources(exposure_sources)
        )
        origins = _load_points(origins_source, "inputs.origins")
        destinations = _load_points(destinations_source, "inputs.destinations")
        # Exact seconds from the matrix; the table reports float minutes so the
        # exposure integral (concentration × minutes) is not built on rounded time.
        frame = TravelCostMatrix(
            streets,
            origins,
            destinations,
            transport_mode=mode,
            output_time_units="seconds",
            exposure=exposure,
            candidates="sweep",
            optimize={objective: parameters["weights"]},
            **group("matrix"),
        )
    frame = pd.DataFrame(frame)
    frame["travel_time"] = frame["travel_time"] / 60.0
    # The layer mean is weighted over traversed street edges only, so the
    # integral takes the on-street share of the trip: the snap connectors at
    # each end carry no sampled concentration. One mode, one speed — time
    # splits as distance does.
    on_network = frame["network_distance_m"]
    total = on_network + frame["connector_distance_m"]
    street_share = (on_network / total).where(total > 0, 0.0)
    frame["on_street_time"] = frame["travel_time"] * street_share
    for name in inputs["exposure"]:
        frame[f"{name}_exposure"] = frame[f"{name}_mean"] * frame["on_street_time"]
    # compare_to_fastest keeps the key and numeric columns in row order; the
    # ridden geometry (``matrix.geometries``) rejoins them as GeoParquet.
    geometry = frame.pop("geometry") if "geometry" in frame.columns else None
    frame = compare_to_fastest(frame)
    if geometry is not None:
        import geopandas

        lines = geopandas.GeoSeries(
            geometry.to_numpy(), index=frame.index, crs="EPSG:4326"
        )
        frame = geopandas.GeoDataFrame(frame, geometry=lines)
    return frame, checksums


def _dependency_versions():
    """The runtime versions that can change a result: the installed cafein
    distribution and, separately, the compiled core (a stale build shows as a
    mismatch), plus the geospatial stack. A missing entry is recorded as
    "unavailable" rather than dropped, so absence itself is visible."""
    import importlib
    from importlib import metadata

    import cafein
    from cafein import _cafein

    try:
        distribution = metadata.version("cafein")
    except metadata.PackageNotFoundError:
        distribution = cafein.__version__
    versions = {
        "cafein": distribution,
        "cafein_core": getattr(_cafein, "__version__", "unavailable"),
    }
    for package in (
        "geopandas",
        "pyrosm",
        "numpy",
        "shapely",
        "pandas",
        "pyarrow",
        "cafein.sampledata",
    ):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            try:
                versions[package] = importlib.import_module(package).__version__
            except Exception:
                versions[package] = "unavailable"
    return versions


def _canonical(value, where):
    """``value`` as JSON data, deterministically: paths and moments as strings,
    tuples as lists, sets ordered; anything else is refused by name."""
    if isinstance(value, dict):
        return {str(k): _canonical(v, f"{where}.{k}") for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(v, where) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_canonical(v, where) for v in value)
    if isinstance(value, (pathlib.PurePath, datetime.date, datetime.timedelta)):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ValueError(f"{where}: {value!r} cannot be recorded in the provenance")


def _effective_parameters(parameters, groups):
    """Every group with its defaults filled in — the complete method — an
    object spelling with its constructor's defaults too."""
    effective = {k: v for k, v in parameters.items() if k not in groups}
    for group, (target, fixed) in groups.items():
        values = _group_surface(target, fixed)
        for key, value in parameters[group].items():
            if (group, key) in _OBJECT_KEYWORDS:
                value = _effective_object(_OBJECT_KEYWORDS[group, key], value)
            values[key] = value
        effective[group] = values
    return effective


def _serialisable(resolved):
    """The resolved recipe with every parameter group's defaults filled in, for
    the provenance record."""
    groups = _RECIPES[resolved["recipe"]].groups()
    return {
        "recipe": resolved["recipe"],
        "version": resolved["version"],
        "requires": resolved["requires"],
        "inputs": resolved["inputs"],
        "parameters": _effective_parameters(resolved["parameters"], groups),
        "outputs": resolved["outputs"],
    }


def _reject_symlinks(out_root, *targets):
    """Refuse a symlinked component below the output root on the table's or
    the record's path: a link swapped in under the run could redirect where
    the files land, or what a rollback reads."""
    for target in targets:
        for component in (*reversed(target.parents), target):
            if component == out_root or out_root not in component.parents:
                continue
            if component.is_symlink():
                raise ValueError(f"output path component '{component}' is a symlink")


def _output_path(out_root, table, protected):
    """The output's absolute path, refused if it escapes ``out_root`` or would
    overwrite the recipe or one of its inputs."""
    unresolved = out_root / table
    record = unresolved.with_name(f"{unresolved.stem}.provenance.json")
    _reject_symlinks(out_root, unresolved, record)
    target = unresolved.resolve()
    if out_root not in target.parents:
        raise ValueError(
            f"outputs.table '{table}' must be a file inside the output root {out_root}"
        )
    provenance = target.with_name(record.name)
    for candidate in (target, provenance):
        for path in protected:
            same = candidate == path or (
                candidate.exists() and path.exists() and candidate.samefile(path)
            )
            if same:
                raise ValueError(
                    f"outputs.table '{table}' would overwrite an input or the "
                    f"recipe itself ({path})"
                )
    return target, provenance


def _identity(path):
    """(device, inode) of the file or directory at ``path``."""
    import os

    info = os.stat(path)
    return info.st_dev, info.st_ino


def run(path, out_dir=None, *, _entry_point="python", _invocation=None):
    """Validate a recipe, run its pipeline, and write its outputs.

    Output paths resolve relative to ``out_dir`` (the current working directory
    by default; resolved once and taken as given), must stay within it, may not
    overwrite the recipe or its inputs, and, below that root, neither the table
    nor its record may pass through a symlink. The target directory is created
    and pinned before the analysis and re-verified after it. Publication is
    serialized per target with a lock file and staged beside the target: the
    ``<stem>.provenance.json`` is moved in first, then the table, each move
    atomic on its own; a failed or interrupted run recovers the pair from what
    is on disk (the previous record returns unless the new table landed), and
    the record's table SHA-256 is the pairing check should a hard crash land
    between the two moves. The record carries the cafein and dependency versions, the
    resolved recipe, every input's SHA-256 (from the bytes the pipeline read),
    the table's own SHA-256, the invocation, and a UTC timestamp. These guards
    address accidents, stale state, and clashing runs in a researcher's own
    output directory; they are not a defence against an adversary racing that
    directory or rewriting the inputs mid-run. Returns the trade-off frame.
    """
    import os
    import shutil
    import tempfile

    recipe_path = pathlib.Path(path).resolve()
    resolved = validate(recipe_path)
    # Sample assets are fetched first, so the output guards see their paths.
    recipe = _RECIPES[resolved["recipe"]]
    resolved["inputs"] = _materialise_inputs(resolved["inputs"], recipe.roles)
    for group, key, source in list(_parameter_sources(resolved["parameters"])):
        resolved["parameters"][group][key] = _materialise(source)
    out_root = pathlib.Path(out_dir or pathlib.Path.cwd()).resolve()
    protected = [recipe_path]
    protected += [
        source["path"] for _, source in _input_sources(resolved["inputs"], recipe.roles)
    ]
    protected += [s["path"] for _, _, s in _parameter_sources(resolved["parameters"])]
    unresolved = out_root / resolved["outputs"]["table"]
    target, provenance = _output_path(out_root, resolved["outputs"]["table"], protected)
    out_root.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    parent_identity = _identity(target.parent)

    frame, checksums = recipe.run(resolved)

    _reject_symlinks(out_root, unresolved, unresolved.with_name(provenance.name))
    if _identity(target.parent) != parent_identity:
        raise ValueError(
            f"the output directory {target.parent} changed while the recipe ran; "
            "refusing to publish"
        )
    lock = target.with_name(f".{target.name}.lock")
    try:
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        raise ValueError(
            f"another run is publishing {target.name} ({lock.name} exists); "
            "wait for it or remove a stale lock"
        ) from None
    try:
        staging = pathlib.Path(
            tempfile.mkdtemp(dir=target.parent, prefix=".cafein-recipe-")
        )
        try:
            staged_table = staging / target.name
            frame.to_parquet(staged_table, index=False)
            record = {
                "cafein_recipe_provenance": 1,
                "recipe": resolved["recipe"],
                "versions": _dependency_versions(),
                "resolved": _serialisable(resolved),
                "inputs": checksums,
                "outputs": {"table": str(target), "sha256": _file_digest(staged_table)},
                "invocation": {
                    **(_invocation or {}),
                    "entry_point": _entry_point,
                    "recipe": str(recipe_path),
                    "out_dir": str(out_root),
                },
                "written_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            staged_record = staging / provenance.name
            staged_record.write_text(
                json.dumps(_canonical(record, "record"), indent=2, sort_keys=True)
            )
            # Record first, table last, recovered from disk state on failure:
            # the new record stays only if the new table landed. Between the
            # two moves a reader verifies pairing by the record's table SHA.
            previous = None
            if provenance.exists():
                previous = staging / "previous.json"
                shutil.copy2(provenance, previous)
            staged_identity = _identity(staged_table)
            try:
                os.replace(staged_record, provenance)
                os.replace(staged_table, target)
            except BaseException:
                try:
                    landed = _identity(target) == staged_identity
                except FileNotFoundError:
                    landed = False
                if not landed:
                    if previous is not None:
                        os.replace(previous, provenance)
                    else:
                        provenance.unlink(missing_ok=True)
                raise
        finally:
            for leftover in staging.iterdir():
                leftover.unlink()
            staging.rmdir()
    finally:
        lock.unlink(missing_ok=True)
    return frame


#: The typed-recipe registry: a recipe type resolves its own sections, so a new
#: analysis type is added by registering it here, not by branching in validate().
_RECIPES = {
    "exposure_tradeoff": _RecipeType(
        "exposure_tradeoff",
        _EXPOSURE_ROLES,
        _exposure_groups,
        _resolve_exposure_tradeoff,
        _run_exposure_tradeoff,
    )
}
