"""The streaming Parquet writer behind ``travel_cost_table(output=...)``
and the matrix computers' ``to_parquet`` classmethods.

One origin batch at a time flows through here: the single-file form
appends each batch as a row group through one ``ParquetWriter`` (staged
under a temporary name, renamed on completion), the directory form
publishes one shard per batch (write-temp + rename) beside an atomically
rewritten ``manifest.json``. Dictionary columns are checked against the
shared domain arrays on every batch — a deviating dictionary is a hard
error, never a silent re-encode.
"""

import dataclasses
import hashlib
import json
import os
import pathlib
import tempfile

import numpy as np

if False:  # noqa: SIM108 — typing-only import; pyarrow stays optional
    import pyarrow

#: Bumped whenever the hashed material or the result semantics change,
#: so fingerprints from other implementation versions can never match.
FINGERPRINT_VERSION = 1
MANIFEST_FORMAT = 1
MANIFEST_NAME = "manifest.json"
DEFAULT_BATCH_SIZE = 500


@dataclasses.dataclass(frozen=True)
class StreamingResult:
    """What a streaming run wrote — the return of every ``output=`` form.

    Attributes
    ----------
    path : pathlib.Path
        The file or directory written.
    mode : str
        ``"file"`` (one Parquet file of row groups) or ``"directory"``
        (one shard per batch plus ``manifest.json``).
    rows : int
        Total rows across all batches.
    batches : int
        Number of batches written.
    schema : pyarrow.Schema
        The written schema.
    manifest : dict or None
        The parsed manifest for the directory form, ``None`` for the
        single-file form.
    """

    path: pathlib.Path
    mode: str
    rows: int
    batches: int
    schema: "pyarrow.Schema"
    manifest: dict | None


def resolve_output(output):
    """The ``(mode, path)`` a streaming run writes to, by the suffix rule.

    A path whose final component ends in ``.parquet``
    (case-insensitive) is the single-file form, any other path the
    directory form. Existing targets are refused — a single file is
    never overwritten, and a directory must be empty.
    """
    path = pathlib.Path(output)
    if path.name.lower().endswith(".parquet"):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(
                f"output {path} already exists; the single-file form never "
                "overwrites — delete it or choose another path"
            )
        try:
            # An exclusive sidecar claim reserves the output while the
            # final name stays absent until completed publication — a
            # concurrent job racing the same output fails here.
            fd = os.open(claim_path(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise FileExistsError(
                f"output {path} is claimed by another running job "
                f"({claim_path(path)} exists); wait for it or remove a "
                "stale claim"
            ) from None
        os.close(fd)
        return "file", path
    if path.exists():
        if not path.is_dir():
            raise ValueError(
                f"output {path} exists and is not a directory; directory-"
                "form outputs (no .parquet suffix) need a directory"
            )
        if any(path.iterdir()):
            raise ValueError(
                f"output directory {path} is not empty; refusing to write "
                "beside existing contents"
            )
    else:
        path.mkdir(parents=True)
    return "directory", path


def network_digest(network):
    """A deterministic identity proxy for the routed network.

    Hashes the public stop and route tables and the trip count. This is
    an identity heuristic, not a content checksum: the artifact
    container's checksum is not exposed for in-memory builds, so two
    networks differing only in state outside these tables (timetables,
    transfers, street data) hash alike — the fingerprint still refuses
    the common mixups (different feeds, different extracts).
    """
    state = hashlib.sha256()
    for stop, latitude, longitude in network.stops:
        state.update(repr((stop, latitude, longitude)).encode())
    for route in network.routes:
        state.update(repr(route).encode())
    state.update(repr(network.trip_count).encode())
    core = getattr(network, "_core", None)
    for extra in ("_multimodal_checksum", "has_walking_hierarchy"):
        state.update(repr(getattr(core, extra, None)).encode())
    return state.hexdigest()


def fingerprint(operation, columns, digest, parameters, from_ids, to_ids, points):
    """The query fingerprint a manifest records.

    Hashes the producing operation and its output columns, the
    fingerprint-algorithm version, the network digest, the resolved
    parameter set, and the ordered, resolved routing inputs — ids *and*
    the coordinates actually routed (``points`` is ``None`` on stop
    queries, where the ids resolve through the network digest). The
    caller excludes ``resume`` and the output location from
    ``parameters``; everything result-affecting, including ``chunk`` and
    ``batch_size``, must be in it. The column names stand in for the
    Arrow schema pre-run; the serialized schema's digest lands in the
    manifest (``schema_digest``) once the first batch fixes it, and
    resume must match both.
    """
    material = {
        "version": FINGERPRINT_VERSION,
        "operation": operation,
        "columns": list(columns),
        "network": digest,
        "parameters": {key: _canonical(value) for key, value in parameters.items()},
        "from_ids": list(from_ids),
        "to_ids": list(to_ids),
        "points": _canonical(points),
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _canonical(value):
    """``value`` reduced to JSON-encodable material with stable float
    text; numpy arrays and scalars hash by their exact bytes."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return float.hex(value)
    if isinstance(value, (bytes, bytearray)):
        return hashlib.sha256(bytes(value)).hexdigest()
    if isinstance(value, np.ndarray):
        state = hashlib.sha256(np.ascontiguousarray(value).tobytes())
        state.update(str(value.dtype).encode())
        state.update(repr(value.shape).encode())
        return state.hexdigest()
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    raise TypeError(f"cannot fingerprint a {type(value).__name__}")


def write_manifest(directory, manifest, claim=False):
    """Atomically publish ``manifest.json``.

    With ``claim=True`` the manifest is created exclusively — a
    concurrent job that already claimed the directory makes this fail
    instead of both jobs interleaving shards. Updates go through a
    unique temporary file (no predictable name to plant a symlink at)
    and ``os.replace``.
    """
    encoded = json.dumps(manifest, indent=1, sort_keys=True)
    target = directory / MANIFEST_NAME
    fd, temporary = tempfile.mkstemp(dir=directory, suffix=".tmp")
    with os.fdopen(fd, "w") as stream:
        stream.write(encoded)
    if claim:
        try:
            # Linking the complete temporary in is both the exclusive
            # claim (fails if a concurrent job already published one)
            # and an atomic first publication — no truncated manifest
            # can ever exist under the final name.
            os.link(temporary, target)
        finally:
            os.unlink(temporary)
        return
    os.replace(temporary, target)


def write_stream(mode, path, batches, manifest_seed, dictionaries):
    """Drain ``batches`` into ``path`` and return the `StreamingResult`.

    ``batches`` yields ``(index, origin_start, origin_stop, table)``;
    the writer holds one table at a time. ``dictionaries`` maps
    dictionary column names to their shared domain arrays — every batch
    must carry exactly those domains.
    """
    import pyarrow.parquet

    rows = batches_written = 0
    schema = None
    if mode == "file":
        # The temporary is written through its mkstemp descriptor — the
        # pathname is never reopened, so a symlink swapped in under the
        # temporary name cannot redirect the write.
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".parquet.tmp")
        stream = os.fdopen(descriptor, "wb")
        writer = None
        try:
            try:
                for _index, _start, _stop, table in batches:
                    if schema is None:
                        schema = table.schema
                        writer = pyarrow.parquet.ParquetWriter(stream, schema)
                    _check_batch(table, schema, dictionaries)
                    # One routing batch is one row group — up to the
                    # Parquet format's 64 Mi rows-per-group cap, past
                    # which pyarrow splits (stated in the docs).
                    writer.write_table(table, row_group_size=max(table.num_rows, 1))
                    rows += table.num_rows
                    batches_written += 1
                    table = None
            finally:
                # Footer and stream failures reach the outer cleanup
                # too, chaining under any in-flight exception.
                try:
                    if writer is not None:
                        writer.close()
                finally:
                    stream.close()
            # A no-replace publication: linking the completed temporary
            # in fails if the final name appeared meanwhile — the
            # no-overwrite promise holds even against a race.
            os.link(temporary, path)
        except BaseException:
            # A failed run leaves nothing behind: the temporary and the
            # claim go, any file at the final name stays untouched.
            _cleanup(temporary, claim_path(path))
            raise
        _cleanup(temporary, claim_path(path))
        return StreamingResult(path, "file", rows, batches_written, schema, None)
    manifest = dict(manifest_seed)
    manifest["format"] = MANIFEST_FORMAT
    manifest["shards"] = []
    write_manifest(path, manifest, claim=True)
    for index, start, stop, table in batches:
        if schema is None:
            schema = table.schema
            manifest["schema_digest"] = hashlib.sha256(
                schema.serialize().to_pybytes()
            ).hexdigest()
        _check_batch(table, schema, dictionaries)
        shard = f"part-{index:05d}.parquet"
        descriptor, temporary = tempfile.mkstemp(dir=path, suffix=".tmp")
        with os.fdopen(descriptor, "wb") as stream:
            pyarrow.parquet.write_table(
                table, stream, row_group_size=max(table.num_rows, 1)
            )
        os.replace(temporary, path / shard)
        manifest["shards"].append(
            {
                "index": index,
                "file": shard,
                "origin_start": start,
                "origin_stop": stop,
                "rows": table.num_rows,
                "completed": True,
            }
        )
        write_manifest(path, manifest)
        rows += table.num_rows
        batches_written += 1
        table = None
    return StreamingResult(path, "directory", rows, batches_written, schema, manifest)


def claim_path(path):
    """The sidecar name that reserves a single-file output while the
    final name stays absent until completed publication."""
    return path.with_name(path.name + ".claim")


def _cleanup(*paths):
    for leftover in paths:
        try:
            os.unlink(leftover)
        except OSError:
            pass


def _check_batch(table, schema, dictionaries):
    """A deviating batch is a hard error, never a silent re-encode."""
    if table.schema != schema:
        raise ValueError(
            "a batch produced a different schema than the stream's first "
            "batch; the writer never casts"
        )
    for name, expected in dictionaries.items():
        column = table.column(name)
        if hasattr(column, "num_chunks"):
            if column.num_chunks != 1:
                raise ValueError(
                    f"a batch carried a {name} column of "
                    f"{column.num_chunks} chunks; the producer contract "
                    "is one chunk indexing the shared domain"
                )
            column = column.chunk(0)
        if _array_identity(column.dictionary) != _array_identity(expected):
            raise ValueError(
                f"a batch carried a deviating {name} dictionary domain; "
                "every batch must index the one shared domain array"
            )


def _array_identity(array):
    """The identity of an array — the same backing buffers *and* the
    same view of them (a slice shares buffers yet is a different
    domain), never mere value equality."""
    return (
        str(array.type),
        len(array),
        array.offset,
        [
            None if buffer is None else (buffer.address, buffer.size)
            for buffer in array.buffers()
        ],
    )
