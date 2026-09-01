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
#: 2: candidates/bucket join the transit parameters; the matrix
#: classmethods' operations and street/time parameter sets arrive.
#: 3: the network digest becomes the artifact content checksum.
#: 4: the slot list joins the parameters, so a scalar moment and a
#: one-slot list never share a fingerprint.
FINGERPRINT_VERSION = 4
MANIFEST_FORMAT = 2
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


def resolve_output(output, resume=False):
    """The ``(mode, path)`` a streaming run writes to, by the suffix rule.

    A path whose final component ends in ``.parquet``
    (case-insensitive) is the single-file form, any other path the
    directory form. Fresh targets are refused when occupied — a single
    file is never overwritten, and a directory must be empty. With
    ``resume=True`` (directory form only) the directory must exist and
    hold a run to continue; `prepare_resume` validates it.
    """
    path = pathlib.Path(output)
    if path.name.lower().endswith(".parquet"):
        if resume:
            raise ValueError(
                "resume=True applies to the directory form; a single "
                ".parquet file carries no manifest to continue from"
            )
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
    if resume:
        if not path.is_dir():
            raise ValueError(
                f"nothing to resume at {path}: the output directory of a "
                "partial run must exist"
            )
        return "directory", path
    if path.exists():
        if not path.is_dir():
            raise ValueError(
                f"output {path} exists and is not a directory; directory-"
                "form outputs (no .parquet suffix) need a directory"
            )
        if any(path.iterdir()):
            raise ValueError(
                f"output directory {path} is not empty; refusing to write "
                "beside existing contents (a matching partial run "
                "continues with resume=True)"
            )
    else:
        path.mkdir(parents=True)
    return "directory", path


def network_digest(network):
    """A deterministic identity for the routed network.

    The core exposes ``_artifact_checksum``: a network loaded from an
    artifact hashes the **file** (stable across processes — the
    identity that lets one HPC job resume another's run over a shared
    artifact), while an in-memory build or a mutated network digests
    the content it would save (strong, but process-local: resume such
    a run from the process that started it, or save and load the
    artifact). The structural summaries below remain as a fallback for
    cores without the getter.
    """
    state = hashlib.sha256()
    core = getattr(network, "_core", None)
    checksum = getattr(core, "_artifact_checksum", None)
    if checksum is not None:
        state.update(repr(("artifact", checksum())).encode())
        return state.hexdigest()
    if not hasattr(network, "stops"):
        # A standalone street network: counts, extent, and elevation
        # provenance stand in the same heuristic way.
        state.update(b"street")
        core = getattr(network, "_core", None)
        for name in ("vertex_count", "edge_count"):
            state.update(repr(getattr(network, name, None)).encode())
        state.update(repr(getattr(core, "_coordinate_bounds", None)).encode())
        state.update(repr(getattr(network, "elevation_metadata", None)).encode())
        return state.hexdigest()
    for stop, latitude, longitude in network.stops:
        state.update(repr((stop, latitude, longitude)).encode())
    for route in network.routes:
        state.update(repr(route).encode())
    state.update(repr(network.trip_count).encode())
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


def prepare_resume(path, fingerprint, size, count, slots=None):
    """The stored manifest and completed batch indices of the run to
    continue — validated against the resuming query, never trusted
    blindly.

    The manifest fingerprint must match the query's exactly (``resume``
    and the output location are outside it; everything result-affecting
    including ``chunk`` and ``batch_size`` is inside). The directory is
    claimed exclusively for the resume, stale temporaries are removed,
    and a part file without a completion marker — a run killed between
    the shard rename and its manifest update — is dropped so its batch
    recomputes.
    """
    target = path / MANIFEST_NAME
    # The exclusive claim comes before anything is read: a validation
    # snapshot taken outside the claim could otherwise be acted on
    # after another resume already finished, destroying its shards.
    claim_run(path)
    try:
        return _validate_resume(path, target, fingerprint, size, count, slots)
    except BaseException:
        _cleanup(path / "run.claim")
        raise


def _validate_resume(path, target, fingerprint, size, count, slots=None):
    if not target.exists():
        raise ValueError(
            f"nothing to resume at {path}: no manifest.json (only "
            "directory-form streaming runs can be resumed)"
        )
    try:
        manifest = json.loads(target.read_text())
    except ValueError:
        raise ValueError(
            f"the manifest at {target} is not valid JSON; the run cannot "
            "be resumed — remove the directory and rerun"
        ) from None
    if manifest.get("fingerprint_version") != FINGERPRINT_VERSION:
        raise ValueError(
            "the manifest was written by fingerprint version "
            f"{manifest.get('fingerprint_version')}, this cafein uses "
            f"{FINGERPRINT_VERSION}; rerun instead of resuming"
        )
    if manifest.get("fingerprint") != fingerprint:
        raise ValueError(
            f"the manifest at {target} records a different query "
            "fingerprint; resuming requires the exact original inputs "
            "and parameters (including chunk and batch_size)"
        )
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError(
            f"the manifest at {target} has format "
            f"{manifest.get('format')}, this cafein writes "
            f"{MANIFEST_FORMAT}; rerun instead of resuming"
        )
    if manifest.get("slots") != slots:
        raise ValueError(
            f"the manifest at {target} records slots that do not match "
            "the resuming query; the run cannot be resumed"
        )
    # Completed shard descriptors are validated, never trusted: the
    # canonical name pins each entry inside the directory, the origin
    # slice must match the current batch plan, and the published file
    # must exist with exactly the recorded rows.
    import pyarrow.parquet

    batches = max(1, -(-count // size))
    shards = [shard for shard in manifest["shards"] if shard["completed"]]
    seen = set()
    for shard in shards:
        index = shard["index"]
        if not isinstance(index, int) or not 0 <= index < batches or index in seen:
            raise ValueError(
                f"the manifest at {target} records shard index {index!r} "
                "outside the query's batch plan; the run cannot be resumed"
            )
        seen.add(index)
        expected_name = f"part-{index:05d}.parquet"
        if shard["file"] != expected_name:
            raise ValueError(
                f"the manifest at {target} names shard {index} "
                f"{shard['file']!r} instead of {expected_name!r}; the run "
                "cannot be resumed"
            )
        if shard["origin_start"] != index * size or shard["origin_stop"] != min(
            (index + 1) * size, count
        ):
            raise ValueError(
                f"the manifest at {target} records an origin slice for "
                f"shard {index} that does not match the batch plan; the "
                "run cannot be resumed"
            )
        part = path / expected_name
        if part.is_symlink() or not part.is_file():
            raise ValueError(
                f"completed shard {expected_name} is missing or not a "
                f"regular file in {path}; the run cannot be resumed"
            )
        published = pyarrow.parquet.ParquetFile(part)
        if published.metadata.num_rows != shard["rows"]:
            raise ValueError(
                f"completed shard {expected_name} holds different rows "
                "than the manifest records; the run cannot be resumed"
            )
        digest = hashlib.sha256(
            published.schema_arrow.serialize().to_pybytes()
        ).hexdigest()
        if digest != manifest.get("schema_digest"):
            raise ValueError(
                f"completed shard {expected_name} holds a different schema "
                "than the manifest records; the run cannot be resumed"
            )
        if _file_sha256(part) != shard.get("sha256"):
            raise ValueError(
                f"completed shard {expected_name} holds different content "
                "than the manifest records; the run cannot be resumed"
            )
    for stray in path.glob("*.tmp"):
        _cleanup(stray)
    manifest["shards"] = shards
    known = {shard["file"] for shard in shards}
    # Only a canonical name for an in-plan incomplete batch can be a
    # crash leftover; anything else in the directory is not this run's
    # to delete.
    expected = {f"part-{index:05d}.parquet" for index in range(batches)}
    for part in path.glob("part-*.parquet"):
        if part.name in known:
            continue
        if part.name in expected:
            _cleanup(part)
        else:
            raise ValueError(
                f"unexpected file {part.name} in the output directory; "
                "refusing to resume beside contents this run did not write"
            )
    return manifest, seen


def claim_run(path):
    """The one directory-wide exclusive claim every run — fresh or
    resumed — holds from before its first write until its last: two
    jobs can never interleave shards or manifests. A run killed while
    holding it leaves the claim behind; the error names the file so a
    stale claim can be removed after the death is confirmed."""
    try:
        fd = os.open(path / "run.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise FileExistsError(
            f"{path} is claimed by another running job (run.claim "
            "exists); wait for it, or remove the stale claim of a dead "
            "one"
        ) from None
    os.close(fd)


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


def write_stream(mode, path, batches, manifest_seed, dictionaries, manifest=None):
    """Drain ``batches`` into ``path`` and return the `StreamingResult`.

    ``batches`` yields ``(index, origin_start, origin_stop, table)``;
    the writer holds one table at a time. ``dictionaries`` maps
    dictionary column names to their shared domain arrays — every batch
    must carry exactly those domains. A ``manifest`` from
    `prepare_resume` continues that run (directory form): its completed
    shards stay untouched and only the remaining batches arrive here.
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
    if manifest is None:
        claim_run(path)
        try:
            manifest = dict(manifest_seed)
            manifest["format"] = MANIFEST_FORMAT
            manifest["shards"] = []
            write_manifest(path, manifest, claim=True)
        except BaseException:
            _cleanup(path / "run.claim")
            raise
    try:
        # The claim holds through the LAST write: schema recovery, the
        # final sort, and the closing manifest publication included.
        for index, start, stop, table in batches:
            if schema is None:
                schema = table.schema
                digest = hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()
                if manifest.get("schema_digest", digest) != digest:
                    raise ValueError(
                        "the resumed run's schema differs from the "
                        "manifest's schema digest; resuming requires the "
                        "exact original query"
                    )
                manifest["schema_digest"] = digest
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
                    # The published bytes, hashed so a resume can prove a
                    # completed shard is still this run's exact output.
                    "sha256": _file_sha256(path / shard),
                    "completed": True,
                }
            )
            write_manifest(path, manifest)
            table = None
        if schema is None and manifest["shards"]:
            # Every batch was already complete: the schema comes from the
            # published shards — and must still match the recorded digest.
            schema = pyarrow.parquet.read_schema(path / manifest["shards"][0]["file"])
            digest = hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()
            if manifest.get("schema_digest", digest) != digest:
                raise ValueError(
                    "the completed shards' schema differs from the manifest's "
                    "schema digest; the output cannot be trusted as this "
                    "query's"
                )
        manifest["shards"].sort(key=lambda shard: shard["index"])
        write_manifest(path, manifest)
    finally:
        _cleanup(path / "run.claim")
    rows = sum(shard["rows"] for shard in manifest["shards"])
    batches_written = len(manifest["shards"])
    return StreamingResult(path, "directory", rows, batches_written, schema, manifest)


def _file_sha256(path):
    """A published file's streamed content hash."""
    state = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            state.update(chunk)
    return state.hexdigest()


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
