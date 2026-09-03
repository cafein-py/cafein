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
import stat
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


#: A manifest larger than this is not one this module wrote: the writer
#: records one small entry per shard, so even a million one-row shards
#: stay far inside it; the bound only refuses an unbounded or absurd file.
MANIFEST_BYTES_LIMIT = 256 << 20
#: A generous bound on one shard's manifest entry (its index, file name,
#: row count, and digest), for the preflight that keeps a planned run's
#: manifest inside the limit.
MANIFEST_ENTRY_BYTES = 512


def _read_manifest(target):
    """The manifest at ``target`` parsed from one descriptor: opened
    without following a symlink, checked to be a regular file within
    ``MANIFEST_BYTES_LIMIT``, and read from that same descriptor."""
    # A link is refused before the open on every platform (O_NOFOLLOW
    # is not universal), and the opened descriptor must be the very
    # file lstat saw, so a swap between the two calls is refused too.
    linked = os.lstat(target)
    if stat.S_ISLNK(linked.st_mode):
        raise ValueError(f"{target} is a symbolic link")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_BINARY", 0)
    fd = os.open(target, flags)
    try:
        status = os.fstat(fd)
        if (status.st_dev, status.st_ino) != (linked.st_dev, linked.st_ino):
            raise ValueError(f"{target} changed while it was being opened")
        if not stat.S_ISREG(status.st_mode):
            raise ValueError(f"{target} is not a regular file")
        if status.st_size > MANIFEST_BYTES_LIMIT:
            raise ValueError(f"{target} is larger than a manifest can be")
        chunks, total = [], 0
        while total <= MANIFEST_BYTES_LIMIT:
            chunk = os.read(fd, MANIFEST_BYTES_LIMIT + 1 - total)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        raw = b"".join(chunks)
    finally:
        os.close(fd)
    if len(raw) > MANIFEST_BYTES_LIMIT:
        raise ValueError(f"{target} is larger than a manifest can be")
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except ValueError as error:
        raise ValueError(f"{target} is not valid JSON: {error}") from None
    if not isinstance(manifest, dict):
        raise ValueError(f"{target} is not valid JSON: not an object")
    return manifest


def stored_batch_size(path):
    """The batch size a directory run's manifest stores, for planning a
    resume before the claim: ``None`` without a manifest; a manifest
    whose ``batch_size`` is not a positive integer is refused. The
    resume's own validation rereads the file under the claim and
    refuses one that changed meanwhile, since the planned batch is part
    of the fingerprint."""
    target = path / MANIFEST_NAME
    try:
        manifest = _read_manifest(target)
    except FileNotFoundError:
        return None
    size = manifest.get("batch_size")
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError(f"{target}: the stored batch_size is not a positive integer")
    return size


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
        manifest = _read_manifest(target)
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
        _verify_shard_file(
            path, shard, expected_name, manifest.get("schema_digest"), "resumed"
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


def _verify_shard_file(path, shard, name, schema_digest, action, buffered=False):
    """One completed shard checked against its manifest record: a
    regular file with the recorded rows, schema, and content hash.

    With ``buffered`` the shard is read once and the returned table is
    parsed from the very bytes the content hash verified — a
    concurrent replacement can never slip between the check and the
    read."""
    import pyarrow
    import pyarrow.parquet

    part = path / name
    if part.is_symlink() or not part.is_file():
        raise ValueError(
            f"completed shard {name} is missing or not a "
            f"regular file in {path}; the run cannot be {action}"
        )
    if buffered:
        try:
            fd = os.open(
                part,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
            )
        except OSError:
            raise ValueError(
                f"completed shard {name} is missing or not a "
                f"regular file in {path}; the run cannot be {action}"
            ) from None
        with open(fd, "rb") as handle:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError(
                    f"completed shard {name} is missing or not a "
                    f"regular file in {path}; the run cannot be {action}"
                )
            data = handle.read()
        if hashlib.sha256(data).hexdigest() != shard.get("sha256"):
            raise ValueError(
                f"completed shard {name} holds different content "
                f"than the manifest records; the run cannot be {action}"
            )
        table = pyarrow.parquet.read_table(pyarrow.BufferReader(data))
        if table.num_rows != shard["rows"]:
            raise ValueError(
                f"completed shard {name} holds different rows "
                f"than the manifest records; the run cannot be {action}"
            )
        digest = hashlib.sha256(table.schema.serialize().to_pybytes()).hexdigest()
        if digest != schema_digest:
            raise ValueError(
                f"completed shard {name} holds a different schema "
                f"than the manifest records; the run cannot be {action}"
            )
        return table
    published = pyarrow.parquet.ParquetFile(part)
    if published.metadata.num_rows != shard["rows"]:
        raise ValueError(
            f"completed shard {name} holds different rows "
            f"than the manifest records; the run cannot be {action}"
        )
    digest = hashlib.sha256(published.schema_arrow.serialize().to_pybytes()).hexdigest()
    if digest != schema_digest:
        raise ValueError(
            f"completed shard {name} holds a different schema "
            f"than the manifest records; the run cannot be {action}"
        )
    if _file_sha256(part) != shard.get("sha256"):
        raise ValueError(
            f"completed shard {name} holds different content "
            f"than the manifest records; the run cannot be {action}"
        )
    return part


#: The streaming operations whose shards form a long matrix frame.
MATRIX_OPERATIONS = frozenset(
    {"TravelTimeMatrix.to_parquet", "TravelCostMatrix.to_parquet", "travel_cost_table"}
)


def read_shards(path):
    """A completed streamed matrix directory as one pandas frame.

    The manifest is verified — format, a matrix-producing operation,
    every shard completed, and each shard's rows, schema digest, and
    content hash — before the shards concatenate in index order. The
    whole matrix loads into memory."""
    import pathlib

    import pyarrow

    path = pathlib.Path(path)
    target = path / MANIFEST_NAME
    if not target.is_file():
        raise ValueError(
            f"no {MANIFEST_NAME} at {path}; only directory-form "
            "streaming runs can be read"
        )
    try:
        manifest = _read_manifest(target)
    except ValueError:
        raise ValueError(
            f"the manifest at {target} is not valid JSON; the run " "cannot be read"
        ) from None
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError(
            f"the manifest at {target} has format "
            f"{manifest.get('format')}, this cafein reads "
            f"{MANIFEST_FORMAT}; recompute the run"
        )
    operation = manifest.get("operation")
    if operation not in MATRIX_OPERATIONS:
        raise ValueError(
            f"the manifest at {target} records operation {operation!r}, "
            "not a matrix producer; only matrix shard directories "
            "aggregate"
        )
    slots = manifest.get("slots")
    if slots is not None and len(slots) > 1:
        raise ValueError(
            f"the run at {path} carries {len(slots)} slots; stream one "
            "moment, or select one slot before aggregating"
        )
    size = manifest.get("batch_size")
    count = manifest.get("origin_count")
    if (
        not isinstance(size, int)
        or not isinstance(count, int)
        or isinstance(size, bool)
        or isinstance(count, bool)
        or size < 1
        or count < 0
    ):
        raise ValueError(
            f"the manifest at {target} records no usable batch plan; "
            "the run cannot be read"
        )
    batches = max(1, -(-count // size))
    descriptors = list(manifest.get("shards", ()))
    if len(descriptors) != batches:
        raise ValueError(
            f"the manifest at {target} records {len(descriptors)} shard "
            f"descriptor(s) for a {batches}-batch plan; the run cannot "
            "be read"
        )
    shards = {}
    for shard in descriptors:
        index = shard.get("index")
        name = f"part-{index:05d}.parquet" if isinstance(index, int) else None
        if (
            not isinstance(index, int)
            or not 0 <= index < batches
            or index in shards
            or shard.get("file") != name
        ):
            raise ValueError(
                f"the manifest at {target} records shard descriptor "
                f"{shard.get('index')!r} outside the batch plan; the run "
                "cannot be read"
            )
        if not shard.get("completed"):
            raise ValueError(
                f"the run at {path} is incomplete: shard {index} of "
                f"{batches} never completed; resume it before aggregating"
            )
        shards[index] = shard
    tables = [
        _verify_shard_file(
            path,
            shards[index],
            f"part-{index:05d}.parquet",
            manifest.get("schema_digest"),
            "read",
            buffered=True,
        )
        for index in range(batches)
    ]
    return pyarrow.concat_tables(tables).to_pandas()


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
    if len(encoded.encode("utf-8")) > MANIFEST_BYTES_LIMIT:
        # The reader's bound, enforced where the file is made.
        raise ValueError(
            f"{target} would exceed {MANIFEST_BYTES_LIMIT} bytes; fewer slots or "
            "batches, or shorter slot labels"
        )
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
