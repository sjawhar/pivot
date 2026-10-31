from __future__ import annotations

import datetime
import hashlib
import json
import uuid
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict, cast

from pivot.storage import cache
from pivot.types import (
    ArtifactTag,
    DepEntry,
    DirHash,
    FileHash,
    HashInfo,
    StageStatus,
    ensure_completion_type,
    identity_key,
    is_dir_hash,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from pivot.types import ArtifactIdentity, ArtifactRef


class StageRunRecord(TypedDict):
    """Record of a stage execution within a run."""

    input_hash: str | None
    status: Literal[
        StageStatus.RAN,
        StageStatus.CACHED,
        StageStatus.BLOCKED,
        StageStatus.CANCELLED,
        StageStatus.FAILED,
    ]
    reason: str
    duration_ms: int


class RunManifest(TypedDict):
    """Record of a complete pipeline run."""

    run_id: str
    started_at: str  # ISO 8601
    ended_at: str
    targeted_stages: list[str]
    execution_order: list[str]
    stages: dict[str, StageRunRecord]


class RunCacheEntry(TypedDict):
    """Run cache entry for skip detection. ``reads`` lists the objects the run read by hash
    (``pivot.read_by_hash``), sorted, each once, when it read any: a restore writes them into
    the lock it writes, and an entry whose objects are gone restores nothing."""

    run_id: str
    output_hashes: list[OutputHashEntry]
    reads: NotRequired[list[str]]


class OutputHashEntry(TypedDict):
    """Single output hash entry in run cache. A directory carries ``size`` and ``nfiles``; its
    listing is the cache object at its hash."""

    path: str
    hash: str
    size: NotRequired[int]
    nfiles: NotRequired[int]


# Part of every run-cache key this pivot files under (compute_input_hash). An older pivot reads a
# run-cache output that lists no files inline as one file, so it would restore a directory's
# listing in the directory's place; keyed apart, it never finds an entry this pivot wrote. This
# pivot still looks an entry up under the unmarked key an older pivot filed it under, with each
# directory dependency hashed as that pivot hashes it (worker.older_pivot_input_hash), and reads
# it in memory (worker._try_skip_via_run_cache_with_store), so an upgrade reruns nothing the run
# cache restored before it.
_DIRECTORY_RECORD = "listing"


def output_hash_to_entry(path: str, oh: HashInfo) -> OutputHashEntry:
    """Convert internal HashInfo to serializable OutputHashEntry."""
    entry = OutputHashEntry(path=path, hash=oh["hash"])
    if is_dir_hash(oh):
        entry["size"] = oh["size"]
        entry["nfiles"] = oh["nfiles"]
    return entry


def entry_to_output_hash(entry: OutputHashEntry) -> FileHash | DirHash | None:
    """Convert serialized OutputHashEntry back to internal HashInfo; None for an entry an older
    pivot filed for a directory, which lists its files inline (the caller converts it), and for
    a directory entry without its size, which no pivot writes."""
    if "manifest" in cast("Mapping[str, object]", entry):
        return None
    if "nfiles" in entry:
        if "size" not in entry:
            return None
        return DirHash(hash=entry["hash"], size=entry["size"], nfiles=entry["nfiles"])
    return FileHash(hash=entry["hash"])


# The keys of an output entry an older pivot filed for a directory, listing its files inline
# (``entry_to_output_hash`` returns None for one; ``listing.from_inline`` reads it).
OLDER_DIRECTORY_KEYS = frozenset({"manifest", "empty_dirs"})
# Every key `pivot gc` accepts in a run-cache row and in one of its output entries, written out
# rather than read from ``RunCacheEntry`` and ``OutputHashEntry``: gc keeps the object at an
# output's ``hash``, every file of a directory's listing, and every object the row's ``reads``
# lists, and nothing else. A field added to either TypedDict that can name a cache object must
# be added here and rooted in ``storage/gc.py`` in the same change; until then gc refuses a row
# that holds it. ``tests/test_run_history.py`` pins these to the TypedDicts.
RUN_CACHE_ROW_KEYS = frozenset({"run_id", "output_hashes"})
RUN_CACHE_ROW_OPTIONAL_KEYS = frozenset({"reads"})
OUTPUT_ENTRY_REQUIRED_KEYS = frozenset({"path", "hash"})
OUTPUT_ENTRY_OPTIONAL_KEYS = frozenset({"size", "nfiles"})


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def run_cache_entry_error(row: object) -> str | None:
    """Why a parsed run-cache row does not hold ``RUN_CACHE_ROW_KEYS`` with any of
    ``RUN_CACHE_ROW_OPTIONAL_KEYS`` (``reads``, a list of content hashes), each output exactly
    ``OUTPUT_ENTRY_REQUIRED_KEYS`` with any of ``OUTPUT_ENTRY_OPTIONAL_KEYS`` (or an older pivot's
    inline directory, ``OLDER_DIRECTORY_KEYS``), or None.

    For a reader that must not skip what it cannot read (`pivot gc`): a key outside the schema
    may be a newer pivot's record of objects. ``deserialize_run_cache_entry`` checks only what a
    lookup needs.
    """
    if not isinstance(row, dict):
        return f"its JSON is {type(row).__name__}, not an object"
    fields = cast("dict[object, object]", row)
    held = {str(key) for key in fields}
    unknown = sorted(held - RUN_CACHE_ROW_KEYS - RUN_CACHE_ROW_OPTIONAL_KEYS)
    missing = sorted(RUN_CACHE_ROW_KEYS - held)
    if unknown or missing:
        expected = " and ".join(sorted(RUN_CACHE_ROW_KEYS))
        return (
            f"its keys are {', '.join(sorted(held)) or 'none'}, not {expected} (and optionally "
            + f"{', '.join(sorted(RUN_CACHE_ROW_OPTIONAL_KEYS))})"
        )
    if not isinstance(fields["run_id"], str):
        return "its run_id is not a string"
    if "reads" in fields:
        read = fields["reads"]
        if not isinstance(read, list) or not all(
            cache.is_valid_hash(item) for item in cast("list[object]", read)
        ):
            return "its reads is not a list of content hashes"
    outputs = fields["output_hashes"]
    if not isinstance(outputs, list):
        return "its output_hashes is not a list"
    allowed = OUTPUT_ENTRY_REQUIRED_KEYS | OUTPUT_ENTRY_OPTIONAL_KEYS | OLDER_DIRECTORY_KEYS
    for index, raw in enumerate(cast("list[object]", outputs)):
        if not isinstance(raw, dict):
            return f"output_hashes[{index}] is not an object"
        output = cast("dict[object, object]", raw)
        keys = {str(key) for key in output}
        extra = sorted(keys - allowed)
        if extra:
            return (
                f"output_hashes[{index}] holds the key {', '.join(extra)}, which this pivot does "
                + "not know"
            )
        missing = sorted(OUTPUT_ENTRY_REQUIRED_KEYS - keys)
        if missing:
            return f"output_hashes[{index}] has no {', '.join(missing)}"
        if not isinstance(output["path"], str) or not isinstance(output["hash"], str):
            return f"output_hashes[{index}] has a path or hash that is not a string"
        for count in ("size", "nfiles"):
            if count in output and not _is_count(output[count]):
                return f"output_hashes[{index}].{count} is not a count"
    return None


def generate_run_id() -> str:
    """Generate unique run ID: YYYYMMDD_HHMMSS_<uuid8>."""
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"{timestamp}_{short_uuid}"


def compute_input_hash(
    code_manifest: dict[str, str],
    params: dict[str, object],
    deps: list[DepEntry],
    out_specs: list[tuple[str, bool]],
    *,
    marked: bool = True,
) -> str:
    """Compute input hash for run cache key.

    Hash is computed from code manifest, params, dependency hashes, and output specs.
    Output specs include both path and cache flag to ensure cache invalidation when
    an output's cache property changes (e.g., Out -> Metric). Unless ``marked`` is False, the
    key also holds ``_DIRECTORY_RECORD``; unmarked, it is the key an older pivot files under.
    """
    data: dict[str, object] = {
        "code_manifest": code_manifest,
        "params": params,
        "deps": sorted([(d["producer"], d["key"], d["hash"]) for d in deps]),
        "out_specs": sorted(out_specs),
    }
    if marked:
        data["directories"] = _DIRECTORY_RECORD
    content = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def stage_input_hash(
    code_manifest: dict[str, str],
    params: dict[str, object],
    deps: Iterable[ArtifactRef],
    dep_hashes: Mapping[ArtifactIdentity, HashInfo],
    outs: Iterable[ArtifactRef],
    *,
    marked: bool = True,
) -> str:
    """Compute a stage's run-cache key from its registered deps and outs (``compute_input_hash``).

    One entry per distinct dependency artifact, so a stage that reads one artifact through
    several parameters gets the same key from every caller (worker, coordinator, commit).
    """
    dep_entries = [
        DepEntry(producer=identity.producer, key=identity.key, hash=dep_hashes[identity]["hash"])
        for identity in {ref.identity for ref in deps}
    ]
    out_specs = [(identity_key(ref.identity), ref.tag is not ArtifactTag.METRIC) for ref in outs]
    return compute_input_hash(code_manifest, params, dep_entries, out_specs, marked=marked)


def serialize_to_bytes(data: RunManifest | RunCacheEntry) -> bytes:
    """Serialize TypedDict to bytes for LMDB storage."""
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()


def deserialize_run_manifest(data: bytes) -> RunManifest:
    """Deserialize bytes to RunManifest with validation."""
    parsed: dict[str, Any] = json.loads(data.decode())
    required = {"run_id", "started_at", "ended_at", "targeted_stages", "execution_order", "stages"}
    missing = required - parsed.keys()
    if missing:
        msg = f"Invalid RunManifest: missing keys {missing}"
        raise ValueError(msg)

    stages: dict[str, StageRunRecord] = {}
    for stage_name, record in parsed["stages"].items():
        raw_status = record["status"]
        if raw_status == "skipped":
            raw_status = "cached"
        try:
            status = StageStatus(raw_status)
        except ValueError:
            msg = f"Invalid status '{record['status']}' for stage '{stage_name}'"
            raise ValueError(msg) from None
        stages[stage_name] = StageRunRecord(
            input_hash=record["input_hash"],
            status=ensure_completion_type(status),
            reason=record["reason"],
            duration_ms=record["duration_ms"],
        )

    execution_order = parsed["execution_order"]

    return RunManifest(
        run_id=parsed["run_id"],
        started_at=parsed["started_at"],
        ended_at=parsed["ended_at"],
        targeted_stages=parsed["targeted_stages"],
        execution_order=execution_order,
        stages=stages,
    )


def deserialize_run_cache_entry(data: bytes) -> RunCacheEntry:
    """Deserialize bytes to RunCacheEntry with validation."""
    parsed: dict[str, Any] = json.loads(data.decode())
    required = {"run_id", "output_hashes"}
    missing = required - parsed.keys()
    if missing:
        msg = f"Invalid RunCacheEntry: missing keys {missing}"
        raise ValueError(msg)
    entry = RunCacheEntry(run_id=parsed["run_id"], output_hashes=parsed["output_hashes"])
    if "reads" in parsed:
        reads = parsed["reads"]
        if not isinstance(reads, list) or not all(
            cache.is_valid_hash(item) for item in cast("list[object]", reads)
        ):
            msg = f"Invalid RunCacheEntry: reads is not a list of content hashes: {reads!r}"
            raise ValueError(msg)
        entry["reads"] = cast("list[str]", reads)
    return entry
