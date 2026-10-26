from __future__ import annotations

import datetime
import hashlib
import json
import uuid
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict, cast

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
    """Run cache entry for skip detection."""

    run_id: str
    output_hashes: list[OutputHashEntry]


class OutputHashEntry(TypedDict):
    """Single output hash entry in run cache. A directory carries ``size`` and ``nfiles``; its
    listing is the cache object at its hash."""

    path: str
    hash: str
    size: NotRequired[int]
    nfiles: NotRequired[int]


def output_hash_to_entry(path: str, oh: HashInfo) -> OutputHashEntry:
    """Convert internal HashInfo to serializable OutputHashEntry."""
    entry = OutputHashEntry(path=path, hash=oh["hash"])
    if is_dir_hash(oh):
        entry["size"] = oh["size"]
        entry["nfiles"] = oh["nfiles"]
    return entry


def entry_to_output_hash(entry: OutputHashEntry) -> FileHash | DirHash | None:
    """Convert serialized OutputHashEntry back to internal HashInfo.

    None for an entry an older pivot filed for a directory, with its files listed inline: its
    hash is not a directory hash this pivot computes, so the entry cannot be used. None too for
    a directory entry without its size, which no pivot writes.
    """
    if "manifest" in cast("Mapping[str, object]", entry):
        return None
    if "nfiles" in entry:
        if "size" not in entry:
            return None
        return DirHash(hash=entry["hash"], size=entry["size"], nfiles=entry["nfiles"])
    return FileHash(hash=entry["hash"])


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
) -> str:
    """Compute input hash for run cache key.

    Hash is computed from code manifest, params, dependency hashes, and output specs.
    Output specs include both path and cache flag to ensure cache invalidation when
    an output's cache property changes (e.g., Out -> Metric).
    """
    data = {
        "code_manifest": code_manifest,
        "params": params,
        "deps": sorted([(d["producer"], d["key"], d["hash"]) for d in deps]),
        "out_specs": sorted(out_specs),
    }
    content = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def stage_input_hash(
    code_manifest: dict[str, str],
    params: dict[str, object],
    deps: Iterable[ArtifactRef],
    dep_hashes: Mapping[ArtifactIdentity, HashInfo],
    outs: Iterable[ArtifactRef],
) -> str:
    """Compute a stage's run-cache key from its registered deps and outs.

    One entry per distinct dependency artifact, so a stage that reads one artifact through
    several parameters gets the same key from every caller (worker, coordinator, commit).
    """
    dep_entries = [
        DepEntry(producer=identity.producer, key=identity.key, hash=dep_hashes[identity]["hash"])
        for identity in {ref.identity for ref in deps}
    ]
    out_specs = [(identity_key(ref.identity), ref.tag is not ArtifactTag.METRIC) for ref in outs]
    return compute_input_hash(code_manifest, params, dep_entries, out_specs)


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
    return RunCacheEntry(run_id=parsed["run_id"], output_hashes=parsed["output_hashes"])
