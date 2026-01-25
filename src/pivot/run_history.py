from __future__ import annotations

import datetime
import hashlib
import json
import uuid
from typing import Any, NotRequired, TypedDict

from pivot.types import (
    DepEntry,
    DirHash,
    DirManifestEntry,
    FileHash,
    LockData,
    OutputHash,
    StageStatus,
)


class StageRunRecord(TypedDict):
    """Record of a stage execution within a run."""

    input_hash: str
    status: StageStatus
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
    """Single output hash entry in run cache."""

    path: str
    hash: str
    manifest: NotRequired[list[DirManifestEntry]]


def output_hash_to_entry(path: str, oh: OutputHash) -> OutputHashEntry | None:
    """Convert internal OutputHash to serializable OutputHashEntry. Returns None for uncached."""
    if oh is None:
        return None
    entry = OutputHashEntry(path=path, hash=oh["hash"])
    if "manifest" in oh:
        entry["manifest"] = oh["manifest"]
    return entry


def entry_to_output_hash(entry: OutputHashEntry) -> FileHash | DirHash:
    """Convert serialized OutputHashEntry back to internal OutputHash."""
    # Null check guards against corrupted JSON where manifest could be null
    if "manifest" in entry and entry["manifest"] is not None:  # pyright: ignore[reportUnnecessaryComparison]
        return DirHash(hash=entry["hash"], manifest=entry["manifest"])
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
        "deps": sorted([(d["path"], d["hash"]) for d in deps]),
        "out_specs": sorted(out_specs),
    }
    content = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def compute_input_hash_from_lock(lock_data: LockData) -> str:
    """Compute input hash from existing lock data.

    Derives output specs from lock data: outputs with None hash are non-cached.
    """
    deps = [
        DepEntry(path=path, hash=info["hash"]) for path, info in lock_data["dep_hashes"].items()
    ]
    # Derive cache flag from output hash: None means cache=False
    out_specs = [(path, oh is not None) for path, oh in lock_data["output_hashes"].items()]
    return compute_input_hash(
        code_manifest=lock_data["code_manifest"],
        params=lock_data["params"],
        deps=deps,
        out_specs=out_specs,
    )


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
        try:
            status = StageStatus(record["status"])
        except ValueError:
            msg = f"Invalid status '{record['status']}' for stage '{stage_name}'"
            raise ValueError(msg) from None
        stages[stage_name] = StageRunRecord(
            input_hash=record["input_hash"],
            status=status,
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
