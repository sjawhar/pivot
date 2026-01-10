from __future__ import annotations

import datetime
import hashlib
import json
import uuid
from typing import Any, TypedDict

from pivot.types import DepEntry, LockData, StageStatus


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


def generate_run_id() -> str:
    """Generate unique run ID: YYYYMMDD_HHMMSS_<uuid8>."""
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"{timestamp}_{short_uuid}"


def compute_input_hash(
    code_manifest: dict[str, str],
    params: dict[str, object],
    deps: list[DepEntry],
    out_paths: list[str],
) -> str:
    """Compute input hash for run cache key.

    Hash is computed from code manifest, params, dependency hashes, and output paths.
    Output hashes are NOT included (they're stored in the value, not the key).
    """
    data = {
        "code_manifest": code_manifest,
        "params": params,
        "deps": sorted([(d["path"], d["hash"]) for d in deps]),
        "out_paths": sorted(out_paths),
    }
    content = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def compute_input_hash_from_lock(lock_data: LockData, out_paths: list[str]) -> str:
    """Compute input hash from existing lock data."""
    deps = [
        DepEntry(path=path, hash=info["hash"]) for path, info in lock_data["dep_hashes"].items()
    ]
    return compute_input_hash(
        code_manifest=lock_data["code_manifest"],
        params=lock_data["params"],
        deps=deps,
        out_paths=out_paths,
    )


def serialize_to_bytes(data: RunManifest | RunCacheEntry) -> bytes:
    """Serialize TypedDict to bytes for LMDB storage."""
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()


def deserialize_run_manifest(data: bytes) -> RunManifest:
    """Deserialize bytes to RunManifest with validation."""
    parsed: dict[str, Any] = json.loads(data.decode())
    required = {"run_id", "started_at", "ended_at", "targeted_stages", "stages"}
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

    # execution_order is optional for backwards compatibility with old entries
    execution_order = parsed.get("execution_order", sorted(stages.keys()))

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
    return RunCacheEntry(
        run_id=parsed["run_id"],
        output_hashes=parsed["output_hashes"],
    )
