# pyright: reportImplicitRelativeImport=false
"""Per-stage lock files for tracking pipeline state.

StageLock provides persistent lock files (.lock) for change detection,
storing fingerprints, params, and hashes to detect when re-runs are needed.

For runtime mutual exclusion during concurrent execution, see
``pivot.storage.artifact_lock`` which uses flock-based artifact locks.
"""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING, Any, TypeGuard, cast

import yaml

from pivot import yaml_config
from pivot.storage import cache
from pivot.types import (
    ArtifactIdentity,
    DepEntry,
    DirHash,
    FileHash,
    HashInfo,
    LockData,
    OutEntry,
    StorageLockData,
    identity_from_key,
    is_dir_hash,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_VALID_STAGE_NAME = re.compile(
    r"^[a-zA-Z0-9_@./-]+$"
)  # Allow / for pipeline-prefixed names, . for DVC matrix keys
_PATH_TRAVERSAL = re.compile(r"(^|/)\.\.(/|$)")  # Reject ../ path traversal
_MAX_STAGE_NAME_LEN = 200  # Leave room for ".lock" suffix within filesystem NAME_MAX (255)
_REQUIRED_LOCK_KEYS = frozenset({"code_manifest", "params", "deps", "outs"})

STAGES_REL_PATH = ".pivot/stages"


def get_stages_dir(state_dir: Path) -> Path:
    """Return the stages directory for lock files.

    Lock files are stored in .pivot/stages/ (git-tracked) rather than
    .pivot/cache/stages/ so they can be versioned for reproducibility.
    """
    return state_dir / "stages"


def is_lock_data(data: object) -> TypeGuard[StorageLockData]:
    """Validate that parsed YAML has valid storage format structure.

    Rejects lock files with null/empty hashes in deps or outs entries,
    which can occur when stages were never executed locally (e.g., pulled
    from remote with incomplete state). Callers already handle None returns
    gracefully (treated as "no lock = needs re-run").
    """
    if not isinstance(data, dict):
        return False
    # Cast to dict[str, object] for type-safe key access
    typed_data = cast("dict[str, object]", data)
    # Require all required keys (allow extra keys for forward compatibility)
    if not _REQUIRED_LOCK_KEYS.issubset(typed_data.keys()):
        return False
    # Reject null values for required keys (corrupted data)
    if not all(typed_data[key] is not None for key in _REQUIRED_LOCK_KEYS):
        return False
    # Validate that deps and outs entries have non-null hash values.
    # YAML `hash: null` deserializes to None, violating the `hash: str` contract
    # on FileHash/DirHash. Reject at the boundary so consumers never see it.
    for list_key in ("deps", "outs"):
        entries = typed_data[list_key]
        if not isinstance(entries, list):
            return False
        entry_list = cast("list[object]", entries)
        for raw_entry in entry_list:
            if not isinstance(raw_entry, dict):
                return False
            typed_entry = cast("dict[str, object]", raw_entry)
            if not typed_entry.get("hash"):
                return False
            if list_key == "deps" and ("producer" not in typed_entry or "key" not in typed_entry):
                return False
            if list_key == "outs" and ("key" not in typed_entry or "tag" not in typed_entry):
                return False
    return True


def _get_output_tag(hash_info: HashInfo) -> str:
    raw: dict[str, object] = cast("dict[str, object]", cast("object", hash_info))
    value = raw.get("tag")
    if isinstance(value, str):
        return value
    return "data"


def _convert_to_storage_format(data: LockData) -> StorageLockData:
    """Convert internal LockData to storage format (list-based, identity keys, sorted)."""
    normalized_deps = _ensure_identity_keys(data["dep_hashes"])
    normalized_outs = _ensure_identity_keys(data["output_hashes"])

    deps_list = list[DepEntry]()
    for identity, hash_info in normalized_deps.items():
        entry = DepEntry(producer=identity.producer, key=identity.key, hash=hash_info["hash"])
        if is_dir_hash(hash_info):
            entry["manifest"] = hash_info["manifest"]
        raw_info = dict(hash_info)  # shallow copy to check extra keys
        if "accessed_keys" in raw_info:
            entry["accessed_keys"] = cast("list[str]", raw_info["accessed_keys"])
        if "accessed_hashes" in raw_info:
            entry["accessed_hashes"] = cast("dict[str, str]", raw_info["accessed_hashes"])
        deps_list.append(entry)
    deps_list.sort(key=lambda e: (e["producer"], e["key"] or ""))

    outs_list = list[OutEntry]()
    for identity, hash_info in normalized_outs.items():
        entry = OutEntry(key=identity.key, hash=hash_info["hash"], tag=_get_output_tag(hash_info))
        if is_dir_hash(hash_info):
            entry["manifest"] = hash_info["manifest"]
        outs_list.append(entry)
    outs_list.sort(key=lambda e: e["key"] or "")

    sorted_code_manifest = dict(sorted(data["code_manifest"].items()))

    storage = StorageLockData(
        schema_version=2,
        code_manifest=sorted_code_manifest,
        params=data["params"],
        deps=deps_list,
        outs=outs_list,
    )
    if "merkle_id" in data:
        mid = data["merkle_id"]
        if mid is not None:
            storage["merkle_id"] = mid
    return storage


def _convert_from_storage_format(data: StorageLockData, *, stage_name: str) -> LockData:
    """Convert storage format (list-based) to internal LockData (identity-keyed)."""
    dep_hashes = dict[ArtifactIdentity, HashInfo]()
    for entry in data["deps"]:
        identity = ArtifactIdentity(entry["producer"], entry["key"])
        if "manifest" in entry:
            hash_info: HashInfo = DirHash(hash=entry["hash"], manifest=entry["manifest"])
        else:
            hash_info = FileHash(hash=entry["hash"])
        if "accessed_keys" in entry or "accessed_hashes" in entry:
            raw = dict[str, object](hash=hash_info["hash"])  # type: ignore[call-overload] - building mutable copy
            if is_dir_hash(hash_info):
                raw["manifest"] = hash_info["manifest"]
            if "accessed_keys" in entry:
                raw["accessed_keys"] = entry["accessed_keys"]
            if "accessed_hashes" in entry:
                raw["accessed_hashes"] = entry["accessed_hashes"]
            dep_hashes[identity] = cast("HashInfo", cast("object", raw))
        else:
            dep_hashes[identity] = hash_info

    output_hashes = dict[ArtifactIdentity, HashInfo]()
    for entry in data["outs"]:
        identity = ArtifactIdentity(stage_name, entry["key"])
        if "manifest" in entry:
            output_hashes[identity] = DirHash(hash=entry["hash"], manifest=entry["manifest"])
        else:
            output_hashes[identity] = FileHash(hash=entry["hash"])

    return LockData(
        code_manifest=data["code_manifest"],
        params=data["params"],
        dep_hashes=dep_hashes,
        output_hashes=output_hashes,
        merkle_id=data.get("merkle_id"),
    )


def _ensure_identity_keys(
    hashes: dict[ArtifactIdentity, HashInfo] | dict[str, HashInfo],
) -> dict[ArtifactIdentity, HashInfo]:
    if not hashes:
        return {}
    first_key = next(iter(hashes))
    if isinstance(first_key, ArtifactIdentity):
        return cast("dict[ArtifactIdentity, HashInfo]", hashes)
    return {identity_from_key(k): v for k, v in cast("dict[str, HashInfo]", hashes).items()}


def _ensure_identity_list(
    paths: list[ArtifactIdentity] | list[str],
) -> list[ArtifactIdentity]:
    if not paths:
        return []
    if isinstance(paths[0], ArtifactIdentity):
        return cast("list[ArtifactIdentity]", paths)
    return [identity_from_key(p) for p in cast("list[str]", paths)]


class StageLock:
    """Manages lock file for a single pipeline stage."""

    stage_name: str
    path: Path

    def __init__(self, stage_name: str, stages_dir: Path) -> None:
        """Initialize a stage lock for the given stage in stages_dir."""
        if (
            not stage_name
            or not _VALID_STAGE_NAME.match(stage_name)
            or _PATH_TRAVERSAL.search(stage_name)
        ):
            raise ValueError(f"Invalid stage name: {stage_name!r}")
        if len(stage_name) > _MAX_STAGE_NAME_LEN:
            raise ValueError(f"Stage name too long ({len(stage_name)} > {_MAX_STAGE_NAME_LEN})")
        self.stage_name = stage_name
        self.path = stages_dir / f"{stage_name}.lock"

    def read(self) -> LockData | None:
        """Read lock file, converting storage format to internal format."""
        try:
            with open(self.path) as f:
                data: object = yaml.load(f, Loader=yaml_config.Loader)
            if not is_lock_data(data):
                if isinstance(data, dict):
                    # Cast to get typed keys for debug logging
                    actual_keys = set(cast("dict[str, object]", data).keys())
                    logger.debug(
                        "Lock file validation failed for %s: keys=%s, expected=%s",
                        self.path,
                        actual_keys,
                        _REQUIRED_LOCK_KEYS,
                    )
                return None  # Treat corrupted/invalid file as missing
            return _convert_from_storage_format(data, stage_name=self.stage_name)
        except FileNotFoundError:
            return None  # Normal case - lock doesn't exist yet
        except (UnicodeDecodeError, yaml.YAMLError) as e:
            logger.warning("Failed to parse lock file %s: %s", self.path, e)
            return None

    def write(self, data: LockData) -> None:
        """Write lock file atomically, converting to storage format."""
        storage_data = _convert_to_storage_format(data)

        def write_yaml(fd: int) -> None:
            with os.fdopen(fd, "w") as f:
                yaml.dump(storage_data, f, Dumper=yaml_config.Dumper, sort_keys=False)

        cache.atomic_write_file(self.path, write_yaml)

    def is_changed(
        self,
        current_fingerprint: dict[str, str],
        current_params: dict[str, Any],
        dep_hashes: dict[ArtifactIdentity, HashInfo] | dict[str, HashInfo],
        out_paths: list[ArtifactIdentity] | list[str] | None = None,
    ) -> tuple[bool, str]:
        lock_data = self.read()
        return self.is_changed_with_lock_data(
            lock_data, current_fingerprint, current_params, dep_hashes, out_paths
        )

    def is_changed_with_lock_data(
        self,
        lock_data: LockData | None,
        current_fingerprint: dict[str, str],
        current_params: dict[str, Any],
        dep_hashes: dict[ArtifactIdentity, HashInfo] | dict[str, HashInfo],
        out_paths: list[ArtifactIdentity] | list[str] | None = None,
    ) -> tuple[bool, str]:
        if lock_data is None:
            return True, "No previous run"

        if lock_data["code_manifest"] != current_fingerprint:
            return True, "Code changed"
        if lock_data["params"] != current_params:
            return True, "Params changed"
        normalized_deps = _ensure_identity_keys(dep_hashes)
        if lock_data["dep_hashes"] != normalized_deps:
            return True, "Input dependencies changed"
        if out_paths is not None:
            locked_out_paths = sorted(lock_data["output_hashes"].keys())
            normalized_outs = _ensure_identity_list(out_paths)
            if sorted(normalized_outs) != locked_out_paths:
                return True, "Output paths changed"

        return False, ""


def find_orphaned_locks(
    stages_dir: Path,
    registered_stages: set[str],
) -> list[str]:
    """Find lock files in stages_dir that don't correspond to any registered stage.

    When a pipeline is restructured (stages renamed/removed), old lock files
    persist and old output files remain on disk. Detecting orphaned locks helps
    users identify stale artifacts from previous pipeline configurations.

    Returns:
        Sorted list of stage names that have lock files but aren't registered.
    """
    if not stages_dir.is_dir():
        return []

    orphaned = list[str]()
    suffix = ".lock"

    for lock_file in stages_dir.rglob(f"*{suffix}"):
        # Derive stage name from relative path (e.g., "base/stage_name.lock" -> "base/stage_name")
        rel = lock_file.relative_to(stages_dir)
        stage_name = str(rel)[: -len(suffix)]
        if stage_name not in registered_stages:
            orphaned.append(stage_name)

    orphaned.sort()
    return orphaned
