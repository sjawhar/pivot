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

from pivot import path_utils, project, yaml_config
from pivot.storage import cache
from pivot.types import (
    DepEntry,
    DirHash,
    FileHash,
    HashInfo,
    LockData,
    OutEntry,
    StorageLockData,
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
    return True


def _convert_to_storage_format(data: LockData) -> StorageLockData:
    """Convert internal LockData to storage format (list-based, relative paths, sorted)."""
    proj_root = project.get_project_root()

    deps_list = list[DepEntry]()
    for abs_path, hash_info in data["dep_hashes"].items():
        rel_path = project.to_relative_path(abs_path, proj_root)
        entry = DepEntry(path=rel_path, hash=hash_info["hash"])
        if is_dir_hash(hash_info):
            entry["manifest"] = hash_info["manifest"]
        deps_list.append(entry)
    deps_list.sort(key=lambda e: e["path"])

    outs_list = list[OutEntry]()
    for abs_path, hash_info in data["output_hashes"].items():
        rel_path = project.to_relative_path(abs_path, proj_root)
        rel_path = path_utils.preserve_trailing_slash(abs_path, rel_path)
        entry = OutEntry(path=rel_path, hash=hash_info["hash"])
        if is_dir_hash(hash_info):
            entry["manifest"] = hash_info["manifest"]
        outs_list.append(entry)
    outs_list.sort(key=lambda e: e["path"])

    # Sort code_manifest keys for deterministic output across interpreter sessions
    sorted_code_manifest = dict(sorted(data["code_manifest"].items()))

    return StorageLockData(
        schema_version=1,
        code_manifest=sorted_code_manifest,
        params=data["params"],
        deps=deps_list,
        outs=outs_list,
    )


def _convert_from_storage_format(data: StorageLockData) -> LockData:
    """Convert storage format (list-based, relative paths) to internal LockData."""
    proj_root = project.get_project_root()

    dep_hashes = dict[str, HashInfo]()
    for entry in data["deps"]:
        abs_path = str(project.to_absolute_path(entry["path"], proj_root))
        if "manifest" in entry:
            dep_hashes[abs_path] = DirHash(hash=entry["hash"], manifest=entry["manifest"])
        else:
            dep_hashes[abs_path] = FileHash(hash=entry["hash"])

    output_hashes = dict[str, HashInfo]()
    for entry in data["outs"]:
        rel_path = entry["path"]
        abs_path = str(project.to_absolute_path(rel_path, proj_root))
        abs_path = path_utils.preserve_trailing_slash(rel_path, abs_path)
        if "manifest" in entry:
            output_hashes[abs_path] = DirHash(hash=entry["hash"], manifest=entry["manifest"])
        else:
            output_hashes[abs_path] = FileHash(hash=entry["hash"])

    result = LockData(
        code_manifest=data["code_manifest"],
        params=data["params"],
        dep_hashes=dep_hashes,
        output_hashes=output_hashes,
    )

    return result


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
            return _convert_from_storage_format(data)
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
        dep_hashes: dict[str, HashInfo],
        out_paths: list[str] | None = None,
    ) -> tuple[bool, str]:
        """Check if stage needs re-run (reads lock file)."""
        lock_data = self.read()
        return self.is_changed_with_lock_data(
            lock_data, current_fingerprint, current_params, dep_hashes, out_paths
        )

    def is_changed_with_lock_data(
        self,
        lock_data: LockData | None,
        current_fingerprint: dict[str, str],
        current_params: dict[str, Any],
        dep_hashes: dict[str, HashInfo],
        out_paths: list[str] | None = None,
    ) -> tuple[bool, str]:
        """Check if stage needs re-run (pure comparison, no I/O)."""
        if lock_data is None:
            return True, "No previous run"

        if lock_data["code_manifest"] != current_fingerprint:
            return True, "Code changed"
        if lock_data["params"] != current_params:
            return True, "Params changed"
        if lock_data["dep_hashes"] != dep_hashes:
            return True, "Input dependencies changed"
        if out_paths is not None:
            locked_out_paths = sorted(lock_data["output_hashes"].keys())
            if sorted(out_paths) != locked_out_paths:
                return True, "Output paths changed"

        return False, ""
