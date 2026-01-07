"""Per-stage lock files for tracking pipeline state.

Each stage gets its own lock file (~1KB) instead of one monolithic file.
This enables parallel writes without contention and O(1) reads per stage.

Atomic Write Pattern:
    We write to a .tmp file first, then rename to the final path. This ensures:
    1. No partial/corrupted files if process is killed mid-write
    2. Readers never see incomplete data (rename is atomic on POSIX)
    3. Original file preserved until new version is complete

    Without this pattern, a crash during write would leave a truncated file
    that fails to parse, breaking all subsequent runs.
"""

import os
import pathlib
import re
from typing import Any, TypeGuard, cast

import yaml

from pivot import cache, yaml_config
from pivot.types import HashInfo, LockData

_VALID_STAGE_NAME = re.compile(r"^[a-zA-Z0-9_@-]+$")
_MAX_STAGE_NAME_LEN = 200  # Leave room for ".lock" suffix within filesystem NAME_MAX (255)
_VALID_LOCK_KEYS = frozenset({"code_manifest", "params", "dep_hashes", "output_hashes"})


def _is_lock_data(data: object) -> TypeGuard[LockData]:
    """Validate that parsed YAML has valid LockData structure."""
    if not isinstance(data, dict):
        return False
    # All keys in LockData are optional, but reject unknown keys
    # YAML dicts have string keys; cast for type checker after isinstance
    return all(key in _VALID_LOCK_KEYS for key in cast("dict[str, object]", data))


class StageLock:
    """Manages lock file for a single pipeline stage."""

    stage_name: str
    path: pathlib.Path

    def __init__(self, stage_name: str, cache_dir: pathlib.Path) -> None:
        if not stage_name or not _VALID_STAGE_NAME.match(stage_name):
            raise ValueError(f"Invalid stage name: {stage_name!r}")
        if len(stage_name) > _MAX_STAGE_NAME_LEN:
            raise ValueError(f"Stage name too long ({len(stage_name)} > {_MAX_STAGE_NAME_LEN})")
        self.stage_name = stage_name
        self.path = cache_dir / "stages" / f"{stage_name}.lock"

    def read(self) -> LockData | None:
        """Read lock file, return None if missing or corrupted."""
        try:
            with open(self.path) as f:
                data: object = yaml.load(f, Loader=yaml_config.Loader)
            if not _is_lock_data(data):
                return None  # Treat corrupted/invalid file as missing
            return data
        except (FileNotFoundError, UnicodeDecodeError, yaml.YAMLError):
            return None

    def write(self, data: LockData) -> None:
        """Write lock file atomically."""

        def write_yaml(fd: int) -> None:
            with os.fdopen(fd, "w") as f:
                yaml.dump(data, f, Dumper=yaml_config.Dumper, sort_keys=False)

        cache.atomic_write_file(self.path, write_yaml)

    def is_changed(
        self,
        current_fingerprint: dict[str, str],
        current_params: dict[str, Any],
        dep_hashes: dict[str, HashInfo],
    ) -> tuple[bool, str]:
        """Check if stage needs re-run."""
        lock_data = self.read()
        if not lock_data:
            return True, "No previous run"

        # Check code_manifest and params directly
        if (lock_data.get("code_manifest") or {}) != current_fingerprint:
            return True, "Code changed"
        if (lock_data.get("params") or {}) != current_params:
            return True, "Params changed"

        # Check dep_hashes with path normalization for cached paths
        # (backward compat: old lock files may have resolved paths)
        from pivot import project

        cached_dep_hashes = lock_data.get("dep_hashes") or {}
        # Normalize cached paths (current dep_hashes already have normalized keys)
        cached_normalized = {
            str(project.normalize_path(p)): h for p, h in cached_dep_hashes.items()
        }

        if cached_normalized != dep_hashes:
            return True, "Input dependencies changed"

        return False, ""
