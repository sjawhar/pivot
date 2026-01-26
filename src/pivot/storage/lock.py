"""Per-stage lock files for tracking pipeline state.

This module provides two locking mechanisms:

1. StageLock - Persistent lock files (.lock) for change detection
   Stores fingerprints, params, and hashes to detect when re-runs are needed.

2. Execution locks - Runtime sentinel files (.running) for mutual exclusion
   Prevents concurrent execution of the same stage across processes.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import tempfile
from typing import TYPE_CHECKING, Any, TypeGuard, cast

import yaml

from pivot import exceptions, project, yaml_config
from pivot.storage import cache
from pivot.types import DepEntry, HashInfo, LockData, OutEntry, OutputHash, StorageLockData

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

logger = logging.getLogger(__name__)

_VALID_STAGE_NAME = re.compile(r"^[a-zA-Z0-9_@.-]+$")  # Allow . for DVC matrix keys like @0.5
_MAX_STAGE_NAME_LEN = 200  # Leave room for ".lock" suffix within filesystem NAME_MAX (255)
_REQUIRED_LOCK_KEYS = frozenset({"code_manifest", "params", "deps", "outs", "dep_generations"})

# Pending directory for --no-commit mode (relative to .pivot/)
_PENDING_DIR = "pending"
STAGES_REL_PATH = ".pivot/stages"


def get_stages_dir(state_dir: Path) -> Path:
    """Return the stages directory for lock files.

    Lock files are stored in .pivot/stages/ (git-tracked) rather than
    .pivot/cache/stages/ so they can be versioned for reproducibility.
    """
    return state_dir / "stages"


def get_pending_stages_dir(project_root: Path) -> Path:
    """Return the pending stages directory for --no-commit mode."""
    return project_root / ".pivot" / _PENDING_DIR / "stages"


def is_lock_data(data: object) -> TypeGuard[StorageLockData]:
    """Validate that parsed YAML has valid storage format structure."""
    if not isinstance(data, dict):
        return False
    typed_data = cast("dict[str, object]", data)
    # Require all required keys (allow extra keys for forward compatibility)
    if not _REQUIRED_LOCK_KEYS.issubset(typed_data.keys()):
        return False
    # Reject null values for required keys (corrupted data)
    return all(typed_data[key] is not None for key in _REQUIRED_LOCK_KEYS)


def _convert_to_storage_format(data: LockData) -> StorageLockData:
    """Convert internal LockData to storage format (list-based, relative paths, sorted)."""
    proj_root = project.get_project_root()

    deps_list = list[DepEntry]()
    for abs_path, hash_info in data["dep_hashes"].items():
        rel_path = project.to_relative_path(abs_path, proj_root)
        entry = DepEntry(path=rel_path, hash=hash_info["hash"])
        if "manifest" in hash_info:
            entry["manifest"] = hash_info["manifest"]
        deps_list.append(entry)
    deps_list.sort(key=lambda e: e["path"])

    outs_list = list[OutEntry]()
    for abs_path, hash_info in data["output_hashes"].items():
        rel_path = project.to_relative_path(abs_path, proj_root)
        if hash_info is None:
            entry = OutEntry(path=rel_path, hash=None)
        else:
            entry = OutEntry(path=rel_path, hash=hash_info["hash"])
            if "manifest" in hash_info:
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
        dep_generations=data["dep_generations"],
    )


def _convert_from_storage_format(data: StorageLockData) -> LockData:
    """Convert storage format (list-based, relative paths) to internal LockData."""
    proj_root = project.get_project_root()

    dep_hashes = dict[str, HashInfo]()
    for entry in data["deps"]:
        abs_path = str(project.to_absolute_path(entry["path"], proj_root))
        if "manifest" in entry:
            dep_hashes[abs_path] = {"hash": entry["hash"], "manifest": entry["manifest"]}
        else:
            dep_hashes[abs_path] = {"hash": entry["hash"]}

    output_hashes = dict[str, OutputHash]()
    for entry in data["outs"]:
        abs_path = str(project.to_absolute_path(entry["path"], proj_root))
        if entry["hash"] is None:
            output_hashes[abs_path] = None
        elif "manifest" in entry:
            output_hashes[abs_path] = {"hash": entry["hash"], "manifest": entry["manifest"]}
        else:
            output_hashes[abs_path] = {"hash": entry["hash"]}

    result = LockData(
        code_manifest=data["code_manifest"],
        params=data["params"],
        dep_hashes=dep_hashes,
        output_hashes=output_hashes,
        dep_generations=data["dep_generations"],
    )

    return result


class StageLock:
    """Manages lock file for a single pipeline stage."""

    stage_name: str
    path: Path

    def __init__(self, stage_name: str, stages_dir: Path) -> None:
        """Initialize a stage lock for the given stage in stages_dir."""
        if not stage_name or not _VALID_STAGE_NAME.match(stage_name):
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


def get_pending_lock(stage_name: str, project_root: Path) -> StageLock:
    """Get StageLock pointing to pending directory for --no-commit mode."""
    return StageLock(stage_name, get_pending_stages_dir(project_root))


def list_pending_stages(project_root: Path) -> list[str]:
    """List all stages with pending lock files."""
    pending_dir = get_pending_stages_dir(project_root)
    if not pending_dir.exists():
        return []
    return sorted(p.stem for p in pending_dir.glob("*.lock"))


# =============================================================================
# Execution Locks - Runtime Mutual Exclusion
# =============================================================================
#
# Prevents concurrent execution of the same stage across processes using
# sentinel files (.running) with PID-based ownership.
#
# Key Scenarios:
# ┌────────────────────────────────────┬─────────────────────────────────────┐
# │ Scenario                           │ Behavior                            │
# ├────────────────────────────────────┼─────────────────────────────────────┤
# │ No lock exists                     │ Create atomically → SUCCESS         │
# │ Lock exists, process alive         │ FAIL immediately with error         │
# │ Lock exists, process dead (stale)  │ Atomic takeover → SUCCESS           │
# │ Lock file corrupted/empty          │ Treat as stale → attempt takeover   │
# │ Race: 2+ processes see stale lock  │ All try takeover, one wins via      │
# │                                    │ verify-after-replace, losers retry  │
# │ Race: loser retries, winner alive  │ Loser sees alive PID → FAIL         │
# │ All retry attempts exhausted       │ FAIL with "after N attempts" error  │
# └────────────────────────────────────┴─────────────────────────────────────┘

_MAX_LOCK_ATTEMPTS = 3


@contextlib.contextmanager
def execution_lock(stage_name: str, cache_dir: Path) -> Generator[Path]:
    """Context manager for stage execution lock.

    Acquires an exclusive lock before yielding, releases on exit.
    """
    sentinel = acquire_execution_lock(stage_name, cache_dir)
    try:
        yield sentinel
    finally:
        sentinel.unlink(missing_ok=True)


def acquire_execution_lock(stage_name: str, cache_dir: Path) -> Path:
    """Acquire exclusive lock for stage execution. Returns sentinel path.

    Flow:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         acquire_execution_lock()                        │
    └─────────────────────────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    │           RETRY LOOP (up to 3x)       │
                    └───────────────────────────────────────┘
                                        │
                                        ▼
                    ┌───────────────────────────────────────┐
                    │  FAST PATH: Atomic Create             │
                    │  os.open(O_CREAT | O_EXCL | O_WRONLY) │
                    └───────────────────────────────────────┘
                              │                │
                        SUCCESS              FAIL
                        (no lock)        (FileExistsError)
                              │                │
                              ▼                ▼
                    ┌─────────────┐   ┌─────────────────────┐
                    │ Write PID   │   │ Read existing PID   │
                    │ RETURN ✓    │   │ _read_lock_pid()    │
                    └─────────────┘   └─────────────────────┘
                                                │
                                                ▼
                                  ┌─────────────────────────┐
                                  │  PID valid AND alive?   │
                                  └─────────────────────────┘
                                        │           │
                                       YES          NO (stale)
                                        │           │
                                        ▼           ▼
                            ┌──────────────┐  ┌─────────────────────┐
                            │ RAISE ERROR  │  │ _atomic_lock_takeover│
                            │ "already     │  └─────────────────────┘
                            │  running"    │            │
                            └──────────────┘      ┌─────┴─────┐
                                               SUCCESS      FAIL
                                               (we won)   (race lost)
                                                  │           │
                                                  ▼           ▼
                                            ┌──────────┐  ┌──────────┐
                                            │ RETURN ✓ │  │  RETRY   │
                                            └──────────┘  └──────────┘
                                                                │
                                                    (after 3 failures)
                                                                │
                                                                ▼
                                                    ┌────────────────────┐
                                                    │ RAISE ERROR        │
                                                    │ "after 3 attempts" │
                                                    └────────────────────┘
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    sentinel = cache_dir / f"{stage_name}.running"

    for _ in range(_MAX_LOCK_ATTEMPTS):
        # Fast path: try atomic create
        try:
            fd = os.open(sentinel, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write(str(os.getpid()))
            return sentinel
        except FileExistsError:
            pass

        # Lock exists - check if it's stale
        existing_pid = _read_lock_pid(sentinel)

        if existing_pid is not None and _is_process_alive(existing_pid):
            raise exceptions.StageAlreadyRunningError(
                f"Stage '{stage_name}' is already running (PID {existing_pid})"
            )

        # Stale lock detected - attempt atomic takeover
        if _atomic_lock_takeover(sentinel, existing_pid):
            return sentinel

    raise exceptions.StageAlreadyRunningError(
        f"Failed to acquire lock for '{stage_name}' after {_MAX_LOCK_ATTEMPTS} attempts"
    )


def _read_lock_pid(sentinel: Path) -> int | None:
    """Read PID from lock file. Returns None if missing/corrupted/invalid."""
    try:
        pid = int(sentinel.read_text().strip())
        return pid if pid > 0 else None
    except (FileNotFoundError, ValueError, OSError):
        return None


def _atomic_lock_takeover(sentinel: Path, stale_pid: int | None) -> bool:
    """Atomically take over a stale lock using temp file + rename.

    Flow:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    _atomic_lock_takeover()                      │
    └─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │ Create temp file with our PID │
                  │ tempfile.mkstemp()            │
                  └───────────────────────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │ Atomic replace                │
                  │ os.replace(tmp, sentinel)     │
                  └───────────────────────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │ Verify: read back PID         │
                  │ Did WE win the race?          │
                  └───────────────────────────────┘
                            │           │
                        OUR PID      OTHER PID
                        (we won)     (they won)
                            │           │
                            ▼           ▼
                       Return True   Return False

    Returns True if we successfully acquired the lock, False otherwise.
    """
    my_pid = os.getpid()
    fd, tmp_path = tempfile.mkstemp(dir=sentinel.parent, prefix=f".{sentinel.name}.")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(str(my_pid))
        os.replace(tmp_path, sentinel)

        # Verify we still hold the lock (another process may have done the same)
        if _read_lock_pid(sentinel) == my_pid:
            if stale_pid is not None:
                logger.warning(f"Removed stale lock file: {sentinel} (was PID {stale_pid})")
            return True
        return False
    except OSError:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        return False


def _is_process_alive(pid: int) -> bool:
    """Check if process is still running."""
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True  # Process exists but owned by different user
    except ProcessLookupError:
        return False
