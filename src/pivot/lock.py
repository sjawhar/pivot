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

from pivot import cache, exceptions, yaml_config

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Generator

    from pivot.types import HashInfo, LockData

logger = logging.getLogger(__name__)

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
def execution_lock(stage_name: str, cache_dir: pathlib.Path) -> Generator[pathlib.Path]:
    """Context manager for stage execution lock.

    Acquires an exclusive lock before yielding, releases on exit.
    """
    sentinel = acquire_execution_lock(stage_name, cache_dir)
    try:
        yield sentinel
    finally:
        sentinel.unlink(missing_ok=True)


def acquire_execution_lock(stage_name: str, cache_dir: pathlib.Path) -> pathlib.Path:
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
                f.write(f"pid: {os.getpid()}\n")
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


def _read_lock_pid(sentinel: pathlib.Path) -> int | None:
    """Read PID from lock file. Returns None if missing/corrupted/invalid."""
    try:
        content = sentinel.read_text()
        pid = int(content.split(":")[1].strip())
        return pid if pid > 0 else None
    except (FileNotFoundError, ValueError, IndexError, OSError):
        return None


def _atomic_lock_takeover(sentinel: pathlib.Path, stale_pid: int | None) -> bool:
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
            f.write(f"pid: {my_pid}\n")
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
