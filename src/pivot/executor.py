"""Pipeline executor - runs stages in dependency order.

Executes registered stages, tracking changes via lock files to skip
unchanged stages on subsequent runs.
"""

import contextlib
import hashlib
import logging
import os
import pathlib
from collections.abc import Generator
from typing import Any

from pivot import dag, exceptions, lock, project
from pivot.registry import REGISTRY

logger = logging.getLogger(__name__)


def run(
    stages: list[str] | None = None,
    cache_dir: pathlib.Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Execute pipeline stages in dependency order.

    Args:
        stages: Specific stages to run (and their dependencies). If None, runs all.
        cache_dir: Directory for lock files. Defaults to .pivot/cache.

    Returns:
        Dict of stage_name -> {status: "ran"|"skipped", reason: str}
    """
    if cache_dir is None:
        cache_dir = project.get_project_root() / ".pivot" / "cache"

    graph = REGISTRY.build_dag(validate=True)

    # Validate requested stages exist
    if stages:
        registered = set(graph.nodes())
        unknown = set(stages) - registered
        if unknown:
            raise exceptions.StageNotFoundError(f"Unknown stages: {', '.join(sorted(unknown))}")

    execution_order = dag.get_execution_order(graph, stages)

    results = dict[str, dict[str, Any]]()

    for stage_name in execution_order:
        stage_info = REGISTRY.get(stage_name)
        result = _run_stage(stage_name, stage_info, cache_dir)
        results[stage_name] = result

    return results


def _run_stage(
    stage_name: str,
    stage_info: dict[str, Any],
    cache_dir: pathlib.Path,
) -> dict[str, Any]:
    """Run a single stage if changed, otherwise skip."""
    stage_lock = lock.StageLock(stage_name, cache_dir)

    current_fingerprint = stage_info["fingerprint"]
    current_params = _extract_params(stage_info)
    dep_hashes, missing = _hash_dependencies(stage_info.get("deps", []))

    if missing:
        raise exceptions.DependencyNotFoundError(
            f"Stage '{stage_name}' has missing dependencies: {', '.join(missing)}"
        )

    changed, reason = stage_lock.is_changed(current_fingerprint, current_params, dep_hashes)

    if not changed:
        logger.info(f"Skipping '{stage_name}' (unchanged)")
        return {"status": "skipped", "reason": "unchanged"}

    # Acquire execution lock to prevent concurrent runs
    with _execution_lock(stage_name, cache_dir):
        logger.info(f"Running '{stage_name}' ({reason})")
        stage_info["func"]()
        stage_lock.write(
            {
                "code_manifest": current_fingerprint,
                "params": current_params,
                "dep_hashes": dep_hashes,
            }
        )

    return {"status": "ran", "reason": reason}


def _extract_params(stage_info: dict[str, Any]) -> dict[str, Any]:
    """Extract parameter values from stage signature defaults."""
    sig = stage_info.get("signature")
    if not sig:
        return {}
    return {
        name: param.default
        for name, param in sig.parameters.items()
        if param.default is not param.empty
    }


def _hash_dependencies(deps: list[str]) -> tuple[dict[str, str], list[str]]:
    """Hash all dependency files. Returns (hashes, missing_files)."""
    hashes = dict[str, str]()
    missing = list[str]()
    for dep in deps:
        try:
            hashes[dep] = hash_file(pathlib.Path(dep))
        except FileNotFoundError:
            missing.append(dep)
    return hashes, missing


def hash_file(path: pathlib.Path) -> str:
    """Hash file contents using SHA256 (first 16 hex chars)."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


_MAX_LOCK_ATTEMPTS = 3


@contextlib.contextmanager
def _execution_lock(stage_name: str, cache_dir: pathlib.Path) -> Generator[pathlib.Path]:
    """Context manager for stage execution lock."""
    sentinel = _acquire_execution_lock(stage_name, cache_dir)
    try:
        yield sentinel
    finally:
        sentinel.unlink(missing_ok=True)


def _acquire_execution_lock(stage_name: str, cache_dir: pathlib.Path) -> pathlib.Path:
    """Acquire exclusive lock for stage execution. Returns sentinel path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    sentinel = cache_dir / f"{stage_name}.running"

    for _ in range(_MAX_LOCK_ATTEMPTS):
        try:
            # Atomic create - fails if file exists
            fd = os.open(sentinel, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"pid: {os.getpid()}\n".encode())
            os.close(fd)
            return sentinel
        except FileExistsError:
            pass

        # Lock exists - check if holder is alive
        try:
            content = sentinel.read_text()
            pid = int(content.split(":")[1].strip())
        except (ValueError, IndexError, OSError):
            pid = None  # Corrupted lock file

        if pid is not None and pid > 0 and _is_process_alive(pid):
            raise exceptions.StageAlreadyRunningError(
                f"Stage '{stage_name}' is already running (PID {pid})"
            )

        # Stale or corrupted lock - remove and retry
        sentinel.unlink(missing_ok=True)
        if pid is not None:
            logger.warning(f"Removed stale lock file: {sentinel} (was PID {pid})")

    raise exceptions.StageAlreadyRunningError(
        f"Failed to acquire lock for '{stage_name}' after {_MAX_LOCK_ATTEMPTS} attempts"
    )


def _is_process_alive(pid: int) -> bool:
    """Check if process is still running."""
    try:
        os.kill(pid, 0)  # Signal 0 just checks existence
        return True
    except PermissionError:
        return True  # Process exists but owned by different user
    except ProcessLookupError:
        return False
