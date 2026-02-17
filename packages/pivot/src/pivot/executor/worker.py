# pyright: reportImplicitRelativeImport=false, reportMissingImports=false
"""Worker process execution for pipeline stages.

Functions that execute in separate processes via ProcessPoolExecutor.
Must be module-level and picklable.
"""

from __future__ import annotations

import collections
import collections.abc as collections_abc
import contextlib
import logging
import os
import pathlib
import queue
import random
import sys
import threading
import traceback
from typing import IO, TYPE_CHECKING, Any, Literal, TypedDict, cast, override

import pydantic

from pivot import (
    exceptions,
    metrics,
    outputs,
    parameters,
    path_utils,
    project,
    run_history,
    stage_def,
    types,
)
from pivot.storage import artifact_lock, cache, lock, state
from pivot.storage import store as store_mod
from pivot.types import (
    DeferredWrites,
    DepEntry,
    DirHash,
    FileHash,
    HashInfo,
    LockData,
    LogMessage,
    OutputMessage,
    OutputMessageKind,
    StageResult,
    StageStatus,
    StateChange,
    is_dir_hash,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence
    from inspect import Signature
    from multiprocessing import Queue
    from types import TracebackType

logger = logging.getLogger(__name__)


class _QueueLoggingHandler(logging.Handler):
    """Logging handler that sends records to the output queue.

    Installed per-task in execute_stage() to capture log messages from worker processes.
    This ensures logging (e.g., stale lock warnings) appears in TUI Logs panel instead
    of corrupting the display by writing to inherited stderr.
    """

    _stage_name: str
    _queue: Queue[OutputMessage]

    def __init__(
        self, stage_name: str, output_queue: Queue[OutputMessage], level: int = logging.INFO
    ) -> None:
        super().__init__(level=level)
        self._stage_name = stage_name
        self._queue = output_queue
        self.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    @override
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            with contextlib.suppress(queue.Full, ValueError, OSError):
                self._queue.put(
                    LogMessage(
                        kind=OutputMessageKind.LOG, stage=self._stage_name, line=msg, is_stderr=True
                    ),
                    block=False,
                )
        except Exception:
            pass  # Never raise from emit - could cause recursion

    @override
    def handleError(self, record: logging.LogRecord) -> None:
        pass  # Suppress default stderr writing to prevent TUI corruption


@contextlib.contextmanager
def _queue_logging(stage_name: str, output_queue: Queue[OutputMessage]) -> Generator[None]:
    """Context manager to capture logging to the output queue.

    Removes stream handlers that write to stdout/stderr (which would corrupt TUI)
    and installs a handler that sends log records to the output queue instead.
    Original handlers are restored on exit.

    Note: Handler manipulation is not fully atomic, but this is acceptable because:
    1. Worker processes are single-threaded for stage execution
    2. Individual addHandler/removeHandler calls are internally synchronized
    3. Worst case is a brief window where a log goes to the wrong handler
    """
    handler = _QueueLoggingHandler(stage_name, output_queue)
    root_logger = logging.getLogger()

    # Remove existing stderr/stdout handlers to prevent TUI corruption
    removed_handlers = list[logging.Handler]()
    for h in root_logger.handlers[:]:
        if isinstance(h, logging.StreamHandler):
            stream_handler = cast("logging.StreamHandler[Any]", h)
            if hasattr(stream_handler, "stream") and stream_handler.stream in (
                sys.stderr,
                sys.stdout,
            ):
                root_logger.removeHandler(stream_handler)
                removed_handlers.append(stream_handler)

    root_logger.addHandler(handler)
    try:
        yield
    finally:
        root_logger.removeHandler(handler)
        for h in removed_handlers:
            root_logger.addHandler(h)


class WorkerStageInfo(TypedDict):
    """Stage info subset passed to worker processes."""

    func: Callable[..., Any]
    fingerprint: dict[str, str]
    deps: dict[str, types.ArtifactRef]
    outs: list[types.ArtifactRef]
    store_spec: store_mod.StoreSpec
    signature: Signature | None
    params: stage_def.StageParams | None
    variant: str | None
    overrides: parameters.ParamsOverrides
    checkout_modes: list[cache.CheckoutMode]
    run_id: str
    force: bool
    no_commit: bool
    params_arg_name: str | None
    project_root: pathlib.Path
    state_dir: pathlib.Path
    collection_params: dict[str, str]


class TrackedDict(dict[str, Any]):
    """Dict wrapper that records which keys were accessed."""

    _accessed_keys: set[str]

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._accessed_keys = set[str]()

    def __getitem__(self, key: str) -> Any:
        self._accessed_keys.add(key)
        return super().__getitem__(key)

    @property
    def accessed_keys(self) -> frozenset[str]:
        return frozenset(self._accessed_keys)


def _make_result(
    status: Literal[StageStatus.RAN, StageStatus.CACHED, StageStatus.FAILED],
    reason: str,
    ring_buffer: _OutputRingBuffer,
    input_hash: str | None = None,
    accessed_dep_keys: dict[str, set[str]] | None = None,
) -> StageResult:
    """Build StageResult with collected metrics for cross-process transfer."""
    result = StageResult(
        status=status,
        reason=reason,
        input_hash=input_hash,
        output_lines=ring_buffer.snapshot(),
        metrics=metrics.get_entries(),
    )
    if accessed_dep_keys is not None:
        result["accessed_dep_keys"] = accessed_dep_keys
    return result


def execute_stage(
    stage_name: str,
    stage_info: WorkerStageInfo,
    cache_dir: pathlib.Path,
    output_queue: Queue[OutputMessage],
) -> StageResult:
    """Worker function executed in separate process. Must be module-level for pickling.

    Flag interactions:
    - --force: Always run stage, even if skip detection says it's unchanged
    """
    # Clear metrics at start - each stage collects its own metrics
    metrics.clear()
    _ = cache_dir
    ring_buffer = _OutputRingBuffer()
    state_db_path = stage_info["state_dir"] / "state.db"
    project_root = stage_info["project_root"]
    store = store_mod.store_from_spec(stage_info["store_spec"])

    # For store-based stages, mark outputs as produced so _resolve_path returns
    # the correct output path (not input path) during run cache skip checks
    if isinstance(store, store_mod.WorkspaceStore):
        for out in stage_info["outs"]:
            store._output_producers.add(out.identity.producer)  # pyright: ignore[reportPrivateUsage]

    # Set project root cache explicitly - workers in reusable pool may have
    # stale cache from previous execution in different project/test.
    project._project_root_cache = project_root  # pyright: ignore[reportPrivateUsage]

    # Ensure worker has correct cwd for this stage (workers in reusable pool
    # may have stale cwd from previous execution in different project).
    # _queue_logging captures log messages to the output queue (for TUI display).
    with contextlib.chdir(project_root), _queue_logging(stage_name, output_queue):
        no_commit = stage_info["no_commit"]

        production_lock = lock.StageLock(stage_name, lock.get_stages_dir(stage_info["state_dir"]))
        current_fingerprint = stage_info["fingerprint"]
        stage_outs = stage_info["outs"]

        params_instance = stage_info["params"]
        overrides = stage_info["overrides"]
        try:
            current_params = parameters.get_effective_params(params_instance, stage_name, overrides)
            if params_instance is not None:
                params_instance = parameters.apply_overrides(params_instance, stage_name, overrides)
        except pydantic.ValidationError as e:
            return _make_result(
                StageStatus.FAILED,
                f"Invalid params override in params.yaml: {e.error_count()} validation error(s)",
                ring_buffer,
            )

        # Acquire artifact locks (READ on deps, WRITE on outs)
        deps_info = stage_info["deps"]
        outs_info = stage_info["outs"]
        lock_requests = artifact_lock.expand_lock_requests(deps_info, outs_info, project_root)
        lock_service = artifact_lock.LocalFlockLockService(stage_info["state_dir"] / "locks")

        def _on_lock_status(_key: str, _mode: artifact_lock.LockMode, _elapsed: float) -> None:
            with contextlib.suppress(queue.Full):
                output_queue.put_nowait(
                    StateChange(
                        kind=OutputMessageKind.STATE, stage=stage_name, state="waiting_on_lock"
                    )
                )

        lock_handle = lock_service.acquire_many(lock_requests, on_status=_on_lock_status)

        input_hash: str | None = None
        try:
            with lock_handle:
                with contextlib.suppress(queue.Full):
                    output_queue.put_nowait(
                        StateChange(kind=OutputMessageKind.STATE, stage=stage_name, state="running")
                    )
                lock_data = production_lock.read()

                with state.StateDB(state_db_path, readonly=True) as state_db:
                    outputs_missing_from_cache = False
                    if lock_data is not None and not stage_info["force"]:
                        out_paths = _get_normalized_out_paths(stage_info)
                        if can_skip_via_generation(
                            stage_name=stage_name,
                            fingerprint=stage_info["fingerprint"],
                            deps=stage_info["deps"],
                            outs_paths=out_paths,
                            current_params=current_params,
                            lock_data=lock_data,
                            state_db=state_db,
                            verify_files=True,
                        ):
                            out_specs = _get_output_specs(stage_info)
                            deps_list = _deps_list_for_input_hash(
                                stage_info, _to_identity_keyed_hashes(lock_data["dep_hashes"])
                            )
                            input_hash = run_history.compute_input_hash(
                                current_fingerprint, current_params, deps_list, out_specs
                            )
                            restored = _outputs_exist_with_store(
                                store,
                                stage_outs,
                                lock_data,
                                stage_info["state_dir"],
                            )
                            if restored:
                                return _make_result(
                                    StageStatus.CACHED,
                                    "unchanged (generation)",
                                    ring_buffer,
                                    input_hash=input_hash,
                                )
                            outputs_missing_from_cache = True

                    dep_hashes, missing, unreadable, file_hash_entries = hash_dependencies(
                        stage_info["deps"], store, state_db
                    )

                    if missing:
                        return _make_result(
                            StageStatus.FAILED,
                            f"missing deps: {', '.join(missing)}",
                            ring_buffer,
                        )

                    if unreadable:
                        return _make_result(
                            StageStatus.FAILED,
                            f"unreadable deps: {', '.join(unreadable)}",
                            ring_buffer,
                        )

                    skip_reason, run_reason, input_hash = _check_skip_or_run(
                        stage_info,
                        production_lock,
                        lock_data,
                        current_fingerprint,
                        current_params,
                        dep_hashes,
                    )

                    if outputs_missing_from_cache and skip_reason is not None:
                        skip_reason = None
                        run_reason = "outputs missing from cache"

                    # Override skip decision if force flag is set
                    if stage_info["force"] and skip_reason is not None:
                        skip_reason = None
                        run_reason = "forced"

                    if skip_reason is not None and lock_data is not None:
                        restored = _outputs_exist_with_store(
                            store,
                            stage_outs,
                            lock_data,
                            stage_info["state_dir"],
                        )
                        if restored:
                            return _make_result(
                                StageStatus.CACHED,
                                skip_reason,
                                ring_buffer,
                                input_hash=input_hash,
                            )
                        run_reason = "outputs missing from cache"

                    # Check run cache for previously executed configuration (skip if forcing)
                    if run_reason and not stage_info["force"]:
                        run_cache_skip = _try_skip_via_run_cache_with_store(
                            stage_name,
                            input_hash,
                            stage_outs,
                            store,
                            state_db,
                        )
                        if run_cache_skip is not None:
                            if no_commit:
                                return _make_result(
                                    StageStatus.CACHED,
                                    "unchanged (run cache)",
                                    ring_buffer,
                                    input_hash=input_hash,
                                )
                            new_lock_data = LockData(
                                code_manifest=current_fingerprint,
                                params=current_params,
                                dep_hashes=_to_identity_hashes(dict(sorted(dep_hashes.items()))),
                                output_hashes=_to_identity_hashes(
                                    dict(sorted(run_cache_skip["output_hashes"].items()))
                                ),
                            )
                            deferred = _commit_lock_and_build_deferred(
                                stage_info,
                                new_lock_data,
                                input_hash,
                                run_cache_skip["output_hashes"],
                                production_lock,
                                state_db,
                                file_hash_entries=file_hash_entries,
                                increment_outputs=False,
                            )
                            return StageResult(
                                status=StageStatus.CACHED,
                                reason="unchanged (run cache)",
                                input_hash=input_hash,
                                output_lines=ring_buffer.snapshot(),
                                metrics=metrics.get_entries(),
                                deferred_writes=deferred,
                            )

                output_paths, accessed_dep_keys = _run_stage_function_with_store(
                    stage_info["func"],
                    stage_name,
                    output_queue,
                    ring_buffer,
                    params_instance,
                    stage_info["deps"],
                    stage_outs,
                    store,
                    stage_info["params_arg_name"],
                    stage_info["collection_params"],
                )
                if no_commit:
                    output_hashes = _hash_output_paths(output_paths)
                else:
                    output_hashes = _commit_outputs_with_store(output_paths, store)

                # For --no-commit, skip lock/cache writes entirely
                if no_commit:
                    return _make_result(
                        StageStatus.RAN,
                        run_reason,
                        ring_buffer,
                        input_hash=input_hash,
                        accessed_dep_keys=accessed_dep_keys,
                    )

                # Build lock data
                new_lock_data = LockData(
                    code_manifest=current_fingerprint,
                    params=current_params,
                    dep_hashes=_to_identity_hashes(dict(sorted(dep_hashes.items()))),
                    output_hashes=_to_identity_hashes(dict(sorted(output_hashes.items()))),
                )

                # Single StateDB open for post-execution work
                with state.StateDB(state_db_path, readonly=True) as state_db:
                    deferred = _commit_lock_and_build_deferred(
                        stage_info,
                        new_lock_data,
                        input_hash,
                        output_hashes,
                        production_lock,
                        state_db,
                        file_hash_entries=file_hash_entries,
                    )
                    result = StageResult(
                        status=StageStatus.RAN,
                        reason=run_reason,
                        input_hash=input_hash,
                        output_lines=ring_buffer.snapshot(),
                        metrics=metrics.get_entries(),
                        deferred_writes=deferred,
                        accessed_dep_keys=accessed_dep_keys,
                    )
                    return result
        except exceptions.OutputMissingError as e:
            return _make_result(StageStatus.FAILED, str(e), ring_buffer, input_hash=input_hash)
        except SystemExit as e:
            return _make_result(
                StageStatus.FAILED,
                f"Stage called sys.exit({e.code})",
                ring_buffer,
                input_hash=input_hash,
            )
        except KeyboardInterrupt:
            return _make_result(
                StageStatus.FAILED,
                "KeyboardInterrupt",
                ring_buffer,
                input_hash=input_hash,
            )
        except Exception:
            return _make_result(
                StageStatus.FAILED,
                traceback.format_exc(),
                ring_buffer,
                input_hash=input_hash,
            )


def _canonicalize_out(path: str) -> str:
    """Canonicalize output path via path_utils.canonicalize_artifact_path."""
    return path_utils.canonicalize_artifact_path(path, project.get_project_root())


def _to_identity_hashes(hashes: dict[str, HashInfo]) -> dict[types.ArtifactIdentity, HashInfo]:
    return {types.identity_from_key(key): info for key, info in hashes.items()}


def _to_identity_keyed_hashes(
    hashes: dict[types.ArtifactIdentity, HashInfo],
) -> dict[str, HashInfo]:
    return {types.identity_key(identity): info for identity, info in hashes.items()}


def _outputs_exist_with_store(
    store: store_mod.Store,
    outs: list[types.ArtifactRef],
    lock_data: LockData | None = None,
    state_dir: pathlib.Path | None = None,
) -> bool:
    if not isinstance(store, store_mod.WorkspaceStore):
        return all(store.exists(ref) for ref in outs)

    for ref in outs:
        if store.exists(ref):
            continue
        if lock_data is None:
            return False
        hash_info = lock_data["output_hashes"].get(ref.identity)
        if hash_info is None:
            return False
        if not store.restore_from_cache(ref, hash_info, state_dir=state_dir):
            return False
    return True


def _get_normalized_out_paths(stage_info: WorkerStageInfo) -> list[str]:
    outs = stage_info.get("outs", [])
    return [types.identity_key(out.identity) for out in outs]


def _get_output_specs(stage_info: WorkerStageInfo) -> list[tuple[str, bool]]:
    outs = stage_info.get("outs", [])
    return [
        (types.identity_key(out.identity), out.tag is not types.ArtifactTag.METRIC) for out in outs
    ]


def _deps_list_for_input_hash(
    stage_info: WorkerStageInfo,
    dep_hashes: dict[str, HashInfo],
) -> list[DepEntry]:
    deps_info = stage_info["deps"]
    entries = list[DepEntry]()
    for ref in deps_info.values():
        dep_key = types.identity_key(ref.identity)
        hash_info = dep_hashes[dep_key]
        entries.append(
            DepEntry(
                producer=ref.identity.producer,
                key=ref.identity.key,
                hash=hash_info["hash"],
            )
        )
    return entries


def _check_skip_or_run(
    stage_info: WorkerStageInfo,
    stage_lock: lock.StageLock,
    lock_data: LockData | None,
    current_fingerprint: dict[str, str],
    current_params: dict[str, Any],
    dep_hashes: dict[str, HashInfo],
) -> tuple[str | None, str, str]:
    """Determine if stage can skip or must run.

    Returns (skip_reason, run_reason, input_hash) where exactly one of skip/run reason is meaningful:
    - If skip_reason is not None: stage can skip, run_reason is empty
    - If skip_reason is None: stage must run, run_reason explains why
    - input_hash is always returned for run cache recording
    """
    out_paths = _get_normalized_out_paths(stage_info)
    out_specs = _get_output_specs(stage_info)
    deps_list = _deps_list_for_input_hash(stage_info, dep_hashes)
    input_hash = run_history.compute_input_hash(
        current_fingerprint, current_params, deps_list, out_specs
    )

    if lock_data is None:
        return None, "No previous run", input_hash

    changed, run_reason = stage_lock.is_changed_with_lock_data(
        lock_data, current_fingerprint, current_params, dep_hashes, out_paths
    )
    if not changed:
        return "unchanged", "", input_hash

    return None, run_reason, input_hash


def _cleanup_restored_paths(restored_paths: list[pathlib.Path]) -> None:
    """Remove partially restored outputs to leave a clean state."""
    for path in restored_paths:
        cache.remove_output(path)


def _restore_outputs(
    output_path_strings: list[str],
    output_hash_map: dict[str, HashInfo],
    files_cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
    *,
    use_normalized_paths: bool = False,
    state_db: state.StateDB | None = None,
    state_dir: pathlib.Path | None = None,
) -> bool:
    """Restore outputs from cache - shared logic for lock file and run cache paths.

    Returns True if all outputs exist or were restored. On failure, cleans up
    any partially restored outputs to leave the filesystem in a clean state.

    For files and directories, also verifies content matches the cached hash and
    reconciles any differences (restores missing/corrupted files, removes extra files).

    Args:
        output_path_strings: Path strings to restore (preserves trailing slash for DirectoryOut)
        output_hash_map: Map of path string -> hash (from lock data or run cache)
        files_cache_dir: Cache directory for file restoration
        checkout_modes: Checkout modes for cache restoration
        use_normalized_paths: If True, normalize paths for lookup (lock data uses
            normalized paths). If False, use raw paths (run cache uses raw paths).
        state_db: Optional state database for hash caching during file verification.
        state_dir: Per-stage state directory for lock files during directory restores.
    """
    restored_paths = list[pathlib.Path]()

    for path_str in output_path_strings:
        path = pathlib.Path(path_str)
        lookup_key = _canonicalize_out(path_str) if use_normalized_paths else path_str

        # Check if output is recorded
        if lookup_key not in output_hash_map:
            _cleanup_restored_paths(restored_paths)
            return False

        output_hash = output_hash_map[lookup_key]

        # Verify content matches cached hash (directories and files)
        if is_dir_hash(output_hash):
            needs_restore = _directory_needs_restore(path, output_hash, state_db)
        else:
            needs_restore = _file_needs_restore(path, output_hash, state_db)

        if not needs_restore:
            continue

        try:
            restored = cache.restore_from_cache(
                path,
                output_hash,
                files_cache_dir,
                checkout_modes=checkout_modes,
                state_dir=state_dir,
            )
        except OSError:
            _cleanup_restored_paths(restored_paths)
            return False

        if restored:
            restored_paths.append(path)
        else:
            _cleanup_restored_paths(restored_paths)
            return False

    return True


def restore_outputs_from_cache(
    stage_outs: Sequence[outputs.ExpandedOut],
    lock_data: LockData,
    files_cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
    *,
    state_db: state.StateDB | None = None,
    state_dir: pathlib.Path | None = None,
) -> bool:
    """Restore missing outputs from cache for lock file skip detection."""
    # Non-cached outputs (Metric) are git-tracked — just verify they exist
    for out in stage_outs:
        if not out.cache and not pathlib.Path(out.path).exists():
            return False

    # Only restore cached outputs from cache
    cached_path_strings = [out.path for out in stage_outs if out.cache]
    output_hashes: dict[str, HashInfo] = _to_identity_keyed_hashes(lock_data["output_hashes"])
    return _restore_outputs(
        cached_path_strings,
        output_hashes,
        files_cache_dir,
        checkout_modes,
        use_normalized_paths=True,
        state_db=state_db,
        state_dir=state_dir,
    )


def _directory_needs_restore(
    path: pathlib.Path, cached_hash: DirHash, state_db: state.StateDB | None = None
) -> bool:
    """Check if directory content differs from cached manifest.

    Returns True if restoration is needed (missing files, extra files,
    or content mismatch).

    Uses hash_directory() internally to ensure the same filtering is applied
    (ignoring __pycache__, .venv, etc.). This ensures consistency between
    hashing and restore checks.
    """
    if not path.exists():
        return True

    try:
        current_hash, _ = cache.hash_directory(path, state_db)
    except OSError:
        return True

    # Compare tree hashes - they include all content and structure
    return current_hash != cached_hash["hash"]


def _file_needs_restore(
    path: pathlib.Path, cached_hash: FileHash, state_db: state.StateDB | None = None
) -> bool:
    """Check if file content differs from cached hash.

    Returns True if restoration is needed (file missing or content mismatch).
    """
    if not path.exists():
        return True

    try:
        current_hash, _ = cache.hash_file(path, state_db)
        return current_hash != cached_hash["hash"]
    except OSError:
        return True


def hash_output(path: pathlib.Path, state_db: state.StateDB | None = None) -> HashInfo:
    """Compute output hash without saving to cache."""
    if path.is_dir():
        tree_hash, manifest = cache.hash_directory(path, state_db)
        return DirHash(hash=tree_hash, manifest=manifest)
    file_hash, _ = cache.hash_file(path, state_db)
    return FileHash(hash=file_hash)


def _set_deterministic_seeds() -> None:
    """Set random seeds for reproducible stage execution.

    Called before each stage to ensure determinism. Users can override
    by calling random.seed() or np.random.seed() in their stage code.
    """
    random.seed(0)
    try:
        import numpy as np

        np.random.seed(0)
    except ImportError:
        pass  # NumPy is optional; stdlib random is still seeded


def _execute_with_joblib_protection(func: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    """Execute stage with joblib threading backend to avoid nested multiprocessing issues.

    By default, configures joblib to use threading backend, which works well for
    GIL-releasing code (NumPy, pandas) and avoids resource_tracker race conditions.

    Users can override via PIVOT_NESTED_PARALLELISM=processes env var, or by wrapping
    their Parallel calls in their own parallel_config() context.
    """
    try:
        from joblib import parallel_config
    except ImportError:
        logger.debug("joblib not installed - nested parallelism protection disabled")
        return func(**kwargs)

    # Environment variable allows users to opt into multiprocessing
    env_override = os.environ.get("PIVOT_NESTED_PARALLELISM")
    if env_override == "processes":
        # Disable memmapping to prevent resource_tracker race conditions.
        # This avoids KeyError tracebacks when Pivot's loky pool and joblib's
        # nested loky pool have concurrent cleanup.
        logger.debug("Nested parallelism: processes mode (memmapping disabled)")
        with parallel_config(backend="loky", max_nbytes=None):
            return func(**kwargs)

    # Default: threading backend (safe for NumPy/pandas workloads)
    logger.debug("Nested parallelism: threading mode")
    with parallel_config(backend="threading"):
        return func(**kwargs)


def _load_dep_from_store(ref: types.ArtifactRef, store: store_mod.Store) -> Any:
    path = store.checkout(ref)
    loader = cast("Any", ref.format)
    if ref.tag is types.ArtifactTag.DIRECTORY:
        dir_path = pathlib.Path(path)
        values = dict[str, Any]()
        for file_path in sorted(p for p in dir_path.rglob("*") if p.is_file()):
            rel_key = file_path.relative_to(dir_path).as_posix()
            values[rel_key] = loader.load(file_path)
        return TrackedDict(values)
    return loader.load(path)


def _resolve_output_values(
    result: Any, outs: list[types.ArtifactRef], stage_name: str
) -> dict[types.ArtifactRef, Any]:
    if not outs:
        if result is not None:
            logger.warning(
                "Stage '%s' returned value but has no Out annotation - discarding", stage_name
            )
        return {}

    if result is None:
        raise RuntimeError(f"Stage '{stage_name}' has output annotations but returned None")

    if len(outs) == 1 and outs[0].identity.key is None:
        return {outs[0]: result}

    if not isinstance(result, collections_abc.Mapping):
        raise RuntimeError(
            f"Stage '{stage_name}' returned {type(result).__name__} but expected a mapping"
        )

    value_map = dict[types.ArtifactRef, Any]()
    missing = list[str]()
    for out in outs:
        key = out.identity.key
        if key is None:
            raise RuntimeError(f"Stage '{stage_name}' has multiple outputs but missing output key")
        if key not in result:
            missing.append(key)
            continue
        value_map[out] = result[key]

    if missing:
        raise KeyError(
            f"Missing return output keys: {sorted(missing)}. Return value keys: {sorted(result.keys())}"
        )

    extra = set(result.keys()) - {out.identity.key for out in outs if out.identity.key is not None}
    if extra:
        logger.warning("Extra keys in return value not declared as outputs: %s", sorted(extra))

    return value_map


def _write_output_with_store(
    ref: types.ArtifactRef, value: Any, store: store_mod.Store
) -> pathlib.Path:
    output_path = store.prepare_output(ref)
    writer = cast("Any", ref.format)
    if ref.tag is types.ArtifactTag.DIRECTORY:
        if not isinstance(value, dict):
            raise RuntimeError(
                f"Directory output for '{ref.identity.producer}' expects dict, got {type(value).__name__}"
            )
        if not value:
            raise ValueError(f"Directory output '{ref.identity.producer}': dict must be non-empty")
        for key, item_value in value.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"Directory output '{ref.identity.producer}': keys must be strings, got {type(key).__name__}"
                )
            normalized = stage_def._validate_directory_out_key(key, ref.identity.producer)
            full_path = output_path / normalized
            full_path.parent.mkdir(parents=True, exist_ok=True)
            writer.save(item_value, full_path)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer.save(value, output_path)
    return output_path


def _reconstruct_list_kwargs(
    kwargs: dict[str, Any],
    collection_params: dict[str, str],
) -> dict[str, Any]:
    # Reverse the ``param[N]`` expansion from compose.Pipeline.build()
    # back into ``param = [val0, val1, …]``.
    lists = dict[str, list[tuple[int, Any]]]()
    result = dict[str, Any]()
    for name, value in kwargs.items():
        bracket = name.rfind("[")
        if bracket != -1 and name.endswith("]"):
            base_name = name[:bracket]
            index = int(name[bracket + 1 : -1])
            if base_name not in lists:
                lists[base_name] = list[tuple[int, Any]]()
            lists[base_name].append((index, value))
        else:
            result[name] = value
    for base_name, indexed in lists.items():
        indexed.sort(key=lambda x: x[0])
        values = [v for _, v in indexed]
        result[base_name] = tuple(values) if collection_params.get(base_name) == "tuple" else values
    for param_name, ctype in collection_params.items():
        if param_name not in result:
            result[param_name] = () if ctype == "tuple" else []
    return result


def _run_stage_function_with_store(
    func: Callable[..., Any],
    stage_name: str,
    output_queue: Queue[OutputMessage],
    ring_buffer: _OutputRingBuffer,
    params: stage_def.StageParams | None,
    deps: dict[str, types.ArtifactRef],
    outs: list[types.ArtifactRef],
    store: store_mod.Store,
    params_arg_name: str | None,
    collection_params: dict[str, str],
) -> tuple[dict[types.ArtifactRef, pathlib.Path], dict[str, set[str]]]:
    with (
        _QueueWriter(stage_name, output_queue, is_stderr=False, ring_buffer=ring_buffer),
        _QueueWriter(stage_name, output_queue, is_stderr=True, ring_buffer=ring_buffer),
    ):
        kwargs = dict[str, Any]()

        if params is not None:
            if params_arg_name is None:
                raise RuntimeError(
                    f"Stage '{stage_name}' has params but params_arg_name is None - this indicates a bug in registration"
                )
            kwargs[params_arg_name] = params

        accessed = dict[str, set[str]]()
        for name, ref in deps.items():
            loaded = _load_dep_from_store(ref, store)
            if isinstance(loaded, TrackedDict):
                kwargs[name] = loaded
            else:
                kwargs[name] = loaded

        kwargs = _reconstruct_list_kwargs(kwargs, collection_params)

        _set_deterministic_seeds()

        result = _execute_with_joblib_protection(func, kwargs)

        output_values = _resolve_output_values(result, outs, stage_name)
        output_paths = dict[types.ArtifactRef, pathlib.Path]()
        for ref, value in output_values.items():
            output_paths[ref] = _write_output_with_store(ref, value, store)

        for name, value in kwargs.items():
            if isinstance(value, TrackedDict):
                accessed[name] = set(value.accessed_keys)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, TrackedDict):
                        accessed.setdefault(name, set()).update(item.accessed_keys)

        return output_paths, accessed


def _hash_output_paths(
    outputs_by_ref: dict[types.ArtifactRef, pathlib.Path],
    state_db: state.StateDB | None = None,
) -> dict[str, HashInfo]:
    output_hashes = dict[str, HashInfo]()
    for ref, path in outputs_by_ref.items():
        if not path.exists():
            raise exceptions.OutputMissingError(f"Stage did not produce output: {path}")
        if path.is_dir():
            tree_hash, manifest = cache.hash_directory(path, state_db)
            output_hashes[types.identity_key(ref.identity)] = DirHash(
                hash=tree_hash, manifest=manifest
            )
        else:
            file_hash, _ = cache.hash_file(path, state_db)
            output_hashes[types.identity_key(ref.identity)] = FileHash(hash=file_hash)
    return output_hashes


def _commit_outputs_with_store(
    outputs_by_ref: dict[types.ArtifactRef, pathlib.Path],
    store: store_mod.Store,
) -> dict[str, HashInfo]:
    output_hashes = dict[str, HashInfo]()
    for ref, path in outputs_by_ref.items():
        if not path.exists():
            raise exceptions.OutputMissingError(f"Stage did not produce output: {path}")
        store.commit(ref, path)
        if isinstance(store, store_mod.WorkspaceStore):
            store.backup_to_cache(ref, path)
        output_hashes[types.identity_key(ref.identity)] = store.hash_artifact(ref)
    return output_hashes


class _OutputRingBuffer:
    """Bounded ring buffer for captured output lines."""

    _lines: collections.deque[tuple[str, bool]]
    _dropped_count: int
    _lock: threading.Lock

    def __init__(self, max_lines: int = 1000) -> None:
        self._lines = collections.deque(maxlen=max_lines)
        self._dropped_count = 0
        self._lock = threading.Lock()

    def append(self, line: str, is_stderr: bool) -> None:
        with self._lock:
            if len(self._lines) == self._lines.maxlen:
                self._dropped_count += 1
            self._lines.append((line, is_stderr))

    def snapshot(self) -> list[tuple[str, bool]]:
        with self._lock:
            lines = list(self._lines)
            if self._dropped_count > 0:
                lines.insert(0, (f"[{self._dropped_count} earlier lines truncated]", False))
            return lines


class _QueueWriter:
    """Context manager for capturing stdout/stderr to a queue.

    Handles stream redirection, output capture, and automatic flushing.
    Implements minimal file-like interface needed by print() and common libraries.
    Thread-safe: multiple threads can write concurrently (needed when nested
    joblib uses threading backend).
    """

    _stage_name: str
    _queue: Queue[OutputMessage]
    _is_stderr: bool
    _ring_buffer: _OutputRingBuffer
    _buffer: str
    _redirect: contextlib.AbstractContextManager[object]
    _lock: threading.Lock
    _read_fd: int | None
    _write_fd: int | None
    _reader_thread: threading.Thread | None

    def __init__(
        self,
        stage_name: str,
        output_queue: Queue[OutputMessage],
        *,
        is_stderr: bool,
        ring_buffer: _OutputRingBuffer,
    ) -> None:
        self._stage_name = stage_name
        self._queue = output_queue
        self._is_stderr = is_stderr
        self._ring_buffer = ring_buffer
        self._buffer = ""
        self._lock = threading.Lock()
        self._read_fd = None
        self._write_fd = None
        self._reader_thread = None
        # Create redirect context manager (not yet entered)
        # _QueueWriter implements write/flush but not full IO[str] interface
        io_target = cast("IO[str]", cast("object", self))
        if is_stderr:
            self._redirect = contextlib.redirect_stderr(io_target)
        else:
            self._redirect = contextlib.redirect_stdout(io_target)

    def _pipe_reader(self) -> None:
        assert self._read_fd is not None  # Guaranteed by _ensure_pipe() before thread starts
        try:
            while True:
                data = os.read(self._read_fd, 8192)
                if not data:
                    break
                self.write(data.decode("utf-8", errors="replace"))
        except OSError:
            pass

    def _ensure_pipe(self) -> None:
        if self._read_fd is not None:
            return
        self._read_fd, self._write_fd = os.pipe()
        self._reader_thread = threading.Thread(
            target=self._pipe_reader, daemon=True, name=f"pipe-reader-{self._stage_name}"
        )
        self._reader_thread.start()

    def __enter__(self) -> _QueueWriter:
        self._redirect.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._redirect.__exit__(exc_type, exc_val, exc_tb)
        if self._write_fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._write_fd)
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=5.0)
        if self._read_fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._read_fd)
        self.flush()

    def _send_line(self, line: str) -> None:
        """Save line locally and send to queue for real-time display."""
        self._ring_buffer.append(line, self._is_stderr)
        # Queue failure only affects real-time display; output is already saved locally
        with contextlib.suppress(queue.Full, ValueError, OSError):
            self._queue.put(
                LogMessage(
                    kind=OutputMessageKind.LOG,
                    stage=self._stage_name,
                    line=line,
                    is_stderr=self._is_stderr,
                ),
                block=False,
            )

    def write(self, s: str) -> int:
        with self._lock:
            self._buffer += s
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                if line:
                    self._send_line(line)
        return len(s)

    def flush(self) -> None:
        with self._lock:
            if self._buffer:
                self._send_line(self._buffer)
                self._buffer = ""

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        self._ensure_pipe()
        if self._write_fd is None:
            raise OSError("QueueWriter pipe not available")
        return self._write_fd


def hash_dependencies(
    deps: dict[str, types.ArtifactRef] | list[str],
    store: store_mod.Store | None = None,
    state_db: state.StateDB | None = None,
) -> tuple[dict[str, HashInfo], list[str], list[str], list[tuple[str, int, int, int, str]]]:
    """Hash all dependency files and directories.

    Returns (hashes, missing_files, unreadable_files).
    For directories, includes full manifest with file hashes/sizes for provenance.
    Paths are normalized (symlinks preserved) for portability in lock files.
    """
    _t = metrics.start()
    hashes = dict[str, HashInfo]()
    missing = list[str]()
    unreadable = list[str]()
    file_hash_entries = list[tuple[str, int, int, int, str]]()
    if isinstance(deps, dict):
        if store is None:
            raise ValueError("store is required for artifact dependency hashing")
        for ref in deps.values():
            key = types.identity_key(ref.identity)
            try:
                hash_info = store.hash_artifact(ref)
                hashes[key] = hash_info
                if not is_dir_hash(hash_info) and isinstance(store, store_mod.WorkspaceStore):
                    path = store.checkout(ref)
                    stat = path.stat()
                    file_hash_entries.append(
                        (
                            str(path),
                            stat.st_mtime_ns,
                            stat.st_size,
                            stat.st_ino,
                            hash_info["hash"],
                        )
                    )
            except FileNotFoundError:
                missing.append(key)
            except OSError:
                unreadable.append(key)
    else:
        for dep in deps:
            normalized_path = project.normalize_path(dep)
            path = pathlib.Path(normalized_path)
            try:
                if path.is_dir():
                    tree_hash, manifest = cache.hash_directory(path, state_db)
                    hashes[dep] = DirHash(hash=tree_hash, manifest=manifest)
                else:
                    file_hash, file_stat = cache.hash_file(path, state_db)
                    hashes[dep] = FileHash(hash=file_hash)
                    file_hash_entries.append(
                        (
                            str(normalized_path),
                            file_stat.st_mtime_ns,
                            file_stat.st_size,
                            file_stat.st_ino,
                            file_hash,
                        )
                    )
            except FileNotFoundError:
                missing.append(dep)
            except OSError:
                unreadable.append(dep)
    metrics.end("worker.hash_dependencies", _t)
    return hashes, missing, unreadable, file_hash_entries


# -----------------------------------------------------------------------------
# Generation tracking for O(1) skip detection
# -----------------------------------------------------------------------------


def can_skip_via_generation(
    stage_name: str,
    fingerprint: dict[str, str],
    deps: dict[str, types.ArtifactRef] | list[str],
    outs_paths: list[str],
    current_params: dict[str, Any],
    lock_data: LockData,
    state_db: state.StateDB,
    verify_files: bool = True,
) -> bool:
    """Check if stage can skip using O(1) generation tracking.

    Generation tracking avoids hashing files by tracking monotonic generation counters.
    Set verify_files=False for status prediction.
    """
    if lock_data["code_manifest"] != fingerprint:
        return False
    if lock_data["params"] != current_params:
        return False

    # Compare output identities (outs_paths is already normalized by _get_normalized_out_paths)
    normalized_outs = sorted(outs_paths)
    locked_out_paths = sorted(
        types.identity_key(k) if isinstance(k, types.ArtifactIdentity) else k
        for k in lock_data["output_hashes"]
    )
    if normalized_outs != locked_out_paths:
        return False

    if not deps:
        return True

    recorded_gens = state_db.get_dep_generations(stage_name)
    if not recorded_gens:
        return False

    dep_keys = (
        [types.identity_key(ref.identity) for ref in deps.values()]
        if isinstance(deps, dict)
        else list(deps)
    )
    current_gens = state_db.get_many_generations(dep_keys)

    for dep_key in dep_keys:
        current_gen = current_gens.get(dep_key)
        if current_gen is None:
            return False
        if current_gen != recorded_gens.get(dep_key):
            return False

    return True


def compute_dep_generation_map(
    deps: dict[str, types.ArtifactRef] | list[str],
    state_db: state.StateDB,
) -> dict[str, int]:
    """Compute dependency identity -> generation map for recording."""
    dep_keys = (
        [types.identity_key(ref.identity) for ref in deps.values()]
        if isinstance(deps, dict)
        else list(deps)
    )
    current_gens = state_db.get_many_generations(dep_keys)

    gen_record = dict[str, int]()
    for dep_key in dep_keys:
        gen = current_gens.get(dep_key)
        if gen is not None:
            gen_record[dep_key] = gen

    return gen_record


def _commit_lock_and_build_deferred(
    stage_info: WorkerStageInfo,
    lock_data: LockData,
    input_hash: str,
    output_hashes: dict[str, HashInfo],
    production_lock: lock.StageLock,
    state_db: state.StateDB,
    *,
    file_hash_entries: list[tuple[str, int, int, int, str]] | None = None,
    increment_outputs: bool = True,
) -> DeferredWrites:
    """Commit lock file and build deferred writes for StateDB.

    Only called in the commit (non --no-commit) path. Writes the production
    lock file and returns DeferredWrites for the coordinator to apply.
    """
    production_lock.write(lock_data)
    return _build_deferred_writes(
        stage_info,
        input_hash,
        output_hashes,
        state_db,
        file_hash_entries=file_hash_entries,
        increment_outputs=increment_outputs,
    )


def _build_deferred_writes(
    stage_info: WorkerStageInfo,
    input_hash: str,
    output_hashes: dict[str, HashInfo],
    state_db: state.StateDB,
    *,
    file_hash_entries: list[tuple[str, int, int, int, str]] | None = None,
    increment_outputs: bool = True,
) -> DeferredWrites:
    """Build deferred writes for coordinator to apply."""
    result: DeferredWrites = {}

    if increment_outputs:
        result["increment_outputs"] = True

    # Dependency generations (read current values)
    gen_record = compute_dep_generation_map(stage_info["deps"], state_db)
    if gen_record:
        result["dep_generations"] = gen_record

    # Run cache entry — only cached outputs belong in run cache
    outs = stage_info["outs"]
    if outs and isinstance(outs[0], outputs.BaseOut):
        expanded_outs = cast("Sequence[outputs.ExpandedOut]", outs)
        cached_paths = {out.path for out in expanded_outs if out.cache}
    else:
        cached_paths = {
            types.identity_key(out.identity)
            for out in outs
            if out.tag is not types.ArtifactTag.METRIC
        }
    output_entries = [
        run_history.output_hash_to_entry(path, oh)
        for path, oh in output_hashes.items()
        if path in cached_paths
    ]
    if output_entries:
        result["run_cache_input_hash"] = input_hash
        result["run_cache_entry"] = run_history.RunCacheEntry(
            run_id=stage_info["run_id"],
            output_hashes=output_entries,
        )

    if file_hash_entries:
        result["file_hash_entries"] = file_hash_entries

    return result


# -----------------------------------------------------------------------------
# Run cache for skip detection (like DVC's run cache)
# -----------------------------------------------------------------------------


class RunCacheSkipResult(TypedDict):
    """Result from successful run cache skip."""

    output_hashes: dict[str, HashInfo]


def _try_skip_via_run_cache_with_store(
    stage_name: str,
    input_hash: str,
    stage_outs: list[types.ArtifactRef],
    store: store_mod.Store,
    state_db: state.StateDB,
) -> RunCacheSkipResult | None:
    """Try to skip using run cache for store-based outputs.

    For store-based stages, outputs are already in the store, so we just need to
    check if they exist and get their hashes.
    """
    entry = state_db.lookup_run_cache(stage_name, input_hash)
    if entry is None:
        return None

    output_hashes: dict[str, HashInfo] = {}
    for ref in stage_outs:
        if not store.exists(ref):
            return None
        identity_key = types.identity_key(ref.identity)
        output_hashes[identity_key] = store.hash_artifact(ref)

    return RunCacheSkipResult(
        output_hashes=output_hashes,
    )


def write_run_cache_entry(
    stage_name: str,
    input_hash: str,
    output_hashes: dict[str, HashInfo],
    run_id: str,
    state_db: state.StateDB,
) -> None:
    """Write run cache entry after successful execution."""
    output_entries = [
        run_history.output_hash_to_entry(path, oh) for path, oh in output_hashes.items()
    ]
    cache_entry = run_history.RunCacheEntry(run_id=run_id, output_hashes=output_entries)
    state_db.write_run_cache(stage_name, input_hash, cache_entry)
