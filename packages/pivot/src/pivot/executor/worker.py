"""Worker process execution for pipeline stages.

Functions that execute in separate processes via ProcessPoolExecutor.
Must be module-level and picklable.
"""

from __future__ import annotations

import codecs
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
from typing import IO, TYPE_CHECKING, Any, Literal, NamedTuple, TypedDict, cast, overload, override

import pydantic

from pivot import (
    config,
    exceptions,
    loaders,
    metrics,
    parameters,
    project,
    run_history,
    skip,
    stage_def,
    types,
)
from pivot.storage import artifact_lock, cache, lock, state
from pivot.storage import store as store_mod
from pivot.types import (
    DeferredWrites,
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
    from collections.abc import Callable, Generator
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
    run_id: str
    force: bool
    no_commit: bool
    params_arg_name: str | None
    project_root: pathlib.Path
    state_dir: pathlib.Path
    collection_params: dict[str, str]
    incremental_inputs: dict[str, types.ArtifactIdentity]


def _make_result(
    status: Literal[StageStatus.RAN, StageStatus.CACHED, StageStatus.FAILED],
    reason: str,
    ring_buffer: _OutputRingBuffer,
    input_hash: str | None = None,
    file_hash_entries: list[tuple[str, int, int, int, str]] | None = None,
) -> StageResult:
    """Build StageResult with collected metrics for cross-process transfer."""
    result = StageResult(
        status=status,
        reason=reason,
        input_hash=input_hash,
        output_lines=ring_buffer.snapshot(),
        metrics=metrics.get_entries(),
    )
    if file_hash_entries:
        result["deferred_writes"] = DeferredWrites(file_hash_entries=file_hash_entries)
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
    state_dir = stage_info["state_dir"]
    project_root = stage_info["project_root"]
    store = store_mod.store_from_spec(stage_info["store_spec"])

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

                with state.StateDB(state_dir, readonly=True) as state_db:
                    outputs_missing_from_cache = False
                    if lock_data is not None and not stage_info["force"]:
                        out_paths = [types.identity_key(out.identity) for out in stage_outs]
                        if can_skip_via_generation(
                            stage_name=stage_name,
                            fingerprint=stage_info["fingerprint"],
                            deps=stage_info["deps"],
                            outs_paths=out_paths,
                            current_params=current_params,
                            lock_data=lock_data,
                            state_db=state_db,
                            store=store,
                        ):
                            input_hash = run_history.stage_input_hash(
                                current_fingerprint,
                                current_params,
                                stage_info["deps"].values(),
                                lock_data["dep_hashes"],
                                stage_outs,
                            )
                            restored_entries = list[tuple[str, int, int, int, str]]()
                            restored = restore_outputs_to_hashes(
                                store,
                                stage_outs,
                                lock_data["output_hashes"],
                                state_db,
                                state_dir=stage_info["state_dir"],
                                file_hash_entries=restored_entries,
                            )
                            if restored:
                                return _make_result(
                                    StageStatus.CACHED,
                                    "unchanged (generation)",
                                    ring_buffer,
                                    input_hash=input_hash,
                                    file_hash_entries=restored_entries,
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

                    skip_reason, run_reason, input_hash, run_cache_allowed = _check_skip_or_run(
                        stage_info,
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
                        restored_entries = list[tuple[str, int, int, int, str]]()
                        restored = restore_outputs_to_hashes(
                            store,
                            stage_outs,
                            lock_data["output_hashes"],
                            state_db,
                            state_dir=stage_info["state_dir"],
                            file_hash_entries=restored_entries,
                        )
                        if restored:
                            return _make_result(
                                StageStatus.CACHED,
                                skip_reason,
                                ring_buffer,
                                input_hash=input_hash,
                                file_hash_entries=restored_entries,
                            )
                        run_reason = "outputs missing from cache"

                    # Check run cache for previously executed configuration (skip if forcing)
                    if run_reason and not stage_info["force"] and run_cache_allowed:
                        run_cache_skip = _try_skip_via_run_cache_with_store(
                            stage_name,
                            input_hash,
                            stage_outs,
                            store,
                            state_db,
                            state_dir=stage_info["state_dir"],
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
                                output_paths=lock_output_paths(stage_outs, store, project_root),
                            )
                            deferred = _commit_lock_and_build_deferred(
                                stage_info,
                                new_lock_data,
                                input_hash,
                                run_cache_skip["output_hashes"],
                                production_lock,
                                state_db,
                                file_hash_entries=file_hash_entries
                                + run_cache_skip["file_hash_entries"],
                                increment_outputs=run_cache_skip["outputs_restored"],
                            )
                            return StageResult(
                                status=StageStatus.CACHED,
                                reason="unchanged (run cache)",
                                input_hash=input_hash,
                                output_lines=ring_buffer.snapshot(),
                                metrics=metrics.get_entries(),
                                deferred_writes=deferred,
                            )

                output_paths = _run_stage_function_with_store(
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
                    stage_info["incremental_inputs"],
                    lock_data,
                    state_dir,
                    project_root,
                )
                if no_commit:
                    output_hashes = _hash_output_paths(output_paths)
                else:
                    output_hashes = _commit_outputs_with_store(output_paths, store)
                    for ref, path in output_paths.items():
                        file_hash_entries.extend(
                            output_file_hash_entries(
                                path, output_hashes[types.identity_key(ref.identity)]
                            )
                        )

                # For --no-commit, skip lock/cache writes entirely
                if no_commit:
                    return _make_result(
                        StageStatus.RAN,
                        run_reason,
                        ring_buffer,
                        input_hash=input_hash,
                    )

                # Build lock data
                new_lock_data = LockData(
                    code_manifest=current_fingerprint,
                    params=current_params,
                    dep_hashes=_to_identity_hashes(dict(sorted(dep_hashes.items()))),
                    output_hashes=_to_identity_hashes(dict(sorted(output_hashes.items()))),
                    output_paths=lock_output_paths(stage_outs, store, project_root),
                )

                # Single StateDB open for post-execution work
                with state.StateDB(state_dir, readonly=True) as state_db:
                    deferred = _commit_lock_and_build_deferred(
                        stage_info,
                        new_lock_data,
                        input_hash,
                        output_hashes,
                        production_lock,
                        state_db,
                        file_hash_entries=file_hash_entries,
                    )
                    return StageResult(
                        status=StageStatus.RAN,
                        reason=run_reason,
                        input_hash=input_hash,
                        output_lines=ring_buffer.snapshot(),
                        metrics=metrics.get_entries(),
                        deferred_writes=deferred,
                    )
        except (exceptions.OutputMissingError, exceptions.CacheRestoreError) as exc:
            return _make_result(StageStatus.FAILED, str(exc), ring_buffer, input_hash=input_hash)
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


def _to_identity_hashes(hashes: dict[str, HashInfo]) -> dict[types.ArtifactIdentity, HashInfo]:
    return {types.identity_from_key(key): info for key, info in hashes.items()}


class OutputRestoreResult(NamedTuple):
    """Whether output restoration succeeded and wrote workspace bytes.

    ``unavailable`` lists outputs that neither match their locked hash nor can be restored
    from the cache. A check (``restore=False``) lists every such output; a restore stops at
    the first, since the stage must run anyway.
    """

    available: bool
    outputs_restored: bool
    unavailable: list[types.ArtifactRef]


@overload
def restore_outputs_to_hashes(
    store: store_mod.Store,
    outs: list[types.ArtifactRef],
    output_hashes: dict[types.ArtifactIdentity, HashInfo],
    state_db: state.StateDB | None,
    *,
    state_dir: pathlib.Path | None = None,
    restore: bool = True,
    report_changes: Literal[False] = False,
    file_hash_entries: list[tuple[str, int, int, int, str]] | None = None,
) -> bool: ...


@overload
def restore_outputs_to_hashes(
    store: store_mod.Store,
    outs: list[types.ArtifactRef],
    output_hashes: dict[types.ArtifactIdentity, HashInfo],
    state_db: state.StateDB | None,
    *,
    state_dir: pathlib.Path | None = None,
    restore: bool = True,
    report_changes: Literal[True],
    file_hash_entries: list[tuple[str, int, int, int, str]] | None = None,
) -> OutputRestoreResult: ...


def restore_outputs_to_hashes(
    store: store_mod.Store,
    outs: list[types.ArtifactRef],
    output_hashes: dict[types.ArtifactIdentity, HashInfo],
    state_db: state.StateDB | None,
    *,
    state_dir: pathlib.Path | None = None,
    restore: bool = True,
    report_changes: bool = False,
    file_hash_entries: list[tuple[str, int, int, int, str]] | None = None,
) -> bool | OutputRestoreResult:
    """Verify outputs match recorded hashes and restore cacheable mismatches.

    ``file_hash_entries``, when given, receives a StateDB file-hash entry for every file
    restored from the cache, so the next check finds it without reading it.
    """
    unavailable = list[types.ArtifactRef]()
    outputs_restored = False
    if not isinstance(store, store_mod.WorkspaceStore):
        unavailable = [ref for ref in outs if not store.exists(ref)]
    else:
        for ref in outs:
            expected_hash = output_hashes.get(ref.identity)
            if expected_hash is not None:
                output_path = store.resolve_display_path(ref)
                try:
                    actual_hash = hash_output(output_path, state_db)
                except OSError:
                    actual_hash = None
                if actual_hash == expected_hash:
                    continue
                if store.cache_contains(ref, expected_hash):
                    if not restore:
                        continue
                    if store.restore_from_cache(ref, expected_hash, state_dir=state_dir):
                        outputs_restored = True
                        if file_hash_entries is not None:
                            file_hash_entries.extend(
                                output_file_hash_entries(output_path, expected_hash)
                            )
                        continue
            unavailable.append(ref)
            if restore:
                break
    if report_changes:
        return OutputRestoreResult(
            available=not unavailable,
            outputs_restored=outputs_restored,
            unavailable=unavailable,
        )
    return not unavailable


def _check_skip_or_run(
    stage_info: WorkerStageInfo,
    lock_data: LockData | None,
    current_fingerprint: dict[str, str],
    current_params: dict[str, Any],
    dep_hashes: dict[str, HashInfo],
) -> tuple[str | None, str, str, bool]:
    """Determine if stage can skip or must run."""
    out_identities = [out.identity for out in stage_info["outs"]]
    dep_hashes_by_identity = _to_identity_hashes(dep_hashes)
    input_hash = run_history.stage_input_hash(
        current_fingerprint,
        current_params,
        stage_info["deps"].values(),
        dep_hashes_by_identity,
        stage_info["outs"],
    )
    decision = skip.check_stage(
        lock_data=lock_data,
        fingerprint=current_fingerprint,
        params=current_params,
        dep_hashes=dep_hashes_by_identity,
        out_paths=out_identities,
        run_cache_allowed=not bool(stage_info["incremental_inputs"]),
    )
    if decision["changed"]:
        return None, decision["reason"], input_hash, decision["run_cache_allowed"]
    return "unchanged", "", input_hash, decision["run_cache_allowed"]


def hash_output(path: pathlib.Path, state_db: state.StateDB | None = None) -> HashInfo:
    """Compute output hash without saving to cache."""
    if path.is_dir():
        tree_hash, manifest = cache.hash_directory(path, state_db)
        return DirHash(hash=tree_hash, manifest=manifest)
    file_hash, _ = cache.hash_file(path, state_db)
    return FileHash(hash=file_hash)


def output_file_hash_entries(
    path: pathlib.Path, output_hash: HashInfo
) -> list[tuple[str, int, int, int, str]]:
    """StateDB file-hash entries for an output whose hash is already known.

    Stats each file (each manifest entry, for a directory) and pairs it with its known hash;
    nothing is read. Call it once the output has its final inode: a hardlink checkout
    replaces the file the stage wrote. An output whose file-hash StateDB key exceeds LMDB's
    limit is omitted: this cache only avoids a later hash and must not fail the stage.
    """
    if is_dir_hash(output_hash):
        files = [(path / entry["relpath"], entry["hash"]) for entry in output_hash["manifest"]]
    else:
        files = [(path, output_hash["hash"])]
    entries = list[tuple[str, int, int, int, str]]()
    for file_path, file_hash in files:
        if not state.file_hash_key_fits(file_path):
            logger.debug("Skipping oversized output file-hash cache entry: %s", file_path)
            continue
        fs_stat = file_path.stat()
        entries.append(
            (str(file_path), fs_stat.st_mtime_ns, fs_stat.st_size, fs_stat.st_ino, file_hash)
        )
    return entries


def lock_output_paths(
    outs: list[types.ArtifactRef],
    store: store_mod.Store,
    project_root: pathlib.Path,
) -> dict[types.ArtifactIdentity, str]:
    """Project-relative, POSIX output paths for a production workspace lock."""
    if not isinstance(store, store_mod.WorkspaceStore):
        return {}
    root = project_root.resolve()
    return {
        ref.identity: store.resolve_display_path(ref).relative_to(root).as_posix() for ref in outs
    }


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
        parallel_config = None
    if parallel_config is None:
        # Called outside the except block so a stage error is not chained onto the ImportError.
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


def _load_dep_from_path(ref: types.ArtifactRef, path: pathlib.Path) -> Any:
    """Load an artifact after the caller has resolved its workspace/cache path."""
    loader = cast("Any", ref.format)
    if ref.tag is types.ArtifactTag.DIRECTORY:
        values = dict[str, Any]()
        for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
            rel_key = file_path.relative_to(path).as_posix()
            values[rel_key] = loader.load(file_path)
        return values
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
    # Cast needed: isinstance narrows Any to Mapping[Unknown, Unknown]; output keys are strings
    result_map = cast("collections_abc.Mapping[str, Any]", result)

    value_map = dict[types.ArtifactRef, Any]()
    missing = list[str]()
    for out in outs:
        key = out.identity.key
        if key is None:
            raise RuntimeError(f"Stage '{stage_name}' has multiple outputs but missing output key")
        if key not in result_map:
            missing.append(key)
            continue
        value_map[out] = result_map[key]

    if missing:
        raise KeyError(
            f"Missing return output keys: {sorted(missing)}. Return value keys: {sorted(result_map.keys())}"
        )

    extra = set(result_map.keys()) - {
        out.identity.key for out in outs if out.identity.key is not None
    }
    if extra:
        logger.warning("Extra keys in return value not declared as outputs: %s", sorted(extra))

    return value_map


def _paths_overlap(first: pathlib.Path, second: pathlib.Path) -> bool:
    """Return whether either path contains the other."""
    return first.is_relative_to(second) or second.is_relative_to(first)


def _reject_path_inside_project(
    ref: types.ArtifactRef,
    value: object,
    project_root: pathlib.Path,
    dependency_paths: collections_abc.Collection[pathlib.Path] = (),
) -> None:
    """A PathOnly output is moved to the output path; never move a file pivot does not own."""
    if not isinstance(value, pathlib.Path):
        raise TypeError(
            f"Stage '{ref.identity.producer}' returned {type(value).__name__}; "
            + "a PathOnly output must return pathlib.Path"
        )
    source = value if value.is_absolute() else project_root / value
    lexical_root = project_root.absolute()
    lexical_source = source.absolute()
    resolved_root = project_root.resolve()
    resolved_source = source.resolve()
    if lexical_source.is_relative_to(lexical_root) or resolved_source.is_relative_to(resolved_root):
        raise exceptions.StageDefinitionError(
            f"Stage '{ref.identity.producer}' returned {value}, which is inside the project "
            + f"({project_root}). Pivot moves a returned path to the stage's output path, so it "
            + "must be a file or directory the stage created outside the project, e.g. under "
            + "tempfile.mkdtemp()."
        )
    for dependency_path in dependency_paths:
        lexical_dependency = dependency_path.absolute()
        resolved_dependency = dependency_path.resolve()
        if any(
            _paths_overlap(source_path, dependency_variant)
            for source_path in (lexical_source, resolved_source)
            for dependency_variant in (lexical_dependency, resolved_dependency)
        ):
            raise exceptions.StageDefinitionError(
                f"Stage '{ref.identity.producer}' returned {value}, which overlaps dependency "
                + f"{dependency_path}. Pivot cannot move a path the stage received as input."
            )


def _write_output_with_store(
    ref: types.ArtifactRef,
    value: Any,
    store: store_mod.Store,
    project_root: pathlib.Path,
    dependency_paths: collections_abc.Collection[pathlib.Path] = (),
) -> pathlib.Path:
    if isinstance(ref.format, loaders.PathOnly) and ref.tag is not types.ArtifactTag.DIRECTORY:
        _reject_path_inside_project(ref, value, project_root, dependency_paths)
    output_path = store.prepare_output(ref)
    writer = cast("Any", ref.format)
    if ref.tag is types.ArtifactTag.DIRECTORY:
        if not isinstance(value, dict):
            raise RuntimeError(
                f"Directory output for '{ref.identity.producer}' expects dict, got {type(value).__name__}"
            )
        if not value:
            raise ValueError(f"Directory output '{ref.identity.producer}': dict must be non-empty")
        # Cast needed: isinstance narrows Any to dict[Unknown, Unknown]; keys are checked below
        value_dict = cast("dict[object, Any]", value)
        members = list[tuple[pathlib.Path, Any]]()
        for key, item_value in value_dict.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"Directory output for '{ref.identity.producer}': keys must be strings, got {type(key).__name__}"
                )
            normalized = stage_def.validate_directory_out_key(key, ref.identity.producer)
            if isinstance(ref.format, loaders.PathOnly):
                _reject_path_inside_project(
                    ref,
                    cast("object", item_value),
                    project_root,
                    dependency_paths,
                )
            members.append((output_path / normalized, item_value))
        for full_path, item_value in members:
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


def _prepare_incremental_inputs(
    stage_name: str,
    outs: list[types.ArtifactRef],
    incremental_inputs: dict[str, types.ArtifactIdentity],
    lock_data: LockData | None,
    store: store_mod.Store,
    state_dir: pathlib.Path,
) -> dict[str, Any]:
    """Restore previous incremental outputs and load them for the stage call."""
    if not incremental_inputs:
        return {}
    if not isinstance(store, store_mod.WorkspaceStore):
        raise RuntimeError("Incremental outputs require a workspace store.")

    outs_by_identity = {out.identity: out for out in outs}
    values = dict[str, Any]()
    for param_name, identity in incremental_inputs.items():
        ref = outs_by_identity[identity]
        loader = cast("loaders.Loader[Any, Any]", ref.format)
        if lock_data is None:
            values[param_name] = loader.empty()
            continue

        output_hash = lock_data["output_hashes"].get(identity)
        stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(state_dir))
        if output_hash is None:
            raise exceptions.CacheRestoreError(
                f"Cache missing for incremental output '{types.identity_key(identity)}'. "
                + f"Run 'pivot pull' to restore it, or remove '{stage_lock.path}' and rerun "
                + "to reset this incremental output."
            )
        output_path = store.resolve_display_path(ref)
        cache.remove_output(output_path)
        if not store.restore_from_cache(
            ref,
            output_hash,
            state_dir=state_dir,
            checkout_modes=[config.CheckoutMode.COPY],
        ):
            raise exceptions.CacheRestoreError(
                f"Cache missing for incremental output '{types.identity_key(identity)}'. "
                + f"Run 'pivot pull' to restore it, or remove '{stage_lock.path}' and rerun "
                + "to reset this incremental output."
            )
        values[param_name] = _load_dep_from_path(ref, output_path)
    return values


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
    incremental_inputs: dict[str, types.ArtifactIdentity],
    lock_data: LockData | None,
    state_dir: pathlib.Path,
    project_root: pathlib.Path,
) -> dict[types.ArtifactRef, pathlib.Path]:
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

        dependency_paths = list[pathlib.Path]()
        for name, ref in deps.items():
            dependency_path = store.checkout(ref)
            dependency_paths.append(dependency_path)
            kwargs[name] = _load_dep_from_path(ref, dependency_path)
        kwargs.update(
            _prepare_incremental_inputs(
                stage_name,
                outs,
                incremental_inputs,
                lock_data,
                store,
                state_dir,
            )
        )

        kwargs = _reconstruct_list_kwargs(kwargs, collection_params)

        _set_deterministic_seeds()

        result = _execute_with_joblib_protection(func, kwargs)

        output_values = _resolve_output_values(result, outs, stage_name)
        output_paths = dict[types.ArtifactRef, pathlib.Path]()
        for ref, value in output_values.items():
            output_paths[ref] = _write_output_with_store(
                ref,
                value,
                store,
                project_root,
                dependency_paths,
            )

        return output_paths


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
        output_hashes[types.identity_key(ref.identity)] = store.commit(ref, path)
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
        # One incremental decoder per pipe: a multi-byte character split across two reads
        # is held until its remaining bytes arrive instead of becoming U+FFFD.
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            while True:
                data = os.read(self._read_fd, 8192)
                if not data:
                    break
                self.write(decoder.decode(data))
        except OSError:
            pass
        # Flush a truncated trailing character so it is reported rather than dropped.
        self.write(decoder.decode(b"", final=True))

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
    """Hash all dependencies and collect file-hash cache write-backs."""
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
                hash_info = store.hash_artifact(ref, state_db)
                hashes[key] = hash_info
                if not is_dir_hash(hash_info) and isinstance(store, store_mod.WorkspaceStore):
                    path = store.checkout(ref)
                    fs_stat = path.stat()
                    file_hash_entries.append(
                        (
                            str(path),
                            fs_stat.st_mtime_ns,
                            fs_stat.st_size,
                            fs_stat.st_ino,
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
                    file_hash, fs_stat = cache.hash_file(path, state_db)
                    hashes[dep] = FileHash(hash=file_hash)
                    file_hash_entries.append(
                        (
                            str(normalized_path),
                            fs_stat.st_mtime_ns,
                            fs_stat.st_size,
                            fs_stat.st_ino,
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
    deps: dict[str, types.ArtifactRef],
    outs_paths: list[str],
    current_params: dict[str, Any],
    lock_data: LockData,
    state_db: state.StateDB,
    store: store_mod.Store,
) -> bool:
    """Check generation counters and cached dependency hashes without rehashing."""
    if lock_data["code_manifest"] != fingerprint:
        return False
    if not skip.params_equal(lock_data["params"], current_params):
        return False

    normalized_outs = sorted(outs_paths)
    locked_out_paths = sorted(
        types.identity_key(identity) for identity in lock_data["output_hashes"]
    )
    if normalized_outs != locked_out_paths:
        return False
    if not deps:
        return True

    recorded_gens = state_db.get_dep_generations(stage_name)
    if not recorded_gens:
        return False
    dep_keys = [types.identity_key(ref.identity) for ref in deps.values()]
    current_gens = state_db.get_many_generations(dep_keys)
    stats = list[tuple[pathlib.Path, os.stat_result]]()
    paths_by_key = dict[str, pathlib.Path]()
    for dep_key, ref in zip(dep_keys, deps.values(), strict=True):
        expected_hash = lock_data["dep_hashes"].get(ref.identity)
        if expected_hash is None or is_dir_hash(expected_hash):
            return False
        current_gen = current_gens.get(dep_key)
        if current_gen is None or current_gen != recorded_gens.get(dep_key):
            return False
        try:
            path = store.checkout(ref)
            stats.append((path, path.stat()))
            paths_by_key[dep_key] = path
        except OSError:
            return False
    cached_hashes = state_db.get_many(stats)
    for dep_key, ref in zip(dep_keys, deps.values(), strict=True):
        cached_hash = cached_hashes[paths_by_key[dep_key]]
        if cached_hash != lock_data["dep_hashes"][ref.identity]["hash"]:
            return False
    return True


def compute_dep_generation_map(
    deps: dict[str, types.ArtifactRef],
    state_db: state.StateDB,
) -> dict[str, int]:
    """Compute dependency identity -> generation map for recording."""
    dep_keys = [types.identity_key(ref.identity) for ref in deps.values()]
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

    # Incremental outputs continue from the latest lock state and never use the run cache.
    outs = stage_info["outs"]
    cached_paths = (
        set[str]()
        if any(out.tag is types.ArtifactTag.INCREMENTAL for out in outs)
        else {
            types.identity_key(out.identity)
            for out in outs
            if out.tag is not types.ArtifactTag.METRIC
        }
    )
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
    outputs_restored: bool
    file_hash_entries: list[tuple[str, int, int, int, str]]


def _try_skip_via_run_cache_with_store(
    stage_name: str,
    input_hash: str,
    stage_outs: list[types.ArtifactRef],
    store: store_mod.Store,
    state_db: state.StateDB,
    *,
    state_dir: pathlib.Path,
) -> RunCacheSkipResult | None:
    """Restore all cached outputs from a matching run-cache entry."""
    if any(out.tag is types.ArtifactTag.METRIC for out in stage_outs):
        return None
    entry = state_db.lookup_run_cache(stage_name, input_hash)
    if entry is None:
        return None
    output_hashes = {
        output["path"]: run_history.entry_to_output_hash(output)
        for output in entry["output_hashes"]
    }
    expected_keys = {types.identity_key(ref.identity) for ref in stage_outs}
    if output_hashes.keys() != expected_keys:
        return None
    identity_hashes = _to_identity_hashes(output_hashes)

    restored_entries = list[tuple[str, int, int, int, str]]()
    restoration = restore_outputs_to_hashes(
        store,
        stage_outs,
        identity_hashes,
        state_db,
        state_dir=state_dir,
        report_changes=True,
        file_hash_entries=restored_entries,
    )
    if not restoration.available:
        return None
    return RunCacheSkipResult(
        output_hashes=output_hashes,
        outputs_restored=restoration.outputs_restored,
        file_hash_entries=restored_entries,
    )
