"""Worker process execution for pipeline stages.

Functions that execute in separate processes via ProcessPoolExecutor.
Must be module-level and picklable.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import pathlib
import queue
import random
import sys
import threading
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast, override

import pydantic

from pivot import exceptions, metrics, outputs, parameters, project, run_history, stage_def
from pivot.storage import cache, lock, state
from pivot.types import (
    DeferredWrites,
    DepEntry,
    DirHash,
    FileHash,
    HashInfo,
    LockData,
    OutputHash,
    OutputMessage,
    StageResult,
    StageStatus,
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
                self._queue.put((self._stage_name, msg, True), block=False)
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
    deps: list[str]
    outs: list[outputs.BaseOut]
    signature: Signature | None
    params: stage_def.StageParams | None
    variant: str | None
    overrides: parameters.ParamsOverrides
    checkout_modes: list[cache.CheckoutMode]
    run_id: str
    force: bool
    no_commit: bool
    no_cache: bool
    dep_specs: dict[str, stage_def.FuncDepSpec]
    out_specs: dict[str, outputs.Out[Any]]
    params_arg_name: str | None
    project_root: pathlib.Path
    state_dir: pathlib.Path


def _make_result(
    status: Literal[StageStatus.RAN, StageStatus.SKIPPED, StageStatus.FAILED],
    reason: str,
    output_lines: list[tuple[str, bool]],
) -> StageResult:
    """Build StageResult with collected metrics for cross-process transfer."""
    return StageResult(
        status=status,
        reason=reason,
        output_lines=output_lines,
        metrics=metrics.get_entries(),
    )


def execute_stage(
    stage_name: str,
    stage_info: WorkerStageInfo,
    cache_dir: pathlib.Path,
    output_queue: Queue[OutputMessage],
) -> StageResult:
    """Worker function executed in separate process. Must be module-level for pickling.

    Flag interactions:
    - --force: Always run stage, even if skip detection says it's unchanged
    - --no-cache: Skip caching entirely; lock file records null output hashes
    - --force + --no-cache: Maximum speed mode - no cache reads/writes, no provenance
      tracking beyond the lock file. Lock file will have null hashes for all outputs.
    """
    # Clear metrics at start - each stage collects its own metrics
    metrics.clear()
    output_lines: list[tuple[str, bool]] = []
    files_cache_dir = cache_dir / "files"
    state_db_path = stage_info["state_dir"] / "state.db"
    project_root = stage_info["project_root"]

    # Ensure worker has correct cwd for this stage (workers in reusable pool
    # may have stale cwd from previous execution in different project).
    # _queue_logging captures log messages to the output queue (for TUI display).
    with contextlib.chdir(project_root), _queue_logging(stage_name, output_queue):
        no_commit = stage_info["no_commit"]
        no_cache = stage_info["no_cache"]

        # Validate --no-cache incompatibility with IncrementalOut
        if no_cache and any(isinstance(out, outputs.IncrementalOut) for out in stage_info["outs"]):
            return StageResult(
                status=StageStatus.FAILED,
                reason="--no-cache is incompatible with IncrementalOut (requires cache for incremental updates)",
                output_lines=[],
            )

        checkout_modes = stage_info["checkout_modes"]

        # Production lock for skip detection, pending lock for --no-commit mode
        production_lock = lock.StageLock(stage_name, lock.get_stages_dir(stage_info["state_dir"]))
        pending_lock = lock.get_pending_lock(stage_name, project_root)
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
                [],
            )

        try:
            with lock.execution_lock(stage_name, cache_dir):
                # Check pending lock first for IncrementalOut restoration, fall back to production
                pending_lock_data = pending_lock.read()
                production_lock_data = production_lock.read()
                lock_data = pending_lock_data or production_lock_data

                with state.StateDB(state_db_path, readonly=True) as state_db:
                    dep_hashes, missing, unreadable = hash_dependencies(
                        stage_info["deps"], state_db
                    )

                    if missing:
                        return _make_result(
                            StageStatus.FAILED, f"missing deps: {', '.join(missing)}", []
                        )

                    if unreadable:
                        return _make_result(
                            StageStatus.FAILED, f"unreadable deps: {', '.join(unreadable)}", []
                        )

                    skip_reason, run_reason, input_hash = _check_skip_or_run(
                        stage_name,
                        stage_info,
                        production_lock,
                        lock_data,
                        state_db,
                        current_fingerprint,
                        current_params,
                        dep_hashes,
                    )

                    # Override skip decision if force flag is set
                    if stage_info["force"] and skip_reason is not None:
                        skip_reason = None
                        run_reason = "forced"

                    if skip_reason is not None and lock_data is not None:
                        restored = _restore_outputs_from_cache(
                            stage_outs, lock_data, files_cache_dir, checkout_modes
                        )
                        if restored:
                            return _make_result(StageStatus.SKIPPED, skip_reason, [])
                        run_reason = "outputs missing from cache"

                    # Check run cache for previously executed configuration (skip if forcing or no_cache)
                    if run_reason and not stage_info["force"] and not no_cache:
                        run_cache_skip = _try_skip_via_run_cache(
                            stage_name,
                            input_hash,
                            stage_outs,
                            files_cache_dir,
                            checkout_modes,
                            state_db,
                        )
                        if run_cache_skip is not None:
                            new_lock_data = LockData(
                                code_manifest=current_fingerprint,
                                params=current_params,
                                dep_hashes=dict(sorted(dep_hashes.items())),
                                output_hashes=dict(sorted(run_cache_skip["output_hashes"].items())),
                                dep_generations={},
                            )
                            deferred = _commit_lock_and_build_deferred(
                                stage_info,
                                new_lock_data,
                                input_hash,
                                run_cache_skip["output_hashes"],
                                pending_lock,
                                production_lock,
                                state_db,
                                no_commit,
                            )
                            if deferred is not None:
                                return StageResult(
                                    status=StageStatus.SKIPPED,
                                    reason="unchanged (run cache)",
                                    output_lines=[],
                                    metrics=metrics.get_entries(),
                                    deferred_writes=deferred,
                                )
                            return run_cache_skip["result"]

                _prepare_outputs_for_execution(stage_outs, lock_data, files_cache_dir)

                _run_stage_function_with_injection(
                    stage_info["func"],
                    stage_name,
                    output_queue,
                    output_lines,
                    params_instance,
                    stage_info["dep_specs"],
                    project_root,
                    stage_info["out_specs"],
                    stage_info["params_arg_name"],
                )

                # Compute output hashes (null for no_cache, actual otherwise)
                if no_cache:
                    _verify_outputs_exist(stage_outs)
                    output_hashes = dict[str, OutputHash](
                        {str(out.path): None for out in stage_outs}
                    )
                else:
                    output_hashes = _save_outputs_to_cache(
                        stage_outs, files_cache_dir, checkout_modes
                    )

                # Build lock data (dep_generations filled below if no_commit)
                new_lock_data = LockData(
                    code_manifest=current_fingerprint,
                    params=current_params,
                    dep_hashes=dict(sorted(dep_hashes.items())),
                    output_hashes=dict(sorted(output_hashes.items())),
                    dep_generations={},
                )

                # Single StateDB open for post-execution work
                with state.StateDB(state_db_path, readonly=True) as state_db:
                    deferred = _commit_lock_and_build_deferred(
                        stage_info,
                        new_lock_data,
                        input_hash,
                        output_hashes,
                        pending_lock,
                        production_lock,
                        state_db,
                        no_commit,
                    )
                    if deferred is not None:
                        return StageResult(
                            status=StageStatus.RAN,
                            reason=run_reason,
                            output_lines=output_lines,
                            metrics=metrics.get_entries(),
                            deferred_writes=deferred,
                        )

            return _make_result(StageStatus.RAN, run_reason, output_lines)

        except exceptions.StageAlreadyRunningError as e:
            return _make_result(StageStatus.FAILED, str(e), output_lines)
        except exceptions.OutputMissingError as e:
            return _make_result(StageStatus.FAILED, str(e), output_lines)
        except SystemExit as e:
            return _make_result(
                StageStatus.FAILED, f"Stage called sys.exit({e.code})", output_lines
            )
        except KeyboardInterrupt:
            return _make_result(StageStatus.FAILED, "KeyboardInterrupt", output_lines)
        except Exception as e:
            return _make_result(StageStatus.FAILED, str(e), output_lines)


def _get_normalized_out_paths(stage_info: WorkerStageInfo) -> list[str]:
    """Get normalized output paths from stage info, matching lock_data format."""
    return [str(project.normalize_path(str(out.path))) for out in stage_info["outs"]]


def _get_output_specs(stage_info: WorkerStageInfo) -> list[tuple[str, bool]]:
    """Get normalized output specs (path, cache flag) for input hash computation."""
    return [(str(project.normalize_path(str(out.path))), out.cache) for out in stage_info["outs"]]


def _check_skip_or_run(
    stage_name: str,
    stage_info: WorkerStageInfo,
    stage_lock: lock.StageLock,
    lock_data: LockData | None,
    state_db: state.StateDB,
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
    deps_list = [DepEntry(path=path, hash=info["hash"]) for path, info in dep_hashes.items()]
    input_hash = run_history.compute_input_hash(
        current_fingerprint, current_params, deps_list, out_specs
    )

    if lock_data is None:
        return None, "No previous run", input_hash

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
        return "unchanged (generation)", "", input_hash

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


def _restore_outputs_from_cache(
    stage_outs: list[outputs.BaseOut],
    lock_data: LockData,
    files_cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
) -> bool:
    """Restore missing outputs from cache for a single stage's skip detection.

    Returns True if all outputs exist or were restored. On failure, cleans up
    any partially restored outputs to leave the filesystem in a clean state,
    allowing the stage to re-run with a clean workspace.
    """
    output_hashes = lock_data["output_hashes"]
    restored_paths = list[pathlib.Path]()

    for out in stage_outs:
        path = pathlib.Path(cast("str", out.path))
        if path.exists():
            continue

        output_hash = output_hashes.get(str(out.path))
        if output_hash is None:
            _cleanup_restored_paths(restored_paths)
            return False

        try:
            restored = cache.restore_from_cache(
                path, output_hash, files_cache_dir, checkout_modes=checkout_modes
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


def _prepare_outputs_for_execution(
    stage_outs: Sequence[outputs.BaseOut],
    lock_data: LockData | None,
    files_cache_dir: pathlib.Path,
) -> None:
    """Prepare outputs before stage execution - delete or restore for incremental."""
    output_hashes = lock_data["output_hashes"] if lock_data else {}

    for out in stage_outs:
        path = pathlib.Path(cast("str", out.path))

        if isinstance(out, outputs.IncrementalOut):
            # IncrementalOut: restore from cache as writable copy
            cache.remove_output(path)  # Clear any stale state first
            out_hash = output_hashes.get(str(out.path))
            if out_hash:
                # COPY mode makes file writable (not symlink to read-only cache)
                restored = cache.restore_from_cache(
                    path, out_hash, files_cache_dir, cache.CheckoutMode.COPY
                )
                if not restored:
                    raise exceptions.CacheRestoreError(
                        f"Failed to restore IncrementalOut '{out.path}' from cache"
                    )
        else:
            # Regular output: delete before run
            cache.remove_output(path)


def _save_outputs_to_cache(
    stage_outs: list[outputs.BaseOut],
    files_cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
) -> dict[str, OutputHash]:
    """Save outputs to cache after successful execution."""
    with metrics.timed("worker.save_outputs_to_cache"):
        output_hashes = dict[str, OutputHash]()

        for out in stage_outs:
            path = pathlib.Path(cast("str", out.path))
            if not path.exists():
                raise exceptions.OutputMissingError(f"Stage did not produce output: {out.path}")

            if out.cache:
                output_hashes[str(out.path)] = cache.save_to_cache(
                    path, files_cache_dir, checkout_modes=checkout_modes
                )
            else:
                output_hashes[str(out.path)] = None

        return output_hashes


def _verify_outputs_exist(stage_outs: list[outputs.BaseOut]) -> None:
    """Verify all outputs exist without caching (for --no-cache mode).

    Note: Only checks existence, not contents. Empty directories pass verification.
    This is intentional - --no-cache mode trusts that the stage produced valid output.
    """
    for out in stage_outs:
        path = pathlib.Path(cast("str", out.path))
        if not path.exists():
            raise exceptions.OutputMissingError(f"Stage did not produce output: {out.path}")


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


def _run_stage_function_with_injection(
    func: Callable[..., Any],
    stage_name: str,
    output_queue: Queue[OutputMessage],
    output_lines: list[tuple[str, bool]],
    params: stage_def.StageParams | None = None,
    dep_specs: dict[str, stage_def.FuncDepSpec] | None = None,
    project_root: pathlib.Path | None = None,
    out_specs: dict[str, outputs.Out[Any]] | None = None,
    params_arg_name: str | None = None,
) -> None:
    """Run stage function with dependency injection and output capture.

    This is the new injection-based execution path for stages using Annotated deps:

        def train(
            config: TrainParams,
            data: Annotated[DataFrame, Dep("input.csv", CSV())],
        ) -> TrainOutputs:
            ...

    The function:
    1. Loads deps from disk based on dep_specs
    2. Builds kwargs dict (params + loaded deps)
    3. Calls the function with kwargs
    4. Saves outputs based on out_specs (resolved at registration time)

    Args:
        out_specs: Output specs resolved at registration time (return key -> Out).
            For single-output stages, uses "_single" key convention.
        params_arg_name: Name of the StageParams parameter (pre-computed at registration).
    """
    with (
        _QueueWriter(stage_name, output_queue, is_stderr=False, output_lines=output_lines),
        _QueueWriter(stage_name, output_queue, is_stderr=True, output_lines=output_lines),
    ):
        kwargs = dict[str, Any]()

        # Add params if provided (using pre-computed arg name from registration)
        if params is not None:
            if params_arg_name is None:
                raise RuntimeError(
                    f"Stage '{stage_name}' has params but params_arg_name is None - this indicates a bug in registration"
                )
            kwargs[params_arg_name] = params

        # Load and inject deps
        root = project_root if project_root is not None else project.get_project_root()
        if dep_specs:
            loaded_deps = stage_def.load_deps_from_specs(dep_specs, root)
            kwargs.update(loaded_deps)

        _set_deterministic_seeds()

        # Execute function with joblib threading protection
        result = _execute_with_joblib_protection(func, kwargs)

        # Save outputs using pre-resolved specs from registration
        if out_specs:
            if result is None:
                raise RuntimeError(f"Stage '{stage_name}' has output annotations but returned None")
            # For single-output stages, out_specs uses SINGLE_OUTPUT_KEY convention
            if stage_def.SINGLE_OUTPUT_KEY in out_specs:
                result = {stage_def.SINGLE_OUTPUT_KEY: result}
            stage_def.save_return_outputs(result, out_specs, root)
        elif result is not None:
            logger.warning(
                "Stage '%s' returned value but has no Out annotation - discarding", stage_name
            )


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
    _output_lines: list[tuple[str, bool]]
    _buffer: str
    _redirect: contextlib.AbstractContextManager[object]
    _lock: threading.Lock

    def __init__(
        self,
        stage_name: str,
        output_queue: Queue[OutputMessage],
        *,
        is_stderr: bool,
        output_lines: list[tuple[str, bool]],
    ) -> None:
        self._stage_name = stage_name
        self._queue = output_queue
        self._is_stderr = is_stderr
        self._output_lines = output_lines
        self._buffer = ""
        self._lock = threading.Lock()
        # Create redirect context manager (not yet entered)
        # _QueueWriter implements write/flush but not full IO[str] interface
        if is_stderr:
            self._redirect = contextlib.redirect_stderr(self)
        else:
            self._redirect = contextlib.redirect_stdout(self)

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
        self.flush()

    def _send_line(self, line: str) -> None:
        """Save line locally and send to queue for real-time display."""
        self._output_lines.append((line, self._is_stderr))
        # Queue failure only affects real-time display; output is already saved locally
        with contextlib.suppress(queue.Full, ValueError, OSError):
            self._queue.put((self._stage_name, line, self._is_stderr), block=False)

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
        """Raise UnsupportedOperation - _QueueWriter has no underlying file descriptor."""
        raise io.UnsupportedOperation("_QueueWriter does not use a file descriptor")


def hash_dependencies(
    deps: list[str], state_db: state.StateDB | None = None
) -> tuple[dict[str, HashInfo], list[str], list[str]]:
    """Hash all dependency files and directories.

    Returns (hashes, missing_files, unreadable_files).
    For directories, includes full manifest with file hashes/sizes for provenance.
    Paths are normalized (symlinks preserved) for portability in lock files.
    """
    with metrics.timed("worker.hash_dependencies"):
        hashes = dict[str, HashInfo]()
        missing = list[str]()
        unreadable = list[str]()
        for dep in deps:
            normalized = str(project.normalize_path(dep))
            path = pathlib.Path(dep)
            try:
                if path.is_dir():
                    tree_hash, manifest = cache.hash_directory(path, state_db)
                    hashes[normalized] = {"hash": tree_hash, "manifest": manifest}
                else:
                    hashes[normalized] = {"hash": cache.hash_file(path, state_db)}
            except FileNotFoundError:
                missing.append(dep)
            except OSError:
                unreadable.append(dep)
        return hashes, missing, unreadable


# -----------------------------------------------------------------------------
# Generation tracking for O(1) skip detection
# -----------------------------------------------------------------------------


def can_skip_via_generation(
    stage_name: str,
    fingerprint: dict[str, str],
    deps: list[str],
    outs_paths: list[str],
    current_params: dict[str, Any],
    lock_data: LockData,
    state_db: state.StateDB,
    verify_files: bool = True,
) -> bool:
    """Check if stage can skip using O(1) generation tracking.

    Generation tracking avoids hashing files by tracking monotonic generation counters.
    Set verify_files=False for status prediction. Falls back to lock_data for --no-commit mode.
    """
    if lock_data["code_manifest"] != fingerprint:
        return False
    if lock_data["params"] != current_params:
        return False

    # Normalize output paths to match lock_data format
    normalized_outs = sorted(str(project.normalize_path(p)) for p in outs_paths)
    locked_out_paths = sorted(lock_data["output_hashes"].keys())
    if normalized_outs != locked_out_paths:
        return False

    # Empty deps is a valid case - skip generation check if no deps
    if not deps:
        return True

    # Try StateDB first, fall back to lock_data (for --no-commit mode)
    recorded_gens = state_db.get_dep_generations(stage_name)
    if recorded_gens is None:
        recorded_gens = lock_data["dep_generations"]
    if not recorded_gens:
        return False

    dep_paths = [pathlib.Path(d) for d in deps]
    current_gens = state_db.get_many_generations(dep_paths)

    # Gather file stats for metadata verification (catches external modifications)
    cached_hashes: dict[pathlib.Path, str | None] | None = None
    if verify_files:
        dep_stats = list[tuple[pathlib.Path, os.stat_result]]()
        for dep in deps:
            path = pathlib.Path(dep)
            try:
                dep_stats.append((path, path.stat()))
            except OSError:
                return False

        # Batch check: verify metadata matches cached values
        cached_hashes = state_db.get_many(dep_stats)

    for dep in deps:
        path = pathlib.Path(dep)
        normalized = str(project.normalize_path(dep))

        # Check generation
        current_gen = current_gens.get(path)
        if current_gen is None:
            return False
        if current_gen != recorded_gens.get(normalized):
            return False

        # Check metadata - if None, file was externally modified or not cached
        if cached_hashes is not None and cached_hashes.get(path) is None:
            return False

    return True


def compute_dep_generation_map(
    deps: list[str],
    state_db: state.StateDB,
) -> dict[str, int]:
    """Compute dependency path -> generation map for recording."""
    dep_paths = [pathlib.Path(d) for d in deps]
    current_gens = state_db.get_many_generations(dep_paths)

    gen_record = dict[str, int]()
    for dep in deps:
        path = pathlib.Path(dep)
        gen = current_gens.get(path)
        if gen is not None:
            normalized = str(project.normalize_path(dep))
            gen_record[normalized] = gen

    return gen_record


def _commit_lock_and_build_deferred(
    stage_info: WorkerStageInfo,
    lock_data: LockData,
    input_hash: str,
    output_hashes: dict[str, OutputHash],
    pending_lock: lock.StageLock,
    production_lock: lock.StageLock,
    state_db: state.StateDB,
    no_commit: bool,
) -> DeferredWrites | None:
    """Commit lock file and build deferred writes.

    For no_commit: computes dep_generations, writes to pending_lock, returns None.
    For commit: writes to production_lock, returns DeferredWrites for StateDB.
    """
    if no_commit:
        dep_gens = compute_dep_generation_map(stage_info["deps"], state_db)
        lock_data["dep_generations"] = dep_gens
        pending_lock.write(lock_data)
        return None
    production_lock.write(lock_data)
    return _build_deferred_writes(stage_info, input_hash, output_hashes, state_db)


def _build_deferred_writes(
    stage_info: WorkerStageInfo,
    input_hash: str,
    output_hashes: dict[str, OutputHash],
    state_db: state.StateDB,
) -> DeferredWrites:
    """Build deferred writes for coordinator to apply."""
    result: DeferredWrites = {}

    # Dependency generations (read current values)
    gen_record = compute_dep_generation_map(stage_info["deps"], state_db)
    if gen_record:
        result["dep_generations"] = gen_record

    # Run cache entry (only if there are cached outputs)
    output_entries = [
        entry
        for path, oh in output_hashes.items()
        if (entry := run_history.output_hash_to_entry(path, oh)) is not None
    ]
    if output_entries:
        result["run_cache_input_hash"] = input_hash
        result["run_cache_entry"] = run_history.RunCacheEntry(
            run_id=stage_info["run_id"],
            output_hashes=output_entries,
        )

    return result


# -----------------------------------------------------------------------------
# Run cache for skip detection (like DVC's run cache)
# -----------------------------------------------------------------------------


class RunCacheSkipResult(TypedDict):
    """Result from successful run cache skip."""

    result: StageResult
    output_hashes: dict[str, OutputHash]


def _try_skip_via_run_cache(
    stage_name: str,
    input_hash: str,
    stage_outs: list[outputs.BaseOut],
    files_cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
    state_db: state.StateDB,
) -> RunCacheSkipResult | None:
    """Try to skip using run cache. Returns result and output hashes if skipped, None if must run.

    On failure, cleans up any partially restored outputs to leave a clean state,
    allowing the stage to re-run with a clean workspace.
    """
    # IncrementalOut stages build on previous outputs - run cache doesn't apply
    if any(isinstance(out, outputs.IncrementalOut) for out in stage_outs):
        return None

    entry = state_db.lookup_run_cache(stage_name, input_hash)
    if entry is None:
        return None

    # Build output hash map preserving manifest for directories
    output_hash_map: dict[str, FileHash | DirHash] = {
        oh["path"]: run_history.entry_to_output_hash(oh) for oh in entry["output_hashes"]
    }

    # Validate cached outputs match expected: run cache entry should only contain
    # outputs that have cache=True. The input hash includes cache flags, so a mismatch
    # here indicates corruption or a bug.
    expected_cached_paths = {str(out.path) for out in stage_outs if out.cache}
    if set(output_hash_map.keys()) != expected_cached_paths:
        return None

    restored_paths = list[pathlib.Path]()

    for out in stage_outs:
        path = pathlib.Path(cast("str", out.path))
        if path.exists():
            continue

        cached_output = output_hash_map.get(str(out.path))
        if cached_output is None:
            # Output doesn't exist on disk and isn't in cache - must re-run
            if restored_paths:
                _cleanup_restored_paths(restored_paths)
            return None

        try:
            restored = cache.restore_from_cache(
                path, cached_output, files_cache_dir, checkout_modes=checkout_modes
            )
        except OSError:
            if restored_paths:
                _cleanup_restored_paths(restored_paths)
            return None

        if restored:
            restored_paths.append(path)
        else:
            if restored_paths:
                _cleanup_restored_paths(restored_paths)
            return None

    # Build output_hashes for lock file update, including non-cached outputs
    output_hashes: dict[str, OutputHash] = {}
    for out in stage_outs:
        out_path = str(out.path)
        if out.cache:
            output_hashes[out_path] = output_hash_map[out_path]
        else:
            output_hashes[out_path] = None

    return RunCacheSkipResult(
        result=_make_result(StageStatus.SKIPPED, "unchanged (run cache)", []),
        output_hashes=output_hashes,
    )


def write_run_cache_entry(
    stage_name: str,
    input_hash: str,
    output_hashes: dict[str, OutputHash],
    run_id: str,
    state_db: state.StateDB,
) -> None:
    """Write run cache entry after successful execution."""
    output_entries = [
        entry
        for path, oh in output_hashes.items()
        if (entry := run_history.output_hash_to_entry(path, oh)) is not None
    ]
    cache_entry = run_history.RunCacheEntry(run_id=run_id, output_hashes=output_entries)
    state_db.write_run_cache(stage_name, input_hash, cache_entry)
