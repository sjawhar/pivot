"""Worker process execution for pipeline stages.

Functions that execute in separate processes via ProcessPoolExecutor.
Must be module-level and picklable.
"""

from __future__ import annotations

import contextlib
import io
import logging
import pathlib
import queue
from typing import TYPE_CHECKING, Any, TypedDict

import pydantic

from pivot import exceptions, outputs, parameters, project
from pivot.storage import cache, lock, state
from pivot.types import (
    HashInfo,
    LockData,
    OutputHash,
    OutputMessage,
    StageResult,
    StageStatus,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from inspect import Signature
    from multiprocessing import Queue
    from types import TracebackType

    from pydantic import BaseModel

logger = logging.getLogger(__name__)


def _working_directory(cwd: pathlib.Path | None) -> contextlib.AbstractContextManager[None]:
    """Return context manager to temporarily change working directory."""
    if cwd is None:
        return contextlib.nullcontext()
    return contextlib.chdir(cwd)


class WorkerStageInfo(TypedDict):
    """Stage info subset passed to worker processes."""

    func: Callable[..., Any]
    fingerprint: dict[str, str]
    deps: list[str]
    outs: list[outputs.BaseOut]
    signature: Signature | None
    params: BaseModel | None
    variant: str | None
    overrides: parameters.ParamsOverrides
    cwd: pathlib.Path | None
    checkout_modes: list[str]


def execute_stage(
    stage_name: str,
    stage_info: WorkerStageInfo,
    cache_dir: pathlib.Path,
    output_queue: Queue[OutputMessage],
) -> StageResult:
    """Worker function executed in separate process. Must be module-level for pickling."""
    output_lines: list[tuple[str, bool]] = []
    files_cache_dir = cache_dir / "files"
    state_db_path = cache_dir.parent / "state.db"

    # Convert string checkout modes to enum (strings required for pickling across processes)
    checkout_modes = [cache.CheckoutMode(m) for m in stage_info["checkout_modes"]]

    stage_lock = lock.StageLock(stage_name, cache_dir)
    current_fingerprint = stage_info["fingerprint"]
    stage_outs = stage_info["outs"]

    params_instance = stage_info["params"]
    overrides = stage_info["overrides"]
    try:
        current_params = parameters.get_effective_params(params_instance, stage_name, overrides)
        if params_instance is not None:
            params_instance = parameters.apply_overrides(params_instance, stage_name, overrides)
    except pydantic.ValidationError as e:
        return StageResult(
            status=StageStatus.FAILED,
            reason=f"Invalid params override in params.yaml: {e.error_count()} validation error(s)",
            output_lines=[],
        )

    try:
        with lock.execution_lock(stage_name, cache_dir):
            lock_data = stage_lock.read()

            with state.StateDB(state_db_path) as state_db:
                dep_hashes, missing, unreadable = hash_dependencies(stage_info["deps"], state_db)

                if missing:
                    return StageResult(
                        status=StageStatus.FAILED,
                        reason=f"missing deps: {', '.join(missing)}",
                        output_lines=[],
                    )

                if unreadable:
                    return StageResult(
                        status=StageStatus.FAILED,
                        reason=f"unreadable deps: {', '.join(unreadable)}",
                        output_lines=[],
                    )

                skip_reason, run_reason = _check_skip_or_run(
                    stage_name,
                    stage_info,
                    stage_lock,
                    lock_data,
                    state_db,
                    current_fingerprint,
                    current_params,
                    dep_hashes,
                )

            if skip_reason is not None and lock_data is not None:
                restored = _restore_outputs_from_cache(
                    stage_outs, lock_data, files_cache_dir, checkout_modes
                )
                if restored:
                    return StageResult(
                        status=StageStatus.SKIPPED, reason=skip_reason, output_lines=[]
                    )
                run_reason = "outputs missing from cache"

            _prepare_outputs_for_execution(stage_outs, lock_data, files_cache_dir)

            with _working_directory(stage_info["cwd"]):
                _run_stage_function_with_capture(
                    stage_info["func"], stage_name, output_queue, output_lines, params_instance
                )

            output_hashes = _save_outputs_to_cache(stage_outs, files_cache_dir, checkout_modes)

            new_lock_data = LockData(
                code_manifest=current_fingerprint,
                params=current_params,
                dep_hashes=dict(sorted(dep_hashes.items())),
                output_hashes=dict(sorted(output_hashes.items())),
            )
            stage_lock.write(new_lock_data)

            with state.StateDB(state_db_path) as state_db:
                _record_generations_after_run(stage_name, stage_info, state_db)

        return StageResult(status=StageStatus.RAN, reason=run_reason, output_lines=output_lines)

    except exceptions.StageAlreadyRunningError as e:
        return StageResult(status=StageStatus.FAILED, reason=str(e), output_lines=output_lines)
    except exceptions.OutputMissingError as e:
        return StageResult(status=StageStatus.FAILED, reason=str(e), output_lines=output_lines)
    except SystemExit as e:
        return StageResult(
            status=StageStatus.FAILED,
            reason=f"Stage called sys.exit({e.code})",
            output_lines=output_lines,
        )
    except KeyboardInterrupt:
        return StageResult(
            status=StageStatus.FAILED, reason="KeyboardInterrupt", output_lines=output_lines
        )
    except Exception as e:
        return StageResult(status=StageStatus.FAILED, reason=str(e), output_lines=output_lines)


def _check_skip_or_run(
    stage_name: str,
    stage_info: WorkerStageInfo,
    stage_lock: lock.StageLock,
    lock_data: LockData | None,
    state_db: state.StateDB,
    current_fingerprint: dict[str, str],
    current_params: dict[str, Any],
    dep_hashes: dict[str, HashInfo],
) -> tuple[str | None, str]:
    """Determine if stage can skip or must run.

    Returns (skip_reason, run_reason) where exactly one is meaningful:
    - If skip_reason is not None: stage can skip, run_reason is empty
    - If skip_reason is None: stage must run, run_reason explains why
    """
    if lock_data is None:
        return None, "No previous run"

    if _can_skip_via_generation(stage_name, stage_info, lock_data, state_db, current_params):
        return "unchanged (generation)", ""

    changed, run_reason = stage_lock.is_changed_with_lock_data(
        lock_data, current_fingerprint, current_params, dep_hashes
    )
    if not changed:
        return "unchanged", ""

    return None, run_reason


def _restore_outputs_from_cache(
    stage_outs: list[outputs.BaseOut],
    lock_data: LockData,
    files_cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
) -> bool:
    """Restore missing outputs from cache. Returns True if all restored successfully."""
    output_hashes = lock_data["output_hashes"]
    for out in stage_outs:
        path = pathlib.Path(out.path)
        if path.exists():
            continue

        output_hash = output_hashes.get(out.path)
        if output_hash is None:
            if out.cache:
                return False
            continue

        if not cache.restore_from_cache(
            path, output_hash, files_cache_dir, checkout_modes=checkout_modes
        ):
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
        path = pathlib.Path(out.path)

        if isinstance(out, outputs.IncrementalOut):
            # IncrementalOut: restore from cache as writable copy
            cache.remove_output(path)  # Clear any stale state first
            out_hash = output_hashes.get(out.path)
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
    output_hashes = dict[str, OutputHash]()

    for out in stage_outs:
        path = pathlib.Path(out.path)
        if not path.exists():
            raise exceptions.OutputMissingError(f"Stage did not produce output: {out.path}")

        if out.cache:
            output_hashes[out.path] = cache.save_to_cache(
                path, files_cache_dir, checkout_modes=checkout_modes
            )
        else:
            output_hashes[out.path] = None

    return output_hashes


def _run_stage_function_with_capture(
    func: Callable[..., Any],
    stage_name: str,
    output_queue: Queue[OutputMessage],
    output_lines: list[tuple[str, bool]],
    params_instance: BaseModel | None = None,
) -> None:
    """Run stage function with stdout/stderr capture, streaming to queue.

    Output is appended to the provided output_lines list, ensuring captured
    output is preserved even if func() raises an exception.
    """
    with (
        _QueueWriter(stage_name, output_queue, is_stderr=False, output_lines=output_lines),
        _QueueWriter(stage_name, output_queue, is_stderr=True, output_lines=output_lines),
    ):
        kwargs = dict[str, Any]()
        if params_instance is not None:
            kwargs["params"] = params_instance
        func(**kwargs)


class _QueueWriter:
    """Context manager for capturing stdout/stderr to a queue.

    Handles stream redirection, output capture, and automatic flushing.
    Implements minimal file-like interface needed by print() and common libraries.
    """

    _stage_name: str
    _queue: Queue[OutputMessage]
    _is_stderr: bool
    _output_lines: list[tuple[str, bool]]
    _buffer: str
    _redirect: contextlib.AbstractContextManager[Any]

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
        self._buffer += s
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line:
                self._send_line(line)
        return len(s)

    def flush(self) -> None:
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


def _can_skip_via_generation(
    stage_name: str,
    stage_info: WorkerStageInfo,
    lock_data: LockData,
    state_db: state.StateDB,
    current_params: dict[str, Any],
) -> bool:
    """Check if stage can skip using O(1) generation tracking."""
    if lock_data["code_manifest"] != stage_info["fingerprint"]:
        return False
    if lock_data["params"] != current_params:
        return False

    recorded_gens = state_db.get_dep_generations(stage_name)
    if recorded_gens is None:
        return False

    dep_paths = [pathlib.Path(d) for d in stage_info["deps"]]
    current_gens = state_db.get_many_generations(dep_paths)

    for dep in stage_info["deps"]:
        path = pathlib.Path(dep)
        normalized = str(project.normalize_path(dep))
        current_gen = current_gens.get(path)
        if current_gen is None:
            return False
        if current_gen != recorded_gens.get(normalized):
            return False

    return True


def _record_generations_after_run(
    stage_name: str,
    stage_info: WorkerStageInfo,
    state_db: state.StateDB,
) -> None:
    """Record dependency generations and increment output generations after successful run."""
    dep_paths = [pathlib.Path(d) for d in stage_info["deps"]]
    current_gens = state_db.get_many_generations(dep_paths)

    gen_record = dict[str, int]()
    for dep in stage_info["deps"]:
        path = pathlib.Path(dep)
        gen = current_gens.get(path)
        if gen is not None:
            normalized = str(project.normalize_path(dep))
            gen_record[normalized] = gen

    if gen_record:
        state_db.record_dep_generations(stage_name, gen_record)

    for out in stage_info["outs"]:
        out_path = pathlib.Path(out.path)
        state_db.increment_generation(out_path)
