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
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

import pydantic

from pivot import exceptions, metrics, outputs, parameters, project, run_history, stage_def
from pivot.storage import cache, lock, state
from pivot.types import (
    DeferredWrites,
    DepEntry,
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
    params: stage_def.StageParams | None
    variant: str | None
    overrides: parameters.ParamsOverrides
    cwd: pathlib.Path | None
    checkout_modes: list[str]
    run_id: str
    force: bool
    no_commit: bool
    no_cache: bool
    dep_specs: dict[str, stage_def.FuncDepSpec]
    out_path_overrides: dict[str, outputs.PathType] | None


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
    state_db_path = cache_dir.parent / "state.db"
    # cache_dir is .pivot/cache, project_root is two levels up
    project_root = cache_dir.parent.parent
    no_commit = stage_info["no_commit"]
    no_cache = stage_info["no_cache"]

    # Validate --no-cache incompatibility with IncrementalOut
    if no_cache and any(isinstance(out, outputs.IncrementalOut) for out in stage_info["outs"]):
        return StageResult(
            status=StageStatus.FAILED,
            reason="--no-cache is incompatible with IncrementalOut (requires cache for incremental updates)",
            output_lines=[],
        )

    # Convert string checkout modes to enum (strings required for pickling across processes)
    checkout_modes = [cache.CheckoutMode(m) for m in stage_info["checkout_modes"]]

    # Production lock for skip detection, pending lock for --no-commit mode
    production_lock = lock.StageLock(stage_name, lock.get_stages_dir(cache_dir))
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
                dep_hashes, missing, unreadable = hash_dependencies(stage_info["deps"], state_db)

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
                    run_cache_result = _try_skip_via_run_cache(
                        stage_name,
                        input_hash,
                        stage_outs,
                        files_cache_dir,
                        checkout_modes,
                        state_db,
                    )
                    if run_cache_result is not None:
                        return run_cache_result

            _prepare_outputs_for_execution(stage_outs, lock_data, files_cache_dir)

            with _working_directory(stage_info["cwd"]):
                _run_stage_function_with_injection(
                    stage_info["func"],
                    stage_name,
                    output_queue,
                    output_lines,
                    params_instance,
                    stage_info["dep_specs"],
                    project_root,
                    stage_info["out_path_overrides"],
                )

            # Compute output hashes (null for no_cache, actual otherwise)
            if no_cache:
                _verify_outputs_exist(stage_outs)
                output_hashes = dict[str, OutputHash]({str(out.path): None for out in stage_outs})
            else:
                output_hashes = _save_outputs_to_cache(stage_outs, files_cache_dir, checkout_modes)

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
                if no_commit:
                    dep_gens = compute_dep_generation_map(stage_info["deps"], state_db)
                    new_lock_data["dep_generations"] = dep_gens
                    pending_lock.write(new_lock_data)
                else:
                    production_lock.write(new_lock_data)
                    deferred = _build_deferred_writes(
                        stage_info, input_hash, output_hashes, state_db
                    )
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
        return _make_result(StageStatus.FAILED, f"Stage called sys.exit({e.code})", output_lines)
    except KeyboardInterrupt:
        return _make_result(StageStatus.FAILED, "KeyboardInterrupt", output_lines)
    except Exception as e:
        return _make_result(StageStatus.FAILED, str(e), output_lines)


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
    # Registry always stores single-file outputs (multi-file are expanded)
    out_paths = [str(out.path) for out in stage_info["outs"]]
    deps_list = [DepEntry(path=path, hash=info["hash"]) for path, info in dep_hashes.items()]
    input_hash = run_history.compute_input_hash(
        current_fingerprint, current_params, deps_list, out_paths
    )

    if lock_data is None:
        return None, "No previous run", input_hash

    if _can_skip_via_generation(stage_name, stage_info, lock_data, state_db, current_params):
        return "unchanged (generation)", "", input_hash

    changed, run_reason = stage_lock.is_changed_with_lock_data(
        lock_data, current_fingerprint, current_params, dep_hashes
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


def _run_stage_function_with_injection(
    func: Callable[..., Any],
    stage_name: str,
    output_queue: Queue[OutputMessage],
    output_lines: list[tuple[str, bool]],
    params: stage_def.StageParams | None = None,
    dep_specs: dict[str, stage_def.FuncDepSpec] | None = None,
    project_root: pathlib.Path | None = None,
    return_out_path_overrides: dict[str, outputs.PathType] | None = None,
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
    4. Saves outputs based on return type annotations
    """
    with (
        _QueueWriter(stage_name, output_queue, is_stderr=False, output_lines=output_lines),
        _QueueWriter(stage_name, output_queue, is_stderr=True, output_lines=output_lines),
    ):
        kwargs = dict[str, Any]()

        # Add params if provided (find the param arg name from signature)
        if params is not None:
            params_arg_name, _ = stage_def.find_params_in_signature(func)
            if params_arg_name is not None:
                kwargs[params_arg_name] = params

        # Load and inject deps
        root = project_root if project_root is not None else project.get_project_root()
        if dep_specs:
            loaded_deps = stage_def.load_deps_from_specs(dep_specs, root)
            kwargs.update(loaded_deps)

        # Execute function
        result = func(**kwargs)

        # Save outputs based on return type
        return_out_specs = stage_def.get_output_specs_from_return(func)
        single_out_spec = stage_def.get_single_output_spec_from_return(func)

        if return_out_specs and result is not None:
            # TypedDict return with multiple outputs
            stage_def.save_return_outputs(result, return_out_specs, root, return_out_path_overrides)
        elif single_out_spec is not None and result is not None:
            # Single annotated return type
            stage_def.save_return_outputs(
                {"_single": result},
                {"_single": single_out_spec},
                root,
                return_out_path_overrides,
            )
        elif result is not None:
            logger.debug(
                "Stage '%s' returned value but has no Out annotation - discarding", stage_name
            )


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
    _redirect: contextlib.AbstractContextManager[object]

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


def _try_skip_via_run_cache(
    stage_name: str,
    input_hash: str,
    stage_outs: list[outputs.BaseOut],
    files_cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
    state_db: state.StateDB,
) -> StageResult | None:
    """Try to skip using run cache. Returns StageResult if skipped, None if must run.

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
    output_hash_map = {
        oh["path"]: run_history.entry_to_output_hash(oh) for oh in entry["output_hashes"]
    }

    restored_paths = list[pathlib.Path]()

    for out in stage_outs:
        path = pathlib.Path(cast("str", out.path))
        if path.exists():
            continue

        cached_output = output_hash_map.get(str(out.path))
        if cached_output is None:
            if out.cache:
                _cleanup_restored_paths(restored_paths)
                return None
            continue

        try:
            restored = cache.restore_from_cache(
                path, cached_output, files_cache_dir, checkout_modes=checkout_modes
            )
        except OSError:
            _cleanup_restored_paths(restored_paths)
            return None

        if restored:
            restored_paths.append(path)
        else:
            _cleanup_restored_paths(restored_paths)
            return None

    return _make_result(StageStatus.SKIPPED, "unchanged (run cache)", [])


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
