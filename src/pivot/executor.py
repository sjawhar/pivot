"""Pipeline executor - runs stages with greedy parallel execution.

Uses ProcessPoolExecutor for true parallelism (separate GIL per process):
- Greedy scheduling: start stages as soon as dependencies complete
- FIRST_COMPLETED waiting: don't wait for entire batches
- forkserver context: warm imports without fork's dangers
- Queue-based output streaming: real-time stdout/stderr from workers
"""

from __future__ import annotations

import collections
import concurrent.futures
import contextlib
import dataclasses
import hashlib
import io
import logging
import multiprocessing as mp
import os
import pathlib
import queue
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Literal, TextIO, TypedDict, cast, override

import loky

from pivot import cache, console, dag, exceptions, lock, outputs, project, registry
from pivot.types import (
    LockData,
    OnError,
    OutputHash,
    OutputMessage,
    StageDisplayStatus,
    StageResult,
    StageStatus,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence
    from inspect import Signature

    from networkx import DiGraph

logger = logging.getLogger(__name__)

_MAX_WORKERS_DEFAULT = 8


class ExecutionSummary(TypedDict):
    """Summary result for a single stage after execution (returned by executor.run)."""

    status: Literal["ran", "skipped", "failed", "unknown"]
    reason: str


class WorkerStageInfo(TypedDict):
    """Stage info subset passed to worker processes."""

    func: Callable[..., Any]
    fingerprint: dict[str, str]
    deps: list[str]
    outs: list[outputs.BaseOut]
    signature: Signature | None


@dataclasses.dataclass
class StageState:
    """Tracks execution state for a single stage."""

    name: str
    info: registry.RegistryStageInfo
    upstream: list[str]
    upstream_unfinished: set[str]
    downstream: list[str]
    mutex: list[str]
    status: StageStatus = StageStatus.READY
    result: StageResult | None = None
    start_time: float | None = None
    end_time: float | None = None


def run(
    stages: list[str] | None = None,
    single_stage: bool = False,
    cache_dir: pathlib.Path | None = None,
    parallel: bool = True,
    max_workers: int | None = None,
    on_error: OnError | str = OnError.FAIL,
    show_output: bool = True,
    force: bool = False,
    stage_timeout: float | None = None,
) -> dict[str, ExecutionSummary]:
    """Execute pipeline stages with greedy parallel execution.

    Args:
        stages: Target stages to run (and their dependencies). If None, runs all.
        single_stage: If True, run only the specified stages without dependencies.
        cache_dir: Directory for lock files. Defaults to .pivot/cache.
        parallel: If True, run independent stages in parallel (default: True).
        max_workers: Max concurrent stages (default: min(cpu_count, 8)).
        on_error: Error handling mode - "fail", "keep_going", or "ignore".
        show_output: If True, print progress and stage output to console.
        force: If True, skip safety checks for uncached IncrementalOut files.
        stage_timeout: Max seconds for each stage to complete (default: no timeout).

    Returns:
        Dict of stage_name -> {status: "ran"|"skipped"|"failed", reason: str}
    """
    if cache_dir is None:
        cache_dir = project.get_project_root() / ".pivot" / "cache"

    if isinstance(on_error, OnError):
        error_mode = on_error
    else:
        try:
            error_mode = OnError(on_error)
        except ValueError:
            raise ValueError(
                f"Invalid on_error mode: {on_error}. Use 'fail', 'keep_going', or 'ignore'"
            ) from None

    graph = registry.REGISTRY.build_dag(validate=True)

    if stages:
        registered = set(graph.nodes())
        unknown = [s for s in stages if s not in registered]
        if unknown:
            raise exceptions.StageNotFoundError(f"Unknown stage(s): {', '.join(unknown)}")

    execution_order = dag.get_execution_order(graph, stages, single_stage=single_stage)

    if not execution_order:
        return {}

    # Check for uncached IncrementalOut files that would be lost
    if not force:
        uncached = _check_uncached_incremental_outputs(execution_order, cache_dir)
        if uncached:
            files_list = "\n".join(f"  - {stage}: {path}" for stage, path in uncached)
            raise exceptions.UncachedIncrementalOutputError(
                f"The following IncrementalOut files exist but are not in cache:\n{files_list}\n\n"
                + "Running the pipeline will DELETE these files and they cannot be restored.\n"
                + "To proceed anyway, use force=True or back up these files first."
            )

    con = console.get_console() if show_output else None
    total_stages = len(execution_order)

    stage_states = _initialize_stage_states(execution_order, graph)

    if not parallel:
        max_workers = 1
    elif max_workers is None:
        max_workers = min(os.cpu_count() or 1, _MAX_WORKERS_DEFAULT, len(execution_order))
    max_workers = max(1, min(max_workers, len(execution_order)))

    start_time = time.perf_counter()

    _execute_greedy(
        stage_states=stage_states,
        cache_dir=cache_dir,
        max_workers=max_workers,
        error_mode=error_mode,
        con=con,
        total_stages=total_stages,
        stage_timeout=stage_timeout,
    )

    results = _build_results(stage_states)

    if con:
        status_counts = collections.Counter(r["status"] for r in results.values())
        total_duration = time.perf_counter() - start_time
        con.summary(
            status_counts["ran"], status_counts["skipped"], status_counts["failed"], total_duration
        )

    return results


def _initialize_stage_states(
    execution_order: list[str],
    graph: DiGraph[str],
) -> dict[str, StageState]:
    """Initialize state tracking for all stages."""
    stages_set = set(execution_order)
    states = dict[str, StageState]()

    for stage_name in execution_order:
        stage_info = registry.REGISTRY.get(stage_name)

        upstream = list(graph.successors(stage_name))
        upstream_in_plan = [u for u in upstream if u in stages_set]

        downstream = list(graph.predecessors(stage_name))
        downstream_in_plan = [d for d in downstream if d in stages_set]

        states[stage_name] = StageState(
            name=stage_name,
            info=stage_info,
            upstream=upstream_in_plan,
            upstream_unfinished=set(upstream_in_plan),
            downstream=downstream_in_plan,
            mutex=stage_info.get("mutex", []),
        )

    return states


def _warn_single_stage_mutex_groups(stage_states: dict[str, StageState]) -> None:
    """Warn if any mutex group contains only one stage (likely a typo)."""
    groups: collections.defaultdict[str, list[str]] = collections.defaultdict(list)
    for name, state in stage_states.items():
        for mutex in state.mutex:
            groups[mutex].append(name)

    for group, members in groups.items():
        if len(members) == 1:
            logger.warning(f"Mutex group '{group}' only contains stage '{members[0]}'")


def _create_executor(max_workers: int) -> concurrent.futures.Executor:
    """Get reusable loky executor - workers persist across calls for efficiency."""
    return cast("concurrent.futures.Executor", loky.get_reusable_executor(max_workers=max_workers))


def _execute_greedy(
    stage_states: dict[str, StageState],
    cache_dir: pathlib.Path,
    max_workers: int,
    error_mode: OnError,
    con: console.Console | None,
    total_stages: int,
    stage_timeout: float | None = None,
) -> None:
    """Execute stages with greedy parallel scheduling using loky ProcessPoolExecutor."""
    completed_count = 0
    futures: dict[concurrent.futures.Future[StageResult], str] = {}
    mutex_counts: collections.defaultdict[str, int] = collections.defaultdict(int)

    _warn_single_stage_mutex_groups(stage_states)

    executor = _create_executor(max_workers)
    output_queue: mp.Queue[OutputMessage] = mp.Manager().Queue()  # pyright: ignore[reportAssignmentType]

    output_thread: threading.Thread | None = None
    if con:
        output_thread = threading.Thread(
            target=_output_queue_reader,
            args=(output_queue, con),
            daemon=True,
        )
        output_thread.start()

    try:
        with executor:
            _start_ready_stages(
                stage_states=stage_states,
                executor=executor,
                futures=futures,
                cache_dir=cache_dir,
                output_queue=output_queue,
                max_stages=max_workers,
                mutex_counts=mutex_counts,
                error_mode=error_mode,
                con=con,
                total_stages=total_stages,
                completed_count=completed_count,
            )

            while futures:
                # Calculate wait timeout based on oldest running stage
                wait_timeout: float | None = None
                if stage_timeout is not None:
                    now = time.perf_counter()
                    for _future, stage_name in futures.items():
                        state = stage_states[stage_name]
                        if state.start_time:
                            elapsed = now - state.start_time
                            remaining = stage_timeout - elapsed
                            if wait_timeout is None or remaining < wait_timeout:
                                wait_timeout = max(0.1, remaining)  # At least 0.1s

                done, _ = concurrent.futures.wait(
                    futures.keys(),
                    timeout=wait_timeout,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

                # Check for timed-out stages if nothing completed
                if not done and stage_timeout is not None:
                    now = time.perf_counter()
                    timed_out = list[tuple[concurrent.futures.Future[StageResult], str]]()
                    for future, stage_name in futures.items():
                        state = stage_states[stage_name]
                        if state.start_time and (now - state.start_time) >= stage_timeout:
                            timed_out.append((future, stage_name))
                    for future, stage_name in timed_out:
                        futures.pop(future)
                        future.cancel()
                        state = stage_states[stage_name]
                        _mark_stage_failed(
                            state,
                            f"Stage timed out after {stage_timeout}s",
                            stage_states,
                            error_mode,
                        )
                        completed_count += 1
                        for mutex in state.mutex:
                            mutex_counts[mutex] -= 1
                    continue

                for future in done:
                    stage_name = futures.pop(future)
                    state = stage_states[stage_name]

                    try:
                        result = future.result()
                        state.result = result
                        state.end_time = time.perf_counter()

                        if result["status"] == "failed":
                            state.status = StageStatus.FAILED
                            _handle_stage_failure(stage_name, stage_states, error_mode)
                        elif result["status"] == "skipped":
                            state.status = StageStatus.SKIPPED
                        else:
                            state.status = StageStatus.COMPLETED

                    except concurrent.futures.BrokenExecutor as e:
                        _mark_stage_failed(state, f"Worker died: {e}", stage_states, error_mode)
                        logger.error(f"Worker process died while running '{stage_name}'")

                    except Exception as e:
                        _mark_stage_failed(state, str(e), stage_states, error_mode)

                    completed_count += 1

                    for downstream_name in state.downstream:
                        downstream_state = stage_states.get(downstream_name)
                        if downstream_state:
                            downstream_state.upstream_unfinished.discard(stage_name)

                    # Release mutex locks (regardless of success/failure)
                    for mutex in state.mutex:
                        mutex_counts[mutex] -= 1
                        if mutex_counts[mutex] < 0:
                            logger.error(
                                f"Mutex '{mutex}' released when not held (bug in mutex tracking)"
                            )
                            mutex_counts[mutex] = 0  # Reset to valid state

                    if con and state.result:
                        duration = (
                            (state.end_time - state.start_time)
                            if (state.start_time is not None and state.end_time is not None)
                            else None
                        )
                        con.stage_result(
                            name=stage_name,
                            index=completed_count,
                            total=total_stages,
                            status=StageStatus(state.result["status"]),
                            reason=state.result["reason"],
                            duration=duration,
                        )

                if error_mode == OnError.FAIL:
                    failed = any(s.status == StageStatus.FAILED for s in stage_states.values())
                    if failed:
                        for f in futures:
                            f.cancel()
                        return

                slots_available = max_workers - len(futures)
                if slots_available > 0:
                    _start_ready_stages(
                        stage_states=stage_states,
                        executor=executor,
                        futures=futures,
                        cache_dir=cache_dir,
                        output_queue=output_queue,
                        max_stages=slots_available,
                        mutex_counts=mutex_counts,
                        error_mode=error_mode,
                        con=con,
                        total_stages=total_stages,
                        completed_count=completed_count,
                    )
    finally:
        # Signal output thread to stop - may fail if queue is broken
        with contextlib.suppress(OSError, ValueError):
            output_queue.put(None)
        if output_thread:
            output_thread.join(timeout=1.0)


def _output_queue_reader(output_q: mp.Queue[OutputMessage], con: console.Console) -> None:
    """Read output messages from worker processes and display them."""
    while True:
        try:
            msg = output_q.get(timeout=0.1)
            if msg is None:
                break
            stage_name, line, is_stderr = msg
            con.stage_output(stage_name, line, is_stderr)
        except queue.Empty:
            continue


def _has_failed_upstream(state: StageState, stage_states: dict[str, StageState]) -> bool:
    """Check if any upstream stage has failed."""
    return any(
        stage_states[u].status == StageStatus.FAILED for u in state.upstream if u in stage_states
    )


def _start_ready_stages(
    stage_states: dict[str, StageState],
    executor: concurrent.futures.Executor,
    futures: dict[concurrent.futures.Future[StageResult], str],
    cache_dir: pathlib.Path,
    output_queue: mp.Queue[OutputMessage],
    max_stages: int,
    mutex_counts: collections.defaultdict[str, int],
    error_mode: OnError,
    con: console.Console | None,
    total_stages: int,
    completed_count: int,
) -> None:
    """Find and start stages that are ready to execute."""
    started = 0

    for stage_name, state in stage_states.items():
        if started >= max_stages:
            break

        if state.status != StageStatus.READY:
            continue

        if state.upstream_unfinished:
            continue

        # Skip stages with failed upstream (unless in ignore mode)
        if error_mode != OnError.IGNORE and _has_failed_upstream(state, stage_states):
            continue

        # Check mutex availability - skip if any mutex group is held
        if any(mutex_counts[m] > 0 for m in state.mutex):
            continue

        # Acquire mutex locks before changing status
        for mutex in state.mutex:
            mutex_counts[mutex] += 1

        worker_info = _prepare_worker_info(state.info)

        try:
            future = executor.submit(
                execute_stage_worker,
                stage_name,
                worker_info,
                cache_dir,
                output_queue,
            )
            futures[future] = stage_name
            started += 1

            # Only mark as in-progress after successful submission
            state.status = StageStatus.IN_PROGRESS
            state.start_time = time.perf_counter()

            if con:
                con.stage_start(
                    name=stage_name,
                    index=completed_count + len(futures),
                    total=total_stages,
                    status=StageDisplayStatus.RUNNING,
                )
        except Exception as e:
            # Rollback mutex acquisition on submission failure
            for mutex in state.mutex:
                mutex_counts[mutex] -= 1
            _mark_stage_failed(state, f"Failed to submit: {e}", stage_states, error_mode)


def _prepare_worker_info(stage_info: registry.RegistryStageInfo) -> WorkerStageInfo:
    """Prepare stage info for pickling to worker process."""
    return WorkerStageInfo(
        func=stage_info["func"],
        fingerprint=stage_info["fingerprint"],
        deps=stage_info["deps"],
        outs=stage_info["outs"],
        signature=stage_info["signature"],
    )


def _mark_stage_failed(
    state: StageState,
    reason: str,
    stage_states: dict[str, StageState],
    error_mode: OnError,
) -> None:
    """Mark a stage as failed and handle downstream effects."""
    state.result = {"status": "failed", "reason": reason, "output_lines": []}
    state.status = StageStatus.FAILED
    state.end_time = time.perf_counter()
    _handle_stage_failure(state.name, stage_states, error_mode)


def _handle_stage_failure(
    failed_stage: str,
    stage_states: dict[str, StageState],
    error_mode: OnError,
) -> None:
    """Handle stage failure by marking downstream stages as skipped."""
    if error_mode == OnError.IGNORE:
        return

    to_skip = set[str]()
    bfs_queue = collections.deque([failed_stage])
    visited = set[str]()

    while bfs_queue:
        current = bfs_queue.popleft()
        if current in visited:
            continue
        visited.add(current)

        state = stage_states.get(current)
        if not state:
            continue

        for downstream in state.downstream:
            if downstream not in visited:
                to_skip.add(downstream)
                bfs_queue.append(downstream)

    for stage_name in to_skip:
        state = stage_states.get(stage_name)
        if state and state.status == StageStatus.READY:
            state.status = StageStatus.SKIPPED
            state.result = {
                "status": "skipped",
                "reason": f"upstream '{failed_stage}' failed",
                "output_lines": [],
            }


def execute_stage_worker(
    stage_name: str,
    stage_info: WorkerStageInfo,
    cache_dir: pathlib.Path,
    output_queue: mp.Queue[OutputMessage],
) -> StageResult:
    """Worker function executed in separate process. Must be module-level for pickling."""
    output_lines: list[tuple[str, bool]] = []
    files_cache_dir = cache_dir / "files"

    stage_lock = lock.StageLock(stage_name, cache_dir)
    current_fingerprint = stage_info["fingerprint"]
    current_params = extract_params(stage_info)
    dep_hashes, missing = hash_dependencies(stage_info.get("deps", []))
    stage_outs = stage_info.get("outs", [])

    if missing:
        return {
            "status": "failed",
            "reason": f"missing deps: {', '.join(missing)}",
            "output_lines": [],
        }

    changed, reason = stage_lock.is_changed(current_fingerprint, current_params, dep_hashes)

    lock_data_prev = stage_lock.read()
    if lock_data_prev is not None and "output_hashes" not in lock_data_prev:
        changed, reason = True, "outputs not cached"

    if not changed:
        restored_all = _restore_outputs_from_cache(stage_outs, lock_data_prev, files_cache_dir)
        if not restored_all:
            changed, reason = True, "outputs missing from cache"

    if not changed:
        return {"status": "skipped", "reason": "unchanged", "output_lines": []}

    try:
        with _execution_lock(stage_name, cache_dir):
            _prepare_outputs_for_execution(stage_outs, lock_data_prev, files_cache_dir)

            output_lines = _run_stage_function_with_capture(
                stage_info["func"], stage_name, output_queue
            )

            output_hashes = _save_outputs_to_cache(stage_outs, files_cache_dir)

            lock_data: LockData = {
                "code_manifest": current_fingerprint,
                "params": current_params,
                "dep_hashes": dict(sorted(dep_hashes.items())),
                "output_hashes": dict(sorted(output_hashes.items())),
            }
            stage_lock.write(lock_data)

        return {"status": "ran", "reason": reason, "output_lines": output_lines}

    except exceptions.StageAlreadyRunningError as e:
        return {"status": "failed", "reason": str(e), "output_lines": output_lines}
    except exceptions.OutputMissingError as e:
        return {"status": "failed", "reason": str(e), "output_lines": output_lines}
    except SystemExit as e:
        return {
            "status": "failed",
            "reason": f"Stage called sys.exit({e.code})",
            "output_lines": output_lines,
        }
    except KeyboardInterrupt:
        return {"status": "failed", "reason": "KeyboardInterrupt", "output_lines": output_lines}
    except Exception as e:
        return {"status": "failed", "reason": str(e), "output_lines": output_lines}


def _restore_outputs_from_cache(
    stage_outs: list[outputs.BaseOut],
    lock_data: LockData | None,
    files_cache_dir: pathlib.Path,
) -> bool:
    """Restore missing outputs from cache. Returns True if all restored successfully."""
    if lock_data is None:
        return False

    output_hashes = lock_data.get("output_hashes", {})
    for out in stage_outs:
        path = pathlib.Path(out.path)
        if path.exists():
            continue

        output_hash = output_hashes.get(out.path)
        if output_hash is None:
            if out.cache:
                return False
            continue

        if not cache.restore_from_cache(path, output_hash, files_cache_dir):
            return False

    return True


def _prepare_outputs_for_execution(
    stage_outs: Sequence[outputs.BaseOut],
    lock_data: LockData | None,
    files_cache_dir: pathlib.Path,
) -> None:
    """Prepare outputs before stage execution - delete or restore for incremental."""
    output_hashes = lock_data.get("output_hashes", {}) if lock_data else {}

    for out in stage_outs:
        path = pathlib.Path(out.path)

        if isinstance(out, outputs.IncrementalOut):
            # IncrementalOut: restore from cache as writable copy
            cache.remove_output(path)  # Clear any stale state first
            out_hash = output_hashes.get(out.path)
            if out_hash:
                # COPY mode makes file writable (not symlink to read-only cache)
                restored = cache.restore_from_cache(
                    path, out_hash, files_cache_dir, cache.LinkMode.COPY
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
) -> dict[str, OutputHash]:
    """Save outputs to cache after successful execution."""
    output_hashes = dict[str, OutputHash]()

    for out in stage_outs:
        path = pathlib.Path(out.path)
        if not path.exists():
            raise exceptions.OutputMissingError(f"Stage did not produce output: {out.path}")

        if out.cache:
            output_hashes[out.path] = cache.save_to_cache(path, files_cache_dir)
        else:
            output_hashes[out.path] = None

    return output_hashes


def _run_stage_function_with_capture(
    func: Callable[..., Any],
    stage_name: str,
    output_queue: mp.Queue[OutputMessage],
) -> list[tuple[str, bool]]:
    """Run stage function with stdout/stderr capture, streaming to queue."""
    output_lines: list[tuple[str, bool]] = []

    old_stdout = sys.stdout
    old_stderr = sys.stderr

    stdout_capture = _QueueStreamCapture(
        stage_name, output_queue, is_stderr=False, output_lines=output_lines
    )
    stderr_capture = _QueueStreamCapture(
        stage_name, output_queue, is_stderr=True, output_lines=output_lines
    )

    try:
        # Cast to TextIO via object - _QueueStreamCapture implements the TextIO protocol
        sys.stdout = cast("TextIO", cast("object", stdout_capture))
        sys.stderr = cast("TextIO", cast("object", stderr_capture))
        func()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        stdout_capture.flush()
        stderr_capture.flush()

    return output_lines


class _QueueStreamCapture(io.TextIOBase):
    """Capture stream output and send to queue for main process.

    Implements the TextIO protocol to allow assignment to sys.stdout/stderr.
    """

    _stage_name: str
    _queue: mp.Queue[OutputMessage]
    _is_stderr: bool
    _output_lines: list[tuple[str, bool]]
    _buffer: str

    def __init__(
        self,
        stage_name: str,
        output_q: mp.Queue[OutputMessage],
        is_stderr: bool,
        output_lines: list[tuple[str, bool]],
    ) -> None:
        self._stage_name = stage_name
        self._queue = output_q
        self._is_stderr = is_stderr
        self._output_lines = output_lines
        self._buffer = ""

    # TextIO protocol attributes as properties (io.TextIOBase declares these read-only)
    @property
    @override
    def encoding(self) -> str:  # pyright: ignore[reportIncompatibleVariableOverride]
        return "utf-8"

    @property
    @override
    def errors(self) -> str | None:  # pyright: ignore[reportIncompatibleVariableOverride]
        return "strict"

    @property
    @override
    def newlines(self) -> str | tuple[str, ...] | None:  # pyright: ignore[reportIncompatibleVariableOverride]
        return None

    @override
    def fileno(self) -> int:
        raise OSError("_QueueStreamCapture has no underlying file descriptor")

    @override
    def isatty(self) -> bool:
        return False

    @override
    def readable(self) -> bool:
        return False

    @override
    def seekable(self) -> bool:
        return False

    @override
    def writable(self) -> bool:
        return True

    def _send_line(self, line: str) -> None:
        """Save line locally and send to queue for real-time display."""
        self._output_lines.append((line, self._is_stderr))
        # Queue failure only affects real-time display; output is already saved locally
        with contextlib.suppress(queue.Full, ValueError, OSError):
            self._queue.put((self._stage_name, line, self._is_stderr), block=False)

    @override
    def write(self, s: str) -> int:
        self._buffer += s
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line:
                self._send_line(line)
        return len(s)

    @override
    def flush(self) -> None:
        if self._buffer:
            self._send_line(self._buffer)
            self._buffer = ""


def _build_results(stage_states: dict[str, StageState]) -> dict[str, ExecutionSummary]:
    """Build results dict from stage states."""
    results = dict[str, ExecutionSummary]()
    for name, state in stage_states.items():
        if state.result:
            results[name] = {
                "status": state.result["status"],
                "reason": state.result["reason"],
            }
        elif state.status == StageStatus.SKIPPED:
            results[name] = {"status": "skipped", "reason": "upstream failed"}
        else:
            results[name] = {"status": "unknown", "reason": "never executed"}
    return results


def _check_uncached_incremental_outputs(
    execution_order: list[str],
    cache_dir: pathlib.Path,
) -> list[tuple[str, str]]:
    """Check for IncrementalOut files that exist but aren't cached.

    Returns list of (stage_name, output_path) tuples for uncached files.
    """
    uncached = list[tuple[str, str]]()

    for stage_name in execution_order:
        stage_info = registry.REGISTRY.get(stage_name)
        stage_outs = stage_info.get("outs", [])

        # Read lock file to get cached output hashes
        stage_lock = lock.StageLock(stage_name, cache_dir)
        lock_data = stage_lock.read()
        output_hashes = lock_data.get("output_hashes", {}) if lock_data else {}

        for out in stage_outs:
            if isinstance(out, outputs.IncrementalOut):
                path = pathlib.Path(out.path)
                # File exists on disk but has no cache entry
                if path.exists() and out.path not in output_hashes:
                    uncached.append((stage_name, out.path))

    return uncached


def extract_params(stage_info: WorkerStageInfo | dict[str, Any]) -> dict[str, Any]:
    """Extract parameter values from stage signature defaults."""
    sig = stage_info.get("signature")
    if not sig:
        return {}
    return {
        name: param.default
        for name, param in sig.parameters.items()
        if param.default is not param.empty
    }


def hash_dependencies(deps: list[str]) -> tuple[dict[str, str], list[str]]:
    """Hash all dependency files and directories. Returns (hashes, missing_files)."""
    hashes = dict[str, str]()
    missing = list[str]()
    for dep in deps:
        path = pathlib.Path(dep)
        try:
            if path.is_dir():
                tree_hash, _ = cache.hash_directory(path)
                hashes[dep] = tree_hash
            else:
                hashes[dep] = hash_file(path)
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
            fd = os.open(sentinel, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"pid: {os.getpid()}\n".encode())
            os.close(fd)
            return sentinel
        except FileExistsError:
            pass

        try:
            content = sentinel.read_text()
            pid = int(content.split(":")[1].strip())
        except (ValueError, IndexError, OSError):
            pid = None

        if pid is not None and pid > 0 and _is_process_alive(pid):
            raise exceptions.StageAlreadyRunningError(
                f"Stage '{stage_name}' is already running (PID {pid})"
            )

        sentinel.unlink(missing_ok=True)
        if pid is not None:
            logger.warning(f"Removed stale lock file: {sentinel} (was PID {pid})")

    raise exceptions.StageAlreadyRunningError(
        f"Failed to acquire lock for '{stage_name}' after {_MAX_LOCK_ATTEMPTS} attempts"
    )


def _is_process_alive(pid: int) -> bool:
    """Check if process is still running."""
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except ProcessLookupError:
        return False
