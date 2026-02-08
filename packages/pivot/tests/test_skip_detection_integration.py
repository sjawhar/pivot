"""Integration tests for skip detection pipeline end-to-end.

These tests exercise the full three-tier skip detection algorithm:
1. O(1) generation check
2. O(n) lock file comparison
3. Run cache lookup

Unlike unit tests that mock internals, these tests use the actual execute_stage()
function and simulate coordinator behavior (applying deferred writes) between runs.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

from pivot import executor, loaders, outputs, stage_def
from pivot.executor import worker
from pivot.storage import cache, lock, state

if TYPE_CHECKING:
    import inspect
    import multiprocessing as mp
    import pathlib
    from collections.abc import Callable

    from pytest_mock import MockerFixture

    from pivot.executor import WorkerStageInfo
    from pivot.types import OutputMessage, StageResult


# =============================================================================
# Module-level helpers (required for pickling in worker processes)
# =============================================================================


def _helper_noop() -> None:
    """No-op stage for skip detection tests."""


def _helper_write_output(path: pathlib.Path) -> None:
    """Write a marker file."""
    path.write_text("output")


def _make_stage_info(
    func: Callable[..., object],
    tmp_path: pathlib.Path,
    *,
    fingerprint: dict[str, str] | None = None,
    deps: list[str] | None = None,
    outs: list[outputs.BaseOut] | None = None,
    params: stage_def.StageParams | None = None,
    signature: inspect.Signature | None = None,
    checkout_modes: list[cache.CheckoutMode] | None = None,
    run_id: str = "test_run",
    force: bool = False,
    no_commit: bool = False,
    dep_specs: dict[str, stage_def.FuncDepSpec] | None = None,
    out_specs: dict[str, outputs.BaseOut] | None = None,
    params_arg_name: str | None = None,
) -> WorkerStageInfo:
    """Create a WorkerStageInfo with sensible defaults for testing."""
    return {
        "func": func,
        "fingerprint": fingerprint or {"self:test": "abc123"},
        "deps": deps or [],
        "signature": signature,
        "outs": outs or [],
        "params": params,
        "variant": None,
        "overrides": {},
        "checkout_modes": checkout_modes
        or [cache.CheckoutMode.HARDLINK, cache.CheckoutMode.SYMLINK, cache.CheckoutMode.COPY],
        "run_id": run_id,
        "force": force,
        "no_commit": no_commit,
        "dep_specs": dep_specs or {},
        "out_specs": out_specs or {},
        "params_arg_name": params_arg_name,
        "project_root": tmp_path,
        "state_dir": tmp_path / ".pivot",
    }


def _apply_deferred_writes(
    stage_name: str,
    stage_info: WorkerStageInfo,
    result: StageResult,
) -> None:
    """Simulate coordinator behavior: apply deferred writes from worker result.

    In production, the coordinator (engine) calls apply_deferred_writes after
    each stage completes. We replicate this to test the full skip detection
    pipeline end-to-end.
    """
    if "deferred_writes" not in result:
        return
    deferred = result["deferred_writes"]
    state_dir = stage_info["state_dir"]

    # Compute output paths (same logic as coordinator)
    out_paths = [str(out.path) for out in stage_info["outs"]]

    with state.StateDB(state_dir / "state.db") as state_db:
        state_db.apply_deferred_writes(stage_name, out_paths, deferred)


# =============================================================================
# Test: Generation skip with Pivot-produced deps
# =============================================================================


def test_generation_skip_with_pivot_produced_deps(
    worker_env: pathlib.Path,
    output_queue: mp.Queue[OutputMessage],
    tmp_path: pathlib.Path,
    mocker: MockerFixture,
) -> None:
    """Stage with all Pivot-produced deps skips via generation (no hashing).

    Two-stage pipeline: step1 produces intermediate.txt, step2 consumes it.
    After both run and deferred writes are applied, step2's second run should
    skip via generation check (Tier 1) without hashing any files.
    """
    (tmp_path / "input.txt").write_text("data")

    def step1_func() -> None:
        data = (tmp_path / "input.txt").read_text()
        (tmp_path / "intermediate.txt").write_text(data.upper())

    def step2_func() -> None:
        data = (tmp_path / "intermediate.txt").read_text()
        (tmp_path / "final.txt").write_text(f"Final: {data}")

    step1_out = outputs.Out(str(tmp_path / "intermediate.txt"), loader=loaders.PathOnly())
    step2_out = outputs.Out(str(tmp_path / "final.txt"), loader=loaders.PathOnly())

    step1_info = _make_stage_info(
        step1_func,
        tmp_path,
        fingerprint={"self:step1": "fp1"},
        deps=["input.txt"],
        outs=[step1_out],
    )
    step2_info = _make_stage_info(
        step2_func,
        tmp_path,
        fingerprint={"self:step2": "fp2"},
        deps=["intermediate.txt"],
        outs=[step2_out],
    )

    # First run: both stages execute
    result1_step1 = executor.execute_stage("step1", step1_info, worker_env, output_queue)
    assert result1_step1["status"] == "ran"
    _apply_deferred_writes("step1", step1_info, result1_step1)

    result1_step2 = executor.execute_stage("step2", step2_info, worker_env, output_queue)
    assert result1_step2["status"] == "ran"
    _apply_deferred_writes("step2", step2_info, result1_step2)

    assert (tmp_path / "final.txt").read_text() == "Final: DATA"

    # Second run: step1 skips (external dep falls through to hash check)
    result2_step1 = executor.execute_stage("step1", step1_info, worker_env, output_queue)
    assert result2_step1["status"] == "skipped"
    # No need to re-apply deferred writes for skipped stages (no deferred_writes in result)

    # Second run: step2 should skip via generation check (intermediate.txt is Pivot-produced)
    hash_spy = mocker.patch(
        "pivot.executor.worker.hash_dependencies",
        autospec=True,
        wraps=worker.hash_dependencies,
    )
    result2_step2 = executor.execute_stage("step2", step2_info, worker_env, output_queue)

    assert result2_step2["status"] == "skipped", f"Expected skip, got: {result2_step2}"
    assert result2_step2["reason"] == "unchanged (generation)", (
        f"Expected generation skip, got: {result2_step2['reason']}"
    )
    hash_spy.assert_not_called()


# =============================================================================
# Test: External dep falls through to hash-based check
# =============================================================================


def test_external_dep_falls_through_to_hash_check(
    worker_env: pathlib.Path,
    output_queue: mp.Queue[OutputMessage],
    tmp_path: pathlib.Path,
    mocker: MockerFixture,
) -> None:
    """Stage with external dep (no generation counter) falls through to hash-based check.

    External files aren't produced by any Pivot stage, so they lack generation
    counters in StateDB. The generation check returns False and execution falls
    through to Tier 2 (lock file comparison with full hashing).
    """
    (tmp_path / "external_data.txt").write_text("external content")

    def stage_func() -> None:
        data = (tmp_path / "external_data.txt").read_text()
        (tmp_path / "output.txt").write_text(data.upper())

    out = outputs.Out(str(tmp_path / "output.txt"), loader=loaders.PathOnly())
    stage_info = _make_stage_info(
        stage_func,
        tmp_path,
        fingerprint={"self:stage": "fp1"},
        deps=["external_data.txt"],
        outs=[out],
    )

    # First run
    result1 = executor.execute_stage("test_stage", stage_info, worker_env, output_queue)
    assert result1["status"] == "ran"
    _apply_deferred_writes("test_stage", stage_info, result1)

    # Second run: should skip via hash check (not generation)
    hash_spy = mocker.patch(
        "pivot.executor.worker.hash_dependencies",
        autospec=True,
        wraps=worker.hash_dependencies,
    )
    result2 = executor.execute_stage("test_stage", stage_info, worker_env, output_queue)

    assert result2["status"] == "skipped"
    assert result2["reason"] == "unchanged", (
        f"Expected hash-based skip (not generation), got: {result2['reason']}"
    )
    hash_spy.assert_called_once()


# =============================================================================
# Test: Cleared StateDB degrades gracefully to lock comparison
# =============================================================================


def test_cleared_statedb_degrades_to_lock_comparison(
    worker_env: pathlib.Path,
    output_queue: mp.Queue[OutputMessage],
    tmp_path: pathlib.Path,
) -> None:
    """After clearing StateDB, skip detection degrades to lock comparison.

    The generation check fails (no generations in cleared DB), but the lock
    file still has valid fingerprint + dep_hashes, so Tier 2 can still skip.
    """
    (tmp_path / "input.txt").write_text("data")

    def stage_func() -> None:
        (tmp_path / "output.txt").write_text("result")

    out = outputs.Out(str(tmp_path / "output.txt"), loader=loaders.PathOnly())
    stage_info = _make_stage_info(
        stage_func,
        tmp_path,
        fingerprint={"self:stage": "fp1"},
        deps=["input.txt"],
        outs=[out],
    )

    # First run - creates lock file and populates StateDB
    result1 = executor.execute_stage("test_stage", stage_info, worker_env, output_queue)
    assert result1["status"] == "ran"
    _apply_deferred_writes("test_stage", stage_info, result1)

    # Clear StateDB (simulates DB corruption or manual reset)
    lmdb_dir = tmp_path / ".pivot" / "state.lmdb"
    if lmdb_dir.exists():
        shutil.rmtree(lmdb_dir)

    # Second run: lock file still exists, so hash comparison should detect unchanged
    result2 = executor.execute_stage("test_stage", stage_info, worker_env, output_queue)
    assert result2["status"] == "skipped", f"Expected skip via lock comparison, got: {result2}"
    assert "unchanged" in result2["reason"], (
        f"Expected 'unchanged' reason, got: {result2['reason']}"
    )
    # Should NOT be generation-based (StateDB was cleared)
    assert "generation" not in result2["reason"], (
        "Should not use generation skip after StateDB clear"
    )


# =============================================================================
# Test: Missing lock file triggers full run
# =============================================================================


def test_missing_lock_file_triggers_full_run(
    worker_env: pathlib.Path,
    output_queue: mp.Queue[OutputMessage],
    tmp_path: pathlib.Path,
) -> None:
    """When lock file doesn't exist, stage must do a full run.

    This is the cold start case — no previous execution recorded.
    """
    (tmp_path / "input.txt").write_text("data")

    counter_file = tmp_path / "run_counter.txt"
    counter_file.write_text("0")

    def stage_func() -> None:
        count = int(counter_file.read_text())
        counter_file.write_text(str(count + 1))
        (tmp_path / "output.txt").write_text("done")

    out = outputs.Out(str(tmp_path / "output.txt"), loader=loaders.PathOnly())
    stage_info = _make_stage_info(
        stage_func,
        tmp_path,
        fingerprint={"self:stage": "fp1"},
        deps=["input.txt"],
        outs=[out],
    )

    # Verify no lock file exists
    stages_dir = lock.get_stages_dir(tmp_path / ".pivot")
    stage_lock = lock.StageLock("test_stage", stages_dir)
    assert stage_lock.read() is None, "Lock file should not exist before first run"

    # First run: must execute (no lock file)
    result1 = executor.execute_stage("test_stage", stage_info, worker_env, output_queue)
    assert result1["status"] == "ran"
    assert result1["reason"] == "No previous run"
    assert counter_file.read_text() == "1", "Stage should have executed once"

    # Lock file should now exist
    assert stage_lock.read() is not None, "Lock file should exist after first run"

    # Apply deferred writes (coordinator behavior)
    _apply_deferred_writes("test_stage", stage_info, result1)

    # Second run: should skip
    result2 = executor.execute_stage("test_stage", stage_info, worker_env, output_queue)
    assert result2["status"] == "skipped"
    assert counter_file.read_text() == "1", "Stage should not have re-executed"


# =============================================================================
# Test: Generation invalidation propagates through pipeline
# =============================================================================


def test_generation_invalidation_propagates(
    worker_env: pathlib.Path,
    output_queue: mp.Queue[OutputMessage],
    tmp_path: pathlib.Path,
) -> None:
    """When upstream re-runs, downstream detects generation change and re-runs.

    step1: input.txt → intermediate.txt
    step2: intermediate.txt → final.txt

    After modifying input.txt, step1 re-runs and gets new output generation.
    step2 detects the generation mismatch and re-runs too.
    """
    (tmp_path / "input.txt").write_text("original")

    def step1_func() -> None:
        data = (tmp_path / "input.txt").read_text()
        (tmp_path / "intermediate.txt").write_text(data.upper())

    def step2_func() -> None:
        data = (tmp_path / "intermediate.txt").read_text()
        (tmp_path / "final.txt").write_text(f"Final: {data}")

    step1_out = outputs.Out(str(tmp_path / "intermediate.txt"), loader=loaders.PathOnly())
    step2_out = outputs.Out(str(tmp_path / "final.txt"), loader=loaders.PathOnly())

    step1_info = _make_stage_info(
        step1_func,
        tmp_path,
        fingerprint={"self:step1": "fp1"},
        deps=["input.txt"],
        outs=[step1_out],
    )
    step2_info = _make_stage_info(
        step2_func,
        tmp_path,
        fingerprint={"self:step2": "fp2"},
        deps=["intermediate.txt"],
        outs=[step2_out],
    )

    # First run: both stages execute
    r1_s1 = executor.execute_stage("step1", step1_info, worker_env, output_queue)
    assert r1_s1["status"] == "ran"
    _apply_deferred_writes("step1", step1_info, r1_s1)

    r1_s2 = executor.execute_stage("step2", step2_info, worker_env, output_queue)
    assert r1_s2["status"] == "ran"
    _apply_deferred_writes("step2", step2_info, r1_s2)

    assert (tmp_path / "final.txt").read_text() == "Final: ORIGINAL"

    # Modify input → step1 must re-run
    (tmp_path / "input.txt").write_text("modified")

    r2_s1 = executor.execute_stage("step1", step1_info, worker_env, output_queue)
    assert r2_s1["status"] == "ran"
    _apply_deferred_writes("step1", step1_info, r2_s1)

    # step2: intermediate.txt generation changed → must re-run
    r2_s2 = executor.execute_stage("step2", step2_info, worker_env, output_queue)
    assert r2_s2["status"] == "ran", f"Expected re-run due to generation change, got: {r2_s2}"
    _apply_deferred_writes("step2", step2_info, r2_s2)

    assert (tmp_path / "final.txt").read_text() == "Final: MODIFIED"


# =============================================================================
# Test: Deferred file hash write-back populates StateDB
# =============================================================================


def test_deferred_file_hash_writeback(
    worker_env: pathlib.Path,
    output_queue: mp.Queue[OutputMessage],
    tmp_path: pathlib.Path,
) -> None:
    """Workers collect file hash entries during dep hashing and write them back.

    After the coordinator applies deferred writes, the file hashes should be
    in StateDB, enabling O(1) cache lookups on subsequent runs.
    """
    (tmp_path / "input.txt").write_text("data")

    def stage_func() -> None:
        (tmp_path / "output.txt").write_text("output")

    out = outputs.Out(str(tmp_path / "output.txt"), loader=loaders.PathOnly())
    stage_info = _make_stage_info(
        stage_func,
        tmp_path,
        fingerprint={"self:stage": "fp1"},
        deps=["input.txt"],
        outs=[out],
    )

    result = executor.execute_stage("test_stage", stage_info, worker_env, output_queue)
    assert result["status"] == "ran", f"Expected ran, got {result['status']}: {result['reason']}"

    assert "deferred_writes" in result, "Result should have deferred_writes"
    deferred = result["deferred_writes"]
    assert "file_hash_entries" in deferred, "Deferred writes should include file_hash_entries"
    assert len(deferred["file_hash_entries"]) >= 1, "Should have at least 1 file hash entry"

    _apply_deferred_writes("test_stage", stage_info, result)

    input_path = tmp_path / "input.txt"
    with state.StateDB(stage_info["state_dir"] / "state.db", readonly=True) as state_db:
        stat = input_path.stat()
        cached = state_db.get_many([(input_path, stat)])
        assert cached.get(input_path) is not None, (
            "File hash should be in StateDB after deferred write-back"
        )
