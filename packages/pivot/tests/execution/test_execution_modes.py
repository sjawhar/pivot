"""Tests for execution modes: --no-commit and run cache."""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING, Any

from pivot import loaders, outputs
from pivot.executor import worker
from pivot.storage import cache, lock, state
from pivot.types import StageStatus

if TYPE_CHECKING:
    import multiprocessing as mp
    import pathlib
    from collections.abc import Callable

    from pivot.types import OutputMessage


def _make_stage_info(
    func: Callable[..., Any],
    tmp_path: pathlib.Path,
    *,
    deps: list[str] | None = None,
    outs: list[outputs.BaseOut] | None = None,
    fingerprint: dict[str, str] | None = None,
    run_id: str = "test_run",
    no_commit: bool = False,
    force: bool = False,
) -> worker.WorkerStageInfo:
    """Create a WorkerStageInfo for testing."""
    return worker.WorkerStageInfo(
        func=func,
        fingerprint=fingerprint or {"self:test": "abc123"},
        deps=deps or [],
        signature=None,
        outs=outs or [],
        params=None,
        variant=None,
        overrides={},
        checkout_modes=[
            cache.CheckoutMode.HARDLINK,
            cache.CheckoutMode.SYMLINK,
            cache.CheckoutMode.COPY,
        ],
        run_id=run_id,
        force=force,
        no_commit=no_commit,
        dep_specs={},
        out_specs={},
        params_arg_name=None,
        project_root=tmp_path,
        state_dir=tmp_path / ".pivot",
    )


# -----------------------------------------------------------------------------
# No-commit mode tests
# -----------------------------------------------------------------------------


def test_no_commit_produces_outputs_without_production_lock(
    worker_env: pathlib.Path, tmp_path: pathlib.Path, output_queue: mp.Queue[OutputMessage]
) -> None:
    """When no_commit=True, outputs exist on disk but no production lock is written."""
    (tmp_path / "input.txt").write_text("input data")

    def stage_func() -> None:
        (tmp_path / "output.txt").write_text("output data")

    stage_info = _make_stage_info(
        stage_func,
        tmp_path,
        deps=[str(tmp_path / "input.txt")],
        outs=[outputs.Out("output.txt", loader=loaders.PathOnly())],
        no_commit=True,
    )

    result = worker.execute_stage("test_stage", stage_info, worker_env, output_queue)

    assert result["status"] == StageStatus.RAN

    # Output should exist on disk
    assert (tmp_path / "output.txt").exists(), "Output should exist"

    # Production lock should NOT exist
    production_lock = lock.StageLock("test_stage", lock.get_stages_dir(tmp_path / ".pivot"))
    assert not production_lock.path.exists(), "Production lock should NOT be written"

    # No deferred writes should be returned (no lock/cache writes to apply)
    assert "deferred_writes" not in result, "No deferred writes in no_commit mode"


def test_no_commit_does_not_write_to_cache(
    worker_env: pathlib.Path, tmp_path: pathlib.Path, output_queue: mp.Queue[OutputMessage]
) -> None:
    """When no_commit=True, outputs are hashed but NOT written to cache."""
    (tmp_path / "input.txt").write_text("input data")

    def stage_func() -> None:
        (tmp_path / "output.txt").write_text("output data")

    stage_info = _make_stage_info(
        stage_func,
        tmp_path,
        deps=[str(tmp_path / "input.txt")],
        outs=[outputs.Out("output.txt", loader=loaders.PathOnly())],
        no_commit=True,
    )

    result = worker.execute_stage("test_stage", stage_info, worker_env, output_queue)

    assert result["status"] == StageStatus.RAN

    # input_hash should still be computed (needed for future run-cache lookups)
    assert result["input_hash"] is not None, "input_hash should be computed even in no_commit mode"
    assert isinstance(result["input_hash"], str) and len(result["input_hash"]) > 0, (
        "input_hash should be a non-empty string for run-cache lookups"
    )

    # Cache should NOT have any files (no_commit skips cache writes)
    files_dir = worker_env / "files"
    if files_dir.exists():
        cache_files = [f for f in files_dir.rglob("*") if f.is_file()]
        assert len(cache_files) == 0, "Cache should have no files in no_commit mode"


def test_normal_run_after_no_commit_reruns_and_commits(
    worker_env: pathlib.Path, tmp_path: pathlib.Path, output_queue: mp.Queue[OutputMessage]
) -> None:
    """A normal run after --no-commit re-runs and writes lock + cache."""
    (tmp_path / "input.txt").write_text("input data")
    execution_count = [0]

    def stage_func() -> None:
        execution_count[0] += 1
        (tmp_path / "output.txt").write_text("output data")

    # First run with no_commit
    stage_info_nc = _make_stage_info(
        stage_func,
        tmp_path,
        deps=[str(tmp_path / "input.txt")],
        outs=[outputs.Out("output.txt", loader=loaders.PathOnly())],
        no_commit=True,
    )
    result1 = worker.execute_stage("test_stage", stage_info_nc, worker_env, output_queue)
    assert result1["status"] == StageStatus.RAN
    assert execution_count[0] == 1

    # No lock exists after no_commit
    production_lock = lock.StageLock("test_stage", lock.get_stages_dir(tmp_path / ".pivot"))
    assert not production_lock.path.exists(), "Lock should NOT exist after no_commit"

    # Second run with commit (no_commit=False)
    stage_info_commit = _make_stage_info(
        stage_func,
        tmp_path,
        deps=[str(tmp_path / "input.txt")],
        outs=[outputs.Out("output.txt", loader=loaders.PathOnly())],
        no_commit=False,
    )
    result2 = worker.execute_stage("test_stage", stage_info_commit, worker_env, output_queue)
    assert result2["status"] == StageStatus.RAN, "Should re-run since no lock exists"
    assert execution_count[0] == 2, "Stage should execute again"

    # Lock should now exist
    assert production_lock.path.exists(), "Lock should exist after normal run"

    # Cache should have files
    files_dir = worker_env / "files"
    cache_files = [f for f in files_dir.rglob("*") if f.is_file()]
    assert len(cache_files) > 0, "Cache should have files after normal run"


# -----------------------------------------------------------------------------
# Run cache directory output tests
# -----------------------------------------------------------------------------


def test_run_cache_restores_directory_output(
    worker_env: pathlib.Path, tmp_path: pathlib.Path, output_queue: mp.Queue[OutputMessage]
) -> None:
    """Run cache should restore directory outputs including manifest."""
    (tmp_path / "input.txt").write_text("input data")

    execution_count = [0]

    def stage_func() -> None:
        execution_count[0] += 1
        out_dir = tmp_path / "output_dir"
        out_dir.mkdir(exist_ok=True)
        (out_dir / "file1.txt").write_text("content1")
        (out_dir / "file2.txt").write_text("content2")

    stage_info = _make_stage_info(
        stage_func,
        tmp_path,
        deps=[str(tmp_path / "input.txt")],
        outs=[outputs.Out("output_dir/", loader=loaders.PathOnly())],
    )

    # First run - should execute and write to run cache
    result1 = worker.execute_stage("test_stage", stage_info, worker_env, output_queue)
    assert result1["status"] == StageStatus.RAN
    assert execution_count[0] == 1

    # Apply deferred writes (simulating what coordinator does)
    assert "deferred_writes" in result1, "Should have deferred writes for directory output"
    state_db_path = worker_env.parent / "state.db"
    output_paths = [str(out.path) for out in stage_info["outs"]]
    with state.StateDB(state_db_path) as db:
        db.apply_deferred_writes("test_stage", output_paths, result1["deferred_writes"])

    # Verify directory output exists
    output_dir = tmp_path / "output_dir"
    assert output_dir.is_dir()
    assert (output_dir / "file1.txt").read_text() == "content1"
    assert (output_dir / "file2.txt").read_text() == "content2"

    # Delete the lock file so run cache is used instead of lock-based skip
    production_lock = lock.StageLock("test_stage", lock.get_stages_dir(tmp_path / ".pivot"))
    production_lock.path.unlink()

    # Delete the directory output
    shutil.rmtree(output_dir)
    assert not output_dir.exists()

    # Second run - should skip via run cache and restore directory
    result2 = worker.execute_stage("test_stage", stage_info, worker_env, output_queue)
    assert result2["status"] == StageStatus.SKIPPED
    assert "run cache" in result2["reason"], "Should skip via run cache"
    assert execution_count[0] == 1, "Should not have executed again"

    # Verify directory was restored from cache
    assert output_dir.is_dir(), "Directory should be restored"
    assert (output_dir / "file1.txt").exists(), "file1.txt should be restored"
    assert (output_dir / "file2.txt").exists(), "file2.txt should be restored"
    assert (output_dir / "file1.txt").read_text() == "content1"
    assert (output_dir / "file2.txt").read_text() == "content2"


def test_run_cache_reruns_when_noncached_output_missing(
    worker_env: pathlib.Path, tmp_path: pathlib.Path, output_queue: mp.Queue[OutputMessage]
) -> None:
    """Run cache should NOT skip when non-cached output (Metric) is missing.

    Regression test for #243: when a non-cached output is deleted after
    running once, the run cache incorrectly skipped execution instead of
    re-running the stage to recreate the output.

    This test uses BOTH a cached output (Out) and a non-cached output (Metric)
    to ensure the run cache entry is created and the skip path is exercised.
    """
    (tmp_path / "input.txt").write_text("input data")

    execution_count: list[int] = [0]

    def stage_func() -> None:
        execution_count[0] += 1
        (tmp_path / "output.txt").write_text("output data")
        (tmp_path / "metrics.json").write_text(json.dumps({"accuracy": 0.95}))

    # Stage with both cached (Out) and non-cached (Metric) outputs
    stage_info = _make_stage_info(
        stage_func,
        tmp_path,
        deps=[str(tmp_path / "input.txt")],
        outs=[
            outputs.Out("output.txt", loader=loaders.PathOnly()),
            outputs.Metric("metrics.json"),
        ],
    )

    # First run - should execute and write to run cache
    result1 = worker.execute_stage("test_stage", stage_info, worker_env, output_queue)
    assert result1["status"] == StageStatus.RAN
    assert execution_count[0] == 1

    # Apply deferred writes (simulating what coordinator does)
    # With a cached output, deferred_writes MUST contain a run cache entry
    assert "deferred_writes" in result1, "Should have deferred writes with cached output"
    state_db_path = worker_env.parent / "state.db"
    output_paths: list[str] = [str(out.path) for out in stage_info["outs"]]
    with state.StateDB(state_db_path) as db:
        db.apply_deferred_writes("test_stage", output_paths, result1["deferred_writes"])

    # Verify both files exist
    output_file = tmp_path / "output.txt"
    metric_file = tmp_path / "metrics.json"
    assert output_file.exists()
    assert metric_file.exists()

    # Delete the lock file so run cache is used instead of lock-based skip
    production_lock = lock.StageLock("test_stage", lock.get_stages_dir(tmp_path / ".pivot"))
    production_lock.path.unlink()

    # Delete the metric file (non-cached output)
    # The cached output.txt remains on disk
    metric_file.unlink()
    assert not metric_file.exists()
    assert output_file.exists(), "Cached output should still exist"

    # Second run - should re-run because the non-cached metric file is missing
    # Before the fix, this incorrectly skipped via run cache
    result2 = worker.execute_stage("test_stage", stage_info, worker_env, output_queue)
    assert result2["status"] == StageStatus.RAN, "Should re-run when non-cached output is missing"
    assert execution_count[0] == 2, "Stage should have executed again"

    # Verify the metric file was recreated
    assert metric_file.exists(), "Metric file should be recreated"
