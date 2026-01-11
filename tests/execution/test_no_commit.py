"""Tests for --no-commit mode and pivot commit command."""

from __future__ import annotations

import datetime
import multiprocessing as mp
from multiprocessing import queues as mp_queues
from typing import TYPE_CHECKING, Any

import pytest

from pivot import outputs, project, reactive, run_history
from pivot.executor import commit as commit_mod
from pivot.executor import worker
from pivot.storage import lock, state
from pivot.types import OutputMessage, StageStatus

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable


@pytest.fixture
def worker_env(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Set up worker execution environment."""
    cache_dir = tmp_path / ".pivot" / "cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "files").mkdir()
    (cache_dir / "stages").mkdir()
    (tmp_path / ".pivot" / "pending" / "stages").mkdir(parents=True)
    (tmp_path / ".pivot").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    monkeypatch.chdir(tmp_path)
    return cache_dir


@pytest.fixture
def output_queue() -> mp_queues.Queue[OutputMessage]:
    """Create a multiprocessing queue for worker output."""
    # mp.Manager().Queue() returns a proxy that's compatible but not the exact type
    return mp.Manager().Queue()  # pyright: ignore[reportReturnType]


def _make_stage_info(
    func: Callable[..., Any],
    tmp_path: pathlib.Path,
    *,
    deps: list[str] | None = None,
    outs: list[outputs.BaseOut] | None = None,
    fingerprint: dict[str, str] | None = None,
    run_id: str = "test_run",
    no_commit: bool = False,
    no_cache: bool = False,
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
        cwd=tmp_path,
        checkout_modes=["hardlink", "symlink", "copy"],
        run_id=run_id,
        force=force,
        no_commit=no_commit,
        no_cache=no_cache,
    )


# -----------------------------------------------------------------------------
# No-commit mode tests
# -----------------------------------------------------------------------------


def test_no_commit_writes_to_pending_lock(
    worker_env: pathlib.Path, tmp_path: pathlib.Path, output_queue: mp_queues.Queue[OutputMessage]
) -> None:
    """When no_commit=True, lock is written to pending directory, not production."""
    (tmp_path / "input.txt").write_text("input data")

    def stage_func() -> None:
        (tmp_path / "output.txt").write_text("output data")

    stage_info = _make_stage_info(
        stage_func,
        tmp_path,
        deps=[str(tmp_path / "input.txt")],
        outs=[outputs.Out("output.txt")],
        no_commit=True,
    )

    result = worker.execute_stage("test_stage", stage_info, worker_env, output_queue)

    assert result["status"] == "ran"

    # Pending lock should exist
    pending_lock = lock.get_pending_lock("test_stage", tmp_path)
    assert pending_lock.path.exists(), "Pending lock should be written"
    pending_data = pending_lock.read()
    assert pending_data is not None

    # Production lock should NOT exist
    production_lock = lock.StageLock("test_stage", worker_env)
    assert not production_lock.path.exists(), "Production lock should NOT be written"


def test_no_commit_still_writes_to_cache(
    worker_env: pathlib.Path, tmp_path: pathlib.Path, output_queue: mp_queues.Queue[OutputMessage]
) -> None:
    """When no_commit=True, outputs are still written to cache (content-addressed)."""
    (tmp_path / "input.txt").write_text("input data")

    def stage_func() -> None:
        (tmp_path / "output.txt").write_text("output data")

    stage_info = _make_stage_info(
        stage_func,
        tmp_path,
        deps=[str(tmp_path / "input.txt")],
        outs=[outputs.Out("output.txt")],
        no_commit=True,
    )

    result = worker.execute_stage("test_stage", stage_info, worker_env, output_queue)

    assert result["status"] == "ran"

    # Check that cache files exist
    cache_files = list((worker_env / "files").rglob("*"))
    # Should have at least the hash directory and file
    cache_files_not_dirs = [f for f in cache_files if f.is_file()]
    assert len(cache_files_not_dirs) > 0, "Cache should have files"


def test_second_no_commit_run_uses_pending_lock_for_skip(
    worker_env: pathlib.Path, tmp_path: pathlib.Path, output_queue: mp_queues.Queue[OutputMessage]
) -> None:
    """Second --no-commit run should skip if inputs unchanged (uses pending lock)."""
    (tmp_path / "input.txt").write_text("input data")

    def stage_func() -> None:
        (tmp_path / "output.txt").write_text("output data")

    stage_info = _make_stage_info(
        stage_func,
        tmp_path,
        deps=[str(tmp_path / "input.txt")],
        outs=[outputs.Out("output.txt")],
        no_commit=True,
    )

    # First run
    result1 = worker.execute_stage("test_stage", stage_info, worker_env, output_queue)
    assert result1["status"] == "ran"

    # Second run should skip (using pending lock for comparison)
    result2 = worker.execute_stage("test_stage", stage_info, worker_env, output_queue)
    assert result2["status"] == "skipped"


# -----------------------------------------------------------------------------
# Commit tests
# -----------------------------------------------------------------------------


def test_list_pending_stages_empty(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """list_pending_stages returns empty list when no pending locks."""
    (tmp_path / ".pivot" / "pending" / "stages").mkdir(parents=True)
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    result = lock.list_pending_stages(tmp_path)
    assert result == []


def test_list_pending_stages_returns_stage_names(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """list_pending_stages returns stage names from pending lock files."""
    pending_stages_dir = tmp_path / ".pivot" / "pending" / "stages"
    pending_stages_dir.mkdir(parents=True)
    (pending_stages_dir / "stage_a.lock").write_text("code_manifest: {}\n")
    (pending_stages_dir / "stage_b.lock").write_text("code_manifest: {}\n")
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    result = lock.list_pending_stages(tmp_path)
    assert result == ["stage_a", "stage_b"]


def test_commit_pending_promotes_to_production(
    worker_env: pathlib.Path, tmp_path: pathlib.Path, output_queue: mp_queues.Queue[OutputMessage]
) -> None:
    """commit_pending promotes pending locks to production."""
    (tmp_path / "input.txt").write_text("input data")

    def stage_func() -> None:
        (tmp_path / "output.txt").write_text("output data")

    stage_info = _make_stage_info(
        stage_func,
        tmp_path,
        deps=[str(tmp_path / "input.txt")],
        outs=[outputs.Out("output.txt")],
        no_commit=True,
    )

    # Run with no_commit
    result = worker.execute_stage("test_stage", stage_info, worker_env, output_queue)
    assert result["status"] == "ran"

    # Pending lock exists, production doesn't
    pending_lock = lock.get_pending_lock("test_stage", tmp_path)
    production_lock = lock.StageLock("test_stage", worker_env)
    assert pending_lock.path.exists()
    assert not production_lock.path.exists()

    # Commit
    committed = commit_mod.commit_pending(worker_env)

    assert committed == ["test_stage"]

    # Now production lock exists, pending doesn't
    assert production_lock.path.exists()
    assert not pending_lock.path.exists()


def test_discard_pending_removes_pending_locks(
    worker_env: pathlib.Path, tmp_path: pathlib.Path, output_queue: mp_queues.Queue[OutputMessage]
) -> None:
    """discard_pending removes pending locks without committing."""
    (tmp_path / "input.txt").write_text("input data")

    def stage_func() -> None:
        (tmp_path / "output.txt").write_text("output data")

    stage_info = _make_stage_info(
        stage_func,
        tmp_path,
        deps=[str(tmp_path / "input.txt")],
        outs=[outputs.Out("output.txt")],
        no_commit=True,
    )

    # Run with no_commit
    result = worker.execute_stage("test_stage", stage_info, worker_env, output_queue)
    assert result["status"] == "ran"

    # Pending lock exists
    pending_lock = lock.get_pending_lock("test_stage", tmp_path)
    assert pending_lock.path.exists()

    # Discard
    discarded = commit_mod.discard_pending()

    assert discarded == ["test_stage"]
    assert not pending_lock.path.exists()

    # Production lock should NOT exist (we discarded, didn't commit)
    production_lock = lock.StageLock("test_stage", worker_env)
    assert not production_lock.path.exists()


def test_commit_nothing_to_commit(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """commit_pending returns empty list when nothing to commit."""
    (tmp_path / ".pivot" / "pending" / "stages").mkdir(parents=True)
    (tmp_path / ".pivot" / "cache" / "stages").mkdir(parents=True)
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    committed = commit_mod.commit_pending()
    assert committed == []


# -----------------------------------------------------------------------------
# Commit correctness tests
# -----------------------------------------------------------------------------


def test_commit_records_generation_at_execution_time(
    worker_env: pathlib.Path, tmp_path: pathlib.Path, output_queue: mp_queues.Queue[OutputMessage]
) -> None:
    """commit_pending records generation from execution time, not commit time.

    When a dependency's generation changes between --no-commit run and commit,
    commit must record the generation from when the stage actually executed
    (stored in pending lock's dep_generations field).
    """
    (tmp_path / "input.txt").write_text("input data v1")
    state_db_path = tmp_path / ".pivot" / "state.db"

    # Set initial generation for input.txt to 5
    with state.StateDB(state_db_path) as db:
        input_path = tmp_path / "input.txt"
        for _ in range(5):  # Increment to gen 5
            db.increment_generation(input_path)
        initial_gen = db.get_generation(input_path)
        assert initial_gen == 5, "Setup: input should be at generation 5"

    def stage_func() -> None:
        (tmp_path / "output.txt").write_text("output data")

    stage_info = _make_stage_info(
        stage_func,
        tmp_path,
        deps=[str(tmp_path / "input.txt")],
        outs=[outputs.Out("output.txt")],
        no_commit=True,
    )

    # Run stage with --no-commit while dep is at generation 5
    result = worker.execute_stage("test_stage", stage_info, worker_env, output_queue)
    assert result["status"] == "ran"

    # Simulate another stage modifying the dep (increment generation to 6)
    with state.StateDB(state_db_path) as db:
        db.increment_generation(tmp_path / "input.txt")
        new_gen = db.get_generation(tmp_path / "input.txt")
        assert new_gen == 6, "After increment: input should be at generation 6"

    # Now commit
    committed = commit_mod.commit_pending(worker_env)
    assert committed == ["test_stage"]

    # Check what generation was recorded for the dependency
    with state.StateDB(state_db_path) as db:
        recorded_gens = db.get_dep_generations("test_stage")
        assert recorded_gens is not None, "Should have recorded dep generations"

        # Get the normalized path key
        normalized_input = str(project.normalize_path(str(tmp_path / "input.txt")))
        recorded_gen = recorded_gens.get(normalized_input)

        # Should record generation 5 (execution time), not 6 (commit time)
        assert recorded_gen == 5, "Should record generation from execution time"


def test_committed_run_cache_entries_survive_pruning(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run cache entries with sentinel run_id survive prune_runs().

    The commit command writes run cache entries with run_id='__committed__',
    a sentinel value that starts with '__' and is never pruned by prune_runs().
    """
    (tmp_path / ".pivot" / "pending" / "stages").mkdir(parents=True)
    (tmp_path / ".pivot" / "cache" / "stages").mkdir(parents=True)
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    state_db_path = tmp_path / ".pivot" / "state.db"

    with state.StateDB(state_db_path) as db:
        # Write a run cache entry with the sentinel run_id (what commit_pending uses)
        commit_entry = run_history.RunCacheEntry(
            run_id=commit_mod.COMMITTED_RUN_ID,
            output_hashes=[run_history.OutputHashEntry(path="output.txt", hash="abc123")],
        )
        db.write_run_cache("test_stage", "input_hash_123", commit_entry)

        # Verify the entry exists
        found = db.lookup_run_cache("test_stage", "input_hash_123")
        assert found is not None, "Setup: commit entry should exist"
        assert found["run_id"] == commit_mod.COMMITTED_RUN_ID

        # Create some actual run history entries
        for i in range(5):
            timestamp = datetime.datetime.now(datetime.UTC).isoformat()
            stage_record = run_history.StageRunRecord(
                input_hash=f"hash_{i}",
                status=StageStatus.RAN,
                reason="changed",
                duration_ms=100,
            )
            run_manifest = run_history.RunManifest(
                run_id=f"run_{i}",
                started_at=timestamp,
                ended_at=timestamp,
                targeted_stages=["stage_a"],
                execution_order=["stage_a"],
                stages={"stage_a": stage_record},
            )
            db.write_run(run_manifest)

        # Now prune with retention=3 (keeps 3 most recent runs)
        deleted = db.prune_runs(retention=3)
        assert deleted == 2, "Should delete 2 oldest runs"

        # Sentinel run_id entries should survive pruning
        found_after_prune = db.lookup_run_cache("test_stage", "input_hash_123")
        assert found_after_prune is not None, "Committed entry should survive pruning"
        assert found_after_prune["run_id"] == commit_mod.COMMITTED_RUN_ID


def test_reactive_engine_passes_no_commit_to_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ReactiveEngine should pass no_commit flag to executor.run."""
    # Track what arguments executor.run was called with
    executor_call_args = dict[str, object]()

    def mock_executor_run(**kwargs: object) -> dict[str, object]:
        executor_call_args.update(kwargs)
        return {}

    # Mock executor.run to capture the call
    monkeypatch.setattr("pivot.executor.run", mock_executor_run)

    # Create engine with no_commit=True
    engine = reactive.ReactiveEngine(
        stages=None,
        single_stage=False,
        cache_dir=None,
        no_commit=True,
    )

    # Call _execute_stages directly (it's what calls executor.run)
    engine._execute_stages(None)

    # Verify no_commit was passed to executor.run
    assert "no_commit" in executor_call_args, "no_commit should be passed to executor.run"
    assert executor_call_args["no_commit"] is True, "no_commit should be True"


def test_reactive_engine_no_commit_defaults_to_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ReactiveEngine should default no_commit to False."""
    executor_call_args = dict[str, object]()

    def mock_executor_run(**kwargs: object) -> dict[str, object]:
        executor_call_args.update(kwargs)
        return {}

    monkeypatch.setattr("pivot.executor.run", mock_executor_run)

    # Create engine without no_commit (should default to False)
    engine = reactive.ReactiveEngine(
        stages=None,
        single_stage=False,
        cache_dir=None,
    )

    engine._execute_stages(None)

    assert "no_commit" in executor_call_args, "no_commit should be passed to executor.run"
    assert executor_call_args["no_commit"] is False, "no_commit should default to False"


# -----------------------------------------------------------------------------
# No-cache mode tests
# -----------------------------------------------------------------------------


def test_no_cache_skips_cache_operations(
    worker_env: pathlib.Path, tmp_path: pathlib.Path, output_queue: mp_queues.Queue[OutputMessage]
) -> None:
    """When no_cache=True, outputs are not saved to cache."""
    (tmp_path / "input.txt").write_text("input data")

    def stage_func() -> None:
        (tmp_path / "output.txt").write_text("output data")

    stage_info = _make_stage_info(
        stage_func,
        tmp_path,
        deps=[str(tmp_path / "input.txt")],
        outs=[outputs.Out("output.txt")],
        no_cache=True,
    )

    result = worker.execute_stage("test_stage", stage_info, worker_env, output_queue)

    assert result["status"] == "ran"

    # Output file should exist
    assert (tmp_path / "output.txt").exists()

    # Cache should NOT have the output file
    cache_files = list((worker_env / "files").rglob("*"))
    cache_files_not_dirs = [f for f in cache_files if f.is_file()]
    assert len(cache_files_not_dirs) == 0, "Cache should be empty when no_cache=True"


def test_no_cache_writes_lock_with_null_hashes(
    worker_env: pathlib.Path, tmp_path: pathlib.Path, output_queue: mp_queues.Queue[OutputMessage]
) -> None:
    """When no_cache=True, lock file is written but with null output hashes."""
    (tmp_path / "input.txt").write_text("input data")

    def stage_func() -> None:
        (tmp_path / "output.txt").write_text("output data")

    stage_info = _make_stage_info(
        stage_func,
        tmp_path,
        deps=[str(tmp_path / "input.txt")],
        outs=[outputs.Out("output.txt")],
        no_cache=True,
    )

    result = worker.execute_stage("test_stage", stage_info, worker_env, output_queue)

    assert result["status"] == "ran"

    # Production lock should exist with null hashes
    production_lock = lock.StageLock("test_stage", worker_env)
    assert production_lock.path.exists(), "Production lock should be written"
    lock_data = production_lock.read()
    assert lock_data is not None
    # Output hash should be None since we didn't cache (uses normalized absolute path as key)
    output_path = str(project.normalize_path("output.txt"))
    assert output_path in lock_data["output_hashes"], f"Expected {output_path} in output_hashes"
    assert lock_data["output_hashes"][output_path] is None


def test_no_cache_second_run_still_skips(
    worker_env: pathlib.Path, tmp_path: pathlib.Path, output_queue: mp_queues.Queue[OutputMessage]
) -> None:
    """Second --no-cache run should skip if inputs unchanged (lock files work)."""
    (tmp_path / "input.txt").write_text("input data")

    execution_count = [0]

    def stage_func() -> None:
        execution_count[0] += 1
        (tmp_path / "output.txt").write_text(f"output data {execution_count[0]}")

    stage_info = _make_stage_info(
        stage_func,
        tmp_path,
        deps=[str(tmp_path / "input.txt")],
        outs=[outputs.Out("output.txt")],
        no_cache=True,
    )

    # First run
    result1 = worker.execute_stage("test_stage", stage_info, worker_env, output_queue)
    assert result1["status"] == "ran"
    assert execution_count[0] == 1

    # Second run should skip (lock file comparison still works)
    result2 = worker.execute_stage("test_stage", stage_info, worker_env, output_queue)
    assert result2["status"] == "skipped"
    assert execution_count[0] == 1  # Should not have executed again


def test_no_cache_incompatible_with_incremental_out(
    worker_env: pathlib.Path, tmp_path: pathlib.Path, output_queue: mp_queues.Queue[OutputMessage]
) -> None:
    """When no_cache=True with IncrementalOut, stage should fail."""
    (tmp_path / "input.txt").write_text("input data")

    def stage_func() -> None:
        (tmp_path / "output.txt").write_text("output data")

    stage_info = _make_stage_info(
        stage_func,
        tmp_path,
        deps=[str(tmp_path / "input.txt")],
        outs=[outputs.IncrementalOut("output.txt")],
        no_cache=True,
    )

    result = worker.execute_stage("test_stage", stage_info, worker_env, output_queue)

    assert result["status"] == "failed"
    assert "IncrementalOut" in result["reason"]


def test_no_cache_with_no_commit(
    worker_env: pathlib.Path, tmp_path: pathlib.Path, output_queue: mp_queues.Queue[OutputMessage]
) -> None:
    """Both --no-cache and --no-commit can be used together."""
    (tmp_path / "input.txt").write_text("input data")

    def stage_func() -> None:
        (tmp_path / "output.txt").write_text("output data")

    stage_info = _make_stage_info(
        stage_func,
        tmp_path,
        deps=[str(tmp_path / "input.txt")],
        outs=[outputs.Out("output.txt")],
        no_cache=True,
        no_commit=True,
    )

    result = worker.execute_stage("test_stage", stage_info, worker_env, output_queue)

    assert result["status"] == "ran"

    # Pending lock should exist (because no_commit=True)
    pending_lock = lock.get_pending_lock("test_stage", tmp_path)
    assert pending_lock.path.exists(), "Pending lock should be written"

    # Production lock should NOT exist
    production_lock = lock.StageLock("test_stage", worker_env)
    assert not production_lock.path.exists(), "Production lock should NOT be written"

    # Cache should be empty (because no_cache=True)
    cache_files = list((worker_env / "files").rglob("*"))
    cache_files_not_dirs = [f for f in cache_files if f.is_file()]
    assert len(cache_files_not_dirs) == 0, "Cache should be empty"


def test_reactive_engine_passes_no_cache_to_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ReactiveEngine should pass no_cache flag to executor.run."""
    executor_call_args = dict[str, object]()

    def mock_executor_run(**kwargs: object) -> dict[str, object]:
        executor_call_args.update(kwargs)
        return {}

    monkeypatch.setattr("pivot.executor.run", mock_executor_run)

    engine = reactive.ReactiveEngine(
        stages=None,
        single_stage=False,
        cache_dir=None,
        no_cache=True,
    )

    engine._execute_stages(None)

    assert "no_cache" in executor_call_args, "no_cache should be passed to executor.run"
    assert executor_call_args["no_cache"] is True, "no_cache should be True"


def test_reactive_engine_no_cache_defaults_to_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ReactiveEngine should default no_cache to False."""
    executor_call_args = dict[str, object]()

    def mock_executor_run(**kwargs: object) -> dict[str, object]:
        executor_call_args.update(kwargs)
        return {}

    monkeypatch.setattr("pivot.executor.run", mock_executor_run)

    engine = reactive.ReactiveEngine(
        stages=None,
        single_stage=False,
        cache_dir=None,
    )

    engine._execute_stages(None)

    assert "no_cache" in executor_call_args, "no_cache should be passed to executor.run"
    assert executor_call_args["no_cache"] is False, "no_cache should default to False"
