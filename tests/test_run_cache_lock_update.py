from __future__ import annotations

from typing import TYPE_CHECKING

from pivot import executor, loaders, outputs
from pivot.storage import cache, lock, state

if TYPE_CHECKING:
    import multiprocessing as mp
    import pathlib
    from collections.abc import Callable
    from typing import Any

    from pivot.executor import WorkerStageInfo
    from pivot.types import OutputMessage, StageResult


def _apply_deferred_writes(
    stage_name: str, stage_info: WorkerStageInfo, result: StageResult, state_db_path: pathlib.Path
) -> None:
    """Apply deferred writes from stage result (normally done by coordinator)."""
    if "deferred_writes" not in result:
        return
    output_paths = [str(out.path) for out in stage_info["outs"]]
    with state.StateDB(state_db_path) as db:
        db.apply_deferred_writes(stage_name, output_paths, result["deferred_writes"])


def _make_stage_info(
    func: Callable[..., Any],
    tmp_path: pathlib.Path,
    *,
    fingerprint: dict[str, str] | None = None,
    deps: list[str] | None = None,
    outs: list[outputs.BaseOut] | None = None,
    run_id: str = "test_run",
) -> WorkerStageInfo:
    """Create a WorkerStageInfo with sensible defaults for testing."""
    return {
        "func": func,
        "fingerprint": fingerprint or {"self:test": "abc123"},
        "deps": deps or [],
        "signature": None,
        "outs": outs or [],
        "params": None,
        "variant": None,
        "overrides": {},
        "checkout_modes": [
            cache.CheckoutMode.HARDLINK,
            cache.CheckoutMode.SYMLINK,
            cache.CheckoutMode.COPY,
        ],
        "run_id": run_id,
        "force": False,
        "no_commit": False,
        "dep_specs": {},
        "out_specs": {},
        "params_arg_name": None,
        "project_root": tmp_path,
        "state_dir": tmp_path / ".pivot",
    }


# =============================================================================
# Run Cache Lock Update Tests
# =============================================================================


def test_run_cache_skip_updates_lock_file(
    worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage], tmp_path: pathlib.Path
) -> None:
    """Run cache skip should update lock file with current state.

    This test verifies the fix for the status/run disagreement bug where:
    1. Run stage (creates lock with state A)
    2. Modify dep to state B
    3. Run again (creates lock with state B, run cache has A and B)
    4. Revert dep to state A
    5. Run again -> skips via run cache BUT lock file should be updated to A
    """
    # Setup: create input file in state A
    input_file = tmp_path / "input.txt"
    input_file.write_text("state_A")

    def stage_func() -> None:
        content = (tmp_path / "input.txt").read_text()
        (tmp_path / "output.txt").write_text(f"processed: {content}")

    out = outputs.Out(str(tmp_path / "output.txt"), loader=loaders.PathOnly())
    stage_info = _make_stage_info(
        stage_func,
        tmp_path,
        fingerprint={"self:stage_func": "fp123"},
        deps=["input.txt"],
        outs=[out],
        run_id="run_1",
    )

    state_db_path = tmp_path / ".pivot" / "state.db"

    # Step 1: First run - creates lock file with state A
    result1 = executor.execute_stage("test_stage", stage_info, worker_env, output_queue)
    assert result1["status"] == "ran"
    assert (tmp_path / "output.txt").read_text() == "processed: state_A"
    _apply_deferred_writes("test_stage", stage_info, result1, state_db_path)

    # Read lock file to get hash of state A
    stage_lock = lock.StageLock("test_stage", lock.get_stages_dir(tmp_path / ".pivot"))
    lock_data_a = stage_lock.read()
    assert lock_data_a is not None
    hash_a = list(lock_data_a["dep_hashes"].values())[0]["hash"]

    # Step 2: Modify input to state B
    input_file.write_text("state_B")

    # Step 3: Run again - creates lock with state B, run cache now has A and B
    stage_info_run2 = _make_stage_info(
        stage_func,
        tmp_path,
        fingerprint={"self:stage_func": "fp123"},
        deps=["input.txt"],
        outs=[out],
        run_id="run_2",
    )
    result2 = executor.execute_stage("test_stage", stage_info_run2, worker_env, output_queue)
    assert result2["status"] == "ran"
    assert (tmp_path / "output.txt").read_text() == "processed: state_B"
    _apply_deferred_writes("test_stage", stage_info_run2, result2, state_db_path)

    # Read lock file to confirm state B
    lock_data_b = stage_lock.read()
    assert lock_data_b is not None
    hash_b = list(lock_data_b["dep_hashes"].values())[0]["hash"]
    assert hash_a != hash_b, "Hashes should differ for different states"

    # Step 4: Revert input to state A
    input_file.write_text("state_A")

    # Step 5: Run again - should skip via run cache AND update lock file
    stage_info_run3 = _make_stage_info(
        stage_func,
        tmp_path,
        fingerprint={"self:stage_func": "fp123"},
        deps=["input.txt"],
        outs=[out],
        run_id="run_3",
    )
    result3 = executor.execute_stage("test_stage", stage_info_run3, worker_env, output_queue)
    assert result3["status"] == "skipped"
    assert "run cache" in result3["reason"], f"Expected run cache skip, got: {result3['reason']}"

    # CRITICAL: Lock file should now have state A (not stale state B)
    lock_data_after = stage_lock.read()
    assert lock_data_after is not None
    hash_after = list(lock_data_after["dep_hashes"].values())[0]["hash"]
    assert hash_after == hash_a, (
        "Lock file should be updated to current state A after run cache skip"
    )


def test_explain_shows_cached_after_run_cache_skip(
    worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage], tmp_path: pathlib.Path
) -> None:
    """Explain module should show stage as unchanged after run cache skip.

    This verifies that `pivot status` will show cached after `pivot run`
    skips via run cache, since status uses explain internally.
    """
    from pivot import explain

    # Setup: create input file in state A
    input_file = tmp_path / "input.txt"
    input_file.write_text("state_A")

    def stage_func() -> None:
        content = (tmp_path / "input.txt").read_text()
        (tmp_path / "output.txt").write_text(f"processed: {content}")

    out = outputs.Out(str(tmp_path / "output.txt"), loader=loaders.PathOnly())
    stage_info = _make_stage_info(
        stage_func,
        tmp_path,
        fingerprint={"self:stage_func": "fp123"},
        deps=["input.txt"],
        outs=[out],
        run_id="run_1",
    )

    state_db_path = tmp_path / ".pivot" / "state.db"

    # Step 1: First run - creates lock file with state A
    result1 = executor.execute_stage("test_stage", stage_info, worker_env, output_queue)
    assert result1["status"] == "ran"
    _apply_deferred_writes("test_stage", stage_info, result1, state_db_path)

    # Step 2: Modify input to state B
    input_file.write_text("state_B")

    # Step 3: Run again - creates lock with state B
    stage_info_run2 = _make_stage_info(
        stage_func,
        tmp_path,
        fingerprint={"self:stage_func": "fp123"},
        deps=["input.txt"],
        outs=[out],
        run_id="run_2",
    )
    result2 = executor.execute_stage("test_stage", stage_info_run2, worker_env, output_queue)
    assert result2["status"] == "ran"
    _apply_deferred_writes("test_stage", stage_info_run2, result2, state_db_path)

    # Step 4: Revert input to state A
    input_file.write_text("state_A")

    # Step 5: Run again - should skip via run cache
    stage_info_run3 = _make_stage_info(
        stage_func,
        tmp_path,
        fingerprint={"self:stage_func": "fp123"},
        deps=["input.txt"],
        outs=[out],
        run_id="run_3",
    )
    result3 = executor.execute_stage("test_stage", stage_info_run3, worker_env, output_queue)
    assert result3["status"] == "skipped"
    assert "run cache" in result3["reason"]

    # CRITICAL: Explain should show stage as NOT needing to run
    explanation = explain.get_stage_explanation(
        stage_name="test_stage",
        fingerprint=stage_info["fingerprint"],
        deps=stage_info["deps"],
        outs_paths=[str(out.path)],
        params_instance=None,
        overrides=None,
        state_dir=tmp_path / ".pivot",
        force=False,
    )

    assert not explanation["will_run"], (
        f"Stage should not need to run after run cache skip, got: {explanation}"
    )
    assert explanation["reason"] == "", f"Reason should be empty, got: {explanation['reason']}"


def test_regular_execution_still_works(
    worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage], tmp_path: pathlib.Path
) -> None:
    """Regular execution should still work (lock file created with current state)."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("test data")

    def stage_func() -> None:
        (tmp_path / "output.txt").write_text("result")

    out = outputs.Out(str(tmp_path / "output.txt"), loader=loaders.PathOnly())
    stage_info = _make_stage_info(
        stage_func,
        tmp_path,
        fingerprint={"self:stage_func": "fp123"},
        deps=["input.txt"],
        outs=[out],
    )

    # First run - should execute and create lock file
    result = executor.execute_stage("test_stage", stage_info, worker_env, output_queue)
    assert result["status"] == "ran"
    assert (tmp_path / "output.txt").read_text() == "result"

    # Lock file should exist with current dep hashes
    stage_lock = lock.StageLock("test_stage", lock.get_stages_dir(tmp_path / ".pivot"))
    lock_data = stage_lock.read()
    assert lock_data is not None
    assert "dep_hashes" in lock_data
    assert len(lock_data["dep_hashes"]) == 1

    # Subsequent run should skip (via generation or hash check)
    result2 = executor.execute_stage("test_stage", stage_info, worker_env, output_queue)
    assert result2["status"] == "skipped"
    assert "unchanged" in result2["reason"]
