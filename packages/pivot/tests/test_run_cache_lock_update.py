from __future__ import annotations

from typing import TYPE_CHECKING

from pivot import executor, loaders
from pivot.storage import cache, lock, state
from pivot.storage import store as store_mod
from pivot.types import ArtifactIdentity, ArtifactRef, ArtifactTag

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
    output_paths = [_identity_key(out) for out in stage_info["outs"]]
    with state.StateDB(state_db_path) as db:
        db.apply_deferred_writes(stage_name, output_paths, result["deferred_writes"])


def _identity_key(ref: ArtifactRef) -> str:
    if ref.identity.key is None:
        return ref.identity.producer
    return f"{ref.identity.producer}:{ref.identity.key}"


def _input_ref(name: str) -> ArtifactRef:
    return ArtifactRef(
        identity=ArtifactIdentity(name, None),
        format=loaders.Text(),
        python_type=str,
        tag=ArtifactTag.DATA,
    )


def _output_ref(stage_name: str) -> ArtifactRef:
    return ArtifactRef(
        identity=ArtifactIdentity(stage_name, None),
        format=loaders.Text(),
        python_type=str,
        tag=ArtifactTag.DATA,
    )


def _output_path(ref: ArtifactRef, tmp_path: pathlib.Path) -> pathlib.Path:
    store = store_mod.WorkspaceStore(
        project_root=tmp_path,
        pipeline_name="test",
        input_bindings={},
    )
    return store.prepare_output(ref)


def _make_stage_info(
    func: Callable[..., Any],
    tmp_path: pathlib.Path,
    *,
    fingerprint: dict[str, str] | None = None,
    deps: dict[str, ArtifactRef] | None = None,
    outs: list[ArtifactRef] | None = None,
    run_id: str = "test_run",
) -> WorkerStageInfo:
    """Create a WorkerStageInfo with sensible defaults for testing."""
    deps = deps or {}
    outs = outs or []
    return {
        "func": func,
        "fingerprint": fingerprint or {"self:test": "abc123"},
        "deps": deps,
        "signature": None,
        "outs": outs,
        "store_spec": {
            "kind": "workspace",
            "cache_dir": str(tmp_path / ".pivot" / "cache"),
            "project_root": str(tmp_path),
            "pipeline_name": "test",
            "input_bindings": {
                ref.identity.producer: ref.identity.producer for ref in deps.values()
            },
        },
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

    def stage_func(_0: str) -> str:
        content = (tmp_path / "input.txt").read_text()
        output = f"processed: {content}"
        (tmp_path / "output.txt").write_text(output)
        return output

    stage_info = _make_stage_info(
        stage_func,
        tmp_path,
        fingerprint={"self:stage_func": "fp123"},
        deps={"_0": _input_ref("input.txt")},
        outs=[_output_ref("test_stage")],
        run_id="run_1",
    )

    state_db_path = tmp_path / ".pivot" / "state.db"

    # Step 1: First run - creates lock file with state A, output gen -> 1
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
        deps={"_0": _input_ref("input.txt")},
        outs=[_output_ref("test_stage")],
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
        deps={"_0": _input_ref("input.txt")},
        outs=[_output_ref("test_stage")],
        run_id="run_3",
    )
    result3 = executor.execute_stage("test_stage", stage_info_run3, worker_env, output_queue)
    assert result3["status"] == "cached"
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

    def stage_func(_0: str) -> str:
        content = (tmp_path / "input.txt").read_text()
        output = f"processed: {content}"
        (tmp_path / "output.txt").write_text(output)
        return output

    stage_info = _make_stage_info(
        stage_func,
        tmp_path,
        fingerprint={"self:stage_func": "fp123"},
        deps={"_0": _input_ref("input.txt")},
        outs=[_output_ref("test_stage")],
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
        deps={"_0": _input_ref("input.txt")},
        outs=[_output_ref("test_stage")],
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
        deps={"_0": _input_ref("input.txt")},
        outs=[_output_ref("test_stage")],
        run_id="run_3",
    )
    result3 = executor.execute_stage("test_stage", stage_info_run3, worker_env, output_queue)
    assert result3["status"] == "cached"
    assert "run cache" in result3["reason"]

    # CRITICAL: Explain should show stage as NOT needing to run
    explanation = explain.get_stage_explanation(
        stage_name="test_stage",
        fingerprint=stage_info["fingerprint"],
        deps=["input.txt"],
        outs_paths=[_identity_key(_output_ref("test_stage"))],
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

    def stage_func(_0: str) -> str:
        result = "result"
        (tmp_path / "output.txt").write_text(result)
        return result

    stage_info = _make_stage_info(
        stage_func,
        tmp_path,
        fingerprint={"self:stage_func": "fp123"},
        deps={"_0": _input_ref("input.txt")},
        outs=[_output_ref("test_stage")],
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
    assert result2["status"] == "cached"
    assert "unchanged" in result2["reason"]


def test_run_cache_skip_does_not_increment_output_generations(
    worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage], tmp_path: pathlib.Path
) -> None:
    """Run cache skip should NOT increment output generations.

    This is the critical integration test for the increment_outputs flag:
    1. First run: outputs get generation 1
    2. Modify dep, second run: outputs get generation 2
    3. Revert dep, third run: skips via run cache, outputs stay at generation 2

    If output generations were wrongly incremented on run cache skip, downstream
    stages would see a generation bump and think they need to re-run even though
    the outputs are identical to what they already processed.
    """
    input_file = tmp_path / "input.txt"
    input_file.write_text("state_A")

    def stage_func(_0: str) -> str:
        content = (tmp_path / "input.txt").read_text()
        output = f"processed: {content}"
        (tmp_path / "output.txt").write_text(output)
        return output

    stage_info = _make_stage_info(
        stage_func,
        tmp_path,
        fingerprint={"self:stage_func": "fp123"},
        deps={"_0": _input_ref("input.txt")},
        outs=[_output_ref("test_stage")],
        run_id="run_1",
    )

    state_db_path = tmp_path / ".pivot" / "state.db"

    # Step 1: First run - creates lock file with state A
    result1 = executor.execute_stage("test_stage", stage_info, worker_env, output_queue)
    assert result1["status"] == "ran"
    _apply_deferred_writes("test_stage", stage_info, result1, state_db_path)

    with state.StateDB(state_db_path) as db:
        gen_after_run1 = db.get_generation(_identity_key(_output_ref("test_stage")))
    assert gen_after_run1 == 1, f"Expected generation 1 after first run, got {gen_after_run1}"

    # Step 2: Modify input to state B
    input_file.write_text("state_B")

    # Step 3: Run again - output gen -> 2
    stage_info_run2 = _make_stage_info(
        stage_func,
        tmp_path,
        fingerprint={"self:stage_func": "fp123"},
        deps={"_0": _input_ref("input.txt")},
        outs=[_output_ref("test_stage")],
        run_id="run_2",
    )
    result2 = executor.execute_stage("test_stage", stage_info_run2, worker_env, output_queue)
    assert result2["status"] == "ran"
    _apply_deferred_writes("test_stage", stage_info_run2, result2, state_db_path)

    with state.StateDB(state_db_path) as db:
        gen_after_run2 = db.get_generation(_identity_key(_output_ref("test_stage")))
    assert gen_after_run2 == 2, f"Expected generation 2 after second run, got {gen_after_run2}"

    # Step 4: Revert input to state A
    input_file.write_text("state_A")

    # Step 5: Run again - should skip via run cache
    stage_info_run3 = _make_stage_info(
        stage_func,
        tmp_path,
        fingerprint={"self:stage_func": "fp123"},
        deps={"_0": _input_ref("input.txt")},
        outs=[_output_ref("test_stage")],
        run_id="run_3",
    )
    result3 = executor.execute_stage("test_stage", stage_info_run3, worker_env, output_queue)
    assert result3["status"] == "cached"
    assert "run cache" in result3["reason"], f"Expected run cache skip, got: {result3['reason']}"

    # Apply deferred writes from the SKIPPED result
    _apply_deferred_writes("test_stage", stage_info_run3, result3, state_db_path)

    # CRITICAL: Output generation should NOT have been incremented
    with state.StateDB(state_db_path) as db:
        gen_after_skip = db.get_generation(_identity_key(_output_ref("test_stage")))
    assert gen_after_skip == 2, (
        f"Output generation should remain at 2 after run cache skip, got {gen_after_skip}. "
        "Run cache skip should not increment output generations since outputs are restored, not rewritten."
    )
