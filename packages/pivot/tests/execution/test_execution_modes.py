"""Tests for execution modes: --no-commit and run cache."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING, Any

from pivot import loaders, types
from pivot.executor import worker
from pivot.storage import cache, lock
from pivot.storage import store as store_mod
from pivot.types import StageStatus

if TYPE_CHECKING:
    import multiprocessing as mp
    import pathlib
    from collections.abc import Callable

    from pivot.types import OutputMessage


def _make_artifact_ref(
    producer: str,
    key: str | None,
    *,
    tag: types.ArtifactTag,
    loader: loaders.Reader[object] | loaders.Writer[object] | loaders.Loader[object, object],
    python_type: type,
) -> types.ArtifactRef:
    return types.ArtifactRef(
        identity=types.ArtifactIdentity(producer=producer, key=key),
        format=loader,
        python_type=python_type,
        tag=tag,
    )


def _artifact_key(ref: types.ArtifactRef) -> str:
    return f"{ref.identity.producer}:{ref.identity.key}"


def _make_stage_info(
    func: Callable[..., Any],
    tmp_path: pathlib.Path,
    *,
    deps: dict[str, types.ArtifactRef] | None = None,
    outs: list[types.ArtifactRef] | None = None,
    fingerprint: dict[str, str] | None = None,
    run_id: str = "test_run",
    no_commit: bool = False,
    force: bool = False,
    input_bindings: dict[str, str] | None = None,
) -> worker.WorkerStageInfo:
    """Create a WorkerStageInfo for testing."""
    return worker.WorkerStageInfo(
        func=func,
        fingerprint=fingerprint or {"self:test": "abc123"},
        deps=deps or {},
        signature=None,
        outs=outs or [],
        store_spec={
            "kind": "workspace",
            "cache_dir": str(tmp_path / ".pivot" / "cache"),
            "project_root": str(tmp_path),
            "pipeline_name": "test",
            "input_bindings": input_bindings or {},
        },
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
        params_arg_name=None,
        project_root=tmp_path,
        state_dir=tmp_path / ".pivot",
        collection_params={},
    )


# -----------------------------------------------------------------------------
# No-commit mode tests
# -----------------------------------------------------------------------------


def test_no_commit_produces_outputs_without_production_lock(
    worker_env: pathlib.Path, tmp_path: pathlib.Path, output_queue: mp.Queue[OutputMessage]
) -> None:
    """When no_commit=True, outputs exist on disk but no production lock is written."""
    (tmp_path / "input.txt").write_text("input data")

    def stage_func(data: str) -> str:
        _ = data
        return "output data"

    input_ref = _make_artifact_ref(
        "input", None, tag=types.ArtifactTag.DATA, loader=loaders.Text(), python_type=str
    )
    output_ref = _make_artifact_ref(
        "test_stage", None, tag=types.ArtifactTag.DATA, loader=loaders.Text(), python_type=str
    )

    stage_info = _make_stage_info(
        stage_func,
        tmp_path,
        deps={"data": input_ref},
        outs=[output_ref],
        input_bindings={"input": "input.txt"},
        no_commit=True,
    )

    result = worker.execute_stage("test_stage", stage_info, worker_env, output_queue)

    assert result["status"] == StageStatus.RAN

    output_path = tmp_path / "data" / "test_stage.txt"
    assert output_path.exists(), "Output should exist"

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

    def stage_func(data: str) -> str:
        _ = data
        return "output data"

    input_ref = _make_artifact_ref(
        "input", None, tag=types.ArtifactTag.DATA, loader=loaders.Text(), python_type=str
    )
    output_ref = _make_artifact_ref(
        "test_stage", None, tag=types.ArtifactTag.DATA, loader=loaders.Text(), python_type=str
    )

    stage_info = _make_stage_info(
        stage_func,
        tmp_path,
        deps={"data": input_ref},
        outs=[output_ref],
        input_bindings={"input": "input.txt"},
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

    def stage_func(data: str) -> str:
        execution_count[0] += 1
        _ = data
        return "output data"

    input_ref = _make_artifact_ref(
        "input", None, tag=types.ArtifactTag.DATA, loader=loaders.Text(), python_type=str
    )
    output_ref = _make_artifact_ref(
        "test_stage", None, tag=types.ArtifactTag.DATA, loader=loaders.Text(), python_type=str
    )

    # First run with no_commit
    stage_info_nc = _make_stage_info(
        stage_func,
        tmp_path,
        deps={"data": input_ref},
        outs=[output_ref],
        input_bindings={"input": "input.txt"},
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
        deps={"data": input_ref},
        outs=[output_ref],
        input_bindings={"input": "input.txt"},
        no_commit=False,
    )
    result2 = worker.execute_stage("test_stage", stage_info_commit, worker_env, output_queue)
    assert result2["status"] == StageStatus.RAN, "Should re-run since no lock exists"
    assert execution_count[0] == 2, "Stage should execute again"

    # Lock should now exist
    assert production_lock.path.exists(), "Lock should exist after normal run"

    output_path = tmp_path / "data" / "test_stage.txt"
    assert output_path.exists(), "Output should exist after normal run"


# -----------------------------------------------------------------------------
# Run cache directory output tests
# -----------------------------------------------------------------------------


def test_run_cache_restores_directory_output(
    worker_env: pathlib.Path, tmp_path: pathlib.Path, output_queue: mp.Queue[OutputMessage]
) -> None:
    """Run cache should restore directory outputs including manifest."""
    (tmp_path / "input.txt").write_text("input data")

    execution_count = [0]

    def stage_func(data: str) -> dict[str, str]:
        execution_count[0] += 1
        _ = data
        return {"file1.txt": "content1", "file2.txt": "content2"}

    input_ref = _make_artifact_ref(
        "input", None, tag=types.ArtifactTag.DATA, loader=loaders.Text(), python_type=str
    )
    output_ref = _make_artifact_ref(
        "test_stage",
        None,
        tag=types.ArtifactTag.DIRECTORY,
        loader=loaders.Text(),
        python_type=dict,
    )

    stage_info = _make_stage_info(
        stage_func,
        tmp_path,
        deps={"data": input_ref},
        outs=[output_ref],
        input_bindings={"input": "input.txt"},
    )

    # First run - should execute and write to run cache
    result1 = worker.execute_stage("test_stage", stage_info, worker_env, output_queue)
    assert result1["status"] == StageStatus.RAN
    assert execution_count[0] == 1

    store = store_mod.store_from_spec(stage_info["store_spec"])
    output_dir = store.checkout(output_ref)
    assert output_dir.is_dir()
    assert (output_dir / "file1.txt").read_text() == "content1"
    assert (output_dir / "file2.txt").read_text() == "content2"

    shutil.rmtree(output_dir)
    assert not output_dir.exists()

    result2 = worker.execute_stage("test_stage", stage_info, worker_env, output_queue)
    assert result2["status"] == StageStatus.CACHED
    assert execution_count[0] == 1

    assert output_dir.is_dir(), "Directory should be recreated"
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

    def stage_func(data: str) -> dict[str, object]:
        execution_count[0] += 1
        _ = data
        return {"output": "output data", "metrics": {"accuracy": 0.95}}

    input_ref = _make_artifact_ref(
        "input", None, tag=types.ArtifactTag.DATA, loader=loaders.Text(), python_type=str
    )
    output_ref = _make_artifact_ref(
        "test_stage", "output", tag=types.ArtifactTag.DATA, loader=loaders.Text(), python_type=str
    )
    metric_ref = _make_artifact_ref(
        "test_stage",
        "metrics",
        tag=types.ArtifactTag.METRIC,
        loader=loaders.JSON(),
        python_type=dict,
    )

    # Stage with both cached (Out) and non-cached (Metric) outputs
    stage_info = _make_stage_info(
        stage_func,
        tmp_path,
        deps={"data": input_ref},
        outs=[output_ref, metric_ref],
        input_bindings={"input": "input.txt"},
    )

    # First run - should execute and write to run cache
    result1 = worker.execute_stage("test_stage", stage_info, worker_env, output_queue)
    assert result1["status"] == StageStatus.RAN
    assert execution_count[0] == 1

    store = store_mod.store_from_spec(stage_info["store_spec"])
    output_file = store.checkout(output_ref)
    metric_file = store.checkout(metric_ref)
    assert output_file.exists()
    assert metric_file.exists()

    metric_file.unlink()
    assert not metric_file.exists()
    assert output_file.exists(), "Cached output should still exist"

    result2 = worker.execute_stage("test_stage", stage_info, worker_env, output_queue)
    assert result2["status"] == StageStatus.RAN
    assert execution_count[0] == 2

    assert metric_file.exists(), "Metric file should be recreated"
