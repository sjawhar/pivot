# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportMissingTypeArgument=false, reportAny=false, reportExplicitAny=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUntypedFunctionDecorator=false, reportUnusedParameter=false, reportUnusedCallResult=false
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import pytest

from pivot import loaders, types
from pivot.engine import engine as engine_mod
from pivot.registry import RegistryStageInfo
from pivot.storage import lock
from pivot.storage import store as store_mod

if TYPE_CHECKING:
    import pathlib

    from pivot.executor import core as executor_core
    from pivot.pipeline import pipeline as pipeline_mod


def _helper_stage_func() -> None:
    return None


def _helper_artifact_ref(stage_name: str, key: str, tag: types.ArtifactTag) -> types.ArtifactRef:
    return types.ArtifactRef(
        identity=types.ArtifactIdentity(stage_name, key),
        format=loaders.Text(),
        python_type=str,
        tag=tag,
    )


def _helper_stage_info(
    stage_name: str,
    out_refs: list[types.ArtifactRef],
    *,
    fingerprint: dict[str, str],
) -> RegistryStageInfo:
    return RegistryStageInfo(
        func=_helper_stage_func,
        name=stage_name,
        deps={},
        outs=out_refs,
        params=None,
        mutex=[],
        variant=None,
        signature=inspect.signature(_helper_stage_func),
        fingerprint=fingerprint,
        params_arg_name=None,
        state_dir=None,
        collection_params={},
    )


def _helper_write_lock(
    state_dir: pathlib.Path,
    stage_name: str,
    out_ref: types.ArtifactRef,
    fingerprint: dict[str, str],
) -> None:
    stages_dir = lock.get_stages_dir(state_dir)
    stage_lock = lock.StageLock(stage_name, stages_dir)
    stage_lock.write(
        types.LockData(
            code_manifest=fingerprint,
            params={},
            dep_hashes={},
            output_hashes={out_ref.identity: types.FileHash(hash="abc123")},
            merkle_id=None,
        )
    )


@pytest.mark.anyio
async def test_compute_output_summary_reads_lock_hashes(
    test_pipeline: pipeline_mod.Pipeline,
) -> None:
    stage_name = "train"
    out_ref = _helper_artifact_ref(stage_name, "model", types.ArtifactTag.DATA)
    stage_info = _helper_stage_info(stage_name, [out_ref], fingerprint={"self": "h1"})
    test_pipeline._registry.add_existing(stage_info)

    _helper_write_lock(test_pipeline.state_dir, stage_name, out_ref, {"self": "h1"})

    async with engine_mod.Engine(pipeline=test_pipeline) as eng:
        summary = await eng._compute_output_summary(stage_name, test_pipeline.state_dir)

    assert summary is not None
    assert len(summary) == 1
    assert summary[0]["path"] == "train:model"
    assert summary[0]["output_type"] == "out"
    assert summary[0]["new_hash"] == "abc123"


@pytest.mark.anyio
async def test_compute_output_summary_with_missing_lock_uses_none_hash(
    test_pipeline: pipeline_mod.Pipeline,
) -> None:
    stage_name = "eval"
    out_ref = _helper_artifact_ref(stage_name, "metrics", types.ArtifactTag.METRIC)
    stage_info = _helper_stage_info(stage_name, [out_ref], fingerprint={"self": "h2"})
    test_pipeline._registry.add_existing(stage_info)

    async with engine_mod.Engine(pipeline=test_pipeline) as eng:
        summary = await eng._compute_output_summary(stage_name, test_pipeline.state_dir)

    assert summary is not None
    assert summary[0]["path"] == "eval:metrics"
    assert summary[0]["output_type"] == "metric"
    assert summary[0]["new_hash"] is None


@pytest.mark.anyio
async def test_try_skip_in_coordinator_returns_false_when_lock_missing(
    test_pipeline: pipeline_mod.Pipeline,
    tmp_path: pathlib.Path,
) -> None:
    stage_name = "prepare"
    out_ref = _helper_artifact_ref(stage_name, "out", types.ArtifactTag.DATA)
    stage_info = _helper_stage_info(stage_name, [out_ref], fingerprint={"self": "h3"})
    test_pipeline._registry.add_existing(stage_info)

    cache_dir = tmp_path / ".pivot" / "cache"
    (cache_dir / "files").mkdir(parents=True)
    results: dict[str, executor_core.ExecutionSummary] = {}

    async with engine_mod.Engine(pipeline=test_pipeline) as eng:
        skipped = await eng._try_skip_in_coordinator(
            stage_name,
            stage_info,
            overrides={},
            force=False,
            cache_dir=cache_dir,
            state_dir=test_pipeline.state_dir,
            project_root=test_pipeline.root,
            results=results,
            run_id="run-1",
        )

    assert skipped is False
    assert results == {}


@pytest.mark.anyio
async def test_try_skip_in_coordinator_marks_stage_cached_on_generation_match(
    test_pipeline: pipeline_mod.Pipeline,
    tmp_path: pathlib.Path,
) -> None:
    stage_name = "predict"
    fingerprint = {"self": "stable"}
    out_ref = _helper_artifact_ref(stage_name, "predictions", types.ArtifactTag.DATA)
    stage_info = _helper_stage_info(stage_name, [out_ref], fingerprint=fingerprint)
    test_pipeline._registry.add_existing(stage_info)

    cache_dir = tmp_path / ".pivot" / "cache"
    (cache_dir / "files").mkdir(parents=True)

    workspace_store = store_mod.WorkspaceStore(
        project_root=test_pipeline.root,
        pipeline_name=test_pipeline.name,
        input_bindings=test_pipeline.input_bindings,
    )
    output_path = workspace_store.resolve_display_path(out_ref)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("cached-output")

    _helper_write_lock(test_pipeline.state_dir, stage_name, out_ref, fingerprint)

    results: dict[str, executor_core.ExecutionSummary] = {}
    async with engine_mod.Engine(pipeline=test_pipeline) as eng:
        skipped = await eng._try_skip_in_coordinator(
            stage_name,
            stage_info,
            overrides={},
            force=False,
            cache_dir=cache_dir,
            state_dir=test_pipeline.state_dir,
            project_root=test_pipeline.root,
            results=results,
            run_id="run-2",
        )

    assert skipped is True
    assert stage_name in results
    stage_result = results[stage_name]
    assert stage_result["status"] == types.StageStatus.CACHED
    assert stage_result["reason"] == "unchanged (generation)"
