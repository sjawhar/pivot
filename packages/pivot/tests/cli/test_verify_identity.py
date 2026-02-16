# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportMissingTypeArgument=false, reportAny=false, reportExplicitAny=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUntypedFunctionDecorator=false, reportUnusedParameter=false, reportUnusedCallResult=false
from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import pytest

from pivot import config, loaders, project, types
from pivot.cli import helpers as cli_helpers
from pivot.cli import remote as cli_remote
from pivot.cli import verify as cli_verify
from pivot.storage import lock

if TYPE_CHECKING:
    from pivot.pipeline.pipeline import Pipeline
    from pivot.registry import RegistryStageInfo


def _helper_ref(
    producer: str,
    key: str | None,
    tag: types.ArtifactTag,
    fmt: loaders.Writer[object] | loaders.Reader[object] | loaders.Loader[object, object],
) -> types.ArtifactRef:
    return types.ArtifactRef(
        identity=types.ArtifactIdentity(producer=producer, key=key),
        format=fmt,
        python_type=pathlib.Path,
        tag=tag,
    )


def _make_stage_info(
    name: str,
    outs: list[types.ArtifactRef],
    *,
    state_dir: pathlib.Path | None = None,
) -> RegistryStageInfo:
    info: RegistryStageInfo = {  # type: ignore[assignment] - test helper
        "func": lambda: None,
        "name": name,
        "deps": {},
        "outs": outs,
        "params": None,
        "mutex": [],
        "variant": None,
        "signature": None,
        "fingerprint": None,
        "params_arg_name": None,
        "state_dir": state_dir,
    }
    return info


def _write_stage_lock(
    stage_name: str,
    state_dir: pathlib.Path,
    *,
    output_hashes: dict[types.ArtifactIdentity, types.HashInfo],
) -> None:
    stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(state_dir))
    stage_lock.write(
        types.LockData(
            code_manifest={},
            params={},
            dep_hashes={},
            output_hashes=output_hashes,
        )
    )


def test_get_stage_lock_hashes_uses_identity_paths(
    mock_discovery: Pipeline,
) -> None:
    data_ref = _helper_ref("train", "model", types.ArtifactTag.DATA, loaders.CSV())
    metric_ref = _helper_ref("train", "metrics", types.ArtifactTag.METRIC, loaders.JSON())
    stage_info = _make_stage_info("train", [data_ref, metric_ref])
    mock_discovery._registry.add_existing(stage_info)  # type: ignore[reportPrivateUsage]

    state_dir = config.get_state_dir()
    _write_stage_lock(
        "train",
        state_dir,
        output_hashes={
            types.ArtifactIdentity("train", "model"): types.FileHash(hash="abc123"),
            types.ArtifactIdentity("train", "metrics"): types.FileHash(hash="def456"),
        },
    )

    output_hashes, dep_hashes = cli_verify._get_stage_lock_hashes("train")

    workspace_store = cli_helpers.get_workspace_store()
    assert workspace_store is not None
    expected_path = project.to_relative_path(workspace_store.resolve_display_path(data_ref))
    assert output_hashes == {expected_path: "abc123"}
    assert dep_hashes == {}


def test_get_stage_lock_hashes_merged_pipeline_paths(
    mock_discovery: Pipeline,
    tmp_path: pathlib.Path,
) -> None:
    stage_name = "alpha/train"
    data_ref = _helper_ref(stage_name, "model", types.ArtifactTag.DATA, loaders.CSV())
    state_dir = tmp_path / "alpha" / ".pivot"
    state_dir.mkdir(parents=True)
    stage_info = _make_stage_info(stage_name, [data_ref], state_dir=state_dir)
    mock_discovery._registry.add_existing(stage_info)  # type: ignore[reportPrivateUsage]

    _write_stage_lock(
        stage_name,
        state_dir,
        output_hashes={
            types.ArtifactIdentity(stage_name, "model"): types.FileHash(hash="abc123"),
        },
    )

    output_hashes, dep_hashes = cli_verify._get_stage_lock_hashes(stage_name)

    workspace_store = cli_helpers.get_workspace_store()
    assert workspace_store is not None
    expected_path = project.to_relative_path(workspace_store.resolve_display_path(data_ref))
    assert output_hashes == {expected_path: "abc123"}
    assert dep_hashes == {}


def test_normalize_cli_targets_preserves_identity_targets(
    set_project_root: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(set_project_root)
    targets = ("train", "train:model")
    normalized = cli_remote._normalize_cli_targets(targets, known_stages={"train"})

    assert normalized == targets
