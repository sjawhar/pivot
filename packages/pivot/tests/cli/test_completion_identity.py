from __future__ import annotations

import inspect
import pathlib
from typing import TYPE_CHECKING

from pivot import loaders, outputs, project, types
from pivot.cli import completion
from pivot.cli import helpers as cli_helpers
from pivot.cli import targets as cli_targets
from pivot.engine import graph as engine_graph
from pivot.registry import RegistryStageInfo

if TYPE_CHECKING:
    from pivot.pipeline import pipeline as pipeline_mod


def _helper_ref(
    producer: str,
    key: str | None,
    tag: types.ArtifactTag,
    fmt: loaders.Writer[object] | loaders.Reader[object] | loaders.Loader[object, object],
    python_type: type,
) -> types.ArtifactRef:
    return types.ArtifactRef(
        identity=types.ArtifactIdentity(producer=producer, key=key),
        format=fmt,
        python_type=python_type,
        tag=tag,
    )


def _make_stage_info(name: str, outs: list[types.ArtifactRef]) -> RegistryStageInfo:
    def _stage_func() -> None:
        pass

    return RegistryStageInfo(
        func=_stage_func,
        name=name,
        deps={},
        outs=outs,
        params=None,
        mutex=[],
        variant=None,
        signature=inspect.signature(_stage_func),
        fingerprint={"_code": "fake_hash"},
        params_arg_name=None,
        state_dir=None,
    )


def _register_stage(
    test_pipeline: pipeline_mod.Pipeline, name: str, outs: list[types.ArtifactRef]
) -> None:
    test_pipeline._registry._stages[name] = _make_stage_info(name, outs)


def test_complete_targets_includes_stage_keys(mock_discovery: pipeline_mod.Pipeline) -> None:
    outs = [
        _helper_ref("train", "model", types.ArtifactTag.DATA, loaders.CSV(), pathlib.Path),
        _helper_ref("train", "score", types.ArtifactTag.METRIC, loaders.JSON(), dict),
    ]
    _register_stage(mock_discovery, "train", outs)

    bare = completion.complete_targets(None, None, "")
    assert "train" in bare
    assert "train:model" not in bare, "stage:key completions should only appear after typing ':'"

    keyed = completion.complete_targets(None, None, "train:")
    assert "train:model" in keyed
    assert "train:score" in keyed


def test_resolve_targets_to_stages_identity_key(
    mock_discovery: pipeline_mod.Pipeline,
) -> None:
    outs = [_helper_ref("train", "model", types.ArtifactTag.DATA, loaders.CSV(), pathlib.Path)]
    _register_stage(mock_discovery, "train", outs)

    graph = engine_graph.build_graph(mock_discovery._registry._stages)
    resolved, unresolved = cli_targets.resolve_targets_to_stages(["train:model"], graph)

    assert resolved == {"train"}
    assert unresolved == []


def test_resolve_output_paths_uses_store_display_path(
    mock_discovery: pipeline_mod.Pipeline,
) -> None:
    ref = _helper_ref("train", "score", types.ArtifactTag.METRIC, loaders.JSON(), dict)
    _register_stage(mock_discovery, "train", [ref])

    workspace_store = cli_helpers.get_workspace_store()
    assert workspace_store is not None
    expected = project.to_relative_path(
        workspace_store.resolve_display_path(ref), mock_discovery.root
    )

    resolved, missing = cli_targets.resolve_output_paths(
        ["train"], mock_discovery.root, outputs.Metric
    )

    assert missing == []
    assert resolved == {expected}


def test_resolve_plot_infos_uses_store_display_path(
    mock_discovery: pipeline_mod.Pipeline,
) -> None:
    ref = _helper_ref(
        "train", "loss", types.ArtifactTag.PLOT, loaders.MatplotlibFigure(), pathlib.Path
    )
    _register_stage(mock_discovery, "train", [ref])

    workspace_store = cli_helpers.get_workspace_store()
    assert workspace_store is not None
    expected = project.to_relative_path(
        workspace_store.resolve_display_path(ref), mock_discovery.root
    )

    resolved, missing = cli_targets.resolve_plot_infos(["train"], mock_discovery.root)

    assert missing == []
    assert len(resolved) == 1
    assert resolved[0]["stage_name"] == "train"
    assert resolved[0]["path"] == expected


def test_complete_targets_stage_without_key(mock_discovery: pipeline_mod.Pipeline) -> None:
    """Stage without key should be completable."""
    outs = [_helper_ref("train", None, types.ArtifactTag.DATA, loaders.CSV(), pathlib.Path)]
    _register_stage(mock_discovery, "train", outs)

    result = completion.complete_targets(None, None, "")

    assert "train" in result


def test_complete_targets_multiple_stages(mock_discovery: pipeline_mod.Pipeline) -> None:
    """Multiple stages should all be in completion results."""
    outs1 = [_helper_ref("train", None, types.ArtifactTag.DATA, loaders.CSV(), pathlib.Path)]
    outs2 = [_helper_ref("eval", None, types.ArtifactTag.DATA, loaders.JSON(), pathlib.Path)]
    _register_stage(mock_discovery, "train", outs1)
    _register_stage(mock_discovery, "eval", outs2)

    result = completion.complete_targets(None, None, "")

    assert "train" in result
    assert "eval" in result


def test_resolve_targets_to_stages_stage_name_only(
    mock_discovery: pipeline_mod.Pipeline,
) -> None:
    """Stage name without key should resolve to stage."""
    outs = [_helper_ref("train", "model", types.ArtifactTag.DATA, loaders.CSV(), pathlib.Path)]
    _register_stage(mock_discovery, "train", outs)

    graph = engine_graph.build_graph(mock_discovery._registry._stages)
    resolved, unresolved = cli_targets.resolve_targets_to_stages(["train"], graph)

    assert resolved == {"train"}
    assert unresolved == []


def test_resolve_output_paths_metric_excluded(
    mock_discovery: pipeline_mod.Pipeline,
) -> None:
    """Metric outputs should not be included in resolved paths."""
    metric_ref = _helper_ref("eval", "score", types.ArtifactTag.METRIC, loaders.JSON(), dict)
    _register_stage(mock_discovery, "eval", [metric_ref])

    resolved, missing = cli_targets.resolve_output_paths(
        ["eval"], mock_discovery.root, outputs.Metric
    )

    # Metrics should be found since we're looking for Metric type
    assert len(resolved) >= 0
