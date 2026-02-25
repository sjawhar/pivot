# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportMissingTypeArgument=false, reportAny=false, reportExplicitAny=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUntypedFunctionDecorator=false, reportUnusedParameter=false, reportUnusedCallResult=false
from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest

from pivot import loaders, types
from pivot.cli import helpers as cli_helpers
from pivot.cli import targets as cli_targets
from pivot.storage import store

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


# ---------------------------------------------------------------------------
# parse_identity_target
# ---------------------------------------------------------------------------


def test_parse_identity_target_single() -> None:
    result = cli_targets.parse_identity_target("train")
    assert result == types.ArtifactIdentity("train", None)


def test_parse_identity_target_with_key() -> None:
    result = cli_targets.parse_identity_target("train:model")
    assert result == types.ArtifactIdentity("train", "model")


# ---------------------------------------------------------------------------
# resolve_cli_target
# ---------------------------------------------------------------------------


def _make_stage_info(outs: list[types.ArtifactRef]) -> RegistryStageInfo:
    info: RegistryStageInfo = {  # type: ignore[assignment] - test helper, minimal shape
        "func": lambda: None,
        "name": "dummy",
        "deps": {},
        "outs": outs,
        "params": None,
        "mutex": [],
        "variant": None,
        "signature": None,
        "fingerprint": None,
        "params_arg_name": None,
        "state_dir": None,
        "collection_params": {},
    }
    return info


def test_resolve_cli_target_stage() -> None:
    """Registered stage name resolves to IdentityTarget with all outputs."""
    ref = _helper_ref("train", None, types.ArtifactTag.DATA, loaders.CSV())
    all_stages = {"train": _make_stage_info([ref])}

    result = cli_targets.resolve_cli_target("train", all_stages, lambda _: False)

    assert result["kind"] == "identity"
    assert result["stage_name"] == "train"
    assert result["identity"] == types.ArtifactIdentity("train", None)
    assert result["refs"] == [ref]


def test_resolve_cli_target_specific_key() -> None:
    """'train:model' resolves to IdentityTarget with only the matching output."""
    ref_model = _helper_ref("train", "model", types.ArtifactTag.DATA, loaders.CSV())
    ref_score = _helper_ref("train", "score", types.ArtifactTag.METRIC, loaders.JSON())
    all_stages = {"train": _make_stage_info([ref_model, ref_score])}

    result = cli_targets.resolve_cli_target("train:model", all_stages, lambda _: False)

    assert result["kind"] == "identity"
    assert result["stage_name"] == "train"
    assert result["identity"] == types.ArtifactIdentity("train", "model")
    assert result["refs"] == [ref_model]


def test_resolve_cli_target_pvt() -> None:
    """File path with .pvt sidecar resolves to PvtTarget."""
    result = cli_targets.resolve_cli_target(
        "data/external/raw.csv", {}, lambda path: path == "data/external/raw.csv"
    )

    assert result["kind"] == "pvt"
    assert result["path"] == "data/external/raw.csv"


def test_resolve_cli_target_unknown_raises() -> None:
    """Unknown target raises TargetValidationError."""
    with pytest.raises(cli_targets.TargetValidationError, match="not_a_stage"):
        cli_targets.resolve_cli_target("not_a_stage", {}, lambda _: False)


def test_resolve_cli_target_stage_with_bad_key_raises() -> None:
    """Stage exists but requested key doesn't match any output."""
    ref = _helper_ref("train", "model", types.ArtifactTag.DATA, loaders.CSV())
    all_stages = {"train": _make_stage_info([ref])}

    with pytest.raises(cli_targets.TargetValidationError, match="no_such_key"):
        cli_targets.resolve_cli_target("train:no_such_key", all_stages, lambda _: False)


# ---------------------------------------------------------------------------
# WorkspaceStore.resolve_display_path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tag", "key", "fmt", "expected"),
    [
        pytest.param(
            types.ArtifactTag.DATA,
            None,
            loaders.CSV(),
            "data/eval/train.csv",
            id="data-single",
        ),
        pytest.param(
            types.ArtifactTag.DATA,
            "model",
            loaders.CSV(),
            "data/eval/train/model.csv",
            id="data-multi",
        ),
        pytest.param(
            types.ArtifactTag.METRIC,
            "score",
            cast("loaders.Writer[object]", loaders.JSON[dict[str, object]]()),
            "metrics/eval/train/score.json",
            id="metric",
        ),
        pytest.param(
            types.ArtifactTag.PLOT,
            "loss",
            loaders.MatplotlibFigure(),
            "plots/eval/train/loss.png",
            id="plot",
        ),
        pytest.param(
            types.ArtifactTag.DIRECTORY,
            None,
            loaders.PathOnly(),
            "data/eval/train",
            id="directory",
        ),
    ],
)
def test_workspace_store_resolve_display_path(
    tmp_path: pathlib.Path,
    tag: types.ArtifactTag,
    key: str | None,
    fmt: loaders.Writer[object] | loaders.Reader[object] | loaders.Loader[object, object],
    expected: str,
) -> None:
    workspace_store = store.WorkspaceStore(
        project_root=tmp_path, pipeline_name="eval", input_bindings={}
    )
    ref = _helper_ref("train", key, tag, fmt)

    resolved = workspace_store.resolve_display_path(ref)
    assert resolved == tmp_path / expected


# ---------------------------------------------------------------------------
# get_workspace_store
# ---------------------------------------------------------------------------


def test_get_workspace_store_with_pipeline(
    mock_discovery: Pipeline,
) -> None:
    """Returns WorkspaceStore when pipeline is in context."""
    ws = cli_helpers.get_workspace_store()
    assert ws is not None
    assert isinstance(ws, store.WorkspaceStore)


def test_get_workspace_store_no_pipeline(
    mocker: pytest.MonkeyPatch,
) -> None:
    """Returns None when no pipeline is in context."""
    from pivot.cli import decorators as cli_decorators

    with patch.object(cli_decorators, "get_pipeline_from_context", return_value=None):
        ws = cli_helpers.get_workspace_store()

    assert ws is None


def test_get_workspace_store_pipeline_name(
    mock_discovery: Pipeline,
) -> None:
    """WorkspaceStore uses the pipeline name for path construction."""
    ws = cli_helpers.get_workspace_store()
    assert ws is not None

    # Verify the store resolves paths using the pipeline name ("test" from mock_discovery)
    ref = _helper_ref("stage", None, types.ArtifactTag.DATA, loaders.CSV())
    path = ws.resolve_display_path(ref)
    # mock_discovery pipeline name is "test" (from conftest test_pipeline)
    assert "test" in path.parts, f"Expected 'test' in path parts: {path}"
