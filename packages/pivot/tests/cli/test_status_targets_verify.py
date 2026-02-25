# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportMissingTypeArgument=false, reportAny=false, reportExplicitAny=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUntypedFunctionDecorator=false, reportUnusedParameter=false, reportUnusedCallResult=false
from __future__ import annotations

import inspect
import json
import pathlib

import click

from pivot import cli, loaders, outputs, types
from pivot.cli import targets as targets_mod
from pivot.registry import RegistryStageInfo


def _helper_ref(stage_name: str, key: str | None) -> types.ArtifactRef:
    return types.ArtifactRef(
        identity=types.ArtifactIdentity(stage_name, key),
        format=loaders.Text(),
        python_type=pathlib.Path,
        tag=types.ArtifactTag.DATA,
    )


def _helper_stage_info(stage_name: str, outs: list[types.ArtifactRef]) -> RegistryStageInfo:
    def _stage() -> None:
        return None

    return RegistryStageInfo(
        func=_stage,
        name=stage_name,
        deps={},
        outs=outs,
        params=None,
        mutex=[],
        variant=None,
        signature=inspect.signature(_stage),
        fingerprint={"self": "x"},
        params_arg_name=None,
        state_dir=None,
        collection_params={},
    )


def test_status_json_stages_only_returns_payload(mock_discovery, runner) -> None:
    result = runner.invoke(cli.cli, ["status", "--json", "--stages-only"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert "stages" in payload
    assert isinstance(payload["stages"], list)


def test_status_remote_only_errors_without_configured_remote(mock_discovery, runner) -> None:
    result = runner.invoke(cli.cli, ["status", "--remote-only"])

    assert result.exit_code != 0
    assert "No remotes configured" in result.output


def test_verify_errors_when_no_stages_registered(mock_discovery, runner) -> None:
    result = runner.invoke(cli.cli, ["verify"])

    assert result.exit_code != 0
    assert "No stages registered. Nothing to verify." in result.output


def test_validate_targets_filters_empty_and_whitespace() -> None:
    valid = targets_mod.validate_targets(("train", "", "   ", "score"))
    assert valid == ["train", "score"]


def test_validate_targets_raises_when_all_targets_invalid() -> None:
    try:
        targets_mod.validate_targets(("", "  "))
    except targets_mod.TargetValidationError as exc:
        assert "All targets are empty or whitespace-only" in str(exc)
    else:
        raise AssertionError("Expected TargetValidationError")


def test_resolve_cli_target_reports_missing_output_key() -> None:
    stage = _helper_stage_info("train", [_helper_ref("train", "model")])
    all_stages: dict[str, RegistryStageInfo] = {"train": stage}

    try:
        targets_mod.resolve_cli_target("train:missing", all_stages, pvt_exists=lambda _: False)
    except targets_mod.TargetValidationError as exc:
        message = str(exc)
        assert "has no output key 'missing'" in message
        assert "train:model" in message
    else:
        raise AssertionError("Expected TargetValidationError")


def test_resolve_and_validate_raises_click_exception_for_unknown_target(
    mock_discovery,
    tmp_path: pathlib.Path,
) -> None:
    unknown_target = "not-a-stage"

    try:
        targets_mod.resolve_and_validate((unknown_target,), tmp_path, output_type=outputs.Metric)
    except click.ClickException as exc:
        assert unknown_target in str(exc)
    else:
        raise AssertionError("Expected ClickException")
