from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import yaml

from pivot import cli, compose, discovery, git, project, stage_def

if TYPE_CHECKING:
    import click.testing
    from pytest_mock import MockerFixture

# =============================================================================
# Module-level StageParams classes for annotation-based registration
# =============================================================================


class _TrainParams(stage_def.StageParams):
    lr: float = 0.01
    epochs: int = 10


class _SimpleParams(stage_def.StageParams):
    x: int = 1


class _PrecisionParams(stage_def.StageParams):
    lr: float = 0.123456789


class _ChangedParams(stage_def.StageParams):
    x: int = 2


# =============================================================================
# Module-level stage functions
# =============================================================================


@compose.stage
def train(params: _TrainParams) -> None:
    pass


@compose.stage
def stage_simple(params: _SimpleParams) -> None:
    pass


@compose.stage
def stage_a(params: _SimpleParams) -> None:
    pass


@compose.stage
def stage_b(params: _SimpleParams) -> None:
    pass


@compose.stage
def stage_c(params: _SimpleParams) -> None:
    pass


@compose.stage
def stage_precision(params: _PrecisionParams) -> None:
    pass


@compose.stage
def stage_changed(params: _ChangedParams) -> None:
    pass


def _head_lock(stage_name: str, params: dict[str, object]) -> dict[str, bytes]:
    """HEAD lock-file content for ``git.read_files_from_head`` (path keyed by qualified name)."""
    content = yaml.dump({"code_manifest": {}, "params": params, "deps": [], "outs": []})
    return {f".pivot/stages/{stage_name}.lock": content.encode()}


# =============================================================================
# Params Show Tests
# =============================================================================


def test_params_show_help(runner: click.testing.CliRunner) -> None:
    """params show command shows help."""
    result = runner.invoke(cli.cli, ["params", "show", "--help"])
    assert result.exit_code == 0
    assert "--json" in result.output
    assert "--md" in result.output
    assert "--precision" in result.output


def test_params_show_no_stages(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shows no params message when no stages registered."""
    monkeypatch.chdir(mock_discovery.root)

    result = runner.invoke(cli.cli, ["params", "show"])

    assert result.exit_code == 0
    assert "No parameters found" in result.output


def test_params_show_with_params(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shows params from registered stage."""
    monkeypatch.chdir(mock_discovery.root)

    with mock_discovery:
        train(params=_TrainParams())

    result = runner.invoke(cli.cli, ["params", "show"])

    assert result.exit_code == 0
    assert "train" in result.output
    assert "lr" in result.output
    assert "0.01" in result.output


def test_params_show_json_format(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """params show --json outputs valid JSON."""
    monkeypatch.chdir(mock_discovery.root)

    with mock_discovery:
        stage_simple(params=_SimpleParams())

    result = runner.invoke(cli.cli, ["params", "show", "--json"])

    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["test/stage_simple"]["x"] == 1


def test_params_show_md_format(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """params show --md outputs markdown table."""
    monkeypatch.chdir(mock_discovery.root)

    with mock_discovery:
        stage_simple(params=_SimpleParams())

    result = runner.invoke(cli.cli, ["params", "show", "--md"])

    assert result.exit_code == 0
    assert "|" in result.output
    assert "---" in result.output


def test_params_show_specific_stages(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """params show filters to specific stages."""
    monkeypatch.chdir(mock_discovery.root)

    with mock_discovery:
        stage_a(params=_SimpleParams())
        stage_b(params=_SimpleParams())
        stage_c(params=_SimpleParams())

    result = runner.invoke(cli.cli, ["params", "show", "test/stage_a", "test/stage_c"])

    assert result.exit_code == 0, result.output
    assert "test/stage_a" in result.output
    assert "test/stage_c" in result.output
    assert "test/stage_b" not in result.output


@pytest.mark.xfail(
    strict=True,
    reason="red test for copilot-fixes (Task 3, #446): bare stage name not resolved",
)
def test_params_show_bare_stage_name_resolves(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """params show accepts a bare stage name in a single-pipeline project (names.py design)."""
    with mock_discovery:
        stage_a(params=_SimpleParams())
        stage_b(params=_SimpleParams())

    result = runner.invoke(cli.cli, ["params", "show", "stage_a"])

    assert result.exit_code == 0, result.output
    assert "test/stage_a" in result.output
    assert "test/stage_b" not in result.output


def test_params_show_precision(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """params show respects --precision flag."""
    monkeypatch.chdir(mock_discovery.root)

    with mock_discovery:
        stage_precision(params=_PrecisionParams())

    result = runner.invoke(cli.cli, ["params", "show", "--precision", "2"])

    assert result.exit_code == 0
    assert "0.12" in result.output
    assert "0.123456789" not in result.output


# =============================================================================
# Params Diff Tests
# =============================================================================


def test_params_diff_help(runner: click.testing.CliRunner) -> None:
    """params diff command shows help."""
    result = runner.invoke(cli.cli, ["params", "diff", "--help"])
    assert result.exit_code == 0
    assert "--json" in result.output
    assert "--md" in result.output
    assert "--precision" in result.output


def test_params_diff_no_stages(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shows message when no stages registered."""
    monkeypatch.chdir(mock_discovery.root)

    result = runner.invoke(cli.cli, ["params", "diff"])

    assert result.exit_code == 0
    assert "No parameters found" in result.output


def test_params_diff_no_changes(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Shows no changes when params match HEAD."""
    monkeypatch.chdir(mock_discovery.root)

    with mock_discovery:
        stage_simple(params=_SimpleParams())

    mocker.patch.object(
        git, "read_files_from_head", return_value=_head_lock("test/stage_simple", {"x": 1})
    )

    result = runner.invoke(cli.cli, ["params", "diff"])

    assert result.exit_code == 0
    assert "No parameter changes" in result.output


def test_params_diff_with_changes(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Shows diff when params changed from HEAD."""
    monkeypatch.chdir(mock_discovery.root)

    with mock_discovery:
        stage_changed(params=_ChangedParams())

    mocker.patch.object(
        git, "read_files_from_head", return_value=_head_lock("test/stage_changed", {"x": 1})
    )

    result = runner.invoke(cli.cli, ["params", "diff"])

    assert result.exit_code == 0
    assert "modified" in result.output
    assert "stage" in result.output


def test_params_diff_json_format(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """params diff --json outputs valid JSON."""
    monkeypatch.chdir(mock_discovery.root)

    with mock_discovery:
        stage_changed(params=_ChangedParams())

    mocker.patch.object(
        git, "read_files_from_head", return_value=_head_lock("test/stage_changed", {"x": 1})
    )

    result = runner.invoke(cli.cli, ["params", "diff", "--json"])

    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert len(parsed) == 1
    assert parsed[0]["change_type"] == "modified"


def test_params_diff_md_format(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """params diff --md outputs markdown table."""
    monkeypatch.chdir(mock_discovery.root)

    with mock_discovery:
        stage_changed(params=_ChangedParams())

    mocker.patch.object(
        git, "read_files_from_head", return_value=_head_lock("test/stage_changed", {"x": 1})
    )

    result = runner.invoke(cli.cli, ["params", "diff", "--md"])

    assert result.exit_code == 0
    assert "|" in result.output
    assert "---" in result.output


# =============================================================================
# Command Group Tests
# =============================================================================


def test_params_group_help(runner: click.testing.CliRunner) -> None:
    """Params group shows subcommands."""
    result = runner.invoke(cli.cli, ["params", "--help"])
    assert result.exit_code == 0
    assert "show" in result.output
    assert "diff" in result.output


def test_params_in_main_help(runner: click.testing.CliRunner) -> None:
    """Params command appears in main help."""
    result = runner.invoke(cli.cli, ["--help"])
    assert result.exit_code == 0
    assert "params" in result.output


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_params_show_unknown_stage_error(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """params show errors on unknown stage names."""
    monkeypatch.chdir(mock_discovery.root)

    result = runner.invoke(cli.cli, ["params", "show", "nonexistent_stage"])

    assert result.exit_code != 0
    assert "Unknown stage(s): nonexistent_stage" in result.output


def test_params_diff_unknown_stage_error(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """params diff errors on unknown stage names."""
    monkeypatch.chdir(mock_discovery.root)

    result = runner.invoke(cli.cli, ["params", "diff", "nonexistent_stage"])

    assert result.exit_code != 0
    assert "Unknown stage(s): nonexistent_stage" in result.output


def test_params_diff_no_git_warning(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """params diff warns when not in git repo."""
    monkeypatch.chdir(mock_discovery.root)

    with mock_discovery:
        stage_simple(params=_SimpleParams())

    mocker.patch.object(git, "read_files_from_head", return_value={})
    mocker.patch.object(git, "is_git_repo_with_head", return_value=False)

    result = runner.invoke(cli.cli, ["params", "diff"])

    assert result.exit_code == 0
    assert "Warning: Not in a git repository" in result.output


# =============================================================================
# Pipeline Discovery Tests
# =============================================================================


def test_params_show_without_prior_run_discovers_pipeline(
    runner: click.testing.CliRunner,
    test_pipeline: compose.Pipeline,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pivot params show discovers pipeline without needing prior command."""
    monkeypatch.chdir(test_pipeline.root)
    (test_pipeline.root / ".pivot").mkdir(exist_ok=True)

    # Mock discover_pipeline to return the test pipeline
    mocker.patch.object(discovery, "discover_pipeline", return_value=test_pipeline)
    mocker.patch.object(project, "_project_root_cache", test_pipeline.root)
    # IMPORTANT: Do NOT mock get_pipeline_from_context - we want to test that
    # ensure_stages_registered() is called and populates the context

    result = runner.invoke(cli.cli, ["params", "show"])

    # Should not raise NoPipelineError - discovery should have been called
    assert "No pipeline found" not in result.output, f"Output: {result.output}"
    assert result.exit_code == 0, f"Output: {result.output}"


def test_params_diff_without_prior_run_discovers_pipeline(
    runner: click.testing.CliRunner,
    test_pipeline: compose.Pipeline,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pivot params diff discovers pipeline without needing prior command."""
    monkeypatch.chdir(test_pipeline.root)
    (test_pipeline.root / ".pivot").mkdir(exist_ok=True)

    # Mock discover_pipeline to return the test pipeline
    mocker.patch.object(discovery, "discover_pipeline", return_value=test_pipeline)
    mocker.patch.object(project, "_project_root_cache", test_pipeline.root)
    # IMPORTANT: Do NOT mock get_pipeline_from_context - we want to test that
    # ensure_stages_registered() is called and populates the context

    result = runner.invoke(cli.cli, ["params", "diff"])

    # Should not raise NoPipelineError - discovery should have been called
    assert "No pipeline found" not in result.output, f"Output: {result.output}"
    assert result.exit_code == 0, f"Output: {result.output}"
