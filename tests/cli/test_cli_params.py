from __future__ import annotations

import json
from typing import TYPE_CHECKING

import yaml

from helpers import register_test_stage
from pivot import cli, stage_def

if TYPE_CHECKING:
    import click.testing
    import pytest
    from pytest_mock import MockerFixture

    from pivot.pipeline.pipeline import Pipeline


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


def _helper_train(params: _TrainParams) -> None:
    pass


def _helper_stage_simple(params: _SimpleParams) -> None:
    pass


def _helper_stage_a(params: _SimpleParams) -> None:
    pass


def _helper_stage_b(params: _SimpleParams) -> None:
    pass


def _helper_stage_c(params: _SimpleParams) -> None:
    pass


def _helper_stage_precision(params: _PrecisionParams) -> None:
    pass


def _helper_stage_changed(params: _ChangedParams) -> None:
    pass


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
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shows no params message when no stages registered."""
    monkeypatch.chdir(mock_discovery.root)

    result = runner.invoke(cli.cli, ["params", "show"])

    assert result.exit_code == 0
    assert "No parameters found" in result.output


def test_params_show_with_params(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shows params from registered stage."""
    monkeypatch.chdir(mock_discovery.root)

    register_test_stage(_helper_train, name="train", params=_TrainParams())

    result = runner.invoke(cli.cli, ["params", "show"])

    assert result.exit_code == 0
    assert "train" in result.output
    assert "lr" in result.output
    assert "0.01" in result.output


def test_params_show_json_format(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """params show --json outputs valid JSON."""
    monkeypatch.chdir(mock_discovery.root)

    register_test_stage(_helper_stage_simple, name="stage", params=_SimpleParams())

    result = runner.invoke(cli.cli, ["params", "show", "--json"])

    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["stage"]["x"] == 1


def test_params_show_md_format(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """params show --md outputs markdown table."""
    monkeypatch.chdir(mock_discovery.root)

    register_test_stage(_helper_stage_simple, name="stage", params=_SimpleParams())

    result = runner.invoke(cli.cli, ["params", "show", "--md"])

    assert result.exit_code == 0
    assert "|" in result.output
    assert "---" in result.output


def test_params_show_specific_stages(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """params show filters to specific stages."""
    monkeypatch.chdir(mock_discovery.root)

    register_test_stage(_helper_stage_a, name="stage_a", params=_SimpleParams())
    register_test_stage(_helper_stage_b, name="stage_b", params=_SimpleParams())
    register_test_stage(_helper_stage_c, name="stage_c", params=_SimpleParams())

    result = runner.invoke(cli.cli, ["params", "show", "stage_a", "stage_c"])

    assert result.exit_code == 0
    assert "stage_a" in result.output
    assert "stage_c" in result.output


def test_params_show_precision(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """params show respects --precision flag."""
    monkeypatch.chdir(mock_discovery.root)

    register_test_stage(_helper_stage_precision, name="stage", params=_PrecisionParams())

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
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shows message when no stages registered."""
    monkeypatch.chdir(mock_discovery.root)

    result = runner.invoke(cli.cli, ["params", "diff"])

    assert result.exit_code == 0
    assert "No parameters found" in result.output


def test_params_diff_no_changes(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Shows no changes when params match HEAD."""
    monkeypatch.chdir(mock_discovery.root)

    register_test_stage(_helper_stage_simple, name="stage", params=_SimpleParams())

    from pivot import git

    lock_content = yaml.dump(
        {
            "code_manifest": {},
            "params": {"x": 1},
            "deps": [],
            "outs": [],
            "dep_generations": {},
        }
    )
    mocker.patch.object(
        git,
        "read_files_from_head",
        return_value={".pivot/stages/stage.lock": lock_content.encode()},
    )

    result = runner.invoke(cli.cli, ["params", "diff"])

    assert result.exit_code == 0
    assert "No parameter changes" in result.output


def test_params_diff_with_changes(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Shows diff when params changed from HEAD."""
    monkeypatch.chdir(mock_discovery.root)

    register_test_stage(_helper_stage_changed, name="stage", params=_ChangedParams())

    from pivot import git

    lock_content = yaml.dump(
        {
            "code_manifest": {},
            "params": {"x": 1},
            "deps": [],
            "outs": [],
            "dep_generations": {},
        }
    )
    mocker.patch.object(
        git,
        "read_files_from_head",
        return_value={".pivot/stages/stage.lock": lock_content.encode()},
    )

    result = runner.invoke(cli.cli, ["params", "diff"])

    assert result.exit_code == 0
    assert "modified" in result.output
    assert "stage" in result.output


def test_params_diff_json_format(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """params diff --json outputs valid JSON."""
    monkeypatch.chdir(mock_discovery.root)

    register_test_stage(_helper_stage_changed, name="stage", params=_ChangedParams())

    from pivot import git

    lock_content = yaml.dump(
        {
            "code_manifest": {},
            "params": {"x": 1},
            "deps": [],
            "outs": [],
            "dep_generations": {},
        }
    )
    mocker.patch.object(
        git,
        "read_files_from_head",
        return_value={".pivot/stages/stage.lock": lock_content.encode()},
    )

    result = runner.invoke(cli.cli, ["params", "diff", "--json"])

    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert len(parsed) == 1
    assert parsed[0]["change_type"] == "modified"


def test_params_diff_md_format(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """params diff --md outputs markdown table."""
    monkeypatch.chdir(mock_discovery.root)

    register_test_stage(_helper_stage_changed, name="stage", params=_ChangedParams())

    from pivot import git

    lock_content = yaml.dump(
        {
            "code_manifest": {},
            "params": {"x": 1},
            "deps": [],
            "outs": [],
            "dep_generations": {},
        }
    )
    mocker.patch.object(
        git,
        "read_files_from_head",
        return_value={".pivot/stages/stage.lock": lock_content.encode()},
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
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """params show errors on unknown stage names."""
    monkeypatch.chdir(mock_discovery.root)

    result = runner.invoke(cli.cli, ["params", "show", "nonexistent_stage"])

    assert result.exit_code != 0
    assert "Unknown stage(s): nonexistent_stage" in result.output


def test_params_diff_unknown_stage_error(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """params diff errors on unknown stage names."""
    monkeypatch.chdir(mock_discovery.root)

    result = runner.invoke(cli.cli, ["params", "diff", "nonexistent_stage"])

    assert result.exit_code != 0
    assert "Unknown stage(s): nonexistent_stage" in result.output


def test_params_diff_no_git_warning(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """params diff warns when not in git repo."""
    monkeypatch.chdir(mock_discovery.root)

    register_test_stage(_helper_stage_simple, name="stage", params=_SimpleParams())

    from pivot import git

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
    test_pipeline: Pipeline,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pivot params show discovers pipeline without needing prior command."""
    from pivot import discovery, project

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
    test_pipeline: Pipeline,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pivot params diff discovers pipeline without needing prior command."""
    from pivot import discovery, project

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
