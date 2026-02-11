"""Tests for failed stages summary in pivot run command."""

from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING, Annotated, TypedDict

from helpers import register_test_stage
from pivot import cli, loaders, outputs

if TYPE_CHECKING:
    import pytest
    from click.testing import CliRunner

    from pivot.pipeline.pipeline import Pipeline

# =============================================================================
# Module-level stage functions for testing (required for pickling)
# =============================================================================


class _FailingTxtOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("failing.txt", loaders.PathOnly())]


class _SucceedingTxtOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("succeeding.txt", loaders.PathOnly())]


def _stage_failing(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _FailingTxtOutputs:
    raise RuntimeError("Intentional failure")


def _stage_succeeding(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _SucceedingTxtOutputs:
    pathlib.Path("succeeding.txt").write_text("success")
    return {"output": pathlib.Path("succeeding.txt")}


# =============================================================================
# run failed stages summary tests
# =============================================================================


def test_run_failed_stages_listed_at_end(
    mock_discovery: Pipeline,
    runner: CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run output contains 'Failed stages:' heading with failed stage name."""
    (tmp_path / "input.txt").write_text("data")

    register_test_stage(_stage_failing, name="failing")

    result = runner.invoke(cli.cli, ["run", "failing"])

    assert result.exit_code == 0
    assert "Failed stages:" in result.output, "Output should contain 'Failed stages:' heading"
    assert "failing" in result.output, "Failed stage name should be listed"


def test_run_jsonl_includes_failed_stages(
    mock_discovery: Pipeline,
    runner: CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run --jsonl output includes failed_stages list in execution_result event."""
    (tmp_path / "input.txt").write_text("data")

    register_test_stage(_stage_failing, name="failing")
    register_test_stage(_stage_succeeding, name="succeeding")

    result = runner.invoke(cli.cli, ["run", "--jsonl", "failing", "succeeding"])

    assert result.exit_code == 0
    # Parse JSONL output
    lines = result.output.strip().split("\n")
    events = [json.loads(line) for line in lines if line.strip()]

    # Find the execution_result event
    execution_result = None
    for event in events:
        if event.get("type") == "execution_result":
            execution_result = event
            break

    assert execution_result is not None, "Should have execution_result event in JSONL output"
    assert "failed_stages" in execution_result, "execution_result should have failed_stages field"
    assert isinstance(execution_result["failed_stages"], list), "failed_stages should be a list"
    assert "failing" in execution_result["failed_stages"], (
        "failing stage should be in failed_stages list"
    )
    assert "succeeding" not in execution_result["failed_stages"], (
        "succeeding stage should not be in failed_stages list"
    )
