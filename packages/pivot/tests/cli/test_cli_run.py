"""Tests for failed stages summary in pivot run command."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pivot import cli, compose

if TYPE_CHECKING:
    import pathlib

    from click.testing import CliRunner

# =============================================================================
# Module-level stage functions for testing (required for pickling)
# =============================================================================


@compose.stage
def failing(input_file: str) -> str:
    raise RuntimeError("Intentional failure")


@compose.stage
def succeeding(input_file: str) -> str:
    return "success"


def _write_input(root: pathlib.Path) -> None:
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "input.txt").write_text("data")


# =============================================================================
# run failed stages summary tests
# =============================================================================


def test_run_failed_stages_listed_at_end(
    mock_discovery: compose.Pipeline,
    runner: CliRunner,
) -> None:
    """run output contains 'Failed stages:' heading with failed stage name."""
    _write_input(mock_discovery.root)
    with mock_discovery:
        failing(mock_discovery.input("input.txt", t=str))

    result = runner.invoke(cli.cli, ["run", "test/failing"])

    assert result.exit_code == 1, "run exits non-zero when a stage fails"
    assert "Failed stages:" in result.output, "Output should contain 'Failed stages:' heading"
    assert "test/failing" in result.output, "Failed stage name should be listed"


def test_run_jsonl_includes_failed_stages(
    mock_discovery: compose.Pipeline,
    runner: CliRunner,
) -> None:
    """run --jsonl output includes failed_stages list in execution_result event."""
    _write_input(mock_discovery.root)
    with mock_discovery:
        raw = mock_discovery.input("input.txt", t=str)
        failing(raw)
        succeeding(raw)

    result = runner.invoke(cli.cli, ["run", "--jsonl", "test/failing", "test/succeeding"])

    assert result.exit_code == 1, "run exits non-zero when a stage fails"
    # Parse JSONL output
    lines = result.output.strip().split("\n")
    events = [json.loads(line) for line in lines if line.strip()]

    # Find the execution_result event
    execution_result = None
    for event in events:
        if event["type"] == "execution_result":
            execution_result = event
            break

    assert execution_result is not None, "Should have execution_result event in JSONL output"
    assert "failed_stages" in execution_result, "execution_result should have failed_stages field"
    assert isinstance(execution_result["failed_stages"], list), "failed_stages should be a list"
    assert "test/failing" in execution_result["failed_stages"], (
        "failing stage should be in failed_stages list"
    )
    assert "test/succeeding" not in execution_result["failed_stages"], (
        "succeeding stage should not be in failed_stages list"
    )
