"""Tests for failed stages summary in pivot repro command."""

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


@compose.stage
def downstream(failing_output: str) -> str:
    return "ran"


def _write_input(root: pathlib.Path) -> None:
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "input.txt").write_text("data")


# =============================================================================
# repro failed stages summary tests
# =============================================================================


def test_repro_failed_stages_listed_at_end(
    mock_discovery: compose.Pipeline,
    runner: CliRunner,
) -> None:
    """repro output contains 'Failed stages:' heading with failed stage name."""
    _write_input(mock_discovery.root)
    with mock_discovery:
        failing(mock_discovery.input("input.txt", t=str))

    result = runner.invoke(cli.cli, ["repro"])

    assert result.exit_code == 1, "repro exits non-zero when a stage fails"
    assert "Failed stages:" in result.output, "Output should contain 'Failed stages:' heading"
    assert "test/failing" in result.output, "Failed stage name should be listed"


def test_repro_no_failed_summary_when_all_succeed(
    mock_discovery: compose.Pipeline,
    runner: CliRunner,
) -> None:
    """repro output does NOT contain 'Failed stages:' when all stages succeed."""
    _write_input(mock_discovery.root)
    with mock_discovery:
        succeeding(mock_discovery.input("input.txt", t=str))

    result = runner.invoke(cli.cli, ["repro"])

    assert result.exit_code == 0, result.output
    assert "Failed stages:" not in result.output, (
        "Output should not contain 'Failed stages:' when all succeed"
    )


def test_repro_failed_stages_excludes_blocked(
    mock_discovery: compose.Pipeline,
    runner: CliRunner,
) -> None:
    """repro --keep-going lists only failed stages, not blocked downstream stages."""
    _write_input(mock_discovery.root)
    with mock_discovery:
        downstream(failing(mock_discovery.input("input.txt", t=str)))

    result = runner.invoke(cli.cli, ["repro", "--keep-going"])

    assert result.exit_code == 1, "repro exits non-zero when a stage fails"
    assert "Failed stages:" in result.output, "Output should contain 'Failed stages:' heading"
    assert "test/failing" in result.output, "Failed stage should be listed"
    # The downstream stage is blocked (skipped), not failed, so it should NOT be in the failed list
    # We check that "downstream" doesn't appear immediately after "Failed stages:" section
    lines = result.output.split("\n")
    failed_section_start = None
    for i, line in enumerate(lines):
        if "Failed stages:" in line:
            failed_section_start = i
            break
    assert failed_section_start is not None, "Failed stages section should exist"
    # Check that only "failing" is in the failed stages section (before next section or end)
    failed_section = "\n".join(lines[failed_section_start : failed_section_start + 5])
    assert "test/failing" in failed_section, "failing should be in failed stages section"
    assert "downstream" not in failed_section, (
        "downstream (blocked) should not be in failed stages section"
    )


def test_repro_jsonl_includes_failed_stages(
    mock_discovery: compose.Pipeline,
    runner: CliRunner,
) -> None:
    """repro --jsonl output includes failed_stages list in execution_result event."""
    _write_input(mock_discovery.root)
    with mock_discovery:
        raw = mock_discovery.input("input.txt", t=str)
        failing(raw)
        succeeding(raw)

    result = runner.invoke(cli.cli, ["repro", "--jsonl"])

    assert result.exit_code == 1, "repro exits non-zero when a stage fails"
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
