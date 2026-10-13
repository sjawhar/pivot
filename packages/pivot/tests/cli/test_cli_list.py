from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pivot import cli, compose

if TYPE_CHECKING:
    import click.testing


# =============================================================================
# Module-level stage functions (required for pickling)
# =============================================================================


@compose.stage
def stage_a(input_file: str) -> str:
    return "output a"


@compose.stage
def stage_b(a_file: str) -> str:
    return "output b"


@compose.stage
def producer(input_file: str) -> str:
    return "intermediate"


@compose.stage
def consumer(intermediate_file: str) -> str:
    return "final"


# =============================================================================
# No Stages Tests
# =============================================================================


def test_list_no_stages_explains_how_to_create(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Empty pipeline shows help text for creating stages."""
    _ = mock_discovery

    result = runner.invoke(cli.cli, ["list"])

    assert result.exit_code == 0
    assert "No stages registered" in result.output
    # Should mention how to create stages
    assert "pipeline.py" in result.output


def test_list_no_stages_json(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Returns {"stages": []} for JSON output with no stages."""
    _ = mock_discovery

    result = runner.invoke(cli.cli, ["list", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "stages" in data
    assert data["stages"] == []


# =============================================================================
# With Stages Tests
# =============================================================================


def test_list_renders_registry_mutex_fixture(
    mock_discovery: compose.Pipeline, runner: click.testing.CliRunner
) -> None:
    """Render a non-empty registry mutex list.

    This is a test-only rendering fixture: @stage has no mutex/resources API yet.
    """
    # Task 19 replaces with @stage(resources=...).
    with mock_discovery:
        stage_b(stage_a(mock_discovery.input("input.txt", t=str)))
    mock_discovery.get_stage("test/stage_a")["mutex"] = ["gpu"]

    result = runner.invoke(cli.cli, ["list", "--json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "stages" in data
    assert len(data["stages"]) == 2

    # Check stage_a has all required fields
    stage_a_list = [s for s in data["stages"] if s["name"] == "test/stage_a"]
    assert len(stage_a_list) == 1, "Expected exactly one stage_a in output"
    stage_a_json = stage_a_list[0]
    assert stage_a_json["deps"] == ["input.txt"]
    assert stage_a_json["outs"] == ["test/stage_a"]
    assert stage_a_json["mutex"] == ["gpu"]
    assert "variant" in stage_a_json  # Present even if None

    stage_b_json = next(s for s in data["stages"] if s["name"] == "test/stage_b")
    assert stage_b_json["deps"] == ["test/stage_a"], "stage_b depends on stage_a's output"
    assert stage_b_json["mutex"] == []


def test_list_deps_shows_source_stage(
    mock_discovery: compose.Pipeline, runner: click.testing.CliRunner
) -> None:
    """--deps shows source stage for dependencies that are outputs of other stages."""
    with mock_discovery:
        consumer(producer(mock_discovery.input("input.txt", t=str)))

    result = runner.invoke(cli.cli, ["list", "--deps"])

    assert result.exit_code == 0, result.output
    assert "test/producer" in result.output
    assert "test/consumer" in result.output
    # Should show that consumer's dep comes from producer
    assert "from: test/producer" in result.output
    # A raw input has no producing stage
    assert "input.txt (from:" not in result.output


# =============================================================================
# No Pipeline Tests
# =============================================================================


def test_list_accepts_all_flag(runner: click.testing.CliRunner) -> None:
    """List should accept the --all flag."""
    result = runner.invoke(cli.cli, ["list", "--help"])
    assert result.exit_code == 0
    assert "--all" in result.output, "List should accept --all flag"
