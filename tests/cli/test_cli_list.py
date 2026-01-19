from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING, Annotated, TypedDict

from helpers import register_test_stage
from pivot import cli, loaders, outputs

if TYPE_CHECKING:
    import click.testing


# =============================================================================
# Output TypedDicts for annotation-based stages
# =============================================================================


class _StageAOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("a.txt", loaders.PathOnly())]


class _StageBOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("b.txt", loaders.PathOnly())]


class _ProducerOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("intermediate.txt", loaders.PathOnly())]


class _ConsumerOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("final.txt", loaders.PathOnly())]


# =============================================================================
# Module-level helper functions for stage registration
# =============================================================================


def _helper_stage_a(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _StageAOutputs:
    _ = input_file
    pathlib.Path("a.txt").write_text("output a")
    return {"output": pathlib.Path("a.txt")}


def _helper_stage_b(
    a_file: Annotated[pathlib.Path, outputs.Dep("a.txt", loaders.PathOnly())],
) -> _StageBOutputs:
    _ = a_file
    pathlib.Path("b.txt").write_text("output b")
    return {"output": pathlib.Path("b.txt")}


def _helper_producer(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _ProducerOutputs:
    _ = input_file
    pathlib.Path("intermediate.txt").write_text("intermediate")
    return {"output": pathlib.Path("intermediate.txt")}


def _helper_consumer(
    intermediate_file: Annotated[pathlib.Path, outputs.Dep("intermediate.txt", loaders.PathOnly())],
) -> _ConsumerOutputs:
    _ = intermediate_file
    pathlib.Path("final.txt").write_text("final")
    return {"output": pathlib.Path("final.txt")}


# =============================================================================
# No Stages Tests
# =============================================================================


def test_list_no_stages_explains_how_to_create(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Empty pipeline shows help text for creating stages."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        result = runner.invoke(cli.cli, ["list"])

        assert result.exit_code == 0
        assert "No stages registered" in result.output
        # Should mention how to create stages
        assert "pivot.yaml" in result.output


def test_list_no_stages_json(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """Returns {"stages": []} for JSON output with no stages."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        result = runner.invoke(cli.cli, ["list", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "stages" in data
        assert data["stages"] == []


# =============================================================================
# With Stages Tests
# =============================================================================


def test_list_with_stages_json(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """JSON output includes name, deps, outs, mutex, variant for all stages."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        register_test_stage(
            _helper_stage_a,
            name="stage_a",
            mutex=["gpu"],
        )
        register_test_stage(
            _helper_stage_b,
            name="stage_b",
        )

        result = runner.invoke(cli.cli, ["list", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "stages" in data
        assert len(data["stages"]) == 2

        # Check stage_a has all required fields
        stage_a_list = [s for s in data["stages"] if s["name"] == "stage_a"]
        assert len(stage_a_list) == 1, "Expected exactly one stage_a in output"
        stage_a = stage_a_list[0]
        # Paths may be absolute, so just check they end with expected filename
        assert len(stage_a["deps"]) == 1
        assert stage_a["deps"][0].endswith("input.txt")
        assert len(stage_a["outs"]) == 1
        assert stage_a["outs"][0].endswith("a.txt")
        assert stage_a["mutex"] == ["gpu"]
        assert "variant" in stage_a  # Present even if None


def test_list_deps_shows_source_stage(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """--deps shows source stage for dependencies that are outputs of other stages."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        register_test_stage(
            _helper_producer,
            name="producer",
        )
        register_test_stage(
            _helper_consumer,
            name="consumer",
        )

        result = runner.invoke(cli.cli, ["list", "--deps"])

        assert result.exit_code == 0
        assert "producer" in result.output
        assert "consumer" in result.output
        # Should show that consumer's dep comes from producer
        assert "from: producer" in result.output
