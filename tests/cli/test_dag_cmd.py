"""CLI integration tests for `pivot dag` command."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Annotated, TypedDict

from helpers import register_test_stage
from pivot import cli, loaders, outputs

if TYPE_CHECKING:
    import click.testing
    import pytest

    from pivot.pipeline.pipeline import Pipeline


# =============================================================================
# Output TypedDicts for annotation-based stages
# =============================================================================


class _ExtractOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("raw.csv", loaders.PathOnly())]


class _TransformOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("clean.csv", loaders.PathOnly())]


class _LoadOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.csv", loaders.PathOnly())]


class _BranchAOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("branch_a.csv", loaders.PathOnly())]


class _BranchBOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("branch_b.csv", loaders.PathOnly())]


class _MergeOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("merged.csv", loaders.PathOnly())]


# =============================================================================
# Module-level helper functions for stage registration
# =============================================================================


def _helper_extract() -> _ExtractOutputs:
    pathlib.Path("raw.csv").write_text("raw")
    return {"output": pathlib.Path("raw.csv")}


def _helper_transform(
    raw: Annotated[pathlib.Path, outputs.Dep("raw.csv", loaders.PathOnly())],
) -> _TransformOutputs:
    _ = raw
    pathlib.Path("clean.csv").write_text("clean")
    return {"output": pathlib.Path("clean.csv")}


def _helper_load(
    clean: Annotated[pathlib.Path, outputs.Dep("clean.csv", loaders.PathOnly())],
) -> _LoadOutputs:
    _ = clean
    pathlib.Path("output.csv").write_text("output")
    return {"output": pathlib.Path("output.csv")}


def _helper_branch_a(
    raw: Annotated[pathlib.Path, outputs.Dep("raw.csv", loaders.PathOnly())],
) -> _BranchAOutputs:
    _ = raw
    pathlib.Path("branch_a.csv").write_text("a")
    return {"output": pathlib.Path("branch_a.csv")}


def _helper_branch_b(
    raw: Annotated[pathlib.Path, outputs.Dep("raw.csv", loaders.PathOnly())],
) -> _BranchBOutputs:
    _ = raw
    pathlib.Path("branch_b.csv").write_text("b")
    return {"output": pathlib.Path("branch_b.csv")}


def _helper_merge(
    a: Annotated[pathlib.Path, outputs.Dep("branch_a.csv", loaders.PathOnly())],
    b: Annotated[pathlib.Path, outputs.Dep("branch_b.csv", loaders.PathOnly())],
) -> _MergeOutputs:
    _ = a, b
    pathlib.Path("merged.csv").write_text("merged")
    return {"output": pathlib.Path("merged.csv")}


# =============================================================================
# Empty pipeline tests
# =============================================================================


def test_dag_empty_pipeline_ascii(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty pipeline shows empty graph placeholder in ASCII (default) format."""
    monkeypatch.chdir(tmp_path)
    pathlib.Path(".git").mkdir()

    result = runner.invoke(cli.cli, ["dag"])

    assert result.exit_code == 0
    assert "(empty graph)" in result.output


def test_dag_empty_pipeline_mermaid(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty pipeline shows valid empty Mermaid flowchart."""
    monkeypatch.chdir(tmp_path)
    pathlib.Path(".git").mkdir()

    result = runner.invoke(cli.cli, ["dag", "--mermaid"])

    assert result.exit_code == 0
    assert result.output.strip() == "flowchart TD"


def test_dag_empty_pipeline_dot(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty pipeline shows minimal DOT graph."""
    monkeypatch.chdir(tmp_path)
    pathlib.Path(".git").mkdir()

    result = runner.invoke(cli.cli, ["dag", "--dot"])

    assert result.exit_code == 0
    assert "digraph {" in result.output
    assert "}" in result.output


def test_dag_empty_pipeline_md(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty pipeline shows Mermaid wrapped in markdown code block."""
    monkeypatch.chdir(tmp_path)
    pathlib.Path(".git").mkdir()

    result = runner.invoke(cli.cli, ["dag", "--md"])

    assert result.exit_code == 0
    assert "```mermaid" in result.output
    assert "```" in result.output
    assert "flowchart TD" in result.output


# =============================================================================
# Single stage tests
# =============================================================================


def test_dag_single_stage_ascii(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single stage shows output artifact in ASCII."""
    monkeypatch.chdir(tmp_path)
    pathlib.Path(".git").mkdir()

    register_test_stage(_helper_extract, name="extract")

    result = runner.invoke(cli.cli, ["dag"])

    assert result.exit_code == 0
    # Should show the artifact path
    assert "raw.csv" in result.output


def test_dag_single_stage_stages_flag(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single stage shows stage name with --stages flag."""
    monkeypatch.chdir(tmp_path)
    pathlib.Path(".git").mkdir()

    register_test_stage(_helper_extract, name="extract")

    result = runner.invoke(cli.cli, ["dag", "--stages"])

    assert result.exit_code == 0
    assert "extract" in result.output


# =============================================================================
# Linear chain tests (A -> B -> C)
# =============================================================================


def test_dag_linear_chain_ascii(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linear pipeline shows all artifacts in ASCII."""
    monkeypatch.chdir(tmp_path)
    pathlib.Path(".git").mkdir()

    register_test_stage(_helper_extract, name="extract")
    register_test_stage(_helper_transform, name="transform")
    register_test_stage(_helper_load, name="load")

    result = runner.invoke(cli.cli, ["dag"])

    assert result.exit_code == 0
    assert "raw.csv" in result.output
    assert "clean.csv" in result.output
    assert "output.csv" in result.output


def test_dag_linear_chain_stages(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linear pipeline shows all stages with --stages flag."""
    monkeypatch.chdir(tmp_path)
    pathlib.Path(".git").mkdir()

    register_test_stage(_helper_extract, name="extract")
    register_test_stage(_helper_transform, name="transform")
    register_test_stage(_helper_load, name="load")

    result = runner.invoke(cli.cli, ["dag", "--stages"])

    assert result.exit_code == 0
    assert "extract" in result.output
    assert "transform" in result.output
    assert "load" in result.output


def test_dag_linear_chain_mermaid(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linear pipeline shows edges in Mermaid format."""
    monkeypatch.chdir(tmp_path)
    pathlib.Path(".git").mkdir()

    register_test_stage(_helper_extract, name="extract")
    register_test_stage(_helper_transform, name="transform")
    register_test_stage(_helper_load, name="load")

    result = runner.invoke(cli.cli, ["dag", "--mermaid", "--stages"])

    assert result.exit_code == 0
    assert "flowchart TD" in result.output
    assert "extract" in result.output
    assert "transform" in result.output
    assert "load" in result.output
    assert "-->" in result.output


def test_dag_linear_chain_dot(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linear pipeline shows edges in DOT format."""
    monkeypatch.chdir(tmp_path)
    pathlib.Path(".git").mkdir()

    register_test_stage(_helper_extract, name="extract")
    register_test_stage(_helper_transform, name="transform")
    register_test_stage(_helper_load, name="load")

    result = runner.invoke(cli.cli, ["dag", "--dot", "--stages"])

    assert result.exit_code == 0
    assert "digraph {" in result.output
    assert "extract" in result.output
    assert "transform" in result.output
    assert "load" in result.output
    assert "->" in result.output


# =============================================================================
# Diamond pattern tests (A -> B, A -> C, B -> D, C -> D)
# =============================================================================


def test_dag_diamond_pattern_stages(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Diamond pattern shows all stages."""
    monkeypatch.chdir(tmp_path)
    pathlib.Path(".git").mkdir()

    register_test_stage(_helper_extract, name="extract")
    register_test_stage(_helper_branch_a, name="branch_a")
    register_test_stage(_helper_branch_b, name="branch_b")
    register_test_stage(_helper_merge, name="merge")

    result = runner.invoke(cli.cli, ["dag", "--stages"])

    assert result.exit_code == 0
    assert "extract" in result.output
    assert "branch_a" in result.output
    assert "branch_b" in result.output
    assert "merge" in result.output


def test_dag_diamond_pattern_mermaid(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Diamond pattern shows correct edges in Mermaid."""
    monkeypatch.chdir(tmp_path)
    pathlib.Path(".git").mkdir()

    register_test_stage(_helper_extract, name="extract")
    register_test_stage(_helper_branch_a, name="branch_a")
    register_test_stage(_helper_branch_b, name="branch_b")
    register_test_stage(_helper_merge, name="merge")

    result = runner.invoke(cli.cli, ["dag", "--mermaid", "--stages"])

    assert result.exit_code == 0
    # Should have multiple edges (fan-out and fan-in)
    assert result.output.count("-->") >= 4


# =============================================================================
# Target filtering tests
# =============================================================================


def test_dag_target_single_stage(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Target filtering by stage name shows stage and its upstream."""
    monkeypatch.chdir(tmp_path)
    pathlib.Path(".git").mkdir()

    register_test_stage(_helper_extract, name="extract")
    register_test_stage(_helper_transform, name="transform")
    register_test_stage(_helper_load, name="load")

    result = runner.invoke(cli.cli, ["dag", "--stages", "transform"])

    assert result.exit_code == 0
    # Should include extract (upstream) and transform
    assert "extract" in result.output
    assert "transform" in result.output
    # Should NOT include load (downstream of transform)
    assert "load" not in result.output


def test_dag_target_artifact_path(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Target filtering by artifact path resolves to producing stage and upstream."""
    monkeypatch.chdir(tmp_path)
    pathlib.Path(".git").mkdir()

    register_test_stage(_helper_extract, name="extract")
    register_test_stage(_helper_transform, name="transform")
    register_test_stage(_helper_load, name="load")

    # Artifact path resolves to the stage that produces it and its upstream deps
    result = runner.invoke(cli.cli, ["dag", "--stages", "clean.csv"])

    assert result.exit_code == 0
    # Should include transform (produces clean.csv) and extract (upstream)
    assert "transform" in result.output
    assert "extract" in result.output
    # Should NOT include load (downstream)
    assert "load" not in result.output


def test_dag_target_multiple_targets(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multiple targets includes all specified stages and their upstreams."""
    monkeypatch.chdir(tmp_path)
    pathlib.Path(".git").mkdir()

    register_test_stage(_helper_extract, name="extract")
    register_test_stage(_helper_branch_a, name="branch_a")
    register_test_stage(_helper_branch_b, name="branch_b")
    register_test_stage(_helper_merge, name="merge")

    result = runner.invoke(cli.cli, ["dag", "--stages", "branch_a", "branch_b"])

    assert result.exit_code == 0
    # Should include extract (upstream of both)
    assert "extract" in result.output
    assert "branch_a" in result.output
    assert "branch_b" in result.output


def test_dag_target_invalid_stage_error(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid target that doesn't match any stage or artifact shows error."""
    monkeypatch.chdir(tmp_path)
    pathlib.Path(".git").mkdir()

    register_test_stage(_helper_extract, name="extract")

    result = runner.invoke(cli.cli, ["dag", "nonexistent_stage"])

    # Should fail with informative error
    assert result.exit_code != 0
    assert "No stages found" in result.output or "Error" in result.output


# =============================================================================
# Output format tests
# =============================================================================


def test_dag_md_format_wraps_mermaid(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--md flag wraps Mermaid output in markdown code block."""
    monkeypatch.chdir(tmp_path)
    pathlib.Path(".git").mkdir()

    register_test_stage(_helper_extract, name="extract")
    register_test_stage(_helper_transform, name="transform")

    result = runner.invoke(cli.cli, ["dag", "--md", "--stages"])

    assert result.exit_code == 0
    # Should have opening code fence
    assert "```mermaid" in result.output
    # Should have closing code fence
    lines = result.output.strip().split("\n")
    assert lines[-1] == "```"
    # Should have Mermaid content inside
    assert "flowchart TD" in result.output


def test_dag_ascii_default_format(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default output format is ASCII."""
    monkeypatch.chdir(tmp_path)
    pathlib.Path(".git").mkdir()

    register_test_stage(_helper_extract, name="extract")
    register_test_stage(_helper_transform, name="transform")

    result = runner.invoke(cli.cli, ["dag"])

    assert result.exit_code == 0
    # ASCII format has box characters
    assert "+" in result.output or "(empty graph)" in result.output


# =============================================================================
# --stages flag tests
# =============================================================================


def test_dag_artifact_view_default(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default view shows artifacts, not stage names."""
    monkeypatch.chdir(tmp_path)
    pathlib.Path(".git").mkdir()

    register_test_stage(_helper_extract, name="extract")

    result = runner.invoke(cli.cli, ["dag"])

    assert result.exit_code == 0
    # Should show artifact, not stage name
    assert "raw.csv" in result.output


def test_dag_stage_view_with_flag(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--stages flag shows stage names instead of artifacts."""
    monkeypatch.chdir(tmp_path)
    pathlib.Path(".git").mkdir()

    register_test_stage(_helper_extract, name="extract")

    result = runner.invoke(cli.cli, ["dag", "--stages"])

    assert result.exit_code == 0
    assert "extract" in result.output


def test_dag_stages_flag_with_mermaid(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--stages flag works with --mermaid format."""
    monkeypatch.chdir(tmp_path)
    pathlib.Path(".git").mkdir()

    register_test_stage(_helper_extract, name="extract")
    register_test_stage(_helper_transform, name="transform")

    result = runner.invoke(cli.cli, ["dag", "--stages", "--mermaid"])

    assert result.exit_code == 0
    assert "extract" in result.output
    assert "transform" in result.output
    # Should not show artifacts in stage view
    assert "raw.csv" not in result.output


def test_dag_stages_flag_with_dot(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--stages flag works with --dot format."""
    monkeypatch.chdir(tmp_path)
    pathlib.Path(".git").mkdir()

    register_test_stage(_helper_extract, name="extract")
    register_test_stage(_helper_transform, name="transform")

    result = runner.invoke(cli.cli, ["dag", "--stages", "--dot"])

    assert result.exit_code == 0
    assert "extract" in result.output
    assert "transform" in result.output


def test_dag_target_partial_resolution(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Command succeeds with partial target resolution (valid + invalid targets)."""
    monkeypatch.chdir(tmp_path)
    pathlib.Path(".git").mkdir()

    register_test_stage(_helper_extract, name="extract")
    register_test_stage(_helper_transform, name="transform")

    # Mix of valid and invalid targets - should succeed with valid targets
    result = runner.invoke(cli.cli, ["dag", "extract", "nonexistent_stage", "--stages"])

    assert result.exit_code == 0
    # Valid target should be in output
    assert "extract" in result.output
    # transform not requested, should not appear in filtered output
    assert "transform" not in result.output
