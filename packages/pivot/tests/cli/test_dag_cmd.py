"""CLI integration tests for `pivot dag` command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pivot import cli, compose

if TYPE_CHECKING:
    import click.testing

# =============================================================================
# Module-level stage functions (required for pickling)
# =============================================================================


@compose.stage
def extract(source: str) -> str:
    return "raw"


@compose.stage
def transform(raw: str) -> str:
    return "clean"


@compose.stage
def load(clean: str) -> str:
    return "output"


@compose.stage
def branch_a(raw: str) -> str:
    return "a"


@compose.stage
def branch_b(raw: str) -> str:
    return "b"


@compose.stage
def merge(a: str, b: str) -> str:
    return "merged"


# The raw input "source.csv" appears as an artifact node in the default view only.
_SOURCE = "source.csv"


def _register_extract(pipeline: compose.Pipeline) -> None:
    with pipeline:
        extract(pipeline.input(_SOURCE, t=str))


def _register_extract_transform(pipeline: compose.Pipeline) -> None:
    with pipeline:
        transform(extract(pipeline.input(_SOURCE, t=str)))


def _register_linear(pipeline: compose.Pipeline) -> None:
    with pipeline:
        load(transform(extract(pipeline.input(_SOURCE, t=str))))


def _register_diamond(pipeline: compose.Pipeline) -> None:
    with pipeline:
        raw = extract(pipeline.input(_SOURCE, t=str))
        merge(branch_a(raw), branch_b(raw))


# =============================================================================
# Empty pipeline tests
# =============================================================================


def test_dag_empty_pipeline_ascii(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Empty pipeline shows empty graph placeholder in ASCII (default) format."""
    result = runner.invoke(cli.cli, ["dag"])

    assert result.exit_code == 0, result.output
    assert "(empty graph)" in result.output


def test_dag_empty_pipeline_mermaid(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Empty pipeline shows valid empty Mermaid flowchart."""
    result = runner.invoke(cli.cli, ["dag", "--mermaid"])

    assert result.exit_code == 0, result.output
    assert result.output.strip() == "flowchart TD"


def test_dag_empty_pipeline_dot(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Empty pipeline shows minimal DOT graph."""
    result = runner.invoke(cli.cli, ["dag", "--dot"])

    assert result.exit_code == 0, result.output
    assert "digraph {" in result.output
    assert "}" in result.output


def test_dag_empty_pipeline_md(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Empty pipeline shows Mermaid wrapped in markdown code block."""
    result = runner.invoke(cli.cli, ["dag", "--md"])

    assert result.exit_code == 0, result.output
    assert "```mermaid" in result.output
    assert "```" in result.output
    assert "flowchart TD" in result.output


# =============================================================================
# Single stage tests
# =============================================================================


def test_dag_single_stage_ascii(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Single stage shows its input and output artifacts in ASCII."""
    _register_extract(mock_discovery)

    result = runner.invoke(cli.cli, ["dag"])

    assert result.exit_code == 0, result.output
    assert _SOURCE in result.output
    assert "test/extract" in result.output


def test_dag_single_stage_stages_flag(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Single stage shows stage name with --stages flag."""
    _register_extract(mock_discovery)

    result = runner.invoke(cli.cli, ["dag", "--stages"])

    assert result.exit_code == 0, result.output
    assert "test/extract" in result.output


# =============================================================================
# Linear chain tests (A -> B -> C)
# =============================================================================


def test_dag_linear_chain_ascii(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Linear pipeline shows all artifacts in ASCII."""
    _register_linear(mock_discovery)

    result = runner.invoke(cli.cli, ["dag"])

    assert result.exit_code == 0, result.output
    assert _SOURCE in result.output
    assert "test/extract" in result.output
    assert "test/transform" in result.output
    assert "test/load" in result.output


def test_dag_linear_chain_stages(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Linear pipeline shows all stages with --stages flag."""
    _register_linear(mock_discovery)

    result = runner.invoke(cli.cli, ["dag", "--stages"])

    assert result.exit_code == 0, result.output
    assert "test/extract" in result.output
    assert "test/transform" in result.output
    assert "test/load" in result.output


def test_dag_linear_chain_mermaid(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Linear pipeline shows edges in Mermaid format."""
    _register_linear(mock_discovery)

    result = runner.invoke(cli.cli, ["dag", "--mermaid", "--stages"])

    assert result.exit_code == 0, result.output
    assert "flowchart TD" in result.output
    assert "test/extract" in result.output
    assert "test/transform" in result.output
    assert "test/load" in result.output
    assert result.output.count("-->") == 2, "Linear chain of 3 stages has 2 edges"


def test_dag_linear_chain_dot(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Linear pipeline shows edges in DOT format."""
    _register_linear(mock_discovery)

    result = runner.invoke(cli.cli, ["dag", "--dot", "--stages"])

    assert result.exit_code == 0, result.output
    assert "digraph {" in result.output
    assert '"test/extract" -> "test/transform"' in result.output
    assert '"test/transform" -> "test/load"' in result.output


# =============================================================================
# Diamond pattern tests (A -> B, A -> C, B -> D, C -> D)
# =============================================================================


def test_dag_diamond_pattern_stages(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Diamond pattern shows all stages."""
    _register_diamond(mock_discovery)

    result = runner.invoke(cli.cli, ["dag", "--stages"])

    assert result.exit_code == 0, result.output
    assert "test/extract" in result.output
    assert "test/branch_a" in result.output
    assert "test/branch_b" in result.output
    assert "test/merge" in result.output


def test_dag_diamond_pattern_mermaid(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Diamond pattern shows correct edges in Mermaid."""
    _register_diamond(mock_discovery)

    result = runner.invoke(cli.cli, ["dag", "--mermaid", "--stages"])

    assert result.exit_code == 0, result.output
    # Fan-out (2) and fan-in (2)
    assert result.output.count("-->") == 4


# =============================================================================
# Target filtering tests
# =============================================================================


def test_dag_target_single_stage(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Target filtering by stage name shows stage and its upstream."""
    _register_linear(mock_discovery)

    result = runner.invoke(cli.cli, ["dag", "--stages", "test/transform"])

    assert result.exit_code == 0, result.output
    # Should include extract (upstream) and transform
    assert "test/extract" in result.output
    assert "test/transform" in result.output
    # Should NOT include load (downstream of transform)
    assert "test/load" not in result.output


def test_dag_target_bare_stage_name(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """dag resolves a bare stage name to its qualified form (names.resolve_stage_name)."""
    _register_linear(mock_discovery)

    result = runner.invoke(cli.cli, ["dag", "--stages", "transform"])

    assert result.exit_code == 0, result.output
    assert "test/extract" in result.output
    assert "test/transform" in result.output
    assert "test/load" not in result.output


def test_dag_target_artifact_path(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Target filtering by artifact path resolves to producing stage and upstream."""
    _register_linear(mock_discovery)

    # transform's output lives at data/test/transform.txt (WorkspaceStore layout)
    result = runner.invoke(cli.cli, ["dag", "--stages", "data/test/transform.txt"])

    assert result.exit_code == 0, result.output
    # Should include transform (produces the file) and extract (upstream)
    assert "test/transform" in result.output
    assert "test/extract" in result.output
    # Should NOT include load (downstream)
    assert "test/load" not in result.output


def test_dag_target_multiple_targets(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Multiple targets includes all specified stages and their upstreams."""
    _register_diamond(mock_discovery)

    result = runner.invoke(cli.cli, ["dag", "--stages", "test/branch_a", "test/branch_b"])

    assert result.exit_code == 0, result.output
    # Should include extract (upstream of both)
    assert "test/extract" in result.output
    assert "test/branch_a" in result.output
    assert "test/branch_b" in result.output
    assert "test/merge" not in result.output


def test_dag_target_invalid_stage_error(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Invalid target that doesn't match any stage or artifact shows error."""
    _register_extract(mock_discovery)

    result = runner.invoke(cli.cli, ["dag", "nonexistent_stage"])

    # Should fail with informative error
    assert result.exit_code != 0
    assert "No stages found" in result.output or "Error" in result.output


# =============================================================================
# Output format tests
# =============================================================================


def test_dag_md_format_wraps_mermaid(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """--md flag wraps Mermaid output in markdown code block."""
    _register_extract_transform(mock_discovery)

    result = runner.invoke(cli.cli, ["dag", "--md", "--stages"])

    assert result.exit_code == 0, result.output
    # Should have opening code fence
    assert "```mermaid" in result.output
    # Should have closing code fence
    lines = result.output.strip().split("\n")
    assert lines[-1] == "```"
    # Should have Mermaid content inside
    assert "flowchart TD" in result.output


def test_dag_ascii_default_format(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Default output format is ASCII."""
    _register_extract_transform(mock_discovery)

    result = runner.invoke(cli.cli, ["dag"])

    assert result.exit_code == 0, result.output
    # ASCII format draws boxes
    assert "+" in result.output
    assert "flowchart" not in result.output
    assert "digraph" not in result.output


# =============================================================================
# --stages flag tests
# =============================================================================


def test_dag_artifact_view_default(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Default view shows artifacts (including raw inputs), not just stages."""
    _register_extract(mock_discovery)

    result = runner.invoke(cli.cli, ["dag"])

    assert result.exit_code == 0, result.output
    assert _SOURCE in result.output


def test_dag_stage_view_with_flag(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """--stages flag shows stage names instead of artifacts."""
    _register_extract(mock_discovery)

    result = runner.invoke(cli.cli, ["dag", "--stages"])

    assert result.exit_code == 0, result.output
    assert "test/extract" in result.output
    assert _SOURCE not in result.output


def test_dag_stages_flag_with_mermaid(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """--stages flag works with --mermaid format."""
    _register_extract_transform(mock_discovery)

    result = runner.invoke(cli.cli, ["dag", "--stages", "--mermaid"])

    assert result.exit_code == 0, result.output
    assert "test/extract" in result.output
    assert "test/transform" in result.output
    # Should not show artifacts in stage view
    assert _SOURCE not in result.output


def test_dag_stages_flag_with_dot(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """--stages flag works with --dot format."""
    _register_extract_transform(mock_discovery)

    result = runner.invoke(cli.cli, ["dag", "--stages", "--dot"])

    assert result.exit_code == 0, result.output
    assert "test/extract" in result.output
    assert "test/transform" in result.output
    assert _SOURCE not in result.output


def test_dag_target_partial_resolution(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Command succeeds with partial target resolution (valid + invalid targets)."""
    _register_extract_transform(mock_discovery)

    # Mix of valid and invalid targets - should succeed with valid targets
    result = runner.invoke(cli.cli, ["dag", "test/extract", "nonexistent_stage", "--stages"])

    assert result.exit_code == 0, result.output
    # Valid target should be in output
    assert "test/extract" in result.output
    # transform not requested, should not appear in filtered output
    assert "test/transform" not in result.output


# =============================================================================
# Flag Tests
# =============================================================================


def test_dag_accepts_all_flag(runner: click.testing.CliRunner) -> None:
    """Dag should accept the --all flag."""
    result = runner.invoke(cli.cli, ["dag", "--help"])
    assert result.exit_code == 0
    assert "--all" in result.output, "Dag should accept --all flag"
