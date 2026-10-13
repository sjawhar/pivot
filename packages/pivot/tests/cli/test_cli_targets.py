from __future__ import annotations

from typing import TYPE_CHECKING

import click
import pytest

from pivot import compose, outputs
from pivot.cli import helpers as cli_helpers
from pivot.cli import targets
from pivot.engine import graph as engine_graph

if TYPE_CHECKING:
    from pathlib import Path

    import networkx as nx

# =============================================================================
# Module-level stage functions (required for pickling)
# =============================================================================


@compose.stage
def my_stage() -> str:
    return "data"


@compose.stage
def data() -> None:
    """Stage whose name collides with a file path."""


@compose.stage
def producer_stage() -> str:
    return "data"


@compose.stage
def stage_a() -> str:
    return "data"


@compose.stage
def stage_b() -> dict[str, int]:
    return {}


def _bipartite_graph() -> nx.DiGraph[str]:
    return engine_graph.build_graph(cli_helpers.get_all_stages())


# --- validate_targets tests ---


def test_validate_targets_empty_tuple() -> None:
    result = targets.validate_targets(())
    assert result == []


def test_validate_targets_filters_whitespace() -> None:
    result = targets.validate_targets(("valid", "", "  ", "also_valid"))
    assert result == ["valid", "also_valid"]


def test_validate_targets_raises_if_all_whitespace() -> None:
    with pytest.raises(targets.TargetValidationError) as exc_info:
        targets.validate_targets(("", "  ", "\t"))

    assert "All targets are empty or whitespace-only" in str(exc_info.value)


def test_validate_targets_logs_warning_for_invalid(caplog: pytest.LogCaptureFixture) -> None:
    targets.validate_targets(("valid", "", "also_valid"))

    assert "Ignoring 1 empty/whitespace-only target(s)" in caplog.text


# --- _classify_targets tests ---


def test_classify_targets_stage_only(
    mock_discovery: compose.Pipeline, set_project_root: Path
) -> None:
    """Target that is only a stage name (bare name resolves to its qualified form)."""
    with mock_discovery:
        my_stage()

    result = targets._classify_targets(["my_stage"], set_project_root)

    assert len(result) == 1
    assert result[0]["target"] == "test/my_stage"
    assert result[0]["is_stage"] is True
    assert result[0]["is_file"] is False


def test_classify_targets_file_only(
    mock_discovery: compose.Pipeline, set_project_root: Path
) -> None:
    """Target that is only a file path."""
    data_file = set_project_root / "data.csv"
    data_file.touch()

    result = targets._classify_targets(["data.csv"], set_project_root)

    assert len(result) == 1
    assert result[0]["target"] == "data.csv"
    assert result[0]["is_stage"] is False
    assert result[0]["is_file"] is True


def test_classify_targets_neither(mock_discovery: compose.Pipeline, set_project_root: Path) -> None:
    """Target that is neither a stage nor existing file."""
    result = targets._classify_targets(["nonexistent"], set_project_root)

    assert len(result) == 1
    assert result[0]["is_stage"] is False
    assert result[0]["is_file"] is False


def test_classify_targets_both_warns(
    mock_discovery: compose.Pipeline,
    set_project_root: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Target that is both a stage name and file should warn."""
    # Register stage with same name as file
    with mock_discovery:
        data()
    data_file = set_project_root / "data"
    data_file.touch()

    result = targets._classify_targets(["data"], set_project_root)

    assert len(result) == 1
    assert result[0]["is_stage"] is True
    assert result[0]["is_file"] is True
    assert "matches both a stage name and a file path" in caplog.text


# --- resolve_output_paths tests ---


def test_resolve_output_paths_file(
    mock_discovery: compose.Pipeline, set_project_root: Path
) -> None:
    """Resolving a file target should return the file path."""
    metrics_file = set_project_root / "my_metrics.yaml"
    metrics_file.touch()

    resolved, missing = targets.resolve_output_paths(
        ["my_metrics.yaml"], set_project_root, outputs.Metric
    )

    assert "my_metrics.yaml" in resolved
    assert missing == []


def test_resolve_output_paths_unknown(
    mock_discovery: compose.Pipeline, set_project_root: Path
) -> None:
    """Unknown targets should be returned in missing list."""
    resolved, missing = targets.resolve_output_paths(
        ["nonexistent.yaml"], set_project_root, outputs.Metric
    )

    assert len(resolved) == 0
    assert missing == ["nonexistent.yaml"]


# --- resolve_plot_infos tests ---


def test_resolve_plot_infos_file(mock_discovery: compose.Pipeline, set_project_root: Path) -> None:
    """Resolving a file target should return PlotInfo with (direct) stage."""
    plot_file = set_project_root / "my_plot.png"
    plot_file.touch()

    resolved, missing = targets.resolve_plot_infos(["my_plot.png"], set_project_root)

    assert len(resolved) == 1
    assert resolved[0]["path"] == "my_plot.png"
    assert resolved[0]["stage_name"] == "(direct)"
    assert resolved[0]["x"] is None
    assert resolved[0]["y"] is None
    assert missing == []


# --- _format_unknown_targets_error tests ---


def test_format_unknown_targets_error_single() -> None:
    result = targets._format_unknown_targets_error(["missing.yaml"])

    assert result == "Target 'missing.yaml' is neither a registered stage nor an existing file"


def test_format_unknown_targets_error_multiple() -> None:
    result = targets._format_unknown_targets_error(["missing.yaml", "other.csv"])

    assert "Targets 'missing.yaml', 'other.csv'" in result
    assert "neither registered stages nor existing files" in result


# --- resolve_and_validate tests ---


def test_resolve_and_validate_empty_targets(
    mock_discovery: compose.Pipeline, set_project_root: Path
) -> None:
    result = targets.resolve_and_validate((), set_project_root, outputs.Metric)

    assert result is None


def test_resolve_and_validate_raises_on_unknown(
    mock_discovery: compose.Pipeline, set_project_root: Path
) -> None:
    """Should raise ClickException with helpful message for unknown targets."""
    with pytest.raises(click.ClickException) as exc_info:
        targets.resolve_and_validate(("nonexistent.yaml",), set_project_root, outputs.Metric)

    assert "neither a registered stage nor an existing file" in str(exc_info.value)


def test_resolve_and_validate_returns_paths(
    mock_discovery: compose.Pipeline, set_project_root: Path
) -> None:
    """Should return resolved paths on success."""
    metrics_file = set_project_root / "data.yaml"
    metrics_file.touch()

    result = targets.resolve_and_validate(("data.yaml",), set_project_root, outputs.Metric)

    assert result is not None
    assert "data.yaml" in result


# --- resolve_targets_to_stages tests ---


def test_resolve_targets_to_stages_stage_name(
    mock_discovery: compose.Pipeline, set_project_root: Path
) -> None:
    """Stage name resolves directly to itself."""
    with mock_discovery:
        my_stage()

    resolved, unresolved = targets.resolve_targets_to_stages(["test/my_stage"], _bipartite_graph())

    assert resolved == {"test/my_stage"}, "Stage name should resolve directly"
    assert unresolved == [], "Should have no unresolved targets"


def test_resolve_targets_to_stages_bare_stage_name(
    mock_discovery: compose.Pipeline, set_project_root: Path
) -> None:
    """Bare stage name resolves to its qualified form."""
    with mock_discovery:
        my_stage()

    resolved, unresolved = targets.resolve_targets_to_stages(["my_stage"], _bipartite_graph())

    assert resolved == {"test/my_stage"}
    assert unresolved == []


def test_resolve_targets_to_stages_file_path(
    mock_discovery: compose.Pipeline, set_project_root: Path
) -> None:
    """Output file path resolves to producer stage."""
    with mock_discovery:
        producer_stage()

    resolved, unresolved = targets.resolve_targets_to_stages(
        ["data/test/producer_stage.txt"], _bipartite_graph()
    )

    assert resolved == {"test/producer_stage"}, "File path should resolve to producer stage"
    assert unresolved == [], "Should have no unresolved targets"


def test_resolve_targets_to_stages_unknown(
    mock_discovery: compose.Pipeline, set_project_root: Path
) -> None:
    """Unknown target goes to unresolved list."""
    with mock_discovery:
        producer_stage()

    resolved, unresolved = targets.resolve_targets_to_stages(
        ["nonexistent.csv"], _bipartite_graph()
    )

    assert resolved == set(), "Should have no resolved stages"
    assert "nonexistent.csv" in unresolved, "Unknown target should be in unresolved"


def test_resolve_targets_to_stages_mixed(
    mock_discovery: compose.Pipeline, set_project_root: Path
) -> None:
    """Stage names and file paths mix correctly."""
    with mock_discovery:
        stage_a()
        stage_b()

    resolved, unresolved = targets.resolve_targets_to_stages(
        ["test/stage_a", "data/test/stage_b.yaml"], _bipartite_graph()
    )

    assert resolved == {"test/stage_a", "test/stage_b"}, (
        "Stage name and stage_b's output path should both resolve"
    )
    assert unresolved == [], "Should have no unresolved targets"


def test_resolve_targets_to_stages_dedup(
    mock_discovery: compose.Pipeline, set_project_root: Path
) -> None:
    """Duplicate paths to same stage deduplicate."""
    with mock_discovery:
        my_stage()

    # Resolve both the stage name and its output file
    resolved, unresolved = targets.resolve_targets_to_stages(
        ["test/my_stage", "data/test/my_stage.txt"], _bipartite_graph()
    )

    assert resolved == {"test/my_stage"}, "Should deduplicate to single stage"
    assert unresolved == [], "Should have no unresolved targets"


def test_resolve_targets_to_stages_absolute_path(
    mock_discovery: compose.Pipeline, set_project_root: Path
) -> None:
    """Absolute path to output file resolves to producer stage."""
    with mock_discovery:
        producer_stage()

    abs_path = str(set_project_root / "data" / "test" / "producer_stage.txt")
    resolved, unresolved = targets.resolve_targets_to_stages([abs_path], _bipartite_graph())

    assert resolved == {"test/producer_stage"}, "Absolute path should resolve to producer stage"
    assert unresolved == [], "Should have no unresolved targets"


def test_resolve_targets_to_stages_dot_relative_path(
    mock_discovery: compose.Pipeline, set_project_root: Path
) -> None:
    """Dot-relative path (./data/test/producer_stage.txt) resolves to producer stage."""
    with mock_discovery:
        producer_stage()

    resolved, unresolved = targets.resolve_targets_to_stages(
        ["./data/test/producer_stage.txt"], _bipartite_graph()
    )

    assert resolved == {"test/producer_stage"}, "Dot-relative path should resolve to producer stage"
    assert unresolved == [], "Should have no unresolved targets"
