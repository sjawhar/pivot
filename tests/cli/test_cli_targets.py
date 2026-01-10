from __future__ import annotations

from typing import TYPE_CHECKING

import click
import pytest

from pivot import outputs, registry
from pivot.cli import targets

if TYPE_CHECKING:
    import pathlib

    from pytest_mock import MockerFixture


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
    set_project_root: pathlib.Path,
    mocker: MockerFixture,
) -> None:
    """Target that is only a stage name."""
    mocker.patch.object(registry.REGISTRY, "list_stages", return_value=["my_stage"])

    result = targets._classify_targets(["my_stage"], set_project_root)

    assert len(result) == 1
    assert result[0]["target"] == "my_stage"
    assert result[0]["is_stage"] is True
    assert result[0]["is_file"] is False


def test_classify_targets_file_only(set_project_root: pathlib.Path, mocker: MockerFixture) -> None:
    """Target that is only a file path."""
    mocker.patch.object(registry.REGISTRY, "list_stages", return_value=[])
    data_file = set_project_root / "data.csv"
    data_file.touch()

    result = targets._classify_targets(["data.csv"], set_project_root)

    assert len(result) == 1
    assert result[0]["target"] == "data.csv"
    assert result[0]["is_stage"] is False
    assert result[0]["is_file"] is True


def test_classify_targets_neither(set_project_root: pathlib.Path, mocker: MockerFixture) -> None:
    """Target that is neither a stage nor existing file."""
    mocker.patch.object(registry.REGISTRY, "list_stages", return_value=[])

    result = targets._classify_targets(["nonexistent"], set_project_root)

    assert len(result) == 1
    assert result[0]["is_stage"] is False
    assert result[0]["is_file"] is False


def test_classify_targets_both_warns(
    set_project_root: pathlib.Path,
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Target that is both a stage name and file should warn."""
    mocker.patch.object(registry.REGISTRY, "list_stages", return_value=["data"])
    data_file = set_project_root / "data"
    data_file.touch()

    result = targets._classify_targets(["data"], set_project_root)

    assert len(result) == 1
    assert result[0]["is_stage"] is True
    assert result[0]["is_file"] is True
    assert "matches both a stage name and a file path" in caplog.text


# --- resolve_output_paths tests ---


def test_resolve_output_paths_stage(
    set_project_root: pathlib.Path,
    mocker: MockerFixture,
) -> None:
    """Resolving a stage target should return its output paths."""
    mocker.patch.object(registry.REGISTRY, "list_stages", return_value=["my_stage"])
    mocker.patch.object(
        registry.REGISTRY,
        "get",
        return_value={
            "func": lambda: None,
            "deps": [],
            "outs": [outputs.Metric(path="metrics.yaml")],
            "params": [],
            "name": "my_stage",
            "cwd": None,
        },
    )

    resolved, missing = targets.resolve_output_paths(["my_stage"], set_project_root, outputs.Metric)

    assert "metrics.yaml" in resolved
    assert missing == []


def test_resolve_output_paths_file(set_project_root: pathlib.Path, mocker: MockerFixture) -> None:
    """Resolving a file target should return the file path."""
    mocker.patch.object(registry.REGISTRY, "list_stages", return_value=[])
    metrics_file = set_project_root / "my_metrics.yaml"
    metrics_file.touch()

    resolved, missing = targets.resolve_output_paths(
        ["my_metrics.yaml"], set_project_root, outputs.Metric
    )

    assert "my_metrics.yaml" in resolved
    assert missing == []


def test_resolve_output_paths_unknown(
    set_project_root: pathlib.Path, mocker: MockerFixture
) -> None:
    """Unknown targets should be returned in missing list."""
    mocker.patch.object(registry.REGISTRY, "list_stages", return_value=[])

    resolved, missing = targets.resolve_output_paths(
        ["nonexistent.yaml"], set_project_root, outputs.Metric
    )

    assert len(resolved) == 0
    assert missing == ["nonexistent.yaml"]


def test_resolve_output_paths_filters_by_type(
    set_project_root: pathlib.Path,
    mocker: MockerFixture,
) -> None:
    """Only outputs matching the specified type should be included."""
    mocker.patch.object(registry.REGISTRY, "list_stages", return_value=["mixed_stage"])
    mocker.patch.object(
        registry.REGISTRY,
        "get",
        return_value={
            "func": lambda: None,
            "deps": [],
            "outs": [
                outputs.Metric(path="metrics.yaml"),
                outputs.Plot(path="plot.png"),
            ],
            "params": [],
            "name": "mixed_stage",
            "cwd": None,
        },
    )

    resolved, _ = targets.resolve_output_paths(["mixed_stage"], set_project_root, outputs.Metric)

    assert "metrics.yaml" in resolved
    assert "plot.png" not in resolved


# --- resolve_plot_infos tests ---


def test_resolve_plot_infos_stage(
    set_project_root: pathlib.Path,
    mocker: MockerFixture,
) -> None:
    """Resolving a stage target should return PlotInfo with metadata."""
    mocker.patch.object(registry.REGISTRY, "list_stages", return_value=["plot_stage"])
    mocker.patch.object(
        registry.REGISTRY,
        "get",
        return_value={
            "func": lambda: None,
            "deps": [],
            "outs": [outputs.Plot(path="train_loss.png", x="epoch", y="loss")],
            "params": [],
            "name": "plot_stage",
            "cwd": None,
        },
    )

    resolved, missing = targets.resolve_plot_infos(["plot_stage"], set_project_root)

    assert len(resolved) == 1
    assert resolved[0]["path"] == "train_loss.png"
    assert resolved[0]["stage_name"] == "plot_stage"
    assert resolved[0]["x"] == "epoch"
    assert resolved[0]["y"] == "loss"
    assert missing == []


def test_resolve_plot_infos_file(set_project_root: pathlib.Path, mocker: MockerFixture) -> None:
    """Resolving a file target should return PlotInfo with (direct) stage."""
    mocker.patch.object(registry.REGISTRY, "list_stages", return_value=[])
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


def test_resolve_and_validate_empty_targets(set_project_root: pathlib.Path) -> None:
    result = targets.resolve_and_validate((), set_project_root, outputs.Metric)

    assert result is None


def test_resolve_and_validate_raises_on_unknown(
    set_project_root: pathlib.Path,
    mocker: MockerFixture,
) -> None:
    """Should raise ClickException with helpful message for unknown targets."""
    mocker.patch.object(registry.REGISTRY, "list_stages", return_value=[])

    with pytest.raises(click.ClickException) as exc_info:
        targets.resolve_and_validate(("nonexistent.yaml",), set_project_root, outputs.Metric)

    assert "neither a registered stage nor an existing file" in str(exc_info.value)


def test_resolve_and_validate_returns_paths(
    set_project_root: pathlib.Path,
    mocker: MockerFixture,
) -> None:
    """Should return resolved paths on success."""
    mocker.patch.object(registry.REGISTRY, "list_stages", return_value=[])
    metrics_file = set_project_root / "data.yaml"
    metrics_file.touch()

    result = targets.resolve_and_validate(("data.yaml",), set_project_root, outputs.Metric)

    assert result is not None
    assert "data.yaml" in result
