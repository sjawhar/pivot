"""Tests for TUI diff panels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pivot import loaders, outputs, project
from pivot.storage import cache
from pivot.tui import diff_panels
from pivot.types import (
    ChangeType,
    CodeChange,
    DepChange,
    LockData,
    MetricValue,
    OutputChange,
    ParamChange,
    StageExplanation,
    StageStatus,
)

if TYPE_CHECKING:
    import pathlib

    from pytest_mock import MockerFixture

    from pivot.pipeline.pipeline import Pipeline
    from pivot.registry import RegistryStageInfo


# =============================================================================
# _get_indicator Tests
# =============================================================================


def test_get_indicator_modified() -> None:
    """_get_indicator returns yellow indicator for modified."""
    result = diff_panels._get_indicator(ChangeType.MODIFIED)
    assert "[yellow]" in result
    assert "~" in result


def test_get_indicator_added() -> None:
    """_get_indicator returns green indicator for added."""
    result = diff_panels._get_indicator(ChangeType.ADDED)
    assert "[green]" in result
    assert "+" in result


def test_get_indicator_removed() -> None:
    """_get_indicator returns red indicator for removed."""
    result = diff_panels._get_indicator(ChangeType.REMOVED)
    assert "[red]" in result
    assert "-" in result


def test_get_indicator_unchanged() -> None:
    """_get_indicator returns dim indicator for unchanged (None)."""
    result = diff_panels._get_indicator(None)
    assert "[dim]" in result


# =============================================================================
# _truncate_hash Tests
# =============================================================================


def test_truncate_hash_long() -> None:
    """_truncate_hash truncates long hash to 8 chars."""
    result = diff_panels._truncate_hash("a1b2c3d4e5f6g7h8")
    assert result == "a1b2c3d4"


def test_truncate_hash_short() -> None:
    """_truncate_hash preserves short hash."""
    result = diff_panels._truncate_hash("abc")
    assert result == "abc"


def test_truncate_hash_none() -> None:
    """_truncate_hash returns placeholder for None."""
    result = diff_panels._truncate_hash(None)
    assert result == "(none)"


def test_truncate_hash_custom_length() -> None:
    """_truncate_hash respects custom length."""
    result = diff_panels._truncate_hash("a1b2c3d4e5f6g7h8", length=4)
    assert result == "a1b2"


# =============================================================================
# _format_hash_change Tests
# =============================================================================


def test_format_hash_change_unchanged() -> None:
    """_format_hash_change shows unchanged for None change_type."""
    result = diff_panels._format_hash_change("abc123", "abc123", None)
    assert result == "(unchanged)"


def test_format_hash_change_added() -> None:
    """_format_hash_change shows none -> hash for added."""
    result = diff_panels._format_hash_change(None, "abc12345", ChangeType.ADDED)
    assert "(none)" in result
    assert "abc12345" in result
    assert "->" in result


def test_format_hash_change_removed() -> None:
    """_format_hash_change shows hash -> deleted for removed."""
    result = diff_panels._format_hash_change("abc12345", None, ChangeType.REMOVED)
    assert "abc12345" in result
    assert "(deleted)" in result
    assert "->" in result


def test_format_hash_change_modified() -> None:
    """_format_hash_change shows old -> new for modified."""
    result = diff_panels._format_hash_change("oldhashabc", "newhashxyz", ChangeType.MODIFIED)
    assert "oldhasha" in result, "Should truncate to 8 chars"
    assert "newhashx" in result, "Should truncate to 8 chars"
    assert "->" in result


# =============================================================================
# _get_relative_path Tests
# =============================================================================


def test_get_relative_path_absolute(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_get_relative_path converts absolute path inside project."""
    (tmp_path / ".pivot").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(project, "_project_root_cache", None)

    abs_path = str(tmp_path / "data" / "file.csv")
    result = diff_panels._get_relative_path(abs_path)
    assert result == "data/file.csv"


def test_get_relative_path_relative(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_get_relative_path preserves relative paths."""
    (tmp_path / ".pivot").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(project, "_project_root_cache", None)

    result = diff_panels._get_relative_path("some/relative/path.txt")
    assert result == "some/relative/path.txt"


# =============================================================================
# _compute_output_changes Tests
# =============================================================================


def test_compute_output_changes_no_lock_shows_added(tmp_path: pathlib.Path) -> None:
    """_compute_output_changes shows new outputs as ADDED when no lock exists."""
    output_file = tmp_path / "output.csv"
    output_file.write_text("data")

    registry_info: RegistryStageInfo = {
        "func": lambda: None,
        "name": "test_stage",
        "deps": {},
        "deps_paths": [],
        "outs": [outputs.Out(path=str(output_file), loader=loaders.PathOnly())],
        "outs_paths": [str(output_file)],
        "params": None,
        "mutex": [],
        "variant": None,
        "signature": None,
        "fingerprint": {},
        "dep_specs": {},
        "out_specs": {},
        "params_arg_name": None,
        "state_dir": None,
    }

    result = diff_panels.compute_output_changes(None, registry_info)

    assert len(result) == 1
    assert result[0]["path"] == str(output_file)
    assert result[0]["old_hash"] is None
    assert result[0]["new_hash"] is not None
    assert result[0]["change_type"] == ChangeType.ADDED
    assert result[0]["output_type"] == "out"


def test_compute_output_changes_missing_file_shows_removed(tmp_path: pathlib.Path) -> None:
    """_compute_output_changes shows missing files as REMOVED."""
    output_file = tmp_path / "missing.csv"

    registry_info: RegistryStageInfo = {
        "func": lambda: None,
        "name": "test_stage",
        "deps": {},
        "deps_paths": [],
        "outs": [outputs.Out(path=str(output_file), loader=loaders.PathOnly())],
        "outs_paths": [str(output_file)],
        "params": None,
        "mutex": [],
        "variant": None,
        "signature": None,
        "fingerprint": {},
        "dep_specs": {},
        "out_specs": {},
        "params_arg_name": None,
        "state_dir": None,
    }

    lock_data: LockData = {
        "code_manifest": {},
        "params": {},
        "dep_hashes": {},
        "output_hashes": {str(output_file): {"hash": "oldhash123"}},
        "dep_generations": {},
    }

    result = diff_panels.compute_output_changes(lock_data, registry_info)

    assert len(result) == 1
    assert result[0]["old_hash"] == "oldhash123"
    assert result[0]["new_hash"] is None
    assert result[0]["change_type"] == ChangeType.REMOVED


def test_compute_output_changes_unchanged(tmp_path: pathlib.Path) -> None:
    """_compute_output_changes returns None change_type for unchanged files."""
    output_file = tmp_path / "output.csv"
    output_file.write_text("data")

    actual_hash = cache.hash_file(output_file)

    registry_info: RegistryStageInfo = {
        "func": lambda: None,
        "name": "test_stage",
        "deps": {},
        "deps_paths": [],
        "outs": [outputs.Out(path=str(output_file), loader=loaders.PathOnly())],
        "outs_paths": [str(output_file)],
        "params": None,
        "mutex": [],
        "variant": None,
        "signature": None,
        "fingerprint": {},
        "dep_specs": {},
        "out_specs": {},
        "params_arg_name": None,
        "state_dir": None,
    }

    lock_data: LockData = {
        "code_manifest": {},
        "params": {},
        "dep_hashes": {},
        "output_hashes": {str(output_file): {"hash": actual_hash}},
        "dep_generations": {},
    }

    result = diff_panels.compute_output_changes(lock_data, registry_info)

    assert len(result) == 1
    assert result[0]["change_type"] is None, "Unchanged files should have None change_type"


def test_compute_output_changes_detects_output_types(tmp_path: pathlib.Path) -> None:
    """_compute_output_changes correctly identifies Out, Metric, Plot types."""
    out_file = tmp_path / "output.csv"
    metric_file = tmp_path / "metrics.json"
    plot_file = tmp_path / "plot.png"

    for f in [out_file, metric_file, plot_file]:
        f.write_text("data")

    registry_info: RegistryStageInfo = {
        "func": lambda: None,
        "name": "test_stage",
        "deps": {},
        "deps_paths": [],
        "outs": [
            outputs.Out(path=str(out_file), loader=loaders.PathOnly()),
            outputs.Metric(path=str(metric_file)),
            outputs.Plot(path=str(plot_file), loader=loaders.PathOnly()),
        ],
        "outs_paths": [str(out_file), str(metric_file), str(plot_file)],
        "params": None,
        "mutex": [],
        "variant": None,
        "signature": None,
        "fingerprint": {},
        "dep_specs": {},
        "out_specs": {},
        "params_arg_name": None,
        "state_dir": None,
    }

    result = diff_panels.compute_output_changes(None, registry_info)

    assert len(result) == 3
    types = {r["output_type"] for r in result}
    assert types == {"out", "metric", "plot"}


# =============================================================================
# Panel Initialization Tests
# =============================================================================


def test_input_diff_panel_init() -> None:
    """InputDiffPanel initializes with None stage_name."""
    panel = diff_panels.InputDiffPanel(id="test-panel")
    assert panel._stage_name is None


def test_input_diff_panel_set_stage_none() -> None:
    """InputDiffPanel.set_stage(None) sets stage_name to None."""
    panel = diff_panels.InputDiffPanel()
    panel.set_stage(None)
    assert panel._stage_name is None


def test_output_diff_panel_init() -> None:
    """OutputDiffPanel initializes with None stage_name."""
    panel = diff_panels.OutputDiffPanel(id="test-panel")
    assert panel._stage_name is None


def test_output_diff_panel_set_stage_none() -> None:
    """OutputDiffPanel.set_stage(None) sets stage_name to None."""
    panel = diff_panels.OutputDiffPanel()
    panel.set_stage(None)
    assert panel._stage_name is None


# =============================================================================
# OutputChange TypedDict Tests
# =============================================================================


def test_output_change_creation() -> None:
    """OutputChange TypedDict can be created with all fields."""
    change = OutputChange(
        path="/path/to/file.csv",
        old_hash="abc123",
        new_hash="def456",
        change_type=ChangeType.MODIFIED,
        output_type="out",
    )
    assert change["path"] == "/path/to/file.csv"
    assert change["old_hash"] == "abc123"
    assert change["new_hash"] == "def456"
    assert change["change_type"] == ChangeType.MODIFIED
    assert change["output_type"] == "out"


def test_output_change_with_metric_type() -> None:
    """OutputChange can have metric output_type."""
    change = OutputChange(
        path="/metrics.json",
        old_hash=None,
        new_hash="xyz789",
        change_type=ChangeType.ADDED,
        output_type="metric",
    )
    assert change["output_type"] == "metric"


def test_output_change_with_plot_type() -> None:
    """OutputChange can have plot output_type."""
    change = OutputChange(
        path="/plot.png",
        old_hash="old123",
        new_hash=None,
        change_type=ChangeType.REMOVED,
        output_type="plot",
    )
    assert change["output_type"] == "plot"


# =============================================================================
# _escape_padded Tests
# =============================================================================


def test_escape_padded_escapes_markup() -> None:
    """_escape_padded escapes Rich markup characters in padded text."""
    # Simple text without markup
    result = diff_panels._escape_padded("simple", 10)
    assert result == "simple    "
    assert len(result) == 10

    # Text with brackets (opening [ gets escaped as \[)
    result_with_brackets = diff_panels._escape_padded("[test]", 20)
    assert "\\[" in result_with_brackets


# =============================================================================
# _get_type_label Tests
# =============================================================================


@pytest.mark.parametrize(
    ("output_type", "expected"),
    [
        ("out", "Output"),
        ("metric", "Metric"),
        ("plot", "Plot"),
    ],
)
def test_get_type_label(output_type: diff_panels.OutputType, expected: str) -> None:
    """_get_type_label returns correct labels for output types."""
    panel = diff_panels.OutputDiffPanel()
    result = panel._get_type_label(output_type)
    assert result == expected


# =============================================================================
# _format_metric_value Tests
# =============================================================================


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(None, "(none)", id="none"),
        pytest.param(3.14159265, "3.1416", id="float"),
        pytest.param(42, "42", id="int"),
        pytest.param("text", "text", id="string"),
    ],
)
def test_format_metric_value(value: MetricValue, expected: str) -> None:
    """_format_metric_value formats metric values correctly."""
    panel = diff_panels.OutputDiffPanel()
    result = panel._format_metric_value(value)
    assert result == expected


# =============================================================================
# _format_metric_delta Tests
# =============================================================================


@pytest.mark.parametrize(
    ("old", "new", "expected_pattern"),
    [
        pytest.param(None, 1.0, "", id="none_old"),
        pytest.param(1.0, None, "", id="none_new"),
        pytest.param(1.0, 2.0, "+1.0000", id="float_increase"),
        pytest.param(5.0, 3.0, "-2.0000", id="float_decrease"),
        pytest.param(10, 15, "+5", id="int_increase"),
        pytest.param(20, 10, "-10", id="int_decrease"),
    ],
)
def test_format_metric_delta(old: MetricValue, new: MetricValue, expected_pattern: str) -> None:
    """_format_metric_delta formats deltas correctly."""
    panel = diff_panels.OutputDiffPanel()
    result = panel._format_metric_delta(old, new)
    if expected_pattern:
        assert expected_pattern in result
    else:
        assert result == ""


# =============================================================================
# InputDiffPanel _render_empty_state Tests
# =============================================================================


def test_input_panel_empty_state_no_stage() -> None:
    """InputDiffPanel._render_empty_state shows 'No stage selected' when stage is None."""
    panel = diff_panels.InputDiffPanel()
    panel._stage_name = None
    result = panel._render_empty_state()
    assert "No stage selected" in result


def test_input_panel_empty_state_no_registry() -> None:
    """InputDiffPanel._render_empty_state shows 'Stage not in registry' when registry_info is None."""
    panel = diff_panels.InputDiffPanel()
    panel._stage_name = "some_stage"
    panel._registry_info = None
    result = panel._render_empty_state()
    assert "Stage not in registry" in result


def test_input_panel_empty_state_no_explanation() -> None:
    """InputDiffPanel._render_empty_state shows 'Error loading' when explanation is None."""
    panel = diff_panels.InputDiffPanel()
    panel._stage_name = "some_stage"
    panel._registry_info = {
        "func": lambda: None,
        "name": "some_stage",
        "deps": {},
        "deps_paths": [],
        "outs": [],
        "outs_paths": [],
        "params": None,
        "mutex": [],
        "variant": None,
        "signature": None,
        "fingerprint": {},
        "dep_specs": {},
        "out_specs": {},
        "params_arg_name": None,
        "state_dir": None,
    }
    panel._explanation = None
    result = panel._render_empty_state()
    assert "Error loading" in result


def test_input_panel_empty_state_no_inputs() -> None:
    """InputDiffPanel._render_empty_state shows 'No inputs' when all fields set."""
    panel = diff_panels.InputDiffPanel()
    panel._stage_name = "some_stage"
    panel._registry_info = {
        "func": lambda: None,
        "name": "some_stage",
        "deps": {},
        "deps_paths": [],
        "outs": [],
        "outs_paths": [],
        "params": None,
        "mutex": [],
        "variant": None,
        "signature": None,
        "fingerprint": {},
        "dep_specs": {},
        "out_specs": {},
        "params_arg_name": None,
        "state_dir": None,
    }
    panel._explanation = StageExplanation(
        stage_name="some_stage",
        will_run=False,
        is_forced=False,
        reason="",
        code_changes=[],
        dep_changes=[],
        param_changes=[],
        upstream_stale=[],
    )
    result = panel._render_empty_state()
    assert "No inputs" in result


# =============================================================================
# OutputDiffPanel _render_empty_state Tests
# =============================================================================


def test_output_panel_empty_state_no_stage() -> None:
    """OutputDiffPanel._render_empty_state shows 'No stage selected' when stage is None."""
    panel = diff_panels.OutputDiffPanel()
    panel._stage_name = None
    result = panel._render_empty_state()
    assert "No stage selected" in result


def test_output_panel_empty_state_no_registry() -> None:
    """OutputDiffPanel._render_empty_state shows 'Stage not in registry' when registry_info is None."""
    panel = diff_panels.OutputDiffPanel()
    panel._stage_name = "some_stage"
    panel._registry_info = None
    result = panel._render_empty_state()
    assert "Stage not in registry" in result


def test_output_panel_empty_state_in_progress() -> None:
    """OutputDiffPanel._render_empty_state shows 'Running' when stage is IN_PROGRESS."""
    panel = diff_panels.OutputDiffPanel()
    panel._stage_name = "some_stage"
    panel._registry_info = {
        "func": lambda: None,
        "name": "some_stage",
        "deps": {},
        "deps_paths": [],
        "outs": [],
        "outs_paths": [],
        "params": None,
        "mutex": [],
        "variant": None,
        "signature": None,
        "fingerprint": {},
        "dep_specs": {},
        "out_specs": {},
        "params_arg_name": None,
        "state_dir": None,
    }
    panel._stage_status = StageStatus.IN_PROGRESS
    result = panel._render_empty_state()
    assert "running" in result.lower()


def test_output_panel_empty_state_no_outputs() -> None:
    """OutputDiffPanel._render_empty_state shows 'No outputs' by default."""
    panel = diff_panels.OutputDiffPanel()
    panel._stage_name = "some_stage"
    panel._registry_info = {
        "func": lambda: None,
        "name": "some_stage",
        "deps": {},
        "deps_paths": [],
        "outs": [],
        "outs_paths": [],
        "params": None,
        "mutex": [],
        "variant": None,
        "signature": None,
        "fingerprint": {},
        "dep_specs": {},
        "out_specs": {},
        "params_arg_name": None,
        "state_dir": None,
    }
    panel._stage_status = None
    result = panel._render_empty_state()
    assert "No outputs" in result


# =============================================================================
# InputDiffPanel _is_changed Tests
# =============================================================================


def test_input_panel_is_changed_code_key() -> None:
    """InputDiffPanel._is_changed returns True for code key in _code_by_key."""
    panel = diff_panels.InputDiffPanel()
    panel._code_by_key = {
        "func_name": CodeChange(
            key="func_name",
            change_type=ChangeType.MODIFIED,
            old_hash="abc",
            new_hash="def",
        )
    }
    panel._dep_by_path = {}
    panel._param_by_key = {}

    assert panel._is_changed("code:func_name") is True
    assert panel._is_changed("code:unknown") is False


def test_input_panel_is_changed_dep_path() -> None:
    """InputDiffPanel._is_changed returns True for dep path in _dep_by_path."""
    panel = diff_panels.InputDiffPanel()
    panel._code_by_key = {}
    panel._dep_by_path = {
        "/path/to/file.csv": DepChange(
            path="/path/to/file.csv",
            change_type=ChangeType.MODIFIED,
            old_hash="abc",
            new_hash="def",
        )
    }
    panel._param_by_key = {}

    assert panel._is_changed("dep:/path/to/file.csv") is True
    assert panel._is_changed("dep:/other/file.csv") is False


def test_input_panel_is_changed_param_key() -> None:
    """InputDiffPanel._is_changed returns True for param key in _param_by_key."""
    panel = diff_panels.InputDiffPanel()
    panel._code_by_key = {}
    panel._dep_by_path = {}
    panel._param_by_key = {
        "learning_rate": ParamChange(
            key="learning_rate",
            change_type=ChangeType.MODIFIED,
            old_value=0.01,
            new_value=0.001,
        )
    }

    assert panel._is_changed("param:learning_rate") is True
    assert panel._is_changed("param:unknown") is False


def test_input_panel_is_changed_unknown_type() -> None:
    """InputDiffPanel._is_changed returns False for unknown item type."""
    panel = diff_panels.InputDiffPanel()
    panel._code_by_key = {}
    panel._dep_by_path = {}
    panel._param_by_key = {}

    assert panel._is_changed("unknown:something") is False


# =============================================================================
# OutputDiffPanel _is_changed Tests
# =============================================================================


def test_output_panel_is_changed_with_change() -> None:
    """OutputDiffPanel._is_changed returns True for output with change_type."""
    panel = diff_panels.OutputDiffPanel()
    panel._output_by_path = {
        "/path/to/output.csv": OutputChange(
            path="/path/to/output.csv",
            old_hash="abc",
            new_hash="def",
            change_type=ChangeType.MODIFIED,
            output_type="out",
        )
    }

    assert panel._is_changed("out:/path/to/output.csv") is True


def test_output_panel_is_changed_unchanged() -> None:
    """OutputDiffPanel._is_changed returns False for output without change_type."""
    panel = diff_panels.OutputDiffPanel()
    panel._output_by_path = {
        "/path/to/output.csv": OutputChange(
            path="/path/to/output.csv",
            old_hash="abc",
            new_hash="abc",
            change_type=None,  # Unchanged
            output_type="out",
        )
    }

    assert panel._is_changed("out:/path/to/output.csv") is False


def test_output_panel_is_changed_not_found() -> None:
    """OutputDiffPanel._is_changed returns False for unknown path."""
    panel = diff_panels.OutputDiffPanel()
    panel._output_by_path = {}

    assert panel._is_changed("out:/unknown/path.csv") is False


# =============================================================================
# set_from_snapshot Tests
# =============================================================================


def test_input_panel_set_from_snapshot(
    test_pipeline: Pipeline, mock_discovery: Pipeline, mocker: MockerFixture
) -> None:
    """InputDiffPanel.set_from_snapshot loads snapshot data correctly."""
    panel = diff_panels.InputDiffPanel()

    # Mock _update_display to verify it's called
    mock_update = mocker.patch.object(panel, "_update_display", autospec=True)
    snapshot = StageExplanation(
        stage_name="test_stage",
        will_run=True,
        is_forced=False,
        reason="Code changed",
        code_changes=[
            CodeChange(
                key="func",
                change_type=ChangeType.MODIFIED,
                old_hash="old",
                new_hash="new",
            )
        ],
        dep_changes=[],
        param_changes=[],
        upstream_stale=[],
    )
    panel.set_from_snapshot(snapshot)

    assert panel._stage_name == "test_stage"
    assert panel._explanation == snapshot
    assert "func" in panel._code_by_key
    mock_update.assert_called_once()


def test_output_panel_set_from_snapshot(
    test_pipeline: Pipeline, mock_discovery: Pipeline, mocker: MockerFixture
) -> None:
    """OutputDiffPanel.set_from_snapshot loads snapshot data correctly."""
    panel = diff_panels.OutputDiffPanel()

    # Mock _update_display to verify it's called
    mock_update = mocker.patch.object(panel, "_update_display", autospec=True)
    changes = [
        OutputChange(
            path="/path/output.csv",
            old_hash="old",
            new_hash="new",
            change_type=ChangeType.MODIFIED,
            output_type="out",
        )
    ]
    panel.set_from_snapshot("test_stage", changes)

    assert panel._stage_name == "test_stage"
    assert "/path/output.csv" in panel._output_by_path
    mock_update.assert_called_once()
