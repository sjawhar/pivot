"""Tests for TUI diff panels."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pivot import loaders, outputs, project
from pivot.storage import cache
from pivot.tui import diff_panels
from pivot.types import ChangeType, LockData, OutputChange

if TYPE_CHECKING:
    import pathlib

    import pytest

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
# _get_registry_info Tests
# =============================================================================


def test_get_registry_info_missing_stage() -> None:
    """_get_registry_info returns None for missing stage."""
    result = diff_panels._get_registry_info("nonexistent_stage_xyz")
    assert result is None


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
        "cwd": None,
        "dep_specs": {},
        "out_path_overrides": None,
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
        "cwd": None,
        "dep_specs": {},
        "out_path_overrides": None,
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
        "cwd": None,
        "dep_specs": {},
        "out_path_overrides": None,
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
            outputs.Plot(path=str(plot_file)),
        ],
        "outs_paths": [str(out_file), str(metric_file), str(plot_file)],
        "params": None,
        "mutex": [],
        "variant": None,
        "signature": None,
        "fingerprint": {},
        "cwd": None,
        "dep_specs": {},
        "out_path_overrides": None,
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
