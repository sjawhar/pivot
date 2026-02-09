from __future__ import annotations

from pivot.types import ChangeType, DataDiffResult, RowChange, SchemaChange
from pivot_tui import diff as data_tui

# =============================================================================
# DiffSummaryPanel Tests
# =============================================================================


def test_diff_summary_panel_basic() -> None:
    """DiffSummaryPanel should initialize without error."""
    result = DataDiffResult(
        path="data.csv",
        old_rows=100,
        new_rows=150,
        old_cols=["id", "name"],
        new_cols=["id", "name", "age"],
        schema_changes=[],
        row_changes=[],
        reorder_only=False,
        truncated=False,
        summary_only=False,
    )
    panel = data_tui.DiffSummaryPanel(result)
    # Verify panel is correct widget type
    assert isinstance(panel, data_tui.DiffSummaryPanel)


def test_diff_summary_panel_reorder_only() -> None:
    """DiffSummaryPanel should handle reorder-only case."""
    result = DataDiffResult(
        path="data.csv",
        old_rows=100,
        new_rows=100,
        old_cols=["id", "name"],
        new_cols=["id", "name"],
        schema_changes=[],
        row_changes=[],
        reorder_only=True,
        truncated=False,
        summary_only=False,
    )
    panel = data_tui.DiffSummaryPanel(result)
    assert isinstance(panel, data_tui.DiffSummaryPanel)


# =============================================================================
# SchemaChangesTable Tests
# =============================================================================


def test_schema_changes_table_init() -> None:
    """SchemaChangesTable should initialize without error."""
    changes = [
        SchemaChange(column="age", old_dtype=None, new_dtype="int64", change_type=ChangeType.ADDED),
        SchemaChange(
            column="status", old_dtype="str", new_dtype=None, change_type=ChangeType.REMOVED
        ),
    ]
    table = data_tui.SchemaChangesTable(changes)
    assert isinstance(table, data_tui.SchemaChangesTable)


# =============================================================================
# RowChangesTable Tests
# =============================================================================


def test_row_changes_table_init() -> None:
    """RowChangesTable should initialize without error."""
    changes = [
        RowChange(
            key="1",
            change_type=ChangeType.ADDED,
            old_values=None,
            new_values={"id": "1", "name": "alice"},
        ),
        RowChange(
            key="2",
            change_type=ChangeType.MODIFIED,
            old_values={"id": "2", "name": "bob"},
            new_values={"id": "2", "name": "bobby"},
        ),
    ]
    columns = ["id", "name"]
    table = data_tui.RowChangesTable(changes, columns)
    assert isinstance(table, data_tui.RowChangesTable)


# =============================================================================
# FileDiffScreen Tests
# =============================================================================


def test_file_diff_screen_init() -> None:
    """FileDiffScreen should initialize with result."""
    result = DataDiffResult(
        path="test.csv",
        old_rows=10,
        new_rows=12,
        old_cols=["a", "b"],
        new_cols=["a", "b"],
        schema_changes=[],
        row_changes=[],
        reorder_only=False,
        truncated=False,
        summary_only=False,
    )
    screen = data_tui.FileDiffScreen(result)
    assert screen._result == result


# =============================================================================
# DataDiffApp Tests
# =============================================================================


def test_data_diff_app_init() -> None:
    """DataDiffApp should initialize with pre-computed results."""
    results = [
        DataDiffResult(
            path="data.csv",
            old_rows=100,
            new_rows=110,
            old_cols=["id", "name"],
            new_cols=["id", "name"],
            schema_changes=[],
            row_changes=[],
            reorder_only=False,
            truncated=False,
            summary_only=False,
        ),
    ]
    app = data_tui.DataDiffApp(results)
    assert app._results == results
    assert app._current_idx == 0


def test_data_diff_app_multiple_results() -> None:
    """DataDiffApp should store multiple results."""
    results = [
        DataDiffResult(
            path="a.csv",
            old_rows=10,
            new_rows=20,
            old_cols=["x"],
            new_cols=["x"],
            schema_changes=[],
            row_changes=[],
            reorder_only=False,
            truncated=False,
            summary_only=False,
        ),
        DataDiffResult(
            path="b.csv",
            old_rows=5,
            new_rows=5,
            old_cols=["y"],
            new_cols=["y"],
            schema_changes=[],
            row_changes=[],
            reorder_only=True,
            truncated=False,
            summary_only=False,
        ),
    ]
    app = data_tui.DataDiffApp(results)
    assert len(app._results) == 2
    assert app._results[0]["path"] == "a.csv"
    assert app._results[1]["path"] == "b.csv"


# =============================================================================
# run_diff_app Tests
# =============================================================================


def test_run_diff_app_callable() -> None:
    """run_diff_app should be callable."""
    # Just verify the function exists and has correct signature
    assert callable(data_tui.run_diff_app)
