from __future__ import annotations

import pathlib

from pivot_tui import diff as data_tui
from pivot.types import ChangeType, DataDiffResult, RowChange, SchemaChange

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
    """DataDiffApp should initialize with diff entries."""
    from pivot.show.data import DataDiffEntry

    entries = [
        DataDiffEntry(
            path="data.csv", old_hash="abc", new_hash="def", change_type=ChangeType.MODIFIED
        ),
    ]
    app = data_tui.DataDiffApp(entries, key_cols=None, max_rows=1000)
    assert app._diff_entries == entries
    assert app._key_cols is None
    assert app._max_rows == 1000
    assert app._current_idx == 0


def test_data_diff_app_with_key_cols() -> None:
    """DataDiffApp should store key columns."""
    from pivot.show.data import DataDiffEntry

    entries = [
        DataDiffEntry(
            path="data.csv", old_hash="abc", new_hash="def", change_type=ChangeType.MODIFIED
        ),
    ]
    app = data_tui.DataDiffApp(entries, key_cols=["id", "name"], max_rows=5000)
    assert app._key_cols == ["id", "name"]
    assert app._max_rows == 5000


# =============================================================================
# run_diff_app Tests
# =============================================================================


def test_run_diff_app_callable() -> None:
    """run_diff_app should be callable."""
    # Just verify the function exists and has correct signature
    assert callable(data_tui.run_diff_app)


def test_data_diff_app_cleanup_temp_files(tmp_path: pathlib.Path) -> None:
    """cleanup_temp_files should remove temp files."""
    from pivot.show.data import DataDiffEntry

    # Create temp files
    temp1 = tmp_path / "temp1.csv"
    temp2 = tmp_path / "temp2.csv"
    temp1.write_text("data")
    temp2.write_text("data")

    entries = [
        DataDiffEntry(
            path="data.csv", old_hash="abc", new_hash="def", change_type=ChangeType.MODIFIED
        ),
    ]
    app = data_tui.DataDiffApp(entries, key_cols=None, max_rows=1000)
    app._temp_files = [temp1, temp2]

    # Verify files exist before cleanup
    assert temp1.exists()
    assert temp2.exists()

    # Call cleanup
    app.cleanup_temp_files()

    # Verify files are removed
    assert not temp1.exists()
    assert not temp2.exists()


def test_data_diff_app_cleanup_temp_files_missing_ok() -> None:
    """cleanup_temp_files should handle missing files gracefully."""
    from pivot.show.data import DataDiffEntry

    entries = [
        DataDiffEntry(
            path="data.csv", old_hash="abc", new_hash="def", change_type=ChangeType.MODIFIED
        ),
    ]
    app = data_tui.DataDiffApp(entries, key_cols=None, max_rows=1000)

    # Add non-existent file to temp_files
    app._temp_files = [pathlib.Path("/nonexistent/file.csv")]

    # Should not raise even though file doesn't exist
    app.cleanup_temp_files()
