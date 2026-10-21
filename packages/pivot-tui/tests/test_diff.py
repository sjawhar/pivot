from __future__ import annotations

from typing import override

import pytest
import textual.app
import textual.widgets

from pivot.types import ChangeType, DataDiffResult, RowChange, SchemaChange
from pivot_tui import diff as data_tui

# =============================================================================
# DiffSummaryPanel Tests
# =============================================================================


def test_diff_summary_panel_basic() -> None:
    """DiffSummaryPanel should initialize without error."""
    result = DataDiffResult(
        path="data.csv",
        change_type=ChangeType.MODIFIED,
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
        change_type=ChangeType.MODIFIED,
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


class _SummaryPanelApp(textual.app.App[None]):
    _result: DataDiffResult

    def __init__(self, result: DataDiffResult) -> None:
        super().__init__()
        self._result = result

    @override
    def compose(self) -> textual.app.ComposeResult:
        yield data_tui.DiffSummaryPanel(self._result)


_CACHE_NOTE = "Its git HEAD content is not in the local cache; run pivot pull to compare it"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("change_type", "old_rows", "new_rows", "rows_line", "note"),
    [
        pytest.param(ChangeType.MODIFIED, None, 3, "Rows: unknown -> 3", True, id="modified-miss"),
        pytest.param(ChangeType.REMOVED, None, None, "Rows: unknown", True, id="removed-miss"),
        pytest.param(ChangeType.MODIFIED, 2, 3, "Rows: 2 -> 3", False, id="modified-known"),
        pytest.param(ChangeType.REMOVED, 2, None, "Rows: 2", False, id="removed-known"),
        pytest.param(ChangeType.ADDED, None, 3, "Rows: 3", False, id="added"),
    ],
)
async def test_diff_summary_panel_shows_the_cache_note_only_for_an_unknown_head_side(
    change_type: ChangeType,
    old_rows: int | None,
    new_rows: int | None,
    rows_line: str,
    note: bool,
) -> None:
    """A HEAD side whose rows are unknown (a cache miss) shows the `pivot pull` note; a known
    HEAD side, and an added file, which has no HEAD side, do not."""
    result = DataDiffResult(
        path="data.csv",
        change_type=change_type,
        old_rows=old_rows,
        new_rows=new_rows,
        old_cols=None if old_rows is None else ["v"],
        new_cols=None if new_rows is None else ["v"],
        schema_changes=[],
        row_changes=[],
        reorder_only=False,
        truncated=False,
        summary_only=True,
    )
    app = _SummaryPanelApp(result)
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(data_tui.DiffSummaryPanel)
        text = str(panel.query_one(textual.widgets.Static).content)

    assert rows_line in text, text
    assert (_CACHE_NOTE in text) is note, text


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
        change_type=ChangeType.MODIFIED,
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
            change_type=ChangeType.MODIFIED,
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
            change_type=ChangeType.MODIFIED,
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
            change_type=ChangeType.MODIFIED,
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
