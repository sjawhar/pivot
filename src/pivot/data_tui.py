from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

import textual.app
import textual.binding
import textual.containers
import textual.widgets

from pivot import data, project
from pivot.types import ChangeType, DataDiffResult, RowChange, SchemaChange

# Rich markup symbols for change types
STATUS_SYMBOLS_RICH: dict[ChangeType, str] = {
    ChangeType.ADDED: "[green]+[/]",
    ChangeType.REMOVED: "[red]-[/]",
    ChangeType.MODIFIED: "[yellow]~[/]",
}

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Sequence

    from pivot.data import DataDiffEntry


class DiffSummaryPanel(textual.widgets.Static):
    """Shows diff summary: row counts, column changes, reorder-only status."""

    _result: DataDiffResult

    def __init__(self, result: DataDiffResult) -> None:
        super().__init__()
        self._result = result

    @override
    def compose(self) -> textual.app.ComposeResult:  # pragma: no cover
        summary_lines = list[str]()

        # File path and status
        if self._result["reorder_only"]:
            summary_lines.append(f"[bold blue]{self._result['path']}[/] - REORDER ONLY")
            summary_lines.append("[dim]Same content, different row order[/]")
        else:
            summary_lines.append(f"[bold]{self._result['path']}[/]")

        # Row counts
        old_rows = self._result["old_rows"]
        new_rows = self._result["new_rows"]
        if old_rows is not None and new_rows is not None:
            diff = new_rows - old_rows
            if diff > 0:
                diff_str = f"[green]+{diff}[/]"
            elif diff < 0:
                diff_str = f"[red]{diff}[/]"
            else:
                diff_str = "0"
            summary_lines.append(f"Rows: {old_rows:,} -> {new_rows:,} ({diff_str})")
        elif old_rows is not None:
            summary_lines.append(f"Rows: {old_rows:,} [red](removed)[/]")
        elif new_rows is not None:
            summary_lines.append(f"Rows: {new_rows:,} [green](added)[/]")

        # Column counts
        old_cols = self._result["old_cols"]
        new_cols = self._result["new_cols"]
        if old_cols is not None and new_cols is not None:
            added = len([c for c in new_cols if c not in old_cols])
            removed = len([c for c in old_cols if c not in new_cols])
            parts = list[str]()
            if added:
                parts.append(f"[green]+{added} added[/]")
            if removed:
                parts.append(f"[red]-{removed} removed[/]")
            col_info = f"{len(old_cols)} -> {len(new_cols)}"
            if parts:
                col_info += f" ({', '.join(parts)})"
            summary_lines.append(f"Columns: {col_info}")

        # Row change summary
        row_changes = self._result["row_changes"]
        if row_changes:
            added = sum(1 for r in row_changes if r["change_type"] == ChangeType.ADDED)
            removed = sum(1 for r in row_changes if r["change_type"] == ChangeType.REMOVED)
            modified = sum(1 for r in row_changes if r["change_type"] == ChangeType.MODIFIED)
            parts = list[str]()
            if added:
                parts.append(f"[green]+{added}[/]")
            if removed:
                parts.append(f"[red]-{removed}[/]")
            if modified:
                parts.append(f"[yellow]~{modified}[/]")
            if self._result["truncated"]:
                parts.append("[dim](truncated)[/]")
            summary_lines.append(f"Changes: {', '.join(parts)}")

        yield textual.widgets.Static("\n".join(summary_lines))


class SchemaChangesTable(textual.widgets.DataTable[str]):
    """Table showing schema (column) changes."""

    _changes: list[SchemaChange]

    def __init__(self, changes: list[SchemaChange]) -> None:
        super().__init__()
        self._changes = changes

    @override
    def on_mount(self) -> None:  # pragma: no cover
        self.add_columns("Status", "Column", "Old Type", "New Type")
        for change in self._changes:
            status = STATUS_SYMBOLS_RICH.get(change["change_type"], "?")
            self.add_row(
                status,
                change["column"],
                change["old_dtype"] or "-",
                change["new_dtype"] or "-",
            )


class RowChangesTable(textual.widgets.DataTable[str]):
    """Table showing row-level changes."""

    _changes: list[RowChange]
    _columns: list[str]

    def __init__(self, changes: list[RowChange], columns: list[str]) -> None:
        super().__init__()
        self._changes = changes
        self._columns = columns

    @override
    def on_mount(self) -> None:  # pragma: no cover
        # Add columns: Status, Key, then all data columns
        self.add_columns("", "Key", *self._columns)

        for change in self._changes:
            status = STATUS_SYMBOLS_RICH.get(change["change_type"], "?")
            key = str(change["key"])
            row_data = [status, key]

            # Build column values with change highlighting
            for col in self._columns:
                old_val = change["old_values"].get(col) if change["old_values"] else None
                new_val = change["new_values"].get(col) if change["new_values"] else None

                match change["change_type"]:
                    case ChangeType.ADDED:
                        row_data.append(f"[green]{new_val}[/]")
                    case ChangeType.REMOVED:
                        row_data.append(f"[red]{old_val}[/]")
                    case _ if old_val != new_val:
                        row_data.append(f"[yellow]{old_val} -> {new_val}[/]")
                    case _:
                        row_data.append(str(new_val) if new_val is not None else "")

            self.add_row(*row_data)


class FileDiffScreen(textual.widgets.Static):
    """Screen showing diff for a single file."""

    _result: DataDiffResult

    def __init__(self, result: DataDiffResult) -> None:
        super().__init__()
        self._result = result

    @override
    def compose(self) -> textual.app.ComposeResult:  # pragma: no cover
        yield DiffSummaryPanel(self._result)

        # Schema changes section
        if self._result["schema_changes"]:
            yield textual.widgets.Static("[bold]Schema Changes[/]")
            yield SchemaChangesTable(self._result["schema_changes"])

        # Row changes section
        if self._result["row_changes"] and not self._result["summary_only"]:
            yield textual.widgets.Static("[bold]Row Changes[/]")
            columns = self._result["new_cols"] or self._result["old_cols"] or []
            yield RowChangesTable(self._result["row_changes"], columns)
        elif self._result["summary_only"]:
            yield textual.widgets.Static(
                "[dim]Row-level diff not available (file too large or --summary used)[/]"
            )


class DataDiffApp(textual.app.App[None]):
    """Interactive data diff viewer."""

    CSS: ClassVar[str] = """
    DiffSummaryPanel {
        padding: 1;
        border: solid green;
        margin-bottom: 1;
    }

    SchemaChangesTable {
        height: auto;
        max-height: 10;
        margin-bottom: 1;
    }

    RowChangesTable {
        height: 1fr;
    }

    .added { color: green; }
    .removed { color: red; }
    .modified { color: yellow; }
    .reorder-only { color: blue; }
    """

    BINDINGS: ClassVar[list[textual.binding.BindingType]] = [
        textual.binding.Binding("q", "quit", "Quit"),
        textual.binding.Binding("j", "scroll_down", "Down"),
        textual.binding.Binding("k", "scroll_up", "Up"),
        textual.binding.Binding("n", "next_file", "Next file"),
        textual.binding.Binding("p", "prev_file", "Prev file"),
    ]

    _diff_entries: list[DataDiffEntry]
    _key_cols: list[str] | None
    _max_rows: int
    _current_idx: int
    _results: list[DataDiffResult]
    _temp_files: list[pathlib.Path]

    def __init__(
        self,
        diff_entries: Sequence[DataDiffEntry],
        key_cols: list[str] | None,
        max_rows: int,
    ) -> None:
        super().__init__()
        self._diff_entries = list(diff_entries)
        self._key_cols = key_cols
        self._max_rows = max_rows
        self._current_idx = 0
        self._results = []
        self._temp_files = []

    @override
    def compose(self) -> textual.app.ComposeResult:  # pragma: no cover
        yield textual.widgets.Header()
        yield textual.containers.VerticalScroll(id="main-content")
        yield textual.widgets.Footer()

    async def on_mount(self) -> None:  # pragma: no cover
        # Load diff results
        self._results = self._load_diff_results()
        self._show_current_file()

    def _load_diff_results(self) -> list[DataDiffResult]:  # pragma: no cover
        """Load diff results for all entries."""
        results = list[DataDiffResult]()
        proj_root = project.get_project_root()

        for entry in self._diff_entries:
            rel_path = entry["path"]
            abs_path = proj_root / rel_path
            old_hash = entry["old_hash"]

            # Restore old file from cache if needed
            old_path: pathlib.Path | None = None
            if old_hash is not None:
                old_path = data.restore_data_from_cache(rel_path, old_hash)
                if old_path is not None:
                    self._temp_files.append(old_path)
            new_path = abs_path if abs_path.exists() else None

            result = data.diff_data_files(
                old_path=old_path,
                new_path=new_path,
                path_display=rel_path,
                key_columns=self._key_cols,
                max_rows=self._max_rows,
            )
            results.append(result)

        return results

    def cleanup_temp_files(self) -> None:
        """Clean up temp files. Call after app.run() returns."""
        for temp_file in self._temp_files:
            temp_file.unlink(missing_ok=True)

    def _show_current_file(self) -> None:  # pragma: no cover
        """Display the current file's diff."""
        if not self._results:
            return

        content = self.query_one("#main-content", textual.containers.VerticalScroll)
        content.remove_children()

        result = self._results[self._current_idx]
        content.mount(FileDiffScreen(result))

        # Update title
        total = len(self._results)
        current = self._current_idx + 1
        self.title = f"pivot data diff ({current}/{total}): {result['path']}"  # pyright: ignore[reportUnannotatedClassAttribute]

    def action_next_file(self) -> None:  # pragma: no cover
        """Move to next file."""
        if self._current_idx < len(self._results) - 1:
            self._current_idx += 1
            self._show_current_file()

    def action_prev_file(self) -> None:  # pragma: no cover
        """Move to previous file."""
        if self._current_idx > 0:
            self._current_idx -= 1
            self._show_current_file()

    def action_scroll_down(self) -> None:  # pragma: no cover
        """Scroll down."""
        content = self.query_one("#main-content", textual.containers.VerticalScroll)
        content.scroll_down()

    def action_scroll_up(self) -> None:  # pragma: no cover
        """Scroll up."""
        content = self.query_one("#main-content", textual.containers.VerticalScroll)
        content.scroll_up()


def run_diff_app(
    diff_entries: Sequence[DataDiffEntry],
    key_cols: list[str] | None,
    max_rows: int,
) -> None:  # pragma: no cover
    """Entry point for TUI."""
    app = DataDiffApp(diff_entries, key_cols, max_rows)
    try:
        app.run()
    finally:
        app.cleanup_temp_files()
