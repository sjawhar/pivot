from __future__ import annotations

from typing import TYPE_CHECKING, override

import rich.markup
import textual.app
import textual.containers
import textual.css.query
import textual.widgets

from pivot.tui.widgets import status
from pivot.types import StageStatus

if TYPE_CHECKING:
    from pivot.tui.types import StageInfo


class StageGroupHeader(textual.widgets.Static):
    """Header for a group of stages with the same base name."""

    _base_name: str
    _stages: list[StageInfo]
    _is_collapsed: bool
    _is_selected: bool

    def __init__(self, base_name: str, stages: list[StageInfo]) -> None:
        super().__init__(classes="stage-group-header")
        self._base_name = base_name
        self._stages = stages
        self._is_collapsed = False
        self._is_selected = False

    @property
    def base_name(self) -> str:
        return self._base_name

    @property
    def is_collapsed(self) -> bool:
        return self._is_collapsed

    def update_display(self, is_selected: bool | None = None) -> None:  # pragma: no cover
        """Update the group header display."""
        if is_selected is not None:
            self._is_selected = is_selected

        counts = status.count_statuses(self._stages)

        # Status summary (show counts for running, completed, failed)
        status_parts = list[str]()
        if counts["running"] > 0:
            status_parts.append(f"[blue bold]▶{counts['running']}[/]")
        if counts["completed"] > 0:
            status_parts.append(f"[green bold]●{counts['completed']}[/]")
        if counts["failed"] > 0:
            status_parts.append(f"[red bold]!{counts['failed']}[/]")
        status_str = " ".join(status_parts) if status_parts else ""

        # Collapse indicator
        collapse_icon = ">" if self._is_collapsed else "v"
        arrow = "→ " if self._is_selected else "  "
        count = len(self._stages)
        name_escaped = rich.markup.escape(self._base_name)

        text = f"{arrow}[bold]{collapse_icon}[/] {name_escaped} ({count})  {status_str}"
        self.update(text)

    def toggle_collapse(self) -> bool:  # pragma: no cover
        """Toggle collapsed state and return new state."""
        self._is_collapsed = not self._is_collapsed
        return self._is_collapsed

    def set_collapsed(self, collapsed: bool) -> None:
        """Set collapsed state."""
        self._is_collapsed = collapsed

    def on_mount(self) -> None:  # pragma: no cover
        self.update_display()


class StageRow(textual.widgets.Static):
    """Single stage row showing index, name, status, and reason."""

    _info: StageInfo
    _is_selected: bool

    def __init__(self, info: StageInfo) -> None:
        super().__init__(classes="stage-row")
        self._info = info
        self._is_selected = False

    @property
    def is_selected(self) -> bool:
        """Return whether this row is selected."""
        return self._is_selected

    def update_display(self, is_selected: bool | None = None) -> None:  # pragma: no cover
        """Update the row display. If is_selected is None, use cached value."""
        if is_selected is not None:
            self._is_selected = is_selected

        symbol, style = status.get_status_symbol(self._info.status)
        index_str = f"{self._info.index:3}"
        name_escaped = rich.markup.escape(self._info.name)

        # Format elapsed time (only for running/completed/failed)
        elapsed_str = ""
        if self._info.elapsed is not None and self._info.status in (
            StageStatus.IN_PROGRESS,
            StageStatus.COMPLETED,
            StageStatus.RAN,
            StageStatus.FAILED,
        ):
            mins, secs = divmod(int(self._info.elapsed), 60)
            elapsed_str = f"{mins}:{secs:02d} "

        # Selection arrow prefix
        arrow = "→ " if self._is_selected else "  "

        # Format: →  3  train@small              1:23 ▶
        text = f"{arrow}{index_str}  {name_escaped:<24} {elapsed_str}[{style}]{symbol}[/]"
        self.update(text)

    def on_mount(self) -> None:  # pragma: no cover
        self.update_display()


class StageListPanel(textual.containers.VerticalScroll):
    """Panel showing all stages with their status, with scrolling and grouping support."""

    _stages: list[StageInfo]
    _stage_by_name: dict[str, StageInfo]
    _rows: dict[str, StageRow]
    _group_headers: dict[str, StageGroupHeader]  # base_name -> header
    _collapsed_groups: set[str]  # base_names of collapsed groups
    _selected_idx: int

    def __init__(
        self,
        stages: list[StageInfo],
        *,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(id=id, classes=classes)
        self._stages = stages
        self._stage_by_name = {s.name: s for s in stages}
        self._rows = {}
        self._group_headers = {}
        self._collapsed_groups = set()
        self._selected_idx = 0

    def _compute_groups(self) -> dict[str, list[StageInfo]]:
        """Group stages by base_name, maintaining order."""
        groups: dict[str, list[StageInfo]] = {}
        for stage in self._stages:
            if stage.base_name not in groups:
                groups[stage.base_name] = []
            groups[stage.base_name].append(stage)
        return groups

    @override
    def compose(self) -> textual.app.ComposeResult:  # pragma: no cover
        yield textual.widgets.Static(
            self._format_header(), id="stages-header", classes="section-header"
        )

        groups = self._compute_groups()
        seen_bases = set[str]()

        for stage_idx, stage in enumerate(self._stages):
            # If this is the first stage of a group with 2+ members, yield header
            if stage.base_name not in seen_bases:
                seen_bases.add(stage.base_name)
                group_stages = groups[stage.base_name]
                if len(group_stages) >= 2:
                    header = StageGroupHeader(stage.base_name, group_stages)
                    header.set_collapsed(stage.base_name in self._collapsed_groups)
                    self._group_headers[stage.base_name] = header
                    yield header

            # Yield stage row (with collapsed class if in collapsed group)
            row = StageRow(stage)
            if stage.base_name in self._collapsed_groups:
                row.add_class("collapsed")
            row.update_display(is_selected=(stage_idx == self._selected_idx))
            self._rows[stage.name] = row
            yield row

    def _format_header(self) -> str:  # pragma: no cover
        """Format header with stage count and status summary."""
        total = len(self._stages)
        counts = status.count_statuses(self._stages)

        summary_parts = list[str]()
        if counts["running"] > 0:
            summary_parts.append(f"[blue bold]▶{counts['running']}[/]")
        if counts["failed"] > 0:
            summary_parts.append(f"[red bold]!{counts['failed']}[/]")

        summary = " " + " ".join(summary_parts) if summary_parts else ""
        return f"[bold]Stages ({total})[/]{summary}"

    def update_header(self) -> None:  # pragma: no cover
        """Update the header to reflect current status counts."""
        try:
            header = self.query_one("#stages-header", textual.widgets.Static)
            header.update(self._format_header())
        except textual.css.query.NoMatches:
            pass  # Header not yet mounted during initial compose

    def update_stage(self, name: str, selected_name: str | None = None) -> None:  # pragma: no cover
        """Update a stage row display."""
        if name not in self._rows:
            self.update_header()
            return

        row = self._rows[name]
        is_selected = (name == selected_name) if selected_name else row.is_selected
        row.update_display(is_selected=is_selected)

        # Update group header if stage is in a group (O(1) lookup)
        stage = self._stage_by_name.get(name)
        if stage and stage.base_name in self._group_headers:
            self._group_headers[stage.base_name].update_display()
        self.update_header()

    def set_selection(self, idx: int, selected_name: str) -> None:  # pragma: no cover
        """Update selection state and scroll to keep it visible."""
        old_idx = self._selected_idx
        self._selected_idx = idx

        # Update old and new selected rows
        if 0 <= old_idx < len(self._stages):
            old_name = self._stages[old_idx].name
            if old_name in self._rows:
                self._rows[old_name].update_display(is_selected=False)
        if selected_name in self._rows:
            self._rows[selected_name].update_display(is_selected=True)
            # Scroll to keep selected row visible
            self._rows[selected_name].scroll_visible()

    def toggle_group(self, base_name: str) -> bool | None:  # pragma: no cover
        """Toggle collapse state for a group. Returns new collapsed state, or None if not found."""
        if base_name not in self._group_headers:
            return None

        header = self._group_headers[base_name]
        is_collapsed = header.toggle_collapse()
        header.update_display()

        # Update collapsed groups set
        if is_collapsed:
            self._collapsed_groups.add(base_name)
        else:
            self._collapsed_groups.discard(base_name)

        # Update CSS class on all rows in this group
        for stage in self._stages:
            if stage.base_name == base_name and stage.name in self._rows:
                row = self._rows[stage.name]
                if is_collapsed:
                    row.add_class("collapsed")
                else:
                    row.remove_class("collapsed")

        return is_collapsed

    def get_group_at_selection(self) -> str | None:  # pragma: no cover
        """Get base_name of group header if selection is on first stage of a group."""
        if not self._stages or self._selected_idx >= len(self._stages):
            return None
        stage = self._stages[self._selected_idx]
        # Check if this is the first stage of a multi-variant group
        groups = self._compute_groups()
        if stage.base_name in groups and len(groups[stage.base_name]) >= 2:
            # Check if this is the first stage of the group
            first_in_group = groups[stage.base_name][0]
            if stage.name == first_in_group.name:
                return stage.base_name
        return None

    def rebuild(self, stages: list[StageInfo]) -> None:  # pragma: no cover
        """Rebuild panel with new stage list."""
        self._stages = stages
        self._stage_by_name = {s.name: s for s in stages}
        self._rows.clear()
        self._group_headers.clear()
        self.refresh(recompose=True)
