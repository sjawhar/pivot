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
    _show_variant_only: bool

    def __init__(self, info: StageInfo, *, show_variant_only: bool = False) -> None:
        super().__init__(classes="stage-row")
        self._info = info
        self._is_selected = False
        self._show_variant_only = show_variant_only

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

        # Show variant only (@current) or full name depending on grouping
        if self._show_variant_only and self._info.variant:
            display_name = f"@{self._info.variant}"
        else:
            display_name = self._info.name
        name_escaped = rich.markup.escape(display_name)

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

        # Format: →  3  @current                 1:23 ▶
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
    _selected_name: str | None  # Track selection by name to avoid index sync issues
    _groups_cache: dict[str, list[StageInfo]] | None

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
        self._selected_name = stages[0].name if stages else None
        self._groups_cache = None

    def _compute_groups(self) -> dict[str, list[StageInfo]]:
        """Group stages by base_name, maintaining order. Results are cached."""
        if self._groups_cache is None:
            groups: dict[str, list[StageInfo]] = {}
            for stage in self._stages:
                if stage.base_name not in groups:
                    groups[stage.base_name] = []
                groups[stage.base_name].append(stage)
            self._groups_cache = groups
        return self._groups_cache

    @override
    def compose(self) -> textual.app.ComposeResult:  # pragma: no cover
        yield textual.widgets.Static(
            self._format_header(), id="stages-header", classes="section-header"
        )

        groups = self._compute_groups()
        seen_bases = set[str]()

        for stage_idx, stage in enumerate(self._stages):
            is_multi_member_group = len(groups[stage.base_name]) >= 2

            # If this is the first stage of a group with 2+ members, yield header
            if stage.base_name not in seen_bases:
                seen_bases.add(stage.base_name)
                if is_multi_member_group:
                    header = StageGroupHeader(stage.base_name, groups[stage.base_name])
                    header.set_collapsed(stage.base_name in self._collapsed_groups)
                    self._group_headers[stage.base_name] = header
                    yield header

            # Yield stage row (with collapsed class if in collapsed group)
            row = StageRow(stage, show_variant_only=is_multi_member_group)
            if stage.base_name in self._collapsed_groups:
                row.add_class("collapsed")
            # Add 'grouped' class for visual indentation in multi-member groups
            if is_multi_member_group:
                row.add_class("grouped")
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
        old_name = self._selected_name
        self._selected_idx = idx
        self._selected_name = selected_name

        # Deselect old row by name (safe even if indices changed)
        if old_name and old_name in self._rows and old_name != selected_name:
            self._rows[old_name].update_display(is_selected=False)
            # Update old group header if in a group
            old_stage = self._stage_by_name.get(old_name)
            if old_stage and old_stage.base_name in self._group_headers:
                self._group_headers[old_stage.base_name].update_display(is_selected=False)

        # Select new row
        if selected_name in self._rows:
            self._rows[selected_name].update_display(is_selected=True)
            self._rows[selected_name].scroll_visible()
            # Update new group header if in a group
            new_stage = self._stage_by_name.get(selected_name)
            if new_stage and new_stage.base_name in self._group_headers:
                self._group_headers[new_stage.base_name].update_display(is_selected=True)

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
        """Get base_name if selection is on any stage in a multi-member group."""
        if not self._stages or self._selected_idx >= len(self._stages):
            return None
        stage = self._stages[self._selected_idx]
        # Check if this stage is in a multi-member group
        groups = self._compute_groups()
        if stage.base_name in groups and len(groups[stage.base_name]) >= 2:
            return stage.base_name
        return None

    def rebuild(self, stages: list[StageInfo]) -> None:  # pragma: no cover
        """Rebuild panel with new stage list."""
        self._stages = stages
        self._stage_by_name = {s.name: s for s in stages}
        self._rows.clear()
        self._group_headers.clear()
        self._groups_cache = None  # Invalidate cache
        self.refresh(recompose=True)

    def is_collapsed(self, stage_name: str) -> bool:  # pragma: no cover
        """Check if a stage is hidden due to collapsed group.

        The first stage of a collapsed group is NOT collapsed (it's navigable
        and represents the group header position).
        """
        stage = self._stage_by_name.get(stage_name)
        if stage is None or stage.base_name not in self._collapsed_groups:
            return False
        # First stage of collapsed group is visible (navigable)
        groups = self._compute_groups()
        group_stages = groups.get(stage.base_name, [])
        return len(group_stages) > 0 and group_stages[0].name != stage_name

    def _set_all_groups_collapsed(self, collapsed: bool) -> None:  # pragma: no cover
        """Set collapse state for all multi-member groups."""
        groups = self._compute_groups()
        # Determine which group currently has selection for header update
        selected_base = None
        if self._selected_name:
            selected_stage = self._stage_by_name.get(self._selected_name)
            if selected_stage:
                selected_base = selected_stage.base_name

        for base_name, stages in groups.items():
            if len(stages) < 2:
                continue
            is_currently_collapsed = base_name in self._collapsed_groups
            if is_currently_collapsed == collapsed:
                continue  # Already in desired state

            if collapsed:
                self._collapsed_groups.add(base_name)
            else:
                self._collapsed_groups.discard(base_name)

            if base_name in self._group_headers:
                self._group_headers[base_name].set_collapsed(collapsed)
                is_selected = base_name == selected_base
                self._group_headers[base_name].update_display(is_selected=is_selected)

            for stage in stages:
                if stage.name in self._rows:
                    if collapsed:
                        self._rows[stage.name].add_class("collapsed")
                    else:
                        self._rows[stage.name].remove_class("collapsed")

    def collapse_all_groups(self) -> None:  # pragma: no cover
        """Collapse all multi-member groups."""
        self._set_all_groups_collapsed(True)

    def expand_all_groups(self) -> None:  # pragma: no cover
        """Expand all collapsed groups."""
        self._set_all_groups_collapsed(False)
