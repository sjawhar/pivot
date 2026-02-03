from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar, override

import rich.markup
import textual.app
import textual.binding
import textual.containers
import textual.css.query
import textual.message
import textual.widgets

from pivot.tui.widgets import status
from pivot.types import StageStatus

if TYPE_CHECKING:
    from textual.events import Resize

    from pivot.tui.types import StageInfo


class FilterEscapePressed(textual.message.Message):
    """Posted when Escape is pressed in the filter input."""

    pass


class FilterInput(textual.widgets.Input):
    """Input with Escape key handling."""

    BINDINGS: ClassVar[list[textual.binding.BindingType]] = [
        textual.binding.Binding("escape", "escape_pressed", "Cancel", show=False),
    ]

    def action_escape_pressed(self) -> None:
        self.post_message(FilterEscapePressed())


def _build_status_summary(counts: status.StatusCounts) -> str:
    """Build status summary string (running/failed indicators) from counts."""
    parts = list[str]()
    if counts["running"] > 0:
        parts.append(f"[blue bold]▶{counts['running']}[/]")
    if counts["failed"] > 0:
        parts.append(f"[red bold]!{counts['failed']}[/]")
    return " " + " ".join(parts) if parts else ""


class StageGroupHeader(textual.widgets.Static):
    """Header for a group of stages with the same base name.

    Note: This widget stores stage names (not StageInfo references) and queries
    the parent panel for fresh data on each update to avoid stale state.
    """

    _base_name: str
    _stage_names: list[str]  # Names only - query parent for fresh StageInfo
    _is_collapsed: bool
    _is_selected: bool

    def __init__(self, base_name: str, stage_names: list[str]) -> None:
        super().__init__(classes="stage-group-header")
        self._base_name = base_name
        self._stage_names = stage_names
        self._is_collapsed = False
        self._is_selected = False

    @property
    def base_name(self) -> str:
        return self._base_name

    @property
    def is_collapsed(self) -> bool:
        return self._is_collapsed

    def update_display(
        self, is_selected: bool | None = None, stages: list[StageInfo] | None = None
    ) -> None:  # pragma: no cover
        """Update the group header display.

        Args:
            is_selected: Whether this group contains the selected stage.
            stages: Fresh StageInfo list for status counts. If None, status is not shown.
        """
        if is_selected is not None:
            self._is_selected = is_selected

        # Compute status counts from fresh stage data
        if stages:
            counts = status.count_statuses(stages)
        else:
            counts: status.StatusCounts = {"running": 0, "completed": 0, "failed": 0}

        # Status summary (show counts for running, completed, failed)
        status_parts = list[str]()
        if counts["running"] > 0:
            status_parts.append(f"[blue bold]▶{counts['running']}[/]")
        if counts["completed"] > 0:
            status_parts.append(f"[green bold]●{counts['completed']}[/]")
        if counts["failed"] > 0:
            status_parts.append(f"[red bold]!{counts['failed']}[/]")
        status_str = " ".join(status_parts) if status_parts else ""

        # Collapse indicator and selection
        collapse_icon = ">" if self._is_collapsed else "v"
        prefix = f" → {collapse_icon} " if self._is_selected else f"{collapse_icon} "
        count = len(self._stage_names)
        count_str = f" ({count})"
        suffix = f"{count_str}  {status_str}" if status_str else count_str

        # Calculate available width for name
        available_width = self.size.width if self.size.width > 0 else 33
        suffix_visible_len = (
            len(count_str)
            + 2
            + sum(
                # Count visible chars in status (exclude markup)
                len(str(counts.get(k, 0))) + 1
                for k in ["running", "completed", "failed"]
                if counts.get(k, 0) > 0
            )
        )
        name_width = max(1, available_width - len(prefix) - suffix_visible_len - 1)

        # Truncate name if needed
        display_name = self._base_name
        if len(display_name) > name_width:
            display_name = display_name[: name_width - 1] + "…"
        name_escaped = rich.markup.escape(display_name)

        text = f"{prefix}[bold]{name_escaped}[/]{suffix}"
        self.update(text)

    def toggle_collapse(self) -> bool:  # pragma: no cover
        """Toggle collapsed state and return new state."""
        self._is_collapsed = not self._is_collapsed
        return self._is_collapsed

    def set_collapsed(self, collapsed: bool) -> None:
        """Set collapsed state."""
        self._is_collapsed = collapsed

    def on_mount(self) -> None:  # pragma: no cover
        # Initial display without status counts - panel will call update_display with fresh data
        self.update_display()

    def on_resize(self, _event: Resize) -> None:  # pragma: no cover
        """Re-render when resized to maintain proper truncation."""
        # Resize without status update - panel will provide fresh data when needed
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

    def set_info(self, info: StageInfo) -> None:
        """Update the StageInfo reference for this row."""
        self._info = info

    def update_display(self, is_selected: bool | None = None) -> None:  # pragma: no cover
        """Update the row display. If is_selected is None, use cached value."""
        if is_selected is not None:
            self._is_selected = is_selected

        # Update selected class for CSS highlighting
        self.set_class(self._is_selected, "selected")

        symbol, style = status.get_status_symbol(self._info.status, self._info.reason)
        symbol = f"{symbol} "  # Add space after symbol for visual separation
        index_str = f"{self._info.index:3}"

        # Show variant only (@current) or full name depending on grouping
        if self._show_variant_only and self._info.variant:
            display_name = f"@{self._info.variant}"
        else:
            display_name = self._info.name
        name_escaped = rich.markup.escape(display_name)

        # Format elapsed time (only for running/completed/failed)
        elapsed_str = ""
        elapsed = self._info.elapsed
        if (
            elapsed is not None
            and not math.isnan(elapsed)
            and not math.isinf(elapsed)
            and self._info.status
            in (
                StageStatus.IN_PROGRESS,
                StageStatus.COMPLETED,
                StageStatus.RAN,
                StageStatus.FAILED,
            )
        ):
            mins, secs = divmod(max(0, int(elapsed)), 60)
            elapsed_str = f"{mins}:{secs:02d} "

        # Selected row has slight indent, others are left-aligned
        prefix = f" {index_str} " if self._is_selected else f"{index_str} "

        # Calculate padding to right-align status symbol using actual widget width
        # Fall back to CSS min-width (35) minus padding (2) if not yet mounted
        available_width = self.size.width if self.size.width > 0 else 33
        suffix_visible_len = len(elapsed_str) + len(symbol)
        name_width = max(1, available_width - len(prefix) - suffix_visible_len - 1)

        # Truncate name with ellipsis if too long
        if len(display_name) > name_width:
            truncated_name = display_name[: name_width - 1] + "…"
            name_escaped = rich.markup.escape(truncated_name)

        # Format: →  3  name                          1:23 ▶
        text = f"{prefix}{name_escaped:<{name_width}} {elapsed_str}[{style}]{symbol}[/]"
        if self._is_selected:
            text = f"[bold]{text}[/]"
        self.update(text)

    def on_mount(self) -> None:  # pragma: no cover
        self.update_display()

    def on_resize(self, _event: Resize) -> None:  # pragma: no cover
        """Re-render when resized to maintain right-alignment."""
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
    _filter_text: str

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
        self._filter_text = ""

    @property
    def has_active_filter(self) -> bool:
        """Return True if a filter is currently active."""
        return bool(self._filter_text)

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

    def _get_group_stages(self, base_name: str) -> list[StageInfo]:
        """Get fresh StageInfo list for a group by base_name."""
        return self._compute_groups().get(base_name, [])

    @override
    def compose(self) -> textual.app.ComposeResult:  # pragma: no cover
        yield FilterInput(placeholder="Filter stages (/)", id="stage-filter")
        yield textual.widgets.Static(
            self._format_header(), id="stages-header", classes="section-header"
        )

        groups = self._compute_groups()

        # Render stages grouped by base_name to keep variants together visually.
        # Groups are yielded in order of first appearance (dict insertion order).
        for base_name, group_stages in groups.items():
            is_multi_member_group = len(group_stages) >= 2

            # For multi-member groups, yield a header first
            if is_multi_member_group:
                stage_names = [s.name for s in group_stages]
                header = StageGroupHeader(base_name, stage_names)
                header.set_collapsed(base_name in self._collapsed_groups)
                self._group_headers[base_name] = header
                yield header

            # Yield all stage rows in this group
            for stage in group_stages:
                row = StageRow(stage, show_variant_only=is_multi_member_group)
                if base_name in self._collapsed_groups:
                    row.add_class("collapsed")
                # Add 'grouped' class for visual indentation in multi-member groups
                if is_multi_member_group:
                    row.add_class("grouped")
                row.update_display(is_selected=(stage.name == self._selected_name))
                self._rows[stage.name] = row
                yield row

    def _format_header(self) -> str:  # pragma: no cover
        """Format header with stage count and status summary."""
        total = len(self._stages)
        counts = status.count_statuses(self._stages)
        summary = _build_status_summary(counts)
        return f"[bold]Stages ({total})[/]{summary}"

    def update_header(self) -> None:  # pragma: no cover
        """Update the header to reflect current status counts."""
        try:
            header = self.query_one("#stages-header", textual.widgets.Static)
            header.update(self._format_header())
        except textual.css.query.NoMatches:
            pass  # Header not yet mounted during initial compose

    def update_stage(
        self, name: str, selected_name: str | None = None, info: StageInfo | None = None
    ) -> None:  # pragma: no cover
        """Update a stage row display.

        Args:
            name: Stage name to update.
            selected_name: Currently selected stage name (for highlighting).
            info: If provided, sync the row's StageInfo reference to this object.
                  This ensures the row displays current state even if references diverged.
        """
        if name not in self._rows:
            self.update_header()
            return

        row = self._rows[name]

        # Sync StageInfo reference if provided (fixes reference divergence bugs)
        if info is not None:
            row.set_info(info)
            self._stage_by_name[name] = info

        is_selected = (name == selected_name) if selected_name else row.is_selected
        row.update_display(is_selected=is_selected)

        # Update group header if stage is in a group (O(1) lookup)
        stage = self._stage_by_name.get(name)
        if stage and stage.base_name in self._group_headers:
            fresh_stages = self._get_group_stages(stage.base_name)
            self._group_headers[stage.base_name].update_display(stages=fresh_stages)
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
                fresh_stages = self._get_group_stages(old_stage.base_name)
                self._group_headers[old_stage.base_name].update_display(
                    is_selected=False, stages=fresh_stages
                )

        # Select new row
        if selected_name in self._rows:
            self._rows[selected_name].update_display(is_selected=True)
            self._rows[selected_name].scroll_visible()
            # Update new group header if in a group
            new_stage = self._stage_by_name.get(selected_name)
            if new_stage and new_stage.base_name in self._group_headers:
                fresh_stages = self._get_group_stages(new_stage.base_name)
                self._group_headers[new_stage.base_name].update_display(
                    is_selected=True, stages=fresh_stages
                )

    def toggle_group(self, base_name: str) -> bool | None:  # pragma: no cover
        """Toggle collapse state for a group. Returns new collapsed state, or None if not found."""
        if base_name not in self._group_headers:
            return None

        header = self._group_headers[base_name]
        is_collapsed = header.toggle_collapse()
        fresh_stages = self._get_group_stages(base_name)
        header.update_display(stages=fresh_stages)

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
        if not self._selected_name:
            return None
        stage = self._stage_by_name.get(self._selected_name)
        if stage is None:
            return None
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
        self._filter_text = ""  # Reset filter to match new empty Input widget

        # Validate selection: keep if still exists, else reset to first stage
        if self._selected_name not in self._stage_by_name:
            self._selected_idx = 0
            self._selected_name = stages[0].name if stages else None

        # Prune collapsed groups that no longer exist
        new_groups = self._compute_groups()
        valid_multi_groups = {name for name, grp in new_groups.items() if len(grp) >= 2}
        self._collapsed_groups &= valid_multi_groups

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
                self._group_headers[base_name].update_display(
                    is_selected=is_selected, stages=stages
                )

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

    # =========================================================================
    # Stage filtering
    # =========================================================================

    def on_input_changed(self, event: textual.widgets.Input.Changed) -> None:  # pragma: no cover
        """Handle filter input changes."""
        if event.input.id == "stage-filter":
            self._filter_text = event.value.lower()
            self._apply_filter()

    def _apply_filter(self) -> None:  # pragma: no cover
        """Apply filter using CSS visibility - no recompose needed."""
        groups = self._compute_groups()

        # Filter individual stage rows
        for name, row in self._rows.items():
            if not self._filter_text or self._filter_text in name.lower():
                row.remove_class("filtered-out")
            else:
                row.add_class("filtered-out")

        # Hide group headers when all their stages are filtered out
        for base_name, header in self._group_headers.items():
            group_stages = groups.get(base_name, [])
            group_visible = any(
                not self._filter_text or self._filter_text in s.name.lower() for s in group_stages
            )
            if group_visible:
                header.remove_class("filtered-out")
            else:
                header.add_class("filtered-out")

        self._update_header_with_filter()

    def _update_header_with_filter(self) -> None:  # pragma: no cover
        """Update header to show filter status."""
        try:
            header = self.query_one("#stages-header", textual.widgets.Static)
        except textual.css.query.NoMatches:
            return

        visible = sum(1 for r in self._rows.values() if "filtered-out" not in r.classes)
        total = len(self._rows)

        if self._filter_text:
            counts = status.count_statuses(
                [
                    self._stage_by_name[n]
                    for n, r in self._rows.items()
                    if "filtered-out" not in r.classes
                ]
            )
            summary = _build_status_summary(counts)
            text = f"[bold]Stages ({visible}/{total})[/]{summary} [dim]/{rich.markup.escape(self._filter_text)}[/]"
        else:
            text = self._format_header()

        header.update(text)

    def get_visible_stage_names(self) -> list[str]:
        """Return stage names that are currently visible (not filtered out or collapsed)."""
        visible = list[str]()
        for stage in self._stages:
            name = stage.name
            if name not in self._rows:
                continue
            row = self._rows[name]
            # Skip if filtered out
            if "filtered-out" in row.classes:
                continue
            # Skip if collapsed (but not filtered)
            if self.is_collapsed(name):
                continue
            visible.append(name)
        return visible

    def clear_filter(self) -> None:  # pragma: no cover
        """Clear the filter text and show all stages."""
        try:
            filter_input = self.query_one("#stage-filter", textual.widgets.Input)
            filter_input.value = ""
        except textual.css.query.NoMatches:
            pass
        self._filter_text = ""
        self._apply_filter()

    def focus_filter(self) -> None:  # pragma: no cover
        """Focus the filter input."""
        try:
            filter_input = self.query_one("#stage-filter", textual.widgets.Input)
            filter_input.focus()
        except textual.css.query.NoMatches:
            pass

    def _blur_filter(self) -> None:  # pragma: no cover
        """Blur filter input and return focus to stage list panel."""
        # Focus the panel itself - it handles j/k navigation
        # Individual StageRow widgets are not focusable (they extend Static)
        self.focus()

    def on_input_submitted(
        self, event: textual.widgets.Input.Submitted
    ) -> None:  # pragma: no cover
        """Handle Enter: confirm filter and return focus to stage list."""
        if event.input.id == "stage-filter":
            self._blur_filter()

    def on_filter_escape_pressed(self, event: FilterEscapePressed) -> None:  # pragma: no cover
        """Handle Escape in filter input: clear filter or return focus to stages."""
        event.stop()
        if self._filter_text:
            self.clear_filter()
        else:
            self._blur_filter()
