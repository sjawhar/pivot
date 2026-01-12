"""Input and Output diff panels for the TUI.

Displays stage change information in the Input and Output tabs:
- Input tab: code changes, dependency changes, parameter changes
- Output tab: output file changes grouped by type (Out, Metric, Plot)

Both panels use a split-view layout with an item list (left) and details pane (right).
Users can navigate with j/k and expand details to full-width with Enter.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Literal, assert_never, override

import rich.markup
import textual.app
import textual.containers
import textual.css.query
import textual.widgets

from pivot import explain, outputs, parameters, project
from pivot.registry import REGISTRY
from pivot.show import metrics as metrics_mod
from pivot.show.metrics import MetricDiff
from pivot.storage import cache, lock
from pivot.types import (
    ChangeType,
    CodeChange,
    DepChange,
    MetricValue,
    OutputChange,
    ParamChange,
    StageExplanation,
)

if TYPE_CHECKING:
    from pivot.registry import RegistryStageInfo
    from pivot.types import LockData, OutputHash

# Type alias for output types matching OutputChange["output_type"]
OutputType = Literal["out", "metric", "plot"]

# Change indicators with brackets and colors (per plan spec)
_INDICATOR_MODIFIED = "[yellow]\\[~][/]"
_INDICATOR_ADDED = "[green]\\[+][/]"
_INDICATOR_REMOVED = "[red]\\[-][/]"
_INDICATOR_UNCHANGED = "[dim]\\[ ][/]"


def _get_indicator(change_type: ChangeType | None) -> str:
    """Get the appropriate indicator for a change type."""
    if change_type is None:
        return _INDICATOR_UNCHANGED
    match change_type:
        case ChangeType.MODIFIED:
            return _INDICATOR_MODIFIED
        case ChangeType.ADDED:
            return _INDICATOR_ADDED
        case ChangeType.REMOVED:
            return _INDICATOR_REMOVED
        case _ as unreachable:  # pyright: ignore[reportUnnecessaryComparison]
            assert_never(unreachable)


def _truncate_hash(hash_str: str | None, length: int = 8) -> str:
    """Truncate hash to specified length, or return placeholder."""
    if hash_str is None:
        return "(none)"
    return hash_str[:length]


def _format_hash_change(
    old_hash: str | None,
    new_hash: str | None,
    change_type: ChangeType | None,
) -> str:
    """Format the hash change display."""
    if change_type is None:
        # Unchanged
        return "(unchanged)"
    match change_type:
        case ChangeType.ADDED:
            return f"(none)   -> {_truncate_hash(new_hash)}"
        case ChangeType.REMOVED:
            return f"{_truncate_hash(old_hash)} -> (deleted)"
        case ChangeType.MODIFIED:
            return f"{_truncate_hash(old_hash)} -> {_truncate_hash(new_hash)}"
        case _ as unreachable:  # pyright: ignore[reportUnnecessaryComparison]
            assert_never(unreachable)


def _get_registry_info(stage_name: str) -> RegistryStageInfo | None:
    """Safely get registry info for a stage."""
    try:
        return REGISTRY.get(stage_name)
    except KeyError:
        return None


def _get_cache_dir() -> pathlib.Path:
    """Get the cache directory path."""
    return project.get_project_root() / ".pivot" / "cache"


def _get_relative_path(abs_path: str) -> str:
    """Convert absolute path to relative path from project root."""
    try:
        proj_root = project.get_project_root()
        path = pathlib.Path(abs_path)
        if path.is_absolute():
            return str(path.relative_to(proj_root))
    except ValueError:
        pass
    return abs_path


def _compute_output_changes(
    lock_data: LockData | None,
    registry_info: RegistryStageInfo,
) -> list[OutputChange]:
    """Compute output changes by comparing lock file with current state."""
    changes = list[OutputChange]()

    # Build maps for easier lookup
    outs = registry_info["outs"]
    outs_paths = registry_info["outs_paths"]

    # Map path -> output type (properly typed as Literal)
    path_to_type: dict[str, OutputType] = {}
    for out, path in zip(outs, outs_paths, strict=True):
        if isinstance(out, outputs.Metric):
            path_to_type[path] = "metric"
        elif isinstance(out, outputs.Plot):
            path_to_type[path] = "plot"
        else:
            path_to_type[path] = "out"

    # Get old hashes from lock
    old_hashes: dict[str, OutputHash] = {}
    if lock_data and "output_hashes" in lock_data:
        old_hashes = lock_data["output_hashes"]

    # Compare each output
    for path in outs_paths:
        old_hash_info = old_hashes.get(path)
        old_hash: str | None = None
        if old_hash_info is not None:
            old_hash = old_hash_info.get("hash")

        # Compute current hash
        new_hash: str | None = None
        path_obj = pathlib.Path(path)
        try:
            if path_obj.exists():
                if path_obj.is_dir():
                    new_hash, _ = cache.hash_directory(path_obj)
                else:
                    new_hash = cache.hash_file(path_obj)
        except OSError:
            # File unreadable
            new_hash = None

        # Determine change type
        change_type: ChangeType | None = None
        if old_hash is None and new_hash is not None:
            change_type = ChangeType.ADDED
        elif old_hash is not None and new_hash is None:
            change_type = ChangeType.REMOVED
        elif old_hash != new_hash and old_hash is not None and new_hash is not None:
            change_type = ChangeType.MODIFIED
        # else: both None or equal -> unchanged (None)

        # Since path_to_type is dict[str, OutputType] and we provide "out" as default,
        # this is already typed as OutputType (Literal["out", "metric", "plot"])
        output_type: OutputType = path_to_type.get(path, "out")

        changes.append(
            OutputChange(
                path=path,
                old_hash=old_hash,
                new_hash=new_hash,
                change_type=change_type,
                output_type=output_type,
            )
        )

    return changes


class _SelectableExpandablePanel(textual.containers.Horizontal):
    """Base class for panels with keyboard-navigable items and expandable details.

    Provides:
    - Split-view layout (item list left, details right)
    - Selection state with j/k navigation
    - Expansion to full-width with Enter, collapse with Esc
    - State reset when stage changes

    Subclasses must implement:
    - _build_items(): Return list of item IDs in display order
    - _render_item_row(): Render a single item row for the list
    - _render_detail_content(): Render detail content for selected item
    """

    _stage_name: str | None
    _selected_idx: int
    _item_ids: list[str]
    _detail_expanded: bool

    def __init__(self, *, id: str | None = None, classes: str | None = None) -> None:
        super().__init__(id=id, classes=classes)
        self._stage_name = None
        self._selected_idx = 0
        self._item_ids = list[str]()
        self._detail_expanded = False

    @override
    def compose(self) -> textual.app.ComposeResult:  # pragma: no cover
        yield textual.widgets.Static(id="item-list")
        yield textual.widgets.Static(id="detail-pane")

    @property
    def is_detail_expanded(self) -> bool:
        """Whether the detail pane is in full-width mode."""
        return self._detail_expanded

    def select_next(self) -> None:  # pragma: no cover
        """Move selection to next item."""
        if self._item_ids and self._selected_idx < len(self._item_ids) - 1:
            self._selected_idx += 1
            self._update_display()

    def select_prev(self) -> None:  # pragma: no cover
        """Move selection to previous item."""
        if self._item_ids and self._selected_idx > 0:
            self._selected_idx -= 1
            self._update_display()

    def expand_details(self) -> None:  # pragma: no cover
        """Expand details pane to full width."""
        if self._item_ids:
            self._detail_expanded = True
            self.add_class("expanded")
            self._update_display()

    def collapse_details(self) -> None:  # pragma: no cover
        """Collapse details pane back to split view."""
        self._detail_expanded = False
        self.remove_class("expanded")
        self._update_display()

    def _reset_selection_state(self) -> None:
        """Reset selection and expansion state. Called by subclasses in set_stage."""
        self._selected_idx = 0
        self._item_ids.clear()
        self._detail_expanded = False
        self.remove_class("expanded")

    def set_stage(self, stage_name: str | None) -> None:  # pragma: no cover
        """Update the displayed stage."""
        self._reset_selection_state()
        self._stage_name = stage_name
        self._update_display()

    def _update_display(self) -> None:  # pragma: no cover
        """Update both item list and detail pane."""
        # Build items first
        self._item_ids = self._build_items()

        # Clamp selection to valid range
        if self._item_ids:
            self._selected_idx = min(self._selected_idx, len(self._item_ids) - 1)
        else:
            self._selected_idx = 0

        # Skip widget updates if not mounted yet
        try:
            item_list = self.query_one("#item-list", textual.widgets.Static)
            detail_pane = self.query_one("#detail-pane", textual.widgets.Static)
        except textual.css.query.NoMatches:
            return

        # Update item list
        if not self._item_ids:
            item_list.update(self._render_empty_state())
        else:
            lines = list[str]()
            for idx, item_id in enumerate(self._item_ids):
                is_selected = idx == self._selected_idx
                lines.append(self._render_item_row(item_id, is_selected))
            item_list.update("\n".join(lines))

        # Update detail pane
        if not self._item_ids:
            detail_pane.update("")
        else:
            selected_item = self._item_ids[self._selected_idx]
            detail_pane.update(self._render_detail_content(selected_item))

    # Abstract methods - subclasses must implement
    def _build_items(self) -> list[str]:
        """Build and return list of item IDs in display order."""
        raise NotImplementedError

    def _render_item_row(self, _item_id: str, _is_selected: bool) -> str:
        """Render a single item row for the list."""
        raise NotImplementedError

    def _render_detail_content(self, _item_id: str) -> str:
        """Render detail content for the selected item."""
        raise NotImplementedError

    def _render_empty_state(self) -> str:
        """Render content when no items are available."""
        return "[dim]No stage selected[/]"

    def _format_status(self, change_type: ChangeType | None) -> str:
        """Format change type as status string."""
        if change_type is None:
            return "[dim]Unchanged[/]"
        match change_type:
            case ChangeType.MODIFIED:
                return "[yellow]Modified[/]"
            case ChangeType.ADDED:
                return "[green]Added[/]"
            case ChangeType.REMOVED:
                return "[red]Removed[/]"
            case _ as unreachable:  # pyright: ignore[reportUnnecessaryComparison]
                assert_never(unreachable)

    def _append_hash_detail(
        self,
        lines: list[str],
        old_hash: str | None,
        new_hash: str | None,
        change_type: ChangeType | None,
    ) -> None:
        """Append hash details and expand hint to lines list."""
        if self._detail_expanded:
            lines.extend(
                [
                    f"Old hash: {rich.markup.escape(old_hash or '(none)')}",
                    f"New hash: {rich.markup.escape(new_hash or '(none)')}",
                ]
            )
        else:
            lines.append(_format_hash_change(old_hash, new_hash, change_type))
            lines.extend(["", "[dim]\\[Enter] Expand[/]"])


class InputDiffPanel(_SelectableExpandablePanel):
    """Panel showing input changes for a stage (code, deps, params)."""

    # Cache for stage data to avoid recomputation on selection changes
    _stage_name: str | None
    _explanation: StageExplanation | None
    _registry_info: RegistryStageInfo | None
    # Dict-based storage for O(1) lookup by key/path
    _code_by_key: dict[str, CodeChange]
    _dep_by_path: dict[str, DepChange]
    _param_by_key: dict[str, ParamChange]

    def __init__(self, *, id: str | None = None, classes: str | None = None) -> None:
        super().__init__(
            id=id, classes="diff-panel" if classes is None else f"diff-panel {classes}"
        )
        self._explanation = None
        self._registry_info = None
        self._code_by_key = dict[str, CodeChange]()
        self._dep_by_path = dict[str, DepChange]()
        self._param_by_key = dict[str, ParamChange]()

    @override
    def set_stage(self, stage_name: str | None) -> None:  # pragma: no cover
        """Update the displayed stage and load data."""
        self._reset_selection_state()
        self._explanation = None
        self._registry_info = None
        self._code_by_key.clear()
        self._dep_by_path.clear()
        self._param_by_key.clear()
        self._stage_name = stage_name

        if stage_name is not None:
            self._load_stage_data(stage_name)

        self._update_display()

    def _load_stage_data(self, stage_name: str) -> None:  # pragma: no cover
        """Load and cache stage data."""
        self._registry_info = _get_registry_info(stage_name)
        if self._registry_info is None:
            return

        cache_dir = _get_cache_dir()
        try:
            self._explanation = explain.get_stage_explanation(
                stage_name=stage_name,
                fingerprint=self._registry_info["fingerprint"],
                deps=self._registry_info["deps"],
                params_instance=self._registry_info["params"],
                overrides=parameters.load_params_yaml(),
                cache_dir=cache_dir,
            )
        except Exception:
            self._explanation = None
            return

        # Cache items as dicts for O(1) lookup
        self._code_by_key = {c["key"]: c for c in self._explanation["code_changes"]}
        self._dep_by_path = {c["path"]: c for c in self._explanation["dep_changes"]}
        self._param_by_key = {c["key"]: c for c in self._explanation["param_changes"]}

    @override
    def _build_items(self) -> list[str]:  # pragma: no cover
        """Build item IDs from code, deps, and params."""
        items = list[str]()

        # Add code items
        for key in self._code_by_key:
            items.append(f"code:{key}")

        # Add dep items
        for path in self._dep_by_path:
            items.append(f"dep:{path}")

        # Add unchanged deps if no changes
        if not self._dep_by_path and self._registry_info:
            for dep_path in self._registry_info["deps"]:
                items.append(f"dep:{dep_path}")

        # Add param items
        for key in self._param_by_key:
            items.append(f"param:{key}")

        return items

    @override
    def _render_item_row(self, item_id: str, is_selected: bool) -> str:  # pragma: no cover
        """Render a single item row."""
        prefix = "[reverse]" if is_selected else ""
        suffix = "[/]" if is_selected else ""

        item_type, item_key = item_id.split(":", 1)

        match item_type:
            case "code":
                change = self._find_code_change(item_key)
                if change:
                    indicator = _get_indicator(change["change_type"])
                    hash_display = _format_hash_change(
                        change["old_hash"],
                        change["new_hash"],
                        change["change_type"],
                    )
                    return f"{prefix}{indicator} {rich.markup.escape(str(change['key'])):<25} {hash_display}{suffix}"
                return f"{prefix}{_INDICATOR_UNCHANGED} {rich.markup.escape(item_key):<25} (unknown){suffix}"

            case "dep":
                change = self._find_dep_change(item_key)
                rel_path = _get_relative_path(item_key)
                if change:
                    indicator = _get_indicator(change["change_type"])
                    hash_display = _format_hash_change(
                        change["old_hash"],
                        change["new_hash"],
                        change["change_type"],
                    )
                    return f"{prefix}{indicator} {rich.markup.escape(rel_path):<25} {hash_display}{suffix}"
                # Unchanged dep
                return f"{prefix}{_INDICATOR_UNCHANGED} {rich.markup.escape(rel_path):<25} (unchanged){suffix}"

            case "param":
                change = self._find_param_change(item_key)
                if change:
                    indicator = _get_indicator(change["change_type"])
                    old_val = (
                        repr(change["old_value"]) if change["old_value"] is not None else "(none)"
                    )
                    new_val = (
                        repr(change["new_value"]) if change["new_value"] is not None else "(none)"
                    )
                    match change["change_type"]:
                        case ChangeType.ADDED:
                            val_display = f"(none) -> {rich.markup.escape(new_val)}"
                        case ChangeType.REMOVED:
                            val_display = f"{rich.markup.escape(old_val)} -> (deleted)"
                        case ChangeType.MODIFIED:
                            val_display = (
                                f"{rich.markup.escape(old_val)} -> {rich.markup.escape(new_val)}"
                            )
                        case _ as unreachable:  # pyright: ignore[reportUnnecessaryComparison]
                            assert_never(unreachable)
                    return f"{prefix}{indicator} {rich.markup.escape(str(change['key'])):<25} {val_display}{suffix}"
                return f"{prefix}{_INDICATOR_UNCHANGED} {rich.markup.escape(item_key):<25} (unknown){suffix}"

            case _:
                return f"{prefix}{rich.markup.escape(item_id)}{suffix}"

    @override
    def _render_detail_content(self, item_id: str) -> str:  # pragma: no cover
        """Render detail content for the selected item."""
        item_type, item_key = item_id.split(":", 1)

        match item_type:
            case "code":
                return self._render_code_detail(item_key)
            case "dep":
                return self._render_dep_detail(item_key)
            case "param":
                return self._render_param_detail(item_key)
            case _:
                return "[dim]Unknown item type[/]"

    def _render_code_detail(self, key: str) -> str:  # pragma: no cover
        """Render detail for a code change."""
        change = self._find_code_change(key)
        if not change:
            return "[dim]No changes[/]"

        lines = [
            f"[bold]{rich.markup.escape(str(change['key']))}[/]",
            "",
            "Type: Code fingerprint",
            f"Status: {self._format_status(change['change_type'])}",
            "",
        ]
        self._append_hash_detail(
            lines, change["old_hash"], change["new_hash"], change["change_type"]
        )
        return "\n".join(lines)

    def _render_dep_detail(self, path: str) -> str:  # pragma: no cover
        """Render detail for a dependency change."""
        change = self._find_dep_change(path)
        rel_path = _get_relative_path(path)

        if not change:
            return f"[bold]{rich.markup.escape(rel_path)}[/]\n\n[dim]No changes[/]"

        lines = [
            f"[bold]{rich.markup.escape(rel_path)}[/]",
            "",
            "Type: Dependency",
            f"Status: {self._format_status(change['change_type'])}",
            "",
        ]
        self._append_hash_detail(
            lines, change["old_hash"], change["new_hash"], change["change_type"]
        )
        return "\n".join(lines)

    def _render_param_detail(self, key: str) -> str:  # pragma: no cover
        """Render detail for a parameter change."""
        change = self._find_param_change(key)

        if not change:
            return f"[bold]{rich.markup.escape(key)}[/]\n\n[dim]No changes[/]"

        lines = [
            f"[bold]{rich.markup.escape(str(change['key']))}[/]",
            "",
            "Type: Parameter",
            f"Status: {self._format_status(change['change_type'])}",
            "",
        ]

        old_val = change["old_value"]
        new_val = change["new_value"]

        if self._detail_expanded:
            # Show full values in expanded view
            lines.extend(
                [
                    f"Old value: {rich.markup.escape(repr(old_val)) if old_val is not None else '(none)'}",
                    f"New value: {rich.markup.escape(repr(new_val)) if new_val is not None else '(none)'}",
                ]
            )
            # Add delta for numeric values
            if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                delta = new_val - old_val
                if old_val != 0:
                    pct = delta / old_val * 100
                elif new_val != 0:
                    pct = float("inf")  # Any change from zero is infinite %
                else:
                    pct = 0.0  # 0 -> 0 is no change
                sign = "+" if delta >= 0 else ""
                lines.append(f"Delta: {sign}{delta} ({sign}{pct:.1f}%)")
        else:
            old_display = rich.markup.escape(repr(old_val)) if old_val is not None else "(none)"
            new_display = rich.markup.escape(repr(new_val)) if new_val is not None else "(none)"
            lines.append(f"{old_display} -> {new_display}")

        if not self._detail_expanded:
            lines.extend(["", "[dim]\\[Enter] Expand[/]"])

        return "\n".join(lines)

    def _find_code_change(self, key: str) -> CodeChange | None:
        """Find a code change by key."""
        return self._code_by_key.get(key)

    def _find_dep_change(self, path: str) -> DepChange | None:
        """Find a dependency change by path."""
        return self._dep_by_path.get(path)

    def _find_param_change(self, key: str) -> ParamChange | None:
        """Find a parameter change by key."""
        return self._param_by_key.get(key)

    @override
    def _render_empty_state(self) -> str:
        if self._stage_name is None:
            return "[dim]No stage selected[/]"
        if self._registry_info is None:
            return "[dim]Stage not in registry[/]"
        if self._explanation is None:
            return "[dim]Error loading stage data[/]"
        return "[dim]No inputs[/]"


class OutputDiffPanel(_SelectableExpandablePanel):
    """Panel showing output changes for a stage (outs, metrics, plots)."""

    # Cache for stage data
    _stage_name: str | None
    _registry_info: RegistryStageInfo | None
    # Dict-based storage for O(1) lookup by path
    _output_by_path: dict[str, OutputChange]
    _metric_diff_cache: dict[str, list[MetricDiff]]
    _head_hashes: dict[str, str | None] | None

    def __init__(self, *, id: str | None = None, classes: str | None = None) -> None:
        super().__init__(
            id=id, classes="diff-panel" if classes is None else f"diff-panel {classes}"
        )
        self._registry_info = None
        self._output_by_path = dict[str, OutputChange]()
        self._metric_diff_cache = dict[str, list[MetricDiff]]()
        self._head_hashes = None

    @override
    def set_stage(self, stage_name: str | None) -> None:  # pragma: no cover
        """Update the displayed stage and load data."""
        self._reset_selection_state()
        self._registry_info = None
        self._output_by_path.clear()
        self._metric_diff_cache.clear()
        self._head_hashes = None
        self._stage_name = stage_name

        if stage_name is not None:
            self._load_stage_data(stage_name)

        self._update_display()

    def _load_stage_data(self, stage_name: str) -> None:  # pragma: no cover
        """Load and cache stage data."""
        self._registry_info = _get_registry_info(stage_name)
        if self._registry_info is None:
            return

        cache_dir = _get_cache_dir()
        stage_lock = lock.StageLock(stage_name, cache_dir)
        lock_data = stage_lock.read()

        output_changes = _compute_output_changes(lock_data, self._registry_info)
        self._output_by_path = {c["path"]: c for c in output_changes}

    @override
    def _build_items(self) -> list[str]:  # pragma: no cover
        """Build item IDs from outputs, metrics, and plots."""
        items = list[str]()

        for path, change in self._output_by_path.items():
            output_type = change["output_type"]
            items.append(f"{output_type}:{path}")

        return items

    @override
    def _render_item_row(self, item_id: str, is_selected: bool) -> str:  # pragma: no cover
        """Render a single item row."""
        prefix = "[reverse]" if is_selected else ""
        suffix = "[/]" if is_selected else ""

        _, item_path = item_id.split(":", 1)
        change = self._find_output_change(item_path)

        if not change:
            rel_path = _get_relative_path(item_path)
            return f"{prefix}{_INDICATOR_UNCHANGED} {rich.markup.escape(rel_path):<25} (unknown){suffix}"

        indicator = _get_indicator(change["change_type"])
        rel_path = _get_relative_path(change["path"])
        hash_display = _format_hash_change(
            change["old_hash"], change["new_hash"], change["change_type"]
        )

        return f"{prefix}{indicator} {rich.markup.escape(rel_path):<25} {hash_display}{suffix}"

    @override
    def _render_detail_content(self, item_id: str) -> str:  # pragma: no cover
        """Render detail content for the selected item."""
        _, item_path = item_id.split(":", 1)
        change = self._find_output_change(item_path)

        if not change:
            return "[dim]No changes[/]"

        rel_path = _get_relative_path(item_path)
        # item_type is always one of OutputType values from _build_items
        item_type: OutputType = change["output_type"]
        type_label = self._get_type_label(item_type)

        lines = [
            f"[bold]{rich.markup.escape(rel_path)}[/]",
            "",
            f"Type: {type_label}",
            f"Status: {self._format_status(change['change_type'])}",
            "",
        ]

        # For metrics, show detailed diff
        if item_type == "metric" and change["change_type"] is not None:
            metric_diffs = self._get_metric_diffs(item_path)
            if metric_diffs:
                lines.append("[bold]Metric Values:[/]")
                for diff in metric_diffs:
                    old_val = diff["old"]
                    new_val = diff["new"]
                    old_str = self._format_metric_value(old_val)
                    new_str = self._format_metric_value(new_val)
                    delta_str = self._format_metric_delta(old_val, new_val)
                    lines.append(
                        f"  {rich.markup.escape(diff['key']):<20} {old_str} -> {new_str}  {delta_str}"
                    )
                lines.append("")

        self._append_hash_detail(
            lines, change["old_hash"], change["new_hash"], change["change_type"]
        )
        return "\n".join(lines)

    def _find_output_change(self, path: str) -> OutputChange | None:
        """Find an output change by path."""
        return self._output_by_path.get(path)

    def _get_type_label(self, item_type: OutputType) -> str:
        """Get human-readable label for output type."""
        match item_type:
            case "out":
                return "Output"
            case "metric":
                return "Metric"
            case "plot":
                return "Plot"
            case _ as unreachable:  # pyright: ignore[reportUnnecessaryComparison]
                assert_never(unreachable)

    def _get_metric_diffs(self, path: str) -> list[MetricDiff]:  # pragma: no cover
        """Get metric diffs for a path, using cache."""
        # Normalize path for cache key (fallback to raw path if resolve fails)
        try:
            cache_key = str(pathlib.Path(path).resolve())
        except OSError:
            cache_key = path

        if cache_key in self._metric_diff_cache:
            return self._metric_diff_cache[cache_key]

        # Load head hashes once per stage
        if self._head_hashes is None:
            try:
                self._head_hashes = metrics_mod.get_metric_info_from_head()
            except Exception:
                self._head_hashes = {}

        rel_path = _get_relative_path(path)

        try:
            head_metrics = metrics_mod.collect_metrics_from_head([rel_path], self._head_hashes)
            current_metrics = metrics_mod.collect_metrics_from_files([path], tolerant=True)
            diffs = metrics_mod.diff_metrics(head_metrics, current_metrics)
        except Exception:
            diffs = list[MetricDiff]()

        self._metric_diff_cache[cache_key] = diffs
        return diffs

    def _format_metric_value(self, value: MetricValue) -> str:
        """Format a metric value for display."""
        if value is None:
            return "(none)"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    def _format_metric_delta(self, old: MetricValue, new: MetricValue) -> str:
        """Format the delta between metric values."""
        if old is None or new is None:
            return ""
        if isinstance(old, (int, float)) and isinstance(new, (int, float)):
            delta = new - old
            sign = "+" if delta >= 0 else ""
            if isinstance(delta, float):
                return f"[dim]({sign}{delta:.4f})[/]"
            return f"[dim]({sign}{delta})[/]"
        return ""

    @override
    def _render_empty_state(self) -> str:
        if self._stage_name is None:
            return "[dim]No stage selected[/]"
        if self._registry_info is None:
            return "[dim]Stage not in registry[/]"
        return "[dim]No outputs[/]"
