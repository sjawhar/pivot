"""Input and Output diff panels for the TUI.

Displays stage change information in the Input and Output tabs:
- Input tab: code changes, dependency changes, parameter changes
- Output tab: output file changes grouped by type (Out, Metric, Plot)
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Literal

import textual.widgets

from pivot import explain, outputs, parameters, project
from pivot.registry import REGISTRY
from pivot.storage import cache, lock
from pivot.types import ChangeType, OutputChange

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


class InputDiffPanel(textual.widgets.Static):
    """Panel showing input changes for a stage (code, deps, params)."""

    _stage_name: str | None

    def __init__(self, *, id: str | None = None, classes: str | None = None) -> None:
        super().__init__(id=id, classes=classes)
        self._stage_name = None

    def set_stage(self, stage_name: str | None) -> None:  # pragma: no cover
        """Update the displayed stage."""
        self._stage_name = stage_name
        self._update_display()

    def _update_display(self) -> None:  # pragma: no cover
        """Render the input changes."""
        if self._stage_name is None:
            self.update("[dim]No stage selected[/]")
            return

        registry_info = _get_registry_info(self._stage_name)
        if registry_info is None:
            self.update("[dim]Stage not in registry[/]")
            return

        # Get stage explanation
        cache_dir = _get_cache_dir()
        try:
            explanation = explain.get_stage_explanation(
                stage_name=self._stage_name,
                fingerprint=registry_info["fingerprint"],
                deps=registry_info["deps"],
                params_instance=registry_info["params"],
                overrides=parameters.load_params_yaml(),
                cache_dir=cache_dir,
            )
        except Exception:
            self.update("[dim]Error loading stage explanation[/]")
            return

        lines = list[str]()

        # Code section
        code_changes = explanation["code_changes"]
        if code_changes:
            lines.append("[bold]Code: Changed[/]")
            for change in code_changes:
                indicator = _get_indicator(change["change_type"])
                hash_display = _format_hash_change(
                    change["old_hash"], change["new_hash"], change["change_type"]
                )
                lines.append(f"  {indicator} {change['key']:<25} {hash_display}")
        else:
            lines.append("[bold]Code:[/] [dim](unchanged)[/]")

        lines.append("")

        # Dependencies section
        dep_changes = explanation["dep_changes"]
        lines.append("[bold]Dependencies:[/]")
        if dep_changes:
            for change in dep_changes:
                indicator = _get_indicator(change["change_type"])
                rel_path = _get_relative_path(change["path"])
                hash_display = _format_hash_change(
                    change["old_hash"], change["new_hash"], change["change_type"]
                )
                lines.append(f"  {indicator} {rel_path:<25} {hash_display}")
        else:
            # Show deps from registry as unchanged
            deps = registry_info["deps"]
            if deps:
                for dep_path in deps:
                    rel_path = _get_relative_path(dep_path)
                    lines.append(f"  {_INDICATOR_UNCHANGED} {rel_path:<25} (unchanged)")
            else:
                lines.append("  [dim]No dependencies[/]")

        lines.append("")

        # Parameters section
        param_changes = explanation["param_changes"]
        if param_changes:
            lines.append("[bold]Parameters: Changed[/]")
            for change in param_changes:
                indicator = _get_indicator(change["change_type"])
                old_val = repr(change["old_value"]) if change["old_value"] is not None else "(none)"
                new_val = repr(change["new_value"]) if change["new_value"] is not None else "(none)"
                match change["change_type"]:
                    case ChangeType.ADDED:
                        val_display = f"(none) -> {new_val}"
                    case ChangeType.REMOVED:
                        val_display = f"{old_val} -> (deleted)"
                    case ChangeType.MODIFIED:
                        val_display = f"{old_val} -> {new_val}"
                lines.append(f"  {indicator} {change['key']:<25} {val_display}")
        else:
            lines.append("[bold]Parameters:[/] [dim](unchanged)[/]")

        self.update("\n".join(lines))


class OutputDiffPanel(textual.widgets.Static):
    """Panel showing output changes for a stage (outs, metrics, plots)."""

    _stage_name: str | None

    def __init__(self, *, id: str | None = None, classes: str | None = None) -> None:
        super().__init__(id=id, classes=classes)
        self._stage_name = None

    def set_stage(self, stage_name: str | None) -> None:  # pragma: no cover
        """Update the displayed stage."""
        self._stage_name = stage_name
        self._update_display()

    def _update_display(self) -> None:  # pragma: no cover
        """Render the output changes."""
        if self._stage_name is None:
            self.update("[dim]No stage selected[/]")
            return

        registry_info = _get_registry_info(self._stage_name)
        if registry_info is None:
            self.update("[dim]Stage not in registry[/]")
            return

        # Read lock file
        cache_dir = _get_cache_dir()
        stage_lock = lock.StageLock(self._stage_name, cache_dir)
        lock_data = stage_lock.read()

        # Compute output changes
        output_changes = _compute_output_changes(lock_data, registry_info)

        # Group by type
        outs_list = [c for c in output_changes if c["output_type"] == "out"]
        metrics_list = [c for c in output_changes if c["output_type"] == "metric"]
        plots_list = [c for c in output_changes if c["output_type"] == "plot"]

        lines = list[str]()

        # Outputs section
        lines.append("[bold]Outputs:[/]")
        if outs_list:
            for change in outs_list:
                indicator = _get_indicator(change["change_type"])
                rel_path = _get_relative_path(change["path"])
                hash_display = _format_hash_change(
                    change["old_hash"], change["new_hash"], change["change_type"]
                )
                lines.append(f"  {indicator} {rel_path:<25} {hash_display}")
        else:
            lines.append("  [dim]No outputs[/]")

        lines.append("")

        # Metrics section
        lines.append("[bold]Metrics:[/]")
        if metrics_list:
            for change in metrics_list:
                indicator = _get_indicator(change["change_type"])
                rel_path = _get_relative_path(change["path"])
                hash_display = _format_hash_change(
                    change["old_hash"], change["new_hash"], change["change_type"]
                )
                lines.append(f"  {indicator} {rel_path:<25} {hash_display}")
        else:
            lines.append("  [dim]No metrics[/]")

        lines.append("")

        # Plots section
        lines.append("[bold]Plots:[/]")
        if plots_list:
            for change in plots_list:
                indicator = _get_indicator(change["change_type"])
                rel_path = _get_relative_path(change["path"])
                hash_display = _format_hash_change(
                    change["old_hash"], change["new_hash"], change["change_type"]
                )
                lines.append(f"  {indicator} {rel_path:<25} {hash_display}")
        else:
            lines.append("  [dim]No plots[/]")

        self.update("\n".join(lines))
