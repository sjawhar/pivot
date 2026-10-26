from __future__ import annotations

import html
import json
import logging
import os
import pathlib
from typing import TYPE_CHECKING, TypedDict

from pivot import config, project, registry, types
from pivot.show import common
from pivot.storage import cache, state
from pivot.types import ChangeType, OutputFormat

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)


class PlotInfo(TypedDict):
    """A plot to render: its workspace path and the stage that produces it."""

    path: str
    stage_name: str


class PlotDiffEntry(TypedDict):
    """Result of comparing a plot file."""

    path: str
    old_hash: str | None
    new_hash: str | None
    change_type: ChangeType


def _plot_outs_by_stage() -> dict[str, list[types.ArtifactRef]]:
    from pivot.cli import helpers as cli_helpers

    outs_by_stage = dict[str, list[types.ArtifactRef]]()
    for stage_name in cli_helpers.list_stages():
        plot_outs = [
            out
            for out in cli_helpers.get_stage(stage_name)["outs"]
            if out.tag is types.ArtifactTag.PLOT
        ]
        if plot_outs:
            outs_by_stage[stage_name] = plot_outs
    return outs_by_stage


def collect_plots_from_stages() -> list[PlotInfo]:
    """List the plot outputs of the Pipeline in context at their workspace paths."""
    from pivot.cli import helpers as cli_helpers

    store = cli_helpers.require_workspace_store()
    proj_root = project.get_project_root()
    return [
        PlotInfo(
            path=project.to_relative_path(store.resolve_display_path(out), proj_root),
            stage_name=stage_name,
        )
        for stage_name, plot_outs in _plot_outs_by_stage().items()
        for out in plot_outs
    ]


def get_plot_hashes_from_head() -> dict[str, str | None]:
    """Map each plot output's workspace path to its hash in the lock files at git HEAD."""
    from pivot.cli import helpers as cli_helpers

    return common.output_hashes_from_head(
        _plot_outs_by_stage(), cli_helpers.require_workspace_store()
    )


def plot_state_dirs() -> dict[str, pathlib.Path]:
    """Map each plot output's workspace path to its stage's state directory, whose StateDB
    holds the output's file-hash entry."""
    from pivot.cli import helpers as cli_helpers

    store = cli_helpers.require_workspace_store()
    proj_root = project.get_project_root()
    default_state_dir = config.get_state_dir()
    state_dirs = dict[str, pathlib.Path]()
    for stage_name, plot_outs in _plot_outs_by_stage().items():
        stage_info = cli_helpers.get_stage(stage_name)
        state_dir = registry.get_stage_state_dir(stage_info, default_state_dir)
        for out in plot_outs:
            path = project.to_relative_path(store.resolve_display_path(out), proj_root)
            state_dirs[path] = state_dir
    return state_dirs


def get_plot_hashes_from_workspace(
    paths: Sequence[str],
    state_dirs: Mapping[str, pathlib.Path] | None = None,
) -> dict[str, str]:
    """Compute current file hashes for plot files.

    Paths are relative to project root. Each file is hashed as ``pivot status`` hashes it: the
    StateDB of ``state_dirs[path]`` (``plot_state_dirs``; the configured state directory for a
    path not in it) is read read-only, none is created, and the hashes read from files are
    saved afterwards with one best-effort ``state.write_back_file_hashes`` per state directory.
    """
    proj_root = project.get_project_root()
    default_state_dir = config.get_state_dir()
    by_state_dir = dict[pathlib.Path, list[tuple[str, pathlib.Path]]]()
    for path_str in paths:
        path = proj_root / path_str
        if path.exists() and path.is_file():
            state_dir = (state_dirs or {}).get(path_str, default_state_dir)
            by_state_dir.setdefault(state_dir, []).append((path_str, path))
    result = dict[str, str]()
    for state_dir, files in by_state_dir.items():
        read = list[tuple[pathlib.Path, os.stat_result, str]]()
        with state.open_readonly(state_dir, create=False) as state_db:
            for path_str, path in files:
                file_hash, _ = cache.hash_file(path, state_db, pending=read)
                result[path_str] = file_hash
        state.write_back_file_hashes(state_dir, cache.deferred_file_hash_entries(read))
    return result


def diff_plots(
    old: Mapping[str, str | None],
    new: Mapping[str, str | None],
) -> list[PlotDiffEntry]:
    """Compare plot hashes between two sources (HEAD vs workspace, or rev vs rev)."""
    diffs = list[PlotDiffEntry]()
    all_paths = set(old.keys()) | set(new.keys())

    for path in sorted(all_paths):
        old_hash = old.get(path)
        new_hash = new.get(path)

        # No previous version (not in old OR old_hash is None)
        if path not in old or old_hash is None:
            if new_hash is not None:
                diffs.append(
                    PlotDiffEntry(
                        path=path, old_hash=None, new_hash=new_hash, change_type=ChangeType.ADDED
                    )
                )
            # Both None: silently ignore (not tracked in either source)
        elif path not in new or new_hash is None:
            diffs.append(
                PlotDiffEntry(
                    path=path, old_hash=old_hash, new_hash=None, change_type=ChangeType.REMOVED
                )
            )
        elif old_hash != new_hash:
            diffs.append(
                PlotDiffEntry(
                    path=path, old_hash=old_hash, new_hash=new_hash, change_type=ChangeType.MODIFIED
                )
            )

    return diffs


def format_diff_table(
    diffs: list[PlotDiffEntry],
    output_format: OutputFormat | None,
    show_path: bool = True,
) -> str:
    """Format diff output as plain text, JSON, or markdown."""
    if output_format == OutputFormat.JSON:
        return json.dumps(diffs, indent=2)

    rows = list[list[str]]()
    for diff in diffs:
        old_str = "-" if diff["old_hash"] is None else diff["old_hash"][:8]
        new_str = "-" if diff["new_hash"] is None else diff["new_hash"][:8]
        row = [old_str, new_str, diff["change_type"]]
        if show_path:
            row.insert(0, diff["path"])
        rows.append(row)

    headers = ["Old", "New", "Change"]
    if show_path:
        headers.insert(0, "Path")

    return common.format_table(rows, headers, output_format, "No plot changes.")


def render_plots_html(
    plots: list[PlotInfo],
    output_path: pathlib.Path,
) -> pathlib.Path:
    """Generate simple HTML image gallery."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    proj_root = project.get_project_root()

    plot_divs = list[str]()
    for plot in plots:
        # Resolve path from project root
        path = pathlib.Path(plot["path"])
        if not path.is_absolute():
            path = proj_root / path
        if not path.exists():
            continue

        # Reference the plot relative to the HTML file so the gallery stays valid when the
        # project directory is moved or opened from another machine.
        rel_path = pathlib.Path(os.path.relpath(path.resolve(), output_path.parent.resolve()))

        # Escape user-provided content to prevent XSS
        safe_path = html.escape(plot["path"])
        safe_stage = html.escape(plot["stage_name"])
        safe_rel_path = html.escape(str(rel_path))

        plot_divs.append(f"""  <div class="plot">
    <h3>{safe_path}</h3>
    <p><small>Stage: {safe_stage}</small></p>
    <img src="{safe_rel_path}" alt="{safe_path}">
  </div>""")

    html_content = f"""<!DOCTYPE html>
<html>
<head>
  <title>Pivot Plots</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
    .plot {{ margin-bottom: 30px; padding: 15px; border: 1px solid #ddd; border-radius: 8px; }}
    .plot img {{ max-width: 100%; height: auto; }}
    .plot h3 {{ margin: 0 0 5px 0; color: #333; }}
    .plot p {{ margin: 0 0 10px 0; color: #666; }}
  </style>
</head>
<body>
  <h1>Pivot Plots</h1>
  <p>{len(plot_divs)} plot(s)</p>
{chr(10).join(plot_divs) if plot_divs else "  <p>No plots found.</p>"}
</body>
</html>"""

    _ = output_path.write_text(html_content)
    return output_path
