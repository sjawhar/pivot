from __future__ import annotations

import html
import json
import pathlib
from typing import TYPE_CHECKING, Any, TypedDict, cast

import tabulate
import yaml

from pivot import cache, git, lock, outputs, project, yaml_config
from pivot.types import ChangeType, OutEntry  # noqa: TC001 - runtime import for TypedDict

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pivot.types import OutputFormat


class PlotInfo(TypedDict):
    """Info about a registered Plot output."""

    path: str
    stage_name: str
    x: str | None
    y: str | None
    template: str | None


class PlotDiffEntry(TypedDict):
    """Result of comparing a plot file."""

    path: str
    old_hash: str | None
    new_hash: str | None
    change: ChangeType


def collect_plots_from_stages() -> list[PlotInfo]:
    """Discover Plot outputs from registry."""
    from pivot import registry

    result = list[PlotInfo]()
    for stage_name in registry.REGISTRY.list_stages():
        info = registry.REGISTRY.get(stage_name)
        for out in info["outs"]:
            if isinstance(out, outputs.Plot):
                result.append(
                    PlotInfo(
                        path=out.path,
                        stage_name=stage_name,
                        x=out.x,
                        y=out.y,
                        template=out.template,
                    )
                )
    return result


def get_plot_hashes_from_lock(
    cache_dir: pathlib.Path | None = None,
) -> dict[str, str | None]:
    """Read output_hashes for plots from lock files.

    Returns paths relative to project root for consistent comparison with user input.
    """
    from pivot import registry

    if cache_dir is None:
        cache_dir = project.get_project_root() / ".pivot" / "cache"

    proj_root = project.get_project_root()
    result = dict[str, str | None]()
    for stage_name in registry.REGISTRY.list_stages():
        info = registry.REGISTRY.get(stage_name)
        stage_lock = lock.StageLock(stage_name, cache_dir)
        lock_data = stage_lock.read()

        for out in info["outs"]:
            if isinstance(out, outputs.Plot):
                # Normalize to absolute for lock data lookup, then convert to relative for result
                abs_path = str(project.normalize_path(out.path))
                rel_path = project.to_relative_path(abs_path, proj_root)
                if lock_data and abs_path in lock_data["output_hashes"]:
                    hash_info = lock_data["output_hashes"][abs_path]
                    result[rel_path] = hash_info["hash"] if hash_info else None
                else:
                    result[rel_path] = None
    return result


def get_plot_hashes_from_head() -> dict[str, str | None]:
    """Read output_hashes for plots from lock files at Git HEAD.

    Returns paths relative to project root for consistent comparison with workspace.
    Returns empty dict if not in a git repo or no HEAD commit exists.
    """
    from pivot import registry

    result = dict[str, str | None]()
    proj_root = project.get_project_root()

    # Collect lock file paths we need to read
    stage_plot_paths: dict[str, list[str]] = {}  # stage_name -> [rel_plot_paths]
    for stage_name in registry.REGISTRY.list_stages():
        info = registry.REGISTRY.get(stage_name)
        for out in info["outs"]:
            if isinstance(out, outputs.Plot):
                if stage_name not in stage_plot_paths:
                    stage_plot_paths[stage_name] = []
                abs_path = str(project.normalize_path(out.path))
                rel_path = project.to_relative_path(abs_path, proj_root)
                stage_plot_paths[stage_name].append(rel_path)
                result[rel_path] = None  # Default to None

    # Read all lock files from HEAD in one batch
    lock_paths = [f".pivot/cache/stages/{name}.lock" for name in stage_plot_paths]
    lock_contents = git.read_files_from_head(lock_paths)

    # Parse lock files and extract plot hashes
    for stage_name, plot_paths in stage_plot_paths.items():
        lock_path = f".pivot/cache/stages/{stage_name}.lock"
        content = lock_contents.get(lock_path)
        if content is None:
            continue

        try:
            data = yaml.load(content, Loader=yaml_config.Loader)
        except yaml.YAMLError:
            continue

        if not isinstance(data, dict) or "outs" not in data:
            continue

        # Build path->hash lookup from storage format (uses relative paths)
        outs_raw = cast("dict[str, Any]", data)["outs"]
        if not isinstance(outs_raw, list):
            continue
        outs_list = cast("list[OutEntry]", outs_raw)

        path_to_hash = dict[str, str | None]()
        for out in outs_list:
            if "path" in out:
                path_to_hash[out["path"]] = out["hash"]

        # Match our plot paths against storage paths
        for plot_rel_path in plot_paths:
            if plot_rel_path in path_to_hash:
                result[plot_rel_path] = path_to_hash[plot_rel_path]

    return result


def get_plot_hashes_from_workspace(
    paths: Sequence[str],
) -> dict[str, str]:
    """Compute current file hashes for plot files.

    Paths are relative to project root.
    """
    proj_root = project.get_project_root()
    result = dict[str, str]()
    for path_str in paths:
        path = proj_root / path_str
        if path.exists() and path.is_file():
            result[path_str] = cache.hash_file(path)
    return result


def diff_plots(
    old: Mapping[str, str | None],
    new: Mapping[str, str],
) -> list[PlotDiffEntry]:
    """Compare lock file hashes vs workspace hashes."""
    diffs = list[PlotDiffEntry]()
    all_paths = set(old.keys()) | set(new.keys())

    for path in sorted(all_paths):
        old_hash = old.get(path)
        new_hash = new.get(path)

        # Not in old OR old_hash is None means no previous version
        if path not in old or old_hash is None:
            if new_hash is not None:
                diffs.append(
                    PlotDiffEntry(
                        path=path, old_hash=None, new_hash=new_hash, change=ChangeType.ADDED
                    )
                )
        elif path not in new or new_hash is None:
            diffs.append(
                PlotDiffEntry(
                    path=path, old_hash=old_hash, new_hash=None, change=ChangeType.REMOVED
                )
            )
        elif old_hash != new_hash:
            diffs.append(
                PlotDiffEntry(
                    path=path, old_hash=old_hash, new_hash=new_hash, change=ChangeType.MODIFIED
                )
            )

    return diffs


def format_diff_table(
    diffs: list[PlotDiffEntry],
    output_format: OutputFormat,
    show_path: bool = True,
) -> str:
    """Format diff output as plain text, JSON, or markdown."""
    if output_format == "json":
        return json.dumps(diffs, indent=2)

    if not diffs:
        return "No plot changes."

    rows = list[list[str]]()
    for diff in diffs:
        old_str = "-" if diff["old_hash"] is None else diff["old_hash"][:8]
        new_str = "-" if diff["new_hash"] is None else diff["new_hash"][:8]
        row = [old_str, new_str, diff["change"]]
        if show_path:
            row.insert(0, diff["path"])
        rows.append(row)

    headers = ["Old", "New", "Change"]
    if show_path:
        headers.insert(0, "Path")

    tablefmt = "github" if output_format == "md" else "plain"
    return tabulate.tabulate(rows, headers=headers, tablefmt=tablefmt, disable_numparse=True)


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

        # Use relative path from output HTML to plot file
        try:
            rel_path = path.resolve().relative_to(output_path.parent.resolve())
        except ValueError:
            rel_path = path.resolve()

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

    output_path.write_text(html_content)
    return output_path
