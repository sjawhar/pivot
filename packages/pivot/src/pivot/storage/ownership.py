"""Keep Pivot from deleting tracked and imported files that overlap stage outputs.

A stage output is Pivot's to replace. A tracked file, an imported file and their ``.pvt``
sidecars are the user's. ``track`` and ``import`` refuse a new path inside any pipeline's
declared output (``discovery.declared_outputs``); a project can still hold one from before that
check, so every write, restore or commit that replaces an output first looks for sidecars where
the replacement reaches and fails, naming them, before anything is removed. A stage can name an
ordinary output ``*.pvt``: the worker passes those exact paths as ``ignored_pvt_paths`` after it
has derived them from the stage return value, so an actual existing sidecar is still protected.
"""

from __future__ import annotations

import os
import pathlib
from typing import TypedDict

from pivot import exceptions, project
from pivot.storage import track


class OutputOverlap(TypedDict):
    """A ``.pvt`` sidecar that replacing ``output_path`` would remove or orphan."""

    tracked_path: pathlib.Path | None
    pvt_path: pathlib.Path
    output_path: pathlib.Path


def paths_overlap(path_a: pathlib.Path, path_b: pathlib.Path) -> bool:
    """Whether two paths are the same location, or one lies inside the other."""
    # Exact match - use samefile if both exist (handles hardlinks, case-insensitivity)
    if path_a.exists() and path_b.exists():
        try:
            if path_a.samefile(path_b):
                return True
        except OSError:
            pass
    return path_a == path_b or path_a.is_relative_to(path_b) or path_b.is_relative_to(path_a)


def find_output_overlaps(
    project_root: pathlib.Path,
    output_paths: list[pathlib.Path],
    ignored_pvt_paths: set[pathlib.Path] | None = None,
) -> list[OutputOverlap]:
    """Find the ``.pvt`` sidecars that replacing ``output_paths`` would remove or orphan.

    Looks only where a replacement reaches: every file named ``*.pvt`` under each output that is
    a real directory (removing a symlink removes nothing it points to), and the sidecars of each
    output and of each directory above it up to the project root, which track the output itself
    or a directory holding it. It never walks the rest of the project, so it does not see a
    hand-edited ``.pvt`` elsewhere whose ``path`` names a file inside an output; Pivot never
    writes one.
    """
    ignored = ignored_pvt_paths or set[pathlib.Path]()
    root = pathlib.Path(os.path.normpath(project_root.absolute()))
    found = dict[pathlib.Path, OutputOverlap]()
    checked = set[pathlib.Path]()
    for output in output_paths:
        output_path = pathlib.Path(os.path.normpath(output))
        current = output_path
        while current != root and current.is_relative_to(root) and current not in checked:
            checked.add(current)
            pvt_path = track.get_pvt_path(current)
            data = track.read_pvt_file(pvt_path)
            if data is not None and pvt_path.parent / data["path"] == current:
                found[pvt_path] = OutputOverlap(
                    tracked_path=current, pvt_path=pvt_path, output_path=output_path
                )
            current = current.parent

        if not output_path.is_dir() or output_path.is_symlink():
            continue
        for dirpath, _dirnames, filenames in os.walk(output_path):
            for name in filenames:
                if not name.endswith(".pvt"):
                    continue
                pvt_path = pathlib.Path(dirpath) / name
                if pvt_path in ignored:
                    continue
                data = track.read_pvt_file(pvt_path)
                found[pvt_path] = OutputOverlap(
                    tracked_path=None if data is None else pvt_path.parent / data["path"],
                    pvt_path=pvt_path,
                    output_path=output_path,
                )
    return sorted(found.values(), key=lambda overlap: overlap["pvt_path"])


def _display(path: pathlib.Path, project_root: pathlib.Path) -> str:
    relative = project.relative_within(path, project_root)
    return str(path if relative is None else relative)


def _describe(overlap: OutputOverlap, project_root: pathlib.Path) -> str:
    output = _display(overlap["output_path"], project_root)
    pvt = _display(overlap["pvt_path"], project_root)
    if overlap["tracked_path"] is None:
        return f"sidecar '{pvt}' overlaps stage output '{output}'"
    tracked = _display(overlap["tracked_path"], project_root)
    return f"tracked path '{tracked}' (sidecar '{pvt}') overlaps stage output '{output}'"


def refuse_output_replacements(
    project_root: pathlib.Path,
    output_paths: list[pathlib.Path],
    ignored_pvt_paths: set[pathlib.Path] | None = None,
) -> None:
    """Fail, naming each one, before replacing outputs would remove a tracked path or sidecar."""
    overlaps = find_output_overlaps(project_root, output_paths, ignored_pvt_paths)
    if not overlaps:
        return
    details = "\n".join(f"  - {_describe(overlap, project_root)}" for overlap in overlaps)
    raise exceptions.OutputOwnershipError(
        "Refusing to replace a stage output: it would remove user-owned files:\n"
        + details
        + "\nMove each tracked path and its sidecar outside the stage output, then rerun."
    )


def saved_input_warnings(project_root: pathlib.Path, output_paths: list[pathlib.Path]) -> list[str]:
    """One warning per sidecar that already overlaps one of ``output_paths``."""
    return [
        f"Warning: {_describe(overlap, project_root)}. Move both outside the output (for example "
        + "under data/raw/); until then Pivot refuses to replace that output."
        for overlap in find_output_overlaps(project_root, output_paths)
    ]
