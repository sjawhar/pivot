"""Keep Pivot from deleting tracked and imported files that overlap stage outputs.

A stage output is Pivot's to replace. A tracked file, an imported file and their ``.pvt``
sidecars are the user's. ``track`` and ``import`` refuse a new path inside any pipeline's
declared output (``discovery.declared_outputs``). A project can still hold one from before that
check, so every write, restore or commit that replaces an output first looks for sidecars where
the replacement reaches and fails, naming them, before anything is removed.

A stage may return a directory member named ``*.pvt``. Such a file is the stage's when the stage
records it (``StageMembers``): a sidecar Pivot cannot parse at a recorded path, or a file whose
content the stage recorded there. Every other ``*.pvt`` under an output is the user's.
"""

from __future__ import annotations

import os
import pathlib
from typing import TYPE_CHECKING, TypedDict

from pivot import exceptions, project, types
from pivot.storage import cache, lock, track

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from pivot.registry import RegistryStageInfo
    from pivot.storage.store import WorkspaceStore

# The ``*.pvt`` files a stage itself writes in one output: path -> the content hashes the stage
# recorded there, or None for a file the stage wrote in this run (any content there is its own).
type StageMembers = dict[pathlib.Path, frozenset[str] | None]


class Replacement(TypedDict):
    """A path Pivot is about to remove or replace."""

    path: pathlib.Path
    stage: str | None
    """The stage whose output ``path`` is, when the caller knows it."""
    destination: bool
    """True for a path the user named (``pivot get --output``), not a stage output."""


class DeclaredOutput(TypedDict):
    """A stage output some pipeline declares: its resolved path and the stage that writes it."""

    resolved: pathlib.Path
    stage: str


class OutputOverlap(TypedDict):
    """A ``.pvt`` sidecar that replacing ``replacement`` would remove or orphan."""

    tracked_path: pathlib.Path | None
    """The path the sidecar tracks; None when Pivot cannot parse the sidecar."""
    pvt_path: pathlib.Path
    replacement: Replacement


def stage_output(path: pathlib.Path, stage: str | None = None) -> Replacement:
    return Replacement(path=path, stage=stage, destination=False)


def destination(path: pathlib.Path) -> Replacement:
    return Replacement(path=path, stage=None, destination=True)


def normalized(path: pathlib.Path) -> pathlib.Path:
    """The logical absolute path the ownership walk compares (symlinks kept)."""
    return pathlib.Path(os.path.normpath(path.absolute()))


def recorded_members(output_path: pathlib.Path, *hashes: types.HashInfo | None) -> StageMembers:
    """The ``*.pvt`` members ``hashes`` record in ``output_path``, with each recorded hash."""
    root = normalized(output_path)
    found = dict[pathlib.Path, set[str]]()
    for output_hash in hashes:
        if output_hash is None or not types.is_dir_hash(output_hash):
            continue
        for entry in output_hash["manifest"]:
            if entry["relpath"].endswith(".pvt"):
                found.setdefault(root / entry["relpath"], set[str]()).add(entry["hash"])
    return {path: frozenset(recorded) for path, recorded in found.items()}


def _same_location(path_a: pathlib.Path, path_b: pathlib.Path) -> bool:
    if path_a == path_b:
        return True
    # samefile when both exist (handles hardlinks, case-insensitivity)
    if path_a.exists() and path_b.exists():
        try:
            return path_a.samefile(path_b)
        except OSError:
            return False
    return False


def paths_overlap(path_a: pathlib.Path, path_b: pathlib.Path) -> bool:
    """Whether two paths are the same location, or one lies inside the other."""
    return (
        _same_location(path_a, path_b)
        or path_a.is_relative_to(path_b)
        or path_b.is_relative_to(path_a)
    )


def _is_stage_member(
    pvt_path: pathlib.Path,
    data: track.PvtData | None,
    members: StageMembers,
    claim_unparsed: bool,
) -> bool:
    if pvt_path not in members:
        return claim_unparsed and data is None
    recorded = members[pvt_path]
    return recorded is None or data is None or cache.hash_file(pvt_path)[0] in recorded


def find_output_overlaps(
    project_root: pathlib.Path,
    replacements: Sequence[Replacement],
    members_of: Callable[[pathlib.Path], StageMembers] | None = None,
    *,
    claim_unparsed: bool = False,
) -> list[OutputOverlap]:
    """Find the ``.pvt`` sidecars that replacing each path would remove or orphan.

    Looks only where a replacement reaches. At the path and at each directory above it up to the
    project root, the sidecar Pivot would write for that path counts unless it parses and names
    another path. Under a path that is a real directory (removing a symlink removes nothing it
    points to), every ``*.pvt`` file counts unless it is one of the stage's own members:
    ``members_of`` maps a replaced path to them and is called only for a path that holds a
    ``*.pvt``. ``claim_unparsed`` (a commit, which records what is on disk as the stage's output)
    also gives the stage every ``*.pvt`` under the path that Pivot cannot parse.

    It never walks the rest of the project, so it does not see a hand-edited ``.pvt`` elsewhere
    whose ``path`` names a file inside an output; Pivot never writes one.
    """
    root = normalized(project_root)
    found = dict[pathlib.Path, OutputOverlap]()
    checked = set[pathlib.Path]()
    for replacement in replacements:
        output_path = normalized(replacement["path"])
        current = output_path
        while current != root and current.is_relative_to(root) and current not in checked:
            checked.add(current)
            pvt_path = track.get_pvt_path(current)
            if pvt_path.is_file() or pvt_path.is_symlink():
                data = track.read_pvt_file(pvt_path)
                if data is None or pvt_path.parent / data["path"] == current:
                    found[pvt_path] = OutputOverlap(
                        tracked_path=None if data is None else current,
                        pvt_path=pvt_path,
                        replacement=replacement,
                    )
            current = current.parent

        if not output_path.is_dir() or output_path.is_symlink():
            continue
        members: StageMembers | None = None
        for dirpath, _dirnames, filenames in os.walk(output_path):
            for name in filenames:
                if not name.endswith(".pvt"):
                    continue
                pvt_path = pathlib.Path(dirpath) / name
                data = track.read_pvt_file(pvt_path)
                if members is None:
                    members = members_of(output_path) if members_of is not None else {}
                if _is_stage_member(pvt_path, data, members, claim_unparsed):
                    continue
                found[pvt_path] = OutputOverlap(
                    tracked_path=None if data is None else pvt_path.parent / data["path"],
                    pvt_path=pvt_path,
                    replacement=replacement,
                )
    return sorted(found.values(), key=lambda overlap: overlap["pvt_path"])


def _display(path: pathlib.Path, project_root: pathlib.Path) -> str:
    relative = project.relative_within(path, project_root)
    return str(path if relative is None else relative)


def _label(replacement: Replacement, project_root: pathlib.Path) -> str:
    path = _display(replacement["path"], project_root)
    if replacement["destination"]:
        return f"destination '{path}'"
    if replacement["stage"] is None:
        return f"stage output '{path}'"
    return f"output '{path}' of stage '{replacement['stage']}'"


def _relation(path: pathlib.Path, replaced: pathlib.Path) -> str:
    if path == replaced:
        return "is"
    if path.is_relative_to(replaced):
        return "is inside"
    return "contains"


def _describe(overlap: OutputOverlap, project_root: pathlib.Path) -> str:
    label = _label(overlap["replacement"], project_root)
    replaced = normalized(overlap["replacement"]["path"])
    pvt = _display(overlap["pvt_path"], project_root)
    tracked_path = overlap["tracked_path"]
    if tracked_path is not None:
        tracked = _display(tracked_path, project_root)
        relation = _relation(normalized(tracked_path), replaced)
        return f"tracked path '{tracked}' (sidecar '{pvt}') {relation} {label}"
    unreadable = f"sidecar '{pvt}', which this version of Pivot cannot read,"
    if overlap["pvt_path"].is_relative_to(replaced):
        return f"{unreadable} is inside {label}"
    data_path = track.get_data_path(overlap["pvt_path"])
    if data_path == replaced:
        return f"{unreadable} may track {label} itself"
    return f"{unreadable} may track '{_display(data_path, project_root)}', which contains {label}"


def refuse_output_replacements(
    project_root: pathlib.Path,
    replacements: Sequence[Replacement],
    members_of: Callable[[pathlib.Path], StageMembers] | None = None,
    *,
    claim_unparsed: bool = False,
) -> None:
    """Fail, naming each one, before replacing these paths would remove a user's sidecar."""
    overlaps = find_output_overlaps(
        project_root, replacements, members_of, claim_unparsed=claim_unparsed
    )
    if not overlaps:
        return
    details = "\n".join(f"  - {_describe(overlap, project_root)}" for overlap in overlaps)
    if all(overlap["replacement"]["destination"] for overlap in overlaps):
        raise exceptions.OutputOwnershipError(
            "Refusing to replace a path that holds user-owned files:\n"
            + details
            + "\nChoose another --output path, or move each tracked path and its sidecar first."
        )
    raise exceptions.OutputOwnershipError(
        "Refusing to replace a stage output that holds user-owned files:\n"
        + details
        + "\nMove each tracked path and its sidecar outside every stage output (for example "
        + "under data/raw/), then rerun."
    )


def find_declared_overlap(
    path: pathlib.Path, outputs: Mapping[str, DeclaredOutput], project_root: pathlib.Path
) -> str | None:
    """Say how ``path`` overlaps the first declared output it overlaps, or return None.

    Paths are compared resolved, so a path that reaches an output through a symlink overlaps it.
    """
    resolved = path.resolve()
    for output, declared in outputs.items():
        out_resolved = declared["resolved"]
        if not paths_overlap(resolved, out_resolved):
            continue
        if _same_location(resolved, out_resolved):
            relation = "is"
        elif resolved.is_relative_to(out_resolved):
            relation = "is inside"
        else:
            relation = "contains"
        text = (
            f"it {relation} output '{_display(pathlib.Path(output), project_root)}' "
            + f"of stage '{declared['stage']}'"
        )
        if resolved != normalized(path) or out_resolved != pathlib.Path(output):
            text += f" (the path resolves to '{resolved}'; the output resolves to '{out_resolved}')"
        return text
    return None


def _warning(overlap: OutputOverlap, project_root: pathlib.Path) -> str:
    moved = "the tracked path and its sidecar" if overlap["tracked_path"] is not None else "it"
    return (
        f"Warning: {_describe(overlap, project_root)}. Move {moved} outside every stage output "
        + "(for example under data/raw/); until then Pivot refuses to replace that output."
    )


def saved_input_warnings(
    project_root: pathlib.Path,
    stages: Mapping[str, RegistryStageInfo],
    workspace: WorkspaceStore,
    default_state_dir: pathlib.Path,
) -> list[str]:
    """One warning per user sidecar already inside or beside an output these stages write.

    A stage's own ``*.pvt`` members, as its lock records them, draw no warning. A lock is read only
    for an output that holds a ``*.pvt``, so a project with none pays one walk per directory output.
    """
    replacements = list[Replacement]()
    owners = dict[pathlib.Path, tuple[str, types.ArtifactIdentity, pathlib.Path]]()
    for stage_name, stage_info in stages.items():
        stages_dir = lock.get_stages_dir(stage_info["state_dir"] or default_state_dir)
        for out in stage_info["outs"]:
            path = normalized(workspace.resolve_display_path(out))
            replacements.append(stage_output(path, stage_name))
            owners[path] = (stage_name, out.identity, stages_dir)

    def members_of(output_path: pathlib.Path) -> StageMembers:
        stage_name, identity, stages_dir = owners[output_path]
        lock_data = lock.StageLock(stage_name, stages_dir).read()
        recorded = None if lock_data is None else lock_data["output_hashes"].get(identity)
        return recorded_members(output_path, recorded)

    return [
        _warning(overlap, project_root)
        for overlap in find_output_overlaps(project_root, replacements, members_of)
    ]
