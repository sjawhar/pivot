"""Keep Pivot from deleting tracked and imported files that overlap stage outputs.

A stage output is Pivot's to replace. A tracked file, an imported file and their ``.pvt``
sidecars are the user's. ``track`` and ``import`` refuse a new path inside any pipeline's
declared output (``discovery.declared_outputs``). A project can still hold one from before that
check, so every write, restore or commit that replaces an output first looks for sidecars where
the replacement reaches and fails, naming them, before anything is removed. A forced restore of a
tracked path replaces that path's own content and nothing else: it fails the same way before it
would remove another tracked file, import or sidecar beneath it.

A stage may return a directory member named ``*.pvt``. Such a file is the stage's when the stage
records it (``StageMembers``): its lock, or its ``nocommit:<stage>`` record, lists it at that
path, and it is a sidecar Pivot cannot parse or a file whose content the stage recorded there.
Pivot adds a member to that record before it writes the file (``record_members``), so no member
it writes is ever unrecorded. Every other ``*.pvt`` under an output is the user's, for
``pivot commit`` too.
"""

from __future__ import annotations

import enum
import os
import pathlib
from typing import TYPE_CHECKING, TypedDict

import lmdb

from pivot import exceptions, project, types
from pivot.storage import cache, lock, state, track

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from pivot.registry import RegistryStageInfo
    from pivot.storage.store import WorkspaceStore

# The ``*.pvt`` files a stage itself writes in one output: path -> the content hashes the stage
# recorded there, or None for a file the stage wrote in this run (any content there is its own).
type StageMembers = dict[pathlib.Path, frozenset[str] | None]

# Records ``*.pvt`` files (path -> content hash) as one stage's own before Pivot writes them:
# ``record_members`` bound to that stage.
type MemberRecorder = Callable[[Mapping[pathlib.Path, str]], None]


class ReplacementKind(enum.StrEnum):
    """What a replaced path is: it decides where the check looks and how a refusal reads."""

    STAGE_OUTPUT = "stage_output"
    """A stage output: everything beneath it, and the sidecar for it and for each parent."""
    TRACKED = "tracked"
    """A tracked or git path restored to its own place: only what lies beneath it."""
    DESTINATION = "destination"
    """A path the user named (``pivot get --output``)."""


class Replacement(TypedDict):
    """A path Pivot is about to remove or replace."""

    path: pathlib.Path
    stage: str | None
    """The stage whose output ``path`` is, when the caller knows it."""
    kind: ReplacementKind


class DeclaredOutput(TypedDict):
    """A stage output some pipeline declares: its resolved path and the stage that writes it."""

    resolved: pathlib.Path
    stage: str


class SidecarOverlap(TypedDict):
    """A ``.pvt`` sidecar that replacing ``replacement`` would remove or orphan."""

    tracked_path: pathlib.Path | None
    """The path the sidecar tracks; None when Pivot cannot parse the sidecar."""
    pvt_path: pathlib.Path
    replacement: Replacement


class UnreadableOverlap(TypedDict):
    """A directory beneath ``replacement`` Pivot cannot read, so it cannot prove no tracked file
    hides inside it. A blocking overlap: Pivot refuses rather than delete what it cannot see."""

    unreadable_dir: pathlib.Path
    replacement: Replacement


# A path a replacement must not remove: a user ``.pvt`` sidecar, or a directory Pivot could not
# read and so cannot clear of user files.
type OutputOverlap = SidecarOverlap | UnreadableOverlap


def overlap_path(overlap: OutputOverlap) -> pathlib.Path:
    """The path an overlap names: its ``.pvt`` sidecar, or the directory Pivot could not read."""
    if "unreadable_dir" in overlap:
        return overlap["unreadable_dir"]
    return overlap["pvt_path"]


def stage_output(path: pathlib.Path, stage: str | None = None) -> Replacement:
    return Replacement(path=path, stage=stage, kind=ReplacementKind.STAGE_OUTPUT)


def tracked_restore(path: pathlib.Path) -> Replacement:
    return Replacement(path=path, stage=None, kind=ReplacementKind.TRACKED)


def destination(path: pathlib.Path) -> Replacement:
    return Replacement(path=path, stage=None, kind=ReplacementKind.DESTINATION)


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


def member_hashes(
    output_path: pathlib.Path, output_hash: types.HashInfo | None
) -> dict[pathlib.Path, str]:
    """The ``*.pvt`` files ``output_hash`` puts in ``output_path``, each with its content hash."""
    if output_hash is None or not types.is_dir_hash(output_hash):
        return {}
    root = normalized(output_path)
    return {
        root / entry["relpath"]: entry["hash"]
        for entry in output_hash["manifest"]
        if entry["relpath"].endswith(".pvt")
    }


def record_members(
    project_root: pathlib.Path,
    state_dir: pathlib.Path,
    stage_name: str,
    members: Mapping[pathlib.Path, str],
) -> None:
    """Record ``members`` (path -> content hash), ``*.pvt`` files Pivot is about to write in the
    stage's outputs, as the stage's own, in its ``nocommit:<stage>`` record.

    Every path that writes such a file calls this first, so a crash leaves at worst a record
    without a file, which claims nothing. Opens no StateDB when there is nothing to record.

    Raises:
        exceptions.MemberRecordError: The record could not be written (the write lock timed
            out, the database is full or read-only, or it cannot be opened). The caller must not
            write the members; it fails, naming the stage and the reason.
    """
    if not members:
        return
    root = normalized(project_root)
    relative = {
        normalized(path).relative_to(root).as_posix(): content_hash
        for path, content_hash in members.items()
    }
    try:
        with state.StateDB(state_dir) as state_db:
            state_db.add_no_commit_pvt_members(stage_name, relative)
    except (
        exceptions.PivotDBWriteTimeoutError,
        state.StateDBError,
        lmdb.Error,
        OSError,
        RuntimeError,
    ) as exc:
        raise exceptions.MemberRecordError(
            f"Stage '{stage_name}' cannot record its .pvt output files "
            + f"({', '.join(sorted(relative))}) as its own before writing them: {exc}. "
            + "Pivot wrote none of them; fix the state database and rerun."
        ) from exc


def no_commit_members(
    project_root: pathlib.Path, record: types.NoCommitRecord | None
) -> StageMembers:
    """The ``*.pvt`` members the stage's ``nocommit:`` record lists that still hold what Pivot
    wrote there, each with that content hash.

    A record outlives a write the lock's outputs later replace, so a member counts only while
    its content is the recorded one: a user's file put at that path later stays the user's.
    """
    if record is None or "pvt_members" not in record:
        return {}
    root = normalized(project_root)
    members = dict[pathlib.Path, frozenset[str] | None]()
    for relpath, content_hash in record["pvt_members"].items():
        path = root / relpath
        if path.is_file() and cache.hash_file(path)[0] == content_hash:
            members[path] = frozenset([content_hash])
    return members


def stored_no_commit_members(
    project_root: pathlib.Path, stages: Iterable[tuple[str, pathlib.Path]]
) -> StageMembers:
    """``no_commit_members`` of these ``(stage, state_dir)`` pairs, from each stage's StateDB.

    Opens each state directory's StateDB once, read-only, and never creates one.
    """
    names_by_dir = dict[pathlib.Path, list[str]]()
    for stage_name, state_dir in stages:
        names_by_dir.setdefault(state_dir, []).append(stage_name)
    found = list[StageMembers]()
    for state_dir, names in names_by_dir.items():
        if not (state_dir / "state.lmdb").exists():
            continue
        with state.StateDB(state_dir, readonly=True) as state_db:
            found.extend(
                no_commit_members(project_root, state_db.get_no_commit_record(name))
                for name in names
            )
    return merged_members(*found)


def merged_members(*sources: StageMembers) -> StageMembers:
    """Every member any source lists, with every hash recorded for it (None: any content)."""
    merged = dict[pathlib.Path, frozenset[str] | None]()
    for source in sources:
        for path, recorded in source.items():
            if path not in merged:
                merged[path] = recorded
                continue
            current = merged[path]
            merged[path] = None if current is None or recorded is None else current | recorded
    return merged


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
    pvt_path: pathlib.Path, data: track.PvtData | None, members: StageMembers
) -> bool:
    if pvt_path not in members:
        return False
    recorded = members[pvt_path]
    return recorded is None or data is None or cache.hash_file(pvt_path)[0] in recorded


def find_output_overlaps(
    project_root: pathlib.Path,
    replacements: Sequence[Replacement],
    members_of: Callable[[pathlib.Path], StageMembers] | None = None,
) -> list[OutputOverlap]:
    """Find what replacing each path would remove or orphan: a user ``.pvt`` sidecar, or a
    directory Pivot cannot read.

    Looks only where a replacement reaches. For a stage output or a destination, at the path and
    at each directory above it up to the project root, the sidecar Pivot would write for that
    path counts unless it parses and names another path; a tracked path's own restore leaves its
    own sidecar and those above it as they are. Under a path that is a real directory (removing a
    symlink removes nothing it points to), every ``*.pvt`` file counts unless it is one of the
    path's own members: ``members_of`` maps a replaced path to them and is called only for a path
    that holds a ``*.pvt``.

    A directory under such a path that Pivot cannot read blocks the replacement too (``os.walk``
    reports it through ``onerror``): Pivot cannot prove no tracked file hides inside, so it
    refuses rather than skip it.

    It never walks the rest of the project, so it does not see a hand-edited ``.pvt`` elsewhere
    whose ``path`` names a file inside an output; Pivot never writes one.
    """
    root = normalized(project_root)
    found = dict[pathlib.Path, OutputOverlap]()
    checked = set[pathlib.Path]()
    for replacement in replacements:
        output_path = normalized(replacement["path"])
        current = output_path if replacement["kind"] is not ReplacementKind.TRACKED else root
        while current != root and current.is_relative_to(root) and current not in checked:
            checked.add(current)
            pvt_path = track.get_pvt_path(current)
            if pvt_path.is_file() or pvt_path.is_symlink():
                data = track.read_pvt_file(pvt_path)
                if data is None or pvt_path.parent / data["path"] == current:
                    found[pvt_path] = SidecarOverlap(
                        tracked_path=None if data is None else current,
                        pvt_path=pvt_path,
                        replacement=replacement,
                    )
            current = current.parent

        if not output_path.is_dir() or output_path.is_symlink():
            continue
        members: StageMembers | None = None
        unreadable = list[OSError]()
        for dirpath, _dirnames, filenames in os.walk(output_path, onerror=unreadable.append):
            for name in filenames:
                if not name.endswith(".pvt"):
                    continue
                pvt_path = pathlib.Path(dirpath) / name
                data = track.read_pvt_file(pvt_path)
                if members is None:
                    members = members_of(output_path) if members_of is not None else {}
                if _is_stage_member(pvt_path, data, members):
                    continue
                found[pvt_path] = SidecarOverlap(
                    tracked_path=None if data is None else pvt_path.parent / data["path"],
                    pvt_path=pvt_path,
                    replacement=replacement,
                )
        # A directory os.walk could not enter may hide a tracked file; block it, never skip it.
        for error in unreadable:
            if error.filename is None:
                continue
            blocked = pathlib.Path(error.filename)
            found.setdefault(
                blocked, UnreadableOverlap(unreadable_dir=blocked, replacement=replacement)
            )
    return sorted(found.values(), key=overlap_path)


def _display(path: pathlib.Path, project_root: pathlib.Path) -> str:
    relative = project.relative_within(path, project_root)
    return str(path if relative is None else relative)


def _label(replacement: Replacement, project_root: pathlib.Path) -> str:
    path = _display(replacement["path"], project_root)
    match replacement["kind"]:
        case ReplacementKind.DESTINATION:
            return f"destination '{path}'"
        case ReplacementKind.TRACKED:
            return f"restored path '{path}'"
        case ReplacementKind.STAGE_OUTPUT:
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
    if "unreadable_dir" in overlap:
        directory = _display(overlap["unreadable_dir"], project_root)
        relation = _relation(normalized(overlap["unreadable_dir"]), replaced)
        return (
            f"directory '{directory}', which Pivot cannot read, {relation} {label}; "
            + "Pivot cannot tell whether a tracked file is inside it"
        )
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
) -> None:
    """Fail, naming each one, before replacing these paths would remove a user's sidecar."""
    overlaps = find_output_overlaps(project_root, replacements, members_of)
    if overlaps:
        raise ownership_error(project_root, overlaps)


def ownership_error(
    project_root: pathlib.Path, overlaps: Sequence[OutputOverlap]
) -> exceptions.OutputOwnershipError:
    """The refusal naming each overlap; its first and last lines fit the kinds of path replaced."""
    details = "\n".join(f"  - {_describe(overlap, project_root)}" for overlap in overlaps)
    kinds = {overlap["replacement"]["kind"] for overlap in overlaps}
    if kinds == {ReplacementKind.DESTINATION}:
        header = "Refusing to replace a path that holds user-owned files:"
        remedy = "Choose another --output path, or move each tracked path and its sidecar first."
    elif kinds == {ReplacementKind.TRACKED}:
        header = "Refusing to restore a path over other user-owned files beneath it:"
        remedy = "Move each tracked path and its sidecar out of the restored path, then rerun."
    elif kinds == {ReplacementKind.STAGE_OUTPUT}:
        header = "Refusing to replace a stage output that holds user-owned files:"
        remedy = (
            "Move each tracked path and its sidecar outside every stage output (for example "
            + "under data/raw/), then rerun."
        )
    else:
        header = "Refusing to replace paths that hold user-owned files:"
        remedy = (
            "Move each tracked path and its sidecar outside every stage output and every "
            + "restored path (for example under data/raw/), then rerun."
        )
    return exceptions.OutputOwnershipError(f"{header}\n{details}\n{remedy}")


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
    if "unreadable_dir" in overlap:
        moved = "it"
    else:
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

    A stage's own ``*.pvt`` members, as its lock or its ``nocommit:`` record lists them,
    draw no warning. Both are read only for an output that holds a ``*.pvt``, so a project with
    none pays one walk per directory output.
    """
    replacements = list[Replacement]()
    owners = dict[pathlib.Path, tuple[str, types.ArtifactIdentity, pathlib.Path]]()
    for stage_name, stage_info in stages.items():
        state_dir = stage_info["state_dir"] or default_state_dir
        for out in stage_info["outs"]:
            path = normalized(workspace.resolve_display_path(out))
            replacements.append(stage_output(path, stage_name))
            owners[path] = (stage_name, out.identity, state_dir)

    def members_of(output_path: pathlib.Path) -> StageMembers:
        stage_name, identity, state_dir = owners[output_path]
        lock_data = lock.StageLock(stage_name, lock.get_stages_dir(state_dir)).read()
        recorded = None if lock_data is None else lock_data["output_hashes"].get(identity)
        return merged_members(
            recorded_members(output_path, recorded),
            stored_no_commit_members(project_root, [(stage_name, state_dir)]),
        )

    return [
        _warning(overlap, project_root)
        for overlap in find_output_overlaps(project_root, replacements, members_of)
    ]
