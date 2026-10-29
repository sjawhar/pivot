"""Keep Pivot from deleting tracked and imported files that overlap stage outputs.

A stage output is Pivot's to replace. A tracked file, an imported file and their ``.pvt``
sidecars are the user's. ``track`` and ``import`` refuse a new path inside any pipeline's
declared output (``discovery.declared_outputs``). A project can still hold one from before that
check, so every write, restore or commit that replaces an output first looks for sidecars where
the replacement reaches and fails, naming them, before anything is removed. A forced restore of a
tracked path replaces that path's own content and nothing else: it fails the same way before it
would remove another tracked file, import or sidecar beneath it.

A stage may write a ``*.pvt`` file in a directory output. Such a file is the stage's when the
stage records it at that path (``stage_members``, the one rule every caller uses): its lock, the
hash a restore writes, its ``nocommit:<stage>`` record, or what this run writes lists it there,
and it is a sidecar Pivot cannot parse or a file whose content the stage recorded there. Pivot
adds a member to that record before it writes the file (``record_members``), so no member it
writes is ever unrecorded. A member is no tracked file: discovery leaves it out
(``own_member_skip``). Every other ``*.pvt`` under an output is the user's, for ``pivot commit``
too. ``forced_restore_refusal`` is the one decision behind every forced restore,
and ``swap_refusal`` makes it again at each target's swap, over the entry the swap removes.
"""

from __future__ import annotations

import contextlib
import contextvars
import enum
import functools
import logging
import os
import pathlib
import stat
from typing import TYPE_CHECKING, NamedTuple, NotRequired, TypedDict

import lmdb

from pivot import config, exceptions, project, types
from pivot import state_dir as state_dir_mod
from pivot.storage import cache, listing, lock, state, track

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence

    from pivot.registry import RegistryStageInfo
    from pivot.storage.store import WorkspaceStore

# The ``*.pvt`` files a stage claims in its outputs: path -> the content hashes the stage
# recorded at that path (none for a path this run returns, before it writes it). A ``*.pvt`` at a
# listed path is the stage's when it does not parse as a sidecar, or when its content is a
# recorded one.
type StageMembers = dict[pathlib.Path, frozenset[str]]

# Records ``*.pvt`` files (path -> content hash) as one stage's own before Pivot writes them:
# ``record_members`` bound to that stage.
type MemberRecorder = Callable[[Mapping[pathlib.Path, str]], None]


class StageClaims(TypedDict):
    """What one stage has recorded as its own ``*.pvt`` files, across its outputs.

    Read once per run or command (``stage_claims``, ``read_stage_claims``); ``stage_members``
    turns it into one output's members.
    """

    locked: Mapping[pathlib.Path, types.HashInfo]
    """Each output's ``normalized`` path -> the hash the stage's lock records for it."""
    no_commit: StageMembers
    """The members the stage's ``nocommit:`` record lists that still hold the recorded content."""
    recorder: MemberRecorder | None
    """Records the ``*.pvt`` files Pivot writes for the stage before it writes them; None when
    the caller writes none."""
    cache_dir: pathlib.Path | None
    """The cache object root (``<cache.dir>/files``) the stage's outputs are cached in."""
    locked_members: NotRequired[Callable[[pathlib.Path], listing.DirListing | None]]
    """An output's ``normalized`` path -> the listing of the directory the lock records for it,
    read from the listing file committed beside the lock (byte-checked, else the cache's copy;
    ``stage_claims``), or None when neither can be read. Without it, or with None, the listing is
    read from the cache alone."""


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


class ForcedRestore(TypedDict):
    """A path a restore replaces (``checkout``, ``pull``, ``get`` or a stage write)."""

    replacement: Replacement
    restored: types.HashInfo | None
    """The hash the restore writes; None for a file restored from git or a stage output."""
    restored_members: NotRequired[listing.DirListing]
    """The listing of a restored directory, when the caller has read it (a committed listing
    file); without it the listing is read by ``read_restored_members``, else from the cache at
    the restored hash."""
    read_restored_members: NotRequired[Callable[[], listing.DirListing]]
    """Reads the listing of a restored directory (``restored_members``), called only when the
    path holds a ``*.pvt``; an error reading it leaves the listing to the cache."""
    claims: Callable[[], StageClaims] | None
    """For a stage output restored to its own path: reads the stage's claims, called only when
    the path holds a ``*.pvt``."""
    written: NotRequired[StageMembers]
    """The ``*.pvt`` members a stage is about to write there."""


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


def _cached_listing(
    output_hash: types.DirHash, cache_dir: pathlib.Path | None
) -> listing.DirListing | None:
    """The listing the cache holds for a directory hash (``cache_dir``, else the configured
    cache), or None when it holds no usable copy: the directory's members are then unknown, so
    no ``*.pvt`` in it counts as the stage's and each is refused as the user's."""
    try:
        files_dir = cache_dir if cache_dir is not None else config.locate_cache_dir() / "files"
        return cache.cached_listing(files_dir, output_hash)
    except (exceptions.CorruptListingError, exceptions.ProjectNotInitializedError):
        return None


def member_hashes(
    output_path: pathlib.Path,
    output_hash: types.HashInfo | None,
    *,
    members: listing.DirListing | None = None,
    cache_dir: pathlib.Path | None = None,
) -> dict[pathlib.Path, str]:
    """The ``*.pvt`` files ``output_hash`` puts in ``output_path``, each with its content hash.

    Only a directory hash lists files, whatever the output's tag is. Its files are in its
    listing: ``members`` when the caller has read it, else the cache's copy (``cache_dir``)."""
    if output_hash is None or not types.is_dir_hash(output_hash):
        return {}
    if members is None:
        members = _cached_listing(output_hash, cache_dir)
        if members is None:
            return {}
    root = normalized(output_path)
    return {
        root / entry["relpath"]: entry["hash"]
        for entry in members["files"]
        if entry["relpath"].endswith(".pvt")
    }


def _hash_members(
    output_path: pathlib.Path,
    hashes: Iterable[tuple[types.HashInfo | None, listing.DirListing | None]],
    cache_dir: pathlib.Path | None,
) -> StageMembers:
    """The ``*.pvt`` files ``hashes`` put in ``output_path``, each with every hash recorded; each
    hash comes with its listing when the caller has read it (``member_hashes``)."""
    found = dict[pathlib.Path, set[str]]()
    for output_hash, members in hashes:
        listed = member_hashes(output_path, output_hash, members=members, cache_dir=cache_dir)
        for path, content_hash in listed.items():
            found.setdefault(path, set[str]()).add(content_hash)
    return {path: frozenset(recorded) for path, recorded in found.items()}


def stage_members(
    output_path: pathlib.Path,
    claims: StageClaims | None,
    *,
    restored: types.HashInfo | None = None,
    restored_members: listing.DirListing | None = None,
    returned: StageMembers | None = None,
    moved_to: pathlib.Path | None = None,
) -> StageMembers:
    """The stage's own ``*.pvt`` files in ``output_path``: the one rule every write, restore,
    commit and warning uses.

    A file is listed when the hash the stage's lock records for the output lists it, or the hash
    a restore writes (``restored``), or the stage's ``nocommit:`` record, or what this run writes
    (``returned``). Whether the output is a directory is read from those hashes and from what
    the run writes, never from the output's tag: a ``PathOnly`` output can be a directory too.

    At a path the lock or the restored hash lists, the file's current content is recorded too
    when Pivot holds it in the cache: a lock that changed through git lists the content of
    another run there, and the stage's own file from the run before is cached. Replacing such a
    file loses nothing the cache cannot restore. That content is read under ``moved_to`` when a
    restore has renamed the output there to swap the restored one in.
    """
    found = dict[pathlib.Path, set[str]]()
    root = normalized(output_path)
    locked = claims["locked"].get(root) if claims is not None else None
    cache_dir = claims["cache_dir"] if claims is not None else None
    locked_members = (
        claims["locked_members"](root)
        if claims is not None and locked is not None and "locked_members" in claims
        else None
    )
    hashes = ((restored, restored_members), (locked, locked_members))
    for path, recorded in _hash_members(output_path, hashes, cache_dir).items():
        found[path] = set(recorded)
        located = path if moved_to is None else moved_to / path.relative_to(root)
        if cache_dir is not None and located.is_file():
            content = cache.hash_file(located)[0]
            if content not in recorded and cache.get_cache_path(cache_dir, content).exists():
                found[path].add(content)
    sources = list[StageMembers]()
    if claims is not None:
        sources.append(claims["no_commit"])
    if returned is not None:
        sources.append(returned)
    for source in sources:
        for path, recorded in source.items():
            found.setdefault(path, set[str]()).update(recorded)
    return {path: frozenset(recorded) for path, recorded in found.items()}


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


def record_restored(
    read_claims: Callable[[], StageClaims],
    output_path: pathlib.Path,
    restored: types.HashInfo,
    members: listing.DirListing | None = None,
) -> None:
    """Record the ``*.pvt`` files restoring ``restored`` writes in ``output_path`` as the stage's,
    before the restore writes them, unless the stage's lock records that hash for the output
    (the lock already lists them). ``members`` is a restored directory's listing when the caller
    has read it. Reads the claims only when there is a file to record."""
    members_written = member_hashes(output_path, restored, members=members)
    if not members_written:
        return
    claims = read_claims()
    if claims["recorder"] is None or claims["locked"].get(normalized(output_path)) == restored:
        return
    claims["recorder"](members_written)


def _no_commit_members(
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
    members = dict[pathlib.Path, frozenset[str]]()
    for relpath, content_hash in record["pvt_members"].items():
        path = root / relpath
        if path.is_file() and cache.hash_file(path)[0] == content_hash:
            members[path] = frozenset([content_hash])
    return members


def stage_claims(
    project_root: pathlib.Path,
    locked: Mapping[pathlib.Path, types.HashInfo],
    no_commit_record: types.NoCommitRecord | None,
    *,
    cache_dir: pathlib.Path | None,
    recorder: MemberRecorder | None = None,
    listing_readers: Mapping[pathlib.Path, Callable[[], listing.DirListing]] | None = None,
) -> StageClaims:
    """A stage's claims: ``locked`` maps each output path to the hash the stage's lock records
    for it, and ``no_commit_record`` is the stage's ``nocommit:`` record. ``listing_readers``
    maps a directory output's path to a reader of the listing committed beside the lock
    (``listing.read``); each is called at most once, when an output holds a ``*.pvt``."""
    readers = {normalized(path): reader for path, reader in (listing_readers or {}).items()}

    @functools.cache
    def locked_members(output_path: pathlib.Path) -> listing.DirListing | None:
        reader = readers.get(output_path)
        if reader is None:
            return None
        try:
            return reader()
        except (exceptions.PivotError, OSError):
            # Neither the committed file nor the cache gives the listing: its members are
            # unknown, so every *.pvt there is refused as the user's
            return None

    return StageClaims(
        locked={normalized(path): output_hash for path, output_hash in locked.items()},
        no_commit=_no_commit_members(project_root, no_commit_record),
        recorder=recorder,
        cache_dir=cache_dir,
        locked_members=locked_members,
    )


def read_stage_claims(
    project_root: pathlib.Path,
    stage_name: str,
    state_dir: pathlib.Path,
    locked: Mapping[pathlib.Path, types.HashInfo],
    *,
    cache_dir: pathlib.Path | None,
    recorder: MemberRecorder | None = None,
    listing_readers: Mapping[pathlib.Path, Callable[[], listing.DirListing]] | None = None,
) -> StageClaims:
    """``stage_claims``, reading the stage's ``nocommit:`` record from the StateDB in
    ``state_dir`` (opened read-only, never created; ``state.open_readonly``).

    A record that cannot be read (a damaged or unwritable database) is taken as absent: a
    ``*.pvt`` only it lists is then the user's, which every restore refuses, and the members the
    lock lists are still the stage's. A command never fails for want of the record.
    """
    record: types.NoCommitRecord | None = None
    try:
        with state.open_readonly(state_dir, create=False) as state_db:
            if state_db is not None:
                record = state_db.get_no_commit_record(stage_name)
    except (OSError, lmdb.Error) as error:
        logger.debug("No nocommit: record for %s in %s: %s", stage_name, state_dir, error)
    return stage_claims(
        project_root,
        locked,
        record,
        cache_dir=cache_dir,
        recorder=recorder,
        listing_readers=listing_readers,
    )


def lock_listing_readers(
    stage_lock: lock.StageLock,
    lock_data: types.LockData | None,
    paths: Mapping[types.ArtifactIdentity, pathlib.Path],
    cache_dir: pathlib.Path | None,
) -> dict[pathlib.Path, Callable[[], listing.DirListing]]:
    """For each directory output ``lock_data`` records at one of ``paths``, a reader of its
    listing: the file committed beside the lock (``StageLock.output_listing_path``), byte-checked,
    else the cache's copy (``listing.read``). A lock an older pivot wrote listed each directory
    inline, and ``StageLock.read`` put that listing in the cache, so only the cache is read
    (``lock.committed_listing``)."""
    readers: dict[pathlib.Path, Callable[[], listing.DirListing]] = {}
    if lock_data is None:
        return readers
    for identity, output_hash in lock_data["output_hashes"].items():
        path = paths.get(identity)
        if path is None or not types.is_dir_hash(output_hash):
            continue
        committed = lock.committed_listing(lock_data, stage_lock.output_listing_path(identity))
        readers[path] = functools.partial(listing.read, committed, output_hash, cache_dir)
    return readers


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


def _lstat_mode(path: str | pathlib.Path) -> int | None:
    """``os.lstat(path).st_mode``, None when nothing is at ``path``, or 0 when ``lstat`` fails
    otherwise.

    The walk below asks this first, so a path where nothing is costs one system call. Every
    other answer goes on to the ``pathlib`` predicates the walk has always used, which decide
    (and raise) as before: 0 is no file type, so it always goes on.
    """
    try:
        return os.lstat(path).st_mode
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return 0


def _is_stage_member(
    pvt_path: pathlib.Path, data: track.PvtData | None, members: StageMembers, located: pathlib.Path
) -> bool:
    """Whether the ``*.pvt`` at ``pvt_path``, whose bytes are at ``located``, is a member."""
    if pvt_path not in members:
        return False
    return data is None or cache.hash_file(located)[0] in members[pvt_path]


def _find_sidecars(
    root: str,
    replacement: Replacement,
    output_path: pathlib.Path,
    found: dict[pathlib.Path, OutputOverlap],
    checked: set[str],
) -> None:
    """Add to ``found`` each sidecar that claims ``output_path`` or a directory above it, up to
    the project root ``root``; a tracked path's own restore looks at none. ``checked`` holds the
    paths already looked at."""
    root_prefix = os.path.join(root, "")
    # Normalized absolute strings: is_relative_to is a prefix test, parent is dirname
    current = str(output_path) if replacement["kind"] is not ReplacementKind.TRACKED else root
    while current != root and current.startswith(root_prefix) and current not in checked:
        checked.add(current)
        # The sidecar get_pvt_path names: the path's own name with ".pvt" appended
        if _lstat_mode(current + ".pvt") is not None:
            current_path = pathlib.Path(current)
            pvt_path = track.get_pvt_path(current_path)
            if pvt_path.is_file() or pvt_path.is_symlink():
                data = track.read_pvt_file(pvt_path)
                if data is None or pvt_path.parent / data["path"] == current_path:
                    found[pvt_path] = SidecarOverlap(
                        tracked_path=None if data is None else current_path,
                        pvt_path=pvt_path,
                        replacement=replacement,
                    )
        current = os.path.dirname(current)


def _find_in_tree(
    output_path: pathlib.Path,
    replacement: Replacement,
    members_of: Callable[[pathlib.Path], StageMembers] | None,
    found: dict[pathlib.Path, OutputOverlap],
    moved_to: pathlib.Path | None = None,
) -> None:
    """Add to ``found`` each ``*.pvt`` under ``output_path`` that is not one of its members, and
    each directory there Pivot cannot read.

    ``moved_to`` is where the entry at ``output_path`` is now, when a restore has renamed it
    aside to swap the restored one in: the walk reads it there, and still names each file at its
    place under ``output_path``.
    """
    walked = output_path if moved_to is None else moved_to
    # A file output, or nothing, is the common case: one lstat answers it
    mode = _lstat_mode(walked)
    if mode is None or stat.S_ISREG(mode):
        return
    if not stat.S_ISDIR(mode) and (not walked.is_dir() or walked.is_symlink()):
        return
    members: StageMembers | None = None
    unreadable = list[OSError]()
    for dirpath, _dirnames, filenames in os.walk(walked, onerror=unreadable.append):
        for name in filenames:
            if not name.endswith(".pvt"):
                continue
            located = pathlib.Path(dirpath) / name
            pvt_path = located if moved_to is None else output_path / located.relative_to(moved_to)
            data = track.read_pvt_file(located)
            if members is None:
                members = members_of(output_path) if members_of is not None else {}
            if _is_stage_member(pvt_path, data, members, located):
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
        if moved_to is not None:
            blocked = output_path / blocked.relative_to(moved_to)
        found.setdefault(
            blocked, UnreadableOverlap(unreadable_dir=blocked, replacement=replacement)
        )


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
    root = str(normalized(project_root))
    found = dict[pathlib.Path, OutputOverlap]()
    checked = set[str]()
    for replacement in replacements:
        output_path = normalized(replacement["path"])
        _find_sidecars(root, replacement, output_path, found, checked)
        _find_in_tree(output_path, replacement, members_of, found)
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


def _restored_listing(target: ForcedRestore) -> listing.DirListing | None:
    """The listing of the directory ``target`` restores, when its caller read it or gave a reader;
    None leaves the listing to the cache (``member_hashes``)."""
    members = target.get("restored_members")
    if members is not None or "read_restored_members" not in target:
        return members
    try:
        return target["read_restored_members"]()
    except (exceptions.PivotError, OSError):
        return None


def _restore_members(
    target: ForcedRestore, output_path: pathlib.Path, *, moved_to: pathlib.Path | None = None
) -> StageMembers:
    """The ``*.pvt`` files under a restored path that its restore may replace; ``moved_to`` is
    where the entry at the path is now (``stage_members``)."""
    match target["replacement"]["kind"]:
        case ReplacementKind.STAGE_OUTPUT:
            claims = target["claims"]() if target["claims"] is not None else None
            return stage_members(
                output_path,
                claims,
                restored=target["restored"],
                restored_members=_restored_listing(target),
                returned=target.get("written"),
                moved_to=moved_to,
            )
        case ReplacementKind.TRACKED:
            # A tracked path's own content: what the restore writes and what its sidecar lists now
            current = track.read_pvt_file(track.get_pvt_path(output_path))
            return _hash_members(
                output_path,
                (
                    (target["restored"], _restored_listing(target)),
                    (track.pvt_to_hash_info(current) if current is not None else None, None),
                ),
                None,
            )
        case ReplacementKind.DESTINATION:
            return {}


def _restore_overlaps(
    project_root: pathlib.Path, target: ForcedRestore, entry: pathlib.Path
) -> list[OutputOverlap]:
    """Find user-owned files the restore of ``target`` removes from ``entry``.

    ``entry`` is either the target path before its replacement or the old directory renamed aside
    for its swap. In the latter case, ``moved_to`` makes each reported path read as where it was
    beneath the target, not the temporary backup name.
    """
    if not (entry.exists() or entry.is_symlink()):
        return []
    replacement = target["replacement"]
    output_path = normalized(replacement["path"])
    location = normalized(entry)
    moved_to = None if location == output_path else location
    found = dict[pathlib.Path, OutputOverlap]()
    _find_sidecars(str(normalized(project_root)), replacement, output_path, found, set[str]())
    _find_in_tree(
        output_path,
        replacement,
        functools.partial(_restore_members, target, moved_to=moved_to),
        found,
        moved_to,
    )
    return sorted(found.values(), key=overlap_path)


def forced_restore_refusal(
    project_root: pathlib.Path, targets: Iterable[ForcedRestore]
) -> exceptions.OutputOwnershipError | None:
    """The refusal naming every target whose restore would remove a user's file, with each file
    that blocks it, or None: the one decision behind ``checkout``, ``pull`` and ``get``.

    A command checks every target with it before it replaces any, and checks each target again,
    under that target's restore lock, just before it builds its replacement: a user file that
    appears in the meantime refuses that target alone, before any work is spent on it. The swap
    itself checks once more (``swap_refusal``). Only a path that exists (or is a symlink) is
    replaced, so only such a path is checked. Each target is checked on its own, so the refusal
    names every refused one. A stage output is checked with ``stage_members`` of its stage's
    claims and the restored hash; a tracked path as its own restore, whose members are its own
    content (the restored hash's and its current sidecar's); a destination with none.
    """
    overlaps = dict[tuple[pathlib.Path, pathlib.Path], OutputOverlap]()
    for target in targets:
        path = target["replacement"]["path"]
        for overlap in _restore_overlaps(project_root, target, path):
            overlaps.setdefault((overlap_path(overlap), normalized(path)), overlap)
    if not overlaps:
        return None
    return ownership_error(project_root, [overlaps[key] for key in sorted(overlaps)])


def swap_refusal(
    project_root: pathlib.Path, target: ForcedRestore, entry: pathlib.Path
) -> exceptions.OutputOwnershipError | None:
    """The refusal for ``target`` at its swap, or None: ``forced_restore_refusal``'s check of the
    entry the swap removes, made once the replacement is built, just before it takes the path.

    ``entry`` is where that entry is now. A restore that puts a directory in place, or removes
    one, first renames the entry at the path aside, beside it, and passes where: once renamed, a
    writer that names its files by path can no longer add one to it, so this walk sees every
    file the swap removes. Each file is still named at its place under the target's path. A file
    that replaces a file or symlink is checked in place, ``entry`` being the path itself.
    """
    overlaps = _restore_overlaps(project_root, target, entry)
    return None if not overlaps else ownership_error(project_root, overlaps)


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


# An output's stage, the output's identity, and the state directory the stage's lock is in
type _Owner = tuple[str, types.ArtifactIdentity, pathlib.Path]


def _members_reader(
    project_root: pathlib.Path, owners: Mapping[pathlib.Path, _Owner], cache_dir: pathlib.Path
) -> Callable[[pathlib.Path], StageMembers]:
    """A reader of an output's own ``*.pvt`` members: ``stage_members`` of the claims its
    stage's lock, the listing committed beside it and the stage's ``nocommit:`` record make, with
    ``cache_dir`` the cache object root. Each output's claims are read once, when asked."""

    @functools.cache
    def members_of(output_path: pathlib.Path) -> StageMembers:
        stage_name, identity, state_dir = owners[output_path]
        stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(state_dir))
        lock_data = stage_lock.read(cache_dir=cache_dir)
        locked = (
            {output_path: lock_data["output_hashes"][identity]}
            if lock_data is not None and identity in lock_data["output_hashes"]
            else {}
        )
        claims = read_stage_claims(
            project_root,
            stage_name,
            state_dir,
            locked,
            cache_dir=cache_dir,
            listing_readers=lock_listing_readers(
                stage_lock, lock_data, {identity: output_path}, cache_dir
            ),
        )
        return stage_members(output_path, claims)

    return members_of


def _registry_owners(
    stages: Mapping[str, RegistryStageInfo],
    placed: Callable[[types.ArtifactRef], pathlib.Path],
    default_state_dir: pathlib.Path,
) -> dict[pathlib.Path, _Owner]:
    """Each output these stages write (``normalized``, at the path ``placed`` gives) -> its
    owner."""
    return {
        normalized(placed(out)): (
            stage_name,
            out.identity,
            stage_info["state_dir"] or default_state_dir,
        )
        for stage_name, stage_info in stages.items()
        for out in stage_info["outs"]
    }


class RecordedOutput(NamedTuple):
    """A directory output a stage lock records at a path (``recorded_directory_outputs``)."""

    stage_name: str
    identity: types.ArtifactIdentity
    lock_files: tuple[pathlib.Path, ...]
    """Every lock file of the stage found in the state directory (one per directory of the
    0.5.2 layout that holds one)."""


type _IndexKey = tuple[pathlib.Path, pathlib.Path, pathlib.Path, frozenset[str]]

# The indexes the member skips of one command build, shared between them (``index_per_command``)
_command_indexes: contextvars.ContextVar[
    dict[_IndexKey, dict[pathlib.Path, RecordedOutput]] | None
] = contextvars.ContextVar("pivot_recorded_output_indexes", default=None)


@contextlib.contextmanager
def index_per_command() -> Iterator[None]:
    """Share the index of recorded directory outputs (``recorded_directory_outputs``) between
    every member skip built in this block: it is built once, by the first that needs it. The CLI
    holds one for each command. A skip built outside one builds its own index."""
    token = _command_indexes.set({})
    try:
        yield
    finally:
        _command_indexes.reset(token)


def recorded_directory_outputs(
    project_root: pathlib.Path,
    default_state_dir: pathlib.Path,
    cache_dir: pathlib.Path,
    declared: Collection[str],
) -> dict[pathlib.Path, RecordedOutput]:
    """Every directory output a stage lock in ``default_state_dir`` records at a path (the
    lock's ``output_paths``), by its ``normalized`` path, for each stage not in ``declared``.

    The locks are those of both directories of the 0.5.2 layout. A stage with a lock in one of
    them has that file read (``StageLock.read_sole_file``); one with a lock in each has the
    current one read (``StageLock.read``). The path is the one the run recorded, never
    re-derived, so any placement (a ``PathOnly`` directory at ``<stage>.<ext>``, a keyed or
    variant output) is found. A directory output whose lock records no path (an older pivot's)
    is left out. Only what runs left behind is read, never a stage definition. When every stage
    is declared (``--all``, ``track``), only the locks of stages no pipeline declares are read.
    Built once per command (``index_per_command``).
    """
    key: _IndexKey = (normalized(project_root), default_state_dir, cache_dir, frozenset(declared))
    shared = _command_indexes.get()
    if shared is not None and key in shared:
        return shared[key]
    root, _, _, excluded = key
    root_str = os.fspath(root)
    stages_dir = lock.get_stages_dir(default_state_dir)
    other_dir = state_dir_mod.other_stages_dir_for(stages_dir)
    scanned = [stages_dir] if other_dir is None else [stages_dir, other_dir]
    found = [lock.stage_lock_names(directory) - excluded for directory in scanned]
    index = dict[pathlib.Path, RecordedOutput]()
    for stage_name in set[str]().union(*found):
        try:
            locks = [
                lock.StageLock(stage_name, directory)
                for directory, names in zip(scanned, found, strict=True)
                if stage_name in names
            ]
        except ValueError:  # a file named like a lock that no stage name can have
            continue
        lock_data = (
            locks[0].read_sole_file(cache_dir=cache_dir)
            if len(locks) == 1
            else locks[0].read(cache_dir=cache_dir)
        )
        if lock_data is None or "output_paths" not in lock_data:
            continue
        lock_files = tuple(stage_lock.path for stage_lock in locks)
        for identity, output_hash in lock_data["output_hashes"].items():
            if not types.is_dir_hash(output_hash) or identity not in lock_data["output_paths"]:
                continue
            path = os.path.normpath(os.path.join(root_str, lock_data["output_paths"][identity]))
            if path.startswith(root_str + os.sep):
                index[pathlib.Path(path)] = RecordedOutput(stage_name, identity, lock_files)
    if shared is not None:
        shared[key] = index
    return index


def _recorded_members(
    project_root: pathlib.Path,
    default_state_dir: pathlib.Path,
    cache_dir: pathlib.Path,
    declared: Collection[str],
) -> tuple[dict[pathlib.Path, RecordedOutput], Callable[[pathlib.Path], StageMembers]]:
    """The index of recorded directory outputs (``recorded_directory_outputs``) and a reader of
    each indexed output's own ``*.pvt`` members (``_members_reader``)."""
    index = recorded_directory_outputs(project_root, default_state_dir, cache_dir, declared)
    owners = {
        output: (recorded.stage_name, recorded.identity, default_state_dir)
        for output, recorded in index.items()
    }
    return index, _members_reader(project_root, owners, cache_dir)


def orphaned_member_claim(
    project_root: pathlib.Path,
    default_state_dir: pathlib.Path,
    cache_dir: pathlib.Path,
    declared: Collection[str],
) -> Callable[[pathlib.Path], str | None]:
    """A reader that says, for a path ``pivot track`` is asked to track, which lock of a stage
    no pipeline ``declared`` lists that path's sidecar (``<path>.pvt``) among the members of its
    output, or returns None.

    Every command skips such a stage's members by its lock (``own_member_skip``). A sidecar
    ``track`` wrote at a listed path could carry the bytes the lock records, and would then be
    one no command reads, so the listed path decides, whatever is there now. Any other path in
    that output tracks as before. The index is built at the first path asked about, and a
    stage's claims are read only for an output that holds the sidecar's path.
    """

    recorded = functools.cache(
        functools.partial(_recorded_members, project_root, default_state_dir, cache_dir, declared)
    )

    def claim(path: pathlib.Path) -> str | None:
        index, members_of = recorded()
        sidecar = normalized(track.get_pvt_path(path))
        for output in sidecar.parents:
            if output not in index or sidecar not in members_of(output):
                continue
            locks = " and ".join(
                f"'{_display(found, project_root)}'" for found in index[output].lock_files
            )
            return (
                f"the lock {locks} records '{_display(sidecar, project_root)}' as a file of stage "
                + f"'{index[output].stage_name}', a stage that no longer exists in any pipeline"
            )
        return None

    return claim


def saved_input_warnings(
    project_root: pathlib.Path,
    stages: Mapping[str, RegistryStageInfo],
    workspace: WorkspaceStore,
    default_state_dir: pathlib.Path,
    cache_dir: pathlib.Path,
) -> list[str]:
    """One warning per user sidecar already inside or beside an output these stages write.

    A stage's own ``*.pvt`` members (``stage_members``, with ``cache_dir`` the cache object
    root) draw no warning. The stage's claims are read only for an output that holds a
    ``*.pvt``, so a project with none pays one walk per directory output.
    """
    owners = _registry_owners(stages, workspace.resolve_display_path, default_state_dir)
    members_of = _members_reader(project_root, owners, cache_dir)
    replacements = [stage_output(path, owner[0]) for path, owner in owners.items()]
    return [
        _warning(overlap, project_root)
        for overlap in find_output_overlaps(project_root, replacements, members_of)
    ]


def own_member_skip(
    project_root: pathlib.Path,
    stages: Mapping[str, RegistryStageInfo],
    default_state_dir: pathlib.Path,
    cache_dir: pathlib.Path,
) -> track.PvtSkip:
    """Whether a ``*.pvt`` file that parses is a stage's own member of its output
    (``stage_members``, the rule every write, restore and commit uses): ``skip`` for
    ``track.discover_pvt_files``, so no command reads a stage's own member as a tracked file.

    A ``*.pvt`` in an output of these stages (the command's) is decided by the registry. Any
    other is decided from recorded state alone: it is a member when a directory output that a
    stage lock records holds it (``recorded_directory_outputs``, built at the first such
    ``*.pvt`` and never for a command that meets none), and the listing committed beside that
    lock (byte-checked, else the cache's copy) or that stage's ``nocommit:`` record lists it
    with its current content. Whether a stage run wrote this file is execution state, not a
    stage definition, so no other pipeline is imported; a pipeline that fails to load, or a
    stage that no longer exists, is decided the same way.

    Any other ``*.pvt`` in an output is the user's: discovery keeps it, and every restore of that
    output refuses it. A stage's claims are read only for an output that holds a ``*.pvt``.
    """
    from pivot.storage import store as store_mod  # store imports this module

    owners = _registry_owners(
        stages, lambda out: project_root / store_mod.relative_output_path(out), default_state_dir
    )
    members_of = _members_reader(project_root, owners, cache_dir)

    recorded = functools.cache(
        functools.partial(_recorded_members, project_root, default_state_dir, cache_dir, stages)
    )

    def skip(pvt_path: pathlib.Path, data: track.PvtData) -> bool:
        path = normalized(pvt_path)
        outputs = [output for output in path.parents if output in owners]
        if outputs:
            return any(_is_stage_member(path, data, members_of(output), path) for output in outputs)
        recorded_owners, recorded_members_of = recorded()
        return any(
            _is_stage_member(path, data, recorded_members_of(output), path)
            for output in path.parents
            if output in recorded_owners
        )

    return skip
