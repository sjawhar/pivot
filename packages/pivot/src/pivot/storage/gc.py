"""Collect the cache objects no live record names.

Liveness is read from disk, never inferred. Six classes of record name an object: a stage
lock (``outs``, ``deps`` and the objects its run read by hash, ``reads``), a ``.pvt`` file, a
run-cache row (its outputs and its ``reads``), a ``nocommit:`` record's ``reads`` (which
``pivot commit`` writes into the lock), a workspace link into the cache, and a ref under
``<cache>/files/refs``. Each contributes a file's hash, or, for a directory, the hash of its
listing — which is itself the cache object at that hash, and the name of its whole-directory
tree under ``<cache>/trees`` — and every file hash that listing names (``storage/listing.py``).
A directory whose listing this collection cannot read refuses it: the members are unknown, and
a collection that cannot read a record never licences a delete.

A cache outside the project root can hold another project's outputs, whose locks this project
cannot read. Pivot therefore keeps a register of the projects that use a cache, at
``<cache>/projects/``, written by every naming unit (``storage/naming.py``: once per ``repro``
or ``run`` before any stage, and through ``config.register_cache_use()`` before ``checkout``
and ``pull`` link objects), and refuses to collect an external cache that records nothing.

Two rules cover the window between publishing an object and writing the record that names it
(``storage/AGENTS.md``): every object whose ``st_ctime`` is inside the grace period is kept,
and so is every object some directory entry outside the cache still hardlinks.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import json
import logging
import os
import pathlib
import socket
import stat
import time
from typing import TYPE_CHECKING, Literal, NamedTuple, TypedDict, cast

import lmdb
import yaml

from pivot import exceptions, project, run_history, yaml_config
from pivot import state_dir as state_dir_mod
from pivot.storage import cache, listing, lock, member_skip, naming, state, track
from pivot.types import is_dir_hash

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable

    from pivot.types import DirHash

logger = logging.getLogger(__name__)

# Cache objects live under ``<cache>/files`` (``naming.OBJECTS_DIR``); the register is beside the
# object store (``naming.store_layout``), so no cache scan sees it (``remote/sync.py`` descends
# two-character prefix directories only).
# Whole-directory trees a symlink checkout links to, one per directory hash
# (``cache.tree_path``), under each cache root. A tree's bytes are copies of its members, so it is
# collected on its own; the record that names the directory keeps both it and the listing object.
TREES_DIR = "trees"
# Held exclusively for a whole collection, beside the register, so two collections never run on
# one object store.
GC_LOCK = "gc.lock"
# Every marker file beside a whole-directory tree object, removed with the tree.
_TREE_MARKER_SUFFIXES = cache.TREE_MARKER_SUFFIXES
DEFAULT_GRACE_HOURS = 24.0


class ProjectScope(TypedDict):
    """One project whose records keep objects in this cache alive."""

    root: pathlib.Path
    state_dir: pathlib.Path
    host: str
    last_write: str


class _RegisteredProject(TypedDict):
    """A register entry, with the evidence a collection checks before it reads the project."""

    scope: ProjectScope
    project_id: str | None
    machine: str
    mount_ns: int | None
    mounts: list[str] | None
    # The state directory the writer recorded, as its mounts are spelled in its view.
    recorded_state_dir: str


class RootCounts(TypedDict):
    """How many records of each class were read."""

    locks: int
    pvt_files: int
    rows: int
    links: int


class Roots(TypedDict):
    """Every cache key the records name, by the class of record that named it first."""

    named: set[str]  # stage locks and .pvt files
    run_cache: set[str]
    linked: set[str]  # workspace links and cache refs
    counts: RootCounts
    # Directories inside a scope that are symlinks leaving the project. The collection does not
    # follow one (BET-301 containment) and does not read its records, so it keeps every object
    # and removes none, naming the links so the user can fix the layout.
    outward_dirs: list[str]


class ObjectCount(TypedDict):
    """A number of cache objects and the bytes they hold."""

    objects: int
    bytes: int


class KeptBy(TypedDict):
    """Why each kept object was kept, counted once per object in this order."""

    locks: int
    run_cache: int
    links: int
    grace: int
    hardlinked: int


class Kept(TypedDict):
    """The objects a collection keeps."""

    objects: int
    bytes: int
    by: KeptBy


class Doomed(TypedDict):
    """One object a collection would remove."""

    path: pathlib.Path
    is_dir: bool
    bytes: int
    linked: bool
    # (st_dev, st_ino) of the prefix directory the plan listed it in: the removal happens in
    # that directory, opened from the store the collection holds, or not at all.
    prefix: tuple[int, int]


class Plan(TypedDict):
    """What a collection found and what it would remove."""

    scanned: ObjectCount
    kept: Kept
    doomed: list[Doomed]
    doomed_bytes: int
    linked: ObjectCount


class GcResult(TypedDict):
    """The document ``pivot gc --json`` writes."""

    cache: str
    scanned: ObjectCount
    kept: Kept
    removed: ObjectCount
    projects: list[str]
    dry_run: bool


# What the user can do about each kind of refusal over a project gc cannot read. Each refusal
# names its own remedy: deleting a register entry is right only for a project that is gone.
REMEDY_REGISTER = (
    "That project may hold outputs this run would remove. Restore access to it, or, for a "
    + "project that is gone\nfor good, delete the register entry named above."
)
REMEDY_ACCESS = (
    "That project may hold outputs this run would remove. Restore access to it, then collect "
    + "again."
)
REMEDY_CONFIG = "Fix that project's configuration, then collect again."
REMEDY_RECORD = (
    "A record pivot cannot fully read may name any object in this cache. Fix or resolve that "
    + "record\n(finish the merge that left it, or restore it from git), then collect again; or "
    + "collect with the\npivot that wrote it."
)
REMEDY_LISTING = (
    "That directory's files are unknown until its listing is back. Restore the listing from "
    + "git, or\ncheck the directory out or track it again, then collect again."
)


class ProjectUnreadableError(exceptions.CacheError):
    """A project in scope cannot be read, so its records are unknown. ``remedy`` is what the
    user can do about this kind of refusal."""

    scope: ProjectScope
    reason: str
    remedy: str

    def __init__(self, scope: ProjectScope, reason: str, remedy: str = REMEDY_ACCESS) -> None:
        super().__init__(f"{scope['root']}: {reason}")
        self.scope = scope
        self.reason = reason
        self.remedy = remedy


# -----------------------------------------------------------------------------
# The register of the projects that use a cache
# -----------------------------------------------------------------------------


def _project_from_entry(path: pathlib.Path) -> _RegisteredProject:
    """Read one register entry. Raises OSError or ValueError when it cannot be used."""
    data: object = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("not a JSON object")
    entry = cast("dict[str, object]", data)
    root = entry.get("root")
    if not isinstance(root, str) or not root:
        raise ValueError("no project root recorded")
    state_dir = entry.get("state_dir")
    host = entry.get("host")
    last_write = entry.get("last_write")
    project_id = entry.get("project_id")
    mounts = entry.get("mounts")
    machine = entry.get("machine")
    mount_ns = entry.get("mount_ns")
    scope = ProjectScope(
        root=pathlib.Path(root),
        # The collector resolves the project's current state directory before it reads records.
        state_dir=pathlib.Path(state_dir)
        if isinstance(state_dir, str)
        else pathlib.Path(root) / ".pivot",
        host=host if isinstance(host, str) else "unknown host",
        last_write=last_write if isinstance(last_write, str) else "an unknown time",
    )
    if mounts is not None and (
        not isinstance(mounts, list)
        or not all(isinstance(mount, str) for mount in cast("list[object]", mounts))
    ):
        raise ValueError("its mount list is not a list of paths")
    return _RegisteredProject(
        scope=scope,
        project_id=project_id
        if isinstance(project_id, str) and naming.ENTRY_ID_RE.fullmatch(project_id)
        else None,
        machine=machine if isinstance(machine, str) else "",
        mount_ns=mount_ns if isinstance(mount_ns, int) and not isinstance(mount_ns, bool) else None,
        mounts=cast("list[str] | None", mounts),
        recorded_state_dir=state_dir if isinstance(state_dir, str) else str(scope["state_dir"]),
    )


def _recorded_where(scope: ProjectScope) -> str:
    """Which host wrote a register entry, as a refusal names it."""
    if scope["host"] != socket.gethostname():
        return f"recorded by host {scope['host']}"
    return "recorded by this host"


def _registered_scope_reason(
    entry: pathlib.Path, scope: ProjectScope, reason: str
) -> tuple[str, str]:
    """Describe a registered project the collector cannot prove it read."""
    where = _recorded_where(scope)
    return (
        scope_label(scope),
        f"{where}, and its project root or state directory is not visible here ({reason}; it may "
        + "be deleted, on a disk that is not mounted, in another mount namespace, or on another "
        + f"host). Make it visible and re-run; if the project is gone for good, delete {entry} "
        + "and re-run",
    )


def _read_scope_directories(scope: ProjectScope) -> None:
    """Open each directory that establishes this scope's records are actually readable."""
    directories = (
        scope["root"],
        scope["root"] / ".pivot",
        scope["state_dir"],
    )
    seen = set[pathlib.Path]()
    for directory in directories:
        if directory in seen:
            continue
        seen.add(directory)
        with os.scandir(directory):
            pass


def _identity_reason(
    entry: pathlib.Path, registered: _RegisteredProject, points: frozenset[str] | None
) -> tuple[str, str] | None:
    """Why a visible registered project is not the one its entry describes, or None."""
    scope = registered["scope"]
    if registered["project_id"] is None:
        return (
            scope_label(scope),
            "its entry records no project id, because its state directory could not be written "
            + "when it registered. Make that state directory writable and run `pivot checkout` "
            + f"there, then delete {entry} and re-run; if the project is gone for good, delete "
            + f"{entry} and re-run",
        )
    try:
        project_id = naming.read_project_id(scope["state_dir"])
    except (OSError, ValueError) as e:
        return _registered_scope_reason(
            entry, scope, f"no project id readable in {scope['state_dir']}: {_error_text(e)}"
        )
    if project_id != registered["project_id"]:
        return (
            scope_label(scope),
            f"{_recorded_where(scope)}, and {scope['state_dir']} holds a different project id. "
            + "This path holds a different project here, or the project's state database was "
            + "reset. Make the registered project visible at this path and re-run; if that "
            + f"project is gone for good or its state database was reset, delete {entry} and "
            + "re-run",
        )
    if (nonce_reason := _view_nonce_reason(entry, registered)) is not None:
        return nonce_reason
    if (caches_reason := _other_caches_reason(entry, registered, project_id)) is not None:
        return caches_reason
    host, machine, mount_ns = naming.this_view()
    if (scope["host"], registered["machine"], registered["mount_ns"]) != (host, machine, mount_ns):
        return _other_view_reason(entry, registered)
    mounts = registered["mounts"]
    if mounts is None or points is None:
        return (
            scope_label(scope),
            "its mount points cannot be checked on this host. Run `pivot checkout` there to "
            + f"register it again; if the project is gone for good, delete {entry} and re-run",
        )
    missing = [mount for mount in mounts if mount not in points]
    if missing:
        return (
            scope_label(scope),
            f"recorded mount point {', '.join(missing)} is not mounted (entry {entry}). Mount it "
            + "and re-run. If you removed that mount on purpose, run `pivot checkout` in the "
            + "project; its naming step records the mounts it has now",
        )
    return None


def _view_nonce_reason(
    entry: pathlib.Path, registered: _RegisteredProject
) -> tuple[str, str] | None:
    """Why this view cannot prove it reads the state the entry's view last wrote, or None.

    An inode number is not proof: containers from one image (overlayfs) and VMs cloned from one
    disk show different state directories with the same inode. The entry's view wrote a fresh
    nonce into its state directory and into the entry's nonce file just before the entry, and
    rewrites both at every naming step, in this cache and in every other cache the project
    lists (``naming.naming_unit``, ``naming._confirm_listed_caches``). Reading the same nonce in
    both here proves this view reads the bytes that view last wrote (one directory, as over
    NFS), not a clone or an overlay copy: a copy keeps the nonce of the moment it was taken,
    and the project's next naming step in any cache it reaches replaces this cache's. The pair
    is read under the shared lock writers take across both writes (``naming.read_nonce_pair``),
    since a step in another cache rewrites this cache's pair without this cache's naming lock.
    """
    scope = registered["scope"]
    view_key = naming.view_key_of(scope["host"], registered["machine"], registered["mount_ns"])
    try:
        # The entry's cache is the one whose register holds it (`<register root>/projects/`)
        cache_id = naming.read_cache_id_at(entry.parent.parent)
        state_nonce, entry_nonce = naming.read_nonce_pair(
            scope["state_dir"], view_key, cache_id, naming.entry_nonce_path(entry)
        )
    except (OSError, ValueError) as e:
        detail = f"no nonce pair for that view readable here: {_error_text(e)}"
    else:
        if state_nonce == entry_nonce:
            return None
        detail = "this state directory holds an older or different nonce for that view"
    return (
        scope_label(scope),
        f"{_recorded_where(scope)}, and this host cannot prove it reads the state that view last "
        + f"wrote in {scope['state_dir']} ({detail}). This path may hold a copy of that project "
        + "(a container's overlay, a cloned disk) rather than the project itself. Run `pivot gc` "
        + "where that view's state directory is visible; if that view is gone for good, delete "
        + f"{entry} and re-run",
    )


def _other_caches_reason(
    entry: pathlib.Path, registered: _RegisteredProject, project_id: str
) -> tuple[str, str] | None:
    """Why this view cannot prove that every other cache the project lists holds the entry this
    state directory last wrote there, or None.

    The entry's own nonce proves this state directory is the one its view last wrote for this
    cache, and every naming step rewrites that pair in every cache the project lists
    (``naming._confirm_listed_caches``). Each other listed cache is checked too: a copy of the
    project (a cloned disk) that holds this cache's root with its object store elsewhere (its
    ``files/`` a link to shared storage) has its own copy of this register, which the original's
    steps never reach, and only the original's entries in a cache both share tell the copy
    apart. So every cache any view of the project lists (``naming.cache_records``) must be
    readable where that view recorded it, hold the same cache, and hold that view's entry for
    the project with the nonce this state directory holds for it. A project that uses one cache
    lists only it, and nothing more is read.
    """
    scope = registered["scope"]
    try:
        this_cache = naming.read_cache_id_at(entry.parent.parent)
        records = naming.cache_records(scope["state_dir"])
    except (OSError, ValueError) as e:
        return (
            scope_label(scope),
            f"{_recorded_where(scope)}, and the list of caches it uses in {scope['state_dir']} "
            + f"cannot be read ({_error_text(e)}). Make it readable and re-run",
        )
    here: str | None = None
    for record in records:
        if record.cache_id == this_cache:
            continue
        problem = naming.listed_cache_problem(record, scope["root"], scope["state_dir"], project_id)
        if problem is None:
            continue
        if here is None:
            here = naming.view_key_of(*naming.this_view())
        where = record.cache_dir if record.cache_dir is not None else record.record
        found = (
            f"{_recorded_where(scope)}, and it also uses the cache {where} (id "
            + f"{record.cache_id}), where {problem.detail}."
        )
        if problem.state is naming.ListedCacheState.STALE:
            remedy = (
                " This path may hold a copy of the project (a cloned disk) that the project has "
                + "moved on from: every naming step of the project brings that entry up to date. "
                + "Run `pivot gc` where the project itself is visible"
            )
        elif record.view_key != here and problem.state is naming.ListedCacheState.NO_ENTRY:
            remedy = (
                " Another view of the project (another host or mount namespace) listed that "
                + f"cache in {record.record}. Run any pivot command of the project in that view "
                + "that reaches the cache: it registers that view there, or, where that view "
                + "cannot write the cache (a read-only mount), removes that record. If that view "
                + f"is gone for good, delete {record.record} and re-run"
            )
        elif record.view_key != here:
            remedy = (
                " Another view of the project (another host or mount namespace) listed that "
                + f"cache in {record.record}. Run `pivot gc` where that view reaches the cache, "
                + f"or, if that view no longer uses it, `pivot forget-cache {record.cache_id}` in "
                + "the project in that view; if that view is gone for good, delete "
                + f"{record.record} and re-run"
            )
        elif problem.state is naming.ListedCacheState.NO_ENTRY:
            remedy = (
                " Run any pivot command in the project that reaches that cache: it registers the "
                + "project there again, or, where this view cannot write the cache (a read-only "
                + "mount), removes the record. Then re-run"
            )
        else:
            remedy = (
                " If the project reaches that cache at another path, run any pivot command in it "
                + "there. If no view of the project uses that cache any more, run `pivot "
                + f"forget-cache {record.cache_id}` in the project itself, never in a copy of it. "
                + "Then re-run"
            )
        return scope_label(scope), found + remedy
    return None


def _other_view_reason(
    entry: pathlib.Path, registered: _RegisteredProject
) -> tuple[str, str] | None:
    """Why an entry from another host or mount namespace cannot be trusted here, or None.

    This view cannot see that writer's mounts. A mount at the project root or the state
    directory is the project's own filesystem, which the project id check above confirmed this
    view reads. Any other recorded mount (a disk mounted under the project in that view) may
    hold records this view does not have, so the collection refuses.
    """
    scope = registered["scope"]
    mounts = registered["mounts"]
    own = {str(scope["root"]), registered["recorded_state_dir"]}
    hidden = None if mounts is None else [mount for mount in mounts if mount not in own]
    if hidden == []:
        return None
    what = (
        "its mount points are unknown"
        if hidden is None
        else f"it records mount point {', '.join(hidden)}"
    )
    return (
        scope_label(scope),
        f"{_recorded_where(scope)} in another mount view, and {what}, which this host cannot "
        + "check: records there may not be visible here. Run `pivot gc` where that mount is "
        + f"visible; if that view of the project is gone for good, delete {entry} and re-run",
    )


def _cache_mount_reasons(
    snapshot: StoreSnapshot, points: frozenset[str] | None
) -> list[tuple[str, str]]:
    """Why the cache's register or refs may be hidden by a mount this view cannot confirm.

    Each view's record is checked: this view's recorded mounts must be mounted now, and another
    view's record may name no mount under the cache root, since this view cannot check it.
    """
    directory = snapshot.layout.register_root / naming.CACHE_MOUNTS_DIR
    try:
        with os.scandir(directory) as entries:
            paths = sorted(pathlib.Path(e.path) for e in entries if e.name.endswith(".json"))
    except FileNotFoundError:
        return []
    except OSError as e:
        return [(str(directory), _error_text(e))]
    reasons = list[tuple[str, str]]()
    for path in paths:
        recorded = naming.entry_from_file(path)
        mounts = None if recorded is None else recorded.get("mounts")
        if recorded is None or (
            mounts is not None
            and (
                not isinstance(mounts, list)
                or not all(isinstance(m, str) for m in cast("list[object]", mounts))
            )
        ):
            reasons.append((str(path), "the cache's mount record cannot be read"))
            continue
        recorded_mounts = cast("list[str] | None", mounts)
        same_view = (
            recorded.get("host"),
            recorded.get("machine"),
            recorded.get("mount_ns"),
        ) == naming.this_view()
        if not same_view:
            if recorded_mounts != []:
                what = "unknown" if recorded_mounts is None else ", ".join(recorded_mounts)
                reasons.append(
                    (
                        str(path),
                        f"another host or mount view of this cache records mount points {what} "
                        + "under the cache root, which this host cannot check. Run `pivot gc` "
                        + f"there; if that view is gone for good, delete {path} and re-run",
                    )
                )
            continue
        if recorded_mounts is None or points is None:
            reasons.append((str(path), "the cache's mount points cannot be checked on this host"))
            continue
        missing = [mount for mount in recorded_mounts if mount not in points]
        if missing:
            reasons.append(
                (
                    str(path),
                    f"recorded cache mount point {', '.join(missing)} is not mounted. Mount it "
                    + "and re-run. If you removed that mount on purpose, run `pivot checkout` "
                    + "in a project that uses this cache; its naming step records the mounts "
                    + "it has now",
                )
            )
    return reasons


def read_register(
    snapshot: StoreSnapshot,
) -> tuple[list[ProjectScope], list[tuple[str, str]]]:
    """Return the register's live projects and the entries a collection cannot use.

    A live entry supplies positive evidence only after the collector finds the entry's project
    id in the registered project's current state directory, which comes from that project's own
    ``core.state_dir`` resolver. On the registering host, every recorded mount below the project
    must still be mounted. ``resolve_scopes`` then opens every scope's root, `.pivot` and state
    directory before any record is read. The register is the one ``snapshot`` resolved.
    """
    register = snapshot.layout.register_root / naming.REGISTER_DIR
    live = list[ProjectScope]()
    points = naming.mount_points(naming.read_mount_table())  # always read now; never the memo
    unreadable = _cache_mount_reasons(snapshot, points)
    try:
        with os.scandir(register) as entries:
            paths = sorted(
                pathlib.Path(entry.path) for entry in entries if entry.name.endswith(".json")
            )
    except FileNotFoundError:
        return live, unreadable
    except OSError as e:
        unreadable.append((str(register), e.strerror or str(e)))
        return live, unreadable
    for path in paths:
        try:
            registered = _project_from_entry(path)
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as e:
            unreadable.append((str(path), str(e)))
            continue
        scope = registered["scope"]
        try:
            scope["state_dir"] = state_dir_mod.locate_state_dir(scope["root"])
        except (exceptions.ConfigError, UnicodeDecodeError) as e:
            # The project is there: its configuration is what needs fixing, not its entry.
            unreadable.append(
                (
                    scope_label(scope),
                    f"{_recorded_where(scope)}, and its configuration cannot be read ({e}). "
                    + "Fix that project's configuration and re-run",
                )
            )
            continue
        except OSError as e:
            unreadable.append(_registered_scope_reason(path, scope, e.strerror or str(e)))
            continue
        reason = _identity_reason(path, registered, points)
        if reason is not None:
            unreadable.append(reason)
            continue
        live.append(scope)
    return live, unreadable


def refusal_external(cache_dir: pathlib.Path, project_root: pathlib.Path) -> str:
    """What a collection says when an external cache records no project that uses it."""
    return (
        f"Refusing to collect {cache_dir}: it is outside this project\n"
        f"({project_root}), so other projects can hold outputs\n"
        "in it, and the cache records no project that uses it. Name every project that shares "
        "this cache; pivot\nrecords them under <cache>/projects/ and later runs need no flag:\n"
        "  pivot gc --projects /path/to/other-project --projects /path/to/another\n"
        "If this project is the only user, run: pivot gc --projects .\n"
        "Nothing was removed."
    )


def refusal_unshared_store(cache_dir: pathlib.Path, layout: naming.StoreLayout) -> str:
    """What a collection says when the cache's object store is the root of a mount of its own,
    so that the register cannot follow it (``naming.StoreLayout.unshared``): ``files/`` is
    itself a mount point, or a link to one. Writers refuse it with the same text."""
    return (
        f"Refusing to collect {cache_dir}: "
        + naming.unshared_store_problem(cache_dir, layout)
        + "\nNothing was removed."
    )


def refusal_store_moved(snapshot: StoreSnapshot, reason: str) -> str:
    """What a collection says when the cache no longer resolves to the store it resolved at its
    start (``StoreSnapshot.check``)."""
    return (
        f"Refusing to go on collecting {snapshot.cache_dir}: {reason}. This collection read the "
        f"register,\nthe records and the plan for the store {snapshot.object_root}, and removes "
        "nothing once the\ncache's layout has changed under it: a files/ link re-pointed or a "
        "store directory replaced\nwhile it ran. Run pivot gc again. Nothing further was removed."
    )


def scope_label(scope: ProjectScope) -> str:
    """How a refusal names one registered project."""
    return f"{scope['root']} (host {scope['host']}, last wrote {scope['last_write']})"


def refusal_not_a_project(cache_dir: pathlib.Path, path: pathlib.Path) -> str:
    """What a collection says when a ``--projects`` path is not a pivot project it can read."""
    return (
        f"Refusing to collect {cache_dir}: --projects {path} is not a pivot project; it has "
        "no readable .pivot\nstate directory, so it names no records and cannot protect "
        "anything in this cache. Name every\nproject that shares this cache by its root (the "
        "directory that holds .pivot). Nothing was removed."
    )


def refusal_unreadable(
    cache_dir: pathlib.Path, recorded: int, unreadable: list[tuple[str, str]], remedy: str
) -> str:
    """What a collection says when a project the cache records cannot be read, ending with
    ``remedy`` (one of the ``REMEDY_*`` texts, for the kind of refusal)."""
    lines = [
        f"Refusing to collect {cache_dir}: it records {recorded} projects and "
        + f"{len(unreadable)} cannot be read:"
    ]
    lines.extend(f"  {label}: {reason}" for label, reason in unreadable)
    lines.append(f"{remedy} Nothing was removed.")
    return "\n".join(lines)


def resolve_scopes(
    snapshot: StoreSnapshot,
    project_root: pathlib.Path,
    state_dir: pathlib.Path,
    extra: Iterable[pathlib.Path],
    *,
    dry_run: bool = False,
) -> list[ProjectScope]:
    """Return the projects whose records this collection must read.

    The project the collection runs in is always in scope. **Every** cache is checked against
    its register (BLK-3) — even one inside the project root, because a peer can point its
    cache at this one and a peer's named objects must be kept. An external cache without any
    registered or named project is refused, and so is a cache with any entry this run cannot
    use (``read_register``). ``dry_run`` suppresses the register writes for ``--projects``
    paths, so that a dry run changes nothing (M1).

    Raises:
        exceptions.CacheCollectionRefusedError: a ``--projects`` path is not a pivot project
            this cache can read, or the cache is external and records no usable project, or an
            entry cannot be read, or the object store is the root of a mount of its own.
    """
    cache_dir = snapshot.cache_dir
    here = ProjectScope(
        root=project_root, state_dir=state_dir, host=socket.gethostname(), last_write="now"
    )
    if snapshot.layout.unshared:
        raise exceptions.CacheCollectionRefusedError(
            refusal_unshared_store(cache_dir, snapshot.layout)
        )
    # Always read the register (BLK-3), before any --projects registration can replace evidence
    # a collection must check: an internal cache can have a registered peer too.
    live, unreadable = read_register(snapshot)
    if unreadable:
        raise exceptions.CacheCollectionRefusedError(
            refusal_unreadable(cache_dir, len(live) + len(unreadable), unreadable, REMEDY_REGISTER)
        )
    named = list[ProjectScope]()
    for path in extra:
        root = path.resolve()
        marker = root / ".pivot"
        if not (marker.is_dir() and os.access(marker, os.R_OK | os.X_OK)):
            raise exceptions.CacheCollectionRefusedError(refusal_not_a_project(cache_dir, root))
        # This project's resolver, not `.pivot`, defines the current state directory. It creates
        # nothing, so a missing or unreadable directory refuses rather than looking empty.
        try:
            scope_state_dir = state_dir_mod.locate_state_dir(root)
        except (exceptions.ConfigError, UnicodeDecodeError) as e:
            raise exceptions.CacheCollectionRefusedError(
                refusal_unreadable(
                    cache_dir, 1, [(str(root), f"cannot read its config: {e}")], REMEDY_CONFIG
                )
            ) from e
        scope = ProjectScope(
            root=root,
            state_dir=scope_state_dir,
            host=socket.gethostname(),
            last_write="now",
        )
        try:
            _read_scope_directories(scope)
        except (OSError, ValueError) as e:
            raise exceptions.CacheCollectionRefusedError(
                refusal_unreadable(
                    cache_dir,
                    1,
                    [
                        (
                            scope_label(scope),
                            f"its state directory cannot be read: {_error_text(e)}",
                        )
                    ],
                    REMEDY_ACCESS,
                )
            ) from e
        named.append(scope)
        if not dry_run:
            naming.register_project(cache_dir, root, scope_state_dir, layout=snapshot.layout)

    # The external-only refusal: a cache outside the project root with no registered or named
    # project, and none on the command line, is someone else's cache.
    is_external = not cache_dir.resolve().is_relative_to(project_root.resolve())
    if is_external and not live and not named:
        raise exceptions.CacheCollectionRefusedError(refusal_external(cache_dir, project_root))
    scopes = _dedupe_scopes([here, *live, *named])
    for scope in scopes:
        try:
            _read_scope_directories(scope)
        except (OSError, ValueError) as e:
            raise exceptions.CacheCollectionRefusedError(
                refusal_unreadable(
                    cache_dir,
                    len(scopes),
                    [(scope_label(scope), _error_text(e))],
                    REMEDY_ACCESS,
                )
            ) from e
    return scopes


def _error_text(error: OSError | ValueError) -> str:
    """An error's reason without its errno prefix; a NUL byte in a path raises ValueError."""
    return error.strerror if isinstance(error, OSError) and error.strerror else str(error)


def _dedupe_scopes(scopes: list[ProjectScope]) -> list[ProjectScope]:
    """One scope per project root, keeping the first."""
    seen = set[pathlib.Path]()
    unique = list[ProjectScope]()
    for scope in scopes:
        root = scope["root"].resolve()
        if root in seen:
            continue
        seen.add(root)
        unique.append(scope)
    return unique


# -----------------------------------------------------------------------------
# The records that name an object
# -----------------------------------------------------------------------------


class _Listings:
    """Reads the members of each directory a record names.

    A directory's listing is the cache object at the directory's hash, and is committed beside
    the record that names it (``storage/listing.py``). This takes the cache's object first,
    because it is the one every reader falls back to and the one pivot itself wrote; a
    committed file is used only when the cache has none, and only when its bytes hash to the
    directory the record names. A directory whose listing neither holds raises
    ``ListingMissingError``, which refuses the collection: its members are unknown.

    ``add`` takes each committed listing file a walk found; they are hashed only when a record
    is not in the cache (a fresh clone, whose listings are in git and whose cache is empty),
    so an ordinary collection reads none of them.
    """

    _object_root: pathlib.Path
    _files: list[pathlib.Path]
    _committed: dict[str, pathlib.Path] | None

    def __init__(self, object_root: pathlib.Path) -> None:
        self._object_root = object_root
        self._files = list[pathlib.Path]()
        self._committed = None

    def add(self, path: pathlib.Path) -> None:
        """Index a committed listing file for a record the cache has no object for."""
        self._files.append(path)

    def _by_digest(self) -> dict[str, pathlib.Path]:
        if self._committed is None:
            committed = dict[str, pathlib.Path]()
            for path in self._files:
                try:
                    committed.setdefault(listing.digest(path.read_bytes()), path)
                except OSError as e:
                    logger.debug("Cannot read the committed listing %s: %s", path, e)
            self._committed = committed
        return self._committed

    def members(self, record: DirHash, source: pathlib.Path) -> listing.DirListing:
        """The files of the directory ``record`` names, as ``source`` records it.

        Raises:
            exceptions.ListingMissingError: no listing for it is in the cache or committed.
            exceptions.CorruptListingError: the listing found does not hold what ``record`` says.
        """
        cached = cache.resolve_cache_object_path(self._object_root, record["hash"])
        if cached is not None:
            with contextlib.suppress(FileNotFoundError):
                return listing.read_found(cached.read_bytes(), str(cached), record, None)
        committed = self._by_digest().get(record["hash"])
        if committed is not None:
            return listing.read_found(committed.read_bytes(), str(committed), record, None)
        raise exceptions.ListingMissingError(
            f"{source} records the directory {record['hash']}, and no listing of it is in the "
            + f"cache ({cache.get_cache_path(self._object_root, record['hash'])}) or committed "
            + "in this project; its files are unknown"
        )


class _UninterpretableRecordError(Exception):
    """A record this pivot cannot fully interpret (a stage lock, a ``.pvt``, a run-cache row):
    ``collect_roots`` refuses and names it. Its message is the record and the reason."""


def _uninterpretable_record(kind: str, path: pathlib.Path | str, reason: str) -> str:
    """The refusal reason for one record this pivot cannot fully read, naming it; the remedy is
    ``REMEDY_RECORD``."""
    return f"cannot fully read the {kind} {path}: {reason}"


def _yaml_problem(error: yaml.YAMLError | UnicodeDecodeError) -> str:
    """A parse error's first line, for a refusal: git conflict markers read as a scanner error."""
    return str(error).strip().splitlines()[0] if str(error).strip() else type(error).__name__


def _lock_keys(path: pathlib.Path, listings: _Listings) -> set[str]:
    """Every key a stage lock names: each ``outs`` and ``deps`` hash, each object its run read
    by hash (``reads``), and every file of each directory it records, read from that
    directory's listing (``lock.recorded_hashes``).

    ``open`` raising ``OSError`` refuses the run (the lock cannot be read, so what it names is
    unknown). A lock this pivot cannot fully interpret refuses it too
    (``_UninterpretableRecordError``, naming the lock): YAML that does not parse (a git merge's
    conflict markers, a truncated file), not a lock, a lock key missing, a schema version that
    is not a whole number or is newer than this pivot's, a key it does not know (a newer
    pivot's record of objects), or a field it cannot read."""
    try:
        with open(path) as f:
            data: object = yaml.load(f, Loader=yaml_config.Loader)
    except (yaml.YAMLError, UnicodeDecodeError) as e:
        raise _UninterpretableRecordError(
            _uninterpretable_record(
                "stage lock", path, f"it does not parse as YAML ({_yaml_problem(e)})"
            )
        ) from e
    recorded = lock.recorded_hashes(data)
    if isinstance(recorded, lock.UninterpretableLock):
        raise _UninterpretableRecordError(
            _uninterpretable_record("stage lock", path, f"it holds {recorded.reason}")
        )
    keys = set(recorded.keys)
    for record in recorded.directories:
        keys.update(entry["hash"] for entry in listings.members(record, path)["files"])
    return keys


def _pvt_keys(
    path: pathlib.Path,
    object_root: pathlib.Path,
    stage_member: Callable[[pathlib.Path, track.PvtData | None], bool],
) -> set[str]:
    """Every key a ``.pvt`` names: its hash and, for a directory, every file of its listing
    (``track.tracked_members``: the listing committed beside it, or the cache's object). An
    older pivot's directory ``.pvt`` lists its files inline: those, and the directory hash it
    records (that format's whole-tree object), are its keys.

    A ``.pvt`` this pivot cannot fully interpret refuses the collection
    (``_UninterpretableRecordError``, naming it): YAML that does not parse (a git merge's
    conflict markers), a key or field it does not read, or a hash that is not a cache key. Since
    #523 a directory's ``.pvt`` names its files only through its listing, so no reading of the
    text alone can say which objects such a ``.pvt`` keeps.

    A ``.pvt`` a stage wrote into its own directory output (#540, ``stage_member``) is that
    stage's data, not a record: nothing restores from it. One that parses keeps its keys, but a
    listing it lacks does not refuse; one that does not parse names nothing. Every other ``.pvt``
    is a tracked file's record.
    """
    try:
        with open(path) as f:
            document: object = yaml.load(f, Loader=yaml_config.Loader)
    except (yaml.YAMLError, UnicodeDecodeError) as e:
        if stage_member(path, None):
            return set()
        raise _UninterpretableRecordError(
            _uninterpretable_record(
                "tracked-file record", path, f"it does not parse as YAML ({_yaml_problem(e)})"
            )
        ) from e
    parsed = track.parse_pvt(document)
    if not isinstance(parsed, tuple):
        if stage_member(path, None):
            return set()
        if isinstance(parsed, str):
            reason = parsed
        elif isinstance(document, dict):
            held = ", ".join(sorted(str(key) for key in cast("dict[object, object]", document)))
            reason = f"no .pvt this pivot reads (its keys: {held})"
        else:
            reason = f"no .pvt: its YAML is {type(document).__name__}, not a mapping"
        raise _UninterpretableRecordError(
            _uninterpretable_record("tracked-file record", path, reason)
        )
    data, inline = parsed
    record = track.pvt_to_hash_info(data)
    if not cache.is_valid_hash(record["hash"]):
        raise _UninterpretableRecordError(
            _uninterpretable_record(
                "tracked-file record", path, f"its hash {record['hash']!r} is not a cache key"
            )
        )
    keys = {record["hash"]}
    if inline is not None:
        legacy = cast("dict[str, object]", document).get("hash")
        if cache.is_valid_hash(legacy):
            keys.add(cast("str", legacy))
        keys.update(entry["hash"] for entry in inline["files"])
        return keys
    if is_dir_hash(record):
        try:
            members = track.tracked_members(track.get_data_path(path), data, object_root)
        except (exceptions.ListingMissingError, exceptions.CorruptListingError):
            if stage_member(path, data):
                return keys
            raise
        keys.update(entry["hash"] for entry in members["files"])
    return keys


class _StateRecords(NamedTuple):
    """What one state directory's ``stages`` directory holds for a collection."""

    locks: list[pathlib.Path]
    listings: list[pathlib.Path]


def _state_records(stages_dir: pathlib.Path) -> _StateRecords:
    """Every lock and every committed listing below ``stages_dir``, following symlinked
    directories once each.

    A dangling link refuses the collection: it may hide stage locks on a disk that is not
    mounted. A directory named ``*.lock`` is returned so its open refuses, as an unreadable
    lock does.
    """
    locks = list[pathlib.Path]()
    listings = list[pathlib.Path]()
    seen = set[tuple[int, int]]()
    stack = [stages_dir]
    while stack:
        directory = stack.pop()
        stats = directory.stat()
        identity = (stats.st_dev, stats.st_ino)
        if identity in seen:
            continue
        seen.add(identity)
        with os.scandir(directory) as entries:
            for entry in entries:
                path = pathlib.Path(entry.path)
                if entry.name.endswith(".lock"):
                    locks.append(path)
                elif entry.name.endswith(listing.SUFFIX) and entry.is_file():
                    listings.append(path)
                elif entry.is_dir():
                    stack.append(path)
                elif entry.is_symlink() and not os.path.exists(path):
                    raise FileNotFoundError(errno.ENOENT, "symlink target is missing", str(path))
    return _StateRecords(sorted(locks), listings)


def _row_keys(output: run_history.OutputHashEntry, source: str, listings: _Listings) -> set[str]:
    """Every key one run-cache output entry ``run_history.run_cache_entry_error`` passed names:
    its own, and each file of a directory's listing — the one an older pivot's row holds inline,
    else the one ``listings`` reads. A hash that is not a cache key refuses."""
    if not cache.is_valid_hash(output["hash"]):
        raise _UninterpretableRecordError(
            _uninterpretable_record(
                "run-cache row", source, f"its output hash {output['hash']!r} is not a cache key"
            )
        )
    record = run_history.entry_to_output_hash(output)
    if record is None:
        # An older pivot's row lists a directory's files inline; one that records neither is a
        # row no pivot writes, so nothing proves what it names and the collection refuses.
        raw = cast("dict[str, object]", cast("object", output))
        inline = listing.from_inline(raw.get("manifest"), raw.get("empty_dirs", []))
        if inline is None:
            raise _UninterpretableRecordError(
                _uninterpretable_record(
                    "run-cache row",
                    source,
                    f"it records the directory {output['hash']} without its size, file count "
                    + "or files",
                )
            )
        return {output["hash"], *(entry["hash"] for entry in inline["files"])}
    keys = {record["hash"]}
    if is_dir_hash(record):
        keys.update(
            entry["hash"] for entry in listings.members(record, pathlib.Path(source))["files"]
        )
    return keys


def _run_cache_keys(
    state_dir: pathlib.Path, listings: _Listings, valid_run_ids: set[str] | None = None
) -> tuple[set[str], int]:
    """Every key the run-cache rows of one project's database name, and the number of rows.

    A row names each output's hash, each file of a directory output's listing, and each object
    the run read by hash (``reads``). The keys also hold every object a stage's `nocommit:`
    record lists as read (``StateDB.list_no_commit_reads``): `pivot commit` writes those into
    the lock, so they are live until then. A run-history prune drops no `nocommit:` record.

    Each row is read as stored, and one this pivot cannot fully interpret (JSON that does not
    parse, a key it does not know, a field of the wrong type, a hash that is not a cache key)
    refuses the collection, naming it (``_UninterpretableRecordError``).

    ``valid_run_ids`` restricts the rows to those a run-history prune would keep — its sentinel
    rows (``__``-prefixed) and the rows of runs within the retention — so the plan counts an
    object whose only record is a row the confirmed run will drop (``StateDB.prune_runs``)."""
    database = state_dir / "state.lmdb"
    if not database.exists():
        return set(), 0
    with state.StateDB(state_dir, readonly=True) as db:
        stored = db.list_run_cache_rows()
        no_commit_reads = db.list_no_commit_reads()
    rows = list[tuple[str, run_history.RunCacheEntry]]()
    for name, data in stored:
        source = f"{database} {name}"
        try:
            row: object = json.loads(data.decode())
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise _UninterpretableRecordError(
                _uninterpretable_record("run-cache row", source, f"it does not parse as JSON ({e})")
            ) from e
        if (error := run_history.run_cache_entry_error(row)) is not None:
            raise _UninterpretableRecordError(
                _uninterpretable_record("run-cache row", source, error)
            )
        rows.append((source, cast("run_history.RunCacheEntry", row)))
    if valid_run_ids is not None:
        rows = [
            (source, entry)
            for source, entry in rows
            if entry["run_id"].startswith("__") or entry["run_id"] in valid_run_ids
        ]
    keys = set(no_commit_reads)
    for source, entry in rows:
        if "reads" in entry:
            keys.update(entry["reads"])
        for output in entry["output_hashes"]:
            keys |= _row_keys(output, source, listings)
    return keys, len(rows)


def _scope_state_dirs(scope: ProjectScope) -> tuple[pathlib.Path, ...]:
    """Every state layout this project can use to name cache objects.

    #503 keeps a 0.5.2 project's old `.pivot` records readable after it configures another
    state directory. A collection must read both: the newer configuration does not prove that
    the legacy directory names nothing.
    """
    current = scope["state_dir"]
    legacy = scope["root"] / ".pivot"
    return (current,) if current == legacy else (current, legacy)


class _WalkResult(NamedTuple):
    """What a walk of one project found for a collection."""

    pvt_paths: list[pathlib.Path]
    keys: set[str]
    outward: list[pathlib.Path]
    listings: list[pathlib.Path]


class StoreSnapshot(NamedTuple):
    """The cache's store layout as one collection resolved it at its start (RP).

    Every lock, register read, record walk, plan and unlink of the collection goes through it,
    never through ``files/`` again: a ``files/`` link re-pointed while the collection runs would
    otherwise send a later step to another store, whose register this collection never read.
    ``object_root`` and ``trees`` are real paths, so a plan's paths name the snapshot's store
    whatever the link says later, and each store directory that exists at the start is held
    open (``anchors``); one that does not stays absent for the whole collection (``anchor``).
    ``check`` tells when a path no longer names the directory the collection opened (a store
    directory renamed and another put in its place). Every prefix directory is opened from the
    held store, never through a path and never following a link (``open_prefix``, through
    ``_open_directory_at``): the plan lists it through such a descriptor and records its
    identity, and the removals open it again in bounded batches, require that identity and
    remove through the descriptor (``execute_plan``).

    ``trees`` holds the whole-directory tree stores: a stage's commit puts a tree beside the
    object store it resolves ``files/`` to, which is the register root; a command given the
    unresolved ``files/`` (``pivot track``, a restore) puts it under the cache root. In a plain
    layout the two are one directory. Both are this cache's: every project that can link either
    is in the register a collection reads.
    """

    cache_dir: pathlib.Path
    layout: naming.StoreLayout
    object_root: pathlib.Path
    trees: tuple[pathlib.Path, ...]
    anchors: tuple[tuple[pathlib.Path, int], ...]

    def check(self) -> None:
        """Refuse to go on when the cache no longer resolves to this snapshot: ``files/`` now
        resolves to another directory, or a directory of the snapshot is no longer the one the
        collection opened.

        Raises:
            exceptions.CacheCollectionRefusedError: the layout changed under the collection.
        """
        current = _resolved_store(self.cache_dir)
        if current != self.object_root:
            raise exceptions.CacheCollectionRefusedError(
                refusal_store_moved(self, f"its files/ now resolves to {current}")
            )
        for path, fd in self.anchors:
            try:
                now = os.stat(path)
            except OSError as e:
                raise exceptions.CacheCollectionRefusedError(
                    refusal_store_moved(self, f"{path} cannot be read ({_error_text(e)})")
                ) from e
            held = os.fstat(fd)
            if (now.st_dev, now.st_ino) != (held.st_dev, held.st_ino):
                raise exceptions.CacheCollectionRefusedError(
                    refusal_store_moved(self, f"{path} is no longer the directory it opened")
                )

    def anchor(self, store: pathlib.Path) -> int | None:
        """The held descriptor of the store directory ``store`` (the object store, or a tree
        store); None when it did not exist when the collection started. Such a store stays
        absent for the whole collection: it gets no listing and no removal, whatever appears at
        its path meanwhile (a directory, or a link to another cache's store)."""
        return next((fd for path, fd in self.anchors if path == store), None)

    def open_prefix(self, prefix: pathlib.Path) -> int | None:
        """Open the prefix directory ``prefix`` (``<store>/<xx>``) from the held store, never
        through its path and never following a link (``_open_directory_at``); None when its
        store is not held or no directory of its own is there (nothing, a file, a link, or an
        entry replaced while it was opened). The caller closes it."""
        anchor = self.anchor(prefix.parent)
        if anchor is None:
            return None
        return _open_directory_at(anchor, prefix.name)

    def open_listed_prefix(self, prefix: pathlib.Path, listed: tuple[int, int]) -> int:
        """Open the prefix directory the plan listed as ``prefix`` (``open_prefix``) and return
        its descriptor; the caller closes it.

        ``listed`` is its identity when the plan listed it (``Doomed.prefix``). A prefix
        directory renamed aside and replaced since (by another directory, a link, or a mount
        over it) is not that directory, and nothing in it is removed.

        Raises:
            exceptions.CacheCollectionRefusedError: no directory is there, or it is not the
                directory the plan listed.
        """
        fd = self.open_prefix(prefix)
        if fd is not None:
            held = os.fstat(fd)
            if (held.st_dev, held.st_ino) == listed:
                return fd
            os.close(fd)
        raise exceptions.CacheCollectionRefusedError(
            refusal_store_moved(
                self,
                f"the prefix directory {prefix} is no longer the directory the plan listed "
                + "(it was replaced, or something is mounted over it), so nothing was removed "
                + "there",
            )
        )


def _resolved_store(cache_dir: pathlib.Path) -> pathlib.Path:
    """Where ``cache_dir``'s ``files/`` resolves to now."""
    return pathlib.Path(os.path.realpath(cache_dir / naming.OBJECTS_DIR))


def _open_directory_at(
    parent_fd: int, name: str, *, expected: tuple[int, int, int] | None = None
) -> int | None:
    """Open the directory ``name`` in the directory open as ``parent_fd`` without following a
    link, and return its descriptor (the caller closes it); None when no directory of its own
    is there: nothing, a file, a link, an entry replaced while it was opened, or a directory
    whose identity is not ``expected``.

    Every prefix directory a collection lists, and every directory it removes from, is opened
    here (``_tree_bytes`` measures a tree with ``os.fwalk``, which makes the same comparison).
    ``O_NOFOLLOW`` alone does not hold: gVisor follows a link to a directory under
    ``O_DIRECTORY | O_NOFOLLOW``, where Linux fails with ENOTDIR. So the entry is ``lstat``-ed
    first and anything but a directory, a link included, is refused; when a caller recorded the
    directory it expects, that identity is required too. Last, the descriptor opened must show
    the device, inode and change time that ``lstat`` saw, which refuses on either kernel a link
    or another directory put at the name between the two calls.
    """
    try:
        seen = os.lstat(name, dir_fd=parent_fd)
    except FileNotFoundError:
        return None
    if not stat.S_ISDIR(seen.st_mode):
        return None
    seen_identity = seen.st_dev, seen.st_ino, seen.st_ctime_ns
    if expected is not None and seen_identity != expected:
        return None
    try:
        fd = os.open(
            name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent_fd
        )
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError as e:
        if e.errno != errno.ELOOP:
            raise
        return None
    opened = os.fstat(fd)
    if (opened.st_dev, opened.st_ino, opened.st_ctime_ns) != seen_identity:
        os.close(fd)
        return None
    return fd


@contextlib.contextmanager
def store_snapshot(cache_dir: pathlib.Path) -> Generator[StoreSnapshot]:
    """Resolve the cache's store layout once, for one collection, holding each of its existing
    store directories open until the collection ends (``StoreSnapshot``)."""
    layout = naming.store_layout(cache_dir)
    object_root = _resolved_store(cache_dir)
    trees = [pathlib.Path(os.path.realpath(layout.register_root / TREES_DIR))]
    own_trees = pathlib.Path(os.path.realpath(cache_dir / TREES_DIR))
    if own_trees != trees[0]:
        trees.append(own_trees)
    register_root = pathlib.Path(os.path.realpath(layout.register_root))
    anchors = list[tuple[pathlib.Path, int]]()
    try:
        for path in dict.fromkeys((register_root, object_root, *trees)):
            try:
                fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            except (FileNotFoundError, NotADirectoryError):
                continue
            anchors.append((path, fd))
        yield StoreSnapshot(cache_dir, layout, object_root, tuple(trees), tuple(anchors))
    finally:
        for _path, fd in anchors:
            os.close(fd)


def _walk_project(
    root: pathlib.Path,
    object_root: pathlib.Path,
    trees: list[pathlib.Path],
    skip: pathlib.Path | None,
) -> _WalkResult:
    """Every ``.pvt`` file under ``root``, every key a symlink there resolves to, every
    committed listing file, and every symlinked directory that leaves the project.

    A file-by-file symlink checkout links each member to its object; a whole-directory
    checkout links the directory itself to its tree under one of ``trees`` (``StoreSnapshot.trees``).
    Both are read here. ``skip`` is the cache itself, listed object by object rather than walked.

    A symlinked directory is walked when the place it really is lies inside ``root``
    (``project.real_location_outside``, BET-301's containment rule, against ``root`` resolved
    once for the walk), as a ``data -> data_real`` link inside the project is. A link that
    leaves the project — ``data -> /mnt/big/data``, or a plain ``.venv -> /opt/venvs/x`` — is
    not followed and is returned instead, because the records under it cannot be read and a
    collection that cannot read a record must never licence a delete: one such link makes the
    collection keep every object. A link into the cache itself is skipped (the cache is listed
    object by object). A directory already walked, by device and inode, is not walked again,
    so a link cycle terminates.
    """
    pvt_paths = list[pathlib.Path]()
    keys = set[str]()
    outward = list[pathlib.Path]()
    listings = list[pathlib.Path]()
    seen = set[tuple[int, int]]()
    resolved_root = root.resolve()
    resolved_objects = object_root.resolve()
    resolved_trees = [tree_root.resolve() for tree_root in trees]
    stack = [root]
    while stack:
        directory = stack.pop()
        stats = directory.stat()
        if (stats.st_dev, stats.st_ino) in seen:
            continue
        seen.add((stats.st_dev, stats.st_ino))
        with os.scandir(directory) as entries:
            for entry in entries:
                path = pathlib.Path(entry.path)
                if entry.is_symlink():
                    try:
                        key = cache.get_symlink_cache_key(path, object_root)
                    except exceptions.UnreadableLinkError:
                        # #519 cannot classify the link as a cache object. It may name a live
                        # object, so preserve every object and name this link in the warning.
                        outward.append(path)
                        continue
                    if key is not None and cache.is_valid_hash(key):
                        # A file-by-file checkout of a cache object: not a subtree.
                        keys.add(key)
                    elif entry.name.endswith(".pvt") and entry.is_file():
                        pvt_paths.append(path)
                    elif entry.is_dir():
                        where = _link_target(path, resolved_root, resolved_objects, resolved_trees)
                        if where.place == "inside":
                            stack.append(path)
                        elif where.place == "outside":
                            outward.append(path)
                        elif where.key is not None:
                            # A whole-directory symlink checkout: the tree is this link's record.
                            keys.add(where.key)
                    elif not os.path.exists(path):
                        # Its target may be a record directory on a disk that is not mounted.
                        outward.append(path)
                elif entry.is_dir(follow_symlinks=False):
                    if path != skip:
                        stack.append(path)
                elif entry.name.endswith(".pvt"):
                    pvt_paths.append(path)
                elif entry.name.endswith(listing.SUFFIX):
                    listings.append(path)
    return _WalkResult(pvt_paths, keys, outward, listings)


class _LinkTarget(NamedTuple):
    """Where a symlinked directory really is, and the tree object it is when it is one."""

    place: Literal["inside", "outside", "cache"]
    key: str | None


def _link_target(
    path: pathlib.Path,
    resolved_root: pathlib.Path,
    resolved_objects: pathlib.Path,
    resolved_trees: list[pathlib.Path],
) -> _LinkTarget:
    """Where the symlinked directory ``path`` really is: inside the project, outside it, or in
    the cache (listed object by object, never walked), with the key of the whole-directory tree
    it links to when it is one.

    ``project.real_location_outside`` is BET-301's containment rule (#533); a link it cannot
    resolve counts as outside, because where it leads is unknown. Only a link into
    ``<cache>/files`` or ``<cache>/trees`` is the cache's own: the cache root can be a broad
    scratch directory that also holds a user's data (``PIVOT_CACHE_DIR=/scratch``), and records
    under it are records like any others, so a link there that leaves the project counts as
    leaving it. The one other link that names objects of this cache is a whole-directory tree
    under another cache root whose ``files/`` resolves to this cache's object store: the
    register is shared by every root of the store (``naming.store_layout``), so a project
    registered through another root links that root's trees, and the tree's key keeps its
    listing and members here (the tree itself is that root's to collect).
    """
    outside = project.real_location_outside(path, resolved_root)
    real = path.resolve() if outside is None else outside
    if real.is_relative_to(resolved_objects):
        return _LinkTarget("cache", None)
    if any(real.is_relative_to(tree_root) for tree_root in resolved_trees):
        key = real.parent.name + real.name
        return _LinkTarget("cache", key if cache.is_valid_hash(key) else None)
    if outside is not None and _tree_of_store(real, resolved_objects):
        key = real.parent.name + real.name
        if cache.is_valid_hash(key):
            return _LinkTarget("cache", key)
    return _LinkTarget("inside" if outside is None else "outside", None)


def _tree_of_store(real: pathlib.Path, resolved_objects: pathlib.Path) -> bool:
    """Whether ``real`` (a resolved path) is ``<root>/trees/XX/YYYY…`` under a cache root whose
    ``files/`` resolves to ``resolved_objects``."""
    trees = real.parent.parent
    return (
        trees.name == TREES_DIR
        and pathlib.Path(os.path.realpath(trees.parent / naming.OBJECTS_DIR)) == resolved_objects
    )


def collect_roots(
    scopes: Iterable[ProjectScope],
    snapshot: StoreSnapshot,
    *,
    prune_state_dir: pathlib.Path | None = None,
    valid_run_ids: set[str] | None = None,
) -> Roots:
    """Read every record of every project in scope and return the keys they name.

    ``prune_state_dir`` and ``valid_run_ids`` restrict that project's run-cache rows to the ones
    a run-history prune keeps, so the plan reflects the confirmed run's prune (B3/B4).

    Raises:
        ProjectUnreadableError: a project in scope cannot be read, so what it names is
            unknown and nothing may be removed: a record it cannot open, a stage lock, ``.pvt``
            or run-cache row it cannot fully interpret (each named), or a directory record whose
            listing is neither in the cache nor committed.
    """
    cache_dir = snapshot.cache_dir
    object_root = snapshot.object_root
    trees = list(snapshot.trees)
    roots = Roots(
        named=set[str](),
        run_cache=set[str](),
        linked=set[str](),
        counts=RootCounts(locks=0, pvt_files=0, rows=0, links=0),
        outward_dirs=list[str](),
    )
    for scope in scopes:
        try:
            # Every listing this project committed is indexed before its records are read, so a
            # record whose listing the cache has lost is still read from the file beside it.
            walked = _walk_project(scope["root"], object_root, trees, cache_dir)
            listings = _Listings(object_root)
            records = dict[pathlib.Path, _StateRecords]()
            for state_dir in _scope_state_dirs(scope):
                stages_dir = lock.get_stages_dir(state_dir)
                if not stages_dir.is_dir():
                    continue
                records[state_dir] = _state_records(stages_dir)
                for listing_path in records[state_dir].listings:
                    listings.add(listing_path)
            for listing_path in walked.listings:
                listings.add(listing_path)
            for state_dir in _scope_state_dirs(scope):
                for lock_path in records[state_dir].locks if state_dir in records else ():
                    roots["counts"]["locks"] += 1
                    roots["named"] |= _lock_keys(lock_path, listings)
                scope_valid = valid_run_ids if state_dir == prune_state_dir else None
                rows_keys, rows = _run_cache_keys(state_dir, listings, scope_valid)
                roots["counts"]["rows"] += rows
                roots["run_cache"] |= rows_keys
            # Read only when a directory .pvt's listing is missing: the locks it needs are the
            # ones `member_skip` reads for every other command's discovery.
            stage_member = member_skip.own_member_skip(
                scope["root"], {}, scope["state_dir"], object_root
            )
            for pvt_path in walked.pvt_paths:
                roots["counts"]["pvt_files"] += 1
                roots["named"] |= _pvt_keys(pvt_path, object_root, stage_member)
            roots["counts"]["links"] += len(walked.keys)
            roots["linked"] |= walked.keys
            roots["outward_dirs"].extend(str(path) for path in walked.outward)
        except (OSError, lmdb.Error, state.StateDBError) as e:
            raise ProjectUnreadableError(
                scope, e.strerror if isinstance(e, OSError) and e.strerror else str(e)
            ) from e
        except (exceptions.ListingMissingError, exceptions.CorruptListingError) as e:
            raise ProjectUnreadableError(scope, str(e), REMEDY_LISTING) from e
        except _UninterpretableRecordError as e:
            raise ProjectUnreadableError(scope, str(e), REMEDY_RECORD) from e

    refs = object_root / "refs"
    if refs.is_dir():
        roots["counts"]["links"] += len(
            ref_keys := _walk_project(refs, object_root, trees, None).keys
        )
        roots["linked"] |= ref_keys
    return roots


def reread_run_cache(roots: Roots, scopes: Iterable[ProjectScope], snapshot: StoreSnapshot) -> None:
    """Replace ``roots``' run-cache keys with every row the scopes' databases hold now.

    Called under the exclusive naming lock after a run-history prune, so the rows the prune
    dropped stop protecting their objects and every row that remains, however recent, does.
    A row that records a directory is read with its listing, as ``collect_roots`` reads one.

    Raises:
        ProjectUnreadableError: a scope's database, a row this pivot cannot fully interpret,
            or the listing of a directory one of its rows records, cannot be read.
    """
    cache_dir = snapshot.cache_dir
    object_root = snapshot.object_root
    trees = list(snapshot.trees)
    keys = set[str]()
    rows = 0
    for scope in scopes:
        try:
            listings = _Listings(object_root)
            for listing_path in _walk_project(
                scope["root"], object_root, trees, cache_dir
            ).listings:
                listings.add(listing_path)
            for state_dir in _scope_state_dirs(scope):
                stages_dir = lock.get_stages_dir(state_dir)
                if stages_dir.is_dir():
                    for listing_path in _state_records(stages_dir).listings:
                        listings.add(listing_path)
                scope_keys, scope_rows = _run_cache_keys(state_dir, listings)
                keys |= scope_keys
                rows += scope_rows
        except (OSError, lmdb.Error, state.StateDBError) as e:
            raise ProjectUnreadableError(
                scope, e.strerror if isinstance(e, OSError) and e.strerror else str(e)
            ) from e
        except (exceptions.ListingMissingError, exceptions.CorruptListingError) as e:
            raise ProjectUnreadableError(scope, str(e), REMEDY_LISTING) from e
        except _UninterpretableRecordError as e:
            raise ProjectUnreadableError(scope, str(e), REMEDY_RECORD) from e
    roots["run_cache"] = keys
    roots["counts"]["rows"] = rows


# -----------------------------------------------------------------------------
# Locking, scanning and removing
# -----------------------------------------------------------------------------


@contextlib.contextmanager
def collection_lock(snapshot: StoreSnapshot) -> Generator[None]:
    """Hold the cache's ``gc.lock`` exclusively so two collections never run on one store.

    Lock order: gc.lock is taken FIRST, before the naming lock.

    Raises:
        exceptions.CacheCollectionRefusedError: another collection holds it, or the cache
            cannot be written, or ``flock`` is refused on this filesystem.
    """
    cache_dir = snapshot.cache_dir
    path = snapshot.layout.register_root / GC_LOCK
    try:
        fd = cache.open_cache_lock(path)
    except OSError as e:
        raise exceptions.CacheCollectionRefusedError(
            f"Refusing to collect {cache_dir}: cannot open {path} ({e.strerror or e}). "
            + "Collecting a cache needs write access to it. Nothing was removed."
        ) from e
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise exceptions.CacheCollectionRefusedError(
                f"Refusing to collect {cache_dir}: another pivot gc holds {path}. "
                + "Wait for it to finish. Nothing was removed."
            ) from None
        except OSError as e:
            if e.errno in cache.NO_FLOCK_ERRNO:
                raise exceptions.CacheCollectionRefusedError(
                    f"Refusing to collect {cache_dir}: this filesystem refuses flock(), so "
                    + "the cache's locks exclude nothing here and a collection cannot safely "
                    + "delete while other processes may be naming objects. Move the cache to a "
                    + "filesystem with working locks, or collect with every pivot process "
                    + "stopped. Nothing was removed."
                ) from e
            raise
        yield
    finally:
        os.close(fd)


@contextlib.contextmanager
def naming_exclusive(snapshot: StoreSnapshot) -> Generator[None]:
    """Hold the naming lock exclusive: no writer can name a cache object while this is held.

    Lock order: gc.lock → naming.turnstile (EX) → naming.lock (EX); no prefix lock under it.
    The turnstile prevents writer starvation: gc takes it exclusive first so new commit
    units queue, then gc takes naming exclusive to wait for in-flight units to finish. The lock
    files are the ones beside the register ``snapshot`` resolved.
    """
    ts_path, nm_path = naming.lock_paths_in(snapshot.layout.register_root)
    ts_fd = cache.open_cache_lock(ts_path)
    nm_fd = cache.open_cache_lock(nm_path)
    try:
        fcntl.flock(ts_fd, fcntl.LOCK_EX)
        fcntl.flock(nm_fd, fcntl.LOCK_EX)
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(nm_fd, fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            fcntl.flock(ts_fd, fcntl.LOCK_UN)
        os.close(nm_fd)
        os.close(ts_fd)


@contextlib.contextmanager
def _prefix_locked(prefix_fd: int) -> Generator[None]:
    """Hold one prefix directory's lock exclusively, excluding every staging publisher and
    the temporary sweep while this directory is listed. ``prefix_fd`` is the directory, opened
    from the held store (``StoreSnapshot.open_prefix``); the lock file is opened in it, not
    through a path.

    Lock order: gc takes this only while it lists, before any naming lock and never under the
    exclusive one; a writer may take a prefix lock shared inside its naming unit."""
    fd = cache.open_cache_lock(pathlib.Path(cache.PREFIX_LOCK), dir_fd=prefix_fd)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError as e:
            if e.errno not in cache.NO_FLOCK_ERRNO:
                raise
        yield
    finally:
        os.close(fd)


def _prefix_names(store_fd: int) -> list[str]:
    """The names of every two-character prefix directory in the held store ``store_fd``, in
    order; a link in their place is not one."""
    with os.scandir(store_fd) as entries:
        return sorted(
            entry.name
            for entry in entries
            if entry.is_dir(follow_symlinks=False) and _is_prefix_name(entry.name)
        )


def _is_prefix_name(name: str) -> bool:
    """Whether a directory of the object root holds objects: the first two hex characters of
    a cache key (``cache.get_cache_path``)."""
    return len(name) == 2 and all(c in "0123456789abcdef" for c in name)


def _tree_bytes(prefix_fd: int, name: str) -> int:
    """The bytes the directory object ``name`` in the held prefix directory ``prefix_fd``
    holds: its members are copies, never links, and the walk follows none."""
    total = 0
    with contextlib.suppress(OSError):
        for _path, _dirs, files, dir_fd in os.fwalk(name, dir_fd=prefix_fd):
            for file in files:
                with contextlib.suppress(OSError):
                    total += os.stat(file, dir_fd=dir_fd, follow_symlinks=False).st_size
    return total


def build_plan(
    snapshot: StoreSnapshot,
    roots: Roots,
    *,
    grace_hours: float,
    remove_linked: bool,
    now: float | None = None,
) -> Plan:
    """List every object in the cache and decide, under each prefix's lock, what may go.

    Both stores are listed: the objects under ``files`` (a file's content, and a directory's
    listing) and the whole-directory trees under ``trees``, which a symlink checkout links to.
    One directory hash names its listing object and its tree, so the record that names the
    directory keeps both. Only the store directories the snapshot holds are listed (a store
    absent when the collection started stays absent), each through its held descriptor, and
    each prefix directory through one opened from it and closed once it is listed
    (``StoreSnapshot.open_prefix``), whose device and inode each condemned object records
    (``Doomed.prefix``) for ``execute_plan``. One prefix descriptor is open at a time, so the
    collection needs a fixed number of descriptors whatever ``ulimit -n`` allows. The plan's
    paths only name what it lists.
    """
    cutoff = (time.time() if now is None else now) - grace_hours * 3600
    plan = Plan(
        scanned=ObjectCount(objects=0, bytes=0),
        kept=Kept(
            objects=0,
            bytes=0,
            by=KeptBy(locks=0, run_cache=0, links=0, grace=0, hardlinked=0),
        ),
        doomed=list[Doomed](),
        doomed_bytes=0,
        linked=ObjectCount(objects=0, bytes=0),
    )
    for store in (snapshot.object_root, *snapshot.trees):
        store_fd = snapshot.anchor(store)
        if store_fd is None:
            continue
        for name in _prefix_names(store_fd):
            prefix_dir = store / name
            prefix_fd = snapshot.open_prefix(prefix_dir)
            if prefix_fd is None:
                continue
            try:
                listed = os.fstat(prefix_fd)
                prefix = (listed.st_dev, listed.st_ino)
                with _prefix_locked(prefix_fd), os.scandir(prefix_fd) as entries:
                    _plan_prefix(
                        plan, prefix_dir, prefix_fd, prefix, entries, roots, cutoff, remove_linked
                    )
            finally:
                os.close(prefix_fd)
    return plan


def _plan_prefix(
    plan: Plan,
    prefix_dir: pathlib.Path,
    prefix_fd: int,
    prefix: tuple[int, int],
    entries: Iterable[os.DirEntry[str]],
    roots: Roots,
    cutoff: float,
    remove_linked: bool,
) -> None:
    """Decide what may go of one prefix directory's ``entries`` (listed through its held
    descriptor ``prefix_fd``, whose identity is ``prefix``), adding to ``plan``."""
    for entry in sorted(entries, key=lambda e: e.name):
        key = prefix_dir.name + entry.name
        # Skips every name that is not an object: the prefix lock, a `.pivot_tmp_*` entry a
        # publisher owns, and a tree's `.content-checked` marker and `.lock` file.
        if not cache.is_valid_hash(key):
            continue
        try:
            stats = entry.stat(follow_symlinks=False)
            is_dir = entry.is_dir(follow_symlinks=False)
        except OSError:
            continue
        size = _tree_bytes(prefix_fd, entry.name) if is_dir else stats.st_size
        plan["scanned"]["objects"] += 1
        plan["scanned"]["bytes"] += size
        # A symlinked directory that leaves a project hides the records under it, so a
        # collection that met one keeps every object and removes none (``collect_roots``).
        # Otherwise the object stays only when a record, the grace or the hardlink rule holds
        # it. A record pivot cannot read refuses before this.
        reason = (
            "locks"
            if roots["outward_dirs"]
            else _keep_reason(
                key, stats, roots, is_dir=is_dir, cutoff=cutoff, remove_linked=remove_linked
            )
        )
        if reason is not None:
            plan["kept"]["objects"] += 1
            plan["kept"]["bytes"] += size
            plan["kept"]["by"][reason] += 1
            continue
        linked = not is_dir and stats.st_nlink > 1
        plan["doomed"].append(
            Doomed(
                path=prefix_dir / entry.name,
                is_dir=is_dir,
                bytes=size,
                linked=linked,
                prefix=prefix,
            )
        )
        plan["doomed_bytes"] += size
        if linked:
            plan["linked"]["objects"] += 1
            plan["linked"]["bytes"] += size


_KeepReason = Literal["locks", "run_cache", "links", "grace", "hardlinked"]


def _keep_reason(
    key: str,
    stats: os.stat_result,
    roots: Roots,
    *,
    is_dir: bool,
    cutoff: float,
    remove_linked: bool,
) -> _KeepReason | None:
    """Why this object stays, or None when no record and no rule holds it.

    ``st_ctime`` is the publication time: ``copy_to_cache`` copies with ``shutil.copy2``, so
    an object's mtime is its source file's and can be arbitrarily older (``storage/cache.py``).
    A directory's ``st_nlink`` counts its subdirectories, so the hardlink rule is for files.
    """
    if key in roots["named"]:
        return "locks"
    if key in roots["run_cache"]:
        return "run_cache"
    if key in roots["linked"]:
        return "links"
    if stats.st_ctime > cutoff:
        return "grace"
    if not is_dir and stats.st_nlink > 1 and not remove_linked:
        return "hardlinked"
    return None


def execute_plan(
    plan: Plan,
    *,
    snapshot: StoreSnapshot,
    roots: Roots,
    grace_hours: float,
    remove_linked: bool,
    now: float | None = None,
) -> ObjectCount:
    """Remove what the plan condemns and return what went.

    The caller holds ``naming_exclusive`` and passes the ``roots`` it collected under that lock,
    so no writer can name an object between the caller's ``collect_roots`` and this pass's
    unlinks. Each object is still re-checked against the grace and the hardlink rule (a checkout
    already in flight before the lock was taken may have hardlinked it).

    No removal goes through a path. Before the first deletion in each prefix directory the
    snapshot is checked again (``StoreSnapshot.check``: ``files/`` resolving elsewhere, or a
    store directory no longer the one the collection opened, refuses), and the prefix directory
    is opened again from the held store, without following a link, and must be the directory the
    plan listed: the ``fstat`` of the opened descriptor must show the device and inode the plan
    recorded (``StoreSnapshot.open_listed_prefix``). Every ``stat``, unlink and tree removal of
    that prefix goes through that descriptor, which is closed before the next prefix is opened.
    So a store directory, or a prefix directory, renamed aside and replaced after the plan (by
    another directory, a link, or a mount over it) receives nothing: the collection refuses,
    naming the prefix, and removes nothing there or after it.

    No prefix lock is taken here: while gc holds the exclusive naming lock it must never wait
    for another process, or every writer on the cache would wait behind it. A publisher staging
    into a prefix (which holds that prefix's lock shared for its whole copy) needs no exclusion
    from these deletions: it publishes outside any naming unit, and its unit re-checks every
    object, publishing again any removed here, before a record names it. Temporary entries are
    never in the plan (``build_plan`` lists each prefix under its lock and takes only keys).

    Raises:
        exceptions.CacheCollectionRefusedError: the cache's layout changed under the collection.
    """
    cutoff = (time.time() if now is None else now) - grace_hours * 3600
    removed = ObjectCount(objects=0, bytes=0)
    current: pathlib.Path | None = None
    prefix_fd: int | None = None
    try:
        for doomed in sorted(plan["doomed"], key=lambda d: d["path"]):
            path = doomed["path"]
            if path.parent != current:
                current = path.parent
                if prefix_fd is not None:
                    os.close(prefix_fd)
                    prefix_fd = None
                snapshot.check()
                prefix_fd = snapshot.open_listed_prefix(path.parent, doomed["prefix"])
            assert prefix_fd is not None
            name = path.name
            try:
                stats = os.stat(name, dir_fd=prefix_fd, follow_symlinks=False)
            except FileNotFoundError:
                removed["objects"] += 1
                removed["bytes"] += doomed["bytes"]
                continue
            key = path.parent.name + name
            if (
                _keep_reason(
                    key,
                    stats,
                    roots,
                    is_dir=doomed["is_dir"],
                    cutoff=cutoff,
                    remove_linked=remove_linked,
                )
                is not None
            ):
                continue
            # M5: unlink markers before the tree, so a partial removal never leaves a marker on
            # a truncated tree that a later checkout would trust.
            if doomed["is_dir"]:
                for suffix in _TREE_MARKER_SUFFIXES:
                    with contextlib.suppress(OSError):
                        os.unlink(name + suffix, dir_fd=prefix_fd)
            try:
                _remove_at(
                    prefix_fd,
                    name,
                    is_dir=stat.S_ISDIR(stats.st_mode),
                    expected=(stats.st_dev, stats.st_ino, stats.st_ctime_ns),
                )
            except OSError as e:
                logger.warning("Cannot remove cache object %s: %s", path, e)
                continue
            removed["objects"] += 1
            removed["bytes"] += doomed["bytes"]
    finally:
        if prefix_fd is not None:
            os.close(prefix_fd)
    return removed


def _remove_at(
    dir_fd: int, name: str, *, is_dir: bool, expected: tuple[int, int, int] | None = None
) -> None:
    """Remove the object ``name`` in the prefix directory open as ``dir_fd``, never through a
    path. A tree is emptied through descriptors, depth first and without recursion: each of its
    directories is opened from the one above it (``_open_directory_at``), and every unlink and
    rmdir is relative to an open directory, so a link or another directory put at a name while
    the removal runs receives nothing. Each directory's identity is recorded while its parent is
    scanned, and the root's is from ``execute_plan``'s non-following ``stat``: a replacement
    directory is not removed. A tree's read-only directories are made writable as the removal
    opens them; a file never is (it may be a hardlink a workspace shares), only the directory
    that holds it.

    Raises:
        OSError: an entry cannot be removed, or a directory of the tree is no longer the one
            the removal reached, and nothing in it was removed.
    """
    if not is_dir:
        _remove_entry(os.unlink, dir_fd, name)
        return
    stack = [_open_for_removal(dir_fd, name, expected=expected)]
    try:
        while stack:
            parent_fd, entry, fd, children = stack[-1]
            if children:
                child, identity = children.pop()
                if identity is not None:
                    stack.append(_open_for_removal(fd, child, expected=identity))
                else:
                    _remove_entry(os.unlink, fd, child)
                continue
            stack.pop()
            os.close(fd)
            _remove_entry(os.rmdir, parent_fd, entry)
    finally:
        for _parent_fd, _entry, fd, _children in stack:
            os.close(fd)


def _open_for_removal(
    parent_fd: int, name: str, *, expected: tuple[int, int, int] | None = None
) -> tuple[int, str, int, list[tuple[str, tuple[int, int, int] | None]]]:
    """Open the directory ``name`` in the directory open as ``parent_fd`` to empty it: its
    parent's descriptor, its name, its own descriptor (made writable) and its entries. A child
    directory keeps the device, inode and change time seen while this parent was scanned; a file
    or link has None and is unlinked without following it.

    Raises:
        OSError: no directory of its own is there (``_open_directory_at``).
    """
    fd = _open_directory_at(parent_fd, name, expected=expected)
    if fd is None:
        raise OSError(
            errno.ENOTDIR,
            "no longer the directory being removed (a link or another directory took its "
            + "name), so nothing in it was removed",
            name,
        )
    try:
        _writable_dir(fd)
        with os.scandir(fd) as entries:
            children = list[tuple[str, tuple[int, int, int] | None]]()
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                identity = (
                    (info.st_dev, info.st_ino, info.st_ctime_ns)
                    if stat.S_ISDIR(info.st_mode)
                    else None
                )
                children.append((entry.name, identity))
    except BaseException:
        os.close(fd)
        raise
    return parent_fd, name, fd, children


def _remove_entry(remove: Callable[..., object], dir_fd: int, name: str) -> None:
    """Remove the entry ``name`` of the directory open as ``dir_fd`` with ``remove``
    (``os.unlink`` or ``os.rmdir``), making that directory writable when it refuses for want
    of permission."""
    try:
        remove(name, dir_fd=dir_fd)
    except PermissionError:
        _writable_dir(dir_fd)
        remove(name, dir_fd=dir_fd)


def _writable_dir(dir_fd: int) -> None:
    """Give the owner write permission on the directory open as ``dir_fd`` when it lacks it;
    a failure leaves it as it is."""
    with contextlib.suppress(OSError):
        mode = stat.S_IMODE(os.fstat(dir_fd).st_mode)
        if not mode & stat.S_IWUSR:
            os.fchmod(dir_fd, mode | stat.S_IWUSR)


def result_document(
    cache_dir: pathlib.Path,
    plan: Plan,
    removed: ObjectCount,
    scopes: Iterable[ProjectScope],
    *,
    dry_run: bool,
) -> GcResult:
    """The JSON document of one collection.

    ``kept`` counts every object still in the cache, so an object a second pass spared, or
    one the filesystem refused to remove, is kept with no reason in ``kept.by``.
    """
    kept = Kept(
        objects=plan["scanned"]["objects"] - removed["objects"],
        bytes=plan["scanned"]["bytes"] - removed["bytes"],
        by=plan["kept"]["by"],
    )
    return GcResult(
        cache=str(cache_dir),
        scanned=plan["scanned"],
        kept=kept,
        removed=removed,
        projects=[str(scope["root"]) for scope in scopes],
        dry_run=dry_run,
    )


def format_bytes(count: int) -> str:
    """Format a byte count as the summary prints it: ``33.5 GB``, ``912.0 MB``, ``17 B``."""
    if count < 1024:
        return f"{count} B"
    value = count / 1024
    for unit in ("KB", "MB", "GB", "TB"):
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} PB"
