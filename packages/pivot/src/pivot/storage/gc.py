"""Collect the cache objects no live record names.

Liveness is read from disk, never inferred. Five classes of record name an object: a stage
lock (``outs`` and ``deps``), a ``.pvt`` file, a run-cache row, a workspace link into the
cache, and a ref under ``<cache>/files/refs``. Each contributes a file's hash, or a
directory's tree hash and every hash in its manifest.

A cache outside the project root can hold another project's outputs, whose locks this project
cannot read. Pivot therefore keeps a register of the projects that use a cache, at
``<cache>/projects/``, written by ``config.get_cache_dir()``, and refuses to collect an
external cache that records nothing.

Two rules cover the window between publishing an object and writing the record that names it
(``storage/AGENTS.md``): every object whose ``st_ctime`` is inside the grace period is kept,
and so is every object some directory entry outside the cache still hardlinks.
"""

from __future__ import annotations

import contextlib
import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import logging
import os
import pathlib
import re
import socket
import time
from typing import TYPE_CHECKING, Literal, TypedDict, cast

import yaml

from pivot import exceptions, run_history, yaml_config
from pivot.storage import cache, lock, state, track
from pivot.types import is_dir_hash

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Iterator

    from pivot.types import HashInfo

logger = logging.getLogger(__name__)

# Cache objects live under this directory of the cache; the register is its sibling, so no
# cache scan sees it (``remote/sync.py`` descends two-character prefix directories only).
OBJECTS_DIR = "files"
REGISTER_DIR = "projects"
# Held exclusively for a whole collection, so two collections never run on one cache.
GC_LOCK = "gc.lock"
DEFAULT_GRACE_HOURS = 24.0
# A process rewrites its register entry at most this often.
_REGISTER_REFRESH_SECONDS = 3600.0
# A cache key as it appears in the text of a record pivot cannot parse.
_KEY_RE = re.compile(r"(?<![0-9a-zA-Z])[0-9a-f]{16}(?![0-9a-zA-Z])")
# (cache, project root) pairs this process has already registered.
_registered = set[tuple[str, str]]()


class ProjectEntry(TypedDict):
    """One project's entry in a cache's register."""

    root: str
    state_dir: str
    host: str
    pivot: str
    last_write: str


class ProjectScope(TypedDict):
    """One project whose records keep objects in this cache alive."""

    root: pathlib.Path
    state_dir: pathlib.Path
    host: str
    last_write: str


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
    # The records (lock files) pivot could not parse. A non-empty list means a record of an
    # unknown shape could name any object, so a collection keeps every object and removes none.
    unparseable: list[str]


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


class ProjectUnreadableError(exceptions.CacheError):
    """A project registered with this cache cannot be read, so its records are unknown."""

    scope: ProjectScope
    reason: str

    def __init__(self, scope: ProjectScope, reason: str) -> None:
        super().__init__(f"{scope['root']}: {reason}")
        self.scope = scope
        self.reason = reason


# -----------------------------------------------------------------------------
# The register of the projects that use a cache
# -----------------------------------------------------------------------------


def register_path(cache_dir: pathlib.Path, project_root: pathlib.Path) -> pathlib.Path:
    """Return the register entry a project owns in a cache."""
    digest = hashlib.sha256(str(project_root).encode()).hexdigest()[:16]
    return cache_dir / REGISTER_DIR / f"{digest}.json"


def register_project(
    cache_dir: pathlib.Path, project_root: pathlib.Path, state_dir: pathlib.Path
) -> None:
    """Record this project as a user of ``cache_dir``, once per process and at most hourly.

    A collection run in another project reads these entries and keeps every object they name.
    Nothing here may fail a command: a cache this user cannot write stays usable, and a cache
    that does not exist yet is not created to hold a register entry.
    """
    memo = (str(cache_dir), str(project_root))
    if memo in _registered:
        return
    if not cache_dir.is_dir():
        return
    path = register_path(cache_dir, project_root)
    try:
        if time.time() - path.stat().st_mtime < _REGISTER_REFRESH_SECONDS:
            _registered.add(memo)
            return
    except OSError:
        pass

    try:
        version = importlib.metadata.version("pivot")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - always installed
        version = "unknown"
    entry = ProjectEntry(
        root=str(project_root),
        state_dir=str(state_dir),
        host=socket.gethostname(),
        pivot=version,
        last_write=datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    def write_entry(fd: int) -> None:
        with os.fdopen(fd, "w") as f:
            json.dump(entry, f)
            f.write("\n")

    try:
        cache.atomic_write_file(path, write_entry)
    except OSError as e:
        logger.debug("Cannot record this project in %s: %s", path, e)
        return
    _registered.add(memo)


def _scope_from_entry(path: pathlib.Path) -> ProjectScope:
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
    return ProjectScope(
        root=pathlib.Path(root),
        # discovery.py permits no other state dir, so an entry an older pivot wrote has one.
        state_dir=pathlib.Path(state_dir)
        if isinstance(state_dir, str)
        else pathlib.Path(root) / ".pivot",
        host=host if isinstance(host, str) else "unknown host",
        last_write=last_write if isinstance(last_write, str) else "an unknown time",
    )


def read_register(
    cache_dir: pathlib.Path,
) -> tuple[list[ProjectScope], list[pathlib.Path], list[tuple[str, str]]]:
    """Return the register's live projects, the roots that are gone, and what cannot be read."""
    register = cache_dir / REGISTER_DIR
    live = list[ProjectScope]()
    gone = list[pathlib.Path]()
    unreadable = list[tuple[str, str]]()
    if not register.is_dir():
        return live, gone, unreadable
    for path in sorted(register.glob("*.json")):
        try:
            scope = _scope_from_entry(path)
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as e:
            unreadable.append((str(path), str(e)))
            continue
        if scope["root"].is_dir():
            live.append(scope)
        else:
            gone.append(scope["root"])
    return live, gone, unreadable


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
    cache_dir: pathlib.Path, recorded: int, unreadable: list[tuple[str, str]]
) -> str:
    """What a collection says when a project the cache records cannot be read."""
    lines = [
        f"Refusing to collect {cache_dir}: it records {recorded} projects and "
        + f"{len(unreadable)} cannot be read:"
    ]
    lines.extend(f"  {label}: {reason}" for label, reason in unreadable)
    lines.append(
        "That project may hold outputs this run would remove. Restore access to "
        + ("it" if len(unreadable) == 1 else "them")
        + ", or pass\n--projects for every project that is still real; an entry whose directory "
        + "is gone is dropped\nautomatically. Nothing was removed."
    )
    return "\n".join(lines)


def dropped_note(gone: list[pathlib.Path]) -> str:
    """The note a collection prints when it drops register entries whose roots are gone."""
    return (
        f"Dropped {len(gone)} projects from the cache register; their roots are gone: "
        + ", ".join(str(root) for root in gone)
        + "."
    )


def _drop_gone_entries(cache_dir: pathlib.Path, gone: list[pathlib.Path]) -> None:
    """Delete the register entry of every project whose root is gone, so the drop is permanent.

    Best-effort: a cache this user cannot write keeps the stale entries, and the next run
    reports them again, but nothing is removed on their account."""
    for root in gone:
        with contextlib.suppress(OSError):
            register_path(cache_dir, root).unlink(missing_ok=True)


def resolve_scopes(
    cache_dir: pathlib.Path,
    project_root: pathlib.Path,
    state_dir: pathlib.Path,
    extra: Iterable[pathlib.Path],
) -> tuple[list[ProjectScope], list[pathlib.Path]]:
    """Return the projects whose records this collection must read, and the roots it dropped.

    The project the collection runs in is always in scope. A cache inside its root has that
    one user by construction; an external cache is collected only when the register names a
    project, or ``--projects`` does, since another project's outputs can be in it. A path
    ``--projects`` names is registered, so the next run needs no flag.

    Raises:
        exceptions.CacheCollectionRefusedError: a ``--projects`` path is not a pivot project
            this cache can read, or the cache is external and records no usable project, or an
            entry cannot be read.
    """
    here = ProjectScope(
        root=project_root, state_dir=state_dir, host=socket.gethostname(), last_write="now"
    )
    named = list[ProjectScope]()
    for path in extra:
        root = path.resolve()
        scope_state_dir = root / ".pivot"
        # A --projects path must be a real pivot project: a readable .pivot state directory.
        # A directory that is not one (the cache itself, a parent, a deleted-and-recreated
        # worktree, a typo) names no records, so accepting it would collect this cache on one
        # project's records alone and remove another project's live objects.
        if not (scope_state_dir.is_dir() and os.access(scope_state_dir, os.R_OK | os.X_OK)):
            raise exceptions.CacheCollectionRefusedError(refusal_not_a_project(cache_dir, root))
        named.append(
            ProjectScope(
                root=root,
                state_dir=scope_state_dir,
                host=socket.gethostname(),
                last_write="now",
            )
        )
        register_project(cache_dir, root, scope_state_dir)

    if cache_dir.resolve().is_relative_to(project_root.resolve()):
        return _dedupe_scopes([here, *named]), []

    live, gone, unreadable = read_register(cache_dir)
    if unreadable:
        raise exceptions.CacheCollectionRefusedError(
            refusal_unreadable(cache_dir, len(live) + len(unreadable), unreadable)
        )
    if not live and not named:
        raise exceptions.CacheCollectionRefusedError(refusal_external(cache_dir, project_root))
    _drop_gone_entries(cache_dir, gone)
    return _dedupe_scopes([here, *live, *named]), gone


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


def _hash_info_keys(hash_info: HashInfo) -> Iterator[str]:
    """A record's own key, and every member key when it records a directory."""
    yield hash_info["hash"]
    if is_dir_hash(hash_info):
        for entry in hash_info["manifest"]:
            yield entry["hash"]


def _text_keys(path: pathlib.Path) -> set[str]:
    """Every cache key in the text of a record pivot cannot parse.

    A record whose shape pivot does not recognise is no evidence that its objects are dead,
    so every key it could name is taken as a root. Over-approximating keeps objects.
    """
    logger.warning("Cannot read %s as a pivot record; keeping every cache object it names", path)
    return set(_KEY_RE.findall(path.read_text(errors="replace")))


def _lock_keys(path: pathlib.Path) -> set[str]:
    """Every key a stage lock names: each ``outs`` and ``deps`` hash and manifest member.

    ``open`` raising ``OSError`` refuses the run (the lock cannot be read, so what it names is
    unknown); ``yaml.load`` raising ``yaml.YAMLError`` (a committed lock left with git conflict
    markers, a truncated one) is caught by ``collect_roots``, which then keeps every object
    rather than let a lock it cannot parse licence a delete."""
    with open(path) as f:
        data: object = yaml.load(f, Loader=yaml_config.Loader)
    if not lock.is_lock_data(data):
        return _text_keys(path)
    entries = [*data["deps"], *data["outs"]]
    return {key for entry in entries for key in _hash_info_keys(lock.entry_hash_info(entry))}


def _pvt_keys(path: pathlib.Path) -> set[str]:
    """Every key a ``.pvt`` names: its hash and, for a directory, every manifest member."""
    data = track.read_pvt_file(path)
    if data is None:
        return _text_keys(path)
    return set(_hash_info_keys(track.pvt_to_hash_info(data)))


def _run_cache_keys(
    state_dir: pathlib.Path, valid_run_ids: set[str] | None = None
) -> tuple[set[str], int]:
    """Every key the run-cache rows of one project's database name, and the number of rows.

    ``valid_run_ids`` restricts the rows to those a run-history prune would keep — its sentinel
    rows (``__``-prefixed) and the rows of runs within the retention — so the plan counts an
    object whose only record is a row the confirmed run will drop (``StateDB.prune_runs``)."""
    if not (state_dir / "state.lmdb").exists():
        return set(), 0
    with state.StateDB(state_dir, readonly=True) as db:
        entries = db.list_run_cache_entries()
    if valid_run_ids is not None:
        entries = [
            entry
            for entry in entries
            if entry["run_id"].startswith("__") or entry["run_id"] in valid_run_ids
        ]
    keys = {
        key
        for entry in entries
        for output in entry["output_hashes"]
        for key in _hash_info_keys(run_history.entry_to_output_hash(output))
    }
    return keys, len(entries)


def _walk_project(
    root: pathlib.Path, object_root: pathlib.Path, skip: pathlib.Path | None
) -> tuple[list[pathlib.Path], set[str]]:
    """Every ``.pvt`` file under ``root`` and every key a symlink there resolves to.

    A file-by-file symlink checkout links each member to its object; a whole-directory
    checkout links the directory itself to its tree object. Both are read here. ``skip`` is
    the cache itself, listed object by object rather than walked.
    """
    pvt_paths = list[pathlib.Path]()
    keys = set[str]()
    stack = [root]
    while stack:
        with os.scandir(stack.pop()) as entries:
            for entry in entries:
                path = pathlib.Path(entry.path)
                if entry.is_symlink():
                    key = cache.get_symlink_cache_key(path, object_root)
                    if key is not None and cache.is_valid_hash(key):
                        keys.add(key)
                    if entry.name.endswith(".pvt") and entry.is_file():
                        pvt_paths.append(path)
                elif entry.is_dir(follow_symlinks=False):
                    if path != skip:
                        stack.append(path)
                elif entry.name.endswith(".pvt"):
                    pvt_paths.append(path)
    return pvt_paths, keys


def collect_roots(
    scopes: Iterable[ProjectScope],
    cache_dir: pathlib.Path,
    *,
    prune_state_dir: pathlib.Path | None = None,
    valid_run_ids: set[str] | None = None,
) -> Roots:
    """Read every record of every project in scope and return the keys they name.

    ``prune_state_dir`` and ``valid_run_ids`` restrict that project's run-cache rows to the ones
    a run-history prune keeps, so the plan reflects the confirmed run's prune (B3/B4).

    Raises:
        ProjectUnreadableError: a project in scope cannot be read, so what it names is
            unknown and nothing may be removed.
    """
    object_root = cache_dir / OBJECTS_DIR
    roots = Roots(
        named=set[str](),
        run_cache=set[str](),
        linked=set[str](),
        counts=RootCounts(locks=0, pvt_files=0, rows=0, links=0),
        unparseable=list[str](),
    )
    for scope in scopes:
        try:
            stages_dir = lock.get_stages_dir(scope["state_dir"])
            if stages_dir.is_dir():
                for lock_path in sorted(stages_dir.rglob("*.lock")):
                    roots["counts"]["locks"] += 1
                    try:
                        roots["named"] |= _lock_keys(lock_path)
                    except yaml.YAMLError:
                        logger.warning(
                            "Cannot parse %s as a stage lock; keeping every cache object because "
                            "a lock pivot cannot read may name any of them",
                            lock_path,
                        )
                        roots["unparseable"].append(str(lock_path))
            pvt_paths, links = _walk_project(scope["root"], object_root, cache_dir)
            for pvt_path in pvt_paths:
                roots["counts"]["pvt_files"] += 1
                roots["named"] |= _pvt_keys(pvt_path)
            roots["counts"]["links"] += len(links)
            roots["linked"] |= links
            scope_valid = valid_run_ids if scope["state_dir"] == prune_state_dir else None
            rows_keys, rows = _run_cache_keys(scope["state_dir"], scope_valid)
            roots["counts"]["rows"] += rows
            roots["run_cache"] |= rows_keys
        except OSError as e:
            raise ProjectUnreadableError(scope, e.strerror or str(e)) from e

    refs = object_root / "refs"
    if refs.is_dir():
        _, ref_keys = _walk_project(refs, object_root, None)
        roots["counts"]["links"] += len(ref_keys)
        roots["linked"] |= ref_keys
    return roots


# -----------------------------------------------------------------------------
# Locking, scanning and removing
# -----------------------------------------------------------------------------


@contextlib.contextmanager
def collection_lock(cache_dir: pathlib.Path) -> Generator[bool]:
    """Hold the cache's ``gc.lock`` exclusively, yielding whether the lock was taken.

    Raises:
        exceptions.CacheCollectionRefusedError: another collection holds it, or the cache
            cannot be written.
    """
    path = cache_dir / GC_LOCK
    try:
        fd = cache.open_cache_lock(path)
    except OSError as e:
        raise exceptions.CacheCollectionRefusedError(
            f"Refusing to collect {cache_dir}: cannot open {path} ({e.strerror or e}). "
            + "Collecting a cache needs write access to it. Nothing was removed."
        ) from e
    try:
        locked = True
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise exceptions.CacheCollectionRefusedError(
                f"Refusing to collect {cache_dir}: another pivot gc holds {path}. "
                + "Wait for it to finish. Nothing was removed."
            ) from None
        except OSError as e:
            if e.errno not in cache.NO_FLOCK_ERRNO:
                raise
            locked = False
        yield locked
    finally:
        os.close(fd)


@contextlib.contextmanager
def _prefix_locked(prefix_dir: pathlib.Path) -> Generator[None]:
    """Hold one prefix directory's lock exclusively, excluding every staging publisher and
    the temporary sweep while this directory is listed and collected."""
    fd = cache.open_cache_lock(prefix_dir / cache.PREFIX_LOCK)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError as e:
            if e.errno not in cache.NO_FLOCK_ERRNO:
                raise
        yield
    finally:
        os.close(fd)


def _prefix_dirs(object_root: pathlib.Path) -> list[pathlib.Path]:
    """Every two-character prefix directory of the object root, in name order."""
    if not object_root.is_dir():
        return []
    with os.scandir(object_root) as entries:
        return sorted(
            pathlib.Path(entry.path)
            for entry in entries
            if entry.is_dir(follow_symlinks=False) and _is_prefix_name(entry.name)
        )


def _is_prefix_name(name: str) -> bool:
    """Whether a directory of the object root holds objects: the first two hex characters of
    a cache key (``cache.get_cache_path``)."""
    return len(name) == 2 and all(c in "0123456789abcdef" for c in name)


def _tree_bytes(path: pathlib.Path) -> int:
    """The bytes a directory object holds: its members are copies, never links."""
    total = 0
    stack = [path]
    while stack:
        try:
            with os.scandir(stack.pop()) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(pathlib.Path(entry.path))
                        continue
                    with contextlib.suppress(OSError):
                        total += entry.stat(follow_symlinks=False).st_size
        except OSError:
            continue
    return total


def build_plan(
    cache_dir: pathlib.Path,
    roots: Roots,
    *,
    grace_hours: float,
    remove_linked: bool,
    now: float | None = None,
) -> Plan:
    """List every object in the cache and decide, under each prefix's lock, what may go."""
    object_root = cache_dir / OBJECTS_DIR
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
    for prefix_dir in _prefix_dirs(object_root):
        with _prefix_locked(prefix_dir), os.scandir(prefix_dir) as entries:
            for entry in sorted(entries, key=lambda e: e.name):
                key = prefix_dir.name + entry.name
                # Skips every name that is not an object: the prefix lock, a `.pivot_tmp_*`
                # entry a publisher owns, and a tree's `.verified` marker and `.lock` file.
                if not cache.is_valid_hash(key):
                    continue
                try:
                    stats = entry.stat(follow_symlinks=False)
                    is_dir = entry.is_dir(follow_symlinks=False)
                except OSError:
                    continue
                size = _tree_bytes(pathlib.Path(entry.path)) if is_dir else stats.st_size
                plan["scanned"]["objects"] += 1
                plan["scanned"]["bytes"] += size
                # A record pivot could not parse may name any object, so a collection that saw
                # one keeps every object and removes none (``collect_roots``). Otherwise the
                # object stays only when a record, the grace or the hardlink rule holds it.
                reason = (
                    "locks"
                    if roots["unparseable"]
                    else _keep_reason(
                        key,
                        stats,
                        roots,
                        is_dir=is_dir,
                        cutoff=cutoff,
                        remove_linked=remove_linked,
                    )
                )
                if reason is not None:
                    plan["kept"]["objects"] += 1
                    plan["kept"]["bytes"] += size
                    plan["kept"]["by"][reason] += 1
                    continue
                linked = not is_dir and stats.st_nlink > 1
                plan["doomed"].append(
                    Doomed(path=pathlib.Path(entry.path), is_dir=is_dir, bytes=size, linked=linked)
                )
                plan["doomed_bytes"] += size
                if linked:
                    plan["linked"]["objects"] += 1
                    plan["linked"]["bytes"] += size
    return plan


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
    scopes: Iterable[ProjectScope],
    cache_dir: pathlib.Path,
    grace_hours: float,
    remove_linked: bool,
    now: float | None = None,
) -> ObjectCount:
    """Remove what the plan condemns, under each prefix's lock, and return what went.

    The liveness records are re-read here, after the plan's scan and immediately before the
    deletions, so an object a concurrent run named between the scan and now — a lock, a
    run-cache row, a workspace link — is kept (``_keep_reason`` against the fresh records). Each
    object is also re-checked against the grace and the hardlink rule, because since the scan a
    checkout can have hardlinked it and a reusing run can have refreshed its timestamp
    (``copy_to_cache``). The residual exposure is an object a run both re-uses and names in the
    gap between this re-read and that object's own unlink — one ``lstat`` per object, not the
    whole scan. An unparseable record seen now keeps every object, as the plan does.

    Raises:
        ProjectUnreadableError: a project became unreadable since the plan; nothing is removed.
    """
    fresh_roots = collect_roots(scopes, cache_dir)
    keep_all = bool(fresh_roots["unparseable"])
    cutoff = (time.time() if now is None else now) - grace_hours * 3600
    removed = ObjectCount(objects=0, bytes=0)
    by_prefix = dict[pathlib.Path, list[Doomed]]()
    for doomed in plan["doomed"]:
        by_prefix.setdefault(doomed["path"].parent, []).append(doomed)

    for prefix_dir, doomed_here in sorted(by_prefix.items()):
        with _prefix_locked(prefix_dir):
            for doomed in doomed_here:
                path = doomed["path"]
                try:
                    stats = path.lstat()
                except FileNotFoundError:
                    removed["objects"] += 1
                    removed["bytes"] += doomed["bytes"]
                    continue
                key = prefix_dir.name + path.name
                if keep_all or (
                    _keep_reason(
                        key,
                        stats,
                        fresh_roots,
                        is_dir=doomed["is_dir"],
                        cutoff=cutoff,
                        remove_linked=remove_linked,
                    )
                    is not None
                ):
                    continue
                try:
                    cache.clear_path(path)
                except OSError as e:
                    logger.warning("Cannot remove cache object %s: %s", path, e)
                    continue
                if doomed["is_dir"]:
                    for suffix in (cache.TREE_MARKER_SUFFIX, cache.LEGACY_TREE_MARKER_SUFFIX):
                        with contextlib.suppress(OSError):
                            path.with_name(path.name + suffix).unlink(missing_ok=True)
                removed["objects"] += 1
                removed["bytes"] += doomed["bytes"]
    return removed


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
