"""The writer side of `pivot gc`'s protocol: the naming lock and the register of projects.

Every command that names a cache object holds the cache's naming lock shared for the naming step
(``naming_unit``), and registers its project in ``<cache>/projects/`` first, with evidence a
collection checks before it trusts the entry: the project id, this view's nonce and the mounts
below the project. ``storage/gc.py`` holds the collector, which reads what this module writes.

Writers import only this module, so a worker process never pays for the collector's import.
"""

from __future__ import annotations

import contextlib
import datetime
import enum
import errno
import fcntl
import hashlib
import json
import logging
import os
import pathlib
import re
import select
import socket
import stat
import subprocess
import time
from typing import TYPE_CHECKING, NamedTuple, TypedDict, cast

from pivot import exceptions
from pivot import state_dir as state_dir_mod
from pivot.storage import cache

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

logger = logging.getLogger(__name__)

# The register of the projects that use a cache: one entry per view of a project. Beside the
# object store (``store_layout``: at the cache root, or in the directory that holds a store
# ``files/`` links to), so no cache scan sees it (``remote/sync.py`` descends two-character
# prefix directories only).
REGISTER_DIR = "projects"
# The naming lock excludes gc-from-writers. Writers hold it shared for each commit unit (from
# the object existence check to the durable record that names it); gc holds it exclusive while
# it deletes (re-read register → prune → collect_roots → unlink). The turnstile prevents
# starvation: gc takes it exclusive first, so new writers queue, then gc takes naming exclusive
# to wait for in-flight writers to finish. Lock order: gc.lock → prefix (listing, before any
# naming lock) → turnstile → naming; gc takes no prefix lock while it holds naming exclusive.
NAMING_LOCK = "naming.lock"
NAMING_TURNSTILE = "naming.turnstile"
# A project's identity lives in its state database directory, which the state directory's
# ignore file keeps out of git, so two checkouts of one repository never share it. A copy (`cp
# -a`, a container image, a cloned disk) keeps the id; the view nonce tells it apart.
PROJECT_ID_FILE = "pivot-project-id"
ENTRY_ID_RE = re.compile(r"[0-9a-f]{32}")
# This process's mount namespace. Entries are keyed by it and the hostname, so each view of a
# project (another host, a container that keeps the hostname) keeps its own mount evidence.
_MOUNT_NAMESPACE = pathlib.Path("/proc/self/ns/mnt")
# The machine id, so two machines that share a hostname (cloned VMs) are different views.
_MACHINE_ID = pathlib.Path("/etc/machine-id")
# One nonce file per view and cache in a project's state directory: proof of what that view last
# wrote for that cache's entry. Keyed by the cache too, so registering in one cache never
# replaces the nonce another cache's entry holds (a run with another `cache_dir`). Beside each
# is a record of where that view found the cache (``CACHE_RECORD_SUFFIX``): the list of caches
# the project uses, which a collection reads to check the project's entry in every one of them.
VIEW_NONCE_DIR = "gc-views"
CACHE_RECORD_SUFFIX = ".cache"
# A random id for each cache, in its register, that names the cache in those nonce files. It is
# read from the cache itself, so every view of the cache finds the same id whatever path it
# mounts the cache at.
CACHE_ID_FILE = "cache-id"
# Mount points under the cache root that each view of the cache (host and mount namespace) saw,
# one file per view, read by a collection before it trusts the register: a register or ref
# directory on a mount that is not mounted holds no entries.
CACHE_MOUNTS_DIR = "mounts"
_MOUNTS_TABLE = pathlib.Path("/proc/self/mounts")
# Anything in /proc is on this, the one procfs mount a watch of the mount table must be on.
_PROC_SELF = pathlib.Path("/proc/self")
_MOUNT_ESCAPE = re.compile(rb"\\([0-7]{3})")
_MOUNT_COMMAND_LINE = re.compile(r"^.* on (.*) \([^()]*\)$")
# How long a register entry's ``last_write`` may lag before a registration rewrites the entry
# that otherwise says exactly what this one would. Every process of a view would write the
# same document, so without this each one rewrites it for its timestamp alone.
_ENTRY_REFRESH_SECONDS = 3600.0


class ProjectEntry(TypedDict):
    """One project's entry in a cache's register."""

    root: str
    state_dir: str
    host: str
    last_write: str
    # None when the state directory cannot be written; a collection refuses such an entry.
    project_id: str | None
    # The writer's machine id ("" where there is none) and mount namespace inode (None where the
    # platform has none): with the hostname, the view this entry speaks for.
    machine: str
    mount_ns: int | None
    # The entry's view nonce is not here: it rotates in every naming unit, so it lives in the
    # fixed-size sibling file `<digest>.nonce`, rewritten in place (`_rotate_nonces`).
    # Mount points under the root and state directory; None where the mount table is unknown.
    mounts: list[str] | None


def _mount_namespace() -> int | None:
    """This process's mount namespace inode, or None where there is none to read."""
    try:
        return os.stat(_MOUNT_NAMESPACE).st_ino
    except OSError:
        return None


def _machine_id() -> str:
    """This machine's systemd machine id, or "" where there is none to read."""
    try:
        return _MACHINE_ID.read_text().strip()
    except OSError:
        return ""


def view_key_of(host: str, machine: str, mount_ns: int | None) -> str:
    """The key of one view of a project: a hostname, a machine id and a mount namespace."""
    return hashlib.sha256(f"{host}\0{machine}\0{mount_ns}".encode()).hexdigest()[:16]


def this_view() -> tuple[str, str, int | None]:
    return socket.gethostname(), _machine_id(), _mount_namespace()


# The object store, under the cache root. When it is a link, the register follows the store it
# resolves to (``store_layout``).
OBJECTS_DIR = "files"


class StoreLayout(NamedTuple):
    """Where one cache root keeps its register (``register_root``); where its ``files/``
    resolves to when it is a link (``linked_store``); and whether that linked store is the root
    of a mount of its own, so that no register can follow it (``unshared``; a ``files/`` that is
    itself a mount point is ``unshared_store``'s to find, for a collection)."""

    register_root: pathlib.Path
    linked_store: pathlib.Path | None
    unshared: bool


def store_layout(cache_root: pathlib.Path) -> StoreLayout:
    """Where ``cache_root`` keeps its register, its cache id, its mount records and its naming
    and collection locks: beside its object store, so that every cache root using one object
    store shares one register and one lock, and a collection through any of them reads every
    project that names objects there.

    In a plain layout (``files/`` is a directory, or does not exist yet) that is the cache root
    itself, as before. When ``files/`` is a link, the object store is the directory it resolves
    to, and the register lives in that directory's parent: a cache root whose own ``files/`` is
    that directory keeps its register there too, and so does every other root that links it. A
    store that is the root of a mount of its own (a mount point, or a bind mount) has a parent
    on another filesystem, which another host or mount namespace does not reach through the
    store: its register stays at the cache root, and the layout is ``unshared``, for which a
    collection refuses (``gc.resolve_scopes``). A plain layout costs one ``lstat``; a linked one,
    once this process has resolved it, an ``lstat`` and a ``stat`` of ``files/`` and one poll of
    the mount watch (``_LINKED_LAYOUTS``).
    """
    files = os.fspath(cache_root) + "/" + OBJECTS_DIR
    try:
        link = os.lstat(files)
    except (OSError, ValueError):
        return StoreLayout(cache_root, None, unshared=False)
    if not stat.S_ISLNK(link.st_mode):
        return StoreLayout(cache_root, None, unshared=False)
    try:
        target = os.stat(files)
    except OSError:
        target = None
    link_identity = (link.st_dev, link.st_ino, link.st_ctime_ns)
    store_identity = None if target is None else (target.st_dev, target.st_ino)
    generation = _mount_generation()
    known = _LINKED_LAYOUTS.get(cache_root)
    if (
        known is not None
        and generation is not None
        and known.identity == (link_identity, store_identity, generation)
    ):
        return known.layout
    store = pathlib.Path(os.path.realpath(files))
    if _mount_root(store):
        layout = StoreLayout(cache_root, store, unshared=True)
    else:
        layout = StoreLayout(store.parent, store, unshared=False)
    _LINKED_LAYOUTS[cache_root] = _KnownLayout((link_identity, store_identity, generation), layout)
    return layout


class _KnownLayout(NamedTuple):
    # The link's (dev, ino, ctime), the directory it resolves to (dev, ino), and the mount-change
    # count, when the layout was resolved.
    identity: tuple[tuple[int, int, int], tuple[int, int] | None, int | None]
    layout: StoreLayout


# This process's resolution of each linked layout, kept while the link, the directory it resolves
# to and the mount table are unchanged: resolving one reads the mount table (``_mount_root``).
# A re-pointed link is a new link (another inode); a re-pointed link further along the path
# resolves to another directory; a mount on the store changes the count.
_LINKED_LAYOUTS = dict[pathlib.Path, _KnownLayout]()


def register_root(cache_root: pathlib.Path) -> pathlib.Path:
    """The directory that holds ``cache_root``'s register (``store_layout``)."""
    return store_layout(cache_root).register_root


def unshared_store(cache_root: pathlib.Path) -> pathlib.Path | None:
    """The object store of ``cache_root`` when it is the root of a mount of its own, so that no
    register can follow it to where every user of the store reaches; None otherwise.

    That is a ``files/`` link to a mount root (``store_layout``'s ``unshared``), and a ``files/``
    that is itself a mount point: on another device than the cache root, or a bind mount the
    mount table lists. Either way the register stays at the cache root, on a filesystem other
    hosts and mount namespaces that mount the store need not see, so a collection refuses
    (``gc.resolve_scopes``) rather than read a register that may not show every project of the
    store. Writers keep working: their register is the cache root's, as before. Only a
    collection calls this (it may read the mount table), never a naming step.
    """
    layout = store_layout(cache_root)
    if layout.linked_store is not None:
        return layout.linked_store if layout.unshared else None
    files = pathlib.Path(os.path.realpath(cache_root / OBJECTS_DIR))
    if not files.is_dir():
        return None
    return files if _mount_root(files) else None


def _mount_root(path: pathlib.Path) -> bool:
    """Whether ``path`` is the root of a mount in this view: on another device than its
    parent, or a mount point the mount table lists (a bind mount keeps its device)."""
    try:
        if os.stat(path).st_dev != os.stat(path.parent).st_dev:
            return True
    except OSError:
        return False
    table = read_mount_table()
    points = None if table is None else mount_points(table)
    return points is not None and str(path) in points


def register_path(
    cache_dir: pathlib.Path, project_root: pathlib.Path, project_id: str, view_key: str
) -> pathlib.Path:
    """Return the register entry one view of a project owns in a cache (in the register beside
    the cache's object store, ``store_layout``).

    The key holds the root, the project id and the view (hostname, machine id and mount
    namespace). Each host and each mount view of a project writes its own entry, so no writer
    replaces another's mount evidence, and host-local projects at one path keep separate entries.
    """
    return _entry_path(register_root(cache_dir), project_root, project_id, view_key)


def _entry_path(
    root: pathlib.Path, project_root: pathlib.Path, project_id: str, view_key: str
) -> pathlib.Path:
    """``register_path`` in the register under ``root``, a ``register_root``."""
    key = f"{project_root}\0{project_id}\0{view_key}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return root / REGISTER_DIR / f"{digest}.json"


def project_register_path(
    cache_dir: pathlib.Path, project_root: pathlib.Path, state_dir: pathlib.Path
) -> pathlib.Path | None:
    """Return this view's entry for the project id in ``state_dir``, or None without an id."""
    try:
        project_id = read_project_id(state_dir)
    except (OSError, ValueError):
        return None
    return register_path(cache_dir, project_root, project_id, view_key_of(*this_view()))


def _view_nonce_path(state_dir: pathlib.Path, view_key: str, cache_id: str) -> pathlib.Path:
    """Where one view's latest registration nonce for one cache lives in a project's state
    directory (inside ``state.lmdb``, which the state directory's ignore file keeps out of git)."""
    return state_dir / "state.lmdb" / VIEW_NONCE_DIR / f"{view_key}-{cache_id}"


def _cache_record_path(state_dir: pathlib.Path, view_key: str, cache_id: str) -> pathlib.Path:
    """Where a project's state directory records where one view found one cache it uses."""
    nonce = _view_nonce_path(state_dir, view_key, cache_id)
    return nonce.with_name(nonce.name + CACHE_RECORD_SUFFIX)


class CacheRecord(NamedTuple):
    """One cache a project's state directory lists: the view that uses it, the cache's id, where
    that view found it (None when the record cannot be read), and since when that view's naming
    steps could not reach it (None while they do)."""

    record: pathlib.Path
    view_key: str
    cache_id: str
    cache_dir: pathlib.Path | None
    unconfirmed_since: str | None


def cache_records(state_dir: pathlib.Path, view_key: str | None = None) -> list[CacheRecord]:
    """Every cache the project in ``state_dir`` lists, for one view or (None) for every view.

    Raises OSError when the list cannot be read; a missing list is empty.
    """
    directory = state_dir / "state.lmdb" / VIEW_NONCE_DIR
    try:
        with os.scandir(directory) as entries:
            names = sorted(entry.name for entry in entries)
    except FileNotFoundError:
        return []
    records = list[CacheRecord]()
    for name in names:
        stem = name.removesuffix(CACHE_RECORD_SUFFIX)
        if stem == name:
            continue
        record_view, _, cache_id = stem.rpartition("-")
        if not ENTRY_ID_RE.fullmatch(cache_id) or (
            view_key is not None and record_view != view_key
        ):
            continue
        path = directory / name
        document = entry_from_file(path) or {}
        cache_dir = document.get("cache")
        since = document.get("unconfirmed_since")
        records.append(
            CacheRecord(
                path,
                record_view,
                cache_id,
                pathlib.Path(cache_dir) if isinstance(cache_dir, str) and cache_dir else None,
                since if isinstance(since, str) else None,
            )
        )
    return records


def _record_cache(
    state_dir: pathlib.Path, view_key: str, cache_id: str, cache_dir: pathlib.Path
) -> None:
    """List ``cache_dir`` among the caches this view of the project uses, as reached now (no
    ``unconfirmed_since``), unless it is listed so already. Raises OSError."""
    path = _cache_record_path(state_dir, view_key, cache_id)
    desired: dict[str, object] = {"cache": os.path.abspath(cache_dir)}
    if entry_from_file(path) != desired:
        _replace_readable(str(path), (json.dumps(desired, sort_keys=True) + "\n").encode())


def _mark_unconfirmed(record: CacheRecord) -> None:
    """Record in this view's record of a listed cache since when its naming steps could not
    reach it (``unconfirmed_since``), once: the first step that reaches it again clears it
    (``_record_cache``)."""
    if record.unconfirmed_since is not None:
        return
    document: dict[str, object] = {
        "unconfirmed_since": _entry_timestamp(time.time()),
        **({} if record.cache_dir is None else {"cache": str(record.cache_dir)}),
    }
    try:
        _replace_readable(
            str(record.record), (json.dumps(document, sort_keys=True) + "\n").encode()
        )
    except OSError as e:
        logger.debug("Cannot record that %s is unconfirmed: %s", record.record, e)


def read_nonce_pair(
    state_dir: pathlib.Path, view_key: str, cache_id: str, entry_nonce: pathlib.Path
) -> tuple[str, str]:
    """Read one view's nonce for one cache in the state directory and the nonce its entry in
    that cache holds, under the shared lock writers take across both writes
    (``_rotate_nonces``), so a rotation in progress is never read half done. Where the
    filesystem refuses the lock both are read unlocked. Raises OSError or ValueError."""
    state_nonce = _view_nonce_path(state_dir, view_key, cache_id)
    fd = os.open(state_nonce, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        locked = True
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
        except OSError:
            locked = False
        try:
            return read_nonce_file(state_nonce), read_nonce_file(entry_nonce)
        finally:
            if locked:
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _open_view_nonce(state_dir: pathlib.Path, view_key: str, cache_id: str) -> int:
    """Open (creating) this view's nonce file for one cache in the state directory. Raises
    OSError or ValueError (a NUL byte in an unusable configured path)."""
    path = _view_nonce_path(state_dir, view_key, cache_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o644)
    os.fchmod(fd, 0o644)  # other users' collections must read it, whatever the umask
    return fd


def _write_entry(path: str, document: ProjectEntry) -> None:
    """Replace ``path`` with ``document`` atomically, readable by every user of the cache."""
    _replace_readable(path, (json.dumps(document, sort_keys=True) + "\n").encode())


def _replace_readable(path: str, data: bytes) -> None:
    """Replace ``path`` with ``data`` atomically, readable by every user (another user's
    collection reads register entries and a project's list of caches)."""
    head, tail = os.path.split(path)
    staged = os.path.join(head, f".{tail}.{os.getpid()}.tmp")
    fd = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_CLOEXEC, 0o644)
    try:
        try:
            os.fchmod(fd, 0o644)
            if os.write(fd, data) != len(data):
                raise OSError(errno.EIO, "short write", str(staged))
        finally:
            os.close(fd)
        os.replace(staged, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(staged)
        raise


def _open_nonce_file(path: pathlib.Path) -> int:
    """Open (creating) a nonce file other users can read. Raises OSError or ValueError."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o644)
    os.fchmod(fd, 0o644)  # other users' collections must read it, whatever the umask
    return fd


def entry_nonce_path(entry: pathlib.Path) -> pathlib.Path:
    """The sibling file that holds a register entry's current view nonce."""
    return entry.with_suffix(".nonce")


def _rotate_nonces(state_fd: int, register_fd: int, *, truncate: bool = False) -> None:
    """Write one fresh nonce to this view's nonce file in the state directory, then to the
    entry's nonce file in the register.

    A collection in any view accepts the entry only if it reads that nonce in its own view of the
    state directory, which proves it reads the bytes this view last wrote, not a clone or an
    overlay copy that shares the directory's inode number. Both are fixed-size files rewritten in
    place: a collection reads them only under the exclusive naming lock, and a writer writes them
    only under the shared one. Processes of one view hold the state nonce file's lock across both
    writes, so the register holds the nonce written last. Where the filesystem refuses the lock
    they write unlocked: an interleaving then leaves a mismatch, which only makes a collection
    refuse. Raises OSError.

    Every nonce is the same length, so only a registration, which may find a longer file at
    either path, truncates (``truncate``); the rotations after it overwrite the whole file.
    A nonce file holding anything else is refused by its reader, so a collection never takes a
    rotation for a stale nonce.
    """
    data = f"{os.urandom(16).hex()}\n".encode()
    locked = True
    try:
        fcntl.flock(state_fd, fcntl.LOCK_EX)
    except OSError:
        locked = False
    try:
        for fd in (state_fd, register_fd):
            if os.pwrite(fd, data, 0) != len(data):
                raise OSError(errno.EIO, "short write of a view nonce")
            if truncate:
                os.ftruncate(fd, len(data))
    finally:
        if locked:
            fcntl.flock(state_fd, fcntl.LOCK_UN)


def read_nonce_file(path: pathlib.Path) -> str:
    """Read a nonce file without following a symlink. Raises OSError or ValueError."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(fd) as file:
        nonce = file.read().strip()
    if not ENTRY_ID_RE.fullmatch(nonce):
        raise ValueError(f"{path} does not hold a pivot view nonce")
    return nonce


class ListedCacheState(enum.Enum):
    """Why a listed cache's entry for the project is not proven current (``listed_cache_problem``)."""

    # The record, the cache at its path, or another cache there: this view cannot reach it.
    UNREACHABLE = "unreachable"
    # The cache holds no entry, or no nonce, for that view of the project.
    NO_ENTRY = "no entry"
    # The entry's nonce is not the one the state directory holds: one of them is a copy's.
    STALE = "stale"


class ListedCacheProblem(NamedTuple):
    state: ListedCacheState
    detail: str


def listed_cache_problem(
    record: CacheRecord, project_root: pathlib.Path, state_dir: pathlib.Path, project_id: str
) -> ListedCacheProblem | None:
    """What stops this process from proving that one cache the project lists holds, for the
    record's view, the entry whose nonce the state directory holds; None when it does. A
    collection refuses on any problem (``gc._other_caches_reason``); `pivot forget-cache`
    refuses to remove a cache that has none."""
    if record.cache_dir is None:
        return ListedCacheProblem(
            ListedCacheState.UNREACHABLE,
            f"the record {record.record} of where that cache is cannot be read",
        )
    try:
        found = read_cache_id(record.cache_dir)
    except (OSError, ValueError) as e:
        return ListedCacheProblem(
            ListedCacheState.UNREACHABLE, f"that cache cannot be read ({_error_text(e)})"
        )
    if found != record.cache_id:
        return ListedCacheProblem(
            ListedCacheState.UNREACHABLE, f"another cache is at that path now (id {found})"
        )
    entry = register_path(record.cache_dir, project_root, project_id, record.view_key)
    try:
        state_nonce, entry_nonce = read_nonce_pair(
            state_dir, record.view_key, record.cache_id, entry_nonce_path(entry)
        )
    except FileNotFoundError as e:
        missing = "its nonce in this state directory" if e.filename is None else e.filename
        return ListedCacheProblem(
            ListedCacheState.NO_ENTRY,
            f"the project's entry there cannot be checked ({missing} is missing)",
        )
    except (OSError, ValueError) as e:
        return ListedCacheProblem(
            ListedCacheState.NO_ENTRY,
            f"the project's entry there cannot be checked ({_error_text(e)})",
        )
    if state_nonce != entry_nonce:
        return ListedCacheProblem(
            ListedCacheState.STALE,
            "the project's entry there holds a different nonce from the one this state "
            + "directory holds for it",
        )
    return None


def _error_text(error: OSError | ValueError) -> str:
    return error.strerror if isinstance(error, OSError) and error.strerror else str(error)


def _cache_id_path(root: pathlib.Path) -> pathlib.Path:
    return root / REGISTER_DIR / CACHE_ID_FILE


def read_cache_id(cache_dir: pathlib.Path) -> str:
    """Read the id of the cache at ``cache_dir`` from its register (``store_layout``) without
    following a symlink. Raises OSError or ValueError."""
    return read_cache_id_at(register_root(cache_dir))


def read_cache_id_at(root: pathlib.Path) -> str:
    """Read the cache id in the register under ``root``, a ``register_root`` (the parent of a
    register entry's directory). Raises OSError or ValueError."""
    path = _cache_id_path(root)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(fd) as file:
        cache_id = file.read().strip()
    if not ENTRY_ID_RE.fullmatch(cache_id):
        raise ValueError(f"{path} does not hold a pivot cache id")
    return cache_id


def _ensure_cache_id(root: pathlib.Path) -> str:
    """Return the id of the cache whose register is under ``root``, creating it once (the first
    writer's link wins). Raises OSError or ValueError."""
    with contextlib.suppress(FileNotFoundError):
        return read_cache_id_at(root)
    register = root / REGISTER_DIR
    register.mkdir(exist_ok=True)
    staged = register / f".{CACHE_ID_FILE}.{os.urandom(8).hex()}"
    try:
        fd = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o644)
        with os.fdopen(fd, "w") as file:
            os.fchmod(file.fileno(), 0o644)  # every user of the cache reads it
            file.write(os.urandom(16).hex() + "\n")
        with contextlib.suppress(FileExistsError):
            os.link(staged, _cache_id_path(root))
    finally:
        staged.unlink(missing_ok=True)
    return read_cache_id_at(root)


def _project_id_path(state_dir: pathlib.Path) -> pathlib.Path:
    return state_dir / "state.lmdb" / PROJECT_ID_FILE


def read_project_id(state_dir: pathlib.Path) -> str:
    """Read the project id without following a symlink. Raises OSError or ValueError."""
    path = _project_id_path(state_dir)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(fd) as file:
        project_id = file.read().strip()
    if not ENTRY_ID_RE.fullmatch(project_id):
        raise ValueError(f"{path} does not hold a pivot project id")
    return project_id


def _ensure_project_id(state_dir: pathlib.Path, project_root: pathlib.Path) -> str:
    """Return this state directory's project id, creating it once. Raises OSError or ValueError.

    A new configured state directory takes the id from the project's `.pivot` state, so changing
    ``core.state_dir`` does not leave the project's current entry under a different identity. A
    copy of a project keeps the id; what tells a copy from the project is the view nonce.
    """
    with contextlib.suppress(FileNotFoundError):
        return read_project_id(state_dir)
    if not state_dir.is_dir():
        raise FileNotFoundError(errno.ENOENT, "state directory does not exist", str(state_dir))
    legacy = project_root / ".pivot"
    try:
        project_id = read_project_id(legacy) if legacy != state_dir else None
    except FileNotFoundError:
        project_id = None
    database_dir = state_dir / "state.lmdb"
    database_dir.mkdir(exist_ok=True)
    staged = database_dir / f".{PROJECT_ID_FILE}.{os.urandom(8).hex()}"
    try:
        staged.write_text((project_id or os.urandom(16).hex()) + "\n")
        with contextlib.suppress(FileExistsError):
            os.link(staged, _project_id_path(state_dir))
    finally:
        staged.unlink(missing_ok=True)
    return read_project_id(state_dir)


def _decode_mount_point(raw: bytes) -> str:
    return os.fsdecode(_MOUNT_ESCAPE.sub(lambda match: bytes([int(match[1], 8)]), raw))


def read_mount_table() -> bytes | None:
    """This mount namespace's mount table, or None where `/proc/self/mounts` cannot be read."""
    try:
        fd = os.open(_MOUNTS_TABLE, os.O_RDONLY | os.O_CLOEXEC)
    except OSError:
        return None
    chunks = list[bytes]()
    try:
        while chunk := os.read(fd, 65536):
            chunks.append(chunk)
    except OSError:
        return None
    finally:
        os.close(fd)
    return b"".join(chunks)


class _MountWatch(NamedTuple):
    """An open mount table and the poll that reports every change to it since it was opened."""

    fd: int
    poller: select.poll
    pid: int


def _watch_mount_table() -> _MountWatch | None:
    """Open `/proc/self/mounts` and arm a watch on this mount namespace's mount table, or None
    where there is none to watch.

    The kernel records the namespace's mount event counter in the open file and reports
    ``POLLPRI`` (with ``POLLERR``) on the next poll after it changes, for every mount, unmount
    and propagated mount in this namespace (``proc(5)``, "this file is pollable"); the poll
    then re-arms. So one poll of this descriptor answers "has the mount table changed since I
    read it", for the price of one syscall, where reading the table costs the kernel a pass
    over every mount.

    The watch is only that file's: a descriptor on anything but the procfs that holds
    ``/proc/self`` reports no such event, so one that is not on it is refused and the caller
    reads the table's bytes instead.
    """
    try:
        fd = os.open(_MOUNTS_TABLE, os.O_RDONLY | os.O_CLOEXEC)
    except OSError:
        return None
    try:
        if os.fstat(fd).st_dev != os.stat(_PROC_SELF).st_dev:
            os.close(fd)
            return None
    except OSError:
        os.close(fd)
        return None
    poller = select.poll()
    poller.register(fd, select.POLLPRI)
    return _MountWatch(fd, poller, os.getpid())


class _SharedMountWatch:
    """The process's one mount watch, and how many changes of the mount table it has reported.
    A poll re-arms the watch, so the registrations share the count, not the poll: each records
    the count when it read the table (``_mount_generation``), and a later count tells it the
    table changed."""

    __slots__: tuple[str, ...] = ("generation", "watch")

    def __init__(self) -> None:
        self.watch: _MountWatch | None = None
        self.generation: int = 0


_SHARED_MOUNT_WATCH = _SharedMountWatch()


def _mount_generation() -> int | None:
    """How many changes of this namespace's mount table this process's watch has seen, polling
    it now; None where there is no watch (``_watch_mount_table``), when no memo engages. Opens
    the watch at the first call in a process (a forked child opens its own)."""
    shared = _SHARED_MOUNT_WATCH
    watch = shared.watch
    if watch is None or watch.pid != os.getpid():
        watch = shared.watch = _watch_mount_table()
        shared.generation += 1
        if watch is None:
            return None
    try:
        events = watch.poller.poll(0)
    except OSError:
        os.close(watch.fd)
        shared.watch = None
        shared.generation += 1
        return None
    if any(event & (select.POLLPRI | select.POLLERR) for _fd, event in events):
        shared.generation += 1
    return shared.generation


def mount_points(table: bytes | None) -> frozenset[str] | None:
    """Every mount point in ``table`` (from `mount` where it is None), or None when unknown."""
    if table is None:
        try:
            result = subprocess.run(
                ["mount"], capture_output=True, check=True, text=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None
        points = set[str]()
        for line in result.stdout.splitlines():
            match = _MOUNT_COMMAND_LINE.fullmatch(line)
            if match is not None:
                points.add(match[1])
        return frozenset(points)
    return frozenset(
        _decode_mount_point(fields[1])
        for line in table.splitlines()
        if len(fields := line.split()) >= 2
    )


# A writer-side shortcut only. Each naming unit registers its project. A full registration reads
# the id and the mount table and writes the entry. This records, per process, what it checked:
# the mount-change count of the process's watch on the mount table (``_mount_generation``: one
# poll reports every mount change since the registration read it) and the identity of the
# project id file and of the entry, with both nonce files kept open. While those are unchanged
# and each open nonce file is still the one at its path, a unit skips the reads but never the
# nonce rotation (`_rotate_view_nonce`). A collection never reads this: `read_register` and
# `collect_roots` always read the files. It holds two descriptors per registration, so it keeps
# the `_MAX_VERIFIED_REGISTRATIONS` latest registrations and closes the rest: a long-lived process
# (a watch or serve session, a library caller) that registers in many caches holds a bounded
# number, and a registration it evicted is only done in full again at its next use, which puts
# it back. A unit that uses the memo does not reorder it, so a hit costs no dictionary write.
_RegistrationKey = tuple[pathlib.Path, pathlib.Path, pathlib.Path]
_FileIdentity = tuple[int, int, int]
_MAX_VERIFIED_REGISTRATIONS = 16


class _VerifiedRegistration(NamedTuple):
    register_root: pathlib.Path
    mount_generation: int
    id_path: str
    id_identity: _FileIdentity
    entry_path: str
    entry_identity: _FileIdentity
    state_nonce_path: str
    state_nonce_fd: int
    register_nonce_path: str
    register_nonce_fd: int
    view_key: str
    cache_id: str


# Oldest registration first.
_VERIFIED_REGISTRATIONS = dict[_RegistrationKey, _VerifiedRegistration]()


def _file_identity(path: pathlib.Path | str) -> _FileIdentity | None:
    """A file's (dev, ino, mtime), or None when it cannot be read."""
    try:
        stats = os.stat(path, follow_symlinks=False)
    except (OSError, ValueError):
        return None
    return (stats.st_dev, stats.st_ino, stats.st_mtime_ns)


def _still_open_at(path: str, fd: int) -> bool:
    """Whether ``fd`` is still the file at ``path``, so a write through it lands there."""
    try:
        at_path = os.stat(path, follow_symlinks=False)
        opened = os.fstat(fd)
    except OSError:
        return False
    return (at_path.st_dev, at_path.st_ino) == (opened.st_dev, opened.st_ino)


def _registration_unchanged(key: _RegistrationKey, root: pathlib.Path) -> bool:
    """Whether this process already registered ``key`` against the evidence it sees now, in
    the register under ``root`` (``register_root``: a re-pointed ``files/`` link moves it)."""
    verified = _VERIFIED_REGISTRATIONS.get(key)
    if verified is None or verified.register_root != root:
        return False
    if _mount_generation() != verified.mount_generation:
        return False
    if _file_identity(verified.id_path) != verified.id_identity:
        return False
    if _file_identity(verified.entry_path) != verified.entry_identity:
        return False
    return _still_open_at(verified.state_nonce_path, verified.state_nonce_fd) and _still_open_at(
        verified.register_nonce_path, verified.register_nonce_fd
    )


def _remember_registration(key: _RegistrationKey, verified: _VerifiedRegistration) -> None:
    """Keep ``verified`` as the latest registration, closing the oldest beyond
    `_MAX_VERIFIED_REGISTRATIONS`."""
    _VERIFIED_REGISTRATIONS[key] = verified
    while len(_VERIFIED_REGISTRATIONS) > _MAX_VERIFIED_REGISTRATIONS:
        _forget_registration(next(iter(_VERIFIED_REGISTRATIONS)))


def _forget_registration(key: _RegistrationKey) -> None:
    verified = _VERIFIED_REGISTRATIONS.pop(key, None)
    if verified is not None:
        os.close(verified.state_nonce_fd)
        os.close(verified.register_nonce_fd)


def _rotate_view_nonce(key: _RegistrationKey) -> None:
    """Rotate the view nonce, for a unit the memo lets skip the rest of the registration.

    The memo never skips this: a clone of the state directory taken while this process runs (a
    VM snapshot, `docker commit`) holds the nonce of the last unit, which this unit replaces
    before it names anything new, so the clone's collection no longer accepts this entry.
    Raises OSError; the caller then registers in full.
    """
    verified = _VERIFIED_REGISTRATIONS[key]
    _rotate_nonces(verified.state_nonce_fd, verified.register_nonce_fd)


def _mounts_under(points: frozenset[str] | None, bases: Iterable[pathlib.Path]) -> list[str] | None:
    """The mount points at or below each base, by real path; None when ``points`` is unknown."""
    if points is None:
        return None
    real_bases = [os.path.realpath(base) for base in bases]
    return sorted(
        point
        for point in points
        if any(point == base or point.startswith(base.rstrip("/") + "/") for base in real_bases)
    )


def entry_from_file(path: pathlib.Path) -> dict[str, object] | None:
    """A readable JSON object at ``path``, or None when absent or unusable."""
    try:
        data: object = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return cast("dict[str, object]", data) if isinstance(data, dict) else None


def _write_json(path: pathlib.Path, value: object) -> None:
    def write(fd: int) -> None:
        with os.fdopen(fd, "w") as file:
            json.dump(value, file, sort_keys=True)
            file.write("\n")

    cache.atomic_write_file(path, write)


def _cache_mounts_path(root: pathlib.Path, view_key: str) -> pathlib.Path:
    return root / CACHE_MOUNTS_DIR / f"{view_key}.json"


def _record_cache_mounts(
    cache_dir: pathlib.Path, layout: StoreLayout, points: frozenset[str] | None
) -> None:
    """Record, for this view, the mount points a read of the cache's register and refs depends
    on: those below the cache root, or, when the register follows a linked object store, those
    below that register and that store (every root of the store records the same)."""
    host, machine, mount_ns = this_view()
    path = _cache_mounts_path(layout.register_root, view_key_of(host, machine, mount_ns))
    if layout.linked_store is None or layout.unshared:
        real_cache = os.path.realpath(cache_dir)
        mounts = _mounts_under(points, [cache_dir])
        if mounts is not None:
            mounts = [mount for mount in mounts if mount != real_cache]
    else:
        mounts = _mounts_under(points, [layout.register_root / REGISTER_DIR, layout.linked_store])
    desired = {"host": host, "machine": machine, "mount_ns": mount_ns, "mounts": mounts}
    if entry_from_file(path) != desired:
        path.parent.mkdir(exist_ok=True)
        _write_json(path, desired)


def _entry_timestamp(when: float) -> str:
    """A register entry's ``last_write``, which compares as text in the order of its time."""
    return datetime.datetime.fromtimestamp(when, datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _entry_is_current(path: pathlib.Path, fields: dict[str, object], *, now: float) -> bool:
    """Whether the entry at ``path`` already records ``fields`` — everything a registration
    writes but its timestamp — and recently enough that its ``last_write`` still shows the
    project in use.

    Every process of one view writes the same document but its timestamp (a worker pool is a
    dozen of them), and the write costs a staged file and a rename over the live entry. So one
    that says the same thing is kept until ``_ENTRY_REFRESH_SECONDS`` have passed, and the
    entry is rewritten then, whoever is running.
    """
    existing = entry_from_file(path)
    if existing is None or set(existing) != {*fields, "last_write"}:
        return False
    recorded = existing["last_write"]
    if not isinstance(recorded, str) or recorded < _entry_timestamp(now - _ENTRY_REFRESH_SECONDS):
        return False
    return all(existing[key] == value for key, value in fields.items())


def register_project(
    cache_dir: pathlib.Path,
    project_root: pathlib.Path,
    state_dir: pathlib.Path,
    *,
    layout: StoreLayout | None = None,
) -> None:
    """Record this project as a user of ``cache_dir`` with evidence a collection can verify.

    The entry records the project id in its state directory, the mount points below its root
    and state directory, and a fresh nonce this view writes into the state directory first. The
    id and the nonce are best-effort: a state directory that cannot be written (#503's unusable
    layouts) gives an entry without them, which a collection refuses, while the writer keeps
    0.5.2's behaviour. Each process registers in full at its first naming unit, and again when
    its mount table or project id changes; every other unit only rotates the nonce. The register
    write itself must succeed before a writer names an object, so its failure refuses that
    writer. ``EROFS`` on the cache is the one exception: no collection can delete from a
    read-only cache. ``layout`` is the cache's ``store_layout`` when the caller already has it (a
    naming unit resolves it once, for its lock and this registration).
    """
    if not cache_dir.is_dir():
        return
    if layout is None:
        layout = store_layout(cache_dir)
    root = layout.register_root
    key = (cache_dir, project_root, state_dir)
    if _registration_unchanged(key, root):
        try:
            _rotate_view_nonce(key)
            return
        except OSError as e:
            logger.debug("Cannot rotate the view nonce in %s: %s", state_dir, e)
    _forget_registration(key)
    path = root / REGISTER_DIR
    # The count before the bytes: the watch reports every mount change after this poll, so a
    # change between the two only costs this process one more registration.
    generation = _mount_generation()
    table = read_mount_table()
    host, machine, mount_ns = this_view()
    view_key = view_key_of(host, machine, mount_ns)
    state_fd: int | None = None
    register_fd: int | None = None
    cache_id = ""
    try:
        project_id: str | None = _ensure_project_id(state_dir, project_root)
        cache_id = _ensure_cache_id(root)
        state_fd = _open_view_nonce(state_dir, view_key, cache_id)
        # Listed before its first nonce, so no nonce for a cache exists that the project's
        # naming steps in its other caches would not rotate (``_confirm_listed_caches``).
        _record_cache(state_dir, view_key, cache_id, cache_dir)
    except (OSError, ValueError) as e:
        logger.debug("Cannot write the project id, cache id or view nonce for %s: %s", state_dir, e)
        project_id = None
    try:
        path = _entry_path(root, project_root, project_id or "", view_key)
        points = mount_points(table)
        try:
            mounts = _mounts_under(points, [project_root, state_dir])
        except (OSError, ValueError) as e:
            logger.debug("Cannot record mounts for %s: %s", project_root, e)
            mounts = None
        now = time.time()
        fields: dict[str, object] = {
            "root": str(project_root),
            "state_dir": str(state_dir),
            "host": host,
            "project_id": project_id,
            "machine": machine,
            "mount_ns": mount_ns,
            "mounts": mounts,
        }
        path.parent.mkdir(exist_ok=True)
        if project_id is not None and state_fd is not None:
            # The nonce file first: an entry without one is refused, never read with a stale one.
            register_fd = _open_nonce_file(entry_nonce_path(path))
            _rotate_nonces(state_fd, register_fd, truncate=True)
        if not _entry_is_current(path, fields, now=now):
            _write_entry(
                str(path),
                ProjectEntry(
                    root=str(project_root),
                    state_dir=str(state_dir),
                    host=host,
                    last_write=_entry_timestamp(now),
                    project_id=project_id,
                    machine=machine,
                    mount_ns=mount_ns,
                    mounts=mounts,
                ),
            )
        _record_cache_mounts(cache_dir, layout, points)
        if project_id is not None and state_fd is not None and register_fd is not None:
            # This view's entry without an id, written while the state directory could not
            # hold one, is superseded: the entry just written covers the same root and view.
            _entry_path(root, project_root, "", view_key).unlink(missing_ok=True)
            id_path = str(_project_id_path(state_dir))
            id_identity = _file_identity(id_path)
            entry_identity = _file_identity(path)
            if generation is not None and id_identity is not None and entry_identity is not None:
                _remember_registration(
                    key,
                    _VerifiedRegistration(
                        register_root=root,
                        mount_generation=generation,
                        id_path=id_path,
                        id_identity=id_identity,
                        entry_path=str(path),
                        entry_identity=entry_identity,
                        state_nonce_path=str(_view_nonce_path(state_dir, view_key, cache_id)),
                        state_nonce_fd=state_fd,
                        register_nonce_path=str(entry_nonce_path(path)),
                        register_nonce_fd=register_fd,
                        view_key=view_key,
                        cache_id=cache_id,
                    ),
                )
                state_fd = register_fd = None  # the memo owns them now
    except (OSError, ValueError) as e:
        if isinstance(e, OSError) and e.errno == errno.EROFS:
            logger.debug("%s is on a read-only filesystem; nothing can delete there", path)
            if cache_id and not path.exists():
                # This view cannot write the register, so it can never hold an entry there
                # and names nothing there (another view, which can write it, registers the
                # project). It does not list the cache either: a collection of its other
                # caches would otherwise wait for an entry no command of this view can write.
                with contextlib.suppress(OSError):
                    _cache_record_path(state_dir, view_key, cache_id).unlink(missing_ok=True)
            return
        raise _unrecorded_project(cache_dir, path, e) from e
    finally:
        for fd in (state_fd, register_fd):
            if fd is not None:
                os.close(fd)


class _ListedCaches(NamedTuple):
    """This process's copy of one project's list of caches (``cache_records``, every view's):
    the list's directory, its identity when it was read (None while it may still change within
    one timestamp tick), and the records."""

    directory: str
    identity: _FileIdentity | None
    records: list[CacheRecord]


# Per state directory. Read again when the directory changes (a record added, marked, cleared or
# removed); kept only once the directory's mtime is 2 s old, so a record added within the same
# timestamp tick is never missed. A nonce rotation rewrites its files in place, so it never
# changes the directory.
_LISTED_CACHES = dict[pathlib.Path, _ListedCaches]()


def _listed_caches(state_dir: pathlib.Path) -> list[CacheRecord]:
    """Every cache any view of the project lists (``cache_records``), one ``stat`` while the
    list is unchanged. Raises OSError when the list cannot be read."""
    cached = _LISTED_CACHES.get(state_dir)
    directory = (
        cached.directory
        if cached is not None
        else os.fspath(state_dir / "state.lmdb" / VIEW_NONCE_DIR)
    )
    try:
        stats = os.stat(directory)
    except FileNotFoundError:
        return []
    identity = (stats.st_dev, stats.st_ino, stats.st_mtime_ns)
    if cached is not None and cached.identity == identity:
        return cached.records
    records = cache_records(state_dir)
    settled = time.time_ns() - stats.st_mtime_ns > 2_000_000_000
    _LISTED_CACHES[state_dir] = _ListedCaches(directory, identity if settled else None, records)
    return records


def _confirm_listed_caches(
    cache_dir: pathlib.Path, project_root: pathlib.Path, state_dir: pathlib.Path
) -> None:
    """At each naming step in ``cache_dir``, rotate this view's nonce in the project's entry in
    every other cache the project lists, in any view, registering this view there first where
    it has no entry yet.

    A copy of the project (a cloned disk) keeps every nonce as it was when it was taken, and
    only a write the project makes to shared storage after that tells the two apart. A step in
    one cache writes records that may name objects in another (a lock that names again an
    object an earlier run left there), so the step rewrites this view's pair for every cache the
    project lists: then a collection of any of them, which checks only its own entries
    (``gc._view_nonce_reason``), refuses the copy once the project took one step anywhere that
    could reach that cache. The list is every view's (a cache another mount namespace or host
    used counts), but a step writes only this view's own nonces and entries: were it to rotate
    another view's, a copy's own steps would make the original's entries match the copy.

    Each other cache costs a nonce rotation while this process's registration there holds
    (``register_project``'s memo); its first step in a process registers there in full. With
    one cache, this reads one ``stat``. A step in another cache rewrites this cache's pair
    without this cache's naming lock, so a collection reads each pair under the lock writers
    hold across both writes (``read_nonce_pair``).

    A cache this view lists that this step cannot reach (it cannot be read, another cache is at
    its path, or its register cannot be written) does not fail the step: this view's record of
    it is marked unconfirmed (``_mark_unconfirmed``) and the command warns once
    (``_warn_unreachable``). Until a step reaches it again, a copy of the project that can reach
    it may collect this project's objects there; they are then fetched or run again, never used
    stale. A cache only other views list, which this view cannot reach (a host cache a
    container does not mount), is passed over without a word: this view never named anything
    there, and a collection still checks that view's entry (``gc._other_caches_reason``).
    """
    verified = _VERIFIED_REGISTRATIONS.get((cache_dir, project_root, state_dir))
    this_cache: str | None
    if verified is not None:
        view_key, this_cache = verified.view_key, verified.cache_id
    else:
        view_key = view_key_of(*this_view())
        try:
            this_cache = read_cache_id(cache_dir)
        except (OSError, ValueError):
            this_cache = None
    try:
        records = _listed_caches(state_dir)
    except (OSError, ValueError) as e:
        # ValueError: a configured state directory no path can be (a NUL byte, #503), which
        # holds no list either; its registration wrote an entry without an id.
        logger.debug("Cannot read the caches %s lists: %s", state_dir, e)
        return
    by_cache = dict[str, list[CacheRecord]]()
    for record in records:
        if record.cache_id != this_cache:
            by_cache.setdefault(record.cache_id, []).append(record)
    for cache_id, listed in by_cache.items():
        own = next((record for record in listed if record.view_key == view_key), None)
        # Where this view found the cache first, then where any other view did.
        paths = list(
            dict.fromkeys(
                record.cache_dir
                for record in sorted(listed, key=lambda record: record is not own)
                if record.cache_dir is not None
            )
        )
        problem = f"its record {listed[0].record} cannot be read"
        reached: pathlib.Path | None = None
        for path in paths:
            why = _confirm_cache(path, cache_id, project_root, state_dir)
            if why is None:
                reached = path
                break
            problem = why
        if reached is not None:
            # A registration that found the register read-only unlisted the cache instead.
            if own is not None and own.unconfirmed_since is not None and own.record.exists():
                with contextlib.suppress(OSError):
                    _record_cache(state_dir, view_key, cache_id, reached)
            continue
        if own is None:
            logger.debug(
                "Cannot reach %s, which another view of %s lists: %s", cache_id, state_dir, problem
            )
            continue
        _mark_unconfirmed(own)
        _warn_unreachable(cache_id, own.cache_dir or own.record, problem)


def _confirm_cache(
    cache_dir: pathlib.Path, cache_id: str, project_root: pathlib.Path, state_dir: pathlib.Path
) -> str | None:
    """Rotate this view's nonce in the project's entry in the cache ``cache_id`` at
    ``cache_dir``, registering there in full when this process holds no registration there;
    why it cannot, or None."""
    key = (cache_dir, project_root, state_dir)
    if _registration_unchanged(key, register_root(cache_dir)):
        try:
            _rotate_view_nonce(key)
            return None
        except OSError as e:
            logger.debug("Cannot rotate the view nonce for %s: %s", cache_dir, e)
    _forget_registration(key)
    try:
        found = read_cache_id(cache_dir)
    except (OSError, ValueError) as e:
        return f"it cannot be read ({_error_text(e)})"
    if found != cache_id:
        return f"another cache is at that path now (id {found})"
    try:
        register_project(cache_dir, project_root, state_dir)
    except exceptions.CacheError as e:
        cause = e.__cause__
        reason = cause.strerror if isinstance(cause, OSError) and cause.strerror else cause or e
        return f"its register cannot be written ({reason})"
    return None


def _warn_unreachable(cache_id: str, where: pathlib.Path, problem: str) -> None:
    """Warn, once per command, that a listed cache's entry for this project was not brought up
    to date (``state_dir.first_unreachable_cache_report``)."""
    if not state_dir_mod.first_unreachable_cache_report(cache_id):
        return
    logger.warning(
        "This project also uses the cache %s (id %s), and this command cannot reach it: %s. "
        + "Its entry for this project there was not brought up to date, so until a pivot "
        + "command in this project reaches that cache again, a copy of this project (a cloned "
        + "disk) that can reach it could collect this project's objects there; they would then "
        + "be fetched or run again. If no view of this project uses that cache any more, run "
        + "`pivot forget-cache %s` here, in the project itself: never in a copy of it.",
        where,
        cache_id,
        problem,
        cache_id,
    )


def _unrecorded_project(
    cache_dir: pathlib.Path, path: pathlib.Path, error: OSError | ValueError
) -> exceptions.CacheError:
    """The refusal when a writer cannot add its project to a shared cache's register."""
    reason = _error_text(error)
    return exceptions.CacheError(
        f"Cannot record this project in the cache register at {path}: {reason}. "
        + f"This command names objects in the cache {cache_dir}, and a `pivot gc` run in another "
        + "project would not see this project's records and could remove them. Make "
        + f"{path.parent} and this project's state directory writable for this user."
    )


def naming_lock_paths(cache_root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """The cache's naming turnstile and naming lock, the one pair every writer and `pivot gc` use.

    ``cache_root`` is the cache root as configured: the directory that holds ``files/``
    (``config.get_cache_dir()``, or the engine's ``cache_dir``), however its path is spelled.
    The lock files live beside the cache's register (``store_layout``): in the cache root in a
    plain layout, and in the directory that holds the object store when ``files/`` is a link to
    it, so every cache root that uses one object store takes one lock, whatever its spelling.
    Never pass the object root itself: its own register root is a different directory, and so
    a different, unshared lock.
    """
    return _lock_paths_at(register_root(cache_root))


def _lock_paths_at(root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """``naming_lock_paths`` in the register root ``root``."""
    return root / NAMING_TURNSTILE, root / NAMING_LOCK


@contextlib.contextmanager
def naming_unit(
    cache_dir: pathlib.Path, *, project: tuple[pathlib.Path, pathlib.Path] | None
) -> Generator[None]:
    """Hold the naming lock shared for one commit unit: from the first existence check on a
    cache object to the durable record (lock, ``.pvt``, workspace link) being on disk.

    ``project`` is the ``(root, state_dir)`` of the project whose records this unit writes. It
    is registered in the cache under the shared lock, before anything is named (one ``stat``
    while its entry is fresh), so every process that names an object is registered first,
    whatever command it runs and whether or not the cache existed when it started. Only a
    test that holds the lock as a stand-in writer passes ``None``.

    Lock order: naming.turnstile (SH) → naming.lock (SH) → turnstile UN → … → naming UN.
    The turnstile SH is released immediately after acquiring naming SH. This unit must NOT
    span stage execution — only the commit step.

    The cache directory is created when missing: the commit this unit guards is about to
    publish into it, and a first commit into a fresh cache must hold the lock like any other
    (otherwise ``gc --grace-hours 0`` started mid-commit could collect its objects). Where
    ``flock`` is refused (``cache.NO_FLOCK_ERRNO``), the unit runs unlocked, because
    ``pivot gc`` refuses to collect there. Where the lock files cannot be opened (they do not
    exist yet and this user cannot create them), the unit refuses: another user's collection
    could create them and delete while this writer named objects unlocked. The one exception
    is a read-only filesystem, where nothing — a collection included — can delete.

    Raises:
        exceptions.CacheError: the lock files cannot be opened on a writable filesystem.
    """
    layout = store_layout(cache_dir)
    ts_path, nm_path = _lock_paths_at(layout.register_root)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        ts_fd = cache.open_cache_lock(ts_path)
    except OSError as e:
        if e.errno != errno.EROFS:
            raise _unopenable_naming_lock(cache_dir, ts_path, e) from e
        logger.debug("%s is on a read-only filesystem; nothing can delete there", ts_path)
        yield
        return
    try:
        nm_fd = cache.open_cache_lock(nm_path)
    except OSError as e:
        os.close(ts_fd)
        if e.errno != errno.EROFS:
            raise _unopenable_naming_lock(cache_dir, nm_path, e) from e
        logger.debug("%s is on a read-only filesystem; nothing can delete there", nm_path)
        yield
        return
    locked = True
    try:
        try:
            fcntl.flock(ts_fd, fcntl.LOCK_SH)
            fcntl.flock(nm_fd, fcntl.LOCK_SH)
            fcntl.flock(ts_fd, fcntl.LOCK_UN)
        except OSError as e:
            if e.errno not in cache.NO_FLOCK_ERRNO:
                raise
            # gc refuses outright on a filesystem that refuses flock (collection_lock), so a
            # writer here is never racing a collector. This is a reasoned exemption, not a
            # silent fallback: if gc ever collects on a flockless mount, the writers must also
            # take the lock. See storage/AGENTS.md "gc refuses, writers proceed."
            logger.debug(
                "flock refused on %s; proceeding unlocked (gc refuses on this mount)", nm_path
            )
            locked = False
        if project is not None:
            register_project(cache_dir, *project, layout=layout)
            _confirm_listed_caches(cache_dir, *project)
        yield
    finally:
        if locked:
            with contextlib.suppress(OSError):
                fcntl.flock(nm_fd, fcntl.LOCK_UN)
        os.close(nm_fd)
        os.close(ts_fd)


def register_use(
    cache_root: pathlib.Path, project_root: pathlib.Path, state_dir: pathlib.Path
) -> None:
    """Register ``project_root`` in ``cache_root`` in one naming unit, before a command links
    or names that cache's objects outside a publishing unit.

    Pass the cache the command reads from: a restore links objects of the cache it restores
    from, and that cache's collection reads only its own register, so registering in another
    cache protects nothing. A cache that does not exist holds nothing to collect, so nothing is
    created or registered for it.
    """
    if cache_root.is_dir():
        with naming_unit(cache_root, project=(project_root, state_dir)):
            pass


def _unopenable_naming_lock(
    cache_dir: pathlib.Path, path: pathlib.Path, error: OSError
) -> exceptions.CacheError:
    """The refusal a writer gives when it cannot take the cache's naming lock."""
    return exceptions.CacheError(
        f"Cannot open {path} ({error.strerror or error}): this command names objects in the "
        + f"cache {cache_dir} and must hold its naming lock, or a `pivot gc` run by another user "
        + "could remove them meanwhile. Make "
        + f"{path.parent} writable for this user (the cache root, or the directory that holds the "
        + "object store its files/ links to), or have someone who can write it run any pivot "
        + "command there once to create the lock files."
    )
