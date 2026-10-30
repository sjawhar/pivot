"""Collect the cache objects no live record names.

Liveness is read from disk, never inferred. Five classes of record name an object: a stage
lock (``outs`` and ``deps``), a ``.pvt`` file, a run-cache row, a workspace link into the
cache, and a ref under ``<cache>/files/refs``. Each contributes a file's hash, or, for a
directory, the hash of its listing — which is itself the cache object at that hash, and the
name of its whole-directory tree under ``<cache>/trees`` — and every file hash that listing
names (``storage/listing.py``). A directory whose listing this collection cannot read refuses
it: the members are unknown, and a collection that cannot read a record never licences a
delete.

A cache outside the project root can hold another project's outputs, whose locks this project
cannot read. Pivot therefore keeps a register of the projects that use a cache, at
``<cache>/projects/``, written by every naming unit (``naming_unit``, and through it by
``config.register_cache_use()`` before a restore links objects), and
refuses to collect an external cache that records nothing.

Two rules cover the window between publishing an object and writing the record that names it
(``storage/AGENTS.md``): every object whose ``st_ctime`` is inside the grace period is kept,
and so is every object some directory entry outside the cache still hardlinks.
"""

from __future__ import annotations

import contextlib
import datetime
import errno
import fcntl
import functools
import hashlib
import importlib.metadata
import json
import logging
import os
import pathlib
import re
import secrets
import select
import socket
import subprocess
import time
from typing import TYPE_CHECKING, Literal, NamedTuple, TypedDict, cast

import lmdb
import yaml

from pivot import exceptions, project, run_history, yaml_config
from pivot import state_dir as state_dir_mod
from pivot.storage import cache, listing, lock, member_skip, state, track
from pivot.types import is_dir_hash

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from pivot.types import DirHash

logger = logging.getLogger(__name__)

# Cache objects live under this directory of the cache; the register is its sibling, so no
# cache scan sees it (``remote/sync.py`` descends two-character prefix directories only).
OBJECTS_DIR = "files"
# Whole-directory trees a symlink checkout links to, one per directory hash
# (``cache.tree_path``). A tree's bytes are copies of its members, so it is collected on its
# own; the record that names the directory keeps both it and the listing object.
TREES_DIR = "trees"
REGISTER_DIR = "projects"
# Held exclusively for a whole collection, so two collections never run on one cache.
GC_LOCK = "gc.lock"
# The naming lock excludes gc-from-writers. Writers hold it shared for each commit unit (from
# the object existence check to the durable record that names it); gc holds it exclusive while
# it deletes (re-read register → prune → collect_roots → unlink). The turnstile prevents
# starvation: gc takes it exclusive first, so new writers queue, then gc takes naming exclusive
# to wait for in-flight writers to finish. Lock order: gc.lock → prefix (listing, before any
# naming lock) → turnstile → naming; gc takes no prefix lock while it holds naming exclusive.
NAMING_LOCK = "naming.lock"
NAMING_TURNSTILE = "naming.turnstile"
# Every marker file beside a whole-directory tree object, removed with the tree.
_TREE_MARKER_SUFFIXES = cache.TREE_MARKER_SUFFIXES
DEFAULT_GRACE_HOURS = 24.0
# A cache key as it appears in the text of a record pivot cannot parse.
_KEY_RE = re.compile(r"(?<![0-9a-zA-Z])[0-9a-f]{16}(?![0-9a-zA-Z])")
# A project's identity lives in its state database directory, which the state directory's
# ignore file keeps out of git, so two checkouts of one repository never share it. A copy (`cp
# -a`, a container image, a cloned disk) keeps the id; the view nonce tells it apart.
PROJECT_ID_FILE = "pivot-project-id"
_ENTRY_ID_RE = re.compile(r"[0-9a-f]{32}")
# This process's mount namespace. Entries are keyed by it and the hostname, so each view of a
# project (another host, a container that keeps the hostname) keeps its own mount evidence.
_MOUNT_NAMESPACE = pathlib.Path("/proc/self/ns/mnt")
# The machine id, so two machines that share a hostname (cloned VMs) are different views.
_MACHINE_ID = pathlib.Path("/etc/machine-id")
# One nonce file per view in a project's state directory: proof of what that view last wrote.
VIEW_NONCE_DIR = "gc-views"
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
    pivot: str
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
    nonce: str | None
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
    # The records a scope held that the collection could not read: locks that did not parse
    # (M6: non-UTF-8/truncated/conflicted) or did not decode. A non-empty list means a record of
    # an unknown shape could name any object, so a collection keeps every object and removes
    # none.
    unreadable: list[str]
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


def _view_key(host: str, machine: str, mount_ns: int | None) -> str:
    """The key of one view of a project: a hostname, a machine id and a mount namespace."""
    return hashlib.sha256(f"{host}\0{machine}\0{mount_ns}".encode()).hexdigest()[:16]


def _this_view() -> tuple[str, str, int | None]:
    return socket.gethostname(), _machine_id(), _mount_namespace()


def register_path(
    cache_dir: pathlib.Path, project_root: pathlib.Path, project_id: str, view_key: str
) -> pathlib.Path:
    """Return the register entry one view of a project owns in a cache.

    The key holds the root, the project id and the view (hostname, machine id and mount
    namespace). Each host and each mount view of a project writes its own entry, so no writer
    replaces another's mount evidence, and host-local projects at one path keep separate entries.
    """
    key = f"{project_root}\0{project_id}\0{view_key}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return cache_dir / REGISTER_DIR / f"{digest}.json"


def project_register_path(
    cache_dir: pathlib.Path, project_root: pathlib.Path, state_dir: pathlib.Path
) -> pathlib.Path | None:
    """Return this view's entry for the project id in ``state_dir``, or None without an id."""
    try:
        project_id = _read_project_id(state_dir)
    except (OSError, ValueError):
        return None
    return register_path(cache_dir, project_root, project_id, _view_key(*_this_view()))


def _view_nonce_path(state_dir: pathlib.Path, view_key: str) -> pathlib.Path:
    """Where one view's latest registration nonce lives in a project's state directory (inside
    ``state.lmdb``, which the state directory's ignore file keeps out of git)."""
    return state_dir / "state.lmdb" / VIEW_NONCE_DIR / view_key


def _open_view_nonce(state_dir: pathlib.Path, view_key: str) -> int:
    """Open (creating) this view's nonce file in the state directory. Raises OSError or
    ValueError (a NUL byte in an unusable configured path)."""
    path = _view_nonce_path(state_dir, view_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o644)
    os.fchmod(fd, 0o644)  # other users' collections must read it, whatever the umask
    return fd


def _write_entry(path: str, document: ProjectEntry) -> None:
    """Replace ``path`` with ``document`` atomically, readable by every user of the cache."""
    data = (json.dumps(document, sort_keys=True) + "\n").encode()
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


def _entry_nonce_path(entry: pathlib.Path) -> pathlib.Path:
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
    data = f"{secrets.token_hex(16)}\n".encode()
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


def _read_nonce_file(path: pathlib.Path) -> str:
    """Read a nonce file without following a symlink. Raises OSError or ValueError."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(fd) as file:
        nonce = file.read().strip()
    if not _ENTRY_ID_RE.fullmatch(nonce):
        raise ValueError(f"{path} does not hold a pivot view nonce")
    return nonce


def _read_view_nonce(state_dir: pathlib.Path, view_key: str) -> str:
    """Read one view's nonce in the state directory. Raises OSError or ValueError."""
    return _read_nonce_file(_view_nonce_path(state_dir, view_key))


def _project_id_path(state_dir: pathlib.Path) -> pathlib.Path:
    return state_dir / "state.lmdb" / PROJECT_ID_FILE


def _read_project_id(state_dir: pathlib.Path) -> str:
    """Read the project id without following a symlink. Raises OSError or ValueError."""
    path = _project_id_path(state_dir)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(fd) as file:
        project_id = file.read().strip()
    if not _ENTRY_ID_RE.fullmatch(project_id):
        raise ValueError(f"{path} does not hold a pivot project id")
    return project_id


def _ensure_project_id(state_dir: pathlib.Path, project_root: pathlib.Path) -> str:
    """Return this state directory's project id, creating it once. Raises OSError or ValueError.

    A new configured state directory takes the id from the project's `.pivot` state, so changing
    ``core.state_dir`` does not leave the project's current entry under a different identity. A
    copy of a project keeps the id; what tells a copy from the project is the view nonce.
    """
    with contextlib.suppress(FileNotFoundError):
        return _read_project_id(state_dir)
    if not state_dir.is_dir():
        raise FileNotFoundError(errno.ENOENT, "state directory does not exist", str(state_dir))
    legacy = project_root / ".pivot"
    try:
        project_id = _read_project_id(legacy) if legacy != state_dir else None
    except FileNotFoundError:
        project_id = None
    database_dir = state_dir / "state.lmdb"
    database_dir.mkdir(exist_ok=True)
    staged = database_dir / f".{PROJECT_ID_FILE}.{secrets.token_hex(8)}"
    try:
        staged.write_text((project_id or secrets.token_hex(16)) + "\n")
        with contextlib.suppress(FileExistsError):
            os.link(staged, _project_id_path(state_dir))
    finally:
        staged.unlink(missing_ok=True)
    return _read_project_id(state_dir)


def _decode_mount_point(raw: bytes) -> str:
    return os.fsdecode(_MOUNT_ESCAPE.sub(lambda match: bytes([int(match[1], 8)]), raw))


def _read_mount_table() -> bytes | None:
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
    return _MountWatch(fd, poller)


def _mount_table_changed(watch: _MountWatch) -> bool:
    """Whether this namespace's mount table changed since ``watch`` was armed or last polled."""
    try:
        events = watch.poller.poll(0)
    except OSError:
        return True
    return any(event & (select.POLLPRI | select.POLLERR) for _fd, event in events)


def _mount_points(table: bytes | None) -> frozenset[str] | None:
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
# a watch on the mount table (``_watch_mount_table``: one poll reports every mount change since
# the registration read it) and the identity of the project id file and of the entry, with both
# nonce files kept open. While those are unchanged and each open nonce file is still the one at
# its path, a unit skips the reads but never the nonce rotation (`_rotate_view_nonce`). A
# collection never reads this: `read_register` and `collect_roots` always read the files.
_RegistrationKey = tuple[pathlib.Path, pathlib.Path, pathlib.Path]
_FileIdentity = tuple[int, int, int]


class _VerifiedRegistration(NamedTuple):
    watch: _MountWatch
    id_path: str
    id_identity: _FileIdentity
    entry_path: str
    entry_identity: _FileIdentity
    state_nonce_path: str
    state_nonce_fd: int
    register_nonce_path: str
    register_nonce_fd: int


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


def _registration_unchanged(key: _RegistrationKey) -> bool:
    """Whether this process already registered ``key`` against the evidence it sees now."""
    verified = _VERIFIED_REGISTRATIONS.get(key)
    if verified is None or _mount_table_changed(verified.watch):
        return False
    if _file_identity(verified.id_path) != verified.id_identity:
        return False
    if _file_identity(verified.entry_path) != verified.entry_identity:
        return False
    return _still_open_at(verified.state_nonce_path, verified.state_nonce_fd) and _still_open_at(
        verified.register_nonce_path, verified.register_nonce_fd
    )


def _forget_registration(key: _RegistrationKey) -> None:
    verified = _VERIFIED_REGISTRATIONS.pop(key, None)
    if verified is not None:
        os.close(verified.state_nonce_fd)
        os.close(verified.register_nonce_fd)
        os.close(verified.watch.fd)


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


def _entry_from_file(path: pathlib.Path) -> dict[str, object] | None:
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


def _cache_mounts_path(cache_dir: pathlib.Path, view_key: str) -> pathlib.Path:
    return cache_dir / CACHE_MOUNTS_DIR / f"{view_key}.json"


def _record_cache_mounts(cache_dir: pathlib.Path, points: frozenset[str] | None) -> None:
    """Record, for this view, the mount points below the cache root a register read depends on."""
    host, machine, mount_ns = _this_view()
    path = _cache_mounts_path(cache_dir, _view_key(host, machine, mount_ns))
    real_cache = os.path.realpath(cache_dir)
    mounts = _mounts_under(points, [cache_dir])
    if mounts is not None:
        mounts = [mount for mount in mounts if mount != real_cache]
    desired = {"host": host, "machine": machine, "mount_ns": mount_ns, "mounts": mounts}
    if _entry_from_file(path) != desired:
        path.parent.mkdir(exist_ok=True)
        _write_json(path, desired)


@functools.cache
def _pivot_version() -> str:
    """This pivot's version, read once: the lookup walks the installation's metadata, which a
    registration in every naming unit must not pay for."""
    try:
        return importlib.metadata.version("pivot")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - always installed
        return "unknown"


def _entry_timestamp(when: float) -> str:
    """A register entry's ``last_write``, which compares as text in the order of its time."""
    return datetime.datetime.fromtimestamp(when, datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _entry_is_current(path: pathlib.Path, fields: dict[str, object], *, now: float) -> bool:
    """Whether the entry at ``path`` already records ``fields`` — everything a registration
    writes but its timestamp and this pivot's version — and recently enough that its
    ``last_write`` still shows the project in use.

    Every process of one view writes the same document but its timestamp (a worker pool is a
    dozen of them), and the write costs a staged file and a rename over the live entry. So one
    that says the same thing is kept until ``_ENTRY_REFRESH_SECONDS`` have passed, and the
    entry is rewritten then, whoever is running; an upgraded pivot records its version at that
    refresh. The version is not compared, because reading it walks this installation's metadata
    (``_pivot_version``), which costs more than every other part of a registration together.
    """
    existing = _entry_from_file(path)
    if existing is None or set(existing) != {*fields, "last_write", "pivot"}:
        return False
    recorded = existing["last_write"]
    if not isinstance(recorded, str) or recorded < _entry_timestamp(now - _ENTRY_REFRESH_SECONDS):
        return False
    return all(existing[key] == value for key, value in fields.items())


def register_project(
    cache_dir: pathlib.Path, project_root: pathlib.Path, state_dir: pathlib.Path
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
    read-only cache.
    """
    if not cache_dir.is_dir():
        return
    key = (cache_dir, project_root, state_dir)
    if _registration_unchanged(key):
        try:
            _rotate_view_nonce(key)
            return
        except OSError as e:
            logger.debug("Cannot rotate the view nonce in %s: %s", state_dir, e)
    _forget_registration(key)
    path = cache_dir / REGISTER_DIR
    # The watch before the bytes: it reports every mount change after it is armed, so a change
    # between the two only costs this process one more registration.
    watch = _watch_mount_table()
    table = _read_mount_table()
    host, machine, mount_ns = _this_view()
    view_key = _view_key(host, machine, mount_ns)
    state_fd: int | None = None
    register_fd: int | None = None
    try:
        project_id: str | None = _ensure_project_id(state_dir, project_root)
        state_fd = _open_view_nonce(state_dir, view_key)
    except (OSError, ValueError) as e:
        logger.debug("Cannot write the project id or view nonce in %s: %s", state_dir, e)
        project_id = None
    try:
        path = register_path(cache_dir, project_root, project_id or "", view_key)
        points = _mount_points(table)
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
            register_fd = _open_nonce_file(_entry_nonce_path(path))
            _rotate_nonces(state_fd, register_fd, truncate=True)
        if not _entry_is_current(path, fields, now=now):
            _write_entry(
                str(path),
                ProjectEntry(
                    root=str(project_root),
                    state_dir=str(state_dir),
                    host=host,
                    pivot=_pivot_version(),
                    last_write=_entry_timestamp(now),
                    project_id=project_id,
                    machine=machine,
                    mount_ns=mount_ns,
                    mounts=mounts,
                ),
            )
        _record_cache_mounts(cache_dir, points)
        if project_id is not None and state_fd is not None and register_fd is not None:
            # This view's entry without an id, written while the state directory could not
            # hold one, is superseded: the entry just written covers the same root and view.
            register_path(cache_dir, project_root, "", view_key).unlink(missing_ok=True)
            id_path = str(_project_id_path(state_dir))
            id_identity = _file_identity(id_path)
            entry_identity = _file_identity(path)
            if watch is not None and id_identity is not None and entry_identity is not None:
                _VERIFIED_REGISTRATIONS[key] = _VerifiedRegistration(
                    watch=watch,
                    id_path=id_path,
                    id_identity=id_identity,
                    entry_path=str(path),
                    entry_identity=entry_identity,
                    state_nonce_path=str(_view_nonce_path(state_dir, view_key)),
                    state_nonce_fd=state_fd,
                    register_nonce_path=str(_entry_nonce_path(path)),
                    register_nonce_fd=register_fd,
                )
                state_fd = register_fd = None  # the memo owns them now
                watch = None
    except (OSError, ValueError) as e:
        if isinstance(e, OSError) and e.errno == errno.EROFS:
            logger.debug("%s is on a read-only filesystem; nothing can delete there", path)
            return
        raise _unrecorded_project(cache_dir, path, e) from e
    finally:
        for fd in (state_fd, register_fd, None if watch is None else watch.fd):
            if fd is not None:
                os.close(fd)


def _unrecorded_project(
    cache_dir: pathlib.Path, path: pathlib.Path, error: OSError | ValueError
) -> exceptions.CacheError:
    """The refusal when a writer cannot add its project to a shared cache's register."""
    reason = error.strerror if isinstance(error, OSError) and error.strerror else str(error)
    return exceptions.CacheError(
        f"Cannot record this project in the cache register at {path}: {reason}. "
        + f"This command names objects in the cache {cache_dir}, and a `pivot gc` run in another "
        + "project would not see this project's records and could remove them. Make "
        + f"{path.parent} and this project's state directory writable for this user."
    )


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
    try:
        nonce: str | None = _read_nonce_file(_entry_nonce_path(path))
    except (OSError, ValueError):
        nonce = None
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
        if isinstance(project_id, str) and _ENTRY_ID_RE.fullmatch(project_id)
        else None,
        machine=machine if isinstance(machine, str) else "",
        mount_ns=mount_ns if isinstance(mount_ns, int) and not isinstance(mount_ns, bool) else None,
        nonce=nonce,
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
        project_id = _read_project_id(scope["state_dir"])
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
    host, machine, mount_ns = _this_view()
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
    nonce into its state directory just before the entry; reading the same nonce here proves this
    view reads those bytes (one directory, as over NFS), not a clone or an overlay copy.
    """
    scope = registered["scope"]
    view_key = _view_key(scope["host"], registered["machine"], registered["mount_ns"])
    if registered["nonce"] is None:
        detail = "the entry has no readable view nonce"
    else:
        try:
            nonce = _read_view_nonce(scope["state_dir"], view_key)
        except (OSError, ValueError) as e:
            detail = f"no nonce for that view readable here: {_error_text(e)}"
        else:
            if nonce == registered["nonce"]:
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
    cache_dir: pathlib.Path, points: frozenset[str] | None
) -> list[tuple[str, str]]:
    """Why the cache's register or refs may be hidden by a mount this view cannot confirm.

    Each view's record is checked: this view's recorded mounts must be mounted now, and another
    view's record may name no mount under the cache root, since this view cannot check it.
    """
    directory = cache_dir / CACHE_MOUNTS_DIR
    try:
        with os.scandir(directory) as entries:
            paths = sorted(pathlib.Path(e.path) for e in entries if e.name.endswith(".json"))
    except FileNotFoundError:
        return []
    except OSError as e:
        return [(str(directory), _error_text(e))]
    reasons = list[tuple[str, str]]()
    for path in paths:
        recorded = _entry_from_file(path)
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
        ) == _this_view()
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
    cache_dir: pathlib.Path,
) -> tuple[list[ProjectScope], list[tuple[str, str]]]:
    """Return the register's live projects and the entries a collection cannot use.

    A live entry supplies positive evidence only after the collector finds the entry's project
    id in the registered project's current state directory, which comes from that project's own
    ``core.state_dir`` resolver. On the registering host, every recorded mount below the project
    must still be mounted. ``resolve_scopes`` then opens every scope's root, `.pivot` and state
    directory before any record is read.
    """
    register = cache_dir / REGISTER_DIR
    live = list[ProjectScope]()
    points = _mount_points(_read_mount_table())  # always read now; never the writer memo
    unreadable = _cache_mount_reasons(cache_dir, points)
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
            unreadable.append(_registered_scope_reason(path, scope, f"cannot read its config: {e}"))
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
        + ", or, for a project that is gone\nfor good, delete the register entry named above. "
        + "Nothing was removed."
    )
    return "\n".join(lines)


def resolve_scopes(
    cache_dir: pathlib.Path,
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
            entry cannot be read.
    """
    here = ProjectScope(
        root=project_root, state_dir=state_dir, host=socket.gethostname(), last_write="now"
    )
    # Always read the register (BLK-3), before any --projects registration can replace evidence
    # a collection must check: an internal cache can have a registered peer too.
    live, unreadable = read_register(cache_dir)
    if unreadable:
        raise exceptions.CacheCollectionRefusedError(
            refusal_unreadable(cache_dir, len(live) + len(unreadable), unreadable)
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
                refusal_unreadable(cache_dir, 1, [(str(root), f"cannot read its config: {e}")])
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
                )
            ) from e
        named.append(scope)
        if not dry_run:
            register_project(cache_dir, root, scope_state_dir)

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
                refusal_unreadable(cache_dir, len(scopes), [(scope_label(scope), _error_text(e))])
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


def _text_keys(path: pathlib.Path) -> set[str]:
    """Every cache key in the text of a record pivot cannot parse.

    A record whose shape pivot does not recognise is no evidence that its objects are dead,
    so every key it could name is taken as a root. Over-approximating keeps objects.
    """
    logger.warning("Cannot read %s as a pivot record; keeping every cache object it names", path)
    return set(_KEY_RE.findall(path.read_text(errors="replace")))


class _UninterpretableLockError(Exception):
    """A stage lock holds a schema or key this pivot does not know; ``collect_roots`` refuses."""


def _lock_keys(path: pathlib.Path, listings: _Listings) -> set[str]:
    """Every key a stage lock names: each ``outs`` and ``deps`` hash, and every file of each
    directory it records, read from that directory's listing (``lock.recorded_hashes``).

    ``open`` raising ``OSError`` refuses the run (the lock cannot be read, so what it names is
    unknown); ``yaml.load`` raising ``yaml.YAMLError`` (a committed lock left with git conflict
    markers, a truncated one) is caught by ``collect_roots``, which then keeps every object
    rather than let a lock it cannot parse licence a delete. A lock with a newer schema, or a
    key this pivot does not know, may name objects there (a newer pivot's list of objects a run
    read by hash): ``_UninterpretableLockError``, which refuses the run and names the lock."""
    with open(path) as f:
        data: object = yaml.load(f, Loader=yaml_config.Loader)
    recorded = lock.recorded_hashes(data)
    if isinstance(recorded, lock.UninterpretableLock):
        raise _UninterpretableLockError(
            f"the stage lock {path} holds {recorded.reason}; a newer pivot may record cache "
            + "objects there, so this pivot cannot tell which objects it names. Collect with "
            + "the pivot that wrote it, or a newer one"
        )
    if recorded is None:
        return _text_keys(path)
    keys = set(recorded.keys)
    for record in recorded.directories:
        keys.update(entry["hash"] for entry in listings.members(record, path)["files"])
    return keys


def _pvt_keys(
    path: pathlib.Path, object_root: pathlib.Path, stage_member: track.PvtSkip
) -> set[str]:
    """Every key a ``.pvt`` names: its hash and, for a directory, every file of its listing
    (``track.tracked_members``: the listing committed beside it, the cache's object, or the one
    an older pivot's ``.pvt`` holds inline).

    A ``.pvt`` a stage wrote into its own directory output (#540, ``stage_member``) is that
    stage's data, not a record: nothing restores from it, so a directory it names may have no
    listing anywhere. Its keys are still kept, but a listing it lacks does not refuse the
    collection. Every other ``.pvt`` is a tracked file, and its missing listing does.
    """
    data = track.read_pvt_file(path)
    if data is None:
        return _text_keys(path)
    record = track.pvt_to_hash_info(data)
    keys = {record["hash"]}
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


def _row_keys(
    output: run_history.OutputHashEntry, source: pathlib.Path, listings: _Listings
) -> set[str]:
    """Every key one run-cache output entry names: its own, and each file of a directory's
    listing — the one an older pivot's row holds inline, else the one ``listings`` reads."""
    record = run_history.entry_to_output_hash(output)
    if record is None:
        # An older pivot's row lists a directory's files inline; one that records neither is a
        # row no pivot writes, so nothing proves what it names and the collection refuses.
        raw = cast("dict[str, object]", cast("object", output))
        inline = listing.from_inline(raw.get("manifest"), raw.get("empty_dirs", []))
        if inline is None:
            raise exceptions.ListingMissingError(
                f"{source} holds a run-cache row recording the directory {output['hash']} "
                + "without its size, file count or files; what it names is unknown"
            )
        return {output["hash"], *(entry["hash"] for entry in inline["files"])}
    keys = {record["hash"]}
    if is_dir_hash(record):
        keys.update(entry["hash"] for entry in listings.members(record, source)["files"])
    return keys


def _run_cache_keys(
    state_dir: pathlib.Path, listings: _Listings, valid_run_ids: set[str] | None = None
) -> tuple[set[str], int]:
    """Every key the run-cache rows of one project's database name, and the number of rows.

    ``valid_run_ids`` restricts the rows to those a run-history prune would keep — its sentinel
    rows (``__``-prefixed) and the rows of runs within the retention — so the plan counts an
    object whose only record is a row the confirmed run will drop (``StateDB.prune_runs``)."""
    database = state_dir / "state.lmdb"
    if not database.exists():
        return set(), 0
    with state.StateDB(state_dir, readonly=True) as db:
        entries = db.list_run_cache_entries()
    if valid_run_ids is not None:
        entries = [
            entry
            for entry in entries
            if entry["run_id"].startswith("__") or entry["run_id"] in valid_run_ids
        ]
    keys = set[str]()
    for entry in entries:
        for output in entry["output_hashes"]:
            keys |= _row_keys(output, database, listings)
    return keys, len(entries)


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


def _walk_project(
    root: pathlib.Path,
    object_root: pathlib.Path,
    tree_root: pathlib.Path,
    skip: pathlib.Path | None,
) -> _WalkResult:
    """Every ``.pvt`` file under ``root``, every key a symlink there resolves to, every
    committed listing file, and every symlinked directory that leaves the project.

    A file-by-file symlink checkout links each member to its object; a whole-directory
    checkout links the directory itself to its tree under ``tree_root``. Both are read here.
    ``skip`` is the cache itself, listed object by object rather than walked.

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
    resolved_trees = tree_root.resolve()
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
    resolved_trees: pathlib.Path,
) -> _LinkTarget:
    """Where the symlinked directory ``path`` really is: inside the project, outside it, or in
    the cache (listed object by object, never walked), with the key of the whole-directory tree
    it links to when it is one.

    ``project.real_location_outside`` is BET-301's containment rule (#533); a link it cannot
    resolve counts as outside, because where it leads is unknown. Only a link into
    ``<cache>/files`` or ``<cache>/trees`` is the cache's own: the cache root can be a broad
    scratch directory that also holds a user's data (``PIVOT_CACHE_DIR=/scratch``), and records
    under it are records like any others, so a link there that leaves the project counts as
    leaving it.
    """
    outside = project.real_location_outside(path, resolved_root)
    real = path.resolve() if outside is None else outside
    if real.is_relative_to(resolved_objects):
        return _LinkTarget("cache", None)
    if real.is_relative_to(resolved_trees):
        key = real.parent.name + real.name
        return _LinkTarget("cache", key if cache.is_valid_hash(key) else None)
    return _LinkTarget("inside" if outside is None else "outside", None)


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
            unknown and nothing may be removed. A directory record whose listing is neither in
            the cache nor committed is one such project: its files are unknown.
    """
    object_root = cache_dir / OBJECTS_DIR
    tree_root = cache_dir / TREES_DIR
    roots = Roots(
        named=set[str](),
        run_cache=set[str](),
        linked=set[str](),
        counts=RootCounts(locks=0, pvt_files=0, rows=0, links=0),
        unreadable=list[str](),
        outward_dirs=list[str](),
    )
    for scope in scopes:
        try:
            # Every listing this project committed is indexed before its records are read, so a
            # record whose listing the cache has lost is still read from the file beside it.
            walked = _walk_project(scope["root"], object_root, tree_root, cache_dir)
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
                    try:
                        roots["named"] |= _lock_keys(lock_path, listings)
                    except (yaml.YAMLError, UnicodeDecodeError):
                        logger.warning(
                            "Cannot read %s as a stage lock; keeping every cache object, because a lock pivot cannot read may name any of them",
                            lock_path,
                        )
                        roots["unreadable"].append(str(lock_path))
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
        except (
            exceptions.ListingMissingError,
            exceptions.CorruptListingError,
            _UninterpretableLockError,
        ) as e:
            raise ProjectUnreadableError(scope, str(e)) from e

    refs = object_root / "refs"
    if refs.is_dir():
        roots["counts"]["links"] += len(
            ref_keys := _walk_project(refs, object_root, tree_root, None).keys
        )
        roots["linked"] |= ref_keys
    return roots


def reread_run_cache(roots: Roots, scopes: Iterable[ProjectScope], cache_dir: pathlib.Path) -> None:
    """Replace ``roots``' run-cache keys with every row the scopes' databases hold now.

    Called under the exclusive naming lock after a run-history prune, so the rows the prune
    dropped stop protecting their objects and every row that remains, however recent, does.
    A row that records a directory is read with its listing, as ``collect_roots`` reads one.

    Raises:
        ProjectUnreadableError: a scope's database, or the listing of a directory one of its
            rows records, cannot be read.
    """
    object_root = cache_dir / OBJECTS_DIR
    tree_root = cache_dir / TREES_DIR
    keys = set[str]()
    rows = 0
    for scope in scopes:
        try:
            listings = _Listings(object_root)
            for listing_path in _walk_project(
                scope["root"], object_root, tree_root, cache_dir
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
            raise ProjectUnreadableError(scope, str(e)) from e
    roots["run_cache"] = keys
    roots["counts"]["rows"] = rows


# -----------------------------------------------------------------------------
# Locking, scanning and removing
# -----------------------------------------------------------------------------


@contextlib.contextmanager
def collection_lock(cache_dir: pathlib.Path) -> Generator[None]:
    """Hold the cache's ``gc.lock`` exclusively so two collections never run on one cache.

    Lock order: gc.lock is taken FIRST, before the naming lock.

    Raises:
        exceptions.CacheCollectionRefusedError: another collection holds it, or the cache
            cannot be written, or ``flock`` is refused on this filesystem.
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


def naming_lock_paths(cache_root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """The cache's naming turnstile and naming lock, the one pair every writer and `pivot gc` use.

    ``cache_root`` is the cache root as configured: the directory that holds ``files/``
    (``config.get_cache_dir()``, or the engine's ``cache_dir``), however its path is spelled. The
    lock files live in that directory, so any spelling of it reaches the same files. Never pass a
    path derived from the object root: ``files`` may be a symlink to another disk, and the parent
    of where it really is would be a different directory, and so a different, unshared lock.
    """
    return cache_root / NAMING_TURNSTILE, cache_root / NAMING_LOCK


@contextlib.contextmanager
def naming_exclusive(cache_dir: pathlib.Path) -> Generator[None]:
    """Hold the naming lock exclusive: no writer can name a cache object while this is held.

    Lock order: gc.lock → naming.turnstile (EX) → naming.lock (EX); no prefix lock under it.
    The turnstile prevents writer starvation: gc takes it exclusive first so new commit
    units queue, then gc takes naming exclusive to wait for in-flight units to finish.
    """
    ts_path, nm_path = naming_lock_paths(cache_dir)
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
    ts_path, nm_path = naming_lock_paths(cache_dir)
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
            register_project(cache_dir, *project)
        yield
    finally:
        if locked:
            with contextlib.suppress(OSError):
                fcntl.flock(nm_fd, fcntl.LOCK_UN)
        os.close(nm_fd)
        os.close(ts_fd)


def _unopenable_naming_lock(
    cache_dir: pathlib.Path, path: pathlib.Path, error: OSError
) -> exceptions.CacheError:
    """The refusal a writer gives when it cannot take the cache's naming lock."""
    return exceptions.CacheError(
        f"Cannot open {path} ({error.strerror or error}): this command names objects in the "
        + f"cache {cache_dir} and must hold its naming lock, or a `pivot gc` run by another user "
        + "could remove them meanwhile. Make the cache root writable for this user, or have "
        + "someone who can write it run any pivot command there once to create the lock files."
    )


@contextlib.contextmanager
def _prefix_locked(prefix_dir: pathlib.Path) -> Generator[None]:
    """Hold one prefix directory's lock exclusively, excluding every staging publisher and
    the temporary sweep while this directory is listed.

    Lock order: gc takes this only while it lists, before any naming lock and never under the
    exclusive one; a writer may take a prefix lock shared inside its naming unit."""
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
    """List every object in the cache and decide, under each prefix's lock, what may go.

    Both stores are listed: the objects under ``files`` (a file's content, and a directory's
    listing) and the whole-directory trees under ``trees``, which a symlink checkout links to.
    One directory hash names its listing object and its tree, so the record that names the
    directory keeps both.
    """
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
    for prefix_dir in [*_prefix_dirs(object_root), *_prefix_dirs(cache_dir / TREES_DIR)]:
        with _prefix_locked(prefix_dir), os.scandir(prefix_dir) as entries:
            for entry in sorted(entries, key=lambda e: e.name):
                key = prefix_dir.name + entry.name
                # Skips every name that is not an object: the prefix lock, a `.pivot_tmp_*`
                # entry a publisher owns, and a tree's `.content-checked` marker and `.lock` file.
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
                    if roots["unreadable"] or roots["outward_dirs"]
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

    No prefix lock is taken here: while gc holds the exclusive naming lock it must never wait
    for another process, or every writer on the cache would wait behind it. A publisher staging
    into a prefix (which holds that prefix's lock shared for its whole copy) needs no exclusion
    from these deletions: it publishes outside any naming unit, and its unit re-checks every
    object, publishing again any removed here, before a record names it. Temporary entries are
    never in the plan (``build_plan`` lists each prefix under its lock and takes only keys).
    """
    cutoff = (time.time() if now is None else now) - grace_hours * 3600
    removed = ObjectCount(objects=0, bytes=0)
    for doomed in sorted(plan["doomed"], key=lambda d: d["path"]):
        path = doomed["path"]
        try:
            stats = path.lstat()
        except FileNotFoundError:
            removed["objects"] += 1
            removed["bytes"] += doomed["bytes"]
            continue
        key = path.parent.name + path.name
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
        # M5: unlink markers before the tree, so a partial rmtree never leaves a marker on a
        # truncated tree that a later checkout would trust.
        if doomed["is_dir"]:
            for suffix in _TREE_MARKER_SUFFIXES:
                with contextlib.suppress(OSError):
                    path.with_name(path.name + suffix).unlink(missing_ok=True)
        try:
            cache.clear_path(path)
        except OSError as e:
            logger.warning("Cannot remove cache object %s: %s", path, e)
            continue
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
