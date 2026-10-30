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
import subprocess
import time
from typing import TYPE_CHECKING, NamedTuple, TypedDict, cast

from pivot import exceptions
from pivot.storage import cache

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

logger = logging.getLogger(__name__)

# The register of the projects that use a cache: one entry per view of a project. A sibling of
# the object store, so no cache scan sees it (``remote/sync.py`` descends two-character prefix
# directories only).
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
# replaces the nonce another cache's entry holds (a run with another `cache_dir`).
VIEW_NONCE_DIR = "gc-views"
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
        project_id = read_project_id(state_dir)
    except (OSError, ValueError):
        return None
    return register_path(cache_dir, project_root, project_id, view_key_of(*this_view()))


def _view_nonce_path(state_dir: pathlib.Path, view_key: str, cache_id: str) -> pathlib.Path:
    """Where one view's latest registration nonce for one cache lives in a project's state
    directory (inside ``state.lmdb``, which the state directory's ignore file keeps out of git)."""
    return state_dir / "state.lmdb" / VIEW_NONCE_DIR / f"{view_key}-{cache_id}"


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


def read_view_nonce(state_dir: pathlib.Path, view_key: str, cache_id: str) -> str:
    """Read one view's nonce for one cache in the state directory. Raises OSError or
    ValueError."""
    return read_nonce_file(_view_nonce_path(state_dir, view_key, cache_id))


def _cache_id_path(cache_dir: pathlib.Path) -> pathlib.Path:
    return cache_dir / REGISTER_DIR / CACHE_ID_FILE


def read_cache_id(cache_dir: pathlib.Path) -> str:
    """Read the cache's id without following a symlink. Raises OSError or ValueError."""
    path = _cache_id_path(cache_dir)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(fd) as file:
        cache_id = file.read().strip()
    if not ENTRY_ID_RE.fullmatch(cache_id):
        raise ValueError(f"{path} does not hold a pivot cache id")
    return cache_id


def _ensure_cache_id(cache_dir: pathlib.Path) -> str:
    """Return the cache's id, creating it once (the first writer's link wins). Raises OSError or
    ValueError."""
    with contextlib.suppress(FileNotFoundError):
        return read_cache_id(cache_dir)
    register = cache_dir / REGISTER_DIR
    register.mkdir(exist_ok=True)
    staged = register / f".{CACHE_ID_FILE}.{os.urandom(8).hex()}"
    try:
        fd = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o644)
        with os.fdopen(fd, "w") as file:
            os.fchmod(file.fileno(), 0o644)  # every user of the cache reads it
            file.write(os.urandom(16).hex() + "\n")
        with contextlib.suppress(FileExistsError):
            os.link(staged, _cache_id_path(cache_dir))
    finally:
        staged.unlink(missing_ok=True)
    return read_cache_id(cache_dir)


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


def _cache_mounts_path(cache_dir: pathlib.Path, view_key: str) -> pathlib.Path:
    return cache_dir / CACHE_MOUNTS_DIR / f"{view_key}.json"


def _record_cache_mounts(cache_dir: pathlib.Path, points: frozenset[str] | None) -> None:
    """Record, for this view, the mount points below the cache root a register read depends on."""
    host, machine, mount_ns = this_view()
    path = _cache_mounts_path(cache_dir, view_key_of(host, machine, mount_ns))
    real_cache = os.path.realpath(cache_dir)
    mounts = _mounts_under(points, [cache_dir])
    if mounts is not None:
        mounts = [mount for mount in mounts if mount != real_cache]
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
    table = read_mount_table()
    host, machine, mount_ns = this_view()
    view_key = view_key_of(host, machine, mount_ns)
    state_fd: int | None = None
    register_fd: int | None = None
    cache_id = ""
    try:
        project_id: str | None = _ensure_project_id(state_dir, project_root)
        cache_id = _ensure_cache_id(cache_dir)
        state_fd = _open_view_nonce(state_dir, view_key, cache_id)
    except (OSError, ValueError) as e:
        logger.debug("Cannot write the project id, cache id or view nonce for %s: %s", state_dir, e)
        project_id = None
    try:
        path = register_path(cache_dir, project_root, project_id or "", view_key)
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
                    state_nonce_path=str(_view_nonce_path(state_dir, view_key, cache_id)),
                    state_nonce_fd=state_fd,
                    register_nonce_path=str(entry_nonce_path(path)),
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
        + "could remove them meanwhile. Make the cache root writable for this user, or have "
        + "someone who can write it run any pivot command there once to create the lock files."
    )
