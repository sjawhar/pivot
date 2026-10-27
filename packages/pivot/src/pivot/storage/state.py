from __future__ import annotations

import contextlib
import errno
import fcntl
import logging
import os
import pathlib
import re
import struct
import threading
import time
import typing
from typing import TYPE_CHECKING, Self

import lmdb

from pivot import exceptions, run_history

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable

    from pivot.fingerprint import AstHashEntry
    from pivot.run_history import RunCacheEntry, RunManifest
    from pivot.types import DeferredWrites, FileHashEntry

# Key prefixes for different entry types
_HASH_PREFIX = b"hash:"  # File hash entries
_INODE_PREFIX = b"inode:"  # File identity (st_dev, st_ino) -> hash: key of one path with it
_GEN_PREFIX = b"gen:"  # Output generation counters (identity-based)
_DEP_PREFIX = b"dep:"  # Stage dependency generations (identity-based)
_REMOTE_PREFIX = b"remote:"  # Remote index entries
_REMOTE_URL_PREFIX = b"remote_url:"  # Remote URL tracking
_RUN_PREFIX = b"run:"  # Run history entries
_RUNCACHE_PREFIX = b"runcache:"  # Run cache entries for skip detection
_FP_PREFIX = b"fp:"  # AST fingerprint/hash cache entries
_SM_PREFIX = b"sm:"  # Stage manifest cache entries

# Default LMDB map size (10GB virtual - grows as needed)
_MAP_SIZE = 10 * 1024 * 1024 * 1024

# LMDB default max key size
_MAX_KEY_SIZE = 511

# Common error message for MapFullError
_DB_FULL_MSG = (
    f"State cache is full ({_MAP_SIZE // (1024**3)}GB limit). Delete .pivot/state.lmdb/ to reset."
)

# How long a best-effort write-back waits for another process's write lock
_WRITE_BACK_TIMEOUT = 2.0

# Filesystem types on which one (st_dev, st_ino) names one live file, so a file there may be
# looked up by identity (``_make_key_file_identity``). ext2 and ext3 keep ext2's on-disk inode
# tables (kernel Documentation/filesystems/ext2.rst). #513's `.pvt` walk memo reads the same
# list; whichever of the two merges second moves both to one shared helper.
_IDENTITY_FS_TYPES = frozenset(
    {b"ext2", b"ext3", b"ext4", b"xfs", b"btrfs", b"tmpfs", b"f2fs", b"zfs"}
)
_MOUNTINFO = pathlib.Path("/proc/self/mountinfo")
_MOUNTINFO_ESCAPE = re.compile(rb"\\([0-7]{3})")


class _SharedEnv(typing.TypedDict):
    """One process-local lease over an LMDB environment."""

    env: lmdb.Environment | None
    identity: tuple[int, int, int, int]
    users: int


_FORK_DRAIN_TIMEOUT = 1.0


@typing.final
class _ForkGate:
    """Allow concurrent transactions while draining them before fork."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._transactions = 0
        self._forking = False

    @contextlib.contextmanager
    def transaction(self) -> Generator[None]:
        with self._condition:
            while self._forking:
                self._condition.wait()
            self._transactions += 1
        try:
            yield
        finally:
            with self._condition:
                self._transactions -= 1
                if self._transactions == 0:
                    self._condition.notify_all()

    def before_fork(self) -> bool:
        """Block new transactions and return whether existing ones drained."""
        self._condition.acquire()
        self._forking = True
        deadline = time.monotonic() + _FORK_DRAIN_TIMEOUT
        while self._transactions:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._forking = False
                self._condition.notify_all()
                self._condition.release()
                return False
            self._condition.wait(remaining)
        return True

    def after_fork_parent(self) -> None:
        self._forking = False
        self._condition.notify_all()
        self._condition.release()

    def after_fork_child(self) -> None:
        self._forking = False
        self._condition.release()

    def abort_fork(self) -> None:
        self._forking = False
        self._condition.notify_all()
        self._condition.release()


# LMDB permits one environment per process/file. Fingerprint, explain,
# coordinator, and worker paths create StateDB independently, so passing one
# explicit handle cannot cover their overlapping lifetimes.
_shared_envs = dict[pathlib.Path, _SharedEnv]()
_shared_envs_lock = threading.RLock()
_fork_gate = _ForkGate()
_fork_drained = False
_inherited_env_paths = set[pathlib.Path]()


def _open_env(lmdb_path: pathlib.Path) -> lmdb.Environment:
    """Open the environment shared by readonly and writable StateDBs.

    It is writable when the database's files are. A database whose data file this process
    cannot write is opened read-only, with LMDB's lock file, so its readers stay registered
    against any writer; ``StateDB`` refuses it to a writer. A lock file this process cannot
    write as well (another user's database, a read-only filesystem) raises PermissionError:
    LMDB would open without it there, and a reader no writer can see is not protected from
    one, so the caller reads files instead (``open_readonly``). Any other failure raises as
    LMDB reports it, a damaged database included, except a damaged database whose data and
    lock files this process can write neither: the lock-file check comes first, so that one
    raises the PermissionError above.
    """
    try:
        return lmdb.open(str(lmdb_path), map_size=_MAP_SIZE, max_spare_txns=0)
    except lmdb.Error:
        data_path = lmdb_path / "data.mdb"
        if not data_path.is_file() or os.access(data_path, os.W_OK):
            raise
        lock_path = lmdb_path / "lock.mdb"
        if not os.access(lock_path, os.W_OK):
            raise PermissionError(
                errno.EACCES, "state database lock file is not writable", str(lock_path)
            ) from None
        return lmdb.open(str(lmdb_path), map_size=_MAP_SIZE, max_spare_txns=0, readonly=True)


def _env_identity(lmdb_path: pathlib.Path) -> tuple[int, int, int, int] | None:
    """Return the identity of the data and lock files for an LMDB environment."""
    try:
        data_stat = (lmdb_path / "data.mdb").stat()
        lock_stat = (lmdb_path / "lock.mdb").stat()
    except FileNotFoundError:
        return None
    return data_stat.st_dev, data_stat.st_ino, lock_stat.st_dev, lock_stat.st_ino


def _acquire_shared_env(lmdb_path: pathlib.Path) -> _SharedEnv:
    """Acquire this process's shared LMDB environment lease for lmdb_path."""
    if lmdb_path in _inherited_env_paths:
        raise StateDBError(
            f"StateDB {lmdb_path} was open during fork; use spawn or forkserver to access it in the child"
        )
    with _shared_envs_lock:
        shared = _shared_envs.get(lmdb_path)
        if shared is not None and _env_identity(lmdb_path) != shared["identity"]:
            shared = None
        if shared is None:
            env = _open_env(lmdb_path)
            identity = _env_identity(lmdb_path)
            if identity is None:
                raise RuntimeError(f"LMDB environment files missing after open: {lmdb_path}")
            shared = _SharedEnv(env=env, identity=identity, users=0)
            _shared_envs[lmdb_path] = shared
        elif shared["env"] is None:
            shared["env"] = _open_env(lmdb_path)
        shared["users"] += 1
        return shared


def _release_shared_env(lmdb_path: pathlib.Path, shared: _SharedEnv) -> None:
    """Release a StateDB's lease and close its environment when it is unused."""
    with _shared_envs_lock:
        shared["users"] -= 1
        if shared["users"] != 0:
            return
        if _shared_envs.get(lmdb_path) is shared:
            del _shared_envs[lmdb_path]
        env = shared["env"]
        shared["env"] = None
        if env is not None:
            env.close()


def _before_fork() -> None:
    """Drain StateDB transactions, then close environments before a fork."""
    global _fork_drained

    _fork_drained = _fork_gate.before_fork()
    if not _fork_drained:
        return
    try:
        with _shared_envs_lock:
            for shared in _shared_envs.values():
                env = shared["env"]
                if env is not None:
                    env.close()
                    shared["env"] = None
    except BaseException:
        _fork_gate.abort_fork()
        _fork_drained = False
        raise


def _after_fork_parent() -> None:
    global _fork_drained

    if _fork_drained:
        _fork_gate.after_fork_parent()
    _fork_drained = False


def _after_fork_child() -> None:
    global _fork_drained, _fork_gate, _inherited_env_paths, _shared_envs, _shared_envs_lock

    if _fork_drained:
        _fork_gate.after_fork_child()
        _inherited_env_paths = set[pathlib.Path]()
    else:
        _inherited_env_paths = set(_shared_envs)
    _fork_drained = False
    _fork_gate = _ForkGate()
    _shared_envs = dict[pathlib.Path, _SharedEnv]()
    _shared_envs_lock = threading.RLock()


os.register_at_fork(
    before=_before_fork,
    after_in_parent=_after_fork_parent,
    after_in_child=_after_fork_child,
)


class StateDBError(Exception):
    """Base exception for StateDB errors."""


class PathTooLongError(StateDBError):
    """Raised when a file path exceeds LMDB's key size limit."""


class DatabaseFullError(StateDBError):
    """Raised when the state database reaches its size limit."""


def _make_key_file_hash(path: str | os.PathLike[str]) -> bytes:
    """Create the LMDB key for a file-hash entry: the absolute path as given, as bytes.

    Symlinks are not followed: the entry's value holds the (mtime_ns, size, device, inode) of
    ``path.stat()``, which follows them. An entry read through a link retargeted at another
    file, on this filesystem or on another one (where an inode number can repeat), no longer
    matches, and the file is hashed again. ``os.fsencode`` keeps names that are not UTF-8.
    Contrast with _make_key_output_generation(), which keys an artifact identity.
    """
    return _HASH_PREFIX + os.fsencode(os.path.abspath(path))


def _make_key_file_identity(dev: int, inode: int) -> bytes:
    """Create the key of the identity index: a file's device and inode number.

    A file-hash entry is keyed by the path the file was read at, so a replica of a project
    (`cp -al`, or a copy that hard-links pivot's read-only outputs) finds no entry for its own
    paths, though its files are the very inodes the copied entries describe. The index maps
    each inode to the key of one file-hash entry written for it (a hint for one path per
    inode, not always the last written: a batch that writes P, Q, P for one inode leaves Q),
    and a lookup that misses by path follows it.

    The identity lookup is never weaker than a path lookup at the recorded path. It runs only
    on a filesystem that keeps one (st_dev, st_ino) for one live file (below), and serves
    another path's entry only when (1) that entry matches this file's stat exactly as a path
    lookup there would (mtime_ns, size, device, inode) and (2) that path, stat'ed now, still
    names this file: the same device and inode, and the same change time. A path lookup at
    that path would then serve the same hash, and the two paths name one inode, so one
    content. In particular:

    - A hard link (`cp -al`, `os.link`) is the same inode, and every link of an inode shares
      its change time: served without reading it.
    - A copy is a new inode, with no index entry: read, as before.
    - Inode reuse: once a recorded file is deleted, the filesystem may give its number to a
      new file, and a copy that keeps mtimes (`cp -a`, `rsync -a`, `shutil.copy2`) can give
      that file the recorded size and mtime as well. The recorded path then names nothing,
      or another inode, so (2) refuses it.
    - Another filesystem: the device is part of the key, of the entry and of check (2).
    - A bind mount shows one filesystem, with one device number, at two places, so its two
      paths are two links to one inode: served while the recorded path still names it.

    The argument assumes that one (st_dev, st_ino) names one live file, as POSIX requires.
    Some filesystems break that: mergerfs with `inodecalc=passthrough` shows each branch's
    inode number under the mount's one device, and other FUSE, network and union filesystems
    make inode numbers up. No stat field tells two such files apart, the change time
    included: one command that changes both (`touch -d @T a b`, `chmod -R`, `chown -R`) gives
    them one change time to the nanosecond. So a file is looked up by identity only when
    ``/proc/self/mountinfo`` mounts its device with a type on ``_IDENTITY_FS_TYPES`` (ext2,
    ext3, ext4, xfs, btrfs, tmpfs, f2fs, zfs; ``_device_allowed``). Any other type, every
    ``fuse.*`` among them, an unknown type, or no mountinfo (macOS) gets the path lookup only:
    such a filesystem loses the replica speedup, never correctness. The change time in (2)
    stays a second check.

    Both change times in (2) are read at lookup time, from what should be one inode, so a hard
    link made before the lookup, which changes the inode's change time, invalidates nothing;
    the entry itself holds no change time (``_FILE_HASH_STAT``) for that reason. A hard link
    made between the stat of the file and the stat of the recorded path (during a long walk)
    turns the file away: one more read, never a wrong hash. The index is only a hint: an
    index entry that is stale or wrong fails (1) or (2) and serves nothing. Check (2) is a
    ``stat``, never a read of the file.
    """
    return _INODE_PREFIX + struct.pack(">QQ", dev, inode)


def _put_file_hashes(txn: lmdb.Transaction, entries: Iterable[FileHashEntry]) -> None:
    """Write file-hash entries, then point the identity index at each.

    The index entries are written after all the hash entries, so in a large write the pages
    of each lie together in the file, and a status whose lookups all hit by path reads no
    index page from disk. An entry this cache cannot hold is left out: a key too long for
    LMDB, or an mtime outside what the unsigned 64-bit field can store, before 1970 or after
    2554 (tmpfs and btrfs keep such times). The cache only saves a later read, so no such
    entry may fail the caller. When a path's previous entry named another inode, that inode's
    index entry is deleted if it still points there, and a path written twice in one batch (a
    worker reads an edited output, then restores it from the run cache as a new inode) is
    indexed only at its last inode, so the index never outgrows the entries.
    """
    index = dict[bytes, bytes]()  # file-hash key -> its identity key; a path's last entry wins
    for path, mtime_ns, size, dev, inode, hash_hex in entries:
        key = _make_key_file_hash(path)
        if len(key) > _MAX_KEY_SIZE:
            logger.debug("Not caching the hash of %s: path too long", path)
            continue
        if not 0 <= mtime_ns <= 0xFFFF_FFFF_FFFF_FFFF:
            logger.debug("Not caching the hash of %s: mtime outside 1970-2554", path)
            continue
        previous = txn.get(key)
        if previous is not None and len(previous) >= _FILE_HASH_STAT.size:
            _, _, old_dev, old_inode, _ = _unpack_value(previous)
            if (old_dev, old_inode) != (dev, inode):
                old_index = _make_key_file_identity(old_dev, old_inode)
                if txn.get(old_index) == key:
                    txn.delete(old_index)
        txn.put(key, _pack_value(mtime_ns, size, dev, inode, hash_hex))
        index[key] = _make_key_file_identity(dev, inode)
    for key, index_key in index.items():
        txn.put(index_key, key)


def _names_inode(path: bytes, fs_stat: os.stat_result) -> bool:
    """Whether ``path`` exists now and is the file ``fs_stat`` describes: the same device and
    inode, and the same change time, which every link of one inode shares (a second check
    behind the filesystem allowlist, ``_make_key_file_identity``)."""
    try:
        current = os.stat(path)
    except OSError:
        return False
    return os.path.samestat(current, fs_stat) and current.st_ctime_ns == fs_stat.st_ctime_ns


def _identity_candidate(
    txn: lmdb.Transaction, key: bytes, fs_stat: os.stat_result
) -> tuple[bytes, str] | None:
    """The path and hash of another path's entry that the identity index offers for a file
    whose own entry (at ``key``) did not match: check (1) of ``_make_key_file_identity``. The
    caller still has to confirm check (2), with ``_names_inode``, outside the transaction."""
    other_key = txn.get(_make_key_file_identity(fs_stat.st_dev, fs_stat.st_ino))
    if other_key is None or other_key == key:
        return None
    other_hash = _match_cached_hash(txn.get(other_key), fs_stat)
    return None if other_hash is None else (other_key[len(_HASH_PREFIX) :], other_hash)


class _Mounts(typing.TypedDict):
    """What ``/proc/self/mountinfo`` says about the identity lookup's allowlist."""

    allowed: frozenset[int]  # devices mounted only with an allowlisted type
    refused: frozenset[int]  # devices mounted with any type off the allowlist
    points: list[tuple[bytes, bytes]]  # (mount point, type), deepest first


def _read_mounts() -> _Mounts:
    """Read the mounts; without the file (not Linux) no device is allowed."""
    try:
        lines = _MOUNTINFO.read_bytes().splitlines()
    except OSError:
        return _Mounts(allowed=frozenset(), refused=frozenset(), points=[])
    allowed = set[int]()
    refused = set[int]()
    points = list[tuple[bytes, bytes]]()
    for line in lines:
        # id parent major:minor root mount-point options [optional fields...] - type source ...
        fields = line.split()
        try:
            fstype = fields[fields.index(b"-", 6) + 1]
            major, minor = fields[2].split(b":")
            device = os.makedev(int(major), int(minor))
        except (ValueError, IndexError):
            continue
        (allowed if fstype in _IDENTITY_FS_TYPES else refused).add(device)
        point = _MOUNTINFO_ESCAPE.sub(lambda match: bytes([int(match[1], 8)]), fields[4])
        points.append((point, fstype))
    points.sort(key=lambda mount: len(mount[0]), reverse=True)
    return _Mounts(allowed=frozenset(allowed - refused), refused=frozenset(refused), points=points)


def _device_allowed(mounts: _Mounts, device: int, path: str | os.PathLike[str]) -> bool:
    """Whether a file on ``device``, at ``path``, may be looked up by identity.

    A device mounted with any type off the allowlist (FUSE, NFS, overlayfs, an unknown type)
    is refused. A device the file does not list is a btrfs subvolume, which reports a device
    of its own: it is allowed when the deepest mount holding ``path`` is btrfs, and refused
    otherwise.
    """
    if device in mounts["allowed"]:
        return True
    if device in mounts["refused"]:
        return False
    real_path = os.fsencode(os.path.realpath(path))
    for point, fstype in mounts["points"]:
        if real_path == point or real_path.startswith(point.rstrip(b"/") + b"/"):
            return fstype == b"btrfs"
    return False


def _make_key_output_generation(identity: str) -> bytes:
    return _GEN_PREFIX + identity.encode()


def _make_key_dep_generation(stage_name: str, dep_identity: str) -> bytes:
    return _DEP_PREFIX + f"{stage_name}:{dep_identity}".encode()


class InvalidAstHashKeyError(Exception):
    """Raised when AST hash key components contain invalid characters."""


def _make_key_ast_hash(
    rel_path: str,
    source_digest: str,
    qualname: str,
    py_version: str,
    schema_version: int,
) -> bytes:
    """Create LMDB key for AST hash entry.

    Key format: fp:{rel_path}\x00{source_digest}\x00{qualname}\x00{py_version}\x00{schema_version}
    Uses null byte separator (can't appear in paths or qualnames). ``source_digest`` is the
    digest of the source file's content: an edit can keep the file's mtime, size and inode.

    Raises:
        InvalidAstHashKeyError: If rel_path, qualname, or py_version contains null bytes or is empty
    """
    if not rel_path:
        raise InvalidAstHashKeyError("rel_path cannot be empty")
    if not qualname:
        raise InvalidAstHashKeyError("qualname cannot be empty")
    if not py_version:
        raise InvalidAstHashKeyError("py_version cannot be empty")
    if "\x00" in rel_path:
        raise InvalidAstHashKeyError(f"rel_path contains null byte: {rel_path!r}")
    if "\x00" in qualname:
        raise InvalidAstHashKeyError(f"qualname contains null byte: {qualname!r}")
    if "\x00" in py_version:
        raise InvalidAstHashKeyError(f"py_version contains null byte: {py_version!r}")

    key = (
        _FP_PREFIX
        + f"{rel_path}\x00{source_digest}\x00{qualname}\x00{py_version}\x00{schema_version}".encode()
    )
    if len(key) > _MAX_KEY_SIZE:
        raise InvalidAstHashKeyError(f"key exceeds {_MAX_KEY_SIZE} bytes: {len(key)}")
    return key


_FILE_HASH_STAT = struct.Struct(">QQQQ")


def _pack_value(mtime_ns: int, size: int, dev: int, inode: int, hash_hex: str) -> bytes:
    """Pack a file's stat fields and hash into a file-hash entry's value.

    The entry holds no ctime: a new hard link changes the inode's ctime, so every hardlink
    checkout, and every replica that hard-links pivot's files, would invalidate its entries.
    """
    return _FILE_HASH_STAT.pack(mtime_ns, size, dev, inode) + hash_hex.encode("ascii")


def _unpack_value(data: bytes) -> tuple[int, int, int, int, str]:
    """Unpack a file-hash entry's value into (mtime_ns, size, dev, inode, hash)."""
    mtime_ns, size, dev, inode = _FILE_HASH_STAT.unpack(data[: _FILE_HASH_STAT.size])
    hash_hex = data[_FILE_HASH_STAT.size :].decode("ascii")
    return mtime_ns, size, dev, inode, hash_hex


def _match_cached_hash(value: bytes | None, fs_stat: os.stat_result) -> str | None:
    """Return the cached hash if the entry's stat fields match ``fs_stat``, else None.

    An entry an earlier pivot wrote, (mtime_ns, size, inode) with no device, reads here with
    its inode as the device and its hash's first characters as the inode, so it matches no
    file: the file is read once again and recorded anew.
    """
    if value is None or len(value) < _FILE_HASH_STAT.size:
        return None
    mtime_ns, size, dev, inode, hash_hex = _unpack_value(value)
    if (
        fs_stat.st_mtime_ns == mtime_ns
        and fs_stat.st_size == size
        and fs_stat.st_dev == dev
        and fs_stat.st_ino == inode
    ):
        return hash_hex
    return None


class StateDB:
    """LMDB cache of file hashes and generation counters.

    StateDB instances for one state directory share one LMDB environment within
    a process. Separate processes open independent environments and LMDB's
    file locks provide MVCC isolation between them. Readonly StateDB instances
    enforce that restriction at the StateDB boundary; the shared environment is
    writable when the database's files are, so a later writer can use it safely,
    and read-only otherwise (``_open_env``), when a writable StateDB raises
    StateDBError at once. Forking closes shared environments before the child
    is created, so both processes reopen their own environment lazily.
    """

    _shared: _SharedEnv
    _lmdb_path: pathlib.Path
    _owner_pid: int
    _closed: bool
    _readonly: bool
    _write_timeout: float
    _write_lock_path: pathlib.Path
    _mounts: _Mounts | None
    _identity_devices: dict[int, bool]

    def __init__(
        self,
        state_dir: pathlib.Path,
        readonly: bool = False,
        write_timeout: float = 30.0,
    ) -> None:
        state_dir.mkdir(parents=True, exist_ok=True)
        lmdb_path = (state_dir / "state.lmdb").resolve()

        with _fork_gate.transaction():
            self._shared = _acquire_shared_env(lmdb_path)
            env = self._shared["env"]
            if not readonly and env is not None and env.flags()["readonly"]:
                _release_shared_env(lmdb_path, self._shared)
                raise StateDBError(f"State database {lmdb_path} cannot be written by this user")
        self._lmdb_path = lmdb_path
        self._owner_pid = os.getpid()
        self._closed = False
        self._readonly = readonly
        self._write_timeout = write_timeout
        self._write_lock_path = lmdb_path / "pivot-write.lock"
        self._mounts = None
        self._identity_devices = dict[int, bool]()

    @property
    def _env(self) -> lmdb.Environment:
        """Return this instance's process-local shared LMDB environment."""
        self._check_closed()
        with _shared_envs_lock:
            env = self._shared["env"]
            if env is None:
                env = _open_env(self._lmdb_path)
                self._shared["env"] = env
            return env

    def _check_closed(self) -> None:
        """Raise if the database is closed or was inherited across a fork."""
        if self._closed:
            raise RuntimeError("Cannot operate on closed StateDB")
        if self._owner_pid != os.getpid():
            raise RuntimeError(
                "Cannot use StateDB inherited across fork; create a new StateDB in the child process"
            )

    def _check_write_allowed(self) -> None:
        """Raise if database is read-only."""
        if self._readonly:
            raise RuntimeError(
                "Internal error: worker attempted write to readonly StateDB. This is a bug in Pivot. Please report it."
            )

    @contextlib.contextmanager
    def _read_transaction(self) -> Generator[lmdb.Transaction]:
        """Begin a short read transaction while excluding fork teardown."""
        with _fork_gate.transaction():
            self._check_closed()
            with self._env.begin() as txn:
                yield txn

    @contextlib.contextmanager
    def _write_transaction(self, timeout: float = 30.0) -> Generator[lmdb.Transaction]:
        """Begin a write transaction with timeout protection.

        The outer flock on pivot-write.lock provides timeout detection only.
        LMDB's internal write serialization (via lock.mdb) handles actual
        mutual exclusion. This wrapper exists because LMDB's env.begin(write=True)
        blocks indefinitely with no timeout mechanism.
        """
        self._check_closed()
        self._check_write_allowed()
        start = time.monotonic()
        with self._write_lock_path.open("a+b") as lock_file:
            while True:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() - start >= timeout:
                        raise exceptions.PivotDBWriteTimeoutError(
                            f"Timed out after {timeout:.1f}s waiting for LMDB write lock"
                        ) from None
                    time.sleep(0.01)
            try:
                with _fork_gate.transaction():
                    self._check_closed()
                    with self._env.begin(write=True) as txn:
                        yield txn
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @property
    def readonly(self) -> bool:
        """Return True if database is opened in readonly mode."""
        return self._readonly

    def get(self, path: pathlib.Path, fs_stat: os.stat_result) -> str | None:
        """Return cached hash if file metadata matches, else None (see ``get_many``)."""
        return self.get_many([(path, fs_stat)])[path]

    def get_many(
        self, items: list[tuple[pathlib.Path, os.stat_result]]
    ) -> dict[pathlib.Path, str | None]:
        """Batch query for multiple files.

        A file whose own path has no matching entry is looked up by its identity, the entry
        of another path that names the same inode, when its filesystem is on the allowlist
        (``_make_key_file_identity`` says when that entry is served). The transaction holds
        only the database reads; the ``stat`` that confirms each identity match comes after it.
        """
        self._check_closed()
        if not items:
            return {}
        results = dict[pathlib.Path, str | None]()
        by_identity = list[tuple[pathlib.Path, os.stat_result, bytes, str]]()
        # Decided before the transaction: _read_mounts reads /proc and _device_allowed may
        # resolve a path, and neither may run while a read transaction pins a snapshot. Each
        # device is decided once, so the call inside the transaction is one dict read.
        for path, fs_stat in items:
            self._identity_allowed(fs_stat.st_dev, path)
        with self._read_transaction() as txn:
            for path, fs_stat in items:
                key = _make_key_file_hash(path)
                found = _match_cached_hash(txn.get(key), fs_stat)
                results[path] = found
                if found is None and self._identity_allowed(fs_stat.st_dev, path):
                    candidate = _identity_candidate(txn, key, fs_stat)
                    if candidate is not None:
                        by_identity.append((path, fs_stat, *candidate))
        for path, fs_stat, other_path, other_hash in by_identity:
            if _names_inode(other_path, fs_stat):
                results[path] = other_hash
        return results

    def _identity_allowed(self, device: int, path: pathlib.Path) -> bool:
        """Whether a file on ``device`` may be looked up by identity (``_device_allowed``).
        The mount table is read on this StateDB's first identity lookup, and each device is
        decided once, so a lookup costs one dict read."""
        allowed = self._identity_devices.get(device)
        if allowed is None:
            if self._mounts is None:
                self._mounts = _read_mounts()
            allowed = self._identity_devices[device] = _device_allowed(self._mounts, device, path)
        return allowed

    def save(self, path: pathlib.Path, fs_stat: os.stat_result, file_hash: str) -> None:
        """Cache file metadata and hash, as ``save_many`` does for one entry."""
        self.save_many([(path, fs_stat, file_hash)])

    def save_many(self, entries: Iterable[tuple[pathlib.Path, os.stat_result, str]]) -> None:
        """Save multiple entries atomically; ``_put_file_hashes`` says which it leaves out."""
        self.save_file_hash_entries(
            (
                os.fspath(path),
                fs_stat.st_mtime_ns,
                fs_stat.st_size,
                fs_stat.st_dev,
                fs_stat.st_ino,
                file_hash,
            )
            for path, fs_stat, file_hash in entries
        )

    def save_file_hash_entries(self, entries: Iterable[FileHashEntry]) -> None:
        """Save ``DeferredWrites`` file-hash entries in one write transaction."""
        self._check_closed()
        self._check_write_allowed()
        try:
            with self._write_transaction(timeout=self._write_timeout) as txn:
                _put_file_hashes(txn, entries)
        except lmdb.MapFullError as e:
            raise DatabaseFullError(_DB_FULL_MSG) from e

    # -------------------------------------------------------------------------
    # AST hash cache for persistent fingerprint caching
    # -------------------------------------------------------------------------

    def get_ast_hash(
        self,
        rel_path: str,
        source_digest: str,
        qualname: str,
        py_version: str,
        schema_version: int,
    ) -> str | None:
        """Get cached AST hash; returns None if not found or key is invalid."""
        self._check_closed()
        try:
            key = _make_key_ast_hash(rel_path, source_digest, qualname, py_version, schema_version)
        except InvalidAstHashKeyError:
            return None
        with self._read_transaction() as txn:
            value = txn.get(key)
        if value is None:
            return None
        return value.decode("ascii")

    def save_ast_hash_many(self, entries: list[AstHashEntry]) -> None:
        """Batch save AST hash entries; skips entries with invalid keys."""
        self._check_closed()
        self._check_write_allowed()
        if not entries:
            return
        try:
            with self._write_transaction(timeout=self._write_timeout) as txn:
                for (
                    rel_path,
                    source_digest,
                    qualname,
                    py_version,
                    schema_version,
                    hash_hex,
                ) in entries:
                    try:
                        key = _make_key_ast_hash(
                            rel_path, source_digest, qualname, py_version, schema_version
                        )
                    except InvalidAstHashKeyError:
                        continue  # Skip invalid entries
                    txn.put(key, hash_hex.encode("ascii"))
        except lmdb.MapFullError as e:
            raise DatabaseFullError(_DB_FULL_MSG) from e

    def clear_fingerprint_caches(self) -> int:
        """Clear all AST hash (`fp:`) and stage manifest (`sm:`) cache entries.

        Returns number of entries deleted.
        """
        self._check_closed()
        self._check_write_allowed()
        deleted = 0
        with self._write_transaction(timeout=self._write_timeout) as txn:
            cursor = txn.cursor()
            keys_to_delete = list[bytes]()
            for prefix in (_FP_PREFIX, _SM_PREFIX):
                if cursor.set_range(prefix):
                    for key, _ in cursor:
                        if not key.startswith(prefix):
                            break
                        keys_to_delete.append(bytes(key))
            for key in keys_to_delete:
                txn.delete(key)
                deleted += 1
        return deleted

    # -------------------------------------------------------------------------
    # Raw key-value access for stage manifest cache
    # -------------------------------------------------------------------------

    def get_raw(self, key: bytes) -> bytes | None:
        """Get raw value by key. Returns None if not found."""
        self._check_closed()
        with self._read_transaction() as txn:
            return txn.get(key)

    def put_raw(self, key: bytes, value: bytes) -> None:
        """Put raw key-value pair."""
        self._check_closed()
        self._check_write_allowed()
        if len(key) > _MAX_KEY_SIZE:
            raise PathTooLongError(
                f"Key too long for state cache ({len(key)} bytes, max {_MAX_KEY_SIZE})"
            )
        try:
            with self._write_transaction(timeout=self._write_timeout) as txn:
                txn.put(key, value)
        except lmdb.MapFullError as e:
            raise DatabaseFullError(_DB_FULL_MSG) from e

    def put_raw_many(self, entries: list[tuple[bytes, bytes]]) -> None:
        """Batch put raw key-value pairs atomically."""
        self._check_closed()
        self._check_write_allowed()
        if not entries:
            return
        try:
            with self._write_transaction(timeout=self._write_timeout) as txn:
                for key, value in entries:
                    if len(key) > _MAX_KEY_SIZE:
                        continue  # Skip oversized keys
                    txn.put(key, value)
        except lmdb.MapFullError as e:
            raise DatabaseFullError(_DB_FULL_MSG) from e

    def delete_raw_many(self, keys: list[bytes]) -> None:
        """Batch delete raw key-value pairs atomically."""
        self._check_closed()
        self._check_write_allowed()
        if not keys:
            return
        try:
            with self._write_transaction(timeout=self._write_timeout) as txn:
                for key in keys:
                    txn.delete(key)
        except lmdb.MapFullError as e:
            raise DatabaseFullError(_DB_FULL_MSG) from e

    def iter_prefix(self, prefix: bytes) -> list[tuple[bytes, bytes]]:
        """Return all raw key-value pairs matching a prefix."""
        self._check_closed()
        results = list[tuple[bytes, bytes]]()
        with self._read_transaction() as txn:
            cursor = txn.cursor()
            if cursor.set_range(prefix):
                for key, value in cursor:
                    if not key.startswith(prefix):
                        break
                    results.append((bytes(key), bytes(value)))
        return results

    # -------------------------------------------------------------------------
    # Generation tracking for O(1) skip detection
    # -------------------------------------------------------------------------

    def get_generation(self, identity: str) -> int | None:
        """Get generation counter for an output identity. Returns None if not tracked."""
        self._check_closed()
        key = _make_key_output_generation(identity)
        with self._read_transaction() as txn:
            value = txn.get(key)
        if value is None:
            return None
        return struct.unpack(">Q", value)[0]

    def get_many_generations(self, identities: list[str]) -> dict[str, int | None]:
        """Batch query for multiple identity generations."""
        self._check_closed()
        if not identities:
            return {}
        results = dict[str, int | None]()
        with self._read_transaction() as txn:
            for identity in identities:
                key = _make_key_output_generation(identity)
                value = txn.get(key)
                if value is None:
                    results[identity] = None
                else:
                    results[identity] = struct.unpack(">Q", value)[0]
        return results

    def increment_generation(self, identity: str) -> int:
        """Increment and return new generation (creates with gen=1 if not exists)."""
        self._check_closed()
        self._check_write_allowed()
        key = _make_key_output_generation(identity)
        if len(key) > _MAX_KEY_SIZE:
            raise PathTooLongError(
                "Generation identity too long for tracking "
                + f"({len(key)} bytes, max {_MAX_KEY_SIZE}): {identity}"
            )
        try:
            with self._write_transaction(timeout=self._write_timeout) as txn:
                value = txn.get(key)
                new_gen = (struct.unpack(">Q", value)[0] + 1) if value else 1
                txn.put(key, struct.pack(">Q", new_gen))
        except lmdb.MapFullError as e:
            raise DatabaseFullError(_DB_FULL_MSG) from e
        return new_gen

    def get_dep_generations(self, stage_name: str) -> dict[str, int] | None:
        self._check_closed()
        prefix = _DEP_PREFIX + stage_name.encode() + b":"
        results = dict[str, int]()
        with self._read_transaction() as txn:
            cursor = txn.cursor()
            if cursor.set_range(prefix):
                for key, value in cursor:
                    if not key.startswith(prefix):
                        break
                    dep_identity = key[len(prefix) :].decode()
                    generation = struct.unpack(">Q", value)[0]
                    results[dep_identity] = generation
        return results if results else None

    def record_dep_generations(self, stage_name: str, deps: dict[str, int]) -> None:
        """Record dependency generations after successful stage execution."""
        self._check_closed()
        self._check_write_allowed()
        prefix = _DEP_PREFIX + stage_name.encode() + b":"
        for identity in deps:
            key = _make_key_dep_generation(stage_name, identity)
            if len(key) > _MAX_KEY_SIZE:
                raise PathTooLongError(
                    "Dependency identity too long for tracking "
                    + f"({len(key)} bytes, max {_MAX_KEY_SIZE}): {identity}"
                )
        try:
            with self._write_transaction(timeout=self._write_timeout) as txn:
                cursor = txn.cursor()
                keys_to_delete = list[bytes]()
                if cursor.set_range(prefix):
                    for key, _ in cursor:
                        if not key.startswith(prefix):
                            break
                        keys_to_delete.append(key)
                for key in keys_to_delete:
                    txn.delete(key)
                for identity, gen in deps.items():
                    key = _make_key_dep_generation(stage_name, identity)
                    txn.put(key, struct.pack(">Q", gen))
        except lmdb.MapFullError as e:
            raise DatabaseFullError(_DB_FULL_MSG) from e

    # -------------------------------------------------------------------------
    # Remote index tracking for avoiding repeated HEAD requests
    # -------------------------------------------------------------------------

    def remote_hash_exists(self, remote_name: str, hash_: str) -> bool:
        """Check if hash is known to exist on remote."""
        self._check_closed()
        key = _REMOTE_PREFIX + f"{remote_name}:{hash_}".encode()
        with self._read_transaction() as txn:
            return txn.get(key) is not None

    def remote_hashes_intersection(self, remote_name: str, hashes: set[str]) -> set[str]:
        """Return subset of hashes known to exist on remote (batch lookup)."""
        self._check_closed()
        if not hashes:
            return set()
        prefix = _REMOTE_PREFIX + remote_name.encode() + b":"
        found = set[str]()
        with self._read_transaction() as txn:
            for hash_ in hashes:
                key = prefix + hash_.encode()
                if txn.get(key) is not None:
                    found.add(hash_)
        return found

    def remote_hashes_add(self, remote_name: str, hashes: Iterable[str]) -> None:
        """Mark hashes as existing on remote."""
        self._check_closed()
        self._check_write_allowed()
        prefix = _REMOTE_PREFIX + remote_name.encode() + b":"
        try:
            with self._write_transaction(timeout=self._write_timeout) as txn:
                for hash_ in hashes:
                    key = prefix + hash_.encode()
                    txn.put(key, b"1")
        except lmdb.MapFullError as e:
            raise DatabaseFullError(_DB_FULL_MSG) from e

    def remote_hashes_remove(self, remote_name: str, hashes: Iterable[str]) -> None:
        """Mark hashes as no longer on remote."""
        self._check_closed()
        self._check_write_allowed()
        prefix = _REMOTE_PREFIX + remote_name.encode() + b":"
        with self._write_transaction(timeout=self._write_timeout) as txn:
            for hash_ in hashes:
                key = prefix + hash_.encode()
                txn.delete(key)

    def remote_get_url(self, remote_name: str) -> str | None:
        """Get stored URL for a remote. Returns None if not tracked."""
        self._check_closed()
        key = _REMOTE_URL_PREFIX + remote_name.encode()
        with self._read_transaction() as txn:
            value = txn.get(key)
        if value is None:
            return None
        return value.decode("utf-8")

    def remote_set_url(self, remote_name: str, url: str) -> None:
        """Store URL for a remote."""
        self._check_closed()
        self._check_write_allowed()
        key = _REMOTE_URL_PREFIX + remote_name.encode()
        value = url.encode("utf-8")
        try:
            with self._write_transaction(timeout=self._write_timeout) as txn:
                txn.put(key, value)
        except lmdb.MapFullError as e:
            raise DatabaseFullError(_DB_FULL_MSG) from e

    def remote_index_clear(self, remote_name: str) -> None:
        """Clear all index entries and stored URL for a remote (force re-indexing)."""
        self._check_closed()
        self._check_write_allowed()
        prefix = _REMOTE_PREFIX + remote_name.encode() + b":"
        with self._write_transaction(timeout=self._write_timeout) as txn:
            cursor = txn.cursor()
            keys_to_delete = list[bytes]()
            if cursor.set_range(prefix):
                for key, _ in cursor:
                    if not key.startswith(prefix):
                        break
                    keys_to_delete.append(key)
            for key in keys_to_delete:
                txn.delete(key)
            # Also delete the remote URL entry
            url_key = _REMOTE_URL_PREFIX + remote_name.encode()
            txn.delete(url_key)

    # -------------------------------------------------------------------------
    # Run history for tracking pipeline executions
    # -------------------------------------------------------------------------

    def write_run(self, manifest: RunManifest) -> None:
        """Write a run manifest to the database."""
        self._check_closed()
        self._check_write_allowed()

        key = _RUN_PREFIX + manifest["run_id"].encode()
        value = run_history.serialize_to_bytes(manifest)
        try:
            with self._write_transaction(timeout=self._write_timeout) as txn:
                txn.put(key, value)
        except lmdb.MapFullError as e:
            raise DatabaseFullError(_DB_FULL_MSG) from e

    def read_run(self, run_id: str) -> RunManifest | None:
        """Read a run manifest by ID. Returns None if not found."""
        self._check_closed()

        key = _RUN_PREFIX + run_id.encode()
        with self._read_transaction() as txn:
            value = txn.get(key)
        if value is None:
            return None
        return run_history.deserialize_run_manifest(value)

    def list_runs(self, limit: int = 100) -> list[RunManifest]:
        """List recent runs, newest first.

        Uses reverse iteration since run IDs are timestamp-prefixed,
        avoiding loading all runs into memory for sorting.
        """
        self._check_closed()

        runs = list[run_history.RunManifest]()
        with self._read_transaction() as txn:
            cursor = txn.cursor()
            # Position at the end of the run: prefix range by seeking past it
            # Use "run;\xff" as upper bound (semicolon > colon in ASCII)
            end_key = _RUN_PREFIX[:-1] + b";\xff"
            if not cursor.set_range(end_key):
                # No keys >= end_key, position at last key in DB
                if not cursor.last():
                    return runs
            else:
                # Move back one to get into the run: range
                if not cursor.prev():
                    return runs

            # Iterate backwards through run: keys
            while True:
                key = cursor.key()
                if not key.startswith(_RUN_PREFIX):
                    break
                value = cursor.value()
                runs.append(run_history.deserialize_run_manifest(value))
                if len(runs) >= limit:
                    break
                if not cursor.prev():
                    break
        return runs

    def prune_runs(self, retention: int) -> int:
        """Remove oldest runs beyond retention limit and orphaned run cache entries.

        Returns number of runs deleted. Run cache entries referencing deleted runs
        are also removed automatically.
        """
        self._check_closed()
        self._check_write_allowed()
        runs = self.list_runs(
            limit=retention + 1000
        )  # Get more than retention to find deletable ones
        if len(runs) <= retention:
            return 0
        to_keep = runs[:retention]
        to_delete = runs[retention:]

        with self._write_transaction(timeout=self._write_timeout) as txn:
            for run in to_delete:
                key = _RUN_PREFIX + run["run_id"].encode()
                txn.delete(key)

        valid_run_ids = {run["run_id"] for run in to_keep}
        self.prune_run_cache(valid_run_ids)

        return len(to_delete)

    # -------------------------------------------------------------------------
    # Run cache for skip detection (like DVC's run cache)
    # -------------------------------------------------------------------------

    def lookup_run_cache(self, stage_name: str, input_hash: str) -> RunCacheEntry | None:
        """Look up run cache entry for skip detection. Returns None if not found."""
        self._check_closed()
        key = _RUNCACHE_PREFIX + f"{stage_name}:{input_hash}".encode()
        with self._read_transaction() as txn:
            value = txn.get(key)
        if value is None:
            return None
        return run_history.deserialize_run_cache_entry(value)

    def prune_run_cache(self, valid_run_ids: set[str]) -> int:
        """Remove run cache entries that reference runs not in valid_run_ids.

        Entries with run_id starting with '__' are sentinel values (e.g., '__committed__')
        and are never pruned.

        Returns number of entries deleted.
        """
        self._check_closed()
        self._check_write_allowed()
        to_delete = list[bytes]()
        with self._read_transaction() as txn:
            cursor = txn.cursor()
            if cursor.set_range(_RUNCACHE_PREFIX):
                key = cursor.key()
                while key.startswith(_RUNCACHE_PREFIX):
                    value = cursor.value()
                    entry = run_history.deserialize_run_cache_entry(value)
                    run_id = entry["run_id"]
                    # Never prune sentinel run_ids (e.g., __committed__ from pivot commit)
                    if not run_id.startswith("__") and run_id not in valid_run_ids:
                        to_delete.append(key)
                    if not cursor.next():
                        break
                    key = cursor.key()

        if not to_delete:
            return 0

        with self._write_transaction(timeout=self._write_timeout) as txn:
            for key in to_delete:
                txn.delete(key)
        return len(to_delete)

    # -------------------------------------------------------------------------
    # Deferred writes for multi-process safety
    # -------------------------------------------------------------------------

    def apply_deferred_writes(
        self,
        stage_name: str,
        output_paths: list[str],
        deferred: DeferredWrites,
    ) -> None:
        """Apply all deferred writes in a single atomic transaction.

        Used by coordinator to apply writes collected from worker processes.
        Workers use readonly StateDB and return deferred writes for the
        coordinator to apply, ensuring LMDB multi-process safety.
        """
        self._check_closed()
        self._check_write_allowed()

        # Pre-validate key lengths before starting transaction
        if "dep_generations" in deferred:
            for identity in deferred["dep_generations"]:
                key = _make_key_dep_generation(stage_name, identity)
                if len(key) > _MAX_KEY_SIZE:
                    raise PathTooLongError(
                        "Dependency identity too long "
                        + f"({len(key)} bytes, max {_MAX_KEY_SIZE}): {identity}"
                    )

        increment_outputs = "increment_outputs" in deferred and deferred["increment_outputs"]
        if increment_outputs:
            for identity in output_paths:
                key = _make_key_output_generation(identity)
                if len(key) > _MAX_KEY_SIZE:
                    raise PathTooLongError(
                        "Output identity too long "
                        + f"({len(key)} bytes, max {_MAX_KEY_SIZE}): {identity}"
                    )

        try:
            with self._write_transaction(timeout=self._write_timeout) as txn:
                # Dependency generations (clear old entries first, like record_dep_generations)
                if "dep_generations" in deferred:
                    prefix = _DEP_PREFIX + stage_name.encode() + b":"
                    cursor = txn.cursor()
                    keys_to_delete = list[bytes]()
                    if cursor.set_range(prefix):
                        for key, _ in cursor:
                            if not key.startswith(prefix):
                                break
                            keys_to_delete.append(key)
                    for key in keys_to_delete:
                        txn.delete(key)
                    for identity, gen in deferred["dep_generations"].items():
                        key = _make_key_dep_generation(stage_name, identity)
                        txn.put(key, struct.pack(">Q", gen))

                # Output generations (increment only when flag is set)
                if increment_outputs:
                    for identity in output_paths:
                        key = _make_key_output_generation(identity)
                        value = txn.get(key)
                        current = struct.unpack(">Q", value)[0] if value else 0
                        txn.put(key, struct.pack(">Q", current + 1))

                # Run cache (only if both keys present)
                if "run_cache_input_hash" in deferred and "run_cache_entry" in deferred:
                    key = (
                        _RUNCACHE_PREFIX
                        + f"{stage_name}:{deferred['run_cache_input_hash']}".encode()
                    )
                    txn.put(key, run_history.serialize_to_bytes(deferred["run_cache_entry"]))
                if "run_cache_delete_input_hash" in deferred:
                    txn.delete(
                        _RUNCACHE_PREFIX
                        + f"{stage_name}:{deferred['run_cache_delete_input_hash']}".encode()
                    )

                # File hash entries; ``_put_file_hashes`` says which it leaves out.
                if "file_hash_entries" in deferred:
                    _put_file_hashes(txn, deferred["file_hash_entries"])

        except lmdb.MapFullError as e:
            raise DatabaseFullError(_DB_FULL_MSG) from e

    def close(self) -> None:
        """Release this StateDB's lease on the shared environment."""
        with _fork_gate.transaction():
            if self._closed:
                return
            self._closed = True
            if self._owner_pid == os.getpid():
                _release_shared_env(self._lmdb_path, self._shared)

    def _check_capacity_warning(self) -> None:
        """Warn if database is approaching capacity limit (80% utilization)."""
        env_info = self._env.info()
        stat = self._env.stat()
        # Used bytes = page size * (last page number + 1) since pages are 0-indexed
        used_bytes = stat["psize"] * (env_info["last_pgno"] + 1)
        if used_bytes > _MAP_SIZE * 0.8:
            used_gb = used_bytes / (1024**3)
            limit_gb = _MAP_SIZE / (1024**3)
            percent = used_bytes / _MAP_SIZE * 100
            logger.warning(
                f"State database at {used_gb:.1f}GB ({percent:.0f}%%), approaching "
                + f"{limit_gb:.0f}GB limit. Run `pivot gc` or delete .pivot/state.lmdb/ to free space."
            )

    def __enter__(self) -> Self:
        with _fork_gate.transaction():
            self._check_closed()
            self._check_capacity_warning()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@contextlib.contextmanager
def open_readonly(state_dir: pathlib.Path, *, create: bool = True) -> Generator[StateDB | None]:
    """Open the state database to read it, yielding None where this user cannot read it safely.

    A database whose lock file this user cannot write cannot register a reader against a
    writer (``_open_env``), so its cache is not used: the caller reads files instead, as it
    would with no database. Only that permission case yields None. A damaged database raises,
    except where this user can write neither its data file nor its lock file: that is the
    permission case, found first, so it yields None too. Opening creates a database where
    there is none, unless ``create`` is False. The caller's own errors propagate with no
    context from the open.
    """
    if not create and not (state_dir / "state.lmdb").exists():
        yield None
        return
    state_db: StateDB | None
    try:
        state_db = StateDB(state_dir, readonly=True)
    except (PermissionError, lmdb.ReadonlyError) as error:
        logger.debug("Reading files without the state database in %s: %s", state_dir, error)
        state_db = None
    except OSError as error:
        if error.errno != errno.EROFS:
            raise
        logger.debug("Reading files without the state database in %s: %s", state_dir, error)
        state_db = None
    if state_db is None:
        yield None
        return
    with state_db:
        yield state_db


def save_best_effort(
    state_dir: pathlib.Path,
    write: Callable[[StateDB], None],
    *,
    what: str,
    create: bool = True,
    timeout: float = _WRITE_BACK_TIMEOUT,
) -> bool:
    """Run ``write`` against a writable StateDB in ``state_dir``, best-effort; True if it ran.

    Every cache a read-only command fills goes through here: the file-hash entries
    (``write_back_file_hashes``) and the fingerprint caches
    (``fingerprint.flush_ast_hash_cache``, ``flush_manifest_cache``). Each one is a cache that
    its own reader validates, and no decision depends on an entry being present, so a write
    that cannot happen at once is skipped with a debug log naming ``what`` instead of failing
    or holding up the command: another process holding the write lock for longer than
    ``timeout`` (2 s, a fraction of a pivot write's own 30 s; 0 makes one attempt), a database
    this user cannot write, a read-only filesystem, a full map. Without ``create``, a state
    directory that has no database is left without one.
    """
    if not create and not (state_dir / "state.lmdb").exists():
        return False
    try:
        with StateDB(state_dir, write_timeout=timeout) as state_db:
            write(state_db)
    except (exceptions.PivotDBWriteTimeoutError, StateDBError, lmdb.Error, OSError) as error:
        logger.debug("Not saving %s in %s: %s", what, state_dir, error)
        return False
    return True


def write_back_file_hashes(
    state_dir: pathlib.Path,
    entries: list[FileHashEntry],
    *,
    timeout: float = _WRITE_BACK_TIMEOUT,
) -> None:
    """Save the file-hash entries a read-only check computed, best-effort, in one transaction.

    ``pivot status``, every explanation, ``pivot diff``, ``pivot plots diff``, the engine's
    skip check and the engine of a ``--no-commit`` run call this with the hashes they had to
    read, so the next check reads none of those files (status's tracked files, ``pivot diff``
    and ``pivot plots diff`` through ``cache.hash_read_only``). The write is best-effort and
    creates no database (``save_best_effort``); with no entries it takes no lock.
    """
    if not entries:
        return
    # A write-back can land after a concurrent `pivot repro` recorded a newer entry for one of
    # these paths: it puts back the stat it read and drops the index entry of the inode repro
    # recorded. Both then fail the next lookup's stat match, so that file is read once more; no
    # wrong hash is served.
    _ = save_best_effort(
        state_dir,
        lambda state_db: state_db.save_file_hash_entries(entries),
        what=f"{len(entries)} file hashes",
        create=False,
        timeout=timeout,
    )
