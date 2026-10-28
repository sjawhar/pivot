from __future__ import annotations

import contextlib
import errno
import fcntl
import hashlib
import json
import logging
import mmap
import operator
import os
import pathlib
import posixpath
import secrets
import shutil
import stat
import tempfile
import time
from typing import TYPE_CHECKING, NewType, TypedDict

import xxhash

from pivot import exceptions, metrics
from pivot import state_dir as state_dir_mod
from pivot.config import CheckoutMode as CheckoutMode
from pivot.types import DirHash, DirManifestEntry, FileHash, FileHashEntry, HashInfo, is_dir_hash

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Mapping, Sequence

    from pivot.storage import state as state_mod

CHUNK_SIZE = 1024 * 1024  # 1MB chunks for hashing
MMAP_THRESHOLD = 10 * 1024 * 1024  # 10MB - use mmap for files larger than this
XXHASH64_HEX_LENGTH = 16  # xxhash64 produces 64-bit hash = 16 hex characters

_RESTORE_TEMP_PREFIX = ".pivot_restore_"
_BACKUP_TEMP_PREFIX = ".pivot_backup_"
_RESTORE_LOCK_PREFIX = ".pivot_restore_lock_"
_RESTORE_TEMP_MAX_AGE = 3600  # 1 hour


def _restore_target_token(target_path: pathlib.Path) -> str:
    """Return a stable token identifying the logical workspace target.

    ``os.fsencode`` keeps a name that is not UTF-8; for every UTF-8 path it gives the bytes
    ``str.encode`` did, so no existing token changes.
    """
    logical_path = target_path.parent.resolve() / target_path.name
    return hashlib.sha256(os.fsencode(logical_path)).hexdigest()[:32]


def restore_staging_path(path: pathlib.Path, token: str | None = None) -> pathlib.Path:
    """Return the deterministic staging path for a file restore."""
    return path.parent / f"{_RESTORE_TEMP_PREFIX}{token or _restore_target_token(path)}.file"


def _get_lock_path_from_token(token: str, state_dir: pathlib.Path) -> pathlib.Path:
    """Get a restore lock path from a previously resolved target token."""
    lock_dir = state_dir / "locks"
    state_dir_mod.make_dirs(lock_dir)
    return lock_dir / f"{_RESTORE_LOCK_PREFIX}{token}"


def _get_lock_path(
    target_path: pathlib.Path, state_dir: pathlib.Path, *, token: str | None = None
) -> pathlib.Path:
    """Get lock file path under state_dir/locks/ for a logical restore target.

    The final path component is intentionally not resolved: a restored symlink must
    retain the same lock as the missing workspace path it replaced.
    """
    return _get_lock_path_from_token(token or _restore_target_token(target_path), state_dir)


@contextlib.contextmanager
def restore_lock(path: pathlib.Path, state_dir: pathlib.Path | None) -> Generator[str | None]:
    """Hold the exclusive restore lock for path and yield its target token.

    No lock is taken when state_dir is None. The lock file is never deleted. Unlinking it
    would let a new restorer lock a fresh file while a waiter still holds the old one.
    """
    if state_dir is None:
        yield None
        return
    token = _restore_target_token(path)
    lock_fd = os.open(_get_lock_path(path, state_dir, token=token), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        yield token
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def atomic_write_file(
    dest: pathlib.Path,
    write_fn: Callable[[int], None],
    mode: int = 0o644,
    *,
    staging_path: pathlib.Path | None = None,
    before_swap: Callable[[pathlib.Path], None] | None = None,
) -> os.stat_result:
    """Atomically write to dest using a temporary file and replacement.

    A caller holding a restore lock can supply its target-tagged staging path so a
    later restore of the same target cleans up a process killed before replacement.

    Args:
        dest: Target file path.
        write_fn: Function that receives the file descriptor and writes content.
                  MUST close fd (typically via os.fdopen which takes ownership).
        mode: File permissions (default 0o644).
        staging_path: Optional deterministic temporary path beside ``dest``.
        before_swap: Checks the entry at ``dest``, once the file is written, just before the
            file replaces it (``restore_from_cache``); not called when nothing is there.

    Returns:
        The written file's stat, taken before it replaced ``dest`` (the same inode), so a
        write after the replacement does not match it.
    """
    state_dir_mod.make_dirs(dest.parent)
    if staging_path is None:
        fd, tmp_path = tempfile.mkstemp(dir=dest.parent, suffix=".tmp")
        tmp = pathlib.Path(tmp_path)
    else:
        tmp = staging_path
        _clear_path(tmp)
        fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    fd_closed = False
    try:
        write_fn(fd)
        fd_closed = True  # write_fn took ownership via os.fdopen
        os.chmod(tmp, mode)
        if before_swap is not None and (dest.exists() or dest.is_symlink()):
            before_swap(dest)
        written = _fstat_staged(tmp)
        tmp.replace(dest)
    except BaseException:
        _clear_path(tmp)
        raise
    finally:
        # Only close fd if write_fn didn't (e.g., exception before os.fdopen)
        if not fd_closed:
            with contextlib.suppress(OSError):
                os.close(fd)
    return written


def _hash_contents(path: str | os.PathLike[str], size: int) -> str:
    """Return the xxhash64 hex digest of the file at ``path``, whose size is ``size``."""
    with open(path, "rb", buffering=0) as f:
        if size >= MMAP_THRESHOLD:
            try:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    return xxhash.xxh64(mm).hexdigest()
            except (ValueError, OSError):
                pass  # mmap fails on an empty file or some network filesystems: read instead
        hasher = xxhash.xxh64()
        while chunk := f.read(CHUNK_SIZE):
            hasher.update(chunk)
        return hasher.hexdigest()


def hash_file(
    path: pathlib.Path,
    state_db: state_mod.StateDB | None = None,
    *,
    pending: list[tuple[pathlib.Path, os.stat_result, str]] | None = None,
) -> tuple[str, os.stat_result]:
    """Compute xxhash64 of file contents, using state cache if available.

    A hash read from the file is saved to a writable ``state_db`` at once, best-effort (a failed
    write is logged, never raised), unless ``pending`` is given: then ``(path, stat, hash)`` is
    appended to it instead, for the caller to save in one transaction, or, when ``state_db`` is
    read-only, to hand to a process that can write.

    Returns:
        Tuple of (file_hash, file_stat) where file_hash is the xxhash64 hex digest
        and file_stat is the os.stat_result for the file.
    """
    file_stat = path.stat()

    if state_db is not None:
        cached = state_db.get(path, file_stat)
        if cached is not None:
            return cached, file_stat

    _t = metrics.start()
    file_hash = _hash_contents(path, file_stat.st_size)
    metrics.end("cache.hash_file", _t)

    if pending is not None:
        pending.append((path, file_stat, file_hash))
    elif state_db is not None and not state_db.readonly:
        _ = state_db.save_many_best_effort([(path, file_stat, file_hash)])

    return file_hash, file_stat


def known_file_hashes(
    path: pathlib.Path, output_hash: HashInfo
) -> list[tuple[pathlib.Path, os.stat_result, str]]:
    """StateDB file-hash entries for an output a stage just committed, whose hash is known.

    Stats each file (each manifest entry, for a directory) and pairs it with its known hash;
    nothing is read. Call it once the output has its final inode: a hardlink checkout
    replaces the file the stage wrote. A restore records what it staged instead
    (``restore_from_cache``'s ``recorded``).

    This lookup only saves work: a path whose stat (a symlink whose ``readlink`` or target
    ``stat`` raises EIO, ESTALE or EACCES) fails gets no entry, so the next status, explanation
    or run hashes it the normal way. It never fails the command that committed the output.
    """
    if is_dir_hash(output_hash):
        files = [(path / entry["relpath"], entry["hash"]) for entry in output_hash["manifest"]]
    else:
        files = [(path, output_hash["hash"])]
    entries = list[tuple[pathlib.Path, os.stat_result, str]]()
    for file_path, file_hash in files:
        try:
            file_stat = file_path.stat()
        except OSError as e:
            logger.debug("No known-hash entry for %s; its stat failed: %s", file_path, e)
            continue
        entries.append((file_path, file_stat, file_hash))
    return entries


def deferred_file_hash_entries(
    entries: Iterable[tuple[pathlib.Path, os.stat_result, str]],
) -> list[FileHashEntry]:
    """``entries`` as ``DeferredWrites`` file-hash entries, for a process that cannot write the
    StateDB to hand to one that can.

    Each path is made absolute here, so the writer keys it the same from any working
    directory. The writer leaves out an entry whose key exceeds LMDB's limit.
    """
    return [
        (
            os.path.abspath(path),
            file_stat.st_mtime_ns,
            file_stat.st_size,
            file_stat.st_dev,
            file_stat.st_ino,
            file_hash,
        )
        for path, file_stat, file_hash in entries
    ]


# ``os.path.realpath`` of a cache's object root (``<cache.dir>/files``), from
# ``resolve_cache_root``. Every component is real, so ``<root>/XX/NAME`` is its own real path
# unless ``XX`` or ``NAME`` is a symlink (``_is_real_object_path``).
CacheRoot = NewType("CacheRoot", str)


def resolve_cache_root(cache_dir: pathlib.Path) -> CacheRoot:
    """Return ``cache_dir``'s real path, the root the cache-object checks take. A caller checking
    or restoring many links resolves it once per walk or restore, so a cache directory re-pointed
    between commands is seen by the next one."""
    return CacheRoot(os.path.realpath(cache_dir))


def _is_real_object_path(object_path: str) -> bool:
    """Whether ``<cache_root>/XX/NAME`` is its own real path. Every component of ``cache_root``
    is real, so only ``XX`` or ``NAME`` can redirect it: two lstats replace realpath's walk of
    every component. Testing the text alone would not do: a link whose text lies in the cache
    still leaves it when ``XX`` or ``NAME`` is a symlink out of it."""
    return not os.path.islink(os.path.dirname(object_path)) and not os.path.islink(object_path)


def _resolve_link(link: str | os.PathLike[str], cache_root: CacheRoot) -> str:
    """Return ``os.path.realpath(link)`` for the symlink ``link``; raise OSError when ``link`` is
    not a symlink or is gone.

    realpath lstats every component of the link's path and of its target, in Python: at 180,000
    links that was most of the time a status took after a symlink checkout. A checkout link
    holds ``<cache_root>/XX/NAME``, the object's real path (``_stage_checkout``); for that text
    ``_is_real_object_path`` proves the answer. Any other text (relative, with ``..``, a link to
    another link, a path outside the cache) is resolved in full, so the answer is realpath's for
    every link. The link is read again on every call.
    """
    target = os.readlink(link)
    prefix = os.path.join(cache_root, "")
    if target.startswith(prefix):
        prefix_dir, sep, name = target[len(prefix) :].partition("/")
        if (
            sep
            and prefix_dir not in ("", ".", "..")
            and name not in ("", ".", "..")
            and "/" not in name
            and _is_real_object_path(target)
        ):
            return target
    return os.path.realpath(link)


def _is_configured_cache_object_path(cache_path: pathlib.Path, cache_root: CacheRoot) -> bool:
    """Whether ``cache_path``'s own object root is ``cache_root`` without resolving every
    component. A #503 fallback can point at legacy ``.pivot/cache/files`` while the configured
    cache root is elsewhere; its path must be resolved in full, not rewritten under ``cache_root``.
    """
    return os.path.abspath(cache_path.parent.parent) == cache_root


def _links_to_object(link: pathlib.Path, cache_path: pathlib.Path, cache_root: CacheRoot) -> bool:
    """Whether the symlink ``link`` resolves where the object path ``cache_path`` resolves, as
    ``link.resolve() == cache_path.resolve()`` answers. ``cache_path`` is
    ``get_cache_path(cache_dir, key)`` for the ``cache_dir`` ``cache_root`` resolves.

    A link whose text is ``<cache_root>/XX/YYYY...``, the text a checkout writes for an object
    whose own root is ``cache_root``, resolves to that text when ``_is_real_object_path`` holds.
    Otherwise the two can resolve apart (an object name linked back to the member stops each
    resolve() at a different link), so that link, like any other, is resolved in full.
    """
    real = os.path.join(cache_root, cache_path.parent.name, cache_path.name)
    try:
        if (
            _is_configured_cache_object_path(cache_path, cache_root)
            and os.readlink(link) == real
            and _is_real_object_path(real)
        ):
            return True
    except OSError:
        pass
    return link.resolve() == cache_path.resolve()


# What ``Path.is_symlink()`` and ``Path.is_file()`` read as "no such file" on Python 3.13
# (pathlib's ``_IGNORED_ERRNOS``), and readlink's answer for a path that is not a symlink. Any
# other error (EACCES, EIO, ESTALE) leaves a link unclassified, a readlink error after a good
# lstat included (non-strict realpath would hide that one): the checks below raise
# ``UnreadableLinkError`` rather than call the link a user symlink and record a partial hash.
# It is not an OSError, so a caller that reads an OSError as "missing" or "changed" (an output
# check that would restore over the link) does not catch it. They test errno themselves, not
# through pathlib: from Python 3.14 ``Path.is_file()`` is ``os.path.isfile``, which hides every
# error: https://github.com/python/cpython/blob/v3.14.0/Lib/pathlib/__init__.py#L690-L696
_NO_FILE_ERRNO = frozenset({errno.ENOENT, errno.ENOTDIR, errno.EBADF, errno.ELOOP})
_NOT_A_LINK_ERRNO = _NO_FILE_ERRNO | {errno.EINVAL}


def _symlink_cache_key(path: str | os.PathLike[str], cache_root: CacheRoot) -> str | None:
    """Return ``XX + YYYY...`` when the symlink ``path`` resolves to ``<cache_root>/XX/YYYY...``
    with 16 characters in all, else None (``path`` is not a symlink, or resolves elsewhere).
    Any other error reading ``path`` raises ``UnreadableLinkError`` (``_NOT_A_LINK_ERRNO``).
    """
    try:
        target = _resolve_link(path, cache_root)
    except OSError as e:
        if e.errno not in _NOT_A_LINK_ERRNO:
            raise exceptions.UnreadableLinkError(os.fspath(path), e.strerror or str(e)) from e
        return None
    except ValueError:
        return None
    prefix = os.path.join(cache_root, "")
    if not target.startswith(prefix):
        return None
    parts = target[len(prefix) :].split("/")
    if len(parts) != 2:
        return None
    key = parts[0] + parts[1]
    return key if len(key) == XXHASH64_HEX_LENGTH else None


def _cache_object_key(path: str | os.PathLike[str], cache_root: CacheRoot | None) -> str | None:
    """Return the key of the cache object the symlink ``path`` stands for, or None.

    A symlink is the cache object ``K`` when following it, through every link on the way, ends at
    a regular file whose real path is ``<cache_root>/K[:2]/K[2:]``, and ``K`` is a cache key. So
    a link to a link to the object, and a relative link, count, as they did when this resolved
    the whole path with ``resolve()``; ``_resolve_link`` gives the same answer cheaply for the
    link a checkout writes. A link that ends outside the cache (its first step inside it or not),
    at another name in it, at a directory, at nothing, or in a loop is a user symlink, even when
    it ends at a file named like a hash. A file-by-file symlink checkout restores each directory
    member as a link to its object. ``cache_root`` is None for no cache. An error that leaves
    the link unclassified raises ``UnreadableLinkError`` (``_NO_FILE_ERRNO``).
    """
    if cache_root is None:
        return None
    key = _symlink_cache_key(path, cache_root)
    if key is None or not is_valid_hash(key):
        return None
    try:
        is_file = stat.S_ISREG(os.stat(path).st_mode)
    except OSError as e:
        if e.errno not in _NO_FILE_ERRNO:
            raise exceptions.UnreadableLinkError(os.fspath(path), e.strerror or str(e)) from e
        return None
    return key if is_file else None


def is_artifact_member_content(
    path: pathlib.Path, file_hash: str, cache_root: CacheRoot | None
) -> bool:
    """Whether a candidate member has the content its cache-object symlink names; ``cache_root``
    is None for no cache."""
    return not path.is_symlink() or file_hash == _cache_object_key(path, cache_root)


def walk_artifact_members(
    directory: str,
    prefix: str,
    cache_root: CacheRoot | None,
    files: list[tuple[str, os.DirEntry[str]]],
    dirs: list[str],
    left_out: list[pathlib.Path],
) -> None:
    """Collect every stored file under ``directory`` with its relpath, ``prefix`` + its name.

    A declared directory artifact is hashed and cached whole: no name is ignored, so a change
    to any file in a directory input reruns its consumers, and every file in a directory output
    survives cache and restore. Ignore rules belong to code fingerprinting and watch mode.
    A symlink to an object in the cache (left by a symlink checkout; see ``_cache_object_key``)
    counts as the file it points to; any other symlink, a symlinked directory included, is not
    stored and is appended to ``left_out``. The walk never follows a symlink, so every regular
    file it collects lies inside the directory. An error that leaves an entry unclassified
    propagates, since a partial hash would be wrong; for a symlink it is ``UnreadableLinkError``
    (EIO, EACCES; see ``_NO_FILE_ERRNO``).

    Args:
        directory: Directory to scan
        prefix: Relpath of ``directory`` under the artifact root, with a trailing ``/`
            (empty at the root)
        cache_root: The cache object root whose links count as content, resolved once for the
            whole walk, or None for no cache
        files: Collects (relpath, entry) for every stored file
        dirs: Collects the relpath of every directory under the root
        left_out: Collects the symlinks that are not part of the artifact
    """
    with os.scandir(directory) as entries:
        for entry in entries:
            relpath = prefix + entry.name
            if entry.is_symlink():
                if _cache_object_key(entry.path, cache_root) is not None:
                    files.append((relpath, entry))
                else:
                    left_out.append(pathlib.Path(entry.path))
            elif entry.is_file():
                files.append((relpath, entry))
            elif entry.is_dir():
                dirs.append(relpath)
                walk_artifact_members(entry.path, relpath + "/", cache_root, files, dirs, left_out)


def _parent_dirs(relpaths: Iterable[str]) -> set[str]:
    """Every directory above each relpath, short of the root, as relpaths."""
    return {
        str(parent) for relpath in relpaths for parent in pathlib.PurePath(relpath).parents[:-1]
    }


def _recorded_empty_dirs(output_hash: DirHash) -> list[str]:
    return output_hash["empty_dirs"] if "empty_dirs" in output_hash else []


# The fields pivot writes for each manifest entry, and the only ones a manifest is compared by.
_MANIFEST_ENTRY_FIELDS = ("hash", "isexec", "relpath", "size")


def _manifest_json(manifest: Sequence[Mapping[str, object]], empty_dirs: list[str]) -> bytes:
    """The JSON ``hash_directory`` names a directory by (its XXH64 is the directory's hash)."""
    hashed: object = {"empty_dirs": empty_dirs, "manifest": manifest} if empty_dirs else manifest
    return json.dumps(hashed, sort_keys=True, separators=(",", ":")).encode()


def _canonical_manifest(output_hash: DirHash) -> bytes | None:
    """``output_hash``'s manifest as ``_manifest_json`` encodes it, from only the fields pivot
    writes for each entry (``_MANIFEST_ENTRY_FIELDS``); a field pivot does not write is left out.
    Entries are sorted by relpath and empty directories sorted, the order ``hash_directory``
    writes, so one set of entries has one canonical form whatever order a .pvt or lock lists
    them in: no ordering can mark a tree for a manifest other than its own.

    None when an entry lacks one of those fields or holds a value JSON cannot encode, as a
    hand-written .pvt or lock can (``when: 2026-01-01``), or nests too deep to encode, and when two
    entries name one relpath or an empty directory is listed twice, which ``hash_directory`` never
    writes. No tree is trusted for such a manifest, so its restore goes file by file, which reads
    none of those values.
    """
    try:
        entries = [
            {field: entry[field] for field in _MANIFEST_ENTRY_FIELDS}
            for entry in output_hash["manifest"]
        ]
        entries.sort(key=operator.itemgetter("relpath"))
        empty_dirs = sorted(_recorded_empty_dirs(output_hash))
        if len({entry["relpath"] for entry in entries}) < len(entries):
            return None
        if len(set(empty_dirs)) < len(empty_dirs):
            return None
        return _manifest_json(entries, empty_dirs)
    except (KeyError, TypeError, ValueError, RecursionError):
        return None


def _restored_dirs(output_hash: DirHash) -> set[str]:
    """Every directory a restore of ``output_hash`` creates under its root, as relpaths."""
    empty_dirs = _recorded_empty_dirs(output_hash)
    file_parents = _parent_dirs(entry["relpath"] for entry in output_hash["manifest"])
    return file_parents | _parent_dirs(empty_dirs) | set(empty_dirs)


def _empty_dir_path(root: pathlib.Path, relpath: str) -> pathlib.Path:
    """Return where the recorded empty directory ``relpath`` belongs under ``root``.

    ``relpath`` may come straight from a lock file, so one leading outside ``root`` is refused,
    as the file-by-file restore refuses a manifest entry's.
    """
    target = root / relpath
    if not target.resolve().is_relative_to(root.resolve()):
        raise exceptions.SecurityValidationError(f"Manifest contains path traversal: {relpath!r}")
    return target


def log_left_out_symlinks(path: pathlib.Path, left_out: list[pathlib.Path]) -> None:
    """Log one warning naming every symlink a directory artifact does not store."""
    absolute = path.absolute()
    cwd = pathlib.Path.cwd()
    shown = absolute.relative_to(cwd) if absolute.is_relative_to(cwd) else path
    names = ", ".join(sorted(str(link.relative_to(path)) for link in left_out))
    one = len(left_out) == 1
    logger.warning(
        "Directory artifact %s leaves out %s %s (a symlink inside a directory artifact is not "
        + "stored); replace %s with a real file or directory",
        shown,
        "symlink" if one else "symlinks",
        names,
        "it" if one else "each",
    )


def hash_directory(
    path: pathlib.Path,
    state_db: state_mod.StateDB | None = None,
    *,
    cache_dir: pathlib.Path | None = None,
    warn_left_out_symlinks: bool = False,
    pending: list[tuple[pathlib.Path, os.stat_result, str]] | None = None,
) -> DirHash:
    """Compute the tree hash of a directory, returning it with the manifest.

    The hash covers the manifest and the directory's empty directories (see ``DirHash``). With
    no empty directories it hashes the manifest alone, so recording them changed no hash of a
    directory that holds none.

    Symlink handling:
    - A symlink to an object in ``cache_dir`` is hashed by its target's content under its own
      relpath; the content must match the object's name
    - Any other symlink inside the directory is left out of the manifest
    - Base path may be a symlink; the walk below it follows none

    With a ``state_db``, every file's cached hash is looked up in one read transaction. The
    hashes read from files are saved to a writable ``state_db`` in one write transaction, unless
    ``pending`` is given: then they are appended to it, as ``hash_file`` does.

    Args:
        path: Directory to hash
        state_db: Optional StateDB for caching file hashes
        cache_dir: The configured cache's object root (``<cache.dir>/files``). Pass it
            wherever one exists, so recording and re-hashing agree on symlink checkouts;
            None means no cache, and then no symlink is content.
        warn_left_out_symlinks: Log one warning naming the symlinks left out. Set where a
            manifest is recorded (a stage output, a tracked directory), not on every re-hash.
        pending: Receives ``(path, stat, hash)`` for each file whose hash was read, instead of
            ``state_db``.
    """
    _t = metrics.start()
    files = list[tuple[str, os.DirEntry[str]]]()
    dirs = list[str]()
    left_out = list[pathlib.Path]()
    cache_root = None if cache_dir is None else resolve_cache_root(cache_dir)
    walk_artifact_members(os.fspath(path), "", cache_root, files, dirs, left_out)
    files.sort(key=operator.itemgetter(0))

    stats = list[tuple[str, os.DirEntry[str], os.stat_result]]()
    for relpath, entry in files:
        try:
            stats.append((relpath, entry, entry.stat()))
        except FileNotFoundError:
            continue  # File deleted between scan and hash

    cached: list[str | None]
    if state_db is not None:
        lookups = [(pathlib.Path(entry.path), file_stat) for _, entry, file_stat in stats]
        found = state_db.get_many(lookups)
        cached = [found[file_path] for file_path, _ in lookups]
    else:
        cached = [None] * len(stats)

    # The hashes read below go to the caller's ``pending``, or are saved here in one write.
    saving = pending is None and state_db is not None and not state_db.readonly
    read = list[tuple[pathlib.Path, os.stat_result, str]]() if saving else pending
    manifest = list[DirManifestEntry]()
    for (relpath, entry, file_stat), file_hash in zip(stats, cached, strict=True):
        if file_hash is None:
            try:
                file_hash = _hash_contents(entry.path, file_stat.st_size)
            except FileNotFoundError:
                continue  # File deleted between scan and hash
            if read is not None:
                read.append((pathlib.Path(entry.path), file_stat, file_hash))
        if entry.is_symlink() and not is_artifact_member_content(
            pathlib.Path(entry.path), file_hash, cache_root
        ):
            left_out.append(pathlib.Path(entry.path))
            continue
        manifest.append(
            DirManifestEntry(
                relpath=relpath,
                hash=file_hash,
                size=file_stat.st_size,
                isexec=bool(file_stat.st_mode & stat.S_IXUSR),
            )
        )

    if saving and read and state_db is not None:
        _ = state_db.save_many_best_effort(read)
    if left_out and warn_left_out_symlinks:
        log_left_out_symlinks(path, left_out)

    # A deepest empty directory holds no stored file and no other directory.
    found_dirs = set(dirs)
    file_dirs = {posixpath.dirname(entry["relpath"]) for entry in manifest}
    has_child = {posixpath.dirname(directory) for directory in found_dirs}
    empty_dirs = sorted(found_dirs - file_dirs - has_child)

    output_hash = DirHash(
        hash=xxhash.xxh64(_manifest_json(manifest, empty_dirs)).hexdigest(), manifest=manifest
    )
    if empty_dirs:
        output_hash["empty_dirs"] = empty_dirs

    metrics.end("cache.hash_directory", _t)
    return output_hash


_VALID_HASH_CHARS = frozenset("0123456789abcdef")


def is_valid_hash(value: object) -> bool:
    """Whether ``value`` is a cache key: an xxhash64 digest, 16 lowercase hex characters."""
    return (
        isinstance(value, str)
        and len(value) == XXHASH64_HEX_LENGTH
        and all(c in _VALID_HASH_CHARS for c in value)
    )


def get_cache_path(cache_dir: pathlib.Path, file_hash: str) -> pathlib.Path:
    """Return the configured cache path for a hash, for writers and primary-cache reads."""
    if len(file_hash) != XXHASH64_HEX_LENGTH:
        raise exceptions.SecurityValidationError(
            f"Hash must be exactly {XXHASH64_HEX_LENGTH} characters, got {len(file_hash)}"
        )
    if not is_valid_hash(file_hash):
        raise exceptions.SecurityValidationError(f"Hash contains invalid characters: {file_hash!r}")
    return cache_dir / file_hash[:2] / file_hash[2:]


# The errnos for which `Path.exists()` reports a path as missing: CPython 3.13
# `pathlib/_abc.py:31` (`_IGNORED_ERRNOS`). A prefix that is a regular file (ENOTDIR) or a symlink
# loop (ELOOP) makes the object missing, as it did on 0.5.2; any other error, such as EACCES, raises.
_ABSENT_ERRNOS = frozenset({errno.ENOENT, errno.ENOTDIR, errno.EBADF, errno.ELOOP})


def is_cache_object_path(path: pathlib.Path, *, directory: bool = False) -> bool:
    """Whether a resolver or scan may treat a final cache file or tree root as present."""
    try:
        mode = os.lstat(path).st_mode
    except OSError as e:
        if e.errno in _ABSENT_ERRNOS:
            return False
        raise
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected(mode):
        kind = "directory tree" if directory else "regular file"
        logger.debug("Ignoring cache object %s: expected %s", path, kind)
        return False
    return True


def resolve_cache_object_path(
    cache_dir: pathlib.Path, file_hash: str, *, directory: bool = False
) -> pathlib.Path | None:
    """Return an existing regular object or requested tree root, or ``None`` when absent."""
    cache_path = get_cache_path(cache_dir, file_hash)
    if is_cache_object_path(cache_path, directory=directory):
        return cache_path
    other_files_dir = state_dir_mod.other_cache_files_dir_for(cache_dir)
    if other_files_dir is not None:
        other_path = get_cache_path(other_files_dir, file_hash)
        if is_cache_object_path(other_path, directory=directory):
            state_dir_mod.report_legacy_state_dir(cache_dir.parent.parent)
            return other_path
    return None


def _make_writable_and_retry(func: Callable[[str], object], path: str, exc: BaseException) -> None:
    """onexc handler for rmtree: make parent directory writable before retrying.

    IMPORTANT: We only chmod directories, never files. Files may be hardlinks to
    the cache, and chmod would corrupt the cache's read-only permissions (hardlinks
    share the same inode, so chmod affects both the output file and cache file).

    To delete a file, you only need write permission on the parent directory.
    To delete a directory, you need write permission on the parent directory.
    """
    # Make parent directory writable so we can modify its contents
    parent = os.path.dirname(path)
    if parent:
        try:
            parent_perm = os.lstat(parent).st_mode
            if not (parent_perm & stat.S_IWUSR):
                os.chmod(parent, parent_perm | stat.S_IWUSR)
        except OSError:
            pass  # Best effort - may not own parent

    # For directories only: make writable so we can delete contents
    # Never chmod files - they may be hardlinks to read-only cache
    try:
        st = os.lstat(path)
        if stat.S_ISDIR(st.st_mode) and not (st.st_mode & stat.S_IWUSR):
            os.chmod(path, st.st_mode | stat.S_IWUSR)
    except OSError as chmod_exc:
        if chmod_exc.errno not in (errno.ENOENT, errno.EPERM):
            raise exc from chmod_exc

    if func is os.open:
        # shutil.rmtree's fd walk hands os.open here when it cannot open a directory
        # (os.open(name, flags, dir_fd=...)); retrying it as func(path) omits the flags argument
        # and raises TypeError. A restore refuses an unreadable directory before any delete
        # reaches rmtree, so this is unreachable there; raise the real error if another caller
        # lands here, never a TypeError.
        raise exc
    func(path)


def _clear_path(path: pathlib.Path) -> None:
    """Remove file, symlink, or directory at path if it exists.

    IMPORTANT: Never chmod files before deletion - they may be hardlinks to the
    cache, and chmod would corrupt the cache's read-only permissions.
    """
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, onexc=_make_writable_and_retry)
    else:
        try:
            path.unlink()
        except PermissionError as perm_err:
            # Make parent directory writable, not the file itself.
            # Files may be hardlinks to read-only cache - chmod would corrupt cache.
            parent = path.parent
            try:
                parent_perm = parent.stat().st_mode
            except FileNotFoundError:
                # Parent was deleted concurrently - re-raise original error
                raise perm_err from None
            if not (parent_perm & stat.S_IWUSR):
                os.chmod(parent, parent_perm | stat.S_IWUSR)
            path.unlink()


def _is_real_dir(path: pathlib.Path) -> bool:
    """Whether ``path`` is a directory, not a symlink to one."""
    return path.is_dir() and not path.is_symlink()


def _cleanup_stale_restore_temps(parent_dir: pathlib.Path) -> None:
    """Remove stale restore temporaries (staged replacements) older than max age.

    It takes no lock, so it never removes a backup (``_BACKUP_TEMP_PREFIX``): a backup is the
    entry a restore renamed aside, which keeps its own old mtime and may still be under that
    restore's check. Only the next directory restore of the same path removes a backup, under
    that path's restore lock (``_cleanup_fresh_restore_orphans``); a file restore removes none.
    """
    cutoff = time.time() - _RESTORE_TEMP_MAX_AGE

    if not parent_dir.exists():
        return

    for entry in os.scandir(parent_dir):
        if entry.name.startswith(_RESTORE_LOCK_PREFIX):
            continue
        if not entry.name.startswith(_RESTORE_TEMP_PREFIX):
            continue
        try:
            entry_stat = entry.stat(follow_symlinks=False)
            if entry.is_dir(follow_symlinks=False):
                age = entry_stat.st_mtime
            elif stat.S_ISREG(entry_stat.st_mode):
                age = entry_stat.st_ctime
            else:
                continue
            if age < cutoff:
                logger.debug(f"Cleaning up stale temp: {entry.path}")
                _clear_path(pathlib.Path(entry.path))
        except OSError as e:
            logger.debug(f"Failed to clean up temp {entry.path}: {e}")


def _cleanup_fresh_restore_orphans(
    path: pathlib.Path,
    token: str,
    before_swap: Callable[[pathlib.Path], None] | None = None,
) -> None:
    """Remove fresh temporaries from a killed restore under ``path``'s restore lock.

    A backup is the old entry a killed directory restore had renamed aside. Before deleting one,
    the same swap check the pending restore would run must prove that it holds no user file. A
    refusal keeps the backup and warns with the refusal, which names the user file.
    """
    restore_prefix = f"{_RESTORE_TEMP_PREFIX}{token}."
    backup_prefix = f"{_BACKUP_TEMP_PREFIX}{token}."
    for entry in os.scandir(path.parent):
        if not entry.name.startswith((restore_prefix, backup_prefix)):
            continue
        orphan = pathlib.Path(entry.path)
        if entry.name.startswith(backup_prefix) and before_swap is not None:
            try:
                before_swap(orphan)
            except exceptions.OutputOwnershipError as error:
                logger.warning("Keeping interrupted restore backup %s: %s", orphan, error)
                continue
        _clear_path(orphan)


def _get_symlink_cache_hash(path: pathlib.Path, cache_dir: pathlib.Path) -> str | None:
    """Extract hash from symlink target if it points to cache, else None.

    For a caller that checks one path: it resolves ``cache_dir`` itself. A caller that checks
    many links resolves the cache directory once and calls ``_symlink_cache_key``.
    """
    if not path.is_symlink():
        return None
    try:
        cache_root = resolve_cache_root(cache_dir)
    except OSError:  # a relative cache_dir under a working directory that is gone
        return None
    return _symlink_cache_key(path, cache_root)


# link() errors on filesystems without hard links: vfat, exFAT, many SMB and FUSE mounts
# (EPERM, ENOTSUP), and FUSE mounts without a link handler on Linux before 6.15 (ENOSYS).
_NO_HARDLINK_ERRNO = frozenset({errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS})


def publish_cache_object(
    staged: str | os.PathLike[str], cache_path: str | os.PathLike[str]
) -> bool:
    """Publish the complete file ``staged`` as the cache object ``cache_path`` unless an object
    is already published there; return whether ``staged`` was published.

    Objects are content-addressed, so one already there holds the same bytes. It is never
    replaced: a checkout hardlinking it has looked ``cache_path`` up and is about to link the
    inode it found, and Linux refuses that link with ENOENT once a rename over ``cache_path``
    has taken the inode's last name, though the name itself never went missing. A symlink
    checkout reads the object's executable bit through the name, so replacing it with a copy
    of another mode would change that checkout's file. link() publishes without replacing.
    The caller removes ``staged`` afterwards.
    """
    try:
        os.link(staged, cache_path)
    except FileExistsError:
        return False
    except OSError as exc:
        if exc.errno not in _NO_HARDLINK_ERRNO:
            raise
        # No hard links here, so nothing can be hardlinking the object either:
        # rename unless another process already published it.
        if os.path.exists(cache_path):
            return False
        os.replace(staged, cache_path)
    return True


# Every temporary entry pivot makes in a cache prefix directory carries this prefix.
_CACHE_TEMP_PREFIX = ".pivot_tmp_"
# Taken shared by every process while its temporary entries exist in the prefix directory, and
# exclusive by the sweep. Never a cache key (cache scans skip it), and never deleted.
_PREFIX_LOCK = ".pivot_cache.lock"
# The sweep of one prefix directory runs at most this often; the lock file's mtime records it.
_CACHE_SWEEP_INTERVAL = 3600
# Opening or creating a lock fails this way in a cache directory this user cannot write.
_CANNOT_WRITE_ERRNO = frozenset({errno.EACCES, errno.EPERM, errno.EROFS})
# flock() fails this way on a filesystem that refuses it: Lustre mounted without `flock`
# (ENOSYS), NFS when its lock manager does not answer (ENOLCK), some FUSE filesystems.
_NO_FLOCK_ERRNO = frozenset({errno.ENOLCK, errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTSUP})


def _open_cache_lock(path: pathlib.Path) -> int:
    """Open the lock file at ``path`` in a cache directory, creating it if needed.

    A cache directory can be shared by several users. The file is created writable by exactly
    those who can write its directory (they could delete it anyway), so each of them can take
    an exclusive lock on it, which NFS allows only on a file opened for writing. A lock file
    this user cannot write is opened read-only: a local filesystem locks it all the same. A new
    lock file's mtime is the epoch: its directory has never been swept.
    """
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o644)
    except FileExistsError:
        try:
            return os.open(path, os.O_RDWR)
        except PermissionError:
            return os.open(path, os.O_RDONLY)
    with contextlib.suppress(OSError):
        os.fchmod(fd, 0o444 | (path.parent.stat().st_mode & 0o222))
    with contextlib.suppress(OSError):
        os.utime(fd, (0, 0))
    return fd


def _sweep_cache_temps(prefix_dir: pathlib.Path) -> None:
    """Remove the temporary entries killed processes left in prefix_dir.

    The caller holds prefix_dir's lock exclusively, so no ``_CACHE_TEMP_PREFIX`` entry has a
    live owner. An older pivot's temporaries (``tmp*.tmp`` files, ``.pivot_download_*``) were
    made without the lock; one is removed once it is untouched for ``_CACHE_SWEEP_INTERVAL``,
    as a write in progress keeps its ctime current. An entry the sweep cannot remove (another
    user's) is reported and left.
    """
    cutoff = time.time() - _CACHE_SWEEP_INTERVAL
    for entry in os.scandir(prefix_dir):
        name = entry.name
        try:
            if name.startswith(_CACHE_TEMP_PREFIX):
                stale = True
            elif (name.startswith("tmp") and name.endswith(".tmp")) or name.startswith(
                ".pivot_download_"
            ):
                # An older pivot renames its temporary file away once its copy is complete.
                stale = entry.stat(follow_symlinks=False).st_ctime < cutoff
            else:
                continue
            if stale:
                _clear_path(pathlib.Path(entry.path))
        except FileNotFoundError:
            continue
        except OSError as e:
            logger.warning(
                "Cannot remove %s, a temporary entry a killed pivot left in the cache (%s); "
                + "remove it by hand",
                entry.path,
                e,
            )


@contextlib.contextmanager
def _staging_in(prefix_dir: pathlib.Path) -> Generator[None]:
    """Hold prefix_dir's lock shared, so this process's temporary entries there survive the
    sweep, after sweeping when no other process holds the lock and the last sweep is older than
    ``_CACHE_SWEEP_INTERVAL``.

    The exclusive lock is only tried, never waited for; a lock the filesystem refuses (NFS on a
    read-only file, or a filesystem that refuses flock() outright) only skips the sweep. Where
    flock() is refused outright the shared lock is skipped too, so this process's temporary
    entries are protected only from processes that are refused as well; a host whose locks do
    work can still sweep them (storage/AGENTS.md).
    The kernel drops a killed process's lock with its file descriptors. Converting the exclusive
    lock to shared may release it first; this process has made no temporary entry yet.
    """
    fd = _open_cache_lock(prefix_dir / _PREFIX_LOCK)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            pass
        else:
            if time.time() - os.fstat(fd).st_mtime >= _CACHE_SWEEP_INTERVAL:
                _sweep_cache_temps(prefix_dir)
                with contextlib.suppress(OSError):
                    os.utime(fd)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
        except OSError as e:
            if e.errno not in _NO_FLOCK_ERRNO:
                raise
        yield
    finally:
        os.close(fd)


@contextlib.contextmanager
def staged_cache_file(cache_path: pathlib.Path) -> Generator[pathlib.Path]:
    """An empty temporary file beside cache object ``cache_path`` to publish it from
    (``publish_cache_object``), removed on exit. A killed process's is swept (``_staging_in``).
    """
    state_dir_mod.make_dirs(cache_path.parent)
    with _staging_in(cache_path.parent):
        fd, name = tempfile.mkstemp(prefix=_CACHE_TEMP_PREFIX, dir=cache_path.parent)
        os.close(fd)
        staged = pathlib.Path(name)
        try:
            yield staged
        finally:
            staged.unlink(missing_ok=True)


def copy_to_cache(
    src: pathlib.Path, cache_path: pathlib.Path, *, executable: bool = False, verify: bool = False
) -> None:
    """Atomically copy file to cache with read-only permissions.

    ``executable`` records the source's executable bit on the cache object itself (0o555
    instead of 0o444): a HARDLINK or SYMLINK checkout has no mode of its own, so it reads
    straight through to whatever mode the cache object carries, and only COPY checkout
    re-chmods afterwards (see ``_stage_checkout``).

    The object is published with ``publish_cache_object`` and never changed afterwards, its
    mode included. An object already there is kept whatever executable bit it carries; a
    directory entry needing the other bit is checked out as a copy (``_checkout_with_fallback``).

    ``verify`` hashes the staged copy and refuses to publish it under ``cache_path``'s key
    unless it matches. A naming unit sets it when it re-publishes an object it hashed before
    the unit: the source may have changed since, and storing new bytes under the old key would
    hand them to every project whose record names that key.

    Raises:
        exceptions.CacheError: ``verify`` is set and the copy does not hash to the key.
    """
    if cache_path.exists():
        return

    with staged_cache_file(cache_path) as staged:
        shutil.copy2(src, staged)
        if verify:
            key = cache_path.parent.name + cache_path.name
            copied, _ = hash_file(staged)
            if copied != key:
                raise exceptions.CacheError(
                    f"{src} changed after pivot hashed it: it now hashes to {copied}, not "
                    + f"{key}. Refusing to store it under the old key, and naming nothing. "
                    + "Run the command again once the file is no longer being written."
                )
        os.chmod(staged, 0o555 if executable else 0o444)
        _ = publish_cache_object(staged, cache_path)


# A link the filesystem cannot make (across devices, refused, or unsupported) falls through to
# the next checkout mode. EMLINK is not here: it is one object's limit, not the filesystem's
# (see _checkout_with_fallback).
FALLBACK_ERRNO = _NO_HARDLINK_ERRNO | {errno.EXDEV, errno.EACCES}


def _object_realpath(cache_path: pathlib.Path, cache_root: CacheRoot | None) -> str:
    """Return ``os.path.realpath(cache_path)`` for ``cache_path = get_cache_path(cache_dir, key)``.

    ``cache_root`` resolves that ``cache_dir`` once for a caller restoring many files, or is None.
    The text shortcut is valid only when the object's own root is that configured root; a #503
    fallback path under legacy ``.pivot/cache/files`` resolves in full instead.
    """
    if cache_root is not None and _is_configured_cache_object_path(cache_path, cache_root):
        real = os.path.join(cache_root, cache_path.parent.name, cache_path.name)
        if _is_real_object_path(real):
            return real
    return os.path.realpath(cache_path)


def _fstat_staged(path: pathlib.Path) -> os.stat_result:
    """Stat a staged file through an open descriptor before its rename exposes it.

    Following a staged symlink is intentional: the stat must describe the cache object the
    final workspace path will resolve to.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        return os.fstat(fd)
    finally:
        os.close(fd)


def _stage_checkout(
    tmp_path: pathlib.Path,
    cache_path: pathlib.Path,
    checkout_mode: CheckoutMode,
    *,
    executable: bool | None,
    cache_root: CacheRoot | None,
) -> None:
    """Create the link or copy of cache_path at tmp_path.

    A symlink holds the object's real path, so it still works from wherever the workspace is
    reached and the cache-object check settles it from its text (``_resolve_link``).
    """
    if checkout_mode == CheckoutMode.SYMLINK:
        tmp_path.symlink_to(_object_realpath(cache_path, cache_root))
    elif checkout_mode == CheckoutMode.HARDLINK:
        os.link(cache_path, tmp_path)
    else:
        shutil.copy2(cache_path, tmp_path)
        os.chmod(tmp_path, 0o755 if executable else 0o644)


def _checkout_from_cache(
    path: pathlib.Path,
    cache_path: pathlib.Path,
    checkout_mode: CheckoutMode,
    *,
    executable: bool | None = None,
    staging_path: pathlib.Path | None = None,
    before_swap: Callable[[pathlib.Path], None] | None = None,
    cache_root: CacheRoot | None = None,
) -> os.stat_result:
    """Create link or copy at path atomically, so readers never see a partial file.

    Args:
        staging_path: Where to build the link or copy before renaming it over path.
            Defaults to a fresh random name beside path.
        before_swap: Checks the entry at path once the link or copy is built (see
            ``restore_from_cache``): a file or symlink in place, a directory renamed aside.
        cache_root: The cache directory holding ``cache_path``, resolved once by a caller
            restoring many files, so a symlink's target is not resolved per file.

    Returns:
        The stat (following symlinks) of what now stands at path, taken before it was renamed
        there: the rename keeps the inode, so a write after it does not match this stat.
    """
    # Idempotency: skip if already correctly linked
    if checkout_mode == CheckoutMode.SYMLINK and path.is_symlink():
        with contextlib.suppress(OSError):
            if path.resolve() == cache_path.resolve():
                return path.stat()
    elif checkout_mode == CheckoutMode.HARDLINK and path.exists() and not path.is_symlink():
        with contextlib.suppress(OSError):
            linked = path.stat()
            if _is_same_file(linked, cache_path):
                return linked

    tmp_path = staging_path or path.parent / f"{_RESTORE_TEMP_PREFIX}{secrets.token_hex(8)}"
    try:
        try:
            _stage_checkout(
                tmp_path, cache_path, checkout_mode, executable=executable, cache_root=cache_root
            )
        except PermissionError:
            # A read-only parent. Replacing an existing entry needs it writable anyway,
            # so make it writable as _clear_path does (never chmod the file: it may be
            # a hardlink to the read-only cache), then retry.
            parent_mode = path.parent.stat().st_mode
            if parent_mode & stat.S_IWUSR or not (path.exists() or path.is_symlink()):
                raise
            os.chmod(path.parent, parent_mode | stat.S_IWUSR)
            _stage_checkout(
                tmp_path, cache_path, checkout_mode, executable=executable, cache_root=cache_root
            )
        staged = _fstat_staged(tmp_path)
        if before_swap is None:
            try:
                tmp_path.replace(path)
            except IsADirectoryError:
                _clear_path(path)
                tmp_path.replace(path)
            return staged
        # A file or symlink is checked in place and replaced atomically; a directory, there
        # from the start or made after that check, is renamed aside and checked there.
        if not _is_real_dir(path):
            if path.exists() or path.is_symlink():
                before_swap(path)
            try:
                tmp_path.replace(path)
                return staged
            except IsADirectoryError:
                pass
        _swap_in(tmp_path, path, _backup_path(path, _restore_target_token(path)), before_swap)
    finally:
        # Also covers a replace() that did nothing: POSIX rename is a no-op when both
        # names are hardlinks to the same inode, leaving tmp_path behind.
        tmp_path.unlink(missing_ok=True)
    return staged


def _cache_object_isexec(cache_path: pathlib.Path) -> bool | None:
    """Whether the cache object at cache_path is stored executable, or None if absent.

    HARDLINK and SYMLINK checkout have no mode of their own: the workspace file shares the
    cache object's permissions directly, so they can only ever produce whatever executable
    bit the cache object itself carries.
    """
    try:
        return bool(cache_path.stat().st_mode & stat.S_IXUSR)
    except FileNotFoundError:
        return None


def _repair_tree_dir_isexec(cache_dir_path: pathlib.Path, manifest: list[DirManifestEntry]) -> None:
    """Chmod each file in a whole-directory SYMLINK cache tree whose executable bit is not the
    one its manifest entry records, to 0o555 or 0o444.

    A tree an older pivot built (every file chmod'd 0o444 regardless of isexec) keeps being
    reused across an upgrade: its tree hash already incorporates isexec, so it never changes
    and neither _store_directory's ``not cache_dir_path.exists()`` branch nor the
    symlink_fast_path restore would otherwise ever revisit it. Chmod'ing in place is safe
    here, unlike the single-file cache: a tree directory is a private ``copytree`` copy with
    no other hardlinks sharing its files.

    ``manifest`` may come straight from a lock file on the restore path, so each relpath is
    checked the same way the file-by-file restore branch already does: a manifest claiming
    ``../../../outside/file`` must not chmod a file outside the tree.
    """
    resolved_tree = cache_dir_path.resolve()
    for entry in manifest:
        f = cache_dir_path / entry["relpath"]
        if not f.resolve().is_relative_to(resolved_tree):
            raise exceptions.SecurityValidationError(
                f"Manifest contains path traversal: {entry['relpath']!r}"
            )
        isexec = _cache_object_isexec(f)
        if isexec is not None and isexec != entry["isexec"]:
            os.chmod(f, 0o555 if entry["isexec"] else 0o444)


# The marker's earlier names, which no pivot trusts: pivot 0.3.4 touched `.layout-checked` after
# checking only a tree's layout, partial trees included, and pivot 0.3.5 to 0.5.11 touched
# `.verified` after checking each file's size, not its content. Every tree so marked is checked
# by content on each restore or commit until it is marked anew, which drops the old markers; a
# user who cannot write its prefix directory cannot mark it.
_TREE_LAYOUT_MARKER_SUFFIX = ".content-checked"
_LEGACY_TREE_MARKER_SUFFIXES = (".layout-checked", ".verified")
# Held exclusively while a tree is checked and repaired; never deleted.
_TREE_LOCK_SUFFIX = ".lock"

# --- the contract the collector reads this module through -------------------------------
# ``storage/gc.py`` is the one consumer outside this module of the cache's own primitives: it
# takes the same locks, reads the same checkout symlinks, removes objects with the same
# remover, and removes the same tree markers. These names are that contract, so a change to
# any of them is type-checked against the collector instead of being an untyped reach-in.
PREFIX_LOCK = _PREFIX_LOCK
NO_FLOCK_ERRNO = _NO_FLOCK_ERRNO
open_cache_lock = _open_cache_lock
clear_path = _clear_path
get_symlink_cache_key = _get_symlink_cache_hash
# Every marker beside a whole-directory tree object: the current content-checked one and the
# layout-only/size-only ones older pivot versions wrote. A collected tree takes all of them.
TREE_MARKER_SUFFIXES = (_TREE_LAYOUT_MARKER_SUFFIX, *_LEGACY_TREE_MARKER_SUFFIXES)


def _tree_layout_marker(cache_dir_path: pathlib.Path) -> pathlib.Path:
    """Return the file beside a whole-directory SYMLINK cache tree that records the manifest the
    tree was checked to hold (``_mark_tree``). The name is never a cache key, so cache scans skip
    it.
    """
    return cache_dir_path.with_name(cache_dir_path.name + _TREE_LAYOUT_MARKER_SUFFIX)


def _manifest_digest(canonical: bytes) -> str:
    """The SHA-256 a tree's marker records for the canonical manifest it was checked to hold
    (``_canonical_manifest``). A second-preimage-resistant digest, unlike the XXH64 a tree is
    named by, so no manifest an attacker writes can match another's."""
    return hashlib.sha256(canonical).hexdigest()


def _read_marker(cache_dir_path: pathlib.Path) -> str | None:
    """What a tree's marker records: None without a marker, else the manifest digest it holds.
    An empty string is a marker an older pivot wrote (empty), or one this user cannot read."""
    try:
        return _tree_layout_marker(cache_dir_path).read_text().strip()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError):
        return ""


def _mark_tree(cache_dir_path: pathlib.Path, digest: str) -> None:
    """Mark a tree checked to hold exactly the manifest whose ``_manifest_digest`` is ``digest``,
    by writing that digest into its marker, renamed into place whole. The marker is staged under
    the prefix directory's shared lock (``_staging_in``), so no concurrent sweep removes it before
    the rename. The marker only saves a check, so a cache the user cannot write still restores
    without one."""
    with contextlib.suppress(OSError), _staging_in(cache_dir_path.parent):
        fd, staged = tempfile.mkstemp(prefix=_CACHE_TEMP_PREFIX, dir=cache_dir_path.parent)
        try:
            with os.fdopen(fd, "w") as file:
                file.write(digest)
            os.chmod(staged, 0o444)
            os.replace(staged, _tree_layout_marker(cache_dir_path))
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(staged)
    for suffix in _LEGACY_TREE_MARKER_SUFFIXES:
        with contextlib.suppress(OSError):
            cache_dir_path.with_name(cache_dir_path.name + suffix).unlink(missing_ok=True)


def _tree_file_holds(
    path: pathlib.Path, path_stat: os.stat_result, entry: DirManifestEntry
) -> bool:
    """Whether a tree file, ``path_stat`` its ``os.lstat``, is a regular file holding ``entry``'s
    bytes: its recorded size and content hash. Hashes the file only once the cheap checks pass.
    A tree check sees only manifests that hold every field pivot writes, ``size`` included
    (``_canonical_manifest``)."""
    return (
        stat.S_ISREG(path_stat.st_mode)
        and path_stat.st_size == entry["size"]
        and hash_file(path)[0] == entry["hash"]
    )


def _tree_matches(
    cache_dir_path: pathlib.Path, output_hash: DirHash, scanned: tuple[list[str], list[str]]
) -> bool:
    """Whether a whole-directory SYMLINK cache tree holds exactly the files (each regular,
    matching its recorded hash, size and executable bit) and the directories ``output_hash``
    records, its empty directories exactly the tree's (as ``hash_directory`` finds them), so only
    one canonical manifest matches a tree. ``scanned`` is ``_scan_entries(cache_dir_path)``. A
    recorded path leading outside the tree raises: ``output_hash`` may come straight from a lock
    file."""
    manifest = output_hash["manifest"]
    resolved_tree = cache_dir_path.resolve()
    for entry in manifest:
        if not (cache_dir_path / entry["relpath"]).resolve().is_relative_to(resolved_tree):
            raise exceptions.SecurityValidationError(
                f"Manifest contains path traversal: {entry['relpath']!r}"
            )
    empty_dirs = _recorded_empty_dirs(output_hash)
    for relpath in empty_dirs:
        _ = _empty_dir_path(cache_dir_path, relpath)
    found_files, found = scanned
    found_dirs = set(found)
    if set(found_files) != {entry["relpath"] for entry in manifest}:
        return False
    if found_dirs - _restored_dirs(output_hash) or not found_dirs.issuperset(empty_dirs):
        return False
    file_dirs = {posixpath.dirname(entry["relpath"]) for entry in manifest}
    has_child = {posixpath.dirname(directory) for directory in found_dirs}
    if set(empty_dirs) != found_dirs - file_dirs - has_child:
        return False
    for entry in manifest:
        tree_file = cache_dir_path / entry["relpath"]
        found_stat = os.lstat(tree_file)
        if bool(found_stat.st_mode & stat.S_IXUSR) != entry["isexec"] or not _tree_file_holds(
            tree_file, found_stat, entry
        ):
            return False
    return True


def _ensure_tree_complete(
    cache_dir_path: pathlib.Path, output_hash: DirHash, cache_dir: pathlib.Path, *, repair: bool
) -> bool:
    """Return whether a whole-directory SYMLINK cache tree holds exactly what ``output_hash``
    records, checking a tree once; with ``repair``, repairing it in place from its files' own
    cache objects first.

    A tree's name is the XXH64 of its manifest, and XXH64 is not second-preimage resistant: a
    manifest someone commits in a .pvt or lock can be made to hash to any tree's name. So the name
    vouches for nothing. A tree is trusted only for the manifest it was checked to hold, recorded
    by its marker as that manifest's SHA-256 (``_mark_tree``), and only on equality of the
    caller's canonical manifest's digest with it:

    - marked for this manifest: trusted, without reading the tree;
    - marked for another manifest: not trusted and never changed, since its content is what
      another manifest was checked against and an output may symlink to it;
    - marked by an older pivot (an empty marker): checked against this manifest, never repaired;
      re-marked if it matches;
    - unmarked (a killed publisher, a tree an older pivot built or marked after checking only
      layout or sizes): checked, and marked only if it holds this manifest exactly.

    Only a commit passes ``repair``: its manifest is ``hash_directory`` of the bytes it commits, so
    repairing toward it rewrites the tree toward bytes that exist. A restore's manifest comes from a
    .pvt or a lock: it is a claim, not a source of truth, so it never drives a write into the
    cache. Repairing toward it and then comparing the result with that same manifest proves
    nothing, and would let a forged manifest rewrite the bytes of every output already linked to
    the tree. A restore whose tree does not match goes file by file. After a repair the tree's
    manifest is recomputed (``hash_directory``) and must equal the caller's, entry for entry,
    before the tree is marked; that catches a cache object whose bytes are not its name's.

    A manifest that is not even nominally the tree's (its XXH64 is another name) or that holds a
    value JSON cannot encode is refused before anything is read: that only spares work, and is not
    what makes a tree trustworthy. The check, the repair and the mark run under the tree's
    exclusive lock, so two repairers never change the tree's directory modes under each other.
    Returns False, the tree left as it was or unmarked, when the tree differs (and, with
    ``repair``, cannot be repaired), or cannot be checked: a file's object is absent, or this user
    cannot read the tree, write it or take its lock.
    """
    canonical = _canonical_manifest(output_hash)
    if canonical is None or xxhash.xxh64(canonical).hexdigest() != output_hash["hash"]:
        return False
    digest = _manifest_digest(canonical)
    marked = _read_marker(cache_dir_path)
    if marked == digest:
        return True
    if marked:
        return False
    fd: int | None
    try:
        fd = _open_cache_lock(cache_dir_path.with_name(cache_dir_path.name + _TREE_LOCK_SUFFIX))
    except OSError as e:
        if e.errno not in _CANNOT_WRITE_ERRNO:
            raise
        fd = None
    try:
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
            except OSError:  # NFS refuses an exclusive lock on a file opened read-only
                os.close(fd)
                fd = None
        marked = _read_marker(cache_dir_path)
        if marked == digest:
            return True
        if marked:
            return False
        try:
            scanned = _scan_entries(cache_dir_path)
            if not _tree_matches(cache_dir_path, output_hash, scanned):
                if not repair or fd is None or marked is not None:
                    return False
                _repair_tree_dir_isexec(cache_dir_path, output_hash["manifest"])
                _repair_tree_dir_layout(cache_dir_path, output_hash, scanned)
                if not _repair_tree_dir_files(cache_dir_path, output_hash["manifest"], cache_dir):
                    return False
                repaired = _canonical_manifest(hash_directory(cache_dir_path, cache_dir=cache_dir))
                if repaired != canonical:
                    return False
        except PermissionError:  # another user's tree, or one with a file this user cannot read
            return False
        _mark_tree(cache_dir_path, digest)
        return True
    finally:
        if fd is not None:
            os.close(fd)


def _repair_tree_dir_files(
    cache_dir_path: pathlib.Path, manifest: list[DirManifestEntry], cache_dir: pathlib.Path
) -> bool:
    """Give a whole-directory SYMLINK cache tree every file its manifest records, copied from
    its files' own cache objects. Return False, changing nothing, when a file is missing or
    holds the wrong content and its own cache object is absent.

    Each repaired file is copied beside its name and renamed in, so a reader of the tree never
    sees part of one. Changing the tree in place is safe for the reason
    ``_repair_tree_dir_isexec`` gives; the caller holds the tree's lock (``_ensure_tree_complete``).
    """
    resolved_tree = cache_dir_path.resolve()
    wrong = list[tuple[DirManifestEntry, pathlib.Path]]()
    for entry in manifest:
        target = cache_dir_path / entry["relpath"]
        if not target.resolve().is_relative_to(resolved_tree):
            raise exceptions.SecurityValidationError(
                f"Manifest contains path traversal: {entry['relpath']!r}"
            )
        try:
            target_stat = os.lstat(target)
        except FileNotFoundError:
            pass
        else:
            if _tree_file_holds(target, target_stat, entry):
                continue
        wrong.append((entry, target))
    if not wrong:
        return True
    objects = [resolve_cache_object_path(cache_dir, entry["hash"]) for entry, _ in wrong]
    if not all(obj is not None and obj.is_file() for obj in objects):
        return False
    modes = dict[pathlib.Path, int]()

    def _make_writable(directory: pathlib.Path) -> None:
        if directory not in modes:
            modes[directory] = directory.stat().st_mode
            os.chmod(directory, modes[directory] | stat.S_IWUSR)

    try:
        for (entry, target), obj in zip(wrong, objects, strict=True):
            assert obj is not None
            existing = target.parent
            while not existing.exists():
                existing = existing.parent
            _make_writable(existing)
            target.parent.mkdir(parents=True, exist_ok=True)
            _make_writable(target.parent)
            fd, staged = tempfile.mkstemp(prefix=_CACHE_TEMP_PREFIX, dir=target.parent)
            os.close(fd)
            try:
                shutil.copyfile(obj, staged)
                os.chmod(staged, 0o555 if entry["isexec"] else 0o444)
                os.replace(staged, target)
            finally:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(staged)
    finally:
        for directory, mode in modes.items():
            os.chmod(directory, mode)
    return True


def _repair_tree_dir_layout(
    cache_dir_path: pathlib.Path, output_hash: DirHash, scanned: tuple[list[str], list[str]]
) -> None:
    """Give a whole-directory SYMLINK cache tree exactly the files and directories
    ``output_hash`` records. ``scanned`` is ``_scan_entries(cache_dir_path)``.

    A tree an older pivot built can hold entries its hash does not record: directories its hash
    left out because they were empty, and copies of the targets of symlinks its hash left out;
    a repair killed midway leaves a temporary file in it. The tree's name never changes, so
    neither _store_directory's ``not cache_dir_path.exists()`` branch nor the symlink_fast_path
    restore would otherwise revisit it, and an output restored through it would not hash to the
    hash it was restored for. Changing the tree in place is safe for the reason
    ``_repair_tree_dir_isexec`` gives; the caller holds the tree's lock.
    """
    found_files, found = scanned
    found_dirs = set(found)
    unlisted = set(found_files) - {entry["relpath"] for entry in output_hash["manifest"]}
    extra_dirs = found_dirs - _restored_dirs(output_hash)
    missing = [
        _empty_dir_path(cache_dir_path, relpath)
        for relpath in _recorded_empty_dirs(output_hash)
        if relpath not in found_dirs
    ]
    if not unlisted and not extra_dirs and not missing:
        return
    modes = dict[pathlib.Path, int]()

    def _make_writable(directory: pathlib.Path) -> None:
        if directory not in modes:
            modes[directory] = directory.stat().st_mode
            os.chmod(directory, modes[directory] | stat.S_IWUSR)

    try:
        for relpath in unlisted:
            target = cache_dir_path / relpath
            _make_writable(target.parent)
            target.unlink(missing_ok=True)
        # Reverse order reaches a directory's children before the directory itself.
        for relpath in sorted(extra_dirs, reverse=True):
            target = cache_dir_path / relpath
            _make_writable(target.parent)
            modes.pop(target, None)
            with contextlib.suppress(FileNotFoundError):  # a concurrent repair got there first
                target.rmdir()
        for target in missing:
            existing = target.parent
            while not existing.exists():
                existing = existing.parent
            _make_writable(existing)
            target.mkdir(parents=True, exist_ok=True)
    finally:
        for directory, mode in modes.items():
            os.chmod(directory, mode)


def _checkout_with_fallback(
    path: pathlib.Path,
    cache_path: pathlib.Path,
    checkout_modes: list[CheckoutMode],
    *,
    executable: bool | None = None,
    staging_path: pathlib.Path | None = None,
    keep_existing_copy: bool = False,
    display_path: pathlib.Path | None = None,
    before_swap: Callable[[pathlib.Path], None] | None = None,
    cache_root: CacheRoot | None = None,
) -> os.stat_result:
    """Try each link mode in order until one succeeds.

    With ``keep_existing_copy``, reaching COPY while ``path`` still exists leaves it in place:
    the caller guarantees it already holds the cached bytes. A failed link attempt leaves
    ``path`` as it was, since each mode builds its entry under a staging name and renames it
    over ``path`` only once it is made.

    A hard link refused with EMLINK means ``cache_path`` alone already has as many links as the
    filesystem allows (65,000 on ext4); other objects still link. That entry is copied, as if
    COPY came next, whatever the order holds: a copy is always correct, where a symlink into the
    cache dangles once the workspace is moved or packed without it.

    ``executable`` is None for a single-file output: ``FileHash`` has no isexec field, so a
    single file's mode is never part of any hash and there is no wrong permission to guard
    against. For a directory entry (``executable`` is a bool), a HARDLINK or SYMLINK whose
    cache object's recorded executable bit does not match it is skipped rather than attempted:
    two files with identical content can disagree on the executable bit (the cache key is
    content-only), and a cache object a remote pull populated by hash alone carries no
    executable bit at all. Either way the shared object cannot represent two permissions at
    once, so checkout falls through to the next mode; only COPY makes a private copy and can
    always re-chmod it correctly. Cache objects never change once published, so the bit read
    here is the bit a link or symlink made after it keeps.

    ``display_path`` names the file in a raised error when ``path`` is a restore-only staging
    location (a caller checking out into a scratch directory before an atomic swap); it
    defaults to ``path`` for every other caller. ``before_swap`` checks the entry at ``path``
    just before the made entry replaces it (``_checkout_from_cache``).

    ``cache_root`` is the cache directory holding ``cache_path``, resolved once by a caller
    restoring many files: a symlink then names the object's real path without resolving it per
    file (``_object_realpath``). None resolves it.

    Returns:
        What ``path`` holds, stat'd before it was renamed there (``_checkout_from_cache``).
    """
    if not checkout_modes:
        raise ValueError("checkout_modes cannot be empty")
    display = display_path if display_path is not None else path
    for i, mode in enumerate(checkout_modes):
        is_last = i == len(checkout_modes) - 1
        if keep_existing_copy and mode == CheckoutMode.COPY and path.exists():
            return path.stat()
        if executable is not None and mode in (CheckoutMode.HARDLINK, CheckoutMode.SYMLINK):
            cached_isexec = _cache_object_isexec(cache_path)
            if cached_isexec is not None and cached_isexec != executable:
                if is_last:
                    cached_state = "executable" if cached_isexec else "non-executable"
                    raise exceptions.CacheModeCollisionError(
                        f"Cache object {cache_path} is recorded {cached_state}, but {display} "
                        + f"needs isexec={executable}."
                    )
                logger.debug(
                    f"Cache object {cache_path} is executable={cached_isexec}, "
                    + f"need {executable}; skipping {mode.value}"
                )
                continue
        try:
            return _checkout_from_cache(
                path,
                cache_path,
                mode,
                executable=executable,
                staging_path=staging_path,
                before_swap=before_swap,
                cache_root=cache_root,
            )
        except OSError as e:
            if mode == CheckoutMode.HARDLINK and e.errno == errno.EMLINK:
                logger.debug(f"Cache object {cache_path} is at the hard-link limit; copying it")
                if keep_existing_copy and path.exists():
                    return path.stat()
                return _checkout_from_cache(
                    path,
                    cache_path,
                    CheckoutMode.COPY,
                    executable=executable,
                    staging_path=staging_path,
                    before_swap=before_swap,
                )
            if e.errno not in FALLBACK_ERRNO or is_last:
                raise
            logger.debug(f"Checkout mode {mode.value} failed ({e}), trying next mode")
    raise AssertionError("the last checkout mode returns or raises")


DEFAULT_CHECKOUT_MODE_ORDER = [CheckoutMode.HARDLINK, CheckoutMode.SYMLINK, CheckoutMode.COPY]


def _resolve_checkout_modes(
    checkout_mode: CheckoutMode | None,
    checkout_modes: list[CheckoutMode] | None,
) -> list[CheckoutMode]:
    """Resolve effective checkout modes from single mode or list."""
    if checkout_mode is not None:
        return [checkout_mode]
    if checkout_modes is not None:
        if not checkout_modes:
            raise ValueError("checkout_modes cannot be empty")
        return checkout_modes
    return DEFAULT_CHECKOUT_MODE_ORDER.copy()


def missing_cache_objects(
    output_hash: HashInfo, cache_dir: pathlib.Path, checkout_modes: list[CheckoutMode]
) -> list[str]:
    """Return hashes required to restore an output that are absent from the cache.

    With SYMLINK first, an existing whole-directory tree for a directory output is checked, never
    repaired, before it is trusted (``_ensure_tree_complete``)."""
    if is_dir_hash(output_hash):
        tree_path = resolve_cache_object_path(cache_dir, output_hash["hash"], directory=True)
        if (
            checkout_modes[0] == CheckoutMode.SYMLINK
            and tree_path is not None
            and tree_path.exists()
            and _ensure_tree_complete(tree_path, output_hash, cache_dir, repair=False)
        ):
            return []
        hashes = [entry["hash"] for entry in output_hash["manifest"]]
    else:
        hashes = [output_hash["hash"]]
    return [
        file_hash
        for file_hash in hashes
        if (cache_path := resolve_cache_object_path(cache_dir, file_hash)) is None
        or not cache_path.exists()
    ]


def save_to_cache(
    path: pathlib.Path,
    cache_dir: pathlib.Path,
    state_db: state_mod.StateDB | None = None,
    checkout_mode: CheckoutMode | None = None,
    checkout_modes: list[CheckoutMode] | None = None,
    state_dir: pathlib.Path | None = None,
) -> HashInfo:
    """Save file or directory to cache, replace with link, return hash info.

    Args:
        path: File or directory to save
        cache_dir: Cache directory
        state_db: Optional state database for hash caching
        checkout_mode: Single link mode, with no fallback except that a hard link refused with
            EMLINK is copied. Takes precedence over checkout_modes.
        checkout_modes: Ordered list of link modes to try with fallback on failure; a hard link
            refused with EMLINK is copied whatever the order is.
        state_dir: State directory for the workspace target's restore lock.
    """
    effective_modes = _resolve_checkout_modes(checkout_mode, checkout_modes)

    if path.is_dir():
        return _save_directory_to_cache(path, cache_dir, state_db, effective_modes, state_dir)
    return _save_file_to_cache(path, cache_dir, state_db, effective_modes, state_dir)


def _save_file_to_cache(
    path: pathlib.Path,
    cache_dir: pathlib.Path,
    state_db: state_mod.StateDB | None,
    checkout_modes: list[CheckoutMode],
    state_dir: pathlib.Path | None,
) -> FileHash:
    """Save single file to cache."""
    # Idempotency: check if already a valid cache symlink (cheap check first)
    if checkout_modes and checkout_modes[0] == CheckoutMode.SYMLINK:
        existing_hash = _get_symlink_cache_hash(path, cache_dir)
        if existing_hash is not None:
            cache_path = get_cache_path(cache_dir, existing_hash)
            if cache_path.exists():
                return FileHash(hash=existing_hash)

    file_hash, _ = hash_file(path, state_db)
    _store_file(
        path,
        file_hash,
        cache_dir,
        checkout_modes,
        keep_existing_copy=False,
        state_dir=state_dir,
    )
    return FileHash(hash=file_hash)


def _store_file(
    path: pathlib.Path,
    file_hash: str,
    cache_dir: pathlib.Path,
    checkout_modes: list[CheckoutMode],
    *,
    keep_existing_copy: bool,
    state_dir: pathlib.Path | None = None,
    verify: bool = False,
) -> None:
    cache_path = get_cache_path(cache_dir, file_hash)
    copy_to_cache(path, cache_path, verify=verify)
    with restore_lock(path, state_dir) as lock_token:
        staging_path = restore_staging_path(path, lock_token)
        _clear_path(staging_path)
        _checkout_with_fallback(
            path,
            cache_path,
            checkout_modes,
            staging_path=staging_path,
            keep_existing_copy=keep_existing_copy,
        )


def _save_directory_to_cache(
    path: pathlib.Path,
    cache_dir: pathlib.Path,
    state_db: state_mod.StateDB | None,
    checkout_modes: list[CheckoutMode],
    state_dir: pathlib.Path | None,
) -> DirHash:
    """Save directory to cache."""
    output_hash = hash_directory(path, state_db, cache_dir=cache_dir)
    if (
        checkout_modes
        and checkout_modes[0] == CheckoutMode.SYMLINK
        and path.is_symlink()
        and _get_symlink_cache_hash(path, cache_dir) == output_hash["hash"]
        and (
            cache_path := resolve_cache_object_path(cache_dir, output_hash["hash"], directory=True)
        )
        is not None
        and cache_path.exists()
    ):
        return output_hash

    _store_directory(
        path,
        output_hash,
        cache_dir,
        checkout_modes,
        keep_existing_copy=False,
        state_dir=state_dir,
    )
    return output_hash


def _publish_tree(
    path: pathlib.Path,
    cache_dir_path: pathlib.Path,
    output_hash: DirHash,
    cache_dir: pathlib.Path,
) -> None:
    """Copy directory output ``path`` into the cache as the whole-directory SYMLINK tree
    ``cache_dir_path`` its hash ``output_hash`` names.

    A restore symlinks to a tree as soon as its name exists, so the tree is built under a
    temporary name beside it and renamed into place complete. rename() will not replace a
    directory that holds anything, so a tree another writer published first is kept and this
    copy dropped; an empty output's tree, which holds nothing, can replace another writer's,
    which is the same empty tree. A later publish into the same prefix directory removes the copy
    a killed publisher left (``_staging_in``).

    The tree is not marked here: the commit that published it checks it and marks it next
    (``_ensure_tree_complete``), since a sweep on another host whose flock is not coherent with
    this host's can remove files from the copy while it is built (see storage/AGENTS.md). An
    unmarked tree is checked by the next restore through it and repaired only by the next commit
    of that output.
    """
    manifest = output_hash["manifest"]

    cache_root = resolve_cache_root(cache_dir)

    def _user_symlinks(directory: str, names: list[str]) -> list[str]:
        # Not stored (see walk_artifact_members); copytree would copy each target.
        return [
            name
            for name in names
            if (link := pathlib.Path(directory, name)).is_symlink()
            and _cache_object_key(link, cache_root) is None
        ]

    state_dir_mod.make_dirs(cache_dir_path.parent)
    with _staging_in(cache_dir_path.parent):
        staged = pathlib.Path(
            tempfile.mkdtemp(prefix=_CACHE_TEMP_PREFIX, dir=cache_dir_path.parent)
        )
        try:
            shutil.copytree(path, staged, ignore=_user_symlinks, dirs_exist_ok=True)
            isexec_by_relpath = {entry["relpath"]: entry["isexec"] for entry in manifest}
            for f in staged.rglob("*"):
                # Skip symlinks to avoid changing permissions on target files outside cache
                if f.is_file() and not f.is_symlink():
                    executable = isexec_by_relpath.get(str(f.relative_to(staged)), False)
                    os.chmod(f, 0o555 if executable else 0o444)
            os.chmod(staged, 0o555)
            try:
                staged.rename(cache_dir_path)
            except OSError as e:
                if e.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                    raise
        finally:
            _clear_path(staged)


def _store_directory(
    path: pathlib.Path,
    output_hash: DirHash,
    cache_dir: pathlib.Path,
    checkout_modes: list[CheckoutMode],
    *,
    keep_existing_copy: bool,
    state_dir: pathlib.Path | None = None,
    verify: bool = False,
) -> None:
    manifest = output_hash["manifest"]
    # Cache individual files first
    for entry in manifest:
        file_path = path / entry["relpath"]
        cache_path = get_cache_path(cache_dir, entry["hash"])
        copy_to_cache(file_path, cache_path, executable=entry["isexec"], verify=verify)

    with restore_lock(path, state_dir):
        if checkout_modes and checkout_modes[0] == CheckoutMode.SYMLINK:
            # SYMLINK mode: cache entire directory, symlink to it
            cache_dir_path = get_cache_path(cache_dir, output_hash["hash"])
            if not cache_dir_path.exists():
                _publish_tree(path, cache_dir_path, output_hash, cache_dir)

            # A tree an older pivot built keeps its 0o444-everywhere files, entries its hash does
            # not record, or files missing from a copy it was killed in, forever otherwise: its name
            # never changes, so this branch never revisits an existing tree directory above.
            if not _ensure_tree_complete(cache_dir_path, output_hash, cache_dir, repair=True):
                raise exceptions.CacheRestoreError(
                    f"Cache tree {cache_dir_path} differs from what its hash records, or cannot "
                    + "be read, and cannot be repaired (a file's cache object is missing, another "
                    + "user owns it, or it is marked for another manifest); remove the tree, or "
                    + "have its owner remove it, to let this commit publish it again"
                )

            _clear_path(path)
            path.symlink_to(cache_dir_path.resolve())
        else:
            # HARDLINK/COPY modes: replace each file with link/copy from cache
            for entry in manifest:
                file_path = path / entry["relpath"]
                cache_path = get_cache_path(cache_dir, entry["hash"])
                staging_path = restore_staging_path(file_path)
                _clear_path(staging_path)
                _checkout_with_fallback(
                    file_path,
                    cache_path,
                    checkout_modes,
                    executable=entry["isexec"],
                    staging_path=staging_path,
                    keep_existing_copy=keep_existing_copy,
                )


def publish_hashed_output(
    path: pathlib.Path,
    output_hash: HashInfo,
    cache_dir: pathlib.Path,
    checkout_modes: list[CheckoutMode] | None = None,
    *,
    state_dir: pathlib.Path | None = None,
) -> None:
    """Copy an output whose hash is known into the cache, without linking the workspace path.

    A commit calls this before its naming-lock unit, so reading and copying a large output holds
    no lock a waiting collection could turn into a stall for every other commit. Inside the unit
    it calls ``store_hashed_output``: each object is then present (a cheap check), any a
    collection removed meanwhile is published again, and only then is the output linked and
    named. ``output_hash`` must be the hash of ``path``'s current contents.
    """
    effective_modes = _resolve_checkout_modes(None, checkout_modes)
    if not is_dir_hash(output_hash):
        copy_to_cache(path, get_cache_path(cache_dir, output_hash["hash"]))
        return
    for entry in output_hash["manifest"]:
        copy_to_cache(
            path / entry["relpath"],
            get_cache_path(cache_dir, entry["hash"]),
            executable=entry["isexec"],
        )
    if effective_modes and effective_modes[0] == CheckoutMode.SYMLINK:
        cache_dir_path = get_cache_path(cache_dir, output_hash["hash"])
        if not cache_dir_path.exists():
            with restore_lock(path, state_dir):
                if not cache_dir_path.exists():
                    _publish_tree(path, cache_dir_path, output_hash, cache_dir)


def store_hashed_output(
    path: pathlib.Path,
    output_hash: HashInfo,
    cache_dir: pathlib.Path,
    checkout_modes: list[CheckoutMode] | None = None,
    *,
    state_dir: pathlib.Path | None = None,
    verify: bool = False,
) -> None:
    """Put an output whose hash is already known into the cache, then link it per mode.

    ``output_hash`` must be the hash of ``path``'s current contents; nothing is hashed again.
    Each file is copied into the cache once and the workspace path is linked to the cache entry
    with the first checkout mode that works. Reaching COPY leaves the workspace file as it is,
    since it already holds the cached bytes. ``verify`` hashes any file it has to copy and
    refuses on a mismatch (``copy_to_cache``): a naming unit sets it, since its hash was taken
    before the unit and the file may have changed since.
    """
    effective_modes = _resolve_checkout_modes(None, checkout_modes)
    if is_dir_hash(output_hash):
        _store_directory(
            path,
            output_hash,
            cache_dir,
            effective_modes,
            keep_existing_copy=True,
            state_dir=state_dir,
            verify=verify,
        )
    else:
        _store_file(
            path,
            output_hash["hash"],
            cache_dir,
            effective_modes,
            keep_existing_copy=True,
            state_dir=state_dir,
            verify=verify,
        )


def restore_from_cache(
    path: pathlib.Path,
    output_hash: HashInfo,
    cache_dir: pathlib.Path,
    checkout_mode: CheckoutMode | None = None,
    checkout_modes: list[CheckoutMode] | None = None,
    state_dir: pathlib.Path | None = None,
    *,
    lock_token: str | None = None,
    before_swap: Callable[[pathlib.Path], None] | None = None,
    recorded: list[tuple[pathlib.Path, os.stat_result, str]] | None = None,
) -> bool:
    """Restore file or directory from cache. Returns True if successful.

    Args:
        path: Target path to restore to
        output_hash: Hash info for the cached output
        cache_dir: Cache directory
        checkout_mode: Single link mode, with no fallback except that a hard link refused with
            EMLINK is copied. Takes precedence over checkout_modes.
        checkout_modes: Ordered list of link modes to try with fallback on failure; a hard link
            refused with EMLINK is copied whatever the order is.
        state_dir: State directory for restore locks and crash cleanup.
        lock_token: Target token yielded by an already-held restore_lock for path.
        before_swap: Checks the entry the restore removes once the replacement is built, just
            before it takes ``path``. It is passed where that entry is: renamed aside, beside
            ``path``, when a directory replaces it or it is a directory, so no writer that names
            files by path can add one to it after the check; else ``path`` itself, which a file
            replaces atomically. When it raises, the entry is left at ``path`` and the error
            propagates. Not called when nothing is at ``path``.
        recorded: Receives ``(path, stat, hash)`` for each file restored, for the caller to
            record in a StateDB. Each stat is taken before the file is renamed into place (the
            same inode), so a write after the restore does not match it.
    """
    effective_modes = _resolve_checkout_modes(checkout_mode, checkout_modes)
    if lock_token is not None and state_dir is None:
        raise ValueError("state_dir is required when lock_token is set")

    if is_dir_hash(output_hash):
        return _restore_directory_from_cache(
            path,
            output_hash,
            cache_dir,
            effective_modes,
            state_dir,
            lock_token=lock_token,
            before_swap=before_swap,
            recorded=recorded,
        )
    return _restore_file_from_cache(
        path,
        output_hash,
        cache_dir,
        effective_modes,
        state_dir,
        lock_token=lock_token,
        before_swap=before_swap,
        recorded=recorded,
    )


def _restore_file_from_cache(
    path: pathlib.Path,
    output_hash: FileHash,
    cache_dir: pathlib.Path,
    checkout_modes: list[CheckoutMode],
    state_dir: pathlib.Path | None = None,
    *,
    lock_token: str | None = None,
    before_swap: Callable[[pathlib.Path], None] | None = None,
    recorded: list[tuple[pathlib.Path, os.stat_result, str]] | None = None,
) -> bool:
    """Restore single file from cache.

    With a state_dir, holds path's restore lock and stages the file under one name
    tagged with path's token, so the next restore removes what a killed one left.
    """
    cache_path = resolve_cache_object_path(cache_dir, output_hash["hash"])
    if cache_path is None or not cache_path.exists():
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    if state_dir is None:
        staged = _checkout_with_fallback(path, cache_path, checkout_modes, before_swap=before_swap)
        if recorded is not None:
            recorded.append((path, staged, output_hash["hash"]))
        return True

    if lock_token is None:
        with restore_lock(path, state_dir) as held_token:
            return _restore_file_from_cache(
                path,
                output_hash,
                cache_dir,
                checkout_modes,
                state_dir,
                lock_token=held_token,
                before_swap=before_swap,
                recorded=recorded,
            )

    staging_path = restore_staging_path(path, lock_token)
    _clear_path(staging_path)
    staged = _checkout_with_fallback(
        path,
        cache_path,
        checkout_modes,
        staging_path=staging_path,
        before_swap=before_swap,
    )
    if recorded is not None:
        recorded.append((path, staged, output_hash["hash"]))
    return True


def _restore_directory_from_cache(
    path: pathlib.Path,
    output_hash: DirHash,
    cache_dir: pathlib.Path,
    checkout_modes: list[CheckoutMode],
    state_dir: pathlib.Path | None = None,
    *,
    lock_token: str | None = None,
    before_swap: Callable[[pathlib.Path], None] | None = None,
    recorded: list[tuple[pathlib.Path, os.stat_result, str]] | None = None,
) -> bool:
    """Restore a directory atomically. ``recorded`` is as ``restore_from_cache`` says: each
    file's stat is taken in the tree or staging directory, before the swap puts it in place."""
    cache_dir_path = resolve_cache_object_path(cache_dir, output_hash["hash"], directory=True)
    symlink_fast_path = (
        checkout_modes[0] == CheckoutMode.SYMLINK
        and cache_dir_path is not None
        and cache_dir_path.exists()
    )
    if not symlink_fast_path:
        for entry in output_hash["manifest"]:
            if resolve_cache_object_path(cache_dir, entry["hash"]) is None:
                return False

    path.parent.mkdir(parents=True, exist_ok=True)
    _cleanup_stale_restore_temps(path.parent)

    if state_dir is not None and lock_token is None:
        with restore_lock(path, state_dir) as held_token:
            return _restore_directory_from_cache(
                path,
                output_hash,
                cache_dir,
                checkout_modes,
                state_dir,
                lock_token=held_token,
                before_swap=before_swap,
                recorded=recorded,
            )

    target_token = lock_token or _restore_target_token(path)
    if lock_token is not None:
        _cleanup_fresh_restore_orphans(path, target_token, before_swap)
    restore_prefix = f"{_RESTORE_TEMP_PREFIX}{target_token}."

    if symlink_fast_path:
        assert cache_dir_path is not None
        if not _ensure_tree_complete(cache_dir_path, output_hash, cache_dir, repair=False):
            # A restore only checks the tree, never repairs it toward the manifest it was given. A
            # tree that does not hold that manifest exactly, or that this user cannot check, is not
            # used: the output is restored file by file from the objects instead, if they are
            # cached.
            logger.warning(
                "Cache tree %s was not checked to hold the manifest given for it, or differs from "
                + "it, or cannot be read; restoring %s file by file",
                cache_dir_path,
                path,
            )
            symlink_fast_path = False
            for entry in output_hash["manifest"]:
                if resolve_cache_object_path(cache_dir, entry["hash"]) is None:
                    return False

    if symlink_fast_path:
        assert cache_dir_path is not None
        temp_symlink = path.parent / f"{restore_prefix}symlink_{secrets.token_hex(4)}"
        try:
            tree_files = (
                [
                    (
                        path / entry["relpath"],
                        (cache_dir_path / entry["relpath"]).stat(),
                        entry["hash"],
                    )
                    for entry in output_hash["manifest"]
                ]
                if recorded is not None
                else []
            )
            temp_symlink.symlink_to(cache_dir_path.resolve())
            if before_swap is None:
                _clear_path(path)
                temp_symlink.replace(path)
            else:
                _swap_in(temp_symlink, path, _backup_path(path, target_token), before_swap)
            if recorded is not None:
                recorded.extend(tree_files)
            return True
        except OSError as error:
            logger.debug(f"Symlink creation failed, will try file-by-file: {error}")
            for entry in output_hash["manifest"]:
                if resolve_cache_object_path(cache_dir, entry["hash"]) is None:
                    return False
        finally:
            if temp_symlink.is_symlink():
                temp_symlink.unlink()

    temp_path = pathlib.Path(tempfile.mkdtemp(prefix=restore_prefix, dir=path.parent))
    resolved_temp = temp_path.resolve()
    restored = list[tuple[pathlib.Path, os.stat_result, str]]()
    try:
        resolved_temp = temp_path.resolve()
        cache_root = resolve_cache_root(cache_dir)
        for file_count, entry in enumerate(output_hash["manifest"], start=1):
            file_cache_path = resolve_cache_object_path(cache_dir, entry["hash"])
            if file_cache_path is None:
                return False
            file_path = temp_path / entry["relpath"]
            if not file_path.resolve().is_relative_to(resolved_temp):
                raise exceptions.SecurityValidationError(
                    f"Manifest contains path traversal: {entry['relpath']!r}"
                )
            file_path.parent.mkdir(parents=True, exist_ok=True)
            staged = _checkout_with_fallback(
                file_path,
                file_cache_path,
                checkout_modes,
                executable=entry["isexec"],
                display_path=path / entry["relpath"],
                cache_root=cache_root,
            )
            restored.append((path / entry["relpath"], staged, entry["hash"]))
            if file_count % 100 == 0:
                try:
                    os.utime(temp_path, None)
                except OSError as e:
                    logger.debug(f"mtime refresh failed (may be NFS with root_squash): {e}")

        for relpath in _recorded_empty_dirs(output_hash):
            _empty_dir_path(temp_path, relpath).mkdir(parents=True, exist_ok=True)

        try:
            os.utime(temp_path, None)
        except OSError as e:
            logger.debug(f"mtime refresh failed (may be NFS with root_squash): {e}")

        if CheckoutMode.COPY in checkout_modes:
            try:
                os.chmod(temp_path, 0o755)
                for item in temp_path.rglob("*"):
                    if item.is_dir() and not item.is_symlink():
                        os.chmod(item, 0o755)
            except OSError as e:
                logger.debug(f"chmod failed (may be NFS with root_squash): {e}")

        _swap_in(temp_path, path, _backup_path(path, target_token), before_swap)
        if recorded is not None:
            recorded.extend(restored)
        return True
    finally:
        if temp_path.exists():
            _clear_path(temp_path)


def _backup_path(path: pathlib.Path, token: str) -> pathlib.Path:
    """A fresh name beside ``path`` to rename it to while a restore swaps in its replacement,
    tagged with ``path``'s restore token so the next directory restore of ``path`` removes an
    orphan. A file restore removes none: a killed one's backup stays beside ``path``."""
    return path.parent / f"{_BACKUP_TEMP_PREFIX}{token}.{secrets.token_hex(4)}"


def _swap_in(
    staged: pathlib.Path,
    path: pathlib.Path,
    backup_path: pathlib.Path,
    before_swap: Callable[[pathlib.Path], None] | None,
) -> None:
    """Rename ``staged`` to ``path``, removing the entry there only once ``staged`` took its place.

    The entry is first renamed to ``backup_path`` and checked there by ``before_swap``
    (``restore_from_cache``): when the check raises, the entry goes back (``_put_back_checked``).
    When ``staged`` cannot take its place, the entry goes back if ``path`` is still free. Either
    way the error propagates and ``staged`` is left for the caller to remove.
    """
    try:
        path.rename(backup_path)
    except FileNotFoundError:
        # Nothing at path: nothing to check or remove
        staged.replace(path)
        return
    if before_swap is not None:
        try:
            before_swap(backup_path)
        except BaseException:
            _put_back_checked(backup_path, path)
            raise
    try:
        staged.replace(path)
    except BaseException:
        _put_back_checked(backup_path, path)
        raise
    try:
        _clear_path(backup_path)
    except OSError as e:
        logger.debug(f"Backup cleanup failed (non-critical): {e}")


def _put_back_checked(moved: pathlib.Path, path: pathlib.Path) -> None:
    """Rename the entry a restore moved from ``path`` to ``moved`` back, after its check refused
    it or failed: it may hold a user's file.

    A directory's rename replaces at most an empty directory made at ``path`` since; any other
    entry goes back only while ``path`` is free. One that cannot go back is never left under its
    backup name, which a later restore removes: it is renamed to a name no restore removes, and a
    warning names it.
    """
    reason: OSError
    try:
        free = not (path.exists() or path.is_symlink())
        if free or (moved.is_dir() and not moved.is_symlink()):
            moved.rename(path)
            return
        reason = FileExistsError(errno.EEXIST, "a new entry took its place", str(path))
    except OSError as e:
        reason = e
    kept = path.with_name(f"{path.name}.pivot-kept-{secrets.token_hex(4)}")
    try:
        moved.rename(kept)
    except OSError as e:
        logger.warning("Could not put %s back (%s), nor keep it at %s: %s", path, reason, kept, e)
        return
    logger.warning("Could not put %s back (%s); it is at %s", path, reason, kept)


def remove_output(path: pathlib.Path) -> None:
    """Remove output file or directory before execution."""
    _clear_path(path)


class MissingRestore(TypedDict):
    """Outcome of restore_missing_in_directory."""

    restored: int  # missing files restored in place
    missing: int  # missing files left missing: a blob is not in the cache, so nothing is written
    local_changes: list[str]  # changed or extra files whose content the cache lacks
    replace: bool  # directory holds only cached content but not the target: replace it whole


def _is_cached(cache_dir: pathlib.Path, content_hash: str | None) -> bool:
    if content_hash is None or not is_valid_hash(content_hash):
        return False
    cache_path = resolve_cache_object_path(cache_dir, content_hash)
    return cache_path is not None and cache_path.exists()


def _scan_entries(path: pathlib.Path) -> tuple[list[str], list[str]]:
    """Return every file and symlink, and every directory, under ``path`` as relative paths.

    Symlinked directories count as symlinks and are not followed.
    """
    files = list[str]()
    dirs = list[str]()

    def _walk(directory: pathlib.Path) -> None:
        with os.scandir(directory) as entries:
            for entry in entries:
                entry_path = pathlib.Path(entry.path)
                relpath = str(entry_path.relative_to(path))
                if entry.is_symlink() or entry.is_file():
                    files.append(relpath)
                elif entry.is_dir():
                    dirs.append(relpath)
                    _walk(entry_path)

    _walk(path)
    return files, dirs


def _is_same_file(file_stat: os.stat_result, cache_path: pathlib.Path) -> bool:
    """Whether file_stat is the cache blob itself (a hardlink checkout). Cache blobs are
    read-only and never rewritten, so sharing the inode proves the content without a read."""
    try:
        blob_stat = cache_path.stat()
    except FileNotFoundError:
        return False
    return (file_stat.st_dev, file_stat.st_ino) == (blob_stat.st_dev, blob_stat.st_ino)


def restore_missing_in_directory(
    path: pathlib.Path,
    output_hash: DirHash,
    cache_dir: pathlib.Path,
    checkout_modes: list[CheckoutMode],
    state_db: state_mod.StateDB | None = None,
    pending: list[tuple[pathlib.Path, os.stat_result, str]] | None = None,
    *,
    before_write: Callable[[], None] | None = None,
) -> MissingRestore:
    """Restore missing target files and empty directories without overwriting local changes.

    When every existing file matches the target, missing files and empty directories are
    restored in place. When a file differs from the target or an extra file or directory
    exists, only a whole-directory swap can restore the target. The swap is safe only when
    every file it removes is in the cache (an extra empty directory removes none); otherwise
    the local changes are returned without writing anything.

    A retained file is read only when nothing else proves its content: a symlink to the
    target's cache path, a hardlink to the target's cache blob, or a matching ``state_db``
    entry each settle it without a read.

    ``pending``, when given, receives the StateDB file-hash entry of every file whose content
    this settled without a matching ``state_db`` entry, and of every file it restored, for the
    caller to save.
    """
    lexical_base = pathlib.Path(os.path.abspath(path))
    resolved_base = path.resolve()
    cache_root = resolve_cache_root(cache_dir)
    expected_paths = set[str]()
    missing = list[tuple[DirManifestEntry, pathlib.Path, pathlib.Path]]()
    # relpath -> hash of the content on disk (None when it cannot be in the cache)
    changed = dict[str, str | None]()

    for entry in output_hash["manifest"]:
        relpath = entry["relpath"]
        expected_paths.add(relpath)
        file_path = path / relpath
        lexical_path = pathlib.Path(os.path.abspath(file_path))
        if not lexical_path.is_relative_to(lexical_base):
            raise exceptions.SecurityValidationError(
                f"Manifest contains path traversal: {relpath!r}"
            )

        resolved_cache_path = resolve_cache_object_path(cache_dir, entry["hash"])
        file_cache_path = resolved_cache_path or get_cache_path(cache_dir, entry["hash"])
        if file_path.is_symlink():
            if not file_path.parent.resolve().is_relative_to(resolved_base):
                raise exceptions.SecurityValidationError(
                    f"Manifest contains path traversal: {relpath!r}"
                )
            if resolved_cache_path is not None and _links_to_object(
                file_path, file_cache_path, cache_root
            ):
                if not file_cache_path.exists():
                    missing.append((entry, file_path, file_cache_path))
                elif pending is not None:
                    pending.append((file_path, file_path.stat(), entry["hash"]))
            else:
                changed[relpath] = _symlink_cache_key(file_path, cache_root)
            continue
        if file_path.exists():
            if not file_path.is_file():
                changed[relpath] = None
                continue
            file_stat = file_path.stat()
            if resolved_cache_path is not None and _is_same_file(file_stat, file_cache_path):
                if pending is not None:
                    pending.append((file_path, file_stat, entry["hash"]))
                continue
            file_hash, _ = hash_file(file_path, state_db, pending=pending)
            if file_hash != entry["hash"]:
                changed[relpath] = file_hash
            continue
        if not file_path.parent.resolve().is_relative_to(resolved_base):
            raise exceptions.SecurityValidationError(
                f"Manifest contains path traversal: {relpath!r}"
            )
        missing.append((entry, file_path, file_cache_path))

    missing_dirs = [
        dir_path
        for dir_path in (
            _empty_dir_path(path, relpath) for relpath in _recorded_empty_dirs(output_hash)
        )
        if not dir_path.exists() and not dir_path.is_symlink()
    ]

    files, dirs = _scan_entries(path)
    for relpath in files:
        if relpath in expected_paths:
            continue
        extra_path = path / relpath
        if extra_path.is_symlink():
            changed[relpath] = _symlink_cache_key(extra_path, cache_root)
        else:
            changed[relpath], _ = hash_file(extra_path, state_db, pending=pending)
    # A hash pivot 0.3.3 recorded lists no empty directories, so on disk none of them is extra.
    extra_dirs = (
        set(dirs) - _restored_dirs(output_hash) if "empty_dirs" in output_hash else set[str]()
    )

    if changed or extra_dirs:
        # Replacing the directory deletes every file it holds; refuse if any is not cached.
        local_changes = sorted(
            rel for rel, content_hash in changed.items() if not _is_cached(cache_dir, content_hash)
        )
        if local_changes:
            return MissingRestore(restored=0, missing=0, local_changes=local_changes, replace=False)
        return MissingRestore(restored=0, missing=0, local_changes=[], replace=True)

    if any(not file_cache_path.exists() for _, _, file_cache_path in missing):
        return MissingRestore(restored=0, missing=len(missing), local_changes=[], replace=False)

    for entry, file_path, file_cache_path in missing:
        if before_write is not None:
            before_write()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        staged = _checkout_with_fallback(
            file_path,
            file_cache_path,
            checkout_modes,
            executable=entry["isexec"],
            cache_root=cache_root,
        )
        if pending is not None:
            pending.append((file_path, staged, entry["hash"]))
    for dir_path in missing_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)

    return MissingRestore(restored=len(missing), missing=0, local_changes=[], replace=False)


def protect(path: pathlib.Path) -> None:
    """Make file read-only (mode 0o444)."""
    os.chmod(path, 0o444)


def unprotect(path: pathlib.Path) -> None:
    """Restore write permission to file.

    WARNING: Do not use on files that might be hardlinks to the cache.
    Hardlinks share the same inode, so chmod would also change the cache
    file's permissions, corrupting its read-only invariant.
    """
    current = path.stat().st_mode
    os.chmod(path, current | stat.S_IWUSR)
