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
from typing import TYPE_CHECKING, TypedDict

import xxhash

from pivot import exceptions, metrics
from pivot.config import CheckoutMode as CheckoutMode
from pivot.storage import state as state_mod
from pivot.types import DirHash, DirManifestEntry, FileHash, HashInfo, is_dir_hash

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable

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
    lock_dir.mkdir(parents=True, exist_ok=True)
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
) -> None:
    """Atomically write to dest using a temporary file and replacement.

    A caller holding a restore lock can supply its target-tagged staging path so a
    later restore of the same target cleans up a process killed before replacement.

    Args:
        dest: Target file path.
        write_fn: Function that receives the file descriptor and writes content.
                  MUST close fd (typically via os.fdopen which takes ownership).
        mode: File permissions (default 0o644).
        staging_path: Optional deterministic temporary path beside ``dest``.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
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
        tmp.replace(dest)
    except BaseException:
        _clear_path(tmp)
        raise
    finally:
        # Only close fd if write_fn didn't (e.g., exception before os.fdopen)
        if not fd_closed:
            with contextlib.suppress(OSError):
                os.close(fd)


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

    A hash read from the file is saved to a writable ``state_db`` at once, unless ``pending``
    is given: then ``(path, stat, hash)`` is appended to it instead, for the caller to save in
    one transaction, or, when ``state_db`` is read-only, to hand to a process that can write.

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
        state_db.save(path, file_stat, file_hash)

    return file_hash, file_stat


def known_file_hashes(
    path: pathlib.Path, output_hash: HashInfo
) -> list[tuple[pathlib.Path, os.stat_result, str]]:
    """StateDB file-hash entries for an output on disk whose hash is already known.

    Stats each file (each manifest entry, for a directory) and pairs it with its known hash;
    nothing is read. Call it once the output has its final inode: a hardlink checkout
    replaces the file the stage wrote, and a restore replaces whatever was there.
    """
    if is_dir_hash(output_hash):
        files = [(path / entry["relpath"], entry["hash"]) for entry in output_hash["manifest"]]
    else:
        files = [(path, output_hash["hash"])]
    return [(file_path, file_path.stat(), file_hash) for file_path, file_hash in files]


def deferred_file_hash_entries(
    entries: Iterable[tuple[pathlib.Path, os.stat_result, str]],
) -> list[tuple[str, int, int, int, str]]:
    """``entries`` as ``DeferredWrites`` file-hash entries, for a process that cannot write the
    StateDB to hand to one that can.

    Each path is made absolute here, so the writer keys it the same from any working
    directory. An entry whose StateDB key exceeds LMDB's limit is left out: this cache only
    saves a later hash, so it must not fail the stage.
    """
    deferred = list[tuple[str, int, int, int, str]]()
    for path, file_stat, file_hash in entries:
        if not state_mod.file_hash_key_fits(path):
            logger.debug("Not caching the hash of %s: path too long", path)
            continue
        deferred.append(
            (
                os.path.abspath(path),
                file_stat.st_mtime_ns,
                file_stat.st_size,
                file_stat.st_ino,
                file_hash,
            )
        )
    return deferred


def _resolve_link(link: str | os.PathLike[str], cache_root: str) -> str:
    """Return ``os.path.realpath(link)`` for the symlink ``link``; raise OSError when ``link`` is
    not a symlink or is gone. ``cache_root`` is ``os.path.realpath`` of the cache's object root.

    realpath lstats every component of the link's path and of its target, in Python: at 180,000
    links that was most of the time a status took after a symlink checkout. A checkout link
    holds ``<cache_root>/XX/NAME``, the object's real path (``_stage_checkout``). Every
    component of ``cache_root`` is already real, so for that text realpath returns the text
    itself unless ``XX`` or ``NAME`` is a symlink: two lstats prove the answer. Testing the
    prefix alone would not: a link whose text lies in the cache still leaves it when ``XX`` or
    ``NAME`` is a symlink out of it. Any other text (relative, with ``..``, a link to another
    link, a path outside the cache) is resolved in full, so the answer is realpath's for every
    link. The link is read again on every call; only ``cache_root`` is resolved once per walk
    or restore, so a cache directory re-pointed between commands is seen by the next one.
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
            and not os.path.islink(prefix + prefix_dir)
            and not os.path.islink(target)
        ):
            return target
    return os.path.realpath(link)


def _links_to_object(link: pathlib.Path, cache_path: pathlib.Path, cache_root: str) -> bool:
    """Whether the symlink ``link`` resolves where the object path ``cache_path`` resolves, as
    ``link.resolve() == cache_path.resolve()`` answers. ``cache_path`` is
    ``get_cache_path(cache_dir, key)`` and ``cache_root`` is ``os.path.realpath(cache_dir)``.

    A link whose text is ``<cache_root>/XX/YYYY...``, the text a checkout writes, resolves
    wherever ``cache_path`` does, symlinks inside the cache included, so one readlink settles
    it. Any other link is resolved in full.
    """
    try:
        if os.readlink(link) == os.path.join(cache_root, cache_path.parent.name, cache_path.name):
            return True
    except OSError:
        pass
    return link.resolve() == cache_path.resolve()


def _symlink_cache_key(path: str | os.PathLike[str], cache_root: str) -> str | None:
    """Return ``XX + YYYY...`` when the symlink ``path`` resolves to ``<cache_root>/XX/YYYY...``
    with 16 characters in all, else None (``path`` is not a symlink, or resolves elsewhere).
    ``cache_root`` is ``os.path.realpath`` of the directory holding the objects.
    """
    try:
        target = _resolve_link(path, cache_root)
    except (OSError, ValueError):
        return None
    prefix = os.path.join(cache_root, "")
    if not target.startswith(prefix):
        return None
    parts = target[len(prefix) :].split("/")
    if len(parts) != 2:
        return None
    key = parts[0] + parts[1]
    return key if len(key) == XXHASH64_HEX_LENGTH else None


def _cache_object_key(path: str | os.PathLike[str], cache_root: str | None) -> str | None:
    """Return the key of the cache object the symlink ``path`` stands for, or None.

    A symlink is the cache object ``K`` when following it, through every link on the way, ends at
    a regular file whose real path is ``<cache_root>/K[:2]/K[2:]``, and ``K`` is a cache key. So
    a link to a link to the object, and a relative link, count, as they did when this resolved
    the whole path with ``resolve()``; ``_resolve_link`` gives the same answer cheaply for the
    link a checkout writes. A link that ends outside the cache (its first step inside it or not),
    at another name in it, at a directory, at nothing, or in a loop is a user symlink, even when
    it ends at a file named like a hash. A file-by-file symlink checkout restores each directory
    member as a link to its object. ``cache_root`` is ``os.path.realpath`` of the cache's object
    root (``<cache.dir>/files``), resolved once by the caller, or None for no cache.
    """
    if cache_root is None:
        return None
    key = _symlink_cache_key(path, cache_root)
    if key is None or not is_valid_hash(key) or not os.path.isfile(path):
        return None
    return key


def is_artifact_member_content(path: pathlib.Path, file_hash: str, cache_root: str | None) -> bool:
    """Whether a candidate member has the content its cache-object symlink names.

    ``cache_root`` is ``os.path.realpath`` of the cache's object root, or None for no cache.
    """
    return not path.is_symlink() or file_hash == _cache_object_key(path, cache_root)


def walk_artifact_members(
    directory: str,
    prefix: str,
    cache_root: str | None,
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
    file it collects lies inside the directory. PermissionError propagates: a partial hash would
    be wrong.

    Args:
        directory: Directory to scan
        prefix: Relpath of ``directory`` under the artifact root, with a trailing ``/``
            (empty at the root)
        cache_root: ``os.path.realpath`` of the cache object root whose links count as content,
            resolved once for the whole walk, or None for no cache
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
    cache_root = None if cache_dir is None else os.path.realpath(cache_dir)
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
        state_db.save_many(read)

    if left_out and warn_left_out_symlinks:
        log_left_out_symlinks(path, left_out)

    # A deepest empty directory holds no stored file and no other directory.
    found_dirs = set(dirs)
    file_dirs = {posixpath.dirname(entry["relpath"]) for entry in manifest}
    has_child = {posixpath.dirname(directory) for directory in found_dirs}
    empty_dirs = sorted(found_dirs - file_dirs - has_child)

    hashed: object = {"empty_dirs": empty_dirs, "manifest": manifest} if empty_dirs else manifest
    hashed_json = json.dumps(hashed, sort_keys=True, separators=(",", ":"))
    output_hash = DirHash(hash=xxhash.xxh64(hashed_json.encode()).hexdigest(), manifest=manifest)
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
    """Get cache path for a hash (XX/XXXX... structure)."""
    if len(file_hash) != XXHASH64_HEX_LENGTH:
        raise exceptions.SecurityValidationError(
            f"Hash must be exactly {XXHASH64_HEX_LENGTH} characters, got {len(file_hash)}"
        )
    # Prevents path traversal via malicious hashes
    if not is_valid_hash(file_hash):
        raise exceptions.SecurityValidationError(f"Hash contains invalid characters: {file_hash!r}")
    return cache_dir / file_hash[:2] / file_hash[2:]


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


def _cleanup_stale_restore_temps(parent_dir: pathlib.Path) -> None:
    """Remove stale restore and backup artifacts older than max age."""
    cutoff = time.time() - _RESTORE_TEMP_MAX_AGE

    if not parent_dir.exists():
        return

    for entry in os.scandir(parent_dir):
        if entry.name.startswith(_RESTORE_LOCK_PREFIX):
            continue
        if not (
            entry.name.startswith(_RESTORE_TEMP_PREFIX)
            or entry.name.startswith(_BACKUP_TEMP_PREFIX)
        ):
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


def _cleanup_fresh_restore_orphans(path: pathlib.Path, token: str) -> None:
    """Remove fresh temp and backup entries left by an interrupted restore of path.

    The caller holds path's restore lock, so entries tagged with token cannot belong to
    a live restore of the same target.
    """
    prefixes = (
        f"{_RESTORE_TEMP_PREFIX}{token}.",
        f"{_BACKUP_TEMP_PREFIX}{token}.",
    )
    for entry in os.scandir(path.parent):
        if entry.name.startswith(prefixes):
            _clear_path(pathlib.Path(entry.path))


def _get_symlink_cache_hash(path: pathlib.Path, cache_dir: pathlib.Path) -> str | None:
    """Extract hash from symlink target if it points to cache, else None.

    For a caller that checks one path: it resolves ``cache_dir`` itself. A caller that checks
    many links resolves the cache directory once and calls ``_symlink_cache_key``.
    """
    try:
        cache_root = os.path.realpath(cache_dir)
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
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with _staging_in(cache_path.parent):
        fd, name = tempfile.mkstemp(prefix=_CACHE_TEMP_PREFIX, dir=cache_path.parent)
        os.close(fd)
        staged = pathlib.Path(name)
        try:
            yield staged
        finally:
            staged.unlink(missing_ok=True)


def copy_to_cache(src: pathlib.Path, cache_path: pathlib.Path, *, executable: bool = False) -> None:
    """Atomically copy file to cache with read-only permissions.

    ``executable`` records the source's executable bit on the cache object itself (0o555
    instead of 0o444): a HARDLINK or SYMLINK checkout has no mode of its own, so it reads
    straight through to whatever mode the cache object carries, and only COPY checkout
    re-chmods afterwards (see ``_stage_checkout``).

    The object is published with ``publish_cache_object`` and never changed afterwards, its
    mode included. An object already there is kept whatever executable bit it carries; a
    directory entry needing the other bit is checked out as a copy (``_checkout_with_fallback``).
    """
    if cache_path.exists():
        return

    with staged_cache_file(cache_path) as staged:
        shutil.copy2(src, staged)
        os.chmod(staged, 0o555 if executable else 0o444)
        _ = publish_cache_object(staged, cache_path)


# A link the filesystem cannot make (across devices, refused, or unsupported) falls through to
# the next checkout mode. EMLINK is not here: it is one object's limit, not the filesystem's
# (see _checkout_with_fallback).
FALLBACK_ERRNO = _NO_HARDLINK_ERRNO | {errno.EXDEV, errno.EACCES}


def _object_realpath(cache_path: pathlib.Path, cache_root: str | None) -> str:
    """Return ``os.path.realpath(cache_path)`` for ``cache_path = get_cache_path(cache_dir, key)``.

    ``cache_root`` is ``os.path.realpath(cache_dir)``, resolved once by a caller restoring many
    files, or None. The object's path under ``cache_root`` is already real unless its
    two-character directory or its own name is a symlink (see ``_resolve_link``), so two lstats
    replace realpath's walk of every component.
    """
    if cache_root is not None:
        real = os.path.join(cache_root, cache_path.parent.name, cache_path.name)
        if not os.path.islink(os.path.dirname(real)) and not os.path.islink(real):
            return real
    return os.path.realpath(cache_path)


def _stage_checkout(
    tmp_path: pathlib.Path,
    cache_path: pathlib.Path,
    checkout_mode: CheckoutMode,
    *,
    executable: bool | None,
    cache_root: str | None = None,
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
    cache_root: str | None = None,
) -> None:
    """Create link or copy at path atomically, so readers never see a partial file.

    Args:
        staging_path: Where to build the link or copy before renaming it over path.
            Defaults to a fresh random name beside path.
        cache_root: ``os.path.realpath`` of the cache directory holding ``cache_path``, from a
            caller restoring many files, so a symlink's target is not resolved per file.
    """
    # Idempotency: skip if already correctly linked
    if checkout_mode == CheckoutMode.SYMLINK and path.is_symlink():
        with contextlib.suppress(OSError):
            if path.resolve() == cache_path.resolve():
                return
    elif checkout_mode == CheckoutMode.HARDLINK and path.exists() and not path.is_symlink():
        with contextlib.suppress(OSError):
            if path.stat().st_ino == cache_path.stat().st_ino:
                return

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
        try:
            tmp_path.replace(path)
        except IsADirectoryError:
            _clear_path(path)
            tmp_path.replace(path)
    finally:
        # Also covers a replace() that did nothing: POSIX rename is a no-op when both
        # names are hardlinks to the same inode, leaving tmp_path behind.
        tmp_path.unlink(missing_ok=True)


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


# The marker's earlier name, which pivot 0.3.4 touched after checking only a tree's layout,
# partial trees included: every tree is checked once more and the old marker dropped.
_TREE_LAYOUT_MARKER_SUFFIX = ".verified"
_LEGACY_TREE_MARKER_SUFFIX = ".layout-checked"
# Held exclusively while a tree is checked and repaired; never deleted.
_TREE_LOCK_SUFFIX = ".lock"


def _tree_layout_marker(cache_dir_path: pathlib.Path) -> pathlib.Path:
    """Return the file beside a whole-directory SYMLINK cache tree that records the tree holds
    exactly what its hash records. The name is never a cache key, so cache scans skip it.
    """
    return cache_dir_path.with_name(cache_dir_path.name + _TREE_LAYOUT_MARKER_SUFFIX)


def _mark_tree(cache_dir_path: pathlib.Path) -> None:
    """Mark a tree known to hold exactly what its hash records. The marker only saves a check,
    so a cache the user cannot write still restores without one."""
    with contextlib.suppress(OSError):
        _tree_layout_marker(cache_dir_path).touch()
    with contextlib.suppress(OSError):
        cache_dir_path.with_name(cache_dir_path.name + _LEGACY_TREE_MARKER_SUFFIX).unlink(
            missing_ok=True
        )


def _tree_matches(
    cache_dir_path: pathlib.Path, output_hash: DirHash, scanned: tuple[list[str], list[str]]
) -> bool:
    """Whether a whole-directory SYMLINK cache tree holds exactly the files (each regular, at its
    recorded size and executable bit) and the directories ``output_hash`` records. ``scanned``
    is ``_scan_entries(cache_dir_path)``. A recorded path leading outside the tree raises:
    ``output_hash`` may come straight from a lock file."""
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
    for entry in manifest:
        found_stat = os.lstat(cache_dir_path / entry["relpath"])
        if (
            not stat.S_ISREG(found_stat.st_mode)
            or found_stat.st_size != entry["size"]
            or bool(found_stat.st_mode & stat.S_IXUSR) != entry["isexec"]
        ):
            return False
    return True


def _ensure_tree_complete(
    cache_dir_path: pathlib.Path, output_hash: DirHash, cache_dir: pathlib.Path
) -> bool:
    """Return whether a whole-directory SYMLINK cache tree holds exactly what ``output_hash``
    records, checking a tree once and repairing it in place from its files' own cache objects.

    The tree publisher marks a tree once it is complete (``_publish_tree``). A tree without the
    mark is checked: one an older pivot built, which can hold entries its hash does not record
    and 0o444-everywhere files, or one it copied straight to its name and was killed while
    copying, leaving files missing or cut short under a name a restore trusts. The check, the
    repair and the mark run under the tree's exclusive lock, so two repairers never change the
    tree's directory modes under each other. Returns False, the tree unmarked, when the tree
    differs and cannot be repaired: a file's object is absent, or this user cannot write the
    tree or take its lock.
    """
    marker = _tree_layout_marker(cache_dir_path)
    if marker.exists():
        return True
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
        if marker.exists():
            return True
        scanned = _scan_entries(cache_dir_path)
        if not _tree_matches(cache_dir_path, output_hash, scanned):
            if fd is None:
                return False
            try:
                _repair_tree_dir_isexec(cache_dir_path, output_hash["manifest"])
                _repair_tree_dir_layout(cache_dir_path, output_hash, scanned)
                if not _repair_tree_dir_files(cache_dir_path, output_hash["manifest"], cache_dir):
                    return False
            except PermissionError:  # another user's tree
                return False
        _mark_tree(cache_dir_path)
        return True
    finally:
        if fd is not None:
            os.close(fd)


def _repair_tree_dir_files(
    cache_dir_path: pathlib.Path, manifest: list[DirManifestEntry], cache_dir: pathlib.Path
) -> bool:
    """Give a whole-directory SYMLINK cache tree every file its manifest records, at its
    recorded size, copied from the file's own cache object. Return False, changing nothing, when
    a file is missing or the wrong size and its object is not cached.

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
            if stat.S_ISREG(target_stat.st_mode) and target_stat.st_size == entry["size"]:
                continue
        wrong.append((entry, target))
    if not wrong:
        return True
    objects = [get_cache_path(cache_dir, entry["hash"]) for entry, _ in wrong]
    if not all(obj.is_file() for obj in objects):
        return False
    modes = dict[pathlib.Path, int]()

    def _make_writable(directory: pathlib.Path) -> None:
        if directory not in modes:
            modes[directory] = directory.stat().st_mode
            os.chmod(directory, modes[directory] | stat.S_IWUSR)

    try:
        for (entry, target), obj in zip(wrong, objects, strict=True):
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
    cache_root: str | None = None,
) -> None:
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
    defaults to ``path`` for every other caller.

    ``cache_root`` is ``os.path.realpath`` of the cache directory holding ``cache_path``, from a
    caller restoring many files: a symlink then names the object's real path without resolving
    it per file (``_object_realpath``). None resolves it.
    """
    if not checkout_modes:
        raise ValueError("checkout_modes cannot be empty")
    display = display_path if display_path is not None else path
    for i, mode in enumerate(checkout_modes):
        is_last = i == len(checkout_modes) - 1
        if keep_existing_copy and mode == CheckoutMode.COPY and path.exists():
            return
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
            _checkout_from_cache(
                path,
                cache_path,
                mode,
                executable=executable,
                staging_path=staging_path,
                cache_root=cache_root,
            )
            return
        except OSError as e:
            if mode == CheckoutMode.HARDLINK and e.errno == errno.EMLINK:
                logger.debug(f"Cache object {cache_path} is at the hard-link limit; copying it")
                if not (keep_existing_copy and path.exists()):
                    _checkout_from_cache(
                        path,
                        cache_path,
                        CheckoutMode.COPY,
                        executable=executable,
                        staging_path=staging_path,
                    )
                return
            if e.errno not in FALLBACK_ERRNO or is_last:
                raise
            logger.debug(f"Checkout mode {mode.value} failed ({e}), trying next mode")


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
    """Return hashes required to restore an output that are absent from the cache."""
    if is_dir_hash(output_hash):
        tree_path = get_cache_path(cache_dir, output_hash["hash"])
        if checkout_modes[0] == CheckoutMode.SYMLINK and tree_path.exists():
            return []
        hashes = [entry["hash"] for entry in output_hash["manifest"]]
    else:
        hashes = [output_hash["hash"]]
    return [file_hash for file_hash in hashes if not get_cache_path(cache_dir, file_hash).exists()]


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
) -> None:
    cache_path = get_cache_path(cache_dir, file_hash)
    copy_to_cache(path, cache_path)
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
    # Idempotency check for SYMLINK mode: already linked to the tree its content hashes to
    if (
        checkout_modes
        and checkout_modes[0] == CheckoutMode.SYMLINK
        and path.is_symlink()
        and _get_symlink_cache_hash(path, cache_dir) == output_hash["hash"]
        and get_cache_path(cache_dir, output_hash["hash"]).exists()
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

    The tree is marked only once it is checked to hold what ``output_hash`` records: a sweep on
    another host whose flock is not coherent with this host's can remove files from the copy
    while it is built (see storage/AGENTS.md). An unmarked tree is checked and repaired by the
    next commit or restore through it (``_ensure_tree_complete``).
    """
    manifest = output_hash["manifest"]

    cache_root = os.path.realpath(cache_dir)

    def _user_symlinks(directory: str, names: list[str]) -> list[str]:
        # Not stored (see walk_artifact_members); copytree would copy each target.
        return [
            name
            for name in names
            if (link := pathlib.Path(directory, name)).is_symlink()
            and _cache_object_key(link, cache_root) is None
        ]

    cache_dir_path.parent.mkdir(parents=True, exist_ok=True)
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
                return
            if _tree_matches(cache_dir_path, output_hash, _scan_entries(cache_dir_path)):
                _mark_tree(cache_dir_path)
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
) -> None:
    manifest = output_hash["manifest"]
    # Cache individual files first
    for entry in manifest:
        file_path = path / entry["relpath"]
        cache_path = get_cache_path(cache_dir, entry["hash"])
        copy_to_cache(file_path, cache_path, executable=entry["isexec"])

    with restore_lock(path, state_dir):
        if checkout_modes and checkout_modes[0] == CheckoutMode.SYMLINK:
            # SYMLINK mode: cache entire directory, symlink to it
            cache_dir_path = get_cache_path(cache_dir, output_hash["hash"])
            if not cache_dir_path.exists():
                _publish_tree(path, cache_dir_path, output_hash, cache_dir)

            # A tree an older pivot built keeps its 0o444-everywhere files, entries its hash does
            # not record, or files missing from a copy it was killed in, forever otherwise: its name
            # never changes, so this branch never revisits an existing tree directory above.
            if not _ensure_tree_complete(cache_dir_path, output_hash, cache_dir):
                raise exceptions.CacheRestoreError(
                    f"Cache tree {cache_dir_path} differs from what its hash records and cannot "
                    + "be repaired (a file's cache object is missing, or another user owns it); "
                    + "remove the tree, or have its owner remove it, to let this commit publish "
                    + "it again"
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


def store_hashed_output(
    path: pathlib.Path,
    output_hash: HashInfo,
    cache_dir: pathlib.Path,
    checkout_modes: list[CheckoutMode] | None = None,
    *,
    state_dir: pathlib.Path | None = None,
) -> None:
    """Put an output whose hash is already known into the cache, then link it per mode.

    ``output_hash`` must be the hash of ``path``'s current contents; nothing is hashed again.
    Each file is copied into the cache once and the workspace path is linked to the cache entry
    with the first checkout mode that works. Reaching COPY leaves the workspace file as it is,
    since it already holds the cached bytes.
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
        )
    else:
        _store_file(
            path,
            output_hash["hash"],
            cache_dir,
            effective_modes,
            keep_existing_copy=True,
            state_dir=state_dir,
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
        )
    return _restore_file_from_cache(
        path, output_hash, cache_dir, effective_modes, state_dir, lock_token=lock_token
    )


def _restore_file_from_cache(
    path: pathlib.Path,
    output_hash: FileHash,
    cache_dir: pathlib.Path,
    checkout_modes: list[CheckoutMode],
    state_dir: pathlib.Path | None = None,
    *,
    lock_token: str | None = None,
) -> bool:
    """Restore single file from cache.

    With a state_dir, holds path's restore lock and stages the file under one name
    tagged with path's token, so the next restore removes what a killed one left.
    """
    cache_path = get_cache_path(cache_dir, output_hash["hash"])
    if not cache_path.exists():
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    if state_dir is None:
        _checkout_with_fallback(path, cache_path, checkout_modes)
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
            )

    staging_path = restore_staging_path(path, lock_token)
    _clear_path(staging_path)
    _checkout_with_fallback(path, cache_path, checkout_modes, staging_path=staging_path)
    return True


def _restore_directory_from_cache(
    path: pathlib.Path,
    output_hash: DirHash,
    cache_dir: pathlib.Path,
    checkout_modes: list[CheckoutMode],
    state_dir: pathlib.Path | None = None,
    *,
    lock_token: str | None = None,
) -> bool:
    """Restore a directory atomically."""
    cache_dir_path = get_cache_path(cache_dir, output_hash["hash"])
    symlink_fast_path = checkout_modes[0] == CheckoutMode.SYMLINK and cache_dir_path.exists()
    if not symlink_fast_path:
        for entry in output_hash["manifest"]:
            if not get_cache_path(cache_dir, entry["hash"]).exists():
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
            )

    target_token = lock_token or _restore_target_token(path)
    if lock_token is not None:
        _cleanup_fresh_restore_orphans(path, target_token)
    restore_prefix = f"{_RESTORE_TEMP_PREFIX}{target_token}."

    if symlink_fast_path and not _ensure_tree_complete(cache_dir_path, output_hash, cache_dir):
        # Same check as _store_directory. A tree this user cannot repair is not used: the
        # output is restored file by file from the objects instead, if they are cached.
        logger.warning(
            "Cache tree %s differs from what its hash records and cannot be repaired; "
            + "restoring %s file by file",
            cache_dir_path,
            path,
        )
        symlink_fast_path = False
        for entry in output_hash["manifest"]:
            if not get_cache_path(cache_dir, entry["hash"]).exists():
                return False

    if symlink_fast_path:
        temp_symlink = path.parent / f"{restore_prefix}symlink_{secrets.token_hex(4)}"
        try:
            temp_symlink.symlink_to(cache_dir_path.resolve())
            _clear_path(path)
            temp_symlink.replace(path)
            return True
        except OSError as e:
            logger.debug(f"Symlink creation failed, will try file-by-file: {e}")
            if temp_symlink.is_symlink():
                temp_symlink.unlink()
            for entry in output_hash["manifest"]:
                if not get_cache_path(cache_dir, entry["hash"]).exists():
                    return False

    temp_path = pathlib.Path(tempfile.mkdtemp(prefix=restore_prefix, dir=path.parent))
    backup_path = path.parent / f"{_BACKUP_TEMP_PREFIX}{target_token}.{secrets.token_hex(4)}"
    swap_completed = False
    backup_restore_failed = False
    try:
        resolved_temp = temp_path.resolve()
        cache_root = os.path.realpath(cache_dir)
        for file_count, entry in enumerate(output_hash["manifest"], start=1):
            file_cache_path = get_cache_path(cache_dir, entry["hash"])
            file_path = temp_path / entry["relpath"]
            if not file_path.resolve().is_relative_to(resolved_temp):
                raise exceptions.SecurityValidationError(
                    f"Manifest contains path traversal: {entry['relpath']!r}"
                )
            file_path.parent.mkdir(parents=True, exist_ok=True)
            _checkout_with_fallback(
                file_path,
                file_cache_path,
                checkout_modes,
                executable=entry["isexec"],
                display_path=path / entry["relpath"],
                cache_root=cache_root,
            )
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

        try:
            if path.exists() or path.is_symlink():
                path.rename(backup_path)
        except FileNotFoundError:
            logger.debug("Path disappeared between check and rename, proceeding")

        temp_path.replace(path)
        swap_completed = True

        try:
            _clear_path(backup_path)
        except OSError as e:
            logger.debug(f"Backup cleanup failed (non-critical): {e}")
        return True
    except BaseException:
        if backup_path.exists() and not path.exists():
            try:
                backup_path.rename(path)
            except OSError as e:
                logger.debug(f"Failed to restore backup: {e}")
                backup_restore_failed = True
        raise
    finally:
        if temp_path.exists():
            _clear_path(temp_path)
        if backup_path.exists() and swap_completed and not backup_restore_failed:
            _clear_path(backup_path)


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
    return (
        content_hash is not None
        and is_valid_hash(content_hash)
        and get_cache_path(cache_dir, content_hash).exists()
    )


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
    cache_root = os.path.realpath(cache_dir)
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

        file_cache_path = get_cache_path(cache_dir, entry["hash"])
        if file_path.is_symlink():
            if not file_path.parent.resolve().is_relative_to(resolved_base):
                raise exceptions.SecurityValidationError(
                    f"Manifest contains path traversal: {relpath!r}"
                )
            if _links_to_object(file_path, file_cache_path, cache_root):
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
            if _is_same_file(file_stat, file_cache_path):
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
        file_path.parent.mkdir(parents=True, exist_ok=True)
        _checkout_with_fallback(
            file_path,
            file_cache_path,
            checkout_modes,
            executable=entry["isexec"],
            cache_root=cache_root,
        )
        if pending is not None:
            pending.append((file_path, file_path.stat(), entry["hash"]))
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
