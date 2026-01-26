from __future__ import annotations

import logging
import os
import pathlib
import struct
from typing import TYPE_CHECKING, Self

import lmdb

from pivot import run_history

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pivot.run_history import RunCacheEntry, RunManifest
    from pivot.types import DeferredWrites

# Key prefixes for different entry types
_HASH_PREFIX = b"hash:"  # File hash entries
_GEN_PREFIX = b"gen:"  # Output generation counters
_DEP_PREFIX = b"dep:"  # Stage dependency generations
_REMOTE_PREFIX = b"remote:"  # Remote index entries
_RUN_PREFIX = b"run:"  # Run history entries
_RUNCACHE_PREFIX = b"runcache:"  # Run cache entries for skip detection
_FP_PREFIX = b"fp:"  # AST fingerprint/hash cache entries

# Default LMDB map size (10GB virtual - grows as needed)
_MAP_SIZE = 10 * 1024 * 1024 * 1024

# LMDB default max key size
_MAX_KEY_SIZE = 511

# Common error message for MapFullError
_DB_FULL_MSG = (
    f"State cache is full ({_MAP_SIZE // (1024**3)}GB limit). Delete .pivot/state.lmdb/ to reset."
)


class StateDBError(Exception):
    """Base exception for StateDB errors."""


class PathTooLongError(StateDBError):
    """Raised when a file path exceeds LMDB's key size limit."""


class DatabaseFullError(StateDBError):
    """Raised when the state database reaches its size limit."""


def _make_key_file_hash(path: pathlib.Path) -> bytes:
    """Create LMDB key for file hash entry (follows symlinks for physical deduplication).

    Uses resolve() to follow symlinks because hash caching is about physical file identity.
    Multiple symlinks pointing to the same file should share one cached hash.
    Contrast with _make_key_output_generation() which preserves symlinks for logical path tracking.
    """
    return _HASH_PREFIX + str(path.resolve()).encode()


def _make_key_output_generation(path: pathlib.Path) -> bytes:
    """Create LMDB key for output generation entry (preserves symlinks for logical path tracking).

    Uses normpath(absolute()), NOT resolve(), because Pivot outputs become symlinks
    to cache after execution. resolve() would follow these symlinks to cache paths
    that change per-run. We track the LOGICAL path the user declared.
    Contrast with _make_key_file_hash() which follows symlinks for physical deduplication.
    """
    return _GEN_PREFIX + os.path.normpath(path.absolute()).encode()


def _make_key_dep_generation(stage_name: str, dep_path: str) -> bytes:
    """Create LMDB key for dependency generation record (stage + dep path)."""
    return _DEP_PREFIX + f"{stage_name}:{dep_path}".encode()


class InvalidAstHashKeyError(Exception):
    """Raised when AST hash key components contain invalid characters."""


def _make_key_ast_hash(rel_path: str, mtime_ns: int, size: int, inode: int, qualname: str) -> bytes:
    """Create LMDB key for AST hash entry.

    Key format: fp:{rel_path}\x00{mtime_ns}\x00{size}\x00{inode}\x00{qualname}
    Uses null byte separator (can't appear in paths or qualnames).

    Raises:
        InvalidAstHashKeyError: If rel_path or qualname contains null bytes or is empty
    """
    if not rel_path:
        raise InvalidAstHashKeyError("rel_path cannot be empty")
    if not qualname:
        raise InvalidAstHashKeyError("qualname cannot be empty")
    if "\x00" in rel_path:
        raise InvalidAstHashKeyError(f"rel_path contains null byte: {rel_path!r}")
    if "\x00" in qualname:
        raise InvalidAstHashKeyError(f"qualname contains null byte: {qualname!r}")

    key = _FP_PREFIX + f"{rel_path}\x00{mtime_ns}\x00{size}\x00{inode}\x00{qualname}".encode()
    if len(key) > _MAX_KEY_SIZE:
        raise InvalidAstHashKeyError(f"key exceeds {_MAX_KEY_SIZE} bytes: {len(key)}")
    return key


def _pack_value(mtime_ns: int, size: int, inode: int, hash_hex: str) -> bytes:
    """Pack metadata and hash into binary value."""
    return struct.pack(">QQQ", mtime_ns, size, inode) + hash_hex.encode("ascii")


def _unpack_value(data: bytes) -> tuple[int, int, int, str]:
    """Unpack binary value into metadata and hash."""
    mtime_ns, size, inode = struct.unpack(">QQQ", data[:24])
    hash_hex = data[24:].decode("ascii")
    return mtime_ns, size, inode, hash_hex


def _match_cached_hash(value: bytes | None, fs_stat: os.stat_result) -> str | None:
    """Return cached hash if stored metadata matches fs_stat, else None."""
    if value is None:
        return None
    mtime_ns, size, inode, hash_hex = _unpack_value(value)
    if fs_stat.st_mtime_ns == mtime_ns and fs_stat.st_size == size and fs_stat.st_ino == inode:
        return hash_hex
    return None


class StateDB:
    """LMDB cache of file hashes and generation counters.

    Multi-process safe: Each process independently instantiates StateDB and opens
    the same LMDB environment file. LMDB handles inter-process synchronization via
    file-system locking (MVCC). Workers open in readonly mode to avoid write contention;
    state changes are deferred and applied atomically by the coordinator.

    Concurrent `pivot run` invocations are safe—each gets its own StateDB instances
    with automatic MVCC snapshot isolation. Readers never block writers and vice versa.
    """

    _env: lmdb.Environment
    _closed: bool
    _readonly: bool

    def __init__(self, db_path: pathlib.Path, readonly: bool = False) -> None:
        lmdb_path = db_path.parent / "state.lmdb"
        lmdb_path.parent.mkdir(parents=True, exist_ok=True)

        # LMDB readonly mode can't create database - create empty one first if needed
        if readonly and not lmdb_path.exists():
            lmdb.open(str(lmdb_path), map_size=_MAP_SIZE).close()

        self._env = lmdb.open(str(lmdb_path), map_size=_MAP_SIZE, readonly=readonly)
        self._closed = False
        self._readonly = readonly

    def _check_closed(self) -> None:
        """Raise if database is closed."""
        if self._closed:
            raise RuntimeError("Cannot operate on closed StateDB")

    def _check_write_allowed(self) -> None:
        """Raise if database is read-only."""
        if self._readonly:
            raise RuntimeError(
                "Internal error: worker attempted write to readonly StateDB. This is a bug in Pivot. Please report it."
            )

    @property
    def readonly(self) -> bool:
        """Return True if database is opened in readonly mode."""
        return self._readonly

    def get(self, path: pathlib.Path, fs_stat: os.stat_result) -> str | None:
        """Return cached hash if file metadata matches, else None."""
        self._check_closed()
        with self._env.begin() as txn:
            value = txn.get(_make_key_file_hash(path))
        return _match_cached_hash(value, fs_stat)

    def get_many(
        self, items: list[tuple[pathlib.Path, os.stat_result]]
    ) -> dict[pathlib.Path, str | None]:
        """Batch query for multiple files."""
        self._check_closed()
        if not items:
            return {}
        results = dict[pathlib.Path, str | None]()
        with self._env.begin() as txn:
            for path, fs_stat in items:
                value = txn.get(_make_key_file_hash(path))
                results[path] = _match_cached_hash(value, fs_stat)
        return results

    def save(self, path: pathlib.Path, fs_stat: os.stat_result, file_hash: str) -> None:
        """Cache file metadata and hash."""
        self._check_closed()
        self._check_write_allowed()
        key = _make_key_file_hash(path)
        if len(key) > _MAX_KEY_SIZE:
            raise PathTooLongError(
                f"Path too long for state cache ({len(key)} bytes, max {_MAX_KEY_SIZE}): {path}"
            )
        value = _pack_value(fs_stat.st_mtime_ns, fs_stat.st_size, fs_stat.st_ino, file_hash)
        try:
            with self._env.begin(write=True) as txn:
                txn.put(key, value)
        except lmdb.MapFullError as e:
            raise DatabaseFullError(_DB_FULL_MSG) from e

    def save_many(self, entries: list[tuple[pathlib.Path, os.stat_result, str]]) -> None:
        """Batch save multiple entries atomically."""
        self._check_closed()
        self._check_write_allowed()
        try:
            with self._env.begin(write=True) as txn:
                for path, fs_stat, file_hash in entries:
                    key = _make_key_file_hash(path)
                    if len(key) > _MAX_KEY_SIZE:
                        raise PathTooLongError(
                            f"Path too long for state cache ({len(key)} bytes, max {_MAX_KEY_SIZE}): {path}"
                        )
                    value = _pack_value(
                        fs_stat.st_mtime_ns, fs_stat.st_size, fs_stat.st_ino, file_hash
                    )
                    txn.put(key, value)
        except lmdb.MapFullError as e:
            raise DatabaseFullError(_DB_FULL_MSG) from e

    # -------------------------------------------------------------------------
    # AST hash cache for persistent fingerprint caching
    # -------------------------------------------------------------------------

    def get_ast_hash(
        self, rel_path: str, mtime_ns: int, size: int, inode: int, qualname: str
    ) -> str | None:
        """Get cached AST hash; returns None if not found or key is invalid."""
        self._check_closed()
        try:
            key = _make_key_ast_hash(rel_path, mtime_ns, size, inode, qualname)
        except InvalidAstHashKeyError:
            return None
        with self._env.begin() as txn:
            value = txn.get(key)
        if value is None:
            return None
        return value.decode("ascii")

    def save_ast_hash_many(self, entries: list[tuple[str, int, int, int, str, str]]) -> None:
        """Batch save AST hash entries; skips entries with invalid keys."""
        self._check_closed()
        self._check_write_allowed()
        if not entries:
            return
        try:
            with self._env.begin(write=True) as txn:
                for rel_path, mtime_ns, size, inode, qualname, hash_hex in entries:
                    try:
                        key = _make_key_ast_hash(rel_path, mtime_ns, size, inode, qualname)
                    except InvalidAstHashKeyError:
                        continue  # Skip invalid entries
                    txn.put(key, hash_hex.encode("ascii"))
        except lmdb.MapFullError as e:
            raise DatabaseFullError(_DB_FULL_MSG) from e

    # -------------------------------------------------------------------------
    # Generation tracking for O(1) skip detection
    # -------------------------------------------------------------------------

    def get_generation(self, path: pathlib.Path) -> int | None:
        """Get generation counter for an output path. Returns None if not tracked."""
        self._check_closed()
        key = _make_key_output_generation(path)
        with self._env.begin() as txn:
            value = txn.get(key)
        if value is None:
            return None
        return struct.unpack(">Q", value)[0]

    def get_many_generations(self, paths: list[pathlib.Path]) -> dict[pathlib.Path, int | None]:
        """Batch query for multiple path generations."""
        self._check_closed()
        if not paths:
            return {}
        results = dict[pathlib.Path, int | None]()
        with self._env.begin() as txn:
            for path in paths:
                key = _make_key_output_generation(path)
                value = txn.get(key)
                if value is None:
                    results[path] = None
                else:
                    results[path] = struct.unpack(">Q", value)[0]
        return results

    def increment_generation(self, path: pathlib.Path) -> int:
        """Increment and return new generation (creates with gen=1 if not exists)."""
        self._check_closed()
        self._check_write_allowed()
        key = _make_key_output_generation(path)
        if len(key) > _MAX_KEY_SIZE:
            raise PathTooLongError(
                f"Path too long for generation tracking ({len(key)} bytes, max {_MAX_KEY_SIZE}): {path}"
            )
        try:
            with self._env.begin(write=True) as txn:
                value = txn.get(key)
                new_gen = (struct.unpack(">Q", value)[0] + 1) if value else 1
                txn.put(key, struct.pack(">Q", new_gen))
        except lmdb.MapFullError as e:
            raise DatabaseFullError(_DB_FULL_MSG) from e
        return new_gen

    def get_dep_generations(self, stage_name: str) -> dict[str, int] | None:
        """Get recorded dependency generations for a stage. Returns None if no record."""
        self._check_closed()
        prefix = _DEP_PREFIX + stage_name.encode() + b":"
        results = dict[str, int]()
        with self._env.begin() as txn:
            cursor = txn.cursor()
            if cursor.set_range(prefix):
                for key, value in cursor:
                    if not key.startswith(prefix):
                        break
                    dep_path = key[len(prefix) :].decode()
                    generation = struct.unpack(">Q", value)[0]
                    results[dep_path] = generation
        return results if results else None

    def record_dep_generations(self, stage_name: str, deps: dict[str, int]) -> None:
        """Record dependency generations after successful stage execution."""
        self._check_closed()
        self._check_write_allowed()
        prefix = _DEP_PREFIX + stage_name.encode() + b":"
        for dep_path in deps:
            key = _make_key_dep_generation(stage_name, dep_path)
            if len(key) > _MAX_KEY_SIZE:
                raise PathTooLongError(
                    f"Dependency path too long for tracking ({len(key)} bytes, max {_MAX_KEY_SIZE}): {dep_path}"
                )
        try:
            with self._env.begin(write=True) as txn:
                cursor = txn.cursor()
                keys_to_delete = list[bytes]()
                if cursor.set_range(prefix):
                    for key, _ in cursor:
                        if not key.startswith(prefix):
                            break
                        keys_to_delete.append(key)
                for key in keys_to_delete:
                    txn.delete(key)
                for dep_path, gen in deps.items():
                    key = _make_key_dep_generation(stage_name, dep_path)
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
        with self._env.begin() as txn:
            return txn.get(key) is not None

    def remote_hashes_intersection(self, remote_name: str, hashes: set[str]) -> set[str]:
        """Return subset of hashes known to exist on remote (batch lookup)."""
        self._check_closed()
        if not hashes:
            return set()
        prefix = _REMOTE_PREFIX + remote_name.encode() + b":"
        found = set[str]()
        with self._env.begin() as txn:
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
            with self._env.begin(write=True) as txn:
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
        with self._env.begin(write=True) as txn:
            for hash_ in hashes:
                key = prefix + hash_.encode()
                txn.delete(key)

    def remote_index_clear(self, remote_name: str) -> None:
        """Clear all index entries for a remote (force re-indexing)."""
        self._check_closed()
        self._check_write_allowed()
        prefix = _REMOTE_PREFIX + remote_name.encode() + b":"
        with self._env.begin(write=True) as txn:
            cursor = txn.cursor()
            keys_to_delete = list[bytes]()
            if cursor.set_range(prefix):
                for key, _ in cursor:
                    if not key.startswith(prefix):
                        break
                    keys_to_delete.append(key)
            for key in keys_to_delete:
                txn.delete(key)

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
            with self._env.begin(write=True) as txn:
                txn.put(key, value)
        except lmdb.MapFullError as e:
            raise DatabaseFullError(_DB_FULL_MSG) from e

    def read_run(self, run_id: str) -> RunManifest | None:
        """Read a run manifest by ID. Returns None if not found."""
        self._check_closed()

        key = _RUN_PREFIX + run_id.encode()
        with self._env.begin() as txn:
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
        with self._env.begin() as txn:
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

        with self._env.begin(write=True) as txn:
            for run in to_delete:
                key = _RUN_PREFIX + run["run_id"].encode()
                txn.delete(key)

        valid_run_ids = {run["run_id"] for run in to_keep}
        self.prune_run_cache(valid_run_ids)

        return len(to_delete)

    # -------------------------------------------------------------------------
    # Run cache for skip detection (like DVC's run cache)
    # -------------------------------------------------------------------------

    def write_run_cache(self, stage_name: str, input_hash: str, entry: RunCacheEntry) -> None:
        """Write a run cache entry for skip detection."""
        self._check_closed()
        self._check_write_allowed()
        key = _RUNCACHE_PREFIX + f"{stage_name}:{input_hash}".encode()
        value = run_history.serialize_to_bytes(entry)
        try:
            with self._env.begin(write=True) as txn:
                txn.put(key, value)
        except lmdb.MapFullError as e:
            raise DatabaseFullError(_DB_FULL_MSG) from e

    def lookup_run_cache(self, stage_name: str, input_hash: str) -> RunCacheEntry | None:
        """Look up run cache entry for skip detection. Returns None if not found."""
        self._check_closed()
        key = _RUNCACHE_PREFIX + f"{stage_name}:{input_hash}".encode()
        with self._env.begin() as txn:
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
        with self._env.begin() as txn:
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

        with self._env.begin(write=True) as txn:
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
            for dep_path in deferred["dep_generations"]:
                key = _make_key_dep_generation(stage_name, dep_path)
                if len(key) > _MAX_KEY_SIZE:
                    raise PathTooLongError(
                        f"Dependency path too long ({len(key)} bytes, max {_MAX_KEY_SIZE}): {dep_path}"
                    )

        for path_str in output_paths:
            key = _make_key_output_generation(pathlib.Path(path_str))
            if len(key) > _MAX_KEY_SIZE:
                raise PathTooLongError(
                    f"Output path too long ({len(key)} bytes, max {_MAX_KEY_SIZE}): {path_str}"
                )

        try:
            with self._env.begin(write=True) as txn:
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
                    for dep_path, gen in deferred["dep_generations"].items():
                        key = _make_key_dep_generation(stage_name, dep_path)
                        txn.put(key, struct.pack(">Q", gen))

                # Output generations (increment)
                for path_str in output_paths:
                    path = pathlib.Path(path_str)
                    key = _make_key_output_generation(path)
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

        except lmdb.MapFullError as e:
            raise DatabaseFullError(_DB_FULL_MSG) from e

    def close(self) -> None:
        """Close the database."""
        if not self._closed:
            self._env.close()
            self._closed = True

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
        self._check_capacity_warning()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
