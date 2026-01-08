from __future__ import annotations

import os
import pathlib
import struct
from typing import Self

import lmdb

# Key prefixes for different entry types
_HASH_PREFIX = b"hash:"  # File hash entries
_GEN_PREFIX = b"gen:"  # Output generation counters
_DEP_PREFIX = b"dep:"  # Stage dependency generations

# Default LMDB map size (10GB virtual - grows as needed)
_MAP_SIZE = 10 * 1024 * 1024 * 1024

# LMDB default max key size
_MAX_KEY_SIZE = 511


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
    """LMDB cache of (path, mtime_ns, size, inode) -> hash to skip rehashing unchanged files."""

    _env: lmdb.Environment
    _closed: bool

    def __init__(self, db_path: pathlib.Path) -> None:
        lmdb_path = db_path.parent / "state.lmdb"
        lmdb_path.parent.mkdir(parents=True, exist_ok=True)
        self._env = lmdb.open(str(lmdb_path), map_size=_MAP_SIZE)
        self._closed = False

    def _check_closed(self) -> None:
        """Raise if database is closed."""
        if self._closed:
            raise RuntimeError("Cannot operate on closed StateDB")

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
            raise DatabaseFullError(
                f"State cache is full ({_MAP_SIZE // (1024**3)}GB limit). Delete .pivot/state.lmdb/ to reset."
            ) from e

    def save_many(self, entries: list[tuple[pathlib.Path, os.stat_result, str]]) -> None:
        """Batch save multiple entries atomically."""
        self._check_closed()
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
            raise DatabaseFullError(
                f"State cache is full ({_MAP_SIZE // (1024**3)}GB limit). Delete .pivot/state.lmdb/ to reset."
            ) from e

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
            raise DatabaseFullError(
                f"State cache is full ({_MAP_SIZE // (1024**3)}GB limit). Delete .pivot/state.lmdb/ to reset."
            ) from e
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
            raise DatabaseFullError(
                f"State cache is full ({_MAP_SIZE // (1024**3)}GB limit). Delete .pivot/state.lmdb/ to reset."
            ) from e

    def close(self) -> None:
        """Close the database."""
        if not self._closed:
            self._env.close()
            self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
