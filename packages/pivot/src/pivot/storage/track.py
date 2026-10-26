from __future__ import annotations

import logging
import os
import pathlib
import re
import stat
import struct
import time
from typing import TYPE_CHECKING, NotRequired, TypedDict, TypeGuard, cast

import lmdb
import yaml

from pivot import config, exceptions, project
from pivot.storage import cache, state

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pivot.types import DirManifestEntry, HashInfo

# Use union types to avoid type: ignore on fallback assignment
_Loader: type[yaml.SafeLoader] | type[yaml.CSafeLoader]
_Dumper: type[yaml.SafeDumper] | type[yaml.CSafeDumper]

try:
    _Loader = yaml.CSafeLoader
    _Dumper = yaml.CSafeDumper
except AttributeError:
    _Loader = yaml.SafeLoader
    _Dumper = yaml.SafeDumper


class ImportSource(TypedDict):
    """Source information for imported artifacts."""

    repo: str  # Source repo URL
    rev: str  # Symbolic ref (branch/tag)
    rev_lock: str  # Resolved commit SHA
    stage: str  # Source stage name (auto-discovered)
    path: str  # Path within source repo
    remote: str  # Source's remote URL (s3:// or gs://)


class PvtData(TypedDict):
    """Data stored in .pvt files."""

    path: str  # Relative path to tracked file/directory
    hash: str  # Content hash
    size: int  # Total size (file or sum of directory)
    num_files: NotRequired[int]  # For directories only
    manifest: NotRequired[list[DirManifestEntry]]  # For directories only
    empty_dirs: NotRequired[list[str]]  # For directories holding any (see DirHash)
    source: NotRequired[ImportSource]  # Import source metadata


_REQUIRED_KEYS = frozenset({"path", "hash", "size"})
_VALID_KEYS = frozenset({"path", "hash", "size", "num_files", "manifest", "empty_dirs", "source"})
_REQUIRED_SOURCE_KEYS = frozenset({"repo", "rev", "rev_lock", "stage", "path", "remote"})


def is_pvt_data(data: object) -> TypeGuard[PvtData]:
    """Validate that parsed YAML has valid PvtData structure."""
    if not isinstance(data, dict):
        return False
    str_data = cast("dict[str, object]", data)
    if not _REQUIRED_KEYS.issubset(str_data.keys()):
        return False
    if not all(key in _VALID_KEYS for key in str_data):
        return False
    if "empty_dirs" in str_data and not _is_empty_dirs(str_data):
        return False
    if "source" in str_data:
        source = str_data["source"]
        if not isinstance(source, dict):
            return False
        source_dict = cast("dict[str, object]", source)
        if not _REQUIRED_SOURCE_KEYS.issubset(source_dict.keys()):
            return False
        if not all(isinstance(source_dict[k], str) for k in _REQUIRED_SOURCE_KEYS):
            return False
    return True


def _is_empty_dirs(data: dict[str, object]) -> bool:
    """Whether a .pvt's empty_dirs is a list of safe relpaths beside a manifest."""
    value = data["empty_dirs"]
    return (
        "manifest" in data
        and isinstance(value, list)
        and all(
            isinstance(relpath, str) and relpath != "" and not has_path_traversal(relpath)
            for relpath in cast("list[object]", value)
        )
    )


def has_path_traversal(path: str) -> bool:
    """Check if path is unsafe: absolute or contains traversal (..)."""
    p = pathlib.Path(path)
    return p.is_absolute() or ".." in p.parts


def _validate_path(path: str) -> None:
    """Validate path is relative and doesn't contain traversal."""
    if has_path_traversal(path):
        raise exceptions.SecurityValidationError(f"Unsafe path (absolute or traversal): {path!r}")


def write_pvt_file(pvt_path: pathlib.Path, data: PvtData) -> None:
    """Write .pvt manifest file atomically with path validation."""
    _validate_path(data["path"])

    def write_yaml(fd: int) -> None:
        with os.fdopen(fd, "w") as f:
            yaml.dump(dict(data), f, Dumper=_Dumper, sort_keys=False)

    cache.atomic_write_file(pvt_path, write_yaml)


def read_pvt_file(pvt_path: pathlib.Path) -> PvtData | None:
    """Read .pvt manifest file; returns None if missing, invalid, or insecure."""
    try:
        with open(pvt_path) as f:
            data: object = yaml.load(f, Loader=_Loader)
        if not is_pvt_data(data):
            return None
        if has_path_traversal(data["path"]):
            return None
        return data
    except (FileNotFoundError, PermissionError, UnicodeDecodeError, yaml.YAMLError):
        return None


def get_pvt_path(data_path: pathlib.Path) -> pathlib.Path:
    """Convert data path to its .pvt manifest path (e.g., file.csv -> file.csv.pvt)."""
    # pathlib normalizes trailing slashes, so "images/" becomes "images"
    return data_path.with_suffix(data_path.suffix + ".pvt")


def get_data_path(pvt_path: pathlib.Path) -> pathlib.Path:
    """Convert .pvt manifest path back to data path (e.g., file.csv.pvt -> file.csv)."""
    if not pvt_path.suffix == ".pvt":
        raise ValueError(f"Expected .pvt file, got: {pvt_path}")
    # Remove .pvt suffix
    return pvt_path.with_suffix("")


# The walk's directory memo. Every directory gets one stat per walk; a directory whose stamp
# (st_dev, st_ino, st_mtime_ns, st_ctime_ns) matches its recorded listing is not listed again.
# The stamp is taken before the listing, so a change made while listing leaves a stamp that no
# longer matches. Adding, removing or renaming an entry sets its directory's mtime and ctime to
# the current time, so the stamp changes, unless the change falls in the same timestamp tick as
# the change the listing saw. A directory that changed less than _MEMO_SETTLE_NS before the walk is
# therefore listed and never recorded: any later change carries a later timestamp. A stamp whose
# mtime and ctime are both whole seconds is never trusted either: on a filesystem with one-second
# timestamps (ext2, ext3, ext4 with 128-byte inodes), a wall clock stepped back into a recorded
# second would let a later change keep its directory's stamp. Such a filesystem is then always
# listed; a nanosecond one loses the memo for about one directory in 10^9. The memo holds names
# only, never a .pvt's content (rewriting a file does not change its directory), and is trusted
# only on filesystems that keep directory times this way. A reused listing is what the user who
# recorded it could list: a reader who cannot list that directory still gets its recorded names.
_MEMO_SETTLE_NS = 2_000_000_000
_NS_PER_S = 1_000_000_000
_MEMO_FS_TYPES = frozenset({b"ext2", b"ext3", b"ext4", b"xfs", b"btrfs", b"tmpfs", b"f2fs", b"zfs"})
_MEMO_WRITE_TIMEOUT_S = 1.0
_MEMO_DB_ERRORS = (exceptions.PivotError, state.StateDBError, RuntimeError, OSError, lmdb.Error)
_MOUNTINFO = pathlib.Path("/proc/self/mountinfo")
_MOUNTINFO_ESCAPE = re.compile(rb"\\([0-7]{3})")
_LISTING_VERSION = b"\x01"
_STAMP = struct.Struct(">QQqq")  # st_dev, st_ino, st_mtime_ns, st_ctime_ns
_SUBDIR_COUNT = struct.Struct(">I")
_LISTING_HEADER = len(_LISTING_VERSION) + _STAMP.size + _SUBDIR_COUNT.size


class _Mounts(TypedDict):
    """What ``/proc/self/mountinfo`` says about the memo's allowlist."""

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
        (allowed if fstype in _MEMO_FS_TYPES else refused).add(device)
        point = _MOUNTINFO_ESCAPE.sub(lambda match: bytes([int(match[1], 8)]), fields[4])
        points.append((point, fstype))
    points.sort(key=lambda mount: len(mount[0]), reverse=True)
    return _Mounts(allowed=frozenset(allowed - refused), refused=frozenset(refused), points=points)


def _device_allowed(mounts: _Mounts, device: int, real_path: str) -> bool:
    """Whether directories on ``device`` keep their times the way the memo needs.

    NFS, overlayfs, FUSE and any other type are refused. A device the file does not list is a
    btrfs subvolume, which reports a device of its own: it is allowed when the deepest mount
    holding ``real_path`` is btrfs, and refused otherwise.
    """
    if device in mounts["allowed"]:
        return True
    if device in mounts["refused"]:
        return False
    path = os.fsencode(real_path)
    for point, fstype in mounts["points"]:
        if path == point or path.startswith(point.rstrip(b"/") + b"/"):
            return fstype == b"btrfs"
    return False


def _list_directory(path: str) -> tuple[list[str], list[str]] | None:
    """List a directory the way ``os.walk(followlinks=False)`` descends it.

    Returns its real subdirectories (not symlinks) and every name ending in ``.pvt``, both in
    listing order, or None when it cannot be listed (``os.walk`` skips it).
    """
    subdirs = list[str]()
    pvt_names = list[str]()
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                name = entry.name
                if name.endswith(".pvt"):
                    pvt_names.append(name)
                try:
                    is_dir = entry.is_dir(follow_symlinks=False)
                except OSError:
                    is_dir = False
                if is_dir:
                    subdirs.append(name)
    except OSError:
        return None
    return subdirs, pvt_names


def _pack_listing(stamp: bytes, subdirs: list[str], pvt_names: list[str]) -> bytes:
    names = b"\0".join(os.fsencode(name) for name in (*subdirs, *pvt_names))
    return _LISTING_VERSION + stamp + _SUBDIR_COUNT.pack(len(subdirs)) + names


def _unpack_listing(listing: bytes) -> tuple[list[str], list[str]] | None:
    """The names a listing holds, or None when it is malformed (the directory is then listed)."""
    if len(listing) < _LISTING_HEADER:
        return None
    (subdir_count,) = _SUBDIR_COUNT.unpack_from(listing, _LISTING_HEADER - _SUBDIR_COUNT.size)
    blob = listing[_LISTING_HEADER:]
    names = blob.split(b"\0") if blob else []
    if subdir_count > len(names) or b"/" in blob or b"" in names or b"." in names or b".." in names:
        return None
    decoded = [os.fsdecode(name) for name in names]
    return decoded[:subdir_count], decoded[subdir_count:]


def _walk(
    root: pathlib.Path, memo: dict[str, bytes], *, record: bool
) -> tuple[list[pathlib.Path], dict[str, bytes]]:
    """Return every .pvt candidate under root, in ``os.walk`` order, and the listings to record.

    Visits the directories ``os.walk(root, followlinks=False)`` visits for the user who recorded
    the memo, with one stat each. A directory is listed unless ``memo`` holds a listing under its
    current stamp, its filesystem is on the allowlist, its mtime and ctime are settled, and they
    are not both whole seconds. With ``record``, the second result holds the listing of every
    directory that may be recorded.
    """
    mounts = _read_mounts()
    allowed_devices = dict[int, bool]()
    real_root = os.path.realpath(root)
    settled_before = time.time_ns() - _MEMO_SETTLE_NS
    candidates = list[pathlib.Path]()
    listings = dict[str, bytes]()
    try:
        root_stat = os.stat(root)  # os.walk lists the root through a symlink
    except OSError:
        return candidates, listings
    if not stat.S_ISDIR(root_stat.st_mode):
        return candidates, listings

    stack = [(os.fspath(root), "", root_stat)]
    while stack:
        path, rel, dir_stat = stack.pop()
        device = dir_stat.st_dev
        if device not in allowed_devices:
            real_path = os.path.join(real_root, rel) if rel else real_root
            allowed_devices[device] = _device_allowed(mounts, device, real_path)
        memoizable = (
            allowed_devices[device]
            and dir_stat.st_mtime_ns < settled_before
            and dir_stat.st_ctime_ns < settled_before
            and (dir_stat.st_mtime_ns % _NS_PER_S != 0 or dir_stat.st_ctime_ns % _NS_PER_S != 0)
        )
        stamp = _STAMP.pack(device, dir_stat.st_ino, dir_stat.st_mtime_ns, dir_stat.st_ctime_ns)
        recorded = memo.get(rel) if memoizable else None
        names = None
        if recorded is not None and recorded.startswith(_LISTING_VERSION + stamp):
            names = _unpack_listing(recorded)
        if names is None:
            names = _list_directory(path)
            if names is None:
                continue
            recorded = _pack_listing(stamp, *names) if memoizable and record else None
        if record and recorded is not None:
            listings[rel] = recorded

        subdirs, pvt_names = names
        candidates.extend(pathlib.Path(path, name) for name in pvt_names)
        for name in reversed(subdirs):
            child = os.path.join(path, name)
            try:
                child_stat = os.lstat(child)
            except OSError:
                continue
            if stat.S_ISDIR(child_stat.st_mode):  # a symlink or file now: not descended
                stack.append((child, os.path.join(rel, name), child_stat))
    return candidates, listings


def _read_pvt_files(candidates: list[pathlib.Path]) -> dict[str, PvtData]:
    result = dict[str, PvtData]()
    for pvt_path in candidates:
        if not pvt_path.is_file():
            continue
        data = read_pvt_file(pvt_path)
        if data is None:
            logger.warning(f"Skipping invalid .pvt file: {pvt_path}")
            continue

        data_path = pvt_path.parent / data["path"]
        normalized = project.normalize_path(data_path)
        result[str(normalized)] = data
    return result


def _memo_root(root: pathlib.Path) -> str:
    """The root a memo is recorded under: a memo from another path (a moved project) is unused."""
    return os.fspath(root.resolve())


def _read_stored_memo() -> state.PvtDirMemo | None:
    """The stored memo, or None when there is no state database or it cannot be read.

    Never creates the database. Any failure to read it (a config error, a read-only or
    unopenable database) means no memo: every directory is listed, as main does.
    """
    try:
        state_dir = config.get_state_dir()
        if not (state_dir / "state.lmdb" / "data.mdb").is_file():
            return None
        with state.StateDB(state_dir, readonly=True) as state_db:
            return state_db.read_pvt_dir_memo()
    except _MEMO_DB_ERRORS as e:
        logger.debug(f"Listing every directory: cannot read the .pvt walk memo: {e}")
        return None


def _trusted_listings(memo: state.PvtDirMemo | None, root: pathlib.Path) -> dict[str, bytes]:
    return memo["dirs"] if memo is not None and memo["root"] == _memo_root(root) else {}


def discover_pvt_files(root: pathlib.Path) -> dict[str, PvtData]:
    """Find all .pvt files under root, return {data_path: PvtData}.

    Finds what ``os.walk`` over the whole tree finds. Directories the memo vouches for are stat'ed
    instead of listed; every .pvt found is read. Never writes the memo: only the repro/run
    pre-flight does (``discover_pvt_files_and_update_memo``).
    """
    candidates, _ = _walk(root, _trusted_listings(_read_stored_memo(), root), record=False)
    return _read_pvt_files(candidates)


def discover_pvt_files_and_update_memo(root: pathlib.Path) -> dict[str, PvtData]:
    """``discover_pvt_files`` for the repro/run pre-flight, the memo's one writer.

    Records the listing of every settled directory on an allowlisted filesystem, and drops the
    listings of directories the walk no longer visits. With no tracked file found, it records
    only into a state database that already exists, so it never creates one. A memo that cannot
    be recorded (another pivot writer holds the lock longer than ``_MEMO_WRITE_TIMEOUT_S``, or
    the database is read-only, full or failing) is skipped. A writer that bypasses pivot's lock
    can hold the memo write until it commits.
    """
    previous = _read_stored_memo()
    candidates, listings = _walk(root, _trusted_listings(previous, root), record=True)
    found = _read_pvt_files(candidates)
    if not found and previous is None:
        return found
    try:
        with state.StateDB(config.get_state_dir()) as state_db:
            state_db.update_pvt_dir_memo(
                previous if previous is not None else state.PvtDirMemo(root=None, dirs={}),
                _memo_root(root),
                listings,
                timeout=_MEMO_WRITE_TIMEOUT_S,
            )
    except _MEMO_DB_ERRORS as e:
        logger.debug(f"Could not record the .pvt walk memo: {e}")
    return found


def is_import(data: PvtData) -> bool:
    """Check if PvtData represents an imported artifact."""
    return "source" in data


def discover_import_pvt_files(root: pathlib.Path) -> dict[str, PvtData]:
    """Find all import .pvt files under root."""
    return {path: data for path, data in discover_pvt_files(root).items() if is_import(data)}


def pvt_to_hash_info(pvt_data: PvtData) -> HashInfo:
    """Convert PvtData to HashInfo format for cache operations."""
    from pivot.types import DirHash, FileHash

    if "manifest" not in pvt_data:
        return FileHash(hash=pvt_data["hash"])
    dir_hash = DirHash(hash=pvt_data["hash"], manifest=pvt_data["manifest"])
    if "empty_dirs" in pvt_data and pvt_data["empty_dirs"]:
        dir_hash["empty_dirs"] = pvt_data["empty_dirs"]
    return dir_hash
