from __future__ import annotations

import fcntl
import logging
import os
import pathlib
import re
import stat
import struct
import time
from typing import TYPE_CHECKING, NotRequired, TypedDict, TypeGuard, cast

import yaml

from pivot import config, exceptions, project
from pivot.storage import cache

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
# listed; a nanosecond one loses the memo for about one directory in 10^9. A time the stamp's
# signed 64-bit nanoseconds cannot hold (before 1677 or after 2262) is never trusted, so such a
# directory is listed every time. The memo holds names only, never a .pvt's content (rewriting a
# file does not change its directory), and is trusted only on filesystems that keep directory
# times this way.
#
# The memo is the account's own file, pvt-memo-<uid> in the state database's directory
# (state.lmdb, which .pivot/.gitignore keeps out of git), created with mode 0600. A walk reads
# it, and the pre-flight writes it, only after an fstat of the open file shows that the running
# account owns it, that no other account may read or write it (no group or other mode bits; a
# POSIX ACL's mask shows in the group bits) and that it has one link; otherwise every directory
# is listed. So only the account that listed a directory reads its names, whatever path reaches
# the file (a symlinked .pivot or state.lmdb, a bind mount) and whatever modes the project's
# directories have, at the cost of the file's open and one fstat per walk. The memo's own path
# is never followed through a symlink, and a hard link fails the check, so the write cannot land
# in another of the account's files. The file is rewritten in place, so the directory holding
# it does not change with each write. A listing in which an entry's type could not be read (the
# lstat a filesystem without d_type needs failed) is never recorded: that entry is not
# descended, as in os.walk, and a walk that can read its type must not reuse that.
_MEMO_SETTLE_NS = 2_000_000_000
_NS_PER_S = 1_000_000_000
_MEMO_FS_TYPES = frozenset({b"ext2", b"ext3", b"ext4", b"xfs", b"btrfs", b"tmpfs", b"f2fs", b"zfs"})
_MEMO_ERRORS = (exceptions.PivotError, OSError, ValueError)  # ValueError: a NUL in core.state_dir
_MEMO_RECORD = struct.Struct(">II")  # a memo file record's key and value sizes
_MEMO_COUNT = struct.Struct(">I")  # the root record's value: how many listings follow it
_MOUNTINFO = pathlib.Path("/proc/self/mountinfo")
_MOUNTINFO_ESCAPE = re.compile(rb"\\([0-7]{3})")
_LISTING_VERSION = b"\x03"  # an earlier version is never reused; version 1 may omit a subdirectory
_STAMP = struct.Struct(">QQqq")  # st_dev, st_ino, st_mtime_ns, st_ctime_ns
_STAMP_TIMES = range(-(2**63), 2**63)  # the times _STAMP's signed 64-bit fields hold
_SUBDIR_COUNT = struct.Struct(">I")
_LISTING_HEADER = len(_LISTING_VERSION) + _STAMP.size + _SUBDIR_COUNT.size


def _times_trusted(dir_stat: os.stat_result, settled_before: int) -> bool:
    """Whether a directory's mtime and ctime let its stamp vouch for a recorded listing.

    Both are settled (before ``settled_before``, the walk start less ``_MEMO_SETTLE_NS``), not
    both whole seconds, and within ``_STAMP_TIMES``, as the comment above argues.
    """
    mtime_ns, ctime_ns = dir_stat.st_mtime_ns, dir_stat.st_ctime_ns
    return (
        mtime_ns < settled_before
        and ctime_ns < settled_before
        and (mtime_ns % _NS_PER_S != 0 or ctime_ns % _NS_PER_S != 0)
        and mtime_ns in _STAMP_TIMES
        and ctime_ns in _STAMP_TIMES
    )


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


def _list_directory(path: str) -> tuple[list[str], list[str], bool] | None:
    """List a directory the way ``os.walk(followlinks=False)`` descends it.

    Returns its real subdirectories (not symlinks) and every name ending in ``.pvt``, both in
    listing order, and whether every entry's type could be read (an entry whose type cannot be
    read is not a subdirectory, as in ``os.walk``). None when it cannot be listed (``os.walk``
    skips it).
    """
    subdirs = list[str]()
    pvt_names = list[str]()
    typed = True
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
                    typed = False
                if is_dir:
                    subdirs.append(name)
    except OSError:
        return None
    return subdirs, pvt_names, typed


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

    Visits the directories ``os.walk(root, followlinks=False)`` visits, with one stat each. A
    directory is listed unless ``memo`` holds a listing under its current stamp, its filesystem is
    on the allowlist, and ``_times_trusted`` accepts its times. With ``record``, the second result
    holds the listing of every such directory whose entries' types could all be read.
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
        memoizable = allowed_devices[device] and _times_trusted(dir_stat, settled_before)
        stamp = (
            _STAMP.pack(device, dir_stat.st_ino, dir_stat.st_mtime_ns, dir_stat.st_ctime_ns)
            if memoizable
            else b""
        )
        recorded = memo.get(rel) if memoizable else None
        names = None
        if recorded is not None and recorded.startswith(_LISTING_VERSION + stamp):
            names = _unpack_listing(recorded)
        if names is None:
            listed = _list_directory(path)
            if listed is None:
                continue
            subdirs, pvt_names, typed = listed
            names = subdirs, pvt_names
            recorded = _pack_listing(stamp, *names) if memoizable and typed and record else None
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


class _Memo(TypedDict):
    """The walk's directory memo, as the account's memo file holds it."""

    root: str  # the project root the listings were taken under
    dirs: dict[str, bytes]  # project-relative directory path ("" is the root) -> packed listing


def _memo_path() -> pathlib.Path:
    """The running account's memo file, beside the state database's files."""
    return config.get_state_dir() / "state.lmdb" / f"pvt-memo-{os.geteuid()}"


def _memo_refusal(memo_stat: os.stat_result) -> str | None:
    """Why an open memo file must not be used, or None when only the running account may use it.

    It must belong to the running account, give group and others no access (a POSIX ACL's mask
    shows in the group bits) and have one link.
    """
    if memo_stat.st_uid != os.geteuid():
        return f"belongs to uid {memo_stat.st_uid}"
    if memo_stat.st_mode & 0o077:
        return f"has mode {stat.filemode(memo_stat.st_mode)}"
    if memo_stat.st_nlink != 1:
        return f"has {memo_stat.st_nlink} links"
    return None


def _pack_memo(root: str, dirs: dict[str, bytes]) -> bytes:
    """A memo file's content: a record whose key is the root and whose value is the number of
    listings, then one record per directory listing."""
    records = [
        (os.fsencode(root), _MEMO_COUNT.pack(len(dirs))),
        *((os.fsencode(rel), item) for rel, item in dirs.items()),
    ]
    return b"".join(_MEMO_RECORD.pack(len(key), len(value)) + key + value for key, value in records)


def _unpack_memo(content: bytes) -> _Memo | None:
    """The memo ``_pack_memo`` made, or None when ``content`` is not one whole memo.

    A memo cut short anywhere (a pre-flight killed mid-write, or a walk reading during a write)
    is not whole: it ends inside a record, or holds fewer listings than its root record counts.
    """
    records = list[tuple[bytes, bytes]]()
    offset = 0
    while offset + _MEMO_RECORD.size <= len(content):
        key_size, value_size = _MEMO_RECORD.unpack_from(content, offset)
        key_at = offset + _MEMO_RECORD.size
        value_at = key_at + key_size
        offset = value_at + value_size
        records.append((content[key_at:value_at], content[value_at:offset]))
    if offset != len(content) or not records:
        return None
    (root, count), *listings = records
    if len(count) != _MEMO_COUNT.size or _MEMO_COUNT.unpack(count)[0] != len(listings):
        return None
    return _Memo(root=os.fsdecode(root), dirs={os.fsdecode(rel): item for rel, item in listings})


def _open_memo(flags: int) -> tuple[int | None, _Memo | None]:
    """Open the account's memo file with ``flags`` and read it: its descriptor and its memo.

    The descriptor is None (and the caller lists every directory) when the file is missing or
    cannot be opened, or when an fstat of the open file fails ``_memo_refusal``; a debug log names
    the reason, except for a missing file. The memo's own path is never followed through a
    symlink, and a FIFO there is not waited on. The memo is None when the file holds no whole
    memo. Any failure means no memo: every directory is listed, as main does.
    """
    try:
        path = _memo_path()
        fd = os.open(path, flags | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, 0o600)
    except FileNotFoundError:
        return None, None
    except _MEMO_ERRORS as e:
        logger.debug(f"Listing every directory: cannot open the .pvt walk memo: {e}")
        return None, None
    try:
        memo_stat = os.fstat(fd)
        refusal = _memo_refusal(memo_stat)
        content = os.pread(fd, memo_stat.st_size, 0) if refusal is None else b""
    except OSError as e:
        refusal = f"cannot be read: {e}"
        content = b""
    if refusal is not None:
        os.close(fd)
        logger.debug(f"Listing every directory: the .pvt walk memo {path} {refusal}")
        return None, None
    memo = _unpack_memo(content)
    if memo is None and content:
        logger.debug(f"Listing every directory: the .pvt walk memo {path} is malformed")
    return fd, memo


def _write_memo(fd: int, content: bytes) -> None:
    """Write ``content`` over the open memo file, unless another pre-flight is writing it.

    The file is emptied first, so a walk reading it meanwhile, or after this process is killed
    mid-write, sees a prefix of the new memo, which does not parse: that walk lists every
    directory. A failure is logged and skipped.
    """
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.ftruncate(fd, 0)
        with open(fd, "wb", closefd=False) as memo_file:
            memo_file.write(content)
    except OSError as e:
        logger.debug(f"Could not record the .pvt walk memo: {e}")


def _trusted_listings(memo: _Memo | None, root: pathlib.Path) -> dict[str, bytes]:
    return memo["dirs"] if memo is not None and memo["root"] == _memo_root(root) else {}


def discover_pvt_files(root: pathlib.Path) -> dict[str, PvtData]:
    """Find all .pvt files under root, return {data_path: PvtData}.

    Finds what ``os.walk`` over the whole tree finds. Directories the memo vouches for are stat'ed
    instead of listed; every .pvt found is read. Never writes or creates the memo: only the
    repro/run pre-flight does (``discover_pvt_files_and_update_memo``).
    """
    fd, memo = _open_memo(os.O_RDONLY)
    if fd is not None:
        os.close(fd)
    candidates, _ = _walk(root, _trusted_listings(memo, root), record=False)
    return _read_pvt_files(candidates)


def discover_pvt_files_and_update_memo(root: pathlib.Path) -> dict[str, PvtData]:
    """``discover_pvt_files`` for the repro/run pre-flight, the memo's one writer.

    Opens the account's memo file read-write, creating it with mode 0600 when it is missing, and
    when the walk's listings differ from it, writes in place the listing of every directory
    ``_walk`` may record, dropping every other listing. It never creates the state database's
    directory, so before the first pivot write there it records nothing. A memo that cannot be
    opened or fails ``_memo_refusal`` is neither read nor written for this walk.
    """
    real_root = _memo_root(root)
    fd, previous = _open_memo(os.O_RDWR | os.O_CREAT)
    try:
        candidates, listings = _walk(root, _trusted_listings(previous, root), record=fd is not None)
        if fd is not None and previous != _Memo(root=real_root, dirs=listings):
            _write_memo(fd, _pack_memo(real_root, listings))
    finally:
        if fd is not None:
            os.close(fd)
    return _read_pvt_files(candidates)


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
