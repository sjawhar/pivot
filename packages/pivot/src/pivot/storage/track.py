from __future__ import annotations

import fcntl
import functools
import logging
import os
import pathlib
import shlex
import stat
import struct
import time
from typing import TYPE_CHECKING, NamedTuple, NotRequired, TypedDict, TypeGuard, cast

import yaml

from pivot import config, exceptions, project
from pivot.storage import cache, listing, mounts

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from pivot.types import HashInfo

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
    """Data stored in .pvt files. A directory's members are in its listing file,
    ``<path>.pvt.listing`` (``listing.tracked_listing_path``)."""

    path: str  # Relative path to tracked file/directory
    hash: str  # Content hash; a directory's is its listing's
    size: int  # Total size (file or sum of directory)
    nfiles: NotRequired[int]  # For directories only
    source: NotRequired[ImportSource]  # Import source metadata


# Whether discovery leaves out a .pvt that parses (its path, its data): ``discover_pvt_files``.
type PvtSkip = Callable[[pathlib.Path, PvtData], bool]


_REQUIRED_KEYS = frozenset({"path", "hash", "size"})
_VALID_KEYS = frozenset({"path", "hash", "size", "nfiles", "source"})
# Keys an older pivot wrote for a directory, with its files listed in the .pvt itself.
_OLD_FORMAT_KEYS = frozenset({"num_files", "manifest", "empty_dirs"})
_REQUIRED_SOURCE_KEYS = frozenset({"repo", "rev", "rev_lock", "stage", "path", "remote"})
# A manifest entry's fields every restore reads, and those it builds a path or cache key from.
# `size` is not required: only a symlink restore through a tree no pivot has checked reads it,
# and it then checks the content by hash alone (`cache._tree_file_holds`).
_MANIFEST_ENTRY_READ_KEYS = ("relpath", "hash", "isexec")
_MANIFEST_ENTRY_STR_KEYS = ("relpath", "hash")


def is_pvt_data(data: object) -> TypeGuard[PvtData]:
    """Validate that parsed YAML has valid PvtData structure."""
    if not isinstance(data, dict):
        return False
    str_data = cast("dict[str, object]", data)
    if not _REQUIRED_KEYS.issubset(str_data.keys()):
        return False
    if not all(key in _VALID_KEYS for key in str_data):
        return False
    if not isinstance(str_data["path"], str) or not isinstance(str_data["hash"], str):
        return False
    if not all(_is_count(str_data[key]) for key in ("size", "nfiles") if key in str_data):
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


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def pvt_field_error(data: Mapping[str, object]) -> str | None:
    """Why a restore cannot use a document shaped like a .pvt, naming the field, or None.

    A .pvt names the fields it has, not what they hold. A restore checks ``path`` for
    traversal, builds a cache key from ``hash``, and, for a directory .pvt an older pivot wrote
    with its files inside it, builds a path from each manifest entry's ``relpath`` and each
    ``empty_dirs`` entry and a cache key from each entry's ``hash``, and reads each entry's
    ``isexec``. So a hand-written .pvt whose all-digit hash YAML reads as an integer, whose
    manifest entry has no hash, or whose relpath or empty directory holds a NUL byte would stop
    the restore with a raw TypeError, KeyError or ValueError. Only those fields are named here:
    a restore never builds a path from ``path`` (``get`` names the file by its target), and
    what the format itself refuses (``is_pvt_data``, ``listing.from_inline``) is not a field
    error but a .pvt pivot does not read.
    """
    for key in ("path", "hash"):
        if not isinstance(data[key], str):
            return f"field {key!r} is not a string"
    empty_dirs = data.get("empty_dirs", [])
    if isinstance(empty_dirs, list):
        for index, relpath in enumerate(cast("list[object]", empty_dirs)):
            if isinstance(relpath, str) and "\0" in relpath:
                return f"field 'empty_dirs[{index}]' holds a NUL byte"
    if "manifest" not in data:
        return None
    manifest = data["manifest"]
    if not isinstance(manifest, list):
        return "field 'manifest' is not a list"
    for index, raw in enumerate(cast("list[object]", manifest)):
        if not isinstance(raw, dict):
            return f"field 'manifest[{index}]' is not a mapping"
        entry = cast("dict[object, object]", raw)
        for key in _MANIFEST_ENTRY_READ_KEYS:
            if key not in entry:
                return f"field 'manifest[{index}].{key}' is missing"
        for key in _MANIFEST_ENTRY_STR_KEYS:
            if not isinstance(entry[key], str):
                return f"field 'manifest[{index}].{key}' is not a string"
        if "\0" in cast("str", entry["relpath"]):
            return f"field 'manifest[{index}].relpath' holds a NUL byte"
    return None


@functools.cache
def _warn_old_format(pvt_path: pathlib.Path) -> None:
    """Warn once per process for each directory .pvt an older pivot wrote, naming the command
    that converts it as typed from the working directory."""
    logger.warning(
        "%s was written by an older pivot, with its files listed inside it; pivot reads that "
        + "list as the directory's listing. Run `pivot track --force %s` to convert it to a "
        + ".pvt and a committed .pvt.listing file",
        os.path.relpath(pvt_path),
        shlex.quote(os.path.relpath(get_data_path(pvt_path))),
    )


@functools.cache
def _warn_stale_listing(pvt_path: pathlib.Path) -> None:
    """Warn once per process for each directory .pvt an older pivot wrote over this pivot's,
    leaving this pivot's ``.pvt.listing`` beside it, naming that file and the command that
    removes it."""
    stale = os.path.relpath(listing.tracked_listing_path(pvt_path))
    logger.warning(
        "%s was written by an older pivot, with its files listed inside it, and pivot reads "
        + "that list; %s beside it is a listing this pivot wrote before, which no .pvt "
        + "records now. Remove it with `git rm %s`",
        os.path.relpath(pvt_path),
        stale,
        shlex.quote(stale),
    )


def _sizeless_directory_error(document: dict[str, object]) -> str | None:
    """The reason an older pivot's directory ``.pvt`` whose inline members lack a ``size`` is
    refused — this pivot records a directory by its listing, whose content-addressed bytes carry
    each member's size — with the remedy; None when ``from_inline`` rejected the ``.pvt`` for
    another reason, which stays a plain invalid ``.pvt``."""
    manifest = document.get("manifest")
    if not isinstance(manifest, list):
        return None
    sizeless = False
    for raw in cast("list[object]", manifest):
        if not isinstance(raw, dict):
            return None
        entry = cast("dict[str, object]", raw)
        relpath, file_hash, size = entry.get("relpath"), entry.get("hash"), entry.get("size")
        if not (isinstance(relpath, str) and isinstance(file_hash, str) and len(file_hash) == 16):
            return None
        if not (isinstance(size, int) and not isinstance(size, bool) and size >= 0):
            sizeless = True
    if not sizeless:
        return None
    path = document.get("path")
    shown = path if isinstance(path, str) else "<path>"
    return (
        "its directory members have no size, which this pivot records in each directory's "
        + f"listing; rewrite the .pvt with `pivot track --force {shown}`"
    )


def parse_pvt(data: object) -> tuple[PvtData, listing.DirListing | None] | str | None:
    """A parsed .pvt's data, why a restore cannot use it, or None if invalid or insecure.

    A directory .pvt an older pivot wrote lists its files inline; it is read as this format's
    record, and that list is returned as the directory's listing (None for any other .pvt).
    A document that has a .pvt's required fields but holds one a restore cannot use is that
    field's message (``pvt_field_error``), which a get reports for the target.
    """
    if isinstance(data, dict):
        document = cast("dict[str, object]", data)
        if document.keys() >= _REQUIRED_KEYS and (error := pvt_field_error(document)) is not None:
            return error
        if _OLD_FORMAT_KEYS & document.keys():
            members = listing.from_inline(document.get("manifest"), document.get("empty_dirs", []))
            if members is None:
                return _sizeless_directory_error(document)
            record = listing.record_of(members, listing.encode(members))
            converted: dict[str, object] = {
                "path": document.get("path"),
                "hash": record["hash"],
                "size": record["size"],
                "nfiles": record["nfiles"],
            }
            if "source" in document:
                converted["source"] = document["source"]
            if not is_pvt_data(converted) or has_path_traversal(converted["path"]):
                return None
            return converted, members
    candidate = cast("object", data)  # not narrowed by the dict check above
    if not is_pvt_data(candidate) or has_path_traversal(candidate["path"]):
        return None
    return candidate, None


def has_path_traversal(path: str) -> bool:
    """Check if path is unsafe: absolute or contains traversal (..)."""
    p = pathlib.Path(path)
    return p.is_absolute() or ".." in p.parts


def _validate_path(path: str) -> None:
    """Validate path is relative and doesn't contain traversal."""
    if has_path_traversal(path):
        raise exceptions.SecurityValidationError(f"Unsafe path (absolute or traversal): {path!r}")


def validate_pvt_path_name(path: str) -> None:
    """Refuse a tracked path name YAML cannot represent before its sidecars are written."""
    try:
        path.encode("utf-8")
    except UnicodeEncodeError as e:
        raise exceptions.InvalidPathError(
            f"Cannot track {path!r}: its name is not valid UTF-8; rename it"
        ) from e


def write_pvt_file(pvt_path: pathlib.Path, data: PvtData) -> None:
    """Write .pvt manifest file atomically with path validation.

    The only path a .pvt holds is the tracked path's own name; one YAML cannot hold (a name
    that is not UTF-8) is refused. A tracked directory's members are in its listing file.
    """
    _validate_path(data["path"])
    validate_pvt_path_name(data["path"])

    def write_yaml(fd: int) -> None:
        with os.fdopen(fd, "w") as f:
            yaml.dump(dict(data), f, Dumper=_Dumper, sort_keys=False)

    cache.atomic_write_file(pvt_path, write_yaml)


def _load_pvt(
    pvt_path: pathlib.Path,
) -> tuple[PvtData, listing.DirListing | None] | str | None:
    """A .pvt file's data and, for an older pivot's directory .pvt, its inline listing (with
    one warning per process saying how to convert it, or how to remove a stale ``.pvt.listing``
    beside it); None if missing, invalid, or insecure; the reason a restore cannot use it when
    its fields are read but refused (``parse_pvt``), for a caller to name.
    """
    try:
        with open(pvt_path) as f:
            data: object = yaml.load(f, Loader=_Loader)
    except (FileNotFoundError, PermissionError, UnicodeDecodeError, yaml.YAMLError):
        return None
    parsed = parse_pvt(data)
    if isinstance(parsed, str):
        return parsed
    if parsed is not None and parsed[1] is not None:
        if listing.tracked_listing_path(pvt_path).exists():
            _warn_stale_listing(pvt_path)
        else:
            _warn_old_format(pvt_path)
    return parsed


def read_pvt_file(pvt_path: pathlib.Path) -> PvtData | None:
    """Read .pvt manifest file; returns None if missing, invalid, or insecure."""
    parsed = _load_pvt(pvt_path)
    return parsed[0] if isinstance(parsed, tuple) else None


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


class RestoreTarget(NamedTuple):
    """The file a discovered .pvt restores, and the one line to tell its reader about it."""

    path: pathlib.Path | None  # None when the .pvt is skipped and drives no write
    warning: str | None


def restore_target(
    pvt_path: pathlib.Path, data_path_value: str, root: pathlib.Path
) -> RestoreTarget:
    """What the ``path:`` value of the .pvt at ``pvt_path`` names, under ``root``.

    A .pvt restores its file only inside the project, so one function answers that for every
    command that reads a sidecar:

    - the value is the .pvt's own name without ``.pvt``, the only form ``pivot track`` and
      ``pivot import`` write: the file beside the .pvt, with no filesystem call;
    - the value names no file (``''``, ``.``, ``./``): skipped, because the path it gives is the
      directory holding the .pvt, which a restore would replace with the file;
    - the file's directory really lies outside the project, through a symlink or because it
      cannot be resolved: skipped, because a .pvt restores its file only inside the project;
    - any other value: the file it names, with a warning that it is not the .pvt's own name, as
      the commands that take a target use the name instead.

    An absolute or ``..`` value never reaches here: ``read_pvt_file`` refuses that .pvt.
    """
    display = os.path.relpath(pvt_path, root)
    if not pathlib.PurePosixPath(data_path_value).parts:
        return RestoreTarget(
            None,
            f"Skipping {display}: its path {data_path_value!r} names no file; set path: to the "
            + "name of the file or directory beside the .pvt.",
        )
    own_name = pvt_path.name.removesuffix(".pvt")
    if data_path_value == own_name:
        return RestoreTarget(pvt_path.parent / own_name, None)

    data_path = pathlib.Path(os.path.normpath(pvt_path.parent / data_path_value))
    resolved_root = root.resolve()
    outside = project.real_location_outside(data_path.parent, resolved_root)
    if outside is not None:
        return RestoreTarget(
            None,
            f"Skipping {display}: its path {data_path_value!r} resolves to "
            + f"'{outside / data_path.name}', outside the project root '{resolved_root}'. "
            + "A .pvt restores its file only inside the project; move it beside its data and "
            + "set path: to the data's name.",
        )
    mismatch = (
        f"{display}: its path {data_path_value!r} is not the .pvt's own name {own_name!r}. "
        + "'pivot checkout' with no target, 'pivot pull' and 'pivot repro --checkout-missing' "
        + f"restore {data_path_value!r}; 'pivot checkout {display}', 'pivot get', "
        + f"'pivot update' and 'pivot push' use the file named {own_name!r} beside the .pvt. "
        + "Set path: to that name, the form 'pivot track' writes."
    )
    return RestoreTarget(data_path, mismatch)


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
# it, and the pre-flight writes it, only after an fstat of the open file shows a regular file on
# a filesystem of the allowlist above (where owner and mode are the file's own and the kernel
# enforces them; bindfs, CIFS and NFS can report the mount's instead), owned by the running
# account, with no group or other mode bits (a POSIX ACL's mask shows in the group bits) and one
# link. Otherwise every directory is listed. A file of another account, of another kind or on
# another filesystem is never written, replaced or removed. When the account's own file fails
# only the mode or link check (its owner loosened it), the pre-flight empties it, unlinks it and
# creates a new 0600 file, so its names leave that path, every descriptor opened on it and any
# other link to it. So no other account reads the memo's names, whatever path reaches the file
# (a symlinked .pivot or state.lmdb, a bind mount) and whatever modes the project's directories
# have, except while the owner leaves the file loose, until the next pre-flight replaces it. The
# cost is the file's open and one fstat per walk (one more when a loosened memo is replaced).
# The memo's own path is never followed through a symlink, and a hard link fails the check, so
# the write cannot land in another of the account's files. The file is rewritten in place, so the
# directory holding it does not change with each write. A listing in which an entry's type could
# not be read (the lstat a filesystem without d_type needs failed) is never recorded: that entry
# is not descended, as in os.walk, and a walk that can read its type must not reuse that.
_MEMO_SETTLE_NS = 2_000_000_000
_NS_PER_S = 1_000_000_000
_MEMO_ERRORS = (exceptions.PivotError, OSError, ValueError)  # ValueError: a NUL in core.state_dir
_MEMO_RECORD = struct.Struct(">II")  # a memo file record's key and value sizes
_MEMO_COUNT = struct.Struct(">I")  # the root record's value: how many listings follow it
_MEMO_OPEN = os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC  # no symlink followed, no FIFO waited on
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
    root: pathlib.Path, memo: dict[str, bytes], mount_table: mounts.Mounts, *, record: bool
) -> tuple[list[pathlib.Path], dict[str, bytes]]:
    """Return every .pvt candidate under root, in ``os.walk`` order, and the listings to record.

    Visits the directories ``os.walk(root, followlinks=False)`` visits, with one stat each. A
    directory is listed unless ``memo`` holds a listing under its current stamp, its filesystem is
    on the allowlist (``mounts``, read once per walk), and ``_times_trusted`` accepts its times.
    With ``record``, the second result holds the listing of every such directory whose entries'
    types could all be read.
    """
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
            allowed_devices[device] = mounts.device_allowed(mount_table, device, real_path)
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


def _read_pvt_files(
    candidates: list[pathlib.Path], root: pathlib.Path, skip: PvtSkip | None
) -> dict[str, PvtData]:
    result = dict[str, PvtData]()
    for pvt_path in candidates:
        if not pvt_path.is_file():
            continue
        parsed = _load_pvt(pvt_path)
        if isinstance(parsed, str):
            logger.warning(f"Skipping {pvt_path}: {parsed}")
            continue
        if parsed is None:
            logger.warning(f"Skipping invalid .pvt file: {pvt_path}")
            continue
        data = parsed[0]
        if skip is not None and skip(pvt_path, data):
            continue

        data_path, warning = restore_target(pvt_path, data["path"], root)
        if warning is not None:
            logger.warning(warning)
        if data_path is None:
            continue
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
    """The running account's memo file, beside the state database's files.

    Located, never created: the memo never creates the state directory (``_create_memo``), and a
    walk is a read, so it must not make a configured ``core.state_dir`` that does not exist yet.
    """
    return config.locate_state_dir() / "state.lmdb" / f"pvt-memo-{os.geteuid()}"


def _memo_refusal(
    memo_stat: os.stat_result, mount_table: mounts.Mounts, directory: pathlib.Path
) -> tuple[str, bool] | None:
    """Why an open memo file must not be used, and whether the pre-flight may replace it; None
    when no other account can reach it.

    It must be a regular file on a filesystem of the allowlist (where owner and mode are the
    file's own and the kernel enforces them), owned by the running account; any other file is
    refused and never replaced. The account's own file must also give group and others no access
    (a POSIX ACL's mask shows in the group bits) and have one link; one that does not was loosened
    by its owner, and the pre-flight replaces it.
    """
    if not stat.S_ISREG(memo_stat.st_mode):
        return f"is not a regular file ({stat.filemode(memo_stat.st_mode)})", False
    if not mounts.device_allowed(mount_table, memo_stat.st_dev, directory):
        return "is on a filesystem off the allowlist", False
    if memo_stat.st_uid != os.geteuid():
        return f"belongs to uid {memo_stat.st_uid}", False
    if memo_stat.st_mode & 0o077:
        return f"has mode {stat.filemode(memo_stat.st_mode)}", True
    if memo_stat.st_nlink != 1:
        return f"has {memo_stat.st_nlink} links", True
    return None


def _unlink_memo(path: pathlib.Path, memo_stat: os.stat_result) -> bool:
    """Unlink ``path`` if it still names the file ``memo_stat`` describes (OSError propagates)."""
    named = os.lstat(path)
    if (named.st_dev, named.st_ino) != (memo_stat.st_dev, memo_stat.st_ino):
        return False
    os.unlink(path)
    return True


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


def _create_memo(path: pathlib.Path, mount_table: mounts.Mounts) -> int | None:
    """Create the account's memo file with mode 0600 and return its descriptor.

    None when the state database's directory does not exist yet (the memo never creates it) or is
    on a filesystem off the allowlist, so nothing is created there, or when another process made
    the file first. A new file that fails ``_memo_refusal`` is removed again.
    """
    try:
        parent_stat = os.stat(path.parent)
        if not mounts.device_allowed(mount_table, parent_stat.st_dev, path.parent):
            logger.debug(f"No .pvt walk memo: {path.parent} is on a filesystem off the allowlist")
            return None
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | _MEMO_OPEN, 0o600)
    except FileNotFoundError:
        return None
    except OSError as e:
        logger.debug(f"No .pvt walk memo: cannot create it: {e}")
        return None
    try:
        memo_stat = os.fstat(fd)
        refusal = _memo_refusal(memo_stat, mount_table, path.parent)
        if refusal is None:
            return fd
        logger.debug(f"No .pvt walk memo: the new file {path} {refusal[0]}")
        _unlink_memo(path, memo_stat)
    except OSError as e:
        logger.debug(f"No .pvt walk memo: cannot check the new file {path}: {e}")
    os.close(fd)
    return None


def _replace_memo(
    fd: int, path: pathlib.Path, memo_stat: os.stat_result, mount_table: mounts.Mounts
) -> int | None:
    """Replace a loosened memo of the account's own, open as ``fd``, with a new private file.

    The old file is emptied, so no descriptor opened on it and no other link to it keeps its
    names, and its path is unlinked if it still names it. Returns the new file's descriptor, or
    None (no memo for this walk).
    """
    try:
        os.ftruncate(fd, 0)
        if not _unlink_memo(path, memo_stat):
            return None
    except OSError as e:
        logger.debug(f"No .pvt walk memo: cannot replace {path}: {e}")
        return None
    return _create_memo(path, mount_table)


def _open_memo(mount_table: mounts.Mounts, *, write: bool) -> tuple[int | None, _Memo | None]:
    """Open the account's memo file and read it: its descriptor and its memo.

    The descriptor is None, and the caller lists every directory, when the file cannot be opened
    or an fstat of the open file fails ``_memo_refusal``; a debug log names the reason (not for a
    missing file). For the pre-flight (``write``), a missing memo is created and a loosened one of
    the account's own is replaced. The memo's own path is never followed through a symlink, and a
    FIFO there is not waited on. The memo is None when the file holds no whole memo. Any failure
    means no memo: every directory is listed, as main does.
    """
    try:
        path = _memo_path()
    except _MEMO_ERRORS as e:
        logger.debug(f"Listing every directory: cannot find the .pvt walk memo: {e}")
        return None, None
    try:
        fd = os.open(path, (os.O_RDWR if write else os.O_RDONLY) | _MEMO_OPEN)
    except FileNotFoundError:
        return (_create_memo(path, mount_table) if write else None), None
    except _MEMO_ERRORS as e:
        logger.debug(f"Listing every directory: cannot open the .pvt walk memo: {e}")
        return None, None
    replacement = None
    try:
        memo_stat = os.fstat(fd)
        refusal = _memo_refusal(memo_stat, mount_table, path.parent)
        if refusal is None:
            content = os.pread(fd, memo_stat.st_size, 0)
            memo = _unpack_memo(content)
            if memo is None and content:
                logger.debug(f"Listing every directory: the .pvt walk memo {path} is malformed")
            return fd, memo
        reason, replaceable = refusal
        if write and replaceable:
            logger.debug(f"Replacing the .pvt walk memo {path}: it {reason}")
            replacement = _replace_memo(fd, path, memo_stat, mount_table)
            reason = "could not be replaced"
    except OSError as e:
        reason = f"cannot be read: {e}"
    os.close(fd)
    if replacement is None:
        logger.debug(f"Listing every directory: the .pvt walk memo {path} {reason}")
    return replacement, None


def _write_memo(fd: int, content: bytes) -> None:
    """Write ``content`` over the open memo file, unless another pre-flight is writing it.

    The file is emptied first, so a walk reading it meanwhile, or after this process is killed
    mid-write, sees a prefix of the new memo, which does not parse: that walk lists every
    directory. A write that fails (a full disk, say) is logged and the file emptied again, so no
    cut memo keeps the space.
    """
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        logger.debug(f"Could not record the .pvt walk memo: {e}")
        return
    try:
        os.ftruncate(fd, 0)
        unwritten = memoryview(content)
        while unwritten:
            unwritten = unwritten[os.write(fd, unwritten) :]
    except OSError as e:
        logger.debug(f"Could not record the .pvt walk memo: {e}")
        try:
            os.ftruncate(fd, 0)
        except OSError as truncate_error:
            logger.debug(f"Could not empty the .pvt walk memo: {truncate_error}")


def _trusted_listings(memo: _Memo | None, root: pathlib.Path) -> dict[str, bytes]:
    return memo["dirs"] if memo is not None and memo["root"] == _memo_root(root) else {}


def discover_pvt_files(root: pathlib.Path, *, skip: PvtSkip | None = None) -> dict[str, PvtData]:
    """Find all .pvt files under root, return {data_path: PvtData}.

    Finds what ``os.walk`` over the whole tree finds. Directories the memo vouches for are stat'ed
    instead of listed; every .pvt found is read. Never writes or creates the memo: only the
    repro/run pre-flight does (``discover_pvt_files_and_update_memo``). A .pvt whose ``path:``
    names no file, or a file outside the project, is left out with a warning, so no command it
    feeds writes there (see ``restore_target``). A .pvt ``skip`` returns True for is left out
    too: a stage's own member of its output (``member_skip.own_member_skip``) is no tracked file.
    """
    mount_table = mounts.read_mounts()
    fd, memo = _open_memo(mount_table, write=False)
    if fd is not None:
        os.close(fd)
    candidates, _ = _walk(root, _trusted_listings(memo, root), mount_table, record=False)
    return _read_pvt_files(candidates, root, skip)


def discover_pvt_files_and_update_memo(
    root: pathlib.Path, *, skip: PvtSkip | None = None
) -> dict[str, PvtData]:
    """``discover_pvt_files`` for the repro/run pre-flight, the memo's one writer.

    Opens the account's memo file read-write (``_open_memo``: a missing memo is created with mode
    0600, a loosened one replaced) and, when the walk's listings differ from it, writes in place
    the listing of every directory ``_walk`` may record, dropping every other listing. It never
    creates the state database's directory, so before the first pivot write there it records
    nothing. A memo that cannot be opened or fails ``_memo_refusal`` is neither read nor written
    for this walk.
    """
    real_root = _memo_root(root)
    mount_table = mounts.read_mounts()
    fd, previous = _open_memo(mount_table, write=True)
    try:
        candidates, listings = _walk(
            root, _trusted_listings(previous, root), mount_table, record=fd is not None
        )
        if fd is not None and previous != _Memo(root=real_root, dirs=listings):
            _write_memo(fd, _pack_memo(real_root, listings))
    finally:
        if fd is not None:
            os.close(fd)
    return _read_pvt_files(candidates, root, skip)


def is_import(data: PvtData) -> bool:
    """Check if PvtData represents an imported artifact."""
    return "source" in data


def discover_import_pvt_files(
    root: pathlib.Path, *, skip: PvtSkip | None = None
) -> dict[str, PvtData]:
    """Find all import .pvt files under root (``discover_pvt_files``). ``skip`` is asked about
    import .pvt files alone, so one that reads more than the file (a stage's own member,
    ``member_skip.own_member_skip``) does so only once an import is found."""

    def other_or_skipped(path: pathlib.Path, data: PvtData) -> bool:
        return not is_import(data) or (skip is not None and skip(path, data))

    return discover_pvt_files(root, skip=other_or_skipped)


def pvt_to_hash_info(pvt_data: PvtData) -> HashInfo:
    """Convert PvtData to HashInfo format for cache operations."""
    from pivot.types import DirHash, FileHash

    if "nfiles" not in pvt_data:
        return FileHash(hash=pvt_data["hash"])
    return DirHash(hash=pvt_data["hash"], size=pvt_data["size"], nfiles=pvt_data["nfiles"])


def tracked_members(
    data_path: pathlib.Path,
    pvt_data: PvtData,
    cache_dir: pathlib.Path | None,
    *,
    strict: bool = False,
) -> listing.DirListing:
    """A tracked directory's listing: for a .pvt an older pivot wrote, the listing it holds
    inline (a ``<path>.pvt.listing`` beside it is a stale one of this pivot's, and is named in
    a warning, not read); else its committed ``<path>.pvt.listing``, or the cache's object at
    its hash (``listing.read``, which ``strict`` passes on). Raises ``CorruptListingError`` or
    ``ListingMissingError``."""
    from pivot.types import is_dir_hash

    record = pvt_to_hash_info(pvt_data)
    assert is_dir_hash(record), "only a tracked directory has a listing"
    pvt_path = get_pvt_path(data_path)
    parsed = _load_pvt(pvt_path)
    if isinstance(parsed, tuple) and parsed[1] is not None and parsed[0]["hash"] == record["hash"]:
        return parsed[1]
    return listing.read(listing.tracked_listing_path(pvt_path), record, cache_dir, strict=strict)
