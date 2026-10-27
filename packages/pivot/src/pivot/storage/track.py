from __future__ import annotations

import functools
import logging
import os
import pathlib
import shlex
from typing import TYPE_CHECKING, NotRequired, TypedDict, TypeGuard, cast

import yaml

from pivot import exceptions
from pivot.storage import cache, listing

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Mapping

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


def discover_pvt_files(root: pathlib.Path) -> dict[str, PvtData]:
    """Find all .pvt files under root, return {data_path: PvtData}."""
    from pivot import project

    result = dict[str, PvtData]()

    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        for fname in filenames:
            if not fname.endswith(".pvt"):
                continue
            pvt_path = pathlib.Path(dirpath) / fname
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

            data_path = pvt_path.parent / data["path"]
            normalized = project.normalize_path(data_path)
            result[str(normalized)] = data

    return result


def is_import(data: PvtData) -> bool:
    """Check if PvtData represents an imported artifact."""
    return "source" in data


def discover_import_pvt_files(root: pathlib.Path) -> dict[str, PvtData]:
    """Find all import .pvt files under root."""
    return {path: data for path, data in discover_pvt_files(root).items() if is_import(data)}


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
