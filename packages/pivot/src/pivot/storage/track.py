from __future__ import annotations

import functools
import logging
import os
import pathlib
from typing import TYPE_CHECKING, NotRequired, TypedDict, TypeGuard, cast

import yaml

from pivot import exceptions
from pivot.storage import cache, listing

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
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


@functools.cache
def _warn_old_format(pvt_path: str, data_path: str) -> None:
    """Warn once per process for each directory .pvt an older pivot wrote."""
    logger.warning(
        "%s was written by an older pivot, with its files listed inside it; pivot reads that "
        + "list as the directory's listing. Run `pivot track --force %s` to convert it to a "
        + ".pvt and a committed .pvt.listing file",
        pvt_path,
        data_path,
    )


def parse_pvt(data: object) -> tuple[PvtData, listing.DirListing | None] | None:
    """A parsed .pvt's data, or None if invalid or insecure.

    A directory .pvt an older pivot wrote lists its files inline; it is read as this format's
    record, and that list is returned as the directory's listing (None for any other .pvt).
    """
    if isinstance(data, dict) and _OLD_FORMAT_KEYS & cast("dict[str, object]", data).keys():
        document = cast("dict[str, object]", data)
        members = listing.from_inline(document.get("manifest"), document.get("empty_dirs", []))
        if members is None:
            return None
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


def _load_pvt(pvt_path: pathlib.Path) -> tuple[PvtData, listing.DirListing | None] | None:
    """A .pvt file's data and, for an older pivot's directory .pvt, its inline listing (with
    one warning per process saying how to convert it); None if missing, invalid, or insecure.
    """
    try:
        with open(pvt_path) as f:
            data: object = yaml.load(f, Loader=_Loader)
    except (FileNotFoundError, PermissionError, UnicodeDecodeError, yaml.YAMLError):
        return None
    parsed = parse_pvt(data)
    if parsed is not None and parsed[1] is not None:
        _warn_old_format(str(pvt_path), str(get_data_path(pvt_path)))
    return parsed


def read_pvt_file(pvt_path: pathlib.Path) -> PvtData | None:
    """Read .pvt manifest file; returns None if missing, invalid, or insecure."""
    parsed = _load_pvt(pvt_path)
    return None if parsed is None else parsed[0]


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
    data_path: pathlib.Path, pvt_data: PvtData, cache_dir: pathlib.Path | None
) -> listing.DirListing:
    """A tracked directory's listing: its committed ``<path>.pvt.listing``; else, for a .pvt an
    older pivot wrote, the listing it holds inline; else the cache's object at its hash.
    Raises ``CorruptListingError`` or ``ListingMissingError``."""
    from pivot.types import is_dir_hash

    record = pvt_to_hash_info(pvt_data)
    assert is_dir_hash(record), "only a tracked directory has a listing"
    pvt_path = get_pvt_path(data_path)
    committed = listing.tracked_listing_path(pvt_path)
    if not committed.exists():
        parsed = _load_pvt(pvt_path)
        if parsed is not None and parsed[1] is not None and parsed[0]["hash"] == record["hash"]:
            return parsed[1]
    return listing.read(committed, record, cache_dir)
