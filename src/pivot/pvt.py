"""PVT file handling for tracked files.

.pvt files are YAML manifests that track arbitrary data files/directories,
similar to DVC's .dvc files. They sit alongside the tracked data for
discoverability.
"""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING, NotRequired, TypedDict, TypeGuard, cast

import yaml

from pivot import cache

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pathlib

    from pivot.types import DirManifestEntry

# Use union types to avoid type: ignore on fallback assignment
_Loader: type[yaml.SafeLoader] | type[yaml.CSafeLoader]
_Dumper: type[yaml.SafeDumper] | type[yaml.CSafeDumper]

try:
    _Loader = yaml.CSafeLoader
    _Dumper = yaml.CSafeDumper
except AttributeError:
    _Loader = yaml.SafeLoader
    _Dumper = yaml.SafeDumper


class PvtData(TypedDict):
    """Data stored in .pvt files."""

    path: str  # Relative path to tracked file/directory
    hash: str  # Content hash
    size: int  # Total size (file or sum of directory)
    num_files: NotRequired[int]  # For directories only
    manifest: NotRequired[list[DirManifestEntry]]  # For directories only


_REQUIRED_KEYS = frozenset({"path", "hash", "size"})
_VALID_KEYS = frozenset({"path", "hash", "size", "num_files", "manifest"})
_PATH_TRAVERSAL = re.compile(r"(^|/)\.\.(/|$)")


def _is_pvt_data(data: object) -> TypeGuard[PvtData]:
    """Validate that parsed YAML has valid PvtData structure."""
    if not isinstance(data, dict):
        return False
    str_data = cast("dict[str, object]", data)
    if not _REQUIRED_KEYS.issubset(str_data.keys()):
        return False
    return all(key in _VALID_KEYS for key in str_data)


def has_path_traversal(path: str) -> bool:
    """Check if path contains traversal components (..)."""
    return bool(_PATH_TRAVERSAL.search(path))


def _validate_path(path: str) -> None:
    """Validate path doesn't contain traversal."""
    if has_path_traversal(path):
        raise ValueError(f"Path contains path traversal: {path!r}")


def write_pvt_file(pvt_path: pathlib.Path, data: PvtData) -> None:
    """Write .pvt file atomically."""
    _validate_path(data["path"])

    def write_yaml(fd: int) -> None:
        with os.fdopen(fd, "w") as f:
            yaml.dump(dict(data), f, Dumper=_Dumper, sort_keys=False)

    cache.atomic_write_file(pvt_path, write_yaml)


def read_pvt_file(pvt_path: pathlib.Path) -> PvtData | None:
    """Read .pvt file, return None if missing or invalid."""
    try:
        with open(pvt_path) as f:
            data: object = yaml.load(f, Loader=_Loader)
        if not _is_pvt_data(data):
            return None
        # Validate path doesn't contain traversal (security check)
        if has_path_traversal(data["path"]):
            return None
        return data
    except (FileNotFoundError, UnicodeDecodeError, yaml.YAMLError):
        return None


def get_pvt_path(data_path: pathlib.Path) -> pathlib.Path:
    """Get .pvt path for a data path (file.csv -> file.csv.pvt, dir/ -> dir.pvt)."""
    # pathlib normalizes trailing slashes, so "images/" becomes "images"
    return data_path.with_suffix(data_path.suffix + ".pvt")


def get_data_path(pvt_path: pathlib.Path) -> pathlib.Path:
    """Get data path from .pvt path."""
    if not pvt_path.suffix == ".pvt":
        raise ValueError(f"Expected .pvt file, got: {pvt_path}")
    # Remove .pvt suffix
    return pvt_path.with_suffix("")


def discover_pvt_files(root: pathlib.Path) -> dict[str, PvtData]:
    """Find all .pvt files under root, return {data_path: PvtData}."""
    from pivot import project

    result = dict[str, PvtData]()

    for pvt_path in root.rglob("*.pvt"):
        if not pvt_path.is_file():
            continue
        data = read_pvt_file(pvt_path)
        if data is None:
            logger.warning(f"Skipping invalid .pvt file: {pvt_path}")
            continue

        # Compute absolute data path from pvt file location + relative path
        # Use normalized path (preserve symlinks) for portability
        data_path = pvt_path.parent / data["path"]
        normalized = project.normalize_path(str(data_path))
        result[str(normalized)] = data

    return result
