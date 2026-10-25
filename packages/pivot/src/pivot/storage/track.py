from __future__ import annotations

import logging
import os
import pathlib
from typing import TYPE_CHECKING, NotRequired, TypedDict, TypeGuard, cast

import yaml

from pivot import exceptions
from pivot.storage import cache

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pivot.registry import RegistryStageInfo
    from pivot.types import DirManifestEntry, HashInfo

# Directories that hold git's or pivot's own state, never a tracked path's .pvt
_STATE_DIR_NAMES = frozenset({".git", ".pivot"})

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
    remote: str  # Source's S3 remote URL


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


def discover_pvt_files(
    root: pathlib.Path, stages: Iterable[RegistryStageInfo]
) -> dict[str, PvtData]:
    """Find all .pvt files under root, return {data_path: PvtData}.

    The walk enters no ``.git`` or ``.pivot`` directory and no output of ``stages``:
    ``pivot track`` refuses a path that overlaps a stage output, so no .pvt lives inside one,
    and an output directory can hold far more files than the rest of the project.
    """
    from pivot import project
    from pivot.storage import store

    outputs = {
        os.path.normpath(store.relative_output_path(out)) for info in stages for out in info["outs"]
    }
    result = dict[str, PvtData]()

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        rel_dir = os.path.relpath(dirpath, root)
        dirnames[:] = [
            name
            for name in dirnames
            if name not in _STATE_DIR_NAMES
            and os.path.normpath(os.path.join(rel_dir, name)) not in outputs
        ]
        for fname in filenames:
            if not fname.endswith(".pvt"):
                continue
            pvt_path = pathlib.Path(dirpath) / fname
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


def is_import(data: PvtData) -> bool:
    """Check if PvtData represents an imported artifact."""
    return "source" in data


def discover_import_pvt_files(
    root: pathlib.Path, stages: Iterable[RegistryStageInfo]
) -> dict[str, PvtData]:
    """Find all import .pvt files under root (``stages`` as for ``discover_pvt_files``)."""
    return {
        path: data for path, data in discover_pvt_files(root, stages).items() if is_import(data)
    }


def pvt_to_hash_info(pvt_data: PvtData) -> HashInfo:
    """Convert PvtData to HashInfo format for cache operations."""
    from pivot.types import DirHash, FileHash

    if "manifest" not in pvt_data:
        return FileHash(hash=pvt_data["hash"])
    dir_hash = DirHash(hash=pvt_data["hash"], manifest=pvt_data["manifest"])
    if "empty_dirs" in pvt_data and pvt_data["empty_dirs"]:
        dir_hash["empty_dirs"] = pvt_data["empty_dirs"]
    return dir_hash
