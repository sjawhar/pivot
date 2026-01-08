from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING, Literal, TypedDict, TypeGuard

import yaml

from pivot import cache, exceptions, git, lock, project, pvt, remote, yaml_config
from pivot.types import DirHash, FileHash

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pivot.pvt import PvtData
    from pivot.types import OutEntry, OutputHash, StorageLockData

logger = logging.getLogger(__name__)


class TargetInfo(TypedDict):
    """Information about a target to retrieve."""

    target_type: Literal["file", "stage"]
    original_target: str
    paths: list[str]
    hashes: dict[str, OutputHash]


def _parse_yaml_bytes[T](
    content: bytes,
    validator: Callable[[object], TypeGuard[T]],
) -> T | None:
    """Parse YAML bytes and validate. Returns None on any failure."""
    try:
        data: object = yaml.load(content, Loader=yaml_config.Loader)
        if validator(data):
            return data
        return None
    except yaml.YAMLError:
        return None


def _parse_lock_data_from_bytes(content: bytes) -> StorageLockData | None:
    """Parse lock file content from bytes. Requires 'outs' key for get operations."""
    data = _parse_yaml_bytes(content, lock.is_lock_data)
    if data is None:
        return None
    # pivot get requires outs to know what to restore
    if "outs" not in data:
        return None
    return data


def _parse_pvt_data_from_bytes(content: bytes) -> PvtData | None:
    """Parse .pvt file content from bytes."""
    data = _parse_yaml_bytes(content, pvt.is_pvt_data)
    if data is None:
        return None
    # Security check: no path traversal in stored path
    if pvt.has_path_traversal(data["path"]):
        return None
    return data


def get_lock_data_from_revision(
    stage_name: str, rev: str, cache_dir: pathlib.Path
) -> StorageLockData | None:
    """Read and parse lock file for a stage from a git revision."""
    rel_path = str(
        cache_dir.relative_to(project.get_project_root()) / "stages" / f"{stage_name}.lock"
    )
    content = git.read_file_from_revision(rel_path, rev)
    if content is None:
        return None
    return _parse_lock_data_from_bytes(content)


def get_pvt_data_from_revision(pvt_rel_path: str, rev: str) -> PvtData | None:
    """Read and parse .pvt file from a git revision."""
    content = git.read_file_from_revision(pvt_rel_path, rev)
    if content is None:
        return None
    return _parse_pvt_data_from_bytes(content)


def _out_entry_to_output_hash(entry: OutEntry) -> OutputHash:
    """Convert OutEntry to OutputHash."""
    if entry["hash"] is None:
        return None
    if "manifest" in entry:
        return DirHash(hash=entry["hash"], manifest=entry["manifest"])
    return FileHash(hash=entry["hash"])


def _normalize_target_path(target: str, proj_root: pathlib.Path) -> str:
    """Normalize target to relative path, validating it's within project."""
    # Security check: reject path traversal in user input
    if pvt.has_path_traversal(target):
        raise exceptions.TargetNotFoundError(f"Path traversal not allowed in target: {target!r}")

    target_path = pathlib.Path(target)
    if not target_path.is_absolute():
        target_path = proj_root / target_path

    try:
        return str(target_path.relative_to(proj_root))
    except ValueError:
        raise exceptions.TargetNotFoundError(
            f"Target path is outside project root: {target!r}"
        ) from None


def resolve_targets(
    targets: Sequence[str],
    rev: str,
    cache_dir: pathlib.Path,
) -> list[TargetInfo]:
    """Resolve targets to TargetInfo, determining if each is a file or stage."""
    proj_root = project.get_project_root()
    results = list[TargetInfo]()

    for target in targets:
        # Try as stage name first (stage names don't have path separators)
        if "/" not in target and "\\" not in target:
            lock_data = get_lock_data_from_revision(target, rev, cache_dir)
            if lock_data is not None and "outs" in lock_data:
                outs = lock_data["outs"]
                paths = [entry["path"] for entry in outs]
                hashes = {entry["path"]: _out_entry_to_output_hash(entry) for entry in outs}
                results.append(
                    TargetInfo(
                        target_type="stage",
                        original_target=target,
                        paths=paths,
                        hashes=hashes,
                    )
                )
                continue

        # Normalize path (validates traversal and project bounds)
        rel_target = _normalize_target_path(target, proj_root)
        pvt_rel_path = rel_target + ".pvt"

        # Try as a .pvt tracked file
        pvt_data = get_pvt_data_from_revision(pvt_rel_path, rev)
        if pvt_data is not None:
            file_hash = pvt_data["hash"]
            hash_info: OutputHash
            if "manifest" in pvt_data:
                hash_info = DirHash(hash=file_hash, manifest=pvt_data["manifest"])
            else:
                hash_info = FileHash(hash=file_hash)
            results.append(
                TargetInfo(
                    target_type="file",
                    original_target=target,
                    paths=[rel_target],
                    hashes={rel_target: hash_info},
                )
            )
            continue

        # Try as a git-tracked file
        content = git.read_file_from_revision(rel_target, rev)
        if content is not None:
            results.append(
                TargetInfo(
                    target_type="file",
                    original_target=target,
                    paths=[rel_target],
                    hashes={rel_target: None},  # None means must use git
                )
            )
            continue

        raise exceptions.TargetNotFoundError(
            f"Target '{target}' not found at revision '{rev}' (not a stage name, tracked file, or git-tracked file)"
        )

    return results


def restore_file(
    rel_path: str,
    output_hash: OutputHash,
    rev: str,
    dest_path: pathlib.Path,
    cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
    force: bool,
) -> str:
    """Restore a single file from revision. Returns status message."""
    if dest_path.exists() and not force:
        return f"Skipped: {dest_path} (already exists, use --force to overwrite)"

    if force and dest_path.exists():
        cache.remove_output(dest_path)

    # Strategy 1: Try local cache (if hash available)
    if output_hash is not None:
        file_hash = output_hash["hash"]
        cached_path = cache.get_cache_path(cache_dir / "files", file_hash)
        if cached_path.exists():
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            success = cache.restore_from_cache(
                dest_path,
                output_hash,
                cache_dir / "files",
                checkout_modes=checkout_modes,
            )
            if success:
                return f"Restored: {dest_path} (from cache)"

    # Strategy 2: Try git fallback
    content = git.read_file_from_revision(rel_path, rev)
    if content is not None:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(content)
        return f"Restored: {dest_path} (from git)"

    # Strategy 3: Try remote fallback (if hash available)
    if output_hash is not None:
        file_hash = output_hash["hash"]
        content = remote.fetch_from_remote(file_hash)
        if content is not None:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(content)
            # Cache the fetched content for future use
            cached_path = cache.get_cache_path(cache_dir / "files", file_hash)
            cached_path.parent.mkdir(parents=True, exist_ok=True)
            cached_path.write_bytes(content)
            return f"Restored: {dest_path} (from remote)"

    raise exceptions.CacheMissError(
        f"Cannot retrieve '{rel_path}': not in local cache, git, or remote"
    )


def restore_targets_from_revision(
    targets: Sequence[str],
    rev: str,
    output: pathlib.Path | None,
    cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
    force: bool,
) -> list[str]:
    """Restore targets from a git revision. Returns list of status messages."""
    proj_root = project.get_project_root()

    # Validate revision exists
    commit_sha = git.resolve_revision(rev)
    if commit_sha is None:
        raise exceptions.RevisionNotFoundError(f"Cannot resolve revision: '{rev}'")

    # Resolve targets
    target_infos = resolve_targets(targets, rev, cache_dir)

    # Validate -o usage
    if output is not None:
        if len(target_infos) != 1:
            raise exceptions.GetError("--output/-o can only be used with a single target")
        if target_infos[0]["target_type"] == "stage":
            raise exceptions.GetError(
                "--output/-o cannot be used with stage names (stages have multiple outputs)"
            )
        if len(target_infos[0]["paths"]) != 1:
            raise exceptions.GetError("--output/-o cannot be used with directory targets")

    messages = list[str]()

    for target_info in target_infos:
        for rel_path in target_info["paths"]:
            output_hash = target_info["hashes"].get(rel_path)
            dest_path = output if output is not None else proj_root / rel_path

            msg = restore_file(
                rel_path=rel_path,
                output_hash=output_hash,
                rev=rev,
                dest_path=dest_path,
                cache_dir=cache_dir,
                checkout_modes=checkout_modes,
                force=force,
            )
            messages.append(msg)

    return messages
