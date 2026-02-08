from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import tabulate
import yaml

from pivot import git, yaml_config
from pivot.storage import lock
from pivot.types import ChangeType, OutputFormat, StorageLockData

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)


def _parse_lock_contents(
    stage_names: Sequence[str],
    lock_contents: dict[str, bytes],
) -> dict[str, StorageLockData | None]:
    """Parse lock file contents into StorageLockData."""
    result = dict[str, StorageLockData | None]()
    for stage_name in stage_names:
        lock_path = f"{lock.STAGES_REL_PATH}/{stage_name}.lock"
        content = lock_contents.get(lock_path)
        if content is None:
            result[stage_name] = None
            continue

        try:
            data: object = yaml.load(content, Loader=yaml_config.Loader)
        except yaml.YAMLError as e:
            logger.debug(f"Failed to parse lock file for {stage_name}: {e}")
            result[stage_name] = None
            continue

        if not lock.is_lock_data(data):
            result[stage_name] = None
            continue

        result[stage_name] = data

    return result


def read_lock_files_from_head(
    stage_names: Sequence[str],
) -> dict[str, StorageLockData | None]:
    """Batch read and parse lock files from git HEAD."""
    if not stage_names:
        return {}
    lock_paths = [f"{lock.STAGES_REL_PATH}/{name}.lock" for name in stage_names]
    lock_contents = git.read_files_from_head(lock_paths)
    return _parse_lock_contents(stage_names, lock_contents)


def read_lock_files_from_revision(
    stage_names: Sequence[str],
    rev: str,
) -> dict[str, StorageLockData | None]:
    """Batch read and parse lock files from a specific git revision."""
    if not stage_names:
        return {}
    lock_paths = [f"{lock.STAGES_REL_PATH}/{name}.lock" for name in stage_names]
    lock_contents = git.read_files_from_revision(lock_paths, rev)
    return _parse_lock_contents(stage_names, lock_contents)


def extract_output_hashes_from_lock(
    lock_data: StorageLockData,
) -> dict[str, str | None]:
    """Extract path -> hash mapping from lock data 'outs' field."""
    return {out["path"]: out["hash"] for out in lock_data["outs"]}


def format_table(
    rows: list[list[str]],
    headers: list[str],
    output_format: OutputFormat | None,
    empty_message: str,
) -> str:
    """Format rows as plain/markdown table."""
    if not rows:
        return empty_message

    tablefmt = "github" if output_format == OutputFormat.MD else "plain"
    return tabulate.tabulate(rows, headers=headers, tablefmt=tablefmt, disable_numparse=True)


def format_json(data: Mapping[str, Any] | list[Any]) -> str:
    """Format data as indented JSON string."""
    return json.dumps(data, indent=2)


def build_two_level_diff[V](
    old: Mapping[str, Mapping[str, V]],
    new: Mapping[str, Mapping[str, V]],
) -> list[tuple[str, str, V | None, V | None, ChangeType]]:
    """Build diff list for two-level nested mappings (e.g., {path: {key: value}})."""
    diffs = list[tuple[str, str, V | None, V | None, ChangeType]]()
    all_keys1 = set(old.keys()) | set(new.keys())

    for key1 in sorted(all_keys1):
        old_inner = old.get(key1, {})
        new_inner = new.get(key1, {})
        all_keys2 = set(old_inner.keys()) | set(new_inner.keys())

        for key2 in sorted(all_keys2):
            old_val = old_inner.get(key2)
            new_val = new_inner.get(key2)

            if key2 not in old_inner:
                diffs.append((key1, key2, None, new_val, ChangeType.ADDED))
            elif key2 not in new_inner:
                diffs.append((key1, key2, old_val, None, ChangeType.REMOVED))
            elif old_val != new_val:
                diffs.append((key1, key2, old_val, new_val, ChangeType.MODIFIED))

    return diffs
