from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import tabulate
import yaml

from pivot import git, yaml_config
from pivot.types import ChangeType, OutputFormat, StorageLockData

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def read_lock_files_from_head(
    stage_names: Sequence[str],
) -> dict[str, StorageLockData | None]:
    """Batch read and parse lock files from git HEAD.

    Returns {stage_name: parsed_lock_data or None}.
    """
    if not stage_names:
        return {}

    lock_paths = [f".pivot/cache/stages/{name}.lock" for name in stage_names]
    lock_contents = git.read_files_from_head(lock_paths)

    result = dict[str, StorageLockData | None]()
    for stage_name in stage_names:
        lock_path = f".pivot/cache/stages/{stage_name}.lock"
        content = lock_contents.get(lock_path)
        if content is None:
            result[stage_name] = None
            continue

        try:
            data: object = yaml.load(content, Loader=yaml_config.Loader)
        except yaml.YAMLError:
            result[stage_name] = None
            continue

        if not isinstance(data, dict):
            result[stage_name] = None
            continue

        # YAML parse result is dict[Unknown, Unknown]; cast through object to StorageLockData
        result[stage_name] = cast("StorageLockData", cast("object", data))

    return result


def extract_output_hashes_from_lock(
    lock_data: StorageLockData,
) -> dict[str, str | None]:
    """Extract path -> hash mapping from lock data 'outs' field."""
    if "outs" not in lock_data:
        return {}

    outs_list = lock_data["outs"]
    result = dict[str, str | None]()
    for out in outs_list:
        if "path" in out:
            result[out["path"]] = out["hash"]
    return result


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
