from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import tabulate
import yaml

from pivot import exceptions, git, project, types, yaml_config
from pivot.storage import lock
from pivot.types import ChangeType, OutputFormat, StorageLockData

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pivot.storage.store import WorkspaceStore

logger = logging.getLogger(__name__)


def _parse_lock_contents(
    stage_names: Sequence[str],
    lock_contents: dict[str, bytes],
    stages_dir: str,
) -> dict[str, StorageLockData | None]:
    """Parse lock file contents into StorageLockData."""
    result = dict[str, StorageLockData | None]()
    for stage_name in stage_names:
        lock_path = f"{stages_dir}/{stage_name}.lock"
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
    try:
        stages_dir = lock.stages_rel_path(project.get_state_dir(), project.get_project_root())
    except ValueError:
        raise exceptions.ConfigError(
            "core.state_dir is outside the project, so lock files are not in git"
        ) from None
    lock_paths = [f"{stages_dir}/{name}.lock" for name in stage_names]
    lock_contents = git.read_files_from_head(lock_paths)
    return _parse_lock_contents(stage_names, lock_contents, stages_dir)


def extract_output_hashes_from_lock(
    lock_data: StorageLockData,
    stage_name: str,
) -> dict[str, str]:
    """Map each output recorded in a stage's lock data to its hash, keyed by identity key."""
    return {
        types.identity_key(types.ArtifactIdentity(stage_name, out["key"])): out["hash"]
        for out in lock_data["outs"]
    }


def output_hashes_from_head(
    outs_by_stage: Mapping[str, Sequence[types.ArtifactRef]],
    store: WorkspaceStore,
) -> dict[str, str | None]:
    """Map each output's workspace path (relative to the project root) to its hash at git HEAD.

    An output maps to None when its stage has no lock file at HEAD or the lock does not record it.
    """
    proj_root = project.get_project_root()
    lock_data_map = read_lock_files_from_head(list(outs_by_stage))
    result = dict[str, str | None]()
    for stage_name, outs in outs_by_stage.items():
        lock_data = lock_data_map[stage_name]
        hashes = {} if lock_data is None else extract_output_hashes_from_lock(lock_data, stage_name)
        for out in outs:
            rel_path = project.to_relative_path(store.resolve_display_path(out), proj_root)
            result[rel_path] = hashes.get(types.identity_key(out.identity))
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
