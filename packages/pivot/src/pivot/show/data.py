from __future__ import annotations

import collections
import csv
import dataclasses
import json
import logging
import os
import pathlib
import re
from typing import TYPE_CHECKING, Any, TypedDict, cast, override

import numpy
import pandas
import tabulate

from pivot import config, loaders, project, types
from pivot.show import common
from pivot.storage import cache, lock
from pivot.types import (
    ChangeType,
    DataDiffResult,
    DataFileFormat,
    OutputFormat,
    RowChange,
    SchemaChange,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pivot.registry import RegistryStageInfo
    from pivot.storage import store as store_mod

# Status symbols for diff display (plain text)
STATUS_SYMBOLS: dict[ChangeType, str] = {
    ChangeType.ADDED: "+",
    ChangeType.REMOVED: "-",
    ChangeType.MODIFIED: "~",
}

logger = logging.getLogger(__name__)


_DATA_EXTENSIONS: dict[str, DataFileFormat] = {
    ".csv": DataFileFormat.CSV,
    ".json": DataFileFormat.JSON,
    ".jsonl": DataFileFormat.JSONL,
}

# Default threshold for in-memory comparison (100MB)
_MAX_IN_MEMORY_SIZE = 100 * 1024 * 1024

# Hash validation pattern for xxhash64 (16 hex characters)
_HASH_PATTERN = re.compile(r"^[0-9a-f]{16}$")


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types by converting to Python equivalents."""

    @override
    def default(self, o: object) -> object:
        if isinstance(o, numpy.integer):
            return int(o)  # pyright: ignore[reportUnknownArgumentType] - numpy.integer stub is generic
        if isinstance(o, numpy.floating):
            return float(o)  # pyright: ignore[reportUnknownArgumentType] - numpy.floating stub is generic
        if isinstance(o, numpy.bool_):
            return bool(o)
        if isinstance(o, numpy.ndarray):
            return o.tolist()
        if isinstance(o, (pandas.Timestamp, pandas.Timedelta)):
            return str(o)
        # Handle NaN/NaT values (float, numpy.floating, pandas.NaT, etc.)
        # Cast to Any for pandas.isna() which accepts dynamic types from JSON
        try:
            if pandas.isna(cast("Any", o)):
                return None
        except (TypeError, ValueError):  # pragma: no cover
            pass
        return super().default(o)  # pragma: no cover


class DataDiffEntry(TypedDict):
    """High-level entry for file-level diff (hash changed or not)."""

    path: str
    old_hash: str | None
    new_hash: str | None
    change_type: ChangeType


class DataError(Exception):
    """Error loading or processing data files."""


def detect_format(path: pathlib.Path) -> DataFileFormat:
    """Detect data file format from extension."""
    suffix = path.suffix.lower()
    return _DATA_EXTENSIONS.get(suffix, DataFileFormat.UNKNOWN)


def _format_from_loader(
    loader: loaders.Reader[object] | loaders.Writer[object] | loaders.Loader[object, object],
) -> DataFileFormat | None:
    if isinstance(loader, loaders.CSV):
        return DataFileFormat.CSV
    if isinstance(loader, loaders.JSON):
        return DataFileFormat.JSON
    if isinstance(loader, (loaders.JSONL, loaders.DataFrameJSONL)):
        return DataFileFormat.JSONL
    return None


def _is_data_output(
    out: types.ArtifactRef, store: store_mod.WorkspaceStore, project_root: pathlib.Path
) -> bool:
    """True for a stage output that `pivot diff` can compare (CSV, JSON or JSONL data)."""
    if out.tag in {types.ArtifactTag.METRIC, types.ArtifactTag.PLOT}:
        return False
    if _format_from_loader(out.format) is not None:
        return True
    rel_path = project.to_relative_path(store.resolve_display_path(out), project_root)
    return detect_format(pathlib.Path(rel_path)) != DataFileFormat.UNKNOWN


def load_dataframe(path: pathlib.Path) -> pandas.DataFrame:
    """Load data file as pandas DataFrame."""
    match detect_format(path):
        case DataFileFormat.CSV:
            return pandas.read_csv(path)  # pyright: ignore[reportUnknownMemberType] - pandas-stubs incomplete
        case DataFileFormat.JSON:
            return pandas.read_json(path)
        case DataFileFormat.JSONL:
            return loaders.DataFrameJSONL().load(path)
        case _:
            raise DataError(f"Unsupported data format: {path.suffix}")


def get_schema(path: pathlib.Path) -> dict[str, str]:
    """Get column name -> dtype mapping without loading full data.

    For large files, reads only the first few rows to infer schema.
    """
    match detect_format(path):
        case DataFileFormat.CSV:
            df = pandas.read_csv(path, nrows=100)  # pyright: ignore[reportUnknownMemberType]
        case DataFileFormat.JSON:
            df = pandas.read_json(path)
        case DataFileFormat.JSONL:
            df = loaders.DataFrameJSONL().load(path, max_rows=100)
        case _:
            raise DataError(f"Unsupported data format: {path.suffix}")
    return {str(col): str(dtype) for col, dtype in df.dtypes.items()}


def get_row_count(path: pathlib.Path) -> int:
    """Get row count without loading full data."""
    match detect_format(path):
        case DataFileFormat.CSV:
            # Use csv.reader to handle embedded newlines in quoted fields
            with open(path, newline="") as f:
                reader = csv.reader(f)
                count = sum(1 for _ in reader)
            # Subtract header row, but don't go negative for empty files
            return max(0, count - 1)
        case DataFileFormat.JSONL:
            return loaders.DataFrameJSONL().row_count(path)
        case _:
            # For JSON, we need to load it
            df = load_dataframe(path)
            return len(df)


def check_reorder_only(old_df: pandas.DataFrame, new_df: pandas.DataFrame) -> bool:
    """Check if two dataframes have identical content but different row order."""
    if old_df.shape != new_df.shape:
        return False
    if list(old_df.columns) != list(new_df.columns):
        return False

    # Sort both by all columns and compare
    sort_cols = list(old_df.columns)
    try:
        old_sorted = old_df.sort_values(by=sort_cols, ignore_index=True)
        new_sorted = new_df.sort_values(by=sort_cols, ignore_index=True)
        return old_sorted.equals(new_sorted)
    except TypeError:  # pragma: no cover
        # Some columns may not be sortable (e.g., mixed types)
        # Fall back to comparing multisets (Counter preserves duplicate counts)
        try:
            old_counter = collections.Counter(tuple(row) for row in old_df.itertuples(index=False))
            new_counter = collections.Counter(tuple(row) for row in new_df.itertuples(index=False))
            return old_counter == new_counter
        except TypeError:
            # Rows contain unhashable types (list, dict) - can't determine reorder
            return False


def _diff_schema(
    old_cols: list[str],
    new_cols: list[str],
    old_dtypes: dict[str, str],
    new_dtypes: dict[str, str],
) -> list[SchemaChange]:
    """Compare schemas and return list of changes."""
    changes = list[SchemaChange]()
    all_cols = set(old_cols) | set(new_cols)

    for col in sorted(all_cols):
        if col not in old_cols:
            changes.append(
                SchemaChange(
                    column=col,
                    old_dtype=None,
                    new_dtype=new_dtypes.get(col),
                    change_type=ChangeType.ADDED,
                )
            )
        elif col not in new_cols:
            changes.append(
                SchemaChange(
                    column=col,
                    old_dtype=old_dtypes.get(col),
                    new_dtype=None,
                    change_type=ChangeType.REMOVED,
                )
            )
        elif old_dtypes.get(col) != new_dtypes.get(col):
            changes.append(
                SchemaChange(
                    column=col,
                    old_dtype=old_dtypes.get(col),
                    new_dtype=new_dtypes.get(col),
                    change_type=ChangeType.MODIFIED,
                )
            )

    return changes


def _row_to_dict(row: pandas.Series[Any], columns: list[str]) -> dict[str, Any]:
    """Convert pandas row to dict, handling NaN values."""
    result = dict[str, Any]()
    for col in columns:
        val = row[col]
        # Convert NaN to None for cleaner output
        if pandas.isna(val):
            result[col] = None
        else:
            result[col] = val
    return result


def _make_row_key(row: pandas.Series[Any], key_columns: list[str]) -> str:
    """Create a collision-free key from row values using JSON serialization."""
    # Convert to list, handling NaN values explicitly
    vals = [None if pandas.isna(row[k]) else row[k] for k in key_columns]
    return json.dumps(vals, sort_keys=True, default=str)


def _diff_rows_by_key(
    old_df: pandas.DataFrame,
    new_df: pandas.DataFrame,
    key_columns: list[str],
    common_cols: list[str],
    max_rows: int,
) -> tuple[list[RowChange], bool]:
    """Diff rows using key columns for matching. Returns (changes, truncated)."""
    changes = list[RowChange]()

    # Build key -> row index mapping, checking for duplicates
    old_keys = dict[str, int]()
    for i in range(len(old_df)):
        row = old_df.iloc[i]
        key = _make_row_key(row, key_columns)
        if key in old_keys:
            msg = f"Duplicate key {key} in old file at rows {old_keys[key]} and {i}. Key columns must be unique for key-based diff."
            raise DataError(msg)
        old_keys[key] = i

    new_keys = dict[str, int]()
    for i in range(len(new_df)):
        row = new_df.iloc[i]
        key = _make_row_key(row, key_columns)
        if key in new_keys:
            msg = f"Duplicate key {key} in new file at rows {new_keys[key]} and {i}. Key columns must be unique for key-based diff."
            raise DataError(msg)
        new_keys[key] = i

    all_keys = set(old_keys.keys()) | set(new_keys.keys())
    truncated = False

    for key in sorted(all_keys):
        if len(changes) >= max_rows:
            truncated = True
            break

        if key not in old_keys:
            # Added row
            new_row = new_df.iloc[new_keys[key]]
            changes.append(
                RowChange(
                    key=key,
                    change_type=ChangeType.ADDED,
                    old_values=None,
                    new_values=_row_to_dict(new_row, common_cols),
                )
            )
        elif key not in new_keys:
            # Removed row
            old_row = old_df.iloc[old_keys[key]]
            changes.append(
                RowChange(
                    key=key,
                    change_type=ChangeType.REMOVED,
                    old_values=_row_to_dict(old_row, common_cols),
                    new_values=None,
                )
            )
        else:
            # Check if modified
            old_row = old_df.iloc[old_keys[key]]
            new_row = new_df.iloc[new_keys[key]]
            old_dict = _row_to_dict(old_row, common_cols)
            new_dict = _row_to_dict(new_row, common_cols)
            if old_dict != new_dict:
                changes.append(
                    RowChange(
                        key=key,
                        change_type=ChangeType.MODIFIED,
                        old_values=old_dict,
                        new_values=new_dict,
                    )
                )

    return changes, truncated


def _diff_rows_positional(
    old_df: pandas.DataFrame,
    new_df: pandas.DataFrame,
    common_cols: list[str],
    max_rows: int,
) -> tuple[list[RowChange], bool]:
    """Diff rows by position (row-by-row). Returns (changes, truncated)."""
    changes = list[RowChange]()
    truncated = False

    max_len = max(len(old_df), len(new_df))
    for i in range(max_len):
        if len(changes) >= max_rows:
            truncated = True
            break

        if i >= len(old_df):
            # Added row
            new_row = new_df.iloc[i]
            changes.append(
                RowChange(
                    key=i,
                    change_type=ChangeType.ADDED,
                    old_values=None,
                    new_values=_row_to_dict(new_row, common_cols),
                )
            )
        elif i >= len(new_df):
            # Removed row
            old_row = old_df.iloc[i]
            changes.append(
                RowChange(
                    key=i,
                    change_type=ChangeType.REMOVED,
                    old_values=_row_to_dict(old_row, common_cols),
                    new_values=None,
                )
            )
        else:
            # Check if modified
            old_row = old_df.iloc[i]
            new_row = new_df.iloc[i]
            old_dict = _row_to_dict(old_row, common_cols)
            new_dict = _row_to_dict(new_row, common_cols)
            if old_dict != new_dict:
                changes.append(
                    RowChange(
                        key=i,
                        change_type=ChangeType.MODIFIED,
                        old_values=old_dict,
                        new_values=new_dict,
                    )
                )

    return changes, truncated


def diff_data_files(
    old_path: pathlib.Path | None,
    new_path: pathlib.Path | None,
    path_display: str,
    key_columns: list[str] | None = None,
    max_rows: int = 10000,
) -> DataDiffResult:
    """Compare two data files and return detailed diff result."""
    # Handle file added/removed cases
    if old_path is None and new_path is not None:
        new_df = load_dataframe(new_path)
        new_cols = list(new_df.columns)
        new_dtypes = {col: str(dtype) for col, dtype in new_df.dtypes.items()}
        return DataDiffResult(
            path=path_display,
            old_rows=None,
            new_rows=len(new_df),
            old_cols=None,
            new_cols=new_cols,
            schema_changes=[
                SchemaChange(
                    column=col,
                    old_dtype=None,
                    new_dtype=new_dtypes[col],
                    change_type=ChangeType.ADDED,
                )
                for col in new_cols
            ],
            row_changes=[],
            reorder_only=False,
            truncated=False,
            summary_only=True,
        )

    if old_path is not None and new_path is None:
        old_df = load_dataframe(old_path)
        old_cols = list(old_df.columns)
        old_dtypes = {col: str(dtype) for col, dtype in old_df.dtypes.items()}
        return DataDiffResult(
            path=path_display,
            old_rows=len(old_df),
            new_rows=None,
            old_cols=old_cols,
            new_cols=None,
            schema_changes=[
                SchemaChange(
                    column=col,
                    old_dtype=old_dtypes[col],
                    new_dtype=None,
                    change_type=ChangeType.REMOVED,
                )
                for col in old_cols
            ],
            row_changes=[],
            reorder_only=False,
            truncated=False,
            summary_only=True,
        )

    if old_path is None or new_path is None:  # pragma: no cover
        raise DataError("Both old and new paths are None")

    # Check file sizes for large file handling - BEFORE loading
    old_size = old_path.stat().st_size
    new_size = new_path.stat().st_size
    total_size = old_size + new_size

    # For large files without key columns, return summary only without loading full data
    if total_size > _MAX_IN_MEMORY_SIZE and not key_columns:
        old_dtypes = get_schema(old_path)
        new_dtypes = get_schema(new_path)
        old_cols = list(old_dtypes.keys())
        new_cols = list(new_dtypes.keys())
        schema_changes = _diff_schema(old_cols, new_cols, old_dtypes, new_dtypes)
        return DataDiffResult(
            path=path_display,
            old_rows=get_row_count(old_path),
            new_rows=get_row_count(new_path),
            old_cols=old_cols,
            new_cols=new_cols,
            schema_changes=schema_changes,
            row_changes=[],
            reorder_only=False,
            truncated=False,
            summary_only=True,
        )

    # Load dataframes for full comparison
    old_df = load_dataframe(old_path)
    new_df = load_dataframe(new_path)

    old_cols = [str(col) for col in old_df.columns]
    new_cols = [str(col) for col in new_df.columns]
    old_dtypes = {str(col): str(dtype) for col, dtype in old_df.dtypes.items()}
    new_dtypes = {str(col): str(dtype) for col, dtype in new_df.dtypes.items()}

    # Validate key columns early (before any computation)
    if key_columns:
        missing_old = [k for k in key_columns if k not in old_cols]
        missing_new = [k for k in key_columns if k not in new_cols]
        if missing_old:
            raise DataError(f"Key column(s) not in old file: {missing_old}")
        if missing_new:
            raise DataError(f"Key column(s) not in new file: {missing_new}")

    # Compute schema changes
    schema_changes = _diff_schema(old_cols, new_cols, old_dtypes, new_dtypes)

    # Check for reorder-only case
    reorder_only = check_reorder_only(old_df, new_df)
    if reorder_only:
        return DataDiffResult(
            path=path_display,
            old_rows=len(old_df),
            new_rows=len(new_df),
            old_cols=old_cols,
            new_cols=new_cols,
            schema_changes=schema_changes,
            row_changes=[],
            reorder_only=True,
            truncated=False,
            summary_only=False,
        )

    # Compute row-level diff
    common_cols = [c for c in old_cols if c in new_cols]

    if key_columns:
        row_changes, truncated = _diff_rows_by_key(
            old_df, new_df, key_columns, common_cols, max_rows
        )
    else:
        row_changes, truncated = _diff_rows_positional(old_df, new_df, common_cols, max_rows)

    return DataDiffResult(
        path=path_display,
        old_rows=len(old_df),
        new_rows=len(new_df),
        old_cols=old_cols,
        new_cols=new_cols,
        schema_changes=schema_changes,
        row_changes=row_changes,
        reorder_only=False,
        truncated=truncated,
        summary_only=False,
    )


@dataclasses.dataclass(frozen=True)
class _DiffOutput:
    """A stage output `pivot diff` can compare, and whether it compares as a directory."""

    path: str
    head: types.HashInfo | None
    is_directory: bool


@dataclasses.dataclass(frozen=True)
class DataSelection:
    """The data files a diff's targets select: hashes at git HEAD, and in the workspace.

    ``workspace_paths`` holds the file to load for each workspace hash. A path absent from it
    is absent on the workspace side and is never read.
    """

    head_hashes: dict[str, str | None]
    workspace_hashes: dict[str, str]
    workspace_paths: dict[str, pathlib.Path]


def _parents(relpath: str) -> list[str]:
    """Every directory above ``relpath``, nearest first, short of the root."""
    return [str(parent) for parent in pathlib.PurePosixPath(relpath).parents[:-1]]


def _member_selected(relpath: str, prefixes: frozenset[str] | None) -> bool:
    """Whether a directory output's member is selected: every member, or one at or below a
    selected prefix."""
    return (
        prefixes is None
        or relpath in prefixes
        or any(parent in prefixes for parent in _parents(relpath))
    )


def _diff_outputs(
    stage_infos: Mapping[str, RegistryStageInfo],
    store: store_mod.WorkspaceStore,
    proj_root: pathlib.Path,
) -> dict[str, _DiffOutput]:
    """Map each comparable stage output's path to its HEAD hash and its kind.

    An output compares as a directory when the registry tags it DIRECTORY, its HEAD lock
    records a manifest, or it is a directory on disk. Any other output is comparable when its
    loader or extension is CSV, JSON or JSONL.
    """
    outs_by_stage = dict[str, list[types.ArtifactRef]]()
    for stage_name, stage_info in stage_infos.items():
        outs = [
            out
            for out in stage_info["outs"]
            if out.tag not in {types.ArtifactTag.METRIC, types.ArtifactTag.PLOT}
        ]
        if outs:
            outs_by_stage[stage_name] = outs

    lock_data_by_stage = common.read_lock_files_from_head(list(outs_by_stage))
    result = dict[str, _DiffOutput]()
    for stage_name, outs in outs_by_stage.items():
        lock_data = lock_data_by_stage[stage_name]
        entries = {} if lock_data is None else {entry["key"]: entry for entry in lock_data["outs"]}
        for out in outs:
            output_path = store.resolve_display_path(out)
            entry = entries.get(out.identity.key)
            head = None if entry is None else lock.entry_hash_info(entry)
            is_directory = (
                out.tag is types.ArtifactTag.DIRECTORY
                or (head is not None and types.is_dir_hash(head))
                or output_path.is_dir()
            )
            if is_directory or _is_data_output(out, store, proj_root):
                rel_path = project.to_relative_path(output_path, proj_root)
                result[rel_path] = _DiffOutput(rel_path, head, is_directory)
    return result


def _select_outputs(
    targets: Sequence[str],
    outputs: Mapping[str, _DiffOutput],
    stage_infos: Mapping[str, RegistryStageInfo],
    store: store_mod.WorkspaceStore,
    proj_root: pathlib.Path,
) -> dict[str, frozenset[str] | None]:
    """Map each selected output to its selected member prefixes, or None for every member.

    ``.`` selects every output, a stage its outputs, and a path the output at it, every output
    below it, or, inside a directory output, the members at or below it. A file that is no
    stage output is never compared: a target that selects no output is named in a warning.
    """
    from pivot.cli import targets as cli_targets

    below = collections.defaultdict[str, list[str]](list)
    for output_path in outputs:
        for parent in _parents(output_path):
            below[parent].append(output_path)

    selected = dict[str, set[str] | None]()
    unmatched = list[str]()
    for target in targets:
        resolved = cli_targets.resolve_cli_target(target, stage_infos)
        if resolved is not None:
            ref_paths = (
                project.to_relative_path(store.resolve_display_path(ref), proj_root)
                for ref in resolved["refs"]
            )
            hits = [ref_path for ref_path in ref_paths if ref_path in outputs]
        else:
            path = project.to_relative_path(project.normalize_path(target), proj_root)
            if path == ".":
                hits = list(outputs)
            elif path in outputs:
                hits = [path]
            else:
                hits = below.get(path, [])
            if not hits:
                container = next((parent for parent in _parents(path) if parent in outputs), None)
                if container is not None and outputs[container].is_directory:
                    prefixes = selected.setdefault(container, set[str]())
                    if prefixes is not None:
                        prefixes.add(path[len(container) + 1 :])
                    continue
        for hit in hits:
            selected[hit] = None
        if not hits:
            unmatched.append(target)

    if unmatched:
        logger.warning(
            "No CSV, JSON or JSONL stage output matches %s, so it is not compared",
            ", ".join(repr(target) for target in unmatched),
        )
    return {
        path: None if prefixes is None else frozenset(prefixes)
        for path, prefixes in selected.items()
    }


def _head_members(output: _DiffOutput, prefixes: frozenset[str] | None) -> dict[str, str | None]:
    """The selected output's comparable files and their hashes at git HEAD."""
    if not output.is_directory:
        return {output.path: None if output.head is None else output.head["hash"]}
    if output.head is None or not types.is_dir_hash(output.head):
        return {}
    return {
        f"{output.path}/{entry['relpath']}": entry["hash"]
        for entry in output.head["manifest"]
        if _member_selected(entry["relpath"], prefixes)
        and detect_format(pathlib.Path(entry["relpath"])) is not DataFileFormat.UNKNOWN
    }


def _workspace_members(
    output: _DiffOutput,
    prefixes: frozenset[str] | None,
    proj_root: pathlib.Path,
    cache_dir: pathlib.Path,
) -> dict[str, tuple[str, pathlib.Path]]:
    """The selected output's comparable workspace files, each with its hash and path.

    A directory output's members come from the record-time walker: a symlink it leaves out,
    and everything below one, is not a member and is never read. The warning names only the
    left-out symlinks inside the selected members, under the output's own path.
    """
    location = proj_root / output.path
    if not output.is_directory:
        if not location.exists():
            return {}
        file_hash, _ = cache.hash_file(location)
        return {output.path: (file_hash, location)}
    if not location.is_dir():
        return {}

    result = dict[str, tuple[str, pathlib.Path]]()
    left_out = list[pathlib.Path]()
    dirs = list[pathlib.Path]()
    for entry in cache.scandir_artifact_members(location, cache_dir, left_out, dirs):
        file_path = pathlib.Path(entry.path)
        relpath = file_path.relative_to(location).as_posix()
        if not _member_selected(relpath, prefixes):
            continue
        if detect_format(file_path) is DataFileFormat.UNKNOWN:
            continue
        file_hash, _ = cache.hash_file(file_path)
        if not cache.is_artifact_member_content(file_path, file_hash, cache_dir):
            left_out.append(file_path)
            continue
        result[f"{output.path}/{relpath}"] = (file_hash, file_path)

    selected_parents = set[str]()
    if prefixes is not None:
        selected_parents = {parent for prefix in prefixes for parent in _parents(prefix)}
    named = list[pathlib.Path]()
    for link in left_out:
        link_relpath = link.relative_to(location).as_posix()
        if _member_selected(link_relpath, prefixes) or link_relpath in selected_parents:
            named.append(link)
    if named:
        cache.log_left_out_symlinks(location, named)
    return result


def select_data_files(targets: Sequence[str]) -> DataSelection | None:
    """Select the data files ``targets`` name, at git HEAD and in the workspace.

    One selection serves both sides (see ``_select_outputs``), and one walk per selected
    directory output builds its workspace members. Returns None when no registered stage has
    an output `pivot diff` can compare.
    """
    from pivot.cli import helpers as cli_helpers

    store = cli_helpers.require_workspace_store()
    proj_root = project.get_project_root()
    stage_infos = {name: cli_helpers.get_stage(name) for name in cli_helpers.list_stages()}
    outputs = _diff_outputs(stage_infos, store, proj_root)
    if not outputs:
        return None

    head_hashes = dict[str, str | None]()
    workspace_hashes = dict[str, str]()
    workspace_paths = dict[str, pathlib.Path]()
    cache_dir = config.get_cache_dir() / "files"
    for path, prefixes in _select_outputs(targets, outputs, stage_infos, store, proj_root).items():
        output = outputs[path]
        head_hashes.update(_head_members(output, prefixes))
        for member, (file_hash, file_path) in _workspace_members(
            output, prefixes, proj_root, cache_dir
        ).items():
            workspace_hashes[member] = file_hash
            workspace_paths[member] = file_path
    return DataSelection(head_hashes, workspace_hashes, workspace_paths)


def diff_data_hashes(
    old_hashes: Mapping[str, str | None],
    new_hashes: Mapping[str, str | None],
) -> list[DataDiffEntry]:
    """Compare data file hashes between old (HEAD) and new (workspace)."""
    diffs = list[DataDiffEntry]()
    all_paths = set(old_hashes.keys()) | set(new_hashes.keys())

    for path in sorted(all_paths):
        old_hash = old_hashes.get(path)
        new_hash = new_hashes.get(path)

        if old_hash is None and new_hash is not None:
            diffs.append(
                DataDiffEntry(
                    path=path, old_hash=None, new_hash=new_hash, change_type=ChangeType.ADDED
                )
            )
        elif old_hash is not None and new_hash is None:
            diffs.append(
                DataDiffEntry(
                    path=path, old_hash=old_hash, new_hash=None, change_type=ChangeType.REMOVED
                )
            )
        elif old_hash != new_hash:
            diffs.append(
                DataDiffEntry(
                    path=path, old_hash=old_hash, new_hash=new_hash, change_type=ChangeType.MODIFIED
                )
            )

    return diffs


def restore_data_from_cache(
    rel_path: str,
    file_hash: str,
) -> pathlib.Path | None:
    """Restore data file from cache to a temp location. Returns temp path or None.

    Note: Caller is responsible for cleaning up the returned temp file.
    """
    import tempfile

    # Validate hash format to prevent path traversal attacks
    if not _HASH_PATTERN.fullmatch(file_hash.lower()):
        return None

    cache_dir = config.get_cache_dir() / "files"
    cached_path = cache_dir / file_hash[:2] / file_hash[2:]

    # Prevent symlink escape attacks
    if cached_path.is_symlink():
        return None

    if not cached_path.exists():
        return None

    # Create temp file with same extension
    suffix = pathlib.Path(rel_path).suffix
    fd, temp_path_str = tempfile.mkstemp(suffix=suffix)
    temp_path = pathlib.Path(temp_path_str)

    try:
        # Set restrictive permissions before writing any data
        os.fchmod(fd, 0o600)
        with open(fd, "wb") as f:
            f.write(cached_path.read_bytes())
        return temp_path
    except OSError:  # pragma: no cover
        os.close(fd)
        temp_path.unlink(missing_ok=True)
        return None


def format_diff_summary(result: DataDiffResult) -> str:
    """Format a single diff result as human-readable summary."""
    lines = list[str]()

    # Header
    if result["reorder_only"]:
        lines.append(f"{result['path']}: REORDER ONLY")
        lines.append("  Same content, different row order")
    else:
        lines.append(f"{result['path']}:")

    # Row counts
    old_rows = result["old_rows"]
    new_rows = result["new_rows"]
    if old_rows is not None and new_rows is not None:
        diff = new_rows - old_rows
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        lines.append(f"  Rows: {old_rows:,} -> {new_rows:,} ({diff_str})")
    elif old_rows is not None:
        lines.append(f"  Rows: {old_rows:,} (removed)")
    elif new_rows is not None:
        lines.append(f"  Rows: {new_rows:,} (added)")

    # Column counts
    old_cols = result["old_cols"]
    new_cols = result["new_cols"]
    if old_cols is not None and new_cols is not None:
        added = len([c for c in new_cols if c not in old_cols])
        removed = len([c for c in old_cols if c not in new_cols])
        lines.append(
            f"  Columns: {len(old_cols)} -> {len(new_cols)} (+{added} added, -{removed} removed)"
        )

    # Schema changes
    if result["schema_changes"]:
        lines.append("  Schema changes:")
        for change in result["schema_changes"]:
            symbol = STATUS_SYMBOLS[change["change_type"]]
            if change["change_type"] == ChangeType.ADDED:
                lines.append(f"    {symbol} {change['column']}: {change['new_dtype']}")
            elif change["change_type"] == ChangeType.REMOVED:
                lines.append(f"    {symbol} {change['column']}: {change['old_dtype']}")
            else:
                lines.append(
                    f"    {symbol} {change['column']}: {change['old_dtype']} -> {change['new_dtype']}"
                )

    # Row change summary
    if result["row_changes"]:
        added = sum(1 for r in result["row_changes"] if r["change_type"] == ChangeType.ADDED)
        removed = sum(1 for r in result["row_changes"] if r["change_type"] == ChangeType.REMOVED)
        modified = sum(1 for r in result["row_changes"] if r["change_type"] == ChangeType.MODIFIED)
        parts = list[str]()
        if added:
            parts.append(f"+{added} added")
        if removed:
            parts.append(f"-{removed} removed")
        if modified:
            parts.append(f"~{modified} modified")
        if result["truncated"]:
            parts.append("(truncated)")
        lines.append(f"  Row changes: {', '.join(parts)}")

    return "\n".join(lines)


def format_diff_table(
    diffs: list[DataDiffResult],
    output_format: OutputFormat | None,
) -> str:
    """Format diff results for display."""
    if output_format == OutputFormat.JSON:
        return json.dumps(diffs, indent=2, cls=_NumpyEncoder)

    if not diffs:
        return "No data file changes."

    # For plain/md output, show summaries
    outputs_list = list[str]()
    for diff in diffs:
        outputs_list.append(format_diff_summary(diff))

    return "\n\n".join(outputs_list)


def format_row_changes_table(
    row_changes: list[RowChange],
    columns: list[str],
    output_format: OutputFormat | None,
) -> str:
    """Format row changes as a table."""
    if output_format == OutputFormat.JSON:
        return json.dumps(row_changes, indent=2, cls=_NumpyEncoder)

    if not row_changes:
        return "No row changes."

    # Build table rows
    rows = list[list[str]]()
    for change in row_changes:
        change_char = STATUS_SYMBOLS[change["change_type"]]
        row = [change_char, str(change["key"])]

        values = (
            change["new_values"]
            if change["change_type"] != ChangeType.REMOVED
            else change["old_values"]
        )
        if values:
            for col in columns:
                val = values.get(col, "")
                row.append(str(val) if val is not None else "")
        rows.append(row)

    headers = ["", "Key"] + columns
    tablefmt = "github" if output_format == "md" else "plain"
    return tabulate.tabulate(rows, headers=headers, tablefmt=tablefmt, disable_numparse=True)
