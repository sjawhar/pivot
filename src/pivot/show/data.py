from __future__ import annotations

import collections
import csv
import json
import logging
import os
import pathlib
import re
from typing import TYPE_CHECKING, Any, TypedDict, cast, override

import numpy
import pandas
import tabulate

from pivot import outputs, project
from pivot.show import common
from pivot.storage import cache
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


def load_dataframe(path: pathlib.Path) -> pandas.DataFrame:
    """Load data file as pandas DataFrame."""
    match detect_format(path):
        case DataFileFormat.CSV:
            return pandas.read_csv(path)  # pyright: ignore[reportUnknownMemberType] - pandas-stubs incomplete
        case DataFileFormat.JSON:
            return pandas.read_json(path)
        case DataFileFormat.JSONL:
            return pandas.read_json(path, lines=True)
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
            df = pandas.read_json(path, lines=True, nrows=100)
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
            with open(path) as f:
                return sum(1 for _ in f)
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


def get_data_outputs_from_registry() -> dict[str, str]:
    """Get all data outputs (Out, not Metric/Plot) from registry.

    Returns dict mapping stage_name -> relative_path for data outputs.
    """
    from pivot import registry

    result = dict[str, str]()
    proj_root = project.get_project_root()

    for stage_name in registry.REGISTRY.list_stages():
        info = registry.REGISTRY.get(stage_name)
        for out in info["outs"]:
            # Only include Out types that are data files (not Metric, Plot, etc.)
            if isinstance(out, outputs.Out) and not isinstance(out, (outputs.Metric, outputs.Plot)):
                abs_path = str(project.normalize_path(out.path))
                rel_path = project.to_relative_path(abs_path, proj_root)
                # Check if it's a supported data format
                fmt = detect_format(pathlib.Path(rel_path))
                if fmt != DataFileFormat.UNKNOWN:
                    result[stage_name] = rel_path

    return result


def get_data_hashes_from_head() -> dict[str, str | None]:
    """Read data output hashes from lock files at git HEAD.

    Returns relative paths mapping to hashes (or None if no hash).
    """
    from pivot import registry

    result = dict[str, str | None]()
    proj_root = project.get_project_root()

    # Collect lock file paths and data paths per stage
    stage_data_paths: dict[str, list[str]] = {}
    for stage_name in registry.REGISTRY.list_stages():
        info = registry.REGISTRY.get(stage_name)
        for out in info["outs"]:
            if isinstance(out, outputs.Out) and not isinstance(out, (outputs.Metric, outputs.Plot)):
                abs_path = str(project.normalize_path(out.path))
                rel_path = project.to_relative_path(abs_path, proj_root)
                fmt = detect_format(pathlib.Path(rel_path))
                if fmt != DataFileFormat.UNKNOWN:
                    if stage_name not in stage_data_paths:
                        stage_data_paths[stage_name] = []
                    stage_data_paths[stage_name].append(rel_path)
                    result[rel_path] = None

    # Read all lock files from HEAD in one batch
    lock_data_map = common.read_lock_files_from_head(list(stage_data_paths.keys()))

    # Extract hashes from lock files
    for stage_name, data_paths in stage_data_paths.items():
        lock_data = lock_data_map.get(stage_name)
        if lock_data is None:
            continue

        path_to_hash = common.extract_output_hashes_from_lock(lock_data)

        for data_rel_path in data_paths:
            if data_rel_path in path_to_hash:
                result[data_rel_path] = path_to_hash[data_rel_path]

    return result


def get_data_hashes_from_workspace(paths: Sequence[str]) -> dict[str, str | None]:
    """Compute hashes for data files in workspace."""
    result = dict[str, str | None]()
    proj_root = project.get_project_root()

    for rel_path in paths:
        abs_path = proj_root / rel_path
        if abs_path.exists():
            result[rel_path] = cache.hash_file(abs_path)
        else:
            result[rel_path] = None

    return result


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

    proj_root = project.get_project_root()
    cache_dir = proj_root / ".pivot" / "cache" / "files"
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
