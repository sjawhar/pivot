from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas
import pytest

from pivot.show import data
from pivot.types import (
    ChangeType,
    DataDiffResult,
    DataFileFormat,
    OutputFormat,
    RowChange,
    SchemaChange,
)

if TYPE_CHECKING:
    from pathlib import Path

# =============================================================================
# Format Detection Tests
# =============================================================================


@pytest.mark.parametrize(
    ("filename", "expected_format"),
    [
        pytest.param("data.csv", DataFileFormat.CSV, id="csv"),
        pytest.param("data.json", DataFileFormat.JSON, id="json"),
        pytest.param("data.jsonl", DataFileFormat.JSONL, id="jsonl"),
        pytest.param("data.parquet", DataFileFormat.UNKNOWN, id="unknown"),
        pytest.param("data.CSV", DataFileFormat.CSV, id="csv_uppercase"),
    ],
)
def test_detect_format(tmp_path: Path, filename: str, expected_format: DataFileFormat) -> None:
    """Format detection correctly identifies file types."""
    path = tmp_path / filename
    assert data.detect_format(path) == expected_format


# =============================================================================
# DataFrame Loading Tests
# =============================================================================


def test_load_dataframe_csv(tmp_path: Path) -> None:
    """Load CSV file as DataFrame."""
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("id,name,value\n1,alice,10\n2,bob,20\n")

    df = data.load_dataframe(csv_file)

    assert len(df) == 2
    assert list(df.columns) == ["id", "name", "value"]
    assert df.iloc[0]["name"] == "alice"


def test_load_dataframe_json(tmp_path: Path) -> None:
    """Load JSON file as DataFrame."""
    json_file = tmp_path / "data.json"
    json_file.write_text(json.dumps([{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}]))

    df = data.load_dataframe(json_file)

    assert len(df) == 2
    assert "id" in df.columns
    assert "name" in df.columns


def test_load_dataframe_jsonl(tmp_path: Path) -> None:
    """Load JSONL file as DataFrame."""
    jsonl_file = tmp_path / "data.jsonl"
    jsonl_file.write_text('{"id": 1, "name": "alice"}\n{"id": 2, "name": "bob"}\n')

    df = data.load_dataframe(jsonl_file)

    assert len(df) == 2
    assert df.iloc[0]["name"] == "alice"


def test_load_dataframe_unsupported(tmp_path: Path) -> None:
    """Unsupported format raises DataError."""
    file = tmp_path / "data.parquet"
    file.write_bytes(b"fake parquet data")

    with pytest.raises(data.DataError, match="Unsupported data format"):
        data.load_dataframe(file)


# =============================================================================
# Schema Tests
# =============================================================================


def test_get_schema_csv(tmp_path: Path) -> None:
    """Get schema from CSV file."""
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("id,name,value\n1,alice,10.5\n2,bob,20.5\n")

    schema = data.get_schema(csv_file)

    assert "id" in schema
    assert "name" in schema
    assert "value" in schema


def test_get_schema_json(tmp_path: Path) -> None:
    """Get schema from JSON file."""
    json_file = tmp_path / "data.json"
    json_file.write_text(json.dumps([{"id": 1, "name": "alice"}]))

    schema = data.get_schema(json_file)

    assert "id" in schema
    assert "name" in schema


def test_get_row_count_csv(tmp_path: Path) -> None:
    """Get row count from CSV without loading all data."""
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("id,name\n1,alice\n2,bob\n3,charlie\n")

    count = data.get_row_count(csv_file)

    assert count == 3


def test_get_row_count_jsonl(tmp_path: Path) -> None:
    """Get row count from JSONL."""
    jsonl_file = tmp_path / "data.jsonl"
    jsonl_file.write_text('{"id": 1}\n{"id": 2}\n')

    count = data.get_row_count(jsonl_file)

    assert count == 2


# =============================================================================
# Reorder Detection Tests
# =============================================================================


def test_check_reorder_only_same_order() -> None:
    """Same data same order is not reorder-only."""
    old_df = pandas.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    new_df = pandas.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})

    result = data.check_reorder_only(old_df, new_df)

    assert result is True, "Identical dataframes should be detected"


def test_check_reorder_only_different_order() -> None:
    """Same data different order is reorder-only."""
    old_df = pandas.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    new_df = pandas.DataFrame({"id": [3, 1, 2], "name": ["c", "a", "b"]})

    result = data.check_reorder_only(old_df, new_df)

    assert result is True, "Reordered dataframes should be detected"


def test_check_reorder_only_different_content() -> None:
    """Different content is not reorder-only."""
    old_df = pandas.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    new_df = pandas.DataFrame({"id": [1, 2, 4], "name": ["a", "b", "d"]})

    result = data.check_reorder_only(old_df, new_df)

    assert result is False, "Different content should not be reorder-only"


def test_check_reorder_only_different_shape() -> None:
    """Different shape is not reorder-only."""
    old_df = pandas.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    new_df = pandas.DataFrame({"id": [1, 2], "name": ["a", "b"]})

    result = data.check_reorder_only(old_df, new_df)

    assert result is False, "Different row count should not be reorder-only"


def test_check_reorder_only_different_columns() -> None:
    """Different columns is not reorder-only."""
    old_df = pandas.DataFrame({"id": [1, 2], "name": ["a", "b"]})
    new_df = pandas.DataFrame({"id": [1, 2], "value": [10, 20]})

    result = data.check_reorder_only(old_df, new_df)

    assert result is False, "Different columns should not be reorder-only"


# =============================================================================
# Schema Diff Tests
# =============================================================================


def test_diff_schema_no_changes() -> None:
    """No schema changes returns empty list."""
    old_cols = ["id", "name"]
    new_cols = ["id", "name"]
    old_dtypes = {"id": "int64", "name": "object"}
    new_dtypes = {"id": "int64", "name": "object"}

    changes = data._diff_schema(old_cols, new_cols, old_dtypes, new_dtypes)

    assert changes == []


@pytest.mark.parametrize(
    (
        "old_cols",
        "new_cols",
        "old_dtypes",
        "new_dtypes",
        "expected_col",
        "expected_type",
        "expected_old_dtype",
        "expected_new_dtype",
    ),
    [
        pytest.param(
            ["id", "name"],
            ["id", "name", "value"],
            {"id": "int64", "name": "object"},
            {"id": "int64", "name": "object", "value": "float64"},
            "value",
            "added",
            None,
            "float64",
            id="added_column",
        ),
        pytest.param(
            ["id", "name", "value"],
            ["id", "name"],
            {"id": "int64", "name": "object", "value": "float64"},
            {"id": "int64", "name": "object"},
            "value",
            "removed",
            "float64",
            None,
            id="removed_column",
        ),
        pytest.param(
            ["id", "value"],
            ["id", "value"],
            {"id": "int64", "value": "int64"},
            {"id": "int64", "value": "float64"},
            "value",
            "modified",
            "int64",
            "float64",
            id="modified_dtype",
        ),
    ],
)
def test_diff_schema_change(
    old_cols: list[str],
    new_cols: list[str],
    old_dtypes: dict[str, str],
    new_dtypes: dict[str, str],
    expected_col: str,
    expected_type: str,
    expected_old_dtype: str | None,
    expected_new_dtype: str | None,
) -> None:
    """_diff_schema detects added/removed/modified columns."""
    changes = data._diff_schema(old_cols, new_cols, old_dtypes, new_dtypes)
    assert len(changes) == 1
    assert changes[0]["column"] == expected_col
    assert changes[0]["change_type"] == expected_type
    assert changes[0]["old_dtype"] == expected_old_dtype
    assert changes[0]["new_dtype"] == expected_new_dtype


# =============================================================================
# Row Diff Tests (Key-based)
# =============================================================================


def test_diff_rows_by_key_no_changes() -> None:
    """No row changes with key matching."""
    old_df = pandas.DataFrame({"id": [1, 2], "name": ["alice", "bob"]})
    new_df = pandas.DataFrame({"id": [1, 2], "name": ["alice", "bob"]})

    changes, truncated = data._diff_rows_by_key(
        old_df, new_df, key_columns=["id"], common_cols=["id", "name"], max_rows=100
    )

    assert changes == []
    assert truncated is False


def test_diff_rows_by_key_added() -> None:
    """Added row detected with key matching."""
    old_df = pandas.DataFrame({"id": [1, 2], "name": ["alice", "bob"]})
    new_df = pandas.DataFrame({"id": [1, 2, 3], "name": ["alice", "bob", "charlie"]})

    changes, truncated = data._diff_rows_by_key(
        old_df, new_df, key_columns=["id"], common_cols=["id", "name"], max_rows=100
    )

    assert len(changes) == 1
    assert changes[0]["change_type"] == ChangeType.ADDED
    # Key is JSON-serialized (numpy int64 -> str via default=str)
    assert changes[0]["key"] == '["3"]'


def test_diff_rows_by_key_removed() -> None:
    """Removed row detected with key matching."""
    old_df = pandas.DataFrame({"id": [1, 2, 3], "name": ["alice", "bob", "charlie"]})
    new_df = pandas.DataFrame({"id": [1, 2], "name": ["alice", "bob"]})

    changes, truncated = data._diff_rows_by_key(
        old_df, new_df, key_columns=["id"], common_cols=["id", "name"], max_rows=100
    )

    assert len(changes) == 1
    assert changes[0]["change_type"] == ChangeType.REMOVED
    # Key is JSON-serialized (numpy int64 -> str via default=str)
    assert changes[0]["key"] == '["3"]'


def test_diff_rows_by_key_modified() -> None:
    """Modified row detected with key matching."""
    old_df = pandas.DataFrame({"id": [1, 2], "name": ["alice", "bob"]})
    new_df = pandas.DataFrame({"id": [1, 2], "name": ["alice", "bobby"]})

    changes, truncated = data._diff_rows_by_key(
        old_df, new_df, key_columns=["id"], common_cols=["id", "name"], max_rows=100
    )

    assert len(changes) == 1
    assert changes[0]["change_type"] == "modified"
    assert changes[0]["old_values"] is not None
    assert changes[0]["new_values"] is not None
    assert changes[0]["old_values"]["name"] == "bob"
    assert changes[0]["new_values"]["name"] == "bobby"


def test_diff_rows_by_key_truncated() -> None:
    """Truncation when max_rows exceeded."""
    old_df = pandas.DataFrame({"id": range(100), "value": range(100)})
    new_df = pandas.DataFrame({"id": range(100, 200), "value": range(100, 200)})

    changes, truncated = data._diff_rows_by_key(
        old_df, new_df, key_columns=["id"], common_cols=["id", "value"], max_rows=10
    )

    assert len(changes) == 10
    assert truncated is True


def test_diff_rows_by_key_duplicate_old() -> None:
    """Error when old file has duplicate keys."""
    old_df = pandas.DataFrame({"id": [1, 1, 2], "name": ["a", "b", "c"]})
    new_df = pandas.DataFrame({"id": [1, 2], "name": ["a", "c"]})

    with pytest.raises(data.DataError, match="Duplicate key.*old file"):
        data._diff_rows_by_key(
            old_df, new_df, key_columns=["id"], common_cols=["id", "name"], max_rows=100
        )


def test_diff_rows_by_key_duplicate_new() -> None:
    """Error when new file has duplicate keys."""
    old_df = pandas.DataFrame({"id": [1, 2], "name": ["a", "c"]})
    new_df = pandas.DataFrame({"id": [1, 1, 2], "name": ["a", "b", "c"]})

    with pytest.raises(data.DataError, match="Duplicate key.*new file"):
        data._diff_rows_by_key(
            old_df, new_df, key_columns=["id"], common_cols=["id", "name"], max_rows=100
        )


def test_row_to_dict_with_nan() -> None:
    """NaN values converted to None in row dict."""
    import numpy

    row = pandas.Series({"id": 1, "value": numpy.nan, "name": "test"})
    result = data._row_to_dict(row, ["id", "value", "name"])

    assert result["id"] == 1
    assert result["value"] is None
    assert result["name"] == "test"


# =============================================================================
# Row Diff Tests (Positional)
# =============================================================================


def test_diff_rows_positional_no_changes() -> None:
    """No row changes with positional matching."""
    old_df = pandas.DataFrame({"id": [1, 2], "name": ["alice", "bob"]})
    new_df = pandas.DataFrame({"id": [1, 2], "name": ["alice", "bob"]})

    changes, truncated = data._diff_rows_positional(
        old_df, new_df, common_cols=["id", "name"], max_rows=100
    )

    assert changes == []
    assert truncated is False


def test_diff_rows_positional_added() -> None:
    """Added rows detected with positional matching."""
    old_df = pandas.DataFrame({"id": [1, 2], "name": ["alice", "bob"]})
    new_df = pandas.DataFrame({"id": [1, 2, 3], "name": ["alice", "bob", "charlie"]})

    changes, truncated = data._diff_rows_positional(
        old_df, new_df, common_cols=["id", "name"], max_rows=100
    )

    assert len(changes) == 1
    assert changes[0]["change_type"] == "added"
    assert changes[0]["key"] == 2  # Row index


def test_diff_rows_positional_removed() -> None:
    """Removed rows detected with positional matching."""
    old_df = pandas.DataFrame({"id": [1, 2, 3], "name": ["alice", "bob", "charlie"]})
    new_df = pandas.DataFrame({"id": [1, 2], "name": ["alice", "bob"]})

    changes, truncated = data._diff_rows_positional(
        old_df, new_df, common_cols=["id", "name"], max_rows=100
    )

    assert len(changes) == 1
    assert changes[0]["change_type"] == "removed"
    assert changes[0]["key"] == 2


def test_diff_rows_positional_modified() -> None:
    """Modified rows detected with positional matching."""
    old_df = pandas.DataFrame({"id": [1, 2], "name": ["alice", "bob"]})
    new_df = pandas.DataFrame({"id": [1, 2], "name": ["alice", "bobby"]})

    changes, truncated = data._diff_rows_positional(
        old_df, new_df, common_cols=["id", "name"], max_rows=100
    )

    assert len(changes) == 1
    assert changes[0]["change_type"] == "modified"


def test_diff_rows_positional_truncated() -> None:
    """Truncation when max_rows exceeded in positional diff."""
    old_df = pandas.DataFrame({"id": range(100), "value": range(100)})
    new_df = pandas.DataFrame({"id": range(100, 200), "value": range(100, 200)})

    changes, truncated = data._diff_rows_positional(
        old_df, new_df, common_cols=["id", "value"], max_rows=10
    )

    assert len(changes) == 10
    assert truncated is True


# =============================================================================
# Full Diff Tests
# =============================================================================


def test_diff_data_files_new_file(tmp_path: Path) -> None:
    """Diff with new file (old is None)."""
    new_file = tmp_path / "data.csv"
    new_file.write_text("id,name\n1,alice\n2,bob\n")

    result = data.diff_data_files(None, new_file, "data.csv")

    assert result["old_rows"] is None
    assert result["new_rows"] == 2
    assert result["old_cols"] is None
    assert result["new_cols"] is not None
    assert len(result["new_cols"]) == 2
    assert result["summary_only"] is True


def test_diff_data_files_deleted_file(tmp_path: Path) -> None:
    """Diff with deleted file (new is None)."""
    old_file = tmp_path / "data.csv"
    old_file.write_text("id,name\n1,alice\n2,bob\n")

    result = data.diff_data_files(old_file, None, "data.csv")

    assert result["old_rows"] == 2
    assert result["new_rows"] is None
    assert result["summary_only"] is True


def test_diff_data_files_reorder_only(tmp_path: Path) -> None:
    """Detect reorder-only diff."""
    old_file = tmp_path / "old.csv"
    new_file = tmp_path / "new.csv"
    old_file.write_text("id,name\n1,alice\n2,bob\n3,charlie\n")
    new_file.write_text("id,name\n3,charlie\n1,alice\n2,bob\n")

    result = data.diff_data_files(old_file, new_file, "data.csv")

    assert result["reorder_only"] is True
    assert result["row_changes"] == []


def test_diff_data_files_with_key(tmp_path: Path) -> None:
    """Diff using key columns."""
    old_file = tmp_path / "old.csv"
    new_file = tmp_path / "new.csv"
    old_file.write_text("id,name\n1,alice\n2,bob\n")
    new_file.write_text("id,name\n1,alice\n2,bobby\n")

    result = data.diff_data_files(old_file, new_file, "data.csv", key_columns=["id"])

    assert result["reorder_only"] is False
    assert len(result["row_changes"]) == 1
    assert result["row_changes"][0]["change_type"] == "modified"


def test_diff_data_files_missing_key_column_old(tmp_path: Path) -> None:
    """Error when key column missing in old file."""
    old_file = tmp_path / "old.csv"
    new_file = tmp_path / "new.csv"
    old_file.write_text("id,name\n1,alice\n")
    new_file.write_text("id,name,extra\n1,alice,x\n")

    with pytest.raises(data.DataError, match="Key column.*not in old file"):
        data.diff_data_files(old_file, new_file, "data.csv", key_columns=["extra"])


def test_diff_data_files_missing_key_column_new(tmp_path: Path) -> None:
    """Error when key column missing in new file."""
    old_file = tmp_path / "old.csv"
    new_file = tmp_path / "new.csv"
    old_file.write_text("id,name,extra\n1,alice,x\n")
    new_file.write_text("id,name\n1,alice\n")

    with pytest.raises(data.DataError, match="Key column.*not in new file"):
        data.diff_data_files(old_file, new_file, "data.csv", key_columns=["extra"])


def test_diff_data_files_large_file_summary_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Large files without key columns return summary-only result."""
    old_file = tmp_path / "old.csv"
    new_file = tmp_path / "new.csv"
    old_file.write_text("id,name\n1,alice\n2,bob\n")
    new_file.write_text("id,name,age\n1,alice,30\n")

    # Monkeypatch the size threshold to trigger large file path
    monkeypatch.setattr(data, "_MAX_IN_MEMORY_SIZE", 1)

    result = data.diff_data_files(old_file, new_file, "data.csv", key_columns=None)

    assert result["summary_only"] is True
    assert result["row_changes"] == []
    assert result["old_rows"] == 2
    assert result["new_rows"] == 1
    assert result["old_cols"] == ["id", "name"]
    assert result["new_cols"] == ["id", "name", "age"]
    assert len(result["schema_changes"]) == 1
    assert result["schema_changes"][0]["column"] == "age"
    assert result["schema_changes"][0]["change_type"] == ChangeType.ADDED


def test_diff_data_files_positional_diff(tmp_path: Path) -> None:
    """Positional diff when no key columns specified."""
    old_file = tmp_path / "old.csv"
    new_file = tmp_path / "new.csv"
    old_file.write_text("id,name\n1,alice\n2,bob\n")
    new_file.write_text("id,name\n1,alice\n2,bobby\n3,carol\n")

    result = data.diff_data_files(old_file, new_file, "data.csv", key_columns=None)

    assert result["reorder_only"] is False
    assert result["summary_only"] is False
    # Row 2 modified (bob -> bobby), row 3 added
    assert len(result["row_changes"]) == 2
    changes_by_key = {c["key"]: c for c in result["row_changes"]}
    assert changes_by_key[1]["change_type"] == ChangeType.MODIFIED
    assert changes_by_key[2]["change_type"] == ChangeType.ADDED


def test_diff_data_files_positional_truncation(tmp_path: Path) -> None:
    """Positional diff truncates at max_rows."""
    old_file = tmp_path / "old.csv"
    new_file = tmp_path / "new.csv"
    # Create files with many different rows
    old_file.write_text("id,name\n1,a\n2,b\n3,c\n4,d\n5,e\n")
    new_file.write_text("id,name\n1,x\n2,y\n3,z\n4,w\n5,v\n")

    result = data.diff_data_files(old_file, new_file, "data.csv", key_columns=None, max_rows=2)

    assert result["truncated"] is True
    # Should only have 2 changes due to max_rows
    assert len(result["row_changes"]) == 2


# =============================================================================
# Hash Diff Tests
# =============================================================================


def test_diff_data_hashes_no_changes() -> None:
    """No changes when hashes match."""
    old = {"data.csv": "abc123"}
    new = {"data.csv": "abc123"}

    result = data.diff_data_hashes(old, new)

    assert result == []


@pytest.mark.parametrize(
    ("old_hashes", "new_hashes", "expected_change_type", "expected_old", "expected_new"),
    [
        pytest.param(
            {"data.csv": "abc123"},
            {"data.csv": "def456"},
            "modified",
            "abc123",
            "def456",
            id="modified",
        ),
        pytest.param({}, {"data.csv": "abc123"}, "added", None, "abc123", id="added"),
        pytest.param({"data.csv": "abc123"}, {}, "removed", "abc123", None, id="removed"),
    ],
)
def test_diff_data_hashes_change(
    old_hashes: dict[str, str | None],
    new_hashes: dict[str, str | None],
    expected_change_type: str,
    expected_old: str | None,
    expected_new: str | None,
) -> None:
    """diff_data_hashes detects added/modified/removed files."""
    result = data.diff_data_hashes(old_hashes, new_hashes)
    assert len(result) == 1
    assert result[0]["change_type"] == expected_change_type
    if expected_old is not None:
        assert result[0]["old_hash"] == expected_old
    if expected_new is not None:
        assert result[0]["new_hash"] == expected_new


# =============================================================================
# Workspace Hash Tests
# =============================================================================


def test_get_data_hashes_from_workspace(
    tmp_path: Path,
    set_project_root: Path,
) -> None:
    """Get hashes from workspace files."""
    data_file = tmp_path / "data.csv"
    data_file.write_text("id,name\n1,alice\n")

    result = data.get_data_hashes_from_workspace(["data.csv"])

    assert "data.csv" in result
    assert result["data.csv"] is not None
    assert len(result["data.csv"]) == 16  # xxhash64 produces 16 hex chars


def test_get_data_hashes_from_workspace_missing_file(
    tmp_path: Path,
    set_project_root: Path,
) -> None:
    """Missing file returns None hash."""
    result = data.get_data_hashes_from_workspace(["nonexistent.csv"])

    assert "nonexistent.csv" in result
    assert result["nonexistent.csv"] is None


# =============================================================================
# Formatting Tests
# =============================================================================


def test_format_diff_summary_basic(tmp_path: Path) -> None:
    """Format basic diff summary."""
    result = DataDiffResult(
        path="data.csv",
        old_rows=100,
        new_rows=110,
        old_cols=["id", "name"],
        new_cols=["id", "name", "value"],
        schema_changes=[
            SchemaChange(
                column="value", old_dtype=None, new_dtype="float64", change_type=ChangeType.ADDED
            )
        ],
        row_changes=[],
        reorder_only=False,
        truncated=False,
        summary_only=True,
    )

    output = data.format_diff_summary(result)

    assert "data.csv" in output
    assert "100" in output
    assert "110" in output
    assert "value" in output
    assert "float64" in output


def test_format_diff_summary_reorder_only() -> None:
    """Format reorder-only summary."""
    result = DataDiffResult(
        path="data.csv",
        old_rows=100,
        new_rows=100,
        old_cols=["id", "name"],
        new_cols=["id", "name"],
        schema_changes=[],
        row_changes=[],
        reorder_only=True,
        truncated=False,
        summary_only=False,
    )

    output = data.format_diff_summary(result)

    assert "REORDER ONLY" in output
    assert "Same content, different row order" in output


def test_format_diff_table_json() -> None:
    """JSON output format."""
    diffs = [
        DataDiffResult(
            path="data.csv",
            old_rows=10,
            new_rows=10,
            old_cols=["id"],
            new_cols=["id"],
            schema_changes=[],
            row_changes=[],
            reorder_only=True,
            truncated=False,
            summary_only=False,
        )
    ]

    output = data.format_diff_table(diffs, OutputFormat.JSON)

    parsed = json.loads(output)
    assert len(parsed) == 1
    assert parsed[0]["reorder_only"] is True


def test_format_diff_table_empty() -> None:
    """Empty diff list shows no changes."""
    output = data.format_diff_table([], None)
    assert "No data file changes" in output


def test_format_diff_table_plain_output() -> None:
    """Plain output shows summaries for each diff."""
    diffs = [
        DataDiffResult(
            path="data.csv",
            old_rows=100,
            new_rows=150,
            old_cols=["id", "name"],
            new_cols=["id", "name", "age"],
            schema_changes=[],
            row_changes=[],
            reorder_only=False,
            truncated=False,
            summary_only=False,
        ),
        DataDiffResult(
            path="train.csv",
            old_rows=50,
            new_rows=50,
            old_cols=["x", "y"],
            new_cols=["x", "y"],
            schema_changes=[],
            row_changes=[],
            reorder_only=True,
            truncated=False,
            summary_only=False,
        ),
    ]

    output = data.format_diff_table(diffs, None)

    # Should contain summaries for both files
    assert "data.csv" in output
    assert "train.csv" in output
    assert "100" in output
    assert "150" in output
    assert "REORDER ONLY" in output


def test_format_row_changes_table() -> None:
    """Format row changes as table."""
    changes = [
        RowChange(
            key="1",
            change_type=ChangeType.ADDED,
            old_values=None,
            new_values={"id": 1, "name": "alice"},
        ),
        RowChange(
            key="2",
            change_type=ChangeType.MODIFIED,
            old_values={"id": 2, "name": "bob"},
            new_values={"id": 2, "name": "bobby"},
        ),
    ]

    output = data.format_row_changes_table(changes, columns=["id", "name"], output_format=None)

    assert "+" in output
    assert "~" in output
    assert "alice" in output
    assert "bobby" in output


# =============================================================================
# Cache Restore Tests
# =============================================================================


def test_restore_data_from_cache(
    tmp_path: Path,
    set_project_root: Path,
) -> None:
    """Restore data file from cache."""
    # Use valid 16-character xxhash64 format (hash validation requires it)
    file_hash = "abc1234567890def"
    cache_dir = tmp_path / ".pivot" / "cache" / "files" / file_hash[:2]
    cache_dir.mkdir(parents=True)
    cached_file = cache_dir / file_hash[2:]
    cached_file.write_text("id,name\n1,alice\n")

    temp_path = data.restore_data_from_cache("data.csv", file_hash)

    assert temp_path is not None
    assert temp_path.exists()
    assert temp_path.read_text() == "id,name\n1,alice\n"
    temp_path.unlink()  # Cleanup


def test_restore_data_from_cache_not_found(
    tmp_path: Path,
    set_project_root: Path,
) -> None:
    """Return None when cache file not found."""
    # Use valid hash format, but file doesn't exist
    result = data.restore_data_from_cache("data.csv", "0123456789abcdef")
    assert result is None


def test_restore_data_from_cache_invalid_hash(
    tmp_path: Path,
    set_project_root: Path,
) -> None:
    """Return None when hash format is invalid (security check)."""
    # Invalid: too short
    assert data.restore_data_from_cache("data.csv", "abc123") is None
    # Invalid: contains path traversal attempt
    assert data.restore_data_from_cache("data.csv", "../../../etc/pass") is None
    # Invalid: contains non-hex characters
    assert data.restore_data_from_cache("data.csv", "abc123ghijklmnop") is None


def test_restore_data_from_cache_symlink_blocked(
    tmp_path: Path,
    set_project_root: Path,
) -> None:
    """Return None when cached file is a symlink (security check)."""
    file_hash = "abc1234567890def"
    cache_dir = tmp_path / ".pivot" / "cache" / "files" / file_hash[:2]
    cache_dir.mkdir(parents=True)

    # Create a real file and a symlink to it
    real_file = tmp_path / "real_file.csv"
    real_file.write_text("id,name\n1,alice\n")
    symlink_file = cache_dir / file_hash[2:]
    symlink_file.symlink_to(real_file)

    # Should return None because it's a symlink
    result = data.restore_data_from_cache("data.csv", file_hash)
    assert result is None


# =============================================================================
# Numpy Encoder Tests
# =============================================================================


def test_numpy_encoder_integer() -> None:
    """Encode numpy integer types."""
    import numpy

    result = json.dumps({"value": numpy.int64(42)}, cls=data._NumpyEncoder)
    assert json.loads(result) == {"value": 42}


def test_numpy_encoder_floating() -> None:
    """Encode numpy floating types (float32 needs custom encoder, float64 is native)."""
    import numpy

    # numpy.float32 needs custom encoder (numpy.float64 serializes natively)
    result = json.dumps({"value": numpy.float32(3.14)}, cls=data._NumpyEncoder)
    parsed = json.loads(result)
    assert abs(parsed["value"] - 3.14) < 0.001


def test_numpy_encoder_bool() -> None:
    """Encode numpy boolean types."""
    import numpy

    result = json.dumps({"value": numpy.bool_(True)}, cls=data._NumpyEncoder)
    assert json.loads(result) == {"value": True}


def test_numpy_encoder_array() -> None:
    """Encode numpy arrays."""
    import numpy

    result = json.dumps({"value": numpy.array([1, 2, 3])}, cls=data._NumpyEncoder)
    assert json.loads(result) == {"value": [1, 2, 3]}


def test_numpy_encoder_timestamp() -> None:
    """Encode pandas Timestamp."""
    result = json.dumps({"value": pandas.Timestamp("2024-01-01")}, cls=data._NumpyEncoder)
    parsed = json.loads(result)
    assert "2024-01-01" in parsed["value"]


def test_numpy_encoder_timedelta() -> None:
    """Encode pandas Timedelta."""
    result = json.dumps({"value": pandas.Timedelta(days=1)}, cls=data._NumpyEncoder)
    parsed = json.loads(result)
    assert "1 day" in parsed["value"]


def test_numpy_encoder_pandas_na() -> None:
    """Encode pandas.NA as None."""
    result = json.dumps({"value": pandas.NA}, cls=data._NumpyEncoder)
    assert json.loads(result) == {"value": None}


def test_numpy_encoder_nat() -> None:
    """Encode NaT as None."""
    result = json.dumps({"value": pandas.NaT}, cls=data._NumpyEncoder)
    assert json.loads(result) == {"value": None}


# =============================================================================
# Row Count Edge Cases
# =============================================================================


def test_get_row_count_json(tmp_path: Path) -> None:
    """Get row count from JSON file (needs full load)."""
    json_file = tmp_path / "data.json"
    json_file.write_text(json.dumps([{"id": 1}, {"id": 2}, {"id": 3}]))

    count = data.get_row_count(json_file)

    assert count == 3


def test_get_row_count_csv_empty(tmp_path: Path) -> None:
    """Empty CSV returns 0 rows."""
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("")

    count = data.get_row_count(csv_file)

    assert count == 0


def test_get_row_count_csv_header_only(tmp_path: Path) -> None:
    """CSV with header only returns 0 rows."""
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("id,name\n")

    count = data.get_row_count(csv_file)

    assert count == 0


def test_get_row_count_csv_embedded_newlines(tmp_path: Path) -> None:
    """CSV with embedded newlines in quoted fields counts correctly."""
    csv_file = tmp_path / "data.csv"
    csv_file.write_text('id,note\n1,"line1\nline2"\n2,"simple"\n')

    count = data.get_row_count(csv_file)

    assert count == 2, "Embedded newlines should not inflate row count"


# =============================================================================
# Reorder Detection Edge Cases
# =============================================================================


def test_check_reorder_only_unsortable_with_counter() -> None:
    """Unsortable columns use Counter fallback."""
    # Create dataframes with mixed types that can't be sorted but can be hashed
    old_df = pandas.DataFrame({"id": [1, 2], "name": ["a", "b"]})
    new_df = pandas.DataFrame({"id": [2, 1], "name": ["b", "a"]})

    # This should work via Counter fallback if sort fails
    result = data.check_reorder_only(old_df, new_df)
    assert result is True


def test_check_reorder_only_unhashable_returns_false() -> None:
    """Unhashable values (dicts/lists) return False as fallback."""
    # DataFrames with list values can't be sorted or hashed
    old_df = pandas.DataFrame({"id": [1, 2], "data": [[1, 2], [3, 4]]})
    new_df = pandas.DataFrame({"id": [2, 1], "data": [[3, 4], [1, 2]]})

    # Should return False due to unhashable types
    result = data.check_reorder_only(old_df, new_df)
    assert result is False


# =============================================================================
# Format Summary Edge Cases
# =============================================================================


def test_format_diff_summary_removed_file() -> None:
    """Format summary for removed file."""
    result = DataDiffResult(
        path="data.csv",
        old_rows=50,
        new_rows=None,
        old_cols=["id", "name"],
        new_cols=None,
        schema_changes=[],
        row_changes=[],
        reorder_only=False,
        truncated=False,
        summary_only=True,
    )

    output = data.format_diff_summary(result)

    assert "50" in output
    assert "(removed)" in output


def test_format_diff_summary_added_file() -> None:
    """Format summary for added file."""
    result = DataDiffResult(
        path="data.csv",
        old_rows=None,
        new_rows=100,
        old_cols=None,
        new_cols=["id", "name"],
        schema_changes=[],
        row_changes=[],
        reorder_only=False,
        truncated=False,
        summary_only=True,
    )

    output = data.format_diff_summary(result)

    assert "100" in output
    assert "(added)" in output


def test_format_diff_summary_with_schema_changes() -> None:
    """Format summary with various schema changes."""
    result = DataDiffResult(
        path="data.csv",
        old_rows=100,
        new_rows=100,
        old_cols=["id", "old_col", "modified_col"],
        new_cols=["id", "new_col", "modified_col"],
        schema_changes=[
            SchemaChange(
                column="new_col", old_dtype=None, new_dtype="int64", change_type=ChangeType.ADDED
            ),
            SchemaChange(
                column="old_col",
                old_dtype="float64",
                new_dtype=None,
                change_type=ChangeType.REMOVED,
            ),
            SchemaChange(
                column="modified_col",
                old_dtype="int64",
                new_dtype="float64",
                change_type=ChangeType.MODIFIED,
            ),
        ],
        row_changes=[],
        reorder_only=False,
        truncated=False,
        summary_only=False,
    )

    output = data.format_diff_summary(result)

    assert "+" in output and "new_col" in output
    assert "-" in output and "old_col" in output
    assert "~" in output and "modified_col" in output


def test_format_diff_summary_with_row_changes() -> None:
    """Format summary with row changes."""
    result = DataDiffResult(
        path="data.csv",
        old_rows=100,
        new_rows=105,
        old_cols=["id", "name"],
        new_cols=["id", "name"],
        schema_changes=[],
        row_changes=[
            RowChange(key="1", change_type=ChangeType.ADDED, old_values=None, new_values={}),
            RowChange(key="2", change_type=ChangeType.ADDED, old_values=None, new_values={}),
            RowChange(key="3", change_type=ChangeType.REMOVED, old_values={}, new_values=None),
            RowChange(key="4", change_type=ChangeType.MODIFIED, old_values={}, new_values={}),
        ],
        reorder_only=False,
        truncated=True,
        summary_only=False,
    )

    output = data.format_diff_summary(result)

    assert "+2 added" in output
    assert "-1 removed" in output
    assert "~1 modified" in output
    assert "(truncated)" in output


# =============================================================================
# Format Row Changes Table Tests
# =============================================================================


def test_format_row_changes_table_json() -> None:
    """Format row changes as JSON."""
    changes = [
        RowChange(
            key="1",
            change_type=ChangeType.ADDED,
            old_values=None,
            new_values={"id": 1, "name": "alice"},
        ),
    ]

    output = data.format_row_changes_table(
        changes, columns=["id", "name"], output_format=OutputFormat.JSON
    )

    parsed = json.loads(output)
    assert len(parsed) == 1
    assert parsed[0]["key"] == "1"


def test_format_row_changes_table_empty() -> None:
    """Format empty row changes."""
    output = data.format_row_changes_table([], columns=["id"], output_format=None)
    assert "No row changes" in output


def test_format_row_changes_table_removed_row() -> None:
    """Format table with removed row shows old values."""
    changes = [
        RowChange(
            key="1",
            change_type=ChangeType.REMOVED,
            old_values={"id": 1, "name": "alice"},
            new_values=None,
        ),
    ]

    output = data.format_row_changes_table(changes, columns=["id", "name"], output_format=None)

    assert "-" in output
    assert "alice" in output


def test_format_row_changes_table_markdown() -> None:
    """Format row changes as markdown."""
    changes = [
        RowChange(
            key="1",
            change_type=ChangeType.ADDED,
            old_values=None,
            new_values={"id": 1, "name": "alice"},
        ),
    ]

    output = data.format_row_changes_table(
        changes, columns=["id", "name"], output_format=OutputFormat.MD
    )

    # Markdown table has | separators
    assert "|" in output


# =============================================================================
# Schema Tests (JSONL)
# =============================================================================


def test_get_schema_jsonl(tmp_path: Path) -> None:
    """Get schema from JSONL file."""
    jsonl_file = tmp_path / "data.jsonl"
    jsonl_file.write_text('{"id": 1, "name": "alice"}\n{"id": 2, "name": "bob"}\n')

    schema = data.get_schema(jsonl_file)

    assert "id" in schema
    assert "name" in schema


def test_get_schema_unsupported(tmp_path: Path) -> None:
    """Unsupported format raises DataError."""
    file = tmp_path / "data.parquet"
    file.write_bytes(b"fake parquet")

    with pytest.raises(data.DataError, match="Unsupported data format"):
        data.get_schema(file)


# =============================================================================
# Registry Integration Tests (with mocking)
# =============================================================================


def test_get_data_outputs_from_stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Get data outputs from mocked cli_helpers."""
    from pivot import loaders, outputs, project
    from pivot.cli import helpers as cli_helpers

    # Mock project root
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    # Define mock stage info
    stage_info: dict[str, dict[str, object]] = {
        "process_data": {
            "outs": [outputs.Out("output.csv", loader=loaders.PathOnly())],
            "deps": [],
            "func": lambda: None,
            "params": None,
        },
        "train_model": {
            "outs": [
                outputs.Out("model.pkl", loader=loaders.PathOnly()),
                outputs.Metric("metrics.json"),
            ],
            "deps": [],
            "func": lambda: None,
            "params": None,
        },
    }
    default_stage: dict[str, object] = {
        "outs": [],
        "deps": [],
        "func": lambda: None,
        "params": None,
    }

    def mock_list_stages() -> list[str]:
        return ["process_data", "train_model"]

    def mock_get_stage(name: str) -> dict[str, object]:
        return stage_info.get(name, default_stage)

    monkeypatch.setattr(cli_helpers, "list_stages", mock_list_stages)
    monkeypatch.setattr(cli_helpers, "get_stage", mock_get_stage)

    result = data.get_data_outputs_from_stages()

    # Only CSV files should be included (not metrics, not pkl)
    assert "process_data" in result
    assert result["process_data"] == "output.csv"
    assert "train_model" not in result  # model.pkl is unknown format


def test_get_data_hashes_from_head(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Get data hashes from mocked cli_helpers and git HEAD."""
    from pivot import git, loaders, outputs, project
    from pivot.cli import helpers as cli_helpers

    # Mock project root
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    # Define mock stage info
    stage_info: dict[str, dict[str, object]] = {
        "process_data": {
            "outs": [outputs.Out("output.csv", loader=loaders.PathOnly())],
            "deps": [],
            "func": lambda: None,
            "params": None,
        },
    }
    default_stage: dict[str, object] = {
        "outs": [],
        "deps": [],
        "func": lambda: None,
        "params": None,
    }

    def mock_list_stages() -> list[str]:
        return ["process_data"]

    def mock_get_stage(name: str) -> dict[str, object]:
        return stage_info.get(name, default_stage)

    monkeypatch.setattr(cli_helpers, "list_stages", mock_list_stages)
    monkeypatch.setattr(cli_helpers, "get_stage", mock_get_stage)

    # Mock git.read_files_from_head to return a lock file
    lock_content = """
code_manifest:
  func:main: abc123
params: {}
deps: []
outs:
  - path: output.csv
    hash: deadbeef
dep_generations: {}
"""

    def mock_read_files(paths: list[str]) -> dict[str, str | None]:
        return {".pivot/stages/process_data.lock": lock_content}

    monkeypatch.setattr(git, "read_files_from_head", mock_read_files)

    result = data.get_data_hashes_from_head()

    assert "output.csv" in result
    assert result["output.csv"] == "deadbeef"


def test_get_data_hashes_from_head_no_lock_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Handle missing lock file gracefully."""
    from pivot import git, loaders, outputs, project
    from pivot.cli import helpers as cli_helpers

    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    def mock_list_stages() -> list[str]:
        return ["process_data"]

    def mock_get_stage(name: str) -> dict[str, object]:
        return {
            "outs": [outputs.Out("output.csv", loader=loaders.PathOnly())],
            "deps": [],
            "func": lambda: None,
            "params": None,
        }

    monkeypatch.setattr(cli_helpers, "list_stages", mock_list_stages)
    monkeypatch.setattr(cli_helpers, "get_stage", mock_get_stage)

    # No lock file found
    def mock_read_files(paths: list[str]) -> dict[str, str | None]:
        return {}

    monkeypatch.setattr(git, "read_files_from_head", mock_read_files)

    result = data.get_data_hashes_from_head()

    assert "output.csv" in result
    assert result["output.csv"] is None  # No hash available


def test_get_data_hashes_from_head_invalid_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Handle invalid YAML in lock file gracefully."""
    from pivot import git, loaders, outputs, project
    from pivot.cli import helpers as cli_helpers

    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    def mock_list_stages() -> list[str]:
        return ["process_data"]

    def mock_get_stage(name: str) -> dict[str, object]:
        return {
            "outs": [outputs.Out("output.csv", loader=loaders.PathOnly())],
            "deps": [],
            "func": lambda: None,
            "params": None,
        }

    monkeypatch.setattr(cli_helpers, "list_stages", mock_list_stages)
    monkeypatch.setattr(cli_helpers, "get_stage", mock_get_stage)

    # Invalid YAML
    def mock_read_files(paths: list[str]) -> dict[str, str | None]:
        return {".pivot/stages/process_data.lock": "invalid: yaml: content: ["}

    monkeypatch.setattr(git, "read_files_from_head", mock_read_files)

    result = data.get_data_hashes_from_head()

    assert "output.csv" in result
    assert result["output.csv"] is None  # Couldn't parse


def test_get_data_hashes_from_head_missing_outs_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Handle lock file missing outs key."""
    from pivot import git, loaders, outputs, project
    from pivot.cli import helpers as cli_helpers

    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    def mock_list_stages() -> list[str]:
        return ["process_data"]

    def mock_get_stage(name: str) -> dict[str, object]:
        return {
            "outs": [outputs.Out("output.csv", loader=loaders.PathOnly())],
            "deps": [],
            "func": lambda: None,
            "params": None,
        }

    monkeypatch.setattr(cli_helpers, "list_stages", mock_list_stages)
    monkeypatch.setattr(cli_helpers, "get_stage", mock_get_stage)

    # Lock file without outs key
    def mock_read_files(paths: list[str]) -> dict[str, str | None]:
        return {".pivot/stages/process_data.lock": "fingerprint: abc123\n"}

    monkeypatch.setattr(git, "read_files_from_head", mock_read_files)

    result = data.get_data_hashes_from_head()

    assert result["output.csv"] is None


def test_get_data_hashes_from_head_outs_not_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Handle lock file where outs is not a list."""
    from pivot import git, loaders, outputs, project
    from pivot.cli import helpers as cli_helpers

    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    def mock_list_stages() -> list[str]:
        return ["process_data"]

    def mock_get_stage(name: str) -> dict[str, object]:
        return {
            "outs": [outputs.Out("output.csv", loader=loaders.PathOnly())],
            "deps": [],
            "func": lambda: None,
            "params": None,
        }

    monkeypatch.setattr(cli_helpers, "list_stages", mock_list_stages)
    monkeypatch.setattr(cli_helpers, "get_stage", mock_get_stage)

    # Lock file with outs as string instead of list
    def mock_read_files(paths: list[str]) -> dict[str, str | None]:
        return {".pivot/stages/process_data.lock": "outs: not_a_list\n"}

    monkeypatch.setattr(git, "read_files_from_head", mock_read_files)

    result = data.get_data_hashes_from_head()

    assert result["output.csv"] is None
