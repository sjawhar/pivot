from __future__ import annotations

import inspect
import json
from typing import TYPE_CHECKING

import pytest
import yaml

from pivot import outputs
from pivot.registry import REGISTRY, RegistryStageInfo
from pivot.show import metrics
from pivot.types import ChangeType, OutputFormat

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

    from pivot.types import MetricValue
    from tests.conftest import ValidLockContentFactory


def _register_metric_stage(
    name: str,
    metric_path: str,
) -> None:
    """Register a test stage with a Metric output directly in the registry.

    This bypasses the annotation-based registration since Metric outputs
    can't be expressed through annotations (they require outputs.Metric).
    """

    def _stage_func() -> None:
        pass

    REGISTRY._stages[name] = RegistryStageInfo(
        func=_stage_func,
        name=name,
        deps={},
        deps_paths=[],
        outs=[outputs.Metric(metric_path)],
        outs_paths=[metric_path],
        params=None,
        mutex=[],
        variant=None,
        signature=inspect.signature(_stage_func),
        fingerprint={"_code": "fake_hash"},
        cwd=None,
        dep_specs={},
        out_path_overrides=None,
    )


# =============================================================================
# Parsing Tests
# =============================================================================


def test_parse_json_file(tmp_path: Path) -> None:
    """Parse JSON metric file and flatten nested dicts."""
    metric_file = tmp_path / "metrics.json"
    metric_file.write_text(json.dumps({"accuracy": 0.95, "loss": 0.05}))

    result = metrics.parse_metric_file(metric_file)

    assert result == {"accuracy": 0.95, "loss": 0.05}


def test_parse_json_file_nested(tmp_path: Path) -> None:
    """Parse JSON with nested dict, flatten using dot notation."""
    metric_file = tmp_path / "metrics.json"
    metric_file.write_text(json.dumps({"train": {"accuracy": 0.95}, "test": {"accuracy": 0.90}}))

    result = metrics.parse_metric_file(metric_file)

    assert result == {"train.accuracy": 0.95, "test.accuracy": 0.90}


def test_parse_yaml_file(tmp_path: Path) -> None:
    """Parse YAML metric file."""
    metric_file = tmp_path / "metrics.yaml"
    metric_file.write_text(yaml.dump({"accuracy": 0.95, "epochs": 10}))

    result = metrics.parse_metric_file(metric_file)

    assert result == {"accuracy": 0.95, "epochs": 10}


def test_parse_yaml_file_yml_extension(tmp_path: Path) -> None:
    """Parse .yml extension as YAML."""
    metric_file = tmp_path / "metrics.yml"
    metric_file.write_text(yaml.dump({"f1": 0.88}))

    result = metrics.parse_metric_file(metric_file)

    assert result == {"f1": 0.88}


def test_parse_csv_file(tmp_path: Path) -> None:
    """Parse CSV file (key,value format)."""
    metric_file = tmp_path / "metrics.csv"
    metric_file.write_text("accuracy,0.95\nloss,0.05\nepochs,10\n")

    result = metrics.parse_metric_file(metric_file)

    assert result["accuracy"] == 0.95
    assert result["loss"] == 0.05
    assert result["epochs"] == 10


def test_parse_csv_file_bool_values(tmp_path: Path) -> None:
    """Parse CSV boolean values."""
    metric_file = tmp_path / "metrics.csv"
    metric_file.write_text("converged,true\nfailed,false\n")

    result = metrics.parse_metric_file(metric_file)

    assert result["converged"] is True
    assert result["failed"] is False


def test_parse_csv_file_string_values(tmp_path: Path) -> None:
    """Parse CSV string values."""
    metric_file = tmp_path / "metrics.csv"
    metric_file.write_text("model_name,bert-base\n")

    result = metrics.parse_metric_file(metric_file)

    assert result["model_name"] == "bert-base"


def test_parse_unsupported_format(tmp_path: Path) -> None:
    """Unsupported format raises MetricsError."""
    metric_file = tmp_path / "metrics.txt"
    metric_file.write_text("data")

    with pytest.raises(metrics.MetricsError, match="Unsupported"):
        metrics.parse_metric_file(metric_file)


def test_parse_invalid_json(tmp_path: Path) -> None:
    """Invalid JSON raises MetricsError."""
    metric_file = tmp_path / "metrics.json"
    metric_file.write_text("{invalid}")

    with pytest.raises(metrics.MetricsError, match="Failed to parse"):
        metrics.parse_metric_file(metric_file)


def test_parse_yaml_non_dict(tmp_path: Path) -> None:
    """YAML file with non-dict content raises MetricsError."""
    metric_file = tmp_path / "metrics.yaml"
    metric_file.write_text("- item1\n- item2\n")

    with pytest.raises(metrics.MetricsError, match="Expected dict"):
        metrics.parse_metric_file(metric_file)


def test_parse_json_non_dict(tmp_path: Path) -> None:
    """JSON file with non-dict content raises MetricsError."""
    metric_file = tmp_path / "metrics.json"
    metric_file.write_text("[1, 2, 3]")

    with pytest.raises(metrics.MetricsError, match="Expected dict"):
        metrics.parse_metric_file(metric_file)


# =============================================================================
# Flatten Dict Tests
# =============================================================================


def test_flatten_dict_empty() -> None:
    """Empty dict returns empty dict."""
    result = metrics._flatten_dict({})
    assert result == {}


def test_flatten_dict_flat() -> None:
    """Flat dict unchanged."""
    result = metrics._flatten_dict({"a": 1, "b": 2})
    assert result == {"a": 1, "b": 2}


def test_flatten_dict_nested() -> None:
    """Nested dict flattened with dot notation."""
    result = metrics._flatten_dict({"outer": {"inner": {"value": 42}}})
    assert result == {"outer.inner.value": 42}


# =============================================================================
# Collection Tests
# =============================================================================


def test_collect_metrics_from_files_single(tmp_path: Path) -> None:
    """Collect metrics from single file."""
    metric_file = tmp_path / "metrics.json"
    metric_file.write_text(json.dumps({"accuracy": 0.95}))

    result = metrics.collect_metrics_from_files([str(metric_file)])

    assert str(metric_file) in result
    assert result[str(metric_file)] == {"accuracy": 0.95}


def test_collect_metrics_from_files_directory(tmp_path: Path) -> None:
    """Collect metrics from directory (non-recursive)."""
    (tmp_path / "a.json").write_text(json.dumps({"a": 1}))
    (tmp_path / "b.yaml").write_text(yaml.dump({"b": 2}))
    subdir = tmp_path / "sub"
    subdir.mkdir()
    (subdir / "c.json").write_text(json.dumps({"c": 3}))

    result = metrics.collect_metrics_from_files([str(tmp_path)], recursive=False)

    assert len(result) == 2, "Non-recursive should skip subdir"
    assert str(tmp_path / "a.json") in result
    assert str(tmp_path / "b.yaml") in result


def test_collect_metrics_from_files_recursive(tmp_path: Path) -> None:
    """Collect metrics from directory recursively."""
    (tmp_path / "a.json").write_text(json.dumps({"a": 1}))
    subdir = tmp_path / "sub"
    subdir.mkdir()
    (subdir / "b.json").write_text(json.dumps({"b": 2}))

    result = metrics.collect_metrics_from_files([str(tmp_path)], recursive=True)

    assert len(result) == 2
    assert str(tmp_path / "a.json") in result
    assert str(subdir / "b.json") in result


def test_collect_metrics_from_files_not_found(tmp_path: Path) -> None:
    """Missing file raises MetricsError."""
    with pytest.raises(metrics.MetricsError, match="not found"):
        metrics.collect_metrics_from_files([str(tmp_path / "nonexistent.json")])


# =============================================================================
# Diff Tests
# =============================================================================


def test_diff_metrics_no_changes() -> None:
    """No changes returns empty list."""
    old = {"metrics.json": {"accuracy": 0.95}}
    new = {"metrics.json": {"accuracy": 0.95}}

    result = metrics.diff_metrics(old, new)

    assert result == []


def test_diff_metrics_modified() -> None:
    """Value change detected as modified."""
    old = {"metrics.json": {"accuracy": 0.90}}
    new = {"metrics.json": {"accuracy": 0.95}}

    result = metrics.diff_metrics(old, new)

    assert len(result) == 1
    assert result[0]["change_type"] == "modified"
    assert result[0]["old"] == 0.90
    assert result[0]["new"] == 0.95


def test_diff_metrics_added() -> None:
    """New key detected as added."""
    old = {"metrics.json": {"accuracy": 0.95}}
    new = {"metrics.json": {"accuracy": 0.95, "f1": 0.88}}

    result = metrics.diff_metrics(old, new)

    assert len(result) == 1
    assert result[0]["change_type"] == "added"
    assert result[0]["key"] == "f1"
    assert result[0]["old"] is None
    assert result[0]["new"] == 0.88


def test_diff_metrics_removed() -> None:
    """Missing key detected as removed."""
    old = {"metrics.json": {"accuracy": 0.95, "f1": 0.88}}
    new = {"metrics.json": {"accuracy": 0.95}}

    result = metrics.diff_metrics(old, new)

    assert len(result) == 1
    assert result[0]["change_type"] == "removed"
    assert result[0]["key"] == "f1"
    assert result[0]["old"] == 0.88
    assert result[0]["new"] is None


def test_diff_metrics_new_file() -> None:
    """New file detected."""
    old: dict[str, dict[str, MetricValue]] = {}
    new = {"metrics.json": {"accuracy": 0.95}}

    result = metrics.diff_metrics(old, new)

    assert len(result) == 1
    assert result[0]["change_type"] == "added"
    assert result[0]["path"] == "metrics.json"


def test_diff_metrics_removed_file() -> None:
    """Removed file detected."""
    old = {"metrics.json": {"accuracy": 0.95}}
    new: dict[str, dict[str, MetricValue]] = {}

    result = metrics.diff_metrics(old, new)

    assert len(result) == 1
    assert result[0]["change_type"] == "removed"
    assert result[0]["path"] == "metrics.json"


# =============================================================================
# Formatting Tests
# =============================================================================


def test_format_metrics_table_plain() -> None:
    """Plain format uses tabulate."""
    data = {"metrics.json": {"accuracy": 0.95, "loss": 0.05}}

    result = metrics.format_metrics_table(data, None, precision=5)

    assert "Path" in result
    assert "Key" in result
    assert "Value" in result
    assert "metrics.json" in result
    assert "accuracy" in result
    assert "0.95000" in result


def test_format_metrics_table_json() -> None:
    """JSON format outputs valid JSON."""
    data = {"metrics.json": {"accuracy": 0.95}}

    result = metrics.format_metrics_table(data, OutputFormat.JSON, precision=5)

    parsed = json.loads(result)
    assert parsed == {"metrics.json": {"accuracy": 0.95}}


def test_format_metrics_table_markdown() -> None:
    """Markdown format uses github table style."""
    data = {"metrics.json": {"accuracy": 0.95}}

    result = metrics.format_metrics_table(data, OutputFormat.MD, precision=5)

    assert "|" in result
    assert "---" in result  # Header separator


def test_format_metrics_table_empty() -> None:
    """Empty metrics shows no metrics message."""
    result = metrics.format_metrics_table({}, None, precision=5)
    assert "No metrics found" in result


def test_format_diff_table_plain() -> None:
    """Plain format for diff."""
    diffs = [
        metrics.MetricDiff(
            path="m.json", key="acc", old=0.9, new=0.95, change_type=ChangeType.MODIFIED
        )
    ]

    result = metrics.format_diff_table(diffs, None, precision=2, show_path=True)

    assert "Path" in result
    assert "Key" in result
    assert "Old" in result
    assert "New" in result
    assert "m.json" in result
    assert "0.90" in result
    assert "0.95" in result


def test_format_diff_table_no_path() -> None:
    """Diff without path column."""
    diffs = [
        metrics.MetricDiff(
            path="m.json", key="acc", old=0.9, new=0.95, change_type=ChangeType.MODIFIED
        )
    ]

    result = metrics.format_diff_table(diffs, None, precision=2, show_path=False)

    assert "Path" not in result
    assert "acc" in result


def test_format_diff_table_empty() -> None:
    """Empty diff shows no changes message."""
    result = metrics.format_diff_table([], None, precision=5)
    assert "No metric changes" in result


def test_format_diff_table_json() -> None:
    """JSON format for diff."""
    diffs = [
        metrics.MetricDiff(
            path="m.json", key="acc", old=0.9, new=0.95, change_type=ChangeType.MODIFIED
        )
    ]

    result = metrics.format_diff_table(diffs, OutputFormat.JSON, precision=2)

    parsed = json.loads(result)
    assert len(parsed) == 1
    assert parsed[0]["change_type"] == "modified"


# =============================================================================
# Format Value Tests
# =============================================================================


def test_format_value_float_precision() -> None:
    """Float values formatted with precision."""
    result = metrics._format_value(0.123456789, precision=3)
    assert result == "0.123"


def test_format_value_int() -> None:
    """Int values converted to string."""
    result = metrics._format_value(42, precision=5)
    assert result == "42"


def test_format_value_string() -> None:
    """String values unchanged."""
    result = metrics._format_value("hello", precision=5)
    assert result == "hello"


def test_format_value_none() -> None:
    """None values formatted as dash."""
    result = metrics._format_value(None, precision=5)
    assert result == "-"


# =============================================================================
# Parse Content Tests
# =============================================================================


def test_parse_metric_content_json() -> None:
    """Parse JSON content string."""
    content = '{"accuracy": 0.95, "nested": {"f1": 0.88}}'
    result = metrics.parse_metric_content(content, "metrics.json")

    assert result["accuracy"] == 0.95
    assert result["nested.f1"] == 0.88


def test_parse_metric_content_json_bytes() -> None:
    """Parse JSON content as bytes."""
    content = b'{"accuracy": 0.95}'
    result = metrics.parse_metric_content(content, "metrics.json")

    assert result["accuracy"] == 0.95


def test_parse_metric_content_yaml() -> None:
    """Parse YAML content string."""
    content = "accuracy: 0.95\nloss: 0.05"
    result = metrics.parse_metric_content(content, "metrics.yaml")

    assert result["accuracy"] == 0.95
    assert result["loss"] == 0.05


def test_parse_metric_content_csv() -> None:
    """Parse CSV content string."""
    content = "accuracy,0.95\nloss,0.05"
    result = metrics.parse_metric_content(content, "metrics.csv")

    assert result["accuracy"] == 0.95
    assert result["loss"] == 0.05


def test_parse_csv_content_empty_rows() -> None:
    """CSV content with empty rows skipped."""
    content = "accuracy,0.95\n\nloss,0.05"
    result = metrics._parse_csv_content(content)

    assert result["accuracy"] == 0.95
    assert result["loss"] == 0.05


def test_parse_csv_content_short_rows(caplog: pytest.LogCaptureFixture) -> None:
    """CSV content with short rows logs warning and skips."""
    content = "accuracy,0.95\norphan_key\nloss,0.05"
    result = metrics._parse_csv_content(content)

    assert result["accuracy"] == 0.95
    assert result["loss"] == 0.05
    assert "orphan_key" not in result
    assert "skipping row with 1 column(s)" in caplog.text


def test_parse_csv_file_short_rows(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """CSV file with short rows logs warning and skips."""
    metric_file = tmp_path / "metrics.csv"
    metric_file.write_text("accuracy,0.95\norphan\nloss,0.05\n")

    result = metrics.parse_metric_file(metric_file)

    assert result["accuracy"] == 0.95
    assert result["loss"] == 0.05
    assert "orphan" not in result
    assert "skipping row with 1 column(s)" in caplog.text


def test_parse_metric_content_unsupported_format() -> None:
    """parse_metric_content raises MetricsError for unsupported format."""
    with pytest.raises(metrics.MetricsError, match="Unsupported"):
        metrics.parse_metric_content("data", "metrics.txt")


def test_parse_metric_content_invalid_json() -> None:
    """parse_metric_content raises MetricsError for invalid JSON."""
    with pytest.raises(metrics.MetricsError, match="Failed to parse"):
        metrics.parse_metric_content("{invalid}", "metrics.json")


def test_parse_metric_content_invalid_yaml() -> None:
    """parse_metric_content raises MetricsError for invalid YAML."""
    with pytest.raises(metrics.MetricsError, match="Failed to parse"):
        metrics.parse_metric_content(":\n  :\n invalid", "metrics.yaml")


# =============================================================================
# Stage Collection Tests
# =============================================================================


def test_collect_metrics_from_stages_with_metric_output(
    tmp_path: Path,
    set_project_root: Path,
) -> None:
    """Collect metrics from stages with Metric outputs."""
    metric_file = tmp_path / "metrics.json"
    metric_file.write_text(json.dumps({"accuracy": 0.95}))

    _register_metric_stage("my_stage", str(metric_file))

    result = metrics.collect_metrics_from_stages()

    assert "my_stage" in result
    assert str(metric_file) in result["my_stage"]
    assert result["my_stage"][str(metric_file)]["accuracy"] == 0.95


def test_collect_metrics_from_stages_missing_file(
    tmp_path: Path,
    set_project_root: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Missing metric file logs warning and skips."""
    _register_metric_stage("my_stage", str(tmp_path / "nonexistent.json"))

    result = metrics.collect_metrics_from_stages()

    assert "my_stage" not in result
    assert "Metric file not found" in caplog.text


def test_collect_metrics_from_stages_parse_error(
    tmp_path: Path,
    set_project_root: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Parse error in metric file logs warning and skips."""
    metric_file = tmp_path / "metrics.json"
    metric_file.write_text("{invalid json}")

    _register_metric_stage("my_stage", str(metric_file))

    result = metrics.collect_metrics_from_stages()

    assert "my_stage" not in result
    assert "Failed to parse metrics" in caplog.text


def test_collect_all_stage_metrics_flat(
    tmp_path: Path,
    set_project_root: Path,
) -> None:
    """Collect and flatten metrics from all stages."""
    metric_file = tmp_path / "metrics.json"
    metric_file.write_text(json.dumps({"accuracy": 0.95}))

    _register_metric_stage("my_stage", str(metric_file))

    result = metrics.collect_all_stage_metrics_flat()

    assert str(metric_file) in result
    assert result[str(metric_file)]["accuracy"] == 0.95


# =============================================================================
# Git/HEAD Collection Tests
# =============================================================================


def test_get_metric_info_from_head_no_stages(set_project_root: Path) -> None:
    """No registered stages returns empty dict."""
    result = metrics.get_metric_info_from_head()
    assert result == {}


def test_get_metric_info_from_head_with_metric_stage(
    tmp_path: Path,
    set_project_root: Path,
    mocker: MockerFixture,
    make_valid_lock_content: ValidLockContentFactory,
) -> None:
    """Collect metric hashes from lock files at HEAD."""
    from pivot import git

    metric_file = tmp_path / "metrics.json"

    _register_metric_stage("my_stage", str(metric_file))

    lock_content = yaml.dump(
        make_valid_lock_content(outs=[{"path": "metrics.json", "hash": "abc123"}])
    )
    mocker.patch.object(
        git,
        "read_files_from_head",
        return_value={".pivot/stages/my_stage.lock": lock_content.encode()},
    )

    result = metrics.get_metric_info_from_head()

    assert "metrics.json" in result
    assert result["metrics.json"] == "abc123"


def test_get_metric_info_from_head_no_lock_file(
    tmp_path: Path,
    set_project_root: Path,
    mocker: MockerFixture,
) -> None:
    """Missing lock file returns None hash."""
    from pivot import git

    metric_file = tmp_path / "metrics.json"

    _register_metric_stage("my_stage", str(metric_file))

    mocker.patch.object(git, "read_files_from_head", return_value={})

    result = metrics.get_metric_info_from_head()

    assert "metrics.json" in result
    assert result["metrics.json"] is None


def test_get_metric_info_from_head_invalid_lock_yaml(
    tmp_path: Path,
    set_project_root: Path,
    mocker: MockerFixture,
) -> None:
    """Invalid YAML in lock file returns None hash."""
    from pivot import git

    metric_file = tmp_path / "metrics.json"

    _register_metric_stage("my_stage", str(metric_file))

    mocker.patch.object(
        git,
        "read_files_from_head",
        return_value={".pivot/stages/my_stage.lock": b":\n  :\n invalid"},
    )

    result = metrics.get_metric_info_from_head()

    assert "metrics.json" in result
    assert result["metrics.json"] is None


def test_get_metric_info_from_head_lock_missing_outs(
    tmp_path: Path,
    set_project_root: Path,
    mocker: MockerFixture,
) -> None:
    """Lock file without 'outs' key returns None hash."""
    from pivot import git

    metric_file = tmp_path / "metrics.json"

    _register_metric_stage("my_stage", str(metric_file))

    lock_content = yaml.dump({"deps": []})
    mocker.patch.object(
        git,
        "read_files_from_head",
        return_value={".pivot/stages/my_stage.lock": lock_content.encode()},
    )

    result = metrics.get_metric_info_from_head()

    assert "metrics.json" in result
    assert result["metrics.json"] is None


def test_get_metric_info_from_head_outs_not_list(
    tmp_path: Path,
    set_project_root: Path,
    mocker: MockerFixture,
) -> None:
    """Lock file with non-list 'outs' returns None hash."""
    from pivot import git

    metric_file = tmp_path / "metrics.json"

    _register_metric_stage("my_stage", str(metric_file))

    lock_content = yaml.dump({"outs": "not a list"})
    mocker.patch.object(
        git,
        "read_files_from_head",
        return_value={".pivot/stages/my_stage.lock": lock_content.encode()},
    )

    result = metrics.get_metric_info_from_head()

    assert "metrics.json" in result
    assert result["metrics.json"] is None


def test_collect_metrics_from_head_from_cache(
    tmp_path: Path,
    set_project_root: Path,
) -> None:
    """Collect metrics from cache when hash is available."""
    cache_dir = tmp_path / ".pivot" / "cache" / "files" / "ab"
    cache_dir.mkdir(parents=True)
    cached_file = cache_dir / "c123"
    cached_file.write_text(json.dumps({"accuracy": 0.95}))

    result = metrics.collect_metrics_from_head(
        paths=["metrics.json"],
        head_hashes={"metrics.json": "abc123"},
    )

    assert "metrics.json" in result
    assert result["metrics.json"]["accuracy"] == 0.95


def test_collect_metrics_from_head_from_git(
    tmp_path: Path,
    set_project_root: Path,
    mocker: MockerFixture,
) -> None:
    """Fall back to git when cache miss."""
    from pivot import git

    mocker.patch.object(
        git,
        "read_files_from_head",
        return_value={"metrics.json": b'{"accuracy": 0.88}'},
    )

    result = metrics.collect_metrics_from_head(
        paths=["metrics.json"],
        head_hashes={"metrics.json": None},
    )

    assert "metrics.json" in result
    assert result["metrics.json"]["accuracy"] == 0.88


def test_collect_metrics_from_head_cache_read_error(
    tmp_path: Path,
    set_project_root: Path,
    mocker: MockerFixture,
) -> None:
    """Cache read error falls back to git."""
    from pivot import git

    cache_dir = tmp_path / ".pivot" / "cache" / "files" / "ab"
    cache_dir.mkdir(parents=True)
    cached_file = cache_dir / "c123"
    cached_file.write_text("{invalid json}")

    mocker.patch.object(
        git,
        "read_files_from_head",
        return_value={"metrics.json": b'{"accuracy": 0.77}'},
    )

    result = metrics.collect_metrics_from_head(
        paths=["metrics.json"],
        head_hashes={"metrics.json": "abc123"},
    )

    assert "metrics.json" in result
    assert result["metrics.json"]["accuracy"] == 0.77


def test_collect_metrics_from_head_git_parse_error(
    tmp_path: Path,
    set_project_root: Path,
    mocker: MockerFixture,
) -> None:
    """Git content with parse error is silently skipped."""
    from pivot import git

    mocker.patch.object(
        git,
        "read_files_from_head",
        return_value={"metrics.json": b"{invalid}"},
    )

    result = metrics.collect_metrics_from_head(
        paths=["metrics.json"],
        head_hashes={"metrics.json": None},
    )

    assert "metrics.json" not in result
