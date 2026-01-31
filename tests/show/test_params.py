from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import yaml

from helpers import register_test_stage
from pivot import stage_def
from pivot.show import params
from pivot.types import ChangeType, OutputFormat

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from pivot.pipeline.pipeline import Pipeline
    from tests.conftest import ValidLockContentFactory


# =============================================================================
# collect_params_from_stages Tests
# =============================================================================


def test_collect_params_from_stages_empty(mock_discovery: Pipeline) -> None:
    """Empty registry returns empty result."""
    result = params.collect_params_from_stages()
    assert result["params"] == {}
    assert result["unknown_stages"] == []


class _MyParams(stage_def.StageParams):
    learning_rate: float = 0.01
    epochs: int = 10


def _helper_params_stage(params: _MyParams) -> None:
    pass


def test_collect_params_from_stages_with_params(mock_discovery: Pipeline) -> None:
    """Collects params from stage with Pydantic model."""
    register_test_stage(
        _helper_params_stage,
        name="train",
        params=_MyParams(),
    )

    result = params.collect_params_from_stages()

    assert "train" in result["params"]
    assert result["params"]["train"]["learning_rate"] == 0.01
    assert result["params"]["train"]["epochs"] == 10
    assert result["unknown_stages"] == []


def _helper_no_params_stage() -> None:
    pass


def test_collect_params_from_stages_without_params(mock_discovery: Pipeline) -> None:
    """Stages without params are not included."""
    register_test_stage(_helper_no_params_stage, name="no_params")

    result = params.collect_params_from_stages()

    assert "no_params" not in result["params"]


class _OverrideParams(stage_def.StageParams):
    learning_rate: float = 0.01


def _helper_override_stage(params: _OverrideParams) -> None:
    pass


def test_collect_params_from_stages_with_overrides(mock_discovery: Pipeline) -> None:
    """Applies params.yaml overrides."""
    register_test_stage(_helper_override_stage, name="train", params=_OverrideParams())

    params_yaml = mock_discovery.root / "params.yaml"
    params_yaml.write_text(yaml.dump({"train": {"learning_rate": 0.05}}))

    result = params.collect_params_from_stages()

    assert result["params"]["train"]["learning_rate"] == 0.05


class _FilterParams(stage_def.StageParams):
    value: int = 1


def _helper_filter_a(params: _FilterParams) -> None:
    pass


def _helper_filter_b(params: _FilterParams) -> None:
    pass


def _helper_filter_c(params: _FilterParams) -> None:
    pass


def test_collect_params_from_stages_filters_by_stage_names(
    mock_discovery: Pipeline,
) -> None:
    """Filters to specific stage names."""
    register_test_stage(_helper_filter_a, name="stage_a", params=_FilterParams())
    register_test_stage(_helper_filter_b, name="stage_b", params=_FilterParams())
    register_test_stage(_helper_filter_c, name="stage_c", params=_FilterParams())

    result = params.collect_params_from_stages(["stage_a", "stage_c"])

    assert "stage_a" in result["params"]
    assert "stage_b" not in result["params"]
    assert "stage_c" in result["params"]


def test_collect_params_from_stages_unknown_stage(
    mock_discovery: Pipeline,
) -> None:
    """Unknown stage name is returned in unknown_stages list."""
    result = params.collect_params_from_stages(["nonexistent"])

    assert "nonexistent" not in result["params"]
    assert result["unknown_stages"] == ["nonexistent"]


# =============================================================================
# get_params_from_head Tests
# =============================================================================


def test_get_params_from_head_no_stages(mock_discovery: Pipeline) -> None:
    """No registered stages returns empty result."""
    result = params.get_params_from_head()
    assert result["params"] == {}
    assert result["git_available"] is True


class _LockParams(stage_def.StageParams):
    lr: float = 0.01


def _helper_lock_stage(params: _LockParams) -> None:
    pass


def test_get_params_from_head_with_lock_file(
    mock_discovery: Pipeline,
    mocker: MockerFixture,
    make_valid_lock_content: ValidLockContentFactory,
) -> None:
    """Returns params from lock file at HEAD."""
    from pivot import git

    register_test_stage(_helper_lock_stage, name="train", params=_LockParams())

    lock_content = yaml.dump(make_valid_lock_content(params={"lr": 0.05, "epochs": 10}))
    mocker.patch.object(
        git,
        "read_files_from_head",
        return_value={".pivot/stages/train.lock": lock_content.encode()},
    )

    result = params.get_params_from_head()

    assert "train" in result["params"]
    assert result["params"]["train"]["lr"] == 0.05
    assert result["params"]["train"]["epochs"] == 10
    assert result["git_available"] is True


def test_get_params_from_head_no_lock_file(
    mock_discovery: Pipeline,
    mocker: MockerFixture,
) -> None:
    """Missing lock file means stage not in result."""
    from pivot import git

    register_test_stage(_helper_lock_stage, name="train", params=_LockParams())
    mocker.patch.object(git, "read_files_from_head", return_value={})
    mocker.patch.object(git, "is_git_repo_with_head", return_value=True)

    result = params.get_params_from_head()

    assert "train" not in result["params"]
    assert result["git_available"] is True


def test_get_params_from_head_no_git_repo(
    mock_discovery: Pipeline,
    mocker: MockerFixture,
) -> None:
    """Returns git_available=False when not in git repo."""
    from pivot import git

    register_test_stage(_helper_lock_stage, name="train", params=_LockParams())
    mocker.patch.object(git, "read_files_from_head", return_value={})
    mocker.patch.object(git, "is_git_repo_with_head", return_value=False)

    result = params.get_params_from_head()

    assert result["params"] == {}
    assert result["git_available"] is False


def test_get_params_from_head_invalid_yaml(
    mock_discovery: Pipeline,
    mocker: MockerFixture,
) -> None:
    """Invalid YAML in lock file is skipped."""
    from pivot import git

    register_test_stage(_helper_lock_stage, name="train", params=_LockParams())
    mocker.patch.object(
        git,
        "read_files_from_head",
        return_value={".pivot/stages/train.lock": b"invalid yaml: {"},
    )

    result = params.get_params_from_head()

    assert "train" not in result["params"]


def test_get_params_from_head_missing_params_key(
    mock_discovery: Pipeline,
    mocker: MockerFixture,
) -> None:
    """Lock file without params key is skipped."""
    from pivot import git

    register_test_stage(_helper_lock_stage, name="train", params=_LockParams())
    lock_content = yaml.dump({"deps": []})
    mocker.patch.object(
        git,
        "read_files_from_head",
        return_value={".pivot/stages/train.lock": lock_content.encode()},
    )

    result = params.get_params_from_head()

    assert "train" not in result["params"]


def test_get_params_from_head_empty_params(
    mock_discovery: Pipeline,
    mocker: MockerFixture,
) -> None:
    """Lock file with empty params dict is skipped."""
    from pivot import git

    register_test_stage(_helper_lock_stage, name="train", params=_LockParams())
    lock_content = yaml.dump({"params": {}})
    mocker.patch.object(
        git,
        "read_files_from_head",
        return_value={".pivot/stages/train.lock": lock_content.encode()},
    )

    result = params.get_params_from_head()

    assert "train" not in result["params"]


def test_get_params_from_head_filters_by_stage_names(
    mock_discovery: Pipeline,
    mocker: MockerFixture,
    make_valid_lock_content: ValidLockContentFactory,
) -> None:
    """Filters to specific stage names."""
    from pivot import git

    register_test_stage(_helper_filter_a, name="stage_a", params=_FilterParams())
    register_test_stage(_helper_filter_b, name="stage_b", params=_FilterParams())

    lock_a = yaml.dump(make_valid_lock_content(params={"value": 1}))
    lock_b = yaml.dump(make_valid_lock_content(params={"value": 2}))
    mocker.patch.object(
        git,
        "read_files_from_head",
        return_value={
            ".pivot/stages/stage_a.lock": lock_a.encode(),
            ".pivot/stages/stage_b.lock": lock_b.encode(),
        },
    )

    result = params.get_params_from_head(["stage_a"])

    assert "stage_a" in result["params"]
    assert "stage_b" not in result["params"]


# =============================================================================
# diff_params Tests
# =============================================================================


def test_diff_params_no_changes() -> None:
    """No changes returns empty list."""
    old = {"train": {"lr": 0.01}}
    new = {"train": {"lr": 0.01}}

    result = params.diff_params(old, new)

    assert result == []


def test_diff_params_modified() -> None:
    """Value change detected as modified."""
    old = {"train": {"lr": 0.01}}
    new = {"train": {"lr": 0.05}}

    result = params.diff_params(old, new)

    assert len(result) == 1
    assert result[0]["change_type"] == "modified"
    assert result[0]["stage"] == "train"
    assert result[0]["key"] == "lr"
    assert result[0]["old"] == 0.01
    assert result[0]["new"] == 0.05


def test_diff_params_added() -> None:
    """New key detected as added."""
    old = {"train": {"lr": 0.01}}
    new = {"train": {"lr": 0.01, "epochs": 10}}

    result = params.diff_params(old, new)

    assert len(result) == 1
    assert result[0]["change_type"] == "added"
    assert result[0]["key"] == "epochs"
    assert result[0]["old"] is None
    assert result[0]["new"] == 10


def test_diff_params_removed() -> None:
    """Missing key detected as removed."""
    old = {"train": {"lr": 0.01, "epochs": 10}}
    new = {"train": {"lr": 0.01}}

    result = params.diff_params(old, new)

    assert len(result) == 1
    assert result[0]["change_type"] == "removed"
    assert result[0]["key"] == "epochs"
    assert result[0]["old"] == 10
    assert result[0]["new"] is None


def test_diff_params_new_stage() -> None:
    """New stage detected."""
    old: dict[str, dict[str, params.ParamValue]] = {}
    new = {"train": {"lr": 0.01}}

    result = params.diff_params(old, new)

    assert len(result) == 1
    assert result[0]["change_type"] == "added"
    assert result[0]["stage"] == "train"


def test_diff_params_removed_stage() -> None:
    """Removed stage detected."""
    old = {"train": {"lr": 0.01}}
    new: dict[str, dict[str, params.ParamValue]] = {}

    result = params.diff_params(old, new)

    assert len(result) == 1
    assert result[0]["change_type"] == "removed"
    assert result[0]["stage"] == "train"


def test_diff_params_nested_values() -> None:
    """Handles nested dict values."""
    old: dict[str, dict[str, params.ParamValue]] = {
        "train": {"optimizer": {"type": "adam", "lr": 0.01}}
    }
    new: dict[str, dict[str, params.ParamValue]] = {
        "train": {"optimizer": {"type": "adam", "lr": 0.05}}
    }

    result = params.diff_params(old, new)

    assert len(result) == 1
    assert result[0]["change_type"] == "modified"
    assert result[0]["key"] == "optimizer"


def test_diff_params_sorted_output() -> None:
    """Results sorted by stage then key."""
    old = {"z_stage": {"b": 1, "a": 2}, "a_stage": {"x": 3}}
    new = {"z_stage": {"b": 10, "a": 20}, "a_stage": {"x": 30}}

    result = params.diff_params(old, new)

    stages = [d["stage"] for d in result]
    assert stages == ["a_stage", "z_stage", "z_stage"]


def test_diff_params_list_values() -> None:
    """Handles list values."""
    old: dict[str, dict[str, params.ParamValue]] = {"train": {"layers": [64, 32, 16]}}
    new: dict[str, dict[str, params.ParamValue]] = {"train": {"layers": [128, 64, 32]}}

    result = params.diff_params(old, new)

    assert len(result) == 1
    assert result[0]["change_type"] == "modified"
    assert result[0]["old"] == [64, 32, 16]
    assert result[0]["new"] == [128, 64, 32]


# =============================================================================
# _values_equal Tests
# =============================================================================


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        pytest.param({"a": 1}, {"a": 1}, True, id="dict_equal"),
        pytest.param({"a": 1}, {"a": 2}, False, id="dict_different"),
        pytest.param({"b": 2, "a": 1}, {"a": 1, "b": 2}, True, id="dict_unordered"),
        pytest.param(1, 1, True, id="int_equal"),
        pytest.param(1, 2, False, id="int_different"),
        pytest.param("a", "a", True, id="str_equal"),
        pytest.param([1, 2, 3], [1, 2, 3], True, id="list_equal"),
        pytest.param([1, 2, 3], [1, 2], False, id="list_different_length"),
        pytest.param(None, None, True, id="none_equal"),
        pytest.param(None, 1, False, id="none_vs_int"),
    ],
)
def test_values_equal(a: params.ParamValue, b: params.ParamValue, expected: bool) -> None:
    """Values compare correctly via JSON normalization."""
    assert params._values_equal(a, b) is expected


# =============================================================================
# format_params_table Tests
# =============================================================================


def test_format_params_table_plain() -> None:
    """Plain format uses tabulate."""
    data = {"train": {"lr": 0.01, "epochs": 10}}

    result = params.format_params_table(data, None, precision=5)

    assert "Stage" in result
    assert "Key" in result
    assert "Value" in result
    assert "train" in result
    assert "lr" in result
    assert "0.01000" in result


def test_format_params_table_json() -> None:
    """JSON format outputs valid JSON."""
    data = {"train": {"lr": 0.01}}

    result = params.format_params_table(data, OutputFormat.JSON, precision=5)

    parsed = json.loads(result)
    assert parsed == {"train": {"lr": 0.01}}


def test_format_params_table_markdown() -> None:
    """Markdown format uses github table style."""
    data = {"train": {"lr": 0.01}}

    result = params.format_params_table(data, OutputFormat.MD, precision=5)

    assert "|" in result
    assert "---" in result


def test_format_params_table_empty() -> None:
    """Empty params shows no params message."""
    result = params.format_params_table({}, None, precision=5)
    assert "No parameters found" in result


# =============================================================================
# format_diff_table Tests
# =============================================================================


def test_format_diff_table_plain() -> None:
    """Plain format for diff."""
    diffs = [
        params.ParamDiff(
            stage="train", key="lr", old=0.01, new=0.05, change_type=ChangeType.MODIFIED
        )
    ]

    result = params.format_diff_table(diffs, None, precision=2)

    assert "Stage" in result
    assert "train" in result
    assert "0.01" in result
    assert "0.05" in result


def test_format_diff_table_empty() -> None:
    """Empty diff shows no changes message."""
    result = params.format_diff_table([], None, precision=5)
    assert "No parameter changes" in result


def test_format_diff_table_json() -> None:
    """JSON format for diff."""
    diffs = [
        params.ParamDiff(
            stage="train", key="lr", old=0.01, new=0.05, change_type=ChangeType.MODIFIED
        )
    ]

    result = params.format_diff_table(diffs, OutputFormat.JSON, precision=2)

    parsed = json.loads(result)
    assert len(parsed) == 1
    assert parsed[0]["change_type"] == "modified"


def test_format_diff_table_markdown() -> None:
    """Markdown format for diff."""
    diffs = [
        params.ParamDiff(
            stage="train", key="lr", old=0.01, new=0.05, change_type=ChangeType.MODIFIED
        )
    ]

    result = params.format_diff_table(diffs, OutputFormat.MD, precision=2)

    assert "|" in result
    assert "---" in result


# =============================================================================
# _format_value Tests
# =============================================================================


@pytest.mark.parametrize(
    ("value", "precision", "expected"),
    [
        pytest.param(0.123456789, 3, "0.123", id="float"),
        pytest.param(42, 5, "42", id="int"),
        pytest.param(None, 5, "-", id="none"),
        pytest.param({"nested": "value"}, 5, '{"nested": "value"}', id="dict"),
        pytest.param([1, 2, 3], 5, "[1, 2, 3]", id="list"),
        pytest.param("hello", 5, "hello", id="str"),
        pytest.param(True, 5, "True", id="bool_true"),
        pytest.param(False, 5, "False", id="bool_false"),
    ],
)
def test_format_value(value: params.ParamValue, precision: int, expected: str) -> None:
    """Values format correctly."""
    assert params._format_value(value, precision) == expected


# =============================================================================
# _apply_precision Tests
# =============================================================================


@pytest.mark.parametrize(
    ("value", "precision", "expected"),
    [
        pytest.param(0.123456789, 3, 0.123, id="float"),
        pytest.param(42, 5, 42, id="int_unchanged"),
        pytest.param(None, 5, None, id="none"),
        pytest.param("hello", 5, "hello", id="str_unchanged"),
        pytest.param(True, 5, True, id="bool_unchanged"),
        pytest.param({"lr": 0.123456789}, 2, {"lr": 0.12}, id="nested_dict"),
        pytest.param([0.123456789, 0.987654321], 2, [0.12, 0.99], id="list"),
        pytest.param(
            {"nested": {"lr": 0.123456789}}, 2, {"nested": {"lr": 0.12}}, id="deep_nested"
        ),
    ],
)
def test_apply_precision(
    value: params.ParamValue, precision: int, expected: params.ParamValue
) -> None:
    """Precision applied correctly to floats."""
    assert params._apply_precision(value, precision) == expected


# =============================================================================
# JSON Precision Tests
# =============================================================================


def test_format_params_table_json_with_precision() -> None:
    """JSON format respects precision for floats."""
    data = {"train": {"lr": 0.123456789}}

    result = params.format_params_table(data, OutputFormat.JSON, precision=2)

    parsed = json.loads(result)
    assert parsed["train"]["lr"] == 0.12


def test_format_diff_table_json_with_precision() -> None:
    """JSON diff format respects precision for floats."""
    diffs = [
        params.ParamDiff(
            stage="train",
            key="lr",
            old=0.123456789,
            new=0.987654321,
            change_type=ChangeType.MODIFIED,
        )
    ]

    result = params.format_diff_table(diffs, OutputFormat.JSON, precision=2)

    parsed = json.loads(result)
    assert parsed[0]["old"] == 0.12
    assert parsed[0]["new"] == 0.99
