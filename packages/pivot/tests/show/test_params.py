# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportMissingTypeArgument=false, reportAny=false, reportExplicitAny=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUntypedFunctionDecorator=false, reportUnusedParameter=false, reportUnusedCallResult=false
from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

from pivot.show import params
from pivot.types import ChangeType, OutputFormat

if TYPE_CHECKING:
    from pivot.show.params import ParamValue


def test_diff_params_reports_added_removed_and_modified() -> None:
    old = {"train": {"lr": 0.1, "batch": 32}, "eval": {"threshold": 0.5}}
    new = {"train": {"lr": 0.2, "dropout": 0.1}, "score": {"topk": 5}}

    diffs = params.diff_params(old, new)

    by_key = {(d["stage"], d["key"]): d["change_type"] for d in diffs}
    assert by_key[("train", "lr")] == ChangeType.MODIFIED
    assert by_key[("train", "batch")] == ChangeType.REMOVED
    assert by_key[("train", "dropout")] == ChangeType.ADDED
    assert by_key[("eval", "threshold")] == ChangeType.REMOVED
    assert by_key[("score", "topk")] == ChangeType.ADDED


def test_format_params_table_json_rounds_float_precision() -> None:
    rendered = params.format_params_table(
        {"train": {"lr": 0.1234567, "flags": [1.234567], "nested": {"v": 2.987654}}},
        OutputFormat.JSON,
        precision=3,
    )

    payload = json.loads(rendered)
    assert payload["train"]["lr"] == 0.123
    assert payload["train"]["flags"][0] == 1.235
    assert payload["train"]["nested"]["v"] == 2.988


def test_format_diff_table_json_rounds_old_and_new_values() -> None:
    diffs = [
        params.ParamDiff(
            stage="train",
            key="lr",
            old=0.123456,
            new=0.678912,
            change_type=ChangeType.MODIFIED,
        )
    ]

    rendered = params.format_diff_table(diffs, OutputFormat.JSON, precision=4)
    payload = json.loads(rendered)

    assert payload[0]["old"] == 0.1235
    assert payload[0]["new"] == 0.6789
    assert payload[0]["change_type"] == ChangeType.MODIFIED


def test_values_equal_normalizes_key_order() -> None:
    left = {"a": 1, "b": {"x": 1, "y": 2}}
    right = {"b": {"y": 2, "x": 1}, "a": 1}

    assert params._values_equal(cast("ParamValue", left), cast("ParamValue", right))


def test_format_value_handles_none_float_and_nested_values() -> None:
    assert params._format_value(None, precision=2) == "-"
    assert params._format_value(1.2345, precision=2) == "1.23"
    assert params._format_value({"a": [1, 2]}, precision=2) == '{"a": [1, 2]}'
