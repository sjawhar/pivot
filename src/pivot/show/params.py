from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, TypedDict, cast

from pivot import git, parameters
from pivot.show import common
from pivot.types import ChangeType

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pivot.types import OutputFormat

logger = logging.getLogger(__name__)

# Recursive type for JSON-compatible parameter values
type ParamValue = str | int | float | bool | None | list[ParamValue] | dict[str, ParamValue]

# Stage params: {param_key: value}
type StageParams = dict[str, ParamValue]


class ParamDiff(TypedDict):
    """Diff info for a single parameter value."""

    stage: str
    key: str
    old: ParamValue
    new: ParamValue
    change: ChangeType


class CollectResult(TypedDict):
    """Result from collect_params_from_stages."""

    params: dict[str, StageParams]
    unknown_stages: list[str]


def collect_params_from_stages(
    stages: Sequence[str] | None = None,
) -> CollectResult:
    """Collect current effective params for stages.

    Returns CollectResult with params dict and list of unknown stage names.
    """
    from pivot import registry

    result = dict[str, StageParams]()
    unknown_stages = list[str]()
    overrides = parameters.load_params_yaml()

    available_stages = set(registry.REGISTRY.list_stages())
    stage_list = list(stages) if stages else list(available_stages)
    for stage_name in stage_list:
        if stage_name not in available_stages:
            unknown_stages.append(stage_name)
            continue
        stage_info = registry.REGISTRY.get(stage_name)
        effective = parameters.get_effective_params(stage_info["params"], stage_name, overrides)
        if effective:
            result[stage_name] = cast("StageParams", effective)

    return CollectResult(params=result, unknown_stages=unknown_stages)


class HeadResult(TypedDict):
    """Result from get_params_from_head."""

    params: dict[str, StageParams]
    git_available: bool


def get_params_from_head(
    stages: Sequence[str] | None = None,
) -> HeadResult:
    """Read params from lock files at git HEAD.

    Returns HeadResult with params dict and git availability flag.
    """
    from pivot import registry

    result = dict[str, StageParams]()

    stage_list = list(stages) if stages else registry.REGISTRY.list_stages()
    if not stage_list:
        return HeadResult(params=result, git_available=True)

    lock_data_map = common.read_lock_files_from_head(stage_list)

    # If all lock files are None, check if git is available
    git_available = (
        any(v is not None for v in lock_data_map.values()) or git.is_git_repo_with_head()
    )

    for stage_name in stage_list:
        lock_data = lock_data_map.get(stage_name)
        if lock_data is None:
            continue

        if "params" not in lock_data:
            continue

        raw_params = lock_data["params"]
        if raw_params:
            result[stage_name] = cast("StageParams", raw_params)

    return HeadResult(params=result, git_available=git_available)


def diff_params(
    old: Mapping[str, Mapping[str, ParamValue]],
    new: Mapping[str, Mapping[str, ParamValue]],
) -> list[ParamDiff]:
    """Compare old vs new params. Returns list of diffs."""
    diffs = list[ParamDiff]()
    all_stages = set(old.keys()) | set(new.keys())

    for stage in sorted(all_stages):
        old_params = old.get(stage, {})
        new_params = new.get(stage, {})
        all_keys = set(old_params.keys()) | set(new_params.keys())

        for key in sorted(all_keys):
            old_val = old_params.get(key)
            new_val = new_params.get(key)

            if key not in old_params:
                diffs.append(
                    ParamDiff(stage=stage, key=key, old=None, new=new_val, change=ChangeType.ADDED)
                )
            elif key not in new_params:
                diffs.append(
                    ParamDiff(
                        stage=stage, key=key, old=old_val, new=None, change=ChangeType.REMOVED
                    )
                )
            elif not _values_equal(old_val, new_val):
                diffs.append(
                    ParamDiff(
                        stage=stage, key=key, old=old_val, new=new_val, change=ChangeType.MODIFIED
                    )
                )

    return diffs


def _values_equal(a: ParamValue, b: ParamValue) -> bool:
    """Compare values using JSON serialization for consistency."""
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _apply_precision(value: ParamValue, precision: int) -> ParamValue:
    """Recursively apply precision to float values."""
    if isinstance(value, float):
        return round(value, precision)
    if isinstance(value, dict):
        return {k: _apply_precision(v, precision) for k, v in value.items()}
    if isinstance(value, list):
        return [_apply_precision(v, precision) for v in value]
    return value


def format_params_table(
    params: Mapping[str, Mapping[str, ParamValue]],
    output_format: OutputFormat,
    precision: int = 5,
) -> str:
    """Format params for display. output_format: None (plain), 'json', or 'md'."""
    if output_format == "json":
        rounded = {
            stage: {k: _apply_precision(v, precision) for k, v in stage_params.items()}
            for stage, stage_params in params.items()
        }
        return json.dumps(rounded, indent=2)

    rows = list[list[str]]()
    for stage, stage_params in sorted(params.items()):
        for key, value in sorted(stage_params.items()):
            rows.append([stage, key, _format_value(value, precision)])

    return common.format_table(
        rows, ["Stage", "Key", "Value"], output_format, "No parameters found."
    )


def format_diff_table(
    diffs: list[ParamDiff],
    output_format: OutputFormat,
    precision: int = 5,
) -> str:
    """Format param diffs for display."""
    if output_format == "json":
        rounded_diffs = [
            ParamDiff(
                stage=d["stage"],
                key=d["key"],
                old=_apply_precision(d["old"], precision),
                new=_apply_precision(d["new"], precision),
                change=d["change"],
            )
            for d in diffs
        ]
        return json.dumps(rounded_diffs, indent=2)

    rows = list[list[str]]()
    for diff in diffs:
        rows.append(
            [
                diff["stage"],
                diff["key"],
                _format_value(diff["old"], precision),
                _format_value(diff["new"], precision),
                diff["change"],
            ]
        )

    return common.format_table(
        rows, ["Stage", "Key", "Old", "New", "Change"], output_format, "No parameter changes."
    )


def _format_value(value: ParamValue, precision: int) -> str:
    """Format value with precision for floats, '-' for None."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    if isinstance(value, dict | list):
        return json.dumps(value)
    return str(value)
