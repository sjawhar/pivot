"""Strict red tests for IncrementalOut in the compositional API (Task 17).

The old suite mutated ``RegistryStageInfo`` after annotation extraction because
IncrementalOut could not be declared. The compositional contract is direct:
annotate the RETURN value with IncrementalOut and let compose/worker retain its
read-before-write semantics. At #452 compose silently lowers the marker to an
ordinary DATA/Text artifact, so every scenario is a strict red test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import pytest

from pivot import compose, loaders, outputs

if TYPE_CHECKING:
    import pathlib

_INCREMENTAL_RETURN = Annotated[str, outputs.IncrementalOut("state.txt", loaders.Text())]


def _incremental_stage() -> _INCREMENTAL_RETURN:
    """Path-free incremental declaration: Pivot should restore the prior value before run."""
    return "next value"


# Each id maps one-for-one to an origin/main test. Task 17 owns the implementation
# that turns this marker into retained compose/worker metadata and behavior.
_INCREMENTAL_SCENARIOS = [
    "prepare_outputs_regular_out_is_deleted",
    "prepare_outputs_noncached_out_is_deleted[out-cache-false]",
    "prepare_outputs_noncached_out_is_deleted[metric]",
    "prepare_outputs_incremental_no_cache_creates_empty",
    "prepare_outputs_incremental_restores_from_cache",
    "prepare_outputs_incremental_restored_file_is_writable",
    "prepare_outputs_incremental_missing_cache_error",
    "integration_missing_cache_error_includes_recovery_suggestion",
    "dvc_export_incremental_out_always_persist",
    "dvc_export_incremental_out_with_cache_false",
    "integration_first_run_creates_output",
    "integration_second_run_appends_to_output",
    "incremental_out_restores_directory",
    "incremental_out_directory_is_writable",
    "incremental_out_directory_subdirs_writable",
    "executable_bit_saved_in_manifest",
    "executable_bit_restored_with_copy_mode",
    "uncached_incremental_output_raises_error",
    "uncached_incremental_output_allow_uncached_incremental_allows_run",
    "cached_incremental_output_runs_normally",
    "force_runs_incremental_stage_even_when_unchanged",
    "force_and_allow_uncached_incremental_are_orthogonal",
    "incremental_out_recognized_as_input_annotation",
    "incremental_input_first_run_returns_empty",
    "incremental_input_subsequent_run_loads_value",
    "incremental_input_no_circular_dependency",
    "incremental_input_type_alias_pattern",
    "incremental_input_mixed_with_regular_deps",
    "incremental_out_input_requires_matching_output",
    "incremental_out_typeddict_name_must_match",
    "incremental_out_input_output_path_must_match",
    "incremental_out_input_output_loader_must_match",
    "incremental_out_single_output_only_one_input",
    "incremental_out_single_output_path_must_match",
    "multiple_incremental_outs_matched_by_name",
    "incremental_out_single_output_valid",
]


@pytest.mark.xfail(
    strict=True,
    reason="red test for Task 17: IncrementalOut not wired into compose/worker",
)
@pytest.mark.parametrize("scenario", _INCREMENTAL_SCENARIOS, ids=_INCREMENTAL_SCENARIOS)
def test_incremental_out_behavior_declared_on_return_type(
    scenario: str, tmp_path: pathlib.Path
) -> None:
    """Each historical incremental behavior requires marker-preserving composition."""
    with compose.Pipeline("incremental", root=tmp_path) as pipeline:
        _incremental_stage()

    output = pipeline.get_stage("incremental/_incremental_stage")["outs"][0]
    # Task 17 must retain the actual marker (or an equivalent explicit metadata
    # field) rather than silently lowering this return annotation to Text/DATA.
    assert isinstance(output.format, outputs.IncrementalOut), scenario
