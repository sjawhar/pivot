"""Strict red DirectoryOut restoration tests pending the OPS-421 design decision.

The intended compositional declaration is an Annotated RETURN value. At #452
compose currently lowers DirectoryOut to an ordinary DATA artifact instead of a
DIRECTORY artifact, so each historical restoration scenario stays strict red.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import pytest

from pivot import compose, loaders, outputs, types

if TYPE_CHECKING:
    import pathlib

_DIRECTORY_RETURN = Annotated[
    dict[str, dict[str, int]],
    outputs.DirectoryOut("results/", loaders.JSON[dict[str, int]]()),
]


def _directory_stage() -> _DIRECTORY_RETURN:
    return {"a.json": {"value": 1}, "nested/b.json": {"value": 2}}


_DIRECTORY_SCENARIOS = [
    "cache_hit_restores_empty_directory",
    "cache_hit_restores_partial_directory",
    "cache_hit_removes_extra_files",
    "cache_hit_replaces_wrong_files",
    "h7_corrupted_files_restored_on_cache_hit",
    "h8_corrupted_file_triggers_rerun_when_cache_empty",
    "h9_missing_directory_restored_on_cache_hit",
    "h10_partial_cache_corruption_triggers_rerun",
    "h11_nested_directories_restored_on_cache_hit",
    "h11_nested_file_corrupted_restored",
    "h12_large_directory_partial_restore",
    "h13_perfect_match_skips_without_modification",
    "h14_multiple_directory_outs_restored",
    "h15_run_cache_restores_directory_out",
    "h15b_run_cache_restores_corrupted_directory",
    "h16_missing_cache_triggers_rerun",
]


@pytest.mark.xfail(
    strict=True,
    reason="pending Sami directory-output decision (OPS-421)",
)
@pytest.mark.parametrize("scenario", _DIRECTORY_SCENARIOS, ids=_DIRECTORY_SCENARIOS)
def test_directory_out_behavior_declared_on_return_type(
    scenario: str, tmp_path: pathlib.Path
) -> None:
    """Each historical directory restoration scenario requires DIRECTORY metadata."""
    with compose.Pipeline("directory", root=tmp_path) as pipeline:
        _directory_stage()

    output = pipeline.get_stage("directory/_directory_stage")["outs"][0]
    assert output.tag is types.ArtifactTag.DIRECTORY, scenario
