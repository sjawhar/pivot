from __future__ import annotations

import pathlib
from typing import Annotated, TypedDict

import pytest

from pivot import exceptions, loaders, outputs, registry

# =============================================================================
# Output TypedDicts for annotation-based stages
# =============================================================================


class _DataDirOutput(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("data/", loaders.PathOnly())]


class _DataTrainOutput(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("data/train.csv", loaders.PathOnly())]


class _DataRawOutput(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("data/raw/", loaders.PathOnly())]


class _DataTestOutput(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("data/test.csv", loaders.PathOnly())]


# =============================================================================
# Module-level stage functions (required for fingerprinting)
# =============================================================================


def _helper_stage_data_dir() -> _DataDirOutput:
    return _DataDirOutput(output=pathlib.Path("data/"))


def _helper_stage_data_train() -> _DataTrainOutput:
    return _DataTrainOutput(output=pathlib.Path("data/train.csv"))


def _helper_stage_data_raw() -> _DataRawOutput:
    return _DataRawOutput(output=pathlib.Path("data/raw/"))


def _helper_stage_data_test() -> _DataTestOutput:
    return _DataTestOutput(output=pathlib.Path("data/test.csv"))


# =============================================================================
# Tests
# =============================================================================


def test_directory_output_vs_file_output_conflict() -> None:
    """Should detect conflict when stage outputs directory and another outputs file inside.

    Example:
        Stage A outputs: data/
        Stage B outputs: data/train.csv

    These conflict - data/ contains data/train.csv
    """
    reg = registry.StageRegistry()

    # Stage A outputs directory
    reg.register(_helper_stage_data_dir, name="stage_a")

    # Stage B outputs file inside that directory (registration succeeds, validation deferred)
    reg.register(_helper_stage_data_train, name="stage_b")

    # Output validation is deferred until validate_outputs() is called
    with pytest.raises(exceptions.ValidationError, match="overlap"):
        reg.validate_outputs()


def test_parent_directory_output_vs_child_directory_output() -> None:
    """Should detect conflict when stage outputs dir and another outputs subdir.

    Example:
        Stage A outputs: data/
        Stage B outputs: data/raw/

    These conflict - data/ contains data/raw/
    """
    reg = registry.StageRegistry()

    # Stage A outputs parent directory
    reg.register(_helper_stage_data_dir, name="stage_a")

    # Stage B outputs child directory (registration succeeds, validation deferred)
    reg.register(_helper_stage_data_raw, name="stage_b")

    # Output validation is deferred until validate_outputs() is called
    with pytest.raises(exceptions.ValidationError, match="overlap"):
        reg.validate_outputs()


def test_sibling_file_outputs_no_conflict() -> None:
    """Should allow sibling files in same directory.

    Example:
        Stage A outputs: data/train.csv
        Stage B outputs: data/test.csv

    These should NOT conflict - different files
    """
    reg = registry.StageRegistry()

    # Both output files in same directory - should be fine
    reg.register(_helper_stage_data_train, name="stage_a")
    reg.register(_helper_stage_data_test, name="stage_b")

    assert "stage_a" in reg.list_stages()
    assert "stage_b" in reg.list_stages()
