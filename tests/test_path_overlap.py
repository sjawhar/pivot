"""Tests to validate path overlap handling.

These tests check if our current implementation correctly handles:
1. Directory outputs vs file outputs (e.g., data/ vs data/train.csv)
2. File dependencies on directory outputs
3. Overlapping output paths
"""

import pytest

from pivot import registry


def test_directory_output_vs_file_output_conflict() -> None:
    """Should detect conflict when stage outputs directory and another outputs file inside.

    Example:
        Stage A outputs: data/
        Stage B outputs: data/train.csv

    These conflict - data/ contains data/train.csv
    """
    reg = registry.StageRegistry()

    def stage_a() -> None:
        pass

    def stage_b() -> None:
        pass

    # Stage A outputs directory
    reg.register(stage_a, name="stage_a", deps=[], outs=["data/"])

    # Stage B outputs file inside that directory - should conflict
    with pytest.raises(registry.ValidationError, match="conflict|overlap"):
        reg.register(stage_b, name="stage_b", deps=[], outs=["data/train.csv"])


def test_parent_directory_output_vs_child_directory_output() -> None:
    """Should detect conflict when stage outputs dir and another outputs subdir.

    Example:
        Stage A outputs: data/
        Stage B outputs: data/raw/

    These conflict - data/ contains data/raw/
    """
    reg = registry.StageRegistry()

    def stage_a() -> None:
        pass

    def stage_b() -> None:
        pass

    # Stage A outputs parent directory
    reg.register(stage_a, name="stage_a", deps=[], outs=["data/"])

    # Stage B outputs child directory - should conflict
    with pytest.raises(registry.ValidationError, match="conflict|overlap"):
        reg.register(stage_b, name="stage_b", deps=[], outs=["data/raw/"])


def test_sibling_file_outputs_no_conflict() -> None:
    """Should allow sibling files in same directory.

    Example:
        Stage A outputs: data/train.csv
        Stage B outputs: data/test.csv

    These should NOT conflict - different files
    """
    reg = registry.StageRegistry()

    def stage_a() -> None:
        pass

    def stage_b() -> None:
        pass

    # Both output files in same directory - should be fine
    reg.register(stage_a, name="stage_a", deps=[], outs=["data/train.csv"])
    reg.register(stage_b, name="stage_b", deps=[], outs=["data/test.csv"])

    assert "stage_a" in reg.list_stages()
    assert "stage_b" in reg.list_stages()
