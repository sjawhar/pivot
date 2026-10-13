"""Definition-time validation of pipeline names and input paths."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pivot import compose, exceptions

if TYPE_CHECKING:
    import pathlib


@compose.stage
def _helper_reader(data: str) -> str:
    return data


def test_valid_inputs_pass_silently(tmp_path: pathlib.Path) -> None:
    """A pipeline with a valid name and input path registers its stage without errors."""
    with compose.Pipeline("valid", root=tmp_path) as pipeline:
        _helper_reader(pipeline.input("data", path="data/input.csv", t=str))

    assert pipeline.list_stages() == ["valid/_helper_reader"]


@pytest.mark.parametrize("name", [pytest.param("", id="empty"), pytest.param("   ", id="spaces")])
def test_blank_pipeline_name_raises_error(tmp_path: pathlib.Path, name: str) -> None:
    """An empty or whitespace-only pipeline name is rejected at definition time."""
    with (
        pytest.raises((exceptions.ValidationError, ValueError)),
        compose.Pipeline(name, root=tmp_path) as pipeline,
    ):
        _helper_reader(pipeline.input("data", path="data/input.csv", t=str))
