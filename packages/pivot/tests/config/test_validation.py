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


_EMPTY_NAME_REASON = (
    "red test (owner TBD): Pipeline accepts an empty or whitespace-only name, "
    + "producing stage names like '/stage'"
)


@pytest.mark.xfail(strict=True, reason=_EMPTY_NAME_REASON)
@pytest.mark.parametrize("name", [pytest.param("", id="empty"), pytest.param("   ", id="spaces")])
def test_blank_pipeline_name_raises_error(tmp_path: pathlib.Path, name: str) -> None:
    """An empty or whitespace-only pipeline name is rejected at definition time."""
    with (
        pytest.raises((exceptions.ValidationError, ValueError)),
        compose.Pipeline(name, root=tmp_path) as pipeline,
    ):
        _helper_reader(pipeline.input("data", path="data/input.csv", t=str))


@pytest.mark.xfail(
    strict=True,
    reason="red test (owner TBD): Pipeline.input accepts paths containing control characters",
)
@pytest.mark.parametrize(
    "path",
    [
        pytest.param("data\x00.csv", id="null-byte"),
        pytest.param("file\nname.csv", id="newline"),
        pytest.param("file\rname.csv", id="carriage-return"),
    ],
)
def test_input_path_with_control_character_raises_error(tmp_path: pathlib.Path, path: str) -> None:
    """Input paths with null bytes or line breaks are rejected as a security error."""
    with (
        pytest.raises(exceptions.SecurityValidationError),
        compose.Pipeline("bad_input", root=tmp_path) as pipeline,
    ):
        _helper_reader(pipeline.input("data", path=path, t=str))
