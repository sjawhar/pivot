"""Build-time behavior of the ``directory`` return-type tag.

A ``dict[str, T]`` tagged ``directory`` is one output: a directory whose files are the dict's
keys, each written by the output's loader (given, or inferred from ``T``).
"""

import pathlib
from typing import Annotated, TypedDict

import matplotlib.figure
import pandas as pd
import pytest

import pivot
from pivot import compose, exceptions, loaders, types
from pivot.storage import store


def _directory_tag() -> object:
    tag = getattr(pivot, "directory", None)
    assert tag is not None, "pivot.directory must be public"
    return tag


@compose.stage
def _helper_split(raw: pd.DataFrame) -> Annotated[dict[str, pd.DataFrame], compose.directory]:
    return {"train.jsonl": raw}


@compose.stage
def _helper_count(parts: dict[str, pd.DataFrame]) -> dict[str, int]:
    return {name: len(frame) for name, frame in parts.items()}


class _HelperParts(TypedDict):
    parts: Annotated[dict[str, dict[str, int]], compose.directory, loaders.JSON[dict[str, int]]()]
    summary: Annotated[dict[str, int], compose.metric]


@compose.stage
def _helper_parts() -> _HelperParts:
    return {"parts": {"a.json": {"v": 1}}, "summary": {"files": 1}}


@compose.stage
def _helper_read_parts(parts: dict[str, dict[str, int]]) -> dict[str, int]:
    return {name: value["v"] for name, value in parts.items()}


def test_directory_is_exported_from_pivot() -> None:
    assert _directory_tag() is compose.directory


@pytest.mark.parametrize(
    ("value_type", "loader_class"),
    [
        pytest.param(pd.DataFrame, loaders.DataFrameJSONL, id="dataframe"),
        pytest.param(dict[str, int], loaders.YAML, id="dict"),
        pytest.param(str, loaders.Text, id="str"),
        pytest.param(pathlib.Path, loaders.PathOnly, id="path"),
    ],
)
def test_directory_loader_is_inferred_from_the_value_type(
    value_type: type, loader_class: type
) -> None:
    """Each file is written as a single output of the value type would be."""
    tag = _directory_tag()

    def split() -> Annotated[dict[str, value_type], tag]:  # pyright: ignore[reportInvalidTypeForm, reportUnknownParameterType] - parametrized type
        return {}

    [spec] = compose._analyze_return_type(split)  # pyright: ignore[reportUnknownArgumentType] - parametrized type

    assert spec.tag is tag
    assert type(spec.format) is loader_class


def test_directory_takes_an_explicit_loader_for_each_file() -> None:
    tag = _directory_tag()

    def split() -> Annotated[dict[str, pd.DataFrame], tag, loaders.CSV[pd.DataFrame]()]:
        return {}

    [spec] = compose._analyze_return_type(split)

    assert isinstance(spec.format, loaders.CSV)


def test_untagged_dict_of_values_stays_one_yaml_file() -> None:
    def split() -> dict[str, pd.DataFrame]:
        return {}

    [spec] = compose._analyze_return_type(split)

    assert spec.tag is None
    assert isinstance(spec.format, loaders.YAML)


@pytest.mark.parametrize(
    "base",
    [
        pytest.param(pd.DataFrame, id="dataframe"),
        pytest.param(dict, id="bare-dict"),
        pytest.param(dict[int, str], id="int-keys"),
        pytest.param(list[str], id="list"),
    ],
)
def test_directory_on_anything_but_a_str_keyed_dict_is_refused(base: type) -> None:
    tag = _directory_tag()

    def split() -> Annotated[base, tag]:  # pyright: ignore[reportInvalidTypeForm, reportUnknownParameterType] - parametrized type
        return {}

    with pytest.raises(exceptions.StageDefinitionError) as exc_info:
        compose._analyze_return_type(split)  # pyright: ignore[reportUnknownArgumentType] - parametrized type

    message = str(exc_info.value)
    assert "Stage 'split': the return value is tagged directory" in message, message
    assert "dict[str, T]" in message, message


def test_directory_value_type_without_a_known_format_is_refused() -> None:
    tag = _directory_tag()

    def split() -> Annotated[dict[str, int], tag]:
        return {}

    with pytest.raises(ValueError, match="Cannot infer serialization format for type int"):
        compose._analyze_return_type(split)


@pytest.mark.parametrize(
    "other",
    [
        pytest.param(compose.metric, id="metric"),
        pytest.param(compose.plot, id="plot"),
        pytest.param(compose.incremental, id="incremental"),
    ],
)
def test_directory_combined_with_another_tag_is_refused(other: object) -> None:
    tag = _directory_tag()

    class Result(TypedDict):
        parts: Annotated[dict[str, dict[str, int]], tag, other]

    def split() -> Result:
        raise AssertionError("annotations are analyzed without calling the stage")

    with pytest.raises(exceptions.StageDefinitionError) as exc_info:
        compose._analyze_return_type(split)

    message = str(exc_info.value)
    assert "Stage 'split': output 'parts' cannot combine directory with" in message, message


def test_typed_dict_directory_field_is_a_directory_artifact(tmp_path: pathlib.Path) -> None:
    """The producer's out and the consumer's dep carry the directory tag and the field's loader."""
    with compose.Pipeline("p", root=tmp_path) as pipeline:
        built = _helper_parts()
        _helper_read_parts(built["parts"])

    outs = {out.identity.key: out for out in pipeline.get_stage("p/_helper_parts")["outs"]}
    dep = pipeline.get_stage("p/_helper_read_parts")["deps"]["parts"]

    assert outs["parts"].tag is types.ArtifactTag.DIRECTORY
    assert outs["summary"].tag is types.ArtifactTag.METRIC
    assert dep.tag is types.ArtifactTag.DIRECTORY
    assert isinstance(dep.format, loaders.JSON)


def test_single_directory_output_is_placed_without_an_extension(tmp_path: pathlib.Path) -> None:
    with compose.Pipeline("p", root=tmp_path) as pipeline:
        parts = _helper_split(pipeline.input("raw.csv", t=pd.DataFrame))
        _helper_count(parts)

    [out] = pipeline.get_stage("p/_helper_split")["outs"]

    assert out.tag is types.ArtifactTag.DIRECTORY
    assert pipeline.get_stage("p/_helper_count")["deps"]["parts"].tag is out.tag
    assert store.relative_output_path(out) == pathlib.Path("data/p/_helper_split")


@compose.stage
def _helper_figures() -> Annotated[
    dict[str, matplotlib.figure.Figure], compose.directory, loaders.MatplotlibFigure()
]:
    return {}


def test_directory_of_write_only_files_is_a_valid_leaf_output(tmp_path: pathlib.Path) -> None:
    """A directory of PNGs needs no reader while nothing consumes it."""
    with compose.Pipeline("p", root=tmp_path) as pipeline:
        _helper_figures()

    [out] = pipeline.get_stage("p/_helper_figures")["outs"]

    assert isinstance(out.format, loaders.MatplotlibFigure)
