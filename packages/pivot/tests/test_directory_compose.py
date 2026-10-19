"""How a pipeline built with the ``directory`` return-type tag is wired.

A ``dict[str, T]`` tagged ``directory`` is one output: a directory whose files are the dict's
keys, each written by the output's loader (given, or inferred from ``T``). Indexing the
output's handle by a file name makes a dependency on that one file.
"""

import pathlib
from typing import Annotated, TypedDict

import matplotlib.figure
import pandas as pd
import pytest

import pivot
from pivot import compose, exceptions, loaders, registry, types
from pivot.engine import graph, watch
from pivot.engine.types import StageExecutionState
from pivot.storage import artifact_lock, store


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


@compose.stage
def _helper_count_one(train: pd.DataFrame) -> dict[str, int]:
    return {"rows": len(train)}


@compose.stage
def _helper_read_one(part: dict[str, int]) -> dict[str, int]:
    return part


@compose.stage
def _helper_show_figure(figure: matplotlib.figure.Figure) -> str:
    return str(figure)


class _HelperOnlyParts(TypedDict):
    parts: Annotated[dict[str, dict[str, int]], compose.directory, loaders.JSON[dict[str, int]]()]


@compose.stage
def _helper_only_parts() -> _HelperOnlyParts:
    return {"parts": {"a.json": {"v": 1}}}


_HelperFileNamedField = TypedDict(
    "_HelperFileNamedField",
    {
        "parts": Annotated[
            dict[str, dict[str, int]], compose.directory, loaders.JSON[dict[str, int]]()
        ],
        "parts/a.json": Annotated[dict[str, int], compose.metric],
    },
)


@compose.stage
def _helper_file_named_field() -> _HelperFileNamedField:
    return {"parts": {"a.json": {"v": 1}}, "parts/a.json": {"m": 2}}


def test_directory_takes_an_explicit_loader_for_each_file(tmp_path: pathlib.Path) -> None:
    def split() -> Annotated[dict[str, pd.DataFrame], pivot.directory, loaders.CSV[pd.DataFrame]()]:
        return {}

    with compose.Pipeline("p", root=tmp_path) as pipeline:
        compose.stage(split)()

    [out] = pipeline.get_stage("p/split")["outs"]
    assert out.tag is types.ArtifactTag.DIRECTORY
    assert isinstance(out.format, loaders.CSV)


@pytest.mark.parametrize(
    "base",
    [
        pytest.param(pd.DataFrame, id="dataframe"),
        pytest.param(dict, id="bare-dict"),
        pytest.param(dict[int, str], id="int-keys"),
        pytest.param(list[str], id="list"),
    ],
)
def test_directory_on_anything_but_a_str_keyed_dict_is_refused(
    tmp_path: pathlib.Path, base: type
) -> None:
    def split() -> Annotated[base, pivot.directory]:  # pyright: ignore[reportInvalidTypeForm, reportUnknownParameterType] - parametrized type
        return {}

    with (
        pytest.raises(exceptions.StageDefinitionError) as exc_info,
        compose.Pipeline("p", root=tmp_path),
    ):
        compose.stage(split)()  # pyright: ignore[reportUnknownArgumentType] - parametrized type

    message = str(exc_info.value)
    assert "Stage 'p/split': the return value is tagged directory" in message, message
    assert "dict[str, T]" in message, message


def test_directory_value_type_without_a_known_format_is_refused(tmp_path: pathlib.Path) -> None:
    def split() -> Annotated[dict[str, int], pivot.directory]:
        return {}

    with (
        pytest.raises(ValueError, match="Cannot infer serialization format for type int"),
        compose.Pipeline("p", root=tmp_path),
    ):
        compose.stage(split)()


@pytest.mark.parametrize(
    "other",
    [
        pytest.param(compose.metric, id="metric"),
        pytest.param(compose.plot, id="plot"),
        pytest.param(compose.incremental, id="incremental"),
    ],
)
def test_directory_combined_with_another_tag_is_refused(
    tmp_path: pathlib.Path, other: object
) -> None:
    class Result(TypedDict):
        parts: Annotated[dict[str, dict[str, int]], pivot.directory, other]

    def split() -> Result:
        raise AssertionError("annotations are analyzed without calling the stage")

    with (
        pytest.raises(exceptions.StageDefinitionError) as exc_info,
        compose.Pipeline("p", root=tmp_path),
    ):
        compose.stage(split)()

    message = str(exc_info.value)
    assert "Stage 'p/split': output 'parts' cannot combine directory with" in message, message


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


# =============================================================================
# One file of a directory output as a dependency
# =============================================================================


def _helper_stages(pipeline: compose.Pipeline) -> dict[str, registry.RegistryStageInfo]:
    return {name: pipeline.get_stage(name) for name in pipeline.list_stages()}


def test_indexing_a_directory_handle_selects_one_file(tmp_path: pathlib.Path) -> None:
    """A file of a single directory output and of a directory field each become one dep.

    The dep is placed at that file inside the producer's directory, read with the producer's
    loader, and its key is normalized as the producer's keys are.
    """
    with compose.Pipeline("p", root=tmp_path) as pipeline:
        parts = _helper_split(pipeline.input("raw.csv", t=pd.DataFrame))
        _helper_count_one(parts["sub//train.jsonl"])
        built = _helper_parts()
        _helper_read_one(built["parts"]["a.json"])

    count_dep = pipeline.get_stage("p/_helper_count_one")["deps"]["train"]
    read_dep = pipeline.get_stage("p/_helper_read_one")["deps"]["part"]

    assert count_dep.identity == types.ArtifactIdentity("p/_helper_split", "sub/train.jsonl")
    assert read_dep.identity == types.ArtifactIdentity("p/_helper_parts", "parts/a.json")
    assert store.relative_output_path(count_dep) == pathlib.Path(
        "data/p/_helper_split/sub/train.jsonl"
    )
    assert store.relative_output_path(read_dep) == pathlib.Path("data/p/_helper_parts/parts/a.json")
    assert isinstance(count_dep.format, loaders.DataFrameJSONL)
    assert isinstance(read_dep.format, loaders.JSON)


def test_one_file_consumer_has_an_edge_from_the_producer(tmp_path: pathlib.Path) -> None:
    with compose.Pipeline("p", root=tmp_path) as pipeline:
        parts = _helper_split(pipeline.input("raw.csv", t=pd.DataFrame))
        _helper_count_one(parts["train.jsonl"])
    stages = _helper_stages(pipeline)

    stage_graph = graph.build_graph(stages)

    assert graph.get_upstream_stages(stage_graph, "p/_helper_count_one") == ["p/_helper_split"]
    assert graph.validate_dependency_sources(stages) == []


def test_one_file_consumer_takes_the_lock_the_producer_writes_under(
    tmp_path: pathlib.Path,
) -> None:
    """Reading a file excludes the producer rewriting its directory in another process."""
    with compose.Pipeline("p", root=tmp_path) as pipeline:
        parts = _helper_split(pipeline.input("raw.csv", t=pd.DataFrame))
        _helper_count_one(parts["train.jsonl"])
    producer = pipeline.get_stage("p/_helper_split")
    consumer = pipeline.get_stage("p/_helper_count_one")

    [write] = artifact_lock.expand_lock_requests({}, producer["outs"], tmp_path)
    [read] = artifact_lock.expand_lock_requests(consumer["deps"], [], tmp_path)

    assert (write["mode"], read["mode"]) == (
        artifact_lock.LockMode.WRITE,
        artifact_lock.LockMode.READ,
    )
    assert read["key"] == write["key"]


def test_one_file_of_a_directory_output_routes_as_the_producers_output(
    tmp_path: pathlib.Path,
) -> None:
    """In watch mode the producer writing the file is its own write, not an input change."""
    with compose.Pipeline("p", root=tmp_path) as pipeline:
        parts = _helper_split(pipeline.input("raw.csv", t=pd.DataFrame))
        _helper_count_one(parts["train.jsonl"])
    stages = _helper_stages(pipeline)
    workspace = store.WorkspaceStore(tmp_path, "p", pipeline.input_bindings)
    coordinator = watch.WatchCoordinator(
        graph.build_graph(stages), watch.build_artifact_paths(stages, workspace)
    )
    member = tmp_path / "data" / "p" / "_helper_split" / "train.jsonl"

    assert coordinator.get_producer(member) == "p/_helper_split"
    assert coordinator.should_filter_path(
        member, get_stage_state=lambda _stage: StageExecutionState.RUNNING
    )
    assert coordinator.get_affected_stages([member]) == ["p/_helper_count_one"]


@pytest.mark.parametrize(
    ("key", "reason"),
    [
        pytest.param("../escape.jsonl", "path traversal not allowed", id="traversal"),
        pytest.param("/abs.jsonl", "absolute path not allowed", id="absolute"),
        pytest.param("train", "must include file extension", id="no-extension"),
        pytest.param("a:b.jsonl", "backslashes, colons and null bytes are not allowed", id="colon"),
        pytest.param(
            "a\\b.jsonl", "backslashes, colons and null bytes are not allowed", id="backslash"
        ),
        pytest.param(
            "a\0b.jsonl", "backslashes, colons and null bytes are not allowed", id="null-byte"
        ),
    ],
)
def test_one_file_key_is_checked_at_build(tmp_path: pathlib.Path, key: str, reason: str) -> None:
    with pytest.raises(ValueError) as exc_info, compose.Pipeline("p", root=tmp_path) as pipeline:
        parts = _helper_split(pipeline.input("raw.csv", t=pd.DataFrame))
        _helper_count_one(parts[key])

    message = str(exc_info.value)
    assert "Stage 'p/_helper_count_one'" in message, message
    assert "stage 'p/_helper_split'" in message, message
    assert repr(key) in message, message
    assert reason in message, message


def test_consumer_of_one_write_only_file_is_refused_at_build(tmp_path: pathlib.Path) -> None:
    with (
        pytest.raises(TypeError, match="whose loader MatplotlibFigure cannot read it back"),
        compose.Pipeline("p", root=tmp_path),
    ):
        _helper_show_figure(_helper_figures()["a.png"])


def test_indexing_a_non_directory_output_by_a_file_name_keeps_the_no_such_output_error(
    tmp_path: pathlib.Path,
) -> None:
    with (
        pytest.raises(KeyError, match="has no output 'a.json'"),
        compose.Pipeline("p", root=tmp_path),
    ):
        built = _helper_parts()
        _ = built["summary"]["a.json"]


def test_one_file_handle_has_no_files_of_its_own(tmp_path: pathlib.Path) -> None:
    with (
        pytest.raises(
            KeyError, match="'a.json' is one file of a directory output of stage 'p/_helper_parts'"
        ),
        compose.Pipeline("p", root=tmp_path),
    ):
        built = _helper_parts()
        _ = built["parts"]["a.json"]["b.json"]


def test_a_lone_typed_dict_directory_field_is_selected_by_name_before_its_files(
    tmp_path: pathlib.Path,
) -> None:
    with compose.Pipeline("p", root=tmp_path) as pipeline:
        built = _helper_only_parts()
        _helper_read_parts(built["parts"])
        _helper_read_one(built["parts"]["a.json"])

    whole = pipeline.get_stage("p/_helper_read_parts")["deps"]["parts"]
    one = pipeline.get_stage("p/_helper_read_one")["deps"]["part"]

    assert whole.identity == types.ArtifactIdentity("p/_helper_only_parts", "parts")
    assert one.identity == types.ArtifactIdentity("p/_helper_only_parts", "parts/a.json")


def test_output_named_like_a_file_of_a_directory_output_is_refused_at_build(
    tmp_path: pathlib.Path,
) -> None:
    """`parts/a.json` would name both a field and a file of `parts`, so the build stops."""
    with (
        pytest.raises(exceptions.StageDefinitionError) as exc_info,
        compose.Pipeline("p", root=tmp_path),
    ):
        built = _helper_file_named_field()
        _helper_read_one(built["parts"]["a.json"])
        _helper_read_one(built["parts/a.json"])

    message = str(exc_info.value)
    assert "Stage 'p/_helper_file_named_field'" in message, message
    assert "output 'parts/a.json'" in message, message
    assert "directory output 'parts'" in message, message
