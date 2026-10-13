import pathlib
from typing import Annotated, TypedDict

import pytest

import pivot
from pivot import compose, exceptions, loaders, outputs, project, types


def _incremental_tag() -> object:
    tag = getattr(pivot, "incremental", None)
    assert tag is not None, "pivot.incremental must be public"
    return tag


def test_incremental_output_registers_its_previous_input_without_a_dag_edge(
    tmp_path: pathlib.Path,
) -> None:
    """An incremental output supplies its own preceding result, not a dependency."""
    tag = _incremental_tag()

    @compose.stage
    def accumulate(previous: dict[str, int]) -> Annotated[dict[str, int], tag]:
        return {"count": previous.get("count", 0) + 1}

    assert accumulate({}) == {"count": 1}

    with compose.Pipeline("updates", root=tmp_path) as pipeline:
        accumulate(previous=pipeline.previous())

    info = pipeline.get_stage("updates/accumulate")
    output = info["outs"][0]
    assert output.tag is types.ArtifactTag.INCREMENTAL
    assert info["incremental_inputs"] == {"previous": output.identity}
    assert list(pipeline.build_dag().edges()) == []


def test_uncached_incremental_output_is_detected_from_artifact_refs(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing incremental workspace artifact without a lock hash is guarded."""
    tag = _incremental_tag()

    @compose.stage
    def update(previous: dict[str, int]) -> Annotated[dict[str, int], tag]:
        return previous

    with compose.Pipeline("updates", root=tmp_path) as pipeline:
        update(previous=pipeline.previous())

    stage_name = "updates/update"
    output_path = tmp_path / "data" / "updates" / "update.yaml"
    output_path.parent.mkdir(parents=True)
    output_path.write_text("count: 1\n")
    monkeypatch.setattr(project, "get_project_root", lambda: tmp_path)

    from pivot.executor import core

    assert core.check_uncached_incremental_outputs(
        [stage_name], {stage_name: pipeline.get_stage(stage_name)}
    ) == [(stage_name, str(output_path))]


def test_previous_key_selects_an_incremental_typed_dict_output(tmp_path: pathlib.Path) -> None:
    """A named previous marker injects exactly its stage's tagged TypedDict field."""
    tag = _incremental_tag()

    class Outputs(TypedDict):
        cache: Annotated[dict[str, int], tag]
        report: dict[str, int]

    @compose.stage
    def update(previous_cache: dict[str, int]) -> Outputs:
        return Outputs(cache=previous_cache, report={"rows": len(previous_cache)})

    with compose.Pipeline("updates", root=tmp_path) as pipeline:
        update(previous_cache=pipeline.previous("cache"))

    info = pipeline.get_stage("updates/update")
    cache = next(out for out in info["outs"] if out.identity.key == "cache")
    assert info["incremental_inputs"] == {"previous_cache": cache.identity}


def test_previous_requires_an_incremental_output(tmp_path: pathlib.Path) -> None:
    """Binding previous to an ordinary output must fail during pipeline definition."""
    tag = _incremental_tag()
    _ = tag

    @compose.stage
    def update(previous: dict[str, int]) -> dict[str, int]:
        return previous

    with (
        pytest.raises(ValueError, match="update.*previous.*incremental"),
        compose.Pipeline("updates", root=tmp_path) as pipeline,
    ):
        update(previous=pipeline.previous())


def test_previous_rejects_unknown_output_key(tmp_path: pathlib.Path) -> None:
    """A previous marker names a real output key rather than silently selecting another one."""
    tag = _incremental_tag()

    @compose.stage
    def update(previous: dict[str, int]) -> Annotated[dict[str, int], tag]:
        return previous

    with (
        pytest.raises(ValueError, match="update.*previous.*missing"),
        compose.Pipeline("updates", root=tmp_path) as pipeline,
    ):
        update(previous=pipeline.previous("missing"))


def test_previous_rejects_two_parameters_for_one_output(tmp_path: pathlib.Path) -> None:
    """One prior artifact cannot be injected into two parameters of the same call."""
    tag = _incremental_tag()

    @compose.stage
    def update(first: dict[str, int], second: dict[str, int]) -> Annotated[dict[str, int], tag]:
        return {**first, **second}

    with (
        pytest.raises(ValueError, match="update.*first.*second"),
        compose.Pipeline("updates", root=tmp_path) as pipeline,
    ):
        update(first=pipeline.previous(), second=pipeline.previous())


def test_previous_marker_must_be_consumed_by_one_stage_call(tmp_path: pathlib.Path) -> None:
    """A marker outside a stage call is a definition error rather than inert state."""
    _ = _incremental_tag()

    with (
        pytest.raises(ValueError, match="previous.*stage call"),
        compose.Pipeline("updates", root=tmp_path) as pipeline,
    ):
        pipeline.previous()


def test_previous_marker_cannot_be_reused_by_another_stage_call(tmp_path: pathlib.Path) -> None:
    """A marker is bound once, preventing a prior value from leaking to a later stage."""
    tag = _incremental_tag()

    @compose.stage
    def update(previous: dict[str, int]) -> Annotated[dict[str, int], tag]:
        return previous

    with (
        pytest.raises(ValueError, match="previous.*already.*update"),
        compose.Pipeline("updates", root=tmp_path) as pipeline,
    ):
        previous = pipeline.previous()
        update(previous=previous)
        update(previous=previous)


def test_incremental_output_requires_a_full_loader() -> None:
    """A writer-only format cannot restore a previous value for an incremental output."""
    tag = _incremental_tag()

    def non_loader() -> Annotated[dict[str, int], tag, loaders.MatplotlibFigure()]:
        return {}

    with pytest.raises(
        exceptions.StageDefinitionError, match="non_loader.*the return value.*MatplotlibFigure"
    ):
        compose._analyze_return_type(non_loader)


@pytest.mark.parametrize(
    "loader",
    [
        pytest.param(loaders.PathOnly(), id="path-only"),
        pytest.param(loaders.Pickle[object](), id="pickle"),
    ],
)
def test_incremental_output_requires_a_loader_with_an_empty_value(loader: object) -> None:
    """A Loader that inherits the raising default cannot initialize an incremental output."""
    tag = _incremental_tag()

    def unsupported_empty_loader() -> Annotated[pathlib.Path, tag, loader]:
        raise AssertionError("annotations are analyzed without calling the stage")

    with pytest.raises(
        exceptions.StageDefinitionError,
        match=rf"unsupported_empty_loader.*the return value.*{type(loader).__name__}",
    ):
        compose._analyze_return_type(unsupported_empty_loader)


def test_incremental_output_rejects_metric_combination() -> None:
    """An output has one storage policy: incremental data cannot also be a metric."""
    tag = _incremental_tag()

    def metric_incremental() -> Annotated[dict[str, int], tag, compose.metric]:
        return {}

    with pytest.raises(
        exceptions.StageDefinitionError, match="metric_incremental.*the return value.*metric"
    ):
        compose._analyze_return_type(metric_incremental)


class Dep:
    pass


@pytest.mark.parametrize(
    ("metadata", "class_name"),
    [
        pytest.param(
            outputs.DirectoryOut("directory/", loaders.JSON[dict[str, int]]()),
            "DirectoryOut",
            id="directory-output",
        ),
        pytest.param(Dep(), "Dep", id="legacy-dependency"),
        pytest.param(object(), "object", id="unknown-object"),
    ],
)
def test_return_type_rejects_unknown_annotated_metadata(metadata: object, class_name: str) -> None:
    """Unknown return metadata is rejected instead of changing output semantics silently."""

    def unknown_metadata() -> Annotated[dict[str, int], metadata]:
        return {}

    with pytest.raises(
        exceptions.StageDefinitionError,
        match=rf"unknown_metadata.*the return value.*{class_name}",
    ):
        compose._analyze_return_type(unknown_metadata)


def test_directory_out_on_return_value_names_it_and_suggests_a_path() -> None:
    """The single output is 'the return value' to users, and a directory is a returned Path."""

    def split() -> Annotated[
        dict[str, dict[str, int]],
        outputs.DirectoryOut("parts/", loaders.YAML[dict[str, int]]()),
    ]:
        return {}

    with pytest.raises(exceptions.StageDefinitionError) as exc_info:
        compose._analyze_return_type(split)

    message = str(exc_info.value)
    assert compose.SINGLE_OUTPUT_KEY not in message, message
    assert (
        "Stage 'split': the return value has unsupported Annotated metadata 'DirectoryOut'."
        in message
    ), message
    assert "return a pathlib.Path" in message, message


class _SplitResult(TypedDict):
    parts: Annotated[
        dict[str, dict[str, int]],
        outputs.DirectoryOut("parts/", loaders.YAML[dict[str, int]]()),
    ]


def test_directory_out_on_typed_dict_field_names_the_field_and_suggests_a_path() -> None:
    """A TypedDict field keeps its own name in the error and gets the same next step."""

    def split() -> _SplitResult:
        raise AssertionError("annotations are analyzed without calling the stage")

    with pytest.raises(exceptions.StageDefinitionError) as exc_info:
        compose._analyze_return_type(split)

    message = str(exc_info.value)
    assert "Stage 'split': output 'parts' has unsupported Annotated metadata" in message, message
    assert "return a pathlib.Path" in message, message
