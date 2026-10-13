import pathlib
from typing import Annotated, TypedDict

import pytest

import pivot
from pivot import compose, exceptions, loaders, outputs, types


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

    with pytest.raises(
        ValueError, match="update.*previous.*incremental"
    ), compose.Pipeline("updates", root=tmp_path) as pipeline:
        update(previous=pipeline.previous())


def test_previous_rejects_unknown_output_key(tmp_path: pathlib.Path) -> None:
    """A previous marker names a real output key rather than silently selecting another one."""
    tag = _incremental_tag()

    @compose.stage
    def update(previous: dict[str, int]) -> Annotated[dict[str, int], tag]:
        return previous

    with pytest.raises(
        ValueError, match="update.*previous.*missing"
    ), compose.Pipeline("updates", root=tmp_path) as pipeline:
        update(previous=pipeline.previous("missing"))


def test_previous_rejects_two_parameters_for_one_output(tmp_path: pathlib.Path) -> None:
    """One prior artifact cannot be injected into two parameters of the same call."""
    tag = _incremental_tag()

    @compose.stage
    def update(first: dict[str, int], second: dict[str, int]) -> Annotated[dict[str, int], tag]:
        return {**first, **second}

    with pytest.raises(
        ValueError, match="update.*first.*second"
    ), compose.Pipeline("updates", root=tmp_path) as pipeline:
        update(first=pipeline.previous(), second=pipeline.previous())


def test_previous_marker_must_be_consumed_by_one_stage_call(tmp_path: pathlib.Path) -> None:
    """A marker outside a stage call is a definition error rather than inert state."""
    _ = _incremental_tag()

    with pytest.raises(
        ValueError, match="previous.*stage call"
    ), compose.Pipeline("updates", root=tmp_path) as pipeline:
        pipeline.previous()


def test_previous_marker_cannot_be_reused_by_another_stage_call(tmp_path: pathlib.Path) -> None:
    """A marker is bound once, preventing a prior value from leaking to a later stage."""
    tag = _incremental_tag()

    @compose.stage
    def update(previous: dict[str, int]) -> Annotated[dict[str, int], tag]:
        return previous

    with pytest.raises(
        ValueError, match="previous.*already.*update"
    ), compose.Pipeline("updates", root=tmp_path) as pipeline:
        previous = pipeline.previous()
        update(previous=previous)
        update(previous=previous)


def test_incremental_output_requires_a_full_loader() -> None:
    """A writer-only format cannot restore a previous value for an incremental output."""
    tag = _incremental_tag()

    def non_loader() -> Annotated[dict[str, int], tag, loaders.MatplotlibFigure()]:
        return {}

    with pytest.raises(
        exceptions.StageDefinitionError, match="non_loader.*_single.*MatplotlibFigure"
    ):
        compose._analyze_return_type(non_loader)


def test_incremental_output_rejects_metric_combination() -> None:
    """An output has one storage policy: incremental data cannot also be a metric."""
    tag = _incremental_tag()

    def metric_incremental() -> Annotated[dict[str, int], tag, compose.metric]:
        return {}

    with pytest.raises(
        exceptions.StageDefinitionError, match="metric_incremental.*_single.*metric"
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
        match=rf"unknown_metadata.*_single.*{class_name}",
    ):
        compose._analyze_return_type(unknown_metadata)
