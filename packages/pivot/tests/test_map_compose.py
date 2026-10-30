"""How ``pipeline.map`` over a directory output is wired.

``pipeline.map(handle)`` stands for one file of a directory output, the file of each key. A stage
called with it is a template: the registry lists it apart from the stages, and a run adds one
instance per key, ``<stage>@<key>``. A stage called with a template's handle is a template over
the same keys.
"""

import pathlib
from collections.abc import Callable
from typing import Annotated, TypedDict

import pytest

from pivot import compose, exceptions, loaders, stage_def, types


@compose.stage
def _helper_plan(
    raw: dict[str, int],
) -> Annotated[dict[str, dict[str, int]], compose.directory, loaders.YAML[dict[str, int]]()]:
    return {f"{name}.yaml": {"value": value} for name, value in raw.items()}


@compose.stage
def _helper_cut(entry: dict[str, int]) -> dict[str, int]:
    return entry


@compose.stage
def _helper_build(cut: dict[str, int]) -> dict[str, int]:
    return cut


@compose.stage
def _helper_cut_with(entry: dict[str, int], config: dict[str, int]) -> dict[str, int]:
    return {**entry, **config}


@compose.stage
def _helper_value() -> dict[str, int]:
    return {"value": 1}


class _HelperPlanParts(TypedDict):
    parts: Annotated[dict[str, dict[str, int]], compose.directory, loaders.YAML[dict[str, int]]()]
    summary: Annotated[dict[str, int], compose.metric]


@compose.stage
def _helper_plan_parts() -> _HelperPlanParts:
    return {"parts": {"a.yaml": {"v": 1}}, "summary": {"files": 1}}


@compose.stage
def _helper_combine(cuts: list[dict[str, int]]) -> dict[str, int]:
    return {"n": len(cuts)}


class _HelperCutParams(stage_def.StageParams):
    factor: int = 1


@compose.stage
def _helper_cut_params(entry: dict[str, int], params: _HelperCutParams) -> dict[str, int]:
    return {"value": entry["value"] * params.factor}


def _plan_input(pipeline: compose.Pipeline) -> compose.ArtifactHandle:
    return pipeline.input("plan.yaml")


# =============================================================================
# Templates
# =============================================================================


def test_a_stage_called_with_a_mapped_handle_is_a_template_not_a_stage(
    tmp_path: pathlib.Path,
) -> None:
    with compose.Pipeline("p", root=tmp_path) as pipeline:
        plan = _helper_plan(_plan_input(pipeline))
        _helper_cut(pipeline.map(plan))

    assert pipeline.list_stages() == ["p/_helper_plan"]
    assert pipeline.list_templates() == ["p/_helper_cut"]
    template = pipeline.get_template("p/_helper_cut")
    assert template["source"] == types.ArtifactIdentity("p/_helper_plan", None)
    assert template["mapped_deps"] == {"entry": None}
    assert template["info"]["name"] == "p/_helper_cut"


def test_a_template_reads_its_source_directory_until_expanded(tmp_path: pathlib.Path) -> None:
    """The declared dependency is the whole directory, so the DAG orders it after its producer."""
    with compose.Pipeline("p", root=tmp_path) as pipeline:
        plan = _helper_plan(_plan_input(pipeline))
        _helper_cut(pipeline.map(plan))

    dep = pipeline.get_template("p/_helper_cut")["info"]["deps"]["entry"]
    assert dep.identity == types.ArtifactIdentity("p/_helper_plan", None)
    assert dep.tag is types.ArtifactTag.DIRECTORY
    assert ("p/_helper_cut", "p/_helper_plan") in pipeline.build_dag().edges()


def test_a_stage_called_with_a_templates_handle_is_a_template_over_the_same_keys(
    tmp_path: pathlib.Path,
) -> None:
    with compose.Pipeline("p", root=tmp_path) as pipeline:
        plan = _helper_plan(_plan_input(pipeline))
        cut = _helper_cut(pipeline.map(plan))
        _helper_build(cut)

    assert pipeline.list_templates() == ["p/_helper_cut", "p/_helper_build"]
    build = pipeline.get_template("p/_helper_build")
    assert build["source"] == types.ArtifactIdentity("p/_helper_plan", None)
    assert build["mapped_deps"] == {"cut": "p/_helper_cut"}
    assert ("p/_helper_build", "p/_helper_cut") in pipeline.build_dag().edges()


def test_a_template_keeps_its_unmapped_dependencies(tmp_path: pathlib.Path) -> None:
    with compose.Pipeline("p", root=tmp_path) as pipeline:
        plan = _helper_plan(_plan_input(pipeline))
        _helper_cut_with(pipeline.map(plan), _helper_value())

    template = pipeline.get_template("p/_helper_cut_with")
    assert template["mapped_deps"] == {"entry": None}
    assert template["info"]["deps"]["config"].identity == types.ArtifactIdentity(
        "p/_helper_value", None
    )


def test_map_over_a_typed_dict_directory_field(tmp_path: pathlib.Path) -> None:
    with compose.Pipeline("p", root=tmp_path) as pipeline:
        produced = _helper_plan_parts()
        _helper_cut(pipeline.map(produced.parts))

    template = pipeline.get_template("p/_helper_cut")
    assert template["source"] == types.ArtifactIdentity("p/_helper_plan_parts", "parts")


def test_a_template_in_a_variant_block_is_named_by_its_variant(tmp_path: pathlib.Path) -> None:
    with compose.Pipeline("p", root=tmp_path) as pipeline:
        plan = _helper_plan(_plan_input(pipeline))
        with pipeline.variant("small"):
            _helper_cut(pipeline.map(plan))

    assert pipeline.list_templates() == ["p/_helper_cut@small"]


def test_a_template_takes_params(tmp_path: pathlib.Path) -> None:
    with compose.Pipeline("p", root=tmp_path) as pipeline:
        plan = _helper_plan(_plan_input(pipeline))
        _helper_cut_params(pipeline.map(plan), params=_HelperCutParams(factor=3))

    params = pipeline.get_template("p/_helper_cut_params")["info"]["params"]
    assert isinstance(params, _HelperCutParams)
    assert params.factor == 3


def test_a_mapped_handle_in_a_list_makes_a_template(tmp_path: pathlib.Path) -> None:
    with compose.Pipeline("p", root=tmp_path) as pipeline:
        plan = _helper_plan(_plan_input(pipeline))
        entry = pipeline.map(plan)
        _helper_combine([entry, _helper_value()])

    assert pipeline.get_template("p/_helper_combine")["mapped_deps"] == {"cuts[0]": None}


def test_include_brings_templates(tmp_path: pathlib.Path) -> None:
    with compose.Pipeline("p", root=tmp_path) as inner:
        plan = _helper_plan(_plan_input(inner))
        _helper_cut(inner.map(plan))
    outer = compose.Pipeline("all", root=tmp_path)

    outer.include(inner)

    assert outer.list_templates() == ["p/_helper_cut"]
    assert outer.list_stages() == ["p/_helper_plan"]


def test_a_template_reached_through_an_imported_handle_comes_with_its_producer(
    tmp_path: pathlib.Path,
) -> None:
    with compose.Pipeline("prep", root=tmp_path) as prep:
        plan = _helper_plan(_plan_input(prep))
        cut = _helper_cut(prep.map(plan))
    with compose.Pipeline("report", root=tmp_path) as report:
        _helper_build(cut)

    assert report.list_stages() == ["prep/_helper_plan"]
    assert report.list_templates() == ["prep/_helper_cut", "report/_helper_build"]


# =============================================================================
# Refusals
# =============================================================================


def test_map_over_an_output_that_is_not_a_directory_is_refused(tmp_path: pathlib.Path) -> None:
    with (
        pytest.raises(exceptions.PipelineConfigError, match="not a directory output"),
        compose.Pipeline("p", root=tmp_path) as pipeline,
    ):
        pipeline.map(_helper_value())


def test_map_over_an_input_is_refused(tmp_path: pathlib.Path) -> None:
    with (
        pytest.raises(exceptions.PipelineConfigError, match="directory output of a stage"),
        compose.Pipeline("p", root=tmp_path) as pipeline,
    ):
        pipeline.map(_plan_input(pipeline))


def test_map_over_one_file_of_a_directory_is_refused(tmp_path: pathlib.Path) -> None:
    with (
        pytest.raises(exceptions.PipelineConfigError, match="one file"),
        compose.Pipeline("p", root=tmp_path) as pipeline,
    ):
        plan = _helper_plan(_plan_input(pipeline))
        pipeline.map(plan["a.yaml"])


def test_map_over_a_mapped_handle_is_refused(tmp_path: pathlib.Path) -> None:
    with (
        pytest.raises(exceptions.PipelineConfigError, match="already mapped"),
        compose.Pipeline("p", root=tmp_path) as pipeline,
    ):
        plan = _helper_plan(_plan_input(pipeline))
        pipeline.map(pipeline.map(plan))


def test_a_stage_mapped_over_two_directories_is_refused(tmp_path: pathlib.Path) -> None:
    with (
        pytest.raises(exceptions.StageDefinitionError, match="mapped over both"),
        compose.Pipeline("p", root=tmp_path) as pipeline,
    ):
        first = _helper_plan(_plan_input(pipeline))
        second = _helper_plan(_plan_input(pipeline))
        _helper_cut_with(pipeline.map(first), pipeline.map(second))


@pytest.mark.parametrize(
    "select",
    [
        pytest.param(lambda entry: entry["x.yaml"], id="item"),
        pytest.param(lambda entry: entry.field, id="attribute"),
    ],
)
def test_a_mapped_entry_has_no_parts(
    tmp_path: pathlib.Path, select: Callable[[compose.ArtifactHandle], object]
) -> None:
    with (
        pytest.raises((KeyError, AttributeError), match="one file of"),
        compose.Pipeline("p", root=tmp_path) as pipeline,
    ):
        plan = _helper_plan(_plan_input(pipeline))
        select(pipeline.map(plan))
