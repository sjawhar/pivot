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


def _plan_input(pipeline: compose.Pipeline) -> dict[str, int]:
    return pipeline.input("plan.yaml", t=dict[str, int])


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
        _helper_cut(pipeline.map(produced["parts"]))

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


def _helper_variant_after(pipeline: compose.Pipeline) -> None:
    plan = _helper_plan(_plan_input(pipeline))
    _helper_cut(pipeline.map(plan))
    with pipeline.variant("a"):
        _helper_cut(plan["a.yaml"])


def _helper_variant_before(pipeline: compose.Pipeline) -> None:
    plan = _helper_plan(_plan_input(pipeline))
    with pipeline.variant("a"):
        _helper_cut(plan["a.yaml"])
    _helper_cut(pipeline.map(plan))


def _helper_second_call(pipeline: compose.Pipeline) -> None:
    plan = _helper_plan(_plan_input(pipeline))
    _helper_cut(pipeline.map(plan))
    _helper_cut(plan["a.yaml"])


def _helper_mapped_in_a_variant(pipeline: compose.Pipeline) -> None:
    plan = _helper_plan(_plan_input(pipeline))
    _helper_cut(pipeline.map(plan))
    with pipeline.variant("a"):
        _helper_cut(pipeline.map(plan))


@pytest.mark.parametrize(
    ("build", "name"),
    [
        pytest.param(_helper_variant_after, "p/_helper_cut@a", id="a-variant-after"),
        pytest.param(_helper_variant_before, "p/_helper_cut@a", id="a-variant-before"),
        pytest.param(_helper_second_call, "p/_helper_cut@1", id="a-second-call"),
        pytest.param(_helper_mapped_in_a_variant, "p/_helper_cut@a", id="a-mapped-variant"),
    ],
)
def test_a_stage_named_as_an_instance_of_a_mapped_stage_is_refused(
    tmp_path: pathlib.Path, build: Callable[[compose.Pipeline], None], name: str
) -> None:
    """Beside the mapped `p/_helper_cut`, a stage `p/_helper_cut@<key>` (a variant, a second
    call, or a mapped stage in a variant) would share its name with an instance: a target, a
    lock and a params.yaml key could each mean either. The load refuses it, naming both."""
    with (
        pytest.raises(exceptions.ValidationError) as refused,
        compose.Pipeline("p", root=tmp_path) as pipeline,
    ):
        build(pipeline)

    assert str(refused.value) == (
        f"'{name}' has the name of an instance of mapped stage 'p/_helper_cut' "
        + "('p/_helper_cut@<key>'): call one of them inside a pipeline.variant(...) block"
    )


def test_an_included_stage_named_as_an_instance_of_a_mapped_stage_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    with compose.Pipeline("p", root=tmp_path) as mapped:
        plan = _helper_plan(_plan_input(mapped))
        _helper_cut(mapped.map(plan))
    with compose.Pipeline("p", root=tmp_path) as static, static.variant("a"):
        _helper_cut(_helper_value())
    combined = compose.Pipeline("all", root=tmp_path)
    combined.include(mapped)

    with pytest.raises(exceptions.ValidationError, match="'p/_helper_cut@a' has the name of an"):
        combined.include(static)

    assert "p/_helper_cut@a" not in combined.list_stages()


def test_variants_of_a_mapped_stage_beside_each_other_load(tmp_path: pathlib.Path) -> None:
    """Neither `p/_helper_cut@a` nor `p/_helper_cut@b` is the other's instance."""
    with compose.Pipeline("p", root=tmp_path) as pipeline:
        plan = _helper_plan(_plan_input(pipeline))
        for variant in ("a", "b"):
            with pipeline.variant(variant):
                _helper_cut(pipeline.map(plan))

    assert pipeline.list_templates() == ["p/_helper_cut@a", "p/_helper_cut@b"]


def _helper_select_item(entry: compose.ArtifactHandle) -> object:
    return entry["x.yaml"]


def _helper_select_attribute(entry: compose.ArtifactHandle) -> object:
    return entry.field


@pytest.mark.parametrize(
    "select",
    [
        pytest.param(_helper_select_item, id="item"),
        pytest.param(_helper_select_attribute, id="attribute"),
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
