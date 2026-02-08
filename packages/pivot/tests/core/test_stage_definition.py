# pyright: reportUnusedFunction=false, reportUnusedParameter=false
"""Tests for StageDefinition dataclass and extract_stage_definition().

Tests the single-pass extraction that replaced individual get_dep_specs_from_signature /
get_output_specs_from_return / get_single_output_spec_from_return calls.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Annotated, TypedDict

import pytest

from pivot import exceptions, loaders, outputs, stage_def

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


# ==============================================================================
# Module-level stage fixtures (required for get_type_hints resolution)
# ==============================================================================


class _SimpleOutput(TypedDict):
    result: Annotated[pathlib.Path, outputs.Out("result.txt", loaders.PathOnly())]


class _MultiOutput(TypedDict):
    model: Annotated[pathlib.Path, outputs.Out("model.bin", loaders.PathOnly())]
    metrics: Annotated[
        dict[str, float], outputs.Out("metrics.json", loaders.JSON[dict[str, float]]())
    ]


class _IncrementalOutput(TypedDict):
    cache: Annotated[
        dict[str, int], outputs.IncrementalOut("cache.json", loaders.JSON[dict[str, int]]())
    ]


class _TestParams(stage_def.StageParams):
    lr: float = 0.01


def _stage_with_dep(
    data: Annotated[pathlib.Path, outputs.Dep("input.csv", loaders.PathOnly())],
) -> _SimpleOutput:
    return _SimpleOutput(result=pathlib.Path("result.txt"))


def _stage_no_annotations() -> None:
    pass


def _stage_outputs_only() -> _SimpleOutput:
    return _SimpleOutput(result=pathlib.Path("result.txt"))


def _stage_with_params(
    params: _TestParams,
    data: Annotated[pathlib.Path, outputs.Dep("input.csv", loaders.PathOnly())],
) -> _SimpleOutput:
    return _SimpleOutput(result=pathlib.Path("result.txt"))


def _stage_with_placeholder(
    data: Annotated[pathlib.Path, outputs.PlaceholderDep(loaders.PathOnly())],
) -> _SimpleOutput:
    return _SimpleOutput(result=pathlib.Path("result.txt"))


def _single_output_stage(
    data: Annotated[pathlib.Path, outputs.Dep("input.csv", loaders.PathOnly())],
) -> Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]:
    return pathlib.Path("output.txt")


def _stage_with_incremental_input(
    cache: Annotated[
        dict[str, int], outputs.IncrementalOut("cache.json", loaders.JSON[dict[str, int]]())
    ],
) -> _IncrementalOutput:
    return _IncrementalOutput(cache=cache or {})


def _stage_multi_dep(
    left: Annotated[pathlib.Path, outputs.Dep("left.csv", loaders.PathOnly())],
    right: Annotated[pathlib.Path, outputs.Dep("right.csv", loaders.PathOnly())],
) -> _SimpleOutput:
    return _SimpleOutput(result=pathlib.Path("result.txt"))


def _stage_with_multi_output() -> _MultiOutput:
    return _MultiOutput(model=pathlib.Path("model.bin"), metrics={"acc": 0.9})


# ==============================================================================
# Basic extraction
# ==============================================================================


def test_basic_extraction_all_fields() -> None:
    """All StageDefinition fields are populated correctly for a dep+output stage."""
    defn = stage_def.extract_stage_definition(_stage_with_dep, "test_stage")
    assert defn.hints_resolved is True
    assert "data" in defn.dep_specs
    assert defn.dep_specs["data"].path == "input.csv"
    assert defn.dep_specs["data"].creates_dep_edge is True
    assert "result" in defn.out_specs
    assert defn.out_specs["result"].path == "result.txt"
    assert defn.single_out_spec is None
    assert defn.params_arg_name is None
    assert defn.params_type is None
    assert defn.placeholder_dep_names == frozenset()


def test_no_annotations() -> None:
    """Empty function yields empty definition with hints_resolved=True."""
    defn = stage_def.extract_stage_definition(_stage_no_annotations, "bare")
    assert defn.hints_resolved is True
    assert defn.dep_specs == {}
    assert defn.out_specs == {}
    assert defn.single_out_spec is None
    assert defn.placeholder_dep_names == frozenset()


def test_outputs_only_no_deps() -> None:
    """Stage with only return outputs and no input deps."""
    defn = stage_def.extract_stage_definition(_stage_outputs_only, "out_only")
    assert defn.dep_specs == {}
    assert "result" in defn.out_specs
    assert defn.out_specs["result"].path == "result.txt"


def test_multi_output_extraction() -> None:
    """Multiple TypedDict output fields are all extracted."""
    defn = stage_def.extract_stage_definition(_stage_with_multi_output, "multi_out")
    assert len(defn.out_specs) == 2
    assert defn.out_specs["model"].path == "model.bin"
    assert defn.out_specs["metrics"].path == "metrics.json"


def test_multi_dep_extraction() -> None:
    """Multiple input deps are all extracted."""
    defn = stage_def.extract_stage_definition(_stage_multi_dep, "multi_dep")
    assert len(defn.dep_specs) == 2
    assert defn.dep_specs["left"].path == "left.csv"
    assert defn.dep_specs["right"].path == "right.csv"


# ==============================================================================
# Params extraction
# ==============================================================================


def test_params_extraction() -> None:
    """StageParams subclass is detected with name and type."""
    defn = stage_def.extract_stage_definition(_stage_with_params, "with_params")
    assert defn.params_arg_name == "params"
    assert defn.params_type is _TestParams


def test_params_not_detected_without_stage_params() -> None:
    """Regular (non-StageParams) parameters are not detected as params."""
    defn = stage_def.extract_stage_definition(_stage_with_dep, "no_params")
    assert defn.params_arg_name is None
    assert defn.params_type is None


# ==============================================================================
# Single output (non-TypedDict)
# ==============================================================================


def test_single_output_extraction() -> None:
    """Annotated return type with Out produces single_out_spec, not out_specs."""
    defn = stage_def.extract_stage_definition(_single_output_stage, "single")
    assert defn.out_specs == {}
    assert defn.single_out_spec is not None
    assert defn.single_out_spec.path == "output.txt"


# ==============================================================================
# PlaceholderDep
# ==============================================================================


def test_placeholder_with_override_resolves() -> None:
    """PlaceholderDep with override is included in dep_specs and placeholder_dep_names."""
    defn = stage_def.extract_stage_definition(
        _stage_with_placeholder,
        "placeholder",
        dep_path_overrides={"data": "override.csv"},
    )
    assert "data" in defn.placeholder_dep_names
    assert defn.dep_specs["data"].path == "override.csv"


def test_placeholder_without_override_records_name() -> None:
    """PlaceholderDep without override records name but does not add to dep_specs."""
    defn = stage_def.extract_stage_definition(_stage_with_placeholder, "placeholder")
    assert "data" in defn.placeholder_dep_names
    assert "data" not in defn.dep_specs


def test_placeholder_empty_string_override_raises() -> None:
    """PlaceholderDep with empty string override raises ValueError."""
    with pytest.raises(ValueError, match="override cannot be empty"):
        stage_def.extract_stage_definition(
            _stage_with_placeholder,
            "placeholder",
            dep_path_overrides={"data": ""},
        )


def test_placeholder_empty_list_override_raises() -> None:
    """PlaceholderDep with empty list override raises ValueError."""
    with pytest.raises(ValueError, match="override contains empty path"):
        stage_def.extract_stage_definition(
            _stage_with_placeholder,
            "placeholder",
            dep_path_overrides={"data": []},
        )


# ==============================================================================
# Dep path overrides (regular Dep)
# ==============================================================================


def test_regular_dep_override() -> None:
    """Regular Dep uses overridden path when dep_path_overrides provides one."""
    defn = stage_def.extract_stage_definition(
        _stage_with_dep,
        "overridden",
        dep_path_overrides={"data": "custom/path.csv"},
    )
    assert defn.dep_specs["data"].path == "custom/path.csv"


# ==============================================================================
# IncrementalOut as input
# ==============================================================================


def test_incremental_out_input_creates_dep_edge_false() -> None:
    """IncrementalOut used as input parameter has creates_dep_edge=False."""
    defn = stage_def.extract_stage_definition(_stage_with_incremental_input, "inc_stage")
    assert "cache" in defn.dep_specs
    assert defn.dep_specs["cache"].creates_dep_edge is False
    assert defn.dep_specs["cache"].path == "cache.json"


def test_incremental_out_output_is_in_out_specs() -> None:
    """IncrementalOut is extracted from TypedDict return as out_spec."""
    defn = stage_def.extract_stage_definition(_stage_with_incremental_input, "inc_stage")
    assert "cache" in defn.out_specs
    assert isinstance(defn.out_specs["cache"], outputs.IncrementalOut)


# ==============================================================================
# Strict / lenient hint resolution
# ==============================================================================


def _make_unresolvable_func() -> object:
    """Create a function with unresolvable type hints via exec."""
    ns: dict[str, object] = {}
    exec(  # noqa: S102
        "def bad_func(x: 'UnresolvableType') -> None: pass\n",
        ns,
    )
    return ns["bad_func"]


def test_strict_raises_on_unresolvable_hints() -> None:
    """strict=True (default) raises StageDefinitionError for bad hints."""
    bad_func = _make_unresolvable_func()
    with pytest.raises(exceptions.StageDefinitionError, match="resolve type hints"):
        stage_def.extract_stage_definition(bad_func, "bad_stage")  # pyright: ignore[reportArgumentType]


def test_lenient_returns_hints_resolved_false() -> None:
    """strict=False returns definition with hints_resolved=False and empty specs."""
    bad_func = _make_unresolvable_func()
    defn = stage_def.extract_stage_definition(bad_func, "bad_stage", strict=False)  # pyright: ignore[reportArgumentType]
    assert defn.hints_resolved is False
    assert defn.dep_specs == {}
    assert defn.out_specs == {}
    assert defn.single_out_spec is None
    assert defn.placeholder_dep_names == frozenset()
    assert defn.params_arg_name is None
    assert defn.params_type is None


# ==============================================================================
# Pipeline integration
# ==============================================================================


def test_pipeline_register_calls_extract_once(
    set_project_root: pathlib.Path, mocker: MockerFixture
) -> None:
    """Pipeline.register should call extract_stage_definition exactly once."""
    from pivot.pipeline.pipeline import Pipeline

    spy = mocker.spy(stage_def, "extract_stage_definition")
    p = Pipeline("test", root=set_project_root)
    p.register(_stage_with_dep, name="stage-with-dep")
    spy.assert_called_once()


def test_pipeline_definition_passed_to_registry(
    set_project_root: pathlib.Path, mocker: MockerFixture
) -> None:
    """Pipeline passes its pre-extracted definition to registry.register."""
    from pivot import registry
    from pivot.pipeline.pipeline import Pipeline

    spy = mocker.spy(registry.StageRegistry, "register")
    p = Pipeline("test", root=set_project_root)
    p.register(_stage_with_dep, name="stage-with-dep")

    # Registry.register was called with a definition kwarg
    _, kwargs = spy.call_args
    assert "definition" in kwargs, "Pipeline should pass definition to registry"
    assert isinstance(kwargs["definition"], stage_def.StageDefinition)
    assert kwargs["definition"].hints_resolved is True
