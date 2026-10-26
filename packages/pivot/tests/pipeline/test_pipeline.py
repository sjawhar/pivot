# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false
from __future__ import annotations

import copy
import importlib.util
import inspect
import sys
from typing import TYPE_CHECKING

import pytest

from pivot import exceptions, loaders, registry, stage_def, types
from pivot.compose import Pipeline

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable
    from types import ModuleType
    from typing import Any


def _helper_make_stage_info(
    name: str,
    deps: dict[str, types.ArtifactRef] | None = None,
    outs: list[types.ArtifactRef] | None = None,
    func: Callable[..., Any] | None = None,
    params: stage_def.StageParams | None = None,
) -> registry.RegistryStageInfo:
    """Each call without ``func`` defines a new ``_fn`` object with one definition (file and
    qualname), as importing a pipeline file again does."""

    def _fn() -> None: ...

    if func is None:
        func = _fn
    if outs is None:
        outs = [
            types.ArtifactRef(
                identity=types.ArtifactIdentity(producer=name, key="out"),
                format=loaders.YAML(),
                python_type=dict,
                tag=types.ArtifactTag.DATA,
            )
        ]

    return registry.RegistryStageInfo(
        func=func,
        name=name,
        deps=deps or {},
        outs=outs,
        incremental_inputs={},
        params=params,
        resources={},
        variant=None,
        signature=inspect.signature(func),
        fingerprint=None,
        params_arg_name=None,
        state_dir=None,
        collection_params={},
        no_fingerprint=False,
        code_deps=[],
        input_paths={},
        unsafe_fingerprint=False,
    )


_make_stage_info = _helper_make_stage_info


def test_include_merges_input_bindings(tmp_path: pathlib.Path) -> None:
    child = Pipeline("child", root=tmp_path)
    with child:
        child.input("ext.jsonl", path="data/external/ext.jsonl")

    parent = Pipeline("parent", root=tmp_path)
    parent.include(child)

    assert parent.input_bindings["ext.jsonl"] == "data/external/ext.jsonl"


def test_add_existing_rejects_mismatched_out_producer(tmp_path: pathlib.Path) -> None:
    pipeline = Pipeline("test", root=tmp_path)
    bad_out = types.ArtifactRef(
        identity=types.ArtifactIdentity(producer="wrong", key="out"),
        format=loaders.YAML(),
        python_type=dict,
        tag=types.ArtifactTag.DATA,
    )
    info = _make_stage_info("my_stage", outs=[bad_out])
    with pytest.raises(ValueError, match="producer.*wrong.*my_stage"):
        pipeline._registry.add_existing(info)


def test_include_preserves_prefixed_names(tmp_path: pathlib.Path) -> None:
    child = Pipeline("child", root=tmp_path)
    child._registry.add_existing(_make_stage_info("child/alpha"))

    parent = Pipeline("parent", root=tmp_path)
    parent._registry.add_existing(_make_stage_info("parent/beta"))
    parent.include(child)

    assert "child/alpha" in parent.list_stages()
    assert "parent/beta" in parent.list_stages()


def test_include_no_collision_with_same_bare_name(tmp_path: pathlib.Path) -> None:
    a = Pipeline("a", root=tmp_path)
    a._registry.add_existing(_make_stage_info("a/train"))

    b = Pipeline("b", root=tmp_path)
    b._registry.add_existing(_make_stage_info("b/train"))

    combined = Pipeline("all", root=tmp_path)
    combined.include(a)
    combined.include(b)

    assert "a/train" in combined.list_stages()
    assert "b/train" in combined.list_stages()


def test_include_dep_identities_need_no_rewriting(tmp_path: pathlib.Path) -> None:
    child = Pipeline("child", root=tmp_path)
    dep_ref = types.ArtifactRef(
        identity=types.ArtifactIdentity(producer="child/upstream", key="out"),
        format=loaders.YAML(),
        python_type=dict,
        tag=types.ArtifactTag.DATA,
    )
    child._registry.add_existing(_make_stage_info("child/upstream"))
    child._registry.add_existing(_make_stage_info("child/downstream", deps={"data": dep_ref}))

    combined = Pipeline("all", root=tmp_path)
    combined.include(child)

    downstream = combined.get_stage("child/downstream")
    assert downstream["deps"]["data"].identity.producer == "child/upstream"


def test_include_keeps_one_stage_included_twice(tmp_path: pathlib.Path) -> None:
    a = Pipeline("same", root=tmp_path)
    a._registry.add_existing(_make_stage_info("same/train"))

    combined = Pipeline("all", root=tmp_path)
    combined.include(a)
    combined.include(a)

    assert combined.list_stages() == ["same/train"]


def _train_a() -> None: ...


def _train_b() -> None: ...


@pytest.mark.parametrize("inside_with", [False, True], ids=["after-block", "inside-block"])
def test_include_refuses_two_functions_under_one_stage_name(
    tmp_path: pathlib.Path, inside_with: bool
) -> None:
    """Two pipelines named 'same' would share same/train's lock and outputs."""
    a = Pipeline("same", root=tmp_path)
    a._registry.add_existing(_make_stage_info("same/train", func=_train_a))

    b = Pipeline("same", root=tmp_path)
    b._registry.add_existing(_make_stage_info("same/train", func=_train_b))

    combined = Pipeline("all", root=tmp_path)
    with pytest.raises(
        exceptions.PipelineConfigError,
        match=r"'same/train' is defined twice with different functions: _train_a and _train_b, "
        + r"both in .*test_pipeline\.py\. Pipeline names must be unique within a project; "
        + "rename one of the pipelines.",
    ):
        if inside_with:
            with combined:
                combined.include(a)
                combined.include(b)
        else:
            combined.include(a)
            combined.include(b)


class _SeedParams(stage_def.StageParams):
    seed: int = 0


def _params_class() -> type[_SeedParams]:
    """A new class object on every call, with one module and qualname, as a re-import gives."""

    class TrainParams(_SeedParams):
        pass

    return TrainParams


class _OtherParams(_SeedParams):
    pass


def _input_ref(
    name: str,
    tag: types.ArtifactTag = types.ArtifactTag.DATA,
    fmt: loaders.Reader[Any] | None = None,
) -> types.ArtifactRef:
    return types.ArtifactRef(
        identity=types.ArtifactIdentity(producer=name, key=None),
        format=fmt if fmt is not None else loaders.YAML(),
        python_type=dict,
        tag=tag,
    )


def _train_info(**changes: Any) -> registry.RegistryStageInfo:
    """same/train reading input x.csv with seed=1, from a new function and params class."""
    info = _make_stage_info(
        "same/train", deps={"x": _input_ref("x.csv")}, params=_params_class()(seed=1)
    )
    info.update(changes)  # pyright: ignore[reportCallIssue, reportArgumentType] - test overrides
    return info


def _include_both(
    tmp_path: pathlib.Path, first: registry.RegistryStageInfo, again: registry.RegistryStageInfo
) -> Pipeline:
    a = Pipeline("same", root=tmp_path)
    a._registry.add_existing(first)
    b = Pipeline("same", root=tmp_path)
    b._registry.add_existing(again)
    combined = Pipeline("all", root=tmp_path)
    combined.include(a)
    combined.include(b)
    return combined


def test_include_keeps_one_stage_whose_definition_was_imported_twice(
    tmp_path: pathlib.Path,
) -> None:
    """`pivot repro path/pipeline.py` imports a discovered file again: new objects, one stage."""
    first, again = _train_info(), _train_info()
    assert first["func"] is not again["func"]
    assert first["params"] != again["params"], "pydantic == needs one class object"

    combined = _include_both(tmp_path, first, again)

    assert combined.list_stages() == ["same/train"]


class _CollectionParams(stage_def.StageParams):
    ids: set[int] = set()
    ceiling: float = 0.0


@pytest.mark.parametrize(
    ("first", "again"),
    [
        (_CollectionParams(ids={0, 3, 11}), copy.deepcopy(_CollectionParams(ids={0, 3, 11}))),
        (_CollectionParams(ceiling=float("nan")), _CollectionParams(ceiling=float("nan"))),
    ],
    ids=["set-iterates-in-another-order", "new-nan"],
)
def test_include_keeps_one_stage_whose_params_encode_alike(
    tmp_path: pathlib.Path, first: _CollectionParams, again: _CollectionParams
) -> None:
    """A deep copy may rebuild a set in another order; a re-import makes a new NaN."""
    assert first.model_dump(mode="json") != again.model_dump(mode="json")

    combined = _include_both(tmp_path, _train_info(params=first), _train_info(params=again))

    assert combined.list_stages() == ["same/train"]


def test_include_keeps_one_stage_whose_python_types_differ(tmp_path: pathlib.Path) -> None:
    """No run-time code reads ArtifactRef.python_type, and a re-imported user type is new."""
    again = _train_info(
        deps={
            "x": types.ArtifactRef(
                types.ArtifactIdentity("x.csv", None), loaders.YAML(), list, types.ArtifactTag.DATA
            )
        },
        outs=[
            types.ArtifactRef(
                identity=types.ArtifactIdentity(producer="same/train", key="out"),
                format=loaders.YAML(),
                python_type=list,
                tag=types.ArtifactTag.DATA,
            )
        ],
    )

    combined = _include_both(tmp_path, _train_info(), again)

    assert combined.list_stages() == ["same/train"]


def test_include_keeps_one_stage_whose_loaders_were_re_created_with_a_user_function(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Importing a file again re-creates `JSON(empty_factory=make_empty)` with a new function."""
    path = tmp_path / "factories.py"
    path.write_text("def make_empty() -> dict:\n    return {}\n")

    def load_factory() -> Callable[[], dict[str, Any]]:
        spec = importlib.util.spec_from_file_location("_t29_factories", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        return module.make_empty

    first_factory, again_factory = load_factory(), load_factory()
    assert first_factory is not again_factory

    def info(factory: Callable[[], dict[str, Any]]) -> registry.RegistryStageInfo:
        loader = loaders.JSON(empty_factory=factory)
        return _train_info(
            deps={"x": _input_ref("x.csv", fmt=loader)},
            outs=[
                types.ArtifactRef(
                    identity=types.ArtifactIdentity(producer="same/train", key="out"),
                    format=loader,
                    python_type=dict,
                    tag=types.ArtifactTag.DATA,
                )
            ],
        )

    combined = _include_both(tmp_path, info(first_factory), info(again_factory))

    assert combined.list_stages() == ["same/train"]


_METRIC_OUT = types.ArtifactRef(
    identity=types.ArtifactIdentity(producer="same/train", key="out"),
    format=loaders.YAML(),
    python_type=dict,
    tag=types.ArtifactTag.METRIC,
)

_DIFFERENCES: dict[str, tuple[dict[str, Any], str]] = {
    "input": ({"deps": {"x": _input_ref("y.csv")}}, "inputs (x=x.csv vs x=y.csv)"),
    "input-format": (
        {"deps": {"x": _input_ref("x.csv", fmt=loaders.CSV(sep=";"))}},
        "input formats (x=YAML(empty_factory=<class 'dict'>) vs "
        + "x=CSV(index_col=None, sep=';', dtype=None))",
    ),
    "input-format-config": (
        {"deps": {"x": _input_ref("x.csv", fmt=loaders.YAML(empty_factory=list))}},
        "input formats (x=YAML(empty_factory=<class 'dict'>) vs "
        + "x=YAML(empty_factory=<class 'list'>))",
    ),
    "input-tag": (
        {"deps": {"x": _input_ref("x.csv", types.ArtifactTag.METRIC)}},
        "inputs (x=x.csv vs x=x.csv (metric))",
    ),
    "input-name": ({"deps": {"y": _input_ref("x.csv")}}, "inputs (x=x.csv vs y=x.csv)"),
    "output-tag": ({"outs": [_METRIC_OUT]}, "outputs (same/train:out vs same/train:out (metric))"),
    "output-format": (
        {
            "outs": [
                types.ArtifactRef(
                    identity=types.ArtifactIdentity(producer="same/train", key="out"),
                    format=loaders.JSON(),
                    python_type=dict,
                    tag=types.ArtifactTag.DATA,
                )
            ]
        },
        "output formats (YAML(empty_factory=<class 'dict'>) vs "
        + "JSON(indent=2, empty_factory=<class 'dict'>))",
    ),
    "incremental": (
        {"incremental_inputs": {"prev": types.ArtifactIdentity("same/train", "out")}},
        "incremental inputs (none vs prev=same/train:out)",
    ),
    "collection": (
        {"collection_params": {"x": "tuple"}},
        "input collections (none vs x=tuple)",
    ),
    "params-value": (
        {"params": _params_class()(seed=2)},
        "params (TrainParams(seed=1) vs TrainParams(seed=2))",
    ),
    "params-class": (
        {"params": _OtherParams(seed=1)},
        "params (TrainParams(seed=1) vs _OtherParams(seed=1))",
    ),
    "params-none": ({"params": None}, "params (TrainParams(seed=1) vs none)"),
    "resources": ({"resources": {"gpu": 1}}, "resources (none vs gpu=1)"),
    "variant": ({"variant": "big"}, "variants (none vs big)"),
    "no-fingerprint": ({"no_fingerprint": True}, "no_fingerprint (False vs True)"),
    "code-deps": ({"code_deps": ["helper.py"]}, "code_deps (none vs helper.py)"),
}


@pytest.mark.parametrize(
    ("changes", "difference"), list(_DIFFERENCES.values()), ids=list(_DIFFERENCES)
)
def test_include_refuses_one_function_called_differently_under_one_stage_name(
    tmp_path: pathlib.Path, changes: dict[str, Any], difference: str
) -> None:
    """Two same-named pipelines calling one shared function differently are two stages."""
    with pytest.raises(exceptions.PipelineConfigError) as excinfo:
        _include_both(tmp_path, _train_info(), _train_info(**changes))

    message = str(excinfo.value)
    assert message.startswith(f"Stage 'same/train' is defined twice with different {difference}")
    assert message.count("test_pipeline.py") == 1, "one definition file is named once"
    assert "both calling _helper_make_stage_info.<locals>._fn in " in message
    assert message.endswith("rename one of the pipelines.")


def test_include_names_the_params_classes_when_both_print_alike(tmp_path: pathlib.Path) -> None:
    """Same class name and values, different module: the message shows what differs."""
    elsewhere = _params_class()
    elsewhere.__module__ = "elsewhere"

    with pytest.raises(
        exceptions.PipelineConfigError,
        match=r"different params \(\(.*TrainParams', '\{\"seed\": 1\}'\) vs "
        + r"\('elsewhere', .*TrainParams', '\{\"seed\": 1\}'\)\)",
    ):
        _include_both(tmp_path, _train_info(), _train_info(params=elsewhere(seed=1)))


def test_include_tells_apart_same_named_functions_behind_one_decorator(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shared functools.wraps decorator must not hide the files that define the stages."""

    def load(name: str, text: str) -> ModuleType:
        path = tmp_path / f"{name}.py"
        path.write_text(text)
        spec = importlib.util.spec_from_file_location(f"_t29_{name}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        return module

    load(
        "deco",
        "import functools\n\n\ndef timed(f):\n    @functools.wraps(f)\n"
        + "    def inner(*args, **kwargs):\n        return f(*args, **kwargs)\n\n    return inner\n",
    )
    left, right = (
        load(side, "from _t29_deco import timed\n\n\n@timed\ndef make() -> None: ...\n")
        for side in ("left", "right")
    )

    with pytest.raises(
        exceptions.PipelineConfigError, match=r"functions: make in left\.py and make in right\.py"
    ):
        _include_both(
            tmp_path,
            _make_stage_info("same/train", func=left.make),
            _make_stage_info("same/train", func=right.make),
        )
