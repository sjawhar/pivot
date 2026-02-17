# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import pytest

from pivot import exceptions, loaders, registry, types
from pivot.pipeline import pipeline as pipeline_mod

if TYPE_CHECKING:
    import pathlib


def _helper_make_stage_info(
    name: str,
    deps: dict[str, types.ArtifactRef] | None = None,
    outs: list[types.ArtifactRef] | None = None,
) -> registry.RegistryStageInfo:
    def _fn() -> None: ...

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
        func=_fn,
        name=name,
        deps=deps or {},
        outs=outs,
        params=None,
        mutex=[],
        variant=None,
        signature=inspect.signature(_fn),
        fingerprint=None,
        params_arg_name=None,
        state_dir=None,
        collection_params={},
    )


_make_stage_info = _helper_make_stage_info


def test_include_merges_input_bindings(tmp_path: pathlib.Path) -> None:
    child = pipeline_mod.Pipeline("child", root=tmp_path)
    child.set_input_bindings({"ext.jsonl": "data/external/ext.jsonl"})

    parent = pipeline_mod.Pipeline("parent", root=tmp_path)
    parent.include(child)

    assert parent.input_bindings["ext.jsonl"] == "data/external/ext.jsonl"


def test_add_existing_rejects_mismatched_out_producer(tmp_path: pathlib.Path) -> None:
    pipeline = pipeline_mod.Pipeline("test", root=tmp_path)
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
    child = pipeline_mod.Pipeline("child", root=tmp_path)
    child._registry.add_existing(_make_stage_info("child/alpha"))

    parent = pipeline_mod.Pipeline("parent", root=tmp_path)
    parent._registry.add_existing(_make_stage_info("parent/beta"))
    parent.include(child)

    assert "child/alpha" in parent.list_stages()
    assert "parent/beta" in parent.list_stages()


def test_include_no_collision_with_same_bare_name(tmp_path: pathlib.Path) -> None:
    a = pipeline_mod.Pipeline("a", root=tmp_path)
    a._registry.add_existing(_make_stage_info("a/train"))

    b = pipeline_mod.Pipeline("b", root=tmp_path)
    b._registry.add_existing(_make_stage_info("b/train"))

    combined = pipeline_mod.Pipeline("all", root=tmp_path)
    combined.include(a)
    combined.include(b)

    assert "a/train" in combined.list_stages()
    assert "b/train" in combined.list_stages()


def test_include_dep_identities_need_no_rewriting(tmp_path: pathlib.Path) -> None:
    child = pipeline_mod.Pipeline("child", root=tmp_path)
    dep_ref = types.ArtifactRef(
        identity=types.ArtifactIdentity(producer="child/upstream", key="out"),
        format=loaders.YAML(),
        python_type=dict,
        tag=types.ArtifactTag.DATA,
    )
    child._registry.add_existing(_make_stage_info("child/upstream"))
    child._registry.add_existing(_make_stage_info("child/downstream", deps={"data": dep_ref}))

    combined = pipeline_mod.Pipeline("all", root=tmp_path)
    combined.include(child)

    downstream = combined.get("child/downstream")
    assert downstream["deps"]["data"].identity.producer == "child/upstream"


def test_include_raises_on_exact_name_collision(tmp_path: pathlib.Path) -> None:
    a = pipeline_mod.Pipeline("same", root=tmp_path)
    a._registry.add_existing(_make_stage_info("same/train"))

    b = pipeline_mod.Pipeline("same", root=tmp_path)
    b._registry.add_existing(_make_stage_info("same/train"))

    combined = pipeline_mod.Pipeline("all", root=tmp_path)
    combined.include(a)

    with pytest.raises(exceptions.ValidationError, match="already registered"):
        combined.include(b)
