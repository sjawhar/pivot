# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import pytest

from pivot import loaders, registry, types
from pivot.compose import Pipeline

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
        no_fingerprint=False,
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


def test_include_skips_duplicate_stage_names(tmp_path: pathlib.Path) -> None:
    a = Pipeline("same", root=tmp_path)
    a._registry.add_existing(_make_stage_info("same/train"))

    b = Pipeline("same", root=tmp_path)
    b._registry.add_existing(_make_stage_info("same/train"))

    combined = Pipeline("all", root=tmp_path)
    combined.include(a)
    combined.include(b)

    assert combined.list_stages() == ["same/train"]
