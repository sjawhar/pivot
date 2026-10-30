"""How a template (``Pipeline.map``) becomes one stage per key.

The key of a file is its path inside the directory without its extension. Each instance is an
ordinary stage named ``<template>@<key>``: its mapped dependency reads that key's file, its
outputs are its own, and it joins the registry under the same name and output-path checks as
every other stage. Expansions last until ``reset_expansions``, so no expansion is saved.
"""

import pathlib
from typing import Annotated, TypedDict

import pytest

from pivot import compose, exceptions, loaders, registry, types


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


class _HelperFields(TypedDict):
    _helper_y: dict[str, int]


@compose.stage
def _helper_fields(entry: dict[str, int]) -> _HelperFields:
    return {"_helper_y": entry}


@compose.stage
def _helper_y() -> dict[str, int]:
    return {"value": 0}


def _chain(tmp_path: pathlib.Path) -> compose.Pipeline:
    with compose.Pipeline("p", root=tmp_path) as pipeline:
        plan = _helper_plan(pipeline.input("plan.yaml"))
        cut = _helper_cut(pipeline.map(plan))
        _helper_build(cut)
    return pipeline


_PAIRS = [("a", "a.yaml"), ("b", "b.yaml")]


# =============================================================================
# Keys
# =============================================================================


def test_instance_keys_drop_the_extension_and_sort() -> None:
    assert registry.map_instance_keys(["c.yaml", "a.yaml", "b.tar.gz"]) == [
        ("a", "a.yaml"),
        ("b.tar", "b.tar.gz"),
        ("c", "c.yaml"),
    ]


@pytest.mark.parametrize(
    "file_key",
    [
        pytest.param("a b.yaml", id="space"),
        pytest.param("sub/a.yaml", id="nested"),
        pytest.param("a@b.yaml", id="at"),
    ],
)
def test_a_file_whose_key_cannot_name_a_stage_is_refused(file_key: str) -> None:
    with pytest.raises(exceptions.PipelineConfigError) as excinfo:
        registry.map_instance_keys([file_key])

    message = str(excinfo.value)
    assert repr(file_key) in message, message
    assert "letters, digits" in message, message


def test_two_files_with_one_key_are_refused_naming_both() -> None:
    with pytest.raises(exceptions.PipelineConfigError) as excinfo:
        registry.map_instance_keys(["a.yaml", "a.json"])

    message = str(excinfo.value)
    assert "'a.json'" in message, message
    assert "'a.yaml'" in message, message


# =============================================================================
# Expansion
# =============================================================================


def test_expansion_adds_one_stage_per_key_reading_its_own_file(tmp_path: pathlib.Path) -> None:
    pipeline = _chain(tmp_path)

    added = pipeline.expand_template("p/_helper_cut", _PAIRS)

    assert added == ["p/_helper_cut@a", "p/_helper_cut@b"]
    assert pipeline.list_stages() == ["p/_helper_plan", "p/_helper_cut@a", "p/_helper_cut@b"]
    info = pipeline.get_stage("p/_helper_cut@a")
    dep = info["deps"]["entry"]
    assert dep.identity == types.ArtifactIdentity("p/_helper_plan", "a.yaml")
    assert dep.directory == types.ArtifactIdentity("p/_helper_plan", None)
    assert [out.identity for out in info["outs"]] == [
        types.ArtifactIdentity("p/_helper_cut@a", None)
    ]
    assert info["variant"] == "a"
    assert pipeline.list_templates() == ["p/_helper_cut", "p/_helper_build"]


def test_a_chained_instance_reads_its_upstream_instance_of_the_same_key(
    tmp_path: pathlib.Path,
) -> None:
    pipeline = _chain(tmp_path)
    pipeline.expand_template("p/_helper_cut", _PAIRS)

    pipeline.expand_template("p/_helper_build", _PAIRS)

    dep = pipeline.get_stage("p/_helper_build@b")["deps"]["cut"]
    assert dep.identity == types.ArtifactIdentity("p/_helper_cut@b", None)
    assert ("p/_helper_build@b", "p/_helper_cut@b") in pipeline.build_dag().edges()


def test_instances_of_lists_a_templates_expansion(tmp_path: pathlib.Path) -> None:
    pipeline = _chain(tmp_path)
    pipeline.expand_template("p/_helper_cut", _PAIRS)

    assert pipeline.instances_of("p/_helper_cut") == ["p/_helper_cut@a", "p/_helper_cut@b"]
    assert pipeline.instances_of("p/_helper_build") == []


def test_reset_expansions_removes_every_instance_and_frees_its_paths(
    tmp_path: pathlib.Path,
) -> None:
    pipeline = _chain(tmp_path)
    pipeline.expand_template("p/_helper_cut", _PAIRS)

    pipeline.reset_expansions()

    assert pipeline.list_stages() == ["p/_helper_plan"]
    assert pipeline.expand_template("p/_helper_cut", [("a", "a.yaml")]) == ["p/_helper_cut@a"]


def test_an_instance_named_like_a_static_stage_is_refused_and_nothing_is_added(
    tmp_path: pathlib.Path,
) -> None:
    with compose.Pipeline("p", root=tmp_path) as pipeline:
        plan = _helper_plan(pipeline.input("plan.yaml"))
        _helper_cut(pipeline.map(plan))
        with pipeline.variant("b"):
            _helper_cut(plan["b.yaml"])

    with pytest.raises(exceptions.ValidationError, match="'p/_helper_cut@b' already registered"):
        pipeline.expand_template("p/_helper_cut", _PAIRS)

    assert pipeline.list_stages() == ["p/_helper_plan", "p/_helper_cut@b"]


def test_an_instance_output_path_taken_by_a_static_stage_is_refused_and_nothing_is_added(
    tmp_path: pathlib.Path,
) -> None:
    """Instance ``p/_helper_fields@b``'s field ``_helper_y`` and pipeline
    ``p/_helper_fields@b``'s stage ``_helper_y`` both write
    ``data/p/_helper_fields@b/_helper_y.yaml``."""
    with compose.Pipeline("p", root=tmp_path) as mapped:
        plan = _helper_plan(mapped.input("plan.yaml"))
        _helper_fields(mapped.map(plan))
    with compose.Pipeline("p/_helper_fields@b", root=tmp_path) as static:
        _helper_y()
    combined = compose.Pipeline("all", root=tmp_path)
    combined.include(mapped)
    combined.include(static)

    with pytest.raises(exceptions.OutputPathConflictError, match="both write"):
        combined.expand_template("p/_helper_fields", _PAIRS)

    assert "p/_helper_fields@a" not in combined.list_stages(), "the expansion was undone whole"
    assert combined.expand_template("p/_helper_fields", [("a", "a.yaml")]) == [
        "p/_helper_fields@a"
    ]


def test_an_instance_name_too_long_for_a_lock_is_refused(tmp_path: pathlib.Path) -> None:
    pipeline = _chain(tmp_path)
    key = "k" * 200

    with pytest.raises(exceptions.PipelineConfigError, match="too long"):
        pipeline.expand_template("p/_helper_cut", [(key, f"{key}.yaml")])

    assert pipeline.list_stages() == ["p/_helper_plan"]
