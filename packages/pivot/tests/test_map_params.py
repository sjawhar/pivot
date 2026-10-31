"""``params.yaml`` keys for a stage mapped over a directory output (``Pipeline.map``).

A template has instances only once a run expands it, so its keys are checked against the
template: its base name applies to every instance, ``<template>@<key>`` to one instance whether
or not a run added it, and each value against the template's params class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import pytest

from pivot import compose, exceptions, loaders, parameters, stage_def

if TYPE_CHECKING:
    import pathlib


@compose.stage
def _helper_plan(
    raw: dict[str, int],
) -> Annotated[dict[str, dict[str, int]], compose.directory, loaders.YAML[dict[str, int]]()]:
    return {f"{name}.yaml": {"value": value} for name, value in raw.items()}


class _HelperCutParams(stage_def.StageParams):
    factor: int = 1


@compose.stage
def _helper_cut(entry: dict[str, int], params: _HelperCutParams) -> dict[str, int]:
    return {"value": entry["value"] * params.factor}


def _pipeline(root: pathlib.Path, *, variant: str | None = None) -> compose.Pipeline:
    with compose.Pipeline("p", root=root) as pipeline:
        plan = _helper_plan(pipeline.input("plan.yaml", t=dict[str, int]))
        if variant is None:
            _helper_cut(pipeline.map(plan), params=_HelperCutParams())
        else:
            with pipeline.variant(variant):
                _helper_cut(pipeline.map(plan), params=_HelperCutParams())
    return pipeline


def _load(root: pathlib.Path, pipeline: compose.Pipeline, text: str) -> parameters.ParamsOverrides:
    path = root / "params.yaml"
    path.write_text(text)
    return parameters.load_params_yaml(pipeline, path)


def test_the_template_name_and_an_unexpanded_instance_are_keys(tmp_path: pathlib.Path) -> None:
    pipeline = _pipeline(tmp_path)

    loaded = _load(
        tmp_path, pipeline, "p/_helper_cut:\n  factor: 2\np/_helper_cut@b:\n  factor: 3\n"
    )

    assert loaded == {"p/_helper_cut": {"factor": 2}, "p/_helper_cut@b": {"factor": 3}}


def test_an_unknown_field_for_an_unexpanded_instance_is_refused(tmp_path: pathlib.Path) -> None:
    pipeline = _pipeline(tmp_path)

    with pytest.raises(exceptions.PipelineConfigError) as excinfo:
        _load(tmp_path, pipeline, "p/_helper_cut@b:\n  facter: 3\n")

    message = str(excinfo.value)
    assert "p/_helper_cut@b" in message and "facter" in message, message
    assert "factor" in message, "the field it could mean is named"


def test_a_value_the_templates_field_rejects_is_refused(tmp_path: pathlib.Path) -> None:
    pipeline = _pipeline(tmp_path)

    with pytest.raises(exceptions.PipelineConfigError) as excinfo:
        _load(tmp_path, pipeline, "p/_helper_cut@a:\n  factor: many\n")

    message = str(excinfo.value)
    assert "p/_helper_cut@a" in message and "factor" in message and "many" in message, message


def test_an_unknown_field_for_the_template_is_refused(tmp_path: pathlib.Path) -> None:
    pipeline = _pipeline(tmp_path)

    with pytest.raises(exceptions.PipelineConfigError, match="facter"):
        _load(tmp_path, pipeline, "p/_helper_cut:\n  facter: 3\n")


def test_an_instance_key_no_file_could_have_is_refused(tmp_path: pathlib.Path) -> None:
    """A key holds no ``@`` or space (``registry.map_instance_keys``), so no instance has it."""
    pipeline = _pipeline(tmp_path)

    with pytest.raises(exceptions.PipelineConfigError, match="p/_helper_cut@a b"):
        _load(tmp_path, pipeline, "p/_helper_cut@a b:\n  factor: 3\n")


def test_a_variant_templates_own_name_is_refused(tmp_path: pathlib.Path) -> None:
    """Instances of ``p/_helper_cut@x`` read ``p/_helper_cut`` and ``p/_helper_cut@x@<key>``:
    the template's own name is a key none of them reads."""
    pipeline = _pipeline(tmp_path, variant="x")

    with pytest.raises(exceptions.PipelineConfigError) as excinfo:
        _load(tmp_path, pipeline, "p/_helper_cut@x:\n  factor: 3\n")

    message = str(excinfo.value)
    assert "names mapped stage 'p/_helper_cut@x'" in message, message
    assert "'p/_helper_cut'" in message and "'p/_helper_cut@x@<key>'" in message, message


def test_a_variant_templates_instance_and_base_are_keys(tmp_path: pathlib.Path) -> None:
    pipeline = _pipeline(tmp_path, variant="x")

    loaded = _load(
        tmp_path, pipeline, "p/_helper_cut:\n  factor: 2\np/_helper_cut@x@a:\n  factor: 5\n"
    )

    assert loaded == {"p/_helper_cut": {"factor": 2}, "p/_helper_cut@x@a": {"factor": 5}}
