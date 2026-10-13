"""Focused coverage for helpers.create_pipeline_py()."""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING

import cloudpickle

from pivot import discovery, executor, project, stage_def
from tests import conftest, helpers

if TYPE_CHECKING:
    import pathlib

    import pytest

    from pivot.registry import PipelineLike


class _GeneratedParams(stage_def.StageParams):
    prefix: str = "default"


def _seed(params: _GeneratedParams) -> str:
    return f"{params.prefix}:seed"


def _consume(value: str) -> str:
    return f"consume:{value}"


def test_create_pipeline_py_renames_stage_binds_params_and_handles(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Generated pipeline executes a renamed StageParams stage and handle consumer."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".pivot").mkdir()
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    pipeline_path = helpers.create_pipeline_py(
        [_seed, _consume],
        path=tmp_path,
        extra_code="""\
class _GeneratedParams(stage_def.StageParams):
    prefix: str = "default"
""",
        names={"_seed": "renamed_seed", "_consume": "consume"},
        args={
            "renamed_seed": {"params": "_GeneratedParams(prefix='configured')"},
            "consume": {"value": "renamed_seed_out"},
        },
    )

    pipeline = discovery.load_pipeline_from_path(pipeline_path)
    assert pipeline is not None
    assert pipeline.list_stages() == ["test/renamed_seed", "test/consume"]

    results = executor.run(pipeline=pipeline)

    assert results["test/renamed_seed"]["status"] == "ran"
    assert results["test/consume"]["status"] == "ran"
    assert (tmp_path / "data/test/consume.txt").read_text() == "consume:configured:seed"


def _discover(pipeline_path: pathlib.Path) -> PipelineLike:
    """Load a root pipeline.py through discover_pipeline, which raises the real load error.

    load_pipeline_from_path logs a failed import at debug level and returns None, so a test
    asserting on its result reports only `assert None is not None`.
    """
    pipeline = discovery.discover_pipeline(project_root=pipeline_path.parent)
    assert pipeline is not None, f"{pipeline_path} defines no pipeline"
    return pipeline


def _write_project_with_sibling_stage(root: pathlib.Path, value: str) -> pathlib.Path:
    (root / ".git").mkdir(parents=True)
    (root / ".pivot").mkdir()
    (root / "stages.py").write_text(
        f"""
from pivot.compose import stage


@stage
def build() -> str:
    return "{value}"
"""
    )
    pipeline_path = root / "pipeline.py"
    pipeline_path.write_text(
        """
import pathlib

from pivot.compose import Pipeline
from stages import build


with Pipeline("test", root=pathlib.Path(__file__).parent) as pipeline:
    build()
"""
    )
    return pipeline_path


def test_reusable_worker_executes_current_project_sibling_module(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A warmed worker must execute project B, not stale same-named modules from project A."""
    first = _write_project_with_sibling_stage(tmp_path / "first", "first")
    second = _write_project_with_sibling_stage(tmp_path / "second", "second")

    monkeypatch.setattr(project, "_project_root_cache", first.parent)
    first_pipeline = _discover(first)
    first_results = executor.run(
        pipeline=first_pipeline,
        parallel=True,
        max_workers=1,
        force=True,
    )
    assert first_results["test/build"]["status"] == "ran"
    assert (first.parent / "data/test/build.txt").read_text() == "first"

    monkeypatch.setattr(project, "_project_root_cache", second.parent)
    second_pipeline = _discover(second)
    second_results = executor.run(
        pipeline=second_pipeline,
        parallel=True,
        max_workers=1,
        force=True,
    )
    assert second_results["test/build"]["status"] == "ran"
    assert (second.parent / "data/test/build.txt").read_text() == "second"


def test_reusable_worker_keeps_generated_params_class_identity_per_submission(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".pivot").mkdir()
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    pipeline_path = tmp_path / "pipeline.py"
    pipeline_path.write_text(
        """
import pathlib

from pivot import StageParams
from pivot.compose import Pipeline, stage


class Params(StageParams):
    value: str = "identity"


@stage
def class_identity(params: Params) -> str:
    return str(type(params) is Params)


with Pipeline("test", root=pathlib.Path(__file__).parent) as pipeline:
    class_identity(params=Params())
"""
    )

    pipeline = _discover(pipeline_path)
    executor.run(pipeline=pipeline, parallel=True, max_workers=1, force=True)
    first = (tmp_path / "data/test/class_identity.txt").read_text()
    executor.run(pipeline=pipeline, parallel=True, max_workers=1, force=True)
    second = (tmp_path / "data/test/class_identity.txt").read_text()
    assert first == "True"
    assert second == "True"


def test_reusable_worker_executes_preimported_current_project_sibling(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller-preimported B sibling is serialized by value, not read from worker A."""
    first = _write_project_with_sibling_stage(tmp_path / "first", "first")
    second = _write_project_with_sibling_stage(tmp_path / "second", "second")

    monkeypatch.setattr(project, "_project_root_cache", first.parent)
    first_pipeline = _discover(first)
    executor.run(pipeline=first_pipeline, parallel=True, max_workers=1, force=True)

    # This test imports B's `stages` as the caller would. Discovery leaves a caller's own import
    # in place, so the isolation block removes it before any later test in this worker runs.
    with conftest.stage_module_isolation(second.parent):
        second_stages = importlib.import_module("stages")
        assert second_stages.build() == "second"

        monkeypatch.setattr(project, "_project_root_cache", second.parent)
        second_pipeline = _discover(second)
        assert "stages" in cloudpickle.list_registry_pickle_by_value()
        second_results = executor.run(
            pipeline=second_pipeline,
            parallel=True,
            max_workers=1,
            force=True,
        )
        assert second_results["test/build"]["status"] == "ran"
        assert (second.parent / "data/test/build.txt").read_text() == "second"

        # End B's discovery session while the isolation block still owns B's sys.path entry.
        discovery.discover_pipeline(project_root=tmp_path / "empty")


def test_discovery_does_not_pickle_editable_src_package_by_value(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    (root / ".git").mkdir(parents=True)
    (root / ".pivot").mkdir()
    package = root / "src/pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        """
def transform(value: str) -> str:
    return value.upper()
"""
    )
    (root / "pipeline.py").write_text(
        """
import pathlib

from pivot.compose import Pipeline, stage
from pkg import transform


@stage
def build() -> str:
    return transform("payload")


with Pipeline("test", root=pathlib.Path(__file__).parent) as pipeline:
    build()
"""
    )
    monkeypatch.syspath_prepend(str(root / "src"))

    pipeline = discovery.discover_pipeline(project_root=root)

    assert pipeline is not None
    assert "pkg" not in cloudpickle.list_registry_pickle_by_value()
    discovery.discover_pipeline(project_root=tmp_path / "empty")
    sys.modules.pop("pkg", None)
