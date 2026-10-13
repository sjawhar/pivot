"""Focused coverage for helpers.create_pipeline_py()."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pivot import discovery, executor, project, stage_def
from tests import helpers

if TYPE_CHECKING:
    import pathlib

    import pytest


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
