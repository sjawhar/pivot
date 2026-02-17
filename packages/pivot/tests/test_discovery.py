from __future__ import annotations

from typing import TYPE_CHECKING

from pivot import discovery

if TYPE_CHECKING:
    import pathlib


def test_find_config_in_dir_ignores_pivot_yaml(tmp_path: pathlib.Path) -> None:
    (tmp_path / "pivot.yaml").write_text("stages:\n  train:\n    python: train.py\n")

    assert discovery.find_config_in_dir(tmp_path) is None


def test_find_config_in_dir_returns_pipeline_py(tmp_path: pathlib.Path) -> None:
    pipeline_path = tmp_path / "pipeline.py"
    pipeline_path.write_text("pipeline = None\n")

    assert discovery.find_config_in_dir(tmp_path) == pipeline_path
