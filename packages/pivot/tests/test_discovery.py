# pyright: reportMissingImports=false
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

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


def _write_pipeline_file(tmp_path: pathlib.Path, content: str) -> pathlib.Path:
    pipeline_path = tmp_path / "pipeline.py"
    pipeline_path.write_text(content)
    return pipeline_path


def test_load_pipeline_from_module_requires_protocol_methods(tmp_path: pathlib.Path) -> None:
    pipeline_path = _write_pipeline_file(
        tmp_path,
        "pipeline = object()\n",
    )

    with pytest.raises(discovery.DiscoveryError) as excinfo:
        discovery._load_pipeline_from_module(pipeline_path)

    message = str(excinfo.value)
    assert "missing required methods" in message
    for attr in [
        "name",
        "list_stages",
        "get_stage",
        "build_dag",
        "ensure_fingerprint",
        "snapshot",
        "restore",
    ]:
        assert attr in message


def test_load_pipeline_from_module_rejects_partial_pipeline_like(
    tmp_path: pathlib.Path,
) -> None:
    pipeline_path = _write_pipeline_file(
        tmp_path,
        """
class FakePipeline:
    name = "fake"

    def list_stages(self):
        return []

    def get_stage(self, name):
        raise KeyError(name)

    def build_dag(self):
        raise NotImplementedError

    def ensure_fingerprint(self, name):
        return {}

    def snapshot(self):
        return {}

    def restore(self, snapshot):
        return None


pipeline = FakePipeline()
""",
    )

    with pytest.raises(discovery.DiscoveryError) as excinfo:
        discovery._load_pipeline_from_module(pipeline_path)

    message = str(excinfo.value)
    assert "missing required methods" in message
