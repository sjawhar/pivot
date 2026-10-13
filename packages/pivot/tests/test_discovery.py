# pyright: reportMissingImports=false
from __future__ import annotations

import pathlib
import sys

import pytest

from pivot import discovery


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


def test_discover_pipeline_fingerprints_helper_defined_in_pipeline(
    tmp_path: pathlib.Path,
) -> None:
    (tmp_path / "pipeline.py").write_text(
        """
import pathlib

from pivot.compose import Pipeline, stage


def normalize(value: str) -> str:
    return value.strip()


@stage
def preprocess(value: str) -> str:
    return normalize(value)


with Pipeline("test", root=pathlib.Path(__file__).parent) as pipeline:
    preprocess(pipeline.input("value.txt", t=str))
"""
    )

    pipeline = discovery.discover_pipeline(project_root=tmp_path)

    assert pipeline is not None
    assert "func:normalize" in pipeline.ensure_fingerprint("test/preprocess")


def test_discover_root_pipeline_imports_sibling_stages_module(tmp_path: pathlib.Path) -> None:
    (tmp_path / "stages.py").write_text(
        """
from pivot.compose import stage


@stage
def clean(value: str) -> str:
    return value.strip()
"""
    )
    (tmp_path / "pipeline.py").write_text(
        """
import pathlib

from pivot.compose import Pipeline
from stages import clean


with Pipeline("test", root=pathlib.Path(__file__).parent) as pipeline:
    clean(pipeline.input("value.txt", t=str))
"""
    )

    pipeline = discovery.discover_pipeline(project_root=tmp_path)

    assert pipeline is not None
    assert pipeline.list_stages() == ["test/clean"]


def test_discover_subdirectory_pipeline_imports_package_helper(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "report"
    report.mkdir()
    (report / "helpers.py").write_text(
        """
def clean(value: str) -> str:
    return value.strip()
"""
    )
    (report / "pipeline.py").write_text(
        """
import pathlib

from pivot.compose import Pipeline, stage
from report.helpers import clean


@stage
def preprocess(value: str) -> str:
    return clean(value)


with Pipeline("report", root=pathlib.Path(__file__).parent) as pipeline:
    preprocess(pipeline.input("value.txt", t=str))
"""
    )
    monkeypatch.chdir(report)

    pipeline = discovery.discover_pipeline(project_root=tmp_path)

    assert pipeline is not None
    assert pipeline.list_stages() == ["report/preprocess"]
    assert "func:clean" in pipeline.ensure_fingerprint("report/preprocess")


def test_discover_all_reuses_pipeline_imported_by_another_pipeline(
    tmp_path: pathlib.Path,
) -> None:
    prep = tmp_path / "prep"
    prep.mkdir()
    (prep / "pipeline.py").write_text(
        """
import pathlib

from pivot.compose import Pipeline, stage

with (pathlib.Path(__file__).with_name("loads.txt")).open("a") as loads:
    loads.write("prep\\n")


@stage
def kept(value: str) -> str:
    return value


with Pipeline("prep", root=pathlib.Path(__file__).parent) as pipeline:
    kept_handle = kept(pipeline.input("value.txt", t=str))
"""
    )
    report = tmp_path / "report"
    report.mkdir()
    (report / "pipeline.py").write_text(
        """
import pathlib

from pivot.compose import Pipeline, stage
from prep.pipeline import kept_handle


@stage
def consume(value: str) -> str:
    return value


with Pipeline("report", root=pathlib.Path(__file__).parent) as pipeline:
    consume(kept_handle)
"""
    )

    pipeline = discovery.discover_pipeline(project_root=tmp_path, all_pipelines=True)

    assert pipeline is not None
    assert set(pipeline.list_stages()) == {"prep/kept", "report/consume"}
    assert (prep / "loads.txt").read_text().splitlines() == ["prep"]


def test_discover_pipeline_replaces_modules_and_import_root_between_projects(
    tmp_path: pathlib.Path,
) -> None:
    """A later discovery does not reuse a same-named module from an earlier project."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root, value in ((first, "first"), (second, "second")):
        root.mkdir()
        (root / "stages.py").write_text(
            f"""
from pivot.compose import stage


@stage
def clean(value: str) -> str:
    return "{value}"
"""
        )
        (root / "pipeline.py").write_text(
            """
import pathlib

from pivot.compose import Pipeline
from stages import clean


with Pipeline("test", root=pathlib.Path(__file__).parent) as pipeline:
    clean(pipeline.input("value.txt", t=str))
"""
        )

    first_pipeline = discovery.discover_pipeline(project_root=first)
    second_pipeline = discovery.discover_pipeline(project_root=second)

    assert first_pipeline is not None
    assert second_pipeline is not None
    module_file = getattr(sys.modules["stages"], "__file__", None)
    assert isinstance(module_file, str)
    assert pathlib.Path(module_file).resolve() == (second / "stages.py")
    assert str(first) not in sys.path
    assert str(second) == sys.path[0]
