# pyright: reportMissingImports=false
from __future__ import annotations

import importlib
import json
import pathlib
import sys
import types as _stdlib_types

import cloudpickle
import pytest

from pivot import discovery
from tests import conftest


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
    assert "func:pipeline.normalize" in pipeline.ensure_fingerprint("test/preprocess")


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
    assert "func:report.helpers.clean" in pipeline.ensure_fingerprint("report/preprocess")


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


def _write_sibling_stage_project(
    root: pathlib.Path, value: str, *, broken: bool = False
) -> pathlib.Path:
    root.mkdir()
    (root / "stages.py").write_text(f'VALUE = "{value}"\n')
    pipeline_path = root / "pipeline.py"
    pipeline_path.write_text(
        f"""
import pathlib

from pivot.compose import Pipeline, stage
from stages import VALUE


@stage
def build() -> str:
    return VALUE


with Pipeline("{root.name}") as pipeline:
    build()
"""
        + ("\nraise RuntimeError('broken pipeline')\n" if broken else "")
    )
    return pipeline_path


def test_failed_pipeline_import_rolls_back_sibling_modules(tmp_path: pathlib.Path) -> None:
    failed = _write_sibling_stage_project(tmp_path / "failed", "a", broken=True)
    with pytest.raises(discovery.DiscoveryError, match="broken pipeline"):
        discovery.discover_pipeline(project_root=failed.parent)

    valid = _write_sibling_stage_project(tmp_path / "valid", "b")
    pipeline = discovery.discover_pipeline(project_root=valid.parent)

    assert pipeline is not None
    assert pipeline.get_stage("valid/build")["func"]() == "b"
    module_file = getattr(sys.modules["stages"], "__file__", None)
    assert pathlib.Path(str(module_file)).resolve() == valid.parent / "stages.py"


def test_direct_pipeline_load_starts_fresh_import_session(tmp_path: pathlib.Path) -> None:
    first = _write_sibling_stage_project(tmp_path / "first", "a")
    second = _write_sibling_stage_project(tmp_path / "second", "b")

    first_pipeline = discovery.load_pipeline_from_path(first)
    second_pipeline = discovery.load_pipeline_from_path(second)

    assert first_pipeline is not None
    assert second_pipeline is not None
    assert second_pipeline.get_stage("second/build")["func"]() == "b"


def test_discover_all_skips_ordinary_broken_sibling(tmp_path: pathlib.Path) -> None:
    valid = tmp_path / "valid"
    valid.mkdir()
    (valid / "pipeline.py").write_text(
        """
import pathlib

from pivot.compose import Pipeline, stage


@stage
def build() -> str:
    return "ok"


with Pipeline("valid", root=pathlib.Path(__file__).parent) as pipeline:
    build()
"""
    )
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "pipeline.py").write_text("raise RuntimeError('broken sibling')\n")

    pipeline = discovery.discover_pipeline(project_root=tmp_path, all_pipelines=True)

    assert pipeline is not None
    assert pipeline.list_stages() == ["valid/build"]


def test_subdirectory_pipeline_displaces_and_restores_stdlib_parent(
    tmp_path: pathlib.Path,
) -> None:
    root = tmp_path / "project"
    collision = root / "json"
    collision.mkdir(parents=True)
    (collision / "pipeline.py").write_text(
        """
import pathlib

from pivot.compose import Pipeline, stage


@stage
def build() -> str:
    return "project"


with Pipeline("project", root=pathlib.Path(__file__).parents[1]) as pipeline:
    build()
"""
    )

    pipeline = discovery.discover_pipeline(project_root=root, all_pipelines=True)

    assert pipeline is not None
    assert pipeline.list_stages() == ["project/build"]
    assert sys.modules["json"] is not json

    assert discovery.discover_pipeline(project_root=tmp_path / "empty") is None
    assert sys.modules["json"] is json


def test_discovery_restores_displaced_pipeline_module_and_caller_path(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    foreign = _stdlib_types.ModuleType("pipeline")
    monkeypatch.setitem(sys.modules, "pipeline", foreign)
    root = _write_sibling_stage_project(tmp_path / "project", "project").parent
    monkeypatch.syspath_prepend(str(root))

    pipeline = discovery.discover_pipeline(project_root=root)

    assert pipeline is not None
    assert sys.modules["pipeline"] is not foreign
    assert discovery.discover_pipeline(project_root=tmp_path / "empty") is None
    assert sys.modules["pipeline"] is foreign
    assert sys.path.count(str(root)) == 1


def test_discovery_keeps_project_module_preimported_by_caller(tmp_path: pathlib.Path) -> None:
    root = _write_sibling_stage_project(tmp_path / "project", "caller").parent
    # The caller's own import outlives discovery by design, so this test must remove it itself.
    with conftest.stage_module_isolation(root):
        stages = importlib.import_module("stages")

        pipeline = discovery.discover_pipeline(project_root=root)

        assert pipeline is not None
        assert sys.modules["stages"] is stages
        assert discovery.discover_pipeline(project_root=tmp_path / "empty") is None
        assert sys.modules["stages"] is stages


def test_discover_pipeline_missing_pipeline_variable_names_instance(
    tmp_path: pathlib.Path,
) -> None:
    """A Pipeline assigned to the wrong module variable names that variable in the error."""
    (tmp_path / "pipeline.py").write_text(
        "import pathlib\nfrom pivot.compose import Pipeline\n"
        + "wrong_name = Pipeline('wrong', root=pathlib.Path('.'))\n"
    )

    with pytest.raises(
        discovery.DiscoveryError, match="Found Pipeline instance named 'wrong_name'"
    ):
        discovery.discover_pipeline(project_root=tmp_path)


def test_discover_all_rolls_back_pickle_registration_for_broken_sibling(
    tmp_path: pathlib.Path,
) -> None:
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "broken_helper.py").write_text("VALUE = 'broken'\n")
    (broken / "pipeline.py").write_text(
        """
from broken.broken_helper import VALUE

raise RuntimeError(VALUE)
"""
    )
    valid = tmp_path / "valid"
    valid.mkdir()
    (valid / "pipeline.py").write_text(
        """
import pathlib

from pivot.compose import Pipeline, stage


@stage
def build() -> str:
    return "valid"


with Pipeline("valid", root=pathlib.Path(__file__).parent) as pipeline:
    build()
"""
    )

    pipeline = discovery.discover_pipeline(project_root=tmp_path, all_pipelines=True)

    assert pipeline is not None
    assert pipeline.list_stages() == ["valid/build"]
    assert "broken.broken_helper" not in sys.modules
    assert "broken.broken_helper" not in cloudpickle.list_registry_pickle_by_value()
