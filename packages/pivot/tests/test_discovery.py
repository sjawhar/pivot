from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import pytest

from pivot import discovery
from pivot.pipeline.pipeline import Pipeline

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def _write_pipeline_module(path: pathlib.Path, body: str) -> None:
    path.write_text(body)


def test_find_config_in_dir_prefers_yaml_and_validates_ambiguity(tmp_path: pathlib.Path) -> None:
    assert discovery.find_config_in_dir(tmp_path) is None

    yaml_path = tmp_path / "pivot.yaml"
    yaml_path.write_text("pipelines: []\n")
    assert discovery.find_config_in_dir(tmp_path) == yaml_path

    yaml_path.unlink()
    py_path = tmp_path / "pipeline.py"
    py_path.write_text("pipeline = None\n")
    assert discovery.find_config_in_dir(tmp_path) == py_path

    yaml_path.write_text("pipelines: []\n")
    with pytest.raises(discovery.DiscoveryError, match="Found both"):
        discovery.find_config_in_dir(tmp_path)


def test_load_pipeline_from_module_handles_missing_wrong_and_valid_pipeline(
    tmp_path: pathlib.Path,
) -> None:
    module_path = tmp_path / "pipeline.py"

    _write_pipeline_module(module_path, "x = 1\n")
    assert discovery._load_pipeline_from_module(module_path) is None  # noqa: SLF001

    _write_pipeline_module(module_path, "pipeline = 123\n")
    with pytest.raises(discovery.DiscoveryError, match="not a Pipeline instance"):
        discovery._load_pipeline_from_module(module_path)  # noqa: SLF001

    _write_pipeline_module(
        module_path,
        """
import pathlib
from pivot.pipeline.pipeline import Pipeline
other_name = Pipeline("valid", root=pathlib.Path(__file__).parent)
""",
    )
    with pytest.raises(discovery.DiscoveryError, match="rename it to 'pipeline'"):
        discovery._load_pipeline_from_module(module_path)  # noqa: SLF001

    _write_pipeline_module(
        module_path,
        """
import pathlib
from pivot.pipeline.pipeline import Pipeline
pipeline = Pipeline("valid", root=pathlib.Path(__file__).parent)
""",
    )
    loaded = discovery._load_pipeline_from_module(module_path)  # noqa: SLF001
    assert loaded is not None
    assert loaded.name == "valid"


def test_discover_pipeline_checks_cwd_then_root_and_loads_yaml(
    mocker: MockerFixture,
    tmp_path: pathlib.Path,
) -> None:
    root = tmp_path
    cwd = root / "sub"
    cwd.mkdir()
    cwd_yaml = cwd / "pivot.yaml"
    cwd_yaml.write_text("pipelines: []\n")
    root_py = root / "pipeline.py"
    root_py.write_text("pipeline = None\n")

    mocker.patch.object(pathlib.Path, "cwd", autospec=True, return_value=cwd)
    loaded = Pipeline("yaml_loaded", root=cwd)
    load_yaml = mocker.patch.object(
        discovery.pipeline_config,
        "load_pipeline_from_yaml",
        autospec=True,
        return_value=loaded,
    )

    pipeline = discovery.discover_pipeline(project_root=root)

    assert pipeline is loaded
    load_yaml.assert_called_once_with(cwd_yaml)


def test_discover_pipeline_wraps_resolution_and_load_errors(
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    mocker.patch.object(pathlib.Path, "cwd", autospec=True, side_effect=OSError("bad cwd"))
    with pytest.raises(discovery.DiscoveryError, match="Failed to resolve paths"):
        discovery.discover_pipeline(project_root=tmp_path)

    root = tmp_path
    yaml_path = root / "pivot.yaml"
    yaml_path.write_text("pipelines: []\n")
    monkeypatch.setattr(pathlib.Path, "cwd", lambda: root)
    mocker.patch.object(
        discovery.pipeline_config,
        "load_pipeline_from_yaml",
        autospec=True,
        side_effect=discovery.pipeline_config.PipelineConfigError("bad yaml"),
    )
    with pytest.raises(discovery.DiscoveryError, match="Failed to load"):
        discovery.discover_pipeline(project_root=root)

    yaml_path.unlink()
    py_path = root / "pipeline.py"
    py_path.write_text("import sys\nsys.exit(2)\n")
    with pytest.raises(discovery.DiscoveryError, match="called sys.exit"):
        discovery.discover_pipeline(project_root=root)


def test_discover_all_pipelines_combines_and_marks_resolved(
    mocker: MockerFixture,
    tmp_path: pathlib.Path,
) -> None:
    a = Pipeline("a", root=tmp_path / "a")
    b = Pipeline("b", root=tmp_path / "b")
    a._registry.add_existing(  # noqa: SLF001
        {
            "func": lambda: None,
            "name": "train",
            "deps": {},
            "outs": [],
            "params": None,
            "mutex": [],
            "variant": None,
            "signature": None,
            "fingerprint": {},
            "params_arg_name": None,
            "state_dir": None,
            "collection_params": {},
        }
    )
    b._registry.add_existing(  # noqa: SLF001
        {
            "func": lambda: None,
            "name": "train",
            "deps": {},
            "outs": [],
            "params": None,
            "mutex": [],
            "variant": None,
            "signature": None,
            "fingerprint": {},
            "params_arg_name": None,
            "state_dir": None,
            "collection_params": {},
        }
    )

    path_a = tmp_path / "a" / "pipeline.py"
    path_b = tmp_path / "b" / "pipeline.py"
    mocker.patch.object(
        discovery, "glob_all_pipelines", autospec=True, return_value=[path_a, path_b]
    )
    mocker.patch.object(discovery, "load_pipeline_from_path", autospec=True, side_effect=[a, b])

    combined = discovery._discover_all_pipelines(tmp_path)  # noqa: SLF001

    assert combined is not None
    assert sorted(combined.list_stages()) == ["b/train", "train"]
    assert combined._external_deps_resolved  # noqa: SLF001


def test_parent_and_dependency_path_discovery(tmp_path: pathlib.Path) -> None:
    root = tmp_path
    child = root / "a" / "b"
    child.mkdir(parents=True)
    (root / "pipeline.py").write_text("pipeline = None\n")
    (root / "a" / "pivot.yaml").write_text("pipelines: []\n")

    parent_paths = list(discovery.find_parent_pipeline_paths(child, root))
    dep_paths = list(discovery.find_pipeline_paths_for_dependency(child / "data.csv", root))

    assert parent_paths == [root / "a" / "pivot.yaml", root / "pipeline.py"]
    assert dep_paths == [root / "a" / "pivot.yaml", root / "pipeline.py"]


def test_glob_all_pipelines_skips_ignored_dirs_and_detects_ambiguity(
    tmp_path: pathlib.Path,
) -> None:
    (tmp_path / "keep").mkdir()
    (tmp_path / "keep" / "pipeline.py").write_text("pipeline = None\n")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "pipeline.py").write_text("pipeline = None\n")

    found = discovery.glob_all_pipelines(tmp_path)
    assert found == [tmp_path / "keep" / "pipeline.py"]

    ambiguous = tmp_path / "bad"
    ambiguous.mkdir()
    (ambiguous / "pipeline.py").write_text("pipeline = None\n")
    (ambiguous / "pivot.yaml").write_text("pipelines: []\n")
    with pytest.raises(discovery.DiscoveryError, match="Found both"):
        discovery.glob_all_pipelines(tmp_path)


def test_load_pipeline_from_path_handles_yaml_python_and_unknown(
    mocker: MockerFixture,
    tmp_path: pathlib.Path,
) -> None:
    yaml_path = tmp_path / "pivot.yaml"
    yaml_path.write_text("pipelines: []\n")
    py_path = tmp_path / "pipeline.py"
    py_path.write_text("pipeline = None\n")
    other_path = tmp_path / "other.txt"
    other_path.write_text("x\n")

    loaded_yaml = Pipeline("yaml", root=tmp_path)
    mocker.patch.object(
        discovery.pipeline_config,
        "load_pipeline_from_yaml",
        autospec=True,
        return_value=loaded_yaml,
    )
    assert discovery.load_pipeline_from_path(yaml_path) is loaded_yaml

    loaded_py = Pipeline("py", root=tmp_path)
    load_module = mocker.patch.object(
        discovery,
        "_load_pipeline_from_module",
        autospec=True,
        return_value=loaded_py,
    )
    assert discovery.load_pipeline_from_path(py_path) is loaded_py

    load_module.side_effect = discovery.DiscoveryError("bad name")
    assert discovery.load_pipeline_from_path(py_path) is None
    assert discovery.load_pipeline_from_path(other_path) is None
