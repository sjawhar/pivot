"""Tests for pivot.discovery auto-discovery functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pivot import discovery, registry

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


@pytest.fixture
def project_root(tmp_path: Path, mocker: MockerFixture) -> Path:
    """Set up a mock project root."""
    mocker.patch("pivot.project._project_root_cache", tmp_path)
    return tmp_path


# =============================================================================
# Basic Discovery Tests
# =============================================================================


def test_discover_returns_none_when_no_files(project_root: Path) -> None:
    """discover_and_register returns None when no pivot.yaml or pipeline.py."""
    result = discovery.discover_and_register(project_root)
    assert result is None


def test_discover_pivot_yaml(project_root: Path) -> None:
    """discover_and_register finds and loads pivot.yaml."""
    import sys

    # Create stages module
    stages_py = project_root / "stages.py"
    stages_py.write_text(
        """\
def preprocess():
    pass
"""
    )

    # Create pivot.yaml
    pivot_yaml = project_root / "pivot.yaml"
    pivot_yaml.write_text(
        """\
stages:
  preprocess:
    python: stages.preprocess
    deps: []
    outs: [output.txt]
"""
    )

    # Add project root to sys.path for module imports
    sys.path.insert(0, str(project_root))
    try:
        result = discovery.discover_and_register(project_root)

        assert result == str(pivot_yaml)
        assert "preprocess" in registry.REGISTRY.list_stages()
    finally:
        sys.path.remove(str(project_root))
        if "stages" in sys.modules:
            del sys.modules["stages"]


def test_discover_pivot_yml(project_root: Path) -> None:
    """discover_and_register finds and loads pivot.yml (alternate extension)."""
    import sys

    # Create stages module
    stages_py = project_root / "stages.py"
    stages_py.write_text(
        """\
def analyze():
    pass
"""
    )

    # Create pivot.yml (note: .yml not .yaml)
    pivot_yml = project_root / "pivot.yml"
    pivot_yml.write_text(
        """\
stages:
  analyze:
    python: stages.analyze
    deps: []
    outs: [result.txt]
"""
    )

    sys.path.insert(0, str(project_root))
    try:
        result = discovery.discover_and_register(project_root)

        assert result == str(pivot_yml)
        assert "analyze" in registry.REGISTRY.list_stages()
    finally:
        sys.path.remove(str(project_root))
        if "stages" in sys.modules:
            del sys.modules["stages"]


def test_discover_pipeline_py(project_root: Path) -> None:
    """discover_and_register finds and loads pipeline.py."""
    # Create pipeline.py that registers a stage
    pipeline_py = project_root / "pipeline.py"
    pipeline_py.write_text(
        """\
from pivot import Pipeline

def my_stage():
    pass

pipeline = Pipeline()
pipeline.add_stage(my_stage, outs=['output.txt'])
"""
    )

    result = discovery.discover_and_register(project_root)

    assert result == str(pipeline_py)
    assert "my_stage" in registry.REGISTRY.list_stages()


def test_discover_both_files_raises_error(project_root: Path) -> None:
    """Having both pivot.yaml and pipeline.py raises DiscoveryError."""
    # Create stages module
    stages_py = project_root / "stages.py"
    stages_py.write_text(
        """\
def yaml_stage():
    pass
"""
    )

    # Create both pivot.yaml and pipeline.py
    pivot_yaml = project_root / "pivot.yaml"
    pivot_yaml.write_text(
        """\
stages:
  yaml_stage:
    python: stages.yaml_stage
    deps: []
    outs: [yaml_output.txt]
"""
    )

    pipeline_py = project_root / "pipeline.py"
    pipeline_py.write_text(
        """\
from pivot import Pipeline

def python_stage():
    pass

pipeline = Pipeline()
pipeline.add_stage(python_stage, outs=['python_output.txt'])
"""
    )

    with pytest.raises(discovery.DiscoveryError, match="Found both pivot.yaml and pipeline.py"):
        discovery.discover_and_register(project_root)


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_discover_invalid_pivot_yaml_raises(project_root: Path) -> None:
    """discover_and_register raises DiscoveryError for invalid pivot.yaml."""
    pivot_yaml = project_root / "pivot.yaml"
    pivot_yaml.write_text(
        """\
stages:
  broken:
    python: nonexistent.module.func
    outs: [out.txt]
"""
    )

    with pytest.raises(discovery.DiscoveryError, match="Failed to load"):
        discovery.discover_and_register(project_root)


def test_discover_invalid_pipeline_py_raises(project_root: Path) -> None:
    """discover_and_register raises DiscoveryError for invalid pipeline.py."""
    pipeline_py = project_root / "pipeline.py"
    pipeline_py.write_text(
        """\
# This will raise an error when executed
raise RuntimeError("intentional error")
"""
    )

    with pytest.raises(discovery.DiscoveryError, match="Failed to load"):
        discovery.discover_and_register(project_root)


def test_discover_pipeline_py_sys_exit_raises(project_root: Path) -> None:
    """discover_and_register raises DiscoveryError when pipeline.py calls sys.exit()."""
    pipeline_py = project_root / "pipeline.py"
    pipeline_py.write_text(
        """\
import sys
sys.exit(1)
"""
    )

    with pytest.raises(discovery.DiscoveryError, match="sys.exit"):
        discovery.discover_and_register(project_root)


# =============================================================================
# Helper Function Tests
# =============================================================================


def test_has_registered_stages_false_when_empty() -> None:
    """has_registered_stages returns False when no stages registered."""
    registry.REGISTRY.clear()
    assert discovery.has_registered_stages() is False


def test_has_registered_stages_true_after_registration(
    project_root: Path,
) -> None:
    """has_registered_stages returns True after stage registration."""

    def test_stage() -> None:
        pass

    registry.REGISTRY.register(test_stage, name="test", deps=[], outs=[])
    assert discovery.has_registered_stages() is True
