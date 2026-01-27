"""Tests for pivot.discovery auto-discovery functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from conftest import stage_module_isolation
from helpers import register_test_stage
from pivot import discovery, outputs, registry

if TYPE_CHECKING:
    from pathlib import Path


# =============================================================================
# Basic Discovery Tests
# =============================================================================


def test_discover_returns_none_when_no_files(set_project_root: Path) -> None:
    """discover_and_register returns None when no pivot.yaml or pipeline.py."""
    result = discovery.discover_and_register(set_project_root)
    assert result is None


def test_discover_pivot_yaml(set_project_root: Path) -> None:
    """discover_and_register finds and loads pivot.yaml."""
    # Create stages module
    stages_py = set_project_root / "stages.py"
    stages_py.write_text(
        """\
def preprocess():
    pass
"""
    )

    # Create pivot.yaml
    pivot_yaml = set_project_root / "pivot.yaml"
    pivot_yaml.write_text(
        """\
stages:
  preprocess:
    python: stages.preprocess
    deps: {}
    outs:
      output: output.txt
"""
    )

    with stage_module_isolation(set_project_root):
        result = discovery.discover_and_register(set_project_root)

        assert result == str(pivot_yaml)
        assert "preprocess" in registry.REGISTRY.list_stages()


def test_discover_pivot_yml(set_project_root: Path) -> None:
    """discover_and_register finds and loads pivot.yml (alternate extension)."""
    # Create stages module
    stages_py = set_project_root / "stages.py"
    stages_py.write_text(
        """\
def analyze():
    pass
"""
    )

    # Create pivot.yml (note: .yml not .yaml)
    pivot_yml = set_project_root / "pivot.yml"
    pivot_yml.write_text(
        """\
stages:
  analyze:
    python: stages.analyze
    deps: {}
    outs:
      result: result.txt
"""
    )

    with stage_module_isolation(set_project_root):
        result = discovery.discover_and_register(set_project_root)

        assert result == str(pivot_yml)
        assert "analyze" in registry.REGISTRY.list_stages()


def test_discover_pipeline_py(set_project_root: Path) -> None:
    """discover_and_register finds and loads pipeline.py."""
    # Create pipeline.py that registers a stage using annotation-based outputs
    pipeline_py = set_project_root / "pipeline.py"
    pipeline_py.write_text(
        """\
import pathlib
from typing import Annotated
from pivot.registry import REGISTRY
from pivot.outputs import Out
from pivot.loaders import PathOnly

def my_stage() -> Annotated[pathlib.Path, Out("output.txt", PathOnly())]:
    pass

REGISTRY.register(my_stage)
"""
    )

    result = discovery.discover_and_register(set_project_root)

    assert result == str(pipeline_py)
    assert "my_stage" in registry.REGISTRY.list_stages()


def test_discover_both_files_raises_error(set_project_root: Path) -> None:
    """Having both pivot.yaml and pipeline.py raises DiscoveryError."""
    # Create stages module
    stages_py = set_project_root / "stages.py"
    stages_py.write_text(
        """\
def yaml_stage():
    pass
"""
    )

    # Create both pivot.yaml and pipeline.py
    pivot_yaml = set_project_root / "pivot.yaml"
    pivot_yaml.write_text(
        """\
stages:
  yaml_stage:
    python: stages.yaml_stage
"""
    )

    pipeline_py = set_project_root / "pipeline.py"
    pipeline_py.write_text(
        """\
from pivot import registry

def python_stage():
    pass

registry.REGISTRY.register(python_stage)
"""
    )

    with pytest.raises(discovery.DiscoveryError, match="Found both pivot.yaml and pipeline.py"):
        discovery.discover_and_register(set_project_root)


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_discover_invalid_pivot_yaml_raises(set_project_root: Path) -> None:
    """discover_and_register raises DiscoveryError for invalid pivot.yaml."""
    pivot_yaml = set_project_root / "pivot.yaml"
    pivot_yaml.write_text(
        """\
stages:
  broken:
    python: nonexistent.module.func
    outs: [out.txt]
"""
    )

    with pytest.raises(discovery.DiscoveryError, match="Failed to load"):
        discovery.discover_and_register(set_project_root)


def test_discover_invalid_pipeline_py_raises(set_project_root: Path) -> None:
    """discover_and_register raises DiscoveryError for invalid pipeline.py."""
    pipeline_py = set_project_root / "pipeline.py"
    pipeline_py.write_text(
        """\
# This will raise an error when executed
raise RuntimeError("intentional error")
"""
    )

    with pytest.raises(discovery.DiscoveryError, match="Failed to load"):
        discovery.discover_and_register(set_project_root)


def test_discover_pipeline_py_sys_exit_raises(set_project_root: Path) -> None:
    """discover_and_register raises DiscoveryError when pipeline.py calls sys.exit()."""
    pipeline_py = set_project_root / "pipeline.py"
    pipeline_py.write_text(
        """\
import sys
sys.exit(1)
"""
    )

    with pytest.raises(discovery.DiscoveryError, match="sys.exit"):
        discovery.discover_and_register(set_project_root)


# =============================================================================
# Helper Function Tests
# =============================================================================


def test_has_registered_stages_false_when_empty() -> None:
    """has_registered_stages returns False when no stages registered."""
    # Registry is already cleared by autouse clean_registry fixture
    assert discovery.has_registered_stages() is False


def test_has_registered_stages_true_after_registration(
    set_project_root: Path,
) -> None:
    """has_registered_stages returns True after stage registration."""
    import pathlib as pathlib_module
    from typing import Annotated

    from pivot import loaders

    def test_stage() -> Annotated[
        pathlib_module.Path, outputs.Out("test_output.txt", loaders.PathOnly())
    ]:
        return pathlib_module.Path("test_output.txt")

    register_test_stage(test_stage, name="test")
    assert discovery.has_registered_stages() is True
