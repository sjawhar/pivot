"""Tests for pivot.discovery auto-discovery functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from conftest import stage_module_isolation
from pivot import discovery

if TYPE_CHECKING:
    from pathlib import Path


# =============================================================================
# Pipeline Discovery Tests (discover_pipeline)
# =============================================================================


def test_discover_pipeline_returns_none_when_no_files(set_project_root: Path) -> None:
    """discover_pipeline returns None when no pivot.yaml or pipeline.py."""
    result = discovery.discover_pipeline(set_project_root)
    assert result is None


def test_discover_pipeline_from_pipeline_py(set_project_root: Path) -> None:
    """discover_pipeline should find Pipeline instance in pipeline.py."""
    from pivot.pipeline.pipeline import Pipeline

    # Create pipeline.py that defines a Pipeline
    pipeline_code = """\
from pivot.pipeline.pipeline import Pipeline

pipeline = Pipeline("test_pipeline")

def _stage():
    pass

pipeline.register(_stage, name="my_stage")
"""
    (set_project_root / "pipeline.py").write_text(pipeline_code)

    result = discovery.discover_pipeline(set_project_root)

    assert result is not None
    assert isinstance(result, Pipeline)
    assert result.name == "test_pipeline"
    assert "my_stage" in result.list_stages()


def test_discover_pipeline_missing_pipeline_variable(set_project_root: Path) -> None:
    """discover_pipeline raises DiscoveryError when pipeline.py has Pipeline but wrong name."""
    # Create pipeline.py with Pipeline assigned to wrong variable name
    pipeline_code = """\
from pivot.pipeline.pipeline import Pipeline

# Note: we're assigning to 'my_pipe' instead of 'pipeline'
my_pipe = Pipeline("oops")
"""
    (set_project_root / "pipeline.py").write_text(pipeline_code)

    with pytest.raises(
        discovery.DiscoveryError,
        match="does not define a 'pipeline' variable.*Found Pipeline instance named 'my_pipe'",
    ):
        discovery.discover_pipeline(set_project_root)


def test_discover_pipeline_wrong_type(set_project_root: Path) -> None:
    """discover_pipeline raises DiscoveryError when 'pipeline' is not a Pipeline instance."""
    # Create pipeline.py with wrong type for 'pipeline'
    pipeline_code = """\
pipeline = "not a Pipeline"
"""
    (set_project_root / "pipeline.py").write_text(pipeline_code)

    with pytest.raises(
        discovery.DiscoveryError,
        match="not a Pipeline instance",
    ):
        discovery.discover_pipeline(set_project_root)


def test_discover_pipeline_both_files_raises_error(set_project_root: Path) -> None:
    """discover_pipeline raises DiscoveryError when both pivot.yaml and pipeline.py exist."""
    # Create pivot.yaml
    (set_project_root / "pivot.yaml").write_text("stages: {}")

    # Create pipeline.py
    pipeline_code = """\
from pivot.pipeline.pipeline import Pipeline
pipeline = Pipeline("test")
"""
    (set_project_root / "pipeline.py").write_text(pipeline_code)

    with pytest.raises(
        discovery.DiscoveryError,
        match="Found both pivot.yaml and pipeline.py",
    ):
        discovery.discover_pipeline(set_project_root)


def test_discover_pipeline_sys_exit_raises(set_project_root: Path) -> None:
    """discover_pipeline raises DiscoveryError when pipeline.py calls sys.exit()."""
    # Create pipeline.py that calls sys.exit
    pipeline_code = """\
import sys
sys.exit(42)
"""
    (set_project_root / "pipeline.py").write_text(pipeline_code)

    with pytest.raises(discovery.DiscoveryError, match=r"sys\.exit\(42\)"):
        discovery.discover_pipeline(set_project_root)


def test_discover_pipeline_runtime_error_raises(set_project_root: Path) -> None:
    """discover_pipeline raises DiscoveryError when pipeline.py has runtime error."""
    # Create pipeline.py with an error
    pipeline_code = """\
raise RuntimeError("intentional error")
"""
    (set_project_root / "pipeline.py").write_text(pipeline_code)

    with pytest.raises(discovery.DiscoveryError, match="Failed to load"):
        discovery.discover_pipeline(set_project_root)


def test_discover_pipeline_from_yaml(set_project_root: Path) -> None:
    """discover_pipeline returns Pipeline when only pivot.yaml exists."""
    from pivot.pipeline.pipeline import Pipeline

    # Create a simple stage module
    stages_py = set_project_root / "stages.py"
    stages_py.write_text(
        """\
import pathlib
from typing import Annotated, TypedDict
from pivot import loaders, outputs

class ProcessOutputs(TypedDict):
    out: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]

def process() -> ProcessOutputs:
    return {"out": pathlib.Path("output.txt")}
"""
    )

    # Create only pivot.yaml (no pipeline.py)
    (set_project_root / "pivot.yaml").write_text(
        """\
pipeline: yaml_pipeline
stages:
  process:
    python: stages.process
"""
    )

    with stage_module_isolation(set_project_root):
        result = discovery.discover_pipeline(set_project_root)

    assert result is not None
    assert isinstance(result, Pipeline)
    assert result.name == "yaml_pipeline"
    assert "process" in result.list_stages()


def test_discover_pipeline_from_yml(set_project_root: Path) -> None:
    """discover_pipeline returns Pipeline when only pivot.yml exists."""
    from pivot.pipeline.pipeline import Pipeline

    # Create a simple stage module
    stages_py = set_project_root / "stages.py"
    stages_py.write_text(
        """\
import pathlib
from typing import Annotated, TypedDict
from pivot import loaders, outputs

class AnalyzeOutputs(TypedDict):
    result: Annotated[pathlib.Path, outputs.Out("result.txt", loaders.PathOnly())]

def analyze() -> AnalyzeOutputs:
    return {"result": pathlib.Path("result.txt")}
"""
    )

    # Create only pivot.yml (note: .yml not .yaml)
    (set_project_root / "pivot.yml").write_text(
        """\
pipeline: yml_pipeline
stages:
  analyze:
    python: stages.analyze
"""
    )

    with stage_module_isolation(set_project_root):
        result = discovery.discover_pipeline(set_project_root)

    assert result is not None
    assert isinstance(result, Pipeline)
    assert result.name == "yml_pipeline"
    assert "analyze" in result.list_stages()


def test_discover_pipeline_invalid_yaml_raises(set_project_root: Path) -> None:
    """discover_pipeline raises DiscoveryError for invalid pivot.yaml."""
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
        discovery.discover_pipeline(set_project_root)
