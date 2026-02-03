"""Tests for pivot.discovery auto-discovery functionality."""

from __future__ import annotations

import logging
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
    """discover_pipeline returns None when no pivot.yaml or pipeline.py exist.

    Prevents regression where discovery fails instead of returning None.
    """
    result = discovery.discover_pipeline(set_project_root)
    assert result is None


def test_discover_pipeline_ignores_directories_with_config_names(set_project_root: Path) -> None:
    """discover_pipeline ignores directories named pivot.yaml or pipeline.py.

    Tests that _find_config_path_in_dir uses is_file() not exists(), preventing
    confusion when a directory happens to be named like a config file.
    """
    # Create directories with config file names
    (set_project_root / "pivot.yaml").mkdir()
    (set_project_root / "pipeline.py").mkdir()

    result = discovery.discover_pipeline(set_project_root)

    # Should return None, not try to parse directories as files
    assert result is None


def test_discover_pipeline_from_pipeline_py(set_project_root: Path) -> None:
    """discover_pipeline finds and loads Pipeline instance from pipeline.py.

    Tests the pipeline.py discovery path including stage registration.
    """
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


def test_discover_pipeline_py_no_pipeline_variable(set_project_root: Path) -> None:
    """discover_pipeline returns None when pipeline.py has no Pipeline at all.

    Tests the case where pipeline.py exists but doesn't define a Pipeline instance
    under any variable name. This is different from the "wrong name" case which
    raises an error.
    """
    # Create pipeline.py with no Pipeline instances
    pipeline_code = """\
# Just some module with no Pipeline
x = 1
y = "hello"
"""
    (set_project_root / "pipeline.py").write_text(pipeline_code)

    result = discovery.discover_pipeline(set_project_root)

    assert result is None


def test_discover_pipeline_missing_pipeline_variable(set_project_root: Path) -> None:
    """discover_pipeline raises DiscoveryError when Pipeline exists with wrong variable name.

    This catches the common mistake of creating a Pipeline but naming it something other
    than 'pipeline'. The error message should guide the user to rename it.
    """
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
    """discover_pipeline raises DiscoveryError when 'pipeline' variable is not a Pipeline.

    Prevents confusion when someone assigns a non-Pipeline value to 'pipeline'.
    """
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


@pytest.mark.parametrize(
    "yaml_name",
    [
        pytest.param("pivot.yaml", id="yaml"),
        pytest.param("pivot.yml", id="yml"),
    ],
)
def test_discover_pipeline_both_yaml_and_pipeline_py_raises_error(
    set_project_root: Path, yaml_name: str
) -> None:
    """discover_pipeline raises DiscoveryError when both YAML config and pipeline.py exist.

    Tests both pivot.yaml and pivot.yml extensions to ensure consistent behavior.
    Prevents ambiguity about which config to use.
    """
    # Create YAML config
    (set_project_root / yaml_name).write_text("stages: {}")

    # Create pipeline.py
    pipeline_code = """\
from pivot.pipeline.pipeline import Pipeline
pipeline = Pipeline("test")
"""
    (set_project_root / "pipeline.py").write_text(pipeline_code)

    with pytest.raises(
        discovery.DiscoveryError,
        match=f"Found both {yaml_name} and pipeline.py",
    ):
        discovery.discover_pipeline(set_project_root)


def test_discover_pipeline_prefers_yaml_over_yml(set_project_root: Path) -> None:
    """discover_pipeline prefers pivot.yaml when both pivot.yaml and pivot.yml exist.

    Tests the PIVOT_YAML_NAMES tuple ordering to ensure .yaml takes precedence.
    """
    from pivot.pipeline.pipeline import Pipeline

    # Create minimal stage module
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

    # Create both files with different names
    (set_project_root / "pivot.yaml").write_text(
        """\
pipeline: yaml_wins
stages:
  process:
    python: stages.process
"""
    )
    (set_project_root / "pivot.yml").write_text(
        """\
pipeline: yml_loses
stages:
  process:
    python: stages.process
"""
    )

    with stage_module_isolation(set_project_root):
        result = discovery.discover_pipeline(set_project_root)

    assert result is not None
    assert isinstance(result, Pipeline)
    # Should load from pivot.yaml, not pivot.yml
    assert result.name == "yaml_wins"


def test_discover_pipeline_sys_exit_raises(set_project_root: Path) -> None:
    """discover_pipeline raises DiscoveryError when pipeline.py calls sys.exit().

    Prevents silent failures when pipeline.py has top-level sys.exit() calls.
    """
    # Create pipeline.py that calls sys.exit
    pipeline_code = """\
import sys
sys.exit(42)
"""
    (set_project_root / "pipeline.py").write_text(pipeline_code)

    with pytest.raises(discovery.DiscoveryError, match=r"sys\.exit\(42\)"):
        discovery.discover_pipeline(set_project_root)


def test_discover_pipeline_runtime_error_raises(set_project_root: Path) -> None:
    """discover_pipeline wraps non-DiscoveryError exceptions in DiscoveryError.

    Tests generic exception handling during pipeline.py loading.
    """
    # Create pipeline.py with an error
    pipeline_code = """\
raise RuntimeError("intentional error")
"""
    (set_project_root / "pipeline.py").write_text(pipeline_code)

    with pytest.raises(discovery.DiscoveryError, match="Failed to load"):
        discovery.discover_pipeline(set_project_root)


def test_discover_pipeline_reraises_discovery_error(set_project_root: Path) -> None:
    """discover_pipeline re-raises DiscoveryError from _load_pipeline_from_module without wrapping.

    Tests that internal DiscoveryErrors (like wrong variable name) are not double-wrapped.
    """
    # Create pipeline.py that will trigger DiscoveryError
    pipeline_code = """\
from pivot.pipeline.pipeline import Pipeline
wrong_name = Pipeline("test")
"""
    (set_project_root / "pipeline.py").write_text(pipeline_code)

    # Should get the original DiscoveryError, not "Failed to load" wrapper
    with pytest.raises(
        discovery.DiscoveryError,
        match="does not define a 'pipeline' variable.*Found Pipeline instance named 'wrong_name'",
    ):
        discovery.discover_pipeline(set_project_root)


@pytest.mark.parametrize(
    "yaml_name,pipeline_name",
    [
        pytest.param("pivot.yaml", "yaml_pipeline", id="yaml"),
        pytest.param("pivot.yml", "yml_pipeline", id="yml"),
    ],
)
def test_discover_pipeline_from_yaml_files(
    set_project_root: Path, yaml_name: str, pipeline_name: str
) -> None:
    """discover_pipeline loads Pipeline from both pivot.yaml and pivot.yml files.

    Parametrized test to ensure both YAML extensions work correctly.
    """
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

    # Create YAML config
    (set_project_root / yaml_name).write_text(
        f"""\
pipeline: {pipeline_name}
stages:
  process:
    python: stages.process
"""
    )

    with stage_module_isolation(set_project_root):
        result = discovery.discover_pipeline(set_project_root)

    assert result is not None
    assert isinstance(result, Pipeline)
    assert result.name == pipeline_name
    assert "process" in result.list_stages()


def test_discover_pipeline_invalid_yaml_raises(set_project_root: Path) -> None:
    """discover_pipeline raises DiscoveryError for invalid pivot.yaml content.

    Tests error handling when YAML config references non-existent modules.
    """
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


# =============================================================================
# Parent Pipeline Discovery Tests (find_parent_pipeline_paths)
# =============================================================================


def test_find_parent_pipeline_paths_finds_pipeline_py(set_project_root: Path) -> None:
    """find_parent_pipeline_paths finds pipeline.py files in parent directories.

    Tests traversal order: closest parents first, stopping at specified directory.
    """
    (set_project_root / "pipeline.py").touch()
    mid = set_project_root / "mid"
    mid.mkdir()
    (mid / "pipeline.py").touch()
    child = mid / "child"
    child.mkdir()

    result = list(discovery.find_parent_pipeline_paths(child, stop_at=set_project_root))

    # Should find mid's pipeline.py first (closest parent), then root's
    assert len(result) == 2
    assert result[0] == mid / "pipeline.py"
    assert result[1] == set_project_root / "pipeline.py"


def test_find_parent_pipeline_paths_finds_pivot_yaml(set_project_root: Path) -> None:
    """find_parent_pipeline_paths finds pivot.yaml in parent directories."""
    (set_project_root / "pivot.yaml").touch()
    child = set_project_root / "child"
    child.mkdir()

    result = list(discovery.find_parent_pipeline_paths(child, stop_at=set_project_root))

    assert result == [set_project_root / "pivot.yaml"]


def test_find_parent_pipeline_paths_finds_pivot_yml(set_project_root: Path) -> None:
    """find_parent_pipeline_paths finds pivot.yml in parent directories."""
    (set_project_root / "pivot.yml").touch()
    child = set_project_root / "child"
    child.mkdir()

    result = list(discovery.find_parent_pipeline_paths(child, stop_at=set_project_root))

    assert result == [set_project_root / "pivot.yml"]


def test_find_parent_pipeline_paths_errors_on_both(set_project_root: Path) -> None:
    """find_parent_pipeline_paths raises DiscoveryError when directory has both configs.

    Prevents ambiguity during parent traversal, same as discover_pipeline.
    """
    (set_project_root / "pipeline.py").touch()
    (set_project_root / "pivot.yaml").touch()
    child = set_project_root / "child"
    child.mkdir()

    with pytest.raises(discovery.DiscoveryError, match="Found both"):
        list(discovery.find_parent_pipeline_paths(child, stop_at=set_project_root))


def test_find_parent_pipeline_paths_skips_own_directory(set_project_root: Path) -> None:
    """find_parent_pipeline_paths does not include start_dir's own config files.

    Tests that traversal starts from start_dir.parent, not start_dir itself.
    """
    (set_project_root / "pipeline.py").touch()

    result = list(
        discovery.find_parent_pipeline_paths(set_project_root, stop_at=set_project_root.parent)
    )

    assert result == []


def test_find_parent_pipeline_paths_stops_at_root(tmp_path: Path) -> None:
    """find_parent_pipeline_paths stops at filesystem root without infinite loop.

    Tests the current.parent == current safety check that prevents infinite loops
    when stop_at is above the actual filesystem root.
    """
    # Create a deep directory structure
    deep_dir = tmp_path / "a" / "b" / "c"
    deep_dir.mkdir(parents=True)

    # No stop_at specified would normally go to filesystem root
    # Should stop when it reaches filesystem root (parent == self)
    import pathlib

    result = list(
        discovery.find_parent_pipeline_paths(deep_dir, stop_at=pathlib.Path(tmp_path.root))
    )

    # Should not raise, should complete (likely finding nothing unless files exist in path)
    assert isinstance(result, list)


def test_find_parent_pipeline_paths_start_equals_stop(set_project_root: Path) -> None:
    """find_parent_pipeline_paths returns empty when start_dir equals stop_at.

    When start_dir equals stop_at, the range is empty (start_dir is exclusive),
    so no configs should be found. This also tests that the function doesn't
    traverse above stop_at.
    """
    # Create config at project root (which equals both start_dir and stop_at)
    (set_project_root / "pipeline.py").touch()

    result = list(discovery.find_parent_pipeline_paths(set_project_root, stop_at=set_project_root))

    # Should return empty - start_dir is exclusive, and parent is above stop_at
    assert result == []


def test_find_parent_pipeline_paths_does_not_traverse_above_stop_at(tmp_path: Path) -> None:
    """find_parent_pipeline_paths stops traversal at stop_at boundary.

    Tests that the function doesn't find configs in directories above stop_at,
    even if they exist.
    """
    # Structure: /root/above/project/child
    # stop_at = /root/above/project, start_dir = /root/above/project/child
    # Config exists at /root/above (should NOT be found)
    above = tmp_path / "above"
    project = above / "project"
    child = project / "child"
    child.mkdir(parents=True)

    # Put config ABOVE stop_at - should not be found
    (above / "pipeline.py").touch()

    result = list(discovery.find_parent_pipeline_paths(child, stop_at=project))

    # Should return empty - the only config is above stop_at
    assert result == []


# =============================================================================
# Load Pipeline From Path Tests (load_pipeline_from_path)
# =============================================================================


def test_load_pipeline_from_path_loads_pipeline_py(set_project_root: Path) -> None:
    """load_pipeline_from_path loads Pipeline from pipeline.py file."""
    pipeline_code = """
from pivot.pipeline import Pipeline
pipeline = Pipeline("test")
"""
    (set_project_root / "pipeline.py").write_text(pipeline_code)

    result = discovery.load_pipeline_from_path(set_project_root / "pipeline.py")

    assert result is not None
    assert result.name == "test"


def test_load_pipeline_from_path_loads_pivot_yaml(set_project_root: Path) -> None:
    """load_pipeline_from_path loads Pipeline from pivot.yaml file."""
    stages_py = set_project_root / "stages.py"
    stages_py.write_text(
        """\
import pathlib
from typing import Annotated, TypedDict
from pivot import loaders, outputs

class ExampleOutputs(TypedDict):
    out: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]

def example() -> ExampleOutputs:
    return {"out": pathlib.Path("output.txt")}
"""
    )
    yaml_content = """\
pipeline: yaml_test
stages:
  example:
    python: stages.example
"""
    (set_project_root / "pivot.yaml").write_text(yaml_content)

    with stage_module_isolation(set_project_root):
        result = discovery.load_pipeline_from_path(set_project_root / "pivot.yaml")

    assert result is not None
    assert result.name == "yaml_test"


def test_load_pipeline_from_path_loads_pivot_yml(set_project_root: Path) -> None:
    """load_pipeline_from_path loads Pipeline from pivot.yml file."""
    stages_py = set_project_root / "stages.py"
    stages_py.write_text(
        """\
import pathlib
from typing import Annotated, TypedDict
from pivot import loaders, outputs

class ExampleOutputs(TypedDict):
    out: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]

def example() -> ExampleOutputs:
    return {"out": pathlib.Path("output.txt")}
"""
    )
    yaml_content = """\
pipeline: yml_test
stages:
  example:
    python: stages.example
"""
    (set_project_root / "pivot.yml").write_text(yaml_content)

    with stage_module_isolation(set_project_root):
        result = discovery.load_pipeline_from_path(set_project_root / "pivot.yml")

    assert result is not None
    assert result.name == "yml_test"


def test_load_pipeline_from_path_returns_none_for_no_pipeline(set_project_root: Path) -> None:
    """load_pipeline_from_path returns None when pipeline.py has no Pipeline."""
    (set_project_root / "pipeline.py").write_text("x = 1\n")

    result = discovery.load_pipeline_from_path(set_project_root / "pipeline.py")

    assert result is None


def test_load_pipeline_from_path_returns_none_for_discovery_error(set_project_root: Path) -> None:
    """load_pipeline_from_path returns None when pipeline.py raises DiscoveryError.

    Tests that DiscoveryErrors (like wrong variable name) are swallowed and return None,
    since this function is designed for optional loading during parent traversal.
    """
    # Create pipeline.py with wrong variable name (triggers DiscoveryError)
    pipeline_code = """\
from pivot.pipeline.pipeline import Pipeline
wrong_name = Pipeline("test")
"""
    (set_project_root / "pipeline.py").write_text(pipeline_code)

    result = discovery.load_pipeline_from_path(set_project_root / "pipeline.py")

    assert result is None


def test_load_pipeline_from_path_logs_errors(
    set_project_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """load_pipeline_from_path logs errors at DEBUG level and returns None.

    Tests error handling for generic exceptions during loading.
    """
    (set_project_root / "pipeline.py").write_text("raise RuntimeError('fail')")

    with caplog.at_level(logging.DEBUG):
        result = discovery.load_pipeline_from_path(set_project_root / "pipeline.py")

    assert result is None
    assert "Failed to load" in caplog.text


def test_load_pipeline_from_path_unknown_file_type(
    set_project_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """load_pipeline_from_path returns None and logs for unknown file types.

    Tests the fallback path when file is neither pivot.yaml, pivot.yml, nor pipeline.py.
    """
    unknown_file = set_project_root / "config.toml"
    unknown_file.write_text("[tool.something]")

    with caplog.at_level(logging.DEBUG):
        result = discovery.load_pipeline_from_path(unknown_file)

    assert result is None
    assert "Unknown pipeline file type" in caplog.text
