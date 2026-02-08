"""Tests for pivot.discovery auto-discovery functionality."""

from __future__ import annotations

import logging
import pathlib

import pytest

from conftest import stage_module_isolation
from pivot import discovery

# =============================================================================
# Pipeline Discovery Tests (discover_pipeline)
# =============================================================================


def test_discover_pipeline_returns_none_when_no_files(set_project_root: pathlib.Path) -> None:
    """discover_pipeline returns None when no pivot.yaml or pipeline.py exist.

    Prevents regression where discovery fails instead of returning None.
    """
    result = discovery.discover_pipeline(set_project_root)
    assert result is None


def test_discover_pipeline_ignores_directories_with_config_names(
    set_project_root: pathlib.Path,
) -> None:
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


def test_discover_pipeline_from_pipeline_py(set_project_root: pathlib.Path) -> None:
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


def test_discover_pipeline_prefers_cwd_over_project_root(
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """discover_pipeline finds pipeline.py in cwd before checking project root.

    When running from a subdirectory that has its own pipeline.py, discovery
    should use that instead of the project root's pipeline.py.
    """
    from pivot.pipeline.pipeline import Pipeline

    # Create pipeline.py at project root
    root_pipeline_code = """\
from pivot.pipeline.pipeline import Pipeline
pipeline = Pipeline("root_pipeline")
"""
    (set_project_root / "pipeline.py").write_text(root_pipeline_code)

    # Create subdirectory with its own pipeline.py
    subdir = set_project_root / "subproject"
    subdir.mkdir()
    subdir_pipeline_code = """\
from pivot.pipeline.pipeline import Pipeline
pipeline = Pipeline("subdir_pipeline")
"""
    (subdir / "pipeline.py").write_text(subdir_pipeline_code)

    # Change cwd to subdirectory
    monkeypatch.chdir(subdir)

    # Discovery should find subdir's pipeline, not root's
    result = discovery.discover_pipeline(set_project_root)

    assert result is not None
    assert isinstance(result, Pipeline)
    assert result.name == "subdir_pipeline"


@pytest.mark.parametrize(
    "yaml_name",
    [
        pytest.param("pivot.yaml", id="yaml"),
        pytest.param("pivot.yml", id="yml"),
    ],
)
def test_discover_pipeline_prefers_cwd_yaml_over_project_root(
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch, yaml_name: str
) -> None:
    """discover_pipeline finds YAML config in cwd before checking project root.

    Tests that cwd-first discovery works for pivot.yaml/yml files, not just pipeline.py.
    This ensures YAML configs in subdirectories are preferred over root configs.
    """
    from pivot.pipeline.pipeline import Pipeline

    # Create minimal stage module for YAML configs
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

    # Create YAML config at project root
    (set_project_root / yaml_name).write_text(
        """\
pipeline: root_pipeline
stages:
  process:
    python: stages.process
"""
    )

    # Create subdirectory with its own YAML config
    subdir = set_project_root / "subproject"
    subdir.mkdir()
    (subdir / yaml_name).write_text(
        """\
pipeline: subdir_pipeline
stages:
  process:
    python: stages.process
"""
    )

    # Change cwd to subdirectory
    monkeypatch.chdir(subdir)

    # Discovery should find subdir's YAML config, not root's
    with stage_module_isolation(set_project_root):
        result = discovery.discover_pipeline(set_project_root)

    assert result is not None
    assert isinstance(result, Pipeline)
    assert result.name == "subdir_pipeline"


def test_discover_pipeline_cwd_invalid_config_raises_error(
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """discover_pipeline raises error when cwd has invalid config even if root is valid.

    When cwd has a broken pipeline config, discovery should fail immediately rather
    than falling back to the root config. This enforces fail-fast behavior and prevents
    confusing scenarios where users think they're using cwd config but root is used.
    """

    # Create valid pipeline.py at project root
    root_pipeline_code = """\
from pivot.pipeline.pipeline import Pipeline
pipeline = Pipeline("root_pipeline")
"""
    (set_project_root / "pipeline.py").write_text(root_pipeline_code)

    # Create subdirectory with BROKEN pipeline.py
    subdir = set_project_root / "subproject"
    subdir.mkdir()
    broken_pipeline_code = """\
from pivot.pipeline.pipeline import Pipeline
# Wrong variable name - should raise DiscoveryError
wrong_name = Pipeline("subdir_pipeline")
"""
    (subdir / "pipeline.py").write_text(broken_pipeline_code)

    # Change cwd to subdirectory with broken config
    monkeypatch.chdir(subdir)

    # Should raise error from cwd config, not silently use root
    with pytest.raises(
        discovery.DiscoveryError,
        match="does not define a 'pipeline' variable.*Found Pipeline instance named 'wrong_name'",
    ):
        discovery.discover_pipeline(set_project_root)


def test_discover_pipeline_cwd_equals_root_checks_once(
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """discover_pipeline doesn't double-check when cwd equals project root.

    When running from the project root, discovery should check the directory only once,
    not twice (once as cwd, once as root). This test verifies the cwd != root_resolved
    guard works correctly.
    """
    from pivot.pipeline.pipeline import Pipeline

    # Create pipeline.py at project root
    root_pipeline_code = """\
from pivot.pipeline.pipeline import Pipeline
pipeline = Pipeline("root_pipeline")
"""
    (set_project_root / "pipeline.py").write_text(root_pipeline_code)

    # Change cwd to project root (cwd == root)
    monkeypatch.chdir(set_project_root)

    result = discovery.discover_pipeline(set_project_root)

    # Should successfully find pipeline (checked once)
    assert result is not None
    assert isinstance(result, Pipeline)
    assert result.name == "root_pipeline"


def test_discover_pipeline_cwd_outside_project_uses_root(
    set_project_root: pathlib.Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """discover_pipeline uses root config when cwd is outside project tree.

    When running from a directory outside the project (e.g., parent or /tmp),
    discovery should skip cwd check and use project root config only.
    """
    from pivot.pipeline.pipeline import Pipeline

    # Create pipeline.py at project root
    root_pipeline_code = """\
from pivot.pipeline.pipeline import Pipeline
pipeline = Pipeline("root_pipeline")
"""
    (set_project_root / "pipeline.py").write_text(root_pipeline_code)

    # Create a separate temp directory outside the project root
    outside_dir = tmp_path_factory.mktemp("outside")
    monkeypatch.chdir(outside_dir)

    result = discovery.discover_pipeline(set_project_root)

    # Should find root config (cwd is outside project tree)
    assert result is not None
    assert isinstance(result, Pipeline)
    assert result.name == "root_pipeline"


def test_discover_pipeline_mixed_config_types_prefers_cwd(
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """discover_pipeline prefers cwd config regardless of type differences.

    When cwd has pipeline.py and root has pivot.yaml (or vice versa),
    the cwd config should win. This verifies cwd-first logic doesn't
    incorrectly prioritize YAML over Python across directories.
    """
    from pivot.pipeline.pipeline import Pipeline

    # Create minimal stage module for YAML config
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

    # Create pivot.yaml at project root
    (set_project_root / "pivot.yaml").write_text(
        """\
pipeline: root_yaml_pipeline
stages:
  process:
    python: stages.process
"""
    )

    # Create subdirectory with pipeline.py (different type)
    subdir = set_project_root / "subproject"
    subdir.mkdir()
    subdir_pipeline_code = """\
from pivot.pipeline.pipeline import Pipeline
pipeline = Pipeline("subdir_py_pipeline")
"""
    (subdir / "pipeline.py").write_text(subdir_pipeline_code)

    # Change cwd to subdirectory
    monkeypatch.chdir(subdir)

    # Discovery should find subdir's pipeline.py, not root's pivot.yaml
    with stage_module_isolation(set_project_root):
        result = discovery.discover_pipeline(set_project_root)

    assert result is not None
    assert isinstance(result, Pipeline)
    assert result.name == "subdir_py_pipeline"


def test_discover_pipeline_py_no_pipeline_variable(set_project_root: pathlib.Path) -> None:
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


def test_discover_pipeline_missing_pipeline_variable(set_project_root: pathlib.Path) -> None:
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


def test_discover_pipeline_wrong_type(set_project_root: pathlib.Path) -> None:
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
    set_project_root: pathlib.Path, yaml_name: str
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


def test_discover_pipeline_prefers_yaml_over_yml(set_project_root: pathlib.Path) -> None:
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


def test_discover_pipeline_sys_exit_raises(set_project_root: pathlib.Path) -> None:
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


def test_discover_pipeline_runtime_error_raises(set_project_root: pathlib.Path) -> None:
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


def test_discover_pipeline_reraises_discovery_error(set_project_root: pathlib.Path) -> None:
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
    set_project_root: pathlib.Path, yaml_name: str, pipeline_name: str
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


def test_discover_pipeline_invalid_yaml_raises(set_project_root: pathlib.Path) -> None:
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


def test_find_parent_pipeline_paths_finds_pipeline_py(set_project_root: pathlib.Path) -> None:
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


def test_find_parent_pipeline_paths_finds_pivot_yaml(set_project_root: pathlib.Path) -> None:
    """find_parent_pipeline_paths finds pivot.yaml in parent directories."""
    (set_project_root / "pivot.yaml").touch()
    child = set_project_root / "child"
    child.mkdir()

    result = list(discovery.find_parent_pipeline_paths(child, stop_at=set_project_root))

    assert result == [set_project_root / "pivot.yaml"]


def test_find_parent_pipeline_paths_finds_pivot_yml(set_project_root: pathlib.Path) -> None:
    """find_parent_pipeline_paths finds pivot.yml in parent directories."""
    (set_project_root / "pivot.yml").touch()
    child = set_project_root / "child"
    child.mkdir()

    result = list(discovery.find_parent_pipeline_paths(child, stop_at=set_project_root))

    assert result == [set_project_root / "pivot.yml"]


def test_find_parent_pipeline_paths_errors_on_both(set_project_root: pathlib.Path) -> None:
    """find_parent_pipeline_paths raises DiscoveryError when directory has both configs.

    Prevents ambiguity during parent traversal, same as discover_pipeline.
    """
    (set_project_root / "pipeline.py").touch()
    (set_project_root / "pivot.yaml").touch()
    child = set_project_root / "child"
    child.mkdir()

    with pytest.raises(discovery.DiscoveryError, match="Found both"):
        list(discovery.find_parent_pipeline_paths(child, stop_at=set_project_root))


def test_find_parent_pipeline_paths_skips_own_directory(set_project_root: pathlib.Path) -> None:
    """find_parent_pipeline_paths does not include start_dir's own config files.

    Tests that traversal starts from start_dir.parent, not start_dir itself.
    """
    (set_project_root / "pipeline.py").touch()

    result = list(
        discovery.find_parent_pipeline_paths(set_project_root, stop_at=set_project_root.parent)
    )

    assert result == []


def test_find_parent_pipeline_paths_stops_at_root(tmp_path: pathlib.Path) -> None:
    """find_parent_pipeline_paths stops at filesystem root without infinite loop.

    Tests the current.parent == current safety check that prevents infinite loops
    when stop_at is above the actual filesystem root.
    """
    # Create a deep directory structure
    deep_dir = tmp_path / "a" / "b" / "c"
    deep_dir.mkdir(parents=True)

    # No stop_at specified would normally go to filesystem root
    # Should stop when it reaches filesystem root (parent == self)
    result = list(
        discovery.find_parent_pipeline_paths(deep_dir, stop_at=pathlib.Path(tmp_path.root))
    )

    # Should not raise, should complete (likely finding nothing unless files exist in path)
    assert isinstance(result, list)


def test_find_parent_pipeline_paths_start_equals_stop(set_project_root: pathlib.Path) -> None:
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


def test_find_parent_pipeline_paths_does_not_traverse_above_stop_at(tmp_path: pathlib.Path) -> None:
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


def test_load_pipeline_from_path_loads_pipeline_py(set_project_root: pathlib.Path) -> None:
    """load_pipeline_from_path loads Pipeline from pipeline.py file."""
    pipeline_code = """
from pivot.pipeline import Pipeline
pipeline = Pipeline("test")
"""
    (set_project_root / "pipeline.py").write_text(pipeline_code)

    result = discovery.load_pipeline_from_path(set_project_root / "pipeline.py")

    assert result is not None
    assert result.name == "test"


def test_load_pipeline_from_path_loads_pivot_yaml(set_project_root: pathlib.Path) -> None:
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


def test_load_pipeline_from_path_loads_pivot_yml(set_project_root: pathlib.Path) -> None:
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


def test_load_pipeline_from_path_returns_none_for_no_pipeline(
    set_project_root: pathlib.Path,
) -> None:
    """load_pipeline_from_path returns None when pipeline.py has no Pipeline."""
    (set_project_root / "pipeline.py").write_text("x = 1\n")

    result = discovery.load_pipeline_from_path(set_project_root / "pipeline.py")

    assert result is None


def test_load_pipeline_from_path_returns_none_for_discovery_error(
    set_project_root: pathlib.Path,
) -> None:
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
    set_project_root: pathlib.Path, caplog: pytest.LogCaptureFixture
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
    set_project_root: pathlib.Path, caplog: pytest.LogCaptureFixture
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


# =============================================================================
# Path Resolution Error Tests
# =============================================================================


def test_discover_pipeline_raises_on_path_resolution_failure(
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """discover_pipeline raises DiscoveryError when path resolution fails.

    Tests error handling for scenarios like broken symlinks or permission errors
    during path resolution.
    """
    original_resolve = pathlib.Path.resolve

    def mock_resolve(self: pathlib.Path, strict: bool = False) -> pathlib.Path:
        if "broken" in str(self):
            raise OSError("Simulated path resolution failure")
        return original_resolve(self, strict=strict)

    monkeypatch.setattr(pathlib.Path, "resolve", mock_resolve)

    # Create a directory whose name triggers the mocked error
    broken_root = set_project_root / "broken_link"
    broken_root.mkdir()

    with pytest.raises(discovery.DiscoveryError, match="Failed to resolve paths"):
        discovery.discover_pipeline(broken_root)


def test_find_parent_pipeline_paths_raises_on_path_resolution_failure(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """find_parent_pipeline_paths raises DiscoveryError when path resolution fails.

    Tests error handling for OSError during path resolution in parent traversal.
    """
    original_resolve = pathlib.Path.resolve

    def mock_resolve(self: pathlib.Path, strict: bool = False) -> pathlib.Path:
        if "broken" in str(self):
            raise OSError("Simulated path resolution failure")
        return original_resolve(self, strict=strict)

    monkeypatch.setattr(pathlib.Path, "resolve", mock_resolve)

    broken_dir = tmp_path / "broken_dir"
    broken_dir.mkdir()

    with pytest.raises(discovery.DiscoveryError, match="Failed to resolve paths"):
        list(discovery.find_parent_pipeline_paths(broken_dir, stop_at=tmp_path))


# =============================================================================
# Dependency Pipeline Discovery Tests (find_pipeline_paths_for_dependency)
# =============================================================================


def test_find_pipeline_paths_for_dependency_finds_sibling(
    tmp_path: pathlib.Path,
) -> None:
    """Should find pipeline in dependency's directory, not just parents."""
    # Create sibling pipeline structure:
    # tmp_path/
    #   sibling_a/pipeline.py  <- consuming pipeline
    #   sibling_b/pipeline.py  <- produces the dependency
    sibling_a = tmp_path / "sibling_a"
    sibling_b = tmp_path / "sibling_b"
    sibling_a.mkdir()
    sibling_b.mkdir()

    (sibling_a / "pipeline.py").write_text("# consumer")
    (sibling_b / "pipeline.py").write_text("# producer")

    # Dependency path is in sibling_b
    dep_path = sibling_b / "data" / "output.csv"

    paths = list(discovery.find_pipeline_paths_for_dependency(dep_path, tmp_path))

    # Should find sibling_b's pipeline (closest to dependency)
    assert sibling_b / "pipeline.py" in paths


def test_find_pipeline_paths_for_dependency_nested(
    tmp_path: pathlib.Path,
) -> None:
    """Should find pipelines at multiple levels when dependency is deeply nested."""
    # Structure:
    # tmp_path/
    #   pipeline.py           <- root pipeline
    #   subdir/
    #     pipeline.py         <- intermediate pipeline
    #     deep/
    #       data/output.csv   <- dependency location

    subdir = tmp_path / "subdir"
    deep = subdir / "deep"
    deep.mkdir(parents=True)

    (tmp_path / "pipeline.py").write_text("# root")
    (subdir / "pipeline.py").write_text("# intermediate")

    dep_path = deep / "data" / "output.csv"

    paths = list(discovery.find_pipeline_paths_for_dependency(dep_path, tmp_path))

    # Should find both, closest first
    assert paths == [subdir / "pipeline.py", tmp_path / "pipeline.py"]


def test_find_pipeline_paths_for_dependency_stops_at_project_root(
    tmp_path: pathlib.Path,
) -> None:
    """Should not traverse above project root."""
    # Dependency outside project root should still be bounded
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "pipeline.py").write_text("# project root")

    # Dep path within project
    dep_path = project_root / "data" / "file.csv"

    paths = list(discovery.find_pipeline_paths_for_dependency(dep_path, project_root))

    assert paths == [project_root / "pipeline.py"]


def test_find_pipeline_paths_for_dependency_no_pipelines(
    tmp_path: pathlib.Path,
) -> None:
    """Should return empty when no pipelines exist."""
    subdir = tmp_path / "empty"
    subdir.mkdir()
    dep_path = subdir / "data.csv"

    paths = list(discovery.find_pipeline_paths_for_dependency(dep_path, tmp_path))

    assert paths == []


# =============================================================================
# glob_all_pipelines Tests
# =============================================================================


def test_glob_all_pipelines_finds_pipeline_py(set_project_root: pathlib.Path) -> None:
    """glob_all_pipelines discovers pipeline.py files in subdirectories."""
    sub = set_project_root / "sub"
    sub.mkdir()
    (sub / "pipeline.py").write_text("pipeline = None")

    result = discovery.glob_all_pipelines(set_project_root)

    assert len(result) == 1
    assert result[0] == sub / "pipeline.py"


def test_glob_all_pipelines_finds_pivot_yaml(set_project_root: pathlib.Path) -> None:
    """glob_all_pipelines discovers pivot.yaml files in subdirectories."""
    sub = set_project_root / "sub"
    sub.mkdir()
    (sub / "pivot.yaml").write_text("stages: {}")

    result = discovery.glob_all_pipelines(set_project_root)

    assert len(result) == 1
    assert result[0] == sub / "pivot.yaml"


def test_glob_all_pipelines_skips_excluded_dirs(set_project_root: pathlib.Path) -> None:
    """glob_all_pipelines skips .venv, __pycache__, .git, etc."""
    venv = set_project_root / ".venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "pipeline.py").write_text("pipeline = None")

    result = discovery.glob_all_pipelines(set_project_root)

    assert len(result) == 0


def test_glob_all_pipelines_deduplicates_by_directory(
    set_project_root: pathlib.Path,
) -> None:
    """glob_all_pipelines raises DiscoveryError if directory has both config types.

    A directory with both pipeline.py and pivot.yaml is ambiguous. This is the
    same constraint as find_config_in_dir but applied during project-wide scan.
    """
    sub = set_project_root / "sub"
    sub.mkdir()
    (sub / "pipeline.py").write_text("pipeline = None")
    (sub / "pivot.yaml").write_text("stages: {}")

    with pytest.raises(discovery.DiscoveryError, match="both"):
        discovery.glob_all_pipelines(set_project_root)


def test_glob_all_pipelines_finds_multiple_pipelines(
    set_project_root: pathlib.Path,
) -> None:
    """glob_all_pipelines finds pipelines in multiple subdirectories."""
    for name in ("alpha", "beta", "gamma"):
        sub = set_project_root / name
        sub.mkdir()
        (sub / "pipeline.py").write_text("pipeline = None")

    result = discovery.glob_all_pipelines(set_project_root)

    assert len(result) == 3


# =============================================================================
# discover_pipeline(all_pipelines=True) Tests
# =============================================================================


def test_discover_all_pipelines_combines_stages(
    set_project_root: pathlib.Path,
) -> None:
    """discover_pipeline(all_pipelines=True) creates a combined pipeline with all stages."""
    from pivot.pipeline.pipeline import Pipeline

    # Create two sub-pipelines with distinct stages
    for name, stage_name in [("alpha", "stage_a"), ("beta", "stage_b")]:
        sub = set_project_root / name
        sub.mkdir()
        code = f"""\
from pivot.pipeline.pipeline import Pipeline

pipeline = Pipeline("{name}", root=__import__("pathlib").Path(__file__).parent)

def _stage():
    pass

pipeline.register(_stage, name="{stage_name}")
"""
        (sub / "pipeline.py").write_text(code)

    result = discovery.discover_pipeline(set_project_root, all_pipelines=True)

    assert result is not None
    assert isinstance(result, Pipeline)
    assert "stage_a" in result.list_stages()
    assert "stage_b" in result.list_stages()


def test_discover_all_pipelines_preserves_state_dir(
    set_project_root: pathlib.Path,
) -> None:
    """Each included stage retains its original pipeline's state_dir."""
    alpha_dir = set_project_root / "alpha"
    alpha_dir.mkdir()
    code_a = """\
from pivot.pipeline.pipeline import Pipeline

pipeline = Pipeline("alpha", root=__import__("pathlib").Path(__file__).parent)

def _stage():
    pass

pipeline.register(_stage, name="stage_a")
"""
    (alpha_dir / "pipeline.py").write_text(code_a)

    result = discovery.discover_pipeline(set_project_root, all_pipelines=True)

    assert result is not None
    info = result.get("stage_a")
    assert info["state_dir"] == alpha_dir / ".pivot"


def test_discover_all_pipelines_name_collision_auto_prefixes(
    set_project_root: pathlib.Path,
) -> None:
    """Name collisions across pipelines are resolved by auto-prefixing."""
    for name in ("alpha", "beta"):
        sub = set_project_root / name
        sub.mkdir()
        code = f"""\
from pivot.pipeline.pipeline import Pipeline

pipeline = Pipeline("{name}", root=__import__("pathlib").Path(__file__).parent)

def _stage():
    pass

pipeline.register(_stage, name="duplicate_name")
"""
        (sub / "pipeline.py").write_text(code)

    result = discovery.discover_pipeline(set_project_root, all_pipelines=True)

    assert result is not None
    stage_names = result.list_stages()
    # One pipeline keeps bare name, the other gets prefixed
    assert "duplicate_name" in stage_names
    prefixed = [n for n in stage_names if "/" in n and n.endswith("/duplicate_name")]
    assert len(prefixed) == 1, f"Expected one prefixed stage, got: {stage_names}"


def test_discover_all_pipelines_returns_none_when_empty(
    set_project_root: pathlib.Path,
) -> None:
    """discover_pipeline(all_pipelines=True) returns None when no pipelines found."""
    result = discovery.discover_pipeline(set_project_root, all_pipelines=True)

    assert result is None


def test_discover_all_pipelines_skips_failed_loads(
    set_project_root: pathlib.Path,
) -> None:
    """discover_pipeline(all_pipelines=True) skips pipelines that fail to load.

    When a sub-pipeline has a broken config (e.g., wrong variable name),
    it should be skipped rather than failing the entire discovery. This
    prevents one bad pipeline from blocking all others.
    """
    from pivot.pipeline.pipeline import Pipeline

    # Create a valid sub-pipeline
    good_dir = set_project_root / "good"
    good_dir.mkdir()
    good_code = """\
from pivot.pipeline.pipeline import Pipeline

pipeline = Pipeline("good", root=__import__("pathlib").Path(__file__).parent)

def _stage():
    pass

pipeline.register(_stage, name="good_stage")
"""
    (good_dir / "pipeline.py").write_text(good_code)

    # Create a broken sub-pipeline (wrong variable name triggers DiscoveryError)
    bad_dir = set_project_root / "bad"
    bad_dir.mkdir()
    bad_code = """\
from pivot.pipeline.pipeline import Pipeline
wrong_name = Pipeline("bad")
"""
    (bad_dir / "pipeline.py").write_text(bad_code)

    result = discovery.discover_pipeline(set_project_root, all_pipelines=True)

    # Should succeed, only including the good pipeline
    assert result is not None
    assert isinstance(result, Pipeline)
    assert "good_stage" in result.list_stages()
    assert len(result.list_stages()) == 1, "Only the good pipeline's stages should be included"


def test_discover_all_pipelines_warns_on_unresolved_deps(
    set_project_root: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """discover_pipeline(all_pipelines=True) warns when deps are not produced by any pipeline.

    When a stage depends on a file not produced by any discovered pipeline,
    a warning is logged. This catches missing cross-pipeline dependencies early.
    """
    # Create a pipeline with a stage that has a dep not produced by any pipeline
    sub_dir = set_project_root / "sub"
    sub_dir.mkdir()
    code = """\
import pathlib
from typing import Annotated, TypedDict
from pivot.pipeline.pipeline import Pipeline
from pivot import loaders, outputs

pipeline = Pipeline("sub", root=__import__("pathlib").Path(__file__).parent)

class _Outputs(TypedDict):
    out: Annotated[pathlib.Path, outputs.Out("result.csv", loaders.PathOnly())]

def _stage(
    external: Annotated[pathlib.Path, outputs.Dep("external_data.csv", loaders.PathOnly())],
) -> _Outputs:
    return {"out": pathlib.Path("result.csv")}

pipeline.register(_stage, name="consumer")
"""
    (sub_dir / "pipeline.py").write_text(code)

    with caplog.at_level(logging.WARNING):
        result = discovery.discover_pipeline(set_project_root, all_pipelines=True)

    assert result is not None
    assert "consumer" in result.list_stages()
    # Should warn about the unresolved dependency
    assert any("not produced by any discovered pipeline" in msg for msg in caplog.messages), (
        f"Expected warning about unresolved deps, got: {caplog.messages}"
    )


def test_glob_all_pipelines_finds_pivot_yml(set_project_root: pathlib.Path) -> None:
    """glob_all_pipelines discovers pivot.yml files in subdirectories."""
    sub = set_project_root / "sub"
    sub.mkdir()
    (sub / "pivot.yml").write_text("stages: {}")

    result = discovery.glob_all_pipelines(set_project_root)

    assert len(result) == 1
    assert result[0] == sub / "pivot.yml"


def test_discover_all_pipelines_returns_none_when_all_fail(
    set_project_root: pathlib.Path,
) -> None:
    """discover_pipeline(all_pipelines=True) returns None when all pipelines fail to load.

    If every discovered config file fails to load (e.g., all have errors),
    the function should return None rather than an empty pipeline.
    """
    # Create two broken sub-pipelines
    for name in ("broken1", "broken2"):
        sub = set_project_root / name
        sub.mkdir()
        (sub / "pipeline.py").write_text("raise RuntimeError('intentional')\n")

    result = discovery.discover_pipeline(set_project_root, all_pipelines=True)

    assert result is None
