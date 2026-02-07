from __future__ import annotations

import logging
import pathlib
import runpy
from typing import TYPE_CHECKING, Final

from pivot import fingerprint, metrics, project
from pivot.pipeline import yaml as pipeline_config

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pivot.pipeline.pipeline import Pipeline

logger = logging.getLogger(__name__)

PIVOT_YAML_NAMES: Final = ("pivot.yaml", "pivot.yml")
PIPELINE_PY_NAME: Final = "pipeline.py"


class DiscoveryError(Exception):
    """Error during pipeline discovery."""


def find_config_in_dir(directory: pathlib.Path) -> pathlib.Path | None:
    """Find the pipeline config file in a directory.

    Returns the path to pivot.yaml/yml or pipeline.py if found.
    Raises DiscoveryError if both exist in the same directory.
    Returns None if neither exists.
    """
    yaml_path: pathlib.Path | None = None
    for yaml_name in PIVOT_YAML_NAMES:
        candidate = directory / yaml_name
        if candidate.is_file():
            yaml_path = candidate
            break

    pipeline_path = directory / PIPELINE_PY_NAME
    pipeline_exists = pipeline_path.is_file()

    if yaml_path and pipeline_exists:
        msg = f"Found both {yaml_path.name} and {PIPELINE_PY_NAME} in {directory}. "
        msg += "Remove one to resolve ambiguity."
        raise DiscoveryError(msg)

    if yaml_path:
        return yaml_path
    if pipeline_exists:
        return pipeline_path
    return None


def discover_pipeline(
    project_root: pathlib.Path | None = None,
    *,
    all_pipelines: bool = False,
) -> Pipeline | None:
    """Discover and return Pipeline from pivot.yaml or pipeline.py.

    Looks for pipeline config in this order:
    1. Current working directory (if within project root)
    2. Project root

    In each location, checks for:
    - pivot.yaml (or pivot.yml) - creates implicit Pipeline
    - pipeline.py - looks for `pipeline` variable (Pipeline instance)

    When all_pipelines=True, discovers ALL pipeline config files in the project,
    loads each, and merges them into a single Pipeline via include(). The combined
    pipeline contains stages from all discovered pipelines, each retaining its
    original state_dir.

    Args:
        project_root: Override project root (default: auto-detect)
        all_pipelines: If True, discover and combine all pipelines in project.

    Returns:
        Pipeline instance, or None if nothing found

    Raises:
        DiscoveryError: If discovery fails, or if both config types exist
    """
    _t = metrics.start()
    try:
        root = project_root or project.get_project_root()

        if all_pipelines:
            return _discover_all_pipelines(root)
        try:
            cwd = pathlib.Path.cwd().resolve()
            root_resolved = root.resolve()
        except OSError as e:
            raise DiscoveryError(f"Failed to resolve paths: {e}") from e

        # Check cwd first if it's within project root but not the root itself
        config_path: pathlib.Path | None = None
        if cwd != root_resolved and cwd.is_relative_to(root_resolved):
            config_path = find_config_in_dir(cwd)

        # Fall back to project root (use resolved path for consistency)
        if config_path is None:
            config_path = find_config_in_dir(root_resolved)

        if config_path is None:
            return None

        logger.info(f"Discovered {config_path}")

        if config_path.name in PIVOT_YAML_NAMES:
            try:
                return pipeline_config.load_pipeline_from_yaml(config_path)
            except pipeline_config.PipelineConfigError as e:
                raise DiscoveryError(f"Failed to load {config_path}: {e}") from e

        # pipeline.py
        _t_module = metrics.start()
        try:
            return _load_pipeline_from_module(config_path)
        except SystemExit as e:
            raise DiscoveryError(f"Pipeline {config_path} called sys.exit({e.code})") from e
        except DiscoveryError:
            # Re-raise DiscoveryError without wrapping
            raise
        except Exception as e:
            raise DiscoveryError(f"Failed to load {config_path}: {e}") from e
        finally:
            metrics.end("discovery.load_module", _t_module)
            fingerprint.flush_ast_hash_cache()
    finally:
        metrics.end("discovery.total", _t)


def _load_pipeline_from_module(path: pathlib.Path) -> Pipeline | None:
    """Load Pipeline instance from a pipeline.py file.

    Returns None if the file doesn't define a 'pipeline' variable.
    Raises DiscoveryError if:
    - 'pipeline' variable exists but isn't a Pipeline instance
    - A Pipeline instance exists under a different variable name (likely typo)
    """
    from pivot.pipeline.pipeline import Pipeline

    module_dict = runpy.run_path(str(path), run_name="_pivot_pipeline")

    # Look for 'pipeline' variable
    pipeline_obj = module_dict.get("pipeline")
    if pipeline_obj is not None:
        if not isinstance(pipeline_obj, Pipeline):
            raise DiscoveryError(
                f"{path} defines 'pipeline' but it's not a Pipeline instance (got {type(pipeline_obj).__name__})"
            )
        return pipeline_obj

    # No 'pipeline' variable - check if there's a Pipeline under a different name
    # This catches cases where user creates a Pipeline but forgets to name it 'pipeline'
    for name, value in module_dict.items():
        if isinstance(value, Pipeline):
            raise DiscoveryError(
                f"{path} does not define a 'pipeline' variable. Found Pipeline instance named '{name}' - rename it to 'pipeline'."
            )

    # No Pipeline found anywhere
    return None


def _discover_all_pipelines(root: pathlib.Path) -> Pipeline | None:
    """Discover all pipelines and combine into one.

    Globs all pipeline config files, loads each, and merges via include().
    """
    from pivot.pipeline.pipeline import Pipeline

    config_paths = glob_all_pipelines(root)
    if not config_paths:
        return None

    pipelines = list[Pipeline]()
    for path in config_paths:
        pipeline = load_pipeline_from_path(path)
        if pipeline is not None:
            pipelines.append(pipeline)
        else:
            logger.warning(f"--all: failed to load pipeline from {path}, skipping")

    if not pipelines:
        return None

    combined = Pipeline("all", root=root)
    for pipeline in pipelines:
        combined.include(pipeline)  # Auto-prefixes on name collision

    # Warn about unresolved external dependencies in --all mode
    local_outputs = set[str]()
    all_deps = set[str]()
    for stage_name in combined.list_stages():
        info = combined.get(stage_name)
        local_outputs.update(info["outs_paths"])
        all_deps.update(info["deps_paths"])
    unresolved = all_deps - local_outputs
    if unresolved:
        sample = ", ".join(sorted(unresolved)[:5])
        suffix = f"... ({len(unresolved)} total)" if len(unresolved) > 5 else ""
        logger.warning(
            f"--all: dependency path(s) not produced by any discovered pipeline: {sample}{suffix}"
        )

    logger.info(
        f"Discovered {len(pipelines)} pipelines with {len(combined.list_stages())} total stages"
    )
    return combined


def find_parent_pipeline_paths(
    start_dir: pathlib.Path,
    stop_at: pathlib.Path,
) -> Iterator[pathlib.Path]:
    """Find pipeline config files in parent directories.

    Traverses up from start_dir (exclusive) to stop_at (inclusive),
    yielding each pivot.yaml or pipeline.py found. Closest parents first.
    Errors if any directory has both.

    Args:
        start_dir: Directory to start from (its config is NOT included).
        stop_at: Stop traversal at this directory (inclusive).

    Yields:
        Paths to pivot.yaml or pipeline.py files.

    Raises:
        DiscoveryError: If a directory has both pivot.yaml and pipeline.py,
            or if path resolution fails.
    """
    try:
        current = start_dir.resolve().parent
        stop_at_resolved = stop_at.resolve()
    except OSError as e:
        raise DiscoveryError(f"Failed to resolve paths: {e}") from e

    while current.is_relative_to(stop_at_resolved):
        config_path = find_config_in_dir(current)
        if config_path:
            yield config_path

        if current == stop_at_resolved or current.parent == current:
            break
        current = current.parent


def find_pipeline_paths_for_dependency(
    dep_path: pathlib.Path,
    stop_at: pathlib.Path,
) -> Iterator[pathlib.Path]:
    """Find pipeline config files starting from a dependency's directory.

    Starts from the directory containing the dependency and traverses up to
    stop_at, yielding each pivot.yaml or pipeline.py found. Closest directories
    first. Unlike find_parent_pipeline_paths (which excludes start_dir), this
    function INCLUDES the dependency's containing directory in the search.

    This enables resolution of sibling pipeline dependencies - if a dependency
    is in ../sibling_b/data/file.csv, we search sibling_b/ for a pipeline that
    produces it.

    Args:
        dep_path: Path to the dependency (file or directory).
        stop_at: Stop traversal at this directory (inclusive).

    Yields:
        Paths to pivot.yaml or pipeline.py files.

    Raises:
        DiscoveryError: If a directory has both pivot.yaml and pipeline.py,
            or if path resolution fails.
    """
    try:
        # Start from dependency's parent directory (the directory containing the dep)
        current = dep_path.resolve().parent
        stop_at_resolved = stop_at.resolve()
    except OSError as e:
        raise DiscoveryError(f"Failed to resolve paths: {e}") from e

    # Traverse up to project root
    while current.is_relative_to(stop_at_resolved):
        config_path = find_config_in_dir(current)
        if config_path:
            yield config_path

        if current == stop_at_resolved or current.parent == current:
            break
        current = current.parent


# Directories excluded from all-pipelines scan
_SCAN_EXCLUDE_DIRS = frozenset(
    {
        ".pivot",
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        ".tox",
        ".nox",
        ".mypy_cache",
        ".ruff_cache",
    }
)


def glob_all_pipelines(project_root: pathlib.Path) -> list[pathlib.Path]:
    """Find all pipeline config files in the project.

    Scans project_root recursively for pipeline.py and pivot.yaml/yml files,
    skipping common non-project directories (.venv, __pycache__, etc.).

    Deduplicates by directory: if a directory contains both pipeline.py and
    pivot.yaml, raises DiscoveryError (same constraint as find_config_in_dir).

    Args:
        project_root: Project root directory to scan.

    Returns:
        List of paths to pipeline config files.

    Raises:
        DiscoveryError: If any directory has both pipeline.py and pivot.yaml.
    """
    # Collect all candidate paths grouped by directory
    by_dir = dict[pathlib.Path, list[pathlib.Path]]()
    target_names = (PIPELINE_PY_NAME, *PIVOT_YAML_NAMES)
    for name in target_names:
        for path in project_root.rglob(name):
            # Only check path components relative to project_root to avoid
            # false exclusions when project_root itself is inside a directory
            # named "venv", "__pycache__", etc.
            try:
                rel_parts = path.relative_to(project_root).parts
            except ValueError:
                continue
            if any(part in _SCAN_EXCLUDE_DIRS for part in rel_parts):
                continue
            by_dir.setdefault(path.parent, list[pathlib.Path]()).append(path)

    # Validate and select canonical config per directory, sorted for deterministic
    # ordering (auto-prefix collision resolution depends on include order).
    results = list[pathlib.Path]()
    for directory in sorted(by_dir):
        paths = by_dir[directory]
        chosen = find_config_in_dir(directory) if len(paths) > 1 else paths[0]
        if chosen is not None:
            results.append(chosen)

    return results


def load_pipeline_from_path(path: pathlib.Path) -> Pipeline | None:
    """Load a Pipeline from a pivot.yaml or pipeline.py file.

    Args:
        path: pathlib.Path to pivot.yaml or pipeline.py file.

    Returns:
        Pipeline instance, or None if file doesn't define one.
        Returns None (with debug log) on load errors.
    """

    # Determine file type and load accordingly
    if path.name in PIVOT_YAML_NAMES:
        try:
            return pipeline_config.load_pipeline_from_yaml(path)
        except Exception as e:
            logger.debug(f"Failed to load pipeline from {path}: {e}")
            return None
    elif path.name == PIPELINE_PY_NAME:
        try:
            return _load_pipeline_from_module(path)
        except DiscoveryError as e:
            # Log at warning level - user likely made a typo (e.g., wrong variable name)
            logger.warning(f"Pipeline discovery issue in {path}: {e}")
            return None
        except Exception as e:
            logger.debug(f"Failed to load pipeline from {path}: {e}")
            return None
        finally:
            fingerprint.flush_ast_hash_cache()
    else:
        logger.debug(f"Unknown pipeline file type: {path}")
        return None
