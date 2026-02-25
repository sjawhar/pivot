# pyright: reportImplicitRelativeImport=false
from __future__ import annotations

import logging
import pathlib
import runpy
from typing import Final, cast

from pivot import fingerprint, metrics, project, types
from pivot.registry import PipelineLike

logger = logging.getLogger(__name__)

PIPELINE_PY_NAME: Final = "pipeline.py"


class DiscoveryError(Exception):
    """Error during pipeline discovery."""


def find_config_in_dir(directory: pathlib.Path) -> pathlib.Path | None:
    """Find the pipeline config file in a directory.

    Returns the path to pipeline.py if found.
    Returns None if not found.
    """
    pipeline_path = directory / PIPELINE_PY_NAME
    if pipeline_path.is_file():
        return pipeline_path
    return None


def discover_pipeline(
    project_root: pathlib.Path | None = None,
    *,
    all_pipelines: bool = False,
) -> PipelineLike | None:
    """Discover and return Pipeline from pipeline.py.

    Looks for pipeline config in this order:
    1. Current working directory (if within project root)
    2. Project root

    In each location, checks for:
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
            fingerprint.flush_manifest_cache()
    finally:
        metrics.end("discovery.total", _t)


def _load_pipeline_from_module(path: pathlib.Path) -> PipelineLike | None:
    """Load Pipeline instance from a pipeline.py file.

    Returns None if the file doesn't define a 'pipeline' variable.
    Raises DiscoveryError if:
    - 'pipeline' variable exists but isn't a Pipeline instance
    - A Pipeline instance exists under a different variable name (likely typo)
    """
    module_dict = cast("dict[str, object]", runpy.run_path(str(path), run_name="_pivot_pipeline"))

    # Look for 'pipeline' variable
    pipeline_obj = module_dict.get("pipeline")
    if pipeline_obj is not None:
        if not isinstance(pipeline_obj, PipelineLike):
            from typing import get_protocol_members

            required = sorted(get_protocol_members(PipelineLike))
            missing = [attr for attr in required if not hasattr(pipeline_obj, attr)]
            if missing:
                raise DiscoveryError(
                    f"{path} defines 'pipeline' but it's missing required methods: {missing}"
                )
            raise DiscoveryError(
                f"{path} defines 'pipeline' but it doesn't satisfy the Pipeline interface "
                f"(got {type(pipeline_obj).__name__})"
            )
        return pipeline_obj

    # No 'pipeline' variable - check if there's a Pipeline under a different name
    # This catches cases where user creates a Pipeline but forgets to name it 'pipeline'
    for name, value in module_dict.items():
        if isinstance(value, PipelineLike):
            raise DiscoveryError(
                f"{path} does not define a 'pipeline' variable. Found Pipeline instance named '{name}' - rename it to 'pipeline'."
            )

    # No Pipeline found anywhere
    return None


def _discover_all_pipelines(root: pathlib.Path) -> PipelineLike | None:
    """Discover all pipelines and combine into one.

    Globs all pipeline config files, loads each, and merges via include().
    """
    from pivot.compose import Pipeline

    config_paths = glob_all_pipelines(root)
    if not config_paths:
        return None

    pipelines = list[PipelineLike]()
    for path in config_paths:
        pipeline = load_pipeline_from_path(path)
        if pipeline is not None:
            pipelines.append(pipeline)
            logger.info(f"  {pipeline.name}: {path} ({len(pipeline.list_stages())} stages)")
        else:
            logger.warning(f"--all: failed to load pipeline from {path}, skipping")

    if not pipelines:
        return None

    combined = Pipeline("all", root=root)
    for pipeline in pipelines:
        combined.include(pipeline)  # Auto-prefixes on name collision

    if logger.isEnabledFor(logging.DEBUG):
        local_outputs = set[str]()
        all_deps = set[str]()
        for stage_name in combined.list_stages():
            info = combined.get_stage(stage_name)
            local_outputs.update(types.identity_key(out.identity) for out in info["outs"])
            all_deps.update(types.identity_key(dep.identity) for dep in info["deps"].values())
        unresolved = all_deps - local_outputs
        if unresolved:
            sample = ", ".join(sorted(unresolved)[:5])
            suffix = f"... ({len(unresolved)} total)" if len(unresolved) > 5 else ""
            logger.debug(
                f"--all: dependency path(s) not produced by any discovered pipeline: {sample}{suffix}"
            )

    logger.debug(
        f"Discovered {len(pipelines)} pipelines with {len(combined.list_stages())} total stages"
    )
    return combined


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
        "site-packages",
        "dist-packages",
        ".eggs",
    }
)


def glob_all_pipelines(project_root: pathlib.Path) -> list[pathlib.Path]:
    """Find all pipeline config files in the project.

    Scans project_root recursively for pipeline.py files,
    skipping common non-project directories (.venv, __pycache__, etc.).

    Args:
        project_root: Project root directory to scan.

    Returns:
        List of paths to pipeline config files.
    """
    # Collect all candidate paths grouped by directory
    by_dir = dict[pathlib.Path, list[pathlib.Path]]()
    for path in project_root.rglob(PIPELINE_PY_NAME):
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
        results.append(by_dir[directory][0])

    return results


def load_pipeline_from_path(path: pathlib.Path) -> PipelineLike | None:
    """Load a Pipeline from a pipeline.py file.

    Args:
        path: pathlib.Path to pipeline.py file.

    Returns:
        Pipeline instance, or None if file doesn't define one.
        Returns None (with debug log) on load errors.
    """

    if path.name != PIPELINE_PY_NAME:
        logger.debug(f"Unknown pipeline file type: {path}")
        return None

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
        fingerprint.flush_manifest_cache()
