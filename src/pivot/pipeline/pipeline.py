from __future__ import annotations

import copy
import inspect
import logging
import pathlib
import re
from typing import TYPE_CHECKING

from pivot import discovery, outputs, path_policy, path_utils, project, registry, stage_def
from pivot.pipeline.yaml import PipelineConfigError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from networkx import DiGraph

    from pivot.types import StageFunc

# Directories excluded from tier 3 full scan
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

# Pipeline name pattern: alphanumeric, underscore, hyphen (like stage names)
_PIPELINE_NAME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]*$")


def _find_pipeline_dir_for_stage(
    info: registry.RegistryStageInfo,
    project_root: pathlib.Path,
) -> str | None:
    """Find the pipeline directory for a stage, relative to project root.

    Walks up from the stage function's source file to find the nearest directory
    containing a pipeline config (pipeline.py or pivot.yaml). Falls back to
    state_dir derivation if inspect fails.
    """
    # Try to find the pipeline dir from the function's source file
    try:
        source_file = pathlib.Path(inspect.getfile(info["func"])).resolve()
        current = source_file.parent
        project_root_resolved = project_root.resolve()
        while current.is_relative_to(project_root_resolved):
            if discovery.find_config_in_dir(current) is not None:
                return str(current.relative_to(project_root_resolved))
            if current == project_root_resolved:
                break
            current = current.parent
    except (TypeError, OSError, ValueError):
        pass

    # Fall back to state_dir derivation
    state_dir = info["state_dir"]
    if state_dir is not None:
        try:
            return str(state_dir.parent.relative_to(project_root))
        except ValueError:
            pass

    return None


def _find_producer_in_pipeline(
    dep_path: str,
    pipeline: Pipeline,
) -> registry.RegistryStageInfo | None:
    """Find the stage in a pipeline that produces dep_path."""
    for name in pipeline.list_stages():
        info = pipeline.get(name)
        if dep_path in info["outs_paths"]:
            return info
    return None


def _load_pipeline(
    path: pathlib.Path,
    loaded_pipelines: dict[pathlib.Path, Pipeline | None],
) -> Pipeline | None:
    """Load a pipeline from path, using the shared cache."""
    if path not in loaded_pipelines:
        loaded_pipelines[path] = discovery.load_pipeline_from_path(path)
    return loaded_pipelines[path]


def _find_producer_via_traversal(
    dep_path: str,
    project_root: pathlib.Path,
    loaded_pipelines: dict[pathlib.Path, Pipeline | None],
) -> registry.RegistryStageInfo | None:
    """Tier 1: Walk up from dep's directory looking for a producer."""
    pipeline_files = discovery.find_pipeline_paths_for_dependency(
        pathlib.Path(dep_path), project_root
    )
    for pipeline_file in pipeline_files:
        pipeline = _load_pipeline(pipeline_file, loaded_pipelines)
        if pipeline is None:
            continue
        producer = _find_producer_in_pipeline(dep_path, pipeline)
        if producer is not None:
            return producer
    return None


def _find_producer_via_index(
    dep_path: str,
    project_root: pathlib.Path,
    loaded_pipelines: dict[pathlib.Path, Pipeline | None],
) -> registry.RegistryStageInfo | None:
    """Tier 2: Read cached output index hint, verify it's still valid."""
    # Index keys are project-relative; dep_path is absolute
    try:
        rel_dep = str(pathlib.Path(dep_path).relative_to(project_root))
    except ValueError:
        return None
    try:
        pipeline_dir = (project_root / ".pivot" / "cache" / "outputs" / rel_dep).read_text().strip()
    except OSError:
        return None

    # Find the pipeline config in the hinted directory
    hint_path = project_root / pipeline_dir
    config_path = discovery.find_config_in_dir(hint_path)
    if config_path is None:
        return None

    pipeline = _load_pipeline(config_path, loaded_pipelines)
    if pipeline is None:
        return None

    # Verify the pipeline still produces this dep
    return _find_producer_in_pipeline(dep_path, pipeline)


def _glob_all_pipelines(project_root: pathlib.Path) -> list[pathlib.Path]:
    """Glob all pipeline.py and pivot.yaml/yml files in the project."""
    results = list[pathlib.Path]()
    target_names = (discovery.PIPELINE_PY_NAME, *discovery.PIVOT_YAML_NAMES)
    for name in target_names:
        for path in project_root.rglob(name):
            if not any(part in _SCAN_EXCLUDE_DIRS for part in path.parts):
                results.append(path)
    return results


def _find_producer_via_scan(
    dep_path: str,
    all_pipeline_paths: list[pathlib.Path],
    loaded_pipelines: dict[pathlib.Path, Pipeline | None],
) -> registry.RegistryStageInfo | None:
    """Tier 3: Full scan of all pipeline files in the project."""
    for pipeline_file in all_pipeline_paths:
        pipeline = _load_pipeline(pipeline_file, loaded_pipelines)
        if pipeline is None:
            continue
        producer = _find_producer_in_pipeline(dep_path, pipeline)
        if producer is not None:
            return producer
    return None


class Pipeline:
    """A pipeline with its own stage registry and state directory.

    Each pipeline maintains isolated state (lock files, state.db) while
    sharing the project-wide cache.

    Args:
        name: Pipeline identifier for logging and display.
        root: Home directory for this pipeline. Defaults to the directory
            containing the file where Pipeline() is called.
    """

    _name: str
    _root: pathlib.Path
    _registry: registry.StageRegistry

    def __init__(
        self,
        name: str,
        *,
        root: pathlib.Path | None = None,
    ) -> None:
        # Validate pipeline name
        if not name:
            raise PipelineConfigError("Pipeline name cannot be empty")
        if not _PIPELINE_NAME_PATTERN.match(name):
            raise PipelineConfigError(
                f"Invalid pipeline name '{name}'. Must start with a letter and contain only alphanumeric characters, underscores, or hyphens."
            )

        self._name = name
        self._registry = registry.StageRegistry()

        if root is not None:
            self._root = root.resolve()
        else:
            # Infer from caller's __file__
            frame = inspect.currentframe()
            try:
                if frame is None or frame.f_back is None:
                    raise RuntimeError("Cannot determine caller frame")
                caller_file = frame.f_back.f_globals.get("__file__")
                if caller_file is None:
                    raise RuntimeError(
                        "Cannot determine caller's __file__. Provide an explicit root= argument when creating Pipeline from interactive code, exec(), or similar contexts."
                    )
                self._root = pathlib.Path(caller_file).resolve().parent
            finally:
                del frame

    @property
    def name(self) -> str:
        """Pipeline name."""
        return self._name

    @property
    def root(self) -> pathlib.Path:
        """Pipeline root directory."""
        return self._root

    @property
    def state_dir(self) -> pathlib.Path:
        """State directory for this pipeline's lock files and state.db."""
        return self._root / ".pivot"

    def _resolve_path(self, annotation_path: str) -> str:
        """Convert pipeline-relative path to project-relative.

        Validation happens AFTER normalization to allow ../ traversal.
        Paths that escape project root are rejected.
        Trailing slashes are preserved (important for DirectoryOut).
        """

        # Reject empty or whitespace-only paths early
        # (before normalization would turn them into a directory)
        if not annotation_path or not annotation_path.strip():
            raise ValueError("Path cannot be empty or whitespace-only")

        # Reject root-only paths (e.g., "/", "\\", "C:\\", "C:/")
        # These don't refer to a specific file and are almost certainly mistakes
        stripped = annotation_path.strip()
        if stripped in ("/", "\\") or (
            len(stripped) == 3
            and stripped[0].isalpha()
            and stripped[1] == ":"
            and stripped[2] in ("/", "\\")
        ):
            raise ValueError(f"Path cannot be a root directory: {annotation_path!r}")

        project_root = project.get_project_root()

        # Absolute paths: normalize but keep absolute
        # Check for Unix absolute (/), UNC paths (\\), and Windows drive letters (C:\ or C:/)
        is_absolute = (
            annotation_path.startswith("/")
            or annotation_path.startswith("\\")
            or (
                len(annotation_path) >= 3
                and annotation_path[0].isalpha()
                and annotation_path[1] == ":"
                and annotation_path[2] in ("/", "\\")
            )
        )
        if is_absolute:
            abs_path = project.normalize_path(annotation_path)
            resolved = abs_path.as_posix()
        else:
            # Relative paths: resolve from pipeline root -> project-relative
            abs_path = project.normalize_path(annotation_path, base=self.root)
            # Check if path escapes project root (reject paths outside project)
            try:
                abs_path.relative_to(project_root)
            except ValueError as e:
                raise ValueError(
                    f"Path '{annotation_path}' resolves to '{abs_path}' which is outside project root '{project_root}'"
                ) from e
            resolved = project.to_relative_path(abs_path)

        # Restore trailing slash for directory paths (DirectoryOut requires it)
        resolved = path_utils.preserve_trailing_slash(annotation_path, resolved)

        # Validate the RESOLVED path (after ../ is collapsed)
        if error := path_policy.validate_path_syntax(resolved):
            raise ValueError(f"Invalid path '{annotation_path}': {error}")

        return resolved

    def _resolve_path_type(self, path: outputs.PathType) -> outputs.PathType:
        """Resolve a PathType (str, list, or tuple of paths).

        Handles single strings, lists, and tuples of paths.
        """
        if isinstance(path, str):
            return self._resolve_path(path)
        elif isinstance(path, tuple):
            return tuple(self._resolve_path(p) for p in path)
        else:
            # list
            return [self._resolve_path(p) for p in path]

    def _resolve_out_override(self, override: registry.OutOverrideInput) -> registry.OutOverride:
        """Resolve path in an output override, preserving other options."""
        # PathType (str, list, tuple) - just resolve and wrap
        if isinstance(override, (str, list, tuple)):
            return registry.OutOverride(path=self._resolve_path_type(override))

        # OutOverride dict: resolve path, preserve cache option
        result = registry.OutOverride(path=self._resolve_path_type(override["path"]))
        if "cache" in override:
            result["cache"] = override["cache"]
        return result

    def register(
        self,
        func: StageFunc,
        *,
        name: str | None = None,
        params: registry.ParamsArg = None,
        mutex: list[str] | None = None,
        variant: str | None = None,
        dep_path_overrides: Mapping[str, outputs.PathType] | None = None,
        out_path_overrides: Mapping[str, registry.OutOverrideInput] | None = None,
    ) -> None:
        """Register a stage with this pipeline.

        Paths in annotations and overrides are resolved relative to pipeline root.
        """

        stage_name = name or func.__name__

        # 1. Extract annotation paths using existing functions
        # Pass dep_path_overrides to handle PlaceholderDep correctly
        dep_specs = stage_def.get_dep_specs_from_signature(func, dep_path_overrides)

        # Handle both TypedDict returns and single-output returns
        out_specs = stage_def.get_output_specs_from_return(func, stage_name)
        if not out_specs:
            single_out = stage_def.get_single_output_spec_from_return(func)
            if single_out is not None:
                out_specs = {stage_def.SINGLE_OUTPUT_KEY: single_out}

        # 2. Resolve annotation paths relative to pipeline root
        # Skip IncrementalOut - registry disallows path overrides for them
        # (IncrementalOut input/output paths must match in annotations)
        resolved_deps: dict[str, outputs.PathType] = {
            dep_name: self._resolve_path_type(spec.path)
            for dep_name, spec in dep_specs.items()
            if spec.creates_dep_edge  # IncrementalOut has creates_dep_edge=False
        }
        resolved_outs: dict[str, registry.OutOverride] = {
            out_name: registry.OutOverride(path=self._resolve_path_type(spec.path))
            for out_name, spec in out_specs.items()
            if not isinstance(spec, outputs.IncrementalOut)
        }

        # 3. Apply explicit overrides (also pipeline-relative)
        if dep_path_overrides:
            for dep_name, path in dep_path_overrides.items():
                resolved_deps[dep_name] = self._resolve_path_type(path)
        if out_path_overrides:
            for out_name, override in out_path_overrides.items():
                resolved_outs[out_name] = self._resolve_out_override(override)

        # 4. Pass all as overrides to registry
        self._registry.register(
            func=func,
            name=name,
            params=params,
            mutex=mutex,
            variant=variant,
            dep_path_overrides=resolved_deps,
            out_path_overrides=resolved_outs,
            state_dir=self.state_dir,
        )

    def list_stages(self) -> list[str]:
        """List all registered stage names."""
        return self._registry.list_stages()

    def get(self, name: str) -> registry.RegistryStageInfo:
        """Get stage info by name."""
        return self._registry.get(name)

    def build_dag(self, validate: bool = True) -> DiGraph[str]:
        """Build DAG from registered stages.

        Automatically resolves external dependencies before building. For each
        dependency without a local producer, searches for pipelines starting from
        the dependency's directory and traversing up to project root.

        Args:
            validate: If True, validate that all dependencies exist

        Returns:
            NetworkX DiGraph with stages as nodes and dependencies as edges

        Raises:
            CyclicGraphError: If graph contains cycles
            DependencyNotFoundError: If dependency doesn't exist (when validate=True)
        """
        # Auto-resolve external dependencies before building
        self.resolve_external_dependencies()
        dag = self._registry.build_dag(validate=validate)
        self._write_output_index()
        return dag

    def snapshot(self) -> dict[str, registry.RegistryStageInfo]:
        """Create a snapshot of current registry state for backup/restore."""
        return self._registry.snapshot()

    def invalidate_dag_cache(self) -> None:
        """Invalidate cached DAG without clearing stages."""
        self._registry.invalidate_dag_cache()

    def restore(self, snapshot: dict[str, registry.RegistryStageInfo]) -> None:
        """Restore registry state from a previous snapshot."""
        self._registry.restore(snapshot)

    def clear(self) -> None:
        """Clear all registered stages (for testing)."""
        self._registry.clear()

    def include(self, other: Pipeline) -> None:
        """Include all stages from another pipeline.

        Stages are deep-copied with their original state_dir preserved, enabling
        composition where sub-pipeline stages maintain independent state tracking.
        The copy is a point-in-time snapshot; subsequent changes to the source
        pipeline are not reflected.

        Args:
            other: Pipeline whose stages to include.

        Raises:
            PipelineConfigError: If ``other`` is ``self`` (self-include) or if
                any stage name in ``other`` already exists in this pipeline.
        """
        if other is self:
            raise PipelineConfigError(f"Pipeline '{self.name}' cannot include itself")

        # Collect stages to add (validates all before adding any - atomic)
        stages_to_add: list[registry.RegistryStageInfo] = []
        existing_names = set(self._registry.list_stages())

        for stage_name in other.list_stages():
            if stage_name in existing_names:
                raise PipelineConfigError(
                    f"Cannot include pipeline '{other.name}': stage '{stage_name}' already exists in '{self.name}'. Rename the stage using name= at registration time."
                )
            # Deep copy to prevent shared mutable state
            stages_to_add.append(copy.deepcopy(other.get(stage_name)))

        # Add all stages (only reached if validation passes)
        for stage_info in stages_to_add:
            self._registry.add_existing(stage_info)

        if stages_to_add:
            logger.debug(
                f"Included {len(stages_to_add)} stages from pipeline '{other.name}' into '{self.name}'"
            )

    def resolve_external_dependencies(self) -> None:
        """Resolve unresolved dependencies using three-tier discovery.

        For each dependency that has no local producer, tries:
        1. Traverse-up from the dependency's file path
        2. Output index cache (``.pivot/cache/outputs/``)
        3. Full scan of all pipeline files in the project

        Uses per-call caching (pipelines loaded once per resolve, discarded after).
        """
        project_root = project.get_project_root()

        # Build set of locally produced outputs and unresolved dependencies in single pass
        local_outputs = set[str]()
        all_deps = set[str]()
        for stage_name in self.list_stages():
            info = self.get(stage_name)
            local_outputs.update(info["outs_paths"])
            all_deps.update(info["deps_paths"])

        # Work queue is deps not satisfied locally
        work = all_deps - local_outputs

        if not work:
            return

        # Per-call caches (fresh every call — safe for watch mode)
        loaded_pipelines: dict[pathlib.Path, Pipeline | None] = {}
        all_pipeline_paths: list[pathlib.Path] | None = None  # lazy, for tier 3

        # Process work queue iteratively
        while work:
            dep_path = work.pop()

            # Skip if already resolved (by a stage we just added)
            if dep_path in local_outputs:
                continue

            # Tier 1: traverse-up (existing behavior)
            producer = _find_producer_via_traversal(dep_path, project_root, loaded_pipelines)

            # Tier 2: output index cache
            if producer is None:
                producer = _find_producer_via_index(dep_path, project_root, loaded_pipelines)

            # Tier 3: full scan
            if producer is None:
                if all_pipeline_paths is None:
                    all_pipeline_paths = _glob_all_pipelines(project_root)
                producer = _find_producer_via_scan(dep_path, all_pipeline_paths, loaded_pipelines)

            if producer is None:
                continue

            producer_name = producer["name"]

            # Skip if already included (idempotency)
            if producer_name in self._registry.list_stages():
                continue

            # Include the producer stage
            stage_info = copy.deepcopy(producer)
            self._registry.add_existing(stage_info)
            local_outputs.update(stage_info["outs_paths"])

            # Add producer's unresolved dependencies to work queue
            work.update(dep for dep in stage_info["deps_paths"] if dep not in local_outputs)

            logger.debug(f"Included stage '{producer_name}' via three-tier discovery")

    def _write_output_index(self) -> None:
        """Write output index cache for future tier 2 lookups.

        For every stage, writes each output path to
        ``.pivot/cache/outputs/{out_path}`` with the producing pipeline's
        project-relative directory as content. Cache writes are never fatal.

        The pipeline directory is derived from the stage function's source file
        by walking up to find a directory containing a pipeline config file
        (pipeline.py or pivot.yaml). This is more robust than using state_dir,
        which can be shared when sub-pipelines use the same project root.
        """
        try:
            project_root = project.get_project_root()
        except Exception:
            return

        cache_dir = project_root / ".pivot" / "cache" / "outputs"
        for stage_name in self.list_stages():
            info = self.get(stage_name)
            pipeline_dir = _find_pipeline_dir_for_stage(info, project_root)
            if pipeline_dir is None:
                continue
            for out_path in info["outs_paths"]:
                # Convert absolute out_path to project-relative for the index key
                try:
                    rel_path = str(pathlib.Path(out_path).relative_to(project_root))
                except ValueError:
                    continue
                target = cache_dir / rel_path
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(pipeline_dir)
                except OSError:
                    logger.debug(f"Failed to write output index for {out_path}")

    # Keep old name as alias for backwards compatibility
    resolve_from_parents = resolve_external_dependencies  # pyright: ignore[reportUnannotatedClassAttribute] - method alias
