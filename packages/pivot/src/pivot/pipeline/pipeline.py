# pyright: reportImplicitRelativeImport=false, reportMissingModuleSource=false
from __future__ import annotations

import copy
import inspect
import logging
import pathlib
import re
from typing import TYPE_CHECKING

from pivot import discovery, project, registry, types
from pivot.pipeline.yaml import PipelineConfigError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from networkx import DiGraph

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
) -> tuple[registry.RegistryStageInfo, str] | None:
    """Find the stage in a pipeline that produces dep_path.

    Returns (stage_info, pipeline_name) or None if not found.
    """
    for name in pipeline.list_stages():
        info = pipeline.get(name)
        if dep_path in [types.identity_key(out.identity) for out in info["outs"]]:
            return info, pipeline.name
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
) -> tuple[registry.RegistryStageInfo, str] | None:
    """Tier 1: Walk up from dep's directory looking for a producer.

    Returns (stage_info, pipeline_name) or None.
    """
    pipeline_files = discovery.find_pipeline_paths_for_dependency(
        pathlib.Path(dep_path), project_root
    )
    for pipeline_file in pipeline_files:
        pipeline = _load_pipeline(pipeline_file, loaded_pipelines)
        if pipeline is None:
            continue
        result = _find_producer_in_pipeline(dep_path, pipeline)
        if result is not None:
            return result
    return None


def _find_producer_via_index(
    dep_path: str,
    project_root: pathlib.Path,
    loaded_pipelines: dict[pathlib.Path, Pipeline | None],
) -> tuple[registry.RegistryStageInfo, str] | None:
    """Tier 2: Read cached output index hint, verify it's still valid.

    Returns (stage_info, pipeline_name) or None.
    """
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


def _find_producer_via_scan(
    dep_path: str,
    all_pipeline_paths: list[pathlib.Path],
    loaded_pipelines: dict[pathlib.Path, Pipeline | None],
) -> tuple[registry.RegistryStageInfo, str] | None:
    """Tier 3: Full scan of all pipeline files in the project.

    Returns (stage_info, pipeline_name) or None.
    """
    for pipeline_file in all_pipeline_paths:
        pipeline = _load_pipeline(pipeline_file, loaded_pipelines)
        if pipeline is None:
            continue
        result = _find_producer_in_pipeline(dep_path, pipeline)
        if result is not None:
            return result
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
    _external_deps_resolved: bool

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
        self._external_deps_resolved = False

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

    def list_stages(self) -> list[str]:
        """List all registered stage names."""
        return self._registry.list_stages()

    def get(self, name: str) -> registry.RegistryStageInfo:
        """Get stage info by name."""
        return self._registry.get(name)

    def get_stage(self, name: str) -> registry.RegistryStageInfo:
        """Look up stage metadata by name. Alias for get(), satisfies StageDataProvider."""
        return self.get(name)

    def ensure_fingerprint(self, name: str) -> dict[str, str]:
        """Compute/return cached code fingerprint for a stage."""
        return self._registry.ensure_fingerprint(name)

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

    def _reset_resolution_cache(self) -> None:
        """Reset resolution state and DAG cache after registry changes."""
        self._external_deps_resolved = False
        self._registry.invalidate_dag_cache()

    def invalidate_dag_cache(self) -> None:
        """Invalidate cached DAG and resolution state."""
        self._reset_resolution_cache()

    def restore(self, snapshot: dict[str, registry.RegistryStageInfo]) -> None:
        """Restore registry state from a previous snapshot."""
        self._reset_resolution_cache()
        self._registry.restore(snapshot)

    def clear(self) -> None:
        """Clear all registered stages (for testing)."""
        self._reset_resolution_cache()
        self._registry.clear()

    def include(self, other: Pipeline) -> None:
        """Include all stages from another pipeline.

        Stages are deep-copied with their original state_dir preserved, enabling
        composition where sub-pipeline stages maintain independent state tracking.
        The copy is a point-in-time snapshot; subsequent changes to the source
        pipeline are not reflected.

        On name collision, all stages from the incoming pipeline are automatically
        prefixed with ``{other.name}/`` to disambiguate. The first pipeline's
        stages keep their bare names.

        Args:
            other: Pipeline whose stages to include.

        Raises:
            PipelineConfigError: If ``other`` is ``self`` (self-include).
        """
        if other is self:
            raise PipelineConfigError(f"Pipeline '{self.name}' cannot include itself")

        # Check if any names collide — if so, prefix all incoming stages
        existing_names = set(self._registry.list_stages())
        needs_prefix = any(name in existing_names for name in other.list_stages())

        # Collect stages to add with deep copy
        stages_to_add = list[registry.RegistryStageInfo]()
        for stage_name in other.list_stages():
            stage_info = copy.deepcopy(other.get(stage_name))
            if needs_prefix:
                prefixed = f"{other.name}/{stage_name}"
                # If the prefixed name also collides (e.g. two pipelines with the
                # same name both have a stage with the same name), add a numeric
                # suffix to guarantee uniqueness.
                if prefixed in existing_names:
                    counter = 2
                    while f"{prefixed}_{counter}" in existing_names:
                        counter += 1
                    prefixed = f"{prefixed}_{counter}"
                stage_info["name"] = prefixed
            stages_to_add.append(stage_info)

        # Add all stages (and track names to avoid intra-batch collisions)
        for stage_info in stages_to_add:
            self._registry.add_existing(stage_info)
            existing_names.add(stage_info["name"])

        if stages_to_add:
            self._reset_resolution_cache()
            prefix_note = f" (prefixed with '{other.name}/')" if needs_prefix else ""
            logger.debug(
                f"Included {len(stages_to_add)} stages from pipeline '{other.name}' into '{self.name}'{prefix_note}"
            )

    def resolve_external_dependencies(self) -> None:
        """Resolve unresolved dependencies using three-tier discovery.

        For each dependency that has no local producer, tries:
        1. Traverse-up from the dependency's file path
        2. Output index cache (``.pivot/cache/outputs/``)
        3. Full scan of all pipeline files in the project

        Uses per-call caching (pipelines loaded once per resolve, discarded after).
        Skipped if already resolved (reset by invalidate_dag_cache, include, etc.).
        """
        if self._external_deps_resolved:
            return

        project_root = project.get_project_root()

        # Build set of locally produced outputs and unresolved dependencies in single pass
        local_outputs = set[str]()
        all_deps = set[str]()
        for stage_name in self.list_stages():
            info = self.get(stage_name)
            local_outputs.update(types.identity_key(out.identity) for out in info["outs"])
            all_deps.update(types.identity_key(dep.identity) for dep in info["deps"].values())

        # Work queue is deps not satisfied locally
        work = all_deps - local_outputs

        if not work:
            self._external_deps_resolved = True
            return

        # Per-call caches (fresh every call — safe for watch mode)
        loaded_pipelines = dict[pathlib.Path, Pipeline | None]()
        all_pipeline_paths: list[pathlib.Path] | None = None  # lazy, for tier 3

        # Pre-seed cache with own config path to prevent re-loading self
        own_config = discovery.find_config_in_dir(self._root)
        if own_config is not None:
            loaded_pipelines[own_config] = self

        # Process work queue iteratively
        while work:
            dep_path = work.pop()

            # Skip if already resolved (by a stage we just added)
            if dep_path in local_outputs:
                continue

            # Tier 1: traverse-up (existing behavior)
            result = _find_producer_via_traversal(dep_path, project_root, loaded_pipelines)

            # Tier 2: output index cache
            if result is None:
                result = _find_producer_via_index(dep_path, project_root, loaded_pipelines)

            # Tier 3: full scan
            if result is None:
                if all_pipeline_paths is None:
                    all_pipeline_paths = discovery.glob_all_pipelines(project_root)
                result = _find_producer_via_scan(dep_path, all_pipeline_paths, loaded_pipelines)

            if result is None:
                continue

            producer, source_pipeline_name = result
            producer_name = producer["name"]

            # Skip if already included (idempotency)
            if producer_name in self._registry.list_stages():
                continue

            # Include the producer stage, prefixing on name collision
            stage_info = copy.deepcopy(producer)
            if stage_info["name"] in self._registry.list_stages():
                stage_info["name"] = f"{source_pipeline_name}/{stage_info['name']}"
            self._registry.add_existing(stage_info)
            local_outputs.update(types.identity_key(out.identity) for out in stage_info["outs"])

            # Add producer's unresolved dependencies to work queue
            work.update(
                dep
                for dep in [types.identity_key(dep.identity) for dep in stage_info["deps"].values()]
                if dep not in local_outputs
            )

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
            for out_path in [types.identity_key(out.identity) for out in info["outs"]]:
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

        self._external_deps_resolved = True
