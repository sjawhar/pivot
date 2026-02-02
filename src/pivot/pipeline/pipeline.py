from __future__ import annotations

import copy
import inspect
import logging
import pathlib
import re
from typing import TYPE_CHECKING

from pivot import outputs, path_policy, project, registry, stage_def
from pivot.pipeline.yaml import PipelineConfigError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from networkx import DiGraph

    from pivot.types import StageFunc

# Pipeline name pattern: alphanumeric, underscore, hyphen (like stage names)
_PIPELINE_NAME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]*$")


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
        # Preserve trailing slash for directory paths (DirectoryOut requires it)
        has_trailing_slash = annotation_path.endswith("/")

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

        # Restore trailing slash if original had it
        if has_trailing_slash and not resolved.endswith("/"):
            resolved = resolved + "/"

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
            out_name: self._resolve_out_override(spec.path)
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

        Args:
            validate: If True, validate that all dependencies exist

        Returns:
            NetworkX DiGraph with stages as nodes and dependencies as edges

        Raises:
            CyclicGraphError: If graph contains cycles
            DependencyNotFoundError: If dependency doesn't exist (when validate=True)
        """
        return self._registry.build_dag(validate=validate)

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
