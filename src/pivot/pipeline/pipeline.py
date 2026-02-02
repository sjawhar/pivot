from __future__ import annotations

import copy
import inspect
import logging
import pathlib
import re
from typing import TYPE_CHECKING

from pivot import outputs, registry
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

        The stage will use this pipeline's state_dir for lock files and state.db.
        """
        self._registry.register(
            func=func,
            name=name,
            params=params,
            mutex=mutex,
            variant=variant,
            dep_path_overrides=dep_path_overrides,
            out_path_overrides=out_path_overrides,
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
