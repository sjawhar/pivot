"""Stage registry for collecting pipeline stages.

Provides the @stage decorator for marking functions as pipeline stages and
a registry for managing all registered stages.

Example:
    >>> from pivot import stage
    >>>
    >>> @stage(deps=['data.csv'], outs=['output.txt'])
    >>> def process(input_file: str = 'data.csv'):
    ...     with open(input_file) as f:
    ...         data = f.read()
    ...     with open('output.txt', 'w') as f:
    ...         f.write(data.upper())
"""

import dataclasses
import inspect
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from pivot import fingerprint, project, trie
from pivot.exceptions import ValidationError, ValidationMode

__all__ = ["stage", "StageRegistry", "REGISTRY", "ValidationError", "ValidationMode"]

F = TypeVar("F", bound=Callable[..., Any])
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class stage:
    """Decorator for marking functions as pipeline stages.

    Args:
        deps: Input dependencies (file paths)
        outs: Output files produced by stage
        params_cls: Optional Pydantic model for parameters

    Example:
        >>> @stage(deps=['input.txt'], outs=['output.txt'])
        >>> def process(input_file: str, output_file: str):
        ...     # Process files...
        ...     pass
    """

    deps: list[str] = dataclasses.field(default_factory=list)
    outs: list[str] = dataclasses.field(default_factory=list)
    params_cls: type[BaseModel] | None = None

    def __call__(self, func: F) -> F:
        """Register function as a stage (returns original function unmodified)."""
        REGISTRY.register(
            func,
            name=func.__name__,
            deps=self.deps,
            outs=self.outs,
            params_cls=self.params_cls,
        )
        return func


class StageRegistry:
    """Global registry for all pipeline stages."""

    def __init__(self, validation_mode: ValidationMode = ValidationMode.ERROR) -> None:
        self._stages: dict[str, dict[str, Any]] = {}
        self.validation_mode: ValidationMode = validation_mode

    def register(
        self,
        func: Callable[..., Any],
        name: str | None = None,
        deps: list[str] | None = None,
        outs: list[str] | None = None,
        params_cls: type[BaseModel] | None = None,
    ) -> None:
        """Register a stage function with metadata."""
        stage_name = name if name is not None else func.__name__
        deps_list = deps if deps is not None else []
        outs_list = outs if outs is not None else []

        # Validate paths BEFORE normalizing (check ".." on original paths)
        _validate_stage_registration(
            self._stages, stage_name, deps_list, outs_list, self.validation_mode
        )

        deps_list = _normalize_paths(deps_list, self.validation_mode)
        outs_list = _normalize_paths(outs_list, self.validation_mode)

        try:
            temp_stages = dict(self._stages)
            temp_stages[stage_name] = {
                "name": stage_name,
                "outs": outs_list,
            }
            trie.build_outs_trie(temp_stages)
        except (trie.OutputDuplicationError, trie.OverlappingOutputPathsError) as e:
            _handle_validation_error(str(e), self.validation_mode)

        self._stages[stage_name] = {
            "func": func,
            "name": stage_name,
            "deps": deps_list,
            "outs": outs_list,
            "params_cls": params_cls,
            "signature": inspect.signature(func),
            "fingerprint": fingerprint.get_stage_fingerprint(func),
        }

    def get(self, name: str) -> dict[str, Any]:
        """Get stage info by name (raises KeyError if not found)."""
        return self._stages[name]

    def list_stages(self) -> list[str]:
        """Get list of all stage names."""
        return list(self._stages.keys())

    def build_dag(self, validate: bool = True):
        """Build DAG from registered stages.

        Args:
            validate: If True, validate that all dependencies exist

        Returns:
            NetworkX DiGraph with stages as nodes and dependencies as edges

        Raises:
            CyclicGraphError: If graph contains cycles
            DependencyNotFoundError: If dependency doesn't exist (when validate=True)
        """
        from pivot import dag

        return dag.build_dag(self._stages, validate=validate)

    def clear(self) -> None:
        """Clear all registered stages (for testing)."""
        self._stages.clear()


def _normalize_paths(paths: list[str], validation_mode: ValidationMode) -> list[str]:
    """Normalize paths to absolute paths, handling errors based on validation mode."""
    normalized = []
    for path in paths:
        try:
            normalized.append(str(project.resolve_path(path)))
        except (ValueError, OSError):
            if validation_mode == ValidationMode.WARN:
                normalized.append(path)  # Use unnormalized path
            else:
                raise
    return normalized


def _validate_stage_registration(
    stages: dict[str, dict[str, Any]],
    stage_name: str,
    deps: list[str],
    outs: list[str],
    validation_mode: ValidationMode,
) -> None:
    """Validate stage registration inputs (before path normalization)."""
    if stage_name in stages:
        _handle_validation_error(
            f"Stage '{stage_name}' already registered. This will overwrite the existing stage.",
            validation_mode,
        )

    if not stage_name or not stage_name.strip():
        _handle_validation_error("Stage name cannot be empty", validation_mode)

    for path in deps + outs:
        _validate_path(path, stage_name, validation_mode)


def _validate_path(path: str, stage_name: str, validation_mode: ValidationMode) -> None:
    """Validate a single path (before normalization)."""
    if ".." in Path(path).parts:
        _handle_validation_error(
            f"Stage '{stage_name}': Path '{path}' contains '..' (path traversal)", validation_mode
        )

    if "\x00" in path:
        _handle_validation_error(
            f"Stage '{stage_name}': Path '{path}' contains null byte", validation_mode
        )


def _handle_validation_error(msg: str, validation_mode: ValidationMode) -> None:
    """Raise error or warn based on validation mode."""
    if validation_mode == ValidationMode.ERROR:
        raise ValidationError(msg)
    logger.warning(msg)


REGISTRY = StageRegistry()
