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

from __future__ import annotations

import dataclasses
import enum
import inspect
import logging
import pathlib
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypedDict, TypeVar

from pivot import exceptions, fingerprint, outputs, project, trie
from pivot.exceptions import ValidationError

if TYPE_CHECKING:
    from inspect import Signature

    from pydantic import BaseModel

F = TypeVar("F", bound=Callable[..., Any])
logger = logging.getLogger(__name__)


class RegistryStageInfo(TypedDict):
    func: Callable[..., Any]
    name: str
    deps: list[str]
    outs: list[outputs.BaseOut]
    outs_paths: list[str]
    params_cls: type[BaseModel] | None
    mutex: list[str]
    signature: Signature | None
    fingerprint: dict[str, str]


class ValidationMode(enum.StrEnum):
    """Validation strictness levels."""

    ERROR = "error"  # Raise exception on validation failure
    WARN = "warn"  # Log warning, allow registration


@dataclasses.dataclass
class stage:
    """Decorator for marking functions as pipeline stages.

    Args:
        deps: Input dependencies (file paths)
        outs: Output files produced by stage (str, Out, Metric, or Plot)
        params_cls: Optional Pydantic model for parameters
        mutex: Mutex groups this stage belongs to (prevents concurrent execution)

    Example:
        >>> @stage(deps=['input.txt'], outs=['output.txt'])
        >>> def process(input_file: str, output_file: str):
        ...     # Process files...
        ...     pass

        >>> @stage(deps=['data.csv'], outs=[Out('model.pkl'), Metric('metrics.json')])
        >>> def train():
        ...     pass

        >>> @stage(deps=['data.csv'], outs=['model.pkl'], mutex=['gpu'])
        >>> def train_gpu():
        ...     # Only one 'gpu' mutex stage runs at a time
        ...     pass
    """

    deps: Sequence[str] = ()
    outs: Sequence[outputs.OutSpec] = ()
    params_cls: type[BaseModel] | None = None
    mutex: Sequence[str] = ()

    def __call__(self, func: F) -> F:
        """Register function as a stage (returns original function unmodified)."""
        REGISTRY.register(
            func,
            name=func.__name__,
            deps=self.deps,
            outs=self.outs,
            params_cls=self.params_cls,
            mutex=self.mutex,
        )
        return func


class StageRegistry:
    """Global registry for all pipeline stages."""

    def __init__(self, validation_mode: ValidationMode = ValidationMode.ERROR) -> None:
        self._stages: dict[str, RegistryStageInfo] = {}
        self.validation_mode: ValidationMode = validation_mode

    def register(
        self,
        func: Callable[..., Any],
        name: str | None = None,
        deps: Sequence[str] | None = None,
        outs: Sequence[outputs.OutSpec] | None = None,
        params_cls: type[BaseModel] | None = None,
        mutex: Sequence[str] | None = None,
    ) -> None:
        """Register a stage function with metadata."""
        stage_name = name if name is not None else func.__name__
        deps_list: Sequence[str] = deps if deps is not None else ()
        outs_list: Sequence[outputs.OutSpec] = outs if outs is not None else ()
        mutex_list: list[str] = [m.strip().lower() for m in mutex] if mutex else []

        # Normalize outputs to BaseOut objects
        outs_normalized = [outputs.normalize_out(o) for o in outs_list]
        outs_paths = [o.path for o in outs_normalized]

        # Validate paths BEFORE normalizing (check ".." on original paths)
        _validate_stage_registration(
            self._stages, stage_name, deps_list, outs_paths, self.validation_mode
        )

        deps_list = _normalize_paths(deps_list, self.validation_mode)
        outs_paths = _normalize_paths(outs_paths, self.validation_mode)

        # Update normalized outputs with absolute paths
        outs_normalized = [
            dataclasses.replace(out, path=path)
            for out, path in zip(outs_normalized, outs_paths, strict=True)
        ]

        try:
            # Build temp_stages with string paths for trie (existing stages use outs_paths)
            temp_stages = {
                name: {"name": name, "outs": info.get("outs_paths", [])}
                for name, info in self._stages.items()
            }
            temp_stages[stage_name] = {
                "name": stage_name,
                "outs": outs_paths,  # Trie uses paths only
            }
            trie.build_outs_trie(temp_stages)
        except (exceptions.OutputDuplicationError, exceptions.OverlappingOutputPathsError) as e:
            _handle_validation_error(str(e), self.validation_mode)

        self._stages[stage_name] = RegistryStageInfo(
            func=func,
            name=stage_name,
            deps=deps_list,
            outs=outs_normalized,
            outs_paths=outs_paths,
            params_cls=params_cls,
            mutex=mutex_list,
            signature=inspect.signature(func),
            fingerprint=fingerprint.get_stage_fingerprint(func),
        )

    def get(self, name: str) -> RegistryStageInfo:
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

    def get_all_output_paths(self) -> set[str]:
        """Get all registered output paths (for watch mode filtering)."""
        result = set[str]()
        for info in self._stages.values():
            for out_path in info["outs_paths"]:
                result.add(str(out_path))
        return result


def _normalize_paths(paths: Sequence[str], validation_mode: ValidationMode) -> list[str]:
    """Normalize paths to absolute paths, handling errors based on validation mode."""
    normalized = list[str]()
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
    stages: dict[str, RegistryStageInfo],
    stage_name: str,
    deps: Sequence[str],
    outs: Sequence[str],
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

    for path in [*deps, *outs]:
        _validate_path(path, stage_name, validation_mode)


def _validate_path(path: str, stage_name: str, validation_mode: ValidationMode) -> None:
    """Validate a single path (before normalization)."""
    if ".." in pathlib.Path(path).parts:
        _handle_validation_error(
            f"Stage '{stage_name}': Path '{path}' contains '..' (path traversal)", validation_mode
        )

    if "\x00" in path:
        _handle_validation_error(
            f"Stage '{stage_name}': Path '{path}' contains null byte", validation_mode
        )

    if "\n" in path or "\r" in path:
        _handle_validation_error(
            f"Stage '{stage_name}': Path '{path}' contains newline character", validation_mode
        )


def _handle_validation_error(msg: str, validation_mode: ValidationMode) -> None:
    """Raise error or warn based on validation mode."""
    if validation_mode == ValidationMode.ERROR:
        raise ValidationError(msg)
    logger.warning(msg)


REGISTRY = StageRegistry()
