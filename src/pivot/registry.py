from __future__ import annotations

import dataclasses
import enum
import inspect
import logging
import pathlib
import re
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypedDict, TypeVar

import pydantic

from pivot import exceptions, fingerprint, outputs, parameters, project, trie
from pivot.exceptions import ParamsError, SecurityValidationError, ValidationError

if TYPE_CHECKING:
    from inspect import Signature

    from networkx import DiGraph

F = TypeVar("F", bound=Callable[..., Any])
logger = logging.getLogger(__name__)

# Type alias for @stage decorator params argument: accepts class, instance, or None
ParamsArg = type[pydantic.BaseModel] | pydantic.BaseModel | None


class RegistryStageInfo(TypedDict):
    func: Callable[..., Any]
    name: str
    deps: list[str]
    outs: list[outputs.BaseOut]
    outs_paths: list[str]
    params: pydantic.BaseModel | None
    mutex: list[str]
    variant: str | None
    signature: Signature | None
    fingerprint: dict[str, str]


class ValidationMode(enum.StrEnum):
    """Validation strictness levels."""

    ERROR = "error"  # Raise exception on validation failure
    WARN = "warn"  # Log warning, allow registration


# Variant name pattern: alphanumeric, underscore, hyphen; max 64 chars
_VARIANT_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
_VARIANT_NAME_MAX_LENGTH = 64


class Variant(pydantic.BaseModel, frozen=True):
    """Variant specification for matrix stages.

    Args:
        name: Unique identifier for this variant (required). Must be alphanumeric
            with underscores/hyphens, max 64 chars.
        deps: Input dependencies (file paths)
        outs: Output files produced by variant
        params: Optional Pydantic model instance for parameters
        mutex: Mutex groups this variant belongs to
    """

    name: str
    deps: Sequence[str] = ()
    outs: Sequence[outputs.OutSpec] = ()
    params: pydantic.BaseModel | None = None
    mutex: Sequence[str] = ()

    @pydantic.field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate variant name format."""
        if not v:
            raise ValueError("variant name cannot be empty")
        if len(v) > _VARIANT_NAME_MAX_LENGTH:
            raise ValueError(
                f"variant name '{v}' exceeds max length of {_VARIANT_NAME_MAX_LENGTH} chars"
            )
        if not _VARIANT_NAME_PATTERN.match(v):
            raise ValueError(
                f"variant name '{v}' must contain only alphanumeric characters, underscores, and hyphens"
            )
        return v


@dataclasses.dataclass
class stage:
    """Decorator for marking functions as pipeline stages.

    Args:
        deps: Input dependencies (file paths)
        outs: Output files produced by stage (str, Out, Metric, or Plot)
        params: Optional Pydantic model class or instance for parameters
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

        >>> @stage(deps=['data.csv'], params=MyParams(threshold=0.5))
        >>> def train_with_params(params: MyParams):
        ...     # Uses pre-configured params instance
        ...     pass
    """

    deps: Sequence[str] = ()
    outs: Sequence[outputs.OutSpec] = ()
    params: ParamsArg = None
    mutex: Sequence[str] = ()
    name: str | None = None  # Optional custom name (enables loop-based registration)

    def __post_init__(self) -> None:
        """Validate stage name doesn't contain @ (reserved for matrix variants)."""
        if self.name is not None and "@" in self.name:
            raise ValueError(
                f"Stage name '{self.name}' cannot contain '@' (reserved for matrix variants)"
            )

    def __call__(self, func: F) -> F:
        """Register function as a stage (returns original function unmodified)."""
        REGISTRY.register(
            func,
            name=self.name or func.__name__,
            deps=self.deps,
            outs=self.outs,
            params=self.params,
            mutex=self.mutex,
        )
        return func

    @classmethod
    def matrix(cls, variants: Sequence[Variant]) -> Callable[[F], F]:
        """Register multiple variant stages from a list of Variant specs.

        Each variant creates a separate stage with name `func_name@variant_name`.

        Example:
            @stage.matrix([
                Variant(name='current', deps=['data/current.csv'], outs=['out/current.json']),
                Variant(name='legacy', deps=['data/legacy.csv', 'extra.csv'], outs=['out/legacy.json']),
            ])
            def process(variant: str):
                ...
        """
        _validate_matrix_variants(variants)

        def decorator(func: F) -> F:
            for variant in variants:
                stage_name = f"{func.__name__}@{variant.name}"
                REGISTRY.register(
                    func,
                    name=stage_name,
                    deps=variant.deps,
                    outs=variant.outs,
                    params=variant.params,
                    mutex=variant.mutex,
                    variant=variant.name,
                )
            return func

        return decorator


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
        params: ParamsArg = None,
        mutex: Sequence[str] | None = None,
        variant: str | None = None,
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

        # Convert params to instance (instantiate class if needed)
        params_instance = _resolve_params(params, func, stage_name)

        deps_list = _normalize_paths(deps_list, self.validation_mode)
        outs_paths = _normalize_paths(outs_paths, self.validation_mode)

        # Update normalized outputs with absolute paths
        outs_normalized = [
            dataclasses.replace(out, path=path)
            for out, path in zip(outs_normalized, outs_paths, strict=True)
        ]

        try:
            # Build temp_stages with string paths for trie (existing stages use outs_paths)
            temp_stages: dict[str, trie.TrieStageInfo] = {
                name: {"name": name, "outs": info["outs_paths"]}
                for name, info in self._stages.items()
            }
            temp_stages[stage_name] = {"name": stage_name, "outs": outs_paths}
            trie.build_outs_trie(temp_stages)
        except (exceptions.OutputDuplicationError, exceptions.OverlappingOutputPathsError) as e:
            _handle_validation_error(str(e), self.validation_mode)

        self._stages[stage_name] = RegistryStageInfo(
            func=func,
            name=stage_name,
            deps=deps_list,
            outs=outs_normalized,
            outs_paths=outs_paths,
            params=params_instance,
            mutex=mutex_list,
            variant=variant,
            signature=inspect.signature(func),
            fingerprint=fingerprint.get_stage_fingerprint(func),
        )

    def get(self, name: str) -> RegistryStageInfo:
        """Get stage info by name (raises KeyError if not found)."""
        return self._stages[name]

    def list_stages(self) -> list[str]:
        """Get list of all stage names."""
        return list(self._stages.keys())

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
    """Normalize paths to absolute paths, preserving symlinks for portability."""
    normalized = list[str]()

    for path in paths:
        try:
            # Use normalized path (preserve symlinks) for portability
            norm_path = project.normalize_path(path)

            # Warn if relative path contains symlinks (skip for absolute paths to avoid
            # caching project root during registration, which can cause issues in tests)
            if not pathlib.Path(path).is_absolute():
                project_root = project.get_project_root()
                if project.contains_symlink_in_path(norm_path, project_root):
                    logger.warning(
                        f"Path '{path}' is inside a symlinked directory. "
                        + "This may affect portability across environments."
                    )

            normalized.append(str(norm_path))
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

    # Extract base name (before @) for validation - matrix variants have format "base@variant"
    base_name = stage_name.split("@")[0] if "@" in stage_name else stage_name
    if base_name and not base_name[0].isalpha():
        _handle_validation_error(
            f"Stage name '{stage_name}' must start with a letter", validation_mode
        )

    for path in [*deps, *outs]:
        _validate_path(path, stage_name)


def _validate_path(path: str, stage_name: str) -> None:
    """Validate path has no security issues (traversal, null bytes, newlines)."""
    parts = pathlib.Path(path).parts
    security_checks = [
        (".." in parts, "contains '..' (path traversal)"),
        ("\x00" in path, "contains null byte"),
        ("\n" in path or "\r" in path, "contains newline character"),
    ]
    for failed, description in security_checks:
        if failed:
            raise SecurityValidationError(f"Stage '{stage_name}': Path '{path}' {description}")


def _handle_validation_error(msg: str, validation_mode: ValidationMode) -> None:
    """Raise error or warn based on validation mode."""
    if validation_mode == ValidationMode.ERROR:
        raise ValidationError(msg)
    logger.warning(msg)


def _resolve_params(
    params_arg: ParamsArg,
    func: Callable[..., Any],
    stage_name: str,
) -> pydantic.BaseModel | None:
    """Resolve params argument to an instance, validating along the way.

    If params is a class, instantiate it with defaults.
    If params is an instance, use it directly.
    If params is None, return None.
    """
    if params_arg is None:
        _warn_orphaned_params(func, stage_name)
        return None

    sig = inspect.signature(func)
    if "params" not in sig.parameters:
        raise ParamsError(
            f"Stage '{stage_name}': function must have a 'params' parameter when params is specified"
        )

    # Check if it's a class (type) or instance
    if isinstance(params_arg, type):
        # It's a class - validate and instantiate
        if not parameters.validate_params_cls(params_arg):
            raise ParamsError(
                f"Stage '{stage_name}': params must be a Pydantic BaseModel subclass, got {params_arg.__name__}"
            )
        return params_arg()
    else:
        # It's an instance - validate the type
        if not parameters.validate_params_cls(type(params_arg)):
            raise ParamsError(
                f"Stage '{stage_name}': params must be a Pydantic BaseModel instance, got {type(params_arg).__name__}"
            )
        return params_arg


def _warn_orphaned_params(func: Callable[..., Any], stage_name: str) -> None:
    """Warn if function has 'params' parameter but no params provided."""
    sig = inspect.signature(func)
    if "params" in sig.parameters:
        logger.warning(
            f"Stage '{stage_name}': function has 'params' parameter but no params specified"
        )


def _validate_matrix_variants(variants: Sequence[Variant]) -> None:
    """Validate matrix variant list for duplicates and emptiness."""
    if not variants:
        raise ValidationError("matrix variants cannot be empty - provide at least one Variant")

    seen_names = set[str]()
    for variant in variants:
        if variant.name in seen_names:
            raise ValidationError(f"Duplicate variant name '{variant.name}' in matrix")
        seen_names.add(variant.name)


REGISTRY = StageRegistry()
