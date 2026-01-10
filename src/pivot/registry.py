from __future__ import annotations

import dataclasses
import enum
import inspect
import logging
import pathlib
import re
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypedDict, TypeVar, get_origin, get_type_hints

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
    """Metadata for a registered stage.

    Attributes:
        func: The stage function to execute.
        name: Unique stage identifier (function name or custom name).
        deps: Input file dependencies (absolute paths).
        outs: Output specifications (Out, Metric, Plot, etc.).
        outs_paths: Output file paths (absolute paths).
        params: Pydantic model instance with parameter values.
        mutex: Mutex groups for exclusive execution.
        variant: Variant name for matrix stages (None for regular stages).
        signature: Function signature for parameter injection.
        fingerprint: Code fingerprint mapping (key -> hash).
        cwd: Working directory for path resolution.
    """

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
    cwd: pathlib.Path | None


class ValidationMode(enum.StrEnum):
    """Validation strictness levels."""

    ERROR = "error"  # Raise exception on validation failure
    WARN = "warn"  # Log warning, allow registration


# Stage name pattern: must start with letter, then alphanumeric/underscore/hyphen
_STAGE_NAME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]*$")

# Variant name pattern: alphanumeric, underscore, hyphen; max 64 chars
_VARIANT_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
_VARIANT_NAME_MAX_LENGTH = 64


class Variant(pydantic.BaseModel, frozen=True):
    """Variant specification for matrix stages.

    Attributes:
        name: Unique identifier for this variant (required). Must be alphanumeric
            with underscores/hyphens, max 64 chars.
        deps: Input dependencies (file paths).
        outs: Output files produced by variant.
        params: Optional Pydantic model instance for parameters.
        mutex: Mutex groups this variant belongs to.
        cwd: Working directory for path resolution and stage execution.
    """

    name: str
    deps: Sequence[str] = ()
    outs: Sequence[outputs.OutSpec] = ()
    params: pydantic.BaseModel | None = None
    mutex: Sequence[str] = ()
    cwd: str | pathlib.Path | None = None

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

    The params class is automatically inferred from the function's type hint,
    so you don't need to pass `params=MyParams` - just type-hint your function.

    Args:
        deps: Input dependencies (file paths, relative to cwd or project root)
        outs: Output files produced by stage (str, Out, Metric, or Plot)
        params: Optional params specification. Can be:
            - None (default): params inferred from function signature, instantiated with defaults
            - Pydantic BaseModel class: instantiated with defaults (validated against type hint)
            - Pydantic BaseModel instance: used directly (validated against type hint)
        mutex: Mutex groups this stage belongs to (prevents concurrent execution)
        cwd: Working directory for path resolution and stage execution (default: project root)

    Example:
        >>> class MyParams(BaseModel):
        ...     threshold: float = 0.5

        >>> # Params class inferred from type hint - no params= needed!
        >>> @stage(deps=['data.csv'], outs=['output.csv'])
        >>> def train(params: MyParams):
        ...     print(params.threshold)  # 0.5 (default)

        >>> # Explicit params instance for custom values
        >>> @stage(deps=['data.csv'], params=MyParams(threshold=0.9))
        >>> def train_explicit(params: MyParams):
        ...     print(params.threshold)  # 0.9

        >>> @stage(deps=['data.csv'], outs=['model.pkl'], mutex=['gpu'])
        >>> def train_gpu():
        ...     # Only one 'gpu' mutex stage runs at a time
        ...     pass
    """

    deps: Sequence[str] = ()
    outs: Sequence[outputs.OutSpec] = ()
    params: ParamsArg = None
    mutex: Sequence[str] = ()
    name: str | None = None  # Optional custom name (enables loop-based registration)
    cwd: str | pathlib.Path | None = None  # Working directory for paths and execution

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
            cwd=self.cwd,
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
                    cwd=variant.cwd,
                )
            return func

        return decorator


class StageRegistry:
    """Global registry for all pipeline stages.

    The registry stores metadata for all stages defined via the `@stage` decorator
    or `Pipeline.add_stage()`. It handles validation, path normalization, and
    dependency graph construction.

    The global `REGISTRY` singleton is used by default. Direct instantiation is
    mainly useful for testing with isolated registries.

    Example:
        ```python
        from pivot import REGISTRY
        REGISTRY.list_stages()  # ['preprocess', 'train']
        info = REGISTRY.get('train')
        info['deps']  # List of dependency paths
        ```
    """

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
        cwd: str | pathlib.Path | None = None,
    ) -> None:
        """Register a stage function with metadata.

        Args:
            func: The function to register as a pipeline stage.
            name: Stage name (defaults to function name).
            deps: Input file dependencies (relative or absolute paths).
            outs: Output specifications (strings, Out, Metric, Plot, etc.).
            params: Pydantic model class or instance for parameters.
            mutex: Mutex groups for exclusive execution.
            variant: Variant name for matrix stages.
            cwd: Working directory for path resolution.

        Raises:
            ValidationError: If stage name is invalid or already registered.
            SecurityValidationError: If paths contain traversal components.
            InvalidPathError: If paths resolve outside project root.
            ParamsError: If params specified but function lacks params argument.
        """
        stage_name = name if name is not None else func.__name__
        deps_list: Sequence[str] = deps if deps is not None else ()
        outs_list: Sequence[outputs.OutSpec] = outs if outs is not None else ()
        mutex_list: list[str] = [m.strip().lower() for m in mutex] if mutex else []

        # Normalize cwd to absolute path and validate within project root
        cwd_path: pathlib.Path | None = None
        if cwd is not None:
            cwd_path = project.normalize_path(str(cwd))
            project_root = project.get_project_root()
            if not cwd_path.is_relative_to(project_root):
                raise exceptions.InvalidPathError(
                    f"Stage '{stage_name}': cwd '{cwd}' resolves to '{cwd_path}' "
                    + f"which is outside project root '{project_root}'"
                )

        # Normalize outputs to BaseOut objects
        outs_normalized = [outputs.normalize_out(o) for o in outs_list]
        outs_paths = [o.path for o in outs_normalized]

        # Validate paths BEFORE normalizing (check ".." on original paths)
        _validate_stage_registration(
            self._stages, stage_name, deps_list, outs_paths, self.validation_mode
        )

        # Convert params to instance (instantiate class if needed)
        params_instance = _resolve_params(params, func, stage_name)

        deps_list = _normalize_paths(deps_list, self.validation_mode, cwd_path)
        outs_paths = _normalize_paths(outs_paths, self.validation_mode, cwd_path)

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
            cwd=cwd_path,
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


def _normalize_paths(
    paths: Sequence[str],
    validation_mode: ValidationMode,
    cwd: pathlib.Path | None = None,
) -> list[str]:
    """Normalize paths to absolute paths, preserving symlinks for portability.

    Args:
        paths: Paths to normalize
        validation_mode: How to handle validation errors
        cwd: Base directory for relative paths (default: project root)

    Raises:
        InvalidPathError: If path is outside project root
    """
    normalized = list[str]()
    project_root = project.get_project_root()

    for path in paths:
        try:
            # If cwd is provided and path is relative, resolve from cwd
            path_to_normalize = path
            if cwd is not None and not pathlib.Path(path).is_absolute():
                path_to_normalize = str(cwd / path)

            # Use normalized path (preserve symlinks) for portability
            norm_path = project.normalize_path(path_to_normalize)

            # Reject paths outside project root (not portable)
            if not norm_path.is_relative_to(project_root):
                raise exceptions.InvalidPathError(
                    f"Path '{path}' resolves to '{norm_path}' which is outside "
                    + f"project root '{project_root}'. All paths must be within the project."
                )

            # Warn if relative path contains symlinks
            is_relative = not pathlib.Path(path).is_absolute()
            if is_relative and project.contains_symlink_in_path(norm_path, project_root):
                logger.warning(
                    f"Path '{path}' is inside a symlinked directory. "
                    + "This may affect portability across environments."
                )

            normalized.append(str(norm_path))
        except (ValueError, OSError, exceptions.InvalidPathError):
            if validation_mode == ValidationMode.WARN:
                normalized.append(str(project.normalize_path(path)))
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

    # Extract base name (before @) for validation - matrix variants have format "base@variant"
    base_name = stage_name.split("@")[0] if "@" in stage_name else stage_name
    if not _STAGE_NAME_PATTERN.match(base_name):
        _handle_validation_error(
            f"Stage name '{stage_name}' must start with a letter and contain only "
            + "alphanumeric characters, underscores, or hyphens",
            validation_mode,
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


def _get_params_type_hint(
    func: Callable[..., Any],
    stage_name: str,
    *,
    strict: bool,
) -> type[pydantic.BaseModel] | None:
    """Get the params type hint from function signature.

    Args:
        func: The stage function to inspect
        stage_name: Name of the stage (for error messages)
        strict: If True, raise ParamsError on issues. If False, return None.

    Returns:
        The params class from the type hint, or None if can't resolve (when strict=False)

    Raises:
        ParamsError: When strict=True and type hints can't be resolved or are invalid
    """
    try:
        type_hints = get_type_hints(func)
    except (NameError, TypeError, AttributeError) as e:
        if strict:
            raise ParamsError(
                f"Stage '{stage_name}': failed to resolve type hints for "
                + f"'{func.__name__}': {e}"
            ) from e
        return None

    if "params" not in type_hints:
        if strict:
            raise ParamsError(
                f"Stage '{stage_name}': function '{func.__name__}' has 'params' parameter "
                + "but no type hint. Add a type hint like 'params: MyParams'"
            )
        return None

    hint = type_hints["params"]

    # Reject Union/Optional types - params must be a concrete BaseModel class
    origin = get_origin(hint)
    if origin is not None:
        raise ParamsError(
            f"Stage '{stage_name}': params type hint must be a concrete Pydantic BaseModel "
            + f"class, not a generic or union type. Got: {hint}"
        )

    if not parameters.validate_params_cls(hint):
        if strict:
            raise ParamsError(
                f"Stage '{stage_name}': params type hint must be a Pydantic BaseModel, "
                + f"got {hint}"
            )
        return None

    return hint


def _resolve_params(
    params_arg: ParamsArg,
    func: Callable[..., Any],
    stage_name: str,
) -> pydantic.BaseModel | None:
    """Resolve params argument to an instance, inferring from function signature if needed.

    Resolution order:
    1. If params_arg is an instance, use it directly (validated against type hint)
    2. If params_arg is a class, instantiate with defaults (validated against type hint)
    3. If params_arg is None, infer class from function signature and instantiate with defaults

    The params class is inferred from the type hint of the function's 'params' parameter.
    """
    sig = inspect.signature(func)
    has_params_param = "params" in sig.parameters

    match params_arg:
        # Case 1: params is an instance - use directly (after validation)
        case pydantic.BaseModel():
            if not has_params_param:
                raise ParamsError(
                    f"Stage '{stage_name}': function must have a 'params' parameter "
                    + "when params is specified"
                )
            expected_cls = _get_params_type_hint(func, stage_name, strict=False)
            if expected_cls is not None and not isinstance(params_arg, expected_cls):
                raise ParamsError(
                    f"Stage '{stage_name}': params type {type(params_arg).__name__} "
                    + f"does not match function type hint {expected_cls.__name__}"
                )
            return params_arg

        # Case 2: params is a class - instantiate with defaults
        case type() as params_cls:
            if not has_params_param:
                raise ParamsError(
                    f"Stage '{stage_name}': function must have a 'params' parameter "
                    + "when params is specified"
                )
            if not parameters.validate_params_cls(params_cls):
                raise ParamsError(
                    f"Stage '{stage_name}': params must be a Pydantic BaseModel subclass, "
                    + f"got {params_cls.__name__}"
                )
            expected_cls = _get_params_type_hint(func, stage_name, strict=False)
            if expected_cls is not None and not issubclass(params_cls, expected_cls):
                raise ParamsError(
                    f"Stage '{stage_name}': params type {params_cls.__name__} "
                    + f"does not match function type hint {expected_cls.__name__}"
                )
            try:
                return params_cls()
            except pydantic.ValidationError as e:
                raise ParamsError(
                    f"Stage '{stage_name}': cannot instantiate params with defaults: {e}"
                ) from e

        # Case 3: params is None - infer class if function has params parameter
        case None:
            if not has_params_param:
                return None
            params_cls = _get_params_type_hint(func, stage_name, strict=True)
            assert params_cls is not None  # strict=True guarantees this
            try:
                return params_cls()
            except pydantic.ValidationError as e:
                raise ParamsError(
                    f"Stage '{stage_name}': cannot instantiate params with defaults: {e}"
                ) from e


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
