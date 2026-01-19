from __future__ import annotations

import dataclasses
import enum
import inspect
import logging
import pathlib
import re
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

import pydantic

from pivot import (
    exceptions,
    fingerprint,
    outputs,
    path_policy,
    project,
    stage_def,
    trie,
)

if TYPE_CHECKING:
    from inspect import Signature

    from networkx import DiGraph
logger = logging.getLogger(__name__)

# Type alias for params argument: accepts class, instance, or None
ParamsArg = type[pydantic.BaseModel] | pydantic.BaseModel | None


class _OutOverrideOptions(TypedDict, total=False):
    """Optional options for output overrides."""

    cache: bool
    persist: bool


class OutOverride(_OutOverrideOptions):
    """Override options for an annotation-defined output.

    path is required, other options are optional and override annotation defaults.
    """

    path: outputs.PathType


class RegistryStageInfo(TypedDict):
    """Metadata for a registered stage.

    Attributes:
        func: The stage function to execute.
        name: Unique stage identifier (function name or custom name).
        deps: Named input file dependencies (name -> path(s), absolute paths).
        deps_paths: Flattened list of all dependency paths (for DAG/worker).
        outs: Output specifications (Out, Metric, Plot, etc.).
        outs_paths: Output file paths (absolute paths).
        params: Pydantic model instance with parameter values.
        mutex: Mutex groups for exclusive execution.
        variant: Variant name for matrix stages (None for regular stages).
        signature: Function signature for parameter injection.
        fingerprint: Code fingerprint mapping (key -> hash).
        cwd: Working directory for path resolution.
        dep_specs: Dependency specs from function annotations.
        out_path_overrides: Path and option overrides for outputs from YAML.
    """

    func: Callable[..., Any]
    name: str
    # deps: Named dependencies for injection (name -> path mapping)
    # deps_paths: Flat list for DAG construction and fingerprint hashing
    deps: dict[str, outputs.PathType]
    deps_paths: list[str]
    outs: list[outputs.Out[Any]]
    outs_paths: list[str]
    params: stage_def.StageParams | None
    mutex: list[str]
    variant: str | None
    signature: Signature | None
    fingerprint: dict[str, str]
    cwd: pathlib.Path | None
    dep_specs: dict[str, stage_def.FuncDepSpec]
    out_path_overrides: dict[str, OutOverride] | None


class ValidationMode(enum.StrEnum):
    """Validation strictness levels."""

    ERROR = "error"  # Raise exception on validation failure
    WARN = "warn"  # Log warning, allow registration


# Stage name pattern: must start with letter, then alphanumeric/underscore/hyphen
_STAGE_NAME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]*$")


def _apply_out_overrides(
    out_spec: outputs.Out[Any],
    override: OutOverride | None,
) -> list[outputs.Out[Any]]:
    """Apply path and option overrides to an Out spec, preserving subclass type and loader.

    For multi-file outputs (path is list/tuple), creates individual Out objects for each path.
    Each returned Out object has a single-string path.

    Returns a list of Out objects (possibly multiple for multi-file outputs).
    """
    # Determine final path (override or annotation default)
    path = override["path"] if override else out_spec.path

    # Determine final cache/persist (override takes precedence, then annotation default)
    # Note: annotation default is already set correctly for Out/Metric/Plot subclasses
    cache = override.get("cache", out_spec.cache) if override else out_spec.cache
    persist = override.get("persist", out_spec.persist) if override else out_spec.persist

    # For multi-file outputs, create individual Out objects for each path
    if isinstance(path, (list, tuple)):
        result = list[outputs.Out[Any]]()
        for p in path:
            # Use dataclasses.replace to preserve the Out subclass type and loader
            result.append(dataclasses.replace(out_spec, path=p, cache=cache, persist=persist))
        return result
    else:
        return [dataclasses.replace(out_spec, path=path, cache=cache, persist=persist)]


class StageRegistry:
    """Global registry for all pipeline stages.

    The registry stores metadata for all stages registered via `REGISTRY.register()`.
    It handles validation, path normalization, and dependency graph construction.

    Stages are registered from pivot.yaml or programmatically. Dependencies and outputs
    are extracted from function annotations (Annotated[T, Dep(...)] and TypedDict
    return types with Out annotations).

    The global `REGISTRY` singleton is used by default. Direct instantiation is
    mainly useful for testing with isolated registries.

    Example:
        ```python
        from pivot.registry import REGISTRY
        REGISTRY.list_stages()  # ['preprocess', 'train']
        info = REGISTRY.get('train')
        info['deps']  # Dict of dependency name -> path(s)
        ```
    """

    def __init__(self, validation_mode: ValidationMode = ValidationMode.ERROR) -> None:
        self._stages: dict[str, RegistryStageInfo] = {}
        self.validation_mode: ValidationMode = validation_mode

    def register(
        self,
        func: Callable[..., Any],
        name: str | None = None,
        params: ParamsArg = None,
        mutex: Sequence[str] | None = None,
        variant: str | None = None,
        cwd: str | pathlib.Path | None = None,
        dep_path_overrides: Mapping[str, outputs.PathType] | None = None,
        out_path_overrides: Mapping[str, OutOverride] | None = None,
    ) -> None:
        """Register a stage function with metadata.

        Dependencies and outputs are extracted from function annotations:
        - Deps: function parameters with `Annotated[T, Dep("path", loader)]`
        - Outs: TypedDict return type with `Annotated[T, Out("path", loader)]` fields

        Args:
            func: The function to register as a pipeline stage.
            name: Stage name (defaults to function name).
            params: Pydantic model class or instance for parameters.
            mutex: Mutex groups for exclusive execution.
            variant: Variant name for matrix stages.
            cwd: Working directory for path resolution.
            dep_path_overrides: Override paths for deps (must match annotation dep names).
            out_path_overrides: Override paths and options for outputs (must match annotation names).

        Raises:
            ValidationError: If stage name is invalid or already registered.
            SecurityValidationError: If paths contain traversal components.
            InvalidPathError: If paths resolve outside project root.
            ParamsError: If params specified but function lacks params argument.
        """
        stage_name = name if name is not None else func.__name__
        mutex_list: list[str] = [m.strip().lower() for m in mutex] if mutex else []

        # Normalize cwd to absolute path and validate within project root
        cwd_path: pathlib.Path | None = None
        if cwd is not None:
            cwd_path = project.normalize_path(cwd)
            project_root = project.get_project_root()
            if not cwd_path.is_relative_to(project_root):
                raise exceptions.InvalidPathError(
                    f"Stage '{stage_name}': cwd '{cwd}' resolves to '{cwd_path}' "
                    + f"which is outside project root '{project_root}'"
                )

        # Convert params to instance (instantiate class if needed)
        params_instance = _resolve_params(params, func, stage_name)

        # Extract deps from function annotations
        dep_specs = stage_def.get_dep_specs_from_signature(func)

        # Validate dep_path_overrides match annotation dep names
        if dep_path_overrides:
            unknown = set(dep_path_overrides.keys()) - set(dep_specs.keys())
            if unknown:
                raise exceptions.ValidationError(
                    f"Stage '{stage_name}': dep_path_overrides contains unknown deps: {unknown}. Available: {list(dep_specs.keys())}"
                )
            # Apply overrides
            dep_specs = stage_def.apply_dep_path_overrides(dep_specs, dep_path_overrides)

        # Build deps dict from specs
        deps_dict: dict[str, outputs.PathType] = {
            dep_name: spec.path for dep_name, spec in dep_specs.items()
        }

        # Flatten deps for validation and DAG
        deps_flat = _flatten_deps(deps_dict)

        # Extract outs from return type annotations
        return_out_specs = stage_def.get_output_specs_from_return(func)
        single_out_spec = stage_def.get_single_output_spec_from_return(func)

        # Build outs from return annotations (preserving Out subclass type and loader)
        outs_from_annotations: list[outputs.Out[Any]] = []
        out_overrides_dict: dict[str, OutOverride] | None = None

        if return_out_specs:
            # Validate out_path_overrides match annotation out names
            if out_path_overrides:
                unknown = set(out_path_overrides.keys()) - set(return_out_specs.keys())
                if unknown:
                    raise exceptions.ValidationError(
                        f"Stage '{stage_name}': out_path_overrides contains unknown outs: {unknown}. Available: {list(return_out_specs.keys())}"
                    )
                out_overrides_dict = dict(out_path_overrides)

            # Apply overrides while preserving Out subclass type and loader
            for out_name, out_spec in return_out_specs.items():
                override = out_path_overrides.get(out_name) if out_path_overrides else None
                outs_from_annotations.extend(_apply_out_overrides(out_spec, override))

        elif single_out_spec is not None:
            # Single annotated return type - any YAML key is valid (maps to the one output)
            override: OutOverride | None = None
            if out_path_overrides:
                if len(out_path_overrides) > 1:
                    raise exceptions.ValidationError(
                        f"Stage '{stage_name}': single-output stage has {len(out_path_overrides)} out_path_overrides keys ({list(out_path_overrides.keys())}). Only one key is allowed for single-output stages."
                    )
                # Get the single override (whatever key the user used)
                override = next(iter(out_path_overrides.values()))
                # Map to internal _single key
                out_overrides_dict = {"_single": override}

            outs_from_annotations.extend(_apply_out_overrides(single_out_spec, override))

        outs_list = outs_from_annotations
        # After _apply_out_overrides, each Out has a single-string path (multi-file paths expanded)
        outs_paths = [str(o.path) for o in outs_list]

        # Validate paths BEFORE normalizing (check ".." on original paths)
        _validate_stage_registration(
            self._stages, stage_name, deps_flat, outs_paths, self.validation_mode
        )

        # Normalize dep paths - flatten, normalize, then rebuild dict
        deps_flat_normalized = _normalize_paths(
            deps_flat, path_policy.PathType.DEP, self.validation_mode, cwd_path
        )
        outs_paths = _normalize_paths(
            outs_paths, path_policy.PathType.OUT, self.validation_mode, cwd_path
        )

        # Rebuild deps dict with normalized paths
        deps_normalized = _normalize_deps_dict(deps_dict, cwd_path, self.validation_mode)

        # Update normalized outputs with absolute paths
        outs_normalized = [
            dataclasses.replace(out, path=path)
            for out, path in zip(outs_list, outs_paths, strict=True)
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

        # Build stage fingerprint (includes loader fingerprints from annotations)
        stage_fp = fingerprint.get_stage_fingerprint(func)
        stage_fp.update(
            _get_annotation_loader_fingerprints(dep_specs, return_out_specs, single_out_spec)
        )

        self._stages[stage_name] = RegistryStageInfo(
            func=func,
            name=stage_name,
            deps=deps_normalized,
            deps_paths=deps_flat_normalized,
            outs=outs_normalized,
            outs_paths=outs_paths,
            params=params_instance,
            mutex=mutex_list,
            variant=variant,
            signature=inspect.signature(func),
            fingerprint=stage_fp,
            cwd=cwd_path,
            dep_specs=dep_specs,
            out_path_overrides=out_overrides_dict,
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

    def snapshot(self) -> dict[str, RegistryStageInfo]:
        """Create a snapshot of current registry state for backup/restore.

        Returns a shallow copy of the internal stages dict. Use with `restore()`
        to implement atomic reload patterns where you want to preserve the previous
        valid state if the reload fails.

        Example:
            backup = REGISTRY.snapshot()
            REGISTRY.clear()
            try:
                reload_stages()
            except Exception:
                REGISTRY.restore(backup)  # Rollback on failure
        """
        return dict(self._stages)

    def restore(self, snapshot: dict[str, RegistryStageInfo]) -> None:
        """Restore registry state from a previous snapshot.

        Replaces all current stages with the snapshot contents. Typically used
        to rollback after a failed reload operation.

        Args:
            snapshot: Previously captured state from `snapshot()`
        """
        self._stages = dict(snapshot)

    def get_all_output_paths(self) -> set[str]:
        """Get all registered output paths (for watch mode filtering)."""
        result = set[str]()
        for info in self._stages.values():
            for out_path in info["outs_paths"]:
                result.add(str(out_path))
        return result


def _normalize_paths(
    paths: Sequence[str],
    path_type: path_policy.PathType,
    validation_mode: ValidationMode,
    cwd: pathlib.Path | None = None,
) -> list[str]:
    """Normalize paths to absolute paths, applying policy-based validation.

    Args:
        paths: Paths to normalize
        path_type: Type of path (DEP or OUT) for policy lookup
        validation_mode: How to handle validation errors
        cwd: Base directory for relative paths (default: project root)

    Raises:
        InvalidPathError: If path violates its type's policy
    """
    normalized = list[str]()
    project_root = project.get_project_root()
    policy = path_policy.POLICIES[path_type]

    for path in paths:
        try:
            # Normalize path to absolute (from cwd or project root)
            if pathlib.Path(path).is_absolute():
                norm_path = pathlib.Path(path)
            elif cwd is not None:
                norm_path = project.normalize_path(cwd / path)
            else:
                norm_path = project.normalize_path(path)

            # Check if path is within project root
            is_within_project = norm_path.is_relative_to(project_root)

            if not is_within_project:
                # Path is outside project root
                if not policy["allow_absolute"]:
                    raise exceptions.InvalidPathError(
                        f"{path_type.value.capitalize()} path '{path}' resolves to '{norm_path}' "
                        + f"which is outside project root '{project_root}'"
                    )
                # Allowed (deps only) - warn about reproducibility
                logger.warning(f"Absolute {path_type.value} path may break reproducibility: {path}")
            else:
                # Path is within project - check symlink escape (for paths that exist)
                if norm_path.exists() and project.contains_symlink_in_path(norm_path, project_root):
                    resolved = norm_path.resolve()
                    if not resolved.is_relative_to(project_root.resolve()):
                        msg = (
                            f"{path_type.value.capitalize()} path '{path}' resolves outside "
                            + f"project via symlink: {resolved}"
                        )
                        if policy["symlink_escape_action"] == "error":
                            raise exceptions.InvalidPathError(msg)
                        logger.warning(msg)
                    else:
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

    # Validate syntax only here (containment checked in _normalize_paths)
    for path in deps:
        error = path_policy.validate_path_syntax(path)
        if error:
            raise exceptions.SecurityValidationError(
                f"Stage '{stage_name}': dependency path {error}: {path}"
            )

    for path in outs:
        error = path_policy.validate_path_syntax(path)
        if error:
            raise exceptions.SecurityValidationError(
                f"Stage '{stage_name}': output path {error}: {path}"
            )


def _handle_validation_error(msg: str, validation_mode: ValidationMode) -> None:
    """Raise error or warn based on validation mode."""
    if validation_mode == ValidationMode.ERROR:
        raise exceptions.ValidationError(msg)
    logger.warning(msg)


def _resolve_params(
    params_arg: ParamsArg,
    func: Callable[..., Any],
    stage_name: str,
) -> stage_def.StageParams | None:
    """Resolve params argument to an instance, inferring from function signature if needed.

    Resolution order:
    1. If params_arg is an instance, use it directly (validated against type hint)
    2. If params_arg is a class, instantiate with defaults (validated against type hint)
    3. If params_arg is None, infer class from function signature and instantiate with defaults

    The params class must be a StageParams subclass (not plain pydantic.BaseModel).
    """
    # Find params in signature using StageParams detection
    params_arg_name, params_type_hint = stage_def.find_params_in_signature(func)
    has_params_param = params_arg_name is not None

    match params_arg:
        # Case 1: params is an instance - use directly (after validation)
        case stage_def.StageParams():
            if not has_params_param:
                raise exceptions.ParamsError(
                    f"Stage '{stage_name}': function must have a StageParams parameter "
                    + "when params is specified"
                )
            if params_type_hint is not None and not isinstance(params_arg, params_type_hint):
                raise exceptions.ParamsError(
                    f"Stage '{stage_name}': params type {type(params_arg).__name__} "
                    + f"does not match function type hint {params_type_hint.__name__}"
                )
            return params_arg

        # Case 1b: plain BaseModel (not StageParams) - error
        case pydantic.BaseModel():
            raise exceptions.ParamsError(
                f"Stage '{stage_name}': params must be a StageParams subclass, "
                + f"got {type(params_arg).__name__} (plain pydantic.BaseModel). "
                + "Inherit from pivot.stage_def.StageParams instead of pydantic.BaseModel."
            )

        # Case 2: params is a class - instantiate with defaults
        case type() as params_cls:
            if not has_params_param:
                raise exceptions.ParamsError(
                    f"Stage '{stage_name}': function must have a StageParams parameter "
                    + "when params is specified"
                )
            if not issubclass(params_cls, stage_def.StageParams):
                raise exceptions.ParamsError(
                    f"Stage '{stage_name}': params must be a StageParams subclass, "
                    + f"got {params_cls.__name__}. "
                    + "Inherit from pivot.stage_def.StageParams instead of pydantic.BaseModel."
                )
            if params_type_hint is not None and not issubclass(params_cls, params_type_hint):
                raise exceptions.ParamsError(
                    f"Stage '{stage_name}': params type {params_cls.__name__} "
                    + f"does not match function type hint {params_type_hint.__name__}"
                )
            try:
                return params_cls()
            except pydantic.ValidationError as e:
                raise exceptions.ParamsError(
                    f"Stage '{stage_name}': cannot instantiate params with defaults: {e}"
                ) from e

        # Case 3: params is None - infer class if function has params parameter
        case None:
            if not has_params_param:
                return None
            assert params_type_hint is not None  # has_params_param guarantees this
            try:
                return params_type_hint()
            except pydantic.ValidationError as e:
                raise exceptions.ParamsError(
                    f"Stage '{stage_name}': cannot instantiate params with defaults: {e}"
                ) from e


def _get_annotation_loader_fingerprints(
    dep_specs: dict[str, stage_def.FuncDepSpec],
    return_out_specs: dict[str, outputs.Out[Any]],
    single_out_spec: outputs.Out[Any] | None,
) -> dict[str, str]:
    """Get fingerprints for all loaders from annotations."""
    result = dict[str, str]()

    for spec in dep_specs.values():
        result.update(fingerprint.get_loader_fingerprint(spec.loader))

    for out in return_out_specs.values():
        result.update(fingerprint.get_loader_fingerprint(out.loader))

    if single_out_spec is not None:
        result.update(fingerprint.get_loader_fingerprint(single_out_spec.loader))

    return result


def _flatten_deps(deps: dict[str, outputs.PathType]) -> list[str]:
    """Flatten named deps dict to a list of paths."""
    result = list[str]()
    for value in deps.values():
        if isinstance(value, (list, tuple)):
            result.extend(value)
        else:
            result.append(value)
    return result


def _normalize_deps_dict(
    deps: dict[str, outputs.PathType],
    cwd: pathlib.Path | None,
    validation_mode: ValidationMode,
) -> dict[str, outputs.PathType]:
    """Normalize all paths in deps dict to absolute paths."""
    result = dict[str, outputs.PathType]()
    for name, value in deps.items():
        if isinstance(value, (list, tuple)):
            normalized = _normalize_paths(
                list(value), path_policy.PathType.DEP, validation_mode, cwd
            )
            # Preserve tuple type for fixed-length deps
            result[name] = tuple(normalized) if isinstance(value, tuple) else normalized
        else:
            normalized = _normalize_paths([value], path_policy.PathType.DEP, validation_mode, cwd)
            result[name] = normalized[0]
    return result


REGISTRY = StageRegistry()
