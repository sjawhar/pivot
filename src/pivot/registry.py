from __future__ import annotations

import dataclasses
import enum
import inspect
import logging
import pathlib
import re
from difflib import get_close_matches
from typing import TYPE_CHECKING, Any, TypedDict, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

import pydantic

from pivot import (
    exceptions,
    fingerprint,
    metrics,
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


class OutOverride(_OutOverrideOptions):
    """Override options for an annotation-defined output.

    path is required, other options are optional and override annotation defaults.
    """

    path: outputs.PathType


# Accept either a simple path string or a full OutOverride dict
OutOverrideInput = outputs.PathType | OutOverride


class RegistryStageInfo(TypedDict):
    """Metadata for a registered stage.

    Attributes:
        func: The stage function to execute.
        name: Unique stage identifier (function name or custom name).
        deps: Named input file dependencies (name -> path(s), absolute paths).
        deps_paths: Flattened list of all dependency paths (for DAG/worker).
        outs: Output specifications (expanded for DAG/caching - one Out per file).
        outs_paths: Output file paths (absolute paths).
        params: Pydantic model instance with parameter values.
        mutex: Mutex groups for exclusive execution.
        variant: Variant name for matrix stages (None for regular stages).
        signature: Function signature for parameter injection.
        fingerprint: Code fingerprint mapping (key -> hash).
        dep_specs: Dependency specs from function annotations.
        out_specs: Output specs from return type (return key -> resolved Out, pre-expansion).
            For single-output stages, uses SINGLE_OUTPUT_KEY (convention for non-TypedDict returns).
        params_arg_name: Name of the StageParams parameter in function signature (or None).
        state_dir: Per-stage state directory (None means use default determined at runtime).
    """

    func: Callable[..., Any]
    name: str
    # deps: Named dependencies for injection (name -> path mapping)
    # deps_paths: Flat list for DAG construction and fingerprint hashing
    deps: dict[str, outputs.PathType]
    deps_paths: list[str]
    outs: list[outputs.BaseOut]
    outs_paths: list[str]
    params: stage_def.StageParams | None
    mutex: list[str]
    variant: str | None
    signature: Signature | None
    fingerprint: dict[str, str]
    dep_specs: dict[str, stage_def.FuncDepSpec]
    out_specs: dict[str, outputs.BaseOut]
    params_arg_name: str | None
    state_dir: pathlib.Path | None


class ValidationMode(enum.StrEnum):
    """Validation strictness levels."""

    ERROR = "error"  # Raise exception on validation failure
    WARN = "warn"  # Log warning, allow registration


# Stage name pattern: must start with letter, then alphanumeric/underscore/hyphen
_STAGE_NAME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]*$")


def _normalize_out_override(value: OutOverrideInput) -> OutOverride:
    """Normalize output override input to OutOverride dict.

    Accepts simple path strings or full OutOverride dicts.
    """
    if isinstance(value, (str, list, tuple)):
        return OutOverride(path=value)
    return value


def _apply_out_overrides(
    out_spec: outputs.BaseOut,
    override: OutOverride | None,
) -> outputs.BaseOut:
    """Apply path and option overrides to an output spec, preserving subclass type and loader.

    Does NOT expand multi-file paths - returns a single output with the resolved path (may be list/tuple).
    Expansion for DAG/caching is handled separately.

    Returns a single output object with overrides applied.
    """
    # DirectoryOut has str path (ends with '/'), can't be overridden the same way
    if isinstance(out_spec, outputs.DirectoryOut):
        if override:
            path = override["path"]
            if not isinstance(path, str) or not path.endswith("/"):
                raise exceptions.ValidationError(
                    f"DirectoryOut path override must be a string ending with '/': {path!r}"
                )
            cache = override.get("cache", out_spec.cache)
            return dataclasses.replace(out_spec, path=path, cache=cache)
        return out_spec

    # IncrementalOut - can update cache but not path (handled by _resolve_out_spec)
    if isinstance(out_spec, outputs.IncrementalOut):
        if override:
            cache = override.get("cache", out_spec.cache)
            path = override["path"] if "path" in override else out_spec.path
            return dataclasses.replace(out_spec, path=path, cache=cache)
        return out_spec

    # Out and subclasses (Metric, Plot)
    # Determine final path (override or annotation default)
    path = override["path"] if override else out_spec.path

    # Determine final cache (override takes precedence, then annotation default)
    # Note: annotation default is already set correctly for Out/Metric/Plot subclasses
    cache = override["cache"] if override and "cache" in override else out_spec.cache

    # Cast to Out for dataclasses.replace - we checked it's not DirectoryOut/IncrementalOut
    out = cast("outputs.Out[Any]", out_spec)
    return dataclasses.replace(out, path=path, cache=cache)


def _expand_out_spec(out_spec: outputs.BaseOut) -> list[outputs.BaseOut]:
    """Expand multi-file output spec into individual output objects for DAG/caching.

    For multi-file outputs (path is list/tuple), creates individual output objects for each path.
    For single-file outputs, returns a single-item list.
    DirectoryOut is not expanded (paths are determined at runtime).
    """
    # DirectoryOut is not expanded - its paths are determined at runtime
    if isinstance(out_spec, outputs.DirectoryOut):
        return [out_spec]

    # IncrementalOut - single path only, return as-is
    if isinstance(out_spec, outputs.IncrementalOut):
        return [out_spec]

    # Out and subclasses (Metric, Plot) - may have list/tuple paths
    out = cast("outputs.Out[Any]", out_spec)
    path = out.path
    if isinstance(path, (list, tuple)):
        return [dataclasses.replace(out, path=p) for p in path]
    else:
        return [out]


def _resolve_out_spec(
    out_name: str,
    out_spec: outputs.BaseOut,
    override: OutOverride | None,
    stage_name: str,
) -> tuple[outputs.BaseOut, list[outputs.BaseOut]]:
    """Apply overrides to an output spec and expand for DAG/caching.

    Validates that IncrementalOut outputs aren't overridden, applies path/cache overrides,
    and expands multi-file specs into individual output objects.

    Args:
        out_name: Output key name (for error messages).
        out_spec: Original output spec from function annotations.
        override: Path/cache override from YAML (or None).
        stage_name: Stage name (for error messages).

    Returns:
        Tuple of (resolved_spec, expanded_specs) where:
        - resolved_spec: Single output with overrides applied (may have list/tuple path)
        - expanded_specs: List of outputs with single-string paths for DAG/caching

    Raises:
        ValidationError: If trying to override an IncrementalOut path.
    """
    # IncrementalOut paths must match between input and output - disallow overrides
    if override is not None and isinstance(out_spec, outputs.IncrementalOut):
        raise exceptions.ValidationError(
            f"Stage '{stage_name}': cannot override IncrementalOut output path for '{out_name}'. "
            + "IncrementalOut paths must match between input and output annotations."
        )

    resolved = _apply_out_overrides(out_spec, override)
    expanded = _expand_out_spec(resolved)
    return resolved, expanded


class StageRegistry:
    """Registry for pipeline stages.

    The registry stores metadata for all stages registered via `register()`.
    It handles validation, path normalization, and dependency graph construction.

    Stages are registered from pivot.yaml or programmatically. Dependencies and outputs
    are extracted from function annotations (Annotated[T, Dep(...)] and TypedDict
    return types with Out annotations).

    Direct instantiation of StageRegistry is mainly useful for testing
    with isolated registries. For production use, use the `Pipeline` class.

    Example:
        ```python
        from pivot.pipeline import Pipeline
        pipeline = Pipeline("my_pipeline")
        pipeline.register(my_stage)
        pipeline.list_stages()  # ['my_stage']
        ```
    """

    def __init__(self, validation_mode: ValidationMode = ValidationMode.ERROR) -> None:
        self._stages: dict[str, RegistryStageInfo] = {}
        self._cached_dag: DiGraph[str] | None = None
        self.validation_mode: ValidationMode = validation_mode

    def register(
        self,
        func: Callable[..., Any],
        name: str | None = None,
        params: ParamsArg = None,
        mutex: Sequence[str] | None = None,
        variant: str | None = None,
        dep_path_overrides: Mapping[str, outputs.PathType] | None = None,
        out_path_overrides: Mapping[str, OutOverrideInput] | None = None,
        state_dir: pathlib.Path | None = None,
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
            dep_path_overrides: Override paths for deps (must match annotation dep names).
            out_path_overrides: Override paths/options for outputs. Accepts simple path strings
                or dicts with path and options: `{"result": "out.csv"}` or
                `{"result": {"path": "out.csv", "cache": False}}`.
            state_dir: Per-stage state directory (None means use default determined at runtime).

        Raises:
            ValidationError: If stage name is invalid or already registered.
            SecurityValidationError: If paths contain traversal components.
            InvalidPathError: If paths resolve outside project root.
            ParamsError: If params specified but function lacks params argument.
        """
        with metrics.timed("registry.register"):
            # Invalidate DAG cache on any registration
            self._cached_dag = None

            stage_name = name if name is not None else func.__name__

            # Warn about lambda functions - their fingerprints are non-deterministic
            if func.__name__ == "<lambda>":
                logger.warning(
                    f"Stage '{stage_name}' uses a lambda function. Lambda fingerprinting is non-deterministic "
                    + "and will cause unnecessary re-runs. Use a named function instead."
                )

            mutex_list: list[str] = [m.strip().lower() for m in mutex] if mutex else []

            # Convert params to instance (instantiate class if needed)
            params_instance = _resolve_params(params, func, stage_name)

            # Identify placeholder deps BEFORE extraction
            placeholder_names = stage_def.get_placeholder_dep_names(func)

            # Validate all placeholders have overrides
            if placeholder_names:
                provided_overrides = (
                    set(dep_path_overrides.keys()) if dep_path_overrides else set[str]()
                )
                missing = placeholder_names - provided_overrides
                if missing:
                    # Check for typos in provided overrides
                    unknown_overrides = list(provided_overrides - placeholder_names)
                    suggestions = list[str]()
                    for m in sorted(missing):
                        matches = get_close_matches(m, unknown_overrides, n=1, cutoff=0.6)
                        if matches:
                            suggestions.append(f"'{matches[0]}' -> '{m}'")
                    msg = (
                        f"Stage '{stage_name}' has invalid dependencies:\n"
                        + f"  - Placeholder dependencies missing overrides: {', '.join(sorted(missing))}"
                    )
                    if suggestions:
                        msg += f"\n  - Did you mean: {', '.join(suggestions)}?"
                    raise exceptions.ValidationError(msg)

            # Extract deps from function annotations (PlaceholderDep resolved via overrides)
            dep_specs = stage_def.get_dep_specs_from_signature(func, dep_path_overrides)

            # Validate dep_path_overrides match annotation dep names
            if dep_path_overrides:
                unknown = set(dep_path_overrides.keys()) - set(dep_specs.keys())
                if unknown:
                    raise exceptions.ValidationError(
                        f"Stage '{stage_name}': dep_path_overrides contains unknown deps: {unknown}. "
                        + f"Available: {list(dep_specs.keys())}"
                    )
                # Disallow overrides for IncrementalOut inputs - path must match output annotation
                incremental_overrides = [
                    name for name in dep_path_overrides if not dep_specs[name].creates_dep_edge
                ]
                if incremental_overrides:
                    raise exceptions.ValidationError(
                        f"Stage '{stage_name}': cannot override IncrementalOut input paths: "
                        + f"{incremental_overrides}. IncrementalOut paths must match between "
                        + "input and output annotations."
                    )

            # Build deps dict from specs (all deps, for loading)
            deps_dict: dict[str, outputs.PathType] = {
                dep_name: spec.path for dep_name, spec in dep_specs.items()
            }

            # Flatten ALL deps for path validation (security check applies to all paths)
            all_deps_flat = _flatten_deps(deps_dict)

            # Flatten deps for DAG, excluding deps that don't create edges
            # (IncrementalOut as input is self-referential, no DAG edge to avoid circular dependency)
            deps_dict_for_dag = {
                dep_name: spec.path for dep_name, spec in dep_specs.items() if spec.creates_dep_edge
            }
            deps_flat = _flatten_deps(deps_dict_for_dag)

            # Extract outs from return type annotations
            return_out_specs = stage_def.get_output_specs_from_return(func, stage_name)
            single_out_spec = stage_def.get_single_output_spec_from_return(func)

            # Validate IncrementalOut input/output matching
            _validate_incremental_out_matching(
                stage_name, dep_specs, return_out_specs, single_out_spec
            )

            # Build out_specs from return annotations (return key -> resolved output, pre-expansion)
            # For single-output stages (non-TypedDict return), uses SINGLE_OUTPUT_KEY convention
            out_specs: dict[str, outputs.BaseOut] = {}
            outs_from_annotations: list[outputs.BaseOut] = []

            if return_out_specs:
                # Validate out_path_overrides match annotation out names
                if out_path_overrides:
                    unknown = set(out_path_overrides.keys()) - set(return_out_specs.keys())
                    if unknown:
                        raise exceptions.ValidationError(
                            f"Stage '{stage_name}': out_path_overrides contains unknown outs: {unknown}. "
                            + f"Available: {list(return_out_specs.keys())}"
                        )

                # Apply overrides and expand each output spec
                for out_name, out_spec in return_out_specs.items():
                    raw_override = out_path_overrides.get(out_name) if out_path_overrides else None
                    override = _normalize_out_override(raw_override) if raw_override else None
                    resolved, expanded = _resolve_out_spec(out_name, out_spec, override, stage_name)
                    out_specs[out_name] = resolved
                    outs_from_annotations.extend(expanded)

            elif single_out_spec is not None:
                # Single annotated return type - uses SINGLE_OUTPUT_KEY convention
                override: OutOverride | None = None
                if out_path_overrides:
                    if len(out_path_overrides) > 1:
                        raise exceptions.ValidationError(
                            f"Stage '{stage_name}': single-output stage has "
                            + f"{len(out_path_overrides)} out_path_overrides keys "
                            + f"({list(out_path_overrides.keys())}). "
                            + "Only one key is allowed for single-output stages."
                        )
                    # Get the single override (whatever key the user used)
                    override = _normalize_out_override(next(iter(out_path_overrides.values())))

                resolved, expanded = _resolve_out_spec(
                    stage_def.SINGLE_OUTPUT_KEY, single_out_spec, override, stage_name
                )
                out_specs[stage_def.SINGLE_OUTPUT_KEY] = resolved
                outs_from_annotations.extend(expanded)

            outs_list = outs_from_annotations
            # After _apply_out_overrides, each Out has a single-string path (multi-file paths expanded)
            outs_paths = [str(o.path) for o in outs_list]

            # Validate paths BEFORE normalizing (check ".." on original paths)
            # Use all_deps_flat to include IncrementalOut paths in security validation
            _validate_stage_registration(
                self._stages, stage_name, all_deps_flat, outs_paths, self.validation_mode
            )

            # Normalize dep paths - flatten, normalize, then rebuild dict
            deps_flat_normalized = _normalize_paths(
                deps_flat, path_policy.PathType.DEP, self.validation_mode
            )
            outs_paths = _normalize_paths(
                outs_paths, path_policy.PathType.OUT, self.validation_mode
            )

            # Rebuild deps dict with normalized paths
            deps_normalized = _normalize_deps_dict(deps_dict, self.validation_mode)

            # Update normalized outputs with absolute paths
            # All concrete output types (Out, DirectoryOut, IncrementalOut) are dataclasses
            outs_normalized: list[outputs.BaseOut] = []
            for out, path in zip(outs_list, outs_paths, strict=True):
                if isinstance(out, (outputs.DirectoryOut, outputs.IncrementalOut)):
                    outs_normalized.append(dataclasses.replace(out, path=path))
                else:
                    # Out and subclasses (Metric, Plot)
                    outs_normalized.append(
                        dataclasses.replace(cast("outputs.Out[Any]", out), path=path)
                    )

            # Output overlap validation is deferred to validate_outputs() for performance
            # (single O(N) pass instead of O(N²) from checking on every register)

            # Build stage fingerprint (includes loader fingerprints from annotations)
            stage_fp = fingerprint.get_stage_fingerprint(func)
            stage_fp.update(
                _get_annotation_loader_fingerprints(dep_specs, return_out_specs, single_out_spec)
            )

            # Get params arg name for worker (avoids re-inspecting signature at execution time)
            params_arg_name, _ = stage_def.find_params_in_signature(func)

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
                dep_specs=dep_specs,
                out_specs=out_specs,
                params_arg_name=params_arg_name,
                state_dir=state_dir,
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
        # Return cached DAG if available and validation matches
        # Only cache when validate=True (the common case for commands)
        if validate and self._cached_dag is not None:
            return self._cached_dag

        from pivot import dag
        from pivot.storage import track

        # Discover tracked files to recognize them as valid dependency sources
        tracked_files = None
        if validate:
            tracked_files = track.discover_pvt_files(project.get_project_root())

        graph = dag.build_dag(self._stages, validate=validate, tracked_files=tracked_files)

        # Validate output paths don't conflict (only when validate=True)
        if validate:
            self.validate_outputs()

        # Cache only when validating (safe to reuse)
        if validate:
            self._cached_dag = graph

        return graph

    def clear(self) -> None:
        """Clear all registered stages (for testing)."""
        self._stages.clear()
        self._cached_dag = None

    def invalidate_dag_cache(self) -> None:
        """Invalidate cached DAG without clearing stages.

        Call when external state changes (code reload, config change) that
        would affect DAG construction but stage registrations haven't changed yet.
        """
        self._cached_dag = None

    def snapshot(self) -> dict[str, RegistryStageInfo]:
        """Create a snapshot of current registry state for backup/restore.

        Returns a shallow copy of the internal stages dict. Use with `restore()`
        to implement atomic reload patterns where you want to preserve the previous
        valid state if the reload fails.

        Example:
            backup = registry.snapshot()
            registry.clear()
            try:
                reload_stages()
            except Exception:
                registry.restore(backup)  # Rollback on failure
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
        self._cached_dag = None

    def get_all_output_paths(self) -> set[str]:
        """Get all registered output paths (for watch mode filtering)."""
        result = set[str]()
        for info in self._stages.values():
            for out_path in info["outs_paths"]:
                result.add(str(out_path))
        return result

    def validate_outputs(self) -> None:
        """Validate no output path conflicts between stages.

        This is called once after all stages are registered, instead of
        checking on every register() call. Raises OutputDuplicationError
        or OverlappingOutputPathsError if conflicts are found.
        """
        if not self._stages:
            return
        temp_stages: dict[str, trie.TrieStageInfo] = {
            name: {"name": name, "outs": info["outs_paths"]} for name, info in self._stages.items()
        }
        trie.build_outs_trie(temp_stages)


def _normalize_paths(
    paths: Sequence[str],
    path_type: path_policy.PathType,
    validation_mode: ValidationMode,
) -> list[str]:
    """Normalize paths to absolute paths, applying policy-based validation.

    All paths are relative to project root.

    Args:
        paths: Paths to normalize
        path_type: Type of path (DEP or OUT) for policy lookup
        validation_mode: How to handle validation errors

    Raises:
        InvalidPathError: If path violates its type's policy
    """
    normalized = list[str]()
    project_root = project.get_project_root()
    policy = path_policy.POLICIES[path_type]

    for path in paths:
        # Preserve trailing slash for directory paths (DirectoryOut)
        is_dir_path = path.endswith("/")

        try:
            # Normalize path to absolute (from project root)
            if pathlib.Path(path).is_absolute():
                norm_path = pathlib.Path(path)
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

            # Restore trailing slash for directory paths (pathlib strips it)
            result_path = str(norm_path) + "/" if is_dir_path else str(norm_path)
            normalized.append(result_path)
        except (ValueError, OSError, exceptions.InvalidPathError):
            if validation_mode == ValidationMode.WARN:
                result_path = str(project.normalize_path(path))
                if is_dir_path:
                    result_path += "/"
                normalized.append(result_path)
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


def _validate_incremental_spec_match(
    stage_name: str,
    input_name: str | None,
    input_spec: stage_def.FuncDepSpec,
    output_spec: outputs.BaseOut,
) -> None:
    """Validate IncrementalOut input/output specs match (path and loader)."""
    name_part = f"'{input_name}' " if input_name else ""

    if input_spec.path != output_spec.path:
        raise exceptions.ValidationError(
            f"Stage '{stage_name}': IncrementalOut input {name_part}path "
            + f"'{input_spec.path}' doesn't match output path '{output_spec.path}'"
        )
    if input_spec.loader != output_spec.loader:
        raise exceptions.ValidationError(
            f"Stage '{stage_name}': IncrementalOut input {name_part}loader "
            + f"{input_spec.loader!r} doesn't match output loader {output_spec.loader!r}"
        )


def _validate_incremental_out_matching(
    stage_name: str,
    dep_specs: dict[str, stage_def.FuncDepSpec],
    return_out_specs: dict[str, outputs.BaseOut],
    single_out_spec: outputs.BaseOut | None,
) -> None:
    """Validate IncrementalOut inputs have matching outputs.

    Matching rules:
    - For TypedDict returns: IncrementalOut output field name must match parameter name
    - For single output (non-TypedDict): Can only have ONE IncrementalOut input parameter
    - Paths and loaders must match between input and output
    """
    # Find IncrementalOut inputs (deps with creates_dep_edge=False are IncrementalOut)
    incremental_inputs = {
        name: spec for name, spec in dep_specs.items() if not spec.creates_dep_edge
    }

    if not incremental_inputs:
        return  # No IncrementalOut inputs, nothing to validate

    # Case 1: TypedDict return
    if return_out_specs:
        incremental_outputs = {
            name: spec
            for name, spec in return_out_specs.items()
            if isinstance(spec, outputs.IncrementalOut)
        }

        for input_name, input_spec in incremental_inputs.items():
            if input_name not in incremental_outputs:
                raise exceptions.ValidationError(
                    f"Stage '{stage_name}': IncrementalOut input parameter '{input_name}' "
                    + "has no matching IncrementalOut output field. "
                    + "For TypedDict returns, the output field name must match the parameter name."
                )
            _validate_incremental_spec_match(
                stage_name, input_name, input_spec, incremental_outputs[input_name]
            )

    # Case 2: Single output (non-TypedDict)
    elif single_out_spec is not None:
        if not isinstance(single_out_spec, outputs.IncrementalOut):
            raise exceptions.ValidationError(
                f"Stage '{stage_name}': has IncrementalOut input but return type "
                + "is not IncrementalOut"
            )

        if len(incremental_inputs) > 1:
            raise exceptions.ValidationError(
                f"Stage '{stage_name}': single-output stages can only have one "
                + f"IncrementalOut input parameter, found {len(incremental_inputs)}: "
                + f"{list(incremental_inputs.keys())}"
            )

        input_spec = next(iter(incremental_inputs.values()))
        _validate_incremental_spec_match(stage_name, None, input_spec, single_out_spec)

    # Case 3: No matching output
    else:
        raise exceptions.ValidationError(
            f"Stage '{stage_name}': has IncrementalOut inputs but no IncrementalOut output. "
            + "Every IncrementalOut input must have a corresponding IncrementalOut output."
        )


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
    return_out_specs: dict[str, outputs.BaseOut],
    single_out_spec: outputs.BaseOut | None,
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
    validation_mode: ValidationMode,
) -> dict[str, outputs.PathType]:
    """Normalize all paths in deps dict to absolute paths."""
    result = dict[str, outputs.PathType]()
    for name, value in deps.items():
        if isinstance(value, (list, tuple)):
            normalized = _normalize_paths(list(value), path_policy.PathType.DEP, validation_mode)
            # Preserve tuple type for fixed-length deps
            result[name] = tuple(normalized) if isinstance(value, tuple) else normalized
        else:
            normalized = _normalize_paths([value], path_policy.PathType.DEP, validation_mode)
            result[name] = normalized[0]
    return result
