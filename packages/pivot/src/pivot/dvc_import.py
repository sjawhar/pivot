from __future__ import annotations

import contextlib
import logging
import os
import re
import tempfile
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import ruamel.yaml
import yaml

from pivot import exceptions, path_policy, project, yaml_config

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

logger = logging.getLogger(__name__)

# Maximum file size for dvc.yaml/dvc.lock (10MB)
MAX_DVC_FILE_SIZE = 10 * 1024 * 1024


class DVCStageConfig(TypedDict, total=False):
    """Raw DVC stage from dvc.yaml."""

    cmd: str | list[str]
    deps: list[str | dict[str, Any]]
    outs: list[str | dict[str, Any]]
    params: list[str | dict[str, Any]]
    metrics: list[str | dict[str, Any]]
    plots: list[str | dict[str, Any]]
    wdir: str
    frozen: bool
    foreach: list[Any] | dict[str, Any]
    do: DVCStageConfig


class DVCConfig(TypedDict):
    """Top-level dvc.yaml structure."""

    stages: dict[str, DVCStageConfig]


class DVCLockStage(TypedDict, total=False):
    """Stage entry from dvc.lock."""

    cmd: str
    deps: list[dict[str, Any]]
    outs: list[dict[str, Any]]
    params: dict[str, dict[str, Any] | None]  # Values can be null in YAML


class DVCLock(TypedDict, total=False):
    """Top-level dvc.lock structure."""

    schema: str
    stages: dict[str, DVCLockStage]


class PivotStageConfig(TypedDict, total=False):
    """Stage config for pivot.yaml output."""

    python: str
    deps: list[str]
    outs: list[str | dict[str, dict[str, bool]]]
    metrics: list[str]
    plots: list[str]
    params: dict[str, Any]
    matrix: dict[str, list[str]]


class MigrationNote(TypedDict):
    """A migration note/warning for the user."""

    stage: str | None
    severity: Literal["error", "warning", "info"]
    message: str
    original_cmd: str | None


class ConversionStats(TypedDict):
    """Statistics about the conversion."""

    stages_converted: int
    stages_with_shell_commands: int
    stages_with_warnings: int
    params_inlined: int


class ConversionResult(TypedDict):
    """Result of converting a DVC pipeline."""

    stages: dict[str, PivotStageConfig]
    notes: list[MigrationNote]
    stats: ConversionStats


class _ResolveParamsResult(TypedDict):
    """Result of resolving params."""

    params: dict[str, Any]
    count: int
    notes: list[MigrationNote]


def parse_dvc_yaml(path: Path) -> DVCConfig:
    """Parse dvc.yaml with size limit and safe loader."""
    _check_file_size(path, "dvc.yaml")

    try:
        with path.open() as f:
            data = yaml.load(f, Loader=yaml_config.Loader)
    except yaml.YAMLError as e:
        raise exceptions.DVCImportError(f"Invalid YAML in {path}: {e}") from e

    if data is None:
        raise exceptions.DVCImportError(f"dvc.yaml is empty: {path}")

    if not isinstance(data, dict):
        raise exceptions.DVCImportError(f"dvc.yaml must be a mapping, got {type(data).__name__}")

    if "stages" not in data:
        raise exceptions.DVCImportError("dvc.yaml missing 'stages' key")

    stages_data: dict[str, DVCStageConfig] = data["stages"]  # pyright: ignore[reportUnknownVariableType] - yaml returns Any
    if not isinstance(stages_data, dict):
        raise exceptions.DVCImportError("dvc.yaml 'stages' must be a mapping")

    return DVCConfig(stages=stages_data)  # pyright: ignore[reportUnknownArgumentType] - validated above


def parse_dvc_lock(path: Path) -> DVCLock | None:
    """Parse dvc.lock if present. Returns None if file doesn't exist."""
    if not path.exists():
        return None

    _check_file_size(path, "dvc.lock")

    try:
        with path.open() as f:
            data = yaml.load(f, Loader=yaml_config.Loader)
    except yaml.YAMLError as e:
        logger.warning("Failed to parse dvc.lock: %s", e)
        return None

    if data is None or not isinstance(data, dict):
        return None

    schema: str = data["schema"] if "schema" in data else ""  # pyright: ignore[reportUnknownVariableType] - yaml returns Any
    stages_raw = data["stages"] if "stages" in data else {}  # pyright: ignore[reportUnknownVariableType]
    stages: dict[str, DVCLockStage] = stages_raw if isinstance(stages_raw, dict) else {}  # pyright: ignore[reportUnknownVariableType]

    return DVCLock(
        schema=schema if isinstance(schema, str) else "",
        stages=stages,
    )


def parse_params_yaml(path: Path) -> tuple[dict[str, Any], list[MigrationNote]]:
    """Parse params.yaml for parameter inlining. Returns (params, notes)."""
    notes = list[MigrationNote]()

    if not path.exists():
        return {}, notes

    _check_file_size(path, "params.yaml")

    try:
        with path.open() as f:
            data = yaml.load(f, Loader=yaml_config.Loader)
    except yaml.YAMLError as e:
        raise exceptions.DVCImportError(f"Invalid YAML in {path}: {e}") from e

    if data is None:
        return {}, notes

    if not isinstance(data, dict):
        raise exceptions.DVCImportError(f"params.yaml must be a mapping, got {type(data).__name__}")

    # Check for non-string keys and warn
    result = dict[str, Any]()
    non_string_keys = list[str]()
    for k, v in data.items():  # pyright: ignore[reportUnknownVariableType] - yaml returns Any
        if isinstance(k, str):
            result[k] = v
        else:
            non_string_keys.append(repr(k))  # pyright: ignore[reportUnknownArgumentType]

    if non_string_keys:
        notes.append(
            MigrationNote(
                stage=None,
                severity="warning",
                message=f"params.yaml contains non-string keys that were skipped: {', '.join(non_string_keys)}",
                original_cmd=None,
            )
        )

    return result, notes


def _check_file_size(path: Path, name: str) -> None:
    """Check file size doesn't exceed limit."""
    try:
        size = path.stat().st_size
        if size > MAX_DVC_FILE_SIZE:
            raise exceptions.DVCImportError(
                f"{name} exceeds maximum size ({MAX_DVC_FILE_SIZE // (1024 * 1024)}MB)"
            )
    except OSError as e:
        raise exceptions.DVCImportError(f"Cannot read {name}: {e}") from e


def convert_pipeline(
    dvc_yaml_path: Path,
    dvc_lock_path: Path | None = None,
    params_yaml_path: Path | None = None,
    project_root: Path | None = None,
) -> ConversionResult:
    """Convert DVC pipeline to Pivot format.

    Args:
        dvc_yaml_path: Path to dvc.yaml
        dvc_lock_path: Path to dvc.lock (optional, for param values)
        params_yaml_path: Path to params.yaml (optional, for param values)
        project_root: Project root for path validation (default: auto-detect)

    Returns:
        ConversionResult with stages, notes, and stats
    """
    proj_root = project_root or project.get_project_root()

    # Parse input files
    dvc_config = parse_dvc_yaml(dvc_yaml_path)
    dvc_lock = parse_dvc_lock(dvc_lock_path) if dvc_lock_path else None

    notes = list[MigrationNote]()
    if params_yaml_path:
        params_yaml, params_notes = parse_params_yaml(params_yaml_path)
        notes.extend(params_notes)
    else:
        params_yaml = {}

    stages = dict[str, PivotStageConfig]()
    stats = ConversionStats(
        stages_converted=0,
        stages_with_shell_commands=0,
        stages_with_warnings=0,
        params_inlined=0,
    )

    for stage_name, dvc_stage in dvc_config["stages"].items():
        # Handle foreach stages
        if "foreach" in dvc_stage:
            converted, stage_notes, params_count = _convert_foreach_stage(
                stage_name, dvc_stage, params_yaml, dvc_lock, proj_root
            )
            stages.update(converted)
            notes.extend(stage_notes)
            stats["stages_converted"] += len(converted)
            # Only count shell commands if do block has cmd
            if "do" in dvc_stage and "cmd" in dvc_stage["do"]:
                stats["stages_with_shell_commands"] += len(converted)
            stats["params_inlined"] += params_count
        else:
            pivot_stage, stage_notes, params_count = _convert_stage(
                stage_name, dvc_stage, params_yaml, dvc_lock, proj_root
            )
            stages[stage_name] = pivot_stage
            notes.extend(stage_notes)
            stats["stages_converted"] += 1
            if "cmd" in dvc_stage:
                stats["stages_with_shell_commands"] += 1
            stats["params_inlined"] += params_count

    # Count warnings
    stats["stages_with_warnings"] = sum(1 for n in notes if n["severity"] == "warning")

    return ConversionResult(stages=stages, notes=notes, stats=stats)


def _convert_stage(
    name: str,
    dvc_stage: DVCStageConfig,
    params_yaml: dict[str, Any],
    dvc_lock: DVCLock | None,
    project_root: Path,
) -> tuple[PivotStageConfig, list[MigrationNote], int]:
    """Convert a single DVC stage to Pivot format. Returns (stage, notes, params_count)."""
    notes = list[MigrationNote]()
    pivot_stage = PivotStageConfig()
    params_count = 0

    # Get wdir prefix (DVC's working directory for the stage)
    wdir: str | None = dvc_stage.get("wdir")
    if wdir:
        notes.append(
            MigrationNote(
                stage=name,
                severity="info",
                message=f"Paths prefixed with wdir '{wdir}' for project-root-relative resolution.",
                original_cmd=None,
            )
        )

    # cmd -> python (PLACEHOLDER)
    pivot_stage["python"] = f"PLACEHOLDER.{name}"
    if "cmd" in dvc_stage:
        cmd = dvc_stage["cmd"]
        cmd_str = cmd if isinstance(cmd, str) else " && ".join(str(c) for c in cmd)
        notes.append(
            MigrationNote(
                stage=name,
                severity="warning",
                message="Shell command requires manual conversion to Python function",
                original_cmd=cmd_str,
            )
        )

    # deps - validate and pass through (with wdir prefix if present)
    if "deps" in dvc_stage:
        deps = _prefix_paths_with_wdir(dvc_stage["deps"], wdir)
        pivot_stage["deps"] = _extract_paths(deps, name, "dep", project_root, notes)

    # outs - convert format (with wdir prefix if present)
    if "outs" in dvc_stage:
        outs = _prefix_paths_with_wdir(dvc_stage["outs"], wdir)
        pivot_stage["outs"] = _convert_outs(outs, name, project_root, notes)

    # metrics (with wdir prefix if present)
    if "metrics" in dvc_stage:
        metrics = _prefix_paths_with_wdir(dvc_stage["metrics"], wdir)
        pivot_stage["metrics"] = _extract_paths(metrics, name, "metric", project_root, notes)

    # plots (with wdir prefix if present)
    if "plots" in dvc_stage:
        plots = _prefix_paths_with_wdir(dvc_stage["plots"], wdir)
        pivot_stage["plots"] = _extract_paths(plots, name, "plot", project_root, notes)

    # params - resolve and inline (no wdir prefix - these are key references, not paths)
    if "params" in dvc_stage:
        resolve_result = _resolve_params(dvc_stage["params"], params_yaml, dvc_lock, name)
        if resolve_result["params"]:
            pivot_stage["params"] = resolve_result["params"]
        params_count = resolve_result["count"]
        notes.extend(resolve_result["notes"])

    # frozen - warning only
    if "frozen" in dvc_stage and dvc_stage["frozen"]:
        notes.append(
            MigrationNote(
                stage=name,
                severity="warning",
                message=(
                    "Stage has 'frozen: true' which is not supported in Pivot. "
                    "Stage will run normally."
                ),
                original_cmd=None,
            )
        )

    return pivot_stage, notes, params_count


def _convert_foreach_stage(
    name: str,
    dvc_stage: DVCStageConfig,
    params_yaml: dict[str, Any],
    dvc_lock: DVCLock | None,
    project_root: Path,
) -> tuple[dict[str, PivotStageConfig], list[MigrationNote], int]:
    """Convert DVC foreach/do stage to Pivot matrix format."""
    notes = list[MigrationNote]()
    stages = dict[str, PivotStageConfig]()
    params_count = 0

    # foreach is guaranteed to exist by caller check (line 258: if "foreach" in dvc_stage)
    foreach = dvc_stage["foreach"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    do_stage = dvc_stage["do"] if "do" in dvc_stage else None

    if do_stage is None:
        notes.append(
            MigrationNote(
                stage=name,
                severity="error",
                message="foreach stage missing 'do' block",
                original_cmd=None,
            )
        )
        return stages, notes, params_count

    # Validate foreach is not empty and convert to matrix dimension
    if isinstance(foreach, list):
        if not foreach:
            notes.append(
                MigrationNote(
                    stage=name,
                    severity="error",
                    message="foreach list is empty",
                    original_cmd=None,
                )
            )
            return stages, notes, params_count
        matrix_values = [str(v) for v in foreach]
    elif isinstance(foreach, dict):  # pyright: ignore[reportUnnecessaryIsInstance] - type narrowing
        if not foreach:
            notes.append(
                MigrationNote(
                    stage=name,
                    severity="error",
                    message="foreach dict is empty",
                    original_cmd=None,
                )
            )
            return stages, notes, params_count
        matrix_values = list(foreach.keys())
        # Warn about lost dict values
        notes.append(
            MigrationNote(
                stage=name,
                severity="warning",
                message=(
                    "foreach dict values are not preserved in matrix conversion. "
                    "Only keys are used as matrix items."
                ),
                original_cmd=None,
            )
        )
    else:
        notes.append(
            MigrationNote(
                stage=name,
                severity="error",
                message=f"Unsupported foreach type: {type(foreach).__name__}",
                original_cmd=None,
            )
        )
        return stages, notes, params_count

    # Create a single stage with matrix
    pivot_stage, stage_notes, params_count = _convert_stage(
        name, do_stage, params_yaml, dvc_lock, project_root
    )
    notes.extend(stage_notes)

    # Add matrix configuration
    pivot_stage["matrix"] = {"item": matrix_values}

    stages[name] = pivot_stage

    notes.append(
        MigrationNote(
            stage=name,
            severity="info",
            message=(
                f"Converted foreach to matrix with {len(matrix_values)} variants. "
                "Verify ${item} substitutions work correctly."
            ),
            original_cmd=None,
        )
    )

    return stages, notes, params_count


def _prefix_paths_with_wdir(
    items: list[str | dict[str, Any]],
    wdir: str | None,
) -> list[str | dict[str, Any]]:
    """Prefix all paths in items with wdir if present.

    Handles both string paths and dict-form paths like {path: {options}}.
    Only skips literal absolute paths (starting with /). Paths with variable
    interpolation are prefixed since we can't determine at parse time whether
    they resolve to absolute or relative paths.
    """
    if not wdir:
        return items

    result = list[str | dict[str, Any]]()
    for item in items:
        if isinstance(item, str):
            # Only skip literal absolute paths
            if item.startswith("/"):
                result.append(item)
            else:
                result.append(f"{wdir}/{item}")
        else:
            # Dict form: {path: {options}} - prefix each key
            prefixed_dict = dict[str, Any]()
            for path, opts in item.items():
                if not path.startswith("/"):
                    prefixed_dict[f"{wdir}/{path}"] = opts
                else:
                    prefixed_dict[path] = opts
            result.append(prefixed_dict)
    return result


def _extract_paths(
    items: list[str | dict[str, Any]],
    stage_name: str,
    path_type: str,
    project_root: Path,
    notes: list[MigrationNote],
) -> list[str]:
    """Extract and validate paths from DVC list format (strings or dicts with path keys)."""
    result = list[str]()
    for item in items:
        if isinstance(item, str):
            if _validate_path(item, stage_name, path_type, project_root, notes):
                result.append(item)
        else:
            # Dict form: {path: {options}} - extract path keys
            for path in item:
                if _validate_path(path, stage_name, path_type, project_root, notes):
                    result.append(path)
    return result


def _convert_outs(
    outs: list[str | dict[str, Any]],
    stage_name: str,
    project_root: Path,
    notes: list[MigrationNote],
) -> list[str | dict[str, dict[str, bool]]]:
    """Convert DVC outs to Pivot format, preserving cache: false options."""
    result = list[str | dict[str, dict[str, bool]]]()
    for out in outs:
        if isinstance(out, str):
            if _validate_path(out, stage_name, "out", project_root, notes):
                result.append(out)
        else:
            # Dict form: {path: {cache: false, ...}}
            for path, opts in out.items():
                if not _validate_path(path, stage_name, "out", project_root, notes):
                    continue
                if isinstance(opts, dict) and "cache" in opts and opts["cache"] is False:
                    result.append({path: {"cache": False}})
                else:
                    result.append(path)
    return result


def _validate_path(
    path: Any,
    stage_name: str,
    path_type: str,
    project_root: Path,
    notes: list[MigrationNote],
) -> bool:
    """Validate path doesn't escape project root."""
    # Check for non-string paths (YAML allows numeric keys)
    if not isinstance(path, str):
        notes.append(
            MigrationNote(
                stage=stage_name,
                severity="error",
                message=f"Invalid {path_type}: expected string path, got {type(path).__name__}",
                original_cmd=None,
            )
        )
        return False

    # For paths with ${} interpolation, validate literal parts don't contain traversal
    if "${" in path:
        literal_parts = re.sub(r"\$\{[^}]*\}", "", path)
        if ".." in literal_parts:
            notes.append(
                MigrationNote(
                    stage=stage_name,
                    severity="error",
                    message=f"Path traversal not allowed in {path_type}: '{path}'",
                    original_cmd=None,
                )
            )
            return False
        return True

    try:
        path_policy.require_valid_path(
            path,
            path_policy.PathType.DEP,  # DEP allows relative paths
            project_root,
            context=f"stage '{stage_name}' {path_type}",
        )
        return True
    except exceptions.SecurityValidationError as e:
        notes.append(
            MigrationNote(
                stage=stage_name,
                severity="error",
                message=f"Invalid {path_type} path '{path}': {e}",
                original_cmd=None,
            )
        )
        return False


def _add_param_with_collision_check(
    param_key: str,
    value: Any,
    stage_name: str,
    result: dict[str, Any],
    seen_keys: dict[str, str],
    notes: list[MigrationNote],
) -> None:
    """Add param value to result, checking for leaf key collisions."""
    leaf_key = param_key.split(".")[-1]

    if leaf_key in seen_keys and seen_keys[leaf_key] != param_key:
        notes.append(
            MigrationNote(
                stage=stage_name,
                severity="warning",
                message=(
                    f"Param key collision: '{param_key}' and '{seen_keys[leaf_key]}' "
                    f"both map to leaf key '{leaf_key}'. Using value from '{param_key}'."
                ),
                original_cmd=None,
            )
        )

    seen_keys[leaf_key] = param_key
    result[leaf_key] = value


def _resolve_params(
    dvc_params: list[str | dict[str, Any]],
    params_yaml: dict[str, Any],
    dvc_lock: DVCLock | None,
    stage_name: str,
) -> _ResolveParamsResult:
    """Resolve DVC param references to actual values.

    DVC param formats:
    - "train.lr" -> params.yaml: train.lr
    - "params.yaml:train.lr" -> explicit file:key
    - "params.yaml:" -> whole file (not supported, skip)
    - {"params.yaml": ["train.lr", "train.epochs"]} -> list of keys
    """
    result = dict[str, Any]()
    notes = list[MigrationNote]()
    count = 0

    # Track seen leaf keys to detect collisions
    seen_keys: dict[str, str] = {}  # leaf_key -> original full key

    # Try to get params from lock file first (actual values used)
    lock_params = dict[str, Any]()
    if dvc_lock and "stages" in dvc_lock:
        dvc_lock_stages = dvc_lock["stages"]
        if stage_name in dvc_lock_stages:
            lock_stage = dvc_lock_stages[stage_name]
            if "params" in lock_stage:
                for file_params in lock_stage["params"].values():
                    if isinstance(file_params, dict):
                        lock_params.update(file_params)

    for param in dvc_params:
        if isinstance(param, str):
            # Handle "file:key" format
            if ":" in param:
                file_part, key_part = param.split(":", 1)
                if not key_part:
                    # Whole file dep - skip (can't inline whole file)
                    continue
                # Warn if using non-default params file
                if file_part != "params.yaml":
                    notes.append(
                        MigrationNote(
                            stage=stage_name,
                            severity="warning",
                            message=(
                                f"Param '{param}' references '{file_part}' which is not loaded. "
                                "Only params.yaml values are inlined."
                            ),
                            original_cmd=None,
                        )
                    )
                param_key = key_part
            else:
                param_key = param

            # Look up value: lock > params.yaml
            value = _get_nested_value(lock_params, param_key)
            if value is None:
                value = _get_nested_value(params_yaml, param_key)

            if value is not None:
                _add_param_with_collision_check(
                    param_key, value, stage_name, result, seen_keys, notes
                )
                count += 1

        else:
            # Dict form: {"params.yaml": ["key1", "key2"]}
            for file_name, keys in param.items():
                # Warn if using non-default params file
                if file_name != "params.yaml":
                    notes.append(
                        MigrationNote(
                            stage=stage_name,
                            severity="warning",
                            message=(
                                f"Params from '{file_name}' are not loaded. "
                                "Only params.yaml values are inlined."
                            ),
                            original_cmd=None,
                        )
                    )
                if not isinstance(keys, list):
                    continue
                for key in keys:  # pyright: ignore[reportUnknownVariableType] - validated above
                    if not isinstance(key, str):
                        continue
                    value = _get_nested_value(lock_params, key)
                    if value is None:
                        value = _get_nested_value(params_yaml, key)
                    if value is not None:
                        _add_param_with_collision_check(
                            key, value, stage_name, result, seen_keys, notes
                        )
                        count += 1

    return _ResolveParamsResult(params=result, count=count, notes=notes)


def _get_nested_value(data: dict[str, Any], dotted_key: str) -> Any:
    """Get value from nested dict using dotted key path."""
    keys = dotted_key.split(".")
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        if key not in current:
            return None
        current = current[key]  # pyright: ignore[reportUnknownVariableType] - traversing Any dict
    return current  # pyright: ignore[reportUnknownVariableType]


@contextlib.contextmanager
def _atomic_write(output_path: Path) -> Generator[int]:
    """Context manager for atomic file writes using temp file + rename."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=output_path.parent, suffix=".tmp")
    try:
        yield fd
        os.rename(tmp_path, output_path)
    except BaseException:
        # Close fd if still open (fdopen may not have been called or may have failed)
        with contextlib.suppress(OSError):
            os.close(fd)
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def write_pivot_yaml(
    stages: dict[str, PivotStageConfig],
    output_path: Path,
    force: bool = False,
) -> None:
    """Write pivot.yaml atomically."""
    if output_path.exists() and not force:
        raise exceptions.DVCImportError(
            f"Output file '{output_path}' already exists. Use --force to overwrite."
        )

    output = {"stages": stages}

    yaml_writer = ruamel.yaml.YAML(typ="rt")
    yaml_writer.default_flow_style = False
    with _atomic_write(output_path) as fd, os.fdopen(fd, "w") as f:
        yaml_writer.dump(output, f)


def write_migration_notes(
    notes: list[MigrationNote],
    stats: ConversionStats,
    output_path: Path,
    force: bool = False,
) -> None:
    """Write migration notes as markdown."""
    if output_path.exists() and not force:
        raise exceptions.DVCImportError(
            f"Notes file '{output_path}' already exists. Use --force to overwrite."
        )

    content = _generate_migration_notes_content(notes, stats)

    with _atomic_write(output_path) as fd, os.fdopen(fd, "w") as f:
        f.write(content)


def _add_notes_section(
    lines: list[str],
    title: str,
    items: list[MigrationNote],
) -> None:
    """Add a section of notes to the markdown output."""
    if not items:
        return
    lines.extend([f"## {title}", ""])
    for note in items:
        prefix = f"Stage '{note['stage']}': " if note["stage"] else ""
        lines.append(f"- {prefix}{note['message']}")
    lines.append("")


def _generate_migration_notes_content(
    notes: list[MigrationNote],
    stats: ConversionStats,
) -> str:
    """Generate migration notes markdown content."""
    lines = [
        "# DVC to Pivot Migration Notes",
        "",
        "## WARNING - Security Review Required",
        "",
        "Shell commands from dvc.yaml are shown below. **Review carefully before executing.**",
        "",
        "## Summary",
        "",
        f"- {stats['stages_converted']} stages converted",
        f"- {stats['stages_with_shell_commands']} require Python function mapping",
        f"- {stats['stages_with_warnings']} warnings",
        f"- {stats['params_inlined']} params inlined",
        "",
    ]

    # Group notes by severity
    errors = [n for n in notes if n["severity"] == "error"]
    warnings = [n for n in notes if n["severity"] == "warning"]
    infos = [n for n in notes if n["severity"] == "info"]

    # Shell command notes (the main ones requiring action)
    shell_notes = [n for n in warnings if n["original_cmd"]]
    if shell_notes:
        lines.extend(["## Required: Python Function Mapping", ""])
        for note in shell_notes:
            stage = note["stage"] or "unknown"
            cmd = note["original_cmd"] or ""
            # Escape backticks in command for markdown
            escaped_cmd = cmd.replace("`", "\\`")
            lines.extend(
                [
                    f"### Stage: {stage}",
                    "",
                    f"- **Original command:** `{escaped_cmd}`",
                    f"- **Current:** `python: PLACEHOLDER.{stage}`",
                    f"- **Action:** Create a Python function and update to e.g. `python: src.{stage}.main`",
                    "",
                ]
            )

    # Use helper for simple sections
    _add_notes_section(lines, "Errors", errors)
    other_warnings = [n for n in warnings if not n["original_cmd"]]
    _add_notes_section(lines, "Warnings", other_warnings)
    _add_notes_section(lines, "Notes", infos)

    # Next steps
    lines.extend(
        [
            "## Next Steps",
            "",
            "1. Create Python functions for each PLACEHOLDER entry",
            "2. Update `pivot.yaml` with correct `python:` paths",
            "3. Run `pivot run --dry-run` to verify configuration",
            "4. Run `pivot run` to execute and build cache",
            "",
            "**Note:** DVC hashes are not migrated. The first `pivot run` will rebuild all caches.",
            "",
        ]
    )

    return "\n".join(lines)
