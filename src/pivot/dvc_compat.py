from __future__ import annotations

import dataclasses
import logging
import pathlib
import subprocess
from typing import TYPE_CHECKING, Any

import yaml

from pivot import outputs, project, registry
from pivot.exceptions import DVCImportError, ExportError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from inspect import Signature

    from dvc.output import Output as DVCOutput
    from dvc.stage import PipelineStage

    from pivot.registry import RegistryStageInfo


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class StageSpec:
    """Parsed DVC stage specification."""

    name: str
    cmd: str | list[str]  # DVC supports multi-command stages as lists
    deps: list[str]
    outs: list[outputs.BaseOut]
    params: dict[str, Any]
    frozen: bool = False
    desc: str | None = None
    cwd: pathlib.Path | None = None  # Working directory for command execution


def _to_relative_path(absolute_path: str, root: pathlib.Path) -> str:
    """Convert absolute path to relative (from project root)."""
    path = pathlib.Path(absolute_path)
    if not path.is_absolute():
        return absolute_path

    try:
        return str(path.relative_to(root))
    except ValueError:
        # Path outside project root - keep absolute with warning
        logger.warning(f"Path '{absolute_path}' is outside project root, keeping absolute")
        return absolute_path


def _generate_cmd(func: Callable[..., Any]) -> str:
    """Generate DVC command from function (module import approach)."""
    module = func.__module__
    name = func.__name__

    if module == "__main__":
        raise ExportError(
            f"Cannot export function '{name}' from __main__ module. Move the function to an importable module (e.g., pipeline.py)."
        )

    if name == "<lambda>":
        raise ExportError("Cannot export lambda functions - they have no importable name.")

    return f"python -c 'from {module} import {name}; {name}()'"


def _extract_param_defaults(sig: Signature) -> dict[str, Any]:
    """Extract parameter defaults from function signature."""
    params = dict[str, Any]()
    for param_name, param in sig.parameters.items():
        if param.default is not param.empty:
            params[param_name] = param.default
    return params


def _generate_params_yaml(
    stages: dict[str, RegistryStageInfo],
    path: pathlib.Path,
) -> dict[str, Any]:
    """Generate params.yaml from Pydantic defaults or function signature defaults."""
    all_params: dict[str, Any] = {}

    for name, info in stages.items():
        params_instance = info["params"]
        if params_instance is not None:
            # Use stored Pydantic model instance
            stage_params = params_instance.model_dump()
        else:
            # Fall back to signature defaults
            sig = info["signature"]
            if not sig:
                continue
            stage_params = _extract_param_defaults(sig)

        if stage_params:
            all_params[name] = stage_params

    if all_params:
        params_path = path.parent / "params.yaml"
        try:
            with open(params_path, "w") as f:
                yaml.dump(all_params, f, sort_keys=False, default_flow_style=False)
        except yaml.YAMLError as e:
            raise ExportError(
                f"Failed to serialize params.yaml: {e}. Parameter defaults must be "
                + "YAML-serializable (str, int, float, bool, list, dict)."
            ) from e
        except OSError as e:
            raise ExportError(f"Failed to write params.yaml to '{params_path}': {e}") from e
        logger.info(f"Generated params.yaml at {params_path}")

    return all_params


def _build_dvc_stage(
    stage_info: RegistryStageInfo,
    root: pathlib.Path,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Build DVC stage dict from Pivot stage decorator."""
    stage: dict[str, Any] = {
        "cmd": _generate_cmd(stage_info["func"]),
    }

    # Add deps section
    deps = [_to_relative_path(dep, root) for dep in stage_info["deps"]]
    if deps:
        stage["deps"] = deps

    # Build outs/metrics/plots sections from BaseOut objects
    outs_section: list[str | dict[str, Any]] = []
    metrics_section: list[str | dict[str, Any]] = []
    plots_section: list[str | dict[str, Any]] = []

    for out in stage_info["outs"]:
        rel_path = _to_relative_path(out.path, root)
        out_entry = _build_out_entry(out, rel_path)

        if isinstance(out, outputs.Plot):
            plots_section.append(out_entry)
        elif isinstance(out, outputs.Metric):
            metrics_section.append(out_entry)
        else:
            # Out or BaseOut → outs section
            outs_section.append(out_entry)

    if outs_section:
        stage["outs"] = outs_section
    if metrics_section:
        stage["metrics"] = metrics_section
    if plots_section:
        stage["plots"] = plots_section

    # Add params reference if stage has parameter defaults
    stage_name = stage_info["name"]
    if stage_name in params:
        stage["params"] = [f"{stage_name}.{p}" for p in params[stage_name]]

    return stage


def _build_out_entry(out: outputs.BaseOut, rel_path: str) -> str | dict[str, Any]:
    """Build DVC output entry from BaseOut object."""
    options: dict[str, Any] = {}

    # Only emit cache option when it differs from type default (Metric=False, others=True)
    default_cache = not isinstance(out, outputs.Metric)
    if out.cache != default_cache:
        options["cache"] = out.cache

    # IncrementalOut always exports with persist: true (DVC won't delete it between runs)
    if isinstance(out, outputs.IncrementalOut) or out.persist:
        options["persist"] = True

    # Plot-specific options
    if isinstance(out, outputs.Plot):
        for attr in ("x", "y", "template"):
            if (value := getattr(out, attr)) is not None:
                options[attr] = value

    return {rel_path: options} if options else rel_path


def export_dvc_yaml(
    path: pathlib.Path | str,
    stages: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Export Pivot stages to dvc.yaml format.

    Generated cmd uses module import: python -c 'from module import func; func()'
    This creates a standalone dvc.yaml that works without Pivot.

    Note: Also generates params.yaml with function parameter defaults. This
    overwrites any existing params.yaml file (does not merge).

    Args:
        path: Output path for dvc.yaml
        stages: Optional list of stage names to export (default: all)

    Returns:
        The generated dvc.yaml dict

    Raises:
        ExportError: If export fails or requested stages don't exist
    """
    path = pathlib.Path(path)
    root = project.get_project_root()

    # Validate requested stages exist
    if stages is None:
        stages = registry.REGISTRY.list_stages()
    else:
        all_stages = {*registry.REGISTRY.list_stages()}
        missing = [name for name in stages if name not in all_stages]
        if missing:
            raise ExportError(f"Stages not found: {missing}")

    stage_dict = {name: registry.REGISTRY.get(name) for name in stages}
    if not stage_dict:
        raise ExportError("No stages registered to export")

    # Generate params.yaml first (need params for stage references)
    params = _generate_params_yaml(stage_dict, path)

    # Build dvc.yaml structure
    dvc_stages: dict[str, Any] = {}
    for name, info in stage_dict.items():
        dvc_stages[name] = _build_dvc_stage(info, root, params)

    dvc_yaml = {"stages": dvc_stages}

    try:
        with open(path, "w") as f:
            yaml.dump(dvc_yaml, f, sort_keys=False, default_flow_style=False)
    except yaml.YAMLError as e:
        raise ExportError(f"Failed to serialize dvc.yaml: {e}") from e
    except OSError as e:
        raise ExportError(f"Failed to write dvc.yaml to '{path}': {e}") from e

    logger.info(f"Exported {len(dvc_stages)} stages to {path}")
    return dvc_yaml


def import_dvc_yaml(
    path: pathlib.Path | str,
    register: bool = True,
) -> dict[str, StageSpec]:
    """Parse dvc.yaml using DVC and optionally register stages with Pivot.

    Uses DVC's own resolver to handle foreach, matrix, vars, interpolation.

    Args:
        path: Path to dvc.yaml
        register: If True, auto-register stages (default). If False, just return specs.

    Returns:
        Dict of stage_name -> StageSpec

    Raises:
        DVCImportError: If DVC not installed or parsing fails
    """
    try:
        import dvc.repo
        from dvc.stage import PipelineStage as DVCPipelineStage
    except ImportError:
        raise DVCImportError("DVC is required for import_dvc_yaml") from None

    path = pathlib.Path(path).resolve()
    if not path.exists():
        raise DVCImportError(f"dvc.yaml not found: {path}")

    # DVC integration requires real DVC repository
    try:  # pragma: no cover
        repo = dvc.repo.Repo(str(path.parent))
        dvc_stages = repo.index.stages
    except Exception as e:  # pragma: no cover
        raise DVCImportError(f"Failed to parse dvc.yaml: {e}") from e

    specs: dict[str, StageSpec] = {}
    for stage in dvc_stages:  # pragma: no cover
        # Only process pipeline stages (not data stages, etc.)
        if not isinstance(stage, DVCPipelineStage):
            continue
        if not stage.name or not stage.cmd:
            continue
        # Filter to stages from the target dvc.yaml file
        if pathlib.Path(stage.path).resolve() != path:
            continue

        spec = StageSpec(
            name=stage.name,
            cmd=stage.cmd,
            deps=[str(d.fs_path) for d in stage.deps],
            outs=_convert_dvc_outputs(stage),
            params=_extract_dvc_params(stage),
            frozen=stage.frozen,
            desc=stage.desc,
            cwd=path.parent,  # Run command from dvc.yaml directory
        )
        specs[stage.name] = spec

        if register:
            _register_imported_stage(spec)

    logger.info(f"Imported {len(specs)} stages from {path}")  # pragma: no cover
    return specs  # pragma: no cover


def _convert_dvc_output(out: DVCOutput) -> outputs.BaseOut:  # pragma: no cover
    """Convert a single DVC output to BaseOut object."""
    path = str(out.fs_path)
    cache = out.use_cache
    persist = out.persist

    if out.plot:
        return outputs.Plot(
            path=path,
            cache=cache,
            persist=persist,
            x=out.plot.get("x") if isinstance(out.plot, dict) else None,
            y=out.plot.get("y") if isinstance(out.plot, dict) else None,
            template=out.plot.get("template") if isinstance(out.plot, dict) else None,
        )
    if out.metric:
        return outputs.Metric(path=path, cache=cache, persist=persist)
    return outputs.Out(path=path, cache=cache, persist=persist)


def _convert_dvc_outputs(stage: PipelineStage) -> list[outputs.BaseOut]:  # pragma: no cover
    """Convert DVC stage outputs to BaseOut objects."""
    return [_convert_dvc_output(out) for out in stage.outs]


def _extract_dvc_params(stage: PipelineStage) -> dict[str, Any]:  # pragma: no cover
    """Extract params from DVC stage."""
    params: dict[str, Any] = {}
    for param_dep in stage.params:
        if param_dep.hash_info:
            value = param_dep.hash_info.value
            if isinstance(value, dict):
                params.update(value)
    return params


def _register_imported_stage(spec: StageSpec) -> None:  # pragma: no cover
    """Register an imported DVC stage with Pivot."""
    _register_imported_stage_with_name(spec, spec.name)


def _register_imported_stage_with_name(spec: StageSpec, name: str) -> None:  # pragma: no cover
    """Register imported DVC stage with explicit name (for collision handling)."""
    # Use spec.cwd if set, otherwise fall back to project root
    cwd = spec.cwd if spec.cwd else project.get_project_root()

    # DVC supports multi-command stages as lists - run each with cwd reset
    # (check=True ensures we stop on first failure, like && would)
    commands = spec.cmd if isinstance(spec.cmd, list) else [spec.cmd]

    def wrapper() -> None:
        for cmd in commands:
            subprocess.run(cmd, shell=True, check=True, cwd=cwd)  # noqa: S602

    wrapper.__name__ = name
    wrapper.__module__ = "pivot.dvc_compat"

    registry.REGISTRY.register(
        wrapper,
        name=name,
        deps=list(spec.deps),
        outs=spec.outs,
    )
