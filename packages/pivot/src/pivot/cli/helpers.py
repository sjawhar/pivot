from __future__ import annotations

import enum
import importlib
import sys
from typing import TYPE_CHECKING, Any, cast, override

import click
from tqdm.asyncio import tqdm as async_tqdm

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Collection

    from networkx import DiGraph

    from pivot.cli import CliContext
    from pivot.expansion import Expansion
    from pivot.registry import PipelineLike, RegistryStageInfo
    from pivot.storage import track
    from pivot.storage.store import WorkspaceStore
    from pivot.types import StageExplanation

from pivot import display, exceptions, names, project
from pivot.cli import decorators as cli_decorators


class NoPipelineError(exceptions.PivotError):
    """Raised when no Pipeline is available in context."""

    @override
    def format_user_message(self) -> str:
        return (
            "No pipeline definition found.\n"
            "\n"
            "This command requires a pipeline to be defined in one of:\n"
            "  - pipeline.py"
        )


def get_pipeline() -> PipelineLike:
    """Get Pipeline from context, raising NoPipelineError if not found."""
    pipeline = cli_decorators.get_pipeline_from_context()
    if pipeline is None:
        raise NoPipelineError()
    return pipeline


def list_stages() -> list[str]:
    """List stage names from Pipeline in context."""
    return get_pipeline().list_stages()


def get_stage(name: str) -> RegistryStageInfo:
    """Get stage info from Pipeline in context."""
    return get_pipeline().get_stage(name)


def get_all_stages() -> dict[str, RegistryStageInfo]:
    """Get all stages as a dict from Pipeline in context."""
    pipeline = get_pipeline()
    return {name: pipeline.get_stage(name) for name in pipeline.list_stages()}


def build_dag() -> DiGraph[str]:
    """Build DAG from Pipeline in context."""
    return get_pipeline().build_dag()


def get_graph_stages(expanded: Collection[str] = ()) -> dict[str, RegistryStageInfo]:
    """Every stage, and every template (``Pipeline.map``) not in ``expanded``, as a run's graph
    holds them: such a template stands for its instances and reads its whole source directory."""
    pipeline = get_pipeline()
    return {
        **get_all_stages(),
        **{
            name: pipeline.get_template(name)["info"]
            for name in pipeline.list_templates()
            if name not in expanded
        },
    }


def expand_mapped_stages(
    *, force: bool = False, allow_missing: bool = False
) -> tuple[Expansion, dict[str, StageExplanation]]:
    """Expand each mapped stage of the Pipeline in context whose producer is current, as a run
    would (``status.expand_current``), and return the expansion with the explanations it read.
    A command that lists stages calls it first, so the stages it lists hold the instances."""
    from pivot import status as status_mod

    return status_mod.expand_current(get_pipeline(), force=force, allow_missing=allow_missing)


def template_target(name: str, templates: list[str]) -> str | None:
    """The template (``Pipeline.map``) a target names, itself or as an instance
    (``<template>@<key>``), qualified or bare as ``names.resolve_stage_name`` takes a stage
    name; None when it names none, or more than one by a bare name."""
    from pivot import registry

    resolved = names.resolve_stage_name(name, dict.fromkeys(templates))
    if resolved in templates:
        return resolved
    template = registry.template_of(name, templates)
    if template is not None:
        return template
    bare = [t for t in templates if registry.template_of(name, [t.rsplit("/", 1)[-1]])]
    return bare[0] if len(bare) == 1 else None


def mapped_name(name: str, templates: list[str]) -> str | None:
    """The qualified name of the template (``Pipeline.map``) or instance a target names, bare or
    qualified (``template_target``): the template's own name, or ``<template>@<key>``; None when
    it names neither."""
    from pivot import registry

    template = template_target(name, templates)
    if template is None:
        return None
    if names.resolve_stage_name(name, dict.fromkeys(templates)) == template:
        return template
    if registry.template_of(name, [template]) == template:
        return name
    return f"{template}{name[len(template.rsplit('/', 1)[-1]) :]}"


def resolve_run_targets(targets: list[str]) -> list[str]:
    """``targets`` for a run that expands mapped stages (``Pipeline.map``) as it goes,
    ``pivot repro``.

    A name of a template or of one of its instances, bare or qualified, becomes its qualified
    name (``mapped_name``); the run reaches an instance through its template and adds that
    instance alone of the template's, with its chain. An instance is checked as ``pivot status``
    checks it (``resolve_mapped_targets``): when its template expands now, the expansion must
    add it. Any other target is left for the caller to resolve as a stage or a path.

    Raises:
        StageNotFoundError: Naming every instance whose template expands now without it, as
            ``pivot status`` names it.
    """
    from pivot import registry

    pipeline = get_pipeline()
    registered = dict.fromkeys(pipeline.list_stages())
    templates = pipeline.list_templates()
    named = list[str]()
    # Each instance named, by the target naming it
    instances = dict[str, str]()
    for target in targets:
        name = None if target in registered else mapped_name(target, templates)
        named.append(target if name is None else name)
        if name is not None and name not in templates:
            instances[target] = name
    if not instances:
        return named
    result, _ = expand_mapped_stages()
    available = [*pipeline.list_stages(), *templates]
    pipeline.reset_expansions()
    unknown = list[str]()
    for target, name in instances.items():
        template = registry.template_of(name, templates)
        if template in result["expanded"] and name not in result["expanded"][template]:
            unknown.append(target)
    if unknown:
        raise exceptions.StageNotFoundError(unknown, available_stages=available)
    return named


def refuse_template_output_targets(targets: list[str]) -> None:
    """Refuse a target that is an output path a mapped stage (``Pipeline.map``) declares. No
    stage writes it, since each instance writes its own, so it names no stage, whether or not
    the mapped stage has expanded.

    Raises:
        click.ClickException: Naming each such target, its mapped stage and the path each of
            its instances writes instead.
    """
    from pivot import registry
    from pivot.storage import store as store_mod

    pipeline = get_pipeline()
    templates = pipeline.list_templates()
    if not templates:
        return
    store = require_workspace_store()
    declared = store.output_identities_by_path(
        out for name in templates for out in pipeline.get_template(name)["info"]["outs"]
    )
    proj_root = project.get_project_root()
    refusals = list[str]()
    for target in targets:
        path = project.to_relative_path(project.normalize_path(target), proj_root)
        if path not in declared:
            continue
        name = declared[path].producer
        instance = registry.instantiate(pipeline.get_template(name), "<key>", "<key>")
        written = ", ".join(str(store_mod.relative_output_path(out)) for out in instance["outs"])
        refusals.append(
            f"'{target}' is the output path of mapped stage '{name}', which no stage writes: "
            + f"each instance writes its own ({written})"
        )
    if refusals:
        raise click.ClickException("\n".join(refusals))


def mapped_target_names(targets: list[str], result: Expansion) -> list[str]:
    """``targets`` with each name of a template (``Pipeline.map``) replaced after an expansion
    (``expand_mapped_stages``): an expanded template by its instances, a template not expanded,
    or an instance named for one, by the template. Any other target, an instance of an expanded
    template among them, is left for the caller to resolve as a stage or a path."""
    pipeline = get_pipeline()
    registered = dict.fromkeys(pipeline.list_stages())
    templates = pipeline.list_templates()
    named = list[str]()
    for target in targets:
        template = None if target in registered else template_target(target, templates)
        if template is None:
            named.append(target)
        elif template not in result["expanded"]:
            named.append(template)
        elif names.resolve_stage_name(target, dict.fromkeys(templates)) == template:
            named.extend(result["expanded"][template])
        else:
            named.append(target)
    return named


def require_expanded_targets(targets: tuple[str, ...], result: Expansion) -> tuple[str, ...]:
    """``mapped_target_names`` for a command that transfers or restores outputs, which a mapped
    stage not yet expanded has none of.

    Raises:
        click.ClickException: A target names such a mapped stage, saying what it waits for.
    """
    named = mapped_target_names(list(targets), result)
    waiting = dict.fromkeys(name for name in named if name in result["not_expanded"])
    if waiting:
        raise click.ClickException(
            "; ".join(
                f"'{name}' is not yet expanded: {result['not_expanded'][name]}" for name in waiting
            )
        )
    return tuple(named)


def resolve_mapped_targets(
    stages: list[str] | None, result: Expansion
) -> tuple[list[str] | None, list[str] | None]:
    """Resolve stage targets after an expansion (``expand_mapped_stages``).

    A bare or qualified name of a stage is resolved as ``resolve_stage_names`` does. An expanded
    template stands for its instances. A template not expanded, or an instance named for one
    (``<template>@<key>``), stands for its producer, whose run would expand it, and is returned
    beside the stages with every template it is mapped over, to show as not yet expanded.
    Returns (stage targets, templates to show; None for every unexpanded one when no target).

    Raises:
        StageNotFoundError: Naming every target that matches no stage or template, an instance
            of an expanded template that has no such key included.
    """
    if not stages:
        return stages, None
    pipeline = get_pipeline()
    registered = pipeline.list_stages()
    templates = pipeline.list_templates()
    known = dict.fromkeys(registered)
    resolved = list[str]()
    shown = list[str]()
    unknown = list[str]()
    for stage in stages:
        name = names.resolve_stage_name(stage, known)
        if name in known:
            resolved.append(name)
            continue
        template = template_target(stage, templates)
        if template is None:
            unknown.append(stage)
        elif template in result["expanded"]:
            if names.resolve_stage_name(stage, dict.fromkeys(templates)) == template:
                resolved.extend(result["expanded"][template])
            else:
                unknown.append(stage)  # an instance of a key the expansion did not add
        else:
            pending = [template]
            while pending:
                current = pending.pop()
                if current in shown:
                    continue
                shown.append(current)
                mapped = pipeline.get_template(current)
                resolved.append(mapped["source"].producer)
                pending.extend(dep for dep in mapped["mapped_deps"].values() if dep is not None)
    if unknown:
        raise exceptions.StageNotFoundError(unknown, available_stages=[*registered, *templates])
    return list(dict.fromkeys(resolved)), shown


def get_workspace_store() -> WorkspaceStore | None:
    """Get a WorkspaceStore for display/path resolution, or None if no pipeline."""
    store_mod = importlib.import_module("pivot.storage.store")

    pipeline = cli_decorators.get_pipeline_from_context()
    if pipeline is None:
        return None
    return store_mod.WorkspaceStore(
        project_root=project.get_project_root(),
        pipeline_name=pipeline.name,
        input_bindings=pipeline.input_bindings,
    )


def require_workspace_store() -> WorkspaceStore:
    """Get the WorkspaceStore for the Pipeline in context, raising NoPipelineError if none."""
    store = get_workspace_store()
    if store is None:
        raise NoPipelineError()
    return store


def own_member_skip(cache_dir: pathlib.Path) -> track.PvtSkip:
    """The skip that leaves a stage's own ``*.pvt`` member out of what this command discovers
    (``member_skip.own_member_skip``): the loaded pipeline's stages (none without a pipeline)
    decide their outputs, and the directory outputs other stages' locks record decide every
    other. ``cache_dir`` is the cache object root."""
    from pivot import config
    from pivot.storage import member_skip

    pipeline = cli_decorators.get_pipeline_from_context()
    return member_skip.own_member_skip(
        project.get_project_root(),
        get_all_stages() if pipeline is not None else {},
        config.locate_state_dir(),
        cache_dir,
    )


def _json_default(obj: Any) -> Any:  # noqa: ANN401 - json.dumps default requires Any
    """Handle non-standard JSON types in JSONL output."""
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, set):
        # Convert to sorted list for JSON serialization (deterministic output)
        # Cast needed because isinstance(obj, set) doesn't narrow the element type
        return sorted(cast("set[Any]", obj), key=str)
    # Let json.dumps raise TypeError for truly unserializable types
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def emit_jsonl(event: object) -> None:
    """Emit a single JSONL event to stdout with flush for streaming.

    Handles StrEnum values automatically. Sets are converted to sorted lists. A non-finite
    float is written as a string (``display.to_json``). Other non-serializable types will
    raise TypeError.
    """
    print(display.to_json(event, default=_json_default), flush=True)


def get_cli_context(ctx: click.Context) -> CliContext:
    """Get CLI context with defaults if not set."""
    if ctx.obj:
        return ctx.obj
    # Return dict matching CliContext structure to avoid circular import
    return {"verbose": False, "quiet": False}


def stages_to_list(stages: tuple[str, ...]) -> list[str] | None:
    """Convert Click's stage tuple to list or None if empty."""
    return list(stages) if stages else None


def resolve_stage_names(stages: list[str] | None) -> list[str] | None:
    """Resolve bare or pipeline-qualified stage names to registered stage names.

    Raises StageNotFoundError naming every target that matches no registered stage.
    """
    if not stages:
        return stages
    registered = list_stages()
    known = dict.fromkeys(registered)
    resolved = [names.resolve_stage_name(stage, known) for stage in stages]
    unknown = [stage for stage, name in zip(stages, resolved, strict=True) if name not in known]
    if unknown:
        raise exceptions.StageNotFoundError(unknown, available_stages=registered)
    return resolved


class TransferProgress:
    """Context manager for transfer progress bars."""

    _action: str
    _bar: async_tqdm[Any] | None
    _show: bool

    def __init__(self, action: str, *, quiet: bool = False) -> None:
        self._action = action
        self._bar = None
        self._show = sys.stderr.isatty() and not quiet

    def __enter__(self) -> TransferProgress:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        if self._bar is not None:
            self._bar.close()

    def callback(self, completed: int, total: int, filename: str) -> None:
        if not self._show:
            return
        if self._bar is None:
            self._bar = async_tqdm(
                total=total,
                file=sys.stderr,
                dynamic_ncols=True,
                leave=False,
                unit="file",
            )
        self._bar.desc = f"{self._action} {display.escape_control(filename)}"
        self._bar.update(completed - self._bar.n)


def print_transfer_errors(errors: list[str], max_shown: int = 5) -> None:
    """Print transfer errors with truncation for long lists."""
    if not errors:
        return
    for err in errors[:max_shown]:
        click.echo(f"  Error: {display.safe_message(err)}", err=True)
    if len(errors) > max_shown:
        click.echo(f"  ... and {len(errors) - max_shown} more errors", err=True)
