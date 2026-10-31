"""The stage names the RPC queries take, and what ``stage_info`` and ``explain`` answer.

A mapped stage (``Pipeline.map``) is a template, not a stage: a run expands it into one instance
per file of its producer's directory, ``<template>@<key>``. A run, ``stage_info`` and
``explain`` take a template's name, and an instance's before a run added it. ``stage_info`` of a
template gives the stage as declared; ``explain`` of an expanded template explains its
instances. A template or an instance not expanded is answered with what it waits for: the
failure of the run that could not expand it, the run's targets that left it out, or the producer
a run must find current. The engine keeps what its running or latest run read and failed
(``Engine.expansion_keys``, ``Engine.expansion_failure``).

``explain`` reads lock files in a worker thread, but takes the registry's stages and the
params.yaml overrides on the event loop first: a run start removes every instance there
(``reset_expansions``), so a thread reading the registry could find an instance gone.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import anyio.to_thread

from pivot import exceptions, parameters, project, types
from pivot import explain as explain_mod
from pivot import registry as registry_mod
from pivot.config import io as config_io
from pivot.storage import state as state_mod
from pivot.storage import store as store_mod
from pivot.types import StageExplanation

if TYPE_CHECKING:
    from pathlib import Path

    from pivot.engine.engine import Engine
    from pivot.registry import PipelineLike, RegistryStageInfo

__all__ = ["explain", "stage_info", "unknown_run_stages"]


def unknown_run_stages(pipeline: PipelineLike, stages: list[str]) -> list[str]:
    """Each of ``stages`` a run cannot take. A run takes a stage, a template, and an instance
    of a template, which it reaches through the template that adds it when the run expands it.
    """
    templates = pipeline.list_templates()
    available = {*pipeline.list_stages(), *templates}
    return [
        stage
        for stage in stages
        if stage not in available and registry_mod.template_of(stage, templates) is None
    ]


def stage_info(engine: Engine, pipeline: PipelineLike, stage: str) -> RegistryStageInfo:
    """``stage``'s registry entry. A template's is the stage as declared, reading its
    producer's whole directory.

    Raises:
        ValueError: ``stage`` names no stage or template, or an instance its template has not
            added (``_not_expanded_reason``).
    """
    with contextlib.suppress(KeyError):
        return pipeline.get_stage(stage)
    if stage in pipeline.list_templates():
        return pipeline.get_template(stage)["info"]
    why = _not_expanded_reason(engine, pipeline, stage)
    raise ValueError(f"'{stage}' is not yet expanded: {why}")


async def explain(engine: Engine, pipeline: PipelineLike, stage: str) -> StageExplanation:
    """Why ``stage`` would run, as ``pivot status --explain`` says it. An expanded template is
    explained by the instances it expanded into; a template not expanded, or an instance its
    template has not added, by what it waits for.

    Raises:
        ValueError: ``stage`` names no stage, template or instance (``_not_expanded_reason``),
            or params.yaml cannot be loaded.
    """
    stages = list[str]()
    with contextlib.suppress(KeyError):
        _ = pipeline.get_stage(stage)
        stages = [stage]
    if not stages and stage in pipeline.list_templates():
        stages = pipeline.instances_of(stage)
    if not stages:
        return _not_expanded_explanation(stage, _not_expanded_reason(engine, pipeline, stage))
    fingerprints = {name: pipeline.ensure_fingerprint(name) for name in stages}
    # Read on the event loop, where a run start removes the instances (module docstring)
    all_stages = {name: pipeline.get_stage(name) for name in pipeline.list_stages()}
    try:
        overrides = parameters.load_params_yaml(pipeline)
    except Exception as e:
        raise ValueError(f"Failed to load params.yaml: {e}") from e
    pipeline_name = pipeline.name
    input_bindings = pipeline.input_bindings

    def _explain_in_thread() -> StageExplanation:
        default_state_dir = config_io.locate_state_dir()
        cache_dir = config_io.locate_cache_dir() / "files"
        store = store_mod.WorkspaceStore(
            project_root=project.get_project_root(),
            pipeline_name=pipeline_name,
            input_bindings=input_bindings,
            cache_dir=cache_dir,
        )
        explanations = [
            _explain_stage(
                name,
                fingerprints[name],
                overrides,
                all_stages,
                store,
                default_state_dir=default_state_dir,
                cache_dir=cache_dir,
            )
            for name in stages
        ]
        if stages == [stage]:
            return explanations[0]
        return _template_explanation(stage, explanations)

    return await anyio.to_thread.run_sync(_explain_in_thread)


def _not_expanded_reason(engine: Engine, pipeline: PipelineLike, stage: str) -> str:
    """What ``stage`` waits for, naming no registered stage: a template (``Pipeline.map``) not
    expanded, or an instance its template has not added. The failure of the running or latest
    run that could not expand the template, or a template it is mapped over, comes first; then,
    for an instance, that the run's targets left it out; else that a run must find the producer
    current.

    Raises:
        ValueError: ``stage`` names no stage, template or instance. An instance whose key no
            file has, of those the run read to expand its template (``Engine.expansion_keys``),
            is refused as ``pivot status`` refuses its name.
    """
    templates = pipeline.list_templates()
    template = stage if stage in templates else registry_mod.template_of(stage, templates)
    if template is None:
        raise ValueError(f"Unknown stage: {stage}")
    if template != stage:
        keys = engine.expansion_keys(template)
        if keys is not None and stage[len(template) + 1 :] not in keys:
            unknown = exceptions.StageNotFoundError(
                [stage], available_stages=[*pipeline.list_stages(), *templates]
            )
            raise ValueError(unknown.format_user_message())
    failed = _failed_template(engine, pipeline, template)
    if failed is not None:
        failure = f"failed in the latest run: {engine.expansion_failure(failed)}"
        if failed == template:
            return f"'{template}' {failure}"
        return f"'{template}' is mapped over '{failed}', which {failure}"
    if template != stage and pipeline.instances_of(template):
        return f"'{template}' added only the instances its run's targets name"
    producer = pipeline.get_template(template)["source"].producer
    return f"'{template}' adds its instances when a run finds '{producer}' current"


def _failed_template(engine: Engine, pipeline: PipelineLike, template: str) -> str | None:
    """``template``, or the nearest template it is mapped over, that the running or latest run
    failed (``Engine.expansion_failure``); None when the run failed none of them."""
    pending = [template]
    seen = set[str]()
    while pending:
        name = pending.pop(0)
        if name in seen:
            continue
        seen.add(name)
        if engine.expansion_failure(name) is not None:
            return name
        mapped_deps = pipeline.get_template(name)["mapped_deps"].values()
        pending.extend(upstream for upstream in mapped_deps if upstream is not None)
    return None


def _not_expanded_explanation(stage: str, why: str) -> StageExplanation:
    """The explanation of a template or instance not yet expanded, as ``pivot status`` gives it:
    a run would expand it, then run its instances."""
    from pivot import status as status_mod

    return StageExplanation(
        stage_name=stage,
        will_run=True,
        is_forced=False,
        reason=f"{status_mod.NOT_EXPANDED_REASON_PREFIX}: {why}",
        code_changes=[],
        param_changes=[],
        dep_changes=[],
        upstream_stale=[],
    )


def _template_explanation(template: str, explanations: list[StageExplanation]) -> StageExplanation:
    """A template's explanation from its instances': it would run when one would, and its reason
    names each of those with its own reason."""
    running = [exp for exp in explanations if exp["will_run"]]
    noun = "stage" if len(explanations) == 1 else "stages"
    reason = f"Expanded into {len(explanations)} {noun}; " + (
        "would run: " + ", ".join(f"{exp['stage_name']} ({exp['reason']})" for exp in running)
        if running
        else "none would run"
    )
    return StageExplanation(
        stage_name=template,
        will_run=bool(running),
        is_forced=any(exp["is_forced"] for exp in running),
        reason=reason,
        code_changes=[],
        param_changes=[],
        dep_changes=[],
        upstream_stale=[],
    )


def _explain_stage(
    stage: str,
    fingerprint: dict[str, str],
    overrides: parameters.ParamsOverrides,
    all_stages: dict[str, RegistryStageInfo],
    store: store_mod.WorkspaceStore,
    *,
    default_state_dir: Path,
    cache_dir: Path,
) -> StageExplanation:
    """Explain why ``stage`` would run, reading its lock and its producers' as ``pivot status``
    does, and save the hash of each file it had to read."""
    reg_info = all_stages[stage]
    producers = {
        ref.identity.producer
        for ref in reg_info["deps"].values()
        if ref.identity.producer in all_stages
    }
    locks = explain_mod.read_locks([stage, *producers], all_stages, default_state_dir, cache_dir)
    unreadable = dict[types.ArtifactIdentity, str]()
    producer_hashes = explain_mod.producer_output_hashes(
        {producer: locks[producer] for producer in producers},
        reg_info["deps"].values(),
        {
            producer: explain_mod.stages_dir(all_stages[producer], default_state_dir)
            for producer in producers
        },
        store.cache_dir,
        unreadable,
    )
    state_dir = registry_mod.get_stage_state_dir(reg_info, default_state_dir)
    file_hash_entries = list[types.FileHashEntry]()
    explanation = explain_mod.get_stage_explanation(
        stage_name=stage,
        fingerprint=fingerprint,
        deps=reg_info["deps"],
        outs=list(reg_info["outs"]),
        params_instance=reg_info["params"],
        overrides=overrides,
        state_dir=state_dir,
        store=store,
        lock_data=locks[stage],
        allow_missing=True,
        producer_hashes=producer_hashes,
        file_hash_entries=file_hash_entries,
        unreadable_listings=unreadable,
    )
    state_mod.save_file_hashes(state_dir, file_hash_entries, create=False)
    return explanation
