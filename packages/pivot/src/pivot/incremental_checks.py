"""The checks a run makes of its incremental outputs before it starts stages.

A run checks its stages before it starts (``engine``) and each mapped stage's instances before
it adds them (``expansion.plan_instances``), and ``pivot repro --dry-run`` checks the same
(``cli.helpers.eligible_templates``). Its importers, ``engine/engine.py``, ``expansion.py`` and
``cli/helpers.py``, import it at their top; it imports none of them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pivot import config, exceptions, project, registry, types
from pivot.storage import lock
from pivot.storage import store as store_mod

if TYPE_CHECKING:
    from pivot.registry import RegistryStageInfo


def check_uncached_incremental_outputs(
    execution_order: list[str],
    all_stages: dict[str, RegistryStageInfo],
) -> list[tuple[str, str]]:
    """Check for incremental artifacts that exist without a recorded cache hash."""
    uncached = list[tuple[str, str]]()
    default_state_dir = config.locate_state_dir()
    workspace = store_mod.WorkspaceStore(project.get_project_root(), "", {})
    cache_dir = config.locate_cache_dir() / "files"

    for stage_name in execution_order:
        stage_info = all_stages[stage_name]
        incremental = [
            out for out in stage_info["outs"] if out.tag is types.ArtifactTag.INCREMENTAL
        ]
        if not incremental:
            continue
        stage_state_dir = registry.get_stage_state_dir(stage_info, default_state_dir)
        stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(stage_state_dir))
        lock_data = stage_lock.read(cache_dir=cache_dir)
        output_hashes = lock_data["output_hashes"] if lock_data else {}

        for out in incremental:
            out_path = workspace.resolve_display_path(out)
            if out_path.exists() and out.identity not in output_hashes:
                uncached.append((stage_name, str(out_path)))

    return uncached


def check_restarted_incremental_outputs(
    execution_order: list[str],
    all_stages: dict[str, RegistryStageInfo],
) -> list[str]:
    """Why each stage in ``execution_order`` whose lock an older pivot rewrote over this
    pivot's, for a stage with an incremental output, is refused (``lock.restart_proofs``)."""
    refused = list[str]()
    default_state_dir = config.locate_state_dir()
    cache_dir = config.locate_cache_dir() / "files"
    for stage_name in execution_order:
        stage_info = all_stages[stage_name]
        if not any(out.tag is types.ArtifactTag.INCREMENTAL for out in stage_info["outs"]):
            continue
        stage_state_dir = registry.get_stage_state_dir(stage_info, default_state_dir)
        stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(stage_state_dir))
        restarted = lock.restart_proofs(
            stage_lock, stage_lock.read(cache_dir=cache_dir), stage_info["outs"]
        )
        if restarted:
            refused.append(f"{stage_name}: {lock.restarted_reason(stage_lock, restarted)}")
    return refused


def incremental_refusal(
    stage_order: list[str],
    stages: dict[str, RegistryStageInfo],
    *,
    allow_uncached_incremental: bool,
    force: bool,
) -> exceptions.PivotError | None:
    """Why a run must not start ``stage_order``, or None: an incremental output on disk that
    the cache does not hold, which running would delete (unless ``allow_uncached_incremental``),
    or an incremental stage whose lock an older pivot rewrote over this pivot's, as that pivot may
    have restarted the output from empty (unless ``force``, which accepts its lock on purpose).
    A run checks its stages before it starts and each mapped stage's instances before it adds
    them; ``pivot repro --dry-run`` checks the same."""
    if not allow_uncached_incremental:
        uncached = check_uncached_incremental_outputs(stage_order, stages)
        if uncached:
            files_list = "\n".join(f"  - {stage}: {path}" for stage, path in uncached)
            return exceptions.UncachedIncrementalOutputError(
                "The following incremental output files exist but are not in cache:\n"
                + f"{files_list}\n\n"
                + "Running the pipeline will DELETE these files and they cannot be restored.\n"
                + "To proceed anyway, use --allow-uncached-incremental or back up these "
                + "files first."
            )
    if not force:
        restarted = check_restarted_incremental_outputs(stage_order, stages)
        if restarted:
            return exceptions.RestartedIncrementalOutputError(
                "Pivot will not run these stages:\n"
                + "\n".join(f"  - {reason}" for reason in restarted)
            )
    return None
