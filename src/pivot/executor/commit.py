"""Commit current workspace state for pipeline stages.

Computes hashes from current disk state, writes lock files, caches outputs,
and updates StateDB. Used by `pivot commit` to record state after running
stages with --no-commit or making manual edits.
"""

from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING, cast

from pivot import config, exceptions, parameters, run_history
from pivot.executor import worker
from pivot.storage import cache, lock
from pivot.storage import state as state_mod
from pivot.types import DeferredWrites, DepEntry, HashInfo, LockData

if TYPE_CHECKING:
    from pivot import registry

logger = logging.getLogger(__name__)


def _get_registry() -> registry.StageRegistry:
    """Get StageRegistry via CLI helpers (lazy import to avoid circular imports)."""
    from pivot.cli import helpers as cli_helpers

    return cli_helpers.get_registry()


def commit_stages(
    stage_names: list[str] | None = None,
    force: bool = False,
) -> tuple[list[str], list[str]]:
    """Commit current workspace state for stages.

    Hashes current deps and outputs on disk, writes lock files, caches outputs,
    and updates StateDB (dep generations, output generations, run cache).

    Args:
        stage_names: Specific stages to commit. None means all stale stages.
        force: If True, commit even if lock file is unchanged.

    Returns:
        Tuple of (committed, failed) stage name lists.
    """
    stage_registry = _get_registry()
    all_stage_names = stage_registry.list_stages()

    # Resolve target stages
    if stage_names is not None:
        registered = set(all_stage_names)
        unknown = [s for s in stage_names if s not in registered]
        if unknown:
            raise exceptions.StageNotFoundError(unknown, available_stages=all_stage_names)
        targets = stage_names
    else:
        targets = all_stage_names

    state_dir = config.get_state_dir()
    state_db_path = config.get_state_db_path()
    cache_dir = config.get_cache_dir()
    files_cache_dir = cache_dir / "files"
    checkout_modes = config.get_checkout_mode_order()
    stages_dir = lock.get_stages_dir(state_dir)

    committed = list[str]()
    failed = list[str]()
    overrides = parameters.load_params_yaml()

    for stage_name in targets:
        stage_info = stage_registry.get(stage_name)

        # 1. Get fingerprint
        fingerprint = stage_registry.ensure_fingerprint(stage_name)

        # 2. Get effective params
        current_params = parameters.get_effective_params(
            stage_info["params"], stage_name, overrides
        )

        # 3. Hash deps
        dep_hashes, missing, unreadable = worker.hash_dependencies(stage_info["deps_paths"])
        if missing:
            logger.error("Stage '%s': missing deps: %s — skipping", stage_name, ", ".join(missing))
            failed.append(stage_name)
            continue
        if unreadable:
            logger.error(
                "Stage '%s': unreadable deps: %s — skipping", stage_name, ", ".join(unreadable)
            )
            failed.append(stage_name)
            continue

        # 4. Compute input_hash
        stage_outs = stage_info["outs"]
        out_specs = [(worker.normalize_out_path(str(out.path)), out.cache) for out in stage_outs]
        deps_list = [
            DepEntry(path=dep_path, hash=info["hash"]) for dep_path, info in dep_hashes.items()
        ]
        input_hash = run_history.compute_input_hash(
            fingerprint, current_params, deps_list, out_specs
        )

        # Compute normalized output paths once (used for skip check, lock data, and StateDB)
        out_paths = [worker.normalize_out_path(str(out.path)) for out in stage_outs]
        production_lock = lock.StageLock(stage_name, stages_dir)

        # 5. If not force and no explicit stages, check lock — skip if unchanged
        if not force and stage_names is None:
            lock_data = production_lock.read()
            if lock_data is not None:
                changed, _ = production_lock.is_changed_with_lock_data(
                    lock_data, fingerprint, current_params, dep_hashes, out_paths
                )
                if not changed:
                    continue

        # 6. Hash and cache outputs
        output_hashes = dict[str, HashInfo]()
        outputs_missing = False

        for out in stage_outs:
            out_path = pathlib.Path(cast("str", out.path))
            if not out_path.exists():
                logger.error("Stage '%s': output missing: %s — skipping", stage_name, out.path)
                outputs_missing = True
                break

            if out.cache:
                output_hashes[str(out.path)] = cache.save_to_cache(
                    out_path, files_cache_dir, checkout_modes=checkout_modes
                )
            else:
                output_hashes[str(out.path)] = worker.hash_output(out_path)

        if outputs_missing:
            failed.append(stage_name)
            continue

        # 7. Write production lock
        new_lock_data = LockData(
            code_manifest=fingerprint,
            params=current_params,
            dep_hashes=dict(sorted(dep_hashes.items())),
            output_hashes=dict(sorted(output_hashes.items())),
            dep_generations={},
        )
        production_lock.write(new_lock_data)

        # 8. Update StateDB: dep generations, output generations, run cache entry
        run_id = run_history.generate_run_id()

        with state_mod.StateDB(state_db_path) as state_db:
            dep_gen_map = worker.compute_dep_generation_map(stage_info["deps_paths"], state_db)

            # Only cached outputs belong in run cache
            cached_paths = {cast("str", out.path) for out in stage_outs if out.cache}
            cached_output_hashes = {
                path: oh for path, oh in output_hashes.items() if path in cached_paths
            }

            output_entries = [
                run_history.output_hash_to_entry(path, oh)
                for path, oh in cached_output_hashes.items()
            ]

            deferred = DeferredWrites()
            if dep_gen_map:
                deferred["dep_generations"] = dep_gen_map
            if output_entries:
                deferred["run_cache_input_hash"] = input_hash
                deferred["run_cache_entry"] = run_history.RunCacheEntry(
                    run_id=run_id,
                    output_hashes=output_entries,
                )

            state_db.apply_deferred_writes(stage_name, out_paths, deferred)

        committed.append(stage_name)

    return committed, failed
