from __future__ import annotations

import pathlib

from pivot import project, run_history
from pivot.executor import worker
from pivot.storage import lock, state

# Sentinel run_id for committed entries - never pruned by prune_run_cache
COMMITTED_RUN_ID = "__committed__"


def commit_pending(cache_dir: pathlib.Path | None = None) -> list[str]:
    """Promote pending locks to production and update StateDB.

    IMPORTANT: Caller must hold pending_state_lock to prevent races with other
    processes that may be executing stages or committing concurrently.

    Returns list of stage names that were committed.
    """
    project_root = project.get_project_root()
    if cache_dir is None:
        cache_dir = project_root / ".pivot" / "cache"

    pending_stages = lock.list_pending_stages(project_root)
    if not pending_stages:
        return []

    state_db_path = project_root / ".pivot" / "state.db"
    committed = list[str]()

    with state.StateDB(state_db_path) as state_db:
        for stage_name in pending_stages:
            pending_lock = lock.get_pending_lock(stage_name, project_root)
            pending_data = pending_lock.read()

            if pending_data is None:
                continue

            # Write to production lock (without dep_generations - that's internal to pending)
            production_lock = lock.StageLock(stage_name, lock.get_stages_dir(cache_dir))
            production_lock.write(pending_data)

            # Record dependency generations from execution time (not commit time!)
            dep_gens = pending_data["dep_generations"]
            if dep_gens:
                state_db.record_dep_generations(stage_name, dep_gens)

            # Increment output generations
            for out_path in pending_data["output_hashes"]:
                state_db.increment_generation(pathlib.Path(out_path))

            # Write run cache entry with sentinel run_id (never pruned)
            out_paths = list(pending_data["output_hashes"].keys())
            input_hash = run_history.compute_input_hash_from_lock(pending_data, out_paths)
            worker.write_run_cache_entry(
                stage_name, input_hash, pending_data["output_hashes"], COMMITTED_RUN_ID, state_db
            )

            # Remove pending lock
            pending_lock.path.unlink(missing_ok=True)
            committed.append(stage_name)

    return committed


def discard_pending() -> list[str]:
    """Discard all pending locks without committing.

    IMPORTANT: Caller must hold pending_state_lock to prevent races with other
    processes that may be executing stages or committing concurrently.

    Returns list of stage names that were discarded.
    """
    project_root = project.get_project_root()
    pending_stages = lock.list_pending_stages(project_root)

    discarded = list[str]()
    for stage_name in pending_stages:
        pending_lock = lock.get_pending_lock(stage_name, project_root)
        pending_lock.path.unlink(missing_ok=True)
        discarded.append(stage_name)

    return discarded
