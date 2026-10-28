"""Commit current workspace state for pipeline stages.

Computes hashes from current disk state, writes lock files, caches outputs,
and updates StateDB. Used by `pivot commit` to record state after running
stages with --no-commit or making manual edits.
"""

from __future__ import annotations

import logging
import pathlib

from pivot import config, exceptions, parameters, project, registry, run_history, skip, types
from pivot import state_dir as state_dir_mod
from pivot.executor import worker
from pivot.storage import cache, lock
from pivot.storage import state as state_mod
from pivot.storage import store as store_mod
from pivot.types import ArtifactTag, DeferredWrites, HashInfo, LockData

logger = logging.getLogger(__name__)


def commit_stages(
    pipeline: registry.PipelineLike,
    stage_names: list[str] | None = None,
    force: bool = False,
    *,
    warn_left_out_symlinks: bool = True,
) -> tuple[list[str], list[str]]:
    """Commit current workspace state for stages.

    Hashes current deps and outputs on disk, writes lock files, caches outputs,
    and updates StateDB (dep generations, output generations, run cache).

    Note: Commits are NOT atomic across stages. If a later stage fails, earlier
    stages' lock files and cache entries remain written.

    Args:
        pipeline: The pipeline whose stages are committed.
        stage_names: Specific stages to commit. None means all stale stages.
        force: If True, commit even if lock file is unchanged.
        warn_left_out_symlinks: Log the warning naming symlinks a directory output leaves
            out. A quiet command turns it off.

    Returns:
        Tuple of (committed, failed) stage name lists.

    A library caller's commit holds one state-directory record, started with the directories
    the pipeline resolved when it loaded (``state_dir.entry_invocation``), so its locks, its
    database writes and its fingerprints use the directory its stages' locks are in.
    """
    with state_dir_mod.entry_invocation(pipeline.resolved_state_dirs):
        return _commit_stages(
            pipeline, stage_names, force, warn_left_out_symlinks=warn_left_out_symlinks
        )


def _commit_stages(
    pipeline: registry.PipelineLike,
    stage_names: list[str] | None,
    force: bool,
    *,
    warn_left_out_symlinks: bool,
) -> tuple[list[str], list[str]]:
    all_stage_names = pipeline.list_stages()

    # Resolve target stages
    if stage_names is not None:
        registered = set(all_stage_names)
        unknown = [s for s in stage_names if s not in registered]
        if unknown:
            raise exceptions.StageNotFoundError(unknown, available_stages=all_stage_names)
        targets = stage_names
    else:
        targets = all_stage_names

    default_state_dir = config.get_state_dir()
    project_root = project.get_project_root()
    store = store_mod.WorkspaceStore(
        project_root=project_root,
        pipeline_name=pipeline.name,
        input_bindings=pipeline.input_bindings,
        cache_dir=config.get_cache_dir() / "files",
        checkout_modes=config.get_checkout_mode_order(),
    )

    committed = list[str]()
    failed = list[str]()
    overrides = parameters.load_params_yaml(pipeline)

    state_dbs = dict[pathlib.Path, state_mod.StateDB]()

    def _get_state_db(stage_state_dir: pathlib.Path) -> state_mod.StateDB:
        if stage_state_dir not in state_dbs:
            state_dbs[stage_state_dir] = state_mod.StateDB(stage_state_dir)
        return state_dbs[stage_state_dir]

    try:
        for stage_name in targets:
            stage_info = pipeline.get_stage(stage_name)
            stage_state_dir = registry.get_stage_state_dir(stage_info, default_state_dir)
            stage_db = _get_state_db(stage_state_dir)
            stages_dir = lock.get_stages_dir(stage_state_dir)

            # 1. Get fingerprint
            fingerprint = pipeline.ensure_fingerprint(stage_name)

            # 2. Get effective params
            current_params = parameters.get_effective_params(
                stage_info["params"], stage_name, overrides
            )

            # 3. Hash deps (pass state_db for hash caching)
            dep_hashes, missing, unreadable, _ = worker.hash_dependencies(
                stage_info["deps"], store, stage_db
            )
            if missing:
                logger.error(
                    "Stage '%s': missing deps: %s — skipping", stage_name, ", ".join(missing)
                )
                failed.append(stage_name)
                continue
            if unreadable:
                logger.error(
                    "Stage '%s': unreadable deps: %s — skipping",
                    stage_name,
                    ", ".join(unreadable),
                )
                failed.append(stage_name)
                continue

            dep_hashes_by_identity = {
                types.identity_from_key(key): info for key, info in dep_hashes.items()
            }
            stage_outs = stage_info["outs"]

            # Compute normalized output paths once (used for skip check, lock data, and StateDB)
            out_paths = [types.identity_key(out.identity) for out in stage_outs]
            production_lock = lock.StageLock(stage_name, stages_dir)
            previous_lock = production_lock.read()
            # A --no-commit run left a record of the code it ran; it decides once the outputs are
            # hashed, so a stage with one is never skipped here. The run writes its two keys
            # together and the reader drops a malformed key on its own, so a record without both
            # (one holding only another feature's key) decides nothing.
            run_record = stage_db.get_no_commit_record(stage_name)
            run_manifest: dict[str, str] | None = None
            run_outputs: dict[str, str] | None = None
            if (
                run_record is not None
                and "code_manifest" in run_record
                and "output_hashes" in run_record
            ):
                run_manifest = run_record["code_manifest"]
                run_outputs = run_record["output_hashes"]
            # The stage's own *.pvt files are the ones its lock or its nocommit: record lists.
            # Every other *.pvt in an output is a user's: the commit refuses before it would
            # claim one, so a later run cannot delete it as the stage's.
            claims = worker.stage_claims(store, stage_outs, previous_lock, run_record, project_root)

            # 5. If not force and no explicit stages, check lock — skip if unchanged
            if (
                not force
                and stage_names is None
                and previous_lock is not None
                and run_manifest is None
            ):
                decision = skip.check_stage(
                    lock_data=previous_lock,
                    fingerprint=fingerprint,
                    params=current_params,
                    dep_hashes=dep_hashes_by_identity,
                    out_paths=[out.identity for out in stage_outs],
                )
                if not decision["changed"]:
                    continue

            # 6. Hash and cache outputs
            output_hashes_by_key = dict[str, HashInfo]()
            output_hashes_by_identity = dict[types.ArtifactIdentity, HashInfo]()
            output_paths_by_identity = dict[types.ArtifactIdentity, str]()
            file_hash_entries = list[types.FileHashEntry]()
            outputs_missing = False
            for out in stage_outs:
                identity_key = types.identity_key(out.identity)
                try:
                    out_path = store.resolve_display_path(out)
                    hash_info = store.commit(
                        out,
                        out_path,
                        state_dir=default_state_dir,
                        warn_left_out_symlinks=warn_left_out_symlinks,
                        claims=claims,
                    )
                    file_hash_entries.extend(
                        cache.deferred_file_hash_entries(
                            cache.known_file_hashes(out_path, hash_info)
                        )
                    )
                    output_hashes_by_key[identity_key] = hash_info
                    output_hashes_by_identity[out.identity] = hash_info
                    output_paths_by_identity[out.identity] = out_path.relative_to(
                        project_root
                    ).as_posix()
                except FileNotFoundError:
                    logger.error(
                        "Stage '%s': output missing: %s — skipping", stage_name, identity_key
                    )
                    outputs_missing = True
                    break
                except OSError:
                    logger.error(
                        "Stage '%s': output unreadable: %s — skipping",
                        stage_name,
                        identity_key,
                    )
                    outputs_missing = True
                    break
                except exceptions.OutputOwnershipError as exc:
                    # This stage's failure alone: the stages before it stay committed, and the
                    # ones after it are still tried
                    logger.error("Stage '%s': %s", stage_name, exc)
                    outputs_missing = True
                    break

            if outputs_missing:
                failed.append(stage_name)
                continue

            # The outputs a --no-commit run wrote came from the code that run fingerprinted, which
            # may differ from the code on disk now: record that run's manifest, its marks
            # included. Otherwise (no record, or outputs edited since) record the outputs on disk
            # against the current code, the "trust me" commit. A run that executed code the
            # fingerprint does not describe marked its lock; committing the same outputs keeps
            # that mark, so the stage is still not reusable.
            if run_manifest is not None and run_outputs == {
                key: hash_info["hash"] for key, hash_info in output_hashes_by_key.items()
            }:
                code_manifest = run_manifest
            else:
                code_manifest = skip.carry_rerun_marks(
                    previous_lock["code_manifest"] if previous_lock is not None else None,
                    fingerprint,
                )
            reusable = skip.rerun_mark_reason(code_manifest) is None
            input_hash = run_history.stage_input_hash(
                skip.strip_rerun_marks(code_manifest),
                current_params,
                stage_info["deps"].values(),
                dep_hashes_by_identity,
                stage_outs,
            )

            # 7. Write production lock
            new_lock_data = LockData(
                code_manifest=code_manifest,
                params=current_params,
                dep_hashes=dict(sorted(dep_hashes_by_identity.items())),
                output_hashes=dict(sorted(output_hashes_by_identity.items())),
                output_paths=output_paths_by_identity,
            )
            try:
                production_lock.write(new_lock_data)
            except exceptions.InvalidPathError as exc:
                logger.error("%s — skipping", exc)
                failed.append(stage_name)
                continue

            # 8. Update StateDB: dep generations, output generations, run cache entry
            run_id = run_history.generate_run_id()

            dep_gen_map = worker.compute_dep_generation_map(stage_info["deps"], stage_db)

            # An unsafe fingerprint may match a run whose captured values differed, and a carried
            # rerun mark means the outputs came from code the fingerprint does not describe: never
            # file either. A carried mark also removes any entry an earlier run filed at this key.
            deferred = DeferredWrites(clear_no_commit_record=True)
            if dep_gen_map:
                deferred["dep_generations"] = dep_gen_map
            if reusable and not stage_info["unsafe_fingerprint"]:
                cached_paths = {
                    types.identity_key(out.identity)
                    for out in stage_outs
                    if out.tag is not ArtifactTag.METRIC
                }
                output_entries = [
                    run_history.output_hash_to_entry(path, oh)
                    for path, oh in output_hashes_by_key.items()
                    if path in cached_paths
                ]
                if output_entries:
                    deferred["run_cache_input_hash"] = input_hash
                    deferred["run_cache_entry"] = run_history.RunCacheEntry(
                        run_id=run_id,
                        output_hashes=output_entries,
                    )
            if not reusable:
                deferred["run_cache_delete_input_hash"] = input_hash
            if file_hash_entries:
                deferred["file_hash_entries"] = file_hash_entries

            stage_db.apply_deferred_writes(stage_name, out_paths, deferred)

            committed.append(stage_name)
    finally:
        for db in state_dbs.values():
            db.close()

    return committed, failed
