"""Commit current workspace state for pipeline stages.

Computes hashes from current disk state, writes lock files, caches outputs,
and updates StateDB. Used by `pivot commit` to record state after running
stages with --no-commit or making manual edits.
"""

from __future__ import annotations

import logging
import pathlib

from pivot import (
    config,
    exceptions,
    parameters,
    project,
    reads,
    registry,
    run_history,
    skip,
    types,
)
from pivot import state_dir as state_dir_mod
from pivot.executor import worker
from pivot.storage import cache, listing, lock, naming
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

    default_state_dir = config.locate_state_dir()
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
            hashed_dirs = dict[str, listing.HashedDirectory]()
            dep_hashes, missing, unreadable, _ = worker.hash_dependencies(
                stage_info["deps"], store, stage_db, hashed_dirs=hashed_dirs
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
            previous_lock = production_lock.read(cache_dir=worker.store_cache_dir(store))
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
            claims = worker.stage_claims(
                store,
                stage_outs,
                previous_lock,
                run_record,
                project_root,
                stage_lock=production_lock,
            )

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

            # 6. Hash and publish every output before the naming unit, so a large output's
            # reading and copying holds no lock a waiting collection would turn into a stall for
            # every other commit on the cache.
            prepared = dict[
                types.ArtifactIdentity,
                tuple[pathlib.Path, types.FileHash | listing.HashedDirectory],
            ]()
            outputs_missing = False
            for out in stage_outs:
                identity_key = types.identity_key(out.identity)
                try:
                    out_path = store.resolve_display_path(out)
                    prepared[out.identity] = (
                        out_path,
                        store.prepare_commit(
                            out,
                            out_path,
                            state_dir=default_state_dir,
                            warn_left_out_symlinks=warn_left_out_symlinks,
                            hashed_dirs=hashed_dirs,
                            claims=claims,
                        ),
                    )
                except FileNotFoundError:
                    logger.error(
                        "Stage '%s': output missing: %s — skipping", stage_name, identity_key
                    )
                    outputs_missing = True
                    break
                except exceptions.UnreadableLinkError as exc:
                    logger.error("Stage '%s': %s — skipping", stage_name, exc)
                    outputs_missing = True
                    break
                except OSError:
                    logger.error(
                        "Stage '%s': output unreadable: %s — skipping", stage_name, identity_key
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

            # A read by hash names nothing until a lock does: an object the run read (its
            # nocommit: record's, or the previous lock's that a trust-me commit keeps) may have
            # left the cache since. Fetch each again before the unit (no download inside one);
            # inside it, the reads the lock will record must be in the cache.
            fetches = reads.Recording(worker.store_cache_dir(store))
            unfetched = reads.fetch_missing(
                [
                    *(
                        run_record["reads"]
                        if run_record is not None and "reads" in run_record
                        else []
                    ),
                    *(
                        previous_lock["reads"]
                        if previous_lock is not None and "reads" in previous_lock
                        else []
                    ),
                ],
                fetches,
            )

            # 7. Under the naming lock (shared), the naming step only: re-check and link each
            # output (an object a collection removed since is published again), record its
            # materialized file hash (#516), then write the lock and the listings that name them.
            with naming.naming_unit(
                config.get_cache_dir(), project=(project_root, config.get_state_dir())
            ):
                output_hashes_by_key = dict[str, HashInfo]()
                output_hashes_by_identity = dict[types.ArtifactIdentity, HashInfo]()
                output_paths_by_identity = dict[types.ArtifactIdentity, str]()
                file_hash_entries = list[types.FileHashEntry]()
                for out in stage_outs:
                    identity_key = types.identity_key(out.identity)
                    out_path, hashed = prepared[out.identity]
                    try:
                        hash_info = store.finish_commit(
                            out, out_path, hashed, state_dir=default_state_dir, claims=claims
                        )
                    except FileNotFoundError:
                        logger.error(
                            "Stage '%s': output missing: %s — skipping", stage_name, identity_key
                        )
                        outputs_missing = True
                        break
                    except exceptions.UnreadableLinkError as exc:
                        logger.error("Stage '%s': %s — skipping", stage_name, exc)
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
                        # #538's swap check, at the link inside the unit: this stage's failure
                        # alone, as at the check before the unit
                        logger.error("Stage '%s': %s", stage_name, exc)
                        outputs_missing = True
                        break
                    try:
                        file_hash_entries.extend(
                            cache.deferred_file_hash_entries(
                                cache.known_file_hashes(
                                    out_path,
                                    hash_info,
                                    hashed_dirs[hash_info["hash"]]["listing"]
                                    if types.is_dir_hash(hash_info)
                                    else None,
                                )
                            )
                        )
                    except exceptions.UnreadableLinkError as exc:
                        logger.error("Stage '%s': %s — skipping", stage_name, exc)
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
                    output_hashes_by_key[identity_key] = hash_info
                    output_hashes_by_identity[out.identity] = hash_info
                    output_paths_by_identity[out.identity] = out_path.relative_to(
                        project_root
                    ).as_posix()

                if outputs_missing:
                    failed.append(stage_name)
                    continue

                # The outputs a --no-commit run wrote came from the code that run fingerprinted,
                # which may differ from the code on disk now: record that run's manifest, its
                # marks included, and the objects it read by hash. Otherwise (no record, or
                # outputs edited since) record the outputs on disk against the current code, the
                # "trust me" commit. A run that executed code the fingerprint does not describe
                # marked its lock; committing the same outputs keeps that mark, so the stage is
                # still not reusable. The previous lock's `reads` stay only when it records this
                # whole provenance: the same code, params, dependencies and outputs. A changed
                # input (a new pin) can leave the outputs' bytes alone while naming other
                # objects, and the old list would have status check what the old pin named and
                # never what the new one does; then no reads are recorded, and only a fresh run
                # or a matching --no-commit record records them.
                read_hashes = list[str]()
                output_content_hashes = {
                    key: hash_info["hash"] for key, hash_info in output_hashes_by_key.items()
                }
                if run_manifest is not None and run_outputs == output_content_hashes:
                    code_manifest = run_manifest
                    if run_record is not None and "reads" in run_record:
                        read_hashes = run_record["reads"]
                else:
                    code_manifest = skip.carry_rerun_marks(
                        previous_lock["code_manifest"] if previous_lock is not None else None,
                        fingerprint,
                    )
                    if (
                        previous_lock is not None
                        and "reads" in previous_lock
                        and previous_lock["code_manifest"] == code_manifest
                        and skip.params_equal(previous_lock["params"], current_params)
                        and previous_lock["dep_hashes"] == dep_hashes_by_identity
                        and {
                            types.identity_key(identity): hash_info["hash"]
                            for identity, hash_info in previous_lock["output_hashes"].items()
                        }
                        == output_content_hashes
                    ):
                        read_hashes = previous_lock["reads"]
                reusable = skip.rerun_mark_reason(code_manifest) is None
                input_hash = run_history.stage_input_hash(
                    skip.strip_rerun_marks(code_manifest),
                    current_params,
                    stage_info["deps"].values(),
                    dep_hashes_by_identity,
                    stage_outs,
                )

                try:
                    reads.require_cached(read_hashes, fetches.cache_dir, unfetched)
                except exceptions.ObjectReadError as exc:
                    logger.error("Stage '%s': %s — skipping", stage_name, exc)
                    failed.append(stage_name)
                    continue

                # 7. Write production lock
                new_lock_data = LockData(
                    code_manifest=code_manifest,
                    params=current_params,
                    dep_hashes=dict(sorted(dep_hashes_by_identity.items())),
                    output_hashes=dict(sorted(output_hashes_by_identity.items())),
                    output_paths=output_paths_by_identity,
                )
                if read_hashes:
                    new_lock_data["reads"] = read_hashes
                try:
                    production_lock.write(
                        new_lock_data,
                        hashed_dirs=hashed_dirs,
                        input_paths=worker.input_listing_paths(
                            stage_info["deps"], store, new_lock_data["dep_hashes"]
                        ),
                        cache_dir=store.cache_dir,
                        outs=stage_outs,
                    )
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
                    if read_hashes:
                        deferred["run_cache_entry"]["reads"] = read_hashes
            if not reusable:
                older_key = worker.older_pivot_input_hash(
                    fingerprint,
                    current_params,
                    stage_info["deps"].values(),
                    dep_hashes_by_identity,
                    stage_outs,
                    hashed_dirs,
                    store.cache_dir,
                )
                deferred["run_cache_delete_input_hashes"] = [
                    input_hash,
                    *([] if older_key is None else [older_key]),
                ]
            if file_hash_entries:
                deferred["file_hash_entries"] = file_hash_entries
            if fetched := fetches.fetched():
                deferred["remote_hashes"] = fetched

            stage_db.apply_deferred_writes(stage_name, out_paths, deferred)

            committed.append(stage_name)
    finally:
        for db in state_dbs.values():
            db.close()

    return committed, failed
