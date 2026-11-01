"""Expanding mapped stages (``Pipeline.map``): the one rule a run and every command follow.

A template (``registry.MapTemplate``) stands for one instance per file of a directory output, its
source. It expands only when the source's producer is current, and then from the listing the
producer's lock records for that directory (``read_keys``): the listing committed beside the lock,
else the cache's object at the directory's hash (``listing.read``). Those are the bytes restore,
push and pull read, so a clone that holds the lock and its listing expands a template as the run
that wrote them did, whether or not the directory is on disk.

A run decides "current" as it goes: the producer ran or was skipped in it (``engine``). A command
decides it by the explain path ``status`` uses (``status.expand_current``): the producer's lock
matches, and its output is in the cache or, missing from it, restorable from the remote.

An instance whose key that listing no longer holds is orphaned (``orphaned_instances``): its lock,
listings and outputs belong to no stage. The run that expands its template removes them
(``engine``); before it, ``pivot status`` lists the instance and ``pivot repro --dry-run`` names
it. A template that does not expand removes nothing, since no plan is known for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import lmdb

from pivot import exceptions, registry, types
from pivot import state_dir as state_dir_mod
from pivot.storage import listing, lock
from pivot.storage import state as state_mod

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Collection, Mapping


class Expansion(TypedDict):
    """What expanding a pipeline's templates gave.

    Attributes:
        expanded: Each expanded template's instances, in key order.
        not_expanded: Each template left unexpanded, with why.
        listings: For a template whose producer is current but whose listing neither the
            committed file nor the cache holds, the hash of that listing: the cache object a
            pull fetches from the remote before it expands again.
    """

    expanded: dict[str, list[str]]
    not_expanded: dict[str, str]
    listings: dict[str, str]


def single_stage_refusal(
    target: str, template: registry.MapTemplate
) -> exceptions.PipelineConfigError:
    """Why a run of named stages only (``pivot run``) cannot take ``target``, which names
    ``template`` or one of its instances: a template expands only in a run that finds its
    producer current, and such a run resolves no producer."""
    source = template["source"]
    return exceptions.PipelineConfigError(
        f"'{target}' is mapped over the files of '{types.identity_key(source)}': it expands only "
        + f"in a run that also runs or skips '{source.producer}'. Run "
        + f"`pivot repro {template['info']['name']}`."
    )


def refusal_reason(source: types.ArtifactIdentity, error: exceptions.PivotError) -> str:
    """Why a template cannot expand over the files of directory output ``source``, in the one
    wording a run and every command use: a listing that is missing names the commands that
    bring it back, and any other refusal (a file whose key cannot name a stage, an unusable
    listing, an instance another stage's name or output path holds) is given as it is."""
    shown = types.identity_key(source)
    if isinstance(error, exceptions.ListingMissingError):
        return (
            f"the listing of '{shown}' is missing ({error}): `pivot pull` fetches it from a "
            + f"remote that holds it, and `pivot repro --force {source.producer}` writes it again"
        )
    return f"cannot expand over the files of '{shown}': {error}"


def stage_lock(
    producer: registry.RegistryStageInfo, default_state_dir: pathlib.Path
) -> lock.StageLock:
    """The lock of ``producer``, beside the listings of its directory outputs."""
    return lock.StageLock(
        producer["name"],
        lock.get_stages_dir(registry.get_stage_state_dir(producer, default_state_dir)),
    )


def recorded_directory(
    source: types.ArtifactIdentity, lock_data: types.LockData | None
) -> types.DirHash | None:
    """The record ``lock_data`` keeps for directory output ``source``, or None when the lock
    is missing or records no directory there."""
    if lock_data is None:
        return None
    record = lock_data["output_hashes"].get(source)
    return record if record is not None and types.is_dir_hash(record) else None


def keys_of(members: listing.DirListing) -> list[tuple[str, str]]:
    """Each file of a listing as (instance key, file key) (``registry.map_instance_keys``)."""
    return registry.map_instance_keys(entry["relpath"] for entry in members["files"])


def read_keys(
    source: types.ArtifactIdentity,
    producer: registry.RegistryStageInfo,
    lock_data: types.LockData | None,
    *,
    default_state_dir: pathlib.Path,
    cache_dir: pathlib.Path | None,
) -> list[tuple[str, str]]:
    """The (instance key, file key) of each file of directory output ``source``, by the listing
    ``producer``'s lock (``lock_data``) records for it.

    Raises:
        ListingMissingError: The lock records no such directory, or neither the committed
            listing nor the cache (``cache_dir``, the object root) holds its listing.
        CorruptListingError: The listing cannot be used (``listing.read``).
        PipelineConfigError: A file's key cannot name a stage, or two files share a key.
    """
    record = recorded_directory(source, lock_data)
    if record is None or lock_data is None:
        raise exceptions.ListingMissingError(
            f"the lock of '{producer['name']}' records no directory "
            + f"'{types.identity_key(source)}'"
        )
    committed = lock.committed_listing(
        lock_data, stage_lock(producer, default_state_dir).output_listing_path(source)
    )
    return keys_of(listing.read(committed, record, cache_dir))


def expand_current(
    pipeline: registry.PipelineLike,
    locks: Mapping[str, types.LockData | None],
    waiting: Mapping[str, str],
    *,
    default_state_dir: pathlib.Path,
    cache_dir: pathlib.Path | None,
) -> Expansion:
    """Expand afresh each template of ``pipeline`` whose source's producer is current.

    ``locks`` holds the lock of each current producer, by name; ``waiting`` says, by producer
    name, why a producer is not current, which its templates report. Every expansion of an
    earlier call is removed first (``reset_expansions``), so a command sees the plan as it
    stands. A template mapped over an upstream template's instances expands with it, over the
    same keys, and stays unexpanded when that one does. Each template expands all or nothing
    (``registry.StageRegistry.expand``); a refusal leaves it unexpanded with the reason.
    """
    pipeline.reset_expansions()
    result = Expansion(expanded={}, not_expanded={}, listings={})
    keys_by_source = dict[types.ArtifactIdentity, list[tuple[str, str]] | str]()
    for name in pipeline.list_templates():
        template = pipeline.get_template(name)
        source = template["source"]
        producer = source.producer
        upstream = sorted(
            {dep for dep in template["mapped_deps"].values() if dep is not None}
            - set(result["expanded"])
        )
        if upstream:
            result["not_expanded"][name] = (
                f"mapped over the instances of '{upstream[0]}', which is not expanded"
            )
            continue
        if producer not in locks:
            why = waiting[producer] if producer in waiting else "it has not run"
            result["not_expanded"][name] = f"waits for '{producer}': {why}"
            continue
        keys = keys_by_source.get(source)
        if keys is None:
            try:
                keys = read_keys(
                    source,
                    pipeline.get_stage(producer),
                    locks[producer],
                    default_state_dir=default_state_dir,
                    cache_dir=cache_dir,
                )
            except exceptions.ListingMissingError as e:
                record = recorded_directory(source, locks[producer])
                if record is not None:
                    result["listings"][name] = record["hash"]
                keys = refusal_reason(source, e)
            except (exceptions.CorruptListingError, exceptions.PipelineConfigError) as e:
                keys = refusal_reason(source, e)
            keys_by_source[source] = keys
        if isinstance(keys, str):
            result["not_expanded"][name] = keys
            continue
        try:
            result["expanded"][name] = pipeline.expand_template(name, keys)
        except exceptions.PivotError as e:
            result["not_expanded"][name] = refusal_reason(source, e)
    return result


def instance_files(
    template: registry.MapTemplate, *, default_state_dir: pathlib.Path
) -> dict[str, list[pathlib.Path]]:
    """Each instance of ``template`` that keeps a file beside its lock, by name: its locks,
    listings and restart proofs, every lock last (``lock.instance_files``), in its state
    directory's stages directory and in the project's other one, of the 0.5.2 layout, when it
    has one (``state_dir.other_stages_dir_for``)."""
    info = template["info"]
    stages_dir = lock.get_stages_dir(registry.get_stage_state_dir(info, default_state_dir))
    other = state_dir_mod.other_stages_dir_for(stages_dir)
    return lock.instance_files([stages_dir] if other is None else [stages_dir, other], info["name"])


def no_commit_instances(
    template: registry.MapTemplate, *, default_state_dir: pathlib.Path
) -> list[str]:
    """Each instance of ``template`` that has a ``nocommit:`` record: a ``--no-commit`` run wrote
    its outputs, which no lock may record. A state database that cannot be read holds none, as
    ``ownership.read_stage_claims`` reads it."""
    name = template["info"]["name"]
    try:
        with state_mod.open_readonly(
            registry.get_stage_state_dir(template["info"], default_state_dir), create=False
        ) as state_db:
            stages = [] if state_db is None else state_db.no_commit_stages(f"{name}@")
    except (OSError, lmdb.Error, state_mod.StateDBError):
        return []
    return [stage for stage in stages if registry.template_of(stage, [name]) == name]


def orphaned_instances(
    template: registry.MapTemplate, keep: Collection[str], *, default_state_dir: pathlib.Path
) -> list[str]:
    """The instances of ``template`` not in ``keep``, the instances its producer's listing
    holds, that keep a file beside a lock (``instance_files``) or have a ``nocommit:`` record
    (``no_commit_instances``), sorted: their key left that listing, so the run that expands
    ``template`` removes them."""
    found = set(instance_files(template, default_state_dir=default_state_dir))
    found.update(no_commit_instances(template, default_state_dir=default_state_dir))
    return sorted(found.difference(keep))


def orphan_reason(template: registry.MapTemplate) -> str:
    """Why an instance of ``template`` is orphaned (``orphaned_instances``), in the one wording
    ``pivot status``, ``pivot repro --dry-run`` and a run use."""
    return f"its key left the listing of '{types.identity_key(template['source'])}'"
