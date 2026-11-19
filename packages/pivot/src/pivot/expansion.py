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

import pathlib
from typing import TYPE_CHECKING, TypedDict

import lmdb

from pivot import exceptions, registry, types
from pivot import state_dir as state_dir_mod
from pivot.storage import listing, lock
from pivot.storage import state as state_mod

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Mapping

# Why a template cannot add the instance of each (key, file key), or None (``expand_current``'s
# ``refuse``)
type InstanceRefusal = Callable[
    [registry.MapTemplate, list[tuple[str, str]]], exceptions.PivotError | None
]


class Expansion(TypedDict):
    """What expanding a pipeline's templates gave.

    Attributes:
        expanded: Each expanded template's instances, in key order.
        not_expanded: Each template left unexpanded, with why.
        listings: For a template whose producer is current but whose listing neither the
            committed file nor the cache holds, the record of that listing: the cache object a
            pull fetches from the remote before it expands again.
        refused: Each template of ``not_expanded`` that a refusal left unexpanded, with that
            refusal, its ``not_expanded`` reason (``refusal_reason``): a run that finds its
            producer current fails it with the same reason.
    """

    expanded: dict[str, list[str]]
    not_expanded: dict[str, str]
    listings: dict[str, types.DirHash]
    refused: dict[str, str]


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
    listing, an instance another stage's name or output path holds, an instance output over a
    missing tracked file) is given as it is."""
    shown = types.identity_key(source)
    if isinstance(error, exceptions.ListingMissingError):
        return (
            f"the listing of '{shown}' is missing ({error}): `pivot pull` fetches it from a "
            + f"remote that holds it, and `pivot repro --force {source.producer}` writes it again"
        )
    return f"cannot expand over the files of '{shown}': {error}"


def mapped_targets(
    stages: list[str], templates: Mapping[str, registry.MapTemplate]
) -> tuple[list[str], dict[str, dict[str, set[str]]]]:
    """Run targets ``stages`` with each instance name (``<template>@<key>``) replaced by its
    template, and the keys each template expands so the run adds the named instances alone,
    each with its chain: an instance's key for its template and for every template that one is
    mapped over, each key with the targets that need it. A template named itself, or upstream
    of one named itself, expands every key and has no entry; so does every template when no
    target names an instance.

    The instances of a run's expansions are gone when this is called (``reset_expansions``), so
    a watch run's affected instance and a name the user gave are both reached this way. The run
    (``Engine``) and its dry run (``pivot repro --dry-run``) both take the keys from here.
    """
    # Each template's keys, each with the targets that need it; None for every key
    wanted = dict[str, dict[str, set[str]] | None]()
    targets = list[str]()
    for stage in stages:
        target = registry.run_target(stage, templates)
        targets.append(target)
        if target not in templates:
            continue
        if target == stage:
            wanted[target] = None
            continue
        keys = wanted.setdefault(target, dict[str, set[str]]())
        if keys is not None:
            keys.setdefault(stage.removeprefix(f"{target}@"), set[str]()).add(stage)
    pending = list(wanted)
    while pending:
        name = pending.pop()
        keys = wanted[name]
        for upstream in {dep for dep in templates[name]["mapped_deps"].values() if dep is not None}:
            before = wanted.get(upstream, dict[str, set[str]]())
            if upstream in wanted and before is None:
                continue
            if keys is None:
                wanted[upstream] = None
                pending.append(upstream)
                continue
            merged = {key: set(needing) for key, needing in (before or {}).items()}
            for key, needing in keys.items():
                merged.setdefault(key, set[str]()).update(needing)
            if upstream in wanted and merged == before:
                continue
            wanted[upstream] = merged
            pending.append(upstream)
    filtered = {name: keys for name, keys in wanted.items() if keys is not None}
    return list(dict.fromkeys(targets)), filtered


def left_out_targets(
    targets: Iterable[str],
    templates: Collection[str],
    left_out: Mapping[str, exceptions.DependencyNotFoundError],
) -> dict[str, exceptions.DependencyNotFoundError]:
    """Each of run targets ``targets``, as given, whose stage a run leaves out, with that stage's
    cause. ``left_out`` holds the stages a run across all pipelines that keeps going past
    failures leaves out for a missing input (``engine.graph.get_runnable_stages``); an instance
    target (``<template>@<key>``) stands for its template (``registry.run_target``). The run
    (``Engine``) and its dry run (``pivot repro --dry-run``) both call this, so they leave out
    the same targets."""
    found = dict[str, exceptions.DependencyNotFoundError]()
    for target in targets:
        stage = registry.run_target(target, templates)
        if stage in left_out:
            found[target] = left_out[stage]
    return found


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
    fetched: Mapping[str, listing.DirListing] | None = None,
    refuse: InstanceRefusal | None = None,
) -> Expansion:
    """Expand afresh each template of ``pipeline`` whose source's producer is current.

    ``locks`` holds the lock of each current producer, by name; ``waiting`` says, by producer
    name, why a producer is not current, which its templates report. Every expansion of an
    earlier call is removed first (``reset_expansions``), so a command sees the plan as it
    stands. A template mapped over an upstream template's instances expands with it, over the
    same keys, and stays unexpanded when that one does. Each template expands all or nothing
    (``registry.StageRegistry.expand``); a refusal leaves it unexpanded with the reason, and
    names it in ``refused``.
    ``fetched`` holds listings read from the remote, by directory hash, already checked against
    their records: a listing neither committed nor cached is taken from it (a dry run's read,
    which writes nothing to the cache). A listing neither holds is named in ``listings`` for
    every template mapped over that source, the ones that reuse its read included. ``refuse``,
    when given, may refuse a template before its instances are added: given the template and
    the (key, file key) of each instance, it returns why the template cannot expand, or None. Its
    refusal is recorded as any other (a dry run of ``pivot repro`` passes the run's check of the
    instances' outputs over missing tracked files, ``ownership.tracked_output_refusal``).
    """
    pipeline.reset_expansions()
    result = Expansion(expanded={}, not_expanded={}, listings={}, refused={})
    keys_by_source = dict[types.ArtifactIdentity, list[tuple[str, str]] | str]()
    # The record of each source's listing that is neither committed, cached nor fetched
    missing_by_source = dict[types.ArtifactIdentity, types.DirHash]()
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
                members = None if record is None or fetched is None else fetched.get(record["hash"])
                if members is not None:
                    try:
                        keys = keys_of(members)
                    except exceptions.PipelineConfigError as error:
                        keys = refusal_reason(source, error)
                else:
                    if record is not None:
                        missing_by_source[source] = record
                    keys = refusal_reason(source, e)
            except (exceptions.CorruptListingError, exceptions.PipelineConfigError) as e:
                keys = refusal_reason(source, e)
            keys_by_source[source] = keys
        if source in missing_by_source:
            result["listings"][name] = missing_by_source[source]
        if isinstance(keys, str):
            reason = keys
        else:
            refusal = None if refuse is None else refuse(template, keys)
            if refusal is None:
                try:
                    result["expanded"][name] = pipeline.expand_template(name, keys)
                    continue
                except exceptions.PivotError as e:
                    refusal = e
            reason = refusal_reason(source, refusal)
        result["not_expanded"][name] = reason
        result["refused"][name] = reason
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


def no_commit_record_dirs(
    template: registry.MapTemplate, *, default_state_dir: pathlib.Path
) -> dict[str, list[pathlib.Path]]:
    """Each instance of ``template`` that has a ``nocommit:`` record, mapped to every state
    directory that holds one: a ``--no-commit`` run wrote its outputs, which no lock may record.
    Both current and legacy state directories are readable during a ``core.state_dir`` move.
    A database that cannot be read contributes none, as ``ownership.read_stage_claims`` reads it.
    """
    info = template["info"]
    state_dir = registry.get_stage_state_dir(info, default_state_dir)
    other = state_dir_mod.other_state_dir_for(state_dir)
    name = info["name"]
    records = dict[str, list[pathlib.Path]]()
    for candidate in [state_dir] if other is None else [state_dir, other]:
        try:
            with state_mod.open_readonly(candidate, create=False) as state_db:
                stages = [] if state_db is None else state_db.no_commit_stages(f"{name}@")
        except (OSError, lmdb.Error, state_mod.StateDBError):
            continue
        for stage in stages:
            if registry.template_of(stage, [name]) == name:
                records.setdefault(stage, []).append(candidate)
    return records


def no_commit_instances(
    template: registry.MapTemplate, *, default_state_dir: pathlib.Path
) -> list[str]:
    """Each instance of ``template`` that has a ``nocommit:`` record."""
    return sorted(no_commit_record_dirs(template, default_state_dir=default_state_dir))


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
