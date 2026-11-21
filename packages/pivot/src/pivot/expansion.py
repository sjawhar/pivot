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
from typing import TYPE_CHECKING, NamedTuple, TypedDict

import lmdb

from pivot import exceptions, incremental_checks, project, registry, types
from pivot import state_dir as state_dir_mod
from pivot.storage import cache, listing, lock, ownership
from pivot.storage import state as state_mod

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from pivot.storage import store as store_mod


class Expansion(TypedDict):
    """What expanding a pipeline's templates gave.

    Attributes:
        expanded: Each expanded template's instances, in key order.
        listed: Each template of ``expanded`` with the instance of each key its producer's
            listing holds, in key order: an instance of the template that is not one of them is
            orphaned (``orphaned_instances``). ``expanded`` holds the same instances unless a
            run's ``checks`` kept some keys alone (``plan_instances``: a run whose targets name
            instances adds those alone).
        not_expanded: Each template left unexpanded, with why.
        listings: For a template whose producer is current but whose listing neither the
            committed file nor the cache holds, the record of that listing: the cache object a
            pull fetches from the remote before it expands again.
        refused: Each template of ``not_expanded`` that a refusal left unexpanded (a listing
            that cannot be read, a file whose key cannot name a stage, a refusal of
            ``plan_instances``, an instance another stage's name or output path holds); its
            reason is the one ``not_expanded`` gives. A template whose producer is current only
            because a pull would bring it up to date (``pulled``) is not in it when no listing
            could be read: a run runs that producer again and reads the keys from what it
            writes, so the producer's directory on disk, which that run replaces, is not read
            for it, under ``--no-commit`` either. Any other refusal of such a template is in it:
            it is over the keys the committed or cached listing gives, which that run is taken
            to write again, so the run meets it after it runs the producer.
            With a run's ``checks`` and, for a ``--no-commit`` run, its ``workspace``, these are
            the templates that run fails, with the reasons it gives: it skips the same
            producers or runs a pulled one again, reads the keys read here, and plans and adds
            the instances as here. A template mapped over the instances of one of them, directly
            or through other templates, is not in it: the run blocks that template.
    """

    expanded: dict[str, list[str]]
    listed: dict[str, list[str]]
    not_expanded: dict[str, str]
    listings: dict[str, types.DirHash]
    refused: set[str]


def empty_expansion() -> Expansion:
    """The expansion of no template."""
    return Expansion(expanded={}, listed={}, not_expanded={}, listings={}, refused=set[str]())


def without_templates(result: Expansion, names: Collection[str]) -> Expansion:
    """``result`` with every record of the templates ``names`` dropped, from each field."""
    return Expansion(
        expanded={name: found for name, found in result["expanded"].items() if name not in names},
        listed={name: found for name, found in result["listed"].items() if name not in names},
        not_expanded={
            name: why for name, why in result["not_expanded"].items() if name not in names
        },
        listings={name: record for name, record in result["listings"].items() if name not in names},
        refused={name for name in result["refused"] if name not in names},
    )


class InstanceChecks(NamedTuple):
    """What a run checks a template's instances against before it adds them
    (``plan_instances``). The run (``engine``) builds one from its targets and its start, and
    its dry run (``pivot repro --all --keep-going --dry-run``) builds the same from the same."""

    wanted: dict[str, dict[str, set[str]]]
    """The keys each template expands when the run's targets name instances, each with the
    targets that need it (``mapped_targets``); a template it does not hold expands every key."""
    missing_tracked: ownership.MissingTracked
    """The tracked files the run start left missing for the stages that read them, with those
    whose ``--checkout-missing`` restore failed (``executor.core.verify_tracked_files``'
    ``tolerate_missing``); none in any other run, which stops on a missing tracked file."""
    allow_uncached_incremental: bool
    force: bool


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
    missing tracked file, an instance's incremental output a run would lose) is given as it
    is."""
    shown = types.identity_key(source)
    if isinstance(error, exceptions.ListingMissingError):
        return (
            f"the listing of '{shown}' is missing ({error}): `pivot pull` fetches it from a "
            + f"remote that holds it, and `pivot repro --force {source.producer}` writes it again"
        )
    return f"cannot expand over the files of '{shown}': {error}"


def plan_instances(
    template: registry.MapTemplate, keys: list[tuple[str, str]], checks: InstanceChecks
) -> list[tuple[str, str]] | str:
    """The (key, file key) of each instance a run adds for ``template``, of the (key, file key)
    ``keys`` of each file of its source, or why the run fails the template, in the words the run
    gives. The one owner of what a run checks before it adds a template's instances: the run
    (``engine``) calls it for each template it expands, and its dry run passes the same
    ``checks`` to ``expand_current``, which calls it, so both fail the same templates.

    When the run's targets name instances of ``template`` (``checks.wanted``), only their keys
    are kept, and a named key no file has fails the template, as `pivot status` refuses the
    instance's name. The instances of the keys kept (``registry.instantiate``) fail it when an
    output is, lies inside or contains a missing tracked file, whose sidecar its write would
    delete (``ownership.tracked_output_refusal``), and when an incremental output on disk is not
    in the cache or was restarted by an older pivot (``incremental_checks.incremental_refusal``),
    each given by ``refusal_reason``. Adding the instances can still fail, on a name or an
    output path another stage holds (``registry.StageRegistry.expand``).
    """
    name = template["info"]["name"]
    if name in checks.wanted:
        wanted = checks.wanted[name]
        unknown = sorted(wanted.keys() - {key for key, _ in keys})
        if unknown:
            return _unknown_keys_reason(name, unknown, wanted)
        keys = [(key, file_key) for key, file_key in keys if key in wanted]
    planned = {
        info["name"]: info
        for info in (registry.instantiate(template, key, file_key) for key, file_key in keys)
    }
    project_root = project.get_project_root()
    refusal: exceptions.PivotError | None
    try:
        refusal = (
            ownership.tracked_output_refusal(
                checks.missing_tracked,
                ownership.stage_output_paths(project_root, planned),
                project_root,
            )
            if checks.missing_tracked.paths
            else None
        )
        if refusal is None:
            refusal = incremental_checks.incremental_refusal(
                list(planned),
                planned,
                allow_uncached_incremental=checks.allow_uncached_incremental,
                force=checks.force,
            )
    except exceptions.PivotError as error:
        # An error a check raises fails the template with it, as a refusal does
        refusal = error
    return keys if refusal is None else refusal_reason(template["source"], refusal)


def _unknown_keys_reason(template: str, missing: list[str], wanted: dict[str, set[str]]) -> str:
    """Why a run whose targets name keys ``missing`` of ``template`` fails it: those instances,
    as `pivot status` refuses their names, and each target further down the chain that needs
    them."""
    unknown = [f"{template}@{key}" for key in missing]
    reason = str(exceptions.StageNotFoundError(unknown))
    needing = sorted({target for key in missing for target in wanted[key]}.difference(unknown))
    if needing:
        noun = "target" if len(needing) == 1 else "targets"
        reason += f", needed by the {noun} {', '.join(needing)}"
    return reason


def workspace_keys(
    directory: pathlib.Path, *, project_root: pathlib.Path, cache_dir: pathlib.Path | None
) -> list[tuple[str, str]]:
    """The (instance key, file key) of each file of a template's source directory as the
    workspace holds it at ``directory``, hashed as recording a listing hashes it
    (``cache.hash_directory``, ``cache_dir`` the object root). A ``--no-commit`` run writes no
    listing, so once the producer has finished it reads the keys here, holding the READ lock on
    the directory (``engine``); its dry run reads here, with no lock, the keys of a directory
    whose listing neither the committed file nor the cache holds in a usable form, of a producer
    the run skips (``expand_current``'s ``workspace``).

    Raises:
        ListingMissingError: The directory is missing.
        PipelineConfigError: A file's key cannot name a stage, or two files share a key.
    """
    try:
        hashed = cache.hash_directory(directory, cache_dir=cache_dir)
    except FileNotFoundError:
        shown = project.to_relative_path(directory, project_root)
        raise exceptions.ListingMissingError(f"the directory {shown} is missing") from None
    return keys_of(hashed["listing"])


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


class KeptTargets(NamedTuple):
    """What a run in leave-out mode keeps of its targets (``keep_targets``)."""

    left_out_targets: dict[str, exceptions.DependencyNotFoundError]
    """Each target, as given, whose stage the run leaves out, with that stage's cause."""
    targets: list[str]
    """The other targets, as given, in their order."""
    stages: list[str]
    """``targets`` with each instance name replaced by its template, each once
    (``mapped_targets``)."""
    wanted: dict[str, dict[str, set[str]]]
    """The keys each template expands for ``targets`` (``mapped_targets``)."""


def keep_targets(
    targets: list[str],
    templates: Mapping[str, registry.MapTemplate],
    left_out: Mapping[str, exceptions.DependencyNotFoundError],
) -> KeptTargets:
    """Run targets ``targets`` split into those whose stage a run leaves out, each with that
    stage's cause, and the others, with what ``mapped_targets`` gives for the others. ``left_out``
    holds the stages a run in leave-out mode leaves out for a missing input
    (``engine.graph.get_runnable_stages``); an instance target (``<template>@<key>``) stands for
    its template (``registry.run_target``). The run (``Engine``) and its dry run
    (``pivot repro --dry-run``) both call this, so they leave out the same targets and expand
    the same keys for the others."""
    found = dict[str, exceptions.DependencyNotFoundError]()
    for target in targets:
        stage = registry.run_target(target, templates)
        if stage in left_out:
            found[target] = left_out[stage]
    kept = [target for target in targets if target not in found]
    stages, wanted = mapped_targets(kept, templates)
    return KeptTargets(left_out_targets=found, targets=kept, stages=stages, wanted=wanted)


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


UPSTREAM_NOT_EXPANDED_REASON = "mapped over the instances of '{}', which is not expanded"
"""Why a template mapped over the instances of another template, which ``format`` names, is not
expanded: that one is not (``expand_current``)."""


def expand_current(
    pipeline: registry.PipelineLike,
    locks: Mapping[str, types.LockData | None],
    waiting: Mapping[str, str],
    *,
    default_state_dir: pathlib.Path,
    cache_dir: pathlib.Path | None,
    fetched: Mapping[str, listing.DirListing] | None = None,
    checks: InstanceChecks | None = None,
    pulled: Collection[str] = (),
    workspace: store_mod.WorkspaceStore | None = None,
) -> Expansion:
    """Expand afresh each template of ``pipeline`` whose source's producer is current.

    ``locks`` holds the lock of each current producer, by name; ``waiting`` says, by producer
    name, why a producer is not current, which its templates report. Every expansion of an
    earlier call is removed first (``reset_expansions``), so a command sees the plan as it
    stands. A template mapped over an upstream template's instances expands with it, over the
    same keys, and stays unexpanded when that one does: each is taken after every template it
    is mapped over (``_mapped_order``), as a run expands it, whatever order the pipeline holds
    them in, and the result holds them in the pipeline's order. Each template expands all or
    nothing (``registry.StageRegistry.expand``); a refusal leaves it unexpanded with the reason,
    and names it in ``refused`` unless its producer is in ``pulled`` and no listing could be
    read. ``fetched`` holds listings read from the remote, by directory hash, already checked
    against their records: a listing neither committed nor cached is taken from it (a dry run's
    read, which writes nothing to the cache). A listing neither holds is named in ``listings``
    for every template mapped over that source, the ones that reuse its read included.
    ``checks``, when given, are a run's: each template adds the instances ``plan_instances``
    plans over them, or fails with its reason, recorded as any other refusal (the dry run of
    ``pivot repro --all --keep-going`` passes the run's own). ``pulled`` names the producers of
    ``locks`` current only because a pull would bring them up to date: a run, which never
    pulls, runs each again and expands its templates from what it writes, taken to list the
    files the listing read here lists. ``workspace``, when given (that dry run of a
    ``--no-commit`` run), reads a listing that neither the committed file nor the cache holds,
    or that cannot be used, as that run reads its keys: from the directory in the workspace
    (``workspace_keys``), for a producer the run skips. The run hashes the directory that skip
    leaves at the lock's record, so a listing either holds lists the same files. It never reads
    the directory of a producer in ``pulled``, which the run replaces when it runs that producer
    again: a listing of such a producer that cannot be read leaves its templates unexpanded and
    not refused.
    """
    pipeline.reset_expansions()
    result = empty_expansion()
    read_by_source = dict[types.ArtifactIdentity, _SourceKeys]()
    for name in _mapped_order(pipeline):
        template = pipeline.get_template(name)
        source = template["source"]
        producer = source.producer
        upstream = sorted(
            {dep for dep in template["mapped_deps"].values() if dep is not None}
            - set(result["expanded"])
        )
        if upstream:
            result["not_expanded"][name] = UPSTREAM_NOT_EXPANDED_REASON.format(upstream[0])
            continue
        if producer not in locks:
            why = waiting[producer] if producer in waiting else "it has not run"
            result["not_expanded"][name] = f"waits for '{producer}': {why}"
            continue
        read = read_by_source.get(source)
        if read is None:
            read = _source_keys(
                source,
                pipeline.get_stage(producer),
                locks[producer],
                default_state_dir=default_state_dir,
                cache_dir=cache_dir,
                fetched=fetched,
                # The run runs a pulled producer again and reads the directory it rewrites
                workspace=None if producer in pulled else workspace,
            )
            read_by_source[source] = read
        if read.missing is not None:
            result["listings"][name] = read.missing
        keys = read.keys
        if isinstance(keys, exceptions.PivotError):
            result["not_expanded"][name] = refusal_reason(source, keys)
            # A run that runs a pulled producer again reads the listing that producer writes
            unread = isinstance(
                keys, (exceptions.ListingMissingError, exceptions.CorruptListingError)
            )
            if not (unread and producer in pulled):
                result["refused"].add(name)
            continue
        planned = keys if checks is None else plan_instances(template, keys, checks)
        if isinstance(planned, str):
            reason = planned
        else:
            try:
                result["expanded"][name] = pipeline.expand_template(name, planned)
                result["listed"][name] = [f"{name}@{key}" for key, _ in keys]
                continue
            except exceptions.PivotError as e:
                reason = refusal_reason(source, e)
        result["not_expanded"][name] = reason
        # A refusal over the keys read here, which a run that runs a pulled producer again meets
        result["refused"].add(name)
    return _in_order(result, pipeline.list_templates())


def _mapped_order(pipeline: registry.PipelineLike) -> list[str]:
    """The templates of ``pipeline`` in an order a run could expand them in: each after every
    template it is mapped over (its ``mapped_deps``). Of the templates that can come next, the
    one the pipeline holds first comes first, so a pipeline that holds each template after the
    ones it is mapped over keeps its order, and their instances register (``list_stages``,
    ``pivot list``) in the order 0.10.4 registers them."""
    import networkx as nx

    names = pipeline.list_templates()
    position = {name: index for index, name in enumerate(names)}
    graph: nx.DiGraph[str] = nx.DiGraph()
    graph.add_nodes_from(names)
    # A template the pipeline lacks never expands, so neither does its reader, which reports it
    # (``expand_current``)
    graph.add_edges_from(
        (upstream, name)
        for name in names
        for upstream in pipeline.get_template(name)["mapped_deps"].values()
        if upstream is not None and upstream in position
    )
    return list(nx.lexicographical_topological_sort(graph, key=position.__getitem__))


def _in_order(result: Expansion, names: list[str]) -> Expansion:
    """``result`` with the templates of each field in the order of ``names``."""
    return Expansion(
        expanded={name: result["expanded"][name] for name in names if name in result["expanded"]},
        listed={name: result["listed"][name] for name in names if name in result["listed"]},
        not_expanded={
            name: result["not_expanded"][name] for name in names if name in result["not_expanded"]
        },
        listings={name: result["listings"][name] for name in names if name in result["listings"]},
        refused=result["refused"],
    )


class _SourceKeys(NamedTuple):
    """What ``_source_keys`` read of a template's source directory."""

    keys: list[tuple[str, str]] | exceptions.PivotError
    """The (key, file key) of each file of the directory, or why its templates cannot expand
    (``refusal_reason`` words it): ``ListingMissingError`` or ``CorruptListingError`` when no
    listing could be read, ``PipelineConfigError`` when a file's key cannot name a stage."""
    missing: types.DirHash | None
    """The record of the directory's listing when neither the committed file, the cache nor
    ``fetched`` holds it, else None."""


def _source_keys(
    source: types.ArtifactIdentity,
    producer: registry.RegistryStageInfo,
    lock_data: types.LockData | None,
    *,
    default_state_dir: pathlib.Path,
    cache_dir: pathlib.Path | None,
    fetched: Mapping[str, listing.DirListing] | None,
    workspace: store_mod.WorkspaceStore | None,
) -> _SourceKeys:
    """The (key, file key) of each file of directory output ``source`` of ``producer``, whose
    lock is ``lock_data``, as ``expand_current`` reads them (``read_keys``, then ``workspace``
    or ``fetched``), or why its templates cannot expand, with the record of a listing that
    neither the committed file, the cache nor ``fetched`` holds."""
    try:
        return _SourceKeys(
            read_keys(
                source,
                producer,
                lock_data,
                default_state_dir=default_state_dir,
                cache_dir=cache_dir,
            ),
            None,
        )
    except (exceptions.ListingMissingError, exceptions.CorruptListingError) as e:
        if workspace is not None:
            output = next(out for out in producer["outs"] if out.identity == source)
            try:
                keys = workspace_keys(
                    workspace.resolve_display_path(output),
                    project_root=project.get_project_root(),
                    cache_dir=cache_dir,
                )
            except (exceptions.ListingMissingError, exceptions.PipelineConfigError) as error:
                return _SourceKeys(error, None)
            return _SourceKeys(keys, None)
        if isinstance(e, exceptions.CorruptListingError):
            return _SourceKeys(e, None)
        record = recorded_directory(source, lock_data)
        members = None if record is None or fetched is None else fetched.get(record["hash"])
        if members is None:
            return _SourceKeys(e, record)
        try:
            return _SourceKeys(keys_of(members), None)
        except exceptions.PipelineConfigError as error:
            return _SourceKeys(error, None)
    except exceptions.PipelineConfigError as e:
        return _SourceKeys(e, None)


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
