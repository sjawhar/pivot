"""Which ``*.pvt`` files tracked-file discovery leaves out as a stage's own members.

A stage may write a ``*.pvt`` file in a directory output; ``ownership.stage_members`` says when
that file is the stage's. Such a member is no tracked file, so every command that discovers the
project's ``.pvt`` files passes ``own_member_skip`` to ``track.discover_pvt_files``.

The command's own stages decide their outputs from the registry. Any other ``.pvt`` is decided
from what runs left behind, never by importing another ``pipeline.py``: the directory outputs
that other stages' locks record (``lock.recorded_directories``). A lock outlives its stage, so
the files a deleted stage wrote stay its members; ``pivot track`` refuses a path whose sidecar
such a stage records (``orphaned_member``), since a sidecar written there could be one no command
reads.
"""

from __future__ import annotations

import functools
import os
import pathlib
from typing import TYPE_CHECKING, NamedTuple

from pivot.storage import lock, ownership, track
from pivot.storage import store as store_mod

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping

    from pivot.registry import RegistryStageInfo


class _Recorded(NamedTuple):
    """The directory outputs other stages' locks record, and their members."""

    outputs: dict[pathlib.Path, lock.RecordedDirectory]
    """By ``ownership.normalized`` path inside the project."""
    members_of: Callable[[pathlib.Path], ownership.StageMembers]


def _recorded(
    project_root: pathlib.Path,
    default_state_dir: pathlib.Path,
    cache_dir: pathlib.Path,
    declared: Collection[str],
) -> _Recorded:
    """``lock.recorded_directories`` for every stage not ``declared``, placed in the project,
    with a reader of each output's own members (``ownership.members_reader``)."""
    root = ownership.normalized(project_root)
    outputs = dict[pathlib.Path, lock.RecordedDirectory]()
    stages_dir = lock.get_stages_dir(default_state_dir)
    for relative, recorded in lock.recorded_directories(stages_dir, declared).items():
        path = pathlib.Path(os.path.normpath(root / relative))
        if path != root and path.is_relative_to(root):
            outputs[path] = recorded

    def locked(output_path: pathlib.Path) -> ownership.LockedOutput:
        recorded = outputs[output_path]
        return ownership.LockedOutput(
            recorded.stage_name,
            recorded.state_dir,
            recorded.record.output_hash,
            functools.partial(recorded.read_listing, cache_dir),
        )

    return _Recorded(outputs, ownership.members_reader(project_root, locked, cache_dir))


def own_member_skip(
    project_root: pathlib.Path,
    stages: Mapping[str, RegistryStageInfo],
    default_state_dir: pathlib.Path,
    cache_dir: pathlib.Path,
) -> track.PvtSkip:
    """Whether a ``*.pvt`` file that parses is a stage's own member of its output
    (``ownership.stage_members``, the rule every write, restore and commit uses): ``skip`` for
    ``track.discover_pvt_files``, with ``cache_dir`` the cache object root.

    A ``*.pvt`` in an output of ``stages`` (the command's) is decided by the registry and the
    stage's lock. Any other is a member when a directory output that another stage's lock
    records holds it, and that lock's listing or the stage's ``nocommit:`` record lists it with
    its current content. Those locks are read at the first such ``*.pvt``, never for a command
    that meets none. When ``stages`` is every pipeline's (``--all``, ``track``), only the locks
    of stages no pipeline declares are read.

    Any other ``*.pvt`` in an output is the user's: discovery keeps it, and every restore of that
    output refuses it. A stage's claims are read only for an output that holds a ``*.pvt``.
    """
    outputs = ownership.registry_outputs(
        stages, lambda out: project_root / store_mod.relative_output_path(out), default_state_dir
    )
    members_of = ownership.members_reader(
        project_root,
        lambda path: ownership.locked_output(path, outputs[path], cache_dir),
        cache_dir,
    )
    recorded = functools.cache(
        functools.partial(_recorded, project_root, default_state_dir, cache_dir, stages)
    )

    def skip(pvt_path: pathlib.Path, data: track.PvtData) -> bool:
        path = ownership.normalized(pvt_path)
        loaded = [output for output in path.parents if output in outputs]
        if loaded:
            return any(
                ownership.is_stage_member(path, data, members_of(output), path) for output in loaded
            )
        other = recorded()
        return any(
            ownership.is_stage_member(path, data, other.members_of(output), path)
            for output in path.parents
            if output in other.outputs
        )

    return skip


class OrphanedMember(NamedTuple):
    """A sidecar path that the lock of a stage no pipeline declares records as that stage's."""

    sidecar: pathlib.Path
    stage_name: str
    lock_files: tuple[pathlib.Path, ...]


def orphaned_member(
    project_root: pathlib.Path,
    default_state_dir: pathlib.Path,
    cache_dir: pathlib.Path,
    declared: Collection[str],
) -> Callable[[pathlib.Path], OrphanedMember | None]:
    """A reader that says, for a path ``pivot track`` is asked to track, whether the lock of a
    stage not ``declared`` lists that path's sidecar (``<path>.pvt``) among the members of its
    output (``OrphanedMember``), or None.

    ``own_member_skip`` skips such a stage's members by its lock, so a sidecar ``track`` wrote at
    a listed path could carry the bytes the lock records and then be one no command reads: the
    listed path decides, whatever is there now. The sidecar is spelled as the ``.pvt`` walk
    reaches it: its directory where it really lies, below the project root, so a path typed
    through a symlinked directory is caught, and an output reached only through a symlinked
    parent is not one the walk (or the skip) would read it in. The sidecar itself is not
    resolved, as the walk does not resolve it: a member that is a symlink to its cache object (a
    checkout across filesystems) is still that member. Any other path in that output tracks as
    before. The locks are read at the first path asked about.
    """
    recorded = functools.cache(
        functools.partial(_recorded, project_root, default_state_dir, cache_dir, declared)
    )

    def claim(path: pathlib.Path) -> OrphanedMember | None:
        pvt = track.get_pvt_path(path)
        real = pvt.parent.resolve() / pvt.name
        real_root = project_root.resolve()
        if not real.is_relative_to(real_root):
            return None
        sidecar = ownership.normalized(project_root) / real.relative_to(real_root)
        found = recorded()
        for output in sidecar.parents:
            if output not in found.outputs or sidecar not in found.members_of(output):
                continue
            output_record = found.outputs[output]
            return OrphanedMember(sidecar, output_record.stage_name, output_record.lock_files)
        return None

    return claim
