"""The restore of a missing tracked path from the local cache.

The pre-flight's ``--checkout-missing`` restore (``executor.core.verify_tracked_files``, through
``restore_tracked``) and the dry run's view of it (``executor.core.find_missing_tracked_files``,
through ``check_tracked``) decide here, from the same reads, what the restore brings back and
how it fails, so they never disagree.
"""

from __future__ import annotations

import enum
import itertools
import logging
import pathlib
from typing import TYPE_CHECKING, NamedTuple

from pivot import config, exceptions, types
from pivot import state_dir as state_dir_mod
from pivot.storage import cache, ownership, track

if TYPE_CHECKING:
    import os

    from pivot.storage import listing

logger = logging.getLogger(__name__)


class Outcome(enum.Enum):
    """How the restore of a missing tracked path ends (``restore_tracked``), or would end
    (``check_tracked``)."""

    RESTORED = enum.auto()
    """The path is back."""
    ABSENT = enum.auto()
    """The local cache lacks what the restore links: the path's object, a member's object, or a
    directory's listing that no committed listing file holds either."""
    FAILED = enum.auto()
    """The restore failed, which only leave-out mode tolerates, with a ``Cannot restore``
    warning naming the path and why: the directory's listing is unavailable
    (``exceptions.ListingUnavailableError``), or reading the cache or writing the path fails
    with ``OSError``."""


class Check(NamedTuple):
    """What the restore of a missing tracked path would do (``check_tracked``)."""

    outcome: Outcome
    created: set[tuple[str, ...]]
    """With ``Outcome.RESTORED``, every path the restore creates inside the path, each as its
    parts relative to it: for a directory, each file and empty directory its listing holds and
    each directory on the way to them; for a file, none. Empty for any other outcome."""


class _Source(NamedTuple):
    """What the restore of a missing tracked path links from the cache (``_source``)."""

    hash: types.HashInfo
    members: listing.DirListing | None
    """A directory's listing; None for a file."""


def _source(
    path: pathlib.Path,
    track_data: track.PvtData,
    cache_dir: pathlib.Path,
    *,
    tolerate_missing: bool,
) -> _Source | Outcome:
    """What the restore of missing tracked path ``path`` links from ``cache_dir``: its hash and,
    for a directory, its committed listing file, else the cache's listing object.
    ``Outcome.ABSENT`` for a directory whose listing neither holds: no restore brings it back.

    A corrupt listing raises ``CorruptListingError``: a collision, a cache object whose bytes do
    not hash to the record, or a listing that does not parse or match the record. With
    ``tolerate_missing`` (leave-out mode), an unavailable listing (``listing._resolve`` raises
    ``ListingUnavailableError``: the committed file is stale, empty or unreadable and the cache
    holds no copy, or the cache's listing object or its prefix directory cannot be read) gives
    ``Outcome.FAILED`` instead, with a warning naming the path: for a lost listing, the one
    ``pivot checkout`` gives, with its remedy; for an object or directory of the cache that
    cannot be read, the error alone (``__cause__`` is the ``OSError``), as no listing file needs
    putting right. ``restore_tracked`` and ``check_tracked`` both read it here."""
    output_hash = track.pvt_to_hash_info(track_data)
    if not types.is_dir_hash(output_hash):
        return _Source(output_hash, None)
    try:
        return _Source(output_hash, track.tracked_members(path, track_data, cache_dir))
    except exceptions.ListingMissingError:
        return Outcome.ABSENT
    except exceptions.ListingUnavailableError as e:
        if not tolerate_missing:
            raise
        if isinstance(e.__cause__, OSError):
            _warn_failed(path, e)
        else:
            logger.warning("Cannot restore %s: %s. %s", path, e, e.REMEDY)
        return Outcome.FAILED


def restore_tracked(
    path: pathlib.Path,
    track_data: track.PvtData,
    cache_dir: pathlib.Path,
    project_root: pathlib.Path,
    state_dir: pathlib.Path,
    recorded: list[tuple[pathlib.Path, os.stat_result, str]],
    *,
    tolerate_missing: bool,
) -> Outcome:
    """Restore missing tracked path ``path``, a file or a directory, from cache without
    replacing a user file that lands while staging, and say how it ended.

    Each restored file is appended to ``recorded`` for the caller to record in StateDB
    (``cache.restore_from_cache``). A directory is restored from its committed listing file,
    else the cache's listing object (``_source``). With ``tolerate_missing`` (leave-out mode),
    a directory whose listing is unavailable and a restore that fails with ``OSError`` give
    ``Outcome.FAILED``, leaving the path missing, each with a warning naming the path; a refusal
    (a ``PivotError``, never an ``OSError``) and a corrupt listing still raise. Without it, they
    raise.
    """
    source = _source(path, track_data, cache_dir, tolerate_missing=tolerate_missing)
    if isinstance(source, Outcome):
        return source
    target = ownership.ForcedRestore(
        replacement=ownership.tracked_restore(path),
        restored=source.hash,
        claims=None,
    )
    if source.members is not None:
        target["restored_members"] = source.members

    def before_swap(entry: pathlib.Path) -> None:
        refusal = ownership.swap_refusal(project_root, target, entry)
        if refusal is not None:
            raise refusal

    # Use default checkout modes (hardlink with copy fallback).
    checkout_modes = config.get_checkout_mode_order()
    try:
        restored = cache.restore_from_cache(
            path,
            source.hash,
            cache_dir,
            checkout_modes=checkout_modes,
            state_dir=state_dir,
            members=source.members,
            before_swap=before_swap,
            recorded=recorded,
        )
    except OSError as e:
        if not tolerate_missing:
            raise
        _warn_failed(path, e)
        return Outcome.FAILED
    return Outcome.RESTORED if restored else Outcome.ABSENT


def check_tracked(
    path: pathlib.Path,
    track_data: track.PvtData,
    cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
) -> Check:
    """How ``restore_tracked`` in leave-out mode would end for missing tracked path ``path``,
    reading only, with the warning it would give.

    The restore's own source (``_source``) must be found: an unavailable listing fails, a
    missing one is absent. The restore must be able to read each cache object it links:
    ``cache.missing_cache_objects``, the rule a forced checkout uses, finds none absent for
    ``checkout_modes`` and opens each one for reading, as the restore's checkout opens it; an
    ``OSError`` from that lookup or open (an object or a directory of the cache this user cannot
    read) fails it, with the warning the restore gives for the error. And the restore must be
    able to write the path: the nearest existing entry above it is a directory this process may
    write in and search (``state_dir.nearest_entry_problem``), where the restore makes each
    missing directory on the way and stages the path; else it fails, with a warning naming the
    path and that entry, where the restore learns by trying.
    """
    source = _source(path, track_data, cache_dir, tolerate_missing=True)
    if isinstance(source, Outcome):
        return Check(source, set())
    try:
        absent = cache.missing_cache_objects(
            source.hash, source.members, cache_dir, checkout_modes, readable=True
        )
    except OSError as e:
        _warn_failed(path, e)
        return Check(Outcome.FAILED, set())
    if absent:
        return Check(Outcome.ABSENT, set())
    problem = state_dir_mod.nearest_entry_problem(path.parent)
    if problem is not None:
        logger.warning("Cannot restore %s: this user cannot write in %s", path, problem[0])
        return Check(Outcome.FAILED, set())
    return Check(Outcome.RESTORED, _created_inside(source.members))


def _warn_failed(path: pathlib.Path, error: Exception) -> None:
    """Warn that the restore of missing tracked path ``path`` failed with ``error``, an
    ``OSError`` or the ``ListingUnavailableError`` one caused, in the one wording the restore
    (``restore_tracked``) and its dry run (``check_tracked``) give."""
    logger.warning("Cannot restore %s: %s", path, error)


def _created_inside(members: listing.DirListing | None) -> set[tuple[str, ...]]:
    """The paths the restore of a missing tracked path creates inside it, each as its parts
    relative to it: for a directory with listing ``members``, each file and empty directory the
    listing holds and each directory on the way to them; for a file (no listing), none."""
    created = set[tuple[str, ...]]()
    if members is None:
        return created
    files = (entry["relpath"] for entry in members["files"])
    for relpath in itertools.chain(files, members["empty_dirs"]):
        parts = pathlib.PurePath(relpath).parts
        # A path held already came with the directories on the way to it
        while parts and parts not in created:
            created.add(parts)
            parts = parts[:-1]
    return created
