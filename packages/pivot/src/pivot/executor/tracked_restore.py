"""The restore of a missing tracked path from the local cache, and the dry run's prediction of it.

The pre-flight's ``--checkout-missing`` restore (``executor.core.verify_tracked_files``) restores
through ``restore_tracked``. The dry run in leave-out mode
(``executor.core.find_missing_tracked_files``) asks ``check_tracked``, which writes nothing. Both
find what the restore links through one read, ``_source``. ``check_tracked`` predicts the rest of
the restore (how it checks each cache object out, which objects it reads, and whether it can
write the path) from reads alone; the agreement tests, which give a dry run and a run the same
tree, guard that prediction.
"""

from __future__ import annotations

import enum
import itertools
import logging
import os
import pathlib
from typing import TYPE_CHECKING, NamedTuple

from pivot import config, exceptions, types
from pivot import state_dir as state_dir_mod
from pivot.storage import cache, naming, ownership, track

if TYPE_CHECKING:
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

    A corrupt listing raises ``CorruptListingError``. With ``tolerate_missing`` (leave-out
    mode), two failures of the read give ``Outcome.FAILED`` instead, with a warning naming the
    path: a listing that is unavailable (``exceptions.ListingUnavailableError``), with the
    warning ``pivot checkout`` gives and its remedy when a committed listing file was refused,
    else with the error alone (``__cause__`` is the ``OSError``), as no listing file needs
    putting right; and an ``OSError``, such as the one a prefix directory of the cache this user
    cannot search gives, which ``storage.listing`` raises as it is. Without it, both raise, the
    ``OSError`` as in 0.10.4. ``restore_tracked`` and ``check_tracked`` both read it here."""
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
        _warn_failed(path, e if isinstance(e.__cause__, OSError) else f"{e}. {e.REMEDY}")
        return Outcome.FAILED
    except OSError as e:
        if not tolerate_missing:
            raise
        _warn_failed(path, e)
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


def register_refuses_links(cache_root: pathlib.Path) -> bool:
    """Whether the run's registration in the cache ``cache_root`` refuses this process's links
    into it, told without writing: the cache's register (``naming.register_root``) lies on a
    filesystem mounted read-only here (``os.statvfs``'s ``ST_RDONLY``). There every write of a
    registration meets ``EROFS``, on which ``naming.naming_unit`` refuses the links
    (``cache.refuse_links_from``) and lets the run go on, so the run's restore copies each
    object, or refuses to link without COPY among its checkout modes. A cache that does not
    exist is registered in by no one (``naming.register_use``).

    A register this user cannot write on a writable filesystem is not one: the run's
    registration then stops the run before its pre-flight, in every mode, where the
    registration writes there (a new or old entry, or a kernel that refuses even the removal of
    an absent file in such a directory), and passes where it writes nothing. Only the write
    tells which, so the dry run does not predict it."""
    if not cache_root.is_dir():
        return False
    return bool(os.statvfs(naming.register_root(cache_root)).f_flag & os.ST_RDONLY)


def check_tracked(
    path: pathlib.Path,
    track_data: track.PvtData,
    cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
    *,
    links_refused: bool,
) -> Check:
    """How ``restore_tracked`` in leave-out mode would end for missing tracked path ``path``,
    predicted from reads alone, warning as the restore would that it fails.

    The restore's own source (``_source``) must be found: an unavailable listing fails, a
    missing one is absent. The restore must be able to read each cache object it links, as it
    checks them out with ``checkout_modes``: the rule a forced checkout uses
    (``cache.missing_cache_objects``) finds none absent and opens each one for reading, as the
    checkout opens it; an ``OSError`` from that lookup or open (an object or a directory of the
    cache this user cannot read) fails it, with the warning the restore gives for the error. A
    symlink restore of a directory whose whole tree the cache holds opens none. With
    ``links_refused`` (``register_refuses_links``: the run's registration refuses links into
    ``cache_dir``'s cache), the restore checks every object out with COPY alone, opening each
    one, or, with no COPY among ``checkout_modes``, refuses to link a path whose objects the
    cache holds, opening none. And the restore must be able to write the path: the nearest
    existing entry above it is a directory this process may write in and search
    (``state_dir.nearest_entry_problem``), where the restore makes each missing directory on the
    way and stages the path; else it fails, with a warning naming that entry, where the
    restore's own warning gives the error its write meets.

    Raises:
        CacheError: With ``links_refused`` and no COPY among ``checkout_modes``, the refusal
            the restore raises (``cache.unregistered_link_error``) for a path whose objects the
            cache holds.
    """
    source = _source(path, track_data, cache_dir, tolerate_missing=True)
    if isinstance(source, Outcome):
        return Check(source, set())
    refuses = links_refused and cache.CheckoutMode.COPY not in checkout_modes
    modes = [cache.CheckoutMode.COPY] if links_refused and not refuses else checkout_modes
    try:
        absent = cache.missing_cache_objects(
            source.hash, source.members, cache_dir, modes, readable=not refuses
        )
    except OSError as e:
        _warn_failed(path, e)
        return Check(Outcome.FAILED, set())
    if absent:
        return Check(Outcome.ABSENT, set())
    if refuses:
        # The cache root, whose object store cache_dir is, as the run's refusal names it
        raise cache.unregistered_link_error(cache_dir, path, cache_dir.parent)
    problem = state_dir_mod.nearest_entry_problem(path.parent)
    if problem is not None:
        entry, _ = problem
        _warn_failed(path, f"this user cannot write in {entry}")
        return Check(Outcome.FAILED, set())
    return Check(Outcome.RESTORED, _created_inside(source.members))


def _warn_failed(path: pathlib.Path, reason: object) -> None:
    """Warn that the restore of missing tracked path ``path`` fails for ``reason``: the one
    ``Cannot restore`` line the restore (``restore_tracked``) and its prediction
    (``check_tracked``) give."""
    logger.warning("Cannot restore %s: %s", path, reason)


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
