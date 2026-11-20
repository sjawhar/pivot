"""The restore of a missing tracked path from the local cache, and the dry run's prediction of it.

The pre-flight's ``--checkout-missing`` restore (``executor.core.verify_tracked_files``) restores
through ``restore_tracked``. The dry run in leave-out mode
(``executor.core.find_missing_tracked_files``) asks ``check_tracked``, which writes nothing. Both
find an entry at the path that the restore in leave-out mode does not replace through one check,
``_occupied``, and what the restore links through one read, ``_source``. ``check_tracked``
predicts the rest of the restore (how it checks each cache object out, which objects it reads,
whether it can write the path, and where it refuses to link from a cache it could not register
in) from reads alone; the agreement tests, which give a dry run and a run the same tree, guard
that prediction.
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
    warning naming the path and why: an entry it does not replace stands at the path
    (``_occupied``), the directory's listing is unavailable
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
    an entry the restore does not replace at the path (``_occupied``: a socket, a FIFO or a
    device, or a symlink to one), a directory whose listing is unavailable and a restore that
    fails with ``OSError`` give ``Outcome.FAILED``, leaving the path as it is, missing, each with
    a warning naming the path; a refusal (a ``PivotError``, never an ``OSError``) and a corrupt
    listing still raise. Without it, they raise, and the restore replaces such an entry, as in
    0.10.4.
    """
    if tolerate_missing and _occupied(path):
        return Outcome.FAILED
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
    cache_root: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
    *,
    links_refused: bool,
) -> Check:
    """How ``restore_tracked`` in leave-out mode would end for missing tracked path ``path``,
    restored from the cache ``cache_root``, predicted from reads alone, warning as the restore
    would that it fails.

    No entry the restore does not replace may stand at the path (``_occupied``, the read the
    restore makes first). The restore's own source (``_source``) must be found: an unavailable
    listing fails, a missing one is absent. The restore must be able to read each cache object
    it links, as it checks them out with ``checkout_modes``: where it links no whole tree of a
    directory as it is (``_linked_tree``), each object is found (``cache.missing_linked_objects``)
    and opened for reading, as the checkout opens it;
    an ``OSError`` from that lookup or open (an object or a directory of the cache this user
    cannot read) fails it, with the warning the restore gives for the error. A symlink restore
    of a directory whose whole tree the cache holds opens none. And the restore must be able to
    write the path (``_write_failure``): one that starts from the cache's whole tree of a
    directory makes the directory that holds the path before it reads the tree or looks up any
    member's object, so that is checked first (``_tree_start_failure``); one that links that
    tree as it is, into an existing directory it cannot write in
    (``state_dir.nearest_entry_problem``), fails at the link and looks each member's object up
    before it restores file by file, so a member the cache lacks makes it absent there, with no
    warning.

    With ``links_refused`` (``naming.register_refuses_links``: the run's registration refuses
    links into the cache), the restore checks every object out as ``cache.refused_link_modes``
    says: with COPY alone, opening each one, or, with no COPY among ``checkout_modes``, it
    refuses where it reaches a link (``_check_refusing``).

    Raises:
        CacheError: With ``links_refused`` and no COPY among ``checkout_modes``, the refusal
            the restore raises where it reaches a link (``_check_refusing``).
    """
    if _occupied(path):
        return Check(Outcome.FAILED, set())
    cache_dir = cache_root / "files"
    source = _source(path, track_data, cache_dir, tolerate_missing=True)
    if isinstance(source, Outcome):
        return Check(source, set())
    modes = checkout_modes
    if links_refused:
        copied = cache.refused_link_modes(checkout_modes)
        if copied is None:
            return _check_refusing(path, source, cache_root, checkout_modes)
        modes = copied
    try:
        failed = _tree_start_failure(path, source, cache_dir, modes)
        if failed is not None:
            return failed
        tree = _linked_tree(source, cache_dir, modes)
        if tree is None:
            absent = cache.missing_linked_objects(
                source.hash, source.members, cache_dir, readable=True
            )
        elif state_dir_mod.nearest_entry_problem(path.parent) is not None:
            # The tree's link into a directory it cannot write in fails: the restore then looks
            # each member's object up, and one the cache lacks makes it absent; else it restores
            # file by file and fails to stage the directory there (``_write_failure``)
            absent = cache.missing_linked_objects(source.hash, source.members, cache_dir)
        else:
            # The restore links the whole tree as it is, opening no object
            absent = list[str]()
    except OSError as e:
        _warn_failed(path, e)
        return Check(Outcome.FAILED, set())
    if absent:
        return Check(Outcome.ABSENT, set())
    failed = _write_failure(path)
    if failed is not None:
        return failed
    return Check(Outcome.RESTORED, _created_inside(source.members))


def _linked_tree(
    source: _Source, cache_dir: pathlib.Path, checkout_modes: list[cache.CheckoutMode]
) -> pathlib.Path | None:
    """The cache's whole tree of the directory ``source`` restores that a restore with
    ``checkout_modes`` links as it is (``cache.linked_tree``, its files stat'd as a restore that
    records what it writes finds them), else None, as for a file."""
    if not types.is_dir_hash(source.hash) or source.members is None:
        return None
    return cache.linked_tree(
        source.hash, source.members, cache_dir, checkout_modes, stat_files=True
    )


def _check_refusing(
    path: pathlib.Path,
    source: _Source,
    cache_root: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
) -> Check:
    """``check_tracked`` where the run's restore of missing tracked path ``path`` from
    ``source`` refuses every link into the cache ``cache_root`` and has no COPY among
    ``checkout_modes`` (``cache.refused_link_modes``): it refuses where it reaches a link, and
    only there, opening no object.

    A file's restore links its object, and a directory's restore that links the cache's whole
    tree (``cache.linked_tree``, as a restore that records what it writes finds it) links that
    tree: each makes the directory that holds the path and refuses before it writes there, so it
    refuses once that directory exists or can be made, and fails, with a warning, where it
    cannot. A file's restore first looks up its object (one the cache lacks makes it absent); a
    directory's restore that starts from the cache's whole tree (``cache.restore_tree``) makes
    that directory before it reads the tree or looks up any member's object, so it fails there
    first, whatever it would then find (``_tree_start_failure``). Any other directory's
    restore, one from a whole tree whose files it cannot stat included (it falls back to
    restoring file by file), looks each member's object up one by one
    (``cache.missing_linked_objects``; one the cache lacks makes it absent) before it reaches a
    link, and stages the directory beside the path, as ``_write_failure`` checks, before it
    links its first file and refuses; a listing that holds no file links nothing, so that
    restore brings the directory back.

    The refusal names the path and the cache's object store, not what the run's names: the
    object or the tree it links, or, for a directory restored file by file, that file and its
    object. The agreement tests compare only the words before those.

    Raises:
        CacheError: The restore reaches a link.
    """
    cache_dir = cache_root / "files"
    try:
        failed = _tree_start_failure(path, source, cache_dir, checkout_modes)
        if failed is not None:
            return failed
        tree = _linked_tree(source, cache_dir, checkout_modes)
        # The restore links a whole tree as it is, looking up no object of it. Any other one
        # looks up each object it links before it reaches a link
        absent = (
            []
            if tree is not None
            else cache.missing_linked_objects(source.hash, source.members, cache_dir)
        )
    except OSError as e:
        _warn_failed(path, e)
        return Check(Outcome.FAILED, set())
    if absent:
        return Check(Outcome.ABSENT, set())
    members = source.members
    if members is None or tree is not None:
        # A restore that links the file or the whole tree makes the directory that holds the
        # path and links before it writes there
        if not path.parent.is_dir():
            failed = _write_failure(path)
            if failed is not None:
                return failed
        raise cache.unregistered_link_error(cache_dir, path, cache_root)
    # Any other stages the directory beside the path, then links its first file
    failed = _write_failure(path)
    if failed is not None:
        return failed
    if members["files"]:
        raise cache.unregistered_link_error(cache_dir, path, cache_root)
    return Check(Outcome.RESTORED, _created_inside(members))


def _tree_start_failure(
    path: pathlib.Path,
    source: _Source,
    cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
) -> Check | None:
    """``Outcome.FAILED`` where the restore of missing tracked directory ``path`` from
    ``source`` with ``checkout_modes``, the modes its checkout uses, starts from the cache's
    whole tree of it (``cache.restore_tree``) and cannot make the directory that holds ``path``
    (``_write_failure``), else None. That restore makes the directory before it reads the tree
    or looks up any member's object, so it fails there whatever it would then find. An
    ``OSError`` of the tree's lookup is raised, as the restore meets it there."""
    if (
        not types.is_dir_hash(source.hash)
        or path.parent.is_dir()
        or cache.restore_tree(source.hash, cache_dir, checkout_modes) is None
    ):
        return None
    return _write_failure(path)


def _write_failure(path: pathlib.Path) -> Check | None:
    """``Outcome.FAILED`` where the restore of missing tracked path ``path`` cannot write in the
    directory that holds it, else None. The nearest existing entry at or above that directory
    must be a directory this process may write in and search
    (``state_dir.nearest_entry_problem``), where the restore makes each missing directory on the
    way and stages the path; else it fails, with a warning naming that entry, where the
    restore's own warning gives the error its write meets."""
    problem = state_dir_mod.nearest_entry_problem(path.parent)
    if problem is None:
        return None
    entry, _ = problem
    _warn_failed(path, f"this user cannot write in {entry}")
    return Check(Outcome.FAILED, set())


def _occupied(path: pathlib.Path) -> bool:
    """Whether the restore of missing tracked path ``path`` in leave-out mode fails before it
    reads the cache, warning that it fails (``_warn_failed``): an entry stands at the path that
    makes the pre-flight count it missing and that the restore does not replace
    (``track.occupant``: a socket, a FIFO or a device, or a symlink to one). The restore never
    replaces such an entry, which another program may be using, and never opens it; the run
    leaves its readers out. Where nothing is at the path, or a symlink to nothing (a symlink
    checkout whose cache object is gone leaves one), the restore goes on and replaces it. Where a
    regular file or a directory landed after the pre-flight found the path missing, it goes on
    too, and its own check at its swap meets that entry (``ownership.swap_refusal``). One
    ``stat`` decides, and an ``OSError`` of it other than "no such file" fails the restore too;
    an ``lstat`` only words the warning ("a symlink to ...")."""
    try:
        occupant = track.occupant(path)
    except OSError as e:
        _warn_failed(path, e)
        return True
    if occupant is None:
        return False
    _warn_failed(path, f"{occupant} is in its place, which the restore does not replace")
    return True


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
