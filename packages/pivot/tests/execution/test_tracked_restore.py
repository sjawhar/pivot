"""How the pre-flight's ``--checkout-missing`` restore of a missing tracked path ends, and the dry
run's prediction of it (``executor.tracked_restore``): the outcome, the warning, and that a run
outside leave-out mode still stops as in 0.10.4."""

from __future__ import annotations

import contextlib
import logging
import os
import stat
from typing import TYPE_CHECKING

import pytest
import xxhash

from pivot import exceptions, project
from pivot.executor import core as executor_core
from pivot.executor import tracked_restore
from pivot.storage import cache, ownership, track
from tests import helpers

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Generator

    import click.testing


@pytest.fixture
def pipeline_dir(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Set a real project root for CLI track plus direct executor calls."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".pivot").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(project, "get_project_root", lambda: tmp_path)
    return tmp_path


def _failed(path: pathlib.Path) -> ownership.MissingTracked:
    """``path`` left missing by a ``--checkout-missing`` restore that failed."""
    return ownership.MissingTracked(
        paths=[str(path)], failed=frozenset({str(path)}), checkout_attempted=True
    )


# A --checkout-missing pre-flight that leaves nothing missing
_NONE_MISSING = ownership.MissingTracked(paths=[], failed=frozenset(), checkout_attempted=True)


@contextlib.contextmanager
def _caught_warnings(caplog: pytest.LogCaptureFixture) -> Generator[list[str]]:
    """The message of each warning the pre-flight's modules (``executor.core`` and the restore's,
    ``executor.tracked_restore``) log in the block, filled in when the block ends. `pivot track`
    sets up the CLI's logging, which takes every handler off the root logger, caplog's among
    them, so the block puts caplog's back on for its length when it is not there."""
    root = logging.getLogger()
    added = caplog.handler not in root.handlers
    if added:
        root.addHandler(caplog.handler)
    caught = list[str]()
    try:
        with caplog.at_level(logging.WARNING):
            caplog.clear()
            yield caught
    finally:
        if added:
            root.removeHandler(caplog.handler)
    caught.extend(
        record.getMessage()
        for record in caplog.records
        if record.name in (executor_core.logger.name, tracked_restore.logger.name)
    )


def _tracked_images(
    runner: click.testing.CliRunner, root: pathlib.Path, *, cached_listing: bool
) -> tuple[pathlib.Path, track.PvtData, pathlib.Path]:
    """Track the directory ``root / "images"``, holding a.jpg, with the real ``pivot track`` and
    remove it from the workspace. Return its path, its .pvt record and the path of its listing's
    object in the cache, which is deleted unless ``cached_listing``."""
    images = root / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"image1")
    helpers.track_then_remove(runner, root, "images")
    record = track.read_pvt_file(track.get_pvt_path(images))
    assert record is not None
    listing_object = cache.get_cache_path(root / ".pivot" / "cache" / "files", record["hash"])
    if not cached_listing:
        listing_object.unlink()
    return images, record, listing_object


@pytest.mark.parametrize("committed", ["stale", "empty"])
def test_a_lost_listing_leaves_its_directory_missing_in_leave_out_mode_and_stops_any_other_run(
    pipeline_dir: pathlib.Path,
    runner: click.testing.CliRunner,
    caplog: pytest.LogCaptureFixture,
    committed: str,
) -> None:
    """With checkout_missing, a missing tracked directory whose committed listing is stale or
    empty, and whose cache holds no copy of the listing to fall back on (a fresh clone), has lost
    its listing (``ListingUnavailableError``). The dry run's view and the leave-out restore
    (tolerate_missing) each leave it missing, as a failed restore, with the one warning `pivot
    checkout` gives for it; a restore outside leave-out mode still stops on it, a
    ``CorruptListingError`` as in 0.10.4."""
    images, record, _ = _tracked_images(runner, pipeline_dir, cached_listing=False)
    listing_file = pipeline_dir / "images.pvt.listing"
    listing_file.write_bytes(
        listing_file.read_bytes().replace(b"a.jpg", b"b.jpg") if committed == "stale" else b""
    )
    problem = (
        f"{listing_file}: its bytes hash to {xxhash.xxh64(listing_file.read_bytes()).hexdigest()}, "
        + f"not to the directory hash {record['hash']} its lock or .pvt records"
    )

    with _caught_warnings(caplog) as dry_run_warnings:
        found = executor_core.find_missing_tracked_files(pipeline_dir, checkout_missing=True)
    with _caught_warnings(caplog) as run_warnings:
        missing = executor_core.verify_tracked_files(
            pipeline_dir, checkout_missing=True, tolerate_missing=True, outputs={}
        )
    with pytest.raises(exceptions.ListingUnavailableError) as exc_info:
        executor_core.verify_tracked_files(pipeline_dir, checkout_missing=True)

    assert found == executor_core.MissingTrackedFiles(
        missing=_failed(images), restorable={}, restored={}
    )
    assert missing == _failed(images)
    warning = f"Cannot restore {images}: {problem}. {exceptions.CorruptListingError.REMEDY}"
    assert dry_run_warnings == [warning]
    assert run_warnings == [warning]
    assert isinstance(exc_info.value, exceptions.CorruptListingError)
    assert str(exc_info.value) == problem
    assert not images.exists()


@pytest.mark.parametrize("corruption", ["listing_disagrees_with_its_pvt", "corrupt_cache_object"])
def test_any_other_corrupt_listing_stops_the_dry_run_as_it_stops_the_leave_out_restore(
    pipeline_dir: pathlib.Path,
    runner: click.testing.CliRunner,
    caplog: pytest.LogCaptureFixture,
    corruption: str,
) -> None:
    """A corrupt listing that is not unavailable still raises ``CorruptListingError``, not the
    ``ListingUnavailableError`` a leave-out restore tolerates, in the dry run's view and in the
    leave-out restore alike, with no warning: a committed listing that hashes to the .pvt's
    record but lists fewer files than the .pvt records, the cache holding no copy; and a stale
    committed listing beside a cache object at the record's hash whose bytes do not hash to
    it."""
    images, record, cache_object = _tracked_images(runner, pipeline_dir, cached_listing=False)
    pvt_path = track.get_pvt_path(images)
    listing_file = pipeline_dir / "images.pvt.listing"
    if corruption == "listing_disagrees_with_its_pvt":
        track.write_pvt_file(
            pvt_path, track.PvtData(path="images", hash=record["hash"], size=6, nfiles=2)
        )
        problem = (
            f"{listing_file}: lists 1 files of 6 bytes, but its lock or .pvt records 2 files of "
            + "6 bytes"
        )
    else:
        listing_file.write_bytes(listing_file.read_bytes().replace(b"a.jpg", b"b.jpg"))
        cache_object.write_bytes(b"junk\n")
        problem = (
            f"{cache_object}: its bytes hash to {xxhash.xxh64(b'junk\n').hexdigest()}, not to the "
            + f"directory hash {record['hash']} its lock or .pvt records"
        )

    with _caught_warnings(caplog) as warnings:
        with pytest.raises(exceptions.CorruptListingError) as dry_run_error:
            executor_core.find_missing_tracked_files(pipeline_dir, checkout_missing=True)
        with pytest.raises(exceptions.CorruptListingError) as run_error:
            executor_core.verify_tracked_files(
                pipeline_dir, checkout_missing=True, tolerate_missing=True, outputs={}
            )

    assert type(dry_run_error.value) is exceptions.CorruptListingError
    assert type(run_error.value) is exceptions.CorruptListingError
    assert str(dry_run_error.value) == problem
    assert str(run_error.value) == problem
    assert warnings == []
    assert not images.exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a cache object whatever its mode")
@pytest.mark.parametrize("committed", ["intact", "stale"])
@pytest.mark.parametrize("unreadable", ["object", "prefix_directory"])
def test_an_unreadable_cached_listing_leaves_its_directory_missing_in_leave_out_mode_only(
    pipeline_dir: pathlib.Path,
    runner: click.testing.CliRunner,
    caplog: pytest.LogCaptureFixture,
    committed: str,
    unreadable: str,
) -> None:
    """With checkout_missing, a missing tracked directory whose listing object in the cache this
    user cannot read, as its mode or its prefix directory's mode is 0o000, cannot be restored,
    whether its committed listing is intact (the read still checks the cache's copy for a
    collision) or stale. The dry run's view and the leave-out restore (tolerate_missing) each
    leave it missing, as a failed restore, with one warning naming the path and the error and no
    listing remedy. A restore outside leave-out mode still stops as in 0.10.4: on the
    ``ListingUnavailableError`` (a ``CorruptListingError``) that reading the object raises from
    the ``PermissionError``, or on the raw ``PermissionError`` of the prefix directory's lookup.
    Once the mode is back, the directory is restorable."""
    images, _, listing_object = _tracked_images(runner, pipeline_dir, cached_listing=True)
    if committed == "stale":
        listing_file = pipeline_dir / "images.pvt.listing"
        listing_file.write_bytes(listing_file.read_bytes().replace(b"a.jpg", b"b.jpg"))
    locked = listing_object if unreadable == "object" else listing_object.parent
    mode = stat.S_IMODE(locked.stat().st_mode)
    denied = f"[Errno 13] Permission denied: '{listing_object}'"
    problem = f"{listing_object}: cannot be read ({denied})" if unreadable == "object" else denied

    locked.chmod(0)
    try:
        with _caught_warnings(caplog) as dry_run_warnings:
            found = executor_core.find_missing_tracked_files(pipeline_dir, checkout_missing=True)
        with _caught_warnings(caplog) as run_warnings:
            missing = executor_core.verify_tracked_files(
                pipeline_dir, checkout_missing=True, tolerate_missing=True, outputs={}
            )
        with pytest.raises((exceptions.ListingUnavailableError, PermissionError)) as exc_info:
            executor_core.verify_tracked_files(pipeline_dir, checkout_missing=True)
    finally:
        locked.chmod(mode)
    readable = executor_core.find_missing_tracked_files(pipeline_dir, checkout_missing=True)

    assert found == executor_core.MissingTrackedFiles(
        missing=_failed(images), restorable={}, restored={}
    )
    assert missing == _failed(images)
    assert dry_run_warnings == [f"Cannot restore {images}: {problem}"]
    assert run_warnings == dry_run_warnings
    if unreadable == "object":
        assert type(exc_info.value) is exceptions.ListingUnavailableError
        assert isinstance(exc_info.value.__cause__, PermissionError)
    else:
        assert type(exc_info.value) is PermissionError
    assert str(exc_info.value) == problem
    assert not images.exists()
    assert list(readable.restorable) == [str(images)]


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes in a read-only directory")
@pytest.mark.parametrize("beside_an_absent_file", [False, True], ids=["alone", "beside_absent"])
def test_a_failed_restore_over_an_output_is_named_as_failed_not_as_absent_from_the_cache(
    pipeline_dir: pathlib.Path, runner: click.testing.CliRunner, beside_an_absent_file: bool
) -> None:
    """With checkout_missing and tolerate_missing, a missing tracked file inside the directory
    output ``locked`` of stage test/locked, whose object the cache holds but whose restore fails
    (``locked`` is read-only), stops the run as an overlap. The error names it as a file whose
    restore failed, not one the local cache lacks, and suggests no pull; beside it, a tracked
    file with no object is still one the local cache lacks. The dry run's view finds the same
    failed restore, and its refusal gives the same text."""
    locked = pipeline_dir / "locked"
    locked.mkdir()
    cached = locked / "cached.csv"
    cached.write_text("cached data")
    helpers.track_then_remove(runner, pipeline_dir, "locked/cached.csv")
    absent = locked / "absent.csv"
    if beside_an_absent_file:
        helpers.track_without_data(absent)
    outputs = {ownership.normalized(locked): "test/locked"}

    locked.chmod(0o555)
    try:
        found = executor_core.find_missing_tracked_files(pipeline_dir, checkout_missing=True)
        with pytest.raises(exceptions.TrackedFileMissingError) as exc_info:
            executor_core.verify_tracked_files(
                pipeline_dir, checkout_missing=True, tolerate_missing=True, outputs=outputs
            )
    finally:
        locked.chmod(0o755)
    dry_run_refusal = ownership.tracked_output_refusal(found.missing, outputs, pipeline_dir)

    overlap = "it is inside output 'locked' of stage 'test/locked'"
    if beside_an_absent_file:
        listed = f"  - {absent}: {overlap}\n  - {cached}: {overlap}; its restore failed\n"
        reason = (
            "A 'Cannot restore' warning above names the cause of each restore that failed. "
            + "The other files are not in local cache."
        )
        suggestion = "Run 'pivot pull' to fetch from remote storage"
    else:
        listed = f"  - {cached}: {overlap}; its restore failed\n"
        reason = "Their restore failed: a 'Cannot restore' warning above names each cause."
        suggestion = "Fix what each 'Cannot restore' warning above names, then run again"
    assert found.missing.failed == frozenset({str(cached)})
    assert str(exc_info.value) == (
        f"The following tracked files are missing:\n{listed}\n{reason}\n{helpers.OVERLAP_STOPS}"
    )
    assert exc_info.value.get_suggestion() == suggestion
    assert dry_run_refusal is not None
    assert str(dry_run_refusal) == str(exc_info.value)
    assert not cached.exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes in a read-only directory")
@pytest.mark.parametrize("nested", [False, True], ids=["in_it", "below_a_missing_directory"])
def test_a_restore_that_cannot_write_leaves_its_path_missing_in_leave_out_mode_only(
    pipeline_dir: pathlib.Path,
    runner: click.testing.CliRunner,
    caplog: pytest.LogCaptureFixture,
    nested: bool,
) -> None:
    """With checkout_missing, a missing tracked file whose object the cache holds, and whose
    nearest existing directory above it, ``locked``, is read-only, cannot be restored. The dry
    run's view leaves it missing, with one warning naming ``locked``; the leave-out restore
    (tolerate_missing) tries, fails with PermissionError and leaves it missing, with one warning
    naming the path and the error; a restore outside leave-out mode still raises the error. The
    nested case's .pvt names ``sub/cached.csv``, whose directory ``sub`` does not exist: the
    restore would make it in ``locked``. Once ``locked`` is writable, the path is restorable
    and the restore brings it back."""
    locked = pipeline_dir / "locked"
    locked.mkdir()
    (locked / "cached.csv").write_text("cached data")
    helpers.track_then_remove(runner, pipeline_dir, "locked/cached.csv")
    data_file = locked / "cached.csv"
    if nested:
        pvt_path = track.get_pvt_path(data_file)
        record = track.read_pvt_file(pvt_path)
        assert record is not None
        track.write_pvt_file(
            pvt_path, track.PvtData(path="sub/cached.csv", hash=record["hash"], size=record["size"])
        )
        data_file = locked / "sub" / "cached.csv"

    locked.chmod(0o555)
    try:
        with _caught_warnings(caplog) as dry_run_warnings:
            found = executor_core.find_missing_tracked_files(pipeline_dir, checkout_missing=True)
        with _caught_warnings(caplog) as run_warnings:
            missing = executor_core.verify_tracked_files(
                pipeline_dir, checkout_missing=True, tolerate_missing=True, outputs={}
            )
        with pytest.raises(PermissionError):
            executor_core.verify_tracked_files(pipeline_dir, checkout_missing=True)
    finally:
        locked.chmod(0o755)
    writable = executor_core.find_missing_tracked_files(pipeline_dir, checkout_missing=True)

    assert found == executor_core.MissingTrackedFiles(
        missing=_failed(data_file), restorable={}, restored={}
    )
    assert dry_run_warnings == [f"Cannot restore {data_file}: this user cannot write in {locked}"]
    assert missing == _failed(data_file)
    assert len(run_warnings) == 1, run_warnings
    assert run_warnings[0].startswith(
        f"Cannot restore {data_file}: [Errno 13] Permission denied: '{locked}/"
    ), run_warnings
    assert not data_file.exists()
    assert list(writable.restorable) == [str(data_file)]
    assert (
        executor_core.verify_tracked_files(
            pipeline_dir, checkout_missing=True, tolerate_missing=True, outputs={}
        )
        == _NONE_MISSING
    )
    assert data_file.read_text() == "cached data"


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a cache object whatever its mode")
@pytest.mark.parametrize("kind", ["file", "directory"])
@pytest.mark.parametrize("unreadable", ["object", "prefix_directory"])
def test_a_restore_that_cannot_read_its_cache_object_leaves_its_path_missing_in_leave_out_mode_only(
    pipeline_dir: pathlib.Path,
    runner: click.testing.CliRunner,
    caplog: pytest.LogCaptureFixture,
    kind: str,
    unreadable: str,
) -> None:
    """With checkout_missing, a missing tracked path whose cache object (a file's, or a member's
    of a directory) this user cannot read, as its mode or its prefix directory's mode is 0o000,
    cannot be restored. The dry run's view reads the object as the restore does (the lookup,
    then an open for reading) and leaves the path missing, with the warning the restore gives
    for the error; the leave-out restore (tolerate_missing) fails with PermissionError and leaves
    it missing, with one such warning; a restore outside leave-out mode still raises the error.
    Once the mode is back, the path is restorable and the restore brings it back."""
    cache_dir = pipeline_dir / ".pivot" / "cache" / "files"
    if kind == "file":
        data_path = pipeline_dir / "cached.csv"
        data_path.write_text("cached data")
    else:
        data_path = pipeline_dir / "images"
        data_path.mkdir()
        (data_path / "a.jpg").write_bytes(b"image1")
    helpers.track_then_remove(runner, pipeline_dir, data_path.name)
    record = track.read_pvt_file(track.get_pvt_path(data_path))
    assert record is not None
    object_hash = (
        record["hash"]
        if kind == "file"
        else track.tracked_members(data_path, record, cache_dir)["files"][0]["hash"]
    )
    cache_object = cache.get_cache_path(cache_dir, object_hash)
    locked = cache_object if unreadable == "object" else cache_object.parent
    mode = stat.S_IMODE(locked.stat().st_mode)

    locked.chmod(0)
    try:
        with _caught_warnings(caplog) as dry_run_warnings:
            found = executor_core.find_missing_tracked_files(pipeline_dir, checkout_missing=True)
        with _caught_warnings(caplog) as run_warnings:
            missing = executor_core.verify_tracked_files(
                pipeline_dir, checkout_missing=True, tolerate_missing=True, outputs={}
            )
        with pytest.raises(PermissionError):
            executor_core.verify_tracked_files(pipeline_dir, checkout_missing=True)
    finally:
        locked.chmod(mode)
    readable = executor_core.find_missing_tracked_files(pipeline_dir, checkout_missing=True)

    assert found == executor_core.MissingTrackedFiles(
        missing=_failed(data_path), restorable={}, restored={}
    )
    assert dry_run_warnings == [
        f"Cannot restore {data_path}: [Errno 13] Permission denied: '{cache_object}'"
    ]
    assert missing == _failed(data_path)
    assert len(run_warnings) == 1, run_warnings
    assert run_warnings[0].startswith(
        f"Cannot restore {data_path}: [Errno 13] Permission denied: "
    ), run_warnings
    assert not data_path.exists()
    assert list(readable.restorable) == [str(data_path)]
    assert (
        executor_core.verify_tracked_files(
            pipeline_dir, checkout_missing=True, tolerate_missing=True, outputs={}
        )
        == _NONE_MISSING
    )
    assert data_path.exists()
