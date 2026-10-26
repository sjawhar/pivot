"""checkout/pull must never overwrite local edits inside an existing directory without --force."""

from __future__ import annotations

import functools
import multiprocessing
import os
import pathlib
import time
from typing import TYPE_CHECKING

from pivot import cli, config, project, types
from pivot.cli import checkout as checkout_mod
from pivot.remote import storage as remote_storage
from pivot.storage import cache, listing, state, track

if TYPE_CHECKING:
    import click.testing
    from pytest_mock import MockerFixture

    from pivot.registry import PipelineLike

_CRASH_EXIT = 73


def test_pull_only_missing_accepts_existing_cache_file_symlinks(
    mock_discovery: PipelineLike, runner: click.testing.CliRunner, mocker: MockerFixture
) -> None:
    data_dir = _helper_track_directory(project.get_project_root())
    cache.remove_output(data_dir)
    restored = runner.invoke(cli.cli, ["checkout", "--checkout-mode", "symlink", "data/ext"])
    assert restored.exit_code == 0, restored.output

    mocker.patch(
        "pivot.remote.sync.create_remote_from_name",
        return_value=(mocker.Mock(spec=remote_storage.S3Remote), "origin"),
    )
    mocker.patch(
        "pivot.remote.sync.pull",
        return_value=types.TransferSummary(transferred=0, skipped=0, failed=0, errors=[]),
    )

    result = runner.invoke(cli.cli, ["pull", "--checkout-mode", "symlink", "data/ext"])

    assert result.exit_code == 0, result.output
    assert (data_dir / "a.txt").is_symlink()
    assert (data_dir / "b.txt").is_symlink()


def _helper_write_tracked_directory(data_dir: pathlib.Path, record: types.HashInfo) -> None:
    """Write data_dir's .pvt and committed .pvt.listing for a cached directory (as `pivot track`)."""
    assert types.is_dir_hash(record)
    pvt_path = track.get_pvt_path(data_dir)
    data = listing.read_bytes(None, record, config.get_cache_dir() / "files")
    listing.write(listing.tracked_listing_path(pvt_path), data)
    track.write_pvt_file(
        pvt_path,
        track.PvtData(
            path=data_dir.name, hash=record["hash"], size=record["size"], nfiles=record["nfiles"]
        ),
    )


def _helper_track_directory(root: pathlib.Path) -> pathlib.Path:
    """Create data/ext with two files, cache them, and track them as `pivot track` does."""
    data_dir = root / "data" / "ext"
    data_dir.mkdir(parents=True)
    (data_dir / "a.txt").write_text("one\n")
    (data_dir / "b.txt").write_text("two\n")
    output_hash = cache.save_to_cache(
        data_dir, config.get_cache_dir() / "files", checkout_mode=config.CheckoutMode.COPY
    )
    _helper_write_tracked_directory(data_dir, output_hash)
    return data_dir


def _helper_cache_directory(
    path: pathlib.Path,
    contents: dict[str, str],
    checkout_mode: config.CheckoutMode = config.CheckoutMode.COPY,
) -> types.DirHash:
    path.mkdir(parents=True)
    for relpath, content in contents.items():
        file_path = path / relpath
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
    output_hash = cache.save_to_cache(
        path, config.get_cache_dir() / "files", checkout_mode=checkout_mode
    )
    assert types.is_dir_hash(output_hash)
    assert output_hash["nfiles"] == len(contents)
    assert output_hash["size"] == sum(len(content) for content in contents.values())
    return output_hash


def test_checkout_only_missing_accepts_existing_cache_file_symlinks(
    mock_discovery: PipelineLike, runner: click.testing.CliRunner
) -> None:
    data_dir = _helper_track_directory(project.get_project_root())
    cache.remove_output(data_dir)

    restored = runner.invoke(cli.cli, ["checkout", "--checkout-mode", "symlink", "data/ext"])
    assert restored.exit_code == 0, restored.output
    assert (data_dir / "a.txt").is_symlink()

    checked = runner.invoke(
        cli.cli, ["checkout", "--only-missing", "--checkout-mode", "symlink", "data/ext"]
    )

    assert checked.exit_code == 0, checked.output
    assert (data_dir / "a.txt").is_symlink()
    assert (data_dir / "b.txt").is_symlink()


def test_pull_replaces_directory_whose_differing_files_are_all_cached(
    mock_discovery: PipelineLike, runner: click.testing.CliRunner, mocker: MockerFixture
) -> None:
    """Changed and extra files that hold cached content are pivot's own output: pull replaces them."""
    root = project.get_project_root()
    data_dir = _helper_track_directory(root)
    new_record = _helper_cache_directory(
        root / "new-ext",
        {"a.txt": "one updated\n", "c.txt": "three\n"},
    )
    _helper_write_tracked_directory(data_dir, new_record)

    mocker.patch(
        "pivot.remote.sync.create_remote_from_name",
        return_value=(mocker.Mock(spec=remote_storage.S3Remote), "origin"),
    )
    mocker.patch(
        "pivot.remote.sync.pull",
        return_value=types.TransferSummary(transferred=0, skipped=0, failed=0, errors=[]),
    )

    result = runner.invoke(cli.cli, ["pull", "data/ext"])

    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in data_dir.iterdir()) == ["a.txt", "c.txt"]
    assert (data_dir / "a.txt").read_text() == "one updated\n"
    assert (data_dir / "c.txt").read_text() == "three\n"


def test_checkout_only_missing_replaces_directory_of_cached_file_symlinks(
    mock_discovery: PipelineLike, runner: click.testing.CliRunner
) -> None:
    """Symlinks into the cache for an older version are pivot's own output, not local edits."""
    root = project.get_project_root()
    data_dir = _helper_track_directory(root)
    cache.remove_output(data_dir)
    restored = runner.invoke(cli.cli, ["checkout", "--checkout-mode", "symlink", "data/ext"])
    assert restored.exit_code == 0, restored.output
    assert (data_dir / "b.txt").is_symlink()
    new_record = _helper_cache_directory(
        root / "new-ext",
        {"a.txt": "one updated\n", "c.txt": "three\n"},
    )
    _helper_write_tracked_directory(data_dir, new_record)

    result = runner.invoke(
        cli.cli, ["checkout", "--only-missing", "--checkout-mode", "symlink", "data/ext"]
    )

    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in data_dir.iterdir()) == ["a.txt", "c.txt"]
    assert (data_dir / "a.txt").read_text() == "one updated\n"


def test_checkout_only_missing_refuses_update_when_a_file_holds_an_uncached_edit(
    mock_discovery: PipelineLike, runner: click.testing.CliRunner
) -> None:
    """One edit the cache lacks blocks the update; only that file is named, nothing is restored."""
    root = project.get_project_root()
    data_dir = _helper_track_directory(root)
    new_record = _helper_cache_directory(
        root / "new-ext",
        {"a.txt": "one updated\n", "c.txt": "three\n"},
    )
    _helper_write_tracked_directory(data_dir, new_record)
    (data_dir / "b.txt").write_text("two EDITED\n")

    result = runner.invoke(cli.cli, ["checkout", "--only-missing", "data/ext"])

    assert result.exit_code != 0
    assert "'ext' has local changes that are not in the cache: b.txt. Use --force" in result.output
    assert (data_dir / "a.txt").read_text() == "one\n"
    assert (data_dir / "b.txt").read_text() == "two EDITED\n"
    assert not (data_dir / "c.txt").exists(), "A refused checkout must not restore anything"


def test_checkout_only_missing_refuses_untracked_entries_when_replacing_directory(
    mock_discovery: PipelineLike, runner: click.testing.CliRunner
) -> None:
    """Every directory entry is part of an artifact; untracked extras block replacement."""
    root = project.get_project_root()
    data_dir = _helper_track_directory(root)
    new_record = _helper_cache_directory(
        root / "new-ext",
        {"a.txt": "one updated\n", "c.txt": "three\n"},
    )
    _helper_write_tracked_directory(data_dir, new_record)
    (data_dir / "build").mkdir()
    (data_dir / "build" / "x.o").write_text("object\n")
    (data_dir / ".a.txt.swp").write_text("swap\n")
    (data_dir / "draft.txt~").write_text("backup\n")

    result = runner.invoke(cli.cli, ["checkout", "--only-missing", "data/ext"])

    assert result.exit_code != 0
    assert "local changes that are not in the cache" in result.output
    for relpath in (".a.txt.swp", "build/x.o", "draft.txt~"):
        assert relpath in result.output
    assert (data_dir / "a.txt").read_text() == "one\n"
    assert (data_dir / "build" / "x.o").read_text() == "object\n"
    assert not (data_dir / "c.txt").exists()


def test_checkout_only_missing_symlink_mode_refuses_untracked_editor_swap_file(
    mock_discovery: PipelineLike, runner: click.testing.CliRunner
) -> None:
    """An editor swap file is an untracked directory entry, not a file to carry forward."""
    root = project.get_project_root()
    data_dir = _helper_track_directory(root)
    new_record = _helper_cache_directory(
        root / "new-ext",
        {"a.txt": "one updated\n", "c.txt": "three\n"},
        checkout_mode=config.CheckoutMode.SYMLINK,
    )
    _helper_write_tracked_directory(data_dir, new_record)
    (data_dir / ".a.txt.swp").write_text("swap\n")

    result = runner.invoke(
        cli.cli, ["checkout", "--only-missing", "--checkout-mode", "symlink", "data/ext"]
    )

    assert result.exit_code != 0
    assert ".a.txt.swp" in result.output
    assert not data_dir.is_symlink()
    assert (data_dir / ".a.txt.swp").read_text() == "swap\n"
    assert (data_dir / "a.txt").read_text() == "one\n"
    assert not (data_dir / "c.txt").exists()


def _helper_restore_layout(root: pathlib.Path) -> tuple[pathlib.Path, types.DirHash]:
    """data/ext at v1; v2 lives in the cache."""
    cache_dir = root / ".pivot" / "cache" / "files"
    v2 = root / "v2src"
    v2.mkdir(parents=True)
    (v2 / "a.txt").write_text("one updated\n")
    (v2 / "c.txt").write_text("three\n")
    v2_hash = cache.save_to_cache(v2, cache_dir, checkout_mode=config.CheckoutMode.COPY)
    assert types.is_dir_hash(v2_hash)
    ext = root / "data" / "ext"
    ext.mkdir(parents=True)
    (ext / "a.txt").write_text("one\n")
    (ext / "b.txt").write_text("two\n")
    _ = cache.save_to_cache(ext, cache_dir, checkout_mode=config.CheckoutMode.COPY)
    return ext, v2_hash


def _helper_restore_only_missing(root: pathlib.Path, v2_hash: types.DirHash) -> str:
    """Run the per-target restore `pivot checkout --only-missing` runs for data/ext."""
    cache_dir = root / ".pivot" / "cache" / "files"
    outcome = checkout_mod._restore_path_sync(
        root / "data" / "ext",
        checkout_mod.Expected(v2_hash, functools.partial(listing.read, None, v2_hash, cache_dir)),
        cache_dir,
        [cache.CheckoutMode.COPY],
        checkout_mod.CheckoutBehavior.SKIP_EXISTING,
        root / ".pivot",
    )
    return outcome.result


def _helper_crash_at_rename(root: pathlib.Path, v2_hash: types.DirHash, crash_at: int) -> None:
    """Run the restore in a child process and exit it, unflushed, at rename number crash_at."""
    calls = 0
    original_rename = pathlib.Path.rename
    original_replace = pathlib.Path.replace

    def crash_rename(self: pathlib.Path, target: pathlib.Path) -> pathlib.Path:
        nonlocal calls
        if calls == crash_at:
            os._exit(_CRASH_EXIT)
        calls += 1
        return original_rename(self, target)

    def crash_replace(self: pathlib.Path, target: pathlib.Path) -> pathlib.Path:
        nonlocal calls
        if calls == crash_at:
            os._exit(_CRASH_EXIT)
        calls += 1
        return original_replace(self, target)

    pathlib.Path.rename = crash_rename  # pyright: ignore[reportAttributeAccessIssue] - crash injection
    pathlib.Path.replace = crash_replace  # pyright: ignore[reportAttributeAccessIssue] - crash injection
    _ = _helper_restore_only_missing(root, v2_hash)


def test_restore_killed_at_any_rename_restores_complete_directory(tmp_path: pathlib.Path) -> None:
    """A killed directory swap leaves a complete target after the next restore."""
    ctx = multiprocessing.get_context("spawn")
    crashes = 0
    for crash_at in range(20):
        root = tmp_path / f"crash-{crash_at}"
        _, v2_hash = _helper_restore_layout(root)
        child = ctx.Process(target=_helper_crash_at_rename, args=(root, v2_hash, crash_at))
        child.start()
        child.join(timeout=120)
        if child.exitcode == 0:
            break
        assert child.exitcode == _CRASH_EXIT, f"rename {crash_at}: child exit {child.exitcode}"
        crashes += 1
        aged = time.time() - 2 * 3600
        for leftover in (root / "data").iterdir():
            if leftover.name.startswith(".pivot_"):
                os.utime(leftover, (aged, aged))

        _ = _helper_restore_only_missing(root, v2_hash)

        ext = root / "data" / "ext"
        assert (ext / "a.txt").read_text() == "one updated\n", f"rename {crash_at}"
        assert (ext / "c.txt").read_text() == "three\n", f"rename {crash_at}"
        assert not (ext / "b.txt").exists(), f"rename {crash_at}"
    else:
        raise AssertionError("the restore never completed without a crash")
    assert crashes >= 2, f"only {crashes} crash points exercised"


def test_restore_after_killed_checkout_cleans_stale_restore_staging(tmp_path: pathlib.Path) -> None:
    """A later restore cleans the temporary paths a killed checkout left behind."""
    ext, v2_hash = _helper_restore_layout(tmp_path)
    child = multiprocessing.get_context("spawn").Process(
        target=_helper_crash_at_rename, args=(tmp_path, v2_hash, 1)
    )
    child.start()
    child.join(timeout=120)
    assert child.exitcode == _CRASH_EXIT

    restored = cache.restore_from_cache(
        ext,
        v2_hash,
        tmp_path / ".pivot" / "cache" / "files",
        checkout_modes=[cache.CheckoutMode.COPY],
        state_dir=tmp_path / ".pivot",
    )

    assert restored
    assert sorted(p.name for p in ext.iterdir()) == ["a.txt", "c.txt"]
    assert (ext / "a.txt").read_text() == "one updated\n"
    assert not any(p.name.startswith(".pivot_") for p in (tmp_path / "data").iterdir())


def test_checkout_only_missing_reads_no_content_of_unchanged_hardlinked_directory(
    mock_discovery: PipelineLike, runner: click.testing.CliRunner, mocker: MockerFixture
) -> None:
    """A workspace file that is a hardlink to its target blob is unchanged without a read."""
    data_dir = project.get_project_root() / "data" / "ext"
    record = _helper_cache_directory(
        data_dir, {"a.txt": "one\n", "b.txt": "two\n"}, checkout_mode=config.CheckoutMode.HARDLINK
    )
    _helper_write_tracked_directory(data_dir, record)
    assert (data_dir / "a.txt").stat().st_nlink == 2
    hash_file = mocker.spy(cache, "hash_file")

    result = runner.invoke(cli.cli, ["checkout", "--only-missing", "data/ext"])

    assert result.exit_code == 0, result.output
    assert "Skipped 2 file(s)" in result.output
    assert hash_file.call_count == 0


def test_checkout_only_missing_takes_unchanged_hashes_from_state_db(
    mock_discovery: PipelineLike, runner: click.testing.CliRunner, mocker: MockerFixture
) -> None:
    """Copied files whose StateDB entry still matches (as after `pivot status`) are not read."""
    data_dir = _helper_track_directory(project.get_project_root())
    with state.StateDB(config.get_state_dir()) as db:
        _ = cache.hash_directory(data_dir, db)
    opened = mocker.patch.object(cache, "open", create=True, side_effect=open)

    result = runner.invoke(cli.cli, ["checkout", "--only-missing", "data/ext"])

    assert result.exit_code == 0, result.output
    assert "Skipped 2 file(s)" in result.output
    assert opened.call_count == 0


def test_checkout_only_missing_reports_edited_file_in_directory(
    mock_discovery: PipelineLike, runner: click.testing.CliRunner
) -> None:
    data_dir = _helper_track_directory(project.get_project_root())
    (data_dir / "a.txt").write_text("one EDITED\n")
    (data_dir / "b.txt").unlink()

    result = runner.invoke(cli.cli, ["checkout", "--only-missing", "data/ext"])

    assert result.exit_code != 0
    assert "a.txt" in result.output
    assert "--force" in result.output
    assert (data_dir / "a.txt").read_text() == "one EDITED\n", "Edited file must not be replaced"
    assert not (data_dir / "b.txt").exists(), "A refused checkout must not partially restore files"


def test_checkout_only_missing_reports_untracked_file_in_directory(
    mock_discovery: PipelineLike, runner: click.testing.CliRunner
) -> None:
    data_dir = _helper_track_directory(project.get_project_root())
    (data_dir / "notes.txt").write_text("mine\n")

    result = runner.invoke(cli.cli, ["checkout", "--only-missing", "data/ext"])

    assert result.exit_code != 0
    assert "notes.txt" in result.output
    assert "--force" in result.output
    assert (data_dir / "notes.txt").read_text() == "mine\n", "Extra file must survive"


def test_checkout_default_refuses_edited_directory_and_force_overwrites(
    mock_discovery: PipelineLike, runner: click.testing.CliRunner
) -> None:
    data_dir = _helper_track_directory(project.get_project_root())
    (data_dir / "a.txt").write_text("one EDITED\n")

    refused = runner.invoke(cli.cli, ["checkout", "data/ext"])

    assert refused.exit_code != 0
    assert "--force" in refused.output, "Error should say how to override"
    assert (data_dir / "a.txt").read_text() == "one EDITED\n"

    forced = runner.invoke(cli.cli, ["checkout", "--force", "data/ext"])

    assert forced.exit_code == 0, forced.output
    assert (data_dir / "a.txt").read_text() == "one\n"


def test_checkout_force_refuses_an_edited_committed_listing_the_cache_cannot_replace(
    mock_discovery: PipelineLike, runner: click.testing.CliRunner
) -> None:
    """A committed listing whose bytes differ from the .pvt hash, with no cache object at that
    hash to read instead, is refused before any write."""
    data_dir = _helper_track_directory(project.get_project_root())
    (data_dir / "a.txt").write_text("one EDITED\n")
    committed = listing.tracked_listing_path(track.get_pvt_path(data_dir))
    original = committed.read_bytes()
    committed.write_bytes(original.replace(b" 4 - b.txt", b" 5 - b.txt"))
    cache.get_cache_path(config.get_cache_dir() / "files", listing.digest(original)).unlink()

    result = runner.invoke(cli.cli, ["checkout", "--force", "data/ext"])

    assert result.exit_code != 0
    assert str(committed) in result.output
    assert "not to the directory hash" in result.output
    assert (data_dir / "a.txt").read_text() == "one EDITED\n"
    assert (data_dir / "b.txt").read_text() == "two\n"


def test_checkout_only_missing_reports_extra_symlink_in_directory(
    mock_discovery: PipelineLike, runner: click.testing.CliRunner
) -> None:
    root = project.get_project_root()
    data_dir = _helper_track_directory(root)
    (root / "external.txt").write_text("outside\n")
    (data_dir / "notes-link").symlink_to(root / "external.txt")

    result = runner.invoke(cli.cli, ["checkout", "--only-missing", "data/ext"])

    assert result.exit_code != 0
    assert "notes-link" in result.output
    assert "--force" in result.output
    assert (data_dir / "notes-link").is_symlink()


def test_checkout_only_missing_rejects_symlink_parent_inside_ignored_directory(
    mock_discovery: PipelineLike, runner: click.testing.CliRunner
) -> None:
    root = project.get_project_root()
    data_dir = _helper_track_directory(root)
    (data_dir / "a.txt").unlink()
    (data_dir / "b.txt").unlink()
    outside = root / "outside"
    outside.mkdir()
    ignored_dir = data_dir / ".pivot"
    ignored_dir.mkdir()
    (ignored_dir / "escape").symlink_to(outside, target_is_directory=True)

    source = root / "new.txt"
    source.write_text("new")
    file_hash, _ = cache.hash_file(source)
    cache.copy_to_cache(source, cache.get_cache_path(config.get_cache_dir() / "files", file_hash))
    members = listing.DirListing(
        files=[
            types.DirManifestEntry(
                relpath=".pivot/escape/new.txt",
                hash=file_hash,
                size=3,
                isexec=False,
            )
        ],
        empty_dirs=[],
    )
    data = listing.encode(members)
    record = listing.record_of(members, data)
    pvt_path = track.get_pvt_path(data_dir)
    listing.write(listing.tracked_listing_path(pvt_path), data)
    track.write_pvt_file(
        pvt_path,
        track.PvtData(
            path="ext", hash=record["hash"], size=record["size"], nfiles=record["nfiles"]
        ),
    )

    result = runner.invoke(cli.cli, ["checkout", "--only-missing", "data/ext"])

    assert result.exit_code != 0
    assert "Manifest contains path traversal" in result.output
    assert not (outside / "new.txt").exists()
