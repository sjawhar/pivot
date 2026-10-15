"""checkout/pull must never overwrite local edits inside an existing directory without --force."""

from __future__ import annotations

import multiprocessing
import os
import pathlib
import time
from typing import TYPE_CHECKING

from pivot import cli, config, project, types
from pivot.cli import checkout as checkout_mod
from pivot.remote import storage as remote_storage
from pivot.storage import cache, state, track

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


def _helper_track_directory(root: pathlib.Path) -> pathlib.Path:
    """Create data/ext with two files, cache them, and write data/ext.pvt (as `pivot track`)."""
    data_dir = root / "data" / "ext"
    data_dir.mkdir(parents=True)
    (data_dir / "a.txt").write_text("one\n")
    (data_dir / "b.txt").write_text("two\n")
    output_hash = cache.save_to_cache(
        data_dir, config.get_cache_dir() / "files", checkout_mode=config.CheckoutMode.COPY
    )
    assert "manifest" in output_hash
    track.write_pvt_file(
        track.get_pvt_path(data_dir),
        track.PvtData(
            path="ext",
            hash=output_hash["hash"],
            size=8,
            num_files=2,
            manifest=output_hash["manifest"],
        ),
    )
    return data_dir


def test_tracked_directory_hash_ignores_symlinks_after_a_regular_file_changes(
    tmp_path: pathlib.Path,
) -> None:
    """A tracked directory keeps its project-tree symlink filter during mismatch checks."""
    directory = tmp_path / "tracked"
    directory.mkdir()
    tracked_file = directory / "tracked.txt"
    tracked_file.write_text("original")
    tree_hash, manifest = cache.hash_directory(directory)
    expected = types.DirHash(hash=tree_hash, manifest=manifest)

    (directory / "ignored-link.txt").symlink_to(tracked_file)
    tracked_file.write_text("changed")

    assert not checkout_mod._on_disk_matches_expected(directory, expected)


def _helper_cache_directory(
    path: pathlib.Path,
    contents: dict[str, str],
    checkout_mode: config.CheckoutMode = config.CheckoutMode.COPY,
) -> track.PvtData:
    path.mkdir(parents=True)
    for relpath, content in contents.items():
        file_path = path / relpath
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
    output_hash = cache.save_to_cache(
        path, config.get_cache_dir() / "files", checkout_mode=checkout_mode
    )
    assert "manifest" in output_hash
    return track.PvtData(
        path="ext",
        hash=output_hash["hash"],
        size=sum(len(content) for content in contents.values()),
        num_files=len(contents),
        manifest=output_hash["manifest"],
    )


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
    new_pvt_data = _helper_cache_directory(
        root / "new-ext",
        {"a.txt": "one updated\n", "c.txt": "three\n"},
    )
    track.write_pvt_file(track.get_pvt_path(data_dir), new_pvt_data)

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
    new_pvt_data = _helper_cache_directory(
        root / "new-ext",
        {"a.txt": "one updated\n", "c.txt": "three\n"},
    )
    track.write_pvt_file(track.get_pvt_path(data_dir), new_pvt_data)

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
    new_pvt_data = _helper_cache_directory(
        root / "new-ext",
        {"a.txt": "one updated\n", "c.txt": "three\n"},
    )
    track.write_pvt_file(track.get_pvt_path(data_dir), new_pvt_data)
    (data_dir / "b.txt").write_text("two EDITED\n")

    result = runner.invoke(cli.cli, ["checkout", "--only-missing", "data/ext"])

    assert result.exit_code != 0
    assert "'ext' has local changes that are not in the cache: b.txt. Use --force" in result.output
    assert (data_dir / "a.txt").read_text() == "one\n"
    assert (data_dir / "b.txt").read_text() == "two EDITED\n"
    assert not (data_dir / "c.txt").exists(), "A refused checkout must not restore anything"


def test_checkout_only_missing_keeps_ignored_entries_when_replacing_directory(
    mock_discovery: PipelineLike, runner: click.testing.CliRunner
) -> None:
    """Entries the directory hash skips are not part of the output: the replaced tree keeps them."""
    root = project.get_project_root()
    data_dir = _helper_track_directory(root)
    new_pvt_data = _helper_cache_directory(
        root / "new-ext",
        {"a.txt": "one updated\n", "c.txt": "three\n"},
    )
    track.write_pvt_file(track.get_pvt_path(data_dir), new_pvt_data)
    (data_dir / "build").mkdir()
    (data_dir / "build" / "x.o").write_text("object\n")
    (data_dir / ".a.txt.swp").write_text("swap\n")
    (data_dir / "draft.txt~").write_text("backup\n")

    result = runner.invoke(cli.cli, ["checkout", "--only-missing", "data/ext"])

    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in data_dir.iterdir()) == [
        ".a.txt.swp",
        "a.txt",
        "build",
        "c.txt",
        "draft.txt~",
    ]
    assert (data_dir / "a.txt").read_text() == "one updated\n"
    assert (data_dir / "build" / "x.o").read_text() == "object\n"
    assert (data_dir / ".a.txt.swp").read_text() == "swap\n"
    assert (data_dir / "draft.txt~").read_text() == "backup\n"


def test_checkout_only_missing_symlink_mode_keeps_editor_swap_file(
    mock_discovery: PipelineLike, runner: click.testing.CliRunner
) -> None:
    """With a cached tree to link to, the swap still keeps an open editor's swap file."""
    root = project.get_project_root()
    data_dir = _helper_track_directory(root)
    new_pvt_data = _helper_cache_directory(
        root / "new-ext",
        {"a.txt": "one updated\n", "c.txt": "three\n"},
        checkout_mode=config.CheckoutMode.SYMLINK,
    )
    track.write_pvt_file(track.get_pvt_path(data_dir), new_pvt_data)
    (data_dir / ".a.txt.swp").write_text("swap\n")

    result = runner.invoke(
        cli.cli, ["checkout", "--only-missing", "--checkout-mode", "symlink", "data/ext"]
    )

    assert result.exit_code == 0, result.output
    assert not data_dir.is_symlink(), "a directory that keeps local entries cannot link the cache"
    assert (data_dir / ".a.txt.swp").read_text() == "swap\n"
    assert (data_dir / "a.txt").read_text() == "one updated\n"
    assert (data_dir / "c.txt").read_text() == "three\n"
    assert not (data_dir / "b.txt").exists()


def _helper_kept_entries_layout(root: pathlib.Path) -> tuple[pathlib.Path, types.DirHash]:
    """data/ext at v1 plus build/x.o and .a.txt.swp (skipped by the hash); v2 in the cache."""
    cache_dir = root / ".pivot" / "cache" / "files"
    v2 = root / "v2src"
    v2.mkdir(parents=True)
    (v2 / "a.txt").write_text("one updated\n")
    (v2 / "c.txt").write_text("three\n")
    v2_hash = cache.save_to_cache(v2, cache_dir, checkout_mode=config.CheckoutMode.COPY)
    assert "manifest" in v2_hash
    ext = root / "data" / "ext"
    ext.mkdir(parents=True)
    (ext / "a.txt").write_text("one\n")
    (ext / "b.txt").write_text("two\n")
    _ = cache.save_to_cache(ext, cache_dir, checkout_mode=config.CheckoutMode.COPY)
    (ext / "build").mkdir()
    (ext / "build" / "x.o").write_text("object\n")
    (ext / ".a.txt.swp").write_text("swap\n")
    return ext, v2_hash


def _helper_restore_only_missing(root: pathlib.Path, v2_hash: types.DirHash) -> str:
    """Run the per-target restore `pivot checkout --only-missing` runs for data/ext."""
    result, _ = checkout_mod._restore_path_sync(
        root / "data" / "ext",
        v2_hash,
        root / ".pivot" / "cache" / "files",
        [cache.CheckoutMode.COPY],
        checkout_mod.CheckoutBehavior.SKIP_EXISTING,
        root / ".pivot",
    )
    return result


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


def test_restore_killed_at_any_rename_loses_no_kept_entry(tmp_path: pathlib.Path) -> None:
    """A kill between any two renames of the swap leaves kept entries where the next restore
    finds them, even after leftovers age past the stale-temp cutoff."""
    ctx = multiprocessing.get_context("spawn")
    crashes = 0
    for crash_at in range(20):
        root = tmp_path / f"crash-{crash_at}"
        _, v2_hash = _helper_kept_entries_layout(root)
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
        where = sorted(
            str(p.relative_to(root)) for p in root.rglob("*") if p.name in ("x.o", ".a.txt.swp")
        )
        assert (ext / "build" / "x.o").read_text() == "object\n", (
            f"rename {crash_at}: x.o at {where}"
        )
        assert (ext / ".a.txt.swp").read_text() == "swap\n", f"rename {crash_at}: .swp at {where}"
        assert (ext / "a.txt").read_text() == "one updated\n", f"rename {crash_at}"
        assert not (ext / "b.txt").exists(), f"rename {crash_at}"
        assert [
            p.name for p in (root / "data").iterdir() if p.name.startswith(".pivot_keep_")
        ] == []
    else:
        raise AssertionError("the restore never completed without a crash")
    assert crashes >= 4, f"only {crashes} crash points exercised"


def test_restore_without_carrying_puts_entries_staged_by_a_killed_checkout_into_the_new_tree(
    tmp_path: pathlib.Path,
) -> None:
    """After a killed `checkout --only-missing`, a restore that carries nothing (as `pivot repro`
    restores a stale output) still puts the staged entries into the tree it writes."""
    ext, v2_hash = _helper_kept_entries_layout(tmp_path)
    child = multiprocessing.get_context("spawn").Process(
        target=_helper_crash_at_rename, args=(tmp_path, v2_hash, 2)
    )
    child.start()
    child.join(timeout=120)
    assert child.exitcode == _CRASH_EXIT
    assert cache.staged_kept_entries(ext) is not None, "the kill must leave entries staged"

    restored = cache.restore_from_cache(
        ext,
        v2_hash,
        tmp_path / ".pivot" / "cache" / "files",
        checkout_modes=[cache.CheckoutMode.COPY],
        state_dir=tmp_path / ".pivot",
    )

    assert restored
    assert sorted(p.name for p in ext.iterdir()) == [".a.txt.swp", "a.txt", "build", "c.txt"]
    assert (ext / "build" / "x.o").read_text() == "object\n"
    assert (ext / ".a.txt.swp").read_text() == "swap\n"
    assert (ext / "a.txt").read_text() == "one updated\n"
    assert cache.staged_kept_entries(ext) is None


def test_checkout_only_missing_reads_no_content_of_unchanged_hardlinked_directory(
    mock_discovery: PipelineLike, runner: click.testing.CliRunner, mocker: MockerFixture
) -> None:
    """A workspace file that is a hardlink to its target blob is unchanged without a read."""
    data_dir = project.get_project_root() / "data" / "ext"
    pvt_data = _helper_cache_directory(
        data_dir, {"a.txt": "one\n", "b.txt": "two\n"}, checkout_mode=config.CheckoutMode.HARDLINK
    )
    track.write_pvt_file(track.get_pvt_path(data_dir), pvt_data)
    assert (data_dir / "a.txt").stat().st_nlink == 2
    hash_file = mocker.spy(cache, "hash_file")

    result = runner.invoke(cli.cli, ["checkout", "--only-missing", "data/ext"])

    assert result.exit_code == 0, result.output
    assert "Skipped 1 file(s)" in result.output
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
    assert "Skipped 1 file(s)" in result.output
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
    manifest_entry = types.DirManifestEntry(
        relpath=".pivot/escape/new.txt",
        hash=file_hash,
        size=3,
        isexec=False,
    )
    track.write_pvt_file(
        track.get_pvt_path(data_dir),
        track.PvtData(
            path="ext",
            hash="aaaaaaaaaaaaaaaa",
            size=3,
            num_files=1,
            manifest=[manifest_entry],
        ),
    )

    result = runner.invoke(cli.cli, ["checkout", "--only-missing", "data/ext"])

    assert result.exit_code != 0
    assert "Manifest contains path traversal" in result.output
    assert not (outside / "new.txt").exists()
