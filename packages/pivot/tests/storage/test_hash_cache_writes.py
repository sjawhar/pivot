"""Hash-cache writes are batched, and hash-cache keys come from the path as given."""

from __future__ import annotations

import os
import pathlib
import stat as stat_mod

import lmdb
import pytest
import xxhash

from pivot.cli import checkout as checkout_mod
from pivot.storage import cache, ownership
from pivot.storage import state as state_mod
from pivot.types import FileHash


def _helper_committed_write_txns(state_dir: pathlib.Path) -> int:
    env = lmdb.open(str(state_dir / "state.lmdb"), readonly=True, lock=False)
    try:
        return int(env.info()["last_txnid"])
    finally:
        env.close()


def test_hash_directory_commits_one_write_transaction(tmp_path: pathlib.Path) -> None:
    """A cold hash of a 50-file directory records every file hash in one write transaction."""
    tree = tmp_path / "tree"
    tree.mkdir()
    for index in range(50):
        (tree / f"f{index:02d}.txt").write_text(str(index))
    state_dir = tmp_path / ".pivot"
    with state_mod.StateDB(state_dir):
        pass
    before = _helper_committed_write_txns(state_dir)

    with state_mod.StateDB(state_dir) as state_db:
        cache.hash_directory(tree, state_db)

    assert _helper_committed_write_txns(state_dir) - before == 1
    with state_mod.StateDB(state_dir, readonly=True) as state_db:
        cached = state_db.get_many([(path, path.stat()) for path in sorted(tree.iterdir())])
    assert all(value is not None for value in cached.values()), "every file hash is recorded"


def test_hash_file_caches_a_non_utf8_name(tmp_path: pathlib.Path) -> None:
    """A file whose name is not UTF-8 is hashed and its hash cached like any other."""
    path = tmp_path / os.fsdecode(b"\xff.bin")
    path.write_bytes(b"payload")

    with state_mod.StateDB(tmp_path / ".pivot") as state_db:
        file_hash, file_stat = cache.hash_file(path, state_db)
        assert state_db.get(path, file_stat) == file_hash


def test_hash_through_a_retargeted_symlink_follows_the_new_target(
    tmp_path: pathlib.Path,
) -> None:
    """Regression guard: a symlink's cached hash never outlives a change of its target."""
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first")
    second.write_text("second, longer")
    link = tmp_path / "link.txt"
    link.symlink_to(first)

    with state_mod.StateDB(tmp_path / ".pivot") as state_db:
        first_hash, _ = cache.hash_file(link, state_db)
        link.unlink()
        link.symlink_to(second)
        second_hash, _ = cache.hash_file(link, state_db)

    assert first_hash == cache.hash_file(first)[0]
    assert second_hash == cache.hash_file(second)[0]


def _helper_twin_files(directory_a: pathlib.Path, directory_b: pathlib.Path) -> None:
    """``f.txt`` in each directory: same size and mtime, different content and inode."""
    for directory, content in ((directory_a, b"aaaa"), (directory_b, b"bbbb")):
        directory.mkdir()
        (directory / "f.txt").write_bytes(content)
        os.utime(directory / "f.txt", ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))


@pytest.mark.parametrize(
    "link_the_directory",
    [
        pytest.param(False, id="file-link"),
        pytest.param(True, id="parent-directory-link"),
    ],
)
def test_a_link_retargeted_on_the_same_filesystem_never_reuses_the_old_targets_hash(
    tmp_path: pathlib.Path, link_the_directory: bool
) -> None:
    """The key is the path as given, not the file it resolves to. A link (to the file, or to a
    directory above it) pointed at another file of the same size and mtime on this filesystem
    still reads the new content: the entry records the inode ``stat`` reaches through the link.
    The next test covers a target on another filesystem, where an inode number can repeat."""
    _helper_twin_files(tmp_path / "a", tmp_path / "b")
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "a" if link_the_directory else tmp_path / "a" / "f.txt")
    path = link / "f.txt" if link_the_directory else link

    with state_mod.StateDB(tmp_path / ".pivot") as state_db:
        first_hash, _ = cache.hash_file(path, state_db)
        link.unlink()
        link.symlink_to(tmp_path / "b" if link_the_directory else tmp_path / "b" / "f.txt")
        second_hash, _ = cache.hash_file(path, state_db)

    assert first_hash == cache.hash_file(tmp_path / "a" / "f.txt")[0]
    assert second_hash == cache.hash_file(tmp_path / "b" / "f.txt")[0], (
        "The hash read through the old target must not be reused"
    )


def test_a_link_re_pointed_at_another_filesystem_reads_the_new_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory symlink re-pointed at another filesystem whose file repeats the old file's
    inode number, size and mtime_ns: only the device tells them apart. CI cannot mount a second
    filesystem, so the new target's stat is made to repeat the old one's on another device."""
    _helper_twin_files(tmp_path / "a", tmp_path / "b")
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "a")
    path = link / "f.txt"
    old_stat = (tmp_path / "a" / "f.txt").stat()
    real_stat = pathlib.Path.stat

    def stat_as_another_mount(
        self: pathlib.Path, *, follow_symlinks: bool = True
    ) -> os.stat_result:
        found = real_stat(self, follow_symlinks=follow_symlinks)
        if self != path or not link.resolve().samefile(tmp_path / "b"):
            return found
        visible = list(old_stat)
        visible[stat_mod.ST_DEV] = old_stat.st_dev + 1
        hidden = {
            name: getattr(old_stat, name)
            for name in (
                "st_atime",
                "st_mtime",
                "st_ctime",
                "st_atime_ns",
                "st_mtime_ns",
                "st_ctime_ns",
            )
        }
        return os.stat_result(visible, hidden)

    with state_mod.StateDB(tmp_path / ".pivot") as state_db:
        first_hash, _ = cache.hash_file(path, state_db)
        link.unlink()
        link.symlink_to(tmp_path / "b")
        monkeypatch.setattr(pathlib.Path, "stat", stat_as_another_mount)
        second_hash, second_stat = cache.hash_file(path, state_db)

    assert (second_stat.st_ino, second_stat.st_size, second_stat.st_mtime_ns) == (
        old_stat.st_ino,
        old_stat.st_size,
        old_stat.st_mtime_ns,
    )
    assert first_hash == xxhash.xxh64(b"aaaa").hexdigest()
    assert second_hash == xxhash.xxh64(b"bbbb").hexdigest(), (
        "The old target's hash must not be served for a file on another filesystem"
    )


def test_hash_file_with_pending_collects_instead_of_saving(tmp_path: pathlib.Path) -> None:
    """With ``pending``, a hash read from the file is handed back for the caller to save."""
    path = tmp_path / "data.txt"
    path.write_text("data")
    pending = list[tuple[pathlib.Path, os.stat_result, str]]()

    with state_mod.StateDB(tmp_path / ".pivot") as state_db:
        file_hash, file_stat = cache.hash_file(path, state_db, pending=pending)
        assert state_db.get(path, file_stat) is None, "Nothing is saved yet"
        assert pending == [(path, file_stat, file_hash)]
        state_db.save_many(pending)
        assert state_db.get(path, file_stat) == file_hash


def test_a_write_just_after_a_copy_restore_is_not_recorded_as_the_cached_content(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checkout records a restored file with the stat of the staged copy, taken before the
    rename puts it in place (the same inode). A write landing just after the rename then
    changes the stat, so the entry does not vouch for the written bytes."""
    content = b"cached"
    cache_dir = tmp_path / ".pivot" / "cache" / "files"
    file_hash = xxhash.xxh64(content).hexdigest()
    cache_object = cache.get_cache_path(cache_dir, file_hash)
    cache_object.parent.mkdir(parents=True)
    cache_object.write_bytes(content)
    target = tmp_path / "out.txt"
    real_replace = pathlib.Path.replace

    def replace_then_write(self: pathlib.Path, destination: str | os.PathLike[str]) -> pathlib.Path:
        moved = real_replace(self, destination)
        if pathlib.Path(destination) == target:
            target.write_bytes(b"edited")
        return moved

    monkeypatch.setattr(pathlib.Path, "replace", replace_then_write)

    batch = state_mod.FileHashBatch()
    outcome = checkout_mod._restore_path_sync(
        target,
        checkout_mod.Expected(FileHash(hash=file_hash)),
        cache_dir,
        [cache.CheckoutMode.COPY],
        checkout_mod.CheckoutBehavior.ERROR,
        tmp_path / ".pivot",
        target=ownership.ForcedRestore(
            replacement=ownership.tracked_restore(target),
            restored=FileHash(hash=file_hash),
            claims=None,
        ),
        batch=batch,
    )
    batch.save()

    assert outcome.result == "restored"
    assert target.read_bytes() == b"edited"
    with state_mod.StateDB(tmp_path / ".pivot", readonly=True) as state_db:
        assert state_db.get(target, target.stat()) is None, (
            "The edited file must be read again, not taken for the cached content"
        )
