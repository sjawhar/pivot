"""A file-hash entry is also found by the file's identity, (st_dev, st_ino), and that match is
never weaker than the path match it stands in for (Dispatch BET-263)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pytest

from pivot.storage import cache, state
from pivot.types import DeferredWrites

if TYPE_CHECKING:
    import pathlib

    from pytest_mock import MockerFixture


def _helper_record(state_dir: pathlib.Path, path: pathlib.Path) -> str:
    """Hash ``path`` through a writable StateDB, which records its entry."""
    with state.StateDB(state_dir) as state_db:
        file_hash, _ = cache.hash_file(path, state_db)
    return file_hash


def _helper_file(path: pathlib.Path, content: bytes) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _helper_stat(
    real: os.stat_result,
    *,
    ino: int,
    dev: int,
    size: int | None = None,
    mtime_ns: int | None = None,
) -> os.stat_result:
    """``real`` with another inode number and device (and optionally size and mtime): what pivot
    sees of a file whose inode number the filesystem took back from a deleted file."""
    fields = list(real)
    fields[1] = ino
    fields[2] = dev
    if size is not None:
        fields[6] = size
    return os.stat_result(
        fields,
        {
            "st_atime_ns": real.st_atime_ns,
            "st_mtime_ns": real.st_mtime_ns if mtime_ns is None else mtime_ns,
            "st_ctime_ns": real.st_ctime_ns,
        },
    )


def test_a_hard_link_is_served_by_identity_without_reading_it(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """A hard link at another path is the recorded inode: its hash comes from the original's
    entry, for one file and for a directory, and no content is read."""
    state_dir = tmp_path / ".pivot"
    live = _helper_file(tmp_path / "live" / "tree" / "f.txt", b"content")
    live_hash = _helper_record(state_dir, live)
    replica = tmp_path / "replica" / "tree" / "f.txt"
    replica.parent.mkdir(parents=True)
    os.link(live, replica)
    reads = mocker.spy(cache, "_hash_contents")

    with state.StateDB(state_dir, readonly=True) as state_db:
        assert cache.hash_file(replica, state_db)[0] == live_hash
        manifest = cache.hash_directory(replica.parent, state_db)["manifest"]

    assert [entry["hash"] for entry in manifest] == [live_hash]
    assert reads.call_count == 0, "A hard link of a recorded file is not read"


@pytest.mark.parametrize(
    ("size_delta", "mtime_delta"),
    [
        pytest.param(0, 0, id="same-size-and-mtime"),
        pytest.param(0, 1, id="other-mtime"),
        pytest.param(1, 0, id="other-size"),
    ],
)
def test_a_reused_inode_number_never_gets_the_deleted_files_hash(
    tmp_path: pathlib.Path, size_delta: int, mtime_delta: int
) -> None:
    """Once the recorded file is deleted, the filesystem may give its inode number to a new
    file with other content. That file is never served the old hash: not when its size or
    mtime differs, and not when a copy that keeps mtimes (`cp -a`, `rsync -a`, `shutil.copy2`)
    gives it the same size and mtime, because no recorded path names its inode any more."""
    state_dir = tmp_path / ".pivot"
    live = _helper_file(tmp_path / "live" / "f.txt", b"content")
    live_hash = _helper_record(state_dir, live)
    link = tmp_path / "replica" / "f.txt"
    link.parent.mkdir()
    os.link(live, link)
    with state.StateDB(state_dir, readonly=True) as state_db:
        assert state_db.get(link, link.stat()) == live_hash, "The identity lookup is in use"
    old = live.stat()
    link.unlink()
    live.unlink()
    new = _helper_file(tmp_path / "new" / "g.txt", b"CONTENT")
    os.utime(new, ns=(old.st_atime_ns, old.st_mtime_ns))
    reused = _helper_stat(
        new.stat(),
        ino=old.st_ino,
        dev=old.st_dev,
        size=old.st_size + size_delta,
        mtime_ns=old.st_mtime_ns + mtime_delta,
    )

    with state.StateDB(state_dir, readonly=True) as state_db:
        assert state_db.get(new, reused) is None
        assert state_db.get_many([(new, reused)]) == {new: None}


def test_an_inode_number_on_another_device_is_not_the_recorded_file(
    tmp_path: pathlib.Path,
) -> None:
    """The identity is the device and the inode number together: a file on another
    filesystem with the recorded inode number, size and mtime is not the recorded file."""
    state_dir = tmp_path / ".pivot"
    live = _helper_file(tmp_path / "live" / "f.txt", b"content")
    _helper_record(state_dir, live)
    twin = _helper_file(tmp_path / "other" / "f.txt", b"CONTENT")
    recorded = live.stat()
    on_another_device = _helper_stat(
        twin.stat(), ino=recorded.st_ino, dev=recorded.st_dev + 1, mtime_ns=recorded.st_mtime_ns
    )

    with state.StateDB(state_dir, readonly=True) as state_db:
        assert state_db.get(twin, on_another_device) is None


@pytest.mark.parametrize(
    ("content", "mtime_delta"),
    [
        pytest.param(b"CONTENT", 1, id="same-size-later-mtime"),
        pytest.param(b"content, longer", 0, id="longer-same-mtime"),
    ],
)
def test_a_hard_link_changed_in_place_is_not_served_the_recorded_hash(
    tmp_path: pathlib.Path, content: bytes, mtime_delta: int
) -> None:
    """A write through a replica's hard link changes the one inode both paths name. The
    recorded path still names it, so only the stat match against the recorded entry refuses
    the old hash."""
    state_dir = tmp_path / ".pivot"
    live = _helper_file(tmp_path / "live" / "f.txt", b"content")
    live_hash = _helper_record(state_dir, live)
    link = tmp_path / "replica" / "f.txt"
    link.parent.mkdir()
    os.link(live, link)
    with state.StateDB(state_dir, readonly=True) as state_db:
        assert state_db.get(link, link.stat()) == live_hash, "The identity lookup is in use"
    recorded = live.stat()
    with link.open("r+b") as f:
        f.write(content)
    os.utime(link, ns=(recorded.st_atime_ns, recorded.st_mtime_ns + mtime_delta))

    with state.StateDB(state_dir, readonly=True) as state_db:
        assert state_db.get(link, link.stat()) is None
        assert state_db.get_many([(link, link.stat())]) == {link: None}
        assert cache.hash_file(link, state_db)[0] != live_hash


def test_a_recorded_path_that_now_names_another_inode_vouches_for_nothing(
    tmp_path: pathlib.Path,
) -> None:
    """The recorded path was replaced by a new file while the old inode lived on elsewhere: a
    file that then gets the old inode number, size and mtime is refused, because the recorded
    path names another inode."""
    state_dir = tmp_path / ".pivot"
    live = _helper_file(tmp_path / "live" / "f.txt", b"content")
    _helper_record(state_dir, live)
    old = live.stat()
    keep = tmp_path / "keep.txt"
    os.link(live, keep)
    replacement = _helper_file(tmp_path / "live" / "f.new", b"CONTENT")
    os.replace(replacement, live)
    keep.unlink()
    new = _helper_file(tmp_path / "new" / "g.txt", b"CONTENT")
    reused = _helper_stat(new.stat(), ino=old.st_ino, dev=old.st_dev, mtime_ns=old.st_mtime_ns)

    with state.StateDB(state_dir, readonly=True) as state_db:
        assert state_db.get(new, reused) is None
        assert state_db.get_many([(new, reused)]) == {new: None}


def test_a_recorded_path_that_now_names_another_devices_inode_vouches_for_nothing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check (2) compares the device as well as the inode number: a recorded path that now
    names the same inode number on another filesystem (a mount over its directory, a volume
    remounted in a container) is another file, so it vouches for nothing."""
    state_dir = tmp_path / ".pivot"
    live = _helper_file(tmp_path / "live" / "f.txt", b"content")
    _helper_record(state_dir, live)
    recorded = live.stat()
    other = _helper_file(tmp_path / "new" / "g.txt", b"CONTENT")
    reused = _helper_stat(
        other.stat(), ino=recorded.st_ino, dev=recorded.st_dev, mtime_ns=recorded.st_mtime_ns
    )
    real_stat = os.stat

    def _stat(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        result = real_stat(path, *args, **kwargs)
        if os.fsdecode(path) == str(live):
            return _helper_stat(result, ino=recorded.st_ino, dev=recorded.st_dev + 1)
        return result

    monkeypatch.setattr(os, "stat", _stat)

    with state.StateDB(state_dir, readonly=True) as state_db:
        assert state_db.get(other, reused) is None
        assert state_db.get_many([(other, reused)]) == {other: None}


def test_rewriting_a_path_drops_its_old_inodes_index_entry(tmp_path: pathlib.Path) -> None:
    """When a path's entry moves to a new inode, the old inode's index entry goes too, so the
    index never holds more entries than the path entries it points at."""
    state_dir = tmp_path / ".pivot"
    path = _helper_file(tmp_path / "f.txt", b"first")
    _helper_record(state_dir, path)
    old = path.stat()
    replacement = _helper_file(tmp_path / "f.new", b"second")
    os.replace(replacement, path)
    _helper_record(state_dir, path)
    new = path.stat()

    with state.StateDB(state_dir, readonly=True) as state_db:
        index = dict(state_db.iter_prefix(b"inode:"))
    assert len(index) == 1
    assert old.st_ino != new.st_ino
    assert list(index.values()) == [b"hash:" + os.fsencode(path)]


def test_one_inode_number_on_two_devices_keeps_an_index_entry_for_each(
    tmp_path: pathlib.Path,
) -> None:
    """The index key holds the device as well as the inode number: an entry for the recorded
    inode number on another filesystem takes nothing from the recorded file, whose hard link
    is still served by identity."""
    state_dir = tmp_path / ".pivot"
    live = _helper_file(tmp_path / "live" / "f.txt", b"content")
    live_hash = _helper_record(state_dir, live)
    recorded = live.stat()
    elsewhere = str(tmp_path / "mnt" / "f.txt")
    with state.StateDB(state_dir) as state_db:
        state_db.save_file_hash_entries(
            [
                (
                    elsewhere,
                    recorded.st_mtime_ns,
                    recorded.st_size,
                    recorded.st_dev + 1,
                    recorded.st_ino,
                    "0" * 16,
                )
            ]
        )
    link = tmp_path / "replica" / "f.txt"
    link.parent.mkdir()
    os.link(live, link)

    with state.StateDB(state_dir, readonly=True) as state_db:
        index = dict(state_db.iter_prefix(b"inode:"))
        assert state_db.get(link, link.stat()) == live_hash
    assert sorted(index.values()) == sorted(
        [b"hash:" + os.fsencode(live), b"hash:" + os.fsencode(elsewhere)]
    )


def test_rewriting_a_path_keeps_its_old_inodes_index_entry_that_names_another_link(
    tmp_path: pathlib.Path,
) -> None:
    """A path that moves to a new inode deletes the old inode's index entry only when that
    entry points at the path itself: one that points at another link of the old inode stays,
    so a further link of it is still served by identity."""
    state_dir = tmp_path / ".pivot"
    live = _helper_file(tmp_path / "live" / "f.txt", b"content")
    live_hash = _helper_record(state_dir, live)
    link = tmp_path / "replica" / "f.txt"
    link.parent.mkdir()
    os.link(live, link)
    with state.StateDB(state_dir) as state_db:
        state_db.save_many([(link, link.stat(), live_hash)])
    replacement = _helper_file(tmp_path / "live" / "f.new", b"rewritten")
    os.replace(replacement, live)
    _helper_record(state_dir, live)
    another = tmp_path / "another" / "f.txt"
    another.parent.mkdir()
    os.link(link, another)

    with state.StateDB(state_dir, readonly=True) as state_db:
        assert state_db.get(another, another.stat()) == live_hash


def test_an_output_read_then_restored_in_one_batch_leaves_one_index_entry(
    tmp_path: pathlib.Path,
) -> None:
    """A repro that restores an edited output from the run cache reads the edited file,
    restores the output as a new inode, and the worker hands both entries to the coordinator
    in one batch (`worker._try_skip_via_run_cache_with_store`). The edited inode keeps no index
    entry: the index holds one entry, for the restored file, and does not grow with each edit
    and restore."""
    state_dir = tmp_path / ".pivot"
    output = _helper_file(tmp_path / "data" / "out.txt", b"content")
    output_hash = _helper_record(state_dir, output)
    os.replace(_helper_file(tmp_path / "data" / "out.edit", b"edited"), output)
    edited_hash, edited = cache.hash_file(output)
    restored = _helper_file(tmp_path / "data" / "out.restore", b"content")
    os.replace(restored, output)
    batch = cache.deferred_file_hash_entries(
        [(output, edited, edited_hash), (output, output.stat(), output_hash)]
    )

    with state.StateDB(state_dir) as state_db:
        state_db.apply_deferred_writes("syn/make", [], DeferredWrites(file_hash_entries=batch))

    with state.StateDB(state_dir, readonly=True) as state_db:
        index = dict(state_db.iter_prefix(b"inode:"))
        assert state_db.get(output, output.stat()) == output_hash
    assert edited.st_ino != output.stat().st_ino
    assert list(index.values()) == [b"hash:" + os.fsencode(output)]


def test_a_batch_that_moves_one_link_to_a_new_inode_keeps_the_other_links_hint(
    tmp_path: pathlib.Path,
) -> None:
    """One batch records two links of an inode, Q then P, and then P restored as a new inode.
    The old inode's index entry stays with Q, which still names it, so a further link of it is
    served by identity."""
    state_dir = tmp_path / ".pivot"
    q = _helper_file(tmp_path / "q" / "f.txt", b"content")
    p = tmp_path / "p" / "f.txt"
    p.parent.mkdir()
    os.link(q, p)
    content_hash, linked = cache.hash_file(q)
    restored = _helper_file(tmp_path / "p" / "f.restore", b"content")
    os.replace(restored, p)
    batch = cache.deferred_file_hash_entries(
        [(q, linked, content_hash), (p, linked, content_hash), (p, p.stat(), content_hash)]
    )
    with state.StateDB(state_dir) as state_db:
        state_db.apply_deferred_writes("syn/make", [], DeferredWrites(file_hash_entries=batch))
    another = tmp_path / "another" / "f.txt"
    another.parent.mkdir()
    os.link(q, another)

    with state.StateDB(state_dir, readonly=True) as state_db:
        index = dict(state_db.iter_prefix(b"inode:"))
        assert state_db.get(another, another.stat()) == content_hash
    assert sorted(index.values()) == sorted([b"hash:" + os.fsencode(q), b"hash:" + os.fsencode(p)])
