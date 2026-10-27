"""The .pvt walk's directory memo: a settled, unchanged directory is stat'ed, not listed.

main's walk (``os.walk`` over the whole project, reading every ``.pvt``) is the oracle: after every
change, cold and warm, the memo walk must find exactly what main finds, in main's order.
"""

from __future__ import annotations

import builtins
import contextlib
import errno
import fcntl
import logging
import os
import pathlib
import random
import shutil
import socket
import stat
import threading
import time
from typing import TYPE_CHECKING

import pytest

from pivot import config, exceptions, project
from pivot.config import io as config_io
from pivot.executor import core as executor_core
from pivot.storage import cache, state, track

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from pytest_mock import MockerFixture

_REAL_TIME_NS = time.time_ns
_NS_PER_S = 1_000_000_000
_PAST_SETTLE_WINDOW_NS = 10_000_000_000
_INSIDE_SETTLE_WINDOW_NS = 1_000_000_000

# A change this long after a walk carries a later timestamp than the walk saw, on any kernel
# timestamp tick (one jiffy, at most 10 ms). The walk's settle window guarantees that gap in real
# use; a test that moves the clock past the window waits the tick itself.
_ONE_TICK_S = 0.05

# How long a test waits for a walk that must not block (on a FIFO, or on a lock another holds).
_NO_WAIT_S = 10.0


@pytest.fixture
def new_project(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """A project root on a filesystem the memo trusts (mountinfo says ext4), with no state
    database yet."""
    root = tmp_path / "project"
    (root / ".pivot").mkdir(parents=True)
    monkeypatch.setattr(project, "_project_root_cache", root)
    _helper_mountinfo(monkeypatch, tmp_path / "mountinfo", root, "ext4")
    return root


@pytest.fixture
def memo_project(new_project: pathlib.Path) -> pathlib.Path:
    """``new_project`` once a pivot command has made its state database, whose directory holds
    the memo file."""
    _helper_create_state_database()
    return new_project


def _helper_mount_line(mount_id: int, device: int, point: str, fstype: str) -> str:
    """One /proc/self/mountinfo line, with optional fields and the escaping the kernel uses."""
    escaped = point.replace("\\", "\\134").replace(" ", "\\040")
    return (
        f"{mount_id} 1 {os.major(device)}:{os.minor(device)} / {escaped} "
        + f"rw,relatime shared:{mount_id} master:3 - {fstype} /dev/sdz1 rw,errors=continue"
    )


def _helper_write_mountinfo(
    monkeypatch: pytest.MonkeyPatch, path: pathlib.Path, lines: list[str]
) -> None:
    decoy = _helper_mount_line(90, os.makedev(0, 4242), "/mnt/other", "nfs4")
    path.write_text("\n".join([decoy, *lines]) + "\n")
    monkeypatch.setattr(track, "_MOUNTINFO", path)


def _helper_mountinfo(
    monkeypatch: pytest.MonkeyPatch, path: pathlib.Path, root: pathlib.Path, fstype: str | None
) -> None:
    """Point the walk at a mountinfo that mounts root's device as ``fstype`` (None: not at all)."""
    device = os.stat(root).st_dev
    lines = list[str]()
    if fstype is not None:
        lines.append(_helper_mount_line(91, device, "/mnt/with space", fstype))
    _helper_write_mountinfo(monkeypatch, path, lines)


def _helper_settle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Move the clock past the settle window, so every directory made so far is settled."""
    monkeypatch.setattr(time, "time_ns", lambda: _REAL_TIME_NS() + _PAST_SETTLE_WINDOW_NS)


def _helper_track(path: pathlib.Path, content: str) -> None:
    """Write ``path`` and the .pvt that ``pivot track`` records for it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    file_hash, _ = cache.hash_file(path)
    track.write_pvt_file(
        track.get_pvt_path(path), track.PvtData(path=path.name, hash=file_hash, size=len(content))
    )


def _helper_main_walk(root: pathlib.Path) -> dict[str, track.PvtData]:
    """main's discover_pvt_files, copied on purpose: os.walk(followlinks=False) over everything.

    The oracle the memo walk must match.
    """
    result = dict[str, track.PvtData]()
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        for fname in filenames:
            if not fname.endswith(".pvt"):
                continue
            pvt_path = pathlib.Path(dirpath) / fname
            if not pvt_path.is_file():
                continue
            data = track.read_pvt_file(pvt_path)
            if data is None:
                continue
            result[str(project.normalize_path(pvt_path.parent / data["path"]))] = data
    return result


def _helper_directories(root: pathlib.Path) -> list[str]:
    """Every directory the walk visits, root first, as absolute paths."""
    return [dirpath for dirpath, _dirnames, _filenames in os.walk(root, followlinks=False)]


def _helper_relative_directories(root: pathlib.Path) -> set[str]:
    """Every directory the walk visits, as the memo keys it (the root is the empty path)."""
    relative = {os.path.relpath(path, root) for path in _helper_directories(root)}
    return {"" if rel == "." else rel for rel in relative}


def _helper_preflight(root: pathlib.Path) -> dict[str, track.PvtData]:
    """The repro/run pre-flight's walk, the memo's one writer."""
    return track.discover_pvt_files_and_update_memo(root)


def _helper_record(root: pathlib.Path) -> None:
    """Record the memo for an unchanging tree.

    The first pre-flight creates the memo file, which changes ``.pivot/state.lmdb``; the second
    records that directory as it then stays (the file is rewritten in place).
    """
    _helper_preflight(root)
    _helper_preflight(root)


def _helper_memo() -> track._Memo:
    """What the account's memo file holds; no root and no listing when there is no whole memo."""
    path = track._memo_path()
    memo = track._unpack_memo(path.read_bytes()) if path.exists() else None
    return memo if memo is not None else track._Memo(root="", dirs={})


def _helper_write_memo(memo: track._Memo) -> None:
    """Write ``memo`` over the account's memo file, keeping the file."""
    track._memo_path().write_bytes(track._pack_memo(memo["root"], memo["dirs"]))


def _helper_assert_matches_main(found: dict[str, track.PvtData], root: pathlib.Path) -> None:
    expected = _helper_main_walk(root)
    assert found == expected, "the memo walk finds exactly what main's walk finds"
    assert list(found) == list(expected), "in main's order"


def _helper_build_tree(root: pathlib.Path) -> None:
    _helper_track(root / "a" / "b" / "x.csv", "x\n")
    _helper_track(root / "c" / "y.csv", "y\n")
    _helper_track(root / "top.csv", "top\n")
    big = root / "data" / "out"
    big.mkdir(parents=True)
    for index in range(50):
        (big / f"part-{index}.csv").write_text(f"{index}\n")
    _helper_track(root / "sub" / ".pivot" / "raw.txt", "raw\n")
    _helper_track(root / "child" / ".git" / "user.txt", "user\n")


# =============================================================================
# Reuse
# =============================================================================


def test_settled_unchanged_directories_are_stat_not_listed(
    memo_project: pathlib.Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    """After the pre-flight records the memo, a walk lists no directory and stats every one, and
    a pre-flight with nothing changed writes nothing."""
    _helper_build_tree(memo_project)
    _helper_settle(monkeypatch)
    directories = _helper_directories(memo_project)

    _helper_assert_matches_main(_helper_preflight(memo_project), memo_project)
    _helper_preflight(memo_project)
    memo = _helper_memo()
    assert memo["root"] == str(memo_project.resolve())
    assert set(memo["dirs"]) == _helper_relative_directories(memo_project), (
        "every settled directory is recorded under its project-relative path"
    )

    listed = mocker.spy(os, "scandir")
    lstat = mocker.spy(os, "lstat")
    warm = track.discover_pvt_files(memo_project)
    listings = listed.call_count
    stated = {os.fspath(call.args[0]) for call in lstat.call_args_list}
    written = mocker.spy(track, "_write_memo")
    _helper_assert_matches_main(_helper_preflight(memo_project), memo_project)

    assert listings == 0, "no settled, unchanged directory is listed again"
    assert set(directories[1:]) <= stated, "every directory below the root still gets its lstat"
    _helper_assert_matches_main(warm, memo_project)
    assert written.call_count == 0, "an unchanged memo is not written again"


def test_memo_file_is_the_accounts_own(memo_project: pathlib.Path) -> None:
    """The pre-flight creates the memo file beside the state database's files, named for the
    running account, owned by it, with mode 0600 and one link."""
    _helper_build_tree(memo_project)

    _helper_preflight(memo_project)

    path = track._memo_path()
    memo_stat = path.stat()
    assert path == config.get_state_dir() / "state.lmdb" / f"pvt-memo-{os.geteuid()}"
    assert (memo_stat.st_uid, stat.S_IMODE(memo_stat.st_mode), memo_stat.st_nlink) == (
        os.geteuid(),
        0o600,
        1,
    )


def test_read_only_walk_does_not_write_the_memo(
    memo_project: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """discover_pvt_files reads the memo but never creates or records one."""
    _helper_build_tree(memo_project)
    _helper_settle(monkeypatch)

    _helper_assert_matches_main(track.discover_pvt_files(memo_project), memo_project)

    assert not track._memo_path().exists()


def test_read_only_walk_creates_no_state_database(new_project: pathlib.Path) -> None:
    """With no state database yet, the read-only walk lists everything and creates none."""
    _helper_build_tree(new_project)

    _helper_assert_matches_main(track.discover_pvt_files(new_project), new_project)

    assert not (config.get_state_dir() / "state.lmdb").exists()


def test_read_only_walk_does_not_create_a_database_in_an_empty_state_directory(
    new_project: pathlib.Path,
) -> None:
    """An empty state.lmdb directory (no data file yet) gets no file from the read-only walk."""
    _helper_build_tree(new_project)
    lmdb_dir = config.get_state_dir() / "state.lmdb"
    lmdb_dir.mkdir(parents=True)

    _helper_assert_matches_main(track.discover_pvt_files(new_project), new_project)

    assert list(lmdb_dir.iterdir()) == [], "the read-only walk created no file"


def test_unreadable_config_means_a_full_listing_not_an_error(
    memo_project: pathlib.Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    """A read-only walk that cannot resolve the state directory lists everything, as main does."""
    _helper_build_tree(memo_project)
    _helper_settle(monkeypatch)
    _helper_record(memo_project)
    (memo_project / ".pivot" / "config.yaml").write_text("core: [unclosed\n")
    config_io.clear_config_cache()
    with pytest.raises(exceptions.ConfigError):
        config.get_state_dir()

    listed = mocker.spy(os, "scandir")
    found = track.discover_pvt_files(memo_project)
    listed_paths = {os.fspath(call.args[0]) for call in listed.call_args_list if call.args}

    assert set(_helper_directories(memo_project)) <= listed_paths, "every directory is listed"
    _helper_assert_matches_main(found, memo_project)


def test_state_directory_with_a_nul_byte_means_a_full_listing_not_an_error(
    memo_project: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``core.state_dir`` holding a NUL byte cannot be opened (``ValueError``): both walks list
    everything and return what main finds; the command fails later, where main's does."""
    _helper_build_tree(memo_project)
    _helper_settle(monkeypatch)
    (memo_project / ".pivot" / "config.yaml").write_text('core:\n  state_dir: "state\\0dir"\n')
    config_io.clear_config_cache()
    assert "\0" in os.fspath(config.get_state_dir())

    _helper_assert_matches_main(track.discover_pvt_files(memo_project), memo_project)
    _helper_assert_matches_main(_helper_preflight(memo_project), memo_project)


def _helper_run_without_waiting(
    call: Callable[[], dict[str, track.PvtData]], release: Callable[[], None]
) -> dict[str, track.PvtData]:
    """Run ``call`` in a thread and return its result. When it is still running after
    ``_NO_WAIT_S``, ``release`` frees what it waits on, and the test fails."""
    outcome = list[dict[str, track.PvtData] | BaseException]()

    def _run() -> None:
        try:
            outcome.append(call())
        except BaseException as error:  # re-raised in the test's thread below
            outcome.append(error)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(_NO_WAIT_S)
    waited = thread.is_alive()
    if waited:
        release()
        thread.join()
    assert not waited, f"the walk waited more than {_NO_WAIT_S} s"
    (result,) = outcome
    if isinstance(result, BaseException):
        raise result
    return result


def test_preflight_does_not_wait_while_another_preflight_writes_the_memo(
    memo_project: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """While another pre-flight holds the memo file's lock, the pre-flight skips its write."""
    _helper_build_tree(memo_project)
    _helper_settle(monkeypatch)
    _helper_record(memo_project)
    before = track._memo_path().read_bytes()
    time.sleep(_ONE_TICK_S)
    _helper_add_pvt(memo_project)

    with track._memo_path().open("rb") as other:
        fcntl.flock(other.fileno(), fcntl.LOCK_EX)
        found = _helper_run_without_waiting(
            lambda: _helper_preflight(memo_project),
            lambda: fcntl.flock(other.fileno(), fcntl.LOCK_UN),
        )

    _helper_assert_matches_main(found, memo_project)
    assert track._memo_path().read_bytes() == before, "nothing was recorded while it was locked"


def test_preflight_survives_a_memo_file_it_cannot_open(
    memo_project: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A memo file the account cannot open (mode 000) means a full listing and is left alone."""
    _helper_build_tree(memo_project)
    _helper_settle(monkeypatch)
    _helper_record(memo_project)
    path = track._memo_path()
    before = path.read_bytes()
    _helper_track(memo_project / "a" / "b" / "after.csv", "after\n")
    path.chmod(0)
    try:
        found = _helper_preflight(memo_project)
    finally:
        path.chmod(0o600)

    _helper_assert_matches_main(found, memo_project)
    assert path.read_bytes() == before


def test_memo_write_that_fails_is_skipped(
    memo_project: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A memo write the filesystem refuses (a full disk, say) is skipped: the pre-flight returns
    what main finds."""
    _helper_build_tree(memo_project)
    _helper_settle(monkeypatch)
    _helper_record(memo_project)
    _helper_track(memo_project / "a" / "b" / "after.csv", "after\n")

    def _disk_full(_fd: int, _length: int) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(os, "ftruncate", _disk_full)

    _helper_assert_matches_main(_helper_preflight(memo_project), memo_project)


def test_memo_write_cut_short_by_a_full_disk_leaves_an_empty_file(
    memo_project: pathlib.Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    """The disk fills partway through the memo write: the pre-flight returns what main finds and
    empties the file again, so no cut memo keeps the space, and the next walk lists every
    directory."""
    _helper_build_tree(memo_project)
    _helper_settle(monkeypatch)
    _helper_record(memo_project)
    time.sleep(_ONE_TICK_S)
    _helper_add_pvt(memo_project)
    path = os.fspath(track._memo_path())
    real_write = os.write
    memo_writes = list[int]()

    def _fills_up(fd: int, data: bytes | memoryview) -> int:
        if os.readlink(f"/proc/self/fd/{fd}") != path:
            return real_write(fd, data)
        if memo_writes:
            raise OSError(errno.ENOSPC, "No space left on device")
        memo_writes.append(real_write(fd, bytes(data)[: len(data) // 2]))
        return memo_writes[0]

    with monkeypatch.context() as patch:
        patch.setattr(os, "write", _fills_up)
        found = _helper_preflight(memo_project)

    assert memo_writes and memo_writes[0] > 0, "the write began, through os.write"
    assert pathlib.Path(path).read_bytes() == b"", "the cut memo is emptied"
    _helper_assert_matches_main(found, memo_project)
    warm, listings = _helper_listings(mocker, lambda: track.discover_pvt_files(memo_project))
    assert listings == len(_helper_directories(memo_project)), "every directory is listed"
    _helper_assert_matches_main(warm, memo_project)


def test_preflight_fstats_the_memo_once_and_writes_through_its_descriptor(
    memo_project: pathlib.Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    """A pre-flight that rewrites the memo makes one fstat of it: the write goes through the
    descriptor (``os.write``), and no file object is built on the descriptor, which would fstat
    it again."""
    _helper_build_tree(memo_project)
    _helper_settle(monkeypatch)
    _helper_record(memo_project)
    time.sleep(_ONE_TICK_S)
    _helper_add_pvt(memo_project)
    fstat = mocker.spy(os, "fstat")
    opened = mocker.spy(builtins, "open")
    written = mocker.spy(track, "_write_memo")

    _helper_assert_matches_main(_helper_preflight(memo_project), memo_project)

    assert written.call_count == 1, "the memo is rewritten"
    assert fstat.call_count == 1, "with one fstat"
    assert [call for call in opened.call_args_list if isinstance(call.args[0], int)] == []


def _helper_record_ends(content: bytes) -> list[int]:
    """Where each record of a memo file's content ends."""
    ends = list[int]()
    offset = 0
    while offset < len(content):
        key_size, value_size = track._MEMO_RECORD.unpack_from(content, offset)
        offset += track._MEMO_RECORD.size + key_size + value_size
        ends.append(offset)
    return ends


def test_memo_cut_short_is_not_used(
    memo_project: pathlib.Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    """The pre-flight empties the memo file and writes it in place, so a walk reading it during
    the write, or after the pre-flight was killed mid-write, sees a prefix of the new memo. Every
    prefix (cut at, around and between record ends, inside a record's sizes, its name and its
    listing) does not parse, so the walk lists every directory and finds what main finds; the
    next pre-flight writes the whole memo again."""
    _helper_build_tree(memo_project)
    _helper_settle(monkeypatch)
    _helper_record(memo_project)
    time.sleep(_ONE_TICK_S)
    _helper_add_pvt(memo_project)
    _helper_preflight(memo_project)
    content = track._memo_path().read_bytes()
    assert track._unpack_memo(content) == _helper_memo() != track._Memo(root="", dirs={})
    ends = _helper_record_ends(content)
    starts = [0, *ends[:-1]]
    cuts = {0, *ends, *(end - 1 for end in ends), *(end + 1 for end in ends)}
    cuts |= {start + track._MEMO_RECORD.size // 2 for start in starts}  # inside the sizes
    cuts |= {start + track._MEMO_RECORD.size + 1 for start in starts}  # inside the name
    cuts |= {(start + end) // 2 for start, end in zip(starts, ends, strict=True)}  # the listing
    directories = len(_helper_directories(memo_project))

    for cut in sorted(cut for cut in cuts if cut < len(content)):
        track._memo_path().write_bytes(content[:cut])
        listed = mocker.spy(os, "scandir")
        found = track.discover_pvt_files(memo_project)
        listings = listed.call_count
        mocker.stop(listed)

        assert track._unpack_memo(content[:cut]) is None, f"a memo cut at {cut} does not parse"
        assert listings == directories, f"cut at {cut}: every directory is listed"
        _helper_assert_matches_main(found, memo_project)

    _helper_preflight(memo_project)
    assert track._memo_path().read_bytes() == content, "the next pre-flight rewrites it whole"


def _helper_build_untracked_tree(root: pathlib.Path) -> None:
    """A project with directories and files but no .pvt anywhere."""
    for rel in ("a/b/x.csv", "c/y.csv", "data/out/part-0.csv", "data/out/part-1.csv"):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{rel}\n")


def _helper_create_state_database() -> pathlib.Path:
    """A state database some earlier pivot write made (data file and write lock present)."""
    with state.StateDB(config.get_state_dir()) as state_db:
        state_db.put_raw(b"probe", b"1")
    return config.get_state_dir() / "state.lmdb"


def test_preflight_with_no_tracked_file_creates_no_state_database(
    new_project: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """As on main, a pre-flight that finds no tracked file creates no state database, so no
    memo file either."""
    _helper_build_untracked_tree(new_project)
    _helper_settle(monkeypatch)

    executor_core.verify_tracked_files(new_project)

    assert not (config.get_state_dir() / "state.lmdb").exists()


def test_preflight_with_no_tracked_file_and_a_read_only_database_succeeds(
    memo_project: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read-only state database directory and nothing tracked: the memo file cannot be
    created, and the pre-flight returns, as main does."""
    _helper_build_untracked_tree(memo_project)
    lmdb_dir = config.get_state_dir() / "state.lmdb"
    _helper_settle(monkeypatch)
    files = list(lmdb_dir.iterdir())
    for path in files:
        path.chmod(0o444)
    lmdb_dir.chmod(0o555)
    try:
        executor_core.verify_tracked_files(memo_project)
    finally:
        lmdb_dir.chmod(0o755)
        for path in files:
            path.chmod(0o644)

    assert not track._memo_path().exists()


def test_preflight_with_no_tracked_file_records_into_an_existing_database(
    memo_project: pathlib.Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    """A project with nothing tracked still gets the memo when its state database exists: the
    second pre-flight lists no directory."""
    _helper_build_untracked_tree(memo_project)
    _helper_settle(monkeypatch)

    executor_core.verify_tracked_files(memo_project)
    listed = mocker.spy(os, "scandir")
    executor_core.verify_tracked_files(memo_project)
    listings = listed.call_count

    assert listings == 0, "the second pre-flight lists no directory"
    assert set(_helper_memo()["dirs"]) == _helper_relative_directories(memo_project)


# =============================================================================
# Validity
# =============================================================================


def _helper_add_pvt(root: pathlib.Path) -> None:
    _helper_track(root / "a" / "b" / "new.csv", "new\n")


def _helper_remove_pvt(root: pathlib.Path) -> None:
    (root / "a" / "b" / "x.csv.pvt").unlink()


def _helper_rename_pvt_within(root: pathlib.Path) -> None:
    (root / "a" / "b" / "x.csv.pvt").rename(root / "a" / "b" / "renamed.csv.pvt")


def _helper_rename_pvt_in(root: pathlib.Path) -> None:
    (root / "c" / "y.csv.pvt").rename(root / "a" / "b" / "y.csv.pvt")


def _helper_rename_dir(root: pathlib.Path) -> None:
    (root / "c").rename(root / "a" / "b" / "c2")


def _helper_add_dir(root: pathlib.Path) -> None:
    _helper_track(root / "a" / "new" / "deep" / "n.csv", "n\n")


def _helper_remove_dir(root: pathlib.Path) -> None:
    shutil.rmtree(root / "c")


def _helper_hardlink_pvt(root: pathlib.Path) -> None:
    os.link(root / "c" / "y.csv.pvt", root / "a" / "b" / "y.csv.pvt")


def _helper_add_pvt_in_hidden_dirs(root: pathlib.Path) -> None:
    _helper_track(root / "sub" / ".pivot" / "more.txt", "more\n")
    _helper_track(root / "child" / ".git" / "more.txt", "more\n")


def _helper_add_pvt_in_output(root: pathlib.Path) -> None:
    _helper_track(root / "data" / "out" / "tracked.csv", "tracked\n")


def _helper_add_pvt_keep_dir_mtime(root: pathlib.Path) -> None:
    """Add a .pvt, then set its directory's mtime back, as ``tar`` or ``rsync -t`` do."""
    directory = root / "a" / "b"
    before = directory.stat()
    _helper_add_pvt(root)
    os.utime(directory, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert directory.stat().st_mtime_ns == before.st_mtime_ns, "only the ctime moved"


def _helper_dir_to_symlink(root: pathlib.Path) -> None:
    shutil.rmtree(root / "a" / "b")
    (root / "a" / "b").symlink_to(root / "c", target_is_directory=True)


def _helper_dir_to_file(root: pathlib.Path) -> None:
    shutil.rmtree(root / "a" / "b")
    (root / "a" / "b").write_text("now a file\n")


def _helper_pvt_symlink_to_file(root: pathlib.Path) -> None:
    _helper_track(root / "elsewhere" / "z.csv", "z\n")
    (root / "a" / "b.csv.pvt").symlink_to(root / "elsewhere" / "z.csv.pvt")


def _helper_pvt_named_dir(root: pathlib.Path) -> None:
    _helper_track(root / "a" / "dir.pvt" / "inner.csv", "inner\n")


def _helper_non_utf8_names(root: pathlib.Path) -> None:
    _helper_track(root / os.fsdecode(b"caf\xe9") / "menu.csv", "menu\n")
    (root / "a" / "b" / os.fsdecode(b"\xff\xfe.bin")).write_bytes(b"x")


_CHANGES = [
    pytest.param(_helper_add_pvt, id="add-pvt"),
    pytest.param(_helper_add_pvt_keep_dir_mtime, id="add-pvt-then-restore-dir-mtime"),
    pytest.param(_helper_remove_pvt, id="remove-pvt"),
    pytest.param(_helper_rename_pvt_within, id="rename-pvt-within-dir"),
    pytest.param(_helper_rename_pvt_in, id="rename-pvt-in-from-another-dir"),
    pytest.param(_helper_rename_dir, id="rename-dir"),
    pytest.param(_helper_add_dir, id="add-dir"),
    pytest.param(_helper_remove_dir, id="remove-dir"),
    pytest.param(_helper_hardlink_pvt, id="hardlink-pvt-in"),
    pytest.param(_helper_add_pvt_in_hidden_dirs, id="add-pvt-under-.pivot-and-.git"),
    pytest.param(_helper_add_pvt_in_output, id="add-pvt-inside-a-big-directory"),
    pytest.param(_helper_dir_to_symlink, id="dir-replaced-by-symlink"),
    pytest.param(_helper_dir_to_file, id="dir-replaced-by-file"),
    pytest.param(_helper_pvt_symlink_to_file, id="pvt-symlink-to-a-file"),
    pytest.param(_helper_pvt_named_dir, id="directory-named-pvt"),
    pytest.param(_helper_non_utf8_names, id="non-utf-8-names"),
]


@pytest.mark.parametrize("change", _CHANGES)
def test_memo_walk_matches_main_after_a_change(
    memo_project: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    change: Callable[[pathlib.Path], None],
) -> None:
    """Added, removed, renamed and replaced entries are found, read-only and in the pre-flight."""
    _helper_build_tree(memo_project)
    _helper_settle(monkeypatch)
    _helper_record(memo_project)
    time.sleep(_ONE_TICK_S)

    change(memo_project)

    _helper_assert_matches_main(track.discover_pvt_files(memo_project), memo_project)
    _helper_assert_matches_main(_helper_preflight(memo_project), memo_project)
    _helper_assert_matches_main(track.discover_pvt_files(memo_project), memo_project)


def test_preflight_drops_the_listing_of_a_directory_it_no_longer_visits(
    memo_project: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The memo holds only what the last pre-flight recorded: a removed directory's listing is
    dropped, not kept beside the new ones."""
    _helper_build_tree(memo_project)
    _helper_settle(monkeypatch)
    _helper_record(memo_project)
    assert "c" in _helper_memo()["dirs"]
    time.sleep(_ONE_TICK_S)

    _helper_remove_dir(memo_project)
    _helper_preflight(memo_project)

    assert set(_helper_memo()["dirs"]) == _helper_relative_directories(memo_project)


def test_memo_walk_matches_main_across_random_changes(
    memo_project: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A seeded run of random changes: after each, the walk finds what main finds."""
    rng = random.Random(513)
    _helper_settle(monkeypatch)
    dirs = [memo_project]
    for step in range(60):
        time.sleep(_ONE_TICK_S)
        parent = rng.choice(dirs)
        pvts = sorted(path for path in memo_project.rglob("*.pvt") if path.is_file())
        action = rng.choice(["track", "track", "mkdir", "unlink", "rename", "rmdir"])
        if action == "track":
            _helper_track(parent / f"f{step}.csv", f"{step}\n")
        elif action == "mkdir":
            child = parent / f"d{step}"
            child.mkdir()
            dirs.append(child)
        elif action == "unlink" and pvts:
            rng.choice(pvts).unlink()
        elif action == "rename" and pvts:
            rng.choice(pvts).rename(parent / f"moved{step}.csv.pvt")
        elif action == "rmdir" and len(dirs) > 1:
            shutil.rmtree(rng.choice(dirs[1:]))
            dirs = [path for path in dirs if path.exists()]
        found = (
            track.discover_pvt_files(memo_project)
            if rng.random() < 0.5
            else _helper_preflight(memo_project)
        )
        _helper_assert_matches_main(found, memo_project)


# =============================================================================
# Settle window
# =============================================================================


def _helper_report_directory_times(
    monkeypatch: pytest.MonkeyPatch,
    root: pathlib.Path,
    times: Callable[[os.stat_result], tuple[int, int]],
) -> None:
    """Report every directory under ``root`` with the (mtime_ns, ctime_ns) that ``times`` gives.

    Simulates a filesystem's timestamp tick or a stepped clock without root.
    """
    prefix = os.fspath(root)

    def _with(result: os.stat_result) -> os.stat_result:
        mtime_ns, ctime_ns = times(result)
        fields = list(result)
        fields[8], fields[9] = mtime_ns // _NS_PER_S, ctime_ns // _NS_PER_S
        return os.stat_result(
            fields,
            {
                "st_atime": result.st_atime,
                "st_mtime": mtime_ns / 1e9,
                "st_ctime": ctime_ns / 1e9,
                "st_atime_ns": result.st_atime_ns,
                "st_mtime_ns": mtime_ns,
                "st_ctime_ns": ctime_ns,
                "st_blksize": result.st_blksize,
                "st_blocks": result.st_blocks,
                "st_rdev": result.st_rdev,
            },
        )

    def _wrap(real: Callable[..., os.stat_result]) -> Callable[..., os.stat_result]:
        def _stat(path: object, *args: object, **kwargs: object) -> os.stat_result:
            result = real(path, *args, **kwargs)
            inside = isinstance(path, str | os.PathLike) and os.fsdecode(path).startswith(prefix)
            return _with(result) if inside and stat.S_ISDIR(result.st_mode) else result

        return _stat

    monkeypatch.setattr(os, "stat", _wrap(os.stat))
    monkeypatch.setattr(os, "lstat", _wrap(os.lstat))


def _helper_freeze_directory_times(
    monkeypatch: pytest.MonkeyPatch, root: pathlib.Path, frozen_ns: int
) -> None:
    """Simulate a filesystem whose timestamp tick has not advanced since ``frozen_ns``.

    Every directory under ``root`` reports that mtime and ctime however it changes, so a change
    in the same tick as a recorded listing leaves the directory's stamp as it was.
    """
    _helper_report_directory_times(monkeypatch, root, lambda _result: (frozen_ns, frozen_ns))


def _helper_tick_now() -> int:
    """The current time as a nanosecond tick that is never a whole second (whole seconds are
    never trusted)."""
    return _REAL_TIME_NS() | 1


def test_directory_changed_within_the_settle_window_is_not_memoized(
    memo_project: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A listing taken less than 2 s after its directory changed is never recorded.

    With the tick frozen, a later change in that tick leaves the stamp unchanged; only the settle
    window keeps a recorded listing from hiding the new .pvt.
    """
    _helper_build_tree(memo_project)
    frozen_ns = _helper_tick_now()
    _helper_freeze_directory_times(monkeypatch, memo_project, frozen_ns)
    monkeypatch.setattr(time, "time_ns", lambda: frozen_ns + _INSIDE_SETTLE_WINDOW_NS)

    _helper_assert_matches_main(_helper_preflight(memo_project), memo_project)
    assert _helper_memo()["dirs"] == {}, "nothing changed in the last 2 s is recorded"

    _helper_add_pvt(memo_project)

    _helper_assert_matches_main(track.discover_pvt_files(memo_project), memo_project)
    _helper_assert_matches_main(_helper_preflight(memo_project), memo_project)


def test_recorded_directory_inside_the_settle_window_is_listed_again(
    memo_project: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recorded stamp less than 2 s old at the walk (the clock went back) is not trusted."""
    _helper_build_tree(memo_project)
    frozen_ns = _helper_tick_now()
    _helper_freeze_directory_times(monkeypatch, memo_project, frozen_ns)
    monkeypatch.setattr(time, "time_ns", lambda: frozen_ns + _PAST_SETTLE_WINDOW_NS)
    _helper_preflight(memo_project)
    assert _helper_memo()["dirs"], "settled directories are recorded"

    monkeypatch.setattr(time, "time_ns", lambda: frozen_ns + _INSIDE_SETTLE_WINDOW_NS)
    _helper_add_pvt(memo_project)

    _helper_assert_matches_main(track.discover_pvt_files(memo_project), memo_project)
    _helper_assert_matches_main(_helper_preflight(memo_project), memo_project)


def test_settle_needs_the_ctime_when_a_restore_sets_the_mtime_back(
    memo_project: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Within one frozen tick, a restore adds a .pvt and sets its directory's mtime back (as tar
    and rsync -t do), the pre-flight runs, and a second restore in that tick does the same. The
    mtime looks settled; only the ctime half of the settle check keeps the listing unrecorded."""
    _helper_build_tree(memo_project)
    frozen_ns = _helper_tick_now()
    restored_mtime_ns = frozen_ns - 100 * _NS_PER_S
    _helper_report_directory_times(
        monkeypatch,
        memo_project,
        lambda result: (min(result.st_mtime_ns, frozen_ns), frozen_ns)
        if result.st_mtime_ns < frozen_ns - 50 * _NS_PER_S
        else (frozen_ns, frozen_ns),
    )
    monkeypatch.setattr(time, "time_ns", lambda: frozen_ns + _INSIDE_SETTLE_WINDOW_NS)
    directory = memo_project / "a" / "b"

    _helper_track(directory / "one.csv", "one\n")
    os.utime(directory, ns=(restored_mtime_ns, restored_mtime_ns))
    _helper_assert_matches_main(_helper_preflight(memo_project), memo_project)
    _helper_track(directory / "two.csv", "two\n")
    os.utime(directory, ns=(restored_mtime_ns, restored_mtime_ns))

    _helper_assert_matches_main(track.discover_pvt_files(memo_project), memo_project)


def test_settle_window_counts_from_the_walk_start_on_a_long_walk(
    memo_project: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A walk that takes over 3 s, within one frozen tick: the root is listed, then changes in
    the same tick. Settled measured from the walk's end would record the root's first listing
    under the unchanged stamp; measured from its start, the root is never recorded."""
    _helper_build_tree(memo_project)
    _helper_freeze_directory_times(monkeypatch, memo_project, _helper_tick_now())
    real_list = track._list_directory
    injected = list[str]()

    def _slow_list(path: str) -> tuple[list[str], list[str], bool] | None:
        names = real_list(path)
        if path == os.fspath(memo_project) and not injected:
            injected.append(path)
            _helper_track(memo_project / "late.csv", "late\n")
        time.sleep(0.35)
        return names

    monkeypatch.setattr(track, "_list_directory", _slow_list)
    start = time.monotonic()
    _helper_preflight(memo_project)
    took = time.monotonic() - start
    monkeypatch.setattr(track, "_list_directory", real_list)

    assert took > 3.0, f"the walk took longer than the settle window ({took:.1f} s)"
    _helper_assert_matches_main(track.discover_pvt_files(memo_project), memo_project)


def test_clock_stepped_back_into_a_recorded_second_hides_nothing(
    memo_project: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One-second timestamps (ext2, ext3, ext4 with 128-byte inodes) and a wall clock stepped back.

    The pre-flight sees a/b under a whole-second stamp. The clock steps back into that second
    and a .pvt is added to a/b; on such a filesystem its mtime and ctime land in the same second,
    so a/b's stamp is unchanged. A whole-second stamp is never trusted, so a/b is listed and the
    new .pvt is found.
    """
    _helper_build_tree(memo_project)
    clock = {"back_ns": 0}

    def _seconds(ns: int) -> int:
        return (ns - clock["back_ns"]) // _NS_PER_S * _NS_PER_S

    _helper_report_directory_times(
        monkeypatch,
        memo_project,
        lambda result: (_seconds(result.st_mtime_ns), _seconds(result.st_ctime_ns)),
    )
    monkeypatch.setattr(
        time, "time_ns", lambda: _REAL_TIME_NS() - clock["back_ns"] + _PAST_SETTLE_WINDOW_NS
    )
    _helper_record(memo_project)
    directory = memo_project / "a" / "b"
    seen_second = os.stat(directory).st_ctime_ns

    clock["back_ns"] = _REAL_TIME_NS() - (seen_second + 300_000_000)
    _helper_add_pvt(memo_project)

    assert (os.stat(directory).st_mtime_ns, os.stat(directory).st_ctime_ns) == (
        seen_second,
        seen_second,
    ), "the add landed in the second the pre-flight saw"
    _helper_assert_matches_main(track.discover_pvt_files(memo_project), memo_project)
    _helper_assert_matches_main(_helper_preflight(memo_project), memo_project)
    _helper_assert_matches_main(track.discover_pvt_files(memo_project), memo_project)


# =============================================================================
# Filesystems
# =============================================================================


@pytest.mark.parametrize(
    "fstype",
    [
        pytest.param("nfs4", id="nfs"),
        pytest.param("nfs", id="nfs-version-3"),
        pytest.param("overlay", id="overlayfs"),
        pytest.param("fuse.sshfs", id="fuse"),
        pytest.param("9p", id="unknown-type"),
        pytest.param(None, id="not-in-mountinfo"),
    ],
)
def test_filesystem_off_the_allowlist_is_always_listed(
    memo_project: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    fstype: str | None,
) -> None:
    """NFS, overlayfs, FUSE and unknown filesystems get a full listing, and no memo file is
    created on them."""
    _helper_mountinfo(monkeypatch, tmp_path / "mountinfo", memo_project, fstype)
    _helper_build_tree(memo_project)
    _helper_settle(monkeypatch)

    _helper_assert_matches_main(_helper_preflight(memo_project), memo_project)
    assert not track._memo_path().exists(), "no memo file is created off the allowlist"

    listed = mocker.spy(os, "scandir")
    found = track.discover_pvt_files(memo_project)
    listings = listed.call_count

    assert listings == len(_helper_directories(memo_project)), "every directory is listed"
    _helper_assert_matches_main(found, memo_project)


def test_without_mountinfo_no_directory_is_trusted(
    memo_project: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No /proc/self/mountinfo (not Linux, or /proc not mounted): every directory is listed."""
    monkeypatch.setattr(track, "_MOUNTINFO", tmp_path / "no-such-mountinfo")
    _helper_build_tree(memo_project)
    _helper_settle(monkeypatch)
    _helper_record(memo_project)

    assert not track._memo_path().exists(), "no memo file is created without mountinfo"
    _helper_assert_matches_main(track.discover_pvt_files(memo_project), memo_project)


def test_whole_second_mtime_alone_still_uses_the_memo(
    memo_project: pathlib.Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    """A directory whose mtime alone is a whole second (an archive with one-second times
    restored it) keeps a nanosecond ctime, so its listing is still recorded and reused."""
    _helper_build_tree(memo_project)
    whole_second_ns = (_REAL_TIME_NS() - 100 * _NS_PER_S) // _NS_PER_S * _NS_PER_S
    os.utime(memo_project / "a" / "b", ns=(whole_second_ns, whole_second_ns))
    assert (memo_project / "a" / "b").stat().st_ctime_ns % _NS_PER_S != 0
    _helper_settle(monkeypatch)
    _helper_record(memo_project)

    listed = mocker.spy(os, "scandir")
    found = track.discover_pvt_files(memo_project)
    listings = listed.call_count

    assert "a/b" in _helper_memo()["dirs"], "a/b is recorded"
    assert listings == 0, "and reused"
    _helper_assert_matches_main(found, memo_project)


@pytest.mark.parametrize("fstype", ["ext2", "ext3", "ext4", "xfs", "btrfs", "tmpfs", "f2fs", "zfs"])
def test_allowlisted_filesystem_uses_the_memo(
    memo_project: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    fstype: str,
) -> None:
    _helper_mountinfo(monkeypatch, tmp_path / "mountinfo", memo_project, fstype)
    _helper_build_tree(memo_project)
    _helper_settle(monkeypatch)
    _helper_record(memo_project)

    listed = mocker.spy(os, "scandir")
    found = track.discover_pvt_files(memo_project)
    listings = listed.call_count

    assert listings == 0, f"a settled directory on {fstype} is not listed again"
    _helper_assert_matches_main(found, memo_project)


def _helper_unlisted_under_btrfs(root: pathlib.Path) -> list[str]:
    """root's device is not listed, as a btrfs subvolume's is not; its parent is a btrfs mount."""
    return [_helper_mount_line(92, os.makedev(0, 4243), os.path.realpath(root.parent), "btrfs")]


def _helper_unlisted_under_ext4(root: pathlib.Path) -> list[str]:
    return [_helper_mount_line(92, os.makedev(0, 4243), os.path.realpath(root.parent), "ext4")]


def _helper_unlisted_under_tmpfs_inside_btrfs(root: pathlib.Path) -> list[str]:
    parent = os.path.realpath(root.parent)
    return [
        _helper_mount_line(92, os.makedev(0, 4243), os.path.dirname(parent), "btrfs"),
        _helper_mount_line(93, os.makedev(0, 4244), parent, "tmpfs"),
    ]


def _helper_listed_as_ext4_and_nfs(root: pathlib.Path) -> list[str]:
    device = os.stat(root).st_dev
    return [
        _helper_mount_line(92, device, "/srv/a", "ext4"),
        _helper_mount_line(93, device, "/srv/b", "nfs4"),
    ]


@pytest.mark.parametrize(
    ("mounts", "used"),
    [
        pytest.param(_helper_unlisted_under_btrfs, True, id="unlisted-device-under-btrfs"),
        pytest.param(_helper_unlisted_under_ext4, False, id="unlisted-device-under-ext4"),
        pytest.param(
            _helper_unlisted_under_tmpfs_inside_btrfs, False, id="the-deepest-mount-decides"
        ),
        pytest.param(_helper_listed_as_ext4_and_nfs, False, id="device-listed-as-ext4-and-nfs"),
    ],
)
def test_device_type_comes_from_mountinfo(
    memo_project: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    mounts: Callable[[pathlib.Path], list[str]],
    used: bool,
) -> None:
    """A device mountinfo does not list is trusted only under a btrfs mount (a subvolume)."""
    _helper_write_mountinfo(monkeypatch, tmp_path / "mountinfo", mounts(memo_project))
    _helper_build_tree(memo_project)
    _helper_settle(monkeypatch)
    _helper_record(memo_project)

    listed = mocker.spy(os, "scandir")
    found = track.discover_pvt_files(memo_project)
    listings = listed.call_count

    assert (listings == 0) is used, "the memo is used exactly when the device is trusted"
    assert bool(_helper_memo()["dirs"]) is used
    _helper_assert_matches_main(found, memo_project)


def test_mount_point_with_a_space_is_unescaped(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    """mountinfo writes a space as \\040; a btrfs mount at such a path still covers the project."""
    mount = tmp_path / "with space"
    root = mount / "project"
    (root / ".pivot").mkdir(parents=True)
    monkeypatch.setattr(project, "_project_root_cache", root)
    _helper_create_state_database()
    btrfs = _helper_mount_line(92, os.makedev(0, 4243), os.path.realpath(mount), "btrfs")
    assert "\\040" in btrfs
    _helper_write_mountinfo(monkeypatch, tmp_path / "mountinfo", [btrfs])
    _helper_build_tree(root)
    _helper_settle(monkeypatch)
    _helper_record(root)

    listed = mocker.spy(os, "scandir")
    found = track.discover_pvt_files(root)
    listings = listed.call_count

    assert listings == 0, "the escaped btrfs mount point matched the project's path"
    _helper_assert_matches_main(found, root)


# =============================================================================
# Moved project, rewritten .pvt
# =============================================================================


def test_moved_project_does_not_reuse_the_memo(
    memo_project: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """A memo recorded under another project root is ignored, then replaced by the pre-flight."""
    _helper_build_tree(memo_project)
    _helper_settle(monkeypatch)
    _helper_record(memo_project)
    moved = tmp_path / "moved"
    memo_project.rename(moved)
    monkeypatch.setattr(project, "_project_root_cache", moved)
    directories = len(_helper_directories(moved))

    listed = mocker.spy(os, "scandir")
    found = track.discover_pvt_files(moved)
    read_only_listings = listed.call_count
    listed.reset_mock()
    recorded = _helper_preflight(moved)
    preflight_listings = listed.call_count

    assert read_only_listings == directories, "a read-only walk of a moved project lists all"
    assert preflight_listings == directories, "so does the pre-flight, which then re-records"
    _helper_assert_matches_main(found, moved)
    _helper_assert_matches_main(recorded, moved)
    memo = _helper_memo()
    assert memo["root"] == str(moved.resolve()), "the pre-flight records the new root"
    assert set(memo["dirs"]) == _helper_relative_directories(moved)


def _helper_rewrite_in_place_and_read(root: pathlib.Path) -> None:
    """Rewrite a/b/x.csv.pvt in place with a new hash; both walks must read the new hash."""
    pvt_path = root / "a" / "b" / "x.csv.pvt"
    old = track.read_pvt_file(pvt_path)
    assert old is not None
    inode = pvt_path.stat().st_ino
    directory_stamp = (pvt_path.parent.stat().st_mtime_ns, pvt_path.parent.stat().st_ctime_ns)

    with pvt_path.open("r+") as pvt_file:
        text = pvt_file.read().replace(old["hash"], "0123456789abcdef")
        pvt_file.seek(0)
        pvt_file.write(text)
        pvt_file.truncate()

    assert pvt_path.stat().st_ino == inode, "rewritten in place"
    assert (pvt_path.parent.stat().st_mtime_ns, pvt_path.parent.stat().st_ctime_ns) == (
        directory_stamp
    ), "its directory did not change"
    data_path = str(root / "a" / "b" / "x.csv")
    assert track.discover_pvt_files(root)[data_path]["hash"] == "0123456789abcdef"
    assert _helper_preflight(root)[data_path]["hash"] == "0123456789abcdef"


def test_pvt_rewritten_in_place_within_the_settle_window_shows_its_new_hash(
    memo_project: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Within 2 s of its directory's last change, nothing is recorded and the .pvt is read anew."""
    before = _REAL_TIME_NS()
    monkeypatch.setattr(time, "time_ns", lambda: before)
    _helper_build_tree(memo_project)
    _helper_record(memo_project)
    assert _helper_memo()["dirs"] == {}, "nothing is recorded inside the settle window"

    _helper_rewrite_in_place_and_read(memo_project)


def test_pvt_rewritten_in_place_after_the_settle_window_shows_its_new_hash(
    memo_project: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The memo holds names only: a recorded directory's .pvt rewritten in place is read anew."""
    _helper_build_tree(memo_project)
    _helper_settle(monkeypatch)
    _helper_record(memo_project)
    assert "a/b" in _helper_memo()["dirs"], "a/b's listing is recorded"

    _helper_rewrite_in_place_and_read(memo_project)


def test_symlinked_project_root_shares_one_memo(
    memo_project: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """A root reached through a symlink is walked as os.walk walks it, and uses the memo."""
    _helper_build_tree(memo_project)
    link = tmp_path / "link"
    link.symlink_to(memo_project, target_is_directory=True)
    _helper_settle(monkeypatch)
    _helper_record(link)

    listed = mocker.spy(os, "scandir")
    through_link = track.discover_pvt_files(link)
    direct = track.discover_pvt_files(memo_project)
    listings = listed.call_count

    assert listings == 0, "the link and the real path record one memo"
    _helper_assert_matches_main(through_link, link)
    _helper_assert_matches_main(direct, memo_project)


def test_unreadable_directory_is_skipped_as_os_walk_skips_it(
    memo_project: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _helper_build_tree(memo_project)
    _helper_settle(monkeypatch)
    _helper_record(memo_project)
    time.sleep(_ONE_TICK_S)
    unreadable = memo_project / "c"
    unreadable.chmod(0)
    try:
        _helper_assert_matches_main(track.discover_pvt_files(memo_project), memo_project)
        _helper_assert_matches_main(_helper_preflight(memo_project), memo_project)
    finally:
        unreadable.chmod(0o755)
    _helper_assert_matches_main(track.discover_pvt_files(memo_project), memo_project)


# Where the listing format puts the subdirectory count: after the version byte and the stamp.
_COUNT_AT = track._LISTING_HEADER - track._SUBDIR_COUNT.size


def _helper_truncated(listing: bytes) -> bytes:
    return listing[: track._LISTING_HEADER - 2]


def _helper_count_too_large(listing: bytes) -> bytes:
    return listing[:_COUNT_AT] + track._SUBDIR_COUNT.pack(999) + listing[track._LISTING_HEADER :]


def _helper_name_with_slash(listing: bytes) -> bytes:
    return listing[:_COUNT_AT] + track._SUBDIR_COUNT.pack(1) + b"../../elsewhere\0x.csv.pvt"


def _helper_empty_name(listing: bytes) -> bytes:
    return listing[:_COUNT_AT] + track._SUBDIR_COUNT.pack(0) + b"x.csv.pvt\0\0new.csv.pvt"


def _helper_version_1(listing: bytes) -> bytes:
    """The listing as version 1 wrote it, which could omit a name: here, every name."""
    return b"\x01" + listing[len(track._LISTING_VERSION) : track._LISTING_HEADER]


def _helper_version_2(listing: bytes) -> bytes:
    """The listing as version 2 (in the state database) wrote it, every name omitted."""
    return b"\x02" + listing[len(track._LISTING_VERSION) : track._LISTING_HEADER]


@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param(_helper_truncated, id="truncated"),
        pytest.param(_helper_count_too_large, id="subdirectory-count-too-large"),
        pytest.param(_helper_name_with_slash, id="name-with-a-slash"),
        pytest.param(_helper_empty_name, id="empty-name"),
        pytest.param(_helper_version_1, id="earlier-listing-version-1"),
        pytest.param(_helper_version_2, id="earlier-listing-version-2"),
    ],
)
def test_malformed_recorded_listing_is_listed_again(
    memo_project: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    corrupt: Callable[[bytes], bytes],
) -> None:
    """A recorded listing that does not parse counts as a miss: the directory is listed."""
    _helper_build_tree(memo_project)
    _helper_settle(monkeypatch)
    _helper_record(memo_project)
    memo = _helper_memo()
    broken = {**memo["dirs"], "a/b": corrupt(memo["dirs"]["a/b"])}
    _helper_write_memo(track._Memo(root=memo["root"], dirs=broken))

    _helper_assert_matches_main(track.discover_pvt_files(memo_project), memo_project)
    _helper_assert_matches_main(_helper_preflight(memo_project), memo_project)
    assert _helper_memo()["dirs"]["a/b"] != broken["a/b"], "the pre-flight records it again"


def test_dotdot_in_a_recorded_listing_never_leaves_the_project(
    memo_project: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recorded root listing naming '..' as a subdirectory is malformed: the walk must not
    climb out of the project and read the sidecars beside it."""
    _helper_build_tree(memo_project)
    _helper_track(memo_project.parent / "outside" / "o.csv", "outside\n")
    _helper_settle(monkeypatch)
    _helper_record(memo_project)
    memo = _helper_memo()
    listing = memo["dirs"][""]
    (count,) = track._SUBDIR_COUNT.unpack_from(listing, _COUNT_AT)
    poisoned = (
        listing[:_COUNT_AT]
        + track._SUBDIR_COUNT.pack(count + 1)
        + b"..\0"
        + listing[track._LISTING_HEADER :]
    )
    _helper_write_memo(track._Memo(root=memo["root"], dirs={**memo["dirs"], "": poisoned}))

    _helper_assert_matches_main(track.discover_pvt_files(memo_project), memo_project)


# =============================================================================
# Listings the memo never records
# =============================================================================


class _UnclassifiableEntry:
    """A DirEntry whose type cannot be read, as is_dir behaves without d_type and without x."""

    name: str

    def __init__(self, entry: os.DirEntry[str]) -> None:
        self.name = entry.name

    def is_dir(self, *, follow_symlinks: bool = True) -> bool:
        raise PermissionError(errno.EACCES, "Permission denied", self.name)


def _helper_scandir_without_types(
    monkeypatch: pytest.MonkeyPatch, directory: pathlib.Path, name: str
) -> None:
    """os.scandir, except that the type of ``name`` in ``directory`` cannot be read."""
    real_scandir = os.scandir

    def _scandir(path: str) -> object:
        listing = real_scandir(path)
        if os.path.normpath(path) != os.path.normpath(directory):
            return listing

        @contextlib.contextmanager
        def _wrapped() -> Iterator[list[object]]:
            with listing as entries:
                yield [_UnclassifiableEntry(e) if e.name == name else e for e in entries]

        return _wrapped()

    monkeypatch.setattr(os, "scandir", _scandir)


def test_listing_with_an_unclassifiable_entry_is_not_reused(
    memo_project: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a filesystem whose listings carry no entry type (XFS with ftype=0, ext4 without the
    filetype feature), is_dir needs an lstat, which an account that may read P but not search it
    cannot make: P/sub counts as no subdirectory, as in os.walk. Such a listing is never
    recorded, so a later walk with full rights still finds P/sub."""
    _helper_track(memo_project / "top.csv", "top\n")
    _helper_track(memo_project / "P" / "sub" / "x.csv", "x\n")
    _helper_settle(monkeypatch)

    with monkeypatch.context() as patch:
        _helper_scandir_without_types(patch, memo_project / "P", "sub")
        _helper_preflight(memo_project)

    assert "P" not in _helper_memo()["dirs"], "the listing with an untyped entry is not recorded"
    _helper_assert_matches_main(track.discover_pvt_files(memo_project), memo_project)


@pytest.mark.parametrize("seconds", [10_413_792_000], ids=["year-2300"])
def test_directory_mtime_past_2262_is_walked(memo_project: pathlib.Path, seconds: int) -> None:
    """ext4 and XFS keep mtimes past 2262, which a signed 64-bit nanosecond count cannot hold:
    both walks list such a directory instead of failing."""
    _helper_track(memo_project / "top.csv", "top\n")
    odd = memo_project / "odd"
    odd.mkdir()
    os.utime(odd, ns=(0, seconds * _NS_PER_S))
    assert odd.stat().st_mtime_ns > 2**63 - 1

    _helper_assert_matches_main(track.discover_pvt_files(memo_project), memo_project)
    _helper_assert_matches_main(_helper_preflight(memo_project), memo_project)


_BEFORE_1677_NS = -(2**63) - 1
_AFTER_2262_NS = 2**63 + 1
_CLOCK_IN_2400_NS = 13_569_465_600 * _NS_PER_S


def _helper_mtime_before_1677(result: os.stat_result) -> tuple[int, int]:
    return _BEFORE_1677_NS, result.st_ctime_ns


def _helper_ctime_before_1677(result: os.stat_result) -> tuple[int, int]:
    return result.st_mtime_ns, _BEFORE_1677_NS


def _helper_mtime_after_2262(result: os.stat_result) -> tuple[int, int]:
    return _AFTER_2262_NS, result.st_ctime_ns


def _helper_ctime_after_2262(result: os.stat_result) -> tuple[int, int]:
    return result.st_mtime_ns, _AFTER_2262_NS


@pytest.mark.parametrize(
    ("times", "clock_ns"),
    [
        pytest.param(_helper_mtime_before_1677, None, id="mtime-before-1677"),
        pytest.param(_helper_ctime_before_1677, None, id="ctime-before-1677"),
        pytest.param(
            _helper_mtime_after_2262, _CLOCK_IN_2400_NS, id="mtime-after-2262-and-the-clock-later"
        ),
        pytest.param(
            _helper_ctime_after_2262, _CLOCK_IN_2400_NS, id="ctime-after-2262-and-the-clock-later"
        ),
    ],
)
def test_directory_time_outside_the_stamp_is_listed_every_walk(
    memo_project: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    times: Callable[[os.stat_result], tuple[int, int]],
    clock_ns: int | None,
) -> None:
    """tmpfs and btrfs keep times before 1677 too. A time the stamp cannot hold is never trusted,
    even when it is settled, so its directory is listed every walk and never recorded."""
    _helper_track(memo_project / "top.csv", "top\n")
    _helper_track(memo_project / "odd" / "inside.csv", "inside\n")
    _helper_report_directory_times(monkeypatch, memo_project / "odd", times)
    if clock_ns is None:
        _helper_settle(monkeypatch)
    else:
        monkeypatch.setattr(time, "time_ns", lambda: clock_ns)

    _helper_assert_matches_main(track.discover_pvt_files(memo_project), memo_project)
    _helper_record(memo_project)
    _helper_assert_matches_main(track.discover_pvt_files(memo_project), memo_project)

    memo = _helper_memo()
    assert "odd" not in memo["dirs"], "a time outside the stamp is never recorded"
    assert "" in memo["dirs"], "the other directories still are"


# =============================================================================
# The memo file: only the account that listed a directory reads its names
# =============================================================================


def _helper_inside(root: pathlib.Path, tmp_path: pathlib.Path) -> pathlib.Path:
    """The default layout: the state directory is the project's .pivot."""
    return root / ".pivot"


def _helper_private_directory(root: pathlib.Path, tmp_path: pathlib.Path) -> pathlib.Path:
    """The default layout, with the untracked names under a mode-700 directory."""
    (root / "notes").chmod(0o700)
    return root / ".pivot"


def _helper_outside_under_a_private_home(
    root: pathlib.Path, tmp_path: pathlib.Path
) -> pathlib.Path:
    """``core.state_dir`` outside the project, and the project under a mode-700 directory."""
    state_dir = tmp_path / "state"
    (root / ".pivot" / "config.yaml").write_text(f"core:\n  state_dir: {state_dir}\n")
    root.parent.chmod(0o700)
    return state_dir


def _helper_symlinked_state_database(root: pathlib.Path, tmp_path: pathlib.Path) -> pathlib.Path:
    """``.pivot/state.lmdb`` is a symlink to a directory outside the project."""
    elsewhere = tmp_path / "elsewhere" / "state.lmdb"
    elsewhere.mkdir(parents=True)
    (root / ".pivot" / "state.lmdb").symlink_to(elsewhere, target_is_directory=True)
    root.parent.chmod(0o700)
    return root / ".pivot"


def _helper_symlinked_data_file(root: pathlib.Path, tmp_path: pathlib.Path) -> pathlib.Path:
    """``.pivot/state.lmdb/data.mdb`` alone is a symlink to a file outside the project."""
    data_file = tmp_path / "elsewhere" / "data.mdb"
    data_file.parent.mkdir(parents=True)
    data_file.touch()
    (root / ".pivot" / "state.lmdb").mkdir()
    (root / ".pivot" / "state.lmdb" / "data.mdb").symlink_to(data_file)
    root.parent.chmod(0o700)
    return root / ".pivot"


@pytest.mark.parametrize(
    "layout",
    [
        pytest.param(_helper_inside, id="state-directory-inside"),
        pytest.param(_helper_private_directory, id="a-private-directory"),
        pytest.param(_helper_outside_under_a_private_home, id="state-directory-outside"),
        pytest.param(_helper_symlinked_state_database, id="symlinked-state-database"),
        pytest.param(_helper_symlinked_data_file, id="symlinked-data-file"),
    ],
)
def test_memo_records_any_directory_and_the_state_database_holds_no_name(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    layout: Callable[[pathlib.Path, pathlib.Path], pathlib.Path],
) -> None:
    """The memo file is the account's own (mode 0600), so it records directories of any mode,
    wherever the state directory or the state database is. The state database, which other
    accounts can read, holds no untracked name."""
    root = tmp_path / "home" / "project"
    (root / ".pivot").mkdir(parents=True)
    monkeypatch.setattr(project, "_project_root_cache", root)
    _helper_mountinfo(monkeypatch, tmp_path / "mountinfo", root, "ext4")
    _helper_build_tree(root)
    (root / "notes" / "q3-layoff-list").mkdir(parents=True)
    state_dir = layout(root, tmp_path)
    config_io.clear_config_cache()
    assert config.get_state_dir() == state_dir
    _helper_create_state_database()
    _helper_settle(monkeypatch)

    _helper_record(root)
    database = (state_dir / "state.lmdb" / "data.mdb").read_bytes()
    assert b"q3-layoff-list" not in database, "no untracked name reaches the state database"

    _helper_assert_matches_main(track.discover_pvt_files(root), root)
    assert {"notes", "notes/q3-layoff-list"} <= set(_helper_memo()["dirs"]), "memo recorded"
    assert stat.S_IMODE(track._memo_path().stat().st_mode) == 0o600


def _helper_mode(mode: int) -> Callable[[pathlib.Path, pathlib.Path, pytest.MonkeyPatch], None]:
    def _spoil(memo: pathlib.Path, _tmp_path: pathlib.Path, _patch: pytest.MonkeyPatch) -> None:
        memo.chmod(mode)

    return _spoil


def _helper_owned_by_another_account(
    memo: pathlib.Path, _tmp_path: pathlib.Path, patch: pytest.MonkeyPatch
) -> None:
    """An fstat of the memo file reports another uid as its owner."""
    _helper_fstat_reports(patch, memo, uid=os.geteuid() + 1)


def _helper_another_accounts_and_loose(
    memo: pathlib.Path, _tmp_path: pathlib.Path, patch: pytest.MonkeyPatch
) -> None:
    """An fstat of the memo file reports another uid as its owner and mode 0666."""
    _helper_fstat_reports(patch, memo, uid=os.geteuid() + 1, mode=0o066)


def _helper_hard_linked(
    memo: pathlib.Path, tmp_path: pathlib.Path, _patch: pytest.MonkeyPatch
) -> None:
    os.link(memo, tmp_path / "second-link")


def _helper_fstat_reports(
    patch: pytest.MonkeyPatch,
    path: pathlib.Path,
    *,
    uid: int | None = None,
    mode: int = 0,
    device: int | None = None,
) -> None:
    """Make an fstat of ``path``'s file report ``uid`` as its owner, ``mode`` bits added and
    ``device``, as a file another account made, a filesystem that ignores modes, or a mount that
    reports its own owner and mode, would show."""
    real_fstat = os.fstat

    def _fstat(fd: int) -> os.stat_result:
        result = real_fstat(fd)
        if not path.exists() or (result.st_dev, result.st_ino) != (
            path.stat().st_dev,
            path.stat().st_ino,
        ):
            return result
        fields = list(result)
        fields[0] = result.st_mode | mode
        fields[2] = result.st_dev if device is None else device
        fields[4] = result.st_uid if uid is None else uid
        return os.stat_result(fields)

    patch.setattr(os, "fstat", _fstat)


def _helper_listings(
    mocker: MockerFixture, walk: Callable[[], dict[str, track.PvtData]]
) -> tuple[dict[str, track.PvtData], int]:
    """Run ``walk``; return what it found and how many directories it listed."""
    listed = mocker.spy(os, "scandir")
    found = walk()
    listings = listed.call_count
    mocker.stop(listed)
    return found, listings


@pytest.mark.parametrize(
    "spoil",
    [
        pytest.param(_helper_owned_by_another_account, id="owned-by-another-account"),
        pytest.param(_helper_another_accounts_and_loose, id="another-accounts-at-mode-666"),
    ],
)
def test_memo_file_of_another_account_is_neither_used_nor_touched(
    memo_project: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    spoil: Callable[[pathlib.Path, pathlib.Path, pytest.MonkeyPatch], None],
) -> None:
    """A memo file whose fstat shows another owner is refused by both walks, which list every
    directory, and the pre-flight never writes, replaces or removes it, loose or not."""
    _helper_build_tree(memo_project)
    _helper_settle(monkeypatch)
    _helper_record(memo_project)
    path = track._memo_path()
    before, inode = path.read_bytes(), path.stat().st_ino
    time.sleep(_ONE_TICK_S)
    _helper_add_pvt(memo_project)
    directories = len(_helper_directories(memo_project))
    spoil(path, tmp_path, monkeypatch)

    found, read_only_listings = _helper_listings(
        mocker, lambda: track.discover_pvt_files(memo_project)
    )
    recorded, preflight_listings = _helper_listings(mocker, lambda: _helper_preflight(memo_project))

    assert read_only_listings == directories, "the read-only walk lists every directory"
    assert preflight_listings == directories, "so does the pre-flight"
    _helper_assert_matches_main(found, memo_project)
    _helper_assert_matches_main(recorded, memo_project)
    assert (path.read_bytes(), path.stat().st_ino) == (before, inode), "the file is left as it was"


@pytest.mark.parametrize(
    "spoil",
    [
        pytest.param(_helper_mode(0o644), id="mode-644"),
        pytest.param(_helper_mode(0o640), id="group-may-read"),
        pytest.param(_helper_mode(0o620), id="group-may-write"),
        pytest.param(_helper_mode(0o604), id="others-may-read"),
        pytest.param(_helper_mode(0o602), id="others-may-write"),
        pytest.param(_helper_hard_linked, id="hard-linked"),
    ],
)
def test_loosened_memo_is_replaced_by_a_new_private_file(
    memo_project: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
    spoil: Callable[[pathlib.Path, pathlib.Path, pytest.MonkeyPatch], None],
) -> None:
    """The account's own memo with group or other bits, or a second link, (``chmod -R go+rX
    .pivot``, say) is refused by the read-only walk, which leaves it. The pre-flight empties it,
    so no descriptor opened on it and no other link keeps its names, and records the memo in a
    new 0600 file with one link; a descriptor opened before reaches only the old, empty file."""
    _helper_build_tree(memo_project)
    _helper_settle(monkeypatch)
    _helper_record(memo_project)
    path = track._memo_path()
    old, inode = path.read_bytes(), path.stat().st_ino
    held = os.open(path, os.O_RDWR)
    try:
        time.sleep(_ONE_TICK_S)
        _helper_add_pvt(memo_project)
        directories = len(_helper_directories(memo_project))
        spoil(path, tmp_path, monkeypatch)
        caplog.set_level(logging.DEBUG, logger=track.logger.name)

        found, read_only_listings = _helper_listings(
            mocker, lambda: track.discover_pvt_files(memo_project)
        )
        assert read_only_listings == directories, "the read-only walk lists every directory"
        assert "Replacing the .pvt walk memo" not in caplog.text, "and tries no replacement"
        assert (path.read_bytes(), path.stat().st_ino) == (old, inode), "it only refuses"
        recorded, preflight_listings = _helper_listings(
            mocker, lambda: _helper_preflight(memo_project)
        )
        new = path.stat()

        assert preflight_listings == directories, "the pre-flight lists every directory once"
        _helper_assert_matches_main(found, memo_project)
        _helper_assert_matches_main(recorded, memo_project)
        assert new.st_ino != inode, "the memo is a new file"
        assert (stat.S_IMODE(new.st_mode), new.st_nlink) == (0o600, 1)
        assert os.fstat(held).st_size == 0, "the old file holds no name"
        if (tmp_path / "second-link").exists():
            assert (tmp_path / "second-link").read_bytes() == b"", "nor does its other link"
        os.pwrite(held, old, 0)
        assert set(_helper_memo()["dirs"]) == _helper_relative_directories(memo_project), (
            "a write through the earlier descriptor does not reach the memo"
        )
    finally:
        os.close(held)
    _, warm_listings = _helper_listings(mocker, lambda: track.discover_pvt_files(memo_project))
    assert warm_listings == 0, "the new memo is used"


def test_memo_the_open_file_cannot_be_read_is_not_used(
    memo_project: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A read of the open memo that the filesystem refuses (an I/O error on a failing disk) is
    not raised: both walks list every directory, find what main finds, and the file is left as
    it is."""
    _helper_build_tree(memo_project)
    _helper_settle(monkeypatch)
    _helper_record(memo_project)
    path = track._memo_path()
    before, inode = path.read_bytes(), path.stat().st_ino
    directories = len(_helper_directories(memo_project))

    def _unreadable(_fd: int, _size: int, _offset: int) -> bytes:
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr(os, "pread", _unreadable)
    caplog.set_level(logging.DEBUG, logger=track.logger.name)

    found, read_only_listings = _helper_listings(
        mocker, lambda: track.discover_pvt_files(memo_project)
    )
    recorded, preflight_listings = _helper_listings(mocker, lambda: _helper_preflight(memo_project))

    assert "cannot be read: [Errno 5]" in caplog.text
    assert (read_only_listings, preflight_listings) == (directories, directories)
    _helper_assert_matches_main(found, memo_project)
    _helper_assert_matches_main(recorded, memo_project)
    assert (path.read_bytes(), path.stat().st_ino) == (before, inode), "the file is left as it is"


def test_replacement_never_removes_a_file_that_took_the_memos_path(
    memo_project: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another file is renamed onto the memo's path while the pre-flight replaces a loosened
    memo: the pre-flight removes only the file it checked, so the other file stays, and that
    walk has no memo."""
    _helper_build_tree(memo_project)
    _helper_settle(monkeypatch)
    _helper_record(memo_project)
    path = track._memo_path()
    path.chmod(0o644)
    other = path.parent / "other"
    other.write_bytes(b"other")
    other_inode = other.stat().st_ino
    real_lstat = os.lstat

    def _swap_then_lstat(
        target: str | os.PathLike[str], *, dir_fd: int | None = None
    ) -> os.stat_result:
        if os.fspath(target) == os.fspath(path) and other.exists():
            other.replace(path)
        return real_lstat(target, dir_fd=dir_fd)

    monkeypatch.setattr(os, "lstat", _swap_then_lstat)

    _helper_assert_matches_main(_helper_preflight(memo_project), memo_project)

    assert not other.exists(), "the other file took the memo's path"
    assert (path.stat().st_ino, path.read_bytes()) == (other_inode, b"other"), "and stays"


def test_new_memo_file_others_could_read_is_removed(
    memo_project: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check runs on a new file too: on a filesystem that shows the new file with group and
    other bits (one that ignores modes), the pre-flight records nothing and removes it."""
    _helper_build_tree(memo_project)
    _helper_settle(monkeypatch)
    _helper_fstat_reports(monkeypatch, track._memo_path(), mode=0o044)

    _helper_assert_matches_main(_helper_preflight(memo_project), memo_project)

    assert not track._memo_path().exists(), "created, refused and removed"


_OTHER_DEVICE = os.makedev(0, 4250)


def _helper_memo_on(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    root: pathlib.Path,
    project: str,
    memo: str,
) -> None:
    """Mount the project's device as ``project`` and the state database's directory, with the
    memo file in it, as ``memo`` on a device of its own (an fstat of the memo file and a stat of
    the directory report it)."""
    path = track._memo_path()
    _helper_write_mountinfo(
        monkeypatch,
        tmp_path / "mountinfo-memo",
        [
            _helper_mount_line(91, os.stat(root).st_dev, "/mnt/project", project),
            _helper_mount_line(94, _OTHER_DEVICE, os.path.realpath(path.parent), memo),
        ],
    )
    _helper_fstat_reports(monkeypatch, path, device=_OTHER_DEVICE)
    real_stat = os.stat

    def _stat(
        target: str | os.PathLike[str], *, dir_fd: int | None = None, follow_symlinks: bool = True
    ) -> os.stat_result:
        result = real_stat(target, dir_fd=dir_fd, follow_symlinks=follow_symlinks)
        if os.fspath(target) != os.fspath(path.parent):
            return result
        return os.stat_result((*tuple(result)[:2], _OTHER_DEVICE, *tuple(result)[3:]))

    monkeypatch.setattr(os, "stat", _stat)


def test_directories_off_the_allowlist_are_not_recorded_in_a_memo_on_it(
    memo_project: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """The project on NFS and its memo on ext4: the memo is created and written, but records no
    directory of the project, and every walk lists every directory."""
    _helper_build_tree(memo_project)
    _helper_settle(monkeypatch)
    _helper_memo_on(monkeypatch, tmp_path, memo_project, "nfs4", "ext4")

    _helper_record(memo_project)
    _, listings = _helper_listings(mocker, lambda: track.discover_pvt_files(memo_project))

    memo = _helper_memo()
    assert memo["root"] == str(memo_project.resolve()), "the memo file is used"
    assert memo["dirs"] == {}, "no directory on NFS is recorded"
    assert listings == len(_helper_directories(memo_project)), "every directory is listed"


@pytest.mark.parametrize(
    ("fstype", "used"),
    [
        pytest.param("fuse.bindfs", False, id="bindfs"),
        pytest.param("cifs", False, id="cifs"),
        pytest.param("nfs4", False, id="nfs"),
        pytest.param("ext4", True, id="ext4"),
    ],
)
def test_memo_on_a_filesystem_off_the_allowlist_is_not_used_or_created(
    memo_project: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    fstype: str,
    used: bool,
) -> None:
    """bindfs, CIFS and NFS can report the mount's owner and mode instead of the file's, so a
    memo whose device mountinfo lists off the allowlist is used and written by neither walk, and
    the pre-flight creates none there, while the project's directories stay on ext4."""
    _helper_build_tree(memo_project)
    _helper_settle(monkeypatch)
    _helper_record(memo_project)
    path = track._memo_path()
    before = path.read_bytes()
    assert track._unpack_memo(before) is not None, "the memo is recorded"
    _helper_memo_on(monkeypatch, tmp_path, memo_project, "ext4", fstype)
    directories = len(_helper_directories(memo_project))

    found, read_only_listings = _helper_listings(
        mocker, lambda: track.discover_pvt_files(memo_project)
    )
    recorded, preflight_listings = _helper_listings(mocker, lambda: _helper_preflight(memo_project))
    path.unlink()
    _helper_preflight(memo_project)

    assert read_only_listings == (0 if used else directories)
    assert preflight_listings == (0 if used else directories)
    _helper_assert_matches_main(found, memo_project)
    _helper_assert_matches_main(recorded, memo_project)
    if not used:
        assert not path.exists(), "no memo is created there"


def test_memo_path_that_is_a_symlink_is_not_followed(
    memo_project: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """A symlink at the memo's path is not followed, even to a private file of the account's
    own that would pass the check: the walks list every directory and nothing is written
    through it."""
    _helper_build_tree(memo_project)
    _helper_settle(monkeypatch)
    _helper_record(memo_project)
    path = track._memo_path()
    target = tmp_path / "private-file"
    shutil.copy2(path, target)
    path.unlink()
    path.symlink_to(target)
    before = target.read_bytes()
    time.sleep(_ONE_TICK_S)
    _helper_add_pvt(memo_project)
    directories = len(_helper_directories(memo_project))

    listed = mocker.spy(os, "scandir")
    found = track.discover_pvt_files(memo_project)
    read_only_listings = listed.call_count
    recorded = _helper_preflight(memo_project)

    assert read_only_listings == directories
    _helper_assert_matches_main(found, memo_project)
    _helper_assert_matches_main(recorded, memo_project)
    assert path.is_symlink() and target.read_bytes() == before, "nothing written through it"


def test_memo_check_reads_the_open_file_not_its_path(
    memo_project: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """The memo file is replaced by a private one just after the walk opens a mode-644 one: the
    check must judge the open file, so the walk lists every directory."""
    _helper_build_tree(memo_project)
    _helper_settle(monkeypatch)
    _helper_record(memo_project)
    path = track._memo_path()
    private = tmp_path / "private-memo"
    shutil.copy2(path, private)
    path.chmod(0o644)
    real_open = os.open

    def _open_then_replace(
        file: str | os.PathLike[str], flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        fd = real_open(file, flags, mode, dir_fd=dir_fd)
        if os.fspath(file) == os.fspath(path) and private.exists():
            os.replace(private, path)
        return fd

    monkeypatch.setattr(os, "open", _open_then_replace)
    listed = mocker.spy(os, "scandir")
    found = track.discover_pvt_files(memo_project)
    listings = listed.call_count

    assert not private.exists(), "the private file replaced the open one"
    assert listings == len(_helper_directories(memo_project)), "every directory is listed"
    _helper_assert_matches_main(found, memo_project)


def test_fifo_at_the_memo_path_is_not_waited_on(
    memo_project: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A FIFO at the memo's path (which anyone who may write the directory can make) does not
    hold up either walk: both list every directory."""
    _helper_build_tree(memo_project)
    _helper_settle(monkeypatch)
    fifo = track._memo_path()
    os.mkfifo(fifo, 0o600)

    def _release() -> None:
        with fifo.open("wb"):
            pass

    found = _helper_run_without_waiting(lambda: track.discover_pvt_files(memo_project), _release)
    recorded = _helper_run_without_waiting(lambda: _helper_preflight(memo_project), _release)

    _helper_assert_matches_main(found, memo_project)
    _helper_assert_matches_main(recorded, memo_project)
    assert stat.S_ISFIFO(os.lstat(fifo).st_mode), "nothing replaced it"


def _helper_fifo(path: pathlib.Path, _patch: pytest.MonkeyPatch) -> Callable[[], None]:
    os.mkfifo(path, 0o600)

    def _release() -> None:  # frees a walk that waits on it, so the test fails instead of hanging
        with path.open("wb"):
            pass

    return _release


def _helper_directory(path: pathlib.Path, _patch: pytest.MonkeyPatch) -> Callable[[], None]:
    path.mkdir(mode=0o700)
    return lambda: None


def _helper_socket(path: pathlib.Path, patch: pytest.MonkeyPatch) -> Callable[[], None]:
    """A Unix socket file, bound by its short relative name (the absolute path may pass the
    108-byte limit of a socket address); the file stays after the socket is closed."""
    patch.chdir(path.parent)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as bound:
        bound.bind(path.name)
    return lambda: None


@pytest.mark.parametrize(
    ("make", "read_only_reason"),
    [
        pytest.param(_helper_fifo, "is not a regular file (prw-------)", id="fifo"),
        pytest.param(_helper_directory, "is not a regular file (drwx------)", id="directory"),
        pytest.param(_helper_socket, "No such device or address", id="socket"),
    ],
)
def test_memo_path_that_is_no_regular_file_is_refused(
    memo_project: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
    make: Callable[[pathlib.Path, pytest.MonkeyPatch], Callable[[], None]],
    read_only_reason: str,
) -> None:
    """The memo must be a regular file: a FIFO or a directory at its path is refused for that
    reason, and a socket cannot be opened at all. Neither walk waits on what is there, both list
    every directory, and nothing replaces it."""
    _helper_build_tree(memo_project)
    _helper_settle(monkeypatch)
    path = track._memo_path()
    release = make(path, monkeypatch)
    kind = stat.S_IFMT(os.lstat(path).st_mode)
    directories = len(_helper_directories(memo_project))
    caplog.set_level(logging.DEBUG, logger=track.logger.name)

    listed = mocker.spy(os, "scandir")
    found = _helper_run_without_waiting(lambda: track.discover_pvt_files(memo_project), release)
    read_only_listings = listed.call_count
    read_only_log = caplog.text
    listed.reset_mock()
    recorded = _helper_run_without_waiting(lambda: _helper_preflight(memo_project), release)
    preflight_listings = listed.call_count

    assert read_only_reason in read_only_log
    assert (read_only_listings, preflight_listings) == (directories, directories)
    _helper_assert_matches_main(found, memo_project)
    _helper_assert_matches_main(recorded, memo_project)
    assert stat.S_IFMT(os.lstat(path).st_mode) == kind, "nothing replaced it"


def test_memo_file_content_round_trips_and_rejects_a_partial_memo() -> None:
    memo = track._Memo(
        root="/p", dirs={"": b"root", "a": b"\0\n\xff", os.fsdecode(b"caf\xe9/x"): b"x"}
    )
    content = track._pack_memo(memo["root"], memo["dirs"])

    assert track._unpack_memo(content) == memo
    for partial in (b"", content[:5], content[:-1], content + b"\0"):
        assert track._unpack_memo(partial) is None, partial


def test_memo_vouched_directory_still_skips_a_pvt_whose_path_leaves_the_project(
    memo_project: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The memo holds names only, so every .pvt it vouches for is still read: on the warm walk,
    which lists no directory, a .pvt whose ``path:`` resolves outside the project through a
    symlink is still skipped with ``restore_target``'s warning, and everything else is found
    as main finds it."""
    _helper_build_tree(memo_project)
    outside = tmp_path / "outside"
    outside.mkdir()
    (memo_project / "c" / "link").symlink_to(outside)
    track.write_pvt_file(
        memo_project / "c" / "escape.csv.pvt",
        track.PvtData(path="link/escape.csv", hash="0" * 16, size=1),
    )
    _helper_settle(monkeypatch)
    _helper_record(memo_project)
    caplog.set_level(logging.WARNING, logger=track.logger.name)
    caplog.clear()

    found, listings = _helper_listings(mocker, lambda: track.discover_pvt_files(memo_project))

    assert listings == 0, "the memo vouches for every directory: each is stat'ed, not listed"
    expected = {
        key: data
        for key, data in _helper_main_walk(memo_project).items()
        if data["path"] != "link/escape.csv"
    }
    assert found == expected, "the escaping .pvt is skipped, everything else is found"
    assert list(found) == list(expected), "in main's order"
    assert "outside the project root" in caplog.text, "with restore_target's warning"
