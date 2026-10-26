"""The .pvt walk's directory memo: a settled, unchanged directory is stat'ed, not listed.

main's walk (``os.walk`` over the whole project, reading every ``.pvt``) is the oracle: after every
change, cold and warm, the memo walk must find exactly what main finds, in main's order.
"""

from __future__ import annotations

import fcntl
import os
import pathlib
import random
import shutil
import stat
import time
from typing import TYPE_CHECKING

import pytest

from pivot import config, exceptions, project
from pivot.config import io as config_io
from pivot.storage import cache, state, track

if TYPE_CHECKING:
    from collections.abc import Callable

    from pytest_mock import MockerFixture

_REAL_TIME_NS = time.time_ns
_PAST_SETTLE_WINDOW_NS = 10_000_000_000
_INSIDE_SETTLE_WINDOW_NS = 1_000_000_000
_WRITE_TIMEOUT_S = 5.0

# A change this long after a walk carries a later timestamp than the walk saw, on any kernel
# timestamp tick (one jiffy, at most 10 ms). The walk's settle window guarantees that gap in real
# use; a test that moves the clock past the window waits the tick itself.
_ONE_TICK_S = 0.05


@pytest.fixture
def memo_project(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """A project root on a filesystem the memo trusts (mountinfo says ext4)."""
    root = tmp_path / "project"
    (root / ".pivot").mkdir(parents=True)
    monkeypatch.setattr(project, "_project_root_cache", root)
    _helper_mountinfo(monkeypatch, tmp_path / "mountinfo", root, "ext4")
    return root


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
    with state.StateDB(config.get_state_dir()) as state_db:
        return track.discover_pvt_files_and_update_memo(root, state_db)


def _helper_record(root: pathlib.Path) -> None:
    """Record the memo for an unchanging tree.

    The first pre-flight's own write creates the state database's write lock, which changes
    ``.pivot/state.lmdb``; the second records that directory as it then stays.
    """
    _helper_preflight(root)
    _helper_preflight(root)


def _helper_memo() -> state.PvtDirMemo:
    with state.StateDB(config.get_state_dir(), readonly=True) as state_db:
        return state_db.read_pvt_dir_memo()


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
    """After the pre-flight records the memo, a walk lists no directory and stats every one."""
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

    assert listings == 0, "no settled, unchanged directory is listed again"
    assert set(directories[1:]) <= stated, "every directory below the root still gets its lstat"
    _helper_assert_matches_main(warm, memo_project)


def test_read_only_walk_does_not_write_the_memo(
    memo_project: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """discover_pvt_files reads the memo but never records one."""
    _helper_build_tree(memo_project)
    _helper_settle(monkeypatch)
    with state.StateDB(config.get_state_dir()):
        pass  # the database exists, so the read-only walk opens it

    _helper_assert_matches_main(track.discover_pvt_files(memo_project), memo_project)

    assert _helper_memo() == state.PvtDirMemo(root=None, dirs={})


def test_read_only_walk_creates_no_state_database(memo_project: pathlib.Path) -> None:
    """With no state database yet, the read-only walk lists everything and creates none."""
    _helper_build_tree(memo_project)

    _helper_assert_matches_main(track.discover_pvt_files(memo_project), memo_project)

    assert not (config.get_state_dir() / "state.lmdb").exists()


def test_read_only_walk_does_not_create_a_database_in_an_empty_state_directory(
    memo_project: pathlib.Path,
) -> None:
    """An empty state.lmdb directory (no data file yet) is not opened, so nothing is created."""
    _helper_build_tree(memo_project)
    lmdb_dir = config.get_state_dir() / "state.lmdb"
    lmdb_dir.mkdir(parents=True)

    _helper_assert_matches_main(track.discover_pvt_files(memo_project), memo_project)

    assert list(lmdb_dir.iterdir()) == [], "the read-only walk created no database files"


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


def test_preflight_does_not_wait_on_a_busy_database_for_the_memo(
    memo_project: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """While another process holds the write lock, the pre-flight skips recording the memo."""
    _helper_build_tree(memo_project)
    _helper_settle(monkeypatch)
    _helper_record(memo_project)
    before = _helper_memo()
    time.sleep(_ONE_TICK_S)
    _helper_add_pvt(memo_project)
    lock_path = config.get_state_dir() / "state.lmdb" / "pivot-write.lock"

    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        start = time.monotonic()
        found = _helper_preflight(memo_project)
        took = time.monotonic() - start

    assert took < 10.0, f"the memo write gave up quickly, not after the 30 s default ({took:.1f} s)"
    _helper_assert_matches_main(found, memo_project)
    assert _helper_memo() == before, "nothing was recorded while the lock was held"


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


def _helper_freeze_directory_times(
    monkeypatch: pytest.MonkeyPatch, root: pathlib.Path, frozen_ns: int
) -> None:
    """Simulate a filesystem whose timestamp tick has not advanced since ``frozen_ns``.

    Every directory under ``root`` reports that mtime and ctime however it changes, so a change
    in the same tick as a recorded listing leaves the directory's stamp as it was.
    """
    prefix = os.fspath(root)

    def _frozen(result: os.stat_result) -> os.stat_result:
        fields = list(result)
        fields[8] = fields[9] = frozen_ns // 1_000_000_000
        return os.stat_result(
            fields,
            {
                "st_atime": result.st_atime,
                "st_mtime": frozen_ns / 1e9,
                "st_ctime": frozen_ns / 1e9,
                "st_atime_ns": result.st_atime_ns,
                "st_mtime_ns": frozen_ns,
                "st_ctime_ns": frozen_ns,
                "st_blksize": result.st_blksize,
                "st_blocks": result.st_blocks,
                "st_rdev": result.st_rdev,
            },
        )

    def _wrap(real: Callable[..., os.stat_result]) -> Callable[..., os.stat_result]:
        def _stat(path: object, *args: object, **kwargs: object) -> os.stat_result:
            result = real(path, *args, **kwargs)
            inside = isinstance(path, str | os.PathLike) and os.fsdecode(path).startswith(prefix)
            return _frozen(result) if inside and stat.S_ISDIR(result.st_mode) else result

        return _stat

    monkeypatch.setattr(os, "stat", _wrap(os.stat))
    monkeypatch.setattr(os, "lstat", _wrap(os.lstat))


def test_directory_changed_within_the_settle_window_is_not_memoized(
    memo_project: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A listing taken less than 2 s after its directory changed is never recorded.

    With the tick frozen, a later change in that tick leaves the stamp unchanged; only the settle
    window keeps a recorded listing from hiding the new .pvt.
    """
    _helper_build_tree(memo_project)
    frozen_ns = _REAL_TIME_NS()
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
    frozen_ns = _REAL_TIME_NS()
    _helper_freeze_directory_times(monkeypatch, memo_project, frozen_ns)
    monkeypatch.setattr(time, "time_ns", lambda: frozen_ns + _PAST_SETTLE_WINDOW_NS)
    _helper_preflight(memo_project)
    assert _helper_memo()["dirs"], "settled directories are recorded"

    monkeypatch.setattr(time, "time_ns", lambda: frozen_ns + _INSIDE_SETTLE_WINDOW_NS)
    _helper_add_pvt(memo_project)

    _helper_assert_matches_main(track.discover_pvt_files(memo_project), memo_project)
    _helper_assert_matches_main(_helper_preflight(memo_project), memo_project)


# =============================================================================
# Filesystems
# =============================================================================


@pytest.mark.parametrize(
    "fstype",
    [
        pytest.param("nfs4", id="nfs"),
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
    """NFS, overlayfs, FUSE and unknown filesystems get a full listing and no memo."""
    _helper_mountinfo(monkeypatch, tmp_path / "mountinfo", memo_project, fstype)
    _helper_build_tree(memo_project)
    _helper_settle(monkeypatch)

    _helper_assert_matches_main(_helper_preflight(memo_project), memo_project)
    assert _helper_memo()["dirs"] == {}, "nothing is recorded off the allowlist"

    listed = mocker.spy(os, "scandir")
    found = track.discover_pvt_files(memo_project)
    listings = listed.call_count

    assert listings == len(_helper_directories(memo_project)), "every directory is listed"
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


def _helper_truncated(listing: bytes) -> bytes:
    return listing[:35]


def _helper_count_too_large(listing: bytes) -> bytes:
    return listing[:33] + (999).to_bytes(4, "big") + listing[37:]


def _helper_name_with_slash(listing: bytes) -> bytes:
    return listing[:33] + (1).to_bytes(4, "big") + b"../../elsewhere\0x.csv.pvt"


def _helper_empty_name(listing: bytes) -> bytes:
    return listing[:33] + (0).to_bytes(4, "big") + b"x.csv.pvt\0\0new.csv.pvt"


@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param(_helper_truncated, id="truncated"),
        pytest.param(_helper_count_too_large, id="subdirectory-count-too-large"),
        pytest.param(_helper_name_with_slash, id="name-with-a-slash"),
        pytest.param(_helper_empty_name, id="empty-name"),
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
    with state.StateDB(config.get_state_dir()) as state_db:
        memo = state_db.read_pvt_dir_memo()
        assert memo["root"] is not None
        broken = {**memo["dirs"], "a/b": corrupt(memo["dirs"]["a/b"])}
        state_db.update_pvt_dir_memo(memo, memo["root"], broken, timeout=_WRITE_TIMEOUT_S)

    _helper_assert_matches_main(track.discover_pvt_files(memo_project), memo_project)
    _helper_assert_matches_main(_helper_preflight(memo_project), memo_project)
    assert _helper_memo()["dirs"]["a/b"] != broken["a/b"], "the pre-flight records it again"


# =============================================================================
# State database
# =============================================================================


def test_update_pvt_dir_memo_records_root_and_replaces_entries(tmp_path: pathlib.Path) -> None:
    with state.StateDB(tmp_path / "state") as state_db:
        empty = state_db.read_pvt_dir_memo()
        assert empty == state.PvtDirMemo(root=None, dirs={})

        state_db.update_pvt_dir_memo(
            empty, "/p", {"": b"root", "a": b"a1", "a/b": b"ab"}, timeout=_WRITE_TIMEOUT_S
        )
        first = state_db.read_pvt_dir_memo()
        assert first == state.PvtDirMemo(root="/p", dirs={"": b"root", "a": b"a1", "a/b": b"ab"})

        state_db.update_pvt_dir_memo(
            first, "/q", {"": b"root", "a": b"a2", "c": b"c"}, timeout=_WRITE_TIMEOUT_S
        )
        assert state_db.read_pvt_dir_memo() == state.PvtDirMemo(
            root="/q", dirs={"": b"root", "a": b"a2", "c": b"c"}
        ), "changed entries are replaced and unvisited ones deleted"


def test_update_pvt_dir_memo_skips_paths_too_long_for_a_key(tmp_path: pathlib.Path) -> None:
    """A directory whose path does not fit an LMDB key is not recorded: it is listed every walk."""
    long_rel = "/".join(["d" * 100] * 6)
    with state.StateDB(tmp_path / "state") as state_db:
        state_db.update_pvt_dir_memo(
            state_db.read_pvt_dir_memo(),
            "/p",
            {"": b"root", long_rel: b"long"},
            timeout=_WRITE_TIMEOUT_S,
        )
        assert state_db.read_pvt_dir_memo()["dirs"] == {"": b"root"}
