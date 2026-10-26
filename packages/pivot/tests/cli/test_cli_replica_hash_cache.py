"""`pivot status` on a replica of a project reads no output it can vouch for (Dispatch BET-263).

A replica made outside pivot (hard links, copies) carries a byte copy of the state database,
whose file-hash entries name the original project's paths. A hard-linked output is the same
inode as the original, so its entry is found by the file's identity and nothing is read. A
copied output is a new file: the first status reads it and saves its hash, so the second
status reads nothing. `pivot diff` and `pivot plots diff` look files up and save what they read
the same way. The tests drive the installed `pivot` binary and record every file of the
replica's outputs that the process opens.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import pathlib
import shutil
import stat
import subprocess
import sys
import time
from typing import TYPE_CHECKING

import lmdb
import pytest

from pivot.storage import state

if TYPE_CHECKING:
    from collections.abc import Callable
    from multiprocessing.synchronize import Event

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")
_FILES = 12
_OUTPUT_FILES = _FILES + 1  # the tree's files and count's output
_PIPELINE = f"""\
import pathlib
import tempfile

from pivot import Pipeline, stage


@stage
def make_tree() -> pathlib.Path:
    out = pathlib.Path(tempfile.mkdtemp()) / "tree"
    for i in range({_FILES}):
        (out / f"d{{i % 3}}").mkdir(parents=True, exist_ok=True)
        (out / f"d{{i % 3}}" / f"f{{i:02d}}.txt").write_text(f"file {{i}}\\n")
    return out


@stage
def count(tree: pathlib.Path) -> str:
    return str(sum(1 for _ in tree.rglob("*.txt")))


with Pipeline("syn") as pipeline:
    count(make_tree())
"""

# Runs the installed `pivot` script in this interpreter and writes, as JSON, every file under
# argv[1] that the process opened. Reading a file's content needs an open; a stat does not.
_RECORD_OPENS = """\
import json, os, runpy, sys

watched = os.path.realpath(sys.argv[1]) + os.sep
record = sys.argv[2]
opened = []


def hook(event, args):
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.path.realpath(os.fsdecode(args[0]))
        if path.startswith(watched):
            opened.append(path)


sys.addaudithook(hook)
sys.argv = sys.argv[3:]
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
finally:
    with open(record, "w") as f:
        json.dump(sorted(set(opened)), f)
"""


def _env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}


def _ok(proc: subprocess.CompletedProcess[str]) -> str:
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    return out


def _pivot(cwd: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_PIVOT, *args], cwd=cwd, env=_env(), capture_output=True, text=True, timeout=180
    )


def _helper_project(root: pathlib.Path) -> pathlib.Path:
    """A project whose two stages ran once."""
    root.mkdir(parents=True)
    root = root.resolve()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    _ok(_pivot(root, "init"))
    (root / "pipeline.py").write_text(_PIPELINE)
    (root / ".gitignore").write_text("/data/\n__pycache__/\n")
    _ok(_pivot(root, "repro"))
    return root


def _helper_status(root: pathlib.Path, tmp_path: pathlib.Path) -> list[str]:
    """Run `pivot status --json -v` in ``root``; return the output files the process opened.

    Every stage must be reported cached: a replica is a copy of a current project.
    """
    record = tmp_path / f"opened-{time.monotonic_ns()}.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _RECORD_OPENS,
            str(root / "data"),
            str(record),
            _PIVOT,
            "status",
            "--json",
            "-v",
        ],
        cwd=root,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=180,
    )
    _ok(proc)
    stages = json.loads(proc.stdout)["stages"]
    assert [(entry["name"], entry["status"]) for entry in stages] == [
        ("syn/make_tree", "cached"),
        ("syn/count", "cached"),
    ], proc.stdout
    return json.loads(record.read_text())


def _cp_al(src: pathlib.Path, dst: pathlib.Path) -> None:
    """`cp -al`: every file a hard link, the state database's files included."""
    shutil.copytree(src, dst, symlinks=True, copy_function=os.link)


def _bet_replica(src: pathlib.Path, dst: pathlib.Path) -> None:
    """The BET lanes' replica (their make_replica.py): a symlink is recreated, a file with no
    write bit (pivot's outputs and cache objects) is hard-linked, and every other file, the
    state database's included, is copied with `shutil.copy2`."""
    for root, dirs, files in os.walk(src):
        out = dst / pathlib.Path(root).relative_to(src)
        out.mkdir(parents=True, exist_ok=True)
        for name in list(dirs):
            path = pathlib.Path(root) / name
            if path.is_symlink():
                (out / name).symlink_to(os.readlink(path))
                dirs.remove(name)
        for name in files:
            path = pathlib.Path(root) / name
            if path.is_symlink():
                (out / name).symlink_to(os.readlink(path))
            elif path.lstat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
                shutil.copy2(path, out / name)
            else:
                os.link(path, out / name)


def _rsync_a(src: pathlib.Path, dst: pathlib.Path) -> None:
    """`rsync -a`: every file copied, mtimes kept."""
    shutil.copytree(src, dst, symlinks=True)


def _cp_r(src: pathlib.Path, dst: pathlib.Path) -> None:
    """`cp -r`: every file copied with its mode, mtimes not kept."""
    shutil.copytree(src, dst, symlinks=True, copy_function=shutil.copy)


@pytest.mark.parametrize(
    "make_replica",
    [pytest.param(_cp_al, id="cp-al"), pytest.param(_bet_replica, id="bet-replica")],
)
def test_a_hard_linked_replicas_first_status_reads_no_output(
    tmp_path: pathlib.Path, make_replica: Callable[[pathlib.Path, pathlib.Path], None]
) -> None:
    """Each output file of the replica is the original's inode, so the entry its copied state
    database holds for the original's path vouches for it, and status reads none of them."""
    root = _helper_project(tmp_path / "live")
    replica = tmp_path / "replica"
    make_replica(root, replica)

    assert _helper_status(replica, tmp_path) == []


@pytest.mark.parametrize(
    "make_replica", [pytest.param(_rsync_a, id="rsync-a"), pytest.param(_cp_r, id="cp-r")]
)
def test_a_copied_replicas_second_status_reads_no_output(
    tmp_path: pathlib.Path, make_replica: Callable[[pathlib.Path, pathlib.Path], None]
) -> None:
    """A copied output is a new file, so the first status reads it; it saves what it read, so
    the second status reads nothing."""
    root = _helper_project(tmp_path / "live")
    replica = tmp_path / "replica"
    make_replica(root, replica)

    assert len(_helper_status(replica, tmp_path)) == _OUTPUT_FILES, "A copy is a new file"
    assert _helper_status(replica, tmp_path) == []


def _helper_hold_write_lock(state_dir: str, held: Event, release: Event) -> None:
    db = state.StateDB(pathlib.Path(state_dir))
    with db, db._write_transaction(timeout=10.0):
        held.set()
        release.wait(120)


def _helper_dump(state_dir: pathlib.Path) -> dict[bytes, bytes]:
    env = lmdb.open(str(state_dir / "state.lmdb"), readonly=True, lock=False)
    try:
        with env.begin() as txn:
            return dict(txn.cursor())
    finally:
        env.close()


def test_status_against_a_locked_state_db_exits_0_and_saves_once_it_is_free(
    tmp_path: pathlib.Path,
) -> None:
    """While another process holds the state database's write lock, status reports, exits 0,
    and leaves the database as it was, waiting no longer than a pivot write does. Once the
    lock is free, status saves the hashes it read."""
    root = _helper_project(tmp_path / "live")
    replica = tmp_path / "replica"
    _rsync_a(root, replica)
    state_dir = replica / ".pivot"
    before = _helper_dump(state_dir)

    ctx = multiprocessing.get_context("spawn")
    held, release = ctx.Event(), ctx.Event()
    holder = ctx.Process(target=_helper_hold_write_lock, args=(str(state_dir), held, release))
    holder.start()
    try:
        assert held.wait(30), "The holder took the write lock"
        started = time.monotonic()
        assert len(_helper_status(replica, tmp_path)) == _OUTPUT_FILES
        assert time.monotonic() - started < 2.0 + 15, (
            "Status gives up the write after its 2 s budget; a pivot write would wait 30 s"
        )
    finally:
        release.set()
        holder.join(30)
    assert holder.exitcode == 0
    assert _helper_dump(state_dir) == before, "A locked database is left as it was"

    assert len(_helper_status(replica, tmp_path)) == _OUTPUT_FILES, (
        "Nothing was saved under the lock"
    )
    assert _helper_dump(state_dir) != before, "With the lock free, status saves what it read"
    assert _helper_status(replica, tmp_path) == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes through any file mode")
@pytest.mark.parametrize(
    ("make_replica", "lock_writable", "opened"),
    [
        pytest.param(_bet_replica, True, 0, id="bet-replica-served"),
        pytest.param(_rsync_a, True, _OUTPUT_FILES, id="rsync-a-read"),
        pytest.param(_bet_replica, False, _OUTPUT_FILES, id="lock-read-only-not-used"),
    ],
)
def test_status_against_a_read_only_state_db_exits_0_and_leaves_it_unchanged(
    tmp_path: pathlib.Path,
    make_replica: Callable[[pathlib.Path, pathlib.Path], None],
    lock_writable: bool,
    opened: int,
) -> None:
    """A state database this user cannot write still serves its entries while its lock file
    lets this reader register against writers; with a lock file it cannot write either (another
    user's project), the database is not used and status reads the files. Either way status
    reports, exits 0, and leaves the database as it was."""
    root = _helper_project(tmp_path / "live")
    replica = tmp_path / "replica"
    make_replica(root, replica)
    state_dir = replica / ".pivot"
    before = _helper_dump(state_dir)
    lmdb_dir = state_dir / "state.lmdb"
    modes = {
        path: path.stat().st_mode
        for path in [lmdb_dir, *lmdb_dir.iterdir()]
        if path.name != "lock.mdb" or not lock_writable
    }
    for path, mode in modes.items():
        path.chmod(mode & ~0o222)
    try:
        assert len(_helper_status(replica, tmp_path)) == opened
    finally:
        for path, mode in modes.items():
            path.chmod(mode)

    assert _helper_dump(state_dir) == before


# Outputs `pivot diff` compares (JSON files in a directory output) and one plot.
_SHOW_PIPELINE = f"""\
import json
import pathlib
import tempfile
from typing import Annotated

import matplotlib.figure

from pivot import Pipeline, plot, stage


@stage
def make_tables() -> pathlib.Path:
    out = pathlib.Path(tempfile.mkdtemp()) / "tables"
    for i in range({_FILES}):
        (out / f"d{{i % 3}}").mkdir(parents=True, exist_ok=True)
        (out / f"d{{i % 3}}" / f"t{{i:02d}}.json").write_text(json.dumps({{"row": i}}))
    return out


@stage
def chart() -> Annotated[matplotlib.figure.Figure, plot]:
    figure = matplotlib.figure.Figure()
    figure.subplots().plot([1, 2])
    return figure


with Pipeline("show") as pipeline:
    make_tables()
    chart()
"""


def _helper_show_project(root: pathlib.Path) -> pathlib.Path:
    """A project whose stages ran once, with their locks committed for `pivot diff` to read."""
    root.mkdir(parents=True)
    root = root.resolve()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    _ok(_pivot(root, "init"))
    (root / "pipeline.py").write_text(_SHOW_PIPELINE)
    (root / ".gitignore").write_text("/data/\n/plots/\n__pycache__/\n")
    _ok(_pivot(root, "repro"))
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "run"],
        cwd=root,
        check=True,
    )
    return root


def _helper_opens(
    root: pathlib.Path, watched: str, tmp_path: pathlib.Path, *args: str
) -> tuple[list[str], str]:
    """Run `pivot <args>` in ``root``; return the files under ``root / watched`` it opened, and
    its stdout."""
    record = tmp_path / f"opened-{time.monotonic_ns()}.json"
    proc = subprocess.run(
        [sys.executable, "-c", _RECORD_OPENS, str(root / watched), str(record), _PIVOT, *args],
        cwd=root,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=180,
    )
    _ok(proc)
    return json.loads(record.read_text()), proc.stdout


def test_a_copied_replicas_second_diff_reads_no_output(tmp_path: pathlib.Path) -> None:
    """`pivot diff` hashes each workspace file it compares as status does: a copied output is
    a new file, so the first diff reads it; it saves what it read, so the second reads none."""
    root = _helper_show_project(tmp_path / "live")
    replica = tmp_path / "replica"
    _cp_r(root, replica)

    first, out = _helper_opens(replica, "data", tmp_path, "diff", "--no-tui", ".")
    assert "No data file changes detected." in out, out
    assert len(first) == _FILES, "A copy is a new file"
    second, out = _helper_opens(replica, "data", tmp_path, "diff", "--no-tui", ".")
    assert "No data file changes detected." in out, out
    assert second == []


def test_a_copied_replicas_second_plots_diff_reads_no_plot(tmp_path: pathlib.Path) -> None:
    """`pivot plots diff` hashes each plot as status does: the first call on a copy reads the
    plot and saves its hash, so the second reads nothing."""
    root = _helper_show_project(tmp_path / "live")
    replica = tmp_path / "replica"
    _cp_r(root, replica)

    first, out = _helper_opens(replica, "plots", tmp_path, "plots", "diff")
    assert "No plot changes." in out, out
    assert [pathlib.Path(path).name for path in first] == ["chart.png"], "A copy is a new file"
    second, out = _helper_opens(replica, "plots", tmp_path, "plots", "diff")
    assert "No plot changes." in out, out
    assert second == []
