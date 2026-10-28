"""`--tui-log` and `import --out`, into a configured state directory that does not exist yet,
make that directory with its `.gitignore`.

Each of these commands creates the parent of a path the user named, before any state-aware
creator runs. A plain `mkdir` there would leave the state directory without its ignore file
for good: every later creator finds the directory already there and writes none, so git sees
the state database, the locks and the cache. So each goes through `state_dir.make_dirs`, as
the parent of a `get -o` destination does (`test_get_output_inside_state_dir.py`).
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest
import yaml

from tests import helpers

if TYPE_CHECKING:
    import pathlib

_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def process(value: str) -> str:
    return value.upper()


with Pipeline("test") as pipeline:
    process(pipeline.input("input.txt", t=str))
"""

_SOURCE_LOCK = {
    "code_manifest": {},
    "params": {},
    "deps": [],
    "outs": [{"key": None, "tag": "data", "hash": "0123456789abcdef", "size": 5}],
}


def _project(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """A committed project whose `core.state_dir` names a directory that does not exist."""
    home = tmp_path / "home"
    root = tmp_path / "project"
    root.mkdir()
    for args in (
        ("git", "init", "-q"),
        ("git", "config", "user.email", "t@example.com"),
        ("git", "config", "user.name", "t"),
    ):
        assert helpers.run_cli(root, home, *args).returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("input\n")
    (root / "pipeline.py").write_text(_PIPELINE)
    (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: altstate\n")
    assert not (root / "altstate").exists()
    return home, root


def _assert_state_dir_is_ignored(root: pathlib.Path, home: pathlib.Path, output: str) -> None:
    """The state directory has its ignore file, and git ignores the state database in it.

    The template ignores `/state.lmdb/` as a directory, so the check needs the database the
    next run writes there: the ignore file has to be in place before that run, because no
    creator writes one into a directory that already exists.
    """
    assert (root / "altstate" / ".gitignore").is_file(), (
        output
        + "\n"
        + "\n".join(sorted(p.relative_to(root).as_posix() for p in (root / "altstate").rglob("*")))
    )
    assert (root / "altstate" / "state.lmdb").is_dir(), output
    ignored = helpers.run_cli(root, home, "git", "check-ignore", "-q", "altstate/state.lmdb")
    assert ignored.returncode == 0, "altstate/state.lmdb is not ignored by git"


@pytest.mark.parametrize(
    "args",
    [
        ("repro", "--tui", "--tui-log", "altstate/tui.log"),
        ("run", "--tui", "--tui-log", "altstate/tui.log", "test/process"),
    ],
    ids=["repro", "run"],
)
def test_a_tui_log_inside_a_missing_state_dir_creates_its_gitignore(
    tmp_path: pathlib.Path, args: tuple[str, ...]
) -> None:
    home, root = _project(tmp_path)

    ran = subprocess.run(
        [helpers.PIVOT, *args],
        cwd=root,
        env=helpers.cli_env(home, TERM="xterm-256color"),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=300,
        check=False,
    )

    output = ran.stdout + ran.stderr
    assert ran.returncode == 0, output
    assert (root / "altstate" / "tui.log").is_file(), output
    _assert_state_dir_is_ignored(root, home, output)


def test_an_import_destination_inside_a_missing_state_dir_creates_its_gitignore(
    tmp_path: pathlib.Path,
) -> None:
    home, root = _project(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    for args in (
        ("git", "init", "-q", "-b", "main"),
        ("git", "config", "user.email", "t@example.com"),
        ("git", "config", "user.name", "t"),
        ("git", "config", "uploadArchive.allowUnreachable", "true"),
    ):
        assert helpers.run_cli(source, home, *args).returncode == 0
    lock = source / ".pivot" / "stages" / "smoke" / "total.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text(yaml.safe_dump(_SOURCE_LOCK))
    (source / ".pivot" / "config.yaml").write_text("remotes:\n  origin: s3://source-cache\n")
    assert helpers.run_cli(source, home, "git", "add", "-A").returncode == 0
    assert helpers.run_cli(source, home, "git", "commit", "-qm", "source").returncode == 0

    imported = helpers.run_pivot(
        root,
        home,
        "import",
        str(source),
        "smoke/total",
        "--out",
        "altstate/got.txt",
        "--no-download",
    )

    output = imported.stdout + imported.stderr
    assert imported.returncode == 0, output
    assert (root / "altstate" / "got.txt.pvt").is_file(), output
    # The import writes no state database. The next command that opens one writes it into the
    # directory the import made, which is where the ignore file has to be already.
    status = helpers.run_pivot(root, home, "status")
    assert status.returncode == 0, status.stdout + status.stderr
    _assert_state_dir_is_ignored(root, home, output + status.stdout + status.stderr)
