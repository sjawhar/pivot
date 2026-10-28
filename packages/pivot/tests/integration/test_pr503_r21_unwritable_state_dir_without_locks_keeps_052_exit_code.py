"""Tester R21 red test: a `core.state_dir` that Pivot 0.5.2 failed on still fails.

`core.state_dir: s1`. A run fills `s1`, then its `stages`, `cache` and `locks` directories are
removed and `s1` is made mode 0555, so `s1/state.lmdb` is still a database Pivot can write while
nothing new can be created in `s1`. Pivot 0.5.2, main and `cac249c2` all fail here: `repro` and
`verify` exit 1, because `s1/locks` cannot be created (`PermissionError: [Errno 13] Permission
denied: '<root>/s1/locks'`). Round 26's record calls such a path a fallback and hands every
reader `.pivot`, so the commands now exit 0 and write their locks, cache objects and run record
to `.pivot`.

The documented contract is 0.5.2's exit code (`docs/concepts/pipelines.md`): "Every state-using
command that Pivot 0.5.2 ran keeps its exit code and uses the default `.pivot`; a state-using
command that 0.5.2 failed still fails in one error line", and the fallback's own justification,
"as 0.5.2 ran such commands", does not hold here: 0.5.2 failed. Either the fallback must need
more than a writable `state.lmdb` directory, or the documented promise must change.
"""

from __future__ import annotations

import os
import shutil
import typing

import pytest

from tests import helpers

if typing.TYPE_CHECKING:
    import pathlib

_PIPELINE = """\
from pivot import Pipeline, stage


def ha(value: str) -> str:
    return value.upper()


def hb(value: str) -> str:
    return value + "b"


@stage
def a(value: str) -> str:
    return ha(value)


@stage
def b(value: str) -> str:
    return hb(value)


with Pipeline("test") as pipeline:
    b(a(pipeline.input("input.txt", t=str)))
"""


@pytest.fixture(autouse=True)
def _not_root() -> None:
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permission bits")


def _project(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """A project whose run filled `s1`, with only `s1/state.lmdb` left and `s1` read-only."""
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "project"
    root.mkdir()
    assert helpers.run_cli(root, home, "git", "init", "-q").returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: s1\n")
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("input\n")
    (root / "pipeline.py").write_text(_PIPELINE)
    first = helpers.run_pivot(root, home, "repro")
    assert first.returncode == 0, first.stdout + first.stderr
    assert (root / "s1" / "state.lmdb").is_dir(), sorted(os.listdir(root / "s1"))
    for name in ("stages", "cache", "locks"):
        if (root / "s1" / name).is_dir():
            shutil.rmtree(root / "s1" / name)
    (root / "data" / "raw" / "input.txt").write_text("second\n")
    return home, root


def test_repro_keeps_052s_exit_code_when_the_state_dir_lost_write_permission_and_locks(
    tmp_path: pathlib.Path,
) -> None:
    home, root = _project(tmp_path)
    s1 = root / "s1"
    s1.chmod(0o555)
    try:
        result = helpers.run_pivot(root, home, "repro")
    finally:
        s1.chmod(0o755)

    # Pivot 0.5.2, main and cac249c2 all exit 1 here, on `s1/locks`; their stage failure
    # prints a worker traceback, so only the exit code is compared.
    output = result.stdout + result.stderr
    assert result.returncode == 1, output


def test_verify_keeps_052s_exit_code_when_the_state_dir_lost_write_permission_and_locks(
    tmp_path: pathlib.Path,
) -> None:
    home, root = _project(tmp_path)
    s1 = root / "s1"
    s1.chmod(0o555)
    try:
        ran = helpers.run_pivot(root, home, "repro")
        result = helpers.run_pivot(root, home, "verify")
    finally:
        s1.chmod(0o755)

    output = result.stdout + result.stderr
    # 0.5.2's `repro` failed here, so its `verify` reports no previous run and exits 1.
    assert (ran.returncode, result.returncode) == (1, 1), ran.stdout + ran.stderr + output
