"""`pivot get -o <dest>`, with `<dest>` inside a configured state directory that does not exist
yet, makes that directory with its `.gitignore`, so git ignores the restore lock `get` leaves
there.

Every directory Pivot creates for cache, locks, state or plots comes from `state_dir.make_dirs`
or `state_dir.make_state_dir`, and the first of those writes the ignore file. `get` locates the
state directory and creates nothing until it writes, so the destination's own parent is the first
directory it may create inside the state directory, and it goes through the same creator. A
destination outside the state directory is the control.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

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


@pytest.mark.parametrize(
    ("dest", "target"),
    [
        ("altstate/copied.txt", "data/raw/input.txt"),
        ("altstate/nested/copied.txt", "data/raw/input.txt"),
        ("altstate/out.txt", "data/test/process.txt"),
        ("elsewhere/copied.txt", "data/raw/input.txt"),
    ],
    ids=["file-in-state-dir", "file-in-nested-dir", "output-in-state-dir", "outside-control"],
)
def test_get_output_inside_missing_state_dir_creates_its_gitignore(
    tmp_path: pathlib.Path, dest: str, target: str
) -> None:
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
    ran = helpers.run_pivot(root, home, "repro")
    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert helpers.run_cli(root, home, "git", "add", "-A").returncode == 0
    assert helpers.run_cli(root, home, "git", "commit", "-qm", "base").returncode == 0
    (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: altstate\n")
    assert not (root / "altstate").exists()

    got = helpers.run_pivot(root, home, "get", "--rev", "HEAD", "-o", dest, target)

    assert got.returncode == 0, got.stdout + got.stderr
    assert (root / dest).is_file()
    assert (root / "altstate" / ".gitignore").is_file(), sorted(
        p.relative_to(root).as_posix() for p in (root / "altstate").rglob("*")
    )
    locks = sorted((root / "altstate" / "locks").iterdir())
    for lock in locks:
        rel = lock.relative_to(root).as_posix()
        ignored = helpers.run_cli(root, home, "git", "check-ignore", "-q", rel)
        assert ignored.returncode == 0, f"{rel} is not ignored by git"
