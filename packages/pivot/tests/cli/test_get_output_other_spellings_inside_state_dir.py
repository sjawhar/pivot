"""`pivot get -o <dest>` makes a missing configured state directory with its `.gitignore`
however `<dest>` names a path inside it.

One destination gets one answer: the creator decides on the path the kernel will open
(`state_dir.make_dirs`), not on the spelling the user typed, because the configured value it
compares with is resolved the same way (`state_dir.resolve_state_dir_value`). So a `..`
segment, a relative path from a subdirectory of the project, and an absolute path through a
symbolic link to the project all name the state directory that `-o altstate/copied.txt` names,
and each of them leaves the ignore file that keeps the restore lock in `<state dir>/locks` out
of git. A destination outside the state directory, from the same subdirectory, is the control:
`get` still creates the state directory for its own lock, with the ignore file.
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
    ("cwd", "dest"),
    [
        ("data", "../altstate/copied.txt"),
        (".", "data/../altstate/copied.txt"),
        (".", "elsewhere/../altstate/copied.txt"),
        (".", "{LINK}/altstate/copied.txt"),
        ("data", "../elsewhere/copied.txt"),
    ],
    ids=[
        "from-subdirectory",
        "dotdot-through-existing-dir",
        "dotdot-through-new-dir",
        "absolute-through-symlink",
        "from-subdirectory-outside-control",
    ],
)
def test_get_output_naming_the_missing_state_dir_another_way_creates_its_gitignore(
    tmp_path: pathlib.Path, cwd: str, dest: str
) -> None:
    home = tmp_path / "home"
    root = tmp_path / "project"
    root.mkdir()
    link = tmp_path / "link"
    link.symlink_to(root)
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
    dest = dest.replace("{LINK}", str(link))

    got = helpers.run_pivot(
        root / cwd, home, "get", "--rev", "HEAD", "-o", dest, "data/raw/input.txt"
    )

    assert got.returncode == 0, got.stdout + got.stderr
    assert ((root / cwd) / dest).is_file()
    assert (root / "altstate" / ".gitignore").is_file(), sorted(
        p.relative_to(root).as_posix() for p in (root / "altstate").rglob("*")
    )
    locks = sorted((root / "altstate" / "locks").iterdir())
    for lock in locks:
        rel = lock.relative_to(root).as_posix()
        ignored = helpers.run_cli(root, home, "git", "check-ignore", "-q", rel)
        assert ignored.returncode == 0, f"{rel} is not ignored by git"
