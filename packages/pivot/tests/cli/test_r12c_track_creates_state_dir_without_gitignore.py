"""R12 completeness B1: `track` creates a configured state directory without its `.gitignore`.

`core.state_dir: fresh` is set in `.pivot/config.yaml` while `fresh/` does not exist yet (a hand
edit, or a state directory the user deleted). `pivot track` writes its cache under
`fresh/cache/` through `config.get_cache_dir()`, which never calls `project.ensure_state_dir`, so
`fresh/` is made by the cache write with no `.gitignore`. Every later command sees an existing
directory and returns from `ensure_state_dir` before writing one, so the cache and the state
database stay visible to git for good.

The C1 ruling says Pivot writes the state directory's `.gitignore` whenever it creates the
directory. At ce74ad41 the pipeline load created it through `ensure_state_dir` first, so the
`pipeline` case passed; 0.5.2 kept the cache in `.pivot/cache`, so nothing `track` wrote was
visible to git.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")

_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def process(value: str) -> str:
    return value.upper()


with Pipeline("test") as pipeline:
    process(pipeline.input("one.txt", t=str))
"""


def _run(cwd: pathlib.Path, home: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "HOME": str(home)}
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False, env=env)


def _exposed_state_files(root: pathlib.Path, home: pathlib.Path) -> list[str]:
    """Untracked, non-ignored files Pivot writes: anything under `fresh/` or `.pivot/` except
    the ignore file, the lock files meant to be committed, and the config the user edited."""
    status = _run(root, home, "git", "status", "--porcelain", "-uall")
    assert status.returncode == 0, status.stderr
    paths = [line[3:] for line in status.stdout.splitlines() if line.startswith("?? ")]
    return [
        path
        for path in paths
        if path.startswith(("fresh/", ".pivot/"))
        and path not in {"fresh/.gitignore", ".pivot/config.yaml"}
        and not path.startswith("fresh/stages/")
    ]


@pytest.mark.parametrize("with_pipeline", [True, False], ids=["pipeline", "no-pipeline"])
def test_track_keeps_what_it_writes_out_of_git(tmp_path: pathlib.Path, with_pipeline: bool) -> None:
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "project"
    root.mkdir()
    for git_args in (
        ("init", "-q"),
        ("config", "user.email", "test@example.com"),
        ("config", "user.name", "test"),
    ):
        assert _run(root, home, "git", *git_args).returncode == 0
    assert _run(root, home, _PIVOT, "init").returncode == 0
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "one.txt").write_text("one\n")
    if with_pipeline:
        (root / "pipeline.py").write_text(_PIPELINE)
    assert _run(root, home, "git", "add", "-A").returncode == 0
    assert _run(root, home, "git", "commit", "-qm", "base").returncode == 0
    (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: fresh\n")
    (root / "data" / "two.txt").write_text("two\n")

    result = _run(root, home, _PIVOT, "track", "data/two.txt")

    assert result.returncode == 0, result.stdout + result.stderr
    if (root / "fresh").exists():
        assert (root / "fresh" / ".gitignore").is_file(), sorted(
            str(p.relative_to(root)) for p in (root / "fresh").rglob("*")
        )
    assert _exposed_state_files(root, home) == []

    # A later state command does not repair it either.
    assert _run(root, home, _PIVOT, "status").returncode in {0, 1}
    assert _exposed_state_files(root, home) == []
