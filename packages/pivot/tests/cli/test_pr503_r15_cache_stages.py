"""Round 15 C2: a cache may not overlap the directory containing committed stage locks."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")
_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def process(value: str) -> str:
    return value.upper()


with Pipeline("test") as pipeline:
    process(pipeline.input("input.txt", t=str))
"""


def _run(
    cwd: pathlib.Path, home: pathlib.Path, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "HOME": str(home), **(env or {})},
    )


def _project(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "project"
    root.mkdir()
    for args in (
        ("init", "-q"),
        ("config", "user.email", "t@example.com"),
        ("config", "user.name", "t"),
    ):
        assert _run(root, home, "git", *args).returncode == 0
    assert _run(root, home, _PIVOT, "init").returncode == 0
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("input\n")
    (root / "pipeline.py").write_text(_PIPELINE)
    assert _run(root, home, _PIVOT, "config", "set", "core.state_dir", "altstate").returncode == 0
    return home, root


def _assert_lock_is_trackable(root: pathlib.Path, home: pathlib.Path) -> None:
    lock = root / "altstate" / "stages" / "test" / "process.lock"
    assert lock.is_file()
    ignored = _run(root, home, "git", "check-ignore", "-q", str(lock.relative_to(root)))
    assert ignored.returncode == 1, ignored.stdout + ignored.stderr
    assert _run(root, home, "git", "add", "-A").returncode == 0
    staged = _run(root, home, "git", "diff", "--cached", "--name-only")
    assert str(lock.relative_to(root)) in staged.stdout.splitlines(), staged.stdout


def test_cache_dir_equal_to_stages_warns_and_falls_back_to_state_cache(
    tmp_path: pathlib.Path,
) -> None:
    home, root = _project(tmp_path)
    assert _run(root, home, _PIVOT, "config", "set", "cache.dir", "altstate/stages").returncode == 0

    result = _run(root, home, _PIVOT, "repro")

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stderr.count("overlaps the stage-lock directory") == 1, result.stderr
    assert (root / "altstate" / "cache" / "files").is_dir()
    _assert_lock_is_trackable(root, home)


def test_environment_cache_dir_under_stages_warns_and_falls_back_to_state_cache(
    tmp_path: pathlib.Path,
) -> None:
    home, root = _project(tmp_path)

    result = _run(root, home, _PIVOT, "repro", env={"PIVOT_CACHE_DIR": "altstate/stages/cache"})

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stderr.count("overlaps the stage-lock directory") == 1, result.stderr
    assert (root / "altstate" / "cache" / "files").is_dir()
    _assert_lock_is_trackable(root, home)


def test_cache_dir_containing_stages_warns_and_falls_back_to_state_cache(
    tmp_path: pathlib.Path,
) -> None:
    home, root = _project(tmp_path)
    assert _run(root, home, _PIVOT, "config", "set", "cache.dir", "altstate").returncode == 0

    result = _run(root, home, _PIVOT, "repro")

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stderr.count("overlaps the stage-lock directory") == 1, result.stderr
    assert (root / "altstate" / "cache" / "files").is_dir()
    _assert_lock_is_trackable(root, home)
