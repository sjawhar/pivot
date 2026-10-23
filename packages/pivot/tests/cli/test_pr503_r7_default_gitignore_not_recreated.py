"""In the default layout, only `init` writes `.pivot/.gitignore`, as on 0.5.2.

A state directory gets its ignore file only when Pivot creates it, and `.pivot` already exists.
So `list`, `dag` and `status` leave a deleted `.pivot/.gitignore` deleted, and run in a `.pivot`
the user cannot write (mode 0555), with or without the file.
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
    process(pipeline.input("input.txt", t=str))
"""


def _run(cwd: pathlib.Path, home: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "HOME": str(home)}
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False, env=env)


def _project(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "project"
    root.mkdir()
    assert _run(root, home, "git", "init", "-q").returncode == 0
    assert _run(root, home, _PIVOT, "init").returncode == 0
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("input\n")
    (root / "pipeline.py").write_text(_PIPELINE)
    assert _run(root, home, _PIVOT, "repro").returncode == 0
    return home, root


@pytest.mark.parametrize("args", [("list",), ("dag",), ("status",)], ids=lambda a: "-".join(a))
def test_read_commands_do_not_recreate_a_deleted_ignore_file(
    tmp_path: pathlib.Path, args: tuple[str, ...]
) -> None:
    """Like 0.5.2, a command other than `init` leaves a deleted `.pivot/.gitignore` deleted."""
    home, root = _project(tmp_path)
    (root / ".pivot" / ".gitignore").unlink()

    result = _run(root, home, _PIVOT, *args)

    assert result.returncode == 0, result.stdout + result.stderr
    assert not (root / ".pivot" / ".gitignore").exists()


@pytest.mark.parametrize("layout", ["absent", "present"])
@pytest.mark.parametrize("args", [("list",), ("dag",), ("status",)], ids=lambda a: "-".join(a))
def test_read_commands_run_in_a_read_only_pivot_directory(
    tmp_path: pathlib.Path, layout: str, args: tuple[str, ...]
) -> None:
    """Like 0.5.2, these exit 0 when `.pivot` is mode 0555, with or without the ignore file."""
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permission bits")
    home, root = _project(tmp_path)
    if layout == "absent":
        (root / ".pivot" / ".gitignore").unlink()
    (root / ".pivot").chmod(0o555)
    try:
        result = _run(root, home, _PIVOT, *args)
    finally:
        (root / ".pivot").chmod(0o755)

    output = result.stdout + result.stderr
    assert "Traceback" not in output, output
    assert result.returncode == 0, output
