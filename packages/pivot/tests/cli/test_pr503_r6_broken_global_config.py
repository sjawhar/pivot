"""R6 defect 2: a project without core.state_dir fails when ~/.config/pivot/config.yaml is broken.

Round 10's resolver reads the global config on every state lookup to decide whether to warn
(`project.warn_if_global_state_dir_ignored` -> `_read_core_state_dir`), and lets a YAML error or a
PermissionError escape. With an invalid or unreadable global file, `init`, `list`, `dag`,
`plots show` and `init --force` now exit 1, where 0.5.2 and 392187f8 exit 0, and `status` and
`history` lose 0.5.2's error message.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

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


def _setup(tmp_path: pathlib.Path, kind: str) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    clean_home = tmp_path / "clean-home"
    clean_home.mkdir()
    root = tmp_path / "project"
    root.mkdir()
    assert _run(root, clean_home, "git", "init", "-q").returncode == 0
    assert _run(root, clean_home, _PIVOT, "init").returncode == 0
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("input\n")
    (root / "pipeline.py").write_text(_PIPELINE)
    assert _run(root, clean_home, _PIVOT, "repro").returncode == 0
    home = tmp_path / "home"
    global_config = home / ".config" / "pivot" / "config.yaml"
    global_config.parent.mkdir(parents=True)
    if kind == "invalid-yaml":
        global_config.write_text("core: [unclosed\n")
    else:
        global_config.write_text("display:\n  precision: 3\n")
        global_config.chmod(0o000)
    return home, root, global_config


@pytest.fixture(params=["invalid-yaml", "unreadable"])
def broken(
    request: pytest.FixtureRequest, tmp_path: pathlib.Path
) -> Iterator[tuple[str, pathlib.Path, pathlib.Path, pathlib.Path]]:
    kind = str(request.param)
    if kind == "unreadable" and os.geteuid() == 0:
        pytest.skip("root ignores file permission bits")
    home, root, global_config = _setup(tmp_path, kind)
    try:
        yield kind, home, root, global_config
    finally:
        global_config.chmod(0o644)


@pytest.mark.parametrize(
    "args",
    [("init",), ("list",), ("dag",), ("plots", "show"), ("init", "--force")],
    ids=lambda args: "-".join(args),
)
def test_commands_that_never_read_global_config_still_work(
    broken: tuple[str, pathlib.Path, pathlib.Path, pathlib.Path],
    tmp_path: pathlib.Path,
    args: tuple[str, ...],
) -> None:
    """0.5.2 runs these with a broken global config: none of them read it there."""
    _, home, root, _ = broken
    cwd = root
    if args == ("init",):
        cwd = tmp_path / "fresh"
        cwd.mkdir()
        assert _run(cwd, home, "git", "init", "-q").returncode == 0

    result = _run(cwd, home, _PIVOT, *args)

    output = result.stdout + result.stderr
    assert "Traceback" not in output, output
    assert result.returncode == 0, output


def test_second_init_after_a_broken_global_config_behaves_like_052(
    broken: tuple[str, pathlib.Path, pathlib.Path, pathlib.Path], tmp_path: pathlib.Path
) -> None:
    """The first init leaves a whole project, so a second init refuses as 0.5.2 does."""
    _, home, _, _ = broken
    cwd = tmp_path / "fresh"
    cwd.mkdir()
    assert _run(cwd, home, "git", "init", "-q").returncode == 0

    first = _run(cwd, home, _PIVOT, "init")
    second = _run(cwd, home, _PIVOT, "init")

    assert first.returncode == 0, first.stdout + first.stderr
    assert sorted(p.name for p in (cwd / ".pivot").iterdir()) == [".gitignore", "stages"]
    assert second.returncode == 1, second.stdout + second.stderr
    assert second.stdout == "", second.stdout
    assert second.stderr == (
        f"Error: Pivot already initialized in {cwd}\n\nTip: Use --force to reinitialize\n"
    ), second.stderr


@pytest.mark.parametrize("args", [("status",), ("history",)], ids=lambda args: "-".join(args))
def test_commands_that_read_global_config_keep_the_052_message(
    broken: tuple[str, pathlib.Path, pathlib.Path, pathlib.Path], args: tuple[str, ...]
) -> None:
    """0.5.2 exits 1 here with one `Error:` line naming the file (392187f8's history exited 0)."""
    kind, home, root, global_config = broken

    result = _run(root, home, _PIVOT, *args)

    assert result.returncode == 1, result.stdout + result.stderr
    errors = [line for line in result.stderr.splitlines() if line.startswith("Error")]
    expected = (
        f"Error: Invalid YAML in {global_config}:"
        if kind == "invalid-yaml"
        else f"Error: Permission denied reading {global_config}"
    )
    assert errors and errors[0].startswith(expected), result.stderr
