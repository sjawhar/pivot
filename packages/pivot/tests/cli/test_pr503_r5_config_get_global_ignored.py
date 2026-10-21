"""R5 defect 5: `config get --global core.state_dir` presents the ignored value as in effect,
while `config list --global` in the same project marks it `(global, ignored)`."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")


def _run(cwd: pathlib.Path, home: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "HOME": str(home)}
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False, env=env)


def test_config_get_global_marks_the_state_dir_ignored(tmp_path: pathlib.Path) -> None:
    """Both global reporting surfaces say the value is ignored, and warn once."""
    home = tmp_path / "home"
    global_config = home / ".config" / "pivot" / "config.yaml"
    global_config.parent.mkdir(parents=True)
    global_config.write_text("core:\n  state_dir: globalstate\n")
    root = tmp_path / "project"
    root.mkdir()
    assert _run(root, home, "git", "init", "-q").returncode == 0
    assert _run(root, home, _PIVOT, "init").returncode == 0
    warning = (
        f"core.state_dir in {global_config} is ignored; set it in {root / '.pivot' / 'config.yaml'}"
    )

    listed = _run(root, home, _PIVOT, "config", "list", "--global")
    got = _run(root, home, _PIVOT, "config", "get", "--global", "core.state_dir")

    # Control, passes at 392187f8.
    assert "core.state_dir = globalstate (global, ignored)" in listed.stdout, listed.stdout
    assert got.returncode == 0, got.stdout + got.stderr
    assert got.stdout == "core.state_dir = globalstate (global, ignored)\n", got.stdout
    assert got.stderr.splitlines().count(warning) == 1, got.stderr
