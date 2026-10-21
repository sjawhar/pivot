"""R7 defect G5: `config list --global` labels a shadowed global `core.state_dir` as `(local)`.

The J1 ruling (21:15Z) says a global `core.state_dir` is ignored and `config get` and `config list`
mark it as ignored. With a global `core.state_dir: globalstate` and a project that sets its own
`core.state_dir: localstate`, `config get --global core.state_dir` prints
`core.state_dir = globalstate (global, ignored)`, but `config list --global` prints
`core.state_dir = globalstate (local)`: the global file's value under the local label.
`cli/config.py:72` takes the label from `get_config_value`, which answers for the merged config
(local wins), while the value comes from the global file. 0.5.2 prints `(local)` here too, so this
is not a regression; it is a case inside J1's contract that misbehaves.

The unshadowed project is the control: `config list --global` prints `(global, ignored)` there.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")


def _run(cwd: pathlib.Path, home: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "HOME": str(home)}
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False, env=env)


@pytest.mark.parametrize("shadowed", [True, False], ids=["shadowed", "unshadowed-control"])
def test_config_list_global_marks_the_global_state_dir_ignored(
    tmp_path: pathlib.Path, shadowed: bool
) -> None:
    """The global value is shown with the `(global, ignored)` label, shadowed or not."""
    home = tmp_path / "home"
    global_config = home / ".config" / "pivot" / "config.yaml"
    global_config.parent.mkdir(parents=True)
    global_config.write_text("core:\n  state_dir: globalstate\n")
    root = tmp_path / "project"
    root.mkdir()
    assert _run(root, home, "git", "init", "-q").returncode == 0
    assert _run(root, home, _PIVOT, "init").returncode == 0
    if shadowed:
        (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: localstate\n")

    result = _run(root, home, _PIVOT, "config", "list", "--global")

    assert result.returncode == 0, result.stdout + result.stderr
    rows = [line for line in result.stdout.splitlines() if line.startswith("core.state_dir")]
    assert rows == ["core.state_dir = globalstate (global, ignored)"], result.stdout
