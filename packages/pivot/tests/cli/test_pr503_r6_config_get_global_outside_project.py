"""R6 defect 1: `config get --global core.state_dir` outside a project exits 1 with a traceback.

Pivot 0.5.2 (and 392187f8) print the global value and exit 0. Round 10's N4 change calls
`project.report_ignored_global_state_dir()`, which needs a project root.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")


def _run(cwd: pathlib.Path, home: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "HOME": str(home)}
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False, env=env)


def _outside_any_project(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    home = tmp_path / "home"
    global_config = home / ".config" / "pivot" / "config.yaml"
    global_config.parent.mkdir(parents=True)
    global_config.write_text("core:\n  state_dir: globalstate\n")
    cwd = tmp_path / "not-a-project"
    cwd.mkdir()
    assert not any((parent / ".pivot").exists() for parent in (cwd, *cwd.parents)), (
        "test needs no .pivot above"
    )
    return home, cwd


@pytest.mark.parametrize("output_json", [False, True], ids=["text", "json"])
def test_config_get_global_state_dir_works_outside_a_project(
    tmp_path: pathlib.Path, output_json: bool
) -> None:
    """Like 0.5.2, the global value is printed and the command exits 0; no project ignores it."""
    home, cwd = _outside_any_project(tmp_path)
    args = ["config", "get", "--global", "core.state_dir", *(["--json"] if output_json else [])]

    result = _run(cwd, home, _PIVOT, *args)

    output = result.stdout + result.stderr
    assert "Traceback" not in output, output
    assert result.returncode == 0, output
    if output_json:
        assert json.loads(result.stdout) == {
            "key": "core.state_dir",
            "value": "globalstate",
            "source": "global",
        }, result.stdout
    else:
        assert result.stdout == "core.state_dir = globalstate (global)\n", result.stdout


@pytest.mark.parametrize("output_json", [False, True], ids=["text", "json"])
def test_config_get_global_state_dir_ignores_an_invalid_local_config(
    tmp_path: pathlib.Path, output_json: bool
) -> None:
    """Like 0.5.2, labeling a global value never parses the project's own config."""
    home, cwd = _outside_any_project(tmp_path)
    (cwd / ".pivot").mkdir()
    (cwd / ".pivot" / "config.yaml").write_text("core: [unclosed\n")
    args = ["config", "get", "--global", "core.state_dir", *(["--json"] if output_json else [])]

    result = _run(cwd, home, _PIVOT, *args)

    output = result.stdout + result.stderr
    assert "Traceback" not in output, output
    assert result.returncode == 0, output
    if output_json:
        assert json.loads(result.stdout) == {
            "key": "core.state_dir",
            "value": "globalstate",
            "source": "global, ignored",
        }, result.stdout
    else:
        assert result.stdout == "core.state_dir = globalstate (global, ignored)\n", result.stdout


def test_config_unset_global_state_dir_works_outside_a_project(tmp_path: pathlib.Path) -> None:
    """Control, passes at 3c3cceb8: the fix the warning names works outside a project."""
    home, cwd = _outside_any_project(tmp_path)

    result = _run(cwd, home, _PIVOT, "config", "unset", "--global", "core.state_dir")

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == "Removed core.state_dir from global config\n", result.stdout
