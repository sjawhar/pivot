"""Round 15 B1: configuration displays do not resolve an unusable state directory."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")
_CASES = {
    "empty": "core:\n  state_dir: ''\n",
    "integer": "core:\n  state_dir: 5\n",
    "list": "core:\n  state_dir: [a, b]\n",
    "file": "core:\n  state_dir: afile\n",
}


def _run(cwd: pathlib.Path, home: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "HOME": str(home)},
    )


@pytest.mark.parametrize("config", _CASES.values(), ids=_CASES)
@pytest.mark.parametrize(
    "command",
    [("config", "get", "core.state_dir"), ("config", "get", "cache.dir"), ("config", "list")],
)
def test_config_display_never_warns_about_a_stored_unusable_state_dir(
    tmp_path: pathlib.Path, config: str, command: tuple[str, ...]
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "project"
    root.mkdir()
    assert _run(root, home, "git", "init", "-q").returncode == 0
    assert _run(root, home, _PIVOT, "init").returncode == 0
    if config == _CASES["file"]:
        (root / "afile").write_text("not a directory\n")
    (root / ".pivot" / "config.yaml").write_text(config)

    result = _run(root, home, _PIVOT, *command)

    assert "Pivot uses the default .pivot." not in result.stderr, result.stderr
