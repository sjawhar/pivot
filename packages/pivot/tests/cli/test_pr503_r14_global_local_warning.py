"""Round 14 B4: an unusable local state directory owns the one warning."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")
_CASES = {
    "empty": "''",
    "integer": "5",
    "list": "[a, b]",
    "mapping": "{x: 1}",
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


@pytest.mark.parametrize("value", _CASES.values(), ids=_CASES)
def test_unusable_local_state_dir_suppresses_the_global_state_dir_notice(
    tmp_path: pathlib.Path, value: str
) -> None:
    home = tmp_path / "home"
    global_config = home / ".config" / "pivot" / "config.yaml"
    global_config.parent.mkdir(parents=True)
    global_config.write_text("core:\n  state_dir: global-state\n")
    root = tmp_path / "project"
    root.mkdir()
    assert _run(root, home, "git", "init", "-q").returncode == 0
    assert _run(root, home, _PIVOT, "init").returncode == 0
    (root / ".pivot" / "config.yaml").write_text(f"core:\n  state_dir: {value}\n")
    (root / "pipeline.py").write_text("from pivot import Pipeline\n\npipeline = Pipeline('test')\n")

    listed = _run(root, home, _PIVOT, "list")

    assert listed.returncode == 0, listed.stdout + listed.stderr
    assert str(global_config) not in listed.stderr
    assert listed.stderr.count("Pivot uses the default .pivot.") == 1, listed.stderr
