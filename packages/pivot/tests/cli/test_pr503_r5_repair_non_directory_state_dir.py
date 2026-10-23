"""`config set` and `config unset` replace a hand-edited `core.state_dir` that names no usable
directory, and exit 0 as on 0.5.2."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")


def _run(cwd: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)


@pytest.mark.parametrize("action", ["set", "unset"])
@pytest.mark.parametrize("old", ["empty-string", "regular-file", "unreadable-directory"])
def test_config_set_and_unset_replace_an_unusable_hand_edited_state_dir(
    tmp_path: pathlib.Path, old: str, action: str
) -> None:
    """Like 0.5.2, the command succeeds with its confirmation and the config holds the new value."""
    if old == "unreadable-directory" and os.geteuid() == 0:
        pytest.skip("root ignores directory permission bits")
    root = tmp_path / "project"
    root.mkdir()
    assert _run(root, "git", "init", "-q").returncode == 0
    assert _run(root, _PIVOT, "init").returncode == 0
    config_path = root / ".pivot" / "config.yaml"
    locked = root / "locked"
    if old == "empty-string":
        config_path.write_text("core:\n  state_dir: ''\n")
    elif old == "regular-file":
        (root / "afile").write_text("not a directory\n")
        config_path.write_text("core:\n  state_dir: afile\n")
    else:
        (locked / "stages").mkdir(parents=True)
        locked.chmod(0o000)
        config_path.write_text("core:\n  state_dir: locked\n")

    try:
        if action == "set":
            result = _run(root, _PIVOT, "config", "set", "core.state_dir", "altstate")
        else:
            result = _run(root, _PIVOT, "config", "unset", "core.state_dir")
    finally:
        if locked.exists():
            locked.chmod(0o755)

    output = result.stdout + result.stderr
    assert "Traceback" not in output, output
    assert result.returncode == 0, output
    if action == "set":
        assert "state_dir: altstate" in config_path.read_text()
        assert output.strip().splitlines()[-1] == "Set core.state_dir = altstate in local config"
    else:
        assert "state_dir" not in config_path.read_text()
        assert output.strip().splitlines()[-1] == "Removed core.state_dir from local config"
