"""R8: exit codes under an unusable local config, against 0.5.2, for cases the head's table leaves out.

Differential rule (ledger, 12:12Z; round 13 ruling, 10:20Z): when `.pivot/config.yaml` cannot be
read or its `core.state_dir` cannot be used, every command 0.5.2 ran keeps 0.5.2's exit code; every
command 0.5.2 failed still fails, in one `Error:` line and without a traceback.

`_052` holds pivot 0.5.2's exit codes, captured on pivot-slices with
`drv/r8_capture.py` (`uv run --no-project --isolated --with pivot==0.5.2 pivot`), only for the
(case, command) pairs this file runs. Each case copies one committed, reproduced project.
"""

from __future__ import annotations

import os
import pathlib
import shutil
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

_CONFIGS = {"empty": "core:\n  state_dir: ''\n"}

_COMMANDS: dict[str, tuple[str, ...]] = {
    "checkout": ("checkout",),
    "commit": ("commit",),
    "fingerprint reset": ("fingerprint", "reset"),
    "get": ("get", "--rev", "HEAD", "data/raw/input.txt"),
    "history": ("history",),
    "repro": ("repro",),
    "run": ("run", "test/process"),
    "status": ("status",),
    "verify": ("verify",),
    "list": ("list",),
}

# pivot 0.5.2, captured with drv/r8_capture.py on pivot-slices (r8cap_052.out).
_052: dict[str, dict[str, int]] = {
    "empty": {
        "checkout": 0,
        "commit": 0,
        "fingerprint reset": 0,
        "get": 0,
        "history": 0,
        "repro": 0,
        "run": 0,
        "status": 0,
        "verify": 0,
        "list": 0,
    },
}


def _run(cwd: pathlib.Path, home: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "HOME": str(home)}
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False, env=env)


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    directory = tmp_path_factory.mktemp("base")
    home = directory / "home"
    home.mkdir()
    root = directory / "project"
    root.mkdir()
    for git_args in (
        ("init", "-q"),
        ("config", "user.email", "t@example.com"),
        ("config", "user.name", "t"),
    ):
        assert _run(root, home, "git", *git_args).returncode == 0
    assert _run(root, home, _PIVOT, "init").returncode == 0
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("input\n")
    (root / "pipeline.py").write_text(_PIPELINE)
    assert _run(root, home, _PIVOT, "repro").returncode == 0
    assert _run(root, home, "git", "add", "-A").returncode == 0
    assert _run(root, home, "git", "commit", "-qm", "base").returncode == 0
    return directory


@pytest.mark.parametrize("case", list(_CONFIGS))
@pytest.mark.parametrize("command", list(_COMMANDS))
def test_exit_code_matches_052(
    base: pathlib.Path, tmp_path: pathlib.Path, case: str, command: str
) -> None:
    """Same exit code as 0.5.2; a failure is one `Error:` line and never a traceback."""
    copy = tmp_path / "copy"
    shutil.copytree(base, copy, symlinks=True)
    root = copy / "project"
    (root / ".pivot" / "config.yaml").write_text(_CONFIGS[case])

    result = _run(root, copy / "home", _PIVOT, *_COMMANDS[command])

    output = result.stdout + result.stderr
    assert "Traceback" not in output, output
    assert result.returncode == _052[case][command], output
    if result.returncode != 0:
        errors = [line for line in result.stderr.splitlines() if line.startswith("Error")]
        assert len(errors) == 1, result.stderr
