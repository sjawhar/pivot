"""`config set core.state_dir` refuses a directory that is not writable and searchable."""

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


def _run(cwd: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
@pytest.mark.parametrize(
    ("mode", "value"),
    [
        # Read-only parent.
        (0o555, "dir/state"),
        # Writable but not searchable (no x bit): nothing can be created or opened inside.
        (0o600, "dir"),
        (0o600, "dir/state"),
    ],
)
def test_config_set_refuses_a_directory_pivot_cannot_use(
    tmp_path: pathlib.Path, mode: int, value: str
) -> None:
    """A target or nearest parent pivot cannot create entries in is refused before writing."""
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permission bits")
    root = tmp_path / "project"
    root.mkdir()
    assert _run(root, "git", "init", "-q").returncode == 0
    assert _run(root, _PIVOT, "init").returncode == 0
    assert _run(root, _PIVOT, "config", "set", "core.max_workers", "2").returncode == 0
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("input\n")
    (root / "pipeline.py").write_text(_PIPELINE)
    config_path = root / ".pivot" / "config.yaml"
    before = config_path.read_bytes()
    restricted = root / "dir"
    restricted.mkdir()
    restricted.chmod(mode)
    try:
        result = _run(root, _PIVOT, "config", "set", "core.state_dir", value)
        status = _run(root, _PIVOT, "status")
    finally:
        restricted.chmod(0o755)

    output = result.stdout + result.stderr
    assert config_path.read_bytes() == before, config_path.read_text()
    assert status.returncode == 0, status.stdout + status.stderr
    assert result.returncode == 1, output
    assert "Traceback" not in output, output
    lines = output.strip().splitlines()
    assert len(lines) == 1 and lines[0].startswith("Error: core.state_dir"), output
