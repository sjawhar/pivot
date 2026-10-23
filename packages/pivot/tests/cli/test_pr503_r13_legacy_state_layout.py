"""A project built by Pivot 0.5.2 remains readable after core.state_dir is set."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

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
    return subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "HOME": str(home)},
    )


def test_uses_legacy_locks_and_cache_after_state_dir_is_configured(tmp_path: pathlib.Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "project"
    root.mkdir()
    for args in (
        ("init", "-q"),
        ("config", "user.email", "t@example.com"),
        ("config", "user.name", "t"),
    ):
        assert _run(root, home, "git", *args).returncode == 0
    assert _run(root, home, _PIVOT, "init").returncode == 0
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("input\n")
    (root / "pipeline.py").write_text(_PIPELINE)
    assert _run(root, home, _PIVOT, "repro").returncode == 0
    assert _run(root, home, "git", "add", "-A").returncode == 0
    assert _run(root, home, "git", "commit", "-qm", "base").returncode == 0

    (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: altstate\n")

    verify = _run(root, home, _PIVOT, "verify")
    assert verify.returncode == 0, verify.stdout + verify.stderr
    assert "Pivot 0.5.2 stored locks and cache" in verify.stderr

    output = root / "data" / "test" / "process.txt"
    output.unlink()
    checkout = _run(root, home, _PIVOT, "checkout", "test/process")
    assert checkout.returncode == 0, checkout.stdout + checkout.stderr
    assert output.read_text() == "INPUT\n"
