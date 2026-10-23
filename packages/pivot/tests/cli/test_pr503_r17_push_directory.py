from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest
import yaml

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")
_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def process(value: str) -> str:
    return value.upper()


with Pipeline("test") as pipeline:
    process(pipeline.input("input.txt", t=str))
"""


def _run(root: pathlib.Path, home: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "HOME": str(home)},
    )


@pytest.mark.parametrize("layout", ["configured", "legacy"])
def test_push_dry_run_ignores_directory_at_cache_object_path(
    tmp_path: pathlib.Path, layout: str
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "project"
    root.mkdir()
    assert _run(root, home, "git", "init", "-q").returncode == 0
    assert _run(root, home, _PIVOT, "init").returncode == 0
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("input\n")
    (root / "pipeline.py").write_text(_PIPELINE)
    if layout == "configured":
        assert (
            _run(root, home, _PIVOT, "config", "set", "core.state_dir", "altstate").returncode == 0
        )
    assert _run(root, home, _PIVOT, "repro").returncode == 0
    if layout == "legacy":
        assert (
            _run(root, home, _PIVOT, "config", "set", "core.state_dir", "altstate").returncode == 0
        )
    assert (
        _run(root, home, _PIVOT, "config", "set", "remotes.origin", "s3://r17-bucket").returncode
        == 0
    )

    state = root / "altstate" if layout == "configured" else root / ".pivot"
    lock = yaml.safe_load((state / "stages" / "test" / "process.lock").read_text())
    file_hash = lock["outs"][0]["hash"]
    cache_object = state / "cache" / "files" / file_hash[:2] / file_hash[2:]
    cache_object.unlink()
    cache_object.mkdir()

    result = _run(root, home, _PIVOT, "push", "--dry-run", "--remote", "origin")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Would push 0 file(s)" in result.stdout
