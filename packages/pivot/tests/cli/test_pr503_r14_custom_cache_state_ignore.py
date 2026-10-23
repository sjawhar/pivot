"""A custom cache below core.state_dir retains its state ignore template."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

from pivot import state_dir as state_dir_mod

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")


def _run(cwd: pathlib.Path, home: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "HOME": str(home)},
    )


def test_track_ignores_a_custom_cache_inside_a_new_state_directory(tmp_path: pathlib.Path) -> None:
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
    (root / "data").mkdir()
    (root / "data" / "item.txt").write_text("item\n")
    assert _run(root, home, "git", "add", "-A").returncode == 0
    assert _run(root, home, "git", "commit", "-qm", "base").returncode == 0
    (root / ".pivot" / "config.yaml").write_text(
        "core:\n  state_dir: fresh\ncache:\n  dir: fresh/custom-cache\n"
    )

    tracked = _run(root, home, _PIVOT, "track", "data/item.txt")

    assert tracked.returncode == 0, tracked.stdout + tracked.stderr
    assert (root / "fresh" / ".gitignore").read_text() == state_dir_mod.STATE_GITIGNORE_CONTENT
    status = _run(root, home, "git", "status", "--porcelain", "-uall")
    assert status.returncode == 0, status.stderr
    assert not any(
        line.endswith("custom-cache") or "custom-cache/" in line
        for line in status.stdout.splitlines()
    )
