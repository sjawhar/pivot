"""`pivot track <path>` never removes or overwrites a file at ``<path>.pvt.listing`` that it did
not write for that path: it refuses, exit 1, naming the file, as it does for a ``.pvt`` that is
another tracked path. A listing pivot wrote for that path's tracked directory (it hashes to the
recorded tree) is replaced or removed as before. From the PR #523 round-2 recheck (slice
``cip``).
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")
_USER_TEXT = "notes a person keeps beside e; pivot does not track this file\n"


def _env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PIVOT_CACHE_DIR")}


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_track_leaves_an_untracked_file_at_the_listing_path_as_it_was(
    tmp_path: pathlib.Path, kind: str
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    init = subprocess.run(
        [_PIVOT, "init"], cwd=tmp_path, env=_env(), capture_output=True, text=True, check=False
    )
    assert init.returncode == 0, init.stdout + init.stderr
    (tmp_path / "e.pvt.listing").write_text(_USER_TEXT)
    if kind == "file":
        (tmp_path / "e").write_text("e\n")
    else:
        (tmp_path / "e").mkdir()
        (tmp_path / "e" / "a.txt").write_text("a\n")

    tracked = subprocess.run(
        [_PIVOT, "track", "e"],
        cwd=tmp_path,
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    out = tracked.stdout + tracked.stderr
    user_file = tmp_path / "e.pvt.listing"
    assert user_file.is_file(), f"rc={tracked.returncode}; e.pvt.listing was removed\n{out}"
    assert user_file.read_text() == _USER_TEXT, (
        f"rc={tracked.returncode}; e.pvt.listing was overwritten with:\n"
        + user_file.read_text()[:200]
        + f"\n{out}"
    )
    assert tracked.returncode == 1, out
    assert "e.pvt.listing" in out and "move it aside" in out, out
    assert not (tmp_path / "e.pvt").exists(), out


def test_track_replaces_and_removes_the_listing_it_wrote_for_the_path(tmp_path: pathlib.Path):
    """Re-tracking a changed directory rewrites its listing; tracking a file in its place
    removes the listing pivot wrote for the directory."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    env = _env()

    def track(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [_PIVOT, "track", *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    assert subprocess.run([_PIVOT, "init"], cwd=tmp_path, env=env, check=False).returncode == 0
    (tmp_path / "e").mkdir()
    (tmp_path / "e" / "a.txt").write_text("a\n")
    first = track("e")
    assert first.returncode == 0, first.stdout + first.stderr
    listing_file = tmp_path / "e.pvt.listing"
    before = listing_file.read_bytes()
    (tmp_path / "e" / "b.txt").write_text("b\n")

    again = track("--force", "e")

    assert again.returncode == 0, again.stdout + again.stderr
    assert listing_file.read_bytes() != before
    (tmp_path / "e" / "a.txt").unlink()
    (tmp_path / "e" / "b.txt").unlink()
    (tmp_path / "e").rmdir()
    (tmp_path / "e").write_text("now a file\n")
    as_file = track("--force", "e")
    assert as_file.returncode == 0, as_file.stdout + as_file.stderr
    assert not listing_file.exists()
