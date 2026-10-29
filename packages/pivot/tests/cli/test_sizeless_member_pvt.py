"""A directory ``.pvt`` whose inline members have no ``size``, which pivot 0.5.12 restored.

Pivot 0.5.12 (#522) made ``size`` optional for a tracked directory and restored such a ``.pvt``
in copy mode. This pivot records a directory by its listing, whose bytes include each member's
size, so a sizeless-member directory has no valid listing (``listing.from_inline`` requires an
``int`` size). It therefore refuses the ``.pvt`` rather than fabricate a size that a later
``verify`` would disagree with: ``pivot checkout`` names the file and exits non-zero, it does not
restore the directory in silence. The cache objects and the ``.pvt`` remain; ``pivot track
--force`` rewrites the ``.pvt`` in this pivot's format.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys

import pytest
import yaml

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")


def _env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}


def _run(cwd: pathlib.Path, *argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv), cwd=cwd, env=_env(), capture_output=True, text=True, timeout=300, check=False
    )


def _git(cwd: pathlib.Path, *args: str) -> None:
    proc = _run(cwd, "git", "-c", "user.email=t@t", "-c", "user.name=t", *args)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def _project_with_sizeless_pvt(root: pathlib.Path) -> None:
    root.mkdir(parents=True)
    _git(root, "init", "-q", "-b", "main")
    assert _run(root, _PIVOT, "init").returncode == 0
    (root / "d").mkdir()
    (root / "d" / "a.txt").write_text("a\n")
    (root / ".gitignore").write_text("/d\n")
    assert _run(root, _PIVOT, "track", "d").returncode == 0
    pvt = yaml.safe_load((root / "d.pvt").read_text())
    member_hash = (root / "d.pvt.listing").read_text().splitlines()[0].split(" ", 1)[0]
    # An older pivot's directory .pvt: its members listed inline, with no `size`, and no listing
    # file beside it (an older pivot writes none).
    (root / "d.pvt").write_text(
        yaml.safe_dump(
            {
                "path": "d",
                "hash": pvt["hash"],
                "num_files": 1,
                "manifest": [{"relpath": "a.txt", "hash": member_hash, "isexec": False}],
                "empty_dirs": [],
            },
            sort_keys=False,
        )
    )
    (root / "d.pvt.listing").unlink()
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "sizeless-member .pvt")
    shutil.rmtree(root / "d")


@pytest.mark.parametrize("mode", ["copy", "symlink", "hardlink"])
def test_checkout_of_a_sizeless_member_pvt_is_refused_not_restored_in_silence(
    tmp_path: pathlib.Path, mode: str
) -> None:
    root = tmp_path / mode
    _project_with_sizeless_pvt(root)

    proc = _run(root, _PIVOT, "checkout", "--checkout-mode", mode, "d")

    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    assert "d.pvt" in out, out
    # One line gives the cause and the remedy, so the message cannot decay to the generic form.
    assert "members have no size" in out, out
    assert "pivot track --force d" in out, out
    assert not (root / "d").exists(), out  # not restored in silence
    # The data is recoverable: the cache object and the .pvt are untouched.
    assert (root / "d.pvt").is_file()
