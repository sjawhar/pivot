"""Tests from the review of sjawhar/pivot#540 (round 8, 436686ac).

Regression locks; each failed at 31f438d2, where the defect was found, and passes from 436686ac:

- `track`'s refusal of a file a deleted stage left ends with the command that tracks it after the
  lock is removed. For a path with a space that command must stay one path when a shell splits
  it, and running it as printed must track that file.
- Reading another stage's listing for the member skip says nothing about state directories: the
  legacy-directory notice (`core.state_dir` set, cache objects left in `.pivot`) is the command's
  own business, not raised by a stage the command does not run.

A test that was red at 436686ac (it passed on main cea092a6 and failed there):

- A lock file under the stages directory whose name is not UTF-8 (git keeps a committed name's
  raw bytes) belongs to no stage the command runs; commands that never run it must not fail.

`_pivot` keeps `PYTHONPATH`, so a mutant run that points it at a mutated `src` reaches the CLI.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shlex
import subprocess
import sys

import pytest

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")

_ROOT_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def alive() -> str:
    return "alive"


with Pipeline("{name}") as pipeline:
    alive()
"""

_SPACED_MEMBER_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, directory, stage


@stage
def tree() -> Annotated[dict[str, str], directory]:
    return {{"my file.txt": "1", "my file.txt.pvt": {sidecar!r}}}


with Pipeline("b") as pipeline:
    tree()
"""

_MEMBER_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, directory, stage


@stage
def tree() -> Annotated[dict[str, str], directory]:
    return {
        "value.txt": "1",
        "fake.pvt": "path: value.txt\\nhash: abcdef0000000001\\nsize: 1\\n",
    }


with Pipeline("b") as pipeline:
    tree()
"""

_LEGACY_NOTICE = "holds stage locks or cache objects"


def _pivot(cwd: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    return subprocess.run(
        [_PIVOT, *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=300
    )


def _ok(proc: subprocess.CompletedProcess[str]) -> subprocess.CompletedProcess[str]:
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc


def _tracked(root: pathlib.Path) -> list[str]:
    status = _ok(_pivot(root, "status", "--tracked-only", "--json"))
    return sorted(info["path"] for info in json.loads(status.stdout)["tracked_files"])


def _init(root: pathlib.Path) -> None:
    root.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    _ok(_pivot(root, "init"))
    (root / "pipeline.py").write_text(_ROOT_PIPELINE.format(name="a"))


def test_the_remedy_for_a_path_with_a_space_is_one_path_and_tracks_it(
    tmp_path: pathlib.Path,
) -> None:
    """b's stage wrote `my file.txt` and a `my file.txt.pvt` with the bytes `pivot track` writes
    for it, then the stage was deleted; its lock stays. `track` refuses the file and prints the
    command that tracks it once the lock is gone. A shell splits that command into
    `pivot track --force` and one path, and running it (after the lock is removed) tracks the
    file: `status` lists it."""
    scratch = tmp_path / "scratch"
    _init(scratch)
    (scratch / "my file.txt").write_text("1")
    _ok(_pivot(scratch, "track", "my file.txt"))
    sidecar = (scratch / "my file.txt.pvt").read_text()
    root = tmp_path / "project"
    _init(root)
    (root / "other").mkdir()
    (root / "other" / "pipeline.py").write_text(_SPACED_MEMBER_PIPELINE.format(sidecar=sidecar))
    for cwd in (root / "other", root):
        _ok(_pivot(cwd, "repro"))
    (root / "other" / "pipeline.py").write_text(_ROOT_PIPELINE.format(name="b"))
    _ok(_pivot(root / "other", "repro"))
    path = "data/b/tree/my file.txt"

    refused = _pivot(root, "track", path)

    out = " ".join((refused.stdout + refused.stderr).split())
    assert refused.returncode != 0, out
    remedy = re.search(r"`(pivot track --force [^`]*)`", out)
    assert remedy is not None, out
    argv = shlex.split(remedy.group(1))
    assert argv == ["pivot", "track", "--force", path], argv
    (root / ".pivot" / "stages" / "b" / "tree.lock").unlink()
    _ok(_pivot(root, *argv[1:]))
    assert _tracked(root) == [path]


def test_another_stages_member_read_raises_no_legacy_state_dir_notice(
    tmp_path: pathlib.Path,
) -> None:
    """b ran with its lock, listing and cache objects in `.pivot`; then `core.state_dir` moved to
    `s1`, where `a` and one tracked file live. From the root, `status --tracked-only` skips b's
    member by b's lock, reading b's listing, whose cache object lies only in `.pivot`; it says
    nothing about state directories for that read, and lists only the tracked file."""
    root = tmp_path / "project"
    _init(root)
    (root / "other").mkdir()
    (root / "other" / "pipeline.py").write_text(_MEMBER_PIPELINE)
    _ok(_pivot(root / "other", "repro"))
    (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: s1\n")
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "kept.txt").write_text("kept")
    _ok(_pivot(root, "track", "data/raw/kept.txt"))
    _ok(_pivot(root, "repro"))

    status = _pivot(root, "status", "--tracked-only")

    out = " ".join((status.stdout + status.stderr).split())
    assert status.returncode == 0, out
    assert _LEGACY_NOTICE not in out, out
    assert _tracked(root) == ["data/raw/kept.txt"]


_TREE_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, directory, stage


@stage
def tree() -> Annotated[dict[str, str], directory]:
    return {"value.txt": "1"}


with Pipeline("b") as pipeline:
    tree()
"""


@pytest.mark.skipif(sys.platform == "darwin", reason="APFS refuses a file name that is not UTF-8")
def test_commands_run_beside_a_lock_whose_file_name_is_not_utf8(tmp_path: pathlib.Path) -> None:
    """Pipeline `a` at the root with one tracked file `data/raw/kept.txt`, since deleted. A lock
    named `<0xff>.lock` (b/tree's, recording `data/raw` as its directory output, so it lies over
    the tracked file's `.pvt`) sits in `.pivot/stages`. `status`, `checkout --only-missing`,
    `repro` and `verify` exit 0, as on main, and the tracked file is restored."""
    root = tmp_path / "project"
    _init(root)
    (root / "other").mkdir()
    (root / "other" / "pipeline.py").write_text(_TREE_PIPELINE)
    _ok(_pivot(root / "other", "repro"))
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "kept.txt").write_text("kept")
    _ok(_pivot(root, "track", "data/raw/kept.txt"))
    _ok(_pivot(root, "repro"))
    lock_text = (root / ".pivot" / "stages" / "b" / "tree.lock").read_text()
    assert "path: data/b/tree" in lock_text, lock_text
    stages = os.fsencode(root / ".pivot" / "stages")
    with open(os.path.join(stages, b"\xff.lock"), "w") as f:
        f.write(lock_text.replace("path: data/b/tree", "path: data/raw"))
    (raw / "kept.txt").unlink()

    results = {
        " ".join(args): _pivot(root, *args)
        for args in (("status",), ("checkout", "--only-missing"), ("repro",), ("verify",))
    }

    for command, proc in results.items():
        assert proc.returncode == 0, f"{command}: {proc.stdout}{proc.stderr}"
    assert (raw / "kept.txt").read_text() == "kept"
