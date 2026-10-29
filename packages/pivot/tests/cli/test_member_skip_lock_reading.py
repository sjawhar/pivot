"""PR540 reviewer, round 5 (sjawhar/pivot#540 at 04baeda3): tests for the index of other
stages' locks.

- Red (passes on main 435707af, fails at the head): a command now reads the lock of every stage
  it does not run, once the project holds a `.pvt` outside its own outputs (any tracked file). A
  lock it cannot read (another account's 0600 lock in a shared state directory, a permission
  mistake) must not fail a command that never runs that stage: main never opened it.
- Red (passes on main, fails at the head): `track`'s refusal of a file a deleted stage left must
  hold when the path reaches it through a symlinked directory.
- Red (passes on main, fails at the head): the lock of a stage the command does not run, in the
  format pivot 0.5.21 wrote (schema 2, every lock before 0.6.0), is read quietly: no warning about
  a stage the command does not run, and no listing written into the cache by `status`.
- Regression lock (passes at the head, fails on main and under mutant m15, an index that reads
  only the configured state directory): after `core.state_dir` moves, a pipeline not run since
  keeps its locks in `.pivot`, the 0.5.2 layout's other directory; its member is still skipped.

`_pivot` keeps `PYTHONPATH`, so a mutant run that points it at a mutated `src` reaches the CLI.
"""

from __future__ import annotations

import json
import os
import pathlib
import stat
import subprocess
import sys

import pytest

from pivot.storage import cache

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")

_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, directory, stage


@stage
def {stage}() -> Annotated[dict[str, str], directory]:
    return {{"value.txt": "1", "fake.pvt": "path: value.txt\\nhash: abcdef0000000001\\nsize: 1\\n"}}


with Pipeline("{name}") as pipeline:
    {stage}()
"""


def _pivot(cwd: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    return subprocess.run(
        [_PIVOT, *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=300
    )


def _ok(proc: subprocess.CompletedProcess[str]) -> None:
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a mode-0 file")
def test_commands_of_one_pipeline_run_beside_another_pipelines_unreadable_lock(
    tmp_path: pathlib.Path,
) -> None:
    """Pipeline `a` at the root, with one tracked file in data/raw; pipeline `b` in `other/`,
    run once, whose lock the user running `a` cannot read. `status`, `checkout --only-missing`,
    `repro` and `verify` run from the root exit 0, as on main, and restore the tracked file."""
    root = tmp_path / "project"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    _ok(_pivot(root, "init"))
    (root / "pipeline.py").write_text(_PIPELINE.format(name="a", stage="own"))
    (root / "other").mkdir()
    (root / "other" / "pipeline.py").write_text(_PIPELINE.format(name="b", stage="tree"))
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "kept.txt").write_text("kept")
    _ok(_pivot(root, "track", "data/raw/kept.txt"))
    for cwd in (root / "other", root):
        _ok(_pivot(cwd, "repro"))
    (raw / "kept.txt").unlink()
    foreign = root / ".pivot" / "stages" / "b" / "tree.lock"
    mode = stat.S_IMODE(foreign.stat().st_mode)
    foreign.chmod(0)
    try:
        results = {
            " ".join(args): _pivot(root, *args)
            for args in (("status",), ("checkout", "--only-missing"), ("repro",), ("verify",))
        }
    finally:
        foreign.chmod(mode)

    for command, proc in results.items():
        assert proc.returncode == 0, f"{command}: {proc.stdout}{proc.stderr}"
    assert (raw / "kept.txt").read_text() == "kept"


_CACHED_MEMBER_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, directory, stage


@stage
def tree() -> Annotated[dict[str, str], directory]:
    return {
        "value.txt": "1",
        "two.txt": "2",
        "fake.pvt": "path: value.txt\\nhash: HASH_OF_2\\nsize: 1\\n",
    }


with Pipeline("b") as pipeline:
    tree()
"""

_ROOT_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def alive() -> str:
    return "alive"


with Pipeline("a") as pipeline:
    alive()
"""


def test_a_member_whose_lock_lies_in_the_other_state_directory_is_skipped(
    tmp_path: pathlib.Path,
) -> None:
    """Pipelines `a` (root) and `b` (`other/`) run with their locks in `.pivot`; then
    `core.state_dir` moves to `s1`, and neither runs again. b's `fake.pvt` names `value.txt` at
    the hash of "2", which the cache holds; b/tree's lock, in `.pivot`, lists it. From the root,
    `status` lists no tracked file and the target-less `checkout --force` leaves `value.txt` "1"."""
    two = tmp_path / "two.txt"
    two.write_text("2")
    root = tmp_path / "project"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    _ok(_pivot(root, "init"))
    (root / "pipeline.py").write_text(_ROOT_PIPELINE)
    (root / "other").mkdir()
    (root / "other" / "pipeline.py").write_text(
        _CACHED_MEMBER_PIPELINE.replace("HASH_OF_2", cache.hash_file(two)[0])
    )
    for cwd in (root / "other", root):
        _ok(_pivot(cwd, "repro"))
    assert (root / ".pivot" / "stages" / "b" / "tree.lock").is_file()
    (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: s1\n")

    status = _pivot(root, "status", "--tracked-only", "--json")
    forced = _pivot(root, "checkout", "--force")

    _ok(status)
    assert json.loads(status.stdout)["tracked_files"] == [], status.stdout
    _ok(forced)
    value = root / "data" / "b" / "tree" / "value.txt"
    assert value.read_text() == "1", forced.stdout + forced.stderr


_SIDECAR_MEMBER_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, directory, stage


@stage
def tree() -> Annotated[dict[str, str], directory]:
    return {{"value.txt": "1", "value.txt.pvt": {sidecar!r}}}


with Pipeline("b") as pipeline:
    tree()
"""


def test_track_through_a_symlink_of_a_file_a_deleted_stage_left_is_tracked_or_refused(
    tmp_path: pathlib.Path,
) -> None:
    """As `test_track_of_a_file_a_deleted_stage_left_is_tracked_or_refused`, with the path
    spelled through `alias`, a symlink to `data/b/tree`. `track` must refuse, or the file must
    then be tracked: listed by `status`, and restored by `checkout --only-missing` once deleted."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=scratch, check=True)
    _ok(_pivot(scratch, "init"))
    (scratch / "value.txt").write_text("1")
    _ok(_pivot(scratch, "track", "value.txt"))
    sidecar = (scratch / "value.txt.pvt").read_text()

    root = tmp_path / "project"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    _ok(_pivot(root, "init"))
    (root / "pipeline.py").write_text(_ROOT_PIPELINE)
    (root / "other").mkdir()
    (root / "other" / "pipeline.py").write_text(_SIDECAR_MEMBER_PIPELINE.format(sidecar=sidecar))
    for cwd in (root / "other", root):
        _ok(_pivot(cwd, "repro"))
    tree = root / "data" / "b" / "tree"
    assert (tree / "value.txt.pvt").read_text() == sidecar
    (root / "other" / "pipeline.py").write_text(_ROOT_PIPELINE.replace('"a"', '"b"'))
    _ok(_pivot(root / "other", "repro"))
    assert (root / ".pivot" / "stages" / "b" / "tree.lock").is_file()
    (root / "alias").symlink_to("data/b/tree")

    tracked = _pivot(root, "track", "alias/value.txt")

    if tracked.returncode != 0:
        return  # refused: the user is told, and nothing claims to track the file
    status = _pivot(root, "status", "--tracked-only", "--json")
    _ok(status)
    listed = [info["path"] for info in json.loads(status.stdout)["tracked_files"]]
    assert listed == ["data/b/tree/value.txt"], tracked.stdout + tracked.stderr + status.stdout
    (tree / "value.txt").unlink()
    _ok(_pivot(root, "checkout", "--only-missing"))
    assert (tree / "value.txt").read_text() == "1"


# The lock pivot 0.5.21 wrote for a stage `b/tree` returning {"value.txt": "1", "two.txt": "2"}
_SCHEMA_2_LOCK = """\
schema_version: 2
code_manifest:
  out:b/tree:loader:pivot.loaders.Text:empty: f346543fc28fb29c
  out:b/tree:loader:pivot.loaders.Text:load: 69b78870ffe7932d
  out:b/tree:loader:pivot.loaders.Text:save: 952962a56fe6457b
  out:b/tree:tag: directory
  self:tree: 2f75655751a070b0
params: {}
deps: []
outs:
- key: null
  hash: c50550725a60b14d
  path: data/b/tree
  manifest:
  - relpath: two.txt
    hash: 6021b5621680598b
    size: 1
    isexec: false
  - relpath: value.txt
    hash: b7b41276360564d4
    size: 1
    isexec: false
"""


def test_status_reads_an_older_pivots_lock_of_a_stage_it_does_not_run_quietly(
    tmp_path: pathlib.Path,
) -> None:
    """Pipeline `a` at the root, with one tracked file in data/raw. Stage `b/tree` is gone; its
    lock, as pivot 0.5.21 wrote it, stays (pivot deletes no lock). `pivot --quiet status
    --tracked-only` exits 0 with nothing on stderr, as on main, and writes no listing object."""
    root = tmp_path / "project"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    _ok(_pivot(root, "init"))
    (root / "pipeline.py").write_text(_ROOT_PIPELINE)
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "kept.txt").write_text("kept")
    _ok(_pivot(root, "track", "data/raw/kept.txt"))
    _ok(_pivot(root, "repro"))
    lock = root / ".pivot" / "stages" / "b" / "tree.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text(_SCHEMA_2_LOCK)
    listing_object = root / ".pivot" / "cache" / "files" / "c5" / "0550725a60b14d"

    status = _pivot(root, "--quiet", "status", "--tracked-only")

    _ok(status)
    assert status.stderr == "", status.stderr
    assert not listing_object.exists()
