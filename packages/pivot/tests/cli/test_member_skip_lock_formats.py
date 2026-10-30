"""Other stages' locks in any format, and lock entries no `stat` can follow.

- Pivot 0.5.21 (lock schema 2, every lock before 0.6.0, with each directory listed inline) runs
  pipeline `b`, whose directory output holds a member. From the root, where only `a` is loaded,
  this pivot skips that member by the inline listing (kills m20, an older pivot's lock recording
  no directory output).
- A `*.lock` entry under the stages directory that is a symlink no `stat` can follow (here one
  that loops) is no lock, as `Path.is_file` reads it: every command that lists other stages'
  locks, and plain `pivot status` (its orphaned-lock count) with no `.pvt`, still exits 0.

The tests are the review's of sjawhar/pivot#540 at ca47c682, unchanged but for line wrapping.
The released pivot runs with `uv tool run` (``PIVOT_TEST_UV``, as
tests/cli/test_mixed_version_teams.py does), without ``PYTHONPATH``; this pivot keeps
``PYTHONPATH``, so a mutant run reaches it.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

from pivot.storage import cache

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")

_ROOT_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def alive() -> str:
    return "alive"


with Pipeline("a") as pipeline:
    alive()
"""

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


def _released(version: str) -> list[str]:
    uv = os.environ.get("PIVOT_TEST_UV") or shutil.which("uv")
    assert uv is not None, "this test runs a released pivot with `uv tool run`"
    return [uv, "tool", "run", "--quiet", "--from", f"pivot=={version}", "pivot"]


def _run(
    cwd: pathlib.Path, *command: str, keep_pythonpath: bool
) -> subprocess.CompletedProcess[str]:
    drop = {"VIRTUAL_ENV"} if keep_pythonpath else {"VIRTUAL_ENV", "PYTHONPATH"}
    env = {k: v for k, v in os.environ.items() if k not in drop}
    return subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=600)


def _ok(proc: subprocess.CompletedProcess[str]) -> None:
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_a_member_of_another_pipeline_whose_lock_an_older_pivot_wrote_is_skipped(
    tmp_path: pathlib.Path,
) -> None:
    """Pivot 0.5.21 runs `b` (in `other/`) and `a` (the root). b's `fake.pvt` names `value.txt`
    at the hash of "2", which the cache holds; b/tree's schema-2 lock lists it inline. From the
    root this pivot's `status` lists no tracked file and the target-less `checkout --force`
    leaves `value.txt` "1"."""
    old = _released("0.5.21")
    two = tmp_path / "two.txt"
    two.write_text("2")
    root = tmp_path / "project"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    _ok(_run(root, *old, "init", keep_pythonpath=False))
    (root / "pipeline.py").write_text(_ROOT_PIPELINE)
    (root / "other").mkdir()
    (root / "other" / "pipeline.py").write_text(
        _CACHED_MEMBER_PIPELINE.replace("HASH_OF_2", cache.hash_file(two)[0])
    )
    for cwd in (root / "other", root):
        _ok(_run(cwd, *old, "repro", keep_pythonpath=False))
    lock = root / ".pivot" / "stages" / "b" / "tree.lock"
    assert lock.read_text().startswith("schema_version: 2\n"), lock.read_text()

    status = _run(root, _PIVOT, "status", "--tracked-only", "--json", keep_pythonpath=True)
    forced = _run(root, _PIVOT, "checkout", "--force", keep_pythonpath=True)

    _ok(status)
    assert json.loads(status.stdout)["tracked_files"] == [], status.stdout
    _ok(forced)
    value = root / "data" / "b" / "tree" / "value.txt"
    assert value.read_text() == "1", forced.stdout + forced.stderr


@pytest.mark.parametrize("tracked", [True, False], ids=["tracked-file", "no-pvt"])
def test_commands_run_beside_a_lock_symlink_that_loops(
    tmp_path: pathlib.Path, tracked: bool
) -> None:
    """Pipeline `a` at the root, run once, with one tracked file in data/raw or none, and
    `.pivot/stages/x.lock`, a symlink to itself. `status`, `checkout --only-missing`, `repro`
    and `verify` exit 0, as on main."""
    root = tmp_path / "project"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    _ok(_run(root, _PIVOT, "init", keep_pythonpath=True))
    (root / "pipeline.py").write_text(_ROOT_PIPELINE)
    if tracked:
        raw = root / "data" / "raw"
        raw.mkdir(parents=True)
        (raw / "kept.txt").write_text("kept")
        _ok(_run(root, _PIVOT, "track", "data/raw/kept.txt", keep_pythonpath=True))
    _ok(_run(root, _PIVOT, "repro", keep_pythonpath=True))
    (root / ".pivot" / "stages" / "x.lock").symlink_to("x.lock")

    results = {
        " ".join(args): _run(root, _PIVOT, *args, keep_pythonpath=True)
        for args in (("status",), ("checkout", "--only-missing"), ("repro",), ("verify",))
    }

    for command, proc in results.items():
        assert proc.returncode == 0, f"{command}: {proc.stdout}{proc.stderr}"
