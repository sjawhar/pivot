"""A member of a pipeline the command did not load is found by the directory output its stage's
lock records (``lock.recorded_directories``), wherever that output was placed.

- A ``PathOnly`` output that returns a directory lies at ``data/<pipeline>/<stage>.dat/``; its
  member is skipped from another pipeline too.
- A member below the top of another pipeline's output, or in its keyed directory output, is
  found by the output that holds it, not by its own directory.
- A lock outlives its stage, and its members stay skipped, so ``pivot track`` refuses a path
  whose sidecar the lock of a stage that no longer exists records.
- ``status --check-imports`` and ``pivot update`` read no import from a stage's own member.
- A lock file is decoded once per state-directory record, and again when it changes.

The first three tests are the review's of sjawhar/pivot#540 at 4f145295; the track test requires
the refusal where the review's accepted either outcome. ``_pivot`` keeps ``PYTHONPATH``, so a
mutant run that points it at a mutated ``src`` reaches the CLI.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import click.testing
import pytest

from pivot import cli, project
from pivot import state_dir as state_dir_mod
from pivot.storage import cache, lock, member_skip, track

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")

_ROOT_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def alive() -> str:
    return "alive"


with Pipeline("{name}") as pipeline:
    alive()
"""

# b/build returns a directory: Pivot places it at data/b/build.dat/
_PATH_ONLY_MEMBER_PIPELINE = """\
import pathlib
import tempfile
from typing import Annotated

from pivot import Pipeline, loaders, stage


@stage
def build() -> Annotated[pathlib.Path, loaders.PathOnly()]:
    out = pathlib.Path(tempfile.mkdtemp()) / "build"
    out.mkdir(parents=True)
    (out / "value.txt").write_text("1")
    (out / "two.txt").write_text("2")
    (out / "fake.pvt").write_text("path: value.txt\\nhash: HASH_OF_2\\nsize: 1\\n")
    return out


with Pipeline("b") as pipeline:
    build()
"""

_SIDECAR_MEMBER_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, directory, stage


@stage
def tree() -> Annotated[dict[str, str], directory]:
    return {{"value.txt": "1", "value.txt.pvt": {sidecar!r}}}


with Pipeline("b") as pipeline:
    tree()
"""

_NESTED_MEMBER_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, directory, stage


@stage
def tree() -> Annotated[dict[str, str], directory]:
    return {
        "sub/deep/value.txt": "1",
        "sub/deep/two.txt": "2",
        "sub/deep/fake.pvt": "path: value.txt\\nhash: HASH_OF_2\\nsize: 1\\n",
    }


with Pipeline("b") as pipeline:
    tree()
"""

_KEYED_MEMBER_PIPELINE = """\
from typing import Annotated, TypedDict

from pivot import Pipeline, directory, stage


class Out(TypedDict):
    files: Annotated[dict[str, str], directory]
    note: str


@stage
def tree() -> Out:
    return Out(
        files={
            "value.txt": "1",
            "two.txt": "2",
            "fake.pvt": "path: value.txt\\nhash: HASH_OF_2\\nsize: 1\\n",
        },
        note="n",
    )


with Pipeline("b") as pipeline:
    tree()
"""


def _pivot(cwd: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    return subprocess.run(
        [_PIVOT, *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=300
    )


def _ok(proc: subprocess.CompletedProcess[str]) -> None:
    assert proc.returncode == 0, proc.stdout + proc.stderr


def _project(tmp_path: pathlib.Path, other_pipeline: str) -> pathlib.Path:
    """Pipeline `a` at the root and pipeline `b` in `other/`, each run once."""
    root = tmp_path / "project"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    _ok(_pivot(root, "init"))
    (root / "pipeline.py").write_text(_ROOT_PIPELINE.format(name="a"))
    (root / "other").mkdir()
    (root / "other" / "pipeline.py").write_text(other_pipeline)
    for cwd in (root / "other", root):
        _ok(_pivot(cwd, "repro"))
    return root


def _tracked_paths(proc: subprocess.CompletedProcess[str]) -> list[str]:
    _ok(proc)
    return sorted(info["path"] for info in json.loads(proc.stdout)["tracked_files"])


@pytest.mark.parametrize(
    ("pipeline", "member_dir"),
    [
        pytest.param(_NESTED_MEMBER_PIPELINE, "data/b/tree/sub/deep", id="nested"),
        pytest.param(_KEYED_MEMBER_PIPELINE, "data/b/tree/files", id="keyed"),
    ],
)
def test_a_member_below_the_stage_name_of_another_pipeline_is_skipped(
    tmp_path: pathlib.Path, pipeline: str, member_dir: str
) -> None:
    """b wrote `fake.pvt` two directories below its output, or in its keyed directory output
    `files`, naming `value.txt` at the hash of "2", which the cache holds. The path spells longer
    names than the stage's (`b/tree/sub/deep`, `b/tree/files`); the lookup still finds b/tree's
    lock, so from the root `status` lists no tracked file and the target-less `checkout --force`
    leaves `value.txt` "1"."""
    two = tmp_path / "two.txt"
    two.write_text("2")
    root = _project(tmp_path, pipeline.replace("HASH_OF_2", cache.hash_file(two)[0]))
    output = root / member_dir
    assert track.read_pvt_file(output / "fake.pvt") is not None

    status = _pivot(root, "status", "--tracked-only", "--json")
    forced = _pivot(root, "checkout", "--force")

    assert _tracked_paths(status) == [], status.stdout
    _ok(forced)
    assert (output / "value.txt").read_text() == "1", forced.stdout + forced.stderr


def test_a_path_only_directory_member_of_another_pipeline_is_skipped(
    tmp_path: pathlib.Path,
) -> None:
    """b's `PathOnly` output is a directory at `data/b/build.dat/`, and b wrote `fake.pvt` there,
    naming `value.txt` at the hash of "2", which the cache holds. From the root, where only `a`
    is loaded, it is b's member as it is from `other/`: `status` lists no tracked file and the
    target-less `checkout --force` leaves `value.txt` "1"."""
    two = tmp_path / "two.txt"
    two.write_text("2")
    root = _project(
        tmp_path, _PATH_ONLY_MEMBER_PIPELINE.replace("HASH_OF_2", cache.hash_file(two)[0])
    )
    (output,) = (root / "data" / "b").glob("build*")
    assert track.read_pvt_file(output / "fake.pvt") is not None

    status = _pivot(root, "status", "--tracked-only", "--json")
    forced = _pivot(root, "checkout", "--force")

    assert _tracked_paths(status) == [], status.stdout
    _ok(forced)
    assert (output / "value.txt").read_text() == "1", forced.stdout + forced.stderr


def test_track_refuses_a_file_a_deleted_stage_left(tmp_path: pathlib.Path) -> None:
    """b's stage wrote `value.txt` and a `value.txt.pvt` with the bytes `pivot track` writes for
    it, then the stage was deleted; its lock stays. Every command reads that `.pvt` as the dead
    stage's own file by its lock, so a sidecar `track` wrote there would be one none reads.
    `track` refuses the path, names the lock, says the stage no longer exists, and gives the
    remedy; it writes nothing. (The review's test accepted either a refusal or a real track;
    the refusal is the outcome this project chose.)"""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=scratch, check=True)
    _ok(_pivot(scratch, "init"))
    (scratch / "value.txt").write_text("1")
    _ok(_pivot(scratch, "track", "value.txt"))
    sidecar = (scratch / "value.txt.pvt").read_text()

    root = _project(tmp_path, _SIDECAR_MEMBER_PIPELINE.format(sidecar=sidecar))
    tree = root / "data" / "b" / "tree"
    assert (tree / "value.txt.pvt").read_text() == sidecar
    (root / "other" / "pipeline.py").write_text(_ROOT_PIPELINE.format(name="b"))
    _ok(_pivot(root / "other", "repro"))
    assert (root / ".pivot" / "stages" / "b" / "tree.lock").is_file()

    tracked = _pivot(root, "track", "data/b/tree/value.txt")

    out = " ".join((tracked.stdout + tracked.stderr).split())
    assert tracked.returncode != 0, out
    assert "Tracked:" not in out, out
    assert ".pivot/stages/b/tree.lock" in out, out
    assert "no longer exists" in out, out
    assert "remove that lock" in out, out
    assert (tree / "value.txt.pvt").read_text() == sidecar


# A `.pvt` shaped as an import (a whole `source:` block) that the stage writes in its output
_IMPORT_MEMBER_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, directory, stage

_IMPORT = (
    "path: imported.txt\\n"
    "hash: abcdef0000000001\\n"
    "size: 1\\n"
    "source:\\n"
    "  repo: /nonexistent/upstream\\n"
    "  rev: main\\n"
    "  rev_lock: 0123456789abcdef0123456789abcdef01234567\\n"
    "  stage: up/numbers\\n"
    "  path: up/numbers\\n"
    "  remote: s3://nonexistent-bucket/cache\\n"
)


@stage
def tree() -> Annotated[dict[str, str], directory]:
    return {"imported.txt": "1", "imported.txt.pvt": _IMPORT}


with Pipeline("b") as pipeline:
    tree()
"""


def test_update_and_check_imports_read_no_import_from_a_stages_own_member(
    tmp_path: pathlib.Path,
) -> None:
    """The stage writes `imported.txt.pvt`, which parses as an import of a repository that does
    not exist. It is the stage's own member, not an import: `status --check-imports` (pipeline
    `b` loaded, the registry decides) reports no import, and `pivot update` (no pipeline loaded,
    the lock index decides) finds none to update, so neither reaches the repository."""
    root = tmp_path / "project"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    _ok(_pivot(root, "init"))
    (root / "pipeline.py").write_text(_IMPORT_MEMBER_PIPELINE)
    _ok(_pivot(root, "repro"))
    member = root / "data" / "b" / "tree" / "imported.txt.pvt"
    member_data = track.read_pvt_file(member)
    assert member_data is not None and track.is_import(member_data)

    status = _pivot(root, "status", "--check-imports", "--json")
    update = _pivot(root, "update")

    _ok(status)
    assert json.loads(status.stdout)["imports"] == [], status.stdout
    _ok(update)
    assert "No imports found to update." in update.stdout, update.stdout + update.stderr


def test_a_lock_is_decoded_once_per_record_and_again_when_it_changes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """From the root, b's `fake.pvt` lies outside every output of the loaded pipeline `a`, so a
    command that discovers the `.pvt` files reads the directory outputs b's lock records. Each
    decode is counted through `lock.parse_committed`, the one decoder of a lock file.

    - In one state-directory record, every skip and `track`'s orphan check share what a lock
      file records: b/tree's lock is decoded once for three callers.
    - A lock file written since is decoded again in the same record, as a `--watch` run needs.
    - `verify` and `repro` (its pre-flight) each decode it once.
    """
    root = _project(tmp_path, _NESTED_MEMBER_PIPELINE.replace("HASH_OF_2", "abcdef0000000002"))
    monkeypatch.chdir(root)
    monkeypatch.setattr(project, "_project_root_cache", None)
    b_lock = root / ".pivot" / "stages" / "b" / "tree.lock"
    decoded = list[bool]()  # one entry per decode: whether it was b/tree's lock
    parse_committed = lock.parse_committed

    def counted(document: object) -> lock.CommittedLock | lock.UnsafeLockPath | None:
        decoded.append("data/b/tree" in json.dumps(document, default=str))
        return parse_committed(document)

    monkeypatch.setattr(lock, "parse_committed", counted)
    member = root / "data" / "b" / "tree" / "sub" / "deep" / "fake.pvt"
    member_data = track.read_pvt_file(member)
    assert member_data is not None
    state_dir, files_dir = root / ".pivot", root / ".pivot" / "cache" / "files"

    token = state_dir_mod.begin_invocation(report=False)
    try:
        first = member_skip.own_member_skip(root, {}, state_dir, files_dir)
        second = member_skip.own_member_skip(root, {}, state_dir, files_dir)
        claim = member_skip.orphaned_member(root, state_dir, files_dir, ())
        skipped = [first(member, member_data), second(member, member_data)]
        claimed = claim(member.with_suffix(""))
        once = decoded.count(True)
        b_lock.write_text(b_lock.read_text() + "\n")  # a run rewrote the lock
        third = member_skip.own_member_skip(root, {}, state_dir, files_dir)
        skipped.append(third(member, member_data))
    finally:
        state_dir_mod.end_invocation(token)

    assert skipped == [True, True, True]
    assert claimed is not None and claimed.stage_name == "b/tree", claimed
    assert once == 1, decoded
    assert decoded.count(True) == 2, decoded

    decoded.clear()
    runner = click.testing.CliRunner()
    verified = runner.invoke(cli.cli, ["verify"])
    verify_decodes = decoded.count(True)
    rerun = runner.invoke(cli.cli, ["repro"])

    assert verified.exit_code == 0, verified.output
    assert verify_decodes == 1, decoded
    assert rerun.exit_code == 0, rerun.output
    assert decoded.count(True) == 2, decoded
