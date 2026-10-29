"""Which ``*.pvt`` files discovery leaves out as a stage's own members, and what it never does.

A user's sidecar at a member's path, with other bytes, stays a tracked file; a member below the
top of its output is still the stage's; a state database that cannot be opened fails no command
main ran; and a member of a pipeline the command did not load, a pipeline that fails to load or
a stage that no longer exists is decided by its lock, so a command run from another pipeline
rewrites nothing it names. The first three came from the review of sjawhar/pivot#540 at a67f4185
and are kept as the reviewer wrote them.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from conftest import isolated_pivot_dir
from pivot import cli
from pivot.storage import cache, track

if TYPE_CHECKING:
    from click.testing import CliRunner

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")

_SIDECAR_SHAPED_MEMBER_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, StageParams, directory, stage


class TreeParams(StageParams):
    n: int = 1


@stage
def tree(params: TreeParams) -> Annotated[dict[str, str], directory]:
    return {
        "value.txt": str(params.n),
        "fake.pvt": f"path: value.txt\\nhash: abcdef{params.n:010x}\\nsize: 1\\n",
    }


with Pipeline("b") as pipeline:
    tree(params=TreeParams())
"""

_NESTED_SIDECAR_SHAPED_MEMBER_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, directory, stage


@stage
def tree() -> Annotated[dict[str, str], directory]:
    return {
        "sub/value.txt": "1",
        "sub/fake.pvt": "path: value.txt\\nhash: abcdef0000000001\\nsize: 1\\n",
    }


with Pipeline("b") as pipeline:
    tree()
"""


def test_a_users_sidecar_at_the_path_of_a_stages_member_stays_a_tracked_file(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """The stage's lock lists `fake.pvt`. The user puts a sidecar of their own there (other
    bytes, naming a tracked file `fake` copied beside it). Its path is a member's, its content is
    not, so discovery keeps it: `status` lists the user's file, and the forced checkout refuses
    the output, naming the sidecar, and changes nothing."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_SIDECAR_SHAPED_MEMBER_PIPELINE)
        assert runner.invoke(cli.cli, ["repro"]).exit_code == 0
        raw = root / "data" / "raw" / "fake"
        raw.parent.mkdir(parents=True)
        raw.write_text("user bytes")
        assert runner.invoke(cli.cli, ["track", "data/raw/fake"]).exit_code == 0
        tree = root / "data" / "b" / "tree"
        stage_fake = (tree / "fake.pvt").read_text()
        user_fake = track.get_pvt_path(raw).read_text()
        assert user_fake != stage_fake
        shutil.copy2(raw, tree / "fake")
        (tree / "fake.pvt").unlink()  # the stage's file is a read-only cache link
        (tree / "fake.pvt").write_text(user_fake)

        status = runner.invoke(cli.cli, ["status", "--tracked-only", "--json"])
        result = runner.invoke(cli.cli, ["checkout", "--force"])

        assert status.exit_code == 0, status.output
        tracked = [info["path"] for info in json.loads(status.stdout)["tracked_files"]]
        assert sorted(tracked) == ["data/b/tree/fake", "data/raw/fake"], status.stdout
        assert result.exit_code != 0, result.output
        assert "Refusing to replace" in result.output, result.output
        assert "data/b/tree/fake.pvt" in result.output, result.output
        assert (tree / "fake").read_text() == "user bytes"
        assert (tree / "fake.pvt").read_text() == user_fake
        assert (tree / "value.txt").read_text() == "1"


@pytest.mark.parametrize("value_present", [False, True], ids=["value-deleted", "value-present"])
def test_targetless_forced_checkout_restores_an_output_whose_nested_member_is_sidecar_shaped(
    runner: CliRunner, tmp_path: pathlib.Path, value_present: bool
) -> None:
    """`sub/fake.pvt` parses as a sidecar naming `sub/value.txt`, with a hash no cache holds, and
    the stage's lock lists it: a member one directory below the output, not a tracked file. A
    checkout with no target restores `b/tree` whole and restores nothing `sub/fake.pvt` names."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_NESTED_SIDECAR_SHAPED_MEMBER_PIPELINE)
        assert runner.invoke(cli.cli, ["repro"]).exit_code == 0
        sub = root / "data" / "b" / "tree" / "sub"
        written = (sub / "fake.pvt").read_text()
        assert track.read_pvt_file(sub / "fake.pvt") is not None
        if not value_present:
            (sub / "value.txt").unlink()

        result = runner.invoke(cli.cli, ["checkout", "--force"])

        assert result.exit_code == 0, result.output
        assert "Restored 2 file(s)" in result.output, result.output
        assert "Missing" not in result.output, result.output
        assert (sub / "value.txt").read_text() == "1"
        assert (sub / "fake.pvt").read_text() == written


def test_a_nested_member_of_a_run_that_wrote_no_lock_is_the_stages(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """`repro --no-commit` writes `sub/fake.pvt` and records it in the stage's `nocommit:`
    record, but writes no lock. No lock can vouch for it then; the loaded stage's own claims do,
    for a member below the top of its output too, so `status` lists no tracked file."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_NESTED_SIDECAR_SHAPED_MEMBER_PIPELINE)
        built = runner.invoke(cli.cli, ["repro", "--no-commit"])
        assert built.exit_code == 0, built.output
        assert not (root / ".pivot" / "stages" / "b" / "tree.lock").exists()
        assert track.read_pvt_file(root / "data" / "b" / "tree" / "sub" / "fake.pvt") is not None

        status = runner.invoke(cli.cli, ["status", "--tracked-only", "--json"])

        assert status.exit_code == 0, status.output
        assert json.loads(status.stdout)["tracked_files"] == [], status.stdout


def _pivot(cwd: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    return subprocess.run(
        [_PIVOT, *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=300
    )


@pytest.mark.parametrize("damage", ["garbage", "unwritable"])
def test_checkout_and_track_run_with_a_state_database_they_cannot_open_beside_a_member(
    tmp_path: pathlib.Path, damage: str
) -> None:
    """Main's `checkout` and `track` never opened the state database to decide which files are
    tracked, so a damaged or unwritable `state.lmdb` left them working
    (`test_checkout_restores_with_a_damaged_state_database`). A project whose output holds the
    stage's own `fake.pvt` must keep that: `checkout --only-missing` restores a deleted tracked
    file and `track` tracks a new one, both exit 0."""
    root = tmp_path / "project"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    assert _pivot(root, "init").returncode == 0
    (root / "pipeline.py").write_text(_SIDECAR_SHAPED_MEMBER_PIPELINE)
    built = _pivot(root, "repro")
    assert built.returncode == 0, built.stdout + built.stderr
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "kept.txt").write_text("kept")
    assert _pivot(root, "track", "data/raw/kept.txt").returncode == 0
    (raw / "kept.txt").unlink()
    (raw / "new.txt").write_text("new")
    lmdb_dir = root / ".pivot" / "state.lmdb"
    modes = {path: path.stat().st_mode for path in [lmdb_dir, *lmdb_dir.iterdir()]}
    if damage == "garbage":
        (lmdb_dir / "data.mdb").write_bytes(b"\x00not an lmdb file" * 8192)
    else:
        for path, mode in modes.items():
            path.chmod(mode & ~0o222)
    try:
        restored = _pivot(root, "checkout", "--only-missing")
        tracked = _pivot(root, "track", "data/raw/new.txt")
    finally:
        for path, mode in modes.items():
            path.chmod(mode)

    assert restored.returncode == 0, restored.stdout + restored.stderr
    assert (raw / "kept.txt").read_text() == "kept"
    assert tracked.returncode == 0, tracked.stdout + tracked.stderr
    assert (raw / "new.txt.pvt").is_file()


_ROOT_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def alive() -> str:
    return "alive"


with Pipeline("a") as pipeline:
    alive()
"""

# `fake.pvt` names `value.txt` at the hash of "2", which the cache holds (`two.txt`'s content)
_CACHED_HASH_MEMBER_PIPELINE = """\
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


def _two_pipelines(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Pipeline `a` at the root and pipeline `b` in `other/`, each run once. b's stage wrote
    `fake.pvt` in `data/b/tree`, naming `value.txt` at a hash the cache holds. Returns the root
    and b's output directory."""
    root = tmp_path / "project"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    assert _pivot(root, "init").returncode == 0
    (root / "pipeline.py").write_text(_ROOT_PIPELINE)
    two = tmp_path / "two.txt"
    two.write_text("2")
    other = root / "other"
    other.mkdir()
    (other / "pipeline.py").write_text(
        _CACHED_HASH_MEMBER_PIPELINE.replace("HASH_OF_2", cache.hash_file(two)[0])
    )
    for cwd in (other, root):
        built = _pivot(cwd, "repro")
        assert built.returncode == 0, built.stdout + built.stderr
    tree = root / "data" / "b" / "tree"
    assert track.read_pvt_file(tree / "fake.pvt") is not None
    return root, tree


def _tracked_paths(proc: subprocess.CompletedProcess[str]) -> list[str]:
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return sorted(info["path"] for info in json.loads(proc.stdout)["tracked_files"])


def test_a_command_run_from_another_pipeline_skips_a_stages_own_member(
    tmp_path: pathlib.Path,
) -> None:
    """The command runs from the root, which loads pipeline `a` alone. b's `fake.pvt` is b's own
    member by b's lock and the listing committed beside it, so the target-less `checkout --force`
    restores nothing it names (before, it rewrote `value.txt` from "1" to "2" and exited 0),
    `status` lists no tracked file, and `repro` of pipeline `a` does not stop on `value.txt` once
    it is deleted."""
    root, tree = _two_pipelines(tmp_path)
    written = (tree / "fake.pvt").read_text()

    forced = _pivot(root, "checkout", "--force")
    status = _pivot(root, "status", "--tracked-only", "--json")

    assert forced.returncode == 0, forced.stdout + forced.stderr
    assert (tree / "value.txt").read_text() == "1", forced.stdout + forced.stderr
    assert (tree / "fake.pvt").read_text() == written
    assert _tracked_paths(status) == [], status.stdout

    (tree / "value.txt").unlink()
    rerun = _pivot(root, "repro")

    assert rerun.returncode == 0, rerun.stdout + rerun.stderr
    assert "tracked files are missing" not in rerun.stdout + rerun.stderr


def test_a_member_of_a_pipeline_that_fails_to_load_is_skipped(tmp_path: pathlib.Path) -> None:
    """After b ran, `other/pipeline.py` raises when imported. A command run from the root never
    imports it, and b's lock still records `fake.pvt` as its member, so the target-less
    `checkout --force` restores nothing it names and `status` lists no tracked file."""
    root, tree = _two_pipelines(tmp_path)
    (root / "other" / "pipeline.py").write_text('raise RuntimeError("b is broken")\n')

    forced = _pivot(root, "checkout", "--force")
    status = _pivot(root, "status", "--tracked-only", "--json")

    assert forced.returncode == 0, forced.stdout + forced.stderr
    assert "b is broken" not in forced.stdout + forced.stderr
    assert (tree / "value.txt").read_text() == "1", forced.stdout + forced.stderr
    assert _tracked_paths(status) == [], status.stdout


def test_a_member_whose_stage_no_longer_exists_is_skipped(tmp_path: pathlib.Path) -> None:
    """After b ran, its stage `tree` is renamed `grove`. Pivot deletes no lock, so
    `stages/b/tree.lock` stays, and so does `data/b/tree` with its `fake.pvt`, which no stage
    writes any more. That lock and its committed listing record `fake.pvt` with its current bytes,
    so it is still a stage's own member, not a tracked file, from either pipeline: the target-less
    `checkout --force` restores nothing it names and `status` lists nothing.

    Skipping it is safe. No command replaces an output that no stage declares, so the directory
    stays as it is; the skip only keeps the file out of the tracked files, and the bytes it holds
    are the ones the lock records pivot writing there. A user's `.pvt` there has other bytes and
    is still discovered (the next test)."""
    root, tree = _two_pipelines(tmp_path)
    other = root / "other"
    renamed = (other / "pipeline.py").read_text().replace("def tree(", "def grove(")
    (other / "pipeline.py").write_text(renamed.replace("    tree()", "    grove()"))
    built = _pivot(other, "repro")
    assert built.returncode == 0, built.stdout + built.stderr
    assert (root / ".pivot" / "stages" / "b" / "tree.lock").is_file()
    written = (tree / "fake.pvt").read_text()

    for cwd in (root, other):
        forced = _pivot(cwd, "checkout", "--force")
        status = _pivot(cwd, "status", "--tracked-only", "--json")

        assert forced.returncode == 0, forced.stdout + forced.stderr
        assert (tree / "value.txt").read_text() == "1", forced.stdout + forced.stderr
        assert (tree / "fake.pvt").read_text() == written
        assert _tracked_paths(status) == [], status.stdout


def test_a_users_sidecar_at_a_member_path_of_another_pipeline_stays_a_tracked_file(
    tmp_path: pathlib.Path,
) -> None:
    """The user puts a sidecar of their own at b's `fake.pvt` (other bytes, naming a tracked file
    `fake` copied beside it). b's lock lists that path with other content, so from the root it is
    no member: `status` lists the user's file. From `other/`, where b is loaded, the forced
    checkout refuses b's output, naming the sidecar, and changes nothing."""
    root, tree = _two_pipelines(tmp_path)
    raw = root / "data" / "raw" / "fake"
    raw.parent.mkdir(parents=True)
    raw.write_text("user bytes")
    tracked = _pivot(root, "track", "data/raw/fake")
    assert tracked.returncode == 0, tracked.stdout + tracked.stderr
    user_fake = track.get_pvt_path(raw).read_text()
    assert user_fake != (tree / "fake.pvt").read_text()
    shutil.copy2(raw, tree / "fake")
    (tree / "fake.pvt").unlink()  # the stage's file is a read-only cache link
    (tree / "fake.pvt").write_text(user_fake)

    status = _pivot(root, "status", "--tracked-only", "--json")
    forced = _pivot(root / "other", "checkout", "--force")

    assert _tracked_paths(status) == ["data/b/tree/fake", "data/raw/fake"], status.stdout
    assert forced.returncode != 0, forced.stdout + forced.stderr
    assert "Refusing to replace" in forced.stdout + forced.stderr
    assert "data/b/tree/fake.pvt" in forced.stdout + forced.stderr
    assert (tree / "fake").read_text() == "user bytes"
    assert (tree / "fake.pvt").read_text() == user_fake
    assert (tree / "value.txt").read_text() == "1"
