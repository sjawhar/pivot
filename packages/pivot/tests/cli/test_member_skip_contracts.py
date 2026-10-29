"""Which ``*.pvt`` files discovery leaves out as a stage's own members, and what it never does.

A user's sidecar at a member's path, with other bytes, stays a tracked file; a member below the
top of its output is still the stage's; a state database that cannot be opened fails no command
main ran; and a member of a pipeline the command did not load is the stage's too, so a command
run from another pipeline rewrites nothing it names. The first three came from the review of
sjawhar/pivot#540 at a67f4185 and are kept as the reviewer wrote them.
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


def test_a_command_run_from_another_pipeline_skips_a_stages_own_member(
    tmp_path: pathlib.Path,
) -> None:
    """Pipeline `b` lives in `other/`; the command runs from the root, which loads pipeline `a`.
    b's stage wrote `fake.pvt`, which names `value.txt` at a hash the cache holds. It is b's own
    member, so the target-less `checkout --force` restores nothing it names (before, it rewrote
    `value.txt` from "1" to "2" and exited 0), `status` lists no tracked file, and `repro` of
    pipeline `a` does not stop on `value.txt` once it is deleted."""
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
    written = (tree / "fake.pvt").read_text()
    assert track.read_pvt_file(tree / "fake.pvt") is not None

    forced = _pivot(root, "checkout", "--force")
    status = _pivot(root, "status", "--tracked-only", "--json")

    assert forced.returncode == 0, forced.stdout + forced.stderr
    assert (tree / "value.txt").read_text() == "1", forced.stdout + forced.stderr
    assert (tree / "fake.pvt").read_text() == written
    assert status.returncode == 0, status.stdout + status.stderr
    assert json.loads(status.stdout)["tracked_files"] == [], status.stdout

    (tree / "value.txt").unlink()
    rerun = _pivot(root, "repro")

    assert rerun.returncode == 0, rerun.stdout + rerun.stderr
    assert "tracked files are missing" not in rerun.stdout + rerun.stderr
