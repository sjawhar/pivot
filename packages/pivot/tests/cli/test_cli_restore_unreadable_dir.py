"""A forced restore must not destroy a tracked file hidden under a directory Pivot cannot read.

``ownership.find_output_overlaps`` walks a directory output with ``os.walk``; a directory it
cannot enter may hide a user ``*.pvt`` sidecar, so the walk must not skip it. A restore that
cannot prove an output holds no user file refuses, naming the directory it could not read, and
deletes nothing.

This is the recheck's red test (BET-265 round 5), kept verbatim; only this module docstring is
this PR's.
"""

from __future__ import annotations

import contextlib
import stat
from typing import TYPE_CHECKING

import pytest

from conftest import isolated_pivot_dir
from pivot import cli
from pivot.storage import track

if TYPE_CHECKING:
    import pathlib

    from click.testing import CliRunner


_A_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def alive() -> str:
    return "alive"


with Pipeline("a") as pipeline:
    alive()
"""

_DIRECTORY_OUTPUT_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, directory, stage


@stage
def tree() -> Annotated[dict[str, str], directory]:
    return {"g.txt": "g"}


with Pipeline("b") as pipeline:
    tree()
"""


@contextlib.contextmanager
def _unreadable(path: pathlib.Path):
    """Make ``path`` mode 0 for the body, then restore it so tmp cleanup can remove it."""
    original = path.stat().st_mode
    path.chmod(0)
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            path.chmod(original | stat.S_IRWXU)


@pytest.mark.parametrize(
    "command",
    [
        pytest.param(["checkout", "b/tree", "--force"], id="checkout-stage-force"),
        pytest.param(["checkout", "--force"], id="checkout-all-force"),
    ],
)
def test_forced_restore_keeps_a_tracked_file_under_an_unreadable_directory(
    runner: CliRunner, tmp_path: pathlib.Path, command: list[str]
) -> None:
    """A tracked file lives inside a stage's directory output, under a subdirectory the user later
    locked to mode 0. The forced restore must keep that tracked file: it cannot read the directory,
    so it cannot prove the output holds no user file, and must refuse rather than delete it."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        # 1. The stage runs and records its directory output (lock + cache).
        (root / "pipeline.py").write_text(_DIRECTORY_OUTPUT_PIPELINE)
        assert runner.invoke(cli.cli, ["repro"]).exit_code == 0

        # 2. The user tracks a file inside the output while the stage is not registered (as main
        #    allowed), so `track` does not refuse it.
        (root / "pipeline.py").write_text(_A_PIPELINE)
        secret_dir = root / "data" / "b" / "tree" / "sekret"
        secret_dir.mkdir(parents=True)
        tracked = secret_dir / "h.txt"
        tracked.write_text("user secret bytes")
        tracked_rel = str(tracked.relative_to(root))
        assert runner.invoke(cli.cli, ["track", tracked_rel]).exit_code == 0
        sidecar = track.get_pvt_path(tracked)
        assert sidecar.is_file()

        # 3. The stage is active again, and the user has locked the directory down.
        (root / "pipeline.py").write_text(_DIRECTORY_OUTPUT_PIPELINE)

        with _unreadable(secret_dir):
            result = runner.invoke(cli.cli, command)

        # The tracked file (and its sidecar) must survive, whatever the command decides.
        assert tracked.is_file(), (
            f"forced restore deleted a tracked file it could not see: {result.output}"
        )
        assert tracked.read_text() == "user secret bytes"
        assert sidecar.is_file()
        # A restore that cannot read the directory must refuse, not silently succeed.
        assert result.exit_code != 0, result.output
