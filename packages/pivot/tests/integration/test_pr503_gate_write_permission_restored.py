"""A run made while the state directory had lost its write permission stays readable after the
permission returns.

`core.state_dir: s1`. A first run fills `s1`; `s1` becomes mode 0555 while its `state.lmdb` and
`locks` stay writable, so Pivot 0.5.2 ran such a project's commands and #503 falls back to `.pivot`
for the second run, after an input edit: its stage lock, run record and cache objects go to
`.pivot`. Then `s1` becomes writable again. The second run is the project's newest state, and it
exists, in `.pivot`. Pivot 0.5.2, main and `cac249c2` list both runs, show the second by its ID,
report the stage cached, keep the present output on `checkout`, and restore the second run's output
when it is deleted. Reading needs no write permission and the fallback decides only where Pivot
writes, so the head must too: no stale lock in `s1` may shadow the newer one in `.pivot`.
"""

from __future__ import annotations

import json
import os
import typing

import pytest

from tests import helpers

if typing.TYPE_CHECKING:
    import pathlib

_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def process(value: str) -> str:
    return value.upper()


with Pipeline("test") as pipeline:
    process(pipeline.input("input.txt", t=str))
"""


@pytest.fixture(autouse=True)
def _not_root() -> None:
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permission bits")


def _restored_project(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, list[str]]:
    """Run 1 in `s1`; `s1` 0555; run 2 (falls back to `.pivot`); `s1` 0755 again."""
    home = tmp_path / "home"
    root = tmp_path / "project"
    root.mkdir()
    assert helpers.run_cli(root, home, "git", "init", "-q").returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: s1\n")
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("first\n")
    (root / "pipeline.py").write_text(_PIPELINE)
    first = helpers.run_pivot(root, home, "repro")
    assert first.returncode == 0, first.stdout + first.stderr
    s1 = root / "s1"
    (s1 / "locks").mkdir(exist_ok=True)
    s1.chmod(0o555)
    try:
        (root / "data" / "raw" / "input.txt").write_text("second\n")
        second = helpers.run_pivot(root, home, "repro")
        assert second.returncode == 0, second.stdout + second.stderr
        listed = helpers.run_pivot(root, home, "history", "--json")
        run_ids = sorted(run["run_id"] for run in json.loads(listed.stdout))
        assert len(run_ids) == 2, listed.stdout
    finally:
        s1.chmod(0o755)
    return home, root, run_ids


def test_history_and_show_still_find_the_run_made_during_the_fallback(
    tmp_path: pathlib.Path,
) -> None:
    home, root, run_ids = _restored_project(tmp_path)

    listed = helpers.run_pivot(root, home, "history", "--json")
    shown = helpers.run_pivot(root, home, "show", run_ids[-1], "--json")

    assert listed.returncode == 0, listed.stdout + listed.stderr
    assert sorted(run["run_id"] for run in json.loads(listed.stdout)) == run_ids, listed.stdout
    assert shown.returncode == 0, shown.stdout + shown.stderr
    assert json.loads(shown.stdout)["run_id"] == run_ids[-1], shown.stdout


def test_status_and_checkout_use_the_newest_lock_not_the_stale_one_in_s1(
    tmp_path: pathlib.Path,
) -> None:
    home, root, _run_ids = _restored_project(tmp_path)
    output = root / "data" / "test" / "process.txt"
    assert output.read_text() == "SECOND\n"

    status = helpers.run_pivot(root, home, "status")
    kept = helpers.run_pivot(root, home, "checkout")
    output.unlink()
    restored = helpers.run_pivot(root, home, "checkout")

    assert status.returncode == 0, status.stdout + status.stderr
    assert "1 cached" in status.stdout, status.stdout
    assert kept.returncode == 0, kept.stdout + kept.stderr
    assert restored.returncode == 0, restored.stdout + restored.stderr
    assert output.read_text() == "SECOND\n", restored.stdout + restored.stderr
