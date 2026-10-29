"""The .pvt walk's directory memo through the CLI.

Only the repro/run pre-flight records the memo; every other command reads it. The memo holds
names only, so a .pvt rewritten in place is read anew by ``status`` and by the pre-flight, both
within the 2 s settle window and after it.
"""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING

import pytest

from pivot import cli, compose
from pivot.storage import cache, track
from pivot.types import TrackedFileStatus
from tests import helpers

if TYPE_CHECKING:
    import pathlib

    from click.testing import CliRunner

_REAL_TIME_NS = time.time_ns
_PAST_SETTLE_WINDOW_NS = 10_000_000_000


@compose.stage
def shout(raw: str) -> str:
    return raw.upper()


@pytest.fixture
def memo_project(
    mock_discovery: compose.Pipeline,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> pathlib.Path:
    """A project whose stage reads ``data/raw/input.txt``, on a filesystem the memo trusts."""
    root = mock_discovery.root
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "input.txt").write_text("original")
    mountinfo = tmp_path_factory.mktemp("mounts") / "mountinfo"
    helpers.mount_table(monkeypatch, mountinfo, [(os.stat(root).st_dev, "/", "ext4")])
    with mock_discovery:
        shout(mock_discovery.input("input.txt", t=str))
    return root


def _helper_settle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Move the clock past the settle window, so every directory made so far is settled."""
    monkeypatch.setattr(time, "time_ns", lambda: _REAL_TIME_NS() + _PAST_SETTLE_WINDOW_NS)


def _helper_memo() -> track._Memo | None:
    """What the account's memo file holds; None when there is no memo file."""
    path = track._memo_path()
    return track._unpack_memo(path.read_bytes()) if path.exists() else None


def _helper_stamp(directory: pathlib.Path) -> tuple[int, int]:
    directory_stat = directory.stat()
    return directory_stat.st_mtime_ns, directory_stat.st_ctime_ns


def test_only_the_run_preflight_records_the_pvt_walk_memo(
    memo_project: pathlib.Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """status, track, checkout, dry-run and explain walk the project and record nothing."""
    assert runner.invoke(cli.cli, ["track", "data/raw/input.txt"]).exit_code == 0
    (memo_project / "notes.txt").write_text("notes")
    _helper_settle(monkeypatch)

    for args in (
        ["status"],
        ["status", "--tracked-only", "--json"],
        ["track", "notes.txt"],
        ["checkout"],
        ["repro", "--dry-run"],
        ["repro", "--explain"],
    ):
        result = runner.invoke(cli.cli, args)
        assert result.exit_code == 0, f"pivot {' '.join(args)}: {result.output}"
        assert _helper_memo() is None, f"pivot {' '.join(args)} records no memo"

    result = runner.invoke(cli.cli, ["repro"])

    assert result.exit_code == 0, result.output
    memo = _helper_memo()
    assert memo is not None, "repro's pre-flight records the memo"
    assert memo["root"] == str(memo_project.resolve())
    assert {"", "data", "data/raw"} <= set(memo["dirs"])


def _helper_track_and_run(runner: CliRunner) -> None:
    """Track the input and run twice: the first run makes the state database, whose directory
    holds the memo file, and the second records the memo."""
    assert runner.invoke(cli.cli, ["track", "data/raw/input.txt"]).exit_code == 0
    for _ in range(2):
        result = runner.invoke(cli.cli, ["repro"])
        assert result.exit_code == 0, result.output


def test_pvt_rewritten_in_place_within_the_settle_window_is_read_anew(
    memo_project: pathlib.Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Within 2 s of data/raw's last change nothing is recorded; the rewritten .pvt is read."""
    before = _REAL_TIME_NS()
    monkeypatch.setattr(time, "time_ns", lambda: before)
    _helper_track_and_run(runner)
    memo = _helper_memo()
    assert memo is not None and "data/raw" not in memo["dirs"], "not recorded inside the window"

    _helper_rewrite_in_place_and_check(memo_project, runner)


def test_pvt_rewritten_in_place_after_the_settle_window_is_read_anew(
    memo_project: pathlib.Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """data/raw's listing is recorded, yet the .pvt rewritten in place is read anew."""
    _helper_settle(monkeypatch)
    _helper_track_and_run(runner)
    memo = _helper_memo()
    assert memo is not None and "data/raw" in memo["dirs"], "recorded once settled"

    _helper_rewrite_in_place_and_check(memo_project, runner)


def _helper_rewrite_in_place_and_check(memo_project: pathlib.Path, runner: CliRunner) -> None:
    """Change the data and rewrite its .pvt in place to match; status and repro see the match."""
    data = memo_project / "data" / "raw" / "input.txt"
    pvt_path = track.get_pvt_path(data)
    old = track.read_pvt_file(pvt_path)
    assert old is not None
    stamp = _helper_stamp(data.parent)
    data.write_text("changed")
    new_hash, _ = cache.hash_file(data)
    with pvt_path.open("r+") as pvt_file:
        text = pvt_file.read()
        pvt_file.seek(0)
        pvt_file.write(
            text.replace(old["hash"], new_hash).replace(f"size: {old['size']}", "size: 7")
        )
        pvt_file.truncate()

    assert _helper_stamp(data.parent) == stamp, "rewriting files in place leaves the directory"
    status = runner.invoke(cli.cli, ["status", "--tracked-only", "--json"])
    assert status.exit_code == 0, status.output
    assert json.loads(status.stdout)["tracked_files"] == [
        {"path": "data/raw/input.txt", "status": TrackedFileStatus.CLEAN, "size": 7}
    ], "status compares the file with the rewritten .pvt"

    second = runner.invoke(cli.cli, ["repro"])

    assert second.exit_code == 0, second.output
    assert "has changed since tracking" not in second.output, (
        "the pre-flight checks the file against the rewritten .pvt"
    )
    assert (memo_project / "data" / "test" / "shout.txt").read_text() == "CHANGED"
