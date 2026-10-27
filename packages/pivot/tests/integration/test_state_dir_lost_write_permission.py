"""A state directory that lost its write permission after a run, while its database stays writable.

`core.state_dir: s1`. A run fills `s1`; then `s1` becomes mode 0555, so nothing new can be
created in it, while its `state.lmdb` can still be written. Pivot 0.5.2 ran such a project's
commands, so the record decides with the value that the path falls back to `.pivot`, for every
reader in every process. One `pivot repro` keeps its locks, its run record and its cache objects
in `.pivot`, and a worker takes the secret parameter's key from `.pivot` as `pivot status` does,
so `status` right after `repro` reports the stage cached.
"""

from __future__ import annotations

import json
import os
import typing

import pytest

from pivot.storage import state
from tests import helpers

if typing.TYPE_CHECKING:
    import pathlib

_TWO_STAGES = """\
from pivot import Pipeline, stage


@stage
def a(value: str) -> str:
    return value.upper()


@stage
def b(value: str) -> str:
    return value + "b"


with Pipeline("test") as pipeline:
    b(a(pipeline.input("input.txt", t=str)))
"""

_SECRET_STAGE = """\
import pydantic

from pivot import Pipeline, StageParams, stage


class SecretParams(StageParams):
    token: pydantic.SecretStr = pydantic.SecretStr("a-secret")


@stage
def secret_length(params: SecretParams) -> dict:
    return {"length": len(params.token.get_secret_value())}


with Pipeline("test") as pipeline:
    secret_length(params=SecretParams())
"""


@pytest.fixture(autouse=True)
def _not_root() -> None:
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permission bits")


def _project(tmp_path: pathlib.Path, pipeline: str) -> tuple[pathlib.Path, pathlib.Path]:
    """A project whose first run filled `s1`; `s1` is then made mode 0555."""
    home = tmp_path / "home"
    root = tmp_path / "project"
    root.mkdir()
    assert helpers.run_cli(root, home, "git", "init", "-q").returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: s1\n")
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("first\n")
    (root / "pipeline.py").write_text(pipeline)
    first = helpers.run_pivot(root, home, "repro")
    assert first.returncode == 0, first.stdout + first.stderr
    return home, root


def _keys(state_dir: pathlib.Path, prefix: bytes) -> int:
    with state.StateDB(state_dir, readonly=True) as db:
        return len(list(db.iter_prefix(prefix)))


def test_a_run_after_the_state_directory_lost_write_permission_keeps_one_directory(
    tmp_path: pathlib.Path,
) -> None:
    home, root = _project(tmp_path, _TWO_STAGES)
    s1 = root / "s1"
    (root / "data" / "raw" / "input.txt").write_text("second\n")
    s1.chmod(0o555)
    try:
        ran = helpers.run_pivot(root, home, "repro")
    finally:
        s1.chmod(0o755)

    output = ran.stdout + ran.stderr
    assert ran.returncode == 0, output
    assert ran.stderr.count("Pivot uses the default .pivot.") == 1, ran.stderr
    pivot_dir = root / ".pivot"
    locks = sorted(path.name for path in (pivot_dir / "stages" / "test").iterdir())
    assert locks == ["a.lock", "b.lock"], output
    # This run's record and fingerprint manifests are in `.pivot`; `s1` keeps the first run's.
    assert (_keys(pivot_dir, b"run:"), _keys(pivot_dir, b"sm:")) == (1, 2), output
    assert _keys(s1, b"run:") == 1, output
    objects = [
        path
        for path in (pivot_dir / "cache" / "files").rglob("*")
        if path.is_file() and not path.name.startswith(".")
    ]
    assert len(objects) == 2, output


def test_status_right_after_repro_reports_a_secret_stage_cached(tmp_path: pathlib.Path) -> None:
    home, root = _project(tmp_path, _SECRET_STAGE)
    s1 = root / "s1"
    s1.chmod(0o555)
    try:
        ran = helpers.run_pivot(root, home, "repro")
        status = helpers.run_pivot(root, home, "status")
    finally:
        s1.chmod(0o755)

    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert status.returncode == 0, status.stdout + status.stderr
    assert "1 cached" in status.stdout, status.stdout
    assert "0 stale" in status.stdout, status.stdout
    keys = sorted(path.relative_to(root).as_posix() for path in root.rglob("secret.key"))
    assert keys == [".pivot/secret.key", "s1/secret.key"]


def test_the_recorded_runs_stay_readable_after_the_directory_lost_write_permission(
    tmp_path: pathlib.Path,
) -> None:
    """Reading needs no write permission: `history`, `history --json` and `show` still find the
    runs recorded in the configured directory, as Pivot 0.5.2 does."""
    home, root = _project(tmp_path, _TWO_STAGES)
    (root / "data" / "raw" / "input.txt").write_text("second\n")
    second = helpers.run_pivot(root, home, "repro")
    assert second.returncode == 0, second.stdout + second.stderr
    assert _keys(root / "s1", b"run:") == 2
    s1 = root / "s1"
    s1.chmod(0o555)
    try:
        listed = helpers.run_pivot(root, home, "history")
        as_json = helpers.run_pivot(root, home, "history", "--json")
        shown = helpers.run_pivot(root, home, "show")
    finally:
        s1.chmod(0o755)

    assert listed.returncode == 0, listed.stdout + listed.stderr
    assert "No runs recorded" not in listed.stdout, listed.stdout
    assert len(listed.stdout.strip().splitlines()) == 4, listed.stdout
    assert as_json.returncode == 0, as_json.stdout + as_json.stderr
    runs = json.loads(as_json.stdout)
    assert len(runs) == 2, as_json.stdout
    assert [run["run_id"] for run in runs] == sorted(
        (run["run_id"] for run in runs), reverse=True
    ), as_json.stdout
    assert shown.returncode == 0, shown.stdout + shown.stderr
    assert runs[0]["run_id"] in shown.stdout, shown.stdout
    # Each command reports the path it cannot write, once, and nothing else.
    for result in (listed, as_json, shown):
        assert result.stderr.count("Pivot uses the default .pivot.") == 1, result.stderr
        assert "holds stage locks or cache objects" not in result.stderr, result.stderr


def test_checkout_restores_from_the_directory_that_lost_write_permission(
    tmp_path: pathlib.Path,
) -> None:
    """A restore reads the lock and the cache object where the run left them."""
    home, root = _project(tmp_path, _TWO_STAGES)
    output = root / "data" / "test" / "b.txt"
    produced = output.read_text()
    output.unlink()
    s1 = root / "s1"
    s1.chmod(0o555)
    try:
        restored = helpers.run_pivot(root, home, "checkout")
    finally:
        s1.chmod(0o755)

    assert restored.returncode == 0, restored.stdout + restored.stderr
    assert output.read_text() == produced, restored.stdout + restored.stderr
