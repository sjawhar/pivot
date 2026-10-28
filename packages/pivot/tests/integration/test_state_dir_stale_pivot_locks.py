"""A stage lock in both `.pivot/stages` and the configured `stages`, the configured one current.

A project reproduced under the default `.pivot`, then given `core.state_dir: s1` and reproduced
again after its inputs changed, keeps each stage's older lock in `.pivot/stages` and its newer one
in `s1/stages`: the layout an upgrade from 0.5.2 leaves after one full run. Pivot reads the
`s1` locks, and tells the user once per command which `.pivot` locks it no longer uses, with the
exact command that removes them. Ordering the two locks opens each state database once per
command and parses each lock file once.
"""

from __future__ import annotations

import os
import pathlib
import shlex
import typing

import pytest
import yaml

from pivot import project
from pivot import state_dir as state_dir_mod
from pivot.storage import lock, state
from tests import helpers

_TWO_INPUTS = """\
from pivot import Pipeline, stage


@stage
def a(value: str) -> str:
    return value.upper()


@stage
def b(value: str) -> str:
    return value + "b"


with Pipeline("test") as pipeline:
    a(pipeline.input("a.txt", t=str))
    b(pipeline.input("b.txt", t=str))
"""

_STALE = "also holds older locks of"


def _upgraded(
    tmp_path: pathlib.Path, *, edit: tuple[str, ...]
) -> tuple[pathlib.Path, pathlib.Path]:
    """A project reproduced under `.pivot`, then under `core.state_dir: s1` after `edit` changed."""
    home = tmp_path / "home"
    root = (tmp_path / "project").resolve()
    root.mkdir()
    assert helpers.run_cli(root, home, "git", "init", "-q").returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    for name in ("a", "b"):
        (raw / f"{name}.txt").write_text(f"first {name}\n")
    (root / "pipeline.py").write_text(_TWO_INPUTS)
    first = helpers.run_pivot(root, home, "repro")
    assert first.returncode == 0, first.stdout + first.stderr
    (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: s1\n")
    for name in edit:
        (raw / f"{name}.txt").write_text(f"second {name}\n")
    second = helpers.run_pivot(root, home, "repro")
    assert second.returncode == 0, second.stdout + second.stderr
    return home, root


def _notices(stderr: str) -> list[str]:
    return [line for line in stderr.splitlines() if _STALE in line]


def test_status_names_every_stale_pivot_lock_with_one_command(tmp_path: pathlib.Path) -> None:
    """Every `.pivot` lock is stale: one notice, `rm -r .pivot/stages`, and that fixes it."""
    home, root = _upgraded(tmp_path, edit=("a", "b"))
    assert (root / ".pivot" / "stages" / "test" / "a.lock").is_file()
    assert (root / "s1" / "stages" / "test" / "a.lock").is_file()

    status = helpers.run_pivot(root, home, "status")

    assert status.returncode == 0, status.stdout + status.stderr
    assert "2 cached" in status.stdout, status.stdout
    notices = _notices(status.stderr)
    assert len(notices) == 1, status.stderr
    command = f"rm -r {shlex.quote(str(root / '.pivot' / 'stages'))}"
    shown = f"older locks of 2 stages whose current lock is in {root / 's1' / 'stages'}"
    assert shown in notices[0], notices[0]
    assert f"`{command}`" in notices[0], notices[0]
    assert "before moving" not in notices[0]

    removed = helpers.run_cli(root, home, *shlex.split(command))
    assert removed.returncode == 0, removed.stderr
    again = helpers.run_pivot(root, home, "status")
    assert again.returncode == 0, again.stdout + again.stderr
    assert "2 cached" in again.stdout, again.stdout
    assert not _notices(again.stderr), again.stderr


def test_a_mixed_layout_lists_only_the_stale_locks(tmp_path: pathlib.Path) -> None:
    """Only `a` was reproduced under `s1`; `b`'s lock is in `.pivot` alone and stays in use."""
    home, root = _upgraded(tmp_path, edit=("a",))
    stale = root / ".pivot" / "stages" / "test" / "a.lock"
    kept = root / ".pivot" / "stages" / "test" / "b.lock"
    assert not (root / "s1" / "stages" / "test" / "b.lock").exists()

    status = helpers.run_pivot(root, home, "status")

    assert status.returncode == 0, status.stdout + status.stderr
    assert "2 cached" in status.stdout, status.stdout
    notices = _notices(status.stderr)
    assert len(notices) == 1, status.stderr
    assert "older locks of 1 stage whose" in notices[0], notices[0]
    assert f"`rm {shlex.quote(str(stale))}`" in notices[0], notices[0]
    assert str(kept) not in notices[0]
    assert f"before moving any other lock out of {root / '.pivot' / 'stages'}" in notices[0]
    assert "holds stage locks or cache objects" in status.stderr, status.stderr

    stale.unlink()
    again = helpers.run_pivot(root, home, "status")
    assert again.returncode == 0, again.stdout + again.stderr
    assert "2 cached" in again.stdout, again.stdout
    assert not _notices(again.stderr), again.stderr


def test_quiet_prints_no_stale_notice(tmp_path: pathlib.Path) -> None:
    home, root = _upgraded(tmp_path, edit=("a", "b"))

    status = helpers.run_pivot(root, home, "--quiet", "status")

    assert status.returncode == 0, status.stdout + status.stderr
    assert not _notices(status.stderr), status.stderr


def test_a_current_pivot_lock_is_not_called_stale(tmp_path: pathlib.Path) -> None:
    """A run that fell back to `.pivot` while `s1` could not be written made the current lock."""
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permission bits")
    home = tmp_path / "home"
    root = tmp_path / "project"
    root.mkdir()
    assert helpers.run_cli(root, home, "git", "init", "-q").returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: s1\n")
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    for name in ("a", "b"):
        (raw / f"{name}.txt").write_text(f"first {name}\n")
    (root / "pipeline.py").write_text(_TWO_INPUTS)
    assert helpers.run_pivot(root, home, "repro").returncode == 0
    s1 = root / "s1"
    (s1 / "locks").mkdir(exist_ok=True)
    s1.chmod(0o555)
    try:
        (raw / "a.txt").write_text("second a\n")
        fallback = helpers.run_pivot(root, home, "repro")
        assert fallback.returncode == 0, fallback.stdout + fallback.stderr
    finally:
        s1.chmod(0o755)
    assert (root / ".pivot" / "stages" / "test" / "a.lock").is_file()

    status = helpers.run_pivot(root, home, "status")

    assert status.returncode == 0, status.stdout + status.stderr
    assert "2 cached" in status.stdout, status.stdout
    assert not _notices(status.stderr), status.stderr


def test_a_record_opens_each_database_once_and_parses_each_lock_once(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reading every stage's lock, twice, in one record: two database opens, one parse per read,
    no database call per stage while nothing writes, and the same answers both times."""
    _home, root = _upgraded(tmp_path, edit=("a", "b"))
    monkeypatch.chdir(root)
    monkeypatch.setattr(project, "_project_root_cache", root)
    opened = list[pathlib.Path]()
    real_state_db = state.StateDB

    def counting_state_db(
        state_dir: pathlib.Path, *args: typing.Any, **kwargs: typing.Any
    ) -> state.StateDB:
        opened.append(state_dir)
        return real_state_db(state_dir, *args, **kwargs)

    parsed = list[object]()
    real_load = yaml.load

    def counting_load(stream: typing.Any, Loader: typing.Any) -> object:  # noqa: N803
        parsed.append(stream)
        return real_load(stream, Loader=Loader)

    calls = list[str]()
    for method in ("newest_run_id", "lookup_run_cache", "list_runs"):
        real_method = getattr(real_state_db, method)

        def counting_method(
            self: state.StateDB,
            *args: typing.Any,
            _name: str = method,
            _real: typing.Any = real_method,
            **kwargs: typing.Any,
        ) -> typing.Any:
            calls.append(_name)
            return _real(self, *args, **kwargs)

        monkeypatch.setattr(real_state_db, method, counting_method)
    monkeypatch.setattr(state, "StateDB", counting_state_db)
    monkeypatch.setattr(yaml, "load", counting_load)
    stages_dir = root / "s1" / "stages"
    names = ["test/a", "test/b"]

    token = state_dir_mod.begin_invocation(report=False)
    try:
        first = [lock.StageLock(name, stages_dir).read() for name in names]
        first_parses, first_calls = len(parsed), list(calls)
        second = [lock.StageLock(name, stages_dir).read() for name in names]
    finally:
        state_dir_mod.end_invocation(token)

    assert sorted(opened) == sorted([root / "s1", root / ".pivot"]), opened
    assert first_parses == len(names), "the configured lock is parsed once, for order and read"
    assert len(parsed) == 2 * len(names), "the kept answer reads no other lock"
    assert sorted(first_calls) == sorted(["newest_run_id", "list_runs"] * 2), first_calls
    assert calls == first_calls, "no database call per stage while nothing writes"
    assert first == second
    assert all(data is not None for data in first)
    configured = [lock.StageLock(name, stages_dir).path for name in names]
    assert [lock.resolve_stage_lock_path(stages_dir, name) for name in names] == configured


def test_a_record_sees_a_run_recorded_while_it_is_open(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """As in a `--watch` session: a run recorded in `s1` while one record is open makes its lock
    the current one for that record's next read."""
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permission bits")
    home = tmp_path / "home"
    root = (tmp_path / "project").resolve()
    root.mkdir()
    assert helpers.run_cli(root, home, "git", "init", "-q").returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: s1\n")
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    for name in ("a", "b"):
        (raw / f"{name}.txt").write_text(f"first {name}\n")
    (root / "pipeline.py").write_text(_TWO_INPUTS)
    assert helpers.run_pivot(root, home, "repro").returncode == 0
    s1 = root / "s1"
    (s1 / "locks").mkdir(exist_ok=True)
    s1.chmod(0o555)
    try:
        (raw / "a.txt").write_text("second a\n")
        assert helpers.run_pivot(root, home, "repro").returncode == 0
    finally:
        s1.chmod(0o755)
    monkeypatch.chdir(root)
    monkeypatch.setattr(project, "_project_root_cache", root)
    stages_dir = s1 / "stages"

    token = state_dir_mod.begin_invocation(report=False)
    try:
        before = lock.resolve_stage_lock_path(stages_dir, "test/a")
        (raw / "a.txt").write_text("third a\n")
        third = helpers.run_pivot(root, home, "repro")
        assert third.returncode == 0, third.stdout + third.stderr
        after = lock.resolve_stage_lock_path(stages_dir, "test/a")
    finally:
        state_dir_mod.end_invocation(token)

    assert before == root / ".pivot" / "stages" / "test" / "a.lock"
    assert after == stages_dir / "test" / "a.lock"
