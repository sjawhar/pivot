"""`pivot repro --all --keep-going --dry-run --jsonl` on mapped stages (``Pipeline.map``) beside a
missing tracked file, or whose expansion the run refuses: the dry run lists the stages the run on
the same tree runs, skips, fails or blocks (spec R5).

Each test drives the installed ``pivot`` binary in a real git project, as
``tests/cli/test_map_removal.py`` does, and parses the dry run's stdout as one JSON document. The
run's stages are its `stage_complete` events.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from pivot import registry
from pivot.storage import cache, track
from tests import helpers

if TYPE_CHECKING:
    import pathlib
    import subprocess
    from collections.abc import Mapping

# left/cut is mapped over left/plan's files and reads data/raw/shared.txt besides; with
# _INCREMENTAL, left/grow also reads it and keeps an incremental output
_LEFT_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, directory, incremental, loaders, stage


@stage
def plan(
    entries: dict[str, int],
) -> Annotated[dict[str, dict[str, int]], directory, loaders.YAML[dict[str, int]]()]:
    return {f"{name}.yaml": {"value": value} for name, value in entries.items()}


@stage
def cut(entry: dict[str, int], shared: str) -> dict[str, int]:
    return {"cut": entry["value"]}


@stage
def grow(previous: str, shared: str) -> Annotated[str, incremental]:
    return previous + shared


with Pipeline("left") as pipeline:
    shared = pipeline.input("shared.txt", t=str)
    cut(pipeline.map(plan(pipeline.input("left_plan.yaml", t=dict[str, int]))), shared)
"""

_INCREMENTAL = "    grow(pipeline.previous(), shared)\n"

# healthy/keep is mapped over healthy/plan's files and reads nothing else
_HEALTHY_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, directory, loaders, stage


@stage
def plan(
    entries: dict[str, int],
) -> Annotated[dict[str, dict[str, int]], directory, loaders.YAML[dict[str, int]]()]:
    return {f"{name}.yaml": {"value": value} for name, value in entries.items()}


@stage
def keep(entry: dict[str, int]) -> dict[str, int]:
    return {"keep": entry["value"]}


with Pipeline("healthy") as pipeline:
    keep(pipeline.map(plan(pipeline.input("healthy_plan.yaml", t=dict[str, int]))))
"""

_ORPHAN_REASON = "its key left the listing of 'healthy/plan'"

# helpers.MAPPED_FILE_PIPELINE with p/build mapped over p/cut's instances and p/final over
# p/build's. A stage that takes a mapped stage's output is mapped too, so these are what can be
# downstream of p/cut
_CHAIN_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, directory, stage


@stage
def plan() -> Annotated[dict[str, str], directory]:
    return {"a.txt": "a", "b.txt": "b"}


@stage
def cut(entry: str) -> str:
    return entry


@stage
def build(cut: str) -> str:
    return cut + "!"


@stage
def final(build: str) -> str:
    return build + "?"


with Pipeline("p") as pipeline:
    final(build(cut(pipeline.map(plan()))))
"""

_UNCHANGED = {"would_run": False, "reason": "unchanged"}
_BLOCKED = {"would_run": False, "reason": "upstream 'p/cut' would fail"}

# p/cut is mapped over p/plan's files and writes one field, y: instance p/cut@<key> writes
# data/p/cut@<key>/y.txt
_FIELDS_PIPELINE = """\
from typing import Annotated, TypedDict

from pivot import Pipeline, directory, stage


class Fields(TypedDict):
    y: str


@stage
def plan() -> Annotated[dict[str, str], directory]:
    return {"a.txt": "a", "b.txt": "b"}


@stage
def cut(entry: str) -> Fields:
    return {"y": entry}


with Pipeline("p") as pipeline:
    cut(pipeline.map(plan()))
"""

# Pipeline p/cut@b's stage p/cut@b/y writes data/p/cut@b/y.txt, the output p/cut@b would write
_CLASHING_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def y() -> str:
    return "static"


with Pipeline("p/cut@b") as pipeline:
    y()
"""

# helpers.MAPPED_FILE_PIPELINE with p/cut's output incremental: p/cut@<key> keeps
# data/p/cut@<key>.txt from its last run
_INCREMENTAL_MAPPED_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, directory, incremental, stage


@stage
def plan() -> Annotated[dict[str, str], directory]:
    return {"a.txt": "a", "b.txt": "b"}


@stage
def cut(entry: str, previous: str) -> Annotated[str, incremental]:
    return previous + entry


with Pipeline("p") as pipeline:
    cut(pipeline.map(plan()), pipeline.previous())
"""

# _CHAIN_PIPELINE with p/cut's output incremental, as _INCREMENTAL_MAPPED_PIPELINE has it
_INCREMENTAL_CHAIN_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, directory, incremental, stage


@stage
def plan() -> Annotated[dict[str, str], directory]:
    return {"a.txt": "a", "b.txt": "b"}


@stage
def cut(entry: str, previous: str) -> Annotated[str, incremental]:
    return previous + entry


@stage
def build(cut: str) -> str:
    return cut + "!"


@stage
def final(build: str) -> str:
    return build + "?"


with Pipeline("p") as pipeline:
    final(build(cut(pipeline.map(plan()), pipeline.previous())))
"""

# _CHAIN_PIPELINE's mapped stages, each standing for its instances while not expanded
_CHAIN_TEMPLATES = ["p/cut", "p/build", "p/final"]


def _ok(proc: subprocess.CompletedProcess[str]) -> str:
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    return out


def _git_project(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """An initialized pivot project in a git repository, and the home its commands run with."""
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "project"
    root.mkdir()
    for args in (
        ("git", "init", "-q"),
        ("git", "config", "user.email", "t@example.com"),
        ("git", "config", "user.name", "t"),
    ):
        _ok(helpers.run_cli(root, home, *args))
    _ok(helpers.run_pivot(root, home, "init"))
    return root, home


def _write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _write_plans(root: pathlib.Path, plan: Mapping[str, int]) -> None:
    for name in ("left_plan.yaml", "healthy_plan.yaml"):
        _write(root / "data" / "raw" / name, yaml.safe_dump(dict(plan)))


def _dry_run(root: pathlib.Path, home: pathlib.Path, *targets: str) -> tuple[dict[str, Any], str]:
    """The dry run's document, parsed from its stdout alone, and its stderr."""
    proc = helpers.run_pivot(
        root, home, "repro", "--all", "--keep-going", "--dry-run", "--jsonl", *targets
    )
    _ok(proc)
    return json.loads(proc.stdout), proc.stderr


def _run(
    root: pathlib.Path, home: pathlib.Path, *targets: str
) -> tuple[subprocess.CompletedProcess[str], list[dict[str, Any]]]:
    """The run on the same tree, and its JSONL events."""
    proc = helpers.run_pivot(root, home, "repro", "--all", "--keep-going", "--jsonl", *targets)
    return proc, [json.loads(line) for line in proc.stdout.splitlines()]


def _statuses(events: list[dict[str, Any]]) -> dict[str, str]:
    return {
        event["stage"]: event["status"] for event in events if event["type"] == "stage_complete"
    }


def _skipping_lines(stderr: str) -> list[str]:
    return [line for line in stderr.splitlines() if "Skipping:" in line]


@pytest.mark.parametrize("mode", ["plain", "uncached-incremental", "stale-producer"])
def test_dry_run_keep_going_leaves_out_a_mapped_stage_and_its_orphan(
    tmp_path: pathlib.Path, mode: str
) -> None:
    """A5 with a left-out mapped stage: left/cut reads the missing tracked file shared.txt, its
    producer is current, and key b left its listing; healthy/keep is mapped over a current
    producer too, and key b left its listing. The dry run lists none of left/cut's instances,
    no row for left/cut and no removal of left/cut@b; it lists healthy/keep@a and removes
    healthy/keep@b. Its stages are those the run on the same tree runs or skips, and the run
    removes what the dry run says. In `uncached-incremental`, the left-out left/grow keeps an
    uncached incremental output, which stops neither the run nor healthy/keep's removal. In
    `stale-producer`, left/plan is stale, so left/cut would be not yet expanded: it still has no
    row."""
    stale = mode == "stale-producer"
    root, home = _git_project(tmp_path)
    (root / "left").mkdir()
    (root / "healthy").mkdir()
    (root / "left" / "pipeline.py").write_text(_LEFT_PIPELINE)
    (root / "healthy" / "pipeline.py").write_text(_HEALTHY_PIPELINE)
    _write_plans(root, {"a": 1, "b": 2})
    shared = root / "data" / "raw" / "shared.txt"
    shared.write_text("shared bytes")
    _ok(helpers.run_pivot(root, home, "track", "data/raw/shared.txt"))
    _ok(helpers.run_pivot(root, home, "repro", "--all"))
    _write_plans(root, {"a": 1})
    _ok(helpers.run_pivot(root, home, "repro", "--all", "left/plan", "healthy/plan"))
    shared.unlink()
    if mode == "uncached-incremental":
        (root / "left" / "pipeline.py").write_text(_LEFT_PIPELINE + _INCREMENTAL)
        _write(root / "data" / "left" / "grow.txt", "uncached\n")
    if stale:
        _write(root / "data" / "raw" / "left_plan.yaml", yaml.safe_dump({"a": 1, "c": 3}))

    document, stderr = _dry_run(root, home)
    ran, events = _run(root, home)

    assert document == {
        "stages": {
            "left/plan": (
                {"would_run": True, "reason": "Input dependencies changed"}
                if stale
                else {"would_run": False, "reason": "unchanged"}
            ),
            "healthy/plan": {"would_run": False, "reason": "unchanged"},
            "healthy/keep@a": {"would_run": False, "reason": "unchanged"},
        },
        "would_remove": {"healthy/keep@b": _ORPHAN_REASON},
    }, stderr
    left_out = [*(["left/grow"] if mode == "uncached-incremental" else []), "left/cut"]
    assert _skipping_lines(stderr) == [
        f"Skipping: Stage '{stage}' depends on input 'shared.txt' "
        + "(path data/raw/shared.txt), which does not exist"
        for stage in left_out
    ], stderr
    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert _skipping_lines(ran.stderr) == _skipping_lines(stderr), ran.stderr
    statuses = _statuses(events)
    assert statuses == {
        "left/plan": "ran" if stale else "cached",
        "healthy/plan": "cached",
        "healthy/keep@a": "cached",
    }, ran.stdout
    assert statuses.keys() == document["stages"].keys()
    expanded = [event for event in events if event["type"] == "stages_expanded"]
    assert expanded == [
        {
            "type": "stages_expanded",
            "template": "healthy/keep",
            "stages": ["healthy/keep@a"],
            "removed": ["healthy/keep@b"],
        }
    ], ran.stdout
    assert (root / ".pivot" / "stages" / "left" / "cut@b.lock").is_file(), "left/cut@b is kept"


def _refused_mapped_project(
    tmp_path: pathlib.Path, code: str = helpers.MAPPED_FILE_PIPELINE
) -> tuple[pathlib.Path, pathlib.Path]:
    """p/plan is current, and data/p/cut@a.txt, the output p/cut@a would write, is a missing
    tracked file; then ``code`` declares p/cut. The file is tracked before p/cut is declared,
    since `pivot track` refuses a path an output the pipeline declares holds."""
    root, home = _git_project(tmp_path)
    (root / "pipeline.py").write_text(helpers.PRODUCER_PIPELINE)
    _ok(helpers.run_pivot(root, home, "repro"))
    tracked = root / "data" / "p" / "cut@a.txt"
    tracked.write_text("tracked bytes")
    _ok(helpers.run_pivot(root, home, "track", "data/p/cut@a.txt"))
    tracked.unlink()
    (root / "pipeline.py").write_text(code)
    return root, home


def test_dry_run_keep_going_refuses_a_mapped_stage_over_a_missing_tracked_file(
    tmp_path: pathlib.Path,
) -> None:
    """R3 for instances in the dry run: p/cut's instance p/cut@a would write over the missing
    tracked file, so the run fails p/cut with its refusal, adds no instance, and blocks p/build
    and p/final, mapped over p/cut's and p/build's instances. The dry run lists the same stages:
    p/cut as one row that would not run, with the reason the run fails it with, and p/build and
    p/final as rows that would not run, naming p/cut. Its text says p/cut would fail and the
    others would be blocked, and so does --explain."""
    root, home = _refused_mapped_project(tmp_path, _CHAIN_PIPELINE)

    document, stderr = _dry_run(root, home)
    text = _ok(helpers.run_pivot(root, home, "repro", "--all", "--keep-going", "--dry-run"))
    explained = _ok(helpers.run_pivot(root, home, "repro", "--all", "--keep-going", "--explain"))
    ran, events = _run(root, home)

    assert ran.returncode == 1, ran.stdout + ran.stderr
    completed = {event["stage"]: event for event in events if event["type"] == "stage_complete"}
    statuses = _statuses(events)
    assert statuses == {
        "p/plan": "cached",
        "p/cut": "failed",
        "p/build": "blocked",
        "p/final": "blocked",
    }, ran.stdout
    reason = completed["p/cut"]["reason"]
    assert reason.startswith(
        "cannot expand over the files of 'p/plan': The following tracked files are missing:\n"
    ), reason
    assert "data/p/cut@a.txt: it is output 'data/p/cut@a.txt' of stage 'p/cut@a'" in reason
    assert document == {
        "stages": {
            "p/plan": _UNCHANGED,
            "p/cut": {"would_run": False, "reason": reason},
            "p/build": _BLOCKED,
            "p/final": _BLOCKED,
        }
    }, document
    assert document["stages"].keys() == statuses.keys()
    assert _skipping_lines(stderr) == [], stderr
    assert f"  p/cut: would fail ({reason})\n" in text, text
    assert "would expand" not in text, text
    assert f"Stage: p/cut\n  Status: WILL FAIL\n  Reason: {reason}\n" in explained, explained
    for chained in ("p/build", "p/final"):
        assert completed[chained]["reason"] == "upstream 'p/cut' failed", ran.stdout
        assert f"  {chained}: would be blocked (upstream 'p/cut' would fail)\n" in text, text
        assert (
            f"Stage: {chained}\n  Status: WILL BE BLOCKED\n  Reason: upstream 'p/cut' would fail\n"
        ) in explained, explained
    assert "Summary: 0 will run, 1 unchanged, 1 will fail, 2 will be blocked\n" in explained, (
        explained
    )


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a cache object whatever its mode")
def test_dry_run_keep_going_names_a_failed_restore_under_a_mapped_stage_as_the_run_does(
    tmp_path: pathlib.Path,
) -> None:
    """R3 for instances with --checkout-missing: data/p/cut@a.txt, the output p/cut@a would
    write, is a missing tracked file whose cache object this user cannot read (0o000), so the
    run start's restore fails and leaves it missing. The run fails p/cut with its refusal, which
    names the file as one whose restore failed, not one the local cache lacks; the dry run on
    the same tree shows p/cut with that same reason, and each warns once that it cannot restore
    the file."""
    root, home = _refused_mapped_project(tmp_path)
    tracked = root / "data" / "p" / "cut@a.txt"
    record = track.read_pvt_file(track.get_pvt_path(tracked))
    assert record is not None
    cache_object = cache.get_cache_path(root / ".pivot" / "cache" / "files", record["hash"])

    cache_object.chmod(0)
    try:
        dry_run = helpers.run_pivot(
            root,
            home,
            "repro",
            "--all",
            "--keep-going",
            "--checkout-missing",
            "--dry-run",
            "--jsonl",
        )
        ran = helpers.run_pivot(
            root, home, "repro", "--all", "--keep-going", "--checkout-missing", "--jsonl"
        )
    finally:
        cache_object.chmod(0o444)

    assert ran.returncode == 1, ran.stdout + ran.stderr
    completed = {
        event["stage"]: event
        for event in (json.loads(line) for line in ran.stdout.splitlines())
        if event["type"] == "stage_complete"
    }
    assert {stage: event["status"] for stage, event in completed.items()} == {
        "p/plan": "cached",
        "p/cut": "failed",
    }, ran.stdout
    reason = completed["p/cut"]["reason"]
    assert (
        "data/p/cut@a.txt: it is output 'data/p/cut@a.txt' of stage 'p/cut@a'; its restore "
        + "failed\n\nTheir restore failed: a 'Cannot restore' warning above names each cause.\n"
    ) in reason, reason
    assert "not in local cache" not in reason, reason
    _ok(dry_run)
    assert json.loads(dry_run.stdout) == {
        "stages": {"p/plan": _UNCHANGED, "p/cut": {"would_run": False, "reason": reason}}
    }, dry_run.stdout
    # The run's error names the path its checkout opened, which its link modes stage first
    assert helpers.restore_warnings(dry_run.stderr) == [
        f"Cannot restore {tracked}: [Errno 13] Permission denied: '{cache_object}'"
    ], dry_run.stderr
    run_warnings = helpers.restore_warnings(ran.stderr)
    assert len(run_warnings) == 1, ran.stderr
    assert run_warnings[0].startswith(f"Cannot restore {tracked}: [Errno 13] Permission denied: ")
    assert not tracked.exists()


def test_dry_run_keep_going_shows_a_mapped_stage_the_run_fails_for_any_refusal(
    tmp_path: pathlib.Path,
) -> None:
    """A refusal over no missing tracked file: p/plan writes a.txt and a.csv, two files of one
    key, so the run fails p/cut and blocks p/build and p/final. The dry run in leave-out mode
    shows p/cut as one row that would not run, with the run's reason, and p/build and p/final as
    blocked by it. Outside leave-out mode (`--all` without `--keep-going`) the dry run shows them
    as 0.10.4 does: mapped stages not yet expanded, which would run."""
    root, home = _git_project(tmp_path)
    (root / "pipeline.py").write_text(
        _CHAIN_PIPELINE.replace('{"a.txt": "a", "b.txt": "b"}', '{"a.txt": "a", "a.csv": "b"}')
    )
    first = helpers.run_pivot(root, home, "repro")
    assert first.returncode == 1, "the first run fails p/cut too"

    document, stderr = _dry_run(root, home)
    outside = helpers.run_pivot(root, home, "repro", "--all", "--dry-run", "--jsonl")
    ran, events = _run(root, home)

    assert ran.returncode == 1, ran.stdout + ran.stderr
    completed = {event["stage"]: event for event in events if event["type"] == "stage_complete"}
    reason = "cannot expand over the files of 'p/plan': files 'a.csv' and 'a.txt' have one key 'a'"
    assert {stage: (event["status"], event["reason"]) for stage, event in completed.items()} == {
        "p/plan": ("cached", "unchanged (generation)"),
        "p/cut": ("failed", reason),
        "p/build": ("blocked", "upstream 'p/cut' failed"),
        "p/final": ("blocked", "upstream 'p/cut' failed"),
    }, ran.stdout
    assert document == {
        "stages": {
            "p/plan": _UNCHANGED,
            "p/cut": {"would_run": False, "reason": reason},
            "p/build": _BLOCKED,
            "p/final": _BLOCKED,
        }
    }, stderr
    _ok(outside)
    assert json.loads(outside.stdout) == {
        "stages": {
            "p/plan": _UNCHANGED,
            "p/cut": {"would_run": True, "reason": f"Not yet expanded: {reason}"},
            "p/build": {
                "would_run": True,
                "reason": "Not yet expanded: mapped over the instances of 'p/cut', which is not "
                + "expanded",
            },
            "p/final": {
                "would_run": True,
                "reason": "Not yet expanded: mapped over the instances of 'p/build', which is not "
                + "expanded",
            },
        }
    }, outside.stdout


def test_dry_run_keep_going_checks_only_the_instances_a_target_names(
    tmp_path: pathlib.Path,
) -> None:
    """With target p/cut@b, the run adds p/cut@b alone, whose output holds no missing tracked
    file, and runs it; the dry run checks that instance alone and lists it."""
    root, home = _refused_mapped_project(tmp_path)

    document, stderr = _dry_run(root, home, "p/cut@b")
    ran, events = _run(root, home, "p/cut@b")

    assert document == {
        "stages": {
            "p/plan": {"would_run": False, "reason": "unchanged"},
            "p/cut@b": {"would_run": True, "reason": "No previous run"},
        }
    }, stderr
    assert ran.returncode == 0, ran.stdout + ran.stderr
    statuses = _statuses(events)
    assert statuses == {"p/plan": "cached", "p/cut@b": "ran"}, ran.stdout
    assert statuses.keys() == document["stages"].keys()
    assert (root / "data" / "p" / "cut@b.txt").read_text() == "b"
    assert not (root / "data" / "p" / "cut@a.txt").exists(), "the tracked file stays missing"


def test_dry_run_keep_going_plans_only_the_keys_an_instance_target_names(
    tmp_path: pathlib.Path,
) -> None:
    """p/cut@b's output data/p/cut@b/y.txt is also the output of pipeline p/cut@b's stage y, so
    a run over every key fails p/cut for the clash, and its dry run says p/cut would fail. With
    target p/cut@a the run plans key a alone (``expansion.plan_instances``), adds p/cut@a, runs
    it and exits 0; the dry run plans the same key and lists p/cut@a, not p/cut."""
    root, home = _git_project(tmp_path)
    _write(root / "p" / "pipeline.py", _FIELDS_PIPELINE)
    _write(root / "q" / "pipeline.py", _CLASHING_PIPELINE)
    first, first_events = _run(root, home)

    every, _ = _dry_run(root, home)
    document, stderr = _dry_run(root, home, "p/cut@a")
    ran, events = _run(root, home, "p/cut@a")

    assert first.returncode == 1, first.stdout + first.stderr
    clash = (
        "cannot expand over the files of 'p/plan': Output path conflict: 'p/cut@b:y' and "
        + "'p/cut@b/y' both write data/p/cut@b/y.txt."
    )
    assert {
        event["stage"]: (event["status"], event["reason"])
        for event in first_events
        if event["type"] == "stage_complete" and event["stage"] == "p/cut"
    } == {"p/cut": ("failed", clash)}, first.stdout
    assert every == {
        "stages": {
            "p/plan": _UNCHANGED,
            "p/cut@b/y": _UNCHANGED,
            "p/cut": {"would_run": False, "reason": clash},
        }
    }, every
    assert document == {
        "stages": {
            "p/plan": _UNCHANGED,
            "p/cut@a": {"would_run": True, "reason": "No previous run"},
        }
    }, stderr
    assert ran.returncode == 0, ran.stdout + ran.stderr
    statuses = _statuses(events)
    assert statuses == {"p/plan": "cached", "p/cut@a": "ran"}, ran.stdout
    assert statuses.keys() == document["stages"].keys()


def test_dry_run_keep_going_removes_no_instance_of_a_key_its_targets_leave_out(
    tmp_path: pathlib.Path,
) -> None:
    """p/cut@a and p/cut@b both ran, and p/plan's listing still holds both keys. With target
    p/cut@a the run adds p/cut@a alone and removes no instance: p/cut@b's key is still listed.
    The dry run, which also adds p/cut@a alone, would remove nothing either: an orphan is an
    instance of no key the listing holds (``Expansion``'s ``listed``), not one it did not add."""
    root, home = _git_project(tmp_path)
    (root / "pipeline.py").write_text(helpers.MAPPED_FILE_PIPELINE)
    _ok(helpers.run_pivot(root, home, "repro"))

    document, stderr = _dry_run(root, home, "p/cut@a")
    ran, events = _run(root, home, "p/cut@a")

    assert document == {"stages": {"p/plan": _UNCHANGED, "p/cut@a": _UNCHANGED}}, stderr
    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert _statuses(events) == {"p/plan": "cached", "p/cut@a": "cached"}, ran.stdout
    assert [event for event in events if event["type"] == "stages_expanded"] == [
        {"type": "stages_expanded", "template": "p/cut", "stages": ["p/cut@a"]}
    ], ran.stdout
    assert (root / ".pivot" / "stages" / "p" / "cut@b.lock").is_file(), "p/cut@b is kept"


def _uncached_incremental_project(
    tmp_path: pathlib.Path, code: str = _INCREMENTAL_MAPPED_PIPELINE
) -> tuple[pathlib.Path, pathlib.Path]:
    """``code``'s p/cut instances keep an incremental output, and every stage ran; then p/cut@a's
    lock is gone, so its output data/p/cut@a.txt is on disk but not in the cache."""
    root, home = _git_project(tmp_path)
    (root / "pipeline.py").write_text(code)
    _ok(helpers.run_pivot(root, home, "repro"))
    (root / ".pivot" / "stages" / "p" / "cut@a.lock").unlink()
    return root, home


def test_dry_run_keep_going_fails_a_mapped_stage_whose_incremental_output_is_not_cached(
    tmp_path: pathlib.Path,
) -> None:
    """p/cut@a's incremental output is on disk but not in the cache: the run fails p/cut with the
    refusal ``executor.core.incremental_refusal`` gives, runs no instance and exits 1, and the
    dry run shows p/cut as a stage that would fail, with that reason, and no instance."""
    root, home = _uncached_incremental_project(tmp_path)

    document, stderr = _dry_run(root, home)
    ran, events = _run(root, home)

    assert ran.returncode == 1, ran.stdout + ran.stderr
    statuses = _statuses(events)
    assert statuses == {"p/plan": "cached", "p/cut": "failed"}, ran.stdout
    reason = next(
        event["reason"]
        for event in events
        if event["type"] == "stage_complete" and event["stage"] == "p/cut"
    )
    assert reason.startswith(
        "cannot expand over the files of 'p/plan': The following incremental output files "
        + "exist but are not in cache:\n  - p/cut@a: "
    ), reason
    assert "cut@a.txt\n\nRunning the pipeline will DELETE these files" in reason, reason
    assert document == {
        "stages": {"p/plan": _UNCHANGED, "p/cut": {"would_run": False, "reason": reason}}
    }, stderr
    assert statuses.keys() == document["stages"].keys()


def test_dry_run_keep_going_expands_a_mapped_stage_over_an_uncached_incremental_output_if_allowed(
    tmp_path: pathlib.Path,
) -> None:
    """p/cut@a's incremental output is on disk but not in the cache. With
    --allow-uncached-incremental the run expands p/cut, runs p/cut@a and skips p/cut@b, and the
    dry run lists them so."""
    root, home = _uncached_incremental_project(tmp_path)

    document, stderr = _dry_run(root, home, "--allow-uncached-incremental")
    ran, events = _run(root, home, "--allow-uncached-incremental")

    assert ran.returncode == 0, ran.stdout + ran.stderr
    statuses = _statuses(events)
    assert statuses == {"p/plan": "cached", "p/cut@a": "ran", "p/cut@b": "cached"}, ran.stdout
    assert document == {
        "stages": {
            "p/plan": _UNCHANGED,
            "p/cut@a": {"would_run": True, "reason": "No previous run"},
            "p/cut@b": _UNCHANGED,
        }
    }, stderr
    assert statuses.keys() == document["stages"].keys()


def _empty_cache(root: pathlib.Path) -> None:
    """Remove every cache object."""
    for entry in (root / ".pivot" / "cache" / "files").iterdir():
        shutil.rmtree(entry)


def _lose_plan_listing(root: pathlib.Path) -> None:
    """Remove every cache object and p/plan's committed listing, as a clone that never pulled
    holds neither."""
    _empty_cache(root)
    (root / ".pivot" / "stages" / "p" / "plan=.listing").unlink()


def test_dry_run_keep_going_leaves_a_mapped_stage_not_expanded_over_a_producer_only_pullable(
    tmp_path: pathlib.Path,
) -> None:
    """p/plan's lock matches, but its directory, its cache objects and its committed listing
    are gone. A remote is configured, so the dry run counts p/plan current, as a pull would
    bring it up to date, and cannot read its listing. The run never pulls: it runs p/plan again,
    expands p/cut, p/build and p/final from what p/plan writes, and exits 0. So the dry run shows
    p/plan as a stage that would run, and p/cut, p/build and p/final as mapped stages not yet
    expanded, as 0.10.4 does, each a row for the instances the run adds; none would fail or be
    blocked."""
    root, home = _git_project(tmp_path)
    (root / "pipeline.py").write_text(_CHAIN_PIPELINE)
    _ok(helpers.run_pivot(root, home, "repro"))
    _ok(helpers.run_pivot(root, home, "config", "set", "remotes.origin", "s3://bucket/x"))
    _lose_plan_listing(root)
    shutil.rmtree(root / "data" / "p" / "plan")

    document, stderr = _dry_run(root, home)
    ran, events = _run(root, home)

    stages = document["stages"]
    assert stages.keys() == {"p/plan", *_CHAIN_TEMPLATES}, document
    assert stages["p/plan"] == {"would_run": True, "reason": "Outputs missing from cache"}
    cut = stages["p/cut"]["reason"]
    assert cut.startswith("Not yet expanded: the listing of 'p/plan' is missing"), cut
    assert cut.endswith("and `pivot repro --force p/plan` writes it again"), cut
    for chained, over in (("p/build", "p/cut"), ("p/final", "p/build")):
        assert stages[chained] == {
            "would_run": True,
            "reason": f"Not yet expanded: mapped over the instances of '{over}', which is not "
            + "expanded",
        }, stderr
    assert all(row["would_run"] for row in stages.values()), document
    assert ran.returncode == 0, ran.stdout + ran.stderr
    statuses = _statuses(events)
    assert statuses == {
        "p/plan": "ran",
        **{f"{template}@{key}": "cached" for template in _CHAIN_TEMPLATES for key in "ab"},
    }, ran.stdout
    assert {registry.template_of(stage, _CHAIN_TEMPLATES) or stage for stage in statuses} == (
        stages.keys()
    )


@pytest.mark.parametrize("refusal", ["overlap", "uncached-incremental"])
def test_dry_run_keep_going_fails_a_mapped_stage_over_a_producer_only_pullable(
    tmp_path: pathlib.Path, refusal: str
) -> None:
    """p/plan's lock matches, but its directory and the cache objects are gone, and its
    committed listing stays. A remote is configured, so the dry run counts p/plan current, as a
    pull would bring it up to date, and reads p/cut's keys from that listing. The run never
    pulls: it runs p/plan again and plans p/cut over the keys p/plan writes, the ones that
    listing gives. In `overlap` data/p/cut@a.txt, p/cut@a's output, is a missing tracked file
    with no cache object; in `uncached-incremental` p/cut@a's incremental output is on disk but
    not in the cache. Either way the run fails p/cut, blocks p/build and p/final and exits 1.
    Only a listing it cannot read shows p/cut as not yet expanded, so the dry run shows p/cut as
    a stage that would fail, with the run's reason, and p/build and p/final as blocked."""
    if refusal == "overlap":
        root, home = _git_project(tmp_path)
        (root / "pipeline.py").write_text(_CHAIN_PIPELINE)
        _ok(helpers.run_pivot(root, home, "repro"))
        tracked = root / "data" / "p" / "cut@a.txt"
        tracked.unlink()
        helpers.track_without_data(tracked)
        refused = "The following tracked files are missing:\n"
        named = "data/p/cut@a.txt: it is output 'data/p/cut@a.txt' of stage 'p/cut@a'"
    else:
        root, home = _uncached_incremental_project(tmp_path, _INCREMENTAL_CHAIN_PIPELINE)
        refused = "The following incremental output files exist but are not in cache:\n"
        named = "  - p/cut@a: "
    _ok(helpers.run_pivot(root, home, "config", "set", "remotes.origin", "s3://bucket/x"))
    _empty_cache(root)
    shutil.rmtree(root / "data" / "p" / "plan")

    document, stderr = _dry_run(root, home)
    ran, events = _run(root, home)

    assert ran.returncode == 1, ran.stdout + ran.stderr
    completed = {event["stage"]: event for event in events if event["type"] == "stage_complete"}
    statuses = {stage: event["status"] for stage, event in completed.items()}
    assert statuses == {
        "p/plan": "ran",
        "p/cut": "failed",
        "p/build": "blocked",
        "p/final": "blocked",
    }, ran.stdout
    reason = completed["p/cut"]["reason"]
    assert reason.startswith(f"cannot expand over the files of 'p/plan': {refused}"), reason
    assert named in reason, reason
    assert document == {
        "stages": {
            "p/plan": {"would_run": True, "reason": "Outputs missing from cache"},
            "p/cut": {"would_run": False, "reason": reason},
            "p/build": _BLOCKED,
            "p/final": _BLOCKED,
        }
    }, stderr
    assert statuses.keys() == document["stages"].keys()


@pytest.mark.parametrize("tree", ["listing_lost", "file_added"])
def test_dry_run_keep_going_reads_the_keys_a_no_commit_run_reads(
    tmp_path: pathlib.Path, tree: str
) -> None:
    """--no-commit: the run reads p/plan's keys from the directory its skip leaves in the
    workspace at its lock's record (``expansion.workspace_keys``). In `listing_lost` the cache
    objects and the committed listing of data/p/plan/ are gone and the directory stays: the run
    expands p/cut, p/build and p/final from the directory and exits 0, and the dry run reads the
    directory as the run does, listing every instance, none that would fail. In `file_added`,
    data/p/plan/c.txt was added after the last run: the run's skip restores data/p/plan/ to the
    lock's record, so the run expands keys a and b alone, and the dry run reads the listing the
    lock records, not the directory on disk."""
    root, home = _git_project(tmp_path)
    (root / "pipeline.py").write_text(_CHAIN_PIPELINE)
    _ok(helpers.run_pivot(root, home, "repro"))
    added = root / "data" / "p" / "plan" / "c.txt"
    if tree == "listing_lost":
        _lose_plan_listing(root)
    else:
        added.write_text("c")

    document, stderr = _dry_run(root, home, "--no-commit")
    ran, events = _run(root, home, "--no-commit")

    instances = [f"{template}@{key}" for template in _CHAIN_TEMPLATES for key in "ab"]
    assert document == {"stages": {"p/plan": _UNCHANGED, **dict.fromkeys(instances, _UNCHANGED)}}, (
        stderr
    )
    assert ran.returncode == 0, ran.stdout + ran.stderr
    statuses = _statuses(events)
    assert statuses == dict.fromkeys(["p/plan", *instances], "cached"), ran.stdout
    assert statuses.keys() == document["stages"].keys()
    assert not added.exists(), "the skip restores the directory to its lock's record"
