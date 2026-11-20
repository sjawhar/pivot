"""`pivot repro --all --keep-going --dry-run --jsonl` on mapped stages (``Pipeline.map``) beside a
missing tracked file, or whose expansion the run refuses: the dry run lists the stages the run on
the same tree runs, skips, fails or blocks (spec R5). Chains of mapped stages are in
``tests/cli/test_dry_run_keep_going_chain.py``.

Each test drives the installed ``pivot`` binary in a real git project (``helpers.git_project``),
as ``tests/cli/test_map_removal.py`` does, and parses the dry run's stdout as one JSON document.
The run's stages are its `stage_complete` events.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import pytest
import yaml

from pivot.storage import cache, track
from tests import helpers

if TYPE_CHECKING:
    import pathlib
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


def _write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _write_plans(root: pathlib.Path, plan: Mapping[str, int]) -> None:
    for name in ("left_plan.yaml", "healthy_plan.yaml"):
        _write(root / "data" / "raw" / name, yaml.safe_dump(dict(plan)))


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
    root, home = helpers.git_project(tmp_path)
    (root / "left").mkdir()
    (root / "healthy").mkdir()
    (root / "left" / "pipeline.py").write_text(_LEFT_PIPELINE)
    (root / "healthy" / "pipeline.py").write_text(_HEALTHY_PIPELINE)
    _write_plans(root, {"a": 1, "b": 2})
    shared = root / "data" / "raw" / "shared.txt"
    shared.write_text("shared bytes")
    helpers.ok(helpers.run_pivot(root, home, "track", "data/raw/shared.txt"))
    helpers.ok(helpers.run_pivot(root, home, "repro", "--all"))
    _write_plans(root, {"a": 1})
    helpers.ok(helpers.run_pivot(root, home, "repro", "--all", "left/plan", "healthy/plan"))
    shared.unlink()
    if mode == "uncached-incremental":
        (root / "left" / "pipeline.py").write_text(_LEFT_PIPELINE + _INCREMENTAL)
        _write(root / "data" / "left" / "grow.txt", "uncached\n")
    if stale:
        _write(root / "data" / "raw" / "left_plan.yaml", yaml.safe_dump({"a": 1, "c": 3}))

    document, stderr = helpers.dry_run_keep_going(root, home)
    ran, events = helpers.run_keep_going(root, home)

    assert document == {
        "stages": {
            "left/plan": (
                {"would_run": True, "reason": "Input dependencies changed"}
                if stale
                else helpers.UNCHANGED_ROW
            ),
            "healthy/plan": helpers.UNCHANGED_ROW,
            "healthy/keep@a": helpers.UNCHANGED_ROW,
        },
        "would_remove": {"healthy/keep@b": _ORPHAN_REASON},
    }, stderr
    left_out = [*(["left/grow"] if mode == "uncached-incremental" else []), "left/cut"]
    assert helpers.skipping_lines(stderr) == [
        f"Skipping: Stage '{stage}' depends on input 'shared.txt' "
        + "(path data/raw/shared.txt), which does not exist"
        for stage in left_out
    ], stderr
    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert helpers.skipping_lines(ran.stderr) == helpers.skipping_lines(stderr), ran.stderr
    statuses = helpers.stage_statuses(events)
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
    root, home = helpers.refused_mapped_project(tmp_path, helpers.MAPPED_FILE_PIPELINE)
    tracked = root / "data" / "p" / "cut@a.txt"
    record = track.read_pvt_file(track.get_pvt_path(tracked))
    assert record is not None
    cache_object = cache.get_cache_path(root / ".pivot" / "cache" / "files", record["hash"])

    cache_object.chmod(0)
    try:
        dry_run = helpers.run_pivot(root, home, *helpers.DRY_RUN_KEEP_GOING, "--checkout-missing")
        ran = helpers.run_pivot(
            root, home, "repro", "--all", "--keep-going", "--checkout-missing", "--jsonl"
        )
    finally:
        cache_object.chmod(0o444)

    assert ran.returncode == 1, ran.stdout + ran.stderr
    completed = helpers.completed_stages(ran.stdout)
    assert {stage: event["status"] for stage, event in completed.items()} == {
        "p/plan": "cached",
        "p/cut": "failed",
    }, ran.stdout
    reason = completed["p/cut"]["reason"]
    assert isinstance(reason, str)
    assert (
        "data/p/cut@a.txt: it is output 'data/p/cut@a.txt' of stage 'p/cut@a'; its restore "
        + "failed\n\nTheir restore failed: a 'Cannot restore' warning above names each cause.\n"
    ) in reason, reason
    assert "not in local cache" not in reason, reason
    helpers.ok(dry_run)
    assert json.loads(dry_run.stdout) == {
        "stages": {"p/plan": helpers.UNCHANGED_ROW, "p/cut": {"would_run": False, "reason": reason}}
    }, dry_run.stdout
    # The run's error names the path its checkout opened, which its link modes stage first
    assert helpers.restore_warnings(dry_run.stderr) == [
        f"Cannot restore {tracked}: [Errno 13] Permission denied: '{cache_object}'"
    ], dry_run.stderr
    run_warnings = helpers.restore_warnings(ran.stderr)
    assert len(run_warnings) == 1, ran.stderr
    assert run_warnings[0].startswith(f"Cannot restore {tracked}: [Errno 13] Permission denied: ")
    assert not tracked.exists()


def test_dry_run_keep_going_checks_only_the_instances_a_target_names(
    tmp_path: pathlib.Path,
) -> None:
    """With target p/cut@b, the run adds p/cut@b alone, whose output holds no missing tracked
    file, and runs it; the dry run checks that instance alone and lists it."""
    root, home = helpers.refused_mapped_project(tmp_path, helpers.MAPPED_FILE_PIPELINE)

    document, stderr = helpers.dry_run_keep_going(root, home, "p/cut@b")
    ran, events = helpers.run_keep_going(root, home, "p/cut@b")

    assert document == {
        "stages": {
            "p/plan": helpers.UNCHANGED_ROW,
            "p/cut@b": {"would_run": True, "reason": "No previous run"},
        }
    }, stderr
    assert ran.returncode == 0, ran.stdout + ran.stderr
    statuses = helpers.stage_statuses(events)
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
    root, home = helpers.git_project(tmp_path)
    _write(root / "p" / "pipeline.py", _FIELDS_PIPELINE)
    _write(root / "q" / "pipeline.py", _CLASHING_PIPELINE)
    first, _ = helpers.run_keep_going(root, home)

    every, _ = helpers.dry_run_keep_going(root, home)
    document, stderr = helpers.dry_run_keep_going(root, home, "p/cut@a")
    ran, events = helpers.run_keep_going(root, home, "p/cut@a")

    assert first.returncode == 1, first.stdout + first.stderr
    clash = (
        "cannot expand over the files of 'p/plan': Output path conflict: 'p/cut@b:y' and "
        + "'p/cut@b/y' both write data/p/cut@b/y.txt."
    )
    cut = helpers.completed_stages(first.stdout)["p/cut"]
    assert (cut["status"], cut["reason"]) == ("failed", clash), first.stdout
    assert every == {
        "stages": {
            "p/plan": helpers.UNCHANGED_ROW,
            "p/cut@b/y": helpers.UNCHANGED_ROW,
            "p/cut": {"would_run": False, "reason": clash},
        }
    }, every
    assert document == {
        "stages": {
            "p/plan": helpers.UNCHANGED_ROW,
            "p/cut@a": {"would_run": True, "reason": "No previous run"},
        }
    }, stderr
    assert ran.returncode == 0, ran.stdout + ran.stderr
    statuses = helpers.stage_statuses(events)
    assert statuses == {"p/plan": "cached", "p/cut@a": "ran"}, ran.stdout
    assert statuses.keys() == document["stages"].keys()


def test_dry_run_keep_going_removes_no_instance_of_a_key_its_targets_leave_out(
    tmp_path: pathlib.Path,
) -> None:
    """p/cut@a and p/cut@b both ran, and p/plan's listing still holds both keys. With target
    p/cut@a the run adds p/cut@a alone and removes no instance: p/cut@b's key is still listed.
    The dry run, which also adds p/cut@a alone, would remove nothing either: an orphan is an
    instance of no key the listing holds (``Expansion``'s ``listed``), not one it did not add."""
    root, home = helpers.git_project(tmp_path)
    (root / "pipeline.py").write_text(helpers.MAPPED_FILE_PIPELINE)
    helpers.ok(helpers.run_pivot(root, home, "repro"))

    document, stderr = helpers.dry_run_keep_going(root, home, "p/cut@a")
    ran, events = helpers.run_keep_going(root, home, "p/cut@a")

    assert document == {
        "stages": {"p/plan": helpers.UNCHANGED_ROW, "p/cut@a": helpers.UNCHANGED_ROW}
    }, stderr
    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert helpers.stage_statuses(events) == {"p/plan": "cached", "p/cut@a": "cached"}, ran.stdout
    assert [event for event in events if event["type"] == "stages_expanded"] == [
        {"type": "stages_expanded", "template": "p/cut", "stages": ["p/cut@a"]}
    ], ran.stdout
    assert (root / ".pivot" / "stages" / "p" / "cut@b.lock").is_file(), "p/cut@b is kept"


def test_dry_run_keep_going_fails_a_mapped_stage_whose_incremental_output_is_not_cached(
    tmp_path: pathlib.Path,
) -> None:
    """p/cut@a's incremental output is on disk but not in the cache: the run fails p/cut with the
    refusal ``executor.core.incremental_refusal`` gives, runs no instance and exits 1, and the
    dry run shows p/cut as a stage that would fail, with that reason, and no instance."""
    root, home = helpers.uncached_incremental_project(tmp_path, _INCREMENTAL_MAPPED_PIPELINE)

    document, stderr = helpers.dry_run_keep_going(root, home)
    ran, events = helpers.run_keep_going(root, home)

    assert ran.returncode == 1, ran.stdout + ran.stderr
    statuses = helpers.stage_statuses(events)
    assert statuses == {"p/plan": "cached", "p/cut": "failed"}, ran.stdout
    reason = helpers.completed_stages(ran.stdout)["p/cut"]["reason"]
    assert isinstance(reason, str)
    assert reason.startswith(
        "cannot expand over the files of 'p/plan': The following incremental output files "
        + "exist but are not in cache:\n  - p/cut@a: "
    ), reason
    assert "cut@a.txt\n\nRunning the pipeline will DELETE these files" in reason, reason
    assert document == {
        "stages": {"p/plan": helpers.UNCHANGED_ROW, "p/cut": {"would_run": False, "reason": reason}}
    }, stderr
    assert statuses.keys() == document["stages"].keys()


def test_dry_run_keep_going_expands_a_mapped_stage_over_an_uncached_incremental_output_if_allowed(
    tmp_path: pathlib.Path,
) -> None:
    """p/cut@a's incremental output is on disk but not in the cache. With
    --allow-uncached-incremental the run expands p/cut, runs p/cut@a and skips p/cut@b, and the
    dry run lists them so."""
    root, home = helpers.uncached_incremental_project(tmp_path, _INCREMENTAL_MAPPED_PIPELINE)

    document, stderr = helpers.dry_run_keep_going(root, home, "--allow-uncached-incremental")
    ran, events = helpers.run_keep_going(root, home, "--allow-uncached-incremental")

    assert ran.returncode == 0, ran.stdout + ran.stderr
    statuses = helpers.stage_statuses(events)
    assert statuses == {"p/plan": "cached", "p/cut@a": "ran", "p/cut@b": "cached"}, ran.stdout
    assert document == {
        "stages": {
            "p/plan": helpers.UNCHANGED_ROW,
            "p/cut@a": {"would_run": True, "reason": "No previous run"},
            "p/cut@b": helpers.UNCHANGED_ROW,
        }
    }, stderr
    assert statuses.keys() == document["stages"].keys()
