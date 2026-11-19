"""`pivot repro --all --keep-going --dry-run --jsonl` on mapped stages (``Pipeline.map``) beside a
missing tracked file: the dry run lists the stages the run on the same tree runs or skips (spec
R5).

Each test drives the installed ``pivot`` binary in a real git project, as
``tests/cli/test_map_removal.py`` does, and parses the dry run's stdout as one JSON document. The
run's stages are its `stage_complete` events.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
import yaml

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


def _refused_mapped_project(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """p/plan is current, and data/p/cut@a.txt, the output p/cut@a would write, is a missing
    tracked file. It is tracked before p/cut is declared, since `pivot track` refuses a path an
    output the pipeline declares holds."""
    root, home = _git_project(tmp_path)
    (root / "pipeline.py").write_text(helpers.PRODUCER_PIPELINE)
    _ok(helpers.run_pivot(root, home, "repro"))
    tracked = root / "data" / "p" / "cut@a.txt"
    tracked.write_text("tracked bytes")
    _ok(helpers.run_pivot(root, home, "track", "data/p/cut@a.txt"))
    tracked.unlink()
    (root / "pipeline.py").write_text(helpers.MAPPED_FILE_PIPELINE)
    return root, home


def test_dry_run_keep_going_refuses_a_mapped_stage_over_a_missing_tracked_file(
    tmp_path: pathlib.Path,
) -> None:
    """R3 for instances in the dry run: p/cut's instance p/cut@a would write over the missing
    tracked file, so p/cut has one row, not yet expanded, with the reason the run fails it with,
    and no instance is listed."""
    root, home = _refused_mapped_project(tmp_path)

    document, stderr = _dry_run(root, home)
    ran, events = _run(root, home)

    stages = document["stages"]
    assert stages.keys() == {"p/plan", "p/cut"}, document
    assert stages["p/plan"] == {"would_run": False, "reason": "unchanged"}, document
    assert "would_remove" not in document, document
    assert _skipping_lines(stderr) == [], stderr
    assert ran.returncode == 1, ran.stdout + ran.stderr
    completed = {event["stage"]: event for event in events if event["type"] == "stage_complete"}
    assert _statuses(events) == {"p/plan": "cached", "p/cut": "failed"}, ran.stdout
    reason = completed["p/cut"]["reason"]
    assert reason.startswith(
        "cannot expand over the files of 'p/plan': The following tracked files are missing:\n"
    ), reason
    assert "data/p/cut@a.txt: it is output 'data/p/cut@a.txt' of stage 'p/cut@a'" in reason
    assert stages["p/cut"] == {"would_run": True, "reason": f"Not yet expanded: {reason}"}, document


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
