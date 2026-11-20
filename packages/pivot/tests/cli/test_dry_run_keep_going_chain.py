"""`pivot repro --all --keep-going --dry-run --jsonl` on a chain of mapped stages
(``Pipeline.map``): p/cut mapped over p/plan's files, p/build over p/cut's instances and p/final
over p/build's (``_CHAIN_PIPELINE``). The dry run lists the stages the run on the same tree runs,
skips, fails or blocks (spec R5): a mapped stage whose expansion the run refuses, those it blocks
behind it, a producer current only because a pull would bring it up to date, targets naming a
mapped stage or one of its instances, and the limit of a producer that must run first.

Each test drives the installed ``pivot`` binary in a real git project (``helpers.git_project``)
and parses the dry run's stdout as one JSON document. The run's stages are its `stage_complete`
events.
"""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

import pytest

from pivot import registry
from tests import helpers

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable

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

# _CHAIN_PIPELINE with p/cut's output incremental: p/cut@<key> keeps data/p/cut@<key>.txt from its
# last run
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

# helpers.MAPPED_FILE_PIPELINE with p/cut also reading p/shared's output, which every instance
# shares: the run schedules p/shared upstream of p/cut
_SHARED_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, directory, stage


@stage
def plan() -> Annotated[dict[str, str], directory]:
    return {"a.txt": "a", "b.txt": "b"}


@stage
def shared() -> str:
    return "shared"


@stage
def cut(entry: str, shared: str) -> str:
    return entry + shared


with Pipeline("p") as pipeline:
    cut(pipeline.map(plan()), shared())
"""

# _CHAIN_PIPELINE's mapped stages, each standing for its instances while not expanded
_CHAIN_TEMPLATES = ["p/cut", "p/build", "p/final"]

# The instances of _CHAIN_PIPELINE's mapped stages, each over keys a and b
_CHAIN_INSTANCES = [f"{template}@{key}" for template in _CHAIN_TEMPLATES for key in "ab"]

# The not-yet-expanded rows of p/build and p/final, mapped over p/cut's and p/build's instances
_CHAINED_ROWS = {
    chained: {
        "would_run": True,
        "reason": f"Not yet expanded: mapped over the instances of '{over}', which is not "
        + "expanded",
    }
    for chained, over in (("p/build", "p/cut"), ("p/final", "p/build"))
}

_BLOCKED = {"would_run": False, "reason": "upstream 'p/cut' would fail"}

# The start of the reason the run fails p/build with when p/build@a's output is missing
_BUILD_REFUSED = (
    "cannot expand over the files of 'p/plan': The following tracked files are missing:\n"
)
_BUILD_OUTPUT_NAMED = "data/p/build@a.txt: it is output 'data/p/build@a.txt' of stage 'p/build@a'"


def _empty_cache(root: pathlib.Path) -> None:
    """Remove every cache object."""
    for entry in (root / ".pivot" / "cache" / "files").iterdir():
        shutil.rmtree(entry)


def _lose_plan_listing(root: pathlib.Path) -> None:
    """Remove every cache object and p/plan's committed listing, as a clone that never pulled
    holds neither."""
    _empty_cache(root)
    (root / ".pivot" / "stages" / "p" / "plan=.listing").unlink()


def _remove_plan_directory(root: pathlib.Path) -> None:
    shutil.rmtree(root / "data" / "p" / "plan")


def _add_plan_file_whose_key_names_no_stage(root: pathlib.Path) -> None:
    (root / "data" / "p" / "plan" / "c d.txt").write_text("c")


def _remove_plan_file_a(root: pathlib.Path) -> None:
    (root / "data" / "p" / "plan" / "a.txt").unlink()


def _missing_output_project(tmp_path: pathlib.Path, path: str) -> tuple[pathlib.Path, pathlib.Path]:
    """_CHAIN_PIPELINE after a run, with ``path`` (project-relative), the output an instance
    writes, a missing tracked file with no cache object."""
    root, home = helpers.git_project(tmp_path)
    (root / "pipeline.py").write_text(_CHAIN_PIPELINE)
    helpers.ok(helpers.run_pivot(root, home, "repro"))
    tracked = root / path
    tracked.unlink()
    helpers.track_without_data(tracked)
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
    root, home = helpers.refused_mapped_project(tmp_path, _CHAIN_PIPELINE)

    document, stderr = helpers.dry_run_keep_going(root, home)
    text = helpers.ok(helpers.run_pivot(root, home, "repro", "--all", "--keep-going", "--dry-run"))
    explained = helpers.ok(
        helpers.run_pivot(root, home, "repro", "--all", "--keep-going", "--explain")
    )
    ran, events = helpers.run_keep_going(root, home)

    assert ran.returncode == 1, ran.stdout + ran.stderr
    completed = helpers.completed_stages(ran.stdout)
    statuses = helpers.stage_statuses(events)
    assert statuses == {
        "p/plan": "cached",
        "p/cut": "failed",
        "p/build": "blocked",
        "p/final": "blocked",
    }, ran.stdout
    reason = completed["p/cut"]["reason"]
    assert isinstance(reason, str)
    assert reason.startswith(
        "cannot expand over the files of 'p/plan': The following tracked files are missing:\n"
    ), reason
    assert "data/p/cut@a.txt: it is output 'data/p/cut@a.txt' of stage 'p/cut@a'" in reason
    assert document == {
        "stages": {
            "p/plan": helpers.UNCHANGED_ROW,
            "p/cut": {"would_run": False, "reason": reason},
            "p/build": _BLOCKED,
            "p/final": _BLOCKED,
        }
    }, document
    assert document["stages"].keys() == statuses.keys()
    assert helpers.skipping_lines(stderr) == [], stderr
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


def test_dry_run_keep_going_shows_a_mapped_stage_the_run_fails_for_any_refusal(
    tmp_path: pathlib.Path,
) -> None:
    """A refusal over no missing tracked file: p/plan writes a.txt and a.csv, two files of one
    key, so the run fails p/cut and blocks p/build and p/final. The dry run in leave-out mode
    shows p/cut as one row that would not run, with the run's reason, and p/build and p/final as
    blocked by it. Outside leave-out mode (`--all` without `--keep-going`) the dry run shows them
    as 0.10.4 does: mapped stages not yet expanded, which would run."""
    root, home = helpers.git_project(tmp_path)
    (root / "pipeline.py").write_text(
        _CHAIN_PIPELINE.replace('{"a.txt": "a", "b.txt": "b"}', '{"a.txt": "a", "a.csv": "b"}')
    )
    first = helpers.run_pivot(root, home, "repro")
    assert first.returncode == 1, "the first run fails p/cut too"

    document, stderr = helpers.dry_run_keep_going(root, home)
    outside = helpers.run_pivot(root, home, "repro", "--all", "--dry-run", "--jsonl")
    ran, _ = helpers.run_keep_going(root, home)

    assert ran.returncode == 1, ran.stdout + ran.stderr
    completed = helpers.completed_stages(ran.stdout)
    reason = "cannot expand over the files of 'p/plan': files 'a.csv' and 'a.txt' have one key 'a'"
    assert {stage: (event["status"], event["reason"]) for stage, event in completed.items()} == {
        "p/plan": ("cached", "unchanged (generation)"),
        "p/cut": ("failed", reason),
        "p/build": ("blocked", "upstream 'p/cut' failed"),
        "p/final": ("blocked", "upstream 'p/cut' failed"),
    }, ran.stdout
    assert document == {
        "stages": {
            "p/plan": helpers.UNCHANGED_ROW,
            "p/cut": {"would_run": False, "reason": reason},
            "p/build": _BLOCKED,
            "p/final": _BLOCKED,
        }
    }, stderr
    helpers.ok(outside)
    assert json.loads(outside.stdout) == {
        "stages": {
            "p/plan": helpers.UNCHANGED_ROW,
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


@pytest.mark.parametrize(
    ("change", "options", "targets", "chained", "added"),
    [
        pytest.param(
            _remove_plan_directory, [], [], _CHAINED_ROWS, _CHAIN_INSTANCES, id="directory_removed"
        ),
        pytest.param(
            _add_plan_file_whose_key_names_no_stage,
            ["--no-commit"],
            [],
            _CHAINED_ROWS,
            _CHAIN_INSTANCES,
            id="no_commit_file_whose_key_names_no_stage",
        ),
        pytest.param(
            _remove_plan_file_a,
            ["--no-commit"],
            [],
            _CHAINED_ROWS,
            _CHAIN_INSTANCES,
            id="no_commit_file_removed",
        ),
        pytest.param(
            _remove_plan_file_a,
            ["--no-commit"],
            ["p/cut@a"],
            {},
            ["p/cut@a"],
            id="no_commit_target_naming_the_removed_file",
        ),
    ],
)
def test_dry_run_keep_going_leaves_a_mapped_stage_not_expanded_over_a_producer_only_pullable(
    tmp_path: pathlib.Path,
    change: Callable[[pathlib.Path], None],
    options: list[str],
    targets: list[str],
    chained: dict[str, dict[str, object]],
    added: list[str],
) -> None:
    """p/plan's lock matches, but its cache objects and its committed listing are gone, and
    ``change`` changes its directory: removed, or, under --no-commit, holding a file whose key
    cannot name a stage, or without a.txt. A remote is configured, so the dry run counts p/plan
    current, as a pull would bring it up to date, and cannot read its listing. The run never
    pulls: it runs p/plan again, which writes a.txt and b.txt anew, expands the mapped stages
    from what p/plan writes (under --no-commit, from the directory it rewrote), adds the
    instances ``added`` and exits 0. So the dry run shows p/plan as a stage that would run, and
    p/cut, with p/build and p/final when no target names an instance (``chained``), as not yet
    expanded, as 0.10.4 does, each a row for the instances the run adds; none would fail or be
    blocked. It never reads p/plan's directory on disk, which the run replaces: read under
    --no-commit, a key that cannot name a stage made p/cut a stage that would fail, and a
    missing a.txt made the target p/cut@a unknown or left only p/cut@b."""
    root, home = helpers.git_project(tmp_path)
    (root / "pipeline.py").write_text(_CHAIN_PIPELINE)
    helpers.ok(helpers.run_pivot(root, home, "repro"))
    helpers.ok(helpers.run_pivot(root, home, "config", "set", "remotes.origin", "s3://bucket/x"))
    _lose_plan_listing(root)
    change(root)

    document, stderr = helpers.dry_run_keep_going(root, home, *options, *targets)
    ran, events = helpers.run_keep_going(root, home, *options, *targets)

    stages = document["stages"]
    assert stages.keys() == {"p/plan", "p/cut", *chained}, document
    assert stages["p/plan"] == {"would_run": True, "reason": "Outputs missing from cache"}
    cut = stages["p/cut"]["reason"]
    assert cut.startswith("Not yet expanded: the listing of 'p/plan' is missing"), cut
    assert cut.endswith("and `pivot repro --force p/plan` writes it again"), cut
    assert {name: stages[name] for name in chained} == chained, stderr
    assert all(row["would_run"] for row in stages.values()), document
    assert ran.returncode == 0, ran.stdout + ran.stderr
    statuses = helpers.stage_statuses(events)
    assert statuses == {"p/plan": "ran", **dict.fromkeys(added, "cached")}, ran.stdout
    assert {registry.template_of(stage, _CHAIN_TEMPLATES) or stage for stage in statuses} == (
        stages.keys()
    )
    assert sorted(path.name for path in (root / "data" / "p" / "plan").iterdir()) == [
        "a.txt",
        "b.txt",
    ], "the run rewrote p/plan's directory"


@pytest.mark.parametrize(
    ("project", "code", "refused", "named"),
    [
        pytest.param(
            helpers.refused_mapped_project,
            _CHAIN_PIPELINE,
            "The following tracked files are missing:\n",
            "data/p/cut@a.txt: it is output 'data/p/cut@a.txt' of stage 'p/cut@a'",
            id="overlap",
        ),
        pytest.param(
            helpers.uncached_incremental_project,
            _INCREMENTAL_CHAIN_PIPELINE,
            "The following incremental output files exist but are not in cache:\n",
            "  - p/cut@a: ",
            id="uncached-incremental",
        ),
    ],
)
def test_dry_run_keep_going_fails_a_mapped_stage_over_a_producer_only_pullable(
    tmp_path: pathlib.Path,
    project: Callable[[pathlib.Path, str], tuple[pathlib.Path, pathlib.Path]],
    code: str,
    refused: str,
    named: str,
) -> None:
    """p/plan's lock matches, but its directory and the cache objects are gone, and its
    committed listing stays. A remote is configured, so the dry run counts p/plan current, as a
    pull would bring it up to date, and reads p/cut's keys from that listing. The run never
    pulls: it runs p/plan again and plans p/cut over the keys p/plan writes, the ones that
    listing gives. ``project`` builds the tree with ``code``: in `overlap` data/p/cut@a.txt,
    p/cut@a's output, is a missing tracked file, whose cache object is gone with the others; in
    `uncached-incremental` p/cut@a's incremental output is on disk but not in the cache. Either
    way the run fails p/cut with the refusal ``refused``, naming ``named``, blocks p/build and
    p/final and exits 1. Only a listing it cannot read shows p/cut as not yet expanded, so the
    dry run shows p/cut as a stage that would fail, with the run's reason, and p/build and
    p/final as blocked."""
    root, home = project(tmp_path, code)
    helpers.ok(helpers.run_pivot(root, home, "config", "set", "remotes.origin", "s3://bucket/x"))
    _empty_cache(root)
    shutil.rmtree(root / "data" / "p" / "plan")

    document, stderr = helpers.dry_run_keep_going(root, home)
    ran, events = helpers.run_keep_going(root, home)

    assert ran.returncode == 1, ran.stdout + ran.stderr
    statuses = helpers.stage_statuses(events)
    assert statuses == {
        "p/plan": "ran",
        "p/cut": "failed",
        "p/build": "blocked",
        "p/final": "blocked",
    }, ran.stdout
    reason = helpers.completed_stages(ran.stdout)["p/cut"]["reason"]
    assert isinstance(reason, str)
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


def test_dry_run_keep_going_lists_the_instances_a_failing_mapped_target_is_mapped_over(
    tmp_path: pathlib.Path,
) -> None:
    """Target p/build, which is mapped over p/cut's instances, with p/plan current: the run
    expands p/cut and skips p/cut@a and p/cut@b, then fails p/build, whose instance p/build@a
    would write over the missing tracked file data/p/build@a.txt, and exits 1. The dry run lists
    the same stages: p/plan, p/cut's instances, and p/build as a stage that would fail, with
    the run's reason, as the run's execution order for the target holds them."""
    root, home = _missing_output_project(tmp_path, "data/p/build@a.txt")

    document, stderr = helpers.dry_run_keep_going(root, home, "p/build")
    ran, events = helpers.run_keep_going(root, home, "p/build")

    assert ran.returncode == 1, ran.stdout + ran.stderr
    statuses = helpers.stage_statuses(events)
    assert statuses == {
        "p/plan": "cached",
        "p/cut@a": "cached",
        "p/cut@b": "cached",
        "p/build": "failed",
    }, ran.stdout
    reason = helpers.completed_stages(ran.stdout)["p/build"]["reason"]
    assert isinstance(reason, str)
    assert reason.startswith(_BUILD_REFUSED), reason
    assert _BUILD_OUTPUT_NAMED in reason, reason
    assert document == {
        "stages": {
            "p/plan": helpers.UNCHANGED_ROW,
            "p/cut@a": helpers.UNCHANGED_ROW,
            "p/cut@b": helpers.UNCHANGED_ROW,
            "p/build": {"would_run": False, "reason": reason},
        }
    }, stderr
    assert statuses.keys() == document["stages"].keys()


def test_dry_run_keep_going_lists_the_instance_a_blocked_instance_target_is_mapped_over(
    tmp_path: pathlib.Path,
) -> None:
    """Target p/final@a, with p/build@a's output data/p/build@a.txt a missing tracked file, and
    p/plan's lock matching while its directory and the cache objects are gone and a remote is
    configured (a producer only a pull brings up to date), its committed listing kept. The run
    runs p/plan again, expands p/cut over key a alone and skips p/cut@a, since p/plan writes the
    same a.txt again, fails p/build, whose p/build@a would write over the missing file, blocks
    p/final, and exits 1. The dry run lists the same stages: p/cut@a too, the instance of p/cut,
    which expands, that the target is mapped over through p/build. It says p/cut@a would run,
    as it says of every stage downstream of one that would run: what p/plan writes is known
    only once it has run."""
    root, home = _missing_output_project(tmp_path, "data/p/build@a.txt")
    helpers.ok(helpers.run_pivot(root, home, "config", "set", "remotes.origin", "s3://bucket/x"))
    _empty_cache(root)
    shutil.rmtree(root / "data" / "p" / "plan")

    document, stderr = helpers.dry_run_keep_going(root, home, "p/final@a")
    ran, events = helpers.run_keep_going(root, home, "p/final@a")

    assert ran.returncode == 1, ran.stdout + ran.stderr
    statuses = helpers.stage_statuses(events)
    assert statuses == {
        "p/plan": "ran",
        "p/cut@a": "cached",
        "p/build": "failed",
        "p/final": "blocked",
    }, ran.stdout
    reason = helpers.completed_stages(ran.stdout)["p/build"]["reason"]
    assert isinstance(reason, str)
    assert reason.startswith(_BUILD_REFUSED), reason
    assert _BUILD_OUTPUT_NAMED in reason, reason
    assert document == {
        "stages": {
            "p/plan": {"would_run": True, "reason": "Outputs missing from cache"},
            "p/cut@a": {"would_run": True, "reason": "Upstream stale (p/plan)"},
            "p/build": {"would_run": False, "reason": reason},
            "p/final": {"would_run": False, "reason": "upstream 'p/build' would fail"},
        }
    }, stderr
    assert statuses.keys() == document["stages"].keys()


@pytest.mark.parametrize(
    ("targets", "instances"),
    [
        pytest.param(["p/cut"], {"p/cut@a": "cached", "p/cut@b": "ran"}, id="mapped_stage"),
        pytest.param(["p/cut@b"], {"p/cut@b": "ran"}, id="instance"),
        pytest.param(["p/plan", "p/cut@a"], {"p/cut@a": "cached"}, id="producer_and_instance"),
    ],
)
def test_dry_run_keep_going_lists_the_stages_upstream_of_a_mapped_target_not_expanded(
    tmp_path: pathlib.Path, targets: list[str], instances: dict[str, str]
) -> None:
    """Targets ``targets`` name p/cut, or one of its instances, while p/cut is not expanded:
    p/plan's code changed, so b.txt changes. The run schedules p/cut with every stage upstream
    of it: p/plan, which it runs, and p/shared, whose output every instance of p/cut reads,
    which it skips; it then expands p/cut and runs or skips its instances ``instances`` (only
    the named one, for an instance target). The dry run lists the same stages, the run's own
    execution order for the targets: p/plan, p/shared, and p/cut as not yet expanded, a row
    standing for the instances the run adds; not p/plan alone."""
    root, home = helpers.git_project(tmp_path)
    (root / "pipeline.py").write_text(_SHARED_PIPELINE)
    helpers.ok(helpers.run_pivot(root, home, "repro"))
    (root / "pipeline.py").write_text(_SHARED_PIPELINE.replace('"b.txt": "b"', '"b.txt": "B"'))

    document, stderr = helpers.dry_run_keep_going(root, home, *targets)
    ran, events = helpers.run_keep_going(root, home, *targets)

    assert document == {
        "stages": {
            "p/plan": {"would_run": True, "reason": "Code changed"},
            "p/shared": helpers.UNCHANGED_ROW,
            "p/cut": {
                "would_run": True,
                "reason": "Not yet expanded: waits for 'p/plan': Code changed",
            },
        }
    }, stderr
    assert ran.returncode == 0, ran.stdout + ran.stderr
    statuses = helpers.stage_statuses(events)
    assert statuses == {"p/plan": "ran", "p/shared": "cached", **instances}, ran.stdout
    assert {registry.template_of(stage, ["p/cut"]) or stage for stage in statuses} == (
        document["stages"].keys()
    )


def test_dry_run_keep_going_limit_a_mapped_stage_its_producer_must_run_for(
    tmp_path: pathlib.Path,
) -> None:
    """R5's limit, pinned: data/p/cut@a.txt, the output p/cut@a writes, is a missing tracked
    file, and p/plan's code changed, so p/plan is not current. Which files p/plan writes, and so
    which instances p/cut adds and what they write, is known only once p/plan has run, and a dry
    run runs nothing: it shows p/cut, p/build and p/final as not yet expanded, rows that would
    run, and exits 0. The run runs p/plan, then fails p/cut, whose instance p/cut@a would write
    over the missing file, blocks p/build and p/final, and exits 1."""
    root, home = _missing_output_project(tmp_path, "data/p/cut@a.txt")
    (root / "pipeline.py").write_text(_CHAIN_PIPELINE.replace('"b.txt": "b"', '"b.txt": "B"'))

    document, stderr = helpers.dry_run_keep_going(root, home)
    ran, events = helpers.run_keep_going(root, home)

    assert document == {
        "stages": {
            "p/plan": {"would_run": True, "reason": "Code changed"},
            "p/cut": {
                "would_run": True,
                "reason": "Not yet expanded: waits for 'p/plan': Code changed",
            },
            **_CHAINED_ROWS,
        }
    }, stderr
    assert ran.returncode == 1, ran.stdout + ran.stderr
    assert helpers.stage_statuses(events) == {
        "p/plan": "ran",
        "p/cut": "failed",
        "p/build": "blocked",
        "p/final": "blocked",
    }, ran.stdout
    reason = helpers.completed_stages(ran.stdout)["p/cut"]["reason"]
    assert isinstance(reason, str)
    assert "data/p/cut@a.txt: it is output 'data/p/cut@a.txt' of stage 'p/cut@a'" in reason, reason


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
    root, home = helpers.git_project(tmp_path)
    (root / "pipeline.py").write_text(_CHAIN_PIPELINE)
    helpers.ok(helpers.run_pivot(root, home, "repro"))
    added = root / "data" / "p" / "plan" / "c.txt"
    if tree == "listing_lost":
        _lose_plan_listing(root)
    else:
        added.write_text("c")

    document, stderr = helpers.dry_run_keep_going(root, home, "--no-commit")
    ran, events = helpers.run_keep_going(root, home, "--no-commit")

    assert document == {
        "stages": {
            "p/plan": helpers.UNCHANGED_ROW,
            **dict.fromkeys(_CHAIN_INSTANCES, helpers.UNCHANGED_ROW),
        }
    }, stderr
    assert ran.returncode == 0, ran.stdout + ran.stderr
    statuses = helpers.stage_statuses(events)
    assert statuses == dict.fromkeys(["p/plan", *_CHAIN_INSTANCES], "cached"), ran.stdout
    assert statuses.keys() == document["stages"].keys()
    assert not added.exists(), "the skip restores the directory to its lock's record"
