"""The commands on a pipeline mapped over a directory output (``Pipeline.map``), end to end.

Each test drives the installed ``pivot`` binary in a real git project, as a user does. The
producer ``p/plan`` writes one file per key of ``data/raw/plan.yaml``; ``p/cut`` is mapped over
those files and ``p/build`` over ``p/cut``'s instances. A command expands each mapped stage as a
run would: from the listing the producer's lock records, when the producer is current; until then
it shows the mapped stage as not yet expanded.
"""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING

import boto3
import pytest
import yaml

from tests import helpers

if TYPE_CHECKING:
    import pathlib
    import subprocess
    from collections.abc import Mapping

_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, directory, loaders, stage


@stage
def plan(
    entries: dict[str, int],
) -> Annotated[dict[str, dict[str, int]], directory, loaders.YAML[dict[str, int]]()]:
    return {f"{name}.yaml": {"value": value} for name, value in entries.items()}


@stage
def cut(entry: dict[str, int]) -> dict[str, int]:
    return {"cut": entry["value"] * 2}


@stage
def build(cut: dict[str, int]) -> dict[str, int]:
    return {"built": cut["cut"] + 1000}


with Pipeline("p") as pipeline:
    planned = plan(pipeline.input("plan.yaml", t=dict[str, int]))
    build(cut(pipeline.map(planned)))
"""

# The mapped pipeline with an independent stage, p/plain, beside it
_PIPELINE_WITH_PLAIN = (
    _PIPELINE.split('with Pipeline("p")')[0]
    + """@stage
def plain(value: str) -> str:
    return value


with Pipeline("p") as pipeline:
    planned = plan(pipeline.input("plan.yaml", t=dict[str, int]))
    build(cut(pipeline.map(planned)))
    plain(pipeline.input("plain.txt", t=str))
"""
)

# The mapped pipeline with a second mapped stage over p/plan's files, p/pair (declared after
# p/cut, two outputs), and p/explode mapped over p/pair's instances (a directory output)
_PIPELINE_WITH_PAIR = (
    _PIPELINE.split('with Pipeline("p")')[0].replace(
        "from typing import Annotated", "from typing import Annotated, TypedDict"
    )
    + """class Pair(TypedDict):
    left: dict[str, int]
    right: dict[str, int]


@stage
def pair(entry: dict[str, int]) -> Pair:
    return {"left": {"left": entry["value"]}, "right": {"right": entry["value"] + 1}}


@stage
def explode(
    left: dict[str, int],
) -> Annotated[dict[str, dict[str, int]], directory, loaders.YAML[dict[str, int]]()]:
    return {f"{name}.yaml": {"value": value} for name, value in left.items()}


with Pipeline("p") as pipeline:
    planned = plan(pipeline.input("plan.yaml", t=dict[str, int]))
    build(cut(pipeline.map(planned)))
    explode(pair(pipeline.map(planned))["left"])
"""
)

_INSTANCES = ["p/cut@a", "p/cut@b", "p/cut@c"]
_CHAINS = sorted([*_INSTANCES, "p/build@a", "p/build@b", "p/build@c"])


def _ok(proc: subprocess.CompletedProcess[str]) -> str:
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    return out


def _stdout(proc: subprocess.CompletedProcess[str]) -> str:
    """A command's stdout, once it succeeded: a ``--json`` document; notes go to stderr."""
    _ok(proc)
    return proc.stdout


def _project(tmp_path: pathlib.Path, plan: Mapping[str, int]) -> tuple[pathlib.Path, pathlib.Path]:
    """A git project holding the mapped pipeline, and the HOME its commands run with."""
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
    (root / "pipeline.py").write_text(_PIPELINE)
    _write_plan(root, plan)
    return root, home


def _project_with_plain(
    tmp_path: pathlib.Path, plan: Mapping[str, int]
) -> tuple[pathlib.Path, pathlib.Path]:
    """A mapped project with an independent plain stage."""
    root, home = _project(tmp_path, plan)
    (root / "pipeline.py").write_text(_PIPELINE_WITH_PLAIN)
    (root / "data" / "raw" / "plain.txt").write_text("plain\n")
    return root, home


def _write_plan(root: pathlib.Path, plan: Mapping[str, int]) -> None:
    (root / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (root / "data" / "raw" / "plan.yaml").write_text(yaml.safe_dump(dict(plan)))


def _status_rows(root: pathlib.Path, home: pathlib.Path, *targets: str) -> dict[str, str]:
    """Each row of ``pivot status --json``: name -> status."""
    ran = helpers.run_pivot(root, home, "status", "--stages-only", "--json", *targets)
    rows = json.loads(_stdout(ran))["stages"]
    return {row["name"]: row["status"] for row in rows}


def _read(root: pathlib.Path, stage: str) -> object:
    return yaml.safe_load((root / "data" / "p" / f"{stage}.yaml").read_text())


# =============================================================================
# Before the producer has run
# =============================================================================


def test_commands_show_mapped_stages_not_yet_expanded_before_the_producer_ran(
    tmp_path: pathlib.Path,
) -> None:
    root, home = _project(tmp_path, {"a": 1, "b": 2})

    status = _ok(helpers.run_pivot(root, home, "status", "--stages-only"))
    rows = _status_rows(root, home)
    listed = json.loads(_stdout(helpers.run_pivot(root, home, "list", "--json")))
    listed_text = _ok(helpers.run_pivot(root, home, "list"))
    dag = _ok(helpers.run_pivot(root, home, "dag", "--stages"))
    explain = _ok(helpers.run_pivot(root, home, "status", "--explain", "--stages-only"))
    dry_run = _ok(helpers.run_pivot(root, home, "repro", "--dry-run"))

    assert rows == {"p/plan": "stale", "p/cut": "not_expanded", "p/build": "not_expanded"}, rows
    assert "2 mapped stages not yet expanded" in status, status
    assert "Not yet expanded: waits for 'p/plan': No previous run" in status, status
    assert [stage["name"] for stage in listed["stages"]] == ["p/plan"], listed
    assert [(t["name"], t["expanded"], t["instances"]) for t in listed["templates"]] == [
        ("p/cut", False, []),
        ("p/build", False, []),
    ], listed
    assert listed["templates"][0]["source"] == "p/plan", listed
    assert "Mapped stages, not yet expanded (2):" in listed_text, listed_text
    assert "p/cut (not yet expanded)" in dag, dag
    assert "NOT YET EXPANDED" in explain and "1 will run, 0 unchanged, 2 not yet" in explain, (
        explain
    )
    assert "p/cut: would expand (Not yet expanded: waits for 'p/plan'" in dry_run, dry_run


# =============================================================================
# Runs and targets
# =============================================================================


def test_after_a_run_the_commands_list_each_instance(tmp_path: pathlib.Path) -> None:
    root, home = _project(tmp_path, {"a": 1, "b": 2, "c": 3})

    repro = _ok(helpers.run_pivot(root, home, "repro"))
    rows = _status_rows(root, home)
    listed = json.loads(_stdout(helpers.run_pivot(root, home, "list", "--json")))
    dag = _ok(helpers.run_pivot(root, home, "dag", "--stages"))
    explain = _ok(helpers.run_pivot(root, home, "status", "--explain", "--stages-only"))

    assert "p/cut: expanded into 3 stages" in repro or "expanded" in repro, repro
    assert _read(root, "build@c") == {"built": 1006}
    assert rows == dict.fromkeys(["p/plan", *_CHAINS], "cached"), rows
    assert sorted(stage["name"] for stage in listed["stages"]) == sorted(["p/plan", *_CHAINS])
    assert [(t["name"], t["expanded"], t["instances"]) for t in listed["templates"]] == [
        ("p/cut", True, _INSTANCES),
        ("p/build", True, ["p/build@a", "p/build@b", "p/build@c"]),
    ], listed
    assert "p/cut@b" in dag and "not yet expanded" not in dag, dag
    assert "Stage: p/build@a" in explain and "NOT YET EXPANDED" not in explain, explain


def test_an_edit_to_one_key_reruns_only_its_chain(tmp_path: pathlib.Path) -> None:
    root, home = _project(tmp_path, {"a": 1, "b": 2, "c": 3})
    _ok(helpers.run_pivot(root, home, "repro"))
    _write_plan(root, {"a": 1, "b": 20, "c": 3})

    before = _status_rows(root, home)
    ran = json.loads(
        _stdout(helpers.run_pivot(root, home, "repro", "--json")).strip().splitlines()[-1]
    )

    assert before == {"p/plan": "stale", "p/cut": "not_expanded", "p/build": "not_expanded"}
    assert sorted(ran["failed_stages"]) == [], ran
    assert ran["ran"] == 3, ran  # p/plan, p/cut@b, p/build@b
    assert _read(root, "build@b") == {"built": 1040}
    assert _status_rows(root, home) == dict.fromkeys(["p/plan", *_CHAINS], "cached")


def test_repro_and_status_take_a_mapped_stage_or_an_instance_as_target(
    tmp_path: pathlib.Path,
) -> None:
    root, home = _project(tmp_path, {"a": 1, "b": 2})

    _ok(helpers.run_pivot(root, home, "repro", "p/cut"))

    assert (root / "data" / "p" / "cut@a.yaml").is_file()
    assert not (root / "data" / "p" / "build@a.yaml").exists(), "p/build is not upstream of p/cut"
    assert _status_rows(root, home, "p/cut@b") == {"p/plan": "cached", "p/cut@b": "cached"}
    assert _status_rows(root, home, "cut") == {
        "p/plan": "cached",
        "p/cut@a": "cached",
        "p/cut@b": "cached",
    }
    bare_instance = _ok(helpers.run_pivot(root, home, "repro", "build@a"))
    assert (root / "data" / "p" / "build@a.yaml").is_file(), bare_instance
    assert not (root / "data" / "p" / "build@b.yaml").exists(), "an instance runs its chain only"
    unknown = helpers.run_pivot(root, home, "status", "p/cut@zzz")
    assert unknown.returncode != 0 and "p/cut@zzz" in unknown.stderr, unknown.stderr


def test_repro_takes_an_instance_output_path_as_pivot_dag_does(tmp_path: pathlib.Path) -> None:
    """A path target resolves through the expansion: the instance writing it runs with its chain
    alone. Before the producer is current no instance writes it, and the refusal names the
    mapped stages not yet expanded."""
    root, home = _project(tmp_path, {"a": 1, "b": 2})

    waiting = helpers.run_pivot(root, home, "repro", "--dry-run", "data/p/cut@a.yaml")
    _ok(helpers.run_pivot(root, home, "repro", "p/plan"))
    dag = _ok(helpers.run_pivot(root, home, "dag", "--stages", "data/p/cut@a.yaml"))
    dry_run = _ok(helpers.run_pivot(root, home, "repro", "--dry-run", "data/p/cut@a.yaml"))
    _ok(helpers.run_pivot(root, home, "repro", "data/p/build@b.yaml"))

    assert waiting.returncode != 0, waiting.stdout
    assert "Unknown target(s): 'data/p/cut@a.yaml'" in waiting.stderr, waiting.stderr
    assert "not yet expanded: p/cut, p/build" in waiting.stderr, waiting.stderr
    assert "p/cut@a" in dag, dag
    assert "p/cut@a" in dry_run and "p/cut@b" not in dry_run, dry_run
    assert (root / "data" / "p" / "build@b.yaml").is_file()
    assert not (root / "data" / "p" / "cut@a.yaml").exists(), "the path's chain runs alone"


@pytest.mark.parametrize(
    "command",
    [
        pytest.param(("repro", "--dry-run"), id="repro"),
        pytest.param(("dag", "--stages"), id="dag"),
    ],
)
def test_the_output_path_a_mapped_stage_declares_is_refused(
    tmp_path: pathlib.Path, command: tuple[str, ...]
) -> None:
    """No stage writes `data/p/cut.yaml`: each instance of `p/cut` writes its own. Before the
    producer runs and after, `repro` and `dag` refuse the target alike, naming the mapped stage
    and its instances' path."""
    root, home = _project(tmp_path, {"a": 1})

    before = helpers.run_pivot(root, home, *command, "data/p/cut.yaml")
    _ok(helpers.run_pivot(root, home, "repro", "p/plan"))
    after = helpers.run_pivot(root, home, *command, "data/p/cut.yaml")

    for refused in (before, after):
        assert refused.returncode != 0, refused.stdout
        message = " ".join(refused.stderr.split())
        assert (
            "'data/p/cut.yaml' is the output path of mapped stage 'p/cut', which no stage "
            + "writes: each instance writes its own (data/p/cut@<key>.yaml)"
        ) in message, refused.stderr


def test_a_run_targeted_at_the_producer_expands_nothing(tmp_path: pathlib.Path) -> None:
    root, home = _project(tmp_path, {"a": 1, "b": 2})

    _ok(helpers.run_pivot(root, home, "repro", "p/plan"))

    assert (root / "data" / "p" / "plan" / "a.yaml").is_file()
    assert not (root / "data" / "p" / "cut@a.yaml").exists()
    # The producer is current now, so the commands expand the mapped stages from its listing
    assert _status_rows(root, home)["p/cut@a"] == "stale"


def test_commit_reports_what_it_committed_before_a_refused_instance(
    tmp_path: pathlib.Path,
) -> None:
    """`pivot commit` commits the producer before it can know the keys, so an instance whose
    key the producer has no file for is refused after it: the output names both."""
    root, home = _project(tmp_path, {"a": 1})
    _ok(helpers.run_pivot(root, home, "repro", "--no-commit"))

    refused = helpers.run_pivot(root, home, "commit", "p/plan", "p/cut@zzz")

    out = refused.stdout + refused.stderr
    assert refused.returncode != 0, out
    assert "Committed 1 stage(s):\n  p/plan" in out
    assert "Unknown stage(s): p/cut@zzz" in out
    assert _status_rows(root, home)["p/plan"] == "cached"


def test_commit_of_a_mapped_stage_commits_its_instances(tmp_path: pathlib.Path) -> None:
    root, home = _project(tmp_path, {"a": 1, "b": 2})
    _ok(helpers.run_pivot(root, home, "repro", "--no-commit"))

    committed = _ok(helpers.run_pivot(root, home, "commit", "plan", "cut"))

    assert "Committed 3 stage(s)" in committed, committed
    rows = _status_rows(root, home)
    assert rows["p/cut@a"] == rows["p/cut@b"] == "cached", rows
    assert rows["p/build@a"] == "stale", rows


def test_params_show_takes_a_mapped_stage_once_it_expands(tmp_path: pathlib.Path) -> None:
    root, home = _project(tmp_path, {"a": 1, "b": 2})

    waiting = helpers.run_pivot(root, home, "params", "show", "p/cut")
    _ok(helpers.run_pivot(root, home, "repro"))
    shown = helpers.run_pivot(root, home, "params", "show", "--json", "cut")

    assert waiting.returncode != 0
    assert "'p/cut' is not yet expanded: waits for 'p/plan'" in waiting.stderr, waiting.stderr
    assert json.loads(_stdout(shown)) == {}, "p/cut declares no params"


_PARAMS_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, StageParams, directory, loaders, stage


class CutParams(StageParams):
    factor: int = 2


@stage
def plan(
    entries: dict[str, int],
) -> Annotated[dict[str, dict[str, int]], directory, loaders.YAML[dict[str, int]]()]:
    return {f"{name}.yaml": {"value": value} for name, value in entries.items()}


@stage
def cut(entry: dict[str, int], params: CutParams) -> dict[str, int]:
    return {"cut": entry["value"] * params.factor}


@stage
def build(cut: dict[str, int]) -> dict[str, int]:
    return {"built": cut["cut"] + 1000}


with Pipeline("p") as pipeline:
    planned = plan(pipeline.input("plan.yaml", t=dict[str, int]))
    build(cut(pipeline.map(planned), params=CutParams()))
"""


def test_params_without_targets_name_the_mapped_stages_not_yet_expanded(
    tmp_path: pathlib.Path,
) -> None:
    """Before the producer has run, the parameters of `p/cut`'s instances cannot be read: `params
    show` and `params diff` say so in place of finding no parameters, beside `--json` on stderr.
    `p/build` takes no parameters, so it hides none."""
    root, home = _project(tmp_path, {"a": 1, "b": 2})
    (root / "pipeline.py").write_text(_PARAMS_PIPELINE)
    note = "Mapped stages, not yet expanded (1):\n  p/cut\n    waits for 'p/plan'"

    shown = _ok(helpers.run_pivot(root, home, "params", "show"))
    as_json = helpers.run_pivot(root, home, "params", "show", "--json")
    diffed = _ok(helpers.run_pivot(root, home, "params", "diff"))
    _ok(helpers.run_pivot(root, home, "repro"))
    expanded = _ok(helpers.run_pivot(root, home, "params", "show"))

    for output in (shown, diffed):
        assert note in output, output
        assert "No parameter" not in output, output
        assert "p/build" not in output, output
    assert json.loads(_stdout(as_json)) == {}, as_json.stdout
    assert note in as_json.stderr, as_json.stderr
    assert "p/cut@a" in expanded and "p/cut@b" in expanded, expanded
    assert "not yet expanded" not in expanded, expanded


def test_a_key_found_missing_after_the_producer_ran_prints_after_the_producer(
    tmp_path: pathlib.Path,
) -> None:
    """The run fails `p/cut` only once `p/plan` has run and written no file for the key, so the
    report prints `p/plan`'s line first."""
    root, home = _project(tmp_path, {"a": 1})

    failed = helpers.run_pivot(root, home, "repro", "p/cut@zzz")

    assert failed.returncode == 1, failed.stdout + failed.stderr
    lines = failed.stdout.splitlines()
    plan = [number for number, line in enumerate(lines) if "p/plan" in line and "done" in line]
    cut = [number for number, line in enumerate(lines) if "p/cut" in line and "FAILED" in line]
    assert len(plan) == 1 and len(cut) == 1, failed.stdout
    assert plan[0] < cut[0], failed.stdout


# =============================================================================
# Outputs: checkout, verify, push and pull
# =============================================================================


def test_checkout_restores_the_instances_and_refuses_a_mapped_stage_not_expanded(
    tmp_path: pathlib.Path,
) -> None:
    root, home = _project(tmp_path, {"a": 1, "b": 2})
    _ok(helpers.run_pivot(root, home, "repro"))
    (root / "data" / "p" / "cut@a.yaml").unlink()
    (root / "data" / "p" / "build@b.yaml").unlink()

    _ok(helpers.run_pivot(root, home, "checkout", "p/cut"))
    assert _read(root, "cut@a") == {"cut": 2}
    assert not (root / "data" / "p" / "build@b.yaml").exists(), "only p/cut's instances"
    _ok(helpers.run_pivot(root, home, "checkout"))
    assert _read(root, "build@b") == {"built": 1004}

    _write_plan(root, {"a": 1, "b": 3})
    refused = helpers.run_pivot(root, home, "checkout", "p/build")
    assert refused.returncode != 0, refused.stdout + refused.stderr
    assert (
        "'p/build' is not yet expanded: mapped over the instances of 'p/cut', which is not "
        + "expanded"
    ) in refused.stderr, refused.stderr
    refused_cut = helpers.run_pivot(root, home, "checkout", "p/cut")
    assert "'p/cut' is not yet expanded: waits for 'p/plan'" in refused_cut.stderr


def test_verify_fails_a_mapped_stage_not_expanded(tmp_path: pathlib.Path) -> None:
    root, home = _project(tmp_path, {"a": 1})
    _ok(helpers.run_pivot(root, home, "repro"))
    passed = _ok(helpers.run_pivot(root, home, "verify"))
    assert "p/build@a: passed" in passed, passed

    _write_plan(root, {"a": 5})
    failed = helpers.run_pivot(root, home, "verify")

    assert failed.returncode == 1, failed.stdout + failed.stderr
    assert "p/cut: failed" in failed.stdout, failed.stdout
    assert "Not yet expanded: waits for 'p/plan'" in failed.stdout, failed.stdout


def _s3_remote(root: pathlib.Path, home: pathlib.Path, endpoint: str) -> dict[str, str]:
    """Make a moto bucket the project's default remote, and return the env that reaches it."""
    env = {
        "AWS_ENDPOINT_URL": endpoint,
        "AWS_ACCESS_KEY_ID": "x",
        "AWS_SECRET_ACCESS_KEY": "x",
        "AWS_DEFAULT_REGION": "us-east-1",
    }
    bucket = f"map-{uuid.uuid4().hex[:8]}"
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name="us-east-1",
        aws_access_key_id="x",
        aws_secret_access_key="x",
    )
    s3.create_bucket(Bucket=bucket)  # pyright: ignore[reportAttributeAccessIssue] - boto3 ships no stubs
    _ok(helpers.run_pivot(root, home, "config", "set", "remotes.origin", f"s3://{bucket}/cache"))
    _ok(helpers.run_pivot(root, home, "config", "set", "default_remote", "origin"))
    return env


def _clone(root: pathlib.Path, home: pathlib.Path, dest: pathlib.Path) -> pathlib.Path:
    """Commit the project as a user does (locks and listings, not outputs) and clone it."""
    (root / ".gitignore").write_text("/data/p/\n")
    _ok(helpers.run_cli(root, home, "git", "add", "-A"))
    _ok(helpers.run_cli(root, home, "git", "commit", "-qm", "base"))
    _ok(helpers.run_cli(root.parent, home, "git", "clone", "-q", str(root), str(dest)))
    return dest


@pytest.mark.usefixtures("moto_services")
def test_a_clone_expands_from_the_committed_listing_and_pull_restores_the_instances(
    tmp_path: pathlib.Path, moto_services: dict[str, str]
) -> None:
    root, home = _project(tmp_path, {"a": 1, "b": 2})
    env = _s3_remote(root, home, moto_services["s3"])
    _ok(helpers.run_pivot(root, home, "repro", extra_env=env))
    _ok(helpers.run_pivot(root, home, "push", extra_env=env))
    clone = _clone(root, home, tmp_path / "clone")

    rows = json.loads(
        _stdout(helpers.run_pivot(clone, home, "status", "--stages-only", "--json", extra_env=env))
    )["stages"]
    pulled_target = _ok(helpers.run_pivot(clone, home, "pull", "p/build", extra_env=env))

    # The clone holds no output and no cache: the keys come from the committed listing
    assert {row["name"]: row["reason"] for row in rows}["p/cut@a"] == "Outputs missing from cache"
    assert _read(clone, "build@a") == {"built": 1002}, pulled_target
    assert _read(clone, "build@b") == {"built": 1004}, pulled_target
    _ok(helpers.run_pivot(clone, home, "pull", extra_env=env))
    assert _read(clone, "cut@b") == {"cut": 4}
    assert "Verification passed" in _ok(helpers.run_pivot(clone, home, "verify", extra_env=env))


@pytest.mark.usefixtures("moto_services")
def test_pull_fetches_a_listing_the_clone_lacks_before_it_expands(
    tmp_path: pathlib.Path, moto_services: dict[str, str]
) -> None:
    root, home = _project(tmp_path, {"a": 1, "b": 2})
    env = _s3_remote(root, home, moto_services["s3"])
    _ok(helpers.run_pivot(root, home, "repro", extra_env=env))
    _ok(helpers.run_pivot(root, home, "push", "p/plan", "p/cut", extra_env=env))
    clone = _clone(root, home, tmp_path / "clone")
    stages_dir = clone / ".pivot" / "stages" / "p"
    for listing in stages_dir.glob("plan=*.listing"):
        listing.unlink()  # as if the listing was never committed

    status = _ok(helpers.run_pivot(clone, home, "status", "--stages-only", extra_env=env))
    pulled = _ok(helpers.run_pivot(clone, home, "pull", "p/cut", extra_env=env))

    assert "the listing of 'p/plan' is missing" in status, status
    assert _read(clone, "cut@a") == {"cut": 2}, pulled
    assert _read(clone, "cut@b") == {"cut": 4}, pulled


@pytest.mark.usefixtures("moto_services")
def test_push_of_a_mapped_stage_sends_the_listing_a_clone_needs_to_expand_it(
    tmp_path: pathlib.Path, moto_services: dict[str, str]
) -> None:
    """`pivot pull p/cut` in a clone without the producer's committed listing fetches that
    listing to expand p/cut, so `pivot push p/cut` sends it beside the instances' outputs."""
    root, home = _project(tmp_path, {"a": 1, "b": 2})
    env = _s3_remote(root, home, moto_services["s3"])
    _ok(helpers.run_pivot(root, home, "repro", extra_env=env))
    _ok(helpers.run_pivot(root, home, "push", "p/cut", extra_env=env))
    clone = _clone(root, home, tmp_path / "clone")
    for listing in (clone / ".pivot" / "stages" / "p").glob("plan=*.listing"):
        listing.unlink()  # as if the listing was never committed

    pulled = _ok(helpers.run_pivot(clone, home, "pull", "p/cut", extra_env=env))

    assert _read(clone, "cut@a") == {"cut": 2}, pulled
    assert _read(clone, "cut@b") == {"cut": 4}, pulled


def _cache_objects(root: pathlib.Path) -> list[pathlib.Path]:
    """Every content object in the project's cache, not its lock files."""
    files = root / ".pivot" / "cache" / "files"
    if not files.is_dir():
        return []
    return sorted(p for p in files.rglob("*") if p.is_file() and not p.name.startswith("."))


@pytest.mark.usefixtures("moto_services")
def test_pull_dry_run_of_a_mapped_stage_counts_what_the_pull_fetches(
    tmp_path: pathlib.Path, moto_services: dict[str, str]
) -> None:
    """In a clone without the producer's committed listing, `pivot pull --dry-run p/cut` reads
    that listing from the remote into a temporary directory to expand p/cut, leaves the cache
    as it was, and counts what the pull fetches: the listing and the instances' outputs."""
    root, home = _project(tmp_path, {"a": 1, "b": 2})
    env = _s3_remote(root, home, moto_services["s3"])
    _ok(helpers.run_pivot(root, home, "repro", extra_env=env))
    _ok(helpers.run_pivot(root, home, "push", "p/cut", extra_env=env))
    clone = _clone(root, home, tmp_path / "clone")
    for listing in (clone / ".pivot" / "stages" / "p").glob("plan=*.listing"):
        listing.unlink()  # as if the listing was never committed

    dry_run = _ok(helpers.run_pivot(clone, home, "pull", "--dry-run", "p/cut", extra_env=env))
    after_dry_run = _cache_objects(clone)
    pulled = _ok(helpers.run_pivot(clone, home, "pull", "p/cut", extra_env=env))

    assert "Would pull 3 file(s)" in dry_run, dry_run
    assert after_dry_run == [], "the dry run wrote into the cache"
    assert "3 transferred" in pulled, pulled
    assert len(_cache_objects(clone)) == 3, pulled


@pytest.mark.usefixtures("moto_services")
def test_fetch_of_a_plain_stage_fetches_no_listing_of_an_unrelated_mapped_stage(
    tmp_path: pathlib.Path, moto_services: dict[str, str]
) -> None:
    """`pivot fetch p/plain` transfers p/plain's output alone, in a clone that lacks the
    listing an unrelated mapped stage expands over, and its summary counts what it moved."""
    root, home = _project_with_plain(tmp_path, {"a": 1, "b": 2})
    env = _s3_remote(root, home, moto_services["s3"])
    _ok(helpers.run_pivot(root, home, "repro", extra_env=env))
    _ok(helpers.run_pivot(root, home, "push", extra_env=env))
    clone = _clone(root, home, tmp_path / "clone")
    for listing in (clone / ".pivot" / "stages" / "p").glob("plan=*.listing"):
        listing.unlink()  # as if the listing was never committed

    dry_run = _ok(helpers.run_pivot(clone, home, "fetch", "--dry-run", "p/plain", extra_env=env))
    fetched = _ok(helpers.run_pivot(clone, home, "fetch", "p/plain", extra_env=env))

    assert "Would fetch 1 file(s)" in dry_run, dry_run
    assert "1 transferred" in fetched, fetched
    assert len(_cache_objects(clone)) == 1, _cache_objects(clone)


@pytest.mark.usefixtures("moto_services")
def test_pull_of_a_mapped_stage_fails_naming_a_listing_the_remote_lacks(
    tmp_path: pathlib.Path, moto_services: dict[str, str]
) -> None:
    """The listing p/cut expands over is neither committed nor on the remote: `pivot pull p/cut`
    and its dry run exit 1, naming the listing and the mapped stage, and restore nothing."""
    root, home = _project_with_plain(tmp_path, {"a": 1, "b": 2})
    env = _s3_remote(root, home, moto_services["s3"])
    _ok(helpers.run_pivot(root, home, "repro", extra_env=env))
    _ok(helpers.run_pivot(root, home, "push", "p/plain", extra_env=env))
    clone = _clone(root, home, tmp_path / "clone")
    for listing in (clone / ".pivot" / "stages" / "p").glob("plan=*.listing"):
        listing.unlink()  # as if the listing was never committed

    for args in (("pull", "--dry-run", "p/cut"), ("pull", "p/cut")):
        proc = helpers.run_pivot(clone, home, *args, extra_env=env)
        out = proc.stdout + proc.stderr
        assert proc.returncode == 1, out
        assert "listing" in out and "p/cut" in out and "p/plan" in out, out
    assert _cache_objects(clone) == []
    assert not (clone / "data" / "p" / "cut@a.yaml").exists()


@pytest.mark.usefixtures("moto_services")
@pytest.mark.parametrize(
    ("args", "line"),
    [
        (("pull",), "2 transferred"),
        (("pull", "--dry-run"), "Would pull 2 file(s)"),
        (("fetch",), "2 transferred"),
        (("fetch", "--dry-run"), "Would fetch 2 file(s)"),
    ],
)
def test_sync_of_an_instance_output_path_fetches_the_listing_it_expands_over(
    tmp_path: pathlib.Path, moto_services: dict[str, str], args: tuple[str, ...], line: str
) -> None:
    """`data/p/cut@a.yaml` is p/cut@a's output. In a clone without the producer's committed
    listing, pull, fetch and their dry runs of that path get the listing p/cut expands over,
    then the output, as they do for `p/cut@a`: two files."""
    root, home = _project(tmp_path, {"a": 1, "b": 2})
    env = _s3_remote(root, home, moto_services["s3"])
    _ok(helpers.run_pivot(root, home, "repro", extra_env=env))
    _ok(helpers.run_pivot(root, home, "push", extra_env=env))
    clone = _clone(root, home, tmp_path / "clone")
    for listing in (clone / ".pivot" / "stages" / "p").glob("plan=*.listing"):
        listing.unlink()  # as if the listing was never committed

    out = _ok(helpers.run_pivot(clone, home, *args, "data/p/cut@a.yaml", extra_env=env))

    assert line in out, out
    if args == ("pull",):
        assert _read(clone, "cut@a") == {"cut": 2}, out


def _cache_bytes(root: pathlib.Path) -> dict[str, bytes]:
    """Every file under the project's cache, lock files included, by its path there."""
    cache = root / ".pivot" / "cache"
    if not cache.is_dir():
        return {}
    return {str(p.relative_to(cache)): p.read_bytes() for p in cache.rglob("*") if p.is_file()}


@pytest.mark.usefixtures("moto_services")
@pytest.mark.parametrize(
    "args",
    [
        ("pull", "--dry-run", "p/cut"),
        ("fetch", "--dry-run", "p/cut"),
        ("push", "--dry-run", "p/cut"),
        ("pull", "--dry-run", "p/plan"),
        ("push", "--dry-run", "p/plan"),
    ],
)
def test_a_dry_run_reads_a_committed_listing_without_caching_it(
    tmp_path: pathlib.Path, moto_services: dict[str, str], args: tuple[str, ...]
) -> None:
    """A clone holds p/plan's committed listing and nothing in its cache. A dry run of the
    mapped stage, or of the directory output it is mapped over, reads that listing without
    storing it as a cache object: the cache is byte for byte as it was."""
    root, home = _project(tmp_path, {"a": 1, "b": 2})
    env = _s3_remote(root, home, moto_services["s3"])
    _ok(helpers.run_pivot(root, home, "repro", extra_env=env))
    _ok(helpers.run_pivot(root, home, "push", extra_env=env))
    clone = _clone(root, home, tmp_path / "clone")
    before = _cache_bytes(clone)

    _ok(helpers.run_pivot(clone, home, *args, extra_env=env))

    assert _cache_bytes(clone) == before


@pytest.mark.usefixtures("moto_services")
@pytest.mark.parametrize("dry_run", [False, True], ids=["pull", "dry-run"])
@pytest.mark.parametrize(
    ("target", "count", "restored", "content"),
    [
        ("p/pair@a", 3, "pair@a/left.yaml", {"left": 1}),
        ("p/pair", 5, "pair@b/right.yaml", {"right": 3}),
        # p/plan's listing and left.yaml; p/explode@a's own listing is committed, so the pull
        # stores it from that file rather than fetching it
        ("p/explode@a", 2, "explode@a/left.yaml", {"value": 1}),
        ("data/p/pair@a/left.yaml", 2, "pair@a/left.yaml", {"left": 1}),
        ("data/p/explode@a", 2, "explode@a/left.yaml", {"value": 1}),
    ],
)
def test_pull_of_a_second_mapped_stage_over_one_producer_fetches_the_listing(
    tmp_path: pathlib.Path,
    moto_services: dict[str, str],
    dry_run: bool,
    target: str,
    count: int,
    restored: str,
    content: dict[str, int],
) -> None:
    """p/pair is mapped over p/plan's files after p/cut, and p/explode over p/pair's instances.
    In a clone without p/plan's committed listing, a pull of either, by stage, instance or
    output path, fetches that listing first, then the target's outputs; its dry run counts the
    same files and restores nothing."""
    root, home = _project(tmp_path, {"a": 1, "b": 2})
    (root / "pipeline.py").write_text(_PIPELINE_WITH_PAIR)
    env = _s3_remote(root, home, moto_services["s3"])
    _ok(helpers.run_pivot(root, home, "repro", extra_env=env))
    _ok(helpers.run_pivot(root, home, "push", extra_env=env))
    clone = _clone(root, home, tmp_path / "clone")
    for listing in (clone / ".pivot" / "stages" / "p").glob("plan=*.listing"):
        listing.unlink()  # as if the listing was never committed
    args = ("pull", "--dry-run", target) if dry_run else ("pull", target)

    out = _ok(helpers.run_pivot(clone, home, *args, extra_env=env))

    output = clone / "data" / "p" / restored
    if dry_run:
        assert f"Would pull {count} file(s)" in out, out
        assert not output.exists(), out
    else:
        assert f"{count} transferred" in out, out
        assert yaml.safe_load(output.read_text()) == content, out


def _clone_without_listing(
    tmp_path: pathlib.Path, endpoint: str
) -> tuple[pathlib.Path, pathlib.Path, dict[str, str]]:
    """A clone of the run, pushed mapped project (keys a and b) without p/plan's committed
    listing; returns the clone, the HOME and the env that reaches the remote."""
    root, home = _project(tmp_path, {"a": 1, "b": 2})
    env = _s3_remote(root, home, endpoint)
    _ok(helpers.run_pivot(root, home, "repro", extra_env=env))
    _ok(helpers.run_pivot(root, home, "push", extra_env=env))
    clone = _clone(root, home, tmp_path / "clone")
    for listing in (clone / ".pivot" / "stages" / "p").glob("plan=*.listing"):
        listing.unlink()  # as if the listing was never committed
    return clone, home, env


@pytest.mark.usefixtures("moto_services")
@pytest.mark.parametrize(
    "args",
    [
        ("pull", "p/cut@zz"),
        ("pull", "data/p/cut@zz.yaml"),
        ("pull", "--dry-run", "p/cut@zz"),
        ("fetch", "data/p/cut@zz.yaml"),
    ],
)
def test_a_key_the_mapped_stage_lacks_is_refused_naming_the_instance(
    tmp_path: pathlib.Path, moto_services: dict[str, str], args: tuple[str, ...]
) -> None:
    """p/cut has keys a and b. A target naming key zz, by instance name or output path, reads
    the listing it expands from, then fails saying p/cut has no such instance, and transfers
    nothing."""
    clone, home, env = _clone_without_listing(tmp_path, moto_services["s3"])

    proc = helpers.run_pivot(clone, home, *args, extra_env=env)
    out = proc.stdout + proc.stderr

    assert proc.returncode == 1, out
    assert "mapped stage 'p/cut' has no instance 'p/cut@zz'" in out, out
    assert "transferred" not in out and "Would" not in out, out


@pytest.mark.usefixtures("moto_services")
@pytest.mark.parametrize("target", ["data/p/cut@a.yaml", "p/cut@a"])
def test_push_of_an_instance_without_its_listing_names_the_listing(
    tmp_path: pathlib.Path, moto_services: dict[str, str], target: str
) -> None:
    """Push fetches nothing, so without p/plan's committed listing or its cache object p/cut
    cannot expand: a push of an instance, by name or output path, fails naming that listing."""
    clone, home, env = _clone_without_listing(tmp_path, moto_services["s3"])

    proc = helpers.run_pivot(clone, home, "push", target, extra_env=env)
    out = proc.stdout + proc.stderr

    assert proc.returncode == 1, out
    assert "'p/cut' is not yet expanded: the listing of 'p/plan' is missing" in out, out


def test_a_mapped_stage_or_an_instance_is_refused_by_pivot_run(tmp_path: pathlib.Path) -> None:
    root, home = _project(tmp_path, {"a": 1})

    template = helpers.run_pivot(root, home, "run", "p/cut")
    instance = helpers.run_pivot(root, home, "run", "cut@a")

    assert template.returncode != 0
    assert "pivot repro p/cut" in template.stdout + template.stderr, template.stderr
    assert instance.returncode != 0
    assert "'cut@a' is mapped over the files of 'p/plan'" in instance.stderr, instance.stderr
