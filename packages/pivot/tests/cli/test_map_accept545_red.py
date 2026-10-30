"""Red tests from the acceptance of sjawhar/pivot#545 (BET-473) at ae91050dd18a.

Run from a checkout of the PR head (on a pivot-slices VM, never on sami-rocqperf):

    cp /mnt/nvme/sami/tmp/BET408/accept/test_map_accept545_red.py packages/pivot/tests/cli/
    uv run pytest packages/pivot/tests/cli/test_map_accept545_red.py -p no:randomly -n 4 --basetemp=/tmp/a545

Each test drives the installed ``pivot`` binary in a real git project, as
``tests/cli/test_map_commands.py`` does. Every test failed at ae91050dd18a.
"""

from __future__ import annotations

import json
import signal
import subprocess
import time
from typing import TYPE_CHECKING

import yaml

from tests import helpers

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable, Mapping

_PIPELINE = """\
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


def _ok(proc: subprocess.CompletedProcess[str]) -> str:
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    return out


def _project(tmp_path: pathlib.Path, plan: Mapping[str, int]) -> tuple[pathlib.Path, pathlib.Path]:
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


def _write_plan(root: pathlib.Path, plan: Mapping[str, int]) -> None:
    (root / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (root / "data" / "raw" / "plan.yaml").write_text(yaml.safe_dump(dict(plan)))


def _read(root: pathlib.Path, stage: str) -> object:
    return yaml.safe_load((root / "data" / "p" / f"{stage}.yaml").read_text())


def _ran(proc: subprocess.CompletedProcess[str]) -> set[str]:
    """The stages a ``pivot repro --json`` run ran (not cached)."""
    ran = set[str]()
    for line in proc.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "stage_complete" and event["status"] == "ran":
            ran.add(event["stage"])
    return ran


def _status_rows(root: pathlib.Path, home: pathlib.Path, *targets: str) -> dict[str, str]:
    proc = helpers.run_pivot(root, home, "status", "--stages-only", "--json", *targets)
    _ok(proc)
    return {row["name"]: row["status"] for row in json.loads(proc.stdout)["stages"]}


# =============================================================================
# Targets naming one instance, or a key no instance has
# =============================================================================


def test_repro_of_a_key_not_in_the_plan_is_refused_and_runs_nothing(
    tmp_path: pathlib.Path,
) -> None:
    """NEGATIVE CONTROL. `pivot status`, `verify` and `checkout` refuse `p/cut@zzz`; `pivot repro`
    took it for `p/cut` and ran every stale instance (exit 0) at ae91050dd18a."""
    root, home = _project(tmp_path, {"a": 1, "b": 2})
    _ok(helpers.run_pivot(root, home, "repro"))
    _write_plan(root, {"a": 1, "b": 20})
    _ok(helpers.run_pivot(root, home, "repro", "p/plan"))  # the producer is current; p/cut@b stale

    dry_run = helpers.run_pivot(root, home, "repro", "--dry-run", "p/cut@zzz")
    run = helpers.run_pivot(root, home, "repro", "--json", "p/cut@zzz")

    assert dry_run.returncode != 0, dry_run.stdout + dry_run.stderr
    assert run.returncode != 0, run.stdout + run.stderr
    assert "p/cut@zzz" in run.stdout + run.stderr
    assert _ran(run) == set(), "a refused target runs nothing"
    assert _read(root, "cut@b") == {"cut": 4}, "p/cut@b must not run for a target naming p/cut@zzz"


def test_repro_of_one_instance_runs_that_instances_chain_only(tmp_path: pathlib.Path) -> None:
    """`pivot status p/build@a` shows p/plan, p/cut@a and p/build@a; `pivot repro p/build@a` ran
    every stale instance of p/cut and p/build at ae91050dd18a."""
    root, home = _project(tmp_path, {"a": 1, "b": 2, "c": 3})
    _ok(helpers.run_pivot(root, home, "repro"))
    _write_plan(root, {"a": 1, "b": 20, "c": 30})
    _ok(helpers.run_pivot(root, home, "repro", "p/plan"))

    shown = _status_rows(root, home, "p/build@a")
    run = helpers.run_pivot(root, home, "repro", "--json", "p/build@a")

    assert set(shown) == {"p/plan", "p/cut@a", "p/build@a"}, shown
    _ok(run)
    assert _ran(run) <= {"p/plan", "p/cut@a", "p/build@a"}, _ran(run)
    assert _read(root, "cut@b") == {"cut": 4}, "p/cut@b is not upstream of p/build@a"
    assert _status_rows(root, home)["p/cut@b"] == "stale"


def test_force_of_one_instance_reruns_that_instance_only(tmp_path: pathlib.Path) -> None:
    """`pivot repro --force p/cut@a` force-reran every instance of p/cut at ae91050dd18a."""
    root, home = _project(tmp_path, {"a": 1, "b": 2, "c": 3})
    _ok(helpers.run_pivot(root, home, "repro"))

    run = helpers.run_pivot(root, home, "repro", "--json", "--force", "p/cut@a")

    _ok(run)
    assert "p/cut@a" in _ran(run), _ran(run)
    assert not _ran(run) & {"p/cut@b", "p/cut@c"}, _ran(run)


# =============================================================================
# pivot commit after pivot repro --no-commit
# =============================================================================


def test_commit_after_a_no_commit_run_commits_the_instances_it_ran(tmp_path: pathlib.Path) -> None:
    """With the static twin (variants p/cut@a ...) `pivot commit` commits p/plan, p/cut@a and
    p/build@a; with p/cut mapped it committed p/plan alone at ae91050dd18a, and the next repro
    ran p/cut@a and p/build@a again."""
    root, home = _project(tmp_path, {"a": 1, "b": 2})
    _ok(helpers.run_pivot(root, home, "repro"))
    _write_plan(root, {"a": 10, "b": 2})
    no_commit = helpers.run_pivot(root, home, "repro", "--json", "--no-commit")
    assert _ran(no_commit) == {"p/plan", "p/cut@a", "p/build@a"}, _ran(no_commit)

    committed = _ok(helpers.run_pivot(root, home, "commit"))

    assert "p/cut@a" in committed and "p/build@a" in committed, committed
    assert set(_status_rows(root, home).values()) == {"cached"}, _status_rows(root, home)
    rerun = helpers.run_pivot(root, home, "repro", "--json")
    assert _ran(rerun) == set(), f"a committed run reruns nothing: {_ran(rerun)}"


def test_commit_takes_an_instance_name(tmp_path: pathlib.Path) -> None:
    """`pivot commit p/plan p/cut@a` failed with "Unknown stage(s): p/cut@a" at ae91050dd18a."""
    root, home = _project(tmp_path, {"a": 1, "b": 2})
    _ok(helpers.run_pivot(root, home, "repro"))
    _write_plan(root, {"a": 10, "b": 2})
    _ok(helpers.run_pivot(root, home, "repro", "--no-commit"))

    committed = _ok(helpers.run_pivot(root, home, "commit", "p/plan", "p/cut@a"))

    assert "p/cut@a" in committed, committed
    rows = _status_rows(root, home)
    assert rows["p/cut@a"] == "cached" and rows["p/build@a"] == "stale", rows


# =============================================================================
# pivot params show
# =============================================================================


def test_params_show_lists_each_instances_params(tmp_path: pathlib.Path) -> None:
    """With the static twin `pivot params show --json` lists p/cut@a and p/cut@b; with p/cut mapped
    it printed `{}` ("No parameters found.") at ae91050dd18a, though the producer is current."""
    root, home = _project(tmp_path, {"a": 1, "b": 2})
    _ok(helpers.run_pivot(root, home, "repro"))
    (root / "params.yaml").write_text("p/cut@a:\n  factor: 7\n")

    shown = helpers.run_pivot(root, home, "params", "show", "--json")

    _ok(shown)
    document = json.loads(shown.stdout[shown.stdout.index("{") :])
    assert document == {"p/cut@a": {"factor": 7}, "p/cut@b": {"factor": 2}}, document


# =============================================================================
# Watch mode: a code change to a mapped stage
# =============================================================================


def _wait_for(predicate: Callable[[], bool], seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.5)
    return predicate()


def _file_says(path: pathlib.Path, expected: object) -> Callable[[], bool]:
    def check() -> bool:
        try:
            return yaml.safe_load(path.read_text()) == expected
        except (OSError, yaml.YAMLError):
            return False

    return check


def test_watch_reruns_the_instances_of_a_mapped_stage_whose_code_changed(
    tmp_path: pathlib.Path,
) -> None:
    """`pivot repro --watch` reloads on the edit of `build`'s code, but its run after the reload
    ran `p/plan` (cached) alone and expanded nothing, while `pivot status` read p/build@a "Code
    changed", at ae91050dd18a. With the static twin the reload run reruns p/build@a and p/build@b."""
    root, home = _project(tmp_path, {"a": 1, "b": 2})
    build_a = root / "data" / "p" / "build@a.yaml"
    watch = subprocess.Popen(
        [helpers.PIVOT, "repro", "--watch", "--json"],
        cwd=root,
        env=helpers.cli_env(home),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        assert _wait_for(_file_says(build_a, {"built": 1002}), 60), "the first watch run"
        time.sleep(2)
        pipeline = root / "pipeline.py"
        pipeline.write_text(pipeline.read_text().replace('cut["cut"] + 1000', 'cut["cut"] + 2000'))

        reran = _wait_for(_file_says(build_a, {"built": 2002}), 45)
    finally:
        watch.send_signal(signal.SIGINT)
        try:
            watch.wait(timeout=30)
        except subprocess.TimeoutExpired:
            watch.kill()
            watch.wait()

    assert reran, f"p/build@a after the code change: {build_a.read_text()!r}"
