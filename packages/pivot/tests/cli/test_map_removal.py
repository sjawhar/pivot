"""A key that leaves a mapped stage's plan (``Pipeline.map``): its instances lose their files.

Each test drives the installed ``pivot`` binary in a real git project. The producer ``p/plan``
writes one file per key of ``data/raw/plan.yaml`` and fails on a negative entry; ``p/cut`` is
mapped over those files and writes one file per key, and ``p/pack`` is mapped over ``p/cut``'s
instances and writes a directory per key, so its instances commit a listing beside their lock.

A run that finds ``p/plan`` current removes each instance whose key its listing no longer holds:
its lock, its listing and its outputs in the workspace. The cache keeps every object. Before the
run, ``pivot status`` lists such an instance as orphaned and ``pivot repro --dry-run`` names it.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import yaml

from pivot.storage import cache
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
    if any(value < 0 for value in entries.values()):
        raise ValueError("a negative entry")
    return {f"{name}.yaml": {"value": value} for name, value in entries.items()}


@stage
def cut(entry: dict[str, int]) -> dict[str, int]:
    return {"cut": entry["value"] * 2}


@stage
def pack(
    cut: dict[str, int],
) -> Annotated[dict[str, dict[str, int]], directory, loaders.YAML[dict[str, int]]()]:
    return {"cut.yaml": cut, "doubled.yaml": {"value": cut["cut"] * 2}}


with Pipeline("p") as pipeline:
    planned = plan(pipeline.input("plan.yaml", t=dict[str, int]))
    pack(cut(pipeline.map(planned)))
"""

_DROPPED = ("p/cut@b", "p/pack@b")
_KEPT = ("p/cut@a", "p/cut@c", "p/pack@a", "p/pack@c")
_REASON = "its key left the listing of 'p/plan'"


def _ok(proc: subprocess.CompletedProcess[str]) -> str:
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    return out


def _stdout(proc: subprocess.CompletedProcess[str]) -> str:
    _ok(proc)
    return proc.stdout


def _write_plan(root: pathlib.Path, plan: Mapping[str, int]) -> None:
    (root / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (root / "data" / "raw" / "plan.yaml").write_text(yaml.safe_dump(dict(plan)))


def _project(
    tmp_path: pathlib.Path, pipeline: str = _PIPELINE, *, no_commit: bool = False
) -> tuple[pathlib.Path, pathlib.Path]:
    """A git project holding a mapped pipeline, run once over keys a, b and c (``no_commit``:
    with ``--no-commit``)."""
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
    (root / "pipeline.py").write_text(pipeline)
    _write_plan(root, {"a": 1, "b": 2, "c": 3})
    _ok(helpers.run_pivot(root, home, "repro", *(["--no-commit"] if no_commit else [])))
    return root, home


def _snapshot(root: pathlib.Path) -> dict[str, bytes]:
    """Every file of the outputs, the locks and listings, and the cache, with its bytes."""
    return {
        str(path.relative_to(root)): path.read_bytes()
        for top in ("data", ".pivot/stages", ".pivot/cache/files")
        for path in (root / top).rglob("*")
        if path.is_file()
    }


def _dropped_by_the_producer_alone(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """The project after key b left the plan in a run of the producer alone: `pivot repro p/plan`
    expands nothing, so it removes nothing, and `p/plan` is current with a listing without b."""
    root, home = _project(tmp_path)
    _write_plan(root, {"a": 1, "c": 3})
    _ok(helpers.run_pivot(root, home, "repro", "p/plan"))
    return root, home


def _instance_files(root: pathlib.Path, instance: str) -> list[str]:
    """The instance's files beside its lock, relative to the stages directory."""
    stages = root / ".pivot" / "stages"
    return sorted(str(path.relative_to(stages)) for path in stages.glob(f"{instance}[.=]*"))


def _outputs(root: pathlib.Path, instance: str) -> pathlib.Path:
    """The instance's output in the workspace: `p/cut`'s a file, `p/pack`'s a directory."""
    name = instance.removeprefix("p/")
    return root / "data" / "p" / (f"{name}.yaml" if name.startswith("cut@") else name)


def _output_objects(root: pathlib.Path, instance: str) -> set[str]:
    """Every cache object the instance's outputs need: each hash its lock records for them (a
    file's, a directory's listing) and each file its outputs hold now."""
    document = yaml.safe_load((root / ".pivot" / "stages" / f"{instance}.lock").read_text())
    records = document["records"] if "records" in document else document
    objects = set[str]()
    for entry in records["outs"]:
        objects.update(entry[key] for key in ("hash", "listing") if key in entry)
    output = _outputs(root, instance)
    files = [output] if output.is_file() else [path for path in output.rglob("*") if path.is_file()]
    objects.update(cache.hash_file(path, None)[0] for path in files)
    return objects


def _status_rows(root: pathlib.Path, home: pathlib.Path) -> dict[str, str]:
    ran = helpers.run_pivot(root, home, "status", "--stages-only", "--json")
    return {row["name"]: row["status"] for row in json.loads(_stdout(ran))["stages"]}


def _assert_kept_whole(root: pathlib.Path, instance: str) -> None:
    assert (root / ".pivot" / "stages" / f"{instance}.lock").is_file(), instance
    assert _outputs(root, instance).exists(), instance
    if instance.startswith("p/pack@"):
        assert f"{instance}=.listing" in _instance_files(root, instance), instance


def test_a_key_that_left_the_listing_loses_its_lock_and_outputs(tmp_path: pathlib.Path) -> None:
    root, home = _project(tmp_path)
    objects = set[str]().union(*(_output_objects(root, instance) for instance in _DROPPED))
    assert _instance_files(root, "p/pack@b") == ["p/pack@b.lock", "p/pack@b=.listing"]
    _write_plan(root, {"a": 1, "c": 3})

    repro = _ok(helpers.run_pivot(root, home, "repro"))

    assert "p/cut: expanded into 2 stages; removed p/cut@b, whose key left the plan" in repro
    assert "p/pack: expanded into 2 stages; removed p/pack@b, whose key left the plan" in repro
    for instance in _DROPPED:
        assert _instance_files(root, instance) == [], instance
        assert not _outputs(root, instance).exists(), instance
    for instance in _KEPT:
        _assert_kept_whole(root, instance)
    files = root / ".pivot" / "cache" / "files"
    assert objects, "the dropped instances recorded outputs"
    assert sorted(o for o in objects if not cache.get_cache_path(files, o).is_file()) == []
    assert _status_rows(root, home) == dict.fromkeys(["p/plan", *_KEPT], "cached")


def test_status_lists_an_orphaned_instance_before_the_run(tmp_path: pathlib.Path) -> None:
    root, home = _dropped_by_the_producer_alone(tmp_path)

    rows = _status_rows(root, home)
    text = _ok(helpers.run_pivot(root, home, "status", "--stages-only"))
    explained = json.loads(
        _stdout(helpers.run_pivot(root, home, "status", "--explain", "--stages-only", "--json"))
    )
    targeted = helpers.run_pivot(root, home, "status", "--stages-only", "--json", "p/cut")
    downstream = helpers.run_pivot(root, home, "status", "--stages-only", "--json", "pack")
    quiet = helpers.run_pivot(root, home, "--quiet", "status", "--stages-only")

    assert rows == {
        **dict.fromkeys(["p/plan", *_KEPT], "cached"),
        **dict.fromkeys(_DROPPED, "orphaned"),
    }, rows
    assert "2 orphaned instances of mapped stages" in text, text
    assert "p/cut@b" in text and f"Orphaned: {_REASON}" in text, text
    assert "the next run that expands 'p/cut' removes its lock and outputs" in text, text
    assert {stage["name"]: stage["status"] for stage in explained["stages"]}["p/pack@b"] == (
        "orphaned"
    ), explained
    assert {row["name"]: row["status"] for row in json.loads(_stdout(targeted))["stages"]} == {
        "p/plan": "cached",
        "p/cut@a": "cached",
        "p/cut@c": "cached",
        "p/cut@b": "orphaned",
    }
    # `pivot repro p/pack` expands p/cut too, so its status lists p/cut's orphan as well
    assert {
        row["name"]: row["status"]
        for row in json.loads(_stdout(downstream))["stages"]
        if row["status"] == "orphaned"
    } == dict.fromkeys(_DROPPED, "orphaned")
    assert quiet.returncode == 1, "an orphaned instance is work the next run does"
    for instance in _DROPPED:
        _assert_kept_whole(root, instance)


def test_dry_run_prints_the_removal_and_removes_nothing(tmp_path: pathlib.Path) -> None:
    root, home = _dropped_by_the_producer_alone(tmp_path)
    before = _snapshot(root)

    dry_run = _ok(helpers.run_pivot(root, home, "repro", "--dry-run"))
    as_json = json.loads(_stdout(helpers.run_pivot(root, home, "repro", "--dry-run", "--json")))
    producer_only = _ok(helpers.run_pivot(root, home, "repro", "--dry-run", "p/plan"))
    after = _snapshot(root)

    assert "Would remove:" in dry_run, dry_run
    for instance in _DROPPED:
        assert f"  {instance}: {_REASON}" in dry_run, dry_run
    assert as_json["would_remove"] == dict.fromkeys(_DROPPED, _REASON), as_json
    assert "Would remove" not in producer_only, "a run of the producer alone expands nothing"
    assert after == before, "a dry run changed a file"
    for instance in _DROPPED:
        _assert_kept_whole(root, instance)

    _ok(helpers.run_pivot(root, home, "repro"))
    for instance in _DROPPED:
        assert _instance_files(root, instance) == [], instance
        assert not _outputs(root, instance).exists(), instance


def test_a_producer_that_is_not_current_removes_nothing(tmp_path: pathlib.Path) -> None:
    """Key b already left `p/plan`'s listing, but `p/plan` is stale and fails when it runs: no
    command reads its plan, so b's instances keep every file."""
    root, home = _dropped_by_the_producer_alone(tmp_path)
    _write_plan(root, {"a": 1, "c": -3})

    rows = _status_rows(root, home)
    dry_run = _ok(helpers.run_pivot(root, home, "repro", "--dry-run"))
    failed = helpers.run_pivot(root, home, "repro")

    assert rows == {"p/plan": "stale", "p/cut": "not_expanded", "p/pack": "not_expanded"}, rows
    assert "Would remove" not in dry_run, dry_run
    assert failed.returncode == 1, failed.stdout + failed.stderr
    assert "removed" not in failed.stdout + failed.stderr
    for instance in (*_DROPPED, *_KEPT):
        _assert_kept_whole(root, instance)


def test_a_no_commit_run_removes_nothing(tmp_path: pathlib.Path) -> None:
    """A `--no-commit` run commits no lock, so it removes none either; the next committed run
    does."""
    root, home = _dropped_by_the_producer_alone(tmp_path)

    dry_run = _ok(helpers.run_pivot(root, home, "repro", "--dry-run", "--no-commit"))
    ran = _ok(helpers.run_pivot(root, home, "repro", "--no-commit"))

    assert "Would remove" not in dry_run, dry_run
    assert "removed" not in ran, ran
    for instance in _DROPPED:
        _assert_kept_whole(root, instance)
    assert _status_rows(root, home)["p/cut@b"] == "orphaned"


def test_an_instance_whose_output_holds_a_tracked_file_is_kept_whole(
    tmp_path: pathlib.Path,
) -> None:
    """A file tracked inside a dropped instance's output is the user's: the run keeps that
    instance whole, says why, and removes the other dropped instances."""
    root, home = _dropped_by_the_producer_alone(tmp_path)
    notes = root / "data" / "p" / "pack@b" / "notes.txt"
    notes.write_text("mine\n")
    _ok(helpers.run_pivot(root, home, "track", "data/p/pack@b/notes.txt"))

    ran = _ok(helpers.run_pivot(root, home, "repro"))

    assert "Cannot remove 'p/pack@b', whose key left the plan" in ran, ran
    assert "notes.txt" in ran, ran
    _assert_kept_whole(root, "p/pack@b")
    assert notes.read_text() == "mine\n"
    assert _instance_files(root, "p/cut@b") == []
    assert not _outputs(root, "p/cut@b").exists()
    assert _status_rows(root, home)["p/pack@b"] == "orphaned"


def test_naming_an_orphaned_instance_is_refused_and_removes_nothing(
    tmp_path: pathlib.Path,
) -> None:
    """`p/plan` has no file for key b, so `p/cut@b` names no stage: `status`, `repro --dry-run`
    and `repro` refuse it alike, as any key the producer has no file for."""
    root, home = _dropped_by_the_producer_alone(tmp_path)

    refused = [
        helpers.run_pivot(root, home, *command, "p/cut@b")
        for command in (("status", "--stages-only"), ("repro", "--dry-run"), ("repro",))
    ]

    for ran in refused:
        out = " ".join((ran.stdout + ran.stderr).split())
        assert ran.returncode != 0, out
        assert "Unknown stage(s): p/cut@b" in out, out
    for instance in _DROPPED:
        _assert_kept_whole(root, instance)


_SPLIT_PIPELINE = """\
from typing import Annotated, TypedDict

from pivot import Pipeline, directory, loaders, stage


class Split(TypedDict):
    left: Annotated[dict[str, dict[str, int]], directory, loaders.YAML[dict[str, int]]()]
    right: Annotated[dict[str, dict[str, int]], directory, loaders.YAML[dict[str, int]]()]


@stage
def plan(
    entries: dict[str, int],
) -> Annotated[dict[str, dict[str, int]], directory, loaders.YAML[dict[str, int]]()]:
    return {f"{name}.yaml": {"value": value} for name, value in entries.items()}


@stage
def split(entry: dict[str, int]) -> Split:
    return Split(left={"l.yaml": entry}, right={"r.yaml": entry})


with Pipeline("p") as pipeline:
    split(pipeline.map(plan(pipeline.input("plan.yaml", t=dict[str, int]))))
"""


def test_a_tracked_file_in_one_output_keeps_every_output_of_the_instance(
    tmp_path: pathlib.Path,
) -> None:
    """`p/split@b` writes two directories; a file tracked in the second refuses the removal
    before the first is touched, so the instance stays whole."""
    root, home = _project(tmp_path, _SPLIT_PIPELINE)
    _write_plan(root, {"a": 1, "c": 3})
    _ok(helpers.run_pivot(root, home, "repro", "p/plan"))
    instance = root / "data" / "p" / "split@b"
    before = {
        str(path.relative_to(root)): path.read_bytes()
        for path in instance.rglob("*")
        if path.is_file()
    }
    notes = instance / "right" / "notes.txt"
    notes.write_text("mine\n")
    _ok(helpers.run_pivot(root, home, "track", "data/p/split@b/right/notes.txt"))
    files = _instance_files(root, "p/split@b")

    ran = _ok(helpers.run_pivot(root, home, "repro"))

    assert "Cannot remove 'p/split@b', whose key left the plan" in ran, ran
    assert {name: data for name, data in before.items() if (root / name).is_file()} == before, (
        "the run removed an output of the kept instance"
    )
    assert notes.read_text() == "mine\n"
    assert files == ["p/split@b.lock", "p/split@b=left.listing", "p/split@b=right.listing"]
    assert _instance_files(root, "p/split@b") == files


def test_the_paths_the_lock_records_go_after_the_output_moved(tmp_path: pathlib.Path) -> None:
    """`p/cut` now writes JSON, so its instances' outputs move from `cut@<key>.yaml` to
    `cut@<key>.json`. The dropped `p/cut@b` wrote `cut@b.yaml`, which its lock records: that is
    the file the run removes."""
    root, home = _project(tmp_path)
    (root / "pipeline.py").write_text(
        _PIPELINE.replace(
            "def cut(entry: dict[str, int]) -> dict[str, int]:",
            "def cut(\n    entry: dict[str, int],\n"
            + ") -> Annotated[dict[str, int], loaders.JSON[dict[str, int]]()]:",
        )
    )
    _write_plan(root, {"a": 1, "c": 3})
    assert (root / "data" / "p" / "cut@b.yaml").is_file()

    ran = _ok(helpers.run_pivot(root, home, "repro"))

    assert "removed p/cut@b, whose key left the plan" in ran, ran
    assert not (root / "data" / "p" / "cut@b.yaml").exists()
    assert _instance_files(root, "p/cut@b") == []
    assert (root / "data" / "p" / "cut@a.json").is_file()


def test_a_committed_run_removes_what_a_no_commit_run_wrote_for_a_dropped_key(
    tmp_path: pathlib.Path,
) -> None:
    """A `--no-commit` run wrote every instance's outputs and no lock. Once b left the plan,
    the next committed run removes b's outputs and its `nocommit:` records, so the run after it
    has nothing left to remove."""
    root, home = _project(tmp_path, no_commit=True)
    assert _outputs(root, "p/cut@b").is_file() and _instance_files(root, "p/cut@b") == []
    _write_plan(root, {"a": 1, "c": 3})

    ran = _ok(helpers.run_pivot(root, home, "repro"))
    again = _ok(helpers.run_pivot(root, home, "repro"))

    for instance in _DROPPED:
        assert f"removed {instance}, whose key left the plan" in ran, ran
        assert not _outputs(root, instance).exists(), instance
    assert "removed" not in again, again
    assert _status_rows(root, home) == dict.fromkeys(["p/plan", *_KEPT], "cached")
