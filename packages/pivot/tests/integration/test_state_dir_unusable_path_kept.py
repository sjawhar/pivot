"""A pipeline keeps the `.pivot` fallback its stages took for a path that could not be a directory.

`core.state_dir: s1`, where `s1` cannot become a directory when the pipeline loads (a regular
file, or a link to a path under a regular file), so its stages record `.pivot`. A library caller
then leaves the file and runs the pipeline it loaded: its locks, its run record, its database
entries, its fingerprint caches and its cache objects all go to `.pivot`, and the configured path
gets nothing. Or the caller removes the file, or re-points the link at a usable directory, and
then runs it: the configuration now names `s1`, where `pivot gc` reads, so the run refuses,
naming both directories, and writes no lock. Or the caller loads `pa` while `s1` is a file,
removes the file, then loads `pb` and includes `pa`, after `pb`'s `with` block or inside it:
`pa` keeps its state in `.pivot` and `pb` in `s1`, so the include is refused, naming both
directories, and nothing runs.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
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

_PA = """\
from pivot import Pipeline, stage


@stage
def a(value: str) -> str:
    return value.upper()


with Pipeline("pa") as pipeline:
    a(pipeline.input("input.txt", t=str))
"""

_PB = """\
from pivot import Pipeline, stage


@stage
def b(value: str) -> str:
    return value + "b"


with Pipeline("pb") as pipeline:
    b(pipeline.input("input.txt", t=str))
"""

# `pb` includes `pa` inside its construction block, so the refusal comes from pb's load.
_PB_INCLUDING_PA = """\
from pivot import Pipeline, stage

import pa


@stage
def b(value: str) -> str:
    return value + "b"


with Pipeline("pb") as pipeline:
    pipeline.include(pa.pipeline)
    b(pipeline.input("input.txt", t=str))
"""

_SINGLE_SHAPES = ("file-removed", "link-re-pointed", "file-kept")

_CALLER = """\
import importlib
import os
import pathlib
import sys

_INCLUDE_SHAPES = ("after-block", "inside-block")


def main() -> None:
    root = pathlib.Path(sys.argv[1])
    os.chdir(root)
    sys.path.insert(0, str(root))
    shape = sys.argv[2]

    from pivot import discovery, exceptions, executor

    if shape not in _INCLUDE_SHAPES:
        pipeline = discovery.discover_pipeline()
    else:
        pa = importlib.import_module("pa").pipeline
    if shape == "link-re-pointed":
        (root / "s1").unlink()
        (root / "real").mkdir()
        (root / "s1").symlink_to("real")
    elif shape != "file-kept":
        (root / "s1").unlink()
    if shape in _INCLUDE_SHAPES:
        try:
            pipeline = importlib.import_module("pb").pipeline
            if shape == "after-block":
                pipeline.include(pa)
        except exceptions.PipelineConfigError as e:
            print("REFUSED " + str(e))
            return
    try:
        results = executor.run(pipeline=pipeline)
    except exceptions.PipelineConfigError as e:
        print("REFUSED " + str(e))
        return
    print("RAN " + " ".join(sorted(f"{k}={v['status']}" for k, v in results.items())))


if __name__ == "__main__":
    main()
"""


def _run(tmp_path: pathlib.Path, shape: str) -> tuple[pathlib.Path, str]:
    home = tmp_path / "home"
    root = tmp_path / "project"
    root.mkdir()
    assert helpers.run_cli(root, home, "git", "init", "-q").returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: s1\n")
    if shape == "link-re-pointed":
        # The link names a path under a regular file, so it cannot become a directory.
        (root / "afile").write_text("a regular file\n")
        (root / "s1").symlink_to("afile/x")
    else:
        (root / "s1").write_text("a regular file where the state directory goes\n")
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("first\n")
    if shape in _SINGLE_SHAPES:
        (root / "pipeline.py").write_text(_TWO_STAGES)
    else:
        (root / "pa.py").write_text(_PA)
        (root / "pb.py").write_text(_PB_INCLUDING_PA if shape == "inside-block" else _PB)
    caller = tmp_path / "caller.py"
    caller.write_text(_CALLER)
    ran = subprocess.run(
        [sys.executable, str(caller), str(root), shape],
        cwd=root,
        env=helpers.cli_env(home, PYTHONPATH=str(root)),
        capture_output=True,
        text=True,
        errors="replace",
        timeout=300,
        check=False,
    )
    assert ran.returncode == 0, ran.stdout + ran.stderr
    return root, ran.stdout + ran.stderr


def _keys(state_dir: pathlib.Path, prefix: bytes) -> int:
    with state.StateDB(state_dir, readonly=True) as db:
        return len(list(db.iter_prefix(prefix)))


def test_a_pipeline_loaded_while_its_path_could_not_be_used_runs_in_pivot(
    tmp_path: pathlib.Path,
) -> None:
    root, output = _run(tmp_path, "file-kept")

    assert "RAN test/a=ran test/b=ran" in output, output
    assert (root / "s1").is_file(), output
    pivot_dir = root / ".pivot"
    locks = sorted(path.name for path in (pivot_dir / "stages" / "test").iterdir())
    assert locks == ["a.lock", "b.lock"], output
    databases = sorted(path.relative_to(root).as_posix() for path in root.rglob("state.lmdb"))
    assert databases == [".pivot/state.lmdb"], output
    # The run's manifest and both stages' fingerprint manifests.
    assert (_keys(pivot_dir, b"run:"), _keys(pivot_dir, b"sm:")) == (1, 2), output
    objects = [
        path
        for path in (pivot_dir / "cache" / "files").rglob("*")
        if path.is_file() and not path.name.startswith(".")
    ]
    assert len(objects) == 2, output

    # The library caller gets the command's warning once, at the load: the path cannot be used
    # and Pivot uses `.pivot`.
    real_root = os.path.realpath(root)
    warning = (
        f"core.state_dir must be a directory: {real_root}/s1 (core.state_dir in "
        + f"{real_root}/.pivot/config.yaml). The state directory cannot be used; Pivot uses the "
        + "default .pivot."
    )
    assert output.splitlines().count(warning) == 1, output
    assert output.count("Pivot uses the default .pivot.") == 1, output
    # The same line a command prints while the path still cannot be used.
    status = helpers.run_pivot(root, tmp_path / "home", "status")
    assert status.stderr.splitlines().count(warning) == 1, status.stderr

    # Once a later command can use `s1`, it reads the locks from `.pivot`; its notice must name
    # every source of locks there, and a move that works although git tracks none of them.
    (root / "s1").unlink()
    status = helpers.run_pivot(root, tmp_path / "home", "status")
    assert status.returncode == 0, status.stdout + status.stderr
    assert (
        "holds stage locks or cache objects of this project: earlier Pivot releases kept them "
        + "there, and so does a run while core.state_dir is unset or cannot be used"
    ) in status.stderr, status.stderr
    move = shlex.join(["mv", f"{real_root}/.pivot/stages", f"{real_root}/s1/stages"])
    assert f"move the locks with `{move}`" in status.stderr, status.stderr
    assert "or delete them if a run that could not use core.state_dir made them." in status.stderr
    # Run as printed, from a subdirectory, with no lock tracked by git.
    moved = subprocess.run(shlex.split(move), cwd=root / "data", capture_output=True, check=False)
    assert moved.returncode == 0, moved.stderr
    assert sorted(path.name for path in (root / "s1" / "stages" / "test").iterdir()) == locks


@pytest.mark.parametrize("shape", ["file-removed", "link-re-pointed"])
def test_a_pipeline_loaded_while_its_path_could_not_be_used_refuses_to_run_once_it_can(
    tmp_path: pathlib.Path, shape: str
) -> None:
    """Its stages record `.pivot`, and the configuration now names `s1`, where `pivot gc`
    reads: a run would record its stages where no collection reads them, so it refuses."""
    root, output = _run(tmp_path, shape)

    real_root = os.path.realpath(root)
    now = f"{real_root}/real" if shape == "link-re-pointed" else f"{real_root}/s1"
    flat = " ".join(output.split())
    assert "REFUSED" in flat and "RAN" not in flat, output
    assert f"kept its state in {real_root}/.pivot" in flat and f"names {now}" in flat, output
    assert "Load the pipeline again" in flat, output
    assert not list(root.rglob("*.lock")), output
    target = root / ("real" if shape == "link-re-pointed" else "s1")
    assert not target.exists() or not list(target.iterdir()), output


@pytest.mark.parametrize("shape", ["after-block", "inside-block"])
def test_include_refuses_a_pipeline_loaded_while_its_path_was_a_file(
    tmp_path: pathlib.Path, shape: str
) -> None:
    root, output = _run(tmp_path, shape)

    real_root = os.path.realpath(root)
    assert (
        f"Pipeline 'pb' cannot include 'pa': 'pb' keeps the state of the project at {real_root} "
        + f"in {real_root}/s1, and 'pa' in {real_root}/.pivot. core.state_dir, or the path it "
        + "names, changed between their loads; load both pipelines after the change."
    ) in output, output
    assert "RAN" not in output, output
    assert not (root / "s1").exists(), output
    assert not list((root / ".pivot" / "stages").rglob("*.lock")), output
