"""`include()` refuses to combine pipelines that keep one project's state in two directories.

A pipeline resolves `core.state_dir` when it loads and records the directory on its stages, and
its runs and caches use the same directory. A library program that loads `pa`, edits the value
from `s1` to `s2` and loads `pb` holds two pipelines in two directories: one run of `pb` with
`pa` included would keep `pa/a`'s lock in `s1` and the rest in `s2`. So the include is refused,
after the `with` block or inside it, naming both directories. Loaded after the edit, both
pipelines keep everything in `s2`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import typing

import pytest

from tests import helpers

if typing.TYPE_CHECKING:
    import pathlib

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

_CALLER = """\
import importlib
import os
import pathlib
import sys


def main() -> None:
    os.chdir(sys.argv[1])
    sys.path.insert(0, sys.argv[1])
    order = sys.argv[2]

    from pivot import exceptions, executor

    config = pathlib.Path(".pivot/config.yaml")
    if order == "after-edit":
        config.write_text("core:\\n  state_dir: s2\\n")
    pa = importlib.import_module("pa").pipeline
    config.write_text("core:\\n  state_dir: s2\\n")
    try:
        pb = importlib.import_module("pb").pipeline
        if order != "inside-block":
            pb.include(pa)
    except exceptions.PipelineConfigError as e:
        print("REFUSED " + str(e))
        return
    results = executor.run(pipeline=pb)
    print("RAN " + " ".join(sorted(f"{k}={v['status']}" for k, v in results.items())))


if __name__ == "__main__":
    main()
"""


def _run(tmp_path: pathlib.Path, order: str) -> tuple[pathlib.Path, str]:
    home = tmp_path / "home"
    root = tmp_path / "project"
    root.mkdir()
    assert helpers.run_cli(root, home, "git", "init", "-q").returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: s1\n")
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("hello\n")
    (root / "pa.py").write_text(_PA)
    (root / "pb.py").write_text(_PB_INCLUDING_PA if order == "inside-block" else _PB)
    caller = tmp_path / "caller.py"
    caller.write_text(_CALLER)
    ran = subprocess.run(
        [sys.executable, str(caller), str(root), order],
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


@pytest.mark.parametrize("order", ["after-block", "inside-block"])
def test_include_refuses_a_pipeline_loaded_before_a_state_dir_edit(
    tmp_path: pathlib.Path, order: str
) -> None:
    root, output = _run(tmp_path, order)

    s1 = os.path.realpath(root / "s1")
    s2 = os.path.realpath(root / "s2")
    assert (
        "Pipeline 'pb' cannot include 'pa': 'pb' keeps the state of the project at "
        + f"{os.path.realpath(root)} in {s2}, and 'pa' in {s1}. core.state_dir, or the path it "
        + "names, changed between their loads; load both pipelines after the change."
    ) in output, output
    assert not (root / "s1" / "stages").exists(), output
    assert not (root / "s2" / "stages").exists(), output


def test_pipelines_loaded_after_the_edit_include_and_run_in_one_state_dir(
    tmp_path: pathlib.Path,
) -> None:
    root, output = _run(tmp_path, "after-edit")

    assert "RAN pa/a=ran pb/b=ran" in output, output
    assert (root / "s2" / "stages" / "pa" / "a.lock").is_file(), output
    assert (root / "s2" / "stages" / "pb" / "b.lock").is_file(), output
    assert not (root / "s1").exists(), output
