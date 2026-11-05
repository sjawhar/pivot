"""A library run must not silently reuse another project's cached root after chdir."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from tests import helpers

if TYPE_CHECKING:
    import pathlib

_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def {name}() -> str:
    return "{name}"


with Pipeline("u") as pipeline:
    {name}()
"""

_CALLER = """\
import json
import os
import pathlib
import sys

from pivot import discovery, exceptions, executor


def main() -> None:
    first_root, second_root, supplied = sys.argv[1:]
    os.chdir(first_root)
    pipeline = discovery.discover_pipeline() if supplied == "supplied" else None
    first = executor.run(pipeline=pipeline, max_workers=1)
    print("FIRST " + json.dumps(first), flush=True)
    os.chdir("subdir")
    within = executor.run(pipeline=pipeline, max_workers=1)
    print("WITHIN " + json.dumps(within), flush=True)
    os.chdir(second_root)
    try:
        second = executor.run(pipeline=pipeline, max_workers=1, force=True)
    except exceptions.PipelineConfigError as error:
        print("REFUSED " + str(error), flush=True)
        os.chdir(first_root)
        returned = executor.run(pipeline=pipeline, max_workers=1)
        print("RETURNED " + json.dumps(returned), flush=True)
        sys.exit(3)
    print("SECOND " + json.dumps(second), flush=True)


if __name__ == "__main__":
    main()
"""


@pytest.mark.parametrize(
    ("destination", "supplied"),
    [
        pytest.param("project", "discovered", id="project-discovered"),
        pytest.param("project", "supplied", id="project-supplied"),
        pytest.param("uninitialized", "discovered", id="uninitialized-discovered"),
    ],
)
def test_library_run_refuses_a_cwd_outside_its_held_project(
    tmp_path: pathlib.Path, supplied: str, destination: str
) -> None:
    """A forced second run must refuse, not run or report the first project's stages."""
    home = tmp_path / "home"
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        root.mkdir()
        assert helpers.run_cli(root, home, "git", "init", "-q").returncode == 0
    assert helpers.run_pivot(first, home, "init").returncode == 0
    (first / "subdir").mkdir()
    (first / "pipeline.py").write_text(_PIPELINE.format(name="a"))
    if destination == "project":
        assert helpers.run_pivot(second, home, "init").returncode == 0
        (second / "pipeline.py").write_text(_PIPELINE.format(name="b"))
    caller = tmp_path / "caller.py"
    caller.write_text(_CALLER)

    ran = subprocess.run(
        [sys.executable, str(caller), str(first), str(second), supplied],
        cwd=first,
        env=helpers.cli_env(home),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    output = ran.stdout + ran.stderr
    assert ran.returncode == 3, output
    records = dict(line.split(" ", 1) for line in ran.stdout.splitlines())
    assert str(first) in records["REFUSED"], output
    assert str(second) in records["REFUSED"], output
    assert "new Python process" in records["REFUSED"], output
    for label, status in (("FIRST", "ran"), ("WITHIN", "cached"), ("RETURNED", "cached")):
        result = json.loads(records[label])
        assert set(result) == {"u/a"}, output
        assert result["u/a"]["status"] == status, output
    assert (first / "data" / "u" / "a.txt").read_text() == "a", output
    assert not (second / "data").exists(), output
    assert list((second / ".pivot" / "stages").rglob("*.lock")) == [], output


def test_library_run_accepts_explicit_pipeline_outside_any_initialized_project(
    tmp_path: pathlib.Path,
) -> None:
    """An explicit selected pipeline still runs when cwd selects no competing project."""
    home = tmp_path / "home"
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    assert helpers.run_cli(first, home, "git", "init", "-q").returncode == 0
    assert helpers.run_pivot(first, home, "init").returncode == 0
    (first / "subdir").mkdir()
    (first / "pipeline.py").write_text(_PIPELINE.format(name="a"))
    caller = tmp_path / "caller.py"
    caller.write_text(_CALLER)

    ran = subprocess.run(
        [sys.executable, str(caller), str(first), str(second), "supplied"],
        cwd=first,
        env=helpers.cli_env(home),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    output = ran.stdout + ran.stderr
    assert ran.returncode == 0, output
    records = dict(line.split(" ", 1) for line in ran.stdout.splitlines())
    for label, status in (("FIRST", "ran"), ("WITHIN", "cached"), ("SECOND", "ran")):
        result = json.loads(records[label])
        assert set(result) == {"u/a"}, output
        assert result["u/a"]["status"] == status, output
    assert (first / "data" / "u" / "a.txt").read_text() == "a", output
    assert list(second.iterdir()) == [], output
