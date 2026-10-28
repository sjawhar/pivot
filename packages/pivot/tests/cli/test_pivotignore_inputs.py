"""`.pivotignore` with the installed CLI: what it keeps from making a stage stale (files in a
directory input) and what it never touches (outputs, tracked data, the cache)."""

# pyright: reportMissingImports=false, reportImplicitRelativeImport=false

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

import pytest

from tests import helpers

if TYPE_CHECKING:
    import pathlib
    import subprocess

_PYC_IGNORE = "*.pyc\n__pycache__/\n"

_INPUT_PIPELINE = """\
import pathlib

from pivot import Pipeline, stage


@stage
def lane(server: pathlib.Path) -> str:
    return (server / "grade.py").read_text()


with Pipeline("p") as pipeline:
    lane(pipeline.input("server", path="tasks/server", t=pathlib.Path))
"""

_OUTPUT_PIPELINE = """\
import pathlib
import tempfile

from pivot import Pipeline, StageParams, stage


class BuildParams(StageParams):
    bytecode: str = "one"


@stage
def build(raw: str, params: BuildParams) -> pathlib.Path:
    out = pathlib.Path(tempfile.mkdtemp()) / "pkg"
    (out / "__pycache__").mkdir(parents=True)
    (out / "mod.py").write_text(raw)
    (out / "__pycache__" / "mod.cpython-312.pyc").write_text(params.bytecode)
    return out


@stage
def use(pkg: pathlib.Path) -> str:
    return (pkg / "mod.py").read_text()


with Pipeline("p") as pipeline:
    use(build(pipeline.input("raw.txt"), params=BuildParams()))
"""


def _ok(result: subprocess.CompletedProcess[str]) -> str:
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    return output


def _project(tmp_path: pathlib.Path, pipeline: str) -> tuple[pathlib.Path, pathlib.Path]:
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "project"
    root.mkdir()
    _ok(helpers.run_cli(root, home, "git", "init", "-q"))
    _ok(helpers.run_pivot(root, home, "init"))
    (root / ".pivotignore").write_text(_PYC_IGNORE)
    (root / "pipeline.py").write_text(pipeline)
    return home, root


def _statuses(root: pathlib.Path, home: pathlib.Path) -> dict[str, str]:
    out = _ok(helpers.run_pivot(root, home, "-q", "status", "--json"))
    return {stage["name"]: stage["status"] for stage in json.loads(out)["stages"]}


def _ran(root: pathlib.Path, home: pathlib.Path) -> dict[str, str]:
    """Run ``pivot repro`` and return each stage's completion status."""
    result = helpers.run_pivot(root, home, "repro", "--json")
    assert result.returncode == 0, result.stdout + result.stderr
    events = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    return {e["stage"]: e["status"] for e in events if e.get("type") == "stage_complete"}


def _source_dir(tmp_path: pathlib.Path, root: pathlib.Path, *, linked: bool) -> pathlib.Path:
    """``tasks/server`` holding ``grade.py``: in the project, or a symlink to a directory
    outside it. Returns the real directory."""
    (root / "tasks").mkdir()
    source = tmp_path / "elsewhere" / "server" if linked else root / "tasks" / "server"
    source.mkdir(parents=True)
    (source / "grade.py").write_text("print('grade')\n")
    if linked:
        (root / "tasks" / "server").symlink_to(source, target_is_directory=True)
    return source


@pytest.mark.parametrize("linked", [False, True], ids=["local", "symlinked-from-outside"])
def test_bytecode_written_into_an_input_directory_leaves_its_stage_cached(
    tmp_path: pathlib.Path, linked: bool
) -> None:
    """A1: Python writes __pycache__/*.pyc into a source directory a stage reads; .pivotignore
    names them, so the stage stays cached, for a local directory and a linked one alike."""
    home, root = _project(tmp_path, _INPUT_PIPELINE)
    source = _source_dir(tmp_path, root, linked=linked)
    assert _ran(root, home) == {"p/lane": "ran"}

    (source / "__pycache__").mkdir()
    (source / "__pycache__" / "grade.cpython-312.pyc").write_bytes(b"\x00bytecode")
    (source / "stray.pyc").write_bytes(b"\x00stray")

    assert _statuses(root, home) == {"p/lane": "cached"}
    check = _ok(helpers.run_pivot(root, home, "check-ignore", "tasks/server/__pycache__/x.pyc"))
    assert "tasks/server/__pycache__/x.pyc" in check, "check-ignore agrees with the hash"
    assert _ran(root, home) == {"p/lane": "cached"}


@pytest.mark.parametrize("linked", [False, True], ids=["local", "symlinked-from-outside"])
def test_edit_to_a_file_pivotignore_does_not_name_makes_the_stage_stale(
    tmp_path: pathlib.Path, linked: bool
) -> None:
    """A2: every file .pivotignore does not name still counts."""
    home, root = _project(tmp_path, _INPUT_PIPELINE)
    source = _source_dir(tmp_path, root, linked=linked)
    assert _ran(root, home) == {"p/lane": "ran"}

    (source / "grade.py").write_text("print('graded')\n")
    (source / "helper.txt").write_text("new file\n")

    assert _statuses(root, home) == {"p/lane": "stale"}
    assert _ran(root, home) == {"p/lane": "ran"}


def test_a_directory_output_holding_bytecode_restores_it_after_checkout(
    tmp_path: pathlib.Path,
) -> None:
    """A3: outputs are cached and restored whole; .pivotignore never applies to them."""
    home, root = _project(tmp_path, _OUTPUT_PIPELINE)
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "raw.txt").write_text("x = 1\n")
    assert _ran(root, home) == {"p/build": "ran", "p/use": "ran"}
    output = root / "data" / "p" / "build.dat"
    pyc = output / "__pycache__" / "mod.cpython-312.pyc"
    assert pyc.read_text() == "one"

    shutil.rmtree(output)
    _ok(helpers.run_pivot(root, home, "checkout", "p/build"))

    assert pyc.read_text() == "one", "the .pyc came back from the cache"
    assert (output / "mod.py").read_text() == "x = 1\n"
    assert _statuses(root, home) == {"p/build": "cached", "p/use": "cached"}


def test_a_tracked_directory_holding_bytecode_restores_it_after_checkout(
    tmp_path: pathlib.Path,
) -> None:
    """A3: `pivot track` records and restores every file, .pivotignore notwithstanding."""
    home, root = _project(tmp_path, _INPUT_PIPELINE)
    tracked = root / "data" / "raw" / "pkg"
    (tracked / "__pycache__").mkdir(parents=True)
    (tracked / "mod.py").write_text("x = 1\n")
    (tracked / "__pycache__" / "mod.cpython-312.pyc").write_bytes(b"\x00bytecode")
    _ok(helpers.run_pivot(root, home, "track", "data/raw/pkg"))
    pvt = (root / "data" / "raw" / "pkg.pvt").read_text()
    assert "__pycache__/mod.cpython-312.pyc" in pvt, pvt

    shutil.rmtree(tracked)
    _ok(helpers.run_pivot(root, home, "checkout", "data/raw/pkg"))

    assert (tracked / "__pycache__" / "mod.cpython-312.pyc").read_bytes() == b"\x00bytecode"
    assert (tracked / "mod.py").read_text() == "x = 1\n"


def test_a_downstream_stage_depends_on_a_directory_output_whole(tmp_path: pathlib.Path) -> None:
    """A4: a dependency on an upstream directory output is that output's whole hash. A stray
    .pyc in the output is removed by the upstream stage's restore, so nothing reruns; a .pyc
    the upstream stage itself writes differently reruns the consumer, as any output change does."""
    home, root = _project(tmp_path, _OUTPUT_PIPELINE)
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "raw.txt").write_text("x = 1\n")
    assert _ran(root, home) == {"p/build": "ran", "p/use": "ran"}
    output = root / "data" / "p" / "build.dat"

    (output / "__pycache__" / "stray.cpython-312.pyc").write_bytes(b"\x00stray")
    assert _statuses(root, home) == {"p/build": "cached", "p/use": "cached"}
    assert _ran(root, home) == {"p/build": "cached", "p/use": "cached"}
    assert not (output / "__pycache__" / "stray.cpython-312.pyc").exists(), (
        "the upstream restore put its output back as recorded"
    )

    (root / "params.yaml").write_text("p/build:\n  bytecode: two\n")
    assert _ran(root, home) == {"p/build": "ran", "p/use": "ran"}


def test_an_input_hashed_before_pivotignore_applied_reruns_once(tmp_path: pathlib.Path) -> None:
    """A lock written when every file counted (pivot 0.5.20, or no .pivotignore) reruns a stage
    whose input holds an ignored file exactly once; then the stage stays cached."""
    home, root = _project(tmp_path, _INPUT_PIPELINE)
    source = _source_dir(tmp_path, root, linked=False)
    (source / "__pycache__").mkdir()
    (source / "__pycache__" / "grade.cpython-312.pyc").write_bytes(b"\x00bytecode")
    (root / ".pivotignore").unlink()
    assert _ran(root, home) == {"p/lane": "ran"}

    (root / ".pivotignore").write_text(_PYC_IGNORE)

    assert _ran(root, home) == {"p/lane": "ran"}
    (source / "__pycache__" / "grade.cpython-312.pyc").write_bytes(b"\x00recompiled")
    assert _ran(root, home) == {"p/lane": "cached"}


def test_an_input_with_no_ignored_file_keeps_its_hash_when_pivotignore_applies(
    tmp_path: pathlib.Path,
) -> None:
    home, root = _project(tmp_path, _INPUT_PIPELINE)
    _source_dir(tmp_path, root, linked=False)
    (root / ".pivotignore").unlink()
    assert _ran(root, home) == {"p/lane": "ran"}

    (root / ".pivotignore").write_text(_PYC_IGNORE)

    assert _ran(root, home) == {"p/lane": "cached"}


def test_a_missing_tracked_input_is_judged_by_its_pvt_without_ignored_files(
    tmp_path: pathlib.Path,
) -> None:
    """`--allow-missing` compares a missing tracked input by its .pvt, which keeps every file;
    the files .pivotignore names are taken out first, so the stage is not called stale."""
    pipeline = _INPUT_PIPELINE.replace('path="tasks/server"', 'path="data/raw/server"')
    home, root = _project(tmp_path, pipeline)
    tracked = root / "data" / "raw" / "server"
    (tracked / "__pycache__").mkdir(parents=True)
    (tracked / "grade.py").write_text("print('grade')\n")
    (tracked / "__pycache__" / "grade.cpython-312.pyc").write_bytes(b"\x00bytecode")
    _ok(helpers.run_pivot(root, home, "track", "data/raw/server"))
    assert _ran(root, home) == {"p/lane": "ran"}

    shutil.rmtree(tracked)
    result = helpers.run_pivot(root, home, "repro", "--dry-run", "--allow-missing", "--json")

    assert result.returncode == 0, result.stdout + result.stderr
    plan = json.loads(result.stdout[result.stdout.index("{") :])
    assert plan["stages"] == {"p/lane": {"would_run": False, "reason": "unchanged"}}, plan
