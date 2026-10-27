"""Round-3 recheck of PR #523: the mixed-version rule, for an incremental output that is a file.

Written by the recheck (lane PR523RecheckR3) against the round-3 head `38234b96`. It is the
story `tests/cli/test_mixed_version_teams.py` already tells for the incremental **directory**
output `p/acc`, told again for the incremental **file** output `p/log`, whose lock this pivot
also writes in the `records:` shape because it records a directory dependency.

Each test drives installed CLIs: this checkout's pivot (``_PIVOT``) and a released pivot
(``uv tool run``; uv named by PIVOT_TEST_UV, else found on PATH).

Drop this file into ``packages/pivot/tests/cli/`` to run it.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys

import pytest

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")

_PIPELINE = """\
import dataclasses
import pathlib
import shutil
import tempfile
from typing import Annotated, TypedDict

from pivot import Pipeline, incremental, loaders, stage


class Copied(TypedDict):
    tree: pathlib.Path


@stage
def copy(corpus: pathlib.Path, seed: str) -> Copied:
    out = pathlib.Path(tempfile.mkdtemp()) / "tree"
    shutil.copytree(corpus, out)
    (out / "seed.txt").write_text(seed)
    return {"tree": out}


@stage
def tally(tree: pathlib.Path) -> dict:
    return {"seed": (tree / "seed.txt").read_text(), "a": (tree / "a.txt").read_text()}


def new_tree() -> pathlib.Path:
    tree = pathlib.Path(tempfile.mkdtemp()) / "tree"
    tree.mkdir()
    return tree


@dataclasses.dataclass(frozen=True)
class DirLoader(loaders.PathOnly):
    def empty(self) -> pathlib.Path:
        return new_tree()


@stage
def acc(raw: str, tree: pathlib.Path) -> Annotated[pathlib.Path, incremental, DirLoader()]:
    out = pathlib.Path(tempfile.mkdtemp()) / "tree"
    shutil.copytree(tree, out)
    n = len(list(out.iterdir()))
    (out / f"f{n:03d}.txt").write_text(raw)
    return out


@stage
def log(notes: pathlib.Path, previous: str) -> Annotated[str, incremental]:
    line = ",".join(f"{p.name}={p.read_text().strip()}" for p in sorted(notes.iterdir()))
    return previous + line + "\\n"


with Pipeline("p") as pipeline:
    copied = copy(pipeline.input("corpus", t=pathlib.Path), pipeline.input("seed.txt"))
    tally(copied["tree"])
    acc(pipeline.input("in.txt"), tree=pipeline.previous())
    log(pipeline.input("notes", t=pathlib.Path), pipeline.previous())
"""

_ACC = pathlib.Path("data/p/acc.dat")
_LOG = pathlib.Path("data/p/log.txt")
_LOG_LOCK = ".pivot/stages/p/log.lock"


def _env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env.update(extra)
    return env


def _run(cwd: pathlib.Path, *argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv), cwd=cwd, env=_env(), capture_output=True, text=True, timeout=600, check=False
    )


def _ok(proc: subprocess.CompletedProcess[str]) -> str:
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    return out


def _git(cwd: pathlib.Path, *args: str) -> str:
    return _ok(_run(cwd, "git", "-c", "user.email=t@t", "-c", "user.name=t", *args))


def _released(version: str) -> list[str]:
    uv = os.environ.get("PIVOT_TEST_UV") or shutil.which("uv")
    assert uv is not None, "this test runs released pivots with `uv tool run`"
    return [uv, "tool", "run", "--quiet", "--from", f"pivot=={version}", "pivot"]


def _set_inputs(root: pathlib.Path, tag: str) -> None:
    (root / "data/raw/in.txt").write_text(f"{tag}\n")
    (root / "data/raw/notes/n.txt").write_text(f"{tag}\n")


def _write_project(root: pathlib.Path) -> None:
    root.mkdir(parents=True)
    _git(root, "init", "-q", "-b", "main")
    (root / "pipeline.py").write_text(_PIPELINE)
    corpus = root / "data/raw/corpus"
    (corpus / "sub").mkdir(parents=True)
    (corpus / "a.txt").write_text("a0")
    (corpus / "sub/c.txt").write_text("c")
    (root / "assets/nested").mkdir(parents=True)
    (root / "assets/x.json").write_text('{"k": 0}\n')
    (root / "assets/nested/y.csv").write_text("a,b\n1,2\n")
    (root / "data/raw/seed.txt").write_text("seed0\n")
    (root / "data/raw/notes").mkdir()
    (root / "data/raw/notes/fixed.txt").write_text("keep\n")
    _set_inputs(root, "one")
    (root / ".gitignore").write_text("__pycache__/\n/data/p/\n/data/raw/corpus/\n/assets/\n")


def _project(root: pathlib.Path, pivot: list[str]) -> pathlib.Path:
    """A project built by ``pivot``: a tracked directory a stage reads (corpus), a directory
    output (copy), a whole-directory consumer (tally), an incremental directory output (acc),
    an incremental file output whose dependency is a directory (log), and a tracked directory
    no stage reads (assets). Built, rebuilt once with new inputs, and committed."""
    _write_project(root)
    _ok(_run(root, *pivot, "init"))
    _ok(_run(root, *pivot, "track", "data/raw/corpus"))
    _ok(_run(root, *pivot, "track", "assets"))
    _ok(_run(root, *pivot, "repro"))
    _set_inputs(root, "two")
    _ok(_run(root, *pivot, "repro"))
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "built")
    return root


def _clone_with_cache(src: pathlib.Path, clone: pathlib.Path) -> pathlib.Path:
    """A fresh clone holding the cache objects and both tracked directories: what a teammate
    has after `pivot fetch`, with nothing left for a tracked-file restore to do."""
    _git(src.parent, "clone", "-q", str(src), str(clone))
    shutil.copytree(src / ".pivot/cache", clone / ".pivot/cache")
    shutil.copytree(src / "data/raw/corpus", clone / "data/raw/corpus")
    shutil.copytree(src / "assets", clone / "assets")
    return clone


def _log_lines(root: pathlib.Path) -> list[str]:
    path = root / _LOG
    return path.read_text().splitlines() if path.is_file() else []


def _acc_values(root: pathlib.Path) -> list[str]:
    acc = root / _ACC
    return [p.read_text() for p in sorted(acc.iterdir())] if acc.is_dir() else []


@pytest.mark.parametrize("version", ["0.5.11", "0.3.11"])
def test_this_pivot_refuses_an_incremental_file_output_an_older_pivot_restarted(
    tmp_path: pathlib.Path, version: str
):
    """A teammate on an older pivot clones the project and runs `pivot repro` with new inputs.
    Their pivot stops at the incremental stages, and offers to remove the lock and rerun; they
    do that for `p/log`, whose incremental output is a file, so their pivot restarts the log
    from empty and they commit it.

    This pivot must not take that output as the stage's history: `status` reports the stage,
    `verify` fails it, and `repro` refuses, each naming `p/log`, why, and how to settle it -
    the same as for `p/acc`, whose incremental output is a directory. Both ways out work.
    """
    old = _released(version)
    src = _project(tmp_path / "src", [_PIVOT])
    assert _log_lines(src) == ["fixed.txt=keep,n.txt=one", "fixed.txt=keep,n.txt=two"]
    clone = _clone_with_cache(src, tmp_path / "clone")
    _run(clone, *old, "checkout")
    _set_inputs(clone, "three")
    stopped = _run(clone, *old, "repro")
    assert stopped.returncode != 0, stopped.stdout + stopped.stderr
    assert "Cache missing for incremental output 'p/log" in stopped.stdout + stopped.stderr

    (clone / _LOG_LOCK).unlink()  # what that message offers, for p/log only
    _ok(_run(clone, *old, "repro", "p/log"))
    assert _log_lines(clone) == ["fixed.txt=keep,n.txt=three"], "the older pivot restarted p/log"
    _git(clone, "add", "-A")
    _git(clone, "commit", "-qm", f"teammate on {version} restarted the incremental file output")

    status = _ok(_run(clone, _PIVOT, "status", "--explain", "p/log"))
    verified = _run(clone, _PIVOT, "verify", "p/log")
    refused = _run(clone, _PIVOT, "repro")

    for proc_out in (status, verified.stdout + verified.stderr, refused.stdout + refused.stderr):
        flat = " ".join(proc_out.split())
        assert "p/log" in flat, proc_out
        assert "an older pivot rewrote" in flat, proc_out
        assert "restarted the output from empty" in flat, proc_out
        assert "pivot repro --force p/log" in flat, proc_out
        assert "git checkout <commit> --" in flat, proc_out
    assert verified.returncode == 1, verified.stdout + verified.stderr
    assert refused.returncode == 1, refused.stdout + refused.stderr
    assert _log_lines(clone) == ["fixed.txt=keep,n.txt=three"], "the refused repro ran nothing"

    # Settle it by restoring the output from the commit before the older pivot's.
    restore = tmp_path / "restore"
    _git(tmp_path, "clone", "-q", str(clone), str(restore))
    shutil.copytree(clone / ".pivot/cache", restore / ".pivot/cache")
    shutil.copytree(clone / "data/raw/corpus", restore / "data/raw/corpus")
    shutil.copytree(clone / "assets", restore / "assets")
    _git(restore, "checkout", "HEAD~1", "--", _LOG_LOCK)
    _ok(_run(restore, _PIVOT, "checkout", "--force", "p/log"))
    assert _log_lines(restore) == [
        "fixed.txt=keep,n.txt=one",
        "fixed.txt=keep,n.txt=two",
    ]
    _ok(_run(restore, _PIVOT, "repro", "p/log"))
    assert _log_lines(restore) == [
        "fixed.txt=keep,n.txt=one",
        "fixed.txt=keep,n.txt=two",
        "fixed.txt=keep,n.txt=three",
    ]

    # Or rerun it from what the older pivot's lock records, on purpose.
    _ok(_run(clone, _PIVOT, "repro", "--force", "p/log"))
    assert _log_lines(clone) == ["fixed.txt=keep,n.txt=three", "fixed.txt=keep,n.txt=three"]
    assert "Verification passed" in _ok(_run(clone, _PIVOT, "verify", "p/log"))


@pytest.mark.parametrize("version", ["0.5.11", "0.3.11"])
def test_this_pivot_runs_a_project_an_older_pivot_built_without_refusing_it(
    tmp_path: pathlib.Path, version: str
):
    """The boundary the refusal must not cross: a project an older pivot built from scratch
    carries older-format locks for every stage, and this pivot reads them in place. Both
    incremental outputs keep what the older pivot accumulated, and nothing is refused.

    This test passes at the round-3 head; it is here so that a fix for the test above cannot
    be a blanket refusal of every older-format lock.
    """
    old = _released(version)
    src = _project(tmp_path / "src", old)
    assert _log_lines(src) == ["fixed.txt=keep,n.txt=one", "fixed.txt=keep,n.txt=two"]
    clone = _clone_with_cache(src, tmp_path / "clone")

    _ok(_run(clone, _PIVOT, "checkout"))
    _set_inputs(clone, "three")
    reproduced = _run(clone, _PIVOT, "repro")

    assert reproduced.returncode == 0, reproduced.stdout + reproduced.stderr
    assert _log_lines(clone) == [
        "fixed.txt=keep,n.txt=one",
        "fixed.txt=keep,n.txt=two",
        "fixed.txt=keep,n.txt=three",
    ]
    assert _acc_values(clone) == ["one\n", "two\n", "three\n"]
    assert "Verification passed" in _ok(_run(clone, _PIVOT, "verify"))
