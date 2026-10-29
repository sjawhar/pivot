"""Red tests from the PR #523 tester gate (head 282d7462): each pins one defect found by driving
the installed `pivot` CLI; and an upgrade from a released pivot's run cache.

D1 and the run-cache upgrade need uv (``PIVOT_TEST_UV``, as CI names it, else on PATH) and
network access to PyPI: they run released pivots (0.5.10 and 0.5.9 share main's lock reader;
0.5.2 the one before) against a project this pivot wrote, or that 0.5.10 wrote.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest
import yaml

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")
_REPO = pathlib.Path(__file__).resolve().parents[4]

# Valid for pivot 0.5.2, 0.5.9 and this head: a pathlib directory output, a dict-of-DataFrames
# directory output, a consumer of one file of it, and a file stage.
_PIPELINE = """\
import pathlib
import shutil
import tempfile
from typing import Annotated, TypedDict

import pandas as pd

from pivot import Pipeline, directory, loaders, metric, stage


class Copied(TypedDict):
    tree: pathlib.Path


@stage
def copy(corpus: pathlib.Path) -> Copied:
    out = pathlib.Path(tempfile.mkdtemp()) / "tree"
    shutil.copytree(corpus, out)
    return {"tree": out}


class Split(TypedDict):
    parts: Annotated[dict[str, pd.DataFrame], directory, loaders.CSV()]


@stage
def split(raw: pd.DataFrame) -> Split:
    return {"parts": {"train.csv": raw.iloc[:2], "test.csv": raw.iloc[2:]}}


@stage
def count(train: pd.DataFrame) -> Annotated[dict, metric]:
    return {"rows": int(len(train))}


@stage
def upper(raw: str) -> str:
    return raw.upper()


with Pipeline("p") as pipeline:
    copy(pipeline.input("corpus", t=pathlib.Path))
    count(split(pipeline.input("t.csv"))["parts"]["train.csv"])
    upper(pipeline.input("in.txt"))
"""

_TREE = pathlib.Path("data/p/copy/tree.dat")
_PARTS = pathlib.Path("data/p/split/parts")


def _env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}


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
    """The command that runs released pivot ``version``."""
    uv = os.environ.get("PIVOT_TEST_UV") or shutil.which("uv")
    assert uv is not None, "this test runs released pivots with `uv tool run`"
    return [uv, "tool", "run", "--quiet", "--from", f"pivot=={version}", "pivot"]


def _project(root: pathlib.Path, pivot: tuple[str, ...] = (_PIVOT,)) -> pathlib.Path:
    root.mkdir(parents=True)
    _git(root, "init", "-q", "-b", "main")
    _ok(_run(root, *pivot, "init"))
    (root / "pipeline.py").write_text(_PIPELINE)
    corpus = root / "data/raw/corpus"
    (corpus / "bin").mkdir(parents=True)
    (corpus / "empty").mkdir()
    (corpus / "a.txt").write_bytes(b"a")
    (corpus / "bin/run.sh").write_bytes(b"#!/bin/sh\n")
    (corpus / "bin/run.sh").chmod(0o755)
    (root / "data/raw/t.csv").write_text("a\n1\n2\n3\n4\n")
    (root / "data/raw/in.txt").write_text("hello\n")
    (root / ".gitignore").write_text("__pycache__/\n/data/p/\n")
    return root


# =============================================================================
# D1: an older pivot reads this head's directory record as a single file
# =============================================================================


@pytest.mark.parametrize("version", ["0.5.11", "0.5.10", "0.5.9", "0.5.2"])
def test_older_pivot_never_restores_a_listing_as_a_file(tmp_path: pathlib.Path, version: str):
    """A project this pivot wrote, opened by an older release (a downgrade, or a teammate who
    has not upgraded), must be refused or decided afresh, never answered wrongly. Today the
    older release reads `{hash, size, nfiles}` as a one-file hash and writes the directory's
    listing bytes as a regular file where the directory output belongs, exit 0."""
    old = _released(version)
    root = _project(tmp_path / "src")
    _ok(_run(root, _PIVOT, "repro"))
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "head wrote this")
    assert (root / _TREE).is_dir() and (root / _PARTS).is_dir()

    # A fresh clone that has the cache objects (what `pivot fetch` gives), then checkout.
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(root), str(clone))
    shutil.copytree(root / ".pivot/cache", clone / ".pivot/cache")
    proc = _run(clone, *old, "checkout")
    for out in (_TREE, _PARTS):
        assert not (clone / out).is_file(), (
            f"pivot {version} checkout (rc {proc.returncode}) wrote {out} as a file holding "
            + f"{(clone / out).read_bytes()[:60]!r}; {proc.stdout + proc.stderr}"
        )

    if version == "0.5.9":
        # main's skip check, with the directory outputs intact on disk.
        proc = _run(root, *old, "repro")
        for out in (_TREE, _PARTS):
            assert (root / out).is_dir(), (
                f"pivot {version} repro (rc {proc.returncode}) replaced the directory output "
                + f"{out} with a file; {proc.stdout + proc.stderr}"
            )


def test_a_lock_an_older_pivot_rewrote_is_read_without_the_listing_left_beside_it(
    tmp_path: pathlib.Path,
):
    """This pivot writes a project; a teammate on pivot 0.5.10 (main's lock reader) clones it,
    changes an input and reruns, writing locks in its format and leaving this pivot's listing
    file beside the directory stage's lock stale. That file is not the old lock's listing: it is
    never named as corrupt. Main's loaded-source fingerprint can safely rerun a consumer once as
    Code changed; after that one-time upgrade run, verify and repro settle cleanly."""
    src = _project(tmp_path / "src")
    _ok(_run(src, _PIVOT, "repro"))
    _git(src, "add", "-A")
    _git(src, "commit", "-qm", "head wrote this")
    root = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(src), str(root))
    (root / "data/raw/t.csv").write_text("a\n1\n20\n3\n4\n")
    _ok(_run(root, *_released("0.5.10"), "repro"))
    split_lock = yaml.safe_load((root / ".pivot/stages/p/split.lock").read_text())
    assert split_lock["schema_version"] == 2, split_lock
    assert (root / ".pivot/stages/p/split=parts.listing").is_file()

    initial_verify = _run(root, _PIVOT, "verify")
    repro = _run(root, _PIVOT, "repro")
    initial = initial_verify.stdout + initial_verify.stderr + repro.stdout + repro.stderr

    assert repro.returncode == 0, initial
    assert "does not hash" not in initial and "split=parts.listing" not in initial, initial
    assert "Code changed" in initial_verify.stdout, initial_verify.stdout
    assert "20" in (root / _PARTS / "train.csv").read_text(), "the output 0.5.10 wrote stays"

    verified = _run(root, _PIVOT, "verify")
    settled = _run(root, _PIVOT, "repro")
    for proc in (verified, settled):
        out = proc.stdout + proc.stderr
        assert proc.returncode == 0, out
        assert "does not hash" not in out and "split=parts.listing" not in out, out
    assert "Verification passed" in verified.stdout, verified.stdout
    assert "0 ran" in settled.stdout + settled.stderr, settled.stdout + settled.stderr


# =============================================================================
# D2: `pivot status` refuses a whole fresh clone over one committed listing
# =============================================================================


def test_status_reports_a_clone_whose_lock_was_put_back_without_its_listing(
    tmp_path: pathlib.Path,
):
    """`git checkout <rev> -- <lock>` without the listing beside it, committed: the case the
    docs name ("left behind by a lock put back without it"). In a fresh clone (no cache), base
    `pivot status` exits 0 and reports the stage stale. This head exits 1 and prints no status,
    because the one-file consumer's explanation reads the producer's listing."""
    root = _project(tmp_path / "src")
    _ok(_run(root, _PIVOT, "repro"))
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "c1")
    c1 = _git(root, "rev-parse", "HEAD").strip()
    (root / "data/raw/t.csv").write_text("a\n1\n2\n30\n40\n")
    _ok(_run(root, _PIVOT, "repro"))
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "c2")
    _git(root, "checkout", c1, "--", ".pivot/stages/p/split.lock")
    _git(root, "commit", "-qm", "put back split.lock")

    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(root), str(clone))
    for args in (("status",), ("status", "--json"), ("status", "--explain")):
        proc = _run(clone, _PIVOT, *args)
        out = proc.stdout + proc.stderr
        assert proc.returncode == 0, f"pivot {' '.join(args)}: {out}"
        assert "p/split" in out, out


# =============================================================================
# D3: the docs say only an old lock that lists a directory's files is read as missing
# =============================================================================


def test_docs_say_every_old_lock_is_read_as_missing(tmp_path: pathlib.Path):
    """A schema-2 lock that records only files (what pivot 0.5.9 writes for a file stage) is
    read as missing and its stage decided afresh, as the PR body's rollout says ("any
    schema_version other than 3"). docs/concepts/outputs.md limits that to "one that lists a
    directory's files inside it"."""
    root = _project(tmp_path / "src")
    _ok(_run(root, _PIVOT, "repro"))
    lock_path = root / ".pivot/stages/p/upper.lock"
    data = yaml.safe_load(lock_path.read_text())
    assert all("nfiles" not in entry for entry in data["deps"] + data["outs"])
    data["schema_version"] = 2
    lock_path.write_text(yaml.safe_dump(data, sort_keys=False))
    out = _ok(_run(root, _PIVOT, "status"))
    assert "upper.lock was written by an older pivot" in out, out

    docs = " ".join((_REPO / "docs/concepts/outputs.md").read_text().split())
    assert "(one that lists a directory's files inside it) is read as missing" not in docs, (
        "docs/concepts/outputs.md says only an old lock that lists a directory's files is read "
        + "as missing; every lock with another schema_version is"
    )


# =============================================================================
# An older pivot's run-cache entry restores after an upgrade
# =============================================================================


def test_a_run_cache_entry_an_older_pivot_filed_restores_after_an_upgrade(tmp_path: pathlib.Path):
    """Pivot 0.5.10 files a run-cache entry under the key without this pivot's directory marker,
    listing a directory output's files inline. After an upgrade, an input put back to a value the
    old pivot ran restores that stage from the old entry instead of running it."""
    old = _released("0.5.10")
    root = _project(tmp_path / "src", tuple(old))
    raw = root / "data/raw/t.csv"
    first_raw = raw.read_text()
    _ok(_run(root, *old, "repro"))
    first_parts = {path.name: path.read_bytes() for path in (root / _PARTS).iterdir()}
    raw.write_text("a\n1\n20\n3\n4\n")
    _ok(_run(root, *old, "repro"))
    raw.write_text(first_raw)

    repro = _run(root, _PIVOT, "repro", "--jsonl")

    assert repro.returncode == 0, repro.stdout + repro.stderr
    events = [json.loads(line) for line in repro.stdout.splitlines()]
    split = next(e for e in events if e["type"] == "stage_complete" and e["stage"] == "p/split")
    assert (split["status"], split["reason"]) == ("cached", "unchanged (run cache)"), split
    assert {path.name: path.read_bytes() for path in (root / _PARTS).iterdir()} == first_parts
    assert "Verification passed" in _ok(_run(root, _PIVOT, "verify"))
