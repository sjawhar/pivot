"""Red tests from the PR #523 tester gate (head 282d7462): each pins one defect found by driving
the installed `pivot` CLI; an upgrade from a released pivot's run cache; and an upgrade from the
locks pivot 0.10.4 wrote for stages that reach installed libraries.

D1 and the two upgrades need uv (``PIVOT_TEST_UV``, as CI names it, else on PATH) and network
access to PyPI: they run released pivots (0.5.10 and 0.5.9 share main's lock reader; 0.5.2 the
one before; 0.10.4 the release before the `dist:` keys) against a project this pivot wrote, or
that a released pivot wrote.
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

from tests import installed_dists

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


# =============================================================================
# An upgrade from 0.10.4 reruns only a stage that reaches a git or local-path library
# =============================================================================

# The libraries of the upgraded project: `uppath` installed from its directory, `upgit` from a
# commit of its git repository, `upidx` from a wheel as from a package index.
_UPPATH = """\
def value():
    return "path-1"
"""
_UPGIT = """\
def _greeting():
    return "git-1"


class Base:
    def greet(self):
        return _greeting()
"""
_UPIDX = """\
def label():
    return "index-label"


def _greeting():
    return "index-1"


class Base:
    def greet(self):
        return _greeting()
"""

# `p/plain` calls the standard library and `upidx.label`, read as a global; `p/path_global` calls
# `uppath.value`, read as a global; `p/git_base` and `p/index_base` each build a user class whose
# base is a library's `Base`, the one form by which they reach that library.
_UPGRADE_PIPELINE = """\
import json

import upgit
import upidx
import uppath

from pivot import Pipeline, stage


class GitGreeter(upgit.Base):
    pass


class IndexGreeter(upidx.Base):
    pass


@stage
def plain() -> str:
    return json.dumps([upidx.label()])


@stage
def path_global() -> str:
    return uppath.value()


@stage
def git_base() -> str:
    return GitGreeter().greet()


@stage
def index_base() -> str:
    return IndexGreeter().greet()


with Pipeline("p") as pipeline:
    plain()
    path_global()
    git_base()
    index_base()
"""

# What each stage of `_UPGRADE_PIPELINE` writes to `data/<stage>.txt`.
_UPGRADE_OUTPUTS = {
    "p/plain": '["index-label"]',
    "p/path_global": "path-1",
    "p/git_base": "git-1",
    "p/index_base": "index-1",
}


def test_an_upgrade_from_0_10_4_reruns_only_stages_that_reach_a_git_or_local_path_library(
    tmp_path: pathlib.Path,
):
    """Pivot 0.10.4 from PyPI, in a venv that holds the three libraries, runs the project's four
    stages and writes their locks. This pivot reads them from a second venv that holds the same
    libraries: `p/plain`, which reaches no git or local-path library, and `p/index_base`, which
    reaches the index library only through a base class, are cached with no code change (a
    `byname:` entry never reaches a lock); `p/path_global` and `p/git_base` are stale, each on its
    library's `dist:` key alone. This pivot's `pivot repro` runs those two only, and then every
    stage is cached."""
    home = tmp_path / "home"
    src = tmp_path / "src"
    head = installed_dists.make_venv(tmp_path / "venv-head")
    uppath = installed_dists.write_library(src, "uppath", "uppath", _UPPATH)
    upgit_requirement = installed_dists.init_git_requirement(
        installed_dists.write_library(src, "upgit", "upgit", _UPGIT), home, "upgit"
    )
    wheels = tmp_path / "wheels"
    installed_dists.build_wheel(
        head, installed_dists.write_library(src, "upidx", "upidx", _UPIDX), wheels
    )
    released = installed_dists.make_standalone_venv(
        tmp_path / "venv-0.10.4", "pivot==0.10.4", str(uppath), upgit_requirement
    )
    installed_dists.install_from_wheels(released, wheels, "upidx")
    version = installed_dists.installed(released, "pivot").version
    assert version == "0.10.4", f"the first venv must hold pivot 0.10.4, not {version}"
    old = installed_dists.make_project_in(tmp_path, released, _UPGRADE_PIPELINE)
    ran, shown = installed_dists.repro_statuses(old)
    assert ran == dict.fromkeys(_UPGRADE_OUTPUTS, "ran"), f"0.10.4 must run every stage:\n{shown}"
    written = {
        stage: (old.root / "data" / f"{stage}.txt").read_text() for stage in _UPGRADE_OUTPUTS
    }
    assert written == _UPGRADE_OUTPUTS, f"each stage must write what its library returns: {written}"
    installed_dists.install_directory(head, uppath)
    installed_dists.install(head, upgit_requirement)
    installed_dists.install_from_wheels(head, wheels, "upidx")
    path_url, git_url = (installed_dists.direct_url(head, name) for name in ("uppath", "upgit"))
    # PEP 610: a `dir_info` without `editable: true` is a path install that is not editable.
    assert path_url is not None and path_url["dir_info"].get("editable") is not True, (
        f"uppath must be installed from its path, not editable: direct_url.json is {path_url!r}"
    )
    assert git_url is not None and git_url["vcs_info"]["vcs"] == "git", (
        f"upgit must be installed from git: direct_url.json is {git_url!r}"
    )
    assert installed_dists.direct_url(head, "upidx") is None, "upidx must be an index install"
    new = installed_dists.Project(root=old.root, home=home, venv=head)

    statuses, status_shown = installed_dists.stage_statuses(new)
    changes, explain_shown = installed_dists.stage_code_changes(new)

    assert (statuses, changes) == (
        {
            "p/plain": "cached",
            "p/path_global": "stale",
            "p/git_base": "stale",
            "p/index_base": "cached",
        },
        {
            "p/plain": [],
            "p/path_global": ["dist:uppath"],
            "p/git_base": ["dist:upgit"],
            "p/index_base": [],
        },
    ), (
        "this pivot must read the locks 0.10.4 wrote as stale only for a stage that reaches a git "
        + f"or local-path library, on its `dist:` key alone:\n{status_shown}\n{explain_shown}"
    )
    reran, shown = installed_dists.repro_statuses(new)
    assert reran == {
        "p/plain": "cached",
        "p/path_global": "ran",
        "p/git_base": "ran",
        "p/index_base": "cached",
    }, f"this pivot's `pivot repro` must run p/path_global and p/git_base only:\n{shown}"
    settled, shown = installed_dists.stage_statuses(new)
    assert settled == dict.fromkeys(_UPGRADE_OUTPUTS, "cached"), (
        f"`pivot status` must report every stage cached after the run:\n{shown}"
    )
