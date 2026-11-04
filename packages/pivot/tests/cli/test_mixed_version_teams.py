"""A team that mixes this pivot with an older release on one project.

Each test drives installed CLIs: this checkout's pivot (``_PIVOT``) and a released pivot
(``uv tool run``; uv named by PIVOT_TEST_UV, else found on PATH). The story is the same in each:
this pivot builds a project, a teammate on an older release works in it, and this pivot reads
what the teammate left. From the PR #523 round-2 recheck (slice ``mix``); the base (#516), whose
lock reader is 0.5.10's, passes each of them.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import uuid

import boto3
import pytest

from tests import helpers

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
_TREE = pathlib.Path("data/p/copy/tree.dat")
_ACC_LOCK = ".pivot/stages/p/acc.lock"
_LOG_LOCK = ".pivot/stages/p/log.lock"
_ACC_LISTING = ".pivot/stages/p/acc=.listing"


def _env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env.update(extra)
    return env


def _run(
    cwd: pathlib.Path, *argv: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        cwd=cwd,
        env=env or _env(),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
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


def _project(root: pathlib.Path) -> pathlib.Path:
    """This pivot's project: a tracked directory read by a pathlib directory output (copy),
    a whole-directory consumer (tally), an incremental directory output (acc), an incremental
    file output that reads a pipeline input directory (log), and a tracked directory no stage
    reads (assets). Built, then rebuilt once with new inputs, and committed."""
    root.mkdir(parents=True)
    _git(root, "init", "-q", "-b", "main")
    _ok(_run(root, _PIVOT, "init"))
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
    _ok(_run(root, _PIVOT, "track", "data/raw/corpus"))
    _ok(_run(root, _PIVOT, "track", "assets"))
    _ok(_run(root, _PIVOT, "repro"))
    _set_inputs(root, "two")
    _ok(_run(root, _PIVOT, "repro"))
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "built by this pivot")
    return root


def _acc_values(root: pathlib.Path) -> list[str]:
    acc = root / _ACC
    return [p.read_text() for p in sorted(acc.iterdir())] if acc.is_dir() else []


def _clone_with_cache(src: pathlib.Path, clone: pathlib.Path) -> pathlib.Path:
    """A fresh clone holding the source's cache objects: what `pivot fetch` gives."""
    _git(src.parent, "clone", "-q", str(src), str(clone))
    shutil.copytree(src / ".pivot/cache", clone / ".pivot/cache")
    return clone


# =============================================================================
# An older pivot stops at an incremental stage instead of restarting it from empty
# =============================================================================


@pytest.mark.parametrize("version", ["0.5.11", "0.5.10", "0.5.9", "0.5.2"])
def test_an_older_pivot_stops_instead_of_restarting_an_incremental_output(
    tmp_path: pathlib.Path, version: str
):
    """A lock this pivot writes for a stage that records a directory keeps its records under
    `records`, with `deps` and `outs` empty. A released pivot reads that as a lock that records
    nothing: in a fresh clone (`pivot checkout`, then `pivot repro` with new inputs) it stops at
    the incremental stages acc and log, exit 1, instead of restarting them from empty, and
    rewrites neither lock. One of them refuses ("Cache missing for incremental output"); the
    other refuses too, or is blocked behind that failure, depending on which a worker reaches
    first."""
    old = _released(version)
    src = _project(tmp_path / "src")
    clone = _clone_with_cache(src, tmp_path / "clone")
    shutil.copytree(src / "data/raw/corpus", clone / "data/raw/corpus")
    _run(clone, *old, "checkout")
    _set_inputs(clone, "three")

    repro = _run(clone, *old, "repro")

    out = repro.stdout + repro.stderr
    assert repro.returncode != 0, out
    stops = {stage: helpers.incremental_stop(out, stage) for stage in ("p/acc", "p/log")}
    assert None not in stops.values(), (stops, out)
    assert "refused" in stops.values(), (stops, out)
    assert _acc_values(clone) == [], out
    assert not (clone / _LOG).exists(), out
    changed = _git(clone, "status", "--porcelain", "--", _ACC_LOCK, _LOG_LOCK)
    assert changed == "", changed


# =============================================================================
# This pivot refuses an incremental output an older pivot restarted from empty
# =============================================================================


def test_this_pivot_refuses_an_incremental_output_an_older_pivot_restarted(
    tmp_path: pathlib.Path,
):
    """A teammate on 0.5.10 clones the project and runs `pivot repro` with new inputs. Their
    pivot must append to what the team accumulated, or refuse; it refuses. If they then remove
    acc's lock, as its message offers, it restarts acc from empty, exit 0, and they commit. This
    pivot then refuses that output: `status` reports acc, `verify` fails, and `repro` refuses,
    each naming the stage, why, and the two ways to settle it; both work."""
    old = _released("0.5.10")
    src = _project(tmp_path / "src")
    assert _acc_values(src) == ["one\n", "two\n"]
    clone = _clone_with_cache(src, tmp_path / "clone")
    shutil.copytree(src / "data/raw/corpus", clone / "data/raw/corpus")
    shutil.copytree(src / "assets", clone / "assets")  # tracked; 0.5.10 skips this pivot's .pvt
    checkout = _run(clone, *old, "checkout")
    _set_inputs(clone, "three")
    repro = _run(clone, *old, "repro")
    out = checkout.stdout + checkout.stderr + repro.stdout + repro.stderr
    if repro.returncode == 0:
        assert _acc_values(clone) == ["one\n", "two\n", "three\n"], (
            f"0.5.10 repro exit 0 dropped the accumulated directory output: {_acc_values(clone)}; "
            + out
        )
        assert (clone / _LOG).read_text().count("\n") == 3, (
            f"0.5.10 repro exit 0 dropped the accumulated lines: {(clone / _LOG).read_text()!r}; "
            + out
        )
        return

    (clone / _ACC_LOCK).unlink()
    _ok(_run(clone, *old, "repro", "p/acc"))
    assert _acc_values(clone) == ["three\n"], "0.5.10 restarted acc from empty"
    _git(clone, "add", "-A")
    _git(clone, "commit", "-qm", "teammate on 0.5.10 restarted acc")
    restarted = _acc_values(clone)

    status = _ok(_run(clone, _PIVOT, "status", "--explain", "p/acc"))
    verified = _run(clone, _PIVOT, "verify")
    refused = _run(clone, _PIVOT, "repro")

    for proc_out in (status, verified.stdout + verified.stderr, refused.stdout + refused.stderr):
        flat = " ".join(proc_out.split())
        assert "p/acc" in flat, proc_out
        assert "an older pivot rewrote" in flat, proc_out
        assert "restarted the output from empty" in flat, proc_out
        assert "pivot repro --force p/acc" in flat, proc_out
        assert "git checkout <commit> --" in flat, proc_out
    assert verified.returncode == 1, verified.stdout + verified.stderr
    assert refused.returncode == 1, refused.stdout + refused.stderr
    assert _acc_values(clone) == restarted, "the refused repro ran nothing"

    # Settle it by restoring the output from the commit before the older pivot's.
    restore = tmp_path / "restore"
    _git(tmp_path, "clone", "-q", str(clone), str(restore))
    shutil.copytree(clone / ".pivot/cache", restore / ".pivot/cache")
    shutil.copytree(clone / "data/raw/corpus", restore / "data/raw/corpus")
    shutil.copytree(clone / "assets", restore / "assets")
    _git(restore, "checkout", "HEAD~1", "--", _ACC_LOCK, _ACC_LISTING)
    _ok(_run(restore, _PIVOT, "checkout", "--force", "p/acc"))
    assert _acc_values(restore) == ["one\n", "two\n"]
    _ok(_run(restore, _PIVOT, "repro", "p/acc"))
    assert _acc_values(restore) == ["one\n", "two\n", "three\n"]

    # Or rerun it from what the older pivot's lock records, on purpose.
    _ok(_run(clone, _PIVOT, "repro", "--force", "p/acc"))
    assert _acc_values(clone) == ["three\n", "three\n"]
    assert "Verification passed" in _ok(_run(clone, _PIVOT, "verify", "p/acc"))


# =============================================================================
# A .pvt 0.5.10 rewrote is read from its own list, not the .pvt.listing beside it
# =============================================================================


def _retracked_by_0_5_10(tmp_path: pathlib.Path) -> pathlib.Path:
    """This pivot tracks `assets` (no stage reads it); a teammate on 0.5.10 changes a file in
    it and re-tracks it (`track --force`, writing its own .pvt with the files inline), then
    commits. The listing this pivot wrote beside the .pvt is left as it was, and no lock holds
    the new directory inline."""
    old = _released("0.5.10")
    root = _project(tmp_path / "src")
    (root / "assets/x.json").unlink()
    (root / "assets/x.json").write_text('{"k": 1}\n')
    _ok(_run(root, *old, "track", "--force", "assets"))
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "teammate on 0.5.10 re-tracked assets")
    return root


def test_verify_reads_a_pvt_0_5_10_rewrote_not_the_listing_left_beside_it(tmp_path: pathlib.Path):
    """`pivot verify` after the teammate's re-track passes, as the base's does: the .pvt.listing
    beside a .pvt an older pivot wrote is not that .pvt's listing. It is named once, in the
    warning that says how to remove it."""
    root = _retracked_by_0_5_10(tmp_path)
    verified = _run(root, _PIVOT, "verify")
    out = verified.stdout + verified.stderr
    assert verified.returncode == 0, out
    flat = " ".join(out.split())
    assert flat.count("assets.pvt.listing") == 2, out  # the stale file, and `git rm` of it
    assert "git rm assets.pvt.listing" in flat, out
    removed = _run(root, "git", "rm", "-q", "assets.pvt.listing")
    assert removed.returncode == 0, removed.stderr
    out = _ok(_run(root, _PIVOT, "verify"))
    assert "assets.pvt.listing" not in out, out


def test_checkout_in_a_fresh_clone_restores_a_directory_0_5_10_retracked(tmp_path: pathlib.Path):
    """A fresh clone with the cache objects (what `pivot fetch` gives) restores the re-tracked
    directory from the .pvt the teammate wrote, as the base does."""
    root = _retracked_by_0_5_10(tmp_path)
    clone = _clone_with_cache(root, tmp_path / "clone")
    proc = _run(clone, _PIVOT, "checkout", "assets")
    out = proc.stdout + proc.stderr
    assert (clone / "assets/x.json").is_file(), out
    assert (clone / "assets/x.json").read_text() == '{"k": 1}\n', out


# =============================================================================
# A consumer's dependency listing is read beside its producer's own lock
# =============================================================================


def _producer_rerun_by_0_5_10(tmp_path: pathlib.Path) -> pathlib.Path:
    """In a clone of this pivot's project, a teammate on 0.5.10 changes the producer's input
    and runs only the producer (`pivot run p/copy`): copy's lock becomes 0.5.10's, the listing
    beside it stays the one this pivot committed, and tally's lock is still this pivot's. This
    pivot's `repro` in that clone then reruns tally, whose lock records the new tree.

    The teammate works in a clone because pivot 0.5.10 cannot read a StateDB this pivot or the
    base wrote (UnicodeDecodeError at both, unrelated to this format); a clone has none."""
    old = _released("0.5.10")
    src = _project(tmp_path / "src")
    root = _clone_with_cache(src, tmp_path / "work")
    shutil.copytree(src / "data/raw/corpus", root / "data/raw/corpus")
    shutil.copytree(src / "assets", root / "assets")
    (root / "data/raw/seed.txt").write_text("seed1\n")
    _ok(_run(root, *old, "run", "p/copy"))
    _ok(_run(root, _PIVOT, "repro"))
    return root


def test_verify_of_a_consumer_after_0_5_10_reran_only_its_producer(tmp_path: pathlib.Path):
    """`pivot verify` passes, as the base's does; the producer's stale listing is not the
    listing of the tree tally's lock records (that lock's producer is 0.5.10's)."""
    root = _producer_rerun_by_0_5_10(tmp_path)
    verified = _run(root, _PIVOT, "verify")
    out = verified.stdout + verified.stderr
    assert verified.returncode == 0, out
    assert "copy=tree.listing" not in out, out


def test_pull_of_a_consumer_after_0_5_10_reran_only_its_producer(
    tmp_path: pathlib.Path, moto_services: dict[str, str]
):
    """`pivot pull p/tally p/copy` in a fresh clone fetches both stages' outputs without a
    warning about the producer's listing, as the base does, and restores the tree the 0.5.10
    producer wrote."""
    root = _producer_rerun_by_0_5_10(tmp_path)
    bucket = f"mix-{uuid.uuid4().hex[:8]}"
    env = _env(
        AWS_ENDPOINT_URL=moto_services["s3"],
        AWS_ACCESS_KEY_ID="x",
        AWS_SECRET_ACCESS_KEY="x",
        AWS_DEFAULT_REGION="us-east-1",
    )
    boto3.client(
        "s3",
        endpoint_url=moto_services["s3"],
        region_name="us-east-1",
        aws_access_key_id="x",
        aws_secret_access_key="x",
    ).create_bucket(Bucket=bucket)  # pyright: ignore[reportAttributeAccessIssue] - boto3 ships no stubs
    for argv in (
        ("config", "set", "remotes.origin", f"s3://{bucket}/cache"),
        ("config", "set", "default_remote", "origin"),
        ("push",),
    ):
        _ok(_run(root, _PIVOT, *argv, env=env))
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "after")
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(root), str(clone))
    for argv in (
        ("config", "set", "remotes.origin", f"s3://{bucket}/cache"),
        ("config", "set", "default_remote", "origin"),
    ):
        _ok(_run(clone, _PIVOT, *argv, env=env))
    pulled = _ok(_run(clone, _PIVOT, "pull", "p/tally", "p/copy", env=env))
    assert "copy=tree.listing" not in pulled, pulled
    assert (clone / _TREE / "seed.txt").read_text() == "seed1\n", pulled
