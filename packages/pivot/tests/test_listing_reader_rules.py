"""Rules every reader of a committed listing keeps, each pinned by one test: `pivot get --rev`
finds a long-named tracked directory's listing under its short name, a committed listing that
cannot be read is named and passed over for the cache's copy, and `pivot verify <stage>` refuses
the stale listing of a tracked directory the stage reads. From the PR #523 round-2 recheck
(slice ``mut``): the code was right, and no test held it.
"""

from __future__ import annotations

import logging
import os
import pathlib
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from pivot import exceptions, project
from pivot.storage import listing, restore, track
from pivot.types import DirManifestEntry

if TYPE_CHECKING:
    from collections.abc import Callable

    GitRepo = tuple[pathlib.Path, Callable[[str], str]]

_FIRST = "abcd123400000000"
_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")


def _helper_members() -> tuple[listing.DirListing, bytes]:
    members = listing.DirListing(
        files=[DirManifestEntry(relpath="a.txt", hash=_FIRST, size=4, isexec=False)],
        empty_dirs=[],
    )
    return members, listing.encode(members)


# =============================================================================
# `get --rev` reads a long-named tracked directory's listing by its short name
# =============================================================================


def test_get_rev_reads_the_short_named_listing_of_a_long_named_tracked_directory(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, git_repo: GitRepo
) -> None:
    """B6: `pivot get --rev` of a tracked directory whose `<name>.pvt.listing` would pass 255
    bytes reads `.~<xxh64>.pvt.listing` at that revision (`restore.resolve_targets`), so a clone
    with no cache copy of the listing still finds the directory's files."""
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    root, commit = git_repo
    name = "d" * 248  # `<name>.pvt` fits NAME_MAX; `<name>.pvt.listing` does not
    members, data = _helper_members()
    record = listing.record_of(members, data)
    pvt_path = root / f"{name}.pvt"
    track.write_pvt_file(
        pvt_path,
        track.PvtData(path=name, hash=record["hash"], size=record["size"], nfiles=record["nfiles"]),
    )
    committed = listing.tracked_listing_path(pvt_path)
    assert committed.name.startswith(".~"), committed
    listing.write(committed, data)
    commit("track a directory with a long name")

    (target,) = restore.resolve_targets([name], "HEAD", {}, cache_dir=tmp_path / "no-cache")

    assert target["members"] == {name: members}


# =============================================================================
# An unreadable committed listing is named and passed over for the cache's copy
# =============================================================================


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a file whatever its mode")
def test_an_unreadable_committed_listing_is_named_and_passed_over_for_the_cache_copy(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """N1 (round 2): a committed listing that cannot be read is treated as one that does not
    hash to its record: the cache's copy is taken with a warning naming the file, and without
    that copy the read is refused naming the file, never a bare PermissionError."""
    members, data = _helper_members()
    record = listing.record_of(members, data)
    cache_dir = tmp_path / "cache"
    listing.store_object(cache_dir, record["hash"], data)
    committed = tmp_path / "p=out.listing"
    committed.write_bytes(data)
    committed.chmod(0)
    try:
        with caplog.at_level(logging.WARNING):
            from_cache = listing.read(committed, record, cache_dir)
        with pytest.raises(exceptions.CorruptListingError, match=r"p=out\.listing: cannot be read"):
            listing.read(committed, record, None)
    finally:
        committed.chmod(0o644)

    assert from_cache == members
    assert "p=out.listing: cannot be read" in caplog.text


# =============================================================================
# `pivot verify <stage>` refuses the stale listing of a tracked directory it reads
# =============================================================================

_TREE_PIPELINE = """\
import pathlib
import shutil
import tempfile
from typing import TypedDict

from pivot import Pipeline, stage


class Copied(TypedDict):
    tree: pathlib.Path


@stage
def copy(corpus: pathlib.Path) -> Copied:
    out = pathlib.Path(tempfile.mkdtemp()) / "tree"
    shutil.copytree(corpus, out)
    return {"tree": out}


with Pipeline("t") as pipeline:
    copy(pipeline.input("corpus", t=pathlib.Path))
"""


def _helper_pivot(cwd: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    return subprocess.run(
        [_PIVOT, *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=300, check=False
    )


def _helper_ok(proc: subprocess.CompletedProcess[str]) -> str:
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    return out


def test_verify_of_a_stage_refuses_the_stale_listing_of_a_tracked_directory_it_reads(
    tmp_path: pathlib.Path,
) -> None:
    """PR body: `pivot verify` refuses a committed listing that does not hash to its record,
    for an output, an input or a tracked directory. `pivot verify <stage>` checks the
    `.pvt.listing` of a tracked directory the stage reads (cli/verify.py
    `_get_stage_lock_hashes`), naming it, though every other listing is right."""
    root = tmp_path / "proj"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    _helper_ok(_helper_pivot(root, "init"))
    (root / "pipeline.py").write_text(_TREE_PIPELINE)
    corpus = root / "data" / "raw" / "corpus"
    (corpus / "bin").mkdir(parents=True)
    (corpus / "a.txt").write_bytes(b"a")
    (corpus / "bin" / "run.sh").write_bytes(b"#!/bin/sh\n")
    _helper_ok(_helper_pivot(root, "track", "data/raw/corpus"))
    _helper_ok(_helper_pivot(root, "repro"))
    assert "Verification passed" in _helper_ok(_helper_pivot(root, "verify", "t/copy"))
    committed = root / "data" / "raw" / "corpus.pvt.listing"
    good = committed.read_bytes()
    committed.write_bytes(good.splitlines(keepends=True)[0])

    verified = _helper_pivot(root, "verify", "t/copy")

    out = verified.stdout + verified.stderr
    assert verified.returncode == 1, out
    assert "corpus.pvt.listing" in out and "its bytes hash to" in out, out
    assert "Traceback" not in out, out
