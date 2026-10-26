"""The one-time lock format change, end to end: directories recorded by hash, with committed
listing files.

Each test drives the installed `pivot` binary in a `git init` project, as a user does.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import uuid

import boto3
import xxhash
import yaml

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")
_A = xxhash.xxh64(b"a").hexdigest()
_RUN = xxhash.xxh64(b"#!/bin/sh\n").hexdigest()
_TREE_LISTING = f"{_A} 1 - a.txt\n{_RUN} 10 x bin/run.sh\n- - - empty/\n".encode()
_OLD_FORMAT = "written by an older pivot"


def _env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env.update(extra)
    return env


def _pivot(
    cwd: pathlib.Path, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_PIVOT, *args],
        cwd=cwd,
        env=env or _env(),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def _ok(proc: subprocess.CompletedProcess[str]) -> str:
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    return out


def _project(root: pathlib.Path, files: dict[str, str]) -> pathlib.Path:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    _ok(_pivot(root, "init"))
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def _commit(cwd: pathlib.Path, message: str) -> None:
    subprocess.run(["git", "add", "-A"], cwd=cwd, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", message],
        cwd=cwd,
        check=True,
    )


def _git(cwd: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)


def _lock(root: pathlib.Path, stage: str) -> dict[str, object]:
    return yaml.safe_load((root / ".pivot" / "stages" / f"{stage}.lock").read_text())


def _helper_entry(
    root: pathlib.Path, stage: str, field: str, name: str | None
) -> dict[str, object]:
    """The lock ``deps`` entry produced by ``name``, or the ``outs`` entry keyed ``name``."""
    entries: list[dict[str, object]] = _lock(root, stage)[field]  # pyright: ignore[reportAssignmentType] - parsed YAML
    match = "producer" if field == "deps" else "key"
    return next(entry for entry in entries if entry[match] == name)


def _helper_write_tree(root: pathlib.Path) -> None:
    """A file, an executable file in a subdirectory, and an empty directory."""
    (root / "bin").mkdir(parents=True)
    (root / "empty").mkdir()
    (root / "a.txt").write_bytes(b"a")
    (root / "bin" / "run.sh").write_bytes(b"#!/bin/sh\n")
    (root / "bin" / "run.sh").chmod(0o755)


def _helper_assert_tree(root: pathlib.Path) -> None:
    assert (root / "a.txt").read_bytes() == b"a"
    assert (root / "bin" / "run.sh").stat().st_mode & 0o100, "the executable bit was lost"
    assert not (root / "a.txt").stat().st_mode & 0o100
    assert (root / "empty").is_dir(), "the empty directory was lost"


# =============================================================================
# A directory input and a directory output: hash, size, file count, committed listing
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


@stage
def count(tree: pathlib.Path) -> dict:
    return {"files": sum(1 for p in tree.rglob("*") if p.is_file())}


with Pipeline("t") as pipeline:
    copied = copy(pipeline.input("corpus", t=pathlib.Path))
    count(copied.tree)
"""

_INPUT_LISTING = pathlib.Path(".pivot/stages/=inputs/data%2Fraw%2Fcorpus.listing")
_OUTPUT_LISTING = pathlib.Path(".pivot/stages/t/copy=tree.listing")


def _helper_tree_project(root: pathlib.Path) -> pathlib.Path:
    _project(root, {"pipeline.py": _TREE_PIPELINE})
    _helper_write_tree(root / "data" / "raw" / "corpus")
    _ok(_pivot(root, "repro"))
    return root


def test_a_directory_is_recorded_by_hash_with_one_committed_listing_per_directory(
    tmp_path: pathlib.Path,
) -> None:
    root = _helper_tree_project(tmp_path)
    tree_hash = xxhash.xxh64(_TREE_LISTING).hexdigest()
    record = {"hash": tree_hash, "size": 11, "nfiles": 2}

    assert _helper_entry(root, "t/copy", "deps", "corpus") == {
        "producer": "corpus",
        "key": None,
        **record,
    }
    out = _helper_entry(root, "t/copy", "outs", "tree")
    assert {k: v for k, v in out.items() if k != "path"} == {"key": "tree", **record}
    assert _helper_entry(root, "t/count", "deps", "t/copy") == {
        "producer": "t/copy",
        "key": "tree",
        **record,
    }
    assert (root / _INPUT_LISTING).read_bytes() == _TREE_LISTING
    assert (root / _OUTPUT_LISTING).read_bytes() == _TREE_LISTING

    out_dir = root / str(out["path"])
    shutil.rmtree(out_dir)
    _ok(_pivot(root, "checkout"))
    _helper_assert_tree(out_dir)
    assert "Verification passed" in _ok(_pivot(root, "verify"))


def test_a_hand_edited_listing_is_refused_naming_it(tmp_path: pathlib.Path) -> None:
    """The listing's bytes must hash to the directory hash its lock records."""
    root = _helper_tree_project(tmp_path)
    listing_file = root / _OUTPUT_LISTING
    assert listing_file.is_file(), "the output's listing must be committed beside its lock"
    listing_file.write_bytes(listing_file.read_bytes().replace(b" 1 - a.txt", b" 2 - a.txt"))

    verified = _pivot(root, "verify")
    out_dir = root / str(_helper_entry(root, "t/copy", "outs", "tree")["path"])
    shutil.rmtree(out_dir)
    checked_out = _pivot(root, "checkout")

    for proc in (verified, checked_out):
        out = proc.stdout + proc.stderr
        assert proc.returncode != 0, out
        assert str(_OUTPUT_LISTING) in out, out
        assert "hash" in out, out
    assert not out_dir.exists(), "checkout must restore nothing from a refused listing"


# =============================================================================
# git diff after one file of a 1,000-file output changes
# =============================================================================

_SHARDS_PIPELINE = """\
import pathlib
import tempfile
from typing import TypedDict

from pivot import Pipeline, stage


class Sharded(TypedDict):
    shards: pathlib.Path


@stage
def shard(seed: str) -> Sharded:
    out = pathlib.Path(tempfile.mkdtemp()) / "shards"
    out.mkdir()
    for i in range(1000):
        (out / f"{i:04d}.txt").write_text(seed if i == 7 else str(i))
    return {"shards": out}


@stage
def count(shards: pathlib.Path) -> dict:
    return {"files": sum(1 for _ in shards.iterdir())}


with Pipeline("big") as pipeline:
    count(shard(pipeline.input("seed.txt")).shards)
"""


def test_one_changed_file_is_one_listing_line_and_no_lock_lists_files(
    tmp_path: pathlib.Path,
) -> None:
    root = _project(
        tmp_path,
        {
            "pipeline.py": _SHARDS_PIPELINE,
            "data/raw/seed.txt": "a",
            ".gitignore": "/data/big/\n__pycache__/\n",
        },
    )
    _ok(_pivot(root, "repro"))
    _commit(root, "v1")

    (root / "data" / "raw" / "seed.txt").write_text("b")
    _ok(_pivot(root, "repro"))
    numstat = {
        line.split("\t")[2]: (line.split("\t")[0], line.split("\t")[1])
        for line in _git(root, "diff", "--numstat", "--", ".pivot/stages").stdout.splitlines()
    }
    lock_diff = _git(root, "diff", "-U0", "--", "*.lock").stdout

    assert numstat.get(".pivot/stages/big/shard=shards.listing") == ("1", "1"), numstat
    assert "relpath" not in lock_diff, lock_diff
    assert all(int(added) <= 2 for path, (added, _) in numstat.items() if path.endswith(".lock"))


# =============================================================================
# A lock in the old format
# =============================================================================

_PLAIN_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def upper(raw: str) -> str:
    return raw.upper()


with Pipeline("p") as pipeline:
    upper(pipeline.input("in.txt"))
"""


def _helper_rewrite_as_schema_2(lock_path: pathlib.Path) -> None:
    """Rewrite a lock the way pivot 0.5.9 wrote it: schema 2."""
    data = yaml.safe_load(lock_path.read_text())
    data["schema_version"] = 2
    lock_path.write_text(yaml.safe_dump(data, sort_keys=False))


def test_an_ordinary_old_lock_warns_once_is_decided_afresh_and_is_rewritten(
    tmp_path: pathlib.Path,
) -> None:
    root = _project(tmp_path, {"pipeline.py": _PLAIN_PIPELINE, "data/raw/in.txt": "abc"})
    _ok(_pivot(root, "repro"))
    lock_path = root / ".pivot" / "stages" / "p" / "upper.lock"
    _helper_rewrite_as_schema_2(lock_path)

    status = _pivot(root, "status")
    assert status.returncode == 0, status.stdout + status.stderr
    assert status.stderr.count(_OLD_FORMAT) == 1, status.stderr
    assert "p/upper.lock" in status.stderr, status.stderr

    repro = _pivot(root, "repro")
    assert repro.returncode == 0, repro.stdout + repro.stderr
    assert (repro.stdout + repro.stderr).count(_OLD_FORMAT) == 1, repro.stdout + repro.stderr
    assert "cached" in repro.stdout or "ran" in repro.stdout, repro.stdout
    assert yaml.safe_load(lock_path.read_text())["schema_version"] == 3
    assert _OLD_FORMAT not in _ok(_pivot(root, "status"))


_INCREMENTAL_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, incremental, stage


@stage
def grow(raw: str, previous: str) -> Annotated[str, incremental]:
    return previous + raw


with Pipeline("p") as pipeline:
    grow(pipeline.input("in.txt"), pipeline.previous())
"""


def test_an_incremental_old_lock_warns_carries_its_output_over_once_and_is_rewritten(
    tmp_path: pathlib.Path,
) -> None:
    """Rerunning an incremental stage from nothing would drop its accumulated output, so its
    old lock still supplies the prior output, once."""
    root = _project(tmp_path, {"pipeline.py": _INCREMENTAL_PIPELINE, "data/raw/in.txt": "abc"})
    _ok(_pivot(root, "repro"))
    lock_path = root / ".pivot" / "stages" / "p" / "grow.lock"
    output = root / str(_helper_entry(root, "p/grow", "outs", None)["path"])
    assert output.read_text() == "abc"
    _helper_rewrite_as_schema_2(lock_path)

    status = _pivot(root, "status")
    assert status.returncode == 0, status.stdout + status.stderr
    assert status.stderr.count(_OLD_FORMAT) == 1, status.stderr

    repro = _pivot(root, "repro")
    out = repro.stdout + repro.stderr
    assert repro.returncode == 0, out
    assert out.count(_OLD_FORMAT) == 1, out
    assert "p/grow.lock" in out, out
    assert output.read_text() == "abcabc", "the incremental output must carry over"
    assert yaml.safe_load(lock_path.read_text())["schema_version"] == 3


# =============================================================================
# Where committed files live: gitignore, and names that cannot collide
# =============================================================================


def test_init_keeps_lock_and_listing_files_visible_under_common_root_ignore_patterns(
    tmp_path: pathlib.Path,
) -> None:
    """BET's root .gitignore ignores `build/`, `lib/` and `parts/` anywhere in the tree."""
    root = _project(tmp_path, {".gitignore": "build/\nlib/\nparts/\n"})
    committed = [
        ".pivot/stages/lib/s.lock",
        ".pivot/stages/build/s=parts.listing",
        ".pivot/stages/=inputs/data%2Fraw%2Fparts.listing",
    ]
    for rel in committed:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("x\n")

    ignored = [rel for rel in committed if _git(root, "check-ignore", "-q", rel).returncode == 0]

    assert ignored == [], f"git ignores committed pivot files: {ignored}"


_IGNORED_TREE_PIPELINE = """\
import pathlib
import tempfile
from typing import TypedDict

from pivot import Pipeline, stage


class Built(TypedDict):
    tree: pathlib.Path


@stage
def make() -> Built:
    tree = pathlib.Path(tempfile.mkdtemp()) / "tree"
    tree.mkdir()
    (tree / "a.txt").write_text("a")
    return {"tree": tree}


with Pipeline("lib") as pipeline:
    make()
"""


def test_repro_warns_when_git_ignores_a_lock_or_listing_it_wrote(tmp_path: pathlib.Path) -> None:
    """A project made before `!/stages/**` existed, under a root `lib/` pattern."""
    root = _project(tmp_path, {"pipeline.py": _IGNORED_TREE_PIPELINE, ".gitignore": "lib/\n"})
    gitignore = root / ".pivot" / ".gitignore"
    gitignore.write_text(gitignore.read_text().replace("!/stages/**\n", ""))

    repro = _pivot(root, "repro")

    out = repro.stdout + repro.stderr
    assert repro.returncode == 0, out
    assert ".pivot/stages/lib/make.lock" in out, out
    assert ".pivot/stages/lib/make=tree.listing" in out, out
    assert "!/stages/**" in out, out


_COLLIDE_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def a() -> str:
    return "a"


@stage
def b() -> str:
    return "b"


with Pipeline("p/a.lock") as other:
    b()

with Pipeline("p") as pipeline:
    a()
    pipeline.include(other)
"""


def test_a_stage_name_whose_part_ends_in_lock_is_listed_like_any_other_name(
    tmp_path: pathlib.Path,
) -> None:
    """Definition accepts `p/a.lock/b`; an actual lock-path collision is still an execution
    concern, as it was before listings existed."""
    root = _project(tmp_path, {"pipeline.py": _COLLIDE_PIPELINE})

    listed = _pivot(root, "list")

    out = listed.stdout + listed.stderr
    assert listed.returncode == 0, out
    assert "p/a" in out and "p/a.lock/b" in out, out


# =============================================================================
# A tracked directory holding a name that is not UTF-8
# =============================================================================


def test_a_tracked_directory_with_a_non_utf8_name_tracks_reports_clean_and_restores(
    tmp_path: pathlib.Path,
) -> None:
    root = _project(tmp_path, {"pipeline.py": _PLAIN_PIPELINE, "data/raw/in.txt": "abc"})
    corpus = root / "data" / "raw" / "corpus"
    corpus.mkdir()
    (corpus / os.fsdecode(b"\xff.bin")).write_bytes(b"odd")

    _ok(_pivot(root, "track", "data/raw/corpus"))
    listing_file = root / "data" / "raw" / "corpus.pvt.listing"
    status = json.loads(_pivot(root, "status", "--json").stdout)
    shutil.rmtree(corpus)
    _ok(_pivot(root, "checkout", "data/raw/corpus"))

    assert (
        listing_file.read_bytes()
        == f'{xxhash.xxh64(b"odd").hexdigest()} 3 - "\\377.bin"\n'.encode()
    )
    assert [entry["status"] for entry in status["tracked_files"]] == ["clean"], status
    assert os.listdir(os.fsencode(corpus)) == [b"\xff.bin"]


# =============================================================================
# push, then pull into a fresh clone
# =============================================================================

_MADE_PIPELINE = """\
import os
import pathlib
import tempfile
from typing import TypedDict

from pivot import Pipeline, stage


class Made(TypedDict):
    tree: pathlib.Path


@stage
def make() -> Made:
    tree = pathlib.Path(tempfile.mkdtemp()) / "tree"
    (tree / "bin").mkdir(parents=True)
    (tree / "empty").mkdir()
    (tree / "a.txt").write_bytes(b"a")
    (tree / "bin" / "run.sh").write_bytes(b"#!/bin/sh\\n")
    os.chmod(tree / "bin" / "run.sh", 0o755)
    return {"tree": tree}


@stage
def count(tree: pathlib.Path) -> dict:
    return {"files": sum(1 for p in tree.rglob("*") if p.is_file())}


with Pipeline("t") as pipeline:
    count(make().tree)
"""


def _helper_s3_env(endpoint: str) -> tuple[dict[str, str], str]:
    bucket = f"listing-{uuid.uuid4().hex[:8]}"
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name="us-east-1",
        aws_access_key_id="x",
        aws_secret_access_key="x",
    )
    s3.create_bucket(Bucket=bucket)  # pyright: ignore[reportAttributeAccessIssue] - boto3 ships no stubs
    env = _env(
        AWS_ENDPOINT_URL=endpoint,
        AWS_ACCESS_KEY_ID="x",
        AWS_SECRET_ACCESS_KEY="x",
        AWS_DEFAULT_REGION="us-east-1",
    )
    return env, bucket


def test_pull_into_a_fresh_clone_restores_a_directory_from_its_committed_listing(
    tmp_path: pathlib.Path, moto_services: dict[str, str]
) -> None:
    env, bucket = _helper_s3_env(moto_services["s3"])
    src = _project(
        tmp_path / "src", {"pipeline.py": _MADE_PIPELINE, ".gitignore": "/data/t/\n__pycache__/\n"}
    )
    _ok(_pivot(src, "config", "set", "remotes.origin", f"s3://{bucket}/cache"))
    _ok(_pivot(src, "config", "set", "default_remote", "origin"))
    _ok(_pivot(src, "repro", env=env))
    _ok(_pivot(src, "push", env=env))
    _commit(src, "source")
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(src), str(clone)], check=True)

    committed = _git(clone, "ls-files", ".pivot/stages").stdout.splitlines()
    assert ".pivot/stages/t/make=tree.listing" in committed, committed
    _ok(_pivot(clone, "pull", env=env))

    out_dir = clone / str(_helper_entry(clone, "t/make", "outs", "tree")["path"])
    _helper_assert_tree(out_dir)
    assert "2 cached" in _ok(_pivot(clone, "status", env=env))
    assert "Verification passed" in _ok(_pivot(clone, "verify", env=env))


# =============================================================================
# A fresh clone, before any pull: one named file of a directory output (BET-58, PR #489)
# =============================================================================

_ONE_FILE_PIPELINE = """\
from typing import Annotated, TypedDict

import pandas as pd

from pivot import Pipeline, directory, loaders, metric, stage


class Split(TypedDict):
    parts: Annotated[dict[str, pd.DataFrame], directory, loaders.CSV()]


@stage
def split(raw: pd.DataFrame) -> Split:
    return {"parts": {"train.csv": raw.iloc[:2], "test.csv": raw.iloc[2:]}}


@stage
def count(train: pd.DataFrame) -> Annotated[dict, metric]:
    return {"rows": int(len(train))}


with Pipeline("s") as pipeline:
    count(split(pipeline.input("t.csv"))["parts"]["train.csv"])
"""


def test_status_in_a_fresh_clone_reports_a_one_file_consumer_up_to_date_before_pull(
    tmp_path: pathlib.Path,
) -> None:
    """The consumer's metric is committed and its one input file's hash comes from the
    committed listing, so with no cache at all it has no change of its own: it is stale only
    because its producer's outputs are missing, and `pivot pull` restores both."""
    src = _project(
        tmp_path / "src",
        {
            "pipeline.py": _ONE_FILE_PIPELINE,
            "data/raw/t.csv": "a\n1\n2\n3\n",
            ".gitignore": "/data/s/split/\n__pycache__/\n",
        },
    )
    _ok(_pivot(src, "repro"))
    _commit(src, "source")
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(src), str(clone)], check=True)
    assert (clone / ".pivot" / "stages" / "s" / "split=parts.listing").is_file()

    explained = _pivot(clone, "status", "--explain", "--json")
    assert explained.returncode == 0, explained.stdout + explained.stderr
    status = json.loads(explained.stdout)

    by_name = {stage["name"]: stage for stage in status["stages"]}
    count = by_name["s/count"]
    assert by_name["s/split"]["reason"] == "Outputs missing from cache", by_name
    assert count["reason"] == "Upstream stale (s/split)", count
    assert (count["code_changes"], count["param_changes"], count["dep_changes"]) == ([], [], [])


# =============================================================================
# A directory tracked by the previous release: its files listed inside the .pvt
# =============================================================================

_TRACKED_INPUT_PIPELINE = """\
import pathlib

from pivot import Pipeline, stage


@stage
def count(corpus: pathlib.Path) -> dict:
    return {"files": sum(1 for p in corpus.rglob("*") if p.is_file())}


with Pipeline("k") as pipeline:
    count(pipeline.input("corpus", t=pathlib.Path))
"""


# The members of `_helper_write_tree` as pivot 0.5.9 listed them inline in a lock or `.pvt`.
_TREE_MANIFEST: list[dict[str, object]] = [
    {"relpath": "a.txt", "hash": _A, "size": 1, "isexec": False},
    {"relpath": "bin/run.sh", "hash": _RUN, "size": 10, "isexec": True},
]
_TREE_EMPTY_DIRS = ["empty"]
_TREE_HASH_0_5_9 = xxhash.xxh64(
    json.dumps(
        {"empty_dirs": _TREE_EMPTY_DIRS, "manifest": _TREE_MANIFEST},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()


def _helper_rewrite_pvt_as_0_5_9(root: pathlib.Path, rel: str) -> str:
    """Rewrite the .pvt of a tracked `_helper_write_tree` the way pivot 0.5.9 wrote it, and
    return its text: the files listed inline and hashed as JSON, with no .pvt.listing file
    beside it and no listing object in the cache."""
    pvt_path = root / f"{rel}.pvt"
    data = yaml.safe_load(pvt_path.read_text())
    old = {
        "path": data["path"],
        "hash": _TREE_HASH_0_5_9,
        "size": data["size"],
        "num_files": len(_TREE_MANIFEST),
        "manifest": _TREE_MANIFEST,
        "empty_dirs": _TREE_EMPTY_DIRS,
    }
    text = yaml.safe_dump(old, sort_keys=False)
    pvt_path.write_text(text)
    (root / f"{rel}.pvt.listing").unlink()
    listing_object = root / ".pivot" / "cache" / "files" / data["hash"][:2] / data["hash"][2:]
    listing_object.unlink()
    return text


def test_a_directory_tracked_by_the_previous_release_is_seen_with_its_files_and_a_warning(
    tmp_path: pathlib.Path,
) -> None:
    """Pivot never rewrites a user's committed .pvt: it reads the files the old .pvt lists as
    the directory's listing, on every run, and says how to convert it."""
    root = _project(tmp_path, {"pipeline.py": _TRACKED_INPUT_PIPELINE})
    corpus = root / "data" / "raw" / "corpus"
    _helper_write_tree(corpus)
    _ok(_pivot(root, "track", "data/raw/corpus"))
    old_text = _helper_rewrite_pvt_as_0_5_9(root, "data/raw/corpus")

    status = _pivot(root, "status", "--json")
    shutil.rmtree(corpus)
    checkout = _pivot(root, "checkout", "data/raw/corpus")
    restored_tree = corpus.is_dir() and (corpus / "bin" / "run.sh").is_file()
    repro = _pivot(root, "repro")

    for proc in (status, checkout, repro):
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert proc.stderr.count(_OLD_FORMAT) == 1, proc.stderr
        assert "corpus.pvt" in proc.stderr and "pivot track --force" in proc.stderr, proc.stderr
    assert [entry["status"] for entry in json.loads(status.stdout)["tracked_files"]] == ["clean"], (
        status.stdout
    )
    assert restored_tree, checkout.stdout + checkout.stderr
    _helper_assert_tree(corpus)
    assert yaml.safe_load((root / "data" / "k" / "count.yaml").read_text()) == {"files": 2}
    assert (root / "data" / "raw" / "corpus.pvt").read_text() == old_text, "never rewritten"
    assert not (root / "data" / "raw" / "corpus.pvt.listing").exists()


# =============================================================================
# A past revision whose locks an older pivot wrote
# =============================================================================


def _helper_old_directory_entry(entry: dict[str, object]) -> dict[str, object]:
    """A schema-3 entry for a `_helper_write_tree` directory as pivot 0.5.9 wrote it: its
    files listed inline and the directory hashed as JSON."""
    old = {k: v for k, v in entry.items() if k not in ("size", "nfiles")}
    old["hash"] = _TREE_HASH_0_5_9
    old["manifest"] = _TREE_MANIFEST
    old["empty_dirs"] = _TREE_EMPTY_DIRS
    return old


def test_get_restores_a_directory_from_a_revision_whose_lock_an_older_pivot_wrote(
    tmp_path: pathlib.Path,
) -> None:
    """A repository's history keeps the lock format each revision was written in, so `pivot
    get --rev` reads an older pivot's lock whole, the directory's inline files included."""
    root = _helper_tree_project(tmp_path)
    lock_path = root / ".pivot" / "stages" / "t" / "copy.lock"
    data = yaml.safe_load(lock_path.read_text())
    data["schema_version"] = 2
    data["deps"] = [_helper_old_directory_entry(entry) for entry in data["deps"]]
    data["outs"] = [_helper_old_directory_entry(entry) for entry in data["outs"]]
    lock_path.write_text(yaml.safe_dump(data, sort_keys=False))
    (root / _OUTPUT_LISTING).unlink()
    (root / ".gitignore").write_text("/data/t/\n__pycache__/\n")
    _commit(root, "written by pivot 0.5.9")
    out_dir = root / str(data["outs"][0]["path"])
    shutil.rmtree(out_dir)

    got = _pivot(root, "get", "t/copy", "--rev", "HEAD", "--checkout-mode", "copy")

    assert got.returncode == 0, got.stdout + got.stderr
    _helper_assert_tree(out_dir)
