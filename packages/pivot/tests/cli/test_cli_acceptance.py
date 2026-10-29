"""End-to-end CLI acceptance tests: drive the installed `pivot` binary in a fresh project.

Each test runs the binary from the test interpreter's venv in a `git init` project,
the way a user does, so it covers discovery, config and the real command wiring.
"""

from __future__ import annotations

import contextlib
import errno
import json
import os
import pathlib
import py_compile
import re
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import sysconfig
import time
import urllib.request
import uuid
from typing import TYPE_CHECKING, Protocol, cast

import boto3
import pytest
import xxhash
import yaml

from pivot import loaders, types
from pivot.remote import keys
from pivot.storage import cache, gc, listing, state
from tests import helpers

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")


def _env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env.update(extra)
    return env


def _pivot(
    cwd: pathlib.Path, *args: str, env: dict[str, str] | None = None, executable: str = _PIVOT
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [executable, *args],
        cwd=cwd,
        env=env or _env(),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def _project(tmp_path: pathlib.Path, files: dict[str, str]) -> pathlib.Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    init = _pivot(tmp_path, "init")
    assert init.returncode == 0, init.stdout + init.stderr
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return tmp_path


def _ok(proc: subprocess.CompletedProcess[str]) -> str:
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    return out


def _helper_output_listing(
    root: pathlib.Path, stage: str, key: str | None = None
) -> listing.DirListing:
    """The committed listing of ``stage``'s directory output ``key``, read from the file beside
    its lock (never the cache) and checked against the hash, size and file count the lock
    records."""
    stages = root / ".pivot" / "stages"
    lock_data = yaml.safe_load((stages / f"{stage}.lock").read_text())
    (out,) = [entry for entry in lock_data["records"]["outs"] if entry["key"] == key]
    record = types.DirHash(hash=out["listing"], size=out["size"], nfiles=out["nfiles"])
    return listing.read(stages / f"{stage}={key or ''}.listing", record, None)


def _helper_tracked_listing(root: pathlib.Path, rel_path: str) -> listing.DirListing:
    """The committed ``<path>.pvt.listing`` of a tracked directory, checked against its .pvt."""
    pvt = yaml.safe_load((root / f"{rel_path}.pvt").read_text())
    record = types.DirHash(hash=pvt["hash"], size=pvt["size"], nfiles=pvt["nfiles"])
    return listing.read(root / f"{rel_path}.pvt.listing", record, None)


# A raw input and a metric dependency are both local; neither is ever in the remote.
# The remote endpoint is unreachable, so the test also pins that nothing local is looked up remotely.
_VERIFY_PIPELINE = """\
from typing import Annotated

import pandas as pd

from pivot import Pipeline, metric, stage


@stage
def summarize(raw: pd.DataFrame) -> Annotated[dict, metric]:
    return {"rows": int(len(raw))}


@stage
def consume(stats: dict) -> dict:
    return {"seen": stats["rows"]}


with Pipeline("v") as pipeline:
    consume(summarize(pipeline.input("t.csv")))
"""


def test_verify_allow_missing_passes_when_everything_is_local(tmp_path: pathlib.Path) -> None:
    root = _project(tmp_path, {"pipeline.py": _VERIFY_PIPELINE, "data/raw/t.csv": "a\n1\n2\n"})
    _ok(_pivot(root, "repro"))
    _ok(_pivot(root, "config", "set", "remotes.origin", "s3://unused-bucket/cache"))
    _ok(_pivot(root, "config", "set", "default_remote", "origin"))
    _ok(_pivot(root, "config", "set", "remote.retries", "0"))
    _ok(_pivot(root, "config", "set", "remote.connect_timeout", "1"))
    _ok(_pivot(root, "verify"))

    env = _env(
        AWS_ENDPOINT_URL="http://127.0.0.1:9",
        AWS_ACCESS_KEY_ID="x",
        AWS_SECRET_ACCESS_KEY="x",
        AWS_DEFAULT_REGION="us-east-1",
    )
    proc = _pivot(root, "verify", "--allow-missing", env=env)
    out = proc.stdout + proc.stderr

    assert proc.returncode == 0, out
    assert "Missing files" not in out, out


_CLONE_PIPELINE = """\
import pandas as pd

from pivot import Pipeline, stage


@stage
def double(raw: pd.DataFrame) -> pd.DataFrame:
    return raw * 2


with Pipeline("c") as pipeline:
    double(pipeline.input("t.csv"))
"""

_DIRECTORY_ARTIFACT_SYMLINK_PIPELINE = """\
import pathlib
import tempfile

from pivot import Pipeline, stage


@stage
def assemble() -> pathlib.Path:
    output = pathlib.Path(tempfile.mkdtemp(prefix="pivot-directory-artifact-"))
    target = output / "target.txt"
    target.write_text("target")
    (output / "link.txt").symlink_to(target)
    return output


with Pipeline("directory") as pipeline:
    assemble()
"""


def test_repro_warns_about_a_symlink_left_out_of_a_directory_artifact(
    tmp_path: pathlib.Path,
) -> None:
    """A user symlink is left out of a directory output, with a warning under the stage."""
    root = _project(tmp_path, {"pipeline.py": _DIRECTORY_ARTIFACT_SYMLINK_PIPELINE})
    output = root / "data" / "directory" / "assemble.dat"

    out = _ok(_pivot(root, "repro"))

    warning = (
        "WARNING: Directory artifact data/directory/assemble.dat leaves out symlink link.txt "
        + "(a symlink inside a directory artifact is not stored); "
        + "replace it with a real file or directory"
    )
    assert warning in " ".join(out.split()), out
    members = _helper_output_listing(root, "directory/assemble")
    assert [entry["relpath"] for entry in members["files"]] == ["target.txt"]

    shutil.rmtree(output)
    assert "0 ran, 1 cached" in _ok(_pivot(root, "repro"))
    assert (output / "target.txt").read_text() == "target"
    assert not (output / "link.txt").exists() and not (output / "link.txt").is_symlink()


def test_track_warns_about_a_symlink_left_out_of_a_tracked_directory(
    tmp_path: pathlib.Path,
) -> None:
    """Tracking a directory with a user symlink records the regular files and warns."""
    root = _project(tmp_path, {"data/tracked/keep.txt": "keep"})
    (root / "data" / "tracked" / "link.txt").symlink_to(root / "data" / "tracked" / "keep.txt")

    out = _ok(_pivot(root, "track", "data/tracked"))

    assert (
        "Directory artifact data/tracked leaves out symlink link.txt "
        + "(a symlink inside a directory artifact is not stored); "
        + "replace it with a real file or directory"
    ) in " ".join(out.split()), out
    members = _helper_tracked_listing(root, "data/tracked")
    assert [entry["relpath"] for entry in members["files"]] == ["keep.txt"]


def test_quiet_track_and_commit_do_not_print_the_left_out_symlink_warning(
    tmp_path: pathlib.Path,
) -> None:
    """-q silences the record-time warning for track and commit, as it does for repro."""
    root = _project(
        tmp_path,
        {"pipeline.py": _DIRECTORY_ARTIFACT_SYMLINK_PIPELINE, "data/tracked/keep.txt": "keep"},
    )
    (root / "data" / "tracked" / "link.txt").symlink_to(root / "data" / "tracked" / "keep.txt")
    _ok(_pivot(root, "track", "data/tracked"))
    _ok(_pivot(root, "repro"))

    for args in (("track", "--force", "data/tracked"), ("commit", "directory/assemble")):
        quiet = _pivot(root, "-q", *args)
        assert quiet.returncode == 0, quiet.stdout + quiet.stderr
        assert quiet.stderr == "", args

        loud = _ok(_pivot(root, *args))
        assert "leaves out symlink link.txt" in " ".join(loud.split()), args


_FORCE_CHECKOUT_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def produce() -> str:
    return "cached"


with Pipeline("checkout") as pipeline:
    produce()
"""


def test_checkout_force_preserves_existing_output_when_its_cache_object_is_missing(
    tmp_path: pathlib.Path,
) -> None:
    """A cache miss is reported before --force can delete the workspace output."""
    root = _project(tmp_path, {"pipeline.py": _FORCE_CHECKOUT_PIPELINE})
    _ok(_pivot(root, "config", "set", "cache.checkout_mode", "copy"))
    _ok(_pivot(root, "repro"))

    output = root / "data" / "checkout" / "produce.txt"
    output.write_text("workspace edit")
    lock_data = yaml.safe_load(
        (root / ".pivot" / "stages" / "checkout" / "produce.lock").read_text()
    )
    output_hash = lock_data["outs"][0]["hash"]
    (root / ".pivot" / "cache" / "files" / output_hash[:2] / output_hash[2:]).unlink()

    proc = _pivot(root, "checkout", "--force", "checkout/produce")
    out = proc.stdout + proc.stderr

    assert proc.returncode == 1, out
    assert output_hash in out
    assert "cache object" in out.lower()
    assert output.read_text() == "workspace edit"


def test_get_killed_before_fallback_replacement_preserves_existing_file(
    tmp_path: pathlib.Path,
) -> None:
    """A killed cache-miss get leaves the prior target intact until atomic replacement."""
    root = _project(tmp_path, {"source.txt": "revision content"})
    subprocess.run(["git", "add", "source.txt"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "initial"],
        cwd=root,
        check=True,
    )
    target = root / "source.txt"
    target.write_text("workspace content")
    hook_dir = root / "hook"
    hook_dir.mkdir()
    ready = root / "ready"
    hook_dir.joinpath("sitecustomize.py").write_text("""\
import os
import pathlib
import time

_target = pathlib.Path(os.environ["PIVOT_TEST_BLOCK_TARGET"])
_ready = pathlib.Path(os.environ["PIVOT_TEST_READY"])
_write_bytes = pathlib.Path.write_bytes
_replace = pathlib.Path.replace

def _pause() -> None:
    _ready.write_text("ready")
    while True:
        time.sleep(1)

def write_bytes(self: pathlib.Path, data: bytes) -> int:
    if self == _target:
        _pause()
    return _write_bytes(self, data)

def replace(self: pathlib.Path, target: pathlib.Path) -> pathlib.Path:
    if pathlib.Path(target) == _target:
        _pause()
    return _replace(self, target)

pathlib.Path.write_bytes = write_bytes
pathlib.Path.replace = replace
""")
    env = _env(
        PIVOT_TEST_BLOCK_TARGET=str(target),
        PIVOT_TEST_READY=str(ready),
    )
    env["PYTHONPATH"] = str(hook_dir) + os.pathsep + env.get("PYTHONPATH", "")
    process = subprocess.Popen(
        [_PIVOT, "get", "--force", "--rev", "HEAD", "source.txt"],
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 30
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "pivot get did not reach the fallback replacement"
    finally:
        if process.poll() is None:
            process.kill()
    process.communicate(timeout=30)

    assert target.read_text() == "workspace content"
    _ok(_pivot(root, "get", "--force", "--rev", "HEAD", "source.txt"))

    assert target.read_text() == "revision content"
    assert list(root.glob("*.tmp")) == []
    assert list(root.glob(".pivot_restore_*.file")) == []


def test_status_in_clone_missing_outputs_suggests_pull_then_repro(tmp_path: pathlib.Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _project(src, {"pipeline.py": _CLONE_PIPELINE, "data/raw/t.csv": "a\n1\n2\n"})
    _ok(_pivot(src, "repro"))
    _ok(_pivot(src, "config", "set", "remotes.origin", "s3://unused-bucket/cache"))
    _ok(_pivot(src, "config", "set", "default_remote", "origin"))
    (src / ".gitignore").write_text("data/c/\n")
    subprocess.run(["git", "add", "-A"], cwd=src, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=src,
        check=True,
    )
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(src), str(clone)], check=True)

    out = _ok(_pivot(clone, "status"))

    assert "Run `pivot pull` to restore outputs of 1 stage from the remote" in out
    assert "Run `pivot repro` to execute 1 stale stage" in out
    assert out.index("pivot pull") < out.index("pivot repro"), out


_IMPORTED_INPUT_SHARED = """\
from pivot.compose import Pipeline


with Pipeline("shared") as inputs:
    source = inputs.input("shared.txt", path="actual.txt")
"""


_IMPORTED_INPUT_PIPELINE = """\
from pivot.compose import Pipeline, stage
from shared import source


@stage
def result(data: str) -> str:
    return data


with Pipeline("demo") as pipeline:
    result(source)
"""


def test_repro_uses_a_directly_imported_inputs_declared_path(tmp_path: pathlib.Path) -> None:
    """A direct imported input handle keeps its path instead of falling back to data/raw/name."""
    root = _project(
        tmp_path,
        {
            "shared.py": _IMPORTED_INPUT_SHARED,
            "pipeline.py": _IMPORTED_INPUT_PIPELINE,
            "actual.txt": "ACTUAL",
            "data/raw/shared.txt": "WRONG DEFAULT",
        },
    )

    _ok(_pivot(root, "repro"))

    assert (root / "data" / "demo" / "result.txt").read_text() == "ACTUAL"


def test_repro_refuses_an_absolute_output_key_before_writing_outside_project(
    tmp_path: pathlib.Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("PRESERVE")
    key = str(sentinel)
    pipeline = (
        "from typing import TypedDict\n\n"
        + "from pivot.compose import Pipeline, stage\n\n"
        + f"Result = TypedDict('Result', {{{key!r}: str}})\n\n"
        + "@stage\n"
        + "def result() -> Result:\n"
        + f"    return {{{key!r}: 'REPLACED'}}\n\n"
        + "with Pipeline('demo') as pipeline:\n"
        + "    result()\n"
    )
    _project(root, {"pipeline.py": pipeline})

    proc = _pivot(root, "repro")
    out = proc.stdout + proc.stderr

    assert proc.returncode == 1, out
    assert "Artifact key cannot be absolute" in out, out
    assert sentinel.read_text() == "PRESERVE"
    assert not (root / "data" / "demo").exists(), "definition rejection prevents execution"


_SYMLINKED_DATA_PIPELINE = """\
from pivot.compose import Pipeline, stage


@stage
def copy(value: str) -> str:
    return value


with Pipeline("symlinked-data") as pipeline:
    copy(pipeline.input("value.txt", t=str))
"""


def test_repro_allows_data_symlinked_outside_the_project(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    external_data = tmp_path / "big-data"
    external_data.mkdir()
    (root / "data").symlink_to(external_data, target_is_directory=True)
    _project(
        root,
        {
            "pipeline.py": _SYMLINKED_DATA_PIPELINE,
            "data/raw/value.txt": "VALUE",
        },
    )

    _ok(_pivot(root, "repro"))

    assert (external_data / "symlinked-data" / "copy.txt").read_text() == "VALUE"


def test_checkout_restores_a_stage_output_in_data_symlinked_outside_the_project(
    tmp_path: pathlib.Path,
) -> None:
    """A stage output is placed by the pipeline, not by a .pvt, so the outward link is kept."""
    root = tmp_path / "project"
    root.mkdir()
    external_data = tmp_path / "big-data"
    external_data.mkdir()
    (root / "data").symlink_to(external_data, target_is_directory=True)
    _project(
        root,
        {
            "pipeline.py": _SYMLINKED_DATA_PIPELINE,
            "data/raw/value.txt": "VALUE",
        },
    )
    _ok(_pivot(root, "repro"))
    output = external_data / "symlinked-data" / "copy.txt"
    output.unlink()

    _ok(_pivot(root, "checkout"))

    assert output.read_text() == "VALUE"


def test_runtime_files_pivot_creates_are_git_ignored(tmp_path: pathlib.Path) -> None:
    """Config lock, artifact locks and the agent socket must not show up as untracked files."""
    root = _project(tmp_path, {"pipeline.py": _CLONE_PIPELINE, "data/raw/t.csv": "a\n1\n2\n"})
    _ok(_pivot(root, "config", "set", "remotes.origin", "s3://unused-bucket/cache"))
    _ok(_pivot(root, "repro"))
    (root / ".pivot" / "agent.sock").touch()

    created = sorted(
        str(p.relative_to(root)) for p in (root / ".pivot").iterdir() if p.name != ".gitignore"
    )
    assert ".pivot/config.yaml.lock" in created, created
    assert ".pivot/locks" in created, created

    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", ".pivot"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    untracked = sorted(line[3:] for line in status.stdout.splitlines())
    # Only committed state (config, ignore file, per-stage lock files) may be untracked.
    runtime = [
        path
        for path in untracked
        if path not in {".pivot/.gitignore", ".pivot/config.yaml"}
        and not path.startswith(".pivot/stages/")
    ]
    assert runtime == [], untracked


def test_stage_locks_of_pipelines_named_like_runtime_dirs_stay_committable(
    tmp_path: pathlib.Path,
) -> None:
    """`.pivot/.gitignore` entries are anchored: a pipeline named `cache` or `locks` keeps its
    lock and listing files."""
    root = _project(tmp_path, {})
    committed = [
        ".pivot/stages/cache/train.lock",
        ".pivot/stages/cache/train=.listing",
        ".pivot/stages/locks/train.lock",
        ".pivot/stages/locks/train=.listing",
        ".pivot/stages/state.lmdb/train.lock",
        ".pivot/stages/state.lmdb/train=.listing",
    ]
    for rel in committed:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("{}")

    # Without --verbose: with it, git counts a match of the `!/stages/**` negation as ignored.
    check = subprocess.run(
        ["git", "check-ignore", *committed],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert check.returncode == 1, f"files ignored:\n{check.stdout}"


_METRIC_ONLY_PIPELINE = """\
from typing import Annotated

import pandas as pd

from pivot import Pipeline, metric, stage


@stage
def summarize(raw: pd.DataFrame) -> Annotated[dict, metric]:
    return {"rows": int(len(raw))}


with Pipeline("m") as pipeline:
    summarize(pipeline.input("t.csv"))
"""


def test_status_does_not_suggest_pull_for_an_edited_metric(tmp_path: pathlib.Path) -> None:
    """Pull never restores metrics (they are git-tracked), so it must not be suggested for one."""
    root = _project(tmp_path, {"pipeline.py": _METRIC_ONLY_PIPELINE, "data/raw/t.csv": "a\n1\n"})
    _ok(_pivot(root, "repro"))
    _ok(_pivot(root, "config", "set", "remotes.origin", "s3://unused-bucket/cache"))
    metric_file = root / "metrics" / "m" / "summarize.yaml"
    assert metric_file.is_file()
    metric_file.write_text("rows: 999\n")

    out = _ok(_pivot(root, "status"))

    assert "pivot pull" not in out, out


_CHAIN_PIPELINE = """\
import pandas as pd

from pivot import Pipeline, stage


@stage
def scale(raw: pd.DataFrame) -> pd.DataFrame:
    return raw * 2


@stage
def total(scaled: pd.DataFrame) -> dict:
    return {"sum": int(scaled["a"].sum())}


with Pipeline("k") as pipeline:
    total(scale(pipeline.input("t.csv")))
"""


def _chain_clone(tmp_path: pathlib.Path) -> pathlib.Path:
    """Run the two-stage chain, commit locks with outputs ignored, and return a fresh clone."""
    src = tmp_path / "src"
    src.mkdir()
    _project(src, {"pipeline.py": _CHAIN_PIPELINE, "data/raw/t.csv": "a\n1\n2\n"})
    _ok(_pivot(src, "repro"))
    _ok(_pivot(src, "config", "set", "remotes.origin", "s3://unused-bucket/cache"))
    (src / ".gitignore").write_text("data/k/\n")
    subprocess.run(["git", "add", "-A"], cwd=src, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=src,
        check=True,
    )
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(src), str(clone)], check=True)
    return clone


def test_status_does_not_suggest_pull_for_stage_whose_upstream_is_stale(
    tmp_path: pathlib.Path,
) -> None:
    """repro recomputes a stage downstream of a stale one, so pulling its outputs is wasted."""
    clone = _chain_clone(tmp_path)
    pipeline_py = clone / "pipeline.py"
    pipeline_py.write_text(pipeline_py.read_text().replace("raw * 2", "raw * 3"))

    out = _ok(_pivot(clone, "status"))

    assert "Run `pivot repro` to execute 2 stale stages" in out, out
    assert "pivot pull" not in out, out


def test_status_in_plain_clone_counts_every_stage_pull_restores(tmp_path: pathlib.Path) -> None:
    """A downstream stage stale only because its upstream's outputs are also missing is pullable."""
    clone = _chain_clone(tmp_path)

    out = _ok(_pivot(clone, "status"))

    assert "Run `pivot pull` to restore outputs of 2 stages from the remote" in out, out


def test_status_quiet_json_keeps_pull_suggestion(tmp_path: pathlib.Path) -> None:
    """`-q` drops chatter, but `--json` still prints suggestions, including the pull line."""
    clone = _chain_clone(tmp_path)

    out = _ok(_pivot(clone, "-q", "status", "--json"))

    suggestions = json.loads(out)["suggestions"]
    assert "Run `pivot pull` to restore outputs of 2 stages from the remote" in suggestions, out


_METRIC_MIDDLE_PIPELINE = """\
from typing import Annotated

import pandas as pd

from pivot import Pipeline, metric, stage


@stage
def clean(raw: pd.DataFrame) -> pd.DataFrame:
    return raw * 2


@stage
def summarize(cleaned: pd.DataFrame) -> Annotated[dict, metric]:
    return {"rows": int(len(cleaned))}


@stage
def report(stats: dict) -> dict:
    return {"seen": stats["rows"]}


with Pipeline("r") as pipeline:
    report(summarize(clean(pipeline.input("t.csv"))))
"""


def test_status_pull_count_passes_through_committed_metric_stage(tmp_path: pathlib.Path) -> None:
    """A committed metric between two data stages must not hide the downstream data stage."""
    src = tmp_path / "src"
    src.mkdir()
    _project(src, {"pipeline.py": _METRIC_MIDDLE_PIPELINE, "data/raw/t.csv": "a\n1\n2\n"})
    _ok(_pivot(src, "repro"))
    _ok(_pivot(src, "config", "set", "remotes.origin", "s3://unused-bucket/cache"))
    (src / ".gitignore").write_text("data/r/\n")
    subprocess.run(["git", "add", "-A"], cwd=src, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=src,
        check=True,
    )
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(src), str(clone)], check=True)
    assert (clone / "metrics" / "r" / "summarize.yaml").is_file()

    out = _ok(_pivot(clone, "status"))

    assert "Run `pivot pull` to restore outputs of 2 stages from the remote" in out, out


_SHARED_DEP_PIPELINE = """\
import pandas as pd

from pivot import Pipeline, stage


@stage
def make(raw: pd.DataFrame) -> pd.DataFrame:
    return raw * 2


@stage
def both(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    return {"sum": int(a["a"].sum() + b["a"].sum())}


with Pipeline("dup") as pipeline:
    made = make(pipeline.input("x.csv"))
    both(made, made)
"""


def test_commit_run_cache_entry_is_hit_for_stage_reading_one_artifact_twice(
    tmp_path: pathlib.Path,
) -> None:
    """`pivot commit` and the worker must file a stage's run-cache entry under one key."""
    root = _project(tmp_path, {"pipeline.py": _SHARED_DEP_PIPELINE, "data/raw/x.csv": "a\n1\n"})
    _ok(_pivot(root, "repro", "--no-commit"))
    _ok(_pivot(root, "commit"))
    raw = root / "data" / "raw" / "x.csv"
    raw.write_text("a\n5\n")
    _ok(_pivot(root, "repro"))
    raw.write_text("a\n1\n")

    out = _ok(_pivot(root, "repro"))

    assert "0 ran, 2 cached" in out, out
    assert (root / "data" / "dup" / "both.yaml").read_text() == "sum: 4\n"


def test_repro_hardlinks_stage_outputs_into_cache_by_default(tmp_path: pathlib.Path) -> None:
    """The default checkout order starts with hardlink, so an output is not stored twice."""
    root = _project(tmp_path, {"pipeline.py": _CHAIN_PIPELINE, "data/raw/t.csv": "a\n1\n2\n"})
    _ok(_pivot(root, "repro"))

    output = root / "data" / "k" / "scale.jsonl"
    cached = [p for p in (root / ".pivot" / "cache" / "files").rglob("*") if p.is_file()]

    assert output.stat().st_nlink == 2
    assert output.stat().st_ino in {p.stat().st_ino for p in cached}


def test_symlink_checkout_mode_applies_when_repro_writes_and_restores_outputs(
    tmp_path: pathlib.Path,
) -> None:
    """`cache.checkout_mode` governs stage outputs on run and on skip-restore, as for checkout."""
    root = _project(tmp_path, {"pipeline.py": _CHAIN_PIPELINE, "data/raw/t.csv": "a\n1\n2\n"})
    _ok(_pivot(root, "config", "set", "cache.checkout_mode", "symlink"))
    output = root / "data" / "k" / "scale.jsonl"

    _ok(_pivot(root, "repro"))
    assert output.is_symlink()

    output.unlink()
    out = _ok(_pivot(root, "repro"))

    assert "0 ran, 2 cached" in out, out
    assert output.is_symlink()
    assert loaders.DataFrameJSONL().load(output).to_dict(orient="records") == [
        {"a": 2},
        {"a": 4},
    ]


def test_noop_status_reads_no_output_bytes_after_repro(tmp_path: pathlib.Path) -> None:
    """Committed outputs are in the StateDB file-hash cache, so status hashes nothing."""
    root = _project(tmp_path, {"pipeline.py": _CHAIN_PIPELINE, "data/raw/t.csv": "a\n1\n2\n"})
    _ok(_pivot(root, "repro"))

    out = _ok(_pivot(root, "status", env=_env(PIVOT_METRICS="1")))

    assert "2 cached" in out, out
    assert "cache.hash_file" not in out, out


def test_noop_status_reads_no_output_bytes_after_pivot_commit(tmp_path: pathlib.Path) -> None:
    """`pivot commit` records the output hashes it computes, as a run does."""
    root = _project(tmp_path, {"pipeline.py": _CHAIN_PIPELINE, "data/raw/t.csv": "a\n1\n2\n"})
    _ok(_pivot(root, "repro", "--no-commit"))
    _ok(_pivot(root, "commit"))

    out = _ok(_pivot(root, "status", env=_env(PIVOT_METRICS="1")))

    assert "2 cached" in out, out
    assert "cache.hash_file" not in out, out


_DIROUT_PIPELINE = """\
import pathlib
import tempfile

import pandas as pd

from pivot import Pipeline, stage


@stage
def shards(raw: pd.DataFrame) -> pathlib.Path:
    out = pathlib.Path(tempfile.mkdtemp())
    for i, row in raw.iterrows():
        (out / f"{i}.txt").write_text(str(int(row.sum())))
    return out


with Pipeline("d") as pipeline:
    shards(pipeline.input("t.csv"))
"""

# The same stage also writes a build/ directory, which is part of its directory artifact.
_DIROUT_BUILD_PIPELINE = _DIROUT_PIPELINE.replace(
    "    return out\n",
    '    (out / "build").mkdir()\n'
    + '    (out / "build" / "log.txt").write_text(f"built {int(raw.sum().sum())}")\n'
    + "    return out\n",
)


def _commit(cwd: pathlib.Path, message: str) -> None:
    subprocess.run(["git", "add", "-A"], cwd=cwd, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", message],
        cwd=cwd,
        check=True,
    )


_GCS_PIPELINE = """\
import random

from pivot.compose import Pipeline, stage


@stage
def generate() -> str:
    return random.Random(0).randbytes(5 * 1024 * 1024).hex()  # 10 MiB, above one download read


@stage
def measure(text: str) -> dict[str, int]:
    return {"chars": len(text)}


with Pipeline("g") as pipeline:
    measure(generate())
"""


def test_gcs_remote_push_pull_import_round_trip(
    tmp_path: pathlib.Path, fake_gcs_server: str, gcs_bucket: str
) -> None:
    """A gs:// remote serves push, pull, status --remote and import like an s3:// one."""
    env = _env(STORAGE_EMULATOR_HOST=fake_gcs_server)
    src = tmp_path / "src"
    src.mkdir()
    _project(src, {"pipeline.py": _GCS_PIPELINE, ".gitignore": "/data/g/\n__pycache__/\n"})
    # `pivot import` defaults to --rev main; a bare `git init` names its branch after init.defaultBranch
    subprocess.run(["git", "checkout", "-q", "-b", "main"], cwd=src, check=True)
    # `git archive --remote` by commit SHA (pivot import) requires this for local repos
    subprocess.run(["git", "config", "uploadArchive.allowUnreachable", "true"], cwd=src, check=True)
    _ok(_pivot(src, "config", "set", "remotes.origin", f"gs://{gcs_bucket}/cache"))
    _ok(_pivot(src, "repro", env=env))
    outputs = [src / "data" / "g" / "generate.txt", src / "data" / "g" / "measure.yaml"]
    originals = [path.read_bytes() for path in outputs]

    assert "2 transferred, 0 skipped, 0 failed" in _ok(_pivot(src, "push", env=env))
    shutil.rmtree(src / ".pivot" / "cache")
    shutil.rmtree(src / "data" / "g")
    _ok(_pivot(src, "pull", env=env))

    assert [path.read_bytes() for path in outputs] == originals
    status = json.loads(_ok(_pivot(src, "--quiet", "status", "--remote-only", "--json", env=env)))
    assert status["remote"]["url"] == f"gs://{gcs_bucket}/cache"
    assert (status["remote"]["push_count"], status["remote"]["pull_count"]) == (0, 0)

    _commit(src, "v1")
    consumer = tmp_path / "consumer"
    consumer.mkdir()
    _project(consumer, {})
    _ok(_pivot(consumer, "import", str(src), "g/generate", "--out", "generate.txt", env=env))

    assert (consumer / "generate.txt").read_bytes() == originals[0]


def test_status_remote_rechecks_after_the_remote_url_changes(
    tmp_path: pathlib.Path, fake_gcs_server: str, gcs_bucket: str
) -> None:
    """`status --remote` must query the bucket the remote points at now, not the previous one.

    `push`, `pull` and `status.get_remote_status` all call `sync.check_remote_url`, which clears
    the cached "known on remote" index when a remote's URL changes. So after a remote is
    repointed at an empty bucket, `status --remote` reports both outputs to push, rather than
    the old bucket's "0 to push" answered from the stale index.
    """
    env = _env(STORAGE_EMULATOR_HOST=fake_gcs_server)
    empty_bucket = f"pivot-test-{uuid.uuid4().hex[:12]}"
    request = urllib.request.Request(
        f"{fake_gcs_server}/storage/v1/b?project=pivot-test",
        data=json.dumps({"name": empty_bucket}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as resp:
        assert resp.status == 200, resp.read()

    src = tmp_path / "src"
    src.mkdir()
    _project(src, {"pipeline.py": _GCS_PIPELINE, ".gitignore": "/data/g/\n__pycache__/\n"})
    _ok(_pivot(src, "config", "set", "remotes.origin", f"gs://{gcs_bucket}/cache"))
    _ok(_pivot(src, "repro", env=env))
    assert "2 transferred" in _ok(_pivot(src, "push", env=env))

    _ok(_pivot(src, "config", "set", "remotes.origin", f"gs://{empty_bucket}/cache"))
    # stdout only: clearing a stale index logs a warning to stderr, which is not JSON
    proc = _pivot(src, "--quiet", "status", "--remote-only", "--json", env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    status = json.loads(proc.stdout)

    assert status["remote"]["url"] == f"gs://{empty_bucket}/cache"
    assert status["remote"]["push_count"] == 2


def _dirout_clone_after_teammate_rerun(
    tmp_path: pathlib.Path, s3_endpoint: str, pipeline_py: str = _DIROUT_PIPELINE
) -> tuple[pathlib.Path, dict[str, str]]:
    """A dev ran a directory-output stage at v1; a teammate re-ran it at v2, pushed and committed.

    Returns the dev's clone after `git pull` (lock at v2, workspace still v1) and the env
    that points pivot at the S3 remote.
    """
    env = _env(
        AWS_ENDPOINT_URL=s3_endpoint,
        AWS_ACCESS_KEY_ID="x",
        AWS_SECRET_ACCESS_KEY="x",
        AWS_DEFAULT_REGION="us-east-1",
    )
    bucket = f"dirout-{uuid.uuid4().hex[:8]}"
    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        region_name="us-east-1",
        aws_access_key_id="x",
        aws_secret_access_key="x",
    )
    s3.create_bucket(Bucket=bucket)  # pyright: ignore[reportAttributeAccessIssue] - boto3 ships no stubs

    src = tmp_path / "src"
    src.mkdir()
    _project(
        src,
        {
            "pipeline.py": pipeline_py,
            "data/raw/t.csv": "a,b\n1,2\n3,4\n",
            ".gitignore": "/data/d/\n__pycache__/\n",
        },
    )
    _ok(_pivot(src, "config", "set", "remotes.origin", f"s3://{bucket}/cache"))
    _ok(_pivot(src, "config", "set", "default_remote", "origin"))
    _ok(_pivot(src, "repro", env=env))
    _ok(_pivot(src, "push", env=env))
    _commit(src, "v1")

    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(src), str(clone)], check=True)
    _ok(_pivot(clone, "repro", env=env))

    (src / "data" / "raw" / "t.csv").write_text("a,b\n5,6\n7,8\n")
    _ok(_pivot(src, "repro", env=env))
    _ok(_pivot(src, "push", env=env))
    _commit(src, "v2")

    subprocess.run(["git", "pull", "-q", "--ff-only"], cwd=clone, check=True)
    return clone, env


class _MotoS3Client(Protocol):
    """The boto3 S3 client calls these tests make; boto3 ships no stubs."""

    def create_bucket(self, **kwargs: object) -> object: ...

    def put_object(self, **kwargs: object) -> object: ...


def _moto_bucket(endpoint: str, prefix: str) -> tuple[dict[str, str], str, _MotoS3Client]:
    """Make a bucket on the moto S3 server at ``endpoint``. Returns the env that points pivot at
    that server, the bucket's name, and a client that plants objects in the bucket."""
    env = _env(
        AWS_ENDPOINT_URL=endpoint,
        AWS_ACCESS_KEY_ID="x",
        AWS_SECRET_ACCESS_KEY="x",
        AWS_DEFAULT_REGION="us-east-1",
    )
    bucket = f"{prefix}-{uuid.uuid4().hex[:8]}"
    s3 = cast(
        "_MotoS3Client",
        boto3.client(
            "s3",
            endpoint_url=endpoint,
            region_name="us-east-1",
            aws_access_key_id="x",
            aws_secret_access_key="x",
        ),
    )
    s3.create_bucket(Bucket=bucket)
    return env, bucket, s3


@pytest.mark.parametrize(
    ("extra_config", "remote_args"),
    [
        pytest.param((("default_remote", "origin"),), (), id="default"),
        pytest.param((), ("--remote", "origin"), id="named"),
    ],
)
def test_get_restores_tracked_file_from_configured_remote(
    tmp_path: pathlib.Path,
    moto_services: dict[str, str],
    extra_config: tuple[tuple[str, str], ...],
    remote_args: tuple[str, ...],
) -> None:
    """get retrieves an uncached tracked file from the default or the named remote, not from the
    remote configured first."""
    env, bucket, _ = _moto_bucket(moto_services["s3"], "get")
    root = _project(tmp_path, {".gitignore": "artifact.txt\n", "artifact.txt": "from remote\n"})
    # backup comes first and has no bucket, so only default_remote or --remote can pick origin
    _ok(_pivot(root, "config", "set", "remotes.backup", "s3://unused-bucket/cache"))
    _ok(_pivot(root, "config", "set", "remotes.origin", f"s3://{bucket}/cache"))
    for key, value in extra_config:
        _ok(_pivot(root, "config", "set", key, value))
    _ok(_pivot(root, "track", "artifact.txt"))
    file_hash, _ = cache.hash_file(root / "artifact.txt")
    _ok(_pivot(root, "push", "--remote", "origin", env=env))
    _commit(root, "track artifact")

    (root / "artifact.txt").unlink()
    cache.get_cache_path(root / ".pivot" / "cache" / "files", file_hash).unlink()

    proc = _pivot(root, "get", *remote_args, "--rev", "HEAD", "artifact.txt", env=env)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "from remote" in proc.stdout
    assert (root / "artifact.txt").read_text() == "from remote\n"
    cached_path = cache.get_cache_path(root / ".pivot" / "cache" / "files", file_hash)
    assert cached_path.read_bytes() == b"from remote\n"
    assert cached_path.stat().st_mode & 0o777 == 0o444
    with state.StateDB(root / ".pivot", readonly=True) as state_db:
        artifact = root / "artifact.txt"
        assert state_db.get(artifact, artifact.stat()) == file_hash, (
            "get from the remote records the restored file's hash, so the next status reads none "
            "of it"
        )

    (root / "artifact.txt").unlink()
    unreachable_env = _env(
        AWS_ENDPOINT_URL="http://127.0.0.1:9",
        AWS_ACCESS_KEY_ID="x",
        AWS_SECRET_ACCESS_KEY="x",
        AWS_DEFAULT_REGION="us-east-1",
    )
    cached_proc = _pivot(
        root, "get", *remote_args, "--rev", "HEAD", "artifact.txt", env=unreachable_env
    )
    assert cached_proc.returncode == 0, cached_proc.stdout + cached_proc.stderr
    assert "from cache" in cached_proc.stdout
    assert (root / "artifact.txt").read_text() == "from remote\n"


def test_get_restores_directory_output_from_configured_remote(
    tmp_path: pathlib.Path, moto_services: dict[str, str]
) -> None:
    """get retrieves every uncached file of a directory output from the configured remote."""
    clone, env = _dirout_clone_after_teammate_rerun(tmp_path, moto_services["s3"])
    shards = clone / "data" / "d" / "shards.dat"
    shutil.rmtree(shards)

    proc = _pivot(clone, "get", "--rev", "HEAD", "d/shards", env=env)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "from remote" in proc.stdout
    assert (shards / "0.txt").read_text() == "11"
    assert (shards / "1.txt").read_text() == "15"
    with state.StateDB(clone / ".pivot", readonly=True) as state_db:
        for name in ("0.txt", "1.txt"):
            member = shards / name
            expected = xxhash.xxh64(member.read_bytes()).hexdigest()
            assert state_db.get(member, member.stat()) == expected, (
                f"get from the remote records {name}'s hash"
            )


def test_get_repairs_partial_directory_cache_tree_from_remote(
    tmp_path: pathlib.Path, moto_services: dict[str, str]
) -> None:
    """get downloads a missing blob when a stale symlink cache tree exists."""
    clone, env = _dirout_clone_after_teammate_rerun(tmp_path, moto_services["s3"])
    _ok(_pivot(clone, "pull", "d/shards", env=env))
    shards = clone / "data" / "d" / "shards.dat"
    shutil.rmtree(shards)

    cache_dir = clone / ".pivot" / "cache" / "files"
    tree_hash, members = _dir_output_record_and_members(clone, "d/shards")
    missing_hash = members["files"][0]["hash"]
    cache.get_cache_path(cache_dir, missing_hash).unlink()
    partial_tree = cache.tree_path(cache_dir, tree_hash)
    partial_tree.mkdir(parents=True)
    (partial_tree / "0.txt").write_text("xx")

    proc = _pivot(
        clone,
        "get",
        "--checkout-mode",
        "symlink",
        "--rev",
        "HEAD",
        "d/shards",
        env=env,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (shards / "0.txt").read_text() == "11"
    assert (shards / "1.txt").read_text() == "15"


def test_get_names_missing_remote_configuration(tmp_path: pathlib.Path) -> None:
    """get explains how to configure its remote fallback."""
    root = _project(tmp_path, {".gitignore": "artifact.txt\n", "artifact.txt": "from remote\n"})
    _ok(_pivot(root, "track", "artifact.txt"))
    file_hash, _ = cache.hash_file(root / "artifact.txt")
    _commit(root, "track artifact")

    (root / "artifact.txt").unlink()
    cache.get_cache_path(root / ".pivot" / "cache" / "files", file_hash).unlink()

    proc = _pivot(root, "get", "--rev", "HEAD", "artifact.txt")
    out = proc.stdout + proc.stderr

    assert proc.returncode == 1, out
    assert "No remotes configured." in out
    assert "pivot config set remotes.<name> <url>" in out


@pytest.mark.parametrize(
    ("remote_body", "reason"),
    [
        pytest.param(None, "Remote cache file not found", id="missing"),
        pytest.param(b"corrupt remote bytes", "Hash mismatch", id="corrupt"),
    ],
)
def test_get_leaves_no_output_or_cache_after_failed_remote_download(
    tmp_path: pathlib.Path, moto_services: dict[str, str], remote_body: bytes | None, reason: str
) -> None:
    """get rejects a missing or corrupt remote object, naming the backend's reason, without
    leaving local state."""
    env, bucket, s3 = _moto_bucket(moto_services["s3"], "get-failed")
    root = _project(tmp_path, {".gitignore": "artifact.txt\n", "artifact.txt": "expected bytes\n"})
    _ok(_pivot(root, "config", "set", "remotes.origin", f"s3://{bucket}/cache"))
    _ok(_pivot(root, "config", "set", "default_remote", "origin"))
    _ok(_pivot(root, "track", "artifact.txt"))
    file_hash, _ = cache.hash_file(root / "artifact.txt")
    if remote_body is not None:
        _ok(_pivot(root, "push", env=env))
        s3.put_object(
            Bucket=bucket,
            Key=keys.hash_to_key("cache/", file_hash),
            Body=remote_body,
        )
    _commit(root, "track artifact")

    (root / "artifact.txt").unlink()
    cached_path = cache.get_cache_path(root / ".pivot" / "cache" / "files", file_hash)
    cached_path.unlink()

    proc = _pivot(root, "get", "--rev", "HEAD", "artifact.txt", env=env)
    out = proc.stdout + proc.stderr

    assert proc.returncode == 1, out
    assert f"failed to download from remote: {reason}" in out, out
    assert not (root / "artifact.txt").exists()
    assert not cached_path.exists()


def _tracked_pair_with_second_uncached(
    tmp_path: pathlib.Path, second: str = "second.txt"
) -> pathlib.Path:
    """A project tracking first.txt and ``second`` (a path from the root), both deleted from the
    workspace and ``second`` also deleted from the cache."""
    root = _project(
        tmp_path,
        {".gitignore": f"first.txt\n{second}\n", "first.txt": "first\n", second: "second\n"},
    )
    _ok(_pivot(root, "track", "first.txt", second))
    second_hash, _ = cache.hash_file(root / second)
    _commit(root, "track files")
    (root / "first.txt").unlink()
    (root / second).unlink()
    cache.get_cache_path(root / ".pivot" / "cache" / "files", second_hash).unlink()
    return root


def test_get_reports_prior_cache_restore_when_remote_is_unconfigured(
    tmp_path: pathlib.Path,
) -> None:
    """get reports both a cache hit and a later remote configuration error."""
    root = _tracked_pair_with_second_uncached(tmp_path)

    proc = _pivot(root, "get", "--rev", "HEAD", "first.txt", "second.txt")
    out = proc.stdout + proc.stderr

    assert proc.returncode == 1, out
    assert "first.txt (from cache)" in out
    assert (root / "first.txt").read_text() == "first\n"
    assert "No remotes configured." in out


_BAD_S3_ENVS = [
    pytest.param({"AWS_PROFILE": "pivot-test-no-such-profile"}, id="missing-profile"),
    pytest.param(
        {"AWS_ENDPOINT_URL": "not-a-url", "AWS_ACCESS_KEY_ID": "x", "AWS_SECRET_ACCESS_KEY": "x"},
        id="invalid-endpoint",
    ),
]


def _bad_s3_env(tmp_path: pathlib.Path, extra: dict[str, str]) -> dict[str, str]:
    """An environment in which pivot cannot set up S3 access for the reason ``extra`` gives, with
    no AWS config or profile from the host to fall back to."""
    empty = tmp_path / "empty-aws-config"
    empty.write_text("")
    env = {k: v for k, v in _env().items() if k not in ("AWS_PROFILE", "AWS_DEFAULT_PROFILE")}
    env.update(
        AWS_CONFIG_FILE=str(empty),
        AWS_SHARED_CREDENTIALS_FILE=str(empty),
        AWS_DEFAULT_REGION="us-east-1",
        **extra,
    )
    return env


@pytest.mark.parametrize("bad_env", _BAD_S3_ENVS)
def test_get_reports_prior_cache_restore_when_the_remote_backend_cannot_start(
    tmp_path: pathlib.Path, bad_env: dict[str, str]
) -> None:
    """A remote whose S3 client cannot start fails only the target that needs it: the earlier
    cache restore is still reported."""
    root = _tracked_pair_with_second_uncached(tmp_path)
    _ok(_pivot(root, "config", "set", "remotes.origin", "s3://bucket-a/cache"))

    proc = _pivot(
        root, "get", "--rev", "HEAD", "first.txt", "second.txt", env=_bad_s3_env(tmp_path, bad_env)
    )
    out = proc.stdout + proc.stderr

    assert proc.returncode == 1, out
    assert "first.txt (from cache)" in out, out
    assert (root / "first.txt").read_text() == "first\n"
    assert "second.txt - failed to download from remote" in out, out
    assert "Cannot set up S3 access for s3://bucket-a/cache" in out, out
    assert not (root / "second.txt").exists()


@pytest.mark.parametrize("bad_env", _BAD_S3_ENVS)
def test_pull_reports_a_remote_backend_that_cannot_start_as_a_remote_error(
    tmp_path: pathlib.Path, bad_env: dict[str, str]
) -> None:
    """pull shows an S3 client that cannot start as a pivot remote error with its tip, not as
    the raw botocore exception."""
    root = _tracked_pair_with_second_uncached(tmp_path)
    _ok(_pivot(root, "config", "set", "remotes.origin", "s3://bucket-a/cache"))

    proc = _pivot(root, "pull", "second.txt", env=_bad_s3_env(tmp_path, bad_env))
    out = proc.stdout + proc.stderr

    assert proc.returncode == 1, out
    assert "Cannot set up S3 access for s3://bucket-a/cache" in out, out
    assert "Tip: Check network connection and remote credentials" in out, out
    assert not (root / "second.txt").exists()


@contextlib.contextmanager
def _refused_endpoint() -> Iterator[str]:
    """The URL of a local port nothing listens on, so a connection to it is refused at once."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    yield f"http://127.0.0.1:{port}"


@contextlib.contextmanager
def _endpoint_that_never_connects() -> Iterator[str]:
    """The URL of a local port that never completes a connection, so connecting to it times out:
    it listens with a full accept queue and never accepts, and Linux drops a SYN to a listener
    whose accept queue is full."""
    with contextlib.ExitStack() as stack:
        listener = stack.enter_context(socket.socket())
        listener.bind(("127.0.0.1", 0))
        listener.listen(0)
        address = listener.getsockname()
        for _ in range(8):
            filler = stack.enter_context(socket.socket())
            filler.settimeout(0.5)
            try:
                filler.connect(address)
            except TimeoutError:
                break
        else:
            pytest.fail("the listener kept completing connections")
        yield f"http://127.0.0.1:{address[1]}"


@pytest.mark.parametrize(
    ("url", "endpoint", "reason"),
    [
        pytest.param(
            "s3://bucket-x/cache",
            _refused_endpoint,
            "Could not connect to the endpoint URL",
            id="s3-refused",
        ),
        pytest.param(
            "gs://bucket-x/cache",
            _refused_endpoint,
            "ClientConnectorError connecting to 127.0.0.1",
            id="gcs-refused",
        ),
        pytest.param(
            "s3://bucket-x/cache",
            _endpoint_that_never_connects,
            "Connect timeout on endpoint URL",
            id="s3-connect-timeout",
        ),
        pytest.param(
            "gs://bucket-x/cache",
            _endpoint_that_never_connects,
            "ConnectionTimeoutError",
            id="gcs-connect-timeout",
        ),
    ],
)
def test_get_tries_an_unreachable_remote_once(
    tmp_path: pathlib.Path,
    url: str,
    endpoint: Callable[[], contextlib.AbstractContextManager[str]],
    reason: str,
) -> None:
    """A remote no connection can be made to (refused, or a connect timeout) fails the first
    target that needs it with that reason; each later target that needs it fails naming the
    remote, without trying it again, so an unreachable remote costs one connection timeout per
    get, not one per target."""
    names = ("a.txt", "b.txt", "c.txt")
    root = _project(
        tmp_path, {".gitignore": "a.txt\nb.txt\nc.txt\n", **{name: name for name in names}}
    )
    _ok(_pivot(root, "config", "set", "remotes.origin", url))
    _ok(_pivot(root, "config", "set", "remote.retries", "0"))
    _ok(_pivot(root, "config", "set", "remote.connect_timeout", "1"))
    _ok(_pivot(root, "track", *names))
    _commit(root, "track files")
    for name in names:
        file_hash, _ = cache.hash_file(root / name)
        cache.get_cache_path(root / ".pivot" / "cache" / "files", file_hash).unlink()
        (root / name).unlink()

    with endpoint() as address:
        env = _env(
            AWS_ENDPOINT_URL=address,
            AWS_ACCESS_KEY_ID="x",
            AWS_SECRET_ACCESS_KEY="x",
            AWS_DEFAULT_REGION="us-east-1",
            STORAGE_EMULATOR_HOST=address,
        )
        proc = _pivot(root, "get", "--rev", "HEAD", *names, env=env)
    out = proc.stdout + proc.stderr

    not_tried = f"{url} could not be reached for an earlier target, so it is not tried again"
    assert proc.returncode == 1, out
    assert "a.txt - failed to download from remote: " in out, out
    assert reason in out, out
    assert out.count(reason) == 1, "only the first target tries the remote"
    assert f"b.txt - failed to download from remote: {not_tried}" in out, out
    assert f"c.txt - failed to download from remote: {not_tried}" in out, out
    assert "Tip: Check network connection and remote credentials" in out, out
    assert "Failed to restore 3 file(s)" in out, out


_EXEC_DIR_PIPELINE = """\
import os
import pathlib
import tempfile

from pivot import Pipeline, stage


@stage
def tools() -> pathlib.Path:
    out = pathlib.Path(tempfile.mkdtemp())
    (out / "notes.txt").write_text("plain")
    script = out / "run.sh"
    script.write_text("#!/bin/sh\\necho run\\n")
    os.chmod(script, 0o755)
    return out


with Pipeline("x") as pipeline:
    tools()
"""

# The same directory output with no executable file.
_PLAIN_DIR_PIPELINE = _EXEC_DIR_PIPELINE.replace("    os.chmod(script, 0o755)\n", "")


def _dir_output_record_and_members(
    root: pathlib.Path, stage: str, key: str | None = None
) -> tuple[str, listing.DirListing]:
    """A directory output's cache-tree hash (the listing hash its lock records) and the committed
    listing members beside the lock."""
    lock_data = yaml.safe_load((root / ".pivot" / "stages" / f"{stage}.lock").read_text())
    (out,) = [entry for entry in lock_data["records"]["outs"] if entry["key"] == key]
    return out["listing"], _helper_output_listing(root, stage, key)


def _plant_unmarked_tree(
    files_dir: pathlib.Path, tree_hash: str, members: listing.DirListing
) -> pathlib.Path:
    """Build by hand the unmarked SYMLINK cache tree of a directory output that hardlink mode left
    without one, each file copied from its own cache object. Returns the tree."""
    tree = cache.tree_path(files_dir, tree_hash)
    assert not tree.exists(), "hardlink mode should build no tree"
    tree.mkdir(parents=True)
    for entry in members["files"]:
        (tree / entry["relpath"]).write_bytes(
            cache.get_cache_path(files_dir, entry["hash"]).read_bytes()
        )
    return tree


def _executable_tools_only_on_remote(
    tmp_path: pathlib.Path, s3_endpoint: str
) -> tuple[pathlib.Path, dict[str, str]]:
    """A project with a tracked first.txt and the x/tools directory output, whose run.sh is
    executable, pushed and committed, then removed from the workspace, x/tools' files also
    removed from the cache. Returns the root and the env that points pivot at the remote."""
    env, bucket, _ = _moto_bucket(s3_endpoint, "exec")
    root = _project(
        tmp_path,
        {
            "pipeline.py": _EXEC_DIR_PIPELINE,
            ".gitignore": "/data/\n__pycache__/\nfirst.txt\n",
            "first.txt": "first\n",
        },
    )
    _ok(_pivot(root, "config", "set", "remotes.origin", f"s3://{bucket}/cache"))
    _ok(_pivot(root, "repro", env=env))
    _ok(_pivot(root, "track", "first.txt"))
    _ok(_pivot(root, "push", env=env))
    _commit(root, "v1")
    files_dir = root / ".pivot" / "cache" / "files"
    _, members = _dir_output_record_and_members(root, "x/tools")
    for entry in members["files"]:
        cache.get_cache_path(files_dir, entry["hash"]).unlink()
    shutil.rmtree(root / "data" / "x")
    (root / "first.txt").unlink()
    return root, env


def _get_restores_first_and_refuses_executable_tools(
    root: pathlib.Path, env: dict[str, str], source: str
) -> None:
    """Run get for first.txt and x/tools with --checkout-mode hardlink, and check that first.txt
    is restored from the cache and that x/tools, restored from ``source`` ("remote" or "cache"),
    fails because its executable member cannot be linked, with that reason and its tip."""
    proc = _pivot(
        root, "get", "--rev", "HEAD", "--checkout-mode", "hardlink", "first.txt", "x/tools", env=env
    )
    out = proc.stdout + proc.stderr

    assert proc.returncode == 1, out
    assert "first.txt (from cache)" in out, out
    assert (root / "first.txt").read_text() == "first\n"
    assert f"tools.dat - failed to restore from {source}: Cache object" in out, out
    assert "needs isexec=True" in out, out
    assert "Tip: Add CheckoutMode.COPY to checkout_modes" in out, out
    assert not (root / "data" / "x" / "tools.dat").exists()


def test_get_reports_earlier_targets_when_the_downloaded_output_cannot_be_linked(
    tmp_path: pathlib.Path, moto_services: dict[str, str]
) -> None:
    """An object get downloads has no executable bit, so with --checkout-mode hardlink an
    executable member cannot be linked: that output fails with the reason, and the target
    restored before it is still reported."""
    root, env = _executable_tools_only_on_remote(tmp_path, moto_services["s3"])

    _get_restores_first_and_refuses_executable_tools(root, env, "remote")


def test_get_reports_earlier_targets_when_a_pulled_output_cannot_be_linked(
    tmp_path: pathlib.Path, moto_services: dict[str, str]
) -> None:
    """The same refusal when `pivot pull` downloaded the objects: get restores from the cache,
    fails that output with the reason, and still reports the target restored before it."""
    root, env = _executable_tools_only_on_remote(tmp_path, moto_services["s3"])
    _ok(_pivot(root, "pull", "x/tools", env=env))
    shutil.rmtree(root / "data" / "x")

    _get_restores_first_and_refuses_executable_tools(root, env, "cache")


def test_get_reports_earlier_targets_when_an_unmarked_tree_member_is_unreadable(
    tmp_path: pathlib.Path,
) -> None:
    """An unmarked SYMLINK cache tree with a member this user cannot read cannot be checked, so
    it is not used: the output is restored file by file from its cache objects, and every
    target is reported."""
    root = _project(
        tmp_path,
        {
            "pipeline.py": _PLAIN_DIR_PIPELINE,
            ".gitignore": "/data/\n__pycache__/\nfirst.txt\n",
            "first.txt": "first\n",
        },
    )
    _ok(_pivot(root, "repro"))
    _ok(_pivot(root, "track", "first.txt"))
    _commit(root, "v1")
    tree_hash, members = _dir_output_record_and_members(root, "x/tools")
    tree = _plant_unmarked_tree(root / ".pivot" / "cache" / "files", tree_hash, members)
    unreadable = tree / "notes.txt"
    unreadable.chmod(0)
    shutil.rmtree(root / "data" / "x")
    (root / "first.txt").unlink()
    tools = root / "data" / "x" / "tools.dat"

    try:
        proc = _pivot(
            root, "get", "--rev", "HEAD", "--checkout-mode", "symlink", "first.txt", "x/tools"
        )
        linked_the_tree = tools.is_symlink()
    finally:
        unreadable.chmod(stat.S_IRUSR | stat.S_IWUSR)
    out = proc.stdout + proc.stderr

    assert proc.returncode == 0, out
    assert "first.txt (from cache)" in out, out
    assert (root / "first.txt").read_text() == "first\n"
    assert "tools.dat (from cache)" in out, out
    assert not linked_the_tree, "the unchecked tree must not be linked into the workspace"
    assert (tools / "notes.txt").read_text() == "plain"


def test_get_reports_earlier_targets_when_a_downloaded_output_cannot_be_written(
    tmp_path: pathlib.Path, moto_services: dict[str, str]
) -> None:
    """A target whose objects download but whose directory this user cannot write fails with the
    reason, and the target restored before it is still reported."""
    env, bucket, _ = _moto_bucket(moto_services["s3"], "readonly")
    root = _project(
        tmp_path,
        {
            ".gitignore": "first.txt\nout/second.txt\n",
            "first.txt": "first\n",
            "out/second.txt": "second\n",
        },
    )
    _ok(_pivot(root, "config", "set", "remotes.origin", f"s3://{bucket}/cache"))
    _ok(_pivot(root, "track", "first.txt", "out/second.txt"))
    second_hash, _ = cache.hash_file(root / "out" / "second.txt")
    _ok(_pivot(root, "push", env=env))
    _commit(root, "track files")
    (root / "first.txt").unlink()
    (root / "out" / "second.txt").unlink()
    cache.get_cache_path(root / ".pivot" / "cache" / "files", second_hash).unlink()
    out_dir = root / "out"
    out_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)

    try:
        proc = _pivot(root, "get", "--rev", "HEAD", "first.txt", "out/second.txt", env=env)
    finally:
        out_dir.chmod(stat.S_IRWXU)
    out = proc.stdout + proc.stderr

    assert proc.returncode == 1, out
    assert "first.txt (from cache)" in out, out
    assert (root / "first.txt").read_text() == "first\n"
    assert "second.txt - failed to restore from remote: [Errno 13] Permission denied" in out, out
    assert not (out_dir / "second.txt").exists()


def test_get_reports_earlier_targets_when_a_target_directory_cannot_be_created(
    tmp_path: pathlib.Path,
) -> None:
    """A target whose directory cannot be made, because a file holds its name, fails with the
    reason, and the target restored before it is still reported."""
    root = _tracked_pair_with_second_uncached(tmp_path, "sub/second.txt")
    shutil.rmtree(root / "sub")
    (root / "sub").write_text("a file where the directory was\n")

    proc = _pivot(root, "get", "--rev", "HEAD", "first.txt", "sub/second.txt")
    out = proc.stdout + proc.stderr

    assert proc.returncode == 1, out
    assert "first.txt (from cache)" in out, out
    assert (root / "first.txt").read_text() == "first\n"
    assert "second.txt - cannot create its directory: [Errno 17] File exists" in out, out


@pytest.mark.parametrize(
    ("pvt", "reason"),
    [
        pytest.param(
            "path: bad.txt\nhash: zzzzzzzzzzzzzzzz\nsize: 1\n",
            "failed to restore from cache: Hash contains invalid characters",
            id="hash-not-a-cache-key",
        ),
        pytest.param(
            "path: bad.txt\nhash: 1234567890123456\nsize: 1\n",
            "bad.txt.pvt at HEAD is not a valid .pvt file: field 'hash' is not a string",
            id="all-digit-hash",
        ),
        pytest.param(
            "path: bad.txt\nhash: 0123456789abcdef\nsize: 1\n"
            + "manifest:\n- relpath: a.txt\n  size: 1\n  isexec: false\n",
            "bad.txt.pvt at HEAD is not a valid .pvt file: field 'manifest[0].hash' is missing",
            id="manifest-entry-without-hash",
        ),
        pytest.param(
            "path: bad.txt\nhash: 0123456789abcdef\nsize: 1\n"
            + 'manifest:\n- relpath: "a\\0b"\n  hash: 0123456789abcdef\n  size: 1\n'
            + "  isexec: false\n",
            "bad.txt.pvt at HEAD is not a valid .pvt file: field 'manifest[0].relpath' holds a "
            + "NUL byte",
            id="manifest-entry-nul-relpath",
        ),
        pytest.param(
            "path: bad.txt\nhash: 0123456789abcdef\nsize: 1\n"
            + "manifest:\n- relpath: a.txt\n  hash: 0123456789abcdef\n  size: 1\n"
            + '  isexec: false\nempty_dirs:\n- "e\\0"\n',
            "bad.txt.pvt at HEAD is not a valid .pvt file: field 'empty_dirs[0]' holds a NUL byte",
            id="nul-empty-dir",
        ),
    ],
)
def test_get_reports_earlier_targets_when_a_tracked_file_has_an_invalid_pvt(
    tmp_path: pathlib.Path, pvt: str, reason: str
) -> None:
    """A committed .pvt that is hand-written wrong (a hash that is not a cache key, an all-digit
    hash YAML reads as an integer, a manifest entry with no hash, a NUL byte in a relpath or an
    empty directory) fails that target with the reason, naming the file and the field where the
    .pvt itself is invalid, and the target restored before it is still reported."""
    root = _project(tmp_path, {".gitignore": "first.txt\n", "first.txt": "first\n"})
    _ok(_pivot(root, "track", "first.txt"))
    (root / "bad.txt.pvt").write_text(pvt)
    _commit(root, "track files")
    (root / "first.txt").unlink()

    proc = _pivot(root, "get", "--rev", "HEAD", "first.txt", "bad.txt")
    out = proc.stdout + proc.stderr

    assert proc.returncode == 1, out
    assert "first.txt (from cache)" in out, out
    assert (root / "first.txt").read_text() == "first\n"
    assert f"bad.txt - {reason}" in out, out
    assert not (root / "bad.txt").exists()


@pytest.mark.parametrize(
    ("force", "returncode", "reported"),
    [
        pytest.param(False, 0, "Skipped: ", id="kept"),
        pytest.param(
            True,
            1,
            "bad.txt.pvt at HEAD is not a valid .pvt file: field 'hash' is not a string",
            id="force",
        ),
    ],
)
def test_get_skips_a_workspace_file_whose_pvt_is_invalid_before_checking_it(
    tmp_path: pathlib.Path, force: bool, returncode: int, reported: str
) -> None:
    """A target already in the workspace is skipped without --force, as for a valid .pvt, so the
    .pvt field check refuses nothing get skipped before it; with --force the target fails naming
    the field and the workspace file is left as it was."""
    root = _project(tmp_path, {".gitignore": "bad.txt\n", "bad.txt": "here\n"})
    (root / "bad.txt.pvt").write_text("path: bad.txt\nhash: 1234567890123456\nsize: 1\n")
    _commit(root, "bad pvt")

    proc = _pivot(root, "get", "--rev", "HEAD", *(["--force"] if force else []), "bad.txt")
    out = proc.stdout + proc.stderr

    assert proc.returncode == returncode, out
    assert reported in out, out
    assert (root / "bad.txt").read_text() == "here\n"


def test_get_refuses_an_unknown_remote_name_before_restoring_anything(
    tmp_path: pathlib.Path,
) -> None:
    """An explicit --remote that names no configured remote fails at once, even when every
    target is in the cache and the remote would never be opened."""
    root = _project(tmp_path, {".gitignore": "artifact.txt\n", "artifact.txt": "cached\n"})
    _ok(_pivot(root, "config", "set", "remotes.origin", "s3://bucket-a/cache"))
    _ok(_pivot(root, "track", "artifact.txt"))
    _commit(root, "track artifact")
    (root / "artifact.txt").unlink()

    proc = _pivot(root, "get", "--remote", "orign", "--rev", "HEAD", "artifact.txt")
    out = proc.stdout + proc.stderr

    assert proc.returncode == 1, out
    assert "Remote 'orign' not found" in out, out
    assert not (root / "artifact.txt").exists()


def _unmarked_shards_tree_with_stale_member(
    tmp_path: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path, str]:
    """A project whose d/shards output is gone from the workspace and whose unmarked SYMLINK
    cache tree holds every recorded file at its recorded size, 0.txt with stale bytes. Returns
    the root, the tree and 0.txt's recorded hash."""
    root = _project(
        tmp_path,
        {
            "pipeline.py": _DIROUT_PIPELINE,
            "data/raw/t.csv": "a,b\n5,6\n7,8\n",
            ".gitignore": "/data/d/\n__pycache__/\n",
        },
    )
    _ok(_pivot(root, "repro"))
    tree_hash, members = _dir_output_record_and_members(root, "d/shards")
    cache_dir = root / ".pivot" / "cache" / "files"
    tree = _plant_unmarked_tree(cache_dir, tree_hash, members)
    (tree / "0.txt").write_text("xx")  # stale, the size of "11"
    shutil.rmtree(root / "data" / "d" / "shards.dat")
    stale_entry = next(e for e in members["files"] if e["relpath"] == "0.txt")
    return root, tree, stale_entry["hash"]


def test_symlink_checkout_restores_file_by_file_past_stale_bytes_in_a_complete_unmarked_tree(
    tmp_path: pathlib.Path,
) -> None:
    """An unmarked SYMLINK cache tree with stale bytes is not trusted. Checkout restores each
    file from its object and leaves the unmarked tree unchanged."""
    root, tree, _ = _unmarked_shards_tree_with_stale_member(tmp_path)
    shards = root / "data" / "d" / "shards.dat"

    proc = _pivot(root, "checkout", "--checkout-mode", "symlink", "d/shards")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (shards / "0.txt").read_text() == "11"
    assert (shards / "1.txt").read_text() == "15"
    assert not shards.is_symlink()
    assert (tree / "0.txt").read_text() == "xx"


# The mark pivot 0.3.5 to 0.5.11 put beside a tree after checking its files' sizes, not content
_SIZE_CHECKED_MARK = ".verified"


@pytest.mark.parametrize(
    ("setup", "command"),
    [
        pytest.param((), ("checkout", "--checkout-mode", "symlink", "d/shards"), id="checkout"),
        pytest.param(
            (), ("get", "--rev", "HEAD", "--checkout-mode", "symlink", "d/shards"), id="get"
        ),
        pytest.param(
            (("config", "set", "cache.checkout_mode", "symlink,hardlink,copy"),),
            ("repro", "--force"),
            id="repro-force",
        ),
    ],
)
def test_symlink_restore_does_not_trust_a_stale_tree_an_older_pivot_marked_after_a_size_check(
    tmp_path: pathlib.Path, setup: tuple[tuple[str, ...], ...], command: tuple[str, ...]
) -> None:
    """A checkout or get never repairs from the supplied listing: an old size-only marker leaves
    the tree untrusted and restores file by file. A forced repro hashes real output bytes, so its
    commit may repair and mark the tree."""
    root, tree, _ = _unmarked_shards_tree_with_stale_member(tmp_path)
    old_mark = tree.with_name(tree.name + _SIZE_CHECKED_MARK)
    old_mark.touch()
    for args in setup:
        _ok(_pivot(root, *args))
    _commit(root, "v1")
    shards = root / "data" / "d" / "shards.dat"

    proc = _pivot(root, *command)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (shards / "0.txt").read_text() == "11"
    if command[0] == "repro":
        assert shards.resolve() == tree.resolve()
        assert not old_mark.exists()
    else:
        assert not shards.is_symlink()
        assert old_mark.exists()


@pytest.mark.skipif(not os.environ.get("CI"), reason="installs pivot 0.5.11 from PyPI; runs in CI")
def test_symlink_checkout_restores_file_by_file_past_a_stale_tree_pivot_0_5_11_built_and_marked(
    tmp_path: pathlib.Path,
) -> None:
    """Pivot 0.5.11 builds a project, then links a stale same-size SYMLINK tree after a size
    check and marks it. After the upgrade, this build does not trust that marker: it restores the
    directory file by file and leaves the tree as it is.

    The other order fails loudly: pivot 0.5.11 cannot read the lock this build writes for a
    directory, so on a project this build made its checkout refuses the stage and writes
    nothing."""
    released = helpers.venv_pivot(tmp_path / "venv", "pivot==0.5.11")
    root = tmp_path / "project"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    _ok(_pivot(root, "init", executable=released))
    (root / "pipeline.py").write_text(_DIROUT_PIPELINE)
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "t.csv").write_text("a,b\n5,6\n7,8\n")
    (root / ".gitignore").write_text("/data/d/\n__pycache__/\n")
    _ok(_pivot(root, "repro", executable=released))

    out = yaml.safe_load((root / ".pivot" / "stages" / "d" / "shards.lock").read_text())["outs"][0]
    assert "manifest" in out, "precondition: pivot 0.5.11 lists the directory inline"
    files_dir = root / ".pivot" / "cache" / "files"
    tree = cache.get_cache_path(files_dir, out["hash"])  # where pivot 0.5.11 keeps a tree
    assert not tree.exists(), "precondition: hardlink mode builds no tree"
    tree.mkdir(parents=True)
    for entry in out["manifest"]:
        (tree / entry["relpath"]).write_bytes(
            cache.get_cache_path(files_dir, entry["hash"]).read_bytes()
        )
    (tree / "0.txt").write_text("xx")  # stale, the size of "11"
    shards = root / "data" / "d" / "shards.dat"
    shutil.rmtree(shards)
    old = _pivot(root, "checkout", "--checkout-mode", "symlink", "d/shards", executable=released)
    assert old.returncode == 0, old.stdout + old.stderr
    assert (shards / "0.txt").read_text() == "xx", "precondition: pivot 0.5.11 links stale bytes"
    assert tree.with_name(tree.name + _SIZE_CHECKED_MARK).exists(), "precondition: and marks them"
    assert shards.is_symlink(), "precondition: as a link to the tree"
    shards.unlink()

    proc = _pivot(root, "checkout", "--checkout-mode", "symlink", "d/shards")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (shards / "0.txt").read_text() == "11"
    assert (shards / "1.txt").read_text() == "15"
    assert not shards.is_symlink()
    assert (tree / "0.txt").read_text() == "xx", "the stale tree is left as it is"

    built = tmp_path / "built"
    built.mkdir()
    built = _project(
        built,
        {
            "pipeline.py": _DIROUT_PIPELINE,
            "data/raw/t.csv": "a,b\n5,6\n7,8\n",
            ".gitignore": "/data/d/\n__pycache__/\n",
        },
    )
    _ok(_pivot(built, "repro"))
    built_shards = built / "data" / "d" / "shards.dat"
    shutil.rmtree(built_shards)

    refused = _pivot(
        built, "checkout", "--checkout-mode", "symlink", "d/shards", executable=released
    )

    refusal = refused.stdout + refused.stderr
    assert refused.returncode != 0, refusal
    assert "Stage 'd/shards' has no cached outputs" in refusal, refusal
    assert not built_shards.exists() and not built_shards.is_symlink(), refusal


@pytest.mark.parametrize(
    "marks",
    [pytest.param((), id="unmarked"), pytest.param((_SIZE_CHECKED_MARK,), id="size-checked")],
)
def test_symlink_checkout_with_stale_tree_and_missing_object_writes_nothing(
    tmp_path: pathlib.Path, marks: tuple[str, ...]
) -> None:
    """A stale tree is not trusted. When the stale member's cache object is missing, checkout
    cannot restore file by file. It fails and leaves the workspace without the output."""
    root, tree, stale_hash = _unmarked_shards_tree_with_stale_member(tmp_path)
    for suffix in marks:
        tree.with_name(tree.name + suffix).touch()
    cache.get_cache_path(root / ".pivot" / "cache" / "files", stale_hash).unlink()
    shards = root / "data" / "d" / "shards.dat"

    proc = _pivot(root, "checkout", "--checkout-mode", "symlink", "d/shards")

    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert not shards.exists() and not shards.is_symlink()


_ALIVE_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def alive() -> str:
    return "alive"


with Pipeline("a") as pipeline:
    alive()
"""


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(("status", "--remote"), id="status"),
        pytest.param(("verify", "--allow-missing"), id="verify"),
        pytest.param(("pull",), id="pull"),
        pytest.param(("get", "--rev", "HEAD", "a/alive"), id="get"),
    ],
)
def test_multi_remote_advice_names_an_option_the_command_takes(
    tmp_path: pathlib.Path, args: tuple[str, ...]
) -> None:
    """With two remotes and no default, every command that reports it tells the user how to
    pick a remote, and each command the advice names takes the option it names."""
    root = _project(tmp_path, {"pipeline.py": _ALIVE_PIPELINE, ".gitignore": "/data/\n"})
    _ok(_pivot(root, "repro"))
    _commit(root, "alive")
    (root / "data" / "a" / "alive.txt").unlink()
    for obj in list((root / ".pivot" / "cache" / "files").rglob("*")):
        if obj.is_file() and not obj.name.startswith("."):
            obj.unlink()  # `get` must reach its remote fallback
    _ok(_pivot(root, "config", "set", "remotes.origin", "s3://bucket-a/cache"))
    _ok(_pivot(root, "config", "set", "remotes.backup", "s3://bucket-b/cache"))

    proc = _pivot(root, *args)
    out = " ".join((proc.stdout + proc.stderr).split())

    assert proc.returncode != 0, out
    assert "multiple remotes configured" in out, out
    assert "pivot config set default_remote <name>" in out, out
    advice = re.search(r"Pass (--[\w-]+) <name> to (.+?), or run", out)
    assert advice is not None, out
    option = advice.group(1)
    for command in re.split(r", | or ", advice.group(2)):
        help_text = " ".join(_ok(_pivot(root, command, "--help")).split())
        assert f"{option} TEXT" in help_text, (
            f"the advice names `pivot {command} {option} <name>`, which it does not take:\n{out}"
        )


_DEMO_TRAIN_PIPELINE = """\
import pandas as pd

from pivot import Pipeline, stage


@stage
def train() -> pd.DataFrame:
    return pd.DataFrame({"score": [0.25, 0.75]})


with Pipeline("demo") as pipeline:
    train()
"""


def test_documented_get_from_a_specific_remote_example_runs(
    tmp_path: pathlib.Path, moto_services: dict[str, str]
) -> None:
    """docs/cli/index.md's `pivot get` example "Get from a specific remote" works as written in
    a project where demo/train writes data/demo/train.jsonl, the path Pivot gives it."""
    docs = (pathlib.Path(__file__).parents[4] / "docs" / "cli" / "index.md").read_text()
    example = re.search(r"# Get from a specific remote\n(pivot get [^\n]+)\n", docs)
    assert example is not None, "the example is gone from docs/cli/index.md"
    argv = shlex.split(example.group(1))[1:]

    env, bucket, _ = _moto_bucket(moto_services["s3"], "docs")
    root = _project(tmp_path, {"pipeline.py": _DEMO_TRAIN_PIPELINE, ".gitignore": "/data/\n"})
    _ok(_pivot(root, "config", "set", "remotes.origin", f"s3://{bucket}/cache"))
    _ok(_pivot(root, "repro", env=env))
    output = root / "data" / "demo" / "train.jsonl"
    assert output.exists(), "demo/train's output path changed; update this test's premise"
    expected = output.read_bytes()
    _ok(_pivot(root, "push", env=env))
    _commit(root, "train")
    rev = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    output.unlink()
    for obj in list((root / ".pivot" / "cache" / "files").rglob("*")):
        if obj.is_file() and not obj.name.startswith("."):
            obj.unlink()

    argv = [rev if arg == "abc123" else arg for arg in argv]
    proc = _pivot(root, *argv, env=env)

    assert proc.returncode == 0, f"pivot {' '.join(argv)}\n{proc.stdout}{proc.stderr}"
    assert output.read_bytes() == expected


def test_pull_replaces_directory_output_after_teammate_rerun(
    tmp_path: pathlib.Path, moto_services: dict[str, str]
) -> None:
    """Files pivot wrote earlier are in the local cache, so pull replaces them with the new version."""
    clone, env = _dirout_clone_after_teammate_rerun(tmp_path, moto_services["s3"])
    shards = clone / "data" / "d" / "shards.dat"
    assert (shards / "0.txt").read_text() == "3", "the dev's workspace should still hold v1"

    _ok(_pivot(clone, "pull", "d/shards", env=env))

    assert sorted(p.name for p in shards.iterdir()) == ["0.txt", "1.txt"]
    assert (shards / "0.txt").read_text() == "11"
    assert (shards / "1.txt").read_text() == "15"


def test_pull_refuses_directory_output_with_hand_edited_file(
    tmp_path: pathlib.Path, moto_services: dict[str, str]
) -> None:
    """A hand edit is not in the cache, so pull refuses, names only that file and keeps it."""
    clone, env = _dirout_clone_after_teammate_rerun(tmp_path, moto_services["s3"])
    shards = clone / "data" / "d" / "shards.dat"
    (shards / "0.txt").unlink()
    (shards / "0.txt").write_text("mine")

    proc = _pivot(clone, "pull", "d/shards", env=env)
    out = proc.stdout + proc.stderr

    assert proc.returncode == 1, out
    assert "'shards.dat' has local changes that are not in the cache: 0.txt. Use --force" in out, (
        out
    )
    assert (shards / "0.txt").read_text() == "mine", "the edit must survive"
    assert (shards / "1.txt").read_text() == "7", "a refused pull must not restore anything"


def test_pull_restores_build_dir_inside_directory_output(
    tmp_path: pathlib.Path, moto_services: dict[str, str]
) -> None:
    """A build directory is cached and restored with the rest of its artifact."""
    clone, env = _dirout_clone_after_teammate_rerun(
        tmp_path, moto_services["s3"], _DIROUT_BUILD_PIPELINE
    )
    shards = clone / "data" / "d" / "shards.dat"
    assert (shards / "build" / "log.txt").read_text() == "built 10"

    _ok(_pivot(clone, "pull", "d/shards", env=env))

    assert sorted(p.name for p in shards.iterdir()) == ["0.txt", "1.txt", "build"]
    assert (shards / "0.txt").read_text() == "11"
    assert (shards / "build" / "log.txt").read_text() == "built 26"


_ARTIFACT_SYMLINK_RESTORE_PIPELINE = """\
import pathlib
import tempfile

from pivot import Pipeline, stage


@stage
def assemble() -> pathlib.Path:
    output = pathlib.Path(tempfile.mkdtemp())
    for relpath, content in {
        "keep.txt": "keep",
        "build/required.txt": "build",
        "dist/required.txt": "dist",
        "node_modules/required.txt": "modules",
        ".venv/required.txt": "venv",
        "editor.swp": "editor",
        "nested/sub/file.txt": "nested",
    }.items():
        path = output / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return output


with Pipeline("dir") as pipeline:
    assemble()
"""


def test_pull_keeps_cache_derived_directory_file_symlinks_as_artifact_members(
    tmp_path: pathlib.Path, moto_services: dict[str, str]
) -> None:
    """A pull that restores a directory file-by-file leaves a usable directory artifact."""
    env = _env(
        AWS_ENDPOINT_URL=moto_services["s3"],
        AWS_ACCESS_KEY_ID="x",
        AWS_SECRET_ACCESS_KEY="x",
        AWS_DEFAULT_REGION="us-east-1",
    )
    bucket = f"artifact-links-{uuid.uuid4().hex[:8]}"
    s3 = boto3.client(
        "s3",
        endpoint_url=moto_services["s3"],
        region_name="us-east-1",
        aws_access_key_id="x",
        aws_secret_access_key="x",
    )
    s3.create_bucket(Bucket=bucket)  # pyright: ignore[reportAttributeAccessIssue] - boto3 ships no stubs

    src = tmp_path / "src"
    src.mkdir()
    _project(
        src,
        {
            "pipeline.py": _ARTIFACT_SYMLINK_RESTORE_PIPELINE,
            ".gitignore": "/data/dir/\n__pycache__/\n",
        },
    )
    _ok(_pivot(src, "config", "set", "remotes.origin", f"s3://{bucket}/cache"))
    _ok(_pivot(src, "config", "set", "default_remote", "origin"))
    _ok(_pivot(src, "config", "set", "cache.checkout_mode", "copy"))
    _ok(_pivot(src, "repro", env=env))
    _ok(_pivot(src, "push", env=env))
    _commit(src, "source")

    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(src), str(clone)], check=True)
    _ok(_pivot(clone, "config", "set", "cache.checkout_mode", "symlink"))

    _ok(_pivot(clone, "pull", "dir/assemble", env=env))

    output = clone / "data" / "dir" / "assemble.dat"
    expected = {
        "keep.txt": "keep",
        "build/required.txt": "build",
        "dist/required.txt": "dist",
        "node_modules/required.txt": "modules",
        ".venv/required.txt": "venv",
        "editor.swp": "editor",
        "nested/sub/file.txt": "nested",
    }
    assert {relpath: (output / relpath).read_text() for relpath in expected} == expected
    assert all((output / relpath).is_symlink() for relpath in expected)
    members = _helper_output_listing(clone, "dir/assemble")
    assert sorted(entry["relpath"] for entry in members["files"]) == sorted(expected)

    assert "1 cached" in _ok(_pivot(clone, "status", env=env))
    assert "0 ran, 1 cached" in _ok(_pivot(clone, "repro", env=env))
    assert "Verification passed" in _ok(_pivot(clone, "verify", env=env))
    assert "leaves out symlink" not in _ok(_pivot(clone, "commit", "dir/assemble", env=env))

    assert _helper_output_listing(clone, "dir/assemble") == members
    assert "1 cached" in _ok(_pivot(clone, "status", env=env))


_EMPTY_DIRS_PIPELINE = """\
import pathlib
import tempfile
from typing import Annotated, TypedDict

from pivot import Pipeline, loaders, stage


class Made(TypedDict):
    tree: pathlib.Path


@stage
def make() -> Made:
    tree = pathlib.Path(tempfile.mkdtemp())
    (tree / "keep.txt").write_text("keep")
    (tree / "bundle" / "classifier_source").mkdir(parents=True)
    (tree / "logs").mkdir()
    return {"tree": tree}


@stage
def pack(tree: pathlib.Path) -> Annotated[dict, loaders.JSON()]:
    return {
        "entries": sorted(
            str(p.relative_to(tree)) + ("/" if p.is_dir() else "") for p in tree.rglob("*")
        )
    }


with Pipeline("empty") as pipeline:
    made = make()
    pack(made.tree)
"""


@pytest.mark.parametrize("mode", ["hardlink", "symlink", "copy"])
def test_pull_restores_empty_dirs_of_directory_output(
    tmp_path: pathlib.Path, moto_services: dict[str, str], mode: str
) -> None:
    """A directory output pulled into a fresh clone is the tree its stage built, empty
    directories included: the clone is up to date, and its consumer packs the same bytes."""
    env = _env(
        AWS_ENDPOINT_URL=moto_services["s3"],
        AWS_ACCESS_KEY_ID="x",
        AWS_SECRET_ACCESS_KEY="x",
        AWS_DEFAULT_REGION="us-east-1",
    )
    bucket = f"empty-dirs-{uuid.uuid4().hex[:8]}"
    s3 = boto3.client(
        "s3",
        endpoint_url=moto_services["s3"],
        region_name="us-east-1",
        aws_access_key_id="x",
        aws_secret_access_key="x",
    )
    s3.create_bucket(Bucket=bucket)  # pyright: ignore[reportAttributeAccessIssue] - boto3 ships no stubs

    src = tmp_path / "src"
    src.mkdir()
    _project(
        src, {"pipeline.py": _EMPTY_DIRS_PIPELINE, ".gitignore": "/data/empty/\n__pycache__/\n"}
    )
    _ok(_pivot(src, "config", "set", "remotes.origin", f"s3://{bucket}/cache"))
    _ok(_pivot(src, "config", "set", "default_remote", "origin"))
    _ok(_pivot(src, "repro", env=env))
    _ok(_pivot(src, "push", env=env))
    _commit(src, "source")
    tree = pathlib.Path("data", "empty", "make", "tree.dat")
    packed = pathlib.Path("data", "empty", "pack.json")
    assert sorted(p.name for p in (src / tree).iterdir()) == ["bundle", "keep.txt", "logs"]
    members = _helper_output_listing(src, "empty/make", "tree")
    assert [entry["relpath"] for entry in members["files"]] == ["keep.txt"]
    assert members["empty_dirs"] == ["bundle/classifier_source", "logs"]

    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(src), str(clone)], check=True)
    _ok(_pivot(clone, "config", "set", "cache.checkout_mode", mode))
    _ok(_pivot(clone, "pull", "empty/make", "empty/pack", env=env))

    assert (clone / tree / "bundle" / "classifier_source").is_dir()
    assert (clone / tree / "logs").is_dir()
    assert "0 ran, 2 cached" in _ok(_pivot(clone, "repro", env=env))
    assert "2 cached" in _ok(_pivot(clone, "status", env=env))
    _ok(_pivot(clone, "run", "--force", "empty/pack", env=env))
    assert (clone / packed).read_text() == (src / packed).read_text()


def _helper_rewrite_lock_as_pivot_033(
    lock_path: pathlib.Path, files: list[types.DirManifestEntry]
) -> int:
    """Rewrite a lock the way pivot 0.3.3 wrote it: schema version 2, and each directory entry
    listing ``files`` inline, without the empty directories, under the hash 0.3.3 gave them
    (the xxhash64 of their sorted-key compact JSON). Return how many entries changed."""
    old_hash = xxhash.xxh64(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    data = yaml.safe_load(lock_path.read_text())
    records = data.pop("records")
    data["schema_version"] = 2
    data["deps"], data["outs"] = records["deps"], records["outs"]
    changed = 0
    for entry in [*data["deps"], *data["outs"]]:
        if "nfiles" not in entry:
            continue
        del entry["size"], entry["nfiles"], entry["listing"]
        entry["hash"] = old_hash
        entry["manifest"] = [dict(file) for file in files]
        changed += 1
    lock_path.write_text(yaml.safe_dump(data, sort_keys=False))
    return changed


@pytest.mark.parametrize("state", ["run-cache-kept", "state-cleared"])
def test_repro_rebuilds_a_pivot_033_lock_and_keeps_the_empty_dirs_it_did_not_record(
    tmp_path: pathlib.Path, state: str
) -> None:
    """A directory output whose lock pivot 0.3.3 wrote lists only its files, while its tree on
    disk holds empty directories. Pivot reads each such lock as it is, warning once that it is
    from an older pivot; the output is not restored to the lock, which would delete the
    directories while every stage stayed cached. The stage runs (or takes its outputs from a
    run-cache entry that records the directories), both locks and the committed listing come
    back byte for byte as a fresh build writes them, and the consumer reads the tree a fresh
    build makes."""
    (tmp_path / "project").mkdir()
    root = _project(tmp_path / "project", {"pipeline.py": _EMPTY_DIRS_PIPELINE})
    _ok(_pivot(root, "repro"))
    tree = root / "data" / "empty" / "make" / "tree.dat"
    packed = (root / "data" / "empty" / "pack.json").read_text()
    stages = root / ".pivot" / "stages" / "empty"
    committed = ("make.lock", "pack.lock", "make=tree.listing")
    fresh = {name: (stages / name).read_bytes() for name in committed}
    members = _helper_output_listing(root, "empty/make", "tree")
    assert members["empty_dirs"] == ["bundle/classifier_source", "logs"]
    assert _helper_rewrite_lock_as_pivot_033(stages / "make.lock", members["files"]) == 1
    assert _helper_rewrite_lock_as_pivot_033(stages / "pack.lock", members["files"]) == 1
    (stages / "make=tree.listing").unlink()  # pivot 0.3.3 wrote no listing file
    if state == "state-cleared":
        shutil.rmtree(root / ".pivot" / "state.lmdb")

    status_proc = _pivot(root, "status", "--explain", "--json")
    _ok(status_proc)
    status = {s["name"]: s for s in json.loads(status_proc.stdout)["stages"]}
    assert {name: s["status"] for name, s in status.items()} == {
        "empty/make": "stale",
        "empty/pack": "stale",
    }, status
    assert status["empty/make"]["reason"].startswith(
        "Output holds empty directories its lock does not record"
    ), status
    warning = "was written by an older pivot; pivot reads it as it is"
    assert " ".join(status_proc.stderr.split()).count(warning) == 2, status_proc.stderr

    _ok(_pivot(root, "repro"))

    assert (tree / "bundle" / "classifier_source").is_dir()
    assert (tree / "logs").is_dir()
    assert (root / "data" / "empty" / "pack.json").read_text() == packed
    assert {name: (stages / name).read_bytes() for name in committed} == fresh
    assert "0 ran, 2 cached" in _ok(_pivot(root, "repro"))


def test_tracked_directory_checked_out_as_symlinks_stays_clean(tmp_path: pathlib.Path) -> None:
    """A symlink checkout of a tracked directory is the tracked content, not a modification."""
    root = _project(tmp_path, {"data/tracked/keep.txt": "keep", "data/tracked/nested/f.txt": "f"})
    _ok(_pivot(root, "track", "data/tracked"))
    shutil.rmtree(root / "data" / "tracked")
    _ok(_pivot(root, "checkout", "--checkout-mode", "symlink", "data/tracked"))
    assert (root / "data" / "tracked" / "keep.txt").is_symlink()

    assert "1 clean" in _ok(_pivot(root, "status", "--tracked-only"))
    assert "leaves out symlink" not in _ok(_pivot(root, "track", "--force", "data/tracked"))

    members = _helper_tracked_listing(root, "data/tracked")
    assert sorted(entry["relpath"] for entry in members["files"]) == ["keep.txt", "nested/f.txt"]


def test_only_links_into_the_configured_cache_count_as_directory_content(
    tmp_path: pathlib.Path,
) -> None:
    """A link to a content-addressed file outside the cache is a user symlink, not content."""
    foreign_content = b"seed bytes"
    digest = xxhash.xxh64(foreign_content).hexdigest()
    foreign_object = tmp_path / "foreign-content-store" / digest[:2] / digest[2:]
    foreign_object.parent.mkdir(parents=True)
    foreign_object.write_bytes(foreign_content)
    real_cache = tmp_path / "real-cache"
    real_cache.mkdir()
    cache_link = tmp_path / "cache-link"
    cache_link.symlink_to(real_cache)
    root = tmp_path / "project"
    root.mkdir()
    _project(root, {"data/tracked/keep.txt": "keep", "data/tracked/nested/f.txt": "f"})
    _ok(_pivot(root, "config", "set", "cache.dir", str(cache_link)))
    tracked = root / "data" / "tracked"
    (tracked / "foreign.txt").symlink_to(foreign_object)

    out = _ok(_pivot(root, "track", "data/tracked"))

    assert "Directory artifact data/tracked leaves out symlink foreign.txt" in " ".join(
        out.split()
    ), out
    pvt_path = root / "data" / "tracked.pvt"
    pvt = pvt_path.read_text()
    members = _helper_tracked_listing(root, "data/tracked")
    assert sorted(entry["relpath"] for entry in members["files"]) == ["keep.txt", "nested/f.txt"]

    # A real symlink checkout through the symlinked custom cache root is still content.
    shutil.rmtree(tracked)
    _ok(_pivot(root, "checkout", "--checkout-mode", "symlink", "data/tracked"))
    assert (tracked / "keep.txt").resolve().is_relative_to((real_cache / "files").resolve())
    assert "1 clean" in _ok(_pivot(root, "status", "--tracked-only"))
    _ok(_pivot(root, "checkout", "data/tracked"))  # existing links match: skipped, not an error
    assert "leaves out symlink" not in _ok(_pivot(root, "track", "--force", "data/tracked"))
    assert pvt_path.read_text() == pvt
    assert _helper_tracked_listing(root, "data/tracked") == members


_PARAM_PIPELINE = """\
import pandas as pd

from pivot import Pipeline, StageParams, stage


class ScaleParams(StageParams):
    factor: int = 10


@stage
def scale(raw: pd.DataFrame, params: ScaleParams) -> pd.DataFrame:
    return raw * params.factor


with Pipeline("p") as pipeline:
    scale(pipeline.input("t.csv"), params=ScaleParams())
"""


def test_noop_status_reads_no_output_bytes_after_run_cache_restore(
    tmp_path: pathlib.Path,
) -> None:
    """A run-cache restore records the restored outputs' hashes."""
    root = _project(tmp_path, {"pipeline.py": _PARAM_PIPELINE, "data/raw/t.csv": "a\n1\n2\n"})
    _ok(_pivot(root, "repro"))
    (root / "params.yaml").write_text("p/scale:\n  factor: 20\n")
    _ok(_pivot(root, "repro"))
    (root / "params.yaml").unlink()
    restored = _ok(_pivot(root, "repro"))
    assert "0 ran, 1 cached" in restored, restored

    out = _ok(_pivot(root, "status", env=_env(PIVOT_METRICS="1")))

    assert "1 cached" in out, out
    assert "cache.hash_file" not in out, out


def test_noop_status_reads_no_output_bytes_after_skip_restore(tmp_path: pathlib.Path) -> None:
    """A skipped stage that copies a deleted output back records the copy's hash."""
    root = _project(tmp_path, {"pipeline.py": _CHAIN_PIPELINE, "data/raw/t.csv": "a\n1\n2\n"})
    _ok(_pivot(root, "config", "set", "cache.checkout_mode", "copy"))
    _ok(_pivot(root, "repro"))
    (root / "data" / "k" / "scale.jsonl").unlink()
    restored = _ok(_pivot(root, "repro"))
    assert "0 ran, 2 cached" in restored, restored

    out = _ok(_pivot(root, "status", env=_env(PIVOT_METRICS="1")))

    assert "2 cached" in out, out
    assert "cache.hash_file" not in out, out


_INPUT_NAMED_LIKE_STAGE_PIPELINE = """\
import pandas as pd

from pivot import Pipeline, stage


@stage
def summary(numbers: pd.DataFrame) -> pd.DataFrame:
    return numbers * 7


@stage
def report(prior: pd.DataFrame) -> dict:
    return {"prior_sum": int(prior.to_numpy().sum())}


with Pipeline("rep") as pipeline:
    summary(pipeline.input("numbers.csv"))
    report(pipeline.input("rep/summary", path="data/raw/prior.jsonl"))
"""


def test_input_named_like_a_stage_is_refused(tmp_path: pathlib.Path) -> None:
    """An input and a stage share one identity namespace, so one name cannot be both."""
    root = _project(
        tmp_path,
        {
            "pipeline.py": _INPUT_NAMED_LIKE_STAGE_PIPELINE,
            "data/raw/numbers.csv": "a\n1\n2\n",
            "data/raw/prior.jsonl": '{"a":100}\n',
        },
    )

    proc = _pivot(root, "list")
    out = proc.stdout + proc.stderr

    assert proc.returncode == 1, out
    assert "Input 'rep/summary'" in out and "stage 'rep/summary'" in out, out


_OTHER_PIPELINE = """\
import pandas as pd

from pivot import Pipeline, stage


@stage
def report(prior: pd.DataFrame) -> dict:
    return {"prior_sum": int(prior.to_numpy().sum())}


with Pipeline("other") as pipeline:
    report(pipeline.input("k/scale", path="data/raw/prior.jsonl"))
"""


def test_input_reads_its_own_file_when_another_pipelines_output_has_its_name(
    tmp_path: pathlib.Path,
) -> None:
    """Whether a dependency is an input is decided by the pipeline, never by what is on disk."""
    root = _project(
        tmp_path,
        {
            "pipeline.py": _CHAIN_PIPELINE,
            "data/raw/t.csv": "a\n1\n2\n",
            "data/raw/prior.jsonl": '{"a":100}\n',
            "other/pipeline.py": _OTHER_PIPELINE,
        },
    )
    other = root / "other"
    _ok(_pivot(other, "repro"))
    _ok(_pivot(root, "repro"))
    assert (root / "data" / "k" / "scale.jsonl").is_file()

    status = _ok(_pivot(other, "status"))
    rerun = _ok(_pivot(other, "repro"))

    assert "1 cached, 0 stale" in status, status
    assert "0 ran, 1 cached" in rerun, rerun
    assert (root / "data" / "other" / "report.yaml").read_text() == "prior_sum: 100\n"


_TWO_INPUTS_PIPELINE = """\
import pandas as pd

from pivot import Pipeline, stage


@stage
def rows(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    return {"rows": int(len(train) + len(test))}


with Pipeline("sym") as pipeline:
    rows(pipeline.input("train.csv"), pipeline.input("test.csv"))
"""


def test_two_inputs_symlinked_to_one_file_run(tmp_path: pathlib.Path) -> None:
    """Inputs are told apart by their declared paths, not by the files the paths point to."""
    root = _project(tmp_path, {"pipeline.py": _TWO_INPUTS_PIPELINE, "shared/base.csv": "a\n1\n2\n"})
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "train.csv").symlink_to("../../shared/base.csv")
    (root / "data" / "raw" / "test.csv").symlink_to("../../shared/base.csv")

    out = _ok(_pivot(root, "repro"))

    assert "1 ran" in out, out
    assert (root / "data" / "sym" / "rows.yaml").read_text() == "rows: 4\n"


_SAME_CONTENT_PIPELINE = """\
import pandas as pd

from pivot import Pipeline, stage


@stage
def left(numbers: pd.DataFrame) -> dict:
    return {"rows": int(len(numbers))}


@stage
def right(numbers: pd.DataFrame) -> dict:
    return {"rows": int(len(numbers))}


@stage
def both(a: dict, b: dict) -> dict:
    return {"total": a["rows"] + b["rows"]}


with Pipeline("sym") as pipeline:
    numbers = pipeline.input("numbers.csv")
    both(left(numbers), right(numbers))
"""


def test_outputs_symlinked_to_one_cache_object_feed_one_stage(tmp_path: pathlib.Path) -> None:
    """Two equal outputs checked out as symlinks share a cache object; their consumer still runs."""
    root = _project(
        tmp_path, {"pipeline.py": _SAME_CONTENT_PIPELINE, "data/raw/numbers.csv": "a\n1\n2\n"}
    )
    _ok(_pivot(root, "config", "set", "cache.checkout_mode", "symlink"))
    _ok(_pivot(root, "repro"))
    left, right = root / "data" / "sym" / "left.yaml", root / "data" / "sym" / "right.yaml"
    assert left.resolve() == right.resolve()

    out = _ok(_pivot(root, "run", "sym/both", "--force"))

    assert "1 ran" in out, out
    assert (root / "data" / "sym" / "both.yaml").read_text() == "total: 4\n"


_LINK_LIMIT_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def left() -> str:
    return ""


@stage
def right() -> str:
    return ""


with Pipeline("lim") as pipeline:
    left()
    right()
"""

# Above ext4's limit of 65,000 hard links per file.
_LINK_LIMIT_PROBE = 70_000


def _helper_fill_hard_links(target: pathlib.Path, scratch: pathlib.Path) -> None:
    """Hard-link ``target`` from ``scratch`` until the filesystem refuses another link with
    EMLINK; skip the test on a filesystem that allows more than _LINK_LIMIT_PROBE links."""
    scratch.mkdir()
    for i in range(_LINK_LIMIT_PROBE):
        try:
            os.link(target, scratch / str(i))
        except OSError as e:
            if e.errno != errno.EMLINK:
                raise
            return
    pytest.skip(f"{target}'s filesystem allows more than {_LINK_LIMIT_PROBE} hard links")


def test_outputs_whose_cache_object_is_at_the_hard_link_limit_are_copied(
    tmp_path: pathlib.Path,
) -> None:
    """Two empty outputs share one cache object, which already has as many hard links as the
    filesystem allows. Checkout, a repro that restores them and a repro that rewrites them each
    leave both as regular files, never symlinks into the cache, and both stages stay cached."""
    (tmp_path / "project").mkdir()
    root = _project(tmp_path / "project", {"pipeline.py": _LINK_LIMIT_PIPELINE})
    _ok(_pivot(root, "repro"))
    outputs = [root / "data" / "lim" / "left.txt", root / "data" / "lim" / "right.txt"]
    empty_object = cache.get_cache_path(
        root / ".pivot" / "cache" / "files", xxhash.xxh64(b"").hexdigest()
    )
    assert {o.stat().st_ino for o in outputs} == {empty_object.stat().st_ino}
    for output in outputs:
        output.unlink()
    _helper_fill_hard_links(empty_object, tmp_path / "links")

    restored = _ok(_pivot(root, "checkout"))
    assert "Restored 2 file(s)" in restored, restored
    assert [stat.S_ISREG(os.lstat(o).st_mode) for o in outputs] == [True, True], "checkout"

    for output in outputs:
        output.unlink()
    out = _ok(_pivot(root, "repro"))
    assert "0 ran, 2 cached" in out, out
    assert [stat.S_ISREG(os.lstat(o).st_mode) for o in outputs] == [True, True], "restore"

    out = _ok(_pivot(root, "repro", "--force"))
    assert "2 ran" in out, out
    assert [stat.S_ISREG(os.lstat(o).st_mode) for o in outputs] == [True, True], "rewrite"
    assert [o.read_text() for o in outputs] == ["", ""]

    assert "2 cached" in _ok(_pivot(root, "status"))
    _ok(_pivot(root, "verify"))


_CPU_PIPELINE = """\
import pandas as pd

from pivot import Pipeline, stage


@stage(resources={"cpu": 2})
def scale(raw: pd.DataFrame) -> pd.DataFrame:
    return raw * 2


@stage
def total(scaled: pd.DataFrame) -> dict:
    return {"sum": int(scaled["a"].sum())}


with Pipeline("res") as pipeline:
    total(scale(pipeline.input("t.csv")))
"""


def test_cpu_request_runs_when_fewer_stages_than_it_requests_are_scheduled(
    tmp_path: pathlib.Path,
) -> None:
    """The cpu capacity is the worker count, not the number of stages in this run."""
    root = _project(tmp_path, {"pipeline.py": _CPU_PIPELINE, "data/raw/t.csv": "a\n1\n2\n"})
    _ok(_pivot(root, "config", "set", "core.max_workers", "4"))

    repro = _ok(_pivot(root, "repro", "res/scale", "--force"))
    run = _ok(_pivot(root, "run", "res/scale", "--force"))

    assert "1 ran" in repro, repro
    assert "1 ran" in run, run


def test_resources_cpu_option_sets_the_cpu_capacity(tmp_path: pathlib.Path) -> None:
    """`--resources cpu=N` is the cpu capacity for the run; it is not replaced by the worker count."""
    root = _project(tmp_path, {"pipeline.py": _CPU_PIPELINE, "data/raw/t.csv": "a\n1\n2\n"})
    _ok(_pivot(root, "config", "set", "core.max_workers", "1"))

    refused = _pivot(root, "repro", "res/scale")
    out = _ok(_pivot(root, "repro", "res/scale", "--resources", "cpu=2"))

    assert refused.returncode == 1, refused.stdout + refused.stderr
    assert "requests 2 units of resource 'cpu', but its capacity is 1" in (
        refused.stdout + refused.stderr
    )
    assert "1 ran" in out, out


def test_status_explain_json_writes_dep_identity_as_an_object(tmp_path: pathlib.Path) -> None:
    """`dep_changes[].identity` is `{producer, key}`, the same shape the RPC sends."""
    root = _project(tmp_path, {"pipeline.py": _CHAIN_PIPELINE, "data/raw/t.csv": "a\n1\n2\n"})
    _ok(_pivot(root, "repro"))
    (root / "data" / "raw" / "t.csv").write_text("a\n5\n")

    out = _ok(_pivot(root, "--quiet", "status", "--explain", "--json"))

    stages = {stage["name"]: stage for stage in json.loads(out)["stages"]}
    assert stages["k/scale"]["dep_changes"][0]["identity"] == {"producer": "t.csv", "key": None}


def test_list_refuses_a_fractional_resource_request(tmp_path: pathlib.Path) -> None:
    """The decorator refuses what the scheduler cannot run, so the error comes at load."""
    pipeline = _CPU_PIPELINE.replace('"cpu": 2', '"gpu": 0.5')
    root = _project(tmp_path, {"pipeline.py": pipeline, "data/raw/t.csv": "a\n1\n2\n"})

    proc = _pivot(root, "list")
    out = proc.stdout + proc.stderr

    assert proc.returncode == 1, out
    assert "Stage resource 'gpu' must request a positive integer or 'all'" in out, out


def test_repro_reruns_for_distinct_equal_union_params(tmp_path: pathlib.Path) -> None:
    """A bool override must not reuse an int run just because Python equates their values."""
    pipeline = """\
from pivot import Pipeline, StageParams, stage


class Params(StageParams):
    value: int | bool = 1


@stage
def show(params: Params) -> str:
    return type(params.value).__name__ + ":" + str(params.value)


with Pipeline("p") as pipeline:
    show(params=Params())
"""
    root = _project(tmp_path, {"pipeline.py": pipeline})
    _ok(_pivot(root, "repro"))
    (root / "params.yaml").write_text("p/show:\n  value: true\n")

    explanation = _ok(_pivot(root, "status", "--explain"))
    repro = _ok(_pivot(root, "repro"))

    assert "Param Changes:" in explanation, explanation
    assert "value" in explanation, explanation
    assert "1 ran" in repro, repro
    assert (root / "data/p/show.txt").read_text() == "bool:True"


def test_cached_run_uses_lock_committed_by_concurrent_incremental_run(
    tmp_path: pathlib.Path,
) -> None:
    """A queued restore must use the lock written by the run that held the artifact lock."""
    gate = tmp_path / "entered"
    pipeline = f"""\
import pathlib
import time
from typing import Annotated

from pivot import Pipeline, StageParams, incremental, stage


class Params(StageParams):
    gate: str = {str(gate)!r}


@stage
def grow(previous: str, params: Params) -> Annotated[str, incremental]:
    pathlib.Path(params.gate).write_text("entered")
    time.sleep(8)
    return previous + "a"


with Pipeline("p") as pipeline:
    grow(pipeline.previous(), params=Params())
"""
    root = _project(tmp_path, {"pipeline.py": pipeline})
    _ok(_pivot(root, "repro"))
    gate.unlink()

    first = subprocess.Popen(
        [_PIVOT, "repro", "--json", "--force"],
        cwd=root,
        env=_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 30
        while not gate.exists():
            assert first.poll() is None and time.monotonic() < deadline
            time.sleep(0.01)

        second = subprocess.Popen(
            [_PIVOT, "repro", "--json"],
            cwd=root,
            env=_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        first_stdout, first_stderr = first.communicate(timeout=60)
        second_stdout, second_stderr = second.communicate(timeout=60)
    finally:
        if first.poll() is None:
            first.kill()
            first.wait()

    assert first.returncode == 0, first_stdout + first_stderr
    assert second.returncode == 0, second_stdout + second_stderr
    assert (root / "data/p/grow.txt").read_text() == "aa"


def test_status_explain_displays_falsy_new_param_values(tmp_path: pathlib.Path) -> None:
    """A changed parameter value of zero is displayed instead of an unavailable marker."""
    pipeline = """\
from pivot import Pipeline, StageParams, stage


class Params(StageParams):
    factor: int = 10


@stage
def scale(params: Params) -> str:
    return str(params.factor)


with Pipeline("p") as pipeline:
    scale(params=Params())
"""
    root = _project(tmp_path, {"pipeline.py": pipeline})
    _ok(_pivot(root, "repro"))
    (root / "params.yaml").write_text("p/scale:\n  factor: 0\n")

    explanation = _ok(_pivot(root, "status", "--explain"))

    assert "Old: 10" in explanation, explanation
    assert "New: 0" in explanation, explanation


def test_doctor_reports_invalid_remote_with_one_actionable_line(tmp_path: pathlib.Path) -> None:
    """An invalid remote is a concise doctor diagnostic rather than a Pydantic traceback."""
    root = _project(tmp_path, {})
    (root / ".pivot/config.yaml").write_text("remotes:\n  broken: not-a-remote\n")

    result = _pivot(root, "doctor", "--remote")
    output = result.stdout + result.stderr

    assert result.returncode == 1, output
    assert output.strip() == (
        "Error: Invalid config remotes.broken: 'not-a-remote'; "
        + "Remote 'broken': Invalid remote URL: not-a-remote. "
        + "Expected format: s3://bucket/prefix or gs://bucket/prefix"
    )


def test_doctor_identifies_invalid_remote_after_valid_remote(tmp_path: pathlib.Path) -> None:
    """Doctor names the invalid remote rather than the mapping's first entry."""
    root = _project(tmp_path, {})
    (root / ".pivot/config.yaml").write_text(
        "remotes:\n  good: s3://valid-bucket/prefix\n  broken: not-a-remote\n"
    )

    result = _pivot(root, "doctor", "--remote")
    output = result.stdout + result.stderr

    assert result.returncode == 1, output
    assert output.strip() == (
        "Error: Invalid config remotes.broken: 'not-a-remote'; "
        + "Remote 'broken': Invalid remote URL: not-a-remote. "
        + "Expected format: s3://bucket/prefix or gs://bucket/prefix"
    )


def test_doctor_reports_remote_name_validation_rule(tmp_path: pathlib.Path) -> None:
    """Doctor preserves the remote-name rule for an invalid remote name."""
    root = _project(tmp_path, {})
    (root / ".pivot/config.yaml").write_text("remotes:\n  'bad name': s3://valid-bucket/prefix\n")

    result = _pivot(root, "doctor", "--remote")
    output = result.stdout + result.stderr

    assert result.returncode == 1, output
    assert output.strip() == (
        "Error: Invalid config remotes.bad name: 's3://valid-bucket/prefix'; "
        + "Invalid remote name 'bad name': must be alphanumeric, hyphens, or underscores"
    )


_DIRECTORY_INPUT_PIPELINE = """\
import pathlib

from pivot import Pipeline, stage


@stage
def listing(tree: pathlib.Path) -> str:
    files = sorted(p for p in tree.rglob("*") if p.is_file())
    return ";".join(f"{p.relative_to(tree).as_posix()}={p.read_text()}" for p in files)


with Pipeline("d") as pipeline:
    listing(pipeline.input("tree"))
"""


@pytest.mark.parametrize("relpath", ["__pycache__/x.pyc", "build/y"])
def test_editing_any_file_in_directory_input_reruns_consumer(
    tmp_path: pathlib.Path, relpath: str
) -> None:
    """A directory input's hash covers every file, whatever its name."""
    root = _project(
        tmp_path,
        {
            "pipeline.py": _DIRECTORY_INPUT_PIPELINE,
            "data/raw/tree/keep.txt": "keep",
            "data/raw/tree/__pycache__/x.pyc": "v1",
            "data/raw/tree/build/y": "v1",
        },
    )
    output = root / "data" / "d" / "listing.txt"
    _ok(_pivot(root, "repro"))
    assert output.read_text() == "__pycache__/x.pyc=v1;build/y=v1;keep.txt=keep"

    (root / "data" / "raw" / "tree" / relpath).write_text("v2")
    out = _ok(_pivot(root, "repro"))

    assert f"{relpath}=v2" in output.read_text(), out


_DIRECTORY_OUTPUT_PIPELINE = """\
import pathlib
import tempfile

from pivot import Pipeline, StageParams, stage


@stage
def bundle(params: StageParams) -> pathlib.Path:
    out = pathlib.Path(tempfile.mkdtemp()) / "bundle"
    for relpath in ("keep.txt", "__pycache__/x.pyc", "build/y"):
        (out / relpath).parent.mkdir(parents=True, exist_ok=True)
        (out / relpath).write_text(relpath)
    return out


with Pipeline("o") as pipeline:
    bundle(params=StageParams())
"""


def _tree_contents(root: pathlib.Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): path.read_text()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_directory_output_keeps_every_file_through_checkout(tmp_path: pathlib.Path) -> None:
    """A directory output is cached and restored whole, whatever its files are named."""
    root = _project(tmp_path, {"pipeline.py": _DIRECTORY_OUTPUT_PIPELINE})
    bundle = root / "data" / "o" / "bundle.dat"
    expected = {name: name for name in ("keep.txt", "__pycache__/x.pyc", "build/y")}
    _ok(_pivot(root, "repro"))
    assert _tree_contents(bundle) == expected

    if bundle.is_symlink():
        bundle.unlink()
    else:
        shutil.rmtree(bundle)
    out = _ok(_pivot(root, "checkout"))

    assert _tree_contents(bundle) == expected, out


def _helper_cache_stale_bytecode(path: pathlib.Path, old: str, new: str) -> None:
    """Write ``old`` to ``path`` and cache its bytecode checked by mtime, as a plain ``python``
    import or an older pivot does, then write ``new``, of the same size, and give the file its
    previous mtime: CPython's check of that bytecode by size and whole-second mtime passes, as
    for an edit made in the second the previous version was written."""
    assert len(old) == len(new)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(old)
    py_compile.compile(str(path), invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
    before = path.stat()
    path.write_text(new)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))


def _helper_env_with_path(*entries: pathlib.Path) -> dict[str, str]:
    env = _env()
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [*(str(entry) for entry in entries), env.get("PYTHONPATH")])
    )
    return env


_BY_NAME_PIPELINE = """\
from pivot import Pipeline, stage

{import_helper}


@stage
def build() -> str:
    return helper.VALUE


with Pipeline("w") as pipeline:
    build()
"""


@pytest.mark.parametrize("layout", ["outside-root", "src-layout"])
@pytest.mark.parametrize("bytecode", ["read-only-pycache", "never-checked"])
def test_worker_runs_the_edited_code_of_a_module_it_imports_by_name(
    tmp_path: pathlib.Path, layout: str, bytecode: str
) -> None:
    """A worker imports by name a module discovery does not send by value (one outside the
    root, or under it in a ``src`` layout), and runs its edited code, not the mtime-checked
    bytecode cached for the previous version: when ``__pycache__`` is read-only so the bytecode
    cannot be replaced, and when pivot runs under ``--check-hash-based-pycs never``, which the
    workers do not inherit."""
    root = tmp_path / "project"
    root.mkdir()
    if layout == "outside-root":
        import_root = tmp_path / "outside"
        helper = import_root / "edited_helper.py"
        import_helper = "import edited_helper as helper"
    else:
        import_root = root / "src"
        helper = import_root / "edited_pkg" / "helper.py"
        import_helper = "from edited_pkg import helper"
        helper.parent.mkdir(parents=True)
        (helper.parent / "__init__.py").write_text("")
    _project(root, {"pipeline.py": _BY_NAME_PIPELINE.format(import_helper=import_helper)})
    _helper_cache_stale_bytecode(helper, 'VALUE = "alpha"\n', 'VALUE = "bravo"\n')
    pycache = helper.parent / "__pycache__"
    command = [_PIVOT, "repro"]
    if bytecode == "never-checked":
        command = [sys.executable, "--check-hash-based-pycs", "never", *command]
    else:
        pycache.chmod(0o555)
    try:
        proc = subprocess.run(
            command,
            cwd=root,
            env=_helper_env_with_path(import_root),
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    finally:
        pycache.chmod(0o755)
    out = _ok(proc)

    assert (root / "data" / "w" / "build.txt").read_text() == "bravo", out


_LAZY_IMPORT_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def build(raw: str) -> str:
    scope = {}
    exec("import lazy_helper as loaded", scope)

    return raw.strip() + "-" + scope["loaded"].VALUE


with Pipeline("w") as pipeline:
    build(pipeline.input("in.txt"))
"""

_LAZY_PIVOT_TUI_PIPELINE = _LAZY_IMPORT_PIPELINE.replace("lazy_helper", "pivot_tui.helpers")


@pytest.mark.parametrize(
    ("pipeline", "files", "helper", "pythonpath"),
    [
        pytest.param(
            _LAZY_IMPORT_PIPELINE,
            {"outside/lazy_helper.py": 'VALUE = "alpha"\n'},
            "outside/lazy_helper.py",
            ["outside"],
            id="module-outside-the-root",
        ),
        pytest.param(
            _LAZY_PIVOT_TUI_PIPELINE,
            {
                "project/pivot_tui/__init__.py": "",
                "project/pivot_tui/helpers.py": 'VALUE = "alpha"\n',
            },
            "project/pivot_tui/helpers.py",
            [],
            id="project-package-named-pivot_tui",
        ),
    ],
)
def test_worker_runs_the_edited_code_of_a_module_a_stage_body_imports(
    tmp_path: pathlib.Path,
    pipeline: str,
    files: dict[str, str],
    helper: str,
    pythonpath: list[str],
) -> None:
    """A module a stage loads only at run time is imported by the worker alone: one outside the
    root on ``PYTHONPATH``, or a project package named ``pivot_tui``, which the worker imports
    from the project root on its ``sys.path``, not pivot-tui's. The worker treats it as user code
    (the stage warns that its fingerprint does not cover it) and caches bytecode for it checked by
    content, so an edit that keeps the file's size, made in the second the previous version was
    written and cached, is run when the stage next runs."""
    root = tmp_path / "project"
    root.mkdir()
    _project(root, {"pipeline.py": pipeline, "data/raw/in.txt": "x"})
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    env = _helper_env_with_path(*(tmp_path / entry for entry in pythonpath))
    output = root / "data" / "w" / "build.txt"
    first = _ok(_pivot(root, "repro", env=env))
    assert output.read_text() == "x-alpha"
    assert "its fingerprint does not cover" in " ".join(first.split()), first
    before = (tmp_path / helper).stat()
    (tmp_path / helper).write_text('VALUE = "bravo"\n')
    os.utime(tmp_path / helper, ns=(before.st_atime_ns, before.st_mtime_ns))
    (root / "data" / "raw" / "in.txt").write_text("y")

    out = _ok(_pivot(root, "repro", env=env))

    assert output.read_text() == "y-bravo", out


@pytest.mark.parametrize(
    ("module", "files", "pythonpath"),
    [
        pytest.param(
            "lazy_helper",
            {"outside/lazy_helper.py": 'VALUE = "alpha"\n'},
            ["outside"],
            id="module-outside-the-root",
        ),
        pytest.param(
            "pivot_tui.helpers",
            {
                "project/pivot_tui/__init__.py": "",
                "project/pivot_tui/helpers.py": 'VALUE = "alpha"\n',
            },
            [],
            id="project-package-named-pivot_tui",
        ),
    ],
)
def test_static_body_import_of_a_user_module_discovery_did_not_load_is_refused(
    tmp_path: pathlib.Path, module: str, files: dict[str, str], pythonpath: list[str]
) -> None:
    """The same modules imported by a static statement in the stage body are user code that
    discovery never loaded, so the stage is refused at definition and names the module."""
    root = tmp_path / "project"
    root.mkdir()
    pipeline = (
        "from pivot import Pipeline, stage\n\n\n"
        "@stage\ndef build(raw: str) -> str:\n"
        f"    import {module}\n\n"
        f"    return raw.strip() + '-' + {module}.VALUE\n\n\n"
        'with Pipeline("w") as pipeline:\n    build(pipeline.input("in.txt"))\n'
    )
    _project(root, {"pipeline.py": pipeline, "data/raw/in.txt": "x"})
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    env = _helper_env_with_path(*(tmp_path / entry for entry in pythonpath))

    proc = _pivot(root, "repro", env=env)

    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    assert f"imports user module '{module}'" in out, out
    assert not (root / "data" / "w" / "build.txt").exists()


_STAGE_BODY_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def build() -> str:
    return "alpha"


with Pipeline("w") as pipeline:
    build()
"""


def test_repro_runs_a_stage_whose_edit_keeps_the_files_mtime_size_and_inode(
    tmp_path: pathlib.Path,
) -> None:
    """An edit can keep a file's mtime to the nanosecond, its size and its inode (two writes
    inside one kernel timestamp tick, or a tool that restores the mtime). Pivot's caches of
    function hashes and stage manifests check the file's content, so the edited stage runs."""
    root = _project(tmp_path, {"pipeline.py": _STAGE_BODY_PIPELINE})
    output = root / "data" / "w" / "build.txt"
    _ok(_pivot(root, "repro"))
    assert output.read_text() == "alpha"
    pipeline_py = root / "pipeline.py"
    before = pipeline_py.stat()
    pipeline_py.write_text(_STAGE_BODY_PIPELINE.replace("alpha", "bravo"))
    os.utime(pipeline_py, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = pipeline_py.stat()
    assert (after.st_mtime_ns, after.st_size, after.st_ino) == (
        before.st_mtime_ns,
        before.st_size,
        before.st_ino,
    )

    out = _ok(_pivot(root, "repro"))

    assert output.read_text() == "bravo", out


_ONE_FILE_PIPELINE = """\
from typing import Annotated

import pandas as pd

from pivot import Pipeline, directory, stage


@stage
def split(raw: pd.DataFrame) -> Annotated[dict[str, pd.DataFrame], directory]:
    return {"train.jsonl": raw.iloc[:2], "test.jsonl": raw.iloc[2:]}


@stage
def count(train: pd.DataFrame) -> dict:
    return {"rows": int(len(train)), "total": int(train["a"].sum())}


with Pipeline("s") as pipeline:
    count(split(pipeline.input("t.csv"))["train.jsonl"])
"""


def test_one_file_consumer_reruns_only_when_its_own_file_changes(tmp_path: pathlib.Path) -> None:
    """A stage that reads one file of a directory output skips when only a sibling changes."""
    root = _project(tmp_path, {"pipeline.py": _ONE_FILE_PIPELINE, "data/raw/t.csv": "a\n1\n2\n3\n"})
    raw = root / "data" / "raw" / "t.csv"
    counted = root / "data" / "s" / "count.yaml"
    _ok(_pivot(root, "repro"))
    assert yaml.safe_load(counted.read_text()) == {"rows": 2, "total": 3}

    raw.write_text("a\n1\n2\n9\n")
    sibling_changed = _ok(_pivot(root, "repro"))
    raw.write_text("a\n5\n2\n9\n")
    own_changed = _ok(_pivot(root, "repro"))

    assert "1 ran, 1 cached" in sibling_changed, sibling_changed
    assert "2 ran, 0 cached" in own_changed, own_changed
    assert yaml.safe_load(counted.read_text()) == {"rows": 2, "total": 7}


def test_status_takes_a_one_file_dependency_from_the_producers_lock_when_it_is_deleted(
    tmp_path: pathlib.Path,
) -> None:
    """With the directory deleted, status reads the file's hash from the producer's lock."""
    root = _project(tmp_path, {"pipeline.py": _ONE_FILE_PIPELINE, "data/raw/t.csv": "a\n1\n2\n3\n"})
    _ok(_pivot(root, "repro"))
    shutil.rmtree(root / "data" / "s" / "split")

    status = _ok(_pivot(root, "status"))
    restored = _ok(_pivot(root, "repro"))

    assert "2 cached, 0 stale" in status, status
    assert "0 ran, 2 cached" in restored, restored
    assert (root / "data" / "s" / "split" / "train.jsonl").is_file()
    assert "Verification passed" in _ok(_pivot(root, "verify"))


_RENAMED_FILE_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, StageParams, directory, stage


class SplitParams(StageParams):
    name: str = "a.yaml"


@stage
def split(params: SplitParams) -> Annotated[dict[str, dict], directory]:
    return {params.name: {"name": params.name}}


with Pipeline("r") as pipeline:
    split(params=SplitParams())
"""


def _helper_snapshot(root: pathlib.Path) -> dict[str, bytes]:
    """Every file under the outputs, the cache and the locks, by path.

    The cache's project register (``cache/projects/``) is pivot's own note that this project
    uses the cache, written by any command that resolves it, so it is not output state.
    """
    tops = [root / "data", root / ".pivot" / "cache", root / ".pivot" / "stages"]
    register = root / ".pivot" / "cache" / gc.REGISTER_DIR
    return {
        str(path.relative_to(root)): path.read_bytes()
        for top in tops
        for path in sorted(top.rglob("*"))
        if path.is_file() and register not in path.parents
    }


def test_refused_directory_key_on_a_rerun_keeps_the_last_good_output(
    tmp_path: pathlib.Path,
) -> None:
    """A refused key fails the stage in one line and changes no output, cache file or lock."""
    root = _project(tmp_path, {"pipeline.py": _RENAMED_FILE_PIPELINE})
    _ok(_pivot(root, "repro"))
    (root / "params.yaml").write_text("r/split:\n  name: ../b.yaml\n")
    before = _helper_snapshot(root)
    status_before = _ok(_pivot(root, "status"))

    refused = _pivot(root, "repro")

    out = refused.stdout + refused.stderr
    assert refused.returncode != 0, out
    assert "Stage 'r/split': the return value has key '../b.yaml'" in out, out
    assert "Traceback" not in out, out
    assert _helper_snapshot(root) == before
    assert (root / "data" / "r" / "split" / "a.yaml").is_file()
    assert _ok(_pivot(root, "status")) == status_before


_THREE_FILES_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, directory, stage


@stage
def split() -> Annotated[dict[str, str], directory]:
    return {"a.txt": "a", "b.txt": "b", "sub/c.txt": "c"}


with Pipeline("n") as pipeline:
    split()
"""


def test_checkout_counts_the_files_it_restores(tmp_path: pathlib.Path) -> None:
    """A directory output counts one per file it holds, as a tracked file counts one."""
    root = _project(tmp_path, {"pipeline.py": _THREE_FILES_PIPELINE, "notes.txt": "n"})
    _ok(_pivot(root, "repro"))
    _ok(_pivot(root, "track", "notes.txt"))
    shutil.rmtree(root / "data" / "n" / "split")
    (root / "notes.txt").unlink()

    restored = _ok(_pivot(root, "checkout"))
    skipped = _ok(_pivot(root, "checkout"))

    assert "Restored 4 file(s)" in restored, restored
    assert "Skipped 4 file(s) (already exist)" in skipped, skipped


def _helper_drop_cached_file(root: pathlib.Path, stage: str, relpath: str) -> None:
    """Delete the cache blob of one file of a directory output."""
    members = _helper_output_listing(root, stage)["files"]
    (entry,) = [e for e in members if e["relpath"] == relpath]
    (root / ".pivot" / "cache" / "files" / entry["hash"][:2] / entry["hash"][2:]).unlink()


def test_checkout_counts_the_files_of_a_missing_directory_output(tmp_path: pathlib.Path) -> None:
    """Missing, like restored and skipped, counts files: a three-file directory is three."""
    root = _project(tmp_path, {"pipeline.py": _THREE_FILES_PIPELINE, "notes.txt": "n"})
    _ok(_pivot(root, "repro"))
    _ok(_pivot(root, "track", "notes.txt"))
    shutil.rmtree(root / "data" / "n" / "split")
    for relpath in ("a.txt", "b.txt", "sub/c.txt"):
        _helper_drop_cached_file(root, "n/split", relpath)

    proc = _pivot(root, "checkout")

    out = proc.stdout + proc.stderr
    assert proc.returncode == 1, out
    assert "Skipped 1 file(s) (already exist)" in out, out
    assert "Missing 3 file(s) from:\n  split\n" in out, out


def test_checkout_only_missing_counts_present_files_when_a_blob_is_gone(
    tmp_path: pathlib.Path,
) -> None:
    """An in-place restore that cannot get one file's blob still counts the files on disk."""
    root = _project(tmp_path, {"pipeline.py": _THREE_FILES_PIPELINE})
    _ok(_pivot(root, "repro"))
    (root / "data" / "n" / "split" / "a.txt").unlink()
    _helper_drop_cached_file(root, "n/split", "a.txt")

    proc = _pivot(root, "checkout", "--only-missing")

    out = proc.stdout + proc.stderr
    assert proc.returncode == 1, out
    assert "Skipped 2 file(s) (already exist)" in out, out
    assert "Missing 1 file(s) from:\n  split\n" in out, out


_ALIAS_PIPELINE = """\
from pivot import Pipeline, stage

from shapes import Shape


@stage
def make(seed: str) -> Shape:
    return {"a.yaml": {"v": 1}}


@stage
def use(payload: Shape) -> dict[str, int]:
    return {"n": len(payload)}


with Pipeline("t") as pipeline:
    use(make(pipeline.input("seed.txt")))
"""

_FILE_SHAPE = "Shape = dict[str, dict[str, int]]\n"

_DIRECTORY_SHAPE = """\
from typing import Annotated

from pivot import directory, loaders

Shape = Annotated[dict[str, dict[str, int]], directory, loaders.YAML()]
"""


@pytest.mark.parametrize(
    ("before", "after", "placed", "kind"),
    [
        pytest.param(_FILE_SHAPE, _DIRECTORY_SHAPE, "data/t/make", "dir", id="file-to-directory"),
        pytest.param(
            _DIRECTORY_SHAPE, _FILE_SHAPE, "data/t/make.yaml", "file", id="directory-to-file"
        ),
    ],
)
def test_switching_directory_through_a_type_alias_reruns_the_producer(
    tmp_path: pathlib.Path, before: str, after: str, placed: str, kind: str
) -> None:
    """The stage's source is unchanged, but its output moves and changes kind."""
    root = _project(
        tmp_path,
        {"pipeline.py": _ALIAS_PIPELINE, "shapes.py": before, "data/raw/seed.txt": "x\n"},
    )
    _ok(_pivot(root, "repro"))
    (root / "shapes.py").write_text(after)

    status = _ok(_pivot(root, "status"))
    rerun = _ok(_pivot(root, "repro"))

    assert "0 stale" not in status, status
    assert "2 ran" in rerun, rerun
    output = root / placed
    assert output.is_dir() if kind == "dir" else output.is_file(), sorted(
        str(p.relative_to(root)) for p in (root / "data" / "t").rglob("*")
    )
    _ok(_pivot(root, "verify"))


def test_verify_fails_when_a_file_sits_at_a_directory_outputs_path(
    tmp_path: pathlib.Path,
) -> None:
    """`pivot commit` records whatever is at the path; verify checks the kind matches the stage."""
    root = _project(
        tmp_path,
        {
            "pipeline.py": _ALIAS_PIPELINE,
            "shapes.py": _DIRECTORY_SHAPE,
            "data/raw/seed.txt": "x\n",
        },
    )
    _ok(_pivot(root, "repro"))
    shutil.rmtree(root / "data" / "t" / "make")
    (root / "data" / "t" / "make").write_text("v: 1\n")
    _ok(_pivot(root, "commit", "t/make"))

    verify = _pivot(root, "verify", "t/make")

    out = verify.stdout + verify.stderr
    assert verify.returncode == 1, out
    assert "data/t/make" in out, out
    assert "directory" in out, out


_MEMBERSHIP_PIPELINE = """\
from typing import Annotated

import pandas as pd

from pivot import Pipeline, directory, stage


@stage
def split(raw: pd.DataFrame) -> Annotated[dict[str, pd.DataFrame], directory]:
    return {"train.jsonl": raw}


@stage
def fit(train: pd.DataFrame) -> dict:
    return {"rows": int(len(train))}


with Pipeline("m") as pipeline:
    parts = split(pipeline.input("t.csv"))
    if "train.jsonl" in parts:
        fit(parts["train.jsonl"])
"""


def test_membership_test_on_a_directory_handle_fails_the_load_at_once(
    tmp_path: pathlib.Path,
) -> None:
    """`in` on a handle must fail pipeline.py's load, not spin: every command loads it first."""
    root = _project(tmp_path, {"pipeline.py": _MEMBERSHIP_PIPELINE, "data/raw/t.csv": "a\n1\n"})

    try:
        proc = subprocess.run(
            [_PIVOT, "list"],
            cwd=root,
            env=_env(),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("pivot list was still loading pipeline.py after 30 s")

    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    assert "is not iterable" in out, out
    assert "parts['train.csv']" in out, out


_DIFF_PIPELINE = """\
import pathlib
import tempfile
from typing import Annotated

import pandas as pd

from pivot import Pipeline, directory, loaders, stage


@stage
def split(raw: pd.DataFrame) -> Annotated[dict[str, pd.DataFrame], directory, loaders.CSV()]:
    return {f"{name}.csv": frame for name, frame in raw.groupby("part")}


@stage
def bundle(raw: pd.DataFrame) -> pathlib.Path:
    out = pathlib.Path(tempfile.mkdtemp()) / "bundle"
    out.mkdir()
    raw.to_csv(out / "all.csv", index=False)
    return out


with Pipeline("d") as pipeline:
    rows = pipeline.input("rows.csv")
    split(rows)
    bundle(rows)
"""


def test_diff_on_a_directory_output_compares_each_of_its_files(tmp_path: pathlib.Path) -> None:
    """A directory output diffs file by file: a changed, a new and a removed file, by stage
    name or by path, and an unchanged file is not listed."""
    root = _project(
        tmp_path,
        {
            "pipeline.py": _DIFF_PIPELINE,
            "data/raw/rows.csv": "part,v\ntrain,1\ntest,2\nkeep,5\n",
        },
    )
    _ok(_pivot(root, "repro"))
    _commit(root, "first run")
    (root / "data" / "raw" / "rows.csv").write_text("part,v\ntrain,1\ntrain,3\nval,4\nkeep,5\n")
    _ok(_pivot(root, "repro"))

    by_stage = _ok(_pivot(root, "diff", "--no-tui", "d/split"))
    by_path = _ok(_pivot(root, "diff", "--no-tui", "data/d/split"))
    path_output = _ok(_pivot(root, "diff", "--no-tui", "d/bundle"))

    assert "data/d/split/train.csv:\n  Rows: 1 -> 2 (+1)" in by_stage, by_stage
    assert "data/d/split/val.csv:\n  Rows: 1 (added)" in by_stage, by_stage
    assert "data/d/split/test.csv:\n  Rows: 1 (removed)" in by_stage, by_stage
    assert "keep.csv" not in by_stage, by_stage
    assert by_path == by_stage
    assert "data/d/bundle.dat/all.csv:\n  Rows: 3 -> 4 (+1)" in path_output, path_output


def test_diff_json_names_each_entry_change_type(tmp_path: pathlib.Path) -> None:
    """Every `--json` entry carries `change_type`, one of `added`, `removed` and `modified`,
    beside its row counts, and no cache-miss warning is logged."""
    root = _project(
        tmp_path,
        {
            "pipeline.py": _DIFF_PIPELINE,
            "data/raw/rows.csv": "part,v\ntrain,1\ntest,2\nkeep,5\n",
        },
    )
    _ok(_pivot(root, "repro"))
    _commit(root, "first run")
    (root / "data" / "raw" / "rows.csv").write_text("part,v\ntrain,1\ntrain,3\nval,4\nkeep,5\n")
    _ok(_pivot(root, "repro"))

    result = _pivot(root, "diff", "--json", "d/split")

    assert result.returncode == 0, result.stdout + result.stderr
    kinds = {
        e["path"]: (e["change_type"], e["old_rows"], e["new_rows"])
        for e in json.loads(result.stdout)
    }
    assert kinds == {
        "data/d/split/test.csv": ("removed", 1, None),
        "data/d/split/train.csv": ("modified", 1, 2),
        "data/d/split/val.csv": ("added", None, 1),
    }, kinds
    assert "local cache" not in result.stderr, result.stderr


def _fresh_clone(tmp_path: pathlib.Path, pipeline_py: str) -> pathlib.Path:
    """A project committed with its outputs, then cloned: the clone has no `.pivot/cache`."""
    source = tmp_path / "source"
    source.mkdir()
    _project(source, {"pipeline.py": pipeline_py, "data/raw/rows.csv": "v\n1\n2\n"})
    _ok(_pivot(source, "repro"))
    _commit(source, "first run")
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(source), str(clone)], check=True)
    assert not (clone / ".pivot" / "cache").exists()
    return clone


@pytest.mark.parametrize(
    ("moved", "expected", "text"),
    [
        pytest.param(
            False,
            {"data/d/make.csv": ("modified", None, 3)},
            ["data/d/make.csv:\n  Rows: unknown -> 3"],
            id="modified",
        ),
        pytest.param(
            True,
            {
                "data/d/make/all.csv": ("removed", None, None),
                "data/d/make.csv": ("added", None, 3),
            },
            [
                "data/d/make/all.csv:\n  Rows: unknown (removed)",
                "data/d/make.csv:\n  Rows: 3 (added)",
            ],
            id="moved",
        ),
    ],
)
def test_diff_in_a_fresh_clone_reports_a_head_side_missing_from_the_cache(
    tmp_path: pathlib.Path,
    moved: bool,
    expected: dict[str, tuple[str, int | None, int | None]],
    text: list[str],
) -> None:
    """A clone that never pulled has no cache object for git HEAD's side. Each such entry is
    listed at its path with its change type and unknown HEAD counts, stderr names it once with
    the fix, and every other entry is compared, in `--json`, `--no-tui` and the TUI."""
    clone = _fresh_clone(tmp_path, _MAKE_DIRECTORY_PIPELINE if moved else _MAKE_FILE_PIPELINE)
    if moved:
        (clone / "pipeline.py").write_text(_MAKE_FILE_PIPELINE)
    (clone / "data" / "raw" / "rows.csv").write_text("v\n1\n2\n3\n")
    tui_capture = _write_tui_capture_module(clone)
    _ok(_pivot(clone, "repro"))
    (missing,) = [path for path, (kind, _, _) in expected.items() if kind != "added"]
    line = (
        f"{missing}: its git HEAD content is not in the local cache; run pivot fetch to compare it"
    )

    for target in ("d/make", "."):
        result = _pivot(clone, "diff", "--json", target)
        assert result.returncode == 0, (target, result.stdout + result.stderr)
        entries = json.loads(result.stdout)
        kinds = {e["path"]: (e["change_type"], e["old_rows"], e["new_rows"]) for e in entries}
        assert kinds == expected, (target, kinds)
        assert result.stderr.count(line) == 1, (target, result.stderr)

    text_result = _pivot(clone, "diff", "--no-tui", "d/make")
    assert text_result.returncode == 0, text_result.stdout + text_result.stderr
    for block in text:
        assert block in text_result.stdout, text_result.stdout
    assert text_result.stderr.count(line) == 1, text_result.stderr

    tui_result = _pivot(clone, "diff", "d/make")
    assert tui_result.returncode == 0, tui_result.stdout + tui_result.stderr
    tui_kinds = {
        e["path"]: (e["change_type"], e["old_rows"], e["new_rows"])
        for e in json.loads(tui_capture.read_text())
    }
    assert tui_kinds == expected, tui_kinds
    assert tui_result.stderr.count(line) == 1, tui_result.stderr


_FETCH_FIX_PIPELINE = """\
from typing import Annotated

import pandas as pd

from pivot import Pipeline, directory, loaders, stage


@stage
def mod(raw: pd.DataFrame) -> Annotated[pd.DataFrame, loaders.CSV()]:
    return raw


@stage
def split(raw: pd.DataFrame) -> Annotated[dict[str, pd.DataFrame], directory, loaders.CSV()]:
    return {f"{name}.csv": frame for name, frame in raw.groupby("part")}


with Pipeline("d") as pipeline:
    rows = pipeline.input("rows.csv")
    mod(rows)
    split(rows)
"""


@pytest.mark.parametrize(
    ("change", "target", "before_fix", "after_fix"),
    [
        pytest.param(
            "modified",
            "data/d/mod.csv",
            ("modified", None, 3),
            ("modified", 2, 3),
            id="modified-output",
        ),
        pytest.param(
            "deleted",
            "data/d/mod.csv",
            ("removed", None, None),
            ("removed", 2, None),
            id="deleted-output",
        ),
        pytest.param(
            "deleted",
            "data/d/split/train.csv",
            ("removed", None, None),
            ("removed", 1, None),
            id="deleted-directory-member",
        ),
    ],
)
def test_diff_names_a_fix_that_compares_the_entry_without_touching_the_workspace(
    tmp_path: pathlib.Path,
    moto_services: dict[str, str],
    change: str,
    target: str,
    before_fix: tuple[str, int | None, int | None],
    after_fix: tuple[str, int | None, int | None],
) -> None:
    """In a fresh clone whose cache lacks git HEAD's objects, the warning names a command.
    Running it makes the same diff compare the entry with git HEAD's counts known, and leaves
    the workspace side as it was: a deleted output stays deleted, a changed one stays changed."""
    env = _env(
        AWS_ENDPOINT_URL=moto_services["s3"],
        AWS_ACCESS_KEY_ID="x",
        AWS_SECRET_ACCESS_KEY="x",
        AWS_DEFAULT_REGION="us-east-1",
    )
    bucket = f"fetch-fix-{uuid.uuid4().hex[:8]}"
    boto3.client(
        "s3",
        endpoint_url=moto_services["s3"],
        region_name="us-east-1",
        aws_access_key_id="x",
        aws_secret_access_key="x",
    ).create_bucket(Bucket=bucket)  # pyright: ignore[reportAttributeAccessIssue] - boto3 ships no stubs

    source = tmp_path / "source"
    source.mkdir()
    _project(
        source,
        {"pipeline.py": _FETCH_FIX_PIPELINE, "data/raw/rows.csv": "part,v\ntrain,1\ntest,2\n"},
    )
    _ok(_pivot(source, "config", "set", "remotes.origin", f"s3://{bucket}/cache"))
    _ok(_pivot(source, "config", "set", "default_remote", "origin"))
    _ok(_pivot(source, "repro", env=env))
    _ok(_pivot(source, "push", env=env))
    _commit(source, "first run")

    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(source), str(clone)], check=True)
    assert not (clone / ".pivot" / "cache").exists()
    if change == "modified":
        (clone / "data" / "raw" / "rows.csv").write_text("part,v\ntrain,1\ntest,2\ntrain,3\n")
        _ok(_pivot(clone, "repro", env=env))
    else:
        (clone / target).unlink()
    workspace_before = (clone / target).read_bytes() if (clone / target).exists() else None

    def _entry(
        result: subprocess.CompletedProcess[str],
    ) -> dict[str, tuple[str, int | None, int | None]]:
        assert result.returncode == 0, result.stdout + result.stderr
        return {
            e["path"]: (e["change_type"], e["old_rows"], e["new_rows"])
            for e in json.loads(result.stdout)
        }

    first = _pivot(clone, "diff", "--json", target, env=env)
    assert _entry(first) == {target: before_fix}, first.stdout
    line = (
        f"{target}: its git HEAD content is not in the local cache; run pivot fetch to compare it"
    )
    assert first.stderr.count(line) == 1, first.stderr

    _ok(_pivot(clone, "fetch", env=env))

    after = _pivot(clone, "diff", "--json", target, env=env)
    assert _entry(after) == {target: after_fix}, after.stdout
    assert "local cache" not in after.stderr, after.stderr
    workspace_after = (clone / target).read_bytes() if (clone / target).exists() else None
    assert workspace_after == workspace_before


def _helper_commit_output_listing(
    root: pathlib.Path, stage: str, data: bytes, *, size: int, nfiles: int
) -> None:
    """Replace the committed listing of ``stage``'s directory output with ``data``, point its
    lock's record at it (hash, size and file count) and commit both, as a hostile commit would."""
    stages = root / ".pivot" / "stages"
    (stages / f"{stage}=.listing").write_bytes(data)
    lock_path = stages / f"{stage}.lock"
    lock_data = yaml.safe_load(lock_path.read_text())
    lock_data["records"]["outs"][0].update(
        listing=xxhash.xxh64(data).hexdigest(), size=size, nfiles=nfiles
    )
    lock_path.write_text(yaml.safe_dump(lock_data, sort_keys=False))
    _commit(root, "hostile listing")


def test_diff_rejects_listing_path_traversal_before_reading_outside_project(
    tmp_path: pathlib.Path,
) -> None:
    """`pivot diff` refuses a committed directory listing whose path leaves the project, naming
    the listing file and line, and never reads the file that path names."""
    root = _project(
        tmp_path,
        {
            "pipeline.py": _DIFF_PIPELINE,
            "data/raw/rows.csv": "part,v\ntrain,1\n",
        },
    )
    _ok(_pivot(root, "repro"))
    _commit(root, "first run")

    secret = tmp_path.parent / f"{tmp_path.name}-secret.csv"
    secret.write_text("TOP_SECRET_VALUE\n1\n")
    size = secret.stat().st_size
    try:
        line = f"{'a' * 16} {size} - ../../../../{secret.name}\n"
        _helper_commit_output_listing(root, "d/split", line.encode(), size=size, nfiles=1)

        result = _pivot(root, "diff", "--json", "d/split")
    finally:
        secret.unlink()

    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert ".pivot/stages/d/split=.listing" in output, output
    assert "line 1: an empty, '.' or '..' segment" in " ".join(output.split()), output
    assert "TOP_SECRET_VALUE" not in output


def test_diff_refuses_a_listing_path_with_a_null_byte(tmp_path: pathlib.Path) -> None:
    """`pivot diff` names and refuses a committed listing path containing a null byte."""
    root = _project(
        tmp_path,
        {
            "pipeline.py": _DIFF_PIPELINE,
            "data/raw/rows.csv": "part,v\ntrain,1\n",
        },
    )
    _ok(_pivot(root, "repro"))
    _commit(root, "first run")

    line = f'{"a" * 16} 20 - "nul\\000.csv"\n'
    _helper_commit_output_listing(root, "d/split", line.encode(), size=20, nfiles=1)

    result = _pivot(root, "diff", "--json", "d/split")

    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert ".pivot/stages/d/split=.listing" in output, output
    assert "line 1: a NUL byte" in " ".join(output.split()), output


_RECORDED_NAME_DIRECTORY_PIPELINE = """\
from typing import Annotated

import pandas as pd

from pivot import Pipeline, directory, loaders, stage


@stage
def make(raw: pd.DataFrame) -> Annotated[dict[str, pd.DataFrame], directory, loaders.CSV()]:
    return {{"plain.csv": raw, {name!r}: raw}}


with Pipeline("d") as pipeline:
    make(pipeline.input("rows.csv"))
"""

# A `pathlib.Path` directory output writes its files itself, so any name the filesystem takes
# reaches the lock: a CSV through pandas, any other name as raw bytes.
_RECORDED_NAME_PATH_PIPELINE = """\
import pathlib
import tempfile

import pandas as pd

from pivot import Pipeline, stage


@stage
def make(raw: pd.DataFrame) -> pathlib.Path:
    out = pathlib.Path(tempfile.mkdtemp()) / "make"
    out.mkdir()
    raw.to_csv(out / "plain.csv", index=False)
    name = {name!r}
    if name.endswith(".csv"):
        raw.to_csv(out / name, index=False)
    else:
        (out / name).write_bytes(b"image-bytes")
    return out


with Pipeline("d") as pipeline:
    make(pipeline.input("rows.csv"))
"""


def _run_recorded_name_project(
    tmp_path: pathlib.Path, pipeline_py: str, output: str, name: str
) -> tuple[pathlib.Path, str]:
    """Run a stage whose directory output holds ``plain.csv`` and ``name``, commit, grow the
    input by one row and run again. Returns the root and both runs' output."""
    root = _project(
        tmp_path,
        {"pipeline.py": pipeline_py.format(name=name), "data/raw/rows.csv": "v\n1\n2\n"},
    )
    first = _ok(_pivot(root, "repro"))
    lock_data = yaml.safe_load((root / ".pivot" / "stages" / "d" / "make.lock").read_text())
    assert lock_data["records"]["outs"][0]["path"] == output
    members = _helper_output_listing(root, "d/make")
    assert sorted(entry["relpath"] for entry in members["files"]) == sorted(["plain.csv", name])
    status = _ok(_pivot(root, "status"))
    assert "1 stages: 1 cached, 0 stale" in status, status
    _commit(root, "first run")
    (root / "data" / "raw" / "rows.csv").write_text("v\n1\n2\n3\n")
    second = _ok(_pivot(root, "repro"))
    return root, first + second


def _assert_recorded_name_is_read(root: pathlib.Path, output: str, name: str) -> None:
    """status, verify, diff --json, params diff and checkout all read the recorded name."""
    status = _ok(_pivot(root, "status"))
    assert "1 stages: 1 cached, 0 stale" in status, status
    assert "Verification passed" in _ok(_pivot(root, "verify"))
    changed = {"plain.csv", name} if name.endswith(".csv") else {"plain.csv"}
    assert {
        (entry["path"], entry["change_type"], entry["old_rows"], entry["new_rows"])
        for entry in _json_diff(root, "d/make")
    } == {(f"{output}/{member}", "modified", 2, 3) for member in changed}
    params = _pivot(root, "params", "diff")
    assert params.returncode == 0, params.stdout + params.stderr
    shutil.rmtree(root / output)
    assert "Restored 2 file(s)" in _ok(_pivot(root, "checkout"))
    assert sorted(path.name for path in (root / output).iterdir()) == sorted(["plain.csv", name])


@pytest.mark.parametrize(
    ("pipeline_py", "output", "name"),
    [
        pytest.param(
            _RECORDED_NAME_DIRECTORY_PIPELINE,
            "data/d/make",
            "Screenshot 2024-01-01 at 10.00.00\u202fAM.csv",
            id="directory-narrow-no-break-space",
        ),
        pytest.param(
            _RECORDED_NAME_DIRECTORY_PIPELINE,
            "data/d/make",
            "caf\u00e9\u00a0menu.csv",
            id="directory-no-break-space",
        ),
        pytest.param(
            _RECORDED_NAME_DIRECTORY_PIPELINE,
            "data/d/make",
            "team\u200dlead.csv",
            id="directory-zero-width-joiner",
        ),
        pytest.param(
            _RECORDED_NAME_PATH_PIPELINE,
            "data/d/make.dat",
            "Screenshot 2024-01-01 at 10.00.00\u202fAM.csv",
            id="path-narrow-no-break-space",
        ),
        pytest.param(
            _RECORDED_NAME_PATH_PIPELINE,
            "data/d/make.dat",
            "Screenshot 2024-01-01 at 10.00.00\u202fAM.png",
            id="path-macos-screenshot-png",
        ),
        pytest.param(
            _RECORDED_NAME_PATH_PIPELINE,
            "data/d/make.dat",
            "\u8cc7\u6599\u3000\u4e00\u89a7.png",
            id="path-ideographic-space-png",
        ),
    ],
)
def test_a_member_name_with_a_space_or_format_character_is_recorded_and_read(
    tmp_path: pathlib.Path, pipeline_py: str, output: str, name: str
) -> None:
    """A name with a no-break, narrow or ideographic space or a zero-width joiner, as in a
    macOS screenshot, is recorded without a warning, and every command reads the lock."""
    root, runs = _run_recorded_name_project(tmp_path, pipeline_py, output, name)

    assert "WARNING" not in runs, runs
    _assert_recorded_name_is_read(root, output, name)


def test_a_member_name_with_a_control_character_is_recorded_with_a_warning(
    tmp_path: pathlib.Path,
) -> None:
    """A TAB in a member name is recorded and read back, as on 0.5.2; each run that records it
    warns once, naming the file, that a later release will refuse it."""
    name = "a\tb.csv"
    root, runs = _run_recorded_name_project(
        tmp_path, _RECORDED_NAME_DIRECTORY_PIPELINE, "data/d/make", name
    )

    warning = (
        "Stage 'd/make' records 'data/d/make/a\\tb.csv', whose name holds a control character; "
        + "a later release will refuse such names, so rename it"
    )
    assert " ".join(runs.split()).count(warning) == 2, runs
    _assert_recorded_name_is_read(root, "data/d/make", name)


def test_a_member_name_with_a_backslash_fails_the_stage_and_records_nothing(
    tmp_path: pathlib.Path,
) -> None:
    """A lock never records a name its read refuses: a backslash fails the run and `pivot
    commit`, naming the file, and writes no lock or listing."""
    root = _project(
        tmp_path,
        {
            "pipeline.py": _RECORDED_NAME_PATH_PIPELINE.format(name="b\\s.csv"),
            "data/raw/rows.csv": "v\n1\n2\n",
        },
    )
    lock_path = root / ".pivot" / "stages" / "d" / "make.lock"
    listing_path = root / ".pivot" / "stages" / "d" / "make=.listing"
    refusal = (
        "Stage 'd/make' cannot record 'data/d/make.dat/b\\\\s.csv': it contains a backslash; "
        + "rename it"
    )

    run = _pivot(root, "repro")
    assert run.returncode == 1, run.stdout + run.stderr
    assert refusal in " ".join((run.stdout + run.stderr).split()), run.stdout + run.stderr
    assert not lock_path.exists()
    assert not listing_path.exists()

    _ok(_pivot(root, "repro", "--no-commit"))
    commit = _pivot(root, "commit", "d/make")
    assert commit.returncode == 1, commit.stdout + commit.stderr
    assert f"{refusal} — skipping" in " ".join(commit.stderr.split()), commit.stderr
    assert not lock_path.exists()
    assert not listing_path.exists()


_EXTERNAL_CACHE_DIFF_PIPELINE = """\
import pandas as pd

from pivot import Pipeline, stage


@stage
def double(numbers: pd.DataFrame) -> pd.DataFrame:
    return numbers * 2


with Pipeline("p") as pipeline:
    double(pipeline.input("numbers.csv"))
"""


def _configure_external_symlink_cache(root: pathlib.Path, cache_dir: pathlib.Path) -> None:
    _ok(_pivot(root, "config", "set", "cache.dir", str(cache_dir)))
    _ok(_pivot(root, "config", "set", "cache.checkout_mode", "symlink"))


def test_diff_supports_external_cache_symlink_checkout_for_a_single_file_output(
    tmp_path: pathlib.Path,
) -> None:
    """`pivot diff` reads a single-file output checked out from an external cache."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    root = _project(
        project_dir,
        {
            "pipeline.py": _EXTERNAL_CACHE_DIFF_PIPELINE,
            "data/raw/numbers.csv": "a\n1\n",
        },
    )
    _configure_external_symlink_cache(root, tmp_path / "cache")
    _ok(_pivot(root, "repro"))
    _commit(root, "first run")
    (root / "data" / "raw" / "numbers.csv").write_text("a\n1\n2\n")
    _ok(_pivot(root, "repro"))

    json_output = _ok(_pivot(root, "diff", "--json", "data/p/double.jsonl"))
    text_output = _ok(_pivot(root, "diff", "--no-tui", "data/p/double.jsonl"))

    assert '"path": "data/p/double.jsonl"' in json_output
    assert "Rows: 1 -> 2 (+1)" in text_output


def test_diff_supports_external_cache_symlink_checkout_for_a_directory_output(
    tmp_path: pathlib.Path,
) -> None:
    """`pivot diff` reads a directory output checked out from an external cache."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    root = _project(
        project_dir,
        {
            "pipeline.py": _DIFF_PIPELINE,
            "data/raw/rows.csv": "part,v\ntrain,1\n",
        },
    )
    _configure_external_symlink_cache(root, tmp_path / "cache")
    _ok(_pivot(root, "repro"))
    _commit(root, "first run")
    (root / "data" / "raw" / "rows.csv").write_text("part,v\ntrain,1\ntrain,2\n")
    _ok(_pivot(root, "repro"))

    json_output = _ok(_pivot(root, "diff", "--json", "d/split"))
    text_output = _ok(_pivot(root, "diff", "--no-tui", "d/split"))

    assert '"path": "data/d/split/train.csv"' in json_output
    assert "Rows: 1 -> 2 (+1)" in text_output


_NESTED_DIFF_PIPELINE = """\
from typing import Annotated

import pandas as pd

from pivot import Pipeline, directory, loaders, stage


@stage
def make(raw: pd.DataFrame) -> Annotated[dict[str, pd.DataFrame], directory, loaders.CSV()]:
    return {"nested/train.csv": raw}


with Pipeline("d") as pipeline:
    make(pipeline.input("rows.csv"))
"""

_TUI_CAPTURE_MODULE = """\
import json
import pathlib


def run_diff_app(results):
    pathlib.Path("tui-results.json").write_text(json.dumps(results, default=str))
"""


def _write_tui_capture_module(root: pathlib.Path) -> pathlib.Path:
    package = root / "pivot_tui"
    package.mkdir()
    (package / "__init__.py").write_text("")
    capture = package / "diff.py"
    capture.write_text(_TUI_CAPTURE_MODULE)
    return root / "tui-results.json"


def _replace_nested_with_outside_directory(
    root: pathlib.Path, tmp_path: pathlib.Path
) -> pathlib.Path:
    """Replace the recorded `data/d/make/nested` with a symlink to an outside secret directory."""
    nested = root / "data" / "d" / "make" / "nested"
    (nested / "train.csv").unlink()
    nested.rmdir()
    external = tmp_path.parent / f"{tmp_path.name}-external"
    external.mkdir()
    (external / "train.csv").write_text("TOP_SECRET_VALUE\n42\n")
    nested.symlink_to(external, target_is_directory=True)
    return external


@pytest.mark.parametrize(
    "target",
    [
        pytest.param("d/make", id="directory-target"),
        pytest.param("data/d/make/nested/train.csv", id="direct-member-target"),
    ],
)
def test_diff_member_map_excludes_descendants_of_a_left_out_directory_symlink(
    tmp_path: pathlib.Path, target: str
) -> None:
    """`pivot diff` never reads a recorded file below a left-out directory symlink."""
    root = _project(
        tmp_path,
        {
            "pipeline.py": _NESTED_DIFF_PIPELINE,
            "data/raw/rows.csv": "value\n1\n",
        },
    )
    tui_capture = _write_tui_capture_module(root)
    _ok(_pivot(root, "repro"))
    _commit(root, "first run")

    external = _replace_nested_with_outside_directory(root, tmp_path)
    try:
        json_output = _ok(_pivot(root, "diff", "--json", target))
        text_output = _ok(_pivot(root, "diff", "--no-tui", target))
        tui_output = _ok(_pivot(root, "diff", target))
        tui_results = tui_capture.read_text()
    finally:
        (external / "train.csv").unlink()
        external.rmdir()

    for output in (json_output, text_output, tui_output):
        assert "Directory artifact data/d/make leaves out symlink nested" in output, output
        assert "TOP_SECRET_VALUE" not in output, output
    assert "data/d/make/nested/train.csv" in tui_results
    assert "TOP_SECRET_VALUE" not in tui_results


@pytest.mark.parametrize(
    "target",
    [
        pytest.param("d/split", id="directory-target"),
        pytest.param("data/d/split/train.csv", id="direct-member-target"),
    ],
)
def test_diff_tui_excludes_an_outside_file_symlink(tmp_path: pathlib.Path, target: str) -> None:
    """The TUI receives no data from a left-out file symlink."""
    root = _project(
        tmp_path,
        {
            "pipeline.py": _DIFF_PIPELINE,
            "data/raw/rows.csv": "part,v\ntrain,1\n",
        },
    )
    tui_capture = _write_tui_capture_module(root)
    _ok(_pivot(root, "repro"))
    _commit(root, "first run")

    secret = tmp_path.parent / f"{tmp_path.name}-secret.csv"
    secret.write_text("TOP_SECRET_VALUE\n1\n")
    member = root / "data" / "d" / "split" / "train.csv"
    member.unlink()
    member.symlink_to(secret)
    try:
        output = _ok(_pivot(root, "diff", target))
        tui_results = tui_capture.read_text()
    finally:
        secret.unlink()

    assert "Directory artifact data/d/split leaves out symlink train.csv" in output
    assert "TOP_SECRET_VALUE" not in output
    assert "data/d/split/train.csv" in tui_results
    assert "TOP_SECRET_VALUE" not in tui_results


def test_diff_excludes_a_cache_shaped_symlink_with_wrong_contents(tmp_path: pathlib.Path) -> None:
    """A symlink to a misnamed cache object is left out like record-time hashing leaves it out."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    root = _project(
        project_dir,
        {
            "pipeline.py": _DIFF_PIPELINE,
            "data/raw/rows.csv": "part,v\ntrain,1\n",
        },
    )
    cache_dir = tmp_path / "cache"
    _ok(_pivot(root, "config", "set", "cache.dir", str(cache_dir)))
    _ok(_pivot(root, "repro"))
    _commit(root, "first run")

    member = root / "data" / "d" / "split" / "train.csv"
    fake_hash = "0123456789abcdef"
    fake_cache_object = cache_dir / "files" / fake_hash[:2] / fake_hash[2:]
    fake_cache_object.parent.mkdir(parents=True)
    fake_cache_object.write_text("TOP_SECRET_VALUE\n42\n")
    member.unlink()
    member.symlink_to(fake_cache_object)

    output = _ok(_pivot(root, "diff", "--json", "d/split"))

    assert "Directory artifact data/d/split leaves out symlink train.csv" in output
    assert "TOP_SECRET_VALUE" not in output


def _row_counts(root: pathlib.Path, target: str) -> dict[str, tuple[object, object]]:
    result = _pivot(root, "diff", "--json", target)
    assert result.returncode == 0, result.stdout + result.stderr
    return {
        entry["path"]: (entry["old_rows"], entry["new_rows"]) for entry in json.loads(result.stdout)
    }


def test_diff_compares_a_cache_object_symlink_member(tmp_path: pathlib.Path) -> None:
    """A directory member that is a checkout link to its cache object diffs as that object."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    root = _project(
        project_dir,
        {
            "pipeline.py": _DIFF_PIPELINE,
            "data/raw/rows.csv": "part,v\ntrain,1\n",
        },
    )
    cache_dir = tmp_path / "cache"
    _ok(_pivot(root, "config", "set", "cache.dir", str(cache_dir)))
    _ok(_pivot(root, "repro"))
    _commit(root, "first run")
    (root / "data" / "raw" / "rows.csv").write_text("part,v\ntrain,1\ntrain,2\n")
    _ok(_pivot(root, "repro"))

    member = root / "data" / "d" / "split" / "train.csv"
    member_hash, _ = cache.hash_file(member)
    cache_object = cache.get_cache_path(cache_dir / "files", member_hash)
    assert cache_object.is_file()
    member.unlink()
    member.symlink_to(cache_object)

    direct = _row_counts(root, "data/d/split/train.csv")
    whole = _row_counts(root, "d/split")

    assert direct == {"data/d/split/train.csv": (1, 2)}
    assert whole["data/d/split/train.csv"] == (1, 2)


def _output_root_symlinked_to_an_in_project_copy(root: pathlib.Path) -> None:
    output = root / "data" / "d" / "split"
    replacement = root / "data" / "d" / "replacement"
    shutil.copytree(output, replacement)
    shutil.rmtree(output)
    output.symlink_to(replacement, target_is_directory=True)


@pytest.mark.parametrize(
    "layout", ["external-cache-symlink-checkout", "output-root-symlink", "data-symlinked-outside"]
)
def test_diff_gives_a_member_one_answer_whatever_the_target(
    tmp_path: pathlib.Path, layout: str
) -> None:
    """A directory member diffs the same whether the target names it, its output, or a
    directory above the output, including the project root, in each supported symlinked layout."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    root = _project(project_dir, {"pipeline.py": _DIFF_PIPELINE})
    rows = root / "data" / "raw" / "rows.csv"
    if layout == "data-symlinked-outside":
        external_data = tmp_path / "data"
        external_data.mkdir()
        (root / "data").symlink_to(external_data, target_is_directory=True)
    if layout == "external-cache-symlink-checkout":
        _configure_external_symlink_cache(root, tmp_path / "cache")
    rows.parent.mkdir(parents=True, exist_ok=True)
    rows.write_text("part,v\ntrain,1\n")
    _ok(_pivot(root, "repro"))
    _commit(root, "first run")
    rows.write_text("part,v\ntrain,1\ntrain,2\n")
    _ok(_pivot(root, "repro"))
    if layout == "output-root-symlink":
        _output_root_symlinked_to_an_in_project_copy(root)
    if layout != "data-symlinked-outside":
        assert (root / "data" / "d" / "split").is_symlink()

    answers = {
        target: _row_counts(root, target).get("data/d/split/train.csv")
        for target in (".", "data", "data/d", "data/d/split", "data/d/split/train.csv", "d/split")
    }

    assert answers == dict.fromkeys(answers, (1, 2))


def test_diff_path_directory_member_never_reads_an_outside_symlink(
    tmp_path: pathlib.Path,
) -> None:
    """A direct member of a `pathlib.Path` directory output uses that output's member map."""
    root = _project(
        tmp_path,
        {
            "pipeline.py": _DIFF_PIPELINE,
            "data/raw/rows.csv": "part,v\ntrain,1\n",
        },
    )
    tui_capture = _write_tui_capture_module(root)
    _ok(_pivot(root, "repro"))
    _commit(root, "first run")

    secret = tmp_path.parent / f"{tmp_path.name}-secret.csv"
    secret.write_text("part,v\nTOP_SECRET_VALUE,42\n")
    member = root / "data" / "d" / "bundle.dat" / "all.csv"
    member.unlink()
    member.symlink_to(secret)
    target = "data/d/bundle.dat/all.csv"
    try:
        json_output = _ok(_pivot(root, "diff", "--json", target))
        text_output = _ok(_pivot(root, "diff", "--no-tui", target))
        tui_output = _ok(_pivot(root, "diff", target))
        tui_results = tui_capture.read_text()
    finally:
        secret.unlink()

    for output in (json_output, text_output, tui_output):
        assert "Directory artifact data/d/bundle.dat leaves out symlink all.csv" in output, output
        assert "TOP_SECRET_VALUE" not in output, output
    assert target in tui_results
    assert "TOP_SECRET_VALUE" not in tui_results


def _json_diff(root: pathlib.Path, target: str) -> list[dict[str, object]]:
    result = _pivot(root, "diff", "--json", target)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def test_diff_of_the_project_root_equals_the_data_directory(tmp_path: pathlib.Path) -> None:
    """`pivot diff .` and the absolute project root select the same outputs as `data`."""
    root = _project(
        tmp_path,
        {
            "pipeline.py": _DIFF_PIPELINE,
            "data/raw/rows.csv": "part,v\ntrain,1\ntest,2\n",
        },
    )
    _ok(_pivot(root, "repro"))
    _commit(root, "first run")
    targets = (".", str(root), "data")

    unchanged = {target: _json_diff(root, target) for target in targets}

    assert unchanged == {target: [] for target in targets}

    (root / "data" / "raw" / "rows.csv").write_text("part,v\ntrain,1\ntrain,3\ntest,2\n")
    _ok(_pivot(root, "repro"))
    changed = {target: _json_diff(root, target) for target in targets}

    assert changed["."] == changed[str(root)] == changed["data"]
    assert {entry["path"] for entry in changed["data"]} == {
        "data/d/bundle.dat/all.csv",
        "data/d/split/train.csv",
    }


def test_diff_warns_only_about_symlinks_inside_the_selected_targets(
    tmp_path: pathlib.Path,
) -> None:
    """A left-out sibling of a direct target is not named; the whole output names it."""
    root = _project(
        tmp_path,
        {
            "pipeline.py": _DIFF_PIPELINE,
            "data/raw/rows.csv": "part,v\ntrain,1\ntest,2\n",
        },
    )
    _ok(_pivot(root, "repro"))
    _commit(root, "first run")

    secret = tmp_path.parent / f"{tmp_path.name}-secret.csv"
    secret.write_text("part,v\nTOP_SECRET_VALUE,42\n")
    sibling = root / "data" / "d" / "split" / "test.csv"
    sibling.unlink()
    sibling.symlink_to(secret)
    try:
        direct = _ok(_pivot(root, "diff", "--json", "data/d/split/train.csv"))
        whole = _ok(_pivot(root, "diff", "--json", "d/split"))
    finally:
        secret.unlink()

    assert "leaves out" not in direct, direct
    assert "Directory artifact data/d/split leaves out symlink test.csv" in whole, whole
    assert "TOP_SECRET_VALUE" not in direct + whole


def _symlinked_project(
    tmp_path: pathlib.Path, files: dict[str, str]
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """A project at ``<tmp>/project``, a symlink ``<tmp>/link`` to it, and a CSV outside it."""
    root = tmp_path / "project"
    root.mkdir()
    _project(root, files)
    link = tmp_path / "link"
    link.symlink_to(root, target_is_directory=True)
    outside = tmp_path / "outside.csv"
    outside.write_text("part,v\nx,1\n")
    return root, link, outside


def test_diff_takes_an_absolute_target_typed_through_a_symlink_to_the_project(
    tmp_path: pathlib.Path,
) -> None:
    """From a cwd reached through a symlink to the project, an absolute target under that
    symlink names the output the relative target names; a path outside stays unmatched."""
    root, link, outside = _symlinked_project(
        tmp_path, {"pipeline.py": _DIFF_PIPELINE, "data/raw/rows.csv": "part,v\ntrain,1\ntest,2\n"}
    )
    _ok(_pivot(root, "repro"))
    _commit(root, "first run")
    (root / "data" / "raw" / "rows.csv").write_text("part,v\ntrain,1\ntrain,3\ntest,2\n")
    _ok(_pivot(root, "repro"))

    relative = _pivot(link, "diff", "--json", "data/d/split/train.csv")
    absolute = _pivot(link, "diff", "--json", str(link / "data" / "d" / "split" / "train.csv"))
    refused = _pivot(link, "diff", "--json", str(outside))

    assert relative.returncode == 0, relative.stdout + relative.stderr
    assert absolute.returncode == 0, absolute.stdout + absolute.stderr
    entries = json.loads(absolute.stdout)
    assert entries == json.loads(relative.stdout)
    assert [(e["path"], e["old_rows"], e["new_rows"]) for e in entries] == [
        ("data/d/split/train.csv", 1, 2)
    ]
    assert "outside base" not in absolute.stderr, absolute.stderr
    assert refused.returncode == 0, refused.stdout + refused.stderr
    assert json.loads(refused.stdout) == []
    assert (
        f"No CSV, JSON or JSONL stage output matches {str(outside)!r}, so it is not compared"
        in " ".join(refused.stderr.split())
    ), refused.stderr


def test_checkout_takes_an_absolute_target_typed_through_a_symlink_to_the_project(
    tmp_path: pathlib.Path,
) -> None:
    """An absolute target under a symlink to the project restores what the relative target
    restores; a path outside the project is refused as before."""
    root, link, outside = _symlinked_project(
        tmp_path, {"pipeline.py": _DIFF_PIPELINE, "data/raw/rows.csv": "part,v\ntrain,1\ntest,2\n"}
    )
    _ok(_pivot(root, "repro"))
    output = root / "data" / "d" / "split"

    for target in ("data/d/split", str(link / "data" / "d" / "split")):
        shutil.rmtree(output)
        assert "Restored 2 file(s)" in _ok(_pivot(link, "checkout", target)), target
        assert sorted(path.name for path in output.iterdir()) == ["test.csv", "train.csv"]
    refused = _pivot(link, "checkout", str(outside))

    assert refused.returncode == 1, refused.stdout + refused.stderr
    assert f"Target '{outside}' resolves outside project root" in refused.stderr, refused.stderr


@pytest.mark.parametrize(
    ("command", "line"),
    [
        ("push", "Would push 3 file(s) to 'origin'"),
        ("fetch", "Would fetch 2 file(s) from 'origin'"),
        ("pull", "Would pull 2 file(s) from 'origin'"),
    ],
)
def test_remote_commands_take_an_absolute_target_typed_through_a_symlink_to_the_project(
    tmp_path: pathlib.Path, command: str, line: str
) -> None:
    """push, fetch and pull read an absolute target under a symlink to the project as the
    relative target; a path outside the project is refused as before."""
    root, link, outside = _symlinked_project(
        tmp_path, {"pipeline.py": _DIFF_PIPELINE, "data/raw/rows.csv": "part,v\ntrain,1\ntest,2\n"}
    )
    _ok(_pivot(root, "repro"))
    _ok(_pivot(root, "config", "set", "remotes.origin", "s3://unused-bucket/cache"))
    _ok(_pivot(root, "config", "set", "default_remote", "origin"))
    # Push names the directory's listing object and both of its files. With the local cache
    # gone, fetch and pull name both files: the listing is stored again from its committed file.
    shutil.rmtree(root / ".pivot" / "cache" / "files")
    env = _env(AWS_ENDPOINT_URL="http://127.0.0.1:9", AWS_DEFAULT_REGION="us-east-1")

    relative = _ok(_pivot(link, command, "--dry-run", "data/d/split", env=env))
    absolute = _ok(_pivot(link, command, "--dry-run", str(link / "data" / "d" / "split"), env=env))
    refused = _pivot(link, command, "--dry-run", str(outside), env=env)

    assert line in relative, relative
    assert line in absolute, absolute
    assert refused.returncode == 1, refused.stdout + refused.stderr
    assert f"Target '{outside}' resolves outside project root" in refused.stderr, refused.stderr


def test_track_takes_an_absolute_path_typed_through_a_symlink_to_the_project(
    tmp_path: pathlib.Path,
) -> None:
    """An absolute path under a symlink to the project is tracked as the relative path is;
    a path outside the project is refused as before."""
    root, link, outside = _symlinked_project(tmp_path, {"notes.txt": "notes\n"})
    pvt = root / "notes.txt.pvt"
    recorded = dict[str, str]()

    for path in ("notes.txt", str(link / "notes.txt")):
        pvt.unlink(missing_ok=True)
        assert f"Tracked: {path}" in _ok(_pivot(link, "track", path)), path
        recorded[path] = pvt.read_text()
    refused = _pivot(link, "track", str(outside))

    assert recorded["notes.txt"] == recorded[str(link / "notes.txt")]
    assert refused.returncode == 1, refused.stdout + refused.stderr
    assert f"Path '{outside}' resolves outside project root" in refused.stderr, refused.stderr


_KIND_DIFF_PIPELINE = """\
import pathlib
import tempfile
from typing import Annotated

import pandas as pd

from pivot import Pipeline, directory, loaders, stage


@stage
def split(raw: pd.DataFrame) -> Annotated[dict[str, pd.DataFrame], directory, loaders.CSV()]:
    return {f"{name}.csv": frame for name, frame in raw.groupby("part")}


@stage
def bundle(raw: pd.DataFrame) -> pathlib.Path:
    out = pathlib.Path(tempfile.mkdtemp()) / "bundle"
    out.mkdir()
    raw.to_csv(out / "all.csv", index=False)
    (out / "notes.txt").write_text(f"rows={len(raw)}\\n")
    return out


@stage
def double(raw: pd.DataFrame) -> Annotated[pd.DataFrame, loaders.CSV()]:
    return raw


with Pipeline("d") as pipeline:
    rows = pipeline.input("rows.csv")
    split(rows)
    bundle(rows)
    double(rows)
"""


def _kind_project(tmp_path: pathlib.Path) -> pathlib.Path:
    """A committed project with a directory, a `pathlib.Path` directory and a file output,
    whose workspace then gains one row."""
    root = _project(
        tmp_path,
        {
            "pipeline.py": _KIND_DIFF_PIPELINE,
            "data/raw/rows.csv": "part,v\ntrain,1\ntest,2\n",
        },
    )
    _ok(_pivot(root, "repro"))
    _commit(root, "first run")
    (root / "data" / "raw" / "rows.csv").write_text("part,v\ntrain,1\ntrain,3\ntest,2\n")
    _ok(_pivot(root, "repro"))
    return root


@pytest.mark.parametrize(
    "target",
    [
        pytest.param("data/d/bundle.dat/notes.txt", id="non-data-member"),
        pytest.param("data/d/split/absent.csv", id="absent-member"),
        pytest.param("data/d/split/sub", id="missing-subdirectory"),
    ],
)
def test_diff_warns_about_a_target_inside_an_output_that_selects_no_data_file(
    tmp_path: pathlib.Path, target: str
) -> None:
    """A target inside a directory output that selects no data file on either side is named
    in the unmatched-target warning, while a data member beside it is compared silently."""
    root = _kind_project(tmp_path)
    warning = f"No CSV, JSON or JSONL stage output matches {target!r}"

    json_result = _pivot(root, "diff", "--json", target)
    text_output = _ok(_pivot(root, "diff", "--no-tui", target))
    control = _ok(_pivot(root, "diff", "--json", "data/d/bundle.dat/all.csv"))

    assert json_result.returncode == 0, json_result.stdout + json_result.stderr
    assert json.loads(json_result.stdout) == []
    assert warning in json_result.stderr, json_result.stderr
    assert warning in text_output, text_output
    assert "No CSV, JSON or JSONL" not in control, control
    assert '"path": "data/d/bundle.dat/all.csv"' in control, control


def _file_output_replaced_by_a_directory(root: pathlib.Path) -> tuple[str, str, str]:
    output = root / "data" / "d" / "double.csv"
    output.unlink()
    output.mkdir()
    (output / "a.csv").write_text("part,v\ntrain,1\n")
    return "data/d/double.csv", "a file", "a directory"


def _directory_output_replaced_by_a_file(root: pathlib.Path) -> tuple[str, str, str]:
    output = root / "data" / "d" / "split"
    shutil.rmtree(output)
    output.write_text("part,v\ntrain,1\n")
    return "data/d/split", "a directory", "a file"


@pytest.mark.parametrize(
    "replace",
    [
        pytest.param(_file_output_replaced_by_a_directory, id="file-replaced-by-directory"),
        pytest.param(_directory_output_replaced_by_a_file, id="directory-replaced-by-file"),
    ],
)
def test_diff_names_an_output_whose_kind_changed_and_compares_nothing_for_it(
    tmp_path: pathlib.Path, replace: Callable[[pathlib.Path], tuple[str, str, str]]
) -> None:
    """An output recorded at git HEAD as one kind and found in the workspace as the other is
    named in one line with both kinds, and none of its files is compared."""
    root = _kind_project(tmp_path)
    output, head_kind, workspace_kind = replace(root)
    line = f"Output {output} is {head_kind} at git HEAD and {workspace_kind} in the workspace"

    json_result = _pivot(root, "diff", "--json", output)
    text_output = _ok(_pivot(root, "diff", "--no-tui", output))
    every_output = _pivot(root, "diff", "--json", ".")

    assert json_result.returncode == 0, json_result.stdout + json_result.stderr
    assert json.loads(json_result.stdout) == []
    assert json_result.stderr.count(line) == 1, json_result.stderr
    assert text_output.count(line) == 1, text_output
    assert "No data file changes detected." in text_output, text_output
    assert every_output.stderr.count(line) == 1, every_output.stderr
    compared = {entry["path"] for entry in json.loads(every_output.stdout)}
    assert not any(path == output or path.startswith(f"{output}/") for path in compared)
    assert "data/d/bundle.dat/all.csv" in compared


_MAKE_DIRECTORY_PIPELINE = """\
from typing import Annotated

import pandas as pd

from pivot import Pipeline, directory, loaders, stage


@stage
def make(raw: pd.DataFrame) -> Annotated[dict[str, pd.DataFrame], directory, loaders.CSV()]:
    return {"all.csv": raw}


with Pipeline("d") as pipeline:
    make(pipeline.input("rows.csv"))
"""

_MAKE_FILE_PIPELINE = """\
from typing import Annotated

import pandas as pd

from pivot import Pipeline, loaders, stage


@stage
def make(raw: pd.DataFrame) -> Annotated[pd.DataFrame, loaders.CSV()]:
    return raw


with Pipeline("d") as pipeline:
    make(pipeline.input("rows.csv"))
"""

_MAKE_JSONL_PIPELINE = _MAKE_FILE_PIPELINE.replace("loaders.CSV()", "loaders.DataFrameJSONL()")


@pytest.mark.parametrize("keep_old_output", [True, False], ids=["old-output-kept", "old-removed"])
@pytest.mark.parametrize(
    ("before", "after", "old_output", "new_output", "removed", "added"),
    [
        pytest.param(
            _MAKE_DIRECTORY_PIPELINE,
            _MAKE_FILE_PIPELINE,
            "data/d/make",
            "data/d/make.csv",
            "data/d/make/all.csv",
            "data/d/make.csv",
            id="directory-to-file",
        ),
        pytest.param(
            _MAKE_FILE_PIPELINE,
            _MAKE_DIRECTORY_PIPELINE,
            "data/d/make.csv",
            "data/d/make",
            "data/d/make.csv",
            "data/d/make/all.csv",
            id="file-to-directory",
        ),
        pytest.param(
            _MAKE_FILE_PIPELINE,
            _MAKE_JSONL_PIPELINE,
            "data/d/make.csv",
            "data/d/make.jsonl",
            "data/d/make.csv",
            "data/d/make.jsonl",
            id="csv-to-dataframe-jsonl",
        ),
    ],
)
def test_diff_compares_a_moved_output_by_path(
    tmp_path: pathlib.Path,
    before: str,
    after: str,
    old_output: str,
    new_output: str,
    removed: str,
    added: str,
    keep_old_output: bool,
) -> None:
    """A stage whose output moved to a new path, by a kind or loader switch, lists HEAD's file
    as removed at the path HEAD recorded, and the workspace file as added at the new one, with
    no warning."""
    root = _project(tmp_path, {"pipeline.py": before, "data/raw/rows.csv": "v\n1\n2\n"})
    _ok(_pivot(root, "repro"))
    _commit(root, "first run")
    (root / "pipeline.py").write_text(after)
    (root / "data" / "raw" / "rows.csv").write_text("v\n1\n2\n3\n")
    _ok(_pivot(root, "repro"))
    old_output_on_disk = root / old_output
    assert old_output_on_disk.exists()
    if not keep_old_output:
        if old_output_on_disk.is_dir():
            shutil.rmtree(old_output_on_disk)
        else:
            old_output_on_disk.unlink()

    expected = {removed: (2, None), added: (None, 3)}
    selections = {
        "d/make": expected,
        ".": expected,
        new_output: {added: (None, 3)},
        old_output: {removed: (2, None)},
    }

    for target, answer in selections.items():
        result = _pivot(root, "diff", "--json", target)
        assert result.returncode == 0, result.stdout + result.stderr
        rows = {e["path"]: (e["old_rows"], e["new_rows"]) for e in json.loads(result.stdout)}
        assert rows == answer, (target, rows)
        assert "at git HEAD and" not in result.stderr, (target, result.stderr)
        assert "No CSV, JSON or JSONL" not in result.stderr, (target, result.stderr)

    text_output = _ok(_pivot(root, "diff", "--no-tui", "d/make"))
    assert "Rows: 2 (removed)" in text_output, text_output
    assert "Rows: 3 (added)" in text_output, text_output


_ALIAS_SHAPE_PIPELINE = """\
import pandas as pd

from pivot import Pipeline, stage

from shapes import Shape, make_value


@stage
def make(raw: pd.DataFrame) -> Shape:
    return make_value(raw)


with Pipeline("k") as pipeline:
    make(pipeline.input("rows.csv"))
"""

_DATAFRAME_FILE_SHAPE = """\
import pandas as pd

Shape = pd.DataFrame


def make_value(raw):
    return raw
"""

_DATAFRAME_DIRECTORY_SHAPE = """\
from typing import Annotated

import pandas as pd

from pivot import directory, loaders

Shape = Annotated[dict[str, pd.DataFrame], directory, loaders.CSV()]


def make_value(raw):
    return {"part.csv": raw}
"""


@pytest.mark.parametrize(
    ("first", "second", "removed", "added"),
    [
        pytest.param(
            _DATAFRAME_DIRECTORY_SHAPE,
            _DATAFRAME_FILE_SHAPE,
            "data/k/make/part.csv",
            "data/k/make.jsonl",
            id="directory-to-file",
        ),
        pytest.param(
            _DATAFRAME_FILE_SHAPE,
            _DATAFRAME_DIRECTORY_SHAPE,
            "data/k/make.jsonl",
            "data/k/make/part.csv",
            id="file-to-directory",
        ),
    ],
)
def test_diff_compares_an_output_moved_by_a_type_alias_switch(
    tmp_path: pathlib.Path, first: str, second: str, removed: str, added: str
) -> None:
    """A type-alias switch that moves an output between `data/k/make` and `data/k/make.jsonl`
    is a move: HEAD's file is removed at the HEAD path and the new file added, with no
    kind-change line naming either path."""
    root = _project(
        tmp_path,
        {
            "pipeline.py": _ALIAS_SHAPE_PIPELINE,
            "shapes.py": first,
            "data/raw/rows.csv": "v\n1\n2\n",
        },
    )
    _ok(_pivot(root, "repro"))
    _commit(root, "first run")
    (root / "shapes.py").write_text(second)
    _ok(_pivot(root, "repro"))

    result = _pivot(root, "diff", "--json", "k/make")

    assert result.returncode == 0, result.stdout + result.stderr
    rows = {e["path"]: (e["old_rows"], e["new_rows"]) for e in json.loads(result.stdout)}
    assert rows == {removed: (2, None), added: (None, 2)}, rows
    assert "git HEAD" not in result.stderr, result.stderr


_MAKE_AND_OTHER_PIPELINE = """\
from typing import Annotated

import pandas as pd

from pivot import Pipeline, loaders, stage


@stage
def make(raw: pd.DataFrame) -> Annotated[pd.DataFrame, loaders.CSV()]:
    return raw


@stage
def other(raw: pd.DataFrame) -> Annotated[pd.DataFrame, loaders.CSV()]:
    return raw.head(1)


with Pipeline("d") as pipeline:
    rows = pipeline.input("rows.csv")
    make(rows)
    other(rows)
"""


@pytest.mark.parametrize(
    ("head_paths", "expected"),
    [
        pytest.param(
            {"other": "data/d/make.csv"},
            {
                "d/make": {"data/d/make.csv": (2, 3)},
                "data/d/make.csv": {"data/d/make.csv": (2, 3)},
                "d/other": {"data/d/other.csv": (None, 1)},
                ".": {"data/d/make.csv": (2, 3), "data/d/other.csv": (None, 1)},
            },
            id="moved-onto-a-current-output",
        ),
        pytest.param(
            {"make": "data/d/old.csv", "other": "data/d/old.csv"},
            {
                "d/make": {"data/d/old.csv": (2, None), "data/d/make.csv": (None, 3)},
                "data/d/old.csv": {"data/d/old.csv": (2, None)},
                "d/other": {"data/d/other.csv": (None, 1)},
                ".": {
                    "data/d/old.csv": (2, None),
                    "data/d/make.csv": (None, 3),
                    "data/d/other.csv": (None, 1),
                },
            },
            id="two-moved-to-one-path",
        ),
        pytest.param(
            {"other": "data/d/make.csv", "make": None},
            {
                "d/other": {"data/d/make.csv": (1, 3), "data/d/other.csv": (None, 1)},
                "d/make": {"data/d/make.csv": (1, 3)},
                "data/d/make.csv": {"data/d/make.csv": (1, 3)},
                ".": {"data/d/make.csv": (1, 3), "data/d/other.csv": (None, 1)},
            },
            id="moved-onto-an-output-with-no-lock",
        ),
    ],
)
def test_diff_keeps_one_head_entry_for_a_path_two_locks_record(
    tmp_path: pathlib.Path,
    head_paths: dict[str, str | None],
    expected: dict[str, dict[str, tuple[int | None, int | None]]],
) -> None:
    """When two committed HEAD locks record one path, which takes a hand-edited lock, the path
    keeps the HEAD entry of the output that lives there, or else the first registered output's;
    the other entry is dropped. A moved entry placed at a current output with no HEAD entry of
    its own stays selectable through its own stage. A None path deletes that stage's lock."""
    root = _project(
        tmp_path, {"pipeline.py": _MAKE_AND_OTHER_PIPELINE, "data/raw/rows.csv": "v\n1\n2\n"}
    )
    _ok(_pivot(root, "repro"))
    for stage_name, head_path in head_paths.items():
        lock_path = root / ".pivot" / "stages" / "d" / f"{stage_name}.lock"
        if head_path is None:
            lock_path.unlink()
            continue
        lock_data = yaml.safe_load(lock_path.read_text())
        lock_data.get("records", lock_data)["outs"][0]["path"] = head_path
        lock_path.write_text(yaml.safe_dump(lock_data, sort_keys=False))
    _commit(root, "hand-edited locks")
    (root / "data" / "raw" / "rows.csv").write_text("v\n1\n2\n3\n")
    _ok(_pivot(root, "repro"))

    for target, answer in expected.items():
        result = _pivot(root, "diff", "--json", target)
        assert result.returncode == 0, result.stdout + result.stderr
        rows = {e["path"]: (e["old_rows"], e["new_rows"]) for e in json.loads(result.stdout)}
        assert rows == answer, (target, rows)


_SHOUT_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def shout(text: str) -> str:
    return text.upper()


with Pipeline("shadow") as pipeline:
    shout(pipeline.input("input.txt", t=str))
"""


@pytest.mark.parametrize(
    "extra_env",
    [pytest.param({}, id="plain"), pytest.param({"PYTHONPATH": "."}, id="pythonpath-root")],
)
def test_repro_runs_with_a_module_named_pivot_tui_in_the_project_root(
    tmp_path: pathlib.Path, extra_env: dict[str, str]
) -> None:
    """A user file ``pivot_tui.py`` at the project root is user code, though it shares the name
    of pivot-tui's package. Workers run with the project root on ``sys.path``, and
    ``PYTHONPATH=.`` puts it there for the coordinator too: pivot still imports and runs."""
    root = _project(
        tmp_path,
        {
            "pipeline.py": _SHOUT_PIPELINE,
            "data/raw/input.txt": "hello\n",
            "pivot_tui.py": 'NOTE = "a user module that shares the TUI name"\n',
        },
    )

    _ok(_pivot(root, "repro", env=_env(**extra_env)))

    assert (root / "data" / "shadow" / "shout.txt").read_text() == "HELLO\n"


_PIVOT_TUI_PACKAGE_PIPELINE = """\
from pivot import Pipeline, stage

import pivot_tui.helpers


@stage
def shout(text: str) -> str:
    return pivot_tui.helpers.tag(text)


with Pipeline("shpkg") as pipeline:
    shout(pipeline.input("input.txt", t=str))
"""


def test_editing_a_project_package_named_pivot_tui_reruns_its_stage(
    tmp_path: pathlib.Path,
) -> None:
    """A user package ``pivot_tui/`` at the project root, imported ahead of pivot-tui's through
    ``PYTHONPATH=.``, is user code: a stage that calls it is fingerprinted against it, and an
    edit to it reruns the stage. Pivot-tui's installed distribution, not whatever ``pivot_tui``
    names on ``sys.path``, decides what is pivot's code."""
    helper_source = 'def tag(text: str) -> str:\n    return text.strip().upper() + " {}"\n'
    root = _project(
        tmp_path,
        {
            "pipeline.py": _PIVOT_TUI_PACKAGE_PIPELINE,
            "data/raw/input.txt": "hello\n",
            "pivot_tui/__init__.py": "",
            "pivot_tui/helpers.py": helper_source.format("u1"),
        },
    )
    env = _env(PYTHONPATH=".")
    output = root / "data" / "shpkg" / "shout.txt"
    _ok(_pivot(root, "repro", env=env))
    assert output.read_text() == "HELLO u1"
    (root / "pivot_tui" / "helpers.py").write_text(helper_source.format("u2"))

    out = _ok(_pivot(root, "repro", env=env))

    assert output.read_text() == "HELLO u2", out


_COMPANYVENDOR_PYPROJECT = """\
[project]
name = "companyvendor"
version = "1.0"

[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
namespaces = true
"""


@pytest.fixture(scope="module")
def pip_checkout_venv(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """A virtualenv where ``pip install -e git+file://...`` installed ``companyvendor``, which
    provides ``company.vendor`` in the PEP 420 namespace package ``company``. pip checks it out in
    ``<venv>/src``: under the interpreter's prefix, outside site-packages. The virtualenv sees
    this environment's packages, pivot among them, and pip builds the package offline with this
    environment's setuptools. ``<venv>/bin/pivot`` runs pivot as its console script does."""
    base = tmp_path_factory.mktemp("pip-checkout")
    repo = base / "companyvendor"
    (repo / "src" / "company").mkdir(parents=True)
    (repo / "src" / "company" / "vendor.py").write_text("X = 1\n")
    (repo / "pyproject.toml").write_text(_COMPANYVENDOR_PYPROJECT)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    _commit(repo, "init")
    venv = base / "venv"
    python = str(venv / "bin" / "python")
    _ok(subprocess.run([sys.executable, "-m", "venv", str(venv)], capture_output=True, text=True))
    venv_site = (
        venv / "lib" / f"python{sys.version_info[0]}.{sys.version_info[1]}" / "site-packages"
    )
    (venv_site / "test_environment.pth").write_text(
        f"import site; site.addsitedir({sysconfig.get_path('purelib')!r})\n"
    )
    install = [python, "-m", "pip", "install", "--isolated", "--disable-pip-version-check", "-q"]
    install += ["--no-build-isolation", "--no-index", "--no-deps"]
    install += ["-e", f"companyvendor @ git+file://{repo}"]
    _ok(subprocess.run(install, env=_env(), capture_output=True, text=True, timeout=180))
    checkout = venv / "src" / "companyvendor" / "src" / "company"
    assert (checkout / "vendor.py").is_file(), "pip checks the package out under <venv>/src"
    (venv / "bin" / "pivot").write_text(
        "import sys\n\nfrom pivot.cli import main\n\nif __name__ == '__main__':\n    sys.exit(main())\n"
    )
    return venv


_NAMESPACE_HELPER = 'def tag(text: str) -> str:\n    return text.strip().upper() + " {}"\n'

_NAMESPACE_STAGE = """\
import company.project

from pivot import {names}


@stage
def shout(text: str) -> str:
    return company.project.tag(text)
"""

_NAMESPACE_WIRING = """
with Pipeline("ns") as pipeline:
    {call}(pipeline.input("input.txt", t=str))
"""


@pytest.mark.parametrize(
    ("files", "helper", "pythonpath"),
    [
        pytest.param(
            {
                "project/pipeline.py": _NAMESPACE_STAGE.format(names="Pipeline, stage")
                + _NAMESPACE_WIRING.format(call="shout"),
                "project/company/project.py": _NAMESPACE_HELPER.format("v1"),
            },
            "project/company/project.py",
            [],
            id="stage-at-the-root",
        ),
        pytest.param(
            {
                "project/pipeline.py": "from ext_stages import stages\n\nfrom pivot import Pipeline\n"
                + _NAMESPACE_WIRING.format(call="stages.shout"),
                "usercode/ext_stages/__init__.py": "",
                "usercode/ext_stages/stages.py": _NAMESPACE_STAGE.format(names="stage"),
                "usercode/company/project.py": _NAMESPACE_HELPER.format("v1"),
            },
            "usercode/company/project.py",
            ["usercode"],
            id="stage-outside-the-root",
        ),
    ],
)
def test_editing_the_user_s_part_of_a_namespace_shared_with_a_pip_checkout_reruns_its_stage(
    tmp_path: pathlib.Path,
    pip_checkout_venv: pathlib.Path,
    files: dict[str, str],
    helper: str,
    pythonpath: list[str],
) -> None:
    """The user's own ``company/project.py`` and a package pip checked out under the
    interpreter's prefix (``pip install -e git+...``) share the namespace package ``company``.
    The user's part is user code, whether the stage that calls it is at the project root or
    outside it: the stage runs, and an edit to the helper reruns it."""
    root = tmp_path / "project"
    root.mkdir()
    _project(root, {"data/raw/input.txt": "hello\n"})
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    command = [str(pip_checkout_venv / "bin" / name) for name in ("python", "pivot")] + ["repro"]
    env = _helper_env_with_path(*(tmp_path / entry for entry in pythonpath))
    output = root / "data" / "ns" / "shout.txt"
    first = _ok(
        subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=180)
    )
    assert output.read_text() == "HELLO v1", first
    (tmp_path / helper).write_text(_NAMESPACE_HELPER.format("v2"))

    out = _ok(
        subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=180)
    )

    assert output.read_text() == "HELLO v2", out
