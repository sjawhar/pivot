"""End-to-end CLI acceptance tests: drive the installed `pivot` binary in a fresh project.

Each test runs the binary from the test interpreter's venv in a `git init` project,
the way a user does, so it covers discovery, config and the real command wiring.
"""

from __future__ import annotations

import json
import os
import pathlib
import py_compile
import shutil
import subprocess
import sys
import sysconfig
import time
import urllib.request
import uuid

import boto3
import pytest
import xxhash
import yaml

from pivot import loaders
from pivot.storage import cache

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")


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
    lock_data = yaml.safe_load(
        (root / ".pivot" / "stages" / "directory" / "assemble.lock").read_text()
    )
    assert [entry["relpath"] for entry in lock_data["outs"][0]["manifest"]] == ["target.txt"]

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
    pvt = yaml.safe_load((root / "data" / "tracked.pvt").read_text())
    assert [entry["relpath"] for entry in pvt["manifest"]] == ["keep.txt"]


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
    """`.pivot/.gitignore` entries are anchored: a pipeline named `cache` or `locks` keeps its locks."""
    root = _project(tmp_path, {})
    lock_files = [
        ".pivot/stages/cache/train.lock",
        ".pivot/stages/locks/train.lock",
        ".pivot/stages/state.lmdb/train.lock",
    ]
    for rel in lock_files:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("{}")

    check = subprocess.run(
        ["git", "check-ignore", "--verbose", *lock_files],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert check.returncode == 1, f"lock files ignored:\n{check.stdout}"


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
    lock_data = yaml.safe_load((clone / ".pivot" / "stages" / "dir" / "assemble.lock").read_text())
    assert len(lock_data["outs"][0]["manifest"]) == len(expected)

    assert "1 cached" in _ok(_pivot(clone, "status", env=env))
    assert "0 ran, 1 cached" in _ok(_pivot(clone, "repro", env=env))
    assert "Verification passed" in _ok(_pivot(clone, "verify", env=env))
    assert "leaves out symlink" not in _ok(_pivot(clone, "commit", "dir/assemble", env=env))

    lock_data = yaml.safe_load((clone / ".pivot" / "stages" / "dir" / "assemble.lock").read_text())
    assert len(lock_data["outs"][0]["manifest"]) == len(expected)
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
    lock_path: pathlib.Path, tree: pathlib.Path, scratch: pathlib.Path
) -> int:
    """Rewrite every directory hash of ``tree`` in a lock the way pivot before 0.3.4 recorded
    it: the hash of its files alone (a copy of them without the empty directories), with no
    ``empty_dirs``. Return how many entries changed."""
    files_only = scratch / "files-only"
    shutil.copytree(tree, files_only)
    for directory in sorted(files_only.rglob("*"), reverse=True):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    old = cache.hash_directory(files_only)
    assert "empty_dirs" not in old
    data = yaml.safe_load(lock_path.read_text())
    changed = 0
    for entry in [*data["deps"], *data["outs"]]:
        if "empty_dirs" not in entry:
            continue
        assert entry["manifest"] == old["manifest"]
        entry["hash"] = old["hash"]
        del entry["empty_dirs"]
        changed += 1
    lock_path.write_text(yaml.safe_dump(data, sort_keys=False))
    return changed


@pytest.mark.parametrize("state", ["run-cache-kept", "state-cleared"])
def test_repro_keeps_empty_dirs_a_pivot_033_lock_does_not_record(
    tmp_path: pathlib.Path, state: str
) -> None:
    """A directory output whose lock pivot before 0.3.4 wrote, and whose tree on disk holds
    empty directories that lock cannot record, is not restored to the lock, which would delete
    the directories while every stage stayed cached. The stage runs (or takes its outputs from
    a run-cache entry that records the directories), both locks come back byte for byte as a
    fresh build writes them, and the consumer reads the tree a fresh build makes."""
    (tmp_path / "project").mkdir()
    root = _project(tmp_path / "project", {"pipeline.py": _EMPTY_DIRS_PIPELINE})
    _ok(_pivot(root, "repro"))
    tree = root / "data" / "empty" / "make" / "tree.dat"
    packed = (root / "data" / "empty" / "pack.json").read_text()
    stages = root / ".pivot" / "stages" / "empty"
    fresh_locks = {name: (stages / name).read_bytes() for name in ("make.lock", "pack.lock")}
    scratch = tmp_path / "scratch"
    assert _helper_rewrite_lock_as_pivot_033(stages / "make.lock", tree, scratch / "make") == 1
    assert _helper_rewrite_lock_as_pivot_033(stages / "pack.lock", tree, scratch / "pack") == 1
    if state == "state-cleared":
        shutil.rmtree(root / ".pivot" / "state.lmdb")

    status_proc = _pivot(root, "status", "--explain", "--json")
    _ok(status_proc)
    status = json.loads(status_proc.stdout)
    make = next(s for s in status["stages"] if s["name"] == "empty/make")
    assert make["status"] == "stale"
    assert make["reason"].startswith("Output holds empty directories its lock does not record")

    _ok(_pivot(root, "repro"))

    assert (tree / "bundle" / "classifier_source").is_dir()
    assert (tree / "logs").is_dir()
    assert (root / "data" / "empty" / "pack.json").read_text() == packed
    assert {name: (stages / name).read_bytes() for name in fresh_locks} == fresh_locks
    make_out = yaml.safe_load((stages / "make.lock").read_text())["outs"][0]
    assert make_out["empty_dirs"] == ["bundle/classifier_source", "logs"]
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

    pvt = yaml.safe_load((root / "data" / "tracked.pvt").read_text())
    assert sorted(entry["relpath"] for entry in pvt["manifest"]) == ["keep.txt", "nested/f.txt"]


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
    manifest = yaml.safe_load(pvt_path.read_text())["manifest"]
    assert sorted(entry["relpath"] for entry in manifest) == ["keep.txt", "nested/f.txt"]

    # A real symlink checkout through the symlinked custom cache root is still content.
    shutil.rmtree(tracked)
    _ok(_pivot(root, "checkout", "--checkout-mode", "symlink", "data/tracked"))
    assert (tracked / "keep.txt").resolve().is_relative_to((real_cache / "files").resolve())
    assert "1 clean" in _ok(_pivot(root, "status", "--tracked-only"))
    _ok(_pivot(root, "checkout", "data/tracked"))  # existing links match: skipped, not an error
    assert "leaves out symlink" not in _ok(_pivot(root, "track", "--force", "data/tracked"))
    assert yaml.safe_load(pvt_path.read_text())["manifest"] == manifest


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
    import lazy_helper

    return raw.strip() + "-" + lazy_helper.VALUE


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
    """A module only a stage body imports is imported by the worker alone: one outside the root
    on ``PYTHONPATH``, or a project package named ``pivot_tui``, which the worker imports from the
    project root on its ``sys.path``, not pivot-tui's. The worker treats it as user code and
    caches bytecode for it checked by content, so an edit that keeps the file's size, made in the
    second the previous version was written and cached, is run when the stage next runs."""
    root = tmp_path / "project"
    root.mkdir()
    _project(root, {"pipeline.py": pipeline, "data/raw/in.txt": "x"})
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    env = _helper_env_with_path(*(tmp_path / entry for entry in pythonpath))
    output = root / "data" / "w" / "build.txt"
    _ok(_pivot(root, "repro", env=env))
    assert output.read_text() == "x-alpha"
    before = (tmp_path / helper).stat()
    (tmp_path / helper).write_text('VALUE = "bravo"\n')
    os.utime(tmp_path / helper, ns=(before.st_atime_ns, before.st_mtime_ns))
    (root / "data" / "raw" / "in.txt").write_text("y")

    out = _ok(_pivot(root, "repro", env=env))

    assert output.read_text() == "y-bravo", out


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
    """Every file under the outputs, the cache and the locks, by path."""
    tops = [root / "data", root / ".pivot" / "cache", root / ".pivot" / "stages"]
    return {
        str(path.relative_to(root)): path.read_bytes()
        for top in tops
        for path in sorted(top.rglob("*"))
        if path.is_file()
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


def _helper_drop_cached_file(root: pathlib.Path, lock_path: str, relpath: str) -> None:
    """Delete the cache blob of one file of a directory output."""
    lock_data = yaml.safe_load((root / ".pivot" / "stages" / lock_path).read_text())
    (entry,) = [e for e in lock_data["outs"][0]["manifest"] if e["relpath"] == relpath]
    (root / ".pivot" / "cache" / "files" / entry["hash"][:2] / entry["hash"][2:]).unlink()


def test_checkout_counts_the_files_of_a_missing_directory_output(tmp_path: pathlib.Path) -> None:
    """Missing, like restored and skipped, counts files: a three-file directory is three."""
    root = _project(tmp_path, {"pipeline.py": _THREE_FILES_PIPELINE, "notes.txt": "n"})
    _ok(_pivot(root, "repro"))
    _ok(_pivot(root, "track", "notes.txt"))
    shutil.rmtree(root / "data" / "n" / "split")
    for relpath in ("a.txt", "b.txt", "sub/c.txt"):
        _helper_drop_cached_file(root, "n/split.lock", relpath)

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
    _helper_drop_cached_file(root, "n/split.lock", "a.txt")

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


def test_diff_rejects_manifest_path_traversal_before_reading_outside_project(
    tmp_path: pathlib.Path,
) -> None:
    """`pivot diff` refuses a committed directory manifest path outside its project."""
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
    malicious_relpath = f"../../../../{secret.name}"
    try:
        lock_path = root / ".pivot" / "stages" / "d" / "split.lock"
        lock_data = yaml.safe_load(lock_path.read_text())
        lock_data["outs"][0]["manifest"] = [
            {
                "relpath": malicious_relpath,
                "hash": "a" * 16,
                "size": secret.stat().st_size,
                "isexec": False,
            }
        ]
        lock_path.write_text(yaml.safe_dump(lock_data, sort_keys=False))
        _commit(root, "malicious lock")

        result = _pivot(root, "diff", "--json", "d/split")
    finally:
        secret.unlink()

    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert ".pivot/stages/d/split.lock" in output
    assert malicious_relpath in output
    assert "TOP_SECRET_VALUE" not in output


def test_diff_refuses_a_workspace_symlink_outside_the_project(tmp_path: pathlib.Path) -> None:
    """`pivot diff` refuses a directory member symlinked to a file outside the project."""
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
    member = root / "data" / "d" / "split" / "train.csv"
    member.unlink()
    member.symlink_to(secret)
    try:
        result = _pivot(root, "diff", "--json", "d/split")
    finally:
        secret.unlink()

    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "data/d/split/train.csv" in output
    assert "TOP_SECRET_VALUE" not in output


def test_diff_refuses_a_manifest_path_with_a_null_byte(tmp_path: pathlib.Path) -> None:
    """`pivot diff` names and refuses a committed manifest path containing a null byte."""
    root = _project(
        tmp_path,
        {
            "pipeline.py": _DIFF_PIPELINE,
            "data/raw/rows.csv": "part,v\ntrain,1\n",
        },
    )
    _ok(_pivot(root, "repro"))
    _commit(root, "first run")

    lock_path = root / ".pivot" / "stages" / "d" / "split.lock"
    lock_data = yaml.safe_load(lock_path.read_text())
    lock_data["outs"][0]["manifest"] = [
        {"relpath": "nul\0.csv", "hash": "a" * 16, "size": 20, "isexec": False}
    ]
    lock_path.write_text(yaml.safe_dump(lock_data, sort_keys=False))
    _commit(root, "malicious lock")

    result = _pivot(root, "diff", "--json", "d/split")

    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert ".pivot/stages/d/split.lock" in output
    assert "not a project-relative POSIX path" in output


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
