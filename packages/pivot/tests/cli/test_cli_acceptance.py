"""End-to-end CLI acceptance tests: drive the installed `pivot` binary in a fresh project.

Each test runs the binary from the test interpreter's venv in a `git init` project,
the way a user does, so it covers discovery, config and the real command wiring.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import uuid

import boto3
import pytest
import yaml

from pivot import loaders

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
        + "Remote 'broken' must be a valid S3 URL (s3://bucket/prefix), got: not-a-remote"
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
        + "Remote 'broken' must be a valid S3 URL (s3://bucket/prefix), got: not-a-remote"
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
