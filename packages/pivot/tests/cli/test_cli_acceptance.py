"""End-to-end CLI acceptance tests: drive the installed `pivot` binary in a fresh project.

Each test runs the binary from the test interpreter's venv in a `git init` project,
the way a user does, so it covers discovery, config and the real command wiring.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

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
    assert output.read_text() == '{"a":2}\n{"a":4}\n'


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
