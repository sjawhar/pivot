"""End-to-end CLI acceptance tests: drive the installed `pivot` binary in a fresh project.

Each test runs the binary from the test interpreter's venv in a `git init` project,
the way a user does, so it covers discovery, config and the real command wiring.
"""

from __future__ import annotations

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
