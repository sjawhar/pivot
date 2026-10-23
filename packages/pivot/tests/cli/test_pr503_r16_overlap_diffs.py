"""Readers use the non-overlapping effective cache directory.

Each fixture keeps its old output out of Git and places it in `<state>/cache`, then makes the
configured or environment-selected cache overlap `<state>/stages`. The reader warns once and
uses `<state>/cache`, reporting the old/new modification.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest
import yaml

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")
_DATA_PIPELINE = """\
from typing import Annotated

import pandas

from pivot import Pipeline, loaders, stage


@stage
def process(value: str) -> Annotated[pandas.DataFrame, loaders.CSV()]:
    return pandas.DataFrame({"id": [1], "value": [value.strip()]})


with Pipeline("test") as pipeline:
    process(pipeline.input("input.txt", t=str))
"""
_METRIC_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, metric, stage


@stage
def process(value: str) -> Annotated[dict, metric]:
    return {"score": int(value.strip())}


with Pipeline("test") as pipeline:
    process(pipeline.input("input.txt", t=str))
"""


def _run(
    cwd: pathlib.Path, home: pathlib.Path, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "HOME": str(home), **(env or {})},
    )


def _base(tmp_path: pathlib.Path, pipeline: str, ignored: str) -> tuple[pathlib.Path, pathlib.Path]:
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "project"
    root.mkdir()
    for args in (
        ("init", "-q"),
        ("config", "user.email", "t@example.com"),
        ("config", "user.name", "t"),
    ):
        assert _run(root, home, "git", *args).returncode == 0
    assert _run(root, home, _PIVOT, "init").returncode == 0
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("1\n")
    (root / "pipeline.py").write_text(pipeline)
    assert _run(root, home, _PIVOT, "repro").returncode == 0
    (root / ".gitignore").write_text(f"/{ignored}/\n")
    assert _run(root, home, "git", "add", "-A").returncode == 0
    assert _run(root, home, "git", "commit", "-qm", "cache-only base").returncode == 0
    return home, root


def _cache_old_content(root: pathlib.Path, content: str) -> None:
    lock_data = yaml.safe_load((root / ".pivot" / "stages" / "test" / "process.lock").read_text())
    file_hash = lock_data["outs"][0]["hash"]
    legacy = root / ".pivot" / "cache" / "files" / file_hash[:2] / file_hash[2:]
    legacy.unlink(missing_ok=True)
    cached = root / "altstate" / "cache" / "files" / file_hash[:2] / file_hash[2:]
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(content)


def _overlap(root: pathlib.Path, source: str) -> dict[str, str] | None:
    config = "core:\n  state_dir: altstate\n"
    if source == "config":
        config += "cache:\n  dir: altstate/stages\n"
    elif source == "control":
        config += "cache:\n  dir: altstate/cache\n"
    (root / ".pivot" / "config.yaml").write_text(config)
    return {"PIVOT_CACHE_DIR": "altstate/stages"} if source == "environment" else None


@pytest.mark.parametrize("source", ["config", "environment", "control"])
def test_data_diff_uses_effective_non_overlapping_cache(
    tmp_path: pathlib.Path, source: str
) -> None:
    home, root = _base(tmp_path, _DATA_PIPELINE, "data/test")
    _cache_old_content(root, "id,value\n1,1\n")
    env = _overlap(root, source)
    output = root / "data" / "test" / "process.csv"
    output.unlink()
    output.write_text("id,value\n1,2\n")

    result = _run(root, home, _PIVOT, "diff", "--no-tui", str(output.relative_to(root)), env=env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Rows: 1 -> 1" in result.stdout
    assert "~1 modified" in result.stdout
    expected_warnings = 0 if source == "control" else 1
    assert result.stderr.count("overlaps the stage-lock directory") == expected_warnings, (
        result.stderr
    )


@pytest.mark.parametrize("source", ["config", "environment", "control"])
def test_metrics_diff_uses_effective_non_overlapping_cache(
    tmp_path: pathlib.Path, source: str
) -> None:
    home, root = _base(tmp_path, _METRIC_PIPELINE, "metrics")
    _cache_old_content(root, "score: 1\n")
    env = _overlap(root, source)
    output = root / "metrics" / "test" / "process.yaml"
    output.write_text("score: 2\n")

    result = _run(root, home, _PIVOT, "metrics", "diff", str(output.relative_to(root)), env=env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "modified" in result.stdout.lower(), result.stdout
    assert "added" not in result.stdout.lower(), result.stdout
    expected_warnings = 0 if source == "control" else 1
    assert result.stderr.count("overlaps the stage-lock directory") == expected_warnings, (
        result.stderr
    )
