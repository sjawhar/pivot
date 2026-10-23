"""`diff` reads the legacy cache after core.state_dir changes."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import yaml

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")


_PIPELINE = """\
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


def _run(cwd: pathlib.Path, home: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "HOME": str(home)},
    )


def test_data_diff_reads_a_legacy_cache_object(tmp_path: pathlib.Path) -> None:
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
    (root / "data" / "raw" / "input.txt").write_text("old\n")
    (root / "pipeline.py").write_text(_PIPELINE)
    assert _run(root, home, _PIVOT, "repro").returncode == 0
    assert _run(root, home, "git", "add", "-A").returncode == 0
    assert _run(root, home, "git", "commit", "-qm", "legacy layout").returncode == 0
    (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: altstate\n")
    output = root / "data" / "test" / "process.csv"
    output.unlink()
    output.write_text("id,value\n1,new\n")

    diff = _run(root, home, _PIVOT, "diff", "--no-tui", "data/test/process.csv")

    assert diff.returncode == 0, diff.stdout + diff.stderr
    assert "Rows: 1 -> 1" in diff.stdout
    assert "~1 modified" in diff.stdout
    assert diff.stderr.count("Pivot 0.5.2 stored locks and cache") == 1, diff.stderr


def test_metrics_diff_reads_a_legacy_cache_object(tmp_path: pathlib.Path) -> None:
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
    (root / "pipeline.py").write_text(_METRIC_PIPELINE)
    assert _run(root, home, _PIVOT, "repro").returncode == 0
    # The metric is not committed, so the old side can come only from the cache, as 0.5.2 read it.
    (root / ".gitignore").write_text("/metrics/\n")
    assert _run(root, home, "git", "add", "-A").returncode == 0
    assert _run(root, home, "git", "commit", "-qm", "legacy layout").returncode == 0
    lock_data = yaml.safe_load((root / ".pivot" / "stages" / "test" / "process.lock").read_text())
    file_hash = lock_data["outs"][0]["hash"]
    cached = root / ".pivot" / "cache" / "files" / file_hash[:2] / file_hash[2:]
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text("score: 1\n")
    (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: altstate\n")
    output = root / "metrics" / "test" / "process.yaml"
    output.write_text("score: 2\n")

    diff = _run(root, home, _PIVOT, "metrics", "diff", "metrics/test/process.yaml")

    assert diff.returncode == 0, diff.stdout + diff.stderr
    assert "modified" in diff.stdout.lower(), diff.stdout
    assert "added" not in diff.stdout.lower(), diff.stdout
    assert diff.stderr.count("Pivot 0.5.2 stored locks and cache") == 1, diff.stderr


def test_metrics_diff_does_not_read_a_symlinked_legacy_cache_object(
    tmp_path: pathlib.Path,
) -> None:
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
    (root / "pipeline.py").write_text(_METRIC_PIPELINE)
    assert _run(root, home, _PIVOT, "repro").returncode == 0
    (root / ".gitignore").write_text("/metrics/\n")
    assert _run(root, home, "git", "add", "-A").returncode == 0
    assert _run(root, home, "git", "commit", "-qm", "metric base").returncode == 0
    lock_data = yaml.safe_load((root / ".pivot" / "stages" / "test" / "process.lock").read_text())
    file_hash = lock_data["outs"][0]["hash"]
    cache_object = root / ".pivot" / "cache" / "files" / file_hash[:2] / file_hash[2:]
    cache_object.parent.mkdir(parents=True, exist_ok=True)
    cache_object.unlink(missing_ok=True)
    private_metric = root / "private-metrics.json"
    private_metric.write_text('{"score": 999}\n')
    cache_object.symlink_to(private_metric)
    (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: altstate\n")
    (root / "metrics" / "test" / "process.yaml").write_text("score: 2\n")

    diff = _run(root, home, _PIVOT, "metrics", "diff", "metrics/test/process.yaml")

    assert diff.returncode == 0, diff.stdout + diff.stderr
    assert "999" not in diff.stdout
