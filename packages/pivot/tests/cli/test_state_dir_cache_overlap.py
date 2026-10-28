"""A cache directory that overlaps the stage-lock directory falls back to the state directory's
cache, and readers use that effective cache."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

from tests import helpers

if TYPE_CHECKING:
    import pathlib

# A cache may not overlap the directory containing committed stage locks.

_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def process(value: str) -> str:
    return value.upper()


with Pipeline("test") as pipeline:
    process(pipeline.input("input.txt", t=str))
"""


def _project(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "project"
    root.mkdir()
    for args in (
        ("init", "-q"),
        ("config", "user.email", "t@example.com"),
        ("config", "user.name", "t"),
    ):
        assert helpers.run_cli(root, home, "git", *args).returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("input\n")
    (root / "pipeline.py").write_text(_PIPELINE)
    assert (
        helpers.run_pivot(root, home, "config", "set", "core.state_dir", "altstate").returncode == 0
    )
    return home, root


def _assert_lock_is_trackable(root: pathlib.Path, home: pathlib.Path) -> None:
    lock = root / "altstate" / "stages" / "test" / "process.lock"
    assert lock.is_file()
    ignored = helpers.run_cli(root, home, "git", "check-ignore", "-q", str(lock.relative_to(root)))
    assert ignored.returncode == 1, ignored.stdout + ignored.stderr
    assert helpers.run_cli(root, home, "git", "add", "-A").returncode == 0
    staged = helpers.run_cli(root, home, "git", "diff", "--cached", "--name-only")
    assert str(lock.relative_to(root)) in staged.stdout.splitlines(), staged.stdout


def test_cache_dir_equal_to_stages_warns_and_falls_back_to_state_cache(
    tmp_path: pathlib.Path,
) -> None:
    home, root = _project(tmp_path)
    assert (
        helpers.run_pivot(root, home, "config", "set", "cache.dir", "altstate/stages").returncode
        == 0
    )

    result = helpers.run_pivot(root, home, "repro")

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stderr.count("overlaps the stage-lock directory") == 1, result.stderr
    assert (root / "altstate" / "cache" / "files").is_dir()
    _assert_lock_is_trackable(root, home)


def test_environment_cache_dir_under_stages_warns_and_falls_back_to_state_cache(
    tmp_path: pathlib.Path,
) -> None:
    home, root = _project(tmp_path)

    result = helpers.run_pivot(
        root, home, "repro", extra_env={"PIVOT_CACHE_DIR": "altstate/stages/cache"}
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stderr.count("overlaps the stage-lock directory") == 1, result.stderr
    assert (root / "altstate" / "cache" / "files").is_dir()
    _assert_lock_is_trackable(root, home)


def test_cache_dir_containing_stages_warns_and_falls_back_to_state_cache(
    tmp_path: pathlib.Path,
) -> None:
    home, root = _project(tmp_path)
    assert helpers.run_pivot(root, home, "config", "set", "cache.dir", "altstate").returncode == 0

    result = helpers.run_pivot(root, home, "repro")

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stderr.count("overlaps the stage-lock directory") == 1, result.stderr
    assert (root / "altstate" / "cache" / "files").is_dir()
    _assert_lock_is_trackable(root, home)


# Readers use the non-overlapping effective cache directory.
#
# Each fixture keeps its old output out of Git and places it in `<state>/cache`, then makes the
# configured or environment-selected cache overlap `<state>/stages`. The reader warns once and
# uses `<state>/cache`, reporting the old/new modification.

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
        assert helpers.run_cli(root, home, "git", *args).returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("1\n")
    (root / "pipeline.py").write_text(pipeline)
    assert helpers.run_pivot(root, home, "repro").returncode == 0
    (root / ".gitignore").write_text(f"/{ignored}/\n")
    assert helpers.run_cli(root, home, "git", "add", "-A").returncode == 0
    assert helpers.run_cli(root, home, "git", "commit", "-qm", "cache-only base").returncode == 0
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

    result = helpers.run_pivot(
        root, home, "diff", "--no-tui", str(output.relative_to(root)), extra_env=env
    )

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

    result = helpers.run_pivot(
        root, home, "metrics", "diff", str(output.relative_to(root)), extra_env=env
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "modified" in result.stdout.lower(), result.stdout
    assert "added" not in result.stdout.lower(), result.stdout
    expected_warnings = 0 if source == "control" else 1
    assert result.stderr.count("overlaps the stage-lock directory") == expected_warnings, (
        result.stderr
    )
