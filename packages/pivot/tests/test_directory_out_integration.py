"""Integration tests for directory outputs: a stage returns ``dict[str, T]`` tagged ``directory``.

Pivot writes each value as one file with the output's loader, then caches and restores the
directory as one artifact. A project-root marker counts executions so each test tells a
restore from the cache apart from a rerun. All tests run real pipelines in temporary projects.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import tempfile
from typing import TYPE_CHECKING, Annotated, Any, TypedDict, cast

import matplotlib.figure
import pandas as pd  # noqa: TC002 - compose resolves postponed stage annotations at runtime
import yaml

from pivot import compose, executor, loaders, stage_def
from pivot.storage import cache
from tests import helpers

if TYPE_CHECKING:
    import pytest

    from pivot.registry import PipelineLike as Pipeline

_JsonValue = str | int | bool


# =============================================================================
# Stage functions
# =============================================================================


class _DirectoryOutResult(TypedDict):
    results: Annotated[
        dict[str, dict[str, _JsonValue]],
        compose.directory,
        loaders.JSON[dict[str, _JsonValue]](),
    ]


def _stage_produces_abc_with_marker() -> _DirectoryOutResult:
    """Return files a, b and c, and count each execution in run_marker.txt."""
    marker_path = pathlib.Path("run_marker.txt")
    run_count = int(marker_path.read_text()) + 1 if marker_path.exists() else 1
    marker_path.write_text(str(run_count))

    return _DirectoryOutResult(
        results={
            "a.json": {"value": 1, "run": run_count},
            "b.json": {"value": 2, "run": run_count},
            "c.json": {"value": 3, "run": run_count},
        }
    )


def _stage_produces_nested_with_marker() -> _DirectoryOutResult:
    """Return files in nested subdirectories."""
    marker_path = pathlib.Path("run_marker.txt")
    run_count = int(marker_path.read_text()) + 1 if marker_path.exists() else 1
    marker_path.write_text(str(run_count))

    return _DirectoryOutResult(
        results={
            "top.json": {"level": "top", "run": run_count},
            "sub/nested.json": {"level": "nested", "run": run_count},
            "sub/deep/deeper.json": {"level": "deeper", "run": run_count},
        }
    )


def _stage_produces_many_files_with_marker() -> _DirectoryOutResult:
    """Return 50 files: enough to exercise bulk restore without being slow."""
    marker_path = pathlib.Path("run_marker.txt")
    run_count = int(marker_path.read_text()) + 1 if marker_path.exists() else 1
    marker_path.write_text(str(run_count))

    return _DirectoryOutResult(
        results={f"file_{i:03d}.json": {"index": i, "run": run_count} for i in range(50)}
    )


class _MultiDirectoryOutResult(TypedDict):
    first: Annotated[
        dict[str, dict[str, _JsonValue]],
        compose.directory,
        loaders.JSON[dict[str, _JsonValue]](),
    ]
    second: Annotated[
        dict[str, dict[str, _JsonValue]],
        compose.directory,
        loaders.JSON[dict[str, _JsonValue]](),
    ]


def _stage_produces_multi_directory_with_marker() -> _MultiDirectoryOutResult:
    """Return one file in each of two directory outputs."""
    marker_path = pathlib.Path("run_marker.txt")
    run_count = int(marker_path.read_text()) + 1 if marker_path.exists() else 1
    marker_path.write_text(str(run_count))

    return _MultiDirectoryOutResult(
        first={"a.json": {"dir": "first", "run": run_count}},
        second={"b.json": {"dir": "second", "run": run_count}},
    )


def _stage_split_rows(raw: pd.DataFrame) -> Annotated[dict[str, pd.DataFrame], compose.directory]:
    """A single directory output whose loader is inferred from the value type."""
    return {"train.jsonl": raw.iloc[:2], "test.jsonl": raw.iloc[2:]}


def _stage_count_rows(parts: dict[str, pd.DataFrame]) -> dict[str, int]:
    return {name: len(frame) for name, frame in parts.items()}


def _stage_read_one_file(part: dict[str, _JsonValue]) -> dict[str, _JsonValue]:
    return part


def _stage_writes_text_files() -> Annotated[dict[str, pathlib.Path], compose.directory]:
    """A directory of files the stage made outside the project, which pivot moves in."""
    scratch = pathlib.Path(tempfile.mkdtemp())
    files = dict[str, pathlib.Path]()
    for name in ("a.txt", "b.txt"):
        (scratch / name).write_text(name.upper())
        files[name] = scratch / name
    return files


def _stage_reads_path(path: pathlib.Path) -> dict[str, str]:
    return {"name": path.name, "directory": path.parent.name, "text": path.read_text()}


def _stage_nested_files() -> Annotated[
    dict[str, dict[str, int]], compose.directory, loaders.JSON[dict[str, int]]()
]:
    return {"x.d/a.json": {"v": 1}}


class _NamedKeysResult(TypedDict):
    files: Annotated[dict[str, dict[str, str]], compose.directory, loaders.JSON[dict[str, str]]()]


def _stage_returns_keys_that_normalize() -> _NamedKeysResult:
    """Keys that are not in normal form: accents in NFD, a hidden file, and ``.`` segments."""
    return {
        "files": {
            "cafe\u0301.json": {"name": "café"},
            "données.json": {"name": "données"},
            "数据.json": {"name": "数据"},
            ".metadata.json": {"name": "metadata"},
            "./baz.json": {"name": "baz"},
            "nested/./file.json": {"name": "file"},
        }
    }


def _stage_returns_executable_and_empty_directory() -> Annotated[
    dict[str, pathlib.Path], compose.directory
]:
    """An executable script and a directory holding an empty directory, made outside the project."""
    scratch = pathlib.Path(tempfile.mkdtemp())
    script = scratch / "run.sh"
    script.write_text("#!/bin/sh\necho hi\n")
    script.chmod(0o755)
    tree = scratch / "tree.d"
    (tree / "empty").mkdir(parents=True)
    (tree / "file.txt").write_text("x")
    return {"run.sh": script, "tree.d": tree}


class _CountParams(stage_def.StageParams):
    count: int = 3


def _stage_numbered_files(params: _CountParams) -> Annotated[dict[str, str], compose.directory]:
    return {f"{i}.txt": str(i) for i in range(params.count)}


class _OutsideParams(stage_def.StageParams):
    outside: str = ""


def _stage_returns_link_and_a_file_beneath_it(
    params: _OutsideParams,
) -> Annotated[dict[str, pathlib.Path], compose.directory]:
    """A returned directory holding a link to ``outside``, and a key whose file lies beneath it."""
    scratch = pathlib.Path(tempfile.mkdtemp())
    tree = scratch / "tree.d"
    tree.mkdir()
    (tree / "link").symlink_to(params.outside, target_is_directory=True)
    (scratch / "x.txt").write_text("x")
    return {"tree.d": tree, "tree.d/link/x.txt": scratch / "x.txt"}


class _MixedResult(TypedDict):
    parts: Annotated[dict[str, dict[str, int]], compose.directory, loaders.JSON[dict[str, int]]()]
    summary: Annotated[dict[str, int], compose.metric]
    chart: Annotated[matplotlib.figure.Figure, compose.plot]
    runs: Annotated[dict[str, int], compose.incremental]


def _stage_mixed(previous: dict[str, int]) -> _MixedResult:
    figure = matplotlib.figure.Figure()
    figure.subplots().plot([1, 2])
    return {
        "parts": {"a.json": {"v": 1}, "sub/b.json": {"v": 2}},
        "summary": {"files": 2},
        "chart": figure,
        "runs": {"count": previous.get("count", 0) + 1},
    }


# =============================================================================
# Helpers
# =============================================================================


def _get_run_count() -> int:
    marker_path = pathlib.Path("run_marker.txt")
    return int(marker_path.read_text()) if marker_path.exists() else 0


def _assert_files_exist(results_dir: pathlib.Path, expected: list[str]) -> None:
    """Assert that exactly the expected files exist in the directory, at any depth."""
    actual = sorted(p.relative_to(results_dir).as_posix() for p in results_dir.rglob("*.json"))
    assert actual == sorted(expected), f"Expected {expected}, got {actual}"


def _assert_file_content(path: pathlib.Path, expected: dict[str, _JsonValue]) -> None:
    actual = json.loads(path.read_text())
    assert actual == expected, f"Expected {expected}, got {actual}"


def _prepare_project(test_pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    root = test_pipeline.root
    monkeypatch.chdir(root)
    (root / ".git").mkdir(exist_ok=True)
    return root


def _register_producer(
    func: Any = _stage_produces_abc_with_marker, key: str = "results"
) -> pathlib.Path:
    helpers.register_test_stage(func, name="produce")
    return helpers.get_output_path("produce", key)


def _corrupt(path: pathlib.Path) -> None:
    """Replace a file's content (unlink first: a cached file may be a read-only hardlink)."""
    path.unlink()
    path.write_text('{"corrupted": true}')


_LOCK_REL = pathlib.Path(".pivot/stages/test/produce.lock")


# =============================================================================
# Cache hit restore scenarios
# =============================================================================


def test_cache_hit_restores_empty_directory(
    mock_discovery: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cache has a, b, c; the directory is empty: a, b, c come back without a rerun."""
    _prepare_project(mock_discovery, monkeypatch)
    results_dir = _register_producer()

    executor.run(pipeline=mock_discovery)
    _assert_files_exist(results_dir, ["a.json", "b.json", "c.json"])
    _assert_file_content(results_dir / "a.json", {"value": 1, "run": 1})
    assert _get_run_count() == 1

    for f in results_dir.glob("*.json"):
        f.unlink()
    assert results_dir.exists(), "Directory should still exist"

    executor.run(pipeline=mock_discovery)

    assert _get_run_count() == 1, "Stage should have skipped, not re-run"
    _assert_files_exist(results_dir, ["a.json", "b.json", "c.json"])
    _assert_file_content(results_dir / "a.json", {"value": 1, "run": 1})


def test_cache_hit_restores_partial_directory(
    mock_discovery: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cache has a, b, c; the directory holds only a: b and c come back, a stays."""
    _prepare_project(mock_discovery, monkeypatch)
    results_dir = _register_producer()

    executor.run(pipeline=mock_discovery)
    assert _get_run_count() == 1

    (results_dir / "b.json").unlink()
    (results_dir / "c.json").unlink()
    _assert_files_exist(results_dir, ["a.json"])

    executor.run(pipeline=mock_discovery)

    assert _get_run_count() == 1, "Stage should have skipped, not re-run"
    _assert_files_exist(results_dir, ["a.json", "b.json", "c.json"])
    _assert_file_content(results_dir / "b.json", {"value": 2, "run": 1})


def test_cache_hit_removes_extra_files(
    mock_discovery: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cache has a, b, c; the directory also holds d: d is removed without a rerun."""
    _prepare_project(mock_discovery, monkeypatch)
    results_dir = _register_producer()

    executor.run(pipeline=mock_discovery)
    assert _get_run_count() == 1

    (results_dir / "d.json").write_text('{"extra": true}')
    _assert_files_exist(results_dir, ["a.json", "b.json", "c.json", "d.json"])

    executor.run(pipeline=mock_discovery)

    assert _get_run_count() == 1, "Stage should have skipped, not re-run"
    _assert_files_exist(results_dir, ["a.json", "b.json", "c.json"])


def test_cache_hit_replaces_wrong_files(
    mock_discovery: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cache has a, b, c; the directory holds a, b, x: c comes back and x goes."""
    _prepare_project(mock_discovery, monkeypatch)
    results_dir = _register_producer()

    executor.run(pipeline=mock_discovery)
    assert _get_run_count() == 1

    (results_dir / "c.json").unlink()
    (results_dir / "x.json").write_text('{"wrong": true}')
    _assert_files_exist(results_dir, ["a.json", "b.json", "x.json"])

    executor.run(pipeline=mock_discovery)

    assert _get_run_count() == 1, "Stage should have skipped, not re-run"
    _assert_files_exist(results_dir, ["a.json", "b.json", "c.json"])
    _assert_file_content(results_dir / "c.json", {"value": 3, "run": 1})


def test_corrupted_file_restored_on_cache_hit(
    mock_discovery: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cache has a, b, c; a holds other content: the cached a comes back without a rerun."""
    _prepare_project(mock_discovery, monkeypatch)
    results_dir = _register_producer()

    executor.run(pipeline=mock_discovery)
    _assert_file_content(results_dir / "a.json", {"value": 1, "run": 1})

    _corrupt(results_dir / "a.json")
    _assert_file_content(results_dir / "a.json", {"corrupted": True})

    executor.run(pipeline=mock_discovery)

    assert _get_run_count() == 1, "Stage should have skipped, not re-run"
    _assert_file_content(results_dir / "a.json", {"value": 1, "run": 1})


def test_corrupted_file_triggers_rerun_when_cache_empty(
    mock_discovery: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the cache emptied, a corrupted file cannot be restored, so the stage runs again."""
    root = _prepare_project(mock_discovery, monkeypatch)
    results_dir = _register_producer()

    executor.run(pipeline=mock_discovery)
    _assert_file_content(results_dir / "a.json", {"value": 1, "run": 1})

    cache_dir = root / ".pivot/cache/files"
    shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)
    _corrupt(results_dir / "a.json")

    executor.run(pipeline=mock_discovery)

    assert _get_run_count() == 2, "Stage should have re-run when cache is empty"
    _assert_files_exist(results_dir, ["a.json", "b.json", "c.json"])
    _assert_file_content(results_dir / "a.json", {"value": 1, "run": 2})


def test_missing_directory_restored_on_cache_hit(
    mock_discovery: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The directory itself is deleted: it is recreated from the cache without a rerun."""
    _prepare_project(mock_discovery, monkeypatch)
    results_dir = _register_producer()

    executor.run(pipeline=mock_discovery)
    assert _get_run_count() == 1

    shutil.rmtree(results_dir)

    executor.run(pipeline=mock_discovery)

    assert _get_run_count() == 1, "Stage should have skipped, not re-run"
    _assert_files_exist(results_dir, ["a.json", "b.json", "c.json"])
    _assert_file_content(results_dir / "a.json", {"value": 1, "run": 1})


def test_partial_cache_corruption_triggers_rerun(
    mock_discovery: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cache lacks two of the files and the workspace copy is corrupted: the stage reruns."""
    root = _prepare_project(mock_discovery, monkeypatch)
    results_dir = _register_producer()

    executor.run(pipeline=mock_discovery)
    assert _get_run_count() == 1

    cache_dir = root / ".pivot/cache/files"
    for name in ("a.json", "b.json"):
        file_hash, _ = cache.hash_file(results_dir / name)
        cached = cache.get_cache_path(cache_dir, file_hash)
        assert cached.exists(), f"{name} should be in the cache"
        cached.unlink()
    _corrupt(results_dir / "a.json")

    executor.run(pipeline=mock_discovery)

    assert _get_run_count() == 2, "Stage should have re-run with corrupted cache"
    _assert_files_exist(results_dir, ["a.json", "b.json", "c.json"])
    _assert_file_content(results_dir / "a.json", {"value": 1, "run": 2})


# =============================================================================
# Nested, large and multiple directory outputs
# =============================================================================


def test_nested_directories_restored_on_cache_hit(
    mock_discovery: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Files in nested subdirectories all come back into an emptied output directory."""
    _prepare_project(mock_discovery, monkeypatch)
    results_dir = _register_producer(_stage_produces_nested_with_marker)

    executor.run(pipeline=mock_discovery)
    _assert_files_exist(results_dir, ["top.json", "sub/nested.json", "sub/deep/deeper.json"])
    assert _get_run_count() == 1

    shutil.rmtree(results_dir)
    results_dir.mkdir()

    executor.run(pipeline=mock_discovery)

    assert _get_run_count() == 1, "Stage should have skipped"
    _assert_files_exist(results_dir, ["top.json", "sub/nested.json", "sub/deep/deeper.json"])
    _assert_file_content(
        results_dir / "sub" / "deep" / "deeper.json", {"level": "deeper", "run": 1}
    )


def test_nested_file_corrupted_restored(
    mock_discovery: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrupted file deep inside the output is restored without a rerun."""
    _prepare_project(mock_discovery, monkeypatch)
    results_dir = _register_producer(_stage_produces_nested_with_marker)

    executor.run(pipeline=mock_discovery)
    assert _get_run_count() == 1

    nested_file = results_dir / "sub" / "deep" / "deeper.json"
    _corrupt(nested_file)

    executor.run(pipeline=mock_discovery)

    assert _get_run_count() == 1, "Stage should have skipped"
    _assert_file_content(nested_file, {"level": "deeper", "run": 1})


def test_large_directory_partial_restore(
    mock_discovery: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Half of 50 files deleted: the missing 25 come back without a rerun."""
    _prepare_project(mock_discovery, monkeypatch)
    results_dir = _register_producer(_stage_produces_many_files_with_marker)

    executor.run(pipeline=mock_discovery)
    assert len(list(results_dir.glob("*.json"))) == 50
    assert _get_run_count() == 1

    for i in range(25):
        (results_dir / f"file_{i:03d}.json").unlink()
    assert len(list(results_dir.glob("*.json"))) == 25

    executor.run(pipeline=mock_discovery)

    assert _get_run_count() == 1, "Stage should have skipped"
    assert len(list(results_dir.glob("*.json"))) == 50


def test_perfect_match_skips_without_modification(
    mock_discovery: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An intact output skips without its files being rewritten."""
    _prepare_project(mock_discovery, monkeypatch)
    results_dir = _register_producer()

    executor.run(pipeline=mock_discovery)
    before = {p.name: p.stat().st_mtime_ns for p in results_dir.glob("*.json")}

    executor.run(pipeline=mock_discovery)

    assert _get_run_count() == 1, "Stage should have skipped"
    after = {p.name: p.stat().st_mtime_ns for p in results_dir.glob("*.json")}
    assert after == before, "Skipping must not rewrite the files"


def test_multiple_directory_outs_restored(
    mock_discovery: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two directory outputs of one stage, one corrupted: both hold their cached files."""
    _prepare_project(mock_discovery, monkeypatch)
    first_dir = _register_producer(_stage_produces_multi_directory_with_marker, "first")
    second_dir = helpers.get_output_path("produce", "second")

    executor.run(pipeline=mock_discovery)
    assert (first_dir / "a.json").exists()
    assert (second_dir / "b.json").exists()
    assert _get_run_count() == 1

    _corrupt(first_dir / "a.json")

    executor.run(pipeline=mock_discovery)

    assert _get_run_count() == 1, "Stage should have skipped"
    _assert_file_content(first_dir / "a.json", {"dir": "first", "run": 1})
    _assert_file_content(second_dir / "b.json", {"dir": "second", "run": 1})


# =============================================================================
# Run cache
# =============================================================================


def test_run_cache_restores_directory_out(
    mock_discovery: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lock file and output directory both deleted: the run cache restores the output."""
    root = _prepare_project(mock_discovery, monkeypatch)
    results_dir = _register_producer()

    executor.run(pipeline=mock_discovery)
    _assert_file_content(results_dir / "a.json", {"value": 1, "run": 1})

    lock_file = root / _LOCK_REL
    assert lock_file.exists(), "the run should have written a lock file"
    lock_file.unlink()
    shutil.rmtree(results_dir)

    executor.run(pipeline=mock_discovery)

    assert _get_run_count() == 1, "Stage should have skipped via run cache"
    _assert_files_exist(results_dir, ["a.json", "b.json", "c.json"])
    _assert_file_content(results_dir / "a.json", {"value": 1, "run": 1})


def test_run_cache_restores_corrupted_directory(
    mock_discovery: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lock file deleted and one file corrupted: the run cache restores the cached file."""
    root = _prepare_project(mock_discovery, monkeypatch)
    results_dir = _register_producer()

    executor.run(pipeline=mock_discovery)
    assert _get_run_count() == 1

    (root / _LOCK_REL).unlink()
    _corrupt(results_dir / "a.json")

    executor.run(pipeline=mock_discovery)

    assert _get_run_count() == 1, "Stage should have skipped via run cache"
    _assert_file_content(results_dir / "a.json", {"value": 1, "run": 1})


def test_missing_cache_triggers_rerun(
    mock_discovery: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the lock, the cache and the run cache all gone, the stage runs again."""
    root = _prepare_project(mock_discovery, monkeypatch)
    results_dir = _register_producer()

    executor.run(pipeline=mock_discovery)
    assert _get_run_count() == 1

    (root / _LOCK_REL).unlink()
    shutil.rmtree(root / ".pivot/cache")
    shutil.rmtree(root / ".pivot/state.lmdb")
    _corrupt(results_dir / "a.json")

    executor.run(pipeline=mock_discovery)

    assert _get_run_count() == 2, "Stage should have re-run when cache is missing"
    _assert_file_content(results_dir / "a.json", {"value": 1, "run": 2})


# =============================================================================
# Writing and reading directory outputs
# =============================================================================


def test_single_return_writes_one_file_per_key_with_the_inferred_loader(
    mock_discovery: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dict of DataFrames is written one JSON Lines file per key, as a DataFrame output is."""
    root = _prepare_project(mock_discovery, monkeypatch)
    (root / "values.csv").write_text("id\n1\n2\n3\n4\n")
    helpers.register_test_stage(_stage_split_rows, name="split", deps={"raw": "values.csv"})
    output = helpers.get_output_path("split")

    results = executor.run(pipeline=mock_discovery)

    assert results["test/split"]["status"] == "ran", results
    assert output == root / "data" / "test" / "split"
    assert sorted(p.name for p in output.iterdir()) == ["test.jsonl", "train.jsonl"]
    train = loaders.DataFrameJSONL().load(output / "train.jsonl")
    test = loaders.DataFrameJSONL().load(output / "test.jsonl")
    assert train["id"].tolist() == [1, 2]
    assert test["id"].tolist() == [3, 4]


def test_consumer_receives_the_whole_directory_as_a_dict(
    mock_discovery: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A downstream stage gets every file of the directory, read with the producer's loader."""
    root = _prepare_project(mock_discovery, monkeypatch)
    (root / "values.csv").write_text("id\n1\n2\n3\n4\n5\n")
    parts = helpers.register_test_stage(_stage_split_rows, name="split", deps={"raw": "values.csv"})
    helpers.register_test_stage(_stage_count_rows, name="count", deps={"parts": parts})

    results = executor.run(pipeline=mock_discovery)

    assert results["test/count"]["status"] == "ran", results
    counts = yaml.safe_load(helpers.get_output_path("count").read_text())
    assert counts == {"test.jsonl": 3, "train.jsonl": 2}


def test_consumer_of_one_typed_dict_directory_file_receives_its_value(
    mock_discovery: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stage that takes one file of a directory field gets that file read with its loader."""
    _prepare_project(mock_discovery, monkeypatch)
    produced = helpers.register_test_stage(_stage_produces_abc_with_marker, name="produce")
    helpers.register_test_stage(
        _stage_read_one_file, name="read", deps={"part": produced.results["b.json"]}
    )

    results = executor.run(pipeline=mock_discovery)

    assert results["test/read"]["status"] == "ran", results
    assert yaml.safe_load(helpers.get_output_path("read").read_text()) == {"value": 2, "run": 1}


def test_one_file_consumer_skips_on_a_rerun_by_generation(
    mock_discovery: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With nothing changed, the one-file consumer skips on its producer's output generation."""
    _prepare_project(mock_discovery, monkeypatch)
    produced = helpers.register_test_stage(_stage_produces_abc_with_marker, name="produce")
    helpers.register_test_stage(
        _stage_read_one_file, name="read", deps={"part": produced.results["b.json"]}
    )
    executor.run(pipeline=mock_discovery)

    results = executor.run(pipeline=mock_discovery)

    assert results["test/read"]["status"] == "cached", results
    assert results["test/read"]["reason"] == "unchanged (generation)", results


def test_consumer_of_a_file_the_producer_did_not_write_fails_naming_both(
    mock_discovery: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_project(mock_discovery, monkeypatch)
    produced = helpers.register_test_stage(_stage_produces_abc_with_marker, name="produce")
    helpers.register_test_stage(
        _stage_read_one_file, name="read", deps={"part": produced.results["z.json"]}
    )

    results = executor.run(pipeline=mock_discovery)

    assert results["test/produce"]["status"] == "ran", results
    assert results["test/read"]["status"] == "failed", results
    reason = results["test/read"]["reason"]
    assert "test/produce" in reason and "results/z.json" in reason, reason


def test_consumer_of_one_file_of_a_path_directory_receives_that_files_path(
    mock_discovery: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each file of a ``dict[str, pathlib.Path]`` directory reads back as its workspace path."""
    _prepare_project(mock_discovery, monkeypatch)
    files = helpers.register_test_stage(_stage_writes_text_files, name="files")
    helpers.register_test_stage(_stage_reads_path, name="reads", deps={"path": files["b.txt"]})

    results = executor.run(pipeline=mock_discovery)

    assert results["test/reads"]["status"] == "ran", results
    assert yaml.safe_load(helpers.get_output_path("reads").read_text()) == {
        "name": "b.txt",
        "directory": "files",
        "text": "B.TXT",
    }


def test_consumer_of_a_subdirectory_of_a_directory_output_fails_naming_the_producer_and_path(
    mock_discovery: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`x.d` is a directory inside the output, not a file, so no loader can read it."""
    _prepare_project(mock_discovery, monkeypatch)
    nested = helpers.register_test_stage(_stage_nested_files, name="nest")
    helpers.register_test_stage(_stage_read_one_file, name="read", deps={"part": nested["x.d"]})

    results = executor.run(pipeline=mock_discovery)

    assert results["test/nest"]["status"] == "ran", results
    assert results["test/read"]["status"] == "failed", results
    reason = results["test/read"]["reason"]
    assert reason == "unreadable deps: test/nest:x.d (a directory, not a file)", reason
    assert "Traceback" not in reason, reason


def test_rerun_writes_inside_the_output_directory_through_no_symlink_left_there(
    mock_discovery: Pipeline,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """A symlink planted where a key's subdirectory goes is removed with the old output.

    Every directory between the output directory and a file is created by the worker, after it
    removed the previous output, so a file is never written through a link to outside.
    """
    root = _prepare_project(mock_discovery, monkeypatch)
    results_dir = _register_producer(_stage_produces_nested_with_marker)
    executor.run(pipeline=mock_discovery)
    outside = tmp_path_factory.mktemp("outside")
    shutil.rmtree(results_dir / "sub")
    (results_dir / "sub").symlink_to(outside, target_is_directory=True)

    results = executor.run(pipeline=mock_discovery, force=True)

    assert results["test/produce"]["status"] == "ran", results
    assert list(outside.iterdir()) == [], "nothing was written through the link"
    assert not (results_dir / "sub").is_symlink()
    assert (results_dir / "sub" / "deep" / "deeper.json").resolve().is_relative_to(root.resolve())


def test_rerun_replaces_a_symlink_left_at_the_output_path(
    mock_discovery: Pipeline,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """The previous output is removed before the directory is made, a link without its target."""
    _prepare_project(mock_discovery, monkeypatch)
    results_dir = _register_producer()
    executor.run(pipeline=mock_discovery)
    outside = tmp_path_factory.mktemp("outside")
    shutil.rmtree(results_dir)
    results_dir.symlink_to(outside, target_is_directory=True)

    results = executor.run(pipeline=mock_discovery, force=True)

    assert results["test/produce"]["status"] == "ran", results
    assert list(outside.iterdir()) == [], "nothing was written through the link"
    assert not results_dir.is_symlink()
    _assert_files_exist(results_dir, ["a.json", "b.json", "c.json"])


def test_returned_path_holding_a_link_cannot_place_another_keys_file_beneath_it(
    mock_discovery: Pipeline,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """A key inside another key's path is refused before anything is moved or written.

    So the only directories on the way to a key's file are ones the worker creates, and a
    link inside a returned directory is never followed by a later write.
    """
    _prepare_project(mock_discovery, monkeypatch)
    outside = tmp_path_factory.mktemp("outside")
    helpers.register_test_stage(
        _stage_returns_link_and_a_file_beneath_it,
        name="linked",
        params=_OutsideParams(outside=str(outside)),
    )

    results = executor.run(pipeline=mock_discovery)

    assert results["test/linked"]["status"] == "failed", results
    assert "cannot be both a file and the directory holding" in results["test/linked"]["reason"]
    assert list(outside.iterdir()) == [], "nothing was written through the link"
    assert not helpers.get_output_path("linked").exists()


def test_keys_are_written_at_their_normal_form(
    mock_discovery: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unicode keys are NFC, a hidden file with an extension is kept, ``.`` segments drop."""
    _prepare_project(mock_discovery, monkeypatch)
    files = _register_producer(_stage_returns_keys_that_normalize, key="files")

    results = executor.run(pipeline=mock_discovery)

    assert results["test/produce"]["status"] == "ran", results
    assert sorted(p.relative_to(files).as_posix() for p in files.rglob("*.json")) == sorted(
        [".metadata.json", "baz.json", "café.json", "données.json", "nested/file.json", "数据.json"]
    )
    assert json.loads((files / "caf\u00e9.json").read_text()) == {"name": "café"}


def test_executable_bit_and_empty_directory_survive_a_cache_restore(
    mock_discovery: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_project(mock_discovery, monkeypatch)
    helpers.register_test_stage(_stage_returns_executable_and_empty_directory, name="tools")
    output = helpers.get_output_path("tools")
    executor.run(pipeline=mock_discovery)
    assert os.access(output / "run.sh", os.X_OK)
    assert (output / "tree.d" / "empty").is_dir()
    cache.remove_output(output)

    results = executor.run(pipeline=mock_discovery)

    assert results["test/tools"]["status"] == "cached", results
    assert os.access(output / "run.sh", os.X_OK), "the executable bit came back from the cache"
    assert (output / "tree.d" / "empty").is_dir(), "the empty directory came back"
    assert list((output / "tree.d" / "empty").iterdir()) == []
    assert (output / "tree.d" / "file.txt").read_text() == "x"


def test_fewer_files_than_last_run_leave_no_stale_file(
    mock_discovery: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rerun that returns fewer keys leaves exactly those files in the directory."""
    _prepare_project(mock_discovery, monkeypatch)
    helpers.register_test_stage(_stage_numbered_files, name="numbered", params=_CountParams())
    output = helpers.get_output_path("numbered")
    executor.run(pipeline=mock_discovery)
    assert sorted(p.name for p in output.iterdir()) == ["0.txt", "1.txt", "2.txt"]

    cast("Any", mock_discovery)._registry.clear()
    helpers.register_test_stage(
        _stage_numbered_files, name="numbered", params=_CountParams(count=2)
    )
    results = executor.run(pipeline=mock_discovery)

    assert results["test/numbered"]["status"] == "ran", results
    assert sorted(p.name for p in output.iterdir()) == ["0.txt", "1.txt"]


def test_typed_dict_directory_field_beside_metric_plot_and_incremental(
    mock_discovery: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory field is written beside metric, plot and incremental fields of one stage."""
    root = _prepare_project(mock_discovery, monkeypatch)
    with compose.Pipeline(mock_discovery.name, root=root) as builder:
        compose.stage(_stage_mixed)(previous=builder.previous("runs"))
    mock_discovery.include(builder)
    parts = helpers.get_output_path("_stage_mixed", "parts")
    runs = helpers.get_output_path("_stage_mixed", "runs")

    executor.run(pipeline=mock_discovery)
    executor.run(pipeline=mock_discovery, force=True)

    assert parts == root / "data" / "test" / "_stage_mixed" / "parts"
    assert {p.relative_to(parts).as_posix() for p in parts.rglob("*.json")} == {
        "a.json",
        "sub/b.json",
    }
    assert json.loads((parts / "sub" / "b.json").read_text()) == {"v": 2}
    summary = helpers.get_output_path("_stage_mixed", "summary")
    assert yaml.safe_load(summary.read_text()) == {"files": 2}
    assert helpers.get_output_path("_stage_mixed", "chart").read_bytes().startswith(b"\x89PNG")
    assert yaml.safe_load(runs.read_text()) == {"count": 2}, "the incremental field kept its state"
