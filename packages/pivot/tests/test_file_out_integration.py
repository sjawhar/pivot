"""Integration tests for JSON file-output cache restoration.

Stages return JSON values; Pivot chooses the workspace path. A project-root marker
records executions so each test distinguishes cache restore from a rerun.
"""

from __future__ import annotations

import json
import pathlib
import shutil
from typing import TYPE_CHECKING, Annotated, Any, cast

import pytest

from pivot import executor, loaders
from pivot.storage import cache
from tests import helpers

if TYPE_CHECKING:
    from pivot.registry import PipelineLike as Pipeline

_JSON_RESULT = Annotated[dict[str, int], loaders.JSON[dict[str, int]]()]


def _stage_produces_single_file_with_marker() -> _JSON_RESULT:
    """Return JSON and increment a marker each time the worker executes this stage."""
    marker_path = pathlib.Path("run_marker.txt")
    run_count = int(marker_path.read_text()) + 1 if marker_path.exists() else 1
    marker_path.write_text(str(run_count))
    return {"value": 42, "run": run_count}


def _get_run_count() -> int:
    marker_path = pathlib.Path("run_marker.txt")
    return int(marker_path.read_text()) if marker_path.exists() else 0


def _assert_json(path: pathlib.Path, expected: dict[str, int | bool]) -> None:
    actual = json.loads(path.read_text())
    assert actual == expected, f"Expected {expected}, got {actual}"


def _register_producer() -> pathlib.Path:
    helpers.register_test_stage(_stage_produces_single_file_with_marker, name="produce")
    return helpers.get_output_path("produce")


def _prepare_project(test_pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    root = test_pipeline.root
    monkeypatch.chdir(root)
    (root / ".git").mkdir(exist_ok=True)
    return root


def test_missing_file_restored_on_cache_hit(
    mock_discovery: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing output is restored from cache without re-executing its stage."""
    root = _prepare_project(mock_discovery, monkeypatch)
    output_file = _register_producer()

    executor.run(pipeline=mock_discovery)
    _assert_json(output_file, {"value": 42, "run": 1})
    assert _get_run_count() == 1

    output_file.unlink()
    executor.run(pipeline=mock_discovery)

    assert _get_run_count() == 1
    _assert_json(output_file, {"value": 42, "run": 1})
    assert output_file.is_relative_to(root)


@pytest.mark.parametrize(
    "unlink_before_corrupt",
    [pytest.param(True, id="always-unlink"), pytest.param(False, id="conditional-unlink")],
)
def test_corrupted_file_restored_on_cache_hit(
    unlink_before_corrupt: bool, mock_discovery: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrupted cached JSON output is restored without re-executing its stage."""
    _prepare_project(mock_discovery, monkeypatch)
    output_file = _register_producer()

    executor.run(pipeline=mock_discovery)
    _assert_json(output_file, {"value": 42, "run": 1})

    if unlink_before_corrupt or output_file.stat().st_mode & 0o777 == 0o444:
        output_file.unlink()
    output_file.write_text('{"corrupted": true}')
    _assert_json(output_file, {"corrupted": True})

    executor.run(pipeline=mock_discovery)

    assert _get_run_count() == 1
    _assert_json(output_file, {"value": 42, "run": 1})


def test_corrupted_file_triggers_rerun_when_cache_empty(
    mock_discovery: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _prepare_project(mock_discovery, monkeypatch)
    output_file = _register_producer()

    executor.run(pipeline=mock_discovery)
    _assert_json(output_file, {"value": 42, "run": 1})

    cache_dir = root / ".pivot/cache/files"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True)
    output_file.unlink()
    output_file.write_text('{"corrupted": true}')

    executor.run(pipeline=mock_discovery)

    assert _get_run_count() == 2
    _assert_json(output_file, {"value": 42, "run": 2})


def test_perfect_match_skips_without_modification(
    mock_discovery: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An intact output and cache entry skip cleanly without a new execution."""
    _prepare_project(mock_discovery, monkeypatch)
    output_file = _register_producer()

    executor.run(pipeline=mock_discovery)
    executor.run(pipeline=mock_discovery)

    assert _get_run_count() == 1
    _assert_json(output_file, {"value": 42, "run": 1})


def test_invalid_json_in_corrupted_file_handled_gracefully(
    mock_discovery: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_project(mock_discovery, monkeypatch)
    output_file = _register_producer()

    executor.run(pipeline=mock_discovery)
    output_file.unlink()
    output_file.write_text('{"invalid": json content missing closing brace')

    executor.run(pipeline=mock_discovery)

    assert _get_run_count() == 1
    _assert_json(output_file, {"value": 42, "run": 1})


def test_cache_files_are_readonly_and_properly_structured(
    mock_discovery: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """JSON cache objects are content-addressed, read-only and correct."""
    root = _prepare_project(mock_discovery, monkeypatch)
    output_file = _register_producer()

    executor.run(pipeline=mock_discovery)
    assert output_file.exists()

    cache_dir = root / ".pivot/cache/files"
    prefix_dirs = [path for path in cache_dir.iterdir() if path.is_dir()]
    assert prefix_dirs, "Cache should contain hash-prefix directories"
    cached_files = [
        path
        for prefix in prefix_dirs
        for path in prefix.iterdir()
        if cache.is_valid_hash(prefix.name + path.name)
    ]
    assert cached_files

    cache_file = cached_files[0]
    assert cache_file.stat().st_mode & 0o777 == 0o444
    _assert_json(cache_file, {"value": 42, "run": 1})


def _stage_fails_after_partial_write() -> _JSON_RESULT:
    """Write a partial JSON payload to Pivot's output path, then fail."""
    output_path = pathlib.Path("data/test/failing.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('{"partial": ')
    raise RuntimeError("Stage failed intentionally")


def test_stage_failure_leaves_workspace_consistent(
    mock_discovery: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Partial failure has no lock/cache entry and same-name retry succeeds."""
    root = _prepare_project(mock_discovery, monkeypatch)
    helpers.register_test_stage(_stage_fails_after_partial_write, name="failing")
    output_file = helpers.get_output_path("failing")
    lock_file = root / ".pivot/stages/test/failing.lock"

    results = executor.run(pipeline=mock_discovery)
    assert results["test/failing"]["status"] == "failed"
    assert not lock_file.exists()
    assert output_file.read_text() == '{"partial": '
    partial_hash, _ = cache.hash_file(output_file)
    assert not cache.get_cache_path(root / ".pivot/cache/files", partial_hash).exists()

    output_file.unlink()
    cast("Any", mock_discovery)._registry.clear()
    helpers.register_test_stage(_stage_produces_single_file_with_marker, name="failing")
    retry = executor.run(pipeline=mock_discovery)

    assert retry["test/failing"]["status"] == "ran"
    _assert_json(output_file, {"value": 42, "run": 1})
