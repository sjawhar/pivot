"""Integration tests for cache reconciliation of file outputs."""

from __future__ import annotations

import json
import pathlib
import shutil
from typing import Annotated

import pytest

from pivot import compose, executor, loaders
from pivot.storage import store as store_mod

_JSON_RESULT = Annotated[dict[str, int], loaders.JSON[dict[str, int]]()]


@compose.stage
def _produce_json_with_marker() -> _JSON_RESULT:
    marker = pathlib.Path("run_marker.txt")
    run_count = int(marker.read_text()) + 1 if marker.exists() else 1
    marker.write_text(str(run_count))
    return {"value": 42, "run": run_count}


def _output_path(pipeline: compose.Pipeline) -> pathlib.Path:
    stage = pipeline.get_stage(f"{pipeline.name}/_produce_json_with_marker")
    workspace = store_mod.WorkspaceStore(
        project_root=pipeline.root,
        pipeline_name=pipeline.name,
        input_bindings=pipeline.input_bindings,
    )
    return workspace.resolve_display_path(stage["outs"][0])


def _prepare(pipeline: compose.Pipeline, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    monkeypatch.chdir(pipeline.root)
    with compose.Pipeline(pipeline.name, root=pipeline.root) as builder:
        _produce_json_with_marker()
    pipeline.include(builder)
    return _output_path(pipeline)


def _read_json(path: pathlib.Path) -> dict[str, int]:
    return json.loads(path.read_text())


@pytest.mark.parametrize(
    "unlink_before_corrupt",
    [pytest.param(True, id="always-unlink"), pytest.param(False, id="conditional-unlink")],
)
def test_corrupted_file_restored_on_cache_hit(
    unlink_before_corrupt: bool,
    mock_discovery: compose.Pipeline,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cache hit repairs a corrupted workspace output without re-executing."""
    output_path = _prepare(mock_discovery, monkeypatch)

    executor.run(pipeline=mock_discovery)
    assert _read_json(output_path) == {"value": 42, "run": 1}

    if unlink_before_corrupt or output_path.stat().st_mode & 0o777 == 0o444:
        output_path.unlink()
    output_path.write_text('{"corrupted": true}')

    executor.run(pipeline=mock_discovery)

    assert pathlib.Path("run_marker.txt").read_text() == "1"
    assert _read_json(output_path) == {"value": 42, "run": 1}


def test_invalid_json_in_corrupted_file_handled_gracefully(
    mock_discovery: compose.Pipeline,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid output bytes restore from cache instead of reaching a loader."""
    output_path = _prepare(mock_discovery, monkeypatch)

    executor.run(pipeline=mock_discovery)
    output_path.unlink()
    output_path.write_text('{"invalid": json content')

    executor.run(pipeline=mock_discovery)

    assert pathlib.Path("run_marker.txt").read_text() == "1"
    assert _read_json(output_path) == {"value": 42, "run": 1}


def test_corrupted_file_triggers_rerun_when_cache_empty(
    mock_discovery: compose.Pipeline,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A corrupt output reruns its stage when no matching cache object remains."""
    output_path = _prepare(mock_discovery, monkeypatch)

    executor.run(pipeline=mock_discovery)
    cache_dir = mock_discovery.root / ".pivot" / "cache" / "files"
    shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)
    output_path.unlink()
    output_path.write_text('{"corrupted": true}')

    executor.run(pipeline=mock_discovery)

    assert pathlib.Path("run_marker.txt").read_text() == "2"
    assert _read_json(output_path) == {"value": 42, "run": 2}
