from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pivot import exceptions, executor, explain, status
from pivot.remote import config as remote_config
from pivot.remote import sync as transfer
from pivot.storage import cache, listing, track
from pivot.storage import state as state_mod
from pivot.types import RemoteStatus, StageExplanation
from tests import helpers

if TYPE_CHECKING:
    import pathlib

    from pytest_mock import MockerFixture

    from pivot.registry import PipelineLike as Pipeline

# =============================================================================
# Note: Tests use set_project_root fixture from conftest.py which patches
# project._project_root_cache using patch.object for safety (fails if attr
# doesn't exist, unlike string-based patches that silently create new attrs).
# =============================================================================


def _helper_stage_a(input_file: str) -> str:
    _ = input_file
    return "output a"


def _helper_stage_b(a_file: str) -> str:
    _ = a_file
    return "output b"


def _helper_register_two_stages() -> None:
    stage_a = helpers.register_test_stage(
        _helper_stage_a, name="stage_a", deps={"input_file": "input.txt"}
    )
    helpers.register_test_stage(_helper_stage_b, name="stage_b", deps={"a_file": stage_a})


def _helper_write_tracked_directory(project_root: pathlib.Path, name: str) -> None:
    """Record a directory as `pivot track` does: its `.pvt` holds the directory's record, and
    the `.pvt.listing` beside it holds the directory's members."""
    hashed = cache.hash_directory(project_root / name)
    record = hashed["record"]
    pvt_path = project_root / f"{name}.pvt"
    listing.write(listing.tracked_listing_path(pvt_path), hashed["data"])
    track.write_pvt_file(
        pvt_path,
        track.PvtData(path=name, hash=record["hash"], size=record["size"], nfiles=record["nfiles"]),
    )


# =============================================================================
# Pipeline Status Tests
# =============================================================================


def test_pipeline_status_all_cached(
    set_project_root: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    test_pipeline: Pipeline,
) -> None:
    """All stages should show cached after successful run."""
    (set_project_root / ".git").mkdir(exist_ok=True)
    (set_project_root / "input.txt").write_text("data")

    helpers.register_test_stage(_helper_stage_a, name="stage_a", deps={"input_file": "input.txt"})

    monkeypatch.chdir(set_project_root)
    executor.run(pipeline=test_pipeline)
    all_stages = test_pipeline.snapshot()
    results, _ = status.get_pipeline_status(
        None,
        single_stage=False,
        all_stages=all_stages,
        pipeline=test_pipeline,
    )

    assert len(results) == 1
    assert results[0]["name"] == "test/stage_a"
    assert results[0]["status"] == "cached"
    assert results[0]["reason"] == ""


def test_pipeline_status_some_stale(
    set_project_root: pathlib.Path, test_pipeline: Pipeline
) -> None:
    """Stages with changed code should show stale."""
    (set_project_root / ".git").mkdir(exist_ok=True)
    (set_project_root / "input.txt").write_text("data")

    helpers.register_test_stage(_helper_stage_a, name="stage_a", deps={"input_file": "input.txt"})

    all_stages = test_pipeline.snapshot()
    results, _ = status.get_pipeline_status(
        None,
        single_stage=False,
        all_stages=all_stages,
        pipeline=test_pipeline,
    )

    assert len(results) == 1
    assert results[0]["name"] == "test/stage_a"
    assert results[0]["status"] == "stale"
    assert results[0]["reason"] == "No previous run"


def test_pipeline_status_upstream_stale(
    set_project_root: pathlib.Path, test_pipeline: Pipeline
) -> None:
    """Stage should be marked stale if upstream is stale."""
    (set_project_root / ".git").mkdir(exist_ok=True)
    (set_project_root / "input.txt").write_text("data")

    _helper_register_two_stages()

    all_stages = test_pipeline.snapshot()
    results, _ = status.get_pipeline_status(
        None,
        single_stage=False,
        all_stages=all_stages,
        pipeline=test_pipeline,
    )

    assert len(results) == 2

    stage_a = next(s for s in results if s["name"] == "test/stage_a")
    stage_b = next(s for s in results if s["name"] == "test/stage_b")

    assert stage_a["status"] == "stale"
    assert stage_a["reason"] == "No previous run"

    assert stage_b["status"] == "stale"
    assert "test/stage_a" in stage_b["upstream_stale"]


def test_pipeline_status_specific_stages(
    set_project_root: pathlib.Path, test_pipeline: Pipeline
) -> None:
    """Should only return status for specified stages."""
    (set_project_root / ".git").mkdir(exist_ok=True)
    (set_project_root / "input.txt").write_text("data")

    _helper_register_two_stages()

    all_stages = test_pipeline.snapshot()
    results, _ = status.get_pipeline_status(
        ["test/stage_a"],
        single_stage=False,
        all_stages=all_stages,
        pipeline=test_pipeline,
    )

    assert len(results) == 1
    assert results[0]["name"] == "test/stage_a"


def test_pipeline_status_uses_per_stage_state_dir(
    set_project_root: pathlib.Path,
    test_pipeline: Pipeline,
    mocker: MockerFixture,
) -> None:
    """get_pipeline_status passes each stage's state_dir, not the global one."""
    (set_project_root / ".git").mkdir(exist_ok=True)
    (set_project_root / "input.txt").write_text("data")

    helpers.register_test_stage(_helper_stage_a, name="stage_a", deps={"input_file": "input.txt"})

    all_stages = test_pipeline.snapshot()

    # Override stage_a's state_dir to a custom path
    custom_state_dir = set_project_root / "custom" / ".pivot"
    all_stages["test/stage_a"]["state_dir"] = custom_state_dir

    # Mock get_stage_explanation and capture calls via autospec (binds positional
    # args to parameter names, so the assertion doesn't break if the signature
    # is reordered or extended)
    mock_explain = mocker.patch.object(
        explain,
        "get_stage_explanation",
        autospec=True,
        return_value=StageExplanation(
            stage_name="test/stage_a",
            will_run=True,
            is_forced=False,
            reason="mocked",
            code_changes=[],
            param_changes=[],
            dep_changes=[],
            upstream_stale=[],
        ),
    )

    status.get_pipeline_status(
        None,
        single_stage=False,
        all_stages=all_stages,
        pipeline=test_pipeline,
    )

    mock_explain.assert_called_once()
    call_kwargs = mock_explain.call_args
    # With autospec, positional args are bound to parameter names
    assert call_kwargs.kwargs.get("state_dir") == custom_state_dir or (
        # Fallback: state_dir passed positionally (7th arg, index 6)
        len(call_kwargs.args) > 6 and call_kwargs.args[6] == custom_state_dir
    ), f"state_dir not set to custom path; call_args={call_kwargs}"


# =============================================================================
# Tracked Files Status Tests
# =============================================================================


def test_tracked_files_clean(set_project_root: pathlib.Path) -> None:
    """Tracked file should show clean when unchanged."""
    (set_project_root / ".git").mkdir(exist_ok=True)

    data_file = set_project_root / "data.txt"
    data_file.write_text("content")
    file_hash, _ = cache.hash_file(data_file)

    pvt_data = track.PvtData(path="data.txt", hash=file_hash, size=7)
    track.write_pvt_file(set_project_root / "data.txt.pvt", pvt_data)

    results = status.get_tracked_files_status(set_project_root)

    assert len(results) == 1
    assert results[0]["path"] == "data.txt"
    assert results[0]["status"] == "clean"


def test_tracked_files_modified(set_project_root: pathlib.Path) -> None:
    """Tracked file should show modified when changed."""
    (set_project_root / ".git").mkdir(exist_ok=True)

    data_file = set_project_root / "data.txt"
    data_file.write_text("original")
    old_hash, _ = cache.hash_file(data_file)

    pvt_data = track.PvtData(path="data.txt", hash=old_hash, size=8)
    track.write_pvt_file(set_project_root / "data.txt.pvt", pvt_data)

    data_file.write_text("modified content")

    results = status.get_tracked_files_status(set_project_root)

    assert len(results) == 1
    assert results[0]["path"] == "data.txt"
    assert results[0]["status"] == "modified"


def test_tracked_files_missing(set_project_root: pathlib.Path) -> None:
    """Tracked file should show missing when deleted."""
    (set_project_root / ".git").mkdir(exist_ok=True)

    pvt_data = track.PvtData(path="data.txt", hash="abc123", size=100)
    track.write_pvt_file(set_project_root / "data.txt.pvt", pvt_data)

    results = status.get_tracked_files_status(set_project_root)

    assert len(results) == 1
    assert results[0]["path"] == "data.txt"
    assert results[0]["status"] == "missing"


def test_tracked_files_empty(set_project_root: pathlib.Path) -> None:
    """Should return empty list when no tracked files."""
    (set_project_root / ".git").mkdir(exist_ok=True)

    results = status.get_tracked_files_status(set_project_root)

    assert results == []


def test_tracked_directory_clean(set_project_root: pathlib.Path) -> None:
    """Tracked directory should show clean when unchanged."""
    (set_project_root / ".git").mkdir(exist_ok=True)

    data_dir = set_project_root / "data"
    data_dir.mkdir()
    (data_dir / "file1.txt").write_text("content1")
    (data_dir / "file2.txt").write_text("content2")

    _helper_write_tracked_directory(set_project_root, "data")

    results = status.get_tracked_files_status(set_project_root)

    assert len(results) == 1, "The .pvt.listing beside data.pvt is not a tracked file"
    assert results[0]["path"] == "data"
    assert results[0]["status"] == "clean"
    assert results[0]["size"] == 16, "Size is the two 8-byte files' total, from the .pvt"


def test_tracked_directory_modified(set_project_root: pathlib.Path) -> None:
    """Tracked directory should show modified when contents change."""
    (set_project_root / ".git").mkdir(exist_ok=True)

    data_dir = set_project_root / "data"
    data_dir.mkdir()
    (data_dir / "file1.txt").write_text("content1")

    _helper_write_tracked_directory(set_project_root, "data")

    (data_dir / "file1.txt").write_text("modified content")

    results = status.get_tracked_files_status(set_project_root)

    assert len(results) == 1
    assert results[0]["path"] == "data"
    assert results[0]["status"] == "modified"


def test_tracked_files_progress_callback(set_project_root: pathlib.Path) -> None:
    """Progress callback should be called with (completed, total) for each file."""
    (set_project_root / ".git").mkdir(exist_ok=True)

    # Create two tracked files
    file1 = set_project_root / "file1.txt"
    file1.write_text("content1")
    hash1, _ = cache.hash_file(file1)
    pvt1 = track.PvtData(path="file1.txt", hash=hash1, size=8)
    track.write_pvt_file(set_project_root / "file1.txt.pvt", pvt1)

    file2 = set_project_root / "file2.txt"
    file2.write_text("content2")
    hash2, _ = cache.hash_file(file2)
    pvt2 = track.PvtData(path="file2.txt", hash=hash2, size=8)
    track.write_pvt_file(set_project_root / "file2.txt.pvt", pvt2)

    progress_calls = list[tuple[int, int]]()

    def on_progress(completed: int, total: int) -> None:
        progress_calls.append((completed, total))

    results = status.get_tracked_files_status(set_project_root, on_progress)

    assert len(results) == 2
    assert progress_calls == [(1, 2), (2, 2)]


# =============================================================================
# Remote Status Tests
# =============================================================================


def test_remote_status_no_remotes_configured(tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
    """Should raise RemoteNotConfiguredError when no remotes exist."""
    mocker.patch.object(remote_config, "list_remotes", return_value={})

    with pytest.raises(exceptions.RemoteNotConfiguredError):
        status.get_remote_status(None, tmp_path)


def test_remote_status_no_local_hashes(tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
    """Should return zero counts when no local cache files exist."""
    mocker.patch.object(remote_config, "list_remotes", return_value={"origin": "s3://bucket"})
    mocker.patch.object(
        transfer, "create_remote_from_name", return_value=(mocker.MagicMock(), "origin")
    )
    mocker.patch.object(remote_config, "get_remote_url", return_value="s3://bucket/prefix")
    mocker.patch.object(transfer, "get_local_cache_hashes", return_value=set())

    result = status.get_remote_status(None, tmp_path)

    assert result["name"] == "origin"
    assert result["url"] == "s3://bucket/prefix"
    assert result["push_count"] == 0
    assert result["pull_count"] == 0


def test_remote_status_with_local_hashes(tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
    """Should return push/pull counts from compare_status."""
    mocker.patch.object(remote_config, "list_remotes", return_value={"origin": "s3://bucket"})
    mocker.patch.object(
        transfer, "create_remote_from_name", return_value=(mocker.MagicMock(), "origin")
    )
    mocker.patch.object(remote_config, "get_remote_url", return_value="s3://bucket/prefix")
    mocker.patch.object(transfer, "get_local_cache_hashes", return_value={"hash1", "hash2"})

    async def mock_compare_status(*args: object) -> RemoteStatus:
        return RemoteStatus(
            local_only={"hash1"},
            remote_only={"hash3", "hash4", "hash5"},
            common={"hash2"},
        )

    mocker.patch.object(transfer, "compare_status", side_effect=mock_compare_status)
    mock_state_db = mocker.MagicMock()
    mock_state_db.__enter__ = mocker.MagicMock(return_value=mock_state_db)
    mock_state_db.__exit__ = mocker.MagicMock(return_value=False)
    mocker.patch.object(state_mod, "StateDB", return_value=mock_state_db)

    result = status.get_remote_status(None, tmp_path)

    assert result["name"] == "origin"
    assert result["url"] == "s3://bucket/prefix"
    assert result["push_count"] == 1, "Should count local_only hashes"
    assert result["pull_count"] == 3, "Should count remote_only hashes"


# =============================================================================
# Suggestions Tests
# =============================================================================


def test_suggestions_stale_stages() -> None:
    """Should suggest repro (runs the whole DAG without arguments) when stages are stale."""
    suggestions = status.get_suggestions(
        stale_count=3, modified_count=0, push_count=0, pull_count=0
    )

    assert len(suggestions) == 1
    assert "pivot repro" in suggestions[0]
    assert "3 stale stages" in suggestions[0]


def test_suggestions_single_stale_stage() -> None:
    """Should use singular 'stage' for count of 1."""
    suggestions = status.get_suggestions(
        stale_count=1, modified_count=0, push_count=0, pull_count=0
    )

    assert "1 stale stage" in suggestions[0]


def test_suggestions_modified_files() -> None:
    """Should suggest track when files are modified."""
    suggestions = status.get_suggestions(
        stale_count=0, modified_count=2, push_count=0, pull_count=0
    )

    assert len(suggestions) == 1
    assert "pivot track" in suggestions[0]
    assert "2 modified files" in suggestions[0]


def test_suggestions_push_files() -> None:
    """Should suggest push when files need uploading."""
    suggestions = status.get_suggestions(
        stale_count=0, modified_count=0, push_count=5, pull_count=0
    )

    assert len(suggestions) == 1
    assert "pivot push" in suggestions[0]
    assert "5 files" in suggestions[0]


def test_suggestions_pull_files() -> None:
    """Should suggest pull when files need downloading."""
    suggestions = status.get_suggestions(
        stale_count=0, modified_count=0, push_count=0, pull_count=3
    )

    assert len(suggestions) == 1
    assert "pivot pull" in suggestions[0]
    assert "3 files" in suggestions[0]


def test_suggestions_missing_outputs_with_remote_suggests_pull_first() -> None:
    """Outputs missing from the cache in a clone with a remote: pull before re-running."""
    suggestions = status.get_suggestions(
        stale_count=2,
        modified_count=0,
        push_count=0,
        pull_count=0,
        missing_outputs_count=2,
        remote_configured=True,
    )

    assert suggestions[0] == "Run `pivot pull` to restore outputs of 2 stages from the remote"
    assert "pivot repro" in suggestions[1]


def test_suggestions_missing_outputs_without_remote_omits_pull() -> None:
    """Without a remote there is nothing to pull from; only repro applies."""
    suggestions = status.get_suggestions(
        stale_count=2,
        modified_count=0,
        push_count=0,
        pull_count=0,
        missing_outputs_count=2,
        remote_configured=False,
    )

    assert len(suggestions) == 1
    assert "pivot repro" in suggestions[0]


def test_suggestions_multiple() -> None:
    """Should generate multiple suggestions when needed."""
    suggestions = status.get_suggestions(
        stale_count=2, modified_count=1, push_count=3, pull_count=1
    )

    assert len(suggestions) == 4


def test_suggestions_none_needed() -> None:
    """Should return empty list when nothing needs action."""
    suggestions = status.get_suggestions(
        stale_count=0, modified_count=0, push_count=0, pull_count=0
    )

    assert suggestions == []


# =============================================================================
# Graph Parameter Tests
# =============================================================================


def _helper_test_stage(input_file: str) -> str:
    return f"output:{input_file}"


def test_get_pipeline_status_uses_provided_graph(
    set_project_root: pathlib.Path,
    test_pipeline: Pipeline,
) -> None:
    """get_pipeline_status uses provided graph instead of building one."""
    from pivot.engine import graph as engine_graph

    (set_project_root / ".git").mkdir(exist_ok=True)
    (set_project_root / "test_input.txt").write_text("data")

    helpers.register_test_stage(
        _helper_test_stage, name="test_stage", deps={"input_file": "test_input.txt"}
    )

    # Build graph externally (simulating Engine)
    all_stages = test_pipeline.snapshot()
    external_graph = engine_graph.build_graph(all_stages)

    # Call status with provided graph
    results, returned_graph = status.get_pipeline_status(
        stages=["test/test_stage"],
        single_stage=False,
        all_stages=all_stages,
        pipeline=test_pipeline,
        graph=external_graph,
    )

    # Verify it found the stage
    assert len(results) == 1
    assert results[0]["name"] == "test/test_stage"

    # Verify returned graph is a stage-only DAG (not the bipartite graph)
    assert "test/test_stage" in returned_graph.nodes()
    # Bipartite graph would have "stage:test/test_stage", not "test/test_stage"
    assert "stage:test/test_stage" not in returned_graph.nodes()


def test_get_pipeline_explanations_uses_provided_graph(
    set_project_root: pathlib.Path,
    test_pipeline: Pipeline,
) -> None:
    """get_pipeline_explanations uses provided graph instead of building one."""
    from pivot.engine import graph as engine_graph

    (set_project_root / ".git").mkdir(exist_ok=True)
    (set_project_root / "test_input.txt").write_text("data")

    helpers.register_test_stage(
        _helper_test_stage, name="test_stage", deps={"input_file": "test_input.txt"}
    )

    # Build graph externally (simulating Engine)
    all_stages = test_pipeline.snapshot()
    external_graph = engine_graph.build_graph(all_stages)

    # Call explanations with provided graph
    explanations = status.get_pipeline_explanations(
        stages=["test/test_stage"],
        single_stage=False,
        all_stages=all_stages,
        pipeline=test_pipeline,
        graph=external_graph,
    )

    # Verify it found the stage
    assert len(explanations) == 1
    assert explanations[0]["stage_name"] == "test/test_stage"


# `what_if_changed` path semantics are owned by Task 3 (Copilot finding 2853197734).
