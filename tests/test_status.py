from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Annotated, TypedDict

import pytest

from helpers import register_test_stage
from pivot import exceptions, executor, explain, loaders, outputs, status
from pivot.remote import config as remote_config
from pivot.remote import sync as transfer
from pivot.storage import cache, track
from pivot.storage import state as state_mod
from pivot.types import RemoteStatus, StageExplanation

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from pivot.pipeline import pipeline as pipeline_mod

# =============================================================================
# Note: Tests use set_project_root fixture from conftest.py which patches
# project._project_root_cache using patch.object for safety (fails if attr
# doesn't exist, unlike string-based patches that silently create new attrs).
# =============================================================================


# =============================================================================
# Output TypedDicts for annotation-based stages
# =============================================================================


class _StageAOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("a.txt", loaders.PathOnly())]


class _StageBOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("b.txt", loaders.PathOnly())]


# =============================================================================
# Pipeline Status Tests
# =============================================================================


def _helper_stage_a(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _StageAOutputs:
    _ = input_file  # deps tracked but not loaded in this simple test
    pathlib.Path("a.txt").write_text("output a")
    return {"output": pathlib.Path("a.txt")}


def _helper_stage_b(
    a_file: Annotated[pathlib.Path, outputs.Dep("a.txt", loaders.PathOnly())],
) -> _StageBOutputs:
    _ = a_file  # deps tracked but not loaded in this simple test
    pathlib.Path("b.txt").write_text("output b")
    return {"output": pathlib.Path("b.txt")}


def test_pipeline_status_all_cached(
    set_project_root: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    test_pipeline: pipeline_mod.Pipeline,
) -> None:
    """All stages should show cached after successful run."""
    (set_project_root / ".git").mkdir(exist_ok=True)
    (set_project_root / "input.txt").write_text("data")

    register_test_stage(_helper_stage_a, name="stage_a")

    monkeypatch.chdir(set_project_root)
    executor.run(pipeline=test_pipeline)
    all_stages = test_pipeline.snapshot()
    results, _ = status.get_pipeline_status(
        None,
        single_stage=False,
        all_stages=all_stages,
        stage_registry=test_pipeline._registry,
    )

    assert len(results) == 1
    assert results[0]["name"] == "stage_a"
    assert results[0]["status"] == "cached"
    assert results[0]["reason"] == ""


def test_pipeline_status_some_stale(
    set_project_root: pathlib.Path, test_pipeline: pipeline_mod.Pipeline
) -> None:
    """Stages with changed code should show stale."""
    (set_project_root / ".git").mkdir(exist_ok=True)
    (set_project_root / "input.txt").write_text("data")

    register_test_stage(_helper_stage_a, name="stage_a")

    all_stages = test_pipeline.snapshot()
    results, _ = status.get_pipeline_status(
        None,
        single_stage=False,
        all_stages=all_stages,
        stage_registry=test_pipeline._registry,
    )

    assert len(results) == 1
    assert results[0]["name"] == "stage_a"
    assert results[0]["status"] == "stale"
    assert results[0]["reason"] == "No previous run"


def test_pipeline_status_upstream_stale(
    set_project_root: pathlib.Path, test_pipeline: pipeline_mod.Pipeline
) -> None:
    """Stage should be marked stale if upstream is stale."""
    (set_project_root / ".git").mkdir(exist_ok=True)
    (set_project_root / "input.txt").write_text("data")

    register_test_stage(_helper_stage_a, name="stage_a")
    register_test_stage(_helper_stage_b, name="stage_b")

    all_stages = test_pipeline.snapshot()
    results, _ = status.get_pipeline_status(
        None,
        single_stage=False,
        all_stages=all_stages,
        stage_registry=test_pipeline._registry,
    )

    assert len(results) == 2

    stage_a = next(s for s in results if s["name"] == "stage_a")
    stage_b = next(s for s in results if s["name"] == "stage_b")

    assert stage_a["status"] == "stale"
    assert stage_a["reason"] == "No previous run"

    assert stage_b["status"] == "stale"
    assert "stage_a" in stage_b["upstream_stale"]


def test_pipeline_status_specific_stages(
    set_project_root: pathlib.Path, test_pipeline: pipeline_mod.Pipeline
) -> None:
    """Should only return status for specified stages."""
    (set_project_root / ".git").mkdir(exist_ok=True)
    (set_project_root / "input.txt").write_text("data")

    register_test_stage(_helper_stage_a, name="stage_a")
    register_test_stage(_helper_stage_b, name="stage_b")

    all_stages = test_pipeline.snapshot()
    results, _ = status.get_pipeline_status(
        ["stage_a"],
        single_stage=False,
        all_stages=all_stages,
        stage_registry=test_pipeline._registry,
    )

    assert len(results) == 1
    assert results[0]["name"] == "stage_a"


def test_pipeline_status_uses_per_stage_state_dir(
    set_project_root: pathlib.Path,
    test_pipeline: pipeline_mod.Pipeline,
    mocker: MockerFixture,
) -> None:
    """get_pipeline_status passes each stage's state_dir, not the global one."""
    (set_project_root / ".git").mkdir(exist_ok=True)
    (set_project_root / "input.txt").write_text("data")

    register_test_stage(_helper_stage_a, name="stage_a")

    all_stages = test_pipeline.snapshot()

    # Override stage_a's state_dir to a custom path
    custom_state_dir = set_project_root / "custom" / ".pivot"
    all_stages["stage_a"]["state_dir"] = custom_state_dir

    # Mock get_stage_explanation and capture calls via autospec (binds positional
    # args to parameter names, so the assertion doesn't break if the signature
    # is reordered or extended)
    mock_explain = mocker.patch.object(
        explain,
        "get_stage_explanation",
        autospec=True,
        return_value=StageExplanation(
            stage_name="stage_a",
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
        stage_registry=test_pipeline._registry,
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
    file_hash = cache.hash_file(data_file)

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
    old_hash = cache.hash_file(data_file)

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

    dir_hash, manifest = cache.hash_directory(data_dir)
    total_size = sum(entry["size"] for entry in manifest)

    pvt_data = track.PvtData(path="data", hash=dir_hash, size=total_size)
    track.write_pvt_file(set_project_root / "data.pvt", pvt_data)

    results = status.get_tracked_files_status(set_project_root)

    assert len(results) == 1
    assert results[0]["path"] == "data"
    assert results[0]["status"] == "clean"


def test_tracked_directory_modified(set_project_root: pathlib.Path) -> None:
    """Tracked directory should show modified when contents change."""
    (set_project_root / ".git").mkdir(exist_ok=True)

    data_dir = set_project_root / "data"
    data_dir.mkdir()
    (data_dir / "file1.txt").write_text("content1")

    dir_hash, manifest = cache.hash_directory(data_dir)
    total_size = sum(entry["size"] for entry in manifest)

    pvt_data = track.PvtData(path="data", hash=dir_hash, size=total_size)
    track.write_pvt_file(set_project_root / "data.pvt", pvt_data)

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
    pvt1 = track.PvtData(path="file1.txt", hash=cache.hash_file(file1), size=8)
    track.write_pvt_file(set_project_root / "file1.txt.pvt", pvt1)

    file2 = set_project_root / "file2.txt"
    file2.write_text("content2")
    pvt2 = track.PvtData(path="file2.txt", hash=cache.hash_file(file2), size=8)
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
    """Should suggest run when stages are stale."""
    suggestions = status.get_suggestions(
        stale_count=3, modified_count=0, push_count=0, pull_count=0
    )

    assert len(suggestions) == 1
    assert "pivot run" in suggestions[0]
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


class _TestStageOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("test_output.txt", loaders.PathOnly())]


def _helper_test_stage(
    input_file: Annotated[pathlib.Path, outputs.Dep("test_input.txt", loaders.PathOnly())],
) -> _TestStageOutputs:
    pathlib.Path("test_output.txt").write_text("output")
    return _TestStageOutputs(output=pathlib.Path("test_output.txt"))


def test_get_pipeline_status_uses_provided_graph(
    set_project_root: pathlib.Path,
    test_pipeline: pipeline_mod.Pipeline,
) -> None:
    """get_pipeline_status uses provided graph instead of building one."""
    from pivot.engine import graph as engine_graph

    (set_project_root / ".git").mkdir(exist_ok=True)
    (set_project_root / "test_input.txt").write_text("data")

    register_test_stage(_helper_test_stage, name="test_stage")

    # Build graph externally (simulating Engine)
    all_stages = test_pipeline.snapshot()
    external_graph = engine_graph.build_graph(all_stages)

    # Call status with provided graph
    results, returned_graph = status.get_pipeline_status(
        stages=["test_stage"],
        single_stage=False,
        all_stages=all_stages,
        stage_registry=test_pipeline._registry,
        graph=external_graph,
    )

    # Verify it found the stage
    assert len(results) == 1
    assert results[0]["name"] == "test_stage"

    # Verify returned graph is a stage-only DAG (not the bipartite graph)
    assert "test_stage" in returned_graph.nodes()
    # Bipartite graph would have "stage:test_stage", not "test_stage"
    assert "stage:test_stage" not in returned_graph.nodes()


def test_get_pipeline_explanations_uses_provided_graph(
    set_project_root: pathlib.Path,
    test_pipeline: pipeline_mod.Pipeline,
) -> None:
    """get_pipeline_explanations uses provided graph instead of building one."""
    from pivot.engine import graph as engine_graph

    (set_project_root / ".git").mkdir(exist_ok=True)
    (set_project_root / "test_input.txt").write_text("data")

    register_test_stage(_helper_test_stage, name="test_stage")

    # Build graph externally (simulating Engine)
    all_stages = test_pipeline.snapshot()
    external_graph = engine_graph.build_graph(all_stages)

    # Call explanations with provided graph
    explanations = status.get_pipeline_explanations(
        stages=["test_stage"],
        single_stage=False,
        all_stages=all_stages,
        stage_registry=test_pipeline._registry,
        graph=external_graph,
    )

    # Verify it found the stage
    assert len(explanations) == 1
    assert explanations[0]["stage_name"] == "test_stage"


# =============================================================================
# what_if_changed Tests
# =============================================================================


def test_what_if_changed_single_path(
    set_project_root: pathlib.Path, test_pipeline: pipeline_mod.Pipeline
) -> None:
    """what_if_changed returns affected stages for a single path."""
    (set_project_root / ".git").mkdir(exist_ok=True)
    (set_project_root / "input.txt").write_text("data")

    register_test_stage(_helper_stage_a, name="stage_a")
    register_test_stage(_helper_stage_b, name="stage_b")

    # input.txt affects stage_a, which in turn affects stage_b
    all_stages = test_pipeline.snapshot()
    affected = status.what_if_changed([pathlib.Path("input.txt")], all_stages=all_stages)

    assert "stage_a" in affected
    assert "stage_b" in affected, "Should include downstream stages"


def test_what_if_changed_intermediate_path(
    set_project_root: pathlib.Path, test_pipeline: pipeline_mod.Pipeline
) -> None:
    """what_if_changed returns only downstream stages for intermediate artifact."""
    (set_project_root / ".git").mkdir(exist_ok=True)
    (set_project_root / "input.txt").write_text("data")

    register_test_stage(_helper_stage_a, name="stage_a")
    register_test_stage(_helper_stage_b, name="stage_b")

    # a.txt is output of stage_a, input to stage_b
    all_stages = test_pipeline.snapshot()
    affected = status.what_if_changed([pathlib.Path("a.txt")], all_stages=all_stages)

    assert "stage_a" not in affected, "stage_a produces a.txt, doesn't consume it"
    assert "stage_b" in affected, "stage_b consumes a.txt"


def test_what_if_changed_unknown_path(
    set_project_root: pathlib.Path, test_pipeline: pipeline_mod.Pipeline
) -> None:
    """what_if_changed returns empty list for unknown paths."""
    (set_project_root / ".git").mkdir(exist_ok=True)
    (set_project_root / "input.txt").write_text("data")

    register_test_stage(_helper_stage_a, name="stage_a")

    all_stages = test_pipeline.snapshot()
    affected = status.what_if_changed([pathlib.Path("unknown.txt")], all_stages=all_stages)

    assert affected == []


def test_what_if_changed_multiple_paths(
    set_project_root: pathlib.Path, test_pipeline: pipeline_mod.Pipeline
) -> None:
    """what_if_changed returns union of affected stages for multiple paths."""
    (set_project_root / ".git").mkdir(exist_ok=True)
    (set_project_root / "input.txt").write_text("data")

    register_test_stage(_helper_stage_a, name="stage_a")
    register_test_stage(_helper_stage_b, name="stage_b")

    # Both paths affect different parts of the pipeline
    all_stages = test_pipeline.snapshot()
    affected = status.what_if_changed(
        [
            pathlib.Path("input.txt"),  # affects stage_a -> stage_b
            pathlib.Path("a.txt"),  # affects stage_b
        ],
        all_stages=all_stages,
    )

    assert "stage_a" in affected
    assert "stage_b" in affected


def test_what_if_changed_with_provided_graph(
    set_project_root: pathlib.Path, test_pipeline: pipeline_mod.Pipeline
) -> None:
    """what_if_changed uses provided graph instead of building one."""
    from pivot.engine import graph as engine_graph

    (set_project_root / ".git").mkdir(exist_ok=True)
    (set_project_root / "input.txt").write_text("data")

    register_test_stage(_helper_stage_a, name="stage_a")

    # Build graph externally
    all_stages = test_pipeline.snapshot()
    external_graph = engine_graph.build_graph(all_stages)

    # Call with provided graph
    affected = status.what_if_changed(
        [pathlib.Path("input.txt")], all_stages=all_stages, graph=external_graph
    )

    assert "stage_a" in affected
