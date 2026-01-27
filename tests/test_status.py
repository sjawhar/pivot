from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Annotated, TypedDict

import pytest

from helpers import register_test_stage
from pivot import exceptions, executor, loaders, outputs, status
from pivot.remote import config as remote_config
from pivot.remote import sync as transfer
from pivot.storage import cache, track
from pivot.storage import state as state_mod
from pivot.types import RemoteStatus

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

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
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All stages should show cached after successful run."""
    (set_project_root / ".git").mkdir()
    (set_project_root / "input.txt").write_text("data")

    register_test_stage(_helper_stage_a, name="stage_a")

    monkeypatch.chdir(set_project_root)
    executor.run(show_output=False)
    results, _ = status.get_pipeline_status(None, single_stage=False)

    assert len(results) == 1
    assert results[0]["name"] == "stage_a"
    assert results[0]["status"] == "cached"
    assert results[0]["reason"] == ""


def test_pipeline_status_some_stale(set_project_root: pathlib.Path) -> None:
    """Stages with changed code should show stale."""
    (set_project_root / ".git").mkdir()
    (set_project_root / "input.txt").write_text("data")

    register_test_stage(_helper_stage_a, name="stage_a")

    results, _ = status.get_pipeline_status(None, single_stage=False)

    assert len(results) == 1
    assert results[0]["name"] == "stage_a"
    assert results[0]["status"] == "stale"
    assert results[0]["reason"] == "No previous run"


def test_pipeline_status_upstream_stale(set_project_root: pathlib.Path) -> None:
    """Stage should be marked stale if upstream is stale."""
    (set_project_root / ".git").mkdir()
    (set_project_root / "input.txt").write_text("data")

    register_test_stage(_helper_stage_a, name="stage_a")
    register_test_stage(_helper_stage_b, name="stage_b")

    results, _ = status.get_pipeline_status(None, single_stage=False)

    assert len(results) == 2

    stage_a = next(s for s in results if s["name"] == "stage_a")
    stage_b = next(s for s in results if s["name"] == "stage_b")

    assert stage_a["status"] == "stale"
    assert stage_a["reason"] == "No previous run"

    assert stage_b["status"] == "stale"
    assert "stage_a" in stage_b["upstream_stale"]


def test_pipeline_status_specific_stages(set_project_root: pathlib.Path) -> None:
    """Should only return status for specified stages."""
    (set_project_root / ".git").mkdir()
    (set_project_root / "input.txt").write_text("data")

    register_test_stage(_helper_stage_a, name="stage_a")
    register_test_stage(_helper_stage_b, name="stage_b")

    results, _ = status.get_pipeline_status(["stage_a"], single_stage=False)

    assert len(results) == 1
    assert results[0]["name"] == "stage_a"


# =============================================================================
# Tracked Files Status Tests
# =============================================================================


def test_tracked_files_clean(set_project_root: pathlib.Path) -> None:
    """Tracked file should show clean when unchanged."""
    (set_project_root / ".git").mkdir()

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
    (set_project_root / ".git").mkdir()

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
    (set_project_root / ".git").mkdir()

    pvt_data = track.PvtData(path="data.txt", hash="abc123", size=100)
    track.write_pvt_file(set_project_root / "data.txt.pvt", pvt_data)

    results = status.get_tracked_files_status(set_project_root)

    assert len(results) == 1
    assert results[0]["path"] == "data.txt"
    assert results[0]["status"] == "missing"


def test_tracked_files_empty(set_project_root: pathlib.Path) -> None:
    """Should return empty list when no tracked files."""
    (set_project_root / ".git").mkdir()

    results = status.get_tracked_files_status(set_project_root)

    assert results == []


def test_tracked_directory_clean(set_project_root: pathlib.Path) -> None:
    """Tracked directory should show clean when unchanged."""
    (set_project_root / ".git").mkdir()

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
    (set_project_root / ".git").mkdir()

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
