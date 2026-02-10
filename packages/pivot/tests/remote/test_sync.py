from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pivot import loaders
from pivot import outputs as outputs_mod
from pivot.registry import RegistryStageInfo
from pivot.remote import storage as remote_storage
from pivot.remote import sync
from pivot.storage import cache as cache_mod
from pivot.storage import lock, track
from pivot.storage import state as state_mod
from pivot.types import DirHash, DirManifestEntry, FileHash, LockData, TransferResult

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable

    from pytest_mock import MockerFixture

# =============================================================================
# Unit tests for _extract_file_hashes_from_hash_info
# =============================================================================


def test_extract_file_hashes_from_file_hash() -> None:
    """FileHash returns its hash as the only element."""
    fh = FileHash(hash="abcdef1234567890")
    result = sync._extract_file_hashes_from_hash_info(fh)
    assert result == {"abcdef1234567890"}


def test_extract_file_hashes_from_dir_hash_excludes_tree_hash() -> None:
    """DirHash returns only manifest file hashes, not the tree hash."""
    dh = DirHash(
        hash="aaaaaaaaaaaaaaaa",  # tree hash — must be excluded
        manifest=[
            DirManifestEntry(relpath="a.csv", hash="1111111111111111", size=100, isexec=False),
            DirManifestEntry(relpath="b.csv", hash="2222222222222222", size=200, isexec=False),
        ],
    )
    result = sync._extract_file_hashes_from_hash_info(dh)
    assert result == {"1111111111111111", "2222222222222222"}
    assert "aaaaaaaaaaaaaaaa" not in result


def test_extract_file_hashes_from_dir_hash_empty_manifest() -> None:
    """DirHash with empty manifest returns empty set (tree hash excluded)."""
    dh = DirHash(hash="aaaaaaaaaaaaaaaa", manifest=[])
    result = sync._extract_file_hashes_from_hash_info(dh)
    assert result == set()


# =============================================================================
# Integration tests for get_stage_output_hashes / get_stage_dep_hashes
# =============================================================================


def _write_lock_with_dir_output(
    stages_dir: pathlib.Path,
    stage_name: str,
    tree_hash: str,
    file_hashes: list[str],
) -> None:
    """Helper: write a lock file with a directory output containing a tree hash."""
    manifest = [
        DirManifestEntry(relpath=f"file{i}.csv", hash=h, size=100, isexec=False)
        for i, h in enumerate(file_hashes)
    ]
    dir_hash = DirHash(hash=tree_hash, manifest=manifest)
    lock_data = LockData(
        code_manifest={},
        params={},
        dep_hashes={},
        output_hashes={"output_dir": dir_hash},
    )
    stage_lock = lock.StageLock(stage_name, stages_dir)
    stage_lock.write(lock_data)


def test_get_stage_output_hashes_excludes_tree_hash(set_project_root: pathlib.Path) -> None:
    """get_stage_output_hashes returns file hashes only, not tree hashes."""
    state_dir = set_project_root / ".pivot"
    stages_dir = lock.get_stages_dir(state_dir)
    stages_dir.mkdir(parents=True, exist_ok=True)

    tree_hash = "aaaaaaaaaaaaaaaa"
    file_hashes = ["1111111111111111", "2222222222222222"]
    _write_lock_with_dir_output(stages_dir, "my_stage", tree_hash, file_hashes)

    result = sync.get_stage_output_hashes(state_dir, ["my_stage"])

    assert "1111111111111111" in result
    assert "2222222222222222" in result
    assert tree_hash not in result


def test_get_stage_dep_hashes_excludes_tree_hash(set_project_root: pathlib.Path) -> None:
    """get_stage_dep_hashes returns file hashes only, not tree hashes."""
    state_dir = set_project_root / ".pivot"
    stages_dir = lock.get_stages_dir(state_dir)
    stages_dir.mkdir(parents=True, exist_ok=True)

    dep_manifest = [
        DirManifestEntry(relpath="dep.csv", hash="3333333333333333", size=50, isexec=False),
    ]
    dep_hash = DirHash(hash="bbbbbbbbbbbbbbbb", manifest=dep_manifest)
    lock_data = LockData(
        code_manifest={},
        params={},
        dep_hashes={str(set_project_root / "input_dir"): dep_hash},
        output_hashes={},
    )
    stage_lock = lock.StageLock("my_stage", stages_dir)
    stage_lock.write(lock_data)

    result = sync.get_stage_dep_hashes(state_dir, ["my_stage"])

    assert "3333333333333333" in result
    assert "bbbbbbbbbbbbbbbb" not in result


# =============================================================================
# get_target_hashes edge cases
# =============================================================================


def _helper_out_cache_false(path: str) -> outputs_mod.Out[pathlib.Path]:
    """Factory for Out(cache=False) — used in parametrized tests."""
    return outputs_mod.Out(path=path, loader=loaders.PathOnly(), cache=False)


def _helper_metric(path: str) -> outputs_mod.Metric:
    """Factory for Metric — used in parametrized tests."""
    return outputs_mod.Metric(path=path)


def test_get_target_hashes_invalid_stage_name_falls_through(
    set_project_root: pathlib.Path,
) -> None:
    """Target with invalid stage name chars (e.g. spaces) falls through to file path resolution."""
    state_dir = set_project_root / ".pivot"
    (state_dir / "stages").mkdir(parents=True, exist_ok=True)

    # "my data.csv" has a space, which is invalid for stage names.
    # Previously this would raise ValueError from StageLock.__init__.
    # After the fix it should fall through and end up in `unresolved`.
    result = sync.get_target_hashes(["my data.csv"], state_dir)
    assert result == set()


@pytest.mark.parametrize(
    "make_noncached",
    [
        pytest.param(_helper_out_cache_false, id="out-cache-false"),
        pytest.param(_helper_metric, id="metric"),
    ],
)
def test_get_target_hashes_excludes_noncached_outputs(
    set_project_root: pathlib.Path,
    make_noncached: Callable[[str], outputs_mod.BaseOut],
) -> None:
    """get_target_hashes excludes cache=False outputs from returned hashes."""
    state_dir = set_project_root / ".pivot"
    stages_dir = lock.get_stages_dir(state_dir)
    stages_dir.mkdir(parents=True, exist_ok=True)

    cached_out = outputs_mod.Out(
        path=str(set_project_root / "output.csv"),
        loader=loaders.PathOnly(),
        cache=True,
    )
    noncached = make_noncached(str(set_project_root / "metrics.json"))

    expanded_cached = outputs_mod.require_expanded(cached_out)
    expanded_noncached = outputs_mod.require_expanded(noncached)

    all_stages = {
        "my_stage": RegistryStageInfo(  # pyright: ignore[reportCallIssue] - partial for test
            state_dir=None,
            outs=[expanded_cached, expanded_noncached],
        )
    }

    cached_hash = FileHash(hash="1111111111111111")
    noncached_hash = FileHash(hash="2222222222222222")
    lock_data = LockData(
        code_manifest={},
        params={},
        dep_hashes={},
        output_hashes={
            str(set_project_root / "output.csv"): cached_hash,
            str(set_project_root / "metrics.json"): noncached_hash,
        },
    )
    lock.StageLock("my_stage", stages_dir).write(lock_data)

    result = sync.get_target_hashes(["my_stage"], state_dir, all_stages=all_stages)

    assert "1111111111111111" in result, "Cached output hash should be included"
    assert "2222222222222222" not in result, "Non-cached output hash should be excluded"


def test_get_target_hashes_file_target_excludes_noncached(
    set_project_root: pathlib.Path,
) -> None:
    """File-path target for a cache=False output returns no hashes."""
    state_dir = set_project_root / ".pivot"
    stages_dir = lock.get_stages_dir(state_dir)
    stages_dir.mkdir(parents=True, exist_ok=True)

    # Stage with one cached and one non-cached output
    cached_out = outputs_mod.Out(
        path=str(set_project_root / "output.csv"),
        loader=loaders.PathOnly(),
        cache=True,
    )
    metric_out = outputs_mod.Metric(path=str(set_project_root / "metrics.json"))

    expanded_cached = outputs_mod.require_expanded(cached_out)
    expanded_metric = outputs_mod.require_expanded(metric_out)

    all_stages = {
        "my_stage": RegistryStageInfo(  # pyright: ignore[reportCallIssue] - partial for test
            state_dir=None,
            outs=[expanded_cached, expanded_metric],
        )
    }

    cached_hash = FileHash(hash="1111111111111111")
    metric_hash = FileHash(hash="2222222222222222")
    lock_data = LockData(
        code_manifest={},
        params={},
        dep_hashes={},
        output_hashes={
            str(set_project_root / "output.csv"): cached_hash,
            str(set_project_root / "metrics.json"): metric_hash,
        },
    )
    lock.StageLock("my_stage", stages_dir).write(lock_data)

    # Target the non-cached file directly by path (not stage name)
    # This exercises _get_file_hash_from_stages (sync.py:107-124)
    result = sync.get_target_hashes(
        [str(set_project_root / "metrics.json")], state_dir, all_stages=all_stages
    )
    assert "2222222222222222" not in result, "Non-cached file target should return no hashes"

    # Target the cached file directly — should return its hash
    result_cached = sync.get_target_hashes(
        [str(set_project_root / "output.csv")], state_dir, all_stages=all_stages
    )
    assert "1111111111111111" in result_cached, "Cached file target should return its hash"


# =============================================================================
# Task 3: Push skips directory cache paths
# =============================================================================


async def test_push_skips_directory_cache_paths(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """Push should never enqueue directory paths for upload."""
    cache_dir = tmp_path / "cache"
    state_dir = tmp_path / ".pivot"
    state_dir.mkdir(parents=True)
    files_dir = cache_dir / "files"

    # Create a file cache entry
    file_hash = "1111111111111111"
    file_cache = files_dir / file_hash[:2] / file_hash[2:]
    file_cache.parent.mkdir(parents=True)
    file_cache.write_text("file content")

    # Create a directory cache entry (simulating SYMLINK mode tree hash)
    dir_hash = "aaaaaaaaaaaaaaaa"
    dir_cache = files_dir / dir_hash[:2] / dir_hash[2:]
    dir_cache.mkdir(parents=True)
    (dir_cache / "some_file.csv").write_text("data")

    # Verify preconditions: both entries exist, one is a file, one is a directory
    file_path = cache_mod.get_cache_path(files_dir, file_hash)
    dir_path = cache_mod.get_cache_path(files_dir, dir_hash)
    assert file_path.is_file(), "File cache entry should be a file"
    assert dir_path.is_dir(), "Dir cache entry should be a directory"

    mock_remote = mocker.Mock(spec=remote_storage.S3Remote)
    mock_state = mocker.Mock(spec=state_mod.StateDB)
    # Both hashes are in local cache; none known on remote
    mock_state.remote_hashes_intersection.return_value = set()
    mock_remote.bulk_exists = mocker.AsyncMock(return_value={file_hash: False, dir_hash: False})
    mock_remote.upload_batch = mocker.AsyncMock(
        return_value=[TransferResult(hash=file_hash, success=True)]
    )

    # Mock get_local_cache_hashes to return both file and directory hashes
    # (normally it filters out directories, but we want to test the filtering in _push_async)
    mocker.patch.object(sync, "get_local_cache_hashes", return_value={file_hash, dir_hash})

    result = await sync._push_async(cache_dir, state_dir, mock_remote, mock_state, "origin")

    # upload_batch should only receive the file entry, not the directory
    mock_remote.upload_batch.assert_called_once()
    uploaded_items = mock_remote.upload_batch.call_args[0][0]
    uploaded_hashes = {h for _, h in uploaded_items}
    assert file_hash in uploaded_hashes, "File hash should be uploaded"
    assert dir_hash not in uploaded_hashes, "Directory hash should be skipped"
    assert result["transferred"] == 1
    # Verify directory was counted in skipped total (skipped_non_file fix from Task 3)
    assert result["skipped"] == 1, "Directory cache path should be counted in skipped total"


# =============================================================================
# get_target_hashes: file path targets without pipeline
# =============================================================================


def test_get_target_hashes_file_path_with_lock_files_no_pipeline(
    set_project_root: pathlib.Path,
) -> None:
    """File path target resolves via lock file even without a pipeline."""
    state_dir = set_project_root / ".pivot"
    stages_dir = lock.get_stages_dir(state_dir)
    stages_dir.mkdir(parents=True, exist_ok=True)

    abs_output = str(set_project_root / "data" / "output.csv")
    output_hash = FileHash(hash="aabbccdd11223344")
    lock_data = LockData(
        code_manifest={},
        params={},
        dep_hashes={},
        output_hashes={abs_output: output_hash},
    )
    lock.StageLock("train", stages_dir).write(lock_data)

    result = sync.get_target_hashes([abs_output], state_dir, all_stages=None)

    assert result == {"aabbccdd11223344"}, "Should resolve hash from lock file without pipeline"


def test_get_target_hashes_pvt_file_target_without_pipeline(
    set_project_root: pathlib.Path,
) -> None:
    """File path target resolves via .pvt tracking file without a pipeline."""
    state_dir = set_project_root / ".pivot"
    (state_dir / "stages").mkdir(parents=True, exist_ok=True)

    pvt_path = set_project_root / "data" / "input.csv.pvt"
    pvt_path.parent.mkdir(parents=True, exist_ok=True)
    track.write_pvt_file(
        pvt_path, track.PvtData(path="input.csv", hash="eeff00112233aabb", size=100)
    )

    result = sync.get_target_hashes(
        [str(set_project_root / "data" / "input.csv")], state_dir, all_stages=None
    )

    assert result == {"eeff00112233aabb"}, "Should resolve hash from .pvt file without pipeline"


def test_get_target_hashes_pvt_suffix_stripped_before_lookup(
    set_project_root: pathlib.Path,
) -> None:
    """Target with .pvt suffix is stripped before lookup, avoiding .pvt.pvt."""
    state_dir = set_project_root / ".pivot"
    (state_dir / "stages").mkdir(parents=True, exist_ok=True)

    pvt_path = set_project_root / "data" / "result.csv.pvt"
    pvt_path.parent.mkdir(parents=True, exist_ok=True)
    track.write_pvt_file(
        pvt_path, track.PvtData(path="result.csv", hash="1122334455667788", size=100)
    )

    result = sync.get_target_hashes(
        [str(set_project_root / "data" / "result.csv.pvt")], state_dir, all_stages=None
    )

    assert result == {"1122334455667788"}, ".pvt suffix should be stripped before hash lookup"


def test_get_target_hashes_unresolved_file_target_returns_empty(
    set_project_root: pathlib.Path,
) -> None:
    """File target that matches nothing returns empty set without crashing."""
    state_dir = set_project_root / ".pivot"
    stages_dir = lock.get_stages_dir(state_dir)
    stages_dir.mkdir(parents=True, exist_ok=True)

    lock_data = LockData(
        code_manifest={},
        params={},
        dep_hashes={},
        output_hashes={str(set_project_root / "other" / "file.csv"): FileHash(hash="aaaa")},
    )
    lock.StageLock("train", stages_dir).write(lock_data)

    result = sync.get_target_hashes(
        [str(set_project_root / "nonexistent.csv")], state_dir, all_stages=None
    )

    assert result == set(), "Unresolved target should return empty set, not crash"
