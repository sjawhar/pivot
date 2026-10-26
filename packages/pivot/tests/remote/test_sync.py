from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from pivot import loaders, types
from pivot.registry import RegistryStageInfo
from pivot.remote import storage as remote_storage
from pivot.remote import sync
from pivot.storage import cache as cache_mod
from pivot.storage import listing, lock, store, track
from pivot.storage import state as state_mod
from pivot.types import DirManifestEntry, FileHash, LockData, TransferResult

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

# =============================================================================
# Directory transfer fixtures
# =============================================================================


def _helper_hashed_directory(file_hashes: list[str]) -> listing.HashedDirectory:
    """Build a directory record together with the listing its hash names."""
    members = listing.DirListing(
        files=[
            DirManifestEntry(relpath=f"file{i}.csv", hash=file_hash, size=100, isexec=False)
            for i, file_hash in enumerate(file_hashes)
        ],
        empty_dirs=[],
    )
    data = listing.encode(members)
    return listing.HashedDirectory(
        record=listing.record_of(members, data),
        listing=members,
        data=data,
    )


def _write_lock_with_dir_output(
    stages_dir: pathlib.Path,
    identity: types.ArtifactIdentity,
    file_hashes: list[str],
) -> listing.HashedDirectory:
    """Write a directory output record and the listing held beside its lock."""
    hashed = _helper_hashed_directory(file_hashes)
    lock.StageLock(identity.producer, stages_dir).write(
        LockData(
            code_manifest={},
            params={},
            dep_hashes={},
            output_hashes={identity: hashed["record"]},
        ),
        hashed_dirs={hashed["record"]["hash"]: hashed},
    )
    return hashed


def _helper_stage_info() -> RegistryStageInfo:
    """Return the fields target transfer resolution needs for one stage."""
    return RegistryStageInfo(  # pyright: ignore[reportCallIssue] - partial for test
        state_dir=None,
        outs=[],
        incremental_inputs={},
        input_paths={},
    )


# =============================================================================
# Targeted directory transfer resolution
# =============================================================================


def test_get_target_hashes_directory_output_includes_listing_and_files(
    set_project_root: pathlib.Path,
) -> None:
    """A targeted directory names its listing object and every listed file object."""
    state_dir = set_project_root / ".pivot"
    stages_dir = lock.get_stages_dir(state_dir)
    file_hashes = ["1111111111111111", "2222222222222222"]
    hashed = _write_lock_with_dir_output(
        stages_dir, types.ArtifactIdentity("my_stage", "output_dir"), file_hashes
    )

    result = sync.get_target_hashes(
        ["my_stage"],
        state_dir,
        all_stages={"my_stage": _helper_stage_info()},
        cache_dir=state_dir / "cache",
    )

    assert result == {hashed["record"]["hash"], *file_hashes}


def test_get_target_hashes_directory_dependency_includes_listing_and_files(
    set_project_root: pathlib.Path,
) -> None:
    """A target's directory dependency names its listing object and every listed file."""
    state_dir = set_project_root / ".pivot"
    stages_dir = lock.get_stages_dir(state_dir)
    dep_file_hashes = ["3333333333333333"]
    dep_identity = types.ArtifactIdentity("upstream", None)
    dependency = _write_lock_with_dir_output(stages_dir, dep_identity, dep_file_hashes)

    output_hash = "4444444444444444"
    lock.StageLock("my_stage", stages_dir).write(
        LockData(
            code_manifest={},
            params={},
            dep_hashes={dep_identity: dependency["record"]},
            output_hashes={
                types.ArtifactIdentity("my_stage", None): FileHash(hash=output_hash),
            },
        )
    )

    result = sync.get_target_hashes(
        ["my_stage"],
        state_dir,
        include_deps=True,
        all_stages={
            "my_stage": _helper_stage_info(),
            "upstream": _helper_stage_info(),
        },
        cache_dir=state_dir / "cache",
    )

    assert result == {output_hash, dependency["record"]["hash"], *dep_file_hashes}


# =============================================================================
# get_target_hashes edge cases
# =============================================================================


def _helper_ref(producer: str, key: str | None, tag: types.ArtifactTag) -> types.ArtifactRef:
    return types.ArtifactRef(
        identity=types.ArtifactIdentity(producer, key),
        format=loaders.CSV() if tag is types.ArtifactTag.DATA else loaders.JSON(),
        python_type=pathlib.Path,
        tag=tag,
    )


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


def test_get_target_hashes_excludes_noncached_outputs(set_project_root: pathlib.Path) -> None:
    """get_target_hashes excludes non-cached (metric) outputs from returned hashes."""
    state_dir = set_project_root / ".pivot"
    cached_ref = _helper_ref("my_stage", "output", types.ArtifactTag.DATA)
    metric_ref = _helper_ref("my_stage", "metrics", types.ArtifactTag.METRIC)
    all_stages = {
        "my_stage": RegistryStageInfo(  # pyright: ignore[reportCallIssue] - partial for test
            state_dir=None,
            outs=[cached_ref, metric_ref],
            incremental_inputs={},
            input_paths={},
        )
    }
    lock.StageLock("my_stage", lock.get_stages_dir(state_dir)).write(
        LockData(
            code_manifest={},
            params={},
            dep_hashes={},
            output_hashes={
                cached_ref.identity: FileHash(hash="1111111111111111"),
                metric_ref.identity: FileHash(hash="2222222222222222"),
            },
        )
    )

    result = sync.get_target_hashes(["my_stage"], state_dir, all_stages=all_stages)

    assert "1111111111111111" in result, "Cached output hash should be included"
    assert "2222222222222222" not in result, "Non-cached output hash should be excluded"


def test_get_target_hashes_file_target_excludes_noncached(
    set_project_root: pathlib.Path,
) -> None:
    """File-path target for a non-cached output returns no hashes; a cached one returns its hash."""
    state_dir = set_project_root / ".pivot"
    cached_ref = _helper_ref("my_stage", "output", types.ArtifactTag.DATA)
    metric_ref = _helper_ref("my_stage", "metrics", types.ArtifactTag.METRIC)
    all_stages = {
        "my_stage": RegistryStageInfo(  # pyright: ignore[reportCallIssue] - partial for test
            state_dir=None,
            outs=[cached_ref, metric_ref],
            incremental_inputs={},
            input_paths={},
        )
    }
    lock.StageLock("my_stage", lock.get_stages_dir(state_dir)).write(
        LockData(
            code_manifest={},
            params={},
            dep_hashes={},
            output_hashes={
                cached_ref.identity: FileHash(hash="1111111111111111"),
                metric_ref.identity: FileHash(hash="2222222222222222"),
            },
        )
    )
    workspace = store.WorkspaceStore(
        project_root=set_project_root, pipeline_name="test", input_bindings={}
    )
    metric_path = workspace.resolve_display_path(metric_ref)
    cached_path = workspace.resolve_display_path(cached_ref)

    # Target the non-cached file directly by its workspace path (not stage name)
    result = sync.get_target_hashes([str(metric_path)], state_dir, all_stages=all_stages)
    assert "2222222222222222" not in result, "Non-cached file target should return no hashes"

    # Target the cached file directly by its workspace path — should return its hash
    result_cached = sync.get_target_hashes([str(cached_path)], state_dir, all_stages=all_stages)
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

    # Create a directory at a cache key; the cache stores directory trees elsewhere.
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


def test_get_target_hashes_identity_target_with_lock_files_no_pipeline(
    set_project_root: pathlib.Path,
) -> None:
    """Identity target resolves via lock file scan even without a pipeline."""
    state_dir = set_project_root / ".pivot"
    output_hash = FileHash(hash="aabbccdd11223344")
    lock.StageLock("train", lock.get_stages_dir(state_dir)).write(
        LockData(
            code_manifest={},
            params={},
            dep_hashes={},
            output_hashes={types.ArtifactIdentity("train", "output"): output_hash},
        )
    )

    result = sync.get_target_hashes(["train:output"], state_dir, all_stages=None)

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

    lock.StageLock("train", stages_dir).write(
        LockData(
            code_manifest={},
            params={},
            dep_hashes={},
            output_hashes={
                types.ArtifactIdentity("train", "file"): FileHash(hash="aaaa000000000000")
            },
        )
    )

    result = sync.get_target_hashes(
        [str(set_project_root / "nonexistent.csv")], state_dir, all_stages=None
    )

    assert result == set(), "Unresolved target should return empty set, not crash"
